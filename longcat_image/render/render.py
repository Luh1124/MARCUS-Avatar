import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pyexr
from PIL import Image

# Assume brdf and lighting modules are in the same directory or on the path
# If imports fail, check the file structure or temporarily copy the related classes here
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from brdf import eval_ggx, eval_diffuse

class TextureRenderLayer(nn.Module):
    """
    [Strict PBR Version - Tangent Space Support]
    Adapt to the ParametricFaceModel coordinate system (camera at +Z).
    Use precomputed TBN maps to convert input tangent-space normals to world space.
    """
    def __init__(self,
                 brdf_type="ggx",
                 spp=1,
                 use_specular=True,
                 disp_scale=0.01, # blender unit is 0.01
                 camera_distance=10.0,
                 ambient_intensity=0.1,  # Pure ambient light intensity (0.0-1.0)
                 fill_light_intensity=0.3,  # Fill light intensity (0.0-1.0)
                 fill_light_dir=None,  # Fill light direction [x, y, z], None = auto (opposite to main light)
                 fast_render=False,  # Fast render mode: skip BRDF, use simple Lambertian
                 position_map_path="./position_map.exr",   # Pos Map contains positive and negative values
                 geo_normal_path="./geo_normal.exr",       # World Geometric Normal contains positive and negative values
                 tan_normal_path="./tan_normal.exr",       # World Geometric Tangent contains positive and negative values
                 device='cuda'):
        super().__init__()
        
        self.use_specular = use_specular
        self.brdf_type = brdf_type
        self.spp = spp
        self.disp_scale = disp_scale
        self.ambient_intensity = ambient_intensity
        self.fill_light_intensity = fill_light_intensity
        self.fast_render = fast_render
        self.device = device
        self.eps = 1e-6
        
        # Fill light direction (can be overridden, or auto-computed from main light)
        if fill_light_dir is not None:
            self.register_buffer(
                'fill_light_dir_fixed',
                F.normalize(torch.tensor(fill_light_dir, device=device).view(1, 1, 3, 1, 1), dim=2)
            )
        else:
            self.fill_light_dir_fixed = None
        
        # Camera position: Z = +10.0
        self.register_buffer(
            'camera_pos', 
            torch.tensor([0.0, 0.0, float(camera_distance)], device=device).view(1, 3, 1, 1)
        )

        # ------------------------------------------------------------------
        # Helper: load EXR and convert to Tensor (B, C, H, W)
        # ------------------------------------------------------------------
        def load_exr_as_tensor(path):
            if not os.path.exists(path):
                # For testing convenience, generate random noise if the file does not exist
                print(f"Warning: File {path} not found. Generating random noise for testing.")
                return torch.randn(1, 3, 512, 512).to(device)

            try:
                data = pyexr.read(path) # (H, W, C)
            except Exception as e:
                raise FileNotFoundError(f"Failed to load EXR file at {path}: {e}")
            
            tensor = torch.from_numpy(data).to(device) # (H, W, C)
            return tensor.permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)

        # 1. Load Position Map and Mask
        pos_data = load_exr_as_tensor(position_map_path)
        
        # Assume the alpha channel of the position map (the fourth channel) stores the mask
        if pos_data.shape[1] >= 4:
            self.register_buffer('pos_map', pos_data[:, :3, ...]) # (1, 3, H, W)
            self.register_buffer('mask', pos_data[:, 3:4, ...])   # (1, 1, H, W)
        else:
            # If there is no alpha channel, assume the whole image is valid
            print(f"Info: No alpha channel in {position_map_path}, assuming full mask.")
            self.register_buffer('pos_map', pos_data[:, :3, ...])
            self.register_buffer('mask', torch.ones_like(pos_data[:, :1, ...]))

        # 2. Load Geometric Normal (World Space)
        self.register_buffer('geom_normal', load_exr_as_tensor(geo_normal_path)[:, :3, ...])

        # 3. Load Geometric Tangent (World Space)
        self.register_buffer('geom_tangent', load_exr_as_tensor(tan_normal_path)[:, :3, ...])


    def forward(
            self,
            lighting_model: nn.Module,
            albedo: torch.Tensor,
            rough: torch.Tensor,
            specular: torch.Tensor,
            normal_map: torch.Tensor, 
            disp: torch.Tensor = None,
            forced_light_dir: torch.Tensor = None,
            camera_pos_override: torch.Tensor = None
    ):
        bn, _, h, w = albedo.shape
        # sRGB2linear important!
        albedo = torch.pow(albedo, 2.2)

        # ============ 1. Build TBN matrix (World Space) ============
        # Broadcast geom_normal and geom_tangent to batch size if needed
        # self.geom_normal: (1, 3, H, W) -> (bn, 3, H, W) via broadcasting
        N_geom = F.normalize(self.geom_normal, dim=1, eps=self.eps)  # (1, 3, H, W) or (bn, 3, H, W)
        T_geom = F.normalize(self.geom_tangent, dim=1, eps=self.eps)  # (1, 3, H, W) or (bn, 3, H, W)
        
        # Right-handed coordinate system check: usually N x T
        B_geom = torch.cross(N_geom, T_geom, dim=1)
        B_geom = F.normalize(B_geom, dim=1, eps=self.eps)

        # ============ 2. Shading Normal (Tangent -> World) ============
        # Decode: [0, 1] -> [-1, 1]
        nm_decoded = normal_map * 2.0 - 1.0 
        
        # TBN Transformation
        N_shading = (T_geom * nm_decoded[:, 0:1, ...] + 
                     B_geom * nm_decoded[:, 1:2, ...] + 
                     N_geom * nm_decoded[:, 2:3, ...])
        N_shading = F.normalize(N_shading, dim=1, eps=self.eps)

        # ============ 3. Geometry (Displacement) ============
        final_pos = self.pos_map # B, 3, H, W
        if disp is not None:
            # Ensure disp is single channel (B, 1, H, W) for broadcasting with N_geom (B, 3, H, W)
            if disp.shape[1] > 1:
                disp = disp[:, 0:1, ...]  # Take first channel if multi-channel
            # Extrude along Geometric Normal
            # N_geom: (B, 3, H, W), disp: (B, 1, H, W) -> broadcasts to (B, 3, H, W)
            displacement_vec = N_geom * disp * self.disp_scale
            final_pos = self.pos_map + displacement_vec

        # ============ 4. Vectors ============
        
        # Wi (View Vector): Surface -> Camera
        # final_pos: (B, 3, H, W), camera_pos: (1, 3, 1, 1)
        current_cam_pos = self.camera_pos
        if camera_pos_override is not None:
            current_cam_pos = camera_pos_override
            
        wi = F.normalize(current_cam_pos - final_pos, dim=1, eps=self.eps)  # (B, 3, H, W)
        
        # Wo (Light Vector): Surface -> Light
        wo_emitter = lighting_model.sample_direction(vpos=None, normal=N_shading, specific_dir=forced_light_dir)
        # wo_emitter: (B, K, 3) where K is number of light samples
        wo_emitter = F.normalize(wo_emitter, dim=2, eps=self.eps)
        
        # Expand wo_emitter to spatial dimensions: (B, K, 3) -> (B, K, 3, H, W)
        wo_emitter = wo_emitter.unsqueeze(-1).unsqueeze(-1)  # (B, K, 3, 1, 1)
        wo = wo_emitter.expand(-1, -1, -1, h, w)  # (B, K, 3, H, W)
        
        # Expand wi to match wo's K dimension: (B, 3, H, W) -> (B, 1, 3, H, W) -> (B, K, 3, H, W)
        wi = wi.unsqueeze(1)  # (B, 1, 3, H, W)
        k = wo.shape[1]  # Get K from wo
        wi = wi.expand(-1, k, -1, -1, -1)  # (B, K, 3, H, W)

        # ============ 5. Visibility Calculation ============
        # Compute the dot product between surface normal and view direction for later weighting
        # N_geom: (B, 3, H, W) -> (B, 1, 3, H, W)
        # wi: (B, K, 3, H, W)
        # Use the geometry normal to determine visibility here because it is more stable
        ndv_geom = torch.sum(N_geom.unsqueeze(1) * wi, dim=2, keepdim=True) # (B, K, 1, H, W)
        vis_mask = torch.clamp(ndv_geom, min=0.0) # back faces are 0

        # ============ 6. World to Tangent Space Transformation ============
        # The BRDF (eval_ggx) assumes vectors in tangent space where Normal = [0, 0, 1]
        # We need to transform world-space wi and wo into a local frame defined by N_shading
        
        # Build local frame from N_shading (using Shading Normal for specular accuracy)
        # N_shading: (B, 3, H, W) - the perturbed normal from normal map
        # We use the geometric tangent as a reference and re-orthogonalize
        
        # Expand N_shading to match wi/wo dimensions: (B, 3, H, W) -> (B, 1, 3, H, W)
        N_local = N_shading.unsqueeze(1)  # (B, 1, 3, H, W)
        
        # For tangent space, we use the geometric tangent, re-orthogonalized to N_shading
        T_local = T_geom.unsqueeze(1)  # (B, 1, 3, H, W)
        # Gram-Schmidt orthogonalization: T' = T - (T·N)*N
        T_local = T_local - torch.sum(T_local * N_local, dim=2, keepdim=True) * N_local
        T_local = F.normalize(T_local, dim=2, eps=self.eps)
        
        # Bitangent: B = N × T
        B_local = torch.cross(N_local, T_local, dim=2)
        B_local = F.normalize(B_local, dim=2, eps=self.eps)
        
        # Transform wi and wo from world space to tangent space
        # v_tangent = [dot(v, T), dot(v, B), dot(v, N)]
        # wi, wo: (B, K, 3, H, W)
        wi_t_x = torch.sum(wi * T_local, dim=2, keepdim=True)  # (B, K, 1, H, W)
        wi_t_y = torch.sum(wi * B_local, dim=2, keepdim=True)
        wi_t_z = torch.sum(wi * N_local, dim=2, keepdim=True)
        wi_tangent = torch.cat([wi_t_x, wi_t_y, wi_t_z], dim=2)  # (B, K, 3, H, W)
        wi_tangent = F.normalize(wi_tangent, dim=2, eps=self.eps)
        
        wo_t_x = torch.sum(wo * T_local, dim=2, keepdim=True)
        wo_t_y = torch.sum(wo * B_local, dim=2, keepdim=True)
        wo_t_z = torch.sum(wo * N_local, dim=2, keepdim=True)
        wo_tangent = torch.cat([wo_t_x, wo_t_y, wo_t_z], dim=2)  # (B, K, 3, H, W)
        wo_tangent = F.normalize(wo_tangent, dim=2, eps=self.eps)

        # ============ 7. Material ============
        # Fast render mode: skip expensive BRDF calculation, use simple Lambertian
        if self.fast_render:
            # Simple Lambertian diffuse: albedo / pi
            eval_diff = albedo / 3.14159  # (B, 3, H, W)
            eval_spec = torch.zeros_like(albedo) if not self.use_specular else specular * 0.1
        else:
            skin_f0 = 0.08 * specular 
            rough_clamped = torch.clamp(rough, min=0.04, max=1.0)
            dummy_metal = torch.zeros_like(rough)

            # ============ 8. BRDF (in Tangent Space) ============
            # Now wi_tangent and wo_tangent are in tangent space where N = [0, 0, 1]
            try:
                eval_diff, eval_spec, _ = eval_ggx(
                    color=albedo, roughness=rough_clamped, metalness=dummy_metal, 
                    wi=wi_tangent, wo=wo_tangent, f0_override=skin_f0 
                )
            except TypeError:
                eval_diff, eval_spec, _ = eval_ggx(
                    color=albedo, roughness=rough_clamped, metalness=dummy_metal, 
                    wi=wi_tangent, wo=wo_tangent
                )
                eval_spec = eval_spec * (2.0 * specular.unsqueeze(1))

        # ============ 9. Shading Integration ============
        
        # light_radiance: (B, K, 3, H, W)
        light_radiance = lighting_model(wo) 

        # ndl: (B, K, 1, H, W)
        # N_shading: (B, 3, H, W) -> (B, 1, 3, H, W)
        n_expanded = N_shading.unsqueeze(1)
        # wo: (B, K, 3, H, W)
        # sum over dim=2 (channels), keepdim=True -> (B, K, 1, H, W)
        ndl = torch.sum(n_expanded * wo, dim=2, keepdim=True)
        ndl = torch.clamp(ndl, min=0.0) 
        
        # Ensure eval_diff and eval_spec have a K dimension (B, K, 3, H, W)
        # Many BRDF implementations return (B, 3, H, W), and need unsqueeze
        if eval_diff.dim() == 4:
            eval_diff = eval_diff.unsqueeze(1) # (B, 1, 3, H, W)
        
        if eval_spec is not None and eval_spec.dim() == 4:
            eval_spec = eval_spec.unsqueeze(1) # (B, 1, 3, H, W)

        # Diffuse Component
        # (B, K, 3, H, W) * (B, K, 3, H, W) * (B, K, 1, H, W)
        diffuse_term = eval_diff * light_radiance * ndl
        # Sum over K (dim=1) -> (B, 3, H, W)
        colorDiffuse = torch.sum(diffuse_term, dim=1)
        
        # Add ambient light to avoid black areas
        # Ambient light is uniform and doesn't depend on light direction
        ambient_term = albedo * self.ambient_intensity
        colorDiffuse = colorDiffuse + ambient_term

        # Specular Component
        if self.use_specular:
            spec_term = eval_spec * light_radiance * ndl
            colorSpec = torch.sum(spec_term, dim=1)
        else:
            colorSpec = torch.zeros_like(colorDiffuse)

        # Shading Map
        # light_radiance mean over channels -> (B, K, 1, H, W)
        light_intensity = torch.mean(light_radiance, dim=2, keepdim=True)
        shading_map = light_intensity * ndl
        shading_map = torch.sum(shading_map, dim=1) # (B, 1, H, W)
        # Add ambient to shading map
        shading_map = shading_map + self.ambient_intensity

        # Masking
        colorDiffuse = colorDiffuse * self.mask
        colorSpec = colorSpec * self.mask
        shading_map = shading_map * self.mask

        # 1. Geometry visibility (Camera Visibility): whether the surface faces the camera
        # vis_mask is (B, K, 1, H, W), average over the K dimension
        # Result: (B, H, W) range [0, 1]
        vis_weight_geometry = torch.mean(vis_mask, dim=1).squeeze(1) 
        
        # 2. UV Mask: whether pixels are inside the valid texture region
        valid_uv_mask = self.mask.squeeze(1) # (B, H, W)
        
        # 3. Final weight: geometry visibility * UV validity
        # Do not multiply by N.L (lighting), because we want albedo in shadowed regions to be trainable too
        vis_weight = vis_weight_geometry * valid_uv_mask
        
        return colorDiffuse, colorSpec, shading_map, vis_weight



# ==========================================
# Test script
# ==========================================
if __name__ == "__main__":
    # Mock Lighting Class for testing
    from lighting import MockLighting, RandomizedDirectionalLight
    from lighting import GlobalIncidentLighting, DirectionalLight, DirectionalLight_LatLong, ConstantIntensity

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Path configuration (make sure files exist under these paths)
    texture_dir = os.environ.get("MARCUS_TEST_TEXTURE_DIR", "./examples/textures")
    
    # Geometry texture paths (EXR)
    # Assume these EXR files have been exported correctly: World Position, World Normal, World Tangent
    # and contain positive and negative values (pyexr preserves original float values after loading)
    pos_path = "./position_map.exr" 
    geo_norm_path = "./geo_normal.exr"
    tan_norm_path = "./tan_normal.exr"

    output_dir = "./test_render"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize renderer
    renderer = TextureRenderLayer(
        position_map_path=pos_path,
        geo_normal_path=geo_norm_path,
        tan_normal_path=tan_norm_path,
        fast_render=False,  # Enable fast render mode (use simple Lambertian and skip GGX BRDF)
        ambient_intensity=0.2,  # increase ambient intensity to ensure full illumination
        device=device
    )

    # Helper: read an image and normalize it to Tensor (B, C, H, W)
    def image_to_tensor(path, is_mono=False):
        if not os.path.exists(path):
            print(f"Warning: Texture {path} not found. Using random tensor.")
            c = 1 if is_mono else 3
            return torch.rand(1, c, 512, 512).to(device)
        
        img = Image.open(path)
        
        if is_mono:
            # Force conversion to grayscale (L mode), to ensure only one channel of data
            # This correctly computes luminance even if the source is RGBA
            img = img.convert('L') 
            img_np = np.array(img).astype(np.float32) / 255.0
            # Grayscale images are read as (H, W), need an extra channel dimension -> (H, W, 1)
            img_np = img_np[..., None]
        else:
            # For color images, ensure RGB and drop alpha channel
            img = img.convert('RGB')
            img_np = np.array(img).astype(np.float32) / 255.0
            # RGB images are already read as (H, W, 3), no change needed
        
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
        return tensor.to(device)

    # Read textures
    print("Loading textures...")
    albedo = image_to_tensor(os.path.join(texture_dir, "Face_Albedo.png"))
    rough = image_to_tensor(os.path.join(texture_dir, "Face_Roughness.png"), is_mono=True)
    specular = image_to_tensor(os.path.join(texture_dir, "Face_Specular.png"), is_mono=True)
    normal_map = image_to_tensor(os.path.join(texture_dir, "Face_Normal.png")) # Tangent Space Normal (Blueish)
    disp = image_to_tensor(os.path.join(texture_dir, "Face_Displacement.png"), is_mono=True)

    # Result-saving function
    def save_tensor_img(tensor, name):
        # 1. Remove batch and potential K dimensions
        while tensor.dim() > 3:
            tensor = tensor.squeeze(0)  # Removethe leading singleton dimension
        
        # 2. (C, H, W) -> (H, W, C)
        img_np = tensor.permute(1, 2, 0).cpu().numpy()
        
        # 3. Handle values
        img_np = np.nan_to_num(img_np)  # Prevent NaN from causing a black image
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        
        # 4. Handle grayscale (single-channel image)
        if img_np.shape[2] == 1:
            img_np = img_np.squeeze(2)  # (H, W, 1) -> (H, W)
            
        Image.fromarray(img_np).save(name)
        print(f"Saved {name}")

    # Define four lighting models for testing
    lighting_scenarios = [
        # 1. Mock Lighting (Fast Mode / Single Dir)
        ("Mock_Lighting", MockLighting(use_fast_mode=True).to(device)),
        
        # 2. Directional Light (Standard, learnable)
        ("Directional_Light", GlobalIncidentLighting(value=DirectionalLight(weight_init=2)).to(device)),
        
        # 3. Directional Light (LatLong Parameterization)
        ("LatLong_Light", GlobalIncidentLighting(value=DirectionalLight_LatLong(weight_init=2)).to(device)),
        
        # 4. Constant Intensity (Ambient/Hemisphere like)
        ("Constant_Light", GlobalIncidentLighting(value=ConstantIntensity(value=(2.0,), exp_val=True)).to(device))
    ]

    # Define a forced lighting direction (pointing toward -Z, directly lighting the face)
    # Shape: (B, 1, 3)
    forced_light_dir = torch.tensor([[[0.0, 0.0, 1.0]]], device=device) 

    for name, lighting_model in lighting_scenarios:
        print(f"\nRendering with {name}...")
        
        # For MockLighting and DirectionalLight, we may want to test forced_light_dir
        # For more complex models, they may sample by themselves
        current_light_dir = forced_light_dir if "Mock" in name or "Directional" in name else None
        
        # For DirectionalLight, set its internal direction manually to ensure frontal illumination
        if "Directional_Light" in name:
             lighting_model.value.direction[:] = torch.tensor([[0.0, 0.0, 1.0]], device=device)

        with torch.no_grad():
            c_diff, c_spec, shading, vis_weight = renderer(
                lighting_model=lighting_model, 
                albedo=albedo, 
                rough=rough, 
                specular=specular, 
                normal_map=normal_map, 
                disp=disp, 
                forced_light_dir=current_light_dir
            )

        # Compose final image (Diff + Spec)
        final_image = c_diff + c_spec
        
        # Gamma Correction for display (Linear -> sRGB)
        final_image = torch.pow(final_image, 1.0/2.2)

        # Save result
        save_tensor_img(final_image, os.path.join(output_dir, f"test_render_{name}_final.png"))
        save_tensor_img(c_diff, os.path.join(output_dir, f"test_render_{name}_diffuse.png"))
        save_tensor_img(c_spec, os.path.join(output_dir, f"test_render_{name}_specular.png"))
        
        # Shading map may be single-channel; save it as grayscale
        if shading.shape[1] == 1:
            shading = shading.repeat(1, 3, 1, 1)
        save_tensor_img(shading, os.path.join(output_dir, f"test_render_{name}_shading.png"))
        
        # Save Visibility Weight
        vis_img = vis_weight.unsqueeze(1).repeat(1, 3, 1, 1) # (1, 1, H, W) -> (1, 3, H, W)
        save_tensor_img(vis_img, os.path.join(output_dir, f"test_render_{name}_vis.png"))

    print("\nAll Done!")
