import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from pathlib import Path
from PIL import Image
import io
import struct
import cv2
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Attributes, Material, PbrMetallicRoughness, TextureInfo, Texture, Image as GLTFImage, Sampler, Buffer, BufferView, Accessor

def save_glb(path, vertices, faces, uvs, face_uvs, albedo, normal, rsd):
    """
    Export mesh and PBR textures as GLB with KHR_materials_specular support.
    """
    # Ensure inputs are NumPy arrays
    if torch.is_tensor(vertices): vertices = vertices.detach().cpu().numpy()
    if torch.is_tensor(faces): faces = faces.detach().cpu().numpy()
    if torch.is_tensor(uvs): uvs = uvs.detach().cpu().numpy()
    if torch.is_tensor(face_uvs): face_uvs = face_uvs.detach().cpu().numpy()
    
    vertices = vertices.astype(np.float32)
    uvs = uvs.astype(np.float32)
    faces = faces.astype(np.uint32)
    face_uvs = face_uvs.astype(np.uint32)
    
    # 1. Mesh processing (duplicate vertices to handle UV seams)
    # This increases vertex count but prevents stretching at UV seams, which is important for research visualization
    v_flat = vertices[faces.flatten()]  # (F*3, 3)
    uv_flat = uvs[face_uvs.flatten()]   # (F*3, 2)
    uv_flat[:, 1] = 1.0 - uv_flat[:, 1] # Flip the V axis for glTF lower-left origin
    
    nb_faces = faces.shape[0]
    # Rebuild indices (vertices are duplicated, so indices are simply 0,1,2,3,4,5...)
    faces_flat = np.arange(nb_faces * 3, dtype=np.uint32).reshape(-1, 3)
    
    # 2. Texture processing
    if rsd.mode != 'RGB':
        rsd = rsd.convert('RGB')
    
    r_img, s_img, d_img = rsd.split()
    width, height = albedo.size
    
    # Build the ORM texture: R=Occlusion (1.0), G=Roughness, B=Metalness (0.0)
    # No AO is baked, so occlusion is set to white (255)
    red_occlusion = Image.new('L', (width, height), 255)
    blue_metal = Image.new('L', (width, height), 0)
    # Green channel comes from the red channel of RSD (i.e. roughness)
    orm_img = Image.merge('RGB', (red_occlusion, r_img, blue_metal))
    
    # Specular texture: converting to RGB is safer
    specular_img = Image.merge('RGB', (s_img, s_img, s_img))

    # Helper: convert to PNG bytes
    def get_png_bytes(img):
        with io.BytesIO() as bio:
            # optimize=True can reduce size slightly but is a bit slower
            img.save(bio, format="PNG", optimize=True)
            return bio.getvalue()

    albedo_bytes = get_png_bytes(albedo)
    normal_bytes = get_png_bytes(normal)
    orm_bytes = get_png_bytes(orm_img)
    specular_bytes = get_png_bytes(specular_img)

    # 3. Build GLTF object
    gltf = GLTF2()
    gltf.asset = {"version": "2.0", "generator": "TVCG Pipeline Exporter"}
    
    # === Key fix: declare extension ===
    gltf.extensionsUsed = ["KHR_materials_specular"]

    buffer_blob = bytearray()
    
    # === Key fix: 4-byte alignment ===
    def align_buffer():
        padding = len(buffer_blob) % 4
        if padding != 0:
            buffer_blob.extend(b'\x00' * (4 - padding))

    buffer_views = []
    accessors = []
    
    # --- Geometry accessors ---
    
    # Position
    align_buffer()
    pos_offset = len(buffer_blob)
    pos_data = v_flat.tobytes()
    buffer_blob.extend(pos_data)
    
    buffer_views.append(BufferView(
        buffer=0, byteOffset=pos_offset, byteLength=len(pos_data), target=34962 # ARRAY_BUFFER
    ))
    accessors.append(Accessor(
        bufferView=len(buffer_views)-1, componentType=5126, count=len(v_flat), type="VEC3",
        min=v_flat.min(axis=0).tolist(), max=v_flat.max(axis=0).tolist()
    ))
    pos_acc_idx = len(accessors)-1

    # UV
    align_buffer()
    uv_offset = len(buffer_blob)
    uv_data = uv_flat.tobytes()
    buffer_blob.extend(uv_data)
    
    buffer_views.append(BufferView(
        buffer=0, byteOffset=uv_offset, byteLength=len(uv_data), target=34962
    ))
    accessors.append(Accessor(
        bufferView=len(buffer_views)-1, componentType=5126, count=len(uv_flat), type="VEC2"
    ))
    uv_acc_idx = len(accessors)-1

    # Indices
    align_buffer()
    ind_offset = len(buffer_blob)
    ind_data = faces_flat.flatten().tobytes()
    buffer_blob.extend(ind_data)
    
    buffer_views.append(BufferView(
        buffer=0, byteOffset=ind_offset, byteLength=len(ind_data), target=34963 # ELEMENT_ARRAY_BUFFER
    ))
    accessors.append(Accessor(
        bufferView=len(buffer_views)-1, componentType=5125, count=len(faces_flat.flatten()), type="SCALAR"
    ))
    ind_acc_idx = len(accessors)-1

    # --- Texture processing ---
    
    images = []
    textures = []
    # Use linear sampler
    samplers = [Sampler(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497)]

    def add_texture(bytes_data):
        align_buffer()
        offset = len(buffer_blob)
        buffer_blob.extend(bytes_data)
        
        buffer_views.append(BufferView(
            buffer=0, byteOffset=offset, byteLength=len(bytes_data)
        ))
        images.append(GLTFImage(
            bufferView=len(buffer_views)-1, mimeType="image/png"
        ))
        textures.append(Texture(source=len(images)-1, sampler=0))
        return len(textures)-1

    # Keep this order for index references
    alb_idx = add_texture(albedo_bytes)
    nrm_idx = add_texture(normal_bytes)
    orm_idx = add_texture(orm_bytes)
    spc_idx = add_texture(specular_bytes)

    # --- Material definition ---
    
    mat = Material(
        name="HumanSkin",
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorTexture=TextureInfo(index=alb_idx),
            metallicRoughnessTexture=TextureInfo(index=orm_idx),
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=1.0, # 1.0 is a multiplier; the actual value is controlled by the blue texture channel (0), so the result is 0
            roughnessFactor=1.0 # 1.0 is a multiplier; the actual value is controlled by the green texture channel
        ),
        normalTexture=TextureInfo(index=nrm_idx, scale=1.0),
        extensions={
            "KHR_materials_specular": {
                "specularTexture": {"index": spc_idx},
                "specularFactor": 1.0
            }
        }
    )

    # --- Scene assembly ---
    
    prim = Primitive(
        attributes=Attributes(POSITION=pos_acc_idx, TEXCOORD_0=uv_acc_idx),
        indices=ind_acc_idx,
        material=0
    )
    
    mesh = Mesh(primitives=[prim])
    node = Node(mesh=0, name="FaceMesh")
    scene = Scene(nodes=[0])
    
    gltf.buffers = [Buffer(byteLength=len(buffer_blob))]
    gltf.bufferViews = buffer_views
    gltf.accessors = accessors
    gltf.materials = [mat]
    gltf.meshes = [mesh]
    gltf.nodes = [node]
    gltf.scenes = [scene]
    gltf.scene = 0
    gltf.images = images
    gltf.textures = textures
    gltf.samplers = samplers

    # 4. Save file
    gltf.save_binary(path, buffer_blob)
    print(f"[SUCCESS] GLB model saved to: {path}")


cache_dir = "cache"
os.makedirs(cache_dir, exist_ok=True)

from utils import read_img, np2tensor, estimate_norm_torch, estimate_norm, pillow2np, np2pillow
from longcat_image.preprocess import Preprocess_API
from longcat_image.face3d_recon import Face3d_Recon_API
from longcat_image.tex import Tex_API

from diffusers import LongCatImageEditPipeline
from peft import LoraConfig, PeftModel

from longcat_image.models.pipeline_intrinsix_image_edit import IntrinsiXEditPipeline
from runtime_paths import BASE_MODEL_PATH, ckpt_path, topo_path

if __name__ == "__main__":

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"


    # ---------------------- Step 1. Init Preprocess_API ----------------------
    preprocess_model = Preprocess_API(
        lm_detector_path=ckpt_path("lm_model/68lm_detector.pb"),
        mtcnn_path=ckpt_path("face_box/large_base_net.pth"),
        lm68_3d_path=topo_path("similarity_Lm3D_all.mat"),
        parsing_pth=ckpt_path("parsing_model/dml_csr_celebA.pth"),
        target_size=224,
        rescale_factor=102.0,
        device=device_str,
    )
                
    # ---------------------- Step 2. Init HiFi3D ----------------------
    face3d_model = Face3d_Recon_API(
        pfm_model_path=topo_path("hifi3dpp_model_info.mat"),
        recon_model_path=ckpt_path("deep3d_model/epoch_latest.pth"),
        image_super_net_path=ckpt_path("sr_model/RealESRGAN_x4plus.pth"),
        focal=1015.0,
        camera_distance=10.0,
        device=device_str,
    )

    # ---------------------- Step 3. Init Tex_API ----------------------
    tex_model = Tex_API(
        unwrap_info_path=topo_path('unwrap_1024_info.mat'),
        unwrap_info_mask_path=topo_path('unwrap_1024_info_mask.png'),
        unwrap_size=1024,
    )

    # ---------------------- Step 3. Init LongCatImageEditPipeline ----------------------
    pipe = LongCatImageEditPipeline.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device=device_str,
    )

    pipe.to(device_str, torch.bfloat16)

    # --- Multi-LoRA Management ---
    # Define your LoRAs here: { "adapter_name": "path_to_checkpoint" }
    lora_configs = {
        "inpainting": ckpt_path("lora_inpainting", "transformer"),
        "delight": ckpt_path("lora_delight", "transformer"),
        "albedo": ckpt_path("lora_albedo", "transformer"),
        "rsd": ckpt_path("lora_rsd", "transformer"),
        "normal": ckpt_path("lora_normal", "transformer")
    }

    # 1. Initialize PeftModel with the first LoRA
    if len(lora_configs) > 0:
        first_lora_name = list(lora_configs.keys())[0]
        first_lora_path = lora_configs[first_lora_name]
        print(f"[INFO] Initializing PeftModel with LoRA: {first_lora_name}")
        
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer, 
            first_lora_path, 
            adapter_name=first_lora_name
        )

        # 2. Load additional LoRAs
        for name, path in lora_configs.items():
            if name == first_lora_name:
                continue
            print(f"[INFO] Loading additional LoRA: {name}")
            pipe.transformer.load_adapter(path, adapter_name=name)
    else:
        print("[WARN] No LoRA configs provided, using base model.")

    def switch_lora(lora_name):
        """Helper to switch active LoRA adapter"""
        if lora_name is None:
            print(f"[INFO] Disabling LoRA (using base model)")
            pipe.transformer.disable_adapters()
            return

        if lora_name in lora_configs:
            print(f"[INFO] Switching to LoRA: {lora_name}")
            pipe.transformer.set_adapter(lora_name)
        else:
            print(f"[WARN] LoRA {lora_name} not found in configs!")

    # ---------------------- Step 4. Load Image ----------------------

    image_path = "examples/000534.jpg"

    image = read_img(image_path)

    align_img, hr_img, trans_params, lm68_2d, seg_mask, skin_mask = preprocess_model(image)

    # 3D face reconstruction
    coeffs = face3d_model.pred_coeffs(np2tensor(align_img, device=device_str))
    
    # Compute face shape
    face_shape, neutral_shape = face3d_model.facemodel.compute_shape(coeffs['id'], coeffs['exp'])
    rotation_matrix = face3d_model.facemodel.compute_rotation(coeffs['angle'])
    face_shape_transformed = face3d_model.facemodel.transform(face_shape, rotation_matrix, coeffs['trans'])

    # Get faces and UVs
    faces = face3d_model.facemodel.head_buf.cpu().numpy()
    uvs = face3d_model.facemodel.vt_list.cpu().numpy()
    face_uvs = face3d_model.facemodel.head_tri_vt.cpu().numpy()

    vertices = face_shape_transformed[0].cpu().numpy()      # (20481, 3)
    neutral_vertices = neutral_shape[0].cpu().numpy()      # (20481, 3)

    # unwrap texture
    projXY, norm = face3d_model.compute_224projXY_norm_by_pin_hole(coeffs)
    projXY, norm = projXY[0].cpu().numpy(), norm[0].cpu().numpy()

    projXY = preprocess_model.trans_projXY_back_to_ori_coord(projXY, trans_params)

    unwrap_uv_tex, remap_mask = tex_model(image, seg_mask, projXY, norm)

    # save unwrap texture
    unwrap_uv_tex = np2pillow(unwrap_uv_tex)
    # unwrap_uv_tex.save(os.path.join(cache_dir, "unwrap_uv_tex.png"))

    # Convert PIL image to OpenCV format (NumPy array)
    ref_cv = np.array(unwrap_uv_tex)
    ref_cv = cv2.cvtColor(ref_cv, cv2.COLOR_RGB2BGR)  # PIL RGB to OpenCV BGR

    # Convert to grayscale and get the foreground mask (assuming black background and nonzero foreground)
    ref_gray = cv2.cvtColor(ref_cv, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(ref_gray, 1, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    # Erode the mask to shrink inward; the inner region is foreground and the outside is background
    kernel = np.ones((12, 12), np.uint8)  # Use stronger erosion to shrink the boundary clearly
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    # Build result: keep the original inside and set edges/outside to pure black
    ref_cv_eroded = np.zeros_like(ref_cv)
    ref_cv_eroded[eroded_mask == 255] = ref_cv[eroded_mask == 255]

    # Convert back to PIL Image; only the center remains, edges/background are black
    eroded_rgb = cv2.cvtColor(ref_cv_eroded, cv2.COLOR_BGR2RGB)
    # Ensure dtype is uint8 because PIL Image requires uint8 (0-255)
    if eroded_rgb.dtype != np.uint8:
        eroded_rgb = np.clip(eroded_rgb, 0, 255).astype(np.uint8)
    ref_image = Image.fromarray(eroded_rgb)
    
    # [FIX] Ensure 'image' variable is available for subsequent steps (e.g. tex_model)
    image = ref_image


    prompt = "Seamlessly restore and complete the missing regions of the provided texture map. Perform context-aware inpainting that strictly preserves the original global illumination, shading gradients, and volumetric shadows. Synthesize high-frequency micro-details that perfectly match the surrounding grain, ensuring pixel-perfect continuity in albedo and surface roughness. The generated fill must exhibit no visible seams, blurring, or lighting inconsistencies, resulting in a coherent, high-fidelity 8K texture."
    negative_prompt = "blur, smoothing, flat shading, mismatched lighting, visible seams, artifacts, low resolution, baked lighting removal, cartoonish, loss of detail, interpolated pixels"

    # Set active LoRA
    switch_lora("inpainting")

    inpaint_image = pipe(
        ref_image,
        prompt,
        negative_prompt=negative_prompt,
        guidance_scale=1.0,
        num_inference_steps=50,
        num_images_per_prompt=1,
        generator=torch.Generator("cuda").manual_seed(43)
    ).images[0]

    inpaint_image.save(os.path.join(cache_dir, "inpainted_result.png"))

    switch_lora("delight")
    delight_image = pipe(
        inpaint_image,
        prompt,
        negative_prompt=negative_prompt,
        guidance_scale=2.0,
        num_inference_steps=50,
        num_images_per_prompt=1,
        generator=torch.Generator("cuda").manual_seed(43)
    ).images[0]

    delight_image.save(os.path.join(cache_dir, "delight_result.png"))

    switch_lora("albedo")
    albedo_image = pipe(
        delight_image,
        prompt,
        negative_prompt=negative_prompt,
        guidance_scale=2.0,
        num_inference_steps=50,
        num_images_per_prompt=1,
        generator=torch.Generator("cuda").manual_seed(43)
    ).images[0]
    albedo_image.save(os.path.join(cache_dir, "albedo_result.png"))

    switch_lora("rsd")
    rsd_image = pipe(
        delight_image,
        prompt,
        negative_prompt=negative_prompt,
        guidance_scale=2.0,
        num_inference_steps=50,
        num_images_per_prompt=1,
        generator=torch.Generator("cuda").manual_seed(43)
    ).images[0]
    rsd_image.save(os.path.join(cache_dir, "rsd_result.png"))

    switch_lora("normal")
    normal_image = pipe(
        delight_image,
        prompt,
        negative_prompt=negative_prompt,
        guidance_scale=2.0,
        num_inference_steps=50,
        num_images_per_prompt=1,
        generator=torch.Generator("cuda").manual_seed(43)
    ).images[0]
    normal_image.save(os.path.join(cache_dir, "normal_result.png"))

    # --- [New Step] Transition to IntrinsiXPipeline ---
    print("[INFO] Transitioning to IntrinsiXPipeline...")
    
    # 1. Disable adapters (using Base Model)
    # The current pipe.transformer is a PeftModel. We need to unload the adapters
    # to get back the pure FluxTransformer2DModel


    # 2. Inject Batched LoRA layers
    # This modifies the base_transformer in-place
    from longcat_image.models.batch_lora import inject_trainable_batched_lora
    from safetensors.torch import load_file
    from longcat_image.models.cross_intrinsic_attention import CrossIntrinsicAttnProcessor2_0

    features = ["albedo", "material", "normal"] # Make sure this matches training config
    lora_configs_align = []
    for feature in features:
         lora_configs_align.append({"r": 32, "dropout_p": 0.0, "scale": 1.0}) # Use config from training

    base_transformer = pipe.transformer.unload()

    inject_trainable_batched_lora(
        model=base_transformer,
        target_modules={"to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out", "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2"},
        lora_configs=lora_configs_align,
        verbose=False
    )

    # 3. Load Align LoRA weights
    align_lora_path = os.environ.get("ALIGN_LORA_PATH", "ckpts/lora_align/18000.safetensors")
    print(f"[INFO] Loading Align LoRA from: {align_lora_path}")
    
    align_state_dict = load_file(align_lora_path)
    # Process keys if necessary (remove "transformer." prefix)
    processed_align_state_dict = {}
    for k, v in align_state_dict.items():
        if k.startswith("transformer."):
            new_key = k.replace("transformer.", "")
        else:
            new_key = k
        processed_align_state_dict[new_key] = v
        
    base_transformer.load_state_dict(processed_align_state_dict, strict=False)

    # 4. Set Attention Processor
    # 5. Re-assemble IntrinsiXPipeline using existing components
    # This avoids reloading VAE, Text Encoder, Tokenizer from disk
    intrinsix_pipeline = IntrinsiXEditPipeline(
        scheduler=pipe.scheduler,
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        text_processor=pipe.text_processor,
        transformer=base_transformer,
    )
    crossattn_processor = CrossIntrinsicAttnProcessor2_0()
    intrinsix_pipeline.set_attn_processor(crossattn_processor)

    intrinsix_pipeline.to(device_str, torch.bfloat16)


    align_prompts = [
        "Ultra-High Definition 8K Diffuse Albedo map of a human face texture, flat lighting, unlit base color, pixel-perfect clarity. The texture reveals natural melanin pigmentation, hemoglobin redness, and distinct subsurface scattering warmth zones. It features high-frequency skin details including specific facial moles, freckles, and capillary variations. The chart is completely void of baked shadows, ambient occlusion, or specular highlights, representing pure biological skin color values in UV space.",
        "Ultra-High Definition 8K Surface Normal map of a human face texture in UV space, displaying sharp high-frequency relief with pixel-perfect clarity. The texture emphasizes intricate pore structures, fine wrinkles, and precise skin micro-geometry orientation relative to UV coordinates, where RGB vectors accurately represent surface angles and bumps without any albedo color or lighting information, achieving absolute biological structural realism.",
        "Ultra-High Definition 8K packed RSD technical texture map of a human face in UV space, consisting of three specific PBR channels. The Red channel encodes detailed micro-surface roughness and oiliness zones; the Green channel defines high-fidelity specular reflection intensity; and the Blue channel represents displacement depth for palpable pores and wrinkles. The map captures purely physical skin surface properties distinct from diffuse color or tangent normals.",
    ]
    negative_prompts = [
        "blur, smoothing, flat shading, mismatched lighting, visible seams, artifacts, low resolution, baked lighting removal, cartoonish, loss of detail, interpolated pixels",
        "blur, smoothing, flat shading, mismatched lighting, visible seams, artifacts, low resolution, baked lighting removal, cartoonish, loss of detail, interpolated pixels",
        "blur, smoothing, flat shading, mismatched lighting, visible seams, artifacts, low resolution, baked lighting removal, cartoonish, loss of detail, interpolated pixels",
    ]

    intrinsix_image_list = intrinsix_pipeline(
        delight_image,
        prompt_albedo=align_prompts[0],
        prompt_normal=align_prompts[1],
        prompt_rsd=align_prompts[2],
        negative_prompt_albedo=negative_prompts[0],
        negative_prompt_normal=negative_prompts[1],
        negative_prompt_rsd=negative_prompts[2],
        guidance_scale=2.0,
        num_inference_steps=50,
        num_images_per_prompt=1,
        generator=torch.Generator("cuda").manual_seed(43)
    ).images
    
    albedo_image, normal_image, rsd_image = intrinsix_image_list

    albedo_image.save(os.path.join(cache_dir, "albedo_align_result.png"))
    normal_image.save(os.path.join(cache_dir, "normal_align_result.png"))

    # rsd_image(R, G, B)are the roughness/specular/displacement textures respectively
    roughness_image = rsd_image.getchannel(0)  # Red channel - roughness
    specular_image = rsd_image.getchannel(1)   # Green channel - specular
    displacement_image = rsd_image.getchannel(2)  # Blue channel - displacement

    roughness_image.save(os.path.join(cache_dir, "roughness_align_result.png"))
    specular_image.save(os.path.join(cache_dir, "specular_align_result.png"))
    displacement_image.save(os.path.join(cache_dir, "displacement_align_result.png"))

    # Export to GLB
    glb_path = os.path.join(cache_dir, "result.glb")
    # print("[INFO] Exporting GLB (Note: Specular/Displacement maps are not supported in standard GLB materials and are saved separately)")
    print("[INFO] Exporting GLB with PBR material...")
    save_glb(
        path=glb_path,
        vertices=vertices,
        faces=faces,
        uvs=uvs,
        face_uvs=face_uvs,
        albedo=albedo_image,
        normal=normal_image,
        rsd=rsd_image
    )


    
    
