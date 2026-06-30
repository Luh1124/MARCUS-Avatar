import einops
import numpy as np
import torch
from torch import nn
from einops import einsum, rearrange

# ====================== LIGHTING ======================

class ConstantIntensity(nn.Module):
    def __init__(self,
                 value,
                 exp_val=True):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float32))
        self.exp_val = exp_val

    def forward(self, direction):
        val = self.value
        if self.exp_val:
            val = torch.exp(val)
        return val.unsqueeze(0).expand_as(direction)[..., :1]

    def reg_loss(self):
        val = self.value
        if self.exp_val:
            val = torch.exp(val)
        return torch.sum(val)

    def sample_direction(self, vpos, normal):
        # (bn, spp, 3, h, w)
        # Constant light comes from "everywhere", but for importance sampling/specular it's tricky.
        # It's usually integrated over the hemisphere.
        # For compatibility, we might return normal direction (emission from surface normal) or None.
        # Returning None might break downstream code expecting a tensor.
        # Let's return the normal as a dummy direction for "view dependent" effects if needed, 
        # but ConstantIntensity usually doesn't have a single direction.
        if normal is not None:
             return normal
        # If normal is None, we need a fallback. 
        # Return a dummy direction (e.g. up) with correct shape (B, 1, 3, 1, 1) or similar if possible.
        # But sample_direction is expected to return (bn, spp, 3, h, w).
        # Without input shape info (bn, h, w), we can't create correct shape.
        # Let's try returning a 1x1x3x1x1 tensor and let broadcasting handle it if possible, 
        # or raise error if vpos/normal required.
        return torch.tensor([[[[[0.0, 0.0, 1.0]]]]], device=self.value.device).permute(0, 1, 4, 2, 3) # (1, 1, 3, 1, 1)

    @property
    def spp(self):
        return 1
    
class DirectionalLight_LatLong(nn.Module):
    def __init__(self,
                 ch=1,
                 solid_angle=1,
                 weight_init=0):
        super().__init__()

        self.Num = 1

        self.ch = ch
        self.solid_angle = np.cos(np.deg2rad(solid_angle))

        self.weight, self.theta, self.phi = self.init_pos(weight_init=weight_init)

        is_enabled = torch.tensor(True)
        self.register_buffer('is_enabled', is_enabled)

    def init_pos(self, weight_init=0):
        weight = nn.Parameter(-torch.ones((1, self.ch), dtype=torch.float32) + weight_init)
        theta = nn.Parameter(torch.ones((1, 1), dtype=torch.float32) * -1 * torch.pi / 2)
        phi = nn.Parameter(torch.zeros((1, 1), dtype=torch.float32))

        return weight, theta, phi

    def deparameterize(self):
        theta = self.theta
        phi = self.phi

        weight = self.deparameterize_weight()

        return weight, theta, phi

    def deparameterize_weight(self):
        weight = torch.exp(self.weight)
        return weight

    def get_axis(self, theta, phi):
        # Get axis
        axisX = torch.sin(theta) * torch.sin(phi)
        axisY = torch.cos(theta)
        axisZ = -torch.sin(theta) * torch.cos(phi)

        axis = torch.cat([axisX, axisY, axisZ], dim=1)

        return axis

    def forward(self, direction):
        if self.is_enabled:
            weight, theta, phi = self.deparameterize()
            axis = self.get_axis(theta, phi)

            if direction.ndim == 2:
                cos_angle = einsum(direction, axis, 'b c, sg c -> b sg')
                cos_angle = rearrange(cos_angle, 'b sg -> b sg 1')
                weight = rearrange(weight, 'sg c -> 1 sg c')
                weight = weight * (cos_angle > self.solid_angle).float()
                val = torch.sum(weight, dim=1)
            elif direction.ndim == 3:
                cos_angle = einsum(direction, axis, 'sg b c, sg c -> b sg')
                cos_angle = rearrange(cos_angle, 'b sg -> b sg 1')
                weight = rearrange(weight, 'sg c -> 1 sg c')
                weight = weight * (cos_angle > self.solid_angle).float()
                val = weight
                val = rearrange(val, 'b sg c -> sg b c')
            else:
                raise NotImplementedError()
        else:
            val = torch.zeros_like(direction)
        return val

    def reg_loss(self):
        if self.is_enabled:
            val = self.deparameterize_weight()
            val = torch.sum(val)

            return val
        else:
            return torch.tensor(0, device=self.weight.device, dtype=torch.float32)

    @torch.no_grad()
    def to_envmap(self, size=(256, 512)):
        envHeight, envWidth = size

        phi = np.linspace(-np.pi, np.pi, envWidth)
        theta = np.linspace(0, np.pi, envHeight)
        phi, theta = np.meshgrid(phi, theta)

        phi = torch.from_numpy(phi)[None, None]
        theta = torch.from_numpy(theta)[None, None]

        directions = self.get_axis(theta, phi)

        directions = einops.rearrange(directions, 'b c h w -> (b h w) c').to(torch.float32).to(self.theta.device)
        envmap = self.forward(directions)
        envmap = einops.rearrange(envmap, '(b h w) c -> b c h w', h=envHeight, w=envWidth)
        return envmap

    def sample_direction(self, vpos, normal):
        # (bn, spp, 3, h, w)
        weight, theta, phi = self.deparameterize()
        axis = self.get_axis(theta, phi)
        return axis[None, :, :, None, None]
        # return axis[None, :, :, None, None].expand_as(normal)

    @property
    def spp(self):
        return 1
    
class DirectionalLight(nn.Module):
    def __init__(self,
                 ch=1,
                 solid_angle=5,
                 weight_init=0):
        super().__init__()

        self.Num = 1

        self.ch = ch
        self.solid_angle = np.cos(np.deg2rad(solid_angle))

        self.init_pos(weight_init=weight_init)

        is_enabled = torch.tensor(True)
        self.register_buffer('is_enabled', is_enabled)

    def init_pos(self, weight_init=0):
        # weight = nn.Parameter(-torch.ones((1, self.ch), dtype=torch.float32) + weight_init)
        # direction = nn.Parameter(torch.tensor(((0, 0, -1),), dtype=torch.float32))
        weight = -torch.ones((1, self.ch), dtype=torch.float32) + weight_init
        direction = torch.tensor(((0, 0, -1),), dtype=torch.float32)

        self.register_buffer('weight', weight)
        self.register_buffer('direction', direction)

        return weight, direction

    def deparameterize(self):
        direction = self.direction

        weight = self.deparameterize_weight()

        return weight, direction

    def deparameterize_weight(self):
        weight = torch.exp(self.weight)
        return weight

    def get_axis(self, direction):
        # Get axis
        direction = torch.nn.functional.normalize(direction, dim=1)

        return direction

    def get_axis_from_angle(self, theta, phi):
        # Get axis
        axisX = torch.sin(theta) * torch.sin(phi)
        axisY = torch.cos(theta)
        axisZ = -torch.sin(theta) * torch.cos(phi)

        axis = torch.cat([axisX, axisY, axisZ], dim=1)

        return axis


    def forward(self, direction):
        if self.is_enabled:
            weight, direction = self.deparameterize()
            axis = self.get_axis(direction)

            if direction.ndim == 2:
                cos_angle = einsum(direction, axis, 'b c, sg c -> b sg')
                cos_angle = rearrange(cos_angle, 'b sg -> b sg 1')
                weight = rearrange(weight, 'sg c -> 1 sg c')
                weight = weight * (cos_angle > self.solid_angle).to(weight.dtype)
                val = torch.sum(weight, dim=1)
            elif direction.ndim == 3:
                cos_angle = einsum(direction, axis, 'sg b c, sg c -> b sg')
                cos_angle = rearrange(cos_angle, 'b sg -> b sg 1')
                weight = rearrange(weight, 'sg c -> 1 sg c')
                weight = weight * (cos_angle > self.solid_angle).to(weight.dtype)
                val = weight
                val = rearrange(val, 'b sg c -> sg b c')
            else:
                raise NotImplementedError()
        else:
            val = torch.zeros_like(direction)
        return val

    def reg_loss(self):
        if self.is_enabled:
            val = self.deparameterize_weight()
            val = torch.sum(val)

            return val
        else:
            return torch.tensor(0, device=self.weight.device, dtype=torch.float32)

    @torch.no_grad()
    def to_envmap(self, size=(256, 512)):
        envHeight, envWidth = size

        phi = np.linspace(-np.pi, np.pi, envWidth)
        theta = np.linspace(0, np.pi, envHeight)
        phi, theta = np.meshgrid(phi, theta)

        phi = torch.from_numpy(phi)[None, None]
        theta = torch.from_numpy(theta)[None, None]

        directions = self.get_axis_from_angle(theta, phi)

        directions = einops.rearrange(directions, 'b c h w -> (b h w) c').to(torch.float32).to(self.theta.device)
        envmap = self.forward(directions)
        envmap = einops.rearrange(envmap, '(b h w) c -> b c h w', h=envHeight, w=envWidth)
        return envmap

    def sample_direction(self, vpos, normal):
        # (bn, spp, 3, h, w)
        weight, direction = self.deparameterize()
        axis = self.get_axis(direction)
        return axis[None, :, :, None, None]
        # return axis[None, :, :, None, None].expand_as(normal)

    @property
    def spp(self):
        return 1

class GlobalIncidentLighting(nn.Module):
    def __init__(self,
                 value=ConstantIntensity((-2, -2, -2), exp_val=True)):
        super().__init__()
        self.value = value

    @property
    def spp(self):
        return self.value.spp

    def sample_direction(self, vpos=None, normal=None, specific_dir=None):
        # If specific_dir is provided, use it instead of sampling
        if specific_dir is not None:
            # specific_dir: (B, 1, 3) or (B, K, 3)
            # Return format: (B, K, 3) where K is number of light samples
            if specific_dir.dim() == 3:
                # Already in (B, K, 3) format
                return specific_dir
            elif specific_dir.dim() == 2:
                # (B, 3) -> (B, 1, 3)
                return specific_dir.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected specific_dir shape: {specific_dir.shape}")
        
        # Otherwise, use the default sampling from value
        # value.sample_direction returns (bn, spp, 3, h, w) format
        # We need to convert to (B, K, 3) format for render.py
        sampled = self.value.sample_direction(vpos, normal)
        
        # Check if sampled is None (e.g. ConstantIntensity with no normal input)
        if sampled is None:
             # Fallback: assume light from camera/view direction or just frontal
             # render.py usually doesn't pass vpos/normal to sample_direction in forward() initially?
             # Actually render.py passes normal=None.
             # Return a default frontal direction (B, 1, 3)
             # We need batch size... hard to guess without input.
             # Return a generic (1, 1, 3) and hope for broadcasting
             return torch.tensor([[[0.0, 0.0, 1.0]]], device=self.value.value.device)

        # sampled: (bn, spp, 3, h, w) -> (bn, spp, 3)
        # Take the first spatial pixel (or could use mean, but first is faster)
        if sampled.dim() == 5:
            bn, spp, c, h, w = sampled.shape
            # Extract direction at center pixel: (bn, spp, 3)
            sampled_reshaped = sampled[:, :, :, h//2, w//2]
            return sampled_reshaped  # (B, K, 3) where K = spp
        elif sampled.dim() == 4:
            # Could be (B, 3, H, W) from ConstantIntensity returning normal
            # or (B, K, 3, 1) from LatLong
            if sampled.shape[1] == 3 and sampled.shape[2] > 3:
                # (B, 3, H, W) format - take center pixel as light direction
                bn, c, h, w = sampled.shape
                sampled_center = sampled[:, :, h//2, w//2]  # (B, 3)
                return sampled_center.unsqueeze(1)  # (B, 1, 3)
            else:
                # (B, K, 3, 1) format - squeeze last dim
                return sampled.squeeze(-1)
        elif sampled.dim() == 3:
            # Already in (B, K, 3) format
            return sampled
        else:
            raise ValueError(f"Unexpected sampled direction shape: {sampled.shape}")

    def pdf_direction(self, vpos, direction):
        # (bn, spp, 3, h, w)
        return torch.ones_like(direction[:, :, :1, ...])

    def forward(self, direction):
        # direction: (B, K, 3, H, W) from render.py
        # value.forward() expects different formats, need to reshape
        
        if direction.dim() == 5:
            # (B, K, 3, H, W) -> need to convert for value.forward()
            bn, k, c, h, w = direction.shape
            
            # Check channels - we expect 3 channels for direction
            if c != 3:
                 raise ValueError(f"Expected 3 channels for direction, got {c}. Input shape: {direction.shape}")

            # Reshape to (B*K*H*W, 3) for processing
            # First move channels to last: (B, K, H, W, 3)
            direction_permuted = direction.permute(0, 1, 3, 4, 2).contiguous()
            direction_flat = direction_permuted.view(-1, 3)  # (B*K*H*W, 3)
            
            # Call value.forward() which expects (B, C) format (ndim=2)
            # DirectionalLight.forward() returns (b, ch) where ch is self.ch (default 1)
            val_flat = self.value(direction_flat)  # Should be (B*K*H*W, ch) where ch=1 typically
            
            # Ensure val_flat has the right shape
            expected_size = bn * k * h * w
            if val_flat.numel() == 1:
                # Scalar or single value - broadcast to all positions
                val_scalar = val_flat.item() if val_flat.dim() == 0 else val_flat[0].item()
                val_reshaped = val_scalar * torch.ones((bn, k, 3, h, w), device=direction.device, dtype=direction.dtype)
                return val_reshaped
            
            # Reshape back to (B, K, ch, H, W)
            if val_flat.dim() == 2:
                b_flat, val_channels = val_flat.shape
                # Check if shape matches expected
                expected_size = bn * k * h * w
                if b_flat != expected_size:
                    # If shape doesn't match, it might be broadcasted or wrong
                    # Try to handle it by repeating or using fallback
                    if b_flat == 1:
                        # Single value returned, broadcast it
                        val_flat = val_flat.expand(expected_size, val_channels)
                    else:
                        # Unexpected shape, use fallback
                        return torch.ones((bn, k, 3, h, w), device=direction.device, dtype=direction.dtype)
                
                # Handle both (B*K*H*W, 1) and (B*K*H*W, ch) cases
                if val_channels == 1:
                    val_reshaped = val_flat.view(bn, k, h, w, 1)  # (B, K, H, W, 1)
                    val_reshaped = val_reshaped.permute(0, 1, 4, 2, 3)  # (B, K, 1, H, W)
                    # Expand to 3 channels (RGB)
                    val_reshaped = val_reshaped.expand(-1, -1, 3, -1, -1)  # (B, K, 3, H, W)
                else:
                    val_reshaped = val_flat.view(bn, k, h, w, val_channels)
                    val_reshaped = val_reshaped.permute(0, 1, 4, 2, 3)  # (B, K, C, H, W)
                    # If not 3 channels, expand or repeat to 3
                    if val_channels != 3:
                        val_reshaped = val_reshaped.expand(-1, -1, 3, -1, -1)  # (B, K, 3, H, W)
                
                return val_reshaped
            elif val_flat.dim() == 1:
                # Single channel output: (B*K*H*W,) or possibly scalar
                if val_flat.numel() == 1:
                    # Scalar value, broadcast to all positions
                    val_reshaped = val_flat.item() * torch.ones((bn, k, 1, h, w), device=direction.device, dtype=direction.dtype)
                    val_reshaped = val_reshaped.expand(-1, -1, 3, -1, -1)  # (B, K, 3, H, W)
                    return val_reshaped
                else:
                    # Vector output
                    val_reshaped = val_flat.view(bn, k, h, w, 1)  # (B, K, H, W, 1)
                    val_reshaped = val_reshaped.permute(0, 1, 4, 2, 3)  # (B, K, 1, H, W)
                    val_reshaped = val_reshaped.expand(-1, -1, 3, -1, -1)  # (B, K, 3, H, W)
                    return val_reshaped
            elif val_flat.dim() == 0:
                # Scalar tensor
                val_scalar = val_flat.item()
                val_reshaped = val_scalar * torch.ones((bn, k, 3, h, w), device=direction.device, dtype=direction.dtype)
                return val_reshaped
            else:
                # Fallback: create uniform light
                return torch.ones((bn, k, 3, h, w), device=direction.device, dtype=direction.dtype)
        else:
            # For other dimensions, use value.forward() directly
            return self.value(direction)

    def val_reg_loss(self):
        return torch.zeros_like(self.value.reg_loss())

    def pos_reg_loss(self):
        return torch.zeros_like(self.value.reg_loss())

class RandomizedDirectionalLight(nn.Module):
    def __init__(self, min_elevation=-30, max_elevation=30, device='cuda'):
        super().__init__()
        self.device = device
        # limit elevation range to avoid light from extreme top or bottom
        self.min_elevation = np.deg2rad(min_elevation)
        self.max_elevation = np.deg2rad(max_elevation)
        
        # default direction (B, 1, 3)
        self.register_buffer('direction', torch.tensor([[[0.0, 0.0, 1.0]]], device=device))
        self.register_buffer('color', torch.tensor([[[1.0, 1.0, 1.0]]], device=device))

    def randomize(self, batch_size=1, camera_pos=None):
        """
        call before each rendering, randomize the lighting
        """
        # Strategy A (70%): light follows camera direction and is perturbed around it (simulate photography light/front light)
        # Strategy B (30%): completely random direction (simulate back light/side light, train contour and Fresnel)
        
        if camera_pos is not None and torch.rand(1) < 0.7:
            # camera_pos shape: (B, 3, 1, 1) -> (B, 1, 3)
            cam_dir = torch.nn.functional.normalize(camera_pos.view(batch_size, 1, 3), dim=2)
            noise = torch.randn_like(cam_dir) * 0.5 # 0.5 perturbation amplitude
            new_dir = torch.nn.functional.normalize(cam_dir + noise, dim=2)
        else:
            # pure random spherical sampling
            # randomly generate (B, 1, 3) vector
            rand_vec = torch.randn(batch_size, 1, 3, device=self.device)
            new_dir = torch.nn.functional.normalize(rand_vec, dim=2)

        self.direction = new_dir # (B, 1, 3)
        
        # optional: slightly randomize light color/intensity (0.8 ~ 1.2)
        intensity = 2.5 + torch.rand(batch_size, 1, 1, device=self.device) * 0.6
        self.color = torch.tensor([[[1.0, 1.0, 1.0]]], device=self.device) * intensity

    def sample_direction(self, vpos, normal, specific_dir=None):
        if specific_dir is not None:
            return specific_dir
        return self.direction

    def forward(self, wo):
        # wo can be:
        #   - (B, K, 3, H, W) when called directly
        #   - (N, 3) when called from GlobalIncidentLighting (flattened)
        if wo.dim() == 5:
            b, k, _, h, w = wo.shape
            # expand color to the whole image
            return self.color.view(1, 1, 3, 1, 1).expand(b, k, -1, h, w)
        elif wo.dim() == 2:
            # Flattened input (N, 3) from GlobalIncidentLighting
            # Return (N, 1) or (N, 3) depending on expected output
            n = wo.shape[0]
            # color is (B, 1, 3), we need to return scalar intensity per direction
            # For simplicity, return the mean color intensity
            intensity = self.color.mean()
            return intensity * torch.ones((n, 1), device=wo.device, dtype=wo.dtype)
        else:
            raise ValueError(f"Unexpected wo shape: {wo.shape}")

class MockLighting(nn.Module):
    def __init__(self, use_fast_mode=True):
        super().__init__()
        self.use_fast_mode = use_fast_mode
    def sample_direction(self, vpos, normal, specific_dir):
        # Fast mode: return single light direction
        # This avoids expensive multi-light BRDF calculations
        return specific_dir  # (B, 1, 3)
        
    def forward(self, wo):
        # wo can be:
        #   - (B, K, 3, H, W) when called directly
        #   - (N, 3) when called from GlobalIncidentLighting (flattened)
        if wo.dim() == 5:
            bn, k, _, h, w = wo.shape
            device = wo.device
            # Fast mode: single bright light
            # Return white light intensity = 1.0
            light_radiance = torch.ones((bn, k, 3, h, w), device=device)
            return light_radiance  # (B, K, 3, H, W)
        elif wo.dim() == 2:
            # Flattened input (N, 3) from GlobalIncidentLighting
            n = wo.shape[0]
            return torch.ones((n, 1), device=wo.device, dtype=wo.dtype)
        else:
            raise ValueError(f"Unexpected wo shape: {wo.shape}")