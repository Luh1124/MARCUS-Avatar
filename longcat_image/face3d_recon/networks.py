"""This script defines deep neural networks for Deep3DFaceRecon_pytorch
"""

import os
import functools
import os
from typing import Any, Callable, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry import warp_affine
from torch.nn import init
from torch.optim import lr_scheduler

from torchvision import models as tv_models
from transformers import AutoModel, AutoConfig
import timm

# from .arcface_torch.backbones import get_model

DEFAULT_DINOV3_PATH = os.environ.get(
    "DINOV3_PATH",
    os.path.join("ckpts", "dinov3-convnext-small-pretrain-lvd1689m"),
)

# DepthAnything3 is imported lazily to avoid libGL.so.1 dependency when not needed
# from depth_anything_3.api import DepthAnything3

from torch import Tensor
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

def resize_n_crop(image, M, dsize=112):
    # image: (b, c, h, w)
    # M   :  (b, 2, 3)
    return warp_affine(image, M, dsize=(dsize, dsize))

def filter_state_dict(state_dict, remove_name='fc'):
    new_state_dict = {}
    for key in state_dict:
        if remove_name in key:
            continue
        new_state_dict[key] = state_dict[key]
    return new_state_dict

def load_convnextv2_weights(init_path: str):
    """
    Load ConvNeXtV2 weights from local path.
    Supports:
    1. Single weight file (.pth, .bin)
    2. Hugging Face format directory (contains pytorch_model.bin or model.safetensors)
    
    Args:
        init_path: Path to weight file or model directory
        
    Returns:
        state_dict or None if loading fails
    """
    if not init_path or not os.path.exists(init_path):
        return None
    
    try:
        # Case 1: Single weight file
        if os.path.isfile(init_path):
            print(f"Loading ConvNeXtV2 weights from file: {init_path}")
            if init_path.endswith('.safetensors'):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(init_path)
                    return state_dict
                except ImportError:
                    print("⚠ Warning: safetensors not installed, cannot load .safetensors file")
                    return None
            else:
                state_dict = torch.load(init_path, map_location='cpu')
                return state_dict
        
        # Case 2: Hugging Face format directory
        elif os.path.isdir(init_path):
            print(f"Loading ConvNeXtV2 weights from directory: {init_path}")
            
            # Try pytorch_model.bin first
            pytorch_model_path = os.path.join(init_path, 'pytorch_model.bin')
            if os.path.isfile(pytorch_model_path):
                state_dict = torch.load(pytorch_model_path, map_location='cpu')
                return state_dict
            
            # Try model.safetensors
            safetensors_path = os.path.join(init_path, 'model.safetensors')
            if os.path.isfile(safetensors_path):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(safetensors_path)
                    return state_dict
                except ImportError:
                    print("⚠ Warning: safetensors not installed, cannot load .safetensors file")
                    return None
            
            print(f"⚠ Warning: No weight file found in {init_path}")
            print("   Expected: pytorch_model.bin or model.safetensors")
            return None
            
    except Exception as e:
        print(f"⚠ Warning: Failed to load weights from {init_path}: {e}")
        return None

def reshape_dinov3_features(feat_map: torch.Tensor) -> torch.Tensor:
    """
    Reshape DINOv3 output from (B, N, C) to (B, C, H, W).
    
    Handles cases where:
    - N is a perfect square (H*W = N)
    - N includes CLS token (N = H*W + 1)
    - N is rectangular (e.g., (H+1)*H = N)
    
    Args:
        feat_map: Input tensor of shape (B, N, C)
    
    Returns:
        Reshaped tensor of shape (B, C, H, W)
    """
    if feat_map.dim() != 3:
        return feat_map
    
    B, N, C = feat_map.shape
    
    # Try to find H and W such that H*W = N or H*W = N-1
    H_sqrt = int(N ** 0.5)
    
    if H_sqrt * H_sqrt == N:
        # Perfect square (no CLS token or CLS already removed)
        H, W = H_sqrt, H_sqrt
        return feat_map.permute(0, 2, 1).view(B, C, H, W)
    elif (H_sqrt + 1) * H_sqrt == N:
        # Rectangular: (H+1) * H = N
        H, W = H_sqrt + 1, H_sqrt
        return feat_map.permute(0, 2, 1).view(B, C, H, W)
    elif H_sqrt * H_sqrt == N - 1:
        # Has CLS token: N = H*W + 1
        H, W = H_sqrt, H_sqrt
        # Remove CLS token (first token)
        feat_map = feat_map[:, 1:, :]  # (B, N-1, C)
        return feat_map.permute(0, 2, 1).view(B, C, H, W)
    else:
        # Fallback: find closest H, W pair
        # Try to find factors of N or N-1
        target_N = N
        if (H_sqrt + 1) * H_sqrt < N:
            # Likely has CLS token
            target_N = N - 1
            feat_map = feat_map[:, 1:, :]  # Remove CLS token
        
        # Find closest square or rectangular shape
        H = int(target_N ** 0.5)
        W = (target_N + H - 1) // H  # Ceiling division
        # Adjust if needed
        if H * W > target_N:
            W = target_N // H
        
        # Reshape with truncation if needed
        if H * W == target_N:
            return feat_map.permute(0, 2, 1).view(B, C, H, W)
        else:
            # Truncate to make it work
            feat_map = feat_map[:, :H*W, :]  # Truncate
            return feat_map.permute(0, 2, 1).view(B, C, H, W)

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    
    For cosine scheduler, supports custom parameters:
        - opt.cosine_T_max: Maximum number of iterations (default: opt.n_epochs)
        - opt.cosine_eta_min: Minimum learning rate (default: 0)
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_epochs, gamma=0.2)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        # Support custom CosineAnnealing parameters
        T_max = getattr(opt, 'cosine_T_max', opt.n_epochs)
        eta_min = getattr(opt, 'cosine_eta_min', 0)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        if hasattr(opt, 'cosine_T_max') or hasattr(opt, 'cosine_eta_min'):
            print(f"Using CosineAnnealingLR with T_max={T_max}, eta_min={eta_min}")
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def define_net_recon(net_recon, use_last_fc=False, init_path=None):
    """
    Create reconstruction network backbone.

    - For standard ResNet backbones, returns ReconNetWrapper as before.
    - For 'spatial_resnet_da3', now returns MultiScaleFusedResNetDA3 (multi-scale fusion, DEPRECATED name).
    - For 'spatial_resnet_dinov3', now returns MultiScaleFusedResNetDINOv3 (multi-scale fusion, DEPRECATED name).
    - For 'spatial_convnextv2_base_dinov3', now returns MultiScaleFusedConvNeXtV2DINOv3 (multi-scale fusion, DEPRECATED name).
    - For 'spatial_convnextv2_base_da3', now returns MultiScaleFusedConvNeXtV2DA3 (multi-scale fusion, DEPRECATED name).
    - For 'multiscale_resnet_da3', returns MultiScaleFusedResNetDA3 (multi-scale fusion).
    - For 'multiscale_convnextv2_base_da3', returns MultiScaleFusedConvNeXtV2DA3 (multi-scale fusion).
    - For 'multiscale_resnet_dinov3', returns MultiScaleFusedResNetDINOv3 (multi-scale fusion with DINOv3).
    - For 'multiscale_convnextv2_base_dinov3', returns MultiScaleFusedConvNeXtV2DINOv3 (multi-scale fusion with DINOv3).
    """
    if net_recon == 'spatial_resnet_da3':
        # DEPRECATED: Now uses multi-scale fusion for better performance.
        # Multi-Scale Fusion: ResNet50 (trainable) + DA3-Small (frozen)
        # Fuse at multiple scales (Layer 1, 2, 3, 4) with HyperColumn aggregation.
        # For the current HIFI ParametricFaceModel, coefficient dim is 1049.
        return MultiScaleFusedResNetDA3(num_outputs=1049, init_path=init_path)

    if net_recon == 'spatial_resnet_dinov3':
        # DEPRECATED: Now uses multi-scale fusion for better performance.
        # Multi-Scale Fusion: ResNet50 (trainable) + DINOv3 (frozen)
        # Fuse at multiple scales (Layer 1, 2, 3, 4) with HyperColumn aggregation.
        # Uses DINOv3's intermediate hidden states for true high-resolution fusion.
        # For the current HIFI ParametricFaceModel, coefficient dim is 1049.
        return MultiScaleFusedResNetDINOv3(num_outputs=1049, init_path=init_path)
    
    if net_recon == 'spatial_convnextv2_base_dinov3':
        # DEPRECATED: Now uses multi-scale fusion for better performance.
        # Multi-Scale Fusion: ConvNeXt V2 Base (trainable) + DINOv3 (frozen)
        # Fuse at multiple scales (Stage 0, 1, 2, 3) with HyperColumn aggregation.
        # For the current HIFI ParametricFaceModel, coefficient dim is 1049.
        return MultiScaleFusedConvNeXtV2DINOv3(num_outputs=1049, init_path=init_path)
    
    if net_recon == 'spatial_convnextv2_base_da3':
        # DEPRECATED: Now uses multi-scale fusion for better performance.
        # Multi-Scale Fusion: ConvNeXt V2 Base (trainable) + DA3-Small (frozen)
        # Fuse at multiple scales (Stage 0, 1, 2, 3) with HyperColumn aggregation.
        # For the current HIFI ParametricFaceModel, coefficient dim is 1049.
        return MultiScaleFusedConvNeXtV2DA3(num_outputs=1049, init_path=init_path)
    
    if net_recon == 'multiscale_resnet_da3':
        # Multi-Scale Fusion: ResNet50 (trainable) + DA3-Small (frozen)
        # Fuse at multiple scales (Layer 1, 2, 3, 4) with HyperColumn aggregation.
        # Uses 3-layer MLP head instead of single Linear layer.
        # For the current HIFI ParametricFaceModel, coefficient dim is 1049.
        return MultiScaleFusedResNetDA3(num_outputs=1049, init_path=init_path)
    
    if net_recon == 'multiscale_convnextv2_base_da3':
        # Multi-Scale Fusion: ConvNeXt V2 Base (trainable) + DA3-Small (frozen)
        # Fuse at multiple scales (Stage 0, 1, 2, 3) with HyperColumn aggregation.
        # Uses 3-layer MLP head instead of single Linear layer.
        # For the current HIFI ParametricFaceModel, coefficient dim is 1049.
        return MultiScaleFusedConvNeXtV2DA3(num_outputs=1049, init_path=init_path)
    
    if net_recon == 'multiscale_resnet_dinov3':
        # Multi-Scale Fusion: ResNet50 (trainable) + DINOv3 (frozen)
        # Fuse at multiple scales (Layer 1, 2, 3, 4) with HyperColumn aggregation.
        # Uses DINOv3's intermediate hidden states for true high-resolution fusion.
        # Uses 3-layer MLP head instead of single Linear layer.
        # For the current HIFI ParametricFaceModel, coefficient dim is 1049.
        return MultiScaleFusedResNetDINOv3(num_outputs=1049, init_path=init_path)
    
    if net_recon == 'multiscale_convnextv2_base_dinov3':
        # Multi-Scale Fusion: ConvNeXt V2 Base (trainable) + DINOv3 (frozen)
        # Fuse at multiple scales (Stage 0, 1, 2, 3) with HyperColumn aggregation.
        # Homogeneous architecture fusion (ConvNeXt + ConvNeXt).
        # Uses 3-layer MLP head instead of single Linear layer.
        # For the current HIFI ParametricFaceModel, coefficient dim is 1049.
        return MultiScaleFusedConvNeXtV2DINOv3(num_outputs=1049, init_path=init_path)
    
    if net_recon == 'multiscale_convnextv2_base_da3_enhanced':
        # Enhanced Multi-Scale Fusion: ConvNeXt V2 Base (trainable) + DA3 (frozen) + Spatial Attention
        # Adds Spatial Attention Modules at each stage after fusion to focus on high-frequency features.
        # For the current HIFI ParametricFaceModel, coefficient dim is 1049.
        return MultiScaleFusedConvNeXtV2DA3_Enhanced(num_outputs=1049, init_path=init_path)
    
    if net_recon == 'multiscale_convnextv2_base_dinov3_enhanced':
        # Enhanced Multi-Scale Fusion: ConvNeXt V2 Base (trainable) + DINOv3 (frozen) + Spatial Attention
        # Adds Spatial Attention Modules at each stage after fusion to focus on high-frequency features.
        # For the current HIFI ParametricFaceModel, coefficient dim is 1049.
        return MultiScaleFusedConvNeXtV2DINOv3_Enhanced(num_outputs=1049, init_path=init_path)

    return ReconNetWrapper(net_recon, use_last_fc=use_last_fc, init_path=init_path)

def define_net_recog(net_recog, pretrained_path=None):
    net = RecogNetWrapper(net_recog=net_recog, pretrained_path=pretrained_path)
    net.eval()
    return net

class ReconNetWrapper(nn.Module):
    fc_dim=532+45+439+3+27+2+1
    def __init__(self, net_recon, use_last_fc=False, init_path=None):
        super(ReconNetWrapper, self).__init__()
        self.use_last_fc = use_last_fc
        if net_recon not in func_dict:
            return  NotImplementedError('network [%s] is not implemented', net_recon)
        func, last_dim = func_dict[net_recon]
        backbone = func(use_last_fc=use_last_fc, num_classes=self.fc_dim)
        if init_path and os.path.isfile(init_path):
            checkpoint = torch.load(init_path, map_location='cpu')
            # Handle full model checkpoint (contains 'net_recon', 'opt_00', etc.)
            if 'net_recon' in checkpoint:
                state_dict = checkpoint['net_recon']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            state_dict = filter_state_dict(state_dict)
            backbone.load_state_dict(state_dict, strict=False)
            print("loading init net_recon %s from %s" %(net_recon, init_path))
        self.backbone = backbone
        if not use_last_fc:
            self.final_layers = nn.ModuleList([
                # conv1x1(last_dim, 80, bias=True), # id layer
                # conv1x1(last_dim, 64, bias=True), # exp layer
                # conv1x1(last_dim, 80, bias=True), # tex layer
                # conv1x1(last_dim, 3, bias=True),  # angle layer
                # conv1x1(last_dim, 27, bias=True), # gamma layer
                # conv1x1(last_dim, 2, bias=True),  # tx, ty
                # conv1x1(last_dim, 1, bias=True)   # tz
                conv1x1(last_dim, 532, bias=True), # id layer
                conv1x1(last_dim, 45, bias=True), # exp layer
                conv1x1(last_dim, 439, bias=True), # tex layer
                conv1x1(last_dim, 3, bias=True),  # angle layer
                conv1x1(last_dim, 27, bias=True), # gamma layer
                conv1x1(last_dim, 2, bias=True),  # tx, ty
                conv1x1(last_dim, 1, bias=True)   # tz
            ])
            for m in self.final_layers:
                nn.init.constant_(m.weight, 0.)
                nn.init.constant_(m.bias, 0.)

    def load_state_dict(self, state_dict, strict=True):
        """
        Smart loading that handles:
        1. Key name mapping: 
           - resnet_* -> backbone.*
           - trainable_backbone.resnet_* -> backbone.*
           - trainable_backbone.* -> backbone.* (for SpatialFusedResNetDINOv3)
             - trainable_backbone.0.* -> backbone.conv1.*
             - trainable_backbone.1.* -> backbone.bn1.*
             - trainable_backbone.2.* -> backbone.relu.*
             - trainable_backbone.3.* -> backbone.maxpool.*
             - trainable_backbone.4.* -> backbone.layer1.*
             - trainable_backbone.5.* -> backbone.layer2.*
             - trainable_backbone.6.* -> backbone.layer3.*
             - trainable_backbone.7.* -> backbone.layer4.*
           - backbone.* -> backbone.* (for ConvNeXt V2 models, keep as-is)
        2. Filtering: Remove da3.*, dinov3.*, adapter.*, pre_norm.*, and gate keys
        3. Shape conversion: Linear [out, in] -> Conv2d [out, in, 1, 1] for final_layers
        4. Dimension mismatch detection: Check if final_layers input dimensions match
        """
        # First, detect dimension mismatch before processing
        checkpoint_dim = None
        model_dim = None
        
        # Find checkpoint's final_layers input dimension
        for key, value in state_dict.items():
            if key.startswith('final_layers.') and 'weight' in key:
                parts = key.split('.')
                if len(parts) >= 3:
                    try:
                        layer_idx = int(parts[1])
                        if parts[2] == 'weight':
                            # Check shape: Linear [out, in] or Conv2d [out, in, 1, 1]
                            if len(value.shape) == 2:  # Linear
                                checkpoint_dim = value.shape[1]  # input dimension
                            elif len(value.shape) == 4:  # Conv2d
                                checkpoint_dim = value.shape[1]  # input dimension
                            break
                    except (ValueError, IndexError):
                        continue
        
        # Get current model's final_layers input dimension
        if hasattr(self, 'final_layers') and len(self.final_layers) > 0:
            first_layer = self.final_layers[0]
            if hasattr(first_layer, 'weight'):
                if len(first_layer.weight.shape) == 4:  # Conv2d
                    model_dim = first_layer.weight.shape[1]
                elif len(first_layer.weight.shape) == 2:  # Linear
                    model_dim = first_layer.weight.shape[1]
        
        # Check for dimension mismatch and provide helpful error message
        if checkpoint_dim is not None and model_dim is not None and checkpoint_dim != model_dim:
            # Determine which net_recon should be used based on checkpoint dimension
            suggested_net_recon = None
            if checkpoint_dim == 1024:
                # ConvNeXt V2 Base models use 1024
                if any('da3' in k for k in state_dict.keys()):
                    suggested_net_recon = 'spatial_convnextv2_base_da3'
                elif any('dinov3' in k for k in state_dict.keys()):
                    suggested_net_recon = 'spatial_convnextv2_base_dinov3'
                else:
                    suggested_net_recon = 'spatial_convnextv2_base_da3'  # default guess
            elif checkpoint_dim == 2048:
                # ResNet50 models use 2048
                if any('da3' in k for k in state_dict.keys()):
                    suggested_net_recon = 'spatial_resnet_da3'
                elif any('dinov3' in k for k in state_dict.keys()):
                    suggested_net_recon = 'spatial_resnet_dinov3'
                else:
                    suggested_net_recon = 'resnet50'  # default
            
            error_msg = f"""
❌ DIMENSION MISMATCH ERROR ❌

Checkpoint expects input dimension: {checkpoint_dim}
Current model has input dimension: {model_dim}

This usually means you're using the wrong --net_recon parameter.

Based on the checkpoint, you should use:
  --net_recon={suggested_net_recon}

Common dimension mappings:
  - 1024: ConvNeXt V2 Base models (spatial_convnextv2_base_da3, spatial_convnextv2_base_dinov3)
  - 2048: ResNet50 models (resnet50, spatial_resnet_da3, spatial_resnet_dinov3)

Please check:
  1. The --name parameter matches the checkpoint directory
  2. The --net_recon parameter matches the training configuration
  3. The checkpoint was trained with the same architecture

Example fix:
  python test.py --name=convnextv2_base_da3 --net_recon=spatial_convnextv2_base_da3 --img_folder=...
"""
            raise RuntimeError(error_msg)
        
        # Mapping from Sequential indices to ResNet module names
        # trainable_backbone is created as: nn.Sequential(*list(model.children())[:-2])
        # This gives: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4]
        sequential_to_resnet = {
            '0': 'conv1',
            '1': 'bn1',
            '2': 'relu',
            '3': 'maxpool',
            '4': 'layer1',
            '5': 'layer2',
            '6': 'layer3',
            '7': 'layer4',
        }
        
        # Create a new state dict with mapped keys
        new_state_dict = {}
        skipped_keys = []
        
        for key, value in state_dict.items():
            # Filter out da3-related keys, dinov3-related keys, adapter keys, pre_norm, and gate
            if 'da3.' in key or 'adapter.' in key or 'dinov3.' in key or 'pre_norm.' in key or 'gate' in key:
                skipped_keys.append(key)
                continue
            
            # Map trainable_backbone.* keys to backbone.*
            # This handles both resnet_* and other backbone structures
            if key.startswith('trainable_backbone.'):
                # Extract the rest after 'trainable_backbone.'
                suffix = key.replace('trainable_backbone.', '')
                parts = suffix.split('.', 1)  # Split into [index, rest]
                
                if len(parts) == 2 and parts[0] in sequential_to_resnet:
                    # Map Sequential index to ResNet module name
                    resnet_module = sequential_to_resnet[parts[0]]
                    new_key = f'backbone.{resnet_module}.{parts[1]}'
                    new_state_dict[new_key] = value
                else:
                    # Fallback: direct replacement (for non-Sequential structures)
                    new_key = 'backbone.' + suffix
                    new_state_dict[new_key] = value
            # Map resnet_* keys to backbone.* (for SpatialFusedResNetDA3)
            elif key.startswith('resnet_'):
                new_key = 'backbone.' + key.replace('resnet_', '')
                new_state_dict[new_key] = value
            # Keep backbone.* keys as-is (for ConvNeXt V2 models)
            elif key.startswith('backbone.'):
                new_state_dict[key] = value
            # Map final_layers from Linear to Conv2d shape
            # Skip final_layers if dimensions don't match (allow loading only backbone)
            elif key.startswith('final_layers.'):
                # Extract layer index and parameter name
                parts = key.split('.')
                if len(parts) >= 3:
                    layer_idx = int(parts[1])
                    param_name = '.'.join(parts[2:])
                    
                    # Check if final_layers dimensions match
                    if param_name == 'weight':
                        # Get expected shape from current model
                        if hasattr(self, 'final_layers') and layer_idx < len(self.final_layers):
                            expected_shape = self.final_layers[layer_idx].weight.shape
                            # Convert checkpoint shape if needed
                            if len(value.shape) == 2:  # Linear: [out, in]
                                checkpoint_shape = (value.shape[0], value.shape[1], 1, 1)
                            else:  # Conv2d: [out, in, 1, 1]
                                checkpoint_shape = value.shape
                            
                            # Check if output dimensions match
                            if checkpoint_shape[0] != expected_shape[0]:
                                # Skip this key - dimensions don't match
                                skipped_keys.append(f"{key} (size mismatch: {checkpoint_shape[0]} vs {expected_shape[0]})")
                                continue
                        
                        # Convert Linear weight [out, in] to Conv2d weight [out, in, 1, 1]
                        if len(value.shape) == 2:  # Linear: [out, in]
                            value = value.unsqueeze(-1).unsqueeze(-1)  # [out, in, 1, 1]
                        new_key = f'final_layers.{layer_idx}.{param_name}'
                        new_state_dict[new_key] = value
                    else:
                        # For bias, check if corresponding weight was skipped
                        if hasattr(self, 'final_layers') and layer_idx < len(self.final_layers):
                            # Check if we have the weight in new_state_dict
                            if f'final_layers.{layer_idx}.weight' not in new_state_dict:
                                # Skip bias if weight was skipped
                                skipped_keys.append(f"{key} (weight was skipped)")
                                continue
                        new_key = f'final_layers.{layer_idx}.{param_name}'
                        new_state_dict[new_key] = value
                else:
                    # Keep as-is if format doesn't match
                    new_state_dict[key] = value
            else:
                # Keep other keys as-is (e.g., gap, avgpool, etc.)
                new_state_dict[key] = value
        
        if skipped_keys:
            print(f"⚠ Skipped {len(skipped_keys)} keys (da3/dinov3/adapter related): {skipped_keys[:5]}..." if len(skipped_keys) > 5 else f"⚠ Skipped keys: {skipped_keys}")
        
        # Load with strict=False to allow missing keys
        missing_keys = []
        unexpected_keys = []
        try:
            result = super().load_state_dict(new_state_dict, strict=False)
            if isinstance(result, tuple):
                missing_keys, unexpected_keys = result
            elif result is not None:
                # Some PyTorch versions return a NamedTuple
                if hasattr(result, 'missing_keys'):
                    missing_keys = result.missing_keys
                if hasattr(result, 'unexpected_keys'):
                    unexpected_keys = result.unexpected_keys
            
            if missing_keys:
                print(f"⚠ Missing keys (will use random init): {len(missing_keys)} keys")
                if len(missing_keys) <= 10:
                    print(f"   Missing: {missing_keys}")
            if unexpected_keys:
                print(f"⚠ Unexpected keys (ignored): {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 10:
                    print(f"   Unexpected: {unexpected_keys}")
        except RuntimeError as e:
            # If there's a size mismatch error, try loading only backbone (skip final_layers)
            if 'size mismatch' in str(e).lower() and 'final_layers' in str(e).lower():
                print("⚠ Size mismatch detected in final_layers. Attempting to load only backbone weights...")
                # Remove all final_layers keys and retry
                backbone_only_dict = {k: v for k, v in new_state_dict.items() if not k.startswith('final_layers.')}
                try:
                    result = super().load_state_dict(backbone_only_dict, strict=False)
                    if isinstance(result, tuple):
                        missing_keys, unexpected_keys = result
                    elif result is not None:
                        if hasattr(result, 'missing_keys'):
                            missing_keys = result.missing_keys
                        if hasattr(result, 'unexpected_keys'):
                            unexpected_keys = result.unexpected_keys
                    
                    print("✅ Successfully loaded backbone weights. Final layers will use random initialization.")
                    if missing_keys:
                        print(f"⚠ Missing keys (will use random init): {len(missing_keys)} keys")
                        if len(missing_keys) <= 10:
                            print(f"   Missing: {missing_keys}")
                    if unexpected_keys:
                        print(f"⚠ Unexpected keys (ignored): {len(unexpected_keys)} keys")
                        if len(unexpected_keys) <= 10:
                            print(f"   Unexpected: {unexpected_keys}")
                    return missing_keys, unexpected_keys
                except RuntimeError as e2:
                    # If still fails, provide helpful error message
                    error_msg = f"""
❌ SIZE MISMATCH ERROR ❌

{str(e2)}

This usually means:
  1. The checkpoint was trained with a different --net_recon than what you're using now
  2. The checkpoint architecture doesn't match the current model architecture

Please check:
  - Your --net_recon parameter matches the training configuration
  - Your --name parameter matches the checkpoint directory
  - The checkpoint was trained with the same architecture

Common issues:
  - Using resnet50 when checkpoint expects spatial_convnextv2_base_da3 (1024 vs 2048 dim)
  - Using spatial_convnextv2_base_da3 when checkpoint expects resnet50 (2048 vs 1024 dim)
"""
                    raise RuntimeError(error_msg) from e2
            elif 'size mismatch' in str(e).lower():
                error_msg = f"""
❌ SIZE MISMATCH ERROR ❌

{str(e)}

This usually means:
  1. The checkpoint was trained with a different --net_recon than what you're using now
  2. The checkpoint architecture doesn't match the current model architecture

Please check:
  - Your --net_recon parameter matches the training configuration
  - Your --name parameter matches the checkpoint directory
  - The checkpoint was trained with the same architecture

Common issues:
  - Using resnet50 when checkpoint expects spatial_convnextv2_base_da3 (1024 vs 2048 dim)
  - Using spatial_convnextv2_base_da3 when checkpoint expects resnet50 (2048 vs 1024 dim)
"""
                raise RuntimeError(error_msg) from e
            else:
                raise
        
        return missing_keys, unexpected_keys

    def forward(self, x):
        x = self.backbone(x)
        if not self.use_last_fc:
            output = []
            for layer in self.final_layers:
                output.append(layer(x))
            x = torch.flatten(torch.cat(output, dim=1), 1)
        return x


class SpatialFusedResNetDINOv3(nn.Module):
    """
    Spatial Fusion Backbone (Optimized Bottleneck Version)
    
    Supports ResNet-50 backbone:
    - 'resnet50': ResNet-50 (2048 dim, baseline)

    Architecture:
    1. Stream A (Trainable): ResNet-50 (ImageNet-pretrained), extracted up to last conv layer -> (B, 2048, 7, 7).
    2. Stream B (Frozen): DINOv3 (ConvNeXt-Small), extracted last_hidden_state -> (B, 768, 7, 7).
    3. Bridge: Bottleneck Adapter (768->512->2048) with Zero-Initialization on the final layer.
    4. Fusion: Spatial element-wise addition (Backbone + Adapter(DINO)).
    """

    def __init__(
        self,
        num_outputs: int = 257,
        dinov3_path: str = DEFAULT_DINOV3_PATH,
        backbone: str = 'resnet50',
        init_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.backbone_name = backbone
        self.init_path = init_path

        # -------------------------
        # Stream A: Trainable Backbone
        # -------------------------
        backbone_dim, backbone_model = self._create_backbone(backbone, init_path)
        self.backbone_dim = backbone_dim
        self.trainable_backbone = backbone_model

        # -------------------------
        # Stream B: DINOv3 (Frozen)
        # -------------------------
        print(f"Loading DINOv3 from local path: {dinov3_path}")
        # Force local loading only, prevent network download
        if not os.path.exists(dinov3_path):
            raise FileNotFoundError(f"DINOv3 model path does not exist: {dinov3_path}. Please ensure the model is downloaded locally.")
        self.dinov3 = AutoModel.from_pretrained(dinov3_path, trust_remote_code=True, local_files_only=True)

        # Freeze DINOv3 completely
        self.dinov3.eval()
        self.dinov3.requires_grad_(False)  # Efficiently freeze all parameters

        # -------------------------
        # Adapter: Bottleneck Structure (Squeeze-Excite style)
        # -------------------------
        # 768 (DINO) -> 512 (Bottleneck) -> ReLU -> backbone_dim (Backbone Space)
        self.adapter = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, backbone_dim, kernel_size=1, bias=True),
        )

        # *** Zero Initialization (ControlNet Trick) ***
        # Only initialize the FINAL layer of the adapter to zero.
        # This ensures: Fused_Feat = Backbone_Feat + 0 (at step 0)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

        # -------------------------
        # Regression head (split into multiple coefficient branches, zero-initialized)
        # -------------------------
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # Split into multiple coefficient branches to match ReconNetWrapper
        self.final_layers = nn.ModuleList([
            nn.Linear(backbone_dim, 532, bias=True),  # id layer
            nn.Linear(backbone_dim, 45, bias=True),   # exp layer
            nn.Linear(backbone_dim, 439, bias=True), # tex layer
            nn.Linear(backbone_dim, 3, bias=True),   # angle layer
            nn.Linear(backbone_dim, 27, bias=True),  # gamma layer
            nn.Linear(backbone_dim, 2, bias=True),  # tx, ty
            nn.Linear(backbone_dim, 1, bias=True)    # tz
        ])
        # Zero-initialize all branches so initial predictions are centered neutral faces
        for m in self.final_layers:
            nn.init.constant_(m.weight, 0.)
            nn.init.constant_(m.bias, 0.)

    def _create_backbone(self, backbone: str, init_path: Optional[str] = None):
        """
        Create trainable backbone and return its feature dimension.
        
        Args:
            backbone: Backbone name ('resnet50')
            init_path: Optional path to pretrained backbone weights. If provided, loads from this path
                       instead of torchvision's ImageNet pretrained weights.
        
        Returns: (feature_dim, backbone_model)
        """
        # Load base model structure (with or without torchvision pretrained weights)
        use_torchvision_pretrained = (init_path is None or not os.path.isfile(init_path))
        
        if backbone == 'resnet50':
            print("Loading ResNet-50 (Trainable)...")
            model = tv_models.resnet50(pretrained=use_torchvision_pretrained)
            backbone_model = nn.Sequential(*list(model.children())[:-2])
            
            # If init_path is provided and file exists, load custom pretrained weights
            if init_path and os.path.isfile(init_path):
                print(f"Loading ResNet-50 backbone weights from: {init_path}")
                state_dict = filter_state_dict(torch.load(init_path, map_location='cpu'))
                # Try to load the filtered state_dict
                # Note: If the state_dict keys don't match exactly (e.g., it's from a full model),
                # we'll try to match by stripping prefixes
                try:
                    backbone_model.load_state_dict(state_dict, strict=False)
                    print("✓ Successfully loaded custom pretrained weights for ResNet-50 backbone")
                except Exception as e:
                    print(f"⚠ Warning: Could not load weights from {init_path}: {e}")
                    print("   Falling back to torchvision ImageNet pretrained weights.")
            
            return 2048, backbone_model
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Only 'resnet50' is supported in this class.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, 224, 224), ImageNet normalized.
        """
        b = x.size(0)

        # 1. Backbone Forward (Trainable)
        # Output: (B, backbone_dim, 7, 7)
        x_backbone = self.trainable_backbone(x)

        # 2. DINOv3 Forward (Frozen)
        # Force eval mode to prevent BatchNorm stats update during training
        self.dinov3.eval()

        with torch.no_grad():
            # dinov3 is a submodule, so it's already on the correct device
            dinov3_out = self.dinov3(x)

            # Extract feature map
            if hasattr(dinov3_out, 'last_hidden_state'):
                feat_map = dinov3_out.last_hidden_state
            else:
                feat_map = dinov3_out[0]

            # Shape Robustness: Ensure (B, C, H, W)
            # Handle (B, N, C) -> (B, C, H, W) conversion
            feat_map = reshape_dinov3_features(feat_map)

        # 3. Spatial Alignment (Safety Check)
        # If DINOv3 output size doesn't match backbone (e.g., if input size changes), interpolate.
        if feat_map.shape[-2:] != x_backbone.shape[-2:]:
            feat_map = F.interpolate(feat_map, size=x_backbone.shape[-2:], mode='bilinear', align_corners=False)

        # 4. Injection (Bottleneck Adapter)
        # Map 768 -> backbone_dim with non-linearity
        adapter_out = self.adapter(feat_map)

        # Spatial Fusion: Backbone features + DINOv3 geometric correction
        fused_feat = x_backbone + adapter_out

        # 5. Global Pooling & Output
        pooled = self.gap(fused_feat)      # (B, backbone_dim, 1, 1)
        pooled = pooled.view(b, -1)        # (B, backbone_dim)
        # Use multiple branches to predict each coefficient group separately
        output = []
        for layer in self.final_layers:
            output.append(layer(pooled))
        out = torch.cat(output, dim=1)      # (B, 1049)
        return out

    def load_state_dict(self, state_dict, strict=True):
        """
        Smart loading that handles:
        1. Key name mapping: trainable_backbone.* -> backbone.* (for testing with resnet50)
           - trainable_backbone.0.* -> backbone.conv1.*
           - trainable_backbone.1.* -> backbone.bn1.*
           - trainable_backbone.2.* -> backbone.relu.*
           - trainable_backbone.3.* -> backbone.maxpool.*
           - trainable_backbone.4.* -> backbone.layer1.*
           - trainable_backbone.5.* -> backbone.layer2.*
           - trainable_backbone.6.* -> backbone.layer3.*
           - trainable_backbone.7.* -> backbone.layer4.*
        2. Filtering: Remove dinov3.* and adapter.* keys
        3. Shape conversion: Conv2d [out, in, 1, 1] -> Linear [out, in] for final_layers
           (This model uses Linear layers, but checkpoint might have Conv2d shape from ReconNetWrapper)
        """
        # Mapping from Sequential indices to ResNet module names
        # trainable_backbone is created as: nn.Sequential(*list(model.children())[:-2])
        # This gives: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4]
        sequential_to_resnet = {
            '0': 'conv1',
            '1': 'bn1',
            '2': 'relu',
            '3': 'maxpool',
            '4': 'layer1',
            '5': 'layer2',
            '6': 'layer3',
            '7': 'layer4',
        }
        
        # Create a new state dict with mapped keys
        new_state_dict = {}
        skipped_keys = []
        
        for key, value in state_dict.items():
            # Filter out dinov3-related keys and adapter keys
            if 'dinov3.' in key or 'adapter.' in key or 'pre_norm.' in key or 'gate' in key:
                skipped_keys.append(key)
                continue
            
            # Map trainable_backbone.* keys to backbone.* (for testing with resnet50)
            if key.startswith('trainable_backbone.'):
                # Extract the rest after 'trainable_backbone.'
                suffix = key.replace('trainable_backbone.', '')
                parts = suffix.split('.', 1)  # Split into [index, rest]
                
                if len(parts) == 2 and parts[0] in sequential_to_resnet:
                    # Map Sequential index to ResNet module name
                    resnet_module = sequential_to_resnet[parts[0]]
                    new_key = f'backbone.{resnet_module}.{parts[1]}'
                    new_state_dict[new_key] = value
                else:
                    # Fallback: direct replacement (shouldn't happen, but just in case)
                    new_key = 'backbone.' + suffix
                    new_state_dict[new_key] = value
            # Map final_layers: Handle shape conversion between Conv2d and Linear
            elif key.startswith('final_layers.'):
                # Extract layer index and parameter name
                parts = key.split('.')
                if len(parts) >= 3:
                    layer_idx = int(parts[1])
                    param_name = '.'.join(parts[2:])
                    
                    if param_name == 'weight':
                        # This model uses Linear layers, but checkpoint might have Conv2d shape
                        # Convert Conv2d weight [out, in, 1, 1] to Linear weight [out, in]
                        if len(value.shape) == 4:  # Conv2d: [out, in, 1, 1]
                            # Squeeze the spatial dimensions
                            if value.shape[2] == 1 and value.shape[3] == 1:
                                value = value.squeeze(-1).squeeze(-1)  # [out, in]
                            else:
                                raise ValueError(f"Cannot convert Conv2d weight with shape {value.shape} to Linear. Expected [out, in, 1, 1]")
                        # If already Linear shape [out, in], keep as-is
                        new_key = f'final_layers.{layer_idx}.{param_name}'
                        new_state_dict[new_key] = value
                    else:
                        # bias and other params remain the same
                        new_key = f'final_layers.{layer_idx}.{param_name}'
                        new_state_dict[new_key] = value
                else:
                    # Keep as-is if format doesn't match
                    new_state_dict[key] = value
            else:
                # Keep other keys as-is (e.g., gap, etc.)
                new_state_dict[key] = value
        
        if skipped_keys:
            print(f"⚠ Skipped {len(skipped_keys)} keys (dinov3/adapter related): {skipped_keys[:5]}..." if len(skipped_keys) > 5 else f"⚠ Skipped keys: {skipped_keys}")
        
        # Load with strict=False to allow missing keys
        missing_keys, unexpected_keys = super().load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"⚠ Missing keys (will use random init): {len(missing_keys)} keys")
            if len(missing_keys) <= 10:
                print(f"   Missing: {missing_keys}")
        if unexpected_keys:
            print(f"⚠ Unexpected keys (ignored): {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 10:
                print(f"   Unexpected: {unexpected_keys}")
        
        return missing_keys, unexpected_keys


class SpatialFusedResNetDA3(nn.Module):
    """
    Spatial Feature Fusion backbone combining:
    - Stream A (trainable): torchvision ResNet50 (ImageNet-pretrained), feature map from layer4 (B, 2048, 7, 7).
    - Stream B (frozen): Depth Anything V3 Small (DA3-Small), spatial feature map (B, 384, H', W').

    DA3 feature maps are projected to 2048 channels via a 1x1 Conv2d adapter (with BN+ReLU),
    then spatially resized to match the ResNet feature map and fused by element-wise addition:

        fused_feat = resnet_feat + adapter(da3_feat_resized)  # (B, 2048, 7, 7)
        gap -> (B, 2048) -> fc -> (B, num_outputs)
    """

    def __init__(
        self,
        num_outputs: int = 1049,
        init_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_outputs = num_outputs

        # -------------------------
        # Stream A: ResNet50 (Trainable)
        # -------------------------
        # Use torchvision ResNet50 as the backbone and keep feature maps up to layer4.
        # Prefer local weights to avoid network download failures
        use_torchvision_pretrained = (init_path is None or not os.path.isfile(init_path))
        
        print("Loading ResNet-50 (Trainable)...")
        base_resnet = tv_models.resnet50(pretrained=use_torchvision_pretrained)
        
        # Load local weights when provided
        if init_path and os.path.isfile(init_path):
            print(f"Loading ResNet-50 backbone weights from: {init_path}")
            state_dict = filter_state_dict(torch.load(init_path, map_location='cpu'))
            try:
                base_resnet.load_state_dict(state_dict, strict=False)
                print("✓ Successfully loaded custom pretrained weights for ResNet-50 backbone")
            except Exception as e:
                print(f"⚠ Warning: Could not load weights from {init_path}: {e}")
                print("   Falling back to torchvision ImageNet pretrained weights.")
        
        self.resnet_conv1 = base_resnet.conv1
        self.resnet_bn1 = base_resnet.bn1
        self.resnet_relu = base_resnet.relu
        self.resnet_maxpool = base_resnet.maxpool
        self.resnet_layer1 = base_resnet.layer1
        self.resnet_layer2 = base_resnet.layer2
        self.resnet_layer3 = base_resnet.layer3
        self.resnet_layer4 = base_resnet.layer4

        # GAP and regression head (split into multiple coefficient branches, zero-initialized)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Split into multiple coefficient branches to match ReconNetWrapper
        self.final_layers = nn.ModuleList([
            nn.Linear(2048, 532, bias=True),  # id layer
            nn.Linear(2048, 45, bias=True),   # exp layer
            nn.Linear(2048, 439, bias=True), # tex layer
            nn.Linear(2048, 3, bias=True),   # angle layer
            nn.Linear(2048, 27, bias=True),  # gamma layer
            nn.Linear(2048, 2, bias=True),  # tx, ty
            nn.Linear(2048, 1, bias=True)    # tz
        ])
        # Zero-initialize all branches so initial predictions are centered neutral faces
        for m in self.final_layers:
            nn.init.constant_(m.weight, 0.)
            nn.init.constant_(m.bias, 0.)

        # -------------------------
        # Stream B: Depth Anything 3 Small (Frozen)
        # -------------------------
        # Load from checkpoints/DA3 to stay consistent with FusedResNetDA3.
        da3_ckpt_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "checkpoints",
            "DA3",
        )
        # Force local loading only, prevent network download
        if not os.path.exists(da3_ckpt_dir):
            raise FileNotFoundError(f"DA3 model directory does not exist: {da3_ckpt_dir}. Please ensure the model is downloaded locally.")
        # Lazy import to avoid libGL.so.1 dependency when not needed
        from depth_anything_3.api import DepthAnything3
        self.da3 = DepthAnything3.from_pretrained(da3_ckpt_dir)
        self.da3.eval()
        for p in self.da3.parameters():
            p.requires_grad = False

        # -------------------------
        # Bridge / Adapter: Bottleneck with Zero Initialization (matching DiNOv3 version)
        # -------------------------
        # Map DA3 384 channels to ResNet 2048 channels using a bottleneck for stability
        # Follow the DINOv3 design: bottleneck + zero initialization for training stability
        # Use a 384->256->2048 structure with fewer parameters (614K), allowing more aggressive extraction of key geometry features
        self.adapter = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=1, bias=False),      # 384 -> 256 (bottleneck, fewer parameters)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2048, kernel_size=1, bias=True),      # 256 -> 2048
        )
        
        # *** Zero Initialization (ControlNet Trick) ***
        # Ensure adapter output is 0 at the start of training so backbone features are unaffected
        # This is key to the stability of the DINOv3 version.
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image with shape (B, 3, 224, 224), already normalized with ImageNet mean and std.

        Returns:
            Coefficient tensor with shape (B, num_outputs), default 1049 and configurable to 257 or other sizes as needed.
        """
        b = x.size(0)

        # -------------------------
        # Stream A: ResNet50 features (trainable)
        # -------------------------
        x_res = self.resnet_conv1(x)
        x_res = self.resnet_bn1(x_res)
        x_res = self.resnet_relu(x_res)
        x_res = self.resnet_maxpool(x_res)

        x_res = self.resnet_layer1(x_res)
        x_res = self.resnet_layer2(x_res)
        x_res = self.resnet_layer3(x_res)
        x_res = self.resnet_layer4(x_res)  # (B, 2048, 7, 7)

        # -------------------------
        # Stream B: DA3-Small spatial features (frozen)
        # -------------------------
        self.da3.to(x.device)
        self.da3.eval()
        with torch.no_grad():
            # DA3 interface expects input shape (B, N, 3, H, W), where N is the number of views; set to 1 here.
            x_da3 = x.unsqueeze(1)  # (B, 1, 3, H, W)
            raw_out = self.da3(
                x_da3,
                export_feat_layers=[11],  # match the layer used in FusedResNetDA3
                infer_gs=False,
            )
            aux = raw_out.aux
            feat_5d = aux["feat_layer_11"]  # (B, N, H', W', C=384) or (B, N, C, H', W')

            # Average over the view dimension and remove it
            feat_sc = feat_5d.mean(dim=1)  # (B, H', W', C) or (B, C, H', W')
            
            # Robustness check: handle both channel-last and channel-first formats
            # DepthAnything3 output dimensions may vary by version and must be detected dynamically
            if feat_sc.shape[-1] == 384:  # Channel last: (B, H', W', 384)
                da3_feat = feat_sc.permute(0, 3, 1, 2).contiguous()  # -> (B, 384, H', W')
            else:  # Channel first: (B, 384, H', W')
                da3_feat = feat_sc

        # -------------------------
        # Bridge: channel mapping + spatial alignment
        # -------------------------
        da3_proj = self.adapter(da3_feat)  # (B, 2048, H', W')
        # Align the spatial resolution of projected DA3 features to ResNet features (7x7)
        # Use bilinear interpolation for alignment, matching the DINOv3 version
        if da3_proj.shape[-2:] != x_res.shape[-2:]:
            da3_proj = F.interpolate(da3_proj, size=x_res.shape[-2:], mode='bilinear', align_corners=False)

        # -------------------------
        # Spatial fusion: element-wise addition
        # -------------------------
        fused_feat = x_res + da3_proj  # (B, 2048, 7, 7)

        # -------------------------
        # GAP + multi-branch regression head
        # -------------------------
        pooled = self.avgpool(fused_feat)  # (B, 2048, 1, 1)
        pooled = pooled.view(b, -1)        # (B, 2048)
        # Use multiple branches to predict each coefficient group separately
        output = []
        for layer in self.final_layers:
            output.append(layer(pooled))
        out = torch.cat(output, dim=1)      # (B, 1049)
        return out

    def load_state_dict(self, state_dict, strict=True):
        """
        Smart loading that handles:
        1. Key name mapping: resnet_* -> backbone.* (for testing with resnet50)
        2. Filtering: Remove da3.* and adapter.* keys
        3. Shape conversion: Linear [out, in] -> Conv2d [out, in, 1, 1] for final_layers
        """
        # Create a new state dict with mapped keys
        new_state_dict = {}
        skipped_keys = []
        
        for key, value in state_dict.items():
            # Filter out da3-related keys and adapter keys
            if 'da3.' in key or 'adapter.' in key:
                skipped_keys.append(key)
                continue
            
            # Map resnet_* keys to backbone.* (for testing with resnet50)
            if key.startswith('resnet_'):
                new_key = 'backbone.' + key.replace('resnet_', '')
                new_state_dict[new_key] = value
            # Map final_layers from Linear to Conv2d shape
            elif key.startswith('final_layers.'):
                # Extract layer index and parameter name
                parts = key.split('.')
                if len(parts) >= 3:
                    layer_idx = int(parts[1])
                    param_name = '.'.join(parts[2:])
                    
                    if param_name == 'weight':
                        # Convert Linear weight [out, in] to Conv2d weight [out, in, 1, 1]
                        if len(value.shape) == 2:  # Linear: [out, in]
                            value = value.unsqueeze(-1).unsqueeze(-1)  # [out, in, 1, 1]
                        new_key = f'final_layers.{layer_idx}.{param_name}'
                        new_state_dict[new_key] = value
                    else:
                        # bias and other params remain the same
                        new_key = f'final_layers.{layer_idx}.{param_name}'
                        new_state_dict[new_key] = value
                else:
                    # Keep as-is if format doesn't match
                    new_state_dict[key] = value
            else:
                # Keep other keys as-is (e.g., avgpool, etc.)
                new_state_dict[key] = value
        
        if skipped_keys:
            print(f"⚠ Skipped {len(skipped_keys)} keys (da3/adapter related): {skipped_keys[:5]}..." if len(skipped_keys) > 5 else f"⚠ Skipped keys: {skipped_keys}")
        
        # Load with strict=False to allow missing keys
        missing_keys, unexpected_keys = super().load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"⚠ Missing keys (will use random init): {len(missing_keys)} keys")
            if len(missing_keys) <= 10:
                print(f"   Missing: {missing_keys}")
        if unexpected_keys:
            print(f"⚠ Unexpected keys (ignored): {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 10:
                print(f"   Unexpected: {unexpected_keys}")
        
        return missing_keys, unexpected_keys


class SpatialFusedConvNeXtV2BaseDINOv3(nn.Module):
    """
    Robust fixed version: adds input normalization (Pre-Norm) and gated fusion (Gated Fusion)
    to resolve epoch 2-10 training collapse / face distortion.
    
    Structure:
    - Trainable: ConvNeXt V2 Base (timm)
    - Frozen: DINOv3 (Local)
    - Pre-Norm: InstanceNorm2d normalizes DINOv3 features to prevent oversized values
    - Adapter: Bottleneck (768->512->1024) with improved initialization
    - Gate: Learnable gate (initialized to 0 for progressive fusion)
    - Fusion: Main + Gate * Adapter(PreNorm(Aux))
    
    Similar to spatial_resnet_dinov3 but uses modern ConvNeXt V2 Base architecture.
    """

    def __init__(
        self,
        num_outputs: int = 1049,
        trainable_model_name: str = 'convnextv2_base',
        dinov3_path: str = DEFAULT_DINOV3_PATH,
        init_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        
        # 1. Trainable Backbone (ConvNeXt V2 Base via timm)
        print(f"Loading Trainable Backbone: {trainable_model_name}...")
        
        # Check if local weights are provided (file or directory)
        has_local_weights = init_path and (os.path.isfile(init_path) or os.path.isdir(init_path))
        
        # Create model without pretrained weights if local path is provided
        self.backbone = timm.create_model(
            trainable_model_name,
            pretrained=(not has_local_weights),  # Don't download if local weights exist
            features_only=True,
            out_indices=(3,)  # Get the final stage
        )
        
        # Load custom pretrained weights if provided
        if has_local_weights:
            state_dict = load_convnextv2_weights(init_path)
            if state_dict is not None:
                try:
                    self.backbone.load_state_dict(state_dict, strict=False)
                    print("✓ Successfully loaded custom pretrained weights for ConvNeXt V2 Base backbone")
                except Exception as e:
                    print(f"⚠ Warning: Could not load weights from {init_path}: {e}")
                    print("   Falling back to random initialization.")
            else:
                print(f"⚠ Warning: Could not load weights from {init_path}")
                print("   Falling back to random initialization.")
        
        # Auto-detect channel dimension (1024 for Base)
        self.backbone_dim = self.backbone.feature_info.channels()[-1]
        print(f"Trainable Backbone Channels: {self.backbone_dim}")

        # 2. Frozen Backbone (DINOv3)
        print(f"Loading DINOv3 from local path: {dinov3_path}")
        # Force local loading only, prevent network download
        if not os.path.exists(dinov3_path):
            raise FileNotFoundError(f"DINOv3 model path does not exist: {dinov3_path}. Please ensure the model is downloaded locally.")
        self.dinov3 = AutoModel.from_pretrained(dinov3_path, trust_remote_code=True, local_files_only=True)
        self.dinov3.eval()
        self.dinov3.requires_grad_(False)
        self.dinov3_dim = 768  # Fixed for DINOv3 ConvNeXt-Small

        # 3. Safety valve A: Pre-normalization (Pre-Norm)
        # DINOv3 feature variance may be large; use InstanceNorm first to bring it back to a standard distribution
        # This prevents oversized adapter inputs from causing gradient explosion
        self.pre_norm = nn.InstanceNorm2d(self.dinov3_dim, affine=False)

        # 4. Adapter (Bottleneck)
        # Maps DINOv3 (768) -> Trainable Backbone (1024)
        print(f"Building Adapter: {self.dinov3_dim} -> {self.backbone_dim}")
        
        self.adapter = nn.Sequential(
            nn.Conv2d(self.dinov3_dim, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.backbone_dim, kernel_size=1, bias=True)
        )
        
        # Weight initialization: Kaiming for the first layer, zero initialization for the last layer
        nn.init.kaiming_normal_(self.adapter[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

        # 5. Safety valve B: learnable gate (Learnable Gate)
        # Initialize to 0, which is stricter than zero-initializing convolution layers.
        # Fusion formula: Main + Gate * Adapter(Aux)
        self.gate = nn.Parameter(torch.zeros(1))

        # 6. Head (split into multiple coefficient branches, zero-initialized)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # Split into multiple coefficient branches to match ReconNetWrapper
        self.final_layers = nn.ModuleList([
            nn.Linear(self.backbone_dim, 532, bias=True),  # id layer
            nn.Linear(self.backbone_dim, 45, bias=True),   # exp layer
            nn.Linear(self.backbone_dim, 439, bias=True), # tex layer
            nn.Linear(self.backbone_dim, 3, bias=True),   # angle layer
            nn.Linear(self.backbone_dim, 27, bias=True),  # gamma layer
            nn.Linear(self.backbone_dim, 2, bias=True),  # tx, ty
            nn.Linear(self.backbone_dim, 1, bias=True)    # tz
        ])
        # Zero-initialize all branches so initial predictions are centered neutral faces
        for m in self.final_layers:
            nn.init.constant_(m.weight, 0.)
            nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        b = x.size(0)

        # Trainable Forward (timm returns a list of features)
        trainable_feat = self.backbone(x)[0]  # (B, C_backbone, 7, 7)

        # Frozen Forward
        self.dinov3.to(x.device)  # Ensure the model is on the correct device
        self.dinov3.eval()
        with torch.no_grad():
            dinov3_out = self.dinov3(x)
            if hasattr(dinov3_out, 'last_hidden_state'):
                frozen_feat = dinov3_out.last_hidden_state
            else:
                frozen_feat = dinov3_out[0]
                
            # Dimension fix if needed (B, N, C) -> (B, C, H, W)
            frozen_feat = reshape_dinov3_features(frozen_feat)

        # Alignment
        if frozen_feat.shape[-2:] != trainable_feat.shape[-2:]:
            frozen_feat = F.interpolate(frozen_feat, size=trainable_feat.shape[-2:], mode='bilinear', align_corners=False)

        # Normalization + injection
        # Normalize DINO features first to prevent oversized values
        normed_frozen = self.pre_norm(frozen_feat)
        
        # Pass through adapter
        adapter_out = self.adapter(normed_frozen)
        
        # Gated fusion (Gated Fusion)
        # Initially self.gate is 0, fully blocking the DINOv3 contribution
        fused_feat = trainable_feat + self.gate * adapter_out

        # Output
        pooled = self.gap(fused_feat).flatten(1)  # (B, backbone_dim)
        # Use multiple branches to predict each coefficient group separately
        output = []
        for layer in self.final_layers:
            output.append(layer(pooled))
        out = torch.cat(output, dim=1)      # (B, 1049)
        return out

    def load_state_dict(self, state_dict, strict=True):
        """
        Smart loading that handles:
        1. Filtering: Remove dinov3.*, adapter.*, pre_norm.*, and gate keys
        2. Shape conversion: Conv2d [out, in, 1, 1] -> Linear [out, in] for final_layers
        3. Keep backbone.* keys as-is (for testing with convnextv2_base)
        """
        # Create a new state dict with mapped keys
        new_state_dict = {}
        skipped_keys = []
        
        for key, value in state_dict.items():
            # Filter out dinov3-related keys, adapter keys, pre_norm, and gate
            if 'dinov3.' in key or 'adapter.' in key or 'pre_norm.' in key or 'gate' in key:
                skipped_keys.append(key)
                continue
            
            # Map final_layers: Convert Conv2d [out, in, 1, 1] to Linear [out, in] shape
            if key.startswith('final_layers.'):
                # Extract layer index and parameter name
                parts = key.split('.')
                if len(parts) >= 3:
                    layer_idx = int(parts[1])
                    param_name = '.'.join(parts[2:])
                    
                    if param_name == 'weight':
                        # Convert Conv2d weight [out, in, 1, 1] to Linear weight [out, in]
                        if len(value.shape) == 4:  # Conv2d: [out, in, 1, 1]
                            value = value.squeeze(-1).squeeze(-1)  # [out, in]
                        elif len(value.shape) == 2:  # Already Linear: [out, in]
                            value = value  # Keep as-is
                        new_key = f'final_layers.{layer_idx}.{param_name}'
                        new_state_dict[new_key] = value
                    else:
                        # bias and other params remain the same
                        new_key = f'final_layers.{layer_idx}.{param_name}'
                        new_state_dict[new_key] = value
                else:
                    # Keep as-is if format doesn't match
                    new_state_dict[key] = value
            else:
                # Keep other keys as-is (e.g., backbone.*, gap, etc.)
                new_state_dict[key] = value
        
        if skipped_keys:
            print(f"⚠ Skipped {len(skipped_keys)} keys (dinov3/adapter related): {skipped_keys[:5]}..." if len(skipped_keys) > 5 else f"⚠ Skipped keys: {skipped_keys}")
        
        # Load with strict=False to allow missing keys
        missing_keys, unexpected_keys = super().load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"⚠ Missing keys (will use random init): {len(missing_keys)} keys")
            if len(missing_keys) <= 10:
                print(f"   Missing: {missing_keys}")
        if unexpected_keys:
            print(f"⚠ Unexpected keys (ignored): {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 10:
                print(f"   Unexpected: {unexpected_keys}")
        
        return missing_keys, unexpected_keys


class SpatialFusedConvNeXtV2BaseDA3(nn.Module):
    """
    Spatial Feature Fusion backbone combining:
    - Stream A (trainable): ConvNeXt V2 Base (timm), feature map from final stage.
    - Stream B (frozen): Depth Anything V3 Small (DA3-Small), spatial feature map.
    
    Similar to spatial_resnet_da3 but uses modern ConvNeXt V2 Base architecture.
    """

    def __init__(
        self,
        num_outputs: int = 1049,
        trainable_model_name: str = 'convnextv2_base',
        init_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_outputs = num_outputs

        # 1. Trainable Backbone (ConvNeXt V2 Base via timm)
        print(f"Loading Trainable Backbone: {trainable_model_name}...")
        
        # Check if local weights are provided (file or directory)
        has_local_weights = init_path and (os.path.isfile(init_path) or os.path.isdir(init_path))
        
        # Create model without pretrained weights if local path is provided
        self.backbone = timm.create_model(
            trainable_model_name,
            pretrained=(not has_local_weights),  # Don't download if local weights exist
            features_only=True,
            out_indices=(3,)  # Get the final stage
        )
        
        # Load custom pretrained weights if provided
        if has_local_weights:
            state_dict = load_convnextv2_weights(init_path)
            if state_dict is not None:
                try:
                    self.backbone.load_state_dict(state_dict, strict=False)
                    print("✓ Successfully loaded custom pretrained weights for ConvNeXt V2 Base backbone")
                except Exception as e:
                    print(f"⚠ Warning: Could not load weights from {init_path}: {e}")
                    print("   Falling back to random initialization.")
            else:
                print(f"⚠ Warning: Could not load weights from {init_path}")
                print("   Falling back to random initialization.")
        
        # Auto-detect channel dimension (1024 for Base)
        self.backbone_dim = self.backbone.feature_info.channels()[-1]
        print(f"Trainable Backbone Channels: {self.backbone_dim}")

        # 2. Frozen Backbone (DA3-Small)
        da3_ckpt_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "checkpoints",
            "DA3",
        )
        print(f"Loading Depth Anything V3 from: {da3_ckpt_dir}")
        # Force local loading only, prevent network download
        if not os.path.exists(da3_ckpt_dir):
            raise FileNotFoundError(f"DA3 model directory does not exist: {da3_ckpt_dir}. Please ensure the model is downloaded locally.")
        # Lazy import to avoid libGL.so.1 dependency when not needed
        from depth_anything_3.api import DepthAnything3
        self.da3 = DepthAnything3.from_pretrained(da3_ckpt_dir)
        self.da3.eval()
        for p in self.da3.parameters():
            p.requires_grad = False
        self.da3_dim = 384  # Fixed for DA3-Small

        # 3. Adapter: Bottleneck + Zero Initialization (matching ResNet DA3 version)
        # Use a 384->256->1024 structure with fewer parameters (614K) to extract key geometry features more aggressively.
        print(f"Building Adapter: {self.da3_dim} -> {self.backbone_dim}")
        self.adapter = nn.Sequential(
            nn.Conv2d(self.da3_dim, 256, kernel_size=1, bias=False),      # 384 -> 256 (bottleneck)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.backbone_dim, kernel_size=1, bias=True),  # 256 -> 1024
        )
        
        # *** Zero Initialization (ControlNet Trick) ***
        # Ensure adapter output is 0 at the start of training so backbone features are unaffected
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

        # 4. Head (split into multiple coefficient branches, zero-initialized)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # Split into multiple coefficient branches to match ReconNetWrapper
        self.final_layers = nn.ModuleList([
            nn.Linear(self.backbone_dim, 532, bias=True),  # id layer
            nn.Linear(self.backbone_dim, 45, bias=True),   # exp layer
            nn.Linear(self.backbone_dim, 439, bias=True), # tex layer
            nn.Linear(self.backbone_dim, 3, bias=True),   # angle layer
            nn.Linear(self.backbone_dim, 27, bias=True),  # gamma layer
            nn.Linear(self.backbone_dim, 2, bias=True),  # tx, ty
            nn.Linear(self.backbone_dim, 1, bias=True)    # tz
        ])
        # Zero-initialize all branches so initial predictions are centered neutral faces
        for m in self.final_layers:
            nn.init.constant_(m.weight, 0.)
            nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        b = x.size(0)

        # Trainable Forward
        trainable_feat = self.backbone(x)[0]  # (B, C_backbone, 7, 7)

        # Frozen Forward (DA3)
        self.da3.to(x.device)
        self.da3.eval()
        with torch.no_grad():
            x_da3 = x.unsqueeze(1)  # (B, 1, 3, H, W)
            raw_out = self.da3(
                x_da3,
                export_feat_layers=[11],
                infer_gs=False,
            )
            aux = raw_out.aux
            feat_5d = aux["feat_layer_11"]  # (B, N, H', W', C=384) or (B, N, C, H', W')

            # Average over the view dimension and remove it
            feat_sc = feat_5d.mean(dim=1)  # (B, H', W', C) or (B, C, H', W')
            
            # Robustness check: handle both channel-last and channel-first formats
            # DepthAnything3 output dimensions may vary by version and must be detected dynamically
            if feat_sc.shape[-1] == 384:  # Channel last: (B, H', W', 384)
                da3_feat = feat_sc.permute(0, 3, 1, 2).contiguous()  # -> (B, 384, H', W')
            else:  # Channel first: (B, 384, H', W')
                da3_feat = feat_sc

        # Bridge: Channel mapping + Spatial alignment
        da3_proj = self.adapter(da3_feat)  # (B, backbone_dim, H', W')
        
        # Spatial alignment to match backbone feature map
        if da3_proj.shape[-2:] != trainable_feat.shape[-2:]:
            da3_proj = F.adaptive_avg_pool2d(da3_proj, output_size=trainable_feat.shape[-2:])

        # Spatial Fusion: Element-wise addition
        fused_feat = trainable_feat + da3_proj  # (B, backbone_dim, 7, 7)

        # Output
        pooled = self.gap(fused_feat)  # (B, backbone_dim, 1, 1)
        pooled = pooled.view(b, -1)    # (B, backbone_dim)
        # Use multiple branches to predict each coefficient group separately
        output = []
        for layer in self.final_layers:
            output.append(layer(pooled))
        out = torch.cat(output, dim=1)      # (B, 1049)
        return out

    def load_state_dict(self, state_dict, strict=True):
        """
        Smart loading that handles:
        1. Filtering: Remove da3.* and adapter.* keys
        2. Shape conversion: Conv2d [out, in, 1, 1] -> Linear [out, in] for final_layers
        3. Keep backbone.* keys as-is (for testing with convnextv2_base)
        """
        # Create a new state dict with mapped keys
        new_state_dict = {}
        skipped_keys = []
        
        for key, value in state_dict.items():
            # Filter out da3-related keys and adapter keys
            if 'da3.' in key or 'adapter.' in key:
                skipped_keys.append(key)
                continue
            
            # Map final_layers: Convert Conv2d [out, in, 1, 1] to Linear [out, in] shape
            if key.startswith('final_layers.'):
                # Extract layer index and parameter name
                parts = key.split('.')
                if len(parts) >= 3:
                    layer_idx = int(parts[1])
                    param_name = '.'.join(parts[2:])
                    
                    if param_name == 'weight':
                        # Convert Conv2d weight [out, in, 1, 1] to Linear weight [out, in]
                        if len(value.shape) == 4:  # Conv2d: [out, in, 1, 1]
                            value = value.squeeze(-1).squeeze(-1)  # [out, in]
                        elif len(value.shape) == 2:  # Already Linear: [out, in]
                            value = value  # Keep as-is
                        new_key = f'final_layers.{layer_idx}.{param_name}'
                        new_state_dict[new_key] = value
                    else:
                        # bias and other params remain the same
                        new_key = f'final_layers.{layer_idx}.{param_name}'
                        new_state_dict[new_key] = value
                else:
                    # Keep as-is if format doesn't match
                    new_state_dict[key] = value
            else:
                # Keep other keys as-is (e.g., backbone.*, gap, etc.)
                new_state_dict[key] = value
        
        if skipped_keys:
            print(f"⚠ Skipped {len(skipped_keys)} keys (da3/adapter related): {skipped_keys[:5]}..." if len(skipped_keys) > 5 else f"⚠ Skipped keys: {skipped_keys}")
        
        # Load with strict=False to allow missing keys
        missing_keys, unexpected_keys = super().load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"⚠ Missing keys (will use random init): {len(missing_keys)} keys")
            if len(missing_keys) <= 10:
                print(f"   Missing: {missing_keys}")
        if unexpected_keys:
            print(f"⚠ Unexpected keys (ignored): {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 10:
                print(f"   Unexpected: {unexpected_keys}")
        
        return missing_keys, unexpected_keys


class MultiScaleFusedResNetDA3(nn.Module):
    """
    Multi-Scale ResNet50 + DA3 Fusion with Deep MLP Head.
    
    Fusion Strategy:
    - Inject DA3 features at Layer 1, 2, 3, and 4 (resolutions 56x56 to 7x7).
    - Aggregate features from ALL layers for the final prediction.
    
    Head Strategy:
    - 3-Layer MLP instead of a single Linear layer.
    """

    def __init__(
        self,
        num_outputs: int = 1049,
        init_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_outputs = num_outputs

        # -------------------------
        # 1. ResNet50 Backbone (Split)
        # -------------------------
        print("Loading ResNet-50 (Trainable) for Multi-Scale Fusion...")
        use_torchvision = (init_path is None or not os.path.isfile(init_path))
        base_resnet = tv_models.resnet50(pretrained=use_torchvision)
        
        if not use_torchvision:
            try:
                state_dict = torch.load(init_path, map_location='cpu')
                if 'state_dict' in state_dict: 
                    state_dict = state_dict['state_dict']
                base_resnet.load_state_dict(state_dict, strict=False)
                print("✓ Loaded custom ResNet weights")
            except Exception as e:
                print(f"⚠ Warning: Fallback to ImageNet weights. {e}")

        # Split into stages
        self.stem = nn.Sequential(base_resnet.conv1, base_resnet.bn1, base_resnet.relu, base_resnet.maxpool)
        self.layer1 = base_resnet.layer1 # 256 ch, 56x56
        self.layer2 = base_resnet.layer2 # 512 ch, 28x28
        self.layer3 = base_resnet.layer3 # 1024 ch, 14x14
        self.layer4 = base_resnet.layer4 # 2048 ch, 7x7

        # -------------------------
        # 2. DA3 (Frozen) - Fix: add error handling
        # -------------------------
        da3_ckpt_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "checkpoints",
            "DA3",
        )
        # Force local loading only, prevent network download
        if not os.path.exists(da3_ckpt_dir):
            raise FileNotFoundError(
                f"DA3 model directory does not exist: {da3_ckpt_dir}. "
                "Please ensure the model is downloaded locally."
            )
        # Lazy import to avoid libGL.so.1 dependency when not needed
        from depth_anything_3.api import DepthAnything3
        self.da3 = DepthAnything3.from_pretrained(da3_ckpt_dir)
        self.da3.eval()
        for p in self.da3.parameters(): 
            p.requires_grad = False
        self.da3_dim = 384

        # -------------------------
        # 3. Multi-Scale Adapters (Bottleneck + Zero Init)
        # -------------------------
        # Create an adapter for EACH ResNet layer
        self.adapter1 = self._make_adapter(self.da3_dim, 256)  # For Layer 1
        self.adapter2 = self._make_adapter(self.da3_dim, 512)  # For Layer 2
        self.adapter3 = self._make_adapter(self.da3_dim, 1024) # For Layer 3
        self.adapter4 = self._make_adapter(self.da3_dim, 2048) # For Layer 4

        # -------------------------
        # 4. Deep MLP Head (3 Layers)
        # -------------------------
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Combined dimension: 256 + 512 + 1024 + 2048 = 3840
        self.fusion_dim = 256 + 512 + 1024 + 2048
        
        # MLP Hidden Dimension
        hidden_dim = 1024
        
        # Define MLP Head Generator
        def make_mlp_head(out_dim):
            return nn.Sequential(
                nn.Linear(self.fusion_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim)
            )

        # Multi-branch Output
        self.final_layers = nn.ModuleList([
            make_mlp_head(532), # id
            make_mlp_head(45),  # exp
            make_mlp_head(439), # tex
            make_mlp_head(3),   # angle
            make_mlp_head(27),  # gamma
            make_mlp_head(2),   # tx, ty
            make_mlp_head(1)    # tz
        ])
        
        # Zero-Init only the LAST Linear layer of each branch
        for branch in self.final_layers:
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def _make_adapter(self, in_dim, out_dim):
        """Creates a bottleneck adapter: In -> In/2 -> Out, with Zero-Init"""
        mid_dim = in_dim // 2
        adapter = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, out_dim, 1, bias=True)
        )
        nn.init.zeros_(adapter[-1].weight)
        nn.init.zeros_(adapter[-1].bias)
        return adapter

    def forward(self, x):
        b = x.size(0)

        # --- DA3 Feature Extraction - Fix: add device management and format compatibility handling ---
        self.da3.to(x.device)  # Fix: explicitly move device
        self.da3.eval()
        with torch.no_grad():
            x_da3 = x.unsqueeze(1)  # (B, 1, 3, H, W)
            raw_out = self.da3(
                x_da3,
                export_feat_layers=[11],
                infer_gs=False
            )
            aux = raw_out.aux
            feat_5d = aux["feat_layer_11"]  # (B, N, H', W', C=384) or (B, N, C, H', W')
            
            # Average over the view dimension and remove it
            feat_sc = feat_5d.mean(dim=1)  # (B, H', W', C) or (B, C, H', W')
            
            # Fix: robustness check for both channel-last and channel-first formats
            if feat_sc.shape[-1] == 384:  # Channel last: (B, H', W', 384)
                da3_feat = feat_sc.permute(0, 3, 1, 2).contiguous()  # -> (B, 384, H', W')
            else:  # Channel first: (B, 384, H', W')
                da3_feat = feat_sc

        # --- Multi-Scale Injection ---
        
        # Stem
        x = self.stem(x)
        
        # Layer 1 (Inject High Res)
        x = self.layer1(x) 
        da3_l1 = F.interpolate(da3_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter1(da3_l1)
        feat_l1 = self.gap(x).view(b, -1) # Save Global Feature

        # Layer 2
        x = self.layer2(x)
        da3_l2 = F.interpolate(da3_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter2(da3_l2)
        feat_l2 = self.gap(x).view(b, -1)

        # Layer 3
        x = self.layer3(x)
        da3_l3 = F.interpolate(da3_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter3(da3_l3)
        feat_l3 = self.gap(x).view(b, -1)

        # Layer 4 (Inject Global Shape)
        x = self.layer4(x)
        da3_l4 = F.interpolate(da3_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter4(da3_l4)
        feat_l4 = self.gap(x).view(b, -1)

        # --- Aggregation & Deep Head ---
        # Concatenate features from ALL scales (HyperColumn)
        combined = torch.cat([feat_l1, feat_l2, feat_l3, feat_l4], dim=1) # Dim: 3840
        
        output = [layer(combined) for layer in self.final_layers]
        return torch.cat(output, dim=1)  # (B, 1049)


class MultiScaleFusedConvNeXtV2DA3(nn.Module):
    """
    Multi-Scale ConvNeXt V2 + DA3 Fusion with Deep MLP Head.
    
    Fusion Strategy:
    - Inject DA3 features at Stage 0, 1, 2, and 3 (resolutions from 56x56 to 7x7).
    - Aggregate features from ALL stages for the final prediction.
    
    Head Strategy:
    - 3-Layer MLP instead of a single Linear layer.
    
    Structure:
    - Backbone: ConvNeXt V2 Base (Split into 4 stages).
    - Adapters: 4 Zero-Init Adapters injecting DA3 features at every stage.
    - Head: 3-Layer MLP aggregating features from all stages.
    """

    def __init__(
        self,
        num_outputs: int = 1049,
        trainable_model_name: str = 'convnextv2_base',
        init_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_outputs = num_outputs

        # -------------------------
        # 1. ConvNeXt Backbone (Split)
        # -------------------------
        print(f"Loading {trainable_model_name} (Trainable) for Multi-Scale Fusion...")
        
        # Check if local weights are provided
        has_local = init_path and (os.path.isfile(init_path) or os.path.isdir(init_path))
        
        # Load full model to extract stem and stages
        full_model = timm.create_model(trainable_model_name, pretrained=(not has_local))
        
        if has_local:
            # Use existing load_convnextv2_weights function for robust weight loading
            state_dict = load_convnextv2_weights(init_path)
            if state_dict is not None:
                try:
                    full_model.load_state_dict(state_dict, strict=False)
                    print("✓ Loaded custom ConvNeXt weights")
                except Exception as e:
                    print(f"⚠ Warning: Fallback to ImageNet weights. {e}")
            else:
                print("⚠ Warning: Could not load weights from init_path. Using ImageNet weights.")

        # Extract Stem and Stages manually
        # ConvNeXt structure: model.stem -> model.stages[0,1,2,3]
        self.stem = full_model.stem
        self.stages = full_model.stages  # nn.Sequential containing 4 ConvNeXtStages
        
        # Dimensions for ConvNeXt Base: [128, 256, 512, 1024]
        # feature_info is a list of dicts with 'num_chs' key
        if isinstance(full_model.feature_info, list):
            dims = [info['num_chs'] for info in full_model.feature_info]
        else:
            dims = full_model.feature_info.channels() 
        self.dims = dims

        # -------------------------
        # 2. DA3 (Frozen) - Fix: add error handling
        # -------------------------
        da3_ckpt_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "checkpoints",
            "DA3",
        )
        # Force local loading only, prevent network download
        if not os.path.exists(da3_ckpt_dir):
            raise FileNotFoundError(
                f"DA3 model directory does not exist: {da3_ckpt_dir}. "
                "Please ensure the model is downloaded locally."
            )
        # Lazy import to avoid libGL.so.1 dependency when not needed
        from depth_anything_3.api import DepthAnything3
        self.da3 = DepthAnything3.from_pretrained(da3_ckpt_dir)
        self.da3.eval()
        for p in self.da3.parameters(): 
            p.requires_grad = False
        self.da3_dim = 384

        # -------------------------
        # 3. Multi-Scale Adapters (Bottleneck + Zero Init)
        # -------------------------
        # Adapter for Stage 0 (128 ch)
        self.adapter0 = self._make_adapter(self.da3_dim, dims[0])
        # Adapter for Stage 1 (256 ch)
        self.adapter1 = self._make_adapter(self.da3_dim, dims[1])
        # Adapter for Stage 2 (512 ch)
        self.adapter2 = self._make_adapter(self.da3_dim, dims[2])
        # Adapter for Stage 3 (1024 ch)
        self.adapter3 = self._make_adapter(self.da3_dim, dims[3])

        # -------------------------
        # 4. Deep MLP Head (3 Layers)
        # -------------------------
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fusion Dim = Sum of all stage channels
        # Base: 128 + 256 + 512 + 1024 = 1920
        self.fusion_dim = sum(dims)
        hidden_dim = 1024

        def make_mlp_head(out_dim):
            return nn.Sequential(
                nn.Linear(self.fusion_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim)
            )

        self.final_layers = nn.ModuleList([
            make_mlp_head(532),  # id
            make_mlp_head(45),   # exp
            make_mlp_head(439),  # tex
            make_mlp_head(3),    # angle
            make_mlp_head(27),   # gamma
            make_mlp_head(2),    # tx, ty
            make_mlp_head(1)     # tz
        ])
        
        # Zero-Init only the LAST Linear layer of each branch
        for branch in self.final_layers:
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def _make_adapter(self, in_dim, out_dim):
        """Creates a bottleneck adapter: In -> In/2 -> Out, with Zero-Init"""
        mid_dim = in_dim // 2
        adapter = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, out_dim, 1, bias=True)
        )
        nn.init.zeros_(adapter[-1].weight)
        nn.init.zeros_(adapter[-1].bias)
        return adapter

    def forward(self, x):
        b = x.size(0)

        # --- DA3 Feature Extraction - Fix: add device management and format compatibility handling ---
        self.da3.to(x.device)  # Fix: explicitly move device
        self.da3.eval()
        with torch.no_grad():
            x_da3 = x.unsqueeze(1)  # (B, 1, 3, H, W)
            raw_out = self.da3(
                x_da3,
                export_feat_layers=[11],
                infer_gs=False
            )
            aux = raw_out.aux
            feat_5d = aux["feat_layer_11"]  # (B, N, H', W', C=384) or (B, N, C, H', W')
            
            # Average over the view dimension and remove it
            feat_sc = feat_5d.mean(dim=1)  # (B, H', W', C) or (B, C, H', W')
            
            # Fix: robustness check for both channel-last and channel-first formats
            if feat_sc.shape[-1] == 384:  # Channel last: (B, H', W', 384)
                da3_feat = feat_sc.permute(0, 3, 1, 2).contiguous()  # -> (B, 384, H', W')
            else:  # Channel first: (B, 384, H', W')
                da3_feat = feat_sc

        # --- Multi-Stage Injection ---
        
        x = self.stem(x)
        
        # Stage 0
        x = self.stages[0](x)
        da3_s0 = F.interpolate(da3_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter0(da3_s0)
        feat_s0 = self.gap(x).view(b, -1)

        # Stage 1
        x = self.stages[1](x)
        da3_s1 = F.interpolate(da3_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter1(da3_s1)
        feat_s1 = self.gap(x).view(b, -1)

        # Stage 2
        x = self.stages[2](x)
        da3_s2 = F.interpolate(da3_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter2(da3_s2)
        feat_s2 = self.gap(x).view(b, -1)

        # Stage 3
        x = self.stages[3](x)
        da3_s3 = F.interpolate(da3_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter3(da3_s3)
        feat_s3 = self.gap(x).view(b, -1)

        # --- Deep Head Aggregation ---
        combined = torch.cat([feat_s0, feat_s1, feat_s2, feat_s3], dim=1)  # 1920 dim
        
        output = [layer(combined) for layer in self.final_layers]
        return torch.cat(output, dim=1)  # (B, 1049)


class MultiScaleFusedResNetDINOv3(nn.Module):
    """
    Multi-Scale Fusion: ResNet50 + DINOv3
    
    Architecture:
    - Inject DINOv3 features at ALL 4 Stages (Layer 1-4).
    - Uses DINOv3's intermediate hidden states for TRUE high-resolution fusion.
    - Aggregates features from all scales (HyperColumn).
    - 3-Layer MLP Head.
    """

    def __init__(
        self,
        num_outputs: int = 1049,
        dinov3_path: str = DEFAULT_DINOV3_PATH,
        init_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_outputs = num_outputs

        # -------------------------
        # 1. ResNet50 Backbone (Split)
        # -------------------------
        print("Loading ResNet-50 (Trainable) for Multi-Scale Fusion...")
        use_torchvision = (init_path is None or not os.path.isfile(init_path))
        base_resnet = tv_models.resnet50(pretrained=use_torchvision)
        
        if not use_torchvision:
            try:
                state_dict = torch.load(init_path, map_location='cpu')
                if 'state_dict' in state_dict: 
                    state_dict = state_dict['state_dict']
                base_resnet.load_state_dict(state_dict, strict=False)
                print("✓ Loaded custom ResNet weights")
            except Exception as e:
                print(f"⚠ Warning: Fallback to ImageNet weights. {e}")

        # Split ResNet into stages
        self.stem = nn.Sequential(base_resnet.conv1, base_resnet.bn1, base_resnet.relu, base_resnet.maxpool)
        self.layer1 = base_resnet.layer1 # 256 ch, 56x56
        self.layer2 = base_resnet.layer2 # 512 ch, 28x28
        self.layer3 = base_resnet.layer3 # 1024 ch, 14x14
        self.layer4 = base_resnet.layer4 # 2048 ch, 7x7

        # -------------------------
        # 2. DINOv3 (Frozen) with Hidden States
        # -------------------------
        print(f"Loading DINOv3 from local path: {dinov3_path}")
        if not os.path.exists(dinov3_path):
            raise FileNotFoundError(
                f"DINOv3 model path does not exist: {dinov3_path}. "
                "Please ensure the model is downloaded locally."
            )
            
        # Configure DINOv3 to output intermediate layers
        config = AutoConfig.from_pretrained(dinov3_path, trust_remote_code=True)
        config.output_hidden_states = True # CRITICAL: Enable intermediate outputs
        
        self.dinov3 = AutoModel.from_pretrained(dinov3_path, config=config, trust_remote_code=True, local_files_only=True)
        self.dinov3.eval()
        self.dinov3.requires_grad_(False)
        
        # DINOv3 (ConvNeXt-Small) widths: [96, 192, 384, 768]
        dino_dims = [96, 192, 384, 768]

        # -------------------------
        # 3. Multi-Scale Adapters & Pre-Norms
        # -------------------------
        # We need to map DINO dims to ResNet dims
        # ResNet dims: [256, 512, 1024, 2048]
        
        # Layer 1: 96 -> 256
        self.norm1 = nn.InstanceNorm2d(dino_dims[0], affine=False)
        self.adapter1 = self._make_adapter(dino_dims[0], 256)
        
        # Layer 2: 192 -> 512
        self.norm2 = nn.InstanceNorm2d(dino_dims[1], affine=False)
        self.adapter2 = self._make_adapter(dino_dims[1], 512)
        
        # Layer 3: 384 -> 1024
        self.norm3 = nn.InstanceNorm2d(dino_dims[2], affine=False)
        self.adapter3 = self._make_adapter(dino_dims[2], 1024)
        
        # Layer 4: 768 -> 2048
        self.norm4 = nn.InstanceNorm2d(dino_dims[3], affine=False)
        self.adapter4 = self._make_adapter(dino_dims[3], 2048)

        # -------------------------
        # 4. Deep MLP Head (3 Layers)
        # -------------------------
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fusion Dim = Sum of ResNet channels
        fusion_dim = 256 + 512 + 1024 + 2048 # 3840
        hidden_dim = 1024

        def make_deep_head(out_dim):
            return nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim)
            )

        self.final_layers = nn.ModuleList([
            make_deep_head(532), # id
            make_deep_head(45),  # exp
            make_deep_head(439), # tex
            make_deep_head(3),   # angle
            make_deep_head(27),  # gamma
            make_deep_head(2),   # tx, ty
            make_deep_head(1)    # tz
        ])
        
        # Zero-Init Last Linear Layer
        for branch in self.final_layers:
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def _make_adapter(self, in_dim, out_dim):
        """Bottleneck Adapter with Zero-Init"""
        mid_dim = in_dim // 2 if in_dim > 64 else in_dim
        adapter = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, out_dim, 1, bias=True)
        )
        nn.init.zeros_(adapter[-1].weight)
        nn.init.zeros_(adapter[-1].bias)
        return adapter

    def forward(self, x):
        b = x.size(0)

        # --- DINOv3 Features (Multi-Scale) ---
        self.dinov3.to(x.device)  # Fix: explicitly move device
        self.dinov3.eval()
        with torch.no_grad():
            outputs = self.dinov3(x, output_hidden_states=True)
            # hidden_states is usually a tuple: (embeddings, stage0, stage1, stage2, stage3)
            # We want the last 4 stages
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states
                # Get the last 4 stages (skip CLS token/embedding if present)
                # hidden_states typically includes: [embedding, stage0, stage1, stage2, stage3]
                # We want stages 0-3 (indices may vary, typically last 4)
                dino_feats = hidden_states[-4:]
            else:
                # Fallback: if hidden_states not available, use last_hidden_state and reshape
                # This is less ideal but provides compatibility
                feat_map = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
                feat_map = reshape_dinov3_features(feat_map)
                # Use the same feature for all stages (not ideal, but functional)
                dino_feats = [feat_map] * 4
            
            # Reshape each feature to (B, C, H, W) format
            dino_feats_reshaped = []
            for feat in dino_feats:
                if feat.dim() == 3:  # (B, N, C) format
                    feat = reshape_dinov3_features(feat)
                elif feat.dim() != 4:  # Already (B, C, H, W) or unexpected format
                    raise ValueError(f"Unexpected DINOv3 feature shape: {feat.shape}")
                dino_feats_reshaped.append(feat)

        # --- Multi-Stage Injection ---
        
        # Stem
        x = self.stem(x)
        
        # Layer 1
        x = self.layer1(x)
        feat_d1 = dino_feats_reshaped[0]
        if feat_d1.shape[-2:] != x.shape[-2:]:
             feat_d1 = F.interpolate(feat_d1, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter1(self.norm1(feat_d1))
        pool_l1 = self.gap(x).view(b, -1)

        # Layer 2
        x = self.layer2(x)
        feat_d2 = dino_feats_reshaped[1]
        if feat_d2.shape[-2:] != x.shape[-2:]:
             feat_d2 = F.interpolate(feat_d2, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter2(self.norm2(feat_d2))
        pool_l2 = self.gap(x).view(b, -1)

        # Layer 3
        x = self.layer3(x)
        feat_d3 = dino_feats_reshaped[2]
        if feat_d3.shape[-2:] != x.shape[-2:]:
             feat_d3 = F.interpolate(feat_d3, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter3(self.norm3(feat_d3))
        pool_l3 = self.gap(x).view(b, -1)

        # Layer 4
        x = self.layer4(x)
        feat_d4 = dino_feats_reshaped[3]
        if feat_d4.shape[-2:] != x.shape[-2:]:
             feat_d4 = F.interpolate(feat_d4, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter4(self.norm4(feat_d4))
        pool_l4 = self.gap(x).view(b, -1)

        # --- Deep Head Aggregation ---
        combined = torch.cat([pool_l1, pool_l2, pool_l3, pool_l4], dim=1)
        
        output = [layer(combined) for layer in self.final_layers]
        return torch.cat(output, dim=1)


class MultiScaleFusedConvNeXtV2DINOv3(nn.Module):
    """
    Multi-Scale Fusion: ConvNeXt V2 + DINOv3
    
    Architecture:
    - Homogeneous Architecture Fusion (ConvNeXt + ConvNeXt).
    - Stage-to-Stage injection (Stage 0->0, 1->1, 2->2, 3->3).
    - Pre-Norm + Zero-Init for stability.
    - 3-Layer Deep MLP Head.
    """

    def __init__(
        self,
        num_outputs: int = 1049,
        trainable_model_name: str = 'convnextv2_base.fcmae_ft_in22k_in1k',
        dinov3_path: str = DEFAULT_DINOV3_PATH,
        init_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_outputs = num_outputs

        # -------------------------
        # 1. ConvNeXt V2 Backbone
        # -------------------------
        print(f"Loading {trainable_model_name} (Trainable)...")
        
        # Check if local weights are provided
        has_local = init_path and (os.path.isfile(init_path) or os.path.isdir(init_path))
        
        # Load full model to extract stem and stages
        full_model = timm.create_model(trainable_model_name, pretrained=(not has_local))
        
        if has_local:
            # Use existing load_convnextv2_weights function for robust weight loading
            state_dict = load_convnextv2_weights(init_path)
            if state_dict is not None:
                try:
                    full_model.load_state_dict(state_dict, strict=False)
                    print("✓ Loaded custom ConvNeXt weights")
                except Exception as e:
                    print(f"⚠ Warning: Fallback to ImageNet weights. {e}")
            else:
                print("⚠ Warning: Could not load weights from init_path. Using ImageNet weights.")

        self.stem = full_model.stem
        self.stages = full_model.stages
        
        # ConvNeXt Base dims: [128, 256, 512, 1024]
        # feature_info is a list of dicts with 'num_chs' key
        if isinstance(full_model.feature_info, list):
            backbone_dims = [info['num_chs'] for info in full_model.feature_info]
        else:
            backbone_dims = full_model.feature_info.channels()

        # -------------------------
        # 2. DINOv3 (Frozen)
        # -------------------------
        print(f"Loading DINOv3 from local path: {dinov3_path}")
        if not os.path.exists(dinov3_path):
            raise FileNotFoundError(
                f"DINOv3 model path does not exist: {dinov3_path}. "
                "Please ensure the model is downloaded locally."
            )
        
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(dinov3_path, trust_remote_code=True)
        config.output_hidden_states = True # Output intermediate layers
        
        self.dinov3 = AutoModel.from_pretrained(dinov3_path, config=config, trust_remote_code=True, local_files_only=True)
        self.dinov3.eval()
        self.dinov3.requires_grad_(False)
        
        # DINOv3 (Small) dims: [96, 192, 384, 768]
        dino_dims = [96, 192, 384, 768]

        # -------------------------
        # 3. Adapters & Pre-Norms
        # -------------------------
        
        # Stage 0
        self.norm0 = nn.InstanceNorm2d(dino_dims[0], affine=False)
        self.adapter0 = self._make_adapter(dino_dims[0], backbone_dims[0])
        
        # Stage 1
        self.norm1 = nn.InstanceNorm2d(dino_dims[1], affine=False)
        self.adapter1 = self._make_adapter(dino_dims[1], backbone_dims[1])
        
        # Stage 2
        self.norm2 = nn.InstanceNorm2d(dino_dims[2], affine=False)
        self.adapter2 = self._make_adapter(dino_dims[2], backbone_dims[2])
        
        # Stage 3
        self.norm3 = nn.InstanceNorm2d(dino_dims[3], affine=False)
        self.adapter3 = self._make_adapter(dino_dims[3], backbone_dims[3])

        # -------------------------
        # 4. Deep MLP Head
        # -------------------------
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fusion Dim = Sum of Backbone Dims
        fusion_dim = sum(backbone_dims) # 1920
        hidden_dim = 1024

        def make_deep_head(out_dim):
            return nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim)
            )

        self.final_layers = nn.ModuleList([
            make_deep_head(532), 
            make_deep_head(45),  
            make_deep_head(439), 
            make_deep_head(3),   
            make_deep_head(27),  
            make_deep_head(2),   
            make_deep_head(1)    
        ])
        
        for branch in self.final_layers:
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def _make_adapter(self, in_dim, out_dim):
        """Bottleneck Adapter with Zero-Init"""
        mid_dim = in_dim // 2
        adapter = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, out_dim, 1, bias=True)
        )
        nn.init.zeros_(adapter[-1].weight)
        nn.init.zeros_(adapter[-1].bias)
        return adapter

    def forward(self, x):
        b = x.size(0)

        # --- DINOv3 Features ---
        self.dinov3.to(x.device)  # Fix: explicitly move device
        self.dinov3.eval()
        with torch.no_grad():
            outputs = self.dinov3(x, output_hidden_states=True)
            # Get the last 4 stages from hidden_states
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states
                # Get the last 4 stages
                dino_feats = hidden_states[-4:]
            else:
                # Fallback: use last_hidden_state
                feat_map = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
                feat_map = reshape_dinov3_features(feat_map)
                dino_feats = [feat_map] * 4
            
            # Reshape each feature to (B, C, H, W) format
            dino_feats_reshaped = []
            for feat in dino_feats:
                if feat.dim() == 3:  # (B, N, C) format
                    feat = reshape_dinov3_features(feat)
                elif feat.dim() != 4:  # Already (B, C, H, W) or unexpected format
                    raise ValueError(f"Unexpected DINOv3 feature shape: {feat.shape}")
                dino_feats_reshaped.append(feat)

        # --- Multi-Stage Fusion ---
        
        x = self.stem(x)
        
        # Stage 0
        x = self.stages[0](x)
        d0 = dino_feats_reshaped[0]
        if d0.shape[-2:] != x.shape[-2:]: 
            d0 = F.interpolate(d0, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter0(self.norm0(d0))
        pool_s0 = self.gap(x).view(b, -1)

        # Stage 1
        x = self.stages[1](x)
        d1 = dino_feats_reshaped[1]
        if d1.shape[-2:] != x.shape[-2:]: 
            d1 = F.interpolate(d1, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter1(self.norm1(d1))
        pool_s1 = self.gap(x).view(b, -1)

        # Stage 2
        x = self.stages[2](x)
        d2 = dino_feats_reshaped[2]
        if d2.shape[-2:] != x.shape[-2:]: 
            d2 = F.interpolate(d2, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter2(self.norm2(d2))
        pool_s2 = self.gap(x).view(b, -1)

        # Stage 3
        x = self.stages[3](x)
        d3 = dino_feats_reshaped[3]
        if d3.shape[-2:] != x.shape[-2:]: 
            d3 = F.interpolate(d3, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter3(self.norm3(d3))
        pool_s3 = self.gap(x).view(b, -1)

        # --- Aggregation ---
        combined = torch.cat([pool_s0, pool_s1, pool_s2, pool_s3], dim=1)
        
        output = [layer(combined) for layer in self.final_layers]
        return torch.cat(output, dim=1)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (Spatial Attention Module, SAM)
    Enhances network attention to high-frequency facial features such as eyes and mouth
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv1(scale)
        return x * self.sigmoid(scale)


class MultiScaleFusedConvNeXtV2DA3_Enhanced(nn.Module):
    """
    Enhanced Multi-Scale ConvNeXt V2 + DA3 Fusion with Spatial Attention.
    
    Adds a Spatial Attention Module on top of the original multi-scale fusion:
    - Apply spatial attention after fusion at each stage and before GAP
    - Helps the network focus on high-frequency facial features such as eyes and mouth
    - Keep all original multi-scale fusion logic, Pre-Norm, and Zero-Init adapters
    """
    
    def __init__(
        self,
        num_outputs: int = 1049,
        trainable_model_name: str = 'convnextv2_base',
        init_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_outputs = num_outputs

        # -------------------------
        # 1. ConvNeXt Backbone (Split)
        # -------------------------
        print(f"Loading {trainable_model_name} (Trainable) for Enhanced Multi-Scale Fusion with Spatial Attention...")
        
        has_local = init_path and (os.path.isfile(init_path) or os.path.isdir(init_path))
        full_model = timm.create_model(trainable_model_name, pretrained=(not has_local))
        
        if has_local:
            state_dict = load_convnextv2_weights(init_path)
            if state_dict is not None:
                try:
                    full_model.load_state_dict(state_dict, strict=False)
                    print("✓ Loaded custom ConvNeXt weights")
                except Exception as e:
                    print(f"⚠ Warning: Fallback to ImageNet weights. {e}")
            else:
                print("⚠ Warning: Could not load weights from init_path. Using ImageNet weights.")

        self.stem = full_model.stem
        self.stages = full_model.stages
        
        if isinstance(full_model.feature_info, list):
            dims = [info['num_chs'] for info in full_model.feature_info]
        else:
            dims = full_model.feature_info.channels() 
        self.dims = dims

        # -------------------------
        # 2. DA3 (Frozen)
        # -------------------------
        da3_ckpt_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "checkpoints",
            "DA3",
        )
        if not os.path.exists(da3_ckpt_dir):
            raise FileNotFoundError(
                f"DA3 model directory does not exist: {da3_ckpt_dir}. "
                "Please ensure the model is downloaded locally."
            )
        # Lazy import to avoid libGL.so.1 dependency when not needed
        from depth_anything_3.api import DepthAnything3
        self.da3 = DepthAnything3.from_pretrained(da3_ckpt_dir)
        self.da3.eval()
        for p in self.da3.parameters(): 
            p.requires_grad = False
        self.da3_dim = 384

        # -------------------------
        # 3. Multi-Scale Adapters (Bottleneck + Zero Init)
        # -------------------------
        self.adapter0 = self._make_adapter(self.da3_dim, dims[0])
        self.adapter1 = self._make_adapter(self.da3_dim, dims[1])
        self.adapter2 = self._make_adapter(self.da3_dim, dims[2])
        self.adapter3 = self._make_adapter(self.da3_dim, dims[3])

        # -------------------------
        # 4. Spatial Attention Modules (NEW)
        # -------------------------
        self.sa0 = SpatialAttention(kernel_size=7)
        self.sa1 = SpatialAttention(kernel_size=7)
        self.sa2 = SpatialAttention(kernel_size=7)
        self.sa3 = SpatialAttention(kernel_size=7)

        # -------------------------
        # 5. Deep MLP Head (3 Layers)
        # -------------------------
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fusion_dim = sum(dims)
        hidden_dim = 1024

        def make_mlp_head(out_dim):
            return nn.Sequential(
                nn.Linear(self.fusion_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim)
            )

        self.final_layers = nn.ModuleList([
            make_mlp_head(532),  # id
            make_mlp_head(45),   # exp
            make_mlp_head(439),  # tex
            make_mlp_head(3),    # angle
            make_mlp_head(27),   # gamma
            make_mlp_head(2),    # tx, ty
            make_mlp_head(1)     # tz
        ])
        
        # Zero-Init only the LAST Linear layer of each branch
        for branch in self.final_layers:
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def _make_adapter(self, in_dim, out_dim):
        """Creates a bottleneck adapter: In -> In/2 -> Out, with Zero-Init"""
        mid_dim = in_dim // 2
        adapter = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, out_dim, 1, bias=True)
        )
        nn.init.zeros_(adapter[-1].weight)
        nn.init.zeros_(adapter[-1].bias)
        return adapter

    def forward(self, x):
        b = x.size(0)

        # --- DA3 Feature Extraction ---
        self.da3.to(x.device)
        self.da3.eval()
        with torch.no_grad():
            x_da3 = x.unsqueeze(1)
            raw_out = self.da3(
                x_da3,
                export_feat_layers=[11],
                infer_gs=False
            )
            aux = raw_out.aux
            feat_5d = aux["feat_layer_11"]
            
            feat_sc = feat_5d.mean(dim=1)
            
            if feat_sc.shape[-1] == 384:
                da3_feat = feat_sc.permute(0, 3, 1, 2).contiguous()
            else:
                da3_feat = feat_sc

        # --- Multi-Stage Injection with Spatial Attention ---
        
        x = self.stem(x)
        
        # Stage 0: Fusion -> Spatial Attention -> GAP
        x = self.stages[0](x)
        da3_s0 = F.interpolate(da3_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter0(da3_s0)
        x = self.sa0(x)  # NEW: Spatial Attention
        feat_s0 = self.gap(x).view(b, -1)

        # Stage 1
        x = self.stages[1](x)
        da3_s1 = F.interpolate(da3_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter1(da3_s1)
        x = self.sa1(x)  # NEW: Spatial Attention
        feat_s1 = self.gap(x).view(b, -1)

        # Stage 2
        x = self.stages[2](x)
        da3_s2 = F.interpolate(da3_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter2(da3_s2)
        x = self.sa2(x)  # NEW: Spatial Attention
        feat_s2 = self.gap(x).view(b, -1)

        # Stage 3
        x = self.stages[3](x)
        da3_s3 = F.interpolate(da3_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter3(da3_s3)
        x = self.sa3(x)  # NEW: Spatial Attention
        feat_s3 = self.gap(x).view(b, -1)

        # --- Deep Head Aggregation ---
        combined = torch.cat([feat_s0, feat_s1, feat_s2, feat_s3], dim=1)
        
        output = [layer(combined) for layer in self.final_layers]
        return torch.cat(output, dim=1)


class MultiScaleFusedConvNeXtV2DINOv3_Enhanced(nn.Module):
    """
    Enhanced Multi-Scale ConvNeXt V2 + DINOv3 Fusion with Spatial Attention.
    
    Adds a Spatial Attention Module on top of the original multi-scale fusion:
    - Apply spatial attention after fusion at each stage and before GAP
    - Helps the network focus on high-frequency facial features such as eyes and mouth
    - Keep all original multi-scale fusion logic, Pre-Norm, and Zero-Init adapters
    """
    
    def __init__(
        self,
        num_outputs: int = 1049,
        trainable_model_name: str = 'convnextv2_base.fcmae_ft_in22k_in1k',
        dinov3_path: str = DEFAULT_DINOV3_PATH,
        init_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_outputs = num_outputs

        # -------------------------
        # 1. ConvNeXt V2 Backbone
        # -------------------------
        print(f"Loading {trainable_model_name} (Trainable) for Enhanced Multi-Scale Fusion with Spatial Attention...")
        
        has_local = init_path and (os.path.isfile(init_path) or os.path.isdir(init_path))
        full_model = timm.create_model(trainable_model_name, pretrained=(not has_local))
        
        if has_local:
            state_dict = load_convnextv2_weights(init_path)
            if state_dict is not None:
                try:
                    full_model.load_state_dict(state_dict, strict=False)
                    print("✓ Loaded custom ConvNeXt weights")
                except Exception as e:
                    print(f"⚠ Warning: Fallback to ImageNet weights. {e}")
            else:
                print("⚠ Warning: Could not load weights from init_path. Using ImageNet weights.")

        self.stem = full_model.stem
        self.stages = full_model.stages
        
        if isinstance(full_model.feature_info, list):
            backbone_dims = [info['num_chs'] for info in full_model.feature_info]
        else:
            backbone_dims = full_model.feature_info.channels()

        # -------------------------
        # 2. DINOv3 (Frozen)
        # -------------------------
        print(f"Loading DINOv3 from local path: {dinov3_path}")
        if not os.path.exists(dinov3_path):
            raise FileNotFoundError(
                f"DINOv3 model path does not exist: {dinov3_path}. "
                "Please ensure the model is downloaded locally."
            )
        
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(dinov3_path, trust_remote_code=True)
        config.output_hidden_states = True
        
        self.dinov3 = AutoModel.from_pretrained(dinov3_path, config=config, trust_remote_code=True, local_files_only=True)
        self.dinov3.eval()
        self.dinov3.requires_grad_(False)
        
        dino_dims = [96, 192, 384, 768]

        # -------------------------
        # 3. Adapters & Pre-Norms
        # -------------------------
        
        # Stage 0
        self.norm0 = nn.InstanceNorm2d(dino_dims[0], affine=False)
        self.adapter0 = self._make_adapter(dino_dims[0], backbone_dims[0])
        
        # Stage 1
        self.norm1 = nn.InstanceNorm2d(dino_dims[1], affine=False)
        self.adapter1 = self._make_adapter(dino_dims[1], backbone_dims[1])
        
        # Stage 2
        self.norm2 = nn.InstanceNorm2d(dino_dims[2], affine=False)
        self.adapter2 = self._make_adapter(dino_dims[2], backbone_dims[2])
        
        # Stage 3
        self.norm3 = nn.InstanceNorm2d(dino_dims[3], affine=False)
        self.adapter3 = self._make_adapter(dino_dims[3], backbone_dims[3])

        # -------------------------
        # 4. Spatial Attention Modules (NEW)
        # -------------------------
        self.sa0 = SpatialAttention(kernel_size=7)
        self.sa1 = SpatialAttention(kernel_size=7)
        self.sa2 = SpatialAttention(kernel_size=7)
        self.sa3 = SpatialAttention(kernel_size=7)

        # -------------------------
        # 5. Deep MLP Head
        # -------------------------
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        fusion_dim = sum(backbone_dims)
        hidden_dim = 1024

        def make_deep_head(out_dim):
            return nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim)
            )

        self.final_layers = nn.ModuleList([
            make_deep_head(532), 
            make_deep_head(45),  
            make_deep_head(439), 
            make_deep_head(3),   
            make_deep_head(27),  
            make_deep_head(2),   
            make_deep_head(1)    
        ])
        
        for branch in self.final_layers:
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def _make_adapter(self, in_dim, out_dim):
        """Bottleneck Adapter with Zero-Init"""
        mid_dim = in_dim // 2
        adapter = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, out_dim, 1, bias=True)
        )
        nn.init.zeros_(adapter[-1].weight)
        nn.init.zeros_(adapter[-1].bias)
        return adapter

    def forward(self, x):
        b = x.size(0)

        # --- DINOv3 Features ---
        self.dinov3.to(x.device)
        self.dinov3.eval()
        with torch.no_grad():
            outputs = self.dinov3(x, output_hidden_states=True)
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states
                dino_feats = hidden_states[-4:]
            else:
                feat_map = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
                feat_map = reshape_dinov3_features(feat_map)
                dino_feats = [feat_map] * 4
            
            dino_feats_reshaped = []
            for feat in dino_feats:
                if feat.dim() == 3:
                    feat = reshape_dinov3_features(feat)
                elif feat.dim() != 4:
                    raise ValueError(f"Unexpected DINOv3 feature shape: {feat.shape}")
                dino_feats_reshaped.append(feat)

        # --- Multi-Stage Fusion with Spatial Attention ---
        
        x = self.stem(x)
        
        # Stage 0: Fusion -> Spatial Attention -> GAP
        x = self.stages[0](x)
        d0 = dino_feats_reshaped[0]
        if d0.shape[-2:] != x.shape[-2:]: 
            d0 = F.interpolate(d0, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter0(self.norm0(d0))
        x = self.sa0(x)  # NEW: Spatial Attention
        pool_s0 = self.gap(x).view(b, -1)

        # Stage 1
        x = self.stages[1](x)
        d1 = dino_feats_reshaped[1]
        if d1.shape[-2:] != x.shape[-2:]: 
            d1 = F.interpolate(d1, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter1(self.norm1(d1))
        x = self.sa1(x)  # NEW: Spatial Attention
        pool_s1 = self.gap(x).view(b, -1)

        # Stage 2
        x = self.stages[2](x)
        d2 = dino_feats_reshaped[2]
        if d2.shape[-2:] != x.shape[-2:]: 
            d2 = F.interpolate(d2, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter2(self.norm2(d2))
        x = self.sa2(x)  # NEW: Spatial Attention
        pool_s2 = self.gap(x).view(b, -1)

        # Stage 3
        x = self.stages[3](x)
        d3 = dino_feats_reshaped[3]
        if d3.shape[-2:] != x.shape[-2:]: 
            d3 = F.interpolate(d3, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.adapter3(self.norm3(d3))
        x = self.sa3(x)  # NEW: Spatial Attention
        pool_s3 = self.gap(x).view(b, -1)

        # --- Aggregation ---
        combined = torch.cat([pool_s0, pool_s1, pool_s2, pool_s3], dim=1)
        
        output = [layer(combined) for layer in self.final_layers]
        return torch.cat(output, dim=1)


# class RecogNetWrapper(nn.Module):
#     def __init__(self, net_recog, pretrained_path=None, input_size=112):
#         super(RecogNetWrapper, self).__init__()
#         net = get_model(name=net_recog, fp16=False)
#         if pretrained_path:
#             state_dict = torch.load(pretrained_path, map_location='cpu')
#             net.load_state_dict(state_dict)
#             print("loading pretrained net_recog %s from %s" %(net_recog, pretrained_path))
#         for param in net.parameters():
#             param.requires_grad = False
#         self.net = net
#         self.preprocess = lambda x: 2 * x - 1
#         self.input_size=input_size
        
#     def forward(self, image, M):
#         image = self.preprocess(resize_n_crop(image, M, self.input_size))
#         id_feature = F.normalize(self.net(image), dim=-1, p=2)
#         return id_feature


# adapted from https://github.com/pytorch/vision/edit/master/torchvision/models/resnet.py
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

def conv1x1_relu(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Sequential:
    """1x1 convolution with ReLU activation (for FFHQ compatibility)"""
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias),
        nn.ReLU(inplace=True)
    )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        use_last_fc: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.use_last_fc = use_last_fc
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if self.use_last_fc:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.use_last_fc:
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


func_dict = {
    'resnet18': (resnet18, 512),
    'resnet50': (resnet50, 2048)
}


