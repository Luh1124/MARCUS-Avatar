"""
3D Face Reconstruction Network Wrapper

This module provides a wrapper for various backbone networks used in 3D face reconstruction,
including Deep3Dhyper multi-scale ConvNeXtV2 + DINOv3, timm ConvNeXt, and ResNet backbones.
"""

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .resnet_backbone import conv1x1, conv1x1_relu, func_dict
from .networks import (  # type: ignore
    MultiScaleFusedConvNeXtV2DINOv3,
    MultiScaleFusedConvNeXtV2DINOv3_Enhanced,
)


DEFAULT_MERGED_RECON_MODEL_PATH = os.environ.get(
    "MERGED_RECON_MODEL_PATH",
    os.path.join("ckpts", "deep3d_merge_model", "epoch_latest.pth"),
)


class ReconNetWrapper(nn.Module):
    """
    3D Face Reconstruction Network Wrapper
    
    Supports multiple backbone architectures:
    - Deep3Dhyper: Multi-scale ConvNeXtV2 + DINOv3 + SpatialAttention
    - timm ConvNeXt: Standard ConvNeXt architectures from timm
    - ResNet: ResNet18/ResNet50 backbones
    
    Args:
        backbone_name: Name of the backbone architecture
        use_last_fc: Whether the backbone outputs coefficients directly
        fc_dim_dict: Dictionary containing dimension info (id_dims, exp_dims, tex_dims)
        limit_exp_range: Whether to limit expression range (use ReLU for exp layer)
        pretrain_model_path: Path to pretrained model checkpoint
        init_SH: Initial spherical harmonics coefficients
        device: Device to run the model on
    """

    def __init__(
        self,
        backbone_name: str = 'multiscale_convnextv2_base_dinov3_enhanced',
        use_last_fc: bool = False,
        fc_dim_dict: Optional[Dict[str, int]] = None,
        limit_exp_range: bool = True,
        pretrain_model_path: str = DEFAULT_MERGED_RECON_MODEL_PATH,
        init_SH: Optional[np.ndarray] = None,
        device: str = 'cuda',
    ):
        super(ReconNetWrapper, self).__init__()

        if fc_dim_dict is None:
            raise ValueError("fc_dim_dict must be provided")
        if init_SH is None:
            init_SH = np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0])

        self.use_last_fc = use_last_fc
        self.fc_dim_dict = fc_dim_dict
        self.device = device

        # Calculate total output dimension
        self.fc_dim = (
            fc_dim_dict['id_dims'] +
            fc_dim_dict['exp_dims'] +
            fc_dim_dict['tex_dims'] +
            3 +  # angle
            27 +  # gamma (SH coefficients)
            2 +  # tx, ty
            1    # tz
        )

        # Initialize backbone network
        backbone_last_dim = self._init_backbone(backbone_name, pretrain_model_path)

        # Initialize final layers if needed
        if not self.use_last_fc:
            if backbone_last_dim is None:
                raise ValueError(
                    "backbone_last_dim must be provided when use_last_fc=False"
                )
            self._init_final_layers(backbone_last_dim, limit_exp_range)

        # Initialize spherical harmonics coefficients
        self.init_SH = torch.from_numpy(
            init_SH.reshape([1, 1, -1]).astype(np.float32)
        ).float().to(device)

        # Move model to device
        self.backbone.to(device)
        if hasattr(self, 'final_layers'):
            self.final_layers.to(device)

        # Load pretrained weights
        self._load_pretrained(pretrain_model_path, map_location=device)
        print(f'[INFO] Loaded recon model from {pretrain_model_path}')

    def _init_backbone(self, backbone_name: str, pretrain_model_path: str) -> Optional[int]:
        """
        Initialize backbone network based on backbone_name.
        
        Args:
            backbone_name: Name of the backbone architecture
            pretrain_model_path: Path to pretrained model (used for offline init)
        
        Returns:
            backbone_last_dim: Last dimension of backbone output (None if use_last_fc=True)
        """
        # Deep3Dhyper multi-scale ConvNeXtV2 + DINOv3 backbones
        if backbone_name in {
            'multiscale_convnextv2_base_dinov3_enhanced',
            'multiscale_convnextv2_base_dinov3',
        }:
            # These networks output coefficients directly
            if not self.use_last_fc:
                self.use_last_fc = True

            # Use existing path to avoid timm downloading weights
            offline_init_path = (
                pretrain_model_path
                if (pretrain_model_path and os.path.exists(pretrain_model_path))
                else None
            )

            if backbone_name == 'multiscale_convnextv2_base_dinov3_enhanced':
                self.backbone = MultiScaleFusedConvNeXtV2DINOv3_Enhanced(
                    num_outputs=self.fc_dim,
                    init_path=offline_init_path
                )
            else:
                self.backbone = MultiScaleFusedConvNeXtV2DINOv3(
                    num_outputs=self.fc_dim,
                    init_path=offline_init_path
                )
            return None

        # timm ConvNeXt backbones
        elif backbone_name.startswith('convnext'):
            try:
                import timm  # type: ignore
            except ImportError as e:
                raise ImportError(
                    f"ConvNeXt backbone requires timm. Please install timm or use a different backbone. "
                    f"Original error: {e}"
                )

            if not self.use_last_fc:
                raise ValueError(
                    "ConvNeXt backbone requires use_last_fc=True "
                    "(timm model outputs coefficients directly). "
                    "Otherwise, two heads will be created."
                )

            self.backbone = timm.create_model(
                backbone_name,
                pretrained=False,
                num_classes=self.fc_dim
            )
            return None

        # ResNet backbones
        else:
            if backbone_name not in func_dict:
                raise NotImplementedError(
                    f'Backbone network [{backbone_name}] is not implemented. '
                    f'Available: {list(func_dict.keys())}'
                )

            func, backbone_last_dim = func_dict[backbone_name]
            self.backbone = func(
                use_last_fc=self.use_last_fc,
                num_classes=self.fc_dim
            )
            return backbone_last_dim

    def _init_final_layers(self, backbone_last_dim: int, limit_exp_range: bool):
        """
        Initialize final fully connected layers.
        
        Args:
            backbone_last_dim: Last dimension of backbone output
            limit_exp_range: Whether to use ReLU for expression layer
        """
        if limit_exp_range:
            self.final_layers = nn.ModuleList([
                conv1x1(backbone_last_dim, self.fc_dim_dict['id_dims'], bias=True),  # id
                conv1x1_relu(backbone_last_dim, self.fc_dim_dict['exp_dims'], bias=True),  # exp (with ReLU)
                conv1x1(backbone_last_dim, self.fc_dim_dict['tex_dims'], bias=True),  # tex
                conv1x1(backbone_last_dim, 3, bias=True),  # angle
                conv1x1(backbone_last_dim, 27, bias=True),  # gamma (SH)
                conv1x1(backbone_last_dim, 2, bias=True),  # tx, ty
                conv1x1(backbone_last_dim, 1, bias=True)  # tz
            ])
        else:
            self.final_layers = nn.ModuleList([
                conv1x1(backbone_last_dim, self.fc_dim_dict['id_dims'], bias=True),  # id
                conv1x1(backbone_last_dim, self.fc_dim_dict['exp_dims'], bias=True),  # exp
                conv1x1(backbone_last_dim, self.fc_dim_dict['tex_dims'], bias=True),  # tex
                conv1x1(backbone_last_dim, 3, bias=True),  # angle
                conv1x1(backbone_last_dim, 27, bias=True),  # gamma (SH)
                conv1x1(backbone_last_dim, 2, bias=True),  # tx, ty
                conv1x1(backbone_last_dim, 1, bias=True)  # tz
            ])

        # Initialize weights
        def init_weights(module):
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.weight, 0.)

        for layer in self.final_layers:
            try:
                if hasattr(layer, 'weight') and layer.weight is not None:
                    nn.init.constant_(layer.weight, 0.)  # type: ignore
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.)  # type: ignore
            except (AttributeError, TypeError):
                layer.apply(init_weights)

    def to(self, device: Union[str, torch.device]):
        """Move model to specified device."""
        self.device = device
        self.init_SH = self.init_SH.to(device)
        self.backbone.to(device)
        if hasattr(self, 'final_layers'):
            self.final_layers.to(device)
        return self

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Dictionary containing reconstruction coefficients:
            - 'id': Identity coefficients
            - 'exp': Expression coefficients
            - 'tex': Texture coefficients
            - 'angle': Rotation angles
            - 'gamma': Spherical harmonics coefficients
            - 'trans': Translation vector
        """
        x = self.backbone(x)
        
        if not self.use_last_fc:
            outputs = []
            for layer in self.final_layers:
                outputs.append(layer(x))
            x = torch.flatten(torch.cat(outputs, dim=1), 1)
        
        return self.get_coeffs(x)

    def _load_pretrained(self, pretrain_model_path: str, map_location: str = 'cpu'):
        """
        Load pretrained checkpoint with compatibility for different formats.
        
        Supports:
        - Full checkpoint dict (with 'net_recon' or 'state_dict' keys)
        - Direct state_dict
        - Wrapper-style keys ('backbone.xxx', 'final_layers.xxx')
        - Backbone-style keys ('stem.xxx', 'stages.xxx')
        - DataParallel keys ('module.xxx')
        
        Args:
            pretrain_model_path: Path to checkpoint file
            map_location: Device to load checkpoint on
        """
        if not os.path.exists(pretrain_model_path):
            raise FileNotFoundError(
                f"Pretrained model not found: {pretrain_model_path}"
            )

        ckpt = torch.load(pretrain_model_path, map_location=map_location)
        
        # Extract state_dict from checkpoint
        if isinstance(ckpt, dict):
            if 'net_recon' in ckpt:
                state_dict = ckpt['net_recon']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                # Direct state_dict
                state_dict = ckpt
        else:
            state_dict = ckpt

        if not isinstance(state_dict, dict):
            raise ValueError(
                f'Unsupported checkpoint format: {type(state_dict)}. '
                f'Expected dict or checkpoint dict.'
            )

        keys = list(state_dict.keys())
        if len(keys) == 0:
            raise ValueError('Empty state_dict in checkpoint.')

        # Detect key style
        is_wrapper_style = (
            any(k.startswith('backbone.') for k in keys) or
            (hasattr(self, 'final_layers') and any(k.startswith('final_layers.') for k in keys))
        )
        
        is_deep3dhyper_style = any(
            k.startswith(prefix)
            for prefix in (
                'stem.', 'stages.', 'dinov3.', 'adapter',
                'sa0.', 'sa1.', 'sa2.', 'sa3.', 'final_layers.'
            )
            for k in keys
        )
        
        is_backbone_style = any(
            k.startswith('stem.') or k.startswith('stages.')
            for k in keys
        )

        # Load based on detected style
        if is_wrapper_style:
            result = super().load_state_dict(state_dict, strict=False)
            missing, unexpected = self._extract_load_result(result)
            print(f'[INFO] Loaded wrapper-style state_dict: '
                  f'missing={len(missing)}, unexpected={len(unexpected)}')
            return

        if is_deep3dhyper_style or is_backbone_style:
            result = self.backbone.load_state_dict(state_dict, strict=False)
            missing, unexpected = self._extract_load_result(result)
            print(f'[INFO] Loaded backbone-style state_dict into self.backbone: '
                  f'missing={len(missing)}, unexpected={len(unexpected)}')
            return

        # Try DataParallel prefix stripping
        if any(k.startswith('module.') for k in keys):
            stripped = {k[len('module.'):]: v for k, v in state_dict.items()}
            result = super().load_state_dict(stripped, strict=False)
            missing, unexpected = self._extract_load_result(result)
            print(f'[INFO] Loaded module.-stripped state_dict: '
                  f'missing={len(missing)}, unexpected={len(unexpected)}')
            return

        # Fallback: try loading into backbone
        result = self.backbone.load_state_dict(state_dict, strict=False)
        missing, unexpected = self._extract_load_result(result)
        print(f'[WARNING] Fallback: loaded state_dict into self.backbone: '
              f'missing={len(missing)}, unexpected={len(unexpected)}')

    @staticmethod
    def _extract_load_result(result: Union[Tuple, object]) -> Tuple[list, list]:
        """Extract missing and unexpected keys from load_state_dict result."""
        if hasattr(result, 'missing_keys') and hasattr(result, 'unexpected_keys'):
            # type: ignore - load_state_dict returns NamedTuple with missing_keys/unexpected_keys
            return list(result.missing_keys), list(result.unexpected_keys)  # type: ignore
        elif isinstance(result, tuple) and len(result) == 2:
            return list(result[0]), list(result[1])
        else:
            return [], []

    def get_coeffs(self, net_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parse network output into coefficient dictionary.
        
        Args:
            net_output: Network output tensor [B, fc_dim]
            
        Returns:
            Dictionary containing parsed coefficients
        """
        id_dims = self.fc_dim_dict['id_dims']
        exp_dims = self.fc_dim_dict['exp_dims']
        tex_dims = self.fc_dim_dict['tex_dims']

        idx = 0
        id_coeffs = net_output[:, idx:idx + id_dims]
        idx += id_dims

        exp_coeffs = net_output[:, idx:idx + exp_dims]
        idx += exp_dims

        tex_coeffs = net_output[:, idx:idx + tex_dims]
        idx += tex_dims

        angle = net_output[:, idx:idx + 3]
        idx += 3

        gamma = net_output[:, idx:idx + 27]
        idx += 27

        translations = net_output[:, idx:]

        # Reshape and add initial SH coefficients
        gamma = gamma.reshape(-1, 3, 9)
        gamma = gamma + self.init_SH

        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angle,
            'gamma': gamma,
            'trans': translations
        }
