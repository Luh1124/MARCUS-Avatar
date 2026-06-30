from typing import Tuple, Union, List, Optional, Dict, Callable, Any
import os

from huggingface_hub import snapshot_download
import numpy as np
import PIL
import torch
import math
from .pipeline_intrinsix_image_edit import IntrinsiXEditPipeline
from diffusers.utils import replace_example_docstring
from huggingface_hub.utils import validate_hf_hub_args

from .cross_intrinsic_attention import CrossIntrinsicAttnProcessor2_0
from .batch_lora import inject_trainable_batched_lora, extract_loras, save_lora_weights


def set_attn_processor(pipe, processor):
    """
    Set the attention processor for all transformer blocks.
    """
    for block in pipe.transformer.transformer_blocks:
        block.attn.set_processor(processor)
    for block in pipe.transformer.single_transformer_blocks:
        block.attn.set_processor(processor)

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = width if width % 16 == 0 else (width // 16 + 1) * 16
    height = height if height % 16 == 0 else (height // 16 + 1) * 16

    width = int(width)
    height = int(height)

    return width, height


class IntrinsiXPipeline(IntrinsiXEditPipeline):
    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, 
                        pretrained_model_name_or_path=None, 
                        base_model_path="meituan-longcat/LongCat-Image-Edit",
                        longcat_image_edit_pipeline=None,
                        **kwargs):
        # Load the Base model
        if longcat_image_edit_pipeline is None:
            pipe = IntrinsiXEditPipeline.from_pretrained(pretrained_model_name_or_path=base_model_path, **kwargs)
        else:
            pipe = longcat_image_edit_pipeline

        # Inject the LoRA modules
        if pretrained_model_name_or_path is not None:
            features = [
                "albedo",
                "normal",
                "rsd",
            ]

            lora_configs = list()
            for feature in features:
                if feature == "albedo":
                    lora_configs.append({
                                        "r": 32,
                                        "dropout_p": 0.0,
                                        "scale": 1.0
                                    })
                elif feature == "normal":
                    lora_configs.append({
                                        "r": 32,
                                        "dropout_p": 0.0,
                                        "scale": 1.0
                                    })
                elif feature == "rsd":
                    lora_configs.append({
                                        "r": 32,
                                        "dropout_p": 0.0,
                                        "scale": 1.0
                                    })
                elif feature == "shading":
                    lora_configs.append({
                                        "r": 32,
                                        "dropout_p": 0.0,
                                        "scale": 1.0
                                    })
            inject_trainable_batched_lora(model=pipe.transformer,
                                        target_modules={"to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out", "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2"},
                                        lora_configs=lora_configs,
                                        verbose=True)
            
            # Load the LoRA weights
            if pretrained_model_name_or_path != base_model_path:
                cache_dir = kwargs.pop("cache_dir", None)
                
                # Check if it's a local path or HuggingFace Hub repo_id
                if os.path.exists(pretrained_model_name_or_path) or os.path.isdir(pretrained_model_name_or_path):
                    # Local path
                    lora_path = pretrained_model_name_or_path
                else:
                    # HuggingFace Hub repo_id
                    lora_path = snapshot_download(repo_id=pretrained_model_name_or_path, cache_dir=cache_dir)
                
                lora_state_dict = cls.lora_state_dict(lora_path)
                
                # Process keys: remove "transformer." prefix if present, or use keys as-is
                loras_state_dict = {}
                for k, v in lora_state_dict.items():
                    # Remove "transformer." prefix if it exists
                    if k.startswith("transformer."):
                        new_key = k.replace("transformer.", "")
                    else:
                        new_key = k
                    loras_state_dict[new_key] = v
                
                pipe.transformer.load_state_dict(loras_state_dict, strict=False)

            # Set Cross-Intrinsic-Attention
            crossattn_processor = CrossIntrinsicAttnProcessor2_0()
            set_attn_processor(pipe, crossattn_processor)

        return pipe

    @classmethod
    def lora_state_dict(
            cls,
            pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
            return_alphas: bool = False,
            **kwargs,
    ):
        r"""
        Return state dict for lora weights and the network alphas.

        <Tip warning={true}>

        We support loading A1111 formatted LoRA checkpoints in a limited capacity.

        This function is experimental and might change in the future.

        </Tip>

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.

        """
        # If input is already a dict, return it directly
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            if return_alphas:
                return pretrained_model_name_or_path_or_dict, None
            else:
                return pretrained_model_name_or_path_or_dict
        
        # Check if it's a directory path
        if isinstance(pretrained_model_name_or_path_or_dict, str):
            # First, try to load from pytorch_lora_weights.safetensors (new format)
            if os.path.isdir(pretrained_model_name_or_path_or_dict):
                lora_path = os.path.join(pretrained_model_name_or_path_or_dict, "pytorch_lora_weights.safetensors")
                if os.path.exists(lora_path):
                    try:
                        from safetensors.torch import load_file
                        state_dict = load_file(lora_path)
                        if return_alphas:
                            return state_dict, None
                        else:
                            return state_dict
                    except Exception as e:
                        # If loading fails, fall through to the original method
                        pass
            
            # Also check if the path itself is a safetensors file
            if pretrained_model_name_or_path_or_dict.endswith(".safetensors"):
                if os.path.exists(pretrained_model_name_or_path_or_dict):
                    try:
                        from safetensors.torch import load_file
                        state_dict = load_file(pretrained_model_name_or_path_or_dict)
                        if return_alphas:
                            return state_dict, None
                        else:
                            return state_dict
                    except Exception as e:
                        # If loading fails, fall through to the original method
                        pass
        
        # Fallback to original method for compatibility
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        state_dict = cls._fetch_state_dict(
            pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
            weight_name=weight_name,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            allow_pickle=allow_pickle,
        )

        if return_alphas:
            return state_dict, None
        else:
            return state_dict
