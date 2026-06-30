import einops
import torch
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention import Attention
from diffusers.models.embeddings import apply_rotary_emb

from torch.nn.modules.module import _global_forward_hooks, _global_forward_hooks_always_called


class CrossIntrinsicAttnProcessor2_0(nn.Module):
    """
    Processor for implementing scaled dot-product attention between multiple images within each batch.
    Supports 'Decoupled Attention' to separate self-refinement from cross-batch alignment.
    """

    def __init__(self, *args, dropout=None, decoupled_attention=False, **kwargs):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CrossIntrinsicAttnProcessor2_0 requires PyTorch 2.0")

        self.dropout = dropout if dropout is not None else 0.0
        self.decoupled_attention = decoupled_attention

    def process_attention(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None) -> torch.FloatTensor:
        
        # Get batch size
        if encoder_hidden_states is not None:
            batch_size = encoder_hidden_states.shape[0]
        else:
            batch_size = hidden_states.shape[0]

        # ===============================================
        # 1. Base projections (Projections)
        # ===============================================
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # ===============================================
        # 2. Inject text conditioning (Text Conditioning) - Key fix: pre-processing
        # ===============================================
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # Along the sequence dimension (dim=2) concatenate text and image
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        # ===============================================
        # 3. Apply RoPE (Rotary Embeddings) - Key fix: pre-processing
        # ===============================================
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # ===============================================
        # 4. Attention strategy branch
        # ===============================================
        if self.decoupled_attention:
            # --- Strategy A: split-head strategy (Split-Head Strategy) ---
            # The first half of heads attend only to self, while the second half attends to the whole batch.
            
            num_heads = attn.heads
            num_self_heads = num_heads // 2
            
            # 4.1 Split heads
            q_self, q_cross = torch.split(query, [num_self_heads, num_heads - num_self_heads], dim=1)
            k_self, k_cross = torch.split(key, [num_self_heads, num_heads - num_self_heads], dim=1)
            v_self, v_cross = torch.split(value, [num_self_heads, num_heads - num_self_heads], dim=1)

            # 4.2 Cross branch: flatten the batch dimension
            # k_cross Shape: [B, H_cross, L, D] -> [1, H_cross, B*L, D] -> [B, H_cross, B*L, D]
            # RoPE has already encoded position information, so flattening is safe
            k_cross = einops.rearrange(k_cross, "b h f d -> 1 h (b f) d").repeat(batch_size, 1, 1, 1)
            v_cross = einops.rearrange(v_cross, "b h f d -> 1 h (b f) d").repeat(batch_size, 1, 1, 1)

            # 4.3 Compute attention separately
            out_self = F.scaled_dot_product_attention(q_self, k_self, v_self, dropout_p=0.0, is_causal=False)
            out_cross = F.scaled_dot_product_attention(q_cross, k_cross, v_cross, dropout_p=0.0, is_causal=False)

            # 4.4 Merge heads
            hidden_states = torch.cat([out_self, out_cross], dim=1)

        else:
            # --- Strategy B: standard mixed strategy (Standard Probabilistic Cross-Batch) ---
            
            use_crossattention = True
            if self.training:
                use_crossattention = float(torch.rand(1)) > self.dropout

            if use_crossattention:
                # Flatten the batch dimension into the sequence dimension
                key_expanded = einops.rearrange(key, "b h f d -> 1 h (b f) d").repeat(batch_size, 1, 1, 1)
                value_expanded = einops.rearrange(value, "b h f d -> 1 h (b f) d").repeat(batch_size, 1, 1, 1)
                
                hidden_states = F.scaled_dot_product_attention(query, key_expanded, value_expanded, dropout_p=0.0, is_causal=False)
            else:
                hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        # ===============================================
        # 5. Output projection (Output Projection)
        # ===============================================
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split back into text and image
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1]:],
            )

            # Linear projection
            hidden_states = attn.to_out[0](hidden_states)
            # Dropout
            hidden_states = attn.to_out[1](hidden_states)
            # Text residual connection
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None) -> torch.FloatTensor:
        
        # Hook handling (standard Diffusers boilerplate)
        called_always_called_hooks = set()
        result = self.process_attention(attn=attn,
                                       hidden_states=hidden_states,
                                       encoder_hidden_states=encoder_hidden_states,
                                       attention_mask=attention_mask,
                                       image_rotary_emb=image_rotary_emb)
        args = []
        kwargs = {"attn": attn,
                 "hidden_states": hidden_states,
                 "encoder_hidden_states": encoder_hidden_states,
                 "attention_mask": attention_mask,
                 "image_rotary_emb": image_rotary_emb}
        
        if _global_forward_hooks or self._forward_hooks:
            for hook_id, hook in (
                    *_global_forward_hooks.items(),
                    *self._forward_hooks.items(),
            ):
                if hook_id in self._forward_hooks_always_called or hook_id in _global_forward_hooks_always_called:
                    called_always_called_hooks.add(hook_id)

                if hook_id in self._forward_hooks_with_kwargs:
                    hook_result = hook(self, args, kwargs, result)
                else:
                    hook_result = hook(self, args, result)

                if hook_result is not None:
                    result = hook_result
        return result
