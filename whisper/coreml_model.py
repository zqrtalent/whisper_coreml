from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from whisper.model_shared import LayerNorm, Linear
from whisper.state_cache import SliceUpdateKeyValueCache

try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False

class MultiHeadAttention_Decoder(nn.Module):
    def __init__(self, n_state: int, n_head: int, n_layer: int, n_text_context:int = 0, n_token_seq_len = 0):
        super().__init__()
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_state = n_state
        self.n_text_context = n_text_context
        self.n_token_seq_len = n_token_seq_len

        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        
    def qkv_attention(
        self,
        q: Tensor,  # (B,C_text,S_text) => (1,>1,384)
        k: Tensor,  # (B,C_text,S_text) or (B,C_audio,S_audio) => (1,>3,384) or (1,1500,384)
        v: Tensor,  # (B,C_text,S_text) or (B,C_audio,S_audio) => (1,>3,384) or (1,1500,384)
        mask: Optional[Tensor] = None,  # (T,T)
        pos: Optional[Tensor] = None,
        
        # v_new: Optional[Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25

        # (B,nH,1,Dh) => (1,6,1,64)
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        # (B,nH,T,Dh) => (1,6,1500 | 448,64)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        # if k_new is not None and v_new is not None:
        #     k_new = k_new.view(*k_new.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        # (B,nH,T,Dh) => (1,6,1500 | 448,64)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        # if k_new is not None and v_new is not None:
        #     v_new = v_new.view(*v_new.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
          
        qk = (q.to(q.dtype) * scale) @ (k.to(q.dtype) * scale).transpose(-1, -2) # (1,6,1,448)
        # if k_new is not None and v_new is not None:
        #     qk_new = (q.to(q.dtype) * scale) @ (k_new.to(q.dtype) * scale).transpose(-1, -2) # (1,6,1,448)
        
        # Apply mask.
        # 0 -inf -inf -inf ... -inf
        # 0   0  -inf -inf ... -inf
        # 0   0   0   -inf ... -inf
        if mask is not None:
            idx = torch.arange(0,self.n_token_seq_len) + pos
            qk = qk + mask[idx,:self.n_text_context]
            
        qk = qk.float()
        # if k_new is not None and v_new is not None:
        #     qk_new = qk_new.float()
        w = F.softmax(qk, dim=-1).to(q.dtype)
        # if k_new is not None and v_new is not None:
        #     w_new = F.softmax(qk_new, dim=-1).to(q.dtype)
        out = (w @ v.to(q.dtype)).permute(0, 2, 1, 3).flatten(start_dim=2)
        # out_new = None
        # if k_new is not None and v_new is not None:
        #     out_new = (w_new @ v_new.to(q.dtype)).permute(0, 2, 1, 3).flatten(start_dim=2)
        return out


class MultiHeadAttention_Decoder_CrossAttn(MultiHeadAttention_Decoder):
    def __init__(self, n_state, n_head, n_layer, n_text_context, n_token_seq_len):
        super().__init__(n_state, n_head, n_layer, n_text_context, n_token_seq_len)

    def forward(
        self,
        x: Tensor,  # (B,Context_token,S)
        xa: Optional[Tensor] = None,  # (B,Context_audio,S)
        mask: Optional[Tensor] = None,  # (B,T)
        cache: Optional[SliceUpdateKeyValueCache] = None
    ):
        q = self.query(x)
        if cache is not None:
            k = cache.k
            v = cache.v
        else:
            k = self.key(xa)
            v = self.value(xa)
        wv = super().qkv_attention(q, k, v, mask)
        return self.out(wv)


class MultiHeadAttention_Decoder_SelfAttn(MultiHeadAttention_Decoder):
    def __init__(self, n_state, n_head, n_layer, n_text_context, n_token_seq_len):
        super().__init__(n_state, n_head, n_layer, n_text_context, n_token_seq_len)
        
        oh = torch.eye(n_text_context, n_text_context, dtype=torch.float32)
        self.register_buffer("OH", oh, persistent=False)
        
    def compare_results(t1, t2):
        if t1.shape != t2.shape:
            print(f"shape are not the same! ml {t1.shape} pt {t2.shape}")
            return False

        diff = t1 - t2
        mae = diff.abs().mean()
        max_err = diff.abs().max()
        l2 = torch.norm(diff) / torch.norm(t1)
        cos = torch.nn.functional.cosine_similarity(t1, t2, dim=-1).mean()
        print(f"ml -> pt")
        print(f"mean = {mae}")
        print(f"max_err = {max_err}")
        print(f"l2 = {l2}")
        print(f"cos = {cos}")

    def forward(
        self,
        x: Tensor,  # (B,Context_token,S)
        pos: Tensor,  # (1)
        xa: Optional[Tensor] = None,  # (B,Context_audio,S)
        mask: Optional[Tensor] = None,  # (B,T)
        cache: Optional[SliceUpdateKeyValueCache] = None
    ):
        # assert x.shape[-2] == self.n_token_seq_len
        q = self.query(x)
        k_new = self.key(x)
        v_new = self.value(x)
        
        if cache is not None:
            # indices = torch.arange(0,self.n_token_seq_len) + pos[0]
            # indices = torch.clamp(indices, 0, self.n_text_context-1)
            
            # select one-hot row for t
            # mask_ = self.OH.index_select(0, pos + self.n_token_seq_len - 1).squeeze(0)  # [1,T,1]
            
            # mask_ = torch.zeros(self.n_text_context, 1, dtype=torch.float)
            # mask_[indices,:] = 1
            
            # # (1, 448, 384) * (1.0 - (448,1)) + (3,384)*(448,1)
            k_slice = cache.k[self.n_layer].to(torch.float32)
            v_slice = cache.v[self.n_layer].to(torch.float32)
             
            i = 0
            tpos = pos
            while i < self.n_token_seq_len:
                # mask_ = self.OH.index_select(0, torch.tensor(i, dtype=torch.int32)).view(-1,1)  # [T,1]
                mask_ = self.OH.index_select(0, tpos).view(-1,1)  # [T,1]
                k_slice = k_slice * (1.0 - mask_)
                k_slice = k_slice + k_new[:,i:i+1,:] * mask_
                v_slice = v_slice * (1.0 - mask_)
                v_slice = v_slice + v_new[:,i:i+1,:] * mask_
                i = i + 1
                tpos = tpos + 1
                     
            # k_slice = (cache.k[self.n_layer].to(torch.float32) * (1.0 - mask_) + k_new * mask_).contiguous()
            # v_slice = (cache.v[self.n_layer].to(torch.float32) * (1.0 - mask_) + v_new * mask_).contiguous()
                        
            cache.k[self.n_layer] = k_slice.to(torch.float16)
            cache.v[self.n_layer] = v_slice.to(torch.float16)
              
            k, v = k_slice, v_slice
            # k, v = k_new, v_new
        else:
            k = self.key(x)
            v = self.value(x)
            
        wv = self.qkv_attention(q, k, v, mask, pos=pos)
        return self.out(wv)


class ResidualAttentionBlock_Decoder(nn.Module):
    def __init__(self, n_state: int, n_head: int, n_layer: int, n_txt_context: int, n_token_seq_len: int = 1):
        super().__init__()

        self.n_layer = n_layer
        self.attn = MultiHeadAttention_Decoder_SelfAttn(
            n_state, n_head, n_layer, n_txt_context, n_token_seq_len)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention_Decoder_CrossAttn(
            n_state, n_head, n_layer, n_txt_context, n_token_seq_len)
        self.cross_attn_ln = LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)
        self.n_state = n_state

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        self_cache: Optional[SliceUpdateKeyValueCache] = None,
        cross_cache: Optional[SliceUpdateKeyValueCache] = None
    ):
        # Self attention
        y_attn = self.attn(self.attn_ln(x), pos, mask=mask, cache=self_cache)
        x = x + y_attn

        # Cross attention
        y_cross = self.cross_attn(self.cross_attn_ln(x), xa, cache=cross_cache)
        x = x + y_cross

        # mlp
        x = x + self.mlp(self.mlp_ln(x))
        return x

class TextDecoder_coreml(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, n_token_seq_len: int = 1
    ):
        super().__init__()

        self.n_token_seq_len = n_token_seq_len
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock_Decoder] = nn.ModuleList(
            [ResidualAttentionBlock_Decoder(
                n_state, n_head, n_layer=i, n_txt_context=n_ctx, n_token_seq_len=n_token_seq_len) for i in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

        cache_dtype = torch.float16
        self_cache_shape = [n_layer, 1, n_ctx, n_state]
        self.self_cache = SliceUpdateKeyValueCache(shape=self_cache_shape, dtype=cache_dtype)
        self.register_buffer("selfKeyCache", self.self_cache.k, persistent=False)
        self.register_buffer("selfValueCache", self.self_cache.v, persistent=False)
        
        self.n_head = n_head
        self.d_head = n_state // n_head
        self.n_layer = n_layer
        self.n_state = n_state
        self.n_ctx = n_ctx

    def forward(self,
                x: Tensor,  # (B, T) -> (1, >=1) - fixed size
                xa: Tensor,  # (B, ContextAudio, S) - (1, 1500, 384)
                pos: Optional[Tensor] = None,  # (1) [int]
                cross_cache_k_1: Optional[Tensor] = None, # (B,ContextAudio, S) => (1,1500,384)
                cross_cache_v_1: Optional[Tensor] = None, # (B,ContextAudio, S) => (1,1500,384)
                cross_cache_k_2: Optional[Tensor] = None, # (B,ContextAudio, S) => (1,1500,384)
                cross_cache_v_2: Optional[Tensor] = None, # (B,ContextAudio, S) => (1,1500,384)
                cross_cache_k_3: Optional[Tensor] = None, # (B,ContextAudio, S) => (1,1500,384)
                cross_cache_v_3: Optional[Tensor] = None, # (B,ContextAudio, S) => (1,1500,384)
                cross_cache_k_4: Optional[Tensor] = None, # (B,ContextAudio, S) => (1,1500,384)
                cross_cache_v_4: Optional[Tensor] = None, # (B,ContextAudio, S) => (1,1500,384)
                ):
        """
        x : torch.LongTensor, shape = (batch_size, == n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        if pos is None:
            pos = torch.tensor([0], dtype=torch.long)

        pos_indices = torch.arange(0,self.n_token_seq_len) + pos[0]
        # tokens = x.gather(1, pos.view(1,1)).view(-1)
        tokens = x.view(-1)

        # x_token = self.token_embedding(tokens) + self.positional_embedding[pos:self.n_token_seq_len]
        x_token = self.token_embedding(tokens) + self.positional_embedding[pos_indices]
        x_token = x_token.view(1, -1, self.n_state).to(xa.dtype)
        
        cache_keys = [cross_cache_k_1, cross_cache_k_2, cross_cache_k_3, cross_cache_k_4]
        cache_values = [cross_cache_v_1, cross_cache_v_2, cross_cache_v_3, cross_cache_v_4]
        
        # start_pos = torch.tensor(0, dtype=torch.int32)
        start_pos = pos[0]
        
        # cross_cache = SliceUpdateKeyValueCache(k = cache_keys[0], v = cache_values[0]) if cache_keys[0] is not None and cache_values is not None else None
        # x_token = self.blocks[0](x_token, start_pos, xa, mask=self.mask,
        #                     self_cache = self.self_cache, cross_cache=cross_cache)
        for (i,block) in enumerate(self.blocks):
            cross_cache = SliceUpdateKeyValueCache(k = cache_keys[i], v = cache_values[i]) if cache_keys[i] is not None and cache_values is not None else None
            x_token = block(x_token, start_pos, xa, mask=self.mask,
                            self_cache = self.self_cache, cross_cache=cross_cache)

        x_token = self.ln(x_token)
        logits = (
            x_token @ torch.transpose(self.token_embedding.weight.to(x_token.dtype), 0, 1)).float()

        return (logits)
    
    def audio_features_kv(self, xa: Tensor) -> tuple[Tensor, Tensor]:
        """
        Generates a key/value cache based on audio features

        xa - audio features tensor with shape (B,C,S) => (1,1500,384)
        """
        k_cache_stack: list[Tensor] = []
        v_cache_stack: list[Tensor] = []
        for block in self.blocks:
            k = block.cross_attn.key(xa).detach()
            v = block.cross_attn.value(xa).detach()
            k_cache_stack.append(k)
            v_cache_stack.append(v)
        k_cache = torch.stack(k_cache_stack, dim=0)
        v_cache = torch.stack(v_cache_stack, dim=0)
        return (k_cache, v_cache)