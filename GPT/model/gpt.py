from dataclasses import dataclass
import math

from numpy import square
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embed: int = 768

def RMS_Norm(x, eps=1e-5):
    squares = x.pow(2)
    mean_sq = squares.mean(dim=-1, keepdim=True)
    RMS = torch.sqrt(mean_sq + eps)
    out = x / RMS
    return out

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)
    out = out.to(x.type)
    return out

class CasualSelfAttetion(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embed = config.n_embed
        self.head_dim = self.n_embed // self.n_head
        
        assert self.n_embed % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.q_proj = nn.Linear(self.n_embed, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.n_embed, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.n_embed, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_embed, self.n_embed, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = RMS_Norm(q), RMS_Norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        
        Tq = q.size(2)
        Tk = k.size(2)

        enable_gqa = self.n_head != self.n_kv_head
        if kv_cache is None or Tq == Tk:
            





