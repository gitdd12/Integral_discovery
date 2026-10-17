"""작은 디코더 전용 트랜스포머 (다음 기호 맞히기). 위치는 RoPE → 학습 때 본 길이보다 길어도 덜 무너진다."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import vocab as V


@dataclass
class Config:
    vocab: int = V.VOCAB_SIZE
    layers: int = 4
    dim: int = 192
    heads: int = 4
    ff: int = 768
    ctx: int = 1024
    rope_base: float = 10000.0


def _rope_cache(cfg: Config, n: int, device):
    hd = cfg.dim // cfg.heads
    inv = 1.0 / (cfg.rope_base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(n, device=device).float()
    ang = torch.outer(t, inv)
    return ang.cos(), ang.sin()


def _rope(x, cos, sin):
    # x: [B, H, T, hd]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[: x.shape[2]], sin[: x.shape[2]]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


def _rope_at(x, c, s):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.h = cfg.heads
        self.n1, self.n2 = nn.LayerNorm(cfg.dim), nn.LayerNorm(cfg.dim)
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.ff)
        self.down = nn.Linear(cfg.ff, cfg.dim)

    def step(self, x, cos, sin, kc, vc, pos, mask):
        """토큰 하나씩 (KV 캐시). x:[B,1,D], pos:[B], kc/vc:[B,H,T,hd], mask:[B,1,1,T]"""
        B, _, D = x.shape
        hd = D // self.h
        q, k, v = self.qkv(self.n1(x)).view(B, 1, 3, self.h, hd).permute(2, 0, 3, 1, 4)
        c, s = cos[pos][:, None, None, :], sin[pos][:, None, None, :]
        q, k = _rope_at(q, c, s), _rope_at(k, c, s)
        ar = torch.arange(B, device=x.device)
        kc[ar, :, pos] = k[:, :, 0]
        vc[ar, :, pos] = v[:, :, 0]
        a = F.scaled_dot_product_attention(q, kc, vc, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, 1, D))
        return x + self.down(F.gelu(self.up(self.n2(x))))

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.down(F.gelu(self.up(self.n2(x))))


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.dim)
        self.out = nn.Linear(cfg.dim, cfg.vocab, bias=False)
        self.out.weight = self.emb.weight
        self.apply(self._init)
        self._cache = None

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ids):
        T = ids.shape[1]
        if self._cache is None or self._cache[0].shape[0] < T or self._cache[0].device != ids.device:
            self._cache = _rope_cache(self.cfg, max(T, 2048), ids.device)
        cos, sin = self._cache
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.out(self.norm(x))

    def init_cache(self, B, T, device="cpu"):
        hd = self.cfg.dim // self.cfg.heads
        z = lambda: torch.zeros(B, self.cfg.heads, T, hd, device=device)
        return [(z(), z()) for _ in self.blocks]

    def step(self, tokens, pos, cache):
        """각 줄이 자기 위치(pos)의 토큰 하나를 처리하고 다음 토큰의 로짓을 돌려준다."""
        T = cache[0][0].shape[2]
        if self._cache is None or self._cache[0].shape[0] < T or self._cache[0].device != tokens.device:
            self._cache = _rope_cache(self.cfg, max(T, 2048), tokens.device)
        cos, sin = self._cache
        mask = (torch.arange(T, device=tokens.device)[None, :] <= pos[:, None])[:, None, None, :]
        x = self.emb(tokens)[:, None, :]
        for b, (kc, vc) in zip(self.blocks, cache):
            x = b.step(x, cos, sin, kc, vc, pos, mask)
        return self.out(self.norm(x))[:, 0]

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def save(self, path, **extra):
        torch.save({"cfg": asdict(self.cfg), "model": self.state_dict(), **extra}, path)

    @staticmethod
    def load(path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        m = GPT(Config(**ck["cfg"]))
        m.load_state_dict(ck["model"])
        return m.to(device), ck
