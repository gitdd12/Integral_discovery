"""B④ 모델: K번 위치를 골라 높이를 받고, 마지막에 log(넓이)를 낸다. 끝까지 미분 가능."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def lerp_query(grid, u):
    """grid:[B,G] 구간을 G-1등분한 높이, u:[B] ∈ [0,1] → 선형 보간한 높이 [B] (u에 대해 미분 가능)."""
    G = grid.shape[1]
    t = u.clamp(0, 1) * (G - 1)
    i0 = t.floor().long().clamp(0, G - 2)
    w = t - i0.float()
    g0 = grid.gather(1, i0[:, None])[:, 0]
    g1 = grid.gather(1, (i0 + 1)[:, None])[:, 0]
    return g0 * (1 - w) + g1 * w


class QueryNet(nn.Module):
    def __init__(self, K=16, d=128, layers=2, heads=4):
        super().__init__()
        self.K = K
        self.inp0 = nn.Linear(2, d)          # (a, 폭) → 첫 토큰
        self.inp = nn.Linear(2, d)           # (이전 위치, 이전 높이) → 다음 토큰
        self.pos = nn.Parameter(torch.zeros(K + 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.u_head = nn.Linear(d, 1)
        self.out_head = nn.Linear(d, 1)

    def forward(self, a, b, grid, K=None, fixed_u=None):
        """a,b:[B], grid:[B,G]. 반환: log넓이 예측 [B], 질의 위치 [B,K]."""
        K = K or self.K
        B = a.shape[0]
        toks = [self.inp0(torch.stack([a / 5.0, torch.log(b - a)], -1))[:, None]]
        us = []
        for t in range(K):
            x = torch.cat(toks, 1) + self.pos[: len(toks)]
            mask = torch.triu(torch.ones(len(toks), len(toks), dtype=torch.bool, device=x.device), 1)
            h = self.norm(self.enc(x, mask=mask))[:, -1]
            u = torch.sigmoid(self.u_head(h))[:, 0] if fixed_u is None else fixed_u[:, t]
            f = lerp_query(grid, u)
            toks.append(self.inp(torch.stack([u, torch.log(f.clamp_min(1e-6))], -1))[:, None])
            us.append(u)
        x = torch.cat(toks, 1) + self.pos[: len(toks)]
        mask = torch.triu(torch.ones(len(toks), len(toks), dtype=torch.bool, device=x.device), 1)
        h = self.norm(self.enc(x, mask=mask))[:, -1]
        return self.out_head(h)[:, 0], torch.stack(us, 1)


def baselines(grid, a, b, K):
    """같은 질의 수 K 의 고전 구적법 (정답에 대한 로그 오차 비교용). 격자에서 균등 위치를 읽는다."""
    G = grid.shape[1]
    w = (b - a)
    pos_mid = (torch.arange(K, device=grid.device).float() + 0.5) / K
    mid = torch.stack([lerp_query(grid, pos_mid[k].expand(grid.shape[0])) for k in range(K)], 1).mean(1) * w
    pos_tr = torch.linspace(0, 1, K, device=grid.device)
    vals = torch.stack([lerp_query(grid, pos_tr[k].expand(grid.shape[0])) for k in range(K)], 1)
    trap = (vals[:, 0] / 2 + vals[:, 1:-1].sum(1) + vals[:, -1] / 2) / (K - 1) * w
    return mid, trap
