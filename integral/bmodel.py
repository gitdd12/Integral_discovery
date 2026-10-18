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


def sinusoid(n, d):
    pos = torch.arange(n).float()[:, None]
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class QueryNet(nn.Module):
    """매 질의 뒤에 그 시점의 넓이 추정(log)을 낸다 (anytime). 위치 정보는 사인 함수 → 학습 때 못 본 긴 K 도 입력 가능."""

    def __init__(self, K=16, d=128, layers=2, heads=4, rect=False, norm=False):
        super().__init__()
        self.K, self.d, self.rect, self.norm = K, d, rect, norm
        self.start = nn.Parameter(torch.zeros(d))
        self.inp0 = nn.Linear(2, d)          # (a, 폭) → 첫 토큰
        self.inp = nn.Linear(4 if rect else (3 if norm else 2), d)  # (이전 위치[, 폭], 이전 높이[, 조각 넓이]) → 다음 토큰
        self.w_head = nn.Linear(d, 1)        # rect 도구: 조각의 폭(구간 길이 대비 비율)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.u_head = nn.Linear(d, 1)
        self.out_head = nn.Linear(d, 1)

    def forward(self, a, b, grid, K=None, fixed_u=None):
        """반환: 추정 [B, K+1] (t번 질의한 뒤의 log넓이; t=0 은 아직 아무것도 안 잼), 질의 위치 [B,K]."""
        K = K or self.K
        B = a.shape[0]
        pe = sinusoid(K + 1, self.d).to(a.device)
        if self.norm:
            toks = [self.start.expand(B, 1, self.d)]
            f1 = None
        else:
            toks = [self.inp0(torch.stack([a / 5.0, torch.log(b - a)], -1))[:, None]]
        ests, us, ws = [], [], []
        for t in range(K + 1):
            n = len(toks)
            x = torch.cat(toks, 1) + pe[:n]
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), 1)
            h = self.norm(self.enc(x, mask=mask))[:, -1]
            if self.norm:
                # 출력 = log(평균 높이 / 첫 높이). 환경이 (첫 높이 × 구간 길이)를 곱해 넓이로 바꿈 → rect(가로,세로)
                z = self.out_head(h)[:, 0]
                ests.append(z + torch.log(f1) + torch.log(b - a) if f1 is not None else torch.zeros_like(z))
            else:
                ests.append(self.out_head(h)[:, 0])
            if t == K:
                break
            u = torch.sigmoid(self.u_head(h))[:, 0] if fixed_u is None else fixed_u[:, t]
            f = lerp_query(grid, u)
            if self.rect:
                w = torch.sigmoid(self.w_head(h))[:, 0]
                r = w * (b - a) * f                      # rect(가로, 세로) = 가로 × 세로 (환경이 계산해서 돌려줌)
                feat = [u, w, torch.log(f.clamp_min(1e-6)), torch.log(r.clamp_min(1e-9))]
                ws.append(w)
            elif self.norm:
                if f1 is None:
                    f1 = f.clamp_min(1e-6)
                ratio = f / f1
                feat = [u, ratio, torch.log(ratio.clamp_min(1e-6))]
            else:
                feat = [u, torch.log(f.clamp_min(1e-6))]
            toks.append(self.inp(torch.stack(feat, -1))[:, None])
            us.append(u)
        self.last_w = torch.stack(ws, 1) if ws else None
        return torch.stack(ests, 1), torch.stack(us, 1)


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
