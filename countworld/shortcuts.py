"""지름길 점검: 개수 정보를 가린 특징으로 정답을 맞히는 분류기가 우연 수준인지 확인한다 (DESIGN3.md §4).

특징 묶음:
  scene1  : 장면 1 의 종류별 개수 + 질의 종류 + 전체 길이  (a 를 포함하지만 b 가 없다)
  scene2  : 장면 2 의 종류별 개수 + 질의 종류 + 전체 길이
  nocount : 전체 길이, 질의 종류 둘이 같은지 (개수는 모두 빠짐)
  both    : 두 장면 전부 — 개수 a, b 가 다 있어서 풀 수 있어야 한다 (환경이 풀 수 있음을 확인)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .scene import N_TYPES, T0, SceneCfg, make_batch


def bag(tok):
    out = np.zeros((tok.shape[0], N_TYPES))
    for t in range(N_TYPES):
        out[:, t] = (tok == T0 + t).sum(1) - 0
    return out


def features(b, kind):
    q1 = np.eye(N_TYPES)[b["qa"]]
    q2 = np.eye(N_TYPES)[b["qb"]]
    T = b["T"][:, None].astype(float)
    # 질의 토큰도 bag 에 1 개씩 들어 있으나 모든 판에 똑같이 들어가 정보가 없다
    c1, c2 = bag(b["s1"]), bag(b["s2"])
    if kind == "scene1":
        return np.hstack([c1, q1, T])
    if kind == "scene2":
        return np.hstack([c2, q2, T])
    if kind == "nocount":
        return np.hstack([T, (b["qa"] == b["qb"])[:, None].astype(float)])
    if kind == "both":
        return np.hstack([c1, c2, q1, q2, T])
    raise ValueError(kind)


def probe(cfg: SceneCfg, kind, n_train=40000, n_test=10000, epochs=60, seed=0, beta=0.0, types="train"):
    rng = np.random.default_rng(seed)
    tr = make_batch(cfg, rng, n_train, beta=beta, types=types)
    te = make_batch(cfg, rng, n_test, beta=beta, types=types)
    xtr, xte = features(tr, kind), features(te, kind)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    xtr, xte = torch.tensor(xtr, dtype=torch.float32), torch.tensor(xte, dtype=torch.float32)
    ytr = torch.tensor(tr["answer"], dtype=torch.float32)
    yte = torch.tensor(te["answer"])
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(xtr.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    for ep in range(epochs):
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 512):
            idx = perm[i:i + 512]
            loss = F.binary_cross_entropy_with_logits(net(xtr[idx]).squeeze(-1), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        acc = ((net(xte).squeeze(-1) > 0) == yte).float().mean().item()
    return acc
