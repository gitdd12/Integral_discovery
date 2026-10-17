"""2단계 학습: 잘 된 풀이만 모아서 따라하기 (PLAN.md 5.2).

한 바퀴:
  1) 곡선 n_curves 개  2) 곡선마다 k 번 풀기(온도 1.0)  3) 보상으로 채점
  4) 곡선마다 상위 keep 개 (너무 비슷한 풀이는 하나만)  5) 남긴 풀이 + 기초 학교 20% 로 따라하기 학습
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from . import school
from . import vocab as V
from .curves import random_curve, siblings
from .generate import run_notebooks_cached
from .notebook import new_episode


@dataclass
class RLConfig:
    n_curves: int = 16
    k: int = 8
    keep: int = 2
    temperature: float = 1.0
    lr: float = 1e-4
    school_frac: float = 0.2
    max_len: int = 1024
    min_keep_score: float = 1e-9     # 0점(문법 오류·실행 불가 등)은 따라하지 않는다
    dup_sim: float = 0.9             # 마지막 제출 글자 유사도가 이 이상이면 같은 풀이로 본다


def _sig(nb) -> str:
    """풀이가 같은 계열인지 보는 지문: 마지막 제출의 토큰열 (숫자 자릿수는 뭉갠다)."""
    if not nb.submissions:
        return ""
    return "".join("0" if t.isdigit() else t for t in nb.submissions[-1][2])


def _similar(a: str, b: str, th: float) -> bool:
    if a == b:
        return True
    n = max(len(a), len(b))
    if n == 0:
        return True
    # 앞쪽 일치 길이로 대충 비교 (길이 차이가 크면 다른 풀이)
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return common / n >= th


def select(nbs, scores, cfg: RLConfig):
    """곡선 하나의 시도들(nbs, scores) 중 상위 keep 개, 비슷한 건 하나만."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    kept, sigs = [], []
    for i in order:
        if scores[i] < cfg.min_keep_score:
            break
        s = _sig(nbs[i])
        if any(_similar(s, t, cfg.dup_sim) for t in sigs):
            continue
        kept.append(i)
        sigs.append(s)
        if len(kept) >= cfg.keep:
            break
    return kept


def collect(model, rng, cfg: RLConfig, device, gen=None, store=None):
    """한 바퀴의 풀이들을 모은다. (notebooks, scores, curves)"""
    curves = [random_curve(rng) for _ in range(cfg.n_curves)]
    seeds = [int(rng.integers(1 << 30)) for _ in curves]
    nbs, groups = [], []
    for c, sd in zip(curves, seeds):
        hidden = siblings(c.kind, sd)         # 같은 곡선의 시도들이 같은 시험 곡선을 쓴다
        for _ in range(cfg.k):
            nbs.append(new_episode(c, sd, store=store, max_len=cfg.max_len))
            nbs[-1]._hidden = hidden
        groups.append(list(range(len(nbs) - cfg.k, len(nbs))))
    run_notebooks_cached(model, nbs, temperature=cfg.temperature, device=device, gen=gen)
    scores = [nb.final_score() for nb in nbs]
    return nbs, scores, curves, groups


def sft_step(model, opt, seqs, masks, device):
    """seqs: 토큰열 목록, masks: 같은 길이의 0/1 (1 = 모델이 쓴 토큰). 한 번 따라하기."""
    model.train()
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), V.PAD_ID, dtype=torch.long)
    m = torch.zeros(len(seqs), L, dtype=torch.bool)
    for r, (s, mk) in enumerate(zip(seqs, masks)):
        x[r, : len(s)] = torch.tensor(s)
        m[r, : len(mk)] = torch.tensor(mk, dtype=torch.bool)
    x, m = x.to(device), m.to(device)
    logits = model(x[:, :-1])
    ls = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="none")
    w = m[:, 1:].reshape(-1).float()
    loss = (ls * w).sum() / w.sum().clamp(min=1)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.detach())


def school_batch(rng, n):
    seqs, masks = [], []
    for _ in range(n):
        ids, mask = school.render_example(school.sample_train(rng))
        seqs.append(ids)
        masks.append(mask)
    return seqs, masks


def round_stats(nbs, scores):
    sc = np.asarray(scores)
    return dict(
        mean=float(sc.mean()), max=float(sc.max()), valid=float((sc > 0).mean()),
        submits=float(np.mean([len(nb.submissions) for nb in nbs])),
        length=float(np.mean([len(nb.ids) for nb in nbs])),
        loops=float(np.mean([any(t == "반복" for s in nb.submissions for t in s[2]) for nb in nbs])),
    )


def thought_sample(nbs, scores, curves, groups, n=3):
    """생각 기록 샘플: 곡선 몇 개의 최고점 풀이 전체."""
    out = []
    for g in groups[:n]:
        best = max(g, key=lambda i: scores[i])
        out.append(dict(score=scores[best], area=None, text=V.decode(nbs[best].ids, nbs[best].mask)))
    return out
