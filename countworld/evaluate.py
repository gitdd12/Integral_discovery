"""평가: E1 정확도, E3 불변성, E4 n 대 n+1, E5 코드 분석 (DESIGN3.md §6)."""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from .scene import SceneCfg, make_batch
from .train import to_t


@torch.no_grad()
def judge_acc(agent, cfg, rng, n=4096, eps=0.0, min_len=0, **kw):
    b = to_t(make_batch(cfg, rng, n, **kw), agent.dev)
    sp = agent.speaker(b["s1"], b["l1"], mode="greedy", eps=eps, min_len=min_len)
    m = agent.listen(sp["syms"], b["s2"], b["l2"])
    return ((m > 0) == b["answer"]).float().mean().item()


@torch.no_grad()
def acc_adjacent(agent, cfg, rng, n=512, eps=0.0, min_len=0, types="train"):
    """n 대 n+1 (양쪽 순서) 정확도를 n 별로."""
    out = {}
    for k in range(cfg.max_count):
        a = np.array([k, k + 1] * (n // 2))
        bb = np.array([k + 1, k] * (n // 2))
        b = to_t(make_batch(cfg, rng, n, a=a, b=bb, types=types), agent.dev)
        sp = agent.speaker(b["s1"], b["l1"], mode="greedy", eps=eps, min_len=min_len)
        m = agent.listen(sp["syms"], b["s2"], b["l2"])
        out[k] = ((m > 0) == b["answer"]).float().mean().item()
    return out


@torch.no_grad()
def code_table(agent, cfg, rng, per=200, eps=0.0, min_len=0, types="train"):
    """개수 a 마다 탐욕 메시지를 모아 일관성·단사성·형태를 잰다."""
    msgs = {}
    for k in range(cfg.max_count + 1):
        a = np.full(per, k)
        b = to_t(make_batch(cfg, rng, per, a=a, b=a, types=types), agent.dev)
        sp = agent.speaker(b["s1"], b["l1"], mode="greedy", eps=eps, min_len=min_len)
        V = agent.V
        msgs[k] = [tuple(int(x) for x in row if x < V) for row in sp["syms"].cpu().numpy()]
    modal = {k: Counter(v).most_common(1)[0][0] for k, v in msgs.items()}
    consistency = float(np.mean([Counter(v).most_common(1)[0][1] / len(v) for v in msgs.values()]))
    ks = sorted(modal)
    pairs = [(i, j) for i in ks for j in ks if i < j]
    injective = float(np.mean([modal[i] != modal[j] for i, j in pairs])) if pairs else 1.0
    lens = np.array([len(modal[k]) for k in ks], float)
    corr = float(np.corrcoef(ks, lens)[0, 1]) if lens.std() > 0 else 0.0
    return dict(consistency=consistency, injective=injective, len_corr=corr, modal={k: modal[k] for k in ks})


def full_report(agent, cfg, rng, eps=0.0, min_len=0):
    rep = dict(
        e1_heldout_pairs=judge_acc(agent, cfg, rng, eps=eps, min_len=min_len, pairs="heldout"),
        e1_train=judge_acc(agent, cfg, rng, eps=eps, min_len=min_len),
        e3_new_types=judge_acc(agent, cfg, rng, eps=eps, min_len=min_len, types="test"),
        e4_adjacent=acc_adjacent(agent, cfg, rng, eps=eps, min_len=min_len),
        e5=code_table(agent, cfg, rng, eps=eps, min_len=min_len),
    )
    return rep
