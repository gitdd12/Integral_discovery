"""메모장과 모델을 이어 붙여 글을 쓰게 한다 (배치). 계산기가 채우는 토큰도 처리한다.

run_notebooks        : 매번 전체를 다시 계산 (느리지만 단순 — 기준 구현, 시험에서 대조용)
run_notebooks_cached : KV 캐시. 한 번에 한 줄이 토큰 하나를 처리한다. 프롬프트·계산기 토큰도
                       같은 방식으로 한 칸씩 읽고, 읽을 게 없을 때만 다음 토큰을 뽑는다.
"""
from __future__ import annotations

import torch

from . import vocab as V

_ALLOWED = {}


def allowed_mask(device):
    key = str(device)
    if key not in _ALLOWED:
        _ALLOWED[key] = torch.tensor(V.model_allowed_mask(), device=device)
    return _ALLOWED[key]


def _mask_logits(logits, rows_nb, allowed):
    """허용 안 된 토큰 + (최소 제출 횟수를 못 채운 줄의) <eos> 를 막는다."""
    m = allowed.unsqueeze(0).repeat(len(rows_nb), 1)
    for r, nb in enumerate(rows_nb):
        if not nb.eos_ok:
            m[r, V.EOS_ID] = False
    return logits.masked_fill(~m, float("-inf"))


def _pick(logits, temperature, gen):
    if temperature > 0:
        return torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1, generator=gen).squeeze(1)
    return logits.argmax(-1)


@torch.no_grad()
def run_notebooks(model, notebooks, max_new=400, temperature=0.0, device="cpu", gen=None):
    model.eval()
    allowed = allowed_mask(device)
    active = [i for i, nb in enumerate(notebooks) if not nb.done]
    steps = 0
    while active and steps < max_new:
        steps += 1
        L = max(len(notebooks[i].ids) for i in active)
        batch = torch.full((len(active), L), V.PAD_ID, dtype=torch.long)
        last = []
        for r, i in enumerate(active):
            ids = notebooks[i].ids
            batch[r, : len(ids)] = torch.tensor(ids)
            last.append(len(ids) - 1)
        logits = model(batch.to(device))[torch.arange(len(active)), torch.tensor(last)]
        nxt = _pick(_mask_logits(logits, [notebooks[i] for i in active], allowed), temperature, gen)
        for r, i in enumerate(active):
            notebooks[i].feed(int(nxt[r]))
        active = [i for i in active if not notebooks[i].done]
    for nb in notebooks:
        nb.done = True
    return notebooks


@torch.no_grad()
def run_notebooks_cached(model, notebooks, temperature=0.0, device="cpu", gen=None):
    """모든 notebook 이 끝날 때까지 (각자의 max_len 까지). 끝난 줄은 건너뛴다."""
    model.eval()
    allowed = allowed_mask(device)
    B = len(notebooks)
    T = max(nb.max_len for nb in notebooks) + 1
    cache = model.init_cache(B, T, device)
    ptr = [0] * B
    alive = [not nb.done for nb in notebooks]
    ar = torch.arange(B)
    while any(alive):
        rows = [i for i in range(B) if alive[i]]
        tok = torch.full((B,), V.PAD_ID, dtype=torch.long)
        pos = torch.zeros(B, dtype=torch.long)
        for i in rows:
            tok[i] = notebooks[i].ids[ptr[i]]
            pos[i] = ptr[i]
        logits = model.step(tok.to(device), pos.to(device), cache)
        need = []
        for i in rows:
            ptr[i] += 1
            if ptr[i] == len(notebooks[i].ids):
                need.append(i)
        if need:
            sel = _mask_logits(logits[torch.tensor(need)], [notebooks[i] for i in need], allowed)
            nxt = _pick(sel, temperature, gen)
            for r, i in enumerate(need):
                notebooks[i].feed(int(nxt[r]))
                if notebooks[i].done:
                    alive[i] = False
    for nb in notebooks:
        nb.done = True
    return notebooks
