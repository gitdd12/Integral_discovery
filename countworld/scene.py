"""장면 생성기 (DESIGN3.md §3, §4).

한 판 = (장면 1, 장면 2, 질의 종류 둘, 정답 같다/다르다). 환경으로 지름길을 막는다:
 - 정답을 먼저 정하고, (개수 a, 개수 b)는 a 의 주변분포와 b 의 주변분포가 정답과 무관(균일)하도록 뽑는다(Sinkhorn).
 - 두 장면의 전체 길이 T 는 늘 같고, T 는 정답·개수와 독립으로 뽑는다.
 - 질의 종류는 매 장면 독립으로 무작위 (종류에 고정된 의미가 없다).
 - 장면 안의 순서는 매번 섞는다.
토큰: 0 PAD, 1 BOS, 2 SEP, 3 QRY, 4.. 물체 종류.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PAD, BOS, SEP, QRY = 0, 1, 2, 3
N_TYPES = 12
T0 = 4
VOCAB = T0 + N_TYPES
TRAIN_TYPES = np.arange(8)
TEST_TYPES = np.arange(8, 12)


@dataclass
class SceneCfg:
    max_count: int = 9          # 개수는 0..max_count
    extra: int = 6              # 전체 길이 T = max_count + U[0, extra]
    omit: tuple = ()            # 학습에서 뺀 개수 (시험에서만 나옴)

    @property
    def max_len(self) -> int:
        return self.max_count + self.extra + 4


def heldout_matrix(n: int, frac: float = 0.1):
    """학습에서 뺄 (개수, 개수) 쌍 (대칭, 고정된 무작위)."""
    u = np.random.default_rng(12345).random((n, n))
    u = np.triu(u) + np.triu(u, 1).T
    return u < frac


_PM_CACHE: dict = {}


def pair_matrix(n: int, beta: float, mode: str, omit: tuple = (), weight=None):
    """서로 다른 (a,b) 쌍의 결합확률. mode: train(학습 쌍만) / heldout(빼 둔 쌍·뺀 개수가 낀 쌍) / all."""
    key = (n, round(beta, 2), mode, omit)
    if weight is None and key in _PM_CACHE:
        return _PM_CACHE[key]
    idx = np.arange(n)
    g = np.abs(idx[:, None] - idx[None, :]).astype(float)
    W = np.where(g > 0, np.maximum(g, 1.0) ** (-beta), 0.0)
    ho = heldout_matrix(n)
    om = np.zeros((n, n), bool)
    for o in omit:
        om[o, :] = True
        om[:, o] = True
    if weight is not None:
        W = W * weight
    if mode == "train":
        W[ho | om] = 0.0
    elif mode == "heldout":
        W[~(ho | om)] = 0.0
    P = W / W.sum()
    if mode == "train":                      # 주변분포를 균일하게 (정답과 무관하도록)
        for _ in range(1000):
            r = P.sum(1, keepdims=True)
            P = np.where(r > 0, P / np.maximum(r, 1e-12) / n, 0.0)
            c = P.sum(0, keepdims=True)
            P = np.where(c > 0, P / np.maximum(c, 1e-12) / n, 0.0)
        P = P / P.sum()
    if weight is None:
        _PM_CACHE[key] = P
    return P


def _scene(rng, counts, T, q, pool, max_len):
    """counts,T,q: (B,) → 토큰 (B, max_len), 길이 (B,)."""
    B = len(counts)
    Tmax = int(T.max())
    P = len(pool)
    pos_q = np.searchsorted(pool, q)
    j = rng.integers(0, P - 1, size=(B, Tmax))
    idx = j + (j >= pos_q[:, None])
    typ = pool[idx]
    col = np.arange(Tmax)[None, :]
    typ = np.where(col < counts[:, None], q[:, None], typ)
    keys = rng.random((B, Tmax))
    keys = np.where(col < T[:, None], keys, 2.0)       # 패딩은 뒤로
    order = np.argsort(keys, axis=1)
    typ = np.take_along_axis(typ, order, axis=1)
    out = np.zeros((B, max_len), dtype=np.int64)
    out[:, 0] = BOS
    out[:, 1:1 + Tmax] = np.where(col < T[:, None], T0 + typ, PAD)
    r = np.arange(B)
    out[r, T + 1] = SEP
    out[r, T + 2] = QRY
    out[r, T + 3] = T0 + q
    return out, T + 4


def make_batch(cfg: SceneCfg, rng, B: int, beta: float = 0.0, types: str = "train",
               pairs: str = "train", a=None, b=None, answer=None, pair_weight=None):
    """pairs: train / heldout. a,b 를 주면 그 개수로 (정답은 a==b)."""
    n = cfg.max_count + 1
    pool = TRAIN_TYPES if types == "train" else TEST_TYPES
    if a is None:
        ans = rng.random(B) < 0.5 if answer is None else np.asarray(answer, bool)
        P = pair_matrix(n, beta, pairs, cfg.omit, pair_weight)
        flat = rng.choice(n * n, size=B, p=P.ravel())
        ad, bd = flat // n, flat % n
        valid = [c for c in range(n) if c not in cfg.omit] if pairs == "train" else list(range(n))
        sa = rng.choice(valid, size=B)
        a = np.where(ans, sa, ad)
        b = np.where(ans, sa, bd)
    else:
        a, b = np.asarray(a), np.asarray(b)
        ans = a == b
    T = cfg.max_count + rng.integers(0, cfg.extra + 1, size=B)
    qa = pool[rng.integers(0, len(pool), size=B)]
    qb = pool[rng.integers(0, len(pool), size=B)]
    s1, l1 = _scene(rng, a, T, qa, pool, cfg.max_len)
    s2, l2 = _scene(rng, b, T, qb, pool, cfg.max_len)
    return dict(s1=s1, l1=l1, s2=s2, l2=l2, answer=ans, a=a, b=b, T=T, qa=qa, qb=qb)
