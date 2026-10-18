"""B④ 실험용 곡선 데이터: 촘촘한 격자에 미리 계산해 둔 높이와 정답 넓이.

학습 분포: 상수(직사각형) / 1차(사다리꼴) / 다항식 / 삼각·지수 / 넓이 공식 없는 곡선(가우스형 군은 제외).
시험 전용: 계단, 뾰족, 긴 구간, 극단 크기, 그리고 제외한 가우스형 군. 높이는 모두 양수.
"""
from __future__ import annotations

import numpy as np

from .curves import Curve, TEST_KINDS, integrate_gl, make_curve, make_test_curve

G = 513  # 격자 점 수 (구간을 512등분)


def grid_of(f_np, a: float, b: float) -> np.ndarray:
    return f_np(np.linspace(a, b, G))


def _simple(kind: str, rng):
    a = round(float(rng.uniform(-5, 5)), 2)
    b = round(a + float(rng.uniform(0.5, 10)), 2)
    c = 10 ** float(rng.uniform(-1, 2))
    if kind == "const":
        f = lambda x, c=c: np.full_like(x, c, dtype=float)
    else:  # linear: 양수 유지
        m = float(rng.uniform(0.05, 0.9))
        s = float(rng.choice([-1, 1]))
        lo, hi = c * m, c * (1 + m)
        y0, y1 = (lo, hi) if s > 0 else (hi, lo)
        f = lambda x, a=a, b=b, y0=y0, y1=y1: y0 + (y1 - y0) * (x - a) / (b - a)
    return Curve(kind, kind, a, b, integrate_gl(f, a, b), None, f)


def is_gauss(text: str) -> bool:
    return "exp(-" in text and "x^2" in text


MIX = [("const", .10), ("linear", .10), ("poly", .25), ("trig", .25), ("nonclosed", .30)]


def sample_train(rng) -> Curve:
    names, ps = zip(*MIX)
    while True:
        k = str(rng.choice(names, p=ps))
        c = _simple(k, rng) if k in ("const", "linear") else make_curve(k, rng)
        if not is_gauss(c.text):
            return c


def sample_gauss_holdout(rng) -> Curve:
    while True:
        c = make_curve("nonclosed", rng)
        if is_gauss(c.text):
            return c


def pack(curves) -> dict:
    return dict(
        a=np.array([c.a for c in curves], dtype=np.float32),
        b=np.array([c.b for c in curves], dtype=np.float32),
        area=np.array([c.area for c in curves], dtype=np.float64),
        grid=np.stack([grid_of(c.f_np, c.a, c.b) for c in curves]).astype(np.float32),
        kind=np.array([c.kind for c in curves]),
    )


def make_sets(n_train=60000, n_val=2000, n_ood=500, seed=0):
    rng = np.random.default_rng(seed)
    out = {"train": pack([sample_train(rng) for _ in range(n_train)]),
           "val": pack([sample_train(np.random.default_rng([seed, 1, i])) for i in range(n_val)])}
    for k in TEST_KINDS:
        out[k] = pack([make_test_curve(k, np.random.default_rng([seed, 2, i])) for i in range(n_ood)])
    out["gauss_holdout"] = pack([sample_gauss_holdout(np.random.default_rng([seed, 3, i])) for i in range(n_ood)])
    return out
