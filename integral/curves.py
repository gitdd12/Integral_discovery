"""곡선 생성기와 정답 넓이.

학습용 3종류: poly(다항식 40%), trig(삼각·지수 30%), nonclosed(깔끔한 넓이 공식이 없는 곡선 30%).
시험 전용(학습에 절대 등장 금지): test_stair, test_spike, test_long, test_extreme.

곡선은 항상 "글자 수식"을 먼저 만들고, 그 글자를 다시 파싱해서 쓴다.
→ 모델이 보는 수식과 실제 계산이 어긋날 수 없다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .expr import EvalError, SCALAR, compile_ast, fmt_num, parse_expr, _numpy_lib
from .vocab import tokenize

TRAIN_KINDS = ("poly", "trig", "nonclosed")
TRAIN_WEIGHTS = (0.4, 0.3, 0.3)
TEST_KINDS = ("test_stair", "test_spike", "test_long", "test_extreme")

_NP = _numpy_lib()
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)


@dataclass
class Curve:
    kind: str
    text: str                       # f(x)= 오른쪽의 수식 글자
    a: float
    b: float
    area: float
    f: Callable[[float], float] = field(repr=False, default=None)      # 스칼라 (제출 절차용)
    f_np: Callable = field(repr=False, default=None)                    # 배열 (정답 계산용)

    @property
    def family(self) -> str:
        return self.kind

    def prompt_tokens(self) -> list[str]:
        """문제 문장(토큰 문자열). '문제:' 로 시작, '넓이? 제출?' 로 끝."""
        return (
            ["문제:", "곡선:", "f(", "x", ")", "="] + tokenize(self.text)
            + ["구간:", "a", "="] + tokenize(fmt_num(self.a))
            + ["b", "="] + tokenize(fmt_num(self.b))
            + ["넓이?", "제출?", "\n"]
        )


def _compile_curve(text: str):
    ast = parse_expr(tokenize(text))
    fs = compile_ast(ast, SCALAR)
    fn = compile_ast(ast, _NP)

    def f(x):
        y = fs({"x": x}, {})
        if not math.isfinite(y):
            raise EvalError("not finite")
        return y

    def f_np(xs):
        with np.errstate(all="ignore"):
            return fn({"x": xs}, {})

    return f, f_np


def integrate_gl(f_np, a: float, b: float) -> float:
    """합성 가우스–르장드르 구적법 (판마다 8점). 정답 넓이용."""
    panels = int(min(max((b - a) * 150, 300), 3000))
    edges = np.linspace(a, b, panels + 1)
    mid = (edges[:-1] + edges[1:]) / 2
    half = (edges[1:] - edges[:-1]) / 2
    xs = mid[:, None] + half[:, None] * _GL_X[None, :]
    ys = f_np(xs)
    return float(np.sum(half[:, None] * _GL_W[None, :] * ys))


def _finish(kind, text, a, b, positive=True):
    f, f_np = _compile_curve(text)
    grid = np.linspace(a, b, 801)
    ys = f_np(grid)
    if not np.all(np.isfinite(ys)):
        return None
    if positive and not np.all(ys > 0):
        return None
    area = integrate_gl(f_np, a, b)
    if not math.isfinite(area) or area <= 0:
        return None
    return Curve(kind, text, a, b, area, f, f_np)


def _terms_text(terms, const):
    """(계수, 인자글자) 목록 + 상수항 → 'c1×인자+c2×인자-c0' 형태의 글자."""
    out = ""
    for coef, fac in terms:
        c = fmt_num(abs(coef))
        sign = "-" if coef < 0 else "+"
        piece = f"{c}×{fac}"
        out += (("-" if sign == "-" else "") if not out else sign) + piece
    c = fmt_num(abs(const))
    if not out:
        return ("-" if const < 0 else "") + c
    return out + ("-" if const < 0 else "+") + c


def _eval_factor(fac, grid):
    _, f_np = _compile_curve(fac)
    return f_np(grid)


def _rnd(v, nd=2):
    return round(float(v), nd)


def _terms_poly(rng, a, b):
    k = rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25])
    pows = sorted(rng.choice([1, 2, 3], size=k, replace=False))
    facs = {1: "x", 2: "x^2", 3: "x^3"}
    return [(float(rng.choice([-1, 1]) * rng.uniform(0.2, 3.0)), facs[p]) for p in pows]


def _terms_trig(rng, a, b):
    M = max(abs(a), abs(b), 1.0)
    def sinus(name):
        k = _rnd(rng.uniform(0.3, 3.0))
        p = _rnd(rng.uniform(0, 6.28))
        return f"{name}({fmt_num(k)}×x+{fmt_num(p)})"
    def expo():
        k = max(_rnd(rng.uniform(0.2, 1.0) * 6.0 / M), 0.01) * rng.choice([-1, 1])
        sgn = "-" if k < 0 else ""
        return f"exp({sgn}{fmt_num(abs(k))}×x)"
    pick = rng.choice(["sin", "cos", "exp", "sin+cos", "sin+exp", "cos+exp"])
    gens = {"sin": lambda: sinus("sin"), "cos": lambda: sinus("cos"), "exp": expo}
    return [(float(rng.choice([-1, 1]) * rng.uniform(0.2, 3.0)), gens[p]()) for p in pick.split("+")]


def _terms_nonclosed(rng, a, b):
    M = max(abs(a), abs(b), 1.0)
    def kfor(n):
        kmax = min(20.0 / (n * M ** (n - 1)), 1.5)
        return max(_rnd(rng.uniform(0.2, 1.0) * kmax), 0.01)
    t = rng.choice(["sin2", "cos2", "gauss", "sin3", "esin", "ecos"])
    if t == "sin2":
        fac = f"sin({fmt_num(kfor(2))}×x^2)"
    elif t == "cos2":
        fac = f"cos({fmt_num(kfor(2))}×x^2)"
    elif t == "gauss":
        fac = f"exp(-{fmt_num(_rnd(rng.uniform(0.05, 1.0)))}×x^2)"
    elif t == "sin3":
        fac = f"sin({fmt_num(kfor(3))}×x^3)"
    elif t == "esin":
        fac = f"exp(sin({fmt_num(_rnd(rng.uniform(0.3, 3.0)))}×x))"
    else:
        fac = f"exp(cos({fmt_num(_rnd(rng.uniform(0.3, 3.0)))}×x))"
    return [(float(rng.choice([-1, 1]) * rng.uniform(0.2, 3.0)), fac)]


_TERMS = {"poly": _terms_poly, "trig": _terms_trig, "nonclosed": _terms_nonclosed}


def _interval(rng, lo=-5.0, hi=5.0, lmin=0.5, lmax=10.0):
    a = _rnd(rng.uniform(lo, hi))
    b = _rnd(a + rng.uniform(lmin, lmax))
    return a, b


def make_curve(kind: str, rng: np.random.Generator) -> Curve:
    """학습용 곡선. 높이 크기는 로그 균등으로 0.1배~100배."""
    terms_fn = _TERMS[kind]
    for _ in range(200):
        a, b = _interval(rng)
        if b - a < 0.5:
            continue
        terms = terms_fn(rng, a, b)
        grid = np.linspace(a, b, 801)
        with np.errstate(all="ignore"):
            g = sum(c * _eval_factor(fac, grid) for c, fac in terms)
        if not np.all(np.isfinite(g)):
            continue
        lo, hi = float(g.min()), float(g.max())
        if hi - lo <= 0:
            continue
        m = rng.uniform(0.05, 0.5)
        # 모양의 변동폭(hi-lo)으로 나눠서, 높이가 늘 s × (m ~ 1+m) 범위에 오게 한다
        # (s = 0.1~100배, 로그 균등). 높이로 곡선 종류를 알아채는 지름길 제거.
        s = 10 ** rng.uniform(-1, 2) / (hi - lo)
        const = (-lo + m * (hi - lo)) * s
        if max(abs(c * s) for c, _ in terms) > 1e8 or const > 1e8:
            continue
        text = _terms_text([(c * s, fac) for c, fac in terms], const)
        cur = _finish(kind, text, a, b)
        if cur is not None:
            return cur
    raise RuntimeError(f"곡선을 못 만들었다: {kind}")


def random_curve(rng: np.random.Generator) -> Curve:
    kind = TRAIN_KINDS[rng.choice(len(TRAIN_KINDS), p=TRAIN_WEIGHTS)]
    return make_curve(kind, rng)


def siblings(kind: str, seed, k: int = 3) -> list[Curve]:
    """모델이 못 본 같은 종류의 곡선 k개 (에피소드 seed 로 재현 가능)."""
    return [make_curve(kind, np.random.default_rng([int(seed) & 0x7FFFFFFF, 7919, j])) for j in range(k)]


# ------------------------------------------------------------ 시험 전용 곡선
def make_test_curve(kind: str, rng: np.random.Generator) -> Curve:
    if kind == "test_stair":
        # 계단: 가파른 S자 곡선(폭 1/60)을 여러 개 더한 것
        for _ in range(200):
            a, b = _interval(rng, -5, 5, 3, 8)
            n = int(rng.integers(3, 7))
            ts = sorted(_rnd(rng.uniform(a, b)) for _ in range(n))
            base = float(rng.uniform(1, 5))
            levels = [base]
            for _ in range(n):
                levels.append(max(0.3, levels[-1] + float(rng.uniform(-2, 2))))
            text = fmt_num(levels[0])
            for j, t in enumerate(ts):
                d = levels[j + 1] - levels[j]
                if abs(d) < 1e-3:
                    continue
                sgn = "-" if d < 0 else "+"
                shift = f"x-{fmt_num(t)}" if t >= 0 else f"x+{fmt_num(-t)}"
                text += f"{sgn}{fmt_num(abs(d))}÷(1+exp(-60×({shift})))"
            cur = _finish(kind, text, a, b)
            if cur is not None:
                return cur
    elif kind == "test_spike":
        # 뾰족한 곡선: 폭 0.05~0.3 의 좁은 봉우리 하나 + 바닥
        for _ in range(200):
            a, b = _interval(rng, -5, 5, 2, 6)
            t = _rnd(rng.uniform(a + 0.3, b - 0.3))
            k = _rnd(rng.uniform(30, 300), 1)
            h = _rnd(rng.uniform(2, 20), 1)
            c = _rnd(rng.uniform(0.2, 1.5), 2)
            shift = f"x-{fmt_num(t)}" if t >= 0 else f"x+{fmt_num(-t)}"
            cur = _finish(kind, f"{fmt_num(h)}×exp(-{fmt_num(k)}×({shift})^2)+{fmt_num(c)}", a, b)
            if cur is not None:
                return cur
    elif kind == "test_long":
        # 아주 긴 구간 (길이 30~60)
        for _ in range(200):
            a = _rnd(rng.uniform(-40, 10))
            b = _rnd(a + rng.uniform(30, 60))
            if abs(a) >= 100 or abs(b) >= 100:
                continue
            terms = [(float(rng.uniform(0.5, 2)), f"sin({fmt_num(_rnd(rng.uniform(0.2, 1.5)))}×x)")]
            grid = np.linspace(a, b, 801)
            g = sum(c * _eval_factor(fac, grid) for c, fac in terms)
            const = -float(g.min()) + float(rng.uniform(0.1, 0.5)) * float(g.max() - g.min())
            cur = _finish(kind, _terms_text(terms, const), a, b)
            if cur is not None:
                return cur
    elif kind == "test_extreme":
        # 극단적 크기: 학습 범위(0.1~100배) 밖
        for _ in range(200):
            base = make_curve(str(rng.choice(TRAIN_KINDS)), rng)
            s = 10 ** float(rng.choice([rng.uniform(-4, -2.5), rng.uniform(3.5, 5)]))
            text = _scale_text(base, s)
            cur = _finish(kind, text, base.a, base.b)
            if cur is not None:
                return cur
    else:
        raise ValueError(kind)
    raise RuntimeError(f"시험 곡선을 못 만들었다: {kind}")


def _scale_text(base: Curve, s: float) -> str:
    return f"{fmt_num(s)}×({base.text})"
