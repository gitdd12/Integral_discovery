"""기초 학교: 도구 쓰는 법만 가르치는 연습문제 자동 생성.

원칙 (PLAN.md 2장):
- 곡선 넓이 풀이 예시 없음. 계단 모양 도형 없음(같은 폭 직사각형 3개 이상 금지).
- 위치를 옮겨가며 함수 값을 재는 문제 없음 (f 는 한 점 x=값 에서만 쓴다).
- 반복은 넓이와 무관한 문제(정수 더하기)에만 쓴다.
- 생각 내용은 점수 주지 않는다. 여기서는 '도구를 부르는 형식'만 보여준다.

예제 하나 = 문제 문장(prompt) + 모델이 쓸 글(response, 계산기 결과는 빠져 있음).
실제로 Notebook 에 먹여서 계산기 칸을 채우고 마스크를 만든다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import vocab as V
from .expr import SCALAR, compile_ast, fmt_num, parse_expr
from .notebook import Notebook
from .vocab import tokenize

# ------------------------------------------------------------------ 예제
@dataclass
class Example:
    category: str
    prompt: list[str]
    response: str
    mode: str                       # 'calc' (마지막 계산기 결과가 정답) | 'submit' (제출 오차 0)
    target: float | bool | None
    x: float | None = None
    f: Callable | None = None
    interval: tuple | None = None
    min_submits: int = 0            # >0 이면 이만큼 제출하기 전엔 코드가 다음 `제출[` 를 써 준다


def build(ex: Example, response: str | None = None) -> Notebook:
    nb = Notebook(ex.prompt, target=ex.target if ex.mode == "submit" else None,
                  x=ex.x, f=ex.f, interval=ex.interval, max_len=512, min_submits=ex.min_submits)
    if response is not None:
        nb.feed_text(response)
    return nb


def render_example(ex: Example) -> tuple[list[int], list[int]]:
    """정답 풀이를 메모장에 먹여 (ids, mask) 를 얻는다. 검증도 같이 한다."""
    nb = build(ex)
    nb.prompt_ids = list(nb.ids)
    nb.feed_text(ex.response)
    if not nb.done:
        nb.feed(V.EOS_ID)
    assert nb.violations == 0
    assert check(nb, ex), (ex.category, nb.render())
    return nb.ids, nb.mask


def check(nb: Notebook, ex: Example) -> bool:
    """메모장 결과가 정답인가 (모델 시험·생성기 검증 둘 다 이걸 쓴다)."""
    if ex.mode == "calc":
        v = nb.last_calc
        if v is None:
            return False
        if isinstance(ex.target, bool):
            return isinstance(v, bool) and v == ex.target
        if isinstance(v, bool):
            return False
        return abs(v - ex.target) <= 2e-3 * abs(ex.target) + 1e-9
    return nb.last_feedback is not None and nb.last_feedback < 1e-9


# ------------------------------------------------------------------ 숫자 뽑기
def exact4(v: float) -> bool:
    """계산기가 4자리로 보여줘도 값이 그대로인가."""
    return math.isclose(float(fmt_num(v)), v, rel_tol=1e-9, abs_tol=1e-12)


def rnum(rng, hi: int, dec: float = 0.2, lo: int = 1) -> float:
    if rng.random() < dec:
        return round(float(rng.integers(lo * 10, hi * 10 + 1)) / 10, 1)
    return float(rng.integers(lo, hi + 1))


def rdig(rng, maxd: int, mind: int = 1, dec: float = 0.0) -> float:
    """자릿수를 mind..maxd 에서 고르게 뽑은 수 (큰 수도 자주 나오게)."""
    d = int(rng.integers(mind, maxd + 1))
    v = float(rng.integers(1 if d == 1 else 10 ** (d - 1), 10 ** d))
    if dec and d <= 2 and rng.random() < dec:
        v += int(rng.integers(1, 10)) / 10
    return round(v, 1)


def ntext(v: float) -> str:
    return fmt_num(v)


def pslot(rng, used=()):
    while True:
        s = f"p{int(rng.integers(0, 32))}"
        if s not in used:
            return s


def pslots(rng, k):
    """서로 다른 이름 k 개. 연속 번호로 주어 이름이 겹치는 실수를 줄인다."""
    base = int(rng.integers(0, 32 - k + 1))
    return [f"p{base + j}" for j in range(k)]


# ------------------------------------------------------------------ 템플릿
def t_rect(rng, maxd=4, mind=1, cat="rect") -> Example:
    w, h = rdig(rng, maxd, mind, .2), rdig(rng, maxd, 1, .2)
    if rng.random() < .5:
        w, h = h, w
    p = ["<bos>", "문제:", "직사각형", "가로"] + tokenize(ntext(w)) + ["세로"] + tokenize(ntext(h)) + ["넓이?", "\n"]
    return Example(cat, p, f"rect({ntext(w)},{ntext(h)})=\n", "calc", w * h)


def _rand_expr_x(rng, family: str) -> str:
    c = lambda lo=1, hi=9: ntext(rnum(rng, hi, 0.25, lo))
    sgn = lambda: "+" if rng.random() < 0.5 else "-"
    if family == "lin":
        return f"{c()}×x{sgn()}{c()}"
    if family == "quad":
        return f"{c()}×x^2{sgn()}{c()}×x{sgn()}{c()}"
    if family == "cubic":
        return f"x^3{sgn()}{c()}×x^2{sgn()}{c()}"
    if family == "trig":
        fn = rng.choice(["sin", "cos"])
        return f"{c()}×{fn}({c(1,4)}×x){sgn()}{c()}"
    if family == "exp":
        return f"{c()}×exp({'-' if rng.random() < .5 else ''}0.{int(rng.integers(1, 9))}×x){sgn()}{c()}"
    if family == "sum":
        return f"{_rand_expr_x(rng, rng.choice(['lin', 'quad']))}{sgn()}{c()}×sin(x)"
    # ---- 학습에 없는 모양 (시험 전용)
    if family == "nest":
        return rng.choice([f"sin(x^2)×{c()}", f"exp(-x^2)+{c()}", f"cos({c()}×x^2){sgn()}{c()}"])
    if family == "prod":
        return f"(x{sgn()}{c()})×(x{sgn()}{c()})"
    if family == "frac":
        return f"{c()}÷(1+x^2)"
    raise ValueError(family)


TRAIN_X_FAMILIES = ["lin", "quad", "cubic", "trig", "exp", "sum"]
TEST_X_FAMILIES = ["nest", "prod", "frac"]


def t_plug(rng, families=TRAIN_X_FAMILIES, cat="plug") -> Example:
    for _ in range(100):
        text = _rand_expr_x(rng, str(rng.choice(families)))
        v = rnum(rng, 9, 0.3, 0)
        if rng.random() < 0.3:
            v = -v
        try:
            fn = compile_ast(parse_expr(tokenize(text)))
            y = fn({"x": v}, {})
        except Exception:
            continue
        if not (math.isfinite(y) and abs(y) < 1e6):
            continue
        f = lambda xx, fn=fn: fn({"x": xx}, {})
        p = ["<bos>", "문제:", "곡선:", "f(", "x", ")", "="] + tokenize(text) + ["\n", "x", "="] + tokenize(ntext(v)) \
            + ["높이?", "\n"]
        return Example(cat, p, f"f({ntext(v)})=\n", "calc", y, x=v, f=f)
    raise RuntimeError


def _rect_list(rng, k, maxd, dec=0.15):
    """계단 모양(같은 폭이 3개 이상)을 피한 직사각형 목록."""
    while True:
        rs = [(rdig(rng, maxd, 1, dec), rdig(rng, maxd, 1, dec)) for _ in range(k)]
        ws = [w for w, _ in rs]
        if k >= 3 and max(ws.count(w) for w in ws) >= 3:
            continue
        return rs


def _shape_prompt(rs):
    p = ["<bos>", "문제:", "도형:"]
    for j, (w, h) in enumerate(rs):
        if j:
            p.append("붙이기")
        p += ["직사각형", "("] + tokenize(ntext(w)) + [","] + tokenize(ntext(h)) + [")"]
    return p + ["넓이?", "\n"]


def t_shape(rng, k=None, maxd=3, cat="shape", rs=None) -> Example:
    k = k or int(rng.choice([2, 3, 4, 5], p=[.3, .3, .25, .15]))
    rs = rs or _rect_list(rng, k, maxd)
    total = sum(w * h for w, h in rs)
    rects = [f"rect({ntext(w)},{ntext(h)})" for w, h in rs]
    style = rng.choice(["each", "define", "inline"])
    if style == "each":     # 하나씩 계산기로 → 결과(4자리로 보임)를 보고 더하기
        lines = [r + "=" for r in rects]
        lines.append("+".join(ntext(w * h) for w, h in rs) + "=")
    elif style == "define":  # 정의로 저장했다가 다시 쓰기
        slots = pslots(rng, len(rs))
        lines = [f"정의 {s}={r}" for s, r in zip(slots, rects)] + ["+".join(slots) + "="]
    else:
        lines = ["+".join(rects) + "="]
    return Example(cat, _shape_prompt(rs), "\n".join(lines) + "\n", "calc", total)


def t_frame_or_t(rng) -> Example:
    """ㅁ 모양(같은 크기 막대 2쌍)과 T 모양: 학습의 무작위 목록과 겹치지 않게 만든 규칙적인 도형."""
    W, H = rdig(rng, 3), rdig(rng, 3)
    w, h = rdig(rng, 2), rdig(rng, 2)
    if rng.random() < .5:   # ㅁ: 위·아래 막대, 왼·오른쪽 막대
        rs = [(W, h), (w, H), (w, H), (W, h)]
    else:                   # T: 윗막대 + 기둥
        rs = [(W, h), (w, H)]
    return t_shape(rng, rs=rs, cat="new_shapes")


def t_nested(rng, maxd=3) -> Example:
    """안에 또 붙이기가 있는 도형 (시험 전용)."""
    rs = _rect_list(rng, 3, maxd)
    (w1, h1), (w2, h2), (w3, h3) = rs
    p = ["<bos>", "문제:", "도형:", "직사각형", "("] + tokenize(ntext(w1)) + [","] + tokenize(ntext(h1)) + [")", "붙이기", "("]
    for j, (w, h) in enumerate(rs[1:]):
        if j:
            p.append("붙이기")
        p += ["직사각형", "("] + tokenize(ntext(w)) + [","] + tokenize(ntext(h)) + [")"]
    p += [")", "넓이?", "\n"]
    total = sum(w * h for w, h in rs)
    r = lambda w, h: f"rect({ntext(w)},{ntext(h)})"
    resp = f"{r(w1,h1)}+({r(w2,h2)}+{r(w3,h3)})=\n"
    return Example("nested", p, resp, "calc", total)


def _sep(rng):
    return str(rng.choice(["\n", ";"]))


def t_loop(rng, max_nd=3, min_nd=1, cat="loop") -> Example:
    """1 부터 K 까지 더하기. (S 부터 E 까지는 모델이 E-S+1 을 암산해야 해서 도구 쓰기 시험이 아니게 된다.)"""
    k = max(2, int(rdig(rng, max_nd, min_nd)))
    slot = pslot(rng)
    body = str(rng.choice(["i+1", "1+i", "N-i"]))
    sep = _sep(rng)
    resp = f"정의 {slot}=0{sep}반복 N={k}: 정의 {slot}={slot}+{body}{sep}{slot}=\n"
    p = ["<bos>", "문제:", "1", "부터"] + tokenize(str(k)) + ["까지", "더하기?", "\n"]
    return Example(cat, p, resp, "calc", float(k * (k + 1) // 2))


def _rand_arith(rng, depth, leaf):
    """(글자, 값) — 값은 파서로 다시 계산."""
    if depth == 0 or rng.random() < 0.25:
        return leaf()
    op = str(rng.choice(["+", "-", "×", "÷"], p=[.3, .25, .3, .15]))
    l, r = _rand_arith(rng, depth - 1, leaf), _rand_arith(rng, depth - 1, leaf)
    txt = f"{_wrap(l)}{op}{_wrap(r)}"
    return txt


def _wrap(t):
    return f"({t})" if any(c in t for c in "+-×÷") else t


def _try_value(text, vars_=None):
    try:
        y = compile_ast(parse_expr(tokenize(text), allow_cmp=True))(vars_ or {}, {})
    except Exception:
        return None
    if isinstance(y, bool):
        return y
    return y if math.isfinite(y) and abs(y) < 1e6 else None


def t_calc(rng, depth=None, maxd=3, cat="calc", with_ab=False) -> Example:
    depth = depth or int(rng.integers(1, 5))
    for _ in range(200):
        a = rnum(rng, 9, .2)
        b = round(a + rnum(rng, 9, .2), 1)
        if with_ab:
            leaf = lambda: str(rng.choice(["a", "b", "a", "b", ntext(rnum(rng, 9, .2))]))
        else:
            leaf = lambda: ntext(rdig(rng, maxd, 1, .15))
        text = _rand_arith(rng, depth, leaf)
        if not any(c in text for c in "+-×÷"):
            continue
        cmpv = (not with_ab) and rng.random() < 0.15
        if cmpv:
            text = f"{text}{rng.choice(['<', '>'])}{ntext(rdig(rng, maxd, 1, .15))}"
        y = _try_value(text, {"a": a, "b": b})
        if y is None:
            continue
        p = ["<bos>", "문제:"] + tokenize(text) + ["계산?"]
        if with_ab:   # 식 → 계산? → 구간 (숫자가 이어 붙어 모호해지는 것 방지)
            p += ["구간:", "a", "="] + tokenize(ntext(a)) + ["b", "="] + tokenize(ntext(b))
        p += ["\n"]
        return Example(cat, p, text + "=\n", "calc", y if isinstance(y, bool) else float(y),
                       interval=(a, b) if with_ab else None)
    raise RuntimeError


def _int(rng, lo, hi):
    return float(rng.integers(lo, hi + 1))


def t_submit(rng, cat="submit") -> Example:
    kind = rng.choice(["loop", "rect", "shape", "calc"], p=[.3, .15, .3, .25])
    sep = _sep(rng)
    if kind == "loop":
        k = max(2, int(rdig(rng, 3)))
        slot = pslot(rng)
        body = str(rng.choice(["i+1", "1+i", "N-i"]))
        resp = f"제출[ 정의 {slot}=0{sep}반복 N={k}: 정의 {slot}={slot}+{body}{sep}{slot} ]\n"
        p = ["<bos>", "문제:", "1", "부터"] + tokenize(str(k)) + ["까지", "더하기?", "제출?", "\n"]
        return Example(cat, p, resp, "submit", float(k * (k + 1) // 2))
    if kind == "rect":
        w, h = float(int(rdig(rng, 4))), float(int(rdig(rng, 4)))
        p = ["<bos>", "문제:", "직사각형", "가로"] + tokenize(ntext(w)) + ["세로"] + tokenize(ntext(h)) + ["넓이?", "제출?", "\n"]
        return Example(cat, p, f"제출[ rect({ntext(w)},{ntext(h)}) ]\n", "submit", w * h)
    if kind == "shape":
        k = int(rng.choice([2, 3, 4, 5], p=[.35, .3, .2, .15]))
        rs = _rect_list(rng, k, 3, dec=0.0)
        p = _shape_prompt(rs)[:-2] + ["제출?", "\n"]
        rects = [f"rect({ntext(w)},{ntext(h)})" for w, h in rs]
        if rng.random() < 0.5:
            resp = "제출[ " + "+".join(rects) + " ]\n"
        else:
            slots = pslots(rng, len(rs))
            resp = "제출[ " + sep.join([f"정의 {s}={r}" for s, r in zip(slots, rects)] + ["+".join(slots)]) + " ]\n"
        return Example(cat, p, resp, "submit", sum(w * h for w, h in rs))
    for _ in range(200):
        ex = t_calc(rng, cat=cat)
        if isinstance(ex.target, bool) or ex.target == 0 or "." in "".join(ex.prompt):
            continue
        expr = ex.response.rstrip("\n")[:-1]
        if not _is_int_expr(expr):
            continue
        p = ex.prompt[:-2] + ["제출?", "\n"]
        return Example(cat, p, f"제출[ {expr} ]\n", "submit", ex.target)
    raise RuntimeError


def t_format(rng, cat="format") -> Example:
    """진짜 문제와 같은 모양(곡선 + 구간)에서 넓이가 아닌 걸 묻는 형식 연습.

    묻는 것: 한 점의 높이(x=a, x=b, x=임의의 정수 — 구간 안팎 무관, 위치가 특별하지 않게) 또는 a, b 로 하는 간단한 계산.
    반복·직사각형 쌓기·넓이 풀이는 없다. 답은 계산기로 읽거나(제출 없음) `제출[ ... ]` 로 낸다.
    """
    from .curves import random_curve

    for _ in range(100):
        c = random_curve(rng)
        sub = rng.random() < 0.55
        kind = str(rng.choice(["a", "b", "int", "calc"], p=[.2, .2, .3, .3]))
        base = ["<bos>", "문제:", "곡선:", "f(", "x", ")", "="] + tokenize(c.text) \
            + ["구간:", "a", "="] + tokenize(fmt_num(c.a)) + ["b", "="] + tokenize(fmt_num(c.b))
        if kind == "calc":
            expr = str(rng.choice(["a+b", "b-a", "a-b", "a×b", "(a+b)÷2", "b÷a", "a+a", "b×b", "b-a+b"]))
            try:
                y = compile_ast(parse_expr(tokenize(expr)))({"a": c.a, "b": c.b}, {})
            except Exception:
                continue
            q = tokenize(expr) + ["계산?"]
            call, x = expr, None
        else:
            v = {"a": c.a, "b": c.b}.get(kind)
            sym = kind
            if kind == "int":
                v = float(rng.integers(-9, 21))
                sym = ntext(v)
            try:
                y = c.f(v)
            except Exception:
                continue
            q = ["\n", "x", "="] + tokenize(sym) + ["높이?"]
            call, x = f"f({sym})", v
        if not (math.isfinite(y) and 1e-6 < abs(y) < 1e6):
            continue
        # 줄바꿈 하나로 문제 끝
        p = base + (["\n"] if kind == "calc" else []) + q + (["제출?"] if sub else []) + ["\n"]
        if kind == "calc":     # 식은 구간 뒤에 오면 숫자와 붙으니 구간 앞으로
            p = ["<bos>", "문제:"] + tokenize(expr) + ["계산?"] + (["제출?"] if sub else []) \
                + base[2:] + ["\n"]
        resp = f"제출[ {call} ]\n" if sub else f"{call}=\n"
        return Example(cat, p, resp, "submit" if sub else "calc", float(y), x=x, f=c.f,
                       interval=(c.a, c.b))
    raise RuntimeError


# ------------------------------------------------------------------ 다시 제출 (A2)
def _slip(rng, toks):
    """맞는 절차(토큰 목록)에서 사소한 실수 하나를 만든다. 반복 횟수(N=)는 건드리지 않는다."""
    opts = []
    nums = []
    i = 0
    while i < len(toks):
        if toks[i] in V.DIGITS:
            j = i
            while j < len(toks) and toks[j] in V.DIGITS:
                j += 1
            if not (i >= 2 and toks[i - 2:i] == ["N", "="]):
                nums.append((i, j))
            i = j
        else:
            i += 1
    if nums:
        opts.append("num")
    if any(t in ("a", "b") for t in toks):
        opts.append("ab")
    ops = [k for k, t in enumerate(toks) if t in ("+", "-", "×", "÷") and k > 0]
    if ops:
        opts.append("op")
    if not opts:
        return None
    kind = str(rng.choice(opts))
    out = list(toks)
    if kind == "num":
        i, j = nums[int(rng.integers(len(nums)))]
        v = int("".join(toks[i:j]))
        d = int(rng.integers(1, max(2, v // 2 + 1)))
        nv = v + d if (v == 0 or rng.random() < .5) else max(0, v - d)
        if nv == v or nv > 10000:
            return None
        out[i:j] = list(str(nv))
    elif kind == "ab":
        k = [k for k, t in enumerate(toks) if t in ("a", "b")][0]
        out[k] = "b" if toks[k] == "a" else "a"
    else:
        k = ops[int(rng.integers(len(ops)))]
        out[k] = {"+": "-", "-": "+", "×": "÷", "÷": "×"}[toks[k]]
    return out


def _body_text(toks):
    return V.decode(V.ids_of(toks)).strip()


def t_retry(rng, cat="retry") -> Example:
    """제출 → 오차 → 다시 제출. 넓이와 무관한 문제에서 '사소한 실수를 한 제출 뒤에 오차를 보고 맞게 고쳐 쓰기'
    (또는 이미 맞았으면 같은 걸 다시). 쪼개기·직사각형 쌓기와는 무관한 일반 습관이다."""
    for _ in range(300):
        base = t_format(rng, cat) if rng.random() < .5 else t_submit(rng, cat)
        if base.mode != "submit":
            continue
        body = base.response.strip()
        assert body.startswith("제출[") and body.endswith("]")
        right = tokenize(body[len("제출["):-1])
        n = int(rng.choice([2, 3], p=[.7, .3]))
        wrongs, fine = [], True
        n_wrong = int(rng.choice([0, 1, n - 1], p=[.2, .5, .3])) if n == 3 else int(rng.choice([0, 1], p=[.2, .8]))
        for _k in range(n_wrong):
            w = _slip(rng, right)
            if w is None:
                fine = False
                break
            wrongs.append(w)
        if not fine:
            continue
        seq = wrongs + [right] * (n - len(wrongs))
        resp = "제출[ " + " ] ".join(_body_text(t) for t in seq) + " ]\n"
        ex = Example(cat, base.prompt, resp, "submit", base.target, x=base.x, f=base.f,
                     interval=base.interval, min_submits=n)
        # 실수한 제출은 실제로 오차가 있어야 하고(유한), 마지막은 맞아야 한다
        nb = build(ex, resp)
        subs = nb.submissions
        if len(subs) != n or not check(nb, ex):
            continue
        errs = []
        for k in range(len(wrongs)):
            nb2 = build(Example(cat, base.prompt, "", "submit", base.target, x=base.x, f=base.f,
                                interval=base.interval))
            nb2.feed_text("제출[ " + _body_text(wrongs[k]) + " ]")
            errs.append(nb2.last_feedback)
        if any(e is None or not (1e-4 < e < 100) for e in errs):
            continue
        return ex
    raise RuntimeError


def _is_int_expr(expr):
    return "." not in expr


# ------------------------------------------------------------------ 섞기
TRAIN_MIX = [
    ("rect", 0.06), ("plug", 0.07), ("shape", 0.17), ("loop", 0.10),
    ("calc", 0.06), ("calc_ab", 0.05), ("submit", 0.15), ("format", 0.16), ("retry", 0.18),
]


def sample_train(rng) -> Example:
    names, ps = zip(*TRAIN_MIX)
    k = str(rng.choice(names, p=ps))
    if k == "rect":
        return t_rect(rng)
    if k == "plug":
        return t_plug(rng)
    if k == "shape":
        return t_shape(rng)
    if k == "loop":
        return t_loop(rng)
    if k == "calc":
        return t_calc(rng)
    if k == "calc_ab":
        return t_calc(rng, with_ab=True, cat="calc_ab")
    if k == "format":
        return t_format(rng)
    if k == "retry":
        return t_retry(rng)
    return t_submit(rng)


TEST_SETS: dict[str, Callable] = {
    # ---- PLAN.md 5.3 의 네 가지
    "new_shapes": lambda rng: t_frame_or_t(rng),
    "big_numbers": lambda rng: t_rect(rng, maxd=4, mind=4, cat="big_numbers"),   # 가로·세로 모두 4자리 = 학습 중 가장 큰 값 (5자리 이상은 실험에 안 쓰므로 시험 안 함)
    "nested": lambda rng: t_nested(rng),
    "new_formulas": lambda rng: t_plug(rng, TEST_X_FAMILIES, "new_formulas"),
    "format_real": lambda rng: t_format(rng, "format_real"),   # 진짜 문제 모양, 처음 보는 곡선
    "retry_fix": lambda rng: t_retry(rng, "retry_fix"),         # 틀린 제출 뒤 오차를 보고 다시 제출
    # ---- 참고용: 학습 범위를 한 칸 넘는 것
    "more_pieces": lambda rng: t_shape(rng, k=int(rng.choice([6, 7])), maxd=3, cat="more_pieces"),
    "longer_loop": lambda rng: t_loop(rng, max_nd=4, min_nd=4, cat="longer_loop"),
    "long_calc": lambda rng: t_calc(rng, depth=5, cat="long_calc"),
}
PASS_SETS = ["train_like", "new_shapes", "big_numbers", "nested", "new_formulas", "format_real", "retry_fix"]
