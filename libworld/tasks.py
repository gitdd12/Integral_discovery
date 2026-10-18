"""과제: 참조 식(정답 풀이)에서 입출력 예시를 만든다. 참조 식은 탐색기에 주지 않는다.

학습 과제 21개(계열별), 시험 과제 24개(학습 개념의 새 조합). 입력은 시드마다 달라진다.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .lang import run

X, Y, A = ("x",), ("y",), ("a",)


def K(n):
    return ("k", n)


def add(p, q): return ("add", p, q)
def sub(p, q): return ("sub", p, q)
def mul(p, q): return ("mul", p, q)
def gt(p, q): return ("gt", p, q)
def eq(p, q): return ("eq", p, q)
def mod(p, q): return ("mod", p, q)
def If(c, t, e): return ("if", c, t, e)
def length(l): return ("len", l)
def rng(n): return ("range", n)
def mp(f, l): return ("map", f, l)
def flt(f, l): return ("filter", f, l)
def fold(f, i, l): return ("fold", f, i, l)


EVEN = eq(mod(Y, K(2)), K(0))
ODD = mod(Y, K(2))
BIG3 = gt(Y, K(3))
SQ = mul(Y, Y)
SUM = lambda l: fold(add(A, Y), K(0), l)
SUMF = lambda f, l: fold(add(A, f), K(0), l)

# (이름, 계열, 입력형, 출력형, 참조 식)
TRAIN = [
    ("합", "집계", "L", "I", SUM(X)),
    ("길이", "집계", "L", "I", length(X)),
    ("제곱의 합", "집계", "L", "I", SUMF(SQ, X)),
    ("곱", "집계", "L", "I", fold(mul(A, Y), K(1), X)),
    ("2배", "변환", "L", "L", mp(add(Y, Y), X)),
    ("제곱", "변환", "L", "L", mp(SQ, X)),
    ("3 더하기", "변환", "L", "L", mp(add(Y, K(3)), X)),
    ("3으로 나눈 나머지", "변환", "L", "L", mp(mod(Y, K(3)), X)),
    ("짝수만", "선택", "L", "L", flt(EVEN, X)),
    ("3보다 큰 것만", "선택", "L", "L", flt(BIG3, X)),
    ("홀수만", "선택", "L", "L", flt(ODD, X)),
    ("0..n-1", "범위", "I", "L", rng(X)),
    ("0..n-1 합", "범위", "I", "I", SUM(rng(X))),
    ("0..n-1 제곱", "범위", "I", "L", mp(SQ, rng(X))),
    ("0..n-1 제곱의 합", "범위", "I", "I", SUMF(SQ, rng(X))),
    ("짝수 개수", "조합", "L", "I", length(flt(EVEN, X))),
    ("n 미만 짝수", "조합", "I", "L", flt(EVEN, rng(X))),
    ("짝수의 2배", "조합", "L", "L", mp(add(Y, Y), flt(EVEN, X))),
    ("3보다 큰 것의 개수", "조합", "L", "I", length(flt(BIG3, X))),
    ("제곱 후 짝수만", "조합", "L", "L", flt(EVEN, mp(SQ, X))),
    ("n 미만 홀수 개수", "조합", "I", "I", length(flt(ODD, rng(X)))),
]

M3 = eq(mod(Y, K(3)), K(0))

TEST = [
    # C: 학습의 부품만으로 새로 조합
    ("짝수의 합", "C", "L", "I", SUM(flt(EVEN, X))),
    ("홀수 제곱의 합", "C", "L", "I", SUMF(SQ, flt(ODD, X))),
    ("n 미만 짝수 제곱의 합", "C", "I", "I", SUMF(SQ, flt(EVEN, rng(X)))),
    ("3보다 큰 것의 제곱", "C", "L", "L", mp(SQ, flt(BIG3, X))),
    ("n 미만 홀수의 합", "C", "I", "I", SUM(flt(ODD, rng(X)))),
    ("짝수 2배의 합", "C", "L", "I", SUMF(add(Y, Y), flt(EVEN, X))),
    # A: 처음 보는 본문
    ("(y+1)의 합", "A", "L", "I", SUMF(add(Y, K(1)), X)),
    ("3배의 합", "A", "L", "I", SUMF(mul(Y, K(3)), X)),
    ("(제곱+1)의 합", "A", "L", "I", SUMF(add(SQ, K(1)), X)),
    ("0..n-1 (y+1)의 합", "A", "I", "I", SUMF(add(Y, K(1)), rng(X))),
    ("0..n-1 3배의 합", "A", "I", "I", SUMF(mul(Y, K(3)), rng(X))),
    ("홀짝 값(y mod 2)의 합", "A", "L", "I", SUMF(mod(Y, K(2)), X)),
    # B: 처음 보는 본문 x 학습한 선택
    ("짝수의 (y+1) 합", "B", "L", "I", SUMF(add(Y, K(1)), flt(EVEN, X))),
    ("홀수 3배의 합", "B", "L", "I", SUMF(mul(Y, K(3)), flt(ODD, X))),
    ("3보다 큰 것의 (y+1) 합", "B", "L", "I", SUMF(add(Y, K(1)), flt(BIG3, X))),
    ("n 미만 짝수 3배의 합", "B", "I", "I", SUMF(mul(Y, K(3)), flt(EVEN, rng(X)))),
    ("짝수의 (y+1) 목록", "B", "L", "L", mp(add(Y, K(1)), flt(EVEN, X))),
    ("홀수 3배 목록", "B", "L", "L", mp(mul(Y, K(3)), flt(ODD, X))),
    # D: 처음 보는 선택 조건
    ("3의 배수의 합", "D", "L", "I", SUM(flt(M3, X))),
    ("3의 배수 개수", "D", "L", "I", length(flt(M3, X))),
    ("n 미만 3의 배수의 합", "D", "I", "I", SUM(flt(M3, rng(X)))),
    ("2인 것의 개수", "D", "L", "I", length(flt(eq(Y, K(2)), X))),
    ("3의 배수 제곱의 합", "D", "L", "I", SUMF(SQ, flt(M3, X))),
    ("3으로 나눈 나머지가 1인 것의 합", "D", "L", "I", SUM(flt(eq(mod(Y, K(3)), K(1)), X))),
]


NSEARCH, NEXT = 12, 30


@dataclass
class Task:
    name: str
    family: str
    incat: str
    outcat: str
    ref: tuple
    all_x: list      # 검색용 예시 + 확인용 예시
    all_y: list

    @property
    def search_x(self):
        return self.all_x[:NSEARCH]

    @property
    def search_y(self):
        return self.all_y[:NSEARCH]


def _gen_input(rng_, incat):
    if incat == "I":
        return rng_.randint(0, 14)
    n = rng_.randint(0, 8)
    return tuple(rng_.randint(0, 9) for _ in range(n))


def make_task(spec, seed, idx):
    name, fam, incat, outcat, ref = spec
    rng_ = random.Random(f"{seed}-{idx}-{name}")
    for _ in range(200):
        xs = [_gen_input(rng_, incat) for _ in range(NSEARCH + NEXT)]
        if incat == "I":
            xs[0], xs[1] = 0, 13          # 경계와 큰 값을 항상 포함
        else:
            xs[0] = ()
        ys = run(ref, {}, xs)
        # 12개에서 서로 다른 출력이 4가지 이상이어야 의미 있는 예시
        if ys is not None and len(set(ys[:NSEARCH])) >= 4:
            return Task(name, fam, incat, outcat, ref, xs, ys)
    raise RuntimeError("예시 생성 실패: " + name)


def make_tasks(seed):
    train = [make_task(s, seed, i) for i, s in enumerate(TRAIN)]
    test = [make_task(s, seed, 100 + i) for i, s in enumerate(TEST)]
    return train, test
