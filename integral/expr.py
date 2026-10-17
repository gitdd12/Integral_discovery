"""수식: 파서, 컴파일러, 숫자 표시.

곡선 수식, 계산기 식, 제출 절차 안의 식이 모두 같은 문법을 쓴다.
  비교  :  add (< | >) add          (계산기 전용)
  add   :  mul ((+|-) mul)*
  mul   :  unary ((×|÷) unary)*
  unary :  - unary | pow
  pow   :  atom (^ unary)?
  atom  :  숫자 | 이름 | f( add ) | rect( add , add ) | sin|cos|exp ( add ) | ( add )
"""
from __future__ import annotations

import math
from decimal import Decimal

from .vocab import DIGITS, SLOTS

VARS = ("a", "b", "x", "i", "N")
NAMES = set(VARS) | set(SLOTS)
FUNCS = ("sin", "cos", "exp")
MAX_NODES = 4000


class ParseError(Exception):
    pass


class EvalError(Exception):
    pass


class Unbound(EvalError):
    """a, b, x, i, N, f 처럼 지금 값이 없는 이름."""


# ---------------------------------------------------------------- 숫자 표시
def fmt_num(v: float) -> str:
    """유효숫자 4자리, 지수 표기 없이. 1e-6 미만은 0, 1e12 이상은 999999999999."""
    if not math.isfinite(v):
        raise EvalError("not finite")
    a = abs(v)
    if a < 1e-6:
        return "0"
    if a >= 1e12:
        s = "999999999999"
    else:
        s = format(Decimal("%.4g" % a), "f")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
    return ("-" if v < 0 else "") + s


# ---------------------------------------------------------------- 파서
class _Parser:
    def __init__(self, toks: list[str]):
        self.t = toks
        self.p = 0
        self.n = 0

    def peek(self):
        return self.t[self.p] if self.p < len(self.t) else None

    def take(self, want=None):
        tok = self.peek()
        if tok is None or (want is not None and tok != want):
            raise ParseError(f"expected {want!r}, got {tok!r}")
        self.p += 1
        return tok

    def node(self, *n):
        self.n += 1
        if self.n > MAX_NODES:
            raise ParseError("too big")
        return n

    def cmp(self):
        l = self.add()
        if self.peek() in ("<", ">"):
            op = self.take()
            return self.node("cmp", op, l, self.add())
        return l

    def add(self):
        l = self.mul()
        while self.peek() in ("+", "-"):
            op = self.take()
            l = self.node("bin", op, l, self.mul())
        return l

    def mul(self):
        l = self.unary()
        while self.peek() in ("×", "÷"):
            op = self.take()
            l = self.node("bin", op, l, self.unary())
        return l

    def unary(self):
        if self.peek() == "-":
            self.take()
            return self.node("neg", self.unary())
        return self.pow()

    def pow(self):
        l = self.atom()
        if self.peek() == "^":
            self.take()
            return self.node("bin", "^", l, self.unary())
        return l

    def atom(self):
        t = self.peek()
        if t is None:
            raise ParseError("unexpected end")
        if t in DIGITS:
            return self.number()
        if t in NAMES:
            self.take()
            return self.node("name", t)
        if t == "(":
            self.take()
            e = self.add()
            self.take(")")
            return e
        if t == "f(":
            self.take()
            e = self.add()
            self.take(")")
            return self.node("f", e)
        if t == "rect(":
            self.take()
            w = self.add()
            self.take(",")
            h = self.add()
            self.take(")")
            return self.node("rect", w, h)
        if t in FUNCS:
            self.take()
            self.take("(")
            e = self.add()
            self.take(")")
            return self.node("fn", t, e)
        raise ParseError(f"unexpected {t!r}")

    def number(self):
        s = ""
        while self.peek() in DIGITS:
            s += self.take()
        if self.peek() == ".":
            self.take()
            if self.peek() not in DIGITS:
                raise ParseError("digits after .")
            s += "."
            while self.peek() in DIGITS:
                s += self.take()
        return self.node("num", float(s), "." not in s)


def parse_expr(toks: list[str], allow_cmp: bool = False):
    if not toks:
        raise ParseError("empty")
    P = _Parser(toks)
    ast = P.cmp() if allow_cmp else P.add()
    if P.p != len(toks):
        raise ParseError(f"extra {toks[P.p]!r}")
    return ast


def ast_nodes(ast):
    yield ast
    for c in ast[1:]:
        if isinstance(c, tuple):
            yield from ast_nodes(c)


def subst(ast, store):
    """슬롯 이름을 지금 저장된 값/식으로 바꿔 넣는다 (게으른 정의의 순환 방지)."""
    k = ast[0]
    if k == "name":
        n = ast[1]
        if n in SLOTS:
            e = store.get(n)
            if e is None:
                raise EvalError(f"undefined {n}")
            if isinstance(e, Lazy):
                return e.ast
            return ("num", float(e), False)
        return ast
    if k == "num":
        return ast
    return (k,) + tuple(subst(c, store) if isinstance(c, tuple) else c for c in ast[1:])


# ---------------------------------------------------------------- 컴파일
class Lazy:
    """a, b, N, i 같은 값이 필요해서 지금은 계산 못 하는 정의. 쓸 때마다 다시 계산."""

    def __init__(self, ast, fn):
        self.ast, self.fn = ast, fn


def _scalar_lib():
    def sat_exp(z):
        return math.exp(min(z, 700.0))

    def div(l, r):
        if r == 0:
            raise EvalError("div0")
        return l / r

    def power(l, r):
        try:
            return math.pow(l, r)
        except (ValueError, OverflowError):
            raise EvalError("pow")

    def wrap(fn):
        def g(z):
            try:
                return fn(z)
            except (ValueError, OverflowError):
                raise EvalError("math")
        return g

    return dict(sin=wrap(math.sin), cos=wrap(math.cos), exp=wrap(sat_exp), div=div, pow=power)


def _numpy_lib():
    import numpy as np

    return dict(
        sin=np.sin, cos=np.cos, exp=lambda z: np.exp(np.minimum(z, 700.0)),
        div=np.divide, pow=np.power,
    )


SCALAR = _scalar_lib()


def compile_ast(ast, lib=SCALAR):
    """식 → 함수 fn(V, S). V: 이름 값 사전(a,b,x,i,N,f), S: 슬롯 저장소."""
    k = ast[0]
    if k == "num":
        v = ast[1]
        return lambda V, S: v
    if k == "name":
        n = ast[1]
        if n in SLOTS:
            def slot(V, S):
                e = S.get(n)
                if e is None:
                    raise EvalError(f"undefined {n}")
                return e.fn(V, S) if isinstance(e, Lazy) else e
            return slot

        def var(V, S):
            try:
                return V[n]
            except KeyError:
                raise Unbound(n)
        return var
    if k == "neg":
        e = compile_ast(ast[1], lib)
        return lambda V, S: -e(V, S)
    if k == "bin":
        op = ast[1]
        l, r = compile_ast(ast[2], lib), compile_ast(ast[3], lib)
        if op == "+":
            return lambda V, S: l(V, S) + r(V, S)
        if op == "-":
            return lambda V, S: l(V, S) - r(V, S)
        if op == "×":
            return lambda V, S: l(V, S) * r(V, S)
        if op == "÷":
            div = lib["div"]
            return lambda V, S: div(l(V, S), r(V, S))
        pw = lib["pow"]
        return lambda V, S: pw(l(V, S), r(V, S))
    if k == "f":
        e = compile_ast(ast[1], lib)

        def call(V, S):
            try:
                f = V["f"]
            except KeyError:
                raise Unbound("f")
            return f(e(V, S))
        return call
    if k == "rect":
        w, h = compile_ast(ast[1], lib), compile_ast(ast[2], lib)
        return lambda V, S: w(V, S) * h(V, S)
    if k == "fn":
        fn, e = lib[ast[1]], compile_ast(ast[2], lib)
        return lambda V, S: fn(e(V, S))
    if k == "cmp":
        op = ast[1]
        l, r = compile_ast(ast[2], lib), compile_ast(ast[3], lib)
        if op == "<":
            return lambda V, S: l(V, S) < r(V, S)
        return lambda V, S: l(V, S) > r(V, S)
    raise ParseError(k)
