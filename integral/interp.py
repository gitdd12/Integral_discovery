"""절차 해석기: 정의 / 반복 / 식.

  정의 p0 = 식        값을 구할 수 있으면 값 저장, a·b·N·i 같은 게 필요하면 식 그대로 저장(Lazy)
  반복 N=정수: 문장   문장을 N번 실행. i = 0..N-1. 결과를 자동으로 더하지 않는다.
  식                  마지막 문장이 식이면 그 값이 절차의 결과
"""
from __future__ import annotations

import math

from .expr import (
    EvalError, Lazy, ParseError, Unbound, ast_nodes, compile_ast, parse_expr, subst,
    MAX_NODES,
)
from .vocab import DIGITS, SLOTS

MAX_LOOP = 10000          # 반복 횟수·제출 안 정수의 상한
MAX_STEPS = 200_000       # 한 번 실행에서 반복 몸통을 도는 총 횟수 상한


class Ctx:
    def __init__(self, max_steps: int = MAX_STEPS):
        self.steps = 0
        self.max_steps = max_steps

    def tick(self):
        self.steps += 1
        if self.steps > self.max_steps:
            raise EvalError("step limit")


def split_statements(toks: list[str]) -> list[list[str]]:
    out, cur = [], []
    for t in toks:
        if t in (";", "\n"):
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        out.append(cur)
    return out


def parse_stmt(toks: list[str]):
    if not toks:
        raise ParseError("empty statement")
    h = toks[0]
    if h == "정의":
        if len(toks) < 4 or toks[1] not in SLOTS or toks[2] != "=":
            raise ParseError("정의 형식")
        ast = parse_expr(toks[3:])
        return ("def", toks[1], ast, compile_ast(ast))
    if h == "반복":
        if len(toks) < 6 or toks[1] != "N" or toks[2] != "=":
            raise ParseError("반복 형식")
        j = 3
        s = ""
        while j < len(toks) and toks[j] in DIGITS:
            s += toks[j]
            j += 1
        if not s or j >= len(toks) or toks[j] != ":":
            raise ParseError("반복 횟수")
        n = int(s)
        if not 1 <= n <= MAX_LOOP:
            raise ParseError("반복 횟수 범위")
        return ("loop", n, parse_stmt(toks[j + 1:]))
    ast = parse_expr(toks)
    return ("expr", ast, compile_ast(ast))


def parse_program(toks: list[str], strict: bool = False):
    """strict(제출 절차): 정수만, 크기 상한 MAX_LOOP, 소수점 금지, 읽기 전용 기호 금지."""
    if strict and "." in toks:
        raise ParseError("소수 금지")
    prog = [parse_stmt(s) for s in split_statements(toks)]
    if not prog:
        raise ParseError("빈 절차")
    if strict:
        for st in prog:
            _check_strict(st)
    return prog


def _check_strict(st):
    if st[0] == "loop":
        _check_strict(st[2])
        return
    ast = st[2] if st[0] == "def" else st[1]
    for n in ast_nodes(ast):
        if n[0] == "num" and not (n[2] and n[1] <= MAX_LOOP):
            raise ParseError("정수 범위")
        if n[0] == "fn" or (n[0] == "bin" and n[1] == "^"):
            raise ParseError("읽기 전용 기호")


def _finite(v):
    if not isinstance(v, (int, float)) or not math.isfinite(v):
        raise EvalError("not finite")
    return float(v)


def exec_stmt(st, V, S, ctx: Ctx):
    k = st[0]
    if k == "def":
        _, slot, ast, fn = st
        try:
            S[slot] = _finite(fn(V, S))
        except Unbound:
            ast2 = subst(ast, S)
            if sum(1 for _ in ast_nodes(ast2)) > MAX_NODES:
                raise EvalError("too big")
            S[slot] = Lazy(ast2, compile_ast(ast2))
        return None
    if k == "loop":
        _, n, body = st
        had = {key: V[key] for key in ("N", "i") if key in V}
        try:
            for i in range(n):
                ctx.tick()
                V["N"], V["i"] = float(n), float(i)
                exec_stmt(body, V, S, ctx)
        finally:
            V.pop("N", None)
            V.pop("i", None)
            V.update(had)
        return None
    return _finite(st[2](V, S))


def run_program(prog, V, S, ctx: Ctx | None = None) -> float:
    """절차를 실행하고 마지막 식의 값을 돌려준다. 마지막이 식이 아니면 오류."""
    ctx = ctx or Ctx()
    val = None
    for st in prog:
        val = exec_stmt(st, V, S, ctx)
    if prog[-1][0] != "expr":
        raise EvalError("값이 없음")
    return val
