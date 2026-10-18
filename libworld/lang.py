"""작은 함수형 언어와 라이브러리(이름 붙은 개념).

식은 튜플: ('add', e1, e2) ... 잎: ('x',) 입력, ('y',) 원소, ('a',) 누적값, ('k', n) 상수, ('H', i) 구멍.
라이브러리 개념 호출: ('app:이름', 인자...).

인자 종류(kind): 'i' 바깥 정수식, 'l' 바깥 목록식, 'Y' y만 쓰는 본문, 'A' a,y를 쓰는 본문, 'K' 상수 잎.
"""
from __future__ import annotations

from dataclasses import dataclass, field

INT, LIST = "int", "list"

# 연산: 인자 종류들, 반환 형
OPS = {
    "add": (("i", "i"), INT), "sub": (("i", "i"), INT), "mul": (("i", "i"), INT),
    "gt": (("i", "i"), INT), "eq": (("i", "i"), INT), "mod": (("i", "i"), INT),
    "if": (("i", "i", "i"), INT),
    "len": (("l",), INT), "head": (("l",), INT),
    "range": (("i",), LIST),
    "map": (("Y", "l"), LIST), "filter": (("Y", "l"), LIST),
    "fold": (("A", "i", "l"), INT),
}
ARITH = ("add", "sub", "mul", "gt", "eq", "mod")
BIG = 10 ** 6
MAXLEN = 40


@dataclass
class Macro:
    name: str
    tmpl: tuple                      # ('H', i) 구멍이 있는 틀
    kinds: tuple                     # 구멍별 종류 'i','l','Y','A','K'
    ret: str                         # INT / LIST
    level: str                       # 'outer' 또는 'body'
    scope: str = "outer"             # body 일 때 'Y' 또는 'A'

    @property
    def arity(self):
        return len(self.kinds)


LEAVES = ("x", "y", "a", "k", "H")


def is_leaf(e) -> bool:
    return e[0] in LEAVES


def size(e) -> int:
    if is_leaf(e):
        return 1
    return 1 + sum(size(c) for c in e[1:])


def children(e):
    return () if is_leaf(e) else e[1:]


def with_children(e, cs):
    return (e[0],) + tuple(cs)


def subst(tmpl, args):
    """틀의 구멍 ('H', i)를 args[i]로 바꾼다."""
    if tmpl[0] == "H":
        return args[tmpl[1]]
    if is_leaf(tmpl):
        return tmpl
    return (tmpl[0],) + tuple(subst(c, args) for c in tmpl[1:])


def expand(e, lib: dict):
    """라이브러리 호출을 풀어서 기본 연산만 있는 식으로."""
    if is_leaf(e):
        return e
    op = e[0]
    if op.startswith("app:"):
        m = lib[op[4:]]
        return expand(subst(m.tmpl, [expand(c, lib) for c in e[1:]]), lib)
    return (op,) + tuple(expand(c, lib) for c in e[1:])


def ret_type(e, lib):
    if e[0].startswith("app:"):
        return lib[e[0][4:]].ret
    if e[0] in ("y", "a", "k"):
        return INT
    return OPS[e[0]][1]


def to_src(e, kinds=None) -> str:
    """기본 연산만 있는 식을 파이썬 식 문자열로."""
    op = e[0]
    if op == "x":
        return "x"
    if op == "y":
        return "y"
    if op == "a":
        return "a"
    if op == "k":
        return str(e[1])
    if op == "H":
        k = kinds[e[1]] if kinds else "I"
        return f"P{e[1]}(y)" if k == "Y" else f"P{e[1]}(a,y)" if k == "A" else f"P{e[1]}"
    c = [to_src(z, kinds) for z in e[1:]]
    if op == "add":
        return f"_c({c[0]}+{c[1]})"
    if op == "sub":
        return f"_c({c[0]}-{c[1]})"
    if op == "mul":
        return f"_c({c[0]}*{c[1]})"
    if op == "gt":
        return f"int({c[0]}>{c[1]})"
    if op == "eq":
        return f"int({c[0]}=={c[1]})"
    if op == "mod":
        return f"_mod({c[0]},{c[1]})"
    if op == "if":
        return f"({c[1]} if {c[0]} else {c[2]})"
    if op == "len":
        return f"len({c[0]})"
    if op == "head":
        return f"_head({c[0]})"
    if op == "range":
        return f"_range({c[0]})"
    if op == "map":
        return f"[{c[0]} for y in {c[1]}]"
    if op == "filter":
        return f"[y for y in {c[1]} if {c[0]}]"
    if op == "fold":
        return f"_fold(lambda a,y: {c[0]}, {c[1]}, {c[2]})"
    raise ValueError(op)


def _c(v):
    return BIG if v > BIG else (-BIG if v < -BIG else v)


def _mod(a, b):
    return a % b if b else 0


def _head(l):
    return l[0] if l else 0


def _range(n):
    return list(range(max(0, min(n, MAXLEN))))


def _fold(f, init, l):
    a = init
    for y in l:
        a = f(a, y)
    return a


ENV = dict(_c=_c, _mod=_mod, _head=_head, _range=_range, _fold=_fold)


def compile_program(e, lib):
    """x 하나를 받는 파이썬 함수 (라이브러리는 풀어서)."""
    src = "lambda x: " + to_src(expand(e, lib))
    return eval(src, dict(ENV))


def run(e, lib, xs):
    f = compile_program(e, lib)
    out = []
    for x in xs:
        try:
            v = f(list(x) if isinstance(x, (list, tuple)) else x)
        except Exception:
            return None
        out.append(tuple(v) if isinstance(v, list) else v)
    return out


def show(e, lib=None) -> str:
    op = e[0]
    if op == "k":
        return str(e[1])
    if op == "H":
        return f"?{e[1]}"
    name = op[4:] if op.startswith("app:") else op
    if is_leaf(e) or len(e) == 1:
        return name
    return f"{name}(" + ",".join(show(c) for c in e[1:]) + ")"
