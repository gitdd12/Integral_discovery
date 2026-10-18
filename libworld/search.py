"""크기 순 상향식 탐색 (출력이 같은 식은 하나만 남긴다).

범주: 'I' 바깥 정수식, 'L' 바깥 목록식, 'Y' y만 쓰는 본문, 'A' a,y를 쓰는 본문.
본문은 격자(y∈[-3,12], a∈[-3,12])에서 계산한 값으로 중복을 거르고, 바깥 식은 과제 예시에서 계산한 값으로 거른다.
비용 = 식 크기 (개념 호출은 1 + 인자 비용).
"""
from __future__ import annotations

import time
from collections import defaultdict
from itertools import product

import numpy as np

from .lang import ARITH, BIG, MAXLEN, ENV, expand, to_src, _c, _mod

YG = np.arange(-3, 13, dtype=np.int64)
AGRID, YGRID = np.meshgrid(YG, YG, indexing="ij")     # [a행, y열]
NG = len(YG)


# ------------------------------------------------------------------ numpy 본문 계산
def np_apply(op, a):
    if op == "add":
        return np.clip(a[0] + a[1], -BIG, BIG)
    if op == "sub":
        return np.clip(a[0] - a[1], -BIG, BIG)
    if op == "mul":
        return np.clip(a[0] * a[1], -BIG, BIG)
    if op == "gt":
        return (a[0] > a[1]).astype(np.int64)
    if op == "eq":
        return (a[0] == a[1]).astype(np.int64)
    if op == "mod":
        d = np.where(a[1] == 0, 1, a[1])
        return np.where(a[1] == 0, 0, np.mod(a[0], d))
    if op == "if":
        return np.where(a[0] != 0, a[1], a[2])
    raise ValueError(op)


def np_eval(e, env):
    op = e[0]
    if op == "y":
        return env["y"]
    if op == "a":
        return env["a"]
    if op == "k":
        return np.full(env["y"].shape, e[1], dtype=np.int64)
    if op == "H":
        return env["H"][e[1]]
    return np_apply(op, [np_eval(c, env) for c in e[1:]])


# ------------------------------------------------------------------ 라이브러리 컴파일
class Compiled:
    """라이브러리의 개념마다 풀어 쓴 틀과 실행 함수."""

    def __init__(self, lib: dict):
        self.lib = lib
        self.xt = {}
        self.fn = {}
        for name, m in lib.items():
            xt = expand(m.tmpl, lib)
            self.xt[name] = xt
            if m.ret in ("I", "L"):
                args = ",".join(f"P{i}" for i in range(m.arity))
                self.fn[name] = eval(f"lambda {args}: " + to_src(xt, m.kinds), dict(ENV))
        self._body = {}

    def body_fn(self, e, scope):
        key = (e, scope)
        f = self._body.get(key)
        if f is None:
            src = to_src(expand(e, self.lib))
            f = eval(("lambda y: " if scope == "Y" else "lambda a,y: ") + src, dict(ENV))
            self._body[key] = f
        return f


def lib_sig(lib: dict):
    return tuple((n, m.tmpl, m.kinds, m.ret) for n, m in lib.items())


# ------------------------------------------------------------------ 본문 풀
def compositions(total, k):
    if k == 0:
        if total == 0:
            yield ()
        return
    if k == 1:
        if total >= 1:
            yield (total,)
        return
    for c in range(1, total - k + 2):
        for rest in compositions(total - c, k - 1):
            yield (c,) + rest


class Bodies:
    def __init__(self, cl: Compiled, max_y=5, max_a=5):
        self.cl, self.max_y, self.max_a = cl, max_y, max_a
        self.Y = defaultdict(list)       # 비용 → [(식, 배열)]
        self.A = defaultdict(list)
        self._build()

    def _build(self):
        cl = self.cl
        seenY, seenA = set(), set()

        def addY(c, e, arr):
            k = arr.tobytes()
            if k not in seenY:
                seenY.add(k)
                self.Y[c].append((e, arr))
                return True
            return False

        def addA(c, e, arr):
            k = arr.tobytes()
            if k not in seenA:
                seenA.add(k)
                self.A[c].append((e, arr))

        def up(arr):
            return np.broadcast_to(arr[None, :], (NG, NG)).copy()

        envY = {"y": YG}
        envA = {"y": YGRID, "a": AGRID}
        # 틀: (인자 범주들, 반환 범주, 식 만들기, 값 계산)
        base = []
        for op in ARITH:
            base.append((("*", "*"), (op,), lambda arrs, op=op: np_apply(op, arrs)))
        base.append((("*", "*", "*"), ("if",), lambda arrs: np_apply("if", arrs)))
        macs = []
        for name, m in cl.lib.items():
            if m.ret in ("Y", "A"):
                macs.append((name, m))

        top = max(self.max_y, self.max_a)
        for c in range(1, top + 1):
            if c == 1:
                termsY = [(("y",), YG)] + [(("k", n), np.full(NG, n, dtype=np.int64)) for n in range(4)]
                if c <= self.max_y:
                    for e, arr in termsY:
                        addY(1, e, arr)
                for name, m in macs:
                    if m.arity == 0:
                        e = ("app:" + name,)
                        if m.ret == "Y" and c <= self.max_y:
                            addY(1, e, np_eval(cl.xt[name], envY))
                        elif m.ret == "A" and c <= self.max_a:
                            addA(1, e, np_eval(cl.xt[name], envA))
                if c <= self.max_a:
                    for e, arr in list(self.Y[1]):
                        addA(1, e, up(arr))
                    addA(1, ("a",), AGRID.copy())
                continue
            # Y 만들기
            if c <= self.max_y:
                for kinds, (op,), f in base:
                    for comp in compositions(c - 1, len(kinds)):
                        for items in product(*[self.Y.get(cc, ()) for cc in comp]):
                            addY(c, (op,) + tuple(i[0] for i in items), f([i[1] for i in items]))
                for name, m in macs:
                    if m.ret != "Y" or m.arity == 0:
                        continue
                    for comp in compositions(c - 1, m.arity):
                        for items in product(*[self.Y.get(cc, ()) for cc in comp]):
                            addY(c, ("app:" + name,) + tuple(i[0] for i in items),
                                 np_eval(cl.xt[name], {**envY, "H": [i[1] for i in items]}))
            if c <= self.max_a:
                for e, arr in self.Y.get(c, ()):
                    addA(c, e, up(arr))
                for kinds, (op,), f in base:
                    for comp in compositions(c - 1, len(kinds)):
                        for items in product(*[self.A.get(cc, ()) for cc in comp]):
                            addA(c, (op,) + tuple(i[0] for i in items), f([i[1] for i in items]))
                for name, m in macs:
                    if m.arity == 0:
                        continue
                    if m.ret == "Y" and c <= self.max_y:
                        pools = None
                    # 반환이 A 인 개념 (인자는 Y 또는 A), 반환이 Y 인 개념은 A 풀에서도 A 인자로는 못 쓴다
                    if m.ret != "A":
                        continue
                    for comp in compositions(c - 1, m.arity):
                        lists = []
                        for k, cc in zip(m.kinds, comp):
                            lists.append(self.Y.get(cc, ()) if k == "Y" else self.A.get(cc, ()))
                        for items in product(*lists):
                            arrs = [up(i[1]) if k == "Y" else i[1] for k, i in zip(m.kinds, items)]
                            addA(c, ("app:" + name,) + tuple(i[0] for i in items),
                                 np_eval(cl.xt[name], {**envA, "H": arrs}))


# ------------------------------------------------------------------ 바깥 탐색
def _clipL(l):
    return l


def _rng_table():
    return [tuple(range(n)) for n in range(MAXLEN + 1)]


RNG = _rng_table()


def _bin(op):
    if op == "add":
        return lambda u, v: _c(u + v)
    if op == "sub":
        return lambda u, v: _c(u - v)
    if op == "mul":
        return lambda u, v: _c(u * v)
    if op == "gt":
        return lambda u, v: int(u > v)
    if op == "eq":
        return lambda u, v: int(u == v)
    return _mod


BIN = {op: _bin(op) for op in ARITH}


class Result:
    def __init__(self, expr, cost, evals, secs):
        self.expr, self.cost, self.evals, self.secs = expr, cost, evals, secs


def solve(task, cl: Compiled, bodies: Bodies, budget=60000, tmax=60.0, max_cost=13):
    """과제의 검색용 예시와 모두 일치하는 가장 작은 식을 찾는다. 못 찾으면 expr=None."""
    from .lang import run

    t0 = time.time()
    n = len(task.search_x)
    target = tuple(task.search_y)
    lib = cl.lib
    pool = {"I": defaultdict(list), "L": defaultdict(list)}
    seen = {"I": set(), "L": set()}
    evals = 0

    def goal(cat, e, vals):
        if cat != task.outcat or vals != target:
            return None
        got = run(e, lib, task.all_x)
        if got is not None and got == list(task.all_y):  # 검색용 + 확인용 예시 전부
            return e
        return None

    def add(c, cat, e, vals):
        r = goal(cat, e, vals)
        if r is not None:
            return r
        if vals in seen[cat]:
            return None
        seen[cat].add(vals)
        pool[cat][c].append((e, vals))
        return None

    xv = tuple(task.search_x)
    res = add(1, task.incat, ("x",), xv)
    if res:
        return Result(res, 1, 0, time.time() - t0)
    for k in range(4):
        res = add(1, "I", ("k", k), (k,) * n)
        if res:
            return Result(res, 1, 0, time.time() - t0)
    for name, m in lib.items():
        if m.ret in ("I", "L") and m.arity == 0:
            v = cl.fn[name]()
            v = tuple(v) if isinstance(v, list) else v
            res = add(1, m.ret, ("app:" + name,), (v,) * n)
            if res:
                return Result(res, 1, 0, time.time() - t0)

    def getbody(cat, c):
        return bodies.Y.get(c, ()) if cat == "Y" else bodies.A.get(c, ())

    # 생산 규칙: (인자 범주, 반환 범주, 값 계산, 식 만들기)
    prods = []
    for op in ARITH:
        f = BIN[op]
        prods.append((("I", "I"), "I", (lambda f: lambda a, b: tuple(map(f, a, b)))(f), op))
    prods.append((("I", "I", "I"), "I",
                  lambda c, t, e: tuple(tt if cc else ee for cc, tt, ee in zip(c, t, e)), "if"))
    prods.append((("L",), "I", lambda l: tuple(len(z) for z in l), "len"))
    prods.append((("L",), "I", lambda l: tuple(z[0] if z else 0 for z in l), "head"))
    prods.append((("I",), "L", lambda v: tuple(RNG[max(0, min(z, MAXLEN))] for z in v), "range"))
    prods.append((("Y", "L"), "L", lambda f, l: tuple(tuple([f(y) for y in z]) for z in l), "map"))
    prods.append((("Y", "L"), "L", lambda f, l: tuple(tuple([y for y in z if f(y)]) for z in l), "filter"))

    def _fold(f, init, l):
        out = []
        for i0, z in zip(init, l):
            a = i0
            for y in z:
                a = f(a, y)
            out.append(a)
        return tuple(out)

    prods.append((("A", "I", "L"), "I", _fold, "fold"))
    for name, m in lib.items():
        if m.ret in ("I", "L") and m.arity > 0:
            fn = cl.fn[name]
            kinds = m.kinds

            def ap(*args, fn=fn, kinds=kinds, ret=m.ret):
                out = []
                for j in range(n):
                    r = fn(*[(a[j] if k in "IL" else a) for a, k in zip(args, kinds)])
                    out.append(tuple(r) if isinstance(r, list) else r)
                return tuple(out)

            prods.append((kinds, m.ret, ap, "app:" + name))

    for c in range(2, max_cost + 1):
        for kinds, ret, f, op in prods:
            for comp in compositions(c - 1, len(kinds)):
                lists = []
                for k, cc in zip(kinds, comp):
                    lists.append(pool[k][cc] if k in "IL" else getbody(k, cc))
                if any(len(z) == 0 for z in lists):
                    continue
                for items in product(*lists):
                    evals += 1
                    if evals > budget or (evals & 1023) == 0 and time.time() - t0 > tmax:
                        return Result(None, None, evals, time.time() - t0)
                    args = []
                    for k, it in zip(kinds, items):
                        if k in "IL":
                            args.append(it[1])
                        else:
                            args.append(cl.body_fn(it[0], k))
                    try:
                        vals = f(*args)
                    except Exception:
                        continue
                    e = (op,) + tuple(it[0] for it in items)
                    res = add(c, ret, e, vals)
                    if res:
                        return Result(res, c, evals, time.time() - t0)
    return Result(None, None, evals, time.time() - t0)
