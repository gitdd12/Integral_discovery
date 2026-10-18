"""추상화(개념 후보 만들기·고르기)와 전체 루프.

개념 = 구멍이 있는 틀. 구멍 종류: 'I' 바깥 정수식, 'L' 바깥 목록식, 'Y' y만 쓰는 본문(함수 인자), 'A' a,y를 쓰는 본문(함수 인자).
풀이는 식 트리이고, 풀이 크기의 총합이 압축의 척도이다.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import random
import time
from collections import defaultdict

from .lang import ARITH, Macro, expand, is_leaf, run, show, size
from .search import Bodies, Compiled, lib_sig, solve
from .tasks import make_tasks

CACHE_DIR = os.environ.get("LIBWORLD_CACHE", "runs2/cache")


# ------------------------------------------------------------------ 맥락
def child_ctxs(e, ctx, lib):
    op = e[0]
    if op in ARITH or op == "if":
        return [ctx] * (len(e) - 1)
    if op in ("len", "head"):
        return ["L"]
    if op == "range":
        return ["I"]
    if op in ("map", "filter"):
        return ["Y", "L"]
    if op == "fold":
        return ["A", "I", "L"]
    if op.startswith("app:"):
        return list(lib[op[4:]].kinds)
    raise ValueError(op)


def has_a(e) -> bool:
    if e[0] == "a":
        return True
    if is_leaf(e):
        return False
    return any(has_a(c) for c in e[1:])


def walk(e, ctx, lib):
    """(부분식, 맥락)을 위에서 아래로."""
    yield e, ctx
    if is_leaf(e):
        return
    for c, cc in zip(e[1:], child_ctxs(e, ctx, lib)):
        yield from walk(c, cc, lib)


def free_vars(t, kinds, bound=frozenset()):
    op = t[0]
    if op in ("y", "a"):
        return set() if op in bound else {op}
    if op == "H":
        k = kinds[t[1]]
        need = {"Y": {"y"}, "A": {"a", "y"}}.get(k, set())
        return need - bound
    if is_leaf(t):
        return set()
    if op in ("map", "filter"):
        return free_vars(t[1], kinds, bound | {"y"}) | free_vars(t[2], kinds, bound)
    if op == "fold":
        return (free_vars(t[1], kinds, bound | {"a", "y"}) | free_vars(t[2], kinds, bound)
                | free_vars(t[3], kinds, bound))
    out = set()
    for c in t[1:]:
        out |= free_vars(c, kinds, bound)
    return out


# ------------------------------------------------------------------ 맞추기와 다시 쓰기
def match(t, e, ctx, m, lib, binds):
    if t[0] == "H":
        i, k = t[1], m.kinds[t[1]]
        if k in "IL":
            ok = ctx == k
        elif k == "Y":
            ok = ctx in ("Y", "A") and not has_a(e)
        else:
            ok = ctx == "A"
        if not ok:
            return False
        if binds[i] is None:
            binds[i] = e
            return True
        return binds[i] == e
    if t[0] != e[0] or len(t) != len(e):
        return False
    if t[0] == "k":
        return t[1] == e[1]
    if is_leaf(t):
        return True
    return all(match(tc, ec, c, m, lib, binds)
               for tc, ec, c in zip(t[1:], e[1:], child_ctxs(e, ctx, lib)))


def root_ok(m, ctx):
    if m.ret in ("I", "L"):
        return ctx == m.ret
    if m.ret == "Y":
        return ctx in ("Y", "A")
    return ctx == "A"


def rewrite(e, ctx, name, m, lib):
    """새 개념 name 으로 e 를 다시 쓴다 (바깥쪽부터)."""
    if root_ok(m, ctx):
        binds = [None] * m.arity
        if match(m.tmpl, e, ctx, m, lib, binds):
            return ("app:" + name,) + tuple(rewrite(b, k, name, m, lib) for b, k in zip(binds, m.kinds))
    if is_leaf(e):
        return e
    return (e[0],) + tuple(rewrite(c, cc, name, m, lib) for c, cc in zip(e[1:], child_ctxs(e, ctx, lib)))


def uses(e, pred, lib):
    """식 안에 pred(개념) 인 호출이 있는가."""
    if is_leaf(e):
        return False
    if e[0].startswith("app:") and pred(lib[e[0][4:]]):
        return True
    return any(uses(c, pred, lib) for c in e[1:])


def count_apps(e):
    if is_leaf(e):
        return 0
    return (1 if e[0].startswith("app:") else 0) + sum(count_apps(c) for c in e[1:])


def has_method_hole(m):
    return any(k in ("Y", "A") for k in m.kinds)


def depth(m, lib):
    d = 0

    def rec(t):
        nonlocal d
        if is_leaf(t):
            return
        if t[0].startswith("app:"):
            d = max(d, depth(lib[t[0][4:]], lib))
        for c in t[1:]:
            rec(c)

    rec(m.tmpl)
    return d + 1


# ------------------------------------------------------------------ 후보 만들기
def gen(e, ctx, lib, kmax, root):
    """e 를 일반화한 (틀, 구멍 수) 목록. 구멍 자리는 ('H', 부분식, 종류) 임시 표기."""
    out = []
    if not root and e[0] != "a":
        if ctx in ("I", "L", "Y"):
            kind = ctx
        else:
            kind = "A" if has_a(e) else "Y"
        out.append((("H", e, kind), 1))
    if is_leaf(e):
        if e[0] != "x":
            out.append((e, 0))
        return out
    parts = [gen(c, cc, lib, kmax, False) for c, cc in zip(e[1:], child_ctxs(e, ctx, lib))]
    combos = [((), 0)]
    for p in parts:
        new = []
        for cs, nh in combos:
            for t, k in p:
                if nh + k <= kmax:
                    new.append((cs + (t,), nh + k))
        combos = new
    for cs, nh in combos:
        out.append(((e[0],) + cs, nh))
    return out


def canon(t):
    """임시 표기를 번호 구멍으로. (틀, 구멍(부분식,종류) 목록)"""
    slots = {}

    def rec(x):
        if x[0] == "H":
            key = (x[1], x[2])
            if key not in slots:
                slots[key] = len(slots)
            return ("H", slots[key])
        if is_leaf(x):
            return x
        return (x[0],) + tuple(rec(c) for c in x[1:])

    tm = rec(t)
    holes = [None] * len(slots)
    for (sub, kind), i in slots.items():
        holes[i] = (sub, kind)
    return tm, holes


def tsize(t):
    return size(t)


class Cand:
    def __init__(self, tmpl, kinds, ret, approx, fams):
        self.tmpl, self.kinds, self.ret = tmpl, kinds, ret
        self.approx, self.fams = approx, fams
        self.gain = None

    def macro(self):
        return Macro("?", self.tmpl, self.kinds, self.ret, "outer")


def total_size(sols):
    return sum(size(e) for e in sols.values())


def make_candidates(sols, outcat, fam, lib, kmax=3, top=60, min_size=3, max_sub=13):
    """sols: 이름→식. outcat: 이름→출력 범주. fam: 이름→계열."""
    table = {}
    gen_cache = {}
    for name, e in sols.items():
        for sub, ctx in walk(e, outcat[name], lib):
            if is_leaf(sub):
                continue
            sz = size(sub)
            if sz < min_size or sz > max_sub:
                continue
            ck = (sub, ctx)
            if ck not in gen_cache:
                res = []
                for t, nh in gen(sub, ctx, lib, kmax, True):
                    tm, holes = canon(t)
                    kinds = tuple(k for _, k in holes)
                    fv = free_vars(tm, kinds)
                    if ctx in ("I", "L"):
                        if fv:
                            continue
                        ret = ctx
                    else:
                        if not fv:
                            continue
                        ret = "A" if "a" in fv else "Y"
                    tsz = tsize(tm)
                    if tsz < 3 and not (tsz >= 2 and nh == 0):
                        continue
                    saving = sz - (1 + sum(size(s) for s, _ in holes))
                    if saving <= 0:
                        continue
                    res.append(((tm, kinds, ret), saving, tsz))
                gen_cache[ck] = res
            for key, saving, tsz in gen_cache[ck]:
                row = table.get(key)
                if row is None:
                    table[key] = [saving, 1, {fam[name]}, tsz]
                else:
                    row[0] += saving
                    row[1] += 1
                    row[2].add(fam[name])
    cands = []
    for (tm, kinds, ret), (sv, cnt, fams, tsz) in table.items():
        if cnt >= 2 and sv - tsz > 0:
            cands.append(Cand(tm, kinds, ret, sv - tsz, fams))
    cands.sort(key=lambda c: (-c.approx, repr(c.tmpl)))
    base = total_size(sols)
    out = []
    for c in cands[:top]:
        m = c.macro()
        lib2 = dict(lib)
        lib2["?"] = m
        new = {k: rewrite(e, outcat[k], "?", m, lib2) for k, e in sols.items()}
        c.gain = base - total_size(new) - tsize(c.tmpl)
        if c.gain > 0:
            out.append(c)
    out.sort(key=lambda c: (-c.gain, repr(c.tmpl)))
    return out


def normalize(e):
    """교환법칙이 성립하는 연산의 인자를 정해진 순서로 (같은 모양의 풀이가 같은 식이 되게)."""
    if is_leaf(e):
        return e
    cs = [normalize(c) for c in e[1:]]
    if e[0] in ("add", "mul", "eq") and repr(cs[0]) > repr(cs[1]):
        cs = [cs[1], cs[0]]
    return (e[0],) + tuple(cs)


# ------------------------------------------------------------------ 풀이 캐시
_COMPILED = {}


def compiled_for(lib):
    sig = lib_sig(lib)
    ent = _COMPILED.get(sig)
    if ent is None:
        if len(_COMPILED) > 6:
            _COMPILED.pop(next(iter(_COMPILED)))
        cl = Compiled(lib)
        ent = (cl, Bodies(cl))
        _COMPILED[sig] = ent
    return ent


def solve_cached(task, lib, seed, budget):
    key = hashlib.sha1(repr((task.name, seed, lib_sig(lib), budget, "v1")).encode()).hexdigest()
    path = os.path.join(CACHE_DIR, key[:2], key + ".pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                e, c, ev = pickle.load(f)
            return (normalize(e) if e is not None else None, c, ev)
        except Exception:
            pass
    cl, bodies = compiled_for(lib)
    r = solve(task, cl, bodies, budget=budget)
    out = (r.expr, r.cost, r.evals)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, path)
    return (normalize(r.expr) if r.expr is not None else None, r.cost, r.evals)


# ------------------------------------------------------------------ 기준별 선택
def new_name(lib):
    return f"f{len(lib) + 1}"


def memorize_candidates(sols, tasks_by_name, lib):
    out = []
    for name, e in sols.items():
        if size(e) < 5 or e[0].startswith("app:"):
            continue
        t = tasks_by_name[name]
        kind = t.incat

        def sub(x):
            if x[0] == "x":
                return ("H", 0)
            if is_leaf(x):
                return x
            return (x[0],) + tuple(sub(c) for c in x[1:])

        out.append(Cand(sub(e), (kind,), t.outcat, 0, {t.family}))
    return out


def choose(criterion, cands, ctx):
    """cands: 압축 이득이 큰 순서의 후보. ctx 에는 rng, trial(c)->새로 푼 수."""
    if not cands:
        return None
    if criterion == "compress":
        return cands[0]
    if criterion == "random":
        return ctx["rng"].choice(cands)
    if criterion == "general":
        multi = [c for c in cands if len(c.fams) >= 2]
        if not multi:
            return None
        return max(multi, key=lambda c: (len(c.fams), c.gain))
    if criterion == "progress":
        best, best_n = None, 0
        for c in cands[:8]:
            n = ctx["trial"](c)
            if n > best_n:
                best, best_n = c, n
        return best
    raise ValueError(criterion)


def run_experiment(criterion, seed, iters=8, budget=150000, per_iter=2, log=print):
    train, test = make_tasks(seed)
    byname = {t.name: t for t in train}
    fam = {t.name: t.family for t in train}
    outcat = {t.name: t.outcat for t in train}
    rng = random.Random(f"{criterion}-{seed}")
    lib: dict = {}
    sols: dict = {}
    t00 = time.time()
    hist = []

    def wake():
        for t in train:
            if t.name in sols:
                continue
            e, c, ev = solve_cached(t, lib, seed, budget)
            if e is not None:
                sols[t.name] = e

    def evaluate(it):
        rows = []
        for t in test:
            e, c, ev = solve_cached(t, lib, seed, budget)
            rows.append({"task": t.name, "solved": e is not None, "cost": c,
                         "expr": show(e) if e is not None else None,
                         "method": bool(e is not None and uses(e, has_method_hole, lib)),
                         "app": bool(e is not None and count_apps(e) > 0)})
        names = list(lib)
        sols_ok = sum(run(e, lib, byname[k].all_x) == list(byname[k].all_y) for k, e in sols.items())
        rec = {
            "sols_ok": sols_ok,
            "iter": it, "lib_size": len(lib), "lib_depth": max([depth(m, lib) for m in lib.values()] or [0]),
            "methods": sum(has_method_hole(m) for m in lib.values()),
            "train_solved": len(sols), "train_size": total_size(sols),
            "test_solved": sum(r["solved"] for r in rows),
            "test_method_used": sum(r["method"] for r in rows),
            "test_app_used": sum(r["app"] for r in rows),
            "test_mean_cost": (sum(r["cost"] for r in rows if r["solved"]) / max(1, sum(r["solved"] for r in rows))),
            "lib": [{"name": n, "tmpl": show(lib[n].tmpl), "kinds": "".join(lib[n].kinds), "ret": lib[n].ret}
                    for n in names],
            "test": rows, "secs": round(time.time() - t00, 1),
        }
        hist.append(rec)
        assert sols_ok == len(sols), "다시 쓴 풀이가 틀어졌다"
        log(f"[{criterion} s{seed}] it{it} lib={len(lib)} train={len(sols)}/{len(train)} "
            f"test={rec['test_solved']}/{len(test)} method_used={rec['test_method_used']} "
            f"({rec['secs']}s)")

    wake()
    evaluate(0)
    for it in range(1, iters + 1):
        for _ in range(per_iter):
            if criterion == "none":
                break
            if criterion == "memorize":
                cands = memorize_candidates(sols, byname, lib)
                cands = [c for c in cands if c.tmpl not in {m.tmpl for m in lib.values()}]
                c = rng.choice(cands) if cands else None
            else:
                cands = make_candidates(sols, outcat, fam, lib)

                def trial(c, lib=lib):
                    lib2 = dict(lib)
                    lib2[new_name(lib)] = c.macro()
                    n = 0
                    for t in train:
                        if t.name not in sols and solve_cached(t, lib2, seed, budget)[0] is not None:
                            n += 1
                    return n

                c = choose(criterion, cands, {"rng": rng, "trial": trial})
            if c is None:
                break
            name = new_name(lib)
            m = c.macro()
            lib[name] = m
            for k in list(sols):
                sols[k] = rewrite(sols[k], outcat[k], name, m, lib)
        wake()
        evaluate(it)
    return hist
