import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libworld.lang import Macro, expand, run, size  # noqa: E402
from libworld.learn import make_candidates, normalize, rewrite  # noqa: E402
from libworld.search import Bodies, Compiled, solve  # noqa: E402
from libworld.tasks import TEST, TRAIN, make_tasks  # noqa: E402

X, Y, A = ("x",), ("y",), ("a",)


def test_run_basic():
    s = ("fold", ("add", A, Y), ("k", 0), X)
    assert run(s, {}, [(1, 2, 3), (), (5,)]) == [6, 0, 5]
    assert run(("range", X), {}, [3]) == [(0, 1, 2)]
    assert run(("mod", X, ("k", 0)), {}, [5]) == [0]


def test_refs_expressible():
    for seed in (0, 1):
        tr, te = make_tasks(seed)
        assert len(tr) == 21 and len(te) == 24
        for t in tr + te:
            assert run(t.ref, {}, t.all_x) == list(t.all_y)


def test_solve_simple():
    tr, _ = make_tasks(0)
    cl = Compiled({})
    b = Bodies(cl)
    t = [t for t in tr if t.name == "제곱"][0]
    r = solve(t, cl, b, budget=50000)
    assert r.expr is not None and run(r.expr, {}, t.all_x) == list(t.all_y)


def test_macro_rewrite_preserves_meaning():
    tr, _ = make_tasks(0)
    cl = Compiled({})
    b = Bodies(cl)
    sols, outcat, fam = {}, {}, {}
    for t in tr:
        r = solve(t, cl, b, budget=100000)
        if r.expr is not None:
            sols[t.name] = normalize(r.expr)
            outcat[t.name], fam[t.name] = t.outcat, t.family
    cands = make_candidates(sols, outcat, fam, {})
    assert cands, "후보가 있어야 한다"
    byname = {t.name: t for t in tr}
    for c in cands[:10]:
        lib = {"f1": c.macro()}
        for k, e in sols.items():
            e2 = rewrite(e, outcat[k], "f1", lib["f1"], lib)
            assert run(e2, lib, byname[k].all_x) == list(byname[k].all_y)
            assert size(e2) <= size(e)


def test_method_hole_macro_works():
    # sumf(F, L) = fold(add(a, F), 0, L): 함수 인자를 받는 개념
    m = Macro("sumf", ("fold", ("add", ("a",), ("H", 0)), ("k", 0), ("H", 1)), ("Y", "L"), "I", "outer")
    lib = {"sumf": m}
    e = ("app:sumf", ("mul", Y, Y), ("range", X))
    assert run(e, lib, [4]) == [0 + 1 + 4 + 9]
    cl = Compiled(lib)
    assert cl.fn["sumf"](lambda y: y * y, [1, 2, 3]) == 14
