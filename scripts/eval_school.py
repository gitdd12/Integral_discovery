"""기초 학교 시험 (PLAN.md 5.3). 학습 때 안 본 조합으로 greedy 풀이를 시켜 정답률을 잰다.

통과 기준: 모든 시험 묶음 정확도 90% 이상.
"""
import argparse
import json
import time

import numpy as np
import torch

from integral import school, vocab as V
from integral.generate import run_notebooks_cached
from integral.model import GPT


def evaluate(model, n, seed, verbose=0):
    rng = np.random.default_rng(seed)
    sets = {"train_like": school.sample_train, **school.TEST_SETS}
    res = {}
    for name, fn in sets.items():
        exs = [fn(rng) for _ in range(n)]
        nbs = [school.build(ex) for ex in exs]
        run_notebooks_cached(model, nbs)
        ok = [school.check(nb, ex) for nb, ex in zip(nbs, exs)]
        res[name] = float(np.mean(ok))
        if verbose:
            bad = [(nb, ex) for nb, ex, o in zip(nbs, exs, ok) if not o][:verbose]
            for nb, ex in bad:
                print(f"--- 틀림 [{name}] 정답={ex.target}\n{nb.render()}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/school/ckpt.pt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--show", type=int, default=2)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    torch.set_num_threads(4)
    model, ck = GPT.load(a.ckpt)
    t = time.time()
    res = evaluate(model, a.n, a.seed, verbose=a.show)
    print(f"\nstep {ck.get('step')}  ({time.time()-t:.0f}초)")
    for k, v in res.items():
        print(f"  {k:14s} {v*100:5.1f}%  {'통과' if v >= .9 else '미달'}")
    print("합격" if all(v >= .9 for v in res.values()) else "불합격")
    if a.json:
        json.dump(res, open(a.json, "w"), ensure_ascii=False, indent=1)
