"""DESIGN2 실험 실행: 기준 × 시드. 결과는 runs2/<기준>_s<시드>.json (이미 있으면 건너뜀)."""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libworld.learn import run_experiment  # noqa: E402

ALL = ["none", "compress", "progress", "general", "random", "memorize"]


def job(a):
    crit, seed, out, budget, iters = a
    path = os.path.join(out, f"{crit}_s{seed}.json")
    if os.path.exists(path):
        return path
    t = time.time()
    hist = run_experiment(crit, seed, iters=iters, budget=budget, log=lambda s: print(s, flush=True))
    with open(path + ".tmp", "w") as f:
        json.dump({"criterion": crit, "seed": seed, "budget": budget, "secs": time.time() - t, "hist": hist},
                  f, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--criteria", nargs="*", default=ALL)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs2")
    ap.add_argument("--budget", type=int, default=150000)
    ap.add_argument("--iters", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.environ.setdefault("LIBWORLD_CACHE", os.path.join(a.out, "cache"))
    jobs = [(c, s, a.out, a.budget, a.iters) for s in range(a.seeds) for c in a.criteria]
    with Pool(a.workers) as p:
        for path in p.imap_unordered(job, jobs):
            print("done", path, flush=True)
