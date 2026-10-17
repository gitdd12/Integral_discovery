"""기초 학교 학습 데이터를 만든다: data/school_train.npz  (ids, mask, 길이)"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from integral import school, vocab as V


def work(args):
    seed, n = args
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ex = school.sample_train(rng)
        ids, mask = school.render_example(ex)
        out.append((ids, mask))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/school_train.npz")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    t = time.time()
    chunk = 5000
    jobs = [(a.seed * 100003 + k, min(chunk, a.n - k * chunk)) for k in range((a.n + chunk - 1) // chunk)]
    with Pool(a.procs) as p:
        rows = [r for part in p.map(work, jobs) for r in part]
    L = max(len(i) for i, _ in rows)
    ids = np.full((len(rows), L), V.PAD_ID, dtype=np.int16)
    mask = np.zeros((len(rows), L), dtype=np.int8)
    for k, (i, m) in enumerate(rows):
        ids[k, : len(i)] = i
        mask[k, : len(m)] = m
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, ids=ids, mask=mask, lens=np.array([len(i) for i, _ in rows], dtype=np.int16))
    print(f"{len(rows)}개, 최대 길이 {L}, 평균 {np.mean([len(i) for i,_ in rows]):.1f}, {time.time()-t:.0f}초")
