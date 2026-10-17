"""2단계 학습 실행기 (코랩에서 쓰도록 이어하기·중간 저장 지원).

  python scripts/train_rl.py --init weights/school.pt --out /content/drive/MyDrive/integral_runs/run1
저장 (--save_every 바퀴마다, 그리고 끝날 때):
  ckpt.pt        가중치 + 옵티마이저 + 바퀴 번호 + 난수 상태  (끊겨도 여기서부터 다시)
  scores.jsonl   바퀴마다 점수 기록
  thoughts/      --thought_every 바퀴마다 생각 기록(메모장 전체) 샘플
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from integral import rl
from integral.model import GPT

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="weights/school.pt")
    ap.add_argument("--out", default="runs/rl")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--n_curves", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--school_frac", type=float, default=0.2)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--thought_every", type=int, default=1000)
    ap.add_argument("--thought_first", type=int, default=20, help="처음 몇 바퀴는 자주 저장")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(a.threads)
    os.makedirs(os.path.join(a.out, "thoughts"), exist_ok=True)
    cfg = rl.RLConfig(n_curves=a.n_curves, k=a.k, keep=a.keep, temperature=a.temperature,
                      lr=a.lr, school_frac=a.school_frac)
    ck_path = os.path.join(a.out, "ckpt.pt")
    rnd = 0
    if os.path.exists(ck_path):
        model, ck = GPT.load(ck_path, device)
        rnd = ck["round"]
        rng = np.random.default_rng(a.seed)
        rng.bit_generator.state = ck["rng"]
        print(f"이어서 시작: 바퀴 {rnd}", flush=True)
    else:
        model, ck = GPT.load(a.init, device)
        rng = np.random.default_rng(a.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    if "opt" in ck:
        opt.load_state_dict(ck["opt"])
    gen = torch.Generator(device="cpu")
    gen.manual_seed(a.seed + rnd)

    def save():
        model.save(ck_path, round=rnd, opt=opt.state_dict(), rng=rng.bit_generator.state)

    log = open(os.path.join(a.out, "scores.jsonl"), "a")
    while rnd < a.rounds:
        t = time.time()
        nbs, scores, curves, groups = rl.collect(model, rng, cfg, device, gen)
        st = rl.round_stats(nbs, scores)
        seqs, masks = [], []
        for g in groups:
            for i in rl.select([nbs[j] for j in g], [scores[j] for j in g], cfg):
                seqs.append(nbs[g[i]].ids)
                masks.append(nbs[g[i]].mask)
        n_kept = len(seqs)
        if n_kept:
            ss, ms = rl.school_batch(rng, max(1, int(round(n_kept * cfg.school_frac / (1 - cfg.school_frac)))))
            loss = rl.sft_step(model, opt, seqs + ss, masks + ms, device)
        else:
            loss = None
        rnd += 1
        rec = dict(round=rnd, kept=n_kept, loss=loss, secs=round(time.time() - t, 1), **st)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)
        if rnd <= a.thought_first or rnd % a.thought_every == 0:
            json.dump(rl.thought_sample(nbs, scores, curves, groups), open(
                os.path.join(a.out, "thoughts", f"round{rnd:06d}.json"), "w"), ensure_ascii=False, indent=1)
        if rnd % a.save_every == 0:
            save()
    save()
    model.save(os.path.join(a.out, "final.pt"), round=rnd)
    print("끝", flush=True)
