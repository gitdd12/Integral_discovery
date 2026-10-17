"""기초 학교 학습 (다음 기호 맞히기). 계산기가 채운 칸과 문제 문장은 손실에서 뺀다 (mask=0).

이어서 하기: --out 에 ckpt.pt 가 있으면 거기서부터 계속한다.
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from integral import vocab as V
from integral.model import GPT, Config


def batches(ids, mask, lens, tok_budget, rng):
    """길이가 비슷한 것끼리 묶어서 패딩 낭비를 줄인다."""
    order = np.argsort(lens + rng.random(len(lens)) * 6)
    out, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, int(lens[i]))
        if cur and m * (len(cur) + 1) > tok_budget:
            out.append(cur)
            cur, m = [], int(lens[i])
        cur.append(i)
        cur_max = m
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def make_batch(ids, mask, lens, idx):
    L = int(lens[idx].max())
    x = torch.from_numpy(ids[idx, :L].astype(np.int64))
    m = torch.from_numpy(mask[idx, :L].astype(np.bool_))
    return x, m


def loss_fn(model, x, m):
    logits = model(x[:, :-1])
    tgt = x[:, 1:]
    w = m[:, 1:]
    ls = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none")
    w = w.reshape(-1).float()
    return (ls * w).sum() / w.sum().clamp(min=1), w.sum()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/school_train.npz")
    ap.add_argument("--out", default="runs/school")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--tok_budget", type=int, default=16000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=768)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--init", default="", help="이 가중치에서 새로 시작(옵티마이저·스텝은 처음부터)")
    ap.add_argument("--max_steps", type=int, default=0, help="시험 실행용")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    d = np.load(a.data)
    ids, mask, lens = d["ids"], d["mask"], d["lens"].astype(np.int64)
    n = len(ids)
    hold = 2000                               # 맨 뒤 2000개는 검증 손실용
    tr = np.arange(n - hold)
    va = np.arange(n - hold, n)

    ck_path = os.path.join(a.out, "ckpt.pt")
    step, rng = 0, np.random.default_rng(0)
    if os.path.exists(ck_path):
        model, ck = GPT.load(ck_path)
        step = ck["step"]
    elif a.init:
        model, _ = GPT.load(a.init)
    else:
        model = GPT(Config(layers=a.layers, dim=a.dim, heads=a.heads, ff=a.ff))
    print(f"파라미터 {model.n_params()/1e6:.2f}M, 시작 step {step}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.01)
    if os.path.exists(ck_path) and "opt" in ck:
        opt.load_state_dict(ck["opt"])

    ep_batches = lambda e: batches(ids[tr], mask[tr], lens[tr], a.tok_budget, np.random.default_rng(100 + e))
    per_epoch = len(ep_batches(0))
    total = int(per_epoch * a.epochs)
    if a.max_steps:
        total = min(total, a.max_steps)
    print(f"에포크당 {per_epoch} 스텝, 총 {total} 스텝", flush=True)
    warm = 100

    def lr_at(s):
        if s < warm:
            return a.lr * (s + 1) / warm
        p = (s - warm) / max(1, total - warm)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1))))

    def val():
        model.eval()
        tot, cnt = 0.0, 0.0
        with torch.no_grad():
            for b in batches(ids[va], mask[va], lens[va], 32000, np.random.default_rng(1)):
                x, m = make_batch(ids[va], mask[va], lens[va], np.array(b))
                l, w = loss_fn(model, x, m)
                tot += float(l) * float(w)
                cnt += float(w)
        model.train()
        return tot / cnt

    model.train()
    t0, toks, run_loss = time.time(), 0, []
    cur_ep, bl = -1, []
    while step < total:
        e, k = divmod(step, per_epoch)
        if e != cur_ep:
            cur_ep, bl = e, ep_batches(e)
        idx = np.array(bl[k])
        x, m = make_batch(ids[tr], mask[tr], lens[tr], idx)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss, _ = loss_fn(model, x, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        toks += x.numel()
        run_loss.append(float(loss.detach()))
        if step % 25 == 0:
            print(f"step {step}/{total} loss {np.mean(run_loss[-25:]):.4f} lr {lr_at(step):.2e} "
                  f"{toks/(time.time()-t0):.0f} tok/s", flush=True)
        if step % a.save_every == 0 or step == total:
            vl = val()
            print(f"  검증 손실 {vl:.4f}", flush=True)
            model.save(ck_path, step=step, opt=opt.state_dict(), val_loss=vl)
    model.save(os.path.join(a.out, "school.pt"), step=step)
    print("끝", flush=True)
