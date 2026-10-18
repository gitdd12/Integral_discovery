"""B④ 학습: 연속 사고 + 점 질의. 손실 = |log 예측 − log 정답|. 주기적으로 시험 곡선 오차와 기준선을 기록한다."""
import argparse, json, math, os, time
import numpy as np, torch
from integral import bdata
from integral.bmodel import QueryNet, baselines

def tens(S):
    return (torch.tensor(S["a"]), torch.tensor(S["b"]), torch.tensor(S["grid"]),
            torch.tensor(S["area"]).float().log())

@torch.no_grad()
def evaluate(m, sets, Ks, fixed_u=None):
    """sets 마다 각 K 에서의 평균 |log 오차|. 반환 {set: {K: err}}."""
    m.eval(); out = {}
    Kmax = max(Ks)
    for name, S in sets.items():
        a, b, g, y = tens(S)
        errs = {K: [] for K in Ks}
        for i in range(0, len(a), 250):
            sl = slice(i, i + 250)
            fu = None if fixed_u is None else fixed_u(len(a[sl]), Kmax)
            e, _ = m(a[sl], b[sl], g[sl], K=Kmax, fixed_u=fu)
            for K in Ks:
                errs[K].append((e[:, K] - y[sl]).abs())
        out[name] = {K: float(torch.cat(v).mean()) for K, v in errs.items()}
    m.train(); return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/b1"); ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--n_train", type=int, default=60000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=500); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rect", action="store_true", help="rect(가로,세로)=가로×세로 도구를 줌")
    ap.add_argument("--norm", action="store_true", help="첫 질의 높이 대비 비율로 입력 정규화, 폭×높이는 환경이 곱함")
    ap.add_argument("--fixed", default="", help="'grid' = 균등 격자 대조군, 'random' = 무작위 고정 위치 대조군")
    a_ = ap.parse_args()
    torch.set_num_threads(a_.threads); torch.manual_seed(a_.seed)
    os.makedirs(a_.out, exist_ok=True)
    t0 = time.time(); S = bdata.make_sets(n_train=a_.n_train, seed=a_.seed)
    print(f"data {time.time()-t0:.0f}s", flush=True)
    a, b, g, y = tens(S["train"]); n = len(a)
    ev = {k: v for k, v in S.items() if k != "train"}
    m = QueryNet(K=a_.K, d=a_.d, layers=a_.layers, rect=a_.rect, norm=a_.norm)
    print("params", sum(p.numel() for p in m.parameters()), flush=True)
    opt = torch.optim.AdamW(m.parameters(), a_.lr, betas=(0.9, 0.98), weight_decay=a_.wd)
    K = a_.K
    if a_.fixed == "grid":
        fx = lambda B, K: ((torch.arange(K).float() + .5) / K).expand(B, K)
    elif a_.fixed == "random":
        rr = torch.rand(K); fx = lambda B, K: rr.expand(B, K)
    else:
        fx = None
    EK = [4, 8, 16, 32, 64, 128]       # 평가하는 질의 횟수 (학습은 최대 K=16; 32 이상은 학습에 없던 횟수)
    base = {}
    for name, Sv in ev.items():
        aa, bb, gg, yy = tens(Sv); base[name] = {}
        for k in EK:
            mid, trap = baselines(gg, aa, bb, k)
            base[name][k] = dict(mid=float((mid.log() - yy).abs().mean()), trap=float((trap.log() - yy).abs().mean()))
    json.dump(base, open(f"{a_.out}/baselines.json", "w"), indent=1)
    log = open(f"{a_.out}/log.jsonl", "a")
    for step in range(1, a_.steps + 1):
        lr = a_.lr * min(1, step / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a_.steps)))
        for gp in opt.param_groups: gp["lr"] = lr
        i = torch.randint(0, n, (a_.bs,))
        e, _ = m(a[i], b[i], g[i], K=K, fixed_u=None if fx is None else fx(a_.bs, K))
        loss = (e[:, 1:] - y[i][:, None]).abs().mean()          # 모든 시점의 오차를 합산
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % a_.eval_every == 0 or step == a_.steps:
            full = step == a_.steps or step % (a_.eval_every * 4) == 0
            r = dict(step=step, train=float(loss.detach()), secs=round(time.time()-t0),
                     ev=evaluate(m, ev if full else {k: ev[k] for k in ("val", "test_stair")},
                                 EK if full else [4, 8, 16], fx))
            log.write(json.dumps(r) + "\n"); log.flush(); print(r, flush=True)
            torch.save({"model": m.state_dict(), "args": vars(a_)}, f"{a_.out}/model.pt")
    print("끝", flush=True)
