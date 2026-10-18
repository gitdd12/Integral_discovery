"""기준 실험: 균등한 M개 지점의 높이를 받아 넓이를 내는 작은 MLP (일반적인 방법 1).

- 입력: 구간을 M등분한 각 조각의 한가운데 높이를 곡선의 최대 높이로 나눈 값 h~ (모양만 봄)
- 출력: q = 평균 높이 비율 (0~1).  넓이 = q × (최대 높이) × (b−a)   ← rect(가로,세로) 는 환경이 계산
- 더하기는 주지 않는다. 분석: ∂q/∂h~_i (진짜 평균 구조라면 모두 1/M)
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn
from integral import bdata

def sample_heights(grid, M, lo=0.0, hi=1.0):
    """grid:[B,513] (구간 [0,1]로 정규화된 위치에서의 높이). [lo,hi] 부분구간을 M등분한 중간 지점 높이 [B,M]."""
    G = grid.shape[1]
    pos = lo + (hi - lo) * (torch.arange(M).float() + 0.5) / M
    t = pos * (G - 1); i0 = t.floor().long().clamp(0, G - 2); w = t - i0.float()
    return grid[:, i0] * (1 - w) + grid[:, i0 + 1] * w

def prep(S, M):
    g = torch.tensor(S["grid"]); a = torch.tensor(S["a"]); b = torch.tensor(S["b"])
    h = sample_heights(g, M); s = h.max(1).values
    return dict(x=h / s[:, None], s=s, width=b - a, grid=g, a=a, b=b, y=torch.tensor(S["area"]).float().log())

def area_from_q(q, s, width):
    return torch.log(q.clamp_min(1e-6)) + torch.log(s) + torch.log(width)

class MLP(nn.Module):
    def __init__(self, M, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(M, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
    def forward(self, x):
        return torch.sigmoid(self.net(x))[:, 0]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=32); ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1); ap.add_argument("--out", default="runs/mlp1")
    ap.add_argument("--load", action="store_true", help="저장된 model.pt 로 분석만")
    ap.add_argument("--n_train", type=int, default=60000); ap.add_argument("--seed", type=int, default=0)
    A = ap.parse_args(); torch.manual_seed(A.seed); os.makedirs(A.out, exist_ok=True)
    S = bdata.make_sets(n_train=A.n_train, seed=A.seed)
    D = {k: prep(v, A.M) for k, v in S.items()}
    tr = D["train"]; n = len(tr["x"]); M = A.M
    m = MLP(M); opt = torch.optim.AdamW(m.parameters(), A.lr, weight_decay=A.wd)
    def err(d):
        with torch.no_grad(): return float((area_from_q(m(d["x"]), d["s"], d["width"]) - d["y"]).abs().mean())
    mid = {k: float((torch.log(d["x"].mean(1)) + torch.log(d["s"]) + torch.log(d["width"]) - d["y"]).abs().mean()) for k, d in D.items() if k != "train"}
    print("중점법(M=%d) 기준선:" % M, {k: round(v, 4) for k, v in mid.items()}, flush=True)
    log = []
    if A.load:
        m.load_state_dict(torch.load(f"{A.out}/model.pt"))
    for step in range(1, 1 if A.load else A.steps + 1):
        for g in opt.param_groups: g["lr"] = A.lr * min(1, step / 200) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / A.steps)))
        i = torch.randint(0, n, (A.bs,))
        loss = (area_from_q(m(tr["x"][i]), tr["s"][i], tr["width"][i]) - tr["y"][i]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0 or step == A.steps:
            r = dict(step=step, train=float(loss), **{k: round(err(d), 4) for k, d in D.items() if k != "train"})
            log.append(r); print(r, flush=True)
    if not A.load:
        torch.save(m.state_dict(), f"{A.out}/model.pt"); json.dump(dict(log=log, mid=mid), open(f"{A.out}/result.json", "w"), indent=1)

    # ---- 분석: 각 입력 높이의 가중치 ∂q/∂h~_i
    m.eval(); v = D["val"]; x = v["x"][:1000].clone().requires_grad_(True)
    q = m(x); q.sum().backward(); J = x.grad  # [N,M]
    print("\n[국소성/가중치] ∂q/∂h~_i  (이상적인 평균 구조: 모두 %.4f)" % (1 / M))
    print("  점별 평균:", np.round(J.mean(0).numpy(), 3).tolist())
    print("  전체 평균 %.4f, 점 간 표준편차 %.4f, 곡선 간 표준편차(평균) %.4f" % (J.mean(), J.mean(0).std(), J.std(0).mean()))
    # ---- 가법성: [a,c] 예측 vs [a,b]+[b,c]
    def pred_area(grid, lo, hi, width_full, a_unused=None):
        h = sample_heights(grid, M, lo, hi); s = h.max(1).values
        with torch.no_grad(): q = m(h / s[:, None])
        return q * s * width_full * (hi - lo)
    g = v["grid"]; W = v["width"]; rng = np.random.default_rng(0); p = torch.tensor(rng.uniform(0.2, 0.8, len(g))).float()
    whole = pred_area(g, 0.0, 1.0, W)
    left = torch.stack([pred_area(g[i:i+1], 0.0, float(p[i]), W[i:i+1]) for i in range(len(g))])[:, 0]
    right = torch.stack([pred_area(g[i:i+1], float(p[i]), 1.0, W[i:i+1]) for i in range(len(g))])[:, 0]
    rel = ((left + right - whole).abs() / whole)
    print("\n[가법성] |왼+오 − 전체| / 전체: 평균 %.4f, 중앙값 %.4f  (정답 쪽 해석 오차 중점법은 ≈%.4f)" % (rel.mean(), rel.median(), mid['val']))
