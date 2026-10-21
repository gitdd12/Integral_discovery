"""눈 점검 (DESIGN4): 플래그 열을 하나씩 받는 LSTM 눈이 '같은 개수인가' 결과 보상만으로 세는지,
훈련(개수 0~20)에 없던 21~40에서도 맞는지.
사용: python scripts/eye_check.py <변형: lstm|lstm_mid> <시드> [단계수]   결과: runs4/eye/<변형>_s<시드>.json
정답 개수는 학습에 쓰지 않고(결과 보상만), 평가·분석에만 쓴다."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from countworld.scene import pair_matrix  # noqa: E402

variant, seed = sys.argv[1], int(sys.argv[2])
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
torch.set_num_threads(int(os.environ.get("THREADS", "1")))
out = os.environ.get("OUT", "runs4/eye")
os.makedirs(out, exist_ok=True)
logf = open(f"{out}/{variant}_s{seed}.log", "a", encoding="utf-8")
TRAIN_MAX, TEST_MAX = int(os.environ.get("TRAIN_MAX", 20)), int(os.environ.get("TEST_MAX", 40))
TMIN, TMAX = int(os.environ.get("TMIN", TRAIN_MAX)), int(os.environ.get("TMAX", TEST_MAX + 10))
D, B, CUTS = 64, 256, 4


def log(s):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


class Eye(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(2, 16)
        self.cell = nn.LSTMCell(16, D)
        self.cmp = nn.Sequential(nn.Linear(8 * D, 64), nn.ReLU(), nn.Linear(64, 1))

    def states(self, flags):                       # (B,T) 0/1 → (B,T,2D) [h,c], 한 토큰씩
        Bn, T = flags.shape
        h = c = torch.zeros(Bn, D)
        e = self.emb(flags)
        outs = []
        for t in range(T):
            h, c = self.cell(e[:, t], (h, c))
            outs.append(torch.cat([h, c], -1))
        return torch.stack(outs, 1)

    def compare(self, sa, sb):
        return self.cmp(torch.cat([sa, sb, sa * sb, (sa - sb).abs()], -1)).squeeze(-1)


def make_streams(rng, counts, T, Tpad):
    Bn = len(counts)
    keys = rng.random((Bn, Tpad))
    keys = np.where(np.arange(Tpad)[None] < T[:, None], keys, 2.0)
    rank = np.argsort(np.argsort(keys, 1), 1)
    return (rank < counts[:, None]).astype(np.int64)


def batch(rng, beta):
    ans = rng.random(B) < 0.5
    P = pair_matrix(TRAIN_MAX + 1, beta, "train")
    flat = rng.choice(P.size, size=B, p=P.ravel())
    ad, bd = flat // (TRAIN_MAX + 1), flat % (TRAIN_MAX + 1)
    sa = rng.integers(0, TRAIN_MAX + 1, B)
    a, b = np.where(ans, sa, ad), np.where(ans, sa, bd)
    T1, T2 = rng.integers(TMIN, TMAX + 1, B), rng.integers(TMIN, TMAX + 1, B)
    return a, b, T1, T2, make_streams(rng, a, T1, TMAX), make_streams(rng, b, T2, TMAX), ans


@torch.no_grad()
def eval_adjacent(eye, rng, n_per=64):
    """n 대 n+1 균형 정확도 (같은 쌍 50% + 다른 쌍 50%), T=50 고정."""
    acc = []
    for n in range(TEST_MAX):
        a = np.full(n_per, n)
        b = np.where(np.arange(n_per) % 2 == 0, n, n + 1)
        T = np.full(n_per, TMAX)
        sa = eye.states(torch.from_numpy(make_streams(rng, a, T, TMAX)))[:, -1]
        sb = eye.states(torch.from_numpy(make_streams(rng, b, T, TMAX)))[:, -1]
        pred = eye.compare(sa, sb) > 0
        acc.append(float((pred.numpy() == (a == b)).mean()))
    return acc


@torch.no_grad()
def probe(eye, rng, n_per=32):
    """상태 → 개수 선형 회귀를 0~20으로 적합, 21~40에서 R² (정답 개수는 분석에만)."""
    X, y = [], []
    for n in range(TEST_MAX + 1):
        s = eye.states(torch.from_numpy(make_streams(rng, np.full(n_per, n), np.full(n_per, TMAX), TMAX)))[:, -1].numpy()
        X.append(s); y += [n] * n_per
    X, y = np.concatenate(X), np.array(y, float)
    res = {}
    for name, sl in (("h", slice(0, D)), ("c", slice(D, 2 * D)), ("hc", slice(0, 2 * D))):
        A = np.c_[X[:, sl], np.ones(len(X))]
        tr, te = y <= TRAIN_MAX, y > TRAIN_MAX
        w = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
        pr = A @ w
        r2 = lambda m: float(1 - ((y[m] - pr[m]) ** 2).sum() / max(((y[m] - y[m].mean()) ** 2).sum(), 1e-9))
        res[name] = dict(r2_train=r2(tr), r2_extrap=r2(te))
    return res


torch.manual_seed(seed)
rng = np.random.default_rng(seed)
erng = np.random.default_rng(999 + seed)
eye = Eye()
opt = torch.optim.AdamW(eye.parameters(), lr=1e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(1.0, i / steps))))
hist = []
for step in range(steps):
    beta = 2.5 * min(1.0, step / (0.5 * steps))
    a, b, T1, T2, fa, fb, ans = batch(rng, beta)
    S = eye.states(torch.from_numpy(np.concatenate([fa, fb])))
    SA, SB = S[:B], S[B:]
    idx = torch.arange(B)
    pairs = [(SA[idx, torch.from_numpy(T1 - 1)], SB[idx, torch.from_numpy(T2 - 1)], torch.from_numpy(ans))]
    if variant == "lstm_mid":                      # 같은 위치 t 에서 '지금까지 개수가 같은가' (접두 개수는 환경이 계산)
        ca, cb = np.cumsum(fa, 1), np.cumsum(fb, 1)
        tmax = np.minimum(T1, T2)
        for _ in range(CUTS):
            t = (rng.random(B) * tmax).astype(int)
            pairs.append((SA[idx, torch.from_numpy(t)], SB[idx, torch.from_numpy(t)],
                          torch.from_numpy(ca[np.arange(B), t] == cb[np.arange(B), t])))
    sigma = 1.0 + (0.3 - 1.0) * min(1.0, step / (0.5 * steps))
    loss, rs = 0.0, []
    for sa, sb, same in pairs:
        m = eye.compare(sa, sb)
        z = (m + sigma * torch.randn_like(m)).detach()
        r = torch.where(same, F.logsigmoid(z), F.logsigmoid(-z))
        adv = r - r.mean()
        loss = loss - (adv.detach() * (-((z - m) ** 2) / (2 * sigma ** 2))).mean() / len(pairs)
        rs.append(r.mean().item())
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(eye.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 500 == 0 or step == steps - 1:
        adj = eval_adjacent(eye, erng)
        tr, ex = float(np.mean(adj[:TRAIN_MAX])), float(np.mean(adj[TRAIN_MAX:]))
        hist.append((step, tr, ex))
        log(f"[{variant} s{seed}] step {step} 점수 {rs[0]:.3f} 훈련범위 이웃 {tr:.3f} 외삽(21~40) 이웃 {ex:.3f}")
adj = eval_adjacent(eye, erng)
res = dict(variant=variant, seed=seed, steps=steps, adjacent=adj, train_adj=float(np.mean(adj[:TRAIN_MAX])),
           extrap_adj=float(np.mean(adj[TRAIN_MAX:])), probe=probe(eye, erng), hist=hist)
json.dump(res, open(f"{out}/{variant}_s{seed}.json", "w"), ensure_ascii=False)
log("RESULT " + json.dumps({k: res[k] for k in ("train_adj", "extrap_adj", "probe")}))
