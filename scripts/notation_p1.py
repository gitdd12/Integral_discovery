"""수 표기 예비 실험 P1 (DESIGN5_notation.md): 크기값(n/100)을 받은 화자가 기호 쪽지를 쓰고,
읽기 잡음(같은 기호 연속 길이가 부정확하게 읽힘)이 있는 판정자가 '같은가'를 맞힌다. 결과 보상만(강화학습).
사용: python scripts/notation_p1.py <조건 C0|C3> <시드> [단계수] [람다]   결과: runs5/p1/<조건>_s<시드>.json
환경변수: OUT, THREADS, LAM(기본 0.01), KAPPA(기본 0.25), LMAX(기본 8)"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from countworld.train import usage_entropy  # noqa: E402

cond, seed = sys.argv[1], int(sys.argv[2])
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
LAM = float(sys.argv[4]) if len(sys.argv) > 4 else float(os.environ.get("LAM", 0.01))
KAPPA = float(os.environ.get("KAPPA", 0.25))
LMAX = int(os.environ.get("LMAX", 8))
V, NMAX, D, B, LP = int(os.environ.get("V", 2)), 100, 64, 512, 24
torch.set_num_threads(int(os.environ.get("THREADS", "1")))
out = os.environ.get("OUT", "runs5/p1")
os.makedirs(out, exist_ok=True)
tag = f"{cond}_l{LAM}_s{seed}"
logf = open(f"{out}/{tag}.log", "a", encoding="utf-8")


def log(s):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


ALL = np.arange(NMAX + 1)
ho = np.random.default_rng(12345).random(NMAX + 1) < 0.4          # 시험 전용 개수 (고정)
TRAIN = ALL[~ho] if cond == "C0" else ALL
HELD = ALL[ho] if cond == "C0" else np.array([], int)


class Speaker(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(1, D), nn.Tanh(), nn.Linear(D, D), nn.Tanh())
        self.emb = nn.Embedding(V + 2, D)                 # 0..V-1 기호, V=종료, V+1=시작
        self.gru = nn.GRUCell(D, D)
        self.head = nn.Linear(D, V + 1)

    def forward(self, x, mode="sample", min_len=1):
        Bn = x.shape[0]
        h = self.inp(x[:, None])
        inp = self.emb(torch.full((Bn,), V + 1))
        alive = torch.ones(Bn, dtype=torch.bool)
        syms = torch.full((Bn, LMAX), V + 1, dtype=torch.long)
        logp = torch.zeros(Bn); ent = torch.zeros(Bn); probs0 = None
        for t in range(LMAX):
            h = self.gru(inp, h)
            logits = self.head(h)
            if t == 0:
                probs0 = F.softmax(logits, -1)
            if t < min_len:
                logits = logits.clone(); logits[:, V] = -1e9
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample() if mode == "sample" else logits.argmax(-1)
            m = alive.float()
            logp = logp + dist.log_prob(a) * m
            ent = ent + dist.entropy() * m
            syms[:, t] = torch.where(alive, a, torch.full_like(a, V + 1))
            alive = alive & (a != V)
            inp = self.emb(a)
        lens = (syms < V).sum(1).float()
        return dict(syms=syms, logp=logp, ent=ent, lens=lens, probs0=probs0)


class Judge(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V + 1, D)                 # V = 패딩
        self.gru = nn.GRU(D, D, batch_first=True)
        self.scene = nn.Sequential(nn.Linear(1, D), nn.Tanh(), nn.Linear(D, D), nn.Tanh())
        self.cmp = nn.Sequential(nn.Linear(4 * D, D), nn.ReLU(), nn.Linear(D, 1))

    def forward(self, ids, lens, x2):
        o, _ = self.gru(self.emb(ids))
        m = o[torch.arange(ids.shape[0]), (lens - 1).clamp(min=0)]
        c = self.scene(x2[:, None])
        return self.cmp(torch.cat([c, m, c * m, (c - m).abs()], -1)).squeeze(-1)


def read_noise(syms, rng, kappa=None):
    """같은 기호 연속 구간 길이 r 이 r' = max(1, round(r + N(0,(κ r)^2))) 로 읽힌다."""
    kappa = KAPPA if kappa is None else kappa
    s = syms.numpy()
    Bn = s.shape[0]
    ids = np.full((Bn, LP), V, dtype=np.int64)
    lens = np.ones(Bn, dtype=np.int64)
    for i in range(Bn):
        seq = []
        for v in s[i]:
            if v >= V:
                break
            seq.append(int(v))
        out_seq, j = [], 0
        while j < len(seq):
            k = j
            while k < len(seq) and seq[k] == seq[j]:
                k += 1
            r = k - j
            rp = max(1, int(round(r + kappa * r * rng.standard_normal()))) if r >= 2 else 1
            out_seq += [seq[j]] * rp
            j = k
        out_seq = out_seq[:LP]
        if not out_seq:
            out_seq = [0]
        ids[i, :len(out_seq)] = out_seq
        lens[i] = len(out_seq)
    return torch.from_numpy(ids[:, :int(lens.max())]), torch.from_numpy(lens)


def sample_pairs(rng, beta):
    ans = rng.random(B) < 0.5
    a = rng.choice(TRAIN, B)
    g = np.abs(TRAIN[None, :] - a[:, None]).astype(float)
    W = np.where(g > 0, np.maximum(g, 1.0) ** (-beta), 0.0)
    W = W / W.sum(1, keepdims=True)
    cum = W.cumsum(1)
    bd = TRAIN[np.minimum((cum < rng.random((B, 1))).sum(1), len(TRAIN) - 1)]
    b = np.where(ans, a, bd)
    return a, b, ans


@torch.no_grad()
def eval_pairs(sp, jd, rng, S, restrict, draws=6):
    """S 의 각 개수 n 에 대해 같은 쌍(n,n)과 다른 쌍(n, n±1 또는 ±2) 균형 정확도 (읽기 잡음 포함)."""
    sy = sp(torch.from_numpy(ALL / NMAX).float(), mode="greedy")["syms"]
    acc = []
    okset = set(restrict.tolist()) if restrict is not None else set(ALL.tolist())
    for _ in range(draws):
        ids, lens = read_noise(sy, rng)
        for n in S:
            for d in (0, 1, -1, 2, -2):
                b = n + d
                if b < 0 or b > NMAX or b not in okset:
                    continue
                m = jd(ids[n:n + 1], lens[n:n + 1], torch.tensor([b / NMAX]).float())
                acc.append(((m.item() > 0) == (d == 0), d == 0))
    same = [x for x, s in acc if s]; diff = [x for x, s in acc if not s]
    return 0.5 * (float(np.mean(same)) + float(np.mean(diff))) if same and diff else float("nan")


@torch.no_grad()
def analyze(sp):
    sy = sp(torch.from_numpy(ALL / NMAX).float(), mode="greedy")["syms"].numpy()
    code = {int(n): "".join(str(int(v)) for v in row if v < V) for n, row in zip(ALL, sy)}
    for n, row in zip(ALL, sy):                                   # 종료 기호 뒤는 버림
        s = ""
        for v in row:
            if v >= V:
                break
            s += str(int(v))
        code[int(n)] = s
    lens = np.array([len(code[n]) for n in ALL], float)
    distinct = len(set(code.values())) / len(ALL)
    max_run = float(np.mean([max((len(m.group()) for m in __import__("re").finditer(r"0+|1+", c)), default=0) for c in code.values()]))
    binexact = float(np.mean([(c != "" and (int(c, 2) == n or int(c[::-1], 2) == n)) for n, c in code.items()]))
    bits = np.array([[(n >> k) & 1 for k in range(7)] for n in ALL])
    mi = []
    for p in range(LMAX):
        col = np.array([int(code[n][p]) if len(code[n]) > p else 2 for n in ALL])
        best = 0.0
        for k in range(7):
            joint = np.zeros((3, 2))
            for c, bb in zip(col, bits[:, k]):
                joint[c, bb] += 1
            joint /= joint.sum()
            px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
            nz = joint > 0
            best = max(best, float((joint[nz] * np.log2(joint[nz] / (px @ py)[nz])).sum()))
        mi.append(best)
    corr_n = float(np.corrcoef(ALL, lens)[0, 1]) if lens.std() > 0 else 0.0
    corr_log = float(np.corrcoef(np.log2(ALL + 1), lens)[0, 1]) if lens.std() > 0 else 0.0
    return dict(code=code, distinct=distinct, mean_len=float(lens.mean()), len_corr_n=corr_n, len_corr_log=corr_log,
                mean_max_run=max_run, binary_exact=binexact, pos_bit_mi=mi)


torch.manual_seed(seed)
rng = np.random.default_rng(seed)
nrng = np.random.default_rng(555 + seed)
sp, jd = Speaker(), Judge()
params = list(sp.parameters()) + list(jd.parameters())
opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, i / steps))))
hist = []
for step in range(steps):
    f = step / steps
    beta = 2.5 * min(1.0, f / 0.5)
    lam = LAM * min(1.0, f / 0.4)
    ramp = min(1.0, f / 0.3)
    sigma = 1.0 + (0.3 - 1.0) * min(1.0, f / 0.5)
    a, b, ans = sample_pairs(rng, beta)
    s = sp(torch.from_numpy(a / NMAX).float())
    ids, lens = read_noise(s["syms"], nrng)
    m = jd(ids, lens, torch.from_numpy(b / NMAX).float())
    z = (m + sigma * torch.randn_like(m)).detach()
    same = torch.from_numpy(ans)
    r_j = torch.where(same, F.logsigmoid(z), F.logsigmoid(-z))
    r_e = r_j - lam * s["lens"]
    adv_j, adv_e = r_j - r_j.mean(), r_e - r_e.mean()
    logp_j = -((z - m) ** 2) / (2 * sigma ** 2)
    loss = (-(adv_j.detach() * logp_j).mean() - (adv_e.detach() * s["logp"]).mean()
            - 0.03 * (1.0 - ramp) * s["ent"].mean() - 0.5 * usage_entropy(s["probs0"], 1.0 * ramp))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step()
    if step % 1000 == 0 or step == steps - 1:
        acc_tr = eval_pairs(sp, jd, nrng, TRAIN[::3], TRAIN, draws=2)
        acc_ho = eval_pairs(sp, jd, nrng, HELD, None, draws=2) if len(HELD) else float("nan")
        hist.append((step, acc_tr, acc_ho, float(s["lens"].mean())))
        log(f"[{tag}] step {step} 점수 {r_j.mean().item():.3f} 길이 {s['lens'].mean().item():.2f} 훈련개수 {acc_tr:.3f} 시험개수 {acc_ho:.3f}")
acc_tr = eval_pairs(sp, jd, nrng, TRAIN, TRAIN, draws=6)
acc_ho = eval_pairs(sp, jd, nrng, HELD, None, draws=6) if len(HELD) else float("nan")
res = dict(cond=cond, seed=seed, lam=LAM, kappa=KAPPA, steps=steps, acc_train=acc_tr, acc_heldout=acc_ho, analysis=analyze(sp), hist=hist)
json.dump(res, open(f"{out}/{tag}.json", "w"), ensure_ascii=False)
log("RESULT " + json.dumps({k: res[k] for k in ("acc_train", "acc_heldout")}) + " " + json.dumps({k: v for k, v in res["analysis"].items() if k != "code"}))
