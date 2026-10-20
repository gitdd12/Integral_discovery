"""단계 A 학습 결과 점검: 장면 읽는 몸통이 개수를 구분하는가(선형 탐침), 화자의 기호 선택이 개수별로 어떻게 생겼는가."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from countworld.scene import SceneCfg, make_batch  # noqa: E402
from countworld.train import Agent, to_t  # noqa: E402

torch.manual_seed(0)
cfg = SceneCfg(max_count=9, extra=6)
rng = np.random.default_rng(7)
for seed in (1, 2, 3):
    ag = Agent(V=10, Lmax=1)
    st = torch.load(f"runs3/stageA/agent_s{seed}.pt")
    ag.bb.load_state_dict(st["bb"]); ag.judge.load_state_dict(st["judge"]); ag.speaker.load_state_dict(st["speaker"])
    X, Y, P = [], [], []
    with torch.no_grad():
        for k in range(10):
            a = np.full(400, k)
            b = to_t(make_batch(cfg, rng, 400, a=a, b=a), "cpu")
            X.append(ag.bb.encode(b["s1"], b["l1"]))
            sp = ag.speaker(b["s1"], b["l1"], mode="greedy", min_len=1)
            h = torch.tanh(ag.speaker.init(X[-1]))
            logits = ag.speaker.head(ag.speaker.gru(ag.speaker.sym_emb(torch.full((400,), 11)), h))
            logits[:, 10] = -1e9
            P.append(F.softmax(logits, -1).mean(0)[:10])
            Y.append(torch.full((400,), k))
    X, Y = torch.cat(X), torch.cat(Y)
    perm = torch.randperm(len(X)); tr, te = perm[:2400], perm[2400:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd
    lin = nn.Linear(X.shape[1], 10)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for _ in range(400):
        loss = F.cross_entropy(lin(Xn[tr]), Y[tr]); opt.zero_grad(); loss.backward(); opt.step()
    pred = lin(Xn[te]).argmax(-1)
    per = [(pred[Y[te] == k] == k).float().mean().item() for k in range(10)]
    print(f"== 시드 {seed}: 몸통 표현에서 개수를 읽는 선형 탐침 정확도 {np.mean(per):.3f}")
    print("   개수별:", [round(p, 2) for p in per])
    print("   화자가 개수별로 고르는 기호 (1위 확률 / 2위 기호 확률):")
    for k in range(10):
        p = P[k]; top = p.topk(2)
        print(f"     개수 {k}: 기호 {top.indices[0].item()} {top.values[0]:.2f} / 기호 {top.indices[1].item()} {top.values[1]:.2f}")
