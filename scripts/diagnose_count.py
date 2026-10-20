"""일관성이 왜 낮은가 진단. 사용: python scripts/diagnose_count.py <가중치 폴더> <시드들...>
같은 개수에서 화자의 기호가 '장면의 어떤 성질'에 따라 달라지는지를 본다.
  - 개수별 기호 분포(탐욕 선택)와 확률적 선택의 엔트로피
  - 기호와 개수의 상호정보량, 기호와 전체 길이 T 의 (개수를 고정한) 조건부 상호정보량
  - 개수가 같을 때 전체 길이 T 별로 최빈 기호가 달라지는 비율
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from countworld.scene import SceneCfg, make_batch  # noqa: E402
from countworld.train import Agent, to_t  # noqa: E402

folder = sys.argv[1]
seeds = [int(x) for x in sys.argv[2:]]
cfg = SceneCfg(max_count=9, extra=6)
rng = np.random.default_rng(5)


def H(c):
    p = np.array(list(c.values()), float)
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def mi(xs, ys):
    cx, cy, cxy = Counter(xs), Counter(ys), Counter(zip(xs, ys))
    return H(cx) + H(cy) - H(cxy)


for seed in seeds:
    path = f"{folder}/agent_s{seed}.pt"
    if not os.path.exists(path):
        print(f"시드 {seed}: 파일 없음"); continue
    st = torch.load(path, map_location="cpu")
    K = len(st["judges"]) if "judges" in st else 1
    ag = Agent(V=10, Lmax=1, K=K)
    ag.bb.load_state_dict(st["bb"]); ag.speaker.load_state_dict(st["speaker"])
    with torch.no_grad():
        counts, Ts, syms, ents, probs = [], [], [], [], []
        for k in range(10):
            a = np.full(600, k)
            b = to_t(make_batch(cfg, rng, 600, a=a, b=a), "cpu")
            h = torch.tanh(ag.speaker.init(ag.bb.encode(b["s1"], b["l1"])))
            logits = ag.speaker.head(ag.speaker.gru(ag.speaker.sym_emb(torch.full((600,), 11)), h))
            logits[:, 10] = -1e9
            p = F.softmax(logits, -1)[:, :10]
            counts += [k] * 600; Ts += b["T"].tolist(); syms += p.argmax(-1).tolist()
            ents += (-(p * (p + 1e-9).log()).sum(-1)).tolist(); probs += p.max(-1).values.tolist()
    counts, Ts, syms = np.array(counts), np.array(Ts), np.array(syms)
    print(f"\n=== 시드 {seed} ===")
    print(f"기호-개수 상호정보량 {mi(counts, syms):.2f} 비트 (최대 {np.log2(10):.2f}), 기호 엔트로피 {H(Counter(syms)):.2f}")
    cm = []
    for t in sorted(set(Ts)):
        m = Ts == t
        cm.append(mi(counts[m], syms[m]))
    print(f"기호-개수 상호정보량을 전체 길이 T 별로 쟀을 때 평균 {np.mean(cm):.2f} 비트")
    # 개수를 고정했을 때 T 가 기호를 얼마나 설명하나 (조건부 상호정보량)
    cond = np.mean([mi(Ts[counts == k], syms[counts == k]) for k in range(10)])
    print(f"개수를 고정했을 때 기호와 전체 길이 T 의 상호정보량(평균) {cond:.3f} 비트  (0에 가까우면 T 는 상관 없음)")
    print("개수별: 최빈 기호(비율) / 전체 길이별 최빈 기호가 서로 다른 개수 / 확률적 선택 시 1위 확률 평균")
    for k in range(10):
        m = counts == k
        c = Counter(syms[m]); top, n = c.most_common(1)[0]
        per_T = {int(t): Counter(syms[m & (Ts == t)]).most_common(1)[0][0] for t in sorted(set(Ts[m]))}
        pm = np.mean(np.array(probs)[m])
        print(f"  개수 {k}: 기호 {top}({n/m.sum():.2f}) | T별 최빈 기호 {sorted(set(per_T.values()))} | 확률적 1위 확률 {pm:.2f}")
