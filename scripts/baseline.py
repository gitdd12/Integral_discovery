"""학습 없이 모델을 평가한다 (대조군). 같은 seed 면 같은 곡선·같은 횟수로 풀어서 짝지어 비교할 수 있다.

  학교 졸업 모델(대조군):  python scripts/baseline.py --ckpt weights/school.pt --out OUT/base_school
  강화 학습한 모델:        python scripts/baseline.py --ckpt OUT/run2/ckpt.pt   --out OUT/base_run2
(두 번 모두 같은 --seed, --min_submits 를 쓸 것.) 가중치는 바꾸지 않는다.
"""
import argparse
import json
import os

import numpy as np
import torch

from integral import rl
from integral.model import GPT

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--n_curves", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--min_submits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=777)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(a.out, "thoughts"), exist_ok=True)
    model, _ = GPT.load(a.ckpt, device)
    cfg = rl.RLConfig(n_curves=a.n_curves, k=a.k, temperature=a.temperature, min_submits=a.min_submits)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(a.seed)
    log = open(os.path.join(a.out, "scores.jsonl"), "w")
    best_all = []
    for r in range(1, a.rounds + 1):
        nbs, scores, curves, groups = rl.collect(model, rng, cfg, device, gen)
        st = rl.round_stats(nbs, scores)
        # 곡선별 '8번 중 최고 점수'의 평균 = 같은 횟수로 비교할 때 쓰는 값
        st["best_of_k"] = float(np.mean([max(scores[i] for i in g) for g in groups]))
        best_all.append(st["best_of_k"])
        log.write(json.dumps(dict(round=r, **st)) + "\n")
        log.flush()
        if r <= 5:
            json.dump(rl.thought_sample(nbs, scores, curves, groups), open(
                os.path.join(a.out, "thoughts", f"round{r:06d}.json"), "w"), ensure_ascii=False, indent=1)
    rows = [json.loads(l) for l in open(os.path.join(a.out, "scores.jsonl"))]
    m = lambda key: float(np.mean([x[key] for x in rows]))
    print(f"{a.ckpt}: {a.rounds}바퀴  평균점수 {m('mean'):.4f}  곡선당최고(8번중) {m('best_of_k'):.4f}  "
          f"최고 {max(x['max'] for x in rows):.3f}  유효 {m('valid')*100:.0f}%  "
          f"제출 {m('submits'):.2f}회  반복사용 {m('loops')*100:.2f}%")
