"""합쳐진 개수 풀기 미세조정 (DESIGN3 A9).
저장된 A8 가중치에서 시작해, 같은 기호를 쓰는(합쳐진) 개수들만 집중적으로:
  (1) 그 개수들 사이의 쌍을 훨씬 자주 출제하고  (2) 그 개수의 장면에서는 화자의 탐색 온도를 높인다.
대조군(control)은 같은 단계 수를 이전 설정 그대로 이어서 학습한다.
사용: python scripts/finetune_merge.py --src runs_local/A8 --seeds 2,3,5,7 --steps 2000 --out runs_local/A9
환경변수: BOOST(합쳐진 쌍 가중치, 기본 20), TEMP(탐색 온도, 기본 3.0)"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from countworld.evaluate import code_table, full_report  # noqa: E402
from countworld.scene import SceneCfg, make_batch  # noqa: E402
from countworld.train import Agent, adjacent_score, evaluate, to_t, usage_entropy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="runs_local/A8")
ap.add_argument("--seeds", default="2,3,5,7")
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--out", default="runs_local/A9")
ap.add_argument("--variants", default="control,force")
args = ap.parse_args()
BOOST, TEMP = float(os.environ.get("BOOST", 20)), float(os.environ.get("TEMP", 3.0))
DEV = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
os.makedirs(args.out, exist_ok=True)
cfg = SceneCfg(max_count=9, extra=6)
N = cfg.max_count + 1


def merged_groups(ag, rng):
    modal = code_table(ag, cfg, rng, min_len=1)["modal"]
    by = {}
    for k, m in modal.items():
        by.setdefault(m, []).append(k)
    return [v for v in by.values() if len(v) > 1], modal


def run(seed, variant, logf):
    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    torch.manual_seed(500 + seed)
    ag = Agent(V=10, Lmax=1, dev=DEV)
    st = torch.load(f"{args.src}/agent_s{seed}.pt", map_location=DEV)
    ag.bb.load_state_dict(st["bb"]); ag.speaker.load_state_dict(st["speaker"])
    ag.judge.load_state_dict(st["judges"][0] if "judges" in st and st["judges"] else st["judge"])
    ag.judges[0] = ag.judge
    rng = np.random.default_rng(900 + seed)
    vrng = np.random.default_rng(777 + seed)
    groups, _ = merged_groups(ag, np.random.default_rng(1000 + seed))
    flat = sorted({k for g in groups for k in g})
    log(f"[s{seed} {variant}] 시작 합쳐진 묶음 {groups}  검증 {adjacent_score(ag, cfg, vrng, min_len=1):.3f}")
    if not groups:
        log("합쳐진 쌍 없음 — 건너뜀")
        return None
    W = np.ones((N, N))
    if variant == "force":
        for g in groups:
            for i in g:
                for j in g:
                    if i != j:
                        W[i, j] = BOOST
    opt = torch.optim.AdamW(ag.params(), lr=2e-4, weight_decay=0.05)
    err = np.full((N, N), 0.5)
    best, best_state, steps = adjacent_score(ag, cfg, vrng, min_len=1), ag.snapshot(), args.steps
    for step in range(steps):
        fw = (err + 0.05) ** 2
        fw = np.maximum(fw, fw.T) * W
        b = to_t(make_batch(cfg, rng, 512, beta=2.5, pair_weight=fw), ag.dev)
        temp = None
        if variant == "force":
            inm = torch.isin(b["a"].to(ag.dev), torch.tensor(flat, device=ag.dev))
            temp = torch.where(inm, torch.full((512,), TEMP, device=ag.dev), torch.ones(512, device=ag.dev))
        sp = ag.speaker(b["s1"], b["l1"], mode="sample", min_len=1, temp=temp)
        m = ag.judge(sp["syms"], b["s2"], b["l2"])
        sigma = 0.3
        z = (m + sigma * torch.randn_like(m)).detach()
        same = b["answer"]
        r = torch.where(same, F.logsigmoid(z), F.logsigmoid(-z))
        wrong = ((z > 0) != same).float().cpu().numpy()
        a_, b_ = b["a"].cpu().numpy(), b["b"].cpu().numpy()
        for x, y, w in zip(np.minimum(a_, b_), np.maximum(a_, b_), wrong):
            err[x, y] = 0.98 * err[x, y] + 0.02 * w
            err[y, x] = err[x, y]
        adv = r - r.mean()
        logp_j = -((z - m) ** 2) / (2 * sigma ** 2)
        loss = (-(adv.detach() * logp_j).mean() - (adv.detach() * sp["logp"]).mean()
                - 1.0 * usage_entropy(sp["probs0"], 1.0))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ag.params(), 1.0)
        opt.step()
        if (step + 1) % 250 == 0:
            val = adjacent_score(ag, cfg, vrng, min_len=1)
            acc, _ = evaluate(ag, cfg, rng, min_len=1)
            gs, _ = merged_groups(ag, np.random.default_rng(1000 + seed))
            log(f"[s{seed} {variant}] step {step+1} 점수 {r.mean().item():.3f} 정확도 {acc:.3f} 검증 {val:.3f} 합쳐진 {gs}")
            if val > best:
                best, best_state = val, ag.snapshot()
    ag.load_snapshot(best_state)
    rep = full_report(ag, cfg, np.random.default_rng(1000 + seed), min_len=1)
    torch.save({"bb": ag.bb.state_dict(), "judge": ag.judge.state_dict(), "speaker": ag.speaker.state_dict()},
               f"{args.out}/agent_{variant}_s{seed}.pt")
    json.dump({"seed": seed, "variant": variant, "report": rep, "best_val": best}, open(f"{args.out}/{variant}_s{seed}.json", "w"),
              ensure_ascii=False, default=str)
    return rep


rows = []
logf = open(f"{args.out}/log.txt", "a", encoding="utf-8")
for seed in [int(x) for x in args.seeds.split(",")]:
    for v in args.variants.split(","):
        rep = run(seed, v, logf)
        if rep:
            e5 = rep["e5"]
            modal = e5["modal"]
            by = {}
            for k, mm in modal.items():
                by.setdefault(mm, []).append(k)
            gs = [x for x in by.values() if len(x) > 1]
            rows.append((seed, v, rep["e1_train"], rep["e4_adjacent"], e5["consistency"], e5["injective"], gs))
print("\n시드 변형 정확도 인접점수 일관성 단사성 합쳐진 묶음")
for r in rows:
    print(r)
