"""단계 A 본 실행 (DESIGN3.md): 시드 하나 = 단계 0(T2 강화학습) → T4 강화학습(로그 점수) → 평가.
사용: python scripts/run_count_A.py <시드> [단계수]   결과: runs3/stageA/s<시드>.json, agent_s<시드>.pt"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from countworld.evaluate import full_report  # noqa: E402
from countworld.scene import SceneCfg  # noqa: E402
from countworld.train import Agent, train_pair, train_rl  # noqa: E402

seed = int(sys.argv[1])
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
torch.set_num_threads(int(os.environ.get("THREADS", "1")))
out = "runs3/stageA"
os.makedirs(out, exist_ok=True)
logf = open(f"{out}/s{seed}.log", "w")


def log(s):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


torch.manual_seed(seed)
cfg = SceneCfg(max_count=9, extra=6)
ag = Agent(V=10, Lmax=1)
train_pair(ag, cfg, steps=4000, mode="rl", lr=5e-4, eval_every=250, stop_acc=0.985, beta_fn=lambda f: 0.0, seed=seed, log=log)
hist = train_rl(ag, cfg, steps=steps, B=512, lr=3e-4, min_len=1, usage=0.5, ent_coef=0.03,
                beta_fn=lambda f: 2.5 * min(1.0, f / 0.5), score="log", eval_every=500, seed=seed, log=log)
torch.save({"bb": ag.bb.state_dict(), "judge": ag.judge.state_dict(), "speaker": ag.speaker.state_dict()}, f"{out}/agent_s{seed}.pt")
rep = full_report(ag, cfg, np.random.default_rng(1000 + seed), min_len=1)
json.dump({"seed": seed, "steps": steps, "hist": hist, "report": rep}, open(f"{out}/s{seed}.json", "w"), ensure_ascii=False, default=str)
log("REPORT " + json.dumps(rep, default=str)[:1500])
