"""단계 A 이어 학습: 저장된 시드의 가중치에서 정규화만 바꿔 5000단계 더 (합쳐진 개수가 풀리는지).
사용: python scripts/finetune_A.py <시드> <변형 이름> <usage> <mi_cond> <ent_coef>"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from countworld.evaluate import full_report  # noqa: E402
from countworld.scene import SceneCfg  # noqa: E402
from countworld.train import Agent, train_rl  # noqa: E402

seed, name, usage, mi, ent = int(sys.argv[1]), sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
torch.set_num_threads(1)
out = "runs3/stageA_ft"
os.makedirs(out, exist_ok=True)
logf = open(f"{out}/{name}_s{seed}.log", "a")


def log(s):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


torch.manual_seed(100 + seed)
cfg = SceneCfg(max_count=9, extra=6)
ag = Agent(V=10, Lmax=1)
st = torch.load(f"runs3/stageA/agent_s{seed}.pt")
ag.bb.load_state_dict(st["bb"]); ag.judge.load_state_dict(st["judge"]); ag.speaker.load_state_dict(st["speaker"])
hist = train_rl(ag, cfg, steps=5000, B=512, lr=2e-4, min_len=1, usage=usage, mi_cond=mi, ent_coef=ent,
                beta_fn=lambda f: 2.5, score="log", eval_every=500, seed=200 + seed, log=log)
rep = full_report(ag, cfg, np.random.default_rng(1000 + seed), min_len=1)
torch.save({"bb": ag.bb.state_dict(), "judge": ag.judge.state_dict(), "speaker": ag.speaker.state_dict()}, f"{out}/agent_{name}_s{seed}.pt")
json.dump({"seed": seed, "variant": name, "hist": hist, "report": rep}, open(f"{out}/{name}_s{seed}.json", "w"), ensure_ascii=False, default=str)
log("REPORT " + json.dumps(rep, default=str)[:1500])
