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
out = os.environ.get("OUT", "runs3/stageA")
os.makedirs(out, exist_ok=True)
logf = open(f"{out}/s{seed}.log", "a")


def log(s):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


torch.manual_seed(seed)
cfg = SceneCfg(max_count=9, extra=6)
K = int(os.environ.get("LISTENERS", 1))
REFRESH = int(os.environ.get("REFRESH", 0))
ag = Agent(V=10, Lmax=1, K=K)
ckpt_path = f"{out}/ckpt_s{seed}.pt"


def state():
    return {"bb": ag.bb.state_dict(), "judge": ag.judge.state_dict(), "speaker": ag.speaker.state_dict(),
            "judges": [j.state_dict() for j in ag.judges], "head0": ag.head0, "ptr": ag.refresh_ptr}


def load(st):
    ag.bb.load_state_dict(st["bb"])
    ag.speaker.load_state_dict(st["speaker"])
    for j, sd in zip(ag.judges, st["judges"]):
        j.load_state_dict(sd)
    ag.head0, ag.refresh_ptr = st["head0"], st["ptr"]


step0, opt_state, hist, pair_done = 0, None, [], False
if os.path.exists(ckpt_path):                     # 컨테이너가 재시작돼도 이어서
    ck = torch.load(ckpt_path)
    load(ck["agent"])
    step0, opt_state, hist, pair_done = ck["step"], ck["opt"], ck["hist"], ck["pair_done"]
    log(f"체크포인트에서 이어하기: 단계 {step0}")
if not pair_done:
    train_pair(ag, cfg, steps=4000, mode="rl", lr=5e-4, eval_every=250, stop_acc=0.985, beta_fn=lambda f: 0.0, seed=seed, log=log)
    ag.sync_listeners()
    torch.save({"agent": state(), "step": 0, "opt": None, "hist": [], "pair_done": True}, ckpt_path)


def on_ckpt(step, opt, h):
    torch.save({"agent": state(), "step": step, "opt": opt.state_dict(), "hist": h, "pair_done": True}, ckpt_path + ".tmp")
    os.replace(ckpt_path + ".tmp", ckpt_path)


USAGE, MI, RAMP = float(os.environ.get("USAGE", 0.5)), float(os.environ.get("MI", 0.0)), float(os.environ.get("RAMP", 0.0))
hist = train_rl(ag, cfg, steps=steps, B=512, lr=3e-4, min_len=1, usage=USAGE, mi_cond=MI, mi_ramp=RAMP, refresh_every=REFRESH, focus=float(os.environ.get("FOCUS", 0)), ent_coef=float(os.environ.get("ENT", 0.03)),
                beta_fn=lambda f: 2.5 * min(1.0, f / 0.5), score="log", eval_every=500, seed=seed, log=log,
                step0=step0, opt_state=opt_state, on_ckpt=on_ckpt, hist=hist)
torch.save(state(), f"{out}/agent_s{seed}.pt")
rep = full_report(ag, cfg, np.random.default_rng(1000 + seed), min_len=1)
json.dump({"seed": seed, "steps": steps, "hist": hist, "report": rep}, open(f"{out}/s{seed}.json", "w"), ensure_ascii=False, default=str)
log("REPORT " + json.dumps(rep, default=str)[:1500])
