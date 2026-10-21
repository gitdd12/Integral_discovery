"""눈 점검 로컬 실행: 변형 x 시드를 병렬로 돌리고 요약을 낸다.
사용: py -3.11 scripts\\run_local_eye.py --variants lstm,lstm_mid --seeds 1-3 --parallel 6 --steps 8000 --out runs_local\\eye
요약만: ... --summary"""
import argparse
import json
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--variants", default="lstm,lstm_mid")
ap.add_argument("--seeds", default="1-3")
ap.add_argument("--parallel", type=int, default=6)
ap.add_argument("--steps", type=int, default=8000)
ap.add_argument("--out", default="runs_local/eye")
ap.add_argument("--summary", action="store_true")
a = ap.parse_args()
lo, hi = (int(x) for x in a.seeds.split("-")) if "-" in a.seeds else (int(a.seeds), int(a.seeds))
jobs = [(v, s) for v in a.variants.split(",") for s in range(lo, hi + 1)]
os.makedirs(a.out, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))


def summary():
    print("\n변형  시드  훈련범위이웃  외삽(21~40)이웃  판독R2(외삽,h/c)")
    for v, s in jobs:
        p = f"{a.out}/{v}_s{s}.json"
        if not os.path.exists(p):
            print(f"{v:9s} {s}  (아직)")
            continue
        r = json.load(open(p, encoding="utf-8"))
        print(f"{v:9s} {s}  {r['train_adj']:.3f}  {r['extrap_adj']:.3f}  {r['probe']['h']['r2_extrap']:.2f}/{r['probe']['c']['r2_extrap']:.2f}")


if a.summary:
    summary()
    sys.exit()
env = dict(os.environ, OUT=a.out, THREADS="1", PYTHONIOENCODING="utf-8")
todo = [j for j in jobs if not os.path.exists(f"{a.out}/{j[0]}_s{j[1]}.json")]
running = []
while todo or running:
    running = [(p, j) for p, j in running if p.poll() is None]
    while todo and len(running) < a.parallel:
        v, s = todo.pop(0)
        errf = open(f"{a.out}/{v}_s{s}.err", "w")
        p = subprocess.Popen([sys.executable, os.path.join(here, "eye_check.py"), v, str(s), str(a.steps)], env=env, stdout=subprocess.DEVNULL, stderr=errf)
        running.append((p, (v, s)))
        print("시작", v, s, flush=True)
    time.sleep(5)
summary()
