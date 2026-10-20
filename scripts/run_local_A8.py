"""내 컴퓨터(GPU)에서 단계 A8 을 돌리는 실행기. 노트북 없이 터미널에서:
    python scripts/run_local_A8.py                 # 시드 1~8, 동시 3개
    python scripts/run_local_A8.py --summary       # 끝난 결과 요약만 보기
끊겨도 같은 명령을 다시 실행하면 체크포인트에서 이어가고, 끝난 시드는 건너뛴다."""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = dict(USAGE="1.0", MI="1.0", RAMP="0.3", ENT="0.03", FOCUS="2", RESTARTS="0.4,0.7", RTHRESH="0.95", RMAX="2")


def summarize(out, seeds):
    import numpy as np
    accs, perfect = [], 0
    for s in seeds:
        p = f"{out}/s{s}.json"
        if not os.path.exists(p):
            print(f"  시드{s}: 아직"); continue
        d = json.load(open(p)); r = d["report"]; e5 = r["e5"]; adj = r["e4_adjacent"]
        ok = sum(v >= 0.9 for v in adj.values())
        merged = {}
        for k, v in e5["modal"].items():
            merged.setdefault(tuple(v), []).append(int(k))
        merged = [g for g in merged.values() if len(g) > 1]
        good = r["e1_train"] >= 0.95 and ok == 9 and e5["consistency"] >= 0.95 and e5["injective"] >= 0.95
        perfect += good; accs.append(r["e1_train"])
        print(f"  시드{s}: 정확도 {r['e1_train']:.3f} | 빼둔쌍 {r['e1_heldout_pairs']:.3f} | n대n+1(균형) ≥90% {ok}/9 | "
              f"일관성 {e5['consistency']:.2f} 단사성 {e5['injective']:.2f} | 합쳐진 개수 {merged} | "
              f"재시작 {d.get('restarts', '?')}회 {d.get('restart_log', '')} | {'완벽' if good else ''}")
    if accs:
        print(f"  >> 끝난 시드 {len(accs)}개 중 완벽한 코드 {perfect}개 ({perfect/len(accs):.0%}), 정확도 평균 {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print("  (완벽 = 정확도≥0.95, 모든 n대n+1(균형) ≥90%, 일관성·단사성 ≥0.95)")


def last_line(out, s):
    p = f"{out}/s{s}.log"
    if not os.path.exists(p):
        return "(시작 전)"
    L = [l for l in open(p, encoding="utf-8").read().splitlines() if l.strip()]
    return L[-1][:110] if L else ""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-8", help="예: 1-8 또는 1,2,5")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--out", default="runs_local/A8")
    ap.add_argument("--pair-steps", type=int, default=4000)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if "-" in a.seeds:
        lo, hi = a.seeds.split("-"); seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x) for x in a.seeds.split(",")]
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    if a.summary:
        summarize(out, seeds); sys.exit()
    try:
        import torch
        print("GPU 사용 가능:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    except Exception as e:
        print("torch 를 불러오지 못했습니다:", e); sys.exit(1)
    jobs = [s for s in seeds if not os.path.exists(f"{out}/s{s}.json")]
    print("돌려야 할 학습:", len(jobs), "개 / 결과 폴더:", out)
    running, t0, tick, started, failed = [], time.time(), 0, {}, []
    while jobs or running:
        for (s, p) in running:                                                 # 비정상 종료 감지
            if p.poll() not in (None, 0) and s not in failed:
                failed.append(s)
                err = open(f"{out}/s{s}.err", encoding="utf-8", errors="replace").read()[-600:]
                print(f"!!! 시드{s} 가 오류로 멈췄습니다 (코드 {p.poll()}). 마지막 오류:\n{err}")
        running = [(s, p) for (s, p) in running if p.poll() is None]
        while jobs and len(running) < a.parallel:
            s = jobs.pop(0)
            env = dict(os.environ, OUT=out, LISTENERS="1", REFRESH="0", THREADS="1", PAIR_STEPS=str(a.pair_steps),
                       DEVICE="cuda" if torch.cuda.is_available() else "cpu", **CFG)
            errf = open(f"{out}/s{s}.err", "w", encoding="utf-8")      # 오류 메시지가 사라지지 않게 파일로 남긴다
            p = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "scripts", "run_count_A.py"), str(s), str(a.steps)],
                                 env=env, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=errf)
            running.append((s, p))
            started[s] = p
        if tick % 6 == 0:
            print(f"--- {int((time.time()-t0)/60)}분 경과, 실행 중 {len(running)}개, 대기 {len(jobs)}개")
            for s, _ in running:
                print(f"  시드{s}:", last_line(out, s))
        tick += 1
        time.sleep(10)
    if failed:
        print(f"!!! 오류로 멈춘 시드: {failed} (자세한 내용은 {out}/s시드.err 파일)")
    print("모두 끝났습니다\n")
    summarize(out, seeds)
