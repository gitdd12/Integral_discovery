"""runs2/*.json → RESULTS2.md (DESIGN2.md §5 의 판정 규칙대로)."""
import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict

out = sys.argv[1] if len(sys.argv) > 1 else "runs2"
ORDER = ["none", "compress", "progress", "general", "random", "memorize"]
runs = defaultdict(dict)
for p in glob.glob(os.path.join(out, "*_s*.json")):
    d = json.load(open(p))
    runs[d["criterion"]][d["seed"]] = d["hist"]


def final(h):
    return h[-1]


def ms(xs):
    return f"{st.mean(xs):.1f} ± {st.pstdev(xs):.1f}" if xs else "-"


L = ["# 결과 2 — 라이브러리 학습 실험 (DESIGN2.md)", "",
     f"시드 수: " + ", ".join(f"{c} {len(runs[c])}" for c in ORDER if c in runs), ""]
L += ["## 시험 과제 해결 수 (24개 중, 8회 후, 평균 ± 표준편차)", "",
      "| 기준 | 해결 수 | 학습 과제 해결(21) | 라이브러리 크기 | 깊이 | 함수 인자 개념 수 | 방법 개념을 쓴 시험 과제 | 개념 호출을 쓴 시험 과제 | 시험 풀이 평균 크기 |",
      "|---|---|---|---|---|---|---|---|---|"]
for c in ORDER:
    if c not in runs:
        continue
    f = [final(h) for h in runs[c].values()]
    L.append(f"| {c} | {ms([x['test_solved'] for x in f])} | {ms([x['train_solved'] for x in f])} | "
             f"{ms([x['lib_size'] for x in f])} | {ms([x['lib_depth'] for x in f])} | {ms([x['methods'] for x in f])} | "
             f"{ms([x['test_method_used'] for x in f])} | {ms([x['test_app_used'] for x in f])} | "
             f"{ms([x['test_mean_cost'] for x in f])} |")
L += ["", "## 회차별 시험 해결 수 (평균)", "", "| 기준 | " + " | ".join(str(i) for i in range(9)) + " |",
      "|---|" + "---|" * 9]
for c in ORDER:
    if c not in runs:
        continue
    row = []
    for i in range(9):
        row.append(f"{st.mean(h[i]['test_solved'] for h in runs[c].values() if len(h) > i):.1f}")
    L.append(f"| {c} | " + " | ".join(row) + " |")

L += ["", "## 묶음별 해결 수 (6개 중, 8회 후 평균)", "",
      "C 학습 부품의 새 조합 / A 처음 보는 본문 / B 처음 보는 본문 × 선택 / D 처음 보는 선택 조건", "",
      "| 기준 | C | A | B | D |", "|---|---|---|---|---|"]
from libworld.tasks import TEST  # noqa: E402
grp = {s[0]: s[1] for s in TEST}
for c in ORDER:
    if c not in runs:
        continue
    cnt = defaultdict(list)
    for h in runs[c].values():
        g = defaultdict(int)
        for r in final(h)["test"]:
            g[grp[r["task"]]] += r["solved"]
        for k in "CABD":
            cnt[k].append(g[k])
    L.append(f"| {c} | " + " | ".join(f"{st.mean(cnt[k]):.1f}" for k in "CABD") + " |")

L += ["", "## 판정 (DESIGN2.md §5)", ""]
mean = {c: st.mean(final(h)["test_solved"] for h in runs[c].values()) for c in runs}
sd = {c: st.pstdev([final(h)["test_solved"] for h in runs[c].values()]) for c in runs}
if "none" in mean and "compress" in mean:
    d = mean["compress"] - mean["none"]
    ok = d >= 3 and d > max(sd["none"], sd["compress"])
    L.append(f"- Q1: compress − none = {d:+.1f} → " + ("**개념 쌓기가 조합 문제에 도움이 된다**" if ok else "이 세계에서는 확인되지 않음"))
if "compress" in mean:
    for c in ORDER:
        if c in ("compress", "none") or c not in mean:
            continue
        d = mean[c] - mean["compress"]
        tag = "기준이 중요하다 (compress 보다 높음)" if d >= 3 and d > max(sd[c], sd["compress"]) else \
              ("compress 보다 낮음" if d <= -3 and -d > max(sd[c], sd["compress"]) else "차이 인정 안 됨")
        L.append(f"- Q2: {c} − compress = {d:+.1f} → {tag}")
    for c in ("random", "memorize"):
        if c in mean:
            ok = mean["compress"] - mean[c] >= 3
            L.append(f"- compress − {c} = {mean['compress'] - mean[c]:+.1f} → " + ("압축이 고르는 효과가 있음" if ok else "압축이 고르는 효과가 확인되지 않음"))
if "compress" in runs:
    mu = [final(h)["test_method_used"] for h in runs["compress"].values()]
    L.append(f"- 방법(함수 인자) 개념을 쓴 시험 과제: compress 평균 {st.mean(mu):.1f}개 " + ("(0 이면 '방법을 쌓았다'고 말하지 않는다)" if st.mean(mu) == 0 else ""))

L += ["", "## 기준별로 쌓인 개념 (시드 0, 8회 후)", ""]
for c in ORDER:
    if c in runs and 0 in runs[c]:
        L.append(f"### {c}")
        for m in final(runs[c][0])["lib"]:
            L.append(f"- `{m['name']}` = `{m['tmpl']}`  (구멍 {m['kinds'] or '없음'} → {m['ret']})")
        L.append("")
open(sys.argv[2] if len(sys.argv) > 2 else "RESULTS2.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
