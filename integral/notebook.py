"""메모장(한 문제의 전체 기록)과 계산기, 제출 실행기, 보상.

모델은 토큰을 한 개씩 `feed` 한다. 계산기·피드백이 채우는 토큰은 `feed` 가 돌려주고
(mask=0), 모델이 쓴 토큰은 mask=1 로 남는다. 학습은 mask=1 인 토큰만 대상으로 한다.

계산기: 생각하는 중에 `식=` 을 쓰면 결과를 끼워 넣는다 (`정의`/`반복` 문장 안의 `=` 는 제외).
제출  : `제출[ 절차 ]` 를 닫는 순간 지금 곡선에 돌려 `→ 오차 12%` (또는 `→ 오류`)를 붙인다.
보상  : 마지막 제출만 채점. 지금 곡선 + 같은 종류의 못 본 곡선 3개에서의 평균 상대 오차
        → 점수 = -log10(평균 상대 오차), 0~6.
"""
from __future__ import annotations

import math
from typing import Callable

from . import vocab as V_
from .expr import EvalError, ParseError, compile_ast, fmt_num, parse_expr
from .interp import Ctx, exec_stmt, parse_program, parse_stmt, run_program, split_statements
from .vocab import tokenize

MAX_SCORE = 6.0
SAFE_ERRORS = (ParseError, EvalError, RecursionError, OverflowError, ZeroDivisionError,
               ValueError, TypeError)


def score_from_errors(rel_errors: list[float]) -> float:
    """상대 오차 목록 → 점수 (= -log10(평균), 0~6으로 자름)."""
    mean = sum(rel_errors) / len(rel_errors)
    if mean <= 0:
        return MAX_SCORE
    return min(max(-math.log10(mean), 0.0), MAX_SCORE)


def _rel_err(v: float, target: float) -> float:
    return abs(v - target) / abs(target)


class Notebook:
    def __init__(
        self,
        prompt: list[str] | list[int],
        *,
        curve=None,
        target: float | None = None,
        hidden: list | Callable[[], list] | None = None,
        store: dict | None = None,
        x: float | None = None,
        f: Callable[[float], float] | None = None,
        interval: tuple[float, float] | None = None,
        max_len: int = 1024,
        max_submits: int = 8,
        min_submits: int = 0,
    ):
        """prompt: 문제 문장(토큰 문자열 또는 id). curve 가 있으면 target 은 그 곡선의 넓이.

        hidden: 못 본 곡선 목록(또는 만드는 함수). 없으면 지금 곡선만으로 채점.
        store : `정의` 저장소 (다음 곡선으로 넘어가도 유지하려면 이전 notebook.store 를 넘긴다).
        x     : 문제에 x=값 이 주어진 경우 그 값.
        f     : 곡선 없이 수식만 주어진 문제(학교)에서 쓸 f.
        interval: 곡선 없이 a, b 만 주어진 문제(학교)의 (a, b).
        """
        self.ids: list[int] = []
        self.mask: list[int] = []
        for t in prompt:
            self._append(t if isinstance(t, int) else V_.TOK2ID[t], 0)
        self.n_prompt = len(self.ids)
        self.curve = curve
        self.target = target if target is not None else (curve.area if curve is not None else None)
        self._hidden = hidden
        self.store: dict = dict(store) if store else {}
        self.vars: dict = {}
        if curve is not None:
            self.vars.update(a=curve.a, b=curve.b, f=curve.f)
        if f is not None:
            self.vars["f"] = f
        if interval is not None:
            self.vars["a"], self.vars["b"] = float(interval[0]), float(interval[1])
        if x is not None:
            self.vars["x"] = float(x)
        self.last_calc = None   # 마지막 계산기 결과 (숫자, True/False, 오류면 None)
        self.max_len, self.max_submits, self.min_submits = max_len, max_submits, min_submits
        self.buf: list[str] = []
        self.sub: list[str] | None = None
        self.sub_bad = False
        self.submissions: list[tuple] = []  # (프로그램 또는 None, 저장소 사본, 토큰)
        self.last_feedback: float | None = None  # 마지막 제출의 지금 곡선 상대 오차
        self.violations = 0
        self.done = False
        self._ctx = Ctx()

    # ---------------------------------------------------------------- 기록
    def _append(self, tok_id: int, m: int):
        self.ids.append(tok_id)
        self.mask.append(m)

    def _force(self, toks: list[str]) -> list[int]:
        out = [V_.TOK2ID[t] for t in toks]
        for i in out:
            self._append(i, 0)
        return out

    @property
    def eos_ok(self) -> bool:
        """끝내도 되는가: 최소 제출 횟수를 채웠으면 True (샘플러가 아니면 <eos> 를 못 뽑게 막는다)."""
        return len(self.submissions) >= self.min_submits

    # ---------------------------------------------------------------- 입력
    def feed(self, tok_id: int) -> list[int]:
        """모델 토큰 하나를 받는다. 계산기가 채운 토큰(id 목록)을 돌려준다."""
        if self.done:
            raise RuntimeError("이미 끝난 메모장")
        t = V_.TOKENS[tok_id]
        self._append(tok_id, 1)
        forced: list[int] = []
        if t == V_.EOS:
            self.done = True
            return forced
        if t not in V_.MODEL_ALLOWED:
            self.violations += 1
        elif self.sub is not None:
            if t == "]":
                forced = self._finish_submit()
            elif t == "제출[":
                self.sub_bad = True
            else:
                self.sub.append(t)
        elif t == "제출[":
            self.sub, self.sub_bad, self.buf = [], False, []
        elif t in (";", "\n"):
            if self.buf:
                forced = self._run_statement()
            self.buf = []
        elif t == "=":
            if self.buf and self.buf[0] not in ("정의", "반복"):
                forced = self._calc()
            else:
                self.buf.append(t)
        else:
            self.buf.append(t)
        if len(self.ids) >= self.max_len:
            self.done = True
        return forced

    def feed_text(self, text: str) -> str:
        """(시험용) 글자로 된 모델 출력을 한 토큰씩 먹인다. 끝나면 멈춘다."""
        for t in tokenize(text):
            if self.done:
                break
            self.feed(V_.TOK2ID[t])
        return V_.decode(self.ids, self.mask)

    # ---------------------------------------------------------------- 계산기
    def _calc(self) -> list[int]:
        toks, self.buf = self.buf, []
        try:
            ast = parse_expr(toks, allow_cmp=True)
            val = compile_ast(ast)(self.vars, self.store)
            if isinstance(val, bool):
                out = ["참" if val else "거짓"]
            else:
                out = tokenize(fmt_num(val))
            self.last_calc = val
        except SAFE_ERRORS:
            out = ["오류"]
            self.last_calc = None
        return self._force(out)

    def _run_statement(self) -> list[int]:
        """`;` 나 줄바꿈으로 끝난 문장 실행. 정의/반복만 의미가 있다."""
        toks = self.buf
        if toks[0] not in ("정의", "반복"):
            return []
        try:
            exec_stmt(parse_stmt(toks), self.vars, self.store, self._ctx)
        except SAFE_ERRORS:
            return self._force(["오류"])
        return []

    # ---------------------------------------------------------------- 제출
    def _finish_submit(self) -> list[int]:
        toks, bad = self.sub, self.sub_bad
        self.sub = None
        prog, err = None, None
        if not bad:
            try:
                prog = parse_program(toks, strict=True)
                v = self._run(prog, self.vars, dict(self.store))
                if self.target is None:
                    raise EvalError("정답 없음")
                err = _rel_err(v, self.target)
                if not math.isfinite(err):
                    raise EvalError("nan")
            except SAFE_ERRORS:
                prog, err = None, None
        self.submissions.append((prog, dict(self.store), toks))
        self.last_feedback = err
        if err is None:
            out = ["→", "오류", "\n"]
        else:
            out = ["→", "오차"] + tokenize(fmt_num(err * 100)) + ["%", "\n"]
        forced = self._force(out)
        if len(self.submissions) >= self.max_submits or len(self.ids) >= self.max_len:
            self.done = True
        elif len(self.submissions) < self.min_submits:
            # 최소 제출 횟수를 못 채웠으면 코드가 `제출[` 를 대신 써 준다 (본문은 모델이 쓴다).
            forced += self._force(["제출["])
            self.sub, self.sub_bad, self.buf = [], False, []
        return forced

    @staticmethod
    def _run(prog, vars_: dict, store: dict) -> float:
        return run_program(prog, dict(vars_), store, Ctx())

    # ---------------------------------------------------------------- 보상
    def hidden_curves(self) -> list:
        h = self._hidden
        if h is None:
            return []
        return h() if callable(h) else list(h)

    def final_score(self) -> float:
        """마지막 제출만 채점. 제출이 없거나 마지막이 실행 불가·위반이면 0점."""
        if self.violations or not self.submissions or self.target is None:
            return 0.0
        prog, store, _ = self.submissions[-1]
        if prog is None:
            return 0.0
        errs = []
        try:
            errs.append(_rel_err(self._run(prog, self.vars, dict(store)), self.target))
            for c in self.hidden_curves():
                v = self._run(prog, {"a": c.a, "b": c.b, "f": c.f}, dict(store))
                errs.append(_rel_err(v, c.area))
        except SAFE_ERRORS:
            return 0.0
        if not all(math.isfinite(e) for e in errs):
            return 0.0
        return score_from_errors(errs)

    def render(self) -> str:
        return V_.decode(self.ids, self.mask)


def new_episode(curve, seed: int, *, store: dict | None = None, **kw) -> Notebook:
    """곡선 하나로 시작하는 진짜 문제. 못 본 곡선 3개는 seed 로 정해진다 (모델에겐 안 보임)."""
    from .curves import siblings

    return Notebook(
        [V_.BOS] + curve.prompt_tokens(),
        curve=curve,
        hidden=lambda: siblings(curve.kind, seed),
        store=store,
        **kw,
    )
