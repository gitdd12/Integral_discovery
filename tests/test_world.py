import math

import numpy as np
import pytest
from scipy.integrate import quad

from integral import vocab as V
from integral.curves import (TEST_KINDS, TRAIN_KINDS, Curve, make_curve, make_test_curve,
                             random_curve, siblings)
from integral.expr import EvalError, ParseError, compile_ast, fmt_num, parse_expr
from integral.interp import Ctx, parse_program, run_program
from integral.notebook import Notebook, new_episode, score_from_errors


def ev(text, **vars_):
    return compile_ast(parse_expr(V.tokenize(text), allow_cmp=True))(vars_, {})


# ------------------------------------------------------------ 토크나이저
def test_roundtrip_and_longest_match():
    for s in ["p10", "p1", "rect(4,2)=", "f(a+i×(b-a)÷N)", "exp(-0.3×x)", "제출[정의 p0=1]"]:
        toks = V.tokenize(s)
        assert V.tokenize(V.decode(V.ids_of(toks))) == toks
    assert V.tokenize("p10") == ["p10"]
    assert V.tokenize("p1 0") == ["p1", "0"]       # 공백이 있으면 갈라진다
    assert V.decode(V.ids_of(["p1", "0"])) == "p1 0"  # decode 가 그 공백을 되살린다
    with pytest.raises(ValueError):
        V.tokenize("é")


def test_every_token_single():
    assert len(V.TOKENS) == len(set(V.TOKENS))
    assert all(t in V.TOK2ID for t in V.MODEL_TOKENS + V.READONLY_TOKENS)
    assert not (set(V.MODEL_TOKENS) & set(V.READONLY_TOKENS))
    # 금지 단어는 어휘에 없다
    for bad in ["쪼개기", "나누기", "조각", "칸", "근사", "무한", "극한", "합계", "Σ", "dx", "h"]:
        assert bad not in V.TOK2ID


# ------------------------------------------------------------ 수식
def test_precedence():
    assert ev("2+3×4") == 14
    assert ev("-2^2") == -4
    assert ev("2^3^2") == 512      # 오른쪽 결합
    assert ev("8÷2÷2") == 2
    assert ev("rect(4,2)") == 8
    assert ev("10-3-2") == 5
    assert ev("3<5") is True and ev("5<3") is False and ev("5>3") is True


def test_errors():
    with pytest.raises(EvalError):
        ev("1÷0")
    for bad in ["1+", "()", "rect(1)", "1..2", "3 3 x +", "1.", "f(1"]:
        with pytest.raises(ParseError):
            parse_expr(V.tokenize(bad))


def test_fmt_num():
    assert fmt_num(12345.678) == "12350"
    assert fmt_num(0.00012345) == "0.0001234"   # 4자리 (round-half-even 은 float 표현 따름)
    assert fmt_num(-3) == "-3"
    assert fmt_num(2.50) == "2.5"
    assert fmt_num(1e-9) == "0"
    assert fmt_num(5e15) == "999999999999"
    assert fmt_num(100) == "100"
    assert "e" not in fmt_num(123456789012.0) and "e" not in fmt_num(1.5e-5)
    with pytest.raises(EvalError):
        fmt_num(float("inf"))


# ------------------------------------------------------------ 절차 해석기
def run(text, a=0.0, b=1.0, f=lambda x: x, store=None, strict=True):
    prog = parse_program(V.tokenize(text), strict=strict)
    return run_program(prog, {"a": a, "b": b, "f": f}, store if store is not None else {}, Ctx())


def test_loop_sum_needs_explicit_accumulate():
    assert run("정의 p0=0; 반복 N=10: 정의 p0=p0+i+1; p0") == 55
    # 반복 자체는 아무것도 더해주지 않는다
    with pytest.raises(EvalError):
        run("반복 N=3: 1")           # 마지막이 식이 아니라 반복 → 값 없음
    with pytest.raises(EvalError):
        run("정의 p0=1")             # 값 없음


def test_riemann_example_runs():
    # 계획서의 '기대하는 발견' 예시: x² 를 0..3 에서 (정답 9)
    v = run("정의 p0=0; 반복 N=1000: 정의 p0=p0+rect((b-a)÷N,f(a+i×(b-a)÷N)); p0",
            a=0, b=3, f=lambda x: x * x)
    assert abs(v - 9) / 9 < 0.0031


def test_lazy_definition_and_reuse_across_curves():
    store = {}
    # N 이 아직 없으니 값이 아니라 '식'으로 저장된다
    run("정의 p1=(b-a)÷N; 1", a=0, b=2, store=store)
    from integral.expr import Lazy
    assert isinstance(store["p1"], Lazy)
    # 다른 곡선(b-a=10)에서 반복 안에서 다시 계산된다
    v = run("정의 p0=0; 반복 N=5: 정의 p0=p0+p1; p0", a=0, b=10, store=store)
    assert v == 10
    # 자기 자신을 가리키는 정의는 순환하지 않는다 (정의 시점에 치환)
    s2 = {}
    run("정의 p2=N; 정의 p2=p2+1; 1", store=s2)
    run("반복 N=7: 정의 p3=p2; 1", store=s2)
    assert s2["p3"] == 8      # 반복 안에서 N=7 로 다시 계산
    with pytest.raises(EvalError):
        run("정의 p5=p9; 1", store={})   # 정의 안 된 슬롯


def test_strict_rules():
    for bad in ["정의 p0=0.5; p0", "10001", "반복 N=10001: 정의 p0=1; 1", "정의 p0=sin(1); p0",
                "정의 p0=2^2; p0", "반복 N=0: 정의 p0=1; 1"]:
        with pytest.raises(ParseError):
            run(bad)
    assert run("10000") == 10000
    assert run("0.5", strict=False) == 0.5   # 생각하는 중에는 소수 가능


def test_step_limit():
    with pytest.raises(EvalError):
        run("반복 N=1000: 반복 N=1000: 반복 N=1000: 정의 p0=1; 1")


def test_loop_vars_restored_and_nested():
    assert run("정의 p0=0; 반복 N=3: 반복 N=4: 정의 p0=p0+1; p0") == 12


# ------------------------------------------------------------ 곡선
@pytest.mark.parametrize("kind", TRAIN_KINDS)
def test_curves_accurate_positive_and_scaled(kind):
    rng = np.random.default_rng(1)
    tops = []
    for _ in range(25):
        c = make_curve(kind, rng)
        v, _ = quad(c.f, c.a, c.b, limit=2000, epsabs=0, epsrel=1e-10)
        assert abs(v - c.area) / c.area < 1e-8
        ys = c.f_np(np.linspace(c.a, c.b, 2001))
        assert ys.min() > 0
        tops.append(ys.max())
        assert -5 <= c.a <= 5 and 0.49 <= c.b - c.a <= 10.01
    assert min(tops) < 5 and max(tops) > 20      # 높이 크기가 넓게 퍼진다


def test_curve_text_is_source_of_truth():
    rng = np.random.default_rng(2)
    c = make_curve("trig", rng)
    toks = c.prompt_tokens()
    assert toks[0] == "문제:" and toks[-3:] == ["넓이?", "제출?", "\n"]
    rhs = toks[toks.index("=") + 1: toks.index("구간:")]
    f = compile_ast(parse_expr(rhs))
    assert math.isclose(f({"x": 0.7}, {}), c.f(0.7))
    # 구간 숫자도 글자 그대로 복원된다
    i = toks.index("구간:")
    assert "".join(toks[i + 3: toks.index("b")]) == fmt_num(c.a)


def test_family_mix():
    rng = np.random.default_rng(0)
    kinds = [random_curve(rng).kind for _ in range(600)]
    for k, p in zip(TRAIN_KINDS, (0.4, 0.3, 0.3)):
        assert abs(kinds.count(k) / 600 - p) < 0.07


def test_test_curves_separate_and_valid():
    rng = np.random.default_rng(5)
    for k in TEST_KINDS:
        assert k not in TRAIN_KINDS
        for _ in range(5):
            c = make_test_curve(k, rng)
            assert c.kind == k and c.area > 0 and c.f(c.a) > 0
            v, _ = quad(c.f, c.a, c.b, limit=5000, epsabs=0, epsrel=1e-8)
            assert abs(v - c.area) / c.area < 1e-6
    # 학습 생성기는 시험 전용 종류를 만들 수 없다
    with pytest.raises(KeyError):
        make_curve("test_stair", rng)
    stair = make_test_curve("test_stair", rng)
    assert "exp(-60×" in stair.text      # 계단 = 가파른 S자의 합
    long_ = make_test_curve("test_long", rng)
    assert long_.b - long_.a >= 29


def test_siblings_reproducible_same_kind():
    a = siblings("nonclosed", 123)
    b = siblings("nonclosed", 123)
    assert [c.text for c in a] == [c.text for c in b] and len(a) == 3
    assert all(c.kind == "nonclosed" for c in a)
    assert [c.text for c in a] != [c.text for c in siblings("nonclosed", 124)]


# ------------------------------------------------------------ 보상
def test_score_formula():
    assert math.isclose(score_from_errors([0.1]), 1.0)
    assert math.isclose(score_from_errors([0.01]), 2.0)
    assert math.isclose(score_from_errors([0.001]), 3.0)
    assert score_from_errors([2.0]) == 0.0
    assert score_from_errors([1e-12]) == 6.0
    assert score_from_errors([0.0]) == 6.0
    assert math.isclose(score_from_errors([0.0, 0.2]), 1.0)   # 평균의 -log10


# ------------------------------------------------------------ 메모장
def tiny_curve():
    f = lambda x: x * x
    fn = lambda x: x * x
    return Curve("poly", "x^2", 0.0, 3.0, 9.0, f, fn)


def nb_for(curve=None, hidden=None):
    curve = curve or tiny_curve()
    return Notebook([V.BOS] + curve.prompt_tokens(), curve=curve, hidden=hidden)


def test_calculator_basics_and_mask():
    nb = nb_for()
    nb.feed_text("rect(4,2)=\nrect(1,3)=\n8+3=\n3<5=\n5<3=\n1÷0=\nf(2)=\n(b-a)÷3=\n")
    text = nb.render()
    for want in ["rect(4,2)=‹8›", "rect(1,3)=‹3›", "8+3=‹11›", "3<5=‹참›", "5<3=‹거짓›",
                 "1÷0=‹오류›", "f(2)=‹4›", "(b-a)÷3=‹1›"]:
        assert want in text, (want, text)
    # 마스크: 프롬프트·계산기 결과는 0, 모델이 쓴 건 1
    assert all(m == 0 for m in nb.mask[:nb.n_prompt])
    forced = [V.TOKENS[i] for i, m in zip(nb.ids[nb.n_prompt:], nb.mask[nb.n_prompt:]) if m == 0]
    assert "".join(forced) == "8" + "3" + "11" + "참" + "거짓" + "오류" + "4" + "1"
    assert nb.violations == 0


def test_definition_equals_does_not_fire():
    nb = nb_for()
    nb.feed_text("정의 p0=5\n반복 N=2: 정의 p0=p0+1\np0=\n")
    assert nb.render().endswith("p0=‹7›\n")
    assert "정의 p0=5\n" in nb.render()    # 정의의 '=' 뒤에는 아무것도 안 끼어든다
    # 저장소는 notebook 밖으로 꺼낼 수 있다 (다음 곡선으로 유지)
    nb2 = Notebook([V.BOS], store=nb.store)
    nb2.feed_text("p0+1=\n")
    assert nb2.render().endswith("‹8›\n")


def test_statement_error_marked():
    nb = nb_for()
    nb.feed_text("정의 p0=1÷0\n")
    assert nb.render().endswith("정의 p0=1÷0\n‹오류›")


def test_submit_feedback_and_scoring():
    nb = nb_for()
    nb.feed_text("제출[ 3 ]\n")
    assert "→ 오차 66.67%›\n" in nb.render()                  # |3-9|/9
    nb.feed_text("제출[ 정의 p0=0; 반복 N=3000: 정의 p0=p0+rect(3÷N,f(3×i÷N)); p0 ]\n")
    assert nb.last_feedback < 1e-3
    assert 2.5 < nb.final_score() < 4
    nb.feed_text("제출[ 정의 p0=0.5; p0 ]\n")          # 소수 → 실행 불가
    assert nb.render().rstrip().endswith("→ 오류›")
    assert nb.final_score() == 0.0                        # 마지막 제출만 채점


def test_only_last_submission_counts():
    nb = nb_for()
    nb.feed_text("제출[ 정의 p0=0; 반복 N=500: 정의 p0=p0+rect(3÷N,f(3×i÷N)); p0 ]\n")
    s_good = nb.final_score()
    nb.feed_text("제출[ 9 ]\n")             # 이 곡선만 정답 → 못 본 곡선에서 틀려야 한다
    assert s_good > 2


def test_unseen_curves_catch_memorizing():
    # 지금 곡선의 넓이를 그대로 외워 쓰면 못 본 곡선에서 0점에 가깝다
    rng = np.random.default_rng(11)
    c = make_curve("poly", rng)
    nb = new_episode(c, seed=77)
    m = 3 - math.floor(math.log10(c.area))      # 정수 상한 10000 안에서 가장 정밀하게
    nb.feed_text(f"제출[ {int(round(c.area * 10 ** m))}÷{10 ** m} ]\n")
    assert nb.last_feedback < 1e-3         # 지금 곡선만 보면 거의 맞다
    assert nb.final_score() < 1.0          # 하지만 4개 평균은 낮다
    # 올바른 방법은 어느 곡선에서도 통한다
    nb2 = new_episode(c, seed=77)
    nb2.feed_text("제출[ 정의 p0=0; 반복 N=2000: 정의 p0=p0+rect((b-a)÷N,f(a+i×(b-a)÷N)); p0 ]\n")
    assert nb2.final_score() > 2.5


def test_constant_shortcut_is_poor_with_random_heights():
    # 높이를 재지 않고 a, b 만 조합하는 지름길: 높이가 0.1~100배로 퍼져 있어 점수가 낮다
    rng = np.random.default_rng(4)
    scores = []
    for s in range(12):
        c = random_curve(rng)
        nb = new_episode(c, seed=s)
        nb.feed_text("제출[ rect(b-a,10) ]\n")
        scores.append(nb.final_score())
    assert np.mean(scores) < 0.5


def test_no_submit_or_violation_is_zero():
    nb = nb_for()
    nb.feed_text("rect(1,2)=\n")
    assert nb.final_score() == 0.0
    nb = nb_for()
    nb.feed(V.TOK2ID["오차"])              # 읽기 전용 기호를 모델이 씀
    assert nb.violations == 1
    nb.feed_text("제출[ 정의 p0=0; 반복 N=100: 정의 p0=p0+rect(3÷N,f(3×i÷N)); p0 ]\n")
    assert nb.final_score() == 0.0


def test_limits_end_episode():
    nb = nb_for()
    for _ in range(8):
        nb.feed_text("제출[ 1 ]\n")
    assert nb.done and len(nb.submissions) == 8
    with pytest.raises(RuntimeError):
        nb.feed(V.TOK2ID["1"])
    nb = Notebook([V.BOS], max_len=20)
    nb.feed_text("1+1=" * 50)
    assert nb.done and len(nb.ids) <= 22
    nb = nb_for()
    nb.feed(V.EOS_ID)
    assert nb.done


def test_nested_or_unclosed_submit():
    nb = nb_for()
    nb.feed_text("제출[ 1 제출[ 2 ]\n")
    assert nb.render().rstrip().endswith("→ 오류›")
    nb = nb_for()
    nb.feed_text("제출[ 1 ")                # 닫지 않고 끝 → 제출 없음
    assert nb.final_score() == 0.0


def test_thinking_may_use_decimals_and_free_numbers():
    nb = nb_for()
    nb.feed_text("f(0.5)=\n")
    assert nb.render().endswith("f(0.5)=‹0.25›\n")


def test_hidden_failure_gives_zero():
    # 지금 곡선에서는 되지만 다른 곡선에서 실행이 안 되는 절차 → 0점
    c = tiny_curve()
    bad = Curve("poly", "x^2", 0.0, 3.0, 9.0, lambda x: (_ for _ in ()).throw(EvalError("x")), None)
    nb = Notebook([V.BOS], curve=c, hidden=[bad])
    nb.feed_text("제출[ f(a) ]\n")
    assert nb.final_score() == 0.0


def test_performance_big_loop():
    import time
    rng = np.random.default_rng(0)
    c = random_curve(rng)
    nb = new_episode(c, seed=1)
    t = time.time()
    nb.feed_text("제출[ 정의 p0=0; 반복 N=10000: 정의 p0=p0+rect((b-a)÷N,f(a+i×(b-a)÷N)); p0 ]\n")
    s = nb.final_score()
    assert time.time() - t < 8 and s > 3


# ------------------------------------------------------------ 모델·생성기
def test_cached_step_matches_full_forward_and_generation():
    import torch
    from integral.generate import run_notebooks, run_notebooks_cached
    from integral.model import GPT, Config

    torch.manual_seed(0)
    m = GPT(Config(layers=2, dim=32, heads=2, ff=64))
    ids = torch.randint(3, V.VOCAB_SIZE, (3, 17))
    full = m(ids)
    cache = m.init_cache(3, 20)
    for t in range(17):
        step = m.step(ids[:, t], torch.full((3,), t), cache)
        assert torch.allclose(step, full[:, t], atol=1e-4)
    # 줄마다 위치가 달라도 같은 결과
    cache = m.init_cache(2, 20)
    a = torch.randint(3, V.VOCAB_SIZE, (2, 6))
    for t in range(6):
        m.step(a[:, t], torch.full((2,), t), cache)
    out = m.step(a[:, 0], torch.tensor([6, 6]), cache)       # 임의 토큰을 7번째 칸에
    ref = m(torch.cat([a, a[:, :1]], 1))[:, 6]
    assert torch.allclose(out, ref, atol=1e-4)

    # 두 생성기가 같은 글을 쓴다 (greedy, 계산기 끼워넣기 포함)
    def fresh():
        return [Notebook([V.BOS] + tiny_curve().prompt_tokens(), curve=tiny_curve(), max_len=60) for _ in range(3)]
    n1, n2 = fresh(), fresh()
    run_notebooks(m, n1, max_new=60)
    run_notebooks_cached(m, n2)
    assert [n.ids for n in n1] == [n.ids for n in n2]
