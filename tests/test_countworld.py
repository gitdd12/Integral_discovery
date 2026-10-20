import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from countworld.scene import BOS, QRY, SEP, T0, SceneCfg, make_batch  # noqa: E402
from countworld.train import Agent, train_pair, train_rl, train_sup  # noqa: E402


def test_labels_and_counts():
    rng = np.random.default_rng(0)
    cfg = SceneCfg()
    b = make_batch(cfg, rng, 500, beta=1.0)
    for i in range(500):
        for s, l, q, c in ((b["s1"][i], b["l1"][i], b["qa"][i], b["a"][i]), (b["s2"][i], b["l2"][i], b["qb"][i], b["b"][i])):
            body = s[1:l - 3]
            assert s[0] == BOS and s[l - 3] == SEP and s[l - 2] == QRY and s[l - 1] == T0 + q
            assert (body == T0 + q).sum() == c
        assert len(b["s1"][i][1:b["l1"][i] - 3]) == b["T1"][i] and len(b["s2"][i][1:b["l2"][i] - 3]) == b["T2"][i]
        assert bool(b["answer"][i]) == (b["a"][i] == b["b"][i])
    assert (b["l1"] != b["l2"]).any()          # 두 장면 전체 길이는 독립이라 서로 다를 수 있다


def test_marginals_independent_of_answer():
    rng = np.random.default_rng(1)
    b = make_batch(SceneCfg(), rng, 60000, beta=1.5)
    ans = b["answer"]
    for name in ("a", "b", "T1", "T2"):
        x = b[name]
        h1 = np.bincount(x[ans], minlength=20) / ans.sum()
        h0 = np.bincount(x[~ans], minlength=20) / (~ans).sum()
        assert np.abs(h1 - h0).max() < 0.02, name


def test_heldout_pairs_disjoint():
    rng = np.random.default_rng(2)
    cfg = SceneCfg(max_count=30)
    tr = make_batch(cfg, rng, 20000)
    ho = make_batch(cfg, rng, 20000, pairs="heldout")
    s_tr = {(min(a, b), max(a, b)) for a, b in zip(tr["a"], tr["b"]) if a != b}
    s_ho = {(min(a, b), max(a, b)) for a, b in zip(ho["a"], ho["b"]) if a != b}
    assert not (s_tr & s_ho)


def test_shortcut_probe_chance():
    from countworld.shortcuts import probe
    cfg = SceneCfg()
    assert abs(probe(cfg, "nocount", n_train=6000, n_test=3000, epochs=10) - 0.5) < 0.05
    assert abs(probe(cfg, "scene2", n_train=6000, n_test=3000, epochs=10) - 0.5) < 0.06


def test_training_steps_run():
    torch.manual_seed(0)
    cfg = SceneCfg(max_count=3, extra=2)
    ag = Agent(V=4, Lmax=2)
    train_pair(ag, cfg, steps=3, B=16, eval_every=2, stop_acc=None, log=lambda s: None)
    train_sup(ag, cfg, steps=3, B=16, log=lambda s: None, smode="st")
    train_rl(ag, cfg, steps=3, B=16, log=lambda s: None, eval_every=2, lam=0.01, eps=0.1)
