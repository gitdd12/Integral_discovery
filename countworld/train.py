"""학습 루프: 강화학습(주) + 지도학습 점검(환경이 풀 수 있는지 확인용)."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import Backbone, Judge, Speaker
from .scene import SceneCfg, make_batch


def usage_entropy(probs0):
    """배치 전체에서 첫 기호가 얼마나 고르게 쓰이는지(주변 엔트로피). 기호를 하나로 몰아 쓰는 것을 막는 약한 정규화."""
    q = probs0.mean(0)
    return -(q * (q + 1e-9).log()).sum()


def to_t(batch, dev):
    return {k: (torch.as_tensor(v, device=dev) if isinstance(v, np.ndarray) else v) for k, v in batch.items()}


class Agent:
    def __init__(self, V, Lmax, d=64, layers=2, dev="cpu"):
        self.V, self.Lmax, self.dev = V, Lmax, dev
        self.bb = Backbone(d, layers).to(dev)
        self.speaker = Speaker(V, Lmax, self.bb, d).to(dev)
        self.judge = Judge(V, Lmax, self.bb, d).to(dev)

    def params(self):
        seen, out = set(), []
        for p in list(self.bb.parameters()) + list(self.speaker.parameters()) + list(self.judge.parameters()):
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        return out


@torch.no_grad()
def evaluate(agent, cfg, rng, n=2048, eps=0.0, min_len=0, **kw):
    """탐욕(가장 확률 높은 기호, 잡음 없는 판정자 평균)으로 정확도."""
    b = to_t(make_batch(cfg, rng, n, **kw), agent.dev)
    sp = agent.speaker(b["s1"], b["l1"], mode="greedy", eps=eps, min_len=min_len)
    m = agent.judge(sp["syms"], b["s2"], b["l2"])
    pred = m > 0
    acc = (pred == b["answer"]).float().mean().item()
    return acc, sp["lens"].mean().item()


def train_sup(agent, cfg, steps=3000, B=256, lr=3e-4, wd=0.05, seed=0, beta_fn=None, log=print, eps=0.0, min_len=0, smode="gumbel", usage=0.0):
    """지도학습 점검: 오차가 메시지를 거쳐 인코더까지 전달(gumbel straight-through). 풀 수 있는 환경인지 보려는 용도."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(agent.params(), lr=lr, weight_decay=wd)
    t0 = time.time()
    for step in range(steps):
        beta = beta_fn(step / steps) if beta_fn else 0.0
        b = to_t(make_batch(cfg, rng, B, beta=beta), agent.dev)
        tau = max(0.5, 2.0 * (1 - step / steps))
        sp = agent.speaker(b["s1"], b["l1"], mode=smode, tau=tau, min_len=min_len)
        soft = torch.stack(sp["soft"], 1)                                  # (B, Lmax, V+2)
        me = soft @ agent.judge.msg_emb.weight
        alive = soft.sum(-1) > 0.5
        m = agent.judge(None, b["s2"], b["l2"], msg_embed=me, msg_alive=alive)
        loss = F.binary_cross_entropy_with_logits(m, b["answer"].float()) - usage * usage_entropy(sp["probs0"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.params(), 1.0)
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            acc, ln = evaluate(agent, cfg, rng, eps=eps, min_len=min_len)
            log(f"[sup] step {step} loss {loss.item():.3f} acc {acc:.3f} len {ln:.2f} ({time.time()-t0:.0f}s)")
    return agent


def train_rl(agent, cfg, steps=20000, B=256, lr=3e-4, wd=0.05, lam=0.0, eps=0.0, sigma0=1.0, sigma1=0.3,
             ent_coef=0.01, score="log", beta_fn=None, seed=0, log=print, eval_every=500, stop_acc=None, min_len=0, usage=0.0,
             step0=0, opt_state=None, on_ckpt=None, ckpt_every=1000, hist=None):
    """강화학습: 점수 숫자 하나만 받아 정책 경사법(평균 기준값). 점수식을 미분해서 쓰지 않는다.
    score: 'log' 확신도 로그 점수 / 'zero_one' 맞음·틀림."""
    rng = np.random.default_rng(seed + step0)
    opt = torch.optim.AdamW(agent.params(), lr=lr, weight_decay=wd)
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, i / steps))))
    for _ in range(step0):
        sched.step()
    t0 = time.time()
    hist = [] if hist is None else hist
    for step in range(step0, steps):
        beta = beta_fn(step / steps) if beta_fn else 0.0
        b = to_t(make_batch(cfg, rng, B, beta=beta), agent.dev)
        sp = agent.speaker(b["s1"], b["l1"], mode="sample", eps=eps, min_len=min_len)
        m = agent.judge(sp["syms"], b["s2"], b["l2"])
        sigma = sigma0 + (sigma1 - sigma0) * min(1.0, step / (0.5 * steps))
        z = (m + sigma * torch.randn_like(m)).detach()                    # 판정자의 행동(로짓), 잡음으로 시도
        same = b["answer"]
        if score == "log":
            r_j = torch.where(same, F.logsigmoid(z), F.logsigmoid(-z))   # log(정답에 준 확률)
        else:
            r_j = ((z > 0) == same).float()
        r_e = r_j - lam * sp["lens"]
        adv_j = r_j - r_j.mean()                                           # 기준값만 빼고 크기는 보존 (표준편차로 나누면 불안정했다)
        adv_e = r_e - r_e.mean()
        logp_j = -((z - m) ** 2) / (2 * sigma ** 2)                       # 가우시안 정책의 로그확률 (상수 제외)
        loss = -(adv_j.detach() * logp_j).mean() - (adv_e.detach() * sp["logp"]).mean() - ent_coef * sp["ent"].mean() - usage * usage_entropy(sp["probs0"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.params(), 1.0)
        opt.step()
        sched.step()
        if on_ckpt and (step + 1) % ckpt_every == 0:
            on_ckpt(step + 1, opt, hist)
        if step % eval_every == 0 or step == steps - 1:
            acc, ln = evaluate(agent, cfg, rng, eps=eps, min_len=min_len)
            hist.append((step, acc, ln, r_j.mean().item()))
            log(f"[rl:{score}] step {step} score {r_j.mean().item():.3f} acc {acc:.3f} len {ln:.2f} "
                f"ent {sp['ent'].mean().item():.2f} ({time.time()-t0:.0f}s)")
            if stop_acc and acc >= stop_acc:
                break
    return hist


def train_pair(agent, cfg, steps=3000, B=256, lr=1e-3, wd=0.05, mode="rl", sigma=0.5, seed=0, log=print,
               eval_every=250, stop_acc=0.97, beta_fn=None, rand_ramp=0.0, rand_lo=0.0, rand_hi=1.0):
    """단계 0: 두 장면이 모두 보이는 개수 비교(T2). 메시지 없음, 개수 라벨 없음 (같다/다르다 결과만).
    mode: 'rl' 점수 하나만 / 'sup' 점검용."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(agent.params(), lr=lr, weight_decay=wd)
    t0 = time.time()
    for step in range(steps):
        beta = beta_fn(step / steps) if beta_fn else 0.0
        if rand_ramp:
            agent.bb.rand_p = rand_lo + (rand_hi - rand_lo) * min(1.0, step / (rand_ramp * steps))
        b = to_t(make_batch(cfg, rng, B, beta=beta), agent.dev)
        m = agent.judge.forward_pair(b["s1"], b["l1"], b["s2"], b["l2"])
        same = b["answer"]
        if mode == "sup":
            loss = F.binary_cross_entropy_with_logits(m, same.float())
        else:
            z = (m + sigma * torch.randn_like(m)).detach()
            r = torch.where(same, F.logsigmoid(z), F.logsigmoid(-z))
            adv = r - r.mean()
            loss = -(adv.detach() * (-((z - m) ** 2) / (2 * sigma ** 2))).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.params(), 1.0)
        opt.step()
        if step % eval_every == 0 or step == steps - 1:
            with torch.no_grad():
                e = to_t(make_batch(cfg, rng, 2048), agent.dev)
                acc = ((agent.judge.forward_pair(e["s1"], e["l1"], e["s2"], e["l2"]) > 0) == e["answer"]).float().mean().item()
            log(f"[pair:{mode}] step {step} acc {acc:.3f} ({time.time()-t0:.0f}s)")
            if stop_acc and acc >= stop_acc:
                break
    return agent


def curriculum_pair(agent, stages, log=print, **kw):
    """개수 범위를 단계적으로 늘리며 T2 학습. stages: [(max_count, extra, rand_lo, rand_hi, max_steps), ...]
    각 단계는 보류 정확도 95%에 닿으면 다음으로."""
    for i, (mc, extra, lo, hi, ms) in enumerate(stages):
        cfg = SceneCfg(max_count=mc, extra=extra)
        log(f"--- 단계 {i}: 개수 0~{mc} (이름 벡터 무작위 비율 {lo}→{hi})")
        train_pair(agent, cfg, steps=ms, stop_acc=0.95, rand_ramp=0.5, rand_lo=lo, rand_hi=hi, log=log, **kw)
    return agent
