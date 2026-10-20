"""학습 루프: 강화학습(주) + 지도학습 점검(환경이 풀 수 있는지 확인용)."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import Backbone, Judge, Speaker
from .scene import SceneCfg, make_batch


def usage_entropy(probs0, mi_cond=0.0):
    """배치 전체에서 첫 기호가 얼마나 고르게 쓰이는지(주변 엔트로피). 기호를 하나로 몰아 쓰는 것을 막는 약한 정규화.
    mi_cond > 0 이면 '장면 하나당 기호는 결정적'일수록 보너스 (= 기호와 장면 사이 상호정보량): 개수와 무관하게 반반 섞어 쓰는 편법을 막는다."""
    q = probs0.mean(0)
    h_marg = -(q * (q + 1e-9).log()).sum()
    if mi_cond == 0.0:
        return h_marg
    h_cond = -(probs0 * (probs0 + 1e-9).log()).sum(-1).mean()
    return h_marg - mi_cond * h_cond


def to_t(batch, dev):
    return {k: (torch.as_tensor(v, device=dev) if isinstance(v, np.ndarray) else v) for k, v in batch.items()}


class Agent:
    def __init__(self, V, Lmax, d=64, layers=2, dev="cpu", K=1):
        self.V, self.Lmax, self.dev, self.K = V, Lmax, dev, K
        self.bb = Backbone(d, layers).to(dev)
        self.speaker = Speaker(V, Lmax, self.bb, d).to(dev)
        self.judges = [Judge(V, Lmax, self.bb, d).to(dev) for _ in range(K)]
        self.judge = self.judges[0]
        self.head0 = None            # 단계 0 직후 비교 머리(새 청자의 시작점)
        self.refresh_ptr = 0

    def params(self):
        seen, out = set(), []
        mods = [self.bb, self.speaker] + self.judges
        for m in mods:
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
        return out

    def sync_listeners(self):
        """단계 0 뒤: 첫 청자의 가중치를 모든 청자에 복사하고 비교 머리를 저장."""
        import copy
        for j in self.judges[1:]:
            j.load_state_dict(self.judges[0].state_dict())
        self.head0 = copy.deepcopy(self.judges[0].head.state_dict())

    def snapshot(self):
        import copy
        return {"bb": copy.deepcopy(self.bb.state_dict()), "speaker": copy.deepcopy(self.speaker.state_dict()),
                "judges": [copy.deepcopy(j.state_dict()) for j in self.judges]}

    def load_snapshot(self, snap):
        self.bb.load_state_dict(snap["bb"])
        self.speaker.load_state_dict(snap["speaker"])
        for j, sd in zip(self.judges, snap["judges"]):
            j.load_state_dict(sd)

    def restart_heads(self, opt=None):
        """눈(몸통)은 두고, 화자의 기호 선택 부분과 판정자의 쪽지 읽는 부분만 처음부터 다시 시작.
        판정자의 비교 머리는 단계 0 직후 상태로 되돌린다."""
        sp = self.speaker
        mods = [sp.init, sp.gru, sp.head, sp.sym_emb]
        for j in self.judges:
            j.msg_emb.reset_parameters()
            j.msg_gru.reset_parameters()
            if self.head0 is not None:
                j.head.load_state_dict(self.head0)
            mods += [j.msg_emb, j.msg_gru, j.head]
        for m in mods[:4]:
            m.reset_parameters()
        if opt is not None:
            for m in mods:
                for p in m.parameters():
                    opt.state.pop(p, None)

    def listen(self, msg, tok, length):
        """청자 평균 로짓 (평가용)."""
        return torch.stack([j(msg, tok, length) for j in self.judges]).mean(0)

    def refresh_listener(self, opt=None):
        """가장 오래된 청자의 기호 읽는 부분을 새로 초기화 (비교 머리는 단계 0 상태로)."""
        j = self.judges[self.refresh_ptr % self.K]
        j.msg_emb.reset_parameters()
        j.msg_gru.reset_parameters()
        j.head.load_state_dict(self.head0)
        if opt is not None:
            for mod in (j.msg_emb, j.msg_gru, j.head):
                for p in mod.parameters():
                    opt.state.pop(p, None)
        self.refresh_ptr += 1


@torch.no_grad()
def evaluate(agent, cfg, rng, n=2048, eps=0.0, min_len=0, **kw):
    """탐욕(가장 확률 높은 기호, 잡음 없는 판정자 평균)으로 정확도."""
    b = to_t(make_batch(cfg, rng, n, **kw), agent.dev)
    sp = agent.speaker(b["s1"], b["l1"], mode="greedy", eps=eps, min_len=min_len)
    m = agent.listen(sp["syms"], b["s2"], b["l2"])
    pred = m > 0
    acc = (pred == b["answer"]).float().mean().item()
    return acc, sp["lens"].mean().item()


def train_sup(agent, cfg, steps=3000, B=256, lr=3e-4, wd=0.05, seed=0, beta_fn=None, log=print, eps=0.0, min_len=0, smode="gumbel", usage=0.0, mi_cond=0.0):
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
        loss = F.binary_cross_entropy_with_logits(m, b["answer"].float()) - usage * usage_entropy(sp["probs0"], mi_cond)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.params(), 1.0)
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            acc, ln = evaluate(agent, cfg, rng, eps=eps, min_len=min_len)
            log(f"[sup] step {step} loss {loss.item():.3f} acc {acc:.3f} len {ln:.2f} ({time.time()-t0:.0f}s)")
    return agent


@torch.no_grad()
def adjacent_score(agent, cfg, rng, n=256, min_len=0, eps=0.0):
    """n 대 n+1 쌍 전체의 균형 잡힌 정확도 (재시작·최선 선택의 검증 점수. 개수 정답이 아니라 같다/다르다 결과만 쓴다)."""
    accs = []
    for k in range(cfg.max_count):
        # 같은 쌍 (k,k),(k+1,k+1) 과 다른 쌍 (k,k+1),(k+1,k) 를 반반 섞는다 (다른 쌍만 내면 늘 "다르다"고 답해도 만점이 된다)
        a = np.array([k, k, k + 1, k + 1] * (n // 4))
        b = np.array([k, k + 1, k, k + 1] * (n // 4))
        bt = to_t(make_batch(cfg, rng, n, a=a, b=b), agent.dev)
        sp = agent.speaker(bt["s1"], bt["l1"], mode="greedy", eps=eps, min_len=min_len)
        m = agent.listen(sp["syms"], bt["s2"], bt["l2"])
        accs.append(((m > 0) == bt["answer"]).float().mean().item())
    return float(np.mean(accs))


def train_rl(agent, cfg, steps=20000, B=256, lr=3e-4, wd=0.05, lam=0.0, eps=0.0, sigma0=1.0, sigma1=0.3,
             ent_coef=0.01, score="log", beta_fn=None, seed=0, log=print, eval_every=500, stop_acc=None, min_len=0, usage=0.0, mi_cond=0.0, mi_ramp=0.0, refresh_every=0, refresh_stop=0.8, focus=0.0, focus_start=500,
             restart_at=(), restart_thresh=0.95, restart_max=2, meta=None,
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
    err_tab = np.full((cfg.max_count + 1, cfg.max_count + 1), 0.5)       # 쌍별 최근 틀린 비율
    if meta is None:
        meta = dict(last_restart=0, done=0, best=-1.0, best_state=None, log=[])
    restart_steps = {int(f * steps) for f in restart_at}
    vrng = np.random.default_rng(777 + seed)
    for step in range(step0, steps):
        if step in restart_steps and step > 0:                              # 재시작 점검: 검증 점수가 낮으면 기호 부분만 다시 시작
            val = adjacent_score(agent, cfg, vrng, min_len=min_len, eps=eps)
            if val > meta["best"]:
                meta["best"], meta["best_state"] = val, agent.snapshot()
            did = bool(val < restart_thresh and meta["done"] < restart_max)
            meta["log"].append((step, round(val, 4), did))
            log(f"[점검] step {step} 검증 점수(n 대 n+1 평균) {val:.3f} → {'재시작' if did else '유지'}")
            if did:
                agent.restart_heads(opt)
                meta["last_restart"] = step
                meta["done"] += 1
        beta = beta_fn(step / steps) if beta_fn else 0.0
        pw = None
        if focus and step >= focus_start:                                  # 실패 중심 출제: 최근에 많이 틀린 (개수, 개수) 쌍을 더 자주
            n_c = cfg.max_count + 1
            if step % 50 == 0 or "fail_w" not in dir():
                fail_w = (err_tab + 0.05) ** focus
                fail_w = np.maximum(fail_w, fail_w.T)
            pw = fail_w
        b = to_t(make_batch(cfg, rng, B, beta=beta, pair_weight=pw), agent.dev)
        if refresh_every and step > 0 and step % refresh_every == 0 and step < refresh_stop * steps:
            agent.refresh_listener(opt)
        sp = agent.speaker(b["s1"], b["l1"], mode="sample", eps=eps, min_len=min_len)
        if agent.K == 1:
            m = agent.judge(sp["syms"], b["s2"], b["l2"])
        else:                                                              # 판마다 무작위 청자 한 명
            who = torch.randint(0, agent.K, (B,), device=agent.dev)
            m = torch.zeros(B, device=agent.dev)
            for k, jk in enumerate(agent.judges):
                idx = (who == k).nonzero(as_tuple=True)[0]
                if len(idx):
                    m = m.index_copy(0, idx, jk(sp["syms"][idx], b["s2"][idx], b["l2"][idx]))
        rel = step - meta["last_restart"]                                   # 마지막 재시작 이후 경과 단계
        sigma = sigma0 + (sigma1 - sigma0) * min(1.0, rel / (0.5 * steps))
        z = (m + sigma * torch.randn_like(m)).detach()                    # 판정자의 행동(로짓), 잡음으로 시도
        same = b["answer"]
        if score == "log":
            r_j = torch.where(same, F.logsigmoid(z), F.logsigmoid(-z))   # log(정답에 준 확률)
        else:
            r_j = ((z > 0) == same).float()
        if focus:
            wrong = (((z > 0) != same).float()).cpu().numpy()
            ia, ib = np.minimum(b["a"].cpu().numpy(), b["b"].cpu().numpy()), np.maximum(b["a"].cpu().numpy(), b["b"].cpu().numpy())
            for x, y, w in zip(ia, ib, wrong):
                err_tab[x, y] = 0.98 * err_tab[x, y] + 0.02 * w
                err_tab[y, x] = err_tab[x, y]
        r_e = r_j - lam * sp["lens"]
        adv_j = r_j - r_j.mean()                                           # 기준값만 빼고 크기는 보존 (표준편차로 나누면 불안정했다)
        adv_e = r_e - r_e.mean()
        logp_j = -((z - m) ** 2) / (2 * sigma ** 2)                       # 가우시안 정책의 로그확률 (상수 제외)
        ramp = min(1.0, rel / (mi_ramp * steps)) if mi_ramp else 1.0       # 초반에는 탐색 보너스, 점차 '더미당 기호 결정' 규칙으로
        loss = (-(adv_j.detach() * logp_j).mean() - (adv_e.detach() * sp["logp"]).mean()
                - ent_coef * (1.0 - ramp if mi_ramp else 1.0) * sp["ent"].mean() - usage * usage_entropy(sp["probs0"], mi_cond * ramp))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.params(), 1.0)
        opt.step()
        sched.step()
        if on_ckpt and (step + 1) % ckpt_every == 0:
            on_ckpt(step + 1, opt, hist, meta)
        if step % eval_every == 0 or step == steps - 1:
            acc, ln = evaluate(agent, cfg, rng, eps=eps, min_len=min_len)
            hist.append((step, acc, ln, r_j.mean().item()))
            log(f"[rl:{score}] step {step} score {r_j.mean().item():.3f} acc {acc:.3f} len {ln:.2f} "
                f"ent {sp['ent'].mean().item():.2f} ({time.time()-t0:.0f}s)")
            if stop_acc and acc >= stop_acc:
                break
    if restart_at:                                                         # 마지막 검증 후 최선의 상태로 복원
        val = adjacent_score(agent, cfg, vrng, min_len=min_len, eps=eps)
        if val >= meta["best"]:
            meta["best"] = val
            log(f"[최종] 검증 점수 {val:.3f} (최선) 유지")
        elif meta["best_state"] is not None:
            log(f"[최종] 검증 점수 {val:.3f} < 이전 최선 {meta['best']:.3f} → 최선 상태로 복원")
            agent.load_snapshot(meta["best_state"])
    agent.meta = meta
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
