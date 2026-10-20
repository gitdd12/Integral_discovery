"""인코더(장면 1 → 기호 메시지)와 판정자(메시지 + 장면 2 → 같을 확률의 로짓).

메시지 기호: 0..V-1, EOS=V, PAD=V+1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scene import N_TYPES, T0, VOCAB

MAXPOS = 512


class Backbone(nn.Module):
    """장면을 읽는 트랜스포머. 화자(장면 1)와 판정자(메시지 + 장면 2)가 같은 몸통을 쓴다.

    입력 토큰: 특수 토큰(PAD/BOS/SEP/QRY) 학습 벡터, 물체는 '질의한 종류와 같은가'만 표시한 두 벡터(같음/다름).
    (사물 구분과 같은 종류 알아보기는 지각의 기본 도구로 주고, 개수 표현은 주지 않는다.)
    """

    def __init__(self, d=64, layers=2, heads=4):
        super().__init__()
        self.sp_emb = nn.Embedding(T0, d)
        self.obj_emb = nn.Embedding(2, d)                                    # 0 다른 종류, 1 질의한 종류
        self.pos = nn.Embedding(MAXPOS, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.d = d

    def embed_scene(self, tok, length):
        B = tok.shape[0]
        query = tok[torch.arange(B, device=tok.device), length - 1]
        is_obj = tok >= T0
        match = (tok == query[:, None]).long()
        sp = self.sp_emb(tok.clamp(max=T0 - 1))
        ob = self.obj_emb(match)
        return torch.where(is_obj[..., None], ob, sp)

    def run(self, x, mask):
        x = x + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        return self.enc(x, src_key_padding_mask=mask)

    def encode(self, tok, length):
        B, L = tok.shape
        mask = torch.arange(L, device=tok.device)[None, :] >= length[:, None]
        h = self.run(self.embed_scene(tok, length), mask)
        return h[torch.arange(B), length - 1]


class Speaker(nn.Module):
    """장면 1 → 기호 열 (GRU 디코더). 강화학습용 표본 추출과 지도학습 점검용 straight-through 둘 다 지원."""

    def __init__(self, V, Lmax, bb, d=64):
        super().__init__()
        self.V, self.Lmax = V, Lmax
        self.bb = bb
        self.init = nn.Linear(d, d)
        self.sym_emb = nn.Embedding(V + 2, d)           # 0..V-1, EOS, PAD(=시작)
        self.gru = nn.GRUCell(d, d)
        self.head = nn.Linear(d, V + 1)

    def forward(self, tok, length, mode="sample", tau=1.0, eps=0.0, min_len=0, temp=None):
        B = tok.shape[0]
        dev = tok.device
        h = torch.tanh(self.init(self.bb.encode(tok, length)))
        V = self.V
        inp = self.sym_emb(torch.full((B,), V + 1, device=dev))
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        syms = torch.full((B, self.Lmax), V + 1, dtype=torch.long, device=dev)
        logp = torch.zeros(B, device=dev)
        ent = torch.zeros(B, device=dev)
        probs0 = None
        soft = []                                         # 지도학습 점검용 (B, Lmax, V+2) one-hot(straight-through)
        alive_f = torch.ones(B, device=dev)
        for t in range(self.Lmax):
            h = self.gru(inp, h)
            logits = self.head(h)
            if t == 0:
                probs0 = F.softmax(logits, -1)
            if t < min_len:                                   # 최소 길이: 아직 끝내기(EOS) 금지
                logits = logits.clone()
                logits[:, V] = -1e9
            if mode in ("gumbel", "st"):
                if mode == "gumbel":
                    y = F.gumbel_softmax(logits, tau=tau, hard=True)           # (B, V+1)
                else:                                                          # 잡음 없는 straight-through
                    soft_y = F.softmax(logits / tau, -1)
                    hard_y = F.one_hot(logits.argmax(-1), V + 1).float()
                    y = hard_y + soft_y - soft_y.detach()
                a = y.argmax(-1)
                oh = torch.cat([y, torch.zeros(B, 1, device=dev)], -1)          # PAD 열 추가
                soft.append(oh * alive_f[:, None])
                inp = oh @ self.sym_emb.weight
                alive_f = alive_f * (1.0 - y[:, V])
                syms[:, t] = torch.where(alive, a, syms[:, t])
                alive = alive & (a != V)
                continue
            if temp is not None:                              # 표본마다 탐색 온도(1보다 크면 더 흩어짐)
                logits = logits / temp[:, None]
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample() if mode == "sample" else logits.argmax(-1)
            m = alive.float()
            logp = logp + dist.log_prob(a) * m
            ent = ent + dist.entropy() * m
            sent = torch.where(alive, a, torch.full_like(a, V + 1))
            syms[:, t] = sent
            alive = alive & (a != V)
            inp = self.sym_emb(a)
        # 전달 잡음: 기호(EOS 제외)가 확률 eps 로 무작위 기호로 바뀜
        if eps > 0:
            flip = (torch.rand(syms.shape, device=dev) < eps) & (syms < V)
            syms = torch.where(flip, torch.randint(0, V, syms.shape, device=dev), syms)
        lens = (syms < V).sum(1).float()
        return dict(syms=syms, logp=logp, ent=ent, lens=lens, soft=soft, probs0=probs0)


class Judge(nn.Module):
    """판정자: 장면 2 를 읽은 벡터와 메시지를 읽은 벡터를 비교한다 (쌍둥이 구조)."""

    def __init__(self, V, Lmax, bb, d=64):
        super().__init__()
        self.V, self.Lmax = V, Lmax
        self.bb = bb
        self.msg_emb = nn.Embedding(V + 2, d)
        self.msg_gru = nn.GRU(d, d, batch_first=True)
        self.from_scene = nn.Linear(d, d)          # 보조 과제(T2)에서 장면 1 벡터를 '메시지 자리'로 옮김
        self.head = nn.Sequential(nn.Linear(4 * d, 2 * d), nn.ReLU(), nn.Linear(2 * d, 1))

    def _compare(self, c, m):
        return self.head(torch.cat([c, m, c * m, (c - m).abs()], -1)).squeeze(-1)

    def forward(self, msg, tok, length, msg_embed=None, msg_alive=None):
        """msg: (B, Lmax) 기호 id. msg_embed(지도학습 점검): (B, Lmax, d) 직접 임베딩, msg_alive: (B, Lmax) 불리언."""
        if msg_embed is None:
            me = self.msg_emb(msg)
            malive = msg != self.V + 1
        else:
            me, malive = msg_embed, msg_alive
        me = me * malive[..., None].float()
        out, _ = self.msg_gru(me)
        n = malive.long().sum(1).clamp(min=1)
        m = out[torch.arange(me.shape[0]), n - 1]
        return self._compare(self.bb.encode(tok, length), m)

    def forward_pair(self, tok1, l1, tok2, l2):
        """보조 과제 T2: 두 장면이 모두 보이는 상태에서 개수 비교 (메시지 없음)."""
        return self._compare(self.bb.encode(tok2, l2), self.from_scene(self.bb.encode(tok1, l1)))
