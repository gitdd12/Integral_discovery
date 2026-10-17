"""기호 단위 토크나이저. 모든 기호는 토큰 1개."""
from __future__ import annotations

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIAL = [PAD, BOS, EOS]

DIGITS = list("0123456789")
SLOTS = [f"p{k}" for k in range(32)]  # `정의`로 만드는 사용자 이름 (고정 슬롯)

# 모델이 직접 쓸 수 있는 기호
MODEL_TOKENS = (
    DIGITS + ["."]
    + ["+", "-", "×", "÷", "(", ")", "=", "<", ">"]
    + ["f(", "rect(", ","]
    + ["a", "b", "x", "i", "N"] + SLOTS
    + ["반복", "정의", "제출[", "]", ":", ";", "\n"]
)

# 읽기 전용: 문제·계산기·피드백이 쓰는 기호. 모델이 쓰면 위반.
# (오류, 가로, 세로, 넓이?, 높이?, 부터, 까지, 더하기?, 계산?, 제출?) 는 계획서 어휘표에
# 없던 것을 문제 문장·오류 알림용으로 보충한 것.
READONLY_TOKENS = [
    "문제:", "곡선:", "구간:", "도형:", "직사각형", "붙이기", "오차", "%",
    "참", "거짓", "sin", "cos", "exp", "^", "→",
    "오류", "가로", "세로", "넓이?", "높이?", "부터", "까지", "더하기?", "계산?", "제출?",
]

TOKENS = SPECIAL + MODEL_TOKENS + READONLY_TOKENS
assert len(set(TOKENS)) == len(TOKENS)

TOK2ID = {t: i for i, t in enumerate(TOKENS)}
VOCAB_SIZE = len(TOKENS)
PAD_ID, BOS_ID, EOS_ID = TOK2ID[PAD], TOK2ID[BOS], TOK2ID[EOS]

READONLY = set(READONLY_TOKENS) | {PAD, BOS}
MODEL_ALLOWED = set(MODEL_TOKENS) | {EOS}

_BY_LEN = sorted((t for t in TOKENS if t not in SPECIAL and t != "\n"), key=len, reverse=True)


def tokenize(text: str) -> list[str]:
    """글자 → 토큰 문자열 목록. 공백은 무시, 가장 긴 토큰부터 맞춘다."""
    out, i = [], 0
    while i < len(text):
        c = text[i]
        if c in " \t\r":
            i += 1
            continue
        if c == "\n":
            out.append("\n")
            i += 1
            continue
        for t in _BY_LEN:
            if text.startswith(t, i):
                out.append(t)
                i += len(t)
                break
        else:
            raise ValueError(f"모르는 글자: {text[i:i+8]!r}")
    return out


def encode(text: str, bos: bool = False) -> list[int]:
    ids = [TOK2ID[t] for t in tokenize(text)]
    return ([BOS_ID] if bos else []) + ids


def ids_of(toks: list[str]) -> list[int]:
    return [TOK2ID[t] for t in toks]


def model_allowed_mask() -> list[bool]:
    """모델이 뽑아도 되는 토큰 True (샘플러에서 나머지 로짓을 막는 용도)."""
    return [t in MODEL_ALLOWED for t in TOKENS]


def _is_word(t: str) -> bool:
    return t in ("sin", "cos", "exp") or ("가" <= t[0] <= "힣")


def decode(ids: list[int], mask: list[int] | None = None) -> str:
    """사람이 읽는 글자로. mask를 주면 계산기가 채운 칸을 ‹ › 로 감싼다."""
    parts: list[str] = []
    prev = None
    in_forced = False
    plen = _prompt_len(mask) if mask is not None else 0
    for k, i in enumerate(ids):
        t = TOKENS[i]
        if t in SPECIAL:
            continue
        forced = mask is not None and mask[k] == 0 and k >= plen
        if mask is not None and forced and not in_forced:
            parts.append("‹")
            in_forced = True
        elif mask is not None and in_forced and (not forced or t == "\n"):
            parts.append("›")
            in_forced = False
        if prev is not None and t != "\n" and prev != "\n":
            func_call = prev in ("sin", "cos", "exp") and t == "("
            need = not func_call and (
                _is_word(t) or _is_word(prev) or prev in (":", ";")
                or (prev[0] == "p" and prev[-1].isdigit() and t.isdigit())
                or (prev[-1].isalnum() and t[0].isalpha() and prev != "(")
            )
            if need and not (parts and parts[-1] in ("‹", "›")):
                parts.append(" ")
        parts.append(t)
        prev = t
    if in_forced:
        parts.append("›")
    return "".join(parts)


def _prompt_len(mask: list[int]) -> int:
    n = 0
    for m in mask:
        if m == 0:
            n += 1
        else:
            break
    return n
