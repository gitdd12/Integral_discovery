"""예시 메모장 출력. (이 풀이들은 사람이 쓴 시험용이며 모델 학습에는 절대 쓰지 않는다.)"""
import numpy as np

from integral.curves import random_curve
from integral.notebook import new_episode

rng = np.random.default_rng(3)
curve = random_curve(rng)

ATTEMPT = (
    "정의 p0=0\n"
    "반복 N=4: 정의 p0=p0+rect((b-a)÷N,f(a+i×(b-a)÷N))\n"
    "p0=\n"
    "4<5=\n"
    "제출[ 정의 p0=0; 반복 N=4: 정의 p0=p0+rect((b-a)÷N,f(a+i×(b-a)÷N)); p0 ]\n"
    "정의 p0=0\n"
    "반복 N=64: 정의 p0=p0+rect((b-a)÷N,f(a+i×(b-a)÷N))\n"
    "p0=\n"
    "제출[ 정의 p0=0; 반복 N=64: 정의 p0=p0+rect((b-a)÷N,f(a+i×(b-a)÷N)); p0 ]\n"
)

nb = new_episode(curve, seed=3)
print(nb.feed_text(ATTEMPT))
print("\n정답 넓이:", curve.area)
print("제출 횟수:", len(nb.submissions), " 점수:", round(nb.final_score(), 3))
print("학습 대상(mask=1) 토큰:", sum(nb.mask), "/", len(nb.mask))
