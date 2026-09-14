"""실험 8: 배수를 얼마로 해야 '거래 가능한' 스윙이 나오는가?

실험 7에서 1.5배 문턱은 하루 130~180회 전환 = 2~3분마다 스윙이 뒤집힌다.
눌림목 매매는 '한 다리(leg)'가 충분히 커야 성립한다. 익절 +0.8%를 노리려면
스윙 한 다리가 최소 1% 이상은 돼야 한다. 배수별로 그게 되는지 잰다.
"""
import numpy as np
from exp7_threshold import make_day, TYPES, MIN

rng = np.random.default_rng(77)


def swing_legs(high, low, close, thresh):
    """확정된 스윙 다리들의 크기(%)를 반환."""
    state, ext, legs, anchor = 1, high[0], [], high[0]
    for i in range(1, high.size):
        if state == 1:
            ext = max(ext, high[i])
            if ext - low[i] >= thresh:
                legs.append(abs(ext - anchor) / close[i] * 100)
                state, anchor, ext = -1, ext, low[i]
        else:
            ext = min(ext, low[i])
            if high[i] - ext >= thresh:
                legs.append(abs(ext - anchor) / close[i] * 100)
                state, anchor, ext = 1, ext, high[i]
    return legs


MULTS = [1.5, 2.5, 4.0, 6.0, 9.0]
print("배수별 스윙 품질  (다리 = 고점→저점 또는 저점→고점 한 구간)\n")
print(f"{'배수':<6}{'종목 유형':<16}{'전환/일':>9}{'평균 다리':>10}{'다리≥1%':>9}{'판정':>16}")
print("─" * 68)
for m in MULTS:
    summ = []
    for name, bs, vov in TYPES:
        fl, allleg = [], []
        for _ in range(40):
            h, l, c = make_day(bs, vov, rng)
            th = np.median(h - l) * m
            lg = swing_legs(h, l, c, th)
            fl.append(len(lg)); allleg += lg
        allleg = np.array(allleg)
        big = (allleg >= 1.0).mean() if allleg.size else 0
        summ.append((name, np.mean(fl), np.mean(allleg), big))
    for name, f, ml, big in summ:
        verdict = "거래 가능" if (8 <= f <= 45 and big > 0.5) else (
                  "너무 잦음" if f > 45 else ("너무 뜸함" if f < 8 else "다리가 작음"))
        print(f"{m:<6}{name:<16}{f:>9.1f}{ml:>9.2f}%{big:>8.0%}{verdict:>16}")
    print()
