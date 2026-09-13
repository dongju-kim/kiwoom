"""실험 6: 방향을 '평활선 기울기'로 볼 때 vs '고점·저점 갱신(돌파)'로 볼 때.

평활(이동평균)은 과거를 섞으므로 정의상 늦다. 돌파는 '지금 막 일어난 일'이라
지연이 적을 수 있다. 대신 잡음에 더 자주 속을 수 있다 — 그 교환비를 잰다.
"""
import numpy as np
from core import make_ohlc, slope_over_atr
from exp3_detection import markov_regimes, BARS, SIG, TPB

rng = np.random.default_rng(23)
K = 0.2

days = []
for _ in range(80):
    tru = markov_regimes(BARS, rng)
    o, h, l, c = make_ohlc(BARS, TPB, SIG, tru * K * SIG, rng)
    days.append((o, h, l, c, tru))


def breakout_state(h, l, c, N, confirm=1):
    """N봉 고점 돌파 -> 상승, N봉 저점 이탈 -> 하락. 반대 돌파 전까지 유지."""
    st = np.zeros(c.size, int)
    cur, pend, cnt = 0, 0, 0
    for i in range(N, c.size):
        hi, lo = h[i - N:i].max(), l[i - N:i].min()
        raw = 1 if c[i] > hi else (-1 if c[i] < lo else cur)
        if raw == cur: cnt, pend = 0, cur
        else:
            cnt = cnt + 1 if raw == pend else 1; pend = raw
            if cnt >= confirm: cur, cnt = raw, 0
        st[i] = cur
    return st


def slope_state(o, h, l, c, q_thresh, confirm=1):
    sn = slope_over_atr(c, h, l, sigma=3.0, span=3, atr_n=14)
    st = np.zeros(c.size, int)
    cur, pend, cnt = 0, 0, 0
    for i in range(c.size):
        if np.isnan(sn[i]): st[i] = cur; continue
        raw = 1 if sn[i] > q_thresh else (-1 if sn[i] < -q_thresh else cur)
        if raw == cur: cnt, pend = 0, cur
        else:
            cnt = cnt + 1 if raw == pend else 1; pend = raw
            if cnt >= confirm: cur, cnt = raw, 0
        st[i] = cur
    return st


def score(states):
    lags, remain, flips, wrong, missed = [], [], [], [], 0
    for (o, h, l, c, tru), st in zip(days, states):
        flips.append(int((np.diff(st) != 0).sum()))
        b = np.flatnonzero(np.diff(np.concatenate([[tru[0] - 1], tru])) != 0)
        for a, z in zip(b, list(b[1:]) + [BARS]):
            d = tru[a]
            if d == 0 or z - a < 20 or a < 80: continue
            hit = np.flatnonzero(st[a:z] == d)
            if hit.size == 0: missed += 1; remain.append(0.0)
            else: lags.append(int(hit[0])); remain.append(1 - hit[0] / (z - a))
            wrong.append(float(np.mean(st[a:z] == -d)))
    return (np.median(lags), np.mean(remain), missed, np.mean(wrong), np.mean(flips))


sn_all = np.concatenate([np.abs(slope_over_atr(d[3], d[1], d[2])) for d in days])
sn_all = sn_all[~np.isnan(sn_all)]

print("방향 판단 방식 비교 (진실 국면 평균 60봉, 추세강도 0.2σ, 1분봉)")
print(f"{'방식':<30}{'탐지지연':>9}{'남은구간':>9}{'놓침':>6}{'역방향':>8}{'전환/일':>8}")
rows = []
for N in (5, 10, 20):
    for conf in (1, 2):
        s = [breakout_state(d[1], d[2], d[3], N, conf) for d in days]
        rows.append((f"{N}봉 고저 돌파, {conf}봉확인", score(s)))
for q in (.55, .70, .85):
    th = np.quantile(sn_all, q)
    for conf in (1, 2):
        s = [slope_state(*d[:4], th, conf) for d in days]
        rows.append((f"가우시안기울기/ATR {int(q*100)}분위, {conf}봉확인", score(s)))
for name, (lag, rem, mis, wr, fl) in rows:
    print(f"{name:<30}{lag:>8.0f}봉{rem:>8.0%}{mis:>6d}{wr:>8.1%}{fl:>8.1f}")
