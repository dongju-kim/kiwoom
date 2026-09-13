"""실험 4: 국면 판단 타임프레임 분리 + 최소유지시간(min_dwell) 효과.

'1분봉으로 스캘핑하니 국면도 1분봉으로 보자'가 맞는지를 잰다.
느린 봉에서 국면을 판단하면 봉 수는 줄지만 1봉의 시간은 길어진다 —
실제로 중요한 것은 '몇 봉 늦었나'가 아니라 '몇 분 늦었나'다.
"""
import numpy as np
from core import (make_ohlc, efficiency_ratio, slope_over_atr, classify)
from exp3_detection import markov_regimes, BARS, SIG, TPB

rng = np.random.default_rng(5)


def agg(o, h, l, c, m):
    n = (o.size // m) * m
    O, H = o[:n:m], h[:n].reshape(-1, m).max(1)
    L, C = l[:n].reshape(-1, m).min(1), c[m - 1:n:m]
    return O, H, L, C


def run(days, m, confirm, min_dwell, q=0.90):
    """m분봉에서 국면 판단 → 1분 해상도로 인과적 전방채움."""
    lags, remain, flips, wrong, missed = [], [], [], [], 0
    E, S = [], []
    for o, h, l, c, _ in days:                       # 문턱 보정용 표본
        O, H, L, C = agg(o, h, l, c, m)
        E.append(efficiency_ratio(C, 10)); S.append(np.abs(slope_over_atr(C, H, L)))
    ei = np.nanquantile(np.concatenate(E), q); si = np.nanquantile(np.concatenate(S), q)
    eo = np.nanquantile(np.concatenate(E), q - .20); so = np.nanquantile(np.concatenate(S), q - .20)
    for o, h, l, c, tru in days:
        O, H, L, C = agg(o, h, l, c, m)
        st_m = classify(efficiency_ratio(C, 10), slope_over_atr(C, H, L),
                        ei, eo, si, so, confirm, min_dwell)
        st = np.zeros(BARS, int)                     # j번째 m분봉 상태는 종가 이후에만 사용 가능
        for j, v in enumerate(st_m):
            a = (j + 1) * m
            st[a:a + m] = v
        flips.append(int((np.diff(st) != 0).sum()))
        b = np.flatnonzero(np.diff(np.concatenate([[tru[0] - 1], tru])) != 0)
        for a, z in zip(b, list(b[1:]) + [BARS]):
            d = tru[a]
            if d == 0 or z - a < 20 or a < 90:
                continue
            hit = np.flatnonzero(st[a:z] == d)
            if hit.size == 0:
                missed += 1; remain.append(0.0)
            else:
                lags.append(int(hit[0])); remain.append(1 - hit[0] / (z - a))
            wrong.append(float(np.mean(st[a:z] == -d)))
    return (np.median(lags) if lags else np.nan, np.mean(remain), missed,
            np.mean(wrong), np.mean(flips))


days = []
for _ in range(80):
    tru = markov_regimes(BARS, rng)
    o, h, l, c = make_ohlc(BARS, TPB, SIG, tru * 0.2 * SIG, rng)
    days.append((o, h, l, c, tru))

print("국면 판단 봉 / 확인·유지 규칙별 성능  (진실 국면 평균 60분 지속, 추세강도 0.2σ)")
print(f"{'판단 타임프레임':<18}{'규칙':<22}{'탐지지연(분)':>12}{'남은구간':>9}{'놓침':>6}{'역방향':>8}{'전환/일':>8}")
for m, mlabel in [(1, "1분봉"), (3, "3분봉"), (5, "5분봉")]:
    for conf, dwell, rl in [(1, 1, "확인없음"), (2, 1, "2봉확인(제안안)"),
                            (1, 5, "최소유지 5봉"), (2, 5, "2봉확인+최소유지5봉")]:
        lag, rem, mis, wr, fl = run(days, m, conf, dwell)
        print(f"{mlabel:<18}{rl:<22}{lag:>11.0f}분{rem:>8.0%}{mis:>6d}{wr:>8.1%}{fl:>8.1f}")
