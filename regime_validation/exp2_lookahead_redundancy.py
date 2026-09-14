"""실험 2: (a) 중심 가우시안의 미래참조, (b) ER 과 기울기/ATR 의 중복도."""
import numpy as np
from core import make_ohlc, efficiency_ratio, slope_over_atr, causal_gauss, centered_gauss

rng = np.random.default_rng(11)
PATHS, BARS = 300, 390          # 390 = 09:00~15:30 1분봉 하루치

ca_ic, ce_ic, er_sn, er_absn = [], [], [], []
for _ in range(PATHS):
    o, h, l, c = make_ohlc(BARS, 60, 0.002, 0.0, rng)   # 드리프트 0 = 예측불가
    fwd = np.concatenate([np.diff(np.log(c)), [np.nan]])  # t -> t+1 미래수익
    ca = slope_over_atr(c, h, l, centered=False)
    ce = slope_over_atr(c, h, l, centered=True)
    er = efficiency_ratio(c, 10)
    m = ~(np.isnan(ca) | np.isnan(ce) | np.isnan(er) | np.isnan(fwd))
    ca_ic.append(np.corrcoef(ca[m], fwd[m])[0, 1])
    ce_ic.append(np.corrcoef(ce[m], fwd[m])[0, 1])
    er_sn.append(np.corrcoef(er[m], np.abs(ca[m]))[0, 1])
    er_absn.append(np.corrcoef(er[m], ca[m])[0, 1])

print("(a) 미래참조 검사 — 드리프트 0인 랜덤워크, 예측력은 0이어야 정상")
print(f"  인과 가우시안 기울기 vs 다음봉 수익 상관 : {np.mean(ca_ic):+.4f}  (정상: ~0)")
print(f"  중심 가우시안 기울기 vs 다음봉 수익 상관 : {np.mean(ce_ic):+.4f}  <-- 미래참조")
print(f"  랜덤워크에서 이 값이 0이 아니면 그만큼이 전부 백테스트 착시다.\n")

print("(b) 중복도 — ER(10) 과 |가우시안기울기/ATR| 은 서로 다른 정보인가?")
print(f"  corr( ER(10), |기울기/ATR| ) = {np.mean(er_sn):+.3f}   (1에 가까울수록 같은 말 반복)")
print(f"  corr( ER(10),  기울기/ATR  ) = {np.mean(er_absn):+.3f}   (ER은 방향 정보 없음 → ~0 기대)")

# 지연(lag) 측정: 중심 필터는 몇 봉의 미래를 보는가
x = np.cumsum(rng.normal(0, 1, 20000))
for sig in (2.0, 3.0, 5.0):
    ce, ca = centered_gauss(x, sig), causal_gauss(x, sig)
    best = max(range(-12, 13), key=lambda k: np.corrcoef(ce[200:-200], np.roll(x, k)[200:-200])[0, 1])
    bestc = max(range(-12, 13), key=lambda k: np.corrcoef(ca[200:-200], np.roll(x, k)[200:-200])[0, 1])
    print(f"  sigma={sig}: 중심필터 정렬 {best:+d}봉(음수=미래를 봄), 인과필터 지연 {bestc:+d}봉")
