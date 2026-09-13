"""실험 7: '중앙값 x 1.5' 문턱이 종목 성격에 따라 다른 의미를 갖는가?

봉폭 분포의 '넓이'가 종목마다 다르면, 같은 배수를 써도 그 문턱이
분포에서 차지하는 위치(분위수)가 달라진다. 그러면 어떤 종목은 문턱이
너무 자주 발동하고, 어떤 종목은 거의 발동하지 않는다.

규칙 A: 문턱 = 봉폭 중앙값 x 1.5
규칙 B: 문턱 = 봉폭 분포의 75분위수   <- 분포 모양과 무관하게 같은 의미
"""
import numpy as np

MIN = 390
TYPES = [
    # (이름, 기본변동성, 변동성의 변동성 = 분포가 얼마나 넓은가)
    ("대형 안정주",   0.0010, 0.25),
    ("보통주",        0.0018, 0.50),
    ("활발한 중소형", 0.0030, 0.80),
    ("테마·급등주",   0.0045, 1.30),
]


def make_day(base_sig, vov, rng, px=50000.0):
    """확률적 변동성: vov 가 클수록 봉폭 분포의 꼬리가 길어진다."""
    z = np.zeros(MIN)
    for i in range(1, MIN):                       # AR(1) 변동성
        z[i] = 0.92 * z[i - 1] + rng.normal(0, 1)
    sig = base_sig * np.exp(vov * z / np.sqrt(1 / (1 - 0.92**2)))
    tick = 60
    r = rng.normal(0, np.repeat(sig / np.sqrt(tick), tick), MIN * tick)
    p = px * np.exp(np.cumsum(r)).reshape(MIN, tick)
    return p.max(1), p.min(1), p[:, -1].copy()


def swing_flips(high, low, thresh):
    """UP/DOWN 스윙 상태기계의 전환 횟수."""
    state, ext, flips = 1, high[0], 0
    for i in range(1, high.size):
        if state == 1:
            ext = max(ext, high[i])
            if ext - low[i] >= thresh:
                flips += 1; state, ext = -1, low[i]
        else:
            ext = min(ext, low[i])
            if high[i] - ext >= thresh:
                flips += 1; state, ext = 1, high[i]
    return flips


rng = np.random.default_rng(31)
print(f"{'종목 유형':<16}{'봉폭중앙값':>10}{'1.5x중앙값':>11}{'그 값의 분위수':>14}"
      f"{'│ A: 전환/일':>13}{'B: 전환/일':>12}")
print("─" * 78)
rows = []
for name, bs, vov in TYPES:
    fa, fb, pct, meds = [], [], [], []
    for _ in range(60):
        h, l, c = make_day(bs, vov, rng)
        rngbar = h - l
        med = np.median(rngbar)
        tha = med * 1.5                      # 규칙 A
        thb = np.quantile(rngbar, 0.75)      # 규칙 B
        pct.append((rngbar < tha).mean())    # 1.5x중앙값이 분포의 몇 % 지점인가
        meds.append(med)
        fa.append(swing_flips(h, l, tha))
        fb.append(swing_flips(h, l, thb))
    rows.append((name, np.mean(meds), np.mean(meds) * 1.5, np.mean(pct),
                 np.mean(fa), np.mean(fb)))
    print(f"{name:<16}{rows[-1][1]:>10.1f}{rows[-1][2]:>11.1f}{rows[-1][3]:>13.1%}"
          f"{rows[-1][4]:>13.1f}{rows[-1][5]:>12.1f}")

pa = [r[4] for r in rows]; pb = [r[5] for r in rows]
print("─" * 78)
print(f"{'전환 횟수 최대/최소 배수':<30}"
      f"규칙 A: {max(pa)/min(pa):>5.2f}배   규칙 B: {max(pb)/min(pb):>5.2f}배")
print(f"{'분위수 위치 편차 (A)':<30}{min(r[3] for r in rows):.0%} ~ {max(r[3] for r in rows):.0%}")
