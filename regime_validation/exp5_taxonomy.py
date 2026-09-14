"""실험 5: 평면 6상태 분류 vs 축 분해(factored) 분류.

'하락/횡보/상승/눌림/급등/급락'을 한 줄의 6개 라벨로 두는 것과,
방향축 x 강도축으로 분해해 조합으로 파생시키는 것을 비교한다.

진실값도 두 축으로 만든다: 방향(3) x 강도(평상/급). 따라서 참 라벨은 6개이고
두 방식 모두 같은 6개를 맞히는 과제를 푼다 — 표현 방식만 다르다.
"""
import numpy as np
from core import make_ohlc, slope_over_atr, atr, classify, causal_gauss

BARS, TPB, SIG, K = 390, 60, 0.002, 0.25
rng = np.random.default_rng(17)


def markov(n, vals, dwell, rng, p=None):
    st, out, i = rng.choice(vals, p=p), np.empty(n, int), 0
    while i < n:
        L = max(6, int(rng.exponential(dwell)))
        out[i:i + L] = st
        i += L
        st = rng.choice([v for v in vals if v != st])
    return out[:n]


def make_day(rng):
    d = markov(BARS, [-1, 0, 1], 60, rng)           # 방향축
    b = markov(BARS, [0, 1], 22, rng)               # 강도축: 0 평상, 1 급(burst)
    b[rng.random(BARS) < 0.55] = 0                  # 급은 드문 사건으로
    sig = SIG * (1 + 2.0 * b)                       # 급 = 변동성 3배
    drift = d * K * sig * (1 + 1.5 * b)             # 급 = 드리프트도 강함
    n = BARS * TPB
    r = rng.normal(np.repeat(drift / TPB, TPB), np.repeat(sig / np.sqrt(TPB), TPB), n)
    p = 100 * np.exp(np.cumsum(r))
    g = p.reshape(BARS, TPB)
    return g[:, 0].copy(), g.max(1), g.min(1), g[:, -1].copy(), d, b


def rolling_median(x, n):
    out = np.full(x.size, np.nan)
    for i in range(n, x.size):
        out[i] = np.nanmedian(x[i - n:i])
    return out


def features(o, h, l, c):
    sn = slope_over_atr(c, h, l, sigma=3.0, span=3, atr_n=14)   # 방향 특징
    a = atr(h, l, c, 14)
    vr = a / np.maximum(rolling_median(a, 60), 1e-9)            # 강도 특징
    return sn, vr


def run_factored(sn, vr, si, so, vi, vo, confirm):
    """축 분해: 방향 상태기계 + 강도 상태기계를 독립으로 돌리고 조합."""
    ones = np.where(np.isnan(sn), np.nan, 1.0)
    d = classify(ones, sn, -np.inf, -np.inf, si, so, confirm)   # 방향: 히스테리시스 적용
    b = np.zeros(vr.size, int)                                  # 강도: 2상태 히스테리시스
    cur, cnt, pend = 0, 0, 0
    for i, v in enumerate(vr):
        if np.isnan(v):
            b[i] = cur; continue
        raw = 1 if (v > vi if cur == 0 else v > vo) else 0
        if raw == cur: cnt, pend = 0, cur
        else:
            cnt = cnt + 1 if raw == pend else 1; pend = raw
            if cnt >= confirm: cur, cnt = raw, 0
        b[i] = cur
    return d, b


def run_flat6(sn, vr, si, vi, confirm):
    """평면 6상태: 범주형이라 축별 히스테리시스를 정의할 수 없고 N봉 확인만 가능.
    (6상태에 진짜 히스테리시스를 주려면 30개 경계쌍을 일일이 지정해야 한다.)"""
    n = sn.size
    raw = np.zeros(n, int)
    for i in range(n):
        if np.isnan(sn[i]) or np.isnan(vr[i]):
            raw[i] = 0; continue
        d = 1 if sn[i] > si else (-1 if sn[i] < -si else 0)
        raw[i] = (d + 1) * 2 + (1 if vr[i] > vi else 0)   # 0..5
    out, cur, cnt, pend = np.zeros(n, int), 2, 0, 2
    for i in range(n):
        if raw[i] == cur: cnt, pend = 0, cur
        else:
            cnt = cnt + 1 if raw[i] == pend else 1; pend = raw[i]
            if cnt >= confirm: cur, cnt = raw[i], 0
        out[i] = cur
    return out


days = [make_day(rng) for _ in range(70)]
F = [features(*d[:4]) for d in days]
SN = np.concatenate([np.abs(s[~np.isnan(s)]) for s, _ in F])
VR = np.concatenate([v[~np.isnan(v)] for _, v in F])
si, so = np.quantile(SN, .80), np.quantile(SN, .60)
vi, vo = np.quantile(VR, .80), np.quantile(VR, .60)

for confirm in (1, 2):
    accD = accB = accJ = flipsJ = flipsD = 0.0
    accD6 = accB6 = accJ6 = flipsJ6 = flipsD6 = 0.0
    for (o, h, l, c, td, tb), (sn, vr) in zip(days, F):
        m = ~(np.isnan(sn) | np.isnan(vr))
        d, b = run_factored(sn, vr, si, so, vi, vo, confirm)
        j = (d + 1) * 2 + b
        tj = (td + 1) * 2 + tb
        accD += (d[m] == td[m]).mean(); accB += (b[m] == tb[m]).mean()
        accJ += (j[m] == tj[m]).mean()
        flipsJ += (np.diff(j) != 0).sum(); flipsD += (np.diff(d) != 0).sum()

        f6 = run_flat6(sn, vr, si, vi, confirm)
        d6, b6 = f6 // 2 - 1, f6 % 2
        accD6 += (d6[m] == td[m]).mean(); accB6 += (b6[m] == tb[m]).mean()
        accJ6 += (f6[m] == tj[m]).mean()
        flipsJ6 += (np.diff(f6) != 0).sum(); flipsD6 += (np.diff(d6) != 0).sum()
    N = len(days)
    print(f"\n=== {confirm}봉 확인 ===")
    print(f"{'방식':<26}{'방향정확도':>10}{'강도정확도':>10}{'6상태정확도':>11}"
          f"{'전체전환/일':>11}{'방향축전환/일':>13}")
    print(f"{'축 분해(방향x강도)':<26}{accD/N:>10.1%}{accB/N:>10.1%}{accJ/N:>11.1%}"
          f"{flipsJ/N:>11.1f}{flipsD/N:>13.1f}")
    print(f"{'평면 6상태':<26}{accD6/N:>10.1%}{accB6/N:>10.1%}{accJ6/N:>11.1%}"
          f"{flipsJ6/N:>11.1f}{flipsD6/N:>13.1f}")

tj_all = np.concatenate([((d[4] + 1) * 2 + d[5]) for d in days])
print("\n참 6상태의 실제 출현 빈도 (1분봉 하루 390봉 기준):")
names = {0: "급락", 1: "급락(고변동)", 2: "횡보", 3: "횡보(고변동)", 4: "상승", 5: "급등"}
lbl = ["하락", "하락+급", "횡보", "횡보+급", "상승", "상승+급"]
for k in range(6):
    f = (tj_all == k).mean()
    print(f"  {lbl[k]:<10} {f:>6.1%}   하루 평균 {f*390:>5.1f}봉")


# ---------------- 상태 개수를 늘리면 개별 상태의 신뢰도가 어떻게 되는가
print("\n\n=== 상태 개수별 신뢰도 (기저비율 대비 향상분이 진짜 정보량) ===")
allD = np.concatenate([d[4] for d in days]); allB = np.concatenate([d[5] for d in days])
allJ = (allD + 1) * 2 + allB
pd_, pb_, pj_, mask = [], [], [], []
for (o, h, l, c, td, tb), (sn, vr) in zip(days, F):
    d, b = run_factored(sn, vr, si, so, vi, vo, 2)
    pd_.append(d); pb_.append(b); pj_.append((d + 1) * 2 + b)
    mask.append(~(np.isnan(sn) | np.isnan(vr)))
pd_, pb_, pj_ = map(np.concatenate, (pd_, pb_, pj_)); mask = np.concatenate(mask)

def lift(pred, true, k):
    a = (pred[mask] == true[mask]).mean()
    base = max((true[mask] == v).mean() for v in np.unique(true[mask]))
    print(f"  {k:<22} 정확도 {a:>6.1%}   최빈클래스 기저 {base:>6.1%}   "
          f"향상 {(a-base)*100:>+6.1f}%p   {'쓸만함' if a-base > .08 else '사실상 무정보'}")

lift(pd_, allD, "3상태 (방향만)")
lift(pb_, allB, "2상태 (강도만)")
lift(pj_, allJ, "6상태 (방향x강도)")

print("\n=== 6상태 클래스별 성능 — 소수 클래스(급등/급락)가 핵심 ===")
print(f"{'상태':<12}{'출현율':>8}{'재현율':>8}{'정밀도':>8}{'해석':>24}")
for k in range(6):
    t, p = allJ[mask] == k, pj_[mask] == k
    rec = (t & p).sum() / max(t.sum(), 1)
    pre = (t & p).sum() / max(p.sum(), 1)
    note = "진입 근거로 쓸 수 없음" if pre < 0.45 else "조건부 사용 가능"
    print(f"{lbl[k]:<12}{t.mean():>8.1%}{rec:>8.1%}{pre:>8.1%}{note:>24}")

print("\n=== '급'을 잘못 부르는 방향 — 비용 비대칭 확인 ===")
burst_true = allB[mask] == 1
print(f"  진짜 급변동인데 평상으로 부름(놓침)   : {((burst_true) & (pb_[mask]==0)).sum()/max(burst_true.sum(),1):>6.1%}")
print(f"  평상인데 급변동으로 부름(헛발질)      : {((~burst_true) & (pb_[mask]==1)).sum()/max((~burst_true).sum(),1):>6.1%}")
dir_ok = pd_[mask] == allD[mask]
dir_rev = (pd_[mask] * allD[mask]) < 0
print(f"  방향 정반대로 부름(치명적)            : {dir_rev.mean():>6.1%}")


# ---------------- 강도축을 빠른 추정량으로 바꾸면?
print("\n=== 강도축 추정량 비교 (느린 ATR vs 빠른 봉폭) ===")
def fast_intensity(o, h, l, c, win=3):
    rng_ = (h - l) / c
    k = np.convolve(rng_, np.ones(win) / win, mode="full")[:rng_.size]  # 인과 이동평균
    med = rolling_median(rng_, 60)
    return k / np.maximum(med, 1e-9)

for name, fn in [("ATR(14)/중앙값(60)  [현재]", lambda d: features(*d[:4])[1]),
                 ("3봉 봉폭/중앙값(60) [빠름]", lambda d: fast_intensity(*d[:4], 3)),
                 ("1봉 봉폭/중앙값(60) [최속]", lambda d: fast_intensity(*d[:4], 1))]:
    vv = [fn(d) for d in days]
    allv = np.concatenate([v[~np.isnan(v)] for v in vv])
    q_i, q_o = np.quantile(allv, .80), np.quantile(allv, .60)
    pb2, tb2, mk = [], [], []
    for d, v in zip(days, vv):
        sn = features(*d[:4])[0]
        _, b = run_factored(sn, v, si, so, q_i, q_o, 2)
        pb2.append(b); tb2.append(d[5]); mk.append(~(np.isnan(sn) | np.isnan(v)))
    pb2, tb2, mk = map(np.concatenate, (pb2, tb2, mk))
    a = (pb2[mk] == tb2[mk]).mean(); base = max((tb2[mk] == v).mean() for v in (0, 1))
    t = tb2[mk] == 1; pr = pb2[mk] == 1
    rec = (t & pr).sum() / max(t.sum(), 1); pre = (t & pr).sum() / max(pr.sum(), 1)
    print(f"  {name:<26} 정확도 {a:>6.1%} (기저 {base:>5.1%}, 향상 {(a-base)*100:>+5.1f}%p)"
          f"  급변동 재현율 {rec:>5.1%} 정밀도 {pre:>5.1%}")
