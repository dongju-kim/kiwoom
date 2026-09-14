"""실험 3: 정답을 아는 합성 국면 데이터에서의 탐지 성능.

마르코프 국면전환(상승/횡보/하락)으로 가격을 만들고, 각 변형이
(1) 국면을 얼마나 늦게 잡는지 (2) 반대로 부르는 비율 (3) 하루 몇 번 뒤집히는지를 잰다.

공정 비교를 위해 모든 변형의 문턱은 '랜덤워크에서 비횡보 상태로 보내는 시간 = 15%'
가 되도록 동일하게 보정한다. 그래야 '더 민감해서 빨리 잡는 것'과 실력을 구분할 수 있다.
"""
import numpy as np
from core import (make_ohlc, efficiency_ratio, slope_over_atr,
                  variance_ratio_z, classify, state_stats)

SIG, TPB, BARS = 0.002, 60, 390
rng = np.random.default_rng(42)


def markov_regimes(n, rng, mean_dwell=60):
    st, out, i = rng.choice([-1, 0, 1]), np.empty(n, int), 0
    while i < n:
        L = max(8, int(rng.exponential(mean_dwell)))
        out[i:i + L] = st
        i += L
        st = rng.choice([s for s in (-1, 0, 1) if s != st])
    return out[:n]


def build(n_days, k, rng, with_truth=True):
    """k = 추세 국면의 봉당 드리프트(시그마 배수). k=0 이면 순수 랜덤워크."""
    days = []
    for _ in range(n_days):
        tru = markov_regimes(BARS, rng) if with_truth else np.zeros(BARS, int)
        o, h, l, c = make_ohlc(BARS, TPB, SIG, tru * k * SIG, rng)
        days.append((o, h, l, c, tru))
    return days


def indicators(o, h, l, c, kind):
    sn = slope_over_atr(c, h, l, sigma=3.0, span=3, atr_n=14)
    if kind == "er10":   return efficiency_ratio(c, 10), sn
    if kind == "er20":   return efficiency_ratio(c, 20), sn
    if kind == "none":   return np.where(np.isnan(sn), np.nan, 1.0), sn   # ER 미사용
    if kind == "vr":     # 분산비 z 를 0..1 로 눌러 ER 자리에 대입
        z = variance_ratio_z(c, q=5, n=60)
        return 1 / (1 + np.exp(-z)), sn
    raise ValueError(kind)


def calibrate(null_days, kind, confirm, target=0.15):
    """랜덤워크에서 비횡보 시간이 target 이 되는 분위수 q 를 이분탐색."""
    cache = [indicators(*d[:4], kind) for d in null_days]
    ers = np.concatenate([e[~np.isnan(e)] for e, _ in cache])
    sns = np.concatenate([np.abs(s[~np.isnan(s)]) for _, s in cache])
    flat_er = kind == "none"        # ER 미사용 변형은 ER 조건을 무력화
    lo, hi = 0.50, 0.9995
    for _ in range(22):
        q = (lo + hi) / 2
        si = np.quantile(sns, q)
        so = np.quantile(sns, max(0.05, q - .20))
        ei = eo = -np.inf if flat_er else None
        if not flat_er:
            ei = np.quantile(ers, q)
            eo = np.quantile(ers, max(0.05, q - .20))
        frac = np.mean([np.mean(classify(e, s, ei, eo, si, so, confirm) != 0) for e, s in cache])
        lo, hi = (q, hi) if frac > target else (lo, q)
    return ei, eo, si, so


def evaluate(days, kind, th, confirm):
    ei, eo, si, so = th
    lags, remain, missed, flips, wrong, acc = [], [], 0, [], [], []
    for o, h, l, c, tru in days:
        e, s = indicators(o, h, l, c, kind)
        st = classify(e, s, ei, eo, si, so, confirm)
        st_v = state_stats(st, tru)
        flips.append(st_v["flips"]); acc.append(st_v["accuracy"])
        b = np.flatnonzero(np.diff(np.concatenate([[tru[0] - 1], tru])) != 0)
        for a, z in zip(b, list(b[1:]) + [BARS]):
            d = tru[a]
            if d == 0 or z - a < 20 or a < 80:
                continue
            hit = np.flatnonzero(st[a:z] == d)
            if hit.size == 0:
                missed += 1; remain.append(0.0)
            else:
                lags.append(int(hit[0])); remain.append(1 - hit[0] / (z - a))
            wrong.append(float(np.mean(st[a:z] == -d)))
    return dict(lag=np.median(lags) if lags else np.nan, remain=np.mean(remain),
                missed=missed, flips=np.mean(flips), wrong=np.mean(wrong), acc=np.mean(acc))


if __name__ == "__main__":
    null = build(60, 0.0, rng, with_truth=False)
    for k in (0.10, 0.20, 0.40):
        real = build(60, k, np.random.default_rng(1000 + int(k * 100)))
        print(f"\n=== 추세 강도 k={k} (봉당 드리프트 {k}σ, 60봉이면 {k*60*0.2:.1f}% 이동) ===")
        print(f"{'구성':<34}{'탐지지연':>8}{'남은구간':>9}{'놓침':>6}{'역방향':>8}{'정확도':>8}{'전환/일':>8}")
        for label, kind, conf in [
            ("ER10+기울기, 확인없음",        "er10", 1),
            ("ER10+기울기, 2봉확인(제안안)", "er10", 2),
            ("ER10+기울기, 3봉확인",        "er10", 3),
            ("ER20+기울기, 2봉확인",        "er20", 2),
            ("기울기만 (ER 제거), 2봉확인",  "none", 2),
            ("분산비VR+기울기, 2봉확인",     "vr",   2),
        ]:
            th = calibrate(null, kind, conf)
            r = evaluate(real, kind, th, conf)
            print(f"{label:<34}{r['lag']:>7.0f}봉{r['remain']:>8.0%}{r['missed']:>6d}"
                  f"{r['wrong']:>8.1%}{r['acc']:>8.1%}{r['flips']:>8.1f}")
