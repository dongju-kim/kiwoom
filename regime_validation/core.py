"""ER(10) x Gaussian-slope/ATR x hysteresis 국면판단 로직의 참조 구현.

검증용으로만 쓰는 최소 구현. 모든 지표는 인과적(causal)이어야 하며,
centered_gauss 는 '이렇게 하면 안 된다'는 것을 보여주기 위한 대조군이다.
"""
import numpy as np


# ---------------------------------------------------------------- 가격 생성
def make_ohlc(n_bars, ticks_per_bar, sigma_bar, drift_bar, rng):
    """틱 단위 랜덤워크를 봉으로 집계해 OHLC 를 만든다.

    drift_bar 는 봉당 드리프트(스칼라 또는 길이 n_bars 배열).
    """
    n_tick = n_bars * ticks_per_bar
    s_t = sigma_bar / np.sqrt(ticks_per_bar)
    mu = np.broadcast_to(np.asarray(drift_bar, float), (n_bars,))
    mu_t = np.repeat(mu / ticks_per_bar, ticks_per_bar)
    r = rng.normal(mu_t, s_t, n_tick)
    p = 100.0 * np.exp(np.cumsum(r))
    g = p.reshape(n_bars, ticks_per_bar)
    return g[:, 0].copy(), g.max(1), g.min(1), g[:, -1].copy()


# ---------------------------------------------------------------- 지표
def rolling_sum(x, n):
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(x.size, np.nan)
    out[n - 1:] = c[n:] - c[:-n]
    return out


def efficiency_ratio(close, n=10):
    """Kaufman ER: |순변화| / 경로길이. 0=완전 횡보, 1=완전 추세."""
    d = np.diff(close, prepend=close[0])
    path = rolling_sum(np.abs(d), n)
    net = np.full(close.size, np.nan)
    net[n:] = np.abs(close[n:] - close[:-n])
    with np.errstate(invalid="ignore", divide="ignore"):
        er = np.where(path > 0, net / path, 0.0)
    er[:n] = np.nan
    return er


def causal_gauss(x, sigma, trunc=3.0):
    """단측(과거만 사용) 가우시안 가중이동평균. 실거래에서 유일하게 허용되는 형태."""
    L = max(2, int(trunc * sigma) + 1)
    k = np.exp(-0.5 * (np.arange(L) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.full(L - 1, x[0]), x])
    return np.convolve(pad, k, mode="valid")


def centered_gauss(x, sigma, trunc=3.0):
    """대칭(중심) 가우시안 = 미래를 본다. 대조군 전용, 실거래 금지."""
    L = max(2, int(trunc * sigma) + 1)
    o = np.arange(-L + 1, L)
    k = np.exp(-0.5 * (o / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.full(L - 1, x[0]), x, np.full(L - 1, x[-1])])
    return np.convolve(pad, k, mode="valid")


def atr(high, low, close, n=14):
    pc = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    a = np.full(close.size, np.nan)
    a[n - 1] = tr[:n].mean()
    for i in range(n, close.size):  # Wilder 평활
        a[i] = (a[i - 1] * (n - 1) + tr[i]) / n
    return a


def slope_over_atr(close, high, low, sigma=3.0, span=3, atr_n=14, centered=False):
    """가우시안 평활선의 span봉 기울기를 ATR로 정규화 → 'ATR 몇 개/봉' 단위."""
    g = centered_gauss(close, sigma) if centered else causal_gauss(close, sigma)
    sl = np.full(close.size, np.nan)
    sl[span:] = (g[span:] - g[:-span]) / span
    a = atr(high, low, close, atr_n)
    floor = 1e-8 + 0.02 * np.nanmedian(a)  # 0-range 봉 방어
    return sl / np.maximum(a, floor)


def variance_ratio_z(close, q=5, n=60):
    """Lo-MacKinlay 분산비 z통계량. >0 추세(모멘텀), <0 평균회귀, 0 랜덤워크.

    ER 과 같은 질문을 던지지만 귀무분포가 알려져 있어 임계값을 통계로 정할 수 있다.
    """
    lr = np.diff(np.log(close), prepend=0.0)
    out = np.full(close.size, np.nan)
    v1 = rolling_sum(lr**2, n) / n
    lq = np.full(close.size, np.nan)
    lq[q:] = np.log(close[q:]) - np.log(close[:-q])
    vq = rolling_sum(np.nan_to_num(lq**2), n) / (n * q)
    with np.errstate(invalid="ignore", divide="ignore"):
        vr = vq / v1
    se = np.sqrt(2.0 * (2 * q - 1) * (q - 1) / (3 * q * n))
    out = (vr - 1.0) / se
    out[: n + q] = np.nan
    return out


# ---------------------------------------------------------------- 상태기계
def classify(er, sn, er_in, er_out, sn_in, sn_out, confirm=1, min_dwell=1):
    """히스테리시스 + N봉 확인 국면 상태기계.

    반환: -1 하락 / 0 횡보 / +1 상승
    er_in>er_out, sn_in>sn_out 이 히스테리시스 밴드.
    """
    n = er.size
    st = np.zeros(n, dtype=int)
    cur, pend, cnt, dwell = 0, 0, 0, 0
    for i in range(n):
        e, s = er[i], sn[i]
        if np.isnan(e) or np.isnan(s):
            st[i] = cur
            continue
        if cur == 0:  # 진입은 엄격한 문턱
            raw = 1 if (e > er_in and s > sn_in) else (-1 if (e > er_in and s < -sn_in) else 0)
        else:  # 유지는 느슨한 문턱 (히스테리시스)
            keep = (e > er_out) and (s > sn_out if cur > 0 else s < -sn_out)
            if keep:
                raw = cur
            else:
                raw = 1 if (e > er_in and s > sn_in) else (-1 if (e > er_in and s < -sn_in) else 0)
        dwell += 1
        if raw == cur:
            pend, cnt = cur, 0
        else:
            if raw == pend:
                cnt += 1
            else:
                pend, cnt = raw, 1
            if cnt >= confirm and dwell >= min_dwell:
                cur, cnt, dwell = raw, 0, 0
        st[i] = cur
    return st


def state_stats(st, truth=None):
    flips = int((np.diff(st) != 0).sum())
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[st[0] - 1], st])) != 0).tolist() + [st.size])
    d = {
        "flips": flips,
        "mean_dwell": float(runs.mean()) if runs.size else float("nan"),
        "pct_up": float((st == 1).mean()),
        "pct_dn": float((st == -1).mean()),
        "pct_flat": float((st == 0).mean()),
    }
    if truth is not None:
        m = ~np.isnan(truth.astype(float))
        d["accuracy"] = float((st[m] == truth[m]).mean())
        d["wrong_sign"] = float(((st[m] * truth[m]) < 0).mean())  # 상승인데 하락이라 부름
    return d
