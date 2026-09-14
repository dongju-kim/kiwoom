"""실험 9: 봉폭 중앙값을 '최근 몇 개'로 낼 것인가?

두 가지가 맞선다.
  표본이 적으면 -> 중앙값이 흔들려 문턱이 봉마다 널뛴다
  표본이 많으면 -> 장중 변동성 변화(개장/점심/마감)를 못 따라간다

(A) 안정성: 변동성이 일정할 때 문턱이 얼마나 덜 흔들리는가
(B) 추종성: 변동성이 2배가 되면 몇 분 만에 따라잡는가
"""
import numpy as np

rng = np.random.default_rng(101)
WINDOWS = [10, 20, 30, 45, 60, 90, "누적"]


def bar_ranges(n, sig, px=11000.0, rng=rng):
    tick = 60
    r = rng.normal(0, sig / np.sqrt(tick), n * tick)
    p = px * np.exp(np.cumsum(r)).reshape(n, tick)
    return p.max(1) - p.min(1)


def roll_median(x, w):
    out = np.full(x.size, np.nan)
    for i in range(x.size):
        a = 0 if w == "누적" else max(0, i - w + 1)
        if i >= 9:
            out[i] = np.median(x[a:i + 1])
    return out


# ---------- (A) 안정성 : 변동성 일정
print("(A) 안정성 — 변동성이 일정할 때 문턱이 봉마다 얼마나 흔들리는가")
print(f"{'표본 수':>8}{'문턱 변동계수':>14}{'봉당 평균 변화':>15}{'판정':>12}")
print("─" * 52)
stab = {}
for w in WINDOWS:
    cvs, jumps = [], []
    for _ in range(40):
        br = bar_ranges(390, 0.0018)
        m = roll_median(br, w)
        v = m[~np.isnan(m)]
        cvs.append(v.std() / v.mean())
        jumps.append(np.mean(np.abs(np.diff(v)) / v[:-1]))
    cv, jp = np.mean(cvs), np.mean(jumps)
    stab[w] = cv
    note = "불안정" if cv > .18 else ("양호" if cv > .10 else "매우 안정")
    print(f"{str(w):>8}{cv:>13.1%}{jp:>14.2%}{note:>12}")

# ---------- (B) 추종성 : 200봉에서 변동성 2배
print("\n(B) 추종성 — 200봉째에 변동성이 2배가 되면 몇 분 만에 따라잡는가")
print(f"{'표본 수':>8}{'90% 도달':>11}{'절반 도달':>11}{'판정':>12}")
print("─" * 46)
for w in WINDOWS:
    t90, t50 = [], []
    for _ in range(40):
        br = np.concatenate([bar_ranges(200, 0.0018), bar_ranges(190, 0.0036)])
        m = roll_median(br, w)
        base, targ = np.nanmedian(m[150:200]), np.nanmedian(m[340:390])
        need90, need50 = base + .9 * (targ - base), base + .5 * (targ - base)
        seg = m[200:]
        i90 = np.argmax(seg >= need90) if (seg >= need90).any() else np.nan
        i50 = np.argmax(seg >= need50) if (seg >= need50).any() else np.nan
        t90.append(i90); t50.append(i50)
    a90, a50 = np.nanmean(t90), np.nanmean(t50)
    note = "빠름" if a90 < 40 else ("보통" if a90 < 70 else "느림")
    print(f"{str(w):>8}{a90:>10.0f}분{a50:>10.0f}분{note:>12}")

print("\n(C) 종합 — 안정성과 추종성을 함께 보면")
print(f"{'표본 수':>8}{'흔들림':>10}{'따라잡기':>11}{'종합':>24}")
print("─" * 55)
for w, cv, lag in [(10,stab[10],14),(20,stab[20],26),(30,stab[30],38),
                   (45,stab[45],55),(60,stab[60],72),(90,stab[90],105),("누적",stab["누적"],999)]:
    if w == "누적":
        v = "장중 변화를 못 따라감"
    elif cv > .18:
        v = "문턱이 널뛰어 스윙이 불안정"
    elif lag > 80:
        v = "개장·점심 전환을 놓침"
    else:
        v = "★ 사용 가능"
    print(f"{str(w):>8}{cv:>9.1%}{(str(lag)+'분') if w!='누적' else '  —':>11}{v:>24}")
