"""실험 10: 거래대금을 매수 신호에 섞으면 실제로 도움이 되는가?

세 가지를 나눠서 본다.
 (1) 중복도  - 거래대금 필터가 이미 있는 봉폭 필터와 같은 신호를 거르는가
 (2) 선별력  - 신호를 몇 개나 줄이는가 (기계적 사실, 순환논리 아님)
 (3) 개선도  - 성공률이 오르는가  <- 심어놓은 구조 덕이라 순환논리. 참고용.

(3)은 '거래량이 가격에 선행한다'는 구조를 내가 데이터에 심어놓았기 때문에
당연히 좋게 나온다. 진짜 판단은 실데이터로만 가능하다. 여기서 볼 것은
'하니스가 그 효과를 잡아내는가'와, 구조가 없을 때(null) 헛되이 좋아지지 않는가다.
"""
import numpy as np

MEDN, MULT, TICK = 30, 5.0, 10.0


def gen(seed, n=240, px=11000.0, planted=True):
    """가격 + 거래대금. planted=False 면 거래대금이 가격 구조와 무관(귀무)."""
    rng = np.random.default_rng(seed)
    sig = 0.0016
    d = np.zeros(n); leg = np.zeros(n, int)     # leg: +1 상승다리 -1 눌림
    segs = [(30,70,.50,1),(70,88,-.42,-1),(88,128,.55,1),(128,146,-.40,-1),
            (146,188,.60,1),(188,204,-.45,-1),(204,240,.65,1)]
    for a,b,k,L in segs:
        d[a:b] = k*sig; leg[a:b] = L
    t = 60
    r = rng.normal(np.repeat(d/t, t), sig/np.sqrt(t), n*t)
    p = px*np.exp(np.cumsum(r)).reshape(n, t)
    q = lambda x: np.round(x/TICK)*TICK
    o,h,l,c = q(p[:,0]), q(p.max(1)), q(p.min(1)), q(p[:,-1])

    rngbar = np.maximum(h-l, TICK)
    base = rngbar / np.median(rngbar)            # 거래대금은 봉폭에 비례하는 게 기본
    noise = np.exp(rng.normal(0, 0.55, n))
    mult = np.ones(n)
    if planted:
        mult[leg == 1] = 1.35                    # 상승 다리에 거래 증가
        mult[leg == -1] = 0.60                   # 눌림에 거래 감소
    else:
        rng.shuffle(mult)                        # 구조 없음
    val = 1e8 * base * noise * mult
    return o,h,l,c,val


def analyse(h, l):
    n = len(h); br = h-l
    th = np.full(n, np.nan)
    for i in range(MEDN-1, n): th[i] = np.median(br[i-MEDN+1:i+1])*MULT
    state, ext, extat, pts = 1, h[MEDN-1], MEDN-1, []
    for i in range(MEDN, n):
        if state == 1:
            if h[i] > ext: ext, extat = h[i], i
            if ext-l[i] >= th[i]: pts.append((i,'H',ext,extat)); state,ext,extat = -1,l[i],i
        else:
            if l[i] < ext: ext, extat = l[i], i
            if h[i]-ext >= th[i]: pts.append((i,'L',ext,extat)); state,ext,extat = 1,h[i],i
    return th, pts


def signals(o,h,l,c,val,th,pts):
    out = []
    for k,(ci,kind,price,at) in enumerate(pts):
        if kind != 'H' or k < 1: continue
        pH = [p for p in pts[:k] if p[1]=='H']; pL = [p for p in pts[:k] if p[1]=='L' and p[3]<at]
        if not pH or not pL: continue
        H0, L0 = pH[-1], pL[-1]
        if price <= H0[2]: continue
        lo, loat = l[ci], ci
        nxt = pts[k+1][0] if k+1 < len(pts) else len(c)
        for i in range(ci+1, min(nxt+1, len(c))):
            if l[i] < lo: lo, loat = l[i], i
            if c[i] > h[i-1] and l[i] >= lo and i > loat:
                back = (price-lo)/(price-L0[2])
                if not (0.25 <= back <= 0.55): break
                entry, stop = c[i], lo-TICK
                risk = (entry-stop)/entry
                if not (0.002 <= risk <= 0.008): break
                tgt = np.round(entry*(1+2*risk)/TICK)*TICK
                hit = next((z for z in range(i+1,len(c)) if h[z] >= tgt), None)
                bad = next((z for z in range(i+1,len(c)) if l[z] <= stop), None)
                win = 1 if (hit is not None and (bad is None or hit < bad)) else 0
                out.append(dict(i=i, win=win,
                    # --- 필터 재료 ---
                    rng_ratio = (h[i]-l[i]) / np.median((h-l)[i-29:i+1]),
                    vol_ratio = val[i] / np.mean(val[i-5:i]),        # 돌파봉 거래 실림
                    pull_vol  = np.mean(val[loat-2:i]) / max(np.mean(val[at-6:at+1]),1),  # 눌림/상승 거래비
                ))
                break
    return out


def run(planted, seeds=500):
    S = []
    for s in range(seeds):
        o,h,l,c,val = gen(s, planted=planted)
        th, pts = analyse(h,l)
        S += signals(o,h,l,c,val,th,pts)
    return S


for planted, tag in [(True, "구조 있음 (거래량이 눌림에서 줄고 돌파에서 실림)"),
                     (False, "구조 없음 = 귀무 (거래량이 가격과 무관)")]:
    S = run(planted)
    n = len(S)
    base = np.mean([s['win'] for s in S])
    rr = np.array([s['rng_ratio'] for s in S])
    vr = np.array([s['vol_ratio'] for s in S])
    pv = np.array([s['pull_vol'] for s in S])
    w  = np.array([s['win'] for s in S])
    print(f"\n{'='*74}\n{tag}\n{'='*74}")
    print(f"전체 신호 {n}건,  기본 성공률 {base:.1%}")
    print(f"corr(봉폭비, 거래대금비) = {np.corrcoef(rr,vr)[0,1]:+.2f}   <- 중복도")
    print(f"\n{'필터':<34}{'남는 신호':>10}{'성공률':>9}{'개선':>8}")
    print("─"*62)
    F = {
      "필터 없음":                 np.ones(n, bool),
      "봉폭비 < 4 (급변동 제외)":   rr < 4.0,
      "거래대금비 ≥ 1.5 (돌파 실림)": vr >= 1.5,
      "눌림거래비 ≤ 0.7 (조용한 눌림)": pv <= 0.7,
      "거래대금 + 눌림거래 둘 다":    (vr >= 1.5) & (pv <= 0.7),
      "봉폭 + 거래대금 + 눌림거래":   (rr < 4.0) & (vr >= 1.5) & (pv <= 0.7),
    }
    for name, m in F.items():
        if m.sum() < 10: print(f"{name:<34}{m.sum():>10}{'—':>9}{'표본부족':>8}"); continue
        print(f"{name:<34}{m.sum():>10}{w[m].mean():>8.1%}{(w[m].mean()-base)*100:>+7.1f}%p")
    a, b = (vr >= 1.5), (rr < 4.0)
    inter = (a & b).sum(); union = (a | b).sum()
    print(f"\n거래대금 필터와 봉폭 필터가 남기는 신호의 겹침(자카드) = {inter/union:.0%}")


# ────────────────────────────────────────────────────────────────────────
# 귀무에서도 거래대금 필터가 좋아진 이유: 거래대금이 봉폭에 비례하기 때문.
# "거래대금비 >= 1.5"가 사실은 "큰 봉을 골라라"였다.
# 해법: 봉폭으로 설명되는 부분을 빼고 '초과 거래'만 남긴다.
# ────────────────────────────────────────────────────────────────────────
def excess(val, h, l, i, w=20):
    """봉폭으로 예상되는 거래대금 대비 실제 거래대금의 비율."""
    br = np.maximum(h-l, TICK)
    s = slice(i-w, i)
    exp_per_unit = np.mean(val[s]) / np.mean(br[s])     # 봉폭 1단위당 평소 거래대금
    return val[i] / max(exp_per_unit * br[i], 1)


def run2(planted, seeds=500):
    S = []
    for s in range(seeds):
        o,h,l,c,val = gen(s, planted=planted)
        th, pts = analyse(h,l)
        for sg in signals(o,h,l,c,val,th,pts):
            sg['exc'] = excess(val,h,l,sg['i'])
            S.append(sg)
    return S


print(f"\n\n{'='*74}\n초과거래로 바꾸면 — 봉폭으로 설명되는 부분을 제거\n{'='*74}")
print(f"{'':<12}{'corr(봉폭비,지표)':>18}{'필터 통과':>10}{'성공률':>9}{'기본대비':>10}")
print("─"*62)
for planted, tag in [(False,"귀무"),(True,"구조있음")]:
    S = run2(planted); n=len(S)
    w  = np.array([s['win'] for s in S])
    rr = np.array([s['rng_ratio'] for s in S])
    vr = np.array([s['vol_ratio'] for s in S])
    ex = np.array([s['exc'] for s in S])
    base = w.mean()
    print(f"{tag:<12}{'기본 성공률 '+format(base,'.1%'):>18}{n:>10}{base:>8.1%}{'':>10}")
    for nm, arr, thr in [("  거래대금비 ≥1.5", vr, 1.5), ("  초과거래  ≥1.3", ex, 1.3)]:
        m = arr >= thr
        cr = np.corrcoef(rr, arr)[0,1]
        print(f"{nm:<12}{cr:>+18.2f}{m.sum():>10}{w[m].mean():>8.1%}{(w[m].mean()-base)*100:>+9.1f}%p")
    print()
