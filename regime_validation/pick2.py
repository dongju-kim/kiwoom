"""가격 조건 + 거래대금(초과거래) 조건을 모두 만족하는 사례를 찾는다."""
import numpy as np
from gen_example import gen, analyse, find_setup, TICK

def volume_for(h, l, n, seed):
    rng = np.random.default_rng(seed + 5000)
    br = np.maximum(h - l, TICK)
    mult = np.ones(n)
    for a, b, k in [(30,70,1.35),(70,88,.60),(88,128,1.35),(128,146,.60),
                    (146,188,1.35),(188,204,.60),(204,240,1.35)]:
        mult[a:b] = k
    return 1e8 * (br/np.median(br)) * np.exp(rng.normal(0,.42,n)) * mult

def excess(val, br, i, w=20):
    sl = slice(i-w, i)
    return val[i] / max(np.mean(val[sl])/np.mean(br[sl]) * br[i], 1)

if __name__ == "__main__":
    best = []
    for seed in range(1200):
        o,h,l,c = gen(seed); n=len(c)
        th,pts = analyse(o,h,l,c)[:2]
        val = volume_for(h,l,n,seed); br = np.maximum(h-l,TICK)
        for s in find_setup(o,h,l,c,th,pts):
            if not (0.28 <= s['back'] <= 0.50): continue
            if not (0.0030 <= s['risk'] <= 0.0060): continue
            i = s['i']
            if i < 25: continue
            ex = excess(val,br,i)
            if ex < 1.4: continue                                  # 돌파봉에 거래가 실렸는가
            pull = np.mean([excess(val,br,k) for k in range(s['H1'][1]+1,i)]) if i>s['H1'][1]+1 else 1
            if pull > 0.95: continue                               # 눌림은 조용했는가
            tgt = np.round(s['entry']*(1+2*s['risk'])/TICK)*TICK
            if abs(tgt - s['H1'][0]) < 30: continue
            hit = next((k for k in range(i+1,n) if h[k]>=tgt), None)
            bad = next((k for k in range(i+1,n) if l[k]<=s['stop']), None)
            if not (hit and (bad is None or hit<bad) and hit-i <= 9): continue
            span = hit - s['H0'][3]
            if span > 62: continue
            s.update(seed=seed,tgt=tgt,hit=hit,span=span,ex=ex,pull=pull,
                     o=o,h=h,l=l,c=c,val=val,br=br)
            best.append(s)

    print(f"가격+거래대금 모두 통과 {len(best)}건")
    b = sorted(best, key=lambda s:(s['span'], -s['ex']))[0]
    lo, i, n = b['H0'][3]-2, b['i'], len(b['c'])
    end = min(n, b['hit']+3)
    print(f"\n■ seed={b['seed']}  {end-lo}봉  문턱={b['th']:.0f}원  봉폭중앙값={b['th']/5:.0f}원")
    print(f"  H0={b['H0'][2]:.0f}  L0={b['L0'][2]:.0f}  H1={b['H1'][0]:.0f}  L1={b['L1']:.0f}")
    print(f"  되돌림={b['back']:.2f}  매수={b['entry']:.0f}  손절={b['stop']:.0f}({-b['risk']:.2%})"
          f"  익절={b['tgt']:.0f}(+{b['tgt']/b['entry']-1:.2%})  {b['hit']-i}분")
    print(f"  ▶ 돌파봉 초과거래={b['ex']:.2f}   눌림 평균 초과거래={b['pull']:.2f}")
    print(f"  손익비 {(b['tgt']-b['entry'])/(b['entry']-b['stop']):.1f}:1")
    print(f"\nconst BARS=[")
    for k in range(lo,end):
        m=9*60+40+(k-lo)
        print(f" {{t:'{m//60:02d}:{m%60:02d}',o:{b['o'][k]:.0f},h:{b['h'][k]:.0f},l:{b['l'][k]:.0f},"
              f"c:{b['c'][k]:.0f},v:{b['val'][k]/1e8:.2f},x:{excess(b['val'],b['br'],k):.2f}}},")
    print("];")
    print(f"H0 발생={b['H0'][3]-lo} 확정={b['H0'][0]-lo} | L0 발생={b['L0'][3]-lo} 확정={b['L0'][0]-lo}")
    print(f"H1 발생={b['H1'][1]-lo} 확정={b['H1'][2]-lo} | L1 발생={b['L1at']-lo}")
    print(f"매수={i-lo}  익절={b['hit']-lo}")
    brs=b['br'][i-29:i+1].astype(int)
    print(f"매수시점 최근30봉 봉폭 중앙값={np.median(brs):.0f}원 → 문턱={np.median(brs)*5:.0f}원")
    print(f"매수봉: 봉폭 {b['br'][i]:.0f}원, 거래대금 {b['val'][i]/1e8:.2f}억, "
          f"예상 {np.mean(b['val'][i-20:i])/np.mean(b['br'][i-20:i])*b['br'][i]/1e8:.2f}억")
