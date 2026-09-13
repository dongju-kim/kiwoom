import numpy as np
from gen_example import gen, analyse, find_setup, MULT, TICK

cands = []
for seed in range(600):
    o, h, l, c = gen(seed)
    th, pts, *_ = analyse(o, h, l, c)
    for s in find_setup(o, h, l, c, th, pts):
        if not (0.28 <= s['back'] <= 0.50): continue
        if not (0.0030 <= s['risk'] <= 0.0060): continue
        tgt = np.round(s['entry'] * (1 + 2*s['risk']) / TICK) * TICK
        if abs(tgt - s['H1'][0]) < 30: continue
        hit = next((k for k in range(s['i']+1, len(c)) if h[k] >= tgt), None)
        bad = next((k for k in range(s['i']+1, len(c)) if l[k] <= s['stop']), None)
        if not (hit and (bad is None or hit < bad) and hit - s['i'] <= 10): continue
        s.update(seed=seed, tgt=tgt, hit=hit, span=hit - s['H0'][3], o=o,h=h,l=l,c=c)
        cands.append(s)

sp = np.array([s['span'] for s in cands])
print(f"사례 {len(cands)}건.  H0부터 익절까지 걸린 시간(분)")
print(f"  최소 {sp.min()}   25% {np.percentile(sp,25):.0f}   중앙 {np.median(sp):.0f}"
      f"   75% {np.percentile(sp,75):.0f}   최대 {sp.max()}")
print("\n→ 제대로 된 문턱을 쓰면 H0→L0→H1→L1→매수 구조 하나가 "
      f"{np.median(sp):.0f}분 안팎 걸린다. 14봉으로는 절대 안 보인다.")

b = sorted(cands, key=lambda s: (s['span'], abs(s['back']-.38)))[0]
lo, i = b['H0'][3]-2, b['i']
end = min(len(b['c']), b['hit']+3)
print(f"\n■ 가장 압축된 사례: seed={b['seed']}  총 {end-lo}봉  문턱={b['th']:.0f}원"
      f"  봉폭중앙값={b['th']/MULT:.0f}원")
print(f"  H0={b['H0'][2]:.0f}  L0={b['L0'][2]:.0f}  H1={b['H1'][0]:.0f}  L1={b['L1']:.0f}")
print(f"  되돌림={b['back']:.2f}  매수={b['entry']:.0f}  손절={b['stop']:.0f}({-b['risk']:.2%})"
      f"  익절={b['tgt']:.0f}(+{b['tgt']/b['entry']-1:.2%})  {b['hit']-i}분 도달"
      f"  손익비 {(b['tgt']-b['entry'])/(b['entry']-b['stop']):.1f}:1")
print(f"\nconst BARS=[")
for k in range(lo, end):
    m = 9*60+40+(k-lo)
    print(f" {{t:'{m//60:02d}:{m%60:02d}',o:{b['o'][k]:.0f},h:{b['h'][k]:.0f},"
          f"l:{b['l'][k]:.0f},c:{b['c'][k]:.0f}}},")
print("];")
print(f"H0 발생={b['H0'][3]-lo} 확정={b['H0'][0]-lo} | L0 발생={b['L0'][3]-lo} 확정={b['L0'][0]-lo}")
print(f"H1 발생={b['H1'][1]-lo} 확정={b['H1'][2]-lo} | L1 발생={b['L1at']-lo}")
print(f"매수={i-lo}  익절={b['hit']-lo}")
br=(b['h']-b['l'])[i-29:i+1].astype(int)
print(f"매수시점 최근30봉 봉폭 정렬: {sorted(br)}")
print(f"  중앙값 = {np.median(br):.0f}원  ->  문턱 = {np.median(br)*5:.0f}원")
