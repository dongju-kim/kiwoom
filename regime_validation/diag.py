import numpy as np
from gen_example import gen, analyse, MEDN, MULT, TICK

fail = {}
rows = []
for seed in range(300):
    o,h,l,c = gen(seed)
    th,pts,state,ext,extat = analyse(o,h,l,c)
    Hs=[p for p in pts if p[1]=='H']; Ls=[p for p in pts if p[1]=='L']
    if len(Hs)<2 or len(Ls)<1 or state!=-1:
        fail['구조부족']=fail.get('구조부족',0)+1; continue
    H1,H0=Hs[-1],Hs[-2]
    L0=[p for p in Ls if p[3]<H1[3]]
    if not L0: fail['L0없음']=fail.get('L0없음',0)+1; continue
    L0=L0[-1]
    if H1[2]<=H0[2]: fail['고점하락']=fail.get('고점하락',0)+1; continue
    if ext<=L0[2]: fail['저점이탈']=fail.get('저점이탈',0)+1; continue
    hit=False
    for i in range(H1[0]+1,len(c)):
        if c[i]>h[i-1] and l[i]>=ext:
            back=(H1[2]-l[i])/(H1[2]-L0[2])
            entry=c[i]; stop=ext-TICK; risk=(entry-stop)/entry
            rows.append((back,risk,th[i],np.median((h-l)[i-29:i+1])))
            hit=True; break
    if not hit: fail['돌파없음']=fail.get('돌파없음',0)+1

print("실패 사유:", fail)
if rows:
    a=np.array(rows)
    print(f"\n돌파까지 도달한 사례 {len(rows)}건")
    for name,col,f in [("되돌림 비율",0,"{:.2f}"),("손절폭(위험)",1,"{:.2%}"),
                       ("문턱(원)",2,"{:.0f}"),("봉폭 중앙값(원)",3,"{:.1f}")]:
        v=a[:,col]
        print(f"  {name:<16} 최소 {f.format(v.min())}  중앙 {f.format(np.median(v))}  최대 {f.format(v.max())}")
    print(f"\n  손절폭 0.6% 이하 비율: {(a[:,1]<=0.006).mean():.0%}")
    print(f"  되돌림 0.22~0.60 비율: {((a[:,0]>=.22)&(a[:,0]<=.60)).mean():.0%}")
