import numpy as np
from gen_example import gen, analyse, find_setup, TICK
from pick2 import volume_for, excess

stage = dict.fromkeys(['전체','되돌림','손절폭','초과거래','눌림조용','승리','기간'],0)
exs, pulls, spans = [], [], []
for seed in range(800):
    o,h,l,c = gen(seed); n=len(c)
    th,pts = analyse(o,h,l,c)[:2]
    val = volume_for(h,l,n,seed); br = np.maximum(h-l,TICK)
    for s in find_setup(o,h,l,c,th,pts):
        stage['전체']+=1
        if not (0.28<=s['back']<=0.50): continue
        stage['되돌림']+=1
        if not (0.0030<=s['risk']<=0.0060): continue
        stage['손절폭']+=1
        i=s['i']
        if i<25: continue
        ex=excess(val,br,i); exs.append(ex)
        if ex<1.4: continue
        stage['초과거래']+=1
        pull=np.mean([excess(val,br,k) for k in range(s['H1'][1]+1,i)]) if i>s['H1'][1]+1 else 1
        pulls.append(pull)
        if pull>0.95: continue
        stage['눌림조용']+=1
        tgt=np.round(s['entry']*(1+2*s['risk'])/TICK)*TICK
        hit=next((k for k in range(i+1,n) if h[k]>=tgt),None)
        bad=next((k for k in range(i+1,n) if l[k]<=s['stop']),None)
        if not (hit and (bad is None or hit<bad) and hit-i<=9): continue
        stage['승리']+=1
        sp=hit-s['H0'][3]; spans.append(sp)
        if sp<=62: stage['기간']+=1

print("단계별 통과 수")
for k,v in stage.items(): print(f"  {k:<10}{v:>6}")
print(f"\n초과거래 분포: 중앙 {np.median(exs):.2f}  75% {np.percentile(exs,75):.2f}"
      f"  90% {np.percentile(exs,90):.2f}  최대 {max(exs):.2f}")
print(f"눌림 평균 초과거래 분포: 중앙 {np.median(pulls):.2f}  25% {np.percentile(pulls,25):.2f}")
if spans: print(f"기간 분포: 최소 {min(spans)}  중앙 {np.median(spans):.0f}  25% {np.percentile(spans,25):.0f}")
