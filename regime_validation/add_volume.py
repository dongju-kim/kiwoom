"""예시 57봉에 거래대금을 붙이고 매수 시점의 초과거래를 계산한다."""
import numpy as np
from gen_example import gen, analyse, find_setup, TICK

o,h,l,c = gen(173)
lo = 20 - 2 + 0          # pick.py 가 쓴 lo 를 재현: H0 발생 -2
th, pts = analyse(o,h,l,c)[:2]
ss = [s for s in find_setup(o,h,l,c,th,pts) if 0.28<=s['back']<=0.50]
s = [x for x in ss if 0.0030<=x['risk']<=0.0060][0]
lo = s['H0'][3]-2
idx = range(lo, s['i']+8)

rng = np.random.default_rng(9)
br = np.maximum(h-l, TICK)
# 거래대금 = 봉폭 비례 x 잡음 x (상승다리 증가 / 눌림 감소)
segs=[(30,70,1.35),(70,88,.60),(88,128,1.35),(128,146,.60),
      (146,188,1.35),(188,204,.60),(204,240,1.35)]
mult=np.ones(len(c))
for a,b,k in segs: mult[a:b]=k
val = 1.0e8*(br/np.median(br))*np.exp(rng.normal(0,.42,len(c)))*mult

def excess(i,w=20):
    sl=slice(i-w,i)
    per=np.mean(val[sl])/np.mean(br[sl])
    return val[i]/max(per*br[i],1)

i=s['i']
print(f"매수봉 인덱스 {i}  (배열 {i-lo}번)")
print(f"  봉폭 {br[i]:.0f}원   거래대금 {val[i]/1e8:.2f}억")
print(f"  최근 20봉 '봉폭 1원당 평소 거래대금' = {np.mean(val[i-20:i])/np.mean(br[i-20:i])/1e6:.2f}백만/원")
print(f"  예상 거래대금 = {np.mean(val[i-20:i])/np.mean(br[i-20:i])*br[i]/1e8:.2f}억")
print(f"  ▶ 초과거래 = {excess(i):.2f}")
print(f"  (원시 거래대금비 = {val[i]/np.mean(val[i-5:i]):.2f})")
print(f"\n눌림 구간({s['L1at']-lo}~{i-lo}번) 평균 초과거래 = "
      f"{np.mean([excess(k) for k in range(s['H1'][1],i)]):.2f}")
print(f"직전 상승구간 평균 초과거래 = "
      f"{np.mean([excess(k) for k in range(s['H1'][1]-12,s['H1'][1])]):.2f}")
print("\nVOL = [  // 억원, 배열 순서")
row=[]
for k in idx:
    row.append(f"{val[k]/1e8:.2f}")
print("  "+", ".join(row)+"\n];")
print(f"\n초과거래 배열:")
row=[f"{excess(k):.2f}" for k in idx]
print("  "+", ".join(row))
