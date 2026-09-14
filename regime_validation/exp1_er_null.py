"""실험 1: ER(n) 의 귀무분포 — 랜덤워크(순수 잡음)에서 ER 은 얼마가 나오는가?

'ER > 0.3 이면 추세' 같은 문턱이 의미가 있으려면, 랜덤워크에서 그 값이
거의 안 나와야 한다. 안 그러면 그 문턱은 잡음을 추세라고 부르는 장치다.
"""
import numpy as np
from core import efficiency_ratio

rng = np.random.default_rng(7)
N = 2_000_000
print(f"{'n':>4} {'이론 1/sqrt(n)':>13} {'평균ER':>8} {'표준편차':>8} {'중앙값':>8} "
      f"{'P(ER>.3)':>9} {'P(ER>.4)':>9} {'P(ER>.5)':>9} {'P(ER>.6)':>9} {'상위5% 값':>9}")
for n in (10, 14, 20, 30, 60):
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, N)))
    er = efficiency_ratio(close, n)
    er = er[~np.isnan(er)]
    print(f"{n:>4} {1/np.sqrt(n):>13.3f} {er.mean():>8.3f} {er.std():>8.3f} "
          f"{np.median(er):>8.3f} {(er>.3).mean():>9.1%} {(er>.4).mean():>9.1%} "
          f"{(er>.5).mean():>9.1%} {(er>.6).mean():>9.1%} {np.quantile(er,.95):>9.3f}")
