"""아티팩트용 예시 생성: 실제 검출기를 돌려 진짜 눌림목 매수 사례를 찾는다.

정한 규칙(봉폭 표본 30개 / 배수 5배)을 그대로 적용해 스윙·눌림·돌파가
실제로 발생하는 구간을 골라낸다. 손으로 지어낸 숫자를 쓰지 않기 위함.
"""
import numpy as np

MEDN, MULT, TICK = 30, 5.0, 10.0


def gen(seed, n=200, px=11000.0):
    """상승 다리와 눌림이 번갈아 나오는 하루 일부를 만든다."""
    rng = np.random.default_rng(seed)
    sig = 0.0016
    d = np.zeros(n)
    segs = [(30, 62, .50), (62, 78, -.42), (78, 115, .55), (115, 132, -.40),
            (132, 168, .60), (168, 182, -.45), (182, 200, .65)]
    for a, b, k in segs:
        d[a:b] = k * sig
    t = 60
    r = rng.normal(np.repeat(d / t, t), sig / np.sqrt(t), n * t)
    p = px * np.exp(np.cumsum(r)).reshape(n, t)
    q = lambda x: np.round(x / TICK) * TICK
    return q(p[:, 0]), q(p.max(1)), q(p.min(1)), q(p[:, -1])


def analyse(o, h, l, c):
    n = len(c)
    br = h - l
    th = np.full(n, np.nan)
    for i in range(MEDN - 1, n):
        th[i] = np.median(br[i - MEDN + 1:i + 1]) * MULT
    state, ext, extat, pts = 1, h[MEDN - 1], MEDN - 1, []
    for i in range(MEDN, n):
        if state == 1:
            if h[i] > ext: ext, extat = h[i], i
            if ext - l[i] >= th[i]:
                pts.append((i, 'H', ext, extat)); state, ext, extat = -1, l[i], i
        else:
            if l[i] < ext: ext, extat = l[i], i
            if h[i] - ext >= th[i]:
                pts.append((i, 'L', ext, extat)); state, ext, extat = 1, h[i], i
    return th, pts, state, ext, extat


def find_setup(o, h, l, c, th, pts):
    """H 확정 이후, 저점을 지키면서 직전봉 고가를 넘는 첫 봉을 찾는다."""
    out = []
    for k, (ci, kind, price, at) in enumerate(pts):
        if kind != 'H' or k < 1: continue
        H1 = (price, at, ci)
        prevH = [p for p in pts[:k] if p[1] == 'H']
        prevL = [p for p in pts[:k] if p[1] == 'L' and p[3] < at]
        if not prevH or not prevL: continue
        H0, L0 = prevH[-1], prevL[-1]
        if H1[0] <= H0[2]: continue
        lo, loat = l[ci], ci                      # 확정 이후 진행 중인 저점
        nxt = pts[k + 1][0] if k + 1 < len(pts) else len(c)
        for i in range(ci + 1, min(nxt + 1, len(c))):
            if l[i] < lo: lo, loat = l[i], i
            if c[i] > h[i - 1] and l[i] >= lo and i > loat:
                back = (H1[0] - lo) / (H1[0] - L0[2])
                entry, stop = c[i], lo - TICK
                out.append(dict(H0=H0, H1=H1, L0=L0, L1=lo, L1at=loat, i=i,
                                back=back, entry=entry, stop=stop,
                                risk=(entry - stop) / entry, th=th[i]))
                break
    return out


if __name__ == "__main__":
    cands = []
    for seed in range(300):
        o, h, l, c = gen(seed)
        th, pts, *_ = analyse(o, h, l, c)
        for s in find_setup(o, h, l, c, th, pts):
            if not (0.25 <= s['back'] <= 0.55): continue
            tgt = np.round(s['entry'] * (1 + 2 * s['risk']) / TICK) * TICK   # 손익비 2:1
            hit = next((k for k in range(s['i'] + 1, len(c)) if h[k] >= tgt), None)
            bad = next((k for k in range(s['i'] + 1, len(c)) if l[k] <= s['stop']), None)
            if hit and (bad is None or hit < bad) and hit - s['i'] <= 12:
                s.update(seed=seed, tgt=tgt, hit=hit, o=o, h=h, l=l, c=c)
                cands.append(s)
    print(f"조건 충족 사례 {len(cands)}건")
    if cands:
        r = np.array([[s['back'], s['risk'], s['th'], s['hit'] - s['i']] for s in cands])
        for nm, j, f in [("되돌림", 0, "{:.2f}"), ("손절폭", 1, "{:.2%}"),
                         ("문턱(원)", 2, "{:.0f}"), ("익절까지(분)", 3, "{:.0f}")]:
            print(f"  {nm:<12} 중앙 {f.format(np.median(r[:,j]))}"
                  f"   범위 {f.format(r[:,j].min())} ~ {f.format(r[:,j].max())}")
        b = sorted(cands, key=lambda s: abs(s['back'] - .38) + abs(s['risk'] - .0045) * 40)[0]
        i, lo = b['i'], max(0, b['i'] - 32)
        print(f"\n■ 선택: seed={b['seed']}  문턱={b['th']:.0f}원"
              f"  봉폭중앙값={b['th']/MULT:.1f}원")
        print(f"  H0={b['H0'][2]:.0f}  H1={b['H1'][0]:.0f}  L0={b['L0'][2]:.0f}  L1={b['L1']:.0f}")
        print(f"  되돌림={b['back']:.2f}  매수={b['entry']:.0f}  손절={b['stop']:.0f}"
              f"({-b['risk']:.2%})  익절={b['tgt']:.0f}(+{b['tgt']/b['entry']-1:.2%})"
              f"  {b['hit']-i}분 만에 도달")
        print(f"\n// 배열 0번 = 원본 {lo}번")
        for k in range(lo, min(len(b['c']), b['hit'] + 3)):
            m = 9 * 60 + 30 + (k - lo)
            print(f" {{t:'{m//60:02d}:{m%60:02d}',o:{b['o'][k]:.0f},h:{b['h'][k]:.0f},"
                  f"l:{b['l'][k]:.0f},c:{b['c'][k]:.0f}}},")
        print(f"\n  H0 발생={b['H0'][3]-lo} 확정={b['H0'][0]-lo}")
        print(f"  L0 발생={b['L0'][3]-lo} 확정={b['L0'][0]-lo}")
        print(f"  H1 발생={b['H1'][1]-lo} 확정={b['H1'][2]-lo}")
        print(f"  L1 발생={b['L1at']-lo}   매수={i-lo}   익절={b['hit']-lo}")
