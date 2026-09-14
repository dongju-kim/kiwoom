# kiwoom

키움 API 기반 국내주식 단기매매 연구 저장소.

## 디렉터리

| 경로 | 내용 |
|---|---|
| `regime_validation/` | 국면 판단·눌림목 로직의 통계 검증 하니스와 측정 결과 |

## regime_validation

1·3분봉 스캘핑용 국면 판단 로직을 몬테카를로로 검증한다. 각 실험은 독립 실행 가능하며,
측정 결과는 `regime_validation/README.md` 에 정리되어 있다.

```
pip install numpy
python regime_validation/exp1_er_null.py
```
