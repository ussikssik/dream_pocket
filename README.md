# Defect A Feature Evidence Booster

반도체 공정에서 여러 불량이 섞인 weak label 상황을 가정한 feature boosting 모듈입니다.

이 모듈은 단순히 Good/Bad 분류 성능을 높이는 feature selector가 아닙니다. 목적은 특정 불량 A와 관련된 원인 후보 feature를 order별로 끌어올리고, XAI/통계/안정성 관점의 evidence를 같이 남기는 것입니다.

## 핵심 원칙

- Good 또는 Bad 한쪽에서만 존재하는 feature는 제거하지 않습니다.
- CatBoost 성능 개선은 최종 ranking의 필수 조건이 아닙니다.
- CatBoost는 설치되어 있을 때 SHAP consistency를 계산하는 보조 XAI probe로 사용합니다.
- 최종 점수는 contrast, 방향 안정성, bootstrap stability, groupwise stability, asymmetric presence, SHAP consistency, confounding risk, redundancy penalty를 합쳐 계산합니다.

## 기본 사용법

```python
from feature_booster import BoosterConfig, DefectAFeatureEvidenceBooster

def load_order(order_id):
    # DB, parquet, CSV 등에서 order별 데이터를 반환
    # label_col, group_cols, feature columns가 한 DataFrame 안에 있으면 됩니다.
    return df

config = BoosterConfig(
    label_col="target_bad_a",
    positive_label=1,
    group_cols=("lot_id", "tool_id"),
    sample_id_cols=("wafer_id",),
)

booster = DefectAFeatureEvidenceBooster(load_order, config)

result = booster.run(
    order_list=list(range(1, 101)),
    top_k_per_order=10,
)
```

## 결과 컬럼

주요 산출 컬럼은 다음과 같습니다.

```text
order_id
final_rank
feature_name
final_score
evidence_reason
presence_type
bad_coverage
good_coverage
direction
effect_size
fdr_pvalue
bootstrap_stability_score
groupwise_stability_score
xai_consistency_score
shap_abs_mean
confounding_risk_score
redundancy_group
quality_warning
confounding_warning
xai_warning
```

`presence_type`은 `bad_only`, `good_only`, `bad_enriched_sparse`, `good_enriched_sparse`, `both_groups` 중 하나입니다.

## CatBoost 사용

`catboost`가 설치되어 있으면 기본적으로 SHAP probe가 동작합니다. 설치되어 있지 않으면 `xai_warning=catboost_not_installed`가 남고, 통계/evidence 기반 ranking은 그대로 수행됩니다.

CatBoost를 끄고 싶으면:

```python
from feature_booster import BoosterConfig, CatBoostProbeConfig

config = BoosterConfig(
    catboost=CatBoostProbeConfig(enabled=False),
)
```

## 예제 실행

```powershell
python examples/run_feature_booster_example.py
```
