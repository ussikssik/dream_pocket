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
    exclude_cols=("eds_bin_no_wf_mean", "eds_yield_wf_mean"),
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

## Toy semiconductor dataset

실제 실행 테스트용 toy dataset은 아래 명령으로 생성합니다.

```powershell
python examples/generate_toy_semiconductor_dataset.py
```

생성 위치:

```text
data/toy_semiconductor/
  order_001.csv
  order_002.csv
  order_003.csv
  order_004.csv
  order_005.csv
  toy_semiconductor_all_orders.csv
  feature_metadata.csv
  target_summary_by_order.csv
```

포함된 컬럼 예시는 다음과 같습니다.

```text
sensor_*                  공정 sensor 값
measure_*                 계측 measure 값
midproc_defect_*_count    최종 불량이 아닌 중간공정 defect count
equipment_name            설비명
chamber_id                chamber
process_step              공정 step
process_time_sec          공정 시간
eds_*                     최종 EDS 기반 y/검증 지표
target_bad_a              booster가 사용하는 binary label
```

`eds_*`와 `sim_*` 컬럼은 결과/검증용 컬럼이라 booster feature에서 제외합니다.

Toy dataset으로 booster를 실행하려면:

```powershell
python examples/run_feature_booster_on_toyset.py
```

결과 CSV는 아래 폴더에 저장됩니다.

```text
outputs/toy_semiconductor_booster/
```

## Wide toyset, 10,000+ features per order

좀 더 실제 상황에 가깝게 order별 1만 개 이상의 candidate feature를 만들고 싶으면 wide toyset을 사용합니다.

기본값은 다음과 같습니다.

```text
orders: 3
rows per order: 240
generated candidate features per order: 10,000
```

생성:

```powershell
python examples/generate_wide_toy_semiconductor_dataset.py
```

실행:

```powershell
python examples/run_feature_booster_on_wide_toyset.py
```

생성 데이터 위치:

```text
data/toy_semiconductor_wide/
```

결과 위치:

```text
outputs/wide_toy_semiconductor_booster/
```

PC가 느리면 feature 수를 줄여 먼저 테스트할 수 있습니다.

```powershell
python examples/run_feature_booster_on_wide_toyset.py --regenerate --orders 1 2 --rows 120 --features-per-order 1000
```

다시 1만 feature로 돌리려면:

```powershell
python examples/run_feature_booster_on_wide_toyset.py --regenerate --orders 1 2 3 --rows 240 --features-per-order 10000
```

CatBoost SHAP probe까지 켜려면 `catboost`를 설치한 뒤 `--catboost` 옵션을 붙입니다.

```powershell
python -m pip install catboost
python examples/run_feature_booster_on_wide_toyset.py --catboost
```

VS Code에서 노트북으로 보고 싶으면 아래 파일을 엽니다.

```text
notebooks/feature_booster_wide_toyset.ipynb
```

## Visual review dashboard

랭킹 결과가 진짜 좋은지 확인하려면 feature별 plot을 봐야 합니다.

아래 노트북을 열면 top feature별로 scatter, Good/Bad boxplot, 설비별 분포, missingness plot을 확인할 수 있습니다.

```text
notebooks/feature_review_dashboard.ipynb
```

plot에 필요한 패키지:

```powershell
python -m pip install matplotlib
```

기본 설정은 wide toyset 결과를 봅니다.

```text
DATA_DIR = data/toy_semiconductor_wide
RESULT_PATH = outputs/wide_toy_semiconductor_booster/combined_feature_evidence.csv
Y_COL = eds_bin_a_wf_mean
FACET_COL = equipment_name
```

## Realistic wide toyset update

Wide toyset은 이제 order별 전체 wafer pool을 먼저 만든 뒤, 그 안에서 booster용 Good/Bad 샘플만 발췌합니다.

기본값:

```text
full wafer rows per order: 2,500
booster Good rows per order: 180
booster Bad rows per order: 70
candidate features per order: 10,000
```

생성되는 파일:

```text
data/toy_semiconductor_wide/wide_order_001_full_pool.csv  # 전체 wafer pool
data/toy_semiconductor_wide/wide_order_001.csv            # booster 입력용 Good/Bad slice
```

기본 실행:

```powershell
python examples/run_feature_booster_on_wide_toyset.py --regenerate
```

빠른 테스트:

```powershell
python examples/run_feature_booster_on_wide_toyset.py --regenerate --orders 1 --rows 600 --booster-good-rows 80 --booster-bad-rows 30 --features-per-order 1000
```

시각 검증은 아래 노트북에서 합니다.

```text
notebooks/feature_review_dashboard.ipynb
```

Visual review dashboard의 main scatter/boxplot은 full pool 파일을 읽어서 `Ignored`, `Good`, `Bad`를 함께 보여줍니다. 별도의 process sequence plot에서는 특정 기간에 sensor feature가 튀고 그 구간에서 Bad가 많이 나오는지도 확인할 수 있습니다.
