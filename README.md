# Defect A Feature Evidence Booster

반도체 공정에서 여러 불량이 섞인 weak label 상황을 가정한 feature boosting 모듈입니다.

이 모듈은 단순히 Good/Bad 분류 성능을 높이는 feature selector가 아닙니다. 목적은 특정 불량 A와 관련된 원인 후보 feature를 order별로 끌어올리고, XAI/통계/안정성 관점의 evidence를 같이 남기는 것입니다.

## Residual-Based Feature Boosting PoC

`feature_boosting/` 패키지는 CatBoost 수율회귀 baseline이 남긴 residual을 sampled candidate feature가 설명할 수 있는지 검증하는 별도 PoC 파이프라인입니다.

VS Code에서 바로 돌릴 때는 아래 노트북을 여세요. YAML 파일은 CLI 반복 실행용이라 필수는 아닙니다.

```text
notebooks/residual_feature_boosting_poc.ipynb
```

처음 clone한 직후에는 실제 데이터가 없어도 됩니다. 노트북 기본값이 `USE_DEMO_DATA = True`라서 실행 중에 toyset을 자동으로 만듭니다.

노트북 기본 toyset 크기는 wafer 14,000매, candidate feature 5,000개입니다. 노트북 상단 설정 셀에서 아래 값만 바꾸면 됩니다.

```python
USE_DEMO_DATA = True
DEMO_N_WAFERS = 2_000
DEMO_N_CANDIDATE_FEATURES = 100
```

처음 빠르게 동작만 확인하고 싶으면 예를 들어 이렇게 줄여서 실행하세요.

```python
DEMO_N_WAFERS = 600
DEMO_N_CANDIDATE_FEATURES = 200
```

Boosting 단계에서 round별 feature 선택 방식도 노트북 설정 셀에서 바꿀 수 있습니다.

```python
# rank 기준으로 round마다 상위 20개 선택
BOOSTING_SELECTION_MODE = "top_k"
SELECT_PER_ROUND = 20
BOOSTING_SELECTION_METRIC = "bad_rmse_reduction"

# after residual / baseline residual 비율이 0.3 이하인 feature 선택
BOOSTING_SELECTION_MODE = "threshold"
BOOSTING_SELECTION_METRIC = "bad_rmse_after_over_baseline"
BOOSTING_SELECTION_THRESHOLD = 0.3
BOOSTING_MAX_SELECT_PER_ROUND = None
```

Overfit guard is enabled by default. A candidate feature can appear in the
ranking chart, but it is not selected if validation residual gets worse than
the base-feature baseline or if validation improvement is much weaker than
train improvement.

```python
AUTO_OVERFIT_SAFE_SETTINGS = True
OVERFIT_GUARD_ENABLED = True
OVERFIT_GUARD_MAX_VALID_AFTER_OVER_BASELINE = 1.0
OVERFIT_GUARD_MAX_VALID_TRAIN_GAP = 0.25
OVERFIT_GUARD_USE_TEST = False  # set True only for exploratory checking
```

정답인자가 있으면 defect별로 넣어두면 candidate rank chart에 별도 마커로 표시됩니다.

```python
ANSWER_FEATURES = {
    "defect_1": [
        "known_root_cause_a",                  # exact feature name
        {"contains_all": ["abc", "step2"]},    # contains both abc and step2
        {"contains_any": ["root", "defect"]},  # contains one of these words
        {"regex": r"abc.*step2"},
    ],
    "defect_2": ["known_root_cause_c"],
}
```

Answer features are highlighted in ranking charts while they are still
candidate features. This is only for visualization; answer features are
selected only when they pass the same top-k or threshold rule as every other
candidate. Once an answer feature is selected, it is removed from later
feature-rank charts like any other selected feature.

```text
data/residual_poc_demo/
  base_dataset.csv
  candidate_features.csv
  base_feature_cols.txt
  groups/
```

터미널에서 toyset만 따로 만들고 싶으면:

```powershell
python scripts/generate_residual_poc_toyset.py
```

크기를 직접 지정할 수도 있습니다.

```powershell
python scripts/generate_residual_poc_toyset.py --rows 14000 --candidate-features 5000
```

이 명령은 아래 경로에 같은 구조의 toyset을 만듭니다.

```text
data/residual_poc_toyset/
```

입력 파일 기본 형태:

```text
data/base_dataset.parquet          # sample_id, yield, split(train/valid/test), base features
data/candidate_features.parquet    # sample_id, sampled candidate features
data/base_feature_cols.txt         # baseline Xb feature list
data/groups/defect_1_bad.csv       # sample_id
data/groups/defect_1_good.csv      # sample_id
```

실제 데이터가 아래 6개 파일 구조라면 노트북에서 `USE_DEMO_DATA = False`, `USE_RAW_SIX_FILE_DATA = True`로 바꾸고 `RAW_*` 설정만 맞추면 됩니다. 파일별 컬럼명이 서로 달라도 설정에서 지정할 수 있습니다.

```text
1) y file:              lot / wf / y
2) candidate file:      lot / wf / candidate feature n개
3) base feature file:   lot / wf / base feature m개
4) defect_1 group file: lot / wf / good_bad
5) defect_2 group file: lot / wf / good_bad
6) defect_3 group file: lot / wf / good_bad
```

노트북은 이 raw 파일들을 먼저 아래 표준 입력으로 변환한 뒤 기존 boosting 코드를 실행합니다.

```text
data/standardized_from_raw/
  base_dataset.csv
  candidate_features.csv
  base_feature_cols.txt
  groups/{defect_id}_bad.csv
  groups/{defect_id}_good.csv
```

VS Code 노트북 대신 CLI로 반복 실행하고 싶을 때:

```powershell
python scripts/run_experiment.py --config configs/experiment.yaml
```

주요 산출물은 `outputs/{run_id}/` 아래에 저장됩니다.

```text
baseline_metrics.csv
baseline_residual_summary.csv
candidate_quality_summary.csv
selected_features.csv
residual_reduction_curve.csv
round_mean_residual_summary.csv
final_model_metrics.csv
final_model_metric_summary.csv
final_feature_set_summary.csv
shap_summary.csv
rankings/{defect_id}_round_{round}.csv
plots/{defect_id}_round_{round}_candidate_loss.png
plots/round_mean_abs_residual_points.png
plots/final_model_metric_comparison.png
plots/final_feature_set_summary.png
models/baseline_model.cbm
models/final_model.cbm
```

가장 중요한 원칙은 residual boosting 단계에서는 `Xb`를 다시 학습하지 않는 것입니다. 각 round에서는 후보 feature `x_j` 하나만으로 현재 residual을 예측하고, feature 선택은 validation bad group의 `bad_rmse_reduction` 기준으로 수행합니다. Test metric은 기록과 검증에만 사용합니다.

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

## Alternative selector comparison

기존 evidence booster와 다른 방식의 selector를 같은 toyset에서 비교할 수 있습니다.

추가된 방식:

```text
nonparametric_random  Mann-Whitney/KS/chi-square/presence test 후 p-value <= 0.05 pool에서 weighted random top-k
distance              Good/Bad 분포 거리 기반 ranking
catboost_shap_gap     CatBoost SHAP의 Bad 평균과 Good 평균 차이가 큰 feature ranking
stability_consensus   bootstrap 반복에서 계속 상위권에 남는 feature ranking
```

이 스크립트는 selector별로 order당 `--top-k`개 feature를 먼저 고른 뒤, 모든 order의 선택 결과를 method별로 합칩니다. 그 다음 모든 order의 Good/Bad rows를 합쳐 method별 global CatBoost를 한 번씩 학습하고, `abs(mean SHAP Bad - mean SHAP Good)` 기준 top feature를 산출합니다. 기본값은 order/method당 15개 선발, global SHAP delta top 15개 리포트입니다.

빠른 테스트:

```powershell
python examples/run_selector_comparison_on_wide_toyset.py --regenerate --orders 1 --rows 600 --booster-good-rows 80 --booster-bad-rows 30 --features-per-order 1000 --top-k 15 --shap-top-n 15 --stability-rounds 2
```

기본 크기 실행:

```powershell
python examples/run_selector_comparison_on_wide_toyset.py --regenerate
```

VS Code/Jupyter에서 표와 plot을 보면서 실행하려면 아래 노트북을 여세요.

```text
notebooks/selector_catboost_shap_review.ipynb
```

결과 파일:

```text
outputs/wide_toy_selector_comparison/combined_selector_comparison.csv
outputs/wide_toy_selector_comparison/method_overlap_jaccard.csv
outputs/wide_toy_selector_comparison/selector_comparison_toy_truth_summary.csv
outputs/wide_toy_selector_comparison/catboost_post_eval/catboost_global_shap_delta_top_features.csv
outputs/wide_toy_selector_comparison/catboost_post_eval/catboost_global_model_metrics_by_method.csv
outputs/wide_toy_selector_comparison/catboost_post_eval/catboost_global_shap_delta_toy_truth_summary.csv
outputs/wide_toy_selector_comparison/catboost_post_eval/catboost_global_method_scoreboard.csv
outputs/wide_toy_selector_comparison/catboost_post_eval/plots/
```

Toyset은 synthetic planted feature를 알고 있으므로, `*_toy_truth_summary.csv`에서 method별 정답 대용 평가를 볼 수 있습니다.

```text
target_defect_a_precision   선택 feature 중 실제 심어둔 A 관련 feature 비율
nonlinear_a_hit_count       nonlinear/interaction A feature를 잡은 개수
sparse_a_hit_count          one-sided sparse A feature를 잡은 개수
other_defect_hit_count      B/C/D 등 다른 불량 feature를 잡은 개수
tool_confounded_hit_count   tool confounding feature를 잡은 개수
noise_hit_count             noise feature를 잡은 개수
toy_truth_score             A hit reward - non-A/noise/confound penalty 요약 점수
```

`catboost_shap_gap`을 실제 SHAP 기준으로 쓰려면 CatBoost가 필요합니다.

```powershell
python -m pip install catboost
```

CatBoost가 설치되어 있지 않으면 post-evaluation은 CSV에 `catboost_not_installed` 경고를 남기고 종료합니다. Plot은 `matplotlib`이 필요합니다.
