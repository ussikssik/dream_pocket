# Toy Semiconductor Dataset

이 폴더는 `DefectAFeatureEvidenceBooster`를 바로 실행해보기 위한 toy dataset입니다.

## Files

```text
order_001.csv ... order_005.csv
toy_semiconductor_all_orders.csv
feature_metadata.csv
target_summary_by_order.csv
```

## Row Unit

각 row는 wafer-level 샘플입니다.

## Feature 후보

```text
sensor_*                  sensor summary 값
measure_*                 계측 measure 값
midproc_defect_*_count    중간공정 defect count
equipment_name            설비명
chamber_id                chamber
process_step              공정 step
route_id                  route
process_time_sec          공정 시간
```

## Label / y 계열

```text
target_bad_a              booster label, 특정 불량 A가 발현된 것으로 보는 binary 값
eds_bin_no_wf_mean        wafer-level EDS bin_no 평균
eds_bin_a_wf_mean         wafer-level EDS defect A bin 평균
eds_bin_b_wf_mean         wafer-level EDS defect B bin 평균
eds_bin_c_wf_mean         wafer-level EDS defect C bin 평균
eds_yield_wf_mean         wafer-level EDS yield 평균
```

`eds_*` 컬럼은 최종 y/검증 지표이므로 runner에서 `exclude_cols`로 제외합니다.

## Simulation-only columns

```text
sim_true_defect_a_flag
sim_true_defect_b_flag
sim_true_defect_c_flag
sim_true_defect_d_flag
sim_dominant_defect
```

이 컬럼들은 toy dataset 검증을 위한 숨은 정답 성격입니다. 실제 booster feature로 쓰면 leakage가 되므로 runner에서 제외합니다.

## Run

```powershell
python examples/run_feature_booster_on_toyset.py
```
