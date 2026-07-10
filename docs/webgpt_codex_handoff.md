# Web GPT 프로젝트 handoff: 코덱스

이 문서는 사내에서 Codex 접속이 안 될 때, 웹 ChatGPT/GPT 프로젝트에서 현재 Codex 대화 맥락을 이어가기 위한 시작 문서입니다.

## 1. 웹 GPT 프로젝트 만들기

프로젝트 이름은 예를 들어 아래처럼 만드세요.

```text
코덱스
```

프로젝트에 업로드하면 좋은 파일:

```text
docs/gpt_context_feature_boosting_rev0.txt
docs/webgpt_codex_handoff.md
notebooks/residual_feature_boosting_poc.ipynb
feature_boosting/reporting.py
feature_boosting/residual_boosting.py
feature_boosting/data_loader.py
feature_boosting/modeling.py
feature_boosting/baseline_model.py
feature_boosting/final_model.py
feature_boosting/validation.py
feature_boosting/answer_features.py
```

가장 간단하게는 `docs/gpt_context_feature_boosting_rev0.txt` 하나만 올려도 됩니다. 이 파일 안에는 주요 코드가 한 번에 붙어 있습니다.

## 2. 프로젝트 지침에 붙여넣을 내용

웹 GPT 프로젝트의 instructions 또는 첫 메시지에 아래 내용을 붙여넣으세요.

```text
너는 Codex처럼 이 feature_boosting_rev0 프로젝트를 도와주는 코딩/분석 파트너다.

나는 Python, VS Code, Git에 익숙하지 않으므로 설명은 아주 구체적으로 해줘.
하지만 너무 장황하게 말하지 말고, 내가 바로 따라 할 수 있게 파일명, 셀 번호, 코드 위치를 명확히 말해줘.

현재 프로젝트의 메인 파일은 notebooks/residual_feature_boosting_poc.ipynb 이다.
이 노트북은 CatBoost baseline 모델 이후 남은 residual을 candidate feature가 설명하는지 확인하는 Residual-Based Feature Boosting PoC다.

중요한 전제:
- 실제 입력 데이터는 CSV 6개 구조를 지원한다.
- demo data도 지원한다.
- base feature만으로 baseline 모델을 학습한다.
- candidate feature를 residual model에 넣어 residual 감소량으로 rank를 매긴다.
- defect별 bad/good group을 따로 평가한다.
- feature rank curve, round residual curve, final model 비교, block 방식, null benchmark가 들어 있다.

내가 원하는 답변 스타일:
- 한국어로 설명한다.
- 내가 초보자라는 전제로, 명령어를 어디서 실행하는지까지 말한다.
- 먼저 개념을 설명하고, 내가 동의하면 코드 수정안을 제안한다.
- 내가 코드 수정을 원하면 어떤 파일의 어떤 함수/셀을 바꿔야 하는지 정확히 말한다.
- 사내에서는 Codex와 git pull이 안 될 수 있으므로, 복붙으로 바꿀 수 있는 형태도 알려준다.
```

## 3. 첫 질문으로 붙여넣을 내용

새 GPT 대화를 시작할 때 아래를 첫 메시지로 넣으면 좋습니다.

```text
첨부한 docs/gpt_context_feature_boosting_rev0.txt 와 docs/webgpt_codex_handoff.md 를 먼저 읽고, 이 프로젝트 맥락을 기억해줘.

나는 residual_feature_boosting_poc.ipynb를 사내 VS Code에서 실행하려고 한다.
앞으로 질문하면 현재 구현 기준으로 답해줘.
특히 아래 기능들이 이미 들어간 상태라고 가정해줘.

1. CSV 6개 입력 구조 지원
2. lot_id/wf_id 분리 또는 lot_wf_id 결합 컬럼 분리 지원
3. residual feature boosting
4. feature rank curve
5. round별 residual/MAE curve
6. final model metric 비교 chart
7. block preselection experiment
8. null/noise feature competition benchmark
9. round MAE plot은 round마다 defect별 1개 point만 나오도록 수정됨

내가 질문하면, 먼저 개념적으로 설명하고 필요한 경우 코드 위치를 알려줘.
```

## 4. 현재 가장 중요한 최신 변경사항

### 4-1. round MAE plot

기존에는 한 round에서 top-k로 여러 feature가 선택되면 같은 round에 여러 점이 찍혔습니다.

현재 의도는 아래입니다.

```text
round 0: boosting 전 baseline 상태
round 1: round 1에서 선택된 모든 feature가 반영된 최종 상태
round 2: round 2까지 반영된 최종 상태
...
```

즉 round별 defect별 point는 1개만 나와야 합니다.

관련 파일:

```text
feature_boosting/reporting.py
```

관련 함수:

```python
round_residual_summary()
plot_round_residual_points()
```

노트북 호출부:

```python
fig = plot_round_residual_points(
    round_mean_residual,
    output_path=OUT_DIR / "plots" / "round_mean_abs_residual_points.png",
    answer_features_by_defect=ANSWER_FEATURES,
    show_answer_markers=False,
    annotate_features=False,
)
```

### 4-2. null/noise benchmark

6-1은 이제 real feature와 null/noise feature를 같은 candidate pool에 넣고 다시 boosting합니다.

해석:

```text
noise가 top rank에 거의 없음 -> real feature ranking 신뢰 가능
noise가 top rank에 자주 섞임 -> 해당 round ranking은 약함
noise가 selected됨 -> 해당 round는 신뢰하기 어려움
```

중요 컬럼:

```text
n_null_in_top_n
n_selected_null
best_real_margin_vs_best_null
judgement
```

관련 결과 저장 위치:

```text
outputs/.../null_benchmark/
```

### 4-3. feature rank curve의 noise 표시

`plot_candidate_loss_ranking()`은 `is_null_feature=True` 컬럼이 있으면 noise feature를 주황색 X로 표시합니다.

관련 파일:

```text
feature_boosting/reporting.py
```

관련 함수:

```python
plot_candidate_loss_ranking()
_plot_loss_axis()
```

## 5. VS Code에서 최신 코드 반영 시 주의점

파일을 바꾼 뒤에는 반드시 커널을 재시작해야 합니다.

```text
VS Code Notebook
-> Kernel Restart
-> Run All
```

또는 임시로 아래를 실행할 수 있습니다.

```python
import importlib
import feature_boosting.reporting as reporting

importlib.reload(reporting)

from feature_boosting.reporting import (
    round_residual_summary,
    plot_round_residual_points,
    plot_candidate_loss_ranking,
)
```

최신 함수가 로드되었는지 확인:

```python
import inspect
from feature_boosting.reporting import plot_round_residual_points

print(inspect.signature(plot_round_residual_points))
```

정상이면 아래 인자가 보여야 합니다.

```text
show_answer_markers: bool = False
annotate_features: bool = False
```

## 6. 사내에서 git pull이 안 될 때

GitHub에서 파일을 직접 다운로드하거나 복붙해야 할 때, 최소한 아래 파일을 최신으로 맞추세요.

```text
notebooks/residual_feature_boosting_poc.ipynb
feature_boosting/reporting.py
```

null benchmark나 data loader까지 전체 맥락을 맞추려면 폴더 전체 ZIP 다운로드가 가장 안전합니다.

```text
feature_boosting/
notebooks/residual_feature_boosting_poc.ipynb
docs/gpt_context_feature_boosting_rev0.txt
```

## 7. 대화상 주요 결정사항

- test set은 feature 선택 기준으로 직접 쓰기보다 최종 검증용으로 보는 것이 더 타당하다.
- valid 기준으로 feature selection을 하고 test residual은 일반화 확인용으로 본다.
- residual feature boosting은 causal proof가 아니라 residual predictive signal discovery다.
- null/noise benchmark는 PowerSHAP과 철학은 비슷하지만, SHAP importance 대신 residual improvement를 기준으로 비교한다.
- block 방식은 candidate가 많을 때 속도 비교용 실험이며, standard 방식과 완전히 같은 수학적 조건은 아니다.
- round MAE plot은 feature별 점이 아니라 round별 최종 누적 상태를 보여줘야 한다.

## 8. 자주 쓰는 확인 코드

round별 MAE table:

```python
display(
    round_mean_residual
    .pivot_table(
        index=["defect_id", "round"],
        columns="split",
        values="mean_abs_residual",
        aggfunc="first",
    )
    .reset_index()
)
```

round마다 point가 1개인지 확인:

```python
display(
    round_mean_residual
    .groupby(["defect_id", "split", "round"])
    .size()
    .reset_index(name="n_points")
)
```

정상이면 모든 `n_points`가 1입니다.

null feature가 실제로 ranking에 들어갔는지 확인:

```python
display(
    null_competition_top_ranking[
        ["defect_id", "round", "rank", "feature_name", "is_null_feature"]
    ].query("is_null_feature == True").head(20)
)
```

