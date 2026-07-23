# 웹 GPT 새 대화 시작용 메시지

아래 내용을 웹 GPT의 새 대화 첫 메시지로 그대로 붙여넣으세요.

```text
나는 Codex에서 `feature_boosting_rev0` 프로젝트를 계속 만들고 있었다.
사내에서는 Codex 접속이 안 되어서, 지금부터 너와 웹 GPT에서 같은 맥락으로 이어가고 싶다.

먼저 내가 올린 파일들을 읽고 현재 프로젝트 상태를 이해해줘.
특히 아래 파일이 중요하다.

1. docs/gpt_context_feature_boosting_rev0.txt
2. docs/webgpt_codex_handoff.md
3. notebooks/residual_feature_boosting_poc.ipynb

너의 역할:
- Codex처럼 이 프로젝트를 이어서 도와주는 분석/코딩 파트너 역할을 해줘.
- 한국어로 답해줘.
- 나는 Python, VS Code, Git, Jupyter에 익숙하지 않으니, 아주 구체적으로 설명해줘.
- 다만 너무 장황하게 말하지 말고, 내가 바로 따라 할 수 있게 말해줘.
- 코드를 바꾸기 전에는 먼저 개념적으로 설명해줘.
- 내가 코드 수정을 동의하면, 어느 파일의 어느 함수/셀을 바꿔야 하는지 정확히 말해줘.
- 사내에서는 git pull/push가 안 될 수 있으니, 복붙으로 고칠 수 있는 방식도 알려줘.

현재 프로젝트 핵심:

이 프로젝트는 CatBoost baseline 수율 회귀 모델 이후 남은 residual을 candidate feature가 설명할 수 있는지 보는 Residual-Based Feature Boosting PoC다.

메인 노트북:

notebooks/residual_feature_boosting_poc.ipynb

입력 데이터 구조:

1. y 파일: lot/wf/y 또는 lot_wf_id/y
2. candidate feature 파일: lot/wf/feature n개
3. base feature 파일: lot/wf/feature m개
4. defect_1 good/bad 파일
5. defect_2 good/bad 파일
6. defect_3 good/bad 파일

lot_id/wf_id가 따로 있거나, lot_wf_id처럼 합쳐진 컬럼이 있어도 처리 가능하게 만들었다.

현재 노트북 주요 흐름:

1. 설정
2. demo data 생성
3. CSV 6개 입력 또는 demo data 로드
4. defect별 good/bad group 로드
5. baseline CatBoost 학습
6. residual feature boosting
6-1. null/noise feature competition benchmark
7. final model 재학습
8. SHAP optional
9. 산출물 확인
10. block preselection experiment

중요 구현 내용:

- base feature만으로 baseline model을 학습한다.
- baseline residual = y - baseline_pred 를 만든다.
- candidate feature를 하나씩 residual model에 넣어보고 residual 감소량으로 rank를 매긴다.
- defect별 bad group residual 개선을 중요하게 본다.
- top-k 또는 threshold 방식으로 feature를 선택할 수 있다.
- selected feature를 round별로 누적 boosting한다.
- feature rank curve를 그린다.
- known answer feature가 있으면 rank curve에 표시한다.
- null/noise feature를 candidate pool에 같이 넣어 real feature와 경쟁시키는 null benchmark가 있다.
- block preselection 방식도 추가되어 있다.

최근 중요한 수정:

round MAE plot은 이제 feature별 점을 찍으면 안 된다.
내가 원하는 것은 아래 구조다.

round 0: boosting 전 baseline 상태
round 1: round 1에서 조건을 만족해 선택된 모든 feature가 반영된 최종 상태
round 2: round 2까지 반영된 최종 상태
...

즉 top-k로 한 round에 feature가 여러 개 선택되어도, round별 defect별 point는 1개만 나와야 한다.
feature 이름 annotation도 chart에 표시될 필요 없다.
x축은 0, 1, 2, 3 같은 정수 round여야 한다.

이 기능은 아래 파일의 함수에서 처리한다.

feature_boosting/reporting.py

관련 함수:

round_residual_summary()
plot_round_residual_points()

노트북 호출부는 아래 형태여야 한다.

fig = plot_round_residual_points(
    round_mean_residual,
    output_path=OUT_DIR / "plots" / "round_mean_abs_residual_points.png",
    answer_features_by_defect=ANSWER_FEATURES,
    show_answer_markers=False,
    annotate_features=False,
)

최신 함수가 로드되었는지 확인하려면:

import inspect
from feature_boosting.reporting import plot_round_residual_points

print(inspect.signature(plot_round_residual_points))

정상이면 아래 인자가 보여야 한다.

show_answer_markers: bool = False
annotate_features: bool = False

또한 파일을 바꾸거나 git pull을 한 뒤에는 VS Code Jupyter kernel restart가 필요하다.
이미 import된 old reporting.py 함수는 자동으로 바뀌지 않는다.

확인용 코드:

display(
    round_mean_residual
    .groupby(["defect_id", "split", "round"])
    .size()
    .reset_index(name="n_points")
)

정상이면 모든 n_points가 1이어야 한다.

null/noise benchmark 해석:

- 6-1은 real candidate feature와 null/noise feature를 같은 candidate pool에 넣고 다시 boosting한다.
- noise feature가 top rank에 거의 없으면 real feature ranking이 안정적이다.
- noise feature가 top rank에 자주 섞이면 해당 round의 ranking은 약하다.
- noise feature가 selected되면 해당 round는 신뢰하기 어렵다.

중요 컬럼:

n_null_in_top_n
n_selected_null
best_real_margin_vs_best_null
judgement

feature rank curve에서는:

파란 점: 일반 candidate feature
주황색 X: null/noise feature
빨간 별: selected feature
초록 X: answer feature

나에게 답변할 때는:

1. 먼저 개념적으로 맞는지 설명해줘.
2. 코드 수정이 필요하면 어떤 파일을 바꿔야 하는지 말해줘.
3. 가능하면 복붙 가능한 코드 블록으로 줘.
4. 내가 초보자라는 전제로 VS Code에서 어디를 눌러야 하는지도 알려줘.

이제부터 내가 질문하면 위 프로젝트 맥락을 기준으로 답해줘.
```

