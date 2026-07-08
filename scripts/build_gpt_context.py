from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "docs" / "gpt_context_feature_boosting_rev0.txt"

TEXT_FILES = [
    "README.md",
    "requirements.txt",
    "configs/experiment.yaml",
    "feature_boosting/__init__.py",
    "feature_boosting/baseline_model.py",
    "feature_boosting/cli.py",
    "feature_boosting/config.py",
    "feature_boosting/data_loader.py",
    "feature_boosting/final_model.py",
    "feature_boosting/metrics.py",
    "feature_boosting/modeling.py",
    "feature_boosting/reporting.py",
    "feature_boosting/residual_boosting.py",
    "feature_boosting/shap_analysis.py",
    "feature_boosting/splitter.py",
    "feature_boosting/validation.py",
    "scripts/build_gpt_context.py",
    "scripts/generate_residual_poc_toyset.py",
    "scripts/run_experiment.py",
    "tests/test_data_loader.py",
    "tests/test_metrics.py",
    "tests/test_reporting.py",
    "tests/test_residual_boosting.py",
    "tests/test_validation.py",
]

NOTEBOOK_FILES = [
    "notebooks/residual_feature_boosting_poc.ipynb",
]


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = [
        "# GPT Context Snapshot: feature_boosting_rev0",
        "",
        "이 파일은 사내망에서 GPT에게 한 번에 복붙해서 프로젝트 구조와 코드를 설명하기 위한 단일 텍스트 스냅샷입니다.",
        "데이터 파일, output 파일, notebook 실행 결과는 제외하고 Residual-Based Feature Boosting PoC 실행에 필요한 코드와 설정을 담았습니다.",
        "",
        "GPT에게 붙여넣을 때는 먼저 이렇게 요청하면 됩니다:",
        '\"아래는 feature_boosting_rev0 프로젝트의 단일 파일 코드 스냅샷이야. 전체 구조를 이해하고, 내가 묻는 코드/수식/실행 오류를 이 맥락 기준으로 설명해줘.\"',
        "",
        "## Included Files",
        "",
        *[f"- {path}" for path in TEXT_FILES],
        *[f"- {path} (extracted notebook cells)" for path in NOTEBOOK_FILES],
        "",
    ]

    for rel_path in TEXT_FILES:
        path = ROOT / rel_path
        parts.append(_section_header(f"FILE: {rel_path}"))
        parts.append(_read_text(path))
        parts.append("")

    for rel_path in NOTEBOOK_FILES:
        path = ROOT / rel_path
        parts.append(_section_header(f"NOTEBOOK: {rel_path}"))
        parts.append(_notebook_to_text(path))
        parts.append("")

    OUTPUT_PATH.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")
    return 0


def _section_header(title: str) -> str:
    line = "=" * 100
    return f"{line}\n{title}\n{line}"


def _read_text(path: Path) -> str:
    if not path.exists():
        return f"[missing file: {path.relative_to(ROOT)}]"
    return path.read_text(encoding="utf-8")


def _notebook_to_text(path: Path) -> str:
    if not path.exists():
        return f"[missing notebook: {path.relative_to(ROOT)}]"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        cell_type = cell.get("cell_type", "unknown")
        source = "".join(cell.get("source", []))
        chunks.append(f"--- CELL {index:02d} [{cell_type}] ---")
        chunks.append(source.rstrip())
        chunks.append("")
    return "\n".join(chunks).rstrip()


if __name__ == "__main__":
    raise SystemExit(main())
