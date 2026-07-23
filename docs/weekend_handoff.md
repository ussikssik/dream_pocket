# Weekend Handoff for Codex

Last updated: 2026-07-23

This document is meant to let a new Codex session on another laptop continue
the current project with minimal context loss.

## Repository

- Main remote: `https://github.com/ussikssik/dream_pocket.git`
- Working branch: `codex/feature-evidence-booster`
- Local project path on the current PC is the current Codex workspace under
  the user's OneDrive Documents folder. The exact path contains Korean
  characters, so prefer using GitHub clone instructions on the new laptop.

Use `origin` as the primary remote unless the user explicitly asks to use
`rev0`.

## New Laptop Setup

Run these commands on the new laptop:

```powershell
git clone https://github.com/ussikssik/dream_pocket.git
cd dream_pocket
git checkout codex/feature-evidence-booster
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks venv activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then open the cloned `dream_pocket` folder in Codex.

## First Prompt for New Codex Session

Paste this into the new laptop's Codex chat:

```text
Read docs/weekend_handoff.md, README.md, and docs/webgpt_conversation_starter.md first.
Then continue this feature boosting project from the current branch.

I am working on a residual-based feature boosting PoC for semiconductor defect
analysis. Please answer in Korean, explain beginner-friendly steps, and when
changing code, tell me exactly which file/function you are editing.

Start by checking git status, confirming the active branch, and summarizing the
main runnable notebooks/scripts.
```

## Project Goal

The main project is a Residual-Based Feature Boosting PoC.

Core idea:

- Train a baseline model using only base process features.
- Compute residuals from the baseline prediction.
- Score candidate features by whether they explain or reduce residual error,
  especially for defect-specific bad groups.
- Select features by top-k or threshold rules.
- Refit cumulative models round by round and compare final metrics.
- Use null/noise feature competition as a sanity benchmark.
- Keep charts and reports interpretable for engineering review.

This is not intended to be causal proof. Treat it as residual predictive signal
discovery with evidence checks.

## Important Files

- `notebooks/residual_feature_boosting_poc.ipynb`
  Main notebook for the residual boosting PoC.

- `feature_boosting/residual_boosting.py`
  Core residual feature boosting logic.

- `feature_boosting/reporting.py`
  Plotting and summary table helpers. Important recent work involved round MAE
  plots and candidate ranking charts.

- `feature_boosting/data_loader.py`
  CSV loading and standardization for demo data and six-file raw input.

- `feature_boosting/answer_features.py`
  Known-answer feature matching and highlighting rules.

- `scripts/generate_residual_poc_toyset.py`
  Creates toy/demo data for the residual PoC.

- `scripts/run_experiment.py`
  CLI entry point using `configs/experiment.yaml`.

- `docs/webgpt_conversation_starter.md`
  Prompt/context starter for moving the work into a GPT or Codex chat.

- `tcat_xtacking_simple_evaluator.py`
  Separate experimental PySide6/Matplotlib GUI for TCAT/Xtacking geometry
  object evaluation. This is adjacent exploratory work, not the main residual
  boosting pipeline.

## Current New/Uncommitted Items Before Handoff

At the time this handoff was created, these files were new in the working tree:

- `docs/webgpt_conversation_starter.md`
- `docs/weekend_handoff.md`
- `data/residual_poc_demo/`
- `tcat_xtacking_simple_evaluator.py`

The current branch was ahead of `origin/codex/feature-evidence-booster` by many
commits, so push this branch before leaving the current PC.

## Main Notebook Notes

When opening `notebooks/residual_feature_boosting_poc.ipynb` on a new machine:

1. Restart the Jupyter kernel.
2. Run all cells from the top.
3. If imports look stale after pulling new code, restart the kernel again.

Useful reload snippet:

```python
import importlib
import feature_boosting.reporting as reporting

importlib.reload(reporting)
```

Expected recent signature:

```python
from feature_boosting.reporting import plot_round_residual_points
import inspect

print(inspect.signature(plot_round_residual_points))
```

It should include:

```text
show_answer_markers: bool = False
annotate_features: bool = False
```

## Recent Behavioral Decisions

- Round residual/MAE plots should show one point per defect/split/round.
- A round point represents the cumulative model state after that round's
  selected features have been applied.
- Feature names do not need to be annotated on the round MAE chart.
- Candidate ranking charts may highlight:
  - normal candidate features
  - selected features
  - answer features
  - null/noise features
- Null/noise benchmark interpretation:
  - few or no null features near the top means the real ranking is more stable
  - frequent null features near the top means that round's ranking is weak
  - selected null features are a warning sign for that round

Important null benchmark columns:

```text
n_null_in_top_n
n_selected_null
best_real_margin_vs_best_null
judgement
```

## Useful Commands

Run tests:

```powershell
python -m pytest
```

Run the configured experiment:

```powershell
python scripts/run_experiment.py --config configs/experiment.yaml
```

Generate residual PoC demo data:

```powershell
python scripts/generate_residual_poc_toyset.py
```

Run a smaller quick demo:

```powershell
python scripts/generate_residual_poc_toyset.py --rows 600 --candidate-features 200
```

## Git Handoff Checklist

Before switching machines:

```powershell
git status
git add .
git commit -m "Add weekend handoff context"
git push origin codex/feature-evidence-booster
```

On the new laptop:

```powershell
git clone https://github.com/ussikssik/dream_pocket.git
cd dream_pocket
git checkout codex/feature-evidence-booster
git status
```

After new work on the laptop:

```powershell
git add .
git commit -m "Continue weekend work"
git push origin codex/feature-evidence-booster
```

When returning to the current PC:

```powershell
git checkout codex/feature-evidence-booster
git pull origin codex/feature-evidence-booster
```

## Cautions

- Do not assume Codex chat/project UI state will fully transfer across devices.
  The durable handoff path is GitHub plus this file.
- Large private or real manufacturing data should not be committed unless the
  user confirms it is safe.
- Generated `outputs/` and model artifacts can be large; commit only when they
  are intentionally part of the handoff.
- If Korean text looks garbled in older docs, prefer this file as the clean
  handoff source.
