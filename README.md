# 87 Semantic Force Closure Revisited

Submission-hardening version: v4

Terminal decision: **KILL_ARCHIVE** for ICLR main conference.

This repository contains a reproducible local evidence audit for the research bet:

> Semantic object categories and task intent can change which contacts count as useful force closure.

The v4 rebuild replaces the template scaffold with a deterministic semantic-force-closure benchmark over four manipulation tasks, five category/task-shift splits, eight methods, ablations, stress sweeps, and negative cases.

## Why This Is Archived

- On the combined hard-shift split, `semantic_force_closure_revisited` reaches `0.25595 +/- 0.02584` task success.
- The strongest success baseline, `language_conditioned_grasp_policy`, reaches `0.24065 +/- 0.03032`.
- The paired task-success difference is only `0.01530 +/- 0.03199`.
- The proposed method beats pure geometry and PONG-like force-closure baselines, but does not decisively beat learned semantic baselines.
- A `language_only_semantic_score` ablation has lower semantic-violation rate (`0.15136`) than the full method (`0.28571`).
- At maximum combined stress, `taskgrasp_semantic_ranker` slightly exceeds the proposed method on success.
- The evidence is local and synthetic, not hardware or accepted high-fidelity benchmark validation.

## Reproduce

```powershell
python src\run_experiment.py
```

The runner writes:

- `results/rollouts.csv`
- `results/raw_seed_metrics.csv`
- `results/metrics.csv`
- `results/pairwise_stats.csv`
- `results/ablation_rollouts.csv`
- `results/ablation_seed_metrics.csv`
- `results/ablation_metrics.csv`
- `results/stress_sweep_raw.csv`
- `results/stress_sweep.csv`
- `results/negative_cases.csv`
- `results/summary.txt`
- `figures/semantic_force_closure_*.png`

## Rebuild PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/87.pdf`
