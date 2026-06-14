# Paper 87 Rebuild Plan

Last update: 2026-06-14 12:53:25 +01:00

## Target Claim

Semantic object categories and task intent can change which contacts count as useful force closure. A grasp planner should therefore evaluate closure in a semantic-task state, not only in geometric contact space. The proposed mechanism is worth keeping only if it beats strong geometry, probabilistic-normal, language-affordance, and risk-aware force-closure baselines under category/task shift.

## Hostile Prior-Work Pressure

The local pool makes this a crowded claim. The closest threats include PONG/probabilistic object normals, language-conditioned task-oriented grasping, VLM grasp pipelines, prompt-guided force-closure analysis, tactile force-control work, soft-gripper force analysis, and open-vocabulary robotic manipulation systems. The v4 rebuild must not claim novelty from "language plus grasping" or "force closure plus uncertainty"; it must isolate whether semantic/task closure changes the selected contact set and downstream task outcome.

## Evidence To Build

Replace the shared probability-template scaffold with a deterministic local grasp benchmark that generates candidate contacts, surface normals, friction, semantic zones, task-critical surfaces, deformation/damage risk, language ambiguity, and execution noise.

### Tasks

- mug pouring and handoff, where the handle/lip/interior have different closure constraints.
- tool use, where grasping a blade/tip may be stable but semantically invalid.
- spray bottle activation, where the trigger/nozzle must remain accessible.
- deformable package handling, where high closure force can crush or tear the object.

### Splits

- `seen_categories_seen_tasks`
- `novel_category_same_semantics`
- `task_intent_shift`
- `ambiguous_language_shift`
- `combined_hard_shift`

### Methods

- `geometry_force_closure`
- `probabilistic_normals_pong`
- `affordance_only_vlm`
- `taskgrasp_semantic_ranker`
- `language_conditioned_grasp_policy`
- `risk_aware_force_closure`
- `semantic_force_closure_revisited` (proposed)
- `oracle_task_closure`

### Metrics

- task success.
- mechanical closure margin.
- semantic violation rate.
- functional-surface blockage.
- slip rate.
- damage rate.
- semantic calibration error.
- regret versus oracle closure.

### Ablations

- full semantic-force-closure planner.
- minus task-intent conditioning.
- minus forbidden-surface semantics.
- minus functional-access constraint.
- minus material/deformation risk.
- geometry-only closure score.
- language-only semantic score.

### Stress Tests

- surface-normal noise.
- friction shift.
- semantic label noise.
- task-language ambiguity.
- material/deformation shift.
- combined stress.

### Terminal Gate

Mark `STRONG_REVISE` only if the proposed method beats the strongest non-oracle baseline on combined hard-shift task success, semantic violations, and functional blockage, while ablations degrade the mechanism and stress tests show a clear robustness margin. Otherwise mark `KILL_ARCHIVE`.

Even a `STRONG_REVISE` outcome is not ICLR-main ready without real robot hardware, an accepted external benchmark, or a high-fidelity dexterous grasp simulator.
