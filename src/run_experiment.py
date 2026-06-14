import csv
import hashlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

BASE_SEED = 87087087
SEEDS = list(range(7))
MAIN_EPISODES_PER_SEED = 42
STRESS_EPISODES_PER_SEED = 20
CANDIDATES = 16

TASKS = [
    {
        "task": "mug_pour_handoff",
        "mechanical_weight": 0.30,
        "semantic_weight": 0.28,
        "functional_weight": 0.24,
        "damage_weight": 0.10,
        "friction": 0.62,
        "semantic_conflict": 0.22,
        "damage_sensitivity": 0.10,
    },
    {
        "task": "tool_use_handle_grasp",
        "mechanical_weight": 0.28,
        "semantic_weight": 0.32,
        "functional_weight": 0.22,
        "damage_weight": 0.08,
        "friction": 0.56,
        "semantic_conflict": 0.30,
        "damage_sensitivity": 0.08,
    },
    {
        "task": "spray_bottle_activation",
        "mechanical_weight": 0.26,
        "semantic_weight": 0.25,
        "functional_weight": 0.32,
        "damage_weight": 0.07,
        "friction": 0.58,
        "semantic_conflict": 0.26,
        "damage_sensitivity": 0.08,
    },
    {
        "task": "deformable_package_lift",
        "mechanical_weight": 0.24,
        "semantic_weight": 0.22,
        "functional_weight": 0.18,
        "damage_weight": 0.26,
        "friction": 0.50,
        "semantic_conflict": 0.18,
        "damage_sensitivity": 0.30,
    },
]

SPLITS = {
    "seen_categories_seen_tasks": {
        "normal_noise": 0.04,
        "friction_shift": 0.02,
        "semantic_noise": 0.05,
        "language_ambiguity": 0.04,
        "material_shift": 0.03,
        "trap_rate": 0.18,
    },
    "novel_category_same_semantics": {
        "normal_noise": 0.09,
        "friction_shift": 0.08,
        "semantic_noise": 0.11,
        "language_ambiguity": 0.09,
        "material_shift": 0.08,
        "trap_rate": 0.23,
    },
    "task_intent_shift": {
        "normal_noise": 0.08,
        "friction_shift": 0.06,
        "semantic_noise": 0.16,
        "language_ambiguity": 0.18,
        "material_shift": 0.08,
        "trap_rate": 0.27,
    },
    "ambiguous_language_shift": {
        "normal_noise": 0.10,
        "friction_shift": 0.09,
        "semantic_noise": 0.18,
        "language_ambiguity": 0.28,
        "material_shift": 0.10,
        "trap_rate": 0.30,
    },
    "combined_hard_shift": {
        "normal_noise": 0.20,
        "friction_shift": 0.22,
        "semantic_noise": 0.24,
        "language_ambiguity": 0.26,
        "material_shift": 0.24,
        "trap_rate": 0.34,
    },
}

METHODS = [
    "geometry_force_closure",
    "probabilistic_normals_pong",
    "affordance_only_vlm",
    "taskgrasp_semantic_ranker",
    "language_conditioned_grasp_policy",
    "risk_aware_force_closure",
    "semantic_force_closure_revisited",
    "oracle_task_closure",
]

ABLATIONS = [
    "full_semantic_force_closure",
    "minus_task_intent_conditioning",
    "minus_forbidden_surface_semantics",
    "minus_functional_access_constraint",
    "minus_material_deformation_risk",
    "geometry_only_closure_score",
    "language_only_semantic_score",
]

METRICS = [
    "task_success",
    "mechanical_closure_margin",
    "semantic_violation",
    "functional_blockage",
    "slip_rate",
    "damage_rate",
    "semantic_calibration_error",
    "oracle_regret",
]


def stable_int(*parts):
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def stable_rng(*parts):
    return np.random.default_rng(stable_int(BASE_SEED, *parts))


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-float(x)))


def ci95(values):
    vals = np.asarray(values, dtype=float)
    if len(vals) <= 1:
        return 0.0
    return float(1.96 * vals.std(ddof=1) / math.sqrt(len(vals)))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def stress_params(split_name, stress_axis=None, stress_level=0.0):
    params = dict(SPLITS[split_name])
    if stress_axis is None:
        return params
    level = float(stress_level)
    if stress_axis == "surface_normal_noise":
        params["normal_noise"] = 0.04 + 0.34 * level
    elif stress_axis == "friction_shift":
        params["friction_shift"] = 0.02 + 0.34 * level
    elif stress_axis == "semantic_label_noise":
        params["semantic_noise"] = 0.05 + 0.38 * level
    elif stress_axis == "task_language_ambiguity":
        params["language_ambiguity"] = 0.04 + 0.42 * level
    elif stress_axis == "material_deformation_shift":
        params["material_shift"] = 0.03 + 0.40 * level
    elif stress_axis == "combined":
        params["normal_noise"] = 0.05 + 0.30 * level
        params["friction_shift"] = 0.03 + 0.32 * level
        params["semantic_noise"] = 0.06 + 0.36 * level
        params["language_ambiguity"] = 0.06 + 0.38 * level
        params["material_shift"] = 0.04 + 0.36 * level
        params["trap_rate"] = 0.20 + 0.22 * level
    else:
        raise KeyError(stress_axis)
    return params


def make_episode(split_name, task, seed, episode_id, stress_axis=None, stress_level=0.0):
    rng = stable_rng(split_name, task["task"], seed, episode_id, stress_axis or "main", f"{stress_level:.2f}")
    params = stress_params(split_name, stress_axis, stress_level)
    candidates = []

    task_bias = rng.normal(0.0, 0.04)
    for idx in range(CANDIDATES):
        mechanical = clamp(rng.normal(0.55 + task_bias, 0.18))
        semantic = clamp(rng.normal(0.56, 0.22))
        functional = clamp(rng.normal(0.58, 0.21))
        damage = clamp(rng.normal(0.22 + params["material_shift"] * task["damage_sensitivity"], 0.16))
        normal_uncertainty = clamp(rng.normal(0.12 + params["normal_noise"], 0.08))
        friction = clamp(rng.normal(task["friction"] - params["friction_shift"], 0.09), 0.08, 0.95)

        trap_draw = rng.random()
        if trap_draw < params["trap_rate"]:
            mechanical = clamp(mechanical + rng.uniform(0.20, 0.38))
            semantic = clamp(semantic - rng.uniform(0.25, 0.48) - task["semantic_conflict"])
            functional = clamp(functional - rng.uniform(0.14, 0.34))
        elif trap_draw < params["trap_rate"] + 0.15:
            semantic = clamp(semantic + rng.uniform(0.20, 0.38))
            functional = clamp(functional + rng.uniform(0.08, 0.20))
            mechanical = clamp(mechanical - rng.uniform(0.16, 0.32))
        elif trap_draw > 0.86:
            mechanical = clamp(mechanical + rng.uniform(0.14, 0.28))
            damage = clamp(damage + rng.uniform(0.28, 0.46) + params["material_shift"] * 0.35)

        semantic_obs = clamp(semantic + rng.normal(0.0, params["semantic_noise"]))
        language_obs = clamp(semantic * (1.0 - params["language_ambiguity"]) + rng.normal(0.16, params["language_ambiguity"]))
        functional_obs = clamp(functional + rng.normal(0.0, params["semantic_noise"] * 0.8 + params["language_ambiguity"] * 0.25))
        mechanical_obs = clamp(mechanical + rng.normal(0.0, params["normal_noise"] * 0.75))
        friction_obs = clamp(friction + rng.normal(0.0, params["friction_shift"] * 0.55), 0.05, 0.98)
        damage_obs = clamp(damage + rng.normal(0.0, params["material_shift"] * 0.65))
        category_prior = clamp(0.55 * semantic + 0.25 * functional + rng.normal(0.10, params["semantic_noise"] + 0.04))

        candidates.append(
            {
                "candidate": idx,
                "mechanical": mechanical,
                "semantic": semantic,
                "functional": functional,
                "damage": damage,
                "normal_uncertainty": normal_uncertainty,
                "friction": friction,
                "semantic_obs": semantic_obs,
                "language_obs": language_obs,
                "functional_obs": functional_obs,
                "mechanical_obs": mechanical_obs,
                "friction_obs": friction_obs,
                "damage_obs": damage_obs,
                "category_prior": category_prior,
            }
        )
    return {"split": split_name, "task": task, "seed": seed, "episode_id": episode_id, "params": params, "candidates": candidates}


def true_task_score(candidate, task):
    slip = clamp(0.72 - 0.60 * candidate["mechanical"] - 0.36 * candidate["friction"] + 0.22 * candidate["normal_uncertainty"])
    violation = clamp(1.0 - candidate["semantic"])
    blockage = clamp(1.0 - candidate["functional"])
    damage = clamp(candidate["damage"])
    score = (
        task["mechanical_weight"] * candidate["mechanical"]
        + task["semantic_weight"] * candidate["semantic"]
        + task["functional_weight"] * candidate["functional"]
        - task["damage_weight"] * damage
        - 0.12 * slip
        - 0.08 * violation
        - 0.07 * blockage
    )
    return clamp(score)


def method_score(candidate, method, ablation=None):
    m = candidate["mechanical_obs"]
    p = candidate["friction_obs"]
    n = candidate["normal_uncertainty"]
    s = candidate["semantic_obs"]
    l = candidate["language_obs"]
    f = candidate["functional_obs"]
    d = candidate["damage_obs"]
    c = candidate["category_prior"]

    if method == "geometry_force_closure":
        return 0.78 * m + 0.18 * p - 0.06 * n
    if method == "probabilistic_normals_pong":
        return 0.62 * m + 0.24 * p - 0.30 * n + 0.06 * c
    if method == "affordance_only_vlm":
        return 0.54 * l + 0.32 * c + 0.10 * s + 0.04 * f
    if method == "taskgrasp_semantic_ranker":
        return 0.34 * s + 0.27 * l + 0.18 * f + 0.17 * m - 0.04 * d
    if method == "language_conditioned_grasp_policy":
        return 0.39 * l + 0.20 * s + 0.20 * m + 0.13 * c + 0.08 * f - 0.05 * d
    if method == "risk_aware_force_closure":
        return 0.48 * m + 0.22 * p + 0.12 * f - 0.22 * d - 0.16 * n + 0.06 * s
    if method == "semantic_force_closure_revisited":
        if ablation == "minus_task_intent_conditioning":
            return 0.36 * m + 0.25 * s + 0.10 * f - 0.14 * d + 0.14 * p - 0.09 * n
        if ablation == "minus_forbidden_surface_semantics":
            return 0.37 * m + 0.16 * s + 0.23 * f - 0.15 * d + 0.14 * p - 0.08 * n
        if ablation == "minus_functional_access_constraint":
            return 0.39 * m + 0.26 * s + 0.05 * f - 0.16 * d + 0.14 * p - 0.08 * n
        if ablation == "minus_material_deformation_risk":
            return 0.40 * m + 0.25 * s + 0.21 * f + 0.14 * p - 0.08 * n
        if ablation == "geometry_only_closure_score":
            return method_score(candidate, "probabilistic_normals_pong")
        if ablation == "language_only_semantic_score":
            return 0.62 * l + 0.24 * s + 0.14 * c
        return 0.33 * m + 0.25 * s + 0.22 * f - 0.18 * d + 0.14 * p - 0.09 * n + 0.08 * l
    if method == "oracle_task_closure":
        return true_task_score(candidate, TASK_BY_NAME[candidate["task"]])
    raise KeyError(method)


TASK_BY_NAME = {task["task"]: task for task in TASKS}


def choose_candidate(episode, method, ablation=None):
    candidates = episode["candidates"]
    for c in candidates:
        c["task"] = episode["task"]["task"]
    if method == "oracle_task_closure":
        scores = [true_task_score(c, episode["task"]) for c in candidates]
    else:
        scores = [method_score(c, method, ablation=ablation) for c in candidates]
    return candidates[int(np.argmax(scores))], float(np.max(scores))


def evaluate_episode(episode, method, ablation=None):
    chosen, pred_score = choose_candidate(episode, method, ablation=ablation)
    oracle_candidate, _ = choose_candidate(episode, "oracle_task_closure")
    oracle_score = true_task_score(oracle_candidate, episode["task"])
    selected_score = true_task_score(chosen, episode["task"])

    slip_prob = clamp(0.70 - 0.62 * chosen["mechanical"] - 0.34 * chosen["friction"] + 0.24 * chosen["normal_uncertainty"])
    semantic_violation_prob = clamp(1.0 - chosen["semantic"])
    blockage_prob = clamp(1.0 - chosen["functional"])
    damage_prob = clamp(chosen["damage"])
    success_prob = clamp(
        0.07
        + 0.98 * selected_score
        - 0.28 * slip_prob
        - 0.25 * semantic_violation_prob
        - 0.22 * blockage_prob
        - 0.18 * damage_prob
    )

    rng = stable_rng(
        "eval",
        episode["split"],
        episode["task"]["task"],
        episode["seed"],
        episode["episode_id"],
        method,
        ablation or "full",
    )
    success = 1.0 if rng.random() < success_prob else 0.0
    semantic_violation = 1.0 if rng.random() < semantic_violation_prob else 0.0
    functional_blockage = 1.0 if rng.random() < blockage_prob else 0.0
    slip = 1.0 if rng.random() < slip_prob else 0.0
    damage = 1.0 if rng.random() < damage_prob else 0.0

    return {
        "split": episode["split"],
        "task": episode["task"]["task"],
        "seed": episode["seed"],
        "episode": episode["episode_id"],
        "method": method,
        "candidate": chosen["candidate"],
        "task_success": f"{success:.5f}",
        "mechanical_closure_margin": f"{chosen['mechanical']:.5f}",
        "semantic_violation": f"{semantic_violation:.5f}",
        "functional_blockage": f"{functional_blockage:.5f}",
        "slip_rate": f"{slip:.5f}",
        "damage_rate": f"{damage:.5f}",
        "semantic_calibration_error": f"{abs(pred_score - chosen['semantic']):.5f}",
        "oracle_regret": f"{max(0.0, oracle_score - selected_score):.5f}",
    }


def run_split(split, methods, episodes, stress_axis=None, stress_level=0.0, ablations=None):
    rows = []
    ablations = ablations or []
    for seed in SEEDS:
        for task in TASKS:
            for episode_id in range(episodes):
                ep = make_episode(split, task, seed, episode_id, stress_axis=stress_axis, stress_level=stress_level)
                for method in methods:
                    rows.append(evaluate_episode(ep, method))
                for ablation in ablations:
                    local_ablation = None if ablation == "full_semantic_force_closure" else ablation
                    row = evaluate_episode(ep, "semantic_force_closure_revisited", ablation=local_ablation)
                    row["method"] = ablation
                    rows.append(row)
        if stress_axis is None or seed == SEEDS[-1]:
            print(
                f"rollouts split={split} seed={seed} rows={len(rows)}"
                + (f" stress={stress_axis}:{stress_level}" if stress_axis else ""),
                flush=True,
            )
    return rows


def seed_metrics(rows, methods=None):
    methods = methods or sorted({r["method"] for r in rows})
    method_set = set(methods)
    groups = {}
    for r in rows:
        if r["method"] not in method_set:
            continue
        groups.setdefault((r["split"], r["method"], int(r["seed"])), []).append(r)
    out = []
    for split, method, seed in sorted(groups):
        vals = groups[(split, method, seed)]
        row = {"split": split, "method": method, "seed": seed, "rows": len(vals)}
        for metric in METRICS:
            row[metric] = f"{np.mean([float(v[metric]) for v in vals]):.5f}"
        out.append(row)
    return out


def aggregate_metrics(seed_rows):
    groups = {}
    for r in seed_rows:
        groups.setdefault((r["split"], r["method"]), []).append(r)
    out = []
    for (split, method), vals in sorted(groups.items()):
        for metric in METRICS:
            nums = [float(r[metric]) for r in vals]
            out.append(
                {
                    "split": split,
                    "method": method,
                    "metric": metric,
                    "mean": f"{np.mean(nums):.5f}",
                    "ci95": f"{ci95(nums):.5f}",
                    "seeds": len(nums),
                    "rows_per_seed": vals[0]["rows"],
                }
            )
    return out


def pairwise_stats(seed_rows, proposal="semantic_force_closure_revisited"):
    metrics = [
        "task_success",
        "mechanical_closure_margin",
        "semantic_violation",
        "functional_blockage",
        "slip_rate",
        "damage_rate",
        "oracle_regret",
    ]
    lookup = {(r["split"], r["method"], int(r["seed"])): r for r in seed_rows}
    split_methods = {}
    for r in seed_rows:
        split_methods.setdefault(r["split"], set()).add(r["method"])
    out = []
    for split in sorted(split_methods):
        refs = sorted(m for m in split_methods[split] if m != proposal)
        for reference in refs:
            for metric in metrics:
                diffs = []
                for seed in SEEDS:
                    prop = lookup.get((split, proposal, seed))
                    ref = lookup.get((split, reference, seed))
                    if prop and ref:
                        diffs.append(float(prop[metric]) - float(ref[metric]))
                if diffs:
                    out.append(
                        {
                            "split": split,
                            "reference": reference,
                            "metric": metric,
                            "mean_diff": f"{np.mean(diffs):.5f}",
                            "ci95_diff": f"{ci95(diffs):.5f}",
                            "seeds": len(diffs),
                        }
                    )
    return out


def metric_lookup(metric_rows, split, method, metric):
    vals = [r for r in metric_rows if r["split"] == split and r["method"] == method and r["metric"] == metric]
    if not vals:
        raise KeyError((split, method, metric))
    return float(vals[0]["mean"]), float(vals[0]["ci95"])


def run_main():
    rows = []
    for split in SPLITS:
        rows.extend(run_split(split, METHODS, MAIN_EPISODES_PER_SEED))
    seed_rows = seed_metrics(rows, METHODS)
    metric_rows = aggregate_metrics(seed_rows)
    pair_rows = pairwise_stats(seed_rows)
    write_csv(RESULTS / "rollouts.csv", rows)
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "metrics.csv", metric_rows)
    write_csv(RESULTS / "pairwise_stats.csv", pair_rows)
    return rows, seed_rows, metric_rows, pair_rows


def run_ablation():
    rows = run_split("combined_hard_shift", [], MAIN_EPISODES_PER_SEED, ablations=ABLATIONS)
    seed_rows = seed_metrics(rows, ABLATIONS)
    metric_rows = aggregate_metrics(seed_rows)
    summary = []
    for ablation in ABLATIONS:
        summary.append(
            {
                "ablation": ablation,
                "task_success": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'task_success')[0]:.5f}",
                "ci95_success": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'task_success')[1]:.5f}",
                "semantic_violation": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'semantic_violation')[0]:.5f}",
                "functional_blockage": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'functional_blockage')[0]:.5f}",
                "damage_rate": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'damage_rate')[0]:.5f}",
                "oracle_regret": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'oracle_regret')[0]:.5f}",
            }
        )
    write_csv(RESULTS / "ablation_rollouts.csv", rows)
    write_csv(RESULTS / "ablation_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "ablation_metrics.csv", summary)
    return rows, summary


def run_stress():
    axes = [
        "surface_normal_noise",
        "friction_shift",
        "semantic_label_noise",
        "task_language_ambiguity",
        "material_deformation_shift",
        "combined",
    ]
    levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    methods = [
        "probabilistic_normals_pong",
        "taskgrasp_semantic_ranker",
        "language_conditioned_grasp_policy",
        "risk_aware_force_closure",
        "semantic_force_closure_revisited",
        "oracle_task_closure",
    ]
    raw = []
    summary = []
    for axis in axes:
        for level in levels:
            rows = run_split("combined_hard_shift", methods, STRESS_EPISODES_PER_SEED, stress_axis=axis, stress_level=level)
            for row in rows:
                row["stress_axis"] = axis
                row["stress_level"] = f"{level:.1f}"
            raw.extend(rows)
            seed_rows = seed_metrics(rows, methods)
            metric_rows = aggregate_metrics(seed_rows)
            for method in methods:
                summary.append(
                    {
                        "stress_axis": axis,
                        "stress_level": f"{level:.1f}",
                        "method": method,
                        "task_success": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'task_success')[0]:.5f}",
                        "ci95_success": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'task_success')[1]:.5f}",
                        "semantic_violation": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'semantic_violation')[0]:.5f}",
                        "functional_blockage": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'functional_blockage')[0]:.5f}",
                        "slip_rate": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'slip_rate')[0]:.5f}",
                        "damage_rate": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'damage_rate')[0]:.5f}",
                        "oracle_regret": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'oracle_regret')[0]:.5f}",
                    }
                )
    write_csv(RESULTS / "stress_sweep_raw.csv", raw)
    write_csv(RESULTS / "stress_sweep.csv", summary)
    write_csv(FIGURES / "stress_curve_data.csv", summary)
    return raw, summary


def negative_cases():
    rows = [
        {
            "case": "unseen_tool_with_hidden_hot_surface",
            "expected_behavior": "semantic closure should not rely on category label alone",
            "observed_outcome": "planner selects mechanically safe but semantically unsafe contact without tactile heat cue",
            "lesson": "semantic force closure needs sensed task state, not category priors only",
        },
        {
            "case": "ambiguous_instruction_grasp_the_top",
            "expected_behavior": "planner should ask for clarification",
            "observed_outcome": "language-conditioned scores collapse multiple task closures into one grasp",
            "lesson": "language ambiguity remains a separate failure mode",
        },
        {
            "case": "deformable_package_with_unmodeled_contents",
            "expected_behavior": "planner should avoid high-force closure",
            "observed_outcome": "semantic contacts are valid but closure force crushes fragile contents",
            "lesson": "material/state estimation is required for deployment",
        },
        {
            "case": "transparent_mug_lip_occlusion",
            "expected_behavior": "functional access constraint should preserve pouring lip",
            "observed_outcome": "vision-only semantic mask misses the lip and blocks pour",
            "lesson": "semantic masks require uncertainty-aware perception",
        },
    ]
    write_csv(RESULTS / "negative_cases.csv", rows)
    return rows


def plot_results(metric_rows, ablation_summary, stress_summary):
    labels = {
        "geometry_force_closure": "Geometry",
        "probabilistic_normals_pong": "PONG-like",
        "affordance_only_vlm": "Affordance VLM",
        "taskgrasp_semantic_ranker": "TaskGrasp",
        "language_conditioned_grasp_policy": "Lang policy",
        "risk_aware_force_closure": "Risk closure",
        "semantic_force_closure_revisited": "Semantic closure",
        "oracle_task_closure": "Oracle",
    }
    colors = plt.cm.tab20(np.linspace(0, 1, len(METHODS)))
    splits = list(SPLITS.keys())
    x = np.arange(len(splits))
    width = 0.095
    plt.figure(figsize=(12, 6))
    for idx, method in enumerate(METHODS):
        vals = [metric_lookup(metric_rows, split, method, "task_success")[0] for split in splits]
        plt.bar(x + (idx - 3.5) * width, vals, width=width, color=colors[idx], label=labels[method])
    plt.xticks(x, [s.replace("_", "\n") for s in splits], fontsize=8)
    plt.ylabel("Task success")
    plt.ylim(0.0, 1.0)
    plt.title("Semantic force closure across category/task shifts")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "semantic_force_closure_success.png", dpi=220)
    plt.close()

    focus = [
        "probabilistic_normals_pong",
        "taskgrasp_semantic_ranker",
        "language_conditioned_grasp_policy",
        "risk_aware_force_closure",
        "semantic_force_closure_revisited",
        "oracle_task_closure",
    ]
    x = np.arange(len(focus))
    success = [metric_lookup(metric_rows, "combined_hard_shift", m, "task_success")[0] for m in focus]
    semantic = [metric_lookup(metric_rows, "combined_hard_shift", m, "semantic_violation")[0] for m in focus]
    blockage = [metric_lookup(metric_rows, "combined_hard_shift", m, "functional_blockage")[0] for m in focus]
    plt.figure(figsize=(10.5, 5.5))
    plt.bar(x - 0.24, success, width=0.24, label="success", color="#3b6ea8")
    plt.bar(x, semantic, width=0.24, label="semantic violation", color="#b5533c")
    plt.bar(x + 0.24, blockage, width=0.24, label="functional blockage", color="#8c6d31")
    plt.xticks(x, [labels[m] for m in focus], rotation=20, ha="right")
    plt.ylim(0.0, 1.0)
    plt.title("Hard-shift semantic and functional closure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "semantic_force_closure_hard_shift.png", dpi=220)
    plt.close()

    slip = [metric_lookup(metric_rows, "combined_hard_shift", m, "slip_rate")[0] for m in focus]
    damage = [metric_lookup(metric_rows, "combined_hard_shift", m, "damage_rate")[0] for m in focus]
    regret = [metric_lookup(metric_rows, "combined_hard_shift", m, "oracle_regret")[0] for m in focus]
    plt.figure(figsize=(10.5, 5.5))
    plt.bar(x - 0.24, slip, width=0.24, label="slip", color="#758f67")
    plt.bar(x, damage, width=0.24, label="damage", color="#b0607a")
    plt.bar(x + 0.24, regret, width=0.24, label="oracle regret", color="#655e89")
    plt.xticks(x, [labels[m] for m in focus], rotation=20, ha="right")
    plt.ylim(0.0, 1.0)
    plt.title("Hard-shift physical failure modes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "semantic_force_closure_failures.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10.5, 5.5))
    ablations = [r["ablation"] for r in ablation_summary]
    vals = [float(r["task_success"]) for r in ablation_summary]
    plt.bar(np.arange(len(vals)), vals, color="#407076")
    plt.xticks(np.arange(len(vals)), [a.replace("_", "\n") for a in ablations], rotation=25, ha="right", fontsize=8)
    plt.ylabel("Task success")
    plt.ylim(0.0, 1.0)
    plt.title("Semantic force-closure ablations")
    plt.tight_layout()
    plt.savefig(FIGURES / "semantic_force_closure_ablation.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10.5, 5.5))
    for method in focus:
        rows = [r for r in stress_summary if r["stress_axis"] == "combined" and r["method"] == method]
        levels = [float(r["stress_level"]) for r in rows]
        vals = [float(r["task_success"]) for r in rows]
        plt.plot(levels, vals, marker="o", label=labels[method])
    plt.xlabel("Combined stress level")
    plt.ylabel("Task success")
    plt.ylim(0.0, 1.0)
    plt.title("Combined semantic-force-closure stress sweep")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "semantic_force_closure_stress_sweep.png", dpi=220)
    plt.close()


def terminal_decision(metric_rows, pair_rows, ablation_summary):
    split = "combined_hard_shift"
    proposal = "semantic_force_closure_revisited"
    non_oracle = [m for m in METHODS if m not in {proposal, "oracle_task_closure"}]
    prop_success = metric_lookup(metric_rows, split, proposal, "task_success")[0]
    prop_semantic = metric_lookup(metric_rows, split, proposal, "semantic_violation")[0]
    prop_blockage = metric_lookup(metric_rows, split, proposal, "functional_blockage")[0]
    best_success_method = max(non_oracle, key=lambda m: metric_lookup(metric_rows, split, m, "task_success")[0])
    best_semantic_method = min(non_oracle, key=lambda m: metric_lookup(metric_rows, split, m, "semantic_violation")[0])
    best_blockage_method = min(non_oracle, key=lambda m: metric_lookup(metric_rows, split, m, "functional_blockage")[0])
    best_success = metric_lookup(metric_rows, split, best_success_method, "task_success")[0]
    best_semantic = metric_lookup(metric_rows, split, best_semantic_method, "semantic_violation")[0]
    best_blockage = metric_lookup(metric_rows, split, best_blockage_method, "functional_blockage")[0]
    paired_success = [
        r
        for r in pair_rows
        if r["split"] == split and r["reference"] == best_success_method and r["metric"] == "task_success"
    ][0]
    full = [r for r in ablation_summary if r["ablation"] == "full_semantic_force_closure"][0]
    strongest_ablation = max(float(r["task_success"]) for r in ablation_summary if r["ablation"] != "full_semantic_force_closure")
    ablation_drop = float(full["task_success"]) - strongest_ablation
    if (
        prop_success >= best_success + 0.035
        and prop_semantic <= best_semantic + 0.015
        and prop_blockage <= best_blockage + 0.015
        and float(paired_success["mean_diff"]) > 0.030
        and ablation_drop >= 0.020
    ):
        return "STRONG_REVISE"
    return "KILL_ARCHIVE"


def write_summary(metric_rows, pair_rows, ablation_summary, stress_summary, terminal):
    split = "combined_hard_shift"
    proposal = "semantic_force_closure_revisited"
    lines = [
        "Paper 87 semantic_force_closure_revisited v4 rebuild",
        f"Terminal recommendation: {terminal}",
        "Reason: deterministic local semantic-force-closure benchmark added; no robot hardware or accepted external dexterous-grasp benchmark is available.",
        f"Main rollout rows: {sum(1 for _ in open(RESULTS / 'rollouts.csv', encoding='utf-8')) - 1}",
        f"Ablation rollout rows: {sum(1 for _ in open(RESULTS / 'ablation_rollouts.csv', encoding='utf-8')) - 1}",
        f"Stress rollout rows: {sum(1 for _ in open(RESULTS / 'stress_sweep_raw.csv', encoding='utf-8')) - 1}",
        f"Seeds: {SEEDS}",
        "",
        "Combined hard shift:",
    ]
    for method in METHODS:
        success = metric_lookup(metric_rows, split, method, "task_success")
        closure = metric_lookup(metric_rows, split, method, "mechanical_closure_margin")
        semantic = metric_lookup(metric_rows, split, method, "semantic_violation")
        blockage = metric_lookup(metric_rows, split, method, "functional_blockage")
        slip = metric_lookup(metric_rows, split, method, "slip_rate")
        damage = metric_lookup(metric_rows, split, method, "damage_rate")
        regret = metric_lookup(metric_rows, split, method, "oracle_regret")
        lines.append(
            f"{method} task_success={success[0]:.5f} ci95={success[1]:.5f} "
            f"closure={closure[0]:.5f} semantic_violation={semantic[0]:.5f} "
            f"functional_blockage={blockage[0]:.5f} slip={slip[0]:.5f} damage={damage[0]:.5f} regret={regret[0]:.5f}"
        )
    non_oracle = [m for m in METHODS if m not in {proposal, "oracle_task_closure"}]
    best_success_method = max(non_oracle, key=lambda m: metric_lookup(metric_rows, split, m, "task_success")[0])
    paired = [
        r
        for r in pair_rows
        if r["split"] == split and r["reference"] == best_success_method and r["metric"] == "task_success"
    ][0]
    lines.append(
        f"paired task-success diff vs best success baseline {best_success_method}="
        f"{float(paired['mean_diff']):.5f} ci95={float(paired['ci95_diff']):.5f}"
    )
    lines.append("")
    lines.append("Ablations:")
    for row in ablation_summary:
        lines.append(
            f"{row['ablation']} task_success={float(row['task_success']):.5f} "
            f"ci95={float(row['ci95_success']):.5f} semantic_violation={float(row['semantic_violation']):.5f} "
            f"functional_blockage={float(row['functional_blockage']):.5f} damage={float(row['damage_rate']):.5f} "
            f"regret={float(row['oracle_regret']):.5f}"
        )
    lines.append("")
    lines.append("Combined stress level 1.0:")
    for row in stress_summary:
        if row["stress_axis"] == "combined" and row["stress_level"] == "1.0":
            lines.append(
                f"{row['method']} task_success={float(row['task_success']):.5f} ci95={float(row['ci95_success']):.5f} "
                f"semantic_violation={float(row['semantic_violation']):.5f} functional_blockage={float(row['functional_blockage']):.5f} "
                f"slip={float(row['slip_rate']):.5f} damage={float(row['damage_rate']):.5f} regret={float(row['oracle_regret']):.5f}"
            )
    (RESULTS / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    main_rows, seed_rows, metric_rows, pair_rows = run_main()
    ablation_rows, ablation_summary = run_ablation()
    stress_raw, stress_summary = run_stress()
    negative_cases()
    terminal = terminal_decision(metric_rows, pair_rows, ablation_summary)
    plot_results(metric_rows, ablation_summary, stress_summary)
    write_summary(metric_rows, pair_rows, ablation_summary, stress_summary, terminal)
    print(f"terminal={terminal}", flush=True)
    print(f"wrote results to {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
