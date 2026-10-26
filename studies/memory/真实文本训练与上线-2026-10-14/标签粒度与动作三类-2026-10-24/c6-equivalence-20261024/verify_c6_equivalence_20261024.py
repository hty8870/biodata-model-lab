from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def softmax(values):
    exp = np.exp(values - values.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def numpy_reference():
    rng = np.random.default_rng(731)
    features = rng.normal(size=(12, 384))
    w3, b3 = rng.normal(0, .02, (3, 384)), rng.normal(0, .02, 3)
    parents = np.array([0, 0, 0, 0, 1, 2])
    w6, b6 = w3[parents].copy(), b3[parents].copy()
    b6[:4] -= np.log(4)
    p3, p6 = softmax(features @ w3.T + b3), softmax(features @ w6.T + b6)
    parent6 = np.column_stack((p6[:, :4].sum(axis=1), p6[:, 4:]))
    labels = np.arange(len(features)) % 3
    g3 = (p3 - np.eye(3)[labels]) / len(features)
    belongs = parents[None, :] == labels[:, None]
    g6 = (p6 - belongs * p6 / parent6[np.arange(len(features)), labels, None]) / len(features)
    fw3, fw6 = g3.T @ features, g6.T @ features
    fb3, fb6 = g3.sum(axis=0), g6.sum(axis=0)
    error = {"probabilities": float(np.max(np.abs(p3 - parent6))),
             "feature_gradient": float(np.max(np.abs(g3 @ w3 - g6 @ w6))),
             "coexist_weight_gradient_times4": float(np.max(np.abs(fw3[0] - 4 * fw6[0]))),
             "coexist_bias_gradient_times4": float(abs(fb3[0] - 4 * fb6[0]))}
    assert max(error.values()) < 1e-12, error
    b6[:4] += np.log(4)
    wrong = softmax(features @ w6.T + b6)
    wrong_parent = np.column_stack((wrong[:, :4].sum(axis=1), wrong[:, 4:]))
    wrong_error = float(np.max(np.abs(p3 - wrong_parent)))
    assert wrong_error > .1
    return {"errors": error, "missing_log4_negative_control": wrong_error, "passed": True}


def main():
    original = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    first = json.loads((ROOT / "results-v2.json").read_text(encoding="utf-8"))
    replay = json.loads((ROOT / "replay-v2.json").read_text(encoding="utf-8"))
    assert original["script_sha256"] == sha(ROOT / "initial_implementation_20261024.py")
    assert first["script_sha256"] == replay["script_sha256"] == sha(ROOT / "check_c6_equivalence_20261024.py")
    comparison_first, comparison_replay = dict(first), dict(replay)
    comparison_first.pop("seconds")
    comparison_replay.pop("seconds")
    assert comparison_first == comparison_replay
    assert (ROOT / "results-v2.steps.jsonl").read_bytes() == (ROOT / "replay-v2.steps.jsonl").read_bytes()
    rows = [json.loads(line) for line in (ROOT / "results-v2.steps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 540
    for name, summary in first["cases"].items():
        selected = [row for row in rows if row["case"] == name]
        assert len(selected) == 60 and [row["step"] for row in selected] == list(range(1, 61))
        assert summary["first_update"] == selected[0] and summary["last_update"] == selected[-1]
        for key, value in summary["max_over_updates"].items():
            assert value == max(row[key] for row in selected), (name, key)
        maximum = summary["clip_norm"]
        if maximum is not None:
            assert all(row[f"{arm}_clipped_norm"] <= maximum + 1e-6 for row in selected for arm in ("a3", "c6"))
            assert summary["clipped_updates"] == {arm: sum(row[f"{arm}_gradient_norm"] > maximum for row in selected) for arm in ("a3", "c6")}
    historical = {}
    for name in ("a3-seed13", "a3-seed20260919"):
        path = ROOT.parent / "runs" / name / "updates.jsonl"
        updates = [row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if row["event"] == "update"]
        assert len(updates) == 300
        values = [row["gradient_norm"] for row in updates]
        historical[name] = {"updates": len(updates), "clip_active": sum(row["gradient_norm"] > row["clip_grad_norm"] for row in updates),
                            "gradient_norm_min": min(values), "gradient_norm_median": median(values), "gradient_norm_max": max(values),
                            "source_sha256": sha(path), "historical_cuda_logs_only": True}
    clipping = [row for row in rows if row["case"] == "adamw_actual_clip1"]
    ratios = [(row["c6_clipped_norm"] / row["c6_gradient_norm"]) /
              (row["a3_clipped_norm"] / row["a3_gradient_norm"]) for row in clipping]
    assert max(ratios) - min(ratios) > 1e-4
    clip_diagnostic = {"multiplier_ratio_min": min(ratios), "multiplier_ratio_max": max(ratios),
                       "multiplier_ratio_first": ratios[0], "multiplier_ratio_last": ratios[-1],
                       "first_step_probability_error": clipping[0]["probability_error"],
                       "last_step_probability_error": clipping[-1]["probability_error"]}
    for path in ROOT.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    result = {"status": "passed", "numpy_independent_reference": numpy_reference(), "replay_equal_excluding_time": True,
              "trajectory_bytes_equal": True, "recomputed_trajectory_rows": len(rows), "historical_a3_clip_diagnostic": historical,
              "clip_multiplier_diagnostic": clip_diagnostic,
              "result_sha256": sha(ROOT / "results-v2.json"), "replay_sha256": sha(ROOT / "replay-v2.json"),
              "verification_script_sha256": sha(Path(__file__))}
    (ROOT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
