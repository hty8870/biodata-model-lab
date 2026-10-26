from __future__ import annotations

import argparse
import json
from pathlib import Path

import conflict_common as K
import score_conflict as S

ROOT = Path(__file__).resolve().parent
ROUTES = ("new-heldout-natural", "new-heldout-counterfactual")
PAIRS = (("r0", "r1"), ("r0", "r2"), ("r0", "r3"), ("r1", "r2"), ("r2", "r3"))


def meets_target(delta):
    keys = ("conflict_recall", "macro_f1", "false_equivalence")
    if any(delta[k] is None for k in keys):
        return None
    return delta["conflict_recall"] >= .10 and delta["macro_f1"] >= -.02 and delta["false_equivalence"] <= .02


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        good = {"conflict_recall": .11, "macro_f1": -.01, "false_equivalence": .01}
        assert meets_target(good) is True
        assert meets_target(dict(good, conflict_recall=-.11)) is False
        assert meets_target(dict(good, false_equivalence=.03)) is False
        assert meets_target(dict(good, conflict_recall=None)) is None
        print(json.dumps({"selftest_checks": 4, "passed": 4}))
        return
    closure = K.read_json(ROOT / "audits/semantic-review-final-20261026.json")
    assert closure["status"] == "accepted_for_fixed_training"
    assert closure["processed_sha256"] == {s: K.sha256_file(K.PROCESSED_DIR / f"{s}.jsonl") for s in K.SPLITS}
    specs = K.read_jsonl(K.SPECS_PATH)
    comparisons, criteria = [], {}
    for seed in K.SEEDS:
        lock = K.load_lock(K.RUNS_DIR / f"final-lock-r1-seed{seed}-terminal.json")
        scored = {}
        for arm in K.SCORE_ROUTES:
            alias = f"{arm}-seed{seed}"
            directory = ROOT / "evaluations" / alias
            assert K.read_json(directory / "independent-reload.json")["ok"] is True
            scored[arm] = {route: K.read_json(directory / f"{route}.json") for route in ROUTES}
            for route, result in scored[arm].items():
                assert result["alias"] == alias and result["checkpoint"] == "terminal-300"
                assert result["route_fingerprint"] == K.route_fingerprint_any(route)[0]
        for route in ROUTES:
            K.check_lock(lock, route=route)
            rows = S.load_rows_for_scoring(route, lock=lock)
            for left, right in PAIRS:
                comparison = S.compare_runs(left=scored[left][route], right=scored[right][route],
                                            rows=rows, rounds=1000, specs=specs)
                comparison.update(seed=seed, route=route, left_arm=left, right_arm=right)
                comparisons.append(comparison)
                if route == "new-heldout-natural" and left == "r0":
                    criteria[f"{right}-seed{seed}"] = {"delta": comparison["delta"],
                                                       "meets_prespecified_target": meets_target(comparison["delta"])}
    result = {"schema": "fixed-terminal-matched-comparison/1", "checkpoint": "terminal-300",
              "comparisons": comparisons, "natural_target_by_seed": criteria,
              "both_seeds_meet_target": {arm: all(criteria[f"{arm}-seed{s}"]["meets_prespecified_target"] is True
                                                   for s in K.SEEDS) for arm in K.TRAIN_ROUTES},
              "interpretation": {"r3-r2": "matched data, initialization and windows; added margin objective",
                                 "r2-r1": "combined data organization, semantic novelty and fine-composition effect",
                                 "negative": "Missing the target is not evidence of no effect when MDES exceeds 10pp"},
              "runner_sha256": K.sha256_file(Path(__file__))}
    path = ROOT / "evaluations/fixed-comparisons-20261026.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(path), "criteria": criteria,
                      "both_seeds_meet_target": result["both_seeds_meet_target"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
