from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("r0", "r1", "r2", "r3")
SEEDS = (13, 42)
ALIASES = tuple(f"{arm}-seed{seed}" for arm in ARMS for seed in SEEDS)
COARSE = ("equivalent", "compatible", "conflict")
ROUTE_FILES = {
    "natural": "new-heldout-natural.json",
    "counterfactual": "new-heldout-counterfactual.json",
    "old_heldout": "old-heldout.json",
    "composite": "composite.json",
    "new_heldout": "new-heldout.json",
}
EXPECTED_N = {"natural": 288, "counterfactual": 144, "old_heldout": 150,
              "composite": 48, "new_heldout": 432}
EXPECTED_GOLD = {
    "natural": {"equivalent": 96, "compatible": 96, "conflict": 96},
    "counterfactual": {"equivalent": 48, "compatible": 48, "conflict": 48},
    "old_heldout": {"equivalent": 30, "compatible": 90, "conflict": 30},
    "composite": {"equivalent": 12, "compatible": 12, "conflict": 24},
}
DYAD_DENOMINATORS = {
    "both_labels_correct": 72,
    "conflict_dyads_both_labels_correct": 48,
    "conflict_detected_and_other_not_conflict": 48,
    "conflict_side_correct": 48,
    "other_label_correct": 48,
    "other_not_conflict": 48,
    "e_to_p_both_labels_correct": 24,
    "margin_satisfied": 48,
}
OCNLI_EXPECTED = {"selected": 3088, "scored": 1502, "unlabelled": 1586,
                  "neutral": 576, "contradiction": 452, "entailment": 474,
                  "clusters": 1099, "sealed": 0}
TARGET_GAIN = 0.10
INPUTS: dict[str, dict] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return os.path.relpath(path, ROOT).replace("\\", "/")


def record(path: Path) -> None:
    key = relative(path)
    if key not in INPUTS:
        INPUTS[key] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def read_json(path: Path):
    record(path)
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path):
    record(path)
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_denominator(name: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"denominator drift {name}: {actual!r} != {expected!r}")


def publish(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def cell_counts(gold: list[str], pred: list[str]) -> dict[str, int]:
    if len(gold) != len(pred):
        raise ValueError(f"prediction length mismatch: {len(gold)} != {len(pred)}")
    counts = {f"{g}->{p}": 0 for g in COARSE for p in COARSE}
    for actual, predicted in zip(gold, pred):
        key = f"{actual}->{predicted}"
        if key not in counts:
            raise ValueError(f"unknown label pair: {key}")
        counts[key] += 1
    return counts


def error_profile(gold: list[str], pred: list[str], meta: list[dict], expected: dict) -> dict:
    check_denominator("n", len(gold), expected["n"])
    for name in COARSE:
        check_denominator(f"gold:{name}", sum(1 for value in gold if value == name), expected[name])
    counts = cell_counts(gold, pred)
    errors = sum(value for key, value in counts.items() if key.split("->")[0] != key.split("->")[1])
    missed_conflict = counts["conflict->compatible"] + counts["conflict->equivalent"]
    false_equivalence = counts["compatible->equivalent"] + counts["conflict->equivalent"]
    false_conflict = counts["equivalent->conflict"] + counts["compatible->conflict"]
    missed_equivalence = counts["equivalent->compatible"] + counts["equivalent->conflict"]
    recipe_rows = Counter(row["recipe"] for row in meta)

    def by_recipe(mask) -> dict[str, int]:
        table = Counter(row["recipe"] for row, keep in zip(meta, mask) if keep)
        return dict(sorted(table.items()))

    return {
        "errors": errors,
        "cells": counts,
        "missed_conflict": {
            "n": missed_conflict,
            "to_compatible": counts["conflict->compatible"],
            "to_equivalent": counts["conflict->equivalent"],
            "by_recipe": by_recipe(g == "conflict" and p != "conflict" for g, p in zip(gold, pred)),
        },
        "false_equivalence": {
            "n": false_equivalence,
            "from_compatible": counts["compatible->equivalent"],
            "from_conflict": counts["conflict->equivalent"],
            "by_recipe": by_recipe(p == "equivalent" and g != "equivalent" for g, p in zip(gold, pred)),
        },
        "false_conflict": {
            "n": false_conflict,
            "from_equivalent": counts["equivalent->conflict"],
            "from_compatible": counts["compatible->conflict"],
        },
        "missed_equivalence": {
            "n": missed_equivalence,
            "to_compatible": counts["equivalent->compatible"],
            "to_conflict": counts["equivalent->conflict"],
        },
        "recipe_rows": dict(sorted(recipe_rows.items())),
    }


def align_by_id(rows: list[dict], expected_ids: set[str]) -> dict[str, dict]:
    table: dict[str, dict] = {}
    for row in rows:
        if row["id"] in table:
            raise ValueError(f"duplicate prediction id: {row['id']}")
        table[row["id"]] = row
    if set(table) != expected_ids or len(rows) != len(expected_ids):
        missing = sorted(expected_ids - set(table))[:3]
        extra = sorted(set(table) - expected_ids)[:3]
        raise ValueError(f"prediction ids mismatch: missing={missing} extra={extra}")
    return table


def crosses_zero(interval) -> bool:
    return interval is None or interval[0] <= 0 <= interval[1]


def target_feasibility(hit: int, n: int, target_gain: float = TARGET_GAIN) -> dict:
    if n <= 0 or not 0 <= hit <= n:
        raise ValueError(f"invalid conflict recall counts: {hit}/{n}")
    rate = hit / n
    maximum = 1 - rate
    return {
        "baseline": {"hit": hit, "n": n, "rate": rate},
        "max_possible_gain": maximum,
        "target_feasible": bool(maximum >= target_gain),
    }


def selftest() -> None:
    perfect_gold = ["conflict", "compatible", "equivalent"]
    expected = {"n": 3, "equivalent": 1, "compatible": 1, "conflict": 1}
    meta = [{"recipe": "X"}, {"recipe": "Y"}, {"recipe": "Z"}]
    perfect = error_profile(perfect_gold, list(perfect_gold), meta, expected)
    require(perfect["errors"] == 0 and perfect["missed_conflict"]["n"] == 0,
            "perfect predictions must have zero errors")

    reversed_conflict = error_profile(perfect_gold, ["equivalent", "compatible", "equivalent"],
                                      meta, expected)
    require(reversed_conflict["missed_conflict"]["n"] == 1,
            "reversed conflict must be counted as a miss")
    require(reversed_conflict["false_equivalence"]["n"] == 1,
            "conflict->equivalent must be counted as false equivalence")

    try:
        align_by_id([{"id": "a"}, {"id": "a"}], {"a", "b"})
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate/missing ids must be rejected")

    try:
        check_denominator("conflict", 95, 96)
    except ValueError:
        pass
    else:
        raise AssertionError("denominator drift must be rejected")

    try:
        error_profile(["conflict"], ["compatible"], [{"recipe": "X"}],
                      {"n": 2, "equivalent": 0, "compatible": 1, "conflict": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("row-count drift must be rejected")

    infeasible = target_feasibility(90, 96)
    require(abs(infeasible["max_possible_gain"] - 0.0625) < 1e-12
            and infeasible["target_feasible"] is False,
            "90/96 baseline cannot reach a +10pp target")
    feasible = target_feasibility(65, 96)
    require(abs(feasible["max_possible_gain"] - (1 - 65 / 96)) < 1e-12
            and feasible["target_feasible"] is True,
            "65/96 baseline can reach a +10pp target")

    print(json.dumps({"selftest_checks": 7, "passed": 7}))


def build_arm(alias: str, natural_meta: dict[str, dict], cf_meta: dict[str, dict],
              registry: dict[str, dict]):
    arm, seed_text = alias.split("-seed")
    seed = int(seed_text)
    directory = ROOT / "evaluations" / alias
    results = {}
    for name, filename in ROUTE_FILES.items():
        result = read_json(directory / filename)
        require(result["alias"] == alias, f"{alias} alias mismatch in {filename}")
        require(result["checkpoint"] == "terminal-300", f"{alias} checkpoint mismatch")
        check_denominator(f"{alias}:{name}:n", result["n"], EXPECTED_N[name])
        require(len(result["predictions"]) == result["n"], f"{alias}:{name} prediction count")
        diagonal = sum(1 for row in result["predictions"]
                       if row["gold_coarse"] == row["pred_coarse"])
        check_denominator(f"{alias}:{name}:correct", diagonal, result["coarse"]["correct"])
        counts = Counter(row["gold_coarse"] for row in result["predictions"])
        if name in EXPECTED_GOLD:
            for label in COARSE:
                check_denominator(f"{alias}:{name}:gold:{label}", counts[label], EXPECTED_GOLD[name][label])
        registry.setdefault(name, {"ids": {}, "gold": {}})
        registry[name]["ids"][alias] = frozenset(row["id"] for row in result["predictions"])
        registry[name]["gold"][alias] = tuple(
            sorted((row["id"], row["gold_coarse"]) for row in result["predictions"]))
        results[name] = result

    natural = results["natural"]
    coarse = natural["coarse"]
    check_denominator("natural.conflict", coarse["conflict_recall"]["n"], 96)
    check_denominator("natural.non_equivalent", coarse["false_equivalence"]["n"], 192)
    natural_pred = align_by_id(natural["predictions"], set(natural_meta))
    for identifier, row in natural_pred.items():
        require(row["gold_coarse"] == natural_meta[identifier]["coarse_label"],
                f"{alias} natural gold mismatch at {identifier}")
    natural_gold = [natural_meta[identifier]["coarse_label"] for identifier in sorted(natural_meta)]
    natural_labels = [natural_pred[identifier]["pred_coarse"] for identifier in sorted(natural_meta)]
    profile = error_profile(natural_gold, natural_labels,
                            [natural_meta[identifier] for identifier in sorted(natural_meta)],
                            {"n": 288, "equivalent": 96, "compatible": 96, "conflict": 96})
    check_denominator("natural.errors", profile["errors"], 288 - coarse["correct"])
    check_denominator("natural.missed_conflict",
                      profile["missed_conflict"]["n"], 96 - coarse["conflict_recall"]["numerator"])
    check_denominator("natural.false_equivalence",
                      profile["false_equivalence"]["n"], coarse["false_equivalence"]["numerator"])

    counterfactual = results["counterfactual"]
    cf = counterfactual["coarse"]
    check_denominator("cf.conflict", cf["conflict_recall"]["n"], 48)
    check_denominator("cf.non_equivalent", cf["false_equivalence"]["n"], 96)
    dyad = counterfactual["dyad"]
    check_denominator("dyad.n", dyad["n"], 72)
    check_denominator("dyad.conflict_dyads", dyad["conflict_dyads"], 48)
    check_denominator("dyad.e_to_p_dyads", dyad["e_to_p_dyads"], 24)
    for name, denominator in DYAD_DENOMINATORS.items():
        check_denominator(f"dyad.denominator:{name}", dyad["denominators"][name], denominator)
    cf_pred = align_by_id(counterfactual["predictions"], set(cf_meta))
    cf_conflict_hit = sum(1 for row in cf_pred.values()
                          if row["gold_coarse"] == "conflict" and row["pred_coarse"] == "conflict")
    check_denominator("cf.conflict_hit", cf_conflict_hit, cf["conflict_recall"]["numerator"])

    old = results["old_heldout"]["coarse"]
    check_denominator("old.conflict", old["conflict_recall"]["n"], 30)
    check_denominator("old.non_equivalent", old["false_equivalence"]["n"], 120)

    composite = results["composite"]
    table = composite["coarse"]
    check_denominator("composite.conflict", table["conflict_recall"]["n"], 24)
    composite_block = composite["composite"]
    check_denominator("composite.groups", composite_block["groups"], 12)
    check_denominator("composite.conflict_groups", composite_block["conflict_groups"], 12)
    check_denominator("composite.group_rows", composite_block["group_rows"], 48)
    check_denominator("composite.families", len(composite["per_family"]), 12)
    for family, entry in composite["per_family"].items():
        check_denominator(f"composite.family:{family}", entry["n"], 4)

    combined = results["new_heldout"]
    combined_ids = set(row["id"] for row in combined["predictions"])
    require(combined_ids == set(natural_meta) | set(cf_meta),
            f"{alias} combined ids do not equal natural+cf")
    require(combined["dyad"] == counterfactual["dyad"], f"{alias} combined dyad differs from cf")

    reload = read_json(directory / "independent-reload.json")
    require(reload["alias"] == alias and reload["ok"] is True,
            f"{alias} independent reload not ok")
    check_denominator("reload.n", reload["n"], 432)
    check_denominator("reload.argmax_mismatches", reload["argmax_mismatches"], 0)
    require(reload["max_probability_delta"] <= 1e-6, f"{alias} reload probability drift")

    return {
        "alias": alias,
        "arm": arm,
        "seed": seed,
        "checkpoint": "terminal-300",
        "independent_reload": {
            "ok": reload["ok"], "n": reload["n"],
            "max_probability_delta": reload["max_probability_delta"],
            "argmax_mismatches": reload["argmax_mismatches"],
        },
        "fingerprints": {name: results[name]["route_fingerprint"] for name in ROUTE_FILES},
        "natural": {
            "n": natural["n"], "correct": coarse["correct"],
            "macro_f1": coarse["macro_f1"], "accuracy": coarse["accuracy"],
            "conflict_recall": {"hit": coarse["conflict_recall"]["numerator"],
                                "n": coarse["conflict_recall"]["n"],
                                "rate": coarse["conflict_recall"]["value"]},
            "false_equivalence": {"n": coarse["false_equivalence"]["numerator"],
                                  "denominator": coarse["false_equivalence"]["n"],
                                  "rate": coarse["false_equivalence"]["value"]},
            "per_class": {label: {key: coarse["per_class"][label][key]
                                  for key in ("support", "precision", "recall", "f1")}
                          for label in COARSE},
            "confusion_matrix": coarse["confusion_matrix"]["matrix"],
            "error_profile": profile,
        },
        "counterfactual": {
            "n": counterfactual["n"], "correct": cf["correct"],
            "macro_f1": cf["macro_f1"], "accuracy": cf["accuracy"],
            "conflict_recall": {"hit": cf["conflict_recall"]["numerator"],
                                "n": cf["conflict_recall"]["n"],
                                "rate": cf["conflict_recall"]["value"]},
            "false_equivalence": {"n": cf["false_equivalence"]["numerator"],
                                  "denominator": cf["false_equivalence"]["n"],
                                  "rate": cf["false_equivalence"]["value"]},
            "dyad": dyad,
        },
        "old_heldout": {
            "n": results["old_heldout"]["n"], "correct": old["correct"],
            "macro_f1": old["macro_f1"], "accuracy": old["accuracy"],
            "conflict_recall": {"hit": old["conflict_recall"]["numerator"],
                                "n": old["conflict_recall"]["n"],
                                "rate": old["conflict_recall"]["value"]},
            "false_equivalence": {"n": old["false_equivalence"]["numerator"],
                                  "denominator": old["false_equivalence"]["n"],
                                  "rate": old["false_equivalence"]["value"]},
        },
        "composite": {
            "n": composite["n"], "correct": table["correct"],
            "macro_f1": table["macro_f1"], "accuracy": table["accuracy"],
            "conflict_recall": {"hit": table["conflict_recall"]["numerator"],
                                "n": table["conflict_recall"]["n"],
                                "rate": table["conflict_recall"]["value"]},
            "group_table": composite_block,
            "per_family_count": len(composite["per_family"]),
        },
    }, {
        "natural": natural_pred,
        "counterfactual": cf_pred,
    }


def build_ocnli(alias: str) -> dict:
    arm, seed_text = alias.split("-seed")
    seed = int(seed_text)
    if arm == "r0":
        path = ROOT.parent / "baseline-regression" / f"direct3-seed{seed}-terminal-ocnli-diagnostic.json"
    else:
        path = ROOT / "runs" / alias / "scores" / "ocnli-diagnostic-terminal.json"
    data = read_json(path)
    require(data["seed"] == seed, f"{alias} ocnli seed mismatch")
    require(data["checkpoint"] == "terminal-300", f"{alias} ocnli checkpoint mismatch")
    if "run" in data:
        require(data["run"] == alias, f"{alias} ocnli run mismatch")
    require(data.get("not_three_class_NLI_score") is True, f"{alias} ocnli not marked diagnostic")
    accounting = data["accounting"]
    check_denominator("ocnli.selected", accounting["selected_diagnostic"], OCNLI_EXPECTED["selected"])
    check_denominator("ocnli.scored", accounting["scored_with_gold"], OCNLI_EXPECTED["scored"])
    check_denominator("ocnli.unlabelled", accounting["unlabelled_not_scored"], OCNLI_EXPECTED["unlabelled"])
    for label in ("neutral", "contradiction", "entailment"):
        check_denominator(f"ocnli.gold:{label}", accounting["gold_counts"][label], OCNLI_EXPECTED[label])
    check_denominator("ocnli.clusters", accounting["clusters_with_gold"], OCNLI_EXPECTED["clusters"])
    check_denominator("ocnli.sealed", accounting["sealed_exposed_to_model"], OCNLI_EXPECTED["sealed"])
    summary = data["summary"]
    check_denominator("ocnli.n", summary["n"], 1502)
    check_denominator("ocnli.support:conflict", summary["per_class"]["conflict"]["support"], 452)
    check_denominator("ocnli.support:not_conflict", summary["per_class"]["not_conflict"]["support"], 1050)
    require(summary["threshold_fitted"] is False, f"{alias} ocnli threshold fitted")
    predictions = data["predictions"]
    require(len(predictions) == 1502 and len({row["id"] for row in predictions}) == 1502,
            f"{alias} ocnli prediction count/uniqueness")
    correct = sum(1 for row in predictions
                  if (row["predicted_coarse"] == "conflict") == (row["gold_nli"] == "contradiction"))
    require(abs(correct / 1502 - summary["accuracy"]) < 1e-12, f"{alias} ocnli accuracy mismatch")
    interval = summary["accuracy_cluster_ci"]
    return {
        "source_file": relative(path),
        "n": summary["n"],
        "accuracy": summary["accuracy"],
        "accuracy_cluster_ci": {key: interval[key]
                                for key in ("point", "ci", "lo", "hi", "n", "group_clusters",
                                            "weighting", "samples", "valid", "invalid")},
        "macro_f1": summary["macro_f1"],
        "conflict": {key: summary["per_class"]["conflict"][key]
                     for key in ("support", "tp", "fp", "fn", "precision", "recall", "f1")},
        "not_conflict": {key: summary["per_class"]["not_conflict"][key]
                         for key in ("support", "tp", "fp", "fn", "precision", "recall", "f1")},
        "auroc": summary["conflict_probability_auc"],
        "majority_not_conflict_accuracy": summary["majority_not_conflict_accuracy"],
        "threshold_fitted": summary["threshold_fitted"],
        "not_three_class_NLI_score": data["not_three_class_NLI_score"],
        "accounting": {
            "selected_diagnostic": accounting["selected_diagnostic"],
            "scored_with_gold": accounting["scored_with_gold"],
            "unlabelled_not_scored": accounting["unlabelled_not_scored"],
            "gold_counts": accounting["gold_counts"],
            "exclusions": accounting["exclusions"],
            "clusters_with_gold": accounting["clusters_with_gold"],
            "sealed_exposed_to_model": accounting["sealed_exposed_to_model"],
            "selected_input_sha256": accounting["selected_input_sha256"],
        },
    }


def comparison_summary(entry: dict) -> dict:
    metrics = {}
    for metric in ("macro_f1", "accuracy", "conflict_recall", "false_equivalence"):
        family = entry["bootstrap_family"]["metrics"][metric]
        cluster = entry["bootstrap_semantic_cluster"]["metrics"][metric]
        metrics[metric] = {
            "delta": entry["delta"][metric],
            "family_ci95": family["delta_ci95"],
            "family_sd": family["delta_sd"],
            "family_mdes_pp_approx": family["retrospective_mdes_pp_approx"],
            "family_ci_includes_zero": crosses_zero(family["delta_ci95"]),
            "semantic_cluster_ci95": cluster["delta_ci95"],
            "semantic_cluster_sd": cluster["delta_sd"],
            "semantic_cluster_mdes_pp_approx": cluster["retrospective_mdes_pp_approx"],
            "semantic_cluster_ci_includes_zero": crosses_zero(cluster["delta_ci95"]),
            "left_ci95": family["left_ci95"],
            "right_ci95": family["right_ci95"],
            "valid_rounds": family["valid_rounds"],
            "undefined_rounds": family["undefined_rounds"],
        }
    return {
        "n": entry["n"],
        "left_arm": entry["left_arm"],
        "right_arm": entry["right_arm"],
        "contrast": f"{entry['right_arm']}-{entry['left_arm']}",
        "delta": entry["delta"],
        "flips": entry["flips"],
        "metrics": metrics,
        "bootstrap": {
            "family_clusters": entry["bootstrap_family"]["clusters"],
            "semantic_clusters": entry["bootstrap_semantic_cluster"]["clusters"],
            "rounds": entry["bootstrap_family"]["rounds"],
            "seed": entry["bootstrap_family"]["seed"],
            "independent_rows_assumed": entry["bootstrap_family"]["independent_rows_assumed"],
        },
    }


def build_comparisons(fixed: dict, arms: dict, predictions: dict, natural_meta: dict,
                      cf_meta: dict) -> tuple[dict, dict]:
    route_rows = {
        "new-heldout-natural": {row["id"]: row for row in natural_meta.values()},
        "new-heldout-counterfactual": {row["id"]: row for row in cf_meta.values()},
    }
    route_key = {"new-heldout-natural": "natural", "new-heldout-counterfactual": "counterfactual"}
    table: dict[str, dict] = {}
    recipes: dict[str, dict] = {}
    for entry in fixed["comparisons"]:
        route = entry["route"]
        require(route in route_key, f"unexpected comparison route {route}")
        key = route_key[route]
        check_denominator(f"{route}:n", entry["n"], EXPECTED_N[key])
        left_alias = f"{entry['left_arm']}-seed{entry['seed']}"
        right_alias = f"{entry['right_arm']}-seed{entry['seed']}"
        for side, alias in (("left", left_alias), ("right", right_alias)):
            stored = arms[alias][key]
            require(abs(entry[side]["macro_f1"] - stored["macro_f1"]) < 1e-12,
                    f"{route} {alias} macro_f1 mismatch")
            require(abs(entry[side]["accuracy"] - stored["accuracy"]) < 1e-12,
                    f"{route} {alias} accuracy mismatch")
            require(abs(entry[side]["conflict_recall"] - stored["conflict_recall"]["rate"]) < 1e-12,
                    f"{route} {alias} conflict recall mismatch")
            require(abs(entry[side]["false_equivalence"] - stored["false_equivalence"]["rate"]) < 1e-12,
                    f"{route} {alias} false equivalence mismatch")
        for metric in ("macro_f1", "accuracy", "conflict_recall", "false_equivalence"):
            require(abs(entry["delta"][metric]
                        - (entry["right"][metric] - entry["left"][metric])) < 1e-12,
                    f"{route} delta mismatch {metric}")
        rows = route_rows[route]
        require(len(entry["per_example"]) == len(rows), f"{route} per-example count")
        require({item["id"] for item in entry["per_example"]} == set(rows),
                f"{route} per-example ids")
        flips = Counter()
        for item in entry["per_example"]:
            require(item["gold"] == rows[item["id"]]["coarse_label"],
                    f"{route} per-example gold mismatch {item['id']}")
            require(item["left"] == predictions[route][left_alias][item["id"]]["pred_coarse"],
                    f"{route} per-example left mismatch {item['id']}")
            require(item["right"] == predictions[route][right_alias][item["id"]]["pred_coarse"],
                    f"{route} per-example right mismatch {item['id']}")
            flips[item["outcome"]] += 1
        for name, value in entry["flips"].items():
            check_denominator(f"{route}:flips:{name}", flips[name], value)
        pair = f"{entry['left_arm']}-{entry['right_arm']}"
        table.setdefault(route, {}).setdefault(pair, {})[str(entry["seed"])] = comparison_summary(entry)
        if route == "new-heldout-natural" and (entry["left_arm"], entry["right_arm"]) == ("r2", "r3"):
            recipe_rows = Counter(row["recipe"] for row in natural_meta.values())
            by_recipe: dict[str, dict] = {}
            for item in entry["per_example"]:
                recipe = natural_meta[item["id"]]["recipe"]
                bucket = by_recipe.setdefault(recipe, {
                    "rows": recipe_rows[recipe], "r3_new_error": 0,
                    "r3_correction": 0, "both_wrong": 0, "both_correct": 0,
                })
                bucket["both_correct" if item["outcome"] == "both_correct" else
                       "r3_new_error" if item["outcome"] == "left_only_correct" else
                       "r3_correction" if item["outcome"] == "right_only_correct" else
                       "both_wrong"] += 1
            totals = {"both_correct": entry["flips"]["both_correct"],
                      "r3_new_error": entry["flips"]["left_only_correct"],
                      "r3_correction": entry["flips"]["right_only_correct"],
                      "both_wrong": entry["flips"]["both_wrong"]}
            recipes[str(entry["seed"])] = {"by_recipe": dict(sorted(by_recipe.items())),
                                           "totals": totals, "rows": entry["n"]}
    return table, recipes


def build_summary() -> dict:
    heldout = read_jsonl(ROOT / "data/processed/heldout.jsonl")
    manifest = read_json(ROOT / "data/processed/manifest.json")
    require(manifest["fixture"] is False, "fixture data")
    require(manifest["files"]["heldout"]["rows"] == len(heldout) == 432, "heldout row count")
    require(manifest["files"]["heldout"]["sha256"] == INPUTS[relative(ROOT / "data/processed/heldout.jsonl")]["sha256"],
            "heldout sha256 does not match manifest")
    natural_rows = [row for row in heldout if row["phenomena"]["pool"] in ("shared", "control")]
    cf_rows = [row for row in heldout if row["phenomena"]["pool"] == "counterfactual"]
    require(len(natural_rows) == 288, "natural pool count")
    require(len(cf_rows) == 144, "counterfactual pool count")
    natural_meta = {row["id"]: {
        "id": row["id"], "recipe": row["phenomena"]["recipe"],
        "setting": row["phenomena"]["setting"], "mechanism": row["phenomena"]["mechanism"],
        "pool": row["phenomena"]["pool"], "coarse_label": row["coarse_label"],
        "family_id": row["family_id"],
    } for row in natural_rows}
    cf_meta = {row["id"]: {
        "id": row["id"], "recipe": row["phenomena"]["recipe"],
        "setting": row["phenomena"]["setting"], "coarse_label": row["coarse_label"],
        "family_id": row["family_id"],
    } for row in cf_rows}

    arms = {}
    registry: dict[str, dict] = {}
    predictions = {"new-heldout-natural": {}, "new-heldout-counterfactual": {}}
    for alias in ALIASES:
        arm, maps = build_arm(alias, natural_meta, cf_meta, registry)
        arms[alias] = arm
        predictions["new-heldout-natural"][alias] = maps["natural"]
        predictions["new-heldout-counterfactual"][alias] = maps["counterfactual"]

    for route in ROUTE_FILES:
        require(len({arms[alias]["fingerprints"][route] for alias in ALIASES}) == 1,
                f"{route} route fingerprints differ across arms")
        require(len(set(registry[route]["ids"].values())) == 1,
                f"{route} prediction id sets differ across arms")
        require(len(set(registry[route]["gold"].values())) == 1,
                f"{route} gold labels differ across arms")

    for alias in ALIASES:
        arms[alias]["ocnli"] = build_ocnli(alias)

    fixed = read_json(ROOT / "evaluations/fixed-comparisons-20261026.json")
    require(fixed["schema"] == "fixed-terminal-matched-comparison/1", "fixed comparison schema")
    require(fixed["checkpoint"] == "terminal-300", "fixed comparison checkpoint")
    require(len(fixed["comparisons"]) == 20, "fixed comparison count")

    comparison_table, recipes = build_comparisons(fixed, arms, predictions, natural_meta, cf_meta)

    ocnli_shas = {arms[alias]["ocnli"]["accounting"]["selected_input_sha256"] for alias in ALIASES}
    require(len(ocnli_shas) == 1, "ocnli selected inputs differ across arms")

    feasibility = {
        str(seed): target_feasibility(
            arms[f"r0-seed{seed}"]["natural"]["conflict_recall"]["hit"],
            arms[f"r0-seed{seed}"]["natural"]["conflict_recall"]["n"])
        for seed in SEEDS
    }

    summary = {
        "schema": "fixed-results-extract/1",
        "extractor_sha256": sha256_file(Path(__file__)),
        "source": {
            "fixed_comparisons": relative(ROOT / "evaluations/fixed-comparisons-20261026.json"),
            "comparisons_schema": fixed["schema"],
            "comparison_runner_sha256": fixed["runner_sha256"],
            "checkpoint": fixed["checkpoint"],
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "aliases": list(ALIASES),
            "interpretation": fixed["interpretation"],
        },
        "inputs": dict(sorted(INPUTS.items())),
        "integrity": {
            "independent_reload_ok": {alias: arms[alias]["independent_reload"]["ok"] for alias in ALIASES},
            "independent_reload_n": {alias: arms[alias]["independent_reload"]["n"] for alias in ALIASES},
            "n": {"natural": 288, "counterfactual": 144, "old_heldout": 150,
                  "composite": 48, "new_heldout": 432, "ocnli": 1502},
            "gold_counts": EXPECTED_GOLD,
            "id_sets_identical_across_arms": {
                route: len(set(registry[route]["ids"].values())) == 1 for route in ROUTE_FILES},
            "gold_labels_identical_across_arms": {
                route: len(set(registry[route]["gold"].values())) == 1 for route in ROUTE_FILES},
            "route_fingerprints": {route: arms[ALIASES[0]]["fingerprints"][route]
                                   for route in ROUTE_FILES},
            "manifest": {
                "schema": manifest["schema"], "fixture": manifest["fixture"],
                "spec_count": manifest["spec_count"], "spec_sha256": manifest["spec_sha256"],
                "heldout_rows": manifest["files"]["heldout"]["rows"],
                "heldout_sha256": manifest["files"]["heldout"]["sha256"],
                "heldout_sha256_matches": True,
            },
            "heldout": {
                "rows": len(heldout), "natural_rows": len(natural_rows),
                "counterfactual_rows": len(cf_rows),
                "pools": dict(Counter(row["phenomena"]["pool"] for row in heldout)),
                "coarse": dict(Counter(row["coarse_label"] for row in heldout)),
                "families": len({row["family_id"] for row in heldout}),
            },
            "ocnli": {
                "n": 1502, "conflict": 452, "not_conflict": 1050, "clusters": 1099,
                "selected_diagnostic": 3088, "unlabelled_not_scored": 1586,
                "sealed_exposed_to_model": 0, "threshold_fitted": False,
                "not_three_class_NLI_score": True,
                "selected_input_sha256": next(iter(ocnli_shas)),
                "same_inputs_across_arms": True,
            },
        },
        "arms": arms,
        "comparisons": comparison_table,
        "targets": {
            "criterion": "natural：Δconflict_recall >= +0.10 且 Δmacro_f1 >= -0.02 且 Δfalse_equivalence <= +0.02",
            "by_seed": fixed["natural_target_by_seed"],
            "both_seeds_meet_target": fixed["both_seeds_meet_target"],
            "mdes_note": fixed["interpretation"]["negative"],
        },
        "target_feasibility": feasibility,
        "r3_vs_r2_natural_recipes": recipes,
        "notes": [
            "所有 headline 指标逐字取自 evaluations/r*-seed*/ 的现成评分与 fixed-comparisons-20261026.json；"
            "仅自然 288 的错误交叉表由已存 predictions 计算。",
            "本提取不训练、不评分、不调阈值、不重跑 bootstrap，也不改变任何既有产物。",
            "R3-R2 与 R2-R1 的负向区间按原值报告；目标未达不等于无效应（MDES 见 comparisons）。",
            "target_feasibility 仅由 R0 各 seed 既有自然 C 召回分子/分母推导："
            "max_possible_gain = 1 - baseline rate；seed42 最大 +6.25pp，+10pp 目标数学上不可达。"
            "both_seeds_meet_target 保持原样 false，不得当作新臂无效的证据。",
        ],
    }
    return summary


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def signed_pp(value: float) -> str:
    return f"{100 * value:+.2f}pp"


def ci_pp(interval) -> str:
    if interval is None:
        return "—"
    return f"[{100 * interval[0]:+.1f}, {100 * interval[1]:+.1f}]pp"


def ci_raw(interval) -> str:
    if interval is None:
        return "—"
    return f"[{interval[0]:+.3f}, {interval[1]:+.3f}]"


def label(alias: str) -> str:
    arm, seed = alias.split("-seed")
    return f"{arm.upper()} seed{seed}"


def delta_cell(entry: dict, metric: str, as_pp: bool) -> str:
    detail = entry["metrics"][metric]
    value = signed_pp(entry["delta"][metric]) if as_pp else signed(entry["delta"][metric])
    family = ci_pp(detail["family_ci95"]) if as_pp else ci_raw(detail["family_ci95"])
    cluster = ci_pp(detail["semantic_cluster_ci95"]) if as_pp else ci_raw(detail["semantic_cluster_ci95"])
    return f"{value} {family} / {cluster}"


def zero_flags(entry: dict) -> str:
    flags = []
    for metric, short in (("macro_f1", "M"), ("conflict_recall", "C"), ("false_equivalence", "E")):
        detail = entry["metrics"][metric]
        if detail["family_ci_includes_zero"] or detail["semantic_cluster_ci_includes_zero"]:
            flags.append(short)
    return "—" if not flags else ",".join(flags)


def render_markdown(summary: dict) -> str:
    arms = summary["arms"]
    comparisons = summary["comparisons"]
    lines = [
        "# 固定评测结果表（2026-10-26）",
        "",
        "只读提取：数据来自 `evaluations/r*-seed*/` 的现成评分、`runs/*/scores/` 的 OCNLI 诊断、"
        "`evaluations/fixed-comparisons-20261026.json` 与 `data/processed/heldout.jsonl`。"
        "本表不重新训练、不重新评分、不调阈值、不重跑 bootstrap；除自然 288 的错误交叉表外，"
        "所有指标逐字取自既有文件。完整 SHA、全部 20 组比较与逐 recipe 计数见 "
        "`evaluations/summary-20261026.json`。",
        "",
        "- checkpoint：`terminal-300`；seed：13 / 42；8 个模型 = R0 旧 direct3 基线 + R1/R2/R3 三个训练臂 × 两 seed。",
        "- 自然 288：三类各 96；C 召回分母 96，误等价分母 192（gold ≠ equivalent）。",
        "- 反事实 144：三类各 48；dyad 72 对（48 对含 conflict，24 对 E↔P）。",
        "- old heldout 150：compatible 90 / conflict 30 / equivalent 30；C 分母 30，误等价分母 120。"
        "仅为旧集回归，不是新终测。",
        "- composite 48：12 组 × 4 行，12 组全部含 conflict；"
        "组内 4 行全对与两处 conflict 均检出分别计。",
        "- OCNLI 1502：矛盾 452 / 非矛盾 1050，1099 个 cluster；"
        "这是 direct3 迁移诊断，不是三头 NLI 分数，阈值未拟合。",
        "- 区间：family / semantic-cluster 两种 cluster bootstrap（1000 轮，seed 20261025），"
        "均为配对 delta 的 95% CI；`含0` 标记任一区间包含 0。",
        "",
        "## 1. 输入完整性",
        "",
        "- 8/8 `independent-reload.json`：`ok=true`，`argmax_mismatches=0`，`max_probability_delta=0`。",
        "- R0 旧 direct3 基线是之前同 Qwen3-0.6B 的 direct3 训练产物；不是大模型，"
        "不是常量多数类分类器，也不意味着在公开 NLI 上训练过。",
        "- 8/8 模型目录的 6 个评分文件齐全；每个路由的预测 ID 集合与 gold 完全一致"
        "（natural 288、CF 144、old 150、composite 48、combined 432）。",
        "- `data/processed/manifest.json`：`fixture=false`；`heldout.jsonl` 432 行，"
        "SHA256 与 manifest 记录一致；pools 为 shared/control/counterfactual 各 144。",
        "- OCNLI：8 份文件均为 n=1502、clusters=1099、`sealed_exposed_to_model=0`、"
        "`threshold_fitted=false`、`not_three_class_NLI_score=true`；selected-input SHA 全同。",
        "",
        "关键 SHA256：",
        "",
        "| 文件 | SHA256 | 字节 |",
        "|---|---|---|",
    ]
    for key in ("evaluations/fixed-comparisons-20261026.json",
                "data/processed/heldout.jsonl", "data/processed/manifest.json"):
        entry = summary["inputs"][key]
        lines.append(f"| `{key}` | `{entry['sha256']}` | {entry['bytes']} |")

    lines += [
        "",
        "## 2. 自然 heldout 288（主指标）",
        "",
        "| 模型 | macroF1 | 正确/288 | C 召回 | 误等价 | ΔmacroF1 vs R0 | ΔC vs R0 | Δ误E vs R0 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for alias in ALIASES:
        natural = arms[alias]["natural"]
        if alias.startswith("r0"):
            macro_delta = c_delta = e_delta = "—"
        else:
            entry = comparisons["new-heldout-natural"][f"r0-{alias.split('-')[0]}"][str(arms[alias]["seed"])]
            macro_delta = signed(entry["delta"]["macro_f1"])
            c_delta = signed_pp(entry["delta"]["conflict_recall"])
            e_delta = signed_pp(entry["delta"]["false_equivalence"])
        lines.append(
            f"| {label(alias)} | {natural['macro_f1']:.3f} | {natural['correct']}/288 | "
            f"{natural['conflict_recall']['hit']}/{natural['conflict_recall']['n']} | "
            f"{natural['false_equivalence']['n']}/{natural['false_equivalence']['denominator']} | "
            f"{macro_delta} | {c_delta} | {e_delta} |")

    lines += [
        "",
        "## 3. 反事实 heldout 144 与 dyad",
        "",
        "| 模型 | macroF1 | C/48 | dyad 两标签全对/72 | C+非C 两侧/48 | margin 达标/48 | E↔P 双对/24 |",
        "|---|---|---|---|---|---|---|",
    ]
    for alias in ALIASES:
        cf = arms[alias]["counterfactual"]
        dyad = cf["dyad"]
        lines.append(
            f"| {label(alias)} | {cf['macro_f1']:.3f} | "
            f"{cf['conflict_recall']['hit']}/{cf['conflict_recall']['n']} | "
            f"{dyad['both_labels_correct']}/{dyad['n']} | "
            f"{dyad['conflict_detected_and_other_not_conflict']}/{dyad['conflict_dyads']} | "
            f"{dyad['margin_satisfied']}/{dyad['conflict_dyads']} | "
            f"{dyad['e_to_p_both_labels_correct']}/{dyad['e_to_p_dyads']} |")

    lines += [
        "",
        "## 4. old heldout 150（旧集回归）",
        "",
        "| 模型 | 正确/150 | macroF1 | C/30 | 误等价/120 |",
        "|---|---|---|---|---|",
    ]
    for alias in ALIASES:
        old = arms[alias]["old_heldout"]
        lines.append(
            f"| {label(alias)} | {old['correct']}/150 | {old['macro_f1']:.3f} | "
            f"{old['conflict_recall']['hit']}/{old['conflict_recall']['n']} | "
            f"{old['false_equivalence']['n']}/{old['false_equivalence']['denominator']} |")

    lines += [
        "",
        "## 5. composite 48",
        "",
        "| 模型 | macroF1 | 正确/48 | C/24 | 四对全对/12 | 两冲突均检出/12 |",
        "|---|---|---|---|---|---|",
    ]
    for alias in ALIASES:
        composite = arms[alias]["composite"]
        group = composite["group_table"]
        lines.append(
            f"| {label(alias)} | {composite['macro_f1']:.3f} | {composite['correct']}/48 | "
            f"{composite['conflict_recall']['hit']}/{composite['conflict_recall']['n']} | "
            f"{group['all_correct']}/{group['groups']} | "
            f"{group['conflict_groups_detected']}/{group['conflict_groups']} |")

    lines += [
        "",
        "## 6. OCNLI 迁移诊断（1502 行）",
        "",
        "| 模型 | 矛盾召回/452 | precision | binaryF1（conflict） | AUROC | accuracy（95% CI） |",
        "|---|---|---|---|---|---|",
    ]
    for alias in ALIASES:
        ocnli = arms[alias]["ocnli"]
        interval = ocnli["accuracy_cluster_ci"]
        lines.append(
            f"| {label(alias)} | {int(ocnli['conflict']['tp'])}/452 | "
            f"{ocnli['conflict']['precision']:.3f} | {ocnli['conflict']['f1']:.3f} | "
            f"{ocnli['auroc']:.3f} | {ocnli['accuracy']:.3f} "
            f"[{interval['lo']:.3f}, {interval['hi']:.3f}] |")

    lines += [
        "",
        "## 7. 现成配对比较（R3−R2、R2−R1）",
        "",
        "每格为 `delta：family 95CI / semantic-cluster 95CI`。`flips` 为 `左对/右对/同对/同错`，"
        "左/右指配对中的左臂/右臂。",
        "",
        "R3−R2：匹配数据、初始化与训练窗口，只加 margin 目标。"
        "R2−R1：合并数据组织、语义新颖性与细粒度组合效应。",
        "",
        "| 路由 | 配对 | seed | ΔmacroF1 | ΔC | Δ误E | flips | CI 含0 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for route, route_name in (("new-heldout-natural", "自然 288"),
                              ("new-heldout-counterfactual", "反事实 144")):
        for pair, pair_name in (("r2-r3", "R3−R2"), ("r1-r2", "R2−R1")):
            for seed in SEEDS:
                entry = comparisons[route][pair][str(seed)]
                lines.append(
                    f"| {route_name} | {pair_name} | {seed} | "
                    f"{delta_cell(entry, 'macro_f1', False)} | "
                    f"{delta_cell(entry, 'conflict_recall', True)} | "
                    f"{delta_cell(entry, 'false_equivalence', True)} | "
                    f"{entry['flips']['left_only_correct']}/{entry['flips']['right_only_correct']}/"
                    f"{entry['flips']['both_correct']}/{entry['flips']['both_wrong']} | "
                    f"{zero_flags(entry)} |")

    lines += [
        "",
        "## 8. R0→各新臂的预设目标判据（自然 288）",
        "",
        "判据：Δconflict_recall ≥ +10pp 且 ΔmacroF1 ≥ −2pp 且 Δfalse_equivalence ≤ +2pp。"
        "按 seed 判定，不平均。"
        "本节表内均为 family bootstrap 95% CI；semantic-cluster 完整区间见 JSON 对应字段"
        "（`evaluations/summary-20261026.json`）。",
        "",
        "| 臂 | seed | ΔC | ΔmacroF1 | Δ误E | 该 seed 达标 |",
        "|---|---|---|---|---|---|",
    ]
    for arm in ("r1", "r2", "r3"):
        for seed in SEEDS:
            alias = f"{arm}-seed{seed}"
            entry = comparisons["new-heldout-natural"][f"r0-{arm}"][str(seed)]
            target = summary["targets"]["by_seed"][alias]
            lines.append(
                f"| {arm.upper()} | {seed} | {signed_pp(entry['delta']['conflict_recall'])} "
                f"{ci_pp(entry['metrics']['conflict_recall']['family_ci95'])} | "
                f"{signed(entry['delta']['macro_f1'])} "
                f"{ci_raw(entry['metrics']['macro_f1']['family_ci95'])} | "
                f"{signed_pp(entry['delta']['false_equivalence'])} "
                f"{ci_pp(entry['metrics']['false_equivalence']['family_ci95'])} | "
                f"{'是' if target['meets_prespecified_target'] else '否'} |")

    both = summary["targets"]["both_seeds_meet_target"]
    lines += [
        "",
        "### 目标天花板（只由既有 R0 数值推导）",
        "",
        "| seed | R0 自然 C 召回 | 最大可能增益 | +10pp 目标可行 |",
        "|---|---|---|---|",
    ]
    for seed in SEEDS:
        item = summary["target_feasibility"][str(seed)]
        baseline = item["baseline"]
        lines.append(
            f"| {seed} | {baseline['hit']}/{baseline['n']}（{pct(baseline['rate'], 2)}） | "
            f"+{100 * item['max_possible_gain']:.2f}pp | "
            f"{'是' if item['target_feasible'] else '否'} |")
    lines += [
        "",
        "seed42 的 R0 自然 C 已达 90/96=93.75%，即使新臂完美 96/96，最大增益也只有 +6.25pp，"
        "低于 +10pp；因此“两 seed 均 +10pp”在 seed42 数学上不可达。"
        "seed13 为 65/96，最大 +32.29pp，目标可行。"
        "这里保留原有目标和 `both_seeds_meet_target=false`，不倒改阈值追分；"
        "统一 false 不能当作新臂无效的证据。",
        "",
        "`both_seeds_meet_target`："
        + "；".join(f"{arm.upper()}={both[arm]}" for arm in ("r1", "r2", "r3")) + "。",
        "",
        "## 9. 自然 288 错误分布（由已存 predictions 交叉表）",
        "",
        "计数为非互斥行为视角：C→E 同时计入“漏 C”和“误 E”。"
        "`→P`=判成 compatible，`→E`=判成 equivalent，`→C`=判成 conflict。",
        "",
        "| 模型 | 错误/288 | 漏 C（→P / →E） | 误 E（P→E / C→E） | 误 C（E→C / P→C） | 漏 E（→P / →C） |",
        "|---|---|---|---|---|---|",
    ]
    for alias in ALIASES:
        profile = arms[alias]["natural"]["error_profile"]
        missed = profile["missed_conflict"]
        false_e = profile["false_equivalence"]
        false_c = profile["false_conflict"]
        missed_e = profile["missed_equivalence"]
        lines.append(
            f"| {label(alias)} | {profile['errors']}/288 | "
            f"{missed['n']}（{missed['to_compatible']} / {missed['to_equivalent']}） | "
            f"{false_e['n']}（{false_e['from_compatible']} / {false_e['from_conflict']}） | "
            f"{false_c['n']}（{false_c['from_equivalent']} / {false_c['from_compatible']}） | "
            f"{missed_e['n']}（{missed_e['to_compatible']} / {missed_e['to_conflict']}） |")

    lines += [
        "",
        "### R3 相对 R2 的新错与纠错（自然 288，按 recipe）",
        "",
        "新错 = R2 对、R3 错（`left_only_correct`）；纠错 = R2 错、R3 对（`right_only_correct`）。"
        "只列至少发生一次变化的 recipe；完整 19 个 recipe 的计数见 JSON。",
        "",
        "| recipe | 自然行数 | seed13 新错 | seed13 纠错 | seed42 新错 | seed42 纠错 |",
        "|---|---|---|---|---|---|",
    ]
    recipe_names = sorted({name for seed in ("13", "42")
                           for name, entry in summary["r3_vs_r2_natural_recipes"][seed]["by_recipe"].items()
                           if entry["r3_new_error"] or entry["r3_correction"]})
    for name in recipe_names:
        seed13 = summary["r3_vs_r2_natural_recipes"]["13"]["by_recipe"].get(
            name, {"rows": 0, "r3_new_error": 0, "r3_correction": 0})
        seed42 = summary["r3_vs_r2_natural_recipes"]["42"]["by_recipe"].get(
            name, {"rows": 0, "r3_new_error": 0, "r3_correction": 0})
        lines.append(
            f"| {name} | {seed13['rows']} | {seed13['r3_new_error']} | {seed13['r3_correction']} | "
            f"{seed42['r3_new_error']} | {seed42['r3_correction']} |")
    totals13 = summary["r3_vs_r2_natural_recipes"]["13"]["totals"]
    totals42 = summary["r3_vs_r2_natural_recipes"]["42"]["totals"]
    lines.append(
        f"| 合计 | 288 | {totals13['r3_new_error']} | {totals13['r3_correction']} | "
        f"{totals42['r3_new_error']} | {totals42['r3_correction']} |")

    lines += [
        "",
        "## 10. 口径与限制",
        "",
        "- 本表是只读提取；headline 指标均来自既有评分文件，未重造统计量。"
        "自然错误交叉表与 R3−R2 recipe 计数由既有 predictions / per-example 加 `processed/heldout.jsonl` 的 "
        "`phenomena.recipe` 连接得到。",
        "- 不把两 seed 平均：R0 自身 seed13 与 seed42 差异很大（自然 macroF1 0.817 vs 0.928），"
        "汇总均值会掩盖 seed 变异。",
        "- 预设目标只适用于自然 288 的 R0→新臂：seed13 的 R1/R2/R3 均达标，seed42 的 R1/R2/R3 均未达标，"
        "`both_seeds_meet_target` 三者全为 false。但 seed42 的 R0 自然 C 已达 90/96=93.75%，"
        "即使完美也只可能 +6.25pp，低于 +10pp；该 seed 的“两 seed 均 +10pp”在数学上不可达。"
        "因此统一 false 不能当作新臂无效的证据，也不回改阈值追分。"
        "未达目标同样不能写成“无效”：例如 R0→R3 seed42 的 ΔC 为 −8.33pp，"
        "family 回溯 MDES≈16.0pp、semantic-cluster≈18.3pp，均大于 10pp 目标，不能排除更小效应。",
        "- R3−R2 与 R2−R1 的负向 delta 及其不含 0 的区间按原值报告；本表不作上线决定，"
        "也不作最终优劣结论。",
        "- OCNLI 是从 3088 条候选中实际评分 1502 条的迁移诊断（1586 条无标注未评分），"
        "矛盾 452 条；不是三头 NLI 准确率，阈值未拟合，也未用 OCNLI 调参。",
        "- composite 的 12 组全部含 conflict：always-escalate 基线在“冲突组检出”上天然为 12/12，"
        "但会误判组内非 conflict 行，故该列不能单独解释为路由成功率或用户效用；"
        "同时看“四对全对/12”。",
        "- 错误分布中的 recipe 是受控设计因子，不是真实用户机制；只描述本固定集的失败形态，"
        "不外推生产分布。",
        "- 旧 heldout 150 只作锁后回归，不是新终测；不得与自然 288 混为同一主指标。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return 0
    summary_path = ROOT / "evaluations/summary-20261026.json"
    markdown_path = ROOT / "evaluations/结果表-2026-10-26.md"
    for path in (summary_path, markdown_path):
        if path.exists():
            raise FileExistsError(f"preserve existing output: {path}")
    summary = build_summary()
    markdown = render_markdown(summary)
    payload = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    publish(summary_path, payload)
    publish(markdown_path, markdown)
    print(json.dumps({
        "summary": relative(summary_path),
        "markdown": relative(markdown_path),
        "summary_bytes": len(payload.encode("utf-8")),
        "markdown_bytes": len(markdown.encode("utf-8")),
        "arms": len(ALIASES),
        "comparisons": sum(len(seeds) for route in summary["comparisons"].values()
                           for seeds in route.values()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
