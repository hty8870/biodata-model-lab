"""只从本次真概率/原回放工件汇总并核验，不调用模型，不重评。"""
from collections import Counter
import json
import math
from pathlib import Path
import numpy as np

import binding as B
import formal_evaluation as R

BASE = B.BASE
REPORTS = BASE / "reports"


def rates(view):
    equal = view["root_equal_primary_known_reference"]
    row = view["row_secondary_known_reference"]
    return {"known_reference_n": view["n_reference"], "n_planned": view["n_planned"],
            "known_reference_roots": view["reference_roots"],
            "root_agreement": equal["agreement"], "root_macro_f1": equal["macro_f1"],
            "root_agreement_ci": equal["bootstrap"]["agreement"]["ci"],
            "root_macro_f1_ci": equal["bootstrap"]["macro_f1"]["ci"],
            "row_agreement": row["agreement"], "row_macro_f1": row["macro_f1"],
            "confirmed_correct": int(row["confirmed_correct_mass"]),
            "completion": view["completion"], "confirmed_correct_overall": view["confirmed_correct_overall"],
            "reference_coverage": view["reference_coverage"], "support": row["support"],
            "confusion": row["confusion"], "root_support": equal["support"],
            "root_confusion": equal["confusion"]}


def root_refs(rows, intervals):
    result = {}
    for source in ("raw", "gated"):
        result[source] = {}
        for role in ("related_ids", "evidence_refs"):
            selected = [r for r in rows if r.get("refs_by_role") and r["refs_by_role"].get(source)]
            weights = R.EM._group_weights([r["group_id"] for r in selected])
            counts = {key: 0.0 for key in ("hits", "predicted", "expected", "unknown", "known_non_expected", "missed_expected")}
            for row, weight in zip(selected, weights):
                score = row["refs_by_role"][source][role]
                counts["hits"] += float(weight) * score["hits"]
                for name, field in (("predicted", "predicted"), ("expected", "expected"), ("unknown", "unknown_ids"),
                                    ("known_non_expected", "known_non_expected"), ("missed_expected", "missed_expected")):
                    counts[name] += float(weight) * len(score[field])
            precision = counts["hits"] / counts["predicted"] if counts["predicted"] else None
            recall = counts["hits"] / counts["expected"] if counts["expected"] else None
            for name, point in (("precision", precision), ("recall", recall)):
                original = intervals[source][role][name]["point"]
                assert point is None and original is None or point is not None and math.isclose(point, original, abs_tol=1e-10)
            result[source][role] = {"weighting": "root_equal", "rows": len(selected),
                                   "roots": len({r["group_id"] for r in selected}), "weighted_counts": counts,
                                   "precision": {"numerator": counts["hits"], "denominator": counts["predicted"],
                                                 "value": precision, "ci": intervals[source][role]["precision"]["ci"]},
                                   "recall": {"numerator": counts["hits"], "denominator": counts["expected"],
                                              "value": recall, "ci": intervals[source][role]["recall"]["ci"]}}
    return result


def main():
    binding = B.verify()
    terminal = B.read_json(REPORTS / "terminal-result.json")
    raw = B.read_jsonl(REPORTS / "terminal-probabilities.jsonl")
    replay = B.read_jsonl(REPORTS / "heldout.predictions.jsonl") + B.read_jsonl(REPORTS / "ood.predictions.jsonl") + B.read_jsonl(REPORTS / "unlabelled.predictions.jsonl")
    cache = B.read_jsonl(REPORTS / "cache-entries.jsonl")
    expected = B.read_jsonl(B.REF / "expected-cases.jsonl")
    expected_ids = {str(r["source_id"]) for r in expected}
    assert len(raw) == len(replay) == len(expected) == len(expected_ids) == 872
    assert {r["id"] for r in raw} == {str(r["row_id"]) for r in replay} == expected_ids
    assert Counter(r["split"] for r in raw) == {"heldout": 445, "ood": 427}
    assert sum(r["reference_label"] in R.CLASSES for r in raw) == 871
    assert sum(r["reference_label"] == "uncertain" for r in raw) == 98
    matrix = R.A._matrix([r["probabilities"] for r in raw], 872, 7)
    assert matrix is not None
    assert terminal["cache"]["model_identity"] == R.identity(binding)
    assert terminal["cache"]["replay_physical_predictions"] == 0
    assert len({r["cache_key"] for r in cache}) == len(cache)
    cached = {r["cache_key"]: r for r in cache}
    for row in raw:
        assert row["model_identity"] == R.identity(binding)
        assert np.allclose(cached[row["cache_key"]]["probabilities"], row["probabilities"], rtol=0, atol=1e-6)
    unknown_raw = next(r for r in raw if r["id"] == "18014002")
    unknown_gated = next(r for r in replay if str(r["row_id"]) == "18014002")
    assert unknown_raw["reference_label"] is unknown_gated["label"] is None
    assert unknown_raw["completed"]
    summary = {"formal_attempt_once": True, "model_prediction_calls_in_this_summary": 0,
               "source_protected_hashes_unchanged": True, "protected_files": len(binding["protected_sha256"]),
               "input_registry_872_exact": True, "human_gold": False,
               "single_model_dir": str(B.MODEL), "model_weights_sha256": B.sha(B.MODEL / "model.pt"),
               "model_meta_sha256": B.sha(B.MODEL / "meta.json"), "lock_sha256": B.sha(B.LOCK),
               "temperature": binding["temperature"], "threshold": None, "risk_certificate": False,
               "classes": list(R.CLASSES), "confusion_outcomes": [*R.CLASSES, "system_failure"],
               "terminal": {}}
    by_id = {str(r["row_id"]): r for r in replay}
    for split, report in terminal["splits"].items():
        selected = [by_id[row["id"]] for row in raw if split == "all" or row["split"] == split]
        summary["terminal"][split] = {
            "planned": report["expected"], "roots": report["roots"],
            "views": {name: rates(view) for name, view in report["views"].items()},
            "completion": report["all_attempt"]["completion"],
            "correct_completed_overall": report["all_attempt"]["correct_completed"],
            "semantic_uncertain": report["semantic_uncertain"], "system_failure": report["system_failure"],
            "failure_causes": report["failure_causes"], "row_operation": report["all_attempt"]["operation"],
            "root_operation": report["group_equal"]["operation"], "refs_by_role_row": report["refs_by_role"],
            "refs_by_role_root": root_refs(selected, report["refs_root_equal"])}
    summary["excluded_no_truth"] = {"id": "18014002", "status": "excluded", "truth": None,
        "raw_classifier": unknown_raw["classifier_relation"], "pair_gate": unknown_raw["pair_relation"],
        "gated_relation": unknown_gated["decision"]["relation"], "gated_completed": unknown_gated["completed"],
        "error_code": unknown_gated["error"], "correctness": None,
        "note": "原作者标签未用作gold；reference缺口不等于预测错误或semantic unknown。"}
    old = B.read_json(REPORTS / "old-exposed-set-result.json")
    old_rows = B.read_jsonl(REPORTS / "old-exposed-set.predictions.jsonl")
    assert len(old_rows) == old["summary"]["all_attempt"]["n"] == 37
    old_by_count = {}
    for count in (0, 1, 8, 64, 65, 129):
        subset = [r for r in old_rows if r["rules"] == count]
        old_by_count[str(count)] = {"planned": len(subset), "completed": sum(r["completed"] for r in subset),
            "correct_completed": sum(r["completed"] and r["primary_agree"] for r in subset),
            "semantic_uncertain": sum(r["semantic_uncertain"] for r in subset),
            "system_failure_causes": dict(Counter(r["error"] or r["row_error"] for r in subset if not r["completed"])),
            "predictor_call_counts": dict(Counter(r["calls"] for r in subset)),
            "pair_attempts": sum(r["classifier_pair_attempts"] for r in subset),
            "oracle_available": sum(r["oracle"]["available"] for r in subset),
            "oracle_correct": sum(r["oracle"].get("primary_agree", False) for r in subset)}
    pairs = []
    for row in old_rows:
        for pair in row["pair_diagnostics"]:
            pairs.append({"group_id": row["group_id"], "reference_label": pair["label"],
                          "classifier_relation": pair["classifier_relation"], "pair_relation": pair["pair_relation"],
                          "completed": True})
    old_pair_views = {}
    for name, field in (("classifier", "classifier_relation"), ("pair_gate", "pair_relation")):
        view = R.view_report([{**pair, "relation": pair[field]} for pair in pairs])
        old_pair_views[name] = rates(view)
    summary["old_set"] = {"input_role": old["input_role"], "planned": 37, "roots": 35,
        "completion": old["summary"]["all_attempt"]["completion"],
        "correct_completed_overall": old["summary"]["all_attempt"]["correct_completed"],
        "semantic_uncertain": old["summary"]["semantic_uncertain_rows"],
        "system_failure": old["summary"]["system_failure_rows"],
        "failure_codes": old["summary"]["errors"], "gated_views": rates(old["known_set_views"]),
        "raw_confusion": old["summary"]["raw_set"]["confusion"],
        "gated_confusion": old["summary"]["gated_set"]["confusion"],
        "pair_views": old_pair_views, "observed_pair_outputs": len(pairs),
        "pairs_without_reference": sum(p["reference_label"] not in R.CLASSES for p in pairs),
        "refs_by_role_row": old["summary"]["refs_by_role"],
        "refs_by_role_root": root_refs(old_rows, old["refs_root_equal"]),
        "oracle_aggregation": old["summary"]["oracle_aggregation"],
        "by_rule_count": old_by_count,
        "physical_predictions": old["physical_predictions"], "cache_hits": old["cache_hits"],
        "predictor_calls": old["summary"]["predictor_calls"], "scored_pairs": old["summary"]["scored_pairs"]}
    structural = B.read_json(REPORTS / "structural-boundary-result.json")
    summary["structural_fixtures"] = [{"rules": r["rules"], "long_rules": r["long_rules"],
                                        "batch_sizes": [batch["rules"] for batch in r["batches"]],
                                        "observed_pairs": r["observed_pairs"], "decision": r["decision"]["relation"],
                                        "context_batches": [b["context_batch"] for b in r["batches"]]}
                                       for r in structural["rows"]]
    summary["physical_prediction_counts"] = B.read_json(REPORTS / "execution-result.json")["physical_predictions"]
    summary["runtime"] = {key: B.read_json(REPORTS / "runtime-final.json")[key] for key in
                          ("loaded_device", "model_training", "hidden_size", "max_length", "parameter_count",
                           "peak_allocated_mb", "peak_reserved_mb", "loader_seconds")}
    B.save_json(REPORTS / "verified-summary.json", summary)
    output_files = sorted(p for p in REPORTS.glob("*") if p.is_file()) + sorted((BASE / "logs").glob("*.log"))
    B.save_json(REPORTS / "output-hashes.json", {str(p.relative_to(BASE)): B.sha(p) for p in output_files})
    compact = {"terminal": {split: {"planned": report["planned"], "roots": report["roots"],
        "classifier": {k: report["views"]["classifier"][k] for k in ("known_reference_n", "root_agreement", "root_macro_f1", "row_agreement", "row_macro_f1", "confirmed_correct")},
        "pair_gate": {k: report["views"]["pair_gate"][k] for k in ("root_agreement", "root_macro_f1", "confirmed_correct")},
        "production_gated": {k: report["views"]["production_gated"][k] for k in ("root_agreement", "root_macro_f1", "confirmed_correct")},
        "completion": report["completion"], "correct_completed_overall": report["correct_completed_overall"],
        "semantic_uncertain": report["semantic_uncertain"], "system_failure": report["system_failure"], "failure_causes": report["failure_causes"]}
        for split, report in summary["terminal"].items()},
        "old_set": {k: summary["old_set"][k] for k in ("planned", "completion", "correct_completed_overall", "semantic_uncertain", "system_failure", "oracle_aggregation", "by_rule_count", "physical_predictions", "scored_pairs")},
        "excluded_no_truth": summary["excluded_no_truth"], "structural_fixtures": summary["structural_fixtures"]}
    B.save_json(REPORTS / "compact-summary.json", compact)
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
