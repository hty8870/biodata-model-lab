                      
"""原子关系 heldout 复算分析器（2026-10-25）。

只读 runs/ 与 baseline/heldout/ 的已落盘结果，重算指标后与生产 summary.json 对账；
不训练、不调用模型、不联网、不改模型或阈值。输出 EXP/analysis/{summary.json,errors.jsonl}。

默认命令先跑 selftest（含必过与必败样本），通过后再分析：
    python EXP/analyze_results_20261025.py

常用变体：
    python EXP/analyze_results_20261025.py --selftest-only
    python EXP/analyze_results_20261025.py --skip-selftest
    python EXP/analyze_results_20261025.py --data-root <实验目录> --out <输出目录>

数据根默认从脚本所在目录向上查找同时含 runs/ 与 baseline/ 的目录（当前为 EXP 的上一级）。
seed42 若尚无 final/，只进 missing_runs，不生成任何替代数字。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SCHEMA = "relation-atomization-analysis/1"
COARSE = ("equivalent", "compatible", "conflict")
FINE = ("equivalent", "old_entails_new_strict", "new_entails_old_strict",
        "disjoint", "overlap_incomparable")
STRICT = ("old_entails_new_strict", "new_entails_old_strict")
FINE_CONFLICT = "disjoint"

ERROR_TYPES = ("conflict_miss", "strict_subset_redundancy", "false_equivalence", "other")
ERROR_TYPE_LABELS = {
    "conflict_miss": "冲突漏判",
    "strict_subset_redundancy": "严格包含冗余",
    "false_equivalence": "等价误接受",
    "other": "其他",
}

EXPECTED_ARMS = ("direct3", "fine5")
EXPECTED_SEEDS = (13, 42)
RUN_PATTERN = re.compile(r"^(?P<arm>[A-Za-z0-9_]+)-seed(?P<seed>\d+)$")

TOLERANCE = 1e-9
PROB_SUM_TOLERANCE = 1e-5


                                                                                        
            
                                                                                        

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{line_number}: invalid JSON ({error})") from error
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8", newline="\n")


                                                                                        
                                                              
                                                                                        

def rate(numerator: int, denominator: int) -> dict:
    return {
        "value": None if denominator == 0 else float(numerator) / float(denominator),
        "numerator": int(numerator),
        "n": int(denominator),
    }


def metric_block(rows: Sequence[dict], classes: Sequence[str], *,
                 gold_key: str, pred_key: str,
                 equivalent: str = "equivalent", conflict: str = "conflict") -> dict:
    gold = [row[gold_key] for row in rows]
    pred = [row[pred_key] for row in rows]
    n = len(rows)

    per_class: dict[str, dict] = {}
    f1_values: list[float] = []
    for label in classes:
        support = sum(1 for value in gold if value == label)
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": support, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }
        f1_values.append(f1)

    matrix = [
        [sum(1 for g, p in zip(gold, pred) if g == gold_label and p == pred_label)
         for pred_label in classes]
        for gold_label in classes
    ]
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    strict_rows = [row for row in rows if row["gold_fine"] in STRICT]

    return {
        "n": n,
        "accuracy": correct / n if n else None,
        "macro_f1": sum(f1_values) / len(classes) if classes else None,
        "correct": correct,
        "classes": list(classes),
        "per_class": per_class,
        "confusion_matrix": {"classes": list(classes), "matrix": matrix},
        "conflict_recall": rate(
            sum(1 for g, p in zip(gold, pred) if g == conflict and p == conflict),
            sum(1 for g in gold if g == conflict)),
        "false_equivalence": rate(
            sum(1 for g, p in zip(gold, pred) if g != equivalent and p == equivalent),
            sum(1 for g in gold if g != equivalent)),
        "strict_subset_redundancy_accept": rate(
            sum(1 for row in strict_rows if row[pred_key] == equivalent), len(strict_rows)),
    }


def gate_block(rows: Sequence[dict], threshold: float) -> dict:
    """单阈值 risk-coverage；numerator 只数 accepted 行，分母仍是该 gold 组全部行。"""
    n = len(rows)
    gold = [row["gold_coarse"] for row in rows]
    pred = [row["pred_coarse"] for row in rows]
    accepted = [row["max_coarse_probability"] >= threshold for row in rows]
    accepted_n = sum(1 for value in accepted if value)
    hits = sum(1 for g, p, ok in zip(gold, pred, accepted) if ok and g == p)
    conflict_total = sum(1 for g in gold if g == "conflict")
    conflict_hits = sum(1 for g, p, ok in zip(gold, pred, accepted)
                        if ok and g == "conflict" and p == "conflict")
    non_equivalent_total = sum(1 for g in gold if g != "equivalent")
    false_equivalence_hits = sum(1 for g, p, ok in zip(gold, pred, accepted)
                                 if ok and g != "equivalent" and p == "equivalent")
    matrix = [
        [sum(1 for g, p, ok in zip(gold, pred, accepted) if ok and g == gold_label and p == pred_label)
         for pred_label in COARSE]
        for gold_label in COARSE
    ]
    return {
        "threshold": float(threshold),
        "n": n,
        "accepted": accepted_n,
        "coverage": rate(accepted_n, n),
        "fallback": rate(n - accepted_n, n),
        "accepted_accuracy": rate(hits, accepted_n),
        "conflict_recall": rate(conflict_hits, conflict_total),
        "conflict_denominator": conflict_total,
        "false_equivalence": rate(false_equivalence_hits, non_equivalent_total),
        "non_equivalent_denominator": non_equivalent_total,
        "confusion_at_acceptance": {"classes": list(COARSE), "matrix": matrix},
    }


                                                                                        
              
                                                                                        

def classify_error(row: dict) -> tuple[str | None, dict]:
    """返回 primary 错误类型与全部命中的 flag；正确行返回 (None, flags)。

    优先级 conflict_miss > strict_subset_redundancy > false_equivalence > other。
    strict_subset 必同时满足 false_equivalence，故 flag 可多项为真，primary 只取最具体一项。
    """
    gold_coarse = row["gold_coarse"]
    pred_coarse = row["pred_coarse"]
    gold_fine = row["gold_fine"]
    wrong = pred_coarse != gold_coarse
    flags = {
        "conflict_miss": bool(wrong and gold_coarse == "conflict"),
        "strict_subset_redundancy": bool(wrong and gold_fine in STRICT and pred_coarse == "equivalent"),
        "false_equivalence": bool(wrong and gold_coarse != "equivalent" and pred_coarse == "equivalent"),
    }
    flags["other"] = bool(wrong and not any(flags.values()))
    if not wrong:
        return None, flags
    for error_type in ERROR_TYPES:
        if flags[error_type]:
            return error_type, flags
    raise AssertionError(f"{row.get('id')}: wrong row without an error type")


def group_error_rates(rows: Sequence[dict], key_name: str, key_fn: Callable[[dict], Any],
                      denominator_text: str) -> dict:
    groups: dict[str, dict] = {}
    for row in rows:
        value = key_fn(row)
        name = "unknown" if value is None else str(value)
        entry = groups.setdefault(name, {"n_rows": 0, "n_errors": 0})
        entry["n_rows"] += 1
        if classify_error(row)[0] is not None:
            entry["n_errors"] += 1
    for entry in groups.values():
        entry["error_rate"] = rate(entry["n_errors"], entry["n_rows"])
    return {
        "key": key_name,
        "rate_denominator": denominator_text,
        "groups": {name: groups[name] for name in sorted(groups)},
    }


def error_stats(rows: Sequence[dict]) -> dict:
    by_type: dict[str, list[str]] = {error_type: [] for error_type in ERROR_TYPES}
    flag_counts = {error_type: 0 for error_type in ERROR_TYPES}
    errors: list[dict] = []
    for row in rows:
        error_type, flags = classify_error(row)
        if error_type is None:
            continue
        errors.append(row)
        by_type[error_type].append(row["id"])
        for name, value in flags.items():
            if value:
                flag_counts[name] += 1
    return {
        "scope": "coarse: pred_coarse != gold_coarse",
        "n_rows": len(rows),
        "n_errors": len(errors),
        "error_rate": rate(len(errors), len(rows)),
        "by_type": {
            error_type: {
                "n": len(by_type[error_type]),
                "label_zh": ERROR_TYPE_LABELS[error_type],
                "ids": sorted(by_type[error_type]),
            }
            for error_type in ERROR_TYPES
        },
        "flag_counts": flag_counts,
        "by_family": group_error_rates(
            rows, "family_id", lambda row: row["family_id"],
            "rows of the same family_id within this run"),
        "by_source": group_error_rates(
            rows, "source", lambda row: row["source"],
            "rows of the same source within this run"),
        "by_phenomena_family": group_error_rates(
            rows, "phenomena_family", lambda row: row.get("phenomena_family"),
            "rows of the same phenomena.family within this run"),
    }


def error_records(rows: Sequence[dict], *, run: str, arm: str, seed: int | None) -> list[dict]:
    records: list[dict] = []
    for row in sorted(rows, key=lambda item: item["id"]):
        error_type, flags = classify_error(row)
        if error_type is None:
            continue
        records.append({
            "run": run,
            "arm": arm,
            "seed": seed,
            "split": row.get("split"),
            "id": row["id"],
            "family_id": row["family_id"],
            "phenomena_family": row.get("phenomena_family"),
            "source": row["source"],
            "author": row.get("author"),
            "gold": {"coarse": row["gold_coarse"], "fine": row["gold_fine"]},
            "pred": {"coarse": row["pred_coarse"], "fine": row["pred_fine"]},
            "probabilities": {
                "coarse": row["coarse_probabilities"],
                "fine": row["fine_probabilities"],
                "max_coarse_probability": row["max_coarse_probability"],
                "max_fine_probability": row["max_fine_probability"],
            },
            "error_type": error_type,
            "error_type_zh": ERROR_TYPE_LABELS[error_type],
            "error_flags": flags,
            "old_text": row.get("old_text"),
            "new_text": row.get("new_text"),
            "text_source": row.get("text_source"),
        })
    return records


                                                                                        
                           
                                                                                        

def compare_arms(rows_a: Sequence[dict], rows_b: Sequence[dict], *,
                 label_a: str, label_b: str, seed: int) -> dict:
    map_a = {row["id"]: row for row in rows_a}
    map_b = {row["id"]: row for row in rows_b}
    shared = sorted(set(map_a) & set(map_b))
    only_a = sorted(set(map_a) - set(map_b))
    only_b = sorted(set(map_b) - set(map_a))

    errors_a = {key for key in shared if map_a[key]["pred_coarse"] != map_a[key]["gold_coarse"]}
    errors_b = {key for key in shared if map_b[key]["pred_coarse"] != map_b[key]["gold_coarse"]}
    misses_a = {key for key in shared if map_a[key]["gold_coarse"] == "conflict"
                and map_a[key]["pred_coarse"] != "conflict"}
    misses_b = {key for key in shared if map_b[key]["gold_coarse"] == "conflict"
                and map_b[key]["pred_coarse"] != "conflict"}
    fixed_by_b = sorted(key for key in misses_a if map_b[key]["pred_coarse"] == "conflict")
    fixed_by_a = sorted(key for key in misses_b if map_a[key]["pred_coarse"] == "conflict")

    return {
        "seed": seed,
        "arms": {"a": label_a, "b": label_b},
        "error_scope": "coarse: pred_coarse != gold_coarse; comparison restricted to shared ids",
        "n_shared_ids": len(shared),
        "ids_only_in_a": only_a,
        "ids_only_in_b": only_b,
        "a_error_count": len(errors_a),
        "b_error_count": len(errors_b),
        "common_errors": {"n": len(errors_a & errors_b), "ids": sorted(errors_a & errors_b)},
        "a_only_errors": {"n": len(errors_a - errors_b), "ids": sorted(errors_a - errors_b)},
        "b_only_errors": {"n": len(errors_b - errors_a), "ids": sorted(errors_b - errors_a)},
        "conflict_miss_recovery": {
            "a_missed": {"n": len(misses_a), "ids": sorted(misses_a)},
            "b_missed": {"n": len(misses_b), "ids": sorted(misses_b)},
            "b_fixed_a": {
                "n": len(fixed_by_b),
                "denominator": len(misses_a),
                "rate": rate(len(fixed_by_b), len(misses_a)),
                "ids": fixed_by_b,
            },
            "a_fixed_b": {
                "n": len(fixed_by_a),
                "denominator": len(misses_b),
                "rate": rate(len(fixed_by_a), len(misses_b)),
                "ids": fixed_by_a,
            },
        },
    }


                                                                                        
                 
                                                                                        

def load_texts(path: Path) -> dict[str, dict]:
    texts: dict[str, dict] = {}
    for row in load_jsonl(path):
        pair = row.get("pair") or {}
        old = pair.get("old") if isinstance(pair.get("old"), dict) else {}
        new = pair.get("new") if isinstance(pair.get("new"), dict) else {}
        texts[row["id"]] = {
            "old_text": old.get("text"),
            "new_text": new.get("text"),
            "phenomena_family": (row.get("phenomena") or {}).get("family"),
            "text_source": f"data/processed/{path.name}",
        }
    return texts


def attach_texts(rows: Sequence[dict], texts: dict[str, dict]) -> list[dict]:
    attached: list[dict] = []
    for row in rows:
        extra = texts.get(row["id"]) or {}
        merged = dict(row)
        merged["old_text"] = extra.get("old_text")
        merged["new_text"] = extra.get("new_text")
        merged["phenomena_family"] = extra.get("phenomena_family")
        merged["text_source"] = extra.get("text_source")
        attached.append(merged)
    return attached


                                                                                        
                         
                                                                                        

def resolve_data_root(script_dir: Path, override: str | None) -> Path:
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / "runs").is_dir():
            raise SystemExit(f"--data-root has no runs/: {root}")
        return root
    for candidate in [script_dir, *list(script_dir.parents)[:3]]:
        if (candidate / "runs").is_dir() and (candidate / "baseline").is_dir():
            return candidate
    raise SystemExit(
        "cannot locate data root containing runs/ and baseline/; pass --data-root explicitly "
        f"(searched from {script_dir} up 3 levels)")


def discover_runs(runs_dir: Path) -> dict[tuple[str, int], Path]:
    found: dict[tuple[str, int], Path] = {}
    if not runs_dir.is_dir():
        return found
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        match = RUN_PATTERN.match(entry.name)
        if not match:
            continue
        found[(match.group("arm"), int(match.group("seed")))] = entry
    return found


def load_run(run_dir: Path, texts: dict[str, dict]) -> tuple[dict | None, list[str]]:
    summary_path = run_dir / "final" / "summary.json"
    predictions_path = run_dir / "final" / "predictions.jsonl"
    missing = [str(path.relative_to(run_dir)).replace("\\", "/")
               for path in (summary_path, predictions_path) if not path.is_file()]
    if missing:
        return None, missing
    summary = load_json(summary_path)
    rows = attach_texts(load_jsonl(predictions_path), texts)
    return {
        "summary": summary,
        "rows": rows,
        "summary_path": summary_path,
        "predictions_path": predictions_path,
        "calibration_path": run_dir / "calibration.json",
    }, []


def load_baseline(baseline_dir: Path, texts: dict[str, dict]) -> dict | None:
    summary_path = baseline_dir / "summary.json"
    predictions_path = baseline_dir / "predictions.jsonl"
    if not summary_path.is_file() or not predictions_path.is_file():
        return None
    return {
        "summary": load_json(summary_path),
        "rows": attach_texts(load_jsonl(predictions_path), texts),
        "summary_path": summary_path,
        "predictions_path": predictions_path,
    }


                                                                                        
                                         
                                                                                        

def _same(left: Any, right: Any, tolerance: float) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= tolerance
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _same(a, b, tolerance) for a, b in zip(left, right))
    return left == right


def _field(name: str, mine: Any, container: dict, key: str,
           mismatches: list, skipped: list, checked: list) -> None:
    if key not in container:
        skipped.append(name)
        return
    recorded = container[key]
    if not _same(mine, recorded, TOLERANCE):
        mismatches.append({"field": name, "recomputed": mine, "recorded": recorded})
    else:
        checked.append(name)


def verify_metric_block(prefix: str, mine: dict, recorded: dict,
                        mismatches: list, skipped: list, checked: list) -> None:
    for key in ("n", "accuracy", "macro_f1", "correct"):
        _field(f"{prefix}.{key}", mine[key], recorded, key, mismatches, skipped, checked)
    recorded_classes = recorded.get("per_class") or {}
    for label, values in mine["per_class"].items():
        target = recorded_classes.get(label) or {}
        for key in ("support", "tp", "fp", "fn", "precision", "recall", "f1"):
            _field(f"{prefix}.per_class.{label}.{key}", values[key], target, key,
                   mismatches, skipped, checked)
    recorded_confusion = recorded.get("confusion_matrix") or {}
    _field(f"{prefix}.confusion_matrix", mine["confusion_matrix"]["matrix"],
           recorded_confusion, "matrix", mismatches, skipped, checked)
    for key in ("conflict_recall", "false_equivalence", "strict_subset_redundancy_accept"):
        target = recorded.get(key) or {}
        for sub in ("numerator", "n"):
            _field(f"{prefix}.{key}.{sub}", mine[key][sub], target, sub,
                   mismatches, skipped, checked)


def verify_gate(prefix: str, mine: dict, recorded: dict,
                mismatches: list, skipped: list, checked: list) -> None:
    for key in ("threshold", "accepted"):
        _field(f"{prefix}.{key}", mine[key], recorded, key, mismatches, skipped, checked)
    for key in ("coverage", "fallback", "accepted_accuracy", "conflict_recall", "false_equivalence"):
        target = recorded.get(key) or {}
        for sub in ("numerator", "n"):
            _field(f"{prefix}.{key}.{sub}", mine[key][sub], target, sub,
                   mismatches, skipped, checked)
    recorded_confusion = recorded.get("confusion_at_acceptance") or {}
    _field(f"{prefix}.confusion_at_acceptance", mine["confusion_at_acceptance"]["matrix"],
           recorded_confusion, "matrix", mismatches, skipped, checked)


                                                                                        
                       
                                                                                        

def _synthetic_rows() -> tuple[list[dict], list[dict], dict[str, dict]]:
    golds = {
        "s1": ("conflict", "disjoint"),
        "s2": ("conflict", "disjoint"),
        "s3": ("compatible", "old_entails_new_strict"),
        "s4": ("compatible", "overlap_incomparable"),
        "s5": ("conflict", "disjoint"),
        "s6": ("equivalent", "equivalent"),
    }
    families = {"s1": "fam-a", "s2": "fam-a", "s3": "fam-a", "s4": "fam-a",
                "s5": "fam-b", "s6": "fam-b"}
    coarse = {
        "s1": [0.01, 0.04, 0.95],
        "s2": [0.05, 0.80, 0.15],
        "s3": [0.90, 0.06, 0.04],
        "s4": [0.70, 0.20, 0.10],
        "s5": [0.02, 0.03, 0.95],
        "s6": [0.30, 0.60, 0.10],
    }
    preds_a = {"s1": "conflict", "s2": "compatible", "s3": "equivalent",
               "s4": "equivalent", "s5": "conflict", "s6": "compatible"}
    preds_b = {"s1": "conflict", "s2": "conflict", "s3": "equivalent",
               "s4": "equivalent", "s5": "compatible", "s6": "compatible"}
    texts = {
        key: {
            "old_text": f"旧要求 {key}：每批条目必须记录随机种子",
            "new_text": f"新要求 {key}：每批条目必须附上评测命令",
            "phenomena_family": "synthetic_phenomenon",
            "text_source": "synthetic",
        }
        for key in golds
    }

    def build(preds: dict[str, str]) -> list[dict]:
        rows = []
        for key in sorted(golds):
            gold_coarse, gold_fine = golds[key]
            probabilities = coarse[key]
            rows.append({
                "id": key,
                "split": "synthetic",
                "family_id": families[key],
                "source": "synthetic-source",
                "author": "selftest",
                "gold_coarse": gold_coarse,
                "gold_fine": gold_fine,
                "pred_coarse": preds[key],
                "pred_fine": None,
                "coarse_probabilities": list(probabilities),
                "fine_probabilities": None,
                "max_coarse_probability": max(probabilities),
                "max_fine_probability": max(probabilities),
            })
        return attach_texts(rows, texts)

    return build(preds_a), build(preds_b), texts


def run_selftest() -> dict:
    rows_a, rows_b, texts = _synthetic_rows()
    checks: list[dict] = []

    def expect_pass(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except AssertionError as error:
            checks.append({"name": name, "expected": "pass", "ok": False, "detail": str(error)})
        else:
            checks.append({"name": name, "expected": "pass", "ok": True, "detail": "ok"})

    def expect_fail(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except AssertionError as error:
            checks.append({"name": name, "expected": "fail", "ok": True,
                           "detail": f"rejected: {error}"})
        else:
            checks.append({"name": name, "expected": "fail", "ok": False,
                           "detail": "wrong expectation was accepted"})

    def check_conflict_miss() -> None:
        block = metric_block(rows_a, COARSE, gold_key="gold_coarse", pred_key="pred_coarse")
        assert block["conflict_recall"] == {"value": 2 / 3, "numerator": 2, "n": 3}, block["conflict_recall"]
        error_type, flags = classify_error(rows_a[1])
        assert rows_a[1]["id"] == "s2" and error_type == "conflict_miss", (error_type, flags)
        assert classify_error(rows_a[0])[0] is None, classify_error(rows_a[0])

    def check_false_equivalence_and_strict() -> None:
        block = metric_block(rows_a, COARSE, gold_key="gold_coarse", pred_key="pred_coarse")
        assert block["false_equivalence"] == {"value": 2 / 5, "numerator": 2, "n": 5}, block["false_equivalence"]
        assert block["strict_subset_redundancy_accept"] == {"value": 1.0, "numerator": 1, "n": 1}, \
            block["strict_subset_redundancy_accept"]
        strict_type, strict_flags = classify_error(rows_a[2])
        assert rows_a[2]["id"] == "s3" and strict_type == "strict_subset_redundancy", strict_type
        assert strict_flags["false_equivalence"] is True, strict_flags
        plain_type, plain_flags = classify_error(rows_a[3])
        assert rows_a[3]["id"] == "s4" and plain_type == "false_equivalence", plain_type
        assert plain_flags["strict_subset_redundancy"] is False, plain_flags

    def check_common_unique_and_recovery() -> None:
        comparison = compare_arms(rows_a, rows_b, label_a="synthA", label_b="synthB", seed=13)
        assert comparison["common_errors"] == {"n": 3, "ids": ["s3", "s4", "s6"]}, comparison["common_errors"]
        assert comparison["a_only_errors"] == {"n": 1, "ids": ["s2"]}, comparison["a_only_errors"]
        assert comparison["b_only_errors"] == {"n": 1, "ids": ["s5"]}, comparison["b_only_errors"]
        recovery = comparison["conflict_miss_recovery"]["b_fixed_a"]
        assert recovery["ids"] == ["s2"] and recovery["n"] == 1 and recovery["denominator"] == 1, recovery
        assert recovery["rate"] == {"value": 1.0, "numerator": 1, "n": 1}, recovery["rate"]

    def check_denominators_and_macro() -> None:
        block = metric_block(rows_a, COARSE, gold_key="gold_coarse", pred_key="pred_coarse")
        assert block["accuracy"] == 2 / 6, block["accuracy"]
        assert abs(block["macro_f1"] - (0.8 + 0.0 + 0.0) / 3) <= 1e-12, block["macro_f1"]
        stats = error_stats(rows_a)
        assert stats["n_errors"] == 4, stats["n_errors"]
        expected_rates = {"fam-a": (3, 4), "fam-b": (1, 2)}
        for family, (errors, total) in expected_rates.items():
            entry = stats["by_family"]["groups"][family]
            assert (entry["n_errors"], entry["n_rows"]) == (errors, total), (family, entry)
            assert entry["error_rate"] == {"value": errors / total, "numerator": errors, "n": total}, entry
        assert stats["by_type"]["conflict_miss"]["n"] == 1, stats["by_type"]
        assert stats["by_type"]["strict_subset_redundancy"]["n"] == 1, stats["by_type"]
        assert stats["by_type"]["false_equivalence"]["n"] == 1, stats["by_type"]
        assert stats["by_type"]["other"]["n"] == 1, stats["by_type"]
        assert stats["flag_counts"]["false_equivalence"] == 2, stats["flag_counts"]

    def check_text_attachment() -> None:
        records = error_records(rows_a, run="synthA", arm="synthA", seed=13)
        lookup = {record["id"]: record for record in records}
        assert set(lookup) == {"s2", "s3", "s4", "s6"}, sorted(lookup)
        assert lookup["s3"]["old_text"] == texts["s3"]["old_text"], lookup["s3"]["old_text"]
        assert lookup["s3"]["new_text"] == texts["s3"]["new_text"], lookup["s3"]["new_text"]
        assert lookup["s3"]["probabilities"]["coarse"] == [0.90, 0.06, 0.04], lookup["s3"]
        assert lookup["s2"]["error_type_zh"] == "冲突漏判", lookup["s2"]

    def reject_wrong_conflict_denominator() -> None:
        block = metric_block(rows_a, COARSE, gold_key="gold_coarse", pred_key="pred_coarse")
        assert block["conflict_recall"]["n"] == 2, "intentionally wrong: denominator is gold conflict rows (3)"

    def reject_correct_row_as_error() -> None:
        error_type, _ = classify_error(rows_a[0])
        assert error_type is not None, "intentionally wrong: s1 is correct"

    def reject_false_equivalence_all_rows_denominator() -> None:
        block = metric_block(rows_a, COARSE, gold_key="gold_coarse", pred_key="pred_coarse")
        assert block["false_equivalence"]["n"] == 6, \
            "intentionally wrong: denominator is gold != equivalent (5)"

    for name, fn in (
        ("conflict_miss_and_denominator", check_conflict_miss),
        ("false_equivalence_and_strict_subset", check_false_equivalence_and_strict),
        ("common_unique_and_conflict_recovery", check_common_unique_and_recovery),
        ("group_denominators_and_macro_f1", check_denominators_and_macro),
        ("error_records_text_and_probabilities", check_text_attachment),
    ):
        expect_pass(name, fn)
    for name, fn in (
        ("wrong_conflict_denominator_must_be_rejected", reject_wrong_conflict_denominator),
        ("correct_row_counted_as_error_must_be_rejected", reject_correct_row_as_error),
        ("all_rows_false_equivalence_denominator_must_be_rejected",
         reject_false_equivalence_all_rows_denominator),
    ):
        expect_fail(name, fn)

    must_pass = [check for check in checks if check["expected"] == "pass"]
    must_fail = [check for check in checks if check["expected"] == "fail"]
    ok = (all(check["ok"] for check in checks) and bool(must_pass) and bool(must_fail))
    return {
        "ok": ok,
        "network": "none",
        "must_pass": len(must_pass),
        "must_fail": len(must_fail),
        "checks": checks,
    }


                                                                                        
                   
                                                                                        

def _probability_deviation(rows: Sequence[dict]) -> float:
    worst = 0.0
    for row in rows:
        for key in ("coarse_probabilities", "fine_probabilities"):
            values = row.get(key)
            if values:
                worst = max(worst, abs(sum(values) - 1.0))
    return worst


def shape_issues(rows: Sequence[dict], label: str) -> list[str]:
    issues: list[str] = []
    unknown_gold = 0
    unknown_pred = 0
    length_bad = 0
    argmax_bad = 0
    max_field_bad = 0
    splits = set()
    for row in rows:
        splits.add(row.get("split"))
        if row.get("gold_coarse") not in COARSE or row.get("gold_fine") not in FINE:
            unknown_gold += 1
        if row.get("pred_coarse") not in COARSE:
            unknown_pred += 1
        coarse = row.get("coarse_probabilities") or []
        if len(coarse) != len(COARSE):
            length_bad += 1
        else:
            if COARSE[max(range(len(coarse)), key=lambda index: coarse[index])] != row.get("pred_coarse"):
                argmax_bad += 1
            if row.get("max_coarse_probability") is not None and abs(
                    max(coarse) - row["max_coarse_probability"]) > TOLERANCE:
                max_field_bad += 1
        fine = row.get("fine_probabilities")
        if fine is not None:
            if len(fine) != len(FINE):
                length_bad += 1
            elif row.get("pred_fine") and FINE[
                    max(range(len(fine)), key=lambda index: fine[index])] != row["pred_fine"]:
                argmax_bad += 1
    if unknown_gold:
        issues.append(f"{label}: {unknown_gold} rows carry gold labels outside the fixed class lists.")
    if unknown_pred:
        issues.append(f"{label}: {unknown_pred} rows carry pred_coarse outside {list(COARSE)}.")
    if length_bad:
        issues.append(f"{label}: {length_bad} probability rows have the wrong arity.")
    if argmax_bad:
        issues.append(f"{label}: {argmax_bad} rows where argmax(probabilities) != recorded pred label.")
    if max_field_bad:
        issues.append(f"{label}: {max_field_bad} rows where max_coarse_probability != max(coarse row).")
    if len(splits) != 1 or splits != {"heldout"}:
        issues.append(f"{label}: unexpected split values {sorted(str(value) for value in splits)}.")
    return issues


def unseen_prediction_notes(rows: Sequence[dict], label: str) -> list[str]:
    notes: list[str] = []
    for classes, key, tag in ((COARSE, "pred_coarse", "coarse"), (FINE, "pred_fine", "fine")):
        if key == "pred_fine" and not any(row.get("pred_fine") for row in rows):
            continue
        seen = {row[key] for row in rows if row.get(key)}
        missing = [value for value in classes if value not in seen]
        if not missing:
            continue
        suffix = (" (equivalent is never predicted, so false_equivalence = 0 is structural here)"
                  if "equivalent" in missing else "")
        notes.append(f"{label}: never predicts {tag} {missing}; those classes get recall 0 and "
                     f"precision 0.0 under the zero-denominator convention{suffix}.")
    return notes


def _file_info(path: Path, root: Path) -> dict:
    return {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_analysis(root: Path, texts_path: Path) -> tuple[dict, list[dict]]:
    runs_dir = root / "runs"
    texts = load_texts(texts_path) if texts_path.is_file() else {}
    discovered = discover_runs(runs_dir)

    wanted = sorted(set(discovered) | {(arm, seed) for arm in EXPECTED_ARMS for seed in EXPECTED_SEEDS})
    run_blocks: dict[str, dict] = {}
    run_rows: dict[str, dict] = {}
    error_lines: list[dict] = []
    missing_runs: list[dict] = []
    mismatches: list[dict] = []
    skipped: list[str] = []
    checked: list[str] = []
    inputs: dict[str, dict] = {}
    notes: list[str] = []

    for arm, seed in wanted:
        name = f"{arm}-seed{seed}"
        run_dir = discovered.get((arm, seed))
        if run_dir is None:
            missing_runs.append({
                "run": name, "arm": arm, "seed": seed, "directory_exists": False,
                "missing_files": ["final/summary.json", "final/predictions.jsonl"],
                "present_entries": [],
            })
            continue
        loaded, missing = load_run(run_dir, texts)
        if loaded is None:
            missing_runs.append({
                "run": name, "arm": arm, "seed": seed, "directory_exists": True,
                "missing_files": missing,
                "present_entries": sorted(entry.name for entry in run_dir.iterdir()),
            })
            continue

        summary = loaded["summary"]
        rows = loaded["rows"]
        coarse = metric_block(rows, COARSE, gold_key="gold_coarse", pred_key="pred_coarse")
        fine = (metric_block(rows, FINE, gold_key="gold_fine", pred_key="pred_fine",
                             conflict=FINE_CONFLICT)
                if all(row.get("pred_fine") for row in rows) else None)

        calibration_path = loaded["calibration_path"]
        calibration = load_json(calibration_path) if calibration_path.is_file() else None
        threshold = (calibration or {}).get("threshold")
        gate = gate_block(rows, threshold) if threshold is not None else None
        if calibration is None:
            notes.append(f"{name}: calibration.json missing; heldout gate not recomputed.")
        elif threshold is None:
            notes.append(f"{name}: calibration.json has threshold=null (no feasible threshold); "
                         "heldout gate not recomputed.")

        stats = error_stats(rows)
        run_blocks[name] = {
            "run": name,
            "arm": arm,
            "seed": seed,
            "split": summary.get("split"),
            "n": len(rows),
            "checkpoint": summary.get("checkpoint"),
            "coarse": coarse,
            "fine": fine,
            "calibration": None if calibration is None else {
                "path": str(calibration_path.relative_to(root)).replace("\\", "/"),
                "sha256": sha256_file(calibration_path),
                "split": calibration.get("split"),
                "n": calibration.get("n"),
                "threshold": threshold,
                "feasible_thresholds": calibration.get("feasible_thresholds"),
                "rule": calibration.get("rule"),
            },
            "gate": gate,
            "errors": stats,
        }
        run_rows[name] = rows
        error_lines.extend(error_records(rows, run=name, arm=arm, seed=seed))
        inputs[name] = {
            "summary.json": _file_info(loaded["summary_path"], root),
            "predictions.jsonl": _file_info(loaded["predictions_path"], root),
        }
        if calibration_path.is_file():
            inputs[name]["calibration.json"] = _file_info(calibration_path, root)

        if len(rows) != summary.get("n"):
            mismatches.append({"field": f"{name}.n", "recomputed": len(rows),
                               "recorded": summary.get("n")})
        verify_metric_block(f"{name}.coarse", coarse, summary.get("coarse") or {},
                            mismatches, skipped, checked)
        if fine is not None:
            verify_metric_block(f"{name}.fine", fine, summary.get("fine") or {},
                                mismatches, skipped, checked)
        elif (summary.get("fine") is not None
              and not any(row.get("pred_fine") for row in rows)):
            notes.append(f"{name}: summary has a fine block but predictions carry no pred_fine; "
                         "fine metrics not recomputed.")
        if gate is not None:
            verify_gate(f"{name}.gate", gate, summary.get("gate") or {},
                        mismatches, skipped, checked)

        if len({row["id"] for row in rows}) != len(rows):
            notes.append(f"{name}: duplicate ids in predictions.jsonl.")
        if any(row.get("pred_fine") for row in rows) and not fine:
            notes.append(f"{name}: only part of the rows carry pred_fine; fine block skipped.")
        notes.extend(shape_issues(rows, name))
        notes.extend(unseen_prediction_notes(rows, name))

    baseline = load_baseline(root / "baseline" / "heldout", texts) if (root / "baseline" / "heldout").is_dir() else None
    baseline_block = None
    if baseline is None:
        notes.append("baseline/heldout/{summary.json,predictions.jsonl} missing; baseline block omitted.")
    else:
        summary = baseline["summary"]
        rows = baseline["rows"]
        coarse = metric_block(rows, COARSE, gold_key="gold_coarse", pred_key="pred_coarse")
        complete_rows = [row for row in rows if row.get("input_complete")]
        coarse_complete = metric_block(complete_rows, COARSE, gold_key="gold_coarse",
                                       pred_key="pred_coarse")
        fine = metric_block(rows, FINE, gold_key="gold_fine", pred_key="pred_fine",
                            conflict=FINE_CONFLICT)
        recorded_gate = summary.get("gate") or {}
        threshold = recorded_gate.get("threshold")
        gate = gate_block(rows, threshold) if threshold is not None else None
        baseline_block = {
            "run": "baseline/heldout",
            "source": summary.get("source"),
            "split": summary.get("split"),
            "n": len(rows),
            "model": summary.get("model"),
            "coarse": coarse,
            "coarse_complete": {
                **coarse_complete,
                "n_incomplete_excluded": len(rows) - len(complete_rows),
                "identical_to_coarse": coarse_complete["n"] == coarse["n"],
                "scope_note": (summary.get("scoring_scope") or {}).get("all_row_metrics_meaning"),
            },
            "fine": fine,
            "gate_at_recorded_threshold": gate,
            "errors": error_stats(rows),
        }
        error_lines.extend(error_records(rows, run="baseline/heldout", arm="baseline", seed=None))
        inputs["baseline/heldout"] = {
            "summary.json": _file_info(baseline["summary_path"], root),
            "predictions.jsonl": _file_info(baseline["predictions_path"], root),
        }
        verify_metric_block("baseline.heldout.coarse", coarse, summary.get("coarse") or {},
                            mismatches, skipped, checked)
        verify_metric_block("baseline.heldout.coarse_complete", coarse_complete,
                            summary.get("coarse_complete") or {}, mismatches, skipped, checked)
        verify_metric_block("baseline.heldout.fine", fine, summary.get("fine") or {},
                            mismatches, skipped, checked)
        if gate is not None:
            verify_gate("baseline.heldout.gate", gate, recorded_gate,
                        mismatches, skipped, checked)
        if len(rows) != summary.get("n"):
            mismatches.append({"field": "baseline.heldout.n", "recomputed": len(rows),
                               "recorded": summary.get("n")})
        notes.extend(shape_issues(rows, "baseline/heldout"))
        notes.extend(unseen_prediction_notes(rows, "baseline/heldout"))

    comparisons: dict[str, dict] = {}
    by_seed: dict[int, dict[str, list[dict]]] = {}
    for name, rows in run_rows.items():
        seed = run_blocks[name]["seed"]
        by_seed.setdefault(seed, {})[name] = rows
    for seed in sorted(by_seed):
        for name_a, name_b in combinations(sorted(by_seed[seed]), 2):
            key = f"{name_a}__vs__{name_b}"
            comparisons[key] = compare_arms(
                by_seed[seed][name_a], by_seed[seed][name_b],
                label_a=name_a, label_b=name_b, seed=seed)

    id_sets = {name: {row["id"] for row in rows} for name, rows in run_rows.items()}
    if baseline_block is not None:
        id_sets["baseline/heldout"] = {row["id"] for row in baseline["rows"]}
    if id_sets:
        reference_name = sorted(id_sets)[0]
        reference = id_sets[reference_name]
        for name in sorted(id_sets):
            if id_sets[name] != reference:
                notes.append(f"id set mismatch: {name} vs {reference_name}.")
    text_ids = set(texts)
    if text_ids:
        for name in sorted(id_sets):
            missing_text = sorted(id_sets[name] - text_ids)
            if missing_text:
                notes.append(f"{name}: {len(missing_text)} predictions lack old/new text "
                             f"(e.g. {missing_text[:3]}).")

    deviations = {(name): _probability_deviation(rows) for name, rows in run_rows.items()}
    if baseline_block is not None:
        deviations["baseline/heldout"] = _probability_deviation(baseline["rows"])
    worst_deviation = max(deviations.values()) if deviations else 0.0
    if worst_deviation > PROB_SUM_TOLERANCE:
        notes.append(f"probability rows deviate from sum=1 by up to {worst_deviation:.3g}.")

    if baseline_block is not None:
        gate = baseline_block["gate_at_recorded_threshold"]
        if gate is not None and gate["accepted"] == 0:
            observed_max = max(row["max_coarse_probability"] for row in baseline["rows"])
            notes.append(
                f"baseline gate at recorded threshold {gate['threshold']:.2f}: accepted "
                f"{gate['accepted']}/{gate['n']} on heldout; the largest max_coarse_probability is "
                f"{observed_max:.4f}, so coverage 0 and accepted_accuracy null follow from the "
                "confidence range, not from a scoring error.")
        if baseline_block["coarse_complete"]["identical_to_coarse"]:
            notes.append("baseline heldout has 0 incomplete rows, so coarse_complete == coarse.")
    notes.append(
        "gate false_equivalence numerator counts accepted rows only while its denominator is all "
        "gold != equivalent rows (producer gate_row); the coarse-block false_equivalence counts all "
        "rows. The two are not interchangeable.")
    notes.append(
        "family grouping uses family_id (scenario family, 15 heldout rows each); "
        "phenomena_family comes from data/processed/heldout.jsonl and is reported separately.")
    if any(block["fine"] is None for block in run_blocks.values()):
        notes.append("runs without pred_fine (direct3) have fine=null by design; fine metrics exist "
                     "only for five-leaf arms and the baseline.")
    if missing_runs:
        notes.append("missing final runs are listed in missing_runs only; no metrics are synthesized: "
                     + ", ".join(entry["run"] for entry in missing_runs) + ".")
    if not texts_path.is_file():
        notes.append(f"pair text source missing ({texts_path}); old/new text is null in errors.jsonl.")

    summary = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_root": str(root),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "selftest": None,
        "metric_definitions": {
            "accuracy": "correct / n rows",
            "macro_f1": "unweighted mean of per-class F1 over the class list; zero-denominator P/R/F1 = 0.0",
            "per_class": "one-vs-rest tp/fp/fn over the same rows",
            "conflict_recall": "pred == conflict among gold conflict; denominator = gold conflict rows",
            "false_equivalence": "pred == equivalent among gold != equivalent; denominator = gold != equivalent rows",
            "strict_subset_redundancy_accept": (
                "pred == equivalent among gold old_entails_new_strict/new_entails_old_strict; "
                "denominator = gold strict-subset rows"),
            "error_type": (
                "coarse-level; priority conflict_miss > strict_subset_redundancy > "
                "false_equivalence > other; error_flags keeps every match, so flags can exceed errors"),
            "gate": (
                "accept local argmax when max_coarse_probability >= threshold; coverage/accepted_accuracy "
                "use accepted rows, conflict_recall and false_equivalence use their full gold-group "
                "denominators while counting accepted rows only (score_and_verify.gate_row)"),
            "group_error_rate": "n_errors / rows in that group of that source; never pooled across sources",
        },
        "inputs": {
            "runs": inputs,
            "texts": ({"path": str(texts_path.relative_to(root)).replace("\\", "/"),
                       "sha256": sha256_file(texts_path),
                       "matched_ids": len(text_ids)} if texts_path.is_file() else None),
        },
        "runs": run_blocks,
        "baseline": {"heldout": baseline_block} if baseline_block is not None else {},
        "comparisons": comparisons,
        "missing_runs": missing_runs,
        "verification": {
            "status": "ok" if not mismatches else "mismatch",
            "tolerance": TOLERANCE,
            "checked_field_count": len(checked),
            "mismatches": mismatches,
            "skipped_fields": skipped,
            "max_probability_row_sum_deviation": worst_deviation,
        },
        "data_shape_notes": notes,
    }
    return summary, error_lines


                                                                                        
                
                                                                                        

def _fmt_rate(value: dict | None) -> str:
    if not value or value.get("n") == 0:
        return "n/a"
    return f"{value['numerator']}/{value['n']}={value['value']:.5f}"


def print_report(summary: dict, out_dir: Path) -> None:
    for name, block in summary["runs"].items():
        coarse = block["coarse"]
        gate = block["gate"]
        gate_text = ("gate=none" if gate is None else
                     f"gate@{gate['threshold']:.2f} cov={_fmt_rate(gate['coverage'])} "
                     f"acc={_fmt_rate(gate['accepted_accuracy'])} "
                     f"conflict={_fmt_rate(gate['conflict_recall'])} "
                     f"falseEq={_fmt_rate(gate['false_equivalence'])}")
        print(f"[run] {name} n={block['n']} acc={coarse['accuracy']:.5f} "
              f"macroF1={coarse['macro_f1']:.5f} "
              f"conflict={_fmt_rate(coarse['conflict_recall'])} "
              f"falseEq={_fmt_rate(coarse['false_equivalence'])} "
              f"strict={_fmt_rate(coarse['strict_subset_redundancy_accept'])} "
              f"errors={block['errors']['n_errors']}/{block['n']} | {gate_text}")
    baseline = summary["baseline"].get("heldout")
    if baseline:
        coarse = baseline["coarse"]
        gate = baseline["gate_at_recorded_threshold"]
        gate_text = ("gate=none" if gate is None else
                     f"gate@{gate['threshold']:.2f} cov={_fmt_rate(gate['coverage'])} "
                     f"acc={_fmt_rate(gate['accepted_accuracy'])}")
        print(f"[baseline] heldout n={baseline['n']} acc={coarse['accuracy']:.5f} "
              f"macroF1={coarse['macro_f1']:.5f} "
              f"conflict={_fmt_rate(coarse['conflict_recall'])} "
              f"falseEq={_fmt_rate(coarse['false_equivalence'])} "
              f"strict={_fmt_rate(coarse['strict_subset_redundancy_accept'])} "
              f"errors={baseline['errors']['n_errors']}/{baseline['n']} | "
              f"complete n={baseline['coarse_complete']['n']} "
              f"acc={baseline['coarse_complete']['accuracy']:.5f} | {gate_text}")
    for name, comparison in summary["comparisons"].items():
        print(f"[compare] {name} shared={comparison['n_shared_ids']} "
              f"common={comparison['common_errors']['n']} "
              f"a_only={comparison['a_only_errors']['n']} "
              f"b_only={comparison['b_only_errors']['n']} "
              f"b_fixed_a_conflict_miss={comparison['conflict_miss_recovery']['b_fixed_a']['n']}"
              f"/{comparison['conflict_miss_recovery']['b_fixed_a']['denominator']} "
              f"ids={comparison['conflict_miss_recovery']['b_fixed_a']['ids']}")
    if summary["missing_runs"]:
        print("[missing] " + ", ".join(entry["run"] for entry in summary["missing_runs"]))
    verification = summary["verification"]
    print(f"[verify] status={verification['status']} "
          f"checked={verification['checked_field_count']} "
          f"mismatches={len(verification['mismatches'])} "
          f"skipped={len(verification['skipped_fields'])} "
          f"max_prob_sum_dev={verification['max_probability_row_sum_deviation']:.3g}")
    for mismatch in verification["mismatches"][:10]:
        print(f"  mismatch {mismatch['field']}: recomputed={mismatch['recomputed']!r} "
              f"recorded={mismatch['recorded']!r}")
    for note in summary["data_shape_notes"]:
        print(f"[note] {note}")
    print(f"[out] {out_dir / 'summary.json'}")
    print(f"[out] {out_dir / 'errors.jsonl'}")


                                                                                        
             
                                                                                        

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute heldout metrics for direct3/fine5 runs and baseline; no training.")
    parser.add_argument("--data-root", default=None,
                        help="experiment dir containing runs/ and baseline/ "
                             "(default: nearest ancestor of this script with both)")
    parser.add_argument("--out", default=None, help="output dir (default: <script dir>/analysis)")
    parser.add_argument("--texts", default=None,
                        help="pair text source (default: <data-root>/data/processed/heldout.jsonl)")
    parser.add_argument("--selftest-only", action="store_true",
                        help="run the selftest and exit without analysing")
    parser.add_argument("--skip-selftest", action="store_true",
                        help="analyse without running the selftest (not recommended)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent

    selftest = None
    if not args.skip_selftest:
        selftest = run_selftest()
        print(f"[selftest] ok={selftest['ok']} must_pass={selftest['must_pass']} "
              f"must_fail={selftest['must_fail']}")
        for check in selftest["checks"]:
            if not check["ok"]:
                print(f"  FAIL {check['name']} ({check['expected']}): {check['detail']}")
        if not selftest["ok"]:
            print("[selftest] failed; analysis skipped")
            return 1
    if args.selftest_only:
        return 0

    root = resolve_data_root(script_dir, args.data_root)
    texts_path = (Path(args.texts).expanduser().resolve() if args.texts
                  else root / "data" / "processed" / "heldout.jsonl")
    out_dir = Path(args.out).expanduser().resolve() if args.out else script_dir / "analysis"

    summary, error_lines = build_analysis(root, texts_path)
    summary["selftest"] = selftest
    summary["outputs"] = {
        "summary": str(out_dir / "summary.json"),
        "errors": str(out_dir / "errors.jsonl"),
    }
    write_jsonl(out_dir / "errors.jsonl", error_lines)
    write_json(out_dir / "summary.json", summary)
    print_report(summary, out_dir)
    return 0 if summary["verification"]["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
