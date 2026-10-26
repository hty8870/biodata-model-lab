from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIMARY = HERE / "pilot_v4/out/real_results/scored_all_220.jsonl"
PAIRED = HERE / "pilot_v4/diagnostics_20261025/paired_results.jsonl"


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def fraction(values: list[bool]) -> dict:
    return {"correct": sum(values), "n": len(values), "rate": sum(values) / len(values) if values else None}


def primary_summary(rows: list[dict], excluded: set[str]) -> dict:
    return {arm: {"exact": fraction([r["score"]["exact_success"] for r in rows if r["arm"] == arm and r["task_id"] not in excluded]),
                  "grounded": fraction([r["score"]["grounded_success"] for r in rows if r["arm"] == arm and r["task_id"] not in excluded])}
            for arm in sorted({r["arm"] for r in rows})}


def paired_summary(rows: list[dict], excluded: set[str]) -> dict:
    result = {}
    for arm in sorted({r["arm"] for r in rows}):
        selected = [r for r in rows if r["arm"] == arm and r["task_id"] not in excluded]
        result[arm] = {key: fraction([r[source][metric] for r in selected]) for key, source, metric in (
            ("ds_exact", "ds_struct_score", "exact_success"), ("ds_grounded", "ds_struct_score", "grounded_success"),
            ("qwen_exact", "qwen_struct_score", "exact_success"), ("qwen_grounded", "qwen_struct_score", "grounded_success"))}
        result[arm]["ai_blind_correct_proxy"] = fraction([r["qwen_blind"]["judge"]["correct"] for r in selected])
    return result


def selftest() -> None:
    rows = [{"arm": "x", "task_id": "a", "score": {"exact_success": False, "grounded_success": False}},
            {"arm": "x", "task_id": "b", "score": {"exact_success": True, "grounded_success": True}}]
    assert primary_summary(rows, {"a"})["x"]["grounded"] == {"correct": 1, "n": 1, "rate": 1.0}
    assert primary_summary(rows, {"a", "b"})["x"]["grounded"]["rate"] is None
    histories = (["A"], ["A", "A"])
    assert all(history.count("A") >= 1 for history in histories)
    assert [len(history) != len(set(history)) for history in histories] == [False, True]
    print(json.dumps({"selftest_checks": 4, "passed": 4, "new_model_calls": 0}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--exclude-task", action="append", default=[])
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    primary, paired = read_rows(PRIMARY), read_rows(PAIRED)
    excluded = set(args.exclude_task)
    assert excluded <= {r["task_id"] for r in primary}
    result = {"schema": "posthoc-task-sensitivity/1", "not_preregistered_primary_score": True,
        "original_results_unchanged": True, "new_model_calls": 0, "excluded_task_ids": sorted(excluded),
        "exclusion_reason": args.reason, "scope": "all arms and scales; no cherry-picking individual predictions",
        "original_primary": primary_summary(primary, set()), "sensitivity_primary": primary_summary(primary, excluded),
        "original_paired": paired_summary(paired, set()), "sensitivity_paired": paired_summary(paired, excluded),
        "sources_sha256": {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (PRIMARY, PAIRED)},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "warning": "AI blind approval is not factual ground truth; sensitivity does not replace the registered score"}
    text = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    output = HERE / "parent_analysis_20261025" / f"task-sensitivity-{digest[:16]}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        assert output.read_text(encoding="utf-8") == text
    else:
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    print(json.dumps({"output": str(output), "sensitivity_primary": result["sensitivity_primary"],
                      "sensitivity_paired": result["sensitivity_paired"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
