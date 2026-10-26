from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

from score_atomic_memory_20261025 import blind_id_of

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ACCEPTED = {"reasonable_alternative", "acceptable_compound"}


def accepted(atom: dict) -> bool:
    value = atom.get("alternative_partition")
    return isinstance(value, dict) and value.get("type") in ACCEPTED


def eligible(atoms: list[dict], gold_id: str) -> list[dict]:
    covering = [atom for atom in atoms if gold_id in atom["core_entailed_gold_ids"]
                and gold_id not in atom["fully_covered_gold_ids"]]
    if not covering or not any(accepted(atom) for atom in covering):
        return []
    if any(atom["entailment"] != "supported" or not atom["persistence_appropriate"]
           or atom["context_errors"] or atom["over_split"] or atom["matched_forbidden_ids"]
           for atom in covering):
        return []
    return [atom for atom in covering if accepted(atom)]


def apply(rows: list[dict], cases_by_id: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    output = copy.deepcopy(rows)
    assignments = []
    for row in output:
        case = cases_by_id[row["case_id"]]
        for gold in case["gold_obligations"]:
            gold_id = gold["gold_id"]
            if any(gold_id in atom["fully_covered_gold_ids"] for atom in row["atoms"]):
                continue
            candidates = eligible(row["atoms"], gold_id)
            if not candidates:
                continue
            chosen = candidates[0]
            chosen["fully_covered_gold_ids"] = sorted(set(chosen["fully_covered_gold_ids"]) | {gold_id})
            chosen["core_entailed_gold_ids"] = sorted(set(chosen["core_entailed_gold_ids"]) | {gold_id})
            assignments.append({"case_id": row["case_id"], "blind_id": row["blind_id"],
                                "gold_id": gold_id, "candidate_id": chosen["candidate_id"],
                                "basis": "all_core_covering_atoms_supported_context_complete_and_alternative_marked"})
    return output, assignments


def selftest() -> dict:
    base = {"candidate_id": "a", "entailment": "supported", "persistence_appropriate": True,
            "non_atomic": False, "over_split": False, "context_errors": [], "matched_forbidden_ids": [],
            "duplicate_group": None, "core_entailed_gold_ids": ["g"], "fully_covered_gold_ids": [],
            "alternative_partition": {"type": "reasonable_alternative", "note": "split"}, "notes": ""}
    case = {"case_id": "c", "gold_obligations": [{"gold_id": "g"}]}
    row = {"case_id": "c", "blind_id": "b", "stage1_complete": True,
           "stage2_complete": True, "adjudicated": True, "atoms": [base]}
    positive, assigned = apply([row], {"c": case})
    missing, missing_assign = apply([{**row, "atoms": [{**base, "alternative_partition": None}]}], {"c": case})
    partial, _ = apply([{**row, "atoms": [{**base, "entailment": "partial"}]}], {"c": case})
    context, _ = apply([{**row, "atoms": [{**base, "context_errors": ["scope"]}]}], {"c": case})
    existing = copy.deepcopy(base)
    existing["fully_covered_gold_ids"] = ["g"]
    existing_out, existing_assign = apply([{**row, "atoms": [existing]}], {"c": case})
    checks = {
        "positive_assigns": len(assigned) == 1 and positive[0]["atoms"][0]["fully_covered_gold_ids"] == ["g"],
        "no_alternative_rejected": not missing_assign and missing[0]["atoms"][0]["fully_covered_gold_ids"] == [],
        "partial_rejected": partial[0]["atoms"][0]["fully_covered_gold_ids"] == [],
        "context_error_rejected": context[0]["atoms"][0]["fully_covered_gold_ids"] == [],
        "existing_full_not_duplicated": existing_out[0]["atoms"][0]["fully_covered_gold_ids"] == ["g"]
            and existing_assign == [],
    }
    return {"checks": checks, "passed": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "run"))
    parser.add_argument("--score-dir", type=Path, default=DATA / "_dev-v2-score")
    parser.add_argument("--output-dir", type=Path, default=DATA / "alt-partition")
    args = parser.parse_args()
    checked = selftest()
    print(json.dumps(checked, ensure_ascii=False))
    if not checked["passed"]:
        raise SystemExit(2)
    if args.command == "check":
        return
    source = args.score_dir / "data"
    output_dir = args.output_dir / "data"
    read = lambda name: [json.loads(line) for line in (source / name).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = read("annotations-v1.jsonl")
    cases = read("cases-v1.jsonl")
    raw_meta = {}
    for arm in ("P0", "A1", "A2"):
        for raw in read(f"raw-{arm}.jsonl"):
            raw_meta[blind_id_of(raw["case_id"], arm, raw["attempt"], raw["content_sha256"])] = {
                "case_id": raw["case_id"], "split": raw["split"]}
    for row in rows:
        row.update(raw_meta[row["blind_id"]])
    output, assignments = apply(rows, {case["case_id"]: case for case in cases})
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("cases-v1.jsonl", "raw-P0.jsonl", "raw-A1.jsonl", "raw-A2.jsonl", "call-ledger.jsonl"):
        shutil.copyfile(source / name, output_dir / name)
    (output_dir / "annotations-v1.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    payload = {"schema": "alt-partition-sensitivity-v1", "assignments": assignments,
               "input_sha256": {
                   "annotations": hashlib.sha256((source / "annotations-v1.jsonl").read_bytes()).hexdigest(),
                   "cases": hashlib.sha256((source / "cases-v1.jsonl").read_bytes()).hexdigest(),
                   "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    (args.output_dir / "assignments.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(ROOT / "score_atomic_memory_20261025.py", args.output_dir / "score_atomic_memory_20261025.py")
    print(json.dumps({"output": str(output_dir), "assignments": len(assignments)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
