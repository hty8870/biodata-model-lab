from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = DATA / "_dev-v2-score" / "data"
ARMS = ("P0", "A1", "A2")
OLD_ANNOTATIONS = [f"annotations-dev-s{i}.jsonl" for i in range(1, 5)]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def blind_id(case_id: str, arm: str, record: dict) -> str:
    payload = f"{case_id}|{arm}|{record['attempt']}|{record['content_sha256']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def merge(working: list[dict], changed: list[dict], p0: list[dict], expected_rows: int = 90) -> list[dict]:
    final_atoms = [atom for row in changed for atom in row["atoms"]]
    final_by_id = {atom["candidate_id"]: atom for atom in final_atoms}
    if len(final_by_id) != len(final_atoms):
        raise ValueError("duplicate final candidate_id")
    merged = []
    for row in working:
        row = dict(row)
        row["atoms"] = [dict(atom) for atom in row["atoms"]]
        for atom in row["atoms"]:
            if atom["candidate_id"] in final_by_id:
                row_atom = final_by_id[atom["candidate_id"]]
                atom.clear()
                atom.update(row_atom)
            elif str(atom["notes"]).startswith("v2 新文本"):
                raise ValueError(f"missing final {atom['candidate_id']}")
        if any(str(atom["notes"]).startswith("v2 新文本") for atom in row["atoms"]):
            raise ValueError(f"placeholder {row['blind_id']}")
        row["stage1_complete"] = row["stage2_complete"] = row["adjudicated"] = True
        merged.append(row)
    rows = sorted(p0 + merged, key=lambda row: row["blind_id"])
    if len(rows) != expected_rows:
        raise ValueError(f"rows={len(rows)} != {expected_rows}")
    for row in rows:
        for atom in row["atoms"]:
            if atom["entailment"] == "supported" and atom["context_errors"]:
                raise ValueError(f"supported with context errors: {atom['candidate_id']}")
    return rows


def p0_annotations() -> list[dict]:
    ids = set()
    for raw in read_jsonl(DATA / "raw-P0.jsonl"):
        ids.add(blind_id(raw["case_id"], "P0", raw))
    return [row for name in OLD_ANNOTATIONS for row in read_jsonl(DATA / name)
            if row["blind_id"] in ids]


def prepare() -> dict:
    working = read_jsonl(DATA / "annotations-v2-dev-working.jsonl")
    changed = [row for name in ("annotations-v2-final-s1.jsonl", "annotations-v2-final-s2.jsonl",
                                "annotations-v2-final-s3.jsonl")
               for row in read_jsonl(DATA / name)]
    rows = merge(working, changed, p0_annotations())
    raw_meta = {}
    for arm in ARMS:
        version = "P0" if arm == "P0" else f"{arm}-v2"
        for raw in read_jsonl(DATA / f"raw-{version}.jsonl"):
            raw_meta[blind_id(raw["case_id"], arm, raw)] = {
                "case_id": raw["case_id"], "split": raw["split"]}
    for row in rows:
        row.update(raw_meta[row["blind_id"]])
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DATA / "_dev-score" / "data" / "annotations-v1.jsonl",
                    ROOT / "_归档" / "annotations-v2-dev-v1.jsonl")
    (DATA / "annotations-v2.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    for source, target in (("annotations-v2.jsonl", "annotations-v1.jsonl"),
                           ("raw-P0.jsonl", "raw-P0.jsonl"),
                           ("raw-A1-v2.jsonl", "raw-A1.jsonl"),
                           ("raw-A2-v2.jsonl", "raw-A2.jsonl")):
        shutil.copyfile(DATA / source, OUT / target)
    p0_attempts = {(row["case_id"], row["attempt"]) for row in read_jsonl(DATA / "raw-P0.jsonl")}
    p0_ledger = [row for row in read_jsonl(DATA / "_dev-score" / "data" / "call-ledger.jsonl")
                 if row["arm"] == "P0" and (row["case_id"], row["attempt"]) in p0_attempts]
    ledger = p0_ledger + read_jsonl(DATA / "call-ledger-v2.jsonl")
    if len(p0_ledger) != len(p0_attempts):
        raise ValueError(f"P0 ledger={len(p0_ledger)} != raw attempts={len(p0_attempts)}")
    (OUT / "call-ledger.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ledger), encoding="utf-8")
    shutil.copyfile(DATA / "_dev-score" / "data" / "cases-v1.jsonl", OUT / "cases-v1.jsonl")
    shutil.copyfile(ROOT / "score_atomic_memory_20261025.py", OUT.parent / "score_atomic_memory_20261025.py")
    return {"rows": len(rows), "p0": len(rows) - len(working), "changed_output_sets": len(changed),
            "output": str(OUT)}


def selftest() -> dict:
    reused = {"candidate_id": "old", "entailment": "supported", "persistence_appropriate": True,
              "non_atomic": False, "over_split": False, "context_errors": [], "matched_forbidden_ids": [],
              "duplicate_group": None, "core_entailed_gold_ids": [], "fully_covered_gold_ids": [],
              "alternative_partition": None, "notes": "old"}
    changed = {"candidate_id": "new", "entailment": "partial", "persistence_appropriate": True,
               "non_atomic": False, "over_split": False, "context_errors": ["scope"],
               "matched_forbidden_ids": [], "duplicate_group": None,
               "core_entailed_gold_ids": [], "fully_covered_gold_ids": [],
               "alternative_partition": None, "notes": "new"}
    working = [{"blind_id": "b", "case_id": "c", "split": "dev", "stage1_complete": False,
                "stage2_complete": False, "adjudicated": False,
                "atoms": [reused, dict(changed, notes="v2 新文本，待两阶段复核")]}]
    rows = merge(working, [{"atoms": [changed]}], [], expected_rows=1)
    checks = {"replacement": rows[0]["atoms"][1]["entailment"] == "partial"
              and rows[0]["atoms"][0]["entailment"] == "supported",
              "flags": rows[0]["adjudicated"] is True}
    override = merge([dict(working[0], atoms=[reused])],
                     [{"atoms": [dict(changed, candidate_id="old")]}], [], expected_rows=1)
    checks["final_overrides_reused"] = override[0]["atoms"][0]["entailment"] == "partial"
    try:
        merge(working, [{"atoms": [changed, changed]}], [], expected_rows=1)
    except ValueError as exc:
        checks["duplicate_final_rejected"] = "duplicate final" in str(exc)
    else:
        checks["duplicate_final_rejected"] = False
    try:
        merge(working, [], [], expected_rows=1)
    except ValueError as exc:
        checks["missing_rejected"] = "missing final" in str(exc)
    else:
        checks["missing_rejected"] = False
    checks["context_rejected"] = False
    try:
        merge([dict(working[0], atoms=[dict(reused, context_errors=["scope"])])], [], [], expected_rows=1)
    except ValueError:
        checks["context_rejected"] = True
    return {"checks": checks, "passed": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "run"))
    args = parser.parse_args()
    checked = selftest()
    print(json.dumps(checked, ensure_ascii=False))
    if not checked["passed"]:
        raise SystemExit(2)
    if args.command == "run":
        print(json.dumps(prepare(), ensure_ascii=False))


if __name__ == "__main__":
    main()
