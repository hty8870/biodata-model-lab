from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OLD_ANNOTATIONS = [f"annotations-dev-s{i}.jsonl" for i in range(1, 5)]
ATOM_FIELDS = ("entailment", "persistence_appropriate", "non_atomic", "over_split",
               "context_errors", "matched_forbidden_ids", "duplicate_group",
               "core_entailed_gold_ids", "fully_covered_gold_ids", "alternative_partition", "notes")


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def key(candidate: dict) -> str:
    return canonical([candidate.get("text"), candidate.get("evidence") or [],
                      candidate.get("tier"), candidate.get("trigger")])


def blind_id(case_id: str, arm: str, record: dict) -> str:
    payload = f"{case_id}|{arm}|{record['attempt']}|{record['content_sha256']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def old_index() -> dict:
    index = {}
    for arm in ("A1", "A2"):
        for raw in read_jsonl(DATA / f"raw-{arm}.jsonl"):
            for candidate in raw["parsed_candidates"]:
                index.setdefault((raw["case_id"], arm, key(candidate)), []).append(candidate)
    return index


def annotation_index() -> dict:
    result = {}
    for name in OLD_ANNOTATIONS:
        for row in read_jsonl(DATA / name):
            for atom in row["atoms"]:
                result[atom["candidate_id"]] = atom
    return result


def placeholder(candidate: dict) -> dict:
    return {"candidate_id": candidate["candidate_id"], "entailment": "partial",
            "persistence_appropriate": False, "non_atomic": False, "over_split": False,
            "context_errors": [], "matched_forbidden_ids": [], "duplicate_group": None,
            "core_entailed_gold_ids": [], "fully_covered_gold_ids": [],
            "alternative_partition": None, "notes": "v2 新文本，待两阶段复核"}


def build() -> tuple[list[dict], list[dict], dict]:
    annotations = annotation_index()
    candidates = old_index()
    raw_map = {}
    for arm in ("A1", "A2"):
        for raw in read_jsonl(DATA / f"raw-{arm}-v2.jsonl"):
            raw_map[blind_id(raw["case_id"], arm, raw)] = (raw["case_id"], raw["split"], arm)
    blind = read_jsonl(DATA / "blind-v2-stage2.jsonl")
    cases = {case["case_id"]: case for case in read_jsonl(DATA / "cases-v1.jsonl")}
    working, review, stats = [], [], {"reused": 0, "needs_review": 0, "empty_sets": 0,
                                      "fully_reused_sets": 0, "partial_sets": 0, "new_sets": 0}
    for row in blind:
        case_id, split, arm = raw_map[row["blind_id"]]
        atoms, changed = [], []
        for candidate in row["candidates"]:
            matches = [old for old in candidates.get((case_id, arm, key(candidate)), [])
                       if old["candidate_id"] in annotations]
            if len(matches) == 1:
                old_annotation = copy.deepcopy(annotations[matches[0]["candidate_id"]])
                old_annotation["candidate_id"] = candidate["candidate_id"]
                old_annotation["notes"] = str(old_annotation.get("notes") or "") + " [v2同文复核]"
                atoms.append(old_annotation)
                stats["reused"] += 1
            else:
                atoms.append(placeholder(candidate))
                changed.append(candidate)
                stats["needs_review"] += 1
        pending = bool(changed)
        working.append({"blind_id": row["blind_id"], "case_id": case_id, "split": split,
                        "stage1_complete": not pending, "stage2_complete": not pending,
                        "adjudicated": not pending, "atoms": atoms})
        if changed:
            review.append({"blind_id": row["blind_id"], "conversations": row["conversations"],
                           "forbidden_inferences": row["forbidden_inferences"],
                           "one_off_hints": row["one_off_hints"], "candidates": changed,
                           "gold_obligations": row["gold_obligations"]})
        if not row["candidates"]:
            stats["empty_sets"] += 1
        elif len(changed) == len(row["candidates"]):
            stats["new_sets"] += 1
        elif not changed:
            stats["fully_reused_sets"] += 1
        else:
            stats["partial_sets"] += 1
    return working, review, stats


def selftest() -> dict:
    global OLD_ANNOTATIONS, DATA
    old_data = DATA
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            DATA = Path(tmp)
            old = {"candidate_id": "old", "text": "甲", "evidence": ["e"], "tier": "explicit"}
            new = {"candidate_id": "new", "text": "甲", "evidence": ["e"], "tier": "explicit"}
            changed = {"candidate_id": "changed", "text": "乙", "evidence": ["e"], "tier": "explicit"}
            (DATA / "raw-A1.jsonl").write_text(json.dumps({"case_id": "c", "parsed_candidates": [old]}) + "\n", encoding="utf-8")
            (DATA / "raw-A2.jsonl").write_text(json.dumps({"case_id": "c", "parsed_candidates": [old]}) + "\n", encoding="utf-8")
            (DATA / "raw-A1-v2.jsonl").write_text(json.dumps({"case_id": "c", "split": "dev", "attempt": 1,
                "content_sha256": "x", "parsed_candidates": [new, changed]}) + "\n", encoding="utf-8")
            (DATA / "raw-A2-v2.jsonl").write_text("", encoding="utf-8")
            annotation = {field: (True if field == "persistence_appropriate" else
                                  False if field in {"non_atomic", "over_split"} else
                                  "supported" if field == "entailment" else
                                  [] if field in {"context_errors", "matched_forbidden_ids",
                                                 "core_entailed_gold_ids", "fully_covered_gold_ids"} else
                                  "note" if field == "notes" else None)
                          for field in ATOM_FIELDS}
            annotation["candidate_id"] = "old"
            (DATA / OLD_ANNOTATIONS[0]).write_text(json.dumps({"atoms": [annotation]}) + "\n", encoding="utf-8")
            for name in OLD_ANNOTATIONS[1:]:
                (DATA / name).write_text("", encoding="utf-8")
            blind = blind_id("c", "A1", {"attempt": 1, "content_sha256": "x"})
            (DATA / "blind-v2-stage2.jsonl").write_text(json.dumps({"blind_id": blind,
                "conversations": [], "forbidden_inferences": [], "one_off_hints": [],
                "candidates": [new, changed], "gold_obligations": []}) + "\n", encoding="utf-8")
            (DATA / "cases-v1.jsonl").write_text(json.dumps({"case_id": "c", "split": "dev"}) + "\n", encoding="utf-8")
            working, review, stats = build()
            checks = {"exact_reused": stats["reused"] == 1 and working[0]["atoms"][0]["candidate_id"] == "new",
                      "changed_not_guessed": stats["needs_review"] == 1 and review[0]["candidates"][0]["text"] == "乙",
                      "placeholder_not_final": working[0]["adjudicated"] is False}
            duplicate = [old, dict(old, candidate_id="old2")]
            (DATA / "raw-A1.jsonl").write_text(json.dumps({"case_id": "c", "parsed_candidates": duplicate}) + "\n", encoding="utf-8")
            conflicting = copy.deepcopy(annotation)
            conflicting["candidate_id"] = "old2"
            conflicting["entailment"] = "partial"
            (DATA / OLD_ANNOTATIONS[1]).write_text(json.dumps({"atoms": [conflicting]}) + "\n", encoding="utf-8")
            _, _, duplicate_stats = build()
            checks["duplicates_not_reused"] = duplicate_stats["reused"] == 0 and duplicate_stats["needs_review"] == 2
            return {"checks": checks, "passed": all(checks.values())}
    finally:
        DATA = old_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "run"))
    args = parser.parse_args()
    checked = selftest()
    print(json.dumps(checked, ensure_ascii=False))
    if not checked["passed"]:
        raise SystemExit(2)
    if args.command == "check":
        return
    working, review, stats = build()
    (DATA / "annotations-v2-dev-working.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in working), encoding="utf-8")
    for stage in (1, 2):
        rows = [dict(row) for row in review]
        if stage == 1:
            for row in rows:
                row.pop("gold_obligations", None)
        (DATA / f"v2-changed-stage{stage}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    payload = {"stats": stats, "input_sha256": {
        "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "annotations": {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in OLD_ANNOTATIONS},
        "blind": hashlib.sha256((DATA / "blind-v2-stage2.jsonl").read_bytes()).hexdigest()}}
    (DATA / "v2-reuse-summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
