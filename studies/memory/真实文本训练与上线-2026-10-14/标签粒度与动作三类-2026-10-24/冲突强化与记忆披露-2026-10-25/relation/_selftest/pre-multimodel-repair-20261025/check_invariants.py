from __future__ import annotations

import json
import sys
from pathlib import Path

REL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REL))

import build_conflict_data as B
import repair_author_outputs_20261025 as R


def main() -> None:
    specs = {s["family_id"]: s for s in B.K.read_jsonl(B.K.SPECS_PATH)}
    checks = []

    def ok(condition: bool, name: str) -> None:
        assert condition, name
        checks.append(name)

    accepted = sorted(R.AUTHORS.glob("*.json"))
    ok(len(accepted) == 36, "36 accepted records")
    for path in accepted:
        record = B.K.read_json(path)
        family = record["family_id"]
        spec = specs[family]
        ok(record["split"] in ("train", "dev"), f"{family} split")
        ok(record["model"] == "deepseek-v4-flash", f"{family} model")
        ok(record["identity"]["channel"] == "chat", f"{family} channel")
        ok(B.verified_record(path, spec, B.K.AUTHOR_PLAN[record["split"]]), f"{family} verified")
        ok(not B.rows_for_family(spec, record)[1], f"{family} rows clean")
    for family in ("cf-001", "cf-003"):
        ok(not (R.AUTHORS / f"{family}.json").exists(), f"{family} not accepted")

    pairs = 0
    regenerated = 0
    for request_path in sorted(R.REPAIRS.glob("*.request.json")):
        family = request_path.name.removesuffix(".request.json")
        outcome_path = R.REPAIRS / f"{family}.outcome.json"
        ok(outcome_path.is_file(), f"{family} outcome")
        request = B.K.read_json(request_path)
        outcome = B.K.read_json(outcome_path)
        raw_path = R.ROOT / request["base_raw"]
        raw = B.K.read_json(raw_path)
        spec = specs[family]
        ok(request["status"] == "reserved", f"{family} reserved")
        ok(raw["split"] == spec["split"] and raw["model"] == "deepseek-v4-flash", f"{family} raw model")
        ok(request["base_raw_sha256"] == B.K.sha256_file(raw_path), f"{family} base raw sha")
        ok(request["spec_sha256"] == raw["spec_sha256"] == B.K.sha256_json(spec), f"{family} spec sha")
        payload = B.parse_author_response(raw["response_text"], spec)
        errors = R.errors_for(spec, payload, raw["model"])
        ok(request["prompt_sha256"] == B.K.sha256_text(request["prompt"]), f"{family} request prompt sha")
        if errors:
            changes = R.changes_for(spec, errors)
            regenerated += request["prompt_sha256"] == B.K.sha256_text(
                R.prompt_for(spec, payload, errors, changes))
        ok(outcome["provider"] == raw["provider"] and outcome["model"] == raw["model"],
           f"{family} outcome identity")
        pairs += 1
    ok(pairs == 11, "11 DS repair pairs")
    for family in ("cf-001", "cf-003"):
        ok(not (R.REPAIRS / f"{family}.request.json").exists()
           and not (R.REPAIRS / f"{family}.outcome.json").exists(), f"{family} no repair files")
    print(json.dumps({"checks": len(checks), "accepted": len(accepted),
                      "repair_pairs": pairs, "prompt_regeneration_matches": regenerated,
                      "passed": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
