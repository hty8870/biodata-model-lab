from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_conflict_data as B

RAW_DIR = ROOT / "data/authors/pilot-r2p1-train/raw"
OUT = ROOT / "data/authors/confirmed-r2p1"
AUDIT = ROOT / "audits/style-diagnostic-replay-20261025.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    specs = {s["family_id"]: s for s in B.K.read_jsonl(B.K.SPECS_PATH)}
    cases = []
    payloads = {}
    raw_paths = sorted(RAW_DIR.glob("*.json"))
    assert len(raw_paths) == 2
    for raw_path in raw_paths:
        raw = B.K.read_json(raw_path)
        spec = specs[raw["family_id"]]
        assert raw["succeeded"] and raw["finish_reason"] == "stop"
        assert raw["spec_sha256"] == B.K.sha256_json(spec)
        assert raw["prompt_sha256"] == B.K.sha256_text(B.build_prompt(spec))
        assert raw["response_sha256"] == B.K.sha256_text(raw["response_text"])
        payload = B.parse_author_response(raw["response_text"], spec)
        errors = B.validate_author_payload(spec, payload)
        assert not errors, errors
        warnings = []
        for item in spec["items"]:
            if item["kind"] == "dyad_flip":
                continue
            for side in ("old", "new"):
                text = payload["items"][item["pair_id"]][side]
                values = B.S.surface_warnings(text, item[side], (item.get("surface") or {}).get(side))
                if values:
                    warnings.append({"pair_id": item["pair_id"], "side": side, "warnings": values})
        payloads[spec["family_id"]] = (spec, payload, raw, raw_path)
        cases.append({"family_id": spec["family_id"], "source_raw": raw_path.relative_to(ROOT).as_posix(),
                      "source_raw_sha256": B.K.sha256_file(raw_path), "errors": errors, "warnings": warnings})
    warned5 = {(x["pair_id"], x["side"]) for x in cases[1]["warnings"]}
    assert {(f"cf-005-{i}", "new") for i in ("s1", "s2", "c3")} <= warned5
    spec, payload, raw, raw_path = payloads["cf-004"]
    setting = B.S.SETTINGS_BY_ID[spec["setting"]]
    first = next(i for i in spec["items"] if i["kind"] == "shared")
    equal_payload = copy.deepcopy(payload)
    equal_payload["items"][first["pair_id"]]["new"] = equal_payload["items"][first["pair_id"]]["old"]
    assert any("old == new" in e for e in B.validate_author_payload(spec, equal_payload))
    empty = dict(first["old"], items_scope=[])
    assert any("empty_applicability" in e for e in B.S.factor_problems(empty, setting))
    dyad = next(i for i in spec["items"] if i["kind"] == "dyad_base" and i["gate"]["new"]["exact_tokens"])
    token = dyad["gate"]["new"]["exact_tokens"][0]
    doubled = payload["items"][dyad["pair_id"]]["new"] + token
    assert any("exact token" in e for e in B.S.check_text_gates(
        doubled, dyad["new"], setting, exact_tokens=(token,)))
    numeric = next(i for i in spec["items"] if i["kind"] == "shared" and i["gate"]["old"]["numbers"])
    number = str(numeric["gate"]["old"]["numbers"][0])
    missing = payload["items"][numeric["pair_id"]]["old"].replace(number, "")
    assert any("missing number" in e for e in B.S.check_text_gates(missing, numeric["old"], setting))
    checks = {"both_real_payloads_pass": True, "known_mixed_style_is_reported": True,
              "identical_pair_still_rejected": True, "empty_scope_still_rejected": True,
              "counterfactual_exact_fragment_still_enforced": True, "missing_number_still_rejected": True}
    if not args.check_only:
        for family, (spec, payload, raw, raw_path) in payloads.items():
            target = OUT / f"{family}.json"
            if not B.verified_record(target, spec, B.K.AUTHOR_PLAN["train"]):
                identity = {key: raw.get(key) for key in (
                    "provider", "model", "base_url_sha256", "max_tokens", "thinking", "finish_reason", "latency_s")}
                identity.update({"channel": "chat", "replayed_without_model_call": True,
                                 "source_raw": raw_path.relative_to(ROOT).as_posix(),
                                 "source_raw_sha256": B.K.sha256_file(raw_path),
                                 "text_gate_sha256": B.K.sha256_file(Path(B.S.__file__))})
                B.write_author_record(OUT, spec, payload, model=raw["model"], prompt=B.build_prompt(spec),
                                      usage=raw["usage"], source="author", identity=identity)
            record = B.K.read_json(target)
            rows, errors = B.rows_for_family(spec, record)
            assert not errors and len(rows) == 18, errors
            case = next(c for c in cases if c["family_id"] == family)
            case.update({"accepted_record": target.relative_to(ROOT).as_posix(),
                         "accepted_record_sha256": B.K.sha256_file(target), "assembled_rows": len(rows)})
        B.K.write_json_atomic(AUDIT, {"schema": "style-diagnostic-replay/1", "new_model_calls": 0,
            "new_tokens": 0, "reused_calls_already_accounted": 2, "cases": cases, "checks": checks,
            "specs_sha256": B.K.sha256_file(B.K.SPECS_PATH),
            "text_gate_sha256": B.K.sha256_file(Path(B.S.__file__)),
            "runner_sha256": B.K.sha256_file(Path(__file__)),
            "policy": "required surface tokens retained; mixed style is diagnostic, not a semantic rejection"})
    print(json.dumps({"selftest_checks": len(checks), "passed": sum(checks.values()),
                      "families": len(cases), "new_model_calls": 0, "check_only": args.check_only,
                      "cases": cases}, ensure_ascii=False))


if __name__ == "__main__":
    main()
