"""Two prompt-only checks, then a read-only full-error summary of the authorized pilot."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile

REL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REL))
import build_conflict_data as B

K = B.K
SNAPSHOT = REL / "_归档" / "prompt-pre-r2p1-20261025"
OLD = REL / "data" / "authors" / "pilot-r2-train"
NEW = REL / "data" / "authors" / "pilot-r2p1-train"
SPECS_SHA = "30964fe030d3781dcdf0a9f400bc5c4628b2a7128a5354584acfa5b5a9928c65"
specs = K.read_jsonl(K.SPECS_PATH)
selected = [spec for spec in specs if spec["split"] == "train"][:2]


def checks():
    def nodes(path):
        return {getattr(node, "name", ast.dump(node.targets[0]) if isinstance(node, ast.Assign) else str(i)):
                ast.dump(node, include_attributes=False)
                for i, node in enumerate(ast.parse(path.read_text(encoding="utf-8")).body)}
    before, after = nodes(SNAPSHOT / "build_conflict_data.py"), nodes(REL / "build_conflict_data.py")
    changed = {name for name in set(before) | set(after) if before.get(name) != after.get(name)}
    allowed = {"build_prompt", "_required_phrases", "Name(id='AUTHOR_PROMPT_VERSION', ctx=Store())"}
    scope_ok = True
    exact_ok = True
    for spec in selected:
        digest = K.sha256_json(spec)
        prompt = B.build_prompt(spec)
        for item in spec["items"]:
            if item["kind"] == "dyad_flip":
                continue
            block = prompt.split("- pair_id: " + item["pair_id"] + "\n", 1)[1].split("- pair_id:", 1)[0]
            for side in ("old", "new"):
                names = item[side].get("items_scope")
                line = next(line for line in block.splitlines() if line.startswith(f"  {side} 必须出现："))
                if names:
                    scope_ok &= "必须逐字使用「只针对」" in line and all(name in line for name in names)
                for token in item["gate"][side]["exact_tokens"]:
                    exact_ok &= f"（逐字且唯一，最高优先级）「{token}」" in line and "恰好出现一次" in line
            if item["kind"] == "dyad_base":
                token = item["dyad"]["factor"]["from_render"]
                exact_ok &= f"最高优先级：new 必须逐字包含「{token}」且恰好一次" in block
        scope_ok &= digest == K.sha256_json(spec)
    K.SELFTEST_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=K.SELFTEST_DIR, prefix="prompt-r2p1-") as directory:
        spec = selected[0]
        record = B.write_author_record(Path(directory), spec, B.fixture_author(spec),
            model=K.AUTHOR_PLAN["train"][0], prompt=B.build_prompt(spec), usage={"calls": 0}, source="author")
        path = Path(directory) / (spec["family_id"] + ".json")
        exact_ok &= B.verified_record(path, spec, K.AUTHOR_PLAN["train"])
        K.write_json_atomic(path, dict(record, prompt_version="conflict-author-prompt-v2.2-r2"))
        exact_ok &= not B.verified_record(path, spec, K.AUTHOR_PLAN["train"])
    result = {"checks": [{"name": "all_items_scope_exposed_without_spec_or_gate_change", "ok": bool(scope_ok and changed == allowed)},
                          {"name": "exact_fragment_priority_and_old_prompt_resume_rejection", "ok": bool(exact_ok)}],
              "changed_top_level_nodes": sorted(changed), "spec_sha256": K.sha256_file(K.SPECS_PATH),
              "prompt_identity": B.POLICY_IDENTITY}
    K.write_json_atomic(REL / "audits" / "prompt-r2p1-selftest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(check["ok"] for check in result["checks"]) and result["spec_sha256"] == SPECS_SHA else 1


def inspect(directory):
    by_id = {spec["family_id"]: spec for spec in specs}
    results = []
    for path in sorted((directory / "raw").glob("*.json")):
        raw = K.read_json(path)
        spec = by_id[raw["family_id"]]
        payload_errors, assembly_errors = [], []
        try:
            payload = B.parse_author_response(raw.get("response_text"), spec)
            payload_errors = B.validate_author_payload(spec, payload)
            record = {"background": payload["background"], "model": raw.get("model"), "source": "author",
                      "items": [dict(pair_id=key, **value) for key, value in payload["items"].items()]}
            _, assembly_errors = B.rows_for_family(spec, record)
        except Exception as error:
            payload_errors.append(f"{type(error).__name__}: {error}")
        results.append({key: raw.get(key) for key in ("family_id", "succeeded", "finish_reason", "usage", "latency_s",
                         "model", "thinking", "max_tokens", "prompt_version", "prompt_sha256", "spec_sha256")}
                       | {"raw": str(path), "raw_sha256": K.sha256_file(path),
                          "author_payload_errors": payload_errors, "assembly_errors": assembly_errors})
    return results


def summarize():
    cases = inspect(NEW)
    ledger = K.read_jsonl(NEW / "calls.jsonl")
    old_files = [OLD / "calls.jsonl", *sorted((OLD / "raw").glob("*.json"))]
    retained = all(K.sha256_file(path) == K.sha256_file(SNAPSHOT / "pilot-r2-train" / path.relative_to(OLD))
                   for path in old_files)
    valid_budget = len(cases) == len(ledger) == 2 and {case["family_id"] for case in cases} == {s["family_id"] for s in selected}
    passed = valid_budget and all(case["succeeded"] and not case["author_payload_errors"] and not case["assembly_errors"] for case in cases)
    report = {"pilot_exit_code": int(sys.argv[2]), "passed_local_validation": passed, "cases": cases,
              "calls": len(ledger), "written": len(list(NEW.glob("cf-*.json"))), "budget_calls": 2,
              "usage_total": {key: sum((case.get("usage") or {}).get(key) or 0 for case in cases)
                              for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
              "old_pilot": inspect(OLD), "old_raw_and_ledger_unchanged": retained,
              "spec_unchanged": K.sha256_file(K.SPECS_PATH) == SPECS_SHA,
              "spec_sha256": K.sha256_file(K.SPECS_PATH), "source_sha256": K.sha256_file(REL / "build_conflict_data.py"),
              "prompt_identity": B.POLICY_IDENTITY, "training_updates": 0,
              "human_semantic_review_passed": False}
    K.write_json_atomic(REL / "audits" / "pilot-r2p1-validation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed and retained and report["spec_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(summarize() if len(sys.argv) > 1 and sys.argv[1] == "after" else checks())
