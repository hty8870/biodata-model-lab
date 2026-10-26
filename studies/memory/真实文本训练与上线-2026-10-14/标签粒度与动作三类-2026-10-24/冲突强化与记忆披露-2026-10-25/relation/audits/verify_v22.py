"""已裁决双合同的离线验收；保留历史红审计，不运行 author/train/smoke。"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RELATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELATION))
import conflict_common as K


def main():
    run = RELATION / "audits" / (f"verification-{K.CONTROL_POLICY_VERSION}-" + time.strftime("%Y%m%d-%H%M%S"))
    run.mkdir(parents=True, exist_ok=False)
    cli_processed = K.SELFTEST_DIR / run.name / "cli-processed"
    cli_audit = run / "fixture-audit-tokenizer.json"
    power_path = run / "mdes-prospective.json"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TEMP=str(K.SELFTEST_DIR),
               TMP=str(K.SELFTEST_DIR))
    K.SELFTEST_DIR.mkdir(parents=True, exist_ok=True)
    commands = [
        (["build_conflict_data.py", "specs"], 0),
        (["build_conflict_data.py", "selftest"], 0),
        (["composite_cases.py", "specs"], 0),
        (["composite_cases.py", "selftest"], 0),
        (["build_conflict_data.py", "fixture-bundle"], 0),
        (["train_conflict_arms.py", "selftest"], 0),
        (["score_conflict.py", "selftest"], 0),
        (["build_conflict_data.py", "assemble", "--fixture", "--specs", "data/specs/specs.jsonl",
          "--out", str(cli_processed)], 0),
        (["build_conflict_data.py", "audit", "--mode", "fixture", "--processed", str(cli_processed),
          "--specs", "data/specs/specs.jsonl", "--with-tokenizer", "--out", str(cli_audit)], 0),
        (["train_conflict_arms.py", "plan-dry-run", "--seed", "13"], 0),
        (["train_conflict_arms.py", "plan-dry-run", "--seed", "42"], 0),
        (["score_conflict.py", "mdes-prospective", "--out", str(power_path)], 0),
    ]
    archive = RELATION / "_归档" / "v2.2-takeover-20261025" / "authors-pre-v2.2" / "deepseek-v4-flash"
    ledger = archive / "calls.jsonl"
    ledger_sha_before = K.sha256_file(ledger)
    historical = [RELATION / "audits" / "fixture-dataset-audit-v2.2.json",
                  RELATION / "audits" / "verification-v2.2.json"]
    historical.append(Path(K.read_json(historical[1])["audit_report"]))
    historical.extend([RELATION / "audits" / "fixture-dataset-audit-v2.2-r1.json",
                       RELATION / "audits" / "verification-control-matching-v2.2-r1.json"])
    historical.append(Path(K.read_json(historical[-1])["audit_report"]))
    historical_shas = {str(path): K.sha256_file(path) for path in historical}
    records = []
    for index, (arguments, expected) in enumerate(commands, 1):
        started = time.perf_counter()
        command = [sys.executable, "-B", *arguments]
        result = subprocess.run(command, cwd=RELATION, env=env, capture_output=True,
                                encoding="utf-8", timeout=240)
        stem = f"{index:02d}-" + Path(arguments[0]).stem + "-" + (arguments[1] if len(arguments) > 1 else "replay")
        (run / (stem + ".stdout.log")).write_text(result.stdout, encoding="utf-8")
        (run / (stem + ".stderr.log")).write_text(result.stderr, encoding="utf-8")
        entry = {"command": command, "exit_code": result.returncode, "expected_exit": expected,
                 "matched_expected_exit": result.returncode == expected,
                 "seconds": round(time.perf_counter() - started, 3), "log": stem + ".stdout.log"}
        try:
            output = json.loads(result.stdout)
            if "checks" in output and isinstance(output["checks"], list):
                entry["selftest_checks"] = len(output["checks"])
                entry["selftest_failures"] = [c for c in output["checks"] if not c["ok"]]
            if "ok" in output:
                entry["reported_ok"] = output["ok"]
        except json.JSONDecodeError:
            pass
        records.append(entry)
        print(json.dumps({"step": index, **entry}, ensure_ascii=False), flush=True)
        if result.returncode != expected:
            break
    report = K.read_json(cli_audit) if cli_audit.is_file() else {}
    contrast = report.get("checks", {}).get("control_matching", {})
    failed = [name for name, entry in report.get("checks", {}).items() if not entry["ok"]]
    equivalent = contrast.get("equivalent_layer", {})
    non_equivalent = contrast.get("non_equivalent_layer", {})
    contracts_verified = (not failed and report.get("ok") is True and len(report["checks"]) == 29
                          and equivalent.get("rows") == equivalent.get("accepted") == 120
                          and non_equivalent.get("rows") == non_equivalent.get("accepted") == 240
                          and non_equivalent.get("semantic_pair_twins") == 0
                          and non_equivalent.get("minimum_effective_factor_changes", 0) >= 2
                          and non_equivalent.get("invalid_domains") == 0
                          and report.get("control_policy_version") == K.CONTROL_POLICY_VERSION)
    import build_conflict_data as B
    import conflict_semantics as S
    from collections import Counter
    before_path = RELATION / "_归档" / "control-policy-pre-r2-20261025" / "specs.jsonl"
    before = K.read_jsonl(before_path)
    after = K.read_jsonl(K.SPECS_PATH)
    def issues(specs):
        return [{"id": item["pair_id"], "split": spec["split"], "side": side, "problems": problems}
                for spec in specs for item in spec["items"] for side in ("old", "new")
                for problems in [S.factor_problems(item[side], S.SETTINGS_BY_ID[spec["setting"]])]
                if problems]
    issues_before, issues_after = issues(before), issues(after)
    before_comparisons = B.control_spec_contrasts(before, include_accepted=True)
    before_by_id = {entry["id"]: entry for entry in before_comparisons}
    changes = []
    fine_label_changes = []
    coarse_labels_unchanged = True
    for old_spec, new_spec in zip(before, after):
        for old, new in zip(old_spec["items"], new_spec["items"]):
            coarse_labels_unchanged &= old["expected_coarse"] == new["expected_coarse"]
            if old["expected_label"] != new["expected_label"]:
                fine_label_changes.append({"id": new["pair_id"], "before": old["expected_label"],
                                           "after": new["expected_label"]})
            if K.sha256_json(old) != K.sha256_json(new):
                causes = [p for entry in issues_before if entry["id"] == old["pair_id"]
                          for p in entry["problems"]]
                comparison = before_by_id.get(old["pair_id"])
                if comparison and len(comparison["effective_factor_changes"]) < 2:
                    causes.append("insufficient effective factor changes")
                changes.append({"id": new["pair_id"], "split": new_spec["split"],
                                "reasons": causes or ["semantic reallocation after candidate filtering"],
                                "old_factors_before": old["old"], "new_factors_before": old["new"],
                                "old_factors_after": new["old"], "new_factors_after": new["new"]})
    changed_items = [entry["id"] for entry in changes]
    backgrounds_changed = [new["family_id"] for old, new in zip(before, after)
                           if old["background_hint"] != new["background_hint"]]
    splits_unchanged = [s["split"] for s in before] == [s["split"] for s in after]
    reference_partitions_present = all(S.reference_partition_text(S.SETTINGS_BY_ID[s["setting"]])
                                       in s["background_hint"] for s in after)
    old_fine = [entry for entry in before_comparisons if entry["coarse"] != "equivalent"]
    recipe_by_id = {item["pair_id"]: item["recipe"] for spec in before for item in spec["items"]}
    old_fine_mismatch = {"n": len(old_fine), "mismatched": sum(not e["fine_subtype_matches"] for e in old_fine),
                         "by_recipe": dict(Counter(recipe_by_id[e["id"]] for e in old_fine
                                                   if not e["fine_subtype_matches"]))}
    minimal_spec_change = coarse_labels_unchanged and splits_unchanged and not issues_after and reference_partitions_present
    historical_unchanged = all(K.sha256_file(path) == sha for path, sha in historical_shas.items())
    selftest_checks = sum(record.get("selftest_checks", 0) for record in records)
    sources = [*RELATION.glob("*.py"), *Path(__file__).parent.glob("*.py")]
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    old_calls = K.read_jsonl(ledger)
    raw_shas = {K.sha256_text(r["response_text"]) if isinstance(r.get("response_text"), str) else None
                for path in (archive / "raw").glob("*.json") for r in [K.read_json(path)]}
    summary = {
        "schema": "conflict-offline-verification-v2.2-r2", "created": K.utc_now(),
        "python": sys.executable, "python_version": sys.version, "bytecode_disabled": sys.dont_write_bytecode,
        "run_directory": str(run), "commands": records, "selftest_checks": selftest_checks,
        "verification_ok": all(r["matched_expected_exit"] for r in records) and contracts_verified
                           and selftest_checks >= 106 and minimal_spec_change and historical_unchanged,
        "dataset_audit_ok": report.get("ok"), "both_control_contracts_verified": contracts_verified,
        "control_policy_version": K.CONTROL_POLICY_VERSION,
        "control_policy_sha256": K.sha256_json(K.CONTROL_POLICY),
        "protocol_sha256": K.sha256_json(K.PROTOCOL),
        "historical_unchanged": historical_unchanged, "historical_file_sha256": historical_shas,
        "spec_change": {"before_sha256": K.sha256_file(before_path), "changed_items": changed_items,
                        "coarse_labels_unchanged": coarse_labels_unchanged, "all_splits_unchanged": splits_unchanged,
                        "fine_label_changes": fine_label_changes, "item_diff": changes,
                        "backgrounds_changed": backgrounds_changed,
                        "reason": "existing candidates filtered by nonempty domains, active exceptions and effective changes; shared disjoint-reference background"},
        "semantic_review": {"issues_before": issues_before, "issues_after": issues_after,
                            "fine_composition_before": old_fine_mismatch,
                            "fine_composition_after": contrast.get("fine_composition"),
                            "reference_partitions_present": reference_partitions_present},
        "audit_checks": len(report.get("checks", {})),
        "audit_passed": sum(entry["ok"] for entry in report.get("checks", {}).values()),
        "audit_failed": failed, "audit_report": str(cli_audit),
        "spec_file_sha256": K.sha256_file(K.SPECS_PATH), "mdes_report": str(power_path),
        "factors_within_pair_excess": {split: block["pair"]["excess_rows"]
                                      for split, block in report.get("factor_twins", {}).get("within", {}).items()},
        "semantic_within_pair_excess": {split: block["pair"]["excess_rows"]
                                       for split, block in report.get("semantic_twins", {}).get("within", {}).items()},
        "code_sha256": {path.relative_to(RELATION).as_posix(): K.sha256_file(path) for path in sources},
        "archive_ledger_unchanged": ledger_sha_before == K.sha256_file(ledger),
        "legacy_calls": len(old_calls), "legacy_usage_total_tokens": sum(
            (entry.get("usage") or {}).get("total_tokens", 0) or 0 for entry in old_calls),
        "legacy_raw_files_available": len(list((archive / "raw").glob("*.json"))),
        "legacy_nonempty_response_shas_missing_from_available_raw": sum(
            bool(entry.get("response_sha256")) and entry["response_sha256"] not in raw_shas for entry in old_calls),
        "new_real_llm_calls": 0, "optimizer_training_updates": 0,
        "fixture_only": True, "production_or_real_model_result": False,
    }
    K.write_json_atomic(run / "summary.json", summary)
    K.write_json_atomic(RELATION / "audits" / f"verification-{K.CONTROL_POLICY_VERSION}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["verification_ok"] and summary["archive_ledger_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
