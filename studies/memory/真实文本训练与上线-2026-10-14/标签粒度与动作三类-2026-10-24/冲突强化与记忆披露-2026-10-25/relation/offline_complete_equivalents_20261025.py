from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_conflict_data as B
import repair_author_outputs_20261025 as R

ROOT = Path(__file__).resolve().parent


def reorder_plain_equivalent(text: str, item: dict) -> str:
    factor = item["new"]
    if item["expected_coarse"] != "equivalent" or factor["measure"]["kind"] != "none" \
            or factor["exception"]["kind"] != "none" or factor["window"]["kind"] != "long_term":
        raise ValueError("Only plain, long-term equivalent obligations may be reordered")
    clauses = text.rstrip("。").split("，")
    if len(clauses) < 2 or any(not part for part in clauses):
        raise ValueError("Expected separate non-empty conjunctive clauses")
    rotated = clauses[1:] + clauses[:1]
    assert sorted(clauses) == sorted(rotated)
    result = "，".join(rotated) + "。"
    assert B.K.normalize_text(result) != B.K.normalize_text(text)
    return result


def selftest() -> None:
    item = {"expected_coarse": "equivalent", "new": {"measure": {"kind": "none"},
            "exception": {"kind": "none"}, "window": {"kind": "long_term"}}}
    source = "同一用户必须核对正文，长期生效，只针对摘要与结论。"
    revised = reorder_plain_equivalent(source, item)
    assert revised == "长期生效，只针对摘要与结论，同一用户必须核对正文。"
    assert revised.count("必须") == source.count("必须") == 1
    for coarse, measure in (("conflict", "none"), ("equivalent", "at_least")):
        bad = {"expected_coarse": coarse, "new": dict(item["new"], measure={"kind": measure})}
        try:
            reorder_plain_equivalent(source, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe clause reordering was accepted")
    print(json.dumps({"selftest_checks": 4, "passed": 4}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    specs = {s["family_id"]: s for s in B.K.read_jsonl(B.K.SPECS_PATH)}
    sources = sorted((R.AUTHORS / "raw").glob("*.json")) + sorted((ROOT / "data/authors/pilot-r2p1-train/raw").glob("*.json"))
    reports = []
    for source_path in sources:
        raw = B.K.read_json(source_path)
        family = raw["family_id"]
        spec = specs[family]
        target = R.AUTHORS / f"{family}.json"
        if B.verified_record(target, spec, B.K.AUTHOR_PLAN[spec["split"]]):
            continue
        payload = B.parse_author_response(raw["response_text"], spec)
        assert raw["spec_sha256"] == B.K.sha256_json(spec)
        assert raw["prompt_sha256"] == B.K.sha256_text(B.build_prompt(spec))
        errors = R.errors_for(spec, payload, raw["model"])
        identity = {k: raw.get(k) for k in ("provider", "model", "base_url_sha256", "thinking", "max_tokens")}
        identity.update({"channel": "chat", "derivation": "offline-surface-completion-v1",
            "prompt_sha256_is_base_generation": True, "base_raw": source_path.relative_to(ROOT).as_posix(),
            "base_raw_sha256": B.K.sha256_file(source_path), "text_gate_sha256": B.K.sha256_file(Path(B.S.__file__))})
        usage = dict(raw["usage"])
        report = {"family_id": family, "base_errors": errors, "clause_reorders": [], "new_model_calls": 0}
        if errors:
            outcome_path = R.REPAIRS / f"{family}.outcome.json"
            request_path = R.REPAIRS / f"{family}.request.json"
            outcome = B.K.read_json(outcome_path)
            assert outcome["succeeded"]
            payload = R.apply_updates(payload, outcome["text"], R.changes_for(spec, errors))
            errors = R.errors_for(spec, payload, raw["model"])
            identity.update({"repair_request_sha256": B.K.sha256_file(request_path),
                             "repair_outcome_sha256": B.K.sha256_file(outcome_path)})
            usage = {k: raw["usage"][k] + outcome["usage"][k] for k in usage}
            if errors and all("old == new after normalisation" in e for e in errors):
                item_map = {i["pair_id"]: i for i in spec["items"]}
                for error in errors:
                    pair_id = error.split(":", 1)[0]
                    original = payload["items"][pair_id]["new"]
                    revised = reorder_plain_equivalent(original, item_map[pair_id])
                    payload["items"][pair_id]["new"] = revised
                    report["clause_reorders"].append({"pair_id": pair_id, "before": original, "after": revised,
                        "operation": "rotate unchanged conjunctive clauses; factor semantics and all tokens unchanged"})
                errors = R.errors_for(spec, payload, raw["model"])
        assert not errors, (family, errors)
        identity["clause_reorders"] = report["clause_reorders"]
        identity["derivation_runner_sha256"] = B.K.sha256_file(Path(__file__))
        B.write_author_record(R.AUTHORS, spec, payload, model=raw["model"], prompt=B.build_prompt(spec),
                              usage=usage, source="author", identity=identity)
        assert B.verified_record(target, spec, B.K.AUTHOR_PLAN[spec["split"]])
        report["accepted_record_sha256"] = B.K.sha256_file(target)
        reports.append(report)
    records = B.load_author_records(R.AUTHORS)
    counts = {split: sum(r["split"] == split for r in records.values()) for split in ("train", "dev")}
    assert counts == {"train": 24, "dev": 12}, counts
    all_errors = {family: B.rows_for_family(specs[family], record)[1] for family, record in records.items()}
    assert not any(all_errors.values()), all_errors
    result = {"schema": "offline-equivalent-completion/1", "new_model_calls": 0,
        "counts": counts, "completed_this_pass": len(reports), "reports": reports,
        "clause_reorders": sum(len(r["clause_reorders"]) for r in reports),
        "gate_sha256": B.K.sha256_file(Path(B.S.__file__)), "runner_sha256": B.K.sha256_file(Path(__file__)),
        "note": "Deterministic clause variants are not independent LLM generations or human gold; no raw response was edited"}
    B.K.write_json_atomic(ROOT / "audits/offline-equivalent-completion-20261025.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
