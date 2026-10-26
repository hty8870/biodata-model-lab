from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import build_conflict_data as B
import prepare_relation_reviews_20261025 as G
import review_relation_texts_20261025 as V

ROOT = Path(__file__).resolve().parent
REVIEWS = ROOT / "data/reviews"
INPUTS = ROOT / "data/review_inputs"
ADJ = ROOT / "data/review_adjudication"
DEFAULT_PREFIXES = ("train-dev-v2", "composite-v2")
PHASE_PREFIXES = {"heldout": ("heldout-v2",)}
QWEN_IDENTITY = {"model": "batiai/qwen3.8-27b:iq3", "channel": "ollama", "provider": "ollama",
                 "thinking": False, "num_ctx": 16384, "max_tokens": 4000}
DEEPSEEK_IDENTITY = {"model": "deepseek-v4-flash", "channel": "chat", "provider": "openai-compatible",
                     "thinking": False, "num_ctx": None, "max_tokens": 4000,
                     "base_url_sha256": "a34e2a4708ed1c61008a151688838dcf1c44d4e7f08054633e72ba7c0b16cfc1"}
PHASE_IDENTITIES = {"train-dev-v2": QWEN_IDENTITY, "composite-v2": QWEN_IDENTITY,
                    "heldout-v2": DEEPSEEK_IDENTITY}


def identity_problems(identity: dict, outcome: dict) -> list[str]:
    checks = {
        "outcome model": outcome.get("model") == identity["model"],
        "outcome provider": outcome.get("provider") == identity["provider"],
        "outcome thinking": outcome.get("thinking") is identity["thinking"],
        "outcome num_ctx": outcome.get("num_ctx") == identity["num_ctx"],
        "outcome max_tokens": outcome.get("max_tokens") == identity["max_tokens"],
    }
    if identity.get("base_url_sha256"):
        checks["outcome base_url_sha256"] = outcome.get("base_url_sha256") == identity["base_url_sha256"]
    return [name for name, value in checks.items() if not value]


def publish_exact(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text(encoding="utf-8") == text, f"Content-addressed output differs: {path}"
        return False
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return True


def load_phase(prefix: str) -> dict:
    version = prefix.rsplit("-", 1)[-1]
    units_path = INPUTS / f"{prefix}.units.jsonl"
    refs_path = INPUTS / f"{prefix}.references.jsonl"
    manifest_path = INPUTS / f"{prefix}.manifest.json"
    run_path = REVIEWS / prefix / "run_manifest.json"
    manifest = B.K.read_json(manifest_path)
    run = B.K.read_json(run_path)
    identity = PHASE_IDENTITIES[prefix]
    assert B.K.sha256_file(units_path) == manifest["units_sha256"]
    assert B.K.sha256_file(refs_path) == manifest["references_sha256"]
    assert run["input_manifest_sha256"] == B.K.sha256_file(manifest_path)
    assert run["prompt_sha256_by_unit"] == manifest["prompt_sha256_by_unit"]
    assert run["model"] == identity["model"] and run["channel"] == identity["channel"]
    assert run["thinking"] is identity["thinking"] and run.get("num_ctx") == identity["num_ctx"]
    assert run["max_output"] == identity["max_tokens"]
    units = B.K.read_jsonl(units_path)
    refs = B.K.read_jsonl(refs_path)
    assert run["planned_calls"] == len(units)
    prompts = {u["id"]: G.prompt_for(u, version) for u in units}
    assert {uid: B.K.sha256_text(prompt) for uid, prompt in prompts.items()} == manifest["prompt_sha256_by_unit"]
    refs_by_id = {}
    for ref in refs:
        assert ref["review_id"] not in refs_by_id
        refs_by_id[ref["review_id"]] = ref
    unit_by_id, pairs_by_id = {}, {}
    for unit in units:
        assert unit["id"] not in unit_by_id
        unit_by_id[unit["id"]] = unit
        for pair in unit["pairs"]:
            assert pair["id"] not in pairs_by_id
            assert pair["id"] in refs_by_id and refs_by_id[pair["id"]]["unit_id"] == unit["id"]
            pairs_by_id[pair["id"]] = {"unit_id": unit["id"], "background": unit["background"],
                                       "old": pair["old"], "new": pair["new"]}
    assert set(refs_by_id) == set(pairs_by_id)
    return {"prefix": prefix, "units": units, "unit_by_id": unit_by_id, "refs_by_id": refs_by_id,
            "pairs_by_id": pairs_by_id, "prompts": prompts, "manifest": manifest, "identity": identity,
            "source": {"units": units_path.relative_to(ROOT).as_posix(),
                       "units_sha256": B.K.sha256_file(units_path),
                       "references": refs_path.relative_to(ROOT).as_posix(),
                       "references_sha256": B.K.sha256_file(refs_path),
                       "manifest": manifest_path.relative_to(ROOT).as_posix(),
                       "manifest_sha256": B.K.sha256_file(manifest_path),
                       "run_manifest": run_path.relative_to(ROOT).as_posix(),
                       "run_manifest_sha256": B.K.sha256_file(run_path)}}


def check_binding(phase: dict, unit: dict, prompt: str) -> dict:
    uid = unit["id"]
    identity = phase["identity"]
    request_path = REVIEWS / phase["prefix"] / f"{uid}.request.json"
    outcome_path = REVIEWS / phase["prefix"] / f"{uid}.outcome.json"
    source = {"phase": phase["prefix"], "request": request_path.relative_to(ROOT).as_posix(),
              "outcome": outcome_path.relative_to(ROOT).as_posix(),
              "input_manifest_sha256": phase["source"]["manifest_sha256"],
              "model": identity["model"], "channel": identity["channel"], "provider": identity["provider"],
              "thinking": identity["thinking"], "num_ctx": identity["num_ctx"],
              "max_tokens": identity["max_tokens"]}
    if not request_path.is_file() or not outcome_path.is_file():
        return {"request": None, "outcome": None, "source": source,
                "problems": [f"missing {'request' if not request_path.is_file() else 'outcome'}"]}
    request = B.K.read_json(request_path)
    outcome = B.K.read_json(outcome_path)
    source.update({"request_sha256": B.K.sha256_file(request_path),
                   "outcome_sha256": B.K.sha256_file(outcome_path),
                   "prompt_sha256": request.get("prompt_sha256")})
    checks = {
        "request unit": request.get("unit_id") == uid,
        "request reserved": request.get("status") == "reserved",
        "request prompt": request.get("prompt") == prompt
                          and request.get("prompt_sha256") == B.K.sha256_text(prompt),
        "request manifest": request.get("input_manifest_sha256") == phase["source"]["manifest_sha256"],
        "outcome unit": outcome.get("unit_id") == uid,
        "outcome prompt": outcome.get("prompt_sha256") == request.get("prompt_sha256"),
        "outcome request": outcome.get("request_sha256") == source["request_sha256"],
        "outcome succeeded": outcome.get("succeeded") is True,
        "outcome text": isinstance(outcome.get("text"), str) and bool(outcome["text"].strip()),
    }
    problems = [name for name, value in checks.items() if not value] + identity_problems(identity, outcome)
    return {"request": request, "outcome": outcome, "source": source, "problems": problems}


def project_rows(raw_rows: list, expected_ids: set[str]) -> tuple[list[dict], dict]:
    expected = set(expected_ids)
    counts = Counter(row.get("id") for row in raw_rows
                     if isinstance(row, dict) and isinstance(row.get("id"), str))
    duplicate_ids = {rid for rid, count in counts.items() if count > 1}
    rows, seen = [], set()
    for index, row in enumerate(raw_rows):
        item = {"row_index": index, "id": None, "relation": None, "ambiguous": None,
                "reason": None, "status": "invalid", "error": None}
        if not isinstance(row, dict):
            item["error"] = "row_not_object"
            rows.append(item)
            continue
        item.update(id=row.get("id"), relation=row.get("relation"),
                    ambiguous=row.get("ambiguous"), reason=row.get("reason"))
        rid = row.get("id")
        if not isinstance(rid, str) or not rid:
            item["error"] = "missing_or_nonstring_id"
            rows.append(item)
            continue
        if rid not in expected:
            item["status"] = "orphan"
            rows.append(item)
            continue
        seen.add(rid)
        if counts[rid] > 1:
            item["error"] = "duplicate_id"
            rows.append(item)
            continue
        if row.get("relation") is None and row.get("ambiguous") is True \
                and isinstance(row.get("reason"), str) and row["reason"].strip():
            item["status"] = "pending"
            rows.append(item)
            continue
        try:
            V.parse_review(json.dumps({"pairs": [row]}, ensure_ascii=False), {"pairs": [{"id": rid}]})
        except (KeyError, TypeError, ValueError) as error:
            item["error"] = str(error)
            rows.append(item)
            continue
        item["status"] = "valid_classified"
        rows.append(item)
    for rid in sorted(expected - seen):
        rows.append({"row_index": None, "id": rid, "relation": None, "ambiguous": None,
                     "reason": None, "status": "missing", "error": None})
    return rows, {
        "actual_rows": len(raw_rows),
        "valid_classified": sum(r["status"] == "valid_classified" for r in rows),
        "pending_unclassified": sum(r["status"] == "pending" for r in rows),
        "invalid_rows": sum(r["status"] == "invalid" for r in rows),
        "orphan_rows": sum(r["status"] == "orphan" for r in rows),
        "missing_ids": sum(r["status"] == "missing" for r in rows),
        "duplicate_ids": len(duplicate_ids),
        "duplicate_rows": sum(counts[rid] for rid in duplicate_ids if rid in expected),
    }


def selftest() -> None:
    checks = 0

    def ok(condition: bool) -> None:
        nonlocal checks
        assert condition
        checks += 1

    good = {"id": "a", "relation": "equivalent", "ambiguous": False, "reason": "same"}
    rows, counts = project_rows([good, {"id": "b", "relation": "other", "ambiguous": False, "reason": "bad"}], {"a", "b"})
    ok(counts["valid_classified"] == 1 and counts["invalid_rows"] == 1
       and any(r["id"] == "a" and r["status"] == "valid_classified" for r in rows))
    rows, counts = project_rows([good, dict(good), dict(good, id="b")], {"a", "b"})
    ok(counts["duplicate_ids"] == 1 and counts["valid_classified"] == 1
       and not any(r["id"] == "a" and r["status"] == "valid_classified" for r in rows))
    rows, counts = project_rows([good], {"a", "b"})
    ok(counts["missing_ids"] == 1
       and any(r["id"] == "b" and r["status"] == "missing" and r["relation"] is None for r in rows))
    rows, counts = project_rows([{"id": "a", "relation": None, "ambiguous": True, "reason": "uncertain"}], {"a"})
    ok(counts["pending_unclassified"] == 1 and rows[0]["status"] == "pending"
       and rows[0]["relation"] is None)
    rows, counts = project_rows([{"id": "x", "relation": "equivalent", "ambiguous": False, "reason": "x"}], {"a"})
    ok(counts["orphan_rows"] == 1 and counts["missing_ids"] == 1)

    deepseek = PHASE_IDENTITIES["heldout-v2"]
    deepseek_outcome = {"model": "deepseek-v4-flash", "provider": "openai-compatible",
                        "thinking": False, "num_ctx": None, "max_tokens": 4000,
                        "base_url_sha256": "a34e2a4708ed1c61008a151688838dcf1c44d4e7f08054633e72ba7c0b16cfc1"}
    ok(identity_problems(deepseek, deepseek_outcome) == [])
    ok(identity_problems(deepseek, {k: v for k, v in deepseek_outcome.items() if k != "num_ctx"}) == [])
    ok("outcome provider" in identity_problems(deepseek, dict(deepseek_outcome, provider="ollama")))
    ok(deepseek["channel"] == "chat" and deepseek["provider"] == "openai-compatible")
    ok(identity_problems(PHASE_IDENTITIES["train-dev-v2"],
                         {"model": "batiai/qwen3.8-27b:iq3", "provider": "ollama", "thinking": False,
                          "num_ctx": 16384, "max_tokens": 4000}) == [])
    print(json.dumps({"selftest_checks": checks, "passed": checks}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--phase", choices=("heldout",), default=None,
                        help="可选 heldout；默认保留 train-dev-v2 + composite-v2")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return

    prefixes = PHASE_PREFIXES.get(args.phase, DEFAULT_PREFIXES)
    phases = [load_phase(prefix) for prefix in prefixes]
    all_rows, units, unit_failures = [], [], []
    for phase in phases:
        for unit in phase["units"]:
            prompt = phase["prompts"][unit["id"]]
            binding = check_binding(phase, unit, prompt)
            unit_info = {"unit_id": unit["id"], "phase": phase["prefix"],
                         "source": binding["source"], "identity_problems": binding["problems"]}
            units.append(unit_info)
            if binding["problems"]:
                unit_failures.append({"unit_id": unit["id"], "phase": phase["prefix"],
                                      "problems": binding["problems"]})
                status = "unit_identity_invalid"
                for pair in unit["pairs"]:
                    all_rows.append({"phase": phase["prefix"], "unit_id": unit["id"], "row_index": None,
                                     "id": pair["id"], "relation": None, "ambiguous": None, "reason": None,
                                     "status": status, "error": ", ".join(binding["problems"]),
                                     "source": binding["source"]})
                continue
            try:
                parsed = json.loads(binding["outcome"]["text"])
                raw_rows = parsed["pairs"]
                assert isinstance(raw_rows, list)
            except (json.JSONDecodeError, KeyError, TypeError, AssertionError) as error:
                unit_failures.append({"unit_id": unit["id"], "phase": phase["prefix"],
                                      "problems": [f"outcome json: {error}"]})
                for pair in unit["pairs"]:
                    all_rows.append({"phase": phase["prefix"], "unit_id": unit["id"], "row_index": None,
                                     "id": pair["id"], "relation": None, "ambiguous": None, "reason": None,
                                     "status": "unit_json_invalid", "error": str(error),
                                     "source": binding["source"]})
                continue
            rows, counts = project_rows(raw_rows, {p["id"] for p in unit["pairs"]})
            unit_info["counts"] = counts
            for row in rows:
                row.update(phase=phase["prefix"], unit_id=unit["id"], source={
                    "request": binding["source"]["request"],
                    "request_sha256": binding["source"]["request_sha256"],
                    "outcome": binding["source"]["outcome"],
                    "outcome_sha256": binding["source"]["outcome_sha256"],
                    "prompt_sha256": binding["source"]["prompt_sha256"],
                    "input_manifest_sha256": binding["source"]["input_manifest_sha256"]})
                reference = phase["refs_by_id"].get(row["id"])
                row["reference"] = reference
                row["match"] = (row["status"] == "valid_classified" and reference is not None
                                and row["relation"] == reference["expected_coarse"])
                all_rows.append(row)

    valid = [r for r in all_rows if r["status"] == "valid_classified"]
    mismatches = [r for r in valid if r["match"] is False]
    ambiguous = [r for r in all_rows if r["ambiguous"] is True]
    valid_ids = {r["id"] for r in valid}
    no_valid = []
    for phase in phases:
        for rid, ref in phase["refs_by_id"].items():
            if rid in valid_ids:
                continue
            candidates = [r for r in all_rows if r["id"] == rid and r["phase"] == phase["prefix"]]
            no_valid.append({"id": rid, "phase": phase["prefix"], "reference": ref,
                             "row_statuses": [r["status"] for r in candidates],
                             "observed_rows": [{k: r[k] for k in ("row_index", "status", "relation", "ambiguous", "reason")}
                                               for r in candidates]})
    no_valid_ids = {r["id"] for r in no_valid}
    mismatch_ids = {r["id"] for r in mismatches}
    ambiguous_ids = {r["id"] for r in ambiguous if r["id"] in {x["review_id"] for p in phases for x in p["refs_by_id"].values()}}
    adjudication_ids = sorted(mismatch_ids | ambiguous_ids | no_valid_ids)
    all_refs = {rid: ref for phase in phases for rid, ref in phase["refs_by_id"].items()}
    all_pairs = {rid: pair for phase in phases for rid, pair in phase["pairs_by_id"].items()}
    all_rows_by_id = {}
    for row in all_rows:
        if isinstance(row.get("id"), str):
            all_rows_by_id.setdefault(row["id"], []).append(row)
    assert set(all_refs) == set(all_pairs)
    assert len(all_refs) == sum(len(phase["refs_by_id"]) for phase in phases)

    def issue_types(rid: str) -> list[str]:
        types = []
        if rid in mismatch_ids:
            types.append("label_disagreement")
        if any(r["ambiguous"] is True for r in all_rows_by_id.get(rid, [])):
            types.append("ambiguous_true")
        if rid not in valid_ids:
            types.append("no_valid_review")
        statuses = {r["status"] for r in all_rows_by_id.get(rid, [])}
        for status, label in (("invalid", "invalid_row"), ("missing", "missing_id"),
                              ("pending", "pending_unclassified"), ("unit_identity_invalid", "unit_identity_invalid"),
                              ("unit_json_invalid", "unit_json_invalid")):
            if status in statuses:
                types.append(label)
        return sorted(set(types))

    cases, references = [], []
    for rid in adjudication_ids:
        ref, pair = all_refs[rid], all_pairs[rid]
        assert pair["unit_id"] == ref["unit_id"]
        case_id = "case-" + B.K.sha256_text(f"row-review|{ref['unit_id']}|{rid}")[:16]
        cases.append({"case_id": case_id, "background": pair["background"],
                      "old": pair["old"], "new": pair["new"]})
        references.append({**{k: ref[k] for k in ("review_id", "unit_id", "source_id", "family_id", "split",
                                                  "expected_coarse", "expected_fine", "label_origin")},
                           "case_id": case_id, "issue_types": issue_types(rid),
                           "observed_rows": [{k: r[k] for k in ("row_index", "status", "relation", "ambiguous", "reason")}
                                             for r in all_rows_by_id.get(rid, []) if r["row_index"] is not None]})
    assert len({case["case_id"] for case in cases}) == len(cases)
    assert all(set(case) == {"case_id", "background", "old", "new"} for case in cases)
    cases_text = "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases)
    cases_sha = B.K.sha256_text(cases_text)
    packet_dir = ADJ / f"row-review-cases-{cases_sha[:16]}"
    cases_path = packet_dir / f"cases-{cases_sha[:16]}.jsonl"
    issue_counts = Counter(issue for ref in references for issue in ref["issue_types"])
    metadata = {"schema": "row-review-adjudication-packet/1", "case_count": len(cases),
                "selection_rule": "按分歧或格式问题选择，不能作为无偏总体准确率样本。",
                "issue_counts": dict(sorted(issue_counts.items())), "cases_sha256": cases_sha,
                "source_manifest_sha256": sorted({p["source"]["manifest_sha256"] for p in phases}),
                "new_model_calls": 0, "cases_contain_oracle": False,
                "cases_contain_reviewer_labels": False, "cases_contain_original_ids": False}
    metadata_text = json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    metadata_sha = B.K.sha256_text(metadata_text)
    metadata_path = packet_dir / f"metadata-{metadata_sha[:16]}.json"
    references_text = "".join(json.dumps(ref, ensure_ascii=False, sort_keys=True) + "\n" for ref in references)
    references_sha = B.K.sha256_text(references_text)
    references_path = ADJ / f"row-review-cases-{cases_sha[:16]}.references-{references_sha[:16]}.jsonl"
    publish_exact(cases_path, cases_text)
    publish_exact(metadata_path, metadata_text)
    publish_exact(references_path, references_text)

    phase_summaries = {}
    for phase in phases:
        rows = [r for r in all_rows if r["phase"] == phase["prefix"]]
        phase_valid = [r for r in rows if r["status"] == "valid_classified"]
        phase_match = sum(r["match"] is True for r in phase_valid)
        phase_summaries[phase["prefix"]] = {
            "planned_units": len(phase["units"]), "planned_pairs": len(phase["refs_by_id"]),
            "actual_rows": sum(1 for r in rows if r["row_index"] is not None),
            "valid_classified": len(phase_valid),
            "pending_unclassified": sum(r["status"] == "pending" for r in rows),
            "invalid_rows": sum(r["status"] == "invalid" for r in rows),
            "orphan_rows": sum(r["status"] == "orphan" for r in rows),
            "missing_ids": sum(r["status"] == "missing" for r in rows),
            "duplicate_ids": sum(1 for r in rows if r["error"] == "duplicate_id") // 2,
            "no_valid_review": sum(1 for r in no_valid if r["phase"] == phase["prefix"]),
            "label_disagreements": sum(1 for r in phase_valid if r["match"] is False),
            "ambiguous_true_rows": sum(r["ambiguous"] is True for r in rows),
            "agreement_on_classified": {"matches": phase_match, "n": len(phase_valid),
                                        "not_accuracy": True},
            "identity": phase["identity"], "source": phase["source"]}

    old_analysis = []
    for prefix in prefixes:
        for path in sorted((REVIEWS / prefix).glob("analysis-*.json")):
            summary = B.K.read_json(path)
            old_analysis.append({"path": path.relative_to(ROOT).as_posix(), "sha256": B.K.sha256_file(path),
                                 "valid_reviewed_pairs": summary.get("valid_reviewed_pairs"),
                                 "failed_unit_count": len(summary.get("failed_units") or [])})

    matches = sum(r["match"] is True for r in valid)
    content = {
        "schema": "row-review-projection/1", "new_model_calls": 0,
        "planned_units": sum(len(p["units"]) for p in phases),
        "planned_pairs": sum(len(p["refs_by_id"]) for p in phases),
        "actual_rows": sum(1 for r in all_rows if r["row_index"] is not None),
        "valid_classified": len(valid), "pending_unclassified": sum(r["status"] == "pending" for r in all_rows),
        "invalid_rows": sum(r["status"] == "invalid" for r in all_rows),
        "orphan_rows": sum(r["status"] == "orphan" for r in all_rows),
        "missing_ids": sum(r["status"] == "missing" for r in all_rows),
        "duplicate_ids": sum(1 for r in all_rows if r["error"] == "duplicate_id") // 2,
        "no_valid_review": len(no_valid), "label_disagreements": len(mismatches),
        "ambiguous_true_rows": len(ambiguous), "adjudication_cases": len(cases),
        "agreement_on_classified": {"matches": matches, "n": len(valid),
                                    "rate": matches / len(valid) if valid else None, "not_accuracy": True},
        "phases": phase_summaries, "units": units, "unit_failures": unit_failures,
        "rows": all_rows, "no_valid_review_rows": no_valid,
        "disagreements": {"label": mismatches, "ambiguous": ambiguous, "no_valid": no_valid},
        "old_analysis": old_analysis,
        "packet": {"cases": cases_path.relative_to(ROOT).as_posix(), "cases_sha256": cases_sha,
                   "metadata": metadata_path.relative_to(ROOT).as_posix(), "metadata_sha256": metadata_sha,
                   "references_for_parent": references_path.relative_to(ROOT).as_posix(),
                   "references_sha256": references_sha},
        "identity_checks": {"phases": {p["prefix"]: p["identity"] for p in phases},
                            "unit_failures": len(unit_failures)},
        "runner_sha256": B.K.sha256_file(Path(__file__)),
        "note": "Per-row projection from existing receipts; agreement with reference is agreement with the written spec, not accuracy. No model call or gold change."}
    content_sha = B.K.sha256_json(content)
    audit_path = ROOT / f"audits/row-review-projection-{content_sha[:16]}.json"
    content["report_path"] = audit_path.relative_to(ROOT).as_posix()
    audit_text = json.dumps(content, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    publish_exact(audit_path, audit_text)
    print(json.dumps({k: content[k] for k in ("planned_units", "planned_pairs", "actual_rows", "valid_classified",
                                              "pending_unclassified", "invalid_rows", "orphan_rows", "missing_ids",
                                              "duplicate_ids", "no_valid_review", "label_disagreements",
                                              "ambiguous_true_rows", "adjudication_cases",
                                              "agreement_on_classified", "report_path", "packet")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
