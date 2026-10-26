from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import build_conflict_data as B
import repair_author_outputs_20261025 as R

ROOT = Path(__file__).resolve().parent
AUTHORS = R.AUTHORS
REPAIRS = R.REPAIRS
RANGE_RE = re.compile(r"在\s*(\d+)\s*到\s*(\d+)\s*([^，。；\s]*)\s*之间")
RULE = "range-to-bounds-synonym-v1"


def _digits(text: str) -> list[str]:
    return re.findall(r"\d+", text)


def _clauses(text: str) -> list[str]:
    return re.split(r"[，。；]", text)


def normalize_numeric_bounds(text: str, item: dict, side: str = "new") -> tuple[str, dict]:
    """把唯一一处「在 low 到 high [单位]之间」改成等义的 bounds 写法。"""
    if not isinstance(text, str) or not text:
        raise ValueError("text is empty")
    surface = (item.get("surface") or {}).get(side)
    if not isinstance(surface, dict) or surface.get("kind") != "numeric" or surface.get("form") != "bounds":
        raise ValueError("surface is not numeric/bounds")
    factor = item.get(side)
    if not isinstance(factor, dict):
        raise ValueError("missing side factor")
    if factor.get("action") != "require":
        raise ValueError("action is not require")
    measure = factor.get("measure") or {}
    low, high = measure.get("low"), measure.get("high")
    if not isinstance(low, int) or isinstance(low, bool) or not isinstance(high, int) or isinstance(high, bool):
        raise ValueError("measure low/high missing")
    matches = list(RANGE_RE.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one range clause, found {len(matches)}")
    match = matches[0]
    low_text, high_text, unit = match.group(1), match.group(2), match.group(3)
    if int(low_text) != low or int(high_text) != high:
        raise ValueError("range numbers do not match measure low/high")
    replacement = f"不少于 {low_text} 且不多于 {high_text}" + (f" {unit}" if unit else "")
    result = text[:match.start()] + replacement + text[match.end():]
    if result == text:
        raise ValueError("normalization produced identical text")
    numbers_before, numbers_after = _digits(text), _digits(result)
    clauses_before, clauses_after = _clauses(text), _clauses(result)
    if numbers_before != numbers_after:
        raise ValueError("number sequence changed")
    if len(clauses_before) != len(clauses_after):
        raise ValueError("clause count changed")
    return result, {
        "pair_id": item.get("pair_id"),
        "side": side,
        "rule": RULE,
        "deterministic": True,
        "match_span": [match.start(), match.end()],
        "match": {"text": match.group(0), "low": low_text, "high": high_text, "unit": unit},
        "before": text,
        "after": result,
        "before_sha256": B.K.sha256_text(text),
        "after_sha256": B.K.sha256_text(result),
        "numbers_before": numbers_before,
        "numbers_after": numbers_after,
        "clauses_before": len(clauses_before),
        "clauses_after": len(clauses_after),
    }


def _sum_usage(left: dict, right: dict) -> dict:
    result = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        a, b = (left or {}).get(key), (right or {}).get(key)
        result[key] = a + b if isinstance(a, int) and not isinstance(a, bool) \
            and isinstance(b, int) and not isinstance(b, bool) else None
    return result


def _group_errors(spec: dict, errors: list[str]) -> tuple[dict, list[str]]:
    groups: dict[tuple[str, str], list[str]] = {}
    unlocalized = []
    for error in errors:
        try:
            keys = R.changes_for(spec, [error])
        except ValueError:
            unlocalized.append(error)
            continue
        for key in keys:
            groups.setdefault(key, []).append(error)
    return groups, unlocalized


def _apply_surface_normalizations(spec: dict, payload: dict, errors: list[str]) -> tuple[dict, list[dict], list[str]]:
    items = {i["pair_id"]: i for i in spec["items"]}
    groups, unlocalized = _group_errors(spec, errors)
    records = []
    for (pair_id, side), group_errors in groups.items():
        if not any("bounds surface" in error for error in group_errors):
            continue
        try:
            new_text, record = normalize_numeric_bounds(payload["items"][pair_id][side], items[pair_id], side)
        except (KeyError, ValueError) as error:
            records.append({"pair_id": pair_id, "side": side, "status": "rejected", "reason": str(error)})
            continue
        payload["items"][pair_id][side] = new_text
        records.append({"status": "applied", **record})
    return payload, records, unlocalized


def _process_family(spec: dict, raw: dict, raw_path: Path, apply: bool) -> dict:
    family = spec["family_id"]
    split = spec["split"]
    expected_models = B.K.AUTHOR_PLAN[split]
    target = AUTHORS / f"{family}.json"
    if B.verified_record(target, spec, expected_models):
        return {"family_id": family, "split": split, "status": "skipped_accepted",
                "accepted_record_sha256": B.K.sha256_file(target)}

    request_path = REPAIRS / f"{family}.request.json"
    outcome_path = REPAIRS / f"{family}.outcome.json"
    if not request_path.is_file() or not outcome_path.is_file():
        return {"family_id": family, "split": split, "status": "skipped_missing_repair"}

    assert raw["split"] == split
    assert raw["model"] in expected_models
    if split == "heldout":
        assert raw["model"] == "batiai/qwen3.8-27b:iq3" and raw["provider"] == "ollama"
    else:
        assert raw["model"] == "deepseek-v4-flash" and raw["provider"] != "ollama"
    assert raw["spec_sha256"] == B.K.sha256_json(spec)
    assert raw["prompt_sha256"] == B.K.sha256_text(B.build_prompt(spec))
    request = B.K.read_json(request_path)
    outcome = B.K.read_json(outcome_path)
    assert request["family_id"] == family
    assert request["base_raw_sha256"] == B.K.sha256_file(raw_path)
    assert request["spec_sha256"] == raw["spec_sha256"]
    assert outcome["succeeded"] is True
    assert outcome.get("provider") == raw["provider"]
    assert outcome.get("model") == raw["model"]

    raw_payload = B.parse_author_response(raw["response_text"], spec)
    raw_errors = R.errors_for(spec, raw_payload, raw["model"])
    assert request["prompt_sha256"] == B.K.sha256_text(request["prompt"])
    requested = json.loads(request["prompt"].split("\n", 1)[1])
    changes = {(target["pair_id"], target["side"]) for target in requested["targets"]}
    assert len(changes) == len(requested["targets"])
    items = {item["pair_id"]: item for item in spec["items"]}
    for change in requested["targets"]:
        pair_id, side = change["pair_id"], change["side"]
        assert change["current_text"] == raw_payload["items"][pair_id][side]
        assert change["intended_semantics"] == items[pair_id]["semantic"][side]
    payload = R.apply_updates(raw_payload, outcome["text"], changes)
    repair_errors = R.errors_for(spec, payload, raw["model"])

    payload, surface_records, unlocalized = _apply_surface_normalizations(spec, payload, repair_errors)
    errors = R.errors_for(spec, payload, raw["model"])
    reorders = []
    if errors and not unlocalized and all("old == new after normalisation" in error for error in errors):
        payload, reorders = R.reorder_equivalent_failures(spec, payload, errors)
        errors = R.errors_for(spec, payload, raw["model"])

    canonical = B.canonical_control_old(spec, payload)
    validate_problems = B.validate_author_payload(spec, canonical)
    row_record = {"background": canonical["background"],
                  "items": [dict(pair_id=key, **value) for key, value in canonical["items"].items()],
                  "model": raw["model"], "source": "author"}
    _, row_problems = B.rows_for_family(spec, row_record)
    source = {"base_raw": raw_path.relative_to(ROOT).as_posix(),
              "base_raw_sha256": B.K.sha256_file(raw_path),
              "repair_request_sha256": B.K.sha256_file(request_path),
              "repair_outcome_sha256": B.K.sha256_file(outcome_path)}
    report = {"family_id": family, "split": split, "status": "blocked",
              "raw_errors": raw_errors, "repair_errors": repair_errors, "final_errors": errors,
              "unlocalized_errors": unlocalized, "surface_normalizations": surface_records,
              "clause_reorders": reorders, "source": source,
              "validate_problems": validate_problems, "row_problems": row_problems}
    if unlocalized or errors or validate_problems or row_problems:
        return report

    derivation = "localized-text-repair-v1"
    if surface_records or reorders:
        derivation += "+deterministic-offline-normalization-v1"
    identity = {key: raw.get(key) for key in ("provider", "model", "base_url_sha256", "thinking", "max_tokens")}
    identity.update({
        "channel": "ollama" if raw["provider"] == "ollama" else "chat",
        "derivation": derivation,
        "prompt_sha256_is_base_generation": True,
        **source,
        "text_gate_sha256": B.K.sha256_file(Path(B.S.__file__)),
        "surface_normalization_rule": RULE,
        "surface_normalization_source": "offline_surface_completion_20261025.py",
        "surface_normalizations": surface_records,
        "normalization_runner_sha256": B.K.sha256_file(Path(__file__)),
    })
    if reorders:
        identity["clause_reorders"] = reorders
        identity["clause_reorder_source"] = "offline_complete_equivalents_20261025.reorder_plain_equivalent"
    report["identity"] = identity
    report["status"] = "ready_not_applied"
    if not apply:
        return report
    usage = _sum_usage(raw["usage"], outcome["usage"])
    B.write_author_record(AUTHORS, spec, canonical, model=raw["model"],
                          prompt=B.build_prompt(spec), usage=usage, source="author", identity=identity)
    assert B.verified_record(target, spec, expected_models)
    report["status"] = "accepted"
    report["accepted_record_sha256"] = B.K.sha256_file(target)
    return report


def selftest() -> None:
    checks = 0

    def ok(condition: bool) -> None:
        nonlocal checks
        assert condition
        checks += 1

    def item(pair_id: str, action: str, form: str, low, high) -> dict:
        return {"pair_id": pair_id,
                "surface": {"new": {"kind": "numeric", "form": form}},
                "new": {"action": action, "measure": {"kind": "within", "low": low, "high": high}}}

    source = "引用必须引用近五年文献，数量在 7 到 9 篇之间，只针对摘要。"
    changed, record = normalize_numeric_bounds(source, item("t1", "require", "bounds", 7, 9))
    ok(changed == "引用必须引用近五年文献，数量不少于 7 且不多于 9 篇，只针对摘要。"
       and record["numbers_before"] == record["numbers_after"] == ["7", "9"]
       and record["clauses_before"] == record["clauses_after"]
       and record["before_sha256"] != record["after_sha256"])

    for bad_item, bad_text, label in (
            (item("t2", "require", "bounds", 7, 9), "引用必须引用近五年文献，数量在 5 到 9 篇之间。", "number mismatch"),
            (item("t3", "prohibit", "bounds", 7, 9), source, "prohibit action"),
            (item("t4", "require", "range", 7, 9), source, "non-bounds surface"),
            (item("t5", "require", "bounds", 7, 9), "在 7 到 9 篇之间，另有在 7 到 9 篇之间。", "multiple ranges")):
        try:
            normalize_numeric_bounds(bad_text, bad_item)
        except ValueError:
            ok(True)
        else:
            raise AssertionError(f"{label} was accepted")
    print(json.dumps({"selftest_checks": checks, "passed": checks}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--split", choices=("train", "dev", "heldout"), default="heldout")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return

    specs = [s for s in B.K.read_jsonl(B.K.SPECS_PATH) if s["split"] == args.split]
    raw_paths = sorted((AUTHORS / "raw").glob("*.json")) \
        + sorted((ROOT / "data/authors/pilot-r2p1-train/raw").glob("*.json"))
    raws = {}
    for path in raw_paths:
        raw = B.K.read_json(path)
        assert raw["family_id"] not in raws
        raws[raw["family_id"]] = (raw, path)
    reports = []
    for spec in sorted(specs, key=lambda s: s["family_id"]):
        if spec["family_id"] not in raws:
            reports.append({"family_id": spec["family_id"], "split": spec["split"], "status": "skipped_no_raw"})
            continue
        raw, raw_path = raws[spec["family_id"]]
        reports.append(_process_family(spec, raw, raw_path, args.apply))
    blocked = [r["family_id"] for r in reports if r["status"] == "blocked"]
    content = {"split": args.split, "apply": bool(args.apply), "reports": reports}
    content_sha256 = B.K.sha256_json(content)
    path = ROOT / f"audits/offline-surface-completion-{args.split}-{content_sha256[:16]}.json"
    summary = {"schema": "offline-surface-completion/1", "split": args.split,
               "apply": bool(args.apply), "new_model_calls": 0,
               "families": len(reports),
               "accepted": sum(r["status"] == "accepted" for r in reports),
               "ready_not_applied": sum(r["status"] == "ready_not_applied" for r in reports),
               "skipped_accepted": sum(r["status"] == "skipped_accepted" for r in reports),
               "blocked": blocked, "report_content_sha256": content_sha256,
               "report_path": path.relative_to(ROOT).as_posix(), "reports": reports,
               "runner_sha256": B.K.sha256_file(Path(__file__)),
               "note": "surface changes are deterministic synonymous normalization from the existing repair payload; no new model call"}
    B.K.write_json_atomic(path, summary)
    print(json.dumps(summary, ensure_ascii=False))
    if blocked:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
