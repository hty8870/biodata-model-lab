from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import build_conflict_data as B
import prepare_relation_reviews_20261025 as G
import project_row_reviews_20261025 as P
import repair_author_outputs_20261025 as R
import review_relation_texts_20261025 as V

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "data/semantic-repairs-followup-20261026"
R2P1 = ROOT / "data/authors/confirmed-r2p2"
R2P2 = ROOT / "data/authors/confirmed-r2p3"
FAMILIES = ("cf-016", "cf-057")
MODEL = "batiai/qwen3.8-27b:iq3"
DS_MODEL = "deepseek-v4-flash"
FAMILY_CHANNELS = {
    "cf-016": {"channel": "chat", "model": DS_MODEL, "provider": "openai-compatible",
               "num_ctx": None, "max_tokens": 4000,
               "base_url_sha256": "a34e2a4708ed1c61008a151688838dcf1c44d4e7f08054633e72ba7c0b16cfc1"},
    "cf-057": {"channel": "ollama", "model": MODEL, "provider": "ollama",
               "num_ctx": 16384, "max_tokens": 4000,
               "base_url_sha256": "b26993598dffd1f14aed97def57ef67f753518a9b773d8a12033c82b4fa545ca"},
}
REVIEW_GROUPS = (
    {"phase": "train-dev", "prefix": "train-dev-v2", "family": "cf-016"},
    {"phase": "heldout", "prefix": "heldout-v2", "family": "cf-057"},
)
INSTRUCTION = (
    "修订已有记忆规则中的文字，只改写targets列出的side；不新增要求，不改变主体、对象、作用域、事项、时间、"
    "数量、例外、否定、指代或标签。只写自然、可直接执行的规则句；不要写生成/标注步骤、分隔符说明或短语清单说明；"
    "不要保留current_text中的元指令。归档、交接、核对都是执行动作，不是列出清单或副本；索取不能写成提供。"
    "严格遵守action：require只能使用必须/需要/应当中的要求词，prohibit只能使用不得/禁止/不允许中的禁止词；"
    "不得把要求写成禁止或反之。surface仅作可选措辞偏好；numeric range/bounds保留数字和对应形式。"
    "exact_tokens每段必须恰好出现一次，并优先保留。不要判断类别，不输出relation/label/score。"
    "只输出JSON对象，键严格为updates；每个target恰好一条，不能多、少、重复。"
    "格式：{\"updates\":[{\"pair_id\":\"...\",\"side\":\"old或new\",\"text\":\"...\"}]}\n"
)


def write_json(path: Path, value: dict) -> None:
    B.K.write_json_atomic(path, value)


def read_json(path: Path) -> dict:
    return B.K.read_json(path)


def append_event(entry: dict) -> None:
    B.append_call_ledger(STAGE / "events.jsonl", entry)


def stop_ollama() -> dict:
    result = {"cmd": ["ollama", "stop", MODEL], "created": B.K.utc_now()}
    try:
        proc = subprocess.run(["ollama", "stop", MODEL], capture_output=True, text=True, timeout=120)
        result.update(returncode=proc.returncode, stdout=proc.stdout.strip(), stderr=proc.stderr.strip())
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    write_json(STAGE / "ollama-stop.json", result)
    return result


def sum_usage(left: dict, right: dict) -> dict:
    result = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        a, b = (left or {}).get(key), (right or {}).get(key)
        result[key] = a + b if isinstance(a, int) and not isinstance(a, bool) \
            and isinstance(b, int) and not isinstance(b, bool) else None
    return result


def prompt_for(family: str, payload: dict, targets: list[dict]) -> str:
    return INSTRUCTION + json.dumps({"family_id": family, "background": payload["background"],
                                     "targets": targets}, ensure_ascii=False)


def preflight() -> int:
    checks = 0

    def ok(condition: bool) -> None:
        nonlocal checks
        assert condition
        checks += 1

    payload = {"background": "bg", "items": {"x": {"old": "old", "new": "new"}}}
    allowed = {("x", "new")}
    changed = R.apply_updates(payload, '{"updates":[{"pair_id":"x","side":"new","text":"revised"}]}', allowed)
    ok(changed["items"]["x"]["new"] == "revised" and payload["items"]["x"]["new"] == "new")
    for bad in ('{"updates":[]}',
                '{"updates":[{"pair_id":"y","side":"new","text":"wrong"}]}',
                '{"updates":[{"pair_id":"x","side":"old","text":"wrong"}]}',
                '{"updates":[{"pair_id":"x","side":"new","text":"a"},{"pair_id":"x","side":"new","text":"b"}]}'):
        try:
            R.apply_updates(payload, bad, allowed)
        except ValueError:
            ok(True)
        else:
            raise AssertionError("invalid updates accepted")
    prompt = prompt_for("cf-006", payload, [{"pair_id": "x", "side": "new", "current_text": "old instruction",
        "intended_semantics": "natural rule", "action": "require", "surface": None, "exact_tokens": []}])
    ok("required_phrases" not in prompt and "expected_coarse" not in prompt)
    return checks


def load_sources() -> tuple[dict, dict, dict]:
    specs = {spec["family_id"]: spec for spec in B.K.read_jsonl(B.K.SPECS_PATH)}
    records, payloads = {}, {}
    for family in FAMILIES:
        path = R2P1 / f"{family}.json"
        record = read_json(path)
        expected = B.K.AUTHOR_PLAN[record["split"]]
        assert record["family_id"] == family
        assert B.verified_record(path, specs[family], expected)
        records[family] = record
        payloads[family] = {"background": record["background"],
                            "items": {entry["pair_id"]: {key: entry[key] for key in ("old", "new")}
                                      for entry in record["items"]}}
    return specs, records, payloads


def build_targets(spec: dict, payload: dict, family: str) -> tuple[list, set, list[dict]]:
    raw = [(item["pair_id"], side) for item in spec["items"] if item["kind"] != "dyad_flip"
           for side in ("old", "new")
           if "逐字" in payload["items"][item["pair_id"]][side]
           or "完整列出" in payload["items"][item["pair_id"]][side]]
    changes = R.changes_for(spec, [f"{pair_id}.{side}: semantic-localized repair" for pair_id, side in raw])
    items = {item["pair_id"]: item for item in spec["items"]}
    targets = []
    for pair_id, side in sorted(changes):
        assert pair_id in payload["items"] and side in ("old", "new")
        item = items[pair_id]
        targets.append({"pair_id": pair_id, "side": side,
                        "current_text": payload["items"][pair_id][side],
                        "intended_semantics": item["semantic"][side],
                        "action": item[side]["action"],
                        "surface": (item.get("surface") or {}).get(side),
                        "exact_tokens": list(item["gate"][side].get("exact_tokens") or [])})
    return raw, changes, targets


def reserve_and_call(family: str, spec: dict, base_record: dict, base_path: Path,
                     payload: dict, changes: set, targets: list[dict]) -> tuple[dict, dict, str]:
    request_path = STAGE / f"{family}.request.json"
    outcome_path = STAGE / f"{family}.outcome.json"
    config = FAMILY_CHANNELS[family]
    prompt = prompt_for(family, payload, targets)
    if request_path.exists() and not outcome_path.exists():
        raise RuntimeError(f"{family}: pending/unknown request; refusing to resend")
    if outcome_path.exists():
        request = read_json(request_path)
        outcome = read_json(outcome_path)
        assert request["prompt_sha256"] == B.K.sha256_text(prompt)
        assert outcome["request_sha256"] == B.K.sha256_file(request_path)
        return request, outcome, "reused"
    request = {"family_id": family, "prompt": prompt, "prompt_sha256": B.K.sha256_text(prompt),
               "base_record": base_path.relative_to(ROOT).as_posix(),
               "base_record_sha256": B.K.sha256_file(base_path),
               "spec_sha256": B.K.sha256_json(spec), "target_keys": sorted(changes),
               "model": config["model"], "channel": config["channel"], "provider": config["provider"],
               "timeout": 600, "num_ctx": config["num_ctx"], "num_predict": 4000,
               "text_gate_sha256": B.K.sha256_file(Path(B.S.__file__)),
               "status": "reserved", "created": B.K.utc_now()}
    write_json(request_path, request)
    append_event({"event": "reserved", "family_id": family, "request_sha256": B.K.sha256_file(request_path),
                  "channel": config["channel"], "model": config["model"], "created": B.K.utc_now()})
    try:
        if config["channel"] == "chat":
            channel = B.ChatChannel(config["model"], max_tokens=4000)
            channel.prepare()
            outcome = channel.complete(prompt)
        else:
            channel = B.OllamaChannel(config["model"], timeout=600, num_ctx=16384)
            outcome = channel.complete(prompt, num_predict=4000)
    except Exception as error:
        outcome = {"text": None, "succeeded": False, "attempted": True, "finish_reason": None,
                   "error": f"{type(error).__name__}: {error}", "usage": None, "latency_s": None,
                   "provider": config["provider"], "model": config["model"], "thinking": False,
                   "num_ctx": config["num_ctx"], "max_tokens": 4000}
    outcome.update({"family_id": family, "request_sha256": B.K.sha256_file(request_path),
                    "prompt_sha256": request["prompt_sha256"], "created": B.K.utc_now()})
    write_json(outcome_path, outcome)
    append_event({"event": "finished", "family_id": family, "outcome_sha256": B.K.sha256_file(outcome_path),
                  "usage": outcome.get("usage"), "succeeded": outcome.get("succeeded"),
                  "channel": config["channel"], "model": config["model"], "created": B.K.utc_now()})
    return request, outcome, "new"


def save_failure(family: str, details: dict) -> Path:
    digest = B.K.sha256_json(details)
    path = STAGE / f"failure-{family}-{digest[:16]}.json"
    write_json(path, details)
    return path


def repair_family(family: str, spec: dict, base_record: dict, payload: dict) -> dict:
    base_path = R2P1 / f"{family}.json"
    config = FAMILY_CHANNELS[family]
    raw_targets, changes, targets = build_targets(spec, payload, family)
    request, outcome, call_status = reserve_and_call(family, spec, base_record, base_path, payload, changes, targets)
    request_path = STAGE / f"{family}.request.json"
    outcome_path = STAGE / f"{family}.outcome.json"
    details = {"family_id": family, "request": request_path.relative_to(ROOT).as_posix(),
               "outcome": outcome_path.relative_to(ROOT).as_posix(),
               "request_sha256": B.K.sha256_file(request_path), "outcome_sha256": B.K.sha256_file(outcome_path),
               "base_record": base_path.relative_to(ROOT).as_posix(),
               "base_record_sha256": B.K.sha256_file(base_path), "targets": targets,
               "outcome_text": outcome.get("text"), "errors": [], "semantic_errors": [],
               "validate_problems": [], "row_problems": []}
    if not outcome.get("succeeded") or not isinstance(outcome.get("text"), str):
        details["errors"] = ["author outcome unsuccessful"]
        raise RuntimeError(f"{family}: author outcome unsuccessful; evidence {save_failure(family, details)}")
    identity_checks = {"model": outcome.get("model") == config["model"],
                       "provider": outcome.get("provider") == config["provider"],
                       "thinking": outcome.get("thinking") is False,
                       "num_ctx": outcome.get("num_ctx") == config["num_ctx"],
                       "max_tokens": outcome.get("max_tokens") == 4000,
                       "base_url_sha256": outcome.get("base_url_sha256") == config["base_url_sha256"]}
    if not all(identity_checks.values()):
        details["errors"] = [f"author identity: {[k for k, v in identity_checks.items() if not v]}"]
        raise RuntimeError(f"{family}: author identity mismatch; evidence {save_failure(family, details)}")
    try:
        updated = R.apply_updates(payload, outcome["text"], changes)
        updated = B.canonical_control_old(spec, updated)
        errors = R.errors_for(spec, updated, base_record["model"])
        reorders = []
        if errors and all("old == new after normalisation" in error for error in errors):
            updated, reorders = R.reorder_equivalent_failures(spec, updated, errors)
            updated = B.canonical_control_old(spec, updated)
            errors = R.errors_for(spec, updated, base_record["model"])
    except (KeyError, TypeError, ValueError) as error:
        details["errors"] = [f"apply/parse: {type(error).__name__}: {error}"]
        raise RuntimeError(f"{family}: apply/parse failed; evidence {save_failure(family, details)}")
    deterministic_fixes = []
    if errors and any("item scope must use 只针对 phrasing" in error for error in errors):
        scope_pattern = re.compile(r"(?<!只)(?<!仅)针对")
        for pair_id, side in sorted(changes):
            text = updated["items"][pair_id][side]
            fixed = scope_pattern.sub("只针对", text.replace("仅针对", "只针对"))
            if fixed != text:
                updated["items"][pair_id][side] = fixed
                deterministic_fixes.append({"pair_id": pair_id, "side": side,
                    "rule": "scope-only-normalization-v1", "before": text, "after": fixed,
                    "before_sha256": B.K.sha256_text(text), "after_sha256": B.K.sha256_text(fixed),
                    "reason": "existing gate requires literal 只针对; normalized 仅针对/针对 to 只针对"})
        updated = B.canonical_control_old(spec, updated)
        errors = R.errors_for(spec, updated, base_record["model"])
    if errors and any("bounds surface" in error for error in errors):
        from offline_surface_completion_20261025 import normalize_numeric_bounds
        items = {item["pair_id"]: item for item in spec["items"]}
        for pair_id, side in sorted(changes):
            if not any(f"{pair_id}.{side}: bounds surface" in error for error in errors):
                continue
            try:
                fixed, record = normalize_numeric_bounds(updated["items"][pair_id][side], items[pair_id], side)
            except (KeyError, ValueError):
                continue
            updated["items"][pair_id][side] = fixed
            deterministic_fixes.append({"status": "applied", **record})
        updated = B.canonical_control_old(spec, updated)
        errors = R.errors_for(spec, updated, base_record["model"])
    if errors and all("old == new after normalisation" in error for error in errors):
        updated, reorders = R.reorder_equivalent_failures(spec, updated, errors)
        updated = B.canonical_control_old(spec, updated)
        errors = R.errors_for(spec, updated, base_record["model"])
    semantic_errors = []
    for pair_id, side in changes:
        text = updated["items"][pair_id][side]
        for term in ("逐字", "完整列出", "最高优先级", "required_phrases"):
            if term in text:
                semantic_errors.append(f"{pair_id}.{side}: leaked term {term!r}")
    for target in targets:
        text = updated["items"][target["pair_id"]][target["side"]]
        for token in target["exact_tokens"]:
            if text.count(token) != 1:
                semantic_errors.append(f"{target['pair_id']}.{target['side']}: exact token {token!r} count {text.count(token)}")
    validate_problems = B.validate_author_payload(spec, updated)
    row_record = {"background": updated["background"],
                  "items": [dict(pair_id=key, **value) for key, value in updated["items"].items()],
                  "model": base_record["model"], "source": "author"}
    rows, row_problems = B.rows_for_family(spec, row_record)
    if len(rows) != 18:
        row_problems = list(row_problems) + [f"expected 18 rows, got {len(rows)}"]
    details.update(errors=errors, semantic_errors=semantic_errors,
                   validate_problems=validate_problems, row_problems=row_problems,
                   reorders=reorders, deterministic_fixes=deterministic_fixes, updated=updated)
    if errors or semantic_errors or validate_problems or row_problems:
        raise RuntimeError(f"{family}: validation failed; evidence {save_failure(family, details)}")
    changed_sides = [f"{pair_id}.{side}" for pair_id, side in sorted(changes)
                     if payload["items"][pair_id][side] != updated["items"][pair_id][side]]
    targets_report = [{**target, "changed": payload["items"][target["pair_id"]][target["side"]]
                       != updated["items"][target["pair_id"]][target["side"]]} for target in targets]
    revisions = {"pair_id": family, "base_record": base_path.relative_to(ROOT).as_posix(),
                 "base_record_sha256": B.K.sha256_file(base_path),
                 "request": request_path.relative_to(ROOT).as_posix(),
                 "request_sha256": B.K.sha256_file(request_path),
                 "outcome": outcome_path.relative_to(ROOT).as_posix(),
                 "outcome_sha256": B.K.sha256_file(outcome_path),
                 "targets": [{target["pair_id"]: target["pair_id"], "side": target["side"],
                              "before": payload["items"][target["pair_id"]][target["side"]],
                              "after": updated["items"][target["pair_id"]][target["side"]],
                              "action": target["action"], "surface": target["surface"],
                              "exact_tokens": target["exact_tokens"],
                              "changed": payload["items"][target["pair_id"]][target["side"]]
                                         != updated["items"][target["pair_id"]][target["side"]]}
                             for target in targets],
                 "changed_sides": changed_sides, "deterministic_reorders": reorders,
                 "deterministic_surface_fixes": deterministic_fixes}
    write_json(STAGE / f"{family}.revisions.json", revisions)
    return {"payload": updated, "outcome": outcome, "request": request, "targets": targets_report,
            "changes": changes, "raw_targets": raw_targets, "changed_sides": changed_sides,
            "reorders": reorders, "deterministic_fixes": deterministic_fixes,
            "call_status": call_status, "rows": rows,
            "request_sha256": B.K.sha256_file(request_path), "outcome_sha256": B.K.sha256_file(outcome_path)}


def build_r2p2(repairs: dict, specs: dict) -> dict:
    R2P2.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(R2P1.glob("*.json")):
        family = path.stem
        if family in FAMILIES:
            continue
        target = R2P2 / path.name
        data = path.read_bytes()
        if target.exists():
            assert target.read_bytes() == data, f"existing r2p3 copy differs: {target}"
        else:
            target.write_bytes(data)
        copied += 1
    assert copied == 58, copied
    validation = {}
    for family in FAMILIES:
        spec, info = specs[family], repairs[family]
        config = FAMILY_CHANNELS[family]
        base_path = R2P1 / f"{family}.json"
        base = read_json(base_path)
        identity = {
            "provider": base["identity"].get("provider"), "model": base["model"],
            "base_url_sha256": base["identity"].get("base_url_sha256"), "thinking": False,
            "max_tokens": base["identity"].get("max_tokens", 8000),
            "channel": base["identity"].get("channel"),
            "derivation": "semantic-localized-repair-v3",
            "prompt_sha256_is_base_generation": True,
            "original_prompt_sha256": base["prompt_sha256"],
            "base_record": base_path.relative_to(ROOT).as_posix(),
            "base_record_sha256": B.K.sha256_file(base_path),
            "repair_request_sha256": info["request_sha256"],
            "repair_outcome_sha256": info["outcome_sha256"],
            "repair_max_tokens": 4000, "repair_num_ctx": config["num_ctx"],
            "repair_channel": config["channel"], "repair_provider": config["provider"],
            "repair_model": config["model"],
            "semantic_repair_runner_sha256": B.K.sha256_file(Path(__file__)),
            "text_gate_sha256": B.K.sha256_file(Path(B.S.__file__)),
            "targets": [{key: target[key] for key in ("pair_id", "side", "action", "surface",
                                                       "exact_tokens", "changed")}
                        for target in info["targets"]],
            "changed_sides": info["changed_sides"],
            "deterministic_reorders": info["reorders"],
            "deterministic_surface_fixes": info["deterministic_fixes"],
        }
        target = R2P2 / f"{family}.json"
        if target.exists():
            existing = read_json(target)
            assert existing["identity"]["derivation"] == "semantic-localized-repair-v3"
            assert existing["identity"]["base_record_sha256"] == identity["base_record_sha256"]
            assert existing["identity"]["repair_outcome_sha256"] == identity["repair_outcome_sha256"]
        else:
            usage = sum_usage(base.get("usage"), info["outcome"].get("usage"))
            B.write_author_record(R2P2, spec, info["payload"], model=base["model"],
                                  prompt=B.build_prompt(spec), usage=usage, source="author", identity=identity)
        assert B.verified_record(target, spec, B.K.AUTHOR_PLAN[spec["split"]])
        rows, problems = B.rows_for_family(spec, read_json(target))
        assert len(rows) == 18 and not problems, (family, problems)
        validation[family] = {"path": target.relative_to(ROOT).as_posix(),
                              "sha256": B.K.sha256_file(target), "verified": True,
                              "rows": len(rows), "row_problems": problems}
    records = B.load_author_records(R2P2)
    assert len(records) == 60, len(records)
    verified = 0
    for family, record in records.items():
        spec = specs[family]
        path = R2P2 / f"{family}.json"
        if B.verified_record(path, spec, B.K.AUTHOR_PLAN[record["split"]]):
            verified += 1
    assert verified == 60, verified
    write_json(STAGE / "r2p3_validation.json",
               {"records": len(records), "verified": verified, "families": validation})
    return {"records": len(records), "verified": verified, "families": validation}


def build_review_inputs(specs: dict) -> dict:
    tokenizer = B._load_tokenizer()
    groups, all_rows = [], []
    for group in REVIEW_GROUPS:
        family, phase, prefix = group["family"], group["phase"], group["prefix"]
        spec = specs[family]
        record = read_json(R2P2 / f"{family}.json")
        rows, problems = B.rows_for_family(spec, record)
        assert len(rows) == 18 and not problems, (family, problems)
        unit_id = G.opaque(family)
        unit = {"id": unit_id, "background": record["background"], "pairs": []}
        references = []
        for row in rows:
            pair_id = G.opaque(row["id"])
            unit["pairs"].append({"id": pair_id, "old": row["pair"]["old"]["text"],
                                  "new": row["pair"]["new"]["text"]})
            references.append({"review_id": pair_id, "unit_id": unit_id, "source_id": row["id"],
                               "family_id": family, "split": row["split"],
                               "expected_coarse": row["coarse_label"], "expected_fine": row["label"],
                               "label_origin": row["label_origin"]})
        G.prompt_for(unit, "v2")
        lengths = [len(tokenizer.encode(row["visible_text"], add_special_tokens=True)) for row in rows]
        assert max(lengths) <= 768, max(lengths)
        units_path = STAGE / f"data/review_inputs/{prefix}.units.jsonl"
        refs_path = STAGE / f"data/review_inputs/{prefix}.references.jsonl"
        manifest_path = STAGE / f"data/review_inputs/{prefix}.manifest.json"
        units_text = json.dumps(unit, ensure_ascii=False, sort_keys=True) + "\n"
        refs_text = "".join(json.dumps(ref, ensure_ascii=False, sort_keys=True) + "\n" for ref in references)
        G.publish(units_path, units_text)
        G.publish(refs_path, refs_text)
        prompt_sha = {unit["id"]: B.K.sha256_text(G.prompt_for(unit, "v2"))}
        review_model = "batiai/qwen3.8-27b:iq3" if phase == "train-dev" else "deepseek-v4-flash"
        manifest = {
            "schema": "blinded-relation-review-inputs/1", "phase": phase,
            "families": 1, "planned_units": 1, "planned_pairs": len(references),
            "rows": len(references), "new_model_calls": 0,
            "student_input_tokens": {"max": max(lengths), "mean": sum(lengths) / len(lengths),
                                     "limit": 768, "truncated": 0},
            "split_counts": dict(Counter(ref["split"] for ref in references)),
            "coarse_counts": dict(Counter(row["coarse_label"] for row in rows)),
            "units_sha256": B.K.sha256_file(units_path), "references_sha256": B.K.sha256_file(refs_path),
            "prompt_sha256_by_unit": prompt_sha,
            "author_sha256_by_family": {family: B.K.sha256_file(R2P2 / f"{family}.json")},
            "specs_sha256": B.K.sha256_file(B.K.SPECS_PATH), "review_model": review_model,
            "max_output": 4000, "blind_fields": ["expected labels", "recipe", "split", "author", "source family/id"],
            "status": "inputs_prepared_not_reviewed_or_trained",
            "generator_sha256": B.K.sha256_file(Path(__file__))}
        G.publish(manifest_path, json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        groups.append({"phase": phase, "prefix": prefix, "family": family,
                       "units": 1, "pairs": len(references),
                       "max_visible_tokens": max(lengths), "mean_visible_tokens": sum(lengths) / len(lengths),
                       "units_path": units_path.relative_to(ROOT).as_posix(),
                       "references_path": refs_path.relative_to(ROOT).as_posix(),
                       "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
                       "manifest_sha256": B.K.sha256_file(manifest_path)})
        all_rows.extend(rows)
    assert len(groups) == 2 and len(all_rows) == 36
    write_json(STAGE / "review_inputs_summary.json",
               {"groups": groups, "units": 2, "pairs": 36,
                "max_visible_tokens": max(group["max_visible_tokens"] for group in groups),
                "limit": 768})
    return {"groups": groups, "rows": all_rows,
            "max_tokens": max(group["max_visible_tokens"] for group in groups)}


def run_reviews() -> dict:
    outputs = []
    for group in REVIEW_GROUPS:
        phase = group["phase"]
        old_root, old_argv = V.ROOT, sys.argv
        buffer = io.StringIO()
        V.ROOT = STAGE
        sys.argv = ["review_relation_texts", "--phase", phase, "--version", "v2",
                    "--collect", "--max-new", "1"]
        try:
            with contextlib.redirect_stdout(buffer):
                V.main()
        finally:
            V.ROOT, sys.argv = old_root, old_argv
            path = STAGE / f"review_main_stdout-{phase}.json"
            path.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")
        outputs.append(path.relative_to(ROOT).as_posix())
    return {"stdout": outputs}


def summarize_reviews() -> dict:
    old = (P.ROOT, P.INPUTS, P.REVIEWS, P.ADJ)
    P.ROOT, P.INPUTS, P.REVIEWS, P.ADJ = STAGE, STAGE / "data/review_inputs", STAGE / "data/reviews", STAGE / "data/review_adjudication"
    try:
        all_rows, failures, groups = [], [], []
        for group in REVIEW_GROUPS:
            phase = P.load_phase(group["prefix"])
            group_rows = []
            for unit in phase["units"]:
                binding = P.check_binding(phase, unit, phase["prompts"][unit["id"]])
                if binding["problems"]:
                    failures.append({"unit_id": unit["id"], "problems": binding["problems"]})
                    continue
                try:
                    raw_rows = json.loads(binding["outcome"]["text"])["pairs"]
                    rows, counts = P.project_rows(raw_rows, {pair["id"] for pair in unit["pairs"]})
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    failures.append({"unit_id": unit["id"], "problems": [f"review json: {error}"]})
                    continue
                for row in rows:
                    row.update(phase=group["phase"], prefix=group["prefix"], unit_id=unit["id"], source={
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
                    group_rows.append(row)
            valid = [row for row in group_rows if row["status"] == "valid_classified"]
            disagreements = [row for row in valid if row["match"] is False]
            ambiguous = [row for row in group_rows if row["ambiguous"] is True]
            invalid = [row for row in group_rows if row["status"] == "invalid"]
            no_valid = [ref for ref in phase["refs_by_id"].values()
                        if ref["review_id"] not in {row["id"] for row in valid}]
            groups.append({"phase": group["phase"], "prefix": group["prefix"],
                           "planned_pairs": len(phase["refs_by_id"]), "valid_classified": len(valid),
                           "disagreements": disagreements, "ambiguous": ambiguous,
                           "invalid": invalid, "no_valid_review": no_valid,
                           "source": phase["source"], "identity": phase["identity"]})
            all_rows.extend(group_rows)
        valid = [row for row in all_rows if row["status"] == "valid_classified"]
        disagreements = [row for row in valid if row["match"] is False]
        ambiguous = [row for row in all_rows if row["ambiguous"] is True]
        invalid = [row for row in all_rows if row["status"] == "invalid"]
        no_valid = [ref for group in groups for ref in group["no_valid_review"]]
        projection = {"schema": "semantic-repair-review-projection/1",
                      "new_model_calls": 0, "planned_pairs": 36,
                      "actual_rows": sum(row["row_index"] is not None for row in all_rows),
                      "valid_classified": len(valid), "disagreements": disagreements,
                      "ambiguous": ambiguous, "invalid": invalid, "no_valid_review": no_valid,
                      "failures": failures, "rows": all_rows, "groups": groups}
        write_json(STAGE / "review_projection.json", projection)
        with (STAGE / "reviews.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in all_rows:
                if row["row_index"] is not None:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        analyses = []
        review_costs = {}
        for group in REVIEW_GROUPS:
            folder = STAGE / "data/reviews" / group["prefix"]
            analyses.extend(sorted(folder.glob("analysis-*.json")))
            for path in sorted(folder.glob("*.outcome.json")):
                outcome = read_json(path)
                review_costs[path.stem.replace(".outcome", "")] = {
                    "phase": group["phase"], "usage": outcome.get("usage"),
                    "latency_s": outcome.get("latency_s"),
                    "succeeded": outcome.get("succeeded")}
        return {"review_projection": (STAGE / "review_projection.json").relative_to(ROOT).as_posix(),
                "reviews": (STAGE / "reviews.jsonl").relative_to(ROOT).as_posix(),
                "analyses": [path.relative_to(ROOT).as_posix() for path in analyses],
                "valid_classified": len(valid), "disagreements": len(disagreements),
                "ambiguous": len(ambiguous), "invalid": len(invalid),
                "no_valid_review": len(no_valid), "failures": failures,
                "disagreement_rows": disagreements, "review_costs": review_costs,
                "groups": groups}
    finally:
        P.ROOT, P.INPUTS, P.REVIEWS, P.ADJ = old


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        checks = preflight()
        print(json.dumps({"selftest_checks": checks, "passed": checks}))
        return
    preflight()
    STAGE.mkdir(parents=True, exist_ok=True)
    specs, base_records, payloads = load_sources()
    base_before = {path.name: B.K.sha256_file(path) for path in sorted(R2P1.glob("*.json"))}
    script_before = {str(path): B.K.sha256_file(Path(path)) for path in
                     (B.__file__, R.__file__, G.__file__, V.__file__, P.__file__)}
    repairs = {}
    try:
        for family in FAMILIES:
            repairs[family] = repair_family(family, specs[family], base_records[family], payloads[family])
        build_r2p2(repairs, specs)
        review_inputs = build_review_inputs(specs)
        run_reviews()
        review_summary = summarize_reviews()
    finally:
        stop_ollama()
    base_after = {path.name: B.K.sha256_file(path) for path in sorted(R2P1.glob("*.json"))}
    assert base_before == base_after
    script_after = {str(path): B.K.sha256_file(Path(path)) for path in
                    (B.__file__, R.__file__, G.__file__, V.__file__, P.__file__)}
    assert script_before == script_after
    write_json(STAGE / "source_hashes.json",
               {"base_records": base_before, "base_unchanged": True,
                "source_scripts": script_before, "source_scripts_unchanged": True})
    leak_terms = ("逐字使用", "完整列出", "最高优先级", "required_phrases")
    leaks = []
    for path in sorted(R2P2.glob("*.json")):
        record = read_json(path)
        for item in record["items"]:
            for side in ("old", "new"):
                text = item[side]
                for term in leak_terms:
                    if term in text:
                        leaks.append({"family_id": record["family_id"], "pair_id": item["pair_id"],
                                      "side": side, "term": term, "text": text})
    assert not leaks, leaks
    write_json(STAGE / "text_leak_scan.json",
               {"terms": list(leak_terms), "scope": "r2p3 item.old/item.new only",
                "hits": len(leaks), "rows": leaks})
    author_costs = {family: {"channel": FAMILY_CHANNELS[family]["channel"],
                             "model": FAMILY_CHANNELS[family]["model"],
                             "call_status": repairs[family]["call_status"],
                             "usage": repairs[family]["outcome"].get("usage"),
                             "latency_s": repairs[family]["outcome"].get("latency_s")}
                    for family in FAMILIES}
    qwen_calls = sum(1 for family in FAMILIES if FAMILY_CHANNELS[family]["channel"] == "ollama") + 1
    deepseek_calls = sum(1 for family in FAMILIES if FAMILY_CHANNELS[family]["channel"] == "chat") + 1
    costs = {"authors": author_costs, "reviews": review_summary["review_costs"],
             "qwen_calls": qwen_calls, "deepseek_calls": deepseek_calls,
             "total_calls": qwen_calls + deepseek_calls}
    write_json(STAGE / "costs.json", costs)
    summary = {"authors": [f"data/authors/confirmed-r2p3/{family}.json" for family in FAMILIES],
              "changed_sides": {family: len(repairs[family]["changed_sides"]) for family in FAMILIES},
              "target_counts": {family: len(repairs[family]["changes"]) for family in FAMILIES},
              "deterministic_surface_fixes": {family: repairs[family]["deterministic_fixes"]
                                              for family in FAMILIES},
              "r2p3_validation": read_json(STAGE / "r2p3_validation.json"),
              "review_inputs": review_inputs["groups"],
              "max_visible_tokens": review_inputs["max_tokens"], "token_limit": 768,
              "review": {key: review_summary[key] for key in
                         ("review_projection", "reviews", "analyses", "valid_classified",
                          "disagreements", "ambiguous", "invalid", "no_valid_review", "failures")},
              "review_disagreements": review_summary["disagreement_rows"],
              "text_leak_scan": {"path": (STAGE / "text_leak_scan.json").relative_to(ROOT).as_posix(),
                                 "hits": len(leaks)},
              "costs": costs,
              "ollama_stop": read_json(STAGE / "ollama-stop.json"),
              "evidence": {"stage": STAGE.relative_to(ROOT).as_posix(),
                           "review_inputs": (STAGE / "review_inputs_summary.json").relative_to(ROOT).as_posix(),
                           "review_manifests": [group["manifest_path"] for group in review_inputs["groups"]],
                           "r2p3_validation": (STAGE / "r2p3_validation.json").relative_to(ROOT).as_posix(),
                           "costs": (STAGE / "costs.json").relative_to(ROOT).as_posix(),
                           "source_hashes": (STAGE / "source_hashes.json").relative_to(ROOT).as_posix(),
                           "text_leak_scan": (STAGE / "text_leak_scan.json").relative_to(ROOT).as_posix()}}
    write_json(STAGE / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
