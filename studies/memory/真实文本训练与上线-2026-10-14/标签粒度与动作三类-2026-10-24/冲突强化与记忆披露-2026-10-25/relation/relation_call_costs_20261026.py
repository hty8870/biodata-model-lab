from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "audits/call-costs-20261026.json"
TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")
SPECS = (
    ("author_raw", "confirmed-r2p1", "data/authors/confirmed-r2p1/raw/*.json"),
    ("author_raw", "pilot-r2p1-train", "data/authors/pilot-r2p1-train/raw/*.json"),
    ("author_raw", "pilot-r2-train", "data/authors/pilot-r2-train/raw/*.json"),
    ("author_raw", "composite-real-r2p1", "data/composite/authors-real-r2p1/raw/*.json"),
    ("author_repair_v1", "confirmed-r2p1", "data/authors/confirmed-r2p1/repair_v1/*.outcome.json"),
    ("blinded_review_v1", "train-dev-v1", "data/reviews/train-dev-v1/*.outcome.json"),
    ("blinded_review_v1", "composite-v1", "data/reviews/composite-v1/*.outcome.json"),
    ("blinded_review_v1", "heldout-v1", "data/reviews/heldout-v1/*.outcome.json"),
    ("blinded_review_v2", "train-dev-v2", "data/reviews/train-dev-v2/*.outcome.json"),
    ("blinded_review_v2", "composite-v2", "data/reviews/composite-v2/*.outcome.json"),
    ("blinded_review_v2", "heldout-v2", "data/reviews/heldout-v2/*.outcome.json"),
    ("semantic_author_repair_v2", "semantic-repairs-20261026", "data/semantic-repairs-20261026/*.outcome.json"),
    ("semantic_author_repair_v3", "semantic-repairs-followup-20261026", "data/semantic-repairs-followup-20261026/*.outcome.json"),
    ("semantic_rereview_v2", "semantic-repairs-20261026/data/reviews", "data/semantic-repairs-20261026/data/reviews/*/*.outcome.json"),
    ("semantic_rereview_v3", "semantic-repairs-followup-20261026/data/reviews", "data/semantic-repairs-followup-20261026/data/reviews/*/*.outcome.json"),
)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def sha_text(text: str) -> str:
    return sha_bytes(text.encode("utf-8"))


def is_accepted_record_path(rel: str) -> bool:
    return bool(re.fullmatch(r"data/authors/confirmed-r2p[123]/[^/]+\.json", rel)
                or re.fullmatch(r"data/composite/authors-real-r2p1/[^/]+\.json", rel))


def is_receipt_path(rel: str) -> bool:
    parts = Path(rel).parts
    if any(part in ("_selftest", "_归档", "fixtures") for part in parts):
        return False
    if is_accepted_record_path(rel):
        return False
    if re.fullmatch(r"data/authors/(?:confirmed-r2p1|pilot-r2p1-train|pilot-r2-train)/raw/[^/]+\.json", rel) \
            or re.fullmatch(r"data/composite/authors-real-r2p1/raw/[^/]+\.json", rel):
        return True
    if re.fullmatch(r"data/authors/confirmed-r2p1/repair_v1/[^/]+\.outcome\.json", rel):
        return True
    if re.fullmatch(r"data/reviews/[^/]+/[^/]+\.outcome\.json", rel):
        return True
    if re.fullmatch(r"data/semantic-repairs-[^/]+/[^/]+\.outcome\.json", rel):
        return True
    if re.fullmatch(r"data/semantic-repairs-[^/]+/data/reviews/[^/]+/[^/]+\.outcome\.json", rel):
        return True
    return False


def is_fixture(data: dict, rel: str) -> bool:
    fields = [data.get("model"), data.get("provider"), data.get("schema"), data.get("source"), rel]
    return any(value is not None and "fixture" in str(value).lower() for value in fields)


def normalize_receipt(path: Path, data: dict, stage: str, subset: str, *, file_sha: str | None = None) -> dict:
    rel = path.relative_to(ROOT).as_posix()
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    tokens = {}
    for key in TOKEN_KEYS:
        value = usage.get(key)
        tokens[key] = value if isinstance(value, int) and not isinstance(value, bool) else None
    complete = all(tokens[key] is not None for key in TOKEN_KEYS)
    text = data.get("response_text") if "response_text" in data else data.get("text")
    completed = isinstance(text, str) and bool(text.strip())
    provider = data.get("provider") if isinstance(data.get("provider"), str) and data.get("provider") else "unknown"
    model = data.get("model") if isinstance(data.get("model"), str) and data.get("model") else "unknown"
    channel = "ollama" if provider == "ollama" else "chat" if provider == "openai-compatible" else "unknown"
    latency = data.get("latency_s")
    latency = latency if isinstance(latency, (int, float)) and not isinstance(latency, bool) else None
    is_outcome = path.name.endswith(".outcome.json")
    request_path = path.with_name(path.name.replace(".outcome.json", ".request.json")) if is_outcome else None
    request_file_sha = sha_file(request_path) if request_path is not None and request_path.is_file() else None
    request_sha = data.get("request_sha256") if isinstance(data.get("request_sha256"), str) else None
    request_sha_source = "outcome" if request_sha else "sibling_request_file" if request_file_sha else "missing"
    request_sha = request_sha or request_file_sha
    prompt_sha = data.get("prompt_sha256") if isinstance(data.get("prompt_sha256"), str) else None
    prompt_sha_source = "outcome" if prompt_sha else "missing"
    if not prompt_sha and request_path is not None and request_path.is_file():
        try:
            prompt_sha = json.loads(request_path.read_text(encoding="utf-8")).get("prompt_sha256")
            prompt_sha_source = "sibling_request_file" if isinstance(prompt_sha, str) else "missing"
        except (OSError, json.JSONDecodeError):
            prompt_sha_source = "missing"
    response_sha = data.get("response_sha256") if isinstance(data.get("response_sha256"), str) else None
    response_sha = response_sha or (sha_text(text) if isinstance(text, str) else None)
    mismatch = bool(data.get("request_sha256") and request_file_sha
                    and data["request_sha256"] != request_file_sha)
    total_mismatch = bool(complete and tokens["total_tokens"] != tokens["prompt_tokens"] + tokens["completion_tokens"])
    return {"path": rel, "stage": stage, "subset": subset, "provider": provider, "model": model,
            "channel": channel, "succeeded": data.get("succeeded"), "completed": completed,
            "fixture": is_fixture(data, rel), "usage_complete": complete, "usage": tokens,
            "usage_raw": usage, "latency_s": latency, "file_sha256": file_sha or sha_file(path),
            "request_sha256": request_sha, "request_sha_source": request_sha_source,
            "request_binding_mismatch": mismatch, "prompt_sha256": prompt_sha,
            "prompt_sha_source": prompt_sha_source, "response_sha256": response_sha,
            "total_mismatch": total_mismatch, "error_present": bool(data.get("error") or data.get("error_kind"))}


def enumerate_receipts() -> tuple[list[dict], list[dict], list[dict]]:
    receipts, unreadable, patterns = [], [], []
    seen = set()
    for stage, subset, pattern in SPECS:
        paths = sorted(ROOT.glob(pattern))
        patterns.append({"stage": stage, "subset": subset, "pattern": pattern, "files": len(paths)})
        for path in paths:
            rel = path.relative_to(ROOT).as_posix()
            if not is_receipt_path(rel) or rel in seen:
                continue
            seen.add(rel)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                unreadable.append({"path": rel, "error": type(error).__name__})
                continue
            receipts.append(normalize_receipt(path, data, stage, subset))
    return receipts, unreadable, patterns


def _empty_group() -> dict:
    return {"attempted": 0, "completed": 0, "succeeded_true": 0, "succeeded_false": 0,
            "succeeded_unknown": 0, "known_complete_usage": 0, "unknown_usage": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "latency_s_known": 0.0, "latency_known_receipts": 0, "latency_unknown_receipts": 0,
            "failed_receipts": 0, "failed_known_total_tokens": 0,
            "usage_total_mismatch": 0, "unknown_identity": 0,
            "request_sha_available": 0, "request_binding_mismatch": 0}


def _add(group: dict, receipt: dict) -> None:
    group["attempted"] += 1
    group["completed"] += int(receipt["completed"])
    if receipt["succeeded"] is True:
        group["succeeded_true"] += 1
    elif receipt["succeeded"] is False:
        group["succeeded_false"] += 1
    else:
        group["succeeded_unknown"] += 1
    if receipt["provider"] == "unknown" or receipt["model"] == "unknown":
        group["unknown_identity"] += 1
    if receipt["request_sha256"]:
        group["request_sha_available"] += 1
    group["request_binding_mismatch"] += int(receipt["request_binding_mismatch"])
    if receipt["usage_complete"]:
        group["known_complete_usage"] += 1
        for key in TOKEN_KEYS:
            group[f"{key}"] += receipt["usage"][key]
        if receipt["succeeded"] is not True:
            group["failed_known_total_tokens"] += receipt["usage"]["total_tokens"]
    else:
        group["unknown_usage"] += 1
    if receipt["latency_s"] is not None:
        group["latency_s_known"] += receipt["latency_s"]
        group["latency_known_receipts"] += 1
    else:
        group["latency_unknown_receipts"] += 1
    if receipt["succeeded"] is not True:
        group["failed_receipts"] += 1
    if receipt["total_mismatch"]:
        group["usage_total_mismatch"] += 1


def summarize(receipts: list[dict], *, accepted_counts: dict | None = None,
              unreadable: list[dict] | None = None) -> dict:
    fixtures = [receipt for receipt in receipts if receipt["fixture"]]
    real = [receipt for receipt in receipts if not receipt["fixture"]]
    groups = defaultdict(_empty_group)
    unknown_usage_rows, unknown_latency_rows, unknown_identity_rows = [], [], []
    failed_rows, mismatch_rows, binding_rows = [], [], []
    for receipt in real:
        key = (receipt["stage"], receipt["subset"], receipt["provider"], receipt["model"], receipt["channel"])
        _add(groups[key], receipt)
        if not receipt["usage_complete"]:
            unknown_usage_rows.append({key: receipt[key] for key in
                                       ("path", "stage", "subset", "provider", "model", "channel",
                                        "usage_raw", "succeeded", "completed", "latency_s", "error_present")})
        if receipt["latency_s"] is None:
            unknown_latency_rows.append(receipt["path"])
        if receipt["provider"] == "unknown" or receipt["model"] == "unknown":
            unknown_identity_rows.append(receipt["path"])
        if receipt["succeeded"] is not True:
            failed_rows.append({key: receipt[key] for key in
                                ("path", "stage", "subset", "provider", "model", "channel", "succeeded",
                                 "usage_complete", "usage", "error_present")})
        if receipt["total_mismatch"]:
            mismatch_rows.append(receipt["path"])
        if receipt["request_binding_mismatch"]:
            binding_rows.append(receipt["path"])
    group_rows = []
    for key in sorted(groups):
        stage, subset, provider, model, channel = key
        row = {"stage": stage, "subset": subset, "provider": provider, "model": model, "channel": channel}
        row.update(groups[key])
        group_rows.append(row)
    stage_map = {}
    for row in group_rows:
        key = (row["stage"], row["subset"])
        stage = stage_map.setdefault(key, {"stage": row["stage"], "subset": row["subset"],
                                           "attempted": 0, "completed": 0, "succeeded_true": 0,
                                           "succeeded_false": 0, "succeeded_unknown": 0,
                                           "known_complete_usage": 0, "unknown_usage": 0,
                                           "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                                           "latency_s_known": 0.0, "latency_known_receipts": 0,
                                           "latency_unknown_receipts": 0, "failed_receipts": 0,
                                           "failed_known_total_tokens": 0, "usage_total_mismatch": 0,
                                           "unknown_identity": 0, "request_sha_available": 0,
                                           "request_binding_mismatch": 0, "by_provider_model": []})
        for field in ("attempted", "completed", "succeeded_true", "succeeded_false", "succeeded_unknown",
                      "known_complete_usage", "unknown_usage", "prompt_tokens", "completion_tokens",
                      "total_tokens", "latency_s_known", "latency_known_receipts",
                      "latency_unknown_receipts", "failed_receipts", "failed_known_total_tokens",
                      "usage_total_mismatch", "unknown_identity", "request_sha_available",
                      "request_binding_mismatch"):
            stage[field] += row[field]
        stage["by_provider_model"].append({key: row[key] for key in
                                           ("provider", "model", "channel", "attempted", "known_complete_usage",
                                            "unknown_usage", "prompt_tokens", "completion_tokens", "total_tokens",
                                            "latency_s_known", "failed_receipts")})
    stage_rows = [stage_map[key] for key in sorted(stage_map)]
    overall = {field: sum(row[field] for row in stage_rows) for field in
               ("attempted", "completed", "succeeded_true", "succeeded_false", "succeeded_unknown",
                "known_complete_usage", "unknown_usage", "prompt_tokens", "completion_tokens",
                "total_tokens", "latency_s_known", "latency_known_receipts", "latency_unknown_receipts",
                "failed_receipts", "failed_known_total_tokens", "usage_total_mismatch",
                "unknown_identity", "request_sha_available", "request_binding_mismatch")}
    overall["coverage_known_complete_usage"] = (overall["known_complete_usage"] / overall["attempted"]
                                                if overall["attempted"] else None)
    duplicates = {}
    for field in ("file_sha256", "request_sha256", "response_sha256", "prompt_sha256"):
        counter = Counter(receipt[field] for receipt in real if receipt[field])
        groups_out = []
        for value, count in counter.items():
            if count > 1:
                groups_out.append({"value": value, "count": count,
                                   "receipts": [{"path": r["path"], "stage": r["stage"], "subset": r["subset"]}
                                                for r in real if r[field] == value]})
        duplicates[field] = groups_out
    duplicates["dedup_applied"] = False
    return {"overall": overall, "stages": stage_rows, "groups": group_rows,
            "unknown_usage_rows": unknown_usage_rows, "unknown_latency_rows": unknown_latency_rows,
            "unknown_identity_rows": unknown_identity_rows, "failed_rows": failed_rows,
            "usage_total_mismatch_rows": mismatch_rows, "request_binding_mismatch_rows": binding_rows,
            "fixture_receipts": [{"path": r["path"], "stage": r["stage"], "model": r["model"],
                                  "provider": r["provider"], "schema": None} for r in fixtures],
            "duplicates": duplicates,
            "accepted_counts": accepted_counts or {}, "unreadable": unreadable or [],
            "request_sha_sources": dict(Counter(receipt["request_sha_source"] for receipt in real)),
            "prompt_sha_sources": dict(Counter(receipt["prompt_sha_source"] for receipt in real)),
            "request_sha_missing": sum(not receipt["request_sha256"] for receipt in real)}


def selftest() -> None:
    checks = 0

    def ok(condition: bool) -> None:
        nonlocal checks
        assert condition
        checks += 1

    raw = normalize_receipt(ROOT / "data/authors/confirmed-r2p1/raw/synthetic.json",
                            {"model": "m", "provider": "ollama", "succeeded": True, "response_text": "{}",
                             "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                             "latency_s": 1.0}, "author_raw", "synthetic", file_sha="1" * 64)
    unknown = normalize_receipt(ROOT / "data/reviews/x/u.outcome.json",
                                {"model": "m", "provider": "openai-compatible", "succeeded": True, "text": "{}",
                                 "usage": None, "latency_s": 2.0}, "blinded_review_v2", "x", file_sha="2" * 64)
    failed = normalize_receipt(ROOT / "data/reviews/x/f.outcome.json",
                               {"model": "m", "provider": "openai-compatible", "succeeded": False, "text": None,
                                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
                                "latency_s": 0.5}, "blinded_review_v2", "x", file_sha="3" * 64)
    fixture = normalize_receipt(ROOT / "data/reviews/x/fixture.outcome.json",
                                {"model": "fixture-only", "provider": "fixture", "succeeded": True, "text": "{}",
                                 "usage": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200}},
                                "blinded_review_v2", "x", file_sha="4" * 64)
    ok(raw["usage_complete"] and raw["succeeded"] is True)
    ok(is_accepted_record_path("data/authors/confirmed-r2p3/cf-006.json")
       and not is_receipt_path("data/authors/confirmed-r2p3/cf-006.json"))
    summary = summarize([raw, unknown, failed, fixture], accepted_counts={"confirmed-r2p3": 1})
    ok(summary["overall"]["attempted"] == 3 and summary["overall"]["known_complete_usage"] == 2)
    ok(summary["overall"]["total_tokens"] == 22 and summary["overall"]["unknown_usage"] == 1)
    ok(summary["overall"]["failed_receipts"] == 1 and summary["overall"]["failed_known_total_tokens"] == 7)
    ok(len(summary["fixture_receipts"]) == 1 and summary["overall"]["attempted"] == 3)
    print(json.dumps({"selftest_checks": checks, "passed": checks}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    receipts, unreadable, patterns = enumerate_receipts()
    accepted_counts = {name: len(list((ROOT / "data/authors" / name).glob("*.json")))
                       for name in ("confirmed-r2p1", "confirmed-r2p2", "confirmed-r2p3")}
    accepted_counts["composite-real-r2p1"] = len(
        list((ROOT / "data/composite/authors-real-r2p1").glob("*.json")))
    result = summarize(receipts, accepted_counts=accepted_counts, unreadable=unreadable)
    result.update({
        "schema": "relation-call-costs/1",
        "scope": "REL relation line only; memory line excluded; programming/Codex tokens excluded",
        "known_receipt_lower_bound": {
            "note": "Only receipts with complete numeric prompt/completion/total usage are summed; "
                    "missing or partial usage is listed separately and never treated as zero.",
            "covered_receipts": result["overall"]["known_complete_usage"],
            "coverage_of_enumerated_receipts": result["overall"]["coverage_known_complete_usage"],
            "prompt_tokens": result["overall"]["prompt_tokens"],
            "completion_tokens": result["overall"]["completion_tokens"],
            "total_tokens": result["overall"]["total_tokens"]},
        "api_pricing": {"computed": False, "currency": None,
                        "note": "No monetary price evidence; Ollama tokens are not API currency."},
        "memory_line": {"calls": 340, "total_tokens": 430194, "included_here": False,
                        "source": "parent-fixed; not added to relation totals"},
        "not_included": [
            "accepted author cumulative records (top-level confirmed-r2p1/r2p2/r2p3 and composite authors-real-r2p1) — usage not summed",
            "fixture receipts — excluded from counts",
            "receipts with missing/partial usage — listed, not counted as zero",
            "calls with no receipt file — unknown; cannot be enumerated",
            "calls.jsonl / events.jsonl ledgers and analysis files — excluded to avoid double counting or derived duplicates",
            "programming/agent tokens and Codex review tokens — unknown, not zero",
            "memory line — fixed separately at 340 calls / 430194 tokens; not added",
        ],
        "source_patterns": patterns,
        "field_limitations": [
            "Every enumerated file is one receipt; duplicate-like receipts are reported, not deduplicated.",
            "completed = non-empty response text; succeeded = explicit succeeded true; failed includes false/unknown.",
            "Known token totals are a lower bound over complete-usage receipts, not a complete spend total.",
            "Latency sums only numeric latency_s; missing latency is listed.",
            "Raw author receipts (including composite raw) have no request_sha256; prompt SHA is used for their duplicate scan.",
            "Request SHA for repair_v1 outcomes comes from the sibling request file when the outcome lacks it.",
        ],
    })
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                     encoding="utf-8", newline="\n")
    print(json.dumps({"schema": result["schema"], "attempted": result["overall"]["attempted"],
                      "completed": result["overall"]["completed"], "succeeded": result["overall"]["succeeded_true"],
                      "known_complete_usage": result["overall"]["known_complete_usage"],
                      "unknown_usage": result["overall"]["unknown_usage"],
                      "total_tokens_lower_bound": result["overall"]["total_tokens"],
                      "coverage": result["overall"]["coverage_known_complete_usage"],
                      "fixtures": len(result["fixture_receipts"]),
                      "accepted_counts": result["accepted_counts"],
                      "duplicate_groups": {key: len(value) for key, value in result["duplicates"].items()
                                           if key != "dedup_applied"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
