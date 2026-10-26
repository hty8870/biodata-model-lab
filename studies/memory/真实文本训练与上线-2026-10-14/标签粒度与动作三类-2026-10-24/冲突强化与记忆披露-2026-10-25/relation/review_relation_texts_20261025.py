from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from unittest.mock import patch

import build_conflict_data as B
import prepare_relation_reviews_20261025 as G

ROOT = Path(__file__).resolve().parent
MODELS = {"train-dev": "batiai/qwen3.8-27b:iq3", "heldout": "deepseek-v4-flash",
          "composite": "batiai/qwen3.8-27b:iq3"}
CHANNELS = {"train-dev": "ollama", "heldout": "chat", "composite": "ollama"}


def parse_review(text: str, unit: dict) -> list[dict]:
    result = json.loads(text)
    rows = result["pairs"]
    expected = {p["id"] for p in unit["pairs"]}
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Review ids must exactly cover the blinded unit")
    for row in rows:
        if row["relation"] not in B.K.COARSE or type(row["ambiguous"]) is not bool \
                or not isinstance(row["reason"], str) or not row["reason"].strip():
            raise ValueError("Invalid relation, ambiguity, or reason")
    return rows


def selftest() -> None:
    unit = {"id": "unit", "background": "", "pairs": [{"id": "p", "old": "x", "new": "x"}]}
    assert parse_review('{"pairs":[{"id":"p","relation":"equivalent","ambiguous":false,"reason":"same requirement"}]}', unit)[0]["relation"] == "equivalent"
    for text in ('{"pairs":[]}', '{"pairs":[{"id":"p","relation":"other","ambiguous":false,"reason":"x"}]}'):
        try:
            parse_review(text, unit)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid review accepted")
    captured = {}
    def transport(request, timeout):
        captured.update(json.loads(request.data))
        return io.BytesIO(json.dumps({"done": True, "done_reason": "stop", "response": "{}",
            "prompt_eval_count": 2, "eval_count": 1, "model": "fixture-only"}).encode())
    with patch.object(B.urllib.request, "urlopen", transport):
        outcome = B.OllamaChannel("fixture-only").complete("offline test", num_predict=4000)
    assert captured["options"]["num_ctx"] == 16384 and captured["options"]["num_predict"] == 4000
    assert captured["format"] == "json" and captured["think"] is False and outcome["succeeded"]
    print(json.dumps({"selftest_checks": 5, "passed": 5, "real_model_calls": 0}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--phase", choices=MODELS, default="train-dev")
    parser.add_argument("--version", choices=G.VERSIONS, default="v1")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--max-new", type=int)
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    prefix = f"{args.phase}-{args.version}"
    units_path = ROOT / "data/review_inputs" / f"{prefix}.units.jsonl"
    refs_path = ROOT / "data/review_inputs" / f"{prefix}.references.jsonl"
    source_path = ROOT / "data/review_inputs" / f"{prefix}.manifest.json"
    units = B.K.read_jsonl(units_path)
    source = B.K.read_json(source_path)
    assert B.K.sha256_file(units_path) == source["units_sha256"]
    assert B.K.sha256_file(refs_path) == source["references_sha256"]
    prompts = {u["id"]: G.prompt_for(u, args.version) for u in units}
    assert {k: B.K.sha256_text(v) for k, v in prompts.items()} == source["prompt_sha256_by_unit"]
    folder = ROOT / "data/reviews" / prefix
    model = MODELS[args.phase]
    channel_name = CHANNELS[args.phase]
    identity = {"model": model, "channel": channel_name, "thinking": False, "max_output": 4000,
                "num_ctx": 16384 if channel_name == "ollama" else None,
                "input_manifest_sha256": B.K.sha256_file(source_path),
                "adapter_sha256": B.K.sha256_file(Path(B.__file__)),
                "prompt_sha256_by_unit": source["prompt_sha256_by_unit"], "planned_calls": len(units)}
    calls = 0
    channel = None
    if args.collect:
        G.publish(folder / "run_manifest.json", json.dumps(identity, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        for unit in units:
            uid = unit["id"]
            request_path = folder / f"{uid}.request.json"
            outcome_path = folder / f"{uid}.outcome.json"
            if request_path.exists():
                continue
            if args.max_new is not None and calls >= args.max_new:
                break
            request = {"unit_id": uid, "prompt": prompts[uid], "prompt_sha256": B.K.sha256_text(prompts[uid]),
                       "input_manifest_sha256": identity["input_manifest_sha256"], "status": "reserved"}
            G.publish(request_path, json.dumps(request, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            if channel is None:
                channel = B.OllamaChannel(model, timeout=600) if channel_name == "ollama" else B.ChatChannel(model, max_tokens=4000)
                if channel_name == "chat":
                    channel.prepare()
            try:
                outcome = channel.complete(prompts[uid], num_predict=4000) if channel_name == "ollama" else channel.complete(prompts[uid])
            except Exception as error:
                outcome = {"attempted": True, "succeeded": False, "text": None, "usage": None,
                           "error_kind": type(error).__name__, "model": model, "provider": channel_name}
            outcome.update(unit_id=uid, prompt_sha256=request["prompt_sha256"], request_sha256=B.K.sha256_file(request_path))
            G.publish(outcome_path, json.dumps(outcome, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
            calls += 1
    references = {r["review_id"]: r for r in B.K.read_jsonl(refs_path)}
    parsed, failures, pending, not_attempted = [], [], [], []
    total_tokens = 0
    usage_unknown = 0
    for unit in units:
        uid = unit["id"]
        request_path = folder / f"{uid}.request.json"
        outcome_path = folder / f"{uid}.outcome.json"
        if not request_path.exists():
            not_attempted.append(uid)
            continue
        if not outcome_path.exists():
            pending.append(uid)
            continue
        outcome = B.K.read_json(outcome_path)
        assert outcome["request_sha256"] == B.K.sha256_file(request_path)
        assert outcome["prompt_sha256"] == B.K.sha256_text(prompts[uid])
        usage = outcome.get("usage") or {}
        total_tokens += usage.get("total_tokens") or 0
        usage_unknown += usage.get("total_tokens") is None
        if not outcome["succeeded"]:
            failures.append({"unit_id": uid, "reason": outcome.get("error_kind", "unsuccessful")})
            continue
        try:
            parsed.extend(parse_review(outcome["text"], unit))
        except (ValueError, KeyError, TypeError) as error:
            failures.append({"unit_id": uid, "reason": str(error)})
    disagreements = [dict(r, reference=references[r["id"]]) for r in parsed
                     if r["relation"] != references[r["id"]]["expected_coarse"] or r["ambiguous"]]
    summary = {"schema": "blinded-relation-review/1", "phase": args.phase, "model": model,
        "planned_units": len(units), "planned_pairs": len(references), "new_calls": calls,
        "valid_reviewed_pairs": len(parsed), "not_attempted_units": len(not_attempted), "pending_unknown": pending,
        "failed_units": failures, "disagreements_or_ambiguity": len(disagreements),
        "unambiguous_agreement": len(parsed) - len(disagreements),
        "tokens_all_completed": total_tokens, "usage_unknown": usage_unknown,
        "input_manifest_sha256": identity["input_manifest_sha256"],
        "runner_sha256": B.K.sha256_file(Path(__file__)),
        "not_human_gold": True, "disagreements": disagreements, "reviews": parsed}
    if folder.exists():
        digest = B.K.sha256_json(summary)
        G.publish(folder / f"analysis-{digest[:16]}.json", json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("disagreements", "reviews")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
