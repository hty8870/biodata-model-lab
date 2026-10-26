from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import build_conflict_data as B

ROOT = Path(__file__).resolve().parent
AUTHORS = ROOT / "data/authors/confirmed-r2p1"
REPAIRS = AUTHORS / "repair_v1"
_CHANNELS: dict[tuple[str, str], object] = {}
POLARITY_WORDS = {"require": ("必须", "需要", "应当"),
                  "prohibit": ("不得", "禁止", "不允许")}


def channel_for(model: str, provider: str) -> tuple[object, str]:
    """按 (model, channel) 缓存通道；ollama 与 chat 不复用同一实例。"""
    kind = "ollama" if provider == "ollama" else "chat"
    key = (model, kind)
    if key not in _CHANNELS:
        _CHANNELS[key] = (B.OllamaChannel(model, timeout=600, num_ctx=16384)
                          if kind == "ollama" else B.ChatChannel(model, max_tokens=4000))
    return _CHANNELS[key], kind


def complete_with(channel: object, kind: str, request: str) -> dict:
    if kind == "ollama":
        return channel.complete(request, num_predict=4000)
    return channel.complete(request)


def report_path(split: str, content_sha256: str) -> Path:
    return ROOT / f"audits/localized-repair-{split}-{content_sha256[:16]}.json"


def errors_for(spec: dict, payload: dict, model: str) -> list[str]:
    record = {"background": payload["background"], "items": [dict(pair_id=k, **v) for k, v in payload["items"].items()],
              "model": model, "source": "author"}
    return list(dict.fromkeys(B.validate_author_payload(spec, payload) + B.rows_for_family(spec, record)[1]))


def changes_for(spec: dict, errors: list[str]) -> set[tuple[str, str]]:
    items = {i["pair_id"]: i for i in spec["items"]}
    changes = set()
    for error in errors:
        address = error.split(":", 1)[0]
        pair_id, side = address.rsplit(".", 1) if "." in address else (address, "new")
        if pair_id not in items or side not in ("old", "new"):
            raise ValueError(f"Not a localized text error: {error}")
        item = items[pair_id]
        if item["kind"] == "dyad_flip":
            pair_id, side = item["derived_from"], "new"
        if item["kind"] == "control" and side == "old":
            pair_id = B.dyad_side_id(item["matched_window_id"], "a")
        changes.add((pair_id, side))
    return changes


def apply_updates(payload: dict, response: str, allowed: set[tuple[str, str]]) -> dict:
    parsed = json.loads(response)
    if not isinstance(parsed, dict) or set(parsed) != {"updates"}:
        raise ValueError("Expected updates object")
    updates = parsed["updates"]
    keys = [(r["pair_id"], r["side"]) for r in updates]
    if len(keys) != len(set(keys)) or set(keys) != allowed:
        raise ValueError("Update keys do not exactly match requested changes")
    result = copy.deepcopy(payload)
    for row in updates:
        if set(row) != {"pair_id", "side", "text"} or not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("Invalid update shape")
        result["items"][row["pair_id"]][row["side"]] = row["text"]
    return result


def reorder_equivalent_failures(spec: dict, payload: dict, errors: list[str]) -> tuple[dict, list[dict]]:
    """唯一失败为 equivalent 的 old==new 时，调用既有 helper 做纯合取分句换序。"""
    if not errors or not all("old == new after normalisation" in error for error in errors):
        return payload, []
    from offline_complete_equivalents_20261025 import reorder_plain_equivalent

    items = {i["pair_id"]: i for i in spec["items"]}
    result = copy.deepcopy(payload)
    reorders = []
    for error in errors:
        address = error.split(":", 1)[0]
        if "." in address or address not in items or address not in result["items"]:
            continue
        before = result["items"][address]["new"]
        try:
            after = reorder_plain_equivalent(before, items[address])
        except (AssertionError, KeyError, TypeError, ValueError):
            continue
        result["items"][address]["new"] = after
        reorders.append({"pair_id": address, "before": before, "after": after,
                         "operation": "rotate unchanged conjunctive clauses; factor semantics and all tokens unchanged"})
    return result, reorders


def prompt_for(spec: dict, payload: dict, errors: list[str], changes: set[tuple[str, str]]) -> str:
    items = {i["pair_id"]: i for i in spec["items"]}
    targets = []
    for pair_id, side in sorted(changes):
        item = items[pair_id]
        gate = dict(item["gate"][side], items_scope=item[side].get("items_scope"))
        action = item[side]["action"]
        targets.append({"pair_id": pair_id, "side": side, "current_text": payload["items"][pair_id][side],
                        "other_side_text": payload["items"][pair_id]["old" if side == "new" else "new"],
                        "intended_semantics": item["semantic"][side], "action": action,
                        "polarity_words": list(POLARITY_WORDS[action]),
                        "required_phrases": B._required_phrases(gate),
                        "surface": (item.get("surface") or {}).get(side)})
    return ("修订已有记忆语料中被点名的文字，不新增要求，不改变作用域、数值、否定、例外、时间或指代。"
            "只改targets列出的side，其余内容由程序原样保留。不能逐字复制另一侧。"
            "严格遵守每个target的action与polarity_words：action=require只能使用必须/需要/应当中的要求词，"
            "action=prohibit只能使用不得/禁止/不允许中的禁止词；不得把要求写成禁止，也不得把禁止写成要求。"
            "required_phrases中的「必须/须」只约束逐字片段，不表示行动极性。"
            "要求只针对的事项要完整列全；引用清单时用所列条目说明清单是范围。"
            "逐字且唯一片段必须恰好出现一次，并优先于一般措辞与surface偏好。"
            "surface仅作可选措辞偏好；数值range/bounds仍按surface写，保留所有对象全称。"
            "只输出JSON对象，不能有Markdown。"
            "格式：{\"updates\":[{\"pair_id\":\"...\",\"side\":\"new\",\"text\":\"...\"}]}\n"
            + json.dumps({"background": payload["background"], "errors": errors, "targets": targets}, ensure_ascii=False))


def selftest() -> None:
    checks = 0

    def ok(condition: bool) -> None:
        nonlocal checks
        assert condition
        checks += 1

    payload = {"background": "unchanged", "items": {"x": {"old": "old", "new": "new"}, "y": {"old": "a", "new": "b"}}}
    changed = apply_updates(payload, '{"updates":[{"pair_id":"x","side":"new","text":"revised"}]}', {("x", "new")})
    ok(changed["items"]["x"]["new"] == "revised" and payload["items"]["x"]["new"] == "new")
    ok(changed["items"]["y"] == payload["items"]["y"] and changed["background"] == payload["background"])
    for response in ('{"updates":[]}', '{"updates":[{"pair_id":"y","side":"new","text":"wrong"}]}',
                     '{"updates":[{"pair_id":"x","side":"new","text":"a"},{"pair_id":"x","side":"new","text":"b"}]}'):
        try:
            apply_updates(payload, response, {("x", "new")})
        except ValueError:
            ok(True)
        else:
            raise AssertionError("Invalid change accepted")

    class _FakeOllama:
        def __init__(self, model, **kwargs):
            self.model, self.kwargs, self.calls = model, kwargs, []

        def complete(self, prompt, **kwargs):
            self.calls.append(kwargs)
            return {"text": "ok", "succeeded": True}

    class _FakeChat:
        def __init__(self, model, **kwargs):
            self.model, self.kwargs, self.calls = model, kwargs, []

        def complete(self, prompt):
            self.calls.append({})
            return {"text": "ok", "succeeded": True}

    original = (B.OllamaChannel, B.ChatChannel)
    B.OllamaChannel, B.ChatChannel = _FakeOllama, _FakeChat
    _CHANNELS.clear()
    try:
        ollama, kind = channel_for("batiai/qwen3.8-27b:iq3", "ollama")
        ok(kind == "ollama" and isinstance(ollama, _FakeOllama)
           and ollama.kwargs == {"timeout": 600, "num_ctx": 16384})
        same, _ = channel_for("batiai/qwen3.8-27b:iq3", "ollama")
        ok(same is ollama)
        chat, chat_kind = channel_for("deepseek-v4-flash", "openai-compatible")
        ok(chat_kind == "chat" and isinstance(chat, _FakeChat)
           and chat.kwargs == {"max_tokens": 4000})
        complete_with(ollama, "ollama", "p")
        complete_with(chat, "chat", "p")
        ok(ollama.calls == [{"num_predict": 4000}] and chat.calls == [{}])
    finally:
        B.OllamaChannel, B.ChatChannel = original
        _CHANNELS.clear()

    equivalent = {"pair_id": "s1", "expected_coarse": "equivalent",
                  "new": {"measure": {"kind": "none"}, "exception": {"kind": "none"},
                          "window": {"kind": "long_term"}}}
    conflict = {"pair_id": "s2", "expected_coarse": "conflict", "new": dict(equivalent["new"])}
    reorder_spec = {"items": [equivalent, conflict]}
    reorder_payload = {"background": "bg", "items": {
        "s1": {"old": "必须核对正文，长期生效，只针对摘要与结论。",
               "new": "同一用户必须核对正文，长期生效，只针对摘要与结论。"},
        "s2": {"old": "必须引用近五年文献，只针对摘要。", "new": "该要求只针对摘要。"}}}
    reordered, lineage = reorder_equivalent_failures(
        reorder_spec, reorder_payload,
        ["s1: old == new after normalisation", "s2: old == new after normalisation"])
    ok(reordered["items"]["s1"]["new"] == "长期生效，只针对摘要与结论，同一用户必须核对正文。"
       and reordered["items"]["s1"]["old"] == reorder_payload["items"]["s1"]["old"]
       and reordered["background"] == "bg")
    ok(reordered["items"]["s2"] == reorder_payload["items"]["s2"]
       and [row["pair_id"] for row in lineage] == ["s1"])
    ok(reorder_equivalent_failures(
        reorder_spec, reorder_payload,
        ["s1: old == new after normalisation", "s2.new: referent must be a partition"])[1] == [])

    def mock_item(pair_id: str, action: str, form: str, items_scope) -> dict:
        factor = {"subject": "同一用户", "items_scope": items_scope, "slot": "citation_year_window",
                  "object": "近五年文献", "scope": "引用", "action": action,
                  "measure": {"kind": "none", "low": None, "high": None},
                  "exception": {"kind": "none", "item": None},
                  "window": {"kind": "long_term", "year": None}, "referent": "待引清单"}
        gate = {"numbers": [], "polarity_classes": [list(POLARITY_WORDS[action])],
                "scope_tokens": ["引用"], "measure_kind": "none", "exception_item": None,
                "window_tokens": [], "referent": "待引清单", "object_phrase": "近五年文献",
                "exact_tokens": []}
        surface = {"kind": "paraphrase", "form": form}
        return {"pair_id": pair_id, "old": dict(factor), "new": dict(factor),
                "semantic": {"old": "旧语义", "new": "新语义"},
                "gate": {"old": dict(gate), "new": dict(gate)},
                "surface": {"old": dict(surface), "new": dict(surface)}}

    prompt_spec = {"items": [mock_item("r", "require", "soft", ["摘要"]),
                             mock_item("p", "prohibit", "strong", None)]}
    prompt_payload = {"background": "bg", "items": {
        "r": {"old": "旧要求", "new": "新要求"},
        "p": {"old": "旧禁止", "new": "新禁止"}}}
    prompt = prompt_for(prompt_spec, prompt_payload, [], {("r", "new"), ("p", "new")})
    prompt_targets = {t["pair_id"]: t for t in json.loads(prompt.split("\n", 1)[1])["targets"]}
    ok(prompt_targets["r"]["action"] == "require"
       and prompt_targets["r"]["polarity_words"] == list(POLARITY_WORDS["require"])
       and prompt_targets["r"]["surface"] == {"kind": "paraphrase", "form": "soft"})
    prohibit_target = json.dumps(prompt_targets["p"], ensure_ascii=False)
    ok(prompt_targets["p"]["action"] == "prohibit"
       and prompt_targets["p"]["polarity_words"] == list(POLARITY_WORDS["prohibit"])
       and not any(word in prohibit_target for word in ("必须", "需要", "应当"))
       and prompt_targets["p"]["surface"] == {"kind": "paraphrase", "form": "strong"}
       and "surface=soft时必须出现需要或应当" not in prompt
       and "surface=strong时必须出现必须或须" not in prompt
       and "action=prohibit只能使用不得/禁止/不允许" in prompt)
    ok(report_path("heldout", "abcdef0123456789") == ROOT / "audits/localized-repair-heldout-abcdef0123456789.json"
       and report_path("heldout", "abcdef0123456789") != ROOT / "audits/localized-repair-20261025.json")
    print(json.dumps({"selftest_checks": checks, "passed": checks}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--budget-calls", type=int, default=0)
    parser.add_argument("--split", choices=("train", "dev", "heldout"))
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if args.collect and args.budget_calls <= 0:
        parser.error("--collect requires --budget-calls")
    specs = {s["family_id"]: s for s in B.K.read_jsonl(B.K.SPECS_PATH)}
    raw_paths = sorted((AUTHORS / "raw").glob("*.json")) + sorted((ROOT / "data/authors/pilot-r2p1-train/raw").glob("*.json"))
    bases = {}
    for path in raw_paths:
        raw = B.K.read_json(path)
        assert raw["family_id"] not in bases
        bases[raw["family_id"]] = (raw, path)
    reports = []
    calls = 0
    for family, (raw, raw_path) in sorted(bases.items()):
        spec = specs[family]
        split = spec["split"]
        if args.split and split != args.split:
            continue
        expected_models = B.K.AUTHOR_PLAN[split]
        assert raw["split"] == split
        assert raw["model"] in expected_models
        if split == "heldout":
            assert raw["model"] == "batiai/qwen3.8-27b:iq3" and raw["provider"] == "ollama"
        else:
            assert raw["model"] == "deepseek-v4-flash" and raw["provider"] != "ollama"
        assert raw["spec_sha256"] == B.K.sha256_json(spec)
        assert raw["prompt_sha256"] == B.K.sha256_text(B.build_prompt(spec))
        channel_kind = "ollama" if raw["provider"] == "ollama" else "chat"
        target = AUTHORS / f"{family}.json"
        if B.verified_record(target, spec, expected_models):
            reports.append({"family_id": family, "split": split,
                            "skipped": "accepted record is read-only"})
            continue
        payload = B.parse_author_response(raw["response_text"], spec)
        before = errors_for(spec, payload, raw["model"])
        report = {"family_id": family, "split": split, "errors_before": before, "new_call": False}
        reports.append(report)
        if not before:
            continue
        changes = changes_for(spec, before)
        report["requested_changes"] = sorted(changes)
        if not args.collect or calls >= args.budget_calls:
            continue
        request = prompt_for(spec, payload, before, changes)
        request_path = REPAIRS / f"{family}.request.json"
        outcome_path = REPAIRS / f"{family}.outcome.json"
        if request_path.exists() and not outcome_path.exists():
            raise RuntimeError(f"Pending or unknown request; never resend {family}")
        if outcome_path.exists() and not request_path.exists():
            raise RuntimeError(f"Outcome without request; refusing {family}")
        if outcome_path.exists():
            outcome = B.K.read_json(outcome_path)
            old_request = B.K.read_json(request_path)
            assert old_request["prompt_sha256"] == B.K.sha256_text(request)
            assert old_request["base_raw_sha256"] == B.K.sha256_file(raw_path)
            assert old_request["spec_sha256"] == raw["spec_sha256"]
            assert outcome.get("provider") == raw["provider"]
        else:
            B.K.write_json_atomic(request_path, {"family_id": family, "prompt": request,
                "prompt_sha256": B.K.sha256_text(request), "base_raw": raw_path.relative_to(ROOT).as_posix(),
                "base_raw_sha256": B.K.sha256_file(raw_path), "spec_sha256": raw["spec_sha256"],
                "text_gate_sha256": B.K.sha256_file(Path(B.S.__file__)), "status": "reserved"})
            B.append_call_ledger(REPAIRS / "events.jsonl", {"event": "reserved", "family_id": family,
                "request_sha256": B.K.sha256_file(request_path), "created": B.K.utc_now()})
            channel, actual_kind = channel_for(raw["model"], raw["provider"])
            assert actual_kind == channel_kind
            if channel_kind == "chat":
                channel.prepare()
            outcome = complete_with(channel, channel_kind, request)
            B.K.write_json_atomic(outcome_path, outcome)
            calls += 1
            report["new_call"] = True
            B.append_call_ledger(REPAIRS / "events.jsonl", {"event": "finished", "family_id": family,
                "outcome_sha256": B.K.sha256_file(outcome_path), "usage": outcome.get("usage"),
                "succeeded": outcome["succeeded"], "created": B.K.utc_now()})
        report["outcome"] = outcome_path.relative_to(ROOT).as_posix()
        report["channel"] = channel_kind
        if not outcome["succeeded"]:
            report["errors_after"] = ["repair response unsuccessful"]
            continue
        try:
            repaired = apply_updates(payload, outcome["text"], changes)
            after = errors_for(spec, repaired, raw["model"])
        except (ValueError, KeyError, TypeError) as error:
            report["errors_after"] = [str(error)]
            continue
        report["errors_after"] = after
        reorders = []
        if after:
            repaired, reorders = reorder_equivalent_failures(spec, repaired, after)
            if reorders:
                report["clause_reorders"] = reorders
                after = errors_for(spec, repaired, raw["model"])
            report["errors_after"] = after
            if after:
                continue
        usage = {k: raw["usage"][k] + outcome["usage"][k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        identity = {k: raw.get(k) for k in ("provider", "model", "base_url_sha256", "thinking", "max_tokens")}
        identity.update({"channel": channel_kind, "derivation": "localized-text-repair-v1",
            "prompt_sha256_is_base_generation": True, "base_raw": raw_path.relative_to(ROOT).as_posix(),
            "base_raw_sha256": B.K.sha256_file(raw_path), "repair_request_sha256": B.K.sha256_file(request_path),
            "repair_outcome_sha256": B.K.sha256_file(outcome_path), "text_gate_sha256": B.K.sha256_file(Path(B.S.__file__))})
        if reorders:
            identity["clause_reorders"] = reorders
            identity["clause_reorder_source"] = "offline_complete_equivalents_20261025.reorder_plain_equivalent"
        B.write_author_record(AUTHORS, spec, repaired, model=raw["model"], prompt=B.build_prompt(spec), usage=usage, source="author", identity=identity)
        assert B.verified_record(target, spec, expected_models)
        report["accepted_record_sha256"] = B.K.sha256_file(target)
    split_label = args.split or "all"
    content = {"split": split_label, "reports": reports}
    content_sha256 = B.K.sha256_json(content)
    path = report_path(split_label, content_sha256)
    summary = {"schema": "localized-author-repair/1", "split": split_label,
               "scanned_raw_families": len(bases), "scope_families": len(reports), "new_calls": calls,
               "originally_failed": sum(bool(r.get("errors_before")) for r in reports),
               "repair_accepted": sum("accepted_record_sha256" in r for r in reports),
               "skipped_accepted": sum("skipped" in r for r in reports),
               "report_content_sha256": content_sha256,
               "report_path": path.relative_to(ROOT).as_posix(), "reports": reports,
               "runner_sha256": B.K.sha256_file(Path(__file__))}
    if args.collect:
        B.K.write_json_atomic(path, summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
