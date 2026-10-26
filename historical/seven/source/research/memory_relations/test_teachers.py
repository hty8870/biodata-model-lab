                       
"""teachers CLI 的离线单测：mock 传输层、禁网络。

覆盖：真实输入闸（schema.materialize_request + 生产 M._request）、生产输出闸、
strict/normalization 报告、账本 attempt 行与崩溃预算、指纹与换 model/mode 重算、
账本尾行 fail-closed 与显式修复、调用边界（URL 白名单/重定向/有界读取）、输出预算与
长度拒绝、usage/字段名/凭据脱敏、外部 manifest 策略、以及输出行通过 schema.validate_row。

schema 模块与测试同目录（集成后为唯一真源）；真实推理不在本套件内执行。
"""
from __future__ import annotations

import importlib
import json
import socket
import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.memory_relations import teachers as T              


def _load_schema():
    try:
        return importlib.import_module("research.memory_relations.schema"), "local"
    except ImportError:
        return None, "missing"


S, SCHEMA_ORIGIN = _load_schema()
pytestmark = pytest.mark.skipif(S is None, reason="schema 模块不可用（未集成且无主树副本）")

QUOTE = "以后汇报先给结论，再给依据。"
REWRITTEN_QUOTE = "汇报时先给结论，然后再给依据。"
KEY = "sk-zen-test-SECRET-4f21"
FIVE = {"relation": "duplicate", "related_ids": ["r1"], "reason": "依据比较。",
        "evidence_refs": ["e1"], "error_code": None}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network call in offline test")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


class FakeTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, url, payload, headers, timeout):
        self.calls.append({"url": url, "payload": payload, "headers": headers, "timeout": timeout})
        if not self.replies:
            raise AssertionError("unexpected extra provider call")
        item = self.replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def ollama_reply(content, done_reason="stop", **extra):
    body = {"message": {"role": "assistant", "content": content}, "done": True,
            "done_reason": done_reason, "prompt_eval_count": 11, "eval_count": 7}
    body.update(extra)
    return T.HttpReply(200, json.dumps(body, ensure_ascii=False))


def openai_reply(content, finish="stop"):
    return T.HttpReply(200, json.dumps({
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 4}}, ensure_ascii=False))


def five(relation="duplicate", related=("r1",), refs=("e1",), reason="依据比较。", code=None):
    return {"relation": relation, "related_ids": list(related), "reason": reason,
            "evidence_refs": list(refs), "error_code": code}


def stored_request(*, quote=QUOTE, candidate_text="汇报先给结论，再给依据。", extra_rule=None):
    existing = [{"id": "r1", "text": "汇报先说结论，再解释原因。", "trigger": "汇报时",
                 "tier": "explicit", "permanence": "persistent", "evidence": [], "scope": "same"}]
    if extra_rule:
        existing.append(extra_rule)
    return {
        "candidate": {"text": candidate_text, "trigger": "汇报时", "tier": "explicit",
                      "permanence": "persistent", "evidence": [quote], "scope": "project"},
        "existing": existing,
        "evidence": [{"id": "e1", "text": quote}],
        "context": {"scope": "project"},
    }


def make_row(rid="x1", *, request=None, label="duplicate", label_origin="construction",
             related=("r1",), refs=("e1",), phenomena=(), **extra):
    row = {"id": rid, "group_id": "g1", "split": "train", "request": request or stored_request(),
           "label": label, "label_origin": label_origin, "related_ids": list(related),
           "evidence_refs": list(refs), "phenomena": dict(phenomena)
           if type(phenomena) is dict else list(phenomena)}
    row.update(extra)
    return row


def write_input(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def ollama_teacher(transport):
    return T.Teacher("ollama", "stub-model", T.OLLAMA_BASE_URL, transport=transport)


def opencode_teacher(transport, key=KEY):
    return T.Teacher("opencode", "deepseek-v4-flash", T.OPENCODE_BASE_URL, key=key, transport=transport)


def ledger_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def attempts_count(lines, step=None):
    return len([line for line in lines if line.get("kind") == "attempt"
                and (step is None or line.get("step") == step)])


def prompt_of(transport, index=0):
    return transport.calls[index]["payload"]["messages"][0]["content"]


def prompt_request_json(prompt, separator="事实 JSON：\n"):
    return json.loads(prompt.split(separator, 1)[1])


def paraphrase_reply():
    return {"candidate": {"text": "先给结论，再补依据。", "trigger": "汇报时",
                          "evidence": [REWRITTEN_QUOTE]},
            "existing": [{"id": "r1", "text": "汇报先讲结论，再解释理由。", "trigger": "汇报时",
                          "evidence": []}],
            "evidence": [{"id": "e1", "text": REWRITTEN_QUOTE}]}


def test_label_roundtrip_resume_ledger_contract(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row("x1"), make_row("x2"), make_row("x3")])
    transport = FakeTransport([ollama_reply(json.dumps(five(), ensure_ascii=False))] * 3)
    teacher = ollama_teacher(transport)

    first = T.run(input_path, output_path, teacher, "label", 2, 180.0)
    assert first["processed"] == 2 and first["ok"] == 2 and first["remaining"] == 1
    second = T.run(input_path, output_path, teacher, "label", 5, 180.0)
    assert second["processed"] == 1 and second["skipped_done"] == 2 and second["remaining"] == 0
    assert len(transport.calls) == 3

    lines = ledger_lines(output_path)
    assert attempts_count(lines) == 3
    entries = T.load_records(output_path)
    assert sorted(entries) == ["x1", "x2", "x3"]
    entry = entries["x1"]
    assert set(entry) == {"kind", "id", "row", "teacher"}
    row = entry["row"]
    assert row["request"] == stored_request()                                                 
    assert set(row) == {"id", "group_id", "split", "request", "label", "label_origin",
                        "related_ids", "evidence_refs", "phenomena"}
    assert row["label"] == "duplicate" and row["label_origin"] == "teacher_single"
    assert row["related_ids"] == ["r1"] and row["evidence_refs"] == ["e1"]

    meta = entry["teacher"]
    assert meta["ok"] is True and meta["error_code"] is None and meta["retryable"] is False
    assert meta["attempts"] == 1 and meta["input_ok"] is True and meta["raw_strict_ok"] is True
    assert len(meta["fingerprint"]) == 64
    assert set(meta["versions"]) == {"prompt_sha256", "paraphrase_prompt_sha256", "validator_sha256"}
    assert len(meta["versions"]["validator_sha256"]) == 64
    call = meta["calls"][0]
    assert call["step"] == "label" and call["provider"] == "ollama" and call["model"] == "stub-model"
    assert call["base_url"] == T.OLLAMA_BASE_URL and call["mode"] == "label"
    assert call["fingerprint"] == meta["fingerprint"]
    assert call["usage"] == {"prompt_tokens": 11, "completion_tokens": 7}
    assert len(call["raw_reply_sha256"]) == 64 and call["raw_reply_chars"] > 0
    assert "raw_reply" not in call                            
    assert len(call["request_sha256"]) == 64 and len(call["model_request_sha256"]) == 64

    payload = transport.calls[0]["payload"]
    prompt = prompt_of(transport)
    assert prompt.startswith(T._PROMPT)
    projected = prompt_request_json(prompt)
    assert set(projected) == {"candidate", "existing", "evidence", "context"}
    assert set(projected["candidate"]) == {"text", "trigger", "tier", "evidence", "scope"}
    assert "permanence" not in json.dumps(projected)
    assert projected["context"]["replaceable_ids"] == ["r1"]
    assert payload["think"] is False and payload["stream"] is False
    assert payload["options"] == {"num_ctx": T.OLLAMA_NUM_CTX, "num_predict": T.OUTPUT_MAX_TOKENS}
    assert transport.calls[0]["timeout"] == 180.0


def test_prompt_projection_excludes_row_and_feature_fields(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    row = make_row("x1", label="LEAKMARK-LABEL-7", phenomena=["LEAKMARK-PHEN-7"],
                   expected="LEAKMARK-EXPECTED-7", notes="LEAKMARK-NOTES-7")
    write_input(input_path, [row])
    transport = FakeTransport([ollama_reply(json.dumps(five(), ensure_ascii=False))])
    assert T.run(input_path, output_path, ollama_teacher(transport), "label", 1, 180.0)["ok"] == 1
    prompt = prompt_of(transport)
    for marker in ("LEAKMARK-LABEL-7", "LEAKMARK-PHEN-7", "LEAKMARK-EXPECTED-7", "LEAKMARK-NOTES-7"):
        assert marker not in prompt
    assert "permanence" not in prompt
    entry = T.load_records(output_path)["x1"]
    assert entry["row"]["notes"] == "LEAKMARK-NOTES-7"               
    assert entry["row"]["phenomena"] == ["LEAKMARK-PHEN-7"]


def test_strict_vs_normalization_reporting(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row(f"x{i}") for i in range(1, 7)])
    extra_field = five()
    extra_field[KEY] = "echo"
    missing_field = five()
    missing_field.pop("reason")
    fenced = "```json\n" + json.dumps(five(), ensure_ascii=False) + "\n```"
    transport = FakeTransport([
        ollama_reply(json.dumps(five(), ensure_ascii=False)),
        ollama_reply(json.dumps(extra_field, ensure_ascii=False)),
        ollama_reply(json.dumps(missing_field, ensure_ascii=False)),
        ollama_reply(fenced),
        ollama_reply("not json"),
        ollama_reply(json.dumps({"related_ids": [], "reason": "x", "evidence_refs": [], "error_code": None},
                                ensure_ascii=False)),
    ])
    summary = T.run(input_path, output_path, ollama_teacher(transport), "label", 6, 180.0)
    assert summary["ok"] == 3 and summary["failed"] == 3
    entries = T.load_records(output_path)
    strict = entries["x1"]["teacher"]
    assert strict["ok"] is True and strict["raw_strict_ok"] is True
    assert strict["calls"][0]["normalizations"] == []
    extra = entries["x2"]["teacher"]
    assert extra["ok"] is True and extra["raw_strict_ok"] is False
    assert extra["calls"][0]["normalizations"] == ["extra_output_fields:1"]
    assert KEY not in output_path.read_text(encoding="utf-8")
    missing = entries["x3"]["teacher"]
    assert missing["ok"] is False and missing["error_code"] == "invalid_reason"
    assert missing["raw_strict_ok"] is False
    assert missing["calls"][0]["normalizations"] == ["missing_output_fields:1"]
    fenced_entry = entries["x4"]["teacher"]
    assert fenced_entry["ok"] is True and fenced_entry["raw_strict_ok"] is False
    assert fenced_entry["calls"][0]["normalizations"] == ["code_fence"]
    broken = entries["x5"]["teacher"]
    assert broken["ok"] is False and broken["error_code"] == "invalid_json"
    assert broken["retryable"] is True and broken["raw_strict_ok"] is False
    no_relation = entries["x6"]["teacher"]
    assert no_relation["ok"] is False and no_relation["error_code"] == "invalid_output_shape"


def test_usage_sanitization_and_adversarial_field_names(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row("x1"), make_row("x2")])
    bad = ollama_reply(json.dumps(five(), ensure_ascii=False))
    bad = T.HttpReply(200, json.dumps({
        "message": {"role": "assistant", "content": json.dumps(five(), ensure_ascii=False)},
        "done": True, "prompt_eval_count": -3, "eval_count": KEY}, ensure_ascii=False))
    good = ollama_reply(json.dumps(five(), ensure_ascii=False), prompt_eval_count=5, eval_count=2)
    transport = FakeTransport([bad, good])
    summary = T.run(input_path, output_path, ollama_teacher(transport), "label", 2, 180.0)
    assert summary["ok"] == 2
    text = output_path.read_text(encoding="utf-8")
    assert KEY not in text
    entries = T.load_records(output_path)
    assert entries["x1"]["teacher"]["calls"][0]["usage"] is None
    assert "usage_fields_dropped:2" in entries["x1"]["teacher"]["calls"][0]["anomalies"]
    assert entries["x2"]["teacher"]["calls"][0]["usage"] == {"prompt_tokens": 5, "completion_tokens": 2}


def test_output_gate_scope_and_unknown_refs(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    inherited = {"id": "r2", "text": "全局偏好：先给依据。", "trigger": "", "tier": "explicit",
                 "permanence": "persistent", "evidence": [], "scope": "inherited_global"}
    request = stored_request(extra_rule=inherited)
    write_input(input_path, [make_row("x1", request=request), make_row("x2", request=request),
                             make_row("x3", request=request)])
    transport = FakeTransport([
        ollama_reply(json.dumps(five(related=("ghost",)), ensure_ascii=False)),
        ollama_reply(json.dumps(five("supersede", related=("r2",)), ensure_ascii=False)),
        ollama_reply(json.dumps(five("supersede", related=("r1",)), ensure_ascii=False)),
    ])
    summary = T.run(input_path, output_path, ollama_teacher(transport), "label", 3, 180.0)
    assert summary["failed_by_code"] == {"unknown_reference": 1, "scope_replacement_forbidden": 1}
    entries = T.load_records(output_path)
    assert entries["x1"]["teacher"]["error_code"] == "unknown_reference"
    assert entries["x2"]["teacher"]["error_code"] == "scope_replacement_forbidden"
    assert entries["x3"]["teacher"]["ok"] is True and entries["x3"]["row"]["label"] == "supersede"


def test_input_gate_rejections_no_call(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    unknown_context = stored_request()
    unknown_context["context"] = {"scope": "project", "expected": ["r1"]}
    with_label = stored_request()
    with_label["context"] = {"scope": "project", "label": "duplicate"}
    with_votes = stored_request()
    with_votes["existing"][0]["votes"] = 3
    wrong_scope = stored_request()
    wrong_scope["candidate"]["scope"] = "global"
    ungrounded = stored_request(quote="从不出现在来源里的引文。")
    ungrounded["evidence"] = [{"id": "e1", "text": QUOTE}]
    rows = [make_row("x1", request=unknown_context), make_row("x2", request=with_label),
            make_row("x3", request=with_votes), make_row("x4", request=wrong_scope),
            make_row("x5", request=ungrounded)]
    write_input(input_path, rows)
    transport = FakeTransport([])
    summary = T.run(input_path, output_path, ollama_teacher(transport), "label", 5, 180.0)
    assert transport.calls == []
    assert summary["failed_by_code"] == {"invalid_context": 3, "invalid_rule": 1, "unverified_evidence": 1}
    entries = T.load_records(output_path)
    for key in entries:
        meta = entries[key]["teacher"]
        assert meta["ok"] is False and meta["retryable"] is False and meta["calls"] == []
        assert "r1" in entries[key]["row"]["related_ids"]                    


def test_ungrounded_input_blocks_paraphrase_too(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    request = stored_request(quote="从不出现在来源里的引文。")
    request["evidence"] = [{"id": "e1", "text": QUOTE}]
    write_input(input_path, [make_row("x1", request=request)])
    transport = FakeTransport([])
    summary = T.run(input_path, output_path, ollama_teacher(transport), "paraphrase", 1, 180.0)
    assert summary["failed_by_code"] == {"unverified_evidence": 1}
    assert transport.calls == []
    entry = T.load_records(output_path)["x1"]
    assert entry["row"]["request"] == request
    assert entry["teacher"]["paraphrase"] is None


def test_paraphrase_carries_only_matching_verdict(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row("x1"), make_row("x2")])
    transport = FakeTransport([
        ollama_reply(json.dumps(paraphrase_reply(), ensure_ascii=False)),
        ollama_reply(json.dumps(five(), ensure_ascii=False)),
        ollama_reply(json.dumps(paraphrase_reply(), ensure_ascii=False)),
        ollama_reply(json.dumps(five("complement", related=("r1",)), ensure_ascii=False)),
    ])
    summary = T.run(input_path, output_path, ollama_teacher(transport), "paraphrase", 2, 180.0)
    assert summary["ok"] == 2 and len(transport.calls) == 4
    entries = T.load_records(output_path)
    carried = entries["x1"]
    assert carried["row"]["label"] == "duplicate" and carried["row"]["label_origin"] == "construction"
    assert carried["row"]["request"]["candidate"]["text"] == "先给结论，再补依据。"
    assert carried["row"]["request"]["candidate"]["permanence"] == "persistent"
    assert carried["row"]["request"]["evidence"][0]["text"] == REWRITTEN_QUOTE
    assert carried["teacher"]["paraphrase"]["label_carried"] is True
    assert carried["teacher"]["paraphrase"]["verdict_matches"] is True
    assert [call["step"] for call in carried["teacher"]["calls"]] == ["paraphrase", "label"]
    verdict_prompt = prompt_of(transport, 1)
    assert REWRITTEN_QUOTE in verdict_prompt and verdict_prompt.startswith(T._PROMPT)

    mismatch = entries["x2"]
    assert mismatch["row"]["label"] is None and mismatch["row"]["label_origin"] is None
    assert mismatch["teacher"]["paraphrase"]["label_carried"] is False
    assert mismatch["teacher"]["paraphrase"]["verdict_matches"] is False
    assert mismatch["teacher"]["paraphrase"]["original_label"] == "duplicate"
    assert mismatch["teacher"]["decision"]["relation"] == "complement"


def test_paraphrase_requires_regate_after_rewrite(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row("x1")])
    broken = paraphrase_reply()
    broken["candidate"]["evidence"] = ["与任何来源都不匹配的引文。"]
    transport = FakeTransport([ollama_reply(json.dumps(broken, ensure_ascii=False))])
    summary = T.run(input_path, output_path, ollama_teacher(transport), "paraphrase", 1, 180.0)
    assert summary["failed_by_code"] == {"unverified_evidence": 1}
    assert len(transport.calls) == 1                                
    entry = T.load_records(output_path)["x1"]
    assert entry["teacher"]["error_code"] == "unverified_evidence"
    assert entry["teacher"]["retryable"] is True
    assert entry["row"]["request"] == stored_request()


def test_paraphrase_shape_drift_rejected(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row("x1")])
    broken = paraphrase_reply()
    broken["existing"] = []
    transport = FakeTransport([ollama_reply(json.dumps(broken, ensure_ascii=False))])
    summary = T.run(input_path, output_path, ollama_teacher(transport), "paraphrase", 1, 180.0)
    assert summary["failed_by_code"] == {"paraphrase_shape": 1}
    entry = T.load_records(output_path)["x1"]
    assert entry["teacher"]["retryable"] is True and entry["row"]["label"] == "duplicate"


def test_paraphrase_resume_reuses_verified_rewrite(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row("x1")])
    transport = FakeTransport([
        ollama_reply(json.dumps(paraphrase_reply(), ensure_ascii=False)),
        ollama_reply("broken json"),
        ollama_reply(json.dumps(five(), ensure_ascii=False)),
    ])
    teacher = ollama_teacher(transport)
    first = T.run(input_path, output_path, teacher, "paraphrase", 1, 180.0)
    assert first["failed"] == 1
    entry = T.load_records(output_path)["x1"]
    assert entry["teacher"]["error_code"] == "invalid_json"
    assert entry["row"]["request"]["candidate"]["text"] == "先给结论，再补依据。"
    assert [call["step"] for call in entry["teacher"]["calls"]] == ["paraphrase", "label"]

    second = T.run(input_path, output_path, teacher, "paraphrase", 1, 180.0)
    assert second["ok"] == 1
    final = T.load_records(output_path)["x1"]
    assert final["row"]["label"] == "duplicate"
    assert [call["step"] for call in final["teacher"]["calls"]] == ["paraphrase", "label", "label"]
    assert final["teacher"]["attempts"] == 3
    assert len(transport.calls) == 3                                             


def test_retry_exhaustion_counts_attempt_events(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row("x1")])
    transport = FakeTransport([ollama_reply("not json")] * 3)
    teacher = ollama_teacher(transport)
    for expected in (1, 2, 3):
        summary = T.run(input_path, output_path, teacher, "label", 1, 180.0)
        assert summary["failed"] == 1
        entry = T.load_records(output_path)["x1"]
        assert entry["teacher"]["attempts"] == expected
        assert entry["teacher"]["error_code"] == "invalid_json"
    assert attempts_count(ledger_lines(output_path), "label") == 3
    last = T.run(input_path, output_path, teacher, "label", 1, 180.0)
    assert last["processed"] == 0 and last["skipped_exhausted"] == 1
    assert len(transport.calls) == 3


def test_crash_between_attempt_and_record_is_counted(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row("x1")])
    crashing = FakeTransport([KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        T.run(input_path, output_path, ollama_teacher(crashing), "label", 1, 180.0)
    lines = ledger_lines(output_path)
    assert attempts_count(lines) == 1 and not [ln for ln in lines if ln.get("kind") == "record"]

    transport = FakeTransport([ollama_reply(json.dumps(five(), ensure_ascii=False))])
    summary = T.run(input_path, output_path, ollama_teacher(transport), "label", 1, 180.0)
    assert summary["ok"] == 1
    entry = T.load_records(output_path)["x1"]
    assert entry["teacher"]["attempts"] == 2                          
    assert attempts_count(ledger_lines(output_path)) == 2


def test_model_and_mode_switches_are_not_skipped(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row("x1")])
    transport_a = FakeTransport([ollama_reply("not json")])
    T.run(input_path, output_path, T.Teacher("ollama", "model-a", T.OLLAMA_BASE_URL,
                                             transport=transport_a), "label", 1, 180.0)
    transport_b = FakeTransport([ollama_reply(json.dumps(five(), ensure_ascii=False))])
    switched = T.run(input_path, output_path, T.Teacher("ollama", "model-b", T.OLLAMA_BASE_URL,
                                                        transport=transport_b), "label", 1, 180.0)
    assert switched["processed"] == 1 and switched["skipped_done"] == 0
    entry = T.load_records(output_path)["x1"]
    assert entry["teacher"]["model"] == "model-b" and entry["teacher"]["ok"] is True

    paraphrase = FakeTransport([
        ollama_reply(json.dumps(paraphrase_reply(), ensure_ascii=False)),
        ollama_reply(json.dumps(five(), ensure_ascii=False)),
    ])
    mode_switched = T.run(input_path, output_path, T.Teacher("ollama", "model-b", T.OLLAMA_BASE_URL,
                                                             transport=paraphrase), "paraphrase", 1, 180.0)
    assert mode_switched["processed"] == 1 and mode_switched["skipped_done"] == 0


def test_changed_input_is_reprocessed(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row("x1")])
    T.run(input_path, output_path, ollama_teacher(
        FakeTransport([ollama_reply(json.dumps(five(), ensure_ascii=False))])), "label", 1, 180.0)
    changed = make_row("x1", request=stored_request(candidate_text="完全不同的候选文本。"))
    write_input(input_path, [changed])
    summary = T.run(input_path, output_path, ollama_teacher(
        FakeTransport([ollama_reply(json.dumps(five(), ensure_ascii=False))])), "label", 1, 180.0)
    assert summary["processed"] == 1 and summary["skipped_done"] == 0
    lines = [ln for ln in ledger_lines(output_path) if ln.get("kind") == "record"]
    assert len(lines) == 2 and lines[-1]["row"]["request"]["candidate"]["text"] == "完全不同的候选文本。"


def test_ledger_torn_tail_fail_closed_and_explicit_repair(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row("x1")])
    first_teacher = ollama_teacher(FakeTransport([ollama_reply(json.dumps(five(), ensure_ascii=False))]))
    T.run(input_path, output_path, first_teacher, "label", 1, 180.0)

    with output_path.open("ab") as handle:
        handle.write(b'{"kind": "record", "id": "x1"')
    with pytest.raises(ValueError) as excinfo:
        T.run(input_path, output_path, first_teacher, "label", 1, 180.0)
    assert "ledger_unreadable_lines:1" in str(excinfo.value)

    write_input(input_path, [make_row("x1"), make_row("x2")])
    repair_teacher = ollama_teacher(
        FakeTransport([ollama_reply(json.dumps(five(), ensure_ascii=False))]))
    repaired = T.run(input_path, output_path, repair_teacher, "label", 1, 180.0,
                     repair_ledger_tail=True)
    assert repaired["processed"] == 1 and repaired["ledger_repair"]["removed_bytes"] > 0
    assert repaired["ledger_repair"]["newline_added"] is False
    assert len(repaired["ledger_repair"]["removed_sha256"]) == 64
    assert T.load_records(output_path)["x1"]["teacher"]["ok"] is True

    with output_path.open("ab") as handle:
        handle.write(json.dumps({"kind": "record", "id": "y", "row": {"id": "y", "label": None},
                                 "teacher": {"ok": True, "fingerprint": "z"}},
                                ensure_ascii=False).encode("utf-8"))
    repaired2 = T.run(input_path, output_path, repair_teacher, "label", 1, 180.0,
                      repair_ledger_tail=True)
    assert repaired2["ledger_repair"]["newline_added"] is True
    assert T.load_records(output_path)["y"]["teacher"]["ok"] is True


def test_ledger_bad_complete_line_is_never_ignored(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row("x1")])
    output_path.write_text("{oops not json}\n", encoding="utf-8")
    teacher = ollama_teacher(FakeTransport([ollama_reply(json.dumps(five(), ensure_ascii=False))]))
    with pytest.raises(ValueError) as excinfo:
        T.run(input_path, output_path, teacher, "label", 1, 180.0)
    assert "ledger_unreadable_lines:1" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo2:
        T.run(input_path, output_path, teacher, "label", 1, 180.0, repair_ledger_tail=True)
    assert "ledger_not_repairable" in str(excinfo2.value)
    assert output_path.read_text(encoding="utf-8") == "{oops not json}\n"


def test_external_manifest_policy(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row("x1", label_origin="human_anchor")])
    transport = FakeTransport([])
    with pytest.raises(ValueError) as excinfo:
        T.run(input_path, output_path, opencode_teacher(transport), "label", 1, 180.0)
    assert "external_manifest_not_allowed" in str(excinfo.value)
    assert transport.calls == [] and not output_path.exists()

    allowed = FakeTransport([openai_reply(json.dumps(five(), ensure_ascii=False))])
    write_input(input_path, [make_row("x1", label_origin="agent_review")])
    summary = T.run(input_path, output_path, opencode_teacher(allowed), "label", 1, 180.0)
    assert summary["ok"] == 1
    call = allowed.calls[0]
    assert call["url"] == "https://opencode.ai/zen/v1/chat/completions"
    assert call["payload"]["max_tokens"] == T.OUTPUT_MAX_TOKENS
    assert "think" not in call["payload"]

    local = FakeTransport([ollama_reply(json.dumps(five(), ensure_ascii=False))])
    write_input(input_path, [make_row("x1", label_origin="human_anchor")])
    output2 = tmp_path / "out2.jsonl"
    assert T.run(input_path, output2, ollama_teacher(local), "label", 1, 180.0)["ok"] == 1


def test_trusted_base_url_allowlist():
    assert T._trusted_base_url("ollama", "http://127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert T._trusted_base_url("ollama", "http://localhost:1234") == "http://localhost:1234"
    assert T._trusted_base_url("ollama", "http://[::1]:9999") == "http://[::1]:9999"
    assert T._trusted_base_url("opencode", "https://opencode.ai/zen/v1") == T.OPENCODE_BASE_URL
    assert T._trusted_base_url("opencode", "https://opencode.ai/zen/v1/") == T.OPENCODE_BASE_URL
    for provider, url in (
        ("opencode", "http://opencode.ai/zen/v1"),
        ("opencode", "https://opencode.ai:8443/zen/v1"),
        ("opencode", "https://opencode.ai/zen/v2"),
        ("opencode", "https://opencode.ai/zen/v1?x=1"),
        ("opencode", "https://opencode.ai/zen/v1#frag"),
        ("opencode", "https://user:pw@opencode.ai/zen/v1"),
        ("opencode", "https://evil.example/zen/v1"),
        ("opencode", "https://opencode.ai:0/zen/v1"),
        ("ollama", "http://10.0.0.1:11434"),
        ("ollama", "http://example.com:11434"),
        ("ollama", "http://127.0.0.1:11434/path"),
        ("ollama", "ftp://127.0.0.1"),
        ("ollama", "http://127.0.0.1:port"),
    ):
        with pytest.raises(ValueError):
            T._trusted_base_url(provider, url)


def test_main_validates_url_before_reading_credentials(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row("x1")])
    monkeypatch.setenv("OPENCODE_API_KEY", KEY)
    read_calls = []
    monkeypatch.setattr(T, "_read_opencode_key", lambda: (read_calls.append(1), KEY)[1])
    code = T.main(["--input", str(input_path), "--output", str(output_path),
                   "--provider", "opencode", "--model", "deepseek-v4-flash",
                   "--base-url", "https://evil.example/zen/v1", "--limit", "1"])
    assert code == 1 and read_calls == []
    assert json.loads(capsys.readouterr().err)["error"] == "unsafe_base_url"
    assert not output_path.exists()


def test_teacher_refuses_unsafe_url_and_redirects_without_request():
    transport = FakeTransport([T.HttpReply(200, "{}")])
    teacher = T.Teacher("opencode", "m", "https://evil.example/zen/v1", key=KEY, transport=transport)
    outcome = teacher.complete("hello", 180.0)
    assert outcome.error_code == "unsafe_base_url" and outcome.retryable is False
    assert transport.calls == []

    redirecting = FakeTransport([T.HttpReply(302, "")])
    redirect_teacher = opencode_teacher(redirecting)
    outcome = redirect_teacher.complete("hello", 180.0)
    assert outcome.error_code == "http_redirect" and outcome.retryable is False


def test_real_transport_bounded_read_and_no_redirect(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def iter_content(self, chunk_size=1):
            yield b"a" * 100
            yield b"b" * (T.RESPONSE_READ_CAP + 10)

        def close(self):
            captured["closed"] = True

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        captured["url"] = url
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    reply = T._requests_transport("https://opencode.ai/zen/v1/chat/completions", {},
                                  {"Authorization": "Bearer x"}, 180.0)
    assert captured["allow_redirects"] is False and captured["stream"] is True
    assert captured["timeout"] == 180.0 and captured["closed"] is True
    assert reply.truncated is True and len(reply.text) == T.RESPONSE_READ_CAP

    oversized = FakeTransport([T.HttpReply(200, "x" * 10, truncated=True)])
    outcome = ollama_teacher(oversized).complete("hello", 180.0)
    assert outcome.error_code == "response_too_large" and outcome.retryable is False


def test_output_token_limits_and_length_rejection(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row("x1"), make_row("x2")])
    thinking = ollama_reply(json.dumps(five(), ensure_ascii=False),
                            message={"role": "assistant", "content": json.dumps(five(), ensure_ascii=False),
                                     "thinking": "内部推理……"})
    transport = FakeTransport([
        ollama_reply(json.dumps(five(), ensure_ascii=False), done_reason="length"),
        thinking,
    ])
    summary = T.run(input_path, output_path, ollama_teacher(transport), "label", 2, 180.0)
    entries = T.load_records(output_path)
    length_entry = entries["x1"]["teacher"]
    assert length_entry["ok"] is False and length_entry["error_code"] == "incomplete_response"
    assert length_entry["retryable"] is True
    thinking_entry = entries["x2"]["teacher"]
    assert thinking_entry["ok"] is True
    assert "thinking_returned" in thinking_entry["calls"][0]["anomalies"]
    assert thinking_entry["calls"][0]["thinking_returned"] is True
    assert summary["ok"] == 1 and summary["failed"] == 1
    assert transport.calls[0]["payload"]["options"]["num_predict"] == T.OUTPUT_MAX_TOKENS


def test_opencode_length_finish_is_incomplete(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row("x1", label_origin="agent_review")])
    transport = FakeTransport([openai_reply(json.dumps(five(), ensure_ascii=False), finish="length")])
    summary = T.run(input_path, output_path, opencode_teacher(transport), "label", 1, 180.0)
    assert summary["failed_by_code"] == {"incomplete_response": 1}
    entry = T.load_records(output_path)["x1"]
    assert entry["teacher"]["retryable"] is True and entry["teacher"]["raw_strict_ok"] is None


def test_input_budget_rejection_no_silent_truncation(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    long_text = "汇报时按照既定顺序给出结论与依据，不要省略任何限定条件。" * 60
    rules = [{"id": f"r{i}", "text": long_text, "trigger": "", "tier": "implicit",
              "permanence": "persistent", "evidence": [], "scope": "same"} for i in range(1, 5)]
    request = stored_request()
    request["existing"] = rules
    write_input(input_path, [make_row("x1", request=request)])
    transport = FakeTransport([])
    summary = T.run(input_path, output_path, ollama_teacher(transport), "label", 1, 180.0)
    assert transport.calls == []
    assert summary["failed_by_code"] == {"input_budget_exceeded": 1}
    entry = T.load_records(output_path)["x1"]
    assert entry["teacher"]["retryable"] is False
    assert any(item.startswith("input_tokens_estimate:") for item in entry["teacher"]["anomalies"])


def test_opencode_key_never_leaks(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row("x1")])
    monkeypatch.setenv("OPENCODE_API_KEY", KEY)
    transport = FakeTransport([
        T.HttpReply(500, f"rejected authorization: Bearer {KEY} key={KEY}"),
    ])
    monkeypatch.setattr(T, "_requests_transport", transport)
    code = T.main(["--input", str(input_path), "--output", str(output_path),
                   "--provider", "opencode", "--model", "minimax-m3", "--mode", "label",
                   "--limit", "1", "--raw-reply", "full"])
    assert code == 0
    captured = capsys.readouterr()
    text = output_path.read_text(encoding="utf-8")
    assert KEY not in text and KEY not in captured.out and KEY not in captured.err
    detail = T.load_records(output_path)["x1"]["teacher"]["calls"][0]["error_detail"]
    assert KEY not in detail and "[REDACTED]" in detail

    extra_field = five()
    extra_field[KEY] = KEY
    reply = T.HttpReply(200, json.dumps({
        "choices": [{"message": {"role": "assistant", "content": json.dumps(extra_field, ensure_ascii=False)},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": KEY, "completion_tokens": -1}}, ensure_ascii=False))
    transport2 = FakeTransport([reply])
    monkeypatch.setattr(T, "_requests_transport", transport2)
    code = T.main(["--input", str(input_path), "--output", str(output_path),
                   "--provider", "opencode", "--model", "minimax-m3", "--mode", "label",
                   "--limit", "1", "--raw-reply", "full"])
    assert code == 0
    captured = capsys.readouterr()
    text = output_path.read_text(encoding="utf-8")
    assert KEY not in text and KEY not in captured.out and KEY not in captured.err
    entry = T.load_records(output_path)["x1"]
    assert "raw_reply" in entry["teacher"]["calls"][-1]                   
    assert "[REDACTED]" in entry["teacher"]["calls"][-1]["raw_reply"]
    assert "extra_output_fields:1" in entry["teacher"]["calls"][-1]["normalizations"]
    assert entry["teacher"]["calls"][-1]["usage"] is None
    assert transport2.calls[0]["headers"]["Authorization"] == f"Bearer {KEY}"


def test_no_key_skips_without_file(tmp_path, monkeypatch, capsys):
    for name in T.OPENCODE_KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row("x1")])
    transport = FakeTransport([])
    monkeypatch.setattr(T, "_requests_transport", transport)
    code = T.main(["--input", str(input_path), "--output", str(output_path),
                   "--provider", "opencode", "--model", "deepseek-v4-flash", "--limit", "1"])
    assert code == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["skipped"] is True and summary["error_code"] == "no_key"
    assert transport.calls == [] and not output_path.exists()


def test_placeholder_key_rejected(monkeypatch):
    for name in T.OPENCODE_KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    assert T._read_opencode_key() is None
    monkeypatch.setenv("OPENCODE_API_KEY", "your_token_here")
    assert T._read_opencode_key() is None
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", KEY)
    assert T._read_opencode_key() == KEY
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-live-first")
    assert T._read_opencode_key() == "sk-live-first"


def test_cli_argument_validation(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row("x1")])
    for argv in (
        ["--input", str(input_path), "--output", str(output_path), "--limit", "1", "--timeout", "200"],
        ["--input", str(input_path), "--output", str(output_path), "--limit", "0"],
        ["--input", str(input_path), "--output", str(output_path), "--limit", "1", "--provider", "opencode"],
        ["--input", str(input_path), "--output", str(output_path), "--limit", "1", "--timeout", "0"],
        ["--input", str(input_path), "--output", str(output_path), "--limit", "1", "--mode", "nope"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            T.main(argv)
        assert excinfo.value.code == 2


def test_main_reports_input_errors(tmp_path, capsys):
    output_path = tmp_path / "out.jsonl"
    missing = tmp_path / "missing.jsonl"
    assert T.main(["--input", str(missing), "--output", str(output_path), "--limit", "1"]) == 1
    assert "error" in json.loads(capsys.readouterr().err)

    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    assert T.main(["--input", str(bad), "--output", str(output_path), "--limit", "1"]) == 1
    assert "invalid_json_line" in json.loads(capsys.readouterr().err)["error"]

    duplicate = tmp_path / "dup.jsonl"
    write_input(duplicate, [make_row("x1"), make_row("x1")])
    assert T.main(["--input", str(duplicate), "--output", str(output_path), "--limit", "1"]) == 1
    assert "duplicate_input_id" in json.loads(capsys.readouterr().err)["error"]


def test_output_rows_validate_against_schema(tmp_path):
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    write_input(input_path, [make_row("x1"), make_row("x2")])
    transport = FakeTransport([
        ollama_reply(json.dumps(five(), ensure_ascii=False)),
        ollama_reply(json.dumps(paraphrase_reply(), ensure_ascii=False)),
        ollama_reply(json.dumps(five(), ensure_ascii=False)),
    ])
    teacher = ollama_teacher(transport)
    T.run(input_path, output_path, teacher, "label", 1, 180.0)
    T.run(input_path, output_path, teacher, "paraphrase", 1, 180.0)
    entries = T.load_records(output_path)
    S.validate_row(entries["x1"]["row"])                             
    for entry in entries.values():
        row = entry["row"]
        if row.get("label") is not None:
            S.validate_row(row)
    carried = [e for e in entries.values() if e["teacher"].get("paraphrase")]
    assert carried, "至少一条改写入账"


def blind_bundle(root, rows=None, *, shard_size=28, seed=17, teacher_id="independent-ds"):
    root.mkdir(parents=True, exist_ok=True)
    source, ledger, blind, mapping = root / "source.jsonl", root / "private" / "ledger.jsonl", \
        root / "blind", root / "private" / "map.json"
    rows = rows or [make_row("duplicate-case-17", label_origin="agent_review", phenomena={
        "root_id": "root-hidden", "domain": "domain-hidden", "family": "family-hidden",
        "author": "author-hidden", "rationale": "rationale-hidden", "variant": "primary-fixture"})]
    write_input(source, rows)
    summary = T.export_blind(source, ledger, blind, mapping, teacher_id=teacher_id,
                             teacher_provider="deepseek", teacher_model="deepseek-v4-flash",
                             shard_size=shard_size, seed=seed, synthetic_only=True)
    private = json.loads(mapping.read_text(encoding="utf-8"))
    return source, ledger, blind, mapping, private, summary


def agent_receipt(item, *, relation="duplicate", code=None, reason="候选与旧要求的条件和长期安排等价。"):
    handles = item["entry"]["reference_handles"]
    return {"blind_id": item["entry"]["blind_id"], "input_sha256": item["entry"]["input_sha256"],
            "teacher": dict(item["teacher"]), "agent_run_id": "stub-agent-run",
            "judgement": five(relation, handles["existing"], handles["evidence"], reason, code)}


def import_receipts(root, bundle, receipts, *, agent_run_id="stub-agent-run", suffix="reply"):
    _source, ledger, blind, mapping, private, _summary = bundle
    reply = root / f"{suffix}.jsonl"
    write_input(reply, receipts)
    shard = blind / private["entries"][0]["shard"]
    return T.import_agent_labels(reply, ledger, shard, mapping, agent_run_id=agent_run_id)


def test_blind_export_exact_student_view_and_no_metadata_leak(tmp_path):
    from research.memory_relations.features import request_view

    request = stored_request(candidate_text="汇报先给结论。 再给依据。")
    row = make_row("duplicate-case-17", request=request, label_origin="agent_review", phenomena={
        "root_id": "root-hidden", "author": "author-hidden", "family": "family-hidden",
        "domain": "domain-hidden", "rationale": "rationale-hidden", "variant": "primary-fixture"})
    bundle = blind_bundle(tmp_path, [row])
    _source, ledger, blind, mapping, private, summary = bundle
    item = private["entries"][0]
    assert item["entry"]["request"] == request_view(row["request"])
    assert item["entry"]["request"]["candidate_text"] == request["candidate"]["text"]
    assert set(item["entry"]) == {"blind_id", "request", "reference_handles", "input_sha256"}
    all_blind = "\n".join(path.read_text(encoding="utf-8") for path in blind.iterdir())
    for marker in (row["id"], row["group_id"], "root-hidden", "author-hidden", "family-hidden",
                   "domain-hidden", "rationale-hidden", '"r1"', '"e1"', '"label":',
                   '"label_origin":', '"split":', '"phenomena":', '"tier":', '"scope":',
                   '"context":', '"permanence":'):
        assert marker not in all_blind
    assert not mapping.is_relative_to(blind)
    assert summary["api_calls"] == 0 and summary["usage"] is None
    events = ledger_lines(ledger)
    assert {event["kind"] for event in events} == {"attempt"}
    assert events[0]["source_row_sha256"] == T._sha256_json(row)
    assert events[0]["versions"]["manual_version"] == "student-text-v3"
    assert events[0]["map_sha256"] == summary["map_sha256"]
    assert item["source_row"] == row


def test_shards_seed_order_and_fresh_random_ids(tmp_path):
    rows = [make_row(f"case{i}") for i in range(5)]
    first = blind_bundle(tmp_path / "first", rows, shard_size=2, seed=9)
    second = blind_bundle(tmp_path / "second", rows, shard_size=2, seed=9)
    assert first[-1]["shards"] == 3
    manifest = json.loads((first[2] / "manifest.json").read_text(encoding="utf-8"))
    assert [shard["rows"] for shard in manifest["shards"]] == [2, 2, 1]
    assert len({shard["teacher"]["id"] for shard in manifest["shards"]}) == 3
    assert [item["source_row"]["id"] for item in first[4]["entries"]] == [
        item["source_row"]["id"] for item in second[4]["entries"]]
    assert not {item["entry"]["blind_id"] for item in first[4]["entries"]}.intersection(
        item["entry"]["blind_id"] for item in second[4]["entries"])
    refs = [ref for item in first[4]["entries"] for handles in item["entry"]["reference_handles"].values()
            for ref in handles]
    assert len(refs) == len(set(refs)) and all(ref.startswith("ref-") for ref in refs)


def test_export_refuses_non_synthetic_self_judging_and_path_overlap(tmp_path):
    source, ledger, blind, mapping = (tmp_path / name for name in ("source.jsonl", "ledger.jsonl", "blind", "map.json"))
    options = {"teacher_id": "judge", "teacher_provider": "deepseek", "teacher_model": "model"}
    write_input(source, [make_row()])
    with pytest.raises(ValueError, match="synthetic_attestation_required"):
        T.export_blind(source, ledger, blind, mapping, **options)
    with pytest.raises(ValueError, match="blind_paths_not_separated"):
        T.export_blind(source, ledger, blind, blind / "private.json", synthetic_only=True, **options)
    write_input(source, [make_row(label_origin="human_anchor")])
    with pytest.raises(ValueError, match="synthetic_manifest_not_allowed"):
        T.export_blind(source, ledger, blind, mapping, synthetic_only=True, **options)
    write_input(source, [make_row(phenomena={"author": "judge"})])
    with pytest.raises(ValueError, match="self_annotation_forbidden"):
        T.export_blind(source, ledger, blind, mapping, synthetic_only=True, **options)
    assert not blind.exists() and not mapping.exists() and not ledger.exists()


def test_export_rejects_original_id_in_semantic_text(tmp_path):
    request = stored_request(candidate_text="请遵守 r1 的汇报顺序。")
    with pytest.raises(ValueError, match="identifier_in_student_text"):
        blind_bundle(tmp_path, [make_row(request=request)])


def test_agent_import_roundtrip_uses_actual_judgement_and_contract(tmp_path):
    bundle = blind_bundle(tmp_path)
    item = bundle[4]["entries"][0]
    receipt = agent_receipt(item, reason="两条都要求汇报先给结论，再给依据。")
    summary = import_receipts(tmp_path, bundle, [receipt])
    assert summary["ok"] == summary["accepted"] == 1 and summary["failed"] == 0
    record = T.load_records(bundle[1])[item["source_row"]["id"]]
    assert set(record["row"]) == set(S.FIELDS)
    S.validate_row(record["row"])
    assert record["row"]["group_id"] == "g1" and record["row"]["related_ids"] == ["r1"]
    assert record["row"]["evidence_refs"] == ["e1"]
    meta = record["teacher"]
    assert meta["decision"]["reason"] == receipt["judgement"]["reason"]
    assert meta["agent_judgement"] == receipt["judgement"]
    assert meta["lineage"]["root_id"] == "root-hidden"
    assert meta["usage"] is None and meta["latency_s"] is None and meta["api_call"] is False
    assert meta["calls"][0]["raw_reply_sha256"] and meta["actual_prompt_sha256"]
    assert attempts_count(ledger_lines(bundle[1])) == 2
    assert import_receipts(tmp_path, bundle, [receipt])["skipped_done"] == 1


def test_disagreement_quarantines_without_uncertain_or_constructed_refs(tmp_path):
    bundle = blind_bundle(tmp_path)
    item = bundle[4]["entries"][0]
    receipt = agent_receipt(item, relation="complement", reason="这是新增要求，不是重复。")
    receipt["judgement"]["related_ids"] = []
    summary = import_receipts(tmp_path, bundle, [receipt])
    record = T.load_records(bundle[1])[item["source_row"]["id"]]
    assert summary["ok"] == summary["disagreements"] == 1 and summary["accepted"] == 0
    assert summary["semantic_uncertain"] == 0
    assert record["row"]["label"] is None and record["row"]["label_origin"] is None
    assert record["row"]["related_ids"] == record["row"]["evidence_refs"] == []
    assert record["teacher"]["decision"]["relation"] == "complement"
    assert record["teacher"]["decision"]["related_ids"] == []
    assert record["teacher"]["decision"]["evidence_refs"] == ["e1"]
    assert record["teacher"]["adjudication_required"] is True
    assert record["teacher"]["source_label"] == "duplicate"


@pytest.mark.parametrize("tamper", ["blind", "map_label", "map_refs", "map_hash", "map_versions"])
def test_blind_hash_tampering_is_fatal_before_ledger_mutation(tmp_path, tamper):
    bundle = blind_bundle(tmp_path)
    item = bundle[4]["entries"][0]
    shard = bundle[2] / item["shard"]
    if tamper == "blind":
        shard.write_text(shard.read_text(encoding="utf-8") + " ", encoding="utf-8")
    else:
        private = bundle[4]
        if tamper == "map_label":
            private["entries"][0]["source_row"]["label"] = "conflict"
        elif tamper == "map_refs":
            private["entries"][0]["reference_map"]["existing"] = {}
        elif tamper == "map_hash":
            private["entries"][0]["entry"]["input_sha256"] = "0" * 64
        else:
            private["versions"]["manual_version"] = "forged-manual"
        bundle[3].write_text(json.dumps(private, ensure_ascii=False), encoding="utf-8")
    before = bundle[1].read_bytes()
    with pytest.raises(ValueError, match="blind_.*mismatch"):
        import_receipts(tmp_path, bundle, [agent_receipt(item)])
    assert bundle[1].read_bytes() == before


@pytest.mark.parametrize("fault,expected", [
    ("hash", "agent_input_hash_mismatch"), ("teacher", "teacher_identity_mismatch"),
    ("model", "teacher_identity_mismatch"), ("role", "teacher_identity_mismatch"),
    ("run", "agent_run_mismatch"), ("missing_field", "invalid_output_shape"),
    ("extra_usage", "invalid_agent_envelope"), ("wrong_role_ref", "unknown_reference"),
    ("original_id", "unknown_reference"), ("duplicate_ref", "duplicate_reference"),
    ("missing_refs", "missing_evidence_refs"), ("missing_reason", "invalid_reason"),
    ("invalid_relation", "invalid_relation"), ("error_type", "invalid_error_code"),
])
def test_agent_import_rejects_forgery_and_never_fills_output(tmp_path, fault, expected):
    bundle = blind_bundle(tmp_path)
    item = bundle[4]["entries"][0]
    receipt = agent_receipt(item)
    judge = receipt["judgement"]
    if fault == "hash":
        receipt["input_sha256"] = "0" * 64
    elif fault in ("teacher", "model", "role"):
        receipt["teacher"][{"teacher": "id", "model": "model", "role": "role"}[fault]] = "forged-author"
    elif fault == "run":
        receipt["agent_run_id"] = "different-run"
    elif fault == "missing_field":
        judge.pop("related_ids")
    elif fault == "extra_usage":
        receipt["usage"] = {"prompt_tokens": 999}
    elif fault == "wrong_role_ref":
        judge["related_ids"] = judge["evidence_refs"]
    elif fault == "original_id":
        judge["related_ids"] = ["r1"]
    elif fault == "duplicate_ref":
        judge["related_ids"] *= 2
    elif fault == "missing_refs":
        judge["evidence_refs"] = []
    elif fault == "missing_reason":
        judge["reason"] = ""
    elif fault == "invalid_relation":
        judge["relation"] = ["duplicate"]
    else:
        judge["relation"], judge["error_code"] = "uncertain", []
    summary = import_receipts(tmp_path, bundle, [receipt])
    assert summary["failed_by_code"] == {expected: 1} and summary["accepted"] == 0
    record = T.load_records(bundle[1])[item["source_row"]["id"]]
    assert record["row"]["label"] is None and record["teacher"]["decision"] is None
    assert record["teacher"]["agent_judgement"] == judge


def test_missing_duplicate_unknown_and_invalid_json_count_independently(tmp_path):
    bundle = blind_bundle(tmp_path, [make_row(f"case{i}") for i in range(4)])
    items = bundle[4]["entries"]
    duplicate = agent_receipt(items[0])
    forged = agent_receipt(items[2])
    forged["teacher"]["role"] = "author"
    unknown = agent_receipt(items[3])
    unknown["blind_id"] = "unknown-blind"
    reply = tmp_path / "reply.jsonl"
    write_input(reply, [duplicate, duplicate, forged, agent_receipt(items[3]), unknown])
    with reply.open("a", encoding="utf-8") as handle:
        handle.write("{broken json}\n")
    summary = T.import_agent_labels(reply, bundle[1], bundle[2] / items[0]["shard"], bundle[3],
                                    agent_run_id="stub-agent-run")
    assert summary["missing_rows"] == 1 and summary["duplicate_rows"] == 1 and summary["unmapped_rows"] == 2
    assert summary["failed"] == 3 and summary["ok"] == 1 and summary["accepted"] == 1
    assert summary["failed_by_code"] == {"unknown_blind_id": 1, "invalid_agent_json": 1,
        "duplicate_agent_judgement": 1, "missing_agent_judgement": 1, "teacher_identity_mismatch": 1}
    records = T.load_records(bundle[1])
    for item in items[:3]:
        assert records[item["source_row"]["id"]]["row"]["label"] is None
    corrected = import_receipts(tmp_path, bundle, [agent_receipt(item) for item in items])
    assert corrected["accepted"] == 3 and corrected["skipped_done"] == 1


def test_semantic_uncertain_is_not_a_missing_judgement(tmp_path):
    bundle = blind_bundle(tmp_path, [make_row(label="uncertain", related=(), refs=())])
    item = bundle[4]["entries"][0]
    receipt = agent_receipt(item, relation="uncertain", code="insufficient_evidence", reason="时间范围未说明。")
    receipt["judgement"]["related_ids"] = receipt["judgement"]["evidence_refs"] = []
    summary = import_receipts(tmp_path, bundle, [receipt])
    assert summary["semantic_uncertain"] == summary["accepted"] == 1 and summary["missing_rows"] == 0
    S.validate_row(T.load_records(bundle[1])[item["source_row"]["id"]]["row"])


def test_production_role_gate_kept_separate_from_student_text(tmp_path):
    request = stored_request()
    request["existing"][0]["scope"] = "inherited_global"
    bundle = blind_bundle(tmp_path, [make_row(request=request, label="contextual")])
    item = bundle[4]["entries"][0]
    summary = import_receipts(tmp_path, bundle, [agent_receipt(item, relation="supersede")])
    assert summary["failed_by_code"] == {"scope_replacement_forbidden": 1}
    assert "scope" not in json.dumps(item["entry"]["request"])
    assert T.load_records(bundle[1])[item["source_row"]["id"]]["teacher"]["agent_judgement"]["relation"] == "supersede"


def test_cli_blind_modes_never_read_credentials_or_call_teachers(tmp_path, monkeypatch, capsys):
    def denied(*_args, **_kwargs):
        raise AssertionError("blind mode must not read keys or call teachers")
    monkeypatch.setattr(T, "_read_opencode_key", denied)
    monkeypatch.setattr(T.Teacher, "complete", denied)
    monkeypatch.setattr(T, "_requests_transport", denied)
    source, ledger, blind, mapping = (tmp_path / name for name in ("source.jsonl", "ledger.jsonl", "blind", "map.json"))
    write_input(source, [make_row()])
    assert T.main(["--input", str(source), "--output", str(ledger), "--mode", "export-blind",
                   "--blind-dir", str(blind), "--blind-map", str(mapping), "--teacher-id", "judge",
                   "--teacher-provider", "deepseek", "--teacher-model", "stub-model",
                   "--shard-size", "1", "--seed", "7", "--synthetic-only"]) == 0
    assert json.loads(capsys.readouterr().out)["shards"] == 1
    item = json.loads(mapping.read_text(encoding="utf-8"))["entries"][0]
    reply = tmp_path / "reply.jsonl"
    write_input(reply, [agent_receipt(item)])
    assert T.main(["--input", str(reply), "--output", str(ledger), "--mode", "import-agent-label",
                   "--blind-input", str(blind / item["shard"]), "--blind-map", str(mapping),
                   "--agent-run-id", "stub-agent-run"]) == 0
    assert json.loads(capsys.readouterr().out)["accepted"] == 1


def test_agent_raw_judgement_redacts_credentials(tmp_path):
    bundle = blind_bundle(tmp_path)
    item = bundle[4]["entries"][0]
    receipt = agent_receipt(item, reason=f"要求等价。api_key={KEY}")
    assert import_receipts(tmp_path, bundle, [receipt])["accepted"] == 1
    assert KEY not in bundle[1].read_text(encoding="utf-8")


def test_two_independent_teachers_keep_disagreement_and_resume_separate(tmp_path):
    bundle = blind_bundle(tmp_path)
    source, ledger, _blind, _mapping, private, _summary = bundle
    first_item = private["entries"][0]
    first = agent_receipt(first_item, relation="complement")
    assert import_receipts(tmp_path, bundle, [first])["disagreements"] == 1
    second_blind, second_map = tmp_path / "second-blind", tmp_path / "private" / "second-map.json"
    T.export_blind(source, ledger, second_blind, second_map, teacher_id="second-judge",
                   teacher_provider="codex", teacher_model="stub-model", synthetic_only=True)
    second_private = json.loads(second_map.read_text(encoding="utf-8"))
    item = second_private["entries"][0]
    second_bundle = (source, ledger, second_blind, second_map, second_private, {})
    summary = import_receipts(tmp_path, second_bundle, [agent_receipt(item)], suffix="second-reply")
    assert summary["disagreements"] == 1 and summary["accepted"] == 0
    latest = T.load_records(ledger)[first_item["source_row"]["id"]]
    assert latest["teacher"]["disagreement_with"] == ["prior_teacher"]
    assert latest["row"]["label"] is None
    before = ledger.read_bytes()
    assert import_receipts(tmp_path, bundle, [first])["skipped_done"] == 1
    assert ledger.read_bytes() == before
    judgments = [event["teacher"]["decision"]["relation"] for event in ledger_lines(ledger)
                 if event["kind"] == "record"]
    assert judgments == ["complement", "duplicate"]


@pytest.mark.parametrize("fault", ["nonempty_refs", "missing_code", "unknown_code"])
def test_invalid_uncertain_is_rejected_not_normalized(tmp_path, fault):
    bundle = blind_bundle(tmp_path)
    item = bundle[4]["entries"][0]
    receipt = agent_receipt(item, relation="uncertain", code="model_uncertain")
    if fault != "nonempty_refs":
        receipt["judgement"]["related_ids"] = receipt["judgement"]["evidence_refs"] = []
        receipt["judgement"]["error_code"] = None if fault == "missing_code" else "llm_failed"
    summary = import_receipts(tmp_path, bundle, [receipt])
    assert summary["failed_by_code"] == {"invalid_uncertain_judgement": 1}
    assert summary["semantic_uncertain"] == 0
    assert T.load_records(bundle[1])[item["source_row"]["id"]]["row"]["label"] is None


def test_empty_existing_true_view_roundtrip(tmp_path):
    request = stored_request()
    request["existing"] = []
    bundle = blind_bundle(tmp_path, [make_row(request=request, label="temporary", related=())])
    item = bundle[4]["entries"][0]
    assert item["entry"]["request"]["rule_texts"] == []
    receipt = agent_receipt(item, relation="temporary", reason="出处限定为本轮例外。")
    summary = import_receipts(tmp_path, bundle, [receipt])
    assert summary["accepted"] == 1
    S.validate_row(T.load_records(bundle[1])[item["source_row"]["id"]]["row"])


def test_malformed_ledger_is_not_repaired_by_blind_mode(tmp_path):
    bundle = blind_bundle(tmp_path)
    item = bundle[4]["entries"][0]
    with bundle[1].open("ab") as handle:
        handle.write(b'{"kind": "attempt"')
    before = bundle[1].read_bytes()
    with pytest.raises(ValueError, match="ledger_unreadable_lines:1"):
        import_receipts(tmp_path, bundle, [agent_receipt(item)])
    assert bundle[1].read_bytes() == before


def test_cli_modes_in_real_subprocess(tmp_path):
    import os
    import subprocess

    source, ledger, blind, mapping = (tmp_path / name for name in ("source.jsonl", "ledger.jsonl", "blind", "map.json"))
    write_input(source, [make_row()])
    command = [sys.executable, "-B", "-m", "research.memory_relations.teachers"]
    env = {"PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
           "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:/Windows")}
    exported = subprocess.run(command + ["--mode", "export-blind", "--input", str(source),
        "--output", str(ledger), "--blind-dir", str(blind), "--blind-map", str(mapping),
        "--teacher-id", "judge", "--teacher-provider", "deepseek", "--teacher-model", "stub-model",
        "--shard-size", "1", "--seed", "13", "--synthetic-only"],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert exported.returncode == 0, exported.stderr
    assert json.loads(exported.stdout)["shards"] == 1
    item = json.loads(mapping.read_text(encoding="utf-8"))["entries"][0]
    reply = tmp_path / "reply.jsonl"
    write_input(reply, [agent_receipt(item)])
    imported = subprocess.run(command + ["--mode", "import-agent-label", "--input", str(reply),
        "--output", str(ledger), "--blind-input", str(blind / item["shard"]),
        "--blind-map", str(mapping), "--agent-run-id", "stub-agent-run"],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert imported.returncode == 0, imported.stderr
    assert json.loads(imported.stdout)["accepted"] == 1
    S.validate_row(T.load_records(ledger)["x1"]["row"])


@pytest.mark.parametrize("fault,expected", [("oversize", "response_too_large"),
    ("duplicate_field", "duplicate_output_field"), ("nonfinite", "invalid_agent_json"),
    ("fence", "invalid_agent_json")])
def test_invalid_receipt_count_is_separate_from_missing(tmp_path, fault, expected):
    bundle = blind_bundle(tmp_path)
    item = bundle[4]["entries"][0]
    text = json.dumps(agent_receipt(item), ensure_ascii=False)
    if fault == "oversize":
        text = "x" * (T._MAX_RESPONSE + 1)
    elif fault == "duplicate_field":
        text = text.replace('"blind_id":', '"blind_id":"forged", "blind_id":', 1)
    elif fault == "nonfinite":
        text = text[:-1] + ', "usage": NaN}'
    else:
        text = "```json " + text + " ```"
    reply = tmp_path / "invalid.jsonl"
    reply.write_text(text + "\n", encoding="utf-8")
    summary = T.import_agent_labels(reply, bundle[1], bundle[2] / item["shard"], bundle[3],
                                    agent_run_id="stub-agent-run")
    assert summary["unmapped_rows"] == summary["missing_rows"] == 1
    assert summary["failed_by_code"] == {expected: 1, "missing_agent_judgement": 1}
    assert summary["accepted"] == summary["semantic_uncertain"] == 0
