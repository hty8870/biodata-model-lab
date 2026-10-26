                      
"""Registered 12-question/S12 paired supplement. Reuses the frozen runtime only."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

D = Path(__file__).resolve().parent
P = D.parent
sys.path.insert(0, str(P))
import pilot_arms_20261025 as arms
import pilot_channels_20261025 as ch
import pilot_real_results_20261025 as main_results
import pilot_score_20261025 as scoring
from disclosure_schema_20261025 import file_sha256, json_write, jsonl_read, jsonl_write, sha256

SEED = "registered-s12-pair-20261025"
QWEN = "batiai/qwen3.8-27b:iq3"
DS = "deepseek-v4-flash"
STAGES = ("qwen_struct", "ds_free", "qwen_blind")
FREE_INSTRUCTION = ("请只依据下面记忆，直接用简短自然语言回答本轮请求。不要输出JSON，"
                    "不要求固定短标签或逐条引用ID。需要用户澄清就直接提问；没有适用记忆规定就说明未规定，"
                    "不要把自行建议冒充记忆规定。未触发的分支不必逐一列出。\n")
BLIND_INSTRUCTION = (
    "你是独立语义评审。只根据给出的任务情境、原始源事实与候选答案判断，不知道答案架构或其他成绩。"
    "评当前任务答案是否正确、是否遗漏影响本题的义务、是否新增无依据行动、是否妥善处理与本题相关的源冲突。"
    "多个并列义务不能漏，有顺序要求须保序；真正同时互斥的要求不能擅自选边，明确时间更新按当前生效规则理解。"
    "没有相关源事实时，应坦诚未规定；不把未声称来源授权的可选建议自动当成来源规则。"
    "不要求JSON动作标签、固定措辞、引用ID或逐条列历史证明；无关槽位的矛盾不应污染本题。"
    "source_conflict_handling.applicable只表示冲突/更新与本题有关，不是库中任意矛盾。"
    "只输出裸JSON并保留全部键："
    '{"correct":true,"omissions":[],"unsupported_actions":[],"source_conflict_handling":'
    '{"applicable":false,"adequate":true,"explanation":"文字依据"},'
    '"answer_evidence":["候选答案中的逐字片段"],"rationale":"对照源事实说明理由"}。'
    "correct/适用/adequate必须为JSON布尔值；omissions、unsupported_actions、answer_evidence为字符串数组。\n"
)


def opaque(prefix, row_id):
    return prefix+"-"+hashlib.sha256((SEED+"|"+row_id).encode()).hexdigest()[:16]


def main_snapshot():
    f = arms.verify_frozen()
    r = main_results.verify_results()
    assert main_results.verify_originals()["ok"]
    return dict(input_manifest_sha256=f["frozen_sha256"], result_manifest_sha256=r["result_manifest_sha256"],
                input_leaves=ch.read_json(P/"out/frozen_manifest.json")["sha256"],
                result_leaves=ch.read_json(P/"out/final_results_manifest_v1.json")["sha256"])


def verify_main():
    before = ch.read_json(D/"main_before.json")
    current = main_snapshot()
    if before != current:
        raise ValueError("main_artifacts_changed")
    return dict(ok=True, input_manifest_sha256=current["input_manifest_sha256"], result_manifest_sha256=current["result_manifest_sha256"],
                input_files=len(current["input_leaves"]), result_files=len(current["result_leaves"]))


def selected():
    tasks = {t["task_id"]: t for t in arms.load()[1]}
    all_rows = jsonl_read(P/"out/disclosures.jsonl")
    rows = [r for r in all_rows if r["scale"] == 12 and r["arm"] in ("P-full", "A-full")]
    if len(rows) != 24 or {r["task_id"] for r in rows} != set(tasks):
        raise ValueError("registered_selection_not_24_all12")
    ordered = []
    for index, tid in enumerate(sorted(tasks, key=lambda t: sha256(SEED+"|"+t))):
        names = ("P-full", "A-full") if index % 2 == 0 else ("A-full", "P-full")
        for arm in names:
            row = next(r for r in rows if r["task_id"] == tid and r["arm"] == arm)
            if sha256(row["prompt"]) != row["prompt_sha256"] or row["placeholder"]:
                raise ValueError("bad_main_prompt_or_artifact")
            ordered.append(row)
    return ordered, tasks


def blind_answer(text):
    value = re.sub(r"(?<![A-Za-z0-9])[mg]-[0-9a-f]{10}(?![0-9a-f])", "[memory-reference]", text or "")
    for arm in arms.ARMS:
        value = value.replace(arm, "[architecture-label]")
    return value


def source_facts():
    groups = jsonl_read(P/"data/groups.jsonl")
    return [dict(text=a["text"], trigger=a["trigger"], title=a["title"]) for g in groups for a in g["atoms"]]


def blind_request(row, task, free_record):
    answer = blind_answer(free_record.get("text") or "")
    payload = dict(source_facts=source_facts(), task=dict(context=task["context"], question=task["question"]), answer=answer)
    prompt = BLIND_INSTRUCTION+json.dumps(payload, ensure_ascii=False)
    if any(x in prompt for x in ("P-full", "A-full", '"expected"', "exact_success", "grounded_success")):
        raise ValueError("blind_prompt_leak")
    return dict(request_id=opaque("qb", row["row_id"]), prompt=prompt,
                input_sha256=sha256(payload), base_row_id=row["row_id"],
                task_id=row["task_id"], arm=row["arm"],
                free_receipt_sha256=sha256(free_record), raw_answer_sha256=sha256(free_record.get("text") or ""),
                blind_answer_sha256=sha256(answer), blind_answer_view=answer)


def build(stage):
    rows, tasks = selected()
    requests = []
    if stage == "qwen_blind":
        free_plan = ch.read_json(D/"plans/ds_free.json")["requests"]
        free_manifest, started, done = ch.audit_ledger(D/"runs/ds_free")
        if len(done) != 24 or set(started) != set(done):
            raise ValueError("free_answers_incomplete_or_pending_unknown_do_not_resend")
        index = {r["base_row_id"]: ch.read_json(D/"runs/ds_free"/(r["request_id"]+".outcome.json")) for r in free_plan}
        for row in sorted(rows, key=lambda r: sha256("blind-order|"+SEED+"|"+r["row_id"])):
            requests.append(blind_request(row, tasks[row["task_id"]], index[row["row_id"]]))
    else:
        for row in rows:
            task = tasks[row["task_id"]]
            prompt = row["prompt"] if stage == "qwen_struct" else FREE_INSTRUCTION+"情境："+task["context"]+"\n请求："+task["question"]+"\n"+row["disclosure"]
            requests.append(dict(request_id=opaque("qs" if stage == "qwen_struct" else "df", row["row_id"]),
                                 prompt=prompt, input_sha256=row["input_sha256"] if stage == "qwen_struct" else sha256(dict(question=task["question"], context=task["context"], disclosure=row["disclosure"])),
                                 base_row_id=row["row_id"], task_id=row["task_id"], arm=row["arm"],
                                 main_prompt_sha256=row["prompt_sha256"], disclosure_sha256=sha256(row["disclosure"])))
    requests = ch.bind_requests(requests, "diagnostics-"+stage)
    if len(requests) != 24 or len({r["request_id"] for r in requests}) != 24:
        raise ValueError("stage_not_exactly_24")
    return requests


def prepare():
    D.mkdir(parents=True, exist_ok=True)
    ch.immutable_json(D/"main_before.json", main_snapshot())
    rows, _ = selected()
    protocol = dict(schema="registered-diagnostics-20261025-v1", seed=SEED, task_n=12, paired_task_n=12, observations_per_group=24,
                    selection="all twelve tasks; S12; P-full/A-full only; no score-dependent selection",
                    collector_sha256=file_sha256(Path(__file__)), main_baseline=dict(calls=256, tokens=294554),
                    models=dict(qwen=QWEN, ds=DS), max_output=1600, max_calls_per_group=24, retry=False,
                    semantic_metric="correct AND no omissions AND no unsupported actions AND conflict adequate if applicable; AI proxy, not human gold",
                    blinding="all 24 original atoms with titles/triggers; no expected/profile/group/arm/scores; memory IDs and literal arm names masked in answer view only",
                    amortization="H=100; initial build before read1; updates after reads20/40/60/80 (4); optionally include maintenance after read100 (5); full rebuild cost projection, not actual updates",
                    row_ids=[r["row_id"] for r in rows])
    ch.immutable_json(D/"protocol.json", protocol)
    for stage in ("qwen_struct", "ds_free"):
        ch.immutable_json(D/"plans"/(stage+".json"), dict(requests=build(stage)))
    return dict(ok=True, paired_tasks=12, selected_rows=24, source_sha256=sha256(ch.source_binding()), main=verify_main())


def collect(stage, enable_llm=False):
    if not enable_llm:
        raise ValueError("explicit_enable_llm_required")
    protocol = ch.read_json(D/"protocol.json")
    if protocol["collector_sha256"] != file_sha256(Path(__file__)):
        raise ValueError("collector_changed_since_protocol_lock")
    verify_main()
    requests = build(stage)
    ch.immutable_json(D/"plans"/(stage+".json"), dict(requests=requests))
    channel = ch.RuntimeChannel("chat" if stage == "ds_free" else "ollama", DS if stage == "ds_free" else QWEN, enable_llm=True, max_output=1600)
    if stage == "ds_free":
        main = ch.read_json(P/"out/consumer/run_manifest.json")["identity"]
        if any(channel.identity.get(k) != main.get(k) for k in ("channel", "model", "provider", "endpoint_sha256", "chat_thinking")):
            raise ValueError("free_reader_different_from_main_ds")
    binding = dict(protocol_sha256=file_sha256(D/"protocol.json"), plan_sha256=sha256(requests),
                   collector_sha256=protocol["collector_sha256"], main_input_sha256=ch.read_json(D/"main_before.json")["input_manifest_sha256"],
                   main_results_sha256=ch.read_json(D/"main_before.json")["result_manifest_sha256"])
    result = ch.execute_requests(requests, channel, D/"runs"/stage, binding=binding, max_calls=24, max_tokens=500_000)
    json_write(D/"runs"/stage/"summary.json", result)
    return result


def selftest():
    rows, tasks = selected()
    checks = []
    assert len(rows) == 24 and len({r["task_id"] for r in rows}) == 12
    checks.append("fixed_all12_pair_selection")
    qs = build("qwen_struct")
    assert all(r["prompt"] == next(x["prompt"] for x in rows if x["row_id"] == r["base_row_id"]) for r in qs)
    checks.append("structured_prompt_byte_identical")
    fs = build("ds_free")
    assert all(FREE_INSTRUCTION in r["prompt"] and arms.INSTRUCTION not in r["prompt"] for r in fs)
    checks.append("free_prompt_removes_output_contract_only")
    br = blind_request(rows[0], tasks[rows[0]["task_id"]], dict(text="P-full m-1234567890记忆未规定。", succeeded=True))
    assert "P-full" not in br["prompt"] and "m-1234567890" not in br["prompt"] and '"expected"' not in br["prompt"]
    checks.append("blind_answer_cue_mask_and_no_gold")
    row = next(r for r in rows if r["task_id"] == "F01-T1" and r["arm"] == "P-full")
    answer = scoring.ideal(row, tasks["F01-T1"])
    assert scoring.score(answer, row, tasks["F01-T1"])["grounded_success"]
    assert not scoring.score(answer | dict(cited_item_ids=[]), row, tasks["F01-T1"])["grounded_success"]
    checks.append("frozen_score_known_pass_fail")
    (D/"tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=D/"tmp", prefix="executor-contract-") as tmp:
        reqs = ch.bind_requests([dict(request_id="local0", prompt="safe0"), dict(request_id="local1", prompt="safe1")], "diagnostics-local-contract")
        fake = ch.FixtureChannel(lambda p, n: "" if n == 1 else "ok")
        first = ch.execute_requests(reqs, fake, Path(tmp), max_calls=2)
        second = ch.execute_requests(reqs, fake, Path(tmp), max_calls=2)
        assert first["completed"] == 2 and first["failed"] == 1 and second["new_calls"] == 0 and fake.calls == 2
    checks.append("executor_fail_denominator_and_no_resend")
    result = dict(ok=True, passed=len(checks), checks=checks, real_model_calls=0)
    json_write(D/"selftest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "selftest", "collect", "verify-main"))
    parser.add_argument("--stage", choices=STAGES, default="qwen_struct")
    parser.add_argument("--enable-llm", action="store_true")
    args = parser.parse_args()
    result = prepare() if args.command == "prepare" else selftest() if args.command == "selftest" else verify_main() if args.command == "verify-main" else collect(args.stage, args.enable_llm)
    print(json.dumps(result, ensure_ascii=False, indent=2))
