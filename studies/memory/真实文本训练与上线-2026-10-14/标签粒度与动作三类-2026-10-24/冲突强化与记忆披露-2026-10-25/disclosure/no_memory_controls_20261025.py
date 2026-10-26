                      
"""Real empty-memory controls for the twelve S12 tasks (prepare/selftest only here).

The control is an explicit empty disclosure, not a missing artifact or fixture.
One control is prepared per task (12 calls), shared by the P-full/A-full pairing.
Real collection is gated by --enable-llm and writes only
pilot_v4/no_memory_control_20261025; prepare/selftest never write there.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
P = HERE / "pilot_v4"
sys.path.insert(0, str(P))
sys.path.insert(0, str(HERE))

import disclosure_schema_20261025 as D              
import pilot_arms_20261025 as arms              
import pilot_channels_20261025 as channels              

TASKS = P / "data/tasks.jsonl"
CONSUMER = P / "out/consumer/run_manifest.json"
REAL_DIR = P / "no_memory_control_20261025"
PREPARE_DIR = HERE / "parent_analysis_20261025"
SELFTEST_DIR = PREPARE_DIR / "_selftest"
STAGE = "no_memory_control"
MAX_OUTPUT = 1600
IDENTITY_FIELDS = ("channel", "model", "provider", "endpoint_sha256", "max_output",
                   "chat_thinking", "adapter_sha256", "backend_attested", "execution", "ollama_format")


def need(path: Path, label: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"refuse: missing {label}: {path}")
    return path


def load_tasks() -> list[dict]:
    need(TASKS, "tasks")
    tasks = D.jsonl_read(TASKS)
    if len(tasks) != 12 or len({t["task_id"] for t in tasks}) != 12:
        raise SystemExit("refuse: tasks.jsonl must contain 12 unique tasks")
    for task in tasks:
        for key in ("task_id", "question", "context", "expected"):
            if key not in task:
                raise SystemExit(f"refuse: task missing {key}")
        if "value_mode" not in task["expected"]:
            raise SystemExit(f"refuse: task {task['task_id']} missing value_mode")
    return tasks


def load_identity() -> dict:
    need(CONSUMER, "consumer run_manifest")
    manifest = D.json_read(CONSUMER)
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise SystemExit("refuse: consumer identity missing")
    if identity.get("channel") != "chat" or identity.get("max_output") != MAX_OUTPUT:
        raise SystemExit("refuse: expected main DS chat identity with max_output=1600")
    for key in IDENTITY_FIELDS:
        if key not in identity:
            raise SystemExit(f"refuse: consumer identity missing {key}")
    return {key: identity[key] for key in IDENTITY_FIELDS}


def empty_disclosure() -> str:
    return D.project_full([], "decide")


def build_requests(tasks: list[dict], identity: dict) -> tuple[list[dict], str, str]:
    empty = empty_disclosure()
    empty_sha = D.sha256(empty)
    tasks_sha = D.file_sha256(TASKS)
    requests = []
    for task in tasks:
        row = {"disclosure": empty, "item_ids": [], "coverage": {}}
        prompt = arms.build_prompt(task, row)
        prompt_sha = D.sha256(prompt)
        source_payload = {
            "tasks_sha256": tasks_sha,
            "task_id": task["task_id"],
            "question": task["question"],
            "context": task["context"],
            "value_mode": task["expected"]["value_mode"],
            "disclosure_sha256": empty_sha,
            "item_ids": [],
            "coverage": {},
        }
        source_sha = D.sha256(source_payload)
        input_sha = D.sha256({"prompt_sha256": prompt_sha, "source_sha256": source_sha,
                              "task_id": task["task_id"], "disclosure_sha256": empty_sha})
        requests.append({
            "request_id": f"no-memory-control__{task['task_id']}__S12",
            "prompt": prompt,
            "prompt_sha256": prompt_sha,
            "input_sha256": input_sha,
            "source_sha256": source_sha,
            "stage": STAGE,
            "blocked": False,
            "placeholder": False,
            "task": {"task_id": task["task_id"], "question": task["question"],
                     "context": task["context"], "value_mode": task["expected"]["value_mode"]},
            "scale": 12,
            "disclosure_sha256": empty_sha,
            "item_ids": [],
            "coverage": {},
            "control_kind": "empty_memory_not_missing_artifact",
        })
    if len(requests) != 12 or len({r["request_id"] for r in requests}) != 12:
        raise SystemExit("refuse: expected one unique control request per task")
    return requests, tasks_sha, empty_sha


def plan_payload(tasks: list[dict], identity: dict, requests: list[dict], tasks_sha: str, empty_sha: str) -> dict:
    script_sha = D.file_sha256(Path(__file__))
    bundle_sha = D.sha256({"tasks_sha256": tasks_sha, "empty_disclosure_sha256": empty_sha,
                           "identity_expected": identity, "requests": requests, "script_sha256": script_sha})
    return {
        "schema": "no-memory-control-prepare/1",
        "status": "prepared_not_collected",
        "stage": STAGE,
        "new_model_calls": 0,
        "planned_calls": 12,
        "one_control_per_task": True,
        "pairing": ["P-full", "A-full"],
        "not_two_arm_independent_calls": True,
        "policy_used_for_selection": False,
        "control_kind": "empty_memory_not_missing_artifact",
        "analysis_note": "compare action/value layer; empty-memory Grounded zero is not a behavioral increment",
        "tasks_sha256": tasks_sha,
        "empty_disclosure_sha256": empty_sha,
        "identity_expected": identity,
        "requests": requests,
        "real_output_dir": str(REAL_DIR.relative_to(HERE)),
        "script_sha256": script_sha,
        "bundle_sha256": bundle_sha,
    }


def publish(path: Path, value: dict) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"refuse: existing output differs: {path}")
    else:
        path.write_text(text, encoding="utf-8", newline="\n")
    return D.file_sha256(path)


def verify_identity(expected: dict, actual: dict) -> dict:
    mismatch = [key for key in IDENTITY_FIELDS if expected.get(key) != actual.get(key)]
    if mismatch:
        raise SystemExit(f"refuse: channel identity mismatch: {mismatch}")
    return {"identity_match": True, "fields": list(IDENTITY_FIELDS)}


def prepare() -> dict:
    tasks, identity = load_tasks(), load_identity()
    requests, tasks_sha, empty_sha = build_requests(tasks, identity)
    plan = plan_payload(tasks, identity, requests, tasks_sha, empty_sha)
    path = PREPARE_DIR / f"no-memory-control-prepare-{plan['bundle_sha256'][:16]}.json"
    plan_sha = publish(path, plan)
    return {"ok": True, "status": plan["status"], "planned_calls": 12, "new_model_calls": 0,
            "output": str(path.relative_to(HERE)), "output_sha256": plan_sha,
            "bundle_sha256": plan["bundle_sha256"], "tasks_sha256": tasks_sha,
            "empty_disclosure_sha256": empty_sha, "real_output_dir": plan["real_output_dir"],
            "policy_used_for_selection": False, "pairing": plan["pairing"]}


def collect(*, enable_llm: bool, max_new: int | None, max_tokens: int) -> dict:
    if not enable_llm:
        raise SystemExit("refuse: real collection requires --enable-llm; prepare/selftest are read-only")
    if max_new is not None and not (1 <= max_new <= 12):
        raise SystemExit("refuse: --max-new must be in 1..12")
    tasks, expected = load_tasks(), load_identity()
    requests, tasks_sha, empty_sha = build_requests(tasks, expected)
    channel = channels.RuntimeChannel("chat", expected["model"], enable_llm=True, max_output=expected["max_output"])
    verify_identity(expected, channel.identity)
    if channel.config is not None and getattr(channel.config, "temperature", None) != 0.0:
        raise SystemExit("refuse: chat temperature is not 0.0")
    plan = plan_payload(tasks, expected, requests, tasks_sha, empty_sha)
    binding = {"tasks_sha256": tasks_sha, "empty_disclosure_sha256": empty_sha,
               "control_plan_sha256": plan["bundle_sha256"], "stage": STAGE}
    result = channels.execute_requests(requests, channel, REAL_DIR, binding=binding,
                                       max_calls=12, max_tokens=max_tokens, max_new=max_new)
    summary = {**result, "schema": "no-memory-control-summary/1", "stage": STAGE,
               "planned_calls": 12, "one_control_per_task": True,
               "pairing": ["P-full", "A-full"], "automatic_retry": False,
               "control_kind": "empty_memory_not_missing_artifact",
               "analysis_note": "compare action/value layer; empty-memory Grounded zero is not a behavioral increment",
               "identity_expected": expected, "identity_returned": channel.identity,
               "real_output_dir": str(REAL_DIR.relative_to(HERE))}
    D.json_write(REAL_DIR / "summary.json", summary)
    return summary


def response(request_id: str) -> dict:
    tasks, identity = load_tasks(), load_identity()
    requests, _, _ = build_requests(tasks, identity)
    request = next((r for r in requests if r["request_id"] == request_id), None)
    if request is None:
        raise SystemExit(f"refuse: unknown request_id: {request_id}")
    record = channels.response(REAL_DIR, request)
    return {"request_id": request_id, "succeeded": record.get("succeeded"),
            "status": record.get("status"), "model": record.get("model"), "provider": record.get("provider"),
            "usage": record.get("usage"), "text_sha256": record.get("text_sha256"), "text": record.get("text")}


def selftest() -> dict:
    tasks, identity = load_tasks(), load_identity()
    requests, tasks_sha, empty_sha = build_requests(tasks, identity)
    empty = empty_disclosure()
    checks = []

    def ck(name: bool, condition: bool):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    ck("twelve_unique_requests", len(requests) == 12 and len({r["request_id"] for r in requests}) == 12)
    ck("one_request_per_task", {r["task"]["task_id"] for r in requests} == {t["task_id"] for t in tasks})
    ck("not_two_arm_24", len(requests) == 12 and all("P-full" not in r["request_id"] and "A-full" not in r["request_id"] for r in requests))
    ck("stage_and_flags", all(r["stage"] == STAGE and r["blocked"] is False and r["placeholder"] is False for r in requests))
    ck("empty_disclosure_exact", all(r["disclosure_sha256"] == empty_sha and r["item_ids"] == [] and r["coverage"] == {} for r in requests))
    ck("prompt_reused_from_arms", all(r["prompt"] == arms.build_prompt(
        next(t for t in tasks if t["task_id"] == r["task"]["task_id"]),
        {"disclosure": empty, "item_ids": [], "coverage": {}}) for r in requests))
    ck("prompt_contains_question_and_context", all(
        r["task"]["question"] in r["prompt"] and r["task"]["context"] in r["prompt"] for r in requests))
    ck("hashes_bind_prompt_source_input", all(
        r["prompt_sha256"] == D.sha256(r["prompt"]) and r["source_sha256"] == D.sha256({
            "tasks_sha256": tasks_sha, "task_id": r["task"]["task_id"],
            "question": r["task"]["question"], "context": r["task"]["context"],
            "value_mode": r["task"]["value_mode"], "disclosure_sha256": empty_sha,
            "item_ids": [], "coverage": {}}) and r["input_sha256"] == D.sha256({
            "prompt_sha256": r["prompt_sha256"], "source_sha256": r["source_sha256"],
            "task_id": r["task"]["task_id"], "disclosure_sha256": empty_sha}) for r in requests))
    ck("main_ds_identity", identity["channel"] == "chat" and identity["model"] == "deepseek-v4-flash"
       and identity["max_output"] == 1600 and identity["provider"] == "openai-compatible"
       and identity["chat_thinking"] is False)
    ck("identity_verify_accepts_expected", verify_identity(identity, identity)["identity_match"] is True)
    mismatched = {**identity, "model": "not-the-main-model"}
    try:
        verify_identity(identity, mismatched)
    except SystemExit:
        ck("identity_verify_rejects_mismatch", True)
    else:
        ck("identity_verify_rejects_mismatch", False)
    ck("real_dir_is_under_pilot", channels.inside(REAL_DIR) == REAL_DIR.resolve())
    ck("prepare_and_selftest_dirs_separate", PREPARE_DIR != REAL_DIR and SELFTEST_DIR != REAL_DIR)
    ck("plan_policy_independent", plan_payload(tasks, identity, requests, tasks_sha, empty_sha)["policy_used_for_selection"] is False)
    result = {"ok": True, "checks": checks, "passed": len(checks), "llm_called": False,
              "script_sha256": D.file_sha256(Path(__file__)), "tasks_sha256": tasks_sha,
              "empty_disclosure_sha256": empty_sha, "planned_calls": 12}
    selftest_path = SELFTEST_DIR / f"no-memory-controls-selftest-{result['script_sha256'][:8]}.json"
    publish(selftest_path, result)
    result["selftest_path"] = str(selftest_path.relative_to(HERE))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "collect", "selftest", "response"), nargs="?", default="prepare")
    parser.add_argument("--enable-llm", action="store_true")
    parser.add_argument("--max-new", type=int)
    parser.add_argument("--max-tokens", type=int, default=2_000_000)
    parser.add_argument("--request-id")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
    elif args.command == "collect":
        result = collect(enable_llm=args.enable_llm, max_new=args.max_new, max_tokens=args.max_tokens)
    elif args.command == "selftest":
        result = selftest()
    else:
        if not args.request_id:
            raise SystemExit("refuse: response requires --request-id")
        result = response(args.request_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
