                      
"""Post-hoc §3.3 controls: metadata-only and no-citation, P-full paired.

Two real controls on the same 12 S12 tasks as the main pilot:
- metadata_only: the actual P-full disclosure JSON view with all free-text
  fields removed; only an explicit machine-metadata whitelist remains.
- no_citation: the byte-identical P-full disclosure and original prompt,
  plus one final instruction forcing cited_item_ids to [] only.

This script only adds files under pilot_v4/missing_controls_20261026/.
It does not modify main disclosures, scores, the empty-memory control,
the frozen policy, or any report. It reuses the existing task/identity
loaders, prompt builder, runtime channel, and layered evaluator.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
P = HERE / "pilot_v4"
sys.path.insert(0, str(P))
sys.path.insert(0, str(HERE))

import disclosure_schema_20261025 as D              
import measurement_correction_20261025 as MC              
import no_memory_controls_20261025 as C              
import pilot_arms_20261025 as arms              
import pilot_channels_20261025 as channels              
import pilot_score_20261025 as scoring              

OUT = P / "missing_controls_20261026"
SELFTEST_DIR = OUT / "selftest"
DISCLOSURES = P / "out/disclosures.jsonl"
SCORED = P / "out/real_results/scored_all_220.jsonl"
EXISTING_NO_MEMORY = P / "no_memory_control_20261025/summary.json"
STAGE = "missing_controls_20261026"
MAX_OUTPUT = 1600
CONTROLS = ("metadata_only", "no_citation")
METADATA_ITEM_FIELDS = ("id", "scope_level", "project_id", "kind", "tier",
                        "permanence", "links", "revision")
REMOVED_FIELDS = ("title", "text", "trigger", "payload")
NO_CITATION_RULE = (
    "\n附加控制（最后执行）：cited_item_ids 必须输出空数组 []，不要提供任何引用；"
    "decision、value、applied_condition、applied_exception 仍严格遵守上面的输出合同与格式，"
    "只移除引用要求。\n"
)
IDENTITY_FIELDS = ("channel", "model", "provider", "endpoint_sha256", "max_output",
                   "chat_thinking", "adapter_sha256", "backend_attested", "execution",
                   "ollama_format")


def need(path: Path, label: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"refuse: missing {label}: {path}")
    return path


def publish(path: Path, value: dict) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"refuse: existing output differs: {path}")
    else:
        path.write_text(text, encoding="utf-8", newline="\n")
    return D.file_sha256(path)


def parse_disclosure(text: str) -> dict:
    if not isinstance(text, str) or not text.startswith(D.SAFETY_PREFIX):
        raise ValueError("disclosure safety prefix missing")
    body = json.loads(text[len(D.SAFETY_PREFIX):])
    if set(body) != {"phase", "items"} or body["phase"] != "decide" or not isinstance(body["items"], list):
        raise ValueError("unexpected disclosure JSON envelope")
    return body


def strings(value):
    if isinstance(value, str):
        if value:
            yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


def free_text_values(body: dict) -> set[str]:
    values: set[str] = set()
    for item in body["items"]:
        for key in ("title", "text", "trigger"):
            value = item.get(key)
            if isinstance(value, str) and value:
                values.add(value)
        values.update(strings(item.get("payload")))
    return values


def metadata_view(disclosure: str) -> tuple[str, list[dict], dict]:
    body = parse_disclosure(disclosure)
    free = free_text_values(body)
    items = []
    for item in body["items"]:
        missing = [key for key in METADATA_ITEM_FIELDS if key not in item]
        if missing:
            raise ValueError(f"metadata field missing: {missing}")
        row = {key: item[key] for key in METADATA_ITEM_FIELDS}
        for key, value in row.items():
            if key in REMOVED_FIELDS or key == "answer":
                raise ValueError(f"forbidden metadata field: {key}")
            if isinstance(value, str) and value in free:
                raise ValueError(f"free text leaked into metadata: {key}")
        items.append(row)
    text = D.SAFETY_PREFIX + json.dumps({"phase": body["phase"], "items": items},
                                        ensure_ascii=False) + "\n"
    for value in free:
        if len(value) >= 4 and value in text:
            raise ValueError("free text leaked into metadata view")
    return text, items, {"free_text_values_checked": len(free)}


def p_full_rows(tasks: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    need(DISCLOSURES, "disclosures")
    rows = D.jsonl_read(DISCLOSURES)
    if len(rows) != 252 or len({r["row_id"] for r in rows}) != 252:
        raise SystemExit("refuse: disclosures must contain 252 unique rows")
    selected = [r for r in rows if r.get("arm") == "P-full" and r.get("scale") == 12]
    task_ids = {t["task_id"] for t in tasks}
    if len(selected) != 12 or {r["task_id"] for r in selected} != task_ids:
        raise SystemExit("refuse: P-full S12 must cover exactly 12 tasks")
    by_task = {r["task_id"]: r for r in selected}
    for task_id, row in by_task.items():
        if row.get("placeholder") or not row.get("real") or row.get("blocked"):
            raise SystemExit(f"refuse: P-full S12 row is not real: {task_id}")
        if D.sha256(row["prompt"]) != row.get("prompt_sha256"):
            raise SystemExit(f"refuse: stored prompt hash mismatch: {task_id}")
        body = parse_disclosure(row["disclosure"])
        if not body["items"]:
            raise SystemExit(f"refuse: empty P-full disclosure: {task_id}")
        if len(body["items"]) != len(row.get("item_ids") or []):
            raise SystemExit(f"refuse: disclosure/item_ids length mismatch: {task_id}")
    return selected, by_task


def selected_rows_digest(rows: list[dict]) -> str:
    return D.sha256([
        {"row_id": r["row_id"], "task_id": r["task_id"],
         "prompt_sha256": r["prompt_sha256"],
         "disclosure_sha256": D.sha256(r["disclosure"])}
        for r in sorted(rows, key=lambda r: r["task_id"])
    ])


def build_plan() -> tuple[list[dict], dict]:
    tasks = C.load_tasks()
    identity = C.load_identity()
    rows, by_task = p_full_rows(tasks)
    tasks_sha = D.file_sha256(C.TASKS)
    disclosures_sha = D.file_sha256(DISCLOSURES)
    rows_sha = selected_rows_digest(rows)
    requests = []
    metadata_audit = {}
    for task in tasks:
        task_id = task["task_id"]
        p_row = by_task[task_id]
        original_prompt = arms.build_prompt(task, p_row)
        if original_prompt != p_row["prompt"]:
            raise SystemExit(f"refuse: prompt reconstruction differs: {task_id}")
        original_disclosure_sha = D.sha256(p_row["disclosure"])
        metadata_disclosure, metadata_items, audit = metadata_view(p_row["disclosure"])
        metadata_audit[task_id] = {"item_count": len(metadata_items), **audit}
        metadata_prompt = arms.build_prompt(task, {**p_row, "disclosure": metadata_disclosure})
        no_citation_prompt = original_prompt + NO_CITATION_RULE
        for control, prompt, disclosure, disclosure_sha, extra in (
            ("metadata_only", metadata_prompt, metadata_disclosure, D.sha256(metadata_disclosure),
             {"metadata_item_fields": list(METADATA_ITEM_FIELDS),
              "metadata_removed_fields": list(REMOVED_FIELDS),
              "metadata_item_count": len(metadata_items),
              "metadata_audit": audit}),
            ("no_citation", no_citation_prompt, p_row["disclosure"], original_disclosure_sha,
             {"original_prompt": original_prompt,
              "original_prompt_sha256": D.sha256(original_prompt),
              "citation_rule": NO_CITATION_RULE,
              "citation_rule_sha256": D.sha256(NO_CITATION_RULE),
              "prompt_before_sha256": D.sha256(original_prompt),
              "prompt_after_sha256": D.sha256(no_citation_prompt),
              "not_identical_system_condition": True}),
        ):
            source_payload = {
                "control": control,
                "task_id": task_id,
                "row_id": p_row["row_id"],
                "tasks_sha256": tasks_sha,
                "p_full_disclosures_sha256": disclosures_sha,
                "p_full_rows_sha256": rows_sha,
                "original_disclosure_sha256": original_disclosure_sha,
                "control_disclosure_sha256": disclosure_sha,
                "question": task["question"],
                "context": task["context"],
                "value_mode": task["expected"]["value_mode"],
                **extra,
            }
            source_sha = D.sha256(source_payload)
            prompt_sha = D.sha256(prompt)
            input_sha = D.sha256({
                "control": control, "task_id": task_id, "prompt_sha256": prompt_sha,
                "source_sha256": source_sha, "disclosure_sha256": disclosure_sha,
            })
            rid = f"{control.replace('_', '-')}__{task_id}__S12"
            requests.append({
                "request_id": rid,
                "prompt": prompt,
                "prompt_sha256": prompt_sha,
                "input_sha256": input_sha,
                "source_sha256": source_sha,
                "source_payload": source_payload,
                "stage": STAGE,
                "task": {"task_id": task_id, "question": task["question"],
                         "context": task["context"], "value_mode": task["expected"]["value_mode"]},
                "scale": 12,
                "control_kind": control,
                "p_full_row_id": p_row["row_id"],
                "original_prompt": original_prompt,
                "original_prompt_sha256": D.sha256(original_prompt),
                "original_disclosure_sha256": original_disclosure_sha,
                "disclosure_sha256": disclosure_sha,
                "control_disclosure": disclosure,
                "item_ids": list(p_row["item_ids"]),
                "coverage": p_row["coverage"] if control == "no_citation" else {},
                **extra,
            })
    if len(requests) != 24 or len({r["request_id"] for r in requests}) != 24:
        raise SystemExit("refuse: expected exactly 24 unique control requests")
    if Counter(r["control_kind"] for r in requests) != {"metadata_only": 12, "no_citation": 12}:
        raise SystemExit("refuse: expected 12 requests per control")
    if {r["task"]["task_id"] for r in requests} != {t["task_id"] for t in tasks}:
        raise SystemExit("refuse: request task coverage mismatch")
    policy = D.json_read(MC.POLICY)
    ambiguous = sorted(t["task_id"] for t in policy["tasks"] if t.get("status") == "ambiguous")
    if ambiguous != ["F03-T2"]:
        raise SystemExit(f"refuse: unexpected ambiguous tasks: {ambiguous}")
    plan = {
        "schema": "missing-controls-prepare/1",
        "status": "prepared_not_collected",
        "stage": STAGE,
        "new_model_calls": 0,
        "planned_calls": 24,
        "controls": {"metadata_only": 12, "no_citation": 12},
        "pairing": {"baseline": "P-full", "tasks": 12, "evaluable_after_ambiguity": 11,
                    "not_A_full": True, "not_independent_new_tasks": True,
                    "excluded_ambiguous_tasks": ambiguous},
        "tasks_sha256": tasks_sha,
        "p_full_disclosures": str(DISCLOSURES.relative_to(HERE)),
        "p_full_disclosures_sha256": disclosures_sha,
        "p_full_rows_sha256": rows_sha,
        "metadata_whitelist": {"envelope_fields": ["phase"],
                               "item_fields": list(METADATA_ITEM_FIELDS),
                               "removed_fields": list(REMOVED_FIELDS),
                               "note": "IDs, scope/kind/tier/permanence, source links and revision only; no title/text/trigger/payload"},
        "no_citation_rule": {"text": NO_CITATION_RULE, "sha256": D.sha256(NO_CITATION_RULE),
                             "note": "appended after the original prompt; only cited_item_ids is overridden"},
        "identity_expected": identity,
        "requests": requests,
        "script_sha256": D.file_sha256(Path(__file__)),
        "real_output_dir": str(OUT.relative_to(HERE)),
        "automatic_retry": False,
        "selection_uses_policy": False,
    }
    plan["bundle_sha256"] = D.sha256({key: plan[key] for key in (
        "tasks_sha256", "p_full_disclosures_sha256", "p_full_rows_sha256",
        "identity_expected", "requests", "script_sha256")})
    return requests, plan


def require_identity(identity: dict) -> dict:
    if identity.get("channel") != "chat" or identity.get("model") != "deepseek-v4-flash":
        raise SystemExit("refuse: unexpected main consumer identity")
    if identity.get("max_output") != MAX_OUTPUT or identity.get("chat_thinking") is not False:
        raise SystemExit("refuse: expected max_output=1600 and thinking=false")
    return C.verify_identity(identity, identity)


def require_real_response(record: dict, identity: dict) -> dict:
    actual = record.get("identity") or {}
    mismatch = [key for key in IDENTITY_FIELDS if actual.get(key) != identity.get(key)]
    if mismatch:
        raise ValueError(f"identity mismatch: {mismatch}")
    if actual.get("execution") != "authorized_model_call":
        raise ValueError("not an authorized model call")
    if not record.get("attempted") or not record.get("succeeded"):
        raise ValueError("response was not a successful attempted call")
    if record.get("model") != identity.get("model") or record.get("provider") != identity.get("provider"):
        raise ValueError("returned model/provider mismatch")
    return record


def prepare() -> dict:
    requests, plan = build_plan()
    plan_sha = publish(OUT / "plan.json", plan)
    return {"ok": True, "status": plan["status"], "planned_calls": 24,
            "controls": plan["controls"], "plan": str((OUT / "plan.json").relative_to(HERE)),
            "plan_sha256": plan_sha, "bundle_sha256": plan["bundle_sha256"],
            "pairing": plan["pairing"], "new_model_calls": 0}


def collect(*, enable_llm: bool, max_new: int | None, max_tokens: int) -> dict:
    if not enable_llm:
        raise SystemExit("refuse: real collection requires --enable-llm")
    if max_new is not None and not (1 <= max_new <= 24):
        raise SystemExit("refuse: --max-new must be in 1..24")
    requests, plan = build_plan()
    plan_sha = publish(OUT / "plan.json", plan)
    identity = plan["identity_expected"]
    channel = channels.RuntimeChannel("chat", identity["model"], enable_llm=True,
                                      max_output=identity["max_output"])
    C.verify_identity(identity, channel.identity)
    require_identity(channel.identity)
    if channel.config is None or getattr(channel.config, "temperature", None) != 0.0:
        raise SystemExit("refuse: chat temperature is not 0.0")
    if channels.inside(OUT) != OUT.resolve():
        raise SystemExit("refuse: output is outside pilot_v4")
    binding = {
        "stage": STAGE,
        "plan_sha256": plan_sha,
        "bundle_sha256": plan["bundle_sha256"],
        "tasks_sha256": plan["tasks_sha256"],
        "p_full_disclosures_sha256": plan["p_full_disclosures_sha256"],
        "planned_calls": 24,
        "controls": plan["controls"],
        "pairing": plan["pairing"],
        "automatic_retry": False,
    }
    result = channels.execute_requests(requests, channel, OUT, binding=binding,
                                       max_calls=24, max_tokens=max_tokens,
                                       max_new=24 if max_new is None else max_new)
    collection = {
        "schema": "missing-controls-collection/1",
        "status": "collected" if result["completed"] == 24 else "partial",
        "plan_sha256": plan_sha,
        "bundle_sha256": plan["bundle_sha256"],
        "identity_expected": identity,
        "identity_returned": channel.identity,
        "execute": result,
        "automatic_retry": False,
        "note": "24 fixed requests; no automatic resend; failed/unknown stays in the ledger.",
    }
    D.json_write(OUT / "collection.json", collection)
    return collection


def _eval_row(answer: dict, disclosure_row: dict, policy_task: dict, public_facts: dict) -> dict:
    if policy_task.get("status") == "ambiguous":
        return {"status": "unknown", "reason": "reference_task_ambiguous",
                "schema_ok": None, "decision_ok": None, "value_ok": None,
                "action_alias_value_match": None, "minimal_evidence_sufficient": None,
                "layered_success": None, "invalid_citations": [], "cited": []}
    return MC.evaluate(answer, disclosure_row, policy_task, public_facts)


def _counts(items: list[dict]) -> dict:
    scored = [x for x in items if x.get("evaluation_status") == "scored"]
    by_decision = {}
    for decision in sorted({x["policy_decision"] for x in scored}):
        subset = [x for x in scored if x["policy_decision"] == decision]
        by_decision[decision] = _simple_counts(subset)
    return {"n_calls": len(items), "n_evaluable": len(scored),
            "excluded_ambiguous": sum(x.get("evaluation_status") == "unknown" for x in items),
            **_simple_counts(scored), "by_policy_decision": by_decision}


def _simple_counts(items: list[dict]) -> dict:
    return {
        "schema_ok": sum(x.get("schema_ok") is True for x in items),
        "decision_ok": sum(x.get("decision_ok") is True for x in items),
        "value_ok": sum(x.get("value_ok") is True for x in items),
        "action_value_match": sum(x.get("action_value_match") is True for x in items),
        "citation_empty": sum(x.get("citation_empty") is True for x in items),
        "invalid_citation_items": sum(bool(x.get("invalid_citations")) for x in items),
        "minimal_evidence_sufficient": sum(x.get("minimal_evidence_sufficient") is True for x in items),
        "layered_success": sum(x.get("layered_success") is True for x in items),
    }


def _usage(rows: list[dict]) -> dict:
    usages = [x["usage"] for x in rows if isinstance(x.get("usage"), dict)]
    known = [u for u in usages if u.get("total_tokens") is not None]
    latencies = [x["latency_s"] for x in rows if isinstance(x.get("latency_s"), (int, float))]
    return {
        "calls": len(rows),
        "usage_records": len(usages),
        "usage_unknown": len(usages) - len(known),
        "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in known),
        "completion_tokens": sum(u.get("completion_tokens") or 0 for u in known),
        "total_tokens_all_sources": sum(u.get("total_tokens") or 0 for u in known),
        "provider_total_tokens": sum((u.get("total_tokens") or 0) for u in known
                                     if u.get("source") == "provider"),
        "latency_s_sum": sum(latencies),
        "latency_s_mean": (sum(latencies) / len(latencies)) if latencies else None,
        "latency_s_max": max(latencies) if latencies else None,
    }


def analyze() -> dict:
    need(OUT / "plan.json", "control plan")
    plan = D.json_read(OUT / "plan.json")
    ctx = MC.load_context()
    identity = plan["identity_expected"]
    rows = []
    failures = []
    p_full_cache = {}
    for request in plan["requests"]:
        control = request["control_kind"]
        task_id = request["task"]["task_id"]
        p_row = ctx["disc_by_id"].get(request["p_full_row_id"])
        if p_row is None:
            failures.append({"request_id": request["request_id"], "control": control,
                             "task_id": task_id, "error": "p_full_row_missing"})
            continue
        try:
            record = channels.response(OUT, request)
            require_real_response(record, identity)
        except Exception as exc:                                         
            failures.append({"request_id": request["request_id"], "control": control,
                             "task_id": task_id, "error": f"{type(exc).__name__}:{str(exc)[:160]}"})
            continue
        answer, parse_reason = scoring.parse(record.get("text") or "")
        policy_task = ctx["policy_by_task"][task_id]
        if control == "metadata_only":
            control_row = {"disclosure": request["control_disclosure"],
                           "item_ids": request["item_ids"], "coverage": {}}
        else:
            control_row = p_row
        evaluated = _eval_row(answer, control_row, policy_task, ctx["public_facts"])
        if task_id not in p_full_cache:
            p_full_answer = ctx["scored_by_id"][request["p_full_row_id"]]["answer"]
            p_full_eval = _eval_row(p_full_answer, p_row, policy_task, ctx["public_facts"])
            p_cited = p_full_eval.get("cited") or []
            p_full_cache[task_id] = {
                "task_id": task_id,
                "policy_status": policy_task.get("status"),
                "policy_decision": policy_task.get("decision"),
                "evaluation_status": p_full_eval.get("status"),
                "schema_ok": p_full_eval.get("schema_ok"),
                "decision_ok": p_full_eval.get("decision_ok"),
                "value_ok": p_full_eval.get("value_ok"),
                "action_value_match": p_full_eval.get("action_alias_value_match"),
                "citation_empty": (len(p_cited) == 0) if p_full_eval.get("schema_ok") else None,
                "invalid_citations": p_full_eval.get("invalid_citations") or [],
                "minimal_evidence_sufficient": p_full_eval.get("minimal_evidence_sufficient"),
                "layered_success": p_full_eval.get("layered_success"),
            }
        p_full_info = p_full_cache[task_id]
        parsed_cited = answer.get("cited_item_ids") if parse_reason == "ok" else None
        analysis = {
            "request_id": request["request_id"],
            "control": control,
            "task_id": task_id,
            "p_full_row_id": request["p_full_row_id"],
            "policy_status": policy_task.get("status"),
            "policy_decision": policy_task.get("decision"),
            "evaluation_status": evaluated.get("status"),
            "excluded_ambiguous": evaluated.get("status") == "unknown",
            "parse_reason": parse_reason,
            "schema_ok": evaluated.get("schema_ok"),
            "decision_ok": evaluated.get("decision_ok"),
            "value_ok": evaluated.get("value_ok"),
            "action_value_match": evaluated.get("action_alias_value_match"),
            "citation_empty": (len(parsed_cited) == 0) if parsed_cited is not None else None,
            "cited_count": len(parsed_cited) if parsed_cited is not None else None,
            "invalid_citations": evaluated.get("invalid_citations") or [],
            "minimal_evidence_sufficient": evaluated.get("minimal_evidence_sufficient"),
            "layered_success": evaluated.get("layered_success"),
            "p_full_schema_ok": p_full_info["schema_ok"],
            "p_full_decision_ok": p_full_info["decision_ok"],
            "p_full_value_ok": p_full_info["value_ok"],
            "p_full_action_value_match": p_full_info["action_value_match"],
            "p_full_citation_empty": p_full_info["citation_empty"],
            "p_full_minimal_evidence_sufficient": p_full_info["minimal_evidence_sufficient"],
            "p_full_layered_success": p_full_info["layered_success"],
            "usage": record.get("usage"),
            "latency_s": record.get("latency_s"),
            "finish_reason": record.get("finish_reason"),
            "status": record.get("status"),
        }
        rows.append(analysis)
    controls = {}
    for control in CONTROLS:
        subset = [x for x in rows if x["control"] == control]
        controls[control] = _counts(subset)
        controls[control]["usage"] = _usage(subset)
        controls[control]["citation_empty_by_task"] = {
            x["task_id"]: x["citation_empty"] for x in subset if x["citation_empty"] is not None}
        controls[control]["citation_observed_calls"] = sum(
            x["citation_empty"] is not None for x in subset)
        controls[control]["citation_empty_all_calls"] = sum(
            x["citation_empty"] is True for x in subset)
        controls[control]["violating_citation_tasks"] = [
            x["task_id"] for x in subset if x["citation_empty"] is False]
    p_full = _counts(list(p_full_cache.values()))
    paired = {}
    for control in CONTROLS:
        subset = [x for x in rows if x["control"] == control
                  and x.get("evaluation_status") == "scored"]
        transitions = Counter()
        for x in subset:
            p, c = x.get("p_full_action_value_match"), x.get("action_value_match")
            key = ("both" if p and c else "p_full_only" if p else
                   "control_only" if c else "neither")
            transitions[key] += 1
        paired[control] = {"n_pairs": len(subset), **dict(transitions)}
    ledger = None
    pending = []
    if (OUT / "run_manifest.json").is_file():
        _, reserved, finished = channels.audit_ledger(OUT)
        pending = sorted(set(reserved) - set(finished))
        ledger = {"reserved": len(reserved), "finished": len(finished),
                  "pending_unknown": pending}
    collection = D.json_read(OUT / "collection.json") if (OUT / "collection.json").is_file() else None
    existing_no_memory = D.json_read(need(EXISTING_NO_MEMORY, "existing no-memory summary"))
    summary = {
        "schema": "missing-controls-summary/1",
        "status": "complete" if len(rows) == 24 and not failures and not pending else "partial",
        "stage": STAGE,
        "plan_sha256": D.file_sha256(OUT / "plan.json"),
        "bundle_sha256": plan["bundle_sha256"],
        "plan_script_sha256": plan["script_sha256"],
        "analysis_script_sha256": D.file_sha256(Path(__file__)),
        "provenance_note": "plan.json was frozen before collection; later edits only changed analysis/reporting, not requests or calls.",
        "source": {
            "disclosures": plan["p_full_disclosures"],
            "disclosures_sha256": plan["p_full_disclosures_sha256"],
            "p_full_rows_sha256": plan["p_full_rows_sha256"],
            "tasks_sha256": plan["tasks_sha256"],
            "selection": "P-full S12 rows only; 12 tasks; no new tasks; not A-full",
        },
        "design": {
            "metadata_only": plan["metadata_whitelist"],
            "no_citation": plan["no_citation_rule"],
            "pairing": plan["pairing"],
            "metadata_count": plan["controls"]["metadata_only"],
            "no_citation_count": plan["controls"]["no_citation"],
            "existing_no_memory_control": {
                "path": str(EXISTING_NO_MEMORY.relative_to(HERE)),
                "sha256": D.file_sha256(EXISTING_NO_MEMORY),
                "planned_calls": existing_no_memory.get("planned_calls"),
                "completed": existing_no_memory.get("completed"),
                "succeeded": existing_no_memory.get("succeeded"),
                "pairing": existing_no_memory.get("pairing"),
                "note": "pre-existing real control; read only, not rerun or modified",
            },
            "post_hoc_supplement": True,
            "preregistered_new_blind": False,
        },
        "collection": collection,
        "ledger": ledger,
        "usage": {"by_control": {control: controls[control]["usage"] for control in CONTROLS},
                  "all": _usage(rows)},
        "failures": failures,
        "analysis": {
            "n_calls": len(rows),
            "n_evaluable": sum(x.get("evaluation_status") == "scored" for x in rows),
            "n_unique_evaluable_tasks": len({x["task_id"] for x in rows
                                             if x.get("evaluation_status") == "scored"}),
            "excluded_ambiguous": [{"task_id": "F03-T2", "reason": "policy status ambiguous"}],
            "controls": controls,
            "p_full": p_full,
            "paired_action_value": paired,
            "layers_note": ("schema/action/value are reported separately. For non-no_memory tasks, "
                            "empty citations make the evidence/grounded layer zero by construction; "
                            "for N1/N2, empty citations are contract-correct. Do not read "
                            "layered_success as a semantic-effect estimate."),
        },
        "limitations": [
            "Post-hoc control supplement to the 12 S12 tasks; not a preregistered new blind sample.",
            "Only P-full is paired; these are not 24 P/A pairs and not new independent tasks.",
            "metadata-only removes title/text/trigger/payload; item IDs, count, kind/scope/tier/"
            "permanence/source IDs/revision can still leak structure, so content isolation is not claimed.",
            "no-citation uses the original disclosure but appends one final instruction; the system "
            "condition is therefore not byte-identical to P-full even though the disclosure is unchanged.",
            "F03-T2 is excluded from evaluation because the frozen policy marks it ambiguous; the call "
            "is retained but not scored.",
            "Failures and unknown outcomes are retained; no automatic retry or replacement sampling.",
        ],
    }
    D.jsonl_write(OUT / "analysis.jsonl", rows)
    D.json_write(OUT / "summary.json", summary)
    return summary


def selftest() -> dict:
    requests, plan = build_plan()
    tasks = C.load_tasks()
    identity = C.load_identity()
    checks = []

    def ck(name: str, condition: bool):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    ck("exactly_24_requests", len(requests) == 24 and len({r["request_id"] for r in requests}) == 24)
    ck("twelve_per_control", Counter(r["control_kind"] for r in requests)
       == {"metadata_only": 12, "no_citation": 12})
    ck("one_per_task_per_control", {
        (r["control_kind"], r["task"]["task_id"]) for r in requests
    } == {(c, t["task_id"]) for c in CONTROLS for t in tasks})
    ck("not_A_full_or_PA_pairs", all("A-full" not in r["request_id"] and
                                     r["control_kind"] in CONTROLS for r in requests))
    metadata_requests = [r for r in requests if r["control_kind"] == "metadata_only"]
    no_citation_requests = [r for r in requests if r["control_kind"] == "no_citation"]
    metadata_items = [item for request in metadata_requests
                      for item in parse_disclosure(request["control_disclosure"])["items"]]
    ck("metadata_no_forbidden_fields", all(
        not (set(item) & {"title", "text", "trigger", "payload", "answer"})
        for item in metadata_items))
    ck("metadata_whitelist_exact", all(
        set(item) == set(METADATA_ITEM_FIELDS) for item in metadata_items))
    _, by_task = p_full_rows(tasks)
    ck("metadata_no_original_free_text", all(
        not any(len(value) >= 4 and value in request["control_disclosure"]
                for value in free_text_values(parse_disclosure(by_task[request["task"]["task_id"]]["disclosure"])))
        for request in metadata_requests))
    ck("no_citation_original_disclosure_unchanged", all(
        request["disclosure_sha256"] == request["original_disclosure_sha256"] ==
        D.sha256(request["control_disclosure"]) for request in no_citation_requests))
    ck("no_citation_original_prompt_preserved", all(
        request["original_prompt_sha256"] == D.sha256(request["original_prompt"]) and
        request["prompt"] == request["original_prompt"] + NO_CITATION_RULE
        for request in no_citation_requests))
    ck("prompt_hashes_recompute", all(
        r["prompt_sha256"] == D.sha256(r["prompt"]) and
        r["input_sha256"] == D.sha256({"control": r["control_kind"], "task_id": r["task"]["task_id"],
                                       "prompt_sha256": r["prompt_sha256"],
                                       "source_sha256": r["source_sha256"],
                                       "disclosure_sha256": r["disclosure_sha256"]})
        for r in requests))
    ck("prompt_contains_question_and_context", all(
        r["task"]["question"] in r["prompt"] and r["task"]["context"] in r["prompt"]
        for r in requests))
    ck("main_ds_identity", identity["channel"] == "chat" and identity["model"] == "deepseek-v4-flash"
       and identity["max_output"] == 1600 and identity["chat_thinking"] is False)
    ck("identity_expected_verifies", C.verify_identity(identity, identity)["identity_match"] is True)
    fake = {"identity": {**identity, "execution": "fixture"}, "attempted": True,
            "succeeded": True, "model": identity["model"], "provider": identity["provider"]}
    try:
        require_real_response(fake, identity)
    except ValueError:
        ck("fake_response_rejected", True)
    else:
        ck("fake_response_rejected", False)
    policy = D.json_read(MC.POLICY)
    ambiguous = sorted(t["task_id"] for t in policy["tasks"] if t.get("status") == "ambiguous")
    ck("f03_t2_excluded_as_ambiguous", ambiguous == ["F03-T2"])
    ck("planned_calls_24", plan["planned_calls"] == 24 and plan["new_model_calls"] == 0)
    ck("output_inside_pilot", channels.inside(OUT) == OUT.resolve())
    result = {
        "ok": True,
        "schema": "missing-controls-selftest/1",
        "checks": checks,
        "passed": len(checks),
        "llm_called": False,
        "script_sha256": D.file_sha256(Path(__file__)),
        "tasks_sha256": plan["tasks_sha256"],
        "p_full_disclosures_sha256": plan["p_full_disclosures_sha256"],
        "p_full_rows_sha256": plan["p_full_rows_sha256"],
        "plan_bundle_sha256": plan["bundle_sha256"],
        "planned_calls": 24,
        "controls": plan["controls"],
    }
    D.json_write(SELFTEST_DIR / "selftest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "selftest", "collect", "analyze", "response"),
                        nargs="?", default="prepare")
    parser.add_argument("--enable-llm", action="store_true")
    parser.add_argument("--max-new", type=int)
    parser.add_argument("--max-tokens", type=int, default=2_000_000)
    parser.add_argument("--request-id")
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare()
    elif args.command == "selftest":
        value = selftest()
    elif args.command == "collect":
        value = collect(enable_llm=args.enable_llm, max_new=args.max_new,
                        max_tokens=args.max_tokens)
    elif args.command == "analyze":
        value = analyze()
    else:
        if not args.request_id:
            raise SystemExit("refuse: response requires --request-id")
        requests, plan = build_plan()
        request = next((r for r in requests if r["request_id"] == args.request_id), None)
        if request is None:
            raise SystemExit(f"refuse: unknown request_id: {args.request_id}")
        record = channels.response(OUT, request)
        value = {"request_id": args.request_id, "succeeded": record.get("succeeded"),
                 "status": record.get("status"), "model": record.get("model"),
                 "provider": record.get("provider"), "usage": record.get("usage"),
                 "latency_s": record.get("latency_s"), "text_sha256": record.get("text_sha256")}
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
