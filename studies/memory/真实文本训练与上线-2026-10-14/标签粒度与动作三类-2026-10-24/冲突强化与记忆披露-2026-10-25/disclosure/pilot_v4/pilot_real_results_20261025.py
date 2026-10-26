                      
"""Deterministic real-run accounting; no model calls, relabels, or result repair."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pilot_arms_20261025 as arms
import pilot_channels_20261025 as ch
import pilot_score_20261025 as score
from disclosure_schema_20261025 import file_sha256, json_write, jsonl_read, jsonl_write, sha256

ROOT = ch.OUT / "real_results"
STAGES = {
    "compiler_v1": ch.CHANNELS / "compiled/responses",
    "compiler_review_v1_rejected": ch.CHANNELS / "compiled/review_responses",
    "compiler_review_v2": ch.artifact_path("compiled", "review_responses"),
    "graph_v1_format_failed": ch.CHANNELS / "graph/responses",
    "graph_v2": ch.artifact_path("graph", "responses"),
    "graph_review_v2": ch.artifact_path("graph", "review_responses"),
    "compiler_evidence_repair_v1": ch.CHANNELS / "compiled/evidence_repair_v1/responses",
    "consumer_all": ch.OUT / "consumer",
}


def protected_files():
    paths = list(ch.DATA.glob("*")) + [ch.CHANNELS / "compiled/candidate.json"]
    for name in ("compiler_v1", "compiler_review_v1_rejected", "graph_v1_format_failed"):
        paths += list(STAGES[name].glob("*"))
    return {str(p.relative_to(ch.EXP)).replace("\\", "/"): file_sha256(p) for p in paths if p.is_file()}


def snapshot():
    value = dict(schema="real-run-original-protection-v1", sha256=protected_files(),
                 source_sha256=sha256(ch.source_binding()), note="Original data, compiler cards, and all first-attempt receipts remain immutable.")
    ch.immutable_json(ROOT / "original_artifacts.json", value)
    return dict(ok=True, protected_files=len(value["sha256"]), source_sha256=value["source_sha256"])


def verify_originals():
    value = ch.read_json(ROOT / "original_artifacts.json")
    if value["sha256"] != protected_files() or value["source_sha256"] != sha256(ch.source_binding()):
        raise ValueError("original_artifacts_changed")
    return dict(ok=True, protected_files=len(value["sha256"]))


def read_records(folder):
    if not (folder / "run_manifest.json").exists():
        return {}, {}, None
    manifest, started, done = ch.audit_ledger(folder)
    return {rid: ch.read_json(folder / f"{rid}.outcome.json") for rid in done}, started, manifest


def cost_table():
    table = []
    for stage, folder in STAGES.items():
        records, started, manifest = read_records(folder)
        if manifest is None:
            continue
        lat = [r["latency_s"] for r in records.values() if isinstance(r.get("latency_s"), (float, int))]
        usage = [r["usage"] for r in records.values()]
        decoded = 0
        serving = []
        for r in records.values():
            try:
                v = ch.decode(r["text"])
                decoded += 1
                if "serving" in v:
                    serving.append(bool(v["serving"]))
            except ValueError:
                pass
        table.append(dict(stage=stage, folder=str(folder.relative_to(ch.EXP)), requests=len(manifest["requests"]),
                          reserved=len(started), completed=len(records), attempted=sum(bool(r["attempted"]) for r in records.values()),
                          transport_success=sum(bool(r["succeeded"]) for r in records.values()),
                          pending_unknown=len(started)-len(records), json_only_outputs=decoded,
                          review_serving=sum(serving) if serving else None, review_rejected=len(serving)-sum(serving) if serving else None,
                          finish_reasons=dict(Counter(str(r.get("finish_reason")) for r in records.values())),
                          identity=manifest["identity"],
                          prompt_tokens=sum(u.get("prompt_tokens") or 0 for u in usage),
                          completion_tokens=sum(u.get("completion_tokens") or 0 for u in usage),
                          total_tokens=sum(u.get("total_tokens") or 0 for u in usage),
                          unknown_usage=sum(u.get("total_tokens") is None for u in usage),
                          latency_s_sum=round(sum(lat), 3), latency_s_median=round(statistics.median(lat), 3) if lat else None,
                          max_latency_s=round(max(lat), 3) if lat else None))
    return table


def error_labels(record, row, task, sc):
    labels = []
    if not record.get("succeeded"):
        labels.append("transport_or_missing")
    answer, reason = score.parse(record.get("text", ""))
    if reason != "ok":
        labels.append("format:"+reason)
        return labels
    expected = task["expected"]
    if answer["decision"] != expected["decision"]:
        labels.append("decision")
    if not score.value_ok(expected["value"], answer["value"], expected["value_mode"]):
        labels.append("value")
    for k in ("applied_condition", "applied_exception"):
        if not score.scope_ok(expected[k], answer[k]):
            labels.append(k)
    if sc["stale_error"]:
        labels.append("stale_value")
    if sc["invalid_citations"]:
        labels.append("invalid_citation")
    if sc["missing_facts"] or sc["missing_evidence"] or sc["missing_groups"]:
        labels.append("citation_or_disclosure_incomplete")
    return labels


def paired(rows, a, b):
    left = {(r["task_id"], r["scale"]): r for r in rows if r["arm"] == a}
    right = {(r["task_id"], r["scale"]): r for r in rows if r["arm"] == b}
    pairs = sorted(set(left) & set(right))
    out = dict(a=a, b=b, paired_n=len(pairs), independent_task_n=len({t for t, _ in pairs}))
    for metric in ("exact_success", "grounded_fact", "grounded_group"):
        cells = Counter((bool(left[p]["score"][metric]), bool(right[p]["score"][metric])) for p in pairs)
        out[metric] = dict(both_correct=cells[(True, True)], a_only=cells[(True, False)], b_only=cells[(False, True)],
                           both_wrong=cells[(False, False)], b_minus_a=(cells[(False, True)]-cells[(True, False)])/len(pairs) if pairs else None)
    return out


def analyze():
    original_check = verify_originals()
    records, started, manifest = read_records(ch.OUT / "consumer")
    disclosure = {r["row_id"]: r for r in jsonl_read(ch.OUT / "disclosures.jsonl")}
    plan = jsonl_read(ch.OUT / "collect_plan.jsonl")
    tasks = {t["task_id"]: t for t in arms.load()[1]}
    rows = []
    for p in plan:
        row = disclosure[p["row_id"]]
        rec = records.get(row["row_id"], {})
        sc = score.score_receipt(rec, row, tasks[row["task_id"]])
        answer, _ = score.parse(rec.get("text", ""))
        rows.append(dict(row_id=row["row_id"], arm=row["arm"], task_id=row["task_id"], scale=row["scale"], split=row["split"],
                         phenomenon=row["phenomenon"], score=sc if manifest else None, answer=answer if rec else None,
                         expected=tasks[row["task_id"]]["expected"],
                         error_labels=error_labels(rec, row, tasks[row["task_id"]], sc) if manifest else [],
                         status=rec.get("status", "pending_unknown" if row["row_id"] in started else "not_attempted"),
                         prompt_sha256=row["prompt_sha256"], receipt_sha256=sha256(rec) if rec else None,
                         usage=rec.get("usage"), latency_s=rec.get("latency_s")))
    by_arm = []
    for arm in arms.ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        metrics = {k: dict(correct=sum(bool(r["score"][k]) for r in rs) if manifest else None,
                           n=len(rs) if manifest else 0, planned_n=len(rs),
                           rate=sum(bool(r["score"][k]) for r in rs)/len(rs) if manifest else None)
                   for k in ("exact_success", "grounded_fact", "grounded_group", "stale_error")}
        by_arm.append(dict(arm=arm, metrics=metrics, attempted=sum(r["row_id"] in records for r in rs),
                           error_counts=dict(Counter(e for r in rs for e in r["error_labels"])),
                           provider_total_tokens=sum((r["usage"] or {}).get("total_tokens") or 0 for r in rs),
                           latency_s_sum=round(sum(r["latency_s"] or 0 for r in rs), 3),
                           mean_citation_count=sum(r["score"]["citation_count"] for r in rs)/len(rs) if manifest else None,
                           mean_citation_fact_coverage=sum(r["score"]["citation_fact_coverage"] for r in rs)/len(rs) if manifest else None,
                           mean_citation_evidence_coverage=sum(r["score"]["citation_evidence_coverage"] for r in rs)/len(rs) if manifest else None))
    states = {k: ch.load_publication(k) for k in ("compiled", "graph")}
    compiler_candidate = ch.candidate_current("compiled")
    review_diagnostics = []
    effective = states["compiled"][0].get("derived_review_views", {}).get("values", {}) if states["compiled"][0] else {}
    for req in ch.review_requests("compiled", save=False):
        rec = ch.response(ch.artifact_path("compiled", "review_responses"), req)
        value = ch.decode(rec["text"])
        card = next(c for c in compiler_candidate["content"]["cards"] if c["card_id"] == req["card_id"])
        raw_gate = ch.semantic_gate("compiled", compiler_candidate, [rec], [req])
        gate = ch.semantic_gate("compiled", compiler_candidate, [rec], [req], effective)
        review_diagnostics.append(dict(card_id=card["card_id"], reviewer_serving=value["serving"], gate=gate, raw_gate=raw_gate,
                                       invalid_original_evidence=[x for x in value["source_checks"] if x.get("evidence_in_card") not in card["text"]],
                                       evidence_repaired=req["request_id"] in effective, receipt_sha256=sha256(rec)))
    costs = cost_table()
    result = dict(schema="pilot-real-results-v2", original_check=original_check,
                  consumer=dict(state="started" if manifest else "not_started_publication_gate_failed", planned=len(plan),
                                main_planned=216, grouped_planned=4, completed=len(records),
                                pending_unknown=len(started)-len(records), failed_transport=sum(not r["succeeded"] for r in records.values()),
                                by_arm=by_arm, error_rows=sum(not r["score"]["grounded_fact"] for r in rows) if manifest else None,
                                note="An intentionally blocked run has n=0 and null rates, not 220 behavioral failures. Once started, missing/failed planned calls remain in the denominator."),
                  publications={k: dict(real=a is not None, reason=reason, path=str(ch.artifact_path(k, "publication").relative_to(ch.EXP)),
                                       sha256=file_sha256(ch.artifact_path(k, "publication")) if ch.artifact_path(k, "publication").exists() else None,
                                       gate=ch.read_json(ch.artifact_path(k, "publication"))["gate"] if ch.artifact_path(k, "publication").exists() else None) for k, (a, reason) in states.items()},
                  compiler_review_diagnostics=review_diagnostics,
                  cost_stages=costs, total_calls=sum(s["reserved"] for s in costs), total_completed=sum(s["completed"] for s in costs),
                  provider_total_tokens=sum(s["total_tokens"] for s in costs),
                  money_cost=None, money_cost_note="No billed invoice or frozen price sheet; provider tokens and wall latency reported, local GPU energy not priced.",
                  paired=[paired(rows, a, b) for a, b in (("P-full", "A-full"), ("P-retrieve", "A-retrieve"),
                                                         ("A-retrieve", "A-graph"), ("A-full", "A-compiled"), ("P-full", "A-grouped"))] if manifest else [],
                  graph_diagnostics=states["graph"][0]["content"]["diagnostics"] if states["graph"][0] else None,
                  limitations=arms.LIMITATIONS,
                  attribution="Twelve tasks share a library and three repeated scales, not memory-heldout or human gold. Citation-unit burden and connectivity differ; no significance/pure-storage claim.")
    if manifest:
        result["consumer"]["state"] = "completed" if len(records) == len(plan) else "running_or_partial"
    result["by_task_arm"] = []
    for (tid, arm), members in sorted(_group_rows(rows).items()):
        result["by_task_arm"].append(dict(task_id=tid, arm=arm, planned_n=len(members), n=len(members) if manifest else 0,
                                          exact=sum(r["score"]["exact_success"] for r in members) if manifest else None,
                                          grounded=sum(r["score"]["grounded_fact"] for r in members) if manifest else None,
                                          failed_scales=[r["scale"] for r in members if r["score"] and not r["score"]["grounded_fact"]]))
    for cell in by_arm:
        eligible = [r for r in rows if r["arm"] == cell["arm"] and tasks[r["task_id"]].get("stale_value") is not None]
        cell["stale_eligible"] = dict(n=len(eligible) if manifest else 0, errors=sum(r["score"]["stale_error"] for r in eligible) if manifest else None)
    for pair in result["paired"]:
        pair["unique_task_n"] = pair.pop("independent_task_n")
    ROOT.mkdir(parents=True, exist_ok=True)
    jsonl_write(ROOT / ("scored_all_220.jsonl" if manifest else "planned_not_attempted.jsonl"), rows)
    jsonl_write(ROOT / "errors.jsonl", [r for r in rows if r["score"] and not r["score"]["grounded_fact"]])
    json_write(ROOT / "summary.json", result)
    return result


def _group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task_id"], row["arm"])].append(row)
    return grouped


def resource_state():
    result = {}
    for name, args in (("ollama_ps", ["ollama", "ps"]), ("gpu", ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"])):
        p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        result[name] = dict(exit_code=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())
    json_write(ROOT / "resource_state.json", result)
    return result


def result_files():
    return {str(p.relative_to(ch.EXP)).replace("\\", "/"): file_sha256(p)
            for folder in (ROOT, ch.OUT / "consumer") for p in folder.rglob("*") if p.is_file()}


def verify_results():
    manifest = ch.read_json(ch.OUT / "final_results_manifest_v1.json")
    inputs = arms.verify_frozen()
    records, started, run = read_records(ch.OUT / "consumer")
    if manifest["input_frozen_sha256"] != inputs["frozen_sha256"] or manifest["sha256"] != result_files():
        raise ValueError("final_result_sha_mismatch")
    if run["binding"]["frozen_sha256"] != inputs["frozen_sha256"] or len(records) != len(started) or len(records) != 220:
        raise ValueError("incomplete_or_different_consumer_run")
    request_index = {r["request_id"]: r for r in run["requests"]}
    for rid, record in records.items():
        raw = ch.OUT / "consumer" / (rid+".raw.txt")
        req = request_index[rid]
        if file_sha256(raw) != record["raw_sha256"] or raw.read_text(encoding="utf-8") != record["text"] or sha256(record["text"]) != record["text_sha256"]:
            raise ValueError("consumer_raw_receipt_mismatch")
        if record["identity"] != run["identity"] or any(record[k] != req[k] for k in ("prompt_sha256", "input_sha256", "source_sha256", "stage")):
            raise ValueError("consumer_request_identity_mismatch")
    verify_originals()
    return dict(ok=True, files=len(manifest["sha256"]), consumer_completed=len(records),
                input_frozen_sha256=inputs["frozen_sha256"], result_manifest_sha256=file_sha256(ch.OUT / "final_results_manifest_v1.json"))


def freeze_results():
    inputs = arms.verify_frozen()
    records, started, run = read_records(ch.OUT / "consumer")
    summary = ch.read_json(ch.OUT / "consumer/collection_summary.json")
    if len(records) != 220 or len(started) != 220 or summary["denominator"] != 220 or summary["completed"] != 220:
        raise ValueError("cannot_seal_incomplete_results")
    manifest = dict(schema="pilot-final-results-v1", input_frozen_sha256=inputs["frozen_sha256"],
                    sha256=result_files(), hash_policy="immutable input manifest -> consumer run; result leaves -> this output manifest, never back into inputs")
    ch.immutable_json(ch.OUT / "final_results_manifest_v1.json", manifest)
    return verify_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("snapshot", "verify-originals", "analyze", "resources", "freeze-results", "verify-results"))
    a = parser.parse_args()
    value = {"snapshot": snapshot, "verify-originals": verify_originals, "analyze": analyze, "resources": resource_state,
             "freeze-results": freeze_results, "verify-results": verify_results}[a.command]()
    if a.command == "analyze":
        value = {k: v for k, v in value.items() if k not in ("cost_stages", "paired", "graph_diagnostics", "limitations")}
    print(json.dumps(value, ensure_ascii=False, indent=2))
