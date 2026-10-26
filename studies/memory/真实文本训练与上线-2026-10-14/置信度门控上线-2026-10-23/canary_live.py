from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/app") if Path("/app/src").is_dir() else Path(os.environ.get("BIODATA_REPO_ROOT", str(HERE.parents[1]))).resolve()
os.environ.setdefault("BIODATA_RELATION_BUNDLE_DIR", str(HERE / "bundle-v2"))
os.environ.setdefault("BIODATA_RELATION_BUNDLE_SHA256", "65a8f9290a01ee038671a9bea068d2047e0897eac5937ecf16cb71bec16ad7b7")
os.environ.setdefault("BIODATA_MODEL_RUNTIME_ROOT", str(HERE / "local-runtime"))
os.environ.setdefault("BIODATA_RELATION_CONFIDENCE_THRESHOLD", "0.80")

from dataset_recommender.agent.memory_relations import assess_relations
from dataset_recommender.llm.llm_client import call_llm, load_llm_config, should_use_llm

cfg = replace(load_llm_config(ROOT), enable_llm=True, mock_llm=False, temperature=0.0, max_tokens=4096, timeout=90.0)
ready, code = should_use_llm(cfg)
if not ready:
    raise SystemExit("Live canary config unavailable: " + str(code))
fp = hashlib.sha256(f"{cfg.base_url}|{cfg.model}".encode()).hexdigest()[:12]
scope = {"principal": "deployment-canary", "endpoint_fp": fp, "project_id": ""}


def entry(text, **extra):
    return {**scope, "text": text, "trigger": "汇报时", "tier": "explicit", "permanence": "persistent", **extra}


def execute(name, candidate, existing, source, edits=()):
    calls, admissions, scores = [], [], []
    def transport(prompt, actual):
        result = call_llm(prompt, actual)
        calls.append({"attempted": result.attempted, "succeeded": result.succeeded, "finish_reason": result.finish_reason, "provider": result.provider, "model": result.model})
        return result
    started = time.perf_counter()
    result = assess_relations(candidate, existing, [{"id": "source-canary", "text": source}], cfg, edits=edits,
        llm_call=transport, before_llm=lambda actual: admissions.append(True), before_score=lambda: scores.append(True))
    trace = result.get("relation_trace", {})
    record = {"name": name, "elapsed_seconds": time.perf_counter() - started, "coverage": result["coverage"], "error_code": result["error_code"], "pairs": [{"old_id": pair["old_id"], "relation": pair["relation"], "error_code": pair["error_code"]} for pair in result["semantic_pairs"]], "edit_checks": [{"entry_id": item["entry_id"], "supported": item["supported"], "error_code": item["error_code"], "preserved": len(item.get("preserved", []))} for item in result["edit_checks"]], "relation_trace": trace, "real_transport": calls, "before_llm_count": len(admissions), "before_score_count": len(scores)}
    assert len(admissions) == len(calls), record
    assert all(pair.get("fallback_reason") != "relation_invalid_identity" for pair in trace.get("pairs", [])), record
    assert result["coverage"]["complete"] and result["error_code"] is None, record
    assert all(call["attempted"] and call["succeeded"] for call in calls), record
    return record


source = "以后汇报先给结论，再给依据。"
candidate = entry("汇报先给结论，再给依据。", evidence=[source])
rows = [entry("汇报先说结论，再解释原因。", id="canary-equal", revision=1), entry("图表要注明单位。", id="canary-independent", revision=1)]
results = []
try:
    results.append(execute("low_confidence_real_model_then_real_llm", candidate, rows, source))
    source_edit = "以后汇报先给结论，再给证据，保留原来的引用来源要求。"
    candidate_edit = entry("汇报时先给结论，然后说明证据；保留引用来源。", id="canary-edit-after", evidence=[source_edit])
    before = entry("汇报时先给证据，然后给结论；保留引用来源。", id="canary-edit-before", revision=1)
    results.append(execute("independent_edit_review", candidate_edit, [], source_edit, edits=[{"before": before, "after": [candidate_edit]}]))
    assert results[0]["before_score_count"] == 1 and results[0]["before_llm_count"] == 1
    assert all(pair["source"] == "llm" and pair["fallback_reason"] == "below_threshold" for pair in results[0]["relation_trace"]["pairs"])
    assert results[1]["before_score_count"] == 0 and results[1]["before_llm_count"] == 1
    assert len(results[1]["edit_checks"]) == 1 and results[1]["edit_checks"][0]["supported"] is True
    passed = True
finally:
    output = {"passed": len(results) == 2 and all(row["error_code"] is None for row in results), "model": cfg.model, "provider": cfg.provider, "threshold": 0.8, "results": results, "limitations": "Synthetic operator-authored input, not human user telemetry or model quality validation. No knowledge committed. High-confidence boundary tested with offline probability fixtures only; actual model probes did not reach 0.80."}
    destination = Path(os.environ.get("BIODATA_CANARY_OUTPUT", str(HERE / "canary-live.json")))
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
