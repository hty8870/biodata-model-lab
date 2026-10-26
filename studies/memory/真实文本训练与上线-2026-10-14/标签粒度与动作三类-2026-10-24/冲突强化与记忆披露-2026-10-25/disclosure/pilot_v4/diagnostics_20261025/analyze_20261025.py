                      
"""Paired supplement, real stale-loader counterexamples and explicit H=100 projections."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import diagnostics_20261025 as diag
from diagnostics_20261025 import D, P, arms, ch, main_results, scoring, file_sha256, json_write, jsonl_read, jsonl_write, sha256


def records(stage):
    folder = D/"runs"/stage
    if not (folder/"run_manifest.json").exists():
        return {}, None, {}
    manifest, started, done = ch.audit_ledger(folder)
    return {rid: ch.read_json(folder/(rid+".outcome.json")) for rid in done}, manifest, started


def verdict(record, answer):
    if not record.get("succeeded"):
        return dict(valid=False, correct=False, complete=False, error="transport_or_missing")
    try:
        v = ch.decode(record["text"])
        if set(v) != {"correct", "omissions", "unsupported_actions", "source_conflict_handling", "answer_evidence", "rationale"} or not isinstance(v["correct"], bool):
            raise ValueError("review_schema")
        for key in ("omissions", "unsupported_actions", "answer_evidence"):
            if not isinstance(v[key], list) or not all(isinstance(x, str) for x in v[key]):
                raise ValueError("review_array")
        conflict = v["source_conflict_handling"]
        if not isinstance(conflict, dict) or set(conflict) != {"applicable", "adequate", "explanation"} or not isinstance(conflict["applicable"], bool) or not isinstance(conflict["adequate"], bool):
            raise ValueError("review_conflict_schema")
        if not isinstance(v["rationale"], str) or not v["rationale"].strip():
            raise ValueError("missing_rationale")
        complete = v["correct"] and not v["omissions"] and not v["unsupported_actions"] and (not conflict["applicable"] or conflict["adequate"])
        return dict(valid=True, correct=v["correct"], complete=bool(complete), judge=v,
                    evidence_literal=all(x and x in answer for x in v["answer_evidence"]),
                    conflicting_judgment=bool(v["correct"] and not complete))
    except (ValueError, TypeError, KeyError) as exc:
        return dict(valid=False, correct=False, complete=False, error=type(exc).__name__+":"+str(exc))


def rate(values, key):
    return dict(correct=sum(bool(x.get(key)) for x in values), n=len(values), rate=sum(bool(x.get(key)) for x in values)/len(values) if values else None)


def pair_cells(left, right):
    pairs = set(left) & set(right)
    cells = Counter((bool(left[k]), bool(right[k])) for k in pairs)
    return dict(n=len(pairs), both_correct=cells[(True, True)], left_only=cells[(True, False)], right_only=cells[(False, True)], both_wrong=cells[(False, False)])


def stage_cost(stage):
    rs, manifest, started = records(stage)
    uses = [r.get("usage") or {} for r in rs.values()]
    return dict(stage=stage, planned=24, reserved=len(started), completed=len(rs), pending_unknown=len(started)-len(rs),
                transport_success=sum(bool(r.get("succeeded")) for r in rs.values()),
                prompt_tokens=sum(u.get("prompt_tokens") or 0 for u in uses), completion_tokens=sum(u.get("completion_tokens") or 0 for u in uses),
                total_tokens=sum(u.get("total_tokens") or 0 for u in uses), unknown_usage=sum(u.get("total_tokens") is None for u in uses),
                latency_s_sum=round(sum(r.get("latency_s") or 0 for r in rs.values()), 3),
                finish_reasons=dict(Counter(str(r.get("finish_reason")) for r in rs.values())),
                identity=manifest["identity"] if manifest else None)


def stale_tests():
    (D/"tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="real-publication-copy-", dir=D/"tmp") as tmp:
        root = Path(tmp)
        shutil.copytree(P/"data", root/"data")
        shutil.copytree(ch.CHANNELS, root/"out/channels")
        with patch.multiple(ch, EXP=root, DATA=root/"data", CHANNELS=root/"out/channels"):
            baseline = {k: ch.load_publication(k) for k in ("compiled", "graph")}
            if not all(v[0] is not None for v in baseline.values()):
                raise ValueError("copied_real_loader_baseline_failed:"+str({k:v[1] for k,v in baseline.items()}))
            manifest = ch.read_json(root/"data/manifest.json")
            manifest["generation"] += 1
            json_write(root/"data/manifest.json", manifest)
            generation = {k: ch.load_publication(k) for k in baseline}
            try:
                ch.source_binding()
            except ValueError as exc:
                generation_source_reason = str(exc)
            else:
                generation_source_reason = None
            shutil.copytree(P/"data", root/"data", dirs_exist_ok=True)
            gs = jsonl_read(root/"data/groups.jsonl")
            old_revision = gs[0]["atoms"][0]["revision"]
            gs[0]["atoms"][0]["revision"] += 1
            jsonl_write(root/"data/groups.jsonl", gs)
            manifest = ch.read_json(root/"data/manifest.json")
            manifest["sha256"]["groups.jsonl"] = file_sha256(root/"data/groups.jsonl")
            json_write(root/"data/manifest.json", manifest)
            new_source = ch.source_binding()
            revision = {k: ch.load_publication(k) for k in baseline}
            result = dict(real_copy_baseline={k:dict(real=v[0] is not None, reason=v[1]) for k,v in baseline.items()},
                          generation_change=dict(new_generation=manifest["generation"]+1, expected_generation=ch.GENERATION,
                                                 source_binding_reason=generation_source_reason,
                                                 loaders={k:dict(rejected=v[0] is None, reason=v[1]) for k,v in generation.items()}),
                          revision_change=dict(old_revision=old_revision, new_revision=old_revision+1, clone_manifest_hash_updated=True,
                                               valid_new_source_sha256=sha256(new_source),
                                               loaders={k:dict(rejected=v[0] is None, reason=v[1]) for k,v in revision.items()}),
                          note="The actual frozen loader is invoked on byte-identical copied real receipts. No real flag/check is patched. Generation mismatch and joint file-hash/revision source binding are tested; automatic refresh/incremental rebuild is not implemented or claimed.")
            assert all(v[0] is None for v in generation.values()) and all(v[0] is None for v in revision.values())
    result["main_unchanged"] = diag.verify_main()
    result["main_old_value_errors"] = dict(errors=0, eligible_observations=19, note="Frozen consumer diagnostic, separate from these cache invalidation mechanism tests")
    json_write(D/"staleness.json", result)
    return result


def add_cost(rows):
    by_provider = defaultdict(lambda: dict(tokens=0.0, latency_s=0.0, calls=0.0))
    for r in rows:
        key = r["identity"]["channel"]+":"+r["identity"]["model"]
        by_provider[key]["tokens"] += r["total_tokens"]
        by_provider[key]["latency_s"] += r["latency_s_sum"]
        by_provider[key]["calls"] += r["completed"]
    return dict(by_provider)


def amortization():
    stages = {r["stage"]:r for r in main_results.cost_table()}
    if sum(r["reserved"] for r in stages.values()) != 256 or sum(r["total_tokens"] for r in stages.values()) != 294554:
        raise ValueError("accepted_main_cost_base_changed")
    builds = {
        "A-compiled": add_cost([stages[k] for k in ("compiler_v1", "compiler_review_v2", "compiler_evidence_repair_v1")]),
        "A-graph": add_cost([stages[k] for k in ("graph_v2", "graph_review_v2")]),
    }
    failures = {"A-compiled":add_cost([stages["compiler_review_v1_rejected"]]), "A-graph":add_cost([stages["graph_v1_format_failed"]])}
    main, _, _ = main_results.read_records(P/"out/consumer")
    disclosures = jsonl_read(P/"out/disclosures.jsonl")
    scenarios, measured = [], []
    for arm in arms.MAIN_ARMS:
        selected = [r for r in disclosures if r["arm"] == arm and r["scale"] == 12]
        obs = [main[r["row_id"]] for r in selected]
        assert len(obs) == 12
        tokens = sum(r["usage"]["total_tokens"] for r in obs)
        latency = sum(r["latency_s"] for r in obs)
        reader = obs[0]["identity"]["channel"]+":"+obs[0]["identity"]["model"]
        measured.append(dict(arm=arm, observed_read_n=12, observed_tokens=tokens, observed_latency_s=round(latency,3),
                             mean_tokens=tokens/12, mean_latency_s=latency/12, initial_build=builds.get(arm,{}), failed_attempts=failures.get(arm,{})))
        for updates in (0,4,5):
            costs = defaultdict(lambda:dict(tokens=0.0, latency_s=0.0, calls=0.0))
            costs[reader] = dict(tokens=100*tokens/12, latency_s=100*latency/12, calls=100)
            for provider, value in builds.get(arm,{}).items():
                for key in ("tokens","latency_s","calls"):
                    costs[provider][key] += (1+updates)*value[key]
            scenarios.append(dict(arm=arm, H=100, initial_builds=1 if arm in builds else 0, updates=updates,
                                  update_rebuilds=updates if arm in builds else 0, projected_by_provider=dict(costs),
                                  projected_total_tokens=sum(v["tokens"] for v in costs.values()),
                                  projected_latency_s=sum(v["latency_s"] for v in costs.values()),
                                  historical_failure_cost_separate=failures.get(arm,{}), is_extrapolation=True))
    result = dict(measured=measured, projected=scenarios,
                  assumptions=["H=100 uses uniform mean of the twelve frozen main S12 DS reads, not 100 new measured reads.",
                               "Initial build occurs before read1. Primary update case updates after20/40/60/80, four rebuilds affecting this horizon. Five-update sensitivity also charges post-read100 maintenance for the next horizon.",
                               "Each projected update is a full-library rebuild priced by the one observed successful release, including compiled quote repair. Actual updated-corpus inference/review costs and repair frequency are unknown.",
                               "P/A full and ordinary retrieval have zero EXTRA LLM build calls in this comparison; authoring, atomization, deterministic CPU/IO, energy and initial dataset costs are not measured as zero.",
                               "Latency sums assume sequential calls and inherit observed cache/load/server conditions; they are not guaranteed serving latency.",
                               "Provider tokens remain split by tokenizer/model; token sums are workload counts, not common monetary units. No currency price or invoice is fabricated.",
                               "Discarded initial reviewer and old graph batches are shown separately, not charged again at every future update. They are real spent costs, not claims that every call in those batches failed transport."],
                  money_cost=None)
    json_write(D/"amortization.json", result)
    return result


def analyze():
    rows, tasks = diag.selected()
    main, _, _ = main_results.read_records(P/"out/consumer")
    qrs, _, _ = records("qwen_struct")
    frs, _, _ = records("ds_free")
    brs, _, _ = records("qwen_blind")
    plans = {s:ch.read_json(D/"plans"/(s+".json"))["requests"] for s in diag.STAGES}
    indexes = {s:{r["base_row_id"]:r for r in plans[s]} for s in diag.STAGES}
    details = []
    for row in rows:
        rid = row["row_id"]
        task = tasks[row["task_id"]]
        qr = qrs.get(indexes["qwen_struct"][rid]["request_id"],{})
        fr = frs.get(indexes["ds_free"][rid]["request_id"],{})
        breq = indexes["qwen_blind"][rid]
        br = brs.get(breq["request_id"],{})
        ds_score = scoring.score_receipt(main[rid],row,task)
        q_score = scoring.score_receipt(qr,row,task)
        semantic = verdict(br,breq["blind_answer_view"])
        semantic["free_transport_success"] = bool(fr.get("succeeded"))
        semantic["complete"] = bool(semantic["complete"] and fr.get("succeeded"))
        details.append(dict(base_row_id=rid,task_id=row["task_id"],arm=row["arm"],question=task["question"],context=task["context"],
                            ds_struct_score=ds_score,qwen_struct_score=q_score,
                            ds_struct_answer=scoring.parse(main[rid]["text"])[0],qwen_struct_answer=scoring.parse(qr.get("text",""))[0],
                            ds_free_text=fr.get("text",""),blind_answer_view=breq["blind_answer_view"],qwen_blind=semantic,
                            hashes=dict(main_receipt=sha256(main[rid]),qwen_receipt=sha256(qr) if qr else None,
                                        free_receipt=sha256(fr) if fr else None,blind_receipt=sha256(br) if br else None)))
    summary=[]
    for arm in ("P-full","A-full"):
        rs=[r for r in details if r["arm"]==arm]
        summary.append(dict(arm=arm,n=12,ds_exact=rate([r["ds_struct_score"] for r in rs],"exact_success"),
                            ds_grounded=rate([r["ds_struct_score"] for r in rs],"grounded_fact"),
                            qwen_exact=rate([r["qwen_struct_score"] for r in rs],"exact_success"),
                            qwen_grounded=rate([r["qwen_struct_score"] for r in rs],"grounded_fact"),
                            ai_blind_correct=rate([r["qwen_blind"] for r in rs],"correct"),
                            ai_blind_complete=rate([r["qwen_blind"] for r in rs],"complete"),
                            valid_blind_reviews=sum(r["qwen_blind"]["valid"] for r in rs),
                            nonliteral_blind_evidence=sum(r["qwen_blind"].get("evidence_literal") is False for r in rs),
                            cross_reader_pair=pair_cells({r["task_id"]:r["ds_struct_score"]["grounded_fact"] for r in rs},{r["task_id"]:r["qwen_struct_score"]["grounded_fact"] for r in rs})))
    costs=[stage_cost(s) for s in diag.STAGES]
    result=dict(schema="registered-s12-diagnostics-v1",paired_task_n=12,observations=24,summary=summary,
                costs=costs,added_calls=sum(c["reserved"] for c in costs),added_tokens=sum(c["total_tokens"] for c in costs),
                cumulative_calls=256+sum(c["reserved"] for c in costs),cumulative_tokens=294554+sum(c["total_tokens"] for c in costs),
                free_reviewer_type="Qwen AI proxy, not human gold; no arm/expected/main score in prompt",
                limitations=["Same shared twelve-task library; no new heldout claims, significance, or score replacement.",
                             "Qwen has provider JSON format plus prompt; DS main has prompt JSON constraint. Difference is not solely reader weights.",
                             "Free-vs-structured changes output requirements and assessor; do not label the gap a pure causal format penalty.",
                             "Blind rationale/evidence quality is reported; nonliteral evidence is not silently rewritten or re-queried."],
                main_unchanged=diag.verify_main())
    jsonl_write(D/"paired_results.jsonl",details)
    json_write(D/"results.json",result)
    return result


def selftest():
    v=dict(correct=True,omissions=[],unsupported_actions=[],source_conflict_handling=dict(applicable=False,adequate=True,explanation="none"),answer_evidence=["记忆未规定"],rationale="source-only fixture")
    ok=verdict(dict(succeeded=True,text=json.dumps(v,ensure_ascii=False)),"记忆未规定")
    bad=verdict(dict(succeeded=True,text=json.dumps(v|dict(unsupported_actions=["invented"]),ensure_ascii=False)),"记忆未规定")
    assert ok["complete"] and not bad["complete"] and bad["conflicting_judgment"]
    return dict(ok=True,known_semantic_pass_fail=True,real_model_calls=0)


def seal():
    diag.verify_main()
    for s in diag.STAGES:
        rs, _, started=records(s)
        if len(rs)!=24 or len(started)!=24:
            raise ValueError("diagnostic_group_incomplete_or_unknown:"+s)
    paths={str(p.relative_to(D)).replace("\\","/"):file_sha256(p) for p in D.rglob("*") if p.is_file() and "tmp" not in p.relative_to(D).parts and p.name!="diagnostics_manifest.json"}
    ch.immutable_json(D/"diagnostics_manifest.json",dict(schema="diagnostics-final-v1",sha256=paths,main=diag.verify_main()))
    return verify()


def verify():
    manifest=ch.read_json(D/"diagnostics_manifest.json")
    paths={str(p.relative_to(D)).replace("\\","/"):file_sha256(p) for p in D.rglob("*") if p.is_file() and "tmp" not in p.relative_to(D).parts and p.name!="diagnostics_manifest.json"}
    if paths!=manifest["sha256"]:
        raise ValueError("diagnostic_result_sha_mismatch")
    return dict(ok=True,files=len(paths),diagnostics_manifest_sha256=file_sha256(D/"diagnostics_manifest.json"),main=diag.verify_main())


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("selftest","cpu","analyze","seal","verify"))
    args=parser.parse_args()
    if args.command=="cpu":
        value=dict(stale=stale_tests(),amortization=amortization())
        print(json.dumps(dict(ok=True,stale_cases=2,projection_rows=len(value["amortization"]["projected"])),ensure_ascii=False))
    else:
        value={"selftest":selftest,"analyze":analyze,"seal":seal,"verify":verify}[args.command]()
        print(json.dumps(value if args.command!="analyze" else {k:v for k,v in value.items() if k!="costs"},ensure_ascii=False,indent=2))
