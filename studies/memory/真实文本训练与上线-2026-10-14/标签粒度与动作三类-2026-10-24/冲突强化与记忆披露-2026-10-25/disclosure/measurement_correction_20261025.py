                      
"""Offline layered re-score for the 220 main observations and 24 Qwen-structure reads.

This is a post-hoc diagnostic, not a new gold set and not a confirmatory score.
It reuses the frozen pilot scorer's parse/value_ok/disclosed_items/support and keeps
the original strict Exact/Grounded fields unchanged and separate.
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
import pilot_score_20261025 as S              

OUTDIR = HERE / "parent_analysis_20261025"
POLICY = OUTDIR / "correction-policy-v1.json"
SOURCE_INPUT = OUTDIR / "source-only-reference-input-v1.json"
SOURCE_RAW = OUTDIR / "source-only-reference-v1.raw.json"
SOURCE_ALIASES = OUTDIR / "source_value_aliases_v1.json"
SCORED = P / "out/real_results/scored_all_220.jsonl"
DISCLOSURES = P / "out/disclosures.jsonl"
PAIRED = P / "diagnostics_20261025/paired_results.jsonl"
ID_MAP = P / "data/id_map.json"
GROUPS = P / "data/groups.jsonl"
TASKS = P / "data/tasks.jsonl"
MAIN_ARMS = ("P-full", "A-full", "P-retrieve", "A-retrieve", "A-graph", "A-compiled")
CONTROL_ARM = "A-grouped"
NEW_METRIC = "动作/有限别名值匹配"
NEW_METRIC_NOTE = ("schema合法、decision==policy.decision，且任一policy.value_variants由原"
                   "S.value_ok(value_mode)匹配；applied_condition/exception不参与该投影。"
                   "这是有限别名动作/值匹配，不是完整NLP语义真值。")
ORIGINAL_NOTE = "原strict Exact/Grounded是未改的字段合同指标，单独保留；本脚本不改写它们。"


def need(path: Path, label: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"refuse: missing {label}: {path}")
    return path


def load_context() -> dict:
    for path, label in ((POLICY, "policy"), (SOURCE_INPUT, "source input"), (SOURCE_RAW, "raw source"),
                        (SOURCE_ALIASES, "source aliases"), (SCORED, "scored_all_220"),
                        (DISCLOSURES, "disclosures"), (PAIRED, "paired_results"),
                        (ID_MAP, "id_map"), (GROUPS, "groups"), (TASKS, "tasks")):
        need(path, label)
    policy = D.json_read(POLICY)
    if policy.get("schema") != "memory-measurement-correction-policy/1":
        raise SystemExit("refuse: unexpected policy schema")
    for key, path, label in (("source_input_sha256", SOURCE_INPUT, "source input"),
                             ("raw_reference_sha256", SOURCE_RAW, "raw source"),
                             ("source_aliases_sha256", SOURCE_ALIASES, "source aliases")):
        if policy.get(key) != D.file_sha256(path):
            raise SystemExit(f"refuse: policy {key} does not match {label}")
    policy_tasks = policy.get("tasks")
    if not isinstance(policy_tasks, list) or len(policy_tasks) != 12:
        raise SystemExit("refuse: policy must contain 12 tasks")
    policy_by_task = {t["task_id"]: t for t in policy_tasks}
    if len(policy_by_task) != 12:
        raise SystemExit("refuse: duplicate policy task ids")
    statuses = Counter(t.get("status") for t in policy_tasks)
    if set(statuses) - {"valid", "ambiguous"}:
        raise SystemExit("refuse: unsupported task status in policy")
    for task_id, task in policy_by_task.items():
        if task["status"] == "ambiguous":
            if task.get("decision") is not None or task.get("minimal_atom_sets") != []:
                raise SystemExit(f"refuse: ambiguous task {task_id} has scoring fields")
            continue
        if task.get("decision") not in S.DECISIONS:
            raise SystemExit(f"refuse: bad policy decision for {task_id}")
        if not task.get("value_variants"):
            raise SystemExit(f"refuse: empty value_variants for {task_id}")
        if task["decision"] != "no_memory" and not task.get("minimal_atom_sets"):
            raise SystemExit(f"refuse: missing minimal_atom_sets for {task_id}")
        if task["decision"] == "no_memory" and task.get("minimal_atom_sets") != []:
            raise SystemExit(f"refuse: no_memory task {task_id} must have empty minimal_atom_sets")

    tasks = {t["task_id"]: t for t in D.jsonl_read(TASKS)}
    if set(tasks) != set(policy_by_task):
        raise SystemExit("refuse: policy/tasks task-id mismatch")
    groups = {g["group_id"]: g for g in D.jsonl_read(GROUPS)}
    id_map = D.json_read(ID_MAP)
    public_facts = defaultdict(set)
    for group in groups.values():
        for logical_id, fact_ids in group["atomic_fact_ids"].items():
            public_id = id_map["items"].get(logical_id)
            if not public_id:
                raise SystemExit(f"refuse: missing public id for {logical_id}")
            public_facts[public_id].update(fact_ids)
    for task in policy_tasks:
        for atom_set in task.get("minimal_atom_sets", []):
            for atom in atom_set:
                if not public_facts.get(atom):
                    raise SystemExit(f"refuse: policy atom {atom} has no fact mapping")

    scored = D.jsonl_read(SCORED)
    if len(scored) != 220 or len({r["row_id"] for r in scored}) != 220:
        raise SystemExit("refuse: scored_all_220 must contain 220 unique rows")
    scored_by_id = {r["row_id"]: r for r in scored}
    disclosures = D.jsonl_read(DISCLOSURES)
    if len(disclosures) != 252 or len({r["row_id"] for r in disclosures}) != 252:
        raise SystemExit("refuse: disclosures must contain 252 unique rows")
    disc_by_id = {r["row_id"]: r for r in disclosures}
    missing_disc = sorted(set(scored_by_id) - set(disc_by_id))
    if missing_disc:
        raise SystemExit(f"refuse: scored rows missing disclosure: {missing_disc[:5]}")
    for row_id, row in scored_by_id.items():
        disc = disc_by_id[row_id]
        for key in ("task_id", "arm", "scale"):
            if row.get(key) != disc.get(key):
                raise SystemExit(f"refuse: {row_id} disclosure/{key} mismatch")
    paired = D.jsonl_read(PAIRED)
    if len(paired) != 24 or len({r["base_row_id"] for r in paired}) != 24:
        raise SystemExit("refuse: paired_results must contain 24 unique base rows")
    for row in paired:
        base_id = row["base_row_id"]
        if base_id not in scored_by_id or base_id not in disc_by_id:
            raise SystemExit(f"refuse: paired base row missing: {base_id}")
        if row["arm"] not in ("P-full", "A-full") or scored_by_id[base_id]["scale"] != 12:
            raise SystemExit(f"refuse: paired row is not P/A-full S12: {base_id}")
        if not isinstance(row.get("qwen_struct_answer"), dict) or not isinstance(row.get("qwen_struct_score"), dict):
            raise SystemExit(f"refuse: paired row missing Qwen structure output: {base_id}")
    if len({(r["task_id"], r["arm"]) for r in paired}) != 24:
        raise SystemExit("refuse: paired Qwen rows do not cover 12 tasks x 2 arms")
    return dict(policy=policy, policy_by_task=policy_by_task, tasks=tasks, groups=groups,
                public_facts=public_facts, scored=scored, scored_by_id=scored_by_id,
                disc_by_id=disc_by_id, paired=paired)


def evaluate(answer: dict, disclosure: dict, policy_task: dict, public_facts: dict) -> dict:
    parsed, parse_reason = S.parse(answer)
    base = {"schema_ok": None, "decision_ok": None, "value_ok": None,
            "action_alias_value_match": None, "minimal_evidence_sufficient": None,
            "layered_success": None, "invalid_citations": [], "cited": [],
            "supported_fact_count": None, "minimal_sets": []}
    if policy_task["status"] == "ambiguous":
        return {**base, "status": "unknown", "reason": "reference_task_ambiguous"}
    if parse_reason != "ok":
        return {**base, "status": "scored", "schema_ok": False, "decision_ok": False,
                "value_ok": False, "action_alias_value_match": False,
                "minimal_evidence_sufficient": False, "layered_success": False,
                "reason": f"schema_invalid:{parse_reason}"}
    decision_ok = parsed["decision"] == policy_task["decision"]
    value_ok = any(S.value_ok(variant, parsed.get("value"), policy_task["value_mode"])
                   for variant in policy_task["value_variants"])
    action = bool(decision_ok and value_ok)
    items = S.disclosed_items(disclosure)
    cited = parsed.get("cited_item_ids", [])
    invalid = sorted(set(cited) - set(items))
    valid = set(cited) & set(items)
    facts, evidence, _ = S.support(disclosure, items, valid)
    supported = facts | evidence
    if policy_task["decision"] == "no_memory":
        minimal = not invalid and not cited
        minimal_sets = []
    else:
        minimal_sets = []
        for atom_set in policy_task["minimal_atom_sets"]:
            required = set().union(*(public_facts[atom] for atom in atom_set)) if atom_set else set()
            minimal_sets.append({"atoms": list(atom_set), "facts": sorted(required),
                                 "missing": sorted(required - supported)})
        minimal = not invalid and any(not entry["missing"] for entry in minimal_sets)
    reasons = []
    if not decision_ok:
        reasons.append("decision_mismatch")
    if not value_ok:
        reasons.append("value_mismatch")
    if invalid:
        reasons.append("invalid_citation")
    if policy_task["decision"] == "no_memory" and cited:
        reasons.append("no_memory_cited_nonempty")
    if not minimal:
        reasons.append("minimal_evidence_not_supported")
    reason = "action_value_and_minimal_evidence_ok" if action and minimal else ";".join(reasons) or "not_matched"
    return {**base, "status": "scored", "schema_ok": True, "decision_ok": bool(decision_ok),
            "value_ok": bool(value_ok), "action_alias_value_match": action,
            "minimal_evidence_sufficient": bool(minimal), "layered_success": bool(action and minimal),
            "invalid_citations": invalid, "cited": cited, "supported_fact_count": len(supported),
            "minimal_sets": minimal_sets, "reason": reason}


def old_strict(score: dict) -> dict:
    return {"exact_success": bool(score.get("exact_success")),
            "grounded_success": bool(score.get("grounded_success")),
            "stale_error": bool(score.get("stale_error")),
            "parse_reason": score.get("parse_reason"),
            "invalid_citations": score.get("invalid_citations", []),
            "cited": score.get("cited", [])}


def build_observations(ctx: dict) -> list[dict]:
    observations = []
    for row in ctx["scored"]:
        disc = ctx["disc_by_id"][row["row_id"]]
        evaluated = evaluate(row["answer"], disc, ctx["policy_by_task"][row["task_id"]], ctx["public_facts"])
        observations.append({
            "source": "main_ds", "row_id": row["row_id"], "task_id": row["task_id"],
            "arm": row["arm"], "scale": row["scale"],
            "role": "control_grouped" if row["arm"] == CONTROL_ARM else "main_arm",
            "policy_status": ctx["policy_by_task"][row["task_id"]]["status"],
            "answer": row["answer"], "old_strict": old_strict(row["score"]),
            "new": evaluated, "reason": evaluated["reason"]})
    for pair in ctx["paired"]:
        base_id = pair["base_row_id"]
        base = ctx["scored_by_id"][base_id]
        disc = ctx["disc_by_id"][base_id]
        evaluated = evaluate(pair["qwen_struct_answer"], disc,
                             ctx["policy_by_task"][base["task_id"]], ctx["public_facts"])
        observations.append({
            "source": "qwen_struct", "row_id": base_id, "task_id": base["task_id"],
            "arm": base["arm"], "scale": base["scale"], "role": "qwen_struct_reader",
            "policy_status": ctx["policy_by_task"][base["task_id"]]["status"],
            "answer": pair["qwen_struct_answer"], "old_strict": old_strict(pair["qwen_struct_score"]),
            "new": evaluated, "reason": evaluated["reason"]})
    return observations


def rate(rows: list[dict], key: str) -> dict:
    n = len(rows)
    correct = sum(bool(row["new"][key]) for row in rows)
    return {"correct": correct, "n": n, "rate": (correct / n) if n else None}


def summarize(observations: list[dict]) -> dict:
    main = [row for row in observations if row["source"] == "main_ds"]
    qwen = [row for row in observations if row["source"] == "qwen_struct"]
    main_six = [row for row in main if row["role"] == "main_arm" and row["new"]["status"] == "scored"]
    grouped = [row for row in main if row["role"] == "control_grouped" and row["new"]["status"] == "scored"]
    qwen_valid = [row for row in qwen if row["new"]["status"] == "scored"]
    unknown = [row for row in observations if row["new"]["status"] == "unknown"]
    if (len(main_six), len(grouped), len(qwen_valid), len(unknown)) != (198, 4, 22, 20):
        raise SystemExit("refuse: unexpected denominators; inputs changed")
    main_old = {"exact_success": sum(row["old_strict"]["exact_success"] for row in main),
                "grounded_success": sum(row["old_strict"]["grounded_success"] for row in main),
                "stale_error": sum(row["old_strict"]["stale_error"] for row in main), "n": len(main)}
    qwen_old = {"exact_success": sum(row["old_strict"]["exact_success"] for row in qwen),
                "grounded_success": sum(row["old_strict"]["grounded_success"] for row in qwen),
                "stale_error": sum(row["old_strict"]["stale_error"] for row in qwen), "n": len(qwen)}
    if main_old != {"exact_success": 142, "grounded_success": 135, "stale_error": 0, "n": 220}:
        raise SystemExit("refuse: original main strict totals changed")
    if qwen_old != {"exact_success": 15, "grounded_success": 15, "stale_error": 0, "n": 24}:
        raise SystemExit("refuse: original Qwen strict totals changed")
    by_arm_scale = []
    for arm in MAIN_ARMS:
        for scale in (4, 8, 12):
            rows = [row for row in main_six if row["arm"] == arm and row["scale"] == scale]
            by_arm_scale.append({"arm": arm, "scale": scale, "n": len(rows),
                                 "action_alias_value_match": rate(rows, "action_alias_value_match"),
                                 "minimal_evidence_sufficient": rate(rows, "minimal_evidence_sufficient"),
                                 "layered_success": rate(rows, "layered_success")})
    by_task = []
    for task_id in sorted({row["task_id"] for row in observations}):
        main_rows = [row for row in main if row["task_id"] == task_id]
        main_arm_rows = [row for row in main_rows if row["role"] == "main_arm"]
        grouped_rows = [row for row in main_rows if row["role"] == "control_grouped"]
        qwen_rows = [row for row in qwen if row["task_id"] == task_id]
        main_new = [row for row in main_arm_rows if row["new"]["status"] == "scored"]
        qwen_new = [row for row in qwen_rows if row["new"]["status"] == "scored"]
        by_task.append({"task_id": task_id, "policy_status": main_rows[0]["policy_status"],
                        "main_old": {"exact_success": sum(r["old_strict"]["exact_success"] for r in main_arm_rows),
                                     "grounded_success": sum(r["old_strict"]["grounded_success"] for r in main_arm_rows),
                                     "n": len(main_arm_rows)},
                        "grouped_old": {"exact_success": sum(r["old_strict"]["exact_success"] for r in grouped_rows),
                                        "grounded_success": sum(r["old_strict"]["grounded_success"] for r in grouped_rows),
                                        "n": len(grouped_rows)},
                        "main_new": {"n": len(main_new), **{k: rate(main_new, k) for k in
                                     ("action_alias_value_match", "minimal_evidence_sufficient", "layered_success")}},
                        "qwen_new": {"n": len(qwen_new), **{k: rate(qwen_new, k) for k in
                                     ("action_alias_value_match", "minimal_evidence_sufficient", "layered_success")}}})
    return {
        "main_220_original_strict": main_old,
        "qwen_struct_original_strict": qwen_old,
        "main_six_arms_valid": {"n": len(main_six),
            "action_alias_value_match": rate(main_six, "action_alias_value_match"),
            "minimal_evidence_sufficient": rate(main_six, "minimal_evidence_sufficient"),
            "layered_success": rate(main_six, "layered_success")},
        "main_control_grouped_valid": {"n": len(grouped), "role": "control_not_architecture",
            "action_alias_value_match": rate(grouped, "action_alias_value_match"),
            "minimal_evidence_sufficient": rate(grouped, "minimal_evidence_sufficient"),
            "layered_success": rate(grouped, "layered_success")},
        "qwen_struct_valid": {"n": len(qwen_valid),
            "action_alias_value_match": rate(qwen_valid, "action_alias_value_match"),
            "minimal_evidence_sufficient": rate(qwen_valid, "minimal_evidence_sufficient"),
            "layered_success": rate(qwen_valid, "layered_success")},
        "ambiguous_unknown": {"main": sum(1 for row in main if row["new"]["status"] == "unknown"),
                              "qwen": sum(1 for row in qwen if row["new"]["status"] == "unknown"),
                              "new_denominator": None},
        "by_arm_scale": by_arm_scale,
        "by_task": by_task,
        "cohort_note": "DS structure 24 is already inside main 220; Qwen structure 24 is separate; no pooling.",
        "grouped_note": "A-grouped 4 is control identity, not an independent architecture.",
        "metric_limits": NEW_METRIC_NOTE,
        "original_metric_note": ORIGINAL_NOTE}


def input_hashes() -> dict:
    paths = {"policy": POLICY, "source_input": SOURCE_INPUT, "source_raw": SOURCE_RAW,
             "source_aliases": SOURCE_ALIASES, "scored_all_220": SCORED, "disclosures": DISCLOSURES,
             "paired_results": PAIRED, "id_map": ID_MAP, "groups": GROUPS, "tasks": TASKS}
    return {name: D.file_sha256(path) for name, path in paths.items()}


def publish(path: Path, value: dict) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"refuse: existing output differs: {path}")
    else:
        path.write_text(text, encoding="utf-8", newline="\n")
    return D.file_sha256(path)


def score() -> dict:
    ctx = load_context()
    observations = build_observations(ctx)
    summary = summarize(observations)
    hashes = input_hashes()
    script_sha = D.file_sha256(Path(__file__))
    bundle_sha = D.sha256({"inputs": hashes, "script_sha256": script_sha})
    output = OUTDIR / f"measurement-correction-{bundle_sha[:16]}.json"
    result = {
        "schema": "measurement-correction/1",
        "status": "posthoc_diagnostic_not_preregistered_confirmatory_score",
        "new_metric": NEW_METRIC,
        "new_metric_note": NEW_METRIC_NOTE,
        "original_metric_note": ORIGINAL_NOTE,
        "policy": {"path": str(POLICY.relative_to(HERE)), "sha256": hashes["policy"],
                   "reference_kind": ctx["policy"].get("reference_kind"),
                   "status": ctx["policy"].get("status")},
        "inputs": hashes,
        "script_sha256": script_sha,
        "bundle_sha256": bundle_sha,
        "summary": summary,
        "observations": observations,
    }
    output_sha = publish(output, result)
    return {"ok": True, "output": str(output.relative_to(HERE)), "output_sha256": output_sha,
            "bundle_sha256": bundle_sha, "script_sha256": script_sha,
            "policy_sha256": hashes["policy"], "new_model_calls": 0,
            "summary": {k: summary[k] for k in ("main_220_original_strict", "main_six_arms_valid",
                                                "main_control_grouped_valid", "qwen_struct_valid",
                                                "ambiguous_unknown")}}


def _find(observations, source, task_id, arm, scale):
    for row in observations:
        if (row["source"], row["task_id"], row["arm"], row["scale"]) == (source, task_id, arm, scale):
            return row
    raise AssertionError(f"missing observation {source}/{task_id}/{arm}/{scale}")


def selftest() -> dict:
    ctx = load_context()
    observations = build_observations(ctx)
    checks = []

    def ck(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    current = _find(observations, "main_ds", "F06-T2", "A-full", 12)
    ck("current_atom_sufficient_without_old_history",
       current["old_strict"]["grounded_success"] is False and current["new"]["action_alias_value_match"] is True
       and current["new"]["minimal_evidence_sufficient"] is True and current["new"]["layered_success"] is True)
    disc = ctx["disc_by_id"][current["row_id"]]
    policy_task = ctx["policy_by_task"]["F06-T2"]
    no_current_citation = evaluate({**current["answer"], "cited_item_ids": []}, disc, policy_task, ctx["public_facts"])
    ck("missing_current_citation_fails",
       no_current_citation["action_alias_value_match"] is True
       and no_current_citation["minimal_evidence_sufficient"] is False)
    compound = _find(observations, "main_ds", "F01-T2", "A-compiled", 12)
    ck("compound_single_citation_covers_multiple_facts",
       compound["new"]["action_alias_value_match"] is True
       and compound["new"]["minimal_evidence_sufficient"] is True
       and compound["new"]["layered_success"] is True)
    multi = _find(observations, "main_ds", "F01-T2", "A-full", 12)
    first_atom = multi["answer"]["cited_item_ids"][0]
    one_atom = evaluate({**multi["answer"], "cited_item_ids": [first_atom]},
                        ctx["disc_by_id"][multi["row_id"]], ctx["policy_by_task"]["F01-T2"], ctx["public_facts"])
    ck("real_multi_obligation_missing_atom_fails",
       one_atom["action_alias_value_match"] is True and one_atom["minimal_evidence_sufficient"] is False)
    f04 = _find(observations, "main_ds", "F04-T2", "A-full", 12)
    f04_disc = ctx["disc_by_id"][f04["row_id"]]
    f04_policy = ctx["policy_by_task"]["F04-T2"]
    wrong_decision = evaluate({**f04["answer"], "decision": "allow"}, f04_disc, f04_policy, ctx["public_facts"])
    wrong_value = evaluate({**f04["answer"], "value": "覆盖"}, f04_disc, f04_policy, ctx["public_facts"])
    ck("wrong_decision_not_passed_by_keywords", wrong_decision["action_alias_value_match"] is False)
    ck("wrong_value_not_passed_by_keywords", wrong_value["action_alias_value_match"] is False)
    n1 = _find(observations, "main_ds", "N1", "A-full", 12)
    n1_disc = ctx["disc_by_id"][n1["row_id"]]
    n1_cited = evaluate({**n1["answer"], "cited_item_ids": [next(iter(S.disclosed_items(n1_disc)))]},
                        n1_disc, ctx["policy_by_task"]["N1"], ctx["public_facts"])
    ck("no_memory_requires_empty_citations",
       n1_cited["action_alias_value_match"] is True and n1_cited["minimal_evidence_sufficient"] is False)
    f01 = _find(observations, "main_ds", "F01-T1", "A-full", 12)
    invalid = evaluate({**f01["answer"], "cited_item_ids": ["m-not-disclosed"]},
                       ctx["disc_by_id"][f01["row_id"]], ctx["policy_by_task"]["F01-T1"], ctx["public_facts"])
    ck("invalid_citation_fails", invalid["minimal_evidence_sufficient"] is False)
    ambiguous = _find(observations, "main_ds", "F03-T2", "P-full", 12)
    ck("ambiguous_uniform_unknown",
       ambiguous["new"]["status"] == "unknown" and ambiguous["new"]["action_alias_value_match"] is None
       and ambiguous["new"]["minimal_evidence_sufficient"] is None and ambiguous["new"]["layered_success"] is None
       and ambiguous["old_strict"]["exact_success"] is False)
    condition_ignored = _find(observations, "main_ds", "F06-T1", "P-full", 12)
    ck("condition_field_ignored_in_action_value_projection",
       condition_ignored["answer"].get("applied_condition") is not None
       and condition_ignored["new"]["action_alias_value_match"] is True)
    summary = summarize(observations)
    ck("denominators_valid_198_control_4_qwen_22_unknown_20",
       summary["main_six_arms_valid"]["n"] == 198 and summary["main_control_grouped_valid"]["n"] == 4
       and summary["qwen_struct_valid"]["n"] == 22
       and summary["ambiguous_unknown"]["main"] == 18 and summary["ambiguous_unknown"]["qwen"] == 2)
    ck("grouped_marked_control_not_architecture", summary["main_control_grouped_valid"]["role"] == "control_not_architecture")
    ck("old_strict_unchanged",
       summary["main_220_original_strict"] == {"exact_success": 142, "grounded_success": 135,
                                               "stale_error": 0, "n": 220})
    result = {"ok": True, "checks": checks, "passed": len(checks), "llm_called": False,
              "policy_sha256": D.file_sha256(POLICY), "script_sha256": D.file_sha256(Path(__file__))}
    selftest_path = OUTDIR / "_selftest" / f"measurement-correction-selftest-{result['script_sha256'][:8]}.json"
    publish(selftest_path, result)
    result["selftest_path"] = str(selftest_path.relative_to(HERE))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score", "selftest"), nargs="?", default="score")
    args = parser.parse_args()
    result = selftest() if args.command == "selftest" else score()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
