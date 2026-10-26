                      
"""Strict JSON answers, finite aliases, group/fact/evidence coverage and stale diagnostics."""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from disclosure_schema_20261025 import SAFETY_PREFIX, json_write, jsonl_read, project_full, sha256
from pilot_data_20261025 import DATA, EXP, LIMITATIONS, SCOPE_ALIASES, VALUE_ALIASES

OUT = EXP / "out"
ANSWER_KEYS = {"decision", "value", "applied_condition", "applied_exception", "cited_item_ids"}
DECISIONS = ("allow", "deny", "ask_user", "no_memory")


def load():
    return ({g["group_id"]: g for g in jsonl_read(DATA / "groups.jsonl")},
            {t["task_id"]: t for t in jsonl_read(DATA / "tasks.jsonl")},
            json.loads((DATA / "id_map.json").read_text(encoding="utf-8")))


def norm(value):
    return re.sub(r"\s+", "", value) if isinstance(value, str) else value


def parse(answer):
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except (json.JSONDecodeError, ValueError):
            return {}, "not_json_only"
    if not isinstance(answer, dict) or set(answer) != ANSWER_KEYS:
        return {}, "answer_schema"
    if answer["decision"] not in DECISIONS:
        return {}, "decision_enum"
    if answer["decision"] in ("ask_user", "no_memory") and answer["value"] is not None:
        return {}, "null_required"
    if not (answer["value"] is None or isinstance(answer["value"], str) or
            isinstance(answer["value"], list) and all(isinstance(v, str) for v in answer["value"])):
        return {}, "value_type"
    if any(answer[k] is not None and not isinstance(answer[k], str) for k in ("applied_condition", "applied_exception")):
        return {}, "scope_type"
    cited = answer["cited_item_ids"]
    if not isinstance(cited, list) or not all(isinstance(c, str) for c in cited) or len(set(cited)) != len(cited):
        return {}, "citation_type_or_duplicate"
    return answer, "ok"


def value_ok(expected, got, mode="exact"):
    if expected is None:
        return got is None
    if mode in ("set", "ordered"):
        if not isinstance(got, list) or len(got) != len(expected):
            return False
        aliases = {norm(v): norm(e) for e in expected for v in [e] + VALUE_ALIASES.get(e, [])}
        values = [aliases.get(norm(v)) for v in got]
        if None in values or len(set(values)) != len(values):
            return False
        return set(map(norm, expected)) == set(values) if mode == "set" else list(map(norm, expected)) == values
    if not isinstance(got, str):
        return False
    return norm(got) in {norm(v) for v in [expected] + VALUE_ALIASES.get(expected, [])}


def scope_ok(expected, got):
    if expected is None:
        return got is None
    return isinstance(got, str) and norm(got) in {norm(v) for v in [expected] + SCOPE_ALIASES.get(expected, [])}


def disclosed_items(row):
    try:
        value = json.loads(row["disclosure"][len(SAFETY_PREFIX):])
        if not row["disclosure"].startswith(SAFETY_PREFIX):
            return {}
        return {i["id"]: i for i in value["items"] if i["id"] in row["item_ids"]}
    except (KeyError, TypeError, ValueError):
        return {}


def support(row, items, ids):
    facts, evidence, groups = set(), set(), set()
    for iid in ids:
        item = items.get(iid, {})
        text = item.get("text", "")
        info = row.get("coverage", {}).get(iid, {})
        if not text or sha256(text) != info.get("text_sha256"):
            continue
        accepted = []
        for f in info.get("facts", []):
            literal = f.get("probe", "") and f["probe"] in text
            reviewed = (info.get("semantic_verified") is True and bool(info.get("publication_sha256")))
            if not (literal or reviewed):
                continue
            accepted.append(f["fact_id"])
            (evidence if f["type"] == "source" else facts).add(f["fact_id"])
        if accepted:
            groups.add(info["group_id"])
    return facts, evidence, groups


def fraction(need, have):
    return len(need & have) / len(need) if need else 1.0


def score(answer, row, task):
    a, parse_reason = parse(answer)
    exp, gold = task["expected"], task["gold"]
    cited = a.get("cited_item_ids", [])
    items = disclosed_items(row)
    invalid = sorted(set(cited) - set(items))
    valid = set(cited) & set(items)
    semantic = bool(a) and a.get("decision") == exp["decision"] and value_ok(exp["value"], a.get("value"), exp["value_mode"]) and all(scope_ok(exp[k], a.get(k)) for k in ("applied_condition", "applied_exception"))
    nf = set(gold["required_facts"] + gold["necessary_context"])
    source_ids = {f["fact_id"] for info in row.get("coverage", {}).values() for f in info.get("facts", []) if f["type"] == "source"}
                                                                                                            
    source_ids |= {fid for fid, typ in row.get("fact_types", {}).items() if typ == "source"}
    ne, ns = nf & source_ids, nf - source_ids
    ng = set(gold["required_groups"])
    df, de, dg = support(row, items, set(items))
    cf, ce, cg = support(row, items, valid)
    forbidden_values = {norm(v) for v in row.get("forbidden_answer_values", [])}
    stale_value = task.get("stale_value")
    if stale_value is not None:
        forbidden_values.add(norm(stale_value))
    values = a.get("value") if isinstance(a.get("value"), list) else [a.get("value")]
    stale_error = bool(a) and any(norm(v) in forbidden_values for v in values if isinstance(v, str))
    grounded_fact = semantic and not invalid and not stale_error and ns <= cf and ne <= ce and ng <= cg
    if exp["decision"] == "no_memory":
        grounded_fact = semantic and not cited
    return {
        "parse_reason": parse_reason, "exact_success": bool(semantic), "grounded_success": bool(grounded_fact),
        "grounded_fact": bool(grounded_fact),
        "grounded_group": bool(semantic and not invalid and not stale_error and ng <= cg and (ng or not cited)),
        "stale_error": stale_error, "forbidden_answer_fact_ids": gold["forbidden_facts"] if stale_error else [],
        "invalid_citations": invalid, "cited": cited,
        "recall_group": fraction(ng, dg), "recall_fact": fraction(ns, df), "recall_evidence": fraction(ne, de),
        "citation_group_coverage": fraction(ng, cg), "citation_fact_coverage": fraction(ns, cf), "citation_evidence_coverage": fraction(ne, ce),
        "required_groups": sorted(ng), "missing_groups": sorted(ng - cg),
        "missing_facts": sorted(ns - cf), "missing_evidence": sorted(ne - ce),
        "irrelevant_logical_groups": len(dg - ng), "citation_count": len(valid),
        "required_fact_count": len(ns), "required_evidence_count": len(ne),
    }


def coverage_for(group, item, version, id_map, *, semantic_verified=False, publication_sha256=None):
    facts = group["facts"] if version in ("compound", "grouped", "compiled") else [f for f in group["facts"] if f["fact_id"] in group["atomic_fact_ids"][item["id"]]]
    return {"group_id": group["group_id"], "facts": facts, "text_sha256": sha256(item["text"]),
            "semantic_verified": semantic_verified, "publication_sha256": publication_sha256}


def full_rows(task, group, id_map, version):
    its = [group["compound"]] if version == "compound" else group["atoms"]
    items = [{**i, "id": id_map["items"][i["id"]]} for i in its]
    cov = {p["id"]: coverage_for(group, i, version, id_map) for i, p in zip(its, items)}
    forbidden = [f["value"] for f in group["facts"] if f["fact_id"] in task["gold"]["forbidden_facts"]]
    return dict(row_id=f"{task['task_id']}__{version}", task_id=task["task_id"], arm=version,
                item_ids=[i["id"] for i in items], disclosure=project_full(items, "decide"), coverage=cov,
                fact_types={f["fact_id"]: f["type"] for f in group["facts"]}, forbidden_answer_values=forbidden)


def ideal(row, task):
    nf = set(task["gold"]["required_facts"] + task["gold"]["necessary_context"])
    cited = [iid for iid, info in row["coverage"].items() if nf & {f["fact_id"] for f in info["facts"]}]
    return {k: task["expected"][k] for k in ("decision", "value", "applied_condition", "applied_exception")} | {"cited_item_ids": cited}


def selftest():
    groups, tasks, ids = load()
    checks = []
    def ck(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)
    for task in tasks.values():
        if not task["target_group"]:
            row = dict(item_ids=[], disclosure=project_full([], "decide"), coverage={})
            ck(task["task_id"]+":no_memory_pass", score(ideal(row, task), row, task)["grounded_success"])
            continue
        for version in ("compound", "atomic"):
            row = full_rows(task, groups[task["target_group"]], ids, version)
            answer = ideal(row, task)
            ck(task["task_id"]+":"+version+":ideal", score(answer, row, task)["grounded_success"])
            ck(task["task_id"]+":"+version+":no_citation", not score(answer | {"cited_item_ids": []}, row, task)["grounded_success"])
            ck(task["task_id"]+":"+version+":no_memory", not score(dict(decision="no_memory", value=None, applied_condition=None, applied_exception=None, cited_item_ids=[]), row, task)["grounded_success"])
            empty = copy.deepcopy(row)
            empty["disclosure"] = project_full([{**i, "text": ""} for i in disclosed_items(row).values()], "decide")
            ck(task["task_id"]+":"+version+":metadata_only", not score(answer, empty, task)["grounded_success"])
            if version == "atomic" and len(answer["cited_item_ids"]) > 1:
                for iid in answer["cited_item_ids"]:
                    sc = score(answer | {"cited_item_ids": [iid]}, row, task)
                    ck(task["task_id"]+":single_atom:"+iid, sc["exact_success"] and not sc["grounded_fact"])
                    rest = [c for c in answer["cited_item_ids"] if c != iid]
                    ck(task["task_id"]+":every_obligation_needed:"+iid, not score(answer | {"cited_item_ids": rest}, row, task)["grounded_fact"])
            ck(task["task_id"]+":"+version+":bad_source_id", not score(answer | {"cited_item_ids": [next(iter(ids["sources"].values()))]}, row, task)["grounded_success"])
    task = tasks["F06-T2"]
    row = full_rows(task, groups["F06-G2"], ids, "compound")
    ck("stale_explicit", score(ideal(row, task) | {"value": "3 条"}, row, task)["stale_error"])
    ck("history_citation_is_legal", score(ideal(row, task), row, task)["grounded_success"])
    ck("failed_receipt_never_passes", not score_receipt(dict(text=json.dumps(ideal(row, task)), succeeded=False), row, task)["grounded_success"])
    wrong_group = copy.deepcopy(task)
    wrong_group["gold"]["required_groups"] = ["not_the_target_group"]
    ck("required_groups_not_dead_metadata", not score(ideal(row, task), row, wrong_group)["grounded_success"])
    no_stale_hint = task | {"stale_value": None}
    ck("forbidden_answer_values_not_dead_metadata", score(ideal(row, task) | {"value": "3 条"}, row, no_stale_hint)["stale_error"])
    for canonical, aliases in SCOPE_ALIASES.items():
        ck("alias:"+canonical, all(scope_ok(canonical, a) for a in aliases + [canonical]))
        for bad in ("", canonical[:2], "不"+canonical, canonical+"且用户为管理员", "无关内容"):
            ck("not_substring:"+canonical+":"+bad, not scope_ok(canonical, bad))
    ck("negation_finite", value_ok("不重复导出", "禁止重复导出") and not value_ok("不重复导出", "导出"))
    ck("null_not_empty", not value_ok(None, "") and not value_ok(None, []))
    ck("set_order_invariant", value_ok(["a", "b"], ["b", "a"], "set"))
    ck("ordered_not_invariant", not value_ok(["a", "b"], ["b", "a"], "ordered"))
    ck("set_no_duplicate", not value_ok(["a", "b"], ["a", "a", "b"], "set"))
    ck("channel_literal_label_alias", value_ok(["对话内提醒", "邮件"], ["电子邮件", "对话内发送"], "set"))
    ck("channel_alias_duplicates_rejected", not value_ok(["对话内提醒", "邮件"], ["对话内", "对话内发送"], "set"))
    ck("json_only", parse('前言 {"decision":"allow"}')[1] != "ok")
    ck("enum_closed", parse(dict(decision="yes", value=None, applied_condition=None, applied_exception=None, cited_item_ids=[]))[1] == "decision_enum")
    return dict(ok=True, passed=len(checks), checks=checks, llm_called=False)


def score_receipt(record, row, task):
    result = score(record.get("text", ""), row, task)
    result["response_succeeded"] = bool(record.get("succeeded"))
    if not record.get("succeeded"):
        for key in ("exact_success", "grounded_success", "grounded_fact", "grounded_group"):
            result[key] = False
    return result


def analyze(rows=None, responses=None, output=None):
    _, tasks, _ = load()
    rows = rows if rows is not None else jsonl_read(OUT / "disclosures.jsonl")
    synthetic = responses is None
    table = defaultdict(list)
    for row in rows:
        task = tasks[row["task_id"]]
        record = {} if synthetic else responses.get(row["row_id"], {})
        sc = score(ideal(row, task), row, task) if synthetic else score_receipt(record, row, task)
        table[(row["arm"], row["scale"])].append((sc, record, row))
    summary = []
    for (arm, scale), values in sorted(table.items()):
        n = len(values)
        cell = dict(arm=arm, scale=scale, n=n, placeholder_rows=sum(r.get("placeholder", False) for _, _, r in values))
        for key in ("exact_success", "grounded_success", "grounded_group", "grounded_fact", "recall_group", "recall_fact", "recall_evidence", "citation_fact_coverage", "citation_evidence_coverage", "stale_error"):
            cell[key] = round(sum(s[key] for s, _, _ in values) / n, 6)
        cell["failed_or_missing"] = 0 if synthetic else sum(not rec.get("succeeded") for _, rec, _ in values)
        cell["usage_total_tokens"] = sum((rec.get("usage") or {}).get("total_tokens") or 0 for _, rec, _ in values)
        cell["provider_usage_rows"] = sum((rec.get("usage") or {}).get("source") == "provider" for _, rec, _ in values)
        cell["prompt_chars_mean"] = round(sum(len(row.get("prompt", "")) for _, _, row in values) / n, 3)
        cell["disclosure_chars_mean"] = round(sum(len(row["disclosure"]) for _, _, row in values) / n, 3)
        latencies = [rec["latency_s"] for _, rec, _ in values if isinstance(rec.get("latency_s"), (int, float))]
        cell["latency_total_s"] = round(sum(latencies), 3) if latencies else None
        cell["latency_mean_recorded_s"] = round(sum(latencies) / len(latencies), 3) if latencies else None
        cell["real_behavior_rate"] = None if synthetic else cell["grounded_success"]
        summary.append(cell)
    result = dict(schema="pilot-v4-r2-analysis", synthetic=synthetic, rows=len(rows), summary=summary, limitations=LIMITATIONS,
                  note="Synthetic ideals test scorer/recall only, NEVER model behavior. Missing/failed calls remain in the denominator. Compound citations are legal but have lower citation burden.")
    json_write(output or OUT / "analysis_synthetic.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", nargs="?", choices=("selftest", "analyze"), default="selftest")
    a = p.parse_args()
    result = selftest() if a.command == "selftest" else analyze()
    if a.command == "selftest":
        json_write(OUT / "score_selftest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
