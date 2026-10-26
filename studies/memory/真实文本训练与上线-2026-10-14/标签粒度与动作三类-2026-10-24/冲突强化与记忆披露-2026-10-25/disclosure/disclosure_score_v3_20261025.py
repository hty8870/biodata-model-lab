                      
"""v3 评分：grounded_success 主指标 + exact 诊断 + 双 recall + 基线。

grounded = 语义正确 ∧ 每个 required/necessary fact 的承载条目既被披露又被 cited
           ∧ 特殊规则（supersede 当前原子、simultaneous 两侧、conditional 应用原子、
             multi 每个义务、source 逐条）。
cited 只接受本行披露的 items[].id；source/link id 记 rejected_source_id。
命令：selftest / analyze [--synthetic] [--answers path]
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from disclosure_schema_20261025 import (              
    DATA, OUT, json_write, jsonl_read, normalize_value, parse_answer, project_full, sha256,
)

ARMS = ("P-full", "A-full", "P-retrieve", "A-retrieve", "A-graph", "A-compiled", "A-grouped")
PAIRS = (("A-full", "A-grouped"), ("P-full", "A-grouped"), ("A-full", "P-full"),
         ("A-full", "A-retrieve"), ("P-retrieve", "A-retrieve"),
         ("A-retrieve", "A-graph"), ("A-retrieve", "A-compiled"))
CONTRAST_LABELS = {
    ("A-full", "A-grouped"): "A-full - A-grouped（平铺呈现效应）",
    ("P-full", "A-grouped"): "P-full - A-grouped（逐字控制，应恒 0）",
    ("A-full", "P-full"): "A-full - P-full（全量呈现总效应，非纯存储）",
}
RETRIEVE_ARMS = ("P-retrieve", "A-retrieve", "A-graph")
SEED = 20261025
BOOTSTRAP = 1000
NEG_EQUIV = {
    "拒绝覆盖": {"拒绝覆盖", "不覆盖", "不得覆盖", "拒绝覆盖同名文件", "拒绝覆盖并换名"},
    "只导出一次": {"只导出一次", "仅导出一次", "去重后导出", "只导出一次引文"},
    "不得引用": {"不得引用", "禁止引用", "不引用", "不得引用内部批次号"},
    "不得静默重建": {"不得静默重建", "禁止静默重建", "不静默重建"},
    "不得包含原始数据": {"不得包含原始数据", "禁止包含原始数据", "不包含原始数据"},
    "不得填充": {"不得填充", "不填充", "禁止填充"},
    "禁止物理删除": {"禁止物理删除", "不得物理删除", "物理删除被禁止"},
}


def _norm(value):
    if isinstance(value, list):
        return [_norm(v) for v in value]
    if value is None:
        return None
    text = normalize_value(value) if isinstance(value, str) else value
    return str(text).strip().strip('"\'“”‘’')


def _value_ok(expected, answer, mode):
    if expected is None:
        return answer in (None, "", [])
    if mode == "ordered":
        return isinstance(answer, list) and _norm(answer) == _norm(expected)
    if isinstance(expected, str) and expected in NEG_EQUIV:
        got = _norm(answer) or ""
        return any(_norm(v) in got or got in _norm(v) for v in NEG_EQUIV[expected])
    return _norm(answer) == _norm(expected)


def _scope_ok(expected, answer):
    if expected is None:
        return answer in (None, "", [])
    if not isinstance(answer, str) or not answer.strip():
        return False
    a, e = _norm(answer), _norm(expected)
    return bool(a) and (a in e or e in a)


def load():
    groups = jsonl_read(DATA / "groups.jsonl")
    tasks = {t["task_id"]: t for t in jsonl_read(DATA / "tasks.jsonl")}
    id_map = json.loads((DATA / "id_map.json").read_text(encoding="utf-8"))
    return groups, tasks, id_map


def full_row(task, group, version, scale=48, position="start"):
    items = [group["compound"]] if version == "compound" else list(group["atoms"])
    public = id_map_global["items"]
    row_items = [{**it, "id": public[it["id"]], "links": []} for it in items]
    internal = [it["id"] for it in items]
    fact_ids = {}
    for it, original in zip(row_items, items):
        if version == "compound":
            fact_ids[it["id"]] = [f["fact_id"] for f in group["facts"]]
        else:
            fact_ids[it["id"]] = [f["fact_id"] for f in group["facts"]
                                  if f["atom_index"] == items.index(original)]
    groups_map = {it["id"]: group["group_id"] for it in row_items}
    def carriers(fact_ids):
        out = set()
        for fid in fact_ids:
            fact = next(f for f in group["facts"] if f["fact_id"] == fid)
            internal_item = (group["compound"]["id"] if version == "compound"
                             else group["atoms"][fact["atom_index"]]["id"])
            out.add(public[internal_item])
        return sorted(out)
    required_all = carriers(task["gold"]["required_facts"])
    context_all = carriers(task["gold"]["necessary_context"])
    block = project_full(row_items, "decide")
    return {"row_id": f"{task['task_id']}__{version}__S{scale}__main",
            "task_id": task["task_id"], "family_id": task["family_id"],
            "split": "dev" if task["family_id"] in ("F01", "F02", "F03", "F04") else "final",
            "phenomenon": task["phenomenon"], "arm": version, "variant": "main",
            "scale": scale, "position": position, "item_ids": [r["id"] for r in row_items],
            "item_internal_ids": internal, "item_fact_ids": fact_ids,
            "item_internal_to_public": {i: r["id"] for i, r in zip(internal, row_items)},
            "item_group_ids": groups_map, "required_items_all": required_all,
            "context_items_all": context_all, "disclosure": block, "chars": len(block),
            "token_estimate": 0, "build_latency_ms": 0.0,
            "budget_chars": None, "snapshot_ok": True, "real_arm_result": False,
            "observation_kind": "independent"}


def ideal_answer(row, task):
    public_to_internal = id_map_global["reverse_items"]
    disclosed = list(row["item_ids"])
    required = set(task["gold"]["required_facts"]) | set(task["gold"]["necessary_context"])
    carriers = {iid for iid in disclosed
                if set(row["item_fact_ids"].get(iid, [])) & required}
    expected = task["expected"]
    return {"decision": expected["decision"], "value": expected["value"],
            "applied_condition": expected["applied_condition"],
            "applied_exception": expected["applied_exception"],
            "cited_item_ids": sorted(carriers)}


def score_answer(value, row, task):
    if isinstance(value, dict):
        answer, reason = dict(value), "ok"
    else:
        answer, reason = parse_answer(value if isinstance(value, str) else "")
    parse_ok = answer is not None
    answer = answer or {}
    expected = task["expected"]
    cited = [c for c in (answer.get("cited_item_ids") or []) if isinstance(c, str)]
    disclosed = set(row["item_ids"])
    source_ids = set(id_map_global["sources"].values())
    rejected = [c for c in cited if c not in disclosed]
    source_cited = [c for c in rejected if c in source_ids]
    valid = [c for c in cited if c in disclosed]
    public_to_internal = id_map_global["reverse_items"]
    cited_internal = {public_to_internal[c] for c in valid if c in public_to_internal}
    disclosed_internal = set(row["item_internal_ids"])
    required = set(task["gold"]["required_facts"])
    context = set(task["gold"]["necessary_context"])
    required_items = set(row.get("required_items_all") or [])
    context_items = set(row.get("context_items_all") or [])
    required_groups = set(task["gold"]["required_groups"])
    disclosed_groups = {row["item_group_ids"][i] for i in row["item_ids"]
                        if i in row["item_group_ids"]}
    group_recall = len(required_groups & disclosed_groups) / len(required_groups) \
        if required_groups else None
    disclosed_facts = set()
    for internal_id, public_id in row.get("item_internal_to_public", {}).items():
        if internal_id in disclosed_internal:
            disclosed_facts |= set(row["item_fact_ids"].get(public_id, []))
    fact_recall = len(required & disclosed_facts) / len(required) if required else None
    exact = (parse_ok and answer.get("decision") == expected["decision"]
             and _value_ok(expected["value"], answer.get("value"),
                           expected.get("value_mode", "exact"))
             and _scope_ok(expected["applied_condition"], answer.get("applied_condition"))
             and _scope_ok(expected["applied_exception"], answer.get("applied_exception")))
    grounded = (exact
                and required_items <= disclosed
                and required_items <= set(valid)
                and context_items <= disclosed
                and context_items <= set(valid)
                and _fact_checks_ok(task, answer, group_of_task(task), row))
    return {"parse_ok": parse_ok, "parse_reason": reason,
            "decision": answer.get("decision"), "value": answer.get("value"),
            "applied_condition": answer.get("applied_condition"),
            "applied_exception": answer.get("applied_exception"),
            "exact_success": bool(exact), "grounded_success": bool(grounded),
            "required_items": sorted(required_items),
            "context_items": sorted(context_items),
            "cited_internal": sorted(cited_internal),
            "missing_required_items": sorted(required_items - disclosed),
            "uncited_required_items": sorted(required_items - set(valid)),
            "rejected_citation_ids": sorted(rejected),
            "source_id_citations": sorted(source_cited),
            "group_recall": group_recall, "fact_recall": fact_recall,
            "irrelevant_logical_groups": len(disclosed_groups - required_groups),
            "stale_error": bool(task["gold"]["forbidden_facts"]
                                and any(_norm(answer.get("value")) == _norm(
                                    fact_value(task, fid)) for fid in task["gold"]["forbidden_facts"])),
            "hallucination": bool(task["phenomenon"] == "no_relevant"
                                  and (answer.get("decision") != "no_memory"
                                       or answer.get("value") not in (None, "", [])))}


def fact_value(task, fid):
    for f in GROUPS[task["target_group"]]["facts"]:
        if f["fact_id"] == fid:
            return f["value"]
    return None


def group_of_task(task):
    return GROUPS.get(task["target_group"]) if task["target_group"] else None


def _fact_checks_ok(task, answer, group, row):
    if group is None:
        return True
    for fid, spec in task["gold"]["fact_checks"].items():
        field = spec["field"]
        got = answer.get(field)
        if "equals" in spec:
            want = spec["equals"]
            if field == "value":
                ok = _value_ok(want, got, task["expected"].get("value_mode", "exact"))
            elif field in ("applied_condition", "applied_exception"):
                ok = _scope_ok(want, got)
            else:
                ok = got == want
        elif "not_equals" in spec:
            ok = _norm(got) != _norm(spec["not_equals"])
        else:
            ok = True
        if not ok:
            return False
    return True


                                                              

def baseline_no_memory(row, task):
    return {"decision": "no_memory", "value": None, "cited_item_ids": [],
            "applied_condition": None, "applied_exception": None}


def baseline_constant_allow(row, task):
    return {"decision": "allow", "value": None, "cited_item_ids": [],
            "applied_condition": None, "applied_exception": None}


def baseline_single_item(row, task):
    """金标字段 + 只引用一个必需条目：应 exact=true、grounded=false（多义务/冲突/条件）。"""
    answer = ideal_answer(row, task)
    answer["cited_item_ids"] = answer["cited_item_ids"][:1]
    return answer


                                                                    

def selftest() -> dict:
    global GROUPS, id_map_global
    GROUPS = {g["group_id"]: g for g in jsonl_read(DATA / "groups.jsonl")}
    tasks = jsonl_read(DATA / "tasks.jsonl")
    id_map_global = json.loads((DATA / "id_map.json").read_text(encoding="utf-8"))
    results = {}
    checked = 0
    for task in tasks:
        if not task["target_group"]:
            continue
        group = GROUPS[task["target_group"]]
        row = full_row(task, group, "compound")
        ideal = ideal_answer(row, task)
        scored = score_answer(ideal, row, task)
        assert scored["exact_success"], (task["task_id"], "ideal exact")
        assert scored["grounded_success"], (task["task_id"], "ideal grounded")
        assert scored["fact_recall"] == 1.0, task["task_id"]
        assert scored["group_recall"] == 1.0, task["task_id"]
        checked += 1
    results["ideal_all_grounded"] = checked

    multi = next(t for t in tasks if t["phenomenon"] == "multi_obligation")
    group = GROUPS[multi["target_group"]]
                                        
    row = full_row(multi, group, "atomic")
    one = baseline_single_item(row, multi)
    s = score_answer(one, row, multi)
    assert s["exact_success"] and not s["grounded_success"] and s["uncited_required_items"]
    results["single_item_not_grounded"] = True
    sim = next(t for t in tasks if t["phenomenon"] == "conflict_update"
               and GROUPS[t["target_group"]]["mode"] == "simultaneous")
    srow = full_row(sim, GROUPS[sim["target_group"]], "atomic")
    assert not score_answer(baseline_single_item(srow, sim), srow, sim)["grounded_success"]
    results["simultaneous_single_item_not_grounded"] = True

    empty_row = dict(row, item_ids=[], item_internal_ids=[], item_fact_ids={},
                     item_group_ids={}, disclosure=project_full([], "decide"), chars=0)
    s = score_answer(ideal_answer(row, multi), empty_row, multi)
    assert not s["grounded_success"] and s["group_recall"] == 0.0
    results["no_target_not_grounded"] = True

    source_id = id_map_global["sources"][group["sources"][0]["source_id"]]
    bad = dict(ideal_answer(row, multi), cited_item_ids=[source_id])
    s = score_answer(bad, row, multi)
    assert not s["grounded_success"] and s["source_id_citations"] == [source_id]
    results["source_id_rejected"] = True

    neg = next(t for t in tasks if t["phenomenon"] == "negation")
    grow = full_row(neg, GROUPS[neg["target_group"]], "compound")
    good = dict(ideal_answer(grow, neg), value=neg["expected"]["value"])
    assert score_answer(good, grow, neg)["grounded_success"]
    bad = dict(good, value="可以覆盖")
    assert not score_answer(bad, grow, neg)["exact_success"]
    results["negation_normalize_positive_negative"] = True

    for task in tasks:
        if not task["target_group"]:
            continue
        grow = full_row(task, GROUPS[task["target_group"]], "compound")
        assert not score_answer(baseline_no_memory(grow, task), grow, task)["grounded_success"]
        assert not score_answer(baseline_constant_allow(grow, task), grow, task)["grounded_success"]
    for task in [t for t in tasks if t["phenomenon"] == "no_relevant"]:
        row = full_row(task, GROUPS["F01-G1"], "compound")
        row.update({"item_ids": [], "item_internal_ids": [], "item_fact_ids": {},
                    "item_group_ids": {}, "required_items_all": [], "context_items_all": []})
        s0 = score_answer({"decision": "no_memory", "value": None, "cited_item_ids": [],
                           "applied_condition": None, "applied_exception": None}, row, task)
        assert s0["exact_success"] and s0["grounded_success"]
    results["no_rel_ideal"] = True
    results["baselines_fail"] = True
    return {"ok": True, "results": results,
            "tasks_checked": checked,
            "grounded_denominator_note": "grounded 只在完整披露+完整引用时为 true"}



def analyze(answers_path=None, synthetic=False):
    groups, tasks, id_map = load()
    global GROUPS, id_map_global
    GROUPS = {g["group_id"]: g for g in groups}
    id_map_global = id_map
    rows = jsonl_read(OUT / "disclosures.jsonl")
    answers = {}
    if synthetic:
        for row in rows:
            task = tasks[row["task_id"]]
            answers[row["row_id"]] = ideal_answer(row, task)
    elif answers_path and Path(answers_path).exists():
        for a in jsonl_read(Path(answers_path)):
            answers[a["row_id"]] = a.get("raw_text") or a.get("answer")
    scored = []
    baseline_hits = {"no_memory": {}, "constant_allow": {}, "single_item": {}}
    for name in baseline_hits:
        baseline_hits[name] = defaultdict(int)
    audit_a8 = []
    for row in rows:
        task = tasks[row["task_id"]]
        ans = answers.get(row["row_id"])
        if ans is not None:
            s = score_answer(ans, row, task)
            s.update({"arm": row["arm"], "split": row["split"], "scale": row["scale"],
                      "chars": row["chars"]})
            scored.append(s)
        if row["arm"] in ("P-full", "A-full", "A-grouped"):
            if not set(row.get("required_items_all") or []) <= set(row["item_ids"]):
                audit_a8.append(row["row_id"])
        if task["phenomenon"] != "no_relevant":
            ph = task["phenomenon"]
            if score_answer(baseline_no_memory(row, task), row, task)["grounded_success"]:
                baseline_hits["no_memory"][ph] += 1
            if score_answer(baseline_constant_allow(row, task), row, task)["grounded_success"]:
                baseline_hits["constant_allow"][ph] += 1
            if row["arm"] == "A-full" and score_answer(
                    baseline_single_item(row, task), row, task)["grounded_success"]:
                baseline_hits["single_item"][ph] += 1
    summary = {}
    for s in scored:
        key = (s["arm"], s["scale"])
        cell = summary.setdefault(key, {"n": 0, "exact": 0, "grounded": 0,
                                        "group": [], "fact": [], "irrelevant": []})
        cell["n"] += 1
        cell["exact"] += int(s["exact_success"])
        cell["grounded"] += int(s["grounded_success"])
        if s["group_recall"] is not None:
            cell["group"].append(s["group_recall"])
        if s["fact_recall"] is not None:
            cell["fact"].append(s["fact_recall"])
        cell["irrelevant"].append(s["irrelevant_logical_groups"])
    table = [{"arm": k[0], "scale": k[1], "n": v["n"],
              "exact_rate": round(v["exact"] / v["n"], 4),
              "grounded_rate": round(v["grounded"] / v["n"], 4),
              "group_recall_mean": round(statistics.fmean(v["group"]), 4) if v["group"] else None,
              "fact_recall_mean": round(statistics.fmean(v["fact"]), 4) if v["fact"] else None,
              "irrelevant_logical_groups_mean": round(statistics.fmean(v["irrelevant"]), 2)}
             for k, v in sorted(summary.items())]
    result = {"schema": "v3-analysis", "synthetic": synthetic,
              "answers": len(scored), "rows": len(rows),
              "audit_a8_errors": audit_a8[:20], "audit_a8_ok": not audit_a8,
              "baseline_grounded_hits": {k: dict(v) for k, v in baseline_hits.items()},
              "summary": table, "llm_called": False,
              "note": "grounded 只在完整披露+完整引用时为 true；exact 仅语义诊断。"}
    json_write(OUT / ("analysis_synthetic.json" if synthetic else "analysis.json"), result)
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "selftest":
        print(json.dumps(selftest(), ensure_ascii=False, indent=2))
    elif sys.argv[1] == "analyze":
        synthetic = "--synthetic" in sys.argv
        path = None
        if "--answers" in sys.argv:
            path = sys.argv[sys.argv.index("--answers") + 1]
        print(json.dumps(analyze(path, synthetic=synthetic), ensure_ascii=False, indent=2))
