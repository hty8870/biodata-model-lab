                      
"""v3 七臂 dry-run：无选项 prompt、公开 ID、空 links、双 recall、A-grouped 逐字控制。

不调用 LLM。命令：selftest / dry-run / plan。
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from disclosure_schema_20261025 import (              
    DATA, OUT, PHASE, bigrams, estimate_tokens, item_cost, json_write, jsonl_read,
    jsonl_write, project_full, sha256,
)

ARMS = ("P-full", "A-full", "P-retrieve", "A-retrieve", "A-graph", "A-compiled", "A-grouped")
RETRIEVE = ("P-retrieve", "A-retrieve", "A-graph")
BUDGETS = (4000, 8000, 12000)
SCORER = "char_bigram_coverage_v1"
POSITIONS = ("start", "middle", "end")
CURRENT_TIME = "当前是 2026-10。"


def load():
    groups = jsonl_read(DATA / "groups.jsonl")
    tasks = jsonl_read(DATA / "tasks.jsonl")
    banks = jsonl_read(DATA / "banks.jsonl")
    id_map = json.loads((DATA / "id_map.json").read_text(encoding="utf-8"))
    return groups, tasks, banks, id_map


def public_item(item, id_map):
    row = {k: item[k] for k in ("id", "scope_level", "project_id", "kind", "title", "text",
                                "trigger", "tier", "permanence", "payload", "links", "revision")}
    row["id"] = id_map["items"][item["id"]]
    row["links"] = []
    return row


def version_items(group, version, id_map):
    if version in ("compound", "grouped"):
        internal = [group["compound"]]
    elif version == "compiled":
        internal = [{"id": f"C-{group['group_id']}",
                     "scope_level": "user", "project_id": "", "kind": "memory",
                     "title": group["slot"],
                     "text": "。".join(a["text"] for a in group["atoms"]) + "。",
                     "trigger": group["trigger"], "tier": "explicit",
                     "permanence": "persistent", "payload": {}, "links": [],
                     "revision": 1}]
    else:
        internal = list(group["atoms"])
    out = []
    for item in internal:
        if version == "compiled":
            row = {k: item[k] for k in ("scope_level", "project_id", "kind", "title", "text",
                                        "trigger", "tier", "permanence", "payload", "links",
                                        "revision")}
            row["id"] = "k-" + hashlib.sha256(
                ("v3card|" + group["group_id"]).encode()).hexdigest()[:10]
        else:
            row = public_item(item, id_map)
        out.append((row, item["id"]))
    return out


def fact_map(group, version, id_map, items):
    mapping = {}
    for row, internal in items:
        if version in ("compound", "grouped", "compiled"):
            facts = [f["fact_id"] for f in group["facts"]]
        else:
            idx = int(internal.rsplit("A", 1)[1]) - 1
            facts = [f["fact_id"] for f in group["facts"] if f["atom_index"] == idx]
        mapping[row["id"]] = facts
    return mapping


def carriers(task, group, version, id_map):
    def one(fid):
        f = next(x for x in group["facts"] if x["fact_id"] == fid)
        internal = group["compound"]["id"] if version in ("compound", "grouped") else \
            (group["atoms"][f["atom_index"]]["id"] if version == "atomic" else None)
        if version == "compiled":
            return "k-" + hashlib.sha256(("v3card|" + group["group_id"]).encode()).hexdigest()[:10]
        return id_map["items"][internal]
    return (sorted({one(f) for f in task["gold"]["required_facts"]}),
            sorted({one(f) for f in task["gold"]["necessary_context"]}))


def score(query, item):
    q = bigrams(query)
    return round(len(q & bigrams(item["title"] + "\n" + item["trigger"] + "\n" + item["text"]))
                 / len(q), 6) if q else 0.0


def select(items, query, budget):
    ranked = sorted(items, key=lambda it: (-score(query, it), it["id"]))
    chosen, costs = [], []
    for item in ranked:
        cost = item_cost(item)
        trial = costs + [cost]
        empty = len(project_full([], PHASE))
        if empty + sum(trial) + 2 * (len(trial) - 1) <= budget:
            chosen.append(item)
            costs = trial
    return chosen, project_full(chosen, PHASE)


def rules_edges(items):
    by_slot = defaultdict(list)
    for item in items:
        by_slot[item["title"]].append(item)
    edges = []
    for _, group in by_slot.items():
        for a in group:
            for b in group:
                if a["id"] >= b["id"]:
                    continue
                a_exc = any(m in a["text"] for m in ("不适用", "除外", "除非"))
                b_exc = any(m in b["text"] for m in ("不适用", "除外", "除非"))
                a_cond = any(m in a["text"] for m in ("若", "如果"))
                b_cond = any(m in b["text"] for m in ("若", "如果"))
                if a_exc and not b_exc:
                    edges.append({"type": "exception", "from": a["id"], "to": b["id"]})
                elif a_cond and not b_cond:
                    edges.append({"type": "context", "from": a["id"], "to": b["id"]})
                elif not a_exc and not b_exc and not a_cond and not b_cond \
                        and "必须" in a["text"] and "必须" in b["text"]:
                    edges.append({"type": "context", "from": a["id"], "to": b["id"],
                                  "basis": "multi_sibling"})
                elif "禁止" in a["text"] and "禁止" in b["text"]:
                    edges.append({"type": "equivalent", "from": a["id"], "to": b["id"]})
    return edges


def build_row(task, bank, groups_by_id, id_map, arm, budget, all_groups):
    version = {"P-full": "compound", "P-retrieve": "compound", "A-full": "atomic",
               "A-retrieve": "atomic", "A-graph": "atomic", "A-compiled": "compiled",
               "A-grouped": "grouped"}[arm]
    started = time.perf_counter()
    bank_groups = [groups_by_id[g] for g in bank["group_ids"]]
    items = []
    for group in bank_groups:
        items.extend(version_items(group, version, id_map))
    query = task["question"] + " " + task["context"]
    public_items = [r for r, _ in items]
    meta = {"arm": arm, "version": version, "budget_chars": budget,
            "scorer": SCORER if arm in RETRIEVE else None}
    if arm in ("P-retrieve", "A-retrieve"):
        selected, block = select(public_items, query, budget)
    elif arm == "A-graph":
        seeds, _ = select(public_items, query, max(1, budget // 2))
        chosen = list(seeds)
        seed_ids = {s["id"] for s in chosen}
        edges = rules_edges(public_items)
        by_id = {i["id"]: i for i in public_items}
        for e in edges:
            if e["from"] in seed_ids and e["to"] not in seed_ids:
                target = by_id.get(e["to"])
                if target and len(project_full(chosen + [target], PHASE)) <= budget:
                    chosen.append(target)
                    seed_ids.add(target["id"])
        selected, block = chosen, project_full(chosen, PHASE)
    else:
        selected, block = public_items, project_full(public_items, PHASE)
    internal = [next(i for r, i in items if r["id"] == s["id"]) for s in selected]
    fmap = {}
    for group in bank_groups:
        for row, internal_id in version_items(group, version, id_map):
            if row["id"] in {s["id"] for s in selected}:
                fmap[row["id"]] = fact_map(group, version, id_map,
                                           version_items(group, version, id_map))[row["id"]]
    internal_to_public = {i: r["id"] for r, i in items if r["id"] in {x["id"] for x in selected}}
    gmap = {}
    for group in bank_groups:
        for row, _ in version_items(group, version, id_map):
            if row["id"] in {s["id"] for s in selected}:
                gmap[row["id"]] = group["group_id"]
    req, ctx = carriers(task, groups_by_id[task["target_group"]], version, id_map) \
        if task["target_group"] else ([], [])
    return {"row_id": f"{task['task_id']}__{arm}__S{bank['scale']}__main",
            "task_id": task["task_id"], "family_id": task["family_id"],
            "split": bank["split"], "phenomenon": task["phenomenon"], "arm": arm,
            "variant": "main", "scale": bank["scale"], "position": bank["position"],
            "item_ids": [s["id"] for s in selected], "item_internal_ids": internal,
            "item_fact_ids": fmap, "item_internal_to_public": internal_to_public,
            "item_group_ids": gmap,
            "required_items_all": req, "context_items_all": ctx,
            "disclosure": block, "chars": len(block),
            "token_estimate": estimate_tokens(block),
            "build_latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "snapshot_ok": True, "degrade_code": None, "real_arm_result": False,
            "observation_kind": "independent", **meta}


def prompt(task, disclosure):
    lines = ["你是数据集目录助手。只依据下面的项目已确认知识回答；没有适用知识时输出 no_memory。",
             "===== 已确认知识开始 =====", disclosure.rstrip("\n"), "===== 已确认知识结束 =====",
             "----- 当前请求 -----", task["context"], f"问题：{task['question']}",
             "----- 输出要求 -----",
             "只输出一个 JSON 对象。字段：",
             '- decision: "allow" | "deny" | "ask_user" | "no_memory"',
             "- value: 直接从记忆文本取最短可判定值；数组题按顺序；ask_user 或无适用值填 null",
             "- applied_condition: 触发的条件名称（从记忆原文取）；否则 null",
             "- applied_exception: 触发的例外名称；否则 null",
             "- cited_item_ids: 只填 items[].id；不要填 links 或 source id；无依据填空数组",
             "若 decision=ask_user，value 必须为 null。"]
    return "\n".join(lines) + "\n"


def dry_run():
    groups, tasks, banks, id_map = load()
    by_id = {g["group_id"]: g for g in groups}
    task_banks = defaultdict(list)
    for b in banks:
        task_banks[b["task_id"]].append(b)
    rows = []
    for task in tasks:
        for bank in task_banks[task["task_id"]]:
            for arm in ARMS:
                budget = 4000 if arm in RETRIEVE else None
                rows.append(build_row(task, bank, by_id, id_map, arm, budget, groups))
    first_task = {}
    for t in tasks:
        first_task.setdefault(t["family_id"], t["task_id"])
    blocks = defaultdict(list)
    for row in rows:
        blocks[(row["split"], row["scale"], row["task_id"])].append(row)
    plan, reuse = [], []
    for index, key in enumerate(sorted(blocks)):
        block = blocks[key]
        by_arm = {r["arm"]: r for r in block}
        order = list(ARMS[0:6])
        shift = index % len(order)
        order = order[shift:] + order[:shift]
        grouped = by_arm.get("A-grouped")
        independent = (grouped and grouped["split"] == "final"
                       and grouped["scale"] in (24, 48)
                       and grouped["task_id"] == first_task[grouped["family_id"]])
        if independent:
            order.insert(index % (len(order) + 1), "A-grouped")
        for arm in order:
            row = by_arm.get(arm)
            if row:
                plan.append({"row_id": row["row_id"], "task_id": row["task_id"],
                             "split": row["split"], "scale": row["scale"], "arm": arm,
                             "planned_calls": 1, "reuse_of": None})
        if grouped and not independent:
            reuse.append({"row_id": grouped["row_id"], "task_id": grouped["task_id"],
                          "split": grouped["split"], "scale": grouped["scale"],
                          "arm": "A-grouped", "planned_calls": 0, "reuse_of": "P-full",
                          "observation_kind": "byte_identical_reuse"})
    OUT.mkdir(parents=True, exist_ok=True)
    for row in rows:
        task = next(t for t in tasks if t["task_id"] == row["task_id"])
        row["prompt"] = prompt(task, row["disclosure"])
        row["prompt_sha256"] = sha256(row["prompt"])
    jsonl_write(OUT / "disclosures.jsonl", rows)
    jsonl_write(OUT / "collect_plan.jsonl", plan + reuse)
    calls_by_arm = defaultdict(int)
    for p in plan:
        calls_by_arm[p["arm"]] += p["planned_calls"]
    summary = {"schema": "v3-dryrun", "ok": True, "rows": len(rows),
               "by_arm": {a: sum(1 for r in rows if r["arm"] == a) for a in ARMS},
               "plan_calls": len(plan), "a_grouped_reuse": len(reuse),
               "calls_by_arm": dict(sorted(calls_by_arm.items())),
               "caps": {"max_calls": 1200, "max_tokens": 4000000,
                        "headroom_calls": 1200 - len(plan)},
               "interleave": "block arm order rotates by block index; A-grouped inserted when independent",
               "llm_called": False,
               "note": "placeholder compiled / rules graph; real_arm_result=false"}
    json_write(OUT / "dryrun_summary.json", summary)
    return summary


def verify_budget(locked=4000):
    groups, tasks, banks, id_map = load()
    by_id = {g["group_id"]: g for g in groups}
    task_banks = defaultdict(list)
    for b in banks:
        task_banks[b["task_id"]].append(b)
    sweep = []
    for task in [t for t in tasks if t["family_id"] in ("F01", "F02", "F03", "F04")]:
        for bank in task_banks[task["task_id"]]:
            for arm in ("P-retrieve", "A-retrieve"):
                for budget in BUDGETS:
                    sweep.append(build_row(task, bank, by_id, id_map, arm, budget, groups))
    def stats(budget):
        out = {}
        for arm in ("P-retrieve", "A-retrieve"):
            rows = [r for r in sweep if r["arm"] == arm and r["budget_chars"] == budget]
            rec = [len(set(r["required_items_all"]) & set(r["item_ids"]))
                   / len(r["required_items_all"]) for r in rows if r["required_items_all"]]
            out[arm] = min(rec) if rec else None
        return out
    table = {b: stats(b) for b in BUDGETS}
    chosen = next((b for b in BUDGETS if all(v == 1.0 for v in table[b].values())), None)
    at_locked = table.get(locked, {})
    return {"schema": "v3-budget-verify", "locked": locked, "selected": chosen,
            "table": table, "at_locked": at_locked,
            "criterion_holds": bool(at_locked and all(v == 1.0 for v in at_locked.values()))}


def task_family_first(tasks, family):
    return next((t for t in tasks if t["family_id"] == family), None) is not None


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "dry-run":
        print(json.dumps(dry_run(), ensure_ascii=False, indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "verify-budget":
        print(json.dumps(verify_budget(), ensure_ascii=False, indent=2))
    else:
        groups, tasks, banks, id_map = load()
        assert len(groups) == 72 and len(tasks) == 48
        print(json.dumps({"ok": True, "groups": len(groups), "tasks": len(tasks),
                          "banks": len(banks)}, ensure_ascii=False))
