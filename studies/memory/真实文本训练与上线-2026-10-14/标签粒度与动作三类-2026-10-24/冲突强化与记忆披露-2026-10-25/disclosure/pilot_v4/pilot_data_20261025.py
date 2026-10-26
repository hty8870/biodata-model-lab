                      
"""Fixed twelve-task mechanism pilot; all tasks share one twelve-group library."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from disclosure_schema_20261025 import file_sha256, json_write, jsonl_read, jsonl_write, sha256

EXP = Path(__file__).resolve().parent
DATA = EXP / "data"
GENERATION = 2
SCALES = (4, 8, 12)
TIME = "当前是 2026-10。"
LIMITATIONS = {
    "split": "dev6/final6 are task-level mechanism exploration, NOT memory holdout; all banks share twelve groups",
    "position": "10 start / 2 middle / 0 end; no position-effect or end-position extrapolation",
    "scope": "user scope only; no project-scope or free-answer/second-consumer claims",
    "unit": "compound/card may cite once for all facts; atom arms require multiple citations; raw differences are not pure storage effects",
    "reference": "author-defined synthetic semantics, not human gold",
}
SCOPE_ALIASES = {
    "未发布数据集不适用默认样式": ["未发布数据集", "未发布数据集不适用默认样式，应先询问用户"],
    "用户开启邮件通知": ["用户已开启邮件通知", "若用户开启邮件通知，则同时发送邮件"],
}
VALUE_ALIASES = {
    "不重复导出": ["拒绝重复导出", "禁止重复导出", "不导出重复条目"],
    "拒绝覆盖": ["不覆盖", "不得覆盖", "禁止覆盖"],
    "对话内提醒": ["对话内", "对话内发送"],
    "邮件": ["邮件通知", "电子邮件"],
    "数据集标题": ["标题"],
    "数据来源链接": ["来源链接"],
}


def pub(prefix, internal):
    return f"{prefix}-{hashlib.sha256(('pilot4|' + internal).encode()).hexdigest()[:10]}"


                                                                 
SPECS = [
    ("F01-G1", "single_fact", "检索结果条数", "检索时", ["单次最多返回 5 条结果"],
     [("obligation", 0, "5 条")]),
    ("F01-G2", "multi_obligation", "检索结果字段", "展示检索结果时",
     ["检索结果必须给出数据集标题", "检索结果必须给出样本数", "检索结果必须给出数据来源链接"],
     [("obligation", 0, "数据集标题"), ("obligation", 1, "样本数"), ("obligation", 2, "数据来源链接")]),
    ("F02-G1", "single_fact", "下载文件大小", "下载文件时", ["单个下载文件最多允许 2 GB"],
     [("obligation", 0, "2 GB")]),
    ("F02-G2", "simultaneous", "下载并发数", "安排本轮并发下载时",
     ["同一时段对同一批下载任务的并发数必须恰好为 2 个", "同一时段对同一批下载任务的并发数必须恰好为 4 个"],
     [("obligation", 0, "2 个"), ("obligation", 1, "4 个")]),
    ("F03-G1", "conditional_exception", "引文样式", "选择引文导出样式时",
     ["引文默认使用 BibTeX 样式", "若用户要求 RIS，则改用 RIS 样式", "未发布数据集不适用默认样式，应先询问用户"],
     [("obligation", 0, "BibTeX 样式"), ("condition", 1, "用户要求 RIS"),
      ("consequence", 1, "RIS 样式"), ("exception", 2, "未发布数据集不适用默认样式"),
      ("consequence", 2, "ask_user", "exception_action")]),
    ("F03-G2", "negation", "引文去重", "导出引文条目时", ["禁止重复导出引文条目"],
     [("polarity", 0, "forbid")]),
    ("F04-G1", "conditional", "混合样本过滤", "筛选混合样本时",
     ["默认排除跨物种混合样本", "若用户明确研究临床混合样本，则保留混合样本"],
     [("obligation", 0, "排除混合样本"), ("condition", 1, "用户明确研究临床混合样本"),
      ("consequence", 1, "保留混合样本")]),
    ("F04-G2", "negation", "数据集覆盖", "保存同名数据集时", ["禁止覆盖已存在的同名数据集"],
     [("polarity", 0, "forbid")]),
    ("F05-G1", "conditional", "回复语言", "生成回复时",
     ["回复默认使用中文", "若用户明确要求英文，则改用英文"],
     [("obligation", 0, "中文"), ("condition", 1, "用户明确要求英文"), ("consequence", 1, "英文")]),
    ("F05-G2", "multi_obligation", "汇报顺序", "组织汇报内容时",
     ["汇报必须先给出结论", "汇报其次给出样本数", "汇报最后给出来源链接"],
     [("obligation", 0, "结论"), ("obligation", 1, "样本数"), ("obligation", 2, "来源链接")]),
    ("F06-G1", "conditional_exception", "提醒渠道", "发送提醒时",
     ["提醒默认在对话内发送", "若用户开启邮件通知，则同时发送邮件", "匿名会话不适用对话内提醒，应先询问是否登录"],
     [("obligation", 0, "对话内提醒"), ("condition", 1, "用户开启邮件通知"),
      ("consequence", 1, "邮件"), ("exception", 2, "匿名会话不适用对话内提醒"),
      ("consequence", 2, "ask_user", "exception_action")]),
    ("F06-G2", "supersede", "提醒条数", "限制本轮提醒数量时",
     ["2026-01 至 2026-06 期间单轮最多 3 条提醒", "自 2026-07 起单轮最多 5 条提醒"],
     [("obligation", 0, "3 条", "historical"), ("time", 0, "2026-01..2026-06"),
      ("obligation", 1, "5 条", "current"), ("time", 1, "2026-07..")]),
]


def make_group(spec):
    gid, profile, slot, trigger, texts, semantic = spec
    base = dict(scope_level="user", project_id="", kind="memory", title=slot,
                trigger=trigger, tier="explicit", permanence="persistent", payload={}, links=[], revision=GENERATION)
    atoms = [{**copy.deepcopy(base), "id": f"{gid}-A{i+1}", "text": text} for i, text in enumerate(texts)]
    facts = []
    for typ, idx, value, *roles in [("source", i, f"{gid}-S{i+1}") for i in range(len(atoms))] + semantic:
        facts.append({"fact_id": f"{gid}-F{idx+1}-{typ}", "type": typ, "atom_index": idx,
                      "probe": texts[idx], "value": value, "role": roles[0] if roles else "rule",
                      "polarity": "forbid" if typ == "polarity" else "require" if typ == "obligation" else "neutral"})
    return {"group_id": gid, "family_id": gid[:3], "profile": profile, "slot": slot,
            "trigger": trigger, "atoms": atoms,
            "compound": {**copy.deepcopy(base), "id": f"{gid}-P", "text": "；".join(texts) + "。"},
            "sources": [{"source_id": f"{gid}-S{i+1}", "title": f"来源{i+1}", "quote": t} for i, t in enumerate(texts)],
            "facts": facts,
            "atomic_fact_ids": {a["id"]: [f["fact_id"] for f in facts if f["atom_index"] == i] for i, a in enumerate(atoms)}}


def make_task(tid, phenomenon, target, question, context, decision, value, mode="exact", condition=None, exception=None, branch=""):
    return {"task_id": tid, "family_id": tid[:3] if tid.startswith("F") else "N",
            "phenomenon": phenomenon, "target_group": target, "question": question, "context": TIME + context,
            "branch": branch, "expected": {"decision": decision, "value": value, "value_mode": mode,
                                          "applied_condition": condition, "applied_exception": exception}}


def task_specs():
    return [
        make_task("F01-T1", "single_fact", "F01-G1", "本次检索最多返回多少条结果？", "", "allow", "5 条", branch="唯一检索条数规则"),
        make_task("F02-T1", "single_fact", "F02-G1", "单个下载文件最大允许多大？", "", "allow", "2 GB", branch="唯一单文件大小规则"),
        make_task("F01-T2", "multi_obligation", "F01-G2", "检索结果必须同时给出哪些内容？", "", "allow", ["数据集标题", "样本数", "数据来源链接"], "set", branch="三个并列义务，非汇报顺序"),
        make_task("F05-T2", "multi_obligation", "F05-G2", "汇报必须按什么顺序给出哪些内容？", "", "allow", ["结论", "样本数", "来源链接"], "ordered", branch="先/其次/最后唯一有序义务"),
        make_task("F03-T1", "conditional_exception", "F03-G1", "本次应如何处理引文样式？", " 目标属于未发布数据集，用户没有要求 RIS。", "ask_user", None, exception="未发布数据集不适用默认样式", branch="未发布例外覆盖默认；RIS未触发"),
        make_task("F06-T1", "conditional_exception", "F06-G1", "本次提醒应通过哪些渠道发送？", " 用户开启邮件通知，会话已登录。", "allow", ["对话内提醒", "邮件"], "set", condition="用户开启邮件通知", branch="邮件追加而非替代默认；匿名例外未触发"),
        make_task("F02-T2", "simultaneous", "F02-G2", "本次应安排几个并发下载？", " 本轮正在安排同一批下载任务。", "ask_user", None, branch="同对象同时段恰好2与恰好4不可同时满足"),
        make_task("F06-T2", "supersede", "F06-G2", "当前单轮最多允许几条提醒？", "", "allow", "5 条", branch="2026-07起生效；旧3条已过期"),
        make_task("F03-T2", "negation", "F03-G2", "本批是否重复导出引文条目？", " 同一引文条目已在本批导出。", "deny", "不重复导出", branch="禁止重复导出；不是导出次数限额"),
        make_task("F04-T2", "negation", "F04-G2", "目标项目已存在同名数据集，是否覆盖？", " 目标项目已存在同名数据集。", "deny", "拒绝覆盖", branch="唯一覆盖禁止规则"),
        make_task("N1", "no_relevant", None, "把检索结果画成雷达图。", "", "no_memory", None, branch="库中无可视化规则；检索字段不能决定图形"),
        make_task("N2", "no_relevant", None, "给反馈包设置访问密码。", "", "no_memory", None, branch="库中无反馈包访问控制或密码规则；不是提醒渠道"),
    ]


def gold(task, by_id):
    if not task["target_group"]:
        return dict(required_facts=[], necessary_context=[], forbidden_facts=[], required_groups=[])
    g = by_id[task["target_group"]]
    ph = task["phenomenon"]
    required, context, forbidden = [], [], []
    for f in g["facts"]:
        typ, idx = f["type"], f["atom_index"]
        if ph == "conditional_exception":
            applied = 2 if task["expected"]["decision"] == "ask_user" else 1
            if idx == applied:
                required.append(f["fact_id"])
            elif idx == 0:
                context.append(f["fact_id"])
        elif ph == "supersede":
            if typ == "obligation" and f["role"] == "historical":
                forbidden.append(f["fact_id"])
            elif idx == 1 and typ in ("obligation", "source"):
                required.append(f["fact_id"])
            else:
                context.append(f["fact_id"])
        else:
            required.append(f["fact_id"])
    return dict(required_facts=required, necessary_context=context, forbidden_facts=forbidden, required_groups=[g["group_id"]])


def materialize():
    groups = [make_group(s) for s in SPECS]
    by_id = {g["group_id"]: g for g in groups}
    tasks = task_specs()
    for t in tasks:
        t["split"] = "dev" if t["family_id"] in ("F01", "F02", "F03") else "final"
        t["gold"] = gold(t, by_id)
        t["stale_value"] = "3 条" if t["phenomenon"] == "supersede" else None
    id_map = {"items": {}, "sources": {}, "groups": {}}
    for g in groups:
        id_map["groups"][g["group_id"]] = pub("g", g["group_id"])
        for a in [g["compound"]] + g["atoms"]:
            id_map["items"][a["id"]] = pub("m", a["id"])
        for s in g["sources"]:
            id_map["sources"][s["source_id"]] = pub("s", s["source_id"])
    for kind in ("items", "sources", "groups"):
        id_map["reverse_" + kind] = {v: k for k, v in id_map[kind].items()}
    banks = []
    for t in tasks:
        pool = sorted((g["group_id"] for g in groups if g["group_id"] != t["target_group"]), key=lambda gid: sha256(t["task_id"] + "|" + gid))
        position = "middle" if t["task_id"] in ("F02-T1", "N2") else "start"
        for scale in SCALES:
            ordered = pool[:scale - bool(t["target_group"])]
            idx = None
            if t["target_group"]:
                idx = len(ordered) // 2 if position == "middle" else 0
                ordered.insert(idx, t["target_group"])
            banks.append(dict(bank_id=f"{t['task_id']}-S{scale}", task_id=t["task_id"], split=t["split"], scale=scale,
                              position=position, relevant_index=idx, relevant_groups=t["gold"]["required_groups"], group_ids=ordered))
    return groups, tasks, banks, id_map


def verify(groups=None, tasks=None, banks=None, id_map=None):
    if groups is None:
        groups, tasks, banks, id_map = (jsonl_read(DATA / "groups.jsonl"), jsonl_read(DATA / "tasks.jsonl"), jsonl_read(DATA / "banks.jsonl"), json.loads((DATA / "id_map.json").read_text(encoding="utf-8")))
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    check("12groups_12tasks_36banks", (len(groups), len(tasks), len(banks)) == (12, 12, 36))
    check("unique_ids", len({g["group_id"] for g in groups}) == 12 and len({t["task_id"] for t in tasks}) == 12)
    check("task_split_6_6", Counter(t["split"] for t in tasks) == {"dev": 6, "final": 6})
    by_id = {g["group_id"]: g for g in groups}
    for g in groups:
        gid = g["group_id"]
        check(gid+":source_quotes", all(s["quote"] == a["text"] for s, a in zip(g["sources"], g["atoms"])) and len(g["sources"]) == len(g["atoms"]))
        check(gid+":grouped_bytes", "；".join(a["text"] for a in g["atoms"]) + "。" == g["compound"]["text"])
        check(gid+":projection", all(a["links"] == [] and a["scope_level"] == "user" and a["project_id"] == "" and a["permanence"] == "persistent" and a["revision"] == GENERATION for a in [g["compound"]] + g["atoms"]))
        check(gid+":trigger", g["trigger"] and "按记忆" not in g["trigger"])
        types = {f["type"] for f in g["facts"]}
        check(gid+":profile", (g["profile"] != "conditional" or ("condition" in types and "exception" not in types and len(g["atoms"]) == 2)) and (g["profile"] != "conditional_exception" or ("exception" in types and len(g["atoms"]) == 3)))
    for t in tasks:
        tid, exp = t["task_id"], t["expected"]
        check(tid+":no_options", not any(k in t for k in ("options", "choices", "answers")))
        check(tid+":ask_null", exp["decision"] not in ("ask_user", "no_memory") or exp["value"] is None)
        check(tid+":decision", exp["decision"] in ("allow", "deny", "ask_user", "no_memory"))
        check(tid+":branch", bool(t["branch"]))
        bs = sorted([b for b in banks if b["task_id"] == tid], key=lambda b: b["scale"])
        check(tid+":nested", len(bs) == 3 and set(bs[0]["group_ids"]) < set(bs[1]["group_ids"]) < set(bs[2]["group_ids"]))
        check(tid+":bank_targets", all(len(b["group_ids"]) == b["scale"] and set(t["gold"]["required_groups"]) <= set(b["group_ids"]) for b in bs))
        if not t["target_group"]:
            check(tid+":negative", not t["gold"]["required_facts"] and not t["gold"]["necessary_context"])
            continue
        g = by_id[t["target_group"]]
        needed = set(t["gold"]["required_facts"] + t["gold"]["necessary_context"])
        facts = {f["fact_id"]: f for f in g["facts"]}
        check(tid+":reachable_facts", needed <= set(facts))
        if t["phenomenon"] in ("single_fact", "multi_obligation", "simultaneous", "negation"):
            check(tid+":all_obligations_and_evidence", set(t["gold"]["required_facts"]) == set(facts))
        if t["phenomenon"] == "conditional_exception":
            applied = 2 if exp["decision"] == "ask_user" else 1
            check(tid+":only_triggered_branch_required", {facts[f]["atom_index"] for f in t["gold"]["required_facts"]} == {applied})
            check(tid+":default_context_not_other_branch", {facts[f]["atom_index"] for f in t["gold"]["necessary_context"]} == {0})
        for fid in needed:
            f = facts[fid]
            src = next(s for s in g["facts"] if s["type"] == "source" and s["atom_index"] == f["atom_index"])
            check(tid+":"+fid+":source", src["fact_id"] in needed)
        for key in ("applied_condition", "applied_exception"):
            v = exp[key]
            check(tid+":"+key, v is None or any(v in a["text"] for a in g["atoms"]))
        vals = exp["value"] if isinstance(exp["value"], list) else [exp["value"]]
        semantic_values = {f["value"] for f in g["facts"]}
        check(tid+":expected_reachable", exp["decision"] in ("ask_user", "deny") or all(v in semantic_values for v in vals))
        check(tid+":forbidden_usage", not t["gold"]["forbidden_facts"] or (t["stale_value"] is not None and all(facts[f]["value"] == t["stale_value"] for f in t["gold"]["forbidden_facts"])))
    sim = by_id["F02-G2"]
    check("simultaneous_exact_same_object", all("同一时段对同一批下载任务" in a["text"] and "必须恰好" in a["text"] for a in sim["atoms"]))
    check("simultaneous_no_question_leak", not any(c.isdigit() for c in next(t for t in tasks if t["task_id"] == "F02-T2")["question"]))
    check("public_opaque_ids", all(v.startswith(prefix) and "F0" not in v for kind, prefix in (("items", "m-"), ("sources", "s-"), ("groups", "g-")) for v in id_map[kind].values()))
    return dict(ok=True, passed=len(checks), checks=checks, counts=dict(groups=12, tasks=12, banks=36), limitations=LIMITATIONS,
                position_counts=dict(Counter(t["position"] for t in banks if t["scale"] == 12)))


def build():
    groups, tasks, banks, id_map = materialize()
    result = verify(groups, tasks, banks, id_map)
    for name, value in (("groups", groups), ("tasks", tasks), ("banks", banks)):
        jsonl_write(DATA / f"{name}.jsonl", value)
    json_write(DATA / "id_map.json", id_map)
    manifest = dict(schema="pilot-v4-r2", generation=GENERATION, counts=result["counts"], limitations=LIMITATIONS,
                    sha256={n: file_sha256(DATA / n) for n in ("groups.jsonl", "tasks.jsonl", "banks.jsonl", "id_map.json")},
                    authoring=dict(author="disclosure-writer", review="author self-check only; independent review pending"))
    json_write(DATA / "manifest.json", manifest)
    json_write(EXP / "out/data_verify.json", result)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("build", "verify"), default="build")
    args = parser.parse_args()
    print(json.dumps(build() if args.command == "build" else verify(), ensure_ascii=False, indent=2))
