                      
"""v3 基准重建：无选项、公开不透明 ID、逐事实来源、嵌套银行、语义修复。

来源：`_archive/benchmark-v2-flawed/data`（只读迁移）。本文件只做数据与任务。
真实 LLM 调用仍禁止。命令：`build` / `selftest`。
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from disclosure_schema_20261025 import (              
    DATA, EXP, file_sha256, json_write, jsonl_read, jsonl_write, sha256,
)

V2 = EXP / "_archive" / "benchmark-v2-flawed" / "data"
SPLIT_SCALES = {"dev": (8, 24), "final": (8, 24, 48)}
POSITIONS = ("start", "middle", "end")
CURRENT_TIME = "当前是 2026-10。"

CONFLICT_REWRITE = {
    "F05-G4": ("回复模板缓存有效期", "2026-01 至 2026-06 期间回复模板缓存最多保留 12 小时",
               "自 2026-07 起回复模板缓存最多保留 24 小时", "24 小时", "12 小时"),
    "F08-G4": ("缓存条目数上限", "自 2026-07 起缓存条目数上限为 3000 条",
               "自 2026-07 起缓存条目数上限为 5000 条", None, None),
    "F09-G4": ("任务包保留天数", "2026-01 至 2026-06 期间任务包最多保留 30 天",
               "自 2026-07 起任务包最多保留 14 天", "14 天", "30 天"),
    "F10-G4": ("比较结果缓存有效期", "2026-01 至 2026-06 期间比较结果缓存最多保留 2 小时",
               "自 2026-07 起比较结果缓存最多保留 4 小时", "4 小时", "2 小时"),
    "F11-G4": ("版本导出保留天数", "2026-01 至 2026-06 期间版本导出文件最多保留 30 天",
               "自 2026-07 起版本导出文件最多保留 7 天", "7 天", "30 天"),
    "F12-G4": ("提醒邮件重试次数", "自 2026-07 起提醒邮件最多重试 2 次",
               "自 2026-07 起提醒邮件最多重试 3 次", None, None),
}
EXCEPTION_TEXT = {
    "F01-G3": "对外部数据集不适用中文默认，应先询问用户选择说明语言",
    "F02-G3": "对只读归档目录不适用直接下载，应先询问用户是否下载",
    "F03-G3": "对未发布数据集不适用任何引文样式，应先询问用户是否需要引文",
    "F04-G3": "对外部贡献者不适用自动通过，应先询问用户是否转人工审核",
    "F05-G3": "对公开交付材料不适用中文默认，应先询问用户使用哪种语言",
    "F06-G3": "对临床来源数据不适用默认排除，应先询问用户是否保留",
    "F07-G3": "对精选数据集不适用该默认排序，应先询问用户排序方式",
    "F08-G3": "对只读历史快照不适用缓存复用，应先询问用户是否读取快照",
    "F09-G3": "对失败运行不适用默认打包范围，应先询问用户是否包含失败运行",
    "F10-G3": "对仅有一个数据集的情况不适用比较，应先询问用户是否继续",
    "F11-G3": "对已归档数据集不适用版本回退，应先询问用户是否解归档",
    "F12-G3": "对匿名会话不适用邮件通知，应先询问用户是否登录",
}
MULTI_ATOMS = {
    "F01": ["检索结果必须给出数据集标题", "检索结果必须给出样本数", "检索结果必须标注数据来源链接"],
    "F02": ["下载前必须确认文件大小", "下载前必须确认文件校验和", "下载前必须确认目标目录剩余空间"],
    "F03": ["每条引文必须包含数据集标题", "每条引文必须包含访问日期", "每条引文必须包含持久标识符"],
    "F04": ["上传登记必须登记数据集标题", "上传登记必须登记样本物种", "上传登记必须登记数据许可"],
    "F05": ["项目回复必须使用项目术语表", "项目回复必须给出数据集编号", "项目回复必须避免内部批次号"],
    "F06": ["单细胞结果必须标注物种", "单细胞结果必须标注组织来源", "单细胞结果必须标注测序平台"],
    "F07": ["结果排序必须优先考虑相关性", "结果排序其次必须考虑数据量", "结果排序最后必须考虑更新时间"],
    "F08": ["缓存维护在缓存过期时必须重建", "缓存维护在索引变更时必须重建", "缓存维护在数据更新时必须重建"],
    "F09": ["反馈任务包必须包含清单文件", "反馈任务包必须包含校验和", "反馈任务包必须包含来源说明"],
    "F10": ["数据集比较必须包含样本数", "数据集比较必须包含物种", "数据集比较必须包含文件大小"],
    "F11": ["版本记录必须包含修订时间", "版本记录必须包含修订人", "版本记录必须包含变更摘要"],
    "F12": ["用户提醒必须说明触发原因", "用户提醒必须给出下一步动作", "用户提醒必须标注是否可关闭"],
}
MULTI_LABEL = {
    "F01": ["给出数据集标题", "给出样本数", "标注数据来源链接"],
    "F02": ["确认文件大小", "确认文件校验和", "确认目标目录剩余空间"],
    "F03": ["包含数据集标题", "包含访问日期", "包含持久标识符"],
    "F04": ["登记数据集标题", "登记样本物种", "登记数据许可"],
    "F05": ["使用项目术语表", "给出数据集编号", "避免内部批次号"],
    "F06": ["标注物种", "标注组织来源", "标注测序平台"],
    "F07": ["优先考虑相关性", "其次考虑数据量", "最后考虑更新时间"],
    "F08": ["缓存过期时重建", "索引变更时重建", "数据更新时重建"],
    "F09": ["包含清单文件", "包含校验和", "包含来源说明"],
    "F10": ["包含样本数", "包含物种", "包含文件大小"],
    "F11": ["包含修订时间", "包含修订人", "包含变更摘要"],
    "F12": ["说明触发原因", "给出下一步动作", "标注是否可关闭"],
}
NEG_MIN = {"F02": "拒绝覆盖", "F03": "只导出一次", "F04": "拒绝覆盖", "F05": "不得引用",
           "F08": "不得静默重建", "F09": "不得包含原始数据", "F10": "不得填充", "F11": "禁止物理删除"}
DISTRACTOR_REWRITE = {
    "F04-G6": ("下载列表排序", "下载列表必须按文件大小升序展示"),
    "F08-G6": ("引文导出编码", "导出引文时必须使用 UTF-8 编码"),
    "F09-G6": ("反馈包压缩格式", "反馈包必须使用 zip 压缩格式"),
    "F10-G6": ("引文版本字段", "导出引文时必须标注数据集版本号"),
    "F11-G6": ("下载断点续传", "下载中断后必须支持断点续传"),
    "F12-G6": ("提醒渠道优先级", "提醒必须优先发送到当前活跃会话"),
    "F05-G6": ("下载完成日志", "下载完成后必须记录下载日志"),
}
NO_REL_Q = {
    "F03": "请按记忆处理本次请求：把结果写入日历。",
    "F04": "请按记忆处理本次请求：给数据集设置封面图。",
    "F05": "请按记忆处理本次请求：给数据集打颜色标签。",
    "F06": "请按记忆处理本次请求：把结果发送到短信。",
    "F09": "请按记忆处理本次请求：生成数据集音频朗读。",
    "F10": "请按记忆处理本次请求：把比较结果写入日历。",
    "F11": "请按记忆处理本次请求：给版本记录设置密码。",
    "F12": "请按记忆处理本次请求：把提醒同步到日历。",
}
COND_LABEL = {"F01-G3": "用户明确要求英文", "F02-G3": "目标目录剩余空间不足 5 GB",
              "F03-G3": "用户要求 RIS", "F04-G3": "上传者属于同一项目",
              "F05-G3": "用户用英文提问", "F06-G3": "用户明确研究混合样本",
              "F07-G3": "用户指定按数据量排序", "F08-G3": "索引在本轮发生变更",
              "F09-G3": "用户要求全部运行", "F10-G3": "两侧缺失同一维度",
              "F11-G3": "用户是项目所有者", "F12-G3": "用户开启邮件通知"}
EXC_LABEL = {"F01-G3": "对外部数据集不适用中文默认", "F02-G3": "对只读归档目录不适用直接下载",
             "F03-G3": "对未发布数据集不适用任何引文样式", "F04-G3": "对外部贡献者不适用自动通过",
             "F05-G3": "对公开交付材料不适用中文默认", "F06-G3": "对临床来源数据不适用默认排除",
             "F07-G3": "对精选数据集不适用该默认排序", "F08-G3": "对只读历史快照不适用缓存复用",
             "F09-G3": "对失败运行不适用默认打包范围", "F10-G3": "对仅有一个数据集的情况不适用比较",
             "F11-G3": "对已归档数据集不适用版本回退", "F12-G3": "对匿名会话不适用邮件通知"}
COND_ACTION = {"F01-G3": "改用英文说明", "F02-G3": "先询问用户", "F03-G3": "改用 RIS 样式",
               "F04-G3": "自动通过审核", "F05-G3": "改用英文回复", "F06-G3": "保留混合样本",
               "F07-G3": "按数据量排序", "F08-G3": "跳过缓存", "F09-G3": "打包全部运行",
               "F10-G3": "省略该维度", "F11-G3": "允许回退任意版本", "F12-G3": "同时发送邮件"}
CONFLICT_TASK = {"F08": ("当前缓存条目数上限是多少？", "ask_user", None),
                 "F09": ("当前任务包最多保留多少天？", "allow", "14 天"),
                 "F10": ("当前比较结果缓存最多保留多久？", "allow", "4 小时")}


def _load_v2():
    return ({g["group_id"]: g for g in jsonl_read(V2 / "groups.jsonl")},
            jsonl_read(V2 / "tasks.jsonl"))


def _public(prefix: str, internal: str) -> str:
    return f"{prefix}-{hashlib.sha256(('v3|' + internal).encode()).hexdigest()[:10]}"


def _group(gid: str, v2: dict) -> dict:
    family = gid.split("-")[0]
    texts = [a["text"] for a in v2["atoms"]]
    title = v2["slot"]
    if gid in CONFLICT_REWRITE:
        title, old_text, new_text, _, _ = CONFLICT_REWRITE[gid]
        if texts[0].startswith("2026-01"):
            texts = [old_text, new_text]
        else:
            texts = [new_text, old_text]
    elif gid in EXCEPTION_TEXT:
        texts[2] = EXCEPTION_TEXT[gid]
    elif gid in DISTRACTOR_REWRITE:
        title, texts = DISTRACTOR_REWRITE[gid][0], [DISTRACTOR_REWRITE[gid][1]]
    elif family in MULTI_ATOMS and gid.endswith("G2"):
        texts = list(MULTI_ATOMS[family])
    atoms = [{"id": f"{gid}-A{i + 1}", "scope_level": "user", "project_id": "",
              "kind": "memory", "title": title, "text": t, "trigger": v2["trigger"],
              "tier": "explicit", "permanence": "persistent", "payload": {},
              "links": [], "revision": 1} for i, t in enumerate(texts)]
    sources = [{"source_id": f"{gid}-S{i + 1}", "title": f"来源{i + 1}", "quote": t}
               for i, t in enumerate(texts)]
    text = "；".join(texts) + "。"
    facts = []
    for i, atom in enumerate(atoms):
        facts.append(_f(gid, i, "source", atom["text"], f"{gid}-S{i + 1}", polarity="neutral"))
    if v2["profile"] == "single_fact":
        facts.append(_f(gid, 0, "obligation", texts[0], texts[0], polarity="require"))
    elif v2["profile"] == "multi_obligation":
        for i in range(3):
            facts.append(_f(gid, i, "obligation", texts[i], MULTI_LABEL[family][i],
                            polarity="require"))
    elif v2["profile"] == "conditional_exception":
        facts += [_f(gid, 0, "obligation", texts[0], texts[0], polarity="require"),
                  _f(gid, 1, "condition", texts[1], COND_LABEL[gid]),
                  _f(gid, 1, "consequence", texts[1], COND_ACTION[gid]),
                  _f(gid, 2, "exception", texts[2], EXC_LABEL[gid]),
                  _f(gid, 2, "consequence", texts[2], "ask_user", role="exception_action")]
    elif v2["profile"] == "negation":
        facts.append(_f(gid, 0, "polarity", texts[0], "forbid", polarity="forbid"))
        m = re.search(r"\d+\s*(分钟|小时|天|周|个月)", texts[0])
        if m:
            facts.append(_f(gid, 0, "time", texts[0], m.group(0)))
    elif v2["profile"] == "conflict_update":
        current = [i for i, t in enumerate(texts) if "自 2026-07" in t]
        old = [i for i in range(len(texts)) if i not in current]
        _, _, _, current_value, old_value = CONFLICT_REWRITE.get(
            gid, (None, None, None, None, None))
        if current_value is None:
            legacy = {}
            for f in v2["facts"]:
                if f["type"] != "obligation":
                    continue
                idx = next((i for i, a in enumerate(v2["atoms"])
                            if f["probe"] in a["text"]), None)
                if idx is not None:
                    legacy[idx] = f["value"]
            current_value = legacy.get(current[0]) if current else None
            old_value = legacy.get(old[0]) if old else None
        for i, atom in enumerate(atoms):
            is_old = i in old
            facts.append(_f(gid, i, "obligation", atom["text"],
                            old_value if is_old else current_value,
                            polarity="require",
                            role="historical" if is_old else "current"))
            facts.append(_f(gid, i, "time", atom["text"],
                            "2026-01..2026-06" if is_old else "2026-07.."))
    else:
        facts.append(_f(gid, 0, "obligation", texts[0], texts[0], polarity="require"))
    return {"group_id": gid, "family_id": family, "profile": v2["profile"],
            "mode": v2.get("mode"), "slot": title, "slot_root": re.sub(r"\s+", "", title),
            "topic": v2["profile"], "trigger": v2["trigger"], "atoms": atoms,
            "compound": {"id": f"{gid}-P", "scope_level": "user", "project_id": "",
                         "kind": "memory", "title": title, "text": text,
                         "trigger": v2["trigger"], "tier": "explicit",
                         "permanence": "persistent", "payload": {}, "links": [],
                         "revision": 1},
            "sources": sources, "facts": facts,
            "atomic_fact_ids": {a["id"]: [f["fact_id"] for f in facts if f["atom_index"] == i]
                                for i, a in enumerate(atoms)}}


def _f(gid, idx, ftype, probe, value, *, polarity="neutral", role="rule"):
    tag = {"obligation": "o", "condition": "c", "consequence": "q", "exception": "e",
           "time": "t", "polarity": "p", "source": "s"}[ftype]
    return {"fact_id": f"{gid}-F{idx + 1}{tag}", "type": ftype, "probe": probe,
            "value": value, "polarity": polarity, "time": None,
            "atom_index": idx, "role": role}


def _tasks(v2_tasks: list[dict], by_id: dict) -> list[dict]:
    out = []
    for t in v2_tasks:
        gid = t["target_group"]
        fam = t["family_id"]
        ph = t["phenomenon"]
        group = by_id.get(gid) if gid else None
        question = t["question"]
        expected = dict(t["expected"])
        expected.update({"value_mode": "exact", "applied_condition": None,
                         "applied_exception": None})
        if ph == "multi_obligation":
            question = f"按优先级从高到低，{group['slot']}必须满足哪些要求？"
            expected["value"] = MULTI_LABEL[fam]
            expected["value_mode"] = "ordered"
        elif ph == "negation":
            question = f"按记忆，本次应如何执行？"
            expected.update({"decision": "deny", "value": NEG_MIN[fam]})
        elif ph == "conditional_exception":
            if fam in ("F03", "F07", "F09"):
                question = "该默认规则是否适用？如不适用，应如何处理？"
                expected.update({"decision": "ask_user", "value": None,
                                 "applied_exception": EXC_LABEL[gid],
                                 "applied_condition": None})
            else:
                question = "按记忆，本次应执行什么动作？"
                expected.update({"decision": "allow", "value": COND_ACTION[gid],
                                 "applied_condition": COND_LABEL[gid],
                                 "applied_exception": None})
        elif ph == "conflict_update":
            if fam in CONFLICT_TASK:
                question, decision, value = CONFLICT_TASK[fam]
                expected.update({"decision": decision, "value": value})
            if group["mode"] == "simultaneous":
                expected.update({"decision": "ask_user", "value": None})
        elif ph == "no_relevant":
            question = NO_REL_Q[fam]
            expected.update({"decision": "no_memory", "value": None})
        out.append({"task_id": t["task_id"], "family_id": fam, "phenomenon": ph,
                    "target_group": gid, "question": question,
                    "context": ("情境：" + CURRENT_TIME) if ph != "no_relevant" else CURRENT_TIME,
                    "expected": expected, "variant": "main",
                    "gold": _gold(ph, group, expected)})
    return out


def _gold(ph, group, expected):
    if ph == "no_relevant":
        return {"required_facts": [], "necessary_context": [], "forbidden_facts": [],
                "fact_checks": {}, "required_groups": []}
    facts = group["facts"]
    by_type = defaultdict(list)
    for f in facts:
        by_type[f["type"]].append(f)
    required, necessary, forbidden, checks = [], [], [], {}
    if ph in ("single_fact", "multi_obligation"):
        required = [f["fact_id"] for f in by_type["obligation"]] + \
                   [f["fact_id"] for f in by_type["source"] if f["atom_index"] == 0]
    elif ph == "conditional_exception":
        if expected["decision"] == "ask_user":
            required = [f["fact_id"] for f in by_type["exception"]] + \
                       [f["fact_id"] for f in by_type["consequence"]
                        if f["role"] == "exception_action"] + \
                       [f["fact_id"] for f in by_type["source"] if f["atom_index"] == 2]
            necessary = []
        else:
            required = [f["fact_id"] for f in by_type["condition"]] + \
                       [f["fact_id"] for f in by_type["consequence"] if f["role"] == "rule"] + \
                       [f["fact_id"] for f in by_type["source"] if f["atom_index"] == 1]
            necessary = []
    elif ph == "conflict_update":
        if group["mode"] == "supersede":
            current = [f for f in by_type["obligation"] if f["role"] == "current"]
            old = [f for f in by_type["obligation"] if f["role"] == "historical"]
            required = [f["fact_id"] for f in current] + \
                       [f["fact_id"] for f in by_type["source"]
                        if f["atom_index"] == current[0]["atom_index"]]
            necessary = []
            forbidden = [f["fact_id"] for f in old]
        else:
            required = [f["fact_id"] for f in by_type["obligation"]] + \
                       [f["fact_id"] for f in by_type["source"]]
            necessary = []
    elif ph == "negation":
        required = [f["fact_id"] for f in by_type["polarity"]] + \
                   [f["fact_id"] for f in by_type["source"]]
        necessary = [f["fact_id"] for f in by_type["time"]]
    by_id = {f["fact_id"]: f for f in facts}
    for fid in required + necessary:
        f = by_id[fid]
        if f["type"] == "obligation":
            checks[fid] = {"field": "value", "equals": expected["value"]}
        elif f["type"] == "consequence":
            checks[fid] = ({"field": "decision", "equals": "ask_user"}
                           if f["role"] == "exception_action"
                           else {"field": "value", "equals": expected["value"]})
        elif f["type"] == "condition":
            checks[fid] = {"field": "applied_condition", "equals": expected["applied_condition"]}
        elif f["type"] == "exception":
            checks[fid] = {"field": "applied_exception", "equals": expected["applied_exception"]}
        elif f["type"] == "time":
            checks[fid] = ({"field": "decision", "equals": "ask_user"}
                           if ph == "conflict_update" and group["mode"] == "simultaneous"
                           else {"field": "value", "equals": expected["value"]})
        elif f["type"] == "polarity":
            checks[fid] = {"field": "decision", "equals": expected["decision"]}
    return {"required_facts": required, "necessary_context": necessary,
            "forbidden_facts": forbidden, "fact_checks": checks,
            "required_groups": [group["group_id"]] if group else []}


def _build_banks(groups: list[dict], tasks: list[dict]) -> list[dict]:
    split_of = {g["group_id"]: ("dev" if g["family_id"] in ("F01", "F02", "F03", "F04")
                                else "final") for g in groups}
    by_id = {g["group_id"]: g for g in groups}
    banks = []
    for task in tasks:
        split = split_of[task["target_group"]] if task["target_group"] else \
            ("dev" if task["family_id"] in ("F01", "F02", "F03", "F04") else "final")
        pool = [g["group_id"] for g in groups if split_of[g["group_id"]] == split]
        if task["target_group"]:
            target = task["target_group"]
            pool = [g for g in pool if g != target]
            forbidden_root = by_id[target]["slot_root"]
            pool = [g for g in pool if by_id[g]["slot_root"] != forbidden_root]
        phen_tasks = [t for t in tasks if t["phenomenon"] == task["phenomenon"]
                      and (split_of[t["target_group"]] if t["target_group"] else
                           ("dev" if t["family_id"] in ("F01", "F02", "F03", "F04") else "final")) == split]
        position = POSITIONS[phen_tasks.index(task) % 3]
        for scale in SPLIT_SCALES[split]:
            need = scale - (1 if task["target_group"] else 0)
            ordered = sorted(pool, key=lambda g: _hash(task["task_id"], g))[:need]
            if len(ordered) != need:
                raise AssertionError(f"{task['task_id']}: S{scale} 组不足 {need}/{len(pool)}")
            if task["target_group"]:
                pos = {"start": 0, "middle": len(ordered) // 2, "end": len(ordered)}[position]
                ordered.insert(pos, task["target_group"])
                idx = pos
            else:
                idx = None
            banks.append({"bank_id": f"{task['task_id']}-S{scale}", "task_id": task["task_id"],
                          "split": split, "scale": scale, "position": position,
                          "relevant_index": idx,
                          "relevant_groups": [task["target_group"]] if task["target_group"] else [],
                          "group_ids": ordered})
    return banks


def _hash(*parts) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _build_id_map(groups: list[dict]) -> dict:
    items, sources = {}, {}
    for g in groups:
        for item in [g["compound"]] + g["atoms"]:
            items[item["id"]] = _public("m", item["id"])
        for s in g["sources"]:
            sources[s["source_id"]] = _public("s", s["source_id"])
    groups_map = {g["group_id"]: _public("g", g["group_id"]) for g in groups}
    return {"items": items, "sources": sources, "groups": groups_map,
            "reverse_items": {v: k for k, v in items.items()},
            "reverse_sources": {v: k for k, v in sources.items()},
            "reverse_groups": {v: k for k, v in groups_map.items()}}


def verify() -> dict:
    groups = jsonl_read(DATA / "groups.jsonl")
    tasks = jsonl_read(DATA / "tasks.jsonl")
    banks = jsonl_read(DATA / "banks.jsonl")
    checks = {}

    def check(code, errs):
        checks[code] = {"ok": not errs, "n": len(errs), "errors": errs[:10]}

    check("V1-counts", [] if (len(groups) == 72 and len(tasks) == 48 and len(banks) == 128)
          else [f"{len(groups)}/{len(tasks)}/{len(banks)}"])
    check("V2-compound", [g["group_id"] for g in groups
                          if g["compound"]["text"] != "；".join(
                              a["text"] for a in g["atoms"]) + "。"])
    check("V3-atom-quote", [f"{g['group_id']}:{i}" for g in groups
                            for i, atom in enumerate(g["atoms"])
                            if next(s for s in g["sources"]
                                    if s["source_id"].endswith(f"S{i + 1}"))["quote"]
                            != atom["text"]])
    check("V4-visible-links", [g["group_id"] for g in groups
                               if any(a["links"] for a in g["atoms"])
                               or g["compound"]["links"]])
    check("V5-fact-probe", [f"{g['group_id']}:{f['fact_id']}" for g in groups
                            for f in g["facts"]
                            if f["probe"] not in g["atoms"][f["atom_index"]]["text"]])
    titles = defaultdict(list)
    for g in groups:
        titles[g["slot_root"]].append(g["group_id"])
    check("V6-slot-root", [f"{k}:{v}" for k, v in titles.items() if len(v) > 1])
    check("V7-tasks", [t["task_id"] for t in tasks
                       if "options" in t
                       or t["expected"]["decision"] == "ask_user"
                       and t["expected"]["value"] is not None
                       or t["phenomenon"] != "no_relevant" and "2026-10" not in t["context"]])
    bad_bank = []
    for split in ("dev", "final"):
        rows = [b for b in banks if b["split"] == split]
        for task in {b["task_id"] for b in rows}:
            sets = {b["scale"]: set(b["group_ids"])
                    for b in rows if b["task_id"] == task}
            scales = sorted(sets)
            for a, b in zip(scales, scales[1:]):
                if not sets[a] <= sets[b]:
                    bad_bank.append(f"{task}:S{a}")
    check("V8-bank-nested", bad_bank)
    manifest_path = DATA / "manifest.json"
    bad_sha = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, digest in manifest.get("sha256", {}).items():
            path = DATA / name
            if path.exists() and file_sha256(path) != digest:
                bad_sha.append(name)
    check("V9-bytes-sha", bad_sha)
    return {"ok": all(v["ok"] for v in checks.values()), "checks": checks,
            "counts": {"groups": len(groups), "tasks": len(tasks), "banks": len(banks),
                       "slot_roots": len(titles)}}


def selftest() -> dict:
    base = verify()
    assert base["ok"], base["checks"]
    import copy
    groups = jsonl_read(DATA / "groups.jsonl")
    tasks = jsonl_read(DATA / "tasks.jsonl")
    banks = jsonl_read(DATA / "banks.jsonl")

    def run(groups2, tasks2, banks2, manifest=None):
                                   
        checks = {}
        def check(code, errs): checks[code] = {"ok": not errs, "n": len(errs)}
        check("V2", [g["group_id"] for g in groups2
                     if g["compound"]["text"] != "；".join(a["text"] for a in g["atoms"]) + "。"])
        check("V3", [g["group_id"] for g in groups2 for i, a in enumerate(g["atoms"])
                     if g["sources"][i]["quote"] != a["text"]])
        check("V4", [g["group_id"] for g in groups2 if any(a["links"] for a in g["atoms"])])
        check("V7", [t["task_id"] for t in tasks2 if "options" in t])
        roots = defaultdict(list)
        for g in groups2:
            roots[g["slot_root"]].append(g["group_id"])
        check("V6", [k for k, v in roots.items() if len(v) > 1])
        return checks

    g_bad = copy.deepcopy(groups); g_bad[0]["atoms"][0]["links"] = ["x"]
    assert not run(g_bad, tasks, banks)["V4"]["ok"]
    g_bad = copy.deepcopy(groups); g_bad[1]["slot_root"] = g_bad[0]["slot_root"]
    assert not run(g_bad, tasks, banks)["V6"]["ok"]
    t_bad = copy.deepcopy(tasks); t_bad[0]["options"] = ["a"]
    assert not run(groups, t_bad, banks)["V7"]["ok"]
    g_bad = copy.deepcopy(groups); g_bad[0]["compound"]["text"] += "X"
    assert not run(g_bad, tasks, banks)["V2"]["ok"]
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    manifest["sha256"]["groups.jsonl"] = "0" * 64
    bad = [n for n, d in manifest["sha256"].items()
           if (DATA / n).exists() and file_sha256(DATA / n) != d]
    assert bad == ["groups.jsonl"]
    return {"ok": True, "checks": {k: v["ok"] for k, v in base["checks"].items()},
            "negative": ["V4-links", "V6-slot-root", "V7-options", "V2-compound",
                         "V9-bytes-sha"]}


def build() -> dict:
    v2_groups, v2_tasks = _load_v2()
    groups = [_group(gid, v2_groups[gid]) for gid in sorted(v2_groups)]
    by_id = {g["group_id"]: g for g in groups}
    tasks = _tasks(v2_tasks, by_id)
    banks = _build_banks(groups, tasks)
    id_map = _build_id_map(groups)
    DATA.mkdir(parents=True, exist_ok=True)
    jsonl_write(DATA / "groups.jsonl", groups)
    jsonl_write(DATA / "tasks.jsonl", tasks)
    jsonl_write(DATA / "banks.jsonl", banks)
    json_write(DATA / "id_map.json", id_map)
    json_write(DATA / "manifest.json", {
        "schema": "disclosure-benchmark-v3",
        "counts": {"groups": len(groups), "tasks": len(tasks), "banks": len(banks)},
        "sha256": {n: file_sha256(DATA / n)
                   for n in ("groups.jsonl", "tasks.jsonl", "banks.jsonl", "id_map.json")},
        "authoring": {"author": "AI 子 Agent（代理证据）",
                      "review": "second independent semantic review pending"},
        "changes": ["no options", "ask_user value=null", "per-atom sources",
                    "visible links empty", "opaque public ids", "user scope",
                    "permanence persistent", "nested banks S8⊂S24⊂S48",
                    "current time in every positive task"],
    })
    report = verify()
    return {"ok": report["ok"], "counts": report["counts"],
            "checks": {k: v["ok"] for k, v in report["checks"].items()}}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "build":
        print(json.dumps(build(), ensure_ascii=False, indent=2))
    elif cmd == "selftest":
        print(json.dumps(selftest(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
