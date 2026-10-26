                      
"""第 4 线配对库与结构化任务：规格化写作 + 内容等价校验 + 数据构建。

方法声明（必须随数据一起读）：
- **不是**实时 LLM 原子化器的产物。复合版与原子版由同一份「子句→事实」规格生成：
  复合版 = 子句按序以「；」连接；原子版 = 每个子句一条条目（逐字引用）。
  这与 `原子化前后配对消融-2026-10-25` 的规格原子做法同源，而不是把有缺陷的抽取器
  混进「存储粒度」主效应。
- 等价性不是靠「同源生成」自证：`verify` 独立重算子句覆盖、残余文本、事实分配、
  极性词表、时间表达、来源链接与 profile 维度集合，任一项不符即失败。
- 作者与审核均为代理（AI），不是 human gold；正式结论须主 Agent/人工复核。

命令：
  python paired_library_20261025.py selftest
  python paired_library_20261025.py build
  python paired_library_20261025.py verify
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from disclosure_schema_20261025 import (              
    BUDGET_CANDIDATES, CONDITION_MARKERS, DATA, DECISIONS, EXCEPTION_MARKERS, FACT_TYPES,
    OUT, PHENOMENA, POLARITIES, POLARITY_MARKERS, PROFILE_BY_PHENOMENON, SPLIT_SCALES,
    TIME_RE, bigrams, estimate_tokens, json_write, jsonl_write, load_groups, load_tasks,
    sha256, stable_hash,
)

DATE_RE = re.compile(r"\d{4}-\d{2}")
CONNECTOR_CHARS = set("，。；：、（）()「」“”\"' \n\t,.;:!?！？-—…·/|+&")
CONNECTOR_WORDS = ("且", "但", "并", "同时", "另外", "以及", "则", "和", "或")
DIMENSION_BY_PROFILE = {
    "single_fact": {"obligation", "source"},
    "multi_obligation": {"obligation", "source"},
    "conditional_exception": {"obligation", "condition", "exception", "source"},
    "conflict_update": {"obligation", "time", "source"},
    "negation": {"polarity", "source"},
    "distractor": {"obligation", "source"},
}
ARM_TOKENS = ("P-full", "A-full", "P-retrieve", "A-retrieve", "A-graph", "A-compiled")
VERIFY_CHECK_IDS = (
    "V1-ids", "V2-splits", "V3-compound", "V4-atom-spans", "V5-residual-only-connectors",
    "V6-fact-partition", "V7-dimension-coverage", "V8-polarity-markers",
    "V9-time-expressions", "V10-source-links", "V11-atom-substring-only",
    "V12-text-hygiene", "V13-task-schema", "V14-split-and-rotation", "V15-negative-controls",
    "V16-duplicate-audit", "V17-arm-leak", "V21-bank-position",
)
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9._\-]{12,}"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),              
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),            
    re.compile(r"(?i)\b(api[_-]?key|secret|token)\s*[:=]"),
)


                                                                  

def grp(gid: str, profile: str, slot: str, trigger: str, clauses: list[str],
        facts: list[tuple], sources: list[tuple], *, mode: str | None = None) -> dict:
    """facts 元组：(key, type, probe, value, polarity, time_value, source_key, atom_index)。

    来源事实由 sources 自动派生（quote 必须落在某一子句内），避免手写漏项。
    """
    assert gid.count("-") == 1 and profile in DIMENSION_BY_PROFILE
    assert 1 <= len(clauses) <= 4, gid
    assert len(set(clauses)) == len(clauses), f"{gid} 子句重复"
    text = "；".join(clauses) + "。"
    built_sources = [{"source_id": f"{gid}-{skey}", "title": title, "quote": quote}
                     for skey, title, quote in sources]
    built_facts = []
    for key, ftype, probe, value, polarity, time_value, skey, atom_index in facts:
        assert ftype in FACT_TYPES and ftype != "source"
        assert probe in clauses[atom_index], (gid, key, probe)
        assert 0 <= atom_index < len(clauses)
        src_id = f"{gid}-{skey}" if skey else None
        assert src_id is None or any(s["source_id"] == src_id for s in built_sources)
        built_facts.append({
            "fact_id": f"{gid}-{key}", "type": ftype, "probe": probe,
            "value": value, "polarity": polarity, "time": time_value,
            "source_id": src_id, "atom_index": atom_index})
    for src in built_sources:
        hits = [i for i, c in enumerate(clauses) if src["quote"] in c]
        assert len(hits) == 1, f"{gid} 来源 {src['source_id']} quote 必须恰好落在一个子句"
        built_facts.append({
            "fact_id": f"{gid}-{src['source_id'].split('-')[-1].lower()}_src",
            "type": "source", "probe": src["quote"], "value": src["source_id"],
            "polarity": "neutral", "time": None,
            "source_id": src["source_id"], "atom_index": hits[0]})
    atoms = []
    for i, clause in enumerate(clauses):
        carried = [f["fact_id"] for f in built_facts if f["atom_index"] == i]
        assert carried, f"{gid} 子句 {i} 未承载任何事实"
        links = sorted({f["source_id"] for f in built_facts
                        if f["atom_index"] == i and f["source_id"]})
        atoms.append({
            "id": f"{gid}-A{i + 1}", "scope_level": "project", "project_id": f"proj-{gid[:3]}",
            "kind": "memory", "title": slot, "text": clause, "trigger": trigger,
            "tier": "explicit", "permanence": "temporary" if profile == "conflict_update"
            else "persistent",
            "payload": {}, "links": links, "revision": 1,
            "payload_fact_ids": carried,                  
        })
    return {
        "group_id": gid, "family_id": gid.split("-")[0], "profile": profile,
        "slot": slot, "trigger": trigger, "mode": mode,
        "compound": {
            "id": f"{gid}-P", "scope_level": "project", "project_id": f"proj-{gid[:3]}",
            "kind": "memory", "title": slot, "text": text, "trigger": trigger,
            "tier": "explicit", "permanence": "temporary" if profile == "conflict_update"
            else "persistent",
            "payload": {}, "links": sorted(s["source_id"] for s in built_sources),
            "revision": 1,
        },
        "atoms": atoms,
        "facts": built_facts,
        "sources": built_sources,
    }


def fam(fid: str, theme: str, split: str, groups: list[dict], tasks: list[dict]) -> dict:
    return {"family_id": fid, "theme": theme, "split": split,
            "groups": groups, "tasks": tasks}


TASK_CODE = {"single_fact": "SI", "multi_obligation": "MU",
             "conditional_exception": "CE", "conflict_update": "CU",
             "negation": "NE", "no_relevant": "NO"}


def task(fid: str, phenomenon: str, target: str | None, question: str, *,
         options: list[str] | None = None, expected_value=None, decision: str = "allow",
         condition: str | None = None, exception: str | None = None,
         context: str | None = None, stale_value: str | None = None,
         value_mode: str = "exact", necessary_context: tuple[str, ...] = (),
         forbidden_bank_slots: tuple[str, ...] = ()) -> dict:
    tid = f"{fid}-T{TASK_CODE[phenomenon]}"
    assert decision in DECISIONS
    if expected_value is not None and options is not None:
        if value_mode == "exact":
            assert expected_value in options, (tid, expected_value)
        else:
            assert set(expected_value) <= set(options), (tid, expected_value)
    return {
        "task_id": tid, "family_id": fid, "phenomenon": phenomenon,
        "target_group": target, "question": question, "options": options,
        "expected": {"decision": decision, "value": expected_value,
                     "applied_condition": condition, "applied_exception": exception,
                     "value_mode": value_mode},
        "context": context, "stale_value": stale_value,
        "necessary_context": list(necessary_context),
        "forbidden_bank_slots": list(forbidden_bank_slots),
    }


                                                                                                

F01 = fam("F01", "检索结果与字段", "dev", [
    grp("F01-G1", "single_fact", "结果条数上限", "用户要求检索公开数据集时",
        ["单次最多返回 5 条结果", "项目手册重复确认：检索结果上限仍为 5 条"],
        [("f1", "obligation", "单次最多返回 5 条结果", "5 条", "require", None, "S1", 0)],
        [("S1", "检索接口约定", "单次最多返回 5 条结果"),
         ("S2", "项目手册", "项目手册重复确认：检索结果上限仍为 5 条")]),
    grp("F01-G2", "multi_obligation", "结果字段要求", "展示检索结果时",
        ["每条结果必须给出数据集标题", "必须给出样本数", "每条结果都必须标注数据来源链接"],
        [("f1", "obligation", "每条结果必须给出数据集标题", "给出数据集标题", "require", None, "S1", 0),
         ("f2", "obligation", "必须给出样本数", "给出样本数", "require", None, "S1", 1),
         ("f3", "obligation", "每条结果都必须标注数据来源链接", "标注数据来源链接", "require", None, "S2", 2)],
        [("S1", "结果字段约定", "每条结果必须给出数据集标题"),
         ("S2", "来源标注要求", "每条结果都必须标注数据来源链接")]),
    grp("F01-G3", "conditional_exception", "结果说明语言", "生成检索结果说明时",
        ["结果说明默认使用中文", "若用户明确要求英文，则改用英文说明", "对外部数据集不适用中文默认"],
        [("f1", "obligation", "结果说明默认使用中文", "中文", "require", None, "S1", 0),
         ("f2", "condition", "若用户明确要求英文", "用户明确要求英文时", "neutral", None, "S2", 1),
         ("f3", "exception", "对外部数据集不适用中文默认", "对外部数据集不适用", "neutral", None, "S2", 2)],
        [("S1", "说明语言默认", "结果说明默认使用中文"),
         ("S2", "说明语言例外", "若用户明确要求英文，则改用英文说明")]),
    grp("F01-G4", "conflict_update", "检索结果排序", "输出检索结果时",
        ["2026-01 至 2026-06 期间按文件名排序", "自 2026-07 起改为按相关性排序"],
        [("f1", "obligation", "2026-01 至 2026-06 期间按文件名排序", "按文件名排序", "neutral", "2026-01..2026-06", "S1", 0),
         ("f2", "time", "2026-01 至 2026-06", "2026-01..2026-06", "neutral", "2026-01..2026-06", "S1", 0),
         ("f3", "obligation", "自 2026-07 起改为按相关性排序", "按相关性排序", "neutral", "2026-07..", "S2", 1),
         ("f4", "time", "自 2026-07 起", "2026-07..", "neutral", "2026-07..", "S2", 1)],
        [("S1", "旧排序规则", "2026-01 至 2026-06 期间按文件名排序"),
         ("S2", "新排序规则", "自 2026-07 起改为按相关性排序")], mode="supersede"),
    grp("F01-G5", "negation", "检索缓存复用", "命中检索缓存时",
        ["禁止复用超过 7 天的检索缓存"],
        [("f1", "polarity", "禁止复用超过 7 天的检索缓存", "forbid", "forbid", None, "S1", 0),
         ("f2", "time", "超过 7 天", "7 天", "neutral", "7 天", "S1", 0)],
        [("S1", "缓存复用限制", "禁止复用超过 7 天的检索缓存")]),
    grp("F01-G6", "distractor", "下载文件命名", "下载数据集文件时",
        ["下载文件时必须使用数据集编号命名"],
        [("f1", "obligation", "下载文件时必须使用数据集编号命名", "数据集编号命名", "require", None, "S1", 0)],
        [("S1", "下载命名规则", "下载文件时必须使用数据集编号命名")]),
], [
    task("F01", "single_fact", "F01-G1", "本次检索公开数据集，单次最多允许返回多少条结果？",
         options=["3 条", "5 条", "10 条", "无上限"], expected_value="5 条"),
    task("F01", "multi_obligation", "F01-G2", "展示检索结果时，必须同时满足哪些字段要求？",
         options=["给出数据集标题", "给出样本数", "标注数据来源链接", "给出 DOI"],
         expected_value=["给出数据集标题", "给出样本数", "标注数据来源链接"], value_mode="set"),
    task("F01", "conditional_exception", "F01-G3",
         "用户明确要求英文说明，目标是内部数据集。结果说明应按哪种方式输出？",
         options=["按默认中文说明", "改用英文说明", "不适用该默认"],
         expected_value="改用英文说明", condition="用户明确要求英文时",
         context="用户明确要求英文，目标是内部数据集。",
         necessary_context=("F01-G3-f2",)),
    task("F01", "conflict_update", "F01-G4",
         "当前是 2026-10，展示检索结果时应按哪种顺序排序？",
         options=["按文件名排序", "按相关性排序"], expected_value="按相关性排序",
         context="当前是 2026-10。", stale_value="按文件名排序",
         necessary_context=("F01-G4-f2", "F01-G4-f4")),
])

F02 = fam("F02", "下载前校验", "dev", [
    grp("F02-G1", "single_fact", "单个文件大小上限", "下载数据集文件时",
        ["单个下载文件最多允许 2 GB"],
        [("f1", "obligation", "单个下载文件最多允许 2 GB", "2 GB", "require", None, "S1", 0)],
        [("S1", "下载体积限制", "单个下载文件最多允许 2 GB")]),
    grp("F02-G2", "multi_obligation", "下载前检查项", "发起下载前",
        ["下载前必须确认文件大小", "必须确认文件校验和", "必须确认目标目录剩余空间"],
        [("f1", "obligation", "下载前必须确认文件大小", "确认文件大小", "require", None, "S1", 0),
         ("f2", "obligation", "必须确认文件校验和", "确认文件校验和", "require", None, "S2", 1),
         ("f3", "obligation", "必须确认目标目录剩余空间", "确认目标目录剩余空间", "require", None, "S2", 2)],
        [("S1", "下载前检查清单", "下载前必须确认文件大小"),
         ("S2", "下载完整性要求", "必须确认文件校验和")]),
    grp("F02-G3", "conditional_exception", "下载确认策略", "准备直接下载时",
        ["默认直接下载", "若目标目录剩余空间不足 5 GB，则先询问用户", "对只读归档目录不适用自动下载"],
        [("f1", "obligation", "默认直接下载", "直接下载", "require", None, "S1", 0),
         ("f2", "condition", "若目标目录剩余空间不足 5 GB", "目标目录剩余空间不足 5 GB", "neutral", None, "S2", 1),
         ("f3", "exception", "对只读归档目录不适用自动下载", "对只读归档目录不适用", "neutral", None, "S2", 2)],
        [("S1", "下载默认策略", "默认直接下载"),
         ("S2", "下载空间条件", "若目标目录剩余空间不足 5 GB，则先询问用户")]),
    grp("F02-G4", "conflict_update", "下载并发数", "同时发起多个下载时",
        ["自 2026-07 起最多允许 2 个并发下载", "自 2026-07 起最多允许 4 个并发下载"],
        [("f1", "obligation", "自 2026-07 起最多允许 2 个并发下载", "2 个", "require", "2026-07..", "S1", 0),
         ("f2", "time", "自 2026-07 起", "2026-07..", "neutral", "2026-07..", "S1", 0),
         ("f3", "obligation", "自 2026-07 起最多允许 4 个并发下载", "4 个", "require", "2026-07..", "S2", 1),
         ("f4", "time", "自 2026-07 起", "2026-07..", "neutral", "2026-07..", "S2", 1)],
        [("S1", "并发下载限制（运维通知）", "自 2026-07 起最多允许 2 个并发下载"),
         ("S2", "并发下载限制（项目约定）", "自 2026-07 起最多允许 4 个并发下载")], mode="simultaneous"),
    grp("F02-G5", "negation", "下载覆盖策略", "目标目录已存在同名文件时",
        ["禁止覆盖已存在的同名下载文件"],
        [("f1", "polarity", "禁止覆盖已存在的同名下载文件", "forbid", "forbid", None, "S1", 0)],
        [("S1", "覆盖禁令", "禁止覆盖已存在的同名下载文件")]),
    grp("F02-G6", "distractor", "引文导出格式", "导出引文时",
        ["导出引文时必须使用 BibTeX 格式"],
        [("f1", "obligation", "导出引文时必须使用 BibTeX 格式", "BibTeX 格式", "require", None, "S1", 0)],
        [("S1", "引文格式约定", "导出引文时必须使用 BibTeX 格式")]),
], [
    task("F02", "single_fact", "F02-G1", "单个下载文件最多允许多大？",
         options=["1 GB", "2 GB", "5 GB"], expected_value="2 GB"),
    task("F02", "multi_obligation", "F02-G2", "发起下载前必须同时确认哪些信息？",
         options=["文件大小", "文件校验和", "目标目录剩余空间", "下载速度"],
         expected_value=["文件大小", "文件校验和", "目标目录剩余空间"], value_mode="set"),
    task("F02", "conditional_exception", "F02-G3",
         "目标目录剩余空间只有 2 GB，且不是只读归档目录。下载前应如何处理？",
         options=["直接下载", "先询问用户", "拒绝下载"], expected_value="先询问用户",
         condition="目标目录剩余空间不足 5 GB",
         context="目标目录剩余空间只有 2 GB，且不是只读归档目录。",
         necessary_context=("F02-G3-f2",)),
    task("F02", "conflict_update", "F02-G4",
         "当前是 2026-10，本次允许几个并发下载？",
         options=["2 个", "4 个", "需要询问用户"], expected_value=None, decision="ask_user",
         context="当前是 2026-10。",
         necessary_context=("F02-G4-f2", "F02-G4-f4")),
    task("F02", "negation", "F02-G5", "目标目录已存在同名文件，是否覆盖？",
         options=["覆盖同名文件", "拒绝覆盖并换名", "询问用户"],
         expected_value="拒绝覆盖并换名", decision="deny"),
    task("F02", "no_relevant", None, "用户要求生成数据集海报，记忆里有适用于本次请求的规则吗？",
         decision="no_memory"),
])

F03 = fam("F03", "引文导出", "dev", [
    grp("F03-G1", "single_fact", "引文条数上限", "导出引文时",
        ["单次导出引文最多 50 条"],
        [("f1", "obligation", "单次导出引文最多 50 条", "50 条", "require", None, "S1", 0)],
        [("S1", "引文导出上限", "单次导出引文最多 50 条")]),
    grp("F03-G2", "multi_obligation", "引文字段要求", "生成引文条目时",
        ["每条引文必须包含数据集标题", "必须包含访问日期", "必须包含持久标识符"],
        [("f1", "obligation", "每条引文必须包含数据集标题", "数据集标题", "require", None, "S1", 0),
         ("f2", "obligation", "必须包含访问日期", "访问日期", "require", None, "S2", 1),
         ("f3", "obligation", "必须包含持久标识符", "持久标识符", "require", None, "S2", 2)],
        [("S1", "引文字段约定", "每条引文必须包含数据集标题"),
         ("S2", "引文持久化要求", "必须包含访问日期")]),
    grp("F03-G3", "conditional_exception", "引文样式", "导出引文样式时",
        ["默认使用 BibTeX 样式", "若用户要求 RIS，则改用 RIS 样式", "对未发布数据集不适用任何引文样式"],
        [("f1", "obligation", "默认使用 BibTeX 样式", "BibTeX 样式", "require", None, "S1", 0),
         ("f2", "condition", "若用户要求 RIS", "用户要求 RIS", "neutral", None, "S2", 1),
         ("f3", "exception", "对未发布数据集不适用任何引文样式", "对未发布数据集不适用任何引文样式", "neutral", None, "S2", 2)],
        [("S1", "引文样式默认", "默认使用 BibTeX 样式"),
         ("S2", "引文样式例外", "若用户要求 RIS，则改用 RIS 样式")]),
    grp("F03-G4", "conflict_update", "引文排序", "导出引文时",
        ["2026-01 至 2026-06 期间按作者排序", "自 2026-07 起改为按访问日期排序"],
        [("f1", "obligation", "2026-01 至 2026-06 期间按作者排序", "按作者排序", "neutral", "2026-01..2026-06", "S1", 0),
         ("f2", "time", "2026-01 至 2026-06", "2026-01..2026-06", "neutral", "2026-01..2026-06", "S1", 0),
         ("f3", "obligation", "自 2026-07 起改为按访问日期排序", "按访问日期排序", "neutral", "2026-07..", "S2", 1),
         ("f4", "time", "自 2026-07 起", "2026-07..", "neutral", "2026-07..", "S2", 1)],
        [("S1", "旧引文排序", "2026-01 至 2026-06 期间按作者排序"),
         ("S2", "新引文排序", "自 2026-07 起改为按访问日期排序")], mode="supersede"),
    grp("F03-G5", "negation", "引文去重", "同一批出现重复条目时",
        ["禁止导出重复的引文条目"],
        [("f1", "polarity", "禁止导出重复的引文条目", "forbid", "forbid", None, "S1", 0)],
        [("S1", "引文去重规则", "禁止导出重复的引文条目")]),
    grp("F03-G6", "distractor", "上传来源登记", "上传数据时",
        ["上传数据时必须登记来源机构"],
        [("f1", "obligation", "上传数据时必须登记来源机构", "来源机构", "require", None, "S1", 0)],
        [("S1", "上传登记要求", "上传数据时必须登记来源机构")]),
], [
    task("F03", "single_fact", "F03-G1", "单次最多导出多少条引文？",
         options=["20 条", "50 条", "100 条"], expected_value="50 条"),
    task("F03", "multi_obligation", "F03-G2", "每条引文必须同时包含哪些信息？",
         options=["数据集标题", "访问日期", "持久标识符", "作者邮箱"],
         expected_value=["数据集标题", "访问日期", "持久标识符"], value_mode="set"),
    task("F03", "conditional_exception", "F03-G3",
         "目标是未发布数据集，用户没有要求 RIS。引文应按哪种样式导出？",
         options=["使用 BibTeX 样式", "改用 RIS 样式", "不适用任何引文样式"],
         expected_value="不适用任何引文样式",
         exception="对未发布数据集不适用任何引文样式",
         context="目标是未发布数据集，用户没有要求 RIS。",
         necessary_context=("F03-G3-f3",)),
    task("F03", "conflict_update", "F03-G4",
         "当前是 2026-10，导出引文时按哪种顺序排序？",
         options=["按作者排序", "按访问日期排序"], expected_value="按访问日期排序",
         context="当前是 2026-10。", stale_value="按作者排序",
         necessary_context=("F03-G4-f2", "F03-G4-f4")),
    task("F03", "negation", "F03-G5", "同一数据集的引文在本批中重复出现，是否重复导出引文条目？",
         options=["重复导出", "只导出一次", "询问用户"],
         expected_value="只导出一次", decision="deny"),
    task("F03", "no_relevant", None, "用户要求把数据集导出为 SPSS 文件，记忆里有适用于本次请求的规则吗？",
         decision="no_memory"),
])

F04 = fam("F04", "上传数据登记", "dev", [
    grp("F04-G1", "single_fact", "上传文件数上限", "上传数据集文件时",
        ["单次最多上传 20 个文件", "来源登记表重复确认：单次上传上限仍为 20 个文件"],
        [("f1", "obligation", "单次最多上传 20 个文件", "20 个", "require", None, "S1", 0)],
        [("S1", "上传数量限制", "单次最多上传 20 个文件"),
         ("S2", "来源登记表", "来源登记表重复确认：单次上传上限仍为 20 个文件")]),
    grp("F04-G2", "multi_obligation", "上传登记字段", "登记上传数据时",
        ["上传时必须登记数据集标题", "必须登记样本物种", "必须登记数据许可"],
        [("f1", "obligation", "上传时必须登记数据集标题", "数据集标题", "require", None, "S1", 0),
         ("f2", "obligation", "必须登记样本物种", "样本物种", "require", None, "S2", 1),
         ("f3", "obligation", "必须登记数据许可", "数据许可", "require", None, "S2", 2)],
        [("S1", "上传登记清单", "上传时必须登记数据集标题"),
         ("S2", "上传元数据要求", "必须登记样本物种")]),
    grp("F04-G3", "conditional_exception", "上传审核策略", "上传者提交数据后",
        ["默认进入待审核状态", "若上传者属于同一项目，则自动通过审核", "对外部贡献者不适用自动通过"],
        [("f1", "obligation", "默认进入待审核状态", "进入待审核", "require", None, "S1", 0),
         ("f2", "condition", "若上传者属于同一项目", "上传者属于同一项目", "neutral", None, "S2", 1),
         ("f3", "exception", "对外部贡献者不适用自动通过", "对外部贡献者不适用自动通过", "neutral", None, "S2", 2)],
        [("S1", "审核默认状态", "默认进入待审核状态"),
         ("S2", "同项目审核例外", "若上传者属于同一项目，则自动通过审核")]),
    grp("F04-G4", "conflict_update", "上传文件大小上限", "上传单个文件时",
        ["2026-01 至 2026-06 期间单个文件最大 5 GB", "自 2026-07 起单个文件最大 2 GB"],
        [("f1", "obligation", "2026-01 至 2026-06 期间单个文件最大 5 GB", "5 GB", "neutral", "2026-01..2026-06", "S1", 0),
         ("f2", "time", "2026-01 至 2026-06", "2026-01..2026-06", "neutral", "2026-01..2026-06", "S1", 0),
         ("f3", "obligation", "自 2026-07 起单个文件最大 2 GB", "2 GB", "neutral", "2026-07..", "S2", 1),
         ("f4", "time", "自 2026-07 起", "2026-07..", "neutral", "2026-07..", "S2", 1)],
        [("S1", "旧上传体积限制", "2026-01 至 2026-06 期间单个文件最大 5 GB"),
         ("S2", "新上传体积限制", "自 2026-07 起单个文件最大 2 GB")], mode="supersede"),
    grp("F04-G5", "negation", "数据集覆盖策略", "目标项目已存在同名数据集时",
        ["禁止覆盖已存在的同名数据集"],
        [("f1", "polarity", "禁止覆盖已存在的同名数据集", "forbid", "forbid", None, "S1", 0)],
        [("S1", "数据集覆盖禁令", "禁止覆盖已存在的同名数据集")]),
    grp("F04-G6", "distractor", "检索结果排序", "展示检索结果时",
        ["展示检索结果时必须按相关性排序"],
        [("f1", "obligation", "展示检索结果时必须按相关性排序", "按相关性排序", "require", None, "S1", 0)],
        [("S1", "结果排序要求", "展示检索结果时必须按相关性排序")]),
], [
    task("F04", "single_fact", "F04-G1", "单次最多上传多少个文件？",
         options=["10 个", "20 个", "50 个"], expected_value="20 个"),
    task("F04", "multi_obligation", "F04-G2", "登记上传数据时必须同时登记哪些信息？",
         options=["数据集标题", "样本物种", "数据许可", "文件格式"],
         expected_value=["数据集标题", "样本物种", "数据许可"], value_mode="set"),
    task("F04", "conditional_exception", "F04-G3",
         "上传者属于同一项目，且不是外部贡献者。上传后应如何处理？",
         options=["进入待审核", "自动通过审核", "拒绝上传"], expected_value="自动通过审核",
         condition="上传者属于同一项目",
         context="上传者属于同一项目，且不是外部贡献者。",
         necessary_context=("F04-G3-f2",)),
    task("F04", "conflict_update", "F04-G4",
         "当前是 2026-10，单个上传文件最多多大？",
         options=["5 GB", "2 GB"], expected_value="2 GB",
         context="当前是 2026-10。", stale_value="5 GB",
         necessary_context=("F04-G4-f2", "F04-G4-f4")),
    task("F04", "negation", "F04-G5", "目标项目已存在同名数据集，是否覆盖？",
         options=["覆盖同名数据集", "拒绝覆盖并登记新版本", "询问用户"],
         expected_value="拒绝覆盖并登记新版本", decision="deny"),
    task("F04", "no_relevant", None, "用户要求生成数据集的引用网络图，记忆里有适用于本次请求的规则吗？",
         decision="no_memory"),
])

F05 = fam("F05", "项目语言与术语", "final", [
    grp("F05-G1", "single_fact", "回复字数上限", "生成回复正文时",
        ["回复正文最多 800 字"],
        [("f1", "obligation", "回复正文最多 800 字", "800 字", "require", None, "S1", 0)],
        [("S1", "回复长度约定", "回复正文最多 800 字")]),
    grp("F05-G2", "multi_obligation", "回复术语要求", "生成项目回复时",
        ["回复必须使用项目术语表", "必须给出数据集编号", "必须避免使用内部批次号"],
        [("f1", "obligation", "回复必须使用项目术语表", "使用项目术语表", "require", None, "S1", 0),
         ("f2", "obligation", "必须给出数据集编号", "给出数据集编号", "require", None, "S2", 1),
         ("f3", "obligation", "必须避免使用内部批次号", "避免内部批次号", "require", None, "S2", 2)],
        [("S1", "项目术语约定", "回复必须使用项目术语表"),
         ("S2", "编号披露约定", "必须给出数据集编号")]),
    grp("F05-G3", "conditional_exception", "回复语言", "生成回复时",
        ["默认使用中文回复", "若用户用英文提问，则改用英文回复", "对公开交付材料不适用中文默认"],
        [("f1", "obligation", "默认使用中文回复", "中文", "require", None, "S1", 0),
         ("f2", "condition", "若用户用英文提问", "用户用英文提问时", "neutral", None, "S2", 1),
         ("f3", "exception", "对公开交付材料不适用中文默认", "对公开交付材料不适用", "neutral", None, "S2", 2)],
        [("S1", "回复语言默认", "默认使用中文回复"),
         ("S2", "回复语言条件", "若用户用英文提问，则改用英文回复")]),
    grp("F05-G4", "conflict_update", "回复字数上限（历史）", "生成回复正文时",
        ["2026-01 至 2026-06 期间回复最多 400 字", "自 2026-07 起回复最多 800 字"],
        [("f1", "obligation", "2026-01 至 2026-06 期间回复最多 400 字", "400 字", "neutral", "2026-01..2026-06", "S1", 0),
         ("f2", "time", "2026-01 至 2026-06", "2026-01..2026-06", "neutral", "2026-01..2026-06", "S1", 0),
         ("f3", "obligation", "自 2026-07 起回复最多 800 字", "800 字", "neutral", "2026-07..", "S2", 1),
         ("f4", "time", "自 2026-07 起", "2026-07..", "neutral", "2026-07..", "S2", 1)],
        [("S1", "旧回复长度", "2026-01 至 2026-06 期间回复最多 400 字"),
         ("S2", "新回复长度", "自 2026-07 起回复最多 800 字")], mode="supersede"),
    grp("F05-G5", "negation", "内部编号引用", "撰写回复内容时",
        ["禁止在回复中引用内部批次号"],
        [("f1", "polarity", "禁止在回复中引用内部批次号", "forbid", "forbid", None, "S1", 0)],
        [("S1", "内部编号禁令", "禁止在回复中引用内部批次号")]),
    grp("F05-G6", "distractor", "下载校验", "下载数据集文件时",
        ["下载前必须确认文件校验和"],
        [("f1", "obligation", "下载前必须确认文件校验和", "确认文件校验和", "require", None, "S1", 0)],
        [("S1", "下载校验要求", "下载前必须确认文件校验和")]),
], [
    task("F05", "single_fact", "F05-G1", "回复正文最多多少字？",
         options=["400 字", "800 字", "1200 字"], expected_value="800 字"),
    task("F05", "multi_obligation", "F05-G2", "生成项目回复时必须同时满足哪些要求？",
         options=["使用项目术语表", "给出数据集编号", "避免内部批次号", "给出作者邮箱"],
         expected_value=["使用项目术语表", "给出数据集编号", "避免内部批次号"], value_mode="set"),
    task("F05", "conditional_exception", "F05-G3",
         "用户用英文提问，目标是内部项目。回复应使用哪种语言？",
         options=["中文", "英文", "不适用该默认"], expected_value="英文",
         condition="用户用英文提问时", context="用户用英文提问，目标是内部项目。",
         necessary_context=("F05-G3-f2",)),
    task("F05", "conflict_update", "F05-G4",
         "当前是 2026-10，回复正文最多多少字？",
         options=["400 字", "800 字"], expected_value="800 字",
         context="当前是 2026-10。", stale_value="400 字",
         necessary_context=("F05-G4-f2", "F05-G4-f4")),
    task("F05", "negation", "F05-G5", "回复中是否可以直接写出内部批次号？",
         options=["可以引用", "不得引用，改用公开编号", "仅在内部讨论中引用"],
         expected_value="不得引用，改用公开编号", decision="deny"),
    task("F05", "no_relevant", None, "用户要求把回复翻译成日文，记忆里有适用于本次请求的规则吗？",
         decision="no_memory"),
])

F06 = fam("F06", "样本物种过滤", "final", [
    grp("F06-G1", "single_fact", "物种覆盖上限", "检索单细胞数据集时",
        ["单次检索最多覆盖 3 个物种"],
        [("f1", "obligation", "单次检索最多覆盖 3 个物种", "3 个", "require", None, "S1", 0)],
        [("S1", "物种覆盖限制", "单次检索最多覆盖 3 个物种")]),
    grp("F06-G2", "multi_obligation", "结果标注字段", "展示单细胞检索结果时",
        ["结果必须标注物种", "必须标注组织来源", "必须标注测序平台"],
        [("f1", "obligation", "结果必须标注物种", "标注物种", "require", None, "S1", 0),
         ("f2", "obligation", "必须标注组织来源", "标注组织来源", "require", None, "S2", 1),
         ("f3", "obligation", "必须标注测序平台", "标注测序平台", "require", None, "S2", 2)],
        [("S1", "物种标注要求", "结果必须标注物种"),
         ("S2", "实验元数据要求", "必须标注组织来源")]),
    grp("F06-G3", "conditional_exception", "混合样本过滤", "过滤单细胞样本时",
        ["默认排除跨物种混合样本", "若用户明确研究混合样本，则保留混合样本", "对临床来源数据不适用自动排除"],
        [("f1", "obligation", "默认排除跨物种混合样本", "排除混合样本", "require", None, "S1", 0),
         ("f2", "condition", "若用户明确研究混合样本", "用户明确研究混合样本", "neutral", None, "S2", 1),
         ("f3", "exception", "对临床来源数据不适用自动排除", "对临床来源数据不适用", "neutral", None, "S2", 2)],
        [("S1", "混合样本默认", "默认排除跨物种混合样本"),
         ("S2", "混合样本例外", "若用户明确研究混合样本，则保留混合样本")]),
    grp("F06-G4", "conflict_update", "默认物种过滤", "未指定物种时",
        ["自 2026-07 起默认只返回人类数据", "自 2026-07 起默认只返回小鼠数据"],
        [("f1", "obligation", "自 2026-07 起默认只返回人类数据", "人类", "neutral", "2026-07..", "S1", 0),
         ("f2", "time", "自 2026-07 起", "2026-07..", "neutral", "2026-07..", "S1", 0),
         ("f3", "obligation", "自 2026-07 起默认只返回小鼠数据", "小鼠", "neutral", "2026-07..", "S2", 1),
         ("f4", "time", "自 2026-07 起", "2026-07..", "neutral", "2026-07..", "S2", 1)],
        [("S1", "物种默认（检索配置）", "自 2026-07 起默认只返回人类数据"),
         ("S2", "物种默认（项目配置）", "自 2026-07 起默认只返回小鼠数据")], mode="simultaneous"),
    grp("F06-G5", "negation", "物种推断", "读取数据文件属性时",
        ["禁止根据文件名推断物种"],
        [("f1", "polarity", "禁止根据文件名推断物种", "forbid", "forbid", None, "S1", 0)],
        [("S1", "物种推断禁令", "禁止根据文件名推断物种")]),
    grp("F06-G6", "distractor", "引文日期", "导出引文时",
        ["导出引文时必须包含访问日期"],
        [("f1", "obligation", "导出引文时必须包含访问日期", "访问日期", "require", None, "S1", 0)],
        [("S1", "引文日期要求", "导出引文时必须包含访问日期")]),
], [
    task("F06", "single_fact", "F06-G1", "单次检索最多覆盖几个物种？",
         options=["2 个", "3 个", "5 个"], expected_value="3 个"),
    task("F06", "multi_obligation", "F06-G2", "展示单细胞检索结果时必须同时标注哪些信息？",
         options=["物种", "组织来源", "测序平台", "作者单位"],
         expected_value=["物种", "组织来源", "测序平台"], value_mode="set"),
    task("F06", "conditional_exception", "F06-G3",
         "用户明确研究跨物种混合样本，数据来自公共数据库。应如何处理混合样本？",
         options=["排除混合样本", "保留混合样本", "不适用该默认"],
         expected_value="保留混合样本", condition="用户明确研究混合样本",
         context="用户明确研究跨物种混合样本，数据来自公共数据库。",
         necessary_context=("F06-G3-f2",)),
    task("F06", "conflict_update", "F06-G4",
         "当前是 2026-10，默认只返回哪个物种的数据？",
         options=["人类", "小鼠", "需要询问用户"], expected_value=None, decision="ask_user",
         context="当前是 2026-10。",
         necessary_context=("F06-G4-f2", "F06-G4-f4")),
    task("F06", "negation", "F06-G5", "文件名包含 mouse 时，是否可以直接推断为小鼠数据？",
         options=["可以推断", "不得推断，读取元数据", "询问用户"],
         expected_value="不得推断，读取元数据", decision="deny"),
    task("F06", "no_relevant", None, "用户要求给数据集起一个中文别名，记忆里有适用于本次请求的规则吗？",
         decision="no_memory"),
])

F07 = fam("F07", "排序与偏好", "final", [
    grp("F07-G1", "single_fact", "默认返回条数", "用户未指定条数时",
        ["未指定条数时默认返回 10 条结果", "偏好表重复确认：默认返回条数仍为 10 条"],
        [("f1", "obligation", "未指定条数时默认返回 10 条结果", "10 条", "require", None, "S1", 0)],
        [("S1", "检索默认条数", "未指定条数时默认返回 10 条结果"),
         ("S2", "偏好表", "偏好表重复确认：默认返回条数仍为 10 条")]),
    grp("F07-G2", "multi_obligation", "排序依据", "对检索结果排序时",
        ["排序必须优先考虑相关性", "其次考虑数据量", "最后考虑更新时间"],
        [("f1", "obligation", "排序必须优先考虑相关性", "相关性", "require", None, "S1", 0),
         ("f2", "obligation", "其次考虑数据量", "数据量", "neutral", None, "S2", 1),
         ("f3", "obligation", "最后考虑更新时间", "更新时间", "neutral", None, "S2", 2)],
        [("S1", "排序主依据", "排序必须优先考虑相关性"),
         ("S2", "排序次依据", "其次考虑数据量")]),
    grp("F07-G3", "conditional_exception", "默认排序方式", "输出检索结果时",
        ["默认按相关性排序", "若用户指定按数据量排序，则改按数据量排序", "对精选数据集不适用该默认排序"],
        [("f1", "obligation", "默认按相关性排序", "按相关性排序", "require", None, "S1", 0),
         ("f2", "condition", "若用户指定按数据量排序", "用户指定按数据量排序", "neutral", None, "S2", 1),
         ("f3", "exception", "对精选数据集不适用该默认排序", "对精选数据集不适用该默认排序", "neutral", None, "S2", 2)],
        [("S1", "默认排序", "默认按相关性排序"),
         ("S2", "排序覆盖条件", "若用户指定按数据量排序，则改按数据量排序")]),
    grp("F07-G4", "conflict_update", "默认排序（历史）", "用户未指定排序时",
        ["2026-01 至 2026-06 期间默认按更新时间排序", "自 2026-07 起默认按相关性排序"],
        [("f1", "obligation", "2026-01 至 2026-06 期间默认按更新时间排序", "按更新时间排序", "neutral", "2026-01..2026-06", "S1", 0),
         ("f2", "time", "2026-01 至 2026-06", "2026-01..2026-06", "neutral", "2026-01..2026-06", "S1", 0),
         ("f3", "obligation", "自 2026-07 起默认按相关性排序", "按相关性排序", "neutral", "2026-07..", "S2", 1),
         ("f4", "time", "自 2026-07 起", "2026-07..", "neutral", "2026-07..", "S2", 1)],
        [("S1", "旧默认排序", "2026-01 至 2026-06 期间默认按更新时间排序"),
         ("S2", "新默认排序", "自 2026-07 起默认按相关性排序")], mode="supersede"),
    grp("F07-G5", "negation", "付费来源排序", "对检索结果排序时",
        ["禁止因付费来源提高排序"],
        [("f1", "polarity", "禁止因付费来源提高排序", "forbid", "forbid", None, "S1", 0)],
        [("S1", "付费排序禁令", "禁止因付费来源提高排序")]),
    grp("F07-G6", "distractor", "上传审核状态", "上传数据后",
        ["上传数据默认进入待审核状态"],
        [("f1", "obligation", "上传数据默认进入待审核状态", "进入待审核", "require", None, "S1", 0)],
        [("S1", "上传审核默认", "上传数据默认进入待审核状态")]),
], [
    task("F07", "single_fact", "F07-G1", "未指定条数时默认返回多少条结果？",
         options=["5 条", "10 条", "20 条"], expected_value="10 条"),
    task("F07", "multi_obligation", "F07-G2", "对检索结果排序时必须同时考虑哪些依据？",
         options=["相关性", "数据量", "更新时间", "文件格式"],
         expected_value=["相关性", "数据量", "更新时间"], value_mode="set"),
    task("F07", "conditional_exception", "F07-G3",
         "目标是精选数据集，用户没有指定排序。应使用哪种排序？",
         options=["按相关性排序", "按数据量排序", "不适用该默认排序"],
         expected_value="不适用该默认排序", exception="对精选数据集不适用该默认排序",
         context="目标是精选数据集，用户没有指定排序。",
         necessary_context=("F07-G3-f3",)),
    task("F07", "conflict_update", "F07-G4",
         "当前是 2026-10，默认按哪种依据排序？",
         options=["按更新时间排序", "按相关性排序"], expected_value="按相关性排序",
         context="当前是 2026-10。", stale_value="按更新时间排序",
         necessary_context=("F07-G4-f2", "F07-G4-f4")),
    task("F07", "negation", "F07-G5", "付费来源的数据集是否可以因此提高排序？",
         options=["可以提高排序", "不得提高排序", "询问用户"],
         expected_value="不得提高排序", decision="deny"),
    task("F07", "no_relevant", None, "用户要求给检索结果加上音频朗读，记忆里有适用于本次请求的规则吗？",
         decision="no_memory"),
])

F08 = fam("F08", "缓存与重建", "final", [
    grp("F08-G1", "single_fact", "检索缓存有效期", "复用检索缓存时",
        ["检索缓存最多保留 24 小时"],
        [("f1", "obligation", "检索缓存最多保留 24 小时", "24 小时", "require", None, "S1", 0)],
        [("S1", "缓存有效期", "检索缓存最多保留 24 小时")]),
    grp("F08-G2", "multi_obligation", "缓存重建触发", "维护检索缓存时",
        ["缓存过期时必须重建", "索引变更时必须重建", "数据更新时必须重建"],
        [("f1", "obligation", "缓存过期时必须重建", "缓存过期", "require", None, "S1", 0),
         ("f2", "obligation", "索引变更时必须重建", "索引变更", "require", None, "S2", 1),
         ("f3", "obligation", "数据更新时必须重建", "数据更新", "require", None, "S2", 2)],
        [("S1", "缓存过期重建", "缓存过期时必须重建"),
         ("S2", "缓存一致性要求", "索引变更时必须重建")]),
    grp("F08-G3", "conditional_exception", "缓存复用条件", "准备复用缓存时",
        ["默认复用未过期缓存", "若索引在本轮发生变更，则跳过缓存", "对只读历史快照不适用缓存重建"],
        [("f1", "obligation", "默认复用未过期缓存", "复用缓存", "require", None, "S1", 0),
         ("f2", "condition", "若索引在本轮发生变更", "索引在本轮发生变更", "neutral", None, "S2", 1),
         ("f3", "exception", "对只读历史快照不适用缓存重建", "对只读历史快照不适用缓存重建", "neutral", None, "S2", 2)],
        [("S1", "缓存复用默认", "默认复用未过期缓存"),
         ("S2", "缓存跳过条件", "若索引在本轮发生变更，则跳过缓存")]),
    grp("F08-G4", "conflict_update", "缓存有效期（双来源）", "复用检索缓存时",
        ["自 2026-07 起缓存最多保留 6 小时", "自 2026-07 起缓存最多保留 12 小时"],
        [("f1", "obligation", "自 2026-07 起缓存最多保留 6 小时", "6 小时", "require", "2026-07..", "S1", 0),
         ("f2", "time", "自 2026-07 起", "2026-07..", "neutral", "2026-07..", "S1", 0),
         ("f3", "obligation", "自 2026-07 起缓存最多保留 12 小时", "12 小时", "require", "2026-07..", "S2", 1),
         ("f4", "time", "自 2026-07 起", "2026-07..", "neutral", "2026-07..", "S2", 1)],
        [("S1", "缓存有效期（运维）", "自 2026-07 起缓存最多保留 6 小时"),
         ("S2", "缓存有效期（项目）", "自 2026-07 起缓存最多保留 12 小时")], mode="simultaneous"),
    grp("F08-G5", "negation", "静默重建", "用户请求进行中",
        ["禁止在用户请求进行中静默重建索引"],
        [("f1", "polarity", "禁止在用户请求进行中静默重建索引", "forbid", "forbid", None, "S1", 0)],
        [("S1", "重建禁令", "禁止在用户请求进行中静默重建索引")]),
    grp("F08-G6", "distractor", "引文去重", "导出引文时",
        ["导出引文时必须去重，每条引文只保留一次"],
        [("f1", "obligation", "导出引文时必须去重，每条引文只保留一次", "每条引文只保留一次", "require", None, "S1", 0)],
        [("S1", "引文去重要求", "导出引文时必须去重，每条引文只保留一次")]),
], [
    task("F08", "single_fact", "F08-G1", "检索缓存最多保留多久？",
         options=["6 小时", "24 小时", "48 小时"], expected_value="24 小时"),
    task("F08", "multi_obligation", "F08-G2", "哪些情况必须重建检索缓存？",
         options=["缓存过期", "索引变更", "数据更新", "用户切换项目"],
         expected_value=["缓存过期", "索引变更", "数据更新"], value_mode="set"),
    task("F08", "conditional_exception", "F08-G3",
         "索引在本轮发生变更，且不是只读历史快照。应如何处理缓存？",
         options=["复用缓存", "跳过缓存", "不适用缓存重建"],
         expected_value="跳过缓存", condition="索引在本轮发生变更",
         context="索引在本轮发生变更，且不是只读历史快照。",
         necessary_context=("F08-G3-f2",)),
    task("F08", "conflict_update", "F08-G4",
         "当前是 2026-10，检索缓存最多保留多久？",
         options=["6 小时", "12 小时", "需要询问用户"], expected_value=None, decision="ask_user",
         context="当前是 2026-10。",
         necessary_context=("F08-G4-f2", "F08-G4-f4")),
    task("F08", "negation", "F08-G5", "用户请求进行中，是否静默重建索引？",
         options=["静默重建", "等请求结束后重建", "询问用户"],
         expected_value="等请求结束后重建", decision="deny"),
    task("F08", "no_relevant", None, "用户要求把缓存目录迁移到新磁盘，记忆里有适用于本次请求的规则吗？",
         decision="no_memory"),
])

F09 = fam("F09", "任务包与反馈", "final", [
    grp("F09-G1", "single_fact", "任务包大小上限", "打包反馈任务包时",
        ["单个任务包最多 500 MB"],
        [("f1", "obligation", "单个任务包最多 500 MB", "500 MB", "require", None, "S1", 0)],
        [("S1", "任务包体积限制", "单个任务包最多 500 MB")]),
    grp("F09-G2", "multi_obligation", "任务包内容", "打包反馈任务包时",
        ["任务包必须包含清单文件", "必须包含校验和", "必须包含来源说明"],
        [("f1", "obligation", "任务包必须包含清单文件", "清单文件", "require", None, "S1", 0),
         ("f2", "obligation", "必须包含校验和", "校验和", "require", None, "S2", 1),
         ("f3", "obligation", "必须包含来源说明", "来源说明", "require", None, "S2", 2)],
        [("S1", "任务包清单要求", "任务包必须包含清单文件"),
         ("S2", "任务包完整性要求", "必须包含校验和")]),
    grp("F09-G3", "conditional_exception", "反馈打包范围", "打包运行反馈时",
        ["默认打包最近 3 次运行", "若用户要求全部运行，则打包全部运行", "对失败运行不适用默认打包范围"],
        [("f1", "obligation", "默认打包最近 3 次运行", "最近 3 次运行", "require", None, "S1", 0),
         ("f2", "condition", "若用户要求全部运行", "用户要求全部运行时", "neutral", None, "S2", 1),
         ("f3", "exception", "对失败运行不适用默认打包范围", "对失败运行不适用默认打包范围", "neutral", None, "S2", 2)],
        [("S1", "反馈打包默认", "默认打包最近 3 次运行"),
         ("S2", "反馈打包例外", "若用户要求全部运行，则打包全部运行")]),
    grp("F09-G4", "conflict_update", "任务包大小（历史）", "打包反馈任务包时",
        ["2026-01 至 2026-06 期间任务包最多 1 GB", "自 2026-07 起任务包最多 500 MB"],
        [("f1", "obligation", "2026-01 至 2026-06 期间任务包最多 1 GB", "1 GB", "neutral", "2026-01..2026-06", "S1", 0),
         ("f2", "time", "2026-01 至 2026-06", "2026-01..2026-06", "neutral", "2026-01..2026-06", "S1", 0),
         ("f3", "obligation", "自 2026-07 起任务包最多 500 MB", "500 MB", "neutral", "2026-07..", "S2", 1),
         ("f4", "time", "自 2026-07 起", "2026-07..", "neutral", "2026-07..", "S2", 1)],
        [("S1", "旧任务包上限", "2026-01 至 2026-06 期间任务包最多 1 GB"),
         ("S2", "新任务包上限", "自 2026-07 起任务包最多 500 MB")], mode="supersede"),
    grp("F09-G5", "negation", "反馈包原始数据", "打包反馈任务包时",
        ["禁止在反馈包中包含原始数据文件"],
        [("f1", "polarity", "禁止在反馈包中包含原始数据文件", "forbid", "forbid", None, "S1", 0)],
        [("S1", "原始数据禁令", "禁止在反馈包中包含原始数据文件")]),
    grp("F09-G6", "distractor", "检索缓存复用", "命中检索缓存时",
        ["复用检索缓存时必须跳过超过 7 天的条目"],
        [("f1", "obligation", "复用检索缓存时必须跳过超过 7 天的条目", "跳过超过 7 天的条目", "require", None, "S1", 0),
         ("f2", "time", "超过 7 天", "7 天", "neutral", "7 天", "S1", 0)],
        [("S1", "缓存跳过期要求", "复用检索缓存时必须跳过超过 7 天的条目")]),
], [
    task("F09", "single_fact", "F09-G1", "单个任务包最多多大？",
         options=["200 MB", "500 MB", "1 GB"], expected_value="500 MB"),
    task("F09", "multi_obligation", "F09-G2", "打包反馈任务包时必须同时包含哪些内容？",
         options=["清单文件", "校验和", "来源说明", "运行日志"],
         expected_value=["清单文件", "校验和", "来源说明"], value_mode="set"),
    task("F09", "conditional_exception", "F09-G3",
         "本次运行失败，用户没有要求全部运行。反馈包应打包哪些运行？",
         options=["最近 3 次运行", "全部运行", "不适用默认打包范围"],
         expected_value="不适用默认打包范围",
         exception="对失败运行不适用默认打包范围",
         context="本次运行失败，用户没有要求全部运行。",
         necessary_context=("F09-G3-f3",)),
    task("F09", "conflict_update", "F09-G4",
         "当前是 2026-10，单个任务包最多多大？",
         options=["1 GB", "500 MB"], expected_value="500 MB",
         context="当前是 2026-10。", stale_value="1 GB",
         necessary_context=("F09-G4-f2", "F09-G4-f4")),
    task("F09", "negation", "F09-G5", "反馈包中是否可以包含原始数据文件？",
         options=["可以包含", "不得包含，只放清单与校验和", "询问用户"],
         expected_value="不得包含，只放清单与校验和", decision="deny"),
    task("F09", "no_relevant", None, "用户要求把任务包上传到公共镜像站，记忆里有适用于本次请求的规则吗？",
         decision="no_memory"),
])

F10 = fam("F10", "并行比较", "final", [
    grp("F10-G1", "single_fact", "比较条数上限", "并行比较数据集时",
        ["单次最多比较 4 个数据集"],
        [("f1", "obligation", "单次最多比较 4 个数据集", "4 个", "require", None, "S1", 0)],
        [("S1", "比较数量限制", "单次最多比较 4 个数据集")]),
    grp("F10-G2", "multi_obligation", "比较维度", "生成数据集比较时",
        ["比较必须包含样本数", "必须包含物种", "必须包含文件大小"],
        [("f1", "obligation", "比较必须包含样本数", "样本数", "require", None, "S1", 0),
         ("f2", "obligation", "必须包含物种", "物种", "require", None, "S2", 1),
         ("f3", "obligation", "必须包含文件大小", "文件大小", "require", None, "S2", 2)],
        [("S1", "比较维度约定", "比较必须包含样本数"),
         ("S2", "比较元数据要求", "必须包含物种")]),
    grp("F10-G3", "conditional_exception", "比较缺失值处理", "生成数据集比较时",
        ["默认把缺失值标记为未知", "若两侧缺失同一维度，则省略该维度", "对仅有一个数据集的情况不适用比较"],
        [("f1", "obligation", "默认把缺失值标记为未知", "标记为未知", "require", None, "S1", 0),
         ("f2", "condition", "若两侧缺失同一维度", "两侧缺失同一维度", "neutral", None, "S2", 1),
         ("f3", "exception", "对仅有一个数据集的情况不适用比较", "对仅有一个数据集的情况不适用", "neutral", None, "S2", 2)],
        [("S1", "缺失值默认", "默认把缺失值标记为未知"),
         ("S2", "缺失维度省略条件", "若两侧缺失同一维度，则省略该维度")]),
    grp("F10-G4", "conflict_update", "比较条数（历史）", "并行比较数据集时",
        ["2026-01 至 2026-06 期间最多比较 2 个数据集", "自 2026-07 起最多比较 4 个数据集"],
        [("f1", "obligation", "2026-01 至 2026-06 期间最多比较 2 个数据集", "2 个", "require", "2026-01..2026-06", "S1", 0),
         ("f2", "time", "2026-01 至 2026-06", "2026-01..2026-06", "neutral", "2026-01..2026-06", "S1", 0),
         ("f3", "obligation", "自 2026-07 起最多比较 4 个数据集", "4 个", "require", "2026-07..", "S2", 1),
         ("f4", "time", "自 2026-07 起", "2026-07..", "neutral", "2026-07..", "S2", 1)],
        [("S1", "旧比较上限", "2026-01 至 2026-06 期间最多比较 2 个数据集"),
         ("S2", "新比较上限", "自 2026-07 起最多比较 4 个数据集")], mode="supersede"),
    grp("F10-G5", "negation", "缺失值填充", "比较维度缺失时",
        ["禁止用默认值填充缺失维度"],
        [("f1", "polarity", "禁止用默认值填充缺失维度", "forbid", "forbid", None, "S1", 0)],
        [("S1", "缺失填充禁令", "禁止用默认值填充缺失维度")]),
    grp("F10-G6", "distractor", "引文版本字段", "导出引文时",
        ["导出引文时必须标注数据集版本号"],
        [("f1", "obligation", "导出引文时必须标注数据集版本号", "数据集版本号", "require", None, "S1", 0)],
        [("S1", "引文版本要求", "导出引文时必须标注数据集版本号")]),
], [
    task("F10", "single_fact", "F10-G1", "单次比较的数据集数量上限是多少？",
         options=["2 个", "4 个", "6 个"], expected_value="4 个"),
    task("F10", "multi_obligation", "F10-G2", "生成数据集比较时必须同时包含哪些维度？",
         options=["样本数", "物种", "文件大小", "测序深度"],
         expected_value=["样本数", "物种", "文件大小"], value_mode="set"),
    task("F10", "conditional_exception", "F10-G3",
         "一侧缺少物种维度，另一侧有该维度。缺失值应如何处理？",
         options=["标记为未知", "省略该维度", "不适用比较"], expected_value="标记为未知",
         context="一侧缺少物种维度，另一侧有该维度。",
         necessary_context=("F10-G3-f2",)),
    task("F10", "conflict_update", "F10-G4",
         "当前是 2026-10，单次最多比较几个数据集？",
         options=["2 个", "4 个"], expected_value="4 个",
         context="当前是 2026-10。", stale_value="2 个",
         necessary_context=("F10-G4-f2", "F10-G4-f4")),
    task("F10", "negation", "F10-G5", "比较维度缺失时，是否用默认值填充？",
         options=["用默认值填充", "不得填充，标记为未知", "询问用户"],
         expected_value="不得填充，标记为未知", decision="deny"),
    task("F10", "no_relevant", None, "用户要求把比较结果做成雷达图，记忆里有适用于本次请求的规则吗？",
         decision="no_memory"),
])

F11 = fam("F11", "历史与版本", "final", [
    grp("F11-G1", "single_fact", "历史版本保留数", "保存数据集历史版本时",
        ["每个数据集最多保留 5 个历史版本", "版本登记表重复确认：历史版本上限仍为 5 个"],
        [("f1", "obligation", "每个数据集最多保留 5 个历史版本", "5 个", "require", None, "S1", 0)],
        [("S1", "历史版本上限", "每个数据集最多保留 5 个历史版本"),
         ("S2", "版本登记表", "版本登记表重复确认：历史版本上限仍为 5 个")]),
    grp("F11-G2", "multi_obligation", "版本记录字段", "写入版本记录时",
        ["版本记录必须包含修订时间", "必须包含修订人", "必须包含变更摘要"],
        [("f1", "obligation", "版本记录必须包含修订时间", "修订时间", "require", None, "S1", 0),
         ("f2", "obligation", "必须包含修订人", "修订人", "require", None, "S2", 1),
         ("f3", "obligation", "必须包含变更摘要", "变更摘要", "require", None, "S2", 2)],
        [("S1", "版本记录约定", "版本记录必须包含修订时间"),
         ("S2", "版本审计要求", "必须包含修订人")]),
    grp("F11-G3", "conditional_exception", "版本回退权限", "执行版本回退时",
        ["默认只允许回退一个版本", "若用户是项目所有者，则允许回退任意版本", "对已归档数据集不适用版本回退"],
        [("f1", "obligation", "默认只允许回退一个版本", "只允许上一个版本", "require", None, "S1", 0),
         ("f2", "condition", "若用户是项目所有者", "用户是项目所有者", "neutral", None, "S2", 1),
         ("f3", "exception", "对已归档数据集不适用版本回退", "对已归档数据集不适用版本回退", "neutral", None, "S2", 2)],
        [("S1", "回退默认权限", "默认只允许回退一个版本"),
         ("S2", "回退所有者例外", "若用户是项目所有者，则允许回退任意版本")]),
    grp("F11-G4", "conflict_update", "历史版本保留数（历史）", "保存数据集历史版本时",
        ["2026-01 至 2026-06 期间最多保留 10 个历史版本", "自 2026-07 起最多保留 5 个历史版本"],
        [("f1", "obligation", "2026-01 至 2026-06 期间最多保留 10 个历史版本", "10 个", "neutral", "2026-01..2026-06", "S1", 0),
         ("f2", "time", "2026-01 至 2026-06", "2026-01..2026-06", "neutral", "2026-01..2026-06", "S1", 0),
         ("f3", "obligation", "自 2026-07 起最多保留 5 个历史版本", "5 个", "neutral", "2026-07..", "S2", 1),
         ("f4", "time", "自 2026-07 起", "2026-07..", "neutral", "2026-07..", "S2", 1)],
        [("S1", "旧版本保留数", "2026-01 至 2026-06 期间最多保留 10 个历史版本"),
         ("S2", "新版本保留数", "自 2026-07 起最多保留 5 个历史版本")], mode="supersede"),
    grp("F11-G5", "negation", "历史物理删除", "清理历史版本时",
        ["禁止物理删除历史版本记录"],
        [("f1", "polarity", "禁止物理删除历史版本记录", "forbid", "forbid", None, "S1", 0)],
        [("S1", "历史删除禁令", "禁止物理删除历史版本记录")]),
    grp("F11-G6", "distractor", "下载文件命名", "下载数据集文件时",
        ["下载文件时必须保留原始扩展名"],
        [("f1", "obligation", "下载文件时必须保留原始扩展名", "保留原始扩展名", "require", None, "S1", 0)],
        [("S1", "下载命名要求", "下载文件时必须保留原始扩展名")]),
], [
    task("F11", "single_fact", "F11-G1", "每个数据集最多保留几个历史版本？",
         options=["3 个", "5 个", "10 个"], expected_value="5 个"),
    task("F11", "multi_obligation", "F11-G2", "写入版本记录时必须同时包含哪些信息？",
         options=["修订时间", "修订人", "变更摘要", "文件大小"],
         expected_value=["修订时间", "修订人", "变更摘要"], value_mode="set"),
    task("F11", "conditional_exception", "F11-G3",
         "用户是项目所有者，目标数据集未归档。允许回退到哪个版本？",
         options=["只允许上一个版本", "允许任意版本", "不适用版本回退"],
         expected_value="允许任意版本", condition="用户是项目所有者",
         context="用户是项目所有者，目标数据集未归档。",
         necessary_context=("F11-G3-f2",)),
    task("F11", "conflict_update", "F11-G4",
         "当前是 2026-10，每个数据集最多保留几个历史版本？",
         options=["10 个", "5 个"], expected_value="5 个",
         context="当前是 2026-10。", stale_value="10 个",
         necessary_context=("F11-G4-f2", "F11-G4-f4")),
    task("F11", "negation", "F11-G5", "是否可以物理删除历史版本记录？",
         options=["可以删除", "禁止物理删除，只做归档", "询问用户"],
         expected_value="禁止物理删除，只做归档", decision="deny"),
    task("F11", "no_relevant", None, "用户要求把版本历史导出为 PDF，记忆里有适用于本次请求的规则吗？",
         decision="no_memory"),
])

F12 = fam("F12", "通知与提醒", "final", [
    grp("F12-G1", "single_fact", "单轮提醒条数", "生成用户提醒时",
        ["单轮最多给出 3 条提醒"],
        [("f1", "obligation", "单轮最多给出 3 条提醒", "3 条", "require", None, "S1", 0)],
        [("S1", "提醒条数限制", "单轮最多给出 3 条提醒")]),
    grp("F12-G2", "multi_obligation", "提醒内容要求", "生成用户提醒时",
        ["提醒必须说明触发原因", "必须给出下一步动作", "必须标注是否可关闭"],
        [("f1", "obligation", "提醒必须说明触发原因", "触发原因", "require", None, "S1", 0),
         ("f2", "obligation", "必须给出下一步动作", "下一步动作", "require", None, "S2", 1),
         ("f3", "obligation", "必须标注是否可关闭", "是否可关闭", "require", None, "S2", 2)],
        [("S1", "提醒内容约定", "提醒必须说明触发原因"),
         ("S2", "提醒可用性要求", "必须给出下一步动作")]),
    grp("F12-G3", "conditional_exception", "提醒渠道", "发送用户提醒时",
        ["默认在对话内提醒", "若用户开启邮件通知，则同时发送邮件", "对匿名会话不适用邮件通知"],
        [("f1", "obligation", "默认在对话内提醒", "对话内提醒", "require", None, "S1", 0),
         ("f2", "condition", "若用户开启邮件通知", "用户开启邮件通知", "neutral", None, "S2", 1),
         ("f3", "exception", "对匿名会话不适用邮件通知", "对匿名会话不适用邮件通知", "neutral", None, "S2", 2)],
        [("S1", "提醒渠道默认", "默认在对话内提醒"),
         ("S2", "提醒渠道条件", "若用户开启邮件通知，则同时发送邮件")]),
    grp("F12-G4", "conflict_update", "提醒条数（双来源）", "生成用户提醒时",
        ["自 2026-07 起单轮最多 3 条提醒", "自 2026-07 起单轮最多 5 条提醒"],
        [("f1", "obligation", "自 2026-07 起单轮最多 3 条提醒", "3 条", "require", "2026-07..", "S1", 0),
         ("f2", "time", "自 2026-07 起", "2026-07..", "neutral", "2026-07..", "S1", 0),
         ("f3", "obligation", "自 2026-07 起单轮最多 5 条提醒", "5 条", "require", "2026-07..", "S2", 1),
         ("f4", "time", "自 2026-07 起", "2026-07..", "neutral", "2026-07..", "S2", 1)],
        [("S1", "提醒上限（产品）", "自 2026-07 起单轮最多 3 条提醒"),
         ("S2", "提醒上限（项目）", "自 2026-07 起单轮最多 5 条提醒")], mode="simultaneous"),
    grp("F12-G5", "negation", "重复提醒", "同一事件再次出现时",
        ["禁止对同一事件重复提醒"],
        [("f1", "polarity", "禁止对同一事件重复提醒", "forbid", "forbid", None, "S1", 0)],
        [("S1", "重复提醒禁令", "禁止对同一事件重复提醒")]),
    grp("F12-G6", "distractor", "比较维度", "生成数据集比较时",
        ["比较必须包含样本数"],
        [("f1", "obligation", "比较必须包含样本数", "样本数", "require", None, "S1", 0)],
        [("S1", "比较维度约定", "比较必须包含样本数")]),
], [
    task("F12", "single_fact", "F12-G1", "单轮最多给出几条提醒？",
         options=["2 条", "3 条", "5 条"], expected_value="3 条"),
    task("F12", "multi_obligation", "F12-G2", "生成用户提醒时，每条提醒必须同时包含哪些信息？",
         options=["触发原因", "下一步动作", "是否可关闭", "发送渠道"],
         expected_value=["触发原因", "下一步动作", "是否可关闭"], value_mode="set"),
    task("F12", "conditional_exception", "F12-G3",
         "用户已开启邮件通知，会话是登录用户。提醒应通过哪些渠道发送？",
         options=["只在对话内提醒", "同时发送邮件", "不适用邮件通知"],
         expected_value="同时发送邮件", condition="用户开启邮件通知",
         context="用户已开启邮件通知，会话是登录用户。",
         necessary_context=("F12-G3-f2",)),
    task("F12", "conflict_update", "F12-G4",
         "当前是 2026-10，单轮最多给出几条提醒？",
         options=["3 条", "5 条", "需要询问用户"], expected_value=None, decision="ask_user",
         context="当前是 2026-10。",
         necessary_context=("F12-G4-f2", "F12-G4-f4")),
    task("F12", "negation", "F12-G5", "同一事件在同一轮再次出现，是否重复提醒？",
         options=["重复提醒", "只提醒一次", "询问用户"],
         expected_value="只提醒一次", decision="deny"),
    task("F12", "no_relevant", None, "用户要求给数据集设置访问密码，记忆里有适用于本次请求的规则吗？",
         decision="no_memory"),
])

SPEC = [F01, F02, F03, F04, F05, F06, F07, F08, F09, F10, F11, F12]
POSITIONS = ("start", "middle", "end")

                                                              

def _bigrams(text: str) -> set[str]:
    return bigrams(text)


def extract_time(probe: str) -> str | None:
    dates = DATE_RE.findall(probe)
    if dates:
        return "..".join(dates) if len(dates) > 1 else dates[0] + ".."
    m = TIME_RE.search(probe)
    return f"{m.group(1)} {m.group(2)}" if m else None


def _residual(compound: str, atoms: list[str]) -> str:
    """按序剥离原子跨度后剩下的字符；只允许连接词与标点。"""
    rest, pos = [], 0
    for atom in atoms:
        idx = compound.find(atom, pos)
        if idx < 0:
            raise AssertionError("原子文本在复合文本中不存在或顺序错乱")
        rest.append(compound[pos:idx])
        pos = idx + len(atom)
    rest.append(compound[pos:])
    return "".join(rest)


def _connector_ok(residual: str) -> bool:
    text = residual
    for word in CONNECTOR_WORDS:
        text = text.replace(word, "")
    return all(ch in CONNECTOR_CHARS for ch in text)


def check_group(group: dict) -> list[str]:
    errs: list[str] = []
    gid = group["group_id"]
    clauses = [a["text"] for a in group["atoms"]]
    compound = group["compound"]["text"]
    if compound != "；".join(clauses) + "。":
        errs.append(f"{gid}: 复合文本不是子句按序连接")
    if len({a["id"] for a in group["atoms"]}) != len(clauses):
        errs.append(f"{gid}: 原子 id 重复")
    for i, atom in enumerate(group["atoms"]):
        if atom["text"] not in compound:
            errs.append(f"{gid}: 原子 {atom['id']} 非逐字子串")
        if any(atom["text"] in other["text"] for j, other in enumerate(group["atoms"]) if j != i):
            errs.append(f"{gid}: 原子 {atom['id']} 是另一原子的子串，不是独立义务")
        if atom["kind"] != "memory" or atom["payload"] != {}:
            errs.append(f"{gid}: 原子 {atom['id']} 不是生产可接受的内存条目")
        for key in ("tier", "permanence"):
            allowed = {"explicit", "implicit"} if key == "tier" else {"persistent", "temporary", "unknown"}
            if atom[key] not in allowed:
                errs.append(f"{gid}: 原子 {atom['id']} 的 {key} 非法")
    try:
        residual = _residual(compound, clauses)
        if not _connector_ok(residual):
            errs.append(f"{gid}: 复合文本存在未被原子覆盖的内容：{residual!r}")
    except AssertionError as exc:
        errs.append(f"{gid}: {exc}")

    facts = group["facts"]
    ids = [f["fact_id"] for f in facts]
    if len(ids) != len(set(ids)):
        errs.append(f"{gid}: 事实 id 重复")
    by_id = {f["fact_id"]: f for f in facts}
    carried: dict[str, list[str]] = {}
    for atom in group["atoms"]:
        for fid in atom["payload_fact_ids"]:
            carried.setdefault(fid, []).append(atom["id"])
    if set(carried) != set(ids):
        errs.append(f"{gid}: 事实分配与事实表不一致 {sorted(set(ids) ^ set(carried))}")
    for fid, atoms in carried.items():
        if len(atoms) != 1:
            errs.append(f"{gid}: 事实 {fid} 被 {len(atoms)} 个原子重复承载")
        fact = by_id.get(fid)
        if fact is None:
            continue
        atom = next(a for a in group["atoms"] if a["id"] == atoms[0])
        if fact["probe"] not in atom["text"]:
            errs.append(f"{gid}: 事实 {fid} 的探针不在其原子内")
    for fid in ids:
        if not any(fid in a["payload_fact_ids"] for a in group["atoms"]):
            errs.append(f"{gid}: 事实 {fid} 未被任何原子承载")

    types = {f["type"] for f in facts}
    missing = DIMENSION_BY_PROFILE[group["profile"]] - types
    if missing:
        errs.append(f"{gid}: profile {group['profile']} 缺少维度 {sorted(missing)}")

    for fact in facts:
        if fact["type"] in ("obligation", "polarity"):
            markers = POLARITY_MARKERS[fact["polarity"]]
            if fact["polarity"] != "neutral" and not any(m in fact["probe"] for m in markers):
                errs.append(f"{gid}: 事实 {fact['fact_id']} 声明极性 {fact['polarity']} 但文本无对应标记")
        if fact["type"] == "time":
            got = extract_time(fact["probe"])
            if not got or got != fact["time"] or fact["value"] != fact["time"]:
                errs.append(f"{gid}: 时间事实 {fact['fact_id']} 表达不规范 ({got!r})")
        if fact["type"] in ("condition", "exception"):
            markers = CONDITION_MARKERS if fact["type"] == "condition" else EXCEPTION_MARKERS
            if not any(m in fact["probe"] for m in markers):
                errs.append(f"{gid}: {fact['type']} 事实 {fact['fact_id']} 缺少标记词")
        if fact["type"] == "source":
            src = next((s for s in group["sources"] if s["source_id"] == fact["value"]), None)
            if src is None:
                errs.append(f"{gid}: 来源事实 {fact['fact_id']} 指向未声明来源")
            elif src["quote"] not in compound:
                errs.append(f"{gid}: 来源引文未出现在复合文本")

    src_ids = {s["source_id"] for s in group["sources"]}
    if not src_ids <= set(group["compound"]["links"]):
        errs.append(f"{gid}: 复合条目未保留全部来源 id")
    atom_links: dict[str, set] = {}
    for atom in group["atoms"]:
        atom_links[atom["id"]] = set(atom["links"])
    for fact in facts:
        if fact["type"] != "source":
            continue
        holders = carried.get(fact["fact_id"]) or []
        if not holders:
            continue                     
        holder = holders[0]
        if fact["value"] not in atom_links[holder]:
            errs.append(f"{gid}: 来源事实 {fact['fact_id']} 的链接未进入承载原子")
    if group["profile"] == "conflict_update":
        if group["mode"] not in ("supersede", "simultaneous"):
            errs.append(f"{gid}: 冲突组缺少合法 mode")
    elif group["mode"] is not None:
        errs.append(f"{gid}: 非冲突组不应带 mode")
    return errs


def _fact_ids_of_type(group: dict, ftype: str) -> list[str]:
    return [f["fact_id"] for f in group["facts"] if f["type"] == ftype]


def derive_gold(task: dict, group: dict) -> dict:
    """从组规格推出 gold：必需事实、必要语境、禁答事实、事实级命题检查。"""
    ph = task["phenomenon"]
    if ph == "no_relevant":
        return {"required_facts": [], "necessary_context": [], "forbidden_facts": [],
                "fact_checks": {}, "required_groups": [], "applicable_slots": []}
    facts = group["facts"]
    by_id = {f["fact_id"]: f for f in facts}
    obs = [f for f in facts if f["type"] == "obligation"]
    sources = [f for f in facts if f["type"] == "source"]
    times = [f for f in facts if f["type"] == "time"]
    if ph == "conflict_update" and group["mode"] == "supersede":
        current = max(obs, key=lambda f: (f["time"] or "").split("..")[0])
        required = [current["fact_id"]] + [f["fact_id"] for f in sources]
        forbidden = [f["fact_id"] for f in obs if f is not current]
        necessary = list(task["necessary_context"])
    elif ph == "conflict_update":
        required = [f["fact_id"] for f in obs] + [f["fact_id"] for f in sources]
        forbidden = []
        necessary = list(task["necessary_context"])
    elif ph == "negation":
        required = [f["fact_id"] for f in facts if f["type"] == "polarity"] \
                   + [f["fact_id"] for f in sources]
        forbidden = []
        necessary = list(task["necessary_context"]) or [f["fact_id"] for f in times]
    else:
        required = [f["fact_id"] for f in obs] + [f["fact_id"] for f in sources]
        forbidden = []
        necessary = list(task["necessary_context"])

    expected = task["expected"]
    checks: dict[str, dict] = {}
    for fid in required + necessary:
        fact = by_id[fid]
        ftype = fact["type"]
        if ftype == "source":
            check = {"field": "cited_group", "contains": task["target_group"]}
        elif ftype == "condition":
            check = {"field": "applied_condition", "equals": expected["applied_condition"]} \
                if expected["applied_condition"] is not None \
                else {"field": "applied_condition", "isnull": True}
        elif ftype == "exception":
            check = {"field": "applied_exception", "equals": expected["applied_exception"]} \
                if expected["applied_exception"] is not None \
                else {"field": "applied_exception", "isnull": True}
        elif ftype == "polarity":
            check = {"field": "decision", "equals": expected["decision"]}
        elif ftype == "time":
            if ph == "conflict_update" and group["mode"] == "supersede":
                check = {"field": "value", "not_equals": task["stale_value"]}
            elif ph == "conflict_update":
                check = {"field": "decision", "equals": "ask_user"}
            else:
                check = {"field": "value", "equals": expected["value"]}
        else:
            if ph == "multi_obligation":
                check = {"field": "value", "contains": fact["value"]}
            elif expected["value"] is None:
                check = {"field": "decision", "equals": expected["decision"]}
            else:
                check = {"field": "value", "equals": expected["value"]}
        checks[fid] = check
    return {"required_facts": required, "necessary_context": necessary,
            "forbidden_facts": forbidden, "fact_checks": checks,
            "required_groups": [task["target_group"]], "applicable_slots": [group["slot"]]}


def check_task(task: dict, group: dict | None) -> list[str]:
    errs: list[str] = []
    tid = task["task_id"]
    ph = task["phenomenon"]
    if ph not in PHENOMENA:
        errs.append(f"{tid}: 未知现象 {ph}")
        return errs
    if ph == "no_relevant":
        if task["target_group"] is not None:
            errs.append(f"{tid}: 负控制不应有目标组")
        if task["expected"]["decision"] != "no_memory" or task["expected"]["value"] is not None:
            errs.append(f"{tid}: 负控制 gold 形状错误")
        return errs
    if group is None:
        return [f"{tid}: 目标组不存在"]
    if group["profile"] != PROFILE_BY_PHENOMENON[ph]:
        errs.append(f"{tid}: 现象 {ph} 与组 profile {group['profile']} 不匹配")
    if group["family_id"] != task["family_id"]:
        errs.append(f"{tid}: 目标组跨 family")
    options = task["options"] or []
    if len(options) != len(set(options)):
        errs.append(f"{tid}: 选项重复")
    expected = task["expected"]
    if expected["decision"] not in DECISIONS:
        errs.append(f"{tid}: 非法 decision")
    if expected["value"] is not None:
        if not options:
            errs.append(f"{tid}: 有期望值但没有选项")
        elif expected["value_mode"] == "set":
            if not set(expected["value"]) <= set(options):
                errs.append(f"{tid}: 期望集合不在选项内")
        elif expected["value"] not in options:
            errs.append(f"{tid}: 期望值不在选项内")
    if task["stale_value"] is not None:
        if task["stale_value"] not in options:
            errs.append(f"{tid}: 陈旧值不在选项内")
        if task["stale_value"] == expected["value"]:
            errs.append(f"{tid}: 陈旧值与正确值相同")
    if task["stale_value"] is not None and ph != "conflict_update":
        errs.append(f"{tid}: 只有冲突/更新任务可带陈旧值")
    gold = derive_gold(task, group)
    fact_ids = {f["fact_id"] for f in group["facts"]}
    for fid in gold["required_facts"] + gold["necessary_context"] + gold["forbidden_facts"]:
        if fid not in fact_ids:
            errs.append(f"{tid}: gold 事实 {fid} 不存在")
    for fid in gold["required_facts"] + gold["necessary_context"]:
        if fid not in gold["fact_checks"]:
            errs.append(f"{tid}: 事实 {fid} 缺少命题检查")
    if not gold["required_facts"]:
        errs.append(f"{tid}: 相关任务必须有必需事实")
    if ph == "conditional_exception":
        if not gold["necessary_context"]:
            errs.append(f"{tid}: 条件/例外任务必须声明必要语境")
        if not any(f["fact_id"] in gold["necessary_context"]
                   and f["type"] in ("condition", "exception") for f in group["facts"]):
            errs.append(f"{tid}: 条件/例外任务的必要语境必须是该组的条件或例外事实")
    if ph == "conflict_update" and not gold["necessary_context"]:
        errs.append(f"{tid}: 冲突/更新任务必须声明必要语境（时间）")
    overlap = _bigrams(task["question"] + (task["context"] or "")) & \
        _bigrams(group["slot"] + group["compound"]["text"])
    if len(overlap) < 3:
        errs.append(f"{tid}: 检索词面重叠不足（{len(overlap)}），检索臂对目标组不公平")
    if any(token in task["question"] for token in ARM_TOKENS):
        errs.append(f"{tid}: 任务文本泄漏臂名")
    return errs


def _build_banks(families: list[dict], tasks: list[dict]) -> list[dict]:
    groups = [g for f in families for g in f["groups"]]
    split_of = {g["group_id"]: f["split"] for f in families for g in f["groups"]}
    banks = []
    for task in tasks:
        split = split_of[task.get("target_group") or task["family_id"] + "-G1"]
        scales = SPLIT_SCALES[split]
        pool = [g["group_id"] for g in groups if split_of[g["group_id"]] == split]
        idx = [t["task_id"] for t in tasks if split_of[(t.get("target_group") or t["family_id"] + "-G1")] == split].index(task["task_id"])
        position = POSITIONS[idx % len(POSITIONS)]
        for scale in scales:
            relevant = [task["target_group"]] if task["target_group"] else []
            need = scale - len(relevant)
            if need < 0 or need > len(pool) - len(relevant):
                raise AssertionError(f"{task['task_id']}: split 组数不足以构造 scale={scale}")
            distractors = sorted([g for g in pool if g not in relevant],
                                 key=lambda g: stable_hash(task["task_id"], scale, g))[:need]
            ordered = list(distractors)
            if relevant:
                pos = {"start": 0, "middle": len(ordered) // 2, "end": len(ordered)}[position]
                ordered.insert(pos, relevant[0])
                pos_index = pos
            else:
                pos_index = None
            banks.append({
                "bank_id": f"{task['task_id']}-S{scale}", "task_id": task["task_id"],
                "split": split, "scale": scale, "position": position if relevant else None,
                "relevant_index": pos_index, "relevant_groups": relevant,
                "distractor_groups": distractors, "group_ids": ordered,
            })
    return banks


def active_phenomena(family_index: int) -> list[str]:
    """预注册轮转：第 i 族取 PHENOMENA 中从 i%6 起的连续 4 个现象。"""
    start = family_index % len(PHENOMENA)
    return [PHENOMENA[(start + k) % len(PHENOMENA)] for k in range(4)]


def active_family_tasks(family: dict, index: int) -> list[dict]:
    want = active_phenomena(index)
    chosen = {t["phenomenon"]: t for t in family["tasks"] if t["phenomenon"] in want}
    return [chosen[ph] for ph in want if ph in chosen]


def active_tasks(families: list[dict]) -> list[dict]:
    """只发出预注册的 4 题/族；其余已写题作为储备池，不进入本轮分母。"""
    out = []
    for index, family in enumerate(families):
        want = active_phenomena(index)
        chosen = {t["phenomenon"]: t for t in family["tasks"] if t["phenomenon"] in want}
        missing = [ph for ph in want if ph not in chosen]
        if missing:
            raise AssertionError(f"{family['family_id']} 缺少预注册现象 {missing}")
        out.extend(chosen[ph] for ph in want)
    return out


def verify(families: list[dict] | None = None) -> dict:
    families = families if families is not None else SPEC
    errors: list[str] = []
    checks: list[str] = []
    groups = [g for f in families for g in f["groups"]]
    try:
        tasks = active_tasks(families)
    except AssertionError as exc:
        tasks = []
        errors.append(f"V14-rotation: {exc}")

    def add(code: str, errs: list[str]) -> None:
        if code not in checks:
            checks.append(code)
        errors.extend(errs)

    add("V1-ids", [e for g in groups if g["group_id"].count("-") != 1
                   for e in [f"{g['group_id']}: 组 id 形状非法"]] +
        ([f"family 重复：{len(families)}"] if len({f['family_id'] for f in families}) != len(families) else []))
    authored_ids = [t["task_id"] for f in families for t in f["tasks"]]
    if len(authored_ids) != len(set(authored_ids)):
        errors.append(f"V1: 题池 task_id 重复（{len(authored_ids)} 条 / {len(set(authored_ids))} 唯一）")
    if len(families) != 12:
        errors.append(f"V1: family 数应为 12，实际 {len(families)}")
    if len(groups) != 72:
        errors.append(f"V1: 组数应为 72，实际 {len(groups)}")
    for index, f in enumerate(families):
        if len(f["groups"]) != 6:
            errors.append(f"V1: {f['family_id']} 组数不是 6")
        if len(f["tasks"]) < 4:
            errors.append(f"V1: {f['family_id']} 题池不足 4")
        for ph in active_phenomena(index):
            if not any(t["phenomenon"] == ph for t in f["tasks"]):
                errors.append(f"V1: {f['family_id']} 预注册现象 {ph} 无题")

    split_counts = {"dev": (0, 0), "final": (0, 0)}
    for f in families:
        if f["split"] not in split_counts:
            errors.append(f"V2: 未知 split {f['split']}")
        fam_c, grp_c = split_counts[f["split"]]
        split_counts[f["split"]] = (fam_c + 1, grp_c + len(f["groups"]))
    if split_counts["dev"] != (4, 24):
        errors.append(f"V2: dev 应为 4 族/24 组，实际 {split_counts['dev']}")
    if split_counts["final"] != (8, 48):
        errors.append(f"V2: final 应为 8 族/48 组，实际 {split_counts['final']}")

    add("V3-compound", [e for g in groups for e in check_group(g)])
    seen_compound, seen_question = {}, {}
    for g in groups:
        digest = sha256(g["compound"]["text"])
        if digest in seen_compound:
            errors.append(f"V16: 复合文本重复 {g['group_id']} / {seen_compound[digest]}")
        seen_compound[digest] = g["group_id"]
    for t in tasks:
        digest = sha256(t["question"])
        if digest in seen_question:
            errors.append(f"V16: 任务问句重复 {t['task_id']} / {seen_question[digest]}")
        seen_question[digest] = t["task_id"]

    by_id = {g["group_id"]: g for g in groups}
    for t in tasks:
        target = by_id.get(t["target_group"]) if t["target_group"] else None
        add("V13-task", check_task(t, target))
        if t["target_group"] and not t["question"].strip():
            errors.append(f"V13: {t['task_id']} 问句为空")

    for f in families:
        for t in f["tasks"]:
            if t["family_id"] != f["family_id"]:
                errors.append(f"V13: {t['task_id']} family 不一致")
            if t.get("target_group") and not any(g["group_id"] == t["target_group"]
                                                 for g in f["groups"]):
                errors.append(f"V13: {t['task_id']} 目标组不在本族")

    counts = {ph: sum(1 for t in tasks if t["phenomenon"] == ph) for ph in PHENOMENA}
    if any(counts[ph] != 8 for ph in PHENOMENA):
        errors.append(f"V14: 现象分布应为各 8，实际 {counts}")
    split_of_family = {f["family_id"]: f["split"] for f in families}
    split_of_group = {g["group_id"]: f["split"] for f in families for g in f["groups"]}
    dev_tasks = [t for t in tasks if split_of_family[t["family_id"]] == "dev"]
    final_tasks = [t for t in tasks if split_of_family[t["family_id"]] == "final"]
    if len(dev_tasks) != 16 or len(final_tasks) != 32:
        errors.append(f"V14: 任务数应为 dev16/final32，实际 {len(dev_tasks)}/{len(final_tasks)}")
    for split, rows in (("dev", dev_tasks), ("final", final_tasks)):
        present = {t["phenomenon"] for t in rows}
        if present != set(PHENOMENA):
            errors.append(f"V14: {split} 未覆盖全部现象，缺 {sorted(set(PHENOMENA) - present)}")
    for t in final_tasks:
        if t["target_group"] and split_of_group[t["target_group"]] != "final":
            errors.append(f"V14: final 任务引用了非 final 组 {t['task_id']}")
    for t in dev_tasks:
        if t["target_group"] and split_of_group[t["target_group"]] != "dev":
            errors.append(f"V14: dev 任务引用了非 dev 组 {t['task_id']}")

    for t in tasks:
        if t["phenomenon"] != "no_relevant":
            continue
        if t["target_group"] is not None:
            errors.append(f"V15: {t['task_id']} 负控制不应有目标组")
        if t["expected"]["decision"] != "no_memory" or t["expected"]["value"] is not None:
            errors.append(f"V15: {t['task_id']} 负控制 gold 形状错误")
        for g in groups:
            if split_of_group[g["group_id"]] != split_of_family[t["family_id"]]:
                continue
            if g["slot"] in t["question"]:
                errors.append(f"V15: 负控制问句命中组标题 {g['slot']}")

    try:
        banks = _build_banks(families, tasks)
    except AssertionError as exc:
        banks = []
        errors.append(f"V21-bank: {exc}")
    for bank in banks:
        if len(bank["group_ids"]) != bank["scale"]:
            errors.append(f"V21: {bank['bank_id']} 组数不等于 scale")
        if len(set(bank["group_ids"])) != len(bank["group_ids"]):
            errors.append(f"V21: {bank['bank_id']} 组重复")
        if bank["relevant_groups"]:
            got = bank["group_ids"][bank["relevant_index"]]
            if got != bank["relevant_groups"][0]:
                errors.append(f"V21: {bank['bank_id']} 相关组位置不符")
        if any(split_of_group[g] != bank["split"] for g in bank["group_ids"]):
            errors.append(f"V21: {bank['bank_id']} 跨 split")
    for split in ("dev", "final"):
        rows = [b for b in banks if b["split"] == split]
        for pos in POSITIONS:
            if not any(b["position"] == pos for b in rows):
                errors.append(f"V21: {split} 缺少位置 {pos}")

    for g in groups:
        for other in groups:
            if g is other or g["family_id"] == other["family_id"]:
                continue
            if g["compound"]["text"] == other["compound"]["text"]:
                errors.append(f"V16: 跨族文本重复 {g['group_id']} / {other['group_id']}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(g["compound"]["text"]):
                errors.append(f"V12: {g['group_id']} 复合文本含敏感模式")
    for t in tasks:
        for pattern in SECRET_PATTERNS:
            if pattern.search(t["question"]):
                errors.append(f"V12: {t['task_id']} 问句含敏感模式")
        if any(token in json.dumps(t, ensure_ascii=False) for token in ARM_TOKENS):
            errors.append(f"V17: {t['task_id']} 含臂名")

    dimension = {g["group_id"]: sorted({f["type"] for f in g["facts"]}) for g in groups}
    corpus_types = {t for types in dimension.values() for t in types}
    if corpus_types != set(FACT_TYPES):
        errors.append(f"V7: 语料未覆盖全部事实类型，缺 {sorted(set(FACT_TYPES) - corpus_types)}")

    return {"schema": "paired-library-verify-v1", "ok": not errors,
            "checks": sorted(set(checks) | set(VERIFY_CHECK_IDS)),
            "error_count": len(errors), "errors": errors,
            "counts": {"families": len(families), "groups": len(groups), "tasks": len(tasks),
                       "dev_tasks": len(dev_tasks), "final_tasks": len(final_tasks),
                       "banks": len(banks), "phenomena": counts},
            "split": {k: {"families": v[0], "groups": v[1]} for k, v in split_counts.items()},
            "dimension_by_profile": {k: sorted(v) for k, v in DIMENSION_BY_PROFILE.items()},
            "atomization_method": "spec_clause_quotes_v1（作者规格化子句，非实时 LLM 原子化器）"}


                                                              

def _materialize(families: list[dict]) -> tuple[list[dict], list[dict]]:
    groups_out, tasks_out = [], []
    active_ids = {t["task_id"] for t in active_tasks(families)}
    for f in families:
        for g in f["groups"]:
            atoms = []
            for atom in g["atoms"]:
                row = {k: v for k, v in atom.items() if k != "payload_fact_ids"}
                atoms.append(row)
            groups_out.append({
                "group_id": g["group_id"], "family_id": g["family_id"],
                "split": f["split"], "profile": g["profile"], "slot": g["slot"],
                "trigger": g["trigger"], "mode": g["mode"],
                "compound": g["compound"], "atoms": atoms,
                "facts": [{k: v for k, v in fact.items() if k != "atom_index"} for fact in g["facts"]],
                "atomic_fact_ids": {atom["id"]: atom["payload_fact_ids"] for atom in g["atoms"]},
                "sources": g["sources"],
            })
        for t in f["tasks"]:
            if t["task_id"] not in active_ids:
                continue
            group = next((g for g in f["groups"] if g["group_id"] == t["target_group"]), None)
            gold = derive_gold(t, group) if group else {
                "required_facts": [], "necessary_context": [], "forbidden_facts": [],
                "fact_checks": {}, "required_groups": [], "applicable_slots": []}
            row = dict(t)
            row["split"] = f["split"]
            row["gold"] = gold
            tasks_out.append(row)
    return groups_out, tasks_out


def build() -> dict:
    report = verify(SPEC)
    if not report["ok"]:
        raise SystemExit("校验失败，拒绝构建：\n" + "\n".join(report["errors"][:20]))
    groups, tasks = _materialize(SPEC)
    banks = _build_banks(SPEC, tasks)
    DATA.mkdir(parents=True, exist_ok=True)
    jsonl_write(DATA / "groups.jsonl", groups)
    jsonl_write(DATA / "tasks.jsonl", tasks)
    jsonl_write(DATA / "banks.jsonl", banks)
    families = [{"family_id": f["family_id"], "theme": f["theme"], "split": f["split"],
                 "group_ids": [g["group_id"] for g in f["groups"]],
                 "task_ids": [t["task_id"] for t in active_family_tasks(f, i)],
                 "task_pool": [t["task_id"] for t in f["tasks"]]}
                for i, f in enumerate(SPEC)]
    json_write(DATA / "families.json", {"schema": "disclosure-families-v1", "families": families})
    json_write(DATA / "splits.json", {
        "schema": "disclosure-splits-v1",
        "splits": {s: {"families": [f["family_id"] for f in SPEC if f["split"] == s],
                       "groups": [g["group_id"] for f in SPEC if f["split"] == s for g in f["groups"]],
                       "scales": list(SPLIT_SCALES[s])} for s in ("dev", "final")},
        "budget_candidates": list(BUDGET_CANDIDATES),
        "position_rule": "task_index_within_split % 3 → start/middle/end",
    })
    manifest = {
        "schema": "disclosure-manifest-v1",
        "counts": report["counts"],
        "split": report["split"],
        "dimension_by_profile": report["dimension_by_profile"],
        "atomization_method": report["atomization_method"],
        "task_pool_by_family": {f["family_id"]: [t["task_id"] for t in f["tasks"]] for f in SPEC},
        "active_tasks_by_family": {f["family_id"]: [t["task_id"] for t in active_family_tasks(f, i)]
                                   for i, f in enumerate(SPEC)},
        "active_phenomena_by_family": {f["family_id"]: active_phenomena(i)
                                       for i, f in enumerate(SPEC)},
        "sha256": {
            "groups": sha256((DATA / "groups.jsonl").read_text(encoding="utf-8")),
            "tasks": sha256((DATA / "tasks.jsonl").read_text(encoding="utf-8")),
            "banks": sha256((DATA / "banks.jsonl").read_text(encoding="utf-8")),
            "families": sha256((DATA / "families.json").read_text(encoding="utf-8")),
            "splits": sha256((DATA / "splits.json").read_text(encoding="utf-8")),
        },
        "authoring": {"author": "AI 子 Agent（代理证据，非 human gold）",
                      "review_status": "pending 主 Agent / 人工复核",
                      "live_atomizer_used": False},
        "notes": [
            "复合版与原子版来自同一「子句→事实」规格；原子版是逐字子句引用，不是实时抽取器输出。",
            "复合版文本 = 子句按序以「；」连接，因此两版内容差只来自条目边界。",
            "任务 gold 使用闭集选项与事实级命题检查，不使用关键词命中率。",
        ],
    }
    json_write(DATA / "manifest.json", manifest)
    json_write(DATA / "verify.json", report)
    return {"ok": True, "manifest": manifest, "verify": report}


                                                                    

def selftest() -> dict:
    import copy
    base = verify(SPEC)
    assert base["ok"], base["errors"][:10]
    assert base["counts"]["groups"] == 72 and base["counts"]["tasks"] == 48
    assert base["counts"]["banks"] == 16 * 2 + 32 * 3

    def mutated(fn) -> list[str]:
        spec = copy.deepcopy(SPEC)
        fn(spec)
        return verify(spec)["errors"]

    def drop_atom_fact(spec):
        spec[0]["groups"][0]["atoms"][1]["payload_fact_ids"] = []

    def leak_arm(spec):
        spec[0]["tasks"][0]["question"] += "（P-full）"

    def bad_polarity(spec):
        g = next(g for f in spec for g in f["groups"] if g["profile"] == "negation")
        fact = next(x for x in g["facts"] if x["type"] == "polarity")
        fact["polarity"] = "require"

    def dup_text(spec):
        spec[1]["groups"][0]["compound"]["text"] = spec[0]["groups"][0]["compound"]["text"]

    def bad_source(spec):
        g = spec[0]["groups"][0]
        fact = next(x for x in g["facts"] if x["type"] == "source")
        fact["value"] = "F99-S9"

    def no_target_negative(spec):
        t = next(t for f in spec for t in f["tasks"] if t["phenomenon"] == "no_relevant")
        t["target_group"] = "F01-G1"

    def weak_overlap(spec):
        spec[0]["tasks"][0]["question"] = "这件事该怎么处理？"

    def conflict_without_context(spec):
        t = next(t for f in spec for t in f["tasks"] if t["phenomenon"] == "conflict_update")
        t["necessary_context"] = []

    def empty_required_via_negative(spec):
                                                             
                                                       
        t = next(t for f in spec for t in f["tasks"] if t["phenomenon"] == "no_relevant")
        t["phenomenon"] = "single_fact"
        t["target_group"] = "F01-G5"
        t["necessary_context"] = []

    cases = {
        "drop_atom_fact": drop_atom_fact, "leak_arm": leak_arm, "bad_polarity": bad_polarity,
        "dup_text": dup_text, "bad_source": bad_source,
        "no_target_negative": no_target_negative, "weak_overlap": weak_overlap,
        "conflict_without_context": conflict_without_context,
        "empty_required_via_negative": empty_required_via_negative,
    }
    caught = {}
    for name, fn in cases.items():
        errs = mutated(fn)
        caught[name] = bool(errs)
        assert errs, f"selftest 未捕获故障：{name}"
    return {"ok": True, "base": {k: base[k] for k in ("ok", "counts", "error_count")},
            "negative_cases_caught": caught, "checks": base["checks"],
            "atomization_method": base["atomization_method"]}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        print(json.dumps(selftest(), ensure_ascii=False, indent=2))
    elif cmd == "verify":
        print(json.dumps(verify(SPEC), ensure_ascii=False, indent=2))
    elif cmd == "build":
        result = build()
        print(json.dumps({"ok": result["ok"], "counts": result["manifest"]["counts"],
                          "sha256": result["manifest"]["sha256"]}, ensure_ascii=False, indent=2))
    else:
        raise SystemExit(f"未知命令：{cmd}")
