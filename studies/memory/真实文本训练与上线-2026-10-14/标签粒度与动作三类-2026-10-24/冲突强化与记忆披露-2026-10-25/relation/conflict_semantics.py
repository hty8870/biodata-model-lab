"""冲突识别线的语义规格层：领域锚点、自然/反事实配方、确定性标签规则与文本门。

设计约束（预注册，不允许在配方里塞标签捷径）：

* 标签只由 ``classify(old, new)`` 从**已声明因素**推出，作者只负责把因素实现成中文句子；
* 自然 OOD 覆盖 7 个预注册机制；每族 6 shared 为 2E/2P/2C，加 control 与 dyad 后各类 6 行；
* 反事实 dyad 只做**一处**声明改动，改动在文本上必须是单点替换，且标签必须翻转；
* 所有坐标（作用域、事项、清单、年份、数量）都来自 ``SETTINGS`` 这一处锚点。
"""
from __future__ import annotations

import re
from typing import Sequence

from conflict_common import COARSE, DYAD_LABEL_PLAN, FINE_TO_COARSE, SHARED_LABEL_PLAN

                                                                                        
                           
                                                                                        

SETTINGS = [
    {"id": "literature-citation", "name": "文献助手引用规范", "subject": "同一用户",
     "verb": "引用", "unit": "篇", "scopes": ("综述引用", "方法学引用", "临床引用"),
     "items": ("章节提纲", "检索记录", "审阅批注"), "lists": ("待引清单", "补充清单"),
     "years": (2024, 2025), "lo": 5, "qmax": 12,
     "slot_main": "citation_year_window", "object_main": "近五年文献",
     "slot_secondary": "citation_numbering", "object_secondary": "文献编号"},
    {"id": "dataset-archive", "name": "数据归档规范", "subject": "同一用户",
     "verb": "归档", "unit": "份", "scopes": ("原始数据", "派生数据", "外部上传"),
     "items": ("汇总表", "附录", "校验记录"), "lists": ("归档清单", "交接清单"),
     "years": (2025, 2026), "lo": 6, "qmax": 12,
     "slot_main": "retention_days", "object_main": "归档副本",
     "slot_secondary": "checksum_policy", "object_secondary": "校验和"},
    {"id": "report-review", "name": "报告复核规范", "subject": "同一用户",
     "verb": "复核", "unit": "页", "scopes": ("周报", "季报", "年报"),
     "items": ("摘要", "方法", "结论"), "lists": ("主报告清单", "附件清单"),
     "years": (2024, 2026), "lo": 8, "qmax": 20,
     "slot_main": "review_length", "object_main": "正文",
     "slot_secondary": "signoff_policy", "object_secondary": "签发记录"},
    {"id": "meeting-handoff", "name": "交接记录规范", "subject": "同一用户",
     "verb": "交接", "unit": "项", "scopes": ("值班交接", "项目交接", "紧急交接"),
     "items": ("交接要点", "风险项", "附件包"), "lists": ("交接清单", "风险清单"),
     "years": (2025, 2027), "lo": 4, "qmax": 12,
     "slot_main": "handoff_completeness", "object_main": "待办事项",
     "slot_secondary": "attachment_policy", "object_secondary": "交接纪要"},
    {"id": "experiment-log", "name": "实验记录规范", "subject": "同一用户",
     "verb": "记录", "unit": "条", "scopes": ("预实验", "正式实验", "复现实验"),
     "items": ("超参数", "环境信息", "失败日志"), "lists": ("实验清单", "复现清单"),
     "years": (2024, 2025), "lo": 3, "qmax": 10,
     "slot_main": "log_completeness", "object_main": "实验过程",
     "slot_secondary": "command_policy", "object_secondary": "评测命令"},
    {"id": "access-grant", "name": "权限开通规范", "subject": "同一用户",
     "verb": "开通", "unit": "项", "scopes": ("只读账号", "编辑账号", "审计账号"),
     "items": ("数据目录", "模型目录", "日志目录"), "lists": ("授权清单", "审计清单"),
     "years": (2025, 2026), "lo": 2, "qmax": 9,
     "slot_main": "access_scope", "object_main": "访问权限",
     "slot_secondary": "expiry_policy", "object_secondary": "登录凭据"},
    {"id": "vendor-quote", "name": "采购比价规范", "subject": "同一用户",
     "verb": "询价", "unit": "家", "scopes": ("硬件采购", "软件采购", "服务采购"),
     "items": ("报价单", "资质文件", "交付计划"), "lists": ("比价清单", "候选清单"),
     "years": (2024, 2026), "lo": 3, "qmax": 10,
     "slot_main": "quote_count", "object_main": "供应商",
     "slot_secondary": "qualification_policy", "object_secondary": "付款条件"},
    {"id": "content-calendar", "name": "内容排期规范", "subject": "同一用户",
     "verb": "排期", "unit": "条", "scopes": ("公众号", "视频号", "邮件简报"),
     "items": ("标题", "封面图", "审阅记录"), "lists": ("排期清单", "素材清单"),
     "years": (2025, 2026), "lo": 4, "qmax": 12,
     "slot_main": "publish_lead_time", "object_main": "发布内容",
     "slot_secondary": "asset_policy", "object_secondary": "视频素材"},
    {"id": "annotate-quality", "name": "标注质检规范", "subject": "同一用户",
     "verb": "抽检", "unit": "条", "scopes": ("训练集标注", "验证集标注", "外部标注"),
     "items": ("抽检记录", "争议记录", "复核记录"), "lists": ("质检清单", "争议清单"),
     "years": (2024, 2025), "lo": 5, "qmax": 15,
     "slot_main": "sample_rate", "object_main": "标注结果",
     "slot_secondary": "dispute_policy", "object_secondary": "争议条目"},
    {"id": "model-release", "name": "模型发布规范", "subject": "同一用户",
     "verb": "发布", "unit": "次", "scopes": ("灰度发布", "正式发布", "回滚发布"),
     "items": ("版本说明", "评测报告", "审批记录"), "lists": ("发布清单", "回滚清单"),
     "years": (2025, 2027), "lo": 2, "qmax": 8,
     "slot_main": "release_gate", "object_main": "发布流程",
     "slot_secondary": "rollback_policy", "object_secondary": "回滚预案"},
    {"id": "ticket-triage", "name": "工单分流规范", "subject": "同一用户",
     "verb": "分流", "unit": "单", "scopes": ("线上故障", "咨询请求", "数据修复"),
     "items": ("优先级", "责任组", "回执"), "lists": ("待分清单", "升级清单"),
     "years": (2024, 2026), "lo": 3, "qmax": 10,
     "slot_main": "triage_deadline", "object_main": "工单",
     "slot_secondary": "receipt_policy", "object_secondary": "处置记录"},
    {"id": "storage-check", "name": "存储核对规范", "subject": "同一用户",
     "verb": "核对", "unit": "份", "scopes": ("冷备", "热备", "离线备份"),
     "items": ("校验和", "存储位置", "失败日志"), "lists": ("核对清单", "异常清单"),
     "years": (2025, 2026), "lo": 6, "qmax": 12,
     "slot_main": "verify_frequency", "object_main": "备份文件",
     "slot_secondary": "failure_policy", "object_secondary": "异常报告"},
]
SETTINGS_BY_ID = {item["id"]: item for item in SETTINGS}


def assert_anchors_disjoint() -> None:
    """v2.2：同一 setting 内 scopes/items/lists/object_main/object_secondary 两两不互相包含。"""
    problems = []
    for setting in SETTINGS:
        anchors = [("scope", value) for value in setting["scopes"]]
        anchors += [("item", value) for value in setting["items"]]
        anchors += [("list", value) for value in setting["lists"]]
        anchors += [("object_main", setting["object_main"]),
                    ("object_secondary", setting["object_secondary"])]
        for index, (kind_a, a) in enumerate(anchors):
            for kind_b, b in anchors[index + 1:]:
                if a == b or a in b or b in a:
                    problems.append(f"{setting['id']}: {kind_a}={a!r} ~ {kind_b}={b!r}")
        for year in setting["years"]:
            if not (setting["years"][0] <= year < setting["years"][0] + 8):                 
                problems.append(f"{setting['id']}: year {year} outside horizon")
    if problems:
        raise AssertionError("anchor collisions: " + "; ".join(problems[:10]))


assert_anchors_disjoint()

                                                                                        
                                     
                                                                                        

MECHANISMS = (
    {"id": "explicit_polarity", "name": "显式极性",
     "recipes": ("E1", "E2", "P1", "P3", "C1", "C3")},
    {"id": "scope_exception", "name": "作用域与例外",
     "recipes": ("E1", "E2", "P1", "P2", "C3", "C1")},
    {"id": "time_window", "name": "时间窗",
     "recipes": ("E1", "E2", "P5", "P1", "C4", "C1")},
    {"id": "numeric_boundary", "name": "数值边界",
     "recipes": ("E1", "E2", "P6", "P8", "C2", "C7")},
    {"id": "composite_cross_slot", "name": "复合与跨维度",
     "recipes": ("E1", "E3", "P4", "P3", "C6", "C3")},
    {"id": "reference_anaphora", "name": "指代与绑定",
     "recipes": ("E1", "E2", "P7", "P1", "C5", "C1")},
    {"id": "independent_obligation", "name": "独立义务",
     "recipes": ("E1", "E2", "P4", "P7", "C1", "C3")},
)
MECHANISMS_BY_ID = {item["id"]: item for item in MECHANISMS}

NATURAL_RECIPES = {
    "E1": {"label": "equivalent", "name": "同义改写", "expected": "equivalent"},
    "E2": {"label": "equivalent", "name": "区间改写", "expected": "equivalent"},
    "P1": {"label": "old_entails_new_strict", "name": "新增例外收窄", "expected": "compatible"},
    "P2": {"label": "new_entails_old_strict", "name": "移除例外放宽", "expected": "compatible"},
    "P3": {"label": "overlap_incomparable", "name": "作用域分离", "expected": "compatible"},
    "P4": {"label": "overlap_incomparable", "name": "跨维度独立", "expected": "compatible"},
    "P5": {"label": "overlap_incomparable", "name": "时间窗分离", "expected": "compatible"},
    "P6": {"label": "overlap_incomparable", "name": "数值区间交叠", "expected": "compatible"},
    "P7": {"label": "overlap_incomparable", "name": "指代不同对象", "expected": "compatible"},
    "P8": {"label": "new_entails_old_strict", "name": "同域数值收紧", "expected": "compatible"},
    "P10": {"label": "old_entails_new_strict", "name": "同域数值放宽", "expected": "compatible"},
    "C1": {"label": "disjoint", "name": "极性相反", "expected": "conflict"},
    "C2": {"label": "disjoint", "name": "数值区间不交", "expected": "conflict"},
    "C3": {"label": "disjoint", "name": "全域禁止覆盖局部要求", "expected": "conflict"},
    "C4": {"label": "disjoint", "name": "一次性窗口内相反", "expected": "conflict"},
    "C5": {"label": "disjoint", "name": "同一指代相反", "expected": "conflict"},
    "C6": {"label": "disjoint", "name": "次维度极性相反", "expected": "conflict"},
    "C7": {"label": "disjoint", "name": "同域数值区间不交", "expected": "conflict"},
    "E3": {"label": "equivalent", "name": "次维度同义改写", "expected": "equivalent"},
}

DYAD_RECIPES = {
    "D1": {"name": "极性单点翻转", "base": "equivalent", "field": "new.action"},
    "D2": {"name": "数值边界单点翻转", "base": "compatible", "field": "new.measure.high"},
    "D3": {"name": "作用域单点翻转", "base": "compatible", "field": "new.scope"},
    "D4": {"name": "时间窗单点翻转", "base": "compatible", "field": "new.window.year"},
    "D5": {"name": "指代单点翻转", "base": "compatible", "field": "new.referent"},
    "D6": {"name": "跨维度极性单点翻转", "base": "equivalent", "field": "new.action"},
                                                                 
                                       
    "D7": {"name": "数值收紧 E→P", "base": "equivalent", "flip": "compatible",
           "field": "new.measure.high"},
}

ALL_RECIPES = set(NATURAL_RECIPES) | set(DYAD_RECIPES)

                                                                        
CONTROL_PLAN = {
    ("D1", "base"): "E1", ("D1", "flip"): "C1",
    ("D2", "base"): "P10", ("D2", "flip"): "C7",
    ("D3", "base"): "P3", ("D3", "flip"): "C1",
    ("D4", "base"): "P5", ("D4", "flip"): "C4",
    ("D5", "base"): "P7", ("D5", "flip"): "C5",
    ("D6", "base"): "E3", ("D6", "flip"): "C6",
    ("D7", "base"): "E2", ("D7", "flip"): "P8",
}


def surface_for(recipe: str, side: str) -> "dict | None":
    """equivalent 类必须给出两种真实表面形式；side ∈ {old, new}。"""
    if recipe in ("E1", "E3"):
        return {"kind": "paraphrase", "form": "strong" if side == "old" else "soft"}
    if recipe in ("E2", "D7"):
        return {"kind": "numeric", "form": "range" if side == "old" else "bounds"}
    return None


                                                                                        
      
                                                                                        

def _measure(kind: str, low=None, high=None) -> dict:
    if kind == "none":
        return {"kind": "none", "low": None, "high": None}
    if kind == "at_least":
        return {"kind": kind, "low": int(low), "high": None}
    if kind == "at_most":
        return {"kind": kind, "low": None, "high": int(high)}
    if kind == "within":
        return {"kind": kind, "low": int(low), "high": int(high)}
    raise ValueError(f"unknown measure {kind!r}")


def factors(setting: dict, *, slot=None, object=None, scope=None, action="require",
            measure=None, exception=None, window=None, referent=None,
            items_scope=None) -> dict:
    return {
        "subject": setting["subject"],
        "items_scope": items_scope,
        "slot": slot or setting["slot_main"],
        "object": object or setting["object_main"],
        "scope": scope or setting["scopes"][0],
        "action": action,
        "measure": measure or _measure("none"),
        "exception": exception or {"kind": "none", "item": None},
        "window": window or {"kind": "long_term", "year": None},
        "referent": referent,
    }


def control_slot(setting: dict, index: int, k: int) -> dict:
    """控制行的坐标：与共享/反事实行错开作用域、数量、事项与年份奇偶。"""
    lo = min(setting["lo"] + ((index + k) % 4) + 1, setting["qmax"] - 5)
    scope_index = (index + k + 1) % len(setting["scopes"])
    first, second = setting["years"]
    if (index + k) % 2 == 0:
        near_year, far_year = first, second
    else:
        near_year, far_year = second, first
    return {"lo": max(lo, 0),
            "near_scope": setting["scopes"][scope_index],
            "far_scope": setting["scopes"][(scope_index + 1) % len(setting["scopes"])],
            "item": setting["items"][(index + k) % len(setting["items"])],
            "near_year": near_year, "far_year": far_year,
            "list_near": setting["lists"][0], "list_far": setting["lists"][1]}


def slot(setting: dict, index: int, position: int) -> dict:
    """族内第 position 行的坐标：数量、作用域、事项、年份都随 (index, position) 平移。

    数量上限固定在 ``qmax - 5``：保证 P6（at_least lo vs at_most lo+2）不会退化成包含关系，
    也保证 D2 的边界翻转两侧都留在同一数量宇宙内。
    """
    lo = min(setting["lo"] + (index % 4) + (position % 3), setting["qmax"] - 5)
    lo = max(lo, 0)
    scope_index = (index + position) % len(setting["scopes"])
    near_scope = setting["scopes"][scope_index]
    far_scope = setting["scopes"][(scope_index + 1) % len(setting["scopes"])]
    item = setting["items"][(index + position) % len(setting["items"])]
    first, second = setting["years"]
    if (index + position) % 2 == 0:
        near_year, far_year = first, second
    else:
        near_year, far_year = second, first
    alt = setting["items"][(setting["items"].index(item) + 1) % len(setting["items"])]
    return {"lo": lo, "near_scope": near_scope, "far_scope": far_scope,
            "item": item, "item_alt": alt, "near_year": near_year, "far_year": far_year,
            "list_near": setting["lists"][0], "list_far": setting["lists"][1]}


def _core(setting: dict, s: dict, **overrides) -> dict:
    base = {"scope": s["near_scope"], "items_scope": s.get("items_scope") or (s["item"],),
            "referent": s.get("referent")}
    base.update(overrides)
    return factors(setting, **base)


def build_natural(recipe: str, setting: dict, s: dict) -> tuple[dict, dict]:
    """自然行配方 → (old, new) 因素；标签由 classify 决定，不在此处声明。"""
    lo = s["lo"]
    if recipe == "E1":
        old = _core(setting, s)
        new = _core(setting, s)
    elif recipe == "E2":
        old = _core(setting, s, measure=_measure("within", lo, lo + 2))
        new = _core(setting, s, measure=_measure("within", lo, lo + 2))
    elif recipe == "P1":
        old = _core(setting, s, items_scope=None)
        new = _core(setting, s, items_scope=None,
                    exception={"kind": "except", "item": s["item"]})
    elif recipe == "P2":
        old = _core(setting, s, items_scope=None, exception={"kind": "except", "item": s["item"]})
        new = _core(setting, s, items_scope=None)
    elif recipe == "P3":
        old = _core(setting, s)
        new = _core(setting, s, scope=s["far_scope"],
                    items_scope=s.get("items_scope_alt") or (s["item_alt"],))
    elif recipe == "P4":
        old = _core(setting, s)
        new = _core(setting, s, slot=setting["slot_secondary"], object=setting["object_secondary"])
    elif recipe == "P5":
        old = _core(setting, s, window={"kind": "one_off", "year": s["near_year"]})
        new = _core(setting, s, window={"kind": "one_off", "year": s["far_year"]})
    elif recipe == "P6":
        old = _core(setting, s, measure=_measure("at_least", lo))
        new = _core(setting, s, measure=_measure("at_most", high=lo + 2))
    elif recipe == "P7":
        old = _core(setting, s, referent=s["list_near"])
        new = _core(setting, s, referent=s["list_far"])
    elif recipe == "P8":
        old = _core(setting, s, measure=_measure("at_least", lo))
        new = _core(setting, s, measure=_measure("at_least", lo + 1))
    elif recipe == "P10":
        old = _core(setting, s, measure=_measure("at_least", lo))
        new = _core(setting, s, measure=_measure("at_least", max(lo - 1, 0)))
    elif recipe == "C1":
        old = _core(setting, s, action="require")
        new = _core(setting, s, action="prohibit")
    elif recipe == "C2":
        old = _core(setting, s, measure=_measure("at_least", lo))
        new = _core(setting, s, measure=_measure("at_most", high=max(lo - 2, 0)))
    elif recipe == "C3":
        old = _core(setting, s, action="require")
        new = _core(setting, s, scope="全部", action="prohibit")
    elif recipe == "C4":
        old = _core(setting, s, action="require", window={"kind": "long_term", "year": None})
        new = _core(setting, s, action="prohibit", window={"kind": "one_off", "year": s["near_year"]})
    elif recipe == "C5":
        old = _core(setting, s, action="require", referent=s["list_near"])
        new = _core(setting, s, action="prohibit", referent=s["list_near"])
    elif recipe == "C6":
        old = _core(setting, s, slot=setting["slot_secondary"],
                    object=setting["object_secondary"], action="require")
        new = _core(setting, s, slot=setting["slot_secondary"],
                    object=setting["object_secondary"], action="prohibit")
    elif recipe == "C7":
        old = _core(setting, s, measure=_measure("at_least", lo))
        new = _core(setting, s, measure=_measure("within", 0, max(lo - 2, 0)))
    elif recipe == "E3":
        old = _core(setting, s, slot=setting["slot_secondary"],
                    object=setting["object_secondary"])
        new = _core(setting, s, slot=setting["slot_secondary"],
                    object=setting["object_secondary"])
    else:
        raise KeyError(f"unknown natural recipe {recipe!r}")
    return old, new


def build_dyad(recipe: str, setting: dict, s: dict) -> dict:
    """反事实：old / 基线 new / 翻转 new + 唯一改动 + 单点替换字面量。"""
    lo = s["lo"]
    if recipe == "D1":
        old = _core(setting, s, action="require")
        base_new = _core(setting, s, action="require")
        conflict_new = _core(setting, s, action="prohibit")
        change = {"field": "new.action", "from": "require", "to": "prohibit"}
        renders = {"from": "必须", "to": "不得"}
    elif recipe == "D2":
        hi, low = lo + 2, max(lo - 2, 0)
        old = _core(setting, s, measure=_measure("at_least", lo))
        base_new = _core(setting, s, measure=_measure("at_most", high=hi))
        conflict_new = _core(setting, s, measure=_measure("at_most", high=low))
        change = {"field": "new.measure.high", "from": hi, "to": low}
        renders = {"from": str(hi), "to": str(low)}
    elif recipe == "D3":
        old = _core(setting, s, action="require")
        base_new = _core(setting, s, scope=s["far_scope"], action="prohibit")
        conflict_new = _core(setting, s, scope=s["near_scope"], action="prohibit")
        change = {"field": "new.scope", "from": s["far_scope"], "to": s["near_scope"]}
        renders = {"from": s["far_scope"], "to": s["near_scope"]}
    elif recipe == "D4":
        old = _core(setting, s, action="require", window={"kind": "one_off", "year": s["near_year"]})
        base_new = _core(setting, s, action="prohibit", window={"kind": "one_off", "year": s["far_year"]})
        conflict_new = _core(setting, s, action="prohibit",
                             window={"kind": "one_off", "year": s["near_year"]})
        change = {"field": "new.window.year", "from": s["far_year"], "to": s["near_year"]}
        renders = {"from": str(s["far_year"]), "to": str(s["near_year"])}
    elif recipe == "D5":
        old = _core(setting, s, action="require", referent=s["list_near"])
        base_new = _core(setting, s, action="prohibit", referent=s["list_far"])
        conflict_new = _core(setting, s, action="prohibit", referent=s["list_near"])
        change = {"field": "new.referent", "from": s["list_far"], "to": s["list_near"]}
        renders = {"from": s["list_far"], "to": s["list_near"]}
    elif recipe == "D6":
        old = _core(setting, s, slot=setting["slot_secondary"], object=setting["object_secondary"],
                    action="require")
        base_new = _core(setting, s, slot=setting["slot_secondary"], object=setting["object_secondary"],
                         action="require")
        conflict_new = _core(setting, s, slot=setting["slot_secondary"],
                             object=setting["object_secondary"], action="prohibit")
        change = {"field": "new.action", "from": "require", "to": "prohibit"}
        renders = {"from": "必须", "to": "不得"}
    elif recipe == "D7":
        hi = lo + 2
        old = _core(setting, s, measure=_measure("within", lo, hi))
        base_new = _core(setting, s, measure=_measure("within", lo, hi))
        conflict_new = _core(setting, s, measure=_measure("within", lo, hi - 1))
        change = {"field": "new.measure.high", "from": hi, "to": hi - 1}
        renders = {"from": str(hi), "to": str(hi - 1)}
    else:
        raise KeyError(f"unknown dyad recipe {recipe!r}")
    return {"old": old, "base_new": base_new, "flip_new": conflict_new,
            "change": change, "renders": renders}


                                                                                        
               
                                                                                        

_HORIZON = 8


def _applicability(f: dict) -> set:
    """(作用域, 事项, 年份) 三元组集合；``全部`` 展开成全部作用域。"""
    scopes = f.get("_scopes") or ()
    scope = f["scope"]
    selected = scopes if scope in ("全部", "all") else (scope,)
    scope_items = f.get("items_scope")
    if scope_items is not None:
        items = list(scope_items) if isinstance(scope_items, (list, tuple)) else [scope_items]
        items = [item for item in items if item in (f.get("_items") or ())]
    else:
        items = list(f.get("_items") or ())
    exception = f.get("exception") or {"kind": "none", "item": None}
    if exception.get("kind") == "except" and exception.get("item") in items:
        items = [item for item in items if item != exception["item"]]
    window = f.get("window") or {"kind": "long_term", "year": None}
    if window.get("kind") == "one_off":
        years = (int(window["year"]),)
    else:
        base = int(f.get("_year_base") or 2024)
        years = tuple(range(base, base + _HORIZON))
    return {(scope_name, item, year) for scope_name in selected for item in items for year in years}


def _allowed(f: dict) -> frozenset:
    qmax = int(f.get("_qmax") or 12)
    universe = frozenset(range(0, qmax + 1))
    measure = f.get("measure") or {"kind": "none"}
    kind = measure["kind"]
    if kind == "none":
        return frozenset() if f["action"] == "prohibit" else universe
    low, high = measure.get("low"), measure.get("high")
    if kind == "at_least":
        inside = frozenset(range(int(low), qmax + 1))
    elif kind == "at_most":
        inside = frozenset(range(0, int(high) + 1))
    elif kind == "within":
        inside = frozenset(range(int(low), int(high) + 1))
    else:
        raise ValueError(kind)
    if f["action"] == "prohibit":
        return universe - inside
    return inside


def with_setting(f: dict, setting: dict) -> dict:
    """把领域坐标注入因素（不改变因素本身的语义字段）。"""
    enriched = dict(f)
    enriched["_scopes"] = tuple(setting["scopes"])
    enriched["_items"] = tuple(setting["items"])
    enriched["_qmax"] = setting["qmax"]
    enriched["_year_base"] = setting["years"][0]
    return enriched


def classify(old: dict, new: dict) -> str:
    """五叶标签：只由因素决定。方向约定：``old_entails_new_strict`` = old 更强、old 蕴含 new。"""
    if old["subject"] != new["subject"] or old["slot"] != new["slot"] or old["object"] != new["object"]:
        return "overlap_incomparable"
    if (old.get("referent") or "") != (new.get("referent") or ""):
        return "overlap_incomparable"
    applies_old, applies_new = _applicability(old), _applicability(new)
    if not applies_old or not applies_new:
        return "overlap_incomparable"                    
    allowed_old, allowed_new = _allowed(old), _allowed(new)
    common = applies_old & applies_new
    if common and not (allowed_old & allowed_new):
        return "disjoint"
    if applies_old == applies_new and allowed_old == allowed_new:
        return "equivalent"
    old_stronger = applies_old >= applies_new and allowed_old <= allowed_new
    new_stronger = applies_new >= applies_old and allowed_new <= allowed_old
    if old_stronger and not new_stronger:
        return "old_entails_new_strict"
    if new_stronger and not old_stronger:
        return "new_entails_old_strict"
    return "overlap_incomparable"


def classify_with(old: dict, new: dict, setting: dict) -> str:
    return classify(with_setting(old, setting), with_setting(new, setting))


def factor_problems(f: dict, setting: dict) -> list[str]:
    problems = []
    selected = setting["items"] if f.get("items_scope") is None else f["items_scope"]
    exception = f.get("exception") or {}
    if exception.get("kind") == "except" and exception.get("item") not in selected:
        problems.append("inert_exception: excluded item is outside the selected item domain")
    if not _applicability(with_setting(f, setting)):
        problems.append("empty_applicability: selected items cannot all be excluded")
    return problems


def effective_factor_differences(left: dict, right: dict, setting: dict) -> list[str]:
    def view(f):
        applicable = _applicability(with_setting(f, setting))
        inside = _allowed(with_setting(dict(f, action="require"), setting))
        return {"subject": f["subject"], "slot": f["slot"], "object": f["object"],
                "referent": f.get("referent") or "", "action": f["action"],
                "scope_domain": tuple(sorted({cell[0] for cell in applicable})),
                "item_domain": tuple(sorted({cell[1] for cell in applicable})),
                "year_domain": tuple(sorted({cell[2] for cell in applicable})),
                "quantity_low": min(inside) if inside else None,
                "quantity_high": max(inside) if inside else None}
    a, b = view(left), view(right)
    return [key for key in a if a[key] != b[key]]


def reference_partition_text(setting: dict) -> str:
    return f"本场景中「{setting['lists'][0]}」与「{setting['lists'][1]}」所列条目互不重叠。"


                                                                                        
          
                                                                                        

POLARITY_TOKENS = {
    "require": ("必须", "需要", "应当", "都要", "须", "务必"),
    "prohibit": ("不得", "禁止", "不允许", "不能", "严禁", "不准"),
}
STRONG_TOKENS = {"require": ("必须", "须", "务必"), "prohibit": ("不得", "严禁")}
SOFT_TOKENS = {"require": ("需要", "应当"), "prohibit": ("禁止", "不允许")}
                          
NEGATION_PATTERNS = {
    "require": ("不需要", "不必", "无须", "无需", "并非", "不是", "不用", "不必再"),
    "prohibit": ("不得不",),
}
MEASURE_TOKENS = {
    "at_least": ("至少", "不少于", "不低于"),
    "at_most": ("至多", "不多于", "不超过", "最多"),
}
WINDOW_TOKENS = {"one_off": ("仅", "只", "一次性", "当年", "当年内")}
FORBIDDEN_TOKENS = ("冲突", "等价", "蕴含", "矛盾", "标签", "规格", "训练数据", "标注规则")
SCOPE_ALL_TOKENS = ("所有", "全部", "任何")
_RANGE_RE = re.compile(r"\d+\s*(到|至|～|~|-)\s*\d+\s*[^。；]{0,8}(之间|以内)")
_RANGE_HEAD = ("在", "从", "介于", "范围是")
_BOUNDS_LOW = ("不少于", "至少", "不低于")
_BOUNDS_HIGH = ("不多于", "至多", "不超过", "最多")
_INCLUSIVE_EXCEPTION_RE = re.compile(r"除了[^。；]{0,24}(还|也|仍|并且|而且)|之外还|外还")


def _has_any(text: str, tokens) -> bool:
    return any(token in text for token in tokens)


def _range_form(text: str) -> bool:
    return bool(_RANGE_RE.search(text)) and _has_any(text, _RANGE_HEAD)


def _bounds_form(text: str) -> bool:
    return _has_any(text, _BOUNDS_LOW) and _has_any(text, _BOUNDS_HIGH)


def expected_tokens(f: dict, setting: dict) -> dict:
    measure = f.get("measure") or {"kind": "none"}
    window = f.get("window") or {"kind": "long_term", "year": None}
    exception = f.get("exception") or {"kind": "none", "item": None}
    numbers = []
    if measure.get("low") is not None:
        numbers.append(int(measure["low"]))
    if measure.get("high") is not None and measure.get("high") != measure.get("low"):
        numbers.append(int(measure["high"]))
    if window.get("kind") == "one_off":
        numbers.append(int(window["year"]))
    scope_tokens = SCOPE_ALL_TOKENS if f["scope"] in ("全部", "all") else (f["scope"],)
    return {
        "numbers": numbers,
        "polarity_classes": [POLARITY_TOKENS[f["action"]]],
        "scope_tokens": list(scope_tokens),
        "measure_kind": measure["kind"],
        "exception_item": exception.get("item") if exception.get("kind") == "except" else None,
        "window_tokens": list(WINDOW_TOKENS["one_off"]) if window["kind"] == "one_off" else [],
        "referent": f.get("referent"),
        "items_scope": f.get("items_scope"),
        "object_phrase": f["object"],
    }


def check_text_gates(text: str, f: dict, setting: dict, *, exact_tokens: Sequence[str] = (),
                     surface: "dict | None" = None) -> list[str]:
    """文本门同时拒绝空适用域与无效例外，不能靠 token 齐全掩盖。"""
    problems: list[str] = factor_problems(f, setting)
    if not isinstance(text, str) or not text.strip():
        return ["empty text"]
    if len(text) > 90:
        problems.append(f"too long ({len(text)} chars)")
    if len(text) < 8:
        problems.append(f"too short ({len(text)} chars)")
    spec = expected_tokens(f, setting)
    for number in spec["numbers"]:
        if str(number) not in text:
            problems.append(f"missing number {number}")
    action = f["action"]
    if _has_any(text, NEGATION_PATTERNS[action]):
        problems.append(f"negated polarity pattern for {action}")
    if not _has_any(text, spec["polarity_classes"][0]):
        problems.append("missing polarity token")
    if not _has_any(text, spec["scope_tokens"]):
        problems.append(f"missing scope token from {spec['scope_tokens']}")
    kind = spec["measure_kind"]
    if kind == "at_least" and not _has_any(text, MEASURE_TOKENS["at_least"]):
        problems.append("missing at_least token")
    if kind == "at_most" and not _has_any(text, MEASURE_TOKENS["at_most"]):
        problems.append("missing at_most token")
    if kind == "within" and not (_range_form(text) or _bounds_form(text)):
        problems.append("within needs explicit range or bounds form")
    if spec["exception_item"]:
        if spec["exception_item"] not in text:
            problems.append(f"missing exception item {spec['exception_item']}")
        if "除" not in text and "除外" not in text:
            problems.append("missing exclusive 除…外 phrasing")
        if _INCLUSIVE_EXCEPTION_RE.search(text):
            problems.append("inclusive 除了…还 pattern is not exclusive")
    if spec["items_scope"]:
        for name in spec["items_scope"]:
            if name not in text:
                problems.append(f"missing item scope {name}")
        if "只针对" not in text:
            problems.append("item scope must use 只针对 phrasing")
    if spec["referent"]:
        if spec["referent"] not in text:
            problems.append(f"missing referent {spec['referent']}")
        if not _has_any(text, ("所列", "范围内", "条目", f"{spec['referent']}中的")):
            problems.append("referent must explicitly identify the listed item domain")
    if spec["object_phrase"] not in text:
        problems.append(f"missing object phrase {spec['object_phrase']}")
    for token in FORBIDDEN_TOKENS:
        if token in text:
            problems.append(f"forbidden token {token}")
    for token in exact_tokens:
        if text.count(token) != 1:
            problems.append(f"exact token {token!r} appears {text.count(token)} times")
    if surface:
        form = surface.get("form")
        if surface.get("kind") == "numeric" and form == "range":
            if not _range_form(text):
                problems.append("range surface must use 在 X 到 Y 之间")
            if _bounds_form(text):
                problems.append("range surface must not use bounds wording")
        elif surface.get("kind") == "numeric" and form == "bounds":
            if not _bounds_form(text):
                problems.append("bounds surface must use 不少于/不多于")
            if _range_form(text):
                problems.append("bounds surface must not use range wording")
    return problems


def surface_warnings(text: str, f: dict, surface: "dict | None") -> list[str]:
    if not surface or surface.get("kind") != "paraphrase":
        return []
    strong = surface.get("form") == "strong"
    preferred = STRONG_TOKENS if strong else SOFT_TOKENS
    other = SOFT_TOKENS if strong else STRONG_TOKENS
    present = _has_any(text, preferred[f["action"]])
    warnings = [] if present else ["preferred_paraphrase_token_missing"]
    if present and _has_any(text, other[f["action"]]):
        warnings.append("mixed_paraphrase_style")
    return warnings


def single_substitution(text: str, from_render: str, to_render: str) -> tuple[str, list[str]]:
    """把文本里的一个 from_render 换成 to_render；要求 from_render 恰好出现一次。"""
    if not from_render or not to_render or from_render == to_render:
        return text, ["invalid substitution tokens"]
    count = text.count(from_render)
    if count != 1:
        return text, [f"from_render {from_render!r} appears {count} times"]
    derived = text.replace(from_render, to_render, 1)
    problems = []
    if derived.count(to_render) != 1:
        problems.append(f"to_render {to_render!r} appears {derived.count(to_render)} times")
    return derived, problems


_CN_DIGITS = "零一二三四五六七八九"


def cn_number(value: int) -> str:
    """1..99 的中文数字（fixture 行标记用，避免阿拉伯数字干扰单点替换）。"""
    if value < 10:
        return _CN_DIGITS[value]
    tens, ones = divmod(value, 10)
    head = "十" if tens == 1 else _CN_DIGITS[tens] + "十"
    return head if ones == 0 else head + _CN_DIGITS[ones]


def render_fixture(f: dict, setting: dict, *, style: str = "old", tag: "str | None" = None,
                   surface: "dict | None" = None) -> str:
    """selftest 专用机械句（不是真实作者文本；manifest 会标 fixture）。

    ``style="new"`` 用另一套语序与近义词表达**同一因素**，让等价对的两句文本不同；
    dyad 的基线 new 必须用 ``style="dyad"``，以保证单点替换字面量恰好出现一次。
    ``tag`` 只给 fixture 用来保证行文本唯一（真实作者靠措辞差异），不进入任何语义门。
    """
    measure = f.get("measure") or {"kind": "none"}
    window = f.get("window") or {"kind": "long_term", "year": None}
    exception = f.get("exception") or {"kind": "none", "item": None}
    scope = "所有场景" if f["scope"] in ("全部", "all") else f["scope"]
    items_phrase = ""
    if f.get("items_scope"):
        names = "、".join(f["items_scope"])
        items_phrase = (f"，只针对「{names}」这一项" if len(f["items_scope"]) == 1
                        else f"，只针对「{names}」这两项")
    referent = f"，只针对「{f['referent']}」所列条目" if f.get("referent") else ""
    exception_text = f"，除「{exception['item']}」外" if exception["kind"] == "except" else ""
    suffix = f"（{tag}）" if tag else ""
    if style == "old":
        polarity = "必须" if f["action"] == "require" else "不得"
        parts = [f"{scope}里，{f['subject']}{polarity}{setting['verb']}{f['object']}"]
        if measure["kind"] == "at_least":
            parts.append(f"，数量不少于 {measure['low']} {setting['unit']}")
        elif measure["kind"] == "at_most":
            parts.append(f"，数量不多于 {measure['high']} {setting['unit']}")
        elif measure["kind"] == "within":
            parts.append(f"，数量在 {measure['low']} 到 {measure['high']} {setting['unit']}之间")
        if window["kind"] == "one_off":
            parts.append(f"，仅限 {window['year']} 年一次性执行")
        else:
            parts.append("，长期有效")
        parts.append(exception_text + items_phrase + referent)
        return "".join(parts) + suffix + "。"
    if style in ("new", "dyad"):
        polarity = ("需要" if f["action"] == "require" else "禁止") if style == "new" \
            else ("必须" if f["action"] == "require" else "不得")
        if window["kind"] == "one_off":
            head = f"{window['year']} 年当年一次性执行："
        else:
            head = "长期生效的要求是：" if style == "new" else "长期有效，"
        parts = [head, f"{scope}范围内{f['subject']}{polarity}{setting['verb']}{f['object']}"]
        if measure["kind"] == "at_least":
            parts.append(f"，至少 {measure['low']} {setting['unit']}")
        elif measure["kind"] == "at_most":
            parts.append(f"，最多 {measure['high']} {setting['unit']}")
        elif measure["kind"] == "within":
            if surface and surface.get("form") == "bounds":
                parts.append(f"，不少于 {measure['low']} 且不多于 {measure['high']} {setting['unit']}")
            else:
                parts.append(f"，在 {measure['low']} 至 {measure['high']} {setting['unit']}之间")
        if exception["kind"] == "except":
            parts.append(f"；「{exception['item']}」除外")
        if f.get("items_scope"):
            names = "、".join(f["items_scope"])
            parts.append(f"；只针对「{names}」这一项" if len(f["items_scope"]) == 1
                         else f"；只针对「{names}」这两项")
        if f.get("referent"):
            parts.append(f"；只针对「{f['referent']}」所列条目")
        return "".join(parts) + suffix + "。"
    raise ValueError(f"unknown render style {style!r}")


def describe(f: dict, setting: dict) -> str:
    """给作者的语义说明（不是最终句；只陈述因素，不含标签）。"""
    measure = f.get("measure") or {"kind": "none"}
    window = f.get("window") or {"kind": "long_term", "year": None}
    exception = f.get("exception") or {"kind": "none", "item": None}
    scope = "所有场景" if f["scope"] in ("全部", "all") else f"「{f['scope']}」"
    action = "必须" if f["action"] == "require" else "禁止"
    pieces = [f"{f['subject']}在{scope}里{action}{setting['verb']}{f['object']}"]
    if measure["kind"] == "at_least":
        pieces.append(f"，数量不少于 {measure['low']} {setting['unit']}")
    elif measure["kind"] == "at_most":
        pieces.append(f"，数量不多于 {measure['high']} {setting['unit']}")
    elif measure["kind"] == "within":
        pieces.append(f"，数量在 {measure['low']} 到 {measure['high']} {setting['unit']}之间")
    if window["kind"] == "one_off":
        pieces.append(f"；只在 {window['year']} 年这一次生效")
    else:
        pieces.append("；长期生效")
    if exception["kind"] == "except":
        pieces.append(f"；「{exception['item']}」除外")
    if f.get("items_scope"):
        names = "、".join(f["items_scope"])
        pieces.append(f"；只针对「{names}」这一项" if len(f["items_scope"]) == 1
                      else f"；只针对「{names}」这两项")
    if f.get("referent"):
        pieces.append(f"；只针对「{f['referent']}」所列条目")
    return "".join(pieces) + "。"


                                                                                        
           
                                                                                        

def family_plan(index: int) -> dict:
    """setting 固定按 5 族分块；机制在块内轮换，避免 setting×mechanism 相关。"""
    setting = setting_of_family(index)
    mechanism = MECHANISMS[(index + index // len(SETTINGS)) % len(MECHANISMS)]
    return {"index": index, "mechanism": mechanism["id"], "setting": setting["id"],
            "recipes": mechanism["recipes"]}


def split_assignment(total: int) -> list[str]:
    """v2.2：12 setting × 5 family；每 setting 2 train / 1 dev / 2 heldout。"""
    if total != len(SETTINGS) * 5:
        raise ValueError(f"v2.2 expects {len(SETTINGS) * 5} families, got {total}")
    import hashlib
    assignment = [None] * total
    for setting_index, setting in enumerate(SETTINGS):
        base = setting_index * 5
        order = sorted(range(5), key=lambda k: hashlib.sha256(
            f"cf-v22-split:{setting['id']}:{k}".encode()).hexdigest())
        for position, k in enumerate(order):
            assignment[base + k] = "train" if position < 2 else ("dev" if position < 3 else "heldout")
    return assignment


def setting_of_family(index: int) -> dict:
    return SETTINGS[(index // 5) % len(SETTINGS)]


def dyad_recipes_for_family(index: int) -> list[tuple[str, str, str]]:
    """3 个 dyad（v2.2）：(recipe, base_label, flip_label)。

    * 第 1 个：base compatible → flip conflict（D2..D5 轮换：数值/作用域/时间/指代）；
    * 第 2 个：base equivalent → flip conflict（D1/D6 交替：主/次维度极性）；
    * 第 3 个：base equivalent → flip compatible（D7：数值收紧），**不含 conflict 侧**。
    """
    cycle = ["D2", "D3", "D4", "D5"]
    conflict_from_compatible = cycle[index % len(cycle)]
    conflict_from_equivalent = "D1" if index % 2 == 0 else "D6"
    return [(conflict_from_compatible, "compatible", "conflict"),
            (conflict_from_equivalent, "equivalent", "conflict"),
            ("D7", "equivalent", "compatible")]


def label_plan_check() -> dict:
    """自检：共享 2/2/2 + dyad(2C+2P+2E) + 控制镜像 2/2/2 → 族内 6/6/6（18 行）。"""
    shared = {name: 0 for name in COARSE}
    for label in SHARED_LABEL_PLAN:
        shared[label] += 1
    if shared != {"conflict": 2, "compatible": 2, "equivalent": 2}:
        raise AssertionError(f"shared label plan {shared} != 2/2/2")
    def coarse_of(label: str) -> str:
        return label if label in COARSE else FINE_TO_COARSE[label]

    dyads = {name: 0 for name in COARSE}
    for _, base, flip in DYAD_LABEL_PLAN:
        dyads[coarse_of(base)] += 1
        dyads[coarse_of(flip)] += 1
    if dyads != {"conflict": 2, "compatible": 2, "equivalent": 2}:
        raise AssertionError(f"dyad label plan {dyads} != 2/2/2")
    total = {name: shared[name] + dyads[name] + dyads[name] for name in COARSE}
    if total != {"conflict": 6, "compatible": 6, "equivalent": 6}:
        raise AssertionError(f"family label total {total} != 6/6/6")
    return {"shared": shared, "dyads": dyads, "family_total": total}


def mechanism_label_check() -> dict:
    """每个机制的 6 个配方都必须给出 2E/2P/2C，否则拒绝构造。"""
    report = {}
    for mechanism in MECHANISMS:
        labels = [NATURAL_RECIPES[recipe]["expected"] for recipe in mechanism["recipes"]]
        counts = {name: labels.count(name) for name in COARSE}
        if counts != {"conflict": 2, "compatible": 2, "equivalent": 2}:
            raise AssertionError(f"{mechanism['id']} label mix {counts}")
        report[mechanism["id"]] = counts
    return report


def recipe_labels(recipe: str) -> str:
    return NATURAL_RECIPES[recipe]["label"] if recipe in NATURAL_RECIPES else "disjoint"
