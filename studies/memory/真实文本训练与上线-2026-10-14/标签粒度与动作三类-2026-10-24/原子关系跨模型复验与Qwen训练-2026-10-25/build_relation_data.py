                      
"""原子关系数据构造与审计 CLI（EXP 内单文件，仅标准库）。

子命令：specs / author-ollama / assemble / audit / selftest。

五叶名称、顺序、pair 校验与唯一序列化都来自 repo
``dataset_recommender.agent.semantic_relations``；本文件不复制五叶校验逻辑，
只负责规格生成、作者调用的记账、最终数据组装与结构审计。标签由规格预分配，
不由模型投票。

叙事口径（content_revision=memory-workflow-v2）：每个 family 描述同一用户在同一
工作流作用域内的一条长期偏好或要求；old 为前提 P，new 为假设 H，比较的是同一批
可能行为方案下的真值集合，而不是一次性物理计数事实。
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.util
import inspect
import json
import re
import shutil
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = Path(r"<archive-root>")
DEFAULT_SOURCE_ROWS = (
    DEFAULT_REPO_ROOT
    / "artifacts/worktrees/semantic-teacher-bind-20261014/research/memory_relations"
    / "runs/v7-fine-development-20261014/teacher-completion-20261015-c71"
    / "delivery/source_rows.jsonl"
)
DEFAULT_SPECS = EXP_ROOT / "data/specs/specs.jsonl"
DEFAULT_PROCESSED = EXP_ROOT / "data/processed"
DEFAULT_MANIFEST = EXP_ROOT / "data/manifest.json"
DEFAULT_QUARANTINE = EXP_ROOT / "data/quarantine.json"
DEFAULT_MODEL = "batiai/qwen3.8-27b:iq3"

CONTENT_REVISION = "memory-workflow-v3.2"
SPEC_SCHEMA = "relation-spec-v3"
NARRATIVE = "assistant-memory-workflow"

SPLITS = ("train", "dev", "calibration", "heldout")
SPLIT_ORDER = ("train", "train", "train", "train", "dev", "calibration", "heldout")
FAMILY_COUNT = 70
ITEMS_PER_FAMILY = 15
PER_FINE = 3
NUM_PREDICT_REQUESTED = 4096
MAX_ATTEMPTS = 2
ROW_SOURCE_LEGACY = "legacy-514"
ROW_SOURCE_NEW = "new-authored"
LEGACY_EXPECTED = {"train": 401, "dev": 113, "calibration": 0, "heldout": 0}
FAMILY_EXPECTED = {"train": 40, "dev": 10, "calibration": 10, "heldout": 10}

_REQUIRED_MECHANISMS = (
    "numeric_threshold",
    "scope_inclusion",
    "extra_conjunct",
    "independent_obligation",
    "negation_exception",
    "longterm_one_off",
)
                                                                 
SPLIT_MECHANISM_QUOTAS = {
    "train": {
        "numeric_threshold": 6, "scope_inclusion": 6, "extra_conjunct": 6,
        "independent_obligation": 6, "negation_exception": 8, "longterm_one_off": 8,
    },
    "dev": {
        "numeric_threshold": 2, "scope_inclusion": 2, "extra_conjunct": 2,
        "independent_obligation": 2, "negation_exception": 1, "longterm_one_off": 1,
    },
    "calibration": {
        "numeric_threshold": 2, "scope_inclusion": 2, "extra_conjunct": 2,
        "independent_obligation": 2, "negation_exception": 1, "longterm_one_off": 1,
    },
    "heldout": {
        "numeric_threshold": 2, "scope_inclusion": 2, "extra_conjunct": 2,
        "independent_obligation": 2, "negation_exception": 1, "longterm_one_off": 1,
    },
}
                                                      
SETTING_FAMILY_SHAPE = {"triples": 10, "pairs": 20}
SETTING_SPLIT_SLOTS = {
    "train": {"triples": 4, "pairs": 14},
    "dev": {"triples": 2, "pairs": 2},
    "calibration": {"triples": 2, "pairs": 2},
    "heldout": {"triples": 2, "pairs": 2},
}
_MAX_BACKGROUND_CHARS = 240
_MIN_SENTENCE_CHARS = 10
_MAX_SENTENCE_CHARS = 26
_BANNED_NARRATIVE = ("登记", "车库", "分区", "货架", "物件", "盘点")
                       
_DIRECTION_WORDS = ("更严", "更宽", "更强", "更弱", "更紧", "更松", "全部遵守", "没有任何", "必然", "一定")
                    
_BACKGROUND_REQUIREMENTS = {
    "numeric_threshold": ("整数",),
    "scope_inclusion": ("高优先级", "需要引用", "本人确认", "非空"),
    "extra_conjunct": ("彼此独立",),
    "independent_obligation": ("彼此独立的义务",),
    "negation_exception": (
        "名单甲真包含名单乙",
        "名单丙、名单丁、名单戊、名单己、名单庚、名单辛",
        "两两互不包含",
        "部分重叠",
        "名单内外都有条目",
    ),
    "longterm_one_off": (
        "时间窗",
        "窗内可追加、暂停或替代",
        "窗外长期要求不变",
        "高优先级",
    ),
}
                                   
_VARIANT_TEXT_REQUIREMENTS = {
    "new_subset_by_scope_restricted": ("名单甲", "豁免", "非高优先级", "一律"),
    "old_subset_by_narrow_exception": ("名单甲", "名单乙"),
    "new_subset_by_narrow_exception": ("名单甲", "名单乙"),
    "distinct_exception_sets": ("名单丙", "名单丁"),
    "exclusive_exemption_list": ("名单戊", "名单己", "不同名单"),
    "old_subset_by_extra_one_off": ("不得在同一维度上放宽",),
    "old_subset_by_two_one_offs": ("不得在同一维度上放宽",),
    "old_subset_by_one_off_scope": ("不得在同一维度上放宽",),
    "new_subset_by_temporary_exception": ("不得在同一维度上放宽",),
    "new_subset_by_two_temporary_exceptions": ("不得在同一维度上放宽",),
    "new_subset_by_one_off_scope": ("不得在同一维度上放宽",),
    "mutually_exclusive_threshold": ("不少于", "不多于"),
    "mutually_exclusive_exact_values": ("恰好", "恰好"),
    "contradictory_negation_with_threshold": ("不少于", "少于"),
    "contradictory_negation": ("全部事项都遵守", "没有任何事项遵守"),
    "contradictory_negation_priority": ("高优先级事项全部遵守", "没有任何高优先级事项遵守"),
    "exclusive_default_choice": ("从严", "从宽"),
    "contradictory_conjunct": ("不得",),
    "contradictory_conjunct_on_other_side": ("不得",),
    "exclusive_conjunct_choice": ("不得",),
    "contradictory_duty": ("不得",),
    "contradictory_duty_window": ("不得",),
    "exclusive_duty_choice": ("只要求", "不得"),
    "one_off_suspends_rule": ("照常适用", "暂停执行"),
    "opposite_one_off_directive": ("不得",),
    "one_off_replaces_rule": ("取代", "只有一份要求"),
}
                                       
_VARIANT_LIST_ALLOWED = {
    "old_subset_by_narrow_exception": frozenset("甲乙"),
    "new_subset_by_narrow_exception": frozenset("甲乙"),
    "distinct_exception_sets": frozenset("丙丁"),
    "exclusive_exemption_list": frozenset("戊己"),
    "new_subset_by_scope_restricted": frozenset("甲"),
}
_LIST_CHARS = "甲乙丙丁戊己庚辛"
                                   
_PREMISE_KEYWORDS = {
    "名单甲": "名单甲",
    "名单乙": "名单甲",
    "名单丙": "名单丙",
    "名单丁": "名单丙",
    "名单戊": "名单戊",
    "名单己": "名单戊",
    "高优先级": "高优先级",
    "需要引用": "需要引用",
    "本人确认": "本人确认",
    "第一周": "时间窗",
    "第三周": "时间窗",
    "整数": "整数",
}
_DISJOINT_SOFT_WORDS = ("不要求", "无需", "可以")
                              
_AUTHOR_DISJOINT_BANNED = ("不要求", "不作要求", "无需", "可以", "允许", "不设", "取消", "不必")


                                                   


class DataError(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8", newline="\n")
    temp.replace(path)


def write_json_atomic(path: Path, value) -> None:
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def write_jsonl_atomic(path: Path, rows) -> None:
    text = "".join(canonical_json(row) + "\n" for row in rows)
    write_text_atomic(path, text)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def bounded(items, limit: int = 20):
    values = list(items)
    return {"count": len(values), "items": values[:limit], "truncated": len(values) > limit}


def emit(payload, *, limit: int = 60000) -> None:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    if len(text) > limit:
        payload = {
            "ok": payload.get("ok"),
            "note": "output_truncated",
            "keys": sorted(payload),
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    print(text)


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "").lower()
    value = re.sub(r"\s+", " ", value).strip()
    return value


                                                           

_SEMANTIC = None


def semantic(repo_root: Path | None = None):
    global _SEMANTIC
    if _SEMANTIC is None:
        root = Path(repo_root) if repo_root else DEFAULT_REPO_ROOT
        src = root / "src"
        module_path = src / "dataset_recommender/agent/semantic_relations.py"
        if not module_path.is_file():
            raise DataError("semantic_module_missing", str(module_path))
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from dataset_recommender.agent import semantic_relations as module

        _SEMANTIC = module
    return _SEMANTIC


def relation_mapping():
    """FINE 顺序取生产常量；COARSE 由唯一映射锚点派生，任何漂移立即报错。"""
    fine = tuple(semantic().SEMANTIC_RELATIONS)
    anchor = {
        "equivalent": "equivalent",
        "old_entails_new_strict": "compatible",
        "new_entails_old_strict": "compatible",
        "disjoint": "conflict",
        "overlap_incomparable": "compatible",
    }
    if tuple(anchor) != fine:
        raise DataError("fine_label_order_mismatch", f"{tuple(anchor)} != {fine}")
    coarse = tuple(dict.fromkeys(anchor[name] for name in fine))
    if coarse != ("equivalent", "compatible", "conflict"):
        raise DataError("coarse_label_order_mismatch", str(coarse))
    return fine, coarse, anchor


                                                           

CHECK_SETS = {
    "search": ("标注数据来源", "记录检索式", "附上筛选理由"),
    "report": ("标注引用编号", "附上章节提纲", "统一术语表"),
    "storage": ("记录校验和", "附上存储位置", "保留失败日志"),
    "experiment": ("记录随机种子", "附上评测命令", "保存失败样本"),
    "archive": ("记录时间戳", "附上归档位置", "保留原始版本"),
    "review": ("标注证据等级", "附上审批记录", "保留拒绝理由"),
}


                                                       
                                           
                                                  

SETTINGS = [
    {
        "id": "search-result-limit",
        "name": "检索结果展示上限",
        "entity": "检索结果",
        "unit": "条",
        "owner": "检索助手",
        "detail": "每次检索只按同一份长期上限决定展示多少条",
        "zones": ["单细胞数据集检索", "乳腺癌项目检索", "方法学文献检索"],
        "check_set": "search",
        "num_bases": [9, 12, 15],
        "step": 3,
    },
    {
        "id": "attachment-size-limit",
        "name": "附件大小上限",
        "entity": "附件",
        "unit": "MB",
        "owner": "上传助手",
        "detail": "单个附件按一份长期上限校验，超限就拒收",
        "zones": ["数据集附件上传", "补充材料上传", "共享盘同步"],
        "check_set": "storage",
        "num_bases": [150, 200, 300],
        "step": 50,
    },
    {
        "id": "citation-count",
        "name": "汇报引用条数",
        "entity": "参考文献",
        "unit": "条",
        "owner": "汇报助手",
        "detail": "每份汇报单独统计引用条数，不跨汇报累计",
        "zones": ["组会周报", "给research collaborator的汇报", "结题汇报"],
        "check_set": "report",
        "num_bases": [12, 15, 18],
        "step": 3,
    },
    {
        "id": "report-length",
        "name": "汇报正文字数",
        "entity": "汇报正文",
        "unit": "字",
        "owner": "汇报助手",
        "detail": "正文按汉字计，附录不计入",
        "zones": ["组会周报", "给导师的汇报", "答辩材料"],
        "check_set": "report",
        "num_bases": [900, 1200, 1500],
        "step": 300,
    },
    {
        "id": "download-concurrency",
        "name": "下载并发上限",
        "entity": "并发下载任务",
        "unit": "个",
        "owner": "下载助手",
        "detail": "同一批次共用一份并发上限",
        "zones": ["数据集批量下载", "引文补充下载", "公共数据库镜像下载"],
        "check_set": "storage",
        "num_bases": [6, 8, 10],
        "step": 2,
    },
    {
        "id": "retry-budget",
        "name": "自动重试次数",
        "entity": "自动重试",
        "unit": "次",
        "owner": "任务助手",
        "detail": "同一任务只累计一份重试预算",
        "zones": ["外部 API 任务", "本地模型任务", "批量文件任务"],
        "check_set": "experiment",
        "num_bases": [4, 5, 6],
        "step": 1,
    },
    {
        "id": "request-timeout",
        "name": "单次请求超时",
        "entity": "超时设置",
        "unit": "秒",
        "owner": "网络助手",
        "detail": "超时按单次请求计算，不跨请求累计",
        "zones": ["外部 API 请求", "本地模型请求", "检索服务请求"],
        "check_set": "experiment",
        "num_bases": [45, 60, 90],
        "step": 15,
    },
    {
        "id": "history-retention",
        "name": "会话历史保留",
        "entity": "历史会话",
        "unit": "天",
        "owner": "记忆助手",
        "detail": "历史按天保留，到期归档",
        "zones": ["检索会话历史", "训练实验历史", "汇报草稿历史"],
        "check_set": "archive",
        "num_bases": [30, 45, 60],
        "step": 10,
    },
    {
        "id": "training-seeds",
        "name": "训练随机种子",
        "entity": "随机种子",
        "unit": "个",
        "owner": "训练助手",
        "detail": "每次训练固定一组种子，不中途更换",
        "zones": ["Qwen 头训练", "基线对照训练", "复现实验"],
        "check_set": "experiment",
        "num_bases": [3, 4, 5],
        "step": 1,
    },
    {
        "id": "eval-case-count",
        "name": "评测案例数",
        "entity": "评测案例",
        "unit": "条",
        "owner": "评测助手",
        "detail": "每轮评测固定一组案例，不临时增删",
        "zones": ["原子关系评测", "检索质量评测", "回归评测"],
        "check_set": "experiment",
        "num_bases": [120, 150, 180],
        "step": 30,
    },
    {
        "id": "sample-size",
        "name": "抽样样本量",
        "entity": "抽样样本",
        "unit": "例",
        "owner": "数据助手",
        "detail": "抽样按例计数，不按批次折算",
        "zones": ["单细胞样本抽样", "随访样本抽样", "负例抽样"],
        "check_set": "search",
        "num_bases": [18, 24, 30],
        "step": 6,
    },
    {
        "id": "time-window",
        "name": "数据时间范围",
        "entity": "数据时间范围",
        "unit": "个月",
        "owner": "数据助手",
        "detail": "时间范围按入库月份计算",
        "zones": ["单细胞图谱数据", "公共队列数据", "本项目采集数据"],
        "check_set": "search",
        "num_bases": [18, 24, 30],
        "step": 6,
    },
    {
        "id": "recommended-sources",
        "name": "推荐来源数量",
        "entity": "推荐来源",
        "unit": "个",
        "owner": "推荐助手",
        "detail": "每次推荐给出固定数量的来源",
        "zones": ["数据集推荐", "文献推荐", "工具推荐"],
        "check_set": "search",
        "num_bases": [5, 6, 8],
        "step": 1,
    },
    {
        "id": "min-rating",
        "name": "最低采纳评分",
        "entity": "采纳评分",
        "unit": "分",
        "owner": "质量助手",
        "detail": "评分满分 10 分，只取整数",
        "zones": ["数据集评审", "模型输出评审", "汇报质量评审"],
        "check_set": "review",
        "num_bases": [6, 7, 8],
        "step": 1,
        "max_value": 10,
    },
    {
        "id": "storage-quota",
        "name": "项目存储上限",
        "entity": "项目存储占用",
        "unit": "GB",
        "owner": "存储助手",
        "detail": "按项目统计占用，不跨项目合并",
        "zones": ["单细胞项目", "乳腺癌项目", "公共数据集镜像"],
        "check_set": "storage",
        "num_bases": [100, 150, 200],
        "step": 25,
    },
    {
        "id": "glossary-entries",
        "name": "术语表条目数",
        "entity": "术语条目",
        "unit": "条",
        "owner": "写作助手",
        "detail": "每份汇报共用一份术语表",
        "zones": ["组会周报", "给导师的汇报", "项目文档"],
        "check_set": "report",
        "num_bases": [18, 24, 30],
        "step": 6,
    },
    {
        "id": "figure-count",
        "name": "汇报图表数量",
        "entity": "图表",
        "unit": "张",
        "owner": "汇报助手",
        "detail": "图表编号与正文引用一一对应",
        "zones": ["组会周报", "答辩材料", "项目申报材料"],
        "check_set": "report",
        "num_bases": [6, 8, 10],
        "step": 2,
    },
    {
        "id": "reference-year-window",
        "name": "引用文献年限",
        "entity": "引用文献年限",
        "unit": "年",
        "owner": "文献助手",
        "detail": "按发表年份计算年限",
        "zones": ["方法学引用", "数据集引用", "综述引用"],
        "check_set": "search",
        "num_bases": [5, 8, 10],
        "step": 1,
    },
    {
        "id": "batch-size",
        "name": "单批处理量",
        "entity": "每批处理条目",
        "unit": "条",
        "owner": "批处理助手",
        "detail": "每批处理量固定，不随负载浮动",
        "zones": ["向量化任务", "评测任务", "数据清洗任务"],
        "check_set": "experiment",
        "num_bases": [48, 64, 96],
        "step": 16,
    },
    {
        "id": "cache-retention",
        "name": "缓存保留时长",
        "entity": "缓存条目",
        "unit": "小时",
        "owner": "缓存助手",
        "detail": "缓存按小时过期，不跨天累计",
        "zones": ["检索缓存", "模型响应缓存", "向量缓存"],
        "check_set": "archive",
        "num_bases": [18, 24, 48],
        "step": 6,
    },
    {
        "id": "access-review-days",
        "name": "权限复核周期",
        "entity": "权限复核周期",
        "unit": "天",
        "owner": "权限助手",
        "detail": "每轮复核覆盖全部已授权项目",
        "zones": ["只读权限", "写权限", "共享链接权限"],
        "check_set": "review",
        "num_bases": [45, 60, 90],
        "step": 15,
    },
    {
        "id": "required-sections",
        "name": "汇报必备章节",
        "entity": "必备章节",
        "unit": "个",
        "owner": "汇报助手",
        "detail": "章节清单固定，不随汇报长度变化",
        "zones": ["组会周报", "结题汇报", "项目申报材料"],
        "check_set": "report",
        "num_bases": [4, 5, 6],
        "step": 1,
    },
    {
        "id": "manual-review-count",
        "name": "人工抽检条数",
        "entity": "抽检条目",
        "unit": "条",
        "owner": "质量助手",
        "detail": "每轮抽检固定条数，不重复抽同一条",
        "zones": ["检索结果抽检", "模型输出抽检", "汇报引用抽检"],
        "check_set": "review",
        "num_bases": [12, 15, 18],
        "step": 3,
    },
    {
        "id": "dedup-similarity",
        "name": "去重相似度阈值",
        "entity": "去重相似度",
        "unit": "分",
        "owner": "去重助手",
        "detail": "相似度按百分制整数计",
        "zones": ["检索结果去重", "引文去重", "样本去重"],
        "check_set": "search",
        "num_bases": [80, 85, 90],
        "step": 5,
        "max_value": 100,
    },
    {
        "id": "batch-approval",
        "name": "批量操作审批条数",
        "entity": "批量操作条目",
        "unit": "条",
        "owner": "审批助手",
        "detail": "超过阈值的批量操作需要一次审批",
        "zones": ["数据删除", "权限变更", "批量导出"],
        "check_set": "review",
        "num_bases": [9, 12, 15],
        "step": 3,
    },
    {
        "id": "weekly-report-cadence",
        "name": "周报提交间隔",
        "entity": "周报提交间隔",
        "unit": "天",
        "owner": "汇报助手",
        "detail": "周报按自然周提交",
        "zones": ["组会周报", "项目周报", "给导师的周报"],
        "check_set": "report",
        "num_bases": [21, 28, 35],
        "step": 7,
    },
    {
        "id": "data-version-count",
        "name": "数据版本保留",
        "entity": "数据版本",
        "unit": "个",
        "owner": "数据助手",
        "detail": "同一数据集只保留有限版本",
        "zones": ["单细胞数据集", "处理后数据集", "公共数据集镜像"],
        "check_set": "archive",
        "num_bases": [3, 4, 5],
        "step": 1,
    },
    {
        "id": "experiment-runs",
        "name": "实验重复次数",
        "entity": "实验重复",
        "unit": "次",
        "owner": "实验助手",
        "detail": "同一配置的重复次数固定",
        "zones": ["训练实验", "评测实验", "数据构造实验"],
        "check_set": "experiment",
        "num_bases": [3, 4, 5],
        "step": 1,
    },
    {
        "id": "export-batch-size",
        "name": "导出批大小",
        "entity": "每批导出条目",
        "unit": "条",
        "owner": "导出助手",
        "detail": "导出按批进行，批大小固定",
        "zones": ["数据集导出", "引文导出", "评测结果导出"],
        "check_set": "storage",
        "num_bases": [48, 64, 96],
        "step": 16,
    },
    {
        "id": "prompt-budget",
        "name": "单次提示预算",
        "entity": "提示预算",
        "unit": "千字",
        "owner": "模型助手",
        "detail": "提示预算按千字计，不跨轮累计",
        "zones": ["检索增强请求", "训练数据构造请求", "评测请求"],
        "check_set": "experiment",
        "num_bases": [6, 8, 10],
        "step": 1,
    },
]
SETTINGS_BY_ID = {item["id"]: item for item in SETTINGS}


                                                          


def _v(label: str, instruction: str, guard: str | None = None):
    return {"label": label, "instruction": instruction, "guard": guard}


VARIANTS = {
                          
    "paraphrase_same_truth": _v(
        "equivalent",
        "用另一种说法复述规格给出的同一长期要求：对象、作用域、数值与方向都不变，只改语序或同义词；不得增删条件。",
    ),
    "threshold_phrase_restatement": _v(
        "equivalent",
        "用与“不少于 {n} {unit}”等价的另一种说法复述（例如“至少 {n} {unit}”“{n} {unit}起”），数值与方向都不变。",
    ),
    "double_negation_threshold_restatement": _v(
        "equivalent",
        "用双重否定复述“不少于 {n} {unit}”（例如“不允许少于 {n} {unit}”），真值条件不变。",
    ),
    "double_negation_plain_restatement": _v(
        "equivalent",
        "用双重否定复述规格中的同一要求（例如“并非不需要”），适用对象与作用域不变。",
    ),
    "scope_synonym_restatement": _v(
        "equivalent",
        "把作用域改写成同义说法（换项目名或场景名的说法），覆盖范围完全不变。",
    ),
    "quantifier_restatement": _v(
        "equivalent",
        "把“全部事项”改写成等价的量化说法（如“没有例外”“无一例外”），覆盖范围不变。",
    ),
    "conjunct_order_restatement": _v(
        "equivalent",
        "把两个并列要求调换顺序复述（“{duty_a}且{duty_b}”写成“{duty_b}且{duty_a}”），内容不变。",
    ),
    "shared_conjunct_synonym_restatement": _v(
        "equivalent",
        "把两个并列要求各换一种同义说法，两个条件与外延都不变。",
    ),
    "duty_order_restatement": _v(
        "equivalent",
        "把两项长期义务的列举顺序换一下（清单不分先后），义务集合不变。",
    ),
    "duty_synonym_restatement": _v(
        "equivalent",
        "把两项长期义务各换一种同义说法，义务内容不变。",
    ),
    "exception_scope_restatement": _v(
        "equivalent",
        "改写例外范围的说法（“除名单外”写成“名单不在其列，其余照旧”），例外集合不变。",
    ),
    "standing_rule_restatement": _v(
        "equivalent",
        "只复述长期要求本身，换一种说法；不涉及任何临时通知或一次性追加。",
    ),
                                                               
    "old_subset_by_threshold": _v(
        "old_entails_new_strict",
        "旧句要求{entity}长期不少于 {n} {unit}，新句只要求不少于 {w} {unit}（{w} < {n}）；两者作用域相同。",
    ),
    "old_subset_by_exact_count": _v(
        "old_entails_new_strict",
        "旧句要求{entity}长期恰好 {n} {unit}，新句要求不少于 {w} {unit}（{w} < {n}）。",
    ),
    "old_subset_by_upper_bound": _v(
        "old_entails_new_strict",
        "旧句要求{entity}长期不多于 {n} {unit}，新句要求不多于 {u} {unit}（{u} > {n}）。",
    ),
    "old_subset_by_priority_scope": _v(
        "old_entails_new_strict",
        "旧句要求这条长期义务适用于{scope}里的全部事项，新句只要求适用于其中标记为高优先级的事项；旧句适用范围更大。",
    ),
    "old_subset_by_evidence_scope": _v(
        "old_entails_new_strict",
        "旧句要求这条长期义务适用于{scope}里的全部事项，新句只要求适用于其中需要引用的事项；旧句适用范围更大。",
    ),
    "old_subset_by_confirmed_scope": _v(
        "old_entails_new_strict",
        "旧句要求这条长期义务适用于{scope}里的全部事项，新句只要求适用于其中由我本人确认过的事项；旧句适用范围更大。",
    ),
    "old_subset_by_extra_conjunct": _v(
        "old_entails_new_strict",
        "旧句要求“{duty_a}且{duty_b}”，新句只要求“{duty_a}”；{duty_b}是额外的并列要求且与前者独立。",
    ),
    "old_subset_by_extra_exclusion_conjunct": _v(
        "old_entails_new_strict",
        "旧句要求“{duty_a}且不出现返工”，新句只要求“{duty_a}”。",
    ),
    "old_subset_by_two_extra_conjuncts": _v(
        "old_entails_new_strict",
        "旧句要求“{duty_a}且{duty_b}且{duty_c}”，新句只要求“{duty_a}”；三个条件彼此独立。",
    ),
    "old_subset_by_extra_duty": _v(
        "old_entails_new_strict",
        "旧句要求两项义务：{duty_a}与{duty_b}；新句只要求{duty_a}。",
    ),
    "old_subset_by_duty_plus_deadline": _v(
        "old_entails_new_strict",
        "旧句要求{duty_a}且必须在当天完成；新句只要求{duty_a}。",
    ),
    "old_subset_by_three_duties": _v(
        "old_entails_new_strict",
        "旧句要求三项义务：{duty_a}、{duty_b}与{duty_c}；新句只要求{duty_a}。",
    ),
    "old_subset_by_no_exception": _v(
        "old_entails_new_strict",
        "旧句规定范围内没有任何例外，新句允许名单内例外；两者作用域相同。",
    ),
    "old_subset_by_narrow_exception": _v(
        "old_entails_new_strict",
        "旧句只豁免名单乙，新句豁免名单甲；名单乙是名单甲的真子集，其余条件相同。",
    ),
    "old_subset_by_no_exception_in_scope": _v(
        "old_entails_new_strict",
        "旧句规定{scope}里没有任何例外，新句允许名单内例外适用于{scope}里的全部事项；范围相同。",
    ),
    "old_subset_by_extra_one_off": _v(
        "old_entails_new_strict",
        "旧句在长期要求之外还要满足一条临时通知的追加要求，新句只要求长期要求；“临时”指只在注明时间窗内有效；追加要求须换一个维度或在同一维度上更严，不得在同一维度上放宽或反转取值。",
    ),
    "old_subset_by_two_one_offs": _v(
        "old_entails_new_strict",
        "旧句在长期要求之外还要满足两条临时通知的追加要求，新句只要求长期要求；每条追加要求都须换一个维度或在同一维度上更严，不得在同一维度上放宽或反转取值。",
    ),
    "old_subset_by_one_off_scope": _v(
        "old_entails_new_strict",
        "旧句在长期要求之外，还对{scope}里的高优先级事项追加一条临时要求；新句只要求长期要求；追加要求须换一个维度或在同一维度上更严，不得在同一维度上放宽或反转取值。",
    ),
                                                               
    "new_subset_by_threshold": _v(
        "new_entails_old_strict",
        "新句要求{entity}长期不少于 {n} {unit}，旧句只要求不少于 {w} {unit}（{w} < {n}）；两者作用域相同。",
    ),
    "new_subset_by_exact_count": _v(
        "new_entails_old_strict",
        "新句要求{entity}长期恰好 {n} {unit}，旧句要求不少于 {w} {unit}（{w} < {n}）。",
    ),
    "new_subset_by_upper_bound": _v(
        "new_entails_old_strict",
        "新句要求{entity}长期不多于 {n} {unit}，旧句要求不多于 {u} {unit}（{u} > {n}）。",
    ),
    "new_subset_by_priority_scope": _v(
        "new_entails_old_strict",
        "新句要求这条长期义务适用于{scope}里的全部事项，旧句只要求适用于其中标记为高优先级的事项；新句适用范围更大。",
    ),
    "new_subset_by_evidence_scope": _v(
        "new_entails_old_strict",
        "新句要求这条长期义务适用于{scope}里的全部事项，旧句只要求适用于其中需要引用的事项；新句适用范围更大。",
    ),
    "new_subset_by_confirmed_scope": _v(
        "new_entails_old_strict",
        "新句要求这条长期义务适用于{scope}里的全部事项，旧句只要求适用于其中由我本人确认过的事项；新句适用范围更大。",
    ),
    "new_subset_by_extra_conjunct": _v(
        "new_entails_old_strict",
        "新句要求“{duty_a}且{duty_b}”，旧句只要求“{duty_a}”；{duty_b}是额外的并列要求且与前者独立。",
    ),
    "new_subset_by_extra_exclusion_conjunct": _v(
        "new_entails_old_strict",
        "新句要求“{duty_a}且不出现返工”，旧句只要求“{duty_a}”。",
    ),
    "new_subset_by_two_extra_conjuncts": _v(
        "new_entails_old_strict",
        "新句要求“{duty_a}且{duty_b}且{duty_c}”，旧句只要求“{duty_a}”；三个条件彼此独立。",
    ),
    "new_subset_by_extra_duty": _v(
        "new_entails_old_strict",
        "新句要求两项义务：{duty_a}与{duty_b}；旧句只要求{duty_a}。",
    ),
    "new_subset_by_duty_plus_deadline": _v(
        "new_entails_old_strict",
        "新句要求{duty_a}且必须在当天完成；旧句只要求{duty_a}。",
    ),
    "new_subset_by_three_duties": _v(
        "new_entails_old_strict",
        "新句要求三项义务：{duty_a}、{duty_b}与{duty_c}；旧句只要求{duty_a}。",
    ),
    "new_subset_by_scope_restricted": _v(
        "new_entails_old_strict",
        "旧句把名单甲内的事项整体豁免，名单甲之外的事项一律执行长期要求；新句把豁免范围收窄到名单甲内的高优先级事项，名单甲内的非高优先级事项与名单甲之外的事项一律执行长期要求。",
    ),
    "new_subset_by_narrow_exception": _v(
        "new_entails_old_strict",
        "新句只豁免名单乙，旧句豁免名单甲；名单乙是名单甲的真子集，其余条件相同。",
    ),
    "new_subset_by_no_exception": _v(
        "new_entails_old_strict",
        "新句规定范围内没有任何例外，旧句允许名单内例外；其余条件相同。",
    ),
    "new_subset_by_temporary_exception": _v(
        "new_entails_old_strict",
        "新句在长期要求之外增加一条临时通知的追加要求，并明确长期要求仍然保留；旧句只要求长期要求；追加要求须换一个维度或在同一维度上更严，不得在同一维度上放宽或反转取值。",
        guard="temporary_exception_retains_longterm",
    ),
    "new_subset_by_two_temporary_exceptions": _v(
        "new_entails_old_strict",
        "新句在长期要求之外增加两条临时通知的追加要求，并明确长期要求仍然保留；旧句只要求长期要求；每条追加要求都须换一个维度或在同一维度上更严，不得在同一维度上放宽或反转取值。",
        guard="temporary_exception_retains_longterm",
    ),
    "new_subset_by_one_off_scope": _v(
        "new_entails_old_strict",
        "新句在长期要求之外，还对{scope}里的高优先级事项追加一条临时要求，并明确长期要求仍然保留；旧句只要求长期要求；追加要求须换一个维度或在同一维度上更严，不得在同一维度上放宽或反转取值。",
        guard="temporary_exception_retains_longterm",
    ),
                        
    "mutually_exclusive_threshold": _v(
        "disjoint",
        "旧句要求{entity}长期不少于 {n} {unit}，新句要求不多于 {q} {unit}（{q} < {n}）。",
    ),
    "mutually_exclusive_exact_values": _v(
        "disjoint",
        "旧句要求{entity}长期恰好 {n} {unit}，新句要求恰好 {m} {unit}（{n} < {m}）。",
    ),
    "contradictory_negation_with_threshold": _v(
        "disjoint",
        "旧句要求{entity}长期不少于 {n} {unit}，新句要求少于 {n} {unit}。",
    ),
    "contradictory_negation": _v(
        "disjoint",
        "旧句说{scope}里的全部事项都遵守同一条长期要求，新句说其中没有任何事项遵守该要求；范围内有事项。",
    ),
    "contradictory_negation_priority": _v(
        "disjoint",
        "旧句说{scope}里的高优先级事项全部遵守同一条要求，新句说其中没有任何高优先级事项遵守该要求；这类事项非空。",
    ),
    "exclusive_default_choice": _v(
        "disjoint",
        "旧句说{scope}里这条要求的默认口径是从严，新句说是从宽；同一项要求只能有一个默认口径。",
    ),
    "contradictory_conjunct": _v(
        "disjoint",
        "旧句要求“{duty_a}且{duty_b}”，新句要求“{duty_a}且不得{duty_b}”。",
    ),
    "contradictory_conjunct_on_other_side": _v(
        "disjoint",
        "旧句要求“{duty_a}且{duty_b}”，新句要求“不得{duty_a}且{duty_b}”。",
    ),
    "exclusive_conjunct_choice": _v(
        "disjoint",
        "旧句要求“{duty_a}且不得{duty_b}”，新句要求“{duty_b}且不得{duty_a}”；两项只能保留一项。",
    ),
    "contradictory_duty": _v(
        "disjoint",
        "旧句要求{duty_a}，新句明确不得{duty_a}；同一项要求只有执行与不执行两种状态。",
    ),
    "contradictory_duty_window": _v(
        "disjoint",
        "旧句要求第一周{duty_a}，新句要求第一周不得{duty_a}。",
    ),
    "exclusive_duty_choice": _v(
        "disjoint",
        "旧句只要求{duty_a}且不得{duty_b}，新句只要求{duty_b}且不得{duty_a}；两项只能保留一项。",
    ),
    "exclusive_exemption_list": _v(
        "disjoint",
        "旧句说豁免名单就是名单戊，新句说豁免名单就是名单己；名单戊与名单己是不同名单，且豁免名单只有一个。",
    ),
    "one_off_suspends_rule": _v(
        "disjoint",
        "旧句说第一周长期要求照常适用，新句说第一周长期要求暂停执行。",
    ),
    "opposite_one_off_directive": _v(
        "disjoint",
        "旧句说第一周照常要求{duty_a}，新句说第一周不得{duty_a}。",
    ),
    "one_off_replaces_rule": _v(
        "disjoint",
        "旧句说第一周仍按长期要求办，新句说第一周改由另一份要求取代长期要求；同一时段只有一份要求生效。",
    ),
                                    
    "crossing_thresholds": _v(
        "overlap_incomparable",
        "旧句要求{entity}长期不少于 {n} {unit}，新句要求不多于 {m} {unit}（{n} < {m}）。",
    ),
    "partial_range_overlap": _v(
        "overlap_incomparable",
        "旧句要求{entity}长期多于 {n} {unit}，新句要求少于 {p} {unit}（{n} < {p}，中间还留有整数）。",
    ),
    "bounded_range_overlap": _v(
        "overlap_incomparable",
        "旧句要求{entity}长期在 {n} 到 {m} {unit}之间，新句要求在 {m} 到 {p} {unit}之间（{n} < {m} < {p}）。",
    ),
    "partial_scope_overlap": _v(
        "overlap_incomparable",
        "旧句说义务只适用于{scope}里标记为高优先级的事项，新句说义务只适用于其中需要引用的事项；两个子集部分重叠且都非空。",
    ),
    "crossing_shift_windows": _v(
        "overlap_incomparable",
        "旧句的要求在第一周生效，新句的要求在第三周生效；两段时间不重叠，两项要求可同时成立。",
    ),
    "priority_partition_requirements": _v(
        "overlap_incomparable",
        "旧句说义务只适用于{scope}里的高优先级事项，新句说义务只适用于其中普通优先级的事项；两类事项都有且非空，两项要求可同时成立。",
    ),
    "independent_extra_conjuncts": _v(
        "overlap_incomparable",
        "旧句要求“{duty_a}且{duty_b}”，新句要求“{duty_a}且{duty_c}”；{duty_b}与{duty_c}彼此独立。",
    ),
    "shared_core_different_side_conditions": _v(
        "overlap_incomparable",
        "旧句要求“{duty_a}且{duty_b}”，新句要求“{duty_c}且{duty_b}”；{duty_a}与{duty_c}彼此独立。",
    ),
    "partially_overlapping_conjunct_sets": _v(
        "overlap_incomparable",
        "旧句要求“{duty_a}且{duty_b}”，新句要求“{duty_b}且{duty_c}”；三项彼此独立。",
    ),
    "independent_obligations": _v(
        "overlap_incomparable",
        "旧句要求{duty_a}，新句要求{duty_b}；两项义务彼此独立。",
    ),
    "partially_overlapping_duty_sets": _v(
        "overlap_incomparable",
        "旧句要求{duty_a}与{duty_b}，新句要求{duty_b}与{duty_c}；三项彼此独立。",
    ),
    "independent_duty_with_shared_prerequisite": _v(
        "overlap_incomparable",
        "旧句要求先完成归档再{duty_a}，新句要求先完成归档再{duty_b}；{duty_a}与{duty_b}彼此独立。",
    ),
    "distinct_exception_sets": _v(
        "overlap_incomparable",
        "旧句说除名单丙外全部适用，新句说除名单丁外全部适用；名单丙与名单丁互不包含且都非空。",
    ),
    "partial_negative_scope": _v(
        "overlap_incomparable",
        "旧句说{scope}里的高优先级事项没有例外，新句说其中需要引用的事项没有例外；两个子集部分重叠且都非空。",
    ),
    "crossing_exception_scopes": _v(
        "overlap_incomparable",
        "旧句的例外只适用于{scope}里的高优先级事项，新句只适用于其中需要引用的事项；两个子集部分重叠且都非空。",
    ),
    "different_one_off_windows": _v(
        "overlap_incomparable",
        "旧句的追加要求在第一周生效，新句的追加要求在第三周生效；两项可同时成立。",
    ),
    "one_off_scope_overlap": _v(
        "overlap_incomparable",
        "旧句的临时要求覆盖第一周{scope}里的高优先级事项，新句覆盖第一周其中需要引用的事项。",
    ),
    "independent_one_off_duties": _v(
        "overlap_incomparable",
        "旧句要求第一周{duty_a}，新句要求第一周{duty_b}；两项彼此独立。",
    ),
}


MECHANISM_PLANS = {
    "numeric_threshold": (
        "paraphrase_same_truth", "threshold_phrase_restatement", "double_negation_threshold_restatement",
        "old_subset_by_threshold", "old_subset_by_exact_count", "old_subset_by_upper_bound",
        "new_subset_by_threshold", "new_subset_by_exact_count", "new_subset_by_upper_bound",
        "mutually_exclusive_threshold", "mutually_exclusive_exact_values", "contradictory_negation_with_threshold",
        "crossing_thresholds", "partial_range_overlap", "bounded_range_overlap",
    ),
    "scope_inclusion": (
        "paraphrase_same_truth", "scope_synonym_restatement", "quantifier_restatement",
        "old_subset_by_priority_scope", "old_subset_by_evidence_scope", "old_subset_by_confirmed_scope",
        "new_subset_by_priority_scope", "new_subset_by_evidence_scope", "new_subset_by_confirmed_scope",
        "contradictory_negation", "exclusive_default_choice", "contradictory_negation_priority",
        "partial_scope_overlap", "crossing_shift_windows", "priority_partition_requirements",
    ),
    "extra_conjunct": (
        "paraphrase_same_truth", "conjunct_order_restatement", "shared_conjunct_synonym_restatement",
        "old_subset_by_extra_conjunct", "old_subset_by_extra_exclusion_conjunct", "old_subset_by_two_extra_conjuncts",
        "new_subset_by_extra_conjunct", "new_subset_by_extra_exclusion_conjunct", "new_subset_by_two_extra_conjuncts",
        "contradictory_conjunct", "contradictory_conjunct_on_other_side", "exclusive_conjunct_choice",
        "independent_extra_conjuncts", "shared_core_different_side_conditions", "partially_overlapping_conjunct_sets",
    ),
    "independent_obligation": (
        "paraphrase_same_truth", "duty_order_restatement", "duty_synonym_restatement",
        "old_subset_by_extra_duty", "old_subset_by_duty_plus_deadline", "old_subset_by_three_duties",
        "new_subset_by_extra_duty", "new_subset_by_duty_plus_deadline", "new_subset_by_three_duties",
        "contradictory_duty", "exclusive_duty_choice", "contradictory_duty_window",
        "independent_obligations", "partially_overlapping_duty_sets", "independent_duty_with_shared_prerequisite",
    ),
    "negation_exception": (
        "paraphrase_same_truth", "exception_scope_restatement", "double_negation_plain_restatement",
        "old_subset_by_no_exception", "old_subset_by_narrow_exception", "old_subset_by_no_exception_in_scope",
        "new_subset_by_scope_restricted", "new_subset_by_narrow_exception", "new_subset_by_no_exception",
        "contradictory_negation", "exclusive_exemption_list", "contradictory_negation_priority",
        "distinct_exception_sets", "partial_negative_scope", "crossing_exception_scopes",
    ),
    "longterm_one_off": (
        "paraphrase_same_truth", "standing_rule_restatement", "double_negation_plain_restatement",
        "old_subset_by_extra_one_off", "old_subset_by_two_one_offs", "old_subset_by_one_off_scope",
        "new_subset_by_temporary_exception", "new_subset_by_two_temporary_exceptions", "new_subset_by_one_off_scope",
        "one_off_suspends_rule", "opposite_one_off_directive", "one_off_replaces_rule",
        "different_one_off_windows", "one_off_scope_overlap", "independent_one_off_duties",
    ),
}


BACKGROUND_TEMPLATES = {
    "numeric_threshold": (
        "{name}（{owner}维护）：{detail}；作用域是{scope}；同一用户只沿用一份长期口径，"
        "数值取整数，两句话按同一份口径判断真假。"
    ),
    "scope_inclusion": (
        "{name}（{owner}维护）：{detail}；作用域是{scope}；同一用户只沿用一份长期范围；"
        "事项的优先级（高优先级与普通优先级）、引用标注（需要引用与否）与本人确认状态可任意组合，"
        "各类子集都非空，两句话按同一份范围判断真假。"
    ),
    "extra_conjunct": (
        "{name}（{owner}维护）：{detail}；作用域是{scope}；同一用户可能同时提出若干"
        "彼此独立的要求，任一项可满足或不满足。"
    ),
    "independent_obligation": (
        "{name}（{owner}维护）：{detail}；作用域是{scope}；同一用户长期执行若干"
        "彼此独立的义务，每项可单独完成或未完成。"
    ),
    "negation_exception": (
        "{name}（{owner}维护）：{detail}；作用域是{scope}；同一用户只沿用一份长期要求；"
        "名单卡：名单甲真包含名单乙，且名单甲内同时含高优先级与非高优先级事项；"
        "名单丙、名单丁、名单戊、名单己、名单庚、名单辛两两互不包含且都非空；"
        "高优先级事项与需要引用的事项部分重叠且都非空；名单内外都有条目。"
    ),
    "longterm_one_off": (
        "{name}（{owner}维护）：{detail}；作用域是{scope}；同一用户只沿用一份长期要求；"
        "临时通知只在注明时间窗内生效，窗内可追加、暂停或替代长期要求，窗外长期要求不变；"
        "事项分高优先级与普通优先级，并区分是否需要引用；"
        "优先级、引用标注与本人确认状态可任意组合，各类子集都非空。"
    ),
}


def render(template: str, context: dict) -> str:
    try:
        text = template.format(**context)
    except KeyError as exc:
        raise DataError("template_placeholder_missing", f"{exc} in {template[:40]}")
    if "{" in text or "}" in text:
        raise DataError("template_placeholder_unresolved", text[:80])
    return text


def split_for_family(index: int) -> str:
    return SPLIT_ORDER[index % len(SPLIT_ORDER)]


def _shuffled_by_digest(values, salt: str):
    """按 (位置, 值) 的摘要排序；同名元素也能被真正打散，而不是整块保持相邻。"""
    items = list(enumerate(values))
    items.sort(key=lambda pair: sha256_text(f"{salt}|{pair[0]}|{pair[1]}"))
    return [value for _, value in items]


def _setting_plan() -> dict:
    """每个 setting 只落在一个 split：10 个 setting 三族，20 个两族。"""
    ordered = sorted(SETTINGS, key=lambda item: sha256_text(f"setting-order|{item['id']}"))
    triples = ordered[: SETTING_FAMILY_SHAPE["triples"]]
    pairs = ordered[SETTING_FAMILY_SHAPE["triples"]:]
    triple_slots = []
    pair_slots = []
    for split in SPLITS:
        slots = SETTING_SPLIT_SLOTS[split]
        triple_slots.extend([split] * slots["triples"])
        pair_slots.extend([split] * slots["pairs"])
    if len(triple_slots) != len(triples) or len(pair_slots) != len(pairs):
        raise DataError("setting_slot_mismatch", f"{len(triple_slots)}/{len(triples)} {len(pair_slots)}/{len(pairs)}")
    plan = {}
    for setting, split in zip(triples, triple_slots):
        plan[setting["id"]] = (split, 3)
    for setting, split in zip(pairs, pair_slots):
        plan[setting["id"]] = (split, 2)
    return plan


def _spec_plan() -> dict:
    setting_split = _setting_plan()
    families_by_split = {
        split: [index for index in range(FAMILY_COUNT) if split_for_family(index) == split]
        for split in SPLITS
    }
    plan = {}
    for split in SPLITS:
        settings_in_split = [item for item in SETTINGS if setting_split[item["id"]][0] == split]
        pool = []
        for setting in settings_in_split:
            pool.extend([setting["id"]] * setting_split[setting["id"]][1])
        if len(pool) != len(families_by_split[split]):
            raise DataError("setting_pool_size", f"{split}:{len(pool)}!={len(families_by_split[split])}")
        mechanisms = []
        for name in _REQUIRED_MECHANISMS:
            mechanisms.extend([name] * SPLIT_MECHANISM_QUOTAS[split][name])
        if len(mechanisms) != len(pool):
            raise DataError("mechanism_pool_size", f"{split}:{len(mechanisms)}!={len(pool)}")
        pool = _shuffled_by_digest(pool, f"setting-pool|{split}")
        mechanisms = _shuffled_by_digest(mechanisms, f"mechanism-pool|{split}")
        occurrence = Counter()
        for index, setting_id, mechanism in zip(families_by_split[split], pool, mechanisms):
            setting = SETTINGS_BY_ID[setting_id]
            slot = occurrence[setting_id]
            occurrence[setting_id] += 1
            plan[index] = {
                "setting": setting,
                "mechanism": mechanism,
                "split": split,
                "scope": setting["zones"][slot],
                "base": setting["num_bases"][slot],
            }
    if set(plan) != set(range(FAMILY_COUNT)):
        raise DataError("spec_plan_incomplete", str(sorted(set(range(FAMILY_COUNT)) - set(plan))[:5]))
    return plan


def _numbers(setting: dict, base: int) -> dict:
    step = setting["step"]
    numbers = {
        "n": base,
        "w": base - step,
        "m": base + step,
        "u": base + step,
        "p": base + 2 * step,
        "q": base - 2 * step,
    }
    if base not in setting["num_bases"]:
        raise DataError("number_base_not_declared", f"{setting['id']}:{base}")
    if numbers["w"] < 1 or numbers["q"] < 0:
        raise DataError("number_scale_invalid", f"{setting['id']}:{numbers}")
    maximum = setting.get("max_value")
    if maximum is not None and numbers["p"] > maximum:
        raise DataError("number_exceeds_max", f"{setting['id']}:{numbers['p']}>{maximum}")
    return numbers


def build_family(*, index: int, setting: dict, mechanism: str, split: str,
                 scope: str | None = None, base: int | None = None) -> dict:
    family_id = f"af-{index:03d}"
    if scope is None:
        scope = setting["zones"][0]
    if scope not in setting["zones"]:
        raise DataError("scope_not_in_setting", f"{setting['id']}:{scope}")
    if base is None:
        base = setting["num_bases"][0]
    numbers = _numbers(setting, base)
    duty_a, duty_b, duty_c = CHECK_SETS[setting["check_set"]]
    context = {
        "entity": setting["entity"],
        "unit": setting["unit"],
        "owner": setting["owner"],
        "scope": scope,
        "name": setting["name"],
        "detail": setting["detail"],
        "duty_a": duty_a,
        "duty_b": duty_b,
        "duty_c": duty_c,
        **numbers,
    }
    background = render(BACKGROUND_TEMPLATES[mechanism], context)
    if len(background) > _MAX_BACKGROUND_CHARS:
        raise DataError("background_too_long", f"{family_id} {len(background)}")
    for word in _BANNED_NARRATIVE:
        if word in background:
            raise DataError("banned_narrative", f"{family_id}:{word}")

    items = []
    for variant_key in MECHANISM_PLANS[mechanism]:
        variant = VARIANTS[variant_key]
        instruction = render(variant["instruction"], context)
        for word in _BANNED_NARRATIVE:
            if word in instruction:
                raise DataError("banned_narrative", f"{family_id}:{variant_key}:{word}")
        items.append(
            {
                "fine_label": variant["label"],
                "variant": variant_key,
                "guard": variant["guard"],
                "instruction": instruction,
            }
        )
    if len(items) != ITEMS_PER_FAMILY:
        raise DataError("family_item_count", f"{family_id} {len(items)}")
    items.sort(key=lambda item: sha256_text(f"{family_id}|{item['variant']}"))
    for position, item in enumerate(items):
        item["pair_id"] = f"{family_id}-p{position:02d}"
        item.pop("guard")

    return {
        "schema": SPEC_SCHEMA,
        "family_id": family_id,
        "split": split,
        "setting": {
            "id": setting["id"],
            "name": setting["name"],
            "entity": setting["entity"],
            "unit": setting["unit"],
            "scope": scope,
        },
        "mechanism": mechanism,
        "background": background,
        "items": [
            {
                "pair_id": item["pair_id"],
                "fine_label": item["fine_label"],
                "variant": item["variant"],
                "instruction": item["instruction"],
            }
            for item in items
        ],
    }


def build_specs() -> list[dict]:
    plan = _spec_plan()
    specs = []
    for index in range(FAMILY_COUNT):
        entry = plan[index]
        specs.append(
            build_family(
                index=index,
                setting=entry["setting"],
                mechanism=entry["mechanism"],
                split=entry["split"],
                scope=entry["scope"],
                base=entry["base"],
            )
        )
    return specs


                                                


def spec_sha256(spec: dict) -> str:
    return sha256_json(spec)


_ARITHMETIC_RE = re.compile(r"(\d+)\s*([<>])\s*(\d+)")


def arithmetic_problems(text: str) -> list[str]:
    """规格里的 A < B / A > B 必须数值成立；模板写反方向时立刻失败。"""
    problems = []
    for left, operator, right in _ARITHMETIC_RE.findall(text or ""):
        first, second = int(left), int(right)
        if operator == "<" and not first < second:
            problems.append(f"{first} < {second}")
        if operator == ">" and not first > second:
            problems.append(f"{first} > {second}")
    return problems


def validate_spec(spec: dict, *, fine=None, mapping=None) -> list[str]:
    fine, _coarse, mapping = relation_mapping() if fine is None else (fine, None, mapping)
    problems = []
    family_id = spec.get("family_id")
    if not isinstance(family_id, str) or not re.fullmatch(r"af-\d{3}", family_id):
        problems.append(f"bad_family_id:{family_id!r}")
        return problems
    if spec.get("split") not in SPLITS:
        problems.append(f"{family_id}:bad_split:{spec.get('split')!r}")
    if spec.get("mechanism") not in MECHANISM_PLANS:
        problems.append(f"{family_id}:bad_mechanism:{spec.get('mechanism')!r}")
    background = spec.get("background")
    mechanism = spec.get("mechanism")
    if not isinstance(background, str) or not background.strip():
        problems.append(f"{family_id}:empty_background")
    else:
        if len(background) > _MAX_BACKGROUND_CHARS:
            problems.append(f"{family_id}:background_too_long:{len(background)}")
        if "同一用户" not in background:
            problems.append(f"{family_id}:background_missing_user_frame")
        setting = spec.get("setting")
        if isinstance(setting, dict) and isinstance(setting.get("scope"), str):
            if setting["scope"] not in background:
                problems.append(f"{family_id}:background_missing_scope")
        for word in _BANNED_NARRATIVE:
            if word in background:
                problems.append(f"{family_id}:banned_narrative:{word}")
        for word in _DIRECTION_WORDS:
            if word in background:
                problems.append(f"{family_id}:background_direction_word:{word}")
        if isinstance(mechanism, str):
            for marker in _BACKGROUND_REQUIREMENTS.get(mechanism, ()):
                if marker not in background:
                    problems.append(f"{family_id}:background_missing_marker:{marker}")
        if mechanism == "longterm_one_off" and "不废止" in background:
            problems.append(f"{family_id}:background_conflicts_suspend")
    setting = spec.get("setting")
    if not isinstance(setting, dict) or setting.get("id") not in SETTINGS_BY_ID:
        problems.append(f"{family_id}:bad_setting")
    elif setting.get("scope") not in SETTINGS_BY_ID[setting["id"]]["zones"]:
        problems.append(f"{family_id}:scope_not_declared:{setting.get('scope')!r}")
    premise_keywords = dict(_PREMISE_KEYWORDS)
    if mechanism != "longterm_one_off":
        premise_keywords.pop("第一周", None)
        premise_keywords.pop("第三周", None)
    items = spec.get("items")
    if not isinstance(items, list) or len(items) != ITEMS_PER_FAMILY:
        problems.append(f"{family_id}:item_count:{len(items) if isinstance(items, list) else 'n/a'}")
        return problems
    seen_ids = set()
    seen_variants = set()
    counts = Counter()
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            problems.append(f"{family_id}:item_not_object:{position}")
            continue
        extra = sorted(set(item) - {"pair_id", "fine_label", "variant", "instruction"})
        if extra:
            problems.append(f"{family_id}:item_extra_keys:{extra}")
        pair_id = item.get("pair_id")
        expected_id = f"{family_id}-p{position:02d}"
        if pair_id != expected_id:
            problems.append(f"{family_id}:pair_id_mismatch:{pair_id!r}!={expected_id}")
        if pair_id in seen_ids:
            problems.append(f"{family_id}:pair_id_duplicate:{pair_id}")
        seen_ids.add(pair_id)
        label = item.get("fine_label")
        variant_key = item.get("variant")
        if label not in mapping:
            problems.append(f"{family_id}:bad_label:{label!r}")
            continue
        if variant_key not in VARIANTS:
            problems.append(f"{family_id}:unknown_variant:{variant_key!r}")
            continue
        seen_variants.add(variant_key)
        variant = VARIANTS[variant_key]
        if variant["label"] != label:
            problems.append(f"{family_id}:variant_label_mismatch:{variant_key}:{label}")
        if variant["guard"] == "temporary_exception_retains_longterm" and label == "disjoint":
            problems.append(f"{family_id}:temporary_exception_marked_disjoint:{pair_id}")
        instruction = item.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            problems.append(f"{family_id}:empty_instruction:{pair_id}")
        elif "{" in instruction or "}" in instruction:
            problems.append(f"{family_id}:unresolved_placeholder:{pair_id}")
        else:
            for bad in arithmetic_problems(instruction):
                problems.append(f"{family_id}:arithmetic_reversed:{pair_id}:{bad}")
            for word in _BANNED_NARRATIVE:
                if word in instruction:
                    problems.append(f"{family_id}:banned_narrative:{pair_id}:{word}")
            for marker in _VARIANT_TEXT_REQUIREMENTS.get(variant_key, ()):
                if marker not in instruction:
                    problems.append(f"{family_id}:instruction_missing_marker:{pair_id}:{marker}")
            if label == "disjoint":
                for word in _DISJOINT_SOFT_WORDS:
                    if word in instruction:
                        problems.append(f"{family_id}:disjoint_soft_wording:{pair_id}:{word}")
            if mechanism == "negation_exception":
                allowed = _VARIANT_LIST_ALLOWED.get(variant_key, frozenset())
                used = {char for char in _LIST_CHARS if char in instruction}
                for char in sorted(used - allowed):
                    problems.append(f"{family_id}:list_algebra:{pair_id}:{char}")
                if variant_key in _VARIANT_LIST_ALLOWED:
                    for char in sorted(allowed - used):
                        problems.append(f"{family_id}:list_missing:{pair_id}:{char}")
            for keyword, marker in premise_keywords.items():
                if keyword in instruction and marker not in (background or ""):
                    problems.append(f"{family_id}:premise_missing:{pair_id}:{keyword}")
        counts[label] += 1
    for label in fine:
        if counts[label] != PER_FINE:
            problems.append(f"{family_id}:fine_count:{label}:{counts[label]}")
    if len(seen_variants) != ITEMS_PER_FAMILY:
        problems.append(f"{family_id}:variant_reuse:{len(seen_variants)}")
    return problems


def validate_spec_set(specs: list[dict], *, full: bool = True) -> list[str]:
    problems = []
    ids = [spec.get("family_id") for spec in specs]
    if len(set(ids)) != len(ids):
        problems.append("duplicate_family_ids")
    for spec in specs:
        problems.extend(validate_spec(spec))
    by_split = Counter(spec.get("split") for spec in specs)
    if full:
        if len(specs) != FAMILY_COUNT:
            problems.append(f"family_total:{len(specs)}")
        for split in SPLITS:
            if by_split.get(split, 0) != FAMILY_EXPECTED[split]:
                problems.append(f"family_split:{split}:{by_split.get(split, 0)}")
        mechanisms = Counter(spec.get("mechanism") for spec in specs)
        for name in _REQUIRED_MECHANISMS:
            if mechanisms.get(name, 0) < 10:
                problems.append(f"mechanism_underused:{name}:{mechanisms.get(name, 0)}")
        for split in SPLITS:
            present = {spec.get("mechanism") for spec in specs if spec.get("split") == split}
            missing = sorted(set(_REQUIRED_MECHANISMS) - present)
            if missing:
                problems.append(f"split_mechanism_gap:{split}:{missing}")
        settings = {spec["setting"]["id"] for spec in specs if isinstance(spec.get("setting"), dict)}
        if len(settings) < 20:
            problems.append(f"setting_variety:{len(settings)}")
        scope_keys = [
            (spec["setting"]["id"], spec["setting"]["scope"])
            for spec in specs
            if isinstance(spec.get("setting"), dict) and "scope" in spec["setting"]
        ]
        if len(set(scope_keys)) != len(scope_keys):
            problems.append(f"duplicate_setting_scope:{len(scope_keys) - len(set(scope_keys))}")
        setting_splits = defaultdict(set)
        for spec in specs:
            if isinstance(spec.get("setting"), dict):
                setting_splits[spec["setting"]["id"]].add(spec.get("split"))
        crossing = sorted(key for key, value in setting_splits.items() if len(value) > 1)
        if crossing:
            problems.append(f"setting_crosses_split:{crossing[:5]}")
    expected_pairs = sum(len(spec.get("items") or []) for spec in specs)
    if expected_pairs != ITEMS_PER_FAMILY * len(specs):
        problems.append(f"item_total:{expected_pairs}")
    return problems


                                                 


def build_prompt(spec: dict) -> str:
    fine, _coarse, _mapping = relation_mapping()
    definitions = {
        "equivalent": "P 与 H 的真值集合完全相同，只是说法不同。",
        "old_entails_new_strict": "P 的真值集合是 H 的真子集；P 成立则 H 必成立，反之不然。",
        "new_entails_old_strict": "H 的真值集合是 P 的真子集；H 成立则 P 必成立，反之不然。",
        "disjoint": "两个真值集合不相交；同一行为方案下不可能同时为真。",
        "overlap_incomparable": "两个真值集合相交但互不包含；存在同时为真的方案，也存在只满足其一的两种方案。",
    }
    if set(definitions) != set(fine):
        raise DataError("prompt_definition_drift", str(sorted(definitions)))
    gloss = "\n".join(f"- {name}：{definitions[name]}" for name in fine)
    setting = spec["setting"]
    payload = {
        "family_id": spec["family_id"],
        "setting": spec["setting"],
        "mechanism": spec["mechanism"],
        "background": spec["background"],
        "items": [
            {
                "pair_id": item["pair_id"],
                "fine_label": item["fine_label"],
                "variant": item["variant"],
                "instruction": item["instruction"],
            }
            for item in spec["items"]
        ],
    }
    return (
        "你在为“长期记忆条目关系”数据集编写中文句对。每条句子都是同一用户在同一工作流里"
        "可以长期保存的偏好或要求，不是一次性的计数事实或事件记录。\n\n"
        "【关系口径】old 固定为前提 P，new 固定为假设 H。判断时穷举同一用户在同一工作流"
        "作用域内所有可能的行为方案（采用哪一档上限、是否要求引用、优先级如何、临时通知是否"
        "生效等），比较两句的真值集合：\n"
        f"{gloss}\n"
        "只在规格声明的作用域与背景内判断，不用现实常识补足未声明的信息；临时通知若只是在"
        "长期要求之外追加要求、长期要求仍然保留，则不是 disjoint。\n\n"
        "【硬性要求】\n"
        "1. 只输出一个 JSON 数组，恰好 15 项，顺序与规格 items 完全一致；不要代码围栏、"
        "不要任何解释文字。\n"
        "2. 每项键恰好为 pair_id、fine_label、old、new、background；old 与 new 的键只允许 text，"
        "不得出现 trigger、permanence 或其他字段；条件与时间窗直接写进 text。\n"
        "3. pair_id 与 fine_label 必须逐字照抄规格；不得改标签、合并、拆分或增删条目。\n"
        "4. 每句是一个原子、完整、自足的偏好或义务：不省略主语，不用代词指向规格之外的内容，"
        "不夹带解释、推理或多余条件；一句话只表达一个可判真假的条件。\n"
        f"5. 句子要写成可长期保存的口径（最多、至少、固定为、必须、不得），围绕“{setting['entity']}”"
        f"展开，作用域是“{setting['scope']}”；不要写成“今天有 6 条”这类一次性事实。\n"
        "6. 两句针对同一用户、同一工作流作用域、同一批事项；只在规格指定的维度上不同。\n"
        "7. 每条 instruction 的必要前提必须能在 old、new 或 background 中直接判定：涉及名单、"
        "优先级、引用要求或时间窗时，只能使用 background 已声明的设定，不得新增前提。\n"
        "8. fine_label 为 disjoint 的条目不得用“不要求”“无需”“可以”表达互斥；必须写成同一"
        "对象的明确禁止或相反谓词（如“必须 X”对“不得 X”）。\n"
        "9. 句子长度 10–26 个汉字；使用规格中的对象与计量单位，不得引入新对象。\n"
        "10. background 逐字复制规格的 background 字段；不得压缩、省略或改写。\n"
        "11. 输出紧凑：不要缩进，不要在标点旁加空格。\n\n"
        "【规格】\n"
        f"{canonical_json(payload)}\n"
    )


                                                     


def _normalize_string_entries(items: list):
    """Qwen 偶尔把 old/new 写成字符串：仅非空字符串归一为 {"text": 原字符串}。

    空串、纯空白、非字符串和非对象一律原样保留，交给 check_author_items 按原规则拒绝。
    返回新列表与归一化次数；未归一化的条目保持原对象不变。
    """
    normalized = 0
    result = []
    for item in items:
        if not isinstance(item, dict):
            result.append(item)
            continue
        copy = None
        for side in ("old", "new"):
            value = item.get(side)
            if isinstance(value, str) and value.strip():
                if copy is None:
                    copy = dict(item)
                copy[side] = {"text": value}
                normalized += 1
        result.append(copy if copy is not None else item)
    return result, normalized


def extract_json_array(text: str):
    if not isinstance(text, str) or not text.strip():
        raise DataError("empty_response")
    value = text.strip()
    mode = "strict"
    if value.startswith("```"):
        value = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", value)
        value = re.sub(r"\s*```$", "", value).strip()
        mode = "fenced"
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("["), value.rfind("]")
        if start < 0 or end <= start:
            raise DataError("no_json_array")
        try:
            parsed = json.loads(value[start:end + 1])
        except json.JSONDecodeError as exc:
            raise DataError("bad_json_array", str(exc))
        mode = "bracketed"
    if not isinstance(parsed, list):
        raise DataError("json_not_array")
    parsed, normalized = _normalize_string_entries(parsed)
    if normalized:
        mode = f"{mode}+string_entries_normalized"
    return parsed, mode


_ITEM_KEYS = {"pair_id", "fine_label", "old", "new", "background"}
_ENTRY_KEYS = {"text"}


def _author_text_gate(family_id: str, spec_item: dict, item: dict) -> list[str]:
    """作者文本级变体门：只检查 old/new 文本，不涉及 instruction 或 legacy 行。"""
    problems = []
    pair_id = spec_item.get("pair_id")
    texts = {}
    for side in ("old", "new"):
        entry = item.get(side)
        texts[side] = entry.get("text") if isinstance(entry, dict) and isinstance(entry.get("text"), str) else ""
    if spec_item.get("fine_label") == "disjoint":
        for side, text in texts.items():
            for word in _AUTHOR_DISJOINT_BANNED:
                if word in text:
                    problems.append(f"{family_id}:author_text_disjoint_soft:{pair_id}:{side}:{word}")
    if spec_item.get("variant") == "crossing_exception_scopes":
        for side, text in texts.items():
            if re.search(r"只有.*免于", text):
                problems.append(f"{family_id}:author_text_only_exempt:{pair_id}:{side}")
            if not ("只适用于" in text or re.search(r"只对.*生效", text)):
                problems.append(f"{family_id}:author_text_scope_phrase:{pair_id}:{side}")
    if spec_item.get("variant") == "exclusive_exemption_list":
        for side, text in texts.items():
            if not re.search(r"唯一|只有一个|其他名单不得", text):
                problems.append(f"{family_id}:author_text_unique_list:{pair_id}:{side}")
    return problems


def check_author_items(items: list, spec: dict) -> list[str]:
    problems = []
    family_id = spec["family_id"]
    if not isinstance(items, list):
        return [f"{family_id}:not_list"]
    if len(items) != ITEMS_PER_FAMILY:
        problems.append(f"{family_id}:item_count:{len(items)}")
    by_id = {}
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            problems.append(f"{family_id}:item_not_object:{position}")
            continue
        extra = sorted(set(item) - _ITEM_KEYS)
        missing = sorted(_ITEM_KEYS - set(item))
        if extra:
            problems.append(f"{family_id}:extra_keys:{extra}")
        if missing:
            problems.append(f"{family_id}:missing_keys:{position}:{missing}")
            continue
        pair_id = item["pair_id"]
        if pair_id in by_id:
            problems.append(f"{family_id}:duplicate_pair_id:{pair_id}")
        by_id[pair_id] = item
        if not isinstance(item["background"], str) or not item["background"].strip():
            problems.append(f"{family_id}:empty_background:{pair_id}")
        for side in ("old", "new"):
            entry = item[side]
            if not isinstance(entry, dict):
                problems.append(f"{family_id}:entry_not_object:{pair_id}:{side}")
                continue
            entry_extra = sorted(set(entry) - _ENTRY_KEYS)
            if entry_extra:
                problems.append(f"{family_id}:entry_extra_keys:{pair_id}:{side}:{entry_extra}")
            text = entry.get("text")
            if not isinstance(text, str) or not text.strip():
                problems.append(f"{family_id}:empty_text:{pair_id}:{side}")
            elif len(text.strip()) > 400:
                problems.append(f"{family_id}:text_too_long:{pair_id}:{side}")
    for spec_item in spec["items"]:
        pair_id = spec_item["pair_id"]
        item = by_id.get(pair_id)
        if item is None:
            problems.append(f"{family_id}:missing_pair:{pair_id}")
            continue
        if item.get("fine_label") != spec_item["fine_label"]:
            problems.append(
                f"{family_id}:label_mismatch:{pair_id}:"
                f"{item.get('fine_label')!r}!={spec_item['fine_label']!r}"
            )
        problems.extend(_author_text_gate(family_id, spec_item, item))
    known = {item["pair_id"] for item in spec["items"]}
    for pair_id in by_id:
        if pair_id not in known:
            problems.append(f"{family_id}:unknown_pair:{pair_id!r}")
    return problems


def build_pair(item: dict, spec: dict):
    """规格背景为 frame 背景；old/new 只接受 text，条件与时限都写在 text 里。"""
    old = {"kind": "memory", "text": item["old"]["text"]}
    new = {"kind": "memory", "text": item["new"]["text"]}
    pair = {
        "frame": {"mode": "same_domain", "background": spec["background"]},
        "old": old,
        "new": new,
    }
    return semantic().validate_semantic_pair(pair)


                                                    


def cmd_specs(args) -> dict:
    specs = build_specs()
    problems = validate_spec_set(specs, full=True)
    if problems:
        return {"ok": False, "command": "specs", "problems": bounded(problems)}
    out_dir = Path(args.out) if args.out else DEFAULT_SPECS.parent
    out_path = out_dir / "specs.jsonl"
    setting_ids_by_split = {
        split: sorted({spec["setting"]["id"] for spec in specs if spec["split"] == split})
        for split in SPLITS
    }
    specs_and_meta = {
        "schema": "relation-specs-manifest-v3",
        "content_revision": CONTENT_REVISION,
        "narrative": NARRATIVE,
        "generated_at_utc": utc_now(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "family_count": len(specs),
        "items_per_family": ITEMS_PER_FAMILY,
        "items_total": sum(len(spec["items"]) for spec in specs),
        "split_families": dict(Counter(spec["split"] for spec in specs)),
        "mechanisms": dict(Counter(spec["mechanism"] for spec in specs)),
        "mechanism_by_split": {
            split: dict(Counter(spec["mechanism"] for spec in specs if spec["split"] == split))
            for split in SPLITS
        },
        "settings": len({spec["setting"]["id"] for spec in specs}),
        "settings_by_split": setting_ids_by_split,
        "setting_count_by_split": {split: len(ids) for split, ids in setting_ids_by_split.items()},
        "unique_setting_scope": len({(spec["setting"]["id"], spec["setting"]["scope"]) for spec in specs}),
        "setting_scope_cross_split": sorted(
            key
            for key, splits in (
                (key, {spec["split"] for spec in specs if (spec["setting"]["id"], spec["setting"]["scope"]) == key})
                for key in {(spec["setting"]["id"], spec["setting"]["scope"]) for spec in specs}
            )
            if len(splits) > 1
        ),
        "specs_jsonl_sha256": sha256_text(
            "".join(canonical_json(spec) + "\n" for spec in specs)
        ),
        "family_ids": [spec["family_id"] for spec in specs],
    }
    if not args.dry_run:
        write_jsonl_atomic(out_path, specs)
        write_json_atomic(out_dir / "manifest.json", specs_and_meta)
    summary = {
        "ok": True,
        "command": "specs",
        "content_revision": CONTENT_REVISION,
        "out": str(out_path),
        "written": not args.dry_run,
        "family_count": len(specs),
        "items_total": specs_and_meta["items_total"],
        "split_families": specs_and_meta["split_families"],
        "mechanisms": specs_and_meta["mechanisms"],
        "mechanism_by_split": specs_and_meta["mechanism_by_split"],
        "settings": specs_and_meta["settings"],
        "setting_count_by_split": specs_and_meta["setting_count_by_split"],
        "unique_setting_scope": specs_and_meta["unique_setting_scope"],
        "setting_scope_cross_split": specs_and_meta["setting_scope_cross_split"],
        "specs_sha256": specs_and_meta["specs_jsonl_sha256"],
    }
    return summary


def load_specs(path: Path) -> list[dict]:
    if not path.is_file():
        raise DataError("specs_missing", str(path))
    specs = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            specs.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise DataError("specs_bad_json", f"line {line_no}: {exc}")
    return specs


                                                


def load_channel(module_path: Path, model: str, num_predict: int):
    if not module_path.is_file():
        raise DataError("channel_module_missing", str(module_path))
    name = "exp_model_channel"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    cls = getattr(module, "OllamaChannel", None)
    if cls is None:
        raise DataError("ollama_channel_missing", str(module_path))
    init_params = inspect.signature(cls.__init__).parameters
    complete_params = inspect.signature(cls.complete).parameters
    init_kwargs = {}
    call_kwargs = {}
    mode = "channel-default"
    if "num_predict" in init_params:
        init_kwargs["num_predict"] = num_predict
        mode = "constructor"
    elif "max_tokens" in init_params:
        init_kwargs["max_tokens"] = num_predict
        mode = "constructor-max_tokens"
    if "options" in complete_params:
        call_kwargs["options"] = {"temperature": 0, "num_predict": num_predict}
        mode = "complete-options"
    elif "num_predict" in complete_params:
        call_kwargs["num_predict"] = num_predict
        mode = "complete-num_predict"
    channel_default = None
    if "num_predict" in init_params and isinstance(init_params["num_predict"].default, int):
        channel_default = int(init_params["num_predict"].default)
    if channel_default is None:
        try:
            source = inspect.getsource(cls.complete)
            match = re.search(r'"num_predict"\s*:\s*(\d+)', source)
            if match:
                channel_default = int(match.group(1))
        except (OSError, TypeError):
            channel_default = None
    meta = {
        "requested_num_predict": num_predict,
        "override_mode": mode,
        "channel_default_num_predict": channel_default,
        "effective_num_predict": num_predict if mode != "channel-default" else channel_default,
        "budget_risk": mode == "channel-default" and (channel_default or 0) < num_predict,
    }

    def factory(_spec):
        return cls(model, **init_kwargs)

    return factory, meta, call_kwargs


                                                            


def model_safe_name(model: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._-")
    return safe or "model-" + sha256_text(model)[:8]


def _read_json_if_exists(path: Path):
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except json.JSONDecodeError:
        raise DataError("record_bad_json", str(path))


def author_specs(
    *,
    specs: list[dict],
    out_dir: Path,
    channel_factory,
    channel_meta: dict,
    call_kwargs: dict,
    max_attempts: int = MAX_ATTEMPTS,
    limit: int | None = None,
    sleep_s: float = 0.0,
    dry_run: bool = False,
) -> dict:
    selected = specs[:limit] if limit else specs
    report = {
        "ok": True,
        "command": "author-ollama",
        "out_dir": str(out_dir),
        "planned": 0,
        "authored": [],
        "skipped_success": [],
        "failed": [],
        "stale_conflicts": [],
        "channel": channel_meta,
        "usage_totals": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    for spec in selected:
        family_id = spec["family_id"]
        current_sha = spec_sha256(spec)
        record_path = out_dir / f"{family_id}.json"
        existing = _read_json_if_exists(record_path)
        if existing is not None:
            if existing.get("spec_sha256") == current_sha and existing.get("status") == "success":
                report["skipped_success"].append(family_id)
                continue
            if existing.get("spec_sha256") != current_sha:
                report["stale_conflicts"].append(
                    {
                        "family_id": family_id,
                        "path": str(record_path),
                        "existing_spec_sha256": existing.get("spec_sha256"),
                        "current_spec_sha256": current_sha,
                    }
                )
                continue
        prompt = build_prompt(spec)
        report["planned"] += 1
        if dry_run:
            continue
        attempts = []
        response_text = None
        extraction = None
        item_problems = []
        channel_model = channel_meta.get("model") or ""
        for attempt in range(1, max_attempts + 1):
            channel = None
            try:
                channel = channel_factory(spec)
                channel_model = getattr(channel, "model", channel_model)
                outcome = channel.complete(prompt, **call_kwargs)
            except Exception as exc:                   
                outcome = {
                    "text": None,
                    "succeeded": False,
                    "status": getattr(exc, "code", "channel_error"),
                    "finish_reason": None,
                    "usage": {},
                    "latency_s": 0.0,
                    "provider": "unknown",
                    "model": channel_model,
                    "error_kind": getattr(exc, "code", "channel_error"),
                    "attempted": True,
                    "diagnostics": {"error": str(exc)[:400]},
                }
            usage = outcome.get("usage") or {}
            attempts.append(
                {
                    "attempt": attempt,
                    "succeeded": bool(outcome.get("succeeded")),
                    "status": outcome.get("status"),
                    "error_kind": outcome.get("error_kind"),
                    "finish_reason": outcome.get("finish_reason"),
                    "latency_s": outcome.get("latency_s"),
                    "usage": usage,
                }
            )
            for key in report["usage_totals"]:
                value = usage.get(key)
                if isinstance(value, int):
                    report["usage_totals"][key] += value
            if not outcome.get("succeeded"):
                continue
            response_text = outcome.get("text") or ""
            try:
                items, extraction = extract_json_array(response_text)
                item_problems = check_author_items(items, spec)
            except DataError as exc:
                item_problems = [f"{family_id}:{exc.code}:{exc.detail}"]
                extraction = None
            if not item_problems:
                break
            response_text = response_text if response_text else None
        status = "success" if (response_text and extraction and not item_problems) else "failed"
        record = {
            "schema": "relation-author-v1",
            "family_id": family_id,
            "split": spec["split"],
            "model": channel_meta.get("model"),
            "model_safe": channel_meta.get("model_safe"),
            "spec_sha256": current_sha,
            "prompt_sha256": sha256_text(prompt),
            "prompt_chars": len(prompt),
            "prompt": prompt,
            "spec": spec,
            "response": response_text,
            "response_sha256": sha256_text(response_text) if response_text else None,
            "response_chars": len(response_text) if response_text else 0,
            "extraction": extraction,
            "status": status,
            "problems": item_problems[:20],
            "attempts": attempts,
            "usage": attempts[-1]["usage"] if attempts else {},
            "channel": channel_meta,
            "created_at_utc": utc_now(),
        }
        write_json_atomic(record_path, record)
        if status == "success":
            report["authored"].append(family_id)
        else:
            report["failed"].append({"family_id": family_id, "problems": item_problems[:5]})
        if sleep_s:
            time.sleep(sleep_s)
    if report["stale_conflicts"] or report["failed"]:
        report["ok"] = False
    report["counts"] = {
        "planned": report["planned"],
        "authored": len(report["authored"]),
        "skipped_success": len(report["skipped_success"]),
        "failed": len(report["failed"]),
        "stale_conflicts": len(report["stale_conflicts"]),
    }
    report["authored"] = report["authored"][:80]
    report["skipped_success"] = report["skipped_success"][:80]
    report["failed"] = report["failed"][:20]
    report["stale_conflicts"] = report["stale_conflicts"][:10]
    return report


def cmd_author_ollama(args) -> dict:
    specs = load_specs(Path(args.specs))
    problems = validate_spec_set(specs, full=False)
    if problems:
        return {"ok": False, "command": "author-ollama", "problems": bounded(problems)}
    if args.split:
        wanted = set(args.split)
        unknown = wanted - set(SPLITS)
        if unknown:
            return {"ok": False, "command": "author-ollama", "error": f"unknown splits {sorted(unknown)}"}
        specs = [spec for spec in specs if spec["split"] in wanted]
    if args.family:
        wanted_families = set(args.family)
        specs = [spec for spec in specs if spec["family_id"] in wanted_families]
    if not specs:
        return {"ok": False, "command": "author-ollama", "error": "no families selected"}
    safe = model_safe_name(args.model)
    out_dir = Path(args.out) if args.out else EXP_ROOT / "data/authors" / safe
    meta = {
        "model": args.model,
        "model_safe": safe,
        "requested_num_predict": args.num_predict,
        "override_mode": "dry-run",
        "channel_default_num_predict": None,
        "effective_num_predict": None,
        "budget_risk": None,
    }
    call_kwargs = {}
    if not args.dry_run:
        module_path = Path(args.channel_module) if args.channel_module else EXP_ROOT / "model_channel.py"
        factory, loaded_meta, call_kwargs = load_channel(module_path, args.model, args.num_predict)
        meta.update(loaded_meta)
        meta["model"] = args.model
        meta["model_safe"] = safe
        factory_for_run = factory
    else:
        factory_for_run = None
    report = author_specs(
        specs=specs,
        out_dir=out_dir,
        channel_factory=factory_for_run,
        channel_meta=meta,
        call_kwargs=call_kwargs,
        max_attempts=args.max_attempts,
        limit=args.limit,
        sleep_s=args.sleep,
        dry_run=args.dry_run,
    )
    return report


                                                    


def load_source_rows(path: Path):
    if not path.is_file():
        raise DataError("source_rows_missing", str(path))
    sr = semantic()
    rows = []
    problems = []
    normalized_pair = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataError("source_rows_bad_json", f"line {line_no}: {exc}")
        missing = [key for key in ("id", "root_id", "cluster_id", "split", "pair", "label") if key not in raw]
        if missing:
            problems.append(f"line {line_no}: missing {missing}")
            continue
        if raw["split"] not in SPLITS:
            problems.append(f"line {line_no}: bad split {raw['split']!r}")
        try:
            pair = sr.validate_semantic_pair(raw["pair"])
        except Exception as exc:
            problems.append(f"line {line_no}: pair invalid {exc}")
            continue
        if pair != raw["pair"]:
            normalized_pair += 1
        rows.append(
            {
                "id": raw["id"],
                "root_id": raw["root_id"],
                "cluster_id": raw["cluster_id"],
                "split": raw["split"],
                "label": raw["label"],
                "label_origin": raw.get("label_origin") or "source_row",
                "pair": raw["pair"],
                "validated_pair": pair,
                "visible_text": sr.semantic_pair_text(pair),
                "phenomena": raw.get("phenomena") or {},
            }
        )
    return rows, problems, {"normalized_pairs": normalized_pair, "rows": len(rows)}


def legacy_row(row: dict) -> dict:
    _fine, _coarse, mapping = relation_mapping()
    return {
        "id": row["id"],
        "root_id": row["root_id"],
        "cluster_id": row["cluster_id"],
        "family_id": None,
        "split": row["split"],
        "label": row["label"],
        "coarse_label": mapping[row["label"]],
        "label_origin": row["label_origin"],
        "source": ROW_SOURCE_LEGACY,
        "author": None,
        "pair": row["pair"],
        "visible_text": row["visible_text"],
        "phenomena": row["phenomena"],
    }


                                                       


AUTHOR_FILE_RE = re.compile(r"^af-\d{3}\.json$")


def load_quarantine(path: Path | None):
    if path is None or not Path(path).is_file():
        return {}
    payload = read_json(Path(path))
    families = payload.get("families") if isinstance(payload, dict) else None
    if not isinstance(families, list):
        raise DataError("quarantine_bad_shape", str(path))
    result = {}
    for entry in families:
        if not isinstance(entry, dict) or not isinstance(entry.get("family_id"), str):
            raise DataError("quarantine_bad_entry", str(entry)[:120])
        result[entry["family_id"]] = {
            "family_id": entry["family_id"],
            "reason": entry.get("reason") or "unspecified",
        }
    return result


def collect_author_records(author_dirs, specs_by_id: dict):
    records = {}
    problems = []
    ignored = []
    inputs = []
    for directory in author_dirs:
        base = Path(directory)
        if not base.is_dir():
            problems.append(f"author_dir_missing:{base}")
            continue
        files = sorted(base.glob("*.json"))
        inputs.append({"dir": str(base), "files": len(files)})
        for path in files:
            if not AUTHOR_FILE_RE.match(path.name):
                ignored.append(str(path))
                continue
            try:
                record = read_json(path)
            except json.JSONDecodeError as exc:
                problems.append(f"record_bad_json:{path}:{exc}")
                continue
            family_id = record.get("family_id") or path.stem
            if family_id not in specs_by_id:
                problems.append(f"unknown_family_record:{family_id}:{path}")
                continue
            if family_id in records:
                problems.append(f"duplicate_family_record:{family_id}")
                continue
            if record.get("spec_sha256") != spec_sha256(specs_by_id[family_id]):
                problems.append(f"stale_spec_record:{family_id}:{path.name}")
                continue
            if record.get("status") != "success":
                problems.append(f"record_not_success:{family_id}")
                continue
            records[family_id] = {"record": record, "path": path}
    return records, problems, ignored, inputs


def assemble_dataset(
    *,
    specs: list[dict],
    source_rows_path: Path,
    author_dirs,
    out_dir: Path,
    quarantine: dict,
    dry_run: bool = False,
):
    sr = semantic()
    fine, coarse, mapping = relation_mapping()
    specs_by_id = {spec["family_id"]: spec for spec in specs}
    problems = validate_spec_set(specs, full=False)
    if problems:
        return {"ok": False, "problems": bounded(problems)}

    source_rows, source_problems, source_stats = load_source_rows(Path(source_rows_path))
    problems.extend(source_problems)
    records, record_problems, ignored_files, author_inputs = collect_author_records(
        author_dirs, specs_by_id
    )
    problems.extend(record_problems)

    active_specs = [spec for spec in specs if spec["family_id"] not in quarantine]
    missing_families = [spec["family_id"] for spec in active_specs if spec["family_id"] not in records]
    if missing_families:
        problems.append(f"families_without_author_record:{len(missing_families)}")
        problems.extend([f"missing_author:{fid}" for fid in missing_families[:20]])
    if problems:
        return {
            "ok": False,
            "problems": bounded(problems),
            "missing_families": missing_families[:20],
        }

    rows_by_split = {split: [] for split in SPLITS}
    family_stats = []
    background_mismatch = 0
    text_short = 0
    for row in source_rows:
        rows_by_split[row["split"]].append(legacy_row(row))

    for spec in active_specs:
        family_id = spec["family_id"]
        record = records[family_id]["record"]
        items, extraction = extract_json_array(record.get("response") or "")
        item_problems = check_author_items(items, spec)
        if item_problems:
            problems.extend(item_problems)
            continue
        by_id = {item["pair_id"]: item for item in items}
        family_rows = []
        for position, spec_item in enumerate(spec["items"]):
            pair_id = spec_item["pair_id"]
            item = by_id[pair_id]
            if normalize_text(item["background"]) != normalize_text(spec["background"]):
                background_mismatch += 1
            try:
                pair = build_pair(item, spec)
                visible = sr.semantic_pair_text(pair)
            except Exception as exc:
                problems.append(f"{family_id}:pair_invalid:{pair_id}:{exc}")
                continue
            for side in ("old", "new"):
                if len(item[side]["text"].strip()) < _MIN_SENTENCE_CHARS:
                    text_short += 1
            label = spec_item["fine_label"]
            family_rows.append(
                {
                    "id": pair_id,
                    "root_id": f"{family_id}-r{position:02d}",
                    "cluster_id": family_id,
                    "family_id": family_id,
                    "split": spec["split"],
                    "label": label,
                    "coarse_label": mapping[label],
                    "label_origin": "spec_author",
                    "source": ROW_SOURCE_NEW,
                    "author": record.get("model"),
                    "pair": pair,
                    "visible_text": visible,
                    "phenomena": {
                        "family": spec["mechanism"],
                        "setting": spec["setting"]["id"],
                        "variant": spec_item["variant"],
                        "language": "zh",
                        "mode": "assertion",
                        "source_ids": [family_id],
                        "author_model": record.get("model"),
                    },
                }
            )
        if len(family_rows) != ITEMS_PER_FAMILY:
            problems.append(f"{family_id}:assembled_rows:{len(family_rows)}")
            continue
        counts = Counter(row["label"] for row in family_rows)
        for label in fine:
            if counts[label] != PER_FINE:
                problems.append(f"{family_id}:assembled_fine_count:{label}:{counts[label]}")
        rows_by_split[spec["split"]].extend(family_rows)
        family_stats.append(
            {
                "family_id": family_id,
                "split": spec["split"],
                "mechanism": spec["mechanism"],
                "setting": spec["setting"]["id"],
                "rows": len(family_rows),
                "author": record.get("model"),
                "extraction": extraction,
            }
        )
    if problems:
        return {"ok": False, "problems": bounded(problems)}

    ordered = {}
    for split in SPLITS:
        ordered[split] = sorted(rows_by_split[split], key=lambda row: sha256_text(f"order|{row['id']}"))

    expected = {}
    for split in SPLITS:
        legacy_expected = LEGACY_EXPECTED[split]
        new_expected = ITEMS_PER_FAMILY * sum(1 for spec in active_specs if spec["split"] == split)
        expected[split] = {"legacy": legacy_expected, "new": new_expected, "total": legacy_expected + new_expected}

    all_rows = [row for split in SPLITS for row in ordered[split]]
    counts = {
        "by_split": {split: len(ordered[split]) for split in SPLITS},
        "by_source": dict(Counter(row["source"] for row in all_rows)),
        "by_label": dict(Counter(row["label"] for row in all_rows)),
        "by_coarse": dict(Counter(row["coarse_label"] for row in all_rows)),
        "by_author": dict(Counter(str(row["author"]) for row in all_rows)),
    }
    processed_paths = {split: out_dir / "processed" / f"{split}.jsonl" for split in SPLITS}
    processed_sha = {}
    if not dry_run:
        for split in SPLITS:
            write_jsonl_atomic(processed_paths[split], ordered[split])
            processed_sha[split] = sha256_file(processed_paths[split])
    model_counts = Counter()
    for item in family_stats:
        model_counts[str(item["author"])] += 1
    manifest = {
        "schema": "relation-dataset-manifest-v3",
        "content_revision": CONTENT_REVISION,
        "narrative": NARRATIVE,
        "built_at_utc": utc_now(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": {
            "source_rows": {
                "path": str(Path(source_rows_path).resolve()),
                "sha256": sha256_file(Path(source_rows_path)),
                "rows": len(source_rows),
                "normalized_pairs": source_stats["normalized_pairs"],
            },
            "specs": {
                "count": len(specs),
                "sha256": sha256_json(specs),
            },
            "authors": [
                {
                    "dir": item["dir"],
                    "files": item["files"],
                    "sha256": {
                        entry["record"].get("family_id"): sha256_file(entry["path"])
                        for entry in records.values()
                        if str(entry["path"]).startswith(item["dir"])
                    },
                }
                for item in author_inputs
            ],
            "ignored_author_files": ignored_files[:20],
        },
        "models": {
            str(model): {"families": count, "model_sha256": sha256_text(str(model))}
            for model, count in sorted(model_counts.items())
        },
        "counts": counts,
        "family_count": len(family_stats),
        "families": family_stats,
        "quarantine": {
            "families": sorted(quarantine.values(), key=lambda item: item["family_id"]),
            "count": len(quarantine),
        },
        "expected": expected,
        "processed": {
            split: {
                "path": str(processed_paths[split]),
                "sha256": processed_sha.get(split),
                "rows": len(ordered[split]),
            }
            for split in SPLITS
        },
        "checks": {
            "background_mismatch_repaired": background_mismatch,
            "short_sentences_under_10_chars": text_short,
        },
    }
    if not dry_run:
        write_json_atomic(out_dir / "manifest.json", manifest)
    return {
        "ok": True,
        "command": "assemble",
        "out_dir": str(out_dir),
        "written": not dry_run,
        "counts": counts,
        "family_count": len(family_stats),
        "expected": expected,
        "processed_sha256": processed_sha,
        "background_mismatch_repaired": background_mismatch,
        "short_sentences_under_10_chars": text_short,
        "quarantine": manifest["quarantine"],
    }


def cmd_assemble(args) -> dict:
    specs = load_specs(Path(args.specs))
    if not args.authors:
        return {"ok": False, "command": "assemble", "error": "--authors is required"}
    quarantine = load_quarantine(Path(args.quarantine) if args.quarantine else None)
    out_dir = Path(args.out) if args.out else EXP_ROOT / "data"
    source_path = Path(args.source_rows) if args.source_rows else DEFAULT_SOURCE_ROWS
    return assemble_dataset(
        specs=specs,
        source_rows_path=source_path,
        author_dirs=[Path(item) for item in args.authors],
        out_dir=out_dir,
        quarantine=quarantine,
        dry_run=args.dry_run,
    )


                                                    


def load_processed(processed_dir: Path):
    sr = semantic()
    fine, coarse, mapping = relation_mapping()
    rows = []
    problems = []
    per_file = {}
    for split in SPLITS:
        path = Path(processed_dir) / f"{split}.jsonl"
        if not path.is_file():
            problems.append(f"missing_processed_file:{split}")
            continue
        count = 0
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"{split}:{line_no}:bad_json:{exc}")
                continue
            if row.get("split") != split:
                problems.append(f"{split}:{line_no}:split_field_mismatch:{row.get('split')!r}")
            rows.append(row)
        per_file[split] = count
    for row in rows:
        missing = [
            key
            for key in (
                "id", "root_id", "cluster_id", "family_id", "split", "label",
                "coarse_label", "label_origin", "source", "author", "pair",
                "visible_text", "phenomena",
            )
            if key not in row
        ]
        if missing:
            problems.append(f"row_missing_keys:{row.get('id')}:{missing}")
            continue
        try:
            validated = sr.validate_semantic_pair(row["pair"])
            expected_text = sr.semantic_pair_text(validated)
        except Exception as exc:
            problems.append(f"pair_invalid:{row['id']}:{exc}")
            continue
        if row["visible_text"] != expected_text:
            problems.append(f"visible_text_not_canonical:{row['id']}")
        if row["label"] not in mapping:
            problems.append(f"label_unknown:{row['id']}:{row['label']!r}")
        elif row["coarse_label"] != mapping[row["label"]]:
            problems.append(f"coarse_mismatch:{row['id']}:{row['coarse_label']!r}")
        if not str(row["visible_text"]).strip():
            problems.append(f"empty_visible_text:{row['id']}")
    return rows, problems, per_file


def _group_rows(rows, key):
    groups = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        groups[value].append(row)
    return groups


def _cross_split_groups(groups):
    result = []
    for value, members in groups.items():
        splits = {row["split"] for row in members}
        if len(splits) > 1:
            result.append({"key": value, "splits": sorted(splits), "ids": [row["id"] for row in members][:6]})
    return result


def _split_impact(groups):
    """把跨 split 冲突拆成只涉及 legacy、以及涉及新数据的两类。"""
    legacy_only = []
    new_involved = []
    for entry in groups:
        members = entry.pop("_members")
        if all(row["source"] == ROW_SOURCE_LEGACY for row in members):
            legacy_only.append(entry)
        else:
            entry["sources"] = sorted({row["source"] for row in members})
            new_involved.append(entry)
    return legacy_only, new_involved


def audit_dataset(
    *,
    processed_dir: Path,
    specs: list[dict],
    source_rows_path: Path,
    quarantine: dict,
    mode: str = "full",
) -> dict:
    rows, load_problems, per_file = load_processed(Path(processed_dir))
    checks = {}

    def check(name: str, ok: bool, **extra):
        checks[name] = {"ok": bool(ok), **extra}

    check("load", not load_problems, problems=bounded(load_problems))

    source_rows, source_problems, source_stats = load_source_rows(Path(source_rows_path))
    check("source_rows", not source_problems, problems=bounded(source_problems), rows=len(source_rows))

    fine, coarse, mapping = relation_mapping()
    specs_by_id = {spec["family_id"]: spec for spec in specs}
    active_specs = [spec for spec in specs if spec["family_id"] not in quarantine]
    quarantined_ids = set(quarantine)

    new_rows = [row for row in rows if row.get("source") == ROW_SOURCE_NEW]
    legacy_rows = [row for row in rows if row.get("source") == ROW_SOURCE_LEGACY]
    unknown_source = [row for row in rows if row.get("source") not in (ROW_SOURCE_NEW, ROW_SOURCE_LEGACY)]
    check("source_partition", not unknown_source, count=len(unknown_source))

                               
    actual_by_split = {split: len([row for row in rows if row.get("split") == split]) for split in SPLITS}
    legacy_by_split = {split: len([row for row in legacy_rows if row.get("split") == split]) for split in SPLITS}
    new_by_split = {split: len([row for row in new_rows if row.get("split") == split]) for split in SPLITS}
    expected_new = {
        split: ITEMS_PER_FAMILY * sum(1 for spec in active_specs if spec["split"] == split)
        for split in SPLITS
    }
    if mode == "full":
        legacy_expected = dict(LEGACY_EXPECTED)
        counts_ok = legacy_by_split == legacy_expected and new_by_split == expected_new
    else:
        legacy_expected = {split: len([row for row in source_rows if row["split"] == split]) for split in SPLITS}
        counts_ok = legacy_by_split == legacy_expected and new_by_split == expected_new
    check(
        "counts",
        counts_ok,
        actual_by_split=actual_by_split,
        legacy_by_split=legacy_by_split,
        new_by_split=new_by_split,
        legacy_expected=legacy_expected,
        new_expected=expected_new,
        total_expected=sum(legacy_expected.values()) + sum(expected_new.values()),
        total_actual=len(rows),
    )

                             
    source_by_id = {row["id"]: row for row in source_rows}
    missing_ids = sorted(set(source_by_id) - {row.get("id") for row in legacy_rows})
    legacy_split_mismatch = [
        row["id"] for row in legacy_rows
        if row["id"] in source_by_id and source_by_id[row["id"]]["split"] != row.get("split")
    ]
    legacy_label_mismatch = [
        row["id"] for row in legacy_rows
        if row["id"] in source_by_id and source_by_id[row["id"]]["label"] != row.get("label")
    ]
    check(
        "legacy_preserved",
        not missing_ids and not legacy_split_mismatch and not legacy_label_mismatch,
        missing=len(missing_ids),
        split_mismatch=bounded(legacy_split_mismatch),
        label_mismatch=bounded(legacy_label_mismatch),
    )

                                    
    family_rows = defaultdict(list)
    for row in new_rows:
        family_rows[row.get("family_id")].append(row)
    family_problems = []
    for spec in active_specs:
        family_id = spec["family_id"]
        members = family_rows.get(family_id, [])
        if len(members) != ITEMS_PER_FAMILY:
            family_problems.append(f"{family_id}:rows:{len(members)}")
        counts = Counter(row["label"] for row in members)
        for label in fine:
            if counts[label] != PER_FINE:
                family_problems.append(f"{family_id}:{label}:{counts[label]}")
        expected_ids = {item["pair_id"] for item in spec["items"]}
        actual_ids = {row["id"] for row in members}
        if expected_ids != actual_ids:
            family_problems.append(f"{family_id}:id_set")
        expected_labels = {item["pair_id"]: item["fine_label"] for item in spec["items"]}
        for row in members:
            if expected_labels.get(row["id"]) not in (None, row["label"]):
                family_problems.append(f"{family_id}:spec_label:{row['id']}")
        if members and members[0]["split"] != spec["split"]:
            family_problems.append(f"{family_id}:split")
    unknown_families = sorted(set(family_rows) - set(specs_by_id))
    if unknown_families:
        family_problems.append(f"unknown_families:{unknown_families[:5]}")
    if quarantined_ids & set(family_rows):
        family_problems.append(f"quarantined_rows_present:{sorted(quarantined_ids & set(family_rows))[:5]}")
    check("family_shape", not family_problems, problems=bounded(family_problems))

    families_per_split = Counter(spec["split"] for spec in active_specs)
    unknown_quarantine = sorted(quarantined_ids - set(specs_by_id))
    if mode == "full":
        quarantined_by_split = Counter(
            spec["split"] for spec in specs if spec["family_id"] in quarantined_ids
        )
        expected_families = {
            split: FAMILY_EXPECTED[split] - quarantined_by_split.get(split, 0) for split in SPLITS
        }
        families_ok = dict(families_per_split) == expected_families and not unknown_quarantine
    else:
        expected_families = dict(families_per_split)
        families_ok = not unknown_quarantine
    check(
        "family_splits",
        families_ok,
        actual=dict(families_per_split),
        expected=expected_families,
        quarantined=sorted(quarantined_ids),
        unknown_quarantine=unknown_quarantine[:10],
    )

                                                                               
    exact_groups = _group_rows(rows, "visible_text")
    exact_cross = _cross_split_groups(exact_groups)
    for entry in exact_cross:
        entry["_members"] = exact_groups[entry["key"]]
    exact_legacy, exact_new = _split_impact(exact_cross)
    check(
        "cross_split_visible_exact",
        not exact_new,
        new_involved=bounded(exact_new),
        legacy_only=bounded(exact_legacy),
    )

    normalized_groups = defaultdict(list)
    for row in rows:
        normalized_groups[normalize_text(row["visible_text"])].append(row)
    norm_cross = []
    for value, members in normalized_groups.items():
        if len({row["split"] for row in members}) > 1:
            norm_cross.append(
                {
                    "key_sha256": sha256_text(value)[:16],
                    "splits": sorted({row["split"] for row in members}),
                    "ids": [row["id"] for row in members][:6],
                    "sources": sorted({row["source"] for row in members}),
                    "_members": members,
                }
            )
    norm_legacy = [entry for entry in norm_cross if all(row["source"] == ROW_SOURCE_LEGACY for row in entry["_members"])]
    norm_new = [entry for entry in norm_cross if entry not in norm_legacy]
    for entry in norm_legacy + norm_new:
        entry.pop("_members", None)
    check(
        "cross_split_visible_normalized",
        not norm_new,
        new_involved=bounded(norm_new),
        legacy_only=bounded(norm_legacy),
    )

    digest_map = defaultdict(list)
    for row in rows:
        digest_map[sha256_text(row["pair"].get("old", {}).get("text", "") + "\u241f"
                               + row["pair"].get("new", {}).get("text", "") + "\u241f"
                               + row["pair"].get("frame", {}).get("background", ""))].append(row)
    digest_cross = []
    for value, members in digest_map.items():
        if len({row["split"] for row in members}) > 1:
            digest_cross.append(
                {
                    "digest": value[:16],
                    "splits": sorted({row["split"] for row in members}),
                    "ids": [row["id"] for row in members][:6],
                    "_members": members,
                }
            )
    digest_legacy = [entry for entry in digest_cross if all(row["source"] == ROW_SOURCE_LEGACY for row in entry["_members"])]
    digest_new = [entry for entry in digest_cross if entry not in digest_legacy]
    for entry in digest_legacy + digest_new:
        entry.pop("_members", None)
    check(
        "cross_split_pair_digest",
        not digest_new,
        new_involved=bounded(digest_new),
        legacy_only=bounded(digest_legacy),
    )

    for key, name in (("root_id", "root"), ("family_id", "family"), ("cluster_id", "cluster")):
        groups = _group_rows(rows, key)
        cross = _cross_split_groups(groups)
        for entry in cross:
            entry["_members"] = groups[entry["key"]]
        legacy_only, new_involved = _split_impact(cross)
        check(
            f"cross_split_{name}",
            not new_involved,
            new_involved=bounded(new_involved),
            legacy_only=bounded(legacy_only),
        )

                   
    same_input_label_problems = []
    for value, members in exact_groups.items():
        labels = {row["label"] for row in members}
        if len(labels) > 1:
            same_input_label_problems.append({"ids": [row["id"] for row in members][:6], "labels": sorted(labels)})
    norm_label_problems = []
    for value, members in normalized_groups.items():
        labels = {row["label"] for row in members}
        if len(labels) > 1:
            norm_label_problems.append(
                {"key_sha256": sha256_text(value)[:16], "labels": sorted(labels)}
            )
    check(
        "same_input_different_label",
        not same_input_label_problems and not norm_label_problems,
        exact=bounded(same_input_label_problems),
        normalized=bounded(norm_label_problems),
    )

                              
    bad_mapping = [row["id"] for row in rows if row.get("coarse_label") != mapping.get(row.get("label"))]
    check("fine_to_coarse", not bad_mapping, bad=len(bad_mapping), mapping=dict(mapping))

                      
    empty_visible = [row.get("id") for row in rows if not str(row.get("visible_text") or "").strip()]
    check("visible_text_nonempty", not empty_visible, empty=len(empty_visible))

                  
    if mode == "full":
        spec_problems = validate_spec_set(specs, full=True)
        check("specs", not spec_problems, problems=bounded(spec_problems))
        check(
            "expected_totals",
            sum(actual_by_split.values()) == sum(LEGACY_EXPECTED.values())
            + ITEMS_PER_FAMILY * len(active_specs),
            total_expected=sum(LEGACY_EXPECTED.values()) + ITEMS_PER_FAMILY * len(active_specs),
            total_actual=sum(actual_by_split.values()),
        )
    else:
        check("expected_totals", sum(actual_by_split.values()) == len(rows), total_actual=len(rows))

    failures = [name for name, value in checks.items() if not value["ok"]]
    legacy_notes = {
        "cross_split_visible_exact": checks["cross_split_visible_exact"].get("legacy_only"),
        "cross_split_visible_normalized": checks["cross_split_visible_normalized"].get("legacy_only"),
        "cross_split_pair_digest": checks["cross_split_pair_digest"].get("legacy_only"),
        "cross_split_root": checks["cross_split_root"].get("legacy_only"),
        "cross_split_family": checks["cross_split_family"].get("legacy_only"),
        "cross_split_cluster": checks["cross_split_cluster"].get("legacy_only"),
    }
    return {
        "ok": not failures,
        "command": "audit",
        "mode": mode,
        "processed_dir": str(processed_dir),
        "checks": checks,
        "failures": failures,
        "legacy_notes": {
            name: value for name, value in legacy_notes.items()
            if value and value.get("count")
        },
        "counts": {
            "by_split": actual_by_split,
            "legacy": legacy_by_split,
            "new": new_by_split,
            "families_per_split": dict(families_per_split),
            "quarantined_families": sorted(quarantined_ids),
            "rows": len(rows),
        },
    }


def cmd_audit(args) -> dict:
    specs = load_specs(Path(args.specs))
    quarantine = load_quarantine(Path(args.quarantine) if args.quarantine else None)
    source_path = Path(args.source_rows) if args.source_rows else DEFAULT_SOURCE_ROWS
    processed_dir = Path(args.processed) if args.processed else DEFAULT_PROCESSED
    return audit_dataset(
        processed_dir=processed_dir,
        specs=specs,
        source_rows_path=source_path,
        quarantine=quarantine,
        mode=args.mode,
    )


                                                    


class _StubChannel:
    """只在 selftest 中使用：按规格生成合法 JSON 数组，不访问网络或真实模型。"""

    name = "stub"
    provider = "stub"

    def __init__(self, spec: dict, *, truncate: bool = False) -> None:
        self.spec = spec
        self.truncate = truncate
        self.model = "stub-model"

    def complete(self, prompt: str, **_kwargs) -> dict:
        items = []
        setting = self.spec["setting"]
        for position, item in enumerate(self.spec["items"]):
            stem = f"{self.spec['family_id']}{setting['name']}{setting['scope']}第{position}条"
            items.append(
                {
                    "pair_id": item["pair_id"],
                    "fine_label": item["fine_label"],
                    "old": {"text": f"{stem}旧口径要求清单齐全并当日复核"},
                    "new": {"text": f"{stem}新口径要求清单齐全"},
                    "background": self.spec["background"],
                }
            )
        text = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
        return {
            "text": text,
            "succeeded": True,
            "status": "ok",
            "finish_reason": None,
            "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
            "latency_s": 0.0,
            "provider": "stub",
            "model": self.model,
            "error_kind": None,
            "attempted": True,
            "diagnostics": {},
        }


class _StringEntryChannel(_StubChannel):
    """只在 selftest 中使用：故意把 old/new 写成非空字符串，验证归一化。"""

    name = "stub-strings"
    provider = "stub"

    def complete(self, prompt: str, **_kwargs) -> dict:
        outcome = dict(super().complete(prompt, **_kwargs))
        items = json.loads(outcome["text"])
        for item in items:
            item["old"] = item["old"]["text"]
            item["new"] = item["new"]["text"]
        outcome["text"] = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
        return outcome


def _fixture_pair(old_text: str, new_text: str, background: str) -> dict:    return semantic().validate_semantic_pair(
        {
            "frame": {"mode": "same_domain", "background": background},
            "old": {"kind": "proposition", "text": old_text},
            "new": {"kind": "proposition", "text": new_text},
        }
    )


def _write_fixture_source(path: Path) -> int:
    sr = semantic()
    rows = []
    for index in range(3):
        pair = _fixture_pair(
            f"源行{index}：所有归还的书都登记过状态",
            f"源行{index}：所有归还的书都登记过状态并复核",
            f"源行{index}背景：还书台只有一份登记表。",
        )
        rows.append(
            {
                "id": f"legacy-{index}",
                "root_id": f"legacy-root-{index}",
                "cluster_id": f"legacy-cluster-{index}",
                "split": "train",
                "label": "new_entails_old_strict",
                "label_origin": "blind_teacher_proxy",
                "pair": pair,
                "visible_text": sr.semantic_pair_text(pair),
                "phenomena": {"family": "fixture"},
            }
        )
    pair = _fixture_pair("源行dev：所有车位都登记过", "源行dev：所有车位都登记过并复核", "源行dev背景：只有一份车位表。")
    rows.append(
        {
            "id": "legacy-dev",
            "root_id": "legacy-root-dev",
            "cluster_id": "legacy-cluster-dev",
            "split": "dev",
            "label": "new_entails_old_strict",
            "label_origin": "blind_teacher_proxy",
            "pair": pair,
            "visible_text": sr.semantic_pair_text(pair),
            "phenomena": {"family": "fixture"},
        }
    )
    write_jsonl_atomic(path, rows)
    return len(rows)


def _audit_fixture(processed_dir: Path, specs, source_path: Path, mode: str = "fixture") -> dict:
    return audit_dataset(
        processed_dir=processed_dir,
        specs=specs,
        source_rows_path=source_path,
        quarantine={},
        mode=mode,
    )


def run_selftest() -> dict:
    results = []
    work = Path(tempfile.mkdtemp(prefix="relation-data-selftest-"))
    try:
        fine, coarse, mapping = relation_mapping()
        sr = semantic()
        results.append(
            {
                "check": "mapping",
                "ok": fine == ("equivalent", "old_entails_new_strict", "new_entails_old_strict",
                               "disjoint", "overlap_incomparable")
                and coarse == ("equivalent", "compatible", "conflict")
                and mapping["disjoint"] == "conflict"
                and mapping["overlap_incomparable"] == "compatible",
                "fine": list(fine),
                "coarse": list(coarse),
            }
        )

        pair = _fixture_pair("所有归还的书都登记过状态", "所有归还的书都登记过状态并复核", "还书台只有一份登记表。")
        text = sr.semantic_pair_text(pair)
        bad_kind = False
        try:
            sr.validate_semantic_pair({"old": {"kind": "unknown", "text": "x"}, "new": {"kind": "proposition", "text": "y"}})
        except Exception:
            bad_kind = True
        results.append(
            {
                "check": "production_serialization",
                "ok": isinstance(pair, dict)
                and set(pair) == {"frame", "old", "new"}
                and sr.semantic_pair_text(pair) == text
                and text == sr.semantic_pair_text(sr.validate_semantic_pair(pair))
                and bad_kind,
                "visible_text_sha256": sha256_text(text),
            }
        )

        specs = build_specs()
        spec_problems = validate_spec_set(specs, full=True)
        results.append(
            {
                "check": "specs_full_set",
                "ok": not spec_problems,
                "problems": bounded(spec_problems),
                "families": len(specs),
                "settings": len({spec["setting"]["id"] for spec in specs}),
                "mechanisms": dict(Counter(spec["mechanism"] for spec in specs)),
                "split_families": dict(Counter(spec["split"] for spec in specs)),
                "settings_by_split": {
                    split: len({spec["setting"]["id"] for spec in specs if spec["split"] == split})
                    for split in SPLITS
                },
                "unique_setting_scope": len(
                    {(spec["setting"]["id"], spec["setting"]["scope"]) for spec in specs}
                ),
            }
        )

        story_spec = json.loads(canonical_json(specs[0]))
        story_spec["background"] = "车库盘点：每件工具的登记取值决定两句真假。"
        story_problems = validate_spec(story_spec)
        results.append(
            {
                "check": "generic_story_must_fail",
                "ok": any("banned_narrative" in problem for problem in story_problems)
                and any("background_missing_user_frame" in problem for problem in story_problems),
                "problems": bounded(story_problems),
            }
        )

        leak_specs = json.loads(canonical_json(specs))
        train_spec = next(spec for spec in leak_specs if spec["split"] == "train")
        heldout_spec = next(spec for spec in leak_specs if spec["split"] == "heldout")
        heldout_spec["setting"] = json.loads(canonical_json(train_spec["setting"]))
        heldout_spec["background"] = train_spec["background"]
        leak_problems = validate_spec_set(leak_specs, full=True)
        results.append(
            {
                "check": "setting_scope_leak_must_fail",
                "ok": any("duplicate_setting_scope" in problem for problem in leak_problems)
                and any("setting_crosses_split" in problem for problem in leak_problems),
                "problems": bounded(leak_problems),
            }
        )

        neg_spec = next(spec for spec in specs if spec["mechanism"] == "negation_exception")
        scope_item = next(
            item for item in neg_spec["items"] if item["variant"] == "new_subset_by_scope_restricted"
        )
        direction_ok = all(marker in scope_item["instruction"] for marker in ("豁免", "非高优先级", "一律"))
        mutated = json.loads(canonical_json(neg_spec))
        target = next(
            item for item in mutated["items"] if item["variant"] == "new_subset_by_scope_restricted"
        )
        target["instruction"] = "新句在例外名单不变的前提下，把覆盖范围再限制到高优先级事项；旧句不限制优先级。"
        mutated_problems = validate_spec(mutated)
        scope_restricted_ok = direction_ok and any(
            "instruction_missing_marker" in problem for problem in mutated_problems
        )
        results.append(
            {
                "check": "scope_restricted_direction",
                "ok": scope_restricted_ok,
                "instruction": scope_item["instruction"],
                "problems": bounded(mutated_problems),
            }
        )

        disjoint_spec = json.loads(canonical_json(neg_spec))
        disjoint_target = next(item for item in disjoint_spec["items"] if item["fine_label"] == "disjoint")
        disjoint_target["instruction"] = "旧句要求名单甲内的事项不得豁免，新句不要求名单甲内的事项不得豁免。"
        disjoint_problems = validate_spec(disjoint_spec)
        disjoint_soft_ok = any("disjoint_soft_wording" in problem for problem in disjoint_problems)
        results.append(
            {
                "check": "disjoint_soft_wording_must_fail",
                "ok": disjoint_soft_ok,
                "problems": bounded(disjoint_problems),
            }
        )

        list_spec = json.loads(canonical_json(neg_spec))
        list_target = next(item for item in list_spec["items"] if item["variant"] == "distinct_exception_sets")
        list_target["instruction"] = "旧句说除名单甲外全部适用，新句说除名单乙外全部适用；两份名单不同且互不包含。"
        list_problems = validate_spec(list_spec)
        list_algebra_ok = any("list_algebra" in problem for problem in list_problems)
        results.append(
            {
                "check": "exception_list_algebra_must_fail",
                "ok": list_algebra_ok,
                "problems": bounded(list_problems),
            }
        )

        premise_spec = next(spec for spec in specs if spec["mechanism"] == "scope_inclusion")
        premise_spec = json.loads(canonical_json(premise_spec))
        premise_target = premise_spec["items"][0]
        premise_target["instruction"] = premise_target["instruction"] + "；名单甲内的事项优先。"
        premise_problems = validate_spec(premise_spec)
        premise_ok = any("premise_missing" in problem for problem in premise_problems)
        results.append(
            {
                "check": "premise_self_sufficiency_must_fail",
                "ok": premise_ok,
                "problems": bounded(premise_problems),
            }
        )

        longterm_spec = next(spec for spec in specs if spec["mechanism"] == "longterm_one_off")
        longterm_bg = longterm_spec["background"]
        longterm_bg_ok = all(
            marker in longterm_bg for marker in ("窗内可追加、暂停或替代", "窗外长期要求不变")
        ) and "不废止" not in longterm_bg
        longterm_mutated = json.loads(canonical_json(longterm_spec))
        longterm_mutated["background"] = (
            f"{longterm_spec['setting']['name']}（测试维护）：同一用户只沿用一份长期要求；"
            f"作用域是{longterm_spec['setting']['scope']}；临时通知只在注明的时间窗内追加要求，不废止长期要求。"
        )
        longterm_problems = validate_spec(longterm_mutated)
        longterm_ok = longterm_bg_ok and any(
            ("background_missing_marker" in problem or "background_conflicts_suspend" in problem)
            for problem in longterm_problems
        )
        results.append(
            {
                "check": "longterm_suspend_background",
                "ok": longterm_ok,
                "problems": bounded(longterm_problems),
            }
        )

        results.append(
            {
                "check": "v2_counterexamples_must_fail",
                "ok": all((scope_restricted_ok, disjoint_soft_ok, list_algebra_ok, premise_ok, longterm_ok)),
                "detail": {
                    "scope_restricted": scope_restricted_ok,
                    "disjoint_soft": disjoint_soft_ok,
                    "list_algebra": list_algebra_ok,
                    "premise": premise_ok,
                    "longterm_background": longterm_ok,
                },
            }
        )

        temp_specs = [
            build_family(index=0, setting=SETTINGS[0], mechanism="numeric_threshold", split="train"),
            build_family(index=1, setting=SETTINGS[1], mechanism="scope_inclusion", split="dev"),
        ]
        guard_spec = build_family(
            index=2, setting=SETTINGS[2], mechanism="longterm_one_off", split="calibration"
        )
        fixture_problems = validate_spec_set(temp_specs, full=False)
        results.append({"check": "fixture_specs", "ok": not fixture_problems, "problems": bounded(fixture_problems)})

        specs_path = work / "specs.jsonl"
        write_jsonl_atomic(specs_path, temp_specs)
        source_path = work / "source_rows.jsonl"
        _write_fixture_source(source_path)

        authors_dir = work / "authors"
        report = author_specs(
            specs=temp_specs,
            out_dir=authors_dir,
            channel_factory=lambda spec: _StubChannel(spec),
            channel_meta={
                "model": "stub-model",
                "model_safe": "stub-model",
                "requested_num_predict": NUM_PREDICT_REQUESTED,
                "override_mode": "test",
                "channel_default_num_predict": None,
                "effective_num_predict": None,
                "budget_risk": False,
            },
            call_kwargs={},
        )
        results.append(
            {
                "check": "author_path_with_stub",
                "ok": report["ok"] and report["counts"]["authored"] == 2,
                "counts": report["counts"],
            }
        )
        records = sorted(authors_dir.glob("af-*.json"))
        results.append(
            {
                "check": "author_record_shape",
                "ok": len(records) == 2
                and all(
                    set(read_json(path)) >= {"model", "prompt", "spec", "response", "usage", "spec_sha256"}
                    for path in records
                ),
                "records": len(records),
            }
        )

        stub_items, _stub_mode = extract_json_array(_StubChannel(temp_specs[0]).complete("")["text"])
        with_trigger = json.loads(canonical_json(stub_items))
        with_trigger[0]["old"]["trigger"] = "工作日"
        trigger_problems = check_author_items(with_trigger, temp_specs[0])
        pair_probe = build_pair(stub_items[0], temp_specs[0])
        results.append(
            {
                "check": "optional_slots_rejected",
                "ok": any("entry_extra_keys" in problem for problem in trigger_problems)
                and set(pair_probe["old"]) == {"kind", "text"}
                and set(pair_probe["new"]) == {"kind", "text"}
                and pair_probe["old"].get("kind") == "memory"
                and pair_probe["new"].get("kind") == "memory",
                "problems": bounded(trigger_problems),
            }
        )

        probe_items, _probe_mode = extract_json_array(_StubChannel(temp_specs[0]).complete("")["text"])
        original_old = "  原样保留的旧句文本  "
        original_new = "原样保留的新句文本"
        string_items = json.loads(canonical_json(probe_items))
        string_items[0]["old"] = original_old
        string_items[0]["new"] = original_new
        normalized_items, normalized_mode = extract_json_array(
            json.dumps(string_items, ensure_ascii=False)
        )
        normalized_ok = (
            normalized_mode.endswith("+string_entries_normalized")
            and normalized_items[0]["old"] == {"text": original_old}
            and normalized_items[0]["new"] == {"text": original_new}
            and not check_author_items(normalized_items, temp_specs[0])
        )
        results.append(
            {
                "check": "string_entries_normalized",
                "ok": normalized_ok,
                "mode": normalized_mode,
            }
        )

        blank_items = json.loads(canonical_json(probe_items))
        blank_items[0]["old"] = "   "
        blank_parsed, blank_mode = extract_json_array(json.dumps(blank_items, ensure_ascii=False))
        blank_problems = check_author_items(blank_parsed, temp_specs[0])

        number_items = json.loads(canonical_json(probe_items))
        number_items[0]["old"] = 123
        number_parsed, number_mode = extract_json_array(json.dumps(number_items, ensure_ascii=False))
        number_problems = check_author_items(number_parsed, temp_specs[0])

        array_items = json.loads(canonical_json(probe_items))
        array_items[0]["new"] = ["文本"]
        array_parsed, array_mode = extract_json_array(json.dumps(array_items, ensure_ascii=False))
        array_problems = check_author_items(array_parsed, temp_specs[0])

        rejected_ok = (
            "string_entries_normalized" not in blank_mode
            and "string_entries_normalized" not in number_mode
            and "string_entries_normalized" not in array_mode
            and any("entry_not_object" in problem for problem in blank_problems)
            and any("entry_not_object" in problem for problem in number_problems)
            and any("entry_not_object" in problem for problem in array_problems)
        )
        results.append(
            {
                "check": "string_entries_rejections",
                "ok": rejected_ok,
                "modes": [blank_mode, number_mode, array_mode],
                "problems": bounded(blank_problems + number_problems + array_problems),
            }
        )

        source_item = {
            "pair_id": "af-000-p00",
            "fine_label": "equivalent",
            "old": {"text": "已有对象形式的旧句"},
            "new": "待归一化的新句",
            "background": temp_specs[0]["background"],
        }
        normalized_probe, normalized_count = _normalize_string_entries([source_item])
        objects_ok = (
            normalized_count == 1
            and source_item["new"] == "待归一化的新句"
            and normalized_probe[0] is not source_item
            and normalized_probe[0]["old"] == {"text": "已有对象形式的旧句"}
            and normalized_probe[0]["new"] == {"text": "待归一化的新句"}
        )
        results.append(
            {
                "check": "string_entries_objects_untouched",
                "ok": objects_ok,
                "normalized": normalized_count,
            }
        )

        string_authors = work / "authors-strings"
        string_report = author_specs(
            specs=[temp_specs[0]],
            out_dir=string_authors,
            channel_factory=lambda spec: _StringEntryChannel(spec),
            channel_meta={"model": "stub-model", "model_safe": "stub-model"},
            call_kwargs={},
        )
        string_record = read_json(string_authors / f"{temp_specs[0]['family_id']}.json")
        string_out = work / "string-data"
        string_asm = assemble_dataset(
            specs=[temp_specs[0]],
            source_rows_path=source_path,
            author_dirs=[string_authors],
            out_dir=string_out,
            quarantine={},
        )
        raw_items = json.loads(string_record["response"])
        processed_string_rows = [
            json.loads(line)
            for line in (string_out / "processed" / "train.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        processed_new = {
            row["id"]: row for row in processed_string_rows if row["source"] == ROW_SOURCE_NEW
        }
        first_id = raw_items[0]["pair_id"]
        both_paths_ok = (
            string_report["ok"]
            and "string_entries_normalized" in (string_record.get("extraction") or "")
            and string_asm.get("ok") is True
            and processed_new[first_id]["pair"]["old"]["text"] == raw_items[0]["old"]
            and processed_new[first_id]["pair"]["new"]["text"] == raw_items[0]["new"]
        )
        results.append(
            {
                "check": "string_entries_both_paths",
                "ok": both_paths_ok,
                "record_extraction": string_record.get("extraction"),
                "assembled": string_asm.get("ok"),
            }
        )

        neg_bg = next(spec["background"] for spec in specs if spec["mechanism"] == "negation_exception")
        results.append(
            {
                "check": "true_subset_background",
                "ok": "名单甲真包含名单乙" in neg_bg
                and "名单甲真包含名单乙" in _BACKGROUND_REQUIREMENTS["negation_exception"],
                "background": neg_bg,
            }
        )

        af009 = next(spec for spec in specs if spec["family_id"] == "af-009")
        gate_items, _gate_mode = extract_json_array(_StubChannel(af009).complete("")["text"])
        variant_by_id = {item["pair_id"]: item["variant"] for item in af009["items"]}
        label_by_id = {item["pair_id"]: item["fine_label"] for item in af009["items"]}
        cross_id = next(pid for pid, value in variant_by_id.items() if value == "crossing_exception_scopes")
        excl_id = next(pid for pid, value in variant_by_id.items() if value == "exclusive_exemption_list")
        disjoint_ids = [
            pid for pid, value in label_by_id.items() if value == "disjoint" and pid != excl_id
        ]

        cross_bad = json.loads(canonical_json(gate_items))
        cross_target = next(item for item in cross_bad if item["pair_id"] == cross_id)
        cross_target["old"] = {"text": "旧句的例外只有名单内的条目免于该要求"}
        cross_target["new"] = {"text": "新句的例外只适用于需要引用的事项"}
        cross_problems = check_author_items(cross_bad, af009)
        results.append(
            {
                "check": "author_text_crossing_gate",
                "ok": cross_id == "af-009-p03"
                and any("author_text_only_exempt" in problem for problem in cross_problems),
                "pair_id": cross_id,
                "problems": bounded(cross_problems),
            }
        )

        excl_bad = json.loads(canonical_json(gate_items))
        excl_target = next(item for item in excl_bad if item["pair_id"] == excl_id)
        excl_target["old"] = {"text": "旧句说豁免名单就是名单戊"}
        excl_target["new"] = {"text": "新句说豁免名单就是名单己"}
        excl_problems = check_author_items(excl_bad, af009)
        results.append(
            {
                "check": "author_text_exemption_gate",
                "ok": excl_id == "af-009-p07"
                and any("author_text_unique_list" in problem for problem in excl_problems),
                "pair_id": excl_id,
                "problems": bounded(excl_problems),
            }
        )

        disjoint_id = disjoint_ids[0]
        disjoint_bad = json.loads(canonical_json(gate_items))
        disjoint_target = next(item for item in disjoint_bad if item["pair_id"] == disjoint_id)
        disjoint_target["new"] = {"text": "新句不要求名单甲内事项必须登记"}
        disjoint_problems = check_author_items(disjoint_bad, af009)
        results.append(
            {
                "check": "author_text_disjoint_gate",
                "ok": any("author_text_disjoint_soft" in problem for problem in disjoint_problems),
                "pair_id": disjoint_id,
                "problems": bounded(disjoint_problems),
            }
        )

        legal_items = json.loads(canonical_json(gate_items))
        legal_cross = next(item for item in legal_items if item["pair_id"] == cross_id)
        legal_cross["old"] = {"text": "旧句的例外只适用于名单内的高优先级事项"}
        legal_cross["new"] = {"text": "新句的例外只对名单内需要引用的事项生效"}
        legal_excl = next(item for item in legal_items if item["pair_id"] == excl_id)
        legal_excl["old"] = {"text": "旧句说豁免名单唯一，就是名单戊"}
        legal_excl["new"] = {"text": "新句说豁免名单只有一个，就是名单己"}
        legal_disjoint = next(item for item in legal_items if item["pair_id"] == disjoint_id)
        legal_disjoint["old"] = {"text": "旧句要求名单甲内事项必须登记"}
        legal_disjoint["new"] = {"text": "新句要求名单甲内事项不得登记"}
        legal_problems = check_author_items(legal_items, af009)
        results.append(
            {
                "check": "author_text_legal_passes",
                "ok": not legal_problems,
                "problems": bounded(legal_problems),
            }
        )

        out_dir = work / "data"
        assembled = assemble_dataset(
            specs=temp_specs,
            source_rows_path=source_path,
            author_dirs=[authors_dir],
            out_dir=out_dir,
            quarantine={},
        )
        results.append(
            {
                "check": "assemble_fixture",
                "ok": assembled.get("ok") is True
                and assembled["counts"]["by_split"] == {"train": 18, "dev": 16, "calibration": 0, "heldout": 0},
                "counts": assembled.get("counts"),
                "problems": assembled.get("problems"),
            }
        )

        assembled_train = [
            json.loads(line)
            for line in (out_dir / "processed" / "train.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        new_kind = next(row for row in assembled_train if row["source"] == ROW_SOURCE_NEW)["pair"]
        legacy_kind = next(row for row in assembled_train if row["source"] == ROW_SOURCE_LEGACY)["pair"]
        results.append(
            {
                "check": "kind_memory_new_legacy_proposition",
                "ok": new_kind["old"].get("kind") == "memory"
                and new_kind["new"].get("kind") == "memory"
                and legacy_kind["old"].get("kind") == "proposition"
                and legacy_kind["new"].get("kind") == "proposition",
                "new_kinds": (new_kind["old"].get("kind"), new_kind["new"].get("kind")),
                "legacy_kinds": (legacy_kind["old"].get("kind"), legacy_kind["new"].get("kind")),
            }
        )

        clean = _audit_fixture(out_dir / "processed", temp_specs, source_path)
        results.append(
            {
                "check": "audit_clean_passes",
                "ok": clean["ok"] is True,
                "failures": clean["failures"],
            }
        )

        os_dir = out_dir / "processed"
        train_rows = [json.loads(line) for line in (os_dir / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        dev_rows = [json.loads(line) for line in (os_dir / "dev.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

        leak_dir = work / "leak"
        shutil.copytree(out_dir / "processed", leak_dir)
        leak_rows = [dict(row) for row in train_rows]
        leak_rows[0]["pair"] = dev_rows[0]["pair"]
        leak_rows[0]["visible_text"] = dev_rows[0]["visible_text"]
        write_jsonl_atomic(leak_dir / "train.jsonl", leak_rows)
        leak = _audit_fixture(leak_dir, temp_specs, source_path)
        results.append(
            {
                "check": "leak_must_fail",
                "ok": leak["ok"] is False
                and "cross_split_visible_exact" in leak["failures"],
                "failures": leak["failures"],
            }
        )

        label_dir = work / "label"
        shutil.copytree(out_dir / "processed", label_dir)
        label_rows = [dict(row) for row in train_rows]
        label_rows[1]["pair"] = label_rows[0]["pair"]
        label_rows[1]["visible_text"] = label_rows[0]["visible_text"]
        label_rows[1]["label"] = "disjoint" if label_rows[0]["label"] != "disjoint" else "equivalent"
        label_rows[1]["coarse_label"] = mapping[label_rows[1]["label"]]
        write_jsonl_atomic(label_dir / "train.jsonl", label_rows)
        mislabeled = _audit_fixture(label_dir, temp_specs, source_path)
        results.append(
            {
                "check": "same_input_different_label_must_fail",
                "ok": mislabeled["ok"] is False
                and "same_input_different_label" in mislabeled["failures"],
                "failures": mislabeled["failures"],
            }
        )

        missing_dir = work / "missing"
        shutil.copytree(out_dir / "processed", missing_dir)
        write_jsonl_atomic(missing_dir / "train.jsonl", train_rows[:-1])
        missing = _audit_fixture(missing_dir, temp_specs, source_path)
        results.append(
            {
                "check": "missing_row_must_fail",
                "ok": missing["ok"] is False
                and ("counts" in missing["failures"] or "family_shape" in missing["failures"]),
                "failures": missing["failures"],
            }
        )

        mutated = json.loads(canonical_json(guard_spec))
        target = next(
            item for item in mutated["items"]
            if item["variant"] in {
                "new_subset_by_temporary_exception",
                "new_subset_by_two_temporary_exceptions",
                "new_subset_by_one_off_scope",
            }
        )
        target["fine_label"] = "disjoint"
        guard_problems = validate_spec(mutated)
        results.append(
            {
                "check": "temporary_exception_as_conflict_must_fail",
                "ok": any("temporary_exception_marked_disjoint" in problem for problem in guard_problems),
                "problems": bounded(guard_problems),
                "note": "合同层检查，不是语义检测；文本语义仍需人工/独立盲审。",
            }
        )

        truncated_dir = work / "truncated-authors"
        trunc_report = author_specs(
            specs=[temp_specs[0]],
            out_dir=truncated_dir,
            channel_factory=lambda spec: _StubChannel(spec),
            channel_meta={"model": "stub-model", "model_safe": "stub-model"},
            call_kwargs={},
        )
        results.append(
            {
                "check": "stub_author_repeatable",
                "ok": trunc_report["ok"] and trunc_report["counts"]["authored"] == 1,
                "counts": trunc_report["counts"],
            }
        )

        resume = author_specs(
            specs=[temp_specs[0]],
            out_dir=truncated_dir,
            channel_factory=lambda spec: _StubChannel(spec),
            channel_meta={"model": "stub-model", "model_safe": "stub-model"},
            call_kwargs={},
        )
        results.append(
            {
                "check": "resume_skips_success",
                "ok": resume["ok"] and resume["counts"]["skipped_success"] == 1 and resume["counts"]["authored"] == 0,
                "counts": resume["counts"],
            }
        )

        stale_dir = work / "stale-authors"
        stale_dir.mkdir(parents=True, exist_ok=True)
        other_spec = json.loads(canonical_json(temp_specs[0]))
        other_spec["background"] = other_spec["background"] + "（另一版）"
        write_json_atomic(
            stale_dir / f"{other_spec['family_id']}.json",
            {
                "schema": "relation-author-v1",
                "family_id": other_spec["family_id"],
                "spec_sha256": spec_sha256(other_spec),
                "status": "success",
                "response": "[]",
                "model": "stub-model",
            },
        )
        stale_report = author_specs(
            specs=[temp_specs[0]],
            out_dir=stale_dir,
            channel_factory=lambda spec: _StubChannel(spec),
            channel_meta={"model": "stub-model", "model_safe": "stub-model"},
            call_kwargs={},
        )
        results.append(
            {
                "check": "stale_spec_not_overwritten",
                "ok": stale_report["ok"] is False and len(stale_report["stale_conflicts"]) == 1,
                "stale_conflicts": len(stale_report["stale_conflicts"]),
            }
        )

        failures = [entry["check"] for entry in results if not entry["ok"]]
        return {
            "ok": not failures,
            "command": "selftest",
            "temp_dir": str(work),
            "checks": results,
            "failures": failures,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


                                               


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="原子关系数据构造与审计 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    specs = sub.add_parser("specs", help="生成 70 个 family 规格")
    specs.add_argument("--out", default=None, help="输出目录（默认 EXP/data/specs）")
    specs.add_argument("--dry-run", action="store_true", help="只校验不写文件")

    author = sub.add_parser("author-ollama", help="调用本地 Ollama 按规格生成句对")
    author.add_argument("--model", default=DEFAULT_MODEL)
    author.add_argument("--specs", default=str(DEFAULT_SPECS))
    author.add_argument("--out", default=None, help="作者目录（默认 EXP/data/authors/<model-safe>）")
    author.add_argument("--split", action="append", choices=SPLITS, help="可重复；默认全部")
    author.add_argument("--family", action="append", help="可重复；只跑指定 family")
    author.add_argument("--limit", type=int, default=None)
    author.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    author.add_argument("--sleep", type=float, default=0.0)
    author.add_argument("--num-predict", type=int, default=NUM_PREDICT_REQUESTED)
    author.add_argument("--channel-module", default=None, help="默认 EXP/model_channel.py")
    author.add_argument("--dry-run", action="store_true", help="只显示计划，不导入通道、不调用")

    assemble = sub.add_parser("assemble", help="校验作者输出并组装 processed/*.jsonl")
    assemble.add_argument("--authors", action="append", required=True, help="可重复；多个作者目录")
    assemble.add_argument("--specs", default=str(DEFAULT_SPECS))
    assemble.add_argument("--source-rows", default=str(DEFAULT_SOURCE_ROWS))
    assemble.add_argument("--quarantine", default=str(DEFAULT_QUARANTINE))
    assemble.add_argument("--out", default=None, help="默认 EXP/data")
    assemble.add_argument("--dry-run", action="store_true")

    audit = sub.add_parser("audit", help="审计 processed/*.jsonl")
    audit.add_argument("--processed", default=str(DEFAULT_PROCESSED))
    audit.add_argument("--specs", default=str(DEFAULT_SPECS))
    audit.add_argument("--source-rows", default=str(DEFAULT_SOURCE_ROWS))
    audit.add_argument("--quarantine", default=str(DEFAULT_QUARANTINE))
    audit.add_argument("--mode", choices=("full", "fixture"), default="full")

    sub.add_parser("selftest", help="在系统临时目录跑合同自检（不访问模型）")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "specs":
            report = cmd_specs(args)
        elif args.command == "author-ollama":
            report = cmd_author_ollama(args)
        elif args.command == "assemble":
            report = cmd_assemble(args)
        elif args.command == "audit":
            report = cmd_audit(args)
        elif args.command == "selftest":
            report = run_selftest()
        else:
            raise DataError("unknown_command", args.command)
    except DataError as exc:
        report = {"ok": False, "command": args.command, "error": exc.code, "detail": exc.detail}
        emit(report)
        return 2
    except Exception as exc:                  
        report = {"ok": False, "command": args.command, "error": type(exc).__name__, "detail": str(exc)[:800]}
        emit(report)
        return 1
    emit(report)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
