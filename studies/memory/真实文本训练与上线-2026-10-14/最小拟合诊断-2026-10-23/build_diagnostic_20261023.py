from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

from dataset_recommender.agent.semantic_relations import SEMANTIC_RELATIONS as CLASSES
from research.memory_relations.semantic_data import oracle_certificate
TOPICS = [
    ("同一份文件", ("文件已经公开", "文件尚未公开"), ("文件已经压缩", "文件尚未压缩")),
    ("同一份报告", ("报告已经签名", "报告尚未签名"), ("报告含有图表", "报告不含图表")),
    ("同一条记录", ("记录已经复核", "记录尚未复核"), ("记录带有附件", "记录不带附件")),
    ("同一个数据集", ("数据集含有FASTQ文件", "数据集不含FASTQ文件"), ("数据集含有表达矩阵", "数据集不含表达矩阵")),
    ("同一次会议", ("会议安排了直播", "会议未安排直播"), ("会议安排了录音", "会议未安排录音")),
    ("同一项任务", ("任务已经完成", "任务尚未完成"), ("任务设有截止日期", "任务未设截止日期")),
    ("同一个软件包", ("软件包提供中文界面", "软件包不提供中文界面"), ("软件包提供英文界面", "软件包不提供英文界面")),
    ("同一份问卷", ("问卷允许匿名填写", "问卷不允许匿名填写"), ("问卷包含开放问题", "问卷不包含开放问题")),
    ("同一封邮件", ("邮件包含附件", "邮件不包含附件"), ("邮件已经标记星标", "邮件尚未标记星标")),
    ("同一个实验批次", ("批次包含对照样本", "批次不包含对照样本"), ("批次包含重复样本", "批次不包含重复样本")),
    ("同一个网页", ("网页允许评论", "网页不允许评论"), ("网页支持全文搜索", "网页不支持全文搜索")),
    ("同一张表格", ("表格含有日期列", "表格不含日期列"), ("表格含有编号列", "表格不含编号列")),
    ("同一个项目", ("项目提供演示版", "项目不提供演示版"), ("项目提供教程", "项目不提供教程")),
    ("同一门课程", ("课程安排了考试", "课程未安排考试"), ("课程安排了小组作业", "课程未安排小组作业")),
    ("同一个下载包", ("下载包包含说明书", "下载包不包含说明书"), ("下载包包含示例文件", "下载包不包含示例文件")),
    ("同一个程序", ("程序启用了日志", "程序未启用日志"), ("程序启用了缓存", "程序未启用缓存")),
    ("同一个文件夹", ("文件夹允许读取", "文件夹不允许读取"), ("文件夹允许写入", "文件夹不允许写入")),
    ("同一条书目", ("书目记录了作者", "书目未记录作者"), ("书目记录了年份", "书目未记录年份")),
    ("同一个服务", ("服务开放了注册", "服务未开放注册"), ("服务提供了导出功能", "服务未提供导出功能")),
    ("同一份申请", ("申请附有推荐信", "申请未附推荐信"), ("申请附有成绩单", "申请未附成绩单")),
]
WORLDS = list(itertools.product((False, True), repeat=2))


def satisfying(literals):
    return {i for i, world in enumerate(WORLDS) if all(world[axis] == value for axis, value in literals)}


def relation(old, new):
    model = {"mode": "assertion", "domains": {"world": {"values": list(range(len(WORLDS))), "exhaustive": True}},
             "domain_condition": True,
             "old": ["or", *[["eq", ["var", "world"], int(i)] for i in sorted(old)]],
             "new": ["or", *[["eq", ["var", "world"], int(i)] for i in sorted(new)]]}
    return oracle_certificate(model)["relation"]


def validate_row(row):
    assert [tuple(world) for world in row["worlds"]] == WORLDS
    old, new = (satisfying(row[side + "_literals"]) for side in ("old", "new"))
    assert row["label"] == relation(old, new), row["id"]
    assert row["old_worlds"] == sorted(old) and row["new_worlds"] == sorted(new)
    assert row["witnesses"] == {"intersection": sorted(old & new), "old_only": sorted(old - new), "new_only": sorted(new - old)}


def build():
    rows = []
    for i, (subject, a, b) in enumerate(TOPICS):
        x, y = bool(i % 2), bool((i // 2) % 2)
        text = {(0, True): a[0], (0, False): a[1], (1, True): b[0], (1, False): b[1]}
        aa, bb = (0, x), (1, y)
        cases = [([aa, bb], [bb, aa]), ([aa, bb], [aa]), ([aa], [aa, bb]), ([aa], [(0, not x)]), ([aa], [bb])]
        background = f"讨论{subject}在同一时刻的状态。这里只考虑两项可独立变化的属性：{a[0]}或{a[1]}；{b[0]}或{b[1]}。每项二选一，四种组合都可能；未提及的属性不受限制。"
        for label, (old_literals, new_literals) in zip(CLASSES, cases):
            old, new = satisfying(old_literals), satisfying(new_literals)
            row = {
                "id": f"diagnostic-{i + 1:02d}-{CLASSES.index(label)}",
                "cluster_id": f"scenario-{i + 1:02d}",
                "split": "train_diagnostic_only",
                "label_origin": "finite_boolean_oracle_with_explicit_frame_not_human_gold",
                "label": label,
                "pair": {
                    "old": {"kind": "proposition", "text": "，并且".join(text[v] for v in old_literals) + "。"},
                    "new": {"kind": "proposition", "text": "，并且".join(text[v] for v in new_literals) + "。"},
                    "frame": {"mode": "same_domain", "background": background},
                },
                "old_literals": old_literals, "new_literals": new_literals,
                "worlds": WORLDS, "old_worlds": sorted(old), "new_worlds": sorted(new),
                "witnesses": {"intersection": sorted(old & new), "old_only": sorted(old - new), "new_only": sorted(new - old)},
            }
            rows.append(row)
    return rows


def selftest(rows):
    expected = [({0}, {0}, CLASSES[0]), ({0}, {0, 1}, CLASSES[1]), ({0, 1}, {0}, CLASSES[2]), ({0}, {1}, CLASSES[3]), ({0, 1}, {1, 2}, CLASSES[4])]
    for old, new, label in expected:
        assert relation(old, new) == label
    for row in rows:
        validate_row(row)
    corrupt = dict(rows[0], label=CLASSES[3])
    rejected = False
    try:
        validate_row(corrupt)
    except AssertionError:
        rejected = True
    assert rejected
    assert Counter(row["label"] for row in rows) == dict.fromkeys(CLASSES, 20)
    assert len({json.dumps(row["pair"], ensure_ascii=False, sort_keys=True) for row in rows}) == 100
    return {"oracle_five_positive_cases": 5, "intentional_wrong_label_rejected": rejected, "all_row_certificates": len(rows)}


def main():
    root = Path(__file__).resolve().parent
    rows = build()
    checks = selftest(rows)
    path = root / "data-100-2026-10-23.jsonl"
    assert not path.exists(), "existing dataset must not be overwritten"
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    result = {"schema": "relation-fit-diagnostic-data-v1", "rows": len(rows), "clusters": len(TOPICS), "classes": dict(Counter(row["label"] for row in rows)), "selftest": checks, "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    (root / "data-summary-2026-10-23.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
