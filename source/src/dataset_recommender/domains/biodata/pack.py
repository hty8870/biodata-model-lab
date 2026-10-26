"""biodata 领域包装配：`get_pack()` 返回 DomainPack。

biodata 是本项目的默认领域，同时充当骨架能力的测试集（冻结评测 784 条基准
驱动同一条通用管线）。品牌/维度/词表/提示词目录等全部领域数据在这里声明，
骨架各界面只经 `domain.registry` 读取。
"""
from __future__ import annotations

from ...domain.pack import DomainBranding, DomainPack
from . import vocabulary as _vocab
from .dimensions import DIMENSIONS


def get_pack() -> DomainPack:
    return DomainPack(
        domain_id="biodata",
        name_zh="BioData Agent",
        tagline_zh="公开单细胞数据集的中文检索与复用助手",
        dimensions=DIMENSIONS,
        catalog=_vocab.CATALOG,
        dim_labels_zh={d.name: d.label_zh for d in DIMENSIONS},                       
        branding=DomainBranding(
            product_name="BioData Agent",
            page_title="BioData Agent · 生物数据集检索",
            dataset_page_title="数据集介绍 · BioData Agent",
            mcp_server_name="biodata",
            download_dir_prefix="BioData数据",
            corpus_blurb_zh="",
        ),
        prompts_dir="prompts/domains/biodata",                                                             
        base_source="10x Genomics",
        facet_order=("source", "species", "tissue", "disease", "modality", "platform", "assay",
                     "has_raw_data", "year"),
        explain_order=("species", "tissue", "disease", "modality", "platform", "assay"),
        raw_data_labels=("有 FASTQ", "无 FASTQ"),
        dim_nouns=(
            ("实验技术", "assay"), ("测序技术", "assay"), ("原始数据", "has_raw_data"),
            ("发表时间", "date"), ("发表年份", "date"), ("数据模态", "modality"),
            ("物种", "species"), ("生物", "species"), ("组织", "tissue"), ("部位", "tissue"),
            ("疾病", "disease"), ("病种", "disease"), ("平台", "platform"), ("技术", "assay"),
            ("模态", "modality"), ("fastq", "has_raw_data"), ("时间", "date"), ("年份", "date"),
        ),
        completeness_fields=("species", "tissue", "disease", "chemistry"),
        vocab=_vocab,
    )
