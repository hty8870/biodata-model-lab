"""DomainPack：领域包契约（骨肉分离架构的「肉」的声明形状）。

一个领域包 = 一个 Python 包，暴露 `get_pack() -> DomainPack`。
骨架只经 `domain.registry` 拿当前包；除 registry 外，骨架通用机制文件
不得 import 任何具体领域包（防火墙由 `tests/test_domain_firewall.py` 机械钉死）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .descriptors import DimensionDescriptor


@dataclass(frozen=True)
class DomainBranding:
    """品牌与产品文案清单（单一真源在领域包；骨架各界面只读这里）。"""

    product_name: str                                    
    page_title: str                               
    dataset_page_title: str                             
    mcp_server_name: str                            
    download_dir_prefix: str                           
    corpus_blurb_zh: str = ""                               


@dataclass(frozen=True)
class DomainPack:
    """一份可插拔领域的全部声明。"""

    domain_id: str
    name_zh: str
    tagline_zh: str
    dimensions: tuple[DimensionDescriptor, ...]
    catalog: dict[str, list[dict]]
    dim_labels_zh: dict[str, str]
    branding: DomainBranding
    prompts_dir: str = ""                                                       
    base_source: str = ""                                                         
    facet_order: tuple[str, ...] = ()                                                   
    explain_order: tuple[str, ...] = ()                   
    raw_data_labels: tuple[str, str] = ("有原始数据", "无原始数据")                       
    dim_nouns: tuple[tuple[str, str], ...] = ()                                    
    completeness_fields: tuple[str, ...] = ()                                       
    vocab: Any = None                                                                      
    corpus_loader: Any = None                                   

    def __post_init__(self) -> None:
        if not self.domain_id or not self.domain_id.strip():
            raise ValueError("DomainPack.domain_id 不能为空")
        if not self.dimensions:
            raise ValueError("DomainPack.dimensions 不能为空")
        names = [d.name for d in self.dimensions]
        if len(set(names)) != len(names):
            raise ValueError(f"维度名重复: {names}")
        extra = set(self.catalog) - set(names)
        if extra:
            raise ValueError(f"catalog 含未声明维度的键: {sorted(extra)}")
        for dim, label in self.dim_labels_zh.items():
            if dim not in set(names):
                raise ValueError(f"dim_labels_zh 含未声明维度: {dim}")

    def dimension_names(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.dimensions)

    def descriptor_for(self, name: str) -> DimensionDescriptor | None:
        for d in self.dimensions:
            if d.name == name:
                return d
        return None
