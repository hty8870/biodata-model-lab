                       
"""retrieval 家族「同一原因只提示一次」stderr 告警的**单一真源**（零本包依赖叶子）。

原住 `vector_recall`（`_WARNED` 集合 + `_warn_once_prefixed`），rerank/recall_api/
vector_index 模块级取私有名，构成家族软环的粘合剂；2026-10-01 仓库整理批抽为本叶子。
各消费模块保留 `_WARNED`/`_warn_once_prefixed` 私有别名——共享的仍是这同一个集合对象，
测试经 `_WARNED.clear()` 复位的语义不变。
"""
from __future__ import annotations

import sys

WARNED: set[str] = set()


def warn_once_prefixed(prefix: str, key: str, message: str) -> None:
    """同一原因只在 stderr 提示一次；绝不抛异常、绝不打断请求。prefix 区分来源模块。"""
    if key not in WARNED:
        WARNED.add(key)
        print(f"{prefix} {message}", file=sys.stderr)
