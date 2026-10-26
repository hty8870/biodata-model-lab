                       
"""兼容转发：OOD 排序对比的唯一实现位于 `scripts/ood_duel.py`。

旧路径的导入与命令行入口保留给既有内部调用；本文件不复制实现、不二次定义函数，
导入后与 `scripts.ood_duel` 是同一个模块对象。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.ood_duel as _impl              

if __name__ == "__main__":
    raise SystemExit(_impl.main())

sys.modules[__name__] = _impl
