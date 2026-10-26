"""设计期风险算术校验（不是实验数据结果）。

用途：在开训前复算本轮预注册风险目标（conditional risk <= 1%、coverage >= 50%）下的
可行 coverage 上界，以及「零危险 allow 根」支持 1% 风险上界所需的独立审计根下限。

口径与来源：
- 均衡七类集安全 allow 占比 3/7、风险 d/c <= 0.01、c - d <= 3/7：主研究计划与
  advisor 评审原文（research/memory_relations/runs/advisor/out/astra-review.md）。
- 零错误精确二项上界 p_U(n) = 1 - alpha ** (1/n)：advisor 原文给定公式；
  单侧 95% 用 alpha=0.05，双侧 95%（等价单侧 97.5%）用 alpha=0.025，
  九固定模型 Bonferroni 用 alpha=0.05/9。
- 本脚本只做标准库算术，不加载模型、不读写实验数据、不写 repo 任何文件。

自测（RA 分析器纪律）：对每个 alpha 同时用闭式公式与逐点复核两种方法求最小 n，
两者必须一致，否则脚本报错而非静默取一个值。
"""

from __future__ import annotations

import math
from math import ceil, exp, log


def zero_error_bound(n: int, alpha: float) -> float:
    """n 个独立同分布接受根、零危险时的单侧精确二项上界。"""
    return 1.0 - alpha ** (1.0 / n)


def min_n_closed(alpha: float, target: float) -> int:
    """闭式：1 - alpha**(1/n) <= target  <=>  n >= ln(alpha)/ln(1-target)。"""
    return ceil(log(alpha) / log(1.0 - target))


def min_n_scan(alpha: float, target: float, nmax: int = 200000) -> int | None:
    """逐点扫描复核；用 log 形式比较避免浮点边界误判。"""
    limit = log(1.0 - target)
    for n in range(1, nmax + 1):
        if log(alpha) / n >= limit:
            return n
    return None


def cross_check(alpha: float, target: float, label: str) -> int:
    a = min_n_closed(alpha, target)
    b = min_n_scan(alpha, target)
    if a != b:
        raise AssertionError(f"{label}: 闭式 {a} 与扫描 {b} 不一致")
    bound = zero_error_bound(a, alpha)
    if not bound <= target:
        raise AssertionError(f"{label}: n={a} 时上界 {bound} 未达 {target}")
    prev = zero_error_bound(a - 1, alpha) if a > 1 else float("inf")
    return a


def main() -> None:
    target = 0.01
    print("== 零错误精确二项上界所需独立 allow 根数（目标上界 <= 1%）==")
    cases = [
        ("单侧 95%（alpha=0.05）", 0.05),
        ("双侧 95% / 单侧 97.5%（alpha=0.025）", 0.025),
        ("九固定模型联合 Bonferroni（alpha=0.05/9）", 0.05 / 9.0),
    ]
    for label, alpha in cases:
        n = cross_check(alpha, target, label)
        print(f"{label}: n_min={n}; 上界(n_min)={zero_error_bound(n, alpha):.6%}; "
              f"上界(n_min-1)={zero_error_bound(n - 1, alpha):.6%}")

    print()
    print("== 均衡七类集的 coverage 上界（七类均衡、安全 allow 占 3/7）==")
    print(f"零危险上限 3/7 = {3 / 7:.6%}")
    print(f"1% 风险下上限 3/(7*0.99) = {3 / (7 * 0.99):.6%}")
    forced_min_risk = 0.5 - 3.0 / 7.0
    print(f"强行 coverage=50% 的最低风险 d/c = (0.5-3/7)/0.5 = {forced_min_risk / 0.5:.6%}")

    print()
    print("== 40 根总校准再三分（audit 角色）下的零错误上界 ==")
    for n in (40, 14, 13):
        print(f"n={n} 全部接受且零危险: p_U = {zero_error_bound(n, 0.05):.6%}")


if __name__ == "__main__":
    main()
