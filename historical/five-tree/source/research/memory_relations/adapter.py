"""pair→set 关系适配器：把逐对评分的轻量 scorer 接成 M.assess_relations 的本地 predictor。

研究专用组件（无生产接线）。逐对操作判定的**唯一真源是 metrics.pair_verdict**
（`pair_verdict(probs, request, threshold) -> relation`：等分 allow/block 不默认放行、
无 trigger 的 contextual 与低置信 max-prob 判 uncertain 等规则都在那里）；本模块只做四件事：

1. pair 构造：候选×单条既有规则（existing 恒 ≤1）。无规则时用空哨兵单条请求（existing=[]）
   仍交 pair_verdict 判断，绝不默认 complement；规则留满、绝不 top-k。
2. 概率矩阵边界校验（形状/有限/取值/行和），不合法 → 整体 uncertain(model_uncertain)。
3. 逐对结论 → 集合级五字段判定的显式聚合（见下）。
4. 引用只用输入真实 id：related_ids 取所选类别的配对规则 id；evidence_refs 取输入
   evidence 全集 id——这只是**宽泛引用（输入证据集合）**，不构成对每条关系的实证。
   权限 scope/implicit/readonly 闸不在此复制，交 M._decision 复核。

聚合口径（显式研究策略，写死在代码里而不是数据里）：
1. 逐对判定（含 pair_verdict 的低置信门；阈值必须显式 [0, 1] 数值——None 未校准在工厂层
   直接拒绝，既不回落 0 也不透传）。绝不对全对概率求和后取最大——那会让大量 complement
   冲淡单个 conflict。
2. 任一高置信 conflict 都保留：兼容的、未定的其它对各不冲淡，也不阻断它。
3. 其余对里 {supersede, temporary, contextual} 出现两类及以上 → uncertain
   （error_code=ambiguous_relation）：条件/时间/替换三种解释无法用单一类别诚实概括。
4. 存在未定对且没有高置信 conflict → uncertain（error_code=model_uncertain）：
   部分覆盖不许冒充完整判断。
5. 全部已定且无冲突 → 按生产 _MERGE_PRIORITY 的严重度序取最高类。
6. 概率矩阵不合法或 pair_verdict 返回未知类别 → 整体 uncertain（model_uncertain）；
   不回退 API、不改道。

**与现行 LLM set 语义非等价（必须随结果披露）**：
- 逐对分解 + 机械聚合只是研究口径的近似；不得用整集特征矩阵（whole-set matrix）冒充
  set 语义，也不能当作生产 LLM 单次 set 判断的等价物或批准依据。
- 跨批合并沿用生产 _merge_decisions：任一批次得 uncertain/错误码则整体 uncertain——
  跨批 uncertain **会覆盖**另一批次的 conflict，与批内「conflict 优先」不等价；非不确定批
  按 _MERGE_PRIORITY 取关系并**并集**各批引用——跨批的 related_ids 可能包含非所选类别的
  批次引用（如实保留在报告里，不擅自剪裁）。
- 批内混合诚实检查（条件/时间/替换 ≥2 → uncertain）**不跨批生效**：批间混合不会被它检出，
  按生产严重度序合并。
本模块不发起网络或 LLM 调用、不读环境。
"""
from __future__ import annotations

import numpy as np

from dataset_recommender.agent.memory_relations import _MERGE_PRIORITY

__all__ = ["make_predictor"]

                                       
_EXPLANATORY = frozenset({"supersede", "temporary", "contextual"})
                                             
_ROW_SUM_TOLERANCE = 1e-3


def _metrics_module():
    """操作判定唯一真源（pair_verdict/类序/允许阻塞划分）的惰性入口；测试可换替身。"""
    from . import metrics

    return metrics


def _threshold(value) -> float:
    """阈值必须是显式 [0, 1] 数值；None（未校准）在工厂层直接拒绝，绝不静默回落到 0。"""
    if value is None:
        raise ValueError("threshold=None（未校准）不允许：运行判定必须显式给出 [0, 1] 数值阈值；"
                         "诊断/试验请显式传数值（如 0），不要用 None 暗代")
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise ValueError("threshold 必须是 [0, 1] 内的数字（None 未校准不接受）")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError("threshold 必须是 [0, 1] 内的数字（None 未校准不接受）")
    return number


def _matrix(result, rows: int, columns: int):
    try:
        matrix = np.asarray(result, dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (rows, columns):
        return None
    if not np.isfinite(matrix).all() or (matrix < 0.0).any() or (matrix > 1.0).any():
        return None
    if not bool(np.allclose(matrix.sum(axis=1), 1.0, atol=_ROW_SUM_TOLERANCE)):
        return None
    return matrix


def _pair_requests(request: dict) -> "tuple[list[dict], list[str | None]]":
    """候选×每条既有规则的逐对请求；无规则时返回空哨兵单条请求（existing=[]）。

    请求与规则对象按投影原样复用（只读）；规则 id 在调用 scorer 前先取好，
    scorer/verdict 即便改动请求对象也不会影响判定引用。
    """
    rules = request["existing"]
    if not rules:
        return [request], [None]
    return ([{"candidate": request["candidate"], "existing": [rule],
              "evidence": request["evidence"], "context": request["context"]} for rule in rules],
            [rule["id"] for rule in rules])


def _tally(outcomes: "list[tuple[str | None, str]]") -> str:
    counts: dict[str, int] = {}
    for _, relation in outcomes:
        counts[relation] = counts.get(relation, 0) + 1
    order = [name for name in _MERGE_PRIORITY if counts.get(name)]
    if counts.get("uncertain"):
        order.append("uncertain")
    return "、".join(f"{name}×{counts[name]}" for name in order)


def _decision(relation: str, related: list[str], refs: list[str], reason: str) -> dict:
    return {"relation": relation, "related_ids": related, "reason": reason,
            "evidence_refs": list(refs), "error_code": None}


def _uncertain(code: str, reason: str) -> dict:
    return {"relation": "uncertain", "related_ids": [], "reason": reason,
            "evidence_refs": [], "error_code": code}


def _aggregate(outcomes: "list[tuple[str | None, str]]", refs: list[str]) -> dict:
    """逐对结论 → 集合级五字段判定；口径见模块文档 1–5（矩阵合法性在调用前已核）。"""
    total = len(outcomes)
    tally = _tally(outcomes)
    decided = [(rule_id, relation) for rule_id, relation in outcomes if relation != "uncertain"]
    if any(relation == "conflict" for _, relation in decided):
        related = [rule_id for rule_id, relation in decided if relation == "conflict" and rule_id]
        reason = (f"本地模型逐对评分 {total} 对（{tally}）：优先保留高置信 conflict，其它对不冲淡；"
                  f"仅模型判断，未经独立核验，仍需人工复核；"
                  f"evidence_refs 为输入证据集合的宽泛引用，非逐条关系实证。")
        return _decision("conflict", related, refs, reason)
    if len({relation for _, relation in decided if relation in _EXPLANATORY}) >= 2:
        reason = (f"本地模型逐对评分 {total} 对（{tally}）：条件/时间/替换类关系混杂，"
                  f"无法诚实归为单一类别；仅模型判断，未经独立核验。")
        return _uncertain("ambiguous_relation", reason)
    if len(decided) != total:
        reason = (f"本地模型逐对评分 {total} 对（{tally}）：存在未取得可核验判断的规则，"
                  f"部分覆盖不许冒充完整判断；仅模型判断，未经独立核验。")
        return _uncertain("model_uncertain", reason)
    for name in _MERGE_PRIORITY:
        if any(relation == name for _, relation in decided):
            related = [rule_id for rule_id, relation in decided if relation == name and rule_id]
            reason = (f"本地模型逐对评分 {total} 对（{tally}）：按严重度优先取 {name}；"
                      f"仅模型判断，未经独立核验，仍需人工复核；"
                      f"evidence_refs 为输入证据集合的宽泛引用，非逐条关系实证。")
            return _decision(name, related, refs, reason)
                                                     
    return _uncertain("model_uncertain",
                      f"本地模型逐对评分 {total} 对（{tally}）：未取得可核验判断；仅模型判断，未经独立核验。")


def make_predictor(scorer, threshold=None):
    """把逐对 scorer 包成 M.assess_relations(predictor=...) 可用的本地判定器。

    返回的 callable 接收 M 的投影请求（candidate/existing/evidence/context），返回恰好五字段的
    原始判定；操作判定逐对调用 metrics.pair_verdict。threshold 必须显式给出 [0, 1] 数值——
    未校准（None，含省略默认值）在工厂层直接拒绝，绝不隐藏回落 0；诊断/试验请显式传数值（如 0）。
    概率矩阵不合法一律映射 uncertain(model_uncertain)，绝不回退 API。scorer 需提供
    predict_proba(requests)->(n, 7)，且 classes 与 metrics.CLASSES 的 spec 固定类序逐位一致
    （pair 请求就是「候选×单条规则」的同一投影形状）。返回对象的 stats（calls/pairs）只供评测
    统计，不参与判定。
    """
    metrics = _metrics_module()
    verdict = getattr(metrics, "pair_verdict", None)
    if not callable(verdict):
        raise ValueError("metrics.pair_verdict 未就绪：操作判定只允许这一个真源")
    classes = tuple(metrics.CLASSES)
    declared = getattr(scorer, "classes", None)
    if not (isinstance(declared, (list, tuple)) and tuple(declared) == classes):
        raise ValueError("scorer.classes 必须与 metrics.CLASSES 的 spec 固定类序逐位一致")
    limit = _threshold(threshold)
    score = getattr(scorer, "predict_proba", None)
    if not callable(score):
        raise ValueError("scorer 必须提供可调用的 predict_proba")
    stats = {"calls": 0, "pairs": 0}

    def predictor(request: dict) -> dict:
        stats["calls"] += 1
        pairs, rule_ids = _pair_requests(request)
        refs = [item["id"] for item in request["evidence"]]
        stats["pairs"] += len(pairs)
        matrix = _matrix(score(pairs), len(pairs), len(classes))
        if matrix is None:
            return _uncertain("model_uncertain",
                              f"本地模型输出概率矩阵不合法（须为 {len(pairs)}×{len(classes)} 的有限概率），"
                              f"未取得可核验判断；仅模型输出，未经独立核验。")
        outcomes = []
        for rule_id, pair, probs in zip(rule_ids, pairs, matrix):
            relation = verdict(probs, pair, limit)
            if relation not in classes:
                return _uncertain("model_uncertain",
                                  "pair_verdict 返回未知类别，未取得可核验判断；仅模型输出，未经独立核验。")
            outcomes.append((rule_id, relation))
        return _aggregate(outcomes, refs)

    predictor.stats = stats
    return predictor
