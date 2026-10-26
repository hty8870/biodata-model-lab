"""共享分类/概率/簇自助数学锚点，以及spec.json约定的七类操作统计。

classification_summary与macro_f1共用混淆/F1内核；失败预测-1只记gold的FN。
probability_summary单列缺失覆盖与零gold概率的正无穷NLL；不epsilon裁剪。
纯语义consumer只消费这些数学接口，不调用下面的七类evaluate/操作门。

- 唯一决策门 pair_verdict（adapter 与本模块共用同一实现）：argmax 最大概率 ≥ τ 才采纳；
  平局同时含允许类与阻塞类 → uncertain（不得静默 allow）；同侧平局按 spec 类序取先；
  候选无 trigger 时 contextual → uncertain；低于 τ → uncertain（low-conf）。
- 分类指标（宏 F1/逐类/混淆/ECE/Brier）与阈值无关，继续全量报告。
- 操作统计以 pair_verdict 结果为准：放行 = verdict ∈ 允许类；误阻塞 = 真实允许类被判阻塞。
  分母明确：conditional risk = 危险放行 / 放行决策；dangerous_allow_rate = 危险放行 /
  真实阻塞；false_block_rate = 误阻塞 / 真实允许；总份额另列（/n）；零分母记 null。
  可行点判据用 conditional risk ≤ target 且覆盖率 ≥ min_coverage。
- calibrate_partitioned 的 candidate 阶段仅 prob 根等权拟合温度、policy 预选 primary 选点，
  拒绝 audit；旧 complete 数学接口保留。独立 audit_locked_partition 只验证 study 锁定点，
  每根一个 primary，风险上界/覆盖下界各 alpha=.025；证据不足阈值为 null，不暗回落 0。
  候选未审计也必须阈值 null。仅同源代理标签，无 human gold。
- group_prior 给模型无关的组等权先验上限 p/(1-.01)，不将类别均衡误作操作覆盖目标可行。
- pair/set 的 cluster_bootstrap 是唯一簇自助引擎；组等权与原行权并列、保留原始分子/分母。
  零分母复制计 invalid。旧行数稀有上界忽略簇相关，仅诊断；[0,0] 不作独立安全证据。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from research.memory_relations import features as F
from research.memory_relations.features import CLASSES

                                                            
ALLOWING_CLASSES = ("duplicate", "complement", "contextual")
BLOCKING_CLASSES = ("conflict", "supersede", "temporary", "uncertain")
CONTEXTUAL_REQUIRES_TRIGGER = True
TARGET_DANGEROUS_ALLOW = 0.01
MIN_COVERAGE = 0.5
BOOTSTRAP_SAMPLES = 500
CONFIDENCE = 0.95

if set(ALLOWING_CLASSES) & set(BLOCKING_CLASSES) or \
        set(ALLOWING_CLASSES) | set(BLOCKING_CLASSES) != set(CLASSES):
    raise RuntimeError("允许/阻塞类别必须构成 spec 七类的一个划分")


def default_thresholds() -> list[float]:
    """风险-覆盖率曲线的默认 τ 网格：0.00–0.95 步长 0.05。"""
    return [round(step * 0.05, 2) for step in range(20)]


def _allowing_positions(classes: Sequence[str]) -> np.ndarray:
    return np.array([name in ALLOWING_CLASSES for name in classes], dtype=bool)


def _blocking_positions(classes: Sequence[str]) -> np.ndarray:
    return np.array([name in BLOCKING_CLASSES for name in classes], dtype=bool)


def _verdict_indices(
    probs: np.ndarray,
    has_trigger: np.ndarray,
    threshold: float,
    *,
    classes: Sequence[str] = CLASSES,
) -> np.ndarray:
    """向量化 pair_verdict：唯一门逻辑，单行接口与评测共用（不允许第二套 gate）。"""
    probs = np.asarray(probs, dtype=np.float64)
    triggers = np.asarray(has_trigger, dtype=bool)
    if probs.ndim != 2 or probs.shape[1] != len(classes):
        raise ValueError("probs 形状必须是 (n, 类别数)")
    if len(triggers) != len(probs):
        raise ValueError("has_trigger 与 probs 长度不符")
    value = float(threshold)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("threshold 必须是非负有限数字")
    max_prob = probs.max(axis=1)
    top = probs == max_prob[:, None]
    first_top = top.argmax(axis=1)
    tie_conflict = ((top.sum(axis=1) > 1)
                    & (top & _allowing_positions(classes)).any(axis=1)
                    & (top & _blocking_positions(classes)).any(axis=1))
    if CONTEXTUAL_REQUIRES_TRIGGER:
        contextual = list(classes).index("contextual")
        contextual_without_trigger = (first_top == contextual) & ~triggers
    else:
        contextual_without_trigger = np.zeros(len(probs), dtype=bool)
    uncertain = list(classes).index("uncertain")
    result = first_top.copy()
    result[max_prob < value] = uncertain
    result[tie_conflict] = uncertain
    result[contextual_without_trigger] = uncertain
    return result


def pair_verdict(probs, request: dict, threshold: float = 0.0) -> str:
    """唯一配对决策门（adapter 消费）：probs 为单行 (7,) 或 (1,7)，返回七类之一。

    采纳条件 = argmax 最大概率 ≥ threshold 且无跨侧平局、且（top 为 contextual 时）候选带
    trigger；否则 uncertain。绝不对 allow/block 平局做 argmax 静默放行。
    """
    row = np.asarray(probs, dtype=np.float64)
    if row.ndim == 2 and row.shape[0] == 1:
        row = row[0]
    if row.ndim != 1 or row.shape[0] != len(CLASSES):
        raise ValueError("pair_verdict 需要形状 (7,) 或 (1,7) 的概率行")
    has_trigger = np.asarray([F.request_has_trigger(request)], dtype=bool)
    return CLASSES[int(_verdict_indices(row[None, :], has_trigger, threshold)[0])]


def _sample_weights(size: int, sample_weight=None) -> np.ndarray:
    weights = np.ones(size, dtype=np.float64) if sample_weight is None else np.asarray(
        sample_weight, dtype=np.float64)
    if weights.shape != (size,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("sample_weight 必须是等长、非负、有限向量")
    return weights


def _group_weights(group_ids: Sequence[str]) -> np.ndarray:
    counts: dict[str, int] = {}
    for group in group_ids:
        counts[group] = counts.get(group, 0) + 1
    return np.asarray([1.0 / counts[group] for group in group_ids], dtype=np.float64)


def operating_stats(
    probs: np.ndarray,
    y_true: np.ndarray,
    has_trigger: np.ndarray,
    threshold: float,
    *,
    classes: Sequence[str] = CLASSES,
    sample_weight=None,
) -> dict:
    """单门统计；可显式组等权，未加权时保留原整数分子/分母。"""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    has_trigger = np.asarray(has_trigger, dtype=bool)
    if probs.ndim != 2 or probs.shape[1] != len(classes) or len(y_true) != len(probs):
        raise ValueError("probs/y_true 形状不符")
    verdicts = _verdict_indices(probs, has_trigger, threshold, classes=classes)
    weights = _sample_weights(len(y_true), sample_weight)

    def total(mask):
        value = weights[mask].sum()
        return int(value) if sample_weight is None else float(value)

    is_allow = _allowing_positions(classes)[verdicts]
    gold_blocking = _blocking_positions(classes)[y_true]
    n = total(np.ones(len(y_true), dtype=bool))
    allow_count = total(is_allow)
    gold_blocking_count = total(gold_blocking)
    gold_allowing_count = n - gold_blocking_count
    dangerous = total(is_allow & gold_blocking)
    false_block = total(~is_allow & ~gold_blocking)
    return {
        "threshold": float(threshold),
        "n": n,
        "coverage": allow_count / n if n else None,
        "allow_count": allow_count,
        "gold_blocking_count": gold_blocking_count,
        "gold_allowing_count": gold_allowing_count,
        "verdict_counts": {classes[index]: total(verdicts == index)
                           for index in range(len(classes))},
        "dangerous_allow_count": dangerous,
        "conditional_risk": dangerous / allow_count if allow_count else None,
        "dangerous_allow_rate": dangerous / gold_blocking_count if gold_blocking_count else None,
        "dangerous_allow_share": dangerous / n if n else None,
        "false_block_count": false_block,
        "false_block_rate": false_block / gold_allowing_count if gold_allowing_count else None,
        "false_block_share": false_block / n if n else None,
    }


def risk_coverage(
    probs: np.ndarray,
    y_true: np.ndarray,
    has_trigger: np.ndarray,
    thresholds: "Sequence[float] | None" = None,
    *,
    classes: Sequence[str] = CLASSES,
) -> list[dict]:
    """各 τ 下的覆盖率/风险曲线（诊断数据；coverage 随 τ 单调不增）。"""
    grid = list(thresholds) if thresholds is not None else default_thresholds()
    return [operating_stats(probs, y_true, has_trigger, tau, classes=classes) for tau in grid]


def select_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    has_trigger: np.ndarray,
    *,
    target: float = TARGET_DANGEROUS_ALLOW,
    min_coverage: float = MIN_COVERAGE,
    thresholds: "Sequence[float] | None" = None,
    classes: Sequence[str] = CLASSES,
) -> "tuple[float | None, dict | None]":
    """在 calibration 上选最小可行 τ：conditional risk ≤ target 且覆盖率 ≥ min_coverage。

    无可行点时返回 (None, None)——报告无可行点、不声明已选操作策略，绝不回退成 τ=0。
    """
    for point in risk_coverage(probs, y_true, has_trigger, thresholds, classes=classes):
        risk = point["conditional_risk"]
        coverage = point["coverage"]
        if coverage is not None and coverage >= min_coverage - 1e-12 and \
                risk is not None and risk <= target + 1e-12:
            return point["threshold"], point
    return None, None


def fit_temperature(logits: np.ndarray, y_true: np.ndarray, *,
                    low: float = 0.3, high: float = 6.0, grid: int = 40,
                    sample_weight=None) -> float:
    """温度缩放：网格 + 黄金分割精修校准 NLL；权重可令每个 prob 根贡献等权。"""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=int)
    if z.ndim != 2 or len(z) != len(y) or len(y) < 2:
        raise ValueError("温度缩放需要 ≥2 条形状正确的校准样本")
    weights = _sample_weights(len(y), sample_weight)
    if not np.isfinite(z).all() or (y < 0).any() or (y >= z.shape[1]).any():
        raise ValueError("logits/labels 含非法值")
    if weights.sum() <= 0 or len(np.unique(y[weights > 0])) < 2:
        raise ValueError("温度缩放需要至少两个类别的校准样本")

    def nll(t: float) -> float:
        scaled = z / max(t, 1e-3)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        logp = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        return float(np.average(-logp[np.arange(len(y)), y], weights=weights))

    line = np.linspace(low, high, grid)
    best = min(line, key=nll)
    lo, hi = max(0.1, best - 0.4), best + 0.4
    ratio = (5 ** 0.5 - 1) / 2
    c, d = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    for _ in range(40):
        if nll(c) < nll(d):
            hi = d
        else:
            lo = c
        c, d = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    return float((lo + hi) / 2)


def group_prior(y_true: np.ndarray, group_ids: Sequence[str]) -> dict:
    """模型无关上限：risk≤r 要求 allow≤p/(1-r)，p 按根等权而非变体数计算。"""
    y = np.asarray(y_true, dtype=int)
    groups = list(group_ids)
    if y.shape != (len(groups),) or (y < 0).any() or (y >= len(CLASSES)).any():
        raise ValueError("group_ids/labels 形状或类序不符")
    weights = _group_weights(groups)
    denominator = float(weights.sum())
    numerator = float(weights[_allowing_positions(CLASSES)[y]].sum())
    prior = numerator / denominator if denominator else None
    bound = min(1.0, prior / (1.0 - TARGET_DANGEROUS_ALLOW)) if prior is not None else None
    return {"sample_count": len(y), "independent_groups": len(set(groups)),
            "allowing_mass": numerator, "group_mass": denominator,
            "allowing_prior": prior, "target_risk": TARGET_DANGEROUS_ALLOW,
            "max_coverage_at_target_risk": bound,
            "min_coverage_possible": bound >= MIN_COVERAGE if bound is not None else None,
            "model_independent": True, "weighting": "group_equal"}


def _partition_inputs(logits, labels, has_trigger, group_ids, roles, primary):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    triggers = np.asarray(has_trigger)
    selected = np.asarray(primary)
    groups, partitions = list(group_ids), list(roles)
    n = len(groups)
    if z.shape != (n, len(CLASSES)) or y.shape != (n,) or triggers.shape != (n,) \
            or selected.shape != (n,) or len(partitions) != n:
        raise ValueError("calibration 向量/类序形状不符")
    if not np.isfinite(z).all() or y.dtype.kind not in "iu" or (y < 0).any() \
            or (y >= len(CLASSES)).any() or triggers.dtype.kind != "b" \
            or selected.dtype.kind != "b":
        raise ValueError("calibration logits/labels/布尔向量非法")
    by_group: dict[str, str] = {}
    primaries: dict[str, int] = {}
    for group, role, is_primary in zip(groups, partitions, selected):
        if not isinstance(group, str) or not group.strip() or role not in ("prob", "policy", "audit"):
            raise ValueError("calibration group/role 非法")
        if group in by_group and by_group[group] != role:
            raise ValueError("calibration 同一 group 跨角色泄漏")
        by_group[group] = role
        primaries[group] = primaries.get(group, 0) + int(is_primary)
    if any(count != 1 for count in primaries.values()):
        raise ValueError("calibration 每个 group 必须恰好一个预选 primary")
    return z, y.astype(int), triggers, np.asarray(groups), np.asarray(partitions), selected


def _binomial_bounds(errors: int, allowed: int, total: int) -> dict:
    from scipy.stats import beta

    alpha = (1.0 - CONFIDENCE) / 2.0
    risk_upper = (float(beta.ppf(1.0 - alpha, errors + 1, allowed - errors))
                  if allowed > errors else 1.0) if allowed else None
    coverage_lower = (float(beta.ppf(alpha, allowed, total - allowed + 1))
                      if allowed else 0.0) if total else None
    return {"confidence": CONFIDENCE, "risk_alpha": alpha, "coverage_alpha": alpha,
            "method": "clopper_pearson_bonferroni_one_sided",
            "conditional_risk_upper": risk_upper, "coverage_lower": coverage_lower}


def _audit_at_locked_point(probs, labels, triggers, groups, primary, candidate,
                           *, point_feasible: bool) -> dict:
    selected = np.asarray(primary, dtype=bool)
    audit_stats = operating_stats(probs[selected], labels[selected], triggers[selected], candidate) \
        if candidate is not None else None
    allowed = audit_stats["allow_count"] if audit_stats is not None else None
    errors = audit_stats["dangerous_allow_count"] if audit_stats is not None else None
    n = int(selected.sum())
    minimum = int(np.ceil(np.log((1.0 - CONFIDENCE) / 2.0) /
                          np.log(1.0 - TARGET_DANGEROUS_ALLOW)))
    bounds = _binomial_bounds(errors, allowed, n) if audit_stats is not None else {
        "confidence": CONFIDENCE, "risk_alpha": (1.0 - CONFIDENCE) / 2.0,
        "coverage_alpha": (1.0 - CONFIDENCE) / 2.0,
        "method": "clopper_pearson_bonferroni_one_sided",
        "conditional_risk_upper": None, "coverage_lower": None}
    enough = allowed is not None and allowed >= minimum
    bounds_feasible = bounds["conditional_risk_upper"] is not None \
        and bounds["conditional_risk_upper"] <= TARGET_DANGEROUS_ALLOW \
        and bounds["coverage_lower"] is not None and bounds["coverage_lower"] >= MIN_COVERAGE
    feasible = bool(point_feasible and enough and bounds_feasible)
    return {"threshold": candidate, "locked": candidate is not None,
            "performed": candidate is not None, "status": "audited" if candidate is not None else "no_candidate",
            "sample_count": len(labels), "independent_groups": len(set(groups[selected])), "primary_count": n,
            "allow_count": allowed, "error_count": errors, "operating": audit_stats, **bounds,
            "minimum_zero_error_allow_groups": minimum, "enough_independent_support": bool(enough),
            "bounds_feasible": bool(bounds_feasible), "sufficient": feasible,
            "prior": group_prior(labels[selected], groups[selected]),
            "scope": "one_preselected_primary_per_group_same_source_proxy_labels",
            "note": "独立根仅为研究抽样单位；没有 human gold，不外推真实用户或生产安全"}


def _deferred_audit() -> dict:
    return {"status": "deferred", "performed": False, "locked": False, "sufficient": False,
            "threshold": None, "sample_count": None, "independent_groups": None, "primary_count": None,
            "allow_count": None, "error_count": None, "conditional_risk_upper": None, "coverage_lower": None,
            "note": "候选阶段未取得audit数据；仅供唯一锁定候选独立一次审计"}


def calibrate_partitioned(logits, labels, has_trigger, group_ids, roles, primary,
                          *, stage: str = "complete") -> dict:
    """共享校准：candidate只见prob/policy；完整六向量旧API保留合成数值兼容。"""
    if stage not in ("complete", "candidate"):
        raise ValueError("calibration stage 无效")
    z, y, triggers, groups, partitions, selected = _partition_inputs(
        logits, labels, has_trigger, group_ids, roles, primary)
    if stage == "candidate" and (partitions == "audit").any():
        raise ValueError("candidate校准禁止audit输入，须物理分离")
    prob = partitions == "prob"
    policy = (partitions == "policy") & selected
    prob_groups = len(set(groups[prob]))
    fitted = prob_groups >= 2 and len(np.unique(y[prob])) >= 2
    temperature = fit_temperature(z[prob], y[prob], sample_weight=_group_weights(groups[prob])) \
        if fitted else 1.0
    probs = F.softmax(z / temperature)
    candidate, point = select_threshold(probs[policy], y[policy], triggers[policy]) \
        if fitted else (None, None)
    selection = {"candidate_threshold": candidate, "point_feasible": point is not None,
                 "point": point, "sample_count": int((partitions == "policy").sum()),
                 "independent_groups": len(set(groups[policy])), "primary_count": int(policy.sum()),
                 "temperature_sample_count": int(prob.sum()), "temperature_groups": prob_groups,
                 "temperature_fitted": bool(fitted), "prior": group_prior(y[policy], groups[policy])}
    if stage == "candidate":
        audit_record = _deferred_audit()
    else:
        audit = partitions == "audit"
        audit_record = _audit_at_locked_point(probs[audit], y[audit], triggers[audit], groups[audit],
                                             selected[audit], candidate, point_feasible=point is not None)
    feasible = audit_record["sufficient"]
    return {"stage": stage, "temperature": float(temperature), "threshold": candidate if feasible else None,
            "threshold_feasible": feasible, "selection": selection, "audit": audit_record}


def audit_locked_partition(logits, labels, has_trigger, group_ids, roles, primary, *,
                           temperature: float, candidate_threshold: "float | None",
                           point_feasible: bool) -> dict:
    """只审已锁温度/阈值，复用唯一CP统计；不调用温度拟合或阈值网格选择。"""
    z, y, triggers, groups, partitions, selected = _partition_inputs(
        logits, labels, has_trigger, group_ids, roles, primary)
    if (partitions != "audit").any() or not np.isfinite(temperature) or temperature <= 0 \
            or type(point_feasible) is not bool:
        raise ValueError("锁点审计仅接受audit及合法锁定温度/可行标记")
    if candidate_threshold is not None and (isinstance(candidate_threshold, bool)
            or not isinstance(candidate_threshold, (int, float)) or not np.isfinite(candidate_threshold)
            or not 0 <= candidate_threshold <= 1):
        raise ValueError("锁定阈值必须为[0,1]数值或null")
    if (candidate_threshold is not None) != point_feasible:
        raise ValueError("锁定候选阈值与point_feasible不一致")
    record = _audit_at_locked_point(F.softmax(z / temperature), y, triggers, groups, selected,
                                   candidate_threshold, point_feasible=point_feasible)
    return {"stage": "locked-audit", "temperature": float(temperature),
            "threshold": candidate_threshold if record["sufficient"] else None,
            "threshold_feasible": record["sufficient"], "audit": record}


def ece(probs: np.ndarray, y_true: np.ndarray, *, n_bins: int = 10,
        sample_weight=None) -> float:
    """多分类 ECE（max-prob 口径，等宽箱）；缺省仍按行等权。"""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    weights = _sample_weights(len(probs), sample_weight)
    if not weights.sum():
        return 0.0
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high) if low > 0 else \
            (confidence >= low) & (confidence <= high)
        mass = weights[mask].sum()
        if mass:
            total += mass * abs(np.average(correct[mask], weights=weights[mask]) -
                                np.average(confidence[mask], weights=weights[mask]))
    return float(total / weights.sum())


def brier(probs: np.ndarray, y_true: np.ndarray, *, sample_weight=None) -> float:
    """多分类 Brier：加权 mean_i Σ_k (p_ik − 1{k=y_i})²。"""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    weights = _sample_weights(len(probs), sample_weight)
    if not weights.sum():
        return 0.0
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.average(np.sum((probs - onehot) ** 2, axis=1), weights=weights))


def _classification_counts(y_true, y_pred, n_classes, sample_weight):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if n_classes < 1 or y_true.ndim != 1 or y_pred.shape != y_true.shape \
            or (y_true < 0).any() or (y_true >= n_classes).any() \
            or (y_pred < -1).any() or (y_pred >= n_classes).any():
        raise ValueError("分类标签须在类序内，失败预测仅用 -1")
    weights = _sample_weights(len(y_true), sample_weight)
    success = y_pred >= 0
    table = np.bincount(y_true[success] * n_classes + y_pred[success],
                        weights=weights[success], minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    failures = np.bincount(y_true[~success], weights=weights[~success], minlength=n_classes)
    return table, failures, weights


def _classification_scores(table, failures):
    true_positive = np.diag(table).astype(np.float64)
    false_positive = table.sum(axis=0) - true_positive
    false_negative = table.sum(axis=1) - true_positive + failures
    def ratio(numerator, denominator):
        return np.divide(numerator, denominator, out=np.zeros(len(table)), where=denominator > 0)
    return (true_positive, false_positive, false_negative,
            ratio(true_positive, true_positive + false_positive),
            ratio(true_positive, true_positive + false_negative),
            ratio(2.0 * true_positive, 2.0 * true_positive + false_positive + false_negative))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, *, n_classes: int = len(CLASSES),
             sample_weight=None) -> float:
    """默认固定七类；缺席类记0，-1失败预测仅贡献gold类FN，不增加类别。"""
    table, failures, _ = _classification_counts(y_true, y_pred, n_classes, sample_weight)
    return float(_classification_scores(table, failures)[-1].mean())


def classification_summary(y_true, y_pred, *, classes=CLASSES, sample_weight=None) -> dict:
    """通用失败感知汇总；合法gold全入分母，失败预测为-1，空/零权子集为null。"""
    table, failures, weights = _classification_counts(y_true, y_pred, len(classes), sample_weight)
    tp, fp, fn, precision, recall, f1 = _classification_scores(table, failures)
    support = table.sum(axis=1) + failures
    mass = float(weights.sum())
    return {"n": len(weights), "weight": mass, "failure_weight": float(failures.sum()),
            "hard_success_weight": float(table.sum()),
            "accuracy": float(tp.sum() / mass) if mass else None,
            "macro_f1": float(f1.mean()) if mass else None,
            "per_class": {name: {"support": float(support[i]), "tp": float(tp[i]),
                                 "fp": float(fp[i]), "fn": float(fn[i]),
                                 "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
                          for i, name in enumerate(classes)},
            "confusion_matrix": {"classes": list(classes), "matrix": table.tolist(),
                                 "failure_by_gold": failures.tolist()}}


def probability_row(probabilities, n_classes: int) -> tuple[str, np.ndarray | None]:
    """校验而不补齐、裁剪或归一化；缺失与坏simplex分别留状态。"""
    if probabilities is None:
        return "missing", None
    try:
        values = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError):
        return "invalid_shape", None
    if values.shape != (n_classes,):
        return "invalid_shape", None
    if not np.isfinite(values).all():
        return "nonfinite", None
    if (values < 0).any() or (values > 1).any() or not np.isclose(values.sum(), 1., rtol=0., atol=1e-6):
        return "invalid_simplex", None
    return "ok", values


def probability_summary(probs, y_true, *, n_classes: int = len(CLASSES), n_bins: int = 10,
                        sample_weight=None) -> dict:
    """概率可用子集的proper指标；覆盖另列，真实零gold质量显式为正无穷。"""
    y = np.asarray(y_true, dtype=int)
    if y.ndim != 1 or len(probs) != len(y) or (y < 0).any() or (y >= n_classes).any():
        raise ValueError("probs/y_true 形状或类序不符")
    weights = _sample_weights(len(y), sample_weight)
    checked = [probability_row(row, n_classes) for row in probs]
    statuses = [item[0] for item in checked]
    valid = np.asarray([status == "ok" for status in statuses], dtype=bool)
    mass, available = float(weights.sum()), float(weights[valid].sum())
    result = {"n": len(y), "available_count": int(valid.sum()),
              "missing_count": statuses.count("missing"),
              "invalid_count": sum(status not in ("ok", "missing") for status in statuses),
              "available_weight": available, "coverage": available / mass if mass else None,
              "brier": None, "ece": None, "zero_gold_count": 0,
              "nll": {"value": None, "status": "not_available"}}
    if available:
        p = np.asarray([checked[i][1] for i in np.flatnonzero(valid)])
        labels, selected_weights = y[valid], weights[valid]
        gold_p = p[np.arange(len(labels)), labels]
        zero = (gold_p == 0) & (selected_weights > 0)
        result.update(brier=brier(p, labels, sample_weight=selected_weights),
                      ece=ece(p, labels, n_bins=n_bins, sample_weight=selected_weights),
                      zero_gold_count=int(zero.sum()))
        if zero.any():
            result["nll"] = {"value": None, "status": "positive_infinity"}
        else:
            positive_weight = selected_weights > 0
            result["nll"] = {"value": float(np.average(-np.log(gold_p[positive_weight]),
                                                       weights=selected_weights[positive_weight])), "status": "finite"}
    return result


def _rare_upper_bound(denominator: int, confidence: float) -> float:
    """零错误时保守单侧上限：精确二项上界 1-(1-confidence)^(1/k)。

    假设：放行决定近似独立同分布（忽略 group 簇相关，故仅作指示上界）。
    """
    if denominator <= 0:
        return 1.0
    return float(1.0 - (1.0 - confidence) ** (1.0 / denominator))


def cluster_bootstrap(group_ids, metric, *, samples: int = BOOTSTRAP_SAMPLES,
                      confidence: float = CONFIDENCE, seed: int = 20260919,
                      weighting: str = "row", numerator=None, denominator=None,
                      errors=None) -> dict:
    """pair/set 共用的唯一簇自助引擎；metric(index, weights) 接收重采样索引与权重。

    group_equal 每次选中根贡献总权重 1；零分母复制计 invalid。二项稀有上界不适用于
    权重伪样本；行等权上界仅旧口径诊断，正式风险证据须独立预选 primary 的联合单侧界。
    """
    if samples < 1 or not 0 < confidence < 1 or weighting not in ("row", "group_equal"):
        raise ValueError("bootstrap 采样数/置信度/权重口径非法")
    groups = list(group_ids)
    clusters: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        clusters.setdefault(str(group), []).append(index)
    rows = [np.asarray(clusters[key], dtype=int) for key in sorted(clusters)]
    index = np.arange(len(groups))
    weights = _group_weights(groups) if weighting == "group_equal" else np.ones(len(groups))
    point = metric(index, weights)
    values = []
    invalid = 0
    rng = np.random.default_rng(seed)
    if len(rows) >= 2:
        for _ in range(samples):
            picked = [rows[position] for position in rng.integers(0, len(rows), size=len(rows))]
            selected = np.concatenate(picked)
            weights = np.concatenate([np.full(len(row), 1.0 / len(row)) for row in picked]) \
                if weighting == "group_equal" else np.ones(len(selected))
            value = metric(selected, weights)
            if value is None:
                invalid += 1
            else:
                values.append(float(value))
    else:
        invalid = samples
    ci = None
    if values:
        alpha = (1.0 - confidence) / 2.0
        ci = [float(np.percentile(values, 100 * alpha)),
              float(np.percentile(values, 100 * (1 - alpha)))]
    rare = (_rare_upper_bound(denominator, confidence)
            if weighting == "row" and errors == 0 and denominator is not None and denominator > 0
            else None)
    return {"point": None if point is None else float(point), "ci": ci,
            "lo": ci[0] if ci else None, "hi": ci[1] if ci else None,
            "n": len(values), "valid": len(values), "invalid": invalid, "samples": samples,
            "group_clusters": len(rows), "weighting": weighting,
            "numerator": numerator, "denominator": denominator, "errors": errors,
            "zero_errors": errors == 0 if errors is not None else None,
            "rare_upper_bound": rare,
            "rare_bound_note": "旧行数二项诊断忽略簇相关，不是独立 primary 风险证据" if rare else None}


def bootstrap_ci(
    probs: np.ndarray,
    y_true: np.ndarray,
    has_trigger: np.ndarray,
    group_ids: Sequence[str],
    threshold: float,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    confidence: float = CONFIDENCE,
    seed: int = 20260919,
    classes: Sequence[str] = CLASSES,
    weighting: str = "row",
) -> dict:
    """共用簇自助引擎；默认保留旧行权，显式 group_equal 补根等权与原始计数。"""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    has_trigger = np.asarray(has_trigger, dtype=bool)
    groups = list(group_ids)
    if len(groups) != len(y_true) or len(y_true) != len(probs):
        raise ValueError("group_ids/y_true/probs 长度不符")
    weights = _group_weights(groups) if weighting == "group_equal" else None
    full = operating_stats(probs, y_true, has_trigger, threshold,
                           classes=classes, sample_weight=weights)
    raw = operating_stats(probs, y_true, has_trigger, threshold, classes=classes)
    denominators = {"macro_f1": "n", "coverage": "n", "conditional_risk": "allow_count",
                    "dangerous_allow_rate": "gold_blocking_count", "false_block_rate": "gold_allowing_count"}
    numerators = {"coverage": "allow_count", "conditional_risk": "dangerous_allow_count",
                  "dangerous_allow_rate": "dangerous_allow_count", "false_block_rate": "false_block_count"}
    result = {"samples": samples, "confidence": confidence, "seed": seed,
              "threshold": float(threshold), "weighting": weighting,
              "raw_operating": raw}
    for name, denominator in denominators.items():
        def metric(index, sampled_weights, name=name):
            if name == "macro_f1":
                return macro_f1(y_true[index], probs[index].argmax(axis=1), n_classes=len(classes),
                                sample_weight=sampled_weights)
            return operating_stats(probs[index], y_true[index], has_trigger[index], threshold,
                                   classes=classes, sample_weight=sampled_weights)[name]

        numerator = full[numerators[name]] if name in numerators else None
        errors = numerator if name not in ("macro_f1", "coverage") else None
        record = cluster_bootstrap(groups, metric, samples=samples, confidence=confidence,
                                   seed=seed, weighting=weighting, numerator=numerator,
                                   denominator=full[denominator], errors=errors)
        record["raw_numerator"] = raw[numerators[name]] if name in numerators else None
        record["raw_denominator"] = raw[denominator]
        result[name] = record
    return result


def evaluate(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    raw_probs: "np.ndarray | None" = None,
    has_trigger: np.ndarray,
    group_ids: Sequence[str],
    threshold: "float | None",
    thresholds: "Sequence[float] | None" = None,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    confidence: float = CONFIDENCE,
    seed: int = 20260919,
    classes: Sequence[str] = CLASSES,
    n_bins: int = 10,
) -> dict:
    """单个评测集的全量指标；threshold=None 表示未选到可行运行点。

    分类/校准指标（宏 F1/逐类/混淆/ECE/Brier）与阈值无关，始终报告；
    operating 仅在 threshold 非 None（已选运行点）时给出；τ=0 一律作为
    uncalibrated diagnostic 单独列出，不冒充已选操作策略。
    """
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    has_trigger = np.asarray(has_trigger, dtype=bool)
    if probs.ndim != 2 or probs.shape[1] != len(classes) or len(y_true) != probs.shape[0]:
        raise ValueError("probs/y_true 形状不符")
    predictions = probs.argmax(axis=1)
    raw = probs if raw_probs is None else np.asarray(raw_probs, dtype=np.float64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=list(range(len(classes))), zero_division=0)
    if threshold is None:
        operating = {"threshold": None, "selected": False}
        gate_threshold = 0.0
        scope = "uncalibrated_tau0_diagnostic"
    else:
        operating = {"selected": True,
                     **operating_stats(probs, y_true, has_trigger, threshold, classes=classes)}
        gate_threshold = float(threshold)
        scope = "operating"
    return {
        "n": int(len(y_true)),
        "label_counts": {classes[index]: int((y_true == index).sum())
                         for index in range(len(classes))},
        "macro_f1": macro_f1(y_true, predictions),
        "accuracy": float((predictions == y_true).mean()) if len(y_true) else 0.0,
        "per_class": {classes[index]: {"precision": float(precision[index]),
                                       "recall": float(recall[index]),
                                       "f1": float(f1[index]),
                                       "support": int(support[index])}
                      for index in range(len(classes))},
        "confusion_matrix": {"classes": list(classes),
                             "matrix": confusion_matrix(
                                 y_true, predictions,
                                 labels=list(range(len(classes)))).tolist()},
        "ece": ece(probs, y_true, n_bins=n_bins),
        "ece_raw": ece(raw, y_true, n_bins=n_bins),
        "brier": brier(probs, y_true),
        "brier_raw": brier(raw, y_true),
        "risk_coverage": risk_coverage(probs, y_true, has_trigger, thresholds, classes=classes),
        "operating": operating,
        "diagnostic_tau0": {"threshold": 0.0, "scope": "uncalibrated_tau0_diagnostic",
                            **operating_stats(probs, y_true, has_trigger, 0.0, classes=classes)},
        "bootstrap": {"scope": scope,
                      **bootstrap_ci(probs, y_true, has_trigger, group_ids, gate_threshold,
                                     samples=bootstrap_samples, confidence=confidence,
                                     seed=seed, classes=classes)},
        "group_prior": group_prior(y_true, group_ids),
        "gated_confusion_matrix": {"scope": scope, "classes": list(classes),
                                   "matrix": confusion_matrix(
                                       y_true, _verdict_indices(probs, has_trigger, gate_threshold,
                                                                classes=classes),
                                       labels=list(range(len(classes)))).tolist()},
        "bootstrap_group_equal": {"scope": scope,
                                   **bootstrap_ci(probs, y_true, has_trigger, group_ids, gate_threshold,
                                                  samples=bootstrap_samples, confidence=confidence,
                                                  seed=seed, classes=classes, weighting="group_equal")},
    }
