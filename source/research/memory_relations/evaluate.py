"""关系模型的 set 级离线回放与对照评测（research-only，无网络）。

真实 M.assess_relations 完整分批回放，不 top-k、不改生产或 adapter 聚合。主口径的 N 是
全部预定逻辑请求；row_error/系统失败不被排除，completion 与 correct_completed/N 分开，
语义 uncertain 是完成的弃权，不伪称系统失败。旧 valid-only 分类/引用并集是 secondary。

make_evaluation_predictor 观察真实 scorer，分开 classifier argmax、pair_verdict、set 原始
输出与生产闸后结果；普通 callable 未观察时 classifier 标 unavailable，不从 set 关系伪造。
related_ids/evidence_refs 分角色计 precision/recall、unknown/known 非 expected/漏标；坏请求的
合法期望引用仍进召回分母。调用尝试与耗时在 finally 记账，异常正文不记录。

风险定义沿用 metrics 的 1% conditional risk + 50% allow；主要独立风险证据只采用每根一个
预分配 operating_primary。行权/组等权统计与原分子分母并列，bootstrap 复用 metrics 唯一
引擎；旧行数二项上界仅诊断。group_prior 的 p/(1-.01) 上限与模型表现无关。全部标签是
同源合成代理，无 human gold，独立根不是独立真人核验或真实用户分布。

跨批合并沿用生产 _merge_decisions：任一批 uncertain/错误码覆盖其它批 conflict；非不确定
批严重度取关系、引用并集（可能污染所选类别引用）。批内混杂检查不跨批。evidence_refs 为
宽泛输入引用，不构成逐条关系实证。可选 oracle-aggregation 用完整参考 pair 标签经真实
adapter/M，仅解释 pair/set 聚合误差，不新建聚合策略，不是人工金标。

CLI: python -m research.memory_relations.evaluate --model-dir DIR --data JSONL --out FILE
     [--threshold F --diagnostic] [--oracle-aggregation]
默认必须 meta 顶层 threshold 为 [0,1] 数值且 threshold_feasible 严格为 true；null/缺失/未
证实均拒绝。显式阈值必须配 --diagnostic；诊断运行不声称已校准，不暗回落 0。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from dataset_recommender.agent import memory_relations as M

from . import adapter

__all__ = ["evaluate_sets", "make_evaluation_predictor", "main"]

_BOOTSTRAP_SEED = 20260919

_SEMANTICS = (
    "逐对评分 + 显式聚合的研究回放与现行 LLM 单次 set 语义非等价，不得以整集特征矩阵"
    "（whole-set matrix）冒充 set 语义。跨批 uncertain 覆盖其它批 conflict，非不确定批取严重度、"
    "并集引用；批内混杂检查不跨批。evidence_refs 为宽泛输入引用，不构成对每条关系的实证。"
    "主口径 N=全部预定逻辑请求，row_error/系统失败不排除；completion 与 correct_completed/N "
    "分开。语义 uncertain 是完成的弃权，不是系统失败；旧 valid-only 分类与并集引用仅 secondary。"
    "classifier/pair gate 与 set raw/gated 分开，related/evidence 分角色计引用质量；unknown=0 "
    "不外推语义质量。主要风险证据仅用每根一个预选 primary；变体不伪独立。全部同源代理标签，"
    "无 human gold，不是独立核验、真实用户分布或生产批准证据。"
)


class RowContractError(ValueError):
    """对照字段/行层契约错误；`code` 是有界机器码。"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _metrics_module():
    from . import metrics

    return metrics


def _materialize(request: object):
    from . import schema

    return schema.materialize_request(request)


def _row_error_type():
    """schema 的行契约异常类型（catch 用）；schema 未定义 RowError 即不可如实归类。"""
    from . import schema

    error = getattr(schema, "RowError", None)
    if not (isinstance(error, type) and issubclass(error, ValueError)):
        raise RuntimeError("schema.RowError 缺失：行契约错误无法如实归类")
    return error


def _load_scorer(model_dir: object):
    from . import features

    return features.load_scorer(model_dir, device="cpu")


def _acceptable_labels(row: dict, classes: tuple) -> list:
    """可接受标签集合：顶层 acceptable_labels → phenomena.acceptable_labels → [label]。"""
    value = row.get("acceptable_labels")
    if value is None:
        phenomena = row.get("phenomena")
        if isinstance(phenomena, dict):
            value = phenomena.get("acceptable_labels")
    if value is None:
        return [row["label"]]
    if (isinstance(value, (list, tuple)) and value
            and all(isinstance(item, str) and item in classes for item in value)):
        return list(value)
    raise RowContractError("invalid_acceptable_labels")


def _expected_refs_by_role(row: dict) -> dict:
    refs = {}
    for name in ("related_ids", "evidence_refs"):
        value = row.get(name)
        if not isinstance(value, list) or any(
                not isinstance(item, str) or not item.strip() for item in value):
            raise RowContractError("invalid_expected_refs")
        refs[name] = list(dict.fromkeys(value))
    return refs


def _expected_refs(row: dict) -> list:
    roles = _expected_refs_by_role(row)
    return list(dict.fromkeys(roles["related_ids"] + roles["evidence_refs"]))


def _prepare(row: object, classes: tuple):
    """行 → 回放输入 + 对照字段；结构/对照违规抛 RowError/RowContractError（fail-zero）。"""
    if not isinstance(row, dict):
        raise RowContractError("invalid_row")
    label = row.get("label")
    if not isinstance(label, str) or label not in classes:
        raise RowContractError("invalid_label")
    group_id = row.get("group_id")
    if not isinstance(group_id, str) or not group_id.strip():
        raise RowContractError("invalid_group_id")
    phenomena = row.get("phenomena")
    primary = phenomena.get("operating_primary") if isinstance(phenomena, dict) else None
    if primary is not None and type(primary) is not bool:
        raise RowContractError("invalid_operating_primary")
    origin = row.get("label_origin")
    refs = _expected_refs_by_role(row)
    meta = {
        "label": label, "group_id": group_id,
        "label_origin": origin if isinstance(origin, str) else None,
        "acceptable": _acceptable_labels(row, classes),
        "expected_refs": list(dict.fromkeys(refs["related_ids"] + refs["evidence_refs"])),
        "expected_refs_by_role": refs, "operating_primary": primary,
    }
    candidate, existing, evidence, cfg = _materialize(row.get("request"))
    return candidate, existing, evidence, cfg, meta


def _scored_pairs(predictor) -> "int | None":
    stats = getattr(predictor, "stats", None)
    if isinstance(stats, dict) and isinstance(stats.get("pairs"), int):
        return stats["pairs"]
    return None


def _rate(part: int, whole: int) -> "float | None":
    return part / whole if whole else None


def _rate_record(part: int, whole: int) -> dict:
    """带分子/分母的比率记录：0 分母 value=null（不用 0 冒充）。"""
    return {"numerator": int(part), "denominator": int(whole),
            "value": (part / whole) if whole else None}


def _cluster_interval(scored: list, metric, *, samples: int, confidence: float, seed: int) -> "dict | None":
    """保留旧 secondary 返回形状，重采样只走 metrics 唯一引擎。"""
    result = _metrics_module().cluster_bootstrap(
        [record["group_id"] for record in scored],
        lambda index, weights: metric([scored[int(i)] for i in index]),
        samples=samples, confidence=confidence, seed=seed)
    if result["point"] is None or result["ci"] is None:
        return None
    return {key: result[key] for key in ("point", "lo", "hi", "n", "samples", "invalid")}


class _ObservedScorer:
    def __init__(self, scorer, threshold):
        self.scorer = scorer
        self.classes = scorer.classes
        self.threshold = threshold
        self.calls = []

    def predict_proba(self, requests):
        record = {"pairs": [], "pair_count": len(requests), "error": None}
        try:
            output = self.scorer.predict_proba(requests)
            matrix = adapter._matrix(output, len(requests), len(self.classes))
            if matrix is None:
                record["error"] = "invalid_probability_matrix"
            else:
                metrics = _metrics_module()
                for request, probs in zip(requests, matrix):
                    record["pairs"].append({
                        "rule_id": request["existing"][0]["id"] if request["existing"] else None,
                        "classifier_relation": self.classes[int(probs.argmax())],
                        "pair_relation": metrics.pair_verdict(probs, request, self.threshold)})
            return output
        except Exception as exc:
            record["error"] = type(exc).__name__
            raise
        finally:
            self.calls.append(record)


def make_evaluation_predictor(scorer, threshold):
    """只加只读观察层；实际 pair/set 判断仍是原 adapter，原始 classifier 不冒充闸后输出。"""
    adapter.make_predictor(scorer, threshold=threshold)
    observed = _ObservedScorer(scorer, threshold)
    predictor = adapter.make_predictor(observed, threshold=threshold)
    predictor.classifier_trace = observed.calls
    return predictor


def _raw_decision(calls, classes):
    outputs = [call["raw"] for call in calls]
    if not outputs or any(not isinstance(raw, dict) or raw.get("relation") not in classes
                          or any(not isinstance(raw.get(name), list) or any(
                              not isinstance(ref, str) for ref in raw[name])
                              for name in ("related_ids", "evidence_refs"))
                          or "error_code" not in raw for raw in outputs):
        return None
    return M._merge_decisions(outputs) if len(outputs) > 1 else dict(outputs[0])


def _completed(decision, calls, classifier_calls):
    if decision is None or any(call["failure"] for call in calls) \
            or any(call["error"] for call in classifier_calls):
        return False
    code = decision["error_code"]
    if code is None:
        return True
    return decision["relation"] == "uncertain" and code in M._UNCERTAIN_CODES and any(
        isinstance(call["raw"], dict) and call["raw"].get("relation") == "uncertain"
        and call["raw"].get("error_code") in (None, *M._UNCERTAIN_CODES) for call in calls)


def _reference_scores(predicted, expected, known) -> dict:
    predicted = list(dict.fromkeys(predicted))
    expected = list(dict.fromkeys(expected))
    hits = len(set(predicted) & set(expected))
    return {"predicted": predicted, "expected": expected, "hits": hits,
            "precision": _rate(hits, len(predicted)), "recall": _rate(hits, len(expected)),
            "unknown_ids": [ref for ref in predicted if ref not in known],
            "known_non_expected": [ref for ref in predicted if ref in known and ref not in expected],
            "missed_expected": [ref for ref in expected if ref not in predicted]}


def _role_scores(decision, expected, known) -> dict:
    return {role: _reference_scores(decision[role] if decision else [], expected[role], known[role])
            for role in ("related_ids", "evidence_refs")}


def _role_summary(records, source="gated") -> dict:
    result = {}
    for role in ("related_ids", "evidence_refs"):
        scores = [r["refs_by_role"][source][role] for r in records if r.get("refs_by_role")]
        predicted = sum(len(s["predicted"]) for s in scores)
        expected = sum(len(s["expected"]) for s in scores)
        hits = sum(s["hits"] for s in scores)
        unknown = sum(len(s["unknown_ids"]) for s in scores)
        result[role] = {"rows": len(scores), "predicted_total": predicted,
                        "expected_total": expected, "hits": hits,
                        "precision": _rate_record(hits, predicted),
                        "recall": _rate_record(hits, expected),
                        "unknown_total": unknown, "unknown_rate": _rate_record(unknown, predicted),
                        "known_non_expected_total": sum(len(s["known_non_expected"]) for s in scores),
                        "missed_expected_total": sum(len(s["missed_expected"]) for s in scores)}
    return result


def _attempt_view(records, classes, allowing, blocking, weights=None, *, source="decision") -> dict:
    weights = np.ones(len(records)) if weights is None else weights
    decisions = [r.get(source) for r in records]
    completed = np.asarray([bool(r.get("completed", not r.get("error") and r.get("decision") is not None))
                            if source == "decision" else d is not None
                            for r, d in zip(records, decisions)], dtype=bool)
    allow = np.asarray([d is not None and d["relation"] in allowing for d in decisions], dtype=bool)
    block = np.asarray([d is not None and d["relation"] in blocking for d in decisions], dtype=bool)
    gold_block = np.asarray([isinstance(r["label"], str) and r["label"] in blocking
                             for r in records], dtype=bool)
    gold_allow = np.asarray([isinstance(r["label"], str) and r["label"] in allowing
                             for r in records], dtype=bool)
    correct = np.asarray([d is not None and isinstance(r["label"], str) and d["relation"] == r["label"]
                          for r, d in zip(records, decisions)], dtype=bool) & completed

    def total(mask):
        return float(np.asarray(weights)[np.asarray(mask, dtype=bool)].sum())

    def rate(mask, denominator):
        part = total(mask)
        return {"numerator": part, "denominator": denominator, "value": _rate(part, denominator)}

    n = float(np.asarray(weights).sum())
    allow_n, block_n = total(allow), total(block)
    dangerous, false_block = allow & gold_block, block & gold_allow & completed
    operation = {"n": n, "scope": "all_attempt_preplanned_logical_requests",
                 "gold_blocking_count": total(gold_block), "gold_allowing_count": total(gold_allow),
                 "allowing_pred_count": allow_n, "blocking_pred_count": block_n,
                 "coverage": rate(allow, n), "dangerous_allow_count": total(dangerous),
                 "dangerous_allow_rate": rate(dangerous, total(gold_block)),
                 "dangerous_allow_share": rate(dangerous, n),
                 "conditional_risk": rate(dangerous, allow_n),
                 "false_block_count": total(false_block),
                 "false_block_rate": rate(false_block, total(gold_allow)),
                 "false_block_share": rate(false_block, n),
                 "false_block_conditional_precision": rate(false_block, block_n),
                 "failed_gold_allowing_count": total(~completed & gold_allow),
                 "unknown_gold_count": total(~(gold_block | gold_allow)),
                 "rate_note": "row_error/系统失败留在 N；false_block 不把系统失败伪作语义判错"}
    return {"n": n, "completed_count": total(completed), "failed_count": total(~completed),
            "correct_completed_count": total(correct),
            "completion": rate(completed, n), "correct_completed": rate(correct, n),
            "agreement_completed": rate(correct, total(completed)), "operation": operation}


def _confusion(records, classes, source, *, failure_column=False):
    outcomes = list(classes) + (["system_failure"] if failure_column else [])
    table = [[0] * len(outcomes) for _ in classes]
    unknown_gold = 0
    missing_output = 0
    for record in records:
        if record["label"] not in classes:
            unknown_gold += 1
            continue
        decision = record.get(source)
        if failure_column and not record.get("completed", decision is not None and not record.get("error")):
            relation = "system_failure"
        else:
            relation = decision["relation"] if decision else None
        if relation not in outcomes:
            missing_output += 1
        else:
            table[classes.index(record["label"])][outcomes.index(relation)] += 1
    return {"classes": list(classes), "outcomes": outcomes, "matrix": table,
            "unknown_gold_count": unknown_gold, "missing_output_count": missing_output}


def _pair_labels(row, existing, label, classes):
    phenomena = row.get("phenomena") if isinstance(row, dict) else None
    annotated = phenomena.get("pair_labels") if isinstance(phenomena, dict) else None
    if annotated is None:
        return {existing[0]["id"] if existing else None: label} if len(existing) <= 1 else None
    if not isinstance(annotated, list):
        return None
    if not existing and not annotated:
        return {None: label}
    labels = {}
    for item in annotated:
        if not isinstance(item, dict) or item.get("relation") not in classes \
                or not isinstance(item.get("rule_id"), str) or item["rule_id"] in labels:
            return None
        labels[item["rule_id"]] = item["relation"]
    return labels if set(labels) == {r["id"] for r in existing} else None


def _oracle_probe(pair_labels, candidate, existing, evidence, cfg, classes):
    if pair_labels is None:
        return {"available": False, "reason": "complete_reference_pair_labels_missing"}

    class ReferenceScorer:
        def __init__(self):
            self.classes = list(classes)

        def predict_proba(self, requests):
            probs = np.zeros((len(requests), len(classes)))
            for index, request in enumerate(requests):
                identifier = request["existing"][0]["id"] if request["existing"] else None
                probs[index, classes.index(pair_labels[identifier])] = 1.0
            return probs

    predictor = adapter.make_predictor(ReferenceScorer(), threshold=0.0)
    decision = M.assess_relations(candidate, existing, evidence, cfg, predictor=predictor)
    return {"available": True, "decision": decision, "pairs": predictor.stats["pairs"],
            "scope": "reference_proxy_pair_labels_real_adapter_and_production_gate_not_human_oracle"}


def _extended_summary(records, metrics, classes, allowing, blocking):
    all_attempt = _attempt_view(records, classes, allowing, blocking)
    groups = [r["group_id"] if isinstance(r["group_id"], str) and r["group_id"] else
              f"unattributed-row-{i}" for i, r in enumerate(records)]
    weights = metrics._group_weights(groups)
    equal = _attempt_view(records, classes, allowing, blocking, weights)
    bootstrap = {}
    for name in ("completion", "correct_completed", "coverage", "conditional_risk", "false_block_rate"):
        def metric(index, weights, name=name):
            view = _attempt_view([records[int(i)] for i in index], classes, allowing, blocking, weights)
            return (view[name] if name in ("completion", "correct_completed") else
                    view["operation"][name])["value"]

        point = (equal[name] if name in ("completion", "correct_completed") else equal["operation"][name])
        raw = (all_attempt[name] if name in ("completion", "correct_completed") else
               all_attempt["operation"][name])
        bootstrap[name] = metrics.cluster_bootstrap(
            groups, metric, samples=metrics.BOOTSTRAP_SAMPLES, confidence=metrics.CONFIDENCE,
            seed=_BOOTSTRAP_SEED, weighting="group_equal", numerator=point["numerator"],
            denominator=point["denominator"])
        bootstrap[name].update(raw_numerator=raw["numerator"], raw_denominator=raw["denominator"])
    primary = {"available": False, "reason": "one_preselected_primary_per_group_required"}
    if records and all(type(r.get("operating_primary")) is bool and isinstance(r["group_id"], str)
                       and r["group_id"] for r in records):
        counts = {group: sum(r["operating_primary"] for r in records if r["group_id"] == group)
                  for group in set(groups)}
        if all(count == 1 for count in counts.values()):
            chosen = [r for r in records if r["operating_primary"]]
            primary = {"available": True, "scope": "one_preselected_primary_per_group_proxy_only",
                       "sample_count": len(records), "independent_groups": len(chosen),
                       **_attempt_view(chosen, classes, allowing, blocking)}
            operation = primary["operation"]
            if not operation["unknown_gold_count"]:
                primary["joint_bounds"] = metrics._binomial_bounds(
                    int(operation["dangerous_allow_count"]), int(operation["allowing_pred_count"]), len(chosen))
                primary["prior"] = metrics.group_prior(
                    np.asarray([classes.index(r["label"]) for r in chosen], dtype=int),
                    [r["group_id"] for r in chosen])
    prior = metrics.group_prior(np.asarray([classes.index(r["label"]) for r in records], dtype=int), groups) \
        if not all_attempt["operation"]["unknown_gold_count"] else {
            "available": False, "reason": "unknown_gold_labels_retained_in_all_attempt"}
    pairs = [pair for r in records for pair in r.get("pair_diagnostics", [])]
    pair_summary = {}
    for source in ("classifier_relation", "pair_relation"):
        table = [[0] * len(classes) for _ in classes]
        known = [p for p in pairs if p["label"] in classes]
        for pair in known:
            table[classes.index(pair["label"])][classes.index(pair[source])] += 1
        correct = sum(p["label"] == p[source] for p in known)
        pair_summary[source] = {"n": len(known), "missing_reference_labels": len(pairs) - len(known),
                                "correct": correct, "errors": len(known) - correct,
                                "agreement": _rate_record(correct, len(known)),
                                "confusion": {"classes": list(classes), "matrix": table},
                                "scope": "raw_argmax" if source == "classifier_relation" else "pair_verdict"}
    oracle = [r for r in records if r.get("oracle", {}).get("available")]
    return {"all_attempt": all_attempt, "group_equal": {**equal, "bootstrap": bootstrap},
            "primary_operating": primary, "group_prior": prior,
            "classifier": {"available": bool(pairs), **pair_summary,
                           "observed": any(r.get("classifier_observed") for r in records),
                           "pair_attempts": sum(r.get("classifier_pair_attempts", 0) for r in records),
                           "missing_outputs": sum(r.get("classifier_pair_attempts", 0) for r in records) - len(pairs),
                           "note": "未观察 scorer 的 callable 不伪造 classifier；失败/缺输出另计，不伪造 confusion；pair 标签不等价 set 标签"},
            "raw_set": {"confusion": _confusion(records, classes, "raw_decision"),
                        "operation": _attempt_view(records, classes, allowing, blocking,
                                                   source="raw_decision")["operation"],
                        "scope": "adapter_output_before_production_gate_already_pair_gated_diagnostic"},
            "gated_set": {"confusion": _confusion(records, classes, "decision", failure_column=True)},
            "refs_by_role": {"scope": "all_attempt_known_expected_refs_role_specific",
                             "raw": _role_summary(records, "raw"), "gated": _role_summary(records)},
            "oracle_aggregation": {"rows": len(oracle),
                                   "agreement": _rate_record(sum(r["oracle"]["primary_agree"] for r in oracle),
                                                             len(oracle)),
                                   "set_relation_errors": sum(not r["oracle"]["primary_agree"] for r in oracle),
                                   "refs_by_role": _role_summary(oracle, "oracle"),
                                   "note": "参考 pair 标签复用真实 adapter/M；聚合错误与模型错误分列，不是 human gold"}}


def _summarize(records: list, row_errors: dict, metrics, classes: tuple,
               allowing, blocking, started: float) -> dict:
    scored = [record for record in records if record["decision"] is not None]
    gate_errored = [record for record in scored if record["error"]]
    valid = [record for record in scored if not record["error"]]
    count = len(classes)
                                                                
    y_true = np.asarray([classes.index(record["label"]) for record in valid], dtype=int)
    y_pred = np.asarray([classes.index(record["decision"]["relation"]) for record in valid], dtype=int)
    table = [[0] * count for _ in range(count)]
    for true, pred in zip(y_true.tolist(), y_pred.tolist()):
        table[true][pred] += 1
    relations: dict = {}
    errors: dict = {}
    for record in scored:
        relations[record["decision"]["relation"]] = relations.get(record["decision"]["relation"], 0) + 1
        if record["error"]:
            errors[record["error"]] = errors.get(record["error"], 0) + 1

    allow_pred = [r for r in scored if r["decision"]["relation"] in allowing]
    block_pred = [r for r in scored if r["decision"]["relation"] in blocking]
    dangerous = [r for r in allow_pred if r["label"] in blocking]
    false_block = [r for r in block_pred if r["label"] in allowing]
    gold_blocking_count = sum(1 for r in scored if r["label"] in blocking)
    gold_allowing_count = sum(1 for r in scored if r["label"] in allowing)
    operation = {
        "n": len(scored),
        "gold_blocking_count": gold_blocking_count,
        "gold_allowing_count": gold_allowing_count,
        "allowing_pred_count": len(allow_pred),
        "blocking_pred_count": len(block_pred),
        "coverage": _rate_record(len(allow_pred), len(scored)),
        "dangerous_allow_count": len(dangerous),
        "dangerous_allow_rate": _rate_record(len(dangerous), gold_blocking_count),
        "dangerous_allow_share": _rate_record(len(dangerous), len(scored)),
        "conditional_risk": _rate_record(len(dangerous), len(allow_pred)),
        "false_block_count": len(false_block),
        "false_block_rate": _rate_record(len(false_block), gold_allowing_count),
        "false_block_share": _rate_record(len(false_block), len(scored)),
        "false_block_conditional_precision": _rate_record(len(false_block), len(block_pred)),
        "gate_error_count": len(gate_errored),
        "gate_error_gold_blocking": sum(1 for r in gate_errored if r["label"] in blocking),
        "gate_error_gold_allowing": sum(1 for r in gate_errored if r["label"] in allowing),
        "rate_note": ("每 rate 带 numerator/denominator；0 分母 value=null。"
                      "dangerous_allow_rate/false_block_rate 分母为 gold 计数（metrics 口径）；"
                      "conditional_risk/false_block_conditional_precision 分母为对应预测集合"),
    }

    predicted_total = sum(len(r["refs"]["predicted"]) for r in scored)
    expected_total = sum(len(r["refs"]["expected"]) for r in scored)
    hit_total = sum(r["refs"]["hits"] for r in scored)
    unknown_total = sum(len(r["refs"]["unknown_ids"]) for r in scored)
    known_unexpected_total = sum(len(r["refs"]["known_non_expected"]) for r in scored)
    missed_total = sum(len(r["refs"]["missed_expected"]) for r in scored)
    refs = {
        "rows_with_expected": sum(1 for r in scored if r["refs"]["expected"]),
        "predicted_total": predicted_total,
        "expected_total": expected_total,
        "hits": hit_total,
        "precision": _rate(hit_total, predicted_total),
        "recall": _rate(hit_total, expected_total),
        "unknown_total": unknown_total,
        "unknown_rate": _rate(unknown_total, predicted_total),
        "known_non_expected_total": known_unexpected_total,
        "missed_expected_total": missed_total,
        "quality_note": ("precision/recall 分开读；unknown=0 只说明产出集合不含未知 id，"
                         "不外推引用语义质量——known 非 expected 与 expected 漏标各自独立计数"),
    }

    samples = int(metrics.BOOTSTRAP_SAMPLES)
    confidence = float(metrics.CONFIDENCE)

    def macro_of(group: list):
        if not group:
            return None
        true = np.asarray([classes.index(r["label"]) for r in group], dtype=int)
        pred = np.asarray([classes.index(r["decision"]["relation"]) for r in group], dtype=int)
        return float(metrics.macro_f1(true, pred))

    def primary_of(group: list):
        return _rate(sum(1 for r in group if r["primary_agree"]), len(group))

    def dangerous_count_of(group: list):
        return sum(1 for r in group
                   if r["decision"]["relation"] in allowing and r["label"] in blocking)

    def dangerous_rate_of(group: list):                           
        return _rate(dangerous_count_of(group),
                     sum(1 for r in group if r["label"] in blocking))

    def conditional_risk_of(group: list):                  
        return _rate(dangerous_count_of(group),
                     sum(1 for r in group if r["decision"]["relation"] in allowing))

    def false_block_count_of(group: list):
        return sum(1 for r in group
                   if r["decision"]["relation"] in blocking and r["label"] in allowing)

    def false_block_rate_of(group: list):                         
        return _rate(false_block_count_of(group),
                     sum(1 for r in group if r["label"] in allowing))

    def false_block_precision_of(group: list):               
        return _rate(false_block_count_of(group),
                     sum(1 for r in group if r["decision"]["relation"] in blocking))

    def _risk_interval(metric, numerator, denominator):
        record = _cluster_interval(scored, metric, samples=samples, confidence=confidence,
                                   seed=_BOOTSTRAP_SEED)
        if record is None:
            return None
        record["numerator"] = int(numerator)
        record["denominator"] = int(denominator)
        record["zero_errors"] = int(numerator) == 0
        bound_fn = getattr(metrics, "_rare_upper_bound", None)
        record["rare_upper_bound"] = (
            float(bound_fn(int(denominator), confidence))
            if int(numerator) == 0 and int(denominator) > 0 and callable(bound_fn) else None)
        if record["rare_upper_bound"] is not None:
            record["rare_bound_note"] = ("零错误：保守单侧二项上界；忽略 group 簇相关，仅作指示，"
                                         "不得把 percentile CI 的 [0,0] 当安全证据")
        return record

    bootstrap = {
        "group_clusters": len({r["group_id"] for r in scored}),
        "samples": samples,
        "confidence": confidence,
        "seed": _BOOTSTRAP_SEED,
        "scope": "与 operation 同命名同口径；invalid=零分母复制数（不按 0 计入）",
        "macro_f1": _cluster_interval(valid, macro_of, samples=samples, confidence=confidence,
                                      seed=_BOOTSTRAP_SEED),
        "primary_agreement": _cluster_interval(valid, primary_of, samples=samples,
                                               confidence=confidence, seed=_BOOTSTRAP_SEED),
        "dangerous_allow_rate": _risk_interval(dangerous_rate_of, len(dangerous), gold_blocking_count),
        "conditional_risk": _risk_interval(conditional_risk_of, len(dangerous), len(allow_pred)),
        "false_block_rate": _risk_interval(false_block_rate_of, len(false_block), gold_allowing_count),
        "false_block_conditional_precision": _risk_interval(false_block_precision_of,
                                                            len(false_block), len(block_pred)),
    }

    trace = [[record["index"], call_index, latency]
             for record in records for call_index, latency in enumerate(record["latency_ms"], 1)]
    pairs = [record["scored_pairs"] for record in records if record["scored_pairs"] is not None]
    extended = _extended_summary(records, metrics, classes, allowing, blocking)
    semantic = sum(bool(r.get("semantic_uncertain")) for r in records)
    return {
        "row_count": len(records),
        "row_error_rows": sum(bool(r["row_error"]) for r in records),
        "scored_rows": len(scored), "gate_error_rows": len(gate_errored), "valid_rows": len(valid),
        "semantic_uncertain_rows": semantic,
        "system_failure_rows": int(extended["all_attempt"]["failed_count"]),
        "classification_scope": "secondary valid_rows（无 gate error）；主口径为 all_attempt.correct_completed/N",
        "row_errors": dict(sorted(row_errors.items())),
        "gate_errors": {
            "count": len(gate_errored), "by_code": dict(sorted(errors.items())),
            "gold_blocking": sum(1 for r in gate_errored if r["label"] in blocking),
            "gold_allowing": sum(1 for r in gate_errored if r["label"] in allowing),
            "semantic_uncertain": semantic,
            "note": "旧 error_code 计数含语义弃权；系统失败见 system_failure_rows，不把 uncertain 全当失败",
        },
        "predictor_calls": sum(record["calls"] for record in records),
        "failed_predictor_calls": sum(len(r.get("call_failures", [])) for r in records),
        "scored_pairs": sum(pairs) if pairs else None,
        "relations": dict(sorted(relations.items())), "errors": dict(sorted(errors.items())),
        "agreement": {
            "primary": _rate(sum(1 for r in valid if r["primary_agree"]), len(valid)),
            "acceptable": _rate(sum(1 for r in valid if r["acceptable_agree"]), len(valid)),
            "n": len(valid), "scope": "secondary valid_rows（无 gate error）",
        },
        "confusion": {"classes": list(classes), "matrix": table,
                      "scope": "secondary valid_rows（无 gate error）"},
        "macro_f1": macro_of(valid), "macro_f1_scope": "secondary_valid_only",
        "operation": extended["all_attempt"]["operation"],
        "secondary": {"operation_scored_only": operation, "refs_union_scored_only": refs,
                      "bootstrap_scored_only": bootstrap, "scope": "legacy_secondary_diagnostic"},
        "refs": {**refs, "scope": "legacy_secondary_union_scored_only_use_refs_by_role"},
        "bootstrap": {**bootstrap, "scope": "legacy_secondary_scored_only_row_weighted"},
        **extended,
        "latency_ms": {
            "total": round(sum(item[2] for item in trace), 3),
            "max": max((item[2] for item in trace), default=None), "trace": trace,
            "scope": "all predictor attempts including exceptions; finally recorded",
        },
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def evaluate_sets(predictor, rows, *, oracle_aggregation: bool = False) -> dict:
    """真实 M 完整回放；all-attempt N 含坏行与失败，valid-only 明确为 secondary。

    语义 uncertain 是已完成的弃权判断，调用/输出失败不是。提供由观察层捕获的 classifier
    与 pair gate、set raw/gated 与角色引用统计；oracle 只复用真实 adapter 做代理标签诊断。
    """
    metrics = _metrics_module()
    classes = tuple(metrics.CLASSES)
    allowing = frozenset(metrics.ALLOWING_CLASSES)
    blocking = frozenset(metrics.BLOCKING_CLASSES)
    row_error = _row_error_type()
    records: list = []
    row_errors: dict = {}
    started = time.perf_counter()
    classifier_trace = getattr(predictor, "classifier_trace", [])
    for index, row in enumerate(rows):
        row_started = time.perf_counter()
        row_id = row.get("id") if isinstance(row, dict) else None
        calls: list = []
        trace_start = len(classifier_trace)

        def wrapped(request, _calls=calls):
            begin = time.perf_counter()
            call = {"raw": None, "failure": None}
            try:
                call["raw"] = predictor(request)
                return call["raw"]
            except Exception as exc:
                call["failure"] = type(exc).__name__
                raise
            finally:
                call["latency_ms"] = round((time.perf_counter() - begin) * 1000.0, 3)
                _calls.append(call)

        try:
            candidate, existing, evidence, cfg, meta = _prepare(row, classes)
        except (row_error, RowContractError) as exc:
            code = str(getattr(exc, "code", None) or type(exc).__name__)
            row_errors[code] = row_errors.get(code, 0) + 1
            phenomena = row.get("phenomena") if isinstance(row, dict) else None
            try:
                expected = _expected_refs_by_role(row) if isinstance(row, dict) else None
            except RowContractError:
                expected = None
            known = {"related_ids": set(), "evidence_refs": set()}
            scores = _role_scores(None, expected, known) if expected is not None else None
            records.append({
                "index": index, "row_id": row_id,
                "group_id": row.get("group_id") if isinstance(row, dict) else None,
                "label": row.get("label") if isinstance(row, dict) else None,
                "label_origin": None, "acceptable": None, "row_error": code,
                "rules": None, "predictions": [], "decision": None, "error": None,
                "raw_decision": None, "completed": False, "semantic_uncertain": False,
                "operating_primary": phenomena.get("operating_primary") if isinstance(phenomena, dict) else None,
                "related": [], "refs": None, "refs_by_role": {"raw": scores, "gated": scores} if scores else None,
                "primary_agree": None, "acceptable_agree": None, "pair_diagnostics": [],
                "calls": 0, "call_failures": [], "latency_ms": [], "scored_pairs": 0,
                "elapsed_ms": round((time.perf_counter() - row_started) * 1000.0, 3),
            })
            continue
        before = _scored_pairs(predictor)
        assessment_failure = None
        try:
            decision = M.assess_relations(candidate, existing, evidence, cfg, predictor=wrapped)
        except Exception as exc:
            decision = None
            assessment_failure = type(exc).__name__
        finally:
            elapsed = round((time.perf_counter() - row_started) * 1000.0, 3)
        after = _scored_pairs(predictor)
        classifier_calls = classifier_trace[trace_start:]
        raw = _raw_decision(calls, classes)
        completed = _completed(decision, calls, classifier_calls)
        known = {"related_ids": {item["id"] for item in existing},
                 "evidence_refs": {item["id"] for item in evidence}}
        expected_roles = meta.get("expected_refs_by_role") or {
            role: [ref for ref in meta["expected_refs"] if ref in identifiers]
            for role, identifiers in known.items()}
        predicted = decision["related_ids"] + decision["evidence_refs"] if decision else []
        refs = _reference_scores(predicted, meta["expected_refs"], known["related_ids"] | known["evidence_refs"])
        labels = _pair_labels(row, existing, meta["label"], classes)
        pair_diagnostics = [{**pair, "label": labels.get(pair["rule_id"]) if labels is not None else None}
                            for call in classifier_calls for pair in call["pairs"]]
        oracle = _oracle_probe(labels, candidate, existing, evidence, cfg, classes) \
            if oracle_aggregation else {"available": False, "reason": "not_requested"}
        if oracle["available"]:
            oracle["primary_agree"] = oracle["decision"]["relation"] == meta["label"]
            oracle["acceptable_agree"] = oracle["decision"]["relation"] in meta["acceptable"]
        records.append({
            "index": index, "row_id": row_id, "group_id": meta["group_id"],
            "label": meta["label"], "label_origin": meta["label_origin"],
            "acceptable": meta["acceptable"], "row_error": None, "rules": len(existing),
            "predictions": [call["raw"] for call in calls],
            "decision": decision, "error": decision["error_code"] if decision else "assessment_failed",
            "raw_decision": raw, "completed": completed,
            "semantic_uncertain": completed and decision["relation"] == "uncertain",
            "operating_primary": meta.get("operating_primary"), "pair_diagnostics": pair_diagnostics,
            "classifier_errors": [call["error"] for call in classifier_calls if call["error"]],
            "classifier_pair_attempts": sum(call["pair_count"] for call in classifier_calls),
            "classifier_observed": hasattr(predictor, "classifier_trace"),
            "oracle": oracle, "assessment_failure": assessment_failure,
            "related": list(decision["related_ids"]) if decision else [], "refs": refs,
            "refs_by_role": {"raw": _role_scores(raw, expected_roles, known),
                             "gated": _role_scores(decision, expected_roles, known),
                             "oracle": _role_scores(oracle["decision"], expected_roles, known)
                             if oracle["available"] else None},
            "primary_agree": decision["relation"] == meta["label"] if decision else False,
            "acceptable_agree": decision["relation"] in meta["acceptable"] if decision else False,
            "calls": len(calls), "call_failures": [call["failure"] for call in calls if call["failure"]],
            "latency_ms": [call["latency_ms"] for call in calls], "elapsed_ms": elapsed,
            "scored_pairs": None if before is None or after is None else after - before,
        })
    summary = _summarize(records, row_errors, metrics, classes, allowing, blocking, started)
    return {"rows": records, "summary": summary, "semantics": _SEMANTICS}


def _operating_threshold(threshold_arg, diagnostic: bool, model_dir: Path) -> dict:
    """默认须 meta 数值 threshold 且 threshold_feasible 严格 true；诊断阈值不声称已校准。"""
    meta_path = model_dir / "meta.json"
    meta = None
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            meta = None
    meta_threshold = None
    if isinstance(meta, dict) and meta.get("threshold") is not None:
        meta_threshold = adapter._threshold(meta["threshold"])
    feasible = meta.get("threshold_feasible") if isinstance(meta, dict) else None
    if threshold_arg is not None:
        threshold_arg = adapter._threshold(threshold_arg)
        if not diagnostic:
            raise ValueError("--threshold 仅用于诊断：请与 --diagnostic 同用，"
                             "或省略以使用 meta 的校准阈值")
    record = {
        "requested_threshold": threshold_arg, "meta_threshold": meta_threshold,
        "meta_path": str(meta_path) if meta_path.is_file() else None,
        "meta_threshold_feasible": feasible,
    }
    if diagnostic:
        if threshold_arg is not None:
            return {**record, "threshold": threshold_arg, "threshold_source": "cli",
                    "calibrated": False, "diagnostic": True,
                    "diagnostic_note": "显式诊断阈值，不是已校准运行点"}
        if meta_threshold is not None:
            return {**record, "threshold": meta_threshold, "threshold_source": "meta",
                    "calibrated": False, "diagnostic": True,
                    "diagnostic_note": "meta 数值阈值仅作分类诊断，不声明已校准操作"}
        raise ValueError("--diagnostic 需要显式数值阈值（--threshold N）或 meta.json 的数值阈值："
                         "未校准（无阈值）不产生运行点")
    if meta_threshold is None or feasible is not True:
        raise ValueError("meta.json 无经独立审计的可行阈值（threshold 数值且 threshold_feasible=true 才可）："
                         "拒绝已校准操作；如需分类诊断请显式 --diagnostic --threshold N")
    return {**record, "threshold": meta_threshold, "threshold_source": "meta", "calibrated": True,
            "diagnostic": False}


def _prediction_row(record: dict) -> dict:
    decision = record["decision"]
    return {
        "index": record["index"], "id": record["row_id"], "group_id": record["group_id"],
        "label": record["label"], "label_origin": record["label_origin"],
        "acceptable": record["acceptable"], "row_error": record["row_error"],
        "relation": decision["relation"] if decision else None,
        "error": record["error"],
        "related_ids": list(decision["related_ids"]) if decision else [],
        "evidence_refs": list(decision["evidence_refs"]) if decision else [],
        "primary_agree": record["primary_agree"],
        "acceptable_agree": record["acceptable_agree"],
        "completed": record["completed"], "semantic_uncertain": record["semantic_uncertain"],
        "operating_primary": record["operating_primary"],
        "raw_relation": record["raw_decision"]["relation"] if record["raw_decision"] else None,
        "pair_diagnostics": record["pair_diagnostics"], "refs_by_role": record["refs_by_role"],
        "oracle": record.get("oracle"), "call_failures": record["call_failures"],
        "calls": record["calls"], "latency_ms": record["latency_ms"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="关系模型 set 级离线回放与对照评测（研究专用）")
    parser.add_argument("--model-dir", required=True, help="train 产出的模型目录（features.load_scorer）")
    parser.add_argument("--data", required=True, help="schema JSONL 数据行")
    parser.add_argument("--out", required=True, help="评测报告 JSON 输出路径（如 artifacts/memory-relations/…）")
    parser.add_argument("--threshold", type=float, default=None,
                        help="显式诊断阈值 [0,1]；必须与 --diagnostic 同用")
    parser.add_argument("--diagnostic", action="store_true",
                        help="分类诊断：允许显式 --threshold；不声明已校准操作")
    parser.add_argument("--oracle-aggregation", action="store_true",
                        help="用完整参考 pair_labels 经真实 adapter/M 探针；仅解释聚合误差")
    args = parser.parse_args(argv)
    from . import schema

    try:
        operation = _operating_threshold(args.threshold, args.diagnostic, Path(args.model_dir))
        predictor = make_evaluation_predictor(_load_scorer(args.model_dir), operation["threshold"])
        report = evaluate_sets(predictor, schema.load_rows(args.data),
                               oracle_aggregation=args.oracle_aggregation)
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    report["operation"] = operation
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = out.with_name(out.stem + ".predictions.jsonl")
    report["predictions_file"] = str(predictions_path)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    predictions_path.write_text(
        "".join(json.dumps(_prediction_row(record), ensure_ascii=False) + "\n"
                for record in report["rows"]),
        encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
