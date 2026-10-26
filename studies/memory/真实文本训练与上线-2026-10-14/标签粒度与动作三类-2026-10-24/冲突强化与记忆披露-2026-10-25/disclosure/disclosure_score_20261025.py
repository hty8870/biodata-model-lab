                      
"""第 4 线确定性评分与审计：闭集命题 oracle、family-cluster 统计、质量—成本前沿。

原则：
- 判分只看答案的**结构化字段**（decision/value/applied_condition/applied_exception）
  与引用是否落在必需组；不做关键词命中率、不做语义相似度近似。
- 0 分母输出 null，不用 0 或 1 填充；空检索、超预算、陈旧视图、位置都有专门断言。
- 未回答的行保留在表里并标 `answered=false`，行为分母不得把它算成失败以外的任何事；
  报告同时给 n/N 与未回答数。

命令：
  python disclosure_score_20261025.py selftest
  python disclosure_score_20261025.py analyze            # 只算披露成本与审计
  python disclosure_score_20261025.py analyze --answers out/answers.jsonl
  python disclosure_score_20261025.py analyze --synthetic  # 仅自检评分机器，产物单独命名
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from disclosure_schema_20261025 import (              
    DECISIONS, OUT, PHASE, json_write, jsonl_read, jsonl_write, normalize_value,
    parse_answer, project_full, sha256,
)

GROUPS_PATH = OUT.parent / "data" / "groups.jsonl"
ARMS = ("P-full", "A-full", "P-retrieve", "A-retrieve", "A-graph", "A-compiled",
        "A-grouped")
PAIRS = (("A-full", "A-grouped"), ("P-full", "A-grouped"), ("A-full", "P-full"),
         ("A-full", "A-retrieve"), ("P-retrieve", "A-retrieve"),
         ("A-retrieve", "A-graph"), ("A-retrieve", "A-compiled"))
CONTRAST_LABELS = {
    ("A-full", "A-grouped"): "A-full - A-grouped（同一原子库的平铺呈现效应）",
    ("P-full", "A-grouped"): "P-full - A-grouped（逐字控制：应为 0，非行为估计）",
    ("A-full", "P-full"): "A-full - P-full（复合/原子全量呈现总效应，非纯存储效应）",
}
RETRIEVE_ARMS = ("P-retrieve", "A-retrieve", "A-graph")
SEED = 20261025
BOOTSTRAP = 1000


                                                                  

def group_of_item(item_id: str) -> str:
    """条目 id 约定：`<Fxx-Gy>-P` / `<Fxx-Gy>-Az` / 编译卡 `C-<Fxx-Gy>`。

    编译卡去掉 `C-` 后本身就是组 id；存储条目去掉最后一段后缀即组 id。
    该约定由数据校验与单元测试保证。
    """
    if item_id.startswith("C-"):
        return item_id[2:]
    return item_id.rsplit("-", 1)[0]


def _norm(value):
    return normalize_value(value)


def _check(answer: dict, spec: dict) -> bool:
    name = spec["field"]
    if name == "cited_group":
        got = [group_of_item(c) for c in (answer.get("cited_item_ids") or [])
               if isinstance(c, str)]
    else:
        got = answer.get(name)
    if "isnull" in spec:
        return got in (None, "", [])
    if "contains" in spec:
        want = _norm(spec["contains"])
        if isinstance(got, list):
            return want in [_norm(v) for v in got]
        return isinstance(got, str) and want in _norm(got)
    if "not_equals" in spec:
        return _norm(got) != _norm(spec["not_equals"])
    return _norm(got) == _norm(spec["equals"])


def score_answer(value, row: dict) -> dict:
    """value 可为模型原文（str）或已解析 dict；返回逐项命题结果与失败标记。"""
    gold = row["gold"]
    expected = gold["expected"]
    if isinstance(value, dict):
        answer, parse_reason = dict(value), "ok"
    else:
        answer, parse_reason = parse_answer(value if isinstance(value, str) else "")
    parse_ok = answer is not None
    answer = answer or {}
    decision = answer.get("decision")
    cited = answer.get("cited_item_ids")
    cited = [c for c in cited if isinstance(c, str)] if isinstance(cited, list) else []
    disclosed = set(row["item_ids"])
    valid_citations = [c for c in cited if c in disclosed]
    invented_citations = [c for c in cited if c not in disclosed]
    required_groups = set(gold["required_groups"] or [])
    cited_required = sorted({group_of_item(c) for c in valid_citations} & required_groups)

    decision_ok = isinstance(decision, str) and decision == expected["decision"]
    if expected["value"] is None:
        value_ok = answer.get("value") in (None, "", [])
    elif expected["value_mode"] == "set":
        value_ok = isinstance(answer.get("value"), list) and \
            _norm(answer["value"]) == _norm(expected["value"])
    else:
        value_ok = _norm(answer.get("value")) == _norm(expected["value"])
    cond_ok = _norm(answer.get("applied_condition")) == _norm(expected["applied_condition"])
    exc_ok = _norm(answer.get("applied_exception")) == _norm(expected["applied_exception"])
    citation_ok = bool(cited_required) if required_groups else True

    props = {fid: _check(answer, spec) for fid, spec in gold["fact_checks"].items()}
    required_ids = [fid for fid in gold["required_facts"] if fid in props]
    context_ids = [fid for fid in gold["necessary_context"] if fid in props]
    required_rate = round(sum(props[f] for f in required_ids) / len(required_ids), 4) \
        if required_ids else None
    context_rate = round(sum(props[f] for f in context_ids) / len(context_ids), 4) \
        if context_ids else None
    stale_value = gold.get("stale_value")
    stale_error = stale_value is not None and _norm(answer.get("value")) == _norm(stale_value)
    hallucination = (expected["decision"] == "no_memory"
                     and (decision != "no_memory" or answer.get("value") not in (None, "", [])))
    spurious_citation = expected["decision"] == "no_memory" and bool(valid_citations)

    return {
        "parse_ok": parse_ok, "parse_reason": parse_reason,
        "decision": decision, "value": answer.get("value"),
        "applied_condition": answer.get("applied_condition"),
        "applied_exception": answer.get("applied_exception"),
        "decision_ok": decision_ok, "value_ok": value_ok,
        "condition_ok": cond_ok, "exception_ok": exc_ok,
        "citation_ok": citation_ok, "citation_required": bool(required_groups),
        "cited_required_groups": cited_required,
        "invented_citation_ids": invented_citations,
        "exact_success": bool(parse_ok and decision_ok and value_ok and cond_ok
                              and exc_ok and citation_ok),
        "required_proposition_rate": required_rate,
        "necessary_context_rate": context_rate,
        "all_required_propositions": (all(props[f] for f in required_ids) if required_ids else None),
        "stale_error": stale_error,
        "hallucination": hallucination,
        "spurious_citation": spurious_citation,
        "proposition_checks": props,
    }


def ideal_answer(row: dict) -> dict:
    """机器自检用的「已知正确」答案：只从 gold 与已披露条目构造。"""
    gold = row["gold"]
    expected = gold["expected"]
    required_groups = set(gold["required_groups"] or [])
    candidates = [iid for iid in row["item_ids"]
                  if group_of_item(iid) in required_groups]
    wanted = set(row["metrics"]["gold_required_item_ids"])
    cited = sorted([iid for iid in candidates if iid in wanted])[:1] or candidates[:1]
    return {"decision": expected["decision"], "value": expected["value"],
            "cited_item_ids": cited,
            "applied_condition": expected["applied_condition"],
            "applied_exception": expected["applied_exception"]}


                                                               

def score_rows(rows: list[dict], answers: dict[str, object]) -> tuple[list[dict], dict]:
    scored, missing = [], []
    for row in rows:
        rid = row_id(row)
        if rid not in answers:
            missing.append(rid)
            continue
        result = score_answer(answers[rid], row)
        scored.append({
            "row_id": rid, "task_id": row["task_id"], "family_id": row["family_id"],
            "split": row["split"], "phenomenon": row["phenomenon"], "variant": row["variant"],
            "arm": row["arm"], "scale": row["scale"], "position": row["position"],
            "chars": row["chars"], "token_estimate": row["token_estimate"],
            "build_latency_ms": row["build_latency_ms"],
            "item_count": row["metrics"]["item_count"],
            "logical_groups": row["metrics"]["logical_groups"],
            "required_recall": row["metrics"]["required_recall"],
            "necessary_context_recall": row["metrics"]["necessary_context_recall"],
            "irrelevant_items": row["metrics"]["irrelevant_items"],
            "budget_chars": row["budget_chars"], "snapshot_ok": row["snapshot_ok"],
            "expected_decision": row["gold"]["expected"]["decision"],
            "observation_kind": row.get("observation_kind", "independent"),
            "independent": row.get("observation_kind", "independent") != "byte_identical_reuse",
            "reused_from": row.get("reuse_of_arm"),
            "real_arm_result": bool(row.get("real_arm_result")),
            "compiler_backend": row.get("compiler_backend"),
            "graph_backend": row.get("graph_backend"),
            **result,
        })
    return scored, {"answered": len(scored), "unanswered": len(missing),
                    "unanswered_row_ids": missing[:50]}


def row_id(row: dict) -> str:
    return f"{row['task_id']}__{row['arm']}__S{row['scale']}__{row['variant']}"


                                                              

def rate_block(values: list[bool]) -> dict:
    n = len(values)
    hits = sum(1 for v in values if v)
    return {"n": n, "hits": hits, "rate": round(hits / n, 4) if n else None}


def summarize(scored: list[dict]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in scored:
        buckets[(row["variant"], row["split"], row["scale"], row["arm"])].append(row)
    out = []
    for key in sorted(buckets, key=lambda k: (k[0], k[1], k[2], ARMS.index(k[3]))):
        all_rows = buckets[key]
        rows = [r for r in all_rows if r.get("independent", True)]
        cell = {"variant": key[0], "split": key[1], "scale": key[2], "arm": key[3],
                "n": len(rows), "reuse_rows": len(all_rows) - len(rows)}
        if not rows:
            cell["exact_success"] = rate_block([])
            cell["required_proposition_rate"] = {"n": 0, "mean": None}
            cell["necessary_context_rate"] = {"n": 0, "mean": None}
            cell["hallucination"] = rate_block([])
            cell["stale_error"] = rate_block([])
            cell["parse_failure"] = rate_block([])
            cell["cost"] = {"chars_mean": None, "tokens_est_mean": None,
                            "latency_ms_mean": None, "items_mean": None, "groups_mean": None}
            cell["required_recall"] = {"n": 0, "mean": None}
            out.append(cell)
            continue
        cell["exact_success"] = rate_block([r["exact_success"] for r in rows])
        for field in ("required_proposition_rate", "necessary_context_rate"):
            vals = [r[field] for r in rows if r[field] is not None]
            cell[field] = {"n": len(vals), "mean": round(sum(vals) / len(vals), 4) if vals else None}
        cell["hallucination"] = rate_block([r["hallucination"] for r in rows
                                            if r["expected_decision"] == "no_memory"])
        cell["stale_error"] = rate_block([r["stale_error"] for r in rows])
        cell["parse_failure"] = rate_block([not r["parse_ok"] for r in rows])
        cell["cost"] = {
            "chars_mean": round(statistics.fmean(r["chars"] for r in rows), 1),
            "tokens_est_mean": round(statistics.fmean(r["token_estimate"] for r in rows), 1),
            "latency_ms_mean": round(statistics.fmean(r["build_latency_ms"] for r in rows), 3),
            "items_mean": round(statistics.fmean(r["item_count"] for r in rows), 2),
            "groups_mean": round(statistics.fmean(r["logical_groups"] for r in rows), 2),
        }
        disclosure_recall = [r["required_recall"] for r in rows if r["required_recall"] is not None]
        cell["required_recall"] = {
            "n": len(disclosure_recall),
            "mean": round(sum(disclosure_recall) / len(disclosure_recall), 4)
            if disclosure_recall else None,
        }
        out.append(cell)
    return out


def _usable(row: dict) -> bool:
    """A-grouped 的 reuse 行不是独立观测，不进配对估计。"""
    return row.get("independent", True)


def _pairs_for(scored: list[dict], arm_a: str, arm_b: str) -> list[tuple[str, float, str]]:
    key = lambda r: (r["task_id"], r["variant"], r["scale"])
    a = {key(r): r for r in scored if r["arm"] == arm_a and _usable(r)}
    b = {key(r): r for r in scored if r["arm"] == arm_b and _usable(r)}
    out = []
    for k in sorted(set(a) & set(b)):
        out.append((a[k]["family_id"], float(a[k]["exact_success"]) - float(b[k]["exact_success"]),
                    f"{k[0]}@S{k[2]}"))
    return out


def cluster_bootstrap(pairs: list[tuple[str, float, str]], *, seed: int = SEED,
                      draws: int = BOOTSTRAP) -> dict:
    if not pairs:
        return {"n_pairs": 0, "n_families": 0, "mean_diff": None,
                "ci95": [None, None], "direction": "no_data"}
    by_family: dict[str, list[float]] = defaultdict(list)
    for family, diff, _ in pairs:
        by_family[family].append(diff)
    families = sorted(by_family)
    point = sum(d for _, d, _ in pairs) / len(pairs)
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        draw = [families[rng.randrange(len(families))] for _ in families]
        vals = [d for fam in draw for d in by_family[fam]]
        means.append(sum(vals) / len(vals))
    means.sort()
    lo = means[int(0.025 * draws)]
    hi = means[int(0.975 * draws) - 1]
    direction = "improves" if lo > 0 else "worse" if hi < 0 else "uncertain"
    return {"n_pairs": len(pairs), "n_families": len(families),
            "mean_diff": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "direction": direction, "seed": seed, "draws": draws,
            "cluster": "task family"}


def _pairs_cross_variant(scored: list[dict], arm_a: str, variant_a: str,
                         arm_b: str, variant_b: str) -> list[tuple[str, float, str]]:
    key = lambda r: (r["task_id"], r["scale"])
    a = {key(r): r for r in scored if r["arm"] == arm_a and r["variant"] == variant_a}
    b = {key(r): r for r in scored if r["arm"] == arm_b and r["variant"] == variant_b}
    out = []
    for k in sorted(set(a) & set(b)):
        out.append((a[k]["family_id"], float(a[k]["exact_success"]) - float(b[k]["exact_success"]),
                    f"{k[0]}@S{k[1]}"))
    return out


def contrasts(scored: list[dict]) -> list[dict]:
    out = []
    for arm_a, arm_b in PAIRS:
        pairs = _pairs_for(scored, arm_a, arm_b)
        label = CONTRAST_LABELS.get((arm_a, arm_b), f"{arm_a} - {arm_b}")
        out.append({"contrast": label, "a": arm_a, "b": arm_b,
                    **cluster_bootstrap(pairs)})
    matched = _pairs_cross_variant(scored, "A-graph", "main", "A-retrieve", "costmatched")
    out.append({"contrast": "A-graph - A-retrieve(cost-matched)",
                "a": "A-graph", "b": "A-retrieve(cost-matched)",
                "note": "检索臂只用 graph 实际字符数；用于排除『graph 更省』被读成『graph 更准』",
                **cluster_bootstrap(matched)})
    return out


def frontier(summary_rows: list[dict]) -> list[dict]:
    """质量—成本前沿：不在单一加权总分里藏成本。"""
    cells = [c for c in summary_rows if c["variant"] == "main" and c["exact_success"]["rate"] is not None]
    rows = []
    for cell in cells:
        rows.append({"arm": cell["arm"], "split": cell["split"], "scale": cell["scale"],
                     "chars_mean": cell["cost"]["chars_mean"],
                     "tokens_est_mean": cell["cost"]["tokens_est_mean"],
                     "latency_ms_mean": cell["cost"]["latency_ms_mean"],
                     "success_rate": cell["exact_success"]["rate"], "n": cell["n"]})
    for row in rows:
        dominated = [other for other in rows
                     if other["split"] == row["split"] and other["scale"] == row["scale"]
                     and other["arm"] != row["arm"]
                     and other["chars_mean"] <= row["chars_mean"]
                     and other["success_rate"] >= row["success_rate"]
                     and (other["chars_mean"] < row["chars_mean"]
                          or other["success_rate"] > row["success_rate"])]
        row["pareto_dominated"] = bool(dominated)
        row["dominated_by"] = sorted({o["arm"] for o in dominated})
    return rows


def cost_table(rows: list[dict]) -> list[dict]:
    """行为未采集时也能出的成本/披露质量表；与行为分表分开，不混算。"""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["variant"], row["split"], row["scale"], row["arm"])].append(row)
    out = []
    for key in sorted(buckets, key=lambda k: (k[0], k[1], k[2], ARMS.index(k[3]))):
        cell_rows = buckets[key]
        recall = [r["metrics"]["required_recall"] for r in cell_rows
                  if r["metrics"]["required_recall"] is not None]
        ctx = [r["metrics"]["necessary_context_recall"] for r in cell_rows
               if r["metrics"]["necessary_context_recall"] is not None]
        out.append({
            "variant": key[0], "split": key[1], "scale": key[2], "arm": key[3],
            "n": len(cell_rows),
            "chars_mean": round(statistics.fmean(r["chars"] for r in cell_rows), 1),
            "tokens_est_mean": round(statistics.fmean(r["token_estimate"] for r in cell_rows), 1),
            "latency_ms_mean": round(statistics.fmean(r["build_latency_ms"] for r in cell_rows), 3),
            "items_mean": round(statistics.fmean(r["metrics"]["item_count"] for r in cell_rows), 2),
            "groups_mean": round(statistics.fmean(r["metrics"]["logical_groups"] for r in cell_rows), 2),
            "irrelevant_items_mean": round(statistics.fmean(
                r["metrics"]["irrelevant_items"] for r in cell_rows), 2),
            "budget_fill_mean": round(statistics.fmean(
                (r["chars"] / r["budget_chars"]) if r["budget_chars"] else 1.0
                for r in cell_rows), 4),
            "required_recall_mean": round(sum(recall) / len(recall), 4) if recall else None,
            "necessary_context_recall_mean": round(sum(ctx) / len(ctx), 4) if ctx else None,
            "empty_disclosures": sum(1 for r in cell_rows if not r["item_ids"]),
            "degraded": sum(1 for r in cell_rows if not r["snapshot_ok"]),
        })
    return out


                                                              

def _item_maps(groups: list[dict], views: dict | None = None,
               stale_groups: list[dict] | None = None) -> dict:
    """条目索引：主视图 + 每个 staged update 各自一份 generation-1 视图。

    编译视图按 split 存储（dict[split] -> view）；卡 id 全库唯一，合并成一张主索引。
    """
    cards = []
    for view in (views or {}).values():
        cards.extend(view.get("cards") or [])

    def build(extra: list[dict]) -> dict[str, dict]:
        out = {}
        for group in groups + extra:
            out[group["compound"]["id"]] = group["compound"]
            for atom in group["atoms"]:
                out[atom["id"]] = atom
        out.update({c["id"]: c for c in cards})
        return out

    return {"main": build([]),
            "by_update": {g["group_id"]: build([g]) for g in (stale_groups or [])}}


def audit(rows: list[dict], groups: list[dict], banks: list[dict], *,
          views: dict | None = None, stale_groups: list[dict] | None = None) -> dict:
    from disclosure_schema_20261025 import estimate_tokens
    errors, checks, failed = [], [], []
    maps = _item_maps(groups, views, stale_groups)
    all_items = {k: v for k, v in maps["main"].items()}
    for mapping in maps["by_update"].values():
        all_items.update(mapping)
    bank_by_id = {b["bank_id"]: b for b in banks}
    version_by_arm = {arm: ("compound" if arm.startswith("P-") else
                            "grouped" if arm == "A-grouped" else "atomic") for arm in ARMS}

    def check(code: str, errs: list[str]) -> None:
        checks.append(code)
        if errs:
            failed.append({"code": code, "n": len(errs), "examples": errs[:5]})
        errors.extend(errs)

    bad_rebuild, bad_chars, bad_budget, bad_full, bad_scale = [], [], [], [], []
    bad_arm, bad_gold, bad_position, bad_token, bad_version = [], [], [], [], []
    for row in rows:
        gen = row.get("store_generation", 0)
        if gen == 0:
            mapping = maps["main"]
        else:
            target = (row["gold"]["required_groups"] or [None])[0]
            mapping = maps["by_update"].get(target)
        if mapping is None:
            bad_rebuild.append(row_id(row) + f":missing_generation_{gen}")
            continue
        rebuilt = project_full([mapping[i] for i in row["item_ids"]], PHASE)
        if rebuilt != row["disclosure"]:
            bad_rebuild.append(row_id(row))
        if len(row["disclosure"]) != row["chars"]:
            bad_chars.append(row_id(row))
        if row["arm"] in RETRIEVE_ARMS and row["chars"] > (row["budget_chars"] or 0):
            bad_budget.append(row_id(row))
        bank = bank_by_id[row["bank_id"]]
        if len(bank["group_ids"]) != row["scale"] or len(row["group_ids"]) != row["scale"]:
            bad_scale.append(row_id(row))
        if row["position"] != bank["position"]:
            bad_position.append(row_id(row))
        if row["arm"] in ("P-full", "A-full", "A-grouped"):
            want = sum(1 for gid in row["group_ids"]
                       for _ in ([all_items[gid + "-P"]] if row["arm"] in ("P-full", "A-grouped")
                                 else [i for i in all_items if i.startswith(gid + "-A")]))
            if len(row["item_ids"]) != want:
                bad_full.append(row_id(row))
        if any(token in row["disclosure"] for token in ARMS):
            bad_arm.append(row_id(row))
        gold_items = set(row["metrics"]["gold_required_item_ids"])
        if not gold_items <= (set(row["item_ids"]) | set(all_items)):
            bad_gold.append(row_id(row))
        if estimate_tokens(row["disclosure"]) != row["token_estimate"]:
            bad_token.append(row_id(row))
        if row["version"] != version_by_arm[row["arm"]]:
            bad_version.append(row_id(row))

    check("A1-disclosure-rebuild-equals-projection", bad_rebuild)
    check("A2-chars-equals-len", bad_chars)
    check("A3-budget-respected", bad_budget)
    check("A4-full-arms-complete", bad_full)
    check("A5-scale-equals-bank", bad_scale)
    check("A6-position-recorded", bad_position)
    check("A7-no-arm-name-in-disclosure", bad_arm)
    check("A8-gold-ids-known", bad_gold)
    check("A11-token-estimate-reproducible", bad_token)
    check("A12-arm-version-consistent", bad_version)

    zero_den = [row_id(r) for r in rows
                if (r["metrics"]["required_recall"] is None) != (not r["gold"]["required_facts"])]
    check("A9-zero-denominator-null", zero_den)

    stale_rows = [r for r in rows if r["variant"] == "stale"]
    stale_errors = []
    for row in stale_rows:
        if row["arm"] == "A-compiled":
            if not row.get("compiled", {}).get("stale_after_update"):
                stale_errors.append(row_id(row))
        if row["arm"] == "A-full" and row.get("compiled"):
            stale_errors.append(row_id(row) + ":A-full-不应有compiled标记")
    check("A10-stale-view-marked", stale_errors)

    return {"ok": not errors, "checks": checks, "failed_checks": failed,
            "error_count": len(errors), "errors": errors[:50], "rows": len(rows),
            "stale_rows": len(stale_rows)}


                                                               

def _load_inputs():
    rows = jsonl_read(OUT / "disclosures.jsonl")
    stale = jsonl_read(OUT / "stale_disclosures.jsonl")
    matched = jsonl_read(OUT / "cost_matched.jsonl")
    groups = jsonl_read(GROUPS_PATH)
    banks = jsonl_read(OUT.parent / "data" / "banks.jsonl")
    views = json.loads((OUT / "compiled_view.json").read_text(encoding="utf-8")) \
        if (OUT / "compiled_view.json").exists() else None
    stale_groups = json.loads((OUT / "stale_groups.json").read_text(encoding="utf-8")) \
        if (OUT / "stale_groups.json").exists() else None
    return rows, stale, matched, groups, banks, views, stale_groups


def analyze(*, answers_path: Path | None = None, synthetic: bool = False) -> dict:
    rows, stale, matched, groups, banks, views, stale_groups = _load_inputs()
    all_rows = rows + stale + matched
    audit_result = audit(all_rows, groups, banks, views=views, stale_groups=stale_groups)
    if synthetic:
        answers = {row_id(r): ideal_answer(r) for r in all_rows}
        source = "synthetic_oracle（仅机器自检，不是模型结果）"
    elif answers_path and answers_path.exists():
        answers = {a["row_id"]: a.get("raw_text") if "raw_text" in a else a.get("answer")
                   for a in jsonl_read(answers_path)}
        source = f"answers:{answers_path}"
    else:
        answers = {}
        source = "none（只报披露成本与审计；行为字段为 null）"
    scored, coverage = score_rows(all_rows, answers)
    summary = summarize(scored)
    result = {
        "schema": "disclosure-analysis-v1",
        "synthetic": bool(synthetic), "answer_source": source,
        "coverage": coverage, "audit": audit_result,
        "summary": summary,
        "contrasts": contrasts(scored) if scored else [],
        "frontier": frontier(summary),
        "disclosure_only": {"by_arm_split_scale": cost_table(all_rows)},
        "provenance": {
            "real_arm_result": bool(any(r.get("real_arm_result") for r in all_rows)),
            "compiler_backends": sorted({r.get("compiler_backend") for r in all_rows
                                         if r.get("compiler_backend")}),
            "graph_backends": sorted({r.get("graph_backend") for r in all_rows
                                      if r.get("graph_backend")}),
            "observation_kinds": {
                "independent": sum(1 for r in all_rows
                                   if r.get("observation_kind", "independent") == "independent"),
                "byte_identical_reuse": sum(1 for r in all_rows
                                            if r.get("observation_kind") == "byte_identical_reuse"),
            },
            "note": "real_arm_result=false 时行为字段只可能来自合成自检，不是真实臂结果。",
        },
        "sha256": {"disclosures": sha256((OUT / "disclosures.jsonl").read_text(encoding="utf-8"))
                   if (OUT / "disclosures.jsonl").exists() else None},
    }
    return result


                                                                    

def selftest() -> dict:
    rows, stale, matched, groups, banks, views, stale_groups = _load_inputs()
    assert rows, "先跑 disclosure_arms dry-run 生成 disclosures.jsonl"
    by_task_arm = {(r["task_id"], r["arm"], r["scale"]): r for r in rows}
    results = {}

                            
    relevant = next(r for r in rows if r["gold"]["required_facts"]
                    and r["arm"] == "A-full" and r["phenomenon"] == "single_fact")
    good = score_answer(ideal_answer(relevant), relevant)
    assert good["exact_success"] and good["required_proposition_rate"] == 1.0
    wrong_value = ideal_answer(relevant)
    wrong_value["value"] = "__不存在的选项__"
    bad = score_answer(wrong_value, relevant)
    assert not bad["exact_success"] and bad["required_proposition_rate"] < 1.0
    no_cite = ideal_answer(relevant)
    no_cite["cited_item_ids"] = []
    assert not score_answer(no_cite, relevant)["exact_success"]
    assert score_answer("这不是 JSON", relevant)["parse_ok"] is False
    results["known_good_and_bad"] = True

                          
    negation = next(r for r in rows if r["phenomenon"] == "negation" and r["arm"] == "P-full")
    hallucinated = ideal_answer(negation)
    hallucinated["decision"] = "allow"
    hallucinated["value"] = negation["gold"]["expected"]["value"]
    scored = score_answer(hallucinated, negation)
    assert not scored["exact_success"] and not scored["decision_ok"]
    results["negation_polarity"] = True

                                    
    stale_row = next(r for r in stale if r["arm"] == "A-compiled")
    old_value = stale_row["gold"]["stale_value"]
    assert old_value is not None
    answered_old = ideal_answer(stale_row)
    answered_old["value"] = old_value
    stale_scored = score_answer(answered_old, stale_row)
    assert stale_scored["stale_error"] and not stale_scored["exact_success"]
    answered_new = score_answer(ideal_answer(stale_row), stale_row)
    assert answered_new["exact_success"] and not answered_new["stale_error"]
    results["stale_view"] = True

                                          
    from disclosure_schema_20261025 import estimate_tokens
    empty = json.loads(json.dumps(next(r for r in rows if r["arm"] == "A-retrieve"
                                        and r["gold"]["required_facts"])))
    empty["item_ids"] = []
    empty["disclosure"] = project_full([], PHASE)
    empty["chars"] = len(empty["disclosure"])
    empty["token_estimate"] = estimate_tokens(empty["disclosure"])
    empty["metrics"] = dict(empty["metrics"], required_recall=0.0, item_count=0,
                            logical_groups=0, irrelevant_items=0,
                            missing_required_facts=list(empty["gold"]["required_facts"]))
    assert empty["metrics"]["required_recall"] == 0.0
    assert not score_answer(ideal_answer(empty), empty)["exact_success"]
    results["empty_retrieval"] = True

                              
    negative = next(r for r in rows if r["phenomenon"] == "no_relevant")
    assert negative["metrics"]["required_recall"] is None
    no_memory = {"decision": "no_memory", "value": None, "cited_item_ids": [],
                 "applied_condition": None, "applied_exception": None}
    assert score_answer(no_memory, negative)["exact_success"]
    halluc = dict(no_memory, decision="allow", value="5 条")
    assert score_answer(halluc, negative)["hallucination"]
    assert rate_block([])["rate"] is None
    results["zero_denominator"] = True

                                            
    over = json.loads(json.dumps(next(r for r in rows if r["arm"] in RETRIEVE_ARMS)))
    over["disclosure"] = over["disclosure"] + "X" * (over["budget_chars"] + 1 - over["chars"])
    over["chars"] = len(over["disclosure"])
    bad_audit = audit([over], groups, banks)
    assert not bad_audit["ok"]
    assert "A3-budget-respected" in {f["code"] for f in bad_audit["failed_checks"]}
    results["over_budget"] = True

                                     
    maps = _item_maps(groups, views, stale_groups)
    base = next(r for r in rows if r["arm"] == "P-full" and r["gold"]["required_facts"]
                and len(r["item_ids"]) >= 3)
    ids = list(base["item_ids"])
    target = next(i for i in ids if group_of_item(i) in set(base["gold"]["required_groups"]))
    rest = [i for i in ids if i != target]
    orders = {"start": [target] + rest,
              "middle": rest[:len(rest) // 2] + [target] + rest[len(rest) // 2:],
              "end": rest + [target]}
    scores = {}
    for label, order in orders.items():
        variant = json.loads(json.dumps(base))
        variant["item_ids"] = order
        variant["disclosure"] = project_full([maps["main"][i] for i in order], PHASE)
        variant["position"] = label
        scores[label] = score_answer(ideal_answer(variant), variant)["exact_success"]
    assert set(scores.values()) == {True}, scores
    results["position_invariance"] = True

                                                    
    def fake_scored(diffs):
        out = []
        for i, diff in enumerate(diffs):
            fam = f"F{i % 4:02d}"
            out.append({"task_id": f"T{i}", "variant": "main", "scale": 8, "arm": "A",
                        "family_id": fam, "exact_success": diff})
            out.append({"task_id": f"T{i}", "variant": "main", "scale": 8, "arm": "B",
                        "family_id": fam, "exact_success": 0.0})
        return out

    improving = cluster_bootstrap(_pairs_for(fake_scored([1.0] * 8), "A", "B"), draws=300)
    assert improving["direction"] == "improves", improving
    nullish = cluster_bootstrap(_pairs_for(fake_scored([1.0, 0.0] * 4), "A", "B"), draws=300)
    assert nullish["direction"] in ("uncertain",), nullish
    results["cluster_bootstrap"] = True

                         
    fake_summary = [
        {"variant": "main", "split": "dev", "scale": 8, "arm": "P-full", "n": 4,
         "exact_success": {"rate": 0.5}, "cost": {"chars_mean": 1000.0, "tokens_est_mean": 300.0,
                                                  "latency_ms_mean": 1.0}},
        {"variant": "main", "split": "dev", "scale": 8, "arm": "A-full", "n": 4,
         "exact_success": {"rate": 0.5}, "cost": {"chars_mean": 1200.0, "tokens_est_mean": 350.0,
                                                  "latency_ms_mean": 1.0}},
        {"variant": "main", "split": "dev", "scale": 8, "arm": "A-graph", "n": 4,
         "exact_success": {"rate": 0.9}, "cost": {"chars_mean": 1200.0, "tokens_est_mean": 350.0,
                                                  "latency_ms_mean": 1.0}},
    ]
    front = frontier(fake_summary)
    assert next(f for f in front if f["arm"] == "A-full")["pareto_dominated"]
    assert not next(f for f in front if f["arm"] == "A-graph")["pareto_dominated"]
    results["frontier"] = True

                             
    real = audit(rows + stale + matched, groups, banks, views=views, stale_groups=stale_groups)
    assert real["ok"], real["errors"][:5]
    results["audit_real_rows"] = True
    return {"ok": True, "results": results,
            "rows": {"main": len(rows), "stale": len(stale)},
            "cluster_bootstrap_demo": {"improving": improving, "nullish": nullish}}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] == "selftest":
        print(json.dumps(selftest(), ensure_ascii=False, indent=2))
        return 0
    if argv[1] == "analyze":
        synthetic = "--synthetic" in argv
        answers = None
        if "--answers" in argv:
            answers = Path(argv[argv.index("--answers") + 1])
        result = analyze(answers_path=answers, synthetic=synthetic)
        name = "analysis_synthetic.json" if synthetic else "analysis.json"
        json_write(OUT / name, result)
        print(json.dumps({"ok": result["audit"]["ok"], "file": f"out/{name}",
                          "coverage": result["coverage"],
                          "audit_errors": result["audit"]["error_count"],
                          "arms_with_behavior": sorted({c["arm"] for c in result["summary"]
                                                        if c["exact_success"]["rate"] is not None})},
                         ensure_ascii=False, indent=2))
        return 0
    raise SystemExit(f"未知命令：{argv[1]}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
