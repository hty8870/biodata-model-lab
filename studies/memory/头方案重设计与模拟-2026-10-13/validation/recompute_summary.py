"""独立包复验：从逐题 JSONL 重算 finalize summary 并比对 expected。

只读 JSONL/summary/protocol/A-dual-hard.json；不重评分、不训练、不调用模型。
用法（在任务包根或任意 cwd 均可）：
  python -B recompute_summary.py --jsonl <questions.jsonl> --summary <summary.json>
      --protocol <head-design-v5.json> [--adual <A-dual-hard.json>] [--out <verification.json>]

核心 summary 不含 Adual：A 的五叶行为由 two_way.combo 经 proto
semantics.combination_leaf 映射得到 hard 5 叶预测，另在 Adual 段比对。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LEAF_ORDER = ["equivalent", "old_entails_new_strict", "new_entails_old_strict",
              "disjoint", "overlap_incomparable"]
PARENT_ORDER = ["E", "N", "C"]
TOL = 1e-9


def import_combination_leaf(source_root: Path):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from research.memory_relations.head_design_semantics import combination_leaf              
    return combination_leaf


def macro_f1(confusion: dict, classes: list[str]) -> float:
    f1s = []
    for c in classes:
        tp = confusion.get((c, c), 0)
        fn = sum(confusion.get((c, p), 0) for p in classes) - tp
        fp = sum(confusion.get((g, c), 0) for g in classes) - tp
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return sum(f1s) / len(f1s)


def adual_metrics(conf: dict, n: int, reject_count: int) -> dict:
    """Adual：reject 是第 6 列（全 FN、不计任何类 FP），recall 分母含 reject。"""
    correct = sum(conf.get((c, c), 0) for c in LEAF_ORDER)
    f1s, recalls = [], {}
    for c in LEAF_ORDER:
        tp = conf.get((c, c), 0)
        fn = sum(v for (g, _p), v in conf.items() if g == c) - tp
        fp = sum(v for (g, p), v in conf.items() if p == c and g in LEAF_ORDER) - tp
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
        support = sum(v for (g, _p), v in conf.items() if g == c)
        recalls[c] = (tp / support) if support > 0 else 0.0
    return {"accuracy": correct / n, "macro_f1": sum(f1s) / len(f1s),
            "reject": reject_count, "total_wrong": n - correct,
            "legal_wrong": n - correct - reject_count, "per_class_recall": recalls}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--source-root", required=True,
                    help="目录，其下有 research/memory_relations/（复制源命名空间）")
    ap.add_argument("--adual", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    combination_leaf = import_combination_leaf(Path(args.source_root))
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    views_expected = summary["views"]

         
    acc: dict[str, dict] = {}
    adual_expected = None
    if args.adual:
        adual_expected = json.loads(Path(args.adual).read_text(encoding="utf-8"))

    rows = 0
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows += 1
            cand = row["candidate"]
            st = acc.setdefault(cand, {
                "n": 0,
                "coarse_conf": {},
                "fine_conf": {},
                "coarse_nll_sum": 0.0, "coarse_nll_cal_sum": 0.0,
                "fine_nll_sum": 0.0, "fine_nll_cal_sum": 0.0,
                "fine_nll_rows": 0,
                "coarse_correct": 0, "fine_correct": 0,
                "adual_conf": {}, "adual_reject": 0, "adual_n": 0,
            })
            st["n"] += 1
            gp, gl = row["gold_parent"], row["gold_leaf"]
            pm, fm = row.get("parent_map"), row.get("fine_map")
            if pm is not None:
                st["coarse_conf"][(gp, pm)] = st["coarse_conf"].get((gp, pm), 0) + 1
                st["coarse_correct"] += (pm == gp)
            if fm is not None:
                st["fine_conf"][(gl, fm)] = st["fine_conf"].get((gl, fm), 0) + 1
                st["fine_correct"] += (fm == gl)
            if row.get("coarse_nll") is not None:
                st["coarse_nll_sum"] += row["coarse_nll"]
            if row.get("fine_nll") is not None:
                st["fine_nll_sum"] += row["fine_nll"]
                st["fine_nll_rows"] += 1
            if row.get("coarse_nll_calibrated") is not None:
                st["coarse_nll_cal_sum"] += row["coarse_nll_calibrated"]
            if row.get("fine_nll_calibrated") is not None:
                st["fine_nll_cal_sum"] += row["fine_nll_calibrated"]
            if row.get("arm") == "A":
                st["adual_n"] += 1
                tw = row.get("two_way") or {}
                combo = tw.get("combo")
                pred = None
                if tw.get("combo_legal") and isinstance(combo, str) and len(combo) == 2:
                    pred = combination_leaf(combo[0], combo[1], protocol)
                if pred is None:
                    st["adual_reject"] += 1
                    st["adual_conf"][(gl, "__reject__")] = st["adual_conf"].get((gl, "__reject__"), 0) + 1
                else:
                    st["adual_conf"][(gl, pred)] = st["adual_conf"].get((gl, pred), 0) + 1

                         
    view_checks = []
    for cand, st in acc.items():
        exp = views_expected[cand]
        n = st["n"]
        checks = {}
        got = {}
        got["rows"] = n
        checks["rows"] = (n == exp["n"])
                
        c_f1 = macro_f1(st["coarse_conf"], PARENT_ORDER)
        got["coarse_macro_f1"] = c_f1
        checks["coarse_macro_f1"] = abs(c_f1 - exp["coarse_macro_f1"]) <= TOL
        c_nll = st["coarse_nll_sum"] / n
        got["coarse_nll"] = c_nll
        checks["coarse_nll"] = abs(c_nll - exp["coarse_nll"]) <= TOL
        c_nll_cal = st["coarse_nll_cal_sum"] / n
        got["coarse_nll_calibrated"] = c_nll_cal
        checks["coarse_nll_calibrated"] = abs(c_nll_cal - exp["coarse_nll_calibrated"]) <= TOL
        c_acc = st["coarse_correct"] / n
        got["coarse_accuracy"] = c_acc
        if exp.get("coarse_accuracy") is not None:
            checks["coarse_accuracy"] = abs(c_acc - exp["coarse_accuracy"]) <= TOL
              
        if st["fine_nll_rows"] > 0:
            f_f1 = macro_f1(st["fine_conf"], LEAF_ORDER)
            got["fine_macro_f1"] = f_f1
            if exp.get("fine_macro_f1") is not None:
                checks["fine_macro_f1"] = abs(f_f1 - exp["fine_macro_f1"]) <= TOL
            f_nll = st["fine_nll_sum"] / st["fine_nll_rows"]
            got["fine_nll"] = f_nll
            if exp.get("fine_nll") is not None:
                checks["fine_nll"] = abs(f_nll - exp["fine_nll"]) <= TOL
            f_nll_cal = st["fine_nll_cal_sum"] / st["fine_nll_rows"]
            got["fine_nll_calibrated"] = f_nll_cal
            exp_cal = exp.get("calibrated_metrics", {}).get("fine", {}).get("nll")
            if exp_cal is not None:
                checks["fine_nll_calibrated"] = abs(f_nll_cal - exp_cal) <= TOL
            f_acc = st["fine_correct"] / n
            got["fine_accuracy"] = f_acc
            if exp.get("fine_accuracy") is not None:
                checks["fine_accuracy"] = abs(f_acc - exp["fine_accuracy"]) <= TOL
        else:
            got["fine_macro_f1"] = None
            got["fine_nll"] = None
            got["fine_nll_calibrated"] = None
               
        if st["adual_n"] > 0:
            got["adual"] = adual_metrics(st["adual_conf"], st["adual_n"], st["adual_reject"])
            checks["adual"] = True                      
        view_checks.append({"view": cand, "got": got, "checks": checks,
                            "all_pass": all(checks.values())})

                                                         
    adual_checks = []
    if adual_expected is not None:
        for cand, st in acc.items():
            if st["adual_n"] == 0:
                continue
            seed = cand.split("-")[1]
            exp = adual_expected["A_dual"][seed]
            nA = st["adual_n"]
            got = adual_metrics(st["adual_conf"], nA, st["adual_reject"])
            ch = {
                "n": nA == exp["n"],
                "accuracy": abs(got["accuracy"] - exp["accuracy"]) <= TOL,
                "macro_f1": abs(got["macro_f1"] - exp["macro_f1"]) <= TOL,
                "reject": got["reject"] == exp["reject"],
                "total_wrong": got["total_wrong"] == exp["total_wrong"],
                "legal_wrong": got["legal_wrong"] == exp["legal_wrong"],
            }
                                 
            for cls in LEAF_ORDER:
                ch[f"recall_all_classes_{cls}"] = abs(
                    got["per_class_recall"][cls] - exp["per_class_recall"][cls]) <= TOL
            adual_checks.append({"view": cand, "got": got, "checks": ch,
                                 "all_pass": all(ch.values())})

                                 
    key = {}
    for cand, st in acc.items():
        n = st["n"]
        entry = {}
        if st["fine_nll_rows"] > 0:
            entry["fine_f1"] = macro_f1(st["fine_conf"], LEAF_ORDER)
            entry["fine_nll"] = st["fine_nll_sum"] / n
            entry["fine_acc"] = st["fine_correct"] / n
        entry["coarse_f1"] = macro_f1(st["coarse_conf"], PARENT_ORDER)
        entry["coarse_nll"] = st["coarse_nll_sum"] / n
        key[cand] = entry

    def mean(vals):
        return sum(vals) / len(vals)

    independent = {
        "C_raw_fine_f1": [key["C-13-fine"]["fine_f1"], key["C-20260919-fine"]["fine_f1"],
                          key["C-424242-fine"]["fine_f1"]],
        "C_raw_fine_nll": [key["C-13-fine"]["fine_nll"], key["C-20260919-fine"]["fine_nll"],
                           key["C-424242-fine"]["fine_nll"]],
        "H_raw_fine_f1": [key["H-13-fine"]["fine_f1"], key["H-20260919-fine"]["fine_f1"],
                          key["H-424242-fine"]["fine_f1"]],
        "H_raw_fine_nll": [key["H-13-fine"]["fine_nll"], key["H-20260919-fine"]["fine_nll"],
                           key["H-424242-fine"]["fine_nll"]],
    }
    independent["C_mean_fine_f1"] = mean(independent["C_raw_fine_f1"])
    independent["C_mean_fine_nll"] = mean(independent["C_raw_fine_nll"])
    independent["H_mean_fine_f1"] = mean(independent["H_raw_fine_f1"])
    independent["H_mean_fine_nll"] = mean(independent["H_raw_fine_nll"])
    for arm in ("A", "B", "C", "H"):
        seeds = ("13", "20260919", "424242")
        independent[f"{arm}_mean_coarse_f1"] = mean([key[f"{arm}-{s}-coarse"]["coarse_f1"] for s in seeds])
        independent[f"{arm}_mean_coarse_nll"] = mean([key[f"{arm}-{s}-coarse"]["coarse_nll"] for s in seeds])
    independent["C_mean_fine_acc"] = mean([key[f"C-{s}-fine"]["fine_acc"] for s in ("13", "20260919", "424242")])
    independent["H_mean_fine_acc"] = mean([key[f"H-{s}-fine"]["fine_acc"] for s in ("13", "20260919", "424242")])
    _a = []
    for _cand, _st in acc.items():
        if _st["adual_n"] > 0:
            _m = adual_metrics(_st["adual_conf"], _st["adual_n"], _st["adual_reject"])
            _a.append((_m["accuracy"], _m["macro_f1"]))
    if _a:
        independent["Adual_mean_accuracy"] = mean([x[0] for x in _a])
        independent["Adual_mean_macro_f1"] = mean([x[1] for x in _a])

    expected_peer = {
        "C_raw_fine_f1": [0.9879949675606327, 0.9975642755952249, 0.9910310248682773],
        "C_raw_fine_nll": [0.023086587772378576, 0.01521048798739431, 0.021071597062079367],
        "C_mean_fine_f1": 0.992196756,
        "C_mean_fine_nll": 0.019789558,
        "H_raw_fine_f1": [0.9911148205001912, 0.9938183528336975, 0.9903780569535202],
        "H_raw_fine_nll": [0.020452556680533363, 0.016588140573890537, 0.022112677702537678],
        "H_mean_fine_f1": 0.991770410,
        "H_mean_fine_nll": 0.019717792,
        "A_mean_coarse_f1": 0.993406112, "A_mean_coarse_nll": 0.009822286,
        "C_mean_coarse_f1": 0.989320263, "C_mean_coarse_nll": 0.015965117,
        "H_mean_coarse_f1": 0.988306090, "H_mean_coarse_nll": 0.015709432,
        "C_mean_fine_acc": 0.996032715,
        "Adual_mean_accuracy": 0.995849609375,
        "Adual_mean_macro_f1": 0.9933186665829412,
    }
    peer_checks = {}
    for k, exp in expected_peer.items():
        got = independent[k]
        if isinstance(exp, list):
            peer_checks[k] = all(abs(g - e) <= 1e-9 for g, e in zip(got, exp))
        else:
            peer_checks[k] = abs(got - exp) <= 1e-9

    all_view_pass = all(v["all_pass"] for v in view_checks)
    all_adual_pass = all(v["all_pass"] for v in adual_checks) if adual_checks else None
    all_peer_pass = all(peer_checks.values())

    result = {
        "tool": "recompute_summary.py（包内复制 code + 逐题 JSONL 流式重算）",
        "rows_streamed": rows,
        "views_checked": len(view_checks),
        "view_checks": view_checks,
        "adual_checks": adual_checks,
        "independent_recomputed": independent,
        "independent_expected": expected_peer,
        "peer_checks": peer_checks,
        "all_views_consistent_with_summary": all_view_pass,
        "all_adual_consistent_with_analysis": all_adual_pass,
        "all_peer_expected_match": all_peer_pass,
        "model_inference_calls": 0,
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "rows_streamed": rows, "views": len(view_checks),
        "all_views_consistent_with_summary": all_view_pass,
        "all_adual_consistent_with_analysis": all_adual_pass,
        "all_peer_expected_match": all_peer_pass,
        "peer_failures": [k for k, v in peer_checks.items() if not v],
        "view_failures": [v["view"] for v in view_checks if not v["all_pass"]],
        "adual_failures": [v["view"] for v in adual_checks if not v["all_pass"]],
    }, ensure_ascii=False))
    return 0 if (all_view_pass and all_peer_pass and (all_adual_pass is not False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
