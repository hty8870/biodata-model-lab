                       
"""弃权类（escalate）三分类评测（2026-09-20，弃权类实验）。

核心研究问题：模型**自评弃权**（B3 的 escalate 类）与**交叉验证弃权**
（现役双判官的分歧信号）是什么关系——重合？互补？谁漏得多？

同一金标（binary_labels 2393 + OOD 108）上对比四条规则：
  R1 B3 单官：escalate→升级，其余 argmax 采纳
  R2 现役双判官（B×L0）：一致→采纳，分歧→升级
  R3 并集（保守）：B3 说 escalate ∨ 两官分歧 → 升级
  R4 交集（激进）：B3 说 escalate ∧ 两官分歧 → 升级（其余全采）
输出 abstain_eval.json：各规则 coverage/acc_on_covered + 弃权集×分歧集
列联（含各格准确率——「B3 独有弃权」「双判官独有分歧」各自抓到了什么）。

Usage:
  .venv/Scripts/python.exe research/route_eval/eval_b3_abstain.py [--limit 0]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

AGENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT_ROOT / "src"))
sys.path.insert(0, str(AGENT_ROOT / "research" / "route_eval"))

from eval_arms_dual import load_arm, load_jsonl              

EVAL_DIR = AGENT_ROOT / "research" / "route_eval"


def rule_stats(rows: list[dict]) -> dict:
    cov = [r for r in rows if not r["abstain"]]
    return {"n": len(rows), "coverage": len(cov) / max(len(rows), 1),
            "acc_on_covered": float(np.mean([r["pred"] == r["gold"] for r in cov])) if cov else None,
            "escalate_rate": 1 - len(cov) / max(len(rows), 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass

    from dataset_recommender.agent import l0_router as l0
    corpus = load_jsonl("binary_labels.jsonl")
    ood = load_jsonl("ood_route_queries.jsonl")
    if args.limit:
        corpus, ood = corpus[: args.limit], ood[: args.limit]
    print(f"corpus={len(corpus)} ood={len(ood)}", flush=True)

    score_b, _ = load_arm("qwen3_decider_b")
    score_b3, spec_b3 = load_arm("qwen3_decider_b3")

    def eval_rows(rows, gold_key, tag):
        out = []
        t0 = time.monotonic()
        for i, r in enumerate(rows):
            utt = r["utterance"]
            gold = r[gold_key]
            b = score_b("binary", utt)
            b3 = score_b3("binary3", utt)
            l = l0.score(l0.ROUTE_BINARY, utt)
            j_label = str(b["label"]) if b else None
            l_label = str(l["label"]) if l else None
            b3_label = str(b3["label"]) if b3 else None
            dual_agree = (j_label is not None and j_label == l_label)
            esc = b3_label == "escalate"
            out.append({
                "utt": utt, "gold": gold,
                "b3_label": b3_label,
                "b3_probs": b3["probs"] if b3 else None,
                "dual_agree": dual_agree, "j_label": j_label, "l_label": l_label,
                "R1_pred": None if esc else b3_label, "R1_abstain": esc,
                "R2_pred": j_label if dual_agree else None, "R2_abstain": not dual_agree,
                "R3_abstain": esc or not dual_agree,
                "R4_abstain": esc and not dual_agree,
            })
            if (i + 1) % 500 == 0:
                print(f"  {tag} {i + 1}/{len(rows)} "
                      f"({(time.monotonic() - t0) / 60:.1f}min)", flush=True)
               
        for r in out:
            esc, agree = r["R1_abstain"], r["dual_agree"]
                                            
            r["R1"] = {"pred": None if esc else r["b3_label"], "abstain": esc}
                                 
            r["R2"] = {"pred": r["j_label"] if agree else None, "abstain": not agree}
                                                        
            abst3 = esc or not agree
            r["R3"] = {"pred": None if abst3 else r["j_label"], "abstain": abst3}
                                                 
            abst4 = esc and not agree
            pred4 = (r["j_label"] if agree else r["b3_label"]) if not abst4 else None
            r["R4"] = {"pred": pred4, "abstain": abst4}
        stats = {rule: rule_stats([{"pred": r[rule]["pred"], "abstain": r[rule]["abstain"],
                                    "gold": r["gold"]} for r in out])
                 for rule in ("R1", "R2", "R3", "R4")}
                                        
        both = [r for r in out if r["R1_abstain"] and r["R2_abstain"]]
        only_b3 = [r for r in out if r["R1_abstain"] and not r["R2_abstain"]]
        only_dual = [r for r in out if not r["R1_abstain"] and r["R2_abstain"]]
        neither = [r for r in out if not r["R1_abstain"] and not r["R2_abstain"]]

        def acc_of(rs, pred_key):
            rs = [r for r in rs if r[pred_key] is not None]
            return float(np.mean([r[pred_key] == r["gold"] for r in rs])) if rs else None

        contingency = {
            "both_abstain": len(both),
            "only_b3_abstain": {"n": len(only_b3),
                                "说明": "双判官一致（R2 会采纳其判决）",
                                "dual_pred_acc": acc_of(only_b3, "j_label")},
            "only_dual_abstain": {"n": len(only_dual),
                                  "说明": "B3 给出判决（R1 会采纳）",
                                  "b3_pred_acc": acc_of(only_dual, "b3_label")},
            "neither": {"n": len(neither),
                        "b3_pred_acc": acc_of(neither, "b3_label"),
                        "dual_pred_acc": acc_of(neither, "j_label")},
        }
        return {"rows": out, "stats": stats, "contingency": contingency}

    corpus_res = eval_rows(corpus, "gate", "corpus")
    ood_res = eval_rows(ood, "gate", "ood")

    report = {"corpus": {"stats": corpus_res["stats"], "contingency": corpus_res["contingency"]},
              "ood": {"stats": ood_res["stats"], "contingency": ood_res["contingency"]},
              "spec_b3_temps": {t: spec_b3["tasks"][t]["temperature"] for t in spec_b3["tasks"]}}
    for tag, res in (("corpus", corpus_res), ("OOD", ood_res)):
        print(f"\n== {tag} ==", flush=True)
        for rule, s in res["stats"].items():
            print(f"  {rule}: coverage={s['coverage']:.4f} acc={s['acc_on_covered']}",
                  flush=True)
        print(f"  列联: {json.dumps(res['contingency'], ensure_ascii=False)}", flush=True)

    out_path = EVAL_DIR / "abstain_eval.json"
                                     
    report["ood_rows"] = [{k: (v if k != "b3_probs" else None) for k, v in r.items()}
                          for r in ood_res["rows"]]
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
