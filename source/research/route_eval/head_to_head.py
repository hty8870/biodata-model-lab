                       
"""头对头汇总：L0 vs L0+aug vs Qwen 复刻（ID test + OOD 108 + 时延 + 校准）。

从既有产物装配对比表（不重算）：
- L0（生产工件）：export_production_report.json + ood_l0_baseline.json
- L0+aug（对照臂）：l0aug_{binary,prelim}/metrics.json + ood_l0aug.json
- Qwen 复刻：qwen_decider/train_report.json
输出 research/route_eval/head_to_head.json + 控制台表。
"""
from __future__ import annotations

import json
from pathlib import Path

E = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((E / name).read_text(encoding="utf-8"))


def main() -> int:
    exp = load("export_production_report.json")["models"]
    ood_l0 = load("ood_l0_baseline.json")
    l0aug_bin = load("l0aug_binary/metrics.json")
    l0aug_pre = load("l0aug_prelim/metrics.json")
    ood_aug = load("ood_l0aug.json")
    qwen = load("qwen_decider/train_report.json")

    def cov90(cascade: list[dict]) -> dict:
        row = next((c for c in cascade if c["coverage"] >= 0.9), cascade[-1])
        return {"coverage": row["coverage"], "accuracy": row["accuracy"]}

    table: dict = {}
                       
    table["binary"] = {
        "L0": {"id_test": exp["l0_route_binary"]["operating_point"]["test_agreement_full"],
               "id_cascade_cov90": exp["l0_route_binary"]["operating_point"],
               "ood": ood_l0["binary"]["ok"] / (ood_l0["binary"]["ok"] + ood_l0["binary"]["fail"])},
        "L0+aug": {"id_test": l0aug_bin["test_accuracy"],
                   "id_cascade_cov90": cov90(l0aug_bin["cascade_calibrated"]),
                   "ood": ood_aug["binary"]["ood_acc"]},
        "decider": {"id_test": qwen["binary"]["test_acc"],
                    "id_cascade_cov90": cov90(qwen["binary"]["cascade"]),
                    "ood": qwen["ood"]["binary"]["ood_acc"]},
    }
                        
    table["prelim"] = {
        "L0": {"id_test": exp["l0_prelim_gate"]["operating_point"]["test_recall"] and None,
               "id_recall": exp["l0_prelim_gate"]["operating_point"]["test_recall"],
               "id_skip_correct": exp["l0_prelim_gate"]["operating_point"]["test_skip_correct_rate"],
               "ood_acc": ood_l0["prelim"]["ok"] / (ood_l0["prelim"]["ok"] + ood_l0["prelim"]["fail"])},
        "L0+aug": {"id_test": l0aug_pre["test_accuracy"],
                   "ood_acc": ood_aug["prelim"]["ood_acc"]},
        "decider": {"id_test": qwen["prelim"]["test_acc"],
                    "id_recall": qwen["prelim"]["test_recall"],
                    "id_skip_correct": qwen["prelim"]["test_skip_correct"],
                    "ood_acc": qwen["ood"]["prelim"]["ood_acc"],
                    "ood_recall": qwen["ood"]["prelim"]["ood_recall"],
                    "ood_skip_correct": qwen["ood"]["prelim"]["ood_skip_correct"]},
    }
                    
    table["route3"] = {
        "L0": {"id_test": exp["l0_route_three_class"]["operating_point"]["test_agreement_full"],
               "ood": None},
        "decider": {"id_test": qwen["route3"]["test_acc"],
                    "ood": qwen["ood"]["route3"]["ood_acc"]},
    }
    table["latency"] = {
        "L0_us_p50": 14.4,
        "decider_ms_per_decision_gpu": qwen.get("latency_ms_per_decision_gpu"),
    }
    out = E / "head_to_head.json"
    out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(table, ensure_ascii=False, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
