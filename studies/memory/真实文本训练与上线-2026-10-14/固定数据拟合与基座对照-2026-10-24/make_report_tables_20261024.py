"""Collect the study numbers into markdown tables so the report does not retype them."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fit_common_20261024 as C              

FIT_TARGET = C.PROTOCOL["fit_target"]


def fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    runs = []
    for path in sorted(C.RUNS.glob("*/result.json")):
        runs.append(C.read_json(path))
    lines = ["# 固定数据拟合与基座对照：数值表（自动生成）", ""]
    lines += ["## 1. 训练预算与拟合", "",
              "| run | 更新 | 实例 | 真实token | terminal acc | terminal CE | 达标 | 首次达标记录 | dev-best step | dev-best macroF1 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for run in runs:
        budget = run["training"]["budget"]
        terminal = run["terminal"]["metrics"]["train"]
        first = next((row["step"] for row in run["training"]["segments"]
                      if row["train"]["accuracy"] >= FIT_TARGET["train_accuracy"]
                      and row["train"]["loss"] <= FIT_TARGET["train_ce"]), None)
        met = terminal["accuracy"] >= FIT_TARGET["train_accuracy"] and terminal["loss"] <= FIT_TARGET["train_ce"]
        lines.append("| {run} | {updates} | {instances} | {tokens} | {acc} | {ce} | {hit} | {first} | {step} | {f1} |".format(
            run=run["run"], updates=budget["actual_updates"], instances=budget["actual_sample_instances"],
            tokens=budget["real_tokens"], acc=fmt(terminal["accuracy"]), ce=fmt(terminal["loss"], 5),
            hit="是" if met else "否",
            first=first if first is not None else "未达", step=run["best_dev"]["step"],
            f1=fmt(run["best_dev"]["metrics"]["dev"]["macro_f1"])))
    lines += ["", "## 2. 泛化与诊断回放（dev 选点冻结后）", "",
              "| run | terminal dev F1 | dev-best dev acc | dev-best dev F1 | 120 acc | 120 macroF1 | 120 CE | 120 三联全对 | 300 acc | 300 macroF1 | 300 CE |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for run in runs:
        terminal_dev = run["terminal"]["metrics"]["dev"]
        best_dev = run["best_dev"]["metrics"]["dev"]
        diag = run["diagnostics"]["metrics"]["diagnostic-120"]
        conf = run["diagnostics"]["metrics"]["confirmation-300"]
        triplets = diag.get("triplets", {})
        lines.append("| {run} | {tf1} | {bacc} | {bf1} | {da} | {df1} | {dce} | {tri} | {ca} | {cf1} | {cce} |".format(
            run=run["run"], tf1=fmt(terminal_dev["macro_f1"]), bacc=fmt(best_dev["accuracy"]),
            bf1=fmt(best_dev["macro_f1"]), da=fmt(diag["accuracy"]), df1=fmt(diag["macro_f1"]),
            dce=fmt(diag["loss"]), tri=f"{triplets.get('triplets_all_correct')}/{triplets.get('triplets_n')}",
            ca=fmt(conf["accuracy"]), cf1=fmt(conf["macro_f1"]), cce=fmt(conf["loss"])))
    lines += ["", "## 3. 诊断集分类与安全口径", "",
              "| run | split | coexist recall | replace recall | merge recall | 漏冲突 | 误整替(merge金标) | coexist被整替 | 丢保留项 | 额外merge | 未覆盖项整替 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for run in runs:
        for split, summary in run["diagnostics"]["metrics"].items():
            recalls = {name: summary["per_class"][name]["recall"] for name in C.ACTIONS}
            lines.append("| {run} | {split} | {co} | {re} | {me} | {miss}/{cd} | {unsafe}/{md} | {coexist_loss}/{pd} | {lost} | {extra} | {total} |".format(
                run=run["run"], split=split, co=fmt(recalls["coexist"]), re=fmt(recalls["replace_all"]),
                me=fmt(recalls["merge_preserve"]), miss=summary.get("missed_conflict_count"),
                cd=summary.get("conflict_denominator"), unsafe=summary.get("unsafe_replace_count"),
                md=summary.get("merge_denominator"), coexist_loss=summary.get("coexist_replace_loss_count"),
                pd=summary.get("preservation_denominator"), lost=summary.get("lost_preserved_items"),
                extra=summary.get("extra_merge_count"), total=summary.get("unsafe_replace_total_count")))
    lines += ["", "## 4. 拟合曲线（每50更新，完整train/dev）", ""]
    for run in runs:
        lines += [f"### {run['run']}", "", "| step | train acc | train CE | dev macroF1 |", "|---|---|---|---|"]
        for row in run["training"]["segments"]:
            lines.append(f"| {row['step']} | {fmt(row['train']['accuracy'])} | {fmt(row['train']['loss'], 5)} | "
                         f"{fmt(row['dev']['macro_f1'])} |")
        lines.append("")
    lines += ["## 5. 资源与产物", "",
              "| run | peak alloc GiB | peak reserved GiB | 训练秒 | checkpoint bytes | checkpoint sha256 |",
              "|---|---|---|---|---|---|"]
    for run in runs:
        checkpoint = run["terminal"]["checkpoint"]
        lines.append("| {run} | {a} | {r} | {s} | {b} | {sha} |".format(
            run=run["run"], a=fmt(run["cuda_peak_allocated_bytes"] / 2 ** 30, 2),
            r=fmt(run["cuda_peak_reserved_bytes"] / 2 ** 30, 2), s=fmt(run["seconds"], 1),
            b=checkpoint.get("bytes") or checkpoint.get("bytes") or checkpoint.get("directory"),
            sha=(checkpoint.get("sha256") or "")[:16]))
    lines += ["", "## 6. 训练前探针（真实CUDA一步）", "",
              "| run | 可训练张量 | 非零梯度张量 | 更新张量 | 未用pooler | 梯度loss | 参数最大步长 |",
              "|---|---|---|---|---|---|---|"]
    for run in runs:
        probe = run["audits"]["gradient_update"]
        lines.append("| {run} | {t} | {g} | {c} | {u} | {l} | {d} |".format(
            run=run["run"], t=probe["trainable_tensors"], g=probe["nonzero_grad_tensors"],
            c=probe["changed_tensors"], u=len(probe["unused_trainable_tensors"]),
            l=fmt(probe["loss"], 3), d=fmt(probe["max_abs_delta"], 8)))
    output = C.HERE / "results" / "report-tables.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    C.dump(C.HERE / "results" / "report-numbers.json",
           {run["run"]: {"config": run["config"], "training": run["training"],
                         "terminal": run["terminal"]["metrics"], "best_dev": run["best_dev"]["metrics"],
                         "diagnostics": run["diagnostics"]["metrics"],
                         "cuda": {"peak_allocated_bytes": run["cuda_peak_allocated_bytes"],
                                  "peak_reserved_bytes": run["cuda_peak_reserved_bytes"]},
                         "seconds": run["seconds"], "frozen_base_sha256": run["frozen_base_sha256"]}
            for run in runs})
    receipt = {"generated": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
               "environment": C.environment(), "protocol": C.PROTOCOL,
               "run_lock_sha256": C.sha256_file(C.HERE / "run-lock.json"),
               "inputs": C.read_json(C.HERE / "run-lock.json")["inputs"],
               "runs": {}, "failures": sorted(str(path.relative_to(C.HERE))
                                              for path in (C.RUNS / "_failed").glob("*")) if (C.RUNS / "_failed").is_dir() else [],
               "no_deploy": True, "human_gold": False, "diagnostics_are_exposed": True}
    for path in sorted(C.RUNS.glob("*/result.json")):
        run = C.read_json(path)
        directory = path.parent
        verification_path = directory / "verification.json"
        receipt["runs"][run["run"]] = {
            "result": {"path": str(path), "sha256": C.sha256_file(path)},
            "kind": run["kind"],
            "verification": ({"path": str(verification_path),
                              "sha256": C.sha256_file(verification_path),
                              "passed": C.read_json(verification_path)["passed"]}
                             if verification_path.is_file() else None),
            "checkpoints": {tag: run[tag]["checkpoint"] for tag in ("terminal", "best_dev")},
            "predictions": {f"{tag}/{name}.json": C.sha256_file(directory / "predictions" / tag / f"{name}.json")
                            for tag in ("terminal", "best-dev")
                            for name in ("train", "dev")}
                           | {f"best-dev/{name}.json": C.sha256_file(directory / "predictions" / "best-dev" / f"{name}.json")
                              for name in ("diagnostic-120", "confirmation-300")},
            "csv": {path.name: C.sha256_file(path) for path in sorted((C.HERE / "results").glob(f"rows-{run['run']}-*.csv"))},
            "metrics": {"terminal": run["terminal"]["metrics"], "best_dev": run["best_dev"]["metrics"],
                        "diagnostics": run["diagnostics"]["metrics"]},
            "budget": run["training"]["budget"], "seconds": run["seconds"],
            "cuda_peak_allocated_bytes": run["cuda_peak_allocated_bytes"],
        }
    C.dump(C.HERE / "回执-20261024.json", receipt)
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
