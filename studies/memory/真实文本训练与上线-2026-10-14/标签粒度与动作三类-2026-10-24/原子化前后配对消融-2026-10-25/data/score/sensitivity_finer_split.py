                      
"""预注册§8.7的“比规格更细”敏感性；不改主预测、主分析或模型。"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "atomic_granularity_ablation_20261025.py"
SPEC = importlib.util.spec_from_file_location("ablation_sensitivity_core", SCRIPT)
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)

                                                          
SENSITIVE_IDS = ("ag-026-comp", "ag-026-conf")
                           
BLOCKED_ID = "ag-049-conf"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    cases = CORE.load_cases()
    fidelity = CORE.load_fidelity(ROOT / "data" / "processed" / "fidelity.json")
    report = {
        "schema": "ag-ablation-finer-split-sensitivity-v1",
        "scope": "post-hoc sensitivity under preregistered design §8.7; primary results unchanged",
        "accepted_case_ids": list(SENSITIVE_IDS),
        "not_recovered_case_ids": [BLOCKED_ID],
        "reason": "C splits a still-composite gold slot into distinct faithful independently editable atoms; strict cardinality is retained in primary metrics",
        "inputs": {
            "script_sha256": sha(SCRIPT),
            "primary_analysis_sha256": sha(ROOT / "data" / "score" / "analysis.json"),
            "fidelity_sha256": sha(ROOT / "data" / "processed" / "fidelity.json"),
        },
        "seeds": {},
    }
    for seed in (13, 42):
        tables = {}
        original = {}
        for arm in "ABC":
            rows = read_jsonl(ROOT / "data" / "score" / f"cases-direct3-seed{seed}-{arm}.jsonl")
            original[arm] = {row["case_id"]: row for row in rows}
            tables[arm] = copy.deepcopy(original[arm])
        for cid in SENSITIVE_IDS:
            row = tables["C"][cid]
            assert row["structure_reasons"] and "duplicate" in row["structure_reasons"][0]
            row["structure_complete"] = True
            row["structure_scope"] = "llm_atoms_finer_split_sensitivity"
            row["route"], row["route_rule"] = CORE.route_from_pairs(row["pair_predictions"], True)
        assert tables["C"][BLOCKED_ID]["structure_complete"] is False
        assert tables["C"][BLOCKED_ID]["route"] == "unknown"
        assert original["C"] != tables["C"]
        metrics = {arm: CORE.route_metrics(list(tables[arm].values()), arm, fidelity,
                                          boot=10000, seed=20261025) for arm in "ABC"}
        comparisons = {
            "B-A": CORE._comparison(tables, cases, "ABC", "A", "B", 10000, 20261025, fidelity),
            "C-B": CORE._comparison(tables, cases, "ABC", "B", "C", 10000, 20261025, fidelity),
            "C-A": CORE._comparison(tables, cases, "ABC", "A", "C", 10000, 20261025, fidelity),
        }
        report["seeds"][str(seed)] = {"metrics": metrics, "comparisons": comparisons}
        print(json.dumps({
            "seed": seed,
            "C_accuracy": metrics["C"]["accuracy"],
            "C_valid_conflict_recall": metrics["C"]["valid_conflict_recall"],
            "C_structure_complete": metrics["C"]["structure_complete"],
            "C_unknown": metrics["C"]["unknown"],
            "C-B_accuracy": comparisons["C-B"]["accuracy"],
            "C-B_valid_conflict_recall": comparisons["C-B"]["valid_conflict_recall"],
        }, ensure_ascii=False))
    out = ROOT / "data" / "score" / "finer-split-sensitivity.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "sha256": sha(out), "primary_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
