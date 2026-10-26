"""交付包复验驱动：在包 D 内真实执行（不重评正式 final）。

- 步骤1：以包内复制 code 从逐题 JSONL 重算 summary 与 Adual（recompute_summary.py）。
- 步骤2：fresh smoke CLI（study_role=smoke，tiny 100 updates 机制预算）入 validation/smoke-study/。
所有命令的 argv / 实际 exit / wall / 日志路径写入 validation/smoke-cli-validation.json 与 logs/。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PYBIN = r"<archive-root>\.venv\Scripts\python.exe"
D = Path(r"<archive-root>\头方案重设计与模拟-2026-10-13")
V = D / "validation"
S = D / "source"                               
R4 = D / "evidence" / "r4"

steps = [
    ("recompute-summary", [
        PYBIN, "-B", str(V / "recompute_summary.py"),
        "--jsonl", str(R4 / "formal-study" / "finalize" / "questions.jsonl"),
        "--summary", str(R4 / "formal-study" / "finalize" / "summary.json"),
        "--protocol", str(S / "research" / "memory_relations" / "head-design-v5.json"),
        "--source-root", str(S),
        "--adual", str(R4 / "formal-analysis-final" / "A-dual-hard.json"),
        "--out", str(V / "recompute-verification.json"),
    ], 0),
    ("smoke-formal", [
        PYBIN, "-B", "-m", "research.memory_relations.head_design", "formal",
        "--study", "smoke", "--out", str(V / "smoke-study" / "formal"), "--allow-external-out",
    ], 0),
    ("smoke-prepare", [
        PYBIN, "-B", "-m", "research.memory_relations.head_design", "prepare",
        "--study-role", "smoke", "--out", str(V / "smoke-study" / "prepare"), "--allow-external-out",
    ], 0),
    ("smoke-train-dev", [
        PYBIN, "-B", "-m", "research.memory_relations.head_design", "train-dev",
        "--prepare", str(V / "smoke-study" / "prepare"),
        "--out", str(V / "smoke-study" / "train-dev"), "--updates", "100", "--allow-external-out",
    ], 0),
    ("smoke-freeze", [
        PYBIN, "-B", "-m", "research.memory_relations.head_design", "freeze",
        "--prepare", str(V / "smoke-study" / "prepare"),
        "--train-dev", str(V / "smoke-study" / "train-dev"),
        "--out", str(V / "smoke-study" / "freeze"), "--allow-external-out",
    ], 0),
    ("smoke-finalize", [
        PYBIN, "-B", "-m", "research.memory_relations.head_design", "finalize",
        "--prepare", str(V / "smoke-study" / "prepare"),
        "--train-dev", str(V / "smoke-study" / "train-dev"),
        "--freeze", str(V / "smoke-study" / "freeze" / "freeze.json"),
        "--out", str(V / "smoke-study" / "finalize"), "--allow-external-out",
    ], 0),
    ("smoke-status", [
        PYBIN, "-B", "-m", "research.memory_relations.head_design", "status",
        "--run", str(V / "smoke-study"),
    ], 0),
]


def main() -> int:
    logs = V / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    records = []
    all_ok = True
    for name, argv, expected in steps:
        t0 = time.time()
        proc = subprocess.run(argv, cwd=str(S), capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        wall = time.time() - t0
        (logs / f"{name}.log").write_text(
            "argv: " + json.dumps(argv, ensure_ascii=False) + "\n"
            + f"exit_code: {proc.returncode}\nwall_seconds: {wall}\n"
            + "----- stdout -----\n" + proc.stdout
            + "\n----- stderr -----\n" + proc.stderr, encoding="utf-8")
        ok = proc.returncode == expected
        all_ok = all_ok and ok
        records.append({
            "stage": name, "argv": argv, "exit_code": proc.returncode,
            "expected_exit_code": expected, "wall_seconds": round(wall, 3),
            "log": f"validation/logs/{name}.log",
        })
        print(f"[{name}] exit={proc.returncode} (expected {expected}) wall={wall:.2f}s ok={ok}")
    record = {
        "role": "package-validation",
        "note": "fresh smoke（tiny 100-update 机制预算，pilot；不重评正式 N10 final）；"
                "recompute 以包内复制 code + 包内 JSONL 重算",
        "cwd": str(S),
        "python": PYBIN,
        "steps": records,
        "formal_scored": False,
        "pilot": True,
    }
    (V / "smoke-cli-validation.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_expected_exit_codes_match": all_ok}, ensure_ascii=False))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
