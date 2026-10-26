"""训练前 QA 证据镜像 + 记录哈希复核 + 正反自测（2026-10-12 晚）。

用途
----
把「训练前 QA / 唯一锁隔离」阶段的关键**审计聚合**镜像到研究目录，并复核该阶段
验证报告里**登记的每一个哈希**是否与磁盘实际文件一致（研究/脚本/日志/工件/权重）。

边界
----
- repo 全只读；只镜像小聚合（≈37 KB）：不复制 QA fixture 数据（`qa-development.jsonl`）、
  pytest 临时目录、GPU 权重目录、任何 shard/receipt 正文或 sealed 内容。
- 只读计数与哈希，不读第三判/教师回执语义。

自测
----
positive：659/14/6601/21、3 updates/48 实例、parity 7.72e-08、peak 2516.0MB、QA-only 标志、
adjudication 预备态（12 tests、final receipts 0）逐项与文件真值比对。
negative：篡改副本必须被哈希拒绝；已知错误等式（含 659→642 冒充）必须抛错。

用法
----
python -B 脚本/preflight-mirror-check-2026-10-12.py [--sync]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
RESEARCH = STAGE.parents[1]
REPO = RESEARCH.parent / "agent"
WT = REPO / ".git/validation-worktrees/mr3-data-20261011"
PRE = REPO / "research/memory_relations/runs/v3-training-preflight-20261012"
ARTIFACT = REPO / "artifacts/quality-memory-relations-v3-training-preflight.json"
SCORER = PRE / "cuda-joint/scorers/finetuned-joint-size239-seed13"

MIRROR = STAGE / "审计镜像-训练前QA-2026-10-12"
OUT_TXT = STAGE / "结果数据/preflight-recheck-2026-10-12.txt"

SOURCES = [
    ("preflight-verification-final.json", PRE / "verification-final.json"),
    ("preflight-gpu-smoke-result.json", PRE / "gpu-smoke-result.json"),
    ("preflight-gpu-export-final.json", PRE / "gpu-export-final.json"),
    ("preflight-qa-manifest.json", PRE / "qa-manifest.json"),
    ("preflight-startup-interface.md", PRE / "startup-interface.md"),
    ("adjudication-plan-verification.json", WT / "research/memory_relations/runs/v3-adjudication-20261012/plan/verification.json"),
]

HASH_CLAIMS = []                            


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expect(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def negative_self_tests(tmp: Path) -> None:
    src = MIRROR / "preflight-verification-final.json"
    tampered = tmp / "t.json"
    data = bytearray(src.read_bytes())
    data[-1] = data[-1] ^ 0x01
    tampered.write_bytes(bytes(data))
    expect(sha256(tampered) != sha256(src), "negative-a: tampered copy matched")
    for expr, msg in ((659 == 642, "negative-b"), (3 == 4, "negative-c")):
        try:
            expect(expr, f"{msg} fired")
            raise AssertionError(f"{msg} did not raise")
        except AssertionError as exc:
            if str(exc) == f"{msg} did not raise":
                raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true")
    args = ap.parse_args()

    lines: list[str] = []
    def out(s: str) -> None:
        print(s)
        lines.append(s)

    MIRROR.mkdir(exist_ok=True)
    if args.sync:
        for name, src in SOURCES:
            expect(src.exists(), f"source missing: {src}")
            shutil.copyfile(src, MIRROR / name)

    out("== 1. 镜像字节哈希（mirror == source）==")
    manifest = []
    for name, src in SOURCES:
        m = MIRROR / name
        expect(m.exists(), f"mirror missing: {m} (run with --sync)")
        ms, ss = sha256(m), sha256(src)
        expect(ms == ss, f"hash mismatch: {name}")
        manifest.append({"mirror": name, "source_repo_relative": str(src.relative_to(REPO)).replace("\\", "/"),
                         "sha256": ms, "bytes": m.stat().st_size})
        out(f"  OK  {name}  {ms[:16]}…  {m.stat().st_size} B")
    (MIRROR / "镜像清单-训练前QA-2026-10-12.json").write_text(
        json.dumps({"synced_from": str(WT.relative_to(REPO)).replace("\\", "/"), "files": manifest},
                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")

    neg = Path(tempfile.mkdtemp(prefix="qa-neg-"))
    try:
        negative_self_tests(neg)
        out("  OK  negative self-tests: tampered-hash rejected; 659→642 与 3→4 冒充均抛错")
    finally:
        shutil.rmtree(neg, ignore_errors=True)

    vf = json.loads((MIRROR / "preflight-verification-final.json").read_text(encoding="utf-8"))
    smoke = json.loads((MIRROR / "preflight-gpu-smoke-result.json").read_text(encoding="utf-8"))
    export = json.loads((MIRROR / "preflight-gpu-export-final.json").read_text(encoding="utf-8"))
    qam = json.loads((MIRROR / "preflight-qa-manifest.json").read_text(encoding="utf-8"))
    adjud = json.loads((MIRROR / "adjudication-plan-verification.json").read_text(encoding="utf-8"))

    out("")
    out("== 2. 登记哈希 vs 磁盘实际（只读，不复制）==")
    claims = [
        ("research-tests-final.log", vf["research"]["sha256"], PRE / "research-tests-final.log"),
        ("full report artifact", vf["full"]["report_sha256"], ARTIFACT),
        ("spec.json", vf["spec_sha256"], REPO / "research/memory_relations/spec.json"),
        ("frozen source-protocol", vf["frozen_protocol_sha256"], REPO / "research/memory_relations/runs/v3-coordination/source-protocol-frozen.json"),
        ("verify_final.py", vf["verification_script_sha256"], PRE / "verify_final.py"),
        ("verify_gpu_export.py", export["verification_script_sha256"], PRE / "verify_gpu_export.py"),
        ("scorer model.pt", export["artifact_sha256"]["model.pt"], SCORER / "model.pt"),
        ("scorer meta.json", export["artifact_sha256"]["meta.json"], SCORER / "meta.json"),
    ]
    for label, claimed, path in claims:
        actual = sha256(path)
        expect(actual == claimed, f"recorded hash mismatch: {label}")
        out(f"  OK  {label}: {actual[:16]}… 与登记一致")
    expect(vf["gpu"]["state_sha256"] == export["state_sha256"] == smoke["export"][0]["restored_sha256"],
           "state_sha256 三处不一致")
    out(f"  OK  state_sha256 三处一致: {vf['gpu']['state_sha256'][:16]}…")

    out("")
    out("== 3. 训练前 QA 关键数值（positive）==")
    r = vf["research"]
    expect((r["passed"], r["failed"], r["skipped"], r["xfailed"]) == (659, 0, 0, 0), "research 659 汇总不一致")
    gates = vf["full"]["gates"]
    expect(len(gates) == 14 and all(g["status"] == "passed" for g in gates), "full 门 != 14 全过")
    expect(vf["full"]["status"] == "passed", "full status != passed")
    expect(vf["full"]["pytest_passed"] == 6601 and vf["full"]["pytest_skipped"] == 21, "项目 pytest 6601/21 不一致")
    expect(vf["qa_only"] is True and vf["formal_comparison_started"] is False, "QA-only 标志被改")
    expect(vf["new_author_or_terminal_data_read"] is False, "读取了新作者/终测数据")
    out(f"  659 专项(0 fail/0 skip/0 xfail)；full {len(gates)}/14 passed；项目 pytest {vf['full']['pytest_passed']}/{vf['full']['pytest_skipped']} skip")

    b = smoke["budget"]
    expect(b["actual_updates"] == b["planned_updates"] == 3 and b["actual_sample_instances"] == b["planned_sample_instances"] == 48, "GPU smoke 预算 3/48 不一致")
    peak = smoke["memory_stages"]["train_and_dev"]
    expect(abs(peak["peak_allocated_mb"] - 2516.00341796875) < 1e-6, "train_and_dev 峰值 != 2516.0MB")
    expect(smoke["qa_only"] is True and smoke["formal_comparison"] is False, "smoke QA-only 标志被改")
    expect("old-construction QA fixture" in smoke["source"], "smoke 来源不是旧 QA fixture")
    expect(smoke["export"][0]["matches_selected_cpu_clone"] is True and smoke["export"][0]["root_loss_is_minimum"] is True, "导出/选择校验未过")
    out(f"  GPU smoke: 3 updates / 48 实例；峰值 allocated {peak['peak_allocated_mb']:.1f}MB（reserved {peak['peak_reserved_mb']}MB，train_and_dev）")

    expect(abs(export["max_probability_abs_diff"] - 7.719342470791091e-08) < 1e-18, "parity 值不一致")
    expect(export["max_probability_abs_diff"] < export["tolerance"] and export["argmax_equal"] is True, "parity 未达容差")
    expect(export["actual_optimizer_updates"] == 0 and export["audit_or_terminal_read"] is False, "export 阶段越界")
    out(f"  CPU/CUDA loader parity: {export['max_probability_abs_diff']:.3e} ≤ {export['tolerance']}；0 新增 updates、未读 audit/终测")

    expect(qam["new_author_data_used"] is False and qam["new_terminal_data_read"] is False and qam["human_gold"] is False, "QA fixture 来源标志被改")
    out(f"  QA fixture: {qam['rows']} 行（train {qam['train_rows']} / dev {qam['dev_rows']}），旧构造、非新数据、非 human gold")

    expect(adjud["tests_passed"] == 12 and adjud["actual_final_receipts"] == 0 and adjud["actual_frozen_arms"] == 0, "adjudication 预备态不一致")
    expect(adjud["status"] == "prepared-tested-not-adjudicated-not-trained" and adjud["third_workers_not_polled"] is True, "adjudication 状态被改")
    out(f"  第三判预备脚本: 12 tests passed；final receipts {adjud['actual_final_receipts']}、frozen arms {adjud['actual_frozen_arms']}、未轮询 third 工作区")

    out("")
    out("== 4. 未完成项（文件自述，不改写）==")
    for item in vf["not_done"]:
        out(f"  - {item}")

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwritten: {OUT_TXT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
