"""正式终评 / 根曲线 / CPU 探针 / 分类闭合轮：复核 + 必要小聚合镜像（2026-10-12；v4 独立复核修订）。

v4 修订（review r1 修复）
----
- §8 引入**冻结 payload vs 派生复核证据**的显式计数边界：本轮派生复核证据
  （`复算/输出-mr-stage-verify-2026-10-12.txt`、`复算/whitelist-verify-2026-10-12.json`）
  明示**不计入 payload**，单独列出件数/字节；payload 计数因此保持可复算、
  不会被复核工具自身输出反复顶漂。两个文件仍受档案清单并集（verify-archive v2）与
  交付索引盘点，不属"未知文件放过"。
- v3 以来的隔离参数与写入边界守卫不变；检查与打印除 §8 分列外与 v3 一致。

用途
----
复核统一报告（`研究报告-完整版-2026-10-12.md`）引用的全部关键数字：
① 终评 compact-summary（F1/CI/一致率/分母/uncertain 保留）；② 根曲线 mean±sample SD；
③ CPU 探针时延/加载/RSS/位级相等；④ REPORT.md 文本与 final-validation 一致性；
⑤ 强范围轮 full 通过链（历史）；⑥ **分类闭合轮最终终态**（61e479…/2e15…、14/14、
   6672/21 skip、公开投影前红后绿、保护 1205）。

镜像覆盖（索引同时区分三类，不互相冒充）
----
- **aggregate_mirror**：本目录逐字节镜像件（mirror==source，保存不改变 fresh 属性）；
- **synthetic_raw_inputs**：`复现资料-2026-10-12/数据-冻结三臂/ + 数据-终评参考/`（合成冻结输入，
  逐件 SHA 在复现档案 manifest/supplement）；
- **research_artifacts**：`复现资料-2026-10-12/` 其余（协议/源码快照/缓存/验收/锁定工件/复算）。

边界
----
- repo 全只读；镜像只复制**报告/小 JSON/时延表**（≈325 KB）。不复制 GPU 权重、
  predictions.jsonl 全文、raw sealed 文本、大日志或 cache-entries。
- 不读终测题目/回执语义；不重新运行模型或评价；保存数据不把已曝光终测变成 fresh，
  归档文件级齐全不等于已做可移植/重载验收。
- **写入边界（v3）**：所有写入必须位于 `--out-root`（默认 = `--inputs-root`）内，
  逐写入路径断言；`--inputs-root` 与 `--out-root` 不同（隔离运行）时禁止 `--sync`。
  隔离运行示例（只读真实档案、只写自有根）：
  `python -B <本脚本> --inputs-root <真实数据修复与新训练目录> --out-root <自有 staging> --repo-root <agent 仓根>`

自测
----
positive：曲线 mean/SD、667/871 与 667/872 两个分母、444/445 与 427/427、98/84 uncertain、
37 分母与 23 正确、1444 pair attempts、CPU p50/p99、位级相等、REPORT 文本关键串逐项核对；
分类闭合轮 14/14、6672/21、185/3、120/2、69/0/0、6.390 s、公开投影 before=失败/after=2 passed。
negative：篡改副本必须被哈希拒绝；两个已知错误等式（667/871==667/872、曲线 60==120 均值）必须抛错；
full exit!=0 与 6670==6601 冒充必须抛错；**before 两 profile 被冒充为通过**与 **6672==6670 冒充**
必须抛错。

用法
----
python -B 脚本/final-report-recheck-2026-10-12.py [--sync]
    [--inputs-root <数据修复与新训练（读取根）>] [--out-root <写入根>] [--repo-root <agent 仓根>]
默认三根从脚本位置推导（与 v2 相同）；检查与打印输出与 v2 逐字一致。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIRROR_NAME = "审计镜像-终评与CPU探针-2026-10-12"
INDEX_NAME = "镜像清单-终评与CPU探针-2026-10-12.json"
RESULTS_DIR = "结果数据"
OUT_TXT_NAME = "final-report-recheck-2026-10-12.txt"
REPRO_NAME = "复现资料-2026-10-12"
REPRO_MANIFEST_NAME = "manifest-复现档案-2026-10-12.json"
REPRO_SUPPLEMENT_NAME = "manifest-补充-锁定工件与分类闭合-2026-10-12.json"
SYNTHETIC_DIR_NAMES = ["数据-冻结三臂", "数据-终评参考"]
                                                
DERIVED_EVIDENCE = [
    "复算/输出-mr-stage-verify-2026-10-12.txt",
    "复算/whitelist-verify-2026-10-12.json",
]


def assert_inside(root: Path, path: Path) -> None:
    """写入边界守卫：path 必须位于 root 内（大小写不敏感；跨根即拒绝）。"""
    root_s = os.path.normcase(str(root.resolve()))
    path_s = os.path.normcase(str(path.resolve()))
    if path_s != root_s and not path_s.startswith(root_s + os.sep):
        raise SystemExit(f"[BOUNDARY] 拒绝写入 out-root 之外：{path}（out-root={root}）")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dir_stats(dirs: list[Path]) -> tuple[int, int]:
    files = 0
    total = 0
    for d in dirs:
        for p in sorted(d.rglob("*")):
            if p.is_file():
                files += 1
                total += p.stat().st_size
    return files, total


def expect(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def near(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(a - b) <= tol


def negative_self_tests(tmp: Path, mirror: Path, cls: dict) -> None:
    src = mirror / "terminal-eval-compact-summary.json"
    t = tmp / "t.json"
    b = bytearray(src.read_bytes())
    b[-1] ^= 0x01
    t.write_bytes(bytes(b))
    expect(sha256(t) != sha256(src), "negative-a: tampered copy matched")
    before_profiles = cls["public_before"]["profiles"]
    for cond, name in (
        ((667 / 871) == (667 / 872), "negative-b 667/871==667/872"),
        (0.726767 == 0.496241, "negative-c curve 60==120"),
        (before_profiles[0]["pytest_returncode"] == 0, "negative-f before profile 被冒充为通过"),
        (cls["pytest_suite"]["passed"] == 6670, "negative-g 6672==6670 冒充"),
    ):
        try:
            expect(cond, f"{name} fired")
            raise AssertionError(f"{name} did not raise")
        except AssertionError as exc:
            if str(exc) == f"{name} did not raise":
                raise


def main() -> int:
    ap = argparse.ArgumentParser(description="final report recheck (v3: isolation roots + write boundary)")
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--inputs-root", type=Path, default=None,
                    help="读取根：审计镜像-终评与CPU探针 / 复现资料-2026-10-12 / 结果数据（默认脚本父目录）")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="写入根：镜像清单与复算输出（默认 = inputs-root）；所有写入必须在其内")
    ap.add_argument("--repo-root", type=Path, default=None,
                    help="agent 仓库根（repo 侧只读；默认由 inputs-root 推导）")
    args = ap.parse_args()

    inputs_root = (args.inputs_root or HERE.parent).resolve()
    out_root = (args.out_root or inputs_root).resolve()
    repo_root = (args.repo_root or (inputs_root.parents[1].parent / "agent")).resolve()
    if args.sync and out_root != inputs_root:
        raise SystemExit("[BOUNDARY] --sync 只允许 out-root == inputs-root（隔离运行不得回写输入档案）")

    fe = repo_root / "research/memory_relations/runs/v3-final-evaluation-20261012/reports"
    curve_dir = repo_root / "research/memory_relations/runs/v3-root-curves-20261012"
    cp = repo_root / ".git/validation-worktrees/mr3-cpu-probe-20261012/research/memory_relations/runs/cpu-probe"
    nine = repo_root / "research/memory_relations/runs/v3-data-ablation-20261012/nine-run-gpu-records.json"
    govfix = repo_root / "research/memory_relations/runs/v3-governance-fix-20261012"
    artifact_strong = repo_root / "artifacts/quality-memory-relations-final-research-strong-scope-20261012.json"
    artifact_classification = repo_root / "artifacts/quality-memory-relations-final-research-classification-20261012.json"

    mirror = inputs_root / MIRROR_NAME
    mirror_out = out_root / MIRROR_NAME
    repro = inputs_root / REPRO_NAME
    repro_manifest = repro / REPRO_MANIFEST_NAME
    repro_supplement = repro / REPRO_SUPPLEMENT_NAME
    synthetic_dirs = [repro / name for name in SYNTHETIC_DIR_NAMES]

    sources = [
        ("terminal-eval-REPORT.md", fe / "REPORT.md"),
        ("terminal-eval-compact-summary.json", fe / "compact-summary.json"),
        ("terminal-eval-verified-summary.json", fe / "verified-summary.json"),
        ("terminal-eval-final-validation.json", fe / "final-validation.json"),
        ("terminal-eval-output-hashes.json", fe / "output-hashes.json"),
        ("root-curves-completed-runs-final.json", curve_dir / "completed-runs-final.json"),
        ("cpu-probe-report-final.md", cp / "report-final.md"),
        ("cpu-probe-verification-final.json", cp / "verification-final.json"),
        ("cpu-probe-bench-03-benchmark_summary.json", cp / "bench-03/benchmark_summary.json"),
        ("governance-fix-terminal-summary-validated.json", govfix / "terminal-summary-validated.json"),
        ("full-strong-scope-20261012.json", artifact_strong),
        ("full-classification-scope-20261012.json", artifact_classification),
        ("governance-fix-terminal-summary-classification.json", govfix / "terminal-summary-classification.json"),
    ]

    lines: list[str] = []
    def out(s: str) -> None:
        print(s)
        lines.append(s)

    out_root.mkdir(parents=True, exist_ok=True)
    if args.sync:
        for name, src in sources:
            expect(src.exists(), f"source missing: {src}")
            dest = mirror_out / name
            assert_inside(out_root, dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)

    out("== 1. 小聚合镜像（mirror == source, sha256）==")
    manifest = []
    for name, src in sources:
        m = mirror / name
        expect(m.exists(), f"mirror missing: {m} (run with --sync)")
        ms, ss = sha256(m), sha256(src)
        expect(ms == ss, f"hash mismatch: {name}")
        manifest.append({"mirror": name, "source_repo_relative": str(src.relative_to(repo_root)).replace("\\", "/"),
                         "sha256": ms, "bytes": m.stat().st_size})
        out(f"  OK  {name}  {ms[:16]}…  {m.stat().st_size} B")

    cls = json.loads((mirror / "governance-fix-terminal-summary-classification.json").read_text(encoding="utf-8"))

    neg = Path(tempfile.mkdtemp(prefix="fin-neg-", dir=str(out_root)))
    try:
        negative_self_tests(neg, mirror, cls)
        out("  OK  negative self-tests: 篡改拒绝；667/871==667/872、曲线均值互换、"
            "before 冒充通过、6672==6670 四式均抛错")
    finally:
        shutil.rmtree(neg, ignore_errors=True)

    cs = json.loads((mirror / "terminal-eval-compact-summary.json").read_text(encoding="utf-8"))
    fv = json.loads((mirror / "terminal-eval-final-validation.json").read_text(encoding="utf-8"))
    curve = json.loads((mirror / "root-curves-completed-runs-final.json").read_text(encoding="utf-8"))
    cpv = json.loads((mirror / "cpu-probe-verification-final.json").read_text(encoding="utf-8"))
    bench = json.loads((mirror / "cpu-probe-bench-03-benchmark_summary.json").read_text(encoding="utf-8"))
    report_txt = (mirror / "terminal-eval-REPORT.md").read_text(encoding="utf-8")
    cp_txt = (mirror / "cpu-probe-report-final.md").read_text(encoding="utf-8")

    out("")
    out("== 2. 终评分母/一致率（positive）==")
    t = cs["terminal"]
    h, o, a = t["heldout"], t["ood"], t["all"]
    expect((h["planned"], o["planned"], a["planned"]) == (445, 427, 872), "planned 分母不符")
    expect((h["classifier"]["known_reference_n"], o["classifier"]["known_reference_n"], a["classifier"]["known_reference_n"]) == (444, 427, 871),
           "known_reference 不符")
    expect(a["classifier"]["confirmed_correct"] == 667, "confirmed 不是 667")
    expect(near(a["classifier"]["row_agreement"], 667 / 871), "行一致率与 667/871 不符")
    expect(near(a["correct_completed_overall"]["value"], 667 / 872), "confirmed/Nplanned 与 667/872 不符")
    expect(near(a["completion"]["value"], 871 / 872), "completion 与 871/872 不符")
    expect((a["semantic_uncertain"], h["semantic_uncertain"], o["semantic_uncertain"]) == (84, 39, 45), "semantic uncertain 不符")
    expect(a["system_failure"] == 1 and a["failure_causes"] == {"missing_related_ids": 1}, "system failure 不符")
    out(f"  reference 444/445 & 427/427 -> all 871/872；confirmed 667/872={667/872:.4%}；行一致 667/871={667/871:.4%}")
    out(f"  semantic uncertain 84 = 39 + 45；system failure 1 = missing_related_ids（heldout）")

    expect(near(h["classifier"]["root_macro_f1"], 0.7522813300497377), "heldout rootF1")
    expect(near(o["classifier"]["root_macro_f1"], 0.8195484181034395), "ood rootF1")
    expect(near(a["classifier"]["root_macro_f1"], 0.785570727725999), "all rootF1")
    expect(near(h["classifier"]["root_agreement"], 0.7382566045066066) and near(o["classifier"]["root_agreement"], 0.7990277777777741),
           "root agreement")
    expect(near(h["production_gated"]["root_macro_f1"], 0.7528087874181927), "gated heldout F1")
    out(f"  root F1: heldout {h['classifier']['root_macro_f1']:.6f} / ood {o['classifier']['root_macro_f1']:.6f} / all {a['classifier']['root_macro_f1']:.6f}")
    out(f"  gated heldout {h['production_gated']['root_macro_f1']:.6f}（仅失败行位移，不是性能变强）")

    for token in ("[70.70%, 79.02%]", "[79.15%, 84.86%]", "667/871", "667/872", "98行（47/51）",
                  "15.49%", "7.53%", "24/861 (2.79%)", "a17.r02", "18014002"):
        expect(token in report_txt, f"REPORT.md 缺少关键串: {token}")
    out("  REPORT.md 文本关键串核对 OK（CI、两个 667 分母、98 行、τ0 诊断、旧 set 引用）")

    out("")
    out("== 3. 根曲线 mean ± sample SD（positive）==")
    groups: dict[int, list[float]] = {}
    for r in curve["runs"]:
        groups.setdefault(r["roots"], []).append(r["record"]["best_dev_root_loss"])
    expect(sorted(groups) == [60, 120], "曲线 roots 配置不是 60/120")
    for roots, exp_mean, exp_sd in ((60, 0.726767, 0.05344), (120, 0.496241, 0.05114)):
        vals = groups[roots]
        expect(len(vals) == 3, f"roots{roots} 不是 3 seed")
        mean, sd = statistics.fmean(vals), statistics.stdev(vals)
        expect(near(mean, exp_mean, 1e-5) and near(sd, exp_sd, 1e-5), f"roots{roots} mean/SD 不符")
        out(f"  roots {roots}: n=3 mean {mean:.6f} sample SD {sd:.5f}")
    nine_data = json.loads(nine.read_text(encoding="utf-8"))
    new_losses = [r["best_dev_root_loss"] for r in nine_data["records"] if r["arm"] == "new-adjudicated"]
    mean235, sd235 = statistics.fmean(new_losses), statistics.stdev(new_losses)
    expect(near(mean235, 0.422099, 1e-5) and near(sd235, 0.01373, 1e-5), "235 点（复用主九 run）不符")
    out(f"  roots 235（复用主九 run new 臂）: n=3 mean {mean235:.6f} sample SD {sd235:.5f}")

    out("")
    out("== 4. 旧 set / oracle / fixtures（positive）==")
    old = cs["old_set"]
    expect(old["planned"] == 37 and old["correct_completed_overall"]["numerator"] == 23, "旧 set 分母不符")
    expect(old["oracle_aggregation"]["rows"] == 10 and old["oracle_aggregation"]["agreement"]["numerator"] == 9, "oracle 9/10 不符")
    expect(old["physical_predictions"] == 1443 and old["scored_pairs"] == 1444, "旧 set pair 计数不符")
    by_rule = old["by_rule_count"]
    expect(sum(v["planned"] for v in by_rule.values()) == 37
           and sum(v["correct_completed"] for v in by_rule.values()) == 23
           and sum(v["pair_attempts"] for v in by_rule.values()) == 1444, "按规则数表不自洽")
    expect(fv["terminal_reference"] == 871 and fv["terminal_uncertain_reference"] == 98
           and fv["physical_terminal_predictions"] == 872 and fv["physical_all_predictions"] == 2322, "final-validation 不符")
    out("  旧 set 37→23/37、oracle 9/10（10/37 覆盖）、pairs 1444=1434无参考+10有参考；final-validation 872/871/98/2322 一致")

    out("")
    out("== 5. CPU 探针（positive）==")
    q = cpv["serial_quantiles_ms"]
    expect(cpv["status"] == "verified" and cpv["serial_count"] == 120 and cpv["probability_roundtrip_bitwise_equal"] is True, "CPU 探针状态不符")
    expect(near(q[0], 56.60125, 1e-4) and near(q[2], 302.681249, 1e-3), "CPU 分位不符")
    expect(bench["runtime"]["cuda_calls"] == 0 and bench["runtime"]["device"] == "cpu", "CPU 探针不是 0 CUDA")
    expect(near(bench["loading"]["first_load_wall_s"], 7.6197266, 1e-4) and near(bench["loading"]["second_load_wall_s"], 2.9589719, 1e-4), "加载时间不符")
    for token in ("2575.14", "6299.54", "21761.41", "unknown", "2C/4GB"):
        expect(token in cp_txt, f"CPU 报告缺少关键串: {token}")
    out(f"  serial n=120 p50 {q[0]:.2f} / p95 {q[1]:.2f} / p99 {q[2]:.2f} ms；加载 {bench['loading']['first_load_wall_s']:.3f}→{bench['loading']['second_load_wall_s']:.3f} s；0 CUDA")
    out("  RSS peak 2575.14 MiB（含 reload）、batch32 p95 6299.54 ms / wall 21761.41 ms、2C/4GB 余量 unknown —— 文本核对 OK")

    out("")
    out("== 6. 强范围轮 full 通过链（positive，保留旧失败因果；历史，非 current）==")
    gov = json.loads((mirror / "governance-fix-terminal-summary-validated.json").read_text(encoding="utf-8"))
    strong = json.loads((mirror / "full-strong-scope-20261012.json").read_text(encoding="utf-8"))
    expect(gov["status"] == "VALIDATED" and gov["full_process_exit_code"] == 0, "治理修复终态不符")
    expect(near(gov["full_process_elapsed_seconds"], 1089.6558032, 1e-4), "full 耗时不符")
    expect(len(gov["gates"]) == 14 and all(g["status"] == "passed" for g in gov["gates"]), "14 门不齐")
    expect(strong["status"] == "passed" and len(strong["gates"]) == 14 and all(g["status"] == "passed" for g in strong["gates"]),
           "strong-scope artifact 不是 14/14 passed")
    expect(gov["pytest_suite"] == {"passed": 6670, "skipped": 21}, "pytest 6670/21 不符")
    expect(gov["fresh_skip_reason_count"] == 21, "fresh skip 计数不符")
    gd = next(g for g in gov["gates"] if g["id"] == "agent-governance")
    expect(near(gd["duration_seconds"], 2.844, 1e-3) and gov["direct_root_governance"]["elapsed_seconds"] > 2.8
           and near(gov["direct_root_governance"]["elapsed_seconds"], 2.828, 1e-2), "governance 时长不符")
    expect(gov["direct_root_governance"]["unchanged_budget_seconds"] == 600
           and gov["policy_manifest_runner_thresholds_changed"] is False, "600 预算被改")
    expect(gov["red_regression"] == {"passed": 59, "failed": 10, "errors": 0, "skipped": 0}
           and gov["green_regression"] == {"passed": 69, "failed": 0, "errors": 0, "skipped": 0}, "红→绿因果不符")
    expect(gov["old_timeout_600_elapsed_855_history_preserved"] is True
           and gov["old_governance_stdout_has_pass_but_timeout_failed"] is True, "旧超时失败未保留")
    expect(gov["original_failed_report_sha256"] == "092a1d28a6dd79d226082450cf987f55c0ca39e49968239bd8113bdd895e9999",
           "旧失败报告哈希不符")
    expect(gov["protected_unchanged_count"] == 1207, "保护路径计数不符")
    expect(gov["critical_sha256"][str(repo_root / "research/memory_relations/runs/v3-study-lock-20261012/lock.json")]
           == "48f6a42cba47ef089769ced67308c38db1d297e7d91babd7f214230d16afae24", "锁哈希不符")
    expect(gov["critical_sha256"][str(repo_root / "research/memory_relations/runs/v3-data-ablation-20261012/new-adjudicated/scorers/finetuned-joint-size3290-seed13/model.pt")]
           == "3b5403da2366bf8f4144ac95ebe6793d13cc82767cf55023cfe0ba4a21167498", "权重哈希不符")
    expect(gov["model_data_locks_predictions_changed"] is False and gov["formal_prediction_training_audit_calls"] == 0
           and gov["research_698_repeated"] is False and gov["fixture_cleanup"] == "none", "边界标志不符")
    expect(gov["full_report_sha256"] == sha256(artifact_strong), "strong-scope artifact 哈希与终态不符")
    for cond, name in ((gov["full_process_exit_code"] != 0, "negative-d full exit!=0 冒充"), (6670 == 6601, "negative-e 6670==6601")):
        try:
            expect(cond, f"{name} fired")
            raise AssertionError(f"{name} did not raise")
        except AssertionError as exc:
            if str(exc) == f"{name} did not raise":
                raise
    out(f"  full exit 0、14/14 passed、{gov['full_process_elapsed_seconds']:.1f}s；pytest {gov['pytest_suite']['passed']}/{gov['pytest_suite']['skipped']} skip（fresh reason {gov['fresh_skip_reason_count']} 条）")
    out(f"  agent-governance 门 {gd['duration_seconds']:.3f}s、独立 root 运行 {gov['direct_root_governance']['elapsed_seconds']:.3f}s、预算 600 未改")
    out(f"  旧链保留：timeout 600/855 → P1×2 → 红 59/10 → 绿 69/0/0/0；旧失败报告 {gov['original_failed_report_sha256'][:16]}… 仍在")
    out("  保护 1207 路径不变；锁/权重/markers 哈希一致；0 formal calls、698 未重跑、无清理、无部署（历史轮）")

    out("")
    out("== 7. 分类闭合轮（positive，最终 current）==")
    cfr = json.loads((mirror / "full-classification-scope-20261012.json").read_text(encoding="utf-8"))
    expect(cls["status"] == "VALIDATED_CLASSIFICATION_CLOSED", "分类闭合 status 不符")
    expect(cls["full_exit_code"] == 0 and near(cls["full_elapsed_seconds"], 1677.1645245, 1e-3), "full exit/耗时不符")
    expect(cls["full_report_sha256"] == sha256(artifact_classification), "classification artifact 哈希不符")
    expect(len(cls["gates"]) == 14 and all(g["status"] == "passed" for g in cls["gates"]), "14 门不齐")
    expect(len(cfr["gates"]) == 14 and all(g["status"] == "passed" for g in cfr["gates"]), "artifact 14 门不齐")
    gdc = next(g for g in cls["gates"] if g["id"] == "agent-governance")
    expect(near(gdc["duration_seconds"], 6.39, 1e-6), "分类轮 governance 门时长不符")
    expect(cls["pytest_suite"] == {"passed": 6672, "skipped": 21}, "pytest 6672/21 不符")
    expect(cls["automation_contracts"] == {"passed": 185, "skipped": 3}, "automation 185/3 不符")
    expect(cls["installer_contracts"] == {"passed": 120, "skipped": 2}, "installer 120/2 不符")
    expect(cls["governance_tests"] == {"passed": 69, "failed": 0, "skipped": 0}, "治理 69/0/0 不符")
    expect(cls["targeted_classification_tests"] == {"passed": 121, "failed": 0, "skipped": 3}, "targeted 121/0/3 不符")
    expect("classification-skip-reasons" in cls["fresh_skip_reasons"], "fresh skip reasons 未指向 classification 现场")
    before, after = cls["public_before"], cls["public_after"]
    expect(before["dirty_certification_rejected"] is True and cls["dirty_release_certification"] is False, "dirty 认证拒绝标志不符")
    b0, b1 = before["profiles"]
    a0, a1 = after["profiles"]
    expect((b0["profile"], b0["file_count"], b0["pytest_returncode"], b0["passed"], b0["skipped"]) == ("assembled", 815, 1, 1, 0)
           and (b1["profile"], b1["file_count"], b1["pytest_returncode"], b1["passed"], b1["skipped"]) == ("skeleton", 755, 1, 1, 0),
           "before 两 profile 真实失败现场不符")
    expect((a0["profile"], a0["file_count"], a0["pytest_returncode"], a0["passed"], a0["skipped"]) == ("assembled", 814, 0, 2, 0)
           and (a1["profile"], a1["file_count"], a1["pytest_returncode"], a1["passed"], a1["skipped"]) == ("skeleton", 754, 0, 2, 0),
           "after 两 profile 全绿现场不符")
    expect(cls["manual_public_copy"] is False, "manual_public_copy 不应为 True")
    expect(cls["protected_source_model_data_lock_prediction_unchanged"] == 1205, "保护计数不是 1205")
    expect(cls["critical_sha256"][str(repo_root / "research/memory_relations/runs/v3-study-lock-20261012/lock.json")]
           == "48f6a42cba47ef089769ced67308c38db1d297e7d91babd7f214230d16afae24", "锁哈希不符")
    expect(cls["critical_sha256"][str(repo_root / "research/memory_relations/runs/v3-data-ablation-20261012/new-adjudicated/scorers/finetuned-joint-size3290-seed13/meta.json")]
           == "7ab44014d1b84b6e50be45e84ddb33685661b00bc3017544e33b39e4b544f948", "meta 哈希不符")
    expect(cls["critical_sha256"][str(repo_root / "research/memory_relations/runs/v3-data-ablation-20261012/new-adjudicated/scorers/finetuned-joint-size3290-seed13/model.pt")]
           == "3b5403da2366bf8f4144ac95ebe6793d13cc82767cf55023cfe0ba4a21167498", "权重哈希不符")
    expect(cls["cached_only_original_governance_test"] is True and cls["staged_test_removed_or_modified"] is False
           and cls["staged_test_sha256"] == "2f27d220a8a221773ffe10e31a6e7d1f06a50501ece026083a04b5e8db2fa95b", "staged 测试契约不符")
    expect(cls["app_runtime_diff"] == "empty" and cls["cuda_mask_added"] is False
           and cls["gpu_training_final_evaluation_probe_run"] is False
           and cls["research_698_repeated"] is False and cls["formal_prediction_training_audit_calls"] == 0
           and cls["fixture_cleanup"] == "none", "分类轮边界标志不符")
    expect(cls["commit_push_reset_checkout_server_deploy_install_download"] is False, "不应有 commit/push/deploy")
    expect("READY_TO_MERGE" in cls["writer_status"], "writer_status 未保留")
    out(f"  full exit 0、14/14 passed、{cls['full_elapsed_seconds']:.4f}s；pytest {cls['pytest_suite']['passed']}/{cls['pytest_suite']['skipped']} skip")
    out(f"  automation {cls['automation_contracts']['passed']}/{cls['automation_contracts']['skipped']}、"
        f"installer {cls['installer_contracts']['passed']}/{cls['installer_contracts']['skipped']}、"
        f"治理 {cls['governance_tests']['passed']}/0/0、targeted {cls['targeted_classification_tests']['passed']}/0/3")
    out(f"  公开投影 before→after：{b0['profile']} {b0['file_count']}件 rc={b0['pytest_returncode']} pass={b0['passed']} → "
        f"{a0['file_count']}件 rc={a0['pytest_returncode']} pass={a0['passed']}；"
        f"{b1['profile']} {b1['file_count']}件 rc={b1['pytest_returncode']} pass={b1['passed']} → "
        f"{a1['file_count']}件 rc={a1['pytest_returncode']} pass={a1['passed']}（前红后绿、0 skip）")
    out(f"  治理门 {gdc['duration_seconds']:.3f}s、预算 600 未改；保护 1205 不变；dirty 认证拒绝（非 clean 发布）")

    out("")
    out("== 8. 镜像覆盖三类区分（aggregate / synthetic / artifacts）==")
    syn_files, syn_bytes = dir_stats(synthetic_dirs)
    total_files, total_bytes = dir_stats([repro])
    ev_files = ev_bytes = 0
    for rel in DERIVED_EVIDENCE:
        p = repro / rel
        if p.is_file():
            ev_files += 1
            ev_bytes += p.stat().st_size
    art_files = total_files - syn_files - ev_files
    art_bytes = total_bytes - syn_bytes - ev_bytes
    mir_files = len(manifest)
    mir_bytes = sum(f["bytes"] for f in manifest)
    repro_manifest_sha = sha256(repro_manifest) if repro_manifest.exists() else None
    supp_manifest_sha = sha256(repro_supplement) if repro_supplement.exists() else None
    expect(syn_files == 15, f"合成原始输入件数不是 15（实际 {syn_files}）")
    expect(repro_manifest.exists() and repro_supplement.exists(), "复现档案 manifest/补充 manifest 缺失")
    out(f"  聚合镜像（本目录）: {mir_files} 件 / {mir_bytes} B —— 逐字节 mirror==source，保存不改变 fresh 属性")
    out(f"  合成原始输入: {syn_files} 件 / {syn_bytes} B（数据-冻结三臂 + 数据-终评参考；已曝光、human_gold=false）")
    out(f"  研究工件（payload）: {art_files} 件 / {art_bytes} B（协议/源码快照/缓存/验收/锁定工件/复算等；不含派生复核证据）")
    out(f"  派生复核证据（明示不计入 payload）: {ev_files} 件 / {ev_bytes} B —— " + "、".join(DERIVED_EVIDENCE))
    out(f"  基线 manifest sha256 {repro_manifest_sha[:16]}…；补充 manifest sha256 {supp_manifest_sha[:16]}…")
    out("  边界：归档保存≠fresh；文件级齐全≠已做可移植/重载验收（跨平台指纹阻塞见补充 manifest）。")

    index = {
        "schema": "memory-relations-mirror-index-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "脚本/final-report-recheck-2026-10-12.py",
        "coverage_classes": {
            "aggregate_mirror": {
                "location": "审计镜像-终评与CPU探针-2026-10-12/",
                "description": "报告引用的小聚合件，逐字节镜像（mirror==source）；保存不改变 fresh 属性",
                "file_count": mir_files, "bytes": mir_bytes, "fresh": False,
            },
            "synthetic_raw_inputs": {
                "location": "复现资料-2026-10-12/数据-冻结三臂/ + 数据-终评参考/",
                "description": "合成冻结输入（已曝光；human_gold=false）；逐件 SHA 见复现档案 manifest",
                "manifest": "复现资料-2026-10-12/manifest-复现档案-2026-10-12.json",
                "manifest_sha256": repro_manifest_sha,
                "file_count": syn_files, "bytes": syn_bytes, "fresh": False,
            },
            "research_artifacts": {
                "location": "复现资料-2026-10-12/（协议-冻结/源码快照-*/缓存-概率与统计/验收-门与修复/锁定工件/复算 等）",
                "description": "研究源码快照、协议、统计缓存、验收证据与锁定工件；逐件 SHA 见基线+补充 manifest",
                "manifest": "复现资料-2026-10-12/manifest-补充-锁定工件与分类闭合-2026-10-12.json",
                "manifest_sha256": supp_manifest_sha,
                "file_count": art_files, "bytes": art_bytes, "fresh": False,
                "derived_evidence_excluded": {
                    "description": "派生复核证据明示不计入 payload 计数（保持计数可复算）；仍受档案清单并集与交付索引盘点",
                    "files": DERIVED_EVIDENCE, "file_count": ev_files, "bytes": ev_bytes,
                },
            },
        },
        "boundaries": [
            "fresh=false：全部为已曝光材料，保存不把已曝光终测变成 fresh/未见。",
            "文件级归档不等于可移植/重载验收：未做加载或迁移验证。",
            "跨平台指纹阻塞（windows 渲染 1416f5fa… vs POSIX 渲染 980924bf…）见补充 manifest。",
        ],
        "files": manifest,
    }
    index_path = mirror_out / INDEX_NAME
    assert_inside(out_root, index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    out(f"  [WRITE] 镜像清单-终评与CPU探针-2026-10-12.json（schema v2，files={mir_files}）")

    out_txt = out_root / RESULTS_DIR / OUT_TXT_NAME
    assert_inside(out_root, out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwritten: {out_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
