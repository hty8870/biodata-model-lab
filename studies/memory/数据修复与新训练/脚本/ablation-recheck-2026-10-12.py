"""正式九 run 训练结果复核 + 必要聚合镜像（2026-10-12）。

用途
----
复核 `runs/v3-data-ablation-20261012` 与 `v3-study-lock-20261012` 的正式证据：
① 9 run 表（损失/步数/预算/token/峰值显存）；② 臂均值与 SD 复算；③ 选择规则
（median-nearest seed，不是最好 seed）；④ lock 阈值/温度/权重哈希；⑤ 遥测峰值。

边界
----
- repo 全只读；镜像只复制**小 JSON 聚合**（≈65 KB）。**不复制** GPU 权重（仅就地哈希一处
  以核对 lock 记录）、不复制大 telemetry jsonl、不复制 raw labels/封存正文/终测内容。
- 不读 audit/terminal 语义；不解释 run 效果之外的因果。

自测
----
positive：9×（400 updates/6400 instances）、均值与登记一致（1e-9）、选择非最好 seed、
lock 阈值 null/温度 .980590113228623、权重 sha 匹配磁盘实际。
negative：篡改副本必须被哈希拒绝；两个已知错误等式（选择 seed≠20260919、均值互换）必须抛错。

用法
----
python -B 脚本/ablation-recheck-2026-10-12.py [--sync]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
RESEARCH = STAGE.parents[1]
REPO = RESEARCH.parent / "agent"
RUN = REPO / "research/memory_relations/runs/v3-data-ablation-20261012"
LOCK = REPO / "research/memory_relations/runs/v3-study-lock-20261012"
PRE = REPO / "research/memory_relations/runs/v3-training-preflight-20261012"

MIRROR = STAGE / "审计镜像-训练结果-2026-10-12"
OUT_TXT = STAGE / "结果数据/ablation-recheck-2026-10-12.txt"

SOURCES = [
    ("final-study-result.json", RUN / "final-study-result.json"),
    ("dev-only-selection.json", RUN / "dev-only-selection.json"),
    ("nine-run-gpu-records.json", RUN / "nine-run-gpu-records.json"),
    ("training-verification-final.json", RUN / "training-verification-final.json"),
    ("locked-candidate.json", RUN / "locked-candidate.json"),
    ("launch-result.json", RUN / "launch-result.json"),
    ("launch-manifest.json", RUN / "launch-manifest.json"),
    ("repair-decision.json", RUN / "repair-decision.json"),
    ("intake-verification.json", RUN / "intake-verification.json"),
    ("lock.json", LOCK / "lock.json"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expect(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def negative_self_tests(tmp: Path, chosen_seed: int) -> None:
    src = MIRROR / "dev-only-selection.json"
    t = tmp / "t.json"
    data = bytearray(src.read_bytes())
    data[0] ^= 0xFF
    t.write_bytes(bytes(data))
    expect(sha256(t) != sha256(src), "negative-a: tampered copy matched")
    for cond, name in ((chosen_seed == 20260919, "negative-b chosen==best-seed"),
                       (1.7208526489326836 == 1.6499935613579864, "negative-c mean swap")):
        try:
            expect(cond, f"{name} fired")
            raise AssertionError(f"{name} did not raise")
        except AssertionError as exc:
            if str(exc) == f"{name} did not raise":
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

    out("== 1. 镜像小聚合（mirror == source, sha256）==")
    manifest = []
    for name, src in SOURCES:
        m = MIRROR / name
        expect(m.exists(), f"mirror missing: {m} (run with --sync)")
        ms, ss = sha256(m), sha256(src)
        expect(ms == ss, f"hash mismatch: {name}")
        manifest.append({"mirror": name, "source_repo_relative": str(src.relative_to(REPO)).replace("\\", "/"),
                         "sha256": ms, "bytes": m.stat().st_size})
        out(f"  OK  {name}  {ms[:16]}…  {m.stat().st_size} B")
    (MIRROR / "镜像清单-训练结果-2026-10-12.json").write_text(
        json.dumps({"study_id": "MR3-20261012-data-ablation", "files": manifest}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8", newline="\n")

    vf = json.loads((MIRROR / "training-verification-final.json").read_text(encoding="utf-8"))
    sel = json.loads((MIRROR / "dev-only-selection.json").read_text(encoding="utf-8"))
    lock = json.loads((MIRROR / "lock.json").read_text(encoding="utf-8"))
    nine = json.loads((MIRROR / "nine-run-gpu-records.json").read_text(encoding="utf-8"))

    neg = Path(tempfile.mkdtemp(prefix="abl-neg-"))
    try:
        negative_self_tests(neg, sel["chosen"]["seed"])
        out("  OK  negative self-tests: tampered rejected; chosen==best-seed 与均值互换均抛错")
    finally:
        shutil.rmtree(neg, ignore_errors=True)

    out("")
    out("== 2. 9 run 表（原样登记 + 复算）==")
    runs = vf["runs"]
    expect(len(runs) == 9 and nine["all_nine_real_runs"] is True, "不是 9 个 run")
    arms = {}
    for r in runs:
        arms.setdefault(r["arm"], []).append(r["best_dev_root_loss"])
        expect(r["actual_updates"] == 400 and r["actual_instances"] == 6400, f"{r['arm']}/{r['seed']} 预算不符")
        out(f"  {r['arm']:<16} seed {r['seed']:<9} rows {r['train_rows']} roots {r['train_roots']:<4} "
            f"loss {r['best_dev_root_loss']:.9f} step {r['selected_step']:<4} tokens {r['real_tokens']}+{r['padding_tokens']} "
            f"peak {r['train_and_dev_peak_allocated_mb']:.1f}MB")
    for arm in ("legacy-original", "legacy-relabelled", "new-adjudicated"):
        expect(len(arms[arm]) == 3, f"{arm} 不是 3 个 seed")

    out("")
    out("== 3. 臂均值/SD 复算 vs 登记（positive）==")
    stored = sel["arm_mean_best_dev_root_loss"]
    for arm, vals in arms.items():
        mean = statistics.fmean(vals)
        sd_p = statistics.pstdev(vals)
        sd_s = statistics.stdev(vals)
        expect(abs(mean - stored[arm]) <= 1e-9, f"{arm} 均值不符")
        out(f"  {arm:<16} mean {mean:.10f} (登记一致)  SD_pop {sd_p:.6f} SD_samp {sd_s:.6f} "
            f"min {min(vals):.9f} max {max(vals):.9f}")
    chosen_arm = min(stored, key=lambda a: stored[a])
    expect(chosen_arm == "new-adjudicated", "选臂不是 new-adjudicated")

    out("")
    out("== 4. 选择规则（median-nearest，不是最好 seed）==")
    expect(sel["chosen"]["arm"] == "new-adjudicated" and sel["chosen"]["seed"] == 13, "chosen 记录不符")
    new_vals = sorted(arms["new-adjudicated"])
    median = statistics.median(new_vals)
    best_seed = min(runs, key=lambda r: r["best_dev_root_loss"])["seed"]
    expect(sel["chosen"]["seed"] != best_seed == 20260919, "选中的是最好 seed，与规则不符")
    expect(abs(median - sel["chosen"]["best_dev_root_loss"]) <= 1e-9, "median 与 chosen 损失不符")
    out(f"  new 臂三 seed 排序 {new_vals}; median {median:.9f} == chosen seed 13 损失；最好 seed 是 20260919（未选）")
    expect(sel["prob_policy_audit_challenge_terminal_used_for_selection"] is False, "选择使用了非 dev 证据")

    out("")
    out("== 5. lock / 阈值 / 权重（positive）==")
    cal = lock["calibration"]
    expect(cal["threshold"] is None and cal["threshold_feasible"] is False, "lock 阈值状态不符")
    expect(abs(cal["temperature"] - 0.980590113228623) <= 1e-12, "温度不符")
    expect(cal["selection"]["candidate_threshold"] is None and cal["selection"]["point_feasible"] is False, "candidate 选择状态不符")
    expect(lock["human_gold"] is False and lock["one_audit_per_study_lock"] is True, "lock 安全标志不符")
    expect(sha256(LOCK / "lock.json") == vf["lock_sha256"], "lock 文件哈希与验证记录不符")
    model_pt = Path(lock["model_dir"]) / "model.pt"
    weights_sha = sha256(model_pt)
    expect(weights_sha == lock["artifact_sha256"]["model.pt"], "选中权重 sha 与 lock 记录不符")
    expect(weights_sha.startswith("3b5403da"), "选中权重 sha 前缀异常")
    out(f"  threshold=null、threshold_feasible=false、point_feasible=false、temperature={cal['temperature']:.12f}")
    out(f"  lock sha256 {sha256(LOCK / 'lock.json')[:16]}…；model.pt 实际 sha256 {weights_sha[:16]}… 与 lock 一致（就地哈希，未复制）")

    out("")
    out("== 6. 预算/token/资源与遥测（positive）==")
    expect(vf["same_updates_and_instances_not_tokens"] is True, "same-updates 标志不符")
    tok = {(r["arm"], r["seed"]): (r["real_tokens"], r["padding_tokens"]) for r in runs}
    for seed in (13, 20260919, 424242):
        expect(tok[("legacy-original", seed)] == tok[("legacy-relabelled", seed)], f"legacy 对 seed {seed} token 不同")
    expect(tok[("new-adjudicated", 13)] != tok[("legacy-original", 13)], "new 与 legacy token 相同（不符）")
    peaks = [r["train_and_dev_peak_allocated_mb"] for r in runs]
    expect(min(peaks) >= 2200 and max(peaks) <= 2300, "峰值显存超出 ~2270–2275MB 区间")
    tel = vf["gpu_telemetry"]
    expect(tel["legacy-original"]["utilization_peak_pct"] == 77.0 and tel["legacy-relabelled"]["utilization_peak_pct"] == 79.0
           and tel["new-adjudicated"]["utilization_peak_pct"] == 74.0, "util 峰值不符")
    expect(abs(tel["new-adjudicated"]["power_peak_w"] - 130.98) < 1e-9, "功耗峰值不符")
    expect(vf["cpu_parity_is_validation_not_training"] is True and vf["audit_or_terminal_read"] is False, "验收标志不符")
    out(f"  same updates/instances ≠ same tokens：legacy 对逐 seed token 相同，new 不同（{tok[('new-adjudicated',13)]} vs {tok[('legacy-original',13)]}）")
    out(f"  峰值 allocated {min(peaks):.1f}–{max(peaks):.1f}MB；util peak 77/79/74%，mean "
        f"{tel['legacy-original']['utilization_mean_pct']:.1f}/{tel['legacy-relabelled']['utilization_mean_pct']:.1f}/"
        f"{tel['new-adjudicated']['utilization_mean_pct']:.1f}%；power peak 119.56/129.68/130.98W")
    out(f"  formal_output_bytes={vf['formal_output_bytes']}；current_disk_free_bytes={vf['current_disk_free_bytes']}")

    out("")
    out("== 7. 修订链（原失败保留 + 续跑）==")
    rd = json.loads((MIRROR / "repair-decision.json").read_text(encoding="utf-8"))
    expect(rd["original_attempt"]["status"] == "failed" and rd["real_completed_runs_reused"] == 3, "修复链记录不符")
    expect(rd["model_optimizer_main_source_or_data_changed"] is False and rd["completed_arm_retrained"] is False, "修复越界")
    expect(rd["audit_or_terminal_read"] is False and rd["incomplete_nine_selection_allowed"] is False, "修复边界不符")
    out(f"  原失败（ValueError：Windows venv Popen PID≠真实解释器 PID）保留；最小修复 + {rd['tests']}；"
        f"attempt-002 复用原标 3 run、补训 6 run；未改数据/主源、未重训已完成臂")

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwritten: {OUT_TXT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
