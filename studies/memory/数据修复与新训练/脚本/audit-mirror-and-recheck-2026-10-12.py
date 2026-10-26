"""审计镜像同步 + 字节哈希复核 + 正反算术自测（2026-10-12）。

用途
----
把「真实独立盲判导入」阶段的关键**审计聚合**镜像到研究目录（只报计数/哈希的聚合文件），
并复核：镜像与源逐字节一致、六层计数闭合、代理一致率与队列算术正确、
以及代码 QA 证据（642 专项 / full 14 门）的来源哈希一致。

边界
----
- repo 全只读；本脚本只读 repo/worktree 源文件，只写本阶段研究目录（镜像 + 结果输出）。
- 只镜像聚合件：不复制旧 runs、不复制任何 shard 正文、封存行、私有 map/ledger 或教师回复。
- `--sync` 才会写镜像；默认只验证并输出报告。

自测纪律
--------
positive：由文件真值复算全部比率/闭合关系，与文件内登记值逐项比对（容差 1e-9）。
negative：a) 篡改副本必须被哈希检查拒绝；b) 已知错误等式必须抛错（正反例都跑）。

用法
----
python -B 脚本/audit-mirror-and-recheck-2026-10-12.py [--sync]
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

MIRROR = STAGE / "审计镜像-import-2026-10-12"
OUT_TXT = STAGE / "结果数据/import-recheck-2026-10-12.txt"

SOURCES = [
    ("collected-source-manifest.json", WT / "research/memory_relations/runs/v3-collected/source-manifest.json"),
    ("collected-trainable-manifest.json", WT / "research/memory_relations/runs/v3-collected/trainable/manifest.json"),
    ("collected-quarantine-conflicts.json", WT / "research/memory_relations/runs/v3-collected/quarantine-conflicts.json"),
    ("prepare-verification.json", WT / "research/memory_relations/runs/v3-blind-prepare/verification.json"),
    ("import-verification.json", WT / "research/memory_relations/runs/v3-blind-import/reports/verification.json"),
    ("import-summary-final.json", WT / "research/memory_relations/runs/v3-blind-import/reports/summary-final.json"),
    ("import-receipt-audit-source-manifest.json", WT / "research/memory_relations/runs/v3-blind-import/provenance/receipt-audit-source-manifest.json"),
    ("import-adjudication-manifest.json", WT / "research/memory_relations/runs/v3-blind-import/adjudication/manifest.json"),
    ("integration-verification-final.json", REPO / "research/memory_relations/runs/v3-coordination/verification-final.json"),
]

FULL_ARTIFACT = REPO / "artifacts/quality-memory-relations-v3-integration.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def expect(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def negative_self_tests(tmp: Path) -> None:
    src = MIRROR / "import-verification.json"
    tampered = tmp / "tampered.json"
    data = bytearray(src.read_bytes())
    data[0] = data[0] ^ 0xFF
    tampered.write_bytes(bytes(data))
    expect(sha256(tampered) != sha256(src), "negative-a failed: tampered copy matched")
    try:
        expect(246 + 24 + 22 + 128 == 419, "negative-b failed to fire")
        raise AssertionError("negative-b did not raise")
    except AssertionError as exc:
        if str(exc) == "negative-b did not raise":
            raise
    try:
        expect(3262 == 3114, "negative-c failed to fire")
        raise AssertionError("negative-c did not raise")
    except AssertionError as exc:
        if str(exc) == "negative-c did not raise":
            raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true", help="把源聚合复制进镜像目录")
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

                     
    out("== 1. 镜像字节哈希复核（mirror == source, sha256）==")
    manifest = []
    for name, src in SOURCES:
        m = MIRROR / name
        expect(m.exists(), f"mirror missing: {m} (run with --sync)")
        ms, ss = sha256(m), sha256(src)
        expect(ms == ss, f"hash mismatch: {name}")
        manifest.append({
            "mirror": name,
            "source_repo_relative": str(src.relative_to(REPO)).replace("\\", "/"),
            "sha256": ms,
            "bytes": m.stat().st_size,
        })
        out(f"  OK  {name}  {ms[:16]}…  {m.stat().st_size} B")
    (MIRROR / "镜像清单-2026-10-12.json").write_text(
        json.dumps({"synced_from": str(WT.relative_to(REPO)).replace("\\", "/"), "files": manifest},
                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")

    neg_dir = Path(tempfile.mkdtemp(prefix="mr-neg-"))
    try:
        negative_self_tests(neg_dir)
        out("  OK  negative self-tests: tampered-hash rejected; wrong equations raised")
    finally:
        shutil.rmtree(neg_dir, ignore_errors=True)

    load = lambda n: json.loads((MIRROR / n).read_text(encoding="utf-8"))
    collected = load("collected-source-manifest.json")
    verify = load("import-verification.json")
    summary = load("import-summary-final.json")
    receipts = load("import-receipt-audit-source-manifest.json")
    adjud = load("import-adjudication-manifest.json")
    integ = load("integration-verification-final.json")

               
    out("")
    out("== 2. 六层/总量闭合（positive）==")
    per_split = {k: v["rows"] for k, v in collected["counts"].items()}
    total = sum(per_split.values())
    out(f"  generated rows by split: {per_split}  total={total}")
    expect(total == 5017 and collected["actual_rows"] == 5017, "generated total != 5017")
    expect(collected["actual_roots"] == 400, "generated roots != 400")
    expect(total == 3360 + 405 + 380 + 445 + 427, "split sum mismatch")

    struct = summary["structural_counts"]
    shard_total = sum(v["completed_shards"] for v in struct.values())
    row_total = sum(v["expected_rows"] for v in struct.values())
    expect(shard_total == 85 == verify["actual_receipt_shards"], "shard count mismatch")
    expect(row_total == 5285, "structural rows != 5285")
    expect(row_total == verify["counts"]["records"] == verify["counts"]["export_attempts"] == verify["counts"]["import_attempts"], "verification counts mismatch")
    expect(len(receipts["files"]) == 85, "receipt manifest files != 85")
    expect(sorted(receipts["ledgers"].keys()) == [f"p0{i}" for i in range(1, 8)], "ledger pools != p01..p07")
    expect(5285 == 5145 + 140, "5285 != DS5145 + primary140")
    expect(5145 == 5017 + 128, "5145 != new5017 + legacy128")
    out(f"  85 receipts = {shard_total} shards; rows {row_total} = 5145(DS) + 140(primary); 5145 = 5017 + 128")

                               
    out("")
    out("== 3. 作者-DS 代理一致率复算（proxy only, not accuracy）==")
    for pool in ("p01", "p02", "p03", "p07"):
        a = summary["author_ds_nonsealed_proxy_agreement"][pool]
        c = a["counts"]
        r = c["rates_valid_only"]
        n = c["valid"]
        expect(n == a["expected"], f"{pool}: valid != expected")
        expect(c["disagree"] == n - c["full_agree"], f"{pool}: disagree != valid - full")
        for key, num in (("label", "label_agree"), ("related", "related_agree"),
                         ("evidence", "evidence_agree"), ("full", "full_agree")):
            rate = c[num] / n
            expect(close(rate, r[key]), f"{pool}: {key} rate mismatch")
        out(f"  {pool}: n={n} label {c['label_agree']}({r['label']:.4%}) related {c['related_agree']}({r['related']:.4%}) "
            f"evidence {c['evidence_agree']}({r['evidence']:.4%}) full {c['full_agree']}({r['full']:.4%}) disagree {c['disagree']}")
    p01 = summary["author_ds_nonsealed_proxy_agreement"]["p01"]["counts"]
    expect(p01["label_agree"] == 3262 and p01["full_agree"] == 3114, "train label/full counts drifted")
    out("  口径提醒：train 3360 中 label 一致 3262、四项全一致 3114 —— 两个不同口径，以 summary 为准")

              
    out("")
    out("== 4. 第三判队列算术（positive）==")
    dis = [summary["author_ds_nonsealed_proxy_agreement"][p]["counts"]["disagree"] for p in ("p01", "p02", "p03")]
    nonsealed_disagree = sum(dis)
    expect(nonsealed_disagree == 246 + 24 + 22 == 292, "nonsealed disagreement sum mismatch")
    expect(nonsealed_disagree + 128 == adjud["required_cases"] == 420, "required_cases != 292 + 128")
    expect(adjud["rows"] == 560 == 420 + 140, "queue rows != 560")
    expect(adjud["shards"][-1]["rows"] == 48 and len(adjud["shards"]) == 9, "shard layout != 8x64+48")
    expect(adjud["seed"] == 20261013 and adjud["status"] == "awaiting-independent-third-judgement-not-adjudicated", "adjudication state drifted")
    out(f"  queue 560 = disagreements {nonsealed_disagree} + legacy128 = 420, + consistent-sample 140; 9 shards; status {adjud['status']}")

                   
    out("")
    out("== 5. 先导 140 三方代理一致（positive）==")
    pc = summary["pilot_three_way"]["counts"]
    expect(pc["valid_three_way"] == 140 and pc["all_three_label"] == 131, "pilot counts drifted")
    expect(summary["pilot_three_way"]["human_gold"] is False and summary["pilot_three_way"]["accuracy"] is None, "pilot human-gold flag drifted")
    out(f"  author-DS {pc['author_ds_label']}/140, author-primary {pc['author_primary_label']}/140, DS-primary {pc['ds_primary_label']}/140, "
        f"three-way {pc['all_three_label']}/140 ({pc['all_three_label']/140:.4%}) —— proxy agreement, not accuracy")

                   
    out("")
    out("== 6. 集成/代码 QA 证据来源核对（positive）==")
    expect(integ["research_tests"]["passed"] == 642 and integ["research_tests"]["failed"] == 0
           and integ["research_tests"]["skipped"] == 0 and integ["research_tests"]["xfailed"] == 0, "642 专项数字不一致")
    expect(integ["full_gates_passed"] == 14, "full gates != 14")
    for flag in ("formal_training", "new_terminal_data_read", "commit_or_deploy"):
        expect(integ[flag] is False, f"{flag} should be False")
    art_hash = sha256(FULL_ARTIFACT)
    expect(art_hash == integ["full_report_sha256"], "full artifact hash mismatch")
    out(f"  642 passed/0 failed/0 skip/0 xfail; full {integ['full_gates_passed']}/14; artifact sha256 {art_hash[:16]}… 匹配")
    out("  口径：代码 QA/工程门只证明代码与合同，不是标签 gold，也不能替代语义裁决")

    out("")
    out("== 7. 状态（不改写）==")
    out(f"  import status: {verify['status']} | summary status: {summary['status']} | student_test_evaluated: {summary['student_test_evaluated']}")
    out(f"  primary_missing: {verify['primary_missing']} | queue_rows: {verify['queue_rows']}")

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwritten: {OUT_TXT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
