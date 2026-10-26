                       
"""复现资料归档校验（2026-10-12，标准库-only；不 torch / 不 nvidia / 不 pytest；v2 清单并集修订）。

v2 修订（review r1 修复）
----
- 清点改为**通用 manifest 清单并集**：baseline 87 + supplement 22 逐件 SHA/字节对账，
  再与**具名档案件表**（EXTRA_ALLOW：README、两份 manifest、复算工具/输出与派生复核证据）
  做集合等式（缺件与清单外文件都 fail）；不再有"复算/ 前缀整体跳过"的盲放。
- 新增两条对账：补充 manifest 逐件、补充 manifest 绑定基线 manifest SHA。
- 历史值说明：初版仅对 baseline 单清单为 `selftests 4/4; checks 43/43`；自补充 22 件与
  派生证据加入后，当前口径为并集清单（见 [DONE] 实际输出）。历史 43/43 不代表当前档案。

两段式：
  A. 自测（合成数据，正/反例）：哈希正反、分子分母、malformed 身份、统计口径换位守卫。
  B. 实跑（cached-data 校对）：对 复现资料-2026-10-12/ 内已归档的
     冻结数据 / 终评参考 / 概率与统计缓存 / 验收工件逐件复核：
     文件 SHA、行数、两旧臂 96+51 差异、参考覆盖与 667/871、667/872 口径、
     风险分子分母（行级 vs 根级不混用）、旧 set 23/37 与 24/861、
     锁定/审计（no_candidate，非 0 风险）、九 run 均值与曲线均值、
     full 门通过件与三次失败件（855 超时 / not tracked / private material）。

退出码：任何自测或校验失败即非 0。
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parents[1]
MANIFEST = ARCHIVE / "manifest-复现档案-2026-10-12.json"
MANIFESTS = [
    MANIFEST,
    ARCHIVE / "manifest-补充-锁定工件与分类闭合-2026-10-12.json",
]
                                                  
EXTRA_ALLOW = {
    "README-复现档案-2026-10-12.md",
    "manifest-复现档案-2026-10-12.json",
    "manifest-补充-锁定工件与分类闭合-2026-10-12.json",
    "复算/build-archive-2026-10-12.py",
    "复算/verify-archive-2026-10-12.py",
    "复算/archive-lock-assets-2026-10-12.py",
    "复算/输出-build-archive-2026-10-12.txt",
    "复算/输出-build-archive-2026-10-12-attempt1-secretstop.txt",
    "复算/manifest-attempt1-含secretstop-2026-10-12.json",
    "复算/输出-verify-archive-2026-10-12.txt",
    "复算/输出-archive-lock-assets-2026-10-12.txt",
                                                           
    "复算/输出-mr-stage-verify-2026-10-12.txt",
    "复算/whitelist-verify-2026-10-12.json",
}

self_test_count = 0
self_test_failed = 0
check_count = 0
check_failed = 0


def selftest(name):
    global self_test_count, self_test_failed
    self_test_count += 1

    def wrap(fn):
        try:
            fn()
            print(f"[SELFTEST-OK] {name}")
        except Exception as exc:                
            self_test_failed += 1
            print(f"[SELFTEST-FAIL] {name}: {exc!r}")
        return fn
    return wrap


def check(label, condition, evidence=""):
    global check_count, check_failed
    check_count += 1
    if condition:
        print(f"[CHECK-OK] {label} {evidence}")
    else:
        check_failed += 1
        print(f"[CHECK-FAIL] {label} {evidence}")


def expect_raises(fn, exc_types, note=""):
    try:
        fn()
    except exc_types:
        return True
    raise AssertionError(f"expected raise {exc_types} {note}")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_of(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


                                         

@selftest("hash 正/反：同字节匹配已知向量，篡改 1 字节必不匹配")
def t_hash():
    good = hashlib.sha256(b"abc").hexdigest()
    assert good == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    tampered = hashlib.sha256(b"abd").hexdigest()
    assert good != tampered

    def guard(actual, expected):
        if actual != expected:
            raise AssertionError("sha mismatch")
    guard(good, good)
    expect_raises(lambda: guard(tampered, good), AssertionError, "guard must reject tampered")


@selftest("分子分母：667/871 ≠ 667/872，换分母必须被守卫拦下")
def t_numerators():
    r871 = Fraction(667, 871)
    r872 = Fraction(667, 872)
    assert close(float(r871), 0.7657864523536165, 1e-15)
    assert close(float(r872), 0.7649082568807339, 1e-15)
    assert float(r871) != float(r872)

    def assert_expected(num, den, expected):
        if abs(float(Fraction(num, den)) - expected) > 1e-15:
            raise ValueError(f"ratio mismatch {num}/{den}")
    assert_expected(667, 871, 0.7657864523536165)
    expect_raises(lambda: assert_expected(667, 872, 0.7657864523536165), ValueError,
                  "swapping denominator must raise")


@selftest("malformed 身份：重复 id / 非 ASCII 数字 id / 缺键 必须被拒")
def t_identity():
    def validate(rows):
        seen = set()
        for row in rows:
            rid = row["id"]
            if not isinstance(rid, str) or not rid.isascii() or not rid.isdigit():
                raise ValueError(f"bad id {rid!r}")
            if rid in seen:
                raise ValueError(f"duplicate id {rid}")
            seen.add(rid)
        return len(seen)

    good = [{"id": "17001001"}, {"id": "17002011"}]
    assert validate(good) == 2
    expect_raises(lambda: validate([{"id": "17001001"}, {"id": "17001001"}]), ValueError, "dup")
    expect_raises(lambda: validate([{"id": "17a01"}]), ValueError, "alpha")
    expect_raises(lambda: validate([{"other": "1"}]), KeyError, "missing key")


@selftest("统计口径：行级 42/265=15.85% 与根级 15.49% 不得互换")
def t_stat_switch():
    row_rate = Fraction(42, 265)
    root_rate = 0.15493810384753712
    assert close(float(row_rate), 0.15849056603773584, 1e-15)
    assert not close(float(row_rate), root_rate, 1e-6)

    def as_row(numerator, denominator, root_value):
        if close(numerator / denominator, root_value, 1e-6):
            raise ValueError("row/root口径混淆")
        return numerator / denominator
    as_row(42, 265, root_rate)
    expect_raises(lambda: as_row(42, 265, 42 / 265), ValueError, "identical values must raise")


                                                        

def main() -> int:
    for m in MANIFESTS:
        if not m.is_file():
            print(f"[FATAL] manifest missing: {m.name}")
            return 2
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    supplement = json.loads(MANIFESTS[1].read_text(encoding="utf-8"))
    if self_test_failed:
        print(f"[SELFTEST] {self_test_count - self_test_failed}/{self_test_count} passed; aborting archive checks")
        return 1
    print(f"[SELFTEST] {self_test_count}/{self_test_count} passed")

                                          
    def audit_listing(entries: dict, label: str) -> int:
        bad = 0
        for arc, entry in entries.items():
            p = ARCHIVE / arc
            if not p.is_file():
                bad += 1
                print(f"[FILE-MISS] {arc}")
                continue
            h = sha256_of(p)
            if h != entry["sha256"] or (entry.get("bytes") is not None and p.stat().st_size != entry["bytes"]):
                bad += 1
                print(f"[FILE-BAD] {arc} {h[:12]} != {entry['sha256'][:12]}")
            elif arc.endswith(".jsonl"):
                n = sum(1 for _ in open(p, "rb"))
                if n != entry.get("rows"):
                    bad += 1
                    print(f"[ROWS-BAD] {arc} {n} != {entry.get('rows')}")
        check(label, bad == 0, f"files={len(entries)} bad={bad}")
        return bad

    listing = {e["archive_path"]: e for e in manifest["files"]}
    listing_supp = {e["archive_path"]: e for e in supplement["files"]}
    bad_files = audit_listing(listing, "manifest 逐件 SHA/字节/行数")
    bad_files += audit_listing(listing_supp, "补充 manifest 逐件 SHA/字节")
    check("补充 manifest 绑定基线 manifest SHA",
          supplement["base_manifest"]["sha256"] == sha256_of(MANIFEST)
          and supplement["base_manifest"]["path"] == MANIFEST.name,
          f"base={supplement['base_manifest']['sha256'][:12]}")

    expected = set(listing) | set(listing_supp) | set(EXTRA_ALLOW)
    walked = sorted(p.relative_to(ARCHIVE).as_posix() for p in ARCHIVE.rglob("*") if p.is_file())
    missing = sorted(expected - set(walked))
    unexpected = sorted(set(walked) - expected)
    check("清单并集对账（baseline 87 + 补充 22 + 具名档案件）",
          not missing and not unexpected,
          f"walked={len(walked)} missing={missing[:4]} unexpected={unexpected[:4]}")

             
    train_arc = {
        "legacy-original": "数据-冻结三臂/legacy-original.train.jsonl",
        "legacy-relabelled": "数据-冻结三臂/legacy-relabelled.train.jsonl",
        "new-adjudicated": "数据-冻结三臂/new-adjudicated.train.jsonl",
    }
    labels = {}
    for arm, arc in train_arc.items():
        rows = list(rows_of(ARCHIVE / arc))
        labels[arm] = rows
        ids = [r["id"] for r in rows]
        ok = len(set(ids)) == len(ids) and all(isinstance(i, str) and i for i in ids)
        if arm == "new-adjudicated":
            ok = ok and all(i.isascii() and i.isdigit() for i in ids)
        check(f"{arm} train 行数与 id 契约", ok, f"rows={len(rows)} id_sample={ids[0]!r}")
    check("train 行数 2522/2522/3290",
          [len(labels[a]) for a in ("legacy-original", "legacy-relabelled", "new-adjudicated")] == [2522, 2522, 3290])

    text_diff = label_diff = ev_diff = 0
    for a, b in zip(labels["legacy-original"], labels["legacy-relabelled"]):
        text_diff += a["request"] != b["request"]
        label_diff += a["label"] != b["label"]
        ev_diff += (a["evidence_refs"] != b["evidence_refs"]) or (a["related_ids"] != b["related_ids"])
    check("两旧臂同文本/96 label/51 evidence",
          (text_diff, label_diff, ev_diff) == (0, 96, 51), f"got {(text_diff, label_diff, ev_diff)}")

    for name, part, want_rows, want_groups in [
        ("shared.dev.jsonl", "dev", 405, 40), (("shared.prob.jsonl"), "prob", 116, 12),
        ("shared.policy.jsonl", "policy", 120, 13)]:
        rows = list(rows_of(ARCHIVE / "数据-冻结三臂" / name))
        groups = {r["group_id"] for r in rows}
        check(f"{name} 行数/组数", len(rows) == want_rows and len(groups) == want_groups,
              f"rows={len(rows)} groups={len(groups)}")
    mapping = manifest["shared_partition_mapping"]
    check("shared 只存 1 份并映射三臂",
          all(m["identical_sha_across_arms"] and len(m["used_by_arms"]) == 3 for m in mapping.values()),
          ",".join(mapping))

             
    exp = list(rows_of(ARCHIVE / "数据-终评参考/expected-cases-872.jsonl"))
    exp_ok = len(exp) == 872 and all(isinstance(r["source_id"], str) and r["source_id"].isascii()
                                     and r["source_id"].isdigit() for r in exp) \
        and len({r["source_id"] for r in exp}) == 872
    check("872 注册表（ASCII 数字 source_id 唯一）", exp_ok, f"rows={len(exp)}")
    ho = list(rows_of(ARCHIVE / "数据-终评参考/final-heldout.jsonl"))
    oo = list(rows_of(ARCHIVE / "数据-终评参考/final-ood.jsonl"))
    check("参考 444/427/872", len(ho) == 444 and len(oo) == 427 and len(ho) + len(oo) + 1 == 872,
          f"heldout={len(ho)} ood={len(oo)}")
    fz = json.loads((ARCHIVE / "数据-终评参考/freeze-manifest.json").read_text(encoding="utf-8"))
    check("冻结清单计数", fz["counts"] == {"accepted": 871, "accepted_uncertain": 98, "excluded": 1,
                                          "expected": 872, "unresolved": 0, "usable": 871} and fz["human_gold"] is False)
    tot = fz["labelled_counts"]["total"]
    check("七类合计", [tot[k] for k in ["complement", "conflict", "contextual", "duplicate", "supersede", "temporary", "uncertain"]]
          == [196, 106, 132, 118, 92, 129, 98], str(tot))
    st = list(rows_of(ARCHIVE / "数据-终评参考/reference_status.jsonl"))
    check("sidecar 仅 excluded（无 truth）", len(st) == 1 and st[0]["status"] == "excluded" and st[0]["human_gold"] is False)

                                 
    probs = list(rows_of(ARCHIVE / "缓存-概率与统计/terminal-probabilities.jsonl"))
    labelled = [r for r in probs if r["reference_label"] is not None]
    correct = [r for r in labelled if r["classifier_relation"] == r["reference_label"]]
    check("terminal-probabilities 行数/覆盖", len(probs) == 872 and len(labelled) == 871,
          f"rows={len(probs)} labelled={len(labelled)}")
    check("667/871=76.58% 与 667/872=76.49%",
          close(len(correct) / len(labelled), 0.7657864523536165, 1e-15)
          and close(len(correct) / len(probs), 0.7649082568807339, 1e-15),
          f"correct={len(correct)}")
    check("uncertain 参考保留 98", sum(1 for r in probs if r["reference_label"] == "uncertain") == 98)

                           
    summ = json.loads((ARCHIVE / "缓存-概率与统计/terminal-eval-verified-summary.json").read_text(encoding="utf-8"))
    terminal = summ["terminal"]
    h_row = terminal["heldout"]["row_operation"]
    h_root = terminal["heldout"]["root_operation"]["conditional_risk"]
    o_row = terminal["ood"]["row_operation"]
    o_root = terminal["ood"]["root_operation"]["conditional_risk"]
    check("heldout 行级 42/265=15.85%", h_row["dangerous_allow_count"] == 42 and h_row["allowing_pred_count"] == 265
          and close(h_row["conditional_risk"]["value"], 0.15849056603773584, 1e-15))
    check("heldout 根级 15.49% 且与行级不同",
          close(h_root["value"], 0.15493810384753712, 1e-12) and not close(h_root["value"], h_row["conditional_risk"]["value"], 1e-4))
    check("OOD 行级 16/213=7.51%", o_row["dangerous_allow_count"] == 16 and o_row["allowing_pred_count"] == 213
          and close(o_row["conditional_risk"]["value"], 0.07511737089201878, 1e-15))
    check("OOD 根级 7.53% 且与行级不同",
          close(o_root["value"], 0.07529839259128414, 1e-12) and not close(o_root["value"], o_row["conditional_risk"]["value"], 1e-4))

                 
    old = summ["old_set"]
    r = old["refs_by_role_row"]["gated"]["related_ids"]
    check("旧 set 23/37 与 24/861=2.79%",
          old["correct_completed_overall"] == {"numerator": 23.0, "denominator": 37.0, "value": 0.6216216216216216}
          and r["predicted_total"] == 861 and r["hits"] == 24 and close(r["precision"]["value"], 0.027874564459930314, 1e-15)
          and r["known_non_expected_total"] == 837 and r["expected_total"] == 28 and r["missed_expected_total"] == 4)
    by_rule = old["by_rule_count"]
    got = [(by_rule[k]["correct_completed"], by_rule[k]["planned"]) for k in ["0", "1", "8", "64", "65", "129"]]
    check("旧 set 按规则数 4/4,4/6,6/10,4/7,2/6,3/4", got == [(4, 4), (4, 6), (6, 10), (4, 7), (2, 6), (3, 4)], str(got))

             
    lock = json.loads((ARCHIVE / "缓存-概率与统计/study-lock.json").read_text(encoding="utf-8"))
    cal = lock["calibration"]
    check("锁：threshold=null 且 point_feasible=false（候选扫描无可行点）",
          cal["threshold"] is None and cal["threshold_feasible"] is False
          and cal["selection"]["point_feasible"] is False and cal["selection"]["point"] is None)
    check("锁：audit deferred（校准支持另限）", cal["audit"]["status"] == "deferred" and cal["audit"]["sufficient"] is False
          and cal["selection"]["prior"]["target_risk"] == 0.01)
    audit = json.loads((ARCHIVE / "缓存-概率与统计/single-audit.json").read_text(encoding="utf-8"))
    ab = audit["audit"]
    check("审计 no_candidate 且界全 null、368 要求可见",
          ab["status"] == "no_candidate" and ab["threshold"] is None and ab["conditional_risk_upper"] is None
          and ab["coverage_lower"] is None and ab["minimum_zero_error_allow_groups"] == 368
          and ab["enough_independent_support"] is False)

              
    tv = json.loads((ARCHIVE / "缓存-概率与统计/training-verification-final.json").read_text(encoding="utf-8"))
    runs = tv["runs"]
    check("九 run：3 臂 × 3 seed、400 updates/6400 实例",
          len(runs) == 9 and all(x["actual_updates"] == 400 and x["actual_instances"] == 6400 and x["device"] == "cuda:0" for x in runs))
    means = tv["selection"]["arm_mean_best_dev_root_loss"]
    computed = {}
    for arm in ("legacy-original", "legacy-relabelled", "new-adjudicated"):
        vals = [x["best_dev_root_loss"] for x in runs if x["arm"] == arm]
        computed[arm] = sum(vals) / len(vals)
        assert len(vals) == 3
    print("       arm means:", json.dumps(computed, ensure_ascii=False))
    check("臂均值 1.720853/1.649994/0.422099（复算=选点表，±5e-6 报告口径）",
          all(close(computed[a], means[a], 1e-12) for a in computed)
          and close(computed["legacy-original"], 1.720853, 5e-6)
          and close(computed["legacy-relabelled"], 1.649994, 5e-6)
          and close(computed["new-adjudicated"], 0.422099, 5e-6))
    cv = json.loads((ARCHIVE / "缓存-概率与统计/curve-verification-final.json").read_text(encoding="utf-8"))
    agg = cv["aggregate"]
    check("曲线 6 新 + 235 复用 3、均值 0.726767/0.496241",
          cv["six_new_real_cuda_runs"] == 6 and cv["main235_three_reused_no_retrain"] == 3
          and close(agg["60"]["root_dev_loss_mean"], 0.7267672881047551, 1e-9)
          and close(agg["120"]["root_dev_loss_mean"], 0.4962409253610873, 1e-9))

             
    strong = json.loads((ARCHIVE / "验收-门与修复-2026-10-12/quality-full-passed-strong-scope.json").read_text(encoding="utf-8"))
    gates = {g["id"]: g for g in strong["gates"]}
    check("full 14/14 passed（含 pytest 6670/21 与 governance 2.844s）",
          strong["status"] == "passed" and len(strong["gates"]) == 14
          and all(g["status"] == "passed" for g in gates.values())
          and "6670 passed, 21 skipped" in gates["pytest-suite"]["commands"][0]["stdout_tail"]
          and close(gates["agent-governance"]["duration_seconds"], 2.844, 1e-9)
          and gates["agent-governance"]["timeout_seconds"] == 600)
    frozen_tail = gates["frozen-recommendation-evaluation"]["commands"][0]["stdout_tail"]
    check("冻结基线 784/97.7/97.7/0/0/10", "97.7" in frozen_tail and "10/10" in frozen_tail and "784" in frozen_tail)
    orig = json.loads((ARCHIVE / "验收-门与修复-2026-10-12/quality-full-failed-original.json").read_text(encoding="utf-8"))
    og = {g["id"]: g for g in orig["gates"]}
    oc = og["agent-governance"]["commands"][0]
    check("原 full failed=855.063s 超 600s 但 stdout 含 PASSED",
          orig["status"] == "failed" and oc["status"] == "timeout"
          and close(oc["duration_seconds"], 855.063, 1e-6)
          and og["agent-governance"]["timeout_seconds"] == 600
          and "AGENT GOVERNANCE CHECK PASSED" in oc["stdout_tail"])
    fixed = json.loads((ARCHIVE / "验收-门与修复-2026-10-12/quality-full-failed-fixed.json").read_text(encoding="utf-8"))
    fg = {g["id"]: g for g in fixed["gates"]}
    check("fixed attempt failed=not tracked", "is not tracked: tests/test_agent_governance.py"
          in fg["public-mirror-source"]["commands"][0]["stderr_tail"])
    strong1 = json.loads((ARCHIVE / "验收-门与修复-2026-10-12/quality-full-failed-strong-attempt1.json").read_text(encoding="utf-8"))
    s1g = {g["id"]: g for g in strong1["gates"]}
    check("strong attempt1 failed=private material", "references private material" in s1g["public-mirror-source"]["commands"][0]["stderr_tail"])
    tsv = json.loads((ARCHIVE / "验收-门与修复-2026-10-12/terminal-summary-validated.json").read_text(encoding="utf-8"))
    check("治理 69 绿 / 59+10 红 / 预算不拆",
          tsv["green_regression"]["passed"] == 69 and tsv["green_regression"]["failed"] == 0
          and tsv["red_regression"]["passed"] == 59 and tsv["red_regression"]["failed"] == 10
          and tsv["direct_root_governance"]["unchanged_budget_seconds"] == 600)
    check("research 698 未重跑 / 0 预测训练审计调用 / 1207 保护件",
          tsv["research_698_repeated"] is False and tsv["formal_prediction_training_audit_calls"] == 0
          and tsv["protected_unchanged_count"] == 1207)
    check("终验收据与归档件 hash 互锁",
          tsv["full_report_sha256"] == listing["验收-门与修复-2026-10-12/quality-full-passed-strong-scope.json"]["sha256"]
          and tsv["original_failed_report_sha256"] == listing["验收-门与修复-2026-10-12/quality-full-failed-original.json"]["sha256"]
          and tsv["fixed_failed_report_sha256"] == listing["验收-门与修复-2026-10-12/quality-full-failed-fixed.json"]["sha256"]
          and tsv["strong_failed_attempt1_sha256"] == listing["验收-门与修复-2026-10-12/quality-full-failed-strong-attempt1.json"]["sha256"])

                                             
    vb = fz["version_binding"]
    check("冻结清单绑定权重 meta 与锁",
          vb["weights_artifact_sha256"]["meta.json"] == listing["缓存-概率与统计/selected-scorer-meta.json"]["sha256"]
          and vb["study_lock"]["sha256"] == listing["缓存-概率与统计/study-lock.json"]["sha256"])
    sb = json.loads((ARCHIVE / "协议-冻结/final-evaluation-source-binding.json").read_text(encoding="utf-8"))
    bind_ok = True
    for src_rel, expected in sb["protected_sha256"].items():
        name = src_rel.replace("\\", "/").rsplit("/", 1)[-1]
        arc = f"源码快照-research-memory-relations/{name}"
        if arc in listing and listing[arc]["sha256"] != expected:
            bind_ok = False
            print(f"       [BIND-DIFF] {name}")
    check("终评绑定与源码快照逐件一致", bind_ok)
    meta_sha = sb["protected_sha256"][[k for k in sb["protected_sha256"] if k.endswith("meta.json")][0]]
    check("绑定 meta/权重 SHA 与锁定候选一致",
          meta_sha == listing["缓存-概率与统计/selected-scorer-meta.json"]["sha256"]
          and "3b5403da2366bf8f4144ac95ebe6793d13cc82767cf55023cfe0ba4a21167498" == tsv["critical_sha256"][[k for k in tsv["critical_sha256"] if k.endswith("model.pt")][0]])

                                                           
    contract_ok = True
    rows_by_arm = {arm: len(labels[arm]) for arm in labels}
    for arm in ("legacy-original", "legacy-relabelled", "new-adjudicated"):
        cj = json.loads((ARCHIVE / f"协议-冻结/training-input-contract-{arm}.json").read_text(encoding="utf-8"))
        rc = cj["training_input_contract"]["row_counts"]
        ok = (rc["train"] == rows_by_arm[arm] and rc["dev"] == 405 and rc["prob"] == 116 and rc["policy"] == 120)
        ok = ok and [b["seed"] for b in cj["budgets"]] == [13, 20260919, 424242]
        ok = ok and all(b["updates"] == 400 and b["sample_instances"] == 6400 for b in cj["budgets"])
        ok = ok and cj["budgets"][0]["n_groups"] == (235 if arm == "new-adjudicated" else 16)
        if not ok:
            contract_ok = False
            print(f"       [CONTRACT-DIFF] {arm}")
    check("输入契约行数/预算与归档数据一致", contract_ok)

    print(f"[DONE] selftests {self_test_count - self_test_failed}/{self_test_count}; checks {check_count - check_failed}/{check_count} passed")
    return 0 if (self_test_failed == 0 and check_failed == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
