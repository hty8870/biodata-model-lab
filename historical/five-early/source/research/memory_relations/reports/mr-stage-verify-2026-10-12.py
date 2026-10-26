                       
"""MR 报告 staging：白名单归档逐件复核 + stage 限定正反复算（2026-10-12，标准库-only；v2 安全/持久化修订）。

用途（独立复核，只读输入、只写自有 out-root）
----
① 对 `复现资料-2026-10-12/` 全树（基线 87 件 + 补充 22 件 + 档案工具/清单/README）与
   `审计镜像-终评与CPU探针-2026-10-12/`（13 镜像件 + 镜像清单）逐件：
   - SHA256 与字节；与基线/补充 manifest 登记对账（fail-closed）；
   - 有 `source_repo_relative` 的重算 repo 源 SHA 做保真比对（漂移如实记录，不冒充一致）；
   - 硬秘密标记（fail-closed）/软标记（与既有复核登记对账）/用户内容模式（邮箱、手机号、已知真名）；
   - reparse 链接检查（junction/symlink 不跟随；白名单树不允许出现链接）。
② 分类闭合轮 7 个 source 哈希独立重算 + backbone 指纹双渲染复算（Windows/POSIX，纯 stdlib）。
③ stage 限定的正反复算：以真实档案为**只读**输入、本 out-root 为唯一**写入**根，运行
   `final-report-recheck-2026-10-12.py`：
   - positive：全检查通过（rc=0），输出与外部当前件对账；
   - negative-1：镜像件被篡改 → 必须哈希拒绝（rc!=0）；
   - negative-2：镜像件缺失 → 必须 fail（rc!=0）；
   - 另测 recheck 写入边界守卫（越界写入必须被 SystemExit 拒绝）与隔离 `--sync` 禁令。
   说明：negative 只覆盖镜像哈希/缺件两道闸与内置值守卫自测（before 冒充/6672==6670 等）；
   repo 侧源只读，不做"源被篡改"的负例。

v2 修订（review r1 修复）
----
- 自测目录改为 **out-root 内 tempfile 独占随机目录**（不读 TEMP/TMP；写入与清理前后均核
  containment 与归属；只删本次新建目录，预存目录一律不触碰）。
- recheck 脚本定位改为**持久默认**：`<study>/数据修复与新训练/脚本/final-report-recheck-2026-10-12.py`
  优先，worktree staging 仅作临时候选；可用 `--recheck-script` 显式指定（只读使用）。

用法
----
python -B mr-stage-verify-2026-10-12.py --out-root <可写根> \
    [--repo-root <agent 仓根>] [--study-root <研究/记忆关系轻量模型-2026-10-11>] \
    [--recheck-script <final-report-recheck 路径>] [--skip-recheck-runs]

边界
----
- 一切写入必须位于 `--out-root` 内（逐路径断言，含自测临时目录）；输入全部只读；
  不加载模型、不跑 GPU、不改任何归档件/正式训练/审计/终测产物。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

MIRROR_DIR_NAME = "审计镜像-终评与CPU探针-2026-10-12"
MIRROR_INDEX_NAME = "镜像清单-终评与CPU探针-2026-10-12.json"
REPRO_DIR_NAME = "复现资料-2026-10-12"
STAGED_RECHECK_REL = Path("staging/deliverable/final-report-recheck-2026-10-12.py")
                                                                            
DERIVED_EVIDENCE_RELS = [
    "复算/输出-mr-stage-verify-2026-10-12.txt",
    "复算/whitelist-verify-2026-10-12.json",
]

                               
BASELINE_MANIFEST_SHA = "f21a83b61627be39110c849fe49bc74a6362666c57aa655068264e6db870c025"
SUPPLEMENT_MANIFEST_SHA = "b68ea6b9d902f0afb3bad19055ee79cb39352c39a6ca7a7df159c3a84bb62daf"
STUDY_TARGET_SHA = {
    "研究报告-完整版-2026-10-12.md":
        "8c3679f78b44f90d07c395eafa4e8bc2734341fe10ba639f3a4d7fd58fcedbae",
    "脚本/final-report-recheck-2026-10-12.py":
        "3f69818a214bbc8d52f6f84774f8d35a18b64685fadeecdad0014b6f2770793c",
    "结果数据/final-report-recheck-2026-10-12.txt":
        "29b8f11e154152302958edb44b67b9d1f84a84309d8a3c889d411da00ebb667c",
    "复现资料-2026-10-12/README-复现档案-2026-10-12.md":
        "3a032b6178db481fec50e6e263f6230133eb78a4ab0f74398c25be2f8f089814",
    "脚本/说明-脚本登记与边界.md":
        "c81c181bec3e1dc88874a63426e1110865abfe1d4d22a955ffee1a35ae15d21a",
    "结果数据/说明-结果登记与边界.md":
        "51dae86776a2034721783e3c32b57c9f0695037c243c9de5826122ddeafc0f34",
    MIRROR_DIR_NAME + "/" + MIRROR_INDEX_NAME:
        "fa217c67bbb22d41af4889e27195a0b92f0b2728f93b84f39e70990a514b9c4a",
    "复现资料-2026-10-12/复算/archive-lock-assets-2026-10-12.py":
        "18f6b0ae1a74db34170fdd2ce0cbea88922e277e8efe58c2084b444a77591d26",
    "复现资料-2026-10-12/复算/输出-archive-lock-assets-2026-10-12.txt":
        "a3867095ee632a4ff0ac17e412be20f15bc37be60897a613a2d8b7adbb61e75c",
}
CLASSIFICATION_SOURCE_SHA = {
    ".deliveryignore": "d7b3097e3eb69d448668da37e749eecbdd60ac0fbb332ecef4ee24cdf97535d5",
    "MODULES.md": "f16806d5c0dfdc7e7eddb5223ca1f01956e60eaccf78395fb0b5aac38020bdd1",
    "packaging/public-mirror/files.skeleton.txt":
        "ca35ae6c4dc98910f315d2609e5a7c1caebe2d4a5ef4d236491b301ef7182c70",
    "packaging/public-mirror/files.txt":
        "fbeb5b24d48c90a6447e9dedb95f9dab6f51a7e2b7e7dfdae1e4bb3eed452c05",
    "scripts/validate-agent-governance.ps1":
        "88280430649e05d64e9548ecb9cdf4f5696c6ea2b0dbdba4ff0340a88a17f08b",
    "tests/test_agent_governance.py":
        "2f27d220a8a221773ffe10e31a6e7d1f06a50501ece026083a04b5e8db2fa95b",
    "tests/test_public_mirror_contract.py":
        "d695dbc183611b94c64c33d06f12854c064567bdd5bdbc6bd665c6e6b2ec0e91",
}
BACKBONE_META_FINGERPRINT_WIN = "1416f5fa5282f9b69713c63450fccdef"
BACKBONE_META_FINGERPRINT_POSIX = "980924bf793294d0962083aa4af04cc9"
BACKBONE_FILES = [
    "config.json", "config_sentence_transformers.json", "configuration.json", "modules.json",
    "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    "sentencepiece.bpe.model", "unigram.json", "1_Pooling/config.json", ".gitattributes", "README.md",
    "model.safetensors", "pytorch_model.bin",
]

HARD_MARKERS = [b"BEGIN RSA PRIVATE KEY", b"BEGIN OPENSSH PRIVATE KEY", b"BEGIN EC PRIVATE KEY",
                b"kimi.pem", b"id_rsa"]
SOFT_MARKERS = [b"BEGIN PRIVATE KEY", b"AKIA", b"ghp_", b"password", b"passwd",
                b"api_key", b"apikey", b"secret_key", b"sk-zen"]
USER_PATTERNS = [
    ("email", re.compile(rb"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("cn_mobile", re.compile(rb"(?<![\w.])1[3-9]\d{9}(?![\w.])")),
    ("real_name", re.compile("BioData contributors".encode("utf-8"))),
    ("gh_id", re.compile(b"hty8870")),
    ("cos", re.compile(b"cos_iterate")),
]
                                         
USER_REVIEW = {
    ("复现资料-2026-10-12/源码快照-research-memory-relations/test_teachers.py", "email"):
        {"verdict": "benign", "evidence": "opencode provider base_url 占位 `https://user:pw@opencode.ai/...`（非真实凭据）"},
    ("复现资料-2026-10-12/验收-门与修复-2026-10-12/research-tests-final.xml", "email"):
        {"verdict": "benign", "evidence": "pytest 用例 ID 合成占位 `name:synthetic-url-password@service.example.invalid`"},
    ("复现资料-2026-10-12/验收-门与修复-2026-10-12/research-tests-initial.xml", "email"):
        {"verdict": "benign", "evidence": "同上（初轮 XML 同类合成用例 ID）"},
    ("复现资料-2026-10-12/验收-门与修复-2026-10-12/classification-close.json", "cos"):
        {"verdict": "benign", "evidence": "git_status 字段记录 `?? cos_iterate.py` 文件名（状态行，无文件内容）"},
    ("复现资料-2026-10-12/验收-门与修复-2026-10-12/readiness-closeout.json", "cos"):
        {"verdict": "benign", "evidence": "同上（git_status 为文件名，无内容）"},
    ("复现资料-2026-10-12/验收-门与修复-2026-10-12/readiness-summary-final.json", "cos"):
        {"verdict": "benign", "evidence": "同上（git_status_before/after 为文件名，无内容）"},
}
                                    
SOFT_REVIEW = {
    "复现资料-2026-10-12/README-复现档案-2026-10-12.md": "档案 README §4 复述软标记复核结论（词名，非凭证）",
    "复现资料-2026-10-12/manifest-复现档案-2026-10-12.json": "扫描器标记清单与复核元数据（marker 字面量，非凭证）",
    "复现资料-2026-10-12/manifest-补充-锁定工件与分类闭合-2026-10-12.json": "补充 manifest 的 secret_scan_review 元数据（marker 名，非凭证）",
    "复现资料-2026-10-12/复算/archive-lock-assets-2026-10-12.py": "补充归档脚本自身的秘密标记定义常量（marker 字面量）",
    "复现资料-2026-10-12/复算/build-archive-2026-10-12.py": "构建脚本自身的秘密标记定义常量（marker 字面量）",
    "复现资料-2026-10-12/复算/manifest-attempt1-含secretstop-2026-10-12.json": "首轮停件半成品清单记录 marker 名（无值）",
    "复现资料-2026-10-12/复算/输出-archive-lock-assets-2026-10-12.txt": "归档复核输出打印软标记名与判定（verbatim 词名）",
    "复现资料-2026-10-12/复算/输出-build-archive-2026-10-12-attempt1-secretstop.txt": "首轮 fail-closed 停件日志（记录 marker 名，无值）",
    "复现资料-2026-10-12/源码快照-research-memory-relations/schema.py": "LOCAL_PROVIDER 默认参数 api_key=None（字段名）",
    "复现资料-2026-10-12/源码快照-research-memory-relations/teachers.py": "urlsplit(base_url).password 属性安全检查（字段名）",
    "复现资料-2026-10-12/源码快照-research-memory-relations/test_adapter.py": "测试夹具合成键 api_key=...（synthetic-knowledge-key）",
    "复现资料-2026-10-12/源码快照-research-memory-relations/test_teachers.py": "离线夹具 KEY=sk-zen-test-SECRET-4f21（显式 test/SECRET 命名）",
    "复现资料-2026-10-12/源码快照-research-memory-relations/test_training.py": "测试夹具 api_key=None（字段名）",
    "复现资料-2026-10-12/源码快照-生产侧/memory_relations.py": "秘密检测正则词表与 cfg.api_key 类型检查（字段名）",
    "复现资料-2026-10-12/源码快照-评估管线/preflight_check.py": "cfg.api_key 字段读取",
    "复现资料-2026-10-12/源码快照-评估管线/test_formal_evaluation.py": "cfg.api_key 字段读取（批次切分测试）",
    "复现资料-2026-10-12/验收-门与修复-2026-10-12/research-tests-final.xml": "pytest 用例 ID 显式合成夹具（synthetic-pem/AAAA/EEEE/ghp_-AAAA/synthetic-json-password）",
    "复现资料-2026-10-12/验收-门与修复-2026-10-12/research-tests-initial.xml": "同上（初轮 XML）",
    "复现资料-2026-10-12/锁定工件-2026-10-12/backbone-tokenizer-config/tokenizer.json": "多语词表成员 `▁password`（普通英文词条）",
    "复现资料-2026-10-12/锁定工件-2026-10-12/backbone-tokenizer-config/sentencepiece.bpe.model": "同词表条目的 BPE 原语（二进制词表）",
    "复现资料-2026-10-12/锁定工件-2026-10-12/backbone-tokenizer-config/unigram.json": "同词表条目的 unigram 词频",
}


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def scan_stream(path: Path, chunk: int = 1 << 22) -> dict:
    """单遍流式扫描：秘密标记（区分真实 PEM 与字面量）+ 用户内容模式（含上下文分类）。

    - hard_real：`-----BEGIN ... KEY-----` 形式的真实 PEM 标记（fail）；
    - hard_literal：`b"..."`/`"..."` 形式的标记字面量（扫描器/清单定义，复核登记）；
    - soft：软标记计数；user：模式命中计数与首个 masked 样本。
    """
    hard_real: dict[str, int] = {}
    hard_literal: dict[str, int] = {}
    soft: dict[str, int] = {}
    user: dict[str, dict] = {}
    overlap = 96
    tail = b""
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            buf = tail + block
            for mk in HARD_MARKERS:
                name = mk.decode("ascii", "replace")
                start = 0
                while True:
                    i = buf.find(mk, start)
                    if i < 0:
                        break
                    if buf[max(0, i - 5):i] == b"-----":
                        hard_real[name] = hard_real.get(name, 0) + 1
                    else:
                        hard_literal[name] = hard_literal.get(name, 0) + 1
                    start = i + len(mk)
            for mk in SOFT_MARKERS:
                name = mk.decode("ascii", "replace")
                c = buf.count(mk)
                if c:
                    soft[name] = soft.get(name, 0) + c
            for name, rx in USER_PATTERNS:
                for m in rx.finditer(buf):
                    rec = user.setdefault(name, {"count": 0, "sample_masked": None})
                    rec["count"] += 1
                    if rec["sample_masked"] is None:
                        s = m.group(0)
                        rec["sample_masked"] = (s[:2] + b"**" + s[-2:]).decode("utf-8", "replace")
            tail = buf[-overlap:]
    return {"hard_real": hard_real, "hard_literal": hard_literal, "soft": soft, "user": user}


def is_reparse(path: Path) -> bool:
    try:
        st = os.lstat(path)
    except OSError:
        return False
    attr = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(st, "st_file_attributes", 0) & attr)


def walk_nofollow(root: Path) -> tuple[list[Path], list[str]]:
    """显式 nofollow 遍历（不跟随 junction/symlink）；返回 (文件列表, 链接列表)。"""
    files: list[Path] = []
    links: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        kept = []
        for d in dirnames:
            p = Path(dirpath) / d
            if is_reparse(p):
                links.append(str(p))
            else:
                kept.append(d)
        dirnames[:] = kept
        for f in filenames:
            p = Path(dirpath) / f
            if is_reparse(p):
                links.append(str(p))
            else:
                files.append(p)
    return sorted(files), links


def is_relative_under(root: Path, path: Path) -> bool:
    root_s = os.path.normcase(str(root.resolve()))
    path_s = os.path.normcase(str(path.resolve()))
    return path_s == root_s or path_s.startswith(root_s + os.sep)


def assert_inside(root: Path, path: Path) -> None:
    if not is_relative_under(root, path):
        raise SystemExit(f"[BOUNDARY] 拒绝写入 out-root 之外：{path}（out-root={root}）")


def guard_no_overwrite(dest: Path, expected_sha: str) -> str:
    if not dest.exists():
        return "copy"
    if sha256_file(dest) == expected_sha:
        return "skip"
    raise RuntimeError(f"拒绝覆盖（已存在且字节不同）：{dest}")


def model_fingerprint_pair(root: Path) -> tuple[str, str, int]:
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    win = hashlib.sha256()
    posix = hashlib.sha256()
    for path in paths:
        rel = path.relative_to(root)
        win.update(str(rel).encode("utf-8"))
        posix.update(rel.as_posix().encode("utf-8"))
        with open(path, "rb") as fh:
            while True:
                block = fh.read(1 << 22)
                if not block:
                    break
                win.update(block)
                posix.update(block)
    return win.hexdigest()[:32], posix.hexdigest()[:32], len(paths)


def run_selftests(lines: list, out_root: Path) -> None:
    ok = 0
    total = 0

    def t(name, cond):
        nonlocal ok, total
        total += 1
        if cond:
            ok += 1
            lines.append(f"[SELFTEST-OK] {name}")
        else:
            lines.append(f"[SELFTEST-FAIL] {name}")
            raise AssertionError(name)

    t("hash 已知向量 sha256('abc')", hashlib.sha256(b"abc").hexdigest()
      == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    outside = out_root.parent / "escape-selftest.txt"
    escaped = False
    try:
        assert_inside(out_root, outside)
    except SystemExit:
        escaped = True
    t("写入边界守卫拒绝 out-root 外路径（negative）", escaped)

                                               
                                                      
    tmp_dir = Path(tempfile.mkdtemp(prefix="mr-stage-verify-selftest-", dir=str(out_root)))
    try:
        assert_inside(out_root, tmp_dir)
        t("自测目录位于 out-root 内", out_root.resolve() == tmp_dir.parent.resolve())
        t("自测目录为本脚本独占命名", tmp_dir.name.startswith("mr-stage-verify-selftest-") and not is_reparse(tmp_dir))

        def w(name: str, data: bytes) -> Path:
            p = tmp_dir / name
            assert_inside(tmp_dir, p)
            p.write_bytes(data)
            return p

        good = w("good.bin", b"stage-verify-selftest")
        g = sha256_file(good)
        b = bytearray(good.read_bytes())
        b[0] ^= 0x01
        tampered = w("t.bin", bytes(b))
        t("篡改 1 字节必不匹配（negative）", sha256_file(tampered) != g)
        dest = w("dest.bin", b"different")
        raised = False
        try:
            guard_no_overwrite(dest, g)
        except RuntimeError:
            raised = True
        t("已存在不同字节件必拒绝覆盖（negative）", raised)
        same = w("same.bin", good.read_bytes())
        t("已存在同字节件允许跳过（positive）", guard_no_overwrite(same, g) == "skip")
        t("自测写入全部位于自测目录内", all(is_relative_under(tmp_dir, tmp_dir / n)
                                            for n in ("good.bin", "t.bin", "dest.bin", "same.bin")))
    finally:
                                               
        if tmp_dir.exists():
            assert_inside(out_root, tmp_dir)
            if not (tmp_dir.name.startswith("mr-stage-verify-selftest-")
                    and tmp_dir.parent.resolve() == out_root.resolve() and not is_reparse(tmp_dir)):
                raise SystemExit(f"[BOUNDARY] 拒绝清理非本脚本独占目录：{tmp_dir}")
            shutil.rmtree(tmp_dir)
        t("自测目录清理后不存在（positive）", not tmp_dir.exists())
    lines.append(f"[SELFTEST] {ok}/{total} passed")


def main() -> int:
    ap = argparse.ArgumentParser(description="mr report staging whitelist verify + stage-limited recheck runs")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--study-root", type=Path, default=None)
    ap.add_argument("--recheck-script", type=Path, default=None,
                    help="被复核的 final-report-recheck 脚本（只读使用）；默认先找 <study>/数据修复与新训练/脚本/"
                         "final-report-recheck-2026-10-12.py（持久最终位置），再回退 worktree staging 候选")
    ap.add_argument("--skip-recheck-runs", action="store_true")
    args = ap.parse_args()

    repo_root = (args.repo_root or HERE.parents[2]).resolve()
    study_root = (args.study_root or (repo_root.parent / "研究" / "记忆关系轻量模型-2026-10-11")).resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    stage_dir = study_root / "数据修复与新训练"
    repro = stage_dir / REPRO_DIR_NAME
    mirror_dir = stage_dir / MIRROR_DIR_NAME

    if args.recheck_script is not None:
        recheck_script = args.recheck_script.resolve()
        recheck_origin = "explicit --recheck-script"
    else:
        recheck_script, recheck_origin = next(
            ((c, o) for c, o in [
                (stage_dir / "脚本" / "final-report-recheck-2026-10-12.py", "study-root 脚本/（持久最终位置）"),
                (HERE / STAGED_RECHECK_REL, "worktree staging 候选（合并/teardown 后不存在）"),
            ] if c.exists()), (None, None))

    lines: list[str] = []

    def out(s: str) -> None:
        print(s)
        lines.append(s)

    out("== 0. 自测 ==")
    run_selftests(lines, out_root)
    out("")

    out("== 1. 输入定位 ==")
    for p, label in ((repo_root, "repo-root"), (stage_dir, "stage(数据修复与新训练)"),
                     (repro, REPRO_DIR_NAME), (mirror_dir, MIRROR_DIR_NAME)):
        expect_exists = p.exists()
        out(f"  {'OK ' if expect_exists else 'FAIL'} {label}: {p}")
        if not expect_exists:
            return 2
    if recheck_script is None:
        out("  FAIL recheck-script：未找到（用 --recheck-script 显式指定，或提供 <study>/数据修复与新训练/"
            "脚本/final-report-recheck-2026-10-12.py）")
        return 2
    out(f"  OK  recheck-script（{recheck_origin}）: {recheck_script}")

    out("")
    out("== 2. 白名单归档逐件复核（SHA/字节/秘密/用户内容/链接）==")
    result: dict = {"schema": "mr-stage-whitelist-verify-v1",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "generated_by": "mr-stage-verify-2026-10-12.py",
                    "out_root": str(out_root), "repo_root": str(repo_root), "study_root": str(study_root)}

    baseline = json.loads((repro / "manifest-复现档案-2026-10-12.json").read_text(encoding="utf-8"))
    supplement = json.loads((repro / "manifest-补充-锁定工件与分类闭合-2026-10-12.json").read_text(encoding="utf-8"))
    base_by_path = {f["archive_path"]: f for f in baseline["files"]}
    supp_by_path = {f["archive_path"]: f for f in supplement["files"]}

    entries: list[dict] = []
    problems: list[str] = []

    for tree_root, tree_label in ((repro, REPRO_DIR_NAME), (mirror_dir, MIRROR_DIR_NAME)):
        files, links = walk_nofollow(tree_root)
        if links:
            problems.append(f"链接出现于 {tree_label}: {links}")
        kept = []
        for p in files:
            rel = p.relative_to(tree_root).as_posix()
            if tree_label == REPRO_DIR_NAME and rel in DERIVED_EVIDENCE_RELS:
                continue
            if tree_label == MIRROR_DIR_NAME and rel == MIRROR_INDEX_NAME:
                continue
            kept.append(p)
        note = ""
        if tree_label == REPRO_DIR_NAME:
            note = f"；派生复核证据按声明排除（{len(DERIVED_EVIDENCE_RELS)} 条路径，工具自身输出不计 SHA 绑定）"
        if tree_label == MIRROR_DIR_NAME:
            note = "；镜像清单 1 件按声明排除（recheck 工具输出，generated_at 易变，不计 SHA 绑定）"
        out(f"-- {tree_label}: {len(kept)} 个受复核文件，reparse 链接 {len(links)} 个{note}")
        for p in kept:
            rel = p.relative_to(tree_root).as_posix()
            key = f"{tree_label}/{rel}"
            sha = sha256_file(p)
            nbytes = p.stat().st_size
            scan = scan_stream(p)
            rec = {"location": key, "bytes": nbytes, "sha256": sha,
                   "hard_real": scan["hard_real"], "hard_literal": scan["hard_literal"],
                   "soft": scan["soft"], "user": scan["user"],
                   "reparse": is_reparse(p), "manifest": None, "manifest_match": None,
                   "source_repo_relative": None, "source_sha256": None, "source_match": None}
            if tree_label == REPRO_DIR_NAME and rel in base_by_path:
                m = base_by_path[rel]
                rec["manifest"] = "baseline"
                rec["manifest_match"] = (sha == m["sha256"] and nbytes == m["bytes"])
                rec["source_repo_relative"] = m.get("source_repo_relative")
            elif tree_label == REPRO_DIR_NAME and rel in supp_by_path:
                m = supp_by_path[rel]
                rec["manifest"] = "supplement"
                rec["manifest_match"] = (sha == m["sha256"] and (m.get("bytes") in (None, nbytes)))
                rec["source_repo_relative"] = m.get("source_repo_relative")
            if rec["manifest_match"] is False:
                problems.append(f"manifest 对账失败: {key}")
            if rec["hard_real"]:
                problems.append(f"真实 PEM 秘密标记: {key} {rec['hard_real']}")
            if rec["hard_literal"] and key not in SOFT_REVIEW:
                problems.append(f"未复核的标记字面量: {key} {rec['hard_literal']}")
            if rec["soft"] and key not in SOFT_REVIEW:
                problems.append(f"未复核软标记: {key} {rec['soft']}")
            for pat, hit in rec["user"].items():
                if (key, pat) not in USER_REVIEW:
                    problems.append(f"未复核用户内容模式: {key} {pat} {hit}")
                else:
                    hit["review"] = USER_REVIEW[(key, pat)]
            if rec["reparse"]:
                problems.append(f"reparse 链接: {key}")
            entries.append(rec)
                                                          
        if tree_label == REPRO_DIR_NAME:
            disk_set = {r["location"].split("/", 1)[1] for r in entries if r["location"].startswith(REPRO_DIR_NAME)}
            missing = sorted((set(base_by_path) | set(supp_by_path)) - disk_set)
            if missing:
                problems.append(f"manifest 登记但磁盘缺失: {missing}")
            out(f"   基线 manifest 覆盖 {len(base_by_path)} 件；补充 {len(supp_by_path)} 件；已逐件复核")

                                                    
    out("")
    out("-- 源保真复核（archive vs repo 源，逐件重算）")
    src_mismatch: list[str] = []
    for rec in entries:
        src_rel = rec["source_repo_relative"]
        if not src_rel:
            continue
        src = repo_root / src_rel.replace("\\", "/")
        if not src.exists():
            rec["source_match"] = False
            src_mismatch.append(f"{rec['location']}: source missing {src_rel}")
            continue
        s = sha256_file(src)
        rec["source_sha256"] = s
        rec["source_match"] = (s == rec["sha256"])
        if not rec["source_match"]:
            src_mismatch.append(f"{rec['location']}: archive {rec['sha256'][:16]} != source {s[:16]}")
    out(f"  有源可比的件数: {sum(1 for r in entries if r['source_repo_relative'])}；"
        f"不一致 {len(src_mismatch)} 件")
    for m in src_mismatch[:20]:
        out(f"    [DRIFT] {m}")

                              
    out("")
    out("-- 外部当前件 vs 记录值（集成冲突闸参考）")
    gate: list[dict] = []
    for rel, expected in STUDY_TARGET_SHA.items():
        p = stage_dir / rel
        if not p.exists():
            gate.append({"target": rel, "status": "missing", "expected": expected})
            out(f"    [MISS] {rel}")
            continue
        actual = sha256_file(p)
        gate.append({"target": rel, "expected_recorded": expected, "actual": actual,
                     "match": actual == expected})
        flag = "OK " if actual == expected else "DIFF"
        out(f"    [{flag}] {rel}  {actual[:16]}…")
    result["expected_gate"] = gate

    out("")
    out("== 3. 分类闭合轮 source 哈希独立重算（7 件） ==")
    cls_src = []
    for rel, expected in CLASSIFICATION_SOURCE_SHA.items():
        p = repo_root / rel
        actual = sha256_file(p) if p.exists() else None
        match = actual == expected
        cls_src.append({"path": rel, "expected": expected, "actual": actual, "match": match})
        out(f"  [{'OK ' if match else 'FAIL'}] {rel}  {(actual or 'missing')[:16]}…")
        if not match:
            problems.append(f"分类源哈希不符: {rel}")
    result["classification_source_recompute"] = cls_src

    out("")
    out("== 4. backbone 指纹双渲染复算（纯 stdlib；不加载模型） ==")
    root_bb = repo_root / "models/embeddings/paraphrase-multilingual-MiniLM-L12-v2"
    arch_bb = repro / "锁定工件-2026-10-12/backbone-tokenizer-config"
    root_files = sorted(p.relative_to(root_bb).as_posix() for p in root_bb.rglob("*") if p.is_file())
    arch_files = sorted(p.relative_to(arch_bb).as_posix() for p in arch_bb.rglob("*") if p.is_file())
    same_set = root_files == arch_files == sorted(BACKBONE_FILES)
    rw, rp, rn = model_fingerprint_pair(root_bb)
    aw, ap_, an = model_fingerprint_pair(arch_bb)
    out(f"  文件集合一致（15 件）: {same_set}；repo 15={rn}、archive 15={an}")
    out(f"  repo    windows-style={rw}  posix-style={rp}")
    out(f"  archive windows-style={aw}  posix-style={ap_}")
    meta_path = repro / "锁定工件-2026-10-12/scorer-finetuned-joint-size3290-seed13/meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mw = meta.get("backbone_fingerprint")
    bp = str(meta.get("backbone_path", ""))
    tp = str(meta.get("tokenizer_path", ""))
    out(f"  meta.backbone_fingerprint={mw}（期望 {BACKBONE_META_FINGERPRINT_WIN}）；"
        f"backbone_path 前缀={'Windows 绝对' if ':' in bp[:4] else '非Windows'}，tokenizer_path 同上={'Windows 绝对' if ':' in tp[:4] else '非Windows'}")
    fp_ok = (same_set and rw == aw == mw == BACKBONE_META_FINGERPRINT_WIN
             and rp == ap_ == BACKBONE_META_FINGERPRINT_POSIX)
    if not fp_ok:
        problems.append("backbone 指纹双渲染不一致")
    result["fingerprint"] = {
        "file_set_equal_15": same_set,
        "repo_windows": rw, "repo_posix": rp, "archive_windows": aw, "archive_posix": ap_,
        "meta_backbone_fingerprint": mw, "meta_backbone_path_windows_abs": ":" in bp[:4],
        "meta_tokenizer_path_windows_abs": ":" in tp[:4],
        "verified": fp_ok,
        "note": "同一字节集合仅相对路径渲染不同；未做加载/迁移验证（不可移植未验证）。",
    }

    out("")
    out("== 5. stage 限定正反复算（写入只落本 out-root） ==")
    runs: dict = {}
    if not args.skip_recheck_runs:
        py = sys.executable
        child_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        common = [py, "-B", str(recheck_script), "--repo-root", str(repo_root)]

        pos_out = out_root / "recheck" / "positive-out"
        cmd = common + ["--inputs-root", str(stage_dir), "--out-root", str(pos_out)]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=child_env)
        pos_txt = pos_out / "结果数据" / "final-report-recheck-2026-10-12.txt"
        pos_index = pos_out / MIRROR_DIR_NAME / MIRROR_INDEX_NAME
        ext_txt = stage_dir / "结果数据" / "final-report-recheck-2026-10-12.txt"
        txt_same = pos_txt.exists() and ext_txt.exists() and sha256_file(pos_txt) == sha256_file(ext_txt)
        txt_delta = []
        if pos_txt.exists() and ext_txt.exists() and not txt_same:
            a = pos_txt.read_text(encoding="utf-8").splitlines()
            b = ext_txt.read_text(encoding="utf-8").splitlines()
            import difflib
            txt_delta = [l for l in difflib.unified_diff(b, a, "external", "staged", lineterm="")
                         if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
        runs["positive"] = {
            "cmd": cmd, "returncode": r.returncode,
            "out_txt": str(pos_txt), "out_txt_sha256": sha256_file(pos_txt) if pos_txt.exists() else None,
            "out_index_sha256": sha256_file(pos_index) if pos_index.exists() else None,
            "external_txt_sha256": sha256_file(ext_txt) if ext_txt.exists() else None,
            "txt_identical_to_external": bool(txt_same), "txt_delta_lines": txt_delta,
            "stdout_tail": r.stdout.splitlines()[-6:] if r.stdout else [],
        }
        out(f"  positive rc={r.returncode}（期望 0）；out txt sha={runs['positive']['out_txt_sha256']}")
        out(f"    与外部当前 txt 逐字节一致: {txt_same}；差异行 {len(txt_delta)}")
        if r.returncode != 0:
            problems.append("stage positive run 未通过")
        if txt_delta:
            for l in txt_delta[:6]:
                out(f"    [TXT-DELTA] {l}")

        def build_neg_tree(name: str, tamper: bool) -> Path:
            tree = out_root / "recheck" / f"{name}-inputs"
            tdir = tree / MIRROR_DIR_NAME
            if tdir.exists():
                shutil.rmtree(tdir)
            tdir.mkdir(parents=True)
            for item in sorted(mirror_dir.iterdir()):
                if item.is_file() and item.name != MIRROR_INDEX_NAME:
                    shutil.copyfile(item, tdir / item.name)
            target = tdir / "governance-fix-terminal-summary-classification.json"
            if tamper:
                s = target.read_text(encoding="utf-8")
                assert s.count('"passed": 6672') == 1
                target.write_text(s.replace('"passed": 6672', '"passed": 6670', 1), encoding="utf-8", newline="\n")
            else:
                target.unlink()
            return tree

        neg1_tree = build_neg_tree("neg1", tamper=True)
        neg1_out = out_root / "recheck" / "neg1-out"
        r1 = subprocess.run(common + ["--inputs-root", str(neg1_tree), "--out-root", str(neg1_out)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", env=child_env)
        neg1_ok = r1.returncode != 0 and "hash mismatch" in (r1.stderr or "")
        runs["negative_1_tampered_mirror"] = {
            "cmd": common + ["--inputs-root", "<staging neg1-inputs>", "--out-root", "<staging neg1-out>"],
            "returncode": r1.returncode, "hash_refusal_observed": "hash mismatch" in (r1.stderr or ""),
            "stderr_tail": (r1.stderr or "").splitlines()[-1:],
        }
        out(f"  negative-1（篡改镜像）rc={r1.returncode}（期望!=0）；哈希拒绝={neg1_ok}")
        if not neg1_ok:
            problems.append("negative-1 未按预期拒绝")

        neg2_tree = build_neg_tree("neg2", tamper=False)
        neg2_out = out_root / "recheck" / "neg2-out"
        r2 = subprocess.run(common + ["--inputs-root", str(neg2_tree), "--out-root", str(neg2_out)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", env=child_env)
        neg2_ok = r2.returncode != 0 and "mirror missing" in (r2.stderr or "")
        runs["negative_2_missing_mirror"] = {
            "returncode": r2.returncode, "missing_refusal_observed": "mirror missing" in (r2.stderr or ""),
            "stderr_tail": (r2.stderr or "").splitlines()[-1:],
        }
        out(f"  negative-2（缺件）rc={r2.returncode}（期望!=0）；缺件拒绝={neg2_ok}")
        if not neg2_ok:
            problems.append("negative-2 未按预期拒绝")

                                                 
        import importlib.util
        spec = importlib.util.spec_from_file_location("staged_recheck", recheck_script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        guard_ok = False
        try:
            mod.assert_inside(out_root, out_root / "guard-ok.txt")
            try:
                mod.assert_inside(out_root, out_root.parent / "guard-escape.txt")
            except SystemExit:
                guard_ok = True
        except Exception:
            guard_ok = False
        runs["boundary_guard"] = {"escape_blocked": guard_ok}
        out(f"  边界守卫（越界写入被 SystemExit 拒绝）: {guard_ok}")
        if not guard_ok:
            problems.append("边界守卫未生效")
    else:
        out("  [SKIP] --skip-recheck-runs")
    result["stage_runs"] = runs

    out("")
    out("== 6. 空间 ==")
    total_files = len(entries)
    total_bytes = sum(r["bytes"] for r in entries)
    vault = shutil.disk_usage(out_root)
    out(f"  归档复核 {total_files} 件 / {total_bytes} B（≈{total_bytes/1024/1024/1024:.3f} GiB）")
    out(f"  磁盘 {out_root.drive}: 总 {vault.total} B / 已用 {vault.used} B / 剩余 {vault.free} B")
    result["space"] = {"files_checked": total_files, "bytes_checked": total_bytes,
                       "disk_total": vault.total, "disk_free": vault.free}

                                     
    known_soft = {r["location"] for r in entries if r["soft"]}
    hard_lit = {r["location"]: r["hard_literal"] for r in entries if r["hard_literal"]}
    user_reviewed = {r["location"]: r["user"] for r in entries if r["user"]}
    out("")
    out(f"  软标记命中件数: {len(known_soft)}（全部经复核登记，见 SOFT_REVIEW；未登记者已列 PROBLEMS）")
    out(f"  硬标记字面量命中（扫描器/清单定义，非 PEM）: {len(hard_lit)} 件")
    out(f"  用户内容模式命中（逐一上下文复核）: {len(user_reviewed)} 件")
    result["soft_hits"] = sorted(known_soft)
    result["hard_literal_hits"] = hard_lit
    result["user_hits_reviewed"] = user_reviewed
    result["problems"] = problems
    result["entries"] = entries

    out("")
    if problems:
        out(f"[PROBLEMS] {len(problems)} 项：")
        for p in problems:
            out(f"  - {p}")
    else:
        out("[DONE] 全部检查通过：逐件 SHA/字节/manifest 对账、无真实 PEM/无未复核秘密标记、"
            "用户内容模式命中均为合成/词表/状态行并已复核、无链接；正反复算与边界守卫符合预期。")
    result["status"] = "passed" if not problems else "problems"

    txt_path = out_root / "输出-mr-stage-verify-2026-10-12.txt"
    json_path = out_root / "whitelist-verify-2026-10-12.json"
    assert_inside(out_root, txt_path)
    assert_inside(out_root, json_path)
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"written: {txt_path}")
    print(f"written: {json_path}")
    return 0 if not problems else 3


if __name__ == "__main__":
    sys.exit(main())
