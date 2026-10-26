                       
"""锁定工件与分类闭合验收补充归档（2026-10-12，标准库-only；v2 只读/边界修订）。

v2 修订（review r1 修复）
----
- `--verify` 默认**只打印、零写入**（含临时目录；自测改为全内存实现，不再使用文件系统 scratch）；
  只有显式 `--out-root <DIR>` 时才把复核输出写入该目录（逐路径 asserts 守卫）。
- 构建模式（无 `--verify`）保持原位写归档（缺件复制 + manifest），并对每个写目标做
  ARCHIVE 内 containment 断言。

背景
----
`manifest-复现档案-2026-10-12.json`（87 件）已按白名单收拢冻结三臂训练集、
终评参考、冻结协议、研究源码快照与概率/统计缓存。**本补充脚本**把当时明确未复制的两类
补齐（不改写、不覆盖基线件）：

① **锁定工件**：被选 scorer 的 meta.json 与全量权重 model.pt（研究锁定候选，训练/终评
   的绑定件）；backbone 目录**全 15 文件**（tokenizer/config/词表/模型卡/两件基础权重；
   按 tower 指示不裁剪、不转换，保持 `features.model_fingerprint` 的 15 文件集合完整）。
② **分类闭合轮验收证据**：full 门最终 passed artifact（classification 轮）、其收口
   terminal-summary、公开投影修复前后与 close 三个现场 JSON。

另附**纯 stdlib 指纹双渲染复算**（不加载模型）：同字节集合在 Windows 反斜杠渲染下
= 锁定 meta 的 `1416f5fa…`、POSIX 渲染下 = M2 只读实测的 `980924bf…`，用于记录
"归档保存 ≠ portable 重载验收"。

纪律
----
- 只复制 WHITELIST；已存在同名件**不覆盖**：字节与 expected 一致→跳过；不一致→fail；
- 双向 SHA256：源文件、归档件都对齐白名单登记的 expected（任一处漂移即 fail）；
- 高精度秘密标记扫描：硬标记命中即 fail；软标记逐件复核登记（benign 也留档）；
- 源目录全部只读；**不运行**任何模型/加载/审计/终评，不安装/下载，不触 cos；
- meta.json **原样字节复制**（其绝对路径字段保持原值；迁移需另行映射，本档案不称已可移植）；
- 补充 manifest 与基线 manifest 交叉登记 SHA；复核输出默认只打印，
  落盘仅在显式 `--out-root`（或构建模式原位）时发生。

用法
----
python -B archive-lock-assets-2026-10-12.py                        # 构建（缺件复制）+ 全量校验 + 写 manifest
python -B archive-lock-assets-2026-10-12.py --verify               # 只读校验：只打印、零写入
python -B archive-lock-assets-2026-10-12.py --verify --out-root D  # 只读校验并把输出文本写到 D/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"<archive-root>")
ARCHIVE = Path(__file__).resolve().parents[1]                            
BASE_MANIFEST = ARCHIVE / "manifest-复现档案-2026-10-12.json"
MANIFEST_OUT = ARCHIVE / "manifest-补充-锁定工件与分类闭合-2026-10-12.json"
OUT_TXT = ARCHIVE / "复算" / "输出-archive-lock-assets-2026-10-12.txt"


def assert_inside(root: Path, path: Path) -> None:
    root_s = os.path.normcase(str(root.resolve()))
    path_s = os.path.normcase(str(path.resolve()))
    if path_s != root_s and not path_s.startswith(root_s + os.sep):
        raise SystemExit(f"[BOUNDARY] 拒绝写入授权根之外：{path}（root={root}）")

SCORER_REL = (
    "research/memory_relations/runs/v3-data-ablation-20261012/new-adjudicated/"
    "scorers/finetuned-joint-size3290-seed13"
)
GOVFIX_REL = "research/memory_relations/runs/v3-governance-fix-20261012"
BACKBONE_REL = "models/embeddings/paraphrase-multilingual-MiniLM-L12-v2"
LOCK_DIR = "锁定工件-2026-10-12"
ACC_DIR = "验收-门与修复-2026-10-12"

                                                                                           
WHITELIST = [
                                     
    ("锁定工件", f"{SCORER_REL}/meta.json",
     f"{LOCK_DIR}/scorer-finetuned-joint-size3290-seed13/meta.json",
     "被选 scorer meta（weights/backbone/tokenizer 指纹、训练预算与校准选点；原样字节，绝对路径保留）",
     "7ab44014d1b84b6e50be45e84ddb33685661b00bc3017544e33b39e4b544f948", 14661),
    ("锁定工件", f"{SCORER_REL}/model.pt",
     f"{LOCK_DIR}/scorer-finetuned-joint-size3290-seed13/model.pt",
     "被选 scorer 全量权重（≈448.9 MiB；训练/终评锁定候选，此前 87 件档案明确未复制）",
     "3b5403da2366bf8f4144ac95ebe6793d13cc82767cf55023cfe0ba4a21167498", 470691565),
                                                      
    ("锁定工件", f"{BACKBONE_REL}/config.json",
     f"{LOCK_DIR}/backbone-tokenizer-config/config.json",
     "骨干 AutoConfig 输入（加载侧配置）",
     "6300193cb75e01cf80c96decef7187dfb33094d97cc1490b7ead6ff134476e4e", 645),
    ("锁定工件", f"{BACKBONE_REL}/config_sentence_transformers.json",
     f"{LOCK_DIR}/backbone-tokenizer-config/config_sentence_transformers.json",
     "sentence-transformers 配置（记录 max_seq_length 等）",
     "b8c64b5cece00d8424b4896ea75b512b6008576088497609dfeb6bd63e6d36b8", 122),
    ("锁定工件", f"{BACKBONE_REL}/configuration.json",
     f"{LOCK_DIR}/backbone-tokenizer-config/configuration.json",
     "模型目录声明（sentence-transformers 侧元数据）",
     "007c616ed9187f4f3d6096ed2d55cece33c3f48866dbd7bc6afefcb9645a37c1", 77),
    ("锁定工件", f"{BACKBONE_REL}/modules.json",
     f"{LOCK_DIR}/backbone-tokenizer-config/modules.json",
     "ST 模块序列（Transformer + Pooling 装配）",
     "8f4b264b80206c830bebbdcae377e137925650a433b689343a63bdc9b3145460", 229),
    ("锁定工件", f"{BACKBONE_REL}/sentence_bert_config.json",
     f"{LOCK_DIR}/backbone-tokenizer-config/sentence_bert_config.json",
     "SBERT 运行参数（max_seq_length 等）",
     "70f4448f31320443fe3557cacea5abf2dcc4915dda8c80646bec9f3bb0aa5a1f", 53),
    ("锁定工件", f"{BACKBONE_REL}/special_tokens_map.json",
     f"{LOCK_DIR}/backbone-tokenizer-config/special_tokens_map.json",
     "tokenizer 特殊 token 映射",
     "378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d", 239),
    ("锁定工件", f"{BACKBONE_REL}/tokenizer.json",
     f"{LOCK_DIR}/backbone-tokenizer-config/tokenizer.json",
     "tokenizer 全量词表/归并规则（fast tokenizer 输入）",
     "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8", 9081518),
    ("锁定工件", f"{BACKBONE_REL}/tokenizer_config.json",
     f"{LOCK_DIR}/backbone-tokenizer-config/tokenizer_config.json",
     "tokenizer 配置（模型类别/特殊 token 参数）",
     "5036ea374ffedd706e3bef33e2e0d6953cb868ef8a490e76e32ba0faa37a6b9b", 526),
    ("锁定工件", f"{BACKBONE_REL}/sentencepiece.bpe.model",
     f"{LOCK_DIR}/backbone-tokenizer-config/sentencepiece.bpe.model",
     "SentencePiece BPE 模型（多语词表原语）",
     "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865", 5069051),
    ("锁定工件", f"{BACKBONE_REL}/unigram.json",
     f"{LOCK_DIR}/backbone-tokenizer-config/unigram.json",
     "unigram 词表（SentencePiece 统计）",
     "71b44701d7efd054205115acfa6ef126c5d2f84bd3affe0c59e48163674d19a6", 14763234),
    ("锁定工件", f"{BACKBONE_REL}/1_Pooling/config.json",
     f"{LOCK_DIR}/backbone-tokenizer-config/1_Pooling/config.json",
     "池化模块配置（CLS/mean 池化声明）",
     "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23", 190),
    ("锁定工件", f"{BACKBONE_REL}/.gitattributes",
     f"{LOCK_DIR}/backbone-tokenizer-config/.gitattributes",
     "模型目录的 git 属性（15 文件指纹集合成员，原样保留）",
     "fd3e5d862331ec272410eb6312742550c3149899ee61827169b78fd0d5889f2e", 1022),
    ("锁定工件", f"{BACKBONE_REL}/README.md",
     f"{LOCK_DIR}/backbone-tokenizer-config/README.md",
     "模型卡（15 文件指纹集合成员，原样保留）",
     "1e98ea05b0de579fcaad3d625b62ea55647142ed674d5f5ebf1440e4bbbb6f23", 3888),
    ("锁定工件", f"{BACKBONE_REL}/model.safetensors",
     f"{LOCK_DIR}/backbone-tokenizer-config/model.safetensors",
     "骨干 safetensors 权重（15 文件指纹集合成员；原样保留，不裁剪不转换）",
     "eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b", 470641600),
    ("锁定工件", f"{BACKBONE_REL}/pytorch_model.bin",
     f"{LOCK_DIR}/backbone-tokenizer-config/pytorch_model.bin",
     "骨干 PyTorch 权重（15 文件指纹集合成员；原样保留，不裁剪不转换）",
     "16cc9e54df6e083272378abec2d75dc34d7a48b5276db3ccc050d18de672ac59", 470693617),
                       
    ("分类闭合验收", "artifacts/quality-memory-relations-final-research-classification-20261012.json",
     f"{ACC_DIR}/classification-passed-artifact.json",
     "full 门最终 passed artifact（classification 轮：14/14、exit 0、1677.1645 s）",
     "61e479adaadd897410252e3cf4517c2827ed3d0cd0ebaa96a2773dedde41282d", 66608),
    ("分类闭合验收", f"{GOVFIX_REL}/terminal-summary-classification.json",
     f"{ACC_DIR}/classification-terminal-summary.json",
     "分类闭合收口（VALIDATED_CLASSIFICATION_CLOSED；6672/21、185/3、120/2、69/0/0、保护 1205）",
     "2e15d9e8afec04d011a7e7006398d3586932d49eacd955ef4a573f1bea382725", None),
    ("分类闭合验收", f"{GOVFIX_REL}/classification-before.json",
     f"{ACC_DIR}/classification-before.json",
     "公开投影修复前现场（两 profile 真实失败：returncode=1、passed=1、skipped=0）",
     "6f7843316e4ed24ffe78092c26ededb21c9c972f6b568df6e9e6671f20fca7cc", 1494),
    ("分类闭合验收", f"{GOVFIX_REL}/classification-after.json",
     f"{ACC_DIR}/classification-after.json",
     "公开投影修复后现场（两 profile 2 passed / 0 skipped、returncode=0）",
     "94e00fc4b21c4817c384987234bc63d5995aba507171765107c50b32f31cdeb0", 1230),
    ("分类闭合验收", f"{GOVFIX_REL}/close-classification.json",
     f"{ACC_DIR}/classification-close.json",
     "分类闭合收尾现场（protected_count=1205、changed_protected=[]、0 formal calls、fixture_cleanup=none）",
     "058bb2742b650610d29d2a2d4373445601673aba8186877d2e749e26f75f45e5", 215393),
]

SECRET_HARD_MARKERS = [
    b"BEGIN RSA PRIVATE KEY", b"BEGIN OPENSSH PRIVATE KEY", b"BEGIN EC PRIVATE KEY",
    b"kimi.pem", b"id_rsa",
]
SECRET_SOFT_MARKERS = [
    b"BEGIN PRIVATE KEY", b"AKIA", b"ghp_", b"password", b"passwd",
    b"api_key", b"apikey", b"secret_key", b"sk-zen",
]
                                     
SECRET_SCAN_REVIEW = {
    "backbone-tokenizer-config/tokenizer.json": {
        "markers": ["password"], "verdict": "benign",
        "evidence": "全文件仅 1 处命中：词表成员 `▁password`（多语 SentencePiece 词表中的普通英文词条），非凭证。"},
    "backbone-tokenizer-config/sentencepiece.bpe.model": {
        "markers": ["password"], "verdict": "benign",
        "evidence": "同词表条目 `password` 的 BPE 原语（二进制词表；1 处命中），非凭证。"},
    "backbone-tokenizer-config/unigram.json": {
        "markers": ["password"], "verdict": "benign",
        "evidence": "同词表条目 `▁password` 的 unigram 词频（1 处命中），非凭证。"},
}

NOT_COPIED = [
    {"item": "逐 arm 原始件与 18 MB provenance-row-ledger.jsonl 等",
     "reason": "基线 manifest 既有决定不变（聚合已归档、原始件留源位）。"},
    {"item": "repo 侧协作/日志/patch/QA runtime/大 log",
     "reason": "内部过程材料与机器运行产物留源位；验收以逐件 SHA 绑定"
               "（见 classification-terminal-summary.json 的 source_sha256/critical_sha256）。"},
    {"item": "backbone 15 文件集合的裁剪或转换",
     "reason": "按 tower 指示（2026-10-12）不裁剪、不转换：features.model_fingerprint 以目录内全部文件参与哈希"
               "（本目录恰为 15 件），裁剪任一成员都会使指纹集合失真。"},
]

BOUNDARIES = [
    "fresh=false：全部为已曝光合成材料/AI 代理标签（human_gold=false）；归档不改变'已曝光'属性，"
    "终测 872 请求已被模型看过一次，不得再称 fresh/未见。",
    "未做加载/重定位验证：本轮明令不运行任何模型/加载/审计/终评；因此本档案只做**文件级齐全**"
    "与 SHA 绑定，不声称可移植、不声称跨机器保真已验证。",
    "跨平台指纹阻塞（tower 转述 M2 只读实测，本档案低耗独立复算复核）：`features.model_fingerprint` "
    "以 `str(relative_path)` 参与哈希，同一字节集合在 Windows 反斜杠渲染下得 1416f5fa…（= 锁定 meta "
    "backbone_fingerprint），POSIX 渲染下得 980924bf…；且 meta.backbone_path/tokenizer_path 为 Windows "
    "绝对路径。**现有 loader 不经迁移不能直接跨机器/Linux 使用**；归档保存 ≠ portable 重载验收。",
    "meta.json 原样字节（绝对路径字段保留原值）；若将来迁移，需按新位置改 meta 或做目录映射，"
    "属未验证变更。",
    "分类闭合轮运行于 source_dirty=true 工作树（dirty_certification_rejected=true）：是验收证据，"
    "不是 clean 发布认证，也没有跑过任何'正式 public full'。",
]

MANIFEST_SCHEMA = "memory-relations-repro-archive-supplement-v1"


def model_fingerprint_pair(root: Path) -> tuple[str, str]:
    """纯 stdlib 复算 features.model_fingerprint 的两种路径渲染（不加载模型）。

    与 `research/memory_relations/features.py:model_fingerprint` 同构造：
    `sorted(root.rglob("*"))` 中 is_file 的项，按顺序 update(str(relative_path)) + 文件字节；
    第二种把相对路径改为 as_posix() 渲染。返回 (windows_style, posix_style) 前 32 位。
    """
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    win = hashlib.sha256()
    posix = hashlib.sha256()
    for path in paths:
        rel = path.relative_to(root)
        win.update(str(rel).encode("utf-8"))
        posix.update(rel.as_posix().encode("utf-8"))
        with open(path, "rb") as fh:
            while True:
                block = fh.read(1 << 20)
                if not block:
                    break
                win.update(block)
                posix.update(block)
    return win.hexdigest()[:32], posix.hexdigest()[:32]



def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def scan_secrets(path: Path) -> tuple[list[str], list[str]]:
    hard, soft = [], []
    with open(path, "rb") as fh:
        data = fh.read()
    for marker in SECRET_HARD_MARKERS:
        if marker in data:
            hard.append(marker.decode("ascii", "replace"))
    for marker in SECRET_SOFT_MARKERS:
        if marker in data:
            soft.append(marker.decode("ascii", "replace"))
    return hard, soft


def guard_no_overwrite(dest: Path, expected_sha: str) -> str:
    """dest 已存在：同字节→skip；不同→抛错（不覆盖、不静默改写）。返回 'skip' 或 'copy'。"""
    if not dest.exists():
        return decide_no_overwrite(False, None, expected_sha)
    return decide_no_overwrite(True, sha256_file(dest), expected_sha)


def decide_no_overwrite(dest_exists: bool, dest_sha: str | None, expected_sha: str) -> str:
    """不覆盖决策（纯逻辑，可内存自测）：不存在→copy；同字节→skip；不同字节→拒绝。"""
    if not dest_exists:
        return "copy"
    if dest_sha == expected_sha:
        return "skip"
    raise RuntimeError("拒绝覆盖（已存在且字节不同）")


def run_selftests(lines: list[str]) -> None:
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

                                                
    good = b"archive-lock-asset-self-test"
    gsha = hashlib.sha256(good).hexdigest()
    raw = bytearray(good)
    raw[0] ^= 0x01
    t("篡改 1 字节必不匹配（negative）", hashlib.sha256(bytes(raw)).hexdigest() != gsha)

    raised = False
    try:
        decide_no_overwrite(True, hashlib.sha256(b"content-different-from-source").hexdigest(), gsha)
    except RuntimeError:
        raised = True
    t("已存在不同字节件必拒绝覆盖（negative）", raised)

    t("已存在同字节件允许跳过（positive）", decide_no_overwrite(True, gsha, gsha) == "skip"
      and decide_no_overwrite(False, None, gsha) == "copy")

    t("白名单 expected SHA/字节齐全且无重复归档路径",
      all(e[4] for e in WHITELIST) and len({e[2] for e in WHITELIST}) == len(WHITELIST))
    lines.append(f"[SELFTEST] {ok}/{total} passed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="只读校验（不复制、不写 manifest；默认零写入）")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="显式可写根：复核输出文本写入 <out-root>/输出-archive-lock-assets-2026-10-12.txt（守卫）")
    args = ap.parse_args()

    write_target: Path | None = None
    if args.out_root is not None:
        write_target = args.out_root.resolve() / OUT_TXT.name
        assert_inside(args.out_root.resolve(), write_target)
    elif not args.verify:
        write_target = OUT_TXT
        assert_inside(ARCHIVE, write_target)

    lines: list[str] = []

    def out(msg: str) -> None:
        print(msg)
        lines.append(msg)

    run_selftests(lines)

    out("== 1. 白名单件：源→归档 SHA/字节/秘密扫描 ==")
    entries = []
    for group, src_rel, arc_rel, purpose, expected, exp_bytes in WHITELIST:
        src = ROOT / src_rel
        dest = ARCHIVE / arc_rel
        out(f"[{group}] {arc_rel}")
        if not src.exists():
            out(f"  FAIL 源缺失：{src}")
            return 2
        src_sha = sha256_file(src)
        if src_sha != expected:
            out(f"  FAIL 源 SHA 漂移：期望 {expected[:16]}… 实际 {src_sha[:16]}…")
            return 2
        if exp_bytes is not None and src.stat().st_size != exp_bytes:
            out(f"  FAIL 源字节漂移：期望 {exp_bytes} 实际 {src.stat().st_size}")
            return 2
        hard, soft = scan_secrets(src)
        if hard:
            out(f"  FAIL 硬秘密标记命中：{hard}（不复制）")
            return 2
        action = "verify-only"
        if not args.verify:
            assert_inside(ARCHIVE, dest)
            action = guard_no_overwrite(dest, expected)
            if action == "copy":
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
        if not dest.exists():
            out(f"  FAIL 归档件缺失：{dest}（{action}）")
            return 2
        dest_sha = sha256_file(dest)
        if dest_sha != expected or dest_sha != src_sha:
            out(f"  FAIL 双向 SHA 不一致：archive {dest_sha[:16]}…")
            return 2
        out(f"  OK  {action}  sha {dest_sha[:16]}…  {dest.stat().st_size} B  软标记={soft or '无'}")
        entries.append({
            "group": group, "archive_path": arc_rel, "source_repo_relative": src_rel,
            "sha256": dest_sha, "bytes": dest.stat().st_size, "purpose": purpose,
            "fresh": False,
            "secret_scan": {"hard": [], "soft_markers": soft,
                            "review": SECRET_SCAN_REVIEW.get(arc_rel)},
        })

    out("")
    out("== 2. backbone 15 文件指纹双渲染复算（纯 stdlib；不加载模型）==")
    win_fp, posix_fp = model_fingerprint_pair(ROOT / BACKBONE_REL)
    if win_fp != "1416f5fa5282f9b69713c63450fccdef":
        out(f"  FAIL windows-style 指纹 {win_fp} != 锁定 meta.backbone_fingerprint 1416f5fa…")
        return 2
    posix_match = posix_fp == "980924bf793294d0962083aa4af04cc9"
    out(f"  windows-style（反斜杠渲染）= {win_fp}  == meta.backbone_fingerprint  OK")
    out(f"  posix-style（正斜杠渲染）  = {posix_fp}  M2 报值 980924bf…  -> {'一致' if posix_match else '不一致'}")
    fingerprint_check = {
        "expected_meta_fingerprint": "1416f5fa5282f9b69713c63450fccdef",
        "windows_style": win_fp,
        "posix_style": posix_fp,
        "m2_readonly_reported_posix": "980924bf793294d0962083aa4af04cc9",
        "posix_matches_m2": posix_match,
        "note": "同一字节集合、仅相对路径渲染不同；现有 loader（features.model_fingerprint 以 str 渲染）"
                "不经迁移不能跨机器/Linux 使用；本复算不加载模型。",
    }

    if not args.verify:
        base_sha = sha256_file(BASE_MANIFEST) if BASE_MANIFEST.exists() else None
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "archive_root_relative": "复现资料-2026-10-12（补充：锁定工件与分类闭合验收）",
            "purpose": "补齐 87 件基线档案明确未复制的锁定 meta/weights、backbone tokenizer/config"
                       "与分类闭合轮（classification-closed）验收证据；冻结三臂/终评参考等合成冻结输入不重复复制。",
            "base_manifest": {
                "path": "manifest-复现档案-2026-10-12.json",
                "sha256": base_sha,
                "note": "基线 87 件已含数据-冻结三臂/数据-终评参考/协议-冻结/源码快照/缓存/既有验收件；本补充不改写基线。",
            },
            "root_git_head_context": "d64d296e57bc9c650a461805a991bd3b3a6bb499",
            "files": entries,
            "fingerprint_check": fingerprint_check,
            "secret_scan_review": {
                "note": "硬标记 0 命中；软标记命中 3 件（verdict=benign，逐件证据见 files[].secret_scan.review）。",
                "reviewed": SECRET_SCAN_REVIEW,
            },
            "not_copied": NOT_COPIED,
            "boundaries": BOUNDARIES,
            "verification": {
                "script": "复算/archive-lock-assets-2026-10-12.py",
                "output": "复算/输出-archive-lock-assets-2026-10-12.txt",
                "checks": "selftests 5/5 + 逐件源/归档双向 SHA + 白名单 expected SHA/字节 + 硬秘密扫描；"
                          "verify 模式重跑同脚本 --verify。",
            },
        }
        assert_inside(ARCHIVE, MANIFEST_OUT)
        MANIFEST_OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
                                encoding="utf-8", newline="\n")
        out(f"\n[WRITE] {MANIFEST_OUT.name}（{len(entries)} 件）")

    out(f"\n[DONE] files={len(entries)} 全部双向 SHA 一致；verify-only={args.verify}")
    if write_target is not None:
        write_target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        print(f"written: {write_target}")
    else:
        print("[MODE] verify-only 且未给 --out-root：只打印，零写入（含临时目录）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
