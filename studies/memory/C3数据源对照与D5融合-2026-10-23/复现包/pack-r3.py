"""r3 构建与验真：只改可照抄示例的输出目录保护/退出码传播 + 版本元数据。

科学成员（61 个）与 4 个 git 源码成员与 r1/r2 逐字节相同；不训练、不做新推理。
r3 说明文本从 r2 ZIP 的 REPRODUCE.txt 派生，只替换头部版本句与两段 PowerShell 示例，
并用逆替换证明没有改到其它任何一行。验证含正例 + 两个负例（已存在输出目录 / 缺索引）。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
import zlib
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = Path(r"<archive-root>")
COMMIT = "b5b911f0bef5642193071875b365163bddf98de7"

R1_ZIP = OUT / "记忆关系轻量模型-C3D5复现包-2026-10-23.zip"
R2_ZIP = OUT / "记忆关系轻量模型-C3D5复现包-2026-10-23-r2.zip"
R3_ZIP = OUT / "记忆关系轻量模型-C3D5复现包-2026-10-23-r3.zip"
R3_MANIFEST = OUT / "记忆关系轻量模型-C3D5复现包-2026-10-23-r3.manifest.sha256"
R3_RECEIPT = OUT / "verify-receipt-r3.json"
PACKER = OUT / "pack.py"

ALLOWED_CHANGES = {"REPRODUCE.txt", "PACKAGE.json", "MANIFEST.sha256"}

OLD_TITLE = "复现资料包（r2）"
NEW_TITLE = "复现资料包（r3）"
OLD_NOTE = "版本：r2（2026-10-23）。相对 r1 只修订本说明与 PACKAGE.json 的版本元数据；"
NEW_NOTE = "版本：r3（2026-10-23）。相对 r2 只改两段可照抄示例的输出目录保护与非零退出传播，以及 PACKAGE.json 版本元数据；"
OLD_RECEIPT_REF = "verify-receipt-r2.json（包外）"
NEW_RECEIPT_REF = "verify-receipt-r3.json（包外）"

TRAIN_BLOCK = """    $RepoRoot   = '<完整仓库绝对路径>'          # 例：D:\\work\\biodata-agent
    $BundleRoot = '<本包解压根>'               # 例：D:\\work\\c3d5-r3
    $Python     = Join-Path $RepoRoot '.venv/Scripts/python.exe'
    $ModelsOut  = '<新的绝对输出目录>/models'   # 必须是尚不存在的新目录
    $env:PYTHONPATH              = "$RepoRoot;$RepoRoot/src"
    $env:PYTHONDONTWRITEBYTECODE = '1'
    Set-Location -LiteralPath $RepoRoot

    # 目录已存在就直接 throw；示例不会删除任何已有目录或结果
    if (Test-Path -LiteralPath $ModelsOut) {
        throw "输出目录已存在：$ModelsOut。请指定未存在的新目录，保留原结果。"
    }

    $TrainArgs = @(
      '-B','-m','research.memory_relations.source_control','train-c3-pair',
      '--official-data', (Join-Path $BundleRoot 'data/official_train.jsonl'),
      '--fine-data',     (Join-Path $BundleRoot 'data/fine_train.jsonl'),
      '--prep-manifest', (Join-Path $BundleRoot 'data/manifest.json'),
      '--embedding-dir','models/embeddings/paraphrase-multilingual-MiniLM-L12-v2',
      '--out',$ModelsOut,'--device','cuda','--seed','13',
      '--batch-size','8','--grad-accum','2','--max-length','384','--budget-epochs','20'
    )
    & $Python @TrainArgs
    if ($LASTEXITCODE -ne 0) { throw "训练失败（python 退出码 $LASTEXITCODE），未覆盖任何已有结果。" }
    "train ok exit=$LASTEXITCODE"
"""

REPLAY_BLOCK = """    $RepoRoot   = '<完整仓库绝对路径>'          # 例：D:\\work\\biodata-agent
    $BundleRoot = '<本包解压根>'               # 例：D:\\work\\c3d5-r3
    $Python     = Join-Path $RepoRoot '.venv/Scripts/python.exe'
    $env:PYTHONPATH              = "$RepoRoot;$RepoRoot/src"
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:HF_HUB_OFFLINE          = '1'
    $env:TRANSFORMERS_OFFLINE    = '1'

    $Out = '<新的绝对输出目录>/d5-replay'       # 必须是尚不存在的新目录
    # 已存在就直接 throw；示例不会删除任何已有目录或结果
    if (Test-Path -LiteralPath $Out) {
        throw "输出目录已存在：$Out。请指定未存在的新目录，保留原结果。"
    }

    Set-Location -LiteralPath $BundleRoot        # 必须在解压根执行：索引内是相对路径
    & $Python -B -m research.memory_relations.source_control d5 `
        --index (Join-Path $BundleRoot 'd5/derived-portable-index.json') `
        --out   $Out `
        --bootstrap-samples 1000
    if ($LASTEXITCODE -ne 0) { throw "D5 重放失败（python 退出码 $LASTEXITCODE），未写入任何结果。" }
    "replay ok exit=$LASTEXITCODE"
"""

BUILD_INFO: dict = {}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def load_packer():
    spec = importlib.util.spec_from_file_location("pack_r1_builder", PACKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def r2_member(name: str) -> bytes:
    with zipfile.ZipFile(R2_ZIP) as archive:
        return archive.read(name)


def section_block(text: str, header: str, next_header: str) -> str:
    _, rest = text.split(header, 1)
    section, _ = rest.split(next_header, 1)
    lines = section.splitlines(keepends=True)
    start = next(index for index, line in enumerate(lines) if line.strip().startswith("$RepoRoot"))
    end = next(index for index, line in enumerate(lines) if line.strip().endswith('exit=$LASTEXITCODE"'))
    return "".join(lines[start:end + 1])


def build_reproduce_txt() -> str:
    base = r2_member("REPRODUCE.txt").decode("utf-8")
    old_train = section_block(base, "3. 重训", "4. 离线重放 D5")
    old_replay = section_block(base, "4. 离线重放 D5", "5. 查评分")

    text = base
    for old, new in [(OLD_TITLE, NEW_TITLE), (OLD_NOTE, NEW_NOTE),
                     (OLD_RECEIPT_REF, NEW_RECEIPT_REF),
                     (old_train, TRAIN_BLOCK), (old_replay, REPLAY_BLOCK)]:
        if text.count(old) != 1:
            raise RuntimeError(f"r2 说明中待替换片段不唯一: {old[:40]!r}")
        text = text.replace(old, new, 1)

    restored = text
    for old, new in [(OLD_TITLE, NEW_TITLE), (OLD_NOTE, NEW_NOTE),
                     (OLD_RECEIPT_REF, NEW_RECEIPT_REF),
                     (old_train, TRAIN_BLOCK), (old_replay, REPLAY_BLOCK)]:
        restored = restored.replace(new, old, 1)

    BUILD_INFO["reproduce"] = {
        "derived_from": "r2 ZIP 的 REPRODUCE.txt",
        "reversible_to_r2": restored == base,
        "no_remove_item": "Remove-Item" not in text,
        "has_output_guard": text.count("输出目录已存在") == 2,
        "has_exit_propagation": text.count("if ($LASTEXITCODE -ne 0) { throw") == 2,
        "replaced_blocks": {"train": old_train != TRAIN_BLOCK, "replay": old_replay != REPLAY_BLOCK},
    }
    if not all([BUILD_INFO["reproduce"]["reversible_to_r2"], BUILD_INFO["reproduce"]["no_remove_item"],
                BUILD_INFO["reproduce"]["has_output_guard"], BUILD_INFO["reproduce"]["has_exit_propagation"]]):
        raise RuntimeError(f"r3 说明构建自检失败: {BUILD_INFO['reproduce']}")
    return text


def build_package_json() -> dict:
    package = json.loads(r2_member("PACKAGE.json").decode("utf-8"))
    for key in ("changes_vs_r1", "unchanged_vs_r1"):
        package.pop(key, None)
    package["bundle_version"] = "r3"
    package["revision_note"] = "只修正可照抄示例的输出目录保护与非零退出传播；科学成员与 r1/r2 逐字节相同。"
    package["built_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    package["supersedes"] = {"version": "r2", "zip": R2_ZIP.name, "zip_sha256": sha256_file(R2_ZIP)}
    package["lineage"] = [
        {"version": "r1", "zip_sha256": sha256_file(R1_ZIP), "note": "首版"},
        {"version": "r2", "zip_sha256": sha256_file(R2_ZIP), "note": "运行环境/cwd/簇数/评测会话措辞修正"},
    ]
    package["changes_vs_r2"] = [
        "REPRODUCE §3/§4：删除示例中 if (Test-Path $Out) { Remove-Item -Recurse -Force } 的自动删除；改为目录已存在即 throw「请指定未存在的新目录，保留原结果」。",
        "REPRODUCE §3/§4：调用 Python 后检查 $LASTEXITCODE，非零即 throw 传播（脚本以非零退出），只有成功才输出 ok exit=0。",
        "REPRODUCE 头部与 §7：版本号 r2→r3、回执名改为 verify-receipt-r3.json。",
        "PACKAGE.json：r3 版本元数据、lineage、changes_vs_r2、output_directory_policy。",
        "MANIFEST.sha256：随 REPRODUCE.txt / PACKAGE.json 重算。",
    ]
    package["unchanged_vs_r2"] = [
        "source/（4 个 git 成员）、data/、d5/、inputs/、audit/、eval-official/、eval-fine/、models/、artifacts/、blind36/、protocol/ 全部成员逐字节相同",
        "REPRODUCE §1 运行环境（scikit-learn/SciPy、无权重/无新增推理/无 CUDA）、§2 簇数（official 289 / fine 60）、§4 cwd 要求、§6 一次锁定会话措辞",
    ]
    package["output_directory_policy"] = {
        "examples_auto_delete_existing": False,
        "existing_output_directory": "throw（请指定未存在的新目录，保留原结果）",
        "nonzero_python_exit": "throw 传播，PowerShell 以非零退出",
        "internal_verification_temp_dirs": "仅构建脚本自建、按 PID 命名的临时目录会自清理，不对外提供",
    }
    return package


def verify_crc() -> dict:
    rows = []
    with zipfile.ZipFile(R3_ZIP) as archive:
        for info in archive.infolist():
            data = archive.read(info)
            rows.append({"path": info.filename, "crc_ok": info.CRC == (zlib.crc32(data) & 0xFFFFFFFF)})
    bad = [row["path"] for row in rows if not row["crc_ok"]]
    return {"checked": len(rows), "mismatch": bad, "ok": not bad}


def verify_prev_comparison() -> dict:
    with zipfile.ZipFile(R2_ZIP) as prev, zipfile.ZipFile(R3_ZIP) as cur:
        prev_names = set(prev.namelist())
        cur_names = set(cur.namelist())
        identical, changed = [], []
        for name in sorted(prev_names & cur_names):
            (identical if prev.read(name) == cur.read(name) else changed).append(name)
        only_prev = sorted(prev_names - cur_names)
        only_cur = sorted(cur_names - prev_names)
    unexpected = [name for name in changed if name not in ALLOWED_CHANGES]
    git_sources = [
        "source/MODULES.md",
        "source/research/memory_relations/README.md",
        "source/research/memory_relations/source_control.py",
        "source/research/memory_relations/test_source_control.py",
    ]
    return {
        "compared_to": R2_ZIP.name,
        "compared_sha256": sha256_file(R2_ZIP),
        "identical_count": len(identical),
        "identical_members": identical,
        "changed_allowed": changed,
        "unexpected_changes": unexpected,
        "only_in_previous": only_prev,
        "only_in_current": only_cur,
        "git_sources_identical": all(name in identical for name in git_sources),
        "ok": (not only_prev and not only_cur and not unexpected
               and set(changed) <= ALLOWED_CHANGES and all(name in identical for name in git_sources)),
    }


def run_readme_block(workdir: Path, bundle: Path, out_path: Path, text: str) -> dict:
    block = section_block(text, "4. 离线重放 D5", "5. 查评分")
    block = "\n".join(line[4:] if line.startswith("    ") else line for line in block.splitlines())
    for placeholder, value in [("<完整仓库绝对路径>", str(ROOT)), ("<本包解压根>", str(bundle)),
                               ("<新的绝对输出目录>/d5-replay", str(out_path))]:
        if placeholder not in block:
            raise RuntimeError(f"REPRODUCE §4 缺少占位符: {placeholder}")
        block = block.replace(placeholder, value)
    script = workdir / "reproduce_section4.ps1"
    script.write_text(block, encoding="utf-8-sig")
    result = subprocess.run(["pwsh", "-NoProfile", "-File", str(script)],
                            cwd=str(workdir), capture_output=True, text=True, encoding="utf-8")
    return {"block": block, "script_sha256": sha256_file(script), "result": result}


def materialise(workdir: Path, with_index: bool = True) -> Path:
    bundle = workdir / "bundle"
    bundle.mkdir(parents=True)
    with zipfile.ZipFile(R3_ZIP) as archive:
        names = [name for name in archive.namelist() if name.startswith("inputs/d5-original/")]
        if with_index:
            names.append("d5/derived-portable-index.json")
        for name in names:
            target = bundle / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    return bundle


def compare_models(summary: dict, archived: dict) -> list[dict]:
    rows = []
    for model in summary["models"]:
        old = next(item for item in archived["models"] if item["model_id"] == model["model_id"])
        rows.append({
            "model_id": model["model_id"],
            "roots": model["roots"],
            "clusters": model["clusters"],
            "baseline_accuracy": model["baseline"]["accuracy"],
            "baseline_macro_f1": model["baseline"]["macro_f1"],
            "d5_accuracy": model["d5"]["accuracy"],
            "d5_macro_f1": model["d5"]["macro_f1"],
            "archived_summary_equal": model == old,
        })
    return rows


def verify_readme_positive() -> dict:
    started = time.time()
    tmp = OUT / f"_verify_r3_positive_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        bundle = materialise(tmp)
        out_dir = tmp / "out" / "d5-replay"
        with zipfile.ZipFile(R3_ZIP) as archive:
            text = archive.read("REPRODUCE.txt").decode("utf-8")
            archived = json.loads(archive.read("d5/summary.json"))
        run = run_readme_block(tmp, bundle, out_dir, text)
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8")) if out_dir.is_dir() else None
        rows = compare_models(summary, archived) if summary else []
        return {
            "block_executed": run["block"],
            "script_sha256": run["script_sha256"],
            "invocation": "pwsh -NoProfile -File reproduce_section4.ps1",
            "process_cwd": str(tmp),
            "bundle_root": str(bundle),
            "out_dir": str(out_dir),
            "exit_code": run["result"].returncode,
            "stdout": run["result"].stdout.strip(),
            "stderr_tail": run["result"].stderr.strip()[-800:],
            "summary_exists": summary is not None,
            "new_training": summary.get("new_training") if summary else None,
            "new_model_inference": summary.get("new_model_inference") if summary else None,
            "models": rows,
            "all_equal": bool(rows) and len(rows) == 6 and all(row["archived_summary_equal"] for row in rows),
            "seconds": round(time.time() - started, 2),
        }
    finally:
        shutil.rmtree(tmp)
        if tmp.exists():
            raise RuntimeError("正例临时目录未能删除")


def verify_negative_existing_output() -> dict:
    started = time.time()
    tmp = OUT / f"_verify_r3_neg_existing_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        bundle = materialise(tmp)
        out_dir = tmp / "existing-output"
        out_dir.mkdir()
        sentinel = out_dir / "sentinel.txt"
        sentinel.write_text("SENTINEL-KEEP-ME r3\n", encoding="utf-8")
        before = {"sentinel_sha256": sha256_file(sentinel), "listing": sorted(p.name for p in out_dir.iterdir())}
        with zipfile.ZipFile(R3_ZIP) as archive:
            text = archive.read("REPRODUCE.txt").decode("utf-8")
        run = run_readme_block(tmp, bundle, out_dir, text)
        after = {"sentinel_exists": sentinel.is_file(),
                 "sentinel_sha256": sha256_file(sentinel) if sentinel.is_file() else None,
                 "listing": sorted(p.name for p in out_dir.iterdir())}
        guarded = "输出目录已存在" in (run["result"].stderr + run["result"].stdout)
        return {
            "block_executed": run["block"],
            "process_cwd": str(tmp),
            "existing_out_dir": str(out_dir),
            "exit_code": run["result"].returncode,
            "stdout": run["result"].stdout.strip(),
            "stderr_tail": run["result"].stderr.strip()[-800:],
            "guard_message_seen": guarded,
            "python_ran": (out_dir / "summary.json").exists(),
            "sentinel_before": before,
            "sentinel_after": after,
            "sentinel_preserved": before["sentinel_sha256"] == after["sentinel_sha256"] and after["sentinel_exists"],
            "dir_unchanged": before["listing"] == after["listing"],
            "ok": (run["result"].returncode != 0 and guarded and not (out_dir / "summary.json").exists()
                   and before["sentinel_sha256"] == after["sentinel_sha256"] and after["sentinel_exists"]
                   and before["listing"] == after["listing"]),
            "seconds": round(time.time() - started, 2),
        }
    finally:
        shutil.rmtree(tmp)
        if tmp.exists():
            raise RuntimeError("负例 a 临时目录未能删除")


def verify_negative_missing_index() -> dict:
    started = time.time()
    tmp = OUT / f"_verify_r3_neg_missing_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        bundle = materialise(tmp, with_index=False)
        out_dir = tmp / "out" / "d5-replay"
        with zipfile.ZipFile(R3_ZIP) as archive:
            text = archive.read("REPRODUCE.txt").decode("utf-8")
        run = run_readme_block(tmp, bundle, out_dir, text)
        combined = run["result"].stdout + run["result"].stderr
        return {
            "block_executed": run["block"],
            "process_cwd": str(tmp),
            "bundle_root": str(bundle),
            "index_present": (bundle / "d5/derived-portable-index.json").exists(),
            "exit_code": run["result"].returncode,
            "stdout": run["result"].stdout.strip(),
            "stderr_tail": run["result"].stderr.strip()[-1200:],
            "python_error_seen": ("FileNotFoundError" in combined or "No such file" in combined),
            "ps1_propagated_failure": "python 退出码" in combined,
            "out_dir_created": out_dir.exists(),
            "ok": (run["result"].returncode != 0
                   and ("FileNotFoundError" in combined or "No such file" in combined)
                   and "python 退出码" in combined),
            "seconds": round(time.time() - started, 2),
        }
    finally:
        shutil.rmtree(tmp)
        if tmp.exists():
            raise RuntimeError("负例 b 临时目录未能删除")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    for path in (R1_ZIP, R2_ZIP, PACKER):
        if not path.is_file():
            raise RuntimeError(f"缺少前序文件: {path}")

    packer = load_packer()
    packer.ZIP_PATH = R3_ZIP
    packer.MANIFEST_PATH = R3_MANIFEST
    packer.RECEIPT_PATH = R3_RECEIPT
    packer.build_reproduce_txt = build_reproduce_txt
    packer.build_package_json = build_package_json

    members = packer.collect_members()
    sanity = packer.sanity_checks(members)
    archive_info = packer.write_zip(members)
    verification = packer.verify_zip(archive_info, members)
    crc = verify_crc()
    prev = verify_prev_comparison()
    positive = verify_readme_positive()
    neg_existing = verify_negative_existing_output()
    neg_missing = verify_negative_missing_index()

    problems = list(verification["problems"])
    if not crc["ok"]:
        problems.append(f"CRC 不符: {crc['mismatch']}")
    if not prev["ok"]:
        problems.append("r2↔r3 成员比对异常")
    if positive["exit_code"] != 0:
        problems.append(f"README 正例退出码 {positive['exit_code']}")
    if not positive["all_equal"]:
        problems.append("README 正例逐模型与归档不一致")
    if positive["new_training"] != 0 or positive["new_model_inference"] != 0:
        problems.append("正例报告了训练或新增推理")
    if not neg_existing["ok"]:
        problems.append("负例 a（已存在输出目录）未按预期失败或动了 sentinel")
    if not neg_missing["ok"]:
        problems.append("负例 b（缺索引）未传播失败")

    receipt = {
        "schema": "c3-d5-reproduction-bundle-verify-receipt-r3-v1",
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_commit": packer.COMMIT,
        "archive": archive_info,
        "supersedes": {"version": "r2", "zip_sha256": prev["compared_sha256"]},
        "sanity": sanity,
        "weights_referenced_not_included": {
            name: {"bytes": meta["bytes"], "sha256": meta["sha256"]}
            for name, meta in packer.WEIGHTS.items()
        },
        "manifest_members": verification["members_checked"],
        "manifest_matches_archive": verification["manifest_matches"],
        "git_show_sources_match": verification["git_show_sources_match"],
        "crc": crc,
        "r2_comparison": prev,
        "reproduce_build": BUILD_INFO.get("reproduce"),
        "function_level_replay": verification["replay"],
        "readme_positive": positive,
        "negative_existing_output": neg_existing,
        "negative_missing_index": neg_missing,
        "problems": problems,
        "ok": not problems,
        "excluded": packer.EXCLUDED,
    }

    R3_RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": not problems,
        "zip": archive_info,
        "crc_checked": crc["checked"],
        "r2_identical": prev["identical_count"],
        "r2_changed_allowed": prev["changed_allowed"],
        "r2_unexpected": prev["unexpected_changes"],
        "reproduce_reversible_to_r2": BUILD_INFO["reproduce"]["reversible_to_r2"],
        "reproduce_has_remove_item": not BUILD_INFO["reproduce"]["no_remove_item"],
        "positive_exit": positive["exit_code"],
        "positive_all_equal": positive["all_equal"],
        "positive_models": len(positive["models"]),
        "negative_existing_exit": neg_existing["exit_code"],
        "negative_existing_sentinel_preserved": neg_existing["sentinel_preserved"],
        "negative_missing_index_exit": neg_missing["exit_code"],
        "receipt": str(R3_RECEIPT),
        "receipt_sha256": sha256_file(R3_RECEIPT),
        "problems": problems,
    }, ensure_ascii=False, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
