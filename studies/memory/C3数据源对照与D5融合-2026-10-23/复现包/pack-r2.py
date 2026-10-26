"""r2 构建与验真：只修 REPRODUCE 说明与包版本元数据，科学成员与 r1 逐字节相同。

复用 pack.py（r1，保留不动）中的成员收集/校验/压缩函数；本文件只改写说明文本、
输出文件名，并新增 r1↔r2 成员比对与「按新 README 的 PowerShell CLI」真实重放。
0 训练、0 新增神经网络推理。
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

R1_ZIP = OUT / "记忆关系轻量模型-C3D5复现包-2026-10-23.zip"
R2_ZIP = OUT / "记忆关系轻量模型-C3D5复现包-2026-10-23-r2.zip"
R2_MANIFEST = OUT / "记忆关系轻量模型-C3D5复现包-2026-10-23-r2.manifest.sha256"
R2_RECEIPT = OUT / "verify-receipt-r2.json"
COMMIT = "b5b911f0bef5642193071875b365163bddf98de7"

PACKER = OUT / "pack.py"
ALLOWED_CHANGES = {"REPRODUCE.txt", "PACKAGE.json", "MANIFEST.sha256"}


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


def git_blob_sha(repo_path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", f"{COMMIT}:{repo_path}"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git rev-parse 失败 {repo_path}")
    return result.stdout.decode().strip()


def build_reproduce_txt() -> str:
    run_summary = json.loads((ROOT / "research/memory_relations/runs/source-control-20261023/d5/summary.json").read_text(encoding="utf-8"))
    index_sha = run_summary["index_sha256"]
    return f"""记忆关系轻量模型 · C3 数据源对照与 D5 融合 —— 复现资料包（r2）
================================================================

版本：r2（2026-10-23）。相对 r1 只修订本说明与 PACKAGE.json 的版本元数据；
source/、data/、d5/、inputs/、audit/、eval-official/、eval-fine/、models/、
artifacts/、blind36/、protocol/ 下的科学成员与 r1 逐字节相同。

本包不含模型权重、不含预训练骨干。两件模型各 470686061 B 的 nli_model.pt
仍留在原仓库 run 目录（research/memory_relations/runs/source-control-20261023/models/{{official,fine}}/），
不随包分发。因此本包不是独立可运行产品，也不是新的 portable 交付物：它是
「既有仓库 + 既有研究环境 + 离线 MiniLM 骨干」条件下的数据、方法与结果复现材料。

0. 包内不包含什么
------------------
- 模型权重 / PT / native / Adam 优化器状态 / tokenizer / 预训练 MiniLM 骨干
- .venv、缓存、git 元数据、任何凭据或秘密
- 研究报告.md（单独交付）、打包时仍在写入的 postmerge-affected.xml

1. 源版本与前置环境
------------------
- 源 commit: b5b911f0bef5642193071875b365163bddf98de7（master，2026-10-23）
- 4 个源码/文档成员是上述 commit 的 git show 字节，不是打包时工作树内容：
    source/MODULES.md
    source/research/memory_relations/README.md
    source/research/memory_relations/source_control.py
    source/research/memory_relations/test_source_control.py
  这 4 个文件不是独立 App：source_control.py 依赖完整仓库的
    dataset_recommender.agent.semantic_relations（SR）、research.memory_relations.metrics（M）、
    nli_reference / finetune / eval_nli_metrics 等既有 NN/FT/metrics 模块，
  以及 models/embeddings/paraphrase-multilingual-MiniLM-L12-v2 离线骨干。
  仅把 4 个文件放进目录无法运行；评分与重放都要求 ROOT 仓库在位。
- 运行环境：直接使用既有仓库的研究环境（ROOT/.venv，Python 3.12.x；numpy 2.5.2，
  scikit-learn、SciPy 等既有研究依赖）。注意 research/memory_relations/metrics.py
  顶层 `from sklearn.metrics import ...`，所以 D5 重放虽然只做确定性后处理，
  仍然需要 scikit-learn/SciPy；不要声称「只需 Python+numpy」。
- D5 重放：不加载任何模型权重、不做新增神经网络推理、不需要 CUDA/GPU；
  但与上面一致，仍需要上述既有研究依赖，无需新装依赖。
- 重训（可选，非本包验收路径）：另需原仓库 .venv 中的 torch / transformers
  与离线 MiniLM 骨干 + CUDA（BF16）；同样无需新装依赖。

2. 固定配方（与 run.json / data/manifest.json 一致）
----------------------------------------------------
- seed = 13
- C3 共同更新 = 361 次（max(1, floor(20 * n / (8 * 2)))），有效 batch = 8*2 = 16
- max_length = 384；BF16 + CUDA；优化器 AdamW（骨干 2e-5 / 头 1e-3 / wd 0.01）
- 两臂同一初始骨干与三类头 tensor；只做 forward CE，不反向扩增、不伪造反向标签
- 数据匹配：official 臂 289 行 / 289 簇；fine 臂 289 行（289 根）/ 60 簇。
  两臂匹配的是独立根数与逐类数量，簇数天然不同（official 一簇一行，fine 多根共享簇），
  不是「各 60 簇」。
- 未标注 gold 的样本不补造标签。

3. 重训（可选；本包验收不要求，勿在执行验收时运行）
--------------------------------------------------
训练入口必须显式传本包内的 JSONL，不依赖历史默认 data 路径。PowerShell：

    $RepoRoot   = '<完整仓库绝对路径>'          # 例：D:\\work\\biodata-agent
    $BundleRoot = '<本包解压根>'               # 例：D:\\work\\c3d5-r2
    $Python     = Join-Path $RepoRoot '.venv/Scripts/python.exe'
    $ModelsOut  = '<新的绝对输出目录>/models'   # 例：D:\\work\\c3d5-r2\\retrain\\models
    $env:PYTHONPATH              = "$RepoRoot;$RepoRoot/src"
    $env:PYTHONDONTWRITEBYTECODE = '1'
    Set-Location -LiteralPath $RepoRoot

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
    "exit=$LASTEXITCODE"

同参数才可能接近原 run；不同 GPU/驱动/torch 版本不保证逐位相同。

4. 离线重放 D5（无需重训、无需权重）
----------------------------------
D5 重放只读逐题预测 JSONL 做双向五叶平均融合与簇自助，属纯确定性后处理。
6 个原始逐题输入在 inputs/d5-original/<model_id>.jsonl（原 SHA 已核），
d5/derived-portable-index.json 是显式派生的便携索引（不是新审计、不改原审计）。

关键一点：派生的便携索引里，输入路径是「相对解压根目录」的，所以必须先
Set-Location 到 $BundleRoot 再执行，否则 analyze_d5 找不到输入。完整 PowerShell：

    $RepoRoot   = '<完整仓库绝对路径>'          # 例：D:\\work\\biodata-agent
    $BundleRoot = '<本包解压根>'               # 例：D:\\work\\c3d5-r2
    $Python     = Join-Path $RepoRoot '.venv/Scripts/python.exe'
    $env:PYTHONPATH              = "$RepoRoot;$RepoRoot/src"
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:HF_HUB_OFFLINE          = '1'
    $env:TRANSFORMERS_OFFLINE    = '1'

    $Out = '<新的绝对输出目录>/d5-replay'       # 例：D:\\work\\c3d5-r2\\replay\\d5-replay
    if (Test-Path -LiteralPath $Out) {{ Remove-Item -LiteralPath $Out -Recurse -Force }}   # analyze_d5 要求目录不存在

    Set-Location -LiteralPath $BundleRoot        # 必须在解压根执行：索引内是相对路径
    & $Python -B -m research.memory_relations.source_control d5 `
        --index (Join-Path $BundleRoot 'd5/derived-portable-index.json') `
        --out   $Out `
        --bootstrap-samples 1000
    "exit=$LASTEXITCODE"

说明：`-m research.memory_relations.source_control` 由 $PYTHONPATH 指向完整仓库里的同名模块，
其字节与包内 source/research/memory_relations/source_control.py 相同（manifest 与 git show 已核）。

只用本包内单文件模块执行（等价做法）：把下面的 Python 片段存成 $BundleRoot\\replay.py，
再按上面的环境设置执行 `& $Python -B replay.py`。

    import importlib.util, pathlib
    p = pathlib.Path('source/research/memory_relations/source_control.py').resolve()
    spec = importlib.util.spec_from_file_location('sc_from_bundle', p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.analyze_d5('d5/derived-portable-index.json', 'replay-out', samples=1000)

（该片段同样要在 $BundleRoot 下运行；'replay-out' 也必须是尚不存在的目录。）

注意：重放产生的 summary.json 中 index_sha256 会不同于归档 d5/summary.json 的
index_sha256（原值 {index_sha}，即原父验收索引文件的 SHA；
派生索引自身 SHA 见 d5/derived-portable-index.json），这是路径派生导致的预期差异，
不是数据变化；逐模型条目应与归档逐字段一致。

5. 查评分（无需重训）
--------------------
- eval-official/eval.json、eval-fine/eval.json：两臂在 ocnli_dev（2950 行）与 fine_final
  （206 根）上的损失/准确率/macro-F1/截断统计，均引用本包内同名 predictions.jsonl 的 SHA。
- d5/summary.json：6 个 C5 工件（v2/v3 × 3 seeds）的 C5 基线与 D5 融合指标、簇自助区间；
  每件 roots=206、clusters=43。
- 逐行复算可用既有仓库 research.memory_relations.metrics.classification_summary(
  y_true, y_pred, classes=("entailment","neutral","contradiction"))，权重口径见 eval.json。

6. 口径与限制（必须原样保留）
----------------------------
- 既有 fine-final 题集在更早已曝光，只能作 posthoc / exploratory，不能作为新确认证据。
- 新写的 35 题短盲确认集（原作者 36 题，1 题双方分歧按规则 held，保存不弃）
  禁止进入任何训练或选点。它只经历过一次锁定的评测会话：同一会话内为导出概率
  并与标准指标核对，对相同权重、相同输入做过重复前向；没有调参、没有选模、没有重训，
  也没有按这 35 题回头改动模型或阈值。
- 该 35 题参考标签是「作者 + 独立盲审代理」的一致性代理，不是真人 gold，
  不构成真人准确率或教师质量认证；报告只能谈预算内来源效用。
- D5 融合是 posthoc 分析（new_training=0, new_model_inference=0），
  平局按既有类序取先，硬等价性不作为保证。
- artifacts/ 内的 JSON/JSONL 保留原始绝对路径与原始 SHA，属于审计原字节；
  本包未做脱敏改写（私有研究材料，非公开交付面）。

7. 成员校验
-----------
MANIFEST.sha256（包内）与同名 .manifest.sha256（包外）逐成员给出
sha256、字节数、相对路径；包外清单额外覆盖 MANIFEST.sha256 自身。
verify-receipt-r2.json（包外）记录生成 ZIP 的 SHA、成员数、逐成员 CRC/SHA、
与 r1 的成员比对、以及按上面 PowerShell 命令真实重放 D5 的命令、cwd、退出码与逐模型对照。
"""


def build_package_json() -> dict:
    return {
        "schema": "c3-d5-private-reproduction-bundle-v1",
        "bundle_version": "r2",
        "revision_note": "只修订 REPRODUCE 说明与包版本元数据；科学成员与 r1 逐字节相同。",
        "built_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "kind": "private_research_reproduction_material",
        "public_release_candidate": False,
        "supersedes": {
            "version": "r1",
            "zip": R1_ZIP.name,
            "zip_sha256": sha256_file(R1_ZIP),
        },
        "changes_vs_r1": [
            "REPRODUCE §1：删除「D5 只需 Python+numpy」的错误说法；改为使用既有仓库研究环境（numpy、scikit-learn、SciPy），原因：research/memory_relations/metrics.py 顶层 import sklearn.metrics。",
            "REPRODUCE §4：D5 示例改为可直接执行的 PowerShell，并明确 $RepoRoot/$BundleRoot/$Python、绝对 PYTHONPATH、Set-Location 到解压根、--index 用 Join-Path 绝对路径。",
            "REPRODUCE §3：训练示例改为 PowerShell 反引号/数组调用并显式 PYTHONPATH，不再混用 Bash 语法。",
            "REPRODUCE §2：簇数拆开表述为 official 289 簇 / fine 60 簇（两者都是 289 行/289 根）。",
            "REPRODUCE §6：35 题确认集改为「一次锁定评测会话，同权重同输入的重复前向用于概率导出与标准指标核对，无调参/选模/重训」。",
            "PACKAGE.json：新增 r2 版本元数据、运行环境声明、supersedes 与变更清单。",
            "MANIFEST.sha256：随上述两个文本成员重算。",
        ],
        "unchanged_vs_r1": [
            "source/、data/、d5/、inputs/、audit/、eval-official/、eval-fine/、models/、artifacts/、blind36/、protocol/ 全部成员逐字节相同",
        ],
        "source": {
            "repository": "biodata-agent-private (local ROOT)",
            "commit": "b5b911f0bef5642193071875b365163bddf98de7",
            "commit_date": "2026-10-23T03:22:36+08:00",
            "git_show_sources": [
                {"repo_path": repo_path, "member": member, "git_blob": git_blob_sha(repo_path)}
                for repo_path, member in [
                    ("MODULES.md", "source/MODULES.md"),
                    ("research/memory_relations/README.md", "source/research/memory_relations/README.md"),
                    ("research/memory_relations/source_control.py", "source/research/memory_relations/source_control.py"),
                    ("research/memory_relations/test_source_control.py", "source/research/memory_relations/test_source_control.py"),
                ]
            ],
        },
        "portability": {
            "standalone": False,
            "requires_existing_repo": True,
            "requires_existing_repo_env": True,
            "requires_offline_backbone_for_retrain": "models/embeddings/paraphrase-multilingual-MiniLM-L12-v2",
            "model_weights_in_bundle": False,
            "note": "4 个源码文件不是独立 App；需完整仓库 b5b911f0 的 SR/M/NR/FT 模块、既有研究环境（含 scikit-learn/SciPy）与离线 MiniLM。",
        },
        "environment": {
            "repo_env": "ROOT/.venv（Python 3.12.x；numpy 2.5.2，scikit-learn、SciPy 等既有研究依赖）",
            "d5_replay": {
                "loads_model_weights": False,
                "new_nn_inference": False,
                "requires_cuda": False,
                "requires_sklearn_scipy": True,
                "reason": "research/memory_relations/metrics.py 顶层 import sklearn.metrics；不需要 torch/CUDA",
            },
            "retrain": {
                "requires_torch": True,
                "requires_transformers": True,
                "requires_cuda_bf16": True,
                "requires_offline_backbone": True,
                "new_dependencies_needed": False,
            },
            "offline_env": ["HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "PYTHONDONTWRITEBYTECODE=1"],
        },
        "external_weights": {
            "official": {
                "path_in_root": "research/memory_relations/runs/source-control-20261023/models/official/nli_model.pt",
                "bytes": 470686061,
                "sha256": "d2d8eedf59373d04fca07023fa0d55cbcff519eacbb0cf63071f2bf49a1f3307",
                "in_bundle": False,
            },
            "fine": {
                "path_in_root": "research/memory_relations/runs/source-control-20261023/models/fine/nli_model.pt",
                "bytes": 470686061,
                "sha256": "1bcbdc226eb7e5ea17fa204bd9e90e3810af437f4995385a52f3a13a6c45acf9",
                "in_bundle": False,
            },
        },
        "data": {
            "official_arm_rows": 289,
            "official_arm_clusters": 289,
            "fine_arm_rows": 289,
            "fine_arm_clusters": 60,
            "matched_independent_roots_per_arm": 289,
            "unlabeled_gold_policy": "不补造未标 gold；只用已标注匹配根",
        },
        "replay": {
            "entry": "research.memory_relations.source_control:analyze_d5",
            "cli": "python -B -m research.memory_relations.source_control d5 --index <bundle>/d5/derived-portable-index.json --out <new dir> --bootstrap-samples 1000",
            "cwd_requirement": "必须在解压根执行（派生索引内是相对路径）",
            "bootstrap_samples": 1000,
            "new_training": 0,
            "new_model_inference": 0,
            "derived_index": "d5/derived-portable-index.json",
            "original_index_member": "audit/parent-verification-index.original.json",
        },
        "excluded": [
            "模型权重（各 470686061 B，留在 ROOT 原目录）",
            "tokenizer/config 与预训练 MiniLM 骨干",
            ".venv、缓存、git 元数据、凭据",
            "研究报告.md 与打包时仍在写的 postmerge-affected.xml",
            "过程日志与探针、eval-plumbing-existing-v6-native/、dual3 臂",
        ],
        "limitations": [
            "fine-final 题集已曝光：仅 posthoc / exploratory。",
            "新的 35 题短盲确认集禁止进入训练或选点；原作者 36 题、1 题分歧 held 保存；只在一次锁定评测会话内做过重复前向（同权重同输入）以导出概率并核对指标，无调参/选模/重训。",
            "盲审参考标签为代理一致性，不是真人 gold，不认证真人或教师准确率。",
            "本包是私有复现材料，非公开交付面；artifacts 保留原始绝对路径与原始 SHA。",
            "重训不保证在其它硬件/驱动/torch 版本上逐位复现；D5 重放为确定性后处理。",
            "D5 重放需要既有仓库研究环境（含 scikit-learn/SciPy），不是仅 numpy 的最小环境。",
        ],
    }


def verify_crc() -> dict:
    rows = []
    with zipfile.ZipFile(R2_ZIP) as archive:
        for info in archive.infolist():
            data = archive.read(info)
            rows.append({
                "path": info.filename,
                "crc32_zip": f"{info.CRC:08x}",
                "crc32_calc": f"{zlib.crc32(data) & 0xFFFFFFFF:08x}",
                "ok": info.CRC == (zlib.crc32(data) & 0xFFFFFFFF),
            })
    bad = [row["path"] for row in rows if not row["ok"]]
    return {"checked": len(rows), "mismatch": bad, "ok": not bad}


def verify_r1_comparison() -> dict:
    with zipfile.ZipFile(R1_ZIP) as r1, zipfile.ZipFile(R2_ZIP) as r2:
        r1_names = set(r1.namelist())
        r2_names = set(r2.namelist())
        identical, changed = [], []
        for name in sorted(r1_names & r2_names):
            (identical if r1.read(name) == r2.read(name) else changed).append(name)
        only_r1 = sorted(r1_names - r2_names)
        only_r2 = sorted(r2_names - r1_names)
    unexpected = [name for name in changed if name not in ALLOWED_CHANGES]
    ok = not only_r1 and not only_r2 and not unexpected and set(changed) <= ALLOWED_CHANGES
    return {
        "r1_zip": R1_ZIP.name,
        "r1_sha256": sha256_file(R1_ZIP),
        "identical_count": len(identical),
        "identical_members": identical,
        "changed_allowed": changed,
        "unexpected_changes": unexpected,
        "only_in_r1": only_r1,
        "only_in_r2": only_r2,
        "ok": ok,
    }


PWSH_TEMPLATE = r"""
$ErrorActionPreference = 'Stop'
$RepoRoot   = 'REPO_ROOT'
$BundleRoot = 'BUNDLE_ROOT'
$Python     = Join-Path $RepoRoot '.venv/Scripts/python.exe'
$env:PYTHONPATH              = "$RepoRoot;$RepoRoot/src"
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HF_HUB_OFFLINE          = '1'
$env:TRANSFORMERS_OFFLINE    = '1'
$env:PYTHONIOENCODING        = 'utf-8'
$Out = 'OUT_DIR'
if (Test-Path -LiteralPath $Out) { Remove-Item -LiteralPath $Out -Recurse -Force }
Set-Location -LiteralPath $BundleRoot
"cwd=$((Get-Location).Path)"
& $Python -B -m research.memory_relations.source_control d5 `
    --index (Join-Path $BundleRoot 'd5/derived-portable-index.json') `
    --out   $Out `
    --bootstrap-samples 1000
"exit=$LASTEXITCODE"
exit $LASTEXITCODE
"""


def verify_cli() -> dict:
    started = time.time()
    tmp = OUT / f"_verify_r2_cli_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    bundle = tmp / "bundle"
    bundle.mkdir(parents=True)
    out_dir = tmp / "d5-replay"
    try:
        with zipfile.ZipFile(R2_ZIP) as archive:
            names = ["d5/derived-portable-index.json"]
            names += [name for name in archive.namelist() if name.startswith("inputs/d5-original/")]
            for name in names:
                target = bundle / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
            archived = json.loads(archive.read("d5/summary.json"))

        script = (PWSH_TEMPLATE
                  .replace("REPO_ROOT", str(ROOT))
                  .replace("BUNDLE_ROOT", str(bundle))
                  .replace("OUT_DIR", str(out_dir)))
        command = ["pwsh", "-NoProfile", "-Command", script]
        result = subprocess.run(command, cwd=str(bundle), capture_output=True, text=True, encoding="utf-8")

        report = {}
        if out_dir.is_dir():
            report = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

        comparison = {"all_equal": False, "models": []}
        if report:
            by_id = {model["model_id"]: model for model in archived["models"]}
            for model in report["models"]:
                old = by_id[model["model_id"]]
                comparison["models"].append({
                    "model_id": model["model_id"],
                    "roots": model["roots"],
                    "clusters": model["clusters"],
                    "baseline_accuracy": model["baseline"]["accuracy"],
                    "baseline_macro_f1": model["baseline"]["macro_f1"],
                    "d5_accuracy": model["d5"]["accuracy"],
                    "d5_macro_f1": model["d5"]["macro_f1"],
                    "source_sha256": model["source_sha256"],
                    "predictions_sha256": model["predictions_sha256"],
                    "archived_summary_equal": model == old,
                })
            comparison["all_equal"] = (
                len(comparison["models"]) == len(archived["models"]) == 6
                and all(row["archived_summary_equal"] for row in comparison["models"])
            )

        return {
            "script": script,
            "command": "pwsh -NoProfile -Command <上面的 script>",
            "cwd": str(bundle),
            "pwsh_exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr_tail": result.stderr.strip()[-1500:],
            "index_argument": str(bundle / "d5/derived-portable-index.json"),
            "index_used_relative_member_paths": True,
            "out_dir": str(out_dir),
            "new_training": report.get("new_training"),
            "new_model_inference": report.get("new_model_inference"),
            "bootstrap_samples": report.get("bootstrap_samples"),
            "models": comparison["models"],
            "all_equal": comparison["all_equal"],
            "seconds": round(time.time() - started, 2),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=False)
        if tmp.exists():
            raise RuntimeError("CLI 临时目录未能删除")


def verify_readme_block() -> dict:
    """按 ZIP 内 REPRODUCE §4 的原文执行，只替换三个占位符；cwd 故意不放在 bundle。"""
    started = time.time()
    tmp = OUT / f"_verify_r2_readme_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    bundle = tmp / "bundle"
    bundle.mkdir(parents=True)
    out_dir = tmp / "out" / "d5-replay"
    try:
        with zipfile.ZipFile(R2_ZIP) as archive:
            text = archive.read("REPRODUCE.txt").decode("utf-8")
            names = ["d5/derived-portable-index.json"]
            names += [name for name in archive.namelist() if name.startswith("inputs/d5-original/")]
            for name in names:
                target = bundle / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
            archived = json.loads(archive.read("d5/summary.json"))

        section = text.split("4. 离线重放 D5", 1)[1].split("5. 查评分", 1)[0]
        lines = section.splitlines()
        start = next(index for index, line in enumerate(lines) if line.strip().startswith("$RepoRoot"))
        end = next(index for index, line in enumerate(lines) if '"exit=$LASTEXITCODE"' in line)
        block = "\n".join(line[4:] if line.startswith("    ") else line for line in lines[start:end + 1])
        for placeholder, value in [
            ("<完整仓库绝对路径>", str(ROOT)),
            ("<本包解压根>", str(bundle)),
            ("<新的绝对输出目录>/d5-replay", str(out_dir)),
        ]:
            if placeholder not in block:
                raise RuntimeError(f"REPRODUCE §4 缺少占位符: {placeholder}")
            block = block.replace(placeholder, value)

        script = tmp / "reproduce_section4.ps1"
        script.write_text(block, encoding="utf-8-sig")
        result = subprocess.run(["pwsh", "-NoProfile", "-File", str(script)],
                                cwd=str(tmp), capture_output=True, text=True, encoding="utf-8")

        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8")) if out_dir.is_dir() else None
        rows = []
        if summary:
            by_id = {model["model_id"]: model for model in archived["models"]}
            for model in summary["models"]:
                old = by_id[model["model_id"]]
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
        return {
            "block_executed": block,
            "script_sha256": sha256_file(script),
            "invocation": f"pwsh -NoProfile -File {script.name}",
            "process_cwd": str(tmp),
            "bundle_root": str(bundle),
            "out_dir": str(out_dir),
            "process_exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr_tail": result.stderr.strip()[-1000:],
            "summary_exists": summary is not None,
            "new_training": summary.get("new_training") if summary else None,
            "new_model_inference": summary.get("new_model_inference") if summary else None,
            "models": rows,
            "all_equal": bool(rows) and len(rows) == 6 and all(row["archived_summary_equal"] for row in rows),
            "seconds": round(time.time() - started, 2),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=False)
        if tmp.exists():
            raise RuntimeError("README 原样执行临时目录未能删除")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    for path in (R1_ZIP, PACKER):
        if not path.is_file():
            raise RuntimeError(f"缺少 r1 文件: {path}")

    packer = load_packer()
    packer.ZIP_PATH = R2_ZIP
    packer.MANIFEST_PATH = R2_MANIFEST
    packer.RECEIPT_PATH = R2_RECEIPT
    packer.build_reproduce_txt = build_reproduce_txt
    packer.build_package_json = build_package_json

    members = packer.collect_members()
    sanity = packer.sanity_checks(members)
    archive_info = packer.write_zip(members)
    verification = packer.verify_zip(archive_info, members)
    crc = verify_crc()
    r1_diff = verify_r1_comparison()
    cli = verify_cli()
    readme = verify_readme_block()

    problems = list(verification["problems"])
    if not crc["ok"]:
        problems.append(f"CRC 不符: {crc['mismatch']}")
    if not r1_diff["ok"]:
        problems.append("r1↔r2 成员比对异常")
    if cli["pwsh_exit_code"] != 0:
        problems.append(f"CLI 重放退出码 {cli['pwsh_exit_code']}")
    if not cli["all_equal"]:
        problems.append("CLI 重放逐模型结果与归档不一致")
    if cli["new_training"] != 0 or cli["new_model_inference"] != 0:
        problems.append("CLI 重放报告了训练或新增推理")
    if readme["process_exit_code"] != 0:
        problems.append(f"README 原样执行退出码 {readme['process_exit_code']}")
    if not readme["all_equal"]:
        problems.append("README 原样执行逐模型结果与归档不一致")
    if readme["new_training"] != 0 or readme["new_model_inference"] != 0:
        problems.append("README 原样执行报告了训练或新增推理")

    receipt = {
        "schema": "c3-d5-reproduction-bundle-verify-receipt-r2-v1",
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_commit": packer.COMMIT,
        "archive": archive_info,
        "supersedes": {"version": "r1", "zip_sha256": r1_diff["r1_sha256"]},
        "sanity": sanity,
        "weights_referenced_not_included": {
            name: {"bytes": meta["bytes"], "sha256": meta["sha256"]}
            for name, meta in packer.WEIGHTS.items()
        },
        "manifest_members": verification["members_checked"],
        "manifest_matches_archive": verification["manifest_matches"],
        "git_show_sources_match": verification["git_show_sources_match"],
        "crc": crc,
        "r1_comparison": r1_diff,
        "function_level_replay": verification["replay"],
        "cli_replay": cli,
        "readme_literal_replay": readme,
        "problems": problems,
        "ok": not problems,
        "excluded": packer.EXCLUDED,
    }

    R2_RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": not problems,
        "zip": archive_info,
        "crc_checked": crc["checked"],
        "r1_identical": r1_diff["identical_count"],
        "r1_changed_allowed": r1_diff["changed_allowed"],
        "r1_unexpected": r1_diff["unexpected_changes"],
        "cli_exit": cli["pwsh_exit_code"],
        "cli_all_equal": cli["all_equal"],
        "cli_models": len(cli["models"]),
        "readme_exit": readme["process_exit_code"],
        "readme_all_equal": readme["all_equal"],
        "readme_models": len(readme["models"]),
        "receipt": str(R2_RECEIPT),
        "receipt_sha256": sha256_file(R2_RECEIPT),
        "problems": problems,
    }, ensure_ascii=False, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
