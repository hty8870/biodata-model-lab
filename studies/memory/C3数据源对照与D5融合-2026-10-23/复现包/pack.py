"""打包并验真 C3 数据源对照 / D5 融合的私有复现资料包。

只读取 ROOT 既有产物，不回写 ROOT；不做训练、不做新模型推理。
打包完成后立刻从生成的 ZIP 本身验真（ZipFile 逐成员校验 + 选择性解压 + 真实 analyze_d5 重放）。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(r"<archive-root>")
EXT = Path(r"<archive-root>\C3数据源对照与D5融合-2026-10-23")
OUT = Path(__file__).resolve().parent

COMMIT = "b5b911f0bef5642193071875b365163bddf98de7"
RUN = ROOT / "research/memory_relations/runs/source-control-20261023"
ART = ROOT / "artifacts/c3-d5-source-control-20261023"
PARENT_INDEX = ROOT / "artifacts/v55-final-preflight-20261021/fine-final-parent-verification.json"

ZIP_PATH = OUT / "记忆关系轻量模型-C3D5复现包-2026-10-23.zip"
MANIFEST_PATH = OUT / "记忆关系轻量模型-C3D5复现包-2026-10-23.manifest.sha256"
RECEIPT_PATH = OUT / "verify-receipt.json"

PYEXE = ROOT / ".venv/Scripts/python.exe"
FIXED_DATE = (2026, 10, 23, 0, 0, 0)

WEIGHTS = {
    "official": {
        "path": RUN / "models/official/nli_model.pt",
        "sha256": "d2d8eedf59373d04fca07023fa0d55cbcff519eacbb0cf63071f2bf49a1f3307",
        "bytes": 470686061,
    },
    "fine": {
        "path": RUN / "models/fine/nli_model.pt",
        "sha256": "1bcbdc226eb7e5ea17fa204bd9e90e3810af437f4995385a52f3a13a6c45acf9",
        "bytes": 470686061,
    },
}

GIT_SOURCES = [
    ("MODULES.md", "source/MODULES.md"),
    ("research/memory_relations/README.md", "source/research/memory_relations/README.md"),
    ("research/memory_relations/source_control.py", "source/research/memory_relations/source_control.py"),
    ("research/memory_relations/test_source_control.py", "source/research/memory_relations/test_source_control.py"),
]

DATA_FILES = [
    "official_train.jsonl",
    "fine_train.jsonl",
    "ocnli_dev.jsonl",
    "fine_final.jsonl",
    "manifest.json",
    "truncation.json",
]

D5_JSONL = [
    "v2-c5-seed13.jsonl",
    "v2-c5-seed20260919.jsonl",
    "v2-c5-seed424242.jsonl",
    "v3-c5-seed13.jsonl",
    "v3-c5-seed20260919.jsonl",
    "v3-c5-seed424242.jsonl",
]

ARTIFACT_FILES = [
    "analyze_c3.py",
    "verify_train_reload.py",
    "run_formal.ps1",
    "prepare_blind36.py",
    "evaluate_blind36.py",
    "independent-blind-plan.json",
    "paired-analysis.json",
    "train-eval-reload.json",
    "formal-preflight.json",
    "formal-stages.jsonl",
    "quality-full.json",
    "quality-fast.json",
]

ROOT_BLIND = [
    ("author-map.json", "blind36/author-map.json"),
    ("authors-lock.json", "blind36/authors-lock.json"),
    ("review-input.json", "blind36/review-input.json"),
    ("evaluation/admission-lock.json", "blind36/evaluation/admission-lock.json"),
    ("evaluation/confirmed.jsonl", "blind36/evaluation/confirmed.jsonl"),
    ("evaluation/fine.predictions.jsonl", "blind36/evaluation/fine.predictions.jsonl"),
    ("evaluation/official.predictions.jsonl", "blind36/evaluation/official.predictions.jsonl"),
    ("evaluation/reference-audit.json", "blind36/evaluation/reference-audit.json"),
    ("evaluation/summary.json", "blind36/evaluation/summary.json"),
]

EXT_BLIND = [
    ("author-A.json", "blind36/author-A.json"),
    ("author-B.json", "blind36/author-B.json"),
    ("reviewer-primary.json", "blind36/reviewer-primary.json"),
]

MODEL_FILES = [
    ("run.json", "models/run.json"),
    ("official.training.json", "models/official.training.json"),
    ("fine.training.json", "models/fine.training.json"),
    ("official/nli_meta.json", "models/official/nli_meta.json"),
    ("fine/nli_meta.json", "models/fine/nli_meta.json"),
]

EXCLUDED = [
    "models/official/nli_model.pt, models/fine/nli_model.pt（各 470686061 B，权重留在 ROOT 原目录）",
    "models/*/tokenizer/、models/*/config/ 与预训练骨干 MiniLM（models/embeddings/…）",
    ".venv/、__pycache__/.pytest_cache/ 等缓存与运行环境",
    ".git/、.env、凭据与任何秘密文件",
    "研究报告.md（单独交付）与 artifacts/…/postmerge-affected.xml（打包时仍在写入）",
    "artifacts/…/{train-pair.log,eval-official.log,eval-fine.log} 与 run 下 probe_*/smoke-*.json（过程日志/探针，不是数据或方法）",
    "eval-plumbing-existing-v6-native/（与 C3/D5 结论无关的管线复测目录）",
    "artifacts/v55-final-preflight-20261021/ 下除父验收索引外的其它模型 report/model 矩阵",
    "全部 dual3 臂（只按任务要求保留 arm=c5 的 6 项）",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def git_show(path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{COMMIT}:{path}"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git show 失败 {path}: {result.stderr.decode('utf-8', 'replace')}")
    return result.stdout


def git_blob_sha(path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", f"{COMMIT}:{path}"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git rev-parse 失败 {path}")
    return result.stdout.decode().strip()


def build_reproduce_txt() -> str:
    return f"""记忆关系轻量模型 · C3 数据源对照与 D5 融合 —— 复现资料包
================================================================

本包不含模型权重、不含预训练骨干。两件模型各 {WEIGHTS['official']['bytes']} B 的 nli_model.pt
仍留在原仓库 run 目录（research/memory_relations/runs/source-control-20261023/models/{{official,fine}}/），
不随包分发。因此本包不是独立可运行产品，也不是新的 portable 交付物：它是
「既有仓库 + 离线 MiniLM 骨干」条件下的数据、方法与结果复现材料。

0. 包内不包含什么
------------------
- 模型权重 / PT / native / Adam 优化器状态 / tokenizer / 预训练 MiniLM 骨干
- .venv、缓存、git 元数据、任何凭据或秘密
- 研究报告.md（单独交付）、打包时仍在写入的 postmerge-affected.xml

1. 源版本与前置环境
------------------
- 源 commit: {COMMIT}（master，2026-10-23）
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
- Python: ROOT/.venv（3.12.x），numpy 2.5.2；D5 重放只需 Python+numpy，不需要 torch/CUDA。
- 训练侧环境（若真要重训，非本包验收路径）: CUDA + BF16，torch 由既有仓库 .venv 提供。
- 全程离线：HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1，不联网、不装依赖。

2. 固定配方（与 run.json / data/manifest.json 一致）
----------------------------------------------------
- seed = 13
- C3 共同更新 = 361 次（max(1, floor(20 * n / (8 * 2)))），有效 batch = 8*2 = 16
- max_length = 384；BF16 + CUDA；优化器 AdamW（骨干 2e-5 / 头 1e-3 / wd 0.01）
- 两臂同一初始骨干与三类头 tensor；只做 forward CE，不反向扩增、不伪造反向标签
- 训练数据只对应匹配到的 289 根/臂（official 289 行、fine 289 行，60 clusters）；
  未标注 gold 的样本不补造标签

3. 重训（可选；本包验收不要求）
------------------------------
训练入口必须显式传本包内的 JSONL，不依赖历史默认 data 路径：

    cd <ROOT>
    .venv/Scripts/python.exe -B -m research.memory_relations.source_control train-c3-pair \\
      --official-data <本包解压>/data/official_train.jsonl \\
      --fine-data     <本包解压>/data/fine_train.jsonl \\
      --prep-manifest <本包解压>/data/manifest.json \\
      --embedding-dir models/embeddings/paraphrase-multilingual-MiniLM-L12-v2 \\
      --out <新 run>/models --device cuda --seed 13 \\
      --batch-size 8 --grad-accum 2 --max-length 384 --budget-epochs 20

同参数才可能接近原 run；不同 GPU/驱动/torch 版本不保证逐位相同。

4. 离线重放 D5（无需重训、无需权重）
----------------------------------
D5 重放只读逐题预测 JSONL 做双向五叶平均融合与簇自助，属纯确定性后处理。
6 个原始逐题输入在 inputs/d5-original/<model_id>.jsonl（原 SHA 已核），
d5/derived-portable-index.json 是显式派生的便携索引（不是新审计、不改原审计）。

从完整仓库执行：

    cd <ROOT>
    PYTHONPATH="<ROOT>;<ROOT>/src" .venv/Scripts/python.exe -B \\
      -m research.memory_relations.source_control d5 \\
      --index <本包解压>/d5/derived-portable-index.json \\
      --out <新目录>/d5-replay --bootstrap-samples 1000

只用本包内单文件模块执行（验证方式，等价于本包 verify-receipt 的做法）：

    import importlib.util, pathlib, sys
    sys.path[:0] = [r"<ROOT>", r"<ROOT>/src"]
    p = pathlib.Path("<本包解压>/source/research/memory_relations/source_control.py")
    spec = importlib.util.spec_from_file_location("sc_from_bundle", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.analyze_d5("<本包解压>/d5/derived-portable-index.json", "<新目录>/d5-replay", samples=1000)

注意：派生索引的路径相对「解压根目录」，因此把 <本包解压> 当 cwd，或把索引里的
相对路径按解压根解析。重放产生的 summary.json 中 index_sha256 会不同于归档
d5/summary.json 的 index_sha256（原值 {json.loads((RUN / "d5/summary.json").read_text(encoding="utf-8"))["index_sha256"]}，
即原父验收索引文件的 SHA；派生索引的 SHA 见 d5/derived-portable-index.json 自身），
这是路径派生导致的预期差异，不是数据变化；逐模型条目应与归档逐字段一致。

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
  禁止进入任何训练或选点；它只在冻结模型上做过一次前向评分。
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
verify-receipt.json（包外）记录生成 ZIP 的 SHA、成员数、逐成员校验结果、
以及从 ZIP 解压后真实调用 analyze_d5 的返回码与逐模型对照。
"""


def build_package_json() -> dict:
    return {
        "schema": "c3-d5-private-reproduction-bundle-v1",
        "built_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "kind": "private_research_reproduction_material",
        "public_release_candidate": False,
        "source": {
            "repository": "biodata-agent-private (local ROOT)",
            "commit": COMMIT,
            "commit_date": "2026-10-23T03:22:36+08:00",
            "git_show_sources": [
                {"repo_path": src, "member": dst, "git_blob": git_blob_sha(src)}
                for src, dst in GIT_SOURCES
            ],
        },
        "portability": {
            "standalone": False,
            "requires_existing_repo": True,
            "requires_offline_backbone": "models/embeddings/paraphrase-multilingual-MiniLM-L12-v2",
            "model_weights_in_bundle": False,
            "note": "4 个源码文件不是独立 App；需完整仓库 b5b911f0 的 NR/FT/metrics 模块与离线 MiniLM。",
        },
        "external_weights": {
            name: {
                "path_in_root": str(meta["path"].relative_to(ROOT)).replace("\\", "/"),
                "bytes": meta["bytes"],
                "sha256": meta["sha256"],
                "in_bundle": False,
            }
            for name, meta in WEIGHTS.items()
        },
        "data": {
            "matched_roots_per_arm": 289,
            "matched_clusters": 60,
            "unlabeled_gold_policy": "不补造未标 gold；只用已标注匹配根",
        },
        "replay": {
            "entry": "research.memory_relations.source_control:analyze_d5",
            "bootstrap_samples": 1000,
            "new_training": 0,
            "new_model_inference": 0,
            "derived_index": "d5/derived-portable-index.json",
            "original_index_member": "audit/parent-verification-index.original.json",
        },
        "excluded": EXCLUDED,
        "limitations": [
            "fine-final 题集已曝光：仅 posthoc / exploratory。",
            "新的 35 题短盲确认集禁止进入训练或选点；原作者 36 题、1 题分歧 held 保存。",
            "盲审参考标签为代理一致性，不是真人 gold，不认证真人或教师准确率。",
            "本包是私有复现材料，非公开交付面；artifacts 保留原始绝对路径与原始 SHA。",
            "重训不保证在其它硬件/驱动/torch 版本上逐位复现；D5 重放为确定性后处理。",
        ],
    }


def collect_members() -> dict[str, tuple[bytes, str]]:
    members: dict[str, tuple[bytes, str]] = {}
    missing: list[str] = []

    def add(arcname: str, data: bytes, origin: str) -> None:
        if arcname in members:
            raise RuntimeError(f"成员重复: {arcname}")
        members[arcname] = (data, origin)

    def add_path(arcname: str, path: Path, origin: str) -> None:
        if not path.is_file():
            missing.append(str(path))
            return
        add(arcname, path.read_bytes(), origin)

    members["REPRODUCE.txt"] = (build_reproduce_txt().encode("utf-8"), "generated")
    members["PACKAGE.json"] = (
        json.dumps(build_package_json(), ensure_ascii=False, indent=2).encode("utf-8") + b"\n",
        "generated",
    )

    for repo_path, arcname in GIT_SOURCES:
        add(arcname, git_show(repo_path), f"git show {COMMIT}:{repo_path}")

    for name in DATA_FILES:
        add_path(f"data/{name}", RUN / "data" / name, "run data")
    add_path("d5/summary.json", RUN / "d5/summary.json", "run d5 summary")
    for name in D5_JSONL:
        add_path(f"d5/{name}", RUN / "d5" / name, "run d5")

    for name in ("eval-official", "eval-fine"):
        add_path(f"{name}/eval.json", RUN / name / "eval.json", "run eval")
        for pred in ("ocnli_dev.predictions.jsonl", "fine_final.predictions.jsonl"):
            add_path(f"{name}/{pred}", RUN / name / pred, "run eval predictions")

    for src, arcname in MODEL_FILES:
        add_path(arcname, RUN / "models" / src, "run model metadata")

    for name in ARTIFACT_FILES:
        add_path(f"artifacts/{name}", ART / name, "analysis artifacts")

    for src, arcname in EXT_BLIND:
        add_path(arcname, EXT / "blind36" / src, "external blind36 author/reviewer")
    for src, arcname in ROOT_BLIND:
        add_path(arcname, ART / "blind36" / src, "root blind36 audit")

    add_path("protocol/实验协议.json", EXT / "实验协议.json", "locked protocol")

    if missing:
        raise RuntimeError("原路径缺失（未做无界递归搜索）:\n" + "\n".join(missing))

                                     
    original_index_bytes = PARENT_INDEX.read_bytes()
    original_index_sha = sha256_bytes(original_index_bytes)
    add("audit/parent-verification-index.original.json", original_index_bytes, "original parent index bytes")

    index = json.loads(original_index_bytes)
    six = [model for model in index["models"] if model.get("arm") == "c5"]
    if len(six) != 6:
        raise RuntimeError(f"arm=c5 应为 6 项，实际 {len(six)}")

    portable_models = []
    for model in six:
        model_id = model["model_id"]
        arcname = f"inputs/d5-original/{model_id}.jsonl"
        source = model["files"]["predictions.jsonl"]
        path = Path(source["path"])
        if not path.is_file():
            missing.append(str(path))
            continue
        data = path.read_bytes()
        if len(data) != source["bytes"]:
            raise RuntimeError(f"{model_id} 原始输入字节数不符: {len(data)} != {source['bytes']}")
        if sha256_bytes(data) != source["sha256"]:
            raise RuntimeError(f"{model_id} 原始输入 SHA 不符")
        add(arcname, data, "original per-question predictions (parent index)")
        rewritten = json.loads(json.dumps(model, ensure_ascii=False))
        rewritten["files"] = {
            "predictions.jsonl": {"path": arcname, "bytes": source["bytes"], "sha256": source["sha256"]}
        }
        portable_models.append(rewritten)

    if missing:
        raise RuntimeError("原路径缺失（未做无界递归搜索）:\n" + "\n".join(missing))

    derived = {
        "schema": "c3-d5-derived-portable-index-v1",
        "status": "derived_for_portable_replay_not_a_new_audit",
        "derived_from_member": "audit/parent-verification-index.original.json",
        "original_index_sha256": original_index_sha,
        "original_index_schema": index.get("schema"),
        "original_index_status": index.get("status"),
        "selection": {"rule": "arm == 'c5'", "kept": len(portable_models), "dropped": len(index["models"]) - len(portable_models)},
        "path_rewrite": {
            "field": "models[].files['predictions.jsonl'].path",
            "from": "原始绝对主机路径",
            "to": "相对本包解压根目录的路径",
        },
        "retained_fields": [
            "model_id",
            "version",
            "arm",
            "seed",
            "correct",
            "accuracy",
            "macro_f1",
            "splits",
            "input_complete_roots",
            "truncation",
            "gold_counts",
            "files['predictions.jsonl'].bytes",
            "files['predictions.jsonl'].sha256",
        ],
        "dropped_fields": ["models[].files['metadata.json']", "models[].files['report.json']"],
        "note": "只做可移植路径派生；baseline accuracy/macro-F1 与原 source SHA 原样保留。不是对原审计的修改或替代。",
        "models": portable_models,
    }
    add(
        "d5/derived-portable-index.json",
        json.dumps(derived, ensure_ascii=False, indent=2).encode("utf-8") + b"\n",
        "derived from parent index",
    )

    return members


def sanity_checks(members: dict[str, tuple[bytes, str]]) -> dict:
    report = {}

    for name, meta in WEIGHTS.items():
        digest = sha256_file(meta["path"])
        size = meta["path"].stat().st_size
        if digest != meta["sha256"] or size != meta["bytes"]:
            raise RuntimeError(f"{name} 权重 SHA/字节不符: {digest} {size}")
        report[name] = {"bytes": size, "sha256": digest, "verified": True}

    data_manifest = json.loads(members["data/manifest.json"][0])
    for name, entry in data_manifest["files"].items():
        actual = sha256_bytes(members[f"data/{name}"][0])
        if actual != entry["sha256"]:
            raise RuntimeError(f"data/{name} 与 data/manifest.json 记录的 SHA 不符")

    for arm in ("official", "fine"):
        eval_json = json.loads(members[f"eval-{arm}/eval.json"][0])
        if eval_json["artifact"]["weights_sha256"] != WEIGHTS[arm]["sha256"]:
            raise RuntimeError(f"eval-{arm} 权重 SHA 与模型不符")
        if eval_json["data_manifest"]["sha256"] != sha256_bytes(members["data/manifest.json"][0]):
            raise RuntimeError(f"eval-{arm} data manifest SHA 不符")
        for cohort, payload in eval_json["cohorts"].items():
            pred_name = cohort + ".predictions.jsonl"
            member = f"eval-{arm}/{pred_name}"
            if member in members:
                if sha256_bytes(members[member][0]) != payload["predictions"]["sha256"]:
                    raise RuntimeError(f"{member} 与 eval.json 记录 SHA 不符")

    d5_summary = json.loads(members["d5/summary.json"][0])
    if d5_summary["index_sha256"] != sha256_bytes(members["audit/parent-verification-index.original.json"][0]):
        raise RuntimeError("d5/summary.json 的 index_sha256 与原父验收索引不符")
    report["d5_index_sha256"] = d5_summary["index_sha256"]

    blind = json.loads(members["blind36/evaluation/summary.json"][0])
    checks = [
        ("reviewer_sha256", blind["admission"]["reviewer_sha256"], "blind36/reviewer-primary.json"),
        ("authors_lock_sha256", blind["admission"]["authors_lock_sha256"], "blind36/authors-lock.json"),
        ("confirmed_sha256", blind["admission"]["confirmed_sha256"], "blind36/evaluation/confirmed.jsonl"),
        ("audit_sha256", blind["admission"]["audit_sha256"], "blind36/evaluation/reference-audit.json"),
        ("admission_lock_sha256", blind["admission_lock_sha256"], "blind36/evaluation/admission-lock.json"),
    ]
    for label, expected, member in checks:
        if sha256_bytes(members[member][0]) != expected:
            raise RuntimeError(f"blind36 {label} 与归档 SHA 不符")
    report["blind36"] = {
        "authored": blind["admission"]["authored"],
        "admitted": blind["admission"]["admitted"],
        "held": blind["admission"]["held"],
        "human_gold": blind["admission"]["human_gold"],
    }
    return report


def write_zip(members: dict[str, tuple[bytes, str]]) -> dict:
    rows = {name: (sha256_bytes(data), len(data)) for name, (data, _) in members.items()}

    def manifest_text(names) -> bytes:
        lines = [f"{rows[n][0]}  {rows[n][1]}  {n}\n" for n in sorted(names)]
        return "".join(lines).encode("utf-8")

    embedded = manifest_text(name for name in members if name != "MANIFEST.sha256")
    members["MANIFEST.sha256"] = (embedded, "generated")
    rows["MANIFEST.sha256"] = (sha256_bytes(embedded), len(embedded))

    external = manifest_text(members)
    MANIFEST_PATH.write_bytes(external)

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(members):
            zinfo = zipfile.ZipInfo(name, date_time=FIXED_DATE)
            zinfo.compress_type = zipfile.ZIP_DEFLATED
            zinfo.external_attr = 0o644 << 16
            archive.writestr(zinfo, members[name][0])

    return {
        "path": str(ZIP_PATH),
        "bytes": ZIP_PATH.stat().st_size,
        "sha256": sha256_file(ZIP_PATH),
        "members": len(members),
        "uncompressed_bytes": sum(size for _, size in rows.values()),
        "manifest": str(MANIFEST_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
    }


def parse_manifest(text: str) -> dict[str, tuple[str, int]]:
    table = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, size, name = line.split("  ", 2)
        table[name.strip()] = (digest, int(size.strip()))
    return table


DRIVER = '''\
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

target = Path("source/research/memory_relations/source_control.py").resolve()
spec = importlib.util.spec_from_file_location("sc_from_bundle", target)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
report = module.analyze_d5(Path("d5/derived-portable-index.json"), Path("replay-out"), samples=1000)
print(json.dumps({
    "module_file": str(Path(module.__file__).resolve()),
    "module_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    "index_sha256": report["index_sha256"],
    "new_training": report["new_training"],
    "new_model_inference": report["new_model_inference"],
    "bootstrap_samples": report["bootstrap_samples"],
    "model_ids": [model["model_id"] for model in report["models"]],
}, ensure_ascii=False))
'''


def verify_zip(archive_info: dict, members: dict[str, tuple[bytes, str]]) -> dict:
    started = time.time()
    table = parse_manifest(MANIFEST_PATH.read_text(encoding="utf-8"))
    problems: list[str] = []

    with zipfile.ZipFile(ZIP_PATH) as archive:
        bad = archive.testzip()
        if bad is not None:
            problems.append(f"testzip 命中损坏成员: {bad}")

        infos = archive.infolist()
        names = [info.filename for info in infos]
        if sorted(names) != sorted(table):
            problems.append("ZIP 成员与包外 manifest 不一致")
        if len(set(names)) != len(names):
            problems.append("ZIP 存在重复成员")

        member_rows = []
        for info in infos:
            data = archive.read(info)
            digest = sha256_bytes(data)
            expected = table.get(info.filename)
            ok = expected is not None and expected[0] == digest and expected[1] == len(data) == info.file_size
            if not ok:
                problems.append(f"成员校验失败: {info.filename}")
            member_rows.append({"path": info.filename, "bytes": len(data), "sha256": digest, "ok": ok})

        embedded = parse_manifest(archive.read("MANIFEST.sha256").decode("utf-8"))
        for name, (digest, size) in embedded.items():
            data = archive.read(name)
            if sha256_bytes(data) != digest or len(data) != size:
                problems.append(f"包内 MANIFEST 与成员不符: {name}")
        if "MANIFEST.sha256" in embedded:
            problems.append("包内 MANIFEST 不应包含自身")

        for repo_path, arcname in GIT_SOURCES:
            if archive.read(arcname) != git_show(repo_path):
                problems.append(f"源码字节与 git show 不一致: {arcname}")

        forbidden = [n for n in names if n.endswith((".pt", ".bin", ".safetensors", ".pth", ".ckpt", ".pyc"))]
        if forbidden:
            problems.append(f"包含禁止的权重/二进制: {forbidden}")

        tmp = OUT / f"_verify_tmp_{os.getpid()}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        replay_block = {}
        try:
            extract = ["source/research/memory_relations/source_control.py", "d5/derived-portable-index.json"]
            extract += [name for name in names if name.startswith("inputs/d5-original/")]
            for name in extract:
                target = tmp / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
            driver = tmp / "_replay_driver.py"
            driver.write_text(DRIVER, encoding="utf-8")

            env = os.environ.copy()
            env.update({
                "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONIOENCODING": "utf-8",
            })
            command = [str(PYEXE), "-B", str(driver)]
            result = subprocess.run(command, cwd=str(tmp), env=env, capture_output=True, text=True, encoding="utf-8")
            if result.returncode != 0:
                problems.append(f"重放返回码 {result.returncode}: {result.stderr[-2000:]}")
                replay_meta = None
            else:
                replay_meta = json.loads(result.stdout.strip().splitlines()[-1])

            comparison = {"all_equal": False, "models": []}
            if replay_meta is not None:
                archived = json.loads(archive.read("d5/summary.json").decode("utf-8"))
                replay = json.loads((tmp / "replay-out/summary.json").read_text(encoding="utf-8"))
                for key in ("schema", "analysis", "fusion", "argmax_tie_rule", "new_training",
                            "new_model_inference", "bootstrap_samples"):
                    if archived[key] != replay[key]:
                        problems.append(f"顶部字段不一致: {key}")
                if archived["index_sha256"] == replay["index_sha256"]:
                    problems.append("派生索引 index_sha256 不应等于原索引 SHA")

                by_id = {model["model_id"]: model for model in archived["models"]}
                if [m["model_id"] for m in replay["models"]] != [m["model_id"] for m in archived["models"]]:
                    problems.append("重放模型顺序/集合与归档不同")
                for model in replay["models"]:
                    model_id = model["model_id"]
                    old = by_id[model_id]
                    fields = {
                        "roots": model["roots"],
                        "clusters": model["clusters"],
                        "baseline_accuracy": model["baseline"]["accuracy"],
                        "baseline_macro_f1": model["baseline"]["macro_f1"],
                        "d5_accuracy": model["d5"]["accuracy"],
                        "d5_macro_f1": model["d5"]["macro_f1"],
                        "source_sha256": model["source_sha256"],
                        "predictions_sha256": model["predictions_sha256"],
                    }
                    expected = {
                        "roots": old["roots"],
                        "clusters": old["clusters"],
                        "baseline_accuracy": old["baseline"]["accuracy"],
                        "baseline_macro_f1": old["baseline"]["macro_f1"],
                        "d5_accuracy": old["d5"]["accuracy"],
                        "d5_macro_f1": old["d5"]["macro_f1"],
                        "source_sha256": old["source_sha256"],
                        "predictions_sha256": old["predictions_sha256"],
                    }
                    equal = fields == expected and model == old
                    if not equal:
                        problems.append(f"{model_id} 重放与归档不一致")
                    comparison["models"].append({
                        "model_id": model_id,
                        "roots": fields["roots"],
                        "clusters": fields["clusters"],
                        "baseline_accuracy": fields["baseline_accuracy"],
                        "baseline_macro_f1": fields["baseline_macro_f1"],
                        "d5_accuracy": fields["d5_accuracy"],
                        "d5_macro_f1": fields["d5_macro_f1"],
                        "source_sha256": fields["source_sha256"],
                        "predictions_sha256": fields["predictions_sha256"],
                        "archived_summary_equal": equal,
                    })
                comparison["all_equal"] = all(row["archived_summary_equal"] for row in comparison["models"])

            replay_block = {
                "command": " ".join(command),
                "cwd": str(tmp),
                "python": str(PYEXE),
                "exit_code": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr_tail": result.stderr.strip()[-1000:],
                "driver_sha256": sha256_bytes(driver.read_bytes()),
                "extracted_members": extract,
                "module_loaded_from_extracted_file": bool(
                    replay_meta and replay_meta["module_file"].endswith("source_control.py")
                    and "_verify_tmp_" in replay_meta["module_file"]
                ),
                "module_sha256": replay_meta["module_sha256"] if replay_meta else None,
                "replay_index_sha256": replay_meta["index_sha256"] if replay_meta else None,
                "derived_index_sha256": sha256_bytes((tmp / "d5/derived-portable-index.json").read_bytes()),
                "comparison": comparison,
                "seconds": round(time.time() - started, 2),
            }
            if replay_meta and replay_meta["new_training"] != 0:
                problems.append("重放报告 new_training 非 0")
            if replay_meta and replay_meta["new_model_inference"] != 0:
                problems.append("重放报告 new_model_inference 非 0")
        finally:
            shutil.rmtree(tmp, ignore_errors=False)

    if (OUT / f"_verify_tmp_{os.getpid()}").exists():
        problems.append("临时验证目录未能删除")

    return {
        "members_checked": len(member_rows),
        "member_rows": member_rows,
        "manifest_matches": len(member_rows) == len(table),
        "git_show_sources_match": not any("git show" in problem for problem in problems),
        "replay": replay_block,
        "problems": problems,
        "ok": not problems,
        "seconds": round(time.time() - started, 2),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if not PYEXE.is_file():
        raise RuntimeError(f"缺少 venv Python: {PYEXE}")

    members = collect_members()
    sanity = sanity_checks(members)
    archive_info = write_zip(members)
    verification = verify_zip(archive_info, members)

    receipt = {
        "schema": "c3-d5-reproduction-bundle-verify-receipt-v1",
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_commit": COMMIT,
        "archive": archive_info,
        "sanity": sanity,
        "weights_referenced_not_included": {
            name: {"bytes": meta["bytes"], "sha256": meta["sha256"]}
            for name, meta in WEIGHTS.items()
        },
        "manifest_members": verification["members_checked"],
        "manifest_matches_archive": verification["manifest_matches"],
        "git_show_sources_match": verification["git_show_sources_match"],
        "replay": verification["replay"],
        "problems": verification["problems"],
        "ok": verification["ok"],
        "excluded": EXCLUDED,
    }

    if not verification["ok"]:
        RECEIPT_PATH.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"ok": False, "problems": verification["problems"]}, ensure_ascii=False, indent=2))
        return 1

    RECEIPT_PATH.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "zip": archive_info,
        "members": verification["members_checked"],
        "replay_all_equal": verification["replay"]["comparison"]["all_equal"],
        "replay_exit_code": verification["replay"]["exit_code"],
        "receipt": str(RECEIPT_PATH),
        "receipt_sha256": sha256_file(RECEIPT_PATH),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
