                       
"""LTR 融合制品的**自动重训 → 转正 → 回滚**管线（攒量监控 / 候选训练 / 四门准入 / 版本化 / 审计）。

为什么（治本）：现役 `models/ltr/` 是 2026-09-14 训练并手工就位的一份制品——固定路径、
无版本指针、无注册表，"重训一次换上去"除了 git 历史之外没有任何机制支撑：既无法回答
「上次训练以来攒了多少新标注」（唯一入口是人工翻 `_ltr_metrics.json` 的时间戳），也无法回答
「候选凭什么比现役好」（heldout / OOD / 冻结门分散在三个目录）。本管线把这条流水线做成
可判定的机制：**攒量 → 候选 → 四门 → 转正 → 回滚**，每一步都留下机械可核验的账。

## 审计语义（谁是真源）

- **git 史只记转正**：`models/ltr/REGISTRY.json` 是 tracked 的版本真源（版本 / 训练口径 /
  数据指纹 / 四门指标 / 转正时间 / git sha / 状态），`models/ltr/_archive/<version>/` 是
  被替代制品的归档（`rollback` 的物理底本，属内部材料、不进公开交付）。「谁在什么时候把
  哪份制品放上了默认路径」只能从 git 追溯。
- **未跟踪流水记全部事件**：`.userdata/ltr_promotions.jsonl` 是 append-only 流水账，
  候选创建、每次门禁裁决（含**拒绝**）、转正、回滚各一行。「试了但没过」这种没有 git 痕迹的
  事件也留档；因为它天然是高频本机产物，不进版本库（与 `uploads_journal.jsonl` 同款分工）。
- **门禁只在候选目录里落账**：`_gates.json` 与四门原始指标同目录；`promote` 只读它，
  不接受口头结论——四门任一非绿即拒绝转正（fail-closed）。

## 四门（全过才放行；判据与阈值都是文件顶部常量）

1. **训练器导出自检**：候选由 `scripts/train_ltr.py` 通过自检（退出码 0、导出重载
   max|ΔnDCG@10| ≤ 1e-6，证据记在 `_candidate.json`），**且**门禁期用独立代码路径重载工件、
   在当前 heldout 上重算指标，须与制品声明的 `heldout_metrics` 一致（`SELFCHECK_TOL`）。
   前者证明"导出的那一刻是忠实的"，后者证明"此刻手里的字节还是那一份"。
2. **heldout 不退化**：重算 nDCG@10 ≥ 现役（同法重算）− `HELDOUT_NDCG10_TOL`。
3. **OOD 决斗不输**：在 OOD 交叉打分集的**只评测层**（`DEFAULT_OOD_REL`）上冻结对照候选
   vs 现役，**未舍入**的 macro nDCG@10 差值 ≥ `OOD_MARGIN_TOL`（复用
   `scripts/ood_duel.py` 的唯一实现，不另写一份）。评测层与训练 mix-in
   （`MIX_IN_PATTERNS`）物理分家——否则候选是在自己见过的行上被裁决。
4. **召回开启的 784**：官方冻结评测**结构性地不开召回**（`retriever.retrieve` 的
   `recall_backend` 默认 `"off"`、官方入口不传），而 LTR 融合只在召回路径里生效——
   也就是说 784 门对 LTR 是瞎的。本门用 `--recall dense`（`fusion="ltr"` 缺省）跑同一套
   54 题外部裁判，判据与冻结门一致：Top1/Top5 ≥ `FROZEN_TOP1/5` − `FROZEN_EPS`、
   硬违规 0、FASTQ 违规 0；**并断言 LTR 融合真的生效**（trace 的 `recall.status=="used"`
   且 `fusion=="ltr"`）——否则比的是回退路径，等于白门。

## 独立纪律（与 train_ltr.py 同）

数据评测只读 JSONL 与制品文件；文件事务复用项目的原子写与可重入跨进程锁：
- `gate` 的第四门需要真实检索管线（`dataset_recommender.retrieval`）才能跑召回路径，
  且需要 `scripts/evaluate_recommendation.py` 的外部裁判；检索依赖仅在该门内惰性 import，
  并临时改绑 `load_ltr_scorer` 指向候选制品（出厂路径零改动、退出即恢复）。
- `gate` 的第三门复用 `scripts/ood_duel.py`（公开工具，唯一实现）；缺失即该门
  fail-closed。

用法：
    python scripts/ltr_promote.py status                  # 攒量：距阈值还差多少
    python scripts/ltr_promote.py candidate               # 用当前数据训一份候选（暂存区）
    python scripts/ltr_promote.py gate  <candidate_dir>   # 四门裁决（无论过不过都落账）
    python scripts/ltr_promote.py promote <candidate_dir> # 归档现役 + 就位候选 + 更新 REGISTRY
    python scripts/ltr_promote.py rollback <version>      # 从归档恢复指定版本
    python scripts/ltr_promote.py history                 # REGISTRY + 未跟踪流水
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache, wraps
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT / "src"))
from dataset_recommender.app.runtime_paths import atomic_write_bytes


                           
                                                       
                                                             
                                                  
                                             
NEW_QUERIES_THRESHOLD = 100

                                                      
HELDOUT_NDCG10_TOL = 0.001

                                                     
                                                    
                                                           
OOD_MARGIN_TOL = 0.0

                                                             
                                                   
SELFCHECK_TOL = 1e-3

                                                                
                                                         
FROZEN_TOP1 = 97.7
FROZEN_TOP5 = 97.7
FROZEN_EPS = 0.05

                                                       
                                                     
                                                                  
                                                               
                                                       
                                             
MIX_IN_PATTERNS = (
    "research/ood_eval/synth_ood_trainable_*.jsonl",
    "research/ltr_data/synth_train_wave*_xscore.jsonl",
)

                                                                           
DEFAULT_OOD_REL = "research/ood_eval/synth_ood_evalonly_xscore.jsonl"

                                                      
                                                                  
                                                        
                                                                
                                                          
                                               
CANDIDATE_FEATURE_SET = "f3"

                                                                      
                                                                 
                                                                   
                                                 
                                             
F3_DATA_REL = "research/ltr_data/_exp_20260930_f3/data"

SPEC_NAME = "ltr_spec.json"
WEIGHTS_NAME = "ltr_mlp.pt"
ARTIFACT_NAMES = (WEIGHTS_NAME, SPEC_NAME)
REGISTRY_NAME = "REGISTRY.json"
CANDIDATE_META = "_candidate.json"
GATE_LEDGER = "_gates.json"
REQUIRED_GATES = frozenset({"trainer_export_selfcheck", "heldout_no_regression",
                            "ood_duel", "frozen_recall_784"})
JOURNAL_NAME = "ltr_promotions.jsonl"
REGISTRY_SCHEMA = 1

SELFCHECK_RE = re.compile(r"export self-check: max per-query \|delta nDCG@10\| = ([0-9.eE+-]+)")


class PromoteError(RuntimeError):
    """可预期的失败（缺件/口径不符/门禁未绿）——CLI 只报一句中文原因，不打印 traceback。"""


                                                                          
@dataclass(frozen=True)
class Paths:
    """一套管线路径。`root` 是项目根：测试用临时根即可整体隔离（不碰真实制品）。"""

    root: Path

    @property
    def ltr_dir(self) -> Path:
        return self.root / "models" / "ltr"

    @property
    def archive_dir(self) -> Path:
        return self.ltr_dir / "_archive"

    @property
    def registry_path(self) -> Path:
        return self.ltr_dir / REGISTRY_NAME

    @property
    def ltr_data_dir(self) -> Path:
        return self.root / "research" / "ltr_data"

    @property
    def candidates_root(self) -> Path:
        return self.ltr_data_dir / "_candidates"

    @property
    def journal_path(self) -> Path:
        return self.root / ".userdata" / JOURNAL_NAME

    @property
    def ood_path(self) -> Path:
        return self.root / DEFAULT_OOD_REL

    @property
    def f3_data_dir(self) -> Path:
        return self.root / F3_DATA_REL

    @property
    def f3_train_dir(self) -> Path:
        """f3 配方的预合并训练目录（基础层+OOD-trainable+wave2，构建器产出）。"""
        return self.f3_data_dir / "d1_f3"

    @property
    def f3_ood_path(self) -> Path:
        """门 3 的 f3 形态只评测集（f3_feats 叠加层版本）。"""
        return self.f3_data_dir / "eval" / "oodonly_f3.jsonl"

    def mix_in_paths(self) -> "list[Path]":
        """按 `MIX_IN_PATTERNS` 解析的 mix-in 文件（排序去重，缺失即空）。"""
        found: set[Path] = set()
        for pattern in MIX_IN_PATTERNS:
            found.update(self.root.glob(pattern))
        return sorted(p for p in found if p.is_file())


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha_map(paths: "list[Path]", root: Path) -> dict:
    """{项目相对路径: SHA-256}（路径排序，天然稳定）。"""
    out: dict[str, str] = {}
    for path in sorted(paths, key=lambda p: p.as_posix()):
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.as_posix()
        out[rel] = sha256_file(path)
    return out


def data_fingerprint(paths: "list[Path]", root: Path) -> str:
    """数据基准指纹：对（项目相对路径, 文件字节 SHA-256）序列求 SHA-256。

    口径 = "训练实际读了哪些字节"，故对"同一次训练重跑"稳定、对"任何一层增删/改写"敏感。
    用相对路径而非绝对路径，跨 worktree 可复现；版本号取它的前 8 位。
    """
    digest = hashlib.sha256()
    for rel, sha in sorted(file_sha_map(paths, root).items()):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload) -> None:
    """原子写 JSON（临时兄弟文件 + 替换）：中断不会留下半截 REGISTRY/门禁账。

    显式 `newline="\\n"`：本仓库 `.gitattributes` 约定文本签入 LF、工作树也是 LF，
    而 tracked 的 `REGISTRY.json` 每次转正都会被本函数重写——让它押着平台默认换行，
    就会在工作树里制造整文件 CRLF/LF 的假 diff。
    """
    atomic_write_bytes(path, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


@contextmanager
def _journal_critical_section(path: Path):
    """流水追加与制品事务的前像/恢复共锁；制品切换固定按模型 → 流水锁序。"""
    from dataset_recommender.corpus.fs_utils import file_critical_section

    with file_critical_section(path.with_name(path.name + ".lock")):
        yield


def append_journal(path: Path, entry: dict) -> None:
    """append-only 流水账：一行一个 JSON 对象。写不进去视作致命（不许静默丢审计）。"""
    with _journal_critical_section(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"ts": now_iso(), **entry}, ensure_ascii=False) + "\n")


def git_sha(root: Path) -> "str | None":
    try:
        raw = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return raw.stdout.strip() or None if raw.returncode == 0 else None


@lru_cache(maxsize=4)
def _train_ltr(root_str: str):
    """惰性 import `scripts/train_ltr.py`——它是训练口径（分层、特征序、成对构造）的唯一真源，
    本文件不另写第二份（分层合并、特征名、pair 计数一律复用它）。

    先查项目根下的 `scripts/`，再回落到本文件所在仓库：**代码位置由本文件决定、数据位置由
    `--project-root` 决定**，故测试用临时根（只放合成数据与制品）时照样复用真实现。
    """
    for base in (Path(root_str), AGENT_ROOT):
        if (base / "scripts" / "train_ltr.py").is_file():
            scripts = str(base / "scripts")
            if scripts not in sys.path:
                sys.path.insert(0, scripts)
            import train_ltr                 

            return train_ltr
    raise PromoteError("训练器 scripts/train_ltr.py 不可用——本管线的一切口径都以它为准")


@lru_cache(maxsize=4)
def _ood_duel():
    """惰性 import `scripts/ood_duel.py` 的冻结对照实现（唯一一份）。

    工具取自本文件所在仓库（代码位置由本文件决定，数据位置由 `--project-root` 决定，
    故测试用临时根时仍能拿到真实现 + 合成数据）。"""
    candidate = AGENT_ROOT / "scripts" / "ood_duel.py"
    if not candidate.is_file():
        raise PromoteError("工具 scripts/ood_duel.py 不可用（缺失即无法裁决 OOD 门）")
    root = str(AGENT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import scripts.ood_duel                 

    return scripts.ood_duel


                                                                          
def compact_stamp(iso: str) -> str:
    """`2026-09-14T00:23:54+08:00` → `20260914T002354`（版本号的稳定前缀）。"""
    cleaned = re.sub(r"[-:]", "", str(iso).strip())
    match = re.match(r"(\d{8})T(\d{6})", cleaned.replace("+", "T"))
    if not match:
        raise PromoteError(f"trained_at 不是可解析的 ISO 时间戳：{iso!r}")
    return f"{match.group(1)}T{match.group(2)}"


def version_of_spec(spec: dict, *, fingerprint: "str | None" = None) -> str:
    """版本 = 训练时间戳（紧凑 ISO）+ 数据指纹短哈希。

    首个条目（历史现役，2026-09-14 就位、当时尚无本管线）只有时间戳：它的训练基准指纹
    已无法从任何记录反推（当时的 OOD 层文件此后被扩充过），凭空补一个哈希等于伪造血缘。
    """
    stamp = compact_stamp(str(spec.get("trained_at") or ""))
    return f"{stamp}-{fingerprint[:8]}" if fingerprint else stamp


def load_spec(model_dir: Path) -> dict:
    path = model_dir / SPEC_NAME
    if not path.is_file():
        raise PromoteError(f"缺制品说明：{path}")
    spec = read_json(path)
    if not isinstance(spec, dict):
        raise PromoteError(f"制品说明不是 JSON 对象：{path}")
    return spec


def require_artifact_files(model_dir: Path) -> None:
    missing = [name for name in ARTIFACT_NAMES if not (model_dir / name).is_file()]
    if missing:
        raise PromoteError(f"制品不完整（缺 {', '.join(missing)}）：{model_dir}")


                                                                      
def load_registry(path: Path) -> dict:
    if not path.is_file():
        return {"schema_version": REGISTRY_SCHEMA, "model": "ltr_mlp", "active": None, "versions": []}
    registry = read_json(path)
    if not isinstance(registry, dict) or not isinstance(registry.get("versions"), list):
        raise PromoteError(f"REGISTRY 结构不符（需 {{versions: [...]}}）：{path}")
    return registry


def registry_entry(registry: dict, version: str) -> "dict | None":
    return next((e for e in registry["versions"] if e.get("version") == version), None)


def active_entry(registry: dict) -> "dict | None":
    version = registry.get("active")
    return registry_entry(registry, version) if version else None


def upsert_registry_entry(registry: dict, entry: dict) -> None:
    existing = registry_entry(registry, str(entry["version"]))
    if existing is None:
        registry["versions"].append(entry)
    else:
        existing.update(entry)


def save_registry(path: Path, registry: dict) -> None:
    registry["versions"] = sorted(registry["versions"], key=lambda e: str(e.get("trained_at") or ""))
    write_json_atomic(path, registry)


def entry_from_spec(spec: dict, *, version: str, fingerprint: "str | None", status: str,
                    artifact_shas: "dict | None" = None, **extra) -> dict:
    entry = {
        "version": version,
        "trained_at": spec.get("trained_at"),
        "train_counts": spec.get("train_counts"),
        "data_fingerprint": fingerprint,
        "label_mode": spec.get("label_mode"),
        "seed": spec.get("seed"),
        "heldout_metrics": spec.get("heldout_metrics"),
        "artifact_sha256": artifact_shas,
        "status": status,
    }
    entry.update(extra)
    return entry


                                                                             
def basis_report(paths: Paths, in_dir: Path, *, seed: int, label: str,
                 mix_ins: "list[Path] | None" = None, feature_set: str = "f0") -> dict:
    """当前训练基准的实测量：分层行数、合并查询数、成对标签数、文件指纹。

    指纹只覆盖**真正被读到的字节**：`load_split` 实际命中的层文件 + mix-in 文件。
    同目录下与训练无关的一次性产物（如 `synth_train_xscore.jsonl` 这类实验导出）不参与，
    否则改一个没人读的文件就会凭空产生"新版本"。全量清单另在 `files` 里如实列出。

    ``mix_ins`` 是**本次训练实际并入的层**（`candidate --mix-in` 可以覆盖缺省层）；
    传 None 才回落到 `MIX_IN_PATTERNS` 的解析结果（`status` 的攒量口径用这一支）。
    必须跟实际训练走：否则两份训练基准不同的候选会算出同一个指纹、进而共用同一个版本号，
    `promote` 的"同版本拒绝"与"指纹变了才需要重训"这两条判断同时失效。
    ``feature_set`` 同理：训练配方（f0/f3）是基准的一部分，必须跟训练命令一致。
    """
    import numpy as np                 

    train_ltr = _train_ltr(str(paths.root))
    train_rows, train_src = train_ltr.load_split(in_dir, "train")
    held_rows, held_src = train_ltr.load_split(in_dir, "heldout")
    mix_ins = list(mix_ins) if mix_ins is not None else paths.mix_in_paths()
    mix_stats: dict = {}
    if mix_ins:
        train_rows, mix_stats = train_ltr.merge_mix_in(train_rows, mix_ins)
    X, y, groups, label_stats = train_ltr.build_training_matrix(train_rows, label, feature_set)
    pairs = 0
    if X.shape[0]:
        left, _right, _sign = train_ltr._pair_indices(y, groups, np.random.default_rng(seed))
        pairs = int(len(left))

    train_files = [in_dir / name for name in train_src["layers"]]
    held_files = [in_dir / name for name in held_src["layers"]]
    listed = sorted(p for p in in_dir.glob("*.jsonl") if p.is_file())
    return {
        "in_dir": str(in_dir),
        "label_mode": label,
        "feature_set": feature_set,
        "seed": seed,
        "train_layers": train_src["layers"],
        "heldout_layers": held_src["layers"],
        "mix_in": {p.relative_to(paths.root).as_posix(): mix_stats.get(p.name, {}) for p in mix_ins},
        "train_queries": len(train_rows),
        "heldout_queries": len(held_rows),
        "train_pairs": pairs,
        "labelled_candidates": label_stats["labelled_candidates"],
        "data_fingerprint": {
            "train": data_fingerprint(train_files + mix_ins, paths.root),
            "heldout": data_fingerprint(held_files, paths.root),
        },
        "files": [
            {
                "path": p.relative_to(paths.root).as_posix(),
                "rows": sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip()),
                "read_by_trainer": p in train_files or p in held_files or p in mix_ins,
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
            }
            for p in listed
        ],
    }


def status_report(paths: Paths, in_dir: Path, *, seed: int, label: str) -> dict:
    basis = basis_report(paths, in_dir, seed=seed, label=label)
    has_incumbent = (paths.ltr_dir / SPEC_NAME).is_file()
    incumbent_spec = load_spec(paths.ltr_dir) if has_incumbent else None
    registry = load_registry(paths.registry_path)
    reg_active = active_entry(registry)
    trained_at = (incumbent_spec or {}).get("trained_at")
    counts = (incumbent_spec or {}).get("train_counts") or {}
    new_queries = basis["train_queries"] - int(counts.get("queries") or 0) if has_incumbent else None
    new_pairs = basis["train_pairs"] - int(counts.get("pairs") or 0) if has_incumbent else None
    reg_fp = (reg_active or {}).get("data_fingerprint")
    return {
        "incumbent": {
            "version": (reg_active or {}).get("version")
            or (version_of_spec(incumbent_spec) if incumbent_spec else None),
            "trained_at": trained_at,
            "train_counts": counts or None,
        },
        "basis": basis,
        "delta": {"queries": new_queries, "pairs": new_pairs},
        "threshold": {
            "new_queries": NEW_QUERIES_THRESHOLD,
            "gap": None if new_queries is None else max(0, NEW_QUERIES_THRESHOLD - new_queries),
            "ready": bool(new_queries is not None and new_queries >= NEW_QUERIES_THRESHOLD),
        },
        "fingerprint": {
            "train": basis["data_fingerprint"]["train"],
            "registry_active": reg_fp,
            "changed_since_registry": (None if reg_fp is None else reg_fp != basis["data_fingerprint"]["train"]),
        },
        "files_changed_since_training": [
            row["path"] for row in basis["files"]
            if trained_at and row["read_by_trainer"] and row["mtime"] > trained_at
        ],
    }


def _print_status(report: dict) -> None:
    basis, inc, thr = report["basis"], report["incumbent"], report["threshold"]
    print(f"现役版本：{inc['version']}（trained_at {inc['trained_at']}）")
    print(f"  训练口径：{inc['train_counts']}")
    print(f"当前基准：train {basis['train_queries']} 查询 / {basis['train_pairs']} 对 · "
          f"heldout {basis['heldout_queries']} 查询 · label={basis['label_mode']} seed={basis['seed']}")
    print(f"  分层：train={basis['train_layers']} heldout={basis['heldout_layers']}")
    for name, stats in basis["mix_in"].items():
        print(f"  mix-in {name}: {stats}")
    delta = report["delta"]
    if delta["queries"] is None:
        print("  ! 无现役制品说明，无法计算增量")
    else:
        print(f"上次训练以来新增：{delta['queries']} 查询 / {delta['pairs']} 对")
    print(f"阈值 NEW_QUERIES_THRESHOLD={thr['new_queries']}："
          + ("已达阈值（可以开候选）" if thr["ready"] else f"还差 {thr['gap']} 查询"))
    reg_fp = report["fingerprint"]["registry_active"]
    print(f"数据指纹（train）：{report['fingerprint']['train'][:16]}…   "
          f"REGISTRY 现役指纹：{(reg_fp or '未记录')[:16] if reg_fp else '未记录'}")
    if report["files_changed_since_training"]:
        touched = report["files_changed_since_training"]
        shown = touched if len(touched) <= 4 else touched[:4] + [f"…共 {len(touched)} 个"]
        print(f"  ! mtime 晚于训练时刻、且训练会读的文件：{shown}")
        print("    （辅助信号：fresh checkout 会把 mtime 整体刷新成签到时间，权威判据是计数差与指纹）")
    print(f"verdict: {'READY' if thr['ready'] else 'NOT_READY'}")


def cmd_status(args) -> int:
    paths = Paths(args.project_root.resolve())
    report = status_report(paths, args.in_dir.resolve(), seed=args.seed, label=args.label)
    _print_status(report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_ready and not report["threshold"]["ready"]:
        return 1
    return 0


def _ensure_f3_data(paths: "Paths", python: str) -> None:
    """f3 特征层缺失时跑构建器（幂等约 1 分钟；构建器失败/契约漂移一律 fail-closed）。"""
    builder = paths.ltr_data_dir / "build_f3_features.py"
    print(f"f3 特征层缺失，先跑构建器 {builder.name}（约 1 分钟，幂等）…")
    completed = subprocess.run([python, str(builder)], cwd=str(paths.root))
    if completed.returncode != 0:
        raise PromoteError(f"f3 特征层构建失败（退出码 {completed.returncode}）——先看构建器输出")
    if not (paths.f3_train_dir / "synth_train.jsonl").is_file():
        raise PromoteError(f"构建器跑完但缺 {paths.f3_train_dir / 'synth_train.jsonl'}——构建器契约漂移")


                                                                                
def train_command(python: str, root: Path, in_dir: Path, out_dir: Path, *,
                  label: str, seed: int, mix_ins: "list[Path]",
                  feature_set: str = "f0") -> "list[str]":
    """候选训练命令（纯函数：便于测试与人工复核；正式执行见 `cmd_candidate`）。

    `--deterministic` 是**固定带上**的：门禁的容差（heldout 1e-3 / OOD 0.0）与 CPU
    归约顺序造成的重训抖动（实测 heldout nDCG@10 样本 SD ≈ 5.7e-4）同量级，不给训练
    子进程钉住数值环境，一次转正裁决就是在噪声里掷硬币。`--feature-set`/`--eval-feature-set`
    显式写进命令行（训练/评测同一配方；审计链上看得见，不靠缺省猜）。
    """
    command = [python, str(root / "scripts" / "train_ltr.py"),
               "--label", label, "--seed", str(seed),
               "--deterministic",
               "--feature-set", feature_set, "--eval-feature-set", feature_set,
               "--in-dir", str(in_dir),
               "--export", str(out_dir),
               "--out", str(out_dir / "_ltr_metrics.json")]
    for path in mix_ins:
        command += ["--mix-in", str(path)]
    return command


def cmd_candidate(args) -> int:
    paths = Paths(args.project_root.resolve())
    feature_set = args.feature_set
    if feature_set == "f3":
                                                                          
                                                                    
        if args.in_dir is None and not (paths.f3_train_dir / "synth_train.jsonl").is_file():
            if args.dry_run:
                print("（f3 特征层缺失：正式执行会先跑 build_f3_features.py 构建，约 1 分钟）")
            else:
                _ensure_f3_data(paths, args.python)
    in_dir = (args.in_dir or (paths.f3_train_dir if feature_set == "f3" else paths.ltr_data_dir)
              ).resolve()
    stamp = args.ts or datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    out_dir = (args.out_dir or (paths.candidates_root / stamp)).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise PromoteError(f"候选目录已存在且非空（拒绝覆盖，避免污染审计链）：{out_dir}")
    if args.mix_in:
        mix_ins = [p.resolve() for p in args.mix_in]
    elif feature_set == "f3" or args.no_default_mix_in:
                                                             
        mix_ins = []
    else:
        mix_ins = paths.mix_in_paths()
    command = train_command(args.python, paths.root, in_dir, out_dir, label=args.label,
                            seed=args.seed, mix_ins=mix_ins, feature_set=feature_set)
    print("候选训练命令：")
    print("  " + subprocess.list2cmdline(command))
    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone()
                                                    
                                    
    stdout_path = out_dir / "_train_stdout.txt"
    with stdout_path.open("w", encoding="utf-8", errors="replace") as sink:
        completed = subprocess.run(command, cwd=str(paths.root), stdout=sink,
                                   stderr=subprocess.STDOUT)
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    print(stdout.rstrip())
    if completed.returncode != 0:
        append_journal(paths.journal_path, {"action": "candidate_failed", "candidate_dir": str(out_dir),
                                            "exit_code": completed.returncode, "command": command})
        raise PromoteError(f"训练失败（退出码 {completed.returncode}），原始输出：{stdout_path}")
    require_artifact_files(out_dir)

    match = SELFCHECK_RE.search(stdout)
    spec = load_spec(out_dir)
    basis = basis_report(paths, in_dir, seed=args.seed, label=args.label, mix_ins=mix_ins,
                         feature_set=feature_set)
    metrics_path = out_dir / "_ltr_metrics.json"
    metrics_payload = read_json(metrics_path) if metrics_path.is_file() else {}
    meta = {
        "schema_version": 1,
        "version": version_of_spec(spec, fingerprint=basis["data_fingerprint"]["train"]),
        "created_at": now_iso(),
        "trained_at": spec.get("trained_at"),
        "label_mode": args.label,
        "feature_set": feature_set,
        "seed": args.seed,
        "in_dir": str(in_dir),
        "mix_in": [p.relative_to(paths.root).as_posix() for p in mix_ins],
        "train_command": command,
        "train_exit_code": completed.returncode,
        "train_seconds": round((datetime.now().astimezone() - started).total_seconds(), 2),
        "selfcheck_max_delta": float(match.group(1)) if match else None,
        "data_fingerprint": basis["data_fingerprint"],
        "basis": {"train_queries": basis["train_queries"], "train_pairs": basis["train_pairs"],
                  "heldout_queries": basis["heldout_queries"]},
        "train_counts": spec.get("train_counts"),
        "heldout_metrics": spec.get("heldout_metrics"),
        "artifact_sha256": file_sha_map([out_dir / name for name in ARTIFACT_NAMES], out_dir),
        "trainer_metrics_generated_at": metrics_payload.get("generated_at"),
        "git_sha": git_sha(paths.root),
    }
    write_json_atomic(out_dir / CANDIDATE_META, meta)
    append_journal(paths.journal_path, {
        "action": "candidate", "version": meta["version"], "candidate_dir": str(out_dir),
        "data_fingerprint": basis["data_fingerprint"]["train"], "train_counts": spec.get("train_counts"),
        "selfcheck_max_delta": meta["selfcheck_max_delta"], "git_sha": meta["git_sha"],
    })
    print(f"\n候选已就位：{out_dir}")
    print(f"  version={meta['version']}  train={spec.get('train_counts')}  "
          f"heldout={spec.get('heldout_metrics')}  自检 Δ={meta['selfcheck_max_delta']}")
    print(f"  下一步：python scripts/ltr_promote.py gate {out_dir}")
    return 0


                                                                                      
def _independent_scorer(model_dir: Path, repo_root: Path):
    """与 `vector_recall.load_ltr_scorer` **独立**的一份最小重载。

    门禁不许和被测件共用实现：这里只读 spec 的 feature_order/mean/std + state_dict，
    按 spec 声明的特征名重新组织列序。spec 出现训练侧没有的特征名 → 响亮报错（契约漂移），
    而不是像生产路径那样 fail-soft 回退。
    """
    import numpy as np                 
    import torch                 

    train_ltr = _train_ltr(str(repo_root))
    spec = load_spec(model_dir)
    names = [str(name) for name in spec["feature_order"]]
    unknown = [name for name in names if name not in train_ltr.FEATURE_NAMES]
    if unknown:
        raise PromoteError(f"制品特征名不在训练侧契约里（推理会静默回退）：{unknown}")
    mean = np.asarray(spec["feature_mean"], dtype=np.float32)
    std = np.asarray(spec["feature_std"], dtype=np.float32)
    net = train_ltr._build_mlp(int(spec["model"]["input_dim"]))
    net.load_state_dict(torch.load(model_dir / WEIGHTS_NAME, map_location="cpu", weights_only=True))
    net.eval()
    index = [train_ltr.FEATURE_NAMES.index(name) for name in names]

    def _score(matrix) -> "np.ndarray":
        """返回 **numpy 数组**（不是 list）：`train_ltr.model_order` 会对分数取负，
        list 会当场 TypeError——与 train_ltr 自家打分器同形是第一要件。"""
        arr = np.asarray(matrix, dtype=np.float32).reshape(-1, len(train_ltr.FEATURE_NAMES))
        scaled = ((arr[:, index] - mean) / std).astype(np.float32)
        with torch.no_grad():
            return net(torch.from_numpy(scaled)).squeeze(1).numpy()

    return _score, spec


def heldout_metrics_of(model_dir: Path, in_dir: Path, *, mode: str, repo_root: Path,
                       feature_set: str = "f0") -> dict:
    """用独立重载在**当前** heldout 上重算指标（门 1 的一致性 + 门 2 的比较都吃这一份）。

    ``feature_set`` = 候选的评测特征口径（门从候选 meta 继承）；现役制品也用同一口径重算——
    R1 之后生产推理永远用共享函数（f3 语义）组特征，「在生产里谁更好」只能在生产口径下比。
    """
    train_ltr = _train_ltr(str(repo_root))
    rows, info = train_ltr.load_split(in_dir, "heldout")
    units = train_ltr.eval_units(rows, mode, feature_set)
    scorer, spec = _independent_scorer(model_dir, repo_root)
    return {"metrics": train_ltr.evaluate(units, train_ltr.model_order(scorer)),
            "declared": spec.get("heldout_metrics"), "spec": spec,
            "heldout_queries": len(rows), "ranked_queries": len(units), "layers": info["layers"]}


def gate_selfcheck(meta: dict, reload_or_error) -> dict:
    """门 1：训练器导出自检（记录 + 独立重算一致性）。"""
    recorded = {"exit_code": meta.get("train_exit_code"), "max_delta": meta.get("selfcheck_max_delta")}
    recorded_ok = (recorded["exit_code"] == 0 and recorded["max_delta"] is not None
                   and float(recorded["max_delta"]) <= 1e-6)
    if isinstance(reload_or_error, str):
        return {"gate": "trainer_export_selfcheck", "passed": False, "reason": reload_or_error,
                "recorded_selfcheck": {**recorded, "ok": recorded_ok}}
    reloaded = reload_or_error
    spec = reloaded["spec"]
    structural: list[str] = []
    if int(spec["model"]["input_dim"]) != len(spec["feature_order"]):
        structural.append("model.input_dim 与 feature_order 长度不符")
    if not (len(spec["feature_mean"]) == len(spec["feature_std"]) == len(spec["feature_order"])):
        structural.append("feature_mean/std 与 feature_order 长度不符")
    if not spec["model"].get("hidden_layers"):
        structural.append("model.hidden_layers 为空")
    declared = reloaded["declared"] or {}
    per_metric = {
        key: abs(float(reloaded["metrics"][key]) - float(declared[key]))
        for key in ("nDCG@5", "nDCG@10", "MRR")
        if declared.get(key) is not None and reloaded["metrics"].get(key) is not None
    }
    max_delta = max(per_metric.values()) if per_metric else None
    consistent = max_delta is not None and max_delta <= SELFCHECK_TOL
    heldout_changed = meta.get("data_fingerprint", {}).get("heldout")
    return {
        "gate": "trainer_export_selfcheck",
        "passed": bool(recorded_ok and consistent and not structural),
        "tolerance": SELFCHECK_TOL,
        "recorded_selfcheck": {**recorded, "ok": recorded_ok},
        "reloaded_vs_declared": {"max_delta": max_delta, "per_metric": per_metric,
                                 "reloaded": reloaded["metrics"], "declared": declared},
        "structural_errors": structural,
        "heldout_queries": reloaded["heldout_queries"],
        "heldout_split_fingerprint_at_training": heldout_changed,
    }


def gate_heldout(candidate_metrics: dict, incumbent_dir: Path, in_dir: Path, *, mode: str,
                 repo_root: Path, blocked_reason: "str | None" = None,
                 feature_set: str = "f0") -> dict:
    """门 2：heldout nDCG@10 不退化。两边都用同一份代码、同一个当前 heldout 重算。"""
    if blocked_reason:
        return {"gate": "heldout_no_regression", "passed": False,
                "reason": f"heldout 指标不可重算：{blocked_reason}"}
    if not (incumbent_dir / SPEC_NAME).is_file():
        return {"gate": "heldout_no_regression", "passed": False, "reason": "现役制品说明缺失，无基线可比"}
    incumbent = heldout_metrics_of(incumbent_dir, in_dir, mode=mode, repo_root=repo_root,
                                   feature_set=feature_set)
    cand10 = candidate_metrics.get("nDCG@10")
    base10 = incumbent["metrics"].get("nDCG@10")
    if cand10 is None or base10 is None:
        return {"gate": "heldout_no_regression", "passed": False,
                "reason": "nDCG@10 缺失（heldout 无可排序查询）"}
    delta = float(cand10) - float(base10)
    return {
        "gate": "heldout_no_regression",
        "passed": delta >= -HELDOUT_NDCG10_TOL,
        "tolerance": HELDOUT_NDCG10_TOL,
        "candidate": candidate_metrics,
        "incumbent_reloaded": incumbent["metrics"],
        "incumbent_declared": incumbent["declared"],
        "delta_nDCG@10": round(delta, 6),
    }


                                                                             
def gate_ood(candidate_dir: Path, incumbent_dir: Path, ood_path: Path, *,
             feature_set: str = "f0") -> dict:
    """门 3：OOD 交叉打分集上冻结对照候选 vs 现役（复用 ood_duel 的唯一实现）。

    ``feature_set`` 随候选 meta：f3 候选用 f3 形态只评测集（生产存活池回放）+ f3 特征——
    现役制品同样按 f3 评测（生产推理侧在 R1 之后就是这么组特征的）。
    """
    if not ood_path.is_file():
        return {"gate": "ood_duel", "passed": False, "reason": f"OOD 数据缺失（内部依赖）：{ood_path}"}
    try:
        ood_duel = _ood_duel()
        result = ood_duel.compare_frozen(candidate_dir=candidate_dir, incumbent_dir=incumbent_dir,
                                        ood_path=ood_path, feature_set=feature_set)
    except PromoteError:
        raise
    except Exception as exc:                                            
        return {"gate": "ood_duel", "passed": False, "reason": f"冻结对照执行失败：{exc!r}"}
    verdict = result["verdict"]
                                                              
                                                               
                                                  
    if "margin_raw" not in verdict:
        return {"gate": "ood_duel", "passed": False,
                "reason": "冻结对照未返回未舍入 margin（ood_duel.compare_frozen 契约漂移）",
                "verdict": verdict}
    margin = float(verdict["margin_raw"])
    return {
        "gate": "ood_duel",
        "passed": margin >= -OOD_MARGIN_TOL,
        "tolerance": OOD_MARGIN_TOL,
        "n_ood": result["n_ood"],
        "overall10": result["overall10"],
        "per_cls10": result.get("per_cls10"),
        "verdict": verdict,
    }


                                                                                
def gate_frozen_recall(candidate_dir: Path, *, report_path: "Path | None" = None, top_k: int = 5) -> dict:
    """门 4：召回开启 + LTR 融合的官方冻结集裁判。

    默认在进程内跑：惰性 import 真实检索管线与官方外部裁判，临时把 `load_ltr_scorer`
    改绑到候选制品（出厂路径与 REGISTRY 零改动、退出即恢复），并要求 trace 证明
    LTR 融合真的生效（`status=="used"` 且 `fusion=="ltr"`）。
    `report_path` 给定时改为消费一份**已跑好的**同形报告（测试与人工复跑用），
    此时跳过生效性断言（该报告来自一次性真跑，命令与关键行记在 docs/QUALITY_GATES.md）。
    """
    if report_path is not None:
        if not report_path.is_file():
            return {"gate": "frozen_recall_784", "passed": False, "reason": f"冻结报告缺失：{report_path}"}
        report = read_json(report_path)
        engagement = {"source": "report", "verified": None}
    else:
        try:
            report, engagement = _run_frozen_recall(candidate_dir, top_k=top_k)
        except PromoteError as exc:
            return {"gate": "frozen_recall_784", "passed": False, "reason": str(exc)}
    metrics = (report or {}).get("metrics") or {}
    try:
        top1 = float(metrics.get("Top1_Accuracy_%"))
        top5 = float(metrics.get("Top5_Hit_Rate_%"))
        violation = float(metrics.get("Constraint_Violation_%"))
        fastq = float(metrics.get("FASTQ_Violation_%"))
    except (TypeError, ValueError):
        return {"gate": "frozen_recall_784", "passed": False,
                "reason": "报告指标缺失或非数值", "metrics": metrics}
    checks = {
        "top1": top1 >= FROZEN_TOP1 - FROZEN_EPS,
        "top5": top5 >= FROZEN_TOP5 - FROZEN_EPS,
        "violation": violation <= 0.0,
        "fastq": fastq <= 0.0,
        "ltr_engaged": engagement.get("verified") is not False,
    }
    return {
        "gate": "frozen_recall_784",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {"top1": FROZEN_TOP1, "top5": FROZEN_TOP5, "eps": FROZEN_EPS},
        "metrics": metrics,
        "n_queries": metrics.get("n_queries"),
        "ltr_engagement": engagement,
    }


def _run_frozen_recall(candidate_dir: Path, *, top_k: int) -> "tuple[dict, dict]":
    """在进程内跑召回开启的冻结裁判，scorer 临时改绑到候选。"""
    for extra in (str(AGENT_ROOT / "src"), str(AGENT_ROOT / "scripts")):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    from dataset_recommender.retrieval import vector_recall                 

    import evaluate_recommendation as ev                 

    scorer = vector_recall.load_ltr_scorer(candidate_dir)
    if scorer is None:
        raise PromoteError("候选制品无法被生产加载器载入（契约不完整）——门 4 无从裁决")
    if vector_recall.load_embedder() is None:
        raise PromoteError(
            "本地嵌入模型不可用（models/embeddings/…）——召回会静默回退，本门此时比的是回退路径、"
            "等于白门，故 fail-closed。先跑 scripts/fetch_embedding_model.py 补模型。"
        )
    original = vector_recall.load_ltr_scorer
    vector_recall.load_ltr_scorer = lambda *a, **k: scorer                            
    try:
        settings, records = ev.load_pipeline()
        queries = ev._load_queries(str(ev.DEFAULT_QUERIES))
        report = ev.evaluate(queries, records, settings, top_k=top_k, recall_backend="dense")
                                                    
                                                   
        engagement = _probe_ltr_engagement(records, queries, settings, top_k=top_k)
    finally:
        vector_recall.load_ltr_scorer = original                            
    return report, engagement


def _probe_ltr_engagement(records, queries, settings, *, top_k: int) -> dict:
    """单查询探针：证明召回路径真的走 `fusion="ltr"`，而不是 fail-soft 回退。"""
    from dataset_recommender.retrieval.query_parser import parse_query                 
    from dataset_recommender.retrieval.retriever import DatasetRetriever                 

    for query in queries:
        intent = parse_query(query["query"], settings.keyword_mapping)
        if getattr(intent, "abstain", False):
            continue
        trace: dict = {}
        DatasetRetriever(top_k=top_k).retrieve(records, intent, top_k=top_k,
                                               recall_backend="dense", execution_trace=trace)
        recall = trace.get("recall") or {}
        if recall.get("backend") == "dense":
            return {"source": "probe", "query": query.get("id"), "trace": recall,
                    "verified": recall.get("status") == "used" and recall.get("fusion") == "ltr"}
    return {"source": "probe", "verified": False, "reason": "没有可探测的非弃权查询"}


                                                                           
def cmd_gate(args) -> int:
    paths = Paths(args.project_root.resolve())
    candidate_dir = args.candidate_dir.resolve()
    require_artifact_files(candidate_dir)
    meta_path = candidate_dir / CANDIDATE_META
    if not meta_path.is_file():
        raise PromoteError(f"候选未经本管线暂存（缺 {CANDIDATE_META}）：{candidate_dir}")
    meta = read_json(meta_path)
    candidate_hashes = file_sha_map([candidate_dir / n for n in ARTIFACT_NAMES], candidate_dir)
    incumbent_hashes = file_sha_map([paths.ltr_dir / n for n in ARTIFACT_NAMES
                                     if (paths.ltr_dir / n).is_file()], paths.ltr_dir)
    if not isinstance(meta, dict):
        raise PromoteError(f"{CANDIDATE_META} 不是 JSON 对象：{meta_path}")
    mode = str(meta.get("label_mode") or "mix")
                                                 
    feature_set = str(meta.get("feature_set") or "f0")
                                                   
                               
    in_dir = Path(meta.get("in_dir") or args.in_dir or paths.ltr_data_dir).resolve()

    try:
        reloaded = heldout_metrics_of(candidate_dir, in_dir, mode=mode, repo_root=paths.root,
                                      feature_set=feature_set)
        reload_error = None
    except PromoteError as exc:
        reloaded, reload_error = None, str(exc)

    gate1 = gate_selfcheck(meta, reload_error if reload_error else reloaded)
    gate2 = gate_heldout((reloaded or {}).get("metrics") or {}, paths.ltr_dir, in_dir, mode=mode,
                         repo_root=paths.root, blocked_reason=reload_error, feature_set=feature_set)
                                                      
    ood_path = (args.ood_path.resolve() if args.ood_path
                else (paths.f3_ood_path if feature_set == "f3" else paths.ood_path))
    gate3 = ({"gate": "ood_duel", "passed": False, "reason": "--skip-ood 指定：本门未跑"}
             if args.skip_ood else gate_ood(candidate_dir, paths.ltr_dir, ood_path,
                                            feature_set=feature_set))
    gate4 = ({"gate": "frozen_recall_784", "passed": False, "reason": "--skip-frozen 指定：本门未跑"}
             if args.skip_frozen else gate_frozen_recall(
                 candidate_dir, report_path=(args.frozen_report.resolve() if args.frozen_report else None)))

    if (candidate_hashes != file_sha_map([candidate_dir / n for n in ARTIFACT_NAMES], candidate_dir)
            or incumbent_hashes != file_sha_map([paths.ltr_dir / n for n in ARTIFACT_NAMES
                                                 if (paths.ltr_dir / n).is_file()], paths.ltr_dir)):
        gate1.update(passed=False, reason="评测期间制品变化，需重新验收")
    gates = {item["gate"]: item for item in (gate1, gate2, gate3, gate4)}
    passed = all(item.get("passed") for item in gates.values())
    ledger = {
        "schema_version": 1,
        "version": meta.get("version"),
        "candidate_dir": str(candidate_dir),
        "gated_at": now_iso(),
        "artifact_sha256": candidate_hashes,
        "incumbent_sha256": incumbent_hashes,
        "verdict": "ACCEPT" if passed else "HOLD",
        "thresholds": {
            "selfcheck_tolerance": SELFCHECK_TOL,
            "heldout_ndcg10_tolerance": HELDOUT_NDCG10_TOL,
            "ood_margin_tolerance": OOD_MARGIN_TOL,
            "frozen_top1": FROZEN_TOP1, "frozen_top5": FROZEN_TOP5, "frozen_eps": FROZEN_EPS,
        },
        "gates": gates,
        "git_sha": git_sha(paths.root),
    }
    write_json_atomic(candidate_dir / GATE_LEDGER, ledger)
    append_journal(paths.journal_path, {
        "action": "gate" if passed else "reject", "version": meta.get("version"),
        "candidate_dir": str(candidate_dir), "verdict": ledger["verdict"],
        "gates": {name: bool(item.get("passed")) for name, item in gates.items()},
        "git_sha": ledger["git_sha"],
    })

    print(f"候选 {meta.get('version')} 四门裁决（{ledger['gated_at']}）")
    for name, item in gates.items():
        print(f"  [{'PASS' if item.get('passed') else 'FAIL'}] {name}: {_gate_summary(name, item)}")
        if not item.get("passed") and item.get("reason"):
            print(f"         reason: {item['reason']}")
    print(f"verdict: {ledger['verdict']}   账：{candidate_dir / GATE_LEDGER}")
    if passed:
        print(f"下一步：python scripts/ltr_promote.py promote {candidate_dir}")
        return 0
    print("不转正：门禁未全过（fail-closed）。")
    return 1


def _gate_summary(name: str, item: dict) -> str:
    if name == "trainer_export_selfcheck":
        rel = item.get("reloaded_vs_declared") or {}
        rec = item.get("recorded_selfcheck") or {}
        return (f"退出码={rec.get('exit_code')} 记录Δ={rec.get('max_delta')} "
                f"重算Δ={rel.get('max_delta')}（容差 {item.get('tolerance')}）")
    if name == "heldout_no_regression":
        return (f"候选 nDCG@10={item.get('candidate', {}).get('nDCG@10')} "
                f"现役={item.get('incumbent_reloaded', {}).get('nDCG@10')} "
                f"Δ={item.get('delta_nDCG@10')}（容差 {item.get('tolerance')}）")
    if name == "ood_duel":
        verdict = item.get("verdict") or {}
        return (f"margin={verdict.get('margin')} 对手={verdict.get('best_opponent')} "
                f"{verdict.get('wins')}胜/{verdict.get('losses')}负/{verdict.get('ties')}平"
                f"（配对 {verdict.get('n_paired')} / OOD 行 {item.get('n_ood')}）")
    metrics = item.get("metrics") or {}
    return (f"Top1={metrics.get('Top1_Accuracy_%')} Top5={metrics.get('Top5_Hit_Rate_%')} "
            f"硬违规={metrics.get('Constraint_Violation_%')} FASTQ={metrics.get('FASTQ_Violation_%')} "
            f"LTR生效={(item.get('ltr_engagement') or {}).get('verified')}")


                                                                                     
def _artifact_payloads(src_dir: Path, expected: "dict | None" = None) -> dict:
    payloads = {name: (src_dir / name).read_bytes() for name in ARTIFACT_NAMES}
    actual = {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()}
    if expected is not None and actual != expected:
        raise PromoteError("制品指纹与认证记录不符，拒绝修改现役；请核对制品及历史证据")
    return payloads


def _archive_expected(archived_dir: Path, entry: "dict | None") -> dict:
    metadata_path = archived_dir / "_archive.json"
    try:
        metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromoteError("归档认证记录损坏，拒绝回滚") from exc
    if not isinstance(metadata, dict):
        raise PromoteError("归档认证记录不是 JSON 对象，拒绝回滚")
    evidence = []
    for record in (entry or {}, metadata):
        shas = record.get("artifact_sha256")
        if shas is None:
            continue
        if (not isinstance(shas, dict) or set(shas) != set(ARTIFACT_NAMES)
                or any(not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)
                       for sha in shas.values())):
            raise PromoteError("历史制品指纹不完整或格式错误，拒绝回滚")
        evidence.append(shas)
    if not evidence:
        raise PromoteError("缺少完整历史制品指纹，拒绝回滚；不能用现有归档字节重新认证")
    if any(shas != evidence[0] for shas in evidence[1:]):
        raise PromoteError("REGISTRY 与归档制品指纹冲突，拒绝回滚")
    return dict(evidence[0])


class _RestoreError(PromoteError):
    def __init__(self, primary, failures):
        self.primary_error = getattr(primary, "primary_error", primary)
        self.restore_failures = failures
        details = "; ".join(f"{path}: {type(error).__name__}: {error}" for path, error in failures)
        super().__init__(f"制品切换失败，恢复不完整；原始错误：{self.primary_error}；恢复失败项：{details}")


def _restore_files(before: dict) -> list:
    """每项独立尝试，只恢复实际变化；一项失败不能阻止其余现役、注册表或流水恢复。"""
    failures = []
    for path, raw in reversed(list(before.items())):
        try:
            if raw is None:
                path.unlink(missing_ok=True)
            elif not path.exists() or path.read_bytes() != raw:
                atomic_write_bytes(path, raw)
        except Exception as exc:
            failures.append((path, exc))
    return failures


@contextmanager
def _files_transaction(discover):
    before = {path: path.read_bytes() if path.exists() else None for path in discover()}
    try:
        yield
    except BaseException as primary:
        failures = []
        try:
            for path in discover():
                before.setdefault(path, None)
        except Exception as exc:
            failures.append(("transaction discovery", exc))
        failures.extend(_restore_files(before))
        if failures:
            raise _RestoreError(primary, failures) from primary
        raise


def _transactional_artifact(function):
    @wraps(function)
    def invoke(args):
        from dataset_recommender.corpus.fs_utils import acquire_file_lock, release_file_lock, path_lock
        paths = Paths(args.project_root.resolve())
        def discover():
            targets = [paths.ltr_dir / name for name in ARTIFACT_NAMES] + [paths.registry_path, paths.journal_path]
            if paths.archive_dir.exists():
                for directory in paths.archive_dir.iterdir():
                    if directory.is_dir():
                        targets.extend(directory / name for name in (*ARTIFACT_NAMES, "_archive.json"))
            return targets
        with path_lock(paths.ltr_dir):
            handle = acquire_file_lock(paths.ltr_dir / ".transaction.lock")
            try:
                with _journal_critical_section(paths.journal_path):
                    with _files_transaction(discover):
                        return function(args)
            except OSError as exc:
                raise PromoteError(f"制品切换失败，现役、注册表与流水已恢复：{exc}") from exc
            finally:
                release_file_lock(handle)
    return invoke


def _install_artifact(src_dir: Path, dst_dir: Path, *, expected: "dict | None" = None) -> dict:
    payloads = _artifact_payloads(src_dir, expected)
    with _files_transaction(lambda: [dst_dir / name for name in ARTIFACT_NAMES]):
        for name, raw in payloads.items():
            atomic_write_bytes(dst_dir / name, raw)
        return file_sha_map([dst_dir / name for name in ARTIFACT_NAMES], dst_dir)


def render_commit_message(version: str, spec: dict, entry: dict, archived_version: "str | None") -> str:
    counts = spec.get("train_counts") or {}
    heldout = (spec.get("heldout_metrics") or {}).get("nDCG@10")
    gate_flags = " ".join(f"{name}={'PASS' if ok else 'FAIL'}" for name, ok in (entry.get("gates") or {}).items())
    archive_note = (f"前代 {archived_version} 归档 models/ltr/_archive/{archived_version}/，"
                    if archived_version else "无前代可归档，")
    return "\n".join([
        f"LTR 转正 {version}：四门全过（heldout nDCG@10 {heldout}；train {counts.get('queries')} 查询/"
        f"{counts.get('pairs')} 对）",
        "",
        f"{archive_note}血缘与四门指标写入 models/ltr/REGISTRY.json；{gate_flags}。",
    ])


@_transactional_artifact
def cmd_promote(args) -> int:
    paths = Paths(args.project_root.resolve())
    candidate_dir = args.candidate_dir.resolve()
    require_artifact_files(candidate_dir)
    ledger_path = candidate_dir / GATE_LEDGER
    if not ledger_path.is_file():
        raise PromoteError(f"缺门禁账 {GATE_LEDGER}——先跑 gate（本命令不接受口头结论）：{candidate_dir}")
    ledger = read_json(ledger_path)
    gates = ledger.get("gates") if isinstance(ledger, dict) else None
    if (not isinstance(gates, dict) or set(gates) != REQUIRED_GATES
            or ledger.get("verdict") != "ACCEPT"
            or any(not isinstance(item, dict) or item.get("passed") is not True
                   for item in gates.values())):
        raise PromoteError("门禁未全过或四门记录不完整，拒绝转正")
    meta = read_json(candidate_dir / CANDIDATE_META)
    candidate_hashes = file_sha_map([candidate_dir / n for n in ARTIFACT_NAMES], candidate_dir)
    incumbent_hashes = file_sha_map([paths.ltr_dir / n for n in ARTIFACT_NAMES
                                     if (paths.ltr_dir / n).is_file()], paths.ltr_dir)
    if (not meta.get("version") or ledger.get("version") != meta["version"]
            or ledger.get("artifact_sha256") != candidate_hashes
            or ledger.get("incumbent_sha256") != incumbent_hashes):
        raise PromoteError("门禁与候选版本/制品/现役对照不符，请重新运行 gate")
    _artifact_payloads(candidate_dir, meta.get("artifact_sha256") or {})
    candidate_spec = load_spec(candidate_dir)
    candidate_version = str(meta.get("version") or version_of_spec(candidate_spec))

    registry = load_registry(paths.registry_path)
    incumbent = active_entry(registry)
    if incumbent is None and (paths.ltr_dir / SPEC_NAME).is_file():
        incumbent_spec = load_spec(paths.ltr_dir)
        incumbent = entry_from_spec(
            incumbent_spec, version=version_of_spec(incumbent_spec), fingerprint=None, status="active",
            artifact_sha256=file_sha_map([paths.ltr_dir / n for n in ARTIFACT_NAMES], paths.ltr_dir))
    if incumbent is not None and incumbent.get("version") == candidate_version:
        raise PromoteError(f"候选与现役同版本（{candidate_version}）——同基准同 seed 重训不产生新制品，无需转正")

    archived_dir = None
    if (paths.ltr_dir / SPEC_NAME).is_file():
        archived_dir = paths.archive_dir / str(incumbent["version"])
        archived_shas = _install_artifact(paths.ltr_dir, archived_dir)
        write_json_atomic(archived_dir / "_archive.json", {
            "archived_at": now_iso(), "reason": f"被 {candidate_version} 转正替代",
            "successor": candidate_version, "artifact_sha256": archived_shas,
            "spec": load_spec(paths.ltr_dir),
        })
    installed = _install_artifact(candidate_dir, paths.ltr_dir, expected=meta["artifact_sha256"])
    for name in ARTIFACT_NAMES:
        if installed[name] != meta["artifact_sha256"][name]:
            raise PromoteError(f"就位字节核验失败：{name} 的 SHA-256 与候选不符（拒绝留下半截状态）")
    archived_version = (incumbent or {}).get("version")
    if incumbent is not None:
        incumbent.update({
            "status": "archived", "archived_at": now_iso(),
            "archived_to": archived_dir.relative_to(paths.root).as_posix() if archived_dir else None,
        })
        upsert_registry_entry(registry, incumbent)
    entry = entry_from_spec(
        candidate_spec, version=candidate_version, fingerprint=meta["data_fingerprint"]["train"],
        status="active", artifact_shas=installed,
        promoted_at=now_iso(), promoted_from=archived_version, git_sha=git_sha(paths.root),
        gates={name: bool(item.get("passed")) for name, item in (ledger.get("gates") or {}).items()},
        gate_ledger=str(ledger_path), basis=meta.get("basis"),
    )
    upsert_registry_entry(registry, entry)
    registry["active"] = candidate_version
    save_registry(paths.registry_path, registry)
    append_journal(paths.journal_path, {
        "action": "promote", "version": candidate_version, "candidate_dir": str(candidate_dir),
        "archived_from": archived_version, "data_fingerprint": meta["data_fingerprint"]["train"],
        "gates": entry["gates"], "git_sha": entry["git_sha"],
    })

    print(f"转正完成：{candidate_version}（前代 {archived_version or '无'}）")
    print(f"  归档前代：{archived_dir.relative_to(paths.root).as_posix() if archived_dir else '（无）'}")
    print(f"  注册表：{paths.registry_path.relative_to(paths.root).as_posix()}")
    print("  建议 commit message：")
    for line in render_commit_message(candidate_version, candidate_spec, entry, archived_version).splitlines():
        print(f"    {line}")
    return 0


@_transactional_artifact
def cmd_rollback(args) -> int:
    paths = Paths(args.project_root.resolve())
    version = args.version
    archived_dir = paths.archive_dir / version
    if not (archived_dir / SPEC_NAME).is_file():
        available = sorted(p.name for p in paths.archive_dir.iterdir()) if paths.archive_dir.is_dir() else []
        raise PromoteError(f"归档中无该版本：{version}（可用：{available or '空'}）")
    try:
        registry = load_registry(paths.registry_path)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromoteError("REGISTRY 认证记录损坏，拒绝回滚") from exc
    current = active_entry(registry)
    current_version = (current or {}).get("version")
    if current_version == version:
        raise PromoteError(f"该版本已是现役：{version}")
    restored = registry_entry(registry, version)
    expected = _archive_expected(archived_dir, restored)
    _artifact_payloads(archived_dir, expected)

    demoted_dir = None
    if (paths.ltr_dir / SPEC_NAME).is_file() and current_version:
        demoted_dir = paths.archive_dir / str(current_version)
        if not (demoted_dir / SPEC_NAME).is_file():
            demoted_shas = _install_artifact(paths.ltr_dir, demoted_dir)
            write_json_atomic(demoted_dir / "_archive.json", {
                "archived_at": now_iso(), "reason": f"回滚到 {version} 时被替换下来",
                "successor": version, "artifact_sha256": demoted_shas, "spec": load_spec(paths.ltr_dir),
            })
        current.update({"status": "rolled_back", "rolled_back_at": now_iso(), "rolled_back_to": version})
        upsert_registry_entry(registry, current)
    installed = _install_artifact(archived_dir, paths.ltr_dir, expected=expected)
    restored = restored or entry_from_spec(
        load_spec(paths.ltr_dir), version=version, fingerprint=None, status="active")
    restored.update({"status": "active", "restored_at": now_iso(), "restored_from": current_version,
                     "artifact_sha256": expected, "git_sha": git_sha(paths.root)})
    upsert_registry_entry(registry, restored)
    registry["active"] = version
    save_registry(paths.registry_path, registry)
    append_journal(paths.journal_path, {
        "action": "rollback", "version": version, "rolled_back_from": current_version,
        "archive_dir": archived_dir.relative_to(paths.root).as_posix(), "git_sha": restored["git_sha"],
    })
    print(f"回滚完成：{current_version} → {version}")
    print(f"  现役字节：{ {k: v[:12] + '…' for k, v in installed.items()} }")
    print(f"  被替换的 {current_version} 已归档："
          f"{demoted_dir.relative_to(paths.root).as_posix() if demoted_dir else '（无）'}")
    print("  建议 commit message：")
    print(f"    LTR 回滚至 {version}：{current_version} 表现不符预期，按 REGISTRY 血缘恢复前代制品"
          f"（{version} 记 active，{current_version} 记 rolled_back）。")
    return 0


                                                                          
def cmd_history(args) -> int:
    paths = Paths(args.project_root.resolve())
    registry = load_registry(paths.registry_path)
    print(f"== REGISTRY（tracked 真源）：{paths.registry_path.relative_to(paths.root).as_posix()} ==")
    if not registry["versions"]:
        print("  （空）")
    for entry in sorted(registry["versions"], key=lambda e: str(e.get("trained_at") or ""), reverse=True):
        mark = "*" if entry.get("version") == registry.get("active") else " "
        counts = entry.get("train_counts") or {}
        fp = entry.get("data_fingerprint")
        print(f" {mark} {entry.get('version')}  {entry.get('status')}  trained_at={entry.get('trained_at')}")
        print(f"      train={counts.get('queries')} 查询/{counts.get('pairs')} 对  "
              f"指纹={fp[:12] + '…' if fp else '未记录'}  heldout={(entry.get('heldout_metrics') or {}).get('nDCG@10')}")
        if entry.get("gates"):
            print(f"      四门：{entry['gates']}")
        if entry.get("promoted_at"):
            print(f"      promoted_at={entry['promoted_at']} git={entry.get('git_sha')}"
                  + (f"  rolled_back_at={entry['rolled_back_at']}" if entry.get("rolled_back_at") else ""))
    journal = paths.journal_path
    print(f"\n== 未跟踪流水：{journal.relative_to(paths.root).as_posix()} ==")
    if not journal.is_file():
        print("  （无）")
        return 0
    lines = [line for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in lines[-args.limit:]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print(f"  ! 坏行（跳过）：{line[:80]}")
            continue
        detail = row.get("verdict") or (row.get("data_fingerprint") or "")[:16] or ""
        print(f"  {row.get('ts')}  {str(row.get('action')):<16} {str(row.get('version') or ''):<24} "
              f"{detail:<18} {row.get('candidate_dir') or ''}")
    print(f"  共 {len(lines)} 行（显示末 {min(args.limit, len(lines))} 行）")
    return 0


                                                                      
def _ensure_cli_utf8_stdout() -> None:
    """CLI-only 编码垫片（Windows 控制台默认 GBK 会让中文输出报错）。

    只在 stdout 明显不是 UTF-8 时才替换——import 本模块、或在 pytest 的 capsys 捕获下运行，
    都绝不触碰 stdout（替换会让捕获对象与调用方持有的引用脱钩）。"""
    stream = sys.stdout
    encoding = str(getattr(stream, "encoding", "") or "").lower().replace("-", "")
    if encoding in ("utf8", "utf8mb4") or not hasattr(stream, "buffer"):
        return
    try:
        import io                 
        sys.stdout = io.TextIOWrapper(stream.buffer, encoding="utf-8")
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=AGENT_ROOT,
                        help="项目根（模型/研究数据/流水账都从它派生；测试用临时根即可整体隔离）")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_train_flags(p):
        p.add_argument("--in-dir", type=Path, default=None, help="训练数据目录（默认 <root>/research/ltr_data）")
        p.add_argument("--label", choices=("det", "sem", "mix"), default="mix")
        p.add_argument("--seed", type=int, default=None, help="训练种子（默认 = train_ltr.DEFAULT_SEED）")

    status = sub.add_parser("status", help="攒量：距阈值还差多少新标注")
    add_train_flags(status)
    status.add_argument("--require-ready", action="store_true", help="未达阈值时退出码 1（cron 用）")
    status.add_argument("--json", action="store_true", help="额外打印完整 JSON 报告")
    status.set_defaults(func=cmd_status)

    candidate = sub.add_parser("candidate", help="包装 train_ltr 训练一份候选到未跟踪暂存区")
    add_train_flags(candidate)
    candidate.add_argument("--feature-set", choices=("f0", "f3"), default=CANDIDATE_FEATURE_SET,
                           help=f"训练/评测特征配方（默认 {CANDIDATE_FEATURE_SET}：生产存活池回放 + "
                                "共享构造函数，训练/服役同口径；f0 = 历史并集池配方，留作对照）")
    candidate.add_argument("--out-dir", type=Path, default=None)
    candidate.add_argument("--ts", default=None, help="候选目录时间戳（默认当前本地时间）")
    candidate.add_argument("--mix-in", type=Path, action="append", default=[],
                           help="额外标注层（重复可给；默认 = MIX_IN_PATTERNS 的解析结果）")
    candidate.add_argument("--no-default-mix-in", action="store_true", help="只用 --mix-in 显式给出的层")
    candidate.add_argument("--python", default=sys.executable, help="训练用的解释器")
    candidate.add_argument("--dry-run", action="store_true", help="只打印训练命令")
    candidate.set_defaults(func=cmd_candidate)

    gate = sub.add_parser("gate", help="四门裁决（无论过不过都落账）")
    gate.add_argument("candidate_dir", type=Path)
    add_train_flags(gate)
    gate.add_argument("--ood-path", type=Path, default=None,
                      help="OOD 交叉打分集（默认 <root>/" + DEFAULT_OOD_REL + "）")
    gate.add_argument("--frozen-report", type=Path, default=None, help="用已跑好的冻结报告代替进程内真跑")
    gate.add_argument("--skip-ood", action="store_true")
    gate.add_argument("--skip-frozen", action="store_true")
    gate.set_defaults(func=cmd_gate)

    promote = sub.add_parser("promote", help="归档现役 + 就位候选 + 更新 REGISTRY")
    promote.add_argument("candidate_dir", type=Path)
    promote.set_defaults(func=cmd_promote)

    rollback = sub.add_parser("rollback", help="从归档恢复指定版本为现役")
    rollback.add_argument("version")
    rollback.set_defaults(func=cmd_rollback)

    history = sub.add_parser("history", help="REGISTRY + 未跟踪流水")
    history.add_argument("--limit", type=int, default=20)
    history.set_defaults(func=cmd_history)
    return parser


def _apply_defaults(args) -> None:
    """把 None/相对默认值补齐到具体路径。"""
    paths = Paths(args.project_root.resolve())
                                                       
                                                             
    if getattr(args, "in_dir", None) is None and hasattr(args, "in_dir") and args.command == "status":
        args.in_dir = paths.ltr_data_dir
    if getattr(args, "seed", None) is None and hasattr(args, "seed"):
        args.seed = _train_ltr(str(paths.root)).DEFAULT_SEED


def main(argv: "list[str] | None" = None) -> int:
    _ensure_cli_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_defaults(args)
    try:
        return args.func(args)
    except PromoteError as exc:
        print(f"! {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
