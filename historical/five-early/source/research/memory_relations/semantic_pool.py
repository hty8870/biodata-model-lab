"""M14 受保护公开池 → C5/dual 共用 canonical JSONL 的机械转换（research-only，CPU，源只读）。

只消费 M14 公开数据 run 的受保护工件并把官方 NLI3 训练记录机械映射为 semantic_model
可消费的 canonical 行；不重造下载/聚簇/保护算法，不训练，不改 gold，不猜 fine/inverse。

映射（唯一真源 semantic_relations；old 固定 P、new 固定 H）：
- old = premise 命题、new = hypothesis 命题；frame 走 validate_semantic_pair 的同域声明。
- label = pool 的 gold_label（仅 nli3 E/N/C 三值）；label_origin = public_official。
- root_id = M14 依赖单位 unit_id；cluster_id = 该单位唯一 cluster（units 1:1，不重分区）。
- id = "public:" + 源 id；source_id 原样保留，producer member 检查按 source_id。

超长输入：唯一既有 raw→学生视图通道的硬界就是 stdlib `MAX_TEXT_CHARS`（无 cap 旋钮、不复制
常量、不写第二截断器）；超限行不截断、不丢文档：整行进 source-too-large.jsonl（保全原文 +
full sha256 / 字符 / coverage 证据），不进可训练切片；分母守恒 = trainable + quarantine = 输入成员。

验证锚点（convert 与 verify 共用同一实现）：manifest schema、必需 hash 表、文件 roles、名称围栏
（只接受正规相对 basename；拒 `../`、绝对路径、分隔符、`~`、重解析别名）与输入/输出 reparse 检查。
verify 从源 pool + 成员资产 + 依赖单位重派生 id / member / split / label / pair / quarantine reason
与守恒，不信任 manifest 计数旗标。位置迁移后 `reanchor` 生成新版本 manifest（readback 哈希 +
当前路径可重放命令，历史路径明确标为已退休），旧版本保留为历史原件。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

from dataset_recommender.agent import semantic_relations as SR
from research.memory_relations import finetune as FT
from research.memory_relations import nli_reference as NR
from research.memory_relations import semantic_model as SM

CONVERTER_SCHEMA = "semantic-public-pool-conversion-v1"
ASSETS_SCHEMA = "semantic-producer-assets-v1"
PRODUCER = "v6-public-pool-20261014"
LABEL_SPACE = "nli3"
LABEL_ORIGIN = "public_official"
ID_PREFIX = "public:"
TRAINABLE_FILE = "public-pool.jsonl"
QUARANTINE_FILE = "source-too-large.jsonl"
ASSETS_FILE = "producer-assets.json"
LEGACY_MANIFEST_FILE = "conversion-manifest.json"
MANIFEST_FILE_TEMPLATE = "conversion-manifest-v{revision}.json"
SIGNATURES_FILE = "source-too-large-signatures.jsonl"
QUARANTINE_REASON = "source_too_large"
ROLE_ASSETS = (
    ("protected-fit", "fit-ids.jsonl", "protected-fit.jsonl"),
    ("protected-dev", "dev-ids.jsonl", "protected-dev.jsonl"),
    ("protection-manifest", "protected.jsonl", "protection-manifest.jsonl"),
)
SPLIT_ROLE = {"train": "protected-fit", "dev": "protected-dev"}
REQUIRED_OUTPUT_FILES = (TRAINABLE_FILE, QUARANTINE_FILE, ASSETS_FILE) + tuple(role[2] for role in ROLE_ASSETS)
REQUIRED_SOURCE_FILES = ("pool.jsonl", "fit-ids.jsonl", "dev-ids.jsonl", "units.jsonl", "protected.jsonl")
ROLE_SOURCE_FILES = {role: source for role, source, _ in ROLE_ASSETS}
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")

                                                                      
EXPECTED_SOURCE_SHA256 = {
    "pool.jsonl": "e30c2f111c303af1ce952bcb301793286b7ac75af772573f7b8b1facc9d16351",
    "fit-ids.jsonl": "d0342ce2c71be0d8f6c5fec8d3c72461ab9f1ac741834197409d4bfb971cb1ef",
    "dev-ids.jsonl": "6028e304ba6a52d5ded9964d15ae6939d48b07059edc197739ae1d3322f257be",
    "units.jsonl": "614593710207042c84f6a57e4efc14ce68d42626784639661dde26e98402338c",
    "dependencies.jsonl": "614593710207042c84f6a57e4efc14ce68d42626784639661dde26e98402338c",
    "protected.jsonl": "cf829ac1486c225c13222c6a53a550a97292eea28232d45ca8fa6b2bc50e2e20",
}


class SemanticPoolError(ValueError):
    def __init__(self, code: str, hint: str) -> None:
        self.code = code
        self.hint = hint
        super().__init__(f"{code}: {hint}")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pair_digest(pair) -> str:
    return hashlib.sha256(json.dumps(pair, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                yield number, json.loads(line)


def _iter_jsonl_safe(path: Path, problems: list, label: str):
    if not path.is_file():
        problems.append(f"{label}_missing:{path.name}")
        return
    try:
        yield from _iter_jsonl(path)
    except (OSError, json.JSONDecodeError) as error:
        problems.append(f"{label}_unreadable:{path.name}:{error}")


def _relative_name(name, *, code: str = "path_not_relative") -> str:
    """唯一名称围栏：只接受正规相对 basename；拒绝对路径、分隔符、`..`、`~`。"""
    if (type(name) is not str or not name or name in (".", "..") or "/" in name or "\\" in name
            or ":" in name or name.startswith("~") or Path(name).name != name):
        raise SemanticPoolError(code, f"只接受正规相对 basename：{name!r}")
    return name


def _assert_plain_path(path: Path, *, code: str) -> Path:
    if NR._reparse_components(Path(path)):
        raise SemanticPoolError(code, f"路径含链接/重解析别名：{path}")
    return path


def _assert_fresh(out_dir: Path) -> None:
    _relative_name(out_dir.name, code="output_not_plain")
    if ".." in out_dir.parts:
        raise SemanticPoolError("output_not_plain", f"输出路径不得含 '..'：{out_dir}")
    if NR._reparse_components(out_dir) or NR._reparse_components(out_dir.parent):
        raise SemanticPoolError("output_not_plain", f"输出路径必须是普通全新目录：{out_dir}")
    if out_dir.exists():
        raise SemanticPoolError("output_exists", f"输出目录已存在，拒绝覆盖：{out_dir}")


def _write_line(handle, value: dict) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def _length_exceeded(record: dict) -> bool:
    """唯一超限判据：原始文本长度 vs stdlib 唯一硬界（不引入第二个可调 cap）。"""
    bound = SR.MAX_TEXT_CHARS
    return (len(record.get("premise") or "") > bound) or (len(record.get("hypothesis") or "") > bound)


def _over_fields(record: dict) -> list:
    return [field for field in ("premise", "hypothesis")
            if len(record.get(field) or "") > SR.MAX_TEXT_CHARS]


def _raw_text_block(text) -> dict:
    return {"chars": len(text) if isinstance(text, str) else None,
            "sha256": _text_sha256(text) if isinstance(text, str) else None}


def _block_text(container, field):
    block = container.get(field) if isinstance(container, dict) else None
    return block.get("text") if isinstance(block, dict) else None


def _source_signature(record: dict, split, units) -> dict:
    """sourceTooLong 成员的 raw canonical projection signature（不写第二份映射，直接用源 record）。"""
    unit = units.get(record.get("id")) if units is not None else None
    return {"source_id": record.get("id"), "split": split, "label": record.get("gold_label"),
            "root_id": unit["unit_id"] if unit else None, "cluster_id": unit["cluster_id"] if unit else None,
            "premise": _raw_text_block(record.get("premise")),
            "hypothesis": _raw_text_block(record.get("hypothesis")),
            "over_fields": _over_fields(record), "cap_chars": SR.MAX_TEXT_CHARS,
            "reason": QUARANTINE_REASON, "windowed_applied": False}


def _reconcile_quarantine(problems: list, record: dict, projection: dict, units, expected_split) -> None:
    """逐条把 quarantine 行与实际源 record 核：文本 sha/字符、label、unit/root/cluster、split、why-too-large。"""
    source_id = record.get("id")
    if projection.get("split") != expected_split:
        problems.append(f"quarantine_split:{source_id}")
    if projection.get("label") != record.get("gold_label"):
        problems.append(f"quarantine_label_mismatch:{source_id}")
    if units is not None:
        unit = units.get(source_id)
        if unit is None:
            problems.append(f"quarantine_unit_missing:{source_id}")
        elif projection.get("root_id") != unit["unit_id"] or projection.get("cluster_id") != unit["cluster_id"]:
            problems.append(f"quarantine_unit_mismatch:{source_id}")
    for field in ("premise", "hypothesis"):
        expected = _raw_text_block(record.get(field))
        stored = projection.get(field) or {}
        if stored.get("sha256") != expected["sha256"] or stored.get("chars") != expected["chars"]:
            problems.append(f"quarantine_text_mismatch:{source_id}:{field}")
    if not _over_fields(record):
        problems.append(f"quarantine_reason_unjustified:{source_id}")


def _load_plan(source_dir: Path):
    plan_path = source_dir / "plan.json"
    if not plan_path.is_file():
        return None, None
    _assert_plain_path(plan_path, code="input_reparse")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return plan, FT.file_sha256(plan_path)


def _verify_sources(source_dir: Path, expected: dict) -> tuple:
    """核验受保护工件 SHA 并返回实测哈希；名称围栏 + 输入 reparse 检查（模块表与 plan 工件表互核）。"""
    _assert_plain_path(source_dir, code="input_reparse")
    plan, plan_sha = _load_plan(source_dir)
    table = {}
    for name, sha in dict(expected).items():
        table[_relative_name(name, code="source_name_invalid")] = sha
    if plan is not None:
        artifacts = plan.get("artifacts")
        if artifacts is not None and not isinstance(artifacts, dict):
            raise SemanticPoolError("source_plan_invalid", "plan.artifacts 必须是映射")
        for name, sha in (artifacts or {}).items():
            name = _relative_name(name, code="source_name_invalid")
            if name in table and table[name] != sha:
                raise SemanticPoolError("source_hash_conflict", f"{name}: 模块表与 plan 工件表不一致")
            table[name] = sha
    hashes = {}
    for name in sorted(table):
        if not _HEX64.match(str(table[name])):
            raise SemanticPoolError("source_hash_invalid", f"{name}: 期望值不是 sha256")
        path = source_dir / name
        if not path.is_file():
            raise SemanticPoolError("source_missing", f"受保护工件缺失：{path}")
        _assert_plain_path(path, code="input_reparse")
        actual = FT.file_sha256(path)
        if actual != table[name]:
            raise SemanticPoolError("source_hash_drift", f"{name}: {actual} != {table[name]}")
        hashes[name] = actual
    if "units.jsonl" in hashes and "dependencies.jsonl" in hashes and hashes["units.jsonl"] != hashes["dependencies.jsonl"]:
        raise SemanticPoolError("source_hash_conflict", "units.jsonl 与 dependencies.jsonl 不再同一字节")
    return hashes, plan, plan_sha


def _load_member_ids(path: Path, role: str) -> list:
    _assert_plain_path(path, code="input_reparse")
    ids = [value for _, value in _iter_jsonl(path)]
    if not ids or any(type(value) is not str or not value for value in ids):
        raise SemanticPoolError("bad_member_asset", f"{role} 成员资产须为非空字符串列表：{path}")
    if len(set(ids)) != len(ids):
        raise SemanticPoolError("duplicate_member", f"{role} 成员资产存在重复 id")
    return ids


def _load_units(path: Path) -> dict:
    _assert_plain_path(path, code="input_reparse")
    units = {}
    owners = {}
    for number, unit in _iter_jsonl(path):
        unit_id = unit.get("unit_id")
        clusters = unit.get("clusters")
        members = unit.get("record_ids")
        if (type(unit_id) is not str or not unit_id or not isinstance(clusters, list)
                or len(clusters) != 1 or type(clusters[0]) is not str
                or not isinstance(members, list) or not members or unit.get("rows") != len(members)):
            raise SemanticPoolError("unsupported_unit_shape", f"{path}:{number} 依赖单位形状不受支持")
        cluster = clusters[0]
        if cluster in owners:
            raise SemanticPoolError("duplicate_cluster", f"{cluster} 同时属于多个单位")
        owners[cluster] = unit_id
        for record_id in members:
            if type(record_id) is not str or record_id in units:
                raise SemanticPoolError("duplicate_unit_membership", f"{path}:{number} 记录重复归属单位")
            units[record_id] = {"unit_id": unit_id, "cluster_id": cluster}
    if not units:
        raise SemanticPoolError("empty_units", f"{path} 无依赖单位")
    return units


def _pair_candidate(record: dict) -> dict:
    return {
        "frame": {"mode": "same_domain", "background": ""},
        "old": {"kind": "proposition", "text": record.get("premise")},
        "new": {"kind": "proposition", "text": record.get("hypothesis")},
    }


def _lineage(record: dict, *, role: str, pool_sha: str) -> dict:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    evidence = record.get("evidence") if isinstance(record.get("evidence"), dict) else {}
    return {
        "pool_sha256": pool_sha,
        "protected_role": role,
        "dataset": record.get("dataset"),
        "official_split": record.get("official_split"),
        "task": record.get("task"),
        "gold_status": metadata.get("gold_status"),
        "label_method": metadata.get("label_method"),
        "original_label": metadata.get("original_label"),
        "group_id": record.get("group_id"),
        "evidence": {key: evidence[key] for key in ("file", "row", "sha256", "source_url") if key in evidence},
    }


def _quarantine_row(record: dict, *, split: str, unit: dict, role: str, pool_sha: str,
                    error: SR.SemanticRelationError) -> dict:
    premise, hypothesis = record.get("premise"), record.get("hypothesis")
    source_id = record["id"]
    cap = SR.MAX_TEXT_CHARS

    def full(text):
        chars = len(text) if isinstance(text, str) else 0
        return {"chars": chars, "sha256": _text_sha256(text) if isinstance(text, str) else None,
                "coverage_if_windowed": (min(chars, cap) / chars) if chars else 1.0,
                "text": text}

    return {
        "id": ID_PREFIX + source_id,
        "source_id": source_id,
        "root_id": unit["unit_id"],
        "cluster_id": unit["cluster_id"],
        "split": split,
        "label_space": LABEL_SPACE,
        "label": record.get("gold_label"),
        "label_origin": LABEL_ORIGIN,
        "dataset": record.get("dataset"),
        "reason": "source_too_large",
        "error": {"code": error.code, "hint": error.hint},
        "windowed_limit_chars": cap,
        "windowed_applied": False,
        "full": {"premise": full(premise), "hypothesis": full(hypothesis)},
        "lineage": _lineage(record, role=role, pool_sha=pool_sha),
    }


def _strata(lengths: list) -> dict:
    ordered = sorted(lengths)
    if not ordered:
        return {"rows": 0}

    def quantile(q):
        return ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]

    buckets = Counter()
    for value in ordered:
        if value <= 1024:
            buckets["<=1024"] += 1
        elif value <= 2048:
            buckets["1025-2048"] += 1
        elif value <= 4096:
            buckets["2049-4096"] += 1
        elif value <= 8192:
            buckets["4097-8192"] += 1
        else:
            buckets[">8192"] += 1
    return {"rows": len(ordered), "min": ordered[0], "p50": quantile(0.5), "p90": quantile(0.9),
            "p95": quantile(0.95), "p99": quantile(0.99), "max": ordered[-1],
            "mean": round(sum(ordered) / len(ordered), 2),
            "buckets": {name: buckets[name] for name in ("<=1024", "1025-2048", "2049-4096", "4097-8192", ">8192")}}


def _manifest_problems(manifest: dict, ownrun_dir: "Path | None" = None) -> list:
    """单一 manifest 形状锚点：schema、必需 hash 表、输出 roles、名称围栏与文件 reparse。"""
    problems = []
    if not isinstance(manifest, dict) or manifest.get("schema") != CONVERTER_SCHEMA:
        problems.append("manifest_schema")
        return problems
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        problems.append("manifest_outputs_missing")
        outputs = {}
    for name in REQUIRED_OUTPUT_FILES:
        if name not in outputs:
            problems.append(f"manifest_outputs_missing_role:{name}")
    for name, item in outputs.items():
        try:
            _relative_name(name, code="output_name_invalid")
        except SemanticPoolError:
            problems.append(f"manifest_outputs_name:{name}")
            continue
        if (not isinstance(item, dict) or not _HEX64.match(str(item.get("sha256")))
                or not isinstance(item.get("rows"), int) or not isinstance(item.get("bytes"), int)):
            problems.append(f"manifest_outputs_item:{name}")
    source = manifest.get("source")
    hashes = source.get("hashes") if isinstance(source, dict) else None
    if not isinstance(hashes, dict) or not hashes:
        problems.append("manifest_source_hashes_missing")
        hashes = {}
    for name in REQUIRED_SOURCE_FILES:
        if name not in hashes:
            problems.append(f"manifest_source_hash_missing_role:{name}")
    for name, value in hashes.items():
        try:
            _relative_name(name, code="source_name_invalid")
        except SemanticPoolError:
            problems.append(f"manifest_source_name:{name}")
            continue
        if not _HEX64.match(str(value)):
            problems.append(f"manifest_source_hash_value:{name}")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        problems.append("manifest_counts_missing")
    else:
        for group, keys in (("input", ("pool", "fit", "dev")),
                            ("trainable", ("train", "dev", "total")),
                            ("quarantine", ("train", "dev", "total"))):
            block = counts.get(group)
            if not isinstance(block, dict) or any(not isinstance(block.get(key), int) for key in keys):
                problems.append(f"manifest_counts_{group}")
    if not isinstance(manifest.get("producer_lineage"), dict):
        problems.append("manifest_producer_lineage_missing")
    signatures = manifest.get("signatures")
    if signatures is not None:
        if not isinstance(signatures, dict):
            problems.append("manifest_signatures_item")
        else:
            try:
                name = _relative_name(signatures.get("file"), code="signatures_name_invalid")
            except SemanticPoolError:
                problems.append("manifest_signatures_shape")
                name = None
            if name is not None and (not _HEX64.match(str(signatures.get("sha256")))
                                     or not isinstance(signatures.get("rows"), int)):
                problems.append("manifest_signatures_shape")
    if ownrun_dir is not None:
        for name in list(outputs) + [ASSETS_FILE, signatures.get("file") if isinstance(signatures, dict) else None]:
            try:
                checked = _relative_name(name)
            except SemanticPoolError:
                continue
            path = ownrun_dir / checked
            if path.exists() and NR._reparse_components(path):
                problems.append(f"output_reparse:{checked}")
    return problems


def _assets_problems(document, base_dir: Path) -> tuple:
    problems = []
    if not isinstance(document, dict) or document.get("schema") != ASSETS_SCHEMA:
        problems.append("assets_schema")
    if not isinstance((document or {}).get("producer"), str) or not (document or {}).get("producer"):
        problems.append("assets_producer")
    assets = (document or {}).get("assets") if isinstance(document, dict) else None
    resolved = {}
    if not isinstance(assets, list):
        problems.append("assets_missing")
        return problems, resolved
    known_roles = {role for role, _, _ in ROLE_ASSETS}
    for item in assets:
        if not isinstance(item, dict):
            problems.append("assets_item")
            continue
        role, name, sha = item.get("role"), item.get("path"), item.get("sha256")
        if role not in known_roles or role in resolved:
            problems.append(f"assets_role:{role}")
            continue
        try:
            name = _relative_name(name, code="asset_path_invalid")
        except SemanticPoolError:
            problems.append(f"assets_path:{role}")
            continue
        if not _HEX64.match(str(sha)):
            problems.append(f"assets_sha:{role}")
            continue
        path = base_dir / name
        if not path.is_file():
            problems.append(f"assets_missing_file:{name}")
            continue
        if NR._reparse_components(path):
            problems.append(f"assets_reparse:{name}")
            continue
        if FT.file_sha256(path) != sha:
            problems.append(f"assets_hash_drift:{name}")
            continue
        resolved[role] = {"path": name, "sha256": sha}
    for role, _, _ in ROLE_ASSETS:
        if role not in resolved:
            problems.append(f"assets_missing_role:{role}")
    return problems, resolved


def _scan(ownrun_dir: Path, source_dir: Path, manifest: dict) -> dict:
    """从一个 manifest 锚点重派生全部事实（verify 与 reanchor 共用；不信任 manifest 计数旗标）。"""
    problems = _manifest_problems(manifest, ownrun_dir=ownrun_dir)
    source_expected = (manifest.get("source") or {}).get("hashes") or {}
    source_hashes = {}
    for name, expected_sha in source_expected.items():
        try:
            name = _relative_name(name)
        except SemanticPoolError:
            continue
        path = source_dir / name
        if not path.is_file():
            problems.append(f"source_missing:{name}")
            continue
        if NR._reparse_components(path):
            problems.append(f"source_reparse:{name}")
            continue
        actual = FT.file_sha256(path)
        source_hashes[name] = actual
        if actual != expected_sha:
            problems.append(f"source_hash_drift:{name}")
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    for name, item in outputs.items():
        try:
            name = _relative_name(name)
        except SemanticPoolError:
            continue
        path = ownrun_dir / name
        if not path.is_file():
            problems.append(f"output_missing:{name}")
            continue
        if NR._reparse_components(path):
            problems.append(f"output_reparse:{name}")
            continue
        if FT.file_sha256(path) != (item.get("sha256") if isinstance(item, dict) else None):
            problems.append(f"output_hash_drift:{name}")

    asset_problems, roles = _assets_problems(_read_json_safe(ownrun_dir / ASSETS_FILE), ownrun_dir)
    problems += asset_problems
    for role, source_file in ROLE_SOURCE_FILES.items():
        if role in roles and source_file in source_hashes and roles[role]["sha256"] != source_hashes[source_file]:
            problems.append(f"asset_source_mismatch:{role}")
    membership_ready = not asset_problems
    fit_set = set(_read_member_safe(roles, "protected-fit", ownrun_dir, problems))
    dev_set = set(_read_member_safe(roles, "protected-dev", ownrun_dir, problems))
    if membership_ready and (not fit_set or not dev_set or fit_set & dev_set):
        problems.append("member_sets_invalid")
        membership_ready = False

    units = {}
    units_ready = False
    if "units.jsonl" in source_hashes or (source_dir / "units.jsonl").is_file():
        try:
            units = _load_units(source_dir / "units.jsonl")
            units_ready = True
        except (SemanticPoolError, OSError, json.JSONDecodeError) as error:
            problems.append(f"units_invalid:{error}")

    content = {}
    trainable_ids = {"train": set(), "dev": set()}
    counts = Counter()
    check_rows = []
    seen_row_ids = set()
    for line_number, row in _iter_jsonl_safe(ownrun_dir / TRAINABLE_FILE, problems, "trainable"):
        if not isinstance(row, dict):
            problems.append(f"row_shape:{line_number}")
            continue
        if row.get("label_space") != LABEL_SPACE or row.get("label_origin") != LABEL_ORIGIN:
            problems.append(f"row_origin:{line_number}")
        try:
            SM.validate_row(row, stage="fit" if row.get("split") == "train" else "dev")
        except (ValueError, KeyError) as error:
            problems.append(f"row_invalid:{line_number}:{error}")
            continue
        source_id = row.get("source_id")
        if row.get("id") != ID_PREFIX + str(source_id):
            problems.append(f"row_id_namespace:{line_number}")
        if source_id in content:
            problems.append(f"duplicate_source_id:{source_id}")
            continue
        if row.get("id") in seen_row_ids:
            problems.append(f"duplicate_row:{row.get('id')}")
        seen_row_ids.add(row.get("id"))
        content[source_id] = (row.get("split"), row.get("label"), _pair_digest(row["pair"]))
        trainable_ids[row.get("split")].add(source_id)
        counts[row.get("split")] += 1
        if units_ready:
            unit = units.get(source_id)
            if unit is None:
                problems.append(f"row_unit_missing:{source_id}")
            elif row.get("root_id") != unit["unit_id"] or row.get("cluster_id") != unit["cluster_id"]:
                problems.append(f"row_unit_mismatch:{source_id}")
        if membership_ready:
            expected_split = "train" if source_id in fit_set else ("dev" if source_id in dev_set else None)
            if expected_split is None:
                problems.append(f"row_not_protected_member:{source_id}")
            elif row.get("split") != expected_split:
                problems.append(f"row_split_mismatch:{source_id}")
        check_rows.append({"id": row.get("id"), "source_id": source_id, "split": row.get("split"),
                           "label_space": row.get("label_space")})

    quarantine_split = {}
    quarantine_projection = {}
    quarantine_counts = Counter()
    for line_number, row in _iter_jsonl_safe(ownrun_dir / QUARANTINE_FILE, problems, "quarantine"):
        if not isinstance(row, dict):
            problems.append(f"quarantine_shape:{line_number}")
            continue
        source_id = row.get("source_id")
        if row.get("id") != ID_PREFIX + str(source_id):
            problems.append(f"quarantine_id_namespace:{line_number}")
        if row.get("reason") != QUARANTINE_REASON or row.get("windowed_applied") is not False:
            problems.append(f"quarantine_shape:{line_number}")
        if row.get("windowed_limit_chars") != SR.MAX_TEXT_CHARS:
            problems.append(f"quarantine_cap:{line_number}")
        if source_id in quarantine_split or source_id in content:
            problems.append(f"quarantine_duplicate:{source_id}")
        quarantine_split[source_id] = row.get("split")
        quarantine_counts[row.get("split")] += 1
        justified = False
        full = row.get("full") if isinstance(row.get("full"), dict) else {}
        for field in ("premise", "hypothesis"):
            block = full.get(field) if isinstance(full.get(field), dict) else {}
            text = block.get("text")
            if not isinstance(text, str) or _text_sha256(text) != block.get("sha256") or len(text) != block.get("chars"):
                problems.append(f"quarantine_original:{line_number}:{field}")
                continue
            if len(text) > SR.MAX_TEXT_CHARS:
                justified = True
        if not justified:
            problems.append(f"quarantine_reason_unjustified:{source_id}")
        quarantine_projection[source_id] = {"split": row.get("split"), "label": row.get("label"),
                                            "root_id": row.get("root_id"), "cluster_id": row.get("cluster_id"),
                                            "premise": _raw_text_block(_block_text(full, "premise")),
                                            "hypothesis": _raw_text_block(_block_text(full, "hypothesis"))}

    source_rows = 0
    source_ids = set()
    signatures = []
    units_lookup = units if units_ready else None
    for line_number, record in _iter_jsonl_safe(source_dir / "pool.jsonl", problems, "source"):
        if not isinstance(record, dict):
            problems.append(f"source_row_shape:{line_number}")
            continue
        source_rows += 1
        source_id = record.get("id")
        if not isinstance(source_id, str) or not source_id:
            problems.append(f"source_row_id:{line_number}")
            continue
        source_ids.add(source_id)
        expected_split = None
        if membership_ready:
            expected_split = "train" if source_id in fit_set else ("dev" if source_id in dev_set else None)
        try:
            pair = SR.validate_semantic_pair(_pair_candidate(record))
        except SR.SemanticRelationError as error:
            if _length_exceeded(record):
                projection = quarantine_projection.get(source_id)
                if projection is None:
                    problems.append(f"missing_quarantine:{source_id}")
                else:
                    _reconcile_quarantine(problems, record, projection, units_lookup, expected_split)
                signatures.append(_source_signature(record, expected_split, units_lookup))
            else:
                problems.append(f"source_row_invalid:{source_id}:{error.code}")
            continue
        if source_id in quarantine_split:
            problems.append(f"quarantine_for_valid_row:{source_id}")
            continue
        if source_id not in content:
            problems.append(f"missing_trainable:{source_id}")
            continue
        split, label, digest = content[source_id]
        if expected_split is not None and split != expected_split:
            problems.append(f"row_split_mismatch:{source_id}")
        if record.get("gold_label") != label:
            problems.append(f"row_label_mismatch:{source_id}")
        if _pair_digest(pair) != digest:
            problems.append(f"row_pair_mismatch:{source_id}")

    output_ids = set(content) | set(quarantine_split)
    missing_ids = sorted(source_ids - output_ids)
    extra_ids = sorted(output_ids - source_ids)
    if missing_ids:
        problems.append(f"source_ids_missing_from_outputs:{len(missing_ids)}:{missing_ids[:5]}")
    if extra_ids:
        problems.append(f"output_ids_not_in_source:{len(extra_ids)}:{extra_ids[:5]}")
    if source_rows and (len(content) + len(quarantine_split)) != source_rows:
        problems.append("conservation_drift")
    if set(content) & set(quarantine_split):
        problems.append("output_overlap")
    if membership_ready:
        for split, members in (("train", fit_set), ("dev", dev_set)):
            produced = trainable_ids[split] | {sid for sid, value in quarantine_split.items() if value == split}
            if produced != members:
                problems.append(f"member_conservation:{split}")
    expected_counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    trainable_expected = (expected_counts.get("trainable") or {}).get("total")
    quarantine_expected = (expected_counts.get("quarantine") or {}).get("total")
    if trainable_expected is not None and counts["train"] + counts["dev"] != trainable_expected:
        problems.append("count_drift:trainable")
    if quarantine_expected is not None and quarantine_counts["train"] + quarantine_counts["dev"] != quarantine_expected:
        problems.append("count_drift:quarantine")
    lineage = None
    try:
        lineage = SM.producer_lineage(ownrun_dir / ASSETS_FILE, check_rows)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        problems.append(f"producer_membership:{error}")
    declared = manifest.get("signatures")
    if isinstance(declared, dict) and isinstance(declared.get("file"), str):
        try:
            declared_name = _relative_name(declared["file"])
        except SemanticPoolError:
            declared_name = None
        if declared_name is not None:
            declared_path = ownrun_dir / declared_name
            if not declared_path.is_file():
                problems.append(f"signatures_missing:{declared_name}")
            elif FT.file_sha256(declared_path) != declared.get("sha256"):
                problems.append(f"signatures_hash_drift:{declared_name}")
            elif declared.get("rows") != len(signatures):
                problems.append(f"signatures_rows_mismatch:{declared.get('rows')}!={len(signatures)}")
    return {"problems": problems, "lineage": lineage, "source_hashes": source_hashes,
            "counts": {"trainable": counts["train"] + counts["dev"], "quarantine": quarantine_counts["train"] + quarantine_counts["dev"],
                       "train": counts["train"], "dev": counts["dev"],
                       "quarantine_train": quarantine_counts["train"], "quarantine_dev": quarantine_counts["dev"]},
            "source_rows": source_rows, "signatures": signatures}


def _read_json_safe(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_member_safe(roles: dict, role: str, base_dir: Path, problems: list) -> list:
    if role not in roles:
        return []
    path = base_dir / roles[role]["path"]
    try:
        return [value for _, value in _iter_jsonl(path)]
    except (OSError, json.JSONDecodeError):
        problems.append(f"member_asset_unreadable:{role}")
        return []


def convert(source_dir, out_dir, *, expected: "dict | None" = None, cli: str = "") -> dict:
    """机械转换受保护公开池为 canonical JSONL；任何失败撤回本调用新建的输出目录。"""
    source_dir = Path(source_dir).absolute()
    out_dir = Path(out_dir)
    table = EXPECTED_SOURCE_SHA256 if expected is None else dict(expected)
    _assert_fresh(out_dir)
    source_hashes, plan, plan_sha = _verify_sources(source_dir, table)
    units = _load_units(source_dir / "units.jsonl")
    fit_ids = _load_member_ids(source_dir / "fit-ids.jsonl", "protected-fit")
    dev_ids = _load_member_ids(source_dir / "dev-ids.jsonl", "protected-dev")
    fit_set, dev_set = set(fit_ids), set(dev_ids)
    if fit_set & dev_set:
        raise SemanticPoolError("member_overlap", "protected-fit 与 protected-dev 成员重叠")
    pool_sha = source_hashes.get("pool.jsonl") or FT.file_sha256(source_dir / "pool.jsonl")

    out_dir.mkdir(parents=True)
    try:
        assets = []
        for role, source_name, target_name in ROLE_ASSETS:
            shutil.copyfile(source_dir / source_name, out_dir / target_name)
            assets.append({"role": role, "path": target_name,
                           "sha256": FT.file_sha256(out_dir / target_name),
                           "source_file": source_name, "source_sha256": source_hashes.get(source_name)})
        assets_document = {
            "schema": ASSETS_SCHEMA,
            "producer": PRODUCER,
            "note": "protected 成员与保护清单的字节级副本；SHA 即副本哈希（LF 源，Git 行尾策略不变字节）。",
            "assets": assets,
        }
        _write_json(out_dir / ASSETS_FILE, assets_document)

        trainable_rows = quarantine_rows = 0
        accepted = Counter()
        quarantined = Counter()
        seen_ids = set()
        seen_fit = seen_dev = 0
        root_split, cluster_split = {}, {}
        labels, datasets, gold_status = {}, {}, {}
        premise_lengths, hypothesis_lengths, pair_lengths = [], [], []
        quarantine_examples, quarantine_datasets, quarantine_labels = [], Counter(), Counter()
        check_rows, quarantine_ids = [], []
        members_sha = hashlib.sha256()

        with open(out_dir / TRAINABLE_FILE, "w", encoding="utf-8", newline="\n") as trainable, \
                open(out_dir / QUARANTINE_FILE, "w", encoding="utf-8", newline="\n") as quarantine:
            for _, record in _iter_jsonl(source_dir / "pool.jsonl"):
                if not isinstance(record, dict):
                    raise SemanticPoolError("bad_record", "pool 行不是 JSON 对象")
                source_id = record.get("id")
                if type(source_id) is not str or not source_id:
                    raise SemanticPoolError("bad_record", "pool 行缺少非空 id")
                if source_id in seen_ids:
                    raise SemanticPoolError("duplicate_source_id", f"pool 重复 id：{source_id}")
                seen_ids.add(source_id)
                if source_id in fit_set:
                    split, seen_fit = "train", seen_fit + 1
                elif source_id in dev_set:
                    split, seen_dev = "dev", seen_dev + 1
                else:
                    raise SemanticPoolError("pool_member_mismatch", f"pool 行不在 fit/dev 成员集：{source_id}")
                unit = units.get(source_id)
                if unit is None:
                    raise SemanticPoolError("missing_unit", f"记录不在依赖单位：{source_id}")
                prior = root_split.setdefault(unit["unit_id"], split)
                if prior != split:
                    raise SemanticPoolError("cross_split_root", f"root 跨 split：{unit['unit_id']}")
                prior = cluster_split.setdefault(unit["cluster_id"], split)
                if prior != split:
                    raise SemanticPoolError("cross_split_cluster", f"cluster 跨 split：{unit['cluster_id']}")
                if record.get("task") != LABEL_SPACE:
                    raise SemanticPoolError("task_not_nli3", f"{source_id} task!=nli3")
                label = record.get("gold_label")
                if type(label) is not str or label not in SR.NLI_CLASSES:
                    raise SemanticPoolError("invalid_gold_label", f"{source_id} gold_label={label!r}")
                role = SPLIT_ROLE[split]
                row = {
                    "id": ID_PREFIX + source_id,
                    "source_id": source_id,
                    "root_id": unit["unit_id"],
                    "cluster_id": unit["cluster_id"],
                    "split": split,
                    "label_space": LABEL_SPACE,
                    "label": label,
                    "label_origin": LABEL_ORIGIN,
                    "pair": _pair_candidate(record),
                }
                try:
                    row = SM.validate_row(row)
                except SR.SemanticRelationError as error:
                    if not _length_exceeded(record):
                        raise SemanticPoolError("unexpected_row_failure", f"{source_id}: {error.code} {error.hint}") from None
                    quarantine_row = _quarantine_row(record, split=split, unit=unit, role=role,
                                                     pool_sha=pool_sha, error=error)
                    _write_line(quarantine, quarantine_row)
                    quarantine_rows += 1
                    quarantined[split] += 1
                    quarantine_datasets[record.get("dataset")] += 1
                    quarantine_labels[label] += 1
                    quarantine_ids.append(source_id)
                    if len(quarantine_examples) < 5:
                        quarantine_examples.append({
                            "source_id": source_id, "split": split, "dataset": record.get("dataset"),
                            "premise_chars": len(record.get("premise") or ""),
                            "hypothesis_chars": len(record.get("hypothesis") or "")})
                    continue
                row["lineage"] = _lineage(record, role=role, pool_sha=pool_sha)
                _write_line(trainable, row)
                trainable_rows += 1
                accepted[split] += 1
                labels[split] = labels.get(split) or Counter()
                labels[split][label] += 1
                datasets[split] = datasets.get(split) or Counter()
                datasets[split][record.get("dataset")] += 1
                metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
                gold_status[split] = gold_status.get(split) or Counter()
                gold_status[split][metadata.get("gold_status")] += 1
                premise_lengths.append(len(row["pair"]["old"]["text"]))
                hypothesis_lengths.append(len(row["pair"]["new"]["text"]))
                pair_lengths.append(len(json.dumps(row["pair"], ensure_ascii=False, separators=(",", ":"))))
                check_rows.append({"id": row["id"], "source_id": source_id, "split": split,
                                   "label_space": LABEL_SPACE, "root_id": row["root_id"],
                                   "cluster_id": row["cluster_id"]})
                members_sha.update(f"{source_id}\x1f{split}\n".encode("utf-8"))

        if seen_fit != len(fit_set) or seen_dev != len(dev_set):
            raise SemanticPoolError("missing_pool_member",
                                    f"成员未被 pool 覆盖：fit {seen_fit}/{len(fit_set)}，dev {seen_dev}/{len(dev_set)}")
        total = trainable_rows + quarantine_rows
        if total != len(seen_ids):
            raise SemanticPoolError("count_mismatch", "输出计数与输入成员不守恒")
        if accepted["train"] + quarantined["train"] != len(fit_set) or accepted["dev"] + quarantined["dev"] != len(dev_set):
            raise SemanticPoolError("count_mismatch", "逐 split 分母不守恒（accepted + quarantine != members）")

        lineage = SM.producer_lineage(out_dir / ASSETS_FILE, check_rows)
        data_names = (TRAINABLE_FILE, QUARANTINE_FILE)
        outputs = {name: {"rows": (trainable_rows if name == TRAINABLE_FILE else quarantine_rows),
                          "bytes": (out_dir / name).stat().st_size,
                          "sha256": FT.file_sha256(out_dir / name)} for name in data_names}
        outputs[ASSETS_FILE] = {"rows": len(assets), "bytes": (out_dir / ASSETS_FILE).stat().st_size,
                                "sha256": FT.file_sha256(out_dir / ASSETS_FILE)}
        for _, _, target_name in ROLE_ASSETS:
            text = (out_dir / target_name).read_text(encoding="utf-8")
            outputs[target_name] = {"rows": sum(1 for line in text.splitlines() if line.strip()),
                                    "bytes": (out_dir / target_name).stat().st_size,
                                    "sha256": FT.file_sha256(out_dir / target_name)}
        manifest = {
            "schema": CONVERTER_SCHEMA,
            "manifest_revision": 1,
            "producer": PRODUCER,
            "created_at": _now(),
            "cli": cli,
            "tool": {name: FT.file_sha256(path) for name, path in {
                "semantic_pool.py": Path(__file__),
                "semantic_relations.py": Path(SR.__file__),
                "semantic_model.py": Path(SM.__file__),
                "finetune.py": Path(FT.__file__),
                "nli_reference.py": Path(NR.__file__),
            }.items()},
            "source": {
                "run_dir": str(source_dir),
                "hashes": source_hashes,
                "plan_sha256": plan_sha,
                "plan_version": (plan or {}).get("plan_version"),
            },
            "source_hashes_after": {name: FT.file_sha256(source_dir / name) for name in source_hashes},
            "label_space": LABEL_SPACE,
            "label_origin": LABEL_ORIGIN,
            "id_prefix": ID_PREFIX,
            "cap": {"chars": SR.MAX_TEXT_CHARS, "source": "semantic_relations.MAX_TEXT_CHARS",
                    "windowed_applied": False, "knob": "removed",
                    "note": "唯一硬界来自 stdlib 契约；不接受可调 cap，超限行不截断、不丢文档，保全原文后整行 quarantine。"},
            "outputs": outputs,
            "counts": {
                "input": {"pool": len(seen_ids), "fit": len(fit_set), "dev": len(dev_set)},
                "trainable": {"train": accepted["train"], "dev": accepted["dev"], "total": trainable_rows},
                "quarantine": {"train": quarantined["train"], "dev": quarantined["dev"], "total": quarantine_rows,
                               "reason": "source_too_large", "by_dataset": dict(quarantine_datasets),
                               "by_label": dict(quarantine_labels), "examples": quarantine_examples},
                "conservation": {
                    "fit": accepted["train"] + quarantined["train"], "dev": accepted["dev"] + quarantined["dev"],
                    "pool": trainable_rows + quarantine_rows, "checked": True},
            },
            "checks": {
                "duplicate_ids": 0, "member_overlap": 0, "fit_dev_root_overlap": 0, "fit_dev_cluster_overlap": 0,
                "units_one_cluster_each": True, "units_split_pure": True,
                "labels": {split: dict(labels.get(split) or {}) for split in ("train", "dev")},
                "datasets": {split: dict(datasets.get(split) or {}) for split in ("train", "dev")},
                "gold_status": {split: dict(gold_status.get(split) or {}) for split in ("train", "dev")},
                "producer_membership_verified": True,
            },
            "char_strata": {
                "premise": _strata(premise_lengths),
                "hypothesis": _strata(hypothesis_lengths),
                "pair_json": _strata(pair_lengths),
            },
            "member_ids_sha256": members_sha.hexdigest(),
            "producer_lineage": lineage,
        }
        shape_problems = _manifest_problems(manifest, ownrun_dir=out_dir)
        if shape_problems:
            raise SemanticPoolError("manifest_shape", f"新鲜 manifest 未过形状锚点：{shape_problems}")
        _write_json(out_dir / LEGACY_MANIFEST_FILE, manifest)
        after = {name: FT.file_sha256(source_dir / name) for name in source_hashes}
        if after != source_hashes:
            raise SemanticPoolError("source_mutated", "转换过程改动了源工件（只读保证失败）")
        return manifest
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise


def _revision_path(ownrun_dir: Path, revision: int) -> Path:
    if revision == 1:
        return ownrun_dir / LEGACY_MANIFEST_FILE
    return ownrun_dir / MANIFEST_FILE_TEMPLATE.format(revision=revision)


def _active_manifest(ownrun_dir: Path) -> Path:
    best = 1
    for revision in range(2, 64):
        if _revision_path(ownrun_dir, revision).is_file():
            best = revision
    return _revision_path(ownrun_dir, best)


def verify(ownrun_dir, source_dir, *, signatures_out=None) -> dict:
    """从一个 manifest 锚点重派生并重验：形状围栏、输出/源哈希、逐行消费者契约、
    id/member/split/label/pair/quarantine reason/units 与守恒（不信任 manifest 旗标）；
    quarantine 行逐条与实际源 record 核（文本 sha/字符、label、unit/root/cluster、split、why-too-large）。"""
    ownrun_dir = Path(ownrun_dir).absolute()
    source_dir = Path(source_dir).absolute()
    manifest_file = _active_manifest(ownrun_dir)
    manifest = _read_json_safe(manifest_file)
    if not isinstance(manifest, dict) or not manifest:
        return {"schema": CONVERTER_SCHEMA, "manifest": manifest_file.name,
                "problems": ["manifest_unreadable"], "counts": {}}
    scan = _scan(ownrun_dir, source_dir, manifest)
    result = {"schema": CONVERTER_SCHEMA, "manifest": manifest_file.name, "problems": scan["problems"],
              "counts": scan["counts"], "source_rows": scan["source_rows"],
              "signatures_rows": len(scan["signatures"])}
    if signatures_out:
        result["signatures_sha256"] = _write_signatures(Path(signatures_out), scan["signatures"])
    return result


def _write_signatures(path: Path, signatures: list) -> str:
    if path.exists():
        existing = _read_signatures(path)
        if existing != signatures:
            raise SemanticPoolError("signatures_drift", f"已有 signatures 文件内容不同，拒绝覆盖：{path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in signatures:
                _write_line(handle, row)
    return FT.file_sha256(path)


def _read_signatures(path: Path) -> list:
    return [value for _, value in _iter_jsonl(path)]


def reanchor(ownrun_dir, source_dir, *, cli_verify: "str | None" = None) -> dict:
    """生成下一版本 manifest：旧版本与旧原件一律保留；新版本标 readback/hashes、
    当前路径可重放命令与 sourceTooLong 成员 raw canonical projection signatures。"""
    ownrun_dir = Path(ownrun_dir).absolute()
    source_dir = Path(source_dir).absolute()
    _assert_plain_path(ownrun_dir, code="input_reparse")
    current = _active_manifest(ownrun_dir)
    if not current.is_file():
        raise SemanticPoolError("legacy_manifest_missing", f"缺少 manifest：{current}")
    current_revision = 1 if current.name == LEGACY_MANIFEST_FILE else int(current.stem.rsplit("-v", 1)[1])
    next_revision = current_revision + 1
    target = _revision_path(ownrun_dir, next_revision)
    if target.exists():
        raise SemanticPoolError("revision_exists", f"新版本已存在，拒绝覆盖：{target}")
    current_sha = FT.file_sha256(current)
    manifest = json.loads(current.read_text(encoding="utf-8"))
    scan = _scan(ownrun_dir, source_dir, manifest)
    if scan["problems"]:
        raise SemanticPoolError("reanchor_blocked", f"当前 manifest 未通过重派生：{scan['problems'][:8]}")
    signatures_sha = _write_signatures(ownrun_dir / SIGNATURES_FILE, scan["signatures"])
    verify_cli = cli_verify or (f"python -m research.memory_relations.semantic_pool verify "
                                f"--ownrun {ownrun_dir} --source-run {source_dir}")
    revision = dict(manifest)
    history = dict(revision.get("cli_history") or {})
    history.update({
        "historical_generation_cli": revision.get("cli"),
        "status": "已退休：生成时 out 目录为顶层 ownruns/（按父 scope 裁定移除）；仅作历史执行记录，不是当前入口",
    })
    revision.update({
        "manifest_revision": next_revision,
        "supersedes": {"file": current.name, "sha256": current_sha},
        "revised_at": _now(),
        "cli": verify_cli,
        "cli_history": history,
        "regenerate_cli": (f"python -m research.memory_relations.semantic_pool convert "
                           f"--source-run {source_dir} --out <fresh dir>"),
        "relocation": {"note": "产物整体 mv 到当前 run 目录，字节未变；本版本重锚 readback 哈希",
                       "run_dir": str(ownrun_dir)},
        "producer_lineage": scan["lineage"],
        "signatures": {"file": SIGNATURES_FILE, "sha256": signatures_sha,
                       "rows": len(scan["signatures"]),
                       "note": "sourceTooLong 成员的 raw canonical projection signature（源文本 sha/字符、label、unit、"
                               "split、why-too-large），verify 逐条与实际源 record 核"},
        "readback": {"checked_at": _now(),
                     "outputs": {name: FT.file_sha256(ownrun_dir / name) for name in manifest.get("outputs", {})},
                     "source": scan["source_hashes"],
                     "superseded_manifest_sha256": current_sha,
                     "data_unchanged": True},
    })
    _write_json(target, revision)
    rescan = _scan(ownrun_dir, source_dir, revision)
    if rescan["problems"]:
        raise SemanticPoolError("revision_verify_failed", f"新版本 manifest 复验失败：{rescan['problems'][:8]}")
    return {"schema": CONVERTER_SCHEMA, "manifest": target.name, "revision": next_revision,
            "supersedes": current_sha, "signatures": {"file": SIGNATURES_FILE, "sha256": signatures_sha,
                                                      "rows": len(scan["signatures"])},
            "problems": [], "counts": rescan["counts"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    converter = sub.add_parser("convert")
    converter.add_argument("--source-run", required=True)
    converter.add_argument("--out", required=True)
    converter.add_argument("--expected-json", help="可选的 filename→sha256 JSON 覆写表（默认冻结 M14 表）")
    checker = sub.add_parser("verify")
    checker.add_argument("--ownrun", required=True)
    checker.add_argument("--source-run", required=True)
    checker.add_argument("--signatures-out", help="可选：把重算的 sourceTooLong raw canonical projection signatures 写入该路径")
    anchor = sub.add_parser("reanchor")
    anchor.add_argument("--ownrun", required=True)
    anchor.add_argument("--source-run", required=True)
    return parser


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    if args.command == "convert":
        expected = None
        if args.expected_json:
            expected = json.loads(Path(args.expected_json).read_text(encoding="utf-8"))
        cli = (f"python -m research.memory_relations.semantic_pool convert "
               f"--source-run {Path(args.source_run).absolute()} --out {Path(args.out).absolute()}")
        manifest = convert(args.source_run, args.out, expected=expected, cli=cli)
        return {"schema": manifest["schema"], "outputs": {name: item["sha256"] for name, item in manifest["outputs"].items()},
                "counts": manifest["counts"], "char_strata": manifest["char_strata"]}
    if args.command == "reanchor":
        return reanchor(args.ownrun, args.source_run)
    return verify(args.ownrun, args.source_run, signatures_out=args.signatures_out)


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result.get("problems") else 0)
