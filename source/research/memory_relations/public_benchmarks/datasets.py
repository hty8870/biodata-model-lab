                       
"""公开 NLI 基准数据集的唯一获取/解析/规范化机制（研究内部件；纯标准库、不触模型）。

一条机制 + 数据源配置（`sources.json`）接入 OCNLI / ANLI / HANS / ContractNLI。四阶段：

- `fetch`：https 白名单（**每一次重定向在发起请求前**校验 scheme/hostname 与跳数预算）、
  超时、声明/实际体积闸、HTML 错误页与魔数检测、有限重试且每次尝试（含失败）入账本；
- `extract`：zip 白名单解包。先整体预检（白名单与 central directory 成员唯一性、Windows
  名称别名/ADS/尾点尾空格/设备名含上标数字；该预检与配置校验共用同一 `_entry_alias_map`，
  公共 API 直接调用也不能绕过），**任何输出（含输出根目录）创建前**完成，再逐条落盘；
  落盘使用**同目录独占新建**的 `.part`（O_CREAT|O_EXCL，拒既存/链接/hardlink），写完复核
  字节哈希；坏 ZIP/CRC、读写 I/O、Unicode 与预检失败统一归类（`bad_zip`/`io_error`/
  `unicode_error`/…）为有界事件账本后原样抛出；
- `normalize`：从**已钉定 zip** 重新流式派生白名单成员 SHA/bytes，与磁盘 extracted 逐项
  校验（drift 直接失败），再按格式适配器产出统一 record；所有派生路径（配置 root/dest/
  输出/中间文件）统一受 run 边界守卫，越界、链接/reparse、hardlink 目标一律拒绝；
- `verify`：**真只读**——复用同一套规范化迭代/计数（`out_dir=None` 的纯摘要模式），
  不写任何产物、不建/删 run 内目录（含未知 `_verify-tmp` 原样保留）、不追加源账本；
  对**全部承诺字段**（集合/来源/路径/SHA/计数/诊断/catalog）重算比对，任何伪造返回问题。
- 写入能力：事件账本一律经显式 sink 对象（`_EventLedger`，见下）落盘。fetch/extract/
  normalize 的**成功、解析失败、前置失败与 I/O 失败**全部走这一接缝；只读摘要与 verify
  的嵌套路径（含 `normalize_split` 的异常分支）不持有 sink，任何失败路径也无处落盘。
- run 边界：写路径先验证**run 基准根本身与其每个已存在祖先分量**为普通目录（逐级 lstat，
  拒链接/reparse，不用 realpath 解析掉证据），再验证全部派生分量；所有 mkdir/`.part`/
  账本写入前完成根检查。属性读取经唯一 helper `_lstat_or_missing`：**仅真实
  `FileNotFoundError`(ENOENT) 可视为缺失分量**，其余 stat I/O 错误（EACCES/ELOOP/未知）
  一律 fail-closed 抛 `unreadable_path`，任何写入前终止。

统一 record 严格对齐 M7 `protocol.py` 记录合同（顶层 10 键、键序固定、行 JSON 确定性）：
    id / dataset / official_split / task / premise / hypothesis /
    gold_label / group_id / metadata / evidence
- `gold_label=null` ⇔ `metadata.gold_status ∈ {no_consensus, missing}`；缺标/无共识行必须
  保留，单列统计，不补标、不伪标。
- `metadata.original_label` 逐字保留源标注并只经 `sources.json.label_map` 映射
  （与 `protocol.NATIVE_LABEL_MAPS` 的非 null 子集一致；官方无共识 marker `'-'` 显式映射
  null，不是未知标签）。
- `group_id` 是数据集级独立单位值本身（无前缀）：ocnli=真实 prem_id（官方无 prem_id 时
  **保留 null**，group 用 `premise_family_key` 派生并以 `metadata.group_id_source=
  "derived_premise_family"` 声明，绝不填 hash 冒充官方单位）、anli=`premise_family_key`
  （NFKC+空白折叠+sha256 前 16 位，逐式镜像 protocol.py）、hans=template、contractnli=doc_id。
- 标签来源如实：ocnli 按有效票对官方 gold 的**严格多数**判 majority（有效票 ≥2 且
  `2×同意票 > 有效票`；`unknown`/`unrelated`/null 不算有效票，五字段均缺即 single），
  有效票 ≥2 但无严格多数时如实降为 single+`label_method=no_majority`（不升级、不改官方
  gold、票原样保留），`'-'`→no_consensus——判据唯一锚点是纯函数 `ocnli_vote_status`
  （生产者/消费方共用，含 votes_valid_count/votes_agree_count 复算计数）；anli 按官方复核票
  实际命中（≥2 票等于 gold 才记 adjudicated，其余 writer_only）；hans 模板规则生成记
  `generated`；contractnli 单一标注记 `single`。
- `evidence={source_url,file,sha256,row}`；Contract gold 跨度在 `metadata.spans`（oracle 用）。
- metadata/evidence 有界（16 KiB / 深度 6）可序列化。
"""
from __future__ import annotations

import argparse
import copy
import csv
import functools
import hashlib
import json
import os
import re
import shutil
import stat
import time
import unicodedata
import urllib.error
import urllib.request
import zipfile
import zlib
from collections import namedtuple
from pathlib import Path

__all__ = [
    "GOLD_LABELS", "GOLD_STATUSES", "NULL_GOLD_STATUSES", "RECORD_KEYS", "TASK_LABELS",
    "UNIT_SOURCES", "DatasetError", "build_record", "default_opener", "extract_archive",
    "fetch_file", "load_config", "main", "normalize_all", "ocnli_vote_status",
    "premise_family_key", "stage_extract", "stage_fetch", "stage_normalize", "stage_verify",
    "validate_config", "verify_run",
]

CONFIG_PATH = Path(__file__).resolve().with_name("sources.json")
RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"

RECORD_KEYS = ("id", "dataset", "official_split", "task", "premise", "hypothesis",
               "gold_label", "group_id", "metadata", "evidence")
TASK_LABELS = {
    "nli3": frozenset({"entailment", "contradiction", "neutral"}),
    "entailment2": frozenset({"entailment", "non-entailment"}),
}
GOLD_LABELS = frozenset().union(*TASK_LABELS.values())
GOLD_STATUSES = frozenset({"adjudicated", "majority", "single", "no_consensus", "missing",
                           "generated"})
NULL_GOLD_STATUSES = frozenset({"no_consensus", "missing"})
UNIT_SOURCES = frozenset({"raw", "derived_premise_family"})
EXPECT_KINDS = frozenset({"zip", "jsonl", "text"})
FILE_KINDS = frozenset({"direct", "archive"})
GROUP_KINDS = frozenset({"field", "premise_family_key"})
GROUP_FALLBACKS = (None, "premise_family_key", "self")
_JOIN_FORMATS = frozenset({"jsonl_keyed"})
_CHUNK = 1 << 16
_MAX_NAME = 512
_MAX_REFS = 50
_MAX_DIFFS = 40
_MAX_TEXT = 500000
_METADATA_MAX_BYTES = 16384
_EVIDENCE_MAX_BYTES = 16384
_MAX_DEPTH = 6
_MAX_REDIRECTS = 5
_DEVICE_NAMES = (frozenset({"CON", "PRN", "AUX", "NUL"})
                 | {"COM%d" % n for n in range(0, 10)}
                 | {"LPT%d" % n for n in range(0, 10)})
_SUPERSCRIPT_DIGITS = {"\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3",
                       "\u2074": "4", "\u2075": "5", "\u2076": "6", "\u2077": "7",
                       "\u2078": "8", "\u2079": "9"}
_USER_AGENT = "biodata-agent-research/1.0 (public benchmark acquisition; stdlib urllib)"
OpenResult = namedtuple("OpenResult", "status headers reader final_url")


class DatasetError(RuntimeError):
    """有界机器码错误；`detail` 只用于账本/日志，不回显无界数据。"""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = _bounded(detail)
        super().__init__(self.code if not self.detail else "%s: %s" % (self.code, self.detail))


def _bounded(value: object, limit: int = 200) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _ledger_append(path: Path, event: dict) -> None:
    """账本只追加：失败/重试/跳过全部保留，不因重跑抹掉历史。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": _utc_now(), **event}, ensure_ascii=False, sort_keys=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


class _EventLedger:
    """显式事件账本写入能力：只读路径不持有 sink（None），所有嵌套写点共享同一接缝。

    每次追加前把 run 根链与账本路径交给同一路径守卫，run 根本身为链接/reparse 时
    账本也无从写到 run 外（守卫先拒绝，不做任何写入）。
    """

    def __init__(self, run_dir: Path, path: Path) -> None:
        self._run_dir = Path(run_dir)
        self.path = Path(path)

    def append(self, event: dict) -> None:
        _assert_plain_chain(self._run_dir, "ledger")
        _assert_plain_path(self._run_dir, self.path, "ledger")
        _ledger_append(self.path, event)


def _ledger_sink(run_dir: Path, ledger: object) -> _EventLedger:
    """把 Path / 既有 sink 统一为写入能力对象（mutating 入口与直接调用方透传）。"""
    if isinstance(ledger, _EventLedger):
        return ledger
    return _EventLedger(run_dir, Path(ledger))


def _classify_error(exc: BaseException) -> tuple:
    """阶段异常归类为有界机器码；原异常保留（由调用方链式抛出或原样 raise）。"""
    if isinstance(exc, DatasetError):
        return exc.code, exc.detail
    if isinstance(exc, (zipfile.BadZipFile, zlib.error)):
        return "bad_zip", _bounded(exc)
    if isinstance(exc, UnicodeError):
        return "unicode_error", _bounded(exc)
    if isinstance(exc, OSError):
        return "io_error", _bounded(exc)
    return type(exc).__name__, _bounded(exc)


def premise_family_key(premise: str) -> str:
    """前置文本的规范族键：NFKC + 空白折叠 + sha256 前 16 位（镜像 protocol.py）。"""
    if not isinstance(premise, str) or not premise:
        raise DatasetError("empty_premise")
    normalized = " ".join(unicodedata.normalize("NFKC", premise).split())
    return "premise:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _depth(value: object) -> int:
    if isinstance(value, dict):
        return 1 + max((_depth(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_depth(item) for item in value), default=0)
    return 0


def _check_jsonable(value: object, code: str, *, max_bytes: int) -> None:
    if _depth(value) > _MAX_DEPTH:
        raise DatasetError(code)
    try:
        serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        raise DatasetError(code) from None
    if len(serialized.encode("utf-8")) > max_bytes:
        raise DatasetError(code)


                                                                              

def _run_relative(value: object, what: str) -> str:
    """run 内相对路径守卫：拒绝对路径、驱动盘、反斜杠、`.`/`..`/空段、冒号与尾点空格。"""
    if not isinstance(value, str) or not value or len(value) > _MAX_NAME:
        raise DatasetError("bad_relpath", what)
    if value.startswith(("/", "\\")) or "\\" in value or re.match(r"^[A-Za-z]:", value):
        raise DatasetError("bad_relpath", what)
    for part in value.split("/"):
        if part in ("", ".", "..") or ":" in part or part.rstrip(" .") != part:
            raise DatasetError("bad_relpath", what)
    return value


def _link_like_info(info: os.stat_result) -> bool:
    """对已取到的 lstat 结果判定链接/reparse：symlink 及 Windows junction/其它 reparse 点。"""
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _lstat_or_missing(path: Path) -> "os.stat_result | None":
    """边界属性读取的**唯一**接缝：仅 `FileNotFoundError`（ENOENT）视为可创建的缺失分量。

    其余 `OSError`（EACCES/ELOOP/磁盘或未知 I/O 故障）一律 fail-closed 抛
    `unreadable_path`——『读不到属性』绝不等同『不存在』；run 根、祖先、派生 parent 与
    target 全部经这一 helper，任何 mkdir/`.part`/账本写入前终止，不用 realpath 绕守卫。
    """
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DatasetError("unreadable_path",
                           "%s:%s" % (type(exc).__name__, exc.errno)) from exc


def _assert_plain_chain(base: Path, what: str) -> None:
    """run 基准根本身与每个已存在祖先分量必须是普通目录（逐级单次 lstat，拒链接/reparse）。

    不从 realpath 解析（那会先把『链接根 + 子路径』的证据解析掉）；任何 mkdir、`.part`
    或账本写入前都必须先过这一闸。仅真实 ENOENT 视作缺失分量（由调用方随后 mkdir 创建）；
    属性读取失败按 `_lstat_or_missing` fail-closed。
    """
    current = Path(os.path.abspath(str(base)))
    while True:
        info = _lstat_or_missing(current)
        if info is not None and (_link_like_info(info) or not stat.S_ISDIR(info.st_mode)):
            raise DatasetError("unsafe_root", what)
        parent = current.parent
        if parent == current:
            return
        current = parent


def _assert_plain_path(base: Path, target: Path, what: str) -> None:
    """目标必须 lexical 位于 base 内且逐级无链接/reparse：不解析符号链接——realpath 会把
    『链接目录 + 子路径』解析回域外从而漏检——base 自身（run 根）与每个已存在派生分量、
    target 逐级单次 lstat 检查；已存普通目录可接受，已存文件拒链接/reparse 与 hardlink
    （nlink>1），其它类型拒绝。base 的祖先链由公共写入口的 `_assert_plain_chain` 前置验证。
    缺失判定与 I/O 失败一律走 `_lstat_or_missing`（仅 ENOENT 可缺失，其余 fail-closed）。"""
    base_abs = os.path.abspath(str(base))
    target_abs = os.path.abspath(str(target))
    if os.path.commonpath([target_abs, base_abs]) != base_abs:
        raise DatasetError("path_escape", what)
    base_info = _lstat_or_missing(base_abs)
    if base_info is not None and (_link_like_info(base_info)
                                  or not stat.S_ISDIR(base_info.st_mode)):
        raise DatasetError("unsafe_path", what)
    relative = Path(target_abs).relative_to(base_abs)
    current = Path(base_abs)
    for part in relative.parts[:-1]:
        current = current / part
        info = _lstat_or_missing(current)
        if info is None:
            continue
        if _link_like_info(info) or not stat.S_ISDIR(info.st_mode):
            raise DatasetError("unsafe_path", what)
    target_info = _lstat_or_missing(target)
    if target_info is None:
        return
    if _link_like_info(target_info):
        raise DatasetError("unsafe_path", what)
    if stat.S_ISDIR(target_info.st_mode):
        return
    if not stat.S_ISREG(target_info.st_mode):
        raise DatasetError("unsafe_path", what)
    if target_info.st_nlink > 1:
        raise DatasetError("hardlinked_target", what)


def _open_exclusive_part(target: Path, what: str):
    """同目录独占新建 `.part`（O_CREAT|O_EXCL）：既存文件/链接/hardlink 一律拒绝、不覆盖。"""
    part = target.with_name(target.name + ".part")
    try:
        descriptor = os.open(str(part), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    except FileExistsError:
        raise DatasetError("part_exists", what) from None
    except OSError as exc:
        raise DatasetError("part_unavailable", "%s:%s" % (what, exc.errno)) from None
    return part, os.fdopen(descriptor, "wb")


                                                                                    


def load_config(path: object = CONFIG_PATH) -> dict:
    """读取并校验数据源配置；任何结构/词表/路径问题 fail-closed。"""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: object) -> None:
    if type(config) is not dict or config.get("schema_version") != 1:
        raise DatasetError("bad_config")
    limits = config.get("limits")
    if type(limits) is not dict:
        raise DatasetError("bad_config")
    for key in ("timeout_seconds", "attempts", "retry_backoff_seconds",
                "max_entries", "max_entry_bytes", "max_total_uncompressed_bytes",
                "min_free_bytes"):
        value = limits.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise DatasetError("bad_config", "limits." + key)
    if not isinstance(config.get("default_run"), str) or not config["default_run"]:
        raise DatasetError("bad_config", "default_run")
    _run_relative(config["default_run"], "default_run")
    datasets = config.get("datasets")
    if type(datasets) is not dict or not datasets:
        raise DatasetError("bad_config", "datasets")
    for name, source in datasets.items():
        _validate_source(name, source)


def _validate_source(name: str, source: object) -> None:
    if type(source) is not dict:
        raise DatasetError("bad_config", name)
    for key in ("title", "home", "license", "task", "allowed_hosts", "label_map",
                "group_id", "unit_key", "files", "splits"):
        if key not in source:
            raise DatasetError("bad_config", "%s.%s" % (name, key))
    task = source["task"]
    if task not in TASK_LABELS:
        raise DatasetError("bad_config", "%s.task" % name)
    label_map = source["label_map"]
    if type(label_map) is not dict or not label_map:
        raise DatasetError("bad_config", "%s.label_map" % name)
    for raw, normalized in label_map.items():
        if type(raw) is not str or not raw:
            raise DatasetError("bad_config", "%s.label_map" % name)
        if normalized is not None and normalized not in TASK_LABELS[task]:
            raise DatasetError("bad_config", "%s.label_map:%s" % (name, raw))
    hosts = source["allowed_hosts"]
    if type(hosts) is not list or not hosts or any(
            type(host) is not str or not host or host != host.lower()
            or "/" in host or ":" in host for host in hosts):
        raise DatasetError("bad_config", "%s.allowed_hosts" % name)
    if type(source["license"]) is not dict or "name" not in source["license"] \
            or "url" not in source["license"]:
        raise DatasetError("bad_config", "%s.license" % name)
    if not isinstance(source["unit_key"], str) or not source["unit_key"]:
        raise DatasetError("bad_config", "%s.unit_key" % name)
    _validate_group(name, source["group_id"])
    roles = _validate_files(name, source)
    archives = [spec for spec in source["files"] if spec.get("kind", "direct") == "archive"]
    if len(archives) > 1:
        raise DatasetError("bad_config", "%s.files(archive>1)" % name)
    split_names = set()
    for split_name, split in source["splits"].items():
        if type(split) is not dict or not isinstance(split.get("official_split"), str) \
                or not split["official_split"] or not isinstance(split.get("format"), str):
            raise DatasetError("bad_config", "%s.splits.%s" % (name, split_name))
        if split_name in split_names or split["format"] not in PARSERS:
            raise DatasetError("bad_config", "%s.splits.%s" % (name, split_name))
        split_names.add(split_name)
        has_file, has_entry = "file_role" in split, "entry" in split
        if has_file == has_entry:
            raise DatasetError("bad_config", "%s.splits.%s" % (name, split_name))
        if has_file and split["file_role"] not in roles:
            raise DatasetError("bad_config", "%s.splits.%s.file_role" % (name, split_name))
        if has_entry:
            if not archives or not isinstance(split["entry"], str):
                raise DatasetError("bad_config", "%s.splits.%s.entry" % (name, split_name))
            entries = archives[0]["extract"]["entries"]
            if split["entry"] not in entries:
                raise DatasetError("bad_config", "%s.splits.%s.entry" % (name, split_name))
        crosscheck = split.get("crosscheck_role")
        if crosscheck is not None and crosscheck not in roles:
            raise DatasetError("bad_config", "%s.splits.%s.crosscheck_role" % (name, split_name))
    for join in source.get("joins", []):
        if type(join) is not dict or join.get("format") not in _JOIN_FORMATS \
                or join.get("role") not in roles \
                or not all(isinstance(join.get(key), str) and join[key]
                           for key in ("key_field", "value_field", "metadata_key")):
            raise DatasetError("bad_config", "%s.joins" % name)
    expected = source.get("expect_hypotheses")
    if expected is not None and (isinstance(expected, bool) or not isinstance(expected, int)
                                 or expected < 1):
        raise DatasetError("bad_config", "%s.expect_hypotheses" % name)


def _validate_group(name: str, rule: object) -> None:
    if type(rule) is not dict or rule.get("kind") not in GROUP_KINDS \
            or rule.get("fallback") not in GROUP_FALLBACKS:
        raise DatasetError("bad_config", "%s.group_id" % name)
    if rule["kind"] == "field" and (not isinstance(rule.get("field"), str)
                                    or not rule["field"]):
        raise DatasetError("bad_config", "%s.group_id.field" % name)


def _validate_files(name: str, source: dict) -> set:
    files = source["files"]
    if type(files) is not list or not files:
        raise DatasetError("bad_config", "%s.files" % name)
    roles = set()
    for spec in files:
        if type(spec) is not dict:
            raise DatasetError("bad_config", "%s.files" % name)
        role = spec.get("role")
        if not isinstance(role, str) or not role or role in roles:
            raise DatasetError("bad_config", "%s.files.role" % name)
        roles.add(role)
        kind = spec.get("kind", "direct")
        if kind not in FILE_KINDS:
            raise DatasetError("bad_config", "%s.files:%s" % (name, role))
        if not isinstance(spec.get("url"), str) or not spec["url"].startswith("https://"):
            raise DatasetError("bad_config", "%s.files:%s.url" % (name, role))
        _run_relative(spec.get("dest"), "%s.files:%s.dest" % (name, role))
        if not isinstance(spec.get("max_bytes"), int) or isinstance(spec["max_bytes"], bool) \
                or spec["max_bytes"] < 1:
            raise DatasetError("bad_config", "%s.files:%s.max_bytes" % (name, role))
        if spec.get("expect") not in EXPECT_KINDS:
            raise DatasetError("bad_config", "%s.files:%s.expect" % (name, role))
        pin = spec.get("sha256")
        if pin is not None and (not isinstance(pin, str) or not re.fullmatch(r"[0-9a-f]{64}", pin)):
            raise DatasetError("bad_config", "%s.files:%s.sha256" % (name, role))
        if kind == "archive":
            extract = spec.get("extract")
            if type(extract) is not dict or type(extract.get("entries")) is not list \
                    or not extract["entries"]:
                raise DatasetError("bad_config", "%s.files:%s.extract" % (name, role))
            _run_relative(extract.get("root"), "%s.files:%s.extract.root" % (name, role))
            try:
                _entry_alias_map(extract["entries"], "%s.files:%s.extract" % (name, role))
            except DatasetError as exc:
                if exc.code != "alias_entry":
                    raise
                raise DatasetError("bad_config", "%s.files:%s.extract.entries(alias)"
                                   % (name, role)) from None
    for join in source.get("joins", []):
        joined = [spec for spec in files if spec.get("role") == join.get("role")]
        if joined and joined[0].get("kind", "direct") != "direct":
            raise DatasetError("bad_config", "%s.joins:%s(direct only)" % (name, join.get("role")))
    return roles


                                                                                   


def _check_redirect_target(url: str, allowed_hosts) -> str:
    """重定向目标在**发起请求前**必须 schemes=https 且 host 在白名单；否则拒绝。"""
    allowed = list(allowed_hosts)
    _check_hosts(url=url, allowed=allowed, what="redirect")
    return url


class _GuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """每一跳重定向前校验 scheme/hostname（含禁域再跳回允许域的链），并限跳数。"""

    max_repeats = 2
    max_redirections = _MAX_REDIRECTS

    def __init__(self, allowed_hosts) -> None:
        self.allowed_hosts = tuple(allowed_hosts)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_redirect_target(newurl, self.allowed_hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _build_guarded_opener(allowed_hosts, extra_handlers=()):
    handlers = [_GuardedRedirectHandler(tuple(allowed_hosts))]
    handlers.extend(extra_handlers)
    return urllib.request.build_opener(*handlers)


def default_opener(url: str, timeout: float, *, allowed_hosts=None) -> OpenResult:
    """默认网络入口：https GET；重定向逐跳守卫；返回状态/头/可读流/最终 URL。"""
    hosts = tuple(allowed_hosts or ())
    if hosts:
        _check_hosts(url=url, allowed=list(hosts), what="request")
        opener = _build_guarded_opener(hosts)
    else:
        opener = urllib.request.build_opener()
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    response = opener.open(request, timeout=timeout)
    headers = {key.lower(): value for key, value in response.headers.items()}
    status = getattr(response, "status", None)
    if status is None:
        status = getattr(response, "code", None)
    return OpenResult(status, headers, response, response.geturl())


def _host_of(url: str) -> str:
    match = re.match(r"^https://([^/?#]+)", url)
    if not match:
        raise DatasetError("bad_url", _bounded(url, 120))
    host = match.group(1).lower()
    if "@" in host:
        raise DatasetError("bad_url", _bounded(url, 120))
    return host.split(":")[0]


def _check_hosts(*, url: str, allowed: list, what: str) -> None:
    if not url.startswith("https://"):
        raise DatasetError("bad_url", what)
    if _host_of(url) not in allowed:
        raise DatasetError("host_not_allowed", what)


def _sniff(head: bytes, expect: str, media_type: str) -> None:
    """错误页/魔数闸：文本类拒绝 HTML 壳，JSONL 必须以 `{`/`[` 开头，zip 必须 PK 头。"""
    if not head:
        raise DatasetError("empty_body")
    if expect == "zip":
        if not head.startswith(b"PK\x03\x04"):
            raise DatasetError("not_zip")
        return
    if "text/html" in media_type.lower():
        raise DatasetError("html_error_page")
    stripped = head.lstrip()[:64].lower()
    if stripped.startswith((b"<!doctype", b"<html", b"<head")):
        raise DatasetError("html_error_page")
    if expect == "jsonl" and stripped[:1] not in (b"{", b"["):
        raise DatasetError("not_jsonl")


def _download_once(spec: dict, dest: Path, *, allowed_hosts: list, timeout: float,
                   max_bytes: int, opener, run_dir: Path) -> dict:
    _check_hosts(url=spec["url"], allowed=allowed_hosts, what=spec["role"])
    opened = None
    try:
        try:
            opened = opener(spec["url"], timeout)
        except urllib.error.HTTPError as exc:
            raise DatasetError("http_status", "%s:%s" % (spec["role"], exc.code)) from None
        except DatasetError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DatasetError("network_error",
                               "%s:%s" % (spec["role"], type(exc).__name__)) from None
        status = int(opened.status)
        if status != 200:
            raise DatasetError("http_status", "%s:%s" % (spec["role"], status))
        headers = {str(key).lower(): value for key, value in (opened.headers or {}).items()}
        _check_hosts(url=str(opened.final_url), allowed=allowed_hosts,
                     what=spec["role"] + ":final")
        declared_bytes = None
        declared = headers.get("content-length")
        if declared is not None:
            try:
                declared_bytes = int(declared)
            except (TypeError, ValueError):
                declared_bytes = None
            if declared_bytes is not None and declared_bytes > max_bytes:
                raise DatasetError("oversize_declared", "%s:%s" % (spec["role"], declared_bytes))
        _assert_plain_path(run_dir, dest, "fetch:" + spec["role"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        part, handle = _open_exclusive_part(dest, "fetch:" + spec["role"])
        digest = hashlib.sha256()
        written = 0
        head = b""
        try:
            while True:
                try:
                    chunk = opened.reader.read(_CHUNK)
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    raise DatasetError("network_error", "%s:%s" % (
                        spec["role"], type(exc).__name__)) from None
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise DatasetError("oversize", "%s>%s" % (spec["role"], max_bytes))
                if len(head) < 64:
                    head += chunk[:64 - len(head)]
                digest.update(chunk)
                handle.write(chunk)
            handle.close()
            if declared_bytes is not None and written != declared_bytes:
                raise DatasetError("length_mismatch", "%s:%s!=%s" % (spec["role"], written,
                                                                     declared_bytes))
            _sniff(head, spec["expect"], str(headers.get("content-type", "")))
            actual = digest.hexdigest()
            pin = spec.get("sha256")
            if pin is not None and actual != pin:
                raise DatasetError("sha256_mismatch", "%s:%s" % (spec["role"], actual[:16]))
            os.replace(part, dest)
        except BaseException:
            if not handle.closed:
                handle.close()
            if part.exists():
                part.unlink()
            raise
    finally:
        if opened is not None and opened.reader is not None:
            try:
                opened.reader.close()
            except OSError:
                pass
    return {"bytes": written, "sha256": actual, "media_type": headers.get("content-type")}


def fetch_file(spec: dict, run_dir: Path, *, dataset: str, allowed_hosts: list, limits: dict,
               ledger_path: Path, opener=None) -> dict:
    """取一个原件：命中钉定哈希即跳过；每次尝试（含失败/重试）都记账本。

    写入前先验证 run 基准根本身与祖先链为普通目录；账本追加复用同一守卫（`_EventLedger`）。
    """
    ledger = _ledger_sink(run_dir, ledger_path)
    _assert_plain_chain(run_dir, "fetch:" + spec["role"])
    if opener is None:
        opener = functools.partial(default_opener, allowed_hosts=tuple(allowed_hosts))
    dest = run_dir / spec["dest"]
    pin = spec.get("sha256")
    attempts = int(limits["attempts"])
    if dest.exists() and pin is not None:
        _assert_plain_path(run_dir, dest, "fetch:" + spec["role"])
        cached = _sha256_file(dest)
        if cached == pin:
            ledger.append({"event": "fetch", "dataset": dataset,
                           "role": spec["role"], "url": spec["url"],
                           "status": "cached", "attempt": 0,
                           "bytes": dest.stat().st_size, "sha256": cached})
            return {"role": spec["role"], "url": spec["url"], "path": spec["dest"],
                    "bytes": dest.stat().st_size, "sha256": cached, "attempts": 0,
                    "skipped": True}
        ledger.append({"event": "fetch", "dataset": dataset,
                       "role": spec["role"], "url": spec["url"],
                       "status": "cached_mismatch", "attempt": 0,
                       "sha256": cached})
    last = None
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        try:
            result = _download_once(spec, dest, allowed_hosts=allowed_hosts,
                                    timeout=float(limits["timeout_seconds"]),
                                    max_bytes=int(spec["max_bytes"]), opener=opener,
                                    run_dir=run_dir)
        except DatasetError as exc:
            last = exc
            ledger.append({"event": "fetch", "dataset": dataset,
                           "role": spec["role"], "url": spec["url"],
                           "status": "error", "attempt": attempt,
                           "code": exc.code, "detail": exc.detail,
                           "seconds": round(time.monotonic() - started, 3)})
            if attempt < attempts:
                time.sleep(float(limits["retry_backoff_seconds"]) * attempt)
            continue
        ledger.append({"event": "fetch", "dataset": dataset,
                       "role": spec["role"], "url": spec["url"],
                       "status": "ok", "attempt": attempt,
                       "bytes": result["bytes"], "sha256": result["sha256"],
                       "media_type": result["media_type"],
                       "seconds": round(time.monotonic() - started, 3)})
        return {"role": spec["role"], "url": spec["url"], "path": spec["dest"],
                "bytes": result["bytes"], "sha256": result["sha256"], "attempts": attempt,
                "skipped": False}
    raise DatasetError("fetch_failed:" + str(last.code if last else "unknown"),
                       spec["role"])


def stage_fetch(config: dict, run_dir: Path, *, only: list | None = None, opener=None) -> dict:
    """下载全部数据集原件（或仅 `only` 指定）；写 `pins.json` 供配置钉定。"""
    _assert_plain_chain(run_dir, "run_root")
    run_dir.mkdir(parents=True, exist_ok=True)
    _check_budget(run_dir, int(config["limits"]["min_free_bytes"]))
    ledger_path = run_dir / "ledger.jsonl"
    pins = {}
    summary = {}
    for name in _selected(config, only):
        source = config["datasets"][name]
        entries = []
        for spec in source["files"]:
            _check_hosts(url=spec["url"], allowed=source["allowed_hosts"],
                         what="%s.%s" % (name, spec["role"]))
            entry = fetch_file(spec, run_dir, dataset=name,
                               allowed_hosts=source["allowed_hosts"],
                               limits=config["limits"], ledger_path=ledger_path, opener=opener)
            entries.append(entry)
            pins["%s/%s" % (name, spec["role"])] = entry["sha256"]
            print("%s/%s %s bytes=%d sha256=%s" % (name, spec["role"],
                                                   "cached" if entry["skipped"] else "ok",
                                                   entry["bytes"], entry["sha256"][:16]))
        summary[name] = entries
    pins_path = run_dir / "pins.json"
    _assert_plain_path(run_dir, pins_path, "pins")
    pins_path.write_text(
        json.dumps(pins, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    return summary


def _check_budget(run_dir: Path, min_free_bytes: int) -> None:
    usage = shutil.disk_usage(str(run_dir))
    if usage.free < min_free_bytes:
        raise DatasetError("disk_budget", "free=%d" % usage.free)


                                                                                     


def _alias_key(name: str) -> str:
    """Windows 别名归一：逐段去尾点/尾空格 + casefold（用于解包目标唯一性预检）。"""
    return "/".join(part.rstrip(" .").casefold() for part in name.split("/"))


def _validate_entry_name(name: object) -> None:
    """解包名称闸：绝对路径/驱动盘/`..`逃逸/反斜杠/NUL、ADS 冒号、尾点空格、设备名
    （含上标数字 COM¹ 等）一律拒绝。"""
    if not isinstance(name, str) or not name or len(name) > _MAX_NAME:
        raise DatasetError("bad_entry_name")
    if "\x00" in name or "\\" in name:
        raise DatasetError("bad_entry_name")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise DatasetError("bad_entry_name")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise DatasetError("bad_entry_name")
    for part in parts:
        if ":" in part or part.rstrip(" .") != part:
            raise DatasetError("bad_entry_name")
        stem = "".join(_SUPERSCRIPT_DIGITS.get(ch, ch) for ch in part.split(".")[0])
        if stem.strip().rstrip(" .").upper() in _DEVICE_NAMES:
            raise DatasetError("bad_entry_name")


def _entry_alias_map(entries: list, what: str) -> dict:
    """白名单成员预检的**唯一**实现：配置校验与公共解包 API 共用同一守卫。

    逐条名称合法（`bad_entry_name`）且 Windows 别名唯一；精确重复与大小写/尾点/尾空格
    归一后的别名碰撞一律 `alias_entry`，在任何输出创建前拒绝。
    """
    seen = {}
    for entry in entries:
        _validate_entry_name(entry)
        alias = _alias_key(entry)
        if alias in seen:
            raise DatasetError("alias_entry", "%s:%s" % (what, entry))
        seen[alias] = entry
    return seen


def _entry_infos(archive: zipfile.ZipFile, entries: list, limits: dict,
                 dataset: str) -> dict:
    """白名单与 central directory 预检：成员唯一、白名单别名唯一、类型/大小合规。"""
    infos = archive.infolist()
    if len(infos) > int(limits["max_entries"]):
        raise DatasetError("archive_too_many_entries", "%s:%d" % (dataset, len(infos)))
    central = {}
    for info in infos:
        if info.filename in central:
            raise DatasetError("duplicate_entry", info.filename)
        central[info.filename] = info
    selected = {}
    for entry in _entry_alias_map(entries, dataset).values():
        info = central.get(entry)
        if info is None:
            raise DatasetError("entry_missing", entry)
        if info.is_dir():
            raise DatasetError("entry_is_dir", entry)
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise DatasetError("entry_symlink", entry)
        if info.file_size > int(limits["max_entry_bytes"]):
            raise DatasetError("entry_oversize", entry)
        selected[entry] = info
    return selected


def _archive_members(archive_path: Path, entries: list, *, limits: dict, dataset: str) -> dict:
    """从（钉定的）zip 流式重派生白名单成员字节哈希：provenance 链的第一段。"""
    members = {}
    with zipfile.ZipFile(archive_path) as archive:
        selected = _entry_infos(archive, entries, limits, dataset)
        total = 0
        for entry, info in selected.items():
            digest = hashlib.sha256()
            written = 0
            with archive.open(info) as source:
                while True:
                    chunk = source.read(_CHUNK)
                    if not chunk:
                        break
                    written += len(chunk)
                    total += len(chunk)
                    if written > int(limits["max_entry_bytes"]) \
                            or total > int(limits["max_total_uncompressed_bytes"]):
                        raise DatasetError("entry_oversize", entry)
                    digest.update(chunk)
            members[entry] = {"entry": entry, "bytes": written, "sha256": digest.hexdigest()}
    return members


def extract_archive(archive_path: Path, dest_root: Path, entries: list, *, limits: dict,
                    dataset: str, ledger_path: Path, run_dir: Path) -> list:
    """整体预检后按白名单解出 zip 条目；独占 `.part` 落盘并复核字节哈希。

    写入前先验证 run 基准根本身与祖先链为普通目录；成员预检（`_entry_infos`，含与配置
    校验共用的白名单唯一性守卫）在任何输出（连输出根目录）创建前完成。任一失败（预检、
    坏 ZIP/CRC、读写 I/O、Unicode、逐条落盘、源件缺失）都归类为有界失败账本事件
    （code/detail/已完成条目数）后原样抛出。
    """
    ledger = _ledger_sink(run_dir, ledger_path)
    _assert_plain_chain(run_dir, "extract_root:" + dataset)
    records = []
    try:
        _assert_plain_path(run_dir, dest_root, "extract_root:" + dataset)
        if not archive_path.exists():
            raise DatasetError("missing_raw_file", "%s:%s" % (dataset, archive_path.name))
        with zipfile.ZipFile(archive_path) as archive:
            selected = _entry_infos(archive, entries, limits, dataset)
                                                  
                                                      
            for entry in selected:
                _assert_plain_path(run_dir, dest_root / entry, "extract:" + entry)
            dest_root.mkdir(parents=True, exist_ok=True)
            total = 0
            for entry, info in selected.items():
                target = dest_root / entry
                _assert_plain_path(run_dir, target, "extract:" + entry)
                target.parent.mkdir(parents=True, exist_ok=True)
                part, handle = _open_exclusive_part(target, "extract:" + entry)
                digest = hashlib.sha256()
                written = 0
                try:
                    with archive.open(info) as source:
                        while True:
                            chunk = source.read(_CHUNK)
                            if not chunk:
                                break
                            written += len(chunk)
                            total += len(chunk)
                            if written > int(limits["max_entry_bytes"]) \
                                    or total > int(limits["max_total_uncompressed_bytes"]):
                                raise DatasetError("entry_oversize", entry)
                            digest.update(chunk)
                            handle.write(chunk)
                    handle.close()
                    member_sha = digest.hexdigest()
                    os.replace(part, target)
                except BaseException:
                    if not handle.closed:
                        handle.close()
                    if part.exists():
                        part.unlink()
                    raise
                if _sha256_file(target) != member_sha:
                    raise DatasetError("write_verify_failed", entry)
                record = {"entry": entry, "path": str(target.relative_to(run_dir)),
                          "bytes": written, "sha256": member_sha}
                records.append(record)
                ledger.append({"event": "extract", "dataset": dataset,
                               "archive": archive_path.name, "entry": entry,
                               "status": "ok", "bytes": written, "sha256": member_sha})
    except Exception as exc:
        code, detail = _classify_error(exc)
        try:
            ledger.append({"event": "extract", "dataset": dataset,
                           "archive": archive_path.name, "status": "error",
                           "code": code, "detail": detail, "partial": len(records)})
        except Exception:
            pass                               
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError(code, detail) from exc
    return records


def stage_extract(config: dict, run_dir: Path, *, only: list | None = None) -> dict:
    summary = {}
    ledger_path = run_dir / "ledger.jsonl"
    for name in _selected(config, only):
        source = config["datasets"][name]
        entries = []
        for spec in source["files"]:
            if spec.get("kind", "direct") != "archive":
                continue
            archive_path = run_dir / spec["dest"]
                                                            
            extracted = extract_archive(
                archive_path, run_dir / spec["extract"]["root"],
                spec["extract"]["entries"], limits=config["limits"], dataset=name,
                ledger_path=ledger_path, run_dir=run_dir)
            entries.extend(extracted)
            print("%s/extract entries=%d" % (name, len(extracted)))
        summary[name] = entries
    return summary


                                                                                       


def _group_id(rule: dict, candidates: dict, premise: str, ref: str) -> str:
    """数据集级独立单位值：ocnli=真实 prem_id（缺失回退 premise_family_key）、anli=族键、
    hans=template、contractnli=doc_id；一律不加前缀（与消费方 unit==group_id 合同一致）。"""
    if rule["kind"] == "premise_family_key":
        return premise_family_key(premise)
    value = candidates.get(rule["field"])
    if not isinstance(value, str) or not value:
        fallback = rule.get("fallback")
        if fallback == "premise_family_key":
            return premise_family_key(premise)
        if fallback == "self":
            value = ref
        else:
            raise DatasetError("missing_group_field", rule["field"])
    return value


def build_record(*, dataset: str, official_split: str, task: str, ref: str, premise: object,
                 hypothesis: object, original_label: object, gold_status: str, label_map: dict,
                 group_rule: dict, unit_key: str, group_id_source: str, candidates: dict,
                 metadata: dict, evidence: dict) -> dict:
    """统一 record 构造与合同校验（形状/标签/状态/单位/有界），任何越界 fail-closed。"""
    if not isinstance(ref, str) or not ref or len(ref) > 200:
        raise DatasetError("bad_ref")
    if not isinstance(premise, str) or not premise or len(premise) > _MAX_TEXT:
        raise DatasetError("empty_premise", ref)
    if not isinstance(hypothesis, str) or not hypothesis or len(hypothesis) > _MAX_TEXT:
        raise DatasetError("empty_hypothesis", ref)
    if original_label is None:
        gold = None
    else:
        if not isinstance(original_label, str) or not original_label:
            raise DatasetError("invalid_original_label", ref)
        if original_label not in label_map:
            raise DatasetError("unknown_label", "%s:%s" % (ref, _bounded(original_label, 40)))
        gold = label_map[original_label]
    if gold is not None and gold not in TASK_LABELS[task]:
        raise DatasetError("bad_gold_label", "%s:%s" % (ref, gold))
    if gold_status not in GOLD_STATUSES:
        raise DatasetError("invalid_gold_status", "%s:%s" % (ref, gold_status))
    if (gold is None) != (gold_status in NULL_GOLD_STATUSES):
        raise DatasetError("gold_status_mismatch", "%s:%s" % (ref, gold_status))
    if gold is not None and (not isinstance(original_label, str) or not original_label.strip()):
        raise DatasetError("missing_original_label", ref)
    if group_id_source not in UNIT_SOURCES:
        raise DatasetError("bad_group_id_source", ref)
    group_id = _group_id(group_rule, candidates, premise, ref)
    if len(group_id) > 256:
        raise DatasetError("bad_group_id", ref)
    metadata_full = {"gold_status": gold_status, "original_label": original_label,
                     "group_id_source": group_id_source, **metadata}
    unit_value = metadata_full.get(unit_key)
    if unit_value is None:
                                                       
        if group_id_source != "derived_premise_family" or group_id != premise_family_key(premise):
            raise DatasetError("unit_mismatch", "%s:%s" % (ref, unit_key))
    elif str(unit_value) != group_id:
        raise DatasetError("unit_mismatch", "%s:%s" % (ref, unit_key))
    if type(metadata_full) is not dict or not metadata_full:
        raise DatasetError("invalid_metadata", ref)
    _check_jsonable(metadata_full, "metadata_oversize", max_bytes=_METADATA_MAX_BYTES)
    evidence_full = {"source_url": evidence.get("source_url"), "file": evidence.get("file"),
                     "sha256": evidence.get("sha256"), "row": evidence.get("row")}
    if not isinstance(evidence_full["source_url"], str) or not evidence_full["source_url"] \
            or len(evidence_full["source_url"]) > 4096:
        raise DatasetError("invalid_evidence", ref)
    if not isinstance(evidence_full["file"], str) or not evidence_full["file"] \
            or len(evidence_full["file"]) > 1024:
        raise DatasetError("invalid_evidence", ref)
    sha = evidence_full["sha256"]
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise DatasetError("invalid_evidence", ref)
    row = evidence_full["row"]
    if not (isinstance(row, int) and not isinstance(row, bool)
            or isinstance(row, str) and row.strip()):
        raise DatasetError("invalid_evidence", ref)
    _check_jsonable(evidence_full, "evidence_oversize", max_bytes=_EVIDENCE_MAX_BYTES)
    record = {
        "id": "%s:%s:%s" % (dataset, official_split, ref),
        "dataset": dataset,
        "official_split": official_split,
        "task": task,
        "premise": premise,
        "hypothesis": hypothesis,
        "gold_label": gold,
        "group_id": group_id,
        "metadata": metadata_full,
        "evidence": evidence_full,
    }
    if tuple(record) != RECORD_KEYS:
        raise DatasetError("bad_record_shape", ref)
    if len(record["id"]) > 256:
        raise DatasetError("bad_id", ref)
    return record


def _jsonl_rows(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise DatasetError("invalid_json_line", "%s:%d" % (path.name, number)) from None
            if type(row) is not dict:
                raise DatasetError("invalid_json_line", "%s:%d" % (path.name, number))
            yield number, row


OCNLI_VOTE_KEYS = ("label0", "label1", "label2", "label3", "label4")
OCNLI_VOTE_LABELS = frozenset({"entailment", "neutral", "contradiction"})


def ocnli_vote_status(original_label: object, votes: list) -> dict:
    """OCNLI 票证据的**纯逻辑锚点**（生产者 `parse_ocnli` 与消费方校验共用同一定义）。

    仅 `entailment`/`neutral`/`contradiction` 计有效票（`unknown`/`unrelated`/null 不计）；
    `original_label=None` → missing/no_label；`"-"` → no_consensus/majority_vote；其余
    以有效票对官方 gold 的**严格多数**判定：有效票 ≥2 且 `2×votes_agree_count >
    votes_valid_count` → majority/majority_vote；有效票 ≥2 但无严格多数 → single/
    `no_majority`（**官方 gold 与全部票原样保留**，不升级不改标；消费者不得据此改 gold）；
    否则 single/single_annotation。返回计数（`votes_valid_count`/`votes_agree_count`）
    供独立复算对账；不读写任何状态，不含 I/O。
    """
    valid = [vote for vote in votes
             if isinstance(vote, str) and vote in OCNLI_VOTE_LABELS]
    agrees = sum(1 for vote in valid if vote == original_label)
    if original_label is None:
        status, method = "missing", "no_label"
    elif original_label == "-":
        status, method = "no_consensus", "majority_vote"
    elif len(valid) >= 2 and 2 * agrees > len(valid):
        status, method = "majority", "majority_vote"
    elif len(valid) >= 2:
                                                      
        status, method = "single", "no_majority"
    else:
        status, method = "single", "single_annotation"
    return {"gold_status": status, "label_method": method,
            "votes_valid_count": len(valid), "votes_agree_count": agrees}


def parse_ocnli(ctx: dict, path: Path):
    """OCNLI：按 `ocnli_vote_status`（纯锚点）判 majority/single、`-` 无共识；
    官方缺 prem_id 保留 null。"""
    for number, row in _jsonl_rows(path):
        for key in ("id", "sentence1", "sentence2"):
            if key not in row:
                raise DatasetError("invalid_row", "%s:%d" % (path.name, number))
        original = row.get("label")
        votes = [row.get(key) for key in OCNLI_VOTE_KEYS]
        decision = ocnli_vote_status(original, votes)
        status, method = decision["gold_status"], decision["label_method"]
        valid_count = decision["votes_valid_count"]
        native = row.get("prem_id")
        if not isinstance(native, str) or not native:
            native = None
            group_id_source = "derived_premise_family"
        else:
            group_id_source = "raw"
        metadata = {"label_method": method, "prem_id": native, "genre": row.get("genre"),
                    "level": row.get("level"), "votes": votes,
                    "votes_valid_count": valid_count}
        yield build_record(
            dataset=ctx["dataset"], official_split=ctx["official_split"], task=ctx["task"],
            ref=str(row["id"]), premise=row["sentence1"], hypothesis=row["sentence2"],
            original_label=original, gold_status=status, label_map=ctx["label_map"],
            group_rule=ctx["group_rule"], unit_key=ctx["unit_key"], group_id_source=group_id_source,
            candidates={"prem_id": native}, metadata=metadata,
            evidence={**ctx["evidence_base"], "row": number})


def parse_anli(ctx: dict, path: Path):
    """ANLI：e/n/c 规范三分类；adjudicated 仅在 ≥2 复核票等于 gold 时成立。"""
    round_name = ctx["official_split"].split("/")[0].lower()
    for number, row in _jsonl_rows(path):
        for key in ("uid", "context", "hypothesis", "label"):
            if key not in row:
                raise DatasetError("invalid_row", "%s:%d" % (path.name, number))
        family = premise_family_key(row["context"])
        verifiers = None
        for join in ctx.get("joins", []):
            index = ctx["join_indexes"].get(join["metadata_key"], {})
            verifiers = index.get(row.get(join["key_field"]))
        if verifiers is None:
            status, method, agrees = "single", "writer_only", None
        else:
            agrees = sum(1 for vote in verifiers if vote == row["label"])
            if agrees >= 2:
                status, method = "adjudicated", "writer_with_verifiers"
            else:
                status, method = "single", "writer_only"
        metadata = {"label_method": method, "premise_hash": family, "round": round_name,
                    "uid": row["uid"], "genre": row.get("genre"), "emturk": row.get("emturk"),
                    "tag": row.get("tag"), "model_label": row.get("model_label"),
                    "reason": row.get("reason"), "verifier_labels": verifiers,
                    "verifier_agree_count": agrees}
        yield build_record(
            dataset=ctx["dataset"], official_split=ctx["official_split"], task=ctx["task"],
            ref=str(row["uid"]), premise=row["context"], hypothesis=row["hypothesis"],
            original_label=row["label"], gold_status=status, label_map=ctx["label_map"],
            group_rule=ctx["group_rule"], unit_key=ctx["unit_key"],
            group_id_source="derived_premise_family", candidates={},
            metadata=metadata, evidence={**ctx["evidence_base"], "row": number})


HANS_FIELDS = ("gold_label", "sentence1", "sentence2", "pairID", "heuristic", "subcase",
               "template")


def parse_hans(ctx: dict, path: Path):
    """HANS TSV：模板规则生成标签（generated），entailment/non-entailment 原样。"""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        header_line = handle.readline().rstrip("\r\n")
        header = header_line.split("\t")
        for field in HANS_FIELDS:
            if field not in header:
                raise DatasetError("bad_header", "%s:%s" % (path.name, field))
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        pairs = {}
        for number, row in enumerate(reader, start=2):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                raise DatasetError("bad_row", "%s:%d" % (path.name, number))
            item = dict(zip(header, row))
            if item["pairID"] in pairs:
                raise DatasetError("duplicate_pair_id", "%s:%d" % (path.name, number))
            metadata = {"label_method": "template_rule", "pairID": item["pairID"],
                        "heuristic": item["heuristic"], "subcase": item["subcase"],
                        "template": item["template"]}
            pairs[item["pairID"]] = item["gold_label"]
            yield build_record(
                dataset=ctx["dataset"], official_split=ctx["official_split"], task=ctx["task"],
                ref=item["pairID"], premise=item["sentence1"], hypothesis=item["sentence2"],
                original_label=item["gold_label"], gold_status="generated",
                label_map=ctx["label_map"], group_rule=ctx["group_rule"],
                unit_key=ctx["unit_key"], group_id_source="raw",
                candidates={"template": item["template"]},
                metadata=metadata, evidence={**ctx["evidence_base"], "row": number})
    crosscheck = ctx.get("crosscheck_path")
    if crosscheck is not None:
        _check_hans_crosscheck(crosscheck, pairs)


def _check_hans_crosscheck(path: Path, pairs: dict) -> None:
    """官方 .jsonl 副本比对 pairID->标签 全表：单文件静默损坏即停（与行序无关）。"""
    expected = {}
    for number, row in _jsonl_rows(path):
        key = row.get("pairID")
        if not isinstance(key, str) or not key or key in expected:
            raise DatasetError("bad_join_row", "%s:%d" % (path.name, number))
        expected[key] = row.get("gold_label")
    if expected != pairs:
        raise DatasetError("crosscheck_mismatch",
                           "%s:%d!=%d" % (path.name, len(pairs), len(expected)))


def parse_contractnli(ctx: dict, path: Path):
    """ContractNLI：每文档 × 全部 17 命题（官方 id 有间断，按数量校验）。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if type(data) is not dict or type(data.get("documents")) is not list \
            or type(data.get("labels")) is not dict:
        raise DatasetError("bad_contract_json", path.name)
    hypotheses = data["labels"]
    expected = ctx.get("expect_hypotheses")
    if expected is not None and len(hypotheses) != expected:
        raise DatasetError("hypotheses_mismatch", "%s:labels=%d" % (path.name, len(hypotheses)))
    for document in data["documents"]:
        if type(document) is not dict:
            raise DatasetError("bad_contract_json", path.name)
        doc_id = document.get("id")
        text = document.get("text")
        spans = document.get("spans")
        annotation_sets = document.get("annotation_sets")
        if isinstance(doc_id, bool) or not isinstance(doc_id, int) \
                or not isinstance(text, str) or not text \
                or type(spans) is not list or type(annotation_sets) is not list:
            raise DatasetError("bad_contract_doc", "%s:%s" % (path.name, _bounded(doc_id)))
        if len(annotation_sets) != 1 or type(annotation_sets[0]) is not dict \
                or type(annotation_sets[0].get("annotations")) is not dict:
            raise DatasetError("annotation_sets_mismatch", "%s:%s" % (path.name, doc_id))
        annotations = annotation_sets[0]["annotations"]
        if set(annotations) != set(hypotheses):
            raise DatasetError("hypotheses_mismatch", "%s:%s" % (path.name, doc_id))
        for hyp_id, hypothesis in hypotheses.items():
            annotation = annotations[hyp_id]
            if type(annotation) is not dict or "choice" not in annotation \
                    or type(annotation.get("spans", [])) is not list:
                raise DatasetError("bad_contract_annotation",
                                   "%s:%s:%s" % (path.name, doc_id, hyp_id))
            indices = annotation.get("spans", [])
            span_out = []
            for index in indices:
                if isinstance(index, bool) or not isinstance(index, int) \
                        or not 0 <= index < len(spans) or type(spans[index]) is not list \
                        or len(spans[index]) != 2:
                    raise DatasetError("bad_span_index", "%s:%s:%s" % (path.name, doc_id, hyp_id))
                start, end = spans[index]
                if not (isinstance(start, int) and not isinstance(start, bool)
                        and isinstance(end, int) and not isinstance(end, bool)
                        and 0 <= start <= end <= len(text)):
                    raise DatasetError("bad_span_range", "%s:%s:%s" % (path.name, doc_id, hyp_id))
                span_text = text[start:end]
                if not span_text.strip():
                                                                           
                                                         
                    ctx["notes"]["spans_skipped_whitespace"] += 1
                    continue
                span_out.append({"text": span_text, "start": start, "end": end})
            metadata = {
                "label_method": "single_annotation", "doc_id": str(doc_id),
                "hypothesis_id": hyp_id, "hypothesis_short": hypothesis.get("short_description"),
                "file_name": document.get("file_name"),
                "document_type": document.get("document_type"),
                "document_url": document.get("url"), "spans": span_out,
            }
            yield build_record(
                dataset=ctx["dataset"], official_split=ctx["official_split"], task=ctx["task"],
                ref="%s:%s" % (doc_id, hyp_id), premise=text,
                hypothesis=hypothesis.get("hypothesis"), original_label=annotation["choice"],
                gold_status="single", label_map=ctx["label_map"], group_rule=ctx["group_rule"],
                unit_key=ctx["unit_key"], group_id_source="raw",
                candidates={"doc_id": str(doc_id)}, metadata=metadata,
                evidence={**ctx["evidence_base"],
                          "row": "doc:%s/hyp:%s" % (doc_id, hyp_id)})


PARSERS = {
    "ocnli_jsonl": parse_ocnli,
    "anli_jsonl": parse_anli,
    "hans_tsv": parse_hans,
    "contractnli_json": parse_contractnli,
}

OMITTED_FIELDS = {
    "hans_tsv": ["sentence1_binary_parse", "sentence2_binary_parse",
                 "sentence1_parse", "sentence2_parse"],
    "contractnli_json": ["zip 内 raw/ 原始合同文件（未入解包白名单；原件 zip 完整保留）"],
    "anli_jsonl": [],
    "ocnli_jsonl": [],
}


def _selected(config: dict, only: list | None) -> list:
    names = list(config["datasets"])
    if only:
        unknown = [name for name in only if name not in names]
        if unknown:
            raise DatasetError("unknown_dataset", ",".join(unknown))
        names = [name for name in names if name in only]
    return names


def _file_map(source: dict) -> dict:
    return {spec["role"]: spec for spec in source["files"]}


def _safe_split_name(official_split: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "-", official_split)


def _build_join_indexes(source: dict, run_dir: Path) -> dict:
    indexes = {}
    for join in source.get("joins", []):
        spec = _file_map(source)[join["role"]]
        path = run_dir / spec["dest"]
        if not path.exists():
            raise DatasetError("missing_raw_file", "%s:%s" % (join["role"], spec["dest"]))
        index = {}
        for number, row in _jsonl_rows(path):
            key = row.get(join["key_field"])
            value = row.get(join["value_field"])
            if key is None or value is None or key in index:
                raise DatasetError("bad_join_row", "%s:%d" % (path.name, number))
            index[key] = value
        indexes[join["metadata_key"]] = index
    return indexes


def _new_diagnostics() -> dict:
    return {"family_group": {}, "family_group_counted": set(),
            "cross_group": {"count": 0, "refs": [], "truncated": False},
            "family_split": {}, "family_split_counted": set(),
            "cross_split": {"count": 0, "refs": [], "truncated": False},
            "pair_first": {}, "pair_duplicates": {"count": 0, "refs": [], "truncated": False}}


def _note(container: dict, ref: str) -> None:
    container["count"] += 1
    if len(container["refs"]) < _MAX_REFS:
        container["refs"].append(ref)
    else:
        container["truncated"] = True


def _update_diagnostics(diag: dict, *, family: str, group_id: str, official_split: str,
                        pair_key: str, record_id: str) -> None:
    """同 premise 跨 group/分区与同有向 pair 碰撞计数（供消费方合并 cluster/泄漏判交）。"""
    previous_group = diag["family_group"].setdefault(family, group_id)
    if previous_group != group_id and family not in diag["family_group_counted"]:
        diag["family_group_counted"].add(family)
        _note(diag["cross_group"], "%s:%s|%s" % (family, previous_group, group_id))
    previous_split = diag["family_split"].setdefault(family, official_split)
    if previous_split != official_split and family not in diag["family_split_counted"]:
        diag["family_split_counted"].add(family)
        _note(diag["cross_split"], "%s:%s|%s" % (family, previous_split, official_split))
    first = diag["pair_first"].setdefault(pair_key, record_id)
    if first != record_id:
        _note(diag["pair_duplicates"], "%s<=%s" % (record_id, first))


def _finalize_diagnostics(diag: dict) -> dict:
    return {
        "same_premise_cross_group": diag["cross_group"],
        "same_premise_cross_split": diag["cross_split"],
        "directed_pair_duplicates": diag["pair_duplicates"],
    }


def _split_source(source: dict, split: dict, run_dir: Path) -> dict:
    files = _file_map(source)
    spec = files[split["file_role"]] if "file_role" in split else files["archive"]
    entry = None if "file_role" in split else split["entry"]
    if "file_role" in split:
        source_path = run_dir / spec["dest"]
        file_ref = spec["dest"]
    else:
        source_path = run_dir / spec["extract"]["root"] / entry
        file_ref = str(source_path.relative_to(run_dir))
    return {"spec": spec, "entry": entry, "path": source_path, "file_ref": file_ref}


def normalize_split(source: dict, dataset: str, split: dict, run_dir: Path, out_dir,
                    ledger, diagnostics: dict) -> dict:
    """一个分区的规范化：解析 -> 合同校验 -> 计数 ->（可选）独占 `.part` 落 JSONL。

    `out_dir=None` 为纯摘要模式（只算哈希/计数、不写任何文件），供只读 verify 复用。
    事件账本经显式 sink（`_EventLedger`；只读模式为 None）落盘：解析失败、pin/来源前置
    失败、输出守卫失败与 I/O 失败走同一接缝，sink 为 None 时任何路径都无处落盘。
    """
    sink = None if ledger is None else _ledger_sink(run_dir, ledger)
    if sink is not None:
        _assert_plain_chain(run_dir, "run_root")
    counts = {"rows": 0, "gold_label": {}, "gold_status": {}, "group_id_source": {},
              "votes_valid": {}, "missing_gold": 0, "no_consensus": 0, "groups": set()}
    seen_ids = set()
    joins_present = {join["metadata_key"]: 0 for join in source.get("joins", [])}
    part = None
    handle = None
    digest = hashlib.sha256()
    written_bytes = 0
    out_path = None
    try:
        resolved = _split_source(source, split, run_dir)
        spec, entry, source_path = resolved["spec"], resolved["entry"], resolved["path"]
        if not source_path.exists():
            raise DatasetError("missing_source_file", str(source_path))
        file_sha = _sha256_file(source_path)
        if spec.get("sha256") is not None and entry is None and file_sha != spec["sha256"]:
            raise DatasetError("pin_mismatch", "%s:%s" % (dataset, spec["role"]))
        crosscheck_path = None
        if split.get("crosscheck_role"):
            crosscheck_path = run_dir / _file_map(source)[split["crosscheck_role"]]["dest"]
            if not crosscheck_path.exists():
                raise DatasetError("missing_raw_file", split["crosscheck_role"])
        ctx = {
            "dataset": dataset, "official_split": split["official_split"],
            "task": source["task"], "label_map": source["label_map"],
            "group_rule": source["group_id"], "unit_key": source["unit_key"],
            "evidence_base": {"source_url": spec["url"], "file": resolved["file_ref"],
                              "sha256": file_sha},
            "joins": source.get("joins", []),
            "join_indexes": _build_join_indexes(source, run_dir) if source.get("joins") else {},
            "crosscheck_path": crosscheck_path,
            "expect_hypotheses": source.get("expect_hypotheses"),
            "notes": {"spans_skipped_whitespace": 0},
        }
        parser = PARSERS.get(split["format"])
        if parser is None:
            raise DatasetError("unknown_format", split["format"])
        output_root = out_dir if out_dir is not None else run_dir / "normalized"
        out_path = output_root / ("%s.%s.jsonl"
                                  % (dataset, _safe_split_name(split["official_split"])))
        if out_dir is not None:
            _assert_plain_path(run_dir, out_dir, "output_root")
            out_dir.mkdir(parents=True, exist_ok=True)
            _assert_plain_path(run_dir, out_path, "output:" + dataset)
            part, handle = _open_exclusive_part(out_path, "output:" + dataset)
        for record in parser(ctx, source_path):
            if tuple(record) != RECORD_KEYS:
                raise DatasetError("bad_record_shape", str(record.get("id")))
            if record["id"] in seen_ids:
                raise DatasetError("duplicate_id", str(record["id"]))
            seen_ids.add(record["id"])
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"),
                              allow_nan=False)
            payload = line.encode("utf-8") + b"\n"
            if handle is not None:
                handle.write(payload)
            digest.update(payload)
            written_bytes += len(payload)
            counts["rows"] += 1
            gold = record["gold_label"]
            status = record["metadata"]["gold_status"]
            counts["gold_status"][status] = counts["gold_status"].get(status, 0) + 1
            counts["group_id_source"][record["metadata"]["group_id_source"]] = \
                counts["group_id_source"].get(record["metadata"]["group_id_source"], 0) + 1
            valid = record["metadata"].get("votes_valid_count")
            if isinstance(valid, int):
                key = str(valid)
                counts["votes_valid"][key] = counts["votes_valid"].get(key, 0) + 1
            if gold is None:
                counts["missing_gold"] += 1
                if status == "no_consensus":
                    counts["no_consensus"] += 1
            else:
                counts["gold_label"][gold] = counts["gold_label"].get(gold, 0) + 1
            counts["groups"].add(record["group_id"])
            family = premise_family_key(record["premise"])
            pair_key = hashlib.sha256(
                ("%s\x00%s" % (family,
                               " ".join(record["hypothesis"].split()))).encode("utf-8")
            ).hexdigest()[:16]
            _update_diagnostics(diagnostics, family=family, group_id=record["group_id"],
                                official_split=record["official_split"], pair_key=pair_key,
                                record_id=record["id"])
            for key in joins_present:
                if record["metadata"].get(key) is not None:
                    joins_present[key] += 1
        if handle is not None:
            handle.close()
            os.replace(part, out_path)
    except BaseException as exc:
        if handle is not None and not handle.closed:
            handle.close()
        if part is not None and part.exists():
            part.unlink()
        if sink is not None and isinstance(exc, Exception):
            code, detail = _classify_error(exc)
            try:
                sink.append({"event": "normalize", "dataset": dataset,
                             "split": split["official_split"], "status": "error",
                             "code": code, "detail": detail, "partial": counts["rows"]})
            except Exception:
                pass                               
        raise
    result = {
        "official_split": split["official_split"],
        "raw_file": spec["dest"],
        "entry": entry,
        "normalized": {"path": str(out_path.relative_to(run_dir)),
                       "bytes": written_bytes, "sha256": digest.hexdigest(),
                       "records": counts["rows"]},
        "counts": {**counts, "groups": len(counts["groups"]),
                   "spans_skipped_whitespace": ctx["notes"]["spans_skipped_whitespace"],
                   "join_present": joins_present,
                   "join_absent": {key: counts["rows"] - value
                                   for key, value in joins_present.items()}},
        "omitted_fields": OMITTED_FIELDS.get(split["format"], []),
    }
    if out_dir is not None and sink is not None:
        sink.append({"event": "normalize", "dataset": dataset,
                     "split": split["official_split"], "status": "ok",
                     "records": counts["rows"],
                     "sha256": result["normalized"]["sha256"]})
    return result


def _catalog(config: dict) -> dict:
    """label_map/单位的机器可读锚点（供消费方 parity 校验，避免第二份平行维护）。"""
    label_map = {name: dict(source["label_map"]) for name, source in config["datasets"].items()}
    payload = json.dumps(label_map, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return {
        "label_map": label_map,
        "label_map_sha256": hashlib.sha256(payload).hexdigest(),
        "tasks": {name: source["task"] for name, source in config["datasets"].items()},
        "unit_keys": {name: source["unit_key"] for name, source in config["datasets"].items()},
    }


def normalize_all(config: dict, run_dir: Path, out_dir, *, only: list | None = None,
                  write: bool = True, config_path: Path = CONFIG_PATH) -> dict:
    """构建（并可选写出）整套 manifest：来源链（zip→成员→normalized）、计数与诊断。

    `out_dir=None` 时纯摘要（不写规范化文件），`write=True` 时另写 manifest.json。
    写入能力：会写产物/manifest 的阶段获得账本 sink（`_EventLedger`，run 根链先验证）；
    纯只读摘要（`out_dir=None` 且 `write=False`，verify 使用）不持有 sink——成功、解析
    失败、前置失败（pin/extracted drift/缺件）与 I/O 失败的嵌套路径都无处落盘。
    """
    readonly = out_dir is None and not write
    ledger = None if readonly else _EventLedger(run_dir, run_dir / "ledger.jsonl")
    if ledger is not None:
        _assert_plain_chain(run_dir, "run_root")
    manifest = {
        "schema_version": 1,
        "run": str(run_dir.name),
        "tool": "research/memory_relations/public_benchmarks/datasets.py",
        "config": {"path": str(config_path), "sha256": _sha256_file(Path(config_path))},
        "generated_at_utc": _utc_now(),
        "catalog": _catalog(config),
        "datasets": {},
        "totals": {"records": 0, "missing_gold": 0, "no_consensus": 0},
    }
    for name in _selected(config, only):
        source = config["datasets"][name]
        files_manifest = []
        extracted_entries = []
        try:
            for spec in source["files"]:
                path = run_dir / spec["dest"]
                if not path.exists():
                    raise DatasetError("missing_raw_file", "%s:%s" % (name, spec["dest"]))
                sha = _sha256_file(path)
                pin = spec.get("sha256")
                files_manifest.append({
                    "role": spec["role"], "url": spec["url"], "dest": spec["dest"],
                    "bytes": path.stat().st_size, "sha256": sha,
                    "pinned": pin is not None, "expect": spec["expect"],
                })
                if pin is not None and sha != pin:
                    raise DatasetError("pin_mismatch", "%s:%s" % (name, spec["role"]))
                if spec.get("kind", "direct") == "archive":
                                                           
                    members = _archive_members(path, spec["extract"]["entries"],
                                               limits=config["limits"], dataset=name)
                    root = run_dir / spec["extract"]["root"]
                    for entry in spec["extract"]["entries"]:
                        entry_path = root / entry
                        if not entry_path.exists():
                            raise DatasetError("missing_extracted_file",
                                               "%s:%s" % (name, entry))
                        on_disk = {"entry": entry,
                                   "path": str(entry_path.relative_to(run_dir)),
                                   "bytes": entry_path.stat().st_size,
                                   "sha256": _sha256_file(entry_path)}
                        member = members[entry]
                        if on_disk["sha256"] != member["sha256"] \
                                or on_disk["bytes"] != member["bytes"]:
                            raise DatasetError("extracted_drift", "%s:%s" % (name, entry))
                        extracted_entries.append(on_disk)
        except Exception as exc:
                                                              
            if ledger is not None:
                code, detail = _classify_error(exc)
                try:
                    ledger.append({"event": "normalize", "dataset": name, "split": None,
                                   "status": "error", "stage": "precondition",
                                   "code": code, "detail": detail})
                except Exception:
                    pass                               
            raise
        diagnostics = _new_diagnostics()
        splits = {}
        for split_name, split in source["splits"].items():
            splits[split_name] = normalize_split(source, name, split, run_dir, out_dir,
                                                 ledger, diagnostics)
        manifest["datasets"][name] = {
            "title": source["title"], "home": source["home"],
            "revision": source.get("revision"), "license": source["license"],
            "task": source["task"], "notes": source.get("notes"),
            "files": files_manifest, "extracted": extracted_entries, "splits": splits,
            "diagnostics": _finalize_diagnostics(diagnostics),
        }
        for split in splits.values():
            manifest["totals"]["records"] += split["normalized"]["records"]
            manifest["totals"]["missing_gold"] += split["counts"]["missing_gold"]
            manifest["totals"]["no_consensus"] += split["counts"]["no_consensus"]
    if write:
        manifest_path = run_dir / "manifest.json"
        try:
            _assert_plain_path(run_dir, manifest_path, "manifest")
            part, handle = _open_exclusive_part(manifest_path, "manifest")
            try:
                handle.write((json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                              + "\n").encode("utf-8"))
                handle.close()
                os.replace(part, manifest_path)
            except BaseException:
                if not handle.closed:
                    handle.close()
                if part.exists():
                    part.unlink()
                raise
        except Exception as exc:
            if ledger is not None:
                code, detail = _classify_error(exc)
                try:
                    ledger.append({"event": "manifest", "dataset": None, "status": "error",
                                   "code": code, "detail": detail})
                except Exception:
                    pass                               
            raise
    return manifest


def stage_normalize(config: dict, run_dir: Path, *, only: list | None = None,
                    config_path: Path = CONFIG_PATH) -> dict:
    manifest = normalize_all(config, run_dir, run_dir / "normalized", only=only, write=True,
                             config_path=config_path)
    for name, data in manifest["datasets"].items():
        for split_name, split in data["splits"].items():
            print("%s/%s rows=%d missing_gold=%d no_consensus=%d groups=%d sha256=%s" % (
                name, split_name, split["counts"]["rows"], split["counts"]["missing_gold"],
                split["counts"]["no_consensus"], split["counts"]["groups"],
                split["normalized"]["sha256"][:16]))
            print("   gold_status=%s group_id_source=%s" % (
                json.dumps(split["counts"]["gold_status"], sort_keys=True),
                json.dumps(split["counts"]["group_id_source"], sort_keys=True)))
        diag = data["diagnostics"]
        print("%s/diagnostics cross_group=%d cross_split=%d pair_dupes=%d" % (
            name, diag["same_premise_cross_group"]["count"],
            diag["same_premise_cross_split"]["count"],
            diag["directed_pair_duplicates"]["count"]))
    print("totals records=%d missing_gold=%d no_consensus=%d" % (
        manifest["totals"]["records"], manifest["totals"]["missing_gold"],
        manifest["totals"]["no_consensus"]))
    return manifest


                                                                                    


def _manifest_view(document: dict) -> dict:
    """比较视图：仅剔除时间与本地路径等易变字段，其余承诺字段全部参与对账。"""
    view = copy.deepcopy(document)
    view.pop("generated_at_utc", None)
    if isinstance(view.get("config"), dict):
        view["config"].pop("path", None)
    return view


def _diff_fields(left: object, right: object, path: str, out: list, limit: int = _MAX_DIFFS) -> None:
    if len(out) >= limit:
        return
    if type(left) is not type(right):
        out.append("%s(type %s!=%s)" % (path, type(left).__name__, type(right).__name__))
        return
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left:
                out.append("%s.%s(missing-left)" % (path, key))
            elif key not in right:
                out.append("%s.%s(missing-right)" % (path, key))
            else:
                _diff_fields(left[key], right[key], "%s.%s" % (path, key), out, limit)
            if len(out) >= limit:
                return
        return
    if isinstance(left, list):
        if len(left) != len(right):
            out.append("%s(len %d!=%d)" % (path, len(left), len(right)))
            return
        for index, (a, b) in enumerate(zip(left, right)):
            _diff_fields(a, b, "%s[%d]" % (path, index), out, limit)
            if len(out) >= limit:
                return
        return
    if left != right:
        out.append("%s(%s!=%s)" % (path, _bounded(left, 60), _bounded(right, 60)))


def verify_run(config: dict, run_dir: Path, *, only: list | None = None,
               config_path: Path = CONFIG_PATH) -> list:
    """真只读复核：不写任何产物、不建/删 run 内目录、不追加源账本。

    复用同一套规范化迭代（`out_dir=None` 纯摘要，不持有账本 sink）重算全部承诺字段
    （来源链 zip→成员→normalized、集合/路径/SHA/计数/诊断/catalog），与存储 manifest
    全字段对账。只读对**所有路径**成立：成功、解析失败（unknown_label 等）、前置失败
    （pin/extracted drift/缺件）与 I/O 失败都只返回结构化问题，零写入。
    """
    problems = []
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return ["manifest_missing"]
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        fresh = normalize_all(config, run_dir, None, only=only, write=False,
                              config_path=config_path)
    except DatasetError as exc:
        return ["normalize_failed:%s" % exc.code]
    except Exception as exc:
                                                                   
        return ["normalize_failed:%s" % _classify_error(exc)[0]]
    stored_view = _manifest_view(stored)
    fresh_view = _manifest_view(fresh)
    if only:
        scope = set(_selected(config, only))
        expected_totals = {"records": 0, "missing_gold": 0, "no_consensus": 0}
        for name, data in stored.get("datasets", {}).items():
            for split in data.get("splits", {}).values():
                expected_totals["records"] += split["normalized"]["records"]
                expected_totals["missing_gold"] += split["counts"]["missing_gold"]
                expected_totals["no_consensus"] += split["counts"]["no_consensus"]
        if expected_totals != stored.get("totals"):
            problems.append("stored_totals_inconsistent")
        stored_subset = {key: value for key, value in stored_view.items()
                         if key not in ("datasets", "totals")}
        stored_subset["datasets"] = {key: value for key, value in stored_view["datasets"].items()
                                     if key in scope}
        fresh_subset = {key: value for key, value in fresh_view.items()
                        if key not in ("datasets", "totals")}
        fresh_subset["datasets"] = fresh_view["datasets"]
        differences = []
        _diff_fields(stored_subset, fresh_subset, "manifest", differences)
    else:
        differences = []
        _diff_fields(stored_view, fresh_view, "manifest", differences)
    problems.extend("manifest_field_mismatch:%s" % item for item in differences)
    scope = set(_selected(config, only))
    for name, data in stored.get("datasets", {}).items():
        if name not in scope:
            continue
        for split_name, split in data.get("splits", {}).items():
            on_disk = run_dir / split["normalized"]["path"]
            if not on_disk.exists():
                problems.append("normalized_file_missing:%s:%s" % (name, split_name))
            elif _sha256_file(on_disk) != split["normalized"]["sha256"]:
                problems.append("normalized_file_drift:%s:%s" % (name, split_name))
    return sorted(problems)


def stage_verify(config: dict, run_dir: Path, *, only: list | None = None,
                 config_path: Path = CONFIG_PATH) -> list:
    scope = "full" if not only else "subset:" + ",".join(sorted(_selected(config, only)))
    print("verify scope=%s (read-only; no ledger/temp writes)" % scope)
    problems = verify_run(config, run_dir, only=only, config_path=config_path)
    for problem in problems:
        print("PROBLEM " + problem)
    print("verify problems=%d" % len(problems))
    if only:
        print("NOTE subset scope only: datasets outside %s not re-verified" % scope)
    return problems


                                                                                 


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="公开基准数据集的获取/解包/规范化/复核（研究内部，纯标准库）")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("fetch", "extract", "normalize", "verify"):
        item = sub.add_parser(command)
        item.add_argument("--config", default=str(CONFIG_PATH))
        item.add_argument("--run", default=None)
        item.add_argument("--only", default=None, help="逗号分隔的数据集 id")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        config_path = Path(args.config)
        run_dir = Path(args.run) if args.run else RUNS_ROOT / config["default_run"]
        only = [item for item in (args.only or "").split(",") if item] or None
        print("run=%s" % run_dir)
        if args.command == "fetch":
            stage_fetch(config, run_dir, only=only)
            return 0
        if args.command == "extract":
            stage_extract(config, run_dir, only=only)
            return 0
        if args.command == "normalize":
            stage_normalize(config, run_dir, only=only, config_path=config_path)
            return 0
        problems = stage_verify(config, run_dir, only=only, config_path=config_path)
        return 1 if problems else 0
    except DatasetError as exc:
        print("ERROR %s %s" % (exc.code, exc.detail))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
