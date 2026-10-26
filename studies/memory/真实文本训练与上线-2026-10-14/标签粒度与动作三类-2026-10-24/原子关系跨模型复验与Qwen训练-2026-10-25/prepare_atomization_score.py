                      
"""跨模型原子化复验：把新模型的 A1 raw 接到已审 v2 评分输入上。

原实验（只读）：../原子记忆真实试点与高效架构-2026-10-25/
  - score_atomic_memory_20261025.py：原评分器，本文件只做字节副本，绝不修改
  - data/_dev-v2-score/{score_atomic_memory_20261025.py,data/}：已审 90 行输入
    （P0/A1/A2 各 30；raw-A1 是 DeepSeek v2）

命令（全部离线，不调用模型/网络；sys.dont_write_bytecode 已开）：
  prepare  --model <id>                             生成 working / pending / summary
  finalize --model <id> --annotations <path> [--allow-same-sha]
  selftest                                          临时目录小夹具自测

prepare 产物（data/<model>/）：
  annotations-working.jsonl   P0 原样 + A1（复用/空/待审 placeholder）+ A2 原样
  review-pending.jsonl        每个待审 case 一包，字段：
                              blind_id/case_id/split/conversations/forbidden_inferences/
                              one_off_hints/gold_obligations/candidates/annotation_template
  reuse-summary.json          输入/输出 SHA、逐 case 决策、复用统计

finalize 产物（data/<model>/score/）：
  score_atomic_memory_20261025.py   原评分器字节副本
  data/{cases-v1,raw-P0,raw-A1,raw-A2,call-ledger,annotations-v1}.jsonl
  manifest.json                     输入/脚本 SHA、模型、复用统计、dry-run 摘要

判据：
- 只有新 A1 与 DeepSeek A1-v2 的**整个输出集的候选有序列表**在
  text/evidence/tier/trigger 上完全一致，才整集复用旧标注；blind_id/candidate_id
  以新 raw 为准。任一候选不同则该 case 全部候选重审，不做单 atom 部分复用
  （duplicate_group / 集合语境会错配）。
- status=valid_empty（合法空输出）可直接 atoms=[] 标完成；其它空/失败输出仍进重审。
- placeholder 一律 adjudicated=false，绝不进 final；finalize 只吃主审完成的 JSONL。
- P0/A2 raw 与标注原样复制，只为满足评分器结构；报告只比较 A1 与 P0，
  不宣称跑了新 A2。
- 新采集器 ledger 用 latency_s，评分器只认 elapsed_s|elapsed_seconds；finalize
  做同值字段适配（不改数值、不伪造），并在 manifest 记账。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
PILOT_DIR = EXP.parent / "原子记忆真实试点与高效架构-2026-10-25"
DEV_SCORE_DIR = PILOT_DIR / "data" / "_dev-v2-score"
DEV_DATA = DEV_SCORE_DIR / "data"
SCORER_PATH = PILOT_DIR / "score_atomic_memory_20261025.py"
DATA_ROOT = EXP / "data"

ARM = "A1"
ARMS = ("P0", "A1", "A2")
COMPARE_FIELDS = ("text", "evidence", "tier", "trigger")
INPUT_NAMES = ("cases-v1.jsonl", "raw-P0.jsonl", "raw-A1.jsonl", "raw-A2.jsonl",
               "call-ledger.jsonl", "annotations-v1.jsonl")
SUMMARY_SCHEMA = "atom-prepare-reuse-summary-v1"
REVIEW_SCHEMA = "atom-review-pending-v1"
MANIFEST_SCHEMA = "atom-score-bundle-manifest-v1"
INSTRUCTIONS = (
    "盲审：只依据 conversations / forbidden_inferences / one_off_hints / gold_obligations "
    "判断每个 candidate 的原子性、蕴含与持久性；atoms 必须与 candidates 一一对应且顺序一致；"
    "完成后把 stage1_complete/stage2_complete/adjudicated 全置 true 再交给 finalize。"
)

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import run_atomization_comparison as collector              


class PrepError(Exception):
    """带机器码的错误；消息只含 case/blind/计数，不含候选原文。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def require(condition: Any, code: str, message: str) -> None:
    if not condition:
        raise PrepError(code, message)


def clip(items: Sequence[Any], limit: int = 12) -> str:
    values = [str(item) for item in items]
    if len(values) <= limit:
        return ", ".join(values)
    return ", ".join(values[:limit]) + f" …(+{len(values) - limit})"


def clip_text(text: Any, limit: int = 400) -> str:
    value = str(text).replace("\n", " ")
    return value if len(value) <= limit else value[:limit] + "…"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_bytes(path: Path) -> bytes:
    require(path.is_file(), "input_missing", f"输入不存在：{path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PrepError("input_unreadable", f"无法读取 {path.name}：{type(exc).__name__}") from None


def jsonl_bytes(rows: Sequence[dict]) -> bytes:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    return text.encode("utf-8")


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_text_atomic(path: Path, text: str) -> None:
    write_bytes_atomic(path, text.encode("utf-8"))


def read_jsonl_strict(path: Path) -> list[dict]:
    require(path.is_file(), "input_missing", f"输入不存在：{path}")
    try:
        rows, _warnings = collector.load_pilot().read_jsonl(path, strict=True)
    except PrepError:
        raise
    except Exception as exc:                        
        message = getattr(exc, "message", None) or str(exc)
        raise PrepError("jsonl_invalid", f"{path.name}: {clip_text(message)}") from None
    require(all(isinstance(row, dict) for row in rows), "jsonl_invalid",
            f"{path.name}: 存在非对象行")
    return rows


                                                   


@dataclass(frozen=True)
class Layout:
    exp: Path
    data_root: Path
    dev_data: Path
    pilot_dir: Path
    scorer: Path


def default_layout() -> Layout:
    return Layout(exp=EXP, data_root=DATA_ROOT, dev_data=DEV_DATA,
                  pilot_dir=PILOT_DIR, scorer=SCORER_PATH)


_SCORERS: dict[tuple[str, str], Any] = {}


def load_scorer(path: Path) -> Any:
    resolved = Path(path).resolve()
    require(resolved.is_file(), "scorer_missing", f"评分器不存在：{resolved}")
    key = (str(resolved), sha_file(resolved))
    module = _SCORERS.get(key)
    if module is None:
        import importlib.util

        name = "atomic_score_" + hashlib.sha256(
            (key[0] + "|" + key[1]).encode("utf-8")).hexdigest()[:12]
        spec = importlib.util.spec_from_file_location(name, resolved)
        require(spec is not None and spec.loader is not None, "scorer_import",
                "无法为评分器创建 import spec")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _SCORERS[key] = module
    return module


def scorer_error(exc: Exception, fallback: str) -> tuple[str, str]:
    return (str(getattr(exc, "code", fallback)),
            clip_text(getattr(exc, "message", None) or str(exc)))


                                                        


@dataclass
class Base:
    layout: Layout
    scorer: Any
    case_by_id: dict[str, dict]
    ordered_cases: list[Any]
    raw_sets: dict[tuple[str, str], Any]
    ann_by_set: dict[tuple[str, str], dict]
    ledger_rows: list[dict]
    shas: dict[str, str]


def load_base(layout: Layout) -> Base:
    """用原评分器自己的校验器读已审输入；任何漂移都 fail-closed。"""
    scorer = load_scorer(layout.scorer)
    sibling = layout.dev_data.parent / layout.scorer.name
    if sibling.is_file():
        require(sha_file(sibling) == sha_file(layout.scorer), "scorer_mismatch",
                "data/_dev-v2-score 下的评分器与 pilot 评分器不是同一字节；拒绝混合输入")

    for name in INPUT_NAMES:
        require((layout.dev_data / name).is_file(), "input_missing",
                f"已审输入缺失：{layout.dev_data / name}")

    case_rows = read_jsonl_strict(layout.dev_data / "cases-v1.jsonl")
    try:
        cases = scorer.validate_cases(case_rows)
    except Exception as exc:              
        code, message = scorer_error(exc, "schema_case")
        raise PrepError(f"base_{code}", f"cases-v1 校验失败：{message}") from None
    case_by_id = {case.case_id: case for case in cases}
    require(len(case_by_id) == len(case_rows), "base_duplicate_case", "cases-v1 有重复 case_id")
    raw_case_rows = {row.get("case_id"): row for row in case_rows}
    require(len(raw_case_rows) == len(case_rows), "base_duplicate_case", "cases-v1 有重复 case_id")

    raw_rows = {arm: read_jsonl_strict(layout.dev_data / f"raw-{arm}.jsonl") for arm in ARMS}
    try:
        raw_sets, _raw_warnings = scorer.load_raw_sets(raw_rows, cases)
    except Exception as exc:
        code, message = scorer_error(exc, "schema_raw")
        raise PrepError(f"base_{code}", f"raw 校验失败：{message}") from None

    annotation_rows = read_jsonl_strict(layout.dev_data / "annotations-v1.jsonl")
    try:
        annotations, _ann_warnings = scorer.load_annotations(annotation_rows, raw_sets, cases)
    except Exception as exc:
        code, message = scorer_error(exc, "schema_annotation")
        raise PrepError(f"base_{code}", f"annotations-v1 校验失败：{message}") from None

    ledger_rows = read_jsonl_strict(layout.dev_data / "call-ledger.jsonl")
    try:
        scorer.load_ledger(ledger_rows, cases)
    except Exception as exc:
        code, message = scorer_error(exc, "schema_ledger")
        raise PrepError(f"base_{code}", f"call-ledger 校验失败：{message}") from None

    ann_rows_by_blind = {row["blind_id"]: row for row in annotation_rows}
    require(len(ann_rows_by_blind) == len(annotation_rows), "base_duplicate_blind",
            "annotations-v1 有重复 blind_id")
    ann_by_set: dict[tuple[str, str], dict] = {}
    for key, annotation in annotations.items():
        row = ann_rows_by_blind.get(annotation.blind_id)
        require(row is not None, "base_annotation_missing",
                f"{key[0]}/{key[1]}: 找不到 blind_id={annotation.blind_id} 的标注行")
        raw = raw_sets[key]
        require(annotation.blind_id == raw.blind_id, "base_blind_mismatch",
                f"{key[0]}/{key[1]}: 标注 blind_id 与 raw 不一致；拒绝复制来源不明的行")
        ann_by_set[key] = row
    require(len(ann_by_set) == len(raw_sets), "base_annotation_mismatch",
            "标注与本输入集的 raw output set 不是一一对应")

    shas = {name: sha_file(layout.dev_data / name) for name in INPUT_NAMES}
    shas["scorer"] = sha_file(layout.scorer)
    ordered = [case_by_id[case_id] for case_id in sorted(case_by_id)]
    return Base(layout=layout, scorer=scorer,
                case_by_id=raw_case_rows, ordered_cases=ordered,
                raw_sets=raw_sets, ann_by_set=ann_by_set, ledger_rows=ledger_rows,
                shas=shas)


                                                               


@dataclass
class Plan:
    case_id: str
    split: str
    case_row: dict
    raw: Any
    raw_row: dict
    mode: str                               
    atoms: list[dict] | None
    placeholder: dict | None
    base_blind_id: str
    base_attempt: int

    @property
    def reused_atoms(self) -> int:
        return len(self.atoms) if self.mode == "reuse" else 0


def _same_candidate_set(new_raw: Any, base_raw: Any) -> bool:
    if len(new_raw.candidates) != len(base_raw.candidates):
        return False
    for new, old in zip(new_raw.candidates, base_raw.candidates):
        if (new.text, new.evidence, new.tier, new.trigger) != (
                old.text, old.evidence, old.tier, old.trigger):
            return False
    return True


def _reuse_atoms(base_ann_row: dict, base_raw: Any, new_raw: Any) -> list[dict]:
    index_of = {candidate.candidate_id: index
                for index, candidate in enumerate(base_raw.candidates)}
    mapped: list[tuple[int, dict]] = []
    seen: set[int] = set()
    for atom in base_ann_row.get("atoms") or []:
        index = index_of.get(atom.get("candidate_id"))
        require(index is not None, "reuse_atom_mismatch",
                f"旧标注 atom 引用了旧 raw 之外的 candidate：{atom.get('candidate_id')}")
        require(index not in seen, "reuse_atom_mismatch", "旧标注 atom 有重复 candidate")
        seen.add(index)
        item = dict(atom)
        item["candidate_id"] = new_raw.candidates[index].candidate_id
        mapped.append((index, item))
    require(len(seen) == len(new_raw.candidates), "reuse_atom_mismatch",
            "旧标注没有覆盖整个输出集；不能整集复用")
    mapped.sort(key=lambda pair: pair[0])
    return [item for _index, item in mapped]


def placeholder_atom(candidate_id: str) -> dict:
    return {"candidate_id": candidate_id, "entailment": "supported",
            "persistence_appropriate": True, "non_atomic": False, "over_split": False,
            "context_errors": [], "matched_forbidden_ids": [], "duplicate_group": None,
            "core_entailed_gold_ids": [], "fully_covered_gold_ids": [],
            "alternative_partition": None}


def a1_row(plan: Plan, atoms: list[dict], *, complete: bool) -> dict:
    return {"blind_id": plan.raw.blind_id,
            "stage1_complete": complete, "stage2_complete": complete,
            "adjudicated": complete, "atoms": atoms,
            "case_id": plan.case_id, "split": plan.split}


def pending_row(plan: Plan) -> dict:
    """待审包：保留原 blind 视图 + 模板；不含 model/channel/kept_status。"""
    pilot = collector.load_pilot()
    blind = pilot.build_blind_row(plan.case_row, ARM, plan.raw_row, 2)
    require(blind["blind_id"] == plan.raw.blind_id, "packet_blind_id",
            f"{plan.case_id}: packet blind_id 与 raw 不一致")
    template = a1_row(plan, [placeholder_atom(c.candidate_id) for c in plan.raw.candidates],
                      complete=False)
    packet = {
        "schema": REVIEW_SCHEMA,
        "blind_id": blind["blind_id"],
        "case_id": plan.case_id,
        "split": plan.split,
        "conversations": blind["conversations"],
        "forbidden_inferences": blind["forbidden_inferences"],
        "one_off_hints": blind["one_off_hints"],
        "gold_obligations": blind["gold_obligations"],
        "candidates": blind["candidates"],
        "annotation_template": template,
        "instructions": INSTRUCTIONS,
    }
    leaked = [key for key in ("model", "channel", "provider", "status", "kept_status")
              if key in packet]
    require(not leaked, "packet_leak", f"待审包泄露实验字段：{leaked}")
    for candidate in packet["candidates"]:
        require("kept_status" not in candidate, "packet_leak",
                f"{plan.case_id}: 候选泄露 kept_status")
    return packet


def _select_new_rows(rows: list[dict], base: Base, scorer: Any) -> dict[str, dict]:
    last: dict[str, dict] = {}
    attempts: dict[str, list[int]] = {}
    for index, row in enumerate(rows, 1):
        require(isinstance(row, dict), "new_raw_schema", f"raw[{index}] 须为对象")
        case_id = row.get("case_id")
        require(isinstance(case_id, str) and case_id in base.case_by_id, "new_raw_unknown_case",
                f"raw[{index}] 引用了 cases 之外的 case：{case_id!r}")
        require(row.get("arm") == ARM, "new_raw_arm",
                f"{case_id}: arm={row.get('arm')!r}，本工具只接 {ARM}")
        require(row.get("split") == base.case_by_id[case_id]["split"], "new_raw_split",
                f"{case_id}: split={row.get('split')!r} 与 cases 不一致")
        attempt = row.get("attempt")
        require(isinstance(attempt, int) and not isinstance(attempt, bool) and attempt >= 1,
                "new_raw_attempt", f"{case_id}: attempt={attempt!r} 须为 >=1 整数")
        attempts.setdefault(case_id, []).append(attempt)
        last[case_id] = row
    missing = sorted(set(base.case_by_id) - set(last))
    require(not missing, "new_raw_missing_cases",
            f"新 raw 还缺 {len(missing)} 个 case：{clip(missing)}")
    for case_id, sequence in attempts.items():
        require(all(left <= right for left, right in zip(sequence, sequence[1:])),
                "new_raw_attempt_order",
                f"{case_id}: attempt 在文件中非递增（{sequence}），last-write-wins 会选错")
    bad = [(case_id, last[case_id].get("status")) for case_id in sorted(last)
           if last[case_id].get("status") not in scorer.STATUS_VALUES]
    require(not bad, "raw_status_unsupported",
            f"原评分器不接受这些 status：{clip([f'{c}:{s}' for c, s in bad])}；"
            "请补跑这些 case，不要改状态或伪造")
    return last


def _parse_new_row(scorer: Any, row: dict, case: Any) -> Any:
    try:
        return scorer.parse_raw_latest(row, ARM, case)
    except Exception as exc:
        code, message = scorer_error(exc, "schema_raw")
        raise PrepError(f"new_raw_{code}", f"{case.case_id}: {message}") from None


def _validate_new_ledger(rows: list[dict], base: Base, plans: list[Plan]) -> tuple[list[dict], int]:
    by_attempt: dict[tuple[str, int], dict] = {}
    for index, row in enumerate(rows, 1):
        require(isinstance(row, dict), "new_ledger_schema", f"ledger[{index}] 须为对象")
        case_id = row.get("case_id")
        require(isinstance(case_id, str) and case_id in base.case_by_id, "new_ledger_case",
                f"ledger[{index}] 引用未知 case：{case_id!r}")
        require(row.get("arm") == ARM, "new_ledger_arm",
                f"{case_id}: ledger arm={row.get('arm')!r}，只接受 {ARM}")
        attempt = row.get("attempt")
        require(isinstance(attempt, int) and not isinstance(attempt, bool) and attempt >= 1,
                "new_ledger_attempt", f"{case_id}: ledger attempt={attempt!r} 非法")
        split = row.get("split", base.case_by_id[case_id]["split"])
        require(split == base.case_by_id[case_id]["split"], "new_ledger_split",
                f"{case_id}: ledger split 与 cases 不一致")
        usage = row.get("usage")
        require(isinstance(usage, dict)
                and all(key in usage for key in
                        ("prompt_tokens", "completion_tokens", "total_tokens")),
                "new_ledger_usage", f"{case_id}/{ARM}#{attempt}: usage 缺 token 字段")
        if "attempted" in row:
            require(row["attempted"] is True, "new_ledger_attempted",
                    f"{case_id}/{ARM}#{attempt}: attempted 不为 true，不能作为已跑成本行")
        by_attempt[(case_id, attempt)] = row

    for plan in plans:
        row = by_attempt.get((plan.case_id, plan.raw.attempt))
        require(row is not None, "ledger_missing",
                f"{plan.case_id}: 缺 A1 attempt={plan.raw.attempt} 的 ledger 行")
        raw_total = (plan.raw.usage or {}).get("total_tokens")
        ledger_total = (row.get("usage") or {}).get("total_tokens")
        require(raw_total == ledger_total, "ledger_usage_mismatch",
                f"{plan.case_id}: raw total_tokens={raw_total} != ledger {ledger_total}")

    out: list[dict] = []
    adapted = 0
    for row in rows:
        item = dict(row)
        if "elapsed_s" not in item and "elapsed_seconds" not in item:
            require("latency_s" in item, "ledger_elapsed_missing",
                    f"{row.get('case_id')}: ledger 既无 elapsed_s 也无 latency_s")
            latency = item.get("latency_s")
            require(isinstance(latency, (int, float)) and not isinstance(latency, bool)
                    and float(latency) >= 0, "ledger_elapsed_missing",
                    f"{row.get('case_id')}: latency_s 非法")
            item["elapsed_s"] = float(latency)
            adapted += 1
        out.append(item)
    return out, adapted


def plan_for_model(layout: Layout, base: Base, model: str) -> tuple[list[Plan], dict]:
    paths = collector.paths_for(model, layout.data_root)
    require(paths.raw.is_file(), "new_raw_missing", f"新模型 raw 不存在：{paths.raw}")
    require(paths.ledger.is_file(), "new_ledger_missing",
            f"新模型 call-ledger 不存在：{paths.ledger}")
    new_rows = read_jsonl_strict(paths.raw)
    require(new_rows, "new_raw_empty", "新模型 raw 为空")
    selected = _select_new_rows(new_rows, base, base.scorer)
    pilot = collector.load_pilot()

    plans: list[Plan] = []
    for case in base.ordered_cases:
        row = selected[case.case_id]
        raw = _parse_new_row(base.scorer, row, case)
        require(isinstance(row.get("content_sha256"), str)
                and row["content_sha256"] == raw.content_sha256, "new_raw_content_sha",
                f"{case.case_id}: content_sha256 缺失或与 candidates 不一致")
        require(pilot.blind_id_of(case.case_id, ARM, row) == raw.blind_id, "new_raw_blind_id",
                f"{case.case_id}: blind_id 与 raw 不一致")
        for index, candidate in enumerate(row.get("parsed_candidates") or []):
            require(candidate.get("candidate_id") == pilot.candidate_id(
                case.case_id, ARM, raw.attempt, index, candidate), "new_raw_candidate_id",
                f"{case.case_id}[{index}]: candidate_id 与 pilot 公式不符")
        base_raw = base.raw_sets[(case.case_id, ARM)]
        base_ann = base.ann_by_set[(case.case_id, ARM)]
        if _same_candidate_set(raw, base_raw):
            plan = Plan(case_id=case.case_id, split=case.split,
                        case_row=base.case_by_id[case.case_id], raw=raw, raw_row=row,
                        mode="reuse", atoms=_reuse_atoms(base_ann, base_raw, raw),
                        placeholder=None, base_blind_id=base_raw.blind_id,
                        base_attempt=base_raw.attempt)
        elif raw.status == "valid_empty":
            plan = Plan(case_id=case.case_id, split=case.split,
                        case_row=base.case_by_id[case.case_id], raw=raw, raw_row=row,
                        mode="auto_empty", atoms=[], placeholder=None,
                        base_blind_id=base_raw.blind_id, base_attempt=base_raw.attempt)
        else:
            plan = Plan(case_id=case.case_id, split=case.split,
                        case_row=base.case_by_id[case.case_id], raw=raw, raw_row=row,
                        mode="review", atoms=None, placeholder=None,
                        base_blind_id=base_raw.blind_id, base_attempt=base_raw.attempt)
            plan.placeholder = pending_row(plan)["annotation_template"]
        plans.append(plan)

    new_ledger_rows = read_jsonl_strict(paths.ledger)
    adapted_rows, adapted = _validate_new_ledger(new_ledger_rows, base, plans)
    old_ledger = [dict(row) for row in base.ledger_rows if row.get("arm") in ("P0", "A2")]
    info = {
        "paths": paths,
        "new_rows": new_rows,
        "new_ledger_rows": new_ledger_rows,
        "ledger_rows": old_ledger + adapted_rows,
        "a1_superseded": len(new_rows) - len(selected),
        "ledger_adapted": adapted,
        "new_raw_sha": sha_file(paths.raw),
        "new_ledger_sha": sha_file(paths.ledger),
    }
    return plans, info


def mode_counts(plans: Sequence[Plan]) -> dict:
    modes = {"reuse": [], "review": [], "auto_empty": []}
    for plan in plans:
        modes[plan.mode].append(plan.case_id)
    return modes


                                                   


def prepare(layout: Layout, model: str) -> dict:
    base = load_base(layout)
    plans, info = plan_for_model(layout, base, model)
    paths = info["paths"]
    require(paths.root.is_dir(), "model_dir_missing",
            f"模型目录不存在（先跑采集）：{paths.root}")
    modes = mode_counts(plans)

    working: list[dict] = [dict(base.ann_by_set[(case.case_id, "P0")])
                           for case in base.ordered_cases]
    for plan in plans:
        if plan.mode == "review":
            working.append(a1_row(plan, plan.placeholder["atoms"], complete=False))
        else:
            working.append(a1_row(plan, plan.atoms, complete=True))
    working.extend(dict(base.ann_by_set[(case.case_id, "A2")])
                   for case in base.ordered_cases)

    pending = [pending_row(plan) for plan in plans if plan.mode == "review"]

    working_path = paths.root / "annotations-working.jsonl"
    pending_path = paths.root / "review-pending.jsonl"
    summary_path = paths.root / "reuse-summary.json"
    working_payload = jsonl_bytes(working)
    pending_payload = jsonl_bytes(pending)
    write_bytes_atomic(working_path, working_payload)
    write_bytes_atomic(pending_path, pending_payload)

    per_case = []
    for plan in plans:
        per_case.append({
            "case_id": plan.case_id, "split": plan.split, "mode": plan.mode,
            "status": plan.raw.status, "attempt": plan.raw.attempt,
            "base_attempt": plan.base_attempt,
            "candidates": len(plan.raw.candidates),
            "base_candidates": len(base.raw_sets[(plan.case_id, ARM)].candidates),
            "atoms": len(plan.atoms) if plan.atoms is not None
            else len(plan.placeholder["atoms"]),
            "blind_id": plan.raw.blind_id, "base_blind_id": plan.base_blind_id,
        })
    reused_atoms = sum(plan.reused_atoms for plan in plans)
    pending_candidates = sum(len(plan.raw.candidates) for plan in plans
                             if plan.mode == "review")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "model": model,
        "generated": now_iso(),
        "compare_fields": list(COMPARE_FIELDS),
        "reuse_rule": "整集有序候选列表在 text/evidence/tier/trigger 上完全一致才复用；"
                      "否则该 case 全部候选重审",
        "inputs": {
            "new_raw": {"path": str(paths.raw), "sha256": info["new_raw_sha"],
                        "rows": len(info["new_rows"])},
            "new_ledger": {"path": str(paths.ledger), "sha256": info["new_ledger_sha"],
                           "rows": len(info["new_ledger_rows"])},
            "scorer": {"path": str(layout.scorer), "sha256": base.shas["scorer"]},
            "base": {name: {"sha256": base.shas[name]} for name in INPUT_NAMES},
            "base_dir": str(layout.dev_data),
        },
        "counts": {
            "cases": len(plans), "reused": len(modes["reuse"]),
            "review": len(modes["review"]), "auto_empty": len(modes["auto_empty"]),
            "reused_atoms": reused_atoms, "pending_candidates": pending_candidates,
            "working_rows": len(working), "pending_rows": len(pending),
        },
        "modes": modes,
        "cases": per_case,
        "outputs": {
            "annotations_working": {"path": str(working_path),
                                    "sha256": sha_bytes(working_payload),
                                    "rows": len(working)},
            "review_pending": {"path": str(pending_path),
                               "sha256": sha_bytes(pending_payload),
                               "rows": len(pending)},
        },
        "notes": ["P0/A2 为已审 v2 输入原样复制，仅供评分器结构",
                  "报告只比较 A1 与 P0，不宣称跑了新 A2",
                  "placeholder adjudicated=false，绝不可直接当 final"],
    }
    write_text_atomic(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return {
        "ok": True, "command": "prepare", "model": model,
        "model_dir": str(paths.root),
        "counts": summary["counts"], "modes": modes,
        "pending_blind_ids": [plan.raw.blind_id for plan in plans if plan.mode == "review"],
        "outputs": {"annotations_working": str(working_path),
                    "review_pending": str(pending_path),
                    "reuse_summary": str(summary_path)},
        "next": "把 review-pending.jsonl 派给主审；完成后 finalize --annotations <完成JSONL>",
    }


                                                    


def _verify_prepare_state(layout: Layout, base: Base, paths: Any,
                          plans: list[Plan]) -> None:
    root = paths.root
    for name in ("annotations-working.jsonl", "review-pending.jsonl", "reuse-summary.json"):
        require((root / name).is_file(), "prepare_missing", f"缺少 {name}；先跑 prepare")
    try:
        summary = json.loads((root / "reuse-summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrepError("prepare_missing", f"reuse-summary.json 不可读：{type(exc).__name__}") from None
    require(summary.get("schema") == SUMMARY_SCHEMA, "prepare_stale",
            "reuse-summary.json schema 不符；重跑 prepare")
    require(summary.get("model") == paths.model, "prepare_stale", "模型 ID 不一致；重跑 prepare")
    inputs = summary.get("inputs") or {}
    checks = [("new_raw", sha_file(paths.raw)), ("new_ledger", sha_file(paths.ledger)),
              ("scorer", base.shas["scorer"])]
    for key, current in checks:
        recorded = (inputs.get(key) or {}).get("sha256")
        require(recorded == current, "prepare_stale",
                f"{key} 在 prepare 后已变化；重跑 prepare")
    for name in INPUT_NAMES:
        recorded = ((inputs.get("base") or {}).get(name) or {}).get("sha256")
        require(recorded == base.shas[name], "prepare_stale",
                f"已审输入 {name} 在 prepare 后变化；重跑 prepare")
    outputs = summary.get("outputs") or {}
    for key, name in (("annotations_working", "annotations-working.jsonl"),
                      ("review_pending", "review-pending.jsonl")):
        recorded = (outputs.get(key) or {}).get("sha256")
        require(recorded == sha_file(root / name), "prepare_stale",
                f"{name} 被改动；重跑 prepare")

    pending_rows = read_jsonl_strict(root / "review-pending.jsonl")
    expected = {plan.raw.blind_id: plan for plan in plans if plan.mode == "review"}
    require(len(pending_rows) == len(expected), "prepare_stale",
            f"review-pending 行数 {len(pending_rows)} != 当前待审 {len(expected)}；重跑 prepare")
    for row in pending_rows:
        blind_id = row.get("blind_id")
        require(blind_id in expected, "prepare_stale",
                f"review-pending 含未知 blind_id：{blind_id}；重跑 prepare")
        plan = expected[blind_id]
        require(row.get("case_id") == plan.case_id, "prepare_stale",
                f"{blind_id}: case_id 与当前 raw 不一致；重跑 prepare")
        ids = [c.get("candidate_id") for c in (row.get("candidates") or [])]
        require(ids == [c.candidate_id for c in plan.raw.candidates], "prepare_stale",
                f"{blind_id}: 候选列表与当前 raw 不一致；重跑 prepare")


def load_review(path: Path, plans: Sequence[Plan], scorer: Any) -> dict[str, dict]:
    rows = read_jsonl_strict(path)
    pending = {plan.raw.blind_id: plan for plan in plans if plan.mode == "review"}
    seen: dict[str, dict] = {}
    for index, row in enumerate(rows, 1):
        require(isinstance(row, dict), "review_schema", f"review[{index}] 须为对象")
        missing = sorted(scorer.ANNOTATION_KEYS - set(row))
        extra = sorted(set(row) - scorer.ANNOTATION_KEYS - scorer.ANNOTATION_OPTIONAL)
        require(not missing and not extra, "review_schema",
                f"review[{index}] 字段不符（缺 {missing}，多 {extra}）")
        blind_id = row.get("blind_id")
        require(isinstance(blind_id, str) and bool(blind_id), "review_schema",
                f"review[{index}].blind_id 须为非空字符串")
        require(blind_id in pending, "review_unknown_blind",
                f"review[{index}]: blind_id 不在待审集合（{blind_id}）；"
                f"期望 {len(pending)} 个：{clip(sorted(pending))}")
        require(blind_id not in seen, "review_duplicate",
                f"review[{index}]: blind_id 重复 {blind_id}")
        for key in ("stage1_complete", "stage2_complete", "adjudicated"):
            require(row.get(key) is True, "review_incomplete",
                    f"{blind_id}.{key} 必须为 true（placeholder 不算完成）")
        plan = pending[blind_id]
        if row.get("case_id") is not None:
            require(row["case_id"] == plan.case_id, "review_case_mismatch",
                    f"{blind_id}: case_id={row.get('case_id')!r} 与 raw 不一致")
        atoms = row.get("atoms")
        require(isinstance(atoms, list), "review_schema", f"{blind_id}.atoms 须为数组")
        require(len(atoms) == len(plan.raw.candidates), "review_candidate_mismatch",
                f"{blind_id}: atoms={len(atoms)} 与 candidates="
                f"{len(plan.raw.candidates)} 不一致")
        for atom_index, (atom, candidate) in enumerate(zip(atoms, plan.raw.candidates)):
            try:
                scorer.validate_atom(atom, f"review[{blind_id}].atoms[{atom_index}]")
            except Exception as exc:
                code, message = scorer_error(exc, "review_atom_schema")
                raise PrepError("review_atom_schema",
                                f"{blind_id}[{atom_index}]: {code}: {message}") from None
            require(atom.get("candidate_id") == candidate.candidate_id,
                    "review_candidate_mismatch",
                    f"{blind_id}[{atom_index}]: candidate_id 与 raw 不一致（要求顺序一致）")
        seen[blind_id] = row
    missing = sorted(set(pending) - set(seen))
    require(not missing, "review_missing",
            f"还有 {len(missing)} 个待审 case 未完成：{clip(missing)}")
    return seen


def final_annotations(base: Base, plans: Sequence[Plan], review: dict[str, dict]) -> list[dict]:
    rows: list[dict] = [dict(base.ann_by_set[(case.case_id, "P0")])
                        for case in base.ordered_cases]
    for plan in plans:
        if plan.mode == "review":
            atoms = [dict(atom) for atom in review[plan.raw.blind_id]["atoms"]]
        else:
            atoms = [dict(atom) for atom in plan.atoms or []]
        rows.append(a1_row(plan, atoms, complete=True))
    rows.extend(dict(base.ann_by_set[(case.case_id, "A2")])
                for case in base.ordered_cases)
    for row in rows:
        require(row.get("adjudicated") is True, "final_incomplete",
                f"final 标注 {row.get('blind_id')} 的 adjudicated 不为 true")
    return rows


def _dry_run(scorer: Any, stage: Path, scorer_name: str) -> dict:
    data = stage / "data"
    try:
        cases_rows, cases_meta = scorer.read_jsonl_file(data / "cases-v1.jsonl")
        raw_rows: dict[str, list[dict]] = {}
        input_meta = {"cases-v1.jsonl": cases_meta}
        for arm in ARMS:
            rows, meta = scorer.read_jsonl_file(data / f"raw-{arm}.jsonl")
            raw_rows[arm] = rows
            input_meta[f"raw-{arm}.jsonl"] = meta
        ledger_rows, ledger_meta = scorer.read_jsonl_file(data / "call-ledger.jsonl")
        input_meta["call-ledger.jsonl"] = ledger_meta
        ann_rows, ann_meta = scorer.read_jsonl_file(data / "annotations-v1.jsonl")
        input_meta["annotations-v1.jsonl"] = ann_meta
        result = scorer.score_bundle(
            cases_rows, raw_rows, ledger_rows, ann_rows,
            input_sha256=input_meta, script_sha256=sha_file(stage / scorer_name),
            synthetic_only=False)
    except Exception as exc:
        code, message = scorer_error(exc, "score_contract")
        raise PrepError("score_contract",
                        f"原评分器拒绝装配结果（{code}）：{clip_text(message)}") from None
    counts = result.get("counts") or {}
    return {
        "counts": {key: counts.get(key) for key in
                   ("cases", "dev", "confirmation", "raw_sets", "annotations",
                    "matched_output_sets", "annotation_atoms", "ledger_rows")},
        "gates": {arm: bool((result.get("gates") or {}).get(arm, {}).get("passed"))
                  for arm in ("A1", "A2")},
        "warnings": len((result.get("validation") or {}).get("warnings") or []),
    }


def _manifest(layout: Layout, model: str, base: Base, plans: Sequence[Plan],
              info: dict, final_rows: Sequence[dict], payloads: dict[str, bytes],
              dry: dict, review_path: Path, reviewed_rows: int) -> dict:
    modes = mode_counts(plans)
    atoms_total = sum(len(row.get("atoms") or []) for row in final_rows)
    return {
        "schema": MANIFEST_SCHEMA,
        "created": now_iso(),
        "model": model,
        "command": "prepare_atomization_score.py finalize",
        "comparison_scope": {
            "report_arms": ["A1", "P0"],
            "a2": "P0/A2 raw 与标注为已审 v2 输入原样复制，仅为满足评分器结构；"
                  "不宣称跑了新 A2",
        },
        "scorer": {"name": layout.scorer.name, "source": str(layout.scorer),
                   "sha256": base.shas["scorer"]},
        "prepare_script": {"name": Path(__file__).name, "sha256": sha_file(Path(__file__))},
        "review": {"path": str(review_path), "sha256": sha_file(review_path),
                   "rows": reviewed_rows},
        "source": {
            "dev_data": str(layout.dev_data),
            "new_raw": str(info["paths"].raw),
            "new_ledger": str(info["paths"].ledger),
            "new_raw_sha256": info["new_raw_sha"],
            "new_ledger_sha256": info["new_ledger_sha"],
            "base": {name: base.shas[name] for name in INPUT_NAMES},
        },
        "inputs": {name: {"sha256": sha_bytes(payloads[name]), "bytes": len(payloads[name])}
                   for name in INPUT_NAMES},
        "counts": {
            "cases": len(plans),
            "rows": len(final_rows),
            "annotations": len(final_rows),
            "atoms": atoms_total,
            "by_arm": {
                "P0": sum(1 for row in final_rows if row.get("blind_id") in {
                    base.raw_sets[(case.case_id, "P0")].blind_id
                    for case in base.ordered_cases}),
                "A1": sum(1 for row in final_rows
                          if row.get("blind_id") in {plan.raw.blind_id for plan in plans}),
                "A2": sum(1 for row in final_rows if row.get("blind_id") in {
                    base.raw_sets[(case.case_id, "A2")].blind_id
                    for case in base.ordered_cases}),
            },
            "a1_rows": len(info["new_rows"]),
            "a1_superseded": info["a1_superseded"],
            "ledger_rows": len(info["ledger_rows"]),
            "reused": len(modes["reuse"]),
            "reviewed": len(modes["review"]),
            "auto_empty": len(modes["auto_empty"]),
        },
        "reuse": {
            "rule": "整集有序候选列表 text/evidence/tier/trigger 全同才复用",
            "modes": modes,
            "cases": [{"case_id": plan.case_id, "mode": plan.mode,
                       "status": plan.raw.status, "attempt": plan.raw.attempt,
                       "candidates": len(plan.raw.candidates),
                       "atoms": len(plan.atoms) if plan.atoms is not None
                       else len(plan.placeholder["atoms"]),
                       "blind_id": plan.raw.blind_id,
                       "base_blind_id": plan.base_blind_id} for plan in plans],
        },
        "ledger_adapter": {
            "fields": {"latency_s": "elapsed_s"},
            "rows_adapted": info["ledger_adapted"],
            "note": "新采集器 ledger 写 latency_s；原评分器只认 elapsed_s，"
                    "同值改名，不改数值、不补行",
        },
        "dry_run": dry,
    }


def _existing_score_dir(score_dir: Path, model: str, base: Base,
                        payloads: dict[str, bytes], allow_same_sha: bool) -> dict:
    manifest_path = score_dir / "manifest.json"
    require(allow_same_sha, "score_dir_exists",
            f"已存在 score 目录：{score_dir}；确认输入完全一致后加 --allow-same-sha")
    require(manifest_path.is_file(), "score_dir_conflict",
            "已存在 score 目录但没有 manifest；拒绝覆盖")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrepError("score_dir_conflict", f"manifest 不可读：{type(exc).__name__}") from None
    require(manifest.get("schema") == MANIFEST_SCHEMA, "score_dir_conflict",
            "已有 score/manifest.json schema 不符")
    same = (manifest.get("model") == model
            and (manifest.get("scorer") or {}).get("sha256") == base.shas["scorer"])
    if same:
        for name in INPUT_NAMES:
            recorded = ((manifest.get("inputs") or {}).get(name) or {}).get("sha256")
            if recorded != sha_bytes(payloads[name]):
                same = False
                break
    require(same, "score_inputs_changed",
            "已有 score 目录的输入与本次不一致；不覆盖，请人工确认后换目录或删除")
    for name in INPUT_NAMES:
        path = score_dir / "data" / name
        require(path.is_file() and sha_file(path) == sha_bytes(payloads[name]),
                "score_dir_tampered",
                f"已有 score/data/{name} 与 manifest 记录不一致；拒绝视为同一结果")
    return {"ok": True, "command": "finalize", "status": "already_current",
            "model": model, "score_dir": str(score_dir),
            "manifest": str(manifest_path),
            "note": "输入与已有 score 完全一致，未重写任何文件"}


def finalize(layout: Layout, model: str, annotations: Path,
             allow_same_sha: bool = False) -> dict:
    base = load_base(layout)
    plans, info = plan_for_model(layout, base, model)
    paths = info["paths"]
    _verify_prepare_state(layout, base, paths, plans)
    review = load_review(Path(annotations), plans, base.scorer)
    final_rows = final_annotations(base, plans, review)

    payloads = {
        "cases-v1.jsonl": file_bytes(layout.dev_data / "cases-v1.jsonl"),
        "raw-P0.jsonl": file_bytes(layout.dev_data / "raw-P0.jsonl"),
        "raw-A2.jsonl": file_bytes(layout.dev_data / "raw-A2.jsonl"),
        "raw-A1.jsonl": jsonl_bytes(info["new_rows"]),
        "call-ledger.jsonl": jsonl_bytes(info["ledger_rows"]),
        "annotations-v1.jsonl": jsonl_bytes(final_rows),
    }
    score_dir = paths.root / "score"
    if score_dir.exists():
        return _existing_score_dir(score_dir, model, base, payloads, allow_same_sha)

    stage = Path(tempfile.mkdtemp(prefix=".score-staging-", dir=str(paths.root)))
    try:
        data_dir = stage / "data"
        data_dir.mkdir()
        shutil.copyfile(layout.scorer, stage / layout.scorer.name)
        require(sha_file(stage / layout.scorer.name) == base.shas["scorer"],
                "scorer_copy", "评分器字节副本 SHA 不一致")
        for name in INPUT_NAMES:
            with open(data_dir / name, "wb") as handle:
                handle.write(payloads[name])
        dry = _dry_run(base.scorer, stage, layout.scorer.name)
        manifest = _manifest(layout, model, base, plans, info, final_rows, payloads,
                             dry, Path(annotations), len(review))
        write_text_atomic(stage / "manifest.json",
                          json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        os.replace(stage, score_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)

    modes = mode_counts(plans)
    return {
        "ok": True, "command": "finalize", "status": "assembled", "model": model,
        "score_dir": str(score_dir),
        "rows": len(final_rows), "by_arm": manifest["counts"]["by_arm"],
        "modes": modes, "counts": manifest["counts"],
        "dry_run": dry, "manifest": str(score_dir / "manifest.json"),
        "compare_arms": ["A1", "P0"],
        "note": "A1 为新模型；A2 仅为结构复用，不宣称新 A2",
    }


                                                    


def _fixture_candidate(pilot: Any, case_id: str, arm: str, attempt: int, index: int,
                       text: str, *, summary: str | None = None,
                       tier: str | None = None, trigger: str | None = None) -> dict:
    item = {"text": text, "summary": summary or f"summary-{index}", "evidence": [text]}
    if tier is not None:
        item["tier"] = tier
    if trigger is not None:
        item["trigger"] = trigger
    candidate = {"candidate_id": pilot.candidate_id(case_id, arm, attempt, index, item),
                 "kept_status": "kept"}
    candidate.update(item)
    return candidate


def _fixture_raw(pilot: Any, case_id: str, arm: str, attempt: int, candidates: list[dict],
                 *, status: str = "ok", raw_item_count: int | None = None,
                 total_tokens: int = 15, raw_text: str | None = None) -> dict:
    count = len(candidates) if raw_item_count is None else raw_item_count
    if raw_text is None:
        raw_text = json.dumps(
            [{key: value for key, value in item.items()
              if key not in ("candidate_id", "kept_status")} for item in candidates],
            ensure_ascii=False)
    return {
        "schema": "atomic-pilot-raw-v1", "ts": "2026-10-25T00:00:00+0800",
        "case_id": case_id, "split": "dev", "arm": arm, "attempt": attempt,
        "status": status, "prompt": f"prompt {case_id} {arm} {attempt}",
        "raw_text": raw_text, "raw_item_count": count,
        "parsed_candidates": candidates,
        "usage": {"prompt_tokens": max(total_tokens - 5, 0), "completion_tokens": 5,
                  "total_tokens": total_tokens},
        "elapsed_s": 0.5, "finish_reason": "stop",
        "call": {"attempted": True, "succeeded": True},
        "content_sha256": pilot.content_sha(candidates),
    }


def _fixture_case(case_id: str, query: str) -> dict:
    return {"schema": "atomic-pilot-case-v1", "case_id": case_id, "family": "fx",
            "cluster_id": f"{case_id}-c", "split": "dev",
            "conversations": [{"query": query, "chat": []}],
            "forbidden_inferences": [{"forbidden_id": f"{case_id}-x1",
                                      "description": "fixture forbidden",
                                      "severity": "major"}],
            "one_off_hints": [], "author_notes": "fixture",
            "gold_obligations": [{
                "gold_id": f"{case_id}-g1", "kind": "rule", "tier": "explicit",
                "meaning": "fixture gold", "evidence_conv_ids": [0],
                "evidence_spans": [{"conv_id": 0, "text": query}],
                "context_requirements": ["scope"]}]}


def _fixture_annotation(pilot: Any, case_id: str, arm: str, row: dict,
                        candidates: list[dict], notes: str | None = None) -> dict:
    atoms = []
    for candidate in candidates:
        atom = {"candidate_id": candidate["candidate_id"], "entailment": "supported",
                "persistence_appropriate": True, "non_atomic": False, "over_split": False,
                "context_errors": [], "matched_forbidden_ids": [],
                "duplicate_group": None, "core_entailed_gold_ids": [f"{case_id}-g1"],
                "fully_covered_gold_ids": [], "alternative_partition": None}
        if notes is not None:
            atom["notes"] = notes
        atoms.append(atom)
    return {"blind_id": pilot.blind_id_of(case_id, arm, row),
            "stage1_complete": True, "stage2_complete": True, "adjudicated": True,
            "atoms": atoms}


def _fixture_ledger(case_id: str, arm: str, attempt: int, total_tokens: int,
                    *, new_style: bool) -> dict:
    row = {"schema": "atomic-pilot-call-v1", "ts": "2026-10-25T00:00:00+0800",
           "case_id": case_id, "split": "dev", "arm": arm, "attempt": attempt,
           "usage": {"prompt_tokens": max(total_tokens - 5, 0),
                     "completion_tokens": 5, "total_tokens": total_tokens}}
    if new_style:
        row["latency_s"] = 0.5
        row["channel"] = "fixture"
        row["status"] = "ok"
        row["usage"]["source"] = "fixture"
    else:
        row["elapsed_s"] = 0.5
    return row


@dataclass
class Fixture:
    layout: Layout
    model: str
    paths: Any
    cases: list[dict]
    base_annotations: list[dict]
    base_raw: dict[str, list[dict]]
    new_rows: list[dict]
    new_ledger: list[dict]


def build_fixture(root: Path) -> Fixture:
    pilot = collector.load_pilot()
    base_dir = root / "研究"
    pilot_dir = base_dir / "pilot"
    dev_score = pilot_dir / "data" / "_dev-v2-score"
    dev_data = dev_score / "data"
    dev_data.mkdir(parents=True)
    require(SCORER_PATH.is_file(), "scorer_missing", f"真实评分器不存在：{SCORER_PATH}")
    shutil.copyfile(SCORER_PATH, dev_score / SCORER_PATH.name)
    shutil.copyfile(SCORER_PATH, pilot_dir / SCORER_PATH.name)
    layout = Layout(exp=base_dir / "exp", data_root=base_dir / "exp" / "data",
                    dev_data=dev_data, pilot_dir=pilot_dir,
                    scorer=pilot_dir / SCORER_PATH.name)
    model = "fixture-model"
    paths = collector.paths_for(model, layout.data_root)
    paths.root.mkdir(parents=True, exist_ok=True)

    case_ids = ["FX-001", "FX-002", "FX-003", "FX-004", "FX-005"]
    cases = [_fixture_case(case_id, f"query {case_id}") for case_id in case_ids]

    p0 = {case_id: [_fixture_candidate(pilot, case_id, "P0", 1, 0, f"p0-{case_id}")]
          for case_id in case_ids}
    base_a1 = {
        "FX-001": [_fixture_candidate(pilot, "FX-001", ARM, 1, 0, "a1x",
                                      tier="explicit", trigger="when q")],
        "FX-002": [_fixture_candidate(pilot, "FX-002", ARM, 1, 0, "a2x"),
                   _fixture_candidate(pilot, "FX-002", ARM, 1, 1, "a2y")],
        "FX-003": [_fixture_candidate(pilot, "FX-003", ARM, 1, 0, "a3x")],
        "FX-004": [_fixture_candidate(pilot, "FX-004", ARM, 1, 0, "a4x"),
                   _fixture_candidate(pilot, "FX-004", ARM, 1, 1, "a4y")],
        "FX-005": [_fixture_candidate(pilot, "FX-005", ARM, 1, 0, "a5x")],
    }
    a2 = {case_id: [dict(_fixture_candidate(pilot, case_id, "A2", 1, 0, f"a2-{case_id}"),
                         parent_p0_candidate_id="unknown",
                         parent_content_sha=None, parent_p0_attempt_sha=None,
                         parent_gate_status="unknown")] for case_id in case_ids}

    base_raw = {
        "P0": [_fixture_raw(pilot, case_id, "P0", 1, p0[case_id]) for case_id in case_ids],
        "A1": [_fixture_raw(pilot, case_id, ARM, 1, base_a1[case_id], total_tokens=77)
               for case_id in case_ids],
        "A2": [_fixture_raw(pilot, case_id, "A2", 1, a2[case_id]) for case_id in case_ids],
    }
    base_annotations: list[dict] = []
    for case_id in case_ids:
        base_annotations.append(_fixture_annotation(pilot, case_id, "P0",
                                                    base_raw["P0"][case_ids.index(case_id)],
                                                    p0[case_id]))
    for index, case_id in enumerate(case_ids):
        base_annotations.append(_fixture_annotation(pilot, case_id, ARM, base_raw["A1"][index],
                                                    base_a1[case_id]))
    for index, case_id in enumerate(case_ids):
        base_annotations.append(_fixture_annotation(pilot, case_id, "A2", base_raw["A2"][index],
                                                    a2[case_id]))

    new_rows = [
        _fixture_raw(pilot, "FX-001", ARM, 2,
                     [_fixture_candidate(pilot, "FX-001", ARM, 2, 0, "a1x",
                                         summary="new-summary", tier="explicit",
                                         trigger="when q")]),
        _fixture_raw(pilot, "FX-002", ARM, 1,
                     [_fixture_candidate(pilot, "FX-002", ARM, 1, 0, "a2x"),
                      _fixture_candidate(pilot, "FX-002", ARM, 1, 1, "a2y-changed")]),
        _fixture_raw(pilot, "FX-003", ARM, 1, [], status="valid_empty",
                     raw_item_count=0, raw_text="[]"),
        _fixture_raw(pilot, "FX-004", ARM, 1,
                     [_fixture_candidate(pilot, "FX-004", ARM, 1, 0, "a4y"),
                      _fixture_candidate(pilot, "FX-004", ARM, 1, 1, "a4x")]),
        _fixture_raw(pilot, "FX-005", ARM, 1, [], status="invalid_json",
                     raw_item_count=0, raw_text="not-json"),
    ]
    new_ledger = [_fixture_ledger("FX-001", ARM, 1, 40, new_style=True)]
    new_ledger.extend(_fixture_ledger(row["case_id"], ARM, row["attempt"], 15,
                                      new_style=True) for row in new_rows)

    write_bytes_atomic(dev_data / "cases-v1.jsonl", jsonl_bytes(cases))
    for arm in ARMS:
        write_bytes_atomic(dev_data / f"raw-{arm}.jsonl", jsonl_bytes(base_raw[arm]))
    write_bytes_atomic(dev_data / "annotations-v1.jsonl", jsonl_bytes(base_annotations))
    ledger = []
    for case_id in case_ids:
        ledger.append(_fixture_ledger(case_id, "P0", 1, 15, new_style=False))
        ledger.append(_fixture_ledger(case_id, ARM, 1, 77, new_style=False))
        ledger.append(_fixture_ledger(case_id, "A2", 1, 15, new_style=False))
    write_bytes_atomic(dev_data / "call-ledger.jsonl", jsonl_bytes(ledger))
    write_bytes_atomic(paths.raw, jsonl_bytes(new_rows))
    write_bytes_atomic(paths.ledger, jsonl_bytes(new_ledger))
    return Fixture(layout=layout, model=model, paths=paths, cases=cases,
                   base_annotations=base_annotations, base_raw=base_raw,
                   new_rows=new_rows, new_ledger=new_ledger)


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _review_rows(pending: list[dict], *, skip: str | None = None,
                 notes: str = "reviewed") -> list[dict]:
    rows = []
    for packet in pending:
        if packet["blind_id"] == skip:
            continue
        template = json.loads(json.dumps(packet["annotation_template"]))
        template["stage1_complete"] = True
        template["stage2_complete"] = True
        template["adjudicated"] = True
        for atom in template["atoms"]:
            atom["notes"] = notes
        rows.append(template)
    return rows


def run_selftest() -> int:
    checks: list[dict] = []

    def record(name: str, passed: bool, **detail: Any) -> None:
        entry = {"name": name, "passed": bool(passed)}
        if detail:
            entry["detail"] = detail
        checks.append(entry)

    def check(name: str, fn: Any) -> None:
        try:
            detail = fn()
            record(name, True, **(detail or {}))
        except Exception as exc:                
            record(name, False, error={"type": type(exc).__name__,
                                       "message": clip_text(exc, 400)})

    def expect_error(name: str, code: str, fn: Any) -> None:
        try:
            fn()
        except PrepError as exc:
            record(name, exc.code == code, expected=code, got=exc.code,
                   message=clip_text(exc.message, 300))
        except Exception as exc:                
            record(name, False, expected=code, got=type(exc).__name__,
                   message=clip_text(exc, 300))
        else:
            record(name, False, expected=code, got="no error")

    with tempfile.TemporaryDirectory(prefix="prepare-atomic-") as raw_root:
        root = Path(raw_root)
        try:
            fixture = build_fixture(root)
        except Exception as exc:                
            record("fixture_build", False,
                   error={"type": type(exc).__name__, "message": clip_text(exc, 400)})
            fixture = None
        if fixture is None:
            payload = {"check": "prepare-atomization-score-selftest", "passed": False,
                       "checks": len(checks), "failed": len(checks), "results": checks}
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            return 2
        layout, model, paths = fixture.layout, fixture.model, fixture.paths
        pilot = collector.load_pilot()

        def body() -> None:
            check("fixture_base_valid",
                  lambda: {"cases": len(load_base(layout).ordered_cases)})

            summary = prepare(layout, model)
            working = _read_rows(paths.root / "annotations-working.jsonl")
            pending = _read_rows(paths.root / "review-pending.jsonl")
            reused_summary = json.loads((paths.root / "reuse-summary.json")
                                        .read_text(encoding="utf-8"))

            check("prepare_counts", lambda: _check_prepare_counts(summary))
            check("prepare_working_rows", lambda: _check_working(working, fixture, pilot))
            check("prepare_packet_shape", lambda: _check_packet(pending, fixture, summary))
            check("reuse_requires_whole_set", lambda: _check_no_partial(working, fixture))
            check("summary_bounded", lambda: _check_summary_file(reused_summary, fixture))

            review = _review_rows(pending)
            bad_dir = paths.root / "review-bad"
            bad_dir.mkdir()

            def write_review(name: str, rows: list[dict]) -> Path:
                path = bad_dir / name
                write_bytes_atomic(path, jsonl_bytes(rows))
                return path

            expect_error("finalize_rejects_placeholder", "review_incomplete",
                         lambda: finalize(layout, model, write_review(
                             "placeholder.jsonl",
                             [packet["annotation_template"] for packet in pending])))
            expect_error("finalize_rejects_missing", "review_missing", lambda: finalize(
                layout, model, write_review("missing.jsonl", _review_rows(pending[1:]))))
            missing_count = _review_rows(pending)
            missing_count[0]["atoms"] = missing_count[0]["atoms"][:1]
            expect_error("finalize_rejects_atom_count", "review_candidate_mismatch",
                         lambda: finalize(layout, model,
                                          write_review("count.jsonl", missing_count)))
            wrong_id = _review_rows(pending)
            wrong_id[0]["atoms"][0]["candidate_id"] = "cand_not_in_raw"
            expect_error("finalize_rejects_candidate_id", "review_candidate_mismatch",
                         lambda: finalize(layout, model,
                                          write_review("candidate.jsonl", wrong_id)))
            ok_path = write_review("ok.jsonl", review)
            done = finalize(layout, model, ok_path)
            check("finalize_success", lambda: _check_score_dir(done, fixture, pilot))
            check("scorer_runs_on_bundle", lambda: _check_scorer_run(paths.root / "score"))
            expect_error("finalize_guard_without_flag", "score_dir_exists",
                         lambda: finalize(layout, model, ok_path))
            check("finalize_same_sha_idempotent",
                  lambda: _check_idempotent(finalize(layout, model, ok_path,
                                                    allow_same_sha=True)))
            changed = _review_rows(pending, notes="reviewed-again")
            expect_error("finalize_same_sha_rejects_changed_inputs",
                         "score_inputs_changed",
                         lambda: finalize(layout, model,
                                          write_review("changed.jsonl", changed),
                                          allow_same_sha=True))
            with open(paths.raw, "ab") as handle:
                handle.write(b"\n")
            expect_error("finalize_rejects_stale_prepare", "prepare_stale",
                         lambda: finalize(layout, model, ok_path))

            second = build_fixture(root / "second")
            expect_error("finalize_requires_prepare", "prepare_missing",
                         lambda: finalize(second.layout, second.model,
                                          second.paths.root / "not-there.jsonl"))

        try:
            body()
        except Exception as exc:                
            record("selftest_crash", False,
                   error={"type": type(exc).__name__, "message": clip_text(exc, 500)})

    failed = [entry for entry in checks if not entry["passed"]]
    payload = {"check": "prepare-atomization-score-selftest",
               "passed": not failed, "checks": len(checks),
               "failed": len(failed), "results": checks}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if not failed else 2


def _check_prepare_counts(summary: dict) -> dict:
    counts = summary["counts"]
    expected = {"cases": 5, "reused": 1, "review": 3, "auto_empty": 1,
                "reused_atoms": 1, "pending_candidates": 4,
                "working_rows": 15, "pending_rows": 3}
    require(counts == expected, "selftest", f"counts={counts} != {expected}")
    require(summary["modes"]["reuse"] == ["FX-001"], "selftest", "reuse 集合不符")
    require(summary["modes"]["auto_empty"] == ["FX-003"], "selftest", "空输出分支不符")
    require(sorted(summary["modes"]["review"]) == ["FX-002", "FX-004", "FX-005"],
            "selftest", "待审集合不符")
    return {"reused": counts["reused"], "review": counts["review"]}


def _check_working(working: list[dict], fixture: Fixture, pilot: Any) -> dict:
    require(len(working) == 15, "selftest", f"working 行数 {len(working)} != 15")
    by_blind = {row["blind_id"]: row for row in working}
    require(len(by_blind) == len(working), "selftest", "working blind_id 重复")
    source_by_blind = {row["blind_id"]: row for row in fixture.base_annotations}
    p0a2_blinds: set[str] = set()
    for arm in ("P0", "A2"):
        for row in fixture.base_raw[arm]:
            p0a2_blinds.add(pilot.blind_id_of(row["case_id"], arm, row))
    for blind in sorted(p0a2_blinds):
        target = by_blind.get(blind)
        require(target is not None, "selftest", f"{blind}: P0/A2 行丢失")
        require(target["atoms"] == source_by_blind[blind]["atoms"], "selftest",
                f"{blind}: P0/A2 行被改动")
        require(target["adjudicated"] is True, "selftest", "P0/A2 adjudicated 被改")
    base_a1_blinds = {pilot.blind_id_of(row["case_id"], ARM, row)
                      for row in fixture.base_raw["A1"]}
    require(not (base_a1_blinds & set(by_blind)), "selftest",
            "DeepSeek A1 旧标注混进了 working")
    new_blind = {row["case_id"]: pilot.blind_id_of(row["case_id"], ARM, row)
                 for row in fixture.new_rows}
    a1_rows = {row["case_id"]: row for row in working if row.get("case_id")}
    require(len(a1_rows) == 5, "selftest", "A1 working 行数不符")
    reuse = a1_rows["FX-001"]
    require(reuse["adjudicated"] is True and reuse["blind_id"] == new_blind["FX-001"],
            "selftest", "复用行未按新 blind_id 记账")
    require(len(reuse["atoms"]) == 1, "selftest", "复用 atom 数不符")
    new_c1 = fixture.new_rows[0]["parsed_candidates"][0]["candidate_id"]
    require(reuse["atoms"][0]["candidate_id"] == new_c1, "selftest",
            "复用 atom 未改为新 candidate_id")
    require(reuse["atoms"][0]["core_entailed_gold_ids"] == ["FX-001-g1"], "selftest",
            "复用 atom 丢了 gold 引用")
    empty = a1_rows["FX-003"]
    require(empty["adjudicated"] is True and empty["atoms"] == [], "selftest",
            "合法空输出未直接完成")
    for case_id in ("FX-002", "FX-004", "FX-005"):
        row = a1_rows[case_id]
        require(row["adjudicated"] is False and row["stage1_complete"] is False
                and row["stage2_complete"] is False, "selftest",
                f"{case_id}: placeholder 标记不全为 false")
        require(len(row["atoms"]) == (2 if case_id != "FX-005" else 0), "selftest",
                f"{case_id}: placeholder atom 数不符")
        require(all(atom["core_entailed_gold_ids"] == [] and "notes" not in atom
                    for atom in row["atoms"]), "selftest",
                f"{case_id}: placeholder 混入了旧判断")
    return {"working_rows": len(working)}


def _check_packet(pending: list[dict], fixture: Fixture, summary: dict) -> dict:
    require(len(pending) == 3, "selftest", f"pending 行数 {len(pending)} != 3")
    require([row["blind_id"] for row in pending] == summary["pending_blind_ids"],
            "selftest", "pending 顺序与 summary 不一致")
    raw_by_case = {row["case_id"]: row for row in fixture.new_rows}
    for packet in pending:
        for key in ("blind_id", "case_id", "split", "conversations",
                    "forbidden_inferences", "one_off_hints", "gold_obligations",
                    "candidates", "annotation_template", "instructions"):
            require(key in packet, "selftest", f"packet 缺 {key}")
        require(packet["schema"] == REVIEW_SCHEMA, "selftest", "packet schema 不符")
        require("model" not in packet and "channel" not in packet, "selftest",
                "packet 泄露模型身份")
        raw = raw_by_case[packet["case_id"]]
        require([c["candidate_id"] for c in packet["candidates"]]
                == [c["candidate_id"] for c in raw["parsed_candidates"]], "selftest",
                "packet 候选顺序与 raw 不一致")
        require(all("kept_status" not in c for c in packet["candidates"]), "selftest",
                "packet 候选泄露 kept_status")
        template = packet["annotation_template"]
        require(template["adjudicated"] is False, "selftest", "模板 adjudicated 不为 false")
        require(len(template["atoms"]) == len(raw["parsed_candidates"]), "selftest",
                "模板 atom 数与候选不一致")
    return {"pending": len(pending)}


def _check_no_partial(working: list[dict], fixture: Fixture) -> dict:
    by_case = {row["case_id"]: row for row in working if row.get("case_id")}
    raw_by_case = {row["case_id"]: row for row in fixture.new_rows}
    for case_id in ("FX-002", "FX-004"):
        row = by_case[case_id]
        require(row["adjudicated"] is False, "selftest", f"{case_id} 应待审")
        require(len(row["atoms"]) == 2, "selftest", f"{case_id} 应全部重审")
        require([atom["candidate_id"] for atom in row["atoms"]]
                == [candidate["candidate_id"]
                    for candidate in raw_by_case[case_id]["parsed_candidates"]],
                "selftest", f"{case_id} placeholder 没有覆盖全部新候选")
        require(all(atom["core_entailed_gold_ids"] == [] for atom in row["atoms"]),
                "selftest", f"{case_id} 出现部分复用残留")
    return {"reviewed": 2}


def _check_summary_file(summary: dict, fixture: Fixture) -> dict:
    require(summary["schema"] == SUMMARY_SCHEMA, "selftest", "summary schema 不符")
    require(summary["inputs"]["new_raw"]["sha256"] == sha_file(fixture.paths.raw),
            "selftest", "summary 未记录新 raw SHA")
    require(summary["outputs"]["review_pending"]["rows"] == 3, "selftest",
            "summary pending rows 不符")
    require(len(summary["cases"]) == 5, "selftest", "summary 逐 case 统计缺失")
    text = json.dumps(summary, ensure_ascii=False)
    require(len(text) < 40000, "selftest", "summary 过大（无界输出）")
    return {"bytes": len(text)}


def _check_score_dir(done: dict, fixture: Fixture, pilot: Any) -> dict:
    require(done["status"] == "assembled", "selftest", f"status={done['status']}")
    score = fixture.paths.root / "score"
    require(score.is_dir(), "selftest", "score 目录不存在")
    require(not list(fixture.paths.root.glob(".score-staging-*")), "selftest",
            "残留 staging 目录")
    copied = score / SCORER_PATH.name
    require(sha_file(copied) == sha_file(SCORER_PATH), "selftest", "评分器不是字节副本")
    data = score / "data"
    names = sorted(path.name for path in data.iterdir())
    require(names == sorted(INPUT_NAMES), "selftest", f"score/data 文件集不符：{names}")
    require((data / "raw-P0.jsonl").read_bytes()
            == (fixture.layout.dev_data / "raw-P0.jsonl").read_bytes(), "selftest",
            "raw-P0 不是字节副本")
    require((data / "raw-A2.jsonl").read_bytes()
            == (fixture.layout.dev_data / "raw-A2.jsonl").read_bytes(), "selftest",
            "raw-A2 不是字节副本")
    require(_read_rows(data / "raw-A1.jsonl") == fixture.new_rows, "selftest",
            "raw-A1 不是新模型 raw")

    annotations = _read_rows(data / "annotations-v1.jsonl")
    require(len(annotations) == 15, "selftest", f"final 行数 {len(annotations)} != 15")
    require(all(row["adjudicated"] is True for row in annotations), "selftest",
            "final 有未完成行")
    by_blind = {row["blind_id"]: row for row in annotations}
    for arm in ARMS:
        rows = _read_rows(data / f"raw-{arm}.jsonl")
        latest: dict[str, dict] = {}
        for row in rows:
            latest[row["case_id"]] = row
        for case_id, row in latest.items():
            blind = pilot.blind_id_of(case_id, arm, row)
            require(blind in by_blind, "selftest", f"{case_id}/{arm} 缺 final 标注")
            require(len(by_blind[blind]["atoms"]) == len(row["parsed_candidates"]),
                    "selftest", f"{case_id}/{arm} atom 数不符")
    reviewed = [row for row in annotations
                if row.get("case_id") in ("FX-002", "FX-004")]
    require(all(atom.get("notes") == "reviewed" for row in reviewed for atom in row["atoms"]),
            "selftest", "final 未采用主审结果")

    ledger = _read_rows(data / "call-ledger.jsonl")
    require(len(ledger) == 16, "selftest", f"ledger 行数 {len(ledger)} != 16")
    require(all("elapsed_s" in row for row in ledger), "selftest", "ledger 缺 elapsed_s")
    require(not any(row.get("usage", {}).get("total_tokens") == 77 for row in ledger),
            "selftest", "旧 A1 ledger 混入")
    retry = [row for row in ledger
             if row.get("attempt") == 1 and row.get("usage", {}).get("total_tokens") == 40]
    require(len(retry) == 1, "selftest", "新 A1 的早先 attempt 成本行被丢掉")

    manifest = json.loads((score / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["schema"] == MANIFEST_SCHEMA, "selftest", "manifest schema 不符")
    require(manifest["model"] == fixture.model, "selftest", "manifest model 不符")
    require(manifest["comparison_scope"]["report_arms"] == ["A1", "P0"], "selftest",
            "manifest 比较范围不符")
    require(manifest["ledger_adapter"]["rows_adapted"] == 6, "selftest",
            f"ledger 适配行数 {manifest['ledger_adapter']['rows_adapted']} != 6")
    require(manifest["counts"]["cases"] == 5 and manifest["counts"]["rows"] == 15,
            "selftest", "manifest counts 不符")
    require(manifest["counts"]["by_arm"] == {"P0": 5, "A1": 5, "A2": 5}, "selftest",
            f"manifest by_arm 不符：{manifest['counts']['by_arm']}")
    for name in INPUT_NAMES:
        require(sha_file(data / name) == manifest["inputs"][name]["sha256"], "selftest",
                f"manifest 输入 SHA 与 {name} 不一致")
    require(sha_file(copied) == manifest["scorer"]["sha256"], "selftest",
            "manifest 评分器 SHA 不一致")
    require(manifest["review"]["rows"] == 3
            and len(manifest["review"]["sha256"]) == 64, "selftest",
            "manifest 未记录主审输入")
    return {"rows": len(annotations), "ledger": len(ledger)}


def _check_scorer_run(score: Path) -> dict:
    import subprocess

    proc = subprocess.run([sys.executable, str(score / SCORER_PATH.name), "score"],
                          cwd=str(score), capture_output=True, text=True, timeout=180)
    require(proc.returncode == 0, "selftest",
            f"评分器 exit={proc.returncode}: {clip_text(proc.stderr, 600)}")
    results = json.loads((score / "data" / "results.json").read_text(encoding="utf-8"))
    counts = results["counts"]
    require(counts["cases"] == 5 and counts["raw_sets"] == 15
            and counts["annotations"] == 15 and counts["matched_output_sets"] == 15,
            "selftest", f"评分器 counts 不符：{counts}")
    return {"cases": counts["cases"], "raw_sets": counts["raw_sets"]}


def _check_idempotent(payload: dict) -> dict:
    require(payload["status"] == "already_current", "selftest",
            f"status={payload['status']}，应为 already_current")
    return {"status": payload["status"]}


                                               


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把新模型 A1 raw 接到已审 v2 评分输入")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="生成 working / pending / summary")
    prepare_parser.add_argument("--model", required=True)
    finalize_parser = sub.add_parser("finalize", help="合并主审结果并装配 score 目录")
    finalize_parser.add_argument("--model", required=True)
    finalize_parser.add_argument("--annotations", required=True)
    finalize_parser.add_argument("--allow-same-sha", action="store_true")
    sub.add_parser("selftest", help="临时目录夹具自测（离线）")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "selftest":
        return run_selftest()

    layout = default_layout()
    try:
        if args.command == "prepare":
            payload = prepare(layout, args.model)
        else:
            payload = finalize(layout, args.model, Path(args.annotations),
                               allow_same_sha=args.allow_same_sha)
    except PrepError as exc:
        print(json.dumps({"ok": False,
                          "error": {"code": exc.code,
                                    "message": clip_text(exc.message, 600)}},
                         ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    except Exception as exc:                                    
        print(json.dumps({"ok": False,
                          "error": {"code": "unexpected",
                                    "type": type(exc).__name__,
                                    "message": clip_text(exc, 600)}},
                         ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
