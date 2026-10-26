                      
"""跨模型原子化采集器：同一 30 dev cases、A1-v2 prompt、生产 parser/证据闸。

原实验（只读）：../原子记忆真实试点与高效架构-2026-10-25/pilot_atomic_memory_20261025.py
复用其 load_cases / build_a1_prompt / head_only_delta / build_raw_record /
build_blind_row / blind_id_of / parse+gate（经 build_raw_record），本文件不复制评分或标签逻辑。

产物（每模型独立，绝不覆盖 DeepSeek 旧数据）：
  data/<model>/raw.jsonl
  data/<model>/call-ledger.jsonl
  data/<model>/blind-stage1.jsonl
  data/<model>/blind-stage2.jsonl

命令（无参数 = selfcheck，完全离线）：
  selfcheck
  preflight [--channel codex|chat|anthropic|ollama --model ...] [--case-id X | --all-dev]
  collect --channel ollama --model qwen3:0.6b (--case-id X | --all-dev) [--dry-run]
  blind --model qwen3:0.6b

Ollama 默认 http://127.0.0.1:11434/api/chat；模型目录把冒号转义为 %3A。
真实调用只发生在显式 `collect`（非 --dry-run）时；selfcheck 用假通道/假 transport，
产物只写系统临时目录。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.parse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True                                 

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import model_channel as mc              

PILOT_DIR = EXP.parent / "原子记忆真实试点与高效架构-2026-10-25"
PILOT_PATH = PILOT_DIR / "pilot_atomic_memory_20261025.py"
DATA_ROOT = EXP / "data"

EXPERIMENT = "atom-cross-model-20261025"
ARM = "A1"
ATOMIC_VERSION = "v2"
SPLIT = "dev"
DEV_CASES = 30
MAX_ATTEMPTS = 2
CONSECUTIVE_FAILURE_STOP = 3
DEFAULT_TOKEN_BUDGET = 150_000
DEFAULT_MAX_CALLS = 60

_PILOT: Any = None


class CollectorError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def require(condition: Any, code: str, message: str) -> None:
    if not condition:
        raise CollectorError(code, message)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_pilot() -> Any:
    global _PILOT
    if _PILOT is None:
        require(PILOT_PATH.exists(), "pilot_missing", f"原实验脚本不存在：{PILOT_PATH}")
        import importlib.util

        spec = importlib.util.spec_from_file_location("pilot_atomic_memory_20261025_shared", PILOT_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _PILOT = module
    return _PILOT


def read_rows(path: Path) -> tuple[list[dict], list[str]]:
    return load_pilot().read_jsonl(path, strict=False)


def dev_cases(*, require_count: bool = True) -> list[dict]:
    cases = [case for case in load_pilot().load_cases() if case["split"] == SPLIT]
    if require_count:
        require(len(cases) == DEV_CASES, "dev_count",
                f"dev case 数 {len(cases)} != {DEV_CASES}；拒绝在漂移的输入上采集。")
    return sorted(cases, key=lambda case: case["case_id"])


def prompt_for(case: dict) -> str:
    """A1-v2：只允许 P0 生产 prompt + 原子段，其它任何差异都中止。"""
    pilot = load_pilot()
    prompt = pilot.build_a1_prompt(case["conversations"], ATOMIC_VERSION)
    p0 = pilot.dream.build_dream_prompt(case["conversations"])
    require(pilot.head_only_delta(prompt, p0, ATOMIC_VERSION), "prompt_contract",
            f"{case['case_id']}: A1-v2 prompt 不是 P0 + 原子段")
    require(pilot.ATOMIC_TEXT_BY_VERSION[ATOMIC_VERSION] in prompt, "prompt_contract",
            f"{case['case_id']}: prompt 缺少 v2 原子约束文本")
    return prompt


def clip_for(case: dict) -> list[dict]:
    clipped = load_pilot().dream._clip_conversations(case["conversations"])
    require(len(clipped) == len(case["conversations"]), "clip_changed",
            f"{case['case_id']}: 生产入参收口改变了对话段数，conv 索引会错位")
    return clipped


                                                   


                                                       
_WINDOWS_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL",
                               *(f"COM{i}" for i in range(1, 10)),
                               *(f"LPT{i}" for i in range(1, 10))})


def safe_model_id(model: str) -> str:
    """可逆、防冲突的目录名：冒号保留为 %3A，大小写差异加哈希，超长名截断加哈希。

    `quote(..., safe="")` 让 `qwen3:0.6b` → `qwen3%3A0.6b`，与 `qwen3_0.6b` 不冲突；
    前缀 `@` 只用于含大写或超长的 ID（普通 ID 的 `@` 会被编码成 `%40`）。
    """
    text = str(model or "").strip()
    require(bool(text), "invalid_model", f"模型 ID 为空：{model!r}")
    encoded = urllib.parse.quote(text, safe="")
    if encoded.endswith("."):
        encoded = encoded[:-1] + "%2E"
    if len(encoded) > 48 or any(char.isupper() for char in text):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        encoded = "@" + encoded[:32] + "-" + digest
    if encoded.split(".")[0].upper() in _WINDOWS_RESERVED:
        encoded = "@r-" + encoded
    require(encoded not in (".", ".."), "invalid_model",
            f"模型 ID 无法安全映射为目录名：{model!r}")
    return encoded


@dataclass(frozen=True)
class ModelPaths:
    model: str
    root: Path
    raw: Path
    ledger: Path
    blind1: Path
    blind2: Path


def paths_for(model: str, root: Path | str = DATA_ROOT) -> ModelPaths:
    base = Path(root).resolve()
    target = (base / safe_model_id(model)).resolve()
    require(target.is_relative_to(base), "invalid_model", "模型目录越界")
    return ModelPaths(model=model, root=target, raw=target / "raw.jsonl",
                      ledger=target / "call-ledger.jsonl",
                      blind1=target / "blind-stage1.jsonl",
                      blind2=target / "blind-stage2.jsonl")


def check_identity(paths: ModelPaths, channel: Any, raw_rows: list[dict],
                   ledger_rows: list[dict]) -> None:
    for row in list(raw_rows) + list(ledger_rows):
        if row.get("model") != channel.model:
            raise CollectorError("collector_identity_conflict",
                                 f"{paths.root.name}: 已有 model={row.get('model')!r}，"
                                 f"拒绝混入 {channel.model!r}")
        if row.get("channel") != channel.name:
            raise CollectorError("collector_identity_conflict",
                                 f"{paths.root.name}: 已有 channel={row.get('channel')!r}，"
                                 f"拒绝混入 {channel.name!r}")
        if row.get("atomic_version") != ATOMIC_VERSION:
            raise CollectorError("collector_identity_conflict",
                                 f"{paths.root.name}: 出现 atomic_version="
                                 f"{row.get('atomic_version')!r}，本采集器只写 {ATOMIC_VERSION}")


def _usage_key(row: dict) -> tuple:
    return (row.get("case_id"), row.get("arm"), row.get("attempt"), row.get("prompt_sha256"))


def _row_attempted(row: dict) -> bool:
    if "attempted" in row:
        return row.get("attempted") is True
    return (row.get("call") or {}).get("attempted") is True


def usage_totals(raw_rows: list[dict], ledger_rows: list[dict]) -> tuple[int, int]:
    """按 (case, arm, attempt, prompt_sha) 合并 raw 与 ledger；ledger 覆盖同键，
    这样任一文件半写（崩溃在两次 append 之间）都不会重复或漏计预算。"""
    merged: dict[tuple, dict] = {}
    for row in raw_rows:
        merged[_usage_key(row)] = row
    for row in ledger_rows:
        merged[_usage_key(row)] = row
    calls = sum(1 for row in merged.values() if _row_attempted(row))
    tokens = sum(int((row.get("usage") or {}).get("total_tokens") or 0)
                 for row in merged.values())
    return calls, tokens


class Budget:
    def __init__(self, calls: int, tokens: int, max_calls: int, token_budget: int) -> None:
        self.calls = int(calls)
        self.tokens = int(tokens)
        self.max_calls = int(max_calls)
        self.token_budget = int(token_budget)

    def cap_reason(self) -> str | None:
        if self.calls >= self.max_calls:
            return "call_cap"
        if self.tokens >= self.token_budget:
            return "token_budget"
        return None

    def account(self, usage: dict, *, attempted: bool) -> None:
        if attempted:
            self.calls += 1
        self.tokens += int((usage or {}).get("total_tokens") or 0)


def case_state(case: dict, prompt: str, raw_rows: list[dict],
               ledger_rows: list[dict]) -> dict:
    case_id, psha = case["case_id"], sha_text(prompt)
    success: dict | None = None
    stale = False
    for row in raw_rows:
        if row.get("case_id") != case_id or row.get("arm") != ARM:
            continue
        if row.get("raw_text") is None:
            continue
        if not (row.get("call") or {}).get("succeeded"):
            continue
        if row.get("prompt_sha256") == psha:
            if success is None or int(row.get("attempt") or 0) >= int(success.get("attempt") or 0):
                success = row
        else:
            stale = True
    raw_attempts = sum(1 for row in raw_rows
                       if row.get("case_id") == case_id and row.get("arm") == ARM
                       and row.get("prompt_sha256") == psha
                       and (row.get("call") or {}).get("attempted") is True)
    ledger_attempts = sum(1 for row in ledger_rows
                          if row.get("case_id") == case_id and row.get("arm") == ARM
                          and row.get("prompt_sha256") == psha
                          and row.get("attempted") is True)
    attempts = max(raw_attempts, ledger_attempts)
    if success is not None:
        state = "done"
    elif attempts >= MAX_ATTEMPTS:
        state = "exhausted"
    else:
        state = "pending"
    return {"case_id": case_id, "prompt_sha256": psha, "prompt_chars": len(prompt),
            "state": state, "attempts_used": attempts,
            "stale_success": bool(stale and success is None),
            "status": success.get("status") if success else None}


def plan_cases(cases: list[dict], prompts: dict[str, str], raw_rows: list[dict],
               ledger_rows: list[dict]) -> list[dict]:
    return [case_state(case, prompts[case["case_id"]], raw_rows, ledger_rows)
            for case in sorted(cases, key=lambda item: item["case_id"])]


def aggregate_plan(entries: list[dict]) -> dict:
    counts = Counter(entry["state"] for entry in entries)
    remaining = sum(max(0, MAX_ATTEMPTS - entry["attempts_used"])
                    for entry in entries if entry["state"] == "pending")
    return {"done": counts.get("done", 0), "pending": counts.get("pending", 0),
            "exhausted": counts.get("exhausted", 0),
            "stale_success": sum(1 for entry in entries if entry["stale_success"]),
            "would_call_max": remaining}


def select_cases(cases: list[dict], case_ids: list[str] | None, all_dev: bool,
                 *, default_all: bool = False) -> list[dict]:
    ids = [item.strip() for value in (case_ids or [])
           for item in str(value).split(",") if item.strip()]
    require(not (all_dev and ids), "case_selection",
            "--all-dev 与 --case-id 不能同时给出")
    if all_dev:
        return sorted(cases, key=lambda case: case["case_id"])
    if not ids:
        if default_all:
            return sorted(cases, key=lambda case: case["case_id"])
        raise CollectorError("case_selection_required",
                             "collect 必须显式 --case-id <id>（可重复/逗号分隔）或 --all-dev")
    known = {case["case_id"]: case for case in cases}
    unknown = sorted(set(ids) - set(known))
    require(not unknown, "unknown_case", f"未知或非 dev 的 case_id：{unknown}")
    return [known[case_id] for case_id in sorted(set(ids))]


                                              


def decorate_record(record: dict, *, channel: Any, outcome: dict, prompt_sha: str) -> dict:
    contract = record.get("contract") or {}
    require(contract.get("head_only_delta") is True, "prompt_contract",
            f"{record.get('case_id')}: raw contract.head_only_delta 不为真")
    require(contract.get("atomic_text_present") is True, "prompt_contract",
            f"{record.get('case_id')}: raw contract.atomic_text_present 不为真")
    record["experiment"] = EXPERIMENT
    record["channel"] = channel.name
    record["provider"] = outcome["provider"]
    record["model"] = channel.model
    record["usage_source"] = (outcome.get("usage") or {}).get("source")
    record["channel_diagnostics"] = outcome.get("diagnostics") or {}
    record["prompt_sha256"] = prompt_sha
    return record


def pilot_outcome(outcome: dict) -> dict:
    return {"text": outcome["text"], "status": outcome["status"],
            "attempted": outcome["attempted"], "succeeded": outcome["succeeded"],
            "finish_reason": outcome["finish_reason"], "usage": outcome["usage"],
            "elapsed_s": outcome["latency_s"], "error_kind": outcome["error_kind"]}


def rebuild_blind(cases: list[dict], paths: ModelPaths) -> dict:
    """由每个 case 的最新成功 raw 生成原版盲审行；不写 model/channel，保持盲态。"""
    pilot = load_pilot()
    by_id = {case["case_id"]: case for case in cases}
    prompts = {case_id: prompt_for(case) for case_id, case in by_id.items()}
    raw_rows, _ = read_rows(paths.raw)
    latest: dict[str, dict] = {}
    for row in raw_rows:
        case_id = row.get("case_id")
        require(case_id in by_id, "unknown_case", f"{paths.raw.name} 引用了未知 case：{case_id}")
        if row.get("raw_text") is None or not (row.get("call") or {}).get("succeeded"):
            continue
        if row.get("prompt_sha256") != sha_text(prompts[case_id]):
            continue
        previous = latest.get(case_id)
        if previous is None or int(row.get("attempt") or 0) >= int(previous.get("attempt") or 0):
            latest[case_id] = row
    if not latest:
        return {"rows": 0, "written": False, "stage1": str(paths.blind1),
                "stage2": str(paths.blind2), "blind_ids": []}
    rows1: list[dict] = []
    rows2: list[dict] = []
    seen: set[str] = set()
    for case_id in sorted(latest):
        record, case = latest[case_id], by_id[case_id]
        row1 = pilot.build_blind_row(case, ARM, record, 1)
        require(row1["blind_id"] not in seen, "blind_id_conflict",
                f"blind_id 冲突：{row1['blind_id']}")
        seen.add(row1["blind_id"])
        _blind_leak_check(row1)
        rows1.append(row1)
        row2 = pilot.build_blind_row(case, ARM, record, 2)
        _blind_leak_check(row2)
        rows2.append(row2)
    rows1.sort(key=lambda row: row["blind_id"])
    rows2.sort(key=lambda row: row["blind_id"])
    pilot.write_jsonl(paths.blind1, rows1)
    pilot.write_jsonl(paths.blind2, rows2)
    return {"rows": len(rows1), "written": True, "stage1": str(paths.blind1),
            "stage2": str(paths.blind2),
            "blind_ids": [row["blind_id"] for row in rows1]}


def _blind_leak_check(row: dict) -> None:
    pilot = load_pilot()
    leaked = sorted(pilot.BLIND_FORBIDDEN_KEYS & set(row))
    require(not leaked, "blind_leak", f"blind 行泄露实验字段：{leaked}")
    for candidate in row.get("candidates") or []:
        leaked = sorted(pilot.BLIND_FORBIDDEN_KEYS & set(candidate))
        require(not leaked, "blind_leak", f"blind 候选泄露实验字段：{leaked}")


def run_collect(*, cases: list[dict], selected: list[dict], channel: Any,
                paths: ModelPaths, token_budget: int, max_calls: int,
                dry_run: bool = False) -> dict:
    pilot = load_pilot()
    require(token_budget > 0 and max_calls > 0, "budget_invalid", "预算与调用上限必须为正")
    known = {case["case_id"] for case in cases}
    require({case["case_id"] for case in selected} <= known, "unknown_case",
            "selected 含 cases 之外的 case")

    raw_rows, raw_warnings = read_rows(paths.raw)
    ledger_rows, ledger_warnings = read_rows(paths.ledger)
    check_identity(paths, channel, raw_rows, ledger_rows)

    prompts = {case["case_id"]: prompt_for(case) for case in cases}
    clipped = {case["case_id"]: clip_for(case) for case in selected}
    plan = plan_cases(selected, prompts, raw_rows, ledger_rows)
    calls_before, tokens_before = usage_totals(raw_rows, ledger_rows)
    budget = Budget(calls_before, tokens_before, max_calls, token_budget)
    available, availability = channel.availability()

    summary: dict[str, Any] = {
        "ok": True, "dry_run": bool(dry_run), "experiment": EXPERIMENT,
        "channel": channel.name, "provider": channel.provider, "model": channel.model,
        "model_dir": str(paths.root), "raw": str(paths.raw), "ledger": str(paths.ledger),
        "selected_cases": [case["case_id"] for case in selected],
        "selected_count": len(selected),
        "skipped_done": 0, "skipped_exhausted": 0, "stale_success": 0,
        "new_records": 0, "provider_failures": 0, "production_mismatch": 0,
        "statuses": {}, "halted": None,
        "available": available, "availability": availability,
        "token_budget": token_budget, "max_calls": max_calls,
        "calls_before": calls_before, "tokens_before": tokens_before,
        "calls_after": calls_before, "tokens_after": tokens_before,
        "raw_warnings": raw_warnings, "ledger_warnings": ledger_warnings,
        "plan": plan, "blind": None,
    }
    if dry_run:
        summary["ok"] = bool(available)
        return summary
    require(available, "channel_unavailable",
            f"通道 {channel.name} 不可用（{availability}）；未发送任何调用")

    statuses: Counter[str] = Counter()
    consecutive_failures = 0
    halted: str | None = None

    for case in selected:
        if halted:
            break
        case_id = case["case_id"]
        entry = next(item for item in plan if item["case_id"] == case_id)
        if entry["state"] == "done":
            summary["skipped_done"] += 1
            continue
        if entry["state"] == "exhausted":
            summary["skipped_exhausted"] += 1
            continue
        if entry["stale_success"]:
            summary["stale_success"] += 1
        attempt = entry["attempts_used"]
        while attempt < MAX_ATTEMPTS:
            cap = budget.cap_reason()
            if cap:
                halted = cap
                break
            attempt += 1
            outcome = channel.complete(prompts[case_id])
            record = pilot.build_raw_record(case=case, arm=ARM, attempt=attempt,
                                            prompt=prompts[case_id],
                                            outcome=pilot_outcome(outcome),
                                            clipped=clipped[case_id],
                                            version=ATOMIC_VERSION)
            decorate_record(record, channel=channel, outcome=outcome,
                            prompt_sha=entry["prompt_sha256"])
            pilot.append_jsonl(paths.raw, record)
            raw_rows.append(record)
            budget.account(outcome["usage"], attempted=outcome["attempted"])
            row = mc.ledger_row(case_id=case_id, split=case["split"], arm=ARM,
                                atomic_version=ATOMIC_VERSION, attempt=attempt,
                                channel=channel, outcome=outcome,
                                prompt_sha256=entry["prompt_sha256"],
                                prompt_chars=entry["prompt_chars"],
                                cumulative_calls=budget.calls,
                                cumulative_tokens=budget.tokens)
            pilot.append_jsonl(paths.ledger, row)
            ledger_rows.append(row)
            summary["new_records"] += 1
            statuses[record["status"]] += 1
            if record.get("contract", {}).get("production_matches_direct") is False:
                summary["production_mismatch"] += 1
            if not outcome["succeeded"]:
                summary["provider_failures"] += 1
                consecutive_failures += 1
                if consecutive_failures >= CONSECUTIVE_FAILURE_STOP:
                    halted = "consecutive_provider_failures"
                    break
            else:
                consecutive_failures = 0
                break
        if halted:
            break

    summary["halted"] = halted
    summary["statuses"] = dict(sorted(statuses.items()))
    summary["calls_after"] = budget.calls
    summary["tokens_after"] = budget.tokens
    summary["blind"] = rebuild_blind(cases, paths)
    summary["ok"] = halted is None
    return summary


def command_collect(args: argparse.Namespace) -> dict:
    cases = dev_cases()
    selected = select_cases(cases, args.case_id, args.all_dev,
                            default_all=bool(args.dry_run))
    channel = mc.make_channel(args.channel, args.model, endpoint=args.base_url,
                              api_key_env=args.api_key_env, effort=args.effort,
                              timeout=args.timeout, binary=args.codex_bin)
    return run_collect(cases=cases, selected=selected, channel=channel,
                       paths=paths_for(args.model), token_budget=args.token_budget,
                       max_calls=args.max_calls, dry_run=args.dry_run)


                                              


def command_preflight(args: argparse.Namespace) -> dict:
    pilot = load_pilot()
    all_cases = pilot.load_cases()
    dev = [case for case in all_cases if case["split"] == SPLIT]
    problems: list[str] = []
    if len(dev) != DEV_CASES:
        problems.append(f"dev_count:{len(dev)}!={DEV_CASES}")

    prompts: dict[str, str] = {}
    per_case: list[dict] = []
    for case in sorted(dev, key=lambda item: item["case_id"]):
        try:
            prompt = prompt_for(case)
        except CollectorError as exc:
            problems.append(f"{case['case_id']}:{exc.code}")
            continue
        prompts[case["case_id"]] = prompt
        per_case.append({"case_id": case["case_id"], "split": case["split"],
                         "prompt_chars": len(prompt), "prompt_sha256": sha_text(prompt)})
    aggregate = sha_text("\n".join(f"{row['case_id']}:{row['prompt_sha256']}"
                                   for row in per_case))

    selected = select_cases(dev, args.case_id, args.all_dev,
                            default_all=not args.case_id and not args.all_dev)
    selected = [case for case in selected if case["case_id"] in prompts]
    status = mc.channel_status(endpoint=args.base_url, api_key_env=args.api_key_env,
                               binary=args.codex_bin)
    requested = None
    if args.channel or args.model:
        if not (args.channel and args.model):
            problems.append("channel_and_model_required_together")
        else:
            channel = mc.make_channel(args.channel, args.model, endpoint=args.base_url,
                                      api_key_env=args.api_key_env, effort=args.effort,
                                      timeout=args.timeout, binary=args.codex_bin)
            available, reason = channel.availability()
            requested = {"channel": channel.name, "model": channel.model,
                         "available": available, "reason": reason}
            if not available:
                problems.append(f"channel_unavailable:{channel.name}:{reason}")

    plan: dict[str, Any] = {"selected_cases": len(selected),
                            "token_budget": args.token_budget,
                            "max_calls": args.max_calls,
                            "attempts_per_case_max": MAX_ATTEMPTS,
                            "consecutive_failure_stop": CONSECUTIVE_FAILURE_STOP}
    if args.model:
        paths = paths_for(args.model)
        raw_rows, _ = read_rows(paths.raw)
        ledger_rows, _ = read_rows(paths.ledger)
        if args.channel:
            try:
                check_identity(paths, mc.make_channel(args.channel, args.model),
                               raw_rows, ledger_rows)
            except CollectorError as exc:
                problems.append(f"{exc.code}:{paths.root.name}")
        calls_used, tokens_used = usage_totals(raw_rows, ledger_rows)
        entries = plan_cases(selected, prompts, raw_rows, ledger_rows)
        plan.update(aggregate_plan(entries))
        plan.update({"calls_used": calls_used, "tokens_used": tokens_used,
                     "identity_checked": bool(args.channel),
                     "per_case": entries,
                     "paths": {"root": str(paths.root), "raw": str(paths.raw),
                               "ledger": str(paths.ledger),
                               "blind1": str(paths.blind1),
                               "blind2": str(paths.blind2)}})
    else:
        plan["note"] = "model_not_given"
        plan["per_case"] = [{"case_id": row["case_id"], "prompt_sha256": row["prompt_sha256"]}
                            for row in per_case if row["case_id"] in
                            {case["case_id"] for case in selected}]

    return {"ok": not problems, "dry_run": True, "experiment": EXPERIMENT,
            "cases": {"total": len(all_cases), "dev": len(dev),
                      "confirmation": sum(1 for case in all_cases
                                          if case["split"] == "confirmation")},
            "atomic_version": ATOMIC_VERSION, "arm": ARM,
            "prompt_sha_aggregate": aggregate, "per_case": per_case,
            "channels": status["available"], "channel_detail": status["detail"],
            "requested_channel": requested, "plan": plan, "problems": problems}


                                                


class ScriptedChannel:
    name = "fake"
    provider = "offline-fake"

    def __init__(self, model: str, script: list[str | None]) -> None:
        self.model = model
        self.script = list(script)
        self.calls = 0

    def availability(self) -> tuple[bool, str]:
        return True, "ready"

    def complete(self, prompt: str) -> dict:
        self.calls += 1
        if not self.script:
            return mc.make_outcome(provider=self.provider, model=self.model,
                                   attempted=True, error_kind="script_exhausted")
        item = self.script.pop(0)
        if item is None:
            return mc.make_outcome(provider=self.provider, model=self.model,
                                   attempted=True, error_kind="timeout")
        return mc.make_outcome(provider=self.provider, model=self.model, text=item,
                               succeeded=True, finish_reason="stop",
                               usage=mc.estimate_usage(prompt, item), latency_s=0.001)


def _selfcheck_response(case: dict) -> str | None:
    """用真实对话里的两段原文作 evidence，保证走生产证据闸时确实 kept。"""
    pilot = load_pilot()
    queries = [str(conv.get("query") or "").strip() for conv in case["conversations"]]
    queries = [query for query in queries if query]
    if len(queries) < 2:
        return None
    item = {"text": "检索前先确认平台与化学版本一致", "summary": "离线自检条目",
            "evidence": [queries[0], queries[1]]}
    text = json.dumps([item], ensure_ascii=False)
    clipped = pilot.dream._clip_conversations(case["conversations"])
    kept, _ = pilot.dream._apply_gates(pilot.dream.parse_dream_output(text), clipped)
    return text if len(kept) == 1 else None


def _pick_selfcheck_cases(cases: list[dict]) -> tuple[dict, str, dict]:
    used: set[str] = set()
    for case in cases:
        response = _selfcheck_response(case)
        if response is not None:
            used.add(case["case_id"])
            first = (case, response)
            break
    else:
        raise AssertionError("找不到能过生产证据闸的 dev case（selfcheck 夹具失效）")
    second = next(case for case in cases if case["case_id"] not in used)
    return first[0], first[1], second


def _scenario_happy(root: Path, cases: list[dict], first: dict, response: str,
                    second: dict) -> dict:
    paths = paths_for("fake-model", root=root)
    channel = ScriptedChannel("fake-model", [response, "[]"])
    summary = run_collect(cases=cases, selected=[first, second], channel=channel,
                          paths=paths, token_budget=DEFAULT_TOKEN_BUDGET,
                          max_calls=DEFAULT_MAX_CALLS)
    assert summary["halted"] is None and summary["new_records"] == 2, summary
    assert summary["statuses"] == {"ok": 1, "valid_empty": 1}, summary["statuses"]
    raw, _ = read_rows(paths.raw)
    by_case = {row["case_id"]: row for row in raw}
    rec_a, rec_b = by_case[first["case_id"]], by_case[second["case_id"]]
    assert rec_a["status"] == "ok" and rec_a["kept"]["count"] >= 1
    assert rec_a["parsed_candidates"][0]["kept_status"] == "kept"
    assert rec_a["contract"]["head_only_delta"] is True
    assert rec_a["contract"]["production_matches_direct"] is True
    assert rec_a["channel"] == "fake" and rec_a["model"] == "fake-model"
    assert rec_b["status"] == "valid_empty" and rec_b["raw_text"] == "[]"
    assert rec_b["kept"]["count"] == 0
    assert rec_b["parsed_candidates"] == []

    pilot = load_pilot()
    rows1, _ = read_rows(paths.blind1)
    rows2, _ = read_rows(paths.blind2)
    assert len(rows1) == 2 and len(rows2) == 2
    expected_ids = {pilot.blind_id_of(first["case_id"], ARM, rec_a),
                    pilot.blind_id_of(second["case_id"], ARM, rec_b)}
    assert {row["blind_id"] for row in rows1} == expected_ids
    assert {row["blind_id"] for row in rows2} == expected_ids
    assert {pilot.canon(row["gold_obligations"]) for row in rows2} == {
        pilot.canon(first["gold_obligations"]), pilot.canon(second["gold_obligations"])}

    channel2 = ScriptedChannel("fake-model", [])
    second_run = run_collect(cases=cases, selected=[first, second], channel=channel2,
                             paths=paths, token_budget=DEFAULT_TOKEN_BUDGET,
                             max_calls=DEFAULT_MAX_CALLS)
    assert second_run["new_records"] == 0 and second_run["skipped_done"] == 2, second_run
    assert channel2.calls == 0
    return {"new_records": summary["new_records"], "statuses": summary["statuses"],
            "blind_rows": len(rows1), "rerun_skipped": second_run["skipped_done"],
            "rerun_calls": channel2.calls}


def _scenario_retry(root: Path, cases: list[dict], case: dict, response: str) -> dict:
    paths = paths_for("fake-model", root=root)
    channel = ScriptedChannel("fake-model", [None, response])
    summary = run_collect(cases=cases, selected=[case], channel=channel, paths=paths,
                          token_budget=DEFAULT_TOKEN_BUDGET, max_calls=DEFAULT_MAX_CALLS)
    assert summary["halted"] is None and summary["new_records"] == 2, summary
    assert summary["statuses"] == {"ok": 1, "timeout": 1}, summary["statuses"]
    assert summary["provider_failures"] == 1
    raw, _ = read_rows(paths.raw)
    assert [row["attempt"] for row in raw] == [1, 2]
    assert [row["status"] for row in raw] == ["timeout", "ok"]
    ledger, _ = read_rows(paths.ledger)
    assert len(ledger) == 2 and all(row["attempted"] is True for row in ledger)
    return {"new_records": summary["new_records"], "statuses": summary["statuses"],
            "attempts": [row["attempt"] for row in raw], "ledger_rows": len(ledger)}


def _scenario_halt(root: Path, cases: list[dict]) -> dict:
    paths = paths_for("fake-model", root=root)
    selected = sorted(cases, key=lambda case: case["case_id"])[:3]
    channel = ScriptedChannel("fake-model", [None] * 6)
    summary = run_collect(cases=cases, selected=selected, channel=channel, paths=paths,
                          token_budget=DEFAULT_TOKEN_BUDGET, max_calls=DEFAULT_MAX_CALLS)
    assert summary["halted"] == "consecutive_provider_failures", summary
    assert summary["new_records"] == 3 and channel.calls == 3, (summary, channel.calls)
    assert summary["ok"] is False
    return {"halted": summary["halted"], "new_records": summary["new_records"],
            "calls": channel.calls}


def _scenario_stale_sha(root: Path, cases: list[dict], case: dict, response: str) -> dict:
    paths = paths_for("fake-model", root=root)
    raw, _ = read_rows(paths.raw)
    for row in raw:
        if row.get("case_id") == case["case_id"]:
            row["prompt_sha256"] = "0" * 64
    load_pilot().write_jsonl(paths.raw, raw)
    channel = ScriptedChannel("fake-model", [response])
    summary = run_collect(cases=cases, selected=[case], channel=channel, paths=paths,
                          token_budget=DEFAULT_TOKEN_BUDGET, max_calls=DEFAULT_MAX_CALLS)
    assert summary["new_records"] == 1 and summary["stale_success"] == 1, summary
    assert channel.calls == 1
    return {"new_records": summary["new_records"],
            "stale_success": summary["stale_success"], "calls": channel.calls}


def _scenario_budget(root: Path, cases: list[dict], first: dict, response: str,
                     second: dict) -> dict:
    """预算按 ledger 累计：先花掉一次，再用 budget=1 采集另一个 pending case。"""
    paths = paths_for("fake-model", root=root)
    seed = ScriptedChannel("fake-model", [response])
    seeded = run_collect(cases=cases, selected=[first], channel=seed, paths=paths,
                         token_budget=DEFAULT_TOKEN_BUDGET, max_calls=DEFAULT_MAX_CALLS)
    assert seeded["new_records"] == 1 and seeded["tokens_after"] > 1, seeded
    channel = ScriptedChannel("fake-model", [response])
    summary = run_collect(cases=cases, selected=[second], channel=channel, paths=paths,
                          token_budget=1, max_calls=DEFAULT_MAX_CALLS)
    assert summary["halted"] == "token_budget" and channel.calls == 0, (summary, channel.calls)
    assert summary["new_records"] == 0 and summary["ok"] is False
    assert summary["tokens_before"] > 1
    return {"halted": summary["halted"], "calls": channel.calls,
            "new_records": summary["new_records"],
            "tokens_before": summary["tokens_before"]}


def _scenario_resume(root: Path, cases: list[dict], case: dict, response: str) -> dict:
    """跨两次运行的断点：第一次只允许 1 次调用后因 call_cap 停下，第二次从 attempt 2 续跑。"""
    paths = paths_for("fake-model", root=root)
    first = ScriptedChannel("fake-model", [None])
    run1 = run_collect(cases=cases, selected=[case], channel=first, paths=paths,
                       token_budget=DEFAULT_TOKEN_BUDGET, max_calls=1)
    assert run1["new_records"] == 1 and run1["halted"] == "call_cap", run1
    second = ScriptedChannel("fake-model", [response])
    run2 = run_collect(cases=cases, selected=[case], channel=second, paths=paths,
                       token_budget=DEFAULT_TOKEN_BUDGET, max_calls=DEFAULT_MAX_CALLS)
    assert run2["new_records"] == 1 and run2["statuses"] == {"ok": 1}, run2
    raw, _ = read_rows(paths.raw)
    assert [row["attempt"] for row in raw] == [1, 2], raw
    third = ScriptedChannel("fake-model", [])
    run3 = run_collect(cases=cases, selected=[case], channel=third, paths=paths,
                       token_budget=DEFAULT_TOKEN_BUDGET, max_calls=DEFAULT_MAX_CALLS)
    assert run3["skipped_done"] == 1 and third.calls == 0, (run3, third.calls)
    return {"run1_statuses": run1["statuses"], "run1_halted": run1["halted"],
            "run2_statuses": run2["statuses"],
            "attempts": [row["attempt"] for row in raw],
            "run3_skipped": run3["skipped_done"]}


def _scenario_exhausted(root: Path, cases: list[dict], case: dict, response: str) -> dict:
    """两次失败后该 case 记为 exhausted；后续运行不再消耗调用。"""
    paths = paths_for("fake-model", root=root)
    first = ScriptedChannel("fake-model", [None, None])
    run1 = run_collect(cases=cases, selected=[case], channel=first, paths=paths,
                       token_budget=DEFAULT_TOKEN_BUDGET, max_calls=DEFAULT_MAX_CALLS)
    assert run1["new_records"] == 2 and run1["halted"] is None, run1
    second = ScriptedChannel("fake-model", [response])
    run2 = run_collect(cases=cases, selected=[case], channel=second, paths=paths,
                       token_budget=DEFAULT_TOKEN_BUDGET, max_calls=DEFAULT_MAX_CALLS)
    assert run2["new_records"] == 0 and run2["skipped_exhausted"] == 1, run2
    assert second.calls == 0
    return {"run1_statuses": run1["statuses"],
            "run2_skipped_exhausted": run2["skipped_exhausted"], "calls": second.calls}


def _scenario_ollama(root: Path, cases: list[dict], first: dict, response: str,
                     second: dict) -> dict:
    """真实 OllamaChannel + 假 transport 走完整采集：冒号目录、usage、thinking 分离。"""
    paths = paths_for("qwen3:0.6b", root=root)
    assert paths.root.name == "qwen3%3A0.6b", paths.root
    seen: list[dict] = []
    thinking = "不应拼进正文"

    def transport(url, headers, payload, timeout):
        seen.append({"url": url, "headers": headers, "payload": payload})
        text = response if len(seen) == 1 else "[]"
        return 200, json.dumps({
            "message": {"role": "assistant", "content": text, "thinking": thinking},
            "done": True, "done_reason": "stop",
            "prompt_eval_count": 10, "eval_count": 4,
        }, ensure_ascii=False)

    channel = mc.OllamaChannel("qwen3:0.6b", transport=transport)
    summary = run_collect(cases=cases, selected=[first, second], channel=channel,
                          paths=paths, token_budget=DEFAULT_TOKEN_BUDGET,
                          max_calls=DEFAULT_MAX_CALLS)
    assert summary["halted"] is None and summary["new_records"] == 2, summary
    assert summary["tokens_after"] == 28, summary
    raw, _ = read_rows(paths.raw)
    by_case = {row["case_id"]: row for row in raw}
    rec_a, rec_b = by_case[first["case_id"]], by_case[second["case_id"]]
    assert rec_a["channel"] == "ollama" and rec_a["model"] == "qwen3:0.6b"
    assert rec_a["status"] == "ok" and rec_a["raw_text"] == response
    assert thinking not in (rec_a["raw_text"] or "")
    assert rec_a["channel_diagnostics"]["thinking_chars"] == len(thinking)
    assert rec_a["channel_diagnostics"]["thinking_tail"] == thinking
    assert rec_b["status"] == "valid_empty"
    assert seen[0]["payload"]["messages"][0]["content"] == prompt_for(first)
    assert all(item["payload"]["model"] == "qwen3:0.6b" for item in seen)
    assert all(item["payload"]["stream"] is False and item["payload"]["think"] is False
               for item in seen)
    assert all(item["payload"]["options"] == {"temperature": 0, "num_predict": 2048}
               for item in seen)
    assert all(item["url"] == mc.DEFAULT_OLLAMA_ENDPOINT for item in seen)
    ledger, _ = read_rows(paths.ledger)
    assert len(ledger) == 2 and all(row["channel"] == "ollama" for row in ledger)
    assert all(row["usage"]["source"] == "provider" for row in ledger)
    return {"model_dir": paths.root.name, "tokens_after": summary["tokens_after"],
            "statuses": summary["statuses"], "calls": len(seen)}


def _scenario_model_dir(root: Path) -> dict:
    """冒号/斜杠/大小写/超长/Windows 保留名的目录映射必须互不冲突。"""
    colon = safe_model_id("qwen3:0.6b")
    assert colon == "qwen3%3A0.6b", colon
    assert colon != safe_model_id("qwen3_0.6b")
    assert safe_model_id("a/b") != safe_model_id("a_b")
    design_model = safe_model_id("batiai/qwen3.8-27b:iq3")
    assert design_model == "batiai%2Fqwen3.8-27b%3Aiq3", design_model
    assert design_model != safe_model_id("batiai_qwen3.8-27b_iq3")
    upper = safe_model_id("Qwen3:0.6b")
    assert upper != colon and len(upper) <= 80, upper
    reserved = safe_model_id("con")
    assert reserved not in ("con", "CON"), reserved
    long_a = safe_model_id("x" * 200 + ":1")
    long_b = safe_model_id("x" * 200 + ":2")
    assert long_a != long_b and len(long_a) <= 80 and len(long_b) <= 80
    assert paths_for("../evil", root=root).root.parent == Path(root).resolve()
    return {"colon": colon, "design_model": design_model,
            "underscore": safe_model_id("qwen3_0.6b"),
            "upper": upper, "reserved": reserved, "long_len": len(long_a)}


def _scenario_cli_parse() -> dict:
    parser = build_parser()
    args = parser.parse_args(["collect", "--channel", "ollama", "--model", "qwen3:0.6b",
                              "--case-id", "F01-001", "--dry-run"])
    assert args.channel == "ollama" and args.model == "qwen3:0.6b", args
    assert args.base_url is None
    pre = parser.parse_args(["preflight", "--channel", "ollama",
                             "--model", "qwen3:0.6b"])
    assert pre.channel == "ollama" and pre.model == "qwen3:0.6b", pre
    return {"channel": args.channel, "model": args.model, "base_url": args.base_url}


def command_selfcheck() -> dict:
    cases = dev_cases(require_count=False)
    require(len(cases) >= 3, "selfcheck_cases", "dev case 不足 3 个，无法跑自检")
    first, response, second = _pick_selfcheck_cases(cases)
    root = Path(tempfile.mkdtemp(prefix="atomize-selfcheck-"))
    data_before = sorted(path.name for path in DATA_ROOT.iterdir()) if DATA_ROOT.exists() else []

    scenarios: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    failures: list[str] = []

    def run(name: str, fn) -> None:
        try:
            scenarios[name] = fn()
            checks[name] = True
        except Exception as exc:
            checks[name] = False
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    run("happy", lambda: _scenario_happy(root / "happy", cases, first, response, second))
    run("stale_sha", lambda: _scenario_stale_sha(root / "happy", cases, first, response))
    run("retry", lambda: _scenario_retry(root / "retry", cases, first, response))
    run("resume", lambda: _scenario_resume(root / "resume", cases, first, response))
    run("exhausted", lambda: _scenario_exhausted(root / "exhausted", cases, first, response))
    run("halt", lambda: _scenario_halt(root / "halt", cases))
    run("budget", lambda: _scenario_budget(root / "budget", cases, first, response, second))
    run("ollama", lambda: _scenario_ollama(root / "ollama", cases, first, response, second))
    run("model_dir", lambda: _scenario_model_dir(root))
    run("cli_parse", _scenario_cli_parse)
    run("model_channel", mc.selftest)

    def preflight_offline() -> dict:
        result = command_preflight(argparse.Namespace(
            case_id=None, all_dev=False, channel=None, model=None, base_url=None,
            api_key_env=None, effort="high", timeout=300.0, codex_bin=None,
            token_budget=DEFAULT_TOKEN_BUDGET, max_calls=DEFAULT_MAX_CALLS))
        assert result["ok"] is True and result["dry_run"] is True, result.get("problems")
        assert result["cases"]["dev"] == DEV_CASES
        assert len(result["per_case"]) == DEV_CASES
        assert set(result["channels"]) == {"codex", "chat", "anthropic", "ollama"}
        assert result["channels"]["ollama"] is True
        assert result["channel_detail"]["ollama"] == "configured"
        return {"cases": result["cases"], "prompt_sha_aggregate":
                result["prompt_sha_aggregate"], "channels": result["channels"]}

    run("preflight_offline", preflight_offline)

    def no_repo_writes() -> dict:
        after = sorted(path.name for path in DATA_ROOT.iterdir()) if DATA_ROOT.exists() else []
        assert after == data_before, f"selfcheck 写入了 data/：{after} != {data_before}"
        assert all(path.is_relative_to(root) for path in root.rglob("*"))
        return {"data_unchanged": True, "temp_files": len(list(root.rglob("*")))}

    run("no_repo_writes", no_repo_writes)

    return {"ok": not failures, "passed": not failures, "experiment": EXPERIMENT,
            "temp_dir": str(root), "checks": checks, "failures": failures,
            "scenarios": scenarios}


                                               


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--case-id", action="append", default=None,
                        help="dev case_id；可重复或逗号分隔")
    parser.add_argument("--all-dev", action="store_true", help="全部 30 个 dev case")


def _add_channel(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--channel",
                        choices=("codex", "chat", "anthropic", "ollama"), default=None)
    parser.add_argument("--model", default=None,
                        help="provider 的模型 ID，如 gpt-6.1-sol、qwen3:0.6b")
    parser.add_argument("--base-url", default=None,
                        help="API/ollama 通道的完整端点 URL；ollama 省略时默认 "
                             "http://127.0.0.1:11434/api/chat")
    parser.add_argument("--api-key-env", default=None,
                        help="读取密钥的环境变量名（默认 OPENCODE_API_KEY / ANTHROPIC_API_KEY；"
                             "ollama 不需要）")


def _add_budget(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_atomization_comparison",
        description="跨模型原子化采集：同一 dev cases / A1-v2 / 生产 parser+证据闸")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("selfcheck", help="离线自检（假通道，产物只写系统临时目录）")
    pre = sub.add_parser("preflight", help="零调用：case 数、prompt SHA、通道可用性、计划")
    _add_selection(pre)
    _add_channel(pre)
    _add_budget(pre)
    pre.add_argument("--effort", default="high")
    pre.add_argument("--timeout", type=float, default=mc.DEFAULT_TIMEOUT)
    pre.add_argument("--codex-bin", default=None)
    collect = sub.add_parser("collect", help="真实采集；必须显式 --channel/--model 与 case 选择")
    _add_selection(collect)
    _add_channel(collect)
    _add_budget(collect)
    collect.add_argument("--effort", default="high")
    collect.add_argument("--timeout", type=float, default=mc.DEFAULT_TIMEOUT)
    collect.add_argument("--codex-bin", default=None)
    collect.add_argument("--dry-run", action="store_true", help="只打印计划，不发送调用")
    blind = sub.add_parser("blind", help="由现有 raw 重建该模型的 blind-stage1/2")
    blind.add_argument("--model", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "selfcheck"
    try:
        if command == "selfcheck":
            result = command_selfcheck()
        elif command == "preflight":
            result = command_preflight(args)
        elif command == "collect":
            require(bool(args.channel and args.model), "channel_required",
                    "collect 必须显式给出 --channel 与 --model")
            result = command_collect(args)
        else:
            result = rebuild_blind(dev_cases(), paths_for(args.model))
            require(result["rows"] > 0, "no_blind_rows",
                    "没有可用成功 raw；先 collect")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if command in ("selfcheck", "preflight") and result.get("ok") is not True:
            raise SystemExit(2)
    except Exception as exc:
        if hasattr(exc, "code") and hasattr(exc, "message"):
            print(json.dumps({"ok": False, "error_code": exc.code,
                              "message": exc.message}, ensure_ascii=False),
                  file=sys.stderr)
            raise SystemExit(2) from None
        raise


if __name__ == "__main__":
    main()
