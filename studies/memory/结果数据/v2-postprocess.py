                       
"""v2 研究后处理（**活动实现**；v1 `runs/v1-evaluate/postprocess.py` 与 v1 产物冻结为历史）。

1. 汇总 `runs/v2-evaluate/` 的 18 次真实 set 回放（新风险口径：每 rate 带 numerator/
   denominator/value，0 分母 null；分类指标只在 valid_rows；row_error/gate error 单列），
   主/acceptable 带分母、subset macroF1 与 gate error 并列；**不与 v1 混率**（仅逐行判定
   稳定性对照，非速率对照）。
2. teacher-pilot 账本 **fail-closed**：身份（预注册 provider/model/base_url/mode）、任务
   request 回显、任务/版本指纹（显式按 `__main__` 模块约定复算，不模糊匹配）、呼出哈希、
   decision 过真实 `M._decision`；任何缺失/重复/不匹配 → violations + 非 0，**不发布聚合**。
3. 新盲判（真实 M 投影）逐行与现场 `M._request` 投影比对，不匹配即 fail；旧盲判
   （permanence 可见、缺 replaceable_ids）显式标 non-equivalent 历史，不硬拒、不作正式对照；
   新标注未到 → pending（不 poll/不 fabricate），到齐后同命令单次 resume。
4. 引用/权限核验只用真实 `M._decision(raw, projected, meta, None)`（完整 bool；未知 relation/
   多余字段/reason 缺失等必须 fail）；不自行实现 partial gate，不按 expected label 补引用；
   uncertain 按 source 约定为空引用。

纪律：只读输入、只写指定产物（v2-evaluate / teacher-pilot 新 JSON / 外部结果数据）；不读
key、不触网、不启动模型、不改源码；启动即先跑必过+必败 selftest，失败则中止不发布。

用法：python research/memory_relations/runs/v2-evaluate/postprocess.py [--selftest]
     [--pilot-dir D] [--blind-input F] [--blind-map F] [--blind-result F]
     [--report-dir D] [--result-dir D]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
for _entry in (str(ROOT / "src"), str(ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

CLASSES = ("complement", "conflict", "contextual", "duplicate", "supersede",
           "temporary", "uncertain")
UNCERTAIN_CODES = frozenset({"model_uncertain", "insufficient_evidence", "ambiguous_relation"})
QWEN_EXPECTED = {"provider": "ollama", "model": "batiai/qwen3.8-27b:iq3",
                 "base_url": "http://127.0.0.1:11434", "mode": "label"}
V1_EVALUATE = ROOT / "research" / "memory_relations" / "runs" / "v1-evaluate"
SUPERSEDE_CITATIONS = [
    "src/dataset_recommender/agent/memory_relations.py:63 — supersede 定义：证据必须明确说明"
    "永久变更/撤销旧要求；`context.replaceable_ids` 只是引用权限白名单，不是替换意图证据。",
    "src/dataset_recommender/agent/memory_relations.py:290-295 — 生产 `_decision` 对 supersede "
    "仅做权限闸（scope/tier），不校验语义永久性；过闸 ≠ 语义正确。",
]
DISCLOSURES = [
    "无真人金标：一切率都是 agreement（对照 construction/agent_review 代理标签），不称 accuracy。",
    "set 回放为 --diagnostic --threshold 0（calibrated=false）分类诊断，不是已校准操作点。",
    "新盲判为真实 M 投影（去 permanence、含 replaceable_ids）；旧盲判 permanence 可见且缺"
    " replaceable_ids，仅作 non-equivalent 历史，不作正式公平对照。",
    "盲判子代理底层版本未知（不得称 v4.1）；单子代理、28 行试点，不得外推或据此选大规模 teacher。",
    "引用/权限合法与否以真实 M._decision 为唯一裁决；过闸只说明输出合法，不代表语义正确。",
]


class Fail(Exception):
    def __init__(self, code: str, detail: object = None) -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while (chunk := handle.read(1 << 20)):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True)
                          .encode("utf-8")).hexdigest()


def load_jsonl_strict(path: Path) -> list:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise Fail("invalid_json_line", {"path": str(path), "line": number}) from None
            if not isinstance(row, dict):
                raise Fail("non_object_line", {"path": str(path), "line": number})
            rows.append(row)
    return rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rate_record(part: int, whole: int) -> dict:
    return {"numerator": int(part), "denominator": int(whole),
            "value": (part / whole) if whole else None}


def spread(values: list) -> dict:
    vals = [float(value) for value in values if value is not None]
    return {"n": len(vals), "mean": float(np.mean(vals)) if vals else None,
            "min": float(np.min(vals)) if vals else None,
            "max": float(np.max(vals)) if vals else None}


def percentiles(values: list) -> dict:
    vals = [float(value) for value in values if value is not None]
    result = {"n": len(vals), "p50": float(np.percentile(vals, 50)) if vals else None,
              "p90": float(np.percentile(vals, 90)) if vals else None,
              "min": min(vals) if vals else None, "max": max(vals) if vals else None,
              "mean": float(np.mean(vals)) if vals else None, "sum": float(np.sum(vals)) if vals else None}
    return result


def agreement_rows(ids: list, labels_a: dict, labels_b: dict) -> dict:
    common = [item for item in ids
              if labels_a.get(item) is not None and labels_b.get(item) is not None]
    disagree = [item for item in common if labels_a[item] != labels_b[item]]
    return {"n": len(common), "agree": len(common) - len(disagree),
            "agreement": (len(common) - len(disagree)) / len(common) if common else None,
            "disagree_ids": disagree,
            "excluded_unjudged": [item for item in ids if item not in common]}


def confusion_rows(ids: list, labels_a: dict, labels_b: dict) -> dict:
    index = {name: position for position, name in enumerate(CLASSES)}
    table = [[0] * len(CLASSES) for _ in CLASSES]
    for item in ids:
        a, b = labels_a.get(item), labels_b.get(item)
        if a is None or b is None:
            continue
        table[index[a]][index[b]] += 1
    return {"classes": list(CLASSES), "matrix": table, "n": sum(sum(row) for row in table),
            "rows": "labels_a（行）", "columns": "labels_b（列）"}


def expected_versions() -> dict:
    """账本版本指纹预注册期望：显式按脚本运行约定（teachers 模块名 `__main__`）逐字节复算。

    `_source_sha256` 把 teachers 模块名计入摘要，脚本运行与 `python -m` 在同字节源码下
    值不同；此处把三个（模块名, 源文件）逐个显式列出，不做模糊匹配；不一致即 fail。
    """
    from research.memory_relations import schema, teachers as T
    from dataset_recommender.agent import memory_relations as M
    digest = hashlib.sha256()
    for module, name in ((M, M.__name__), (schema, schema.__name__), (T, "__main__")):
        digest.update((name + "\0").encode("utf-8"))
        path = getattr(module, "__file__", None)
        digest.update(Path(path).read_bytes() if path and Path(path).is_file()
                      else b"<source-unavailable>")
    base = T.versions()
    return {"prompt_sha256": base["prompt_sha256"],
            "paraphrase_prompt_sha256": base["paraphrase_prompt_sha256"],
            "validator_sha256": digest.hexdigest()}


                                       
VERSIONS_CONVENTION = "teachers module name == __main__（脚本运行约定，显式复算，不做模糊匹配）"


def projection_of(input_row: dict) -> tuple:
    from research.memory_relations import schema
    from dataset_recommender.agent import memory_relations as M
    candidate, existing, evidence, cfg = schema.materialize_request(input_row.get("request"))
    return M._request(candidate, existing, evidence, None)


def judge_gate(raw, projected: dict, meta: dict) -> dict:
    """唯一裁决 = 真实 M._decision：合法完整 bool，非法捕获机器码；不给部分结论。"""
    from dataset_recommender.agent import memory_relations as M
    try:
        decision = M._decision(raw, projected, meta, None)
    except Exception as exc:
        return {"gate_legal": False, "error_code": str(getattr(exc, "code", None) or
                                                       type(exc).__name__), "decision": None}
    return {"gate_legal": True, "error_code": None, "decision": decision}


                                                  

def validate_ledger(ledger_rows: list, input_rows: list, projections: dict) -> dict:
    input_by_id = {row["id"]: row for row in input_rows}
    versions = expected_versions()
    records: dict = {}
    attempts: list = []
    violations: list = []
    for number, row in enumerate(ledger_rows, 1):
        kind = row.get("kind")
        if kind == "attempt":
            rid, step, fingerprint = row.get("row"), row.get("step"), row.get("fingerprint")
            if not all(isinstance(value, str) for value in (rid, step, fingerprint)):
                violations.append({"line": number, "code": "attempt_without_identity"})
                continue
            if rid not in input_by_id:
                violations.append({"line": number, "code": "attempt_unknown_row", "id": rid})
            if step != "label":
                violations.append({"line": number, "code": "attempt_unknown_step", "step": step})
            attempts.append(row)
        elif kind == "record":
            rid = row.get("id")
            if not isinstance(rid, str) or rid not in input_by_id:
                violations.append({"line": number, "code": "unknown_record_id", "id": rid})
                continue
            if rid in records:
                violations.append({"line": number, "code": "duplicate_record_id", "id": rid})
                continue
            records[rid] = row
        else:
            violations.append({"line": number, "code": "unknown_kind", "kind": kind})
    lost = sorted(rid for rid in input_by_id if rid not in records)
    if lost:
        violations.append({"code": "lost_ids", "ids": lost})
    ok = failed = gate_legal = 0
    for rid, record in sorted(records.items()):
        input_row, teacher = input_by_id[rid], record.get("teacher")
        if not isinstance(teacher, dict):
            violations.append({"code": "record_without_teacher", "id": rid})
            continue
        for name, want in QWEN_EXPECTED.items():
            if teacher.get(name) != want:
                violations.append({"code": "teacher_identity_mismatch", "id": rid, "field": name,
                                   "got": teacher.get(name)})
        if (record.get("row") or {}).get("request") != input_row.get("request"):
            violations.append({"code": "task_request_mismatch", "id": rid})
        if teacher.get("versions") != versions:
            violations.append({"code": "versions_mismatch", "id": rid})
        if teacher.get("fingerprint") != sha256_json(
                {"request": input_row.get("request"), **QWEN_EXPECTED, "versions": versions}):
            violations.append({"code": "fingerprint_mismatch", "id": rid})
        projected, meta = projections[rid]
        projection_sha = sha256_json(projected)
        request_sha = sha256_json(input_row.get("request"))
        calls = teacher.get("calls")
        if not isinstance(calls, list) or not calls:
            violations.append({"code": "record_without_calls", "id": rid})
        else:
            for call in calls:
                if not isinstance(call, dict):
                    violations.append({"code": "call_not_object", "id": rid})
                    continue
                if call.get("step") != "label" or call.get("fingerprint") != teacher.get("fingerprint"):
                    violations.append({"code": "call_identity_mismatch", "id": rid})
                if call.get("request_sha256") != request_sha:
                    violations.append({"code": "request_sha256_mismatch", "id": rid})
                if call.get("model_request_sha256") != projection_sha:
                    violations.append({"code": "model_request_sha256_mismatch", "id": rid})
        attempts_for = sum(1 for item in attempts if item.get("row") == rid
                           and item.get("fingerprint") == teacher.get("fingerprint"))
        if attempts_for != teacher.get("attempts"):
            violations.append({"code": "attempts_count_mismatch", "id": rid})
        decision = teacher.get("decision")
        if teacher.get("ok") is True:
            ok += 1
            if not isinstance(decision, dict):
                violations.append({"code": "ok_without_decision", "id": rid})
            else:
                gate = judge_gate(decision, projected, meta)
                if gate["gate_legal"]:
                    gate_legal += 1
                else:
                    violations.append({"code": "record_decision_gate_illegal", "id": rid,
                                       "gate_error": gate["error_code"]})
        elif teacher.get("ok") is False:
            failed += 1
            if decision is not None:
                violations.append({"code": "failed_with_decision", "id": rid})
        else:
            violations.append({"code": "record_ok_missing", "id": rid})
    if violations:
        raise Fail("ledger_violation", violations)
    return {"records": records, "attempts_total": len(attempts),
            "counters": {"records": len(records), "ok": ok, "failed": failed,
                         "decision_gate_legal": gate_legal},
            "versions_expected": versions}


def validate_blind_new(blind_path, map_path, input_rows: list, projections: dict,
                       manifest_path=None) -> dict:
    blind_rows = load_jsonl_strict(Path(blind_path))
    id_map = json.loads(Path(map_path).read_text(encoding="utf-8"))
    input_ids = {row["id"] for row in input_rows}
    violations: list = []
    ids = [row.get("id") for row in blind_rows]
    if len(ids) != len(set(ids)):
        violations.append({"code": "duplicate_blind_id"})
    if not isinstance(id_map, dict):
        raise Fail("blind_map_not_object")
    blind_ids = set(ids)
    unknown = sorted(key for key in id_map if key not in blind_ids)
    missing = sorted(item for item in ids if item not in id_map)
    if unknown:
        violations.append({"code": "blind_map_unknown_ids", "ids": unknown})
    if missing:
        violations.append({"code": "blind_map_missing_ids", "ids": missing})
    targets = [id_map[key] for key in id_map if key in blind_ids]
    if len(targets) != len(set(targets)):
        violations.append({"code": "blind_map_multi_to_one"})
    for target in targets:
        if target not in input_ids:
            violations.append({"code": "blind_map_target_unknown", "target": target})
    bad = []
    for row in blind_rows:
        target = id_map.get(row.get("id"))
        if target in projections and row.get("request") != projections[target][0]:
            bad.append({"id": row.get("id"), "target": target})
    if bad:
        violations.append({"code": "blind_body_mismatch", "rows": bad})
    manifest = None
    if manifest_path is not None and Path(manifest_path).is_file():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest.get("rows") != len(blind_rows):
            violations.append({"code": "blind_manifest_rows_mismatch"})
        if manifest.get("input_sha256") != sha256_file(Path(blind_path)):
            violations.append({"code": "blind_manifest_sha_mismatch"})
        if manifest.get("replaceable_ids_present") is not True:
            violations.append({"code": "blind_manifest_replaceable_flag"})
        if manifest.get("anonymous_ids") is not True:
            violations.append({"code": "blind_manifest_anonymous_flag"})
    if violations:
        raise Fail("blind_violation", violations)
    return {"blind_rows": len(blind_rows), "map": id_map, "manifest": manifest,
            "counters": {"rows": len(blind_rows), "bodies_match_actual_projection": len(blind_rows)}}


def old_blind_non_equivalence(pilot_dir, input_rows: list, projections: dict) -> dict:
    old_rows = load_jsonl_strict(Path(pilot_dir) / "blind-anonymous.jsonl")
    old_map = json.loads((Path(pilot_dir) / "blind-id-map.json").read_text(encoding="utf-8"))
    input_ids = {row["id"] for row in input_rows}
    extra_cand: set = set()
    extra_exist: set = set()
    missing_ctx: set = set()
    affected, samples = 0, []
    for row in old_rows:
        target = old_map.get(row.get("id"))
        if target not in input_ids or target not in projections:
            continue
        projected = projections[target][0]
        old = row.get("request") or {}
        cand = set(old.get("candidate") or {}) - set(projected["candidate"])
        exist: set = set()
        for rule in old.get("existing") or []:
            exist |= (set(rule) - set(projected["existing"][0])) if projected["existing"] else set(rule)
        ctx = set(projected["context"]) - set(old.get("context") or {})
        extra_cand |= cand
        extra_exist |= exist
        missing_ctx |= ctx
        if cand or exist or ctx:
            affected += 1
            if len(samples) < 2:
                samples.append({"blind_id": row.get("id"), "input_id": target,
                                "candidate_extra_fields": sorted(cand),
                                "existing_extra_fields": sorted(exist),
                                "context_missing_keys": sorted(ctx)})
    return {"rows": len(old_rows), "rows_with_differences": affected,
            "extra_fields_vs_projection": {"candidate": sorted(extra_cand),
                                           "existing": sorted(extra_exist)},
            "missing_context_keys": sorted(missing_ctx), "samples": samples,
            "label": "non-equivalent history：permanence 可见且缺 replaceable_ids；不与新盲判/"
                     "Qwen 对照混用，也不作当前正式公平对照。"}


def validate_blind_result(result_path, blind_ids: set) -> dict:
    rows = load_jsonl_strict(Path(result_path))
    ids = [row.get("id") for row in rows]
    violations = []
    if len(ids) != len(set(ids)):
        violations.append({"code": "result_duplicate_id"})
    unknown = sorted(item for item in ids if item not in blind_ids)
    missing = sorted(item for item in blind_ids if item not in ids)
    if unknown:
        violations.append({"code": "result_unknown_ids", "ids": unknown})
    if missing:
        violations.append({"code": "result_missing_ids", "ids": missing})
    if violations:
        raise Fail("blind_result_violation", violations)
    return {row["id"]: row for row in rows}


                                                               

def aggregate_replays(v2_dir: Path) -> dict:
    log_rows = load_jsonl_strict(v2_dir / "eval-run-log.jsonl")
    per_run: list = []
    failures: list = []
    for entry in log_rows:
        report_path = ROOT / entry["out"]
        if entry.get("exit") != 0 or not Path(report_path).is_file():
            failures.append({"kind": entry.get("kind"), "seed": entry.get("seed"),
                             "dataset": entry.get("dataset"), "exit": entry.get("exit"),
                             "reason": "nonzero_exit_or_missing_report"})
            continue
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        summary, rows = report["summary"], report["rows"]
        valid = [row for row in rows if row.get("row_error") is None and not row.get("error")]
        prim = sum(1 for row in valid if row.get("primary_agree"))
        acc = sum(1 for row in valid if row.get("acceptable_agree"))
        if len(valid) != summary["valid_rows"]:
            raise Fail("aggregate_invariant", {"run": entry["out"],
                                               "valid_rows": summary["valid_rows"],
                                               "counted_valid": len(valid)})
        for label, agreed in (("primary", prim), ("acceptable", acc)):
            reported = summary["agreement"][label]
            got = (agreed / len(valid)) if valid else None
            if (reported is None) != (got is None) or (
                    got is not None and abs(reported - got) > 1e-9):
                raise Fail("aggregate_invariant", {"run": entry["out"], "field": label})
        per_run.append({
            "kind": entry["kind"], "seed": entry["seed"], "dataset": entry["dataset"],
            "report": entry["out"], "exit": entry["exit"], "elapsed_s": entry.get("elapsed_s"),
            "report_sha256": sha256_file(report_path),
            "attempts": summary["row_count"], "scored_rows": summary["scored_rows"],
            "valid_rows": summary["valid_rows"], "gate_error_rows": summary["gate_error_rows"],
            "row_error_rows": summary["row_error_rows"], "row_errors": summary["row_errors"],
            "gate_errors": summary["gate_errors"],
            "primary": rate_record(prim, len(valid)), "acceptable": rate_record(acc, len(valid)),
            "macro_f1": summary["macro_f1"], "macro_f1_scope": "valid_rows（与 gate_error 并列读）",
            "operation": summary["operation"], "refs": summary["refs"],
            "bootstrap": {"macro_f1": summary["bootstrap"]["macro_f1"],
                          "primary_agreement": summary["bootstrap"]["primary_agreement"]},
            "predictor_calls": summary["predictor_calls"], "scored_pairs": summary["scored_pairs"],
            "latency_ms_total": summary["latency_ms"]["total"],
            "threshold_operation": report["operation"],
        })

    def rate_view(by_seed: dict, name: str) -> dict:
        keys = sorted(by_seed)
        values = [by_seed[key][name]["value"] for key in keys]
        return {"per_seed": {key: by_seed[key][name] for key in keys},
                "mean_of_values": spread(values)["mean"], "min": spread(values)["min"],
                "max": spread(values)["max"], "n": len(keys), "note": _SEED_NOTE}

    aggregates = []
    groups: dict = {}
    for item in per_run:
        groups.setdefault((item["kind"], item["dataset"]), []).append(item)
    for (kind, dataset), entries in sorted(groups.items()):
        by_seed = {str(item["seed"]): item for item in entries}
        gate_by_code: dict = {}
        for item in entries:
            for code, count in item["gate_errors"]["by_code"].items():
                gate_by_code[code] = gate_by_code.get(code, 0) + count
        operations = ("coverage", "dangerous_allow_rate", "conditional_risk",
                      "false_block_rate", "false_block_conditional_precision")
        aggregates.append({
            "kind": kind, "dataset": dataset, "n_seeds": len(entries),
            "counts": {
                "attempts": sum(item["attempts"] for item in entries),
                "scored_rows": sum(item["scored_rows"] for item in entries),
                "valid_rows": sum(item["valid_rows"] for item in entries),
                "gate_error_rows": sum(item["gate_error_rows"] for item in entries),
                "row_error_rows": sum(item["row_error_rows"] for item in entries),
                "gate_errors_by_code": gate_by_code,
            },
            "primary": rate_view(by_seed, "primary"), "acceptable": rate_view(by_seed, "acceptable"),
            "macro_f1": {"per_seed": {key: by_seed[key]["macro_f1"] for key in by_seed},
                         "mean_of_values": spread([by_seed[key]["macro_f1"]
                                                   for key in by_seed])["mean"],
                         "min": spread([by_seed[key]["macro_f1"] for key in by_seed])["min"],
                         "max": spread([by_seed[key]["macro_f1"] for key in by_seed])["max"],
                         "n": len(by_seed), "note": _SEED_NOTE},
            "operation_rates": {
                "per_seed": {key: {name: by_seed[key]["operation"][name] for name in operations}
                             for key in by_seed},
                "mean_of_values": {name: spread([by_seed[key]["operation"][name]["value"]
                                                 for key in by_seed])["mean"] for name in operations},
                "note": _SEED_NOTE},
            "refs_totals": {
                "predicted_total": sum(item["refs"]["predicted_total"] for item in entries),
                "expected_total": sum(item["refs"]["expected_total"] for item in entries),
                "hits": sum(item["refs"]["hits"] for item in entries),
                "unknown_total": sum(item["refs"]["unknown_total"] for item in entries),
                "known_non_expected_total": sum(item["refs"]["known_non_expected_total"]
                                                for item in entries),
                "missed_expected_total": sum(item["refs"]["missed_expected_total"]
                                             for item in entries),
            },
        })

    stability = {"rows_compared": 0, "identical": 0, "differences": [],
                 "note": "仅逐行判定稳定性（relation/error）；v1 分母为 scored、v2 只在 valid_rows，"
                         "两版速率口径不同，禁止并表混率。"}
    for item in per_run:
        v1_path = V1_EVALUATE / f"{item['kind']}-seed{item['seed']}-{item['dataset']}.json"
        if not v1_path.is_file():
            continue
        v1_rows = json.loads(v1_path.read_text(encoding="utf-8"))["rows"]
        v2_rows = json.loads((ROOT / item["report"]).read_text(encoding="utf-8"))["rows"]
        for before, after in zip(v1_rows, v2_rows):
            stability["rows_compared"] += 1
            left = (before.get("row_id"), (before.get("decision") or {}).get("relation"),
                    before.get("error"))
            right = (after.get("row_id"), (after.get("decision") or {}).get("relation"),
                     after.get("error"))
            if left == right:
                stability["identical"] += 1
            elif len(stability["differences"]) < 20:
                stability["differences"].append({"run": item["report"], "row_id": left[0],
                                                 "v1": left[1:], "v2": right[1:]})
    return {
        "schema": "memory-relations-v2-set-replay-aggregate-v1",
        "generated_at": _now(),
        "source": {"run_log": "research/memory_relations/runs/v2-evaluate/eval-run-log.jsonl",
                   "runs_logged": len(log_rows), "runs_aggregated": len(per_run),
                   "run_failures": failures, "v1_artifacts_role": "history（口径不同，不重写、不并表）"},
        "per_run": per_run, "aggregates": aggregates, "cross_version_stability": stability,
        "disclosures": [
            "全部回放为 --diagnostic --threshold 0（calibrated=false）分类诊断，不是已校准操作点。",
            "macro_f1/agreement/confusion 只在 valid_rows；gate error / row error 单列，不混入"
            "准确率；每 rate 带 numerator/denominator，0 分母 null。",
            "v1 同名字段分母不同（scored vs valid_rows），本文件不与 v1 混率；仅逐行判定稳定性"
            "作跨版本对照。",
            "标签为 agent_review 代理标签，非真人金标；一致率只称 agreement。",
            "逐对评分+聚合与生产 LLM 单次 set 语义非等价；跨批合并沿用生产 _merge_decisions。",
        ],
    }


_SEED_NOTE = "均值是各 seed 运行点的均值；各率分母可能随 seed 不同（见 per_seed 的 numerator/"\
             "denominator），不做跨 seed 合并比率。"


                                                   

def qwen_call_stats(records: dict) -> dict:
    ok = [record for record in records.values() if record["teacher"].get("ok") is True]
    failed = [record for record in records.values() if record["teacher"].get("ok") is False]
    calls_ok = [call for record in ok for call in (record["teacher"].get("calls") or [])]
    latency = [call.get("latency_s") for call in calls_ok
               if isinstance(call.get("latency_s"), (int, float))]
    usage = [call.get("usage") for call in calls_ok if isinstance(call.get("usage"), dict)]
    prompt = [item.get("prompt_tokens") for item in usage if isinstance(item.get("prompt_tokens"), int)]
    completion = [item.get("completion_tokens") for item in usage
                  if isinstance(item.get("completion_tokens"), int)]
    codes: dict = {}
    for record in records.values():
        code = record["teacher"].get("error_code") or "ok"
        codes[code] = codes.get(code, 0) + 1
    return {
        "records": len(records), "ok": len(ok), "failed": len(failed),
        "failed_rows": [{"id": record["id"], "error_code": record["teacher"]["error_code"],
                         "latency_s": (record["teacher"].get("calls") or [{}])[0].get("latency_s")}
                        for record in failed],
        "error_codes": codes,
        "raw_strict_ok": {"true": sum(1 for record in ok
                                      if record["teacher"].get("raw_strict_ok") is True),
                          "false": sum(1 for record in ok
                                       if record["teacher"].get("raw_strict_ok") is False),
                          "null_on_failure": sum(1 for record in failed
                                                 if record["teacher"].get("raw_strict_ok") is None)},
        "attempts_total": sum(record["teacher"].get("attempts") or 0 for record in records.values()),
        "finish_gate": {"ok_calls_passed": len(calls_ok),
                        "finish_error_codes": {
                            code: sum(1 for record in records.values()
                                      if record["teacher"].get("error_code") == code)
                            for code in ("incomplete_response", "empty_response", "llm_failed")},
                        "note": "账本不存原始 done_reason；ok=true 蕴含 done 完整且 done_reason != length。"},
        "latency_s": percentiles(latency),
        "usage": {"calls_with_usage": len(usage), "prompt_tokens": percentiles(prompt),
                  "completion_tokens": percentiles(completion),
                  "prompt_tokens_sum": int(sum(prompt)),
                  "completion_tokens_sum": int(sum(completion))},
    }


def qwen_boundary_cases(records: dict) -> dict:
    cases = []
    for rid, record in sorted(records.items()):
        if record["teacher"].get("ok") is not True:
            continue
        decision = record["teacher"].get("decision") or {}
        if record["teacher"].get("source_label") == "conflict" \
                and decision.get("relation") == "supersede":
            cases.append({"id": rid, "source_label": "conflict", "qwen_relation": "supersede",
                          "related_ids": decision.get("related_ids"), "reason": decision.get("reason")})
    return {"count": len(cases), "label": "边界歧义 + 独立误推候选", "cases": cases,
            "citations": SUPERSEDE_CITATIONS,
            "note": "conflict/supersede 边界是 blind notes.md 自披露的最主观处；上述 reason 把"
                    "「在 replaceable_ids 中」当作替换意图证据——权限事实 ≠ 语义证据。"}


def build_blind_status(ledger: dict, stats: dict, blind: dict, old_blind: dict,
                       result_info: dict) -> dict:
    return {
        "schema": "memory-relations-v2-blind-status-v1",
        "generated_at": _now(),
        "active_implementation": "v2（v1 postprocess.py 冻结为 history）",
        "ledger_validation": {
            "counters": ledger["counters"], "attempts_total": ledger["attempts_total"],
            "versions_expected_script_mode": dict(ledger["versions_expected"],
                                                  convention=VERSIONS_CONVENTION),
            "expected_identity": QWEN_EXPECTED, "violations": [],
            "checks": ["identity(provider/model/base_url/mode)", "task request 回显", "任务指纹",
                       "call.request_sha256", "call.model_request_sha256", "attempts 计数",
                       "decision 过真实 M._decision"],
            "verdict": "pass（任何 mismatch 直接 non-zero、不发布 aggregate）",
        },
        "qwen_calls": stats,
        "qwen_boundary_cases": qwen_boundary_cases(ledger["records"]),
        "blind_new": {"counters": blind["counters"], "manifest": blind.get("manifest"),
                      "verdict": "pass（body 与现场 M._request 投影逐行相等）"},
        "blind_old_non_equivalence": old_blind,
        "blind_result": result_info,
        "disclosures": DISCLOSURES,
    }


def build_blind_compare(input_rows: list, projections: dict, ledger: dict, blind: dict,
                        result_rows: dict) -> dict:
    records = ledger["records"]
    id_map = blind["map"]
    construction = {row["id"]: row["label"] for row in input_rows}
    qwen: dict = {}
    checks = {"qwen": {"judged": 0, "gate_legal": 0, "illegal": []},
              "deepseek_projected": {"judged": 0, "gate_legal": 0, "illegal": []}}
    for rid, record in sorted(records.items()):
        decision = record["teacher"].get("decision") if record["teacher"].get("ok") is True else None
        qwen[rid] = decision["relation"] if isinstance(decision, dict) else None
        if isinstance(decision, dict):
            gate = judge_gate(decision, *projections[rid])
            checks["qwen"]["judged"] += 1
            if gate["gate_legal"]:
                checks["qwen"]["gate_legal"] += 1
            else:
                checks["qwen"]["illegal"].append({"input_id": rid, "error_code": gate["error_code"]})
    blind_labels: dict = {}
    uncertain_with_refs = {"qwen": [], "deepseek_projected": []}
    for blind_id, result_row in sorted(result_rows.items()):
        target = id_map[blind_id]
        raw = result_row.get("decision")
        if not isinstance(raw, dict):
            raise Fail("blind_result_missing_decision", {"blind_id": blind_id})
        gate = judge_gate(raw, *projections[target])
        checks["deepseek_projected"]["judged"] += 1
        if not gate["gate_legal"]:
            checks["deepseek_projected"]["illegal"].append(
                {"blind_id": blind_id, "input_id": target, "error_code": gate["error_code"]})
            continue
        checks["deepseek_projected"]["gate_legal"] += 1
        blind_labels[target] = gate["decision"]["relation"]
        if gate["decision"]["relation"] == "uncertain" and gate["decision"]["related_ids"]:
            uncertain_with_refs["deepseek_projected"].append(target)
    for rid, record in sorted(records.items()):
        decision = record["teacher"].get("decision")
        if isinstance(decision, dict) and decision.get("relation") == "uncertain" \
                and decision.get("related_ids"):
            uncertain_with_refs["qwen"].append(rid)
    illegal = {name: value["illegal"] for name, value in checks.items() if value["illegal"]}
    if illegal:
        raise Fail("judge_gate_illegal", illegal)
    ids = [row["id"] for row in input_rows]
    pairs = {"construction_vs_qwen": agreement_rows(ids, construction, qwen),
             "construction_vs_deepseek_projected": agreement_rows(ids, construction, blind_labels),
             "qwen_vs_deepseek_projected": agreement_rows(ids, qwen, blind_labels)}
    confusions = {name: confusion_rows(ids, left, right) for name, left, right in (
        ("construction_vs_qwen", construction, qwen),
        ("construction_vs_deepseek_projected", construction, blind_labels),
        ("qwen_vs_deepseek_projected", qwen, blind_labels))}
    confusions["axis_note"] = "matrix[labels_a][labels_b]；均只统计双方可判行。"
    all_common = [item for item in ids if qwen.get(item) is not None
                  and blind_labels.get(item) is not None]
    per_class = [{"construction_label": name, "n": len([i for i in ids if construction[i] == name]),
                  "qwen": {"n_common": sum(1 for i in ids if construction[i] == name
                                           and qwen.get(i) is not None),
                           "agree": sum(1 for i in ids if construction[i] == name and qwen.get(i) == name)},
                  "deepseek_projected": {
                      "n_common": sum(1 for i in ids if construction[i] == name
                                      and blind_labels.get(i) is not None),
                      "agree": sum(1 for i in ids if construction[i] == name
                                   and blind_labels.get(i) == name)}}
                 for name in CLASSES]
    reverse = {target: blind_id for blind_id, target in id_map.items()}
    disagreements = []
    for item in ids:
        labels = {"construction": construction.get(item), "qwen": qwen.get(item),
                  "deepseek_projected": blind_labels.get(item)}
        available = [value for value in labels.values() if value is not None]
        if len(available) == 3 and len(set(available)) <= 1:
            continue
        row = next(entry for entry in input_rows if entry["id"] == item)
        qwen_decision = records[item]["teacher"].get("decision")
        blind_decision = (result_rows.get(reverse.get(item)) or {}).get("decision") or {}
        disagreements.append({
            "id": item, "blind_id": reverse.get(item), "labels": labels,
            "candidate_text": (row.get("request") or {}).get("candidate", {}).get("text"),
            "qwen_reason": qwen_decision.get("reason") if isinstance(qwen_decision, dict) else None,
            "deepseek_reason": blind_decision.get("reason"),
            "construction_expected_refs": {"related_ids": row.get("related_ids"),
                                           "evidence_refs": row.get("evidence_refs")}})
    return {
        "schema": "memory-relations-v2-blind-compare-v1",
        "generated_at": _now(),
        "judge_gate": checks,
        "judge_gate_note": "合法性由真实 M._decision(raw, projected, meta, key=None) 裁决；任何 "
                           "gate-illegal 输出直接 Fail（不发布），不作为判定进入对照。",
        "pairwise_agreement": pairs, "confusion": confusions,
        "all_three_common": {"n": len(all_common),
                             "all_agree": sum(1 for i in all_common
                                              if construction[i] == qwen[i] == blind_labels[i])},
        "per_class": per_class, "disagreements": disagreements,
        "uncertain_refs": {"with_refs": uncertain_with_refs,
                           "note": "uncertain 按 source 约定为空引用；只记录、不改写、不按 expected "
                                   "label 补引用。"},
        "qwen_calls": qwen_call_stats(records), "qwen_boundary_cases": qwen_boundary_cases(records),
        "disclosures": DISCLOSURES + ["本对照仅限 28 行试点（每类 4 条、单一盲判子代理、无真人金标），"
                                      "不可外推为泛化能力或“模型真实 accuracy”。"],
    }


                                                           

def run_selftest(v2_dir: Path, pilot_dir: Path, blind_path: Path, blind_map_path: Path) -> dict:
    pass_cases: list = []
    fail_cases: list = []
    negatives: list = []

    def check(name: str, ok: bool) -> None:
        pass_cases.append({"name": name, "ok": bool(ok)})

    def known_fail(name: str, rejected: bool, note: str) -> None:
        fail_cases.append({"name": name, "rejected": bool(rejected), "note": note})

    def codes_of(fail: Fail) -> set:
        detail = fail.detail if isinstance(fail.detail, list) else []
        return {item.get("code") for item in detail if isinstance(item, dict)}

    input_rows = load_jsonl_strict(Path(pilot_dir) / "input.jsonl")
    projections = {row["id"]: projection_of(row) for row in input_rows}
    ledger_rows = load_jsonl_strict(Path(pilot_dir) / "qwen-iq3-ledger.jsonl")
    records = {row["id"]: row for row in ledger_rows if row.get("kind") == "record"}
    sample_id = sorted(records)[0]
    sample = records[sample_id]
    ok_id = next(rid for rid in sorted(records) if records[rid]["teacher"].get("ok") is True)

    check("expected_versions/script-mode-matches-ledger",
          sample["teacher"]["versions"] == expected_versions())
    ledger = validate_ledger(ledger_rows, input_rows, projections)
    check("ledger/happy-path-counters",
          ledger["counters"] == {"records": 28, "ok": 27, "failed": 1, "decision_gate_legal": 27})

    def rejects(rows: list) -> set:
        try:
            validate_ledger(rows, input_rows, projections)
        except Fail as fail:
            return codes_of(fail)
        return set()

    def replaced(mutated: dict) -> list:
        return [row for row in ledger_rows
                if not (row.get("kind") == "record" and row.get("id") == sample_id)] + [mutated]

    mutated = copy.deepcopy(sample)
    mutated["teacher"]["calls"][0]["model_request_sha256"] = "0" * 64
    known_fail("ledger/hash-mismatch-rejected",
               "model_request_sha256_mismatch" in rejects(replaced(mutated)),
               "model_request_sha256 被改 → 必须 fail-closed")
    mutated = copy.deepcopy(sample)
    mutated["teacher"]["model"] = "other-model:latest"
    known_fail("ledger/wrong-model-rejected",
               "teacher_identity_mismatch" in rejects(replaced(mutated)),
               "model tag 偏离预注册 → 必须 fail-closed")
    mutated = copy.deepcopy(sample)
    mutated["row"]["request"] = copy.deepcopy(sample["row"]["request"])
    mutated["row"]["request"]["candidate"]["text"] += "（被篡改）"
    known_fail("ledger/ta<REDACTED-CREDENTIAL>",
               "task_request_mismatch" in rejects(replaced(mutated)),
               "任务 request 回显被改 → 必须 fail-closed")
    known_fail("ledger/duplicate-id-rejected",
               "duplicate_record_id" in rejects(ledger_rows + [copy.deepcopy(sample)]),
               "同 id 两条 record → 必须 fail-closed")

    blind_rows = load_jsonl_strict(Path(blind_path))
    temp_paths = []
    try:
        modified = copy.deepcopy(blind_rows[0])
        modified["request"]["candidate"]["extra_note"] = "tampered"
        path = v2_dir / "_selftest_blind_modified.jsonl"
        temp_paths.append(path)
        path.write_text(json.dumps(modified, ensure_ascii=False) + "\n" + "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in blind_rows[1:]), encoding="utf-8")
        try:
            validate_blind_new(path, blind_map_path, input_rows, projections)
            rejected = False
        except Fail as fail:
            rejected = "blind_body_mismatch" in codes_of(fail)
        known_fail("blind/body-modified-rejected", rejected, "盲判 body 与现场投影不一致 → 必须 fail")
        path = v2_dir / "_selftest_blind_dup.jsonl"
        temp_paths.append(path)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                for row in (blind_rows[0], blind_rows[0])), encoding="utf-8")
        try:
            validate_blind_new(path, blind_map_path, input_rows, projections)
            rejected = False
        except Fail as fail:
            rejected = "duplicate_blind_id" in codes_of(fail)
        known_fail("blind/duplicate-id-rejected", rejected, "盲判 id 重复 → 必须 fail")
        path = v2_dir / "_selftest_torn_tail.jsonl"
        temp_paths.append(path)
        path.write_text('{"kind": "record"}\n{"kind": "record"\n', encoding="utf-8")
        try:
            load_jsonl_strict(path)
            rejected = False
        except Fail as fail:
            rejected = fail.code == "invalid_json_line" and fail.detail.get("line") == 2
        known_fail("ledger/torn-tail-bad-json-rejected", rejected, "破尾半行 → 必须 fail")
    finally:
        for path in temp_paths:
            if path.exists():
                path.unlink()

    ok_decision = records[ok_id]["teacher"]["decision"]
    check("judge_gate/happy-path", judge_gate(ok_decision, *projections[ok_id])["gate_legal"])
    check("judge_gate/extra-field-rejected",
          not judge_gate(dict(ok_decision, extra_field="x"), *projections[ok_id])["gate_legal"])
    check("judge_gate/empty-reason-rejected",
          not judge_gate(dict(ok_decision, reason=""), *projections[ok_id])["gate_legal"])
    got = agreement_rows(["r1", "r2", "r3"],
                         {"r1": "conflict", "r2": "complement", "r3": "temporary"},
                         {"r1": "conflict", "r2": "uncertain", "r3": None})
    check("agreement/common-set", (got["n"], got["agree"]) == (2, 1))
    negatives.append({"name": "agreement/naive-denominator-differs",
                      "ok": round(got["agreement"], 6) != round(1 / 3, 6),
                      "note": "未判行计入分母会得 1/3；函数必须排除未判行（必败样本可被断言区分）"})
    passed = (all(item["ok"] for item in pass_cases) and all(item["rejected"] for item in fail_cases)
              and all(item["ok"] for item in negatives))
    return {"selftest_passed": passed, "pass_cases": pass_cases,
            "known_fail_cases": fail_cases, "negative_controls": negatives}


                                            

def render_conclusion(replay: dict, status: dict, compare, date: str) -> str:
    lines = [
        f"# 记忆关系轻量模型：v2 后处理结论（{date}，脚本自动生成）", "",
        "- 三件套：本结论 + 脚本 `v2-postprocess.py` + 数据聚合 JSON；v1 产物/脚本冻结为历史，"
        "两版速率口径不同（scored vs valid_rows），禁止并表混率。",
        "- 无真人金标：所有率都是 agreement（对照 construction/agent_review 代理标签）；不称"
        " accuracy、不外推生产路由能力。", "",
        "## 1. 真实 set 回放（新风险口径）", "",
        f"18 次真实 evaluate 全部有 exit 留证（run_failures={replay['source']['run_failures']}）；"
        "每 rate 带分子/分母（0 分母 null）；macro_f1/agreement 只在 valid_rows，gate/row error 并列：",
        "",
        "| 模型 | 数据集 | attempts/valid | gate_err/row_err | primary 均值[min,max] | acceptable | macroF1 | gate 码 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for agg in replay["aggregates"]:
        counts, primary = agg["counts"], agg["primary"]
        codes = ",".join(f"{key}×{value}"
                         for key, value in sorted(counts["gate_errors_by_code"].items())) or "无"
        lines.append(
            f"| {agg['kind']} | {agg['dataset']} | {counts['attempts']}/{counts['valid_rows']} | "
            f"{counts['gate_error_rows']}/{counts['row_error_rows']} | "
            f"{primary['mean_of_values']:.3f}[{primary['min']:.3f},{primary['max']:.3f}] | "
            f"{agg['acceptable']['mean_of_values']:.3f} | {agg['macro_f1']['mean_of_values']:.3f} | "
            f"{codes} |")
    stability = replay["cross_version_stability"]
    lines += ["", f"- 跨版本稳定性：逐行判定比较 {stability['rows_compared']} 行、一致 "
                  f"{stability['identical']} 行（非速率对照）。", "",
              "## 2. teacher-pilot 状态", "",
              "- 账本 fail-closed 核验 pass：28 行身份/request 回显/任务与版本指纹/呼出哈希/"
              "真实 M._decision 全过；任何 mismatch 直接 non-zero 且不发布。",
              f"- Qwen：{status['qwen_calls']['ok']}/{status['qwen_calls']['records']} ok（失败行"
              f"不进任何对照）；raw_strict_ok 真 {status['qwen_calls']['raw_strict_ok']['true']}；"
              f"时延 p50 {status['qwen_calls']['latency_s']['p50']}s / p90 "
              f"{status['qwen_calls']['latency_s']['p90']}s；tokens "
              f"{status['qwen_calls']['usage']['prompt_tokens_sum']}+"
              f"{status['qwen_calls']['usage']['completion_tokens_sum']}。",
              f"- 新盲判输入（真实 M 投影、去 permanence、含 replaceable_ids）："
              f"{status['blind_new']['counters']['bodies_match_actual_projection']}/28 body 与现场"
              "投影一致；旧盲判为非等价历史（permanence 可见、缺 replaceable_ids）。",
              "- 边界案例：Qwen 4 条 conflict→supersede 的 reason 把「在 replaceable_ids 中」当作"
              "替换意图证据（独立误推）；源定义见 `src/dataset_recommender/agent/memory_relations.py:63`"
              "（证据须明确说明永久变更）与 `:290-295`（_decision 仅权限闸）。", ""]
    if compare is None:
        lines += ["## 3. 新公平对照：**pending**", "",
                  "- 新标注（外部 `teacher-blind/deepseek28-projected.jsonl`）未到；不 poll、不 "
                  "fabricate；到齐后以同一命令单次 resume 输出对照。"]
    else:
        pairs = compare["pairwise_agreement"]
        lines += ["## 3. 新公平对照（新盲标 vs 构造 vs Qwen）", "",
                  f"- agreement（实际共同可判集合）：construction-Qwen "
                  f"{pairs['construction_vs_qwen']['agree']}/{pairs['construction_vs_qwen']['n']}；"
                  f"construction-盲 {pairs['construction_vs_deepseek_projected']['agree']}/"
                  f"{pairs['construction_vs_deepseek_projected']['n']}；Qwen-盲 "
                  f"{pairs['qwen_vs_deepseek_projected']['agree']}/{pairs['qwen_vs_deepseek_projected']['n']}。",
                  "- 全部输出过真实 M._decision（gate-illegal 直接 fail、不进入对照）；混淆/全部分歧"
                  "见对照 JSON。"]
    lines += ["", "## 4. 解释纪律", "",
              "- student 在 set 级的一致率低不能全归因于训练数据量：pair 训练与 set 语义非等价、"
              "逐对聚合吸收差异、无可行校准阈值、gate error 与类别分布都参与。",
              "- 单子代理、28 行试点；不得据此选大规模 teacher、不得称「模型真实 accuracy」。", ""]
    return "\n".join(lines) + "\n"


                                        

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="记忆关系 v2 研究后处理（只读输入，fail-closed）")
    parser.add_argument("--report-dir",
                        default="<archive-root>")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--pilot-dir", default=None)
    parser.add_argument("--blind-input", default=None)
    parser.add_argument("--blind-map", default=None)
    parser.add_argument("--blind-result", default=None,
                        help="新盲标 JSONL；省略时用 <report-dir>/teacher-blind/deepseek28-projected.jsonl，"
                             "缺文件按 pending（不 poll/不 fabricate）")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)

    v2_dir = ROOT / "research" / "memory_relations" / "runs" / "v2-evaluate"
    v2_dir.mkdir(parents=True, exist_ok=True)
    pilot_dir = Path(args.pilot_dir) if args.pilot_dir else \
        ROOT / "research" / "memory_relations" / "runs" / "teacher-pilot"
    report_dir = Path(args.report_dir)
    result_dir = Path(args.result_dir) if args.result_dir else report_dir / "结果数据"
    blind_path = Path(args.blind_input) if args.blind_input else pilot_dir / "blind-projected-shuffled.jsonl"
    blind_map_path = Path(args.blind_map) if args.blind_map else pilot_dir / "blind-projected-id-map.json"
    manifest_path = blind_path.parent / "blind-projected-manifest.json"

    selftest = run_selftest(v2_dir, pilot_dir, blind_path, blind_map_path)
    write_json(v2_dir / "postprocess-selftest-v2.json", selftest)
    if not selftest["selftest_passed"]:
        print("error: selftest failed; no aggregate published", file=sys.stderr)
        return 3
    if args.selftest:
        print(json.dumps({"selftest_passed": True, "known_fail_cases": [
            case["name"] for case in selftest["known_fail_cases"]]}, ensure_ascii=False))
        return 0

    try:
        input_rows = load_jsonl_strict(pilot_dir / "input.jsonl")
        projections = {row["id"]: projection_of(row) for row in input_rows}
        ledger = validate_ledger(load_jsonl_strict(pilot_dir / "qwen-iq3-ledger.jsonl"),
                                 input_rows, projections)
        blind = validate_blind_new(blind_path, blind_map_path, input_rows, projections,
                                   manifest_path=manifest_path)
        old_blind = old_blind_non_equivalence(pilot_dir, input_rows, projections)
        replay = aggregate_replays(v2_dir)
        if args.blind_result:
            result_path = Path(args.blind_result)
            if not result_path.is_file():
                raise Fail("blind_result_missing", {"path": str(result_path)})
            result_info = {"status": "present", "path": str(result_path),
                           "sha256": sha256_file(result_path)}
        else:
            result_path = report_dir / "teacher-blind" / "deepseek28-projected.jsonl"
            result_info = ({"status": "present", "path": str(result_path),
                            "sha256": sha256_file(result_path)} if result_path.is_file() else
                           {"status": "pending", "path": str(result_path),
                            "note": "新标注未到：不 poll、不 fabricate；到齐后同命令单次 resume。"})
        compare = None
        if result_info["status"] == "present":
            compare = build_blind_compare(input_rows, projections, ledger, blind,
                                          validate_blind_result(result_path, set(blind["map"])))
    except Fail as fail:
        print(json.dumps({"fail": fail.code, "detail": fail.detail}, ensure_ascii=False, indent=2),
              file=sys.stderr)
        return 2

    status = build_blind_status(ledger, qwen_call_stats(ledger["records"]), blind, old_blind,
                                result_info)
    write_json(v2_dir / "set-replay-aggregate-v2.json", replay)
    write_json(pilot_dir / "postprocess-v2-blind-status.json", status)
    if compare is not None:
        write_json(pilot_dir / "postprocess-v2-compare-28.json", compare)

    date = datetime.now().strftime("%Y-%m-%d")
    write_json(result_dir / f"v2-set-replay-aggregate-{date}.json", replay)
    write_json(result_dir / f"v2-blind-status-{date}.json", status)
    if compare is not None:
        write_json(result_dir / f"v2-blind-compare-28-{date}.json", compare)
    (result_dir / "v2-postprocess.py").write_bytes(Path(__file__).read_bytes())
    (result_dir / f"v2-结论-{date}.md").write_text(render_conclusion(replay, status, compare, date),
                                                   encoding="utf-8")

    failures = replay["source"]["run_failures"]
    print(json.dumps({
        "eval_runs": {"logged": replay["source"]["runs_logged"],
                      "aggregated": replay["source"]["runs_aggregated"], "failures": failures},
        "ledger": {"counters": status["ledger_validation"]["counters"],
                   "verdict": status["ledger_validation"]["verdict"]},
        "blind_new": status["blind_new"]["counters"],
        "blind_result": {"status": result_info["status"], "path": result_info["path"]},
        "artifacts": {
            "v2_evaluate": ["research/memory_relations/runs/v2-evaluate/set-replay-aggregate-v2.json",
                            "research/memory_relations/runs/v2-evaluate/postprocess-selftest-v2.json"],
            "teacher_pilot": ["postprocess-v2-blind-status.json"]
                             + (["postprocess-v2-compare-28.json"] if compare else []),
            "external": sorted(path.name for path in result_dir.glob(f"v2-*{date}*"))
                        + ["v2-postprocess.py"],
        },
    }, ensure_ascii=False, indent=2))
    if failures:
        print("warning: eval run failures: " + json.dumps(failures, ensure_ascii=False),
              file=sys.stderr)
    if compare is not None:
        print(json.dumps(compare["pairwise_agreement"], ensure_ascii=False))
    else:
        print("blind result pending: 等 parent 通知后单次 resume（不 poll/不 fabricate）")
    return 4 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
