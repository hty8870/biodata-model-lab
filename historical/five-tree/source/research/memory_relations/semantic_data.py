"""离线五叶研究行、满足集证书及盲任务；不调用模型、不替代生产权限裁决。

盲导出默认 canonical（保留源方向；盲性由随机 blind id 与行打乱提供）。reciprocal 只在空
shared background 或带 source 证据的声明（role_invariant / 逐字 explicit_inverse）下交换，
不以文本规则改写 frame；未证明安全即拒。

唯一学生输入复用 semantic_relations；旧七类 request 不适合五叶，因此只薄封装
build_data/teachers 的 JSON、hash、归一化、隔离路径和账本原语。oracle 只证明
作者明确模型内的关系；语言保真、自作者代理与独立裁决分别留来源，均不冒充人类金标。
CLI 错误为有界 SemanticDataError.code，失败回执隔离且返回非零；不回显秘密。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import re
import secrets
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dataset_recommender.agent import semantic_relations as S
from dataset_recommender.secret_patterns import find_secret_patterns, redact
from research.memory_relations import build_data as B
from research.memory_relations import teachers as T

FORMAT = "memory-relations-semantic-data-v6"
JOB_SPEC = Path(__file__).parent / "data" / "semantic-author-jobs-v6.json"
SPLITS = ("pilot", "train", "dev", "probability", "policy", "blind_final", "ood_final")
SEALED = frozenset({"blind_final", "ood_final"})
STATUSES = frozenset({"resolved", "ambiguous", "unresolved", "inconsistent_domain",
                      "empty_denotation", "not_comparable"})
ORIGINS = frozenset({"author_controlled_oracle", "author_proxy", "author_unresolved",
                     "official_coarse", "blind_teacher_proxy"})
ROW_FIELDS = {"id", "root_id", "cluster_id", "split", "label_space", "label", "pair",
              "label_origin", "oracle_relation", "NL_fidelity_origin", "adjudication_origin",
              "resolution_status", "domain", "phenomena", "oracle", "teacher_evidence"}
KINDS = frozenset({"proposition", "memory", "skill", "example"})
SIDE_FIELDS = {"kind", "text", "title", "trigger", "permanence", "payload"}
MODES = ("assertion", "constraint_satisfaction")
MAX_WORLDS = 65536


class SemanticDataError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def digest(value: object) -> str:
    return T._sha256_json(value)


def read_json(path: object) -> object:
    return T._strict_json(Path(path).read_text(encoding="utf-8"))


def load_rows(path: object) -> list[dict]:
    return [T._strict_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _token(value: object) -> str:
    if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value):
        raise SemanticDataError("invalid_identifier")
    return value


def _nonsecret(value: object) -> None:
    try:
        text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        raise SemanticDataError("invalid_json") from None
    if len(text) > 524288 or find_secret_patterns(text) or T._redacted_agent_value(value) != value:
        raise SemanticDataError("secret_or_oversized_record")


def student_pair(row: dict) -> dict:
    return S.validate_semantic_pair(row["pair"])


def student_text(row: dict) -> str:
    return S.semantic_pair_text(student_pair(row))


def pair_fingerprints(pair: dict) -> dict:
    forward = S.validate_semantic_pair(pair)
    reverse = S.swap_pair(forward)
    normal = B._normal_student_value(forward)
    inverse = B._normal_student_value(reverse)
    return {"forward": digest(forward), "reverse": digest(reverse),
            "unordered": digest(sorted([digest(normal), digest(inverse)]))}


INVERSE_QUALIFICATION_FORMAT = "semantic-inverse-qualification-v6"
SUPERVISION_RULE = "v6.1-forward-unless-inverse-qualified"


def _check_inverse_qualification(row: dict, qualification: dict) -> dict:
    """机械接收数据侧声明：完整逆向 pair + 同 frame.mode + 审读记录 hash 与源行 hash 边界。

    不校验、也不声称程序验证了自然语言保真；NL 保真为数据侧声明，仅按 hash 边界留痕。
    """
    required = {"id", "kind", "inverse_pair", "source_row_sha256", "review_kind",
                "review_record_sha256"}
    if type(qualification) is not dict or set(qualification) != required:
        raise SemanticDataError("invalid_inverse_qualification")
    if qualification["id"] != row["id"]:
        raise SemanticDataError("inverse_qualification_id_mismatch")
    if qualification["kind"] != "explicit_inverse":
        raise SemanticDataError("invalid_inverse_qualification_kind")
    _token(qualification["review_kind"])
    if type(qualification["review_record_sha256"]) is not str \
            or not re.fullmatch(r"[0-9a-f]{64}", qualification["review_record_sha256"]):
        raise SemanticDataError("invalid_inverse_review_record_sha256")
    if qualification["source_row_sha256"] != digest(row):
        raise SemanticDataError("inverse_qualification_source_drift")
    inverse = qualification["inverse_pair"]
    if type(inverse) is not dict or set(inverse) != {"old", "new", "frame"}:
        raise SemanticDataError("invalid_inverse_pair")
    _nonsecret(inverse)
    clean = S.validate_semantic_pair(inverse)
    if clean != inverse or clean != S.swap_pair(row["pair"]):
        raise SemanticDataError("inverse_pair_not_exact_swap")
    return {"status": "inverse_qualified", "format": INVERSE_QUALIFICATION_FORMAT,
            "evidence_sha256": digest(qualification), "review_kind": qualification["review_kind"],
            "basis": "mechanical_bounds_and_data_side_declaration"}


def _qualification_index(qualifications: list | None) -> dict:
    index = {}
    for record in qualifications or []:
        if type(record) is not dict or type(record.get("id")) is not str:
            raise SemanticDataError("invalid_inverse_qualification")
        _token(record["id"])
        if record["id"] in index:
            raise SemanticDataError("duplicate_inverse_qualification")
        index[record["id"]] = record
    return index


def supervision(row: dict, *, qualification: dict | None = None) -> dict:
    row = validate_row(row)
    label = row["label"]
    if label is None:
        return {"fine_targets": [None, None], "nli_targets": [None, None], "loss_mask": [0, 0],
                "inverse_qualification": {"status": "unverified"}}
    if row["label_space"] == "nli3":
        return {"fine_targets": [None, None], "nli_targets": [label, None], "loss_mask": [1, 0],
                "inverse_qualification": {"status": "unverified"}}
    bound = _check_inverse_qualification(row, qualification) if qualification is not None else None
    if bound is None:
        return {"fine_targets": [label, None], "nli_targets": [S.SEMANTIC_TO_NLI[label], None],
                "loss_mask": [1, 0], "inverse_qualification": {"status": "unverified"}}
    reverse = S.swap_relation(label)
    return {"fine_targets": [label, reverse],
            "nli_targets": [S.SEMANTIC_TO_NLI[label], S.SEMANTIC_TO_NLI[reverse]],
            "loss_mask": [1, 1], "inverse_qualification": bound}


class SourceBoundary:
    """每任务预分配owner及成员；越scope只归档，不作补票、first-appearance或多数。"""
    def __init__(self, document: dict):
        if document.get("schema") != "semantic-source-boundary-v1" or type(document.get("tasks")) is not list:
            raise SemanticDataError("invalid_source_boundary")
        self.tasks, members = {}, set()
        for task in document["tasks"]:
            if set(task) != {"task_id", "owner_id", "member_ids"} or type(task["member_ids"]) is not list:
                raise SemanticDataError("invalid_source_boundary_task")
            name, owner = _token(task["task_id"]), _token(task["owner_id"])
            ids = [_token(identifier) for identifier in task["member_ids"]]
            if name in self.tasks or len(set(ids)) != len(ids) or members.intersection(ids):
                raise SemanticDataError("ambiguous_source_boundary")
            self.tasks[name] = (owner, set(ids))
            members.update(ids)

    def authorized(self, owner_id: str, task_id: str, member_id: str) -> bool:
        if any(type(value) is not str for value in (owner_id, task_id, member_id)):
            return False
        owner, members = self.tasks.get(task_id, (None, set()))
        return owner_id == owner and member_id in members


def jsonl_records_bytes(path) -> list[dict]:
    """原记录bytes只去一个行分隔符；不strip/重dump，另存D.digest语义行hash。"""
    result = []
    for number, physical in enumerate(Path(path).read_bytes().splitlines(keepends=True), 1):
        content = physical[:-2] if physical.endswith(b"\r\n") else physical[:-1] if physical.endswith(b"\n") else physical
        if not content.strip():
            continue
        row = T._strict_json(content.decode("utf-8"))
        if type(row) is not dict:
            raise SemanticDataError("invalid_jsonl_record")
        result.append({"row": row, "line": number, "bytes_sha256": hashlib.sha256(content).hexdigest()})
    return result


def inverse_blind_input_sha256(entry: dict) -> str:
    """复用盲包producer的D.digest四个实际输入字段（不含input_sha256），不造新hash口径。"""
    keys = {"blind_id", "forward_pair", "candidate_inverse_pair", "instructions", "input_sha256"}
    if type(entry) is not dict or set(entry) != keys or type(entry["instructions"]) is not str \
            or not entry["instructions"].strip():
        raise SemanticDataError("invalid_actual_blind_entry")
    return digest({key: entry[key] for key in keys - {"input_sha256"}})


def validate_inverse_review(receipt: dict, *, identity: dict, entry: dict) -> dict:
    """唯一review入口：echo及严格9字段；NL保真仍是数据声明，不是程序语义证明。"""
    fields = {"inverse_decision", "forward_label", "inverse_label", "actually_reversed", "same_domain",
              "nl_fidelity", "complete_semantics", "role_reference_conflict", "reason"}
    if type(receipt) is not dict or set(receipt) != {"blind_id", "input_sha256", "reviewer", "review"} \
            or type(receipt.get("review")) is not dict or set(receipt["review"]) != fields:
        raise SemanticDataError("invalid_inverse_review_schema")
    identity_fields = {"id", "provider", "model", "family", "revision"}
    if type(identity) is not dict or set(identity) != identity_fields \
            or type(receipt["reviewer"]) is not dict or set(receipt["reviewer"]) != identity_fields:
        raise SemanticDataError("invalid_inverse_reviewer_schema")
    for value in identity.values():
        _token(value)
    if receipt["blind_id"] != entry["blind_id"] or receipt["input_sha256"] != inverse_blind_input_sha256(entry) \
            or receipt["reviewer"] != identity:
        raise SemanticDataError("review_identity_or_actual_input_drift")
    review = receipt["review"]
    if review["inverse_decision"] not in ("accept", "reject") \
            or any(value is not None and value not in S.SEMANTIC_RELATIONS for value in (review["forward_label"], review["inverse_label"])) \
            or any(type(review[key]) is not bool for key in ("actually_reversed", "same_domain", "nl_fidelity", "complete_semantics", "role_reference_conflict")) \
            or type(review["reason"]) is not str or not review["reason"].strip() or len(review["reason"]) > 2000:
        raise SemanticDataError("invalid_inverse_review_fields")
    if review["inverse_decision"] == "accept" and (review["forward_label"] is None or review["inverse_label"] is None \
            or any(review[key] is not True for key in ("actually_reversed", "same_domain", "nl_fidelity", "complete_semantics")) \
            or review["role_reference_conflict"] is not False or review["inverse_label"] != S.swap_relation(review["forward_label"])):
        raise SemanticDataError("incoherent_inverse_accept")
    return review


def load_qualified_source(manifest_path, *, construction_scope: object | None = None,
                          run_id: str | None = None) -> dict:
    """绑定实际审读与原row；默认仅开发来源，sealed 资格核验须显式构造声明。"""
    context = _construction_context(construction_scope, run_id)
    manifest_path = Path(manifest_path)
    doc = read_json(manifest_path)
    if doc.get("schema") != "semantic-qualified-source-v2":
        raise SemanticDataError("invalid_qualified_source_manifest")
    if context is not None:
        if doc.get("construction_scope") != context["audit"]:
            raise SemanticDataError("qualified_construction_scope_mismatch")
    elif "construction_scope" in doc:
        raise SemanticDataError("sealed_construction_scope_required")
    required = {"source_rows", "qualifications", "reviews", "review_scope", "packet_map", "import_manifest", "blind_manifest"}
    if type(doc.get("assets")) is not dict or set(doc["assets"]) != required:
        raise SemanticDataError("qualified_source_assets_missing")
    def bound_asset(name, reference):
        path = Path(reference["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        measured = hashlib.sha256(path.read_bytes()).hexdigest()
        if measured != reference["sha256"]:
            raise SemanticDataError("qualified_source_asset_drift:" + name)
        return {"path": str(path.absolute()), "sha256": measured}
    assets = {name: bound_asset(name, reference) for name, reference in doc["assets"].items()}
    source_records = jsonl_records_bytes(assets["source_rows"]["path"])
    rows = {record["row"]["id"]: record for record in source_records}
    if len(rows) != len(source_records):
        raise SemanticDataError("duplicate_source_row")
    if context is not None:
        source_rows = [record["row"] for record in source_records]
        _construction_members(source_rows, context, rows=True)
        cluster_rows(source_rows)
        for row in source_rows:
            evidence = row.get("teacher_evidence", {})
            if type(evidence) is dict and "construction_scope" in evidence \
                    and (evidence["construction_scope"] != context["audit"] or evidence.get("run_id") != run_id):
                raise SemanticDataError("teacher_construction_scope_mismatch")
    qual_records = jsonl_records_bytes(assets["qualifications"]["path"])
    qualifications = _qualification_index([record["row"] for record in qual_records])
    qual_bytes = {record["row"]["id"]: record["bytes_sha256"] for record in qual_records}
    import_doc = read_json(assets["import_manifest"]["path"])
    if import_doc.get("scope_corrected") is not True \
            or import_doc.get("files_sha256", {}).get("qualifications.jsonl") != assets["qualifications"]["sha256"]:
        raise SemanticDataError("qualified_import_scope_or_manifest_drift")
    packet = read_json(assets["packet_map"]["path"])
    by_blind = {item["blind_id"]: item for item in packet["items"]}
    if len(by_blind) != len(packet["items"]):
        raise SemanticDataError("duplicate_packet_binding")
    boundary_doc = doc["source_boundary"]
    boundary = SourceBoundary(boundary_doc)
    blind_manifest = read_json(assets["blind_manifest"]["path"])
    shards = doc.get("blind_shards")
    if type(shards) is not dict or set(shards) != set(blind_manifest.get("shards_sha256", {})) or set(shards) != set(boundary.tasks):
        raise SemanticDataError("actual_blind_shards_missing_or_scope_drift")
    actual_entries, blind_assets = {}, {}
    for task_id, reference in shards.items():
        asset = bound_asset(task_id, reference)
        if asset["sha256"] != blind_manifest["shards_sha256"][task_id]:
            raise SemanticDataError("frozen_blind_shard_drift")
        blind_assets[task_id] = asset
        records = jsonl_records_bytes(asset["path"])
        if {record["row"]["blind_id"] for record in records} != boundary.tasks[task_id][1]:
            raise SemanticDataError("actual_blind_scope_member_drift")
        for record in records:
            entry = record["row"]
            blind_id = entry["blind_id"]
            if blind_id in actual_entries or inverse_blind_input_sha256(entry) != entry["input_sha256"]:
                raise SemanticDataError("actual_blind_input_fingerprint_drift")
            if "instructions_sha256" in blind_manifest and digest(entry["instructions"]) != blind_manifest["instructions_sha256"]:
                raise SemanticDataError("frozen_blind_instructions_drift")
            actual_entries[blind_id] = (entry, task_id, record["bytes_sha256"])
    if not set(actual_entries) <= set(by_blind):
        raise SemanticDataError("map_actual_blind_id_drift")
    if "map_file_sha256" in blind_manifest and blind_manifest["map_file_sha256"] != assets["packet_map"]["sha256"]:
        raise SemanticDataError("frozen_blind_map_drift")
    if "identity_sha256" in blind_manifest and blind_manifest["identity_sha256"] != digest(packet["reviewer"]):
        raise SemanticDataError("frozen_blind_reviewer_drift")
    scope = read_json(assets["review_scope"]["path"])
    occurrences = scope.get("occurrences")
    review_records = jsonl_records_bytes(assets["reviews"]["path"])
    if type(occurrences) is not list or len(occurrences) != len(review_records):
        raise SemanticDataError("missing_review_bytes_or_scope_records")
    authorized, archived = {}, 0
    for record, occurrence in zip(review_records, occurrences):
        review = record["row"]
        if record["bytes_sha256"] != occurrence["line_sha256"]:
            raise SemanticDataError("review_scope_bytes_mismatch")
        blind = review.get("blind_id")
        if not boundary.authorized(occurrence["agent"], occurrence["item"], blind):
            archived += 1
            continue
        if blind not in by_blind or blind in authorized or actual_entries[blind][1] != occurrence["item"]:
            raise SemanticDataError("unknown_or_duplicate_authorized_review")
        validate_inverse_review(review, identity=packet["reviewer"], entry=actual_entries[blind][0])
        authorized[blind] = (record, occurrence)
    source_to_packet = {item["source_id"]: item for item in by_blind.values()}
    if len(source_to_packet) != len(by_blind):
        raise SemanticDataError("duplicate_packet_source_id")
    supervision_by_id, evidence_by_id, qualified = {}, {}, []
    for identifier, qualification in qualifications.items():
        if identifier not in rows or identifier not in source_to_packet:
            raise SemanticDataError("unknown_qualified_source_id")
        record = rows[identifier]
        row = validate_row(record["row"])
        if (context is None and row["split"] not in ("pilot", "train", "dev")) \
                or row["label_space"] != "semantic5" or row["label"] is None:
            raise SemanticDataError("inapplicable_formal_fine_source")
        supervised = supervision(row, qualification=qualification)
        if supervised["loss_mask"] != [1, 1] or supervised["inverse_qualification"]["status"] != "inverse_qualified":
            raise SemanticDataError("inverse_not_qualified")
        binding = source_to_packet[identifier]
        if binding["source_row_sha256"] != digest(row) or binding["label"] != row["label"] \
                or binding["root_id"] != row["root_id"] or binding["cluster_id"] != row["cluster_id"]:
            raise SemanticDataError("packet_source_row_drift")
        blind = binding["blind_id"]
        if blind not in authorized:
            raise SemanticDataError("missing_authorized_review")
        receipt, occurrence = authorized[blind]
        if receipt["bytes_sha256"] != qualification["review_record_sha256"]:
            raise SemanticDataError("qualified_review_bytes_drift")
        entry, actual_task, entry_bytes_sha = actual_entries[blind]
        expected_forward = student_pair(row)
        expected_inverse = S.swap_pair(expected_forward)
        if entry["forward_pair"] != expected_forward or entry["candidate_inverse_pair"] != expected_inverse \
                or qualification["inverse_pair"] != entry["candidate_inverse_pair"] \
                or binding["input_sha256"] != inverse_blind_input_sha256(entry) \
                or ("forward_pair_sha256" in binding and binding["forward_pair_sha256"] != digest(expected_forward)) \
                or ("candidate_inverse_pair_sha256" in binding and binding["candidate_inverse_pair_sha256"] != digest(expected_inverse)):
            raise SemanticDataError("actual_blind_source_pair_drift")
        review = validate_inverse_review(receipt["row"], identity=packet["reviewer"], entry=entry)
        if review["inverse_decision"] != "accept" or review["forward_label"] != row["label"] \
                or review["inverse_label"] != supervised["fine_targets"][1]:
            raise SemanticDataError("review_not_qualified_or_label_drift")
        evidence_by_id[identifier] = {"actual_blind_entry_bytes_sha256": entry_bytes_sha,
                                     "actual_blind_input_sha256": inverse_blind_input_sha256(entry),
                                     "source_row_sha256": digest(row), "source_row_bytes_sha256": record["bytes_sha256"],
                                     "qualification_sha256": digest(qualification), "qualification_bytes_sha256": qual_bytes[identifier],
                                     "review_record_sha256": receipt["bytes_sha256"], "reviewer_owner": occurrence["agent"],
                                     "task_id": occurrence["item"], "blind_id": blind,
                                     "frame_role_independent": True, "frame_basis": "authorized_proxy_review_no_role_reference_conflict"}
        supervision_by_id[identifier] = supervised
        qualified.append(row)
    lineage = {"manifest_path": str(manifest_path.absolute()), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "assets": assets, "actual_blind_shards": blind_assets,
               "source_boundary_sha256": digest(boundary_doc), "qualified_rows": len(qualified),
               "authorized_records": len(authorized), "archived_out_of_scope": archived,
               "supervision_rule": SUPERVISION_RULE, "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "proxy_only": True, "human_or_backend_certification": False}
    if context is not None:
        lineage["construction_scope"] = context["audit"]
    return {"rows": qualified, "supervision_by_id": supervision_by_id, "evidence_by_id": evidence_by_id, "lineage": lineage}


def _value(term: object, world: dict) -> object:
    if type(term) is list and len(term) == 2 and term[0] == "var":
        if term[1] not in world:
            raise SemanticDataError("unknown_oracle_variable")
        return world[term[1]]
    if type(term) in (bool, int, str) or term is None:
        return term
    raise SemanticDataError("invalid_oracle_term")


def _eval(expr: object, world: dict) -> bool:
    if type(expr) is bool:
        return expr
    if type(expr) is not list or not expr or type(expr[0]) is not str:
        raise SemanticDataError("invalid_oracle_expression")
    op, args = expr[0], expr[1:]
    if op in ("and", "or") and args:
        values = [_eval(arg, world) for arg in args]
        return all(values) if op == "and" else any(values)
    if op == "not" and len(args) == 1:
        return not _eval(args[0], world)
    if op == "implies" and len(args) == 2:
        return (not _eval(args[0], world)) or _eval(args[1], world)
    if op == "var" and len(args) == 1:
        value = _value(expr, world)
        if type(value) is not bool:
            raise SemanticDataError("nonboolean_oracle_atom")
        return value
    if op in ("eq", "lt", "le", "gt", "ge") and len(args) == 2:
        a, b = (_value(arg, world) for arg in args)
        if op == "eq":
            return type(a) is type(b) and a == b
        if type(a) is not int or type(b) is not int:
            raise SemanticDataError("invalid_oracle_comparison")
        return {"lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}[op]
    if op == "contains" and len(args) == 2:
        value, member = (_value(arg, world) for arg in args)
        if type(value) is not list:
            raise SemanticDataError("invalid_oracle_set")
        return member in value
    if op == "before" and len(args) == 3:
        order, a, b = (_value(arg, world) for arg in args)
        if type(order) is not list or a not in order or b not in order or len(set(order)) != len(order):
            raise SemanticDataError("invalid_oracle_order")
        return order.index(a) < order.index(b)
    raise SemanticDataError("invalid_oracle_expression")


def _cuts(expressions: list, variable: str) -> set[int]:
    cuts = set()
    def walk(expr):
        if type(expr) is not list or not expr:
            return
        if expr[0] in ("eq", "lt", "le", "gt", "ge") and len(expr) == 3:
            if expr[1] == ["var", variable]:
                if type(expr[2]) is not int:
                    raise SemanticDataError("nonconstant_integer_comparison")
                cuts.add(expr[2])
            if expr[2] == ["var", variable]:
                if type(expr[1]) is not int:
                    raise SemanticDataError("nonconstant_integer_comparison")
                cuts.add(expr[1])
        for arg in expr[1:]:
            walk(arg)
    for expr in expressions:
        walk(expr)
    if not cuts:
        raise SemanticDataError("unused_integer_domain")
    return cuts


def oracle_worlds(model: dict) -> tuple[list[dict], dict]:
    if type(model) is not dict or set(model) != {"mode", "domains", "domain_condition", "old", "new"} \
            or model["mode"] not in MODES or type(model["domains"]) is not dict or not model["domains"]:
        raise SemanticDataError("invalid_oracle_model")
    columns, abstraction = {}, {}
    exprs = [model["domain_condition"], model["old"], model["new"]]
    size = 1
    for name, domain in sorted(model["domains"].items()):
        _token(name)
        if type(domain) is not dict:
            raise SemanticDataError("invalid_oracle_domain")
        if set(domain) == {"values", "exhaustive"}:
            values = domain["values"]
            if domain["exhaustive"] is not True or type(values) is not list or not values \
                    or len({digest(v) for v in values}) != len(values):
                raise SemanticDataError("empty_or_nonexhaustive_domain")
            _nonsecret(values)
            columns[name] = copy.deepcopy(values)
            abstraction[name] = {"kind": "finite", "cells": copy.deepcopy(values)}
        elif set(domain) == {"integer", "unit"}:
            bounds = domain["integer"]
            if type(bounds) is not dict or set(bounds) != {"minimum", "maximum"} \
                    or type(bounds["minimum"]) is not int or not domain["unit"] \
                    or (bounds["maximum"] is not None and (type(bounds["maximum"]) is not int
                        or bounds["maximum"] < bounds["minimum"])):
                raise SemanticDataError("invalid_integer_domain")
            low, high = bounds["minimum"], bounds["maximum"]
            points = sorted(point for point in _cuts(exprs, name)
                            if point >= low and (high is None or point <= high))
            cells, start = [], low
            for point in points:
                if start < point:
                    cells.append({"minimum": start, "maximum": point - 1, "representative": start})
                cells.append({"minimum": point, "maximum": point, "representative": point})
                start = point + 1
            if high is None or start <= high:
                cells.append({"minimum": start, "maximum": high, "representative": start})
            columns[name] = [cell["representative"] for cell in cells]
            abstraction[name] = {"kind": "integer_comparison_partition", "unit": domain["unit"],
                                 "cells": cells, "tail_exhaustive": True}
        else:
            raise SemanticDataError("invalid_oracle_domain")
        size *= len(columns[name])
        if size > MAX_WORLDS:
            raise SemanticDataError("oracle_domain_too_large")
    worlds = [dict(zip(columns, values)) for values in itertools.product(*columns.values())]
    worlds = [world for world in worlds if _eval(model["domain_condition"], world)]
    if not worlds:
        raise SemanticDataError("empty_common_domain")
    return worlds, abstraction


def oracle_certificate(model: dict) -> dict:
    worlds, abstraction = oracle_worlds(model)
    p = {i for i, world in enumerate(worlds) if _eval(model["old"], world)}
    h = {i for i, world in enumerate(worlds) if _eval(model["new"], world)}
    if not p or not h:
        raise SemanticDataError("empty_denotation")
    if p == h:
        relation = S.SEMANTIC_RELATIONS[0]
    elif p < h:
        relation = S.SEMANTIC_RELATIONS[1]
    elif h < p:
        relation = S.SEMANTIC_RELATIONS[2]
    elif not p & h:
        relation = S.SEMANTIC_RELATIONS[3]
    else:
        relation = S.SEMANTIC_RELATIONS[4]
    parts = {"domain": set(range(len(worlds))), "old": p, "new": h,
             "intersection": p & h, "old_only": p - h, "new_only": h - p}
    return {"format": "semantic-satisfaction-certificate-v1", "relation": relation,
            "mode": model["mode"], "model_sha256": digest(model), "abstraction": abstraction,
            "regions": {name: {"cells": len(indices),
                               "witness": copy.deepcopy(worlds[min(indices)]) if indices else None,
                               "absence_proof": None if indices else "exhaustive_partition"}
                        for name, indices in parts.items()},
            "language_fidelity": "not_proved_by_oracle"}


def validate_row(row: dict) -> dict:
    if type(row) is not dict or set(row) - ROW_FIELDS \
            or not (ROW_FIELDS - {"oracle", "teacher_evidence"}) <= set(row):
        raise SemanticDataError("invalid_row_shape")
    _nonsecret(row)
    for name in ("id", "root_id", "cluster_id"):
        _token(row[name])
    if row["split"] not in SPLITS or row["label_space"] not in ("semantic5", "nli3") \
            or row["label_origin"] not in ORIGINS or row["resolution_status"] not in STATUSES:
        raise SemanticDataError("invalid_row_metadata")
    for name in ("NL_fidelity_origin", "adjudication_origin"):
        origin = row[name]
        if type(origin) is not dict or not {"kind", "independent", "human"} <= set(origin) \
                or type(origin["independent"]) is not bool or origin["human"] is not False:
            raise SemanticDataError("unsupported_origin_claim")
    pair = row["pair"]
    if type(pair) is not dict or set(pair) != {"old", "new", "frame"} \
            or type(pair["frame"]) is not dict or set(pair["frame"]) != {"mode", "background"}:
        raise SemanticDataError("invalid_pair_shape")
    for name in ("old", "new"):
        if type(pair[name]) is not dict or set(pair[name]) - SIDE_FIELDS:
            raise SemanticDataError("nonsemantic_pair_field")
    clean = S.validate_semantic_pair(pair)
    if clean != pair:
        raise SemanticDataError("pair_would_lose_information")
    label = row["label"]
    classes = S.SEMANTIC_RELATIONS if row["label_space"] == "semantic5" else S.NLI_CLASSES
    if label is not None and (type(label) is not str or label not in classes):
        raise SemanticDataError("invalid_label")
    if row["resolution_status"] != "resolved" and label is not None:
        raise SemanticDataError("nonresolved_label_must_be_null")
    if row["resolution_status"] == "resolved" and label is None:
        raise SemanticDataError("resolved_label_missing")
    domain = row["domain"]
    if type(domain) is not dict or set(domain) != {"consistent", "nonempty", "old_nonempty", "new_nonempty", "basis"}:
        raise SemanticDataError("invalid_domain_declaration")
    if label is not None and row["label_space"] == "semantic5" \
            and (row["resolution_status"] != "resolved"
                 or not all(domain[k] is True for k in ("consistent", "nonempty", "old_nonempty", "new_nonempty"))
                 or not pair["frame"]["background"].strip()):
        raise SemanticDataError("fine_without_nonempty_common_domain")
    if row["label_space"] == "nli3" and (row["oracle_relation"] is not None or row.get("oracle") is not None
                                           or row["label_origin"] != "official_coarse"):
        raise SemanticDataError("coarse_must_not_invent_fine")
    if row["oracle_relation"] is not None and row["oracle_relation"] not in S.SEMANTIC_RELATIONS:
        raise SemanticDataError("invalid_oracle_relation")
    oracle = row.get("oracle")
    if oracle is not None:
        if type(oracle) is not dict or set(oracle) != {"model", "certificate", "mapping_notes"}:
            raise SemanticDataError("invalid_oracle_evidence")
        cert = oracle_certificate(oracle["model"])
        for declared in oracle["model"]["domains"].values():
            finite_integer = ("integer" in declared and declared["integer"]["maximum"] is not None) or (
                "values" in declared and any(type(value) is int for value in declared["values"]))
            if finite_integer:
                quote = oracle["mapping_notes"].get("closed_domain_quote")
                if type(quote) is not str or not quote.strip() or quote not in pair["frame"]["background"]:
                    raise SemanticDataError("finite_numeric_domain_without_textual_boundary")
        if cert != oracle["certificate"] or cert["relation"] != row["oracle_relation"] \
                or (label is not None and label != cert["relation"]):
            raise SemanticDataError("oracle_label_mismatch")
    elif row["oracle_relation"] is not None or row["label_origin"] == "author_controlled_oracle":
        raise SemanticDataError("missing_oracle")
    meta = row["phenomena"]
    if type(meta) is not dict or not {"family", "language", "mode", "scenario", "skeleton", "source_ids", "author"} <= set(meta) \
            or meta["mode"] not in MODES or type(meta["source_ids"]) is not list or not meta["source_ids"]:
        raise SemanticDataError("missing_dependency_lineage")
    if oracle is not None and oracle["model"]["mode"] != meta["mode"]:
        raise SemanticDataError("mixed_interpretation_modes")
    for value in [meta["scenario"], meta["skeleton"], *meta["source_ids"]]:
        _token(value)
    if label is None and not meta.get("nonresolved_reason"):
        raise SemanticDataError("missing_nonresolved_reason")
    if label is not None and row["resolution_status"] != "resolved":
        raise SemanticDataError("invalid_resolution")
    return copy.deepcopy(row)


def _model_skeleton(model: dict) -> str:
    names = {name: "v" + str(i) for i, name in enumerate(sorted(model["domains"]))}
    literals = {}
    def collect(value):
        if type(value) is str and value not in literals:
            literals[value] = "item" + str(len(literals))
        elif type(value) is list:
            for item in value:
                collect(item)
    for domain in model["domains"].values():
        collect(domain.get("values", []))
    def shape(value):
        if type(value) is bool or value is None:
            return value
        if type(value) is int:
            return "number"
        if type(value) is str:
            return names.get(value, literals.get(value, value))
        if type(value) is list:
            return [shape(item) for item in value]
        return {names.get(key, key): "unit" if key == "unit" else shape(item)
                for key, item in sorted(value.items()) if key != "mode"}
    shaped = shape(model)
    shaped["old"], shaped["new"] = sorted([shaped["old"], shaped["new"]], key=digest)
    return digest(shaped)


def cluster_rows(rows: list[dict], *, merge: bool = False) -> tuple[list[dict], dict]:
    rows = [validate_row(row) for row in rows]
    if len({row["id"] for row in rows}) != len(rows):
        raise SemanticDataError("duplicate_row_id")
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(a, b):
        parents[find(b)] = find(a)
    keys, labels, reasons = {}, {}, Counter()
    for i, row in enumerate(rows):
        meta = row["phenomena"]
        fp = pair_fingerprints(row["pair"])
        linked = [("root", row["root_id"]), ("cluster", row["cluster_id"]),
                  ("scenario", meta["scenario"]), ("skeleton", meta["skeleton"]),
                  ("inverse_fingerprint", fp["unordered"])] + [("source", source) for source in meta["source_ids"]]
        if row.get("oracle"):
            linked.append(("oracle_skeleton", _model_skeleton(row["oracle"]["model"])))
        for key in linked:
            if key in keys:
                union(i, keys[key])
                reasons[key[0]] += 1
            else:
                keys[key] = i
        if row["label_space"] == "semantic5" and row["label"] is not None:
            for orientation, label in ((fp["forward"], row["label"]),
                                       (fp["reverse"], S.swap_relation(row["label"]))):
                if orientation in labels and labels[orientation] != label:
                    raise SemanticDataError("inverse_label_conflict")
                labels[orientation] = label
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    for indices in groups.values():
        if len({rows[i]["split"] for i in indices}) > 1:
            raise SemanticDataError("cross_split_dependency")
        ids = sorted({rows[i]["cluster_id"] for i in indices})
        if len(ids) > 1 and not merge:
            raise SemanticDataError("dependency_cluster_not_merged")
        if merge:
            cluster = "sc-" + digest(sorted({rows[i]["root_id"] for i in indices}))[:20]
            for i in indices:
                rows[i]["cluster_id"] = cluster
    counts = Counter(row["label"] or "nonresolved" for row in rows)
    family_support = {}
    for family in sorted({r["phenomena"]["family"] for r in rows}):
        members = [r for r in rows if r["phenomena"]["family"] == family]
        labels_in_family = {r["label"] for r in members if r["label_space"] == "semantic5" and r["label"]}
        family_support[family] = {"rows": len(members), "clusters": len({r["cluster_id"] for r in members}),
                                  "label_counts": dict(Counter(r["label"] or "nonresolved" for r in members)),
                                  "missing_leaves": sorted(set(S.SEMANTIC_RELATIONS) - labels_in_family)}
    return rows, {"rows": len(rows), "candidate_root_ids": len({r["root_id"] for r in rows}),
                  "dependency_clusters": len(groups), "independent_human_roots": 0,
                  "independence_status": "author_or_proxy_only_not_independent_gold",
                  "label_counts": dict(sorted(counts.items())), "dependency_edges": dict(reasons),
                  "resolved_rows": sum(row["label"] is not None for row in rows),
                  "resolved_clusters": len({r["cluster_id"] for r in rows if r["label"] is not None}),
                  "nonresolved_rows": sum(row["label"] is None for row in rows),
                  "label_origin_counts": dict(Counter(r["label_origin"] for r in rows)),
                  "resolution_counts": dict(Counter(r["resolution_status"] for r in rows)),
                  "family_support": family_support,
                  "cluster_primary_root": {rows[indices[0]]["cluster_id"]: min(rows[i]["root_id"] for i in indices)
                                           for indices in groups.values()},
                  "root_support_shortfall_vs_72": max(0, 72 - len(groups))}


def preallocate(spec: dict) -> list[dict]:
    plan = spec["first_wave"]
    rows = []
    for family_index, family in enumerate(spec["families"], 1):
        cluster_index = 0
        for split, count in plan["clusters_per_family_by_split"].items():
            for _ in range(count):
                cluster_index += 1
                cluster = f"v6-f{family_index:02}-c{cluster_index:02}"
                for root in range(1, plan["roots_per_planned_cluster"] + 1):
                    rows.append({"root_id": f"{cluster}-r{root}", "cluster_id": cluster,
                                 "family": family, "split": split, "content_status": "unwritten_slot",
                                 "accepted_root": False, "sealed": split in SEALED,
                                 "primary": root == 1})
    if len(rows) != 900 or len({r["cluster_id"] for r in rows}) != 180:
        raise SemanticDataError("invalid_preallocation_plan")
    return rows


CONSTRUCTION_SCOPE_FORMAT = "semantic-sealed-construction-scope-v1"


def _construction_context(scope_path: object | None, run_id: str | None) -> dict | None:
    """显式构造声明，不是密钥或认证；只绑定本次 run 与已锁定 allocation 元数据。"""
    if scope_path is None:
        return None
    scope_path = Path(scope_path)
    doc = read_json(scope_path)
    fields = {"schema", "run_id", "splits", "allocation", "scope_sha256"}
    if type(doc) is not dict or set(doc) != fields or doc["schema"] != CONSTRUCTION_SCOPE_FORMAT:
        raise SemanticDataError("invalid_construction_scope")
    _nonsecret(doc)
    if doc["scope_sha256"] != digest({key: value for key, value in doc.items() if key != "scope_sha256"}):
        raise SemanticDataError("construction_scope_hash_mismatch")
    _token(doc["run_id"])
    if run_id != doc["run_id"]:
        raise SemanticDataError("construction_run_mismatch")
    splits = doc["splits"]
    if type(splits) is not list or not splits or any(type(split) is not str or split not in SEALED for split in splits) \
            or len(set(splits)) != len(splits):
        raise SemanticDataError("invalid_construction_splits")
    reference = doc["allocation"]
    if type(reference) is not dict or set(reference) != {"path", "sha256", "members_sha256"} \
            or type(reference["path"]) is not str or not reference["path"].strip():
        raise SemanticDataError("invalid_construction_allocation")
    path = Path(reference["path"])
    if not path.is_absolute():
        path = scope_path.parent / path
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
        raise SemanticDataError("construction_allocation_hash_mismatch")
    members, clusters = {}, {}
    keys = {"root_id", "cluster_id", "family", "split"}
    for slot in load_rows(path):
        if type(slot) is not dict or not keys <= set(slot) \
                or set(slot) - keys - {"content_status", "accepted_root", "sealed", "primary"}:
            raise SemanticDataError("invalid_construction_allocation")
        member = {key: _token(slot[key]) for key in keys}
        cluster, partition = member["cluster_id"], (member["family"], member["split"])
        if member["root_id"] in members or member["split"] not in splits \
                or (cluster in clusters and clusters[cluster] != partition):
            raise SemanticDataError("invalid_construction_allocation")
        members[member["root_id"]] = member
        clusters[cluster] = partition
    if not members or digest(sorted(members.values(), key=lambda item: item["root_id"])) != reference["members_sha256"]:
        raise SemanticDataError("construction_members_hash_mismatch")
    audit = {"schema": CONSTRUCTION_SCOPE_FORMAT, "run_id": run_id, "splits": sorted(splits),
             "allocation_sha256": reference["sha256"], "members_sha256": reference["members_sha256"],
             "scope_sha256": doc["scope_sha256"]}
    return {"members": members, "audit": audit}


def _construction_members(items: list[dict], context: dict, *, rows: bool = False) -> None:
    for item in items:
        member = {key: item[key] for key in ("root_id", "cluster_id", "split")}
        member["family"] = item["phenomena"]["family"] if rows else item["family"]
        if context["members"].get(member["root_id"]) != member:
            raise SemanticDataError("construction_member_mismatch")


def author_jobs(assignments: list[dict], spec: dict, *, construction_scope: object | None = None,
                run_id: str | None = None) -> list[dict]:
    context = _construction_context(construction_scope, run_id)
    if context is not None:
        _construction_members(assignments, context)
    jobs = []
    for item in assignments:
        if item["split"] in SEALED and context is None:
            continue
        job = {"job_id": "author-" + item["root_id"], "root_id": item["root_id"],
               "cluster_id": item["cluster_id"], "family": item["family"],
               "instructions": spec["author_instructions"], "expected_schema": FORMAT, "human_gold": False}
        if context is not None:
            job["construction_scope"] = context["audit"]
        jobs.append(job)
    return jobs


def import_authors(receipts: list[dict], assignments: list[dict], *, construction_scope: object | None = None,
                   run_id: str | None = None) -> dict:
    context = _construction_context(construction_scope, run_id)
    if context is not None:
        _construction_members(assignments, context)
    slots = {row["root_id"]: row for row in assignments}
    accepted, rejected, seen = [], [], set()
    for number, raw in enumerate(receipts, 1):
        try:
            row = validate_row(raw)
            slot = slots.get(row["root_id"])
            if slot is None or (slot["split"] in SEALED and context is None) or row["split"] != slot["split"] \
                    or row["cluster_id"] != slot["cluster_id"] or row["phenomena"]["family"] != slot["family"]:
                raise SemanticDataError("unauthorized_author_slot")
            if row["root_id"] in seen:
                raise SemanticDataError("duplicate_author_root")
            if row["label_origin"] not in ("author_proxy", "author_controlled_oracle", "author_unresolved"):
                raise SemanticDataError("author_origin_mismatch")
            seen.add(row["root_id"])
            accepted.append(row)
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            rejected.append({"line": number, "code": getattr(exc, "code", "invalid_author_receipt"),
                             "raw_redacted": T._redacted_agent_value(raw)})
    accepted, audit = cluster_rows(accepted, merge=context is None)
    if context is not None:
        audit["construction_scope"] = context["audit"]
    return {"accepted": accepted, "quarantine": rejected, "audit": audit,
            "accepted_count": len(accepted), "quarantine_count": len(rejected),
            "failed_by_code": dict(Counter(item["code"] for item in rejected)),
            "received": len(receipts), "missing_slots": len(slots) - len(seen), "api_calls": 0}


def _snapshot_leaves(value: object, prefix: str = "") -> dict:
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            result.update(_snapshot_leaves(item, f"{prefix}.{key}".strip(".")))
        return result or {prefix: {}}
    if type(value) is list:
        result = {}
        for i, item in enumerate(value):
            result.update(_snapshot_leaves(item, f"{prefix}.{i}"))
        return result or {prefix: []}
    return {prefix: value}


def assess_action(event: dict) -> dict:
    _nonsecret(event)
    for name in ("before", "proposal", "after", "authority", "source", "cas", "joint_model"):
        if name not in event or type(event[name]) is not dict:
            raise SemanticDataError("invalid_action_event")
    before, after = event["before"], event["after"]
    proposal = event["proposal"]
    if proposal.get("operation") not in ("noop", "append", "edit", "replace") \
            or type(proposal.get("authorized_paths")) is not list or type(proposal.get("patch")) is not dict \
            or any(type(path) is not str for path in proposal["authorized_paths"]):
        raise SemanticDataError("invalid_action_proposal")
    authority = event["authority"]
    if authority.get("authorized") is not True or any(
            type(authority.get(k)) is not str or not authority[k].strip()
            or authority[k] != authority.get("expected_" + k) for k in ("actor", "project", "endpoint")):
        return {"action": "reject", "reason": "authority_mismatch", "preserves_unrelated": False}
    source = event["source"]
    if source.get("fresh") is not True or type(source.get("id")) is not str or not source["id"].strip() \
            or any(type(source.get(k)) is not int or source[k] < 0
                   for k in ("expected_version", "observed_version")) \
            or source["expected_version"] != source["observed_version"]:
        return {"action": "defer", "reason": "source_version_or_freshness", "preserves_unrelated": None}
    cas = event["cas"]
    if any(type(cas.get(k)) is not int or cas[k] < 0 for k in ("expected", "observed")) \
            or type(before.get("revision")) is not int or cas["expected"] != cas["observed"] \
            or cas["observed"] != before["revision"]:
        return {"action": "defer", "reason": "cas_mismatch", "preserves_unrelated": None}
    a, b = _snapshot_leaves(before), _snapshot_leaves(after)
    changed = {path for path in set(a) | set(b) if (path not in a or path not in b or a[path] != b[path])}
    allowed = set(proposal["authorized_paths"]) | {"revision"}
    if changed - allowed:
        return {"action": "reject", "reason": "unauthorized_clause_change", "preserves_unrelated": False}
    patch = proposal["patch"]
    if changed - {"revision"} != set(patch) or any(path not in b or b[path] != value for path, value in patch.items()):
        return {"action": "reject", "reason": "proposal_patch_mismatch", "preserves_unrelated": True}
    expected_revision = before["revision"] + (proposal["operation"] != "noop")
    if after.get("revision") != expected_revision:
        return {"action": "reject", "reason": "after_revision_mismatch", "preserves_unrelated": True}
    worlds, _ = oracle_worlds(event["joint_model"])
    if not any(_eval(event["joint_model"]["old"], world) for world in worlds):
        return {"action": "reject", "reason": "joint_inconsistent", "preserves_unrelated": True}
    return {"action": proposal["operation"], "reason": "mechanical_research_checks_passed",
            "preserves_unrelated": True}


def validate_action_receipt(event: dict, receipt: dict) -> dict:
    result = assess_action(event)
    if type(receipt) is not dict or receipt != result \
            or (receipt.get("action") in ("noop", "append", "edit", "replace")
                and receipt.get("preserves_unrelated") is not True):
        raise SemanticDataError("false_action_receipt")
    return result


def _contract_sha() -> str:
    return hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest()


EXPORT_RULE = "v6.2-canonical-default"
SWAP_POLICIES = ("canonical", "reciprocal")


def _swap_evidence_index(evidence: list | None) -> dict:
    index = {}
    for record in evidence or []:
        if type(record) is not dict or not {"id", "kind", "evidence"} <= set(record) \
                or not set(record) <= {"id", "kind", "evidence", "inverse_pair"}:
            raise SemanticDataError("invalid_swap_evidence")
        _token(record["id"])
        if record["id"] in index:
            raise SemanticDataError("duplicate_swap_evidence")
        if record["kind"] not in ("role_invariant", "explicit_inverse"):
            raise SemanticDataError("invalid_swap_evidence_kind")
        _nonsecret(record)
        index[record["id"]] = record
    return index


def _swap_decision(row: dict, policy: str, evidence: dict | None) -> dict:
    """canonical 永不交换；reciprocal 只在空 shared background 或带 source 证据声明下交换。"""
    if policy == "canonical":
        return {"pair": student_pair(row), "swapped": False, "basis": "canonical"}
    if not row["pair"]["frame"]["background"].strip():
        return {"pair": S.swap_pair(student_pair(row)), "swapped": True, "basis": "empty_frame"}
    if evidence is None:
        raise SemanticDataError("unproven_swap_safety")
    if type(evidence["evidence"]) is not str or not evidence["evidence"].strip():
        raise SemanticDataError("missing_swap_evidence_text")
    if evidence["kind"] == "role_invariant":
        return {"pair": S.swap_pair(student_pair(row)), "swapped": True,
                "basis": "role_invariant", "evidence_sha256": digest(evidence)}
    inverse = evidence.get("inverse_pair")
    if type(inverse) is not dict or set(inverse) != {"old", "new", "frame"}:
        raise SemanticDataError("invalid_inverse_pair")
    _nonsecret(inverse)
    clean = S.validate_semantic_pair(inverse)
    if clean != inverse or digest(inverse["old"]) != digest(row["pair"]["new"]) \
            or digest(inverse["new"]) != digest(row["pair"]["old"]):
        raise SemanticDataError("inverse_pair_not_exact_swap")
    return {"pair": clean, "swapped": True, "basis": "explicit_inverse",
            "evidence_sha256": digest(evidence)}


def export_blind(rows: list[dict], blind_dir: object, map_path: object, *, teacher: dict,
                 shard_size: int = 12, source: object, swap_policy: str = "canonical",
                 swap_evidence: list | None = None, construction_scope: object | None = None,
                 run_id: str | None = None) -> dict:
    """默认保留源方向；sealed 仅接显式构造声明，reciprocal 仍须独立的交换证据。"""
    context = _construction_context(construction_scope, run_id)
    rows, audit = cluster_rows(rows)
    if context is not None:
        _construction_members(rows, context, rows=True)
    elif any(row["split"] in SEALED for row in rows):
        raise SemanticDataError("sealed_content_export_forbidden")
    if any(row["label_space"] != "semantic5" for row in rows):
        raise SemanticDataError("fine_teacher_requires_semantic5")
    if type(shard_size) is not int or shard_size < 1 or shard_size > 100:
        raise SemanticDataError("invalid_shard_size")
    if swap_policy not in SWAP_POLICIES:
        raise SemanticDataError("invalid_swap_policy")
    if swap_policy == "canonical" and swap_evidence is not None:
        raise SemanticDataError("swap_evidence_without_reciprocal_policy")
    evidence_index = _swap_evidence_index(swap_evidence)
    unknown = set(evidence_index) - {row["id"] for row in rows}
    if unknown:
        raise SemanticDataError("unknown_swap_evidence_id")
    for key in ("id", "provider", "model", "family", "revision"):
        _token(teacher[key])
    _nonsecret(teacher)
    blind_dir, map_path = Path(blind_dir), Path(map_path)
    T._outside_blind_paths(blind_dir, map_path, map_path.with_suffix(".ledger.jsonl"), Path(source))
    if blind_dir.exists() or map_path.exists():
        raise SemanticDataError("blind_output_exists")
    spec = read_json(JOB_SPEC)
    instructions = spec["teacher_instructions"]
    entries, items = [], []
    for row in rows:
        decision = _swap_decision(row, swap_policy, evidence_index.get(row["id"]))
        entry = {"blind_id": "blind-" + secrets.token_hex(16), "pair": decision["pair"],
                 "instructions": instructions}
        entry["input_sha256"] = digest(entry)
        entries.append(entry)
        item = {"entry": entry, "source_row": row, "source_sha256": digest(row),
                "swapped": decision["swapped"], "swap_basis": decision["basis"]}
        if "evidence_sha256" in decision:
            item["swap_evidence_sha256"] = decision["evidence_sha256"]
        items.append(item)
    secrets.SystemRandom().shuffle(items)
    entries = [item["entry"] for item in items]
    payloads = {f"shard-{i // shard_size + 1:03}.jsonl": B._jsonl_bytes(entries[i:i + shard_size])
                for i in range(0, len(entries), shard_size)}
    construction = {"construction_scope": context["audit"]} if context is not None else {}
    map_doc = {"format": FORMAT + "-blind-map", "teacher": teacher,
               "contract_sha256": _contract_sha(), "prompt_sha256": digest(instructions),
               "export_rule": EXPORT_RULE, "swap_policy": swap_policy,
               "swap_evidence_sha256": digest(swap_evidence) if swap_evidence is not None else None,
               "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(), "items": items,
               "shards": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}, **construction}
    B._write_new_directory(blind_dir, payloads)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_bytes(B._json_bytes(map_doc))
    T._append_event(map_path.with_suffix(".ledger.jsonl"),
                    {"kind": "export", "map_sha256": digest(map_doc), "api_call": False, **construction})
    return {"rows": len(rows), "shards": len(payloads), "map_sha256": digest(map_doc),
            "swap_policy": swap_policy, "swapped_rows": sum(item["swapped"] for item in items),
            "human_gold": False, "api_calls": 0, "dependency_clusters": audit["dependency_clusters"], **construction}


def import_teacher(receipts: list[dict], blind_dir: object, map_path: object, ledger_path: object,
                   *, run_id: str, construction_scope: object | None = None) -> dict:
    _token(run_id)
    context = _construction_context(construction_scope, run_id)
    construction = {"construction_scope": context["audit"]} if context is not None else {}
    blind_dir, map_path, ledger_path = Path(blind_dir), Path(map_path), Path(ledger_path)
    T._outside_blind_paths(blind_dir, map_path, ledger_path, map_path.with_suffix(".source.jsonl"))
    doc = read_json(map_path)
    if doc["format"] != FORMAT + "-blind-map" or doc["contract_sha256"] != _contract_sha():
        raise SemanticDataError("blind_version_mismatch")
    export_events = load_rows(map_path.with_suffix(".ledger.jsonl"))
    if not any(event.get("kind") == "export" and event.get("map_sha256") == digest(doc)
               for event in export_events):
        raise SemanticDataError("blind_map_hash_mismatch")
    source_rows = [item["source_row"] for item in doc["items"]]
    if context is not None:
        if doc.get("construction_scope") != context["audit"]:
            raise SemanticDataError("blind_construction_scope_mismatch")
        _construction_members(source_rows, context, rows=True)
    elif "construction_scope" in doc or any(row["split"] in SEALED for row in source_rows):
        raise SemanticDataError("sealed_construction_scope_required")
    exported = []
    for name, expected in sorted(doc["shards"].items()):
        if Path(name).name != name or hashlib.sha256((blind_dir / name).read_bytes()).hexdigest() != expected:
            raise SemanticDataError("blind_shard_hash_mismatch")
        exported += load_rows(blind_dir / name)
    if exported != [item["entry"] for item in doc["items"]]:
        raise SemanticDataError("blind_map_content_mismatch")
    items = {item["entry"]["blind_id"]: item for item in doc["items"]}
    if len(items) != len(doc["items"]):
        raise SemanticDataError("duplicate_blind_id")
    prior = load_rows(ledger_path) if ledger_path.exists() else []
    seen_done = {event["fingerprint"]: event for event in prior if event.get("kind") == "record"}
    grouped = defaultdict(list)
    for raw in receipts:
        identifier = raw.get("blind_id") if type(raw) is dict else None
        grouped[identifier if type(identifier) is str else None].append(raw)
    accepted, quarantine, skipped = [], [], 0
    for identifier, replies in grouped.items():
        fp = digest({"map": digest(doc), "run": run_id, "replies": T._redacted_agent_value(replies)})
        if fp in seen_done:
            skipped += 1
            cached = seen_done[fp]
            if cached.get("ok"):
                accepted.append(validate_row(cached["row"]))
            else:
                quarantine.append({"code": cached["error_code"], "blind_id": identifier,
                                   "raw_redacted": T._redacted_agent_value(replies)})
            continue
        result, code = None, None
        try:
            if identifier not in items or len(replies) != 1:
                raise SemanticDataError("unknown_or_duplicate_blind_id")
            raw, item = replies[0], items[identifier]
            _nonsecret(raw)
            if set(raw) != {"blind_id", "input_sha256", "teacher", "judgement"} \
                    or raw["input_sha256"] != item["entry"]["input_sha256"] or raw["teacher"] != doc["teacher"]:
                raise SemanticDataError("receipt_identity_or_input_mismatch")
            if digest(item["source_row"]) != item["source_sha256"]:
                raise SemanticDataError("source_row_hash_mismatch")
            judgement = raw["judgement"]
            if type(judgement) is not dict or set(judgement) != {"label", "resolution_status", "reason", "nl_fidelity", "complete_semantics", "reconstruction"}:
                raise SemanticDataError("invalid_teacher_judgement")
            label, status = judgement["label"], judgement["resolution_status"]
            if status not in STATUSES or (label is not None and label not in S.SEMANTIC_RELATIONS):
                raise SemanticDataError("invalid_label")
            if status == "resolved" and (label is None or judgement["nl_fidelity"] is not True
                                         or judgement["complete_semantics"] is not True):
                raise SemanticDataError("false_fidelity_or_incomplete_clause_receipt")
            if status != "resolved" and label is not None:
                raise SemanticDataError("nonresolved_label_must_be_null")
            if type(judgement["reason"]) is not str or not judgement["reason"].strip():
                raise SemanticDataError("missing_teacher_reason")
            reconstruction = judgement["reconstruction"]
            if reconstruction is not None and oracle_certificate(reconstruction)["relation"] != label:
                raise SemanticDataError("teacher_reconstruction_mismatch")
            if item["source_row"].get("oracle") is not None and status == "resolved" and reconstruction is None:
                raise SemanticDataError("missing_blind_reconstruction")
            result = copy.deepcopy(item["source_row"])
            result["label"] = S.swap_relation(label) if label and item["swapped"] else label
            result["resolution_status"] = status
            result["label_origin"] = "blind_teacher_proxy"
            result["NL_fidelity_origin"] = {"kind": "blind_teacher_proxy", "independent": False, "human": False}
            result["adjudication_origin"] = {"kind": "pending_independent_adjudication", "independent": False, "human": False}
            result["teacher_evidence"] = {"teacher": doc["teacher"], "run_id": run_id,
                                           "prompt_sha256": doc["prompt_sha256"], "receipt_sha256": digest(raw),
                                           "judgement": judgement, "swapped": item["swapped"],
                                           "author_agreement": result["label"] == item["source_row"]["label"], **construction}
            if result.get("oracle") and result["label"] is not None and result["label"] != result["oracle_relation"]:
                raise SemanticDataError("teacher_author_oracle_disagreement")
            if label is None:
                result["phenomena"]["nonresolved_reason"] = judgement["reason"]
            validate_row(result)
            accepted.append(result)
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            code = getattr(exc, "code", "invalid_teacher_receipt")
            quarantine.append({"code": code, "blind_id": identifier,
                               "raw_redacted": T._redacted_agent_value(replies)})
        T._append_event(ledger_path, {"kind": "attempt", "fingerprint": fp, "run_id": run_id, "api_call": False, **construction})
        T._append_event(ledger_path, {"kind": "record", "fingerprint": fp, "blind_id": identifier,
                                     "ok": code is None, "error_code": code, "row": result if code is None else None, **construction})
    return {"accepted": accepted, "quarantine": quarantine, "received": len(receipts),
            "accepted_count": len(accepted), "quarantine_count": len(quarantine),
            "failed_by_code": dict(Counter(item["code"] for item in quarantine)),
            "expected": len(items), "missing": len(set(items) - set(grouped)), "skipped_done": skipped,
            "api_calls": 0, "human_gold": False, **construction}


def build_pilot(raw: dict) -> tuple[list[dict], dict]:
    if raw.get("author", {}).get("human") is not False or raw.get("author", {}).get("independent") is not False:
        raise SemanticDataError("unsupported_author_claim")
    rows = []
    for index, case in enumerate(raw["cases"], 1):
        oracle = case.get("oracle")
        certificate = oracle_certificate(oracle) if oracle is not None else None
        label = certificate["relation"] if certificate else case["proxy_label"]
        status = "resolved" if label else case["resolution_status"]
        root = f"pilot-v6-{index:03}"
        row = {"id": root + "-primary", "root_id": root, "cluster_id": root, "split": "pilot",
               "label_space": "semantic5", "label": label, "pair": copy.deepcopy(case["pair"]),
               "label_origin": "author_controlled_oracle" if oracle else "author_proxy" if label else "author_unresolved",
               "oracle_relation": certificate["relation"] if certificate else None,
               "NL_fidelity_origin": {"kind": "author_self_checked", "independent": False, "human": False},
               "adjudication_origin": {"kind": "none", "independent": False, "human": False},
               "resolution_status": status,
               "domain": case["domain"],
               "phenomena": {"family": case["family"], "language": "zh", "mode": case["mode"],
                              "scenario": case["scenario"], "skeleton": case["skeleton"],
                              "source_ids": case["source_ids"], "author": raw["author"],
                              "nonresolved_reason": case.get("nonresolved_reason"),
                              "cohort": "development_pilot_not_first_wave_900"}}
        if oracle:
            row["oracle"] = {"model": oracle, "certificate": certificate, "mapping_notes": case["mapping_notes"]}
        rows.append(validate_row(row))
    rows, audit = cluster_rows(rows, merge=True)
    for event in raw.get("actions", []):
        validate_action_receipt(event, event["expected_receipt"])
    audit["action_events"] = len(raw.get("actions", []))
    audit["human_gold"] = False
    audit["blind_reviews_received"] = 0
    audit["oracle_rows"] = sum(row.get("oracle") is not None for row in rows)
    audit["free_author_proxy_rows"] = sum(row.get("oracle") is None for row in rows)
    return rows, audit


def write_dataset(rows: list[dict], out_dir: object, *, audit: dict | None = None,
                  extra: dict | None = None, qualifications: list | None = None) -> dict:
    rows, checked = cluster_rows(rows)
    index = _qualification_index(qualifications)
    by_id = {row["id"]: row for row in rows}
    unknown = set(index) - set(by_id)
    if unknown:
        raise SemanticDataError("unknown_inverse_qualification_id")
    for identifier in index:
        target = by_id[identifier]
        if target["label"] is None or target["label_space"] != "semantic5":
            raise SemanticDataError("inapplicable_inverse_qualification")
    audit = {**(audit or {}), **checked, "supervision_rule": SUPERVISION_RULE,
             "inverse_qualification_format": INVERSE_QUALIFICATION_FORMAT,
             "inverse_qualifications": {
                 "records": len(index),
                 "given_sha256": digest(qualifications) if qualifications is not None else None,
                 "qualified_rows": sum(1 for row in rows if row["id"] in index)}}
    resolved = [row for row in rows if row["label"] is not None]
    nonresolved = [row for row in rows if row["label"] is None]
    views = [{"id": row["id"], "root_id": row["root_id"], "cluster_id": row["cluster_id"],
              "split": row["split"], "label_space": row["label_space"], "label": row["label"],
              "pair": student_pair(row),
              "supervision": supervision(row, qualification=index.get(row["id"]))} for row in resolved]
    payloads = {"all.jsonl": B._jsonl_bytes(rows), "resolved.jsonl": B._jsonl_bytes(resolved),
                "nonresolved.jsonl": B._jsonl_bytes(nonresolved), "student.jsonl": B._jsonl_bytes(views),
                "audit.json": B._json_bytes(audit)}
    for split in sorted({row["split"] for row in rows}):
        payloads[f"{split}.jsonl"] = B._jsonl_bytes([row for row in rows if row["split"] == split])
    payloads.update(extra or {})
    manifest = {"format": FORMAT, "status": "development_author_proxy_not_gold",
                "protocol_status": "unfrozen_draft_not_sha_bound", "protocol_sha256": None,
                "classes": list(S.SEMANTIC_RELATIONS), "parent_by_name": dict(S.SEMANTIC_TO_NLI),
                "reverse_by_name": dict(S.SWAPPED_RELATIONS), "contract_sha256": _contract_sha(),
                "model_input": "dataset_recommender.agent.semantic_relations.semantic_pair_text(pair)",
                "full_semantic_fields": sorted(SIDE_FIELDS), "proof_in_student": False,
                "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "job_spec_sha256": hashlib.sha256(JOB_SPEC.read_bytes()).hexdigest(),
                "shared_primitives_sha256": {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                             for module in (B, T)},
                "oriented_inputs": 2 * len(resolved),
                "unverified": ["independent_NL_fidelity", "tokenization_and_truncation", "human_adjudication"],
                "counts": audit, "human_gold": False, "api_calls": 0,
                "files": {name: {"sha256": hashlib.sha256(data).hexdigest(),
                                  "bytes": len(data), "rows": len(data.splitlines()) if name.endswith(".jsonl") else None}
                          for name, data in sorted(payloads.items())}}
    payloads["manifest.json"] = B._json_bytes(manifest)
    manifest_hash = hashlib.sha256(payloads["manifest.json"]).hexdigest()
    payloads["manifest.sha256"] = (manifest_hash + "  manifest.json\n").encode("ascii")
    B._write_new_directory(Path(out_dir), payloads)
    return {"manifest_sha256": manifest_hash, "counts": audit, "files": manifest["files"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--inverse-qualifications")
    p = sub.add_parser("preallocate")
    p.add_argument("--out", required=True)
    p = sub.add_parser("author-jobs")
    p.add_argument("--allocation", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("import-authors")
    p.add_argument("--input", required=True)
    p.add_argument("--allocation", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--inverse-qualifications")
    p = sub.add_parser("blind-jobs")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--map", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--swap-policy", choices=SWAP_POLICIES, default="canonical")
    p.add_argument("--swap-evidence")
    p = sub.add_parser("import-teacher")
    p.add_argument("--input", required=True)
    p.add_argument("--blind", required=True)
    p.add_argument("--map", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--inverse-qualifications")
    p = sub.add_parser("qualified-source")
    p.add_argument("--input", required=True)
    for command in ("author-jobs", "import-authors", "blind-jobs", "import-teacher", "qualified-source"):
        p = sub.choices[command]
        p.add_argument("--construction-scope", help="显式 sealed 构造声明 JSON；非密钥或认证")
        if command != "import-teacher":
            p.add_argument("--run-id")
    p = sub.add_parser("audit")
    p.add_argument("--input", required=True)
    args = parser.parse_args(argv)
    failed = False
    try:
        if args.command == "pilot":
            raw = read_json(args.input)
            rows, audit = build_pilot(raw)
            result = write_dataset(rows, args.out, audit=audit,
                                   extra={"actions.jsonl": B._jsonl_bytes(raw.get("actions", []))},
                                   qualifications=load_rows(args.inverse_qualifications)
                                   if args.inverse_qualifications else None)
        elif args.command == "preallocate":
            spec = read_json(JOB_SPEC)
            allocation = preallocate(spec)
            payloads = {"slots.jsonl": B._jsonl_bytes(allocation),
                        "author-jobs.jsonl": B._jsonl_bytes(author_jobs(allocation, spec))}
            result = {"planned_slots": len(allocation), "planned_clusters": 180, "accepted_roots": 0,
                      "counts_by_split": dict(Counter(item["split"] for item in allocation)),
                      "sealed_content_generated": False,
                      "files": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}}
            payloads["manifest.json"] = B._json_bytes(result)
            B._write_new_directory(Path(args.out), payloads)
        elif args.command == "author-jobs":
            context = _construction_context(args.construction_scope, args.run_id)
            jobs = author_jobs(load_rows(args.allocation), read_json(JOB_SPEC),
                               construction_scope=args.construction_scope, run_id=args.run_id)
            payloads = {"author-jobs.jsonl": B._jsonl_bytes(jobs)}
            result = {"author_jobs": len(jobs), "accepted_roots": 0, "api_calls": 0,
                      "files": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}}
            if context is not None:
                result["construction_scope"] = context["audit"]
            payloads["manifest.json"] = B._json_bytes(result)
            B._write_new_directory(Path(args.out), payloads)
        elif args.command == "import-authors":
            result = import_authors(load_rows(args.input), load_rows(args.allocation),
                                    construction_scope=args.construction_scope, run_id=args.run_id)
            failed = bool(result["quarantine"])
            write_dataset(result["accepted"], args.out, audit=result["audit"],
                          extra={"quarantine.jsonl": B._jsonl_bytes(result["quarantine"])},
                          qualifications=load_rows(args.inverse_qualifications)
                          if args.inverse_qualifications else None)
            result = {k: v for k, v in result.items() if k not in ("accepted", "quarantine")}
        elif args.command == "blind-jobs":
            result = export_blind(load_rows(args.input), args.out, args.map,
                                  teacher=read_json(args.teacher), source=args.input,
                                  swap_policy=args.swap_policy, construction_scope=args.construction_scope,
                                  run_id=args.run_id, swap_evidence=load_rows(args.swap_evidence)
                                  if args.swap_evidence else None)
        elif args.command == "import-teacher":
            result = import_teacher(load_rows(args.input), args.blind, args.map, args.ledger,
                                    run_id=args.run_id, construction_scope=args.construction_scope)
            failed = bool(result["quarantine"])
            construction = {"construction_scope": result["construction_scope"]} if "construction_scope" in result else None
            write_dataset(result["accepted"], args.out, audit=construction,
                          extra={"quarantine.jsonl": B._jsonl_bytes(result["quarantine"])},
                          qualifications=load_rows(args.inverse_qualifications)
                          if args.inverse_qualifications else None)
            result = {k: v for k, v in result.items() if k not in ("accepted", "quarantine")}
        elif args.command == "qualified-source":
            bundle = load_qualified_source(args.input, construction_scope=args.construction_scope, run_id=args.run_id)
            result = bundle["lineage"]
        else:
            _, result = cluster_rows(load_rows(args.input))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 2 if failed else 0
    except (ValueError, TypeError, KeyError, OSError, RecursionError) as exc:
        print(json.dumps({"ok": False, "code": getattr(exc, "code", "invalid_input_or_io")}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
