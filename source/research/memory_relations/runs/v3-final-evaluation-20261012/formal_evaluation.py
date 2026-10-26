"""一次冻结参考评价：真概率缓存复用于原adapter/M；所有结果仅proxy agreement。"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import copy
import gc
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import binding as B
from research.memory_relations import adapter as A
from research.memory_relations import evaluate as E
from research.memory_relations import features as F
from research.memory_relations import finetune as FT
from research.memory_relations import metrics as EM
from research.memory_relations import schema as S
from dataset_recommender.agent import memory_relations as M

CLASSES = tuple(F.CLASSES)
SEED = 20260919


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def save_rows(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def build_registry(expected, references, statuses, originals, *, expected_counts=None):
    expected_counts = expected_counts or {"heldout": 445, "ood": 427}
    registry = {}
    for entry in expected:
        identifier = str(entry["source_id"])
        if identifier in registry or entry["split"] not in expected_counts:
            raise ValueError("invalid_expected_registry")
        registry[identifier] = {"id": identifier, "split": entry["split"]}
    if dict(Counter(r["split"] for r in registry.values())) != expected_counts:
        raise ValueError("expected_split_count_mismatch")
    originals_by_id = {}
    for row in originals:
        identifier = str(row["id"])
        if identifier in originals_by_id:
            raise ValueError("duplicate_original_request")
        originals_by_id[identifier] = row
    if set(originals_by_id) != set(registry):
        raise ValueError("original_request_registry_mismatch")
    for row in references:
        identifier = str(row["id"])
        if identifier not in registry or "reference" in registry[identifier] or row["label"] not in CLASSES:
            raise ValueError("reference_registry_mismatch")
        if row["split"] != registry[identifier]["split"]:
            raise ValueError("reference_split_mismatch")
        if S.canonical_request(row["request"]) != S.canonical_request(originals_by_id[identifier]["request"]):
            raise ValueError("reference_request_changed")
        registry[identifier]["reference"] = row
        registry[identifier]["reference_status"] = "labelled_proxy"
    for row in statuses:
        identifier = str(row["source_id"])
        if (identifier not in registry or "reference_status" in registry[identifier]
                or row.get("reference_label") is not None or row.get("label") is not None):
            raise ValueError("unlabelled_status_has_truth_or_bad_registry")
        if row["split"] != registry[identifier]["split"] or row["status"] not in ("excluded", "unresolved"):
            raise ValueError("unlabelled_status_mismatch")
        if S.canonical_request(row["request"]) != S.canonical_request(originals_by_id[identifier]["request"]):
            raise ValueError("unlabelled_original_request_changed")
        registry[identifier]["reference_status"] = row["status"]
        registry[identifier]["reference_note"] = row.get("exclusion_reason")
    for identifier, entry in registry.items():
        if "reference_status" not in entry:
            raise ValueError("expected_request_unaccounted_for")
        original = originals_by_id[identifier]
        if original["split"] != entry["split"]:
            raise ValueError("original_split_mismatch")
        entry.update(request=copy.deepcopy(original["request"]), group_id=original["group_id"],
                     operating_primary=original.get("phenomena", {}).get("operating_primary"))
        if not isinstance(entry["group_id"], str) or not entry["group_id"]:
            raise ValueError("missing_root_id")
    return registry


def identity(binding):
    return B.canonical_hash({"classes": list(CLASSES), "temperature": binding["temperature"],
                            "protected_sha256": binding["protected_sha256"]})


class ProbabilityCache:
    def __init__(self, model_identity):
        self.model_identity = model_identity
        self.entries = {}

    def key(self, request):
        return B.canonical_hash({"model_identity": self.model_identity,
                                 "input_recipe": FT.INPUT_RECIPE,
                                 "joint_text": FT.joint_text(request)})

    def add(self, request, probabilities, *, expected_identity=None):
        if expected_identity is not None and expected_identity != self.model_identity:
            raise ValueError("cache_model_identity_mismatch")
        matrix = A._matrix([probabilities], 1, len(CLASSES))
        if matrix is None:
            raise ValueError("invalid_probability_matrix")
        key = self.key(request)
        if key in self.entries and not np.allclose(self.entries[key], matrix[0], atol=1e-6, rtol=0):
            raise ValueError("same_input_inconsistent_probabilities")
        self.entries.setdefault(key, matrix[0].copy())
        return key

    def get(self, request, *, expected_identity=None):
        if expected_identity is not None and expected_identity != self.model_identity:
            raise ValueError("cache_model_identity_mismatch")
        key = self.key(request)
        if key not in self.entries:
            raise KeyError("cache_miss_no_terminal_reprediction")
        return self.entries[key].copy()


class CacheScorer:
    classes = CLASSES

    def __init__(self, cache, *, scorer=None, ledger=None):
        self.cache = cache
        self.scorer = scorer
        self.ledger = ledger
        self.hits = self.misses = self.physical_pairs = self.physical_calls = 0
        self.phase = "terminal_cache_only"

    def predict_proba(self, requests):
        missing = []
        seen = set()
        for request in requests:
            key = self.cache.key(request)
            if key not in self.cache.entries and key not in seen:
                seen.add(key)
                missing.append(request)
        if missing:
            if self.scorer is None:
                raise KeyError("cache_miss_no_terminal_reprediction")
            self.misses += len(missing)
            output = physical_score(self.scorer, missing, self.phase, self.ledger)
            self.physical_calls += 1
            self.physical_pairs += len(missing)
            for request, probs in zip(missing, output):
                self.cache.add(request, probs)
        self.hits += len(requests) - len(missing)
        return np.asarray([self.cache.get(request) for request in requests]).reshape(len(requests), len(CLASSES))


def physical_score(scorer, requests, phase, ledger):
    started = time.perf_counter()
    record = {"phase": phase, "started_at": B.now(), "requested_predictions": len(requests),
              "input_feature_sha256": [B.canonical_hash(FT.joint_text(r)) for r in requests],
              "device": scorer.device, "inference_batch_size": scorer.batch_size,
              "planned_forward_batches": math.ceil(len(requests) / scorer.batch_size),
              "training": False, "completed_predictions": 0, "error_type": None}
    try:
        probabilities = scorer.predict_proba(requests)
        matrix = A._matrix(probabilities, len(requests), len(CLASSES))
        if matrix is None:
            raise ValueError("invalid_probability_matrix")
        record.update(completed_predictions=len(requests), shape=list(matrix.shape), finite=True,
                      min_probability=float(matrix.min()) if len(matrix) else None,
                      max_probability=float(matrix.max()) if len(matrix) else None,
                      max_row_sum_error=float(np.abs(matrix.sum(axis=1) - 1).max()) if len(matrix) else 0)
        return matrix
    except Exception as exc:
        record["error_type"] = type(exc).__name__
        raise
    finally:
        record["elapsed_ms"] = (time.perf_counter() - started) * 1000
        if ledger is not None:
            append_jsonl(ledger, record)


def gpu_snapshot():
    queries = {}
    for name, query in (("gpu", "--query-gpu=name,memory.total,memory.used,utilization.gpu"),
                        ("contexts", "--query-compute-apps=pid,process_name,used_gpu_memory")):
        done = subprocess.run(["nvidia-smi", query, "--format=csv,noheader"], capture_output=True, text=True)
        queries[name] = {"exit_code": done.returncode, "stdout": done.stdout.strip(),
                         "stderr": done.stderr.strip()}
    return {"at": B.now(), **queries}


def raw_record(entry, probabilities, failure=None):
    relation = CLASSES[int(np.argmax(probabilities))] if probabilities is not None else None
    return {"id": entry["id"], "split": entry["split"], "group_id": entry["group_id"],
            "reference_status": entry["reference_status"],
            "reference_label": entry.get("reference", {}).get("label"),
            "request_sha256": B.canonical_hash(entry["request"]),
            "feature_sha256": B.canonical_hash(FT.joint_text(entry["request"])),
            "classifier_relation": relation, "completed": probabilities is not None,
            "pair_relation": EM.pair_verdict(probabilities, entry["request"], 0.0) if probabilities is not None else None,
            "probabilities": probabilities.tolist() if probabilities is not None else None,
            "classes": list(CLASSES), "temperature": 0.980590113228623,
            "cause": failure, "physical_original_request_prediction": True}


def replay_unlabelled(entry, scorer):
    predictor = E.make_evaluation_predictor(scorer, 0.0)
    calls = []
    decision = None
    failure = None
    try:
        candidate, existing, evidence, cfg = S.materialize_request(entry["request"])
        def wrapped(request):
            started = time.perf_counter()
            call = {"raw": None, "failure": None}
            try:
                call["raw"] = predictor(request)
                return call["raw"]
            except Exception as exc:
                call["failure"] = type(exc).__name__
                raise
            finally:
                call["latency_ms"] = (time.perf_counter() - started) * 1000
                calls.append(call)
        decision = M.assess_relations(candidate, existing, evidence, cfg, predictor=wrapped)
    except Exception as exc:
        failure = type(exc).__name__
    completed = E._completed(decision, calls, predictor.classifier_trace)
    return {"row_id": entry["id"], "group_id": entry["group_id"], "label": None,
            "reference_status": entry["reference_status"], "reference_note": entry.get("reference_note"),
            "label_origin": None, "acceptable": None, "row_error": failure,
            "decision": decision, "raw_decision": E._raw_decision(calls, CLASSES),
            "completed": completed, "semantic_uncertain": bool(completed and decision["relation"] == "uncertain"),
            "operating_primary": entry["operating_primary"], "error": decision["error_code"] if decision else failure,
            "predictions": [c["raw"] for c in calls], "calls": len(calls),
            "latency_ms": [c["latency_ms"] for c in calls],
            "call_failures": [c["failure"] for c in calls if c["failure"]],
            "classifier_errors": [c["error"] for c in predictor.classifier_trace if c["error"]],
            "classifier_pair_attempts": sum(c["pair_count"] for c in predictor.classifier_trace),
            "classifier_observed": True,
            "pair_diagnostics": [{**p, "label": None} for c in predictor.classifier_trace for p in c["pairs"]],
            "refs_by_role": None, "oracle": {"available": False, "reason": "no_reference_label"},
            "primary_agree": None, "acceptable_agree": None, "assessment_failure": failure}


def class_stats(records, weights):
    known = [(r, float(w)) for r, w in zip(records, weights) if r.get("reference_label") in CLASSES]
    table = np.zeros((len(CLASSES), len(CLASSES) + 1))
    correct = completed = 0.0
    for row, weight in known:
        truth = CLASSES.index(row["reference_label"])
        relation = row.get("relation")
        column = CLASSES.index(relation) if row.get("completed") and relation in CLASSES else len(CLASSES)
        table[truth, column] += weight
        completed += weight * bool(row.get("completed"))
        correct += weight * bool(row.get("completed") and relation == row["reference_label"])
    support = table.sum(axis=1)
    true_positive = np.diag(table[:, :len(CLASSES)])
    predicted = table[:, :len(CLASSES)].sum(axis=0)
    precision = np.divide(true_positive, predicted, out=np.zeros(len(CLASSES)), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros(len(CLASSES)), where=support > 0)
    per_class_f1 = np.divide(2 * true_positive, support + predicted,
                             out=np.zeros(len(CLASSES)), where=support + predicted > 0)
    mass = float(support.sum())
    macro = float(per_class_f1.mean()) if mass else None
    if known and all(r.get("completed") and r.get("relation") in CLASSES for r, _ in known):
        macro = EM.macro_f1(np.asarray([CLASSES.index(r["reference_label"]) for r, _ in known]),
                            np.asarray([CLASSES.index(r["relation"]) for r, _ in known]),
                            sample_weight=np.asarray([w for _, w in known]))
    return {"reference_mass": mass, "reference_rows": len(known), "confirmed_correct_mass": correct,
            "completed_known_mass": completed, "agreement": correct / mass if mass else None,
            "macro_f1": macro, "classes": list(CLASSES), "outcomes": [*CLASSES, "system_failure"],
            "confusion": table.tolist(), "support": {c: float(n) for c, n in zip(CLASSES, support)},
            "per_class": {c: {"precision": float(p), "recall": float(r), "f1": float(f), "support": float(n)}
                          for c, p, r, f, n in zip(CLASSES, precision, recall, per_class_f1, support)}}


def view_report(records):
    groups = [r["group_id"] for r in records]
    known = [r for r in records if r.get("reference_label") in CLASSES]
    row = class_stats(records, np.ones(len(records)))
    equal = class_stats(known, EM._group_weights([r["group_id"] for r in known])) if known else class_stats([], [])
    intervals = {}
    known_groups = [r["group_id"] for r in known]
    for name in ("agreement", "macro_f1"):
        intervals[name] = EM.cluster_bootstrap(
            known_groups, lambda index, weights, name=name: class_stats(
                [known[int(i)] for i in index], weights)[name], weighting="group_equal", seed=SEED,
            numerator=equal["confirmed_correct_mass"] if name == "agreement" else None,
            denominator=equal["reference_mass"])
    completed = sum(bool(r["completed"]) for r in records)
    confirmed = int(row["confirmed_correct_mass"])
    return {"scope": "proxy_agreement_known_reference_all_known_including_failure",
            "n_planned": len(records), "n_reference": len(known), "reference_roots": len(set(known_groups)),
            "completion": E._rate_record(completed, len(records)),
            "confirmed_correct_overall": E._rate_record(confirmed, len(records)),
            "reference_coverage": E._rate_record(len(known), len(records)),
            "unlabelled_completed": sum(bool(r["completed"]) and r.get("reference_label") is None for r in records),
            "unlabelled_correctness": "not_assessable_not_wrong_not_semantic_unknown",
            "root_equal_primary_known_reference": {**equal, "bootstrap": intervals},
            "row_secondary_known_reference": row,
            "all_attempt_root_intervals": {
                name: EM.cluster_bootstrap(groups, lambda index, weights, name=name: (
                    float(np.average([bool(records[int(i)]["completed"]) for i in index], weights=weights))
                    if name == "completion" else float(np.average([
                        bool(records[int(i)]["completed"]) and records[int(i)].get("reference_label") in CLASSES
                        and records[int(i)].get("relation") == records[int(i)].get("reference_label") for i in index], weights=weights))),
                    weighting="group_equal", seed=SEED)
                for name in ("completion", "confirmed_correct_overall")}}


def refs_root_report(records):
    reports = {}
    for source in ("raw", "gated"):
        reports[source] = {}
        for role in ("related_ids", "evidence_refs"):
            subset = [r for r in records if r.get("refs_by_role") and r["refs_by_role"].get(source)]
            groups = [r["group_id"] for r in subset]
            def counts(index, weights, metric):
                hits = sum(float(w) * subset[int(i)]["refs_by_role"][source][role]["hits"] for i, w in zip(index, weights))
                field = "predicted" if metric == "precision" else "expected"
                denominator = sum(float(w) * len(subset[int(i)]["refs_by_role"][source][role][field]) for i, w in zip(index, weights))
                return hits / denominator if denominator else None
            reports[source][role] = {metric: EM.cluster_bootstrap(
                groups, lambda index, weights, metric=metric: counts(index, weights, metric),
                weighting="group_equal", seed=SEED) for metric in ("precision", "recall")}
    return reports


def terminal_reports(registry, raw_rows, replayed):
    raw_by_id = {r["id"]: r for r in raw_rows}
    replay_by_id = {str(r["row_id"]): r for r in replayed}
    if set(raw_by_id) != set(registry) or set(replay_by_id) != set(registry):
        raise ValueError("predictions_do_not_cover_expected_registry")
    result = {"schema": "locked-terminal-proxy-agreement-v1", "human_gold": False,
              "production_safe": False, "risk_certificate": False, "model_reselectable": False,
              "policy_threshold": None, "diagnostic_threshold": 0.0,
              "uncertain_reference_rows": sum(r.get("reference", {}).get("label") == "uncertain" for r in registry.values()),
              "bootstrap": {"engine": "research.memory_relations.metrics.cluster_bootstrap",
                            "samples": EM.BOOTSTRAP_SAMPLES, "confidence": EM.CONFIDENCE,
                            "resampling_unit": "root", "rows_are_not_independent_roots": True}, "splits": {}}
    for split in ("heldout", "ood", "all"):
        entries = [r for r in registry.values() if split == "all" or r["split"] == split]
        views = {name: [] for name in ("classifier", "pair_gate", "adapter_raw", "production_gated")}
        records = []
        for entry in entries:
            raw, replay = raw_by_id[entry["id"]], replay_by_id[entry["id"]]
            records.append(replay)
            common = {"id": entry["id"], "group_id": entry["group_id"],
                      "reference_label": entry.get("reference", {}).get("label")}
            for name, field in (("classifier", "classifier_relation"), ("pair_gate", "pair_relation")):
                views[name].append({**common, "relation": raw[field], "completed": raw["completed"]})
            for name, field in (("adapter_raw", "raw_decision"), ("production_gated", "decision")):
                decision = replay[field]
                views[name].append({**common, "relation": decision["relation"] if decision else None,
                                    "completed": replay["completed"] if field == "decision" else decision is not None})
        extended = E._extended_summary(records, EM, CLASSES, EM.ALLOWING_CLASSES, EM.BLOCKING_CLASSES)
        result["splits"][split] = {"expected": len(entries), "roots": len({r["group_id"] for r in entries}),
                                    "views": {name: view_report(rows) for name, rows in views.items()},
                                    "semantic_uncertain": sum(r["semantic_uncertain"] for r in records),
                                    "system_failure": sum(not r["completed"] for r in records),
                                    "failure_causes": dict(Counter(r.get("row_error") or r.get("error") or "unknown_failure"
                                                                  for r in records if not r["completed"])),
                                    "all_attempt": extended["all_attempt"], "group_equal": extended["group_equal"],
                                    "refs_by_role": extended["refs_by_role"], "refs_root_equal": refs_root_report(records),
                                    "raw_set": extended["raw_set"], "gated_set": extended["gated_set"],
                                    "observed_pair_diagnostics": extended["classifier"]}
    return result


def structural_fixture(count, *, long=False):
    text = "在公开合成示例中先写摘要，再写数据来源。"
    candidate = {"text": text, "trigger": "公开合成示例", "evidence": [text],
                 "tier": "explicit", "permanence": "persistent", "scope": "project"}
    existing = [{"id": f"fixture-rule-{i}", "text": ("公开结构示例。" * 180 if long else text),
                 "trigger": "公开合成示例", "evidence": [], "scope": "same",
                 "tier": "explicit", "permanence": "persistent"} for i in range(count)]
    return {"candidate": candidate, "existing": existing, "evidence": [{"id": "fixture-source", "text": text}],
            "context": {"scope": "project"}}


def structural_run(cache_scorer):
    reports = []
    for count, long in [(n, False) for n in (0, 1, 8, 64, 65, 129)] + [(64, True)]:
        request = structural_fixture(count, long=long)
        candidate, existing, evidence, cfg = S.materialize_request(request)
        predictor = E.make_evaluation_predictor(cache_scorer, 0.0)
        batches = []
        def wrapped(projected):
            raw = predictor(projected)
            batches.append({"rules": len(projected["existing"]),
                            "json_characters": len(json.dumps(projected, ensure_ascii=False)),
                            "utf8_bytes": len(json.dumps(projected, ensure_ascii=False).encode("utf-8")),
                            "context_batch": projected["context"].get("batch"), "raw": raw})
            return raw
        decision = M.assess_relations(candidate, existing, evidence, cfg, predictor=wrapped)
        reports.append({"fixture": True, "human_gold": False, "semantic_reference": None,
                        "rules": count, "long_rules": long, "batches": batches, "decision": decision,
                        "observed_pairs": predictor.stats["pairs"],
                        "complete_scoring": predictor.stats["pairs"] == max(1, count)})
    if not all(r["complete_scoring"] for r in reports):
        raise ValueError("structural_complete_scoring_failed")
    return {"rows": reports, "note": "公开合成结构fixture，仅测真实模型/闸的分批，不当人工gold或新终测。"
            "当前生产_MAX_REQUEST比较json字符数而非UTF8字节，双方均记录；单条/证据超限不由切批治愈。"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--diagnostic", action="store_true", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--oracle-aggregation", action="store_true", required=True)
    args = parser.parse_args(argv)
    if args.threshold != 0.0:
        parser.error("locked_no_policy_only_explicit_tau0_diagnostic")
    binding = B.verify()
    B.save_json(B.BASE / "execution-started.json", {"at": B.now(), "argv": sys.argv,
                "single_attempt": True, "binding_sha256": B.canonical_hash(binding),
                "runner_sha256": B.sha(__file__), "binding_script_sha256": B.sha(B.__file__)})
    reports_dir = B.BASE / "reports"
    reports_dir.mkdir(exist_ok=True)
    physical_ledger = reports_dir / "physical-prediction-calls.jsonl"
    started = time.perf_counter()
    status = {"started_at": B.now(), "status": "running", "training": False, "device_requested": args.device}
    try:
        manifest = B.read_json(B.REF / "ref-ready-manifest.json")
        references = S.load_rows(B.REF / "final-heldout.jsonl") + S.load_rows(B.REF / "final-ood.jsonl")
        originals = []
        for info in manifest["version_binding"]["sealed"].values():
            originals.extend(B.read_jsonl(info["path"]))
        registry = build_registry(B.read_jsonl(B.REF / "expected-cases.jsonl"), references,
                                  B.read_jsonl(B.REF / "reference_status.jsonl"), originals)
        if Counter(entry.get("reference", {}).get("label") for entry in registry.values())["uncertain"] != 98:
            raise ValueError("uncertain_reference_count_changed")
        B.save_json(reports_dir / "expected-registry-check.json", {
            "expected": len(registry), "reference": len(references), "unlabelled": 1,
            "splits": {split: {"expected": sum(r["split"] == split for r in registry.values()),
                               "roots": len({r["group_id"] for r in registry.values() if r["split"] == split})}
                       for split in ("heldout", "ood")}, "original_requests_exactly_registered": True,
            "original_label_used_as_gold": False, "label_counts": dict(Counter(r["label"] for r in references))})
        import torch
        torch.set_num_threads(6)
        resources = {"before": gpu_snapshot(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                     "python": sys.version, "executable": sys.executable,
                     "current_pid": __import__("os").getpid(), "cuda_available": torch.cuda.is_available(),
                     "shared_idle_context_note": "Pre-existing contexts left intact; coordinator reserves CUDA for this inference only; CPU probe actor does not use GPU. No training/optimizer."}
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("requested_cuda_unavailable_no_silent_fallback")
        if args.device.startswith("cuda"):
            resources["device_name"] = torch.cuda.get_device_name(0)
            torch.cuda.reset_peak_memory_stats(0)
        scorer_started = time.perf_counter()
        scorer = F.load_scorer(B.MODEL, device=args.device)
        resources.update(loader_seconds=time.perf_counter() - scorer_started, loaded_device=scorer.device,
                         kind=scorer.kind, hidden_size=scorer.model.classifier.in_features,
                         classes=list(scorer.classes), max_length=scorer.max_length, temperature=scorer.temperature,
                         threshold=scorer.threshold, threshold_feasible=scorer.threshold_feasible,
                         parameter_count=sum(p.numel() for p in scorer.model.parameters()), model_training=scorer.model.training)
        if tuple(scorer.classes) != CLASSES or scorer.temperature != binding["temperature"] or scorer.model.training:
            raise ValueError("loaded_model_contract_mismatch")
        resources["after_loader"] = gpu_snapshot()
        B.save_json(reports_dir / "runtime-start.json", resources)
        cache = ProbabilityCache(identity(binding))
        entries = list(registry.values())
        raw_rows = []
        probabilities_path = reports_dir / "terminal-probabilities.jsonl"
        for start in range(0, len(entries), 32):
            chunk = entries[start:start + 32]
            requests = [entry["request"] for entry in chunk]
            try:
                output = physical_score(scorer, requests, "terminal_original_requests", physical_ledger)
                for entry, request, probs in zip(chunk, requests, output):
                    key = cache.add(request, probs)
                    record = {**raw_record(entry, probs), "model_identity": cache.model_identity, "cache_key": key}
                    append_jsonl(probabilities_path, record)
                    raw_rows.append(record)
            except Exception as exc:
                for entry in chunk:
                    record = {**raw_record(entry, None, type(exc).__name__), "model_identity": cache.model_identity}
                    append_jsonl(probabilities_path, record)
                    raw_rows.append(record)
            print(json.dumps({"phase": "terminal_original_requests", "attempted": min(start + 32, len(entries)),
                              "planned": len(entries)}), flush=True)
        resources["after_terminal_prediction"] = gpu_snapshot()
        replay_scorer = CacheScorer(cache)
        replayed = []
        for split in ("heldout", "ood"):
            split_rows = [r for r in references if r["split"] == split]
            predictor = E.make_evaluation_predictor(replay_scorer, 0.0)
            evaluation = E.evaluate_sets(predictor, split_rows, oracle_aggregation=True)
            replayed.extend(evaluation["rows"])
            save_rows(reports_dir / f"{split}.predictions.jsonl", evaluation["rows"])
            B.save_json(reports_dir / f"{split}-evaluate-summary.json", {
                "summary": evaluation["summary"], "semantics": evaluation["semantics"],
                "rows_file": f"{split}.predictions.jsonl", "operation": E._operating_threshold(0.0, True, B.MODEL),
                "probabilities_reused": True, "physical_predictions_here": 0})
        for entry in entries:
            if "reference" not in entry:
                record = replay_unlabelled(entry, replay_scorer)
                replayed.append(record)
                save_rows(reports_dir / "unlabelled.predictions.jsonl", [record])
        terminal = terminal_reports(registry, raw_rows, replayed)
        terminal["cache"] = {"model_identity": cache.model_identity, "entries": len(cache.entries),
                             "replay_hits": replay_scorer.hits, "replay_misses": replay_scorer.misses,
                             "replay_physical_predictions": replay_scorer.physical_pairs,
                             "original_request_rows_attempted": len(raw_rows),
                             "original_request_rows_completed": sum(r["completed"] for r in raw_rows)}
        B.save_json(reports_dir / "terminal-result.json", terminal)
        mistakes = []
        for entry in entries:
            reference = entry.get("reference")
            raw = next(r for r in raw_rows if r["id"] == entry["id"])
            replay = next(r for r in replayed if str(r["row_id"]) == entry["id"])
            if reference and (raw["classifier_relation"] != reference["label"] or not replay["completed"]
                              or replay["decision"]["relation"] != reference["label"]):
                mistakes.append({"id": entry["id"], "split": entry["split"], "group_id": entry["group_id"],
                                 "reference_label": reference["label"], "classifier": raw["classifier_relation"],
                                 "pair_gate": raw["pair_relation"], "gated": replay["decision"],
                                 "completed": replay["completed"], "candidate": entry["request"]["candidate"],
                                 "existing": entry["request"]["existing"], "evidence": entry["request"]["evidence"],
                                 "related_reference": reference["related_ids"], "evidence_reference": reference["evidence_refs"]})
                if len(mistakes) >= 10:
                    break
        B.save_json(reports_dir / "bounded-case-explanations.json", {"limit": 10, "cases": mistakes})
        probe = [structural_fixture(1), structural_fixture(0)]
        reloaded = F.load_scorer(B.MODEL, device="cpu")
        before = physical_score(scorer, probe, "public_reload_probe_original_device", physical_ledger)
        after = physical_score(reloaded, probe, "public_reload_probe_cpu", physical_ledger)
        difference = float(np.abs(before - after).max())
        reload_ok = bool(np.array_equal(before.argmax(axis=1), after.argmax(axis=1)) and difference <= 1e-6
                         and scorer.temperature == reloaded.temperature and scorer.threshold == reloaded.threshold
                         and scorer.threshold_feasible == reloaded.threshold_feasible)
        B.save_json(reports_dir / "public-reload-parity.json", {
            "ok": reload_ok, "public_loader": "features.load_scorer", "max_abs_diff": difference,
            "argmax_match": bool(np.array_equal(before.argmax(axis=1), after.argmax(axis=1))),
            "probe_role": "public_synthetic_structure_not_new_terminal", "physical_predictions": 4,
            "original_device": args.device, "reload_device": "cpu", "tolerance": 1e-6})
        del reloaded
        gc.collect()
        if not reload_ok:
            raise ValueError("public_reload_parity_failed")
        B.verify()
        diagnostic_scorer = CacheScorer(cache, scorer=scorer, ledger=physical_ledger)
        diagnostic_scorer.phase = "old_exposed_set_pair_inputs"
        old_rows = S.load_rows(B.OLD)
        old_predictor = E.make_evaluation_predictor(diagnostic_scorer, 0.0)
        old = E.evaluate_sets(old_predictor, old_rows, oracle_aggregation=True)
        save_rows(reports_dir / "old-exposed-set.predictions.jsonl", old["rows"])
        old_views = []
        for row in old["rows"]:
            old_views.append({"group_id": row["group_id"], "reference_label": row["label"],
                              "relation": row["decision"]["relation"] if row["decision"] else None,
                              "completed": row["completed"]})
        B.save_json(reports_dir / "old-exposed-set-result.json", {
            "input_role": "old_exposed_synthetic_diagnostic_not_unseen_terminal", "human_gold": False,
            "data_sha256": B.sha(B.OLD), "model_dir": str(B.MODEL), "weights_sha256": B.sha(B.MODEL / "model.pt"),
            "meta_source": "original_locked_candidate_not_single_audit_export", "operation": E._operating_threshold(0.0, True, B.MODEL),
            "summary": old["summary"], "semantics": old["semantics"], "known_set_views": view_report(old_views),
            "refs_root_equal": refs_root_report(old["rows"]), "rule_count_distribution": dict(Counter(len(r["request"]["existing"]) for r in old_rows)),
            "observed_batch_count_distribution": dict(Counter(r["calls"] for r in old["rows"])),
            "physical_predictions": diagnostic_scorer.physical_pairs, "physical_calls": diagnostic_scorer.physical_calls,
            "cache_hits": diagnostic_scorer.hits, "oracle_role": "complete_proxy_pair_labels_not_human_gold"})
        diagnostic_scorer.phase = "public_structural_boundary_fixture"
        structural = structural_run(diagnostic_scorer)
        B.save_json(reports_dir / "structural-boundary-result.json", structural)
        save_rows(reports_dir / "cache-entries.jsonl", ({"model_identity": cache.model_identity,
                  "cache_key": key, "classes": list(CLASSES), "probabilities": value.tolist()}
                  for key, value in cache.entries.items()))
        resources["after_all_prediction"] = gpu_snapshot()
        if args.device.startswith("cuda"):
            resources.update(peak_allocated_mb=torch.cuda.max_memory_allocated(0) / 2**20,
                             peak_reserved_mb=torch.cuda.max_memory_reserved(0) / 2**20)
        B.save_json(reports_dir / "runtime-final.json", resources)
        cli = [sys.executable, "-B", "-m", "research.memory_relations.evaluate", "--model-dir", str(B.MODEL),
               "--data", str(B.OLD), "--out", str(reports_dir / "unavailable-policy-cli-output.json")]
        done = subprocess.run(cli, cwd=B.ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        B.save_json(reports_dir / "policy-cli-rejection.json", {"argv": cli, "returncode": done.returncode,
                    "stdout": done.stdout, "stderr": done.stderr,
                    "output_exists": (reports_dir / "unavailable-policy-cli-output.json").exists(),
                    "expected_rejection": done.returncode == 2, "policy_threshold": None})
        if done.returncode != 2:
            raise ValueError("default_policy_cli_did_not_reject_null_policy")
        B.verify()
        physical = B.read_jsonl(physical_ledger)
        physical_counts = {}
        for phase in sorted({r["phase"] for r in physical}):
            phase_rows = [r for r in physical if r["phase"] == phase]
            physical_counts[phase] = {"calls": len(phase_rows),
                                      "requested_predictions": sum(r["requested_predictions"] for r in phase_rows),
                                      "completed_predictions": sum(r["completed_predictions"] for r in phase_rows),
                                      "forward_batches": sum(r["planned_forward_batches"] for r in phase_rows),
                                      "failures": sum(r["error_type"] is not None for r in phase_rows)}
        status.update(status="completed", ended_at=B.now(), elapsed_seconds=time.perf_counter() - started,
                      unchanged_protected_hashes=True, protected_file_count=len(binding["protected_sha256"]),
                      physical_predictions=physical_counts, terminal_result_file="reports/terminal-result.json",
                      old_set_result_file="reports/old-exposed-set-result.json", no_model_selection_after_scores=True)
        B.save_json(reports_dir / "execution-result.json", status)
        print(json.dumps({"status": status["status"], "expected": len(registry),
                          "references": len(references), "physical_predictions": physical_counts}), flush=True)
        return 0
    except Exception as exc:
        status.update(status="failed_attempt_retained", ended_at=B.now(), elapsed_seconds=time.perf_counter() - started,
                      error_type=type(exc).__name__, error_code=str(exc) if len(str(exc)) <= 160 else "see_local_failure_trace")
        B.save_json(reports_dir / "execution-failure.json", status)
        import traceback
        with (reports_dir / "execution-failure-trace.log").open("x", encoding="utf-8") as handle:
            traceback.print_exc(file=handle)
        print(json.dumps({"status": status["status"], "error_type": type(exc).__name__, "error_code": status["error_code"]}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
