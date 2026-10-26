"""五叶研究评测的薄编排；只消费qualified来源、公共双向scorer及metrics数学锚点。

开发默认无scope，封存评测须先核显式确认与不可变候选/来源/统计锁，再交数据loader。
锁与construction scope仅是本机操作声明，不是ACL；逐题输出须由获准的独立评测者保管。
不走七类action评分、不拟温度或动作阈值、不把合成测试或代理标签当风险认证。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dataset_recommender.agent import semantic_relations as SR
from research.memory_relations import metrics as M
from research.memory_relations import semantic_data as D
from research.memory_relations import semantic_model as SM

FORMAT = "semantic-fine-evaluation-v1"
LOCK_FORMAT = "semantic-fine-evaluation-lock-v1"
STATISTICS = {"bootstrap_samples": 2000, "confidence": .95, "seed": 20261014,
              "weighting": "row", "unit": "one_canonical_per_root", "n_bins": 10,
              "temperature": 1.0, "temperature_status": "fixed_uncalibrated"}
TECHNICAL = frozenset({"input_failure", "inference_failure", "timeout", "missing_probability",
                       "invalid_shape", "nonfinite", "invalid_simplex"})


class SemanticEvaluationError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _splits(mode, splits):
    values = sorted(splits)
    if not values or len(values) != len(set(values)):
        raise SemanticEvaluationError("explicit_unique_splits_required")
    if mode == "development":
        if values != ["dev"]:
            raise SemanticEvaluationError("development_requires_dev")
    elif mode != "final" or not set(values) <= D.SEALED:
        raise SemanticEvaluationError("invalid_evaluation_phase")
    return values


IDENTITY_FORMAT = "semantic-evaluation-identity-v1"
COHORT_FORMAT = "semantic-evaluation-cohort-v1"


def _model_identity(directory, identity_index=None):
    from research.memory_relations import semantic_artifacts as A

    try:
        measured = A.identity(directory)
    except A.ArtifactError as error:
        raise SemanticEvaluationError(error.code) from None
    if identity_index is not None:
        document = D.read_json(identity_index)
        if document.get("schema") != IDENTITY_FORMAT or type(document.get("model")) is not dict:
            raise SemanticEvaluationError("invalid_model_identity_index")
        if document["model"] != measured:
            raise SemanticEvaluationError("model_identity_index_drift")
        measured["identity_index_sha256"] = SM.FT.file_sha256(identity_index)
    return measured


def _source_specs(source_manifest, cohort, *, mode, allocation=None, construction_scope=None, run_id=None):
    if cohort is None:
        if source_manifest is None:
            raise SemanticEvaluationError("source_or_cohort_required")
        specs = [{"source_id": "source", "source_manifest": source_manifest, "allocation": allocation,
                  "construction_scope": construction_scope, "run_id": run_id}]
    else:
        if any(value is not None for value in (source_manifest, allocation, construction_scope, run_id)):
            raise SemanticEvaluationError("cohort_and_single_source_arguments_conflict")
        path = Path(cohort)
        document = D.read_json(path)
        if document.get("schema") != COHORT_FORMAT or type(document.get("sources")) is not list or not document["sources"]:
            raise SemanticEvaluationError("invalid_evaluation_cohort")
        specs = []
        for entry in document["sources"]:
            if type(entry) is not dict or not {"source_id", "source_manifest"} <= set(entry) \
                    or set(entry) - {"source_id", "source_manifest", "allocation", "construction_scope", "run_id"}:
                raise SemanticEvaluationError("invalid_cohort_source")
            item = {key: entry.get(key) for key in ("source_id", "source_manifest", "allocation", "construction_scope", "run_id")}
            for key in ("source_manifest", "allocation", "construction_scope"):
                if item[key] is not None:
                    member = Path(item[key])
                    item[key] = member if member.is_absolute() else path.parent / member
            specs.append(item)
    ids = [item["source_id"] for item in specs]
    if any(type(value) is not str or not value.strip() for value in ids) or len(ids) != len(set(ids)):
        raise SemanticEvaluationError("duplicate_or_invalid_cohort_source_id")
    for item in specs:
        if mode == "development" and (item["construction_scope"] is not None or item["run_id"] is not None):
            raise SemanticEvaluationError("development_scope_forbidden")
        if mode == "final" and (item["construction_scope"] is None or not item["run_id"] or item["allocation"] is not None):
            raise SemanticEvaluationError("final_scope_run_required")
    return sorted(specs, key=lambda item: item["source_id"])


def _evaluation_binding(model_dir, specs, *, mode, splits, cohort, identity_index):
    sources = [{"source_id": item["source_id"], "source_manifest_sha256": SM.FT.file_sha256(item["source_manifest"]),
                "allocation_sha256": SM.FT.file_sha256(item["allocation"]) if item["allocation"] is not None else None,
                "scope_file_sha256": SM.FT.file_sha256(item["construction_scope"]) if item["construction_scope"] is not None else None,
                "run_id": item["run_id"]} for item in specs]
    single = sources[0] if cohort is None else {}
    return {"model": _model_identity(model_dir, identity_index), "scientific_sources": SM.source_binding(),
            "consumer_sources": {"semantic_evaluate": SM.FT.file_sha256(__file__), "metrics": SM.FT.file_sha256(M.__file__)},
            **{key: single.get(key) for key in ("source_manifest_sha256", "allocation_sha256", "scope_file_sha256", "run_id")},
            "sources": sources, "cohort_sha256": SM.FT.file_sha256(cohort) if cohort is not None else None,
            "mode": mode, "splits": list(splits), "statistics": dict(STATISTICS)}


def evaluation_binding(model_dir, source_manifest, *, mode, splits, allocation=None,
                       construction_scope=None, run_id=None, cohort=None, identity_index=None):
    """绑定完整来源集合；不打开来源assets，不加载PT或调用construction loader。"""
    selected = _splits(mode, splits)
    specs = _source_specs(source_manifest, cohort, mode=mode, allocation=allocation,
                          construction_scope=construction_scope, run_id=run_id)
    return _evaluation_binding(model_dir, specs, mode=mode, splits=selected, cohort=cohort, identity_index=identity_index)


def write_final_lock(model_dir, source_manifest, lock_path, *, splits, construction_scope=None, run_id=None,
                     cohort=None, identity_index=None):
    binding = evaluation_binding(model_dir, source_manifest, mode="final", splits=splits,
                                 construction_scope=construction_scope, run_id=run_id, cohort=cohort, identity_index=identity_index)
    document = {"schema": LOCK_FORMAT, "binding": binding, "binding_sha256": D.digest(binding)}
    _write_json(Path(lock_path), document)
    return {"lock": str(Path(lock_path)), "binding_sha256": document["binding_sha256"]}


def _check_final_lock(lock_path, expected):
    lock = D.read_json(lock_path)
    if lock.get("schema") != LOCK_FORMAT or lock.get("binding_sha256") != D.digest(lock.get("binding")):
        raise SemanticEvaluationError("invalid_final_lock")
    if lock["binding"] != expected:
        raise SemanticEvaluationError("final_lock_mismatch")


def verify_final_lock(lock_path, *, confirm_final, model_dir, source_manifest, splits,
                      construction_scope=None, run_id=None, cohort=None, identity_index=None):
    if not confirm_final or lock_path is None:
        raise SemanticEvaluationError("final_confirmation_and_lock_required")
    expected = evaluation_binding(model_dir, source_manifest, mode="final", splits=splits,
                                  construction_scope=construction_scope, run_id=run_id, cohort=cohort, identity_index=identity_index)
    _check_final_lock(lock_path, expected)
    return expected


def _load_scorer(model_dir, identity, device):
    from research.memory_relations import semantic_artifacts as A

    scorer, meta, payload = A.load_inference_artifact(model_dir, device=device)
    if meta["schema"] != identity["schema"] or meta["train_sha256"] != identity["certificate_sha256"] \
            or scorer.arm != identity["arm"] or scorer.max_length != identity["input"]["max_length"]:
        raise SemanticEvaluationError("loaded_model_binding_mismatch")
    del meta, payload
    return scorer


def _source_rows(source_manifest, *, mode, splits, allocation, construction_scope, run_id, allow_empty=False):
    manifest = D.read_json(source_manifest)
    if mode == "development" and "construction_scope" in manifest:
        raise SemanticEvaluationError("sealed_source_in_development")
    bundle = D.load_qualified_source(source_manifest, construction_scope=construction_scope, run_id=run_id)
    rows = bundle["rows"]
    allocation_lineage = None
    if allocation is not None:
        rows, allocation_lineage = SM.qualified_stage_rows(source_manifest, allocation, bundle=bundle)
    if mode == "development" and any(row["split"] in D.SEALED for row in rows):
        raise SemanticEvaluationError("sealed_source_in_development")
    selected = [row for row in rows if row["split"] in splits]
    if not selected and not allow_empty:
        raise SemanticEvaluationError("empty_selected_source")
    roots = [row["root_id"] for row in selected]
    if len(roots) != len(set(roots)):
        raise SemanticEvaluationError("one_canonical_per_root_required")
    cluster_splits = defaultdict(set)
    for row in rows:
        cluster_splits[row["cluster_id"]].add(row["split"])
    if any(len(values) > 1 for values in cluster_splits.values()):
        raise SemanticEvaluationError("cluster_crosses_splits")
    return selected, dict(bundle, evaluation_cluster_splits=cluster_splits,
                           evaluation_ids={row["id"] for row in rows},
                           evaluation_roots={row["root_id"] for row in rows}), allocation_lineage


def _cohort_rows(specs, *, mode, splits, allow_empty_sources=False):
    combined, lineages, ids, roots = [], [], set(), set()
    cluster_splits = defaultdict(set)
    for spec in specs:
        rows, bundle, allocation_lineage = _source_rows(spec["source_manifest"], mode=mode, splits=splits,
                                                       allocation=spec["allocation"], construction_scope=spec["construction_scope"],
                                                       run_id=spec["run_id"], allow_empty=allow_empty_sources)
        if ids & bundle["evaluation_ids"]:
            raise SemanticEvaluationError("duplicate_cohort_canonical_id")
        if roots & bundle["evaluation_roots"]:
            raise SemanticEvaluationError("duplicate_cohort_root")
        ids.update(bundle["evaluation_ids"])
        roots.update(bundle["evaluation_roots"])
        for cluster, partitions in bundle["evaluation_cluster_splits"].items():
            cluster_splits[cluster].update(partitions)
        for row in rows:
            combined.append((spec["source_id"], row, bundle))
        lineages.append({"source_id": spec["source_id"], "qualified_lineage": bundle["lineage"],
                         "allocation_lineage": allocation_lineage, "selected_roots": len(rows)})
    if any(len(values) > 1 for values in cluster_splits.values()):
        raise SemanticEvaluationError("cohort_cluster_crosses_splits")
    if not combined:
        raise SemanticEvaluationError("empty_selected_source")
    return combined, lineages


def legal_dyads():
    """父投影与交换关系只来自SR；硬预测不在合法表上重新argmax。"""
    return {(SR.SEMANTIC_TO_NLI[leaf], SR.SEMANTIC_TO_NLI[SR.swap_relation(leaf)]): leaf
            for leaf in SR.SEMANTIC_RELATIONS}


def _probabilities(value, size):
    status, row = M.probability_row(value, size)
    return {"status": status, "probabilities": row.tolist() if row is not None else None}


def decode_both(probabilities, arm):
    if probabilities is None:
        return {direction: {"status": "abstain", "prediction": None, "probabilities": None}
                for direction in ("forward", "reverse")}
    if not isinstance(probabilities, dict):
        probabilities = {}
    checked = {}
    for direction in ("forward", "reverse"):
        value = probabilities.get(direction)
        if value is not None:
            value = np.asarray(value)
            if value.ndim == 2 and len(value) == 1:
                value = value[0]
        checked[direction] = _probabilities(value, len(SM.ARM_CLASSES[arm]))
    if arm == "c5":
        return {direction: {**item, "prediction": SR.SEMANTIC_RELATIONS[int(np.argmax(item["probabilities"]))]
                            if item["status"] == "ok" else None,
                            "status": "missing_probability" if item["status"] == "missing" else item["status"]}
                for direction, item in checked.items()}
    parents = {direction: {**item, "prediction": SR.NLI_CLASSES[int(np.argmax(item["probabilities"]))]
                          if item["status"] == "ok" else None} for direction, item in checked.items()}
    result = {"parents": parents}
    if any(item["status"] != "ok" for item in parents.values()):
        status = next(item["status"] for item in parents.values() if item["status"] != "ok")
        status = "missing_probability" if status == "missing" else status
        result.update({direction: {"status": status, "prediction": None, "probabilities": None,
                                   "valid_mass": None, "invalid_mass": None}
                       for direction in ("forward", "reverse")})
        return result
    forward, reverse = [np.asarray(parents[direction]["probabilities"]) for direction in ("forward", "reverse")]
    q = np.asarray([forward[SR.NLI_CLASSES.index(SR.SEMANTIC_TO_NLI[leaf])] *
                    reverse[SR.NLI_CLASSES.index(SR.SEMANTIC_TO_NLI[SR.swap_relation(leaf)])]
                    for leaf in SR.SEMANTIC_RELATIONS])
    valid_mass = float(q.sum())
    legal = legal_dyads()
    invalid_mass = float(sum(forward[i] * reverse[j]
                             for i, a in enumerate(SR.NLI_CLASSES) for j, b in enumerate(SR.NLI_CLASSES)
                             if (a, b) not in legal))
    conditional = (q / valid_mass).tolist() if valid_mass > 0 else None
    dyad = tuple(parents[direction]["prediction"] for direction in ("forward", "reverse"))
    predicted = legal.get(dyad)
    swapped_q = [float(q[SR.SEMANTIC_RELATIONS.index(SR.swap_relation(leaf))]) for leaf in SR.SEMANTIC_RELATIONS]
    for direction in ("forward", "reverse"):
        prediction = predicted if direction == "forward" or predicted is None else SR.swap_relation(predicted)
        result[direction] = {"status": "ok" if prediction is not None else "invalid_dyad",
                             "prediction": prediction,
                             "probabilities": conditional if direction == "forward" or conditional is None else
                             list(SR.swap_probabilities(conditional)),
                             "probability_status": "conditional_on_valid_dyad" if valid_mass > 0 else "zero_valid_mass",
                             "valid_mass": valid_mass, "invalid_mass": invalid_mass,
                             "legal_cell_mass": q.tolist() if direction == "forward" else swapped_q}
    return result


def input_audit(prepared):
    result = {key: prepared[key] for key in ("recipe", "input_limits", "raw_pair_sha256", "visible_information_sha256",
                                            "max_length", "complete", "full_visible_token_lengths",
                                            "inference_frame_declaration", "frame_condition")}
    result["directions"] = {direction: {**{key: view[key] for key in ("view_sha256", "visible_information_sha256", "fields",
                                                                        "token_length", "special_tokens", "role_body_covered",
                                                                        "attention_mask_sum", "complete")},
                                        "encoding_sha256": D.digest(view["encoding"])}
                            for direction, view in prepared["directions"].items()}
    return result


def predict_one(scorer, row, bundle):
    identifier = row["id"]
    evidence = bundle["evidence_by_id"][identifier]
    supervision = bundle["supervision_by_id"][identifier]
    source_row = next(item for item in bundle["rows"] if item["id"] == identifier)
    result = {"id": identifier, "root_id": row["root_id"], "cluster_id": row["cluster_id"],
              "split": row["split"], "original_split": source_row["split"], "gold": row["label"],
              "reverse_gold": supervision["fine_targets"][1], "evidence": evidence,
              "family": row["phenomena"]["family"], "language": row["phenomena"]["language"],
              "kind": "/".join(row["pair"][role]["kind"] for role in ("old", "new")),
              "input_audit": None, "truncation": "unknown", "error_code": None}
    stage = "input_failure"
    try:
        pair = D.student_pair(row)
        declaration = evidence["frame_role_independent"]
        prepared = scorer.prepare_inputs(pair, role_independent_frame=declaration)[0]
        result["input_audit"] = input_audit(prepared)
        result["truncation"] = "complete" if prepared["complete"] else "truncated"
        stage = "inference_failure"
        output = scorer.predict_both_proba(pair, role_independent_frame=declaration)
        result.update(decode_both(output, scorer.arm))
    except (ValueError, RuntimeError, OSError, TimeoutError) as error:
        status = "timeout" if isinstance(error, TimeoutError) else stage
        result["error_code"] = getattr(error, "code", type(error).__name__)
        result.update({direction: {"status": status, "prediction": None, "probabilities": None}
                       for direction in ("forward", "reverse")})
    for direction in ("forward", "reverse"):
        result[direction]["hard_success"] = result[direction]["prediction"] is not None
        result[direction]["probability_available"] = result[direction]["probabilities"] is not None
    result["technical_failure"] = any(result[direction]["status"] in TECHNICAL for direction in ("forward", "reverse"))
    return result


def _summary(records, direction):
    classes = SR.SEMANTIC_RELATIONS
    labels = [classes.index(row["gold" if direction == "forward" else "reverse_gold"]) for row in records]
    predictions = [classes.index(row[direction]["prediction"]) if row[direction]["prediction"] is not None else -1
                   for row in records]
    probs = [row[direction]["probabilities"] for row in records]
    masses = [row[direction] for row in records if row[direction].get("valid_mass") is not None]
    return {"classification": M.classification_summary(labels, predictions, classes=classes),
            "probability": M.probability_summary(probs, labels, n_classes=len(classes), n_bins=STATISTICS["n_bins"]),
            "dyad_mass": {"available_count": len(masses),
                          "mean_valid": float(np.mean([item["valid_mass"] for item in masses])) if masses else None,
                          "mean_invalid": float(np.mean([item["invalid_mass"] for item in masses])) if masses else None},
            "status_counts": dict(Counter(row[direction]["status"] for row in records)),
            "clusters": len({row["cluster_id"] for row in records})}


def _bootstrap(records):
    groups = [row["cluster_id"] for row in records]
    classes = SR.SEMANTIC_RELATIONS
    y = np.asarray([classes.index(row["gold"]) for row in records])
    pred = np.asarray([classes.index(row["forward"]["prediction"]) if row["forward"]["prediction"] is not None else -1
                       for row in records])
    probs = [row["forward"]["probabilities"] for row in records]
    full_probability = M.probability_summary(probs, y, n_classes=len(classes))
    result = {}
    for name in ("accuracy", "macro_f1", "brier", "ece", "nll"):
        def metric(index, weights, name=name):
            if name in ("accuracy", "macro_f1"):
                return M.classification_summary(y[index], pred[index], classes=classes, sample_weight=weights)[name]
            if name == "nll" and full_probability["nll"]["status"] == "positive_infinity":
                return None
            summary = M.probability_summary([probs[i] for i in index], y[index], n_classes=len(classes),
                                             n_bins=STATISTICS["n_bins"], sample_weight=weights)
            return summary[name]["value"] if name == "nll" else summary[name]
        result[name] = M.cluster_bootstrap(groups, metric, samples=STATISTICS["bootstrap_samples"],
                                           confidence=STATISTICS["confidence"], seed=STATISTICS["seed"], weighting="row")
        if name == "nll":
            result[name]["point_status"] = full_probability["nll"]["status"]
    return result


def summarize(records):
    roots = [row["root_id"] for row in records]
    if len(roots) != len(set(roots)):
        raise SemanticEvaluationError("one_canonical_per_root_required")
    strata = {}
    for field in ("split", "family", "language", "kind", "truncation"):
        strata[field] = {value: _summary([row for row in records if row[field] == value], "forward")
                         for value in sorted({row[field] for row in records})}
    return {"schema": FORMAT, "statistics": dict(STATISTICS), "resolved_roots": len(records),
            "technical_failures": sum(row["technical_failure"] for row in records),
            "primary_forward": _summary(records, "forward"), "reverse_diagnostic": _summary(records, "reverse"),
            "primary_cluster_bootstrap": _bootstrap(records), "strata": strata,
            "scope": "one_artifact; qualified_proxy_labels; no_action_threshold_or_risk_certification",
            "probability_scope": "available_only; dual3 conditional_on_valid_mass; hard_invalid_not_repaired",
            "strata_intervals": "not_computed; primary overall intervals resample dependency clusters"}


def _write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def evaluate(model_dir, source_manifest, output_dir, *, mode, splits, allocation=None,
             construction_scope=None, run_id=None, lock_path=None, confirm_final=False,
             cohort=None, identity_index=None, device="cpu"):
    selected = _splits(mode, splits)
    if mode == "final" and (not confirm_final or lock_path is None):
        raise SemanticEvaluationError("final_confirmation_and_lock_required")
    if mode == "development" and (lock_path is not None or confirm_final):
        raise SemanticEvaluationError("development_final_arguments_forbidden")
    if device not in ("cpu", "cuda"):
        raise SemanticEvaluationError("invalid_evaluation_device")
    specs = _source_specs(source_manifest, cohort, mode=mode, allocation=allocation,
                          construction_scope=construction_scope, run_id=run_id)
    binding = _evaluation_binding(model_dir, specs, mode=mode, splits=selected, cohort=cohort, identity_index=identity_index)
    if mode == "final":
        _check_final_lock(lock_path, binding)
    else:
        for spec in specs:
            if "construction_scope" in D.read_json(spec["source_manifest"]):
                raise SemanticEvaluationError("sealed_source_in_development")
    scorer = _load_scorer(model_dir, binding["model"], device)
    rows, lineages = _cohort_rows(specs, mode=mode, splits=selected, allow_empty_sources=cohort is not None)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    single = lineages[0] if cohort is None else {}
    metadata = {"schema": FORMAT, "status": "running", "binding": binding, "binding_sha256": D.digest(binding),
                "qualified_lineage": single.get("qualified_lineage"), "allocation_lineage": single.get("allocation_lineage"),
                "cohort_sources": lineages, "device": device,
                "final_lock_sha256": SM.FT.file_sha256(lock_path) if lock_path is not None else None}
    _write_json(output / "metadata.json", metadata)
    records = []
    with (output / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for source_id, row, bundle in rows:
            record = dict(predict_one(scorer, row, bundle), source_id=source_id,
                          source_manifest_sha256=bundle["lineage"]["manifest_sha256"])
            json.dump(record, handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            records.append(record)
    report = summarize(records)
    _write_json(output / "report.json", report)
    metadata["status"] = "completed_with_technical_failures" if report["technical_failures"] else "completed"
    metadata["files_sha256"] = {name: SM.FT.file_sha256(output / name) for name in ("predictions.jsonl", "report.json")}
    with (output / "metadata.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return {"status": metadata["status"], "resolved_roots": len(records), "binding_sha256": metadata["binding_sha256"],
            "predictions": str(output / "predictions.jsonl"), "report": str(output / "report.json"),
            "metadata": str(output / "metadata.json")}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "lock-final"):
        command = commands.add_parser(name)
        command.add_argument("--model", required=True)
        sources = command.add_mutually_exclusive_group(required=True)
        sources.add_argument("--source-manifest")
        sources.add_argument("--cohort")
        command.add_argument("--identity-index")
        command.add_argument("--split", action="append", required=True)
        command.add_argument("--construction-scope")
        command.add_argument("--run-id")
        if name == "evaluate":
            command.add_argument("--mode", choices=("development", "final"), default="development")
            command.add_argument("--allocation")
            command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
            command.add_argument("--lock")
            command.add_argument("--confirm-final", action="store_true")
            command.add_argument("--out", required=True)
        else:
            command.add_argument("--out-lock", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "lock-final":
            result = write_final_lock(args.model, args.source_manifest, args.out_lock, splits=args.split,
                                      construction_scope=args.construction_scope, run_id=args.run_id,
                                      cohort=args.cohort, identity_index=args.identity_index)
        else:
            result = evaluate(args.model, args.source_manifest, args.out, mode=args.mode, splits=args.split,
                              allocation=args.allocation, construction_scope=args.construction_scope, run_id=args.run_id,
                              lock_path=args.lock, confirm_final=args.confirm_final, cohort=args.cohort,
                              identity_index=args.identity_index, device=args.device)
    except (ValueError, RuntimeError, OSError) as error:
        print(json.dumps({"ok": False, "code": getattr(error, "code", type(error).__name__), "phase": "run_preflight_or_output"}), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 1 if result.get("status") == "completed_with_technical_failures" else 0


if __name__ == "__main__":
    raise SystemExit(main())
