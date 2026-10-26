"""Research-only text C5/dual3 controls; shared encoder/fit/artifact mechanisms, no final-data entry.

Input is generic canonical JSONL from the data writer. Public rows must already be derived from
protected fit/dev outputs; this module does not repartition or convert raw public benchmarks.
Tiny CPU runs verify mechanics only. Default budget is 32 canonical (16 public + 16 fine),
64 encoded orientations, microbatch 8 x accumulation 8; source means each receive weight 0.5.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dataset_recommender.agent import semantic_relations as SR
from research.memory_relations import finetune as FT
from research.memory_relations import nli_reference as NR
from research.memory_relations import semantic_data as D

ARM_CLASSES = {"c5": SR.SEMANTIC_RELATIONS, "dual3": SR.NLI_CLASSES}
SEEDS = (13, 20260919, 424242)
INPUT_RECIPE = SR.SEMANTIC_INPUT_RECIPE
INPUT_LIMITS = SR.SEMANTIC_LIMITS_RESEARCH
ARTIFACT_SCHEMA = "semantic-text-control-scorer-v6"
DATA_SCHEMA = "semantic-canonical-prepared-v6"
META_FILE = "semantic_meta.json"
WEIGHTS_FILE = "semantic_model.pt"
DIRECTION = "old_as_premise"
ROLES = {DIRECTION: {"premise": "old", "hypothesis": "new"}}


def _digest(value) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _state_sha256(state) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(_digest([name, list(tensor.shape), str(tensor.dtype)]).encode("ascii"))
        digest.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def initial_state_sha256(model) -> str:
    return _state_sha256(FT.cpu_state(model))


def source_binding() -> dict:
    import inspect

    files = {"semantic_model": Path(__file__), "finetune": Path(FT.__file__),
             "nli_reference": Path(NR.__file__), "semantic_relations": Path(SR.__file__),
             "redaction": Path(inspect.getsourcefile(SR.redact)), "semantic_data": Path(D.__file__),
             "data_job_spec": D.JOB_SPEC}
    return {name: FT.file_sha256(path) for name, path in files.items()}


def qualified_stage_rows(fine_manifest, allocation, *, bundle=None):
    """只变stage metadata；原资格始终在完整immutable source row上验证，不重签hash。"""
    bundle = D.load_qualified_source(fine_manifest) if bundle is None else bundle
    allocation = Path(allocation)
    document = D.read_json(allocation)
    if document.get("schema") != "semantic-source-allocation-v1" or type(document.get("items")) is not list:
        raise ValueError("invalid external fine allocation")
    rows = {row["id"]: row for row in bundle["rows"]}
    items = {item["source_id"]: item for item in document["items"]}
    if len(items) != len(document["items"]) or set(items) != set(rows):
        raise ValueError("external allocation source coverage/duplicate drift")
    for name, key in (("source_rows", "source_rows"), ("qualified", "qualifications"), ("evidence_manifest", "import_manifest")):
        reference = document.get("sources", {}).get(name)
        if reference is not None and reference["sha256"] != bundle["lineage"]["assets"][key]["sha256"]:
            raise ValueError("allocation asset lineage drift")
    cluster_stage, output = {}, []
    allocation_sha = FT.file_sha256(allocation)
    for identifier, source in rows.items():
        item = items[identifier]
        if item.get("stage_split") not in ("train", "dev") or item.get("source_row_sha256") != D.digest(source) \
                or item.get("root_id") != source["root_id"] or item.get("cluster_id") != source["cluster_id"]:
            raise ValueError("allocation source row/stage/hash drift")
        split = item["stage_split"]
        cluster = source["cluster_id"]
        if cluster in cluster_stage and cluster_stage[cluster] != split:
            raise ValueError("external allocation cluster crosses stages")
        cluster_stage[cluster] = split
        binding = {"format": "semantic-qualified-stage-row-v1", "manifest_path": bundle["lineage"]["manifest_path"],
                   "manifest_sha256": bundle["lineage"]["manifest_sha256"], "allocation_path": str(allocation.absolute()),
                   "allocation_sha256": allocation_sha, "allocation_item_sha256": D.digest(item), "source_id": identifier,
                   "original_split": source["split"], "stage_split": split,
                   "evidence": bundle["evidence_by_id"][identifier], "supervision": bundle["supervision_by_id"][identifier],
                   "source_lineage": bundle["lineage"]}
        staged = copy.deepcopy(source)
        staged["split"] = split
        staged["source_binding"] = binding
        output.append(staged)
    return output, {"qualified_source": bundle["lineage"], "allocation_path": str(allocation.absolute()),
                    "allocation_sha256": allocation_sha, "allocation_source_sha256": FT.file_sha256(__file__),
                    "original_to_stage": [{"source_id": row["id"], "original_row_sha256": row["source_binding"]["evidence"]["source_row_sha256"],
                                           "original_line_bytes_sha256": row["source_binding"]["evidence"]["source_row_bytes_sha256"],
                                           "stage_split": row["split"], "stage_row_sha256": D.digest(row)} for row in output]}


def _qualified_row_context(rows):
    context = {}
    for row in rows:
        binding = row.get("source_binding") if isinstance(row, dict) else None
        if binding:
            key = (binding.get("manifest_path"), binding.get("allocation_path"))
            if key not in context:
                staged, _ = qualified_stage_rows(*key)
                context[key] = {item["id"]: item for item in staged}
    return context


def validate_row(row, *, stage=None, fine_context=None) -> dict:
    if not isinstance(row, dict):
        raise ValueError("canonical row must be an object")
    for key in ("id", "root_id", "cluster_id", "split", "label_origin"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"canonical row requires nonempty {key}")
    if row.get("label_space") not in ("semantic5", "nli3") or "label" not in row:
        raise ValueError("label_space must be semantic5/nli3 with label name or null")
    classes = SR.SEMANTIC_RELATIONS if row["label_space"] == "semantic5" else SR.NLI_CLASSES
    if row["label"] is not None and row["label"] not in classes:
        raise ValueError("label is not in its declared label_space; coarse gold cannot imply a fine leaf")
    if row["label_space"] == "semantic5" and (stage is not None or row["label"] is not None):
        binding = row.get("source_binding")
        if type(binding) is not dict or binding.get("format") != "semantic-qualified-stage-row-v1":
            raise ValueError("fine row requires bound inverse qualification before formal supervision")
        key = (binding["manifest_path"], binding["allocation_path"])
        context = _qualified_row_context([row]) if fine_context is None else fine_context
        expected = context.get(key, {}).get(binding["source_id"])
        if expected is None or row != expected:
            raise ValueError("fine original/allocation/stage/evidence content drift")
    if stage is not None:
        allowed = ("train",) if stage == "fit" else ("dev",)
        if row.get("sealed") or row["split"] not in allowed:
            raise ValueError("sealed/final/wrong split cannot enter fit or checkpoint selection")
        if row["label"] is None:
            raise ValueError("fit/dev requires observed gold; null is inference-only")
    return dict(row, pair=SR.validate_semantic_pair(row.get("pair"), limits=INPUT_LIMITS))


def checked_pools(rows, dev_rows=(), *, require_both=True, require_coverage=True):
    rows, dev_rows = list(rows), list(dev_rows)
    context = _qualified_row_context(rows + dev_rows)
    fit = [validate_row(row, stage="fit", fine_context=context) for row in rows]
    dev = [validate_row(row, stage="dev", fine_context=context) for row in dev_rows]
    for key in ("id", "root_id", "cluster_id"):
        if {row[key] for row in fit} & {row[key] for row in dev}:
            raise ValueError(f"fit/dev {key} leakage")
    def pair_keys(pool):
        return {_digest(sorted((SR.semantic_pair_text(row["pair"], limits=INPUT_LIMITS),
                                SR.semantic_pair_text(SR.swap_pair(row["pair"], limits=INPUT_LIMITS), limits=INPUT_LIMITS)))) for row in pool}

    if pair_keys(fit) & pair_keys(dev):
        raise ValueError("fit/dev renamed or reversed semantic pair leakage")
    for pool in (fit, dev):
        if len({row["id"] for row in pool}) != len(pool):
            raise ValueError("duplicate canonical id")
        if require_both and pool and {row["label_space"] for row in pool} != {"nli3", "semantic5"}:
            raise ValueError("both public nli3 and fine semantic5 sources are required")
    if not fit:
        raise ValueError("empty fit pool")
    for space, classes in (("nli3", SR.NLI_CLASSES), ("semantic5", SR.SEMANTIC_RELATIONS)):
        if require_coverage and {row["label"] for row in fit if row["label_space"] == space} != set(classes):
            raise ValueError(f"fit pool must cover all classes in {space}")
    return fit, dev


def prepare_input_pairs(pairs, tokenizer, max_length, *, frame_declarations=None):
    """只消费已核数据声明或显式inference声明；recipe不自行声称frame角色无关。"""
    declarations = [False] * len(pairs) if frame_declarations is None else list(frame_declarations)
    if len(declarations) != len(pairs) or any(type(value) is not bool for value in declarations):
        raise ValueError("frame declaration length/type invalid")
    cache, result = {}, []
    for pair, declaration in zip(pairs, declarations):
        key = (SR.semantic_pair_text(pair, limits=INPUT_LIMITS), declaration)
        if key not in cache:
            cache[key] = SR.prepare_semantic_input(pair, tokenizer, max_length=max_length,
                                                   role_independent_frame=declaration, limits=INPUT_LIMITS)
        result.append(cache[key])
    return result


def oriented_rows(rows, tokenizer=None, max_length=None):
    rows = list(rows)
    context = _qualified_row_context(rows)
    checked = [validate_row(row, fine_context=context) for row in rows]
    declarations = [row["source_binding"]["evidence"]["frame_role_independent"] if row["label_space"] == "semantic5" and row["label"] is not None
                    else row.get("frame_role_independent", False) for row in checked]
    prepared = prepare_input_pairs([row["pair"] for row in checked], tokenizer, max_length,
                                   frame_declarations=declarations) if tokenizer is not None else None
    result = []
    for index, row in enumerate(checked):
        fine = row["label_space"] == "semantic5"
        supervised = row["source_binding"]["supervision"] if fine and row["label"] is not None else None
        for reverse in (False, True):
            direction = int(reverse)
            label = supervised["fine_targets"][direction] if supervised else row["label"] if not reverse else None
            nli_target = supervised["nli_targets"][direction] if supervised else label
            loss_mask = supervised["loss_mask"][direction] if supervised else int(label is not None)
            view = prepared[index]["directions"]["reverse" if reverse else "forward"] if prepared else None
            item = {"text": view["text"] if view else SR.semantic_pair_text(SR.swap_pair(row["pair"], limits=INPUT_LIMITS) if reverse else row["pair"], limits=INPUT_LIMITS),
                    "label": label, "label_space": row["label_space"], "loss_mask": loss_mask, "nli_target": nli_target,
                    "source": "fine" if fine else "public", "root_id": row["root_id"], "reverse": reverse,
                    "pair": row["pair"], "frame_role_independent": declarations[index]}
            if view:
                item.update({"input_view": view, "encoding": view["encoding"], "prepared": prepared[index]})
            result.append(item)
    return result


def orientation_losses(logits, orientations, arm):
    classes = ARM_CLASSES.get(arm)
    if classes is None or logits.ndim != 2 or logits.shape != (len(orientations), len(classes)):
        raise ValueError("semantic logits shape/arm mismatch")
    if not torch.isfinite(logits).all():
        raise ValueError("non-finite semantic logits")
    losses, hits = [], []
    for index, row in enumerate(orientations):
        label = row["label"]
        if label is None:
            losses.append(logits[index].sum() * 0)
            hits.append(0.0)
            continue
        if arm == "c5" and row["label_space"] == "nli3":
            leaves = [i for i, name in enumerate(SR.SEMANTIC_RELATIONS) if SR.SEMANTIC_TO_NLI[name] == label]
            loss = torch.logsumexp(logits[index].float(), dim=0) - torch.logsumexp(logits[index, leaves].float(), dim=0)
            parent_scores = torch.stack([torch.logsumexp(logits[index, [i for i, name in enumerate(classes)
                                                                       if SR.SEMANTIC_TO_NLI[name] == parent]].float(), dim=0)
                                         for parent in SR.NLI_CLASSES])
            hit = float(int(parent_scores.detach().argmax()) == SR.NLI_CLASSES.index(label))
        else:
            target_name = row.get("nli_target", SR.SEMANTIC_TO_NLI[label]) if arm == "dual3" and row["label_space"] == "semantic5" else label
            target = classes.index(target_name)
            loss = -torch.log_softmax(logits[index].float(), dim=0)[target]
            hit = float(int(logits[index].detach().argmax()) == target)
        losses.append(loss)
        hits.append(hit)
    return torch.stack(losses), logits.new_tensor(hits)


def source_weights(orientations, *, root_equal=False):
    weights = np.zeros(len(orientations), dtype=np.float64)
    for source in ("public", "fine"):
        indexes = [i for i, row in enumerate(orientations) if row["source"] == source and row["label"] is not None]
        if not indexes:
            raise ValueError("source has no supervised directions")
        roots = Counter(orientations[i]["root_id"] for i in indexes)
        for i in indexes:
            weights[i] = (0.5 / len(roots) / roots[orientations[i]["root_id"]]
                          if root_equal else 0.5 / len(indexes))
    return weights


def balanced_windows(rows, *, seed, updates, canonical_batch=32, microbatch=8, grad_accum=8):
    if canonical_batch < 2 or canonical_batch % 2 or microbatch * grad_accum != canonical_batch * 2:
        raise ValueError("budget must encode exactly two orientations per balanced canonical batch")
    public = [i for i, row in enumerate(rows) if row["label_space"] == "nli3"]
    fine_roots = {}
    for index, row in enumerate(rows):
        if row["label_space"] == "semantic5":
            fine_roots.setdefault(row["root_id"], []).append(index)
    roots = sorted(fine_roots)
    if not public or not roots:
        raise ValueError("both sources required")
    streams = [FT.training_windows(size, epochs=1, batch_size=canonical_batch // 2,
                                    grad_accum=1, max_steps=updates, rng=np.random.default_rng(seed + index))
               for index, size in enumerate((len(public), len(roots)))]
    variants = np.random.default_rng(seed + 2)
    for left, right in zip(*streams):
        canonical = [public[i] for i in left[2][0]]
        canonical.extend(int(variants.choice(fine_roots[roots[i]])) for i in right[2][0])
        oriented = np.asarray([i * 2 + reverse for i in canonical for reverse in (0, 1)], dtype=np.int64)
        chunks = [oriented[start:start + microbatch] for start in range(0, len(oriented), microbatch)]
        yield left[0], max(left[1], right[1]), chunks, left[3] or right[3]


def require_model(model, *, arm, where="semantic"):
    if arm not in ARM_CLASSES:
        raise ValueError("unknown arm")
    head = getattr(model, "classifier", None)
    hidden = getattr(getattr(getattr(model, "backbone", None), "config", None), "hidden_size", None)
    if getattr(head, "out_features", None) != len(ARM_CLASSES[arm]) or getattr(head, "in_features", None) != hidden:
        raise ValueError(f"{where}: class/hidden shape mismatch")
    return hidden


def initialize_arm(mother, arm):
    NR._require_nli_model(mother, where="semantic mother3")
    if arm not in ARM_CLASSES:
        raise ValueError("unknown arm")
    model = copy.deepcopy(mother).cpu()
    if arm == "c5":
        head = nn.Linear(mother.classifier.in_features, len(SR.SEMANTIC_RELATIONS))
        counts = Counter(SR.SEMANTIC_TO_NLI.values())
        with torch.no_grad():
            for index, name in enumerate(SR.SEMANTIC_RELATIONS):
                parent = SR.SEMANTIC_TO_NLI[name]
                source = SR.NLI_CLASSES.index(parent)
                head.weight[index].copy_(mother.classifier.weight[source].detach().cpu())
                head.bias[index].copy_(mother.classifier.bias[source].detach().cpu() - math.log(counts[parent]))
        model.classifier = head
    require_model(model, arm=arm)
    return model


def truncation_profile(tokenizer, orientations, max_length):
    if all("prepared" in row for row in orientations):
        items = [row["prepared"] for row in orientations]
    else:
        items = prepare_input_pairs([row["pair"] for row in orientations], tokenizer, max_length,
                                     frame_declarations=[row["frame_role_independent"] for row in orientations])
    views = [item["directions"]["reverse" if row["reverse"] else "forward"] for row, item in zip(orientations, items)]
    return {"recipe": INPUT_RECIPE, "input_limits": dict(INPUT_LIMITS), "oriented_rows": len(views),
            "truncated_rows": sum(not view["complete"] for view in views), "max_length": max_length,
            "untruncated_tokens": sum(item["full_visible_token_lengths"][int(row["reverse"])] for row, item in zip(orientations, items)),
            "encoded_tokens_without_padding": sum(view["token_length"] for view in views),
            "same_visible_truncation_implemented": True,
            "requires_complete_input_or_llm_fallback": any(not view["complete"] for view in views),
            "raw_pair_hashes": [item["raw_pair_sha256"] for item in items],
            "visible_information_hashes": [item["visible_information_sha256"] for item in items],
            "audit": [{"raw_pair_sha256": item["raw_pair_sha256"], "view_sha256": view["view_sha256"],
                       "reverse": row["reverse"], "token_length": view["token_length"],
                       "role_body_covered": view["role_body_covered"], "loss_mask": row["loss_mask"],
                       "fields": view["fields"]} for row, item, view in zip(orientations, items, views)]}


def joint_parent_losses(logits, orientations, arm):
    parents = [dict(row, label=SR.SEMANTIC_TO_NLI[row["label"]]
                    if row["label_space"] == "semantic5" and row["label"] is not None else row["label"],
                    label_space="nli3") for row in orientations]
    return orientation_losses(logits, parents, arm)


def eval_semantic(model, tokenizer, rows, *, arm, max_length, device, batch_size=16):
    orientations = oriented_rows(rows, tokenizer, max_length)
    weights = source_weights(orientations, root_equal=True)
    losses, hits, fine_losses, fine_hits = [], [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(orientations), batch_size):
            chunk = orientations[start:start + batch_size]
            encoded = FT.encode_prepared_views(tokenizer, [row["encoding"] for row in chunk], device)
            logits = model(**encoded)
            loss, correct = joint_parent_losses(logits, chunk, arm)
            diagnostic_loss, diagnostic_hits = orientation_losses(logits, chunk, arm)
            losses.extend(loss.cpu().tolist())
            hits.extend(correct.cpu().tolist())
            fine_losses.extend(diagnostic_loss.cpu().tolist())
            fine_hits.extend(diagnostic_hits.cpu().tolist())
    valid = [i for i, row in enumerate(orientations) if row["label"] is not None]
    fine = [i for i, row in enumerate(orientations) if row["source"] == "fine"]
    fine_weights = weights[fine] * 2
    diagnostics = {"fine_parent_accuracy": float(np.dot(fine_weights, np.asarray(hits)[fine])),
                   "fine_parent_nll": float(np.dot(fine_weights, np.asarray(losses)[fine]))}
    if arm == "c5":
        diagnostics.update({"fine_leaf_nll": float(np.dot(fine_weights, np.asarray(fine_losses)[fine])),
                            "fine_leaf_accuracy": float(np.dot(fine_weights, np.asarray(fine_hits)[fine]))})
    return {"loss": float(np.dot(weights, losses)), "row_loss": float(np.asarray(losses)[valid].mean()),
            "accuracy": float(np.dot(weights, hits)), "macro_f1": None,
            "criterion": "source_half_root_equal_joint_parent_nll", "diagnostics": diagnostics,
            "weighting": "source_half_then_root_equal", "n_groups": len({row["root_id"] for row in rows})}


def _prepared_training_data(directory, *, mechanics_only):
    if directory is None or type(mechanics_only) is not bool:
        raise ValueError("training requires immutable prepared data and explicit mechanics scope")
    directory = Path(directory).absolute()
    fit, dev = read_prepared(directory)
    meta_path = directory / "prepare.json"
    meta = D.read_json(meta_path)
    if not mechanics_only and meta["formal_fit_source_ready"] is not True:
        raise ValueError("formal training requires protected public and qualified fine prepared sources")
    members = [{"source_id": row["id"], "root_id": row["root_id"], "cluster_id": row["cluster_id"],
                "stage": row["split"], "label_space": row["label_space"], "row_sha256": D.digest(row)} for row in fit + dev]
    reference = {"directory": str(directory), "metadata_sha256": FT.file_sha256(meta_path),
                 "files_sha256": meta["files_sha256"], "members_sha256": _digest(members)}
    return fit, dev, meta, reference


def _fine_training_evidence(rows):
    return [{"source_id": row["id"], "root_id": row["root_id"], "cluster_id": row["cluster_id"],
             "stage": row["split"], "stage_row_sha256": D.digest(row),
             "manifest_path": row["source_binding"]["manifest_path"],
             "allocation_path": row["source_binding"]["allocation_path"],
             "manifest_sha256": row["source_binding"]["manifest_sha256"],
             "allocation_sha256": row["source_binding"]["allocation_sha256"],
             "evidence": row["source_binding"]["evidence"]} for row in rows if row["label_space"] == "semantic5"]


def _source_batch_record(fit, step, chunks):
    canonical = np.concatenate(chunks)[::2] // 2
    members = [fit[index] for index in canonical]
    return {"step": step, "source_ids": [row["id"] for row in members],
            "root_ids": [row["root_id"] for row in members], "member_sha256": _digest([D.digest(row) for row in members])}


def train_semantic(model, tokenizer, rows, *, arm, dev_rows=(), seed=13, updates,
                   canonical_batch=32, microbatch=8, grad_accum=8, max_length=384,
                   device=torch.device("cpu"), bf16=False, checkpoint_selection="best-dev",
                   eval_every_steps=25, lr_backbone=FT.LR_BACKBONE, lr_head=FT.LR_HEAD,
                   weight_decay=FT.WEIGHT_DECAY, resume_state=None, event_callback=None,
                   clip_grad_norm=1.0, producer_lineage=None, prepared_directory=None, mechanics_only=False):
    frozen_source = source_binding()
    require_model(model, arm=arm)
    NR._require_trainable_backbone(model, where="semantic full")
    FT.validate_training_options(epochs=1, batch_size=microbatch, grad_accum=grad_accum,
                                 max_length=max_length, checkpoint_selection=checkpoint_selection,
                                 selection_metric="dev-loss", patience=0, min_delta=0.0,
                                 max_steps=updates, eval_every_steps=eval_every_steps, eval_batch_size=16)
    if type(updates) is not int or updates < 1 or type(seed) is not int or not 0 <= seed < 2**32 - 1 \
            or eval_every_steps < 1 or not math.isfinite(clip_grad_norm) or clip_grad_norm <= 0:
        raise ValueError("fixed updates/seed/grid/clip invalid")
    if max_length > model.backbone.config.max_position_embeddings:
        raise ValueError("max_length exceeds backbone positions")
    fit, dev = checked_pools(rows, dev_rows)
    prepared_fit, prepared_dev, prepared_meta, prepared_binding = _prepared_training_data(prepared_directory, mechanics_only=mechanics_only)
    if fit != prepared_fit or dev != prepared_dev or producer_lineage != prepared_meta["producer_lineage"]:
        raise ValueError("training rows/producer do not match immutable prepared members")
    if mechanics_only and device.type != "cpu":
        raise ValueError("mechanics fixture must remain CPU only, not formal certification")
    if checkpoint_selection == "best-dev" and not dev:
        raise ValueError("best-dev requires independent dev")
    orientations = oriented_rows(fit, tokenizer, max_length)
    dev_orientations = oriented_rows(dev, tokenizer, max_length) if dev else []
    input_hashes = lambda items: [{"raw": row["prepared"]["raw_pair_sha256"],
                                  "visible": row["prepared"]["visible_information_sha256"],
                                  "view": row["input_view"]["view_sha256"], "loss_mask": row["loss_mask"]} for row in items]
    binding = {"data_sha256": _digest(fit), "dev_sha256": _digest(dev), "seed": seed, "arm": arm,
               "prepared_data": prepared_binding, "mechanics_only": mechanics_only,
               "canonical_batch": canonical_batch, "microbatch": microbatch, "grad_accum": grad_accum,
               "max_length": max_length, "bf16": bf16, "checkpoint_selection": checkpoint_selection,
               "eval_every_steps": eval_every_steps, "lr_backbone": lr_backbone, "lr_head": lr_head,
               "weight_decay": weight_decay, "dropout": float(model.dropout.p),
               "initial_state_sha256": initial_state_sha256(model), "clip_grad_norm": clip_grad_norm,
               "tokenizer_encoding_sha256": NR.tokenizer_encoding_sha256(tokenizer),
               "backbone_config_sha256": NR._config_digest(model.backbone.config),
               "source_sha256": frozen_source, "producer_lineage": producer_lineage,
               "fine_supervision_rule": D.SUPERVISION_RULE, "fine_evidence": _fine_training_evidence(fit + dev),
               "fine_sampling": "root_uniform_then_variant_uniform", "input_recipe": INPUT_RECIPE, "input_limits": dict(INPUT_LIMITS),
               "train_input_hashes": input_hashes(orientations), "dev_input_hashes": input_hashes(dev_orientations),
               "selection_estimand": "source_half_root_equal_joint_parent_nll"}
    if resume_state is not None and (not isinstance(resume_state, dict) or not resume_state
                                     or resume_state.get("training_binding") != binding):
        raise ValueError("continuation missing or data/recipe/budget/initialization/tokenizer/source drift")
    torch.manual_seed(seed)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested CUDA unavailable; no CPU fallback")
        torch.cuda.manual_seed_all(seed)
    model.to(device)
    optimizer = FT.build_joint_optimizer(model, train_backbone=True, lr_backbone=lr_backbone,
                                         lr_head=lr_head, weight_decay=weight_decay)
    covered = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if covered != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError("optimizer coverage mismatch")
    texts = [row["text"] for row in orientations]

    def loss_fn(logits, chunk, window):
        items = [orientations[index] for index in chunk]
        all_items = [orientations[index] for index in window]
        all_weights = source_weights(all_items)
        by_index = dict(zip(window.tolist(), all_weights))
        weights = logits.new_tensor([by_index[index] for index in chunk])
        losses, hits = orientation_losses(logits, items, arm)
        return (losses * weights).sum() * len(window), float((hits * weights).sum().detach()) * len(window)

    def evaluate():
        return eval_semantic(model, tokenizer, dev, arm=arm, max_length=max_length, device=device)

    batch_order = []
    def recorded_windows():
        for step, epoch, chunks, end in balanced_windows(fit, seed=seed, updates=updates, canonical_batch=canonical_batch,
                                                        microbatch=microbatch, grad_accum=grad_accum):
            batch_order.append(_source_batch_record(fit, step, chunks))
            yield step, epoch, chunks, end

    result = FT.fit_joint(model, tokenizer, texts, optimizer=optimizer,
                          windows=recorded_windows(),
                          planned_steps=updates, planned_instances=updates * canonical_batch * 2,
                          max_length=max_length, device=device, bf16=bf16, loss_fn=loss_fn,
                          evaluate=evaluate if dev else None, checkpoint_selection=checkpoint_selection,
                          eval_every_steps=eval_every_steps, require_backbone_grad=True,
                          resume_state=resume_state, event_callback=event_callback,
                          clip_grad_norm=clip_grad_norm,
                          encode_fn=lambda chunk: FT.encode_prepared_views(
                              tokenizer, [orientations[index]["encoding"] for index in chunk], device))
    if source_binding() != frozen_source:
        raise ValueError("source drift during training; refusing artifact")
    result["continuation"]["training_binding"] = binding
    result["training_binding"] = binding
    result["input_limits"] = dict(INPUT_LIMITS)
    result["mechanics_only"] = mechanics_only
    result["purpose"] = "mechanics-fixture" if mechanics_only else "formal-control"
    result["source_end_sha256"] = frozen_source
    result["source_batch_order"] = batch_order
    result["continuation"]["source_batch_order"] = batch_order
    result["budget"]["source_batch_order_sha256"] = _digest(batch_order)
    result["selection"].update({"weighting": "source_half_then_root_equal",
                                "estimand": "source_half_root_equal_joint_parent_nll",
                                "eval_every_steps": eval_every_steps})
    result["budget"].update({"canonical_batch": canonical_batch, "source_canonical_per_update": canonical_batch // 2,
                              "oriented_batch": canonical_batch * 2, "microbatch": microbatch, "grad_accum": grad_accum,
                              "actual_canonical_instances": result["budget"]["actual_updates"] * canonical_batch,
                              "supervised_directions_per_update": canonical_batch * 3 // 2,
                              "source_weights": {"public": 0.5, "fine": 0.5}, "public_reverse_mask": 0,
                              "qualified_fine_only": True, "fine_supervision_rule": D.SUPERVISION_RULE,
                              "qualification_manifest_sha256": sorted({item["manifest_sha256"] for item in binding["fine_evidence"]}),
                              "allocation_sha256": sorted({item["allocation_sha256"] for item in binding["fine_evidence"]}),
                              "allocation_source_sha256": frozen_source["semantic_model"]})
    result["truncation"] = {"train": truncation_profile(tokenizer, orientations, max_length)}
    if dev:
        result["truncation"]["dev"] = truncation_profile(tokenizer, oriented_rows(dev), max_length)
    result["guard"] = {"full_optimizer_coverage": True, "backbone_grad_seen": result.pop("backbone_grad_seen"),
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                        "device": str(device), "bf16": bf16, "arm": arm, "classes": list(ARM_CLASSES[arm])}
    return result


class SemanticScorer:
    def __init__(self, *, model, tokenizer, max_length, arm, device="cpu", batch_size=16, direction=DIRECTION):
        if direction != DIRECTION or max_length < 1 or batch_size < 1:
            raise ValueError("scorer direction/window invalid")
        require_model(model, arm=arm)
        self.arm, self.classes, self.direction = arm, ARM_CLASSES[arm], direction
        self.model, self.tokenizer = model, tokenizer
        self.max_length, self.batch_size = int(max_length), int(batch_size)
        self.device = str(FT.resolve_device(device))
        self.model.to(self.device).eval()

    def prepare_inputs(self, pairs, *, role_independent_frame=False):
        items = [pairs] if isinstance(pairs, dict) else list(pairs)
        prepared = prepare_input_pairs(items, self.tokenizer, self.max_length,
                                        frame_declarations=[role_independent_frame] * len(items))
        return [dict(item, inference_frame_declaration=role_independent_frame,
                     use="prediction_not_formal_qualification") for item in prepared]

    def _predict_views(self, views):
        output = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(views), self.batch_size):
                chunk = views[start:start + self.batch_size]
                logits = self.model(**FT.encode_prepared_views(self.tokenizer, [view["encoding"] for view in chunk],
                                                               torch.device(self.device)))
                if logits.shape != (len(chunk), len(self.classes)) or not torch.isfinite(logits).all():
                    raise ValueError("non-finite/shape-invalid scorer output")
                output.extend(torch.softmax(logits.float(), dim=1).cpu().tolist())
        return np.asarray(output, dtype=np.float64).reshape(-1, len(self.classes))

    def predict_proba(self, pairs, *, role_independent_frame=False):
        return self._predict_views([item["directions"]["forward"] for item in self.prepare_inputs(pairs, role_independent_frame=role_independent_frame)])

    def predict_both_proba(self, pairs, *, role_independent_frame=False):
        prepared = self.prepare_inputs(pairs, role_independent_frame=role_independent_frame)
        return {direction: self._predict_views([item["directions"][direction] for item in prepared])
                for direction in ("forward", "reverse")}

    def predict_rows(self, rows):
        oriented = oriented_rows(rows, self.tokenizer, self.max_length)
        return self._predict_views([item["input_view"] for item in oriented[::2]])

    def predict_nli_proba(self, pairs, *, role_independent_frame=False):
        probabilities = self.predict_proba(pairs, role_independent_frame=role_independent_frame)
        if self.arm == "dual3":
            return probabilities
        return np.asarray([SR.project_nli_probabilities(row.tolist()) for row in probabilities]).reshape(-1, 3)


def validate_training_ledger(report, continuation, *, state_dict=None, input_params=None):
    def require(ok, detail):
        if not ok:
            raise ValueError("training ledger mismatch: " + detail)

    def integer(value, *, positive=False):
        require(type(value) is int and value >= int(positive), "invalid counter")
        return value

    bound = report["input_parameters"] if report.get("purpose") == "untrained-initializer" else report["training_binding"]
    require(report["input_limits"] == bound["input_limits"] == INPUT_LIMITS, "input limits")
    if input_params is not None:
        require(input_params["recipe"] == INPUT_RECIPE and input_params["max_length"] == bound["max_length"], "scorer input budget/recipe")
        require(input_params.get("input_limits") == bound["input_limits"], "scorer input limits")
    if isinstance(report, dict) and report.get("purpose") == "untrained-initializer":
        require(continuation is None and report["budget"] == {"actual_updates": 0, "actual_sample_instances": 0},
                "initializer cannot claim updates or Adam continuation")
        require(report["initialization"]["kind"] == "raw-backbone-untrained"
                and report["initialization"]["optimizer_updates"] == 0, "initializer must use raw baseline")
        require(report["input_recipe"] == INPUT_RECIPE and report["source_sha256"] == source_binding(), "initializer recipe/source")
        if state_dict is not None:
            require(_state_sha256(state_dict) == report["initial_state_sha256"], "initializer weights")
        return
    require(isinstance(report, dict) and isinstance(continuation, dict) and bool(continuation),
            "missing continuation; cold start is forbidden")
    budget, binding, selection = report["budget"], report["training_binding"], report["selection"]
    updates, history = report["updates"], report["history"]
    actual = integer(budget["actual_updates"], positive=True)
    require(actual == integer(continuation["completed_steps"], positive=True) == len(updates), "completed/actual updates")
    require(integer(budget["planned_updates"], positive=True) == actual, "fixed updates not completed")
    canonical = integer(binding["canonical_batch"], positive=True)
    micro, accum = integer(binding["microbatch"], positive=True), integer(binding["grad_accum"], positive=True)
    require(canonical % 2 == 0 and micro * accum == 2 * canonical, "canonical/oriented budget")
    expected = {"mode": "fixed-updates", "planned_sample_instances": actual * canonical * 2,
                "actual_sample_instances": actual * canonical * 2, "actual_canonical_instances": actual * canonical,
                "canonical_batch": canonical, "source_canonical_per_update": canonical // 2,
                "oriented_batch": canonical * 2, "microbatch": micro, "grad_accum": accum,
                "supervised_directions_per_update": canonical * 3 // 2, "public_reverse_mask": 0,
                "source_weights": {"public": 0.5, "fine": 0.5}}
    require(all(budget.get(key) == value for key, value in expected.items()), "budget/binding counters")
    require(binding == continuation.get("training_binding"), "training binding")
    require(binding["source_sha256"] == report["source_end_sha256"] == source_binding(), "start/end source drift")
    require(binding["input_recipe"] == INPUT_RECIPE, "visible recipe drift")
    require(binding["fine_supervision_rule"] == budget["fine_supervision_rule"] == D.SUPERVISION_RULE
            and budget["qualified_fine_only"] is True, "fine inverse rule")
    require(report["source_batch_order"] == continuation["source_batch_order"]
            and len(report["source_batch_order"]) == actual
            and _digest(report["source_batch_order"]) == budget["source_batch_order_sha256"], "source/root batch order")
    require(type(binding.get("prepared_data")) is dict and bool(binding["prepared_data"]), "missing immutable prepared binding")
    require(type(binding.get("mechanics_only")) is bool and report.get("mechanics_only") is binding["mechanics_only"]
            and report.get("purpose") == ("mechanics-fixture" if binding["mechanics_only"] else "formal-control"), "mechanics/formal scope")
    fit, dev, prepared_meta, prepared_binding = _prepared_training_data(binding["prepared_data"].get("directory"),
                                                                      mechanics_only=binding["mechanics_only"])
    require(prepared_binding == binding["prepared_data"] and _digest(fit) == binding["data_sha256"]
            and _digest(dev) == binding["dev_sha256"] and prepared_meta["producer_lineage"] == binding["producer_lineage"], "immutable prepared members/data digest")
    expected_evidence = _fine_training_evidence(fit + dev)
    require(bool(expected_evidence) and binding["fine_evidence"] == expected_evidence, "fine evidence must cover immutable train/dev members")
    manifests = sorted({item["manifest_sha256"] for item in expected_evidence})
    allocations = sorted({item["allocation_sha256"] for item in expected_evidence})
    require(bool(manifests) and bool(allocations) and manifests == budget["qualification_manifest_sha256"]
            and allocations == budget["allocation_sha256"]
            and budget["allocation_source_sha256"] == binding["source_sha256"]["semantic_model"], "qualification/allocation budget binding")
    expected_batches = [_source_batch_record(fit, step, chunks) for step, _, chunks, _ in balanced_windows(
        fit, seed=binding["seed"], updates=actual, canonical_batch=canonical, microbatch=micro, grad_accum=accum)]
    require(report["source_batch_order"] == expected_batches, "participating source/root/member digest coverage")
    for split, pool in (("train", fit), ("dev", dev)):
        hashes = binding[split + "_input_hashes"]
        profile = report["truncation"].get(split)
        require(len(hashes) == 2 * len(pool), "prepared split input coverage")
        if not hashes:
            require(profile is None, "empty split input binding")
            continue
        require(profile["recipe"] == INPUT_RECIPE and profile["input_limits"] == bound["input_limits"]
                and profile["same_visible_truncation_implemented"], "input recipe/limits/profile")
        require(len(hashes) % 2 == 0 and len(profile["audit"]) == len(hashes), "input audit count")
        for index, (item, audit) in enumerate(zip(hashes, profile["audit"])):
            row = pool[index // 2]
            mask = int(index % 2 == 0 or row["label_space"] == "semantic5")
            require(item["raw"] == SR.semantic_pair_pieces(row["pair"], limits=INPUT_LIMITS)["raw_pair_sha256"] and item["loss_mask"] == mask, "prepared member raw input/mask")
            require(item["raw"] == audit["raw_pair_sha256"] == profile["raw_pair_hashes"][index]
                    and item["visible"] == profile["visible_information_hashes"][index]
                    and item["view"] == audit["view_sha256"] and item["loss_mask"] == audit["loss_mask"], "raw/visible/view hash binding")
            require(audit["token_length"] <= binding["max_length"] and audit["role_body_covered"] == {"old": True, "new": True}, "body/budget audit")
            if index % 2:
                require(item["raw"] == hashes[index - 1]["raw"] and item["visible"] == hashes[index - 1]["visible"], "same-visible reversal")
    require(selection["estimand"] == binding["selection_estimand"] == "source_half_root_equal_joint_parent_nll",
            "checkpoint estimand")
    require(selection["eval_every_steps"] == binding["eval_every_steps"] >= 1, "evaluation grid")
    require(updates == continuation.get("updates") and history == continuation.get("history"), "history/update records")
    for step, update in enumerate(updates, 1):
        require(update["step"] == step and update["sample_instances"] == canonical * 2
                and update["cumulative_sample_instances"] == step * canonical * 2
                and update["microbatch_sizes"] == [micro] * accum, "per-update exposure")
        for key in ("real_tokens", "padding_tokens"):
            integer(update[key])
        require(math.isfinite(update["loss"]) and math.isfinite(update["gradient_norm"]), "nonfinite update")
        require(update["clip_grad_norm"] == binding["clip_grad_norm"]
                and math.isfinite(update["clipped_gradient_norm"])
                and 0 <= update["clipped_gradient_norm"] <= binding["clip_grad_norm"] + 1e-6, "gradient clipping")
    for key in ("real_tokens", "padding_tokens"):
        require(integer(budget[key]) == sum(update[key] for update in updates) == integer(continuation[key]), "tokens")
    require(continuation["instances"] == budget["actual_sample_instances"], "instances")
    require([row["step"] for row in history] == sorted({row["step"] for row in history})
            and all(1 <= row["step"] <= actual for row in history) and history[-1]["step"] == actual, "evaluation history")
    selected = integer(selection["selected_step"], positive=True)
    require(selected <= actual and selection["mode"] == binding["checkpoint_selection"], "selection range/mode")
    require(selection["best_dev_value"] == continuation["best_value"], "best-dev value")
    if selection["mode"] == "best-dev":
        require(selected == continuation["selected_step"] and selection["selected_epoch"] == continuation["selected_epoch"], "selected checkpoint")
        dev = [row for row in history if "dev_loss" in row]
        require(bool(dev) and all(math.isfinite(row["dev_loss"]) for row in dev), "finite dev evaluations")
        best = min(dev, key=lambda row: row["dev_loss"])
        require(best["step"] == selected and best["dev_loss"] == selection["best_dev_value"], "best proper dev selection")
        chosen = continuation["best_state"]
    else:
        require(selected == actual, "final checkpoint")
        chosen = continuation["terminal_state"]
    for snapshot in (continuation["terminal_state"], chosen):
        require(isinstance(snapshot, dict) and bool(snapshot) and all(torch.is_tensor(value) and torch.isfinite(value).all()
                                                                    for value in snapshot.values()), "finite snapshots")
    if state_dict is not None:
        require(set(state_dict) == set(chosen) and all(torch.equal(state_dict[key], chosen[key]) for key in chosen), "selected weights")
    optimizer = continuation["optimizer_state"]
    require(bool(optimizer["state"]) and bool(optimizer["param_groups"]), "Adam state")
    for values in optimizer["state"].values():
        require(int(values["step"]) == actual and all(not torch.is_tensor(value) or torch.isfinite(value).all()
                                                      for value in values.values()), "Adam steps/finite values")


def artifact_spec(arm):
    if arm not in ARM_CLASSES:
        raise ValueError("unknown arm")
    return {"schema": ARTIFACT_SCHEMA, "kind": "semantic-text-" + arm, "classes": ARM_CLASSES[arm],
            "label_space": "semantic5" if arm == "c5" else "nli3", "recipe": INPUT_RECIPE,
            "weights_file": WEIGHTS_FILE, "meta_file": META_FILE, "roles": ROLES,
            "model_guard": lambda model, where: require_model(model, arm=arm, where=where),
            "scorer_type": lambda **kwargs: SemanticScorer(arm=arm, **kwargs),
            "validate_training": validate_training_ledger, "require_input_parameters": True,
            "input_limits": dict(INPUT_LIMITS)}


def artifact_binding(arm):
    return {"source_sha256": source_binding(), "arm": arm, "input_recipe": INPUT_RECIPE, "input_limits": dict(INPUT_LIMITS),
            "semantic_to_nli": dict(SR.SEMANTIC_TO_NLI), "swap": dict(SR.SWAPPED_RELATIONS)}


def save_semantic_initializer(scorer, directory, *, backbone_dir, mother_initialization):
    """显式零更新工件只作 factory/export 验证；没有 Adam，也不能冒充训练或续训。"""
    if mother_initialization.get("kind") != "raw-backbone-untrained" or mother_initialization.get("optimizer_updates") != 0:
        raise ValueError("initializer requires raw untrained mother, not native warm-start")
    report = {"purpose": "untrained-initializer", "initialization": mother_initialization,
              "budget": {"actual_updates": 0, "actual_sample_instances": 0}, "input_recipe": INPUT_RECIPE, "input_limits": dict(INPUT_LIMITS),
              "input_parameters": {"recipe": INPUT_RECIPE, "max_length": int(scorer.max_length), "input_limits": dict(INPUT_LIMITS)},
              "source_sha256": source_binding(), "initial_state_sha256": initial_state_sha256(scorer.model)}
    return NR.save_joint_artifact(scorer, directory, spec=artifact_spec(scorer.arm), backbone_dir=backbone_dir,
                                  train_meta=report, bindings=artifact_binding(scorer.arm))


def save_semantic_artifact(scorer, directory, *, backbone_dir, training):
    report = {key: value for key, value in training.items() if key != "continuation"}
    frozen_source = report["training_binding"]["source_sha256"]
    if frozen_source != report["source_end_sha256"] or source_binding() != frozen_source:
        raise ValueError("source drift since training start; refusing artifact")
    bindings = artifact_binding(scorer.arm)
    bindings["source_sha256"] = frozen_source
    return NR.save_joint_artifact(scorer, directory, spec=artifact_spec(scorer.arm), backbone_dir=backbone_dir,
                                  train_meta=report, bindings=bindings,
                                  extra_payload=training.get("continuation"))


def load_semantic_artifact(directory, *, device="cpu"):
    meta = json.loads((Path(directory) / META_FILE).read_text(encoding="utf-8"))
    arm = meta.get("bindings", {}).get("arm")
    return NR.load_joint_artifact(directory, spec=artifact_spec(arm), device=device, bindings=artifact_binding(arm))


def initialize_untrained_mother(backbone_dir, directory, *, seed=13, max_length=384):
    if seed not in SEEDS:
        raise ValueError("initialization seed is outside registered seeds")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model, config = FT.build_joint_model(backbone_dir, num_labels=len(SR.NLI_CLASSES))
    tokenizer = NR._reconstruct_tokenizer(backbone_dir, use_fast=True, where="raw initialization")
    scorer = NR.NLIScorer(model=model, tokenizer=tokenizer, max_length=max_length, device="cpu")
    initialization = {"kind": "raw-backbone-untrained", "optimizer_updates": 0, "seed": seed,
                      "initial_state_sha256": initial_state_sha256(model), "effective_dtype": str(config.dtype)}
    return NR.save_nli_artifact(scorer, directory, backbone_dir=backbone_dir,
                                train_meta={"initialization": initialization})


def resume_continuation(directory):
    payload = load_semantic_artifact(directory, device="cpu")[2]
    continuation = payload.get("continuation")
    if not isinstance(continuation, dict) or not continuation:
        raise ValueError("resume requires continuation; cold start forbidden")
    return continuation


def producer_lineage(manifest_path, rows):
    if manifest_path is None:
        return None
    manifest_path = Path(manifest_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("schema") != "semantic-producer-assets-v1" or not document.get("producer"):
        raise ValueError("producer asset manifest schema/name missing")
    assets = {}
    for item in document["assets"]:
        role, path = item["role"], Path(item["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if role in assets or path.suffix not in (".json", ".jsonl") or FT.file_sha256(path) != item["sha256"]:
            raise ValueError("producer asset drift/duplicate/non-JSON")
        assets[role] = {"path": str(path.absolute()), "sha256": item["sha256"]}
    if not {"protected-fit", "protected-dev", "protection-manifest"} <= set(assets):
        raise ValueError("producer must bind protected fit/dev and protection manifest")
    for role, split in (("protected-fit", "train"), ("protected-dev", "dev")):
        path = Path(assets[role]["path"])
        text = path.read_text(encoding="utf-8")
        members = json.loads(text) if path.suffix == ".json" else [json.loads(line) for line in text.splitlines() if line.strip()]
        if not isinstance(members, list):
            raise ValueError("protected member asset must be a JSON array or JSONL")
        ids = {item if isinstance(item, str) else item["id"] for item in members}
        for row in rows:
            if row["label_space"] == "nli3" and row["split"] == split and row.get("source_id", row["id"]) not in ids:
                raise ValueError("public row absent from protected producer member set")
    return {"manifest_path": str(manifest_path.absolute()), "manifest_sha256": FT.file_sha256(manifest_path),
            "producer": document["producer"], "assets": assets,
            "scope": "asset hashes and protected member IDs verified; protection remains producer responsibility"}


def prepare(data, directory, *, producer_manifest=None, fine_manifest=None, allocation=None):
    public = D.load_rows(data) if data is not None else []
    if any(row.get("label_space") != "nli3" for row in public):
        raise ValueError("generic data input is public nli3 only; fine must use immutable qualified source allocation")
    public = [validate_row(row) for row in public]
    if (fine_manifest is None) != (allocation is None):
        raise ValueError("fine manifest and external allocation must be supplied together")
    fine, fine_lineage = qualified_stage_rows(fine_manifest, allocation) if fine_manifest is not None else ([], None)
    rows = public + fine
    if any(row.get("sealed") or row["split"] not in ("train", "dev") for row in rows):
        raise ValueError("prepare accepts only unsealed protected/stage train/dev outputs")
    fit = [row for row in rows if row["split"] == "train"]
    dev = [row for row in rows if row["split"] == "dev"]
    checked_pools(fit, dev, require_both=False, require_coverage=False)
    lineage = producer_lineage(producer_manifest, public)
    directory = Path(directory)
    if directory.exists() or NR._reparse_components(directory) or ".." in directory.parts:
        raise ValueError("prepare destination must be fresh and plain")
    directory.mkdir(parents=True)
    hashes = {}
    for name, pool in (("train", fit), ("dev", dev)):
        target = directory / (name + ".jsonl")
        target.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in pool), encoding="utf-8")
        hashes[name] = FT.file_sha256(target)
    ready = fine_lineage is not None and lineage is not None and all({row["label_space"] for row in pool} == {"nli3", "semantic5"} for pool in (fit, dev))
    meta = {"schema": DATA_SCHEMA, "input_recipe": INPUT_RECIPE, "input_limits": dict(INPUT_LIMITS), "files_sha256": hashes,
            "source_file_sha256": FT.file_sha256(data) if data is not None else None,
            "counts": {"train": len(fit), "dev": len(dev)}, "formal_fit_source_ready": ready,
            "producer_lineage": lineage, "fine_lineage": fine_lineage, "source_code_sha256": source_binding(),
            "raw_pair_sha256": {row["id"]: SR.semantic_pair_pieces(row["pair"], limits=INPUT_LIMITS)["raw_pair_sha256"] for row in rows},
            "raw_preserved": True, "frame_condition": "qualified authorized proxy declaration, or explicit public/inference declaration; not semantic certification"}
    (directory / "prepare.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def read_prepared(directory):
    directory = Path(directory)
    meta = json.loads((directory / "prepare.json").read_text(encoding="utf-8"))
    if meta.get("schema") != DATA_SCHEMA or meta.get("input_recipe") != INPUT_RECIPE or meta.get("input_limits") != INPUT_LIMITS:
        raise ValueError("prepared schema/recipe/input limits mismatch")
    pools = []
    for name in ("train", "dev"):
        path = directory / (name + ".jsonl")
        if FT.file_sha256(path) != meta["files_sha256"][name]:
            raise ValueError("prepared data hash drift")
        pools.append([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
    public = [row for row in pools[0] + pools[1] if row["label_space"] == "nli3"]
    lineage = meta.get("producer_lineage")
    if lineage is not None and producer_lineage(lineage["manifest_path"], public) != lineage:
        raise ValueError("prepared producer lineage drift")
    fine = meta.get("fine_lineage")
    if fine is not None:
        stages, current = qualified_stage_rows(fine["qualified_source"]["manifest_path"], fine["allocation_path"])
        actual_fine = {row["id"]: row for row in pools[0] + pools[1] if row["label_space"] == "semantic5"}
        if current != fine or actual_fine != {row["id"]: row for row in stages}:
            raise ValueError("prepared fine qualification/allocation member coverage or lineage drift")
    elif any(row["label_space"] == "semantic5" for row in pools[0] + pools[1]):
        raise ValueError("prepared fine members require qualification/allocation lineage")
    ready = fine is not None and lineage is not None and all({row["label_space"] for row in pool} == {"nli3", "semantic5"} for pool in pools)
    members = pools[0] + pools[1]
    if meta.get("formal_fit_source_ready") is not ready or meta.get("counts") != {"train": len(pools[0]), "dev": len(pools[1])} \
            or meta.get("raw_pair_sha256") != {row["id"]: SR.semantic_pair_pieces(row["pair"], limits=INPUT_LIMITS)["raw_pair_sha256"] for row in members}:
        raise ValueError("prepared member metadata/readiness drift")
    if meta.get("source_code_sha256") != source_binding():
        raise ValueError("prepared producer/allocator code drift")
    return checked_pools(*pools, require_both=False, require_coverage=False)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data")
    prep.add_argument("--out", required=True)
    prep.add_argument("--producer-manifest")
    prep.add_argument("--fine-manifest")
    prep.add_argument("--allocation")
    init = sub.add_parser("init")
    init.add_argument("--backbone", required=True)
    init.add_argument("--out", required=True)
    init.add_argument("--seed", type=int, choices=SEEDS, default=13)
    init.add_argument("--max-length", type=int, default=384)
    status = sub.add_parser("status")
    status.add_argument("--artifact", required=True)
    for command in ("fit", "budget"):
        child = sub.add_parser(command)
        child.add_argument("--prepared", required=True)
        child.add_argument("--updates", type=int, default=50)
        child.add_argument("--canonical-batch", type=int, default=32)
        child.add_argument("--microbatch", type=int, default=8)
        child.add_argument("--grad-accum", type=int, default=8)
        child.add_argument("--max-length", type=int, default=384)
        child.add_argument("--seed", type=int, choices=SEEDS, default=13)
        child.add_argument("--eval-every-steps", type=int, default=25)
        child.add_argument("--clip-grad-norm", type=float, default=1.0)
        child.add_argument("--tiny-cpu", action="store_true")
        child.add_argument("--mother3", required=True)
        if command == "fit":
            child.add_argument("--out", required=True)
            child.add_argument("--arm", choices=tuple(ARM_CLASSES), required=True)
            child.add_argument("--device", default="cpu")
            child.add_argument("--bf16", choices=("on", "off", "auto"), default="auto")
            child.add_argument("--resume")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "init":
        meta = initialize_untrained_mother(args.backbone, args.out, seed=args.seed, max_length=args.max_length)
        result = {"schema": meta["schema"], "initialization": meta["train"]["initialization"],
                  "effective_config_sha256": meta["backbone_config_sha256"]}
    elif args.command == "prepare":
        result = prepare(args.data, args.out, producer_manifest=args.producer_manifest,
                         fine_manifest=args.fine_manifest, allocation=args.allocation)
    elif args.command == "status":
        scorer, meta, payload = load_semantic_artifact(args.artifact, device="cpu")
        result = {"schema": meta["schema"], "arm": scorer.arm, "classes": list(scorer.classes),
                  "weights_sha256": meta["weights_sha256"], "continuation": "continuation" in payload,
                  "budget": meta["train"]["budget"], "binding_verified": True,
                  "training_ledger_verified": meta["train"].get("purpose") != "untrained-initializer",
                  "initializer_only": meta["train"].get("purpose") == "untrained-initializer",
                  "effective_dtype": str(scorer.model.backbone.config.dtype), "truncation": meta["train"].get("truncation")}
    else:
        if args.updates < 1 or args.max_length < 1 or args.eval_every_steps < 1 \
                or not math.isfinite(args.clip_grad_norm) or args.clip_grad_norm <= 0:
            raise ValueError("updates/window/grid/clip must be positive finite")
        if not args.tiny_cpu and (args.canonical_batch, args.microbatch, args.grad_accum) != (32, 8, 8):
            raise ValueError("main budget is canonical32/micro8/accum8; use explicit --tiny-cpu for mechanics")
        if args.tiny_cpu and args.command == "fit" and args.device != "cpu":
            raise ValueError("tiny-cpu cannot use CUDA")
        fit, dev = read_prepared(args.prepared)
        prepared_meta = json.loads((Path(args.prepared) / "prepare.json").read_text(encoding="utf-8"))
        lineage = prepared_meta.get("producer_lineage")
        if not args.tiny_cpu and (lineage is None or not prepared_meta.get("formal_fit_source_ready")):
            raise ValueError("formal run requires protected public and verified inverse-qualified allocated fine source lineage")
        mother_meta = json.loads((Path(args.mother3) / NR.NLI_META_FILE).read_text(encoding="utf-8"))
        initialization = mother_meta["train"].get("initialization", {})
        if not args.tiny_cpu and (initialization.get("kind") != "raw-backbone-untrained"
                                 or initialization.get("optimizer_updates") != 0 or initialization.get("seed") != args.seed):
            raise ValueError("raw baseline requires explicit untrained same-seed mother3; native warm-start forbidden")
        mother = NR.load_nli_scorer(args.mother3, device="cpu")
        list(balanced_windows(fit, seed=args.seed, updates=1, canonical_batch=args.canonical_batch,
                              microbatch=args.microbatch, grad_accum=args.grad_accum))
        if args.command == "budget":
            result = {"updates": args.updates, "canonical_instances": args.updates * args.canonical_batch,
                      "encoder_instances": args.updates * args.canonical_batch * 2,
                      "supervised_directions": args.updates * args.canonical_batch * 3 // 2,
                      "canonical_batch": args.canonical_batch, "microbatch": args.microbatch,
                      "grad_accum": args.grad_accum, "source_weights": {"public": 0.5, "fine": 0.5},
                      "eval_every_steps": args.eval_every_steps, "clip_grad_norm": args.clip_grad_norm,
                      "mechanics_only": args.tiny_cpu, "producer_lineage": lineage,
                      "truncation": truncation_profile(mother.tokenizer, oriented_rows(fit), args.max_length)}
        else:
            device = FT.resolve_device(args.device)
            bf16 = FT.resolve_bf16(args.bf16, device)
            model = initialize_arm(mother.model, args.arm)
            resume = resume_continuation(args.resume) if args.resume else None
            training = train_semantic(model, mother.tokenizer, fit, arm=args.arm, dev_rows=dev,
                                       seed=args.seed, updates=args.updates, canonical_batch=args.canonical_batch,
                                       microbatch=args.microbatch, grad_accum=args.grad_accum, max_length=args.max_length,
                                       device=device, bf16=bf16, resume_state=resume, eval_every_steps=args.eval_every_steps,
                                       clip_grad_norm=args.clip_grad_norm, producer_lineage=lineage,
                                       prepared_directory=args.prepared, mechanics_only=args.tiny_cpu)
            scorer = SemanticScorer(model=model, tokenizer=mother.tokenizer, arm=args.arm, max_length=args.max_length,
                                     device=str(device))
            training["mother_weights_sha256"] = mother_meta["weights_sha256"]
            training["initialization"] = initialization
            training["mechanics_only"] = args.tiny_cpu
            result = save_semantic_artifact(scorer, args.out, backbone_dir=mother_meta["backbone_path"], training=training)
            reloaded = load_semantic_artifact(args.out, device="cpu")[0]
            probe = dev[:2]
            difference = float(np.max(np.abs(scorer.predict_rows(probe) - reloaded.predict_rows(probe))))
            if not np.isfinite(difference) or difference > 1e-6:
                raise ValueError("CPU reload parity failed")
            result = {"schema": result["schema"], "arm": args.arm, "budget": training["budget"],
                      "selection": training["selection"], "guard": training["guard"], "truncation": training["truncation"],
                      "reload_max_abs_diff": difference, "mechanics_only": args.tiny_cpu}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
