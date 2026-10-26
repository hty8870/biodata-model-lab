"""复用既有训练与指标原语：D5 双向融合 posthoc 分析与 C3 单源对照薄入口。

C3 只做数据匹配、同配方训练和交叉评测编排；数据/训练/工件/指标分别复用
canonical prepared、qualified source_rows、public_benchmarks 已完成 run、
nli_reference.train_nli/save_nli_artifact、finetune.build_joint_model 与
eval_nli_metrics，不复制第二套引擎，也不把来源效用写成教师质量认证。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dataset_recommender.agent import semantic_relations as SR
from research.memory_relations import metrics as M


FIVE = SR.SEMANTIC_RELATIONS
THREE = SR.NLI_CLASSES


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fuse_five(forward, reverse):
    forward = SR.validate_probabilities(forward)
    reverse = SR.swap_probabilities(reverse)
    return SR.validate_probabilities(tuple((a + b) / 2 for a, b in zip(forward, reverse)))


def _hard(probabilities):
    return int(np.argmax(probabilities)) if probabilities is not None else -1


def _direction(row, name):
    item = row.get(name, {})
    status, probabilities = M.probability_row(item.get("probabilities"), len(FIVE))
    if item.get("status") != "ok" or status != "ok":
        return None
    return probabilities.tolist()


def d5_rows(rows):
    seen = set()
    result = []
    for row in rows:
        root = row["root_id"]
        if root in seen:
            raise ValueError("D5输入每个独立根只能有一条预测")
        seen.add(root)
        gold = row["gold"]
        if row["reverse_gold"] != SR.swap_relation(gold):
            raise ValueError("D5正反向gold不符合共同交换映射")
        forward, reverse = (_direction(row, name) for name in ("forward", "reverse"))
        fused = fuse_five(forward, reverse) if forward is not None and reverse is not None else None
        raw_f, raw_r = _hard(forward), _hard(reverse)
        fused_f = _hard(fused)
        fused_r = _hard(SR.swap_probabilities(fused)) if fused is not None else -1
        valid = forward is not None and reverse is not None
        raw_consistent = valid and raw_r == FIVE.index(SR.swap_relation(FIVE[raw_f]))
        tie = bool(fused is not None and np.count_nonzero(np.asarray(fused) == max(fused)) > 1)
        result.append({
            "id": row["id"], "root_id": root, "cluster_id": row["cluster_id"],
            "split": row["split"], "gold": gold, "reverse_gold": row["reverse_gold"],
            "language": row.get("language"), "family": row.get("family"),
            "complete": row.get("input_audit", {}).get("complete"),
            "forward": raw_f, "reverse": raw_r, "d5": fused_f, "d5_reverse": fused_r,
            "d5_probabilities": fused, "both_probabilities_available": valid,
            "raw_swap_consistent": bool(raw_consistent),
            "raw_both_correct": bool(raw_f == FIVE.index(gold) and raw_r == FIVE.index(row["reverse_gold"])),
            "d5_both_correct": bool(fused_f == FIVE.index(gold) and fused_r == FIVE.index(row["reverse_gold"])),
            "d5_argmax_tie": tie,
            "raw_swap_l1": float(np.abs(np.asarray(forward) - np.asarray(SR.swap_probabilities(reverse))).sum()) if valid else None,
        })
    if not result:
        raise ValueError("D5输入为空")
    return result


def _d5_summary(rows):
    truth = np.asarray([FIVE.index(row["gold"]) for row in rows])
    baseline = M.classification_summary(truth, [row["forward"] for row in rows], classes=FIVE)
    fused = M.classification_summary(truth, [row["d5"] for row in rows], classes=FIVE)
    available = [row for row in rows if row["both_probabilities_available"]]
    return {
        "roots": len(rows), "clusters": len({row["cluster_id"] for row in rows}),
        "baseline": baseline, "d5": fused,
        "delta_accuracy": fused["accuracy"] - baseline["accuracy"],
        "delta_macro_f1": fused["macro_f1"] - baseline["macro_f1"],
        "both_probabilities_available": len(available),
        "raw_swap_consistency_all_roots": sum(row["raw_swap_consistent"] for row in rows) / len(rows),
        "raw_swap_consistency_available": sum(row["raw_swap_consistent"] for row in available) / len(available) if available else None,
        "raw_both_correct": sum(row["raw_both_correct"] for row in rows) / len(rows),
        "d5_both_correct": sum(row["d5_both_correct"] for row in rows) / len(rows),
        "d5_argmax_ties": sum(row["d5_argmax_tie"] for row in rows),
        "raw_swap_l1_available": float(np.mean([row["raw_swap_l1"] for row in available])) if available else None,
    }


def summarize_d5(rows, *, samples=1000, seed=20261023):
    result = _d5_summary(rows)
    truth = np.asarray([FIVE.index(row["gold"]) for row in rows])
    baseline = np.asarray([row["forward"] for row in rows])
    fused = np.asarray([row["d5"] for row in rows])
    result["paired_cluster_delta"] = {}
    for name in ("accuracy", "macro_f1"):
        def difference(index, weights):
            return (M.classification_summary(truth[index], fused[index], classes=FIVE, sample_weight=weights)[name]
                    - M.classification_summary(truth[index], baseline[index], classes=FIVE, sample_weight=weights)[name])
        result["paired_cluster_delta"][name] = M.cluster_bootstrap(
            [row["cluster_id"] for row in rows], difference, samples=samples, seed=seed, weighting="row")
    result["splits"] = {split: _d5_summary([row for row in rows if row["split"] == split])
                        for split in sorted({row["split"] for row in rows})}
    return result


def analyze_d5(index_path, output, *, samples=1000):
    index_path, output = Path(index_path), Path(output)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    models = [model for model in index["models"] if model["arm"] == "c5"]
    if not models:
        raise ValueError("索引没有C5工件")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "c5-bidirectional-fusion-v1", "analysis": "posthoc_existing_fine_final",
              "index_sha256": file_sha256(index_path), "fusion": "mean_forward_and_swapped_reverse",
              "argmax_tie_rule": "first_existing_class_order; probability equivariance does not guarantee hard equivariance at ties",
              "new_training": 0, "new_model_inference": 0, "bootstrap_samples": samples, "models": []}
    cohort = None
    for model in models:
        source = model["files"]["predictions.jsonl"]
        if file_sha256(source["path"]) != source["sha256"]:
            raise ValueError("原逐题预测SHA不符")
        rows = d5_rows(read_jsonl(source["path"]))
        signature = sorted((row["root_id"], row["cluster_id"], row["split"], row["gold"]) for row in rows)
        if cohort is not None and signature != cohort:
            raise ValueError("模型之间评测根/簇/标签不一致")
        cohort = signature
        summary = summarize_d5(rows, samples=samples)
        if not np.isclose(summary["baseline"]["accuracy"], model["accuracy"], rtol=0, atol=1e-12):
            raise ValueError("C5基线准确率未复现")
        if not np.isclose(summary["baseline"]["macro_f1"], model["macro_f1"], rtol=0, atol=1e-12):
            raise ValueError("C5基线macro-F1未复现")
        target = output / f"{model['model_id']}.jsonl"
        write_jsonl(target, rows)
        report["models"].append({"model_id": model["model_id"], "version": model["version"], "seed": model["seed"],
                                 "source_sha256": source["sha256"], "predictions_sha256": file_sha256(target), **summary})
    write_json(output / "summary.json", report)
    return report


C3_PREP_SCHEMA = "memory-relations-source-control-c3-prep-v1"
C3_TRAIN_SCHEMA = "memory-relations-source-control-c3-train-v1"
C3_EVAL_SCHEMA = "memory-relations-source-control-c3-eval-v1"
C3_PURPOSE = {
    "experiment": "new_single_source_c3",
    "prior_fine_final": "exposed_exploratory_only_posthoc",
    "teacher_quality": "not_certified",
    "claim_scope": "matched_budget_source_utility_not_teacher_accuracy",
}
C3_DEFAULTS = {
    "canonical_dir": "artifacts/worktrees/semantic-formal-v12-20261016/research/memory_relations/runs/v34-v3-joint-preparation-20261019/prepared",
    "fine_source": "artifacts/worktrees/semantic-teacher-bind-20261014/research/memory_relations/runs/v7-fine-development-20261014/teacher-completion-20261015-c71/delivery/source_rows.jsonl",
    "public_data_run": "artifacts/worktrees/semantic-assets-20261014/research/memory_relations/runs/v6-public-assets-20261014/v4-public-data-close-20261013",
    "fine_cohort": "artifacts/v55-final-preflight-20261021/fine-final-cohort.json",
    "embedding_dir": "models/embeddings/paraphrase-multilingual-MiniLM-L12-v2",
    "run_dir": "research/memory_relations/runs/source-control-20261023",
}
ADAPTER_RECIPE = {
    "role_order": "candidate=new/hypothesis; existing=[frame+old]/premise",
    "serializer": "canonical semantic pair entry compact JSON (frame -> old -> new), no second text-only projection",
    "frame": "frame.mode always preserved; background block included in existing payload only when non-empty; original JSON entry fields are not dropped",
    "official_native": "premise/hypothesis are wrapped as kind=proposition pair entries, then the same function is used",
    "truncation": "right truncation at max_length; measured per role, never silently dropped",
}


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _default_path(key):
    return _repo_root() / C3_DEFAULTS[key]


def _resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else _repo_root() / path


def _stream_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(f"{path}:{number} 不是合法 JSONL") from None


def _compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def request_for_pair(pair):
    """唯一 pair→NLI request 适配：new 为 candidate，frame+old 为单一 existing。

    官方原生 premise/hypothesis 先包成 kind=proposition 的 pair，再走本函数；两侧都保留
    canonical pair 的全部可见字段（kind/text/title/trigger/permanence/payload），不另做
    text-only 投影，也不新增第二份序列化。
    """
    pair = SR.validate_semantic_pair(pair)
    frame, old, new = pair["frame"], pair["old"], pair["new"]
    premise = {"frame": frame, "old": old} if frame["background"] else old
    return {
        "candidate": {"text": _compact(new), "trigger": ""},
        "existing": [{"text": _compact(premise), "trigger": ""}],
        "evidence": [],
    }


def official_native_pair(premise, hypothesis):
    return SR.validate_semantic_pair({
        "frame": {"mode": "same_domain", "background": ""},
        "old": {"kind": "proposition", "text": premise},
        "new": {"kind": "proposition", "text": hypothesis},
    })


def _pair_signature(pair):
    return _compact(SR.validate_semantic_pair(pair))


def _load_canonical_official(canonical_dir):
    directory = Path(canonical_dir)
    result = {"train": [], "dev": []}
    for filename, split in (("train.jsonl", "train"), ("dev.jsonl", "dev")):
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"canonical {split} 不存在：{path}")
        rows = []
        for row in _stream_jsonl(path):
            if row.get("label_space") != "nli3":
                continue
            lineage = row.get("lineage") or {}
            if lineage.get("dataset") != "ocnli" or row.get("split") != split:
                continue
            if row.get("label") not in THREE:
                raise ValueError(f"canonical {split} 出现非三类标签：{row.get('label')!r}")
            rows.append(dict(row, pair=SR.validate_semantic_pair(row["pair"])))
        result[split] = rows
        result[f"{split}_path"] = path
    return result


def _load_fine_source(source_rows):
    items, roots = [], set()
    for row in _stream_jsonl(source_rows):
        if row.get("label_space") != "semantic5":
            raise ValueError(f"fine source 行标签空间不是 semantic5：{row.get('id')}")
        label5 = row.get("label")
        if label5 not in SR.SEMANTIC_TO_NLI:
            raise ValueError(f"fine source 五类标签未知：{label5!r}")
        root = row.get("root_id")
        if not isinstance(root, str) or not root.strip():
            raise ValueError(f"fine source 行缺少 root_id：{row.get('id')}")
        if root in roots:
            raise ValueError(f"fine source 每根只允许一条，重复 root：{root}")
        roots.add(root)
        items.append({
            "row": row,
            "pair": SR.validate_semantic_pair(row["pair"]),
            "label5": label5,
            "label": SR.SEMANTIC_TO_NLI[label5],
            "language": (row.get("phenomena") or {}).get("language"),
        })
    if not items:
        raise ValueError("fine source 为空")
    return items


def _fine_train_selection(items):
    train = [item for item in items if item["row"].get("split") == "train"]
    languages = Counter(item["language"] for item in train)
    missing_language = sum(1 for item in train if not isinstance(item["language"], str) or not item["language"].strip())
    selected = [item for item in train
                if isinstance(item["language"], str) and item["language"].strip() == "zh"]
    for item in selected:
        if (item["row"].get("teacher_evidence") or {}).get("swapped") is True:
            raise ValueError(f"fine train 根 {item['row']['root_id']} 是反向样本：不作为独立样本")
    if not selected:
        raise ValueError("fine 中文 train 根为空")
    counts = Counter(item["label"] for item in selected)
    missing = [label for label in THREE if counts[label] == 0]
    if missing:
        raise ValueError(f"fine 中文 train 未覆盖三类，缺 {missing}")
    return selected, {
        "train_rows": len(train),
        "train_languages": dict(languages),
        "excluded_missing_language": missing_language,
        "excluded_non_zh": sum(value for key, value in languages.items()
                               if isinstance(key, str) and key.strip() and key != "zh"),
        "selected_rows": len(selected),
        "selected_roots": len({item["row"]["root_id"] for item in selected}),
        "selected_clusters": len({item["row"]["cluster_id"] for item in selected}),
        "selected_classes": {label: counts[label] for label in THREE},
        "selected_label5": dict(Counter(item["label5"] for item in selected)),
        "selected_modes": dict(Counter(item["pair"]["frame"]["mode"] for item in selected)),
        "selected_roles": dict(Counter(((item["row"].get("phenomena") or {}).get("author") or {}).get("role")
                                       for item in selected)),
        "selected_label_origin": dict(Counter(item["row"].get("label_origin") for item in selected)),
        "selected_teacher_fidelity": dict(Counter(
            ((item["row"].get("NL_fidelity_origin") or {}).get("kind")) for item in selected)),
        "selected_adjudication": dict(Counter(
            ((item["row"].get("adjudication_origin") or {}).get("kind")) for item in selected)),
        "selected_swapped": dict(Counter((item["row"].get("teacher_evidence") or {}).get("swapped")
                                         for item in selected)),
        "language_basis": "source_rows 每行显式 phenomena.language；只取 zh，不按文本猜语种",
        "human_confirmation": False,
    }


def _select_official_match(rows, wanted, *, seed):
    """每根先取一条确定性代表行，再按类做 seed 定序的无放回抽样；根不复用。"""
    by_root = {}
    for row in rows:
        root = row["root_id"]
        if root not in by_root or row["id"] < by_root[root]["id"]:
            by_root[root] = row
    representatives = list(by_root.values())
    by_class = {label: [] for label in THREE}
    for row in representatives:
        by_class[row["label"]].append(row)
    selected, available = [], {}
    for label in THREE:
        pool = sorted(by_class[label], key=lambda row: row["id"])
        available[label] = len(pool)
        need = int(wanted[label])
        if len(pool) < need:
            raise ValueError(f"官方 {label} 根不足：需要 {need}，只有 {len(pool)}")
        ranked = sorted(pool, key=lambda row: hashlib.sha256(
            f"{seed}\x1fofficial\x1f{label}\x1f{row['root_id']}".encode("utf-8")).hexdigest())
        selected.extend(ranked[:need])
    roots = [row["root_id"] for row in selected]
    if len(roots) != len(set(roots)):
        raise ValueError("官方匹配出现重复根")
    return selected, {
        "root_candidates_by_class": available,
        "representative_rows": len(representatives),
        "representative_roots": len(by_root),
        "selection_rule": "root representative = smallest canonical id; per-class rank = sha256(seed|official|class|root); first k",
        "seed": int(seed),
    }


def _load_official_dev(public_data_run):
    from research.memory_relations.public_benchmarks import evaluate as E

    run_dir = Path(public_data_run)
    loaded = E.load_data_run(run_dir)
    raw = [row for row in loaded["rows"]
           if row["dataset"] == "ocnli" and row["official_split"] == "dev"]
    if not raw:
        raise ValueError("public data run 没有 OCNLI dev 行")
    scored, missing = [], 0
    for record in raw:
        if record.get("gold_label") is None:
            missing += 1
            continue
        if record["gold_label"] not in THREE:
            raise ValueError(f"OCNLI dev gold 不在三类内：{record['gold_label']!r}")
        pair = official_native_pair(record["premise"], record["hypothesis"])
        scored.append({
            "id": record["id"], "root_id": record["id"], "cluster_id": record["group_id"],
            "group_id": record["group_id"], "label": record["gold_label"], "pair": pair,
            "source_split": record["official_split"],
            "gold_status": (record.get("metadata") or {}).get("gold_status"),
        })
    files = [item for item in loaded["files"] if item["dataset"] == "ocnli" and item["official_split"] == "dev"]
    if not files:
        raise ValueError("public data manifest 缺少 OCNLI dev 文件声明")
    return scored, {
        "public_data_run": str(run_dir),
        "manifest_sha256": loaded["manifest_sha256"],
        "file": files[0],
        "rows": len(raw),
        "scored": len(scored),
        "missing_gold_excluded": missing,
        "groups": len({item["group_id"] for item in scored}),
        "groups_all_rows": len({item["group_id"] for item in raw}),
        "classes": {label: sum(1 for item in scored if item["label"] == label) for label in THREE},
        "gold_status": dict(Counter(item["gold_status"] for item in scored)),
        "language": "zh",
        "language_basis": "study-v4.json datasets.ocnli.language=zh + sources.json OCNLI dataset + canonical lineage.dataset=ocnli",
        "human_confirmation": False,
    }


def _load_fine_final(cohort_path):
    from research.memory_relations import semantic_data as D

    path = Path(cohort_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "semantic-evaluation-cohort-v1" or not document.get("sources"):
        raise ValueError("fine cohort schema 不符")
    items, sources, roots = [], [], set()
    for entry in document["sources"]:
        manifest = Path(entry["source_manifest"])
        scope = Path(entry["construction_scope"])
        bundle = D.load_qualified_source(manifest, construction_scope=scope, run_id=entry["run_id"])
        for row in bundle["rows"]:
            label5 = row.get("label")
            if label5 not in SR.SEMANTIC_TO_NLI:
                raise ValueError(f"fine final 五类标签未知：{label5!r}")
            root = row["root_id"]
            if root in roots:
                raise ValueError(f"fine final 每根只允许一条，重复 root：{root}")
            roots.add(root)
            items.append({
                "id": row["id"], "root_id": root, "cluster_id": row["cluster_id"],
                "label5": label5, "label": SR.SEMANTIC_TO_NLI[label5],
                "pair": SR.validate_semantic_pair(row["pair"]),
                "source_split": row["split"],
                "language": (row.get("phenomena") or {}).get("language"),
                "row": row,
            })
        lineage = bundle.get("lineage") or {}
        sources.append({
            "source_id": entry["source_id"],
            "source_manifest": str(manifest),
            "manifest_sha256": lineage.get("manifest_sha256"),
            "qualified_rows": lineage.get("qualified_rows"),
            "proxy_only": lineage.get("proxy_only"),
            "human_or_backend_certification": lineage.get("human_or_backend_certification"),
        })
    if not items:
        raise ValueError("fine final cohort 为空")
    return items, {
        "cohort": str(path),
        "cohort_sha256": file_sha256(path),
        "sources": sources,
        "rows": len(items),
        "roots": len(roots),
        "clusters": len({item["cluster_id"] for item in items}),
        "classes": {label: sum(1 for item in items if item["label"] == label) for label in THREE},
        "label5": dict(Counter(item["label5"] for item in items)),
        "splits": dict(Counter(item["source_split"] for item in items)),
        "languages": dict(Counter(item["language"] for item in items)),
        "roles": dict(Counter(((item["row"].get("phenomena") or {}).get("author") or {}).get("role")
                              for item in items)),
        "modes": dict(Counter(item["pair"]["frame"]["mode"] for item in items)),
        "language_basis": "source_rows 每行显式 phenomena.language；不按文本猜语种",
        "human_confirmation": False,
    }


def _checked_train_rows(rows, *, where):
    if not rows:
        raise ValueError(f"{where} 训练行为空")
    counts = Counter()
    for index, row in enumerate(rows):
        if not isinstance(row.get("request"), dict):
            raise ValueError(f"{where} 第 {index} 行缺少 request")
        if row.get("split") != "train" or row.get("sealed"):
            raise ValueError(f"{where} 第 {index} 行不是纯 train：split={row.get('split')!r}")
        if row.get("label") not in THREE:
            raise ValueError(f"{where} 第 {index} 行标签不在三类：{row.get('label')!r}")
        counts[row["label"]] += 1
    missing = [label for label in THREE if counts[label] == 0]
    if missing:
        raise ValueError(f"{where} 未覆盖三类，缺 {missing}")
    return counts


def _checked_eval_rows(rows, *, where):
    if not rows:
        raise ValueError(f"{where} 评测行为空")
    ids, roots = set(), set()
    counts = Counter()
    for index, row in enumerate(rows):
        if not isinstance(row.get("request"), dict):
            raise ValueError(f"{where} 第 {index} 行缺少 request")
        if row.get("split") == "train":
            raise ValueError(f"{where} 第 {index} 行标记为 train：拒绝把训练行当评测")
        if row.get("label") not in THREE:
            raise ValueError(f"{where} 第 {index} 行标签不在三类：{row.get('label')!r}")
        if row.get("id") in ids:
            raise ValueError(f"{where} 第 {index} 行 id 重复：{row.get('id')}")
        ids.add(row["id"])
        if row.get("source") == "fine_final":
            if row.get("root_id") in roots:
                raise ValueError(f"{where} fine final 根重复：{row.get('root_id')}")
            roots.add(row["root_id"])
        counts[row["label"]] += 1
    missing = [label for label in THREE if counts[label] == 0]
    if missing:
        raise ValueError(f"{where} 未覆盖三类，缺 {missing}")
    return counts


def common_updates(n, *, batch_size, grad_accum, epochs=20):
    """两臂共同预算：20 个完整遍历的实例量 / 有效 batch，向下取整并至少 1 update。"""
    if min(int(n), int(batch_size), int(grad_accum), int(epochs)) < 1:
        raise ValueError("预算参数必须为正")
    effective = int(batch_size) * int(grad_accum)
    return max(1, (int(epochs) * int(n)) // effective)


def _official_train_manifest(rows, selected, stats, canonical):
    return {
        "source": "ocnli_train50k",
        "canonical_train": str(canonical["train_path"]),
        "canonical_train_sha256": file_sha256(canonical["train_path"]),
        "canonical_dev": str(canonical["dev_path"]),
        "canonical_dev_sha256": file_sha256(canonical["dev_path"]),
        "rows_seen": len(rows),
        "official_splits": dict(Counter(row["lineage"].get("official_split") for row in rows)),
        "roles": dict(Counter(row["lineage"].get("protected_role") for row in rows)),
        "label_origin": dict(Counter(row.get("label_origin") for row in rows)),
        "gold_status": dict(Counter(row["lineage"].get("gold_status") for row in rows)),
        "label_method": dict(Counter(row["lineage"].get("label_method") for row in rows)),
        **stats,
        "selected_rows": len(selected),
        "selected_roots": len({row["root_id"] for row in selected}),
        "selected_clusters": len({row["cluster_id"] for row in selected}),
        "selected_classes": {label: sum(1 for row in selected if row["label"] == label) for label in THREE},
        "selected_roles": dict(Counter(row["lineage"].get("protected_role") for row in selected)),
        "selected_gold_status": dict(Counter(row["lineage"].get("gold_status") for row in selected)),
        "selected_label_method": dict(Counter(row["lineage"].get("label_method") for row in selected)),
        "selected_ids_sha256": hashlib.sha256("\n".join(sorted(row["id"] for row in selected)).encode("utf-8")).hexdigest(),
        "language": "zh",
        "language_basis": "OCNLI 数据集显式语言声明（study-v4.json/sources.json）+ canonical lineage.dataset=ocnli；未做文本语种猜测",
        "human_confirmation": False,
    }


def prepare_c3(*, out_dir, canonical_dir=None, fine_source=None, public_data_run=None,
               fine_cohort=None, seed=13, budget_epochs=20, batch_size=8, grad_accum=2,
               max_length=384, official_dev_rows=None, fine_final_rows=None):
    """C3 两源匹配数据与小 manifest；不训练、不读大池、不复制原数据。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    fine_source_path = Path(fine_source or _default_path("fine_source"))
    study_path = _repo_root() / "research/memory_relations/public_benchmarks/study-v4.json"
    sources_path = _repo_root() / "research/memory_relations/public_benchmarks/sources.json"
    canonical = _load_canonical_official(canonical_dir or _default_path("canonical_dir"))
    fine_items = _load_fine_source(fine_source_path)
    fine_selected, fine_stats = _fine_train_selection(fine_items)
    wanted = {label: fine_stats["selected_classes"][label] for label in THREE}
    official_selected, official_stats = _select_official_match(canonical["train"], wanted, seed=seed)
    if len(official_selected) != len(fine_selected):
        raise ValueError("两源匹配后行数不同")

    if official_dev_rows is None:
        official_dev, official_dev_stats = _load_official_dev(public_data_run or _default_path("public_data_run"))
    else:
        official_dev, official_dev_stats = official_dev_rows
    if fine_final_rows is None:
        fine_final, fine_final_stats = _load_fine_final(fine_cohort or _default_path("fine_cohort"))
    else:
        fine_final, fine_final_stats = fine_final_rows

    fine_dev = [item for item in fine_items if item["row"].get("split") == "dev"]
    official_dev_overlap = {
        "pair": len({_pair_signature(item["pair"]) for item in official_selected}
                    & {_pair_signature(item["pair"]) for item in official_dev}),
        "group": len({row["lineage"].get("group_id") for row in official_selected}
                     & {item["group_id"] for item in official_dev}),
    }
    fine_final_overlap = {
        "root": len({item["row"]["root_id"] for item in fine_selected}
                    & {item["root_id"] for item in fine_final}),
        "pair": len({_pair_signature(item["pair"]) for item in fine_selected}
                    & {_pair_signature(item["pair"]) for item in fine_final}),
        "cluster": len({item["row"]["cluster_id"] for item in fine_selected}
                       & {item["cluster_id"] for item in fine_final}),
    }
    fine_dev_overlap = {
        "root": len({item["row"]["root_id"] for item in fine_selected}
                    & {item["row"]["root_id"] for item in fine_dev}),
        "pair": len({_pair_signature(item["pair"]) for item in fine_selected}
                    & {_pair_signature(item["pair"]) for item in fine_dev}),
    }
    canonical_dev_overlap = {
        "root": len({row["root_id"] for row in canonical["train"]} & {row["root_id"] for row in canonical["dev"]}),
        "pair": len({_pair_signature(row["pair"]) for row in canonical["train"]}
                    & {_pair_signature(row["pair"]) for row in canonical["dev"]}),
        "group": len({row["lineage"].get("group_id") for row in canonical["train"]}
                     & {row["lineage"].get("group_id") for row in canonical["dev"]}),
    }
    if any(official_dev_overlap.values()):
        raise ValueError(f"官方匹配 train 与 OCNLI dev 有 exact overlap：{official_dev_overlap}")
    if any(fine_final_overlap.values()):
        raise ValueError(f"fine train 与 fine final 有 exact overlap：{fine_final_overlap}")
    if any(fine_dev_overlap.values()):
        raise ValueError(f"fine train 与 fine dev 有 exact overlap：{fine_dev_overlap}")

    def official_train_row(row):
        return {
            "id": row["id"], "root_id": row["root_id"], "cluster_id": row["cluster_id"],
            "split": "train", "source": "ocnli_train50k", "source_split": row["split"],
            "label": row["label"], "label5": None, "language": "zh",
            "language_basis": "dataset_declaration_not_text_guess",
            "official_split": row["lineage"].get("official_split"),
            "gold_status": row["lineage"].get("gold_status"),
            "request": request_for_pair(row["pair"]),
        }

    def fine_train_row(item):
        row = item["row"]
        return {
            "id": row["id"], "root_id": row["root_id"], "cluster_id": row["cluster_id"],
            "split": "train", "source": "fine_zh_train", "source_split": row["split"],
            "label": item["label"], "label5": item["label5"], "language": item["language"],
            "language_basis": "phenomena.language",
            "label_origin": row.get("label_origin"),
            "request": request_for_pair(item["pair"]),
        }

    def official_dev_row(item):
        return {
            "id": item["id"], "root_id": item["root_id"], "cluster_id": item["cluster_id"],
            "split": "dev", "source": "ocnli_dev", "source_split": item["source_split"],
            "label": item["label"], "group_id": item["group_id"], "language": "zh",
            "language_basis": "dataset_declaration_not_text_guess",
            "gold_status": item["gold_status"], "request": request_for_pair(item["pair"]),
        }

    def fine_final_row(item):
        return {
            "id": item["id"], "root_id": item["root_id"], "cluster_id": item["cluster_id"],
            "split": "dev", "source": "fine_final", "source_split": item["source_split"],
            "label": item["label"], "label5": item["label5"], "language": item["language"],
            "language_basis": "phenomena.language", "request": request_for_pair(item["pair"]),
        }

    official_rows = [official_train_row(row) for row in official_selected]
    fine_rows = [fine_train_row(item) for item in fine_selected]
    official_eval_rows = [official_dev_row(item) for item in official_dev]
    fine_eval_rows = [fine_final_row(item) for item in fine_final]
    _checked_train_rows(official_rows, where="official")
    _checked_train_rows(fine_rows, where="fine")
    _checked_eval_rows(official_eval_rows, where="ocnli_dev")
    _checked_eval_rows(fine_eval_rows, where="fine_final")
    n = len(official_rows)
    updates = common_updates(n, batch_size=batch_size, grad_accum=grad_accum, epochs=budget_epochs)

    files = {}
    for name, rows in (("official_train.jsonl", official_rows), ("fine_train.jsonl", fine_rows),
                       ("ocnli_dev.jsonl", official_eval_rows), ("fine_final.jsonl", fine_eval_rows)):
        target = out / name
        write_jsonl(target, rows)
        files[name] = {"sha256": file_sha256(target), "rows": len(rows)}

    manifest = {
        "schema": C3_PREP_SCHEMA,
        "purpose": C3_PURPOSE,
        "seed": int(seed),
        "adapter": ADAPTER_RECIPE,
        "language": {
            "official": {
                "value": "zh", "basis": "dataset_declaration_not_text_guess",
                "study_v4": str(study_path), "study_v4_sha256": file_sha256(study_path),
                "sources": str(sources_path), "sources_sha256": file_sha256(sources_path),
                "human_confirmation": False,
            },
            "fine": {
                "value": "zh", "basis": "per_row_phenomena_language_not_text_guess",
                "source_rows": str(fine_source_path), "source_rows_sha256": file_sha256(fine_source_path),
                "human_confirmation": False,
            },
        },
        "budget": {
            "epochs": int(budget_epochs), "batch_size": int(batch_size),
            "grad_accum": int(grad_accum), "effective_batch": int(batch_size) * int(grad_accum),
            "max_length": int(max_length), "common_updates": updates,
            "instances_per_arm": updates * int(batch_size) * int(grad_accum),
            "epochs_equivalent": updates * int(batch_size) * int(grad_accum) / n,
            "formula": "max(1, floor(epochs*n/(batch_size*grad_accum)))",
            "checkpoint": "final (fixed common updates)",
            "bf16": "cuda_on", "optimizer": "AdamW backbone 2e-5 / head 1e-3 / wd 0.01",
        },
        "official_train": _official_train_manifest(canonical["train"], official_selected, official_stats, canonical),
        "fine_train": {
            "source": "fine_source_rows",
            "path": str(fine_source_path),
            "sha256": file_sha256(fine_source_path),
            "total_rows": len(fine_items),
            "split_counts": dict(Counter(item["row"].get("split") for item in fine_items)),
            **fine_stats,
        },
        "ocnli_dev": official_dev_stats,
        "fine_final": fine_final_stats,
        "overlaps": {
            "official_train_vs_ocnli_dev": official_dev_overlap,
            "fine_train_vs_fine_final": fine_final_overlap,
            "fine_train_vs_fine_source_dev": fine_dev_overlap,
            "canonical_ocnli_train_vs_canonical_dev": canonical_dev_overlap,
        },
        "files": files,
        "notes": [
            "两源只做 forward CE；fine 每根一条，不改写、不反向扩增。",
            "官方 OCNLI dev 的 50 条无 gold 行如实排除，不逆推反向 gold。",
            "官方 train 与 fine 中文 train 逐类同 n；不宣称每类 1/3。",
            "旧 fine final 只作 posthoc 交叉评测，不进入 fit、选点或调参。",
            "来源效用不等于教师标注质量；不认证教师质量或人工确认。",
            "max_length=384 的实际截断在 train/eval 的 truncation_profile 中逐 role 报告。",
        ],
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def _snapshot_state(model):
    import torch
    return {name: tensor.detach().to("cpu").clone()
            for name, tensor in model.state_dict().items()}


def _training_device(name):
    from research.memory_relations import finetune as FT
    device = FT.resolve_device(name)
    if str(name).startswith("cuda") and device.type != "cuda":
        raise RuntimeError("请求 CUDA 但解析结果不是 CUDA：拒绝回退 CPU")
    return device


def _load_tokenizer(embedding_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(embedding_dir), local_files_only=True)


def _train_meta(*, arm, data_path, rows, training, seed, init_sha, max_length, batch_size,
                grad_accum, updates, prep_manifest=None):
    return {
        "schema": C3_TRAIN_SCHEMA, "arm": arm,
        "purpose": C3_PURPOSE,
        "data": {"path": str(data_path), "sha256": file_sha256(data_path), "rows": len(rows),
                 "classes": dict(Counter(row["label"] for row in rows)),
                 "roots": len({row["root_id"] for row in rows}),
                 "clusters": len({row["cluster_id"] for row in rows})},
        "training": training["data"],
        "budget": training["budget"],
        "selection": training["selection"],
        "device_log": training["device_log"],
        "guard": training["guard"],
        "truncation": training["truncation"],
        "recipe": {"adapter": ADAPTER_RECIPE, "seed": int(seed), "max_length": int(max_length),
                   "batch_size": int(batch_size), "grad_accum": int(grad_accum),
                   "common_updates": int(updates), "checkpoint_selection": "final",
                   "objective": "forward cross-entropy only", "bf16": True},
        "initial_state_sha256": init_sha,
        "prep_manifest": ({"path": str(prep_manifest), "sha256": file_sha256(prep_manifest)}
                          if prep_manifest else None),
    }


def train_pair(*, official_data, fine_data, out_dir, embedding_dir=None, device="cuda",
               seed=13, batch_size=8, grad_accum=2, max_length=384, budget_epochs=20,
               threads=8, prep_manifest=None):
    """同初始 tensor、同配方训练两条单源臂；只保存两个正式工件与小型训练日志。"""
    import gc
    import time

    import torch

    from research.memory_relations import finetune as FT
    from research.memory_relations import nli_reference as NR
    from research.memory_relations import semantic_model as SM

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    official_rows = read_jsonl(official_data)
    fine_rows = read_jsonl(fine_data)
    _checked_train_rows(official_rows, where="official")
    _checked_train_rows(fine_rows, where="fine")
    if len(official_rows) != len(fine_rows):
        raise ValueError("两臂训练行数不同：无法共享共同 update 预算")
    n = len(official_rows)
    updates = common_updates(n, batch_size=batch_size, grad_accum=grad_accum, epochs=budget_epochs)
    device_obj = _training_device(device)
    bf16 = FT.resolve_bf16("on" if device_obj.type == "cuda" else "off", device_obj)
    if device_obj.type == "cuda" and not bf16:
        raise ValueError("正式 CUDA 训练要求 BF16")
    if int(threads) < 1:
        raise ValueError("threads 必须为正")
    torch.set_num_threads(int(threads))
    embedding_dir = Path(embedding_dir or _default_path("embedding_dir"))
    if not embedding_dir.is_dir():
        raise FileNotFoundError(f"骨干目录不存在：{embedding_dir}")
    tokenizer = _load_tokenizer(embedding_dir)

    torch.manual_seed(int(seed))
    base_model, _ = FT.build_joint_model(embedding_dir, num_labels=len(THREE))
    init_sha = SM.initial_state_sha256(base_model)
    init_state = _snapshot_state(base_model)
    del base_model
    gc.collect()
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()

    run = {
        "schema": C3_TRAIN_SCHEMA, "seed": int(seed), "init_state_sha256": init_sha,
        "budget": {"common_updates": updates, "sample_instances": updates * batch_size * grad_accum,
                   "batch_size": batch_size, "grad_accum": grad_accum,
                   "max_length": max_length, "device": str(device_obj), "bf16": bf16},
        "purpose": C3_PURPOSE, "prep_manifest": prep_manifest, "arms": {},
    }
    for arm, rows, data_path in (("official", official_rows, Path(official_data)),
                                 ("fine", fine_rows, Path(fine_data))):
        started = time.perf_counter()
        model, _ = FT.build_joint_model(embedding_dir, num_labels=len(THREE))
        model.load_state_dict(init_state, strict=True)
        if SM.initial_state_sha256(model) != init_sha:
            raise RuntimeError(f"{arm} 初始 tensor 与共同初始 state 不一致")
        training = NR.train_nli(
            model, tokenizer, rows, seed=int(seed), epochs=1,
            batch_size=int(batch_size), grad_accum=int(grad_accum), max_length=int(max_length),
            device=device_obj, bf16=bf16, checkpoint_selection="final", max_steps=updates,
        )
        if training["budget"]["actual_updates"] != updates:
            raise RuntimeError(f"{arm} 实际 updates {training['budget']['actual_updates']} 与共同预算 {updates} 不符")
        if training["selection"]["selected_step"] != updates:
            raise RuntimeError(f"{arm} 未在 fixed common updates 终点取权重")
        artifact = out / arm
        scorer = NR.NLIScorer(model=model, tokenizer=tokenizer, max_length=int(max_length),
                               device=str(device_obj))
        meta = NR.save_nli_artifact(
            scorer, artifact, weights="full", backbone_dir=embedding_dir,
            train_meta=_train_meta(arm=arm, data_path=data_path, rows=rows, training=training,
                                   seed=seed, init_sha=init_sha, max_length=max_length,
                                   batch_size=batch_size, grad_accum=grad_accum, updates=updates,
                                   prep_manifest=prep_manifest),
        )
        write_json(out / f"{arm}.training.json", {
            "budget": training["budget"], "selection": training["selection"],
            "history": training["history"], "updates": training["updates"],
            "data": training["data"], "truncation": training["truncation"],
            "device_log": training["device_log"], "guard": training["guard"],
        })
        run["arms"][arm] = {
            "artifact": str(artifact), "meta_sha256": file_sha256(artifact / "nli_meta.json"),
            "weights_sha256": meta["weights_sha256"], "seconds": round(time.perf_counter() - started, 3),
            "classes": dict(Counter(row["label"] for row in rows)),
        }
        del scorer, model
        gc.collect()
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()
    write_json(out / "run.json", run)
    return run


def evaluate_artifact(*, artifact, data_dir, out_dir, device="cuda", batch_size=32):
    """加载正式 NLI 工件，对 OCNLI dev 与 fine final 做同一 forward 交叉评测。"""
    import torch

    from research.memory_relations import finetune as FT
    from research.memory_relations import nli_reference as NR

    artifact = Path(artifact)
    data_dir = Path(data_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    scorer = NR.load_nli_scorer(artifact, device=device)
    scorer.batch_size = int(batch_size)
    report = {
        "schema": C3_EVAL_SCHEMA,
        "purpose": C3_PURPOSE,
        "artifact": {
            "path": str(artifact),
            "meta_sha256": file_sha256(artifact / "nli_meta.json"),
            "weights_sha256": file_sha256(artifact / "nli_model.pt"),
            "classes": list(scorer.classes), "direction": scorer.direction,
            "max_length": int(scorer.max_length), "device": scorer.device,
        },
        "data_manifest": {"path": str(data_dir / "manifest.json"),
                          "sha256": file_sha256(data_dir / "manifest.json")},
        "cohorts": {},
    }
    for cohort, filename in (("ocnli_dev", "ocnli_dev.jsonl"), ("fine_final", "fine_final.jsonl")):
        rows = read_jsonl(data_dir / filename)
        _checked_eval_rows(rows, where=cohort)
        requests = [row["request"] for row in rows]
        metrics = NR.eval_nli_metrics(
            scorer.model, scorer.tokenizer, rows, max_length=scorer.max_length,
            device=torch.device(scorer.device), batch_size=int(batch_size),
        )
        probabilities = scorer.predict_proba(requests)
        if probabilities.shape != (len(rows), len(THREE)):
            raise RuntimeError(f"{cohort} 概率形状不符：{probabilities.shape}")
        predictions = probabilities.argmax(axis=1)
        prediction_path = out / f"{cohort}.predictions.jsonl"
        write_jsonl(prediction_path, [
            {"id": row["id"], "root_id": row["root_id"], "cluster_id": row["cluster_id"],
             "group_id": row.get("group_id"), "source_split": row.get("source_split"),
             "label": row["label"], "prediction": THREE[int(index)],
             "probabilities": [float(value) for value in probability]}
            for row, index, probability in zip(rows, predictions, probabilities)
        ])
        cohort_report = {
            "rows": len(rows), "classes": dict(Counter(row["label"] for row in rows)),
            "metrics": metrics,
            "truncation": FT.truncation_profile(scorer.tokenizer, requests, scorer.max_length),
            "predictions": {"path": str(prediction_path),
                            "sha256": file_sha256(prediction_path), "rows": len(rows)},
        }
        if cohort == "ocnli_dev":
            cohort_report["metrics_row_native"] = NR.eval_nli_metrics(
                scorer.model, scorer.tokenizer,
                [{key: value for key, value in row.items() if key != "group_id"} for row in rows],
                max_length=scorer.max_length, device=torch.device(scorer.device),
                batch_size=int(batch_size),
            )
        report["cohorts"][cohort] = cohort_report
    write_json(out / "eval.json", report)
    return report


def smoke_overfit(*, data, out, embedding_dir=None, device="cuda", seed=13, per_class=8,
                  updates=100, max_length=384, threads=8, min_accuracy=0.9):
    """同一小 train 子集的有界机制诊断；不保存权重、不声称泛化、不启动正式两臂。"""
    import gc
    import time

    import torch

    from research.memory_relations import finetune as FT
    from research.memory_relations import nli_reference as NR
    from research.memory_relations import semantic_model as SM

    out_path = Path(out)
    if out_path.exists():
        raise FileExistsError(f"smoke 结果已存在，拒绝覆盖：{out_path}")
    rows = read_jsonl(data)
    _checked_train_rows(rows, where="smoke")
    by_class = {label: sorted([row for row in rows if row["label"] == label], key=lambda row: row["id"])
                for label in THREE}
    available = {label: len(by_class[label]) for label in THREE}
    short = [label for label in THREE if available[label] < 2]
    if short:
        raise ValueError(f"小拟合每类至少 2 条，缺 {short}；不虚凑根")
    selected, support = [], {}
    for label in THREE:
        take = min(int(per_class), available[label])
        support[label] = take
        selected.extend(by_class[label][:take])
    device_obj = _training_device(device)
    bf16 = FT.resolve_bf16("on" if device_obj.type == "cuda" else "off", device_obj)
    if device_obj.type == "cuda" and not bf16:
        raise ValueError("CUDA smoke 要求 BF16")
    if int(threads) < 1 or int(updates) < 1:
        raise ValueError("threads/updates 必须为正")
    torch.set_num_threads(int(threads))
    embedding_dir = Path(embedding_dir or _default_path("embedding_dir"))
    tokenizer = _load_tokenizer(embedding_dir)
    torch.manual_seed(int(seed))
    model, _ = FT.build_joint_model(embedding_dir, num_labels=len(THREE))
    init_sha = SM.initial_state_sha256(model)
    started = time.perf_counter()
    training = NR.train_nli(
        model, tokenizer, selected, seed=int(seed), epochs=1,
        batch_size=min(8, len(selected)), grad_accum=1, max_length=int(max_length),
        device=device_obj, bf16=bf16, checkpoint_selection="final", max_steps=int(updates),
    )
    same_rows = NR.eval_nli_metrics(
        model, tokenizer, selected, max_length=int(max_length), device=device_obj,
        batch_size=min(8, len(selected)),
    )
    passed = bool(same_rows["accuracy"] is not None and same_rows["accuracy"] >= float(min_accuracy))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "memory-relations-source-control-c3-smoke-v1",
        "purpose": {**C3_PURPOSE, "diagnostic": "same_train_subset_train_fit_only_not_generalization"},
        "data": {"path": str(Path(data)), "sha256": file_sha256(data), "rows": len(selected),
                 "support": support, "available": available},
        "seed": int(seed), "updates": int(updates), "max_length": int(max_length),
        "initial_state_sha256": init_sha,
        "device_log": training["device_log"], "guard": training["guard"],
        "training": {"budget": training["budget"], "selection": training["selection"],
                     "history": training["history"], "truncation": training["truncation"]},
        "same_rows_eval": same_rows,
        "threshold": float(min_accuracy), "passed": passed,
        "seconds": round(time.perf_counter() - started, 3),
        "weights_saved": False,
    }
    write_json(out_path, result)
    del model
    gc.collect()
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    d5 = commands.add_parser("d5")
    d5.add_argument("--index", required=True, type=Path)
    d5.add_argument("--out", required=True, type=Path)
    d5.add_argument("--bootstrap-samples", type=int, default=1000)

    prepare = commands.add_parser("prepare-c3", help="生成两源匹配小数据与 manifest")
    prepare.add_argument("--out", type=Path, default=_default_path("run_dir") / "data")
    prepare.add_argument("--canonical-dir", default=C3_DEFAULTS["canonical_dir"])
    prepare.add_argument("--fine-source", default=C3_DEFAULTS["fine_source"])
    prepare.add_argument("--public-data-run", default=C3_DEFAULTS["public_data_run"])
    prepare.add_argument("--fine-cohort", default=C3_DEFAULTS["fine_cohort"])
    prepare.add_argument("--seed", type=int, default=13)
    prepare.add_argument("--budget-epochs", type=int, default=20)
    prepare.add_argument("--batch-size", type=int, default=8)
    prepare.add_argument("--grad-accum", type=int, default=2)
    prepare.add_argument("--max-length", type=int, default=384)

    train = commands.add_parser("train-c3-pair", help="同初始 tensor/同配方训练两臂")
    train.add_argument("--official-data", type=Path, required=True)
    train.add_argument("--fine-data", type=Path, required=True)
    train.add_argument("--out", type=Path, default=_default_path("run_dir") / "models")
    train.add_argument("--embedding-dir", default=C3_DEFAULTS["embedding_dir"])
    train.add_argument("--device", default="cuda")
    train.add_argument("--seed", type=int, default=13)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--grad-accum", type=int, default=2)
    train.add_argument("--max-length", type=int, default=384)
    train.add_argument("--budget-epochs", type=int, default=20)
    train.add_argument("--threads", type=int, default=8)
    train.add_argument("--prep-manifest", type=Path, default=None)

    evaluate = commands.add_parser("eval-c3", help="加载正式工件做 OCNLI/fine final 交叉评测")
    evaluate.add_argument("--artifact", type=Path, required=True)
    evaluate.add_argument("--data-dir", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--batch-size", type=int, default=32)

    smoke = commands.add_parser("smoke-c3", help="同一小 train 子集的有界机制诊断")
    smoke.add_argument("--data", type=Path, required=True)
    smoke.add_argument("--out", type=Path, default=_default_path("run_dir") / "smoke.json")
    smoke.add_argument("--embedding-dir", default=C3_DEFAULTS["embedding_dir"])
    smoke.add_argument("--device", default="cuda")
    smoke.add_argument("--seed", type=int, default=13)
    smoke.add_argument("--per-class", type=int, default=8)
    smoke.add_argument("--updates", type=int, default=100)
    smoke.add_argument("--max-length", type=int, default=384)
    smoke.add_argument("--threads", type=int, default=8)
    smoke.add_argument("--min-accuracy", type=float, default=0.9)

    args = parser.parse_args(argv)
    if args.command == "d5":
        report = analyze_d5(args.index, args.out, samples=args.bootstrap_samples)
        print(json.dumps({"models": len(report["models"]), "output": str(args.out), "new_training": 0}, ensure_ascii=False))
        return 0
    if args.command == "prepare-c3":
        manifest = prepare_c3(
            out_dir=_resolve_path(args.out), canonical_dir=_resolve_path(args.canonical_dir),
            fine_source=_resolve_path(args.fine_source),
            public_data_run=_resolve_path(args.public_data_run),
            fine_cohort=_resolve_path(args.fine_cohort), seed=args.seed,
            budget_epochs=args.budget_epochs, batch_size=args.batch_size,
            grad_accum=args.grad_accum, max_length=args.max_length,
        )
        print(json.dumps({"output": str(args.out), "common_updates": manifest["budget"]["common_updates"],
                          "official": manifest["official_train"]["selected_rows"],
                          "fine": manifest["fine_train"]["selected_rows"],
                          "ocnli_dev": manifest["ocnli_dev"]["scored"],
                          "fine_final": manifest["fine_final"]["rows"]}, ensure_ascii=False))
        return 0
    if args.command == "train-c3-pair":
        run = train_pair(
            official_data=_resolve_path(args.official_data), fine_data=_resolve_path(args.fine_data),
            out_dir=_resolve_path(args.out),
            embedding_dir=_resolve_path(args.embedding_dir), device=args.device, seed=args.seed,
            batch_size=args.batch_size, grad_accum=args.grad_accum, max_length=args.max_length,
            budget_epochs=args.budget_epochs, threads=args.threads,
            prep_manifest=(str(_resolve_path(args.prep_manifest)) if args.prep_manifest else None),
        )
        print(json.dumps({"output": str(args.out), "arms": sorted(run["arms"]),
                          "common_updates": run["budget"]["common_updates"]}, ensure_ascii=False))
        return 0
    if args.command == "eval-c3":
        report = evaluate_artifact(artifact=_resolve_path(args.artifact), data_dir=_resolve_path(args.data_dir),
                                   out_dir=_resolve_path(args.out), device=args.device,
                                   batch_size=args.batch_size)
        print(json.dumps({cohort: {"accuracy": value["metrics"]["accuracy"],
                                   "macro_f1": value["metrics"]["macro_f1"],
                                   "n": value["metrics"]["n"]}
                          for cohort, value in report["cohorts"].items()}, ensure_ascii=False))
        return 0
    if args.command == "smoke-c3":
        result = smoke_overfit(
            data=_resolve_path(args.data), out=_resolve_path(args.out),
            embedding_dir=_resolve_path(args.embedding_dir),
            device=args.device, seed=args.seed, per_class=args.per_class, updates=args.updates,
            max_length=args.max_length, threads=args.threads, min_accuracy=args.min_accuracy,
        )
        print(json.dumps({"output": str(args.out), "passed": result["passed"],
                          "train_accuracy": result["same_rows_eval"]["accuracy"],
                          "updates": result["updates"], "seconds": result["seconds"]}, ensure_ascii=False))
        return 0 if result["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
