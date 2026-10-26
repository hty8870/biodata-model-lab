from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("BIODATA_REPO_ROOT", str(HERE.parents[1]))).resolve()
BUNDLE = Path(os.environ.get("BIODATA_RELATION_BUNDLE_DIR", str(HERE / "bundle-v2")))
PIN = os.environ.get("BIODATA_RELATION_BUNDLE_SHA256", "65a8f9290a01ee038671a9bea068d2047e0897eac5937ecf16cb71bec16ad7b7")
CASES = [
    ("equal_zh", "默认使用中文回复。", "默认使用中文回复。"),
    ("disjoint_zh", "所有结果必须按时间升序排列。", "所有结果必须按时间降序排列。"),
    ("old_strict_zh", "报告必须包含研究方法和局限。", "报告必须包含研究方法。"),
    ("new_strict_zh", "报告必须包含研究方法。", "报告必须包含研究方法和局限。"),
    ("overlap_zh", "报告需要说明研究方法。", "使用图表展示计算资源消耗。"),
    ("equal_en", "The report must include limitations.", "The report must include limitations."),
    ("disjoint_en", "Use only local CPU inference.", "Never use local CPU inference."),
    ("old_strict_en", "The report must include methods and limitations.", "The report must include methods."),
    ("new_strict_en", "The report must include methods.", "The report must include methods and limitations."),
    ("overlap_en", "Write the report in English.", "Include resource usage figures in the report."),
    ("long_incomplete", "每次汇报都要详细解释方法、实验设计和局限。" * 180, "每次汇报都要说明方法。"),
]
PAIRS = [{"old": {"kind": "memory", "text": old}, "new": {"kind": "memory", "text": new}} for _, old, new in CASES]


def run(mode):
    start = time.perf_counter()
    if mode == "worker":
        os.environ["BIODATA_RELATION_BUNDLE_DIR"] = str(BUNDLE)
        os.environ["BIODATA_RELATION_BUNDLE_SHA256"] = PIN
        os.environ.setdefault("BIODATA_MODEL_RUNTIME_ROOT", str(HERE / "local-runtime"))
        from dataset_recommender.retrieval.model_runtime import external_relation_scorer
        scorer = external_relation_scorer()
        value = scorer.score_pairs(PAIRS)
        cold = time.perf_counter() - start
        warm_start = time.perf_counter()
        again = scorer.score_pairs(PAIRS)
        warm = time.perf_counter() - warm_start
        assert value == again, "warm CPU result mismatch"
        scorer.close()
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location("relation_worker_reference", ROOT / "src/dataset_recommender/retrieval/model_worker.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module._relation_import_paths(BUNDLE / "source")
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        from research.memory_relations.semantic_artifacts import load_runtime_artifact
        manifest = json.loads((BUNDLE / "bundle.json").read_text(encoding="utf-8"))
        scorer, meta, payload = load_runtime_artifact(BUNDLE / "runtime", expected_meta_sha256=manifest["meta_sha256"], device="cpu")
        del payload
        gc.collect()
        prepared = scorer.prepare_inputs(PAIRS)
        probabilities = scorer.predict_proba(PAIRS)
        value = {"classes": list(meta["classes"]), "probabilities": probabilities.tolist(), "input_complete": [item["complete"] for item in prepared], "model_id": manifest["model_id"], "meta_sha256": manifest["meta_sha256"]}
        cold = time.perf_counter() - start
        warm = None
    from dataset_recommender.agent.semantic_relations import validate_probabilities
    for row in value["probabilities"]:
        validate_probabilities(row)
    result = {"mode": mode, "device": "cpu", "cases": [item[0] for item in CASES], "input_sha256": hashlib.sha256(json.dumps(PAIRS, ensure_ascii=False, sort_keys=True).encode()).hexdigest(), "cold_seconds": cold, "warm_seconds": warm, "result": value, "routing_at_0_8": [{"case": case[0], "max_probability": max(p), "class": value["classes"][max(range(5), key=p.__getitem__)], "route": "local" if complete and max(p) >= 0.8 else "llm", "complete": complete} for case, p, complete in zip(CASES, value["probabilities"], value["input_complete"])], "limitation": "Handwritten integration probes, not a held-out accuracy benchmark or a calibration study."}
    target = HERE / (mode + "-probe.json")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["worker", "direct", "compare"])
    mode = parser.parse_args().mode
    if mode != "compare":
        run(mode)
    else:
        left, right = [json.loads((HERE / (name + "-probe.json")).read_text(encoding="utf-8")) for name in ("worker", "direct")]
        assert left["input_sha256"] == right["input_sha256"]
        for key in ("classes", "model_id", "meta_sha256", "input_complete"):
            assert left["result"][key] == right["result"][key], key
        delta = max(abs(a - b) for x, y in zip(left["result"]["probabilities"], right["result"]["probabilities"]) for a, b in zip(x, y))
        assert delta <= 1e-6, delta
        output = {"passed": True, "max_absolute_probability_difference": delta, "cases": len(CASES), "worker_seconds_cold": left["cold_seconds"], "worker_seconds_warm": left["warm_seconds"]}
        (HERE / "cpu-parity.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(output))
