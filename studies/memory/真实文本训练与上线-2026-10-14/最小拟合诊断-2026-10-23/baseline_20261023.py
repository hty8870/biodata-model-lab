from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from research.memory_relations.semantic_artifacts import load_runtime_artifact
from diagnostic_common_20261023 import ROOT, analysis_selftest, dump, evaluate, load_inputs, sha

META_SHA = "f600b2d46fa41f0fc18d697b7b1f139a3cabb457340f89ff30547bb89c39d742"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    assert not (ROOT / "online-baseline-2026-10-23.json").exists()
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "CUDA unavailable; no CPU fallback"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    scorer, meta, payload = load_runtime_artifact(args.repo_root / "artifacts/relation-confidence-20261023/bundle-v2/runtime", expected_meta_sha256=META_SHA, device="cpu")
    del payload
    loaded = time.perf_counter()
    rows, encodings, labels, ids, audit = load_inputs(scorer.tokenizer)
    prepared = time.perf_counter()
    scorer.model.to(device)
    metrics, predictions = evaluate(scorer.model, scorer.tokenizer, encodings, labels, ids, device)
    torch.cuda.synchronize()
    result = {"schema": "relation-fit-online-baseline-v1", "meta_sha256": META_SHA,
              "weights_sha256": meta["weights_sha256"], "classes": meta["classes"],
              "device": str(device), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "input_audit": audit, "analysis_selftest": analysis_selftest(), "metrics": metrics,
              "seconds": {"load": loaded - started, "prepare": prepared - loaded, "total": time.perf_counter() - started},
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    dump(ROOT / "online-baseline-2026-10-23.json", result)
    dump(ROOT / "online-baseline-predictions-2026-10-23.json", predictions)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
