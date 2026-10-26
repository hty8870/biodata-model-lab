from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
LOG = HERE / "download-20261024.jsonl"

                                                                                   
os.environ.setdefault("HF_HOME", str(HERE / "hf-home"))
os.environ.setdefault("TORCH_HOME", str(HERE / "torch-home"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from huggingface_hub import snapshot_download              

BASES = {
    "macbert": {
        "repo_id": "hfl/chinese-macbert-base",
        "revision": "a986e004d2a7f2a1c2f5a3edef4e20604a974ed1",
        "allow_patterns": ["config.json", "pytorch_model.bin", "vocab.txt", "tokenizer_config.json",
                           "tokenizer.json", "special_tokens_map.json", "added_tokens.json"],
        "expected_weights": ["pytorch_model.bin"],
    },
    "qwen3": {
        "repo_id": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "allow_patterns": ["config.json", "generation_config.json", "model.safetensors",
                           "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
                           "special_tokens_map.json", "added_tokens.json"],
        "expected_weights": ["model.safetensors"],
    },
}


def log(record: dict) -> None:
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


def main() -> int:
    only = sys.argv[1:] or list(BASES)
    failures = 0
    for name in only:
        spec = BASES[name]
        target = MODELS / name
        target.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        log({"event": "download_start", "name": name, "repo_id": spec["repo_id"],
             "revision": spec["revision"], "local_dir": str(target), "hf_home": os.environ["HF_HOME"]})
        try:
            path = snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"], token=False,
                                     local_dir=str(target), allow_patterns=spec["allow_patterns"])
            files = sorted(item.relative_to(target).as_posix() for item in target.rglob("*")
                           if item.is_file() and ".cache" not in item.parts)
            weights = {item: (target / item).stat().st_size for item in spec["expected_weights"]}
            missing = [item for item in spec["expected_weights"] if not (target / item).is_file()]
            if missing:
                raise RuntimeError(f"missing expected weights: {missing}")
            log({"event": "download_done", "name": name, "path": path, "files": files, "weight_bytes": weights,
                 "seconds": round(time.perf_counter() - started, 2)})
        except Exception as error:                                                               
            failures += 1
            log({"event": "download_failed", "name": name, "error": f"{type(error).__name__}: {error}",
                 "seconds": round(time.perf_counter() - started, 2)})
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
