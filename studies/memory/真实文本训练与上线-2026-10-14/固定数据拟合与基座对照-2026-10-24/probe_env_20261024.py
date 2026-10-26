from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def main() -> None:
    info = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    for name in ("transformers", "peft", "accelerate", "huggingface_hub", "numpy", "sklearn"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "unknown")
        except Exception as error:                                             
            info[name] = f"MISSING: {error!r}"
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["capability"] = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        info["total_memory_bytes"] = [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]
    info["disk_free_gib_c"] = round(shutil.disk_usage("C:/").free / 2 ** 30, 2)
    info["cwd"] = os.getcwd()
    info["hf_home_env"] = os.environ.get("HF_HOME")
    print(json.dumps(info, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
