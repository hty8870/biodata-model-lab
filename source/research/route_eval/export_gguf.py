                       
"""LoRA 适配器 → 合并 → GGUF → 多档量化（真·小 Jev 批，2026-09-19）。

链路：base(modelscope 缓存) + adapter(训练臂产物) --peft merge_and_unload-->
merged HF(bf16 safetensors) --convert_hf_to_gguf.py--> GGUF bf16 --llama-quantize-->
Q4_K_M / Q3_K_M / Q2_K（服务器 2C4G 预算：Q4≈1.1GB / Q3≈0.9GB / Q2≈0.7GB，
逐级精度代价由 eval_gguf_decider.py 实测，选「不掉精度的最小档」）。

Usage:
  .venv/Scripts/python.exe research/route_eval/export_gguf.py \
      --arm qwen3_decider_b --base Qwen/Qwen3-1.7B [--levels Q4_K_M,Q3_K_M,Q2_K]
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = AGENT_ROOT / "research" / "route_eval"
LLAMA = EVAL_DIR / "_llamacpp"
CONVERT = LLAMA / "llama.cpp-b11052" / "convert_hf_to_gguf.py"
QUANTIZE = LLAMA / "cpu-x64" / "llama-quantize.exe"
PY = AGENT_ROOT / ".venv" / "Scripts" / "python.exe"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, help="训练臂目录名（research/route_eval/<arm>/adapter）")
    ap.add_argument("--base", required=True)
    ap.add_argument("--levels", default="Q4_K_M,Q3_K_M,Q2_K")
    args = ap.parse_args()
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass

    arm_dir = EVAL_DIR / args.arm
    adapter = arm_dir / "adapter"
    merged = arm_dir / "merged"
    gguf_dir = EVAL_DIR / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    if not merged.exists():
        print("== merge LoRA ==", flush=True)
        import torch
        from modelscope import snapshot_download
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        base_path = snapshot_download(args.base)
        tok = AutoTokenizer.from_pretrained(base_path)
        base = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, adapter)
        model = model.merge_and_unload()
        model.save_pretrained(merged, safe_serialization=True)
        tok.save_pretrained(merged)
        print(f"-> {merged}", flush=True)
    else:
        print(f"merged 已存在，跳过（{merged}）", flush=True)

    bf16_gguf = gguf_dir / f"{args.arm}-bf16.gguf"
    if not bf16_gguf.exists():
        print("== convert HF -> GGUF bf16 ==", flush=True)
        subprocess.run([str(PY), str(CONVERT), str(merged), "--outfile", str(bf16_gguf),
                        "--outtype", "bf16"], check=True)
    print(f"bf16: {bf16_gguf.name} {bf16_gguf.stat().st_size / 1e6:.0f} MB", flush=True)

    outs = {"bf16": str(bf16_gguf)}
    for level in [lv.strip() for lv in args.levels.split(",") if lv.strip()]:
        out = gguf_dir / f"{args.arm}-{level}.gguf"
        if not out.exists():
            print(f"== quantize {level} ==", flush=True)
            subprocess.run([str(QUANTIZE), str(bf16_gguf), str(out), level], check=True,
                           stdout=subprocess.DEVNULL)
        print(f"{level}: {out.name} {out.stat().st_size / 1e6:.0f} MB", flush=True)
        outs[level] = str(out)

    (gguf_dir / f"{args.arm}-gguf-index.json").write_text(
        json.dumps(outs, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
