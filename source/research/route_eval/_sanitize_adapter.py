                       
"""peft adapter 脱敏（通用版，2026-09-19 真·小 Jev 批重写）。

peft 保存的 adapter_config.json / README.md 会内嵌本机 modelscope 缓存绝对路径
（<user-home>\\...）——交付面红线（家路径泄漏）。本脚本扫描
research/route_eval/*/adapter 与 src/dataset_recommender/agent/router_models/*/adapter，
把一切 modelscope 缓存绝对路径改回规范模型名（Org--Model 目录名 → Org/Model）。

纪律：train_qwen_decider.py / export_gguf.py 每次产出新 adapter 后必须重跑
（检查由 tests/test_delivery_safety.py + test_release_builder.py 机械把关）。
"""
import json
import re
from pathlib import Path

                                            
PAT = re.compile(r"[A-Za-z0-9_./-]*\\Users\\[^\\\s]+\\\.cache\\modelscope\\models\\([^\\\s]+)--([^\\\s]+)\\snapshots\\[^\\\s]+")
                                  
PAT_JSON = re.compile(r"[A-Za-z0-9_./-]*\\\\Users\\\\[^\\\\\s]+\\\\\.cache\\\\modelscope\\\\models\\\\([^\\\\\s]+)--([^\\\\\s]+)\\\\snapshots\\\\[^\\\\\s]+")

ROOTS = ("research/route_eval", "src/dataset_recommender/agent/router_models")


def canon(m: re.Match) -> str:
    return f"{m.group(1)}/{m.group(2)}"


def sanitize_dir(adapter: Path) -> None:
    cfg_p = adapter / "adapter_config.json"
    if cfg_p.exists():
        cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        bmp = str(cfg.get("base_model_name_or_path") or "")
        m = PAT.search(bmp)
        if m:
            cfg["base_model_name_or_path"] = canon(m)
            cfg_p.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    rd_p = adapter / "README.md"
    if rd_p.exists():
        t = rd_p.read_text(encoding="utf-8")
        t2 = PAT.sub(canon, PAT_JSON.sub(canon, t))
        if t2 != t:
            rd_p.write_text(t2, encoding="utf-8", newline="\n")
    print("sanitized", adapter)


def main() -> None:
    n = 0
    for root in ROOTS:
        rp = Path(root)
        if not rp.exists():
            continue
        for adapter in sorted(rp.glob("*/adapter")):
            if adapter.is_dir():
                sanitize_dir(adapter)
                n += 1
    print(f"done ({n} adapter dirs)")


if __name__ == "__main__":
    main()
