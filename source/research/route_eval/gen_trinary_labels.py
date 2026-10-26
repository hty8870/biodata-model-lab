                       
"""三分类（含弃权类 escalate）软标签派生（2026-09-20，弃权类实验）。

从 rlcd_soft_labels.jsonl（教师 20 票）派生三路软分布：
  confident（top_share ≥ CONF_TH）→ escalate=0，write/read 按票归一化
  拿不准（top_share < CONF_TH）→ escalate = 1 - top_share，
    write/read 按票分剩下的 top_share（教师分裂越大弃权质量越大——RLCD 原生软法）
输出 rlcd_soft3.jsonl，形状与 gen_rlcd_soft_labels.py 产物一致
（{qid, split, task:"binary3", votes:{write,read,escalate}, gold}），
train_qwen_decider.py 的软标签机制原样可用。
gold（参考真值）沿用 combined_all 的 gate；escalate 的「金标」由教师分裂度定义。
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = AGENT_ROOT / "research" / "route_eval"
SRC = EVAL_DIR / "rlcd_soft_labels.jsonl"
OUT = EVAL_DIR / "rlcd_soft3.jsonl"

CONF_TH = 0.9                              


def main() -> int:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    n_in = n_out = n_esc = 0
    esc_mass: list[float] = []
    with OUT.open("w", encoding="utf-8") as fh:
        for line in SRC.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d["task"] != "binary":
                continue
            n_in += 1
            v = d["votes"]
            w, r = v.get("write", 0), v.get("read", 0)
            total = w + r
            if not total:
                continue
            top = max(w, r) / total
            e = 0.0 if top >= CONF_TH else (1.0 - top)
            n = d["n"]
            votes = {"write": round(w / total * (1 - e) * n),
                     "read": round(r / total * (1 - e) * n),
                     "escalate": round(e * n)}
            if votes["escalate"] > 0:
                n_esc += 1
                esc_mass.append(e)
            rec = {"qid": d["qid"], "split": d["split"], "task": "binary3",
                   "utterance": d["utterance"], "n": n, "votes": votes,
                   "invalid": d.get("invalid", 0), "gold": d["gold"],
                   "teacher_majority": d.get("teacher_majority", ""),
                   "ts": d.get("ts", "")}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
    import numpy as np
    print(f"in={n_in} out={n_out} 含弃权质量={n_esc} ({n_esc / max(n_out, 1):.2%}) "
          f"mean_esc_mass={float(np.mean(esc_mass)) if esc_mass else 0:.3f}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
