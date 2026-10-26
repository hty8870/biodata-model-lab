                       
"""xdata-reuse WP3: 判官竞标场（统一版，取代 validate_cheap_judges*.py 三份一次性脚本）。

对任意 OpenCode Go 模型跑「相关性判官资格审查」：
- 采样协议与历史竞标逐位一致（--seed 20260911 洗牌后 low 全取 + high 取前 130，
  可用 --n / --n-high 覆盖）——不同模型在同一可比样本上对决。
- 打分提示词唯一真源：scripts/llm_label_pairs.PROMPT；网关三端点适配唯一真源：
  scripts/llm_cross_score.call_gateway / score_one（不写平行实现）。
- key 只走环境变量 OPENCODE_GO_API_KEY，绝不落盘。
输出：_audition_<model>.json（exact/±1/quad-κ/MAE/分布/失败数/耗时/粗估 token）+ 打印摘要。

用法：
  OPENCODE_GO_API_KEY=... python research/ltr_data/judge_audition.py --model deepseek-flash --n 12
  OPENCODE_GO_API_KEY=... python research/ltr_data/judge_audition.py --model glm-5.3-flash
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llm_cross_score import score_one              
from llm_label_pairs import load_uid_texts              

DATA = Path(__file__).resolve().parent


def quad_kappa(a, b, k=4):
    n = len(a)
    if n == 0:
        return None
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1
    ha = [sum(row) for row in obs]
    hb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / ((k - 1) ** 2)
            num += w * obs[i][j]
            den += w * ha[i] * hb[j] / n
    return round(1 - num / den, 4) if den else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=0, help="总样本量上限（0=按历史协议全 low + n-high）")
    ap.add_argument("--n-high", type=int, default=130)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    key = os.environ.get("OPENCODE_GO_API_KEY")
    if not key:
        print("ERROR: OPENCODE_GO_API_KEY 未设置", file=sys.stderr)
        return 2
    texts = load_uid_texts()
    rows = [json.loads(l) for l in (DATA / "synth_heldout_xscore.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    low, high = [], []
    for row in rows:
        for c in row["candidates"]:
            cons = c.get("llm_semantic")
            if cons is None:
                continue
            (low if cons <= 1 else high).append((row["query"], c["uid"], cons))
    rng = random.Random(args.seed)
    rng.shuffle(high)
    sample = low + high[:args.n_high]
    rng.shuffle(sample)
    if args.n and len(sample) > args.n:
        sample = sample[:args.n]
    print(f"audition: model={args.model} sample={len(sample)} seed={args.seed}", flush=True)

    t0 = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(score_one, q, u, texts, args.model, key, args.max_tokens): (q, u, cons)
                for q, u, cons in sample}
        done = 0
        for fut in as_completed(futs):
            q, u, cons = futs[fut]
            _, score = fut.result()
            results[(q, u)] = {"cons": cons, "score": score}
            done += 1
            if done % 25 == 0:
                print(f"  [{done}/{len(sample)}]", flush=True)

    valid = [(v["score"], v["cons"]) for v in results.values() if v["score"] is not None]
    n_fail = len(sample) - len(valid)
    a = [s for s, _ in valid]
    b = [c for _, c in valid]
    exact = sum(1 for s, c in valid if s == c) / max(1, len(valid))
    pm1 = sum(1 for s, c in valid if abs(s - c) <= 1) / max(1, len(valid))
    mae = sum(abs(s - c) for s, c in valid) / max(1, len(valid))
    dist = {}
    for s, _ in valid:
        dist[str(s)] = dist.get(str(s), 0) + 1
    elapsed = time.time() - t0
    est_tokens = len(sample) * 800
    report = {
        "model": args.model, "seed": args.seed, "n_sample": len(sample),
        "n_fail": n_fail, "exact": round(exact, 4), "pm1": round(pm1, 4),
        "mae": round(mae, 4), "quad_kappa": quad_kappa(a, b),
        "score_dist": dist, "elapsed_s": round(elapsed, 1),
        "est_tokens": est_tokens,
        "protocol": "heldout 五判官融合共识为金标；采样=seed 洗牌后 low 全取 + high 前 n-high",
    }
    out = DATA / f"_audition_{args.model.replace('/', '_')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
