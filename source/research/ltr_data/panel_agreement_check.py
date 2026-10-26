                       
"""How far does a substitute judge panel drift from the panel the corpus was labelled with?

Background: the training corpus was labelled by a three-model gateway panel
(``deepseek-v4-pro`` + ``mimo-v2.5`` + ``gpt-5.6-luna``, consensus = rounded median).
When that gateway is not reachable, the same labelling code can be pointed at another
OpenAI-compatible endpoint (``llm_cross_score.py --endpoint``). Switching the judge set
changes the *labels*, not the metrics -- gates measure the incumbent's own eval sets --
but the size of that change decides how much of a candidate's result may be attributed
to the new data rather than to the new label source. This script measures it directly:
every pair in the OOD layer already carries the gateway consensus in ``llm_semantic``,
so re-labelling a sample with the substitute panel gives an exact agreement figure.

Sampling is stratified by the stored consensus so the estimate is not dominated by the
most common grade. Internal research tool; keys come from the repository LLM config and
are never printed.

Usage:
  python research/ltr_data/panel_agreement_check.py --per-grade 40 --concurrency 12
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llm_cross_score import aggregate, resolve_api_key, score_one              
from llm_label_pairs import load_uid_texts              

CORPUS = ROOT / "research" / "ood_eval" / "synth_ood_label_input_xscore.jsonl"
OUT = ROOT / "research" / "ltr_data" / "_panel_agreement.json"


def sample_pairs(per_grade: int, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    buckets: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        for cand in row.get("candidates") or []:
            stored = cand.get("llm_semantic")
            if cand.get("uid") and isinstance(stored, int):
                buckets[stored].append({"query_id": row["query_id"], "uid": cand["uid"],
                                        "query": row["query"], "stored": stored})
    rng = random.Random(seed)
    sample = []
    for grade in sorted(buckets):
        pool = buckets[grade]
        rng.shuffle(pool)
        sample.extend(pool[:per_grade])
    return sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-grade", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--models", default="deepseek-v4-pro,deepseek-flash")
    parser.add_argument("--endpoint", default="https://api.deepseek.com/v1/chat/completions")
    parser.add_argument("--api-key-env", default="OPENCODE_GO_API_KEY")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    key = resolve_api_key(args.api_key_env, args.endpoint)
    if not key:
        print("no api key resolved", file=sys.stderr)
        return 2
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    texts = load_uid_texts()
    sample = sample_pairs(args.per_grade, args.seed)
    print(f"sampled {len(sample)} pairs x {len(models)} judges "
          f"= {len(sample) * len(models)} calls", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = []
        for item in sample:
            for model in models:
                futures.append((item, model, pool.submit(
                    score_one, item["query"], item["uid"], texts, model, key, 4096, 3, args.endpoint)))
        for item, model, fut in futures:
            _m, score = fut.result()
            item.setdefault("scores", {})[model] = score

    per_judge: dict[str, dict] = {}
    for model in models:
        pairs = [(item["scores"].get(model), item["stored"]) for item in sample
                 if item["scores"].get(model) is not None]
        per_judge[model] = _agreement([p[0] for p in pairs], [p[1] for p in pairs])

    consensus_pairs = []
    for item in sample:
        value, n, spread = aggregate({m: item["scores"].get(m) for m in models})
        item["consensus"], item["n_judges"], item["spread"] = value, n, spread
        if value is not None:
            consensus_pairs.append((value, item["stored"]))

    report = {
        "schema_version": 1,
        "endpoint": args.endpoint,
        "models": models,
        "corpus": CORPUS.name,
        "n_pairs": len(sample),
        "calls": len(sample) * len(models),
        "seed": args.seed,
        "per_grade_sample": args.per_grade,
        "stored_distribution": dict(Counter(item["stored"] for item in sample)),
        "per_judge": per_judge,
        "consensus": _agreement([p[0] for p in consensus_pairs], [p[1] for p in consensus_pairs]),
        "judge_spread": dict(Counter(item["spread"] for item in sample if item.get("spread") is not None)),
        "elapsed_s": round(time.time() - t0, 1),
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _agreement(substitute: list, stored: list) -> dict:
    if not substitute:
        return {"n": 0}
    diffs = [int(a) - int(b) for a, b in zip(substitute, stored)]
    exact = sum(1 for d in diffs if d == 0) / len(diffs)
    within1 = sum(1 for d in diffs if abs(d) <= 1) / len(diffs)
    return {
        "n": len(diffs),
        "exact": round(exact, 4),
        "within1": round(within1, 4),
        "mae": round(sum(abs(d) for d in diffs) / len(diffs), 4),
        "mean_shift": round(statistics.fmean(diffs), 4),
        "delta_distribution": dict(sorted(Counter(diffs).items())),
    }


if __name__ == "__main__":
    raise SystemExit(main())
