                       
"""Validate /responses-endpoint judge candidates (gpt-5.6-luna and, once the
workspace data-policy opt-in is enabled, muse-spark-1.3-contributor).

Same protocol and seeded 300-pair sample as validate_cheap_judges.py.
Note: the /responses endpoint does not accept a temperature parameter
(defaults to 1), so scores carry sampling variance by design.
Internal research tool; key via OPENCODE_GO_API_KEY env var only.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llm_cross_score import USER_AGENT              
from llm_label_pairs import PROMPT, load_uid_texts              

RESPONSES_URL = "https://opencode.ai/zen/go/v1/responses"
CANDIDATES = [m for m in sys.argv[1:] if m] or ["gpt-5.6-luna"]
N_HIGH = 130


def main() -> None:
    key = os.environ["OPENCODE_GO_API_KEY"]
    texts = load_uid_texts()
    rows = [json.loads(l) for l in (ROOT / "research/ltr_data/synth_heldout_xscore.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    low, high = [], []
    for row in rows:
        for c in row["candidates"]:
            cons = c.get("llm_semantic")
            if cons is None:
                continue
            item = (row["query"], c["uid"], cons)
            (low if cons <= 1 else high).append(item)
    rng = random.Random(20260911)
    rng.shuffle(high)
    sample = low + high[:N_HIGH]
    rng.shuffle(sample)
    print(f"sample: {len(sample)} pairs; candidates={CANDIDATES}", flush=True)

    def judge_one(item, model):
        query, uid, cons = item
        meta = texts.get(uid) or {"title": uid, "description": ""}
        prompt = PROMPT.format(query=query, title=meta["title"], description=meta["description"])
        for attempt in range(3):
            body = json.dumps({"model": model, "max_output_tokens": 4096, "input": prompt}).encode()
            req = urllib.request.Request(RESPONSES_URL, data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json",
                "Accept": "application/json", "User-Agent": USER_AGENT,
                "x-opencode-session": str(uuid.uuid4())})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    d = json.loads(r.read())
                dt = time.time() - t0
                if d.get("status") not in (None, "completed"):
                    raise ValueError(f"status={d.get('status')}")
                text = "".join(
                    blk.get("text", "") for out in d.get("output", [])
                    if out.get("type") == "message"
                    for blk in out.get("content", [])
                    if isinstance(blk, dict) and blk.get("type") == "output_text")
                start = text.find("{")
                payload, _ = json.JSONDecoder().raw_decode(text, start)
                score = int(payload["score"])
                if not 0 <= score <= 3:
                    raise ValueError("range")
                u = d.get("usage", {})
                return score, dt, u.get("input_tokens", 0), u.get("output_tokens", 0)
            except Exception:
                time.sleep(2 * (attempt + 1))
        return None, None, 0, 0

    out = {}
    per_pair = {}
    for model in CANDIDATES:
        scores, lat, tp, tc = [], [], 0, 0
        per_pair[model] = {}
        t_start = time.time()
        with ThreadPoolExecutor(max_workers=16) as pool:
            futs = {pool.submit(judge_one, item, model): item for item in sample}
            for fut in as_completed(futs):
                item = futs[fut]
                s, dt, p_, c_ = fut.result()
                scores.append((item[2], s))
                per_pair[model][f"{item[0]}\t{item[1]}"] = s
                if dt:
                    lat.append(dt)
                tp += p_
                tc += c_
        pairs = [(a, b) for a, b in scores if b is not None]
        n = len(pairs)
        exact = sum(1 for a, b in pairs if a == b) / max(1, n)
        w1 = sum(1 for a, b in pairs if abs(a - b) <= 1) / max(1, n)
        shift = statistics.mean(b - a for a, b in pairs) if pairs else 0
        nulls = len(scores) - n
        out[model] = {
            "n": n, "nulls": nulls, "exact_vs_consensus": round(exact, 4),
            "within1": round(w1, 4), "mean_shift": round(shift, 3),
            "lat_p50_s": round(statistics.median(lat), 1) if lat else None,
            "prompt_tokens": tp, "completion_tokens": tc,
            "wall_s": round(time.time() - t_start, 0),
        }
        print(f"{model:26s} exact={exact:.3f} ±1={w1:.3f} shift={shift:+.3f} nulls={nulls} "
              f"p50={out[model]['lat_p50_s']}s tok={tp}/{tc}", flush=True)
    (ROOT / "research/ltr_data/_cheap_judge_validation3.json").write_text(
        json.dumps({"metrics": out, "per_pair": per_pair}, ensure_ascii=False), encoding="utf-8")
    print("wrote _cheap_judge_validation3.json")


if __name__ == "__main__":
    main()
