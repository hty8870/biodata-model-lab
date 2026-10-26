                       
"""Validate /messages-endpoint (Anthropic-shape) judge candidates.

Same protocol and same seeded sample as validate_cheap_judges.py, adapted for
the gateway's Anthropic-style /messages endpoint (qwen3.8-flash, minimax-m3).
Thinking blocks are skipped; only ``type == "text"`` blocks are parsed.
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

MESSAGES_URL = "https://opencode.ai/zen/go/v1/messages"
CANDIDATES = ["qwen3.8-flash", "minimax-m3"]
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
    print(f"sample: {len(sample)} pairs (cons<=1: {len(low)}, cons>=2: {min(N_HIGH, len(high))})", flush=True)

    def judge_one(item, model):
        query, uid, cons = item
        meta = texts.get(uid) or {"title": uid, "description": ""}
        prompt = PROMPT.format(query=query, title=meta["title"], description=meta["description"])
        for attempt in range(3):
            body = json.dumps({"model": model, "max_tokens": 4096, "temperature": 0,
                               "messages": [{"role": "user", "content": prompt}]}).encode()
            req = urllib.request.Request(MESSAGES_URL, data=body, headers={
                "Authorization": f"Bearer {key}", "x-api-key": key,
                "anthropic-version": "2023-06-01", "Content-Type": "application/json",
                "User-Agent": USER_AGENT, "x-opencode-session": str(uuid.uuid4())})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    d = json.loads(r.read())
                dt = time.time() - t0
                text = "".join(b.get("text", "") for b in d.get("content", [])
                               if isinstance(b, dict) and b.get("type") == "text")
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
    for model in CANDIDATES:
        scores, lat, tp, tc = [], [], 0, 0
        t_start = time.time()
        with ThreadPoolExecutor(max_workers=16) as pool:
            futs = {pool.submit(judge_one, item, model): item for item in sample}
            for fut in as_completed(futs):
                s, dt, p_, c_ = fut.result()
                scores.append((futs[fut][2], s))
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
        print(f"{model:18s} exact={exact:.3f} ±1={w1:.3f} shift={shift:+.3f} nulls={nulls} "
              f"p50={out[model]['lat_p50_s']}s tok={tp}/{tc}", flush=True)
    (ROOT / "research/ltr_data/_cheap_judge_validation2.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote _cheap_judge_validation2.json")


if __name__ == "__main__":
    main()
