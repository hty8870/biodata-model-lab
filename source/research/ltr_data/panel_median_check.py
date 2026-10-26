                       
"""Per-pair panel-median acceptance check against the gold 5-judge consensus.

Re-judges the seeded 300-pair validation sample with the three shortlisted
cheap panelists (mimo-v2.5, qwen3.8-flash, deepseek-flash), saving per-pair
scores; deepseek-v4-pro scores are reused from the stored heldout xscore.
Then evaluates candidate triplets' median vs the gold consensus.
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

from llm_cross_score import API_URL, USER_AGENT              
from llm_label_pairs import PROMPT, load_uid_texts              

MESSAGES_URL = "https://opencode.ai/zen/go/v1/messages"
CHEAP = ["mimo-v2.5", "qwen3.8-flash", "deepseek-flash"]
N_HIGH = 130
TRIPLETS = {
    "A[v4pro,mimo,qwen]": ("deepseek-v4-pro", "mimo-v2.5", "qwen3.8-flash"),
    "B[dsflash,mimo,qwen]": ("deepseek-flash", "mimo-v2.5", "qwen3.8-flash"),
}


def call_chat(prompt, model, key):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 4096, "temperature": 0}).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "User-Agent": USER_AGENT, "x-opencode-session": str(uuid.uuid4())})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or ""


def call_messages(prompt, model, key):
    body = json.dumps({"model": model, "max_tokens": 4096, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(MESSAGES_URL, data=body, headers={
        "Authorization": f"Bearer {key}", "x-api-key": key,
        "anthropic-version": "2023-06-01", "Content-Type": "application/json",
        "User-Agent": USER_AGENT, "x-opencode-session": str(uuid.uuid4())})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    return "".join(b.get("text", "") for b in d.get("content", [])
                   if isinstance(b, dict) and b.get("type") == "text")


def judge(item, model, key):
    query, uid, _cons = item
    meta = TEXTS.get(uid) or {"title": uid, "description": ""}
    prompt = PROMPT.format(query=query, title=meta["title"], description=meta["description"])
    call = call_messages if model in ("qwen3.8-flash",) else call_chat
    for attempt in range(3):
        try:
            text = call(prompt, model, key)
            start = text.find("{")
            payload, _ = json.JSONDecoder().raw_decode(text, start)
            score = int(payload["score"])
            if 0 <= score <= 3:
                return score
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


TEXTS = {}


def main() -> None:
    global TEXTS
    key = os.environ["OPENCODE_GO_API_KEY"]
    TEXTS = load_uid_texts()
    rows = [json.loads(l) for l in (ROOT / "research/ltr_data/synth_heldout_xscore.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    low, high = [], []
    v4pro = {}
    for row in rows:
        for c in row["candidates"]:
            cons = c.get("llm_semantic")
            if cons is None:
                continue
            item = (row["query"], c["uid"], cons)
            (low if cons <= 1 else high).append(item)
            v4pro[(row["query"], c["uid"])] = (c.get("xscore") or {}).get("deepseek-v4-pro")
    rng = random.Random(20260911)
    rng.shuffle(high)
    sample = low + high[:N_HIGH]
    rng.shuffle(sample)
    print(f"sample: {len(sample)} pairs", flush=True)

    per_model: dict[str, dict] = {m: {} for m in CHEAP}
    for model in CHEAP:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=16) as pool:
            futs = {pool.submit(judge, item, model, key): item for item in sample}
            for fut in as_completed(futs):
                item = futs[fut]
                per_model[model][(item[0], item[1])] = fut.result()
        print(f"  {model} done in {time.time()-t0:.0f}s", flush=True)

    def median3(vals):
        vals = [v for v in vals if v is not None]
        return int(round(statistics.median(vals))) if vals else None

    report = {}
    for name, members in TRIPLETS.items():
        exact = within1 = n = 0
        for item in sample:
            query, uid, cons = item
            vals = []
            for m in members:
                v = v4pro[(query, uid)] if m == "deepseek-v4-pro" else per_model[m].get((query, uid))
                vals.append(v)
            med = median3(vals)
            if med is None:
                continue
            n += 1
            exact += med == cons
            within1 += abs(med - cons) <= 1
        report[name] = {"n": n, "exact": round(exact / max(1, n), 4),
                        "within1": round(within1 / max(1, n), 4)}
        print(f"{name}: median vs gold exact={exact/max(1,n):.4f} ±1={within1/max(1,n):.4f} (n={n})")
    (ROOT / "research/ltr_data/_panel_median_check.json").write_text(
        json.dumps({"triplets": report,
                    "per_pair": {m: {f"{q}\t{u}": v for (q, u), v in d.items()}
                                 for m, d in per_model.items()}}, ensure_ascii=False),
        encoding="utf-8")
    print("wrote _panel_median_check.json")


if __name__ == "__main__":
    main()
