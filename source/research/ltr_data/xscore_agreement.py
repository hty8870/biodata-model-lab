                       
"""Cross-model labelling agreement analysis for LTR pseudo-labels.

Reads ``synth_heldout_xscore.jsonl`` (and optionally train) and reports:
- per-model label distribution, null rate, mean grade
- pairwise exact / within-1 / quadratic-kappa matrix between judges
- each judge vs consensus median; old single-model label vs consensus
- consensus vs deterministic constraint grade (the other label family)

Internal research tool; not part of the public delivery surface.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

IN_DIR = Path(__file__).resolve().parent


def qkappa(pairs):
    """Quadratic weighted kappa on 0..3 integer labels."""
    if not pairs:
        return None
    ha = [0] * 4
    hb = [0] * 4
    for a, b in pairs:
        ha[a] += 1
        hb[b] += 1
    n = len(pairs)
    num = den = 0.0
    for i in range(4):
        for j in range(4):
            w = (i - j) ** 2 / 9.0
            obs = sum(1 for a, b in pairs if a == i and b == j) / n
            exp = ha[i] * hb[j] / (n * n)
            num += w * obs
            den += w * exp
    return round(1 - num / den, 4) if den else None


def agree(pairs):
    if not pairs:
        return {"n": 0}
    n = len(pairs)
    return {
        "n": n,
        "exact": round(sum(1 for a, b in pairs if a == b) / n, 4),
        "within1": round(sum(1 for a, b in pairs if abs(a - b) <= 1) / n, 4),
        "kappa_q": qkappa(pairs),
    }


def load_pairs(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main(split="heldout"):
    rows = load_pairs(IN_DIR / f"synth_{split}_xscore.jsonl")
    items = []                                           
    for row in rows:
        for c in row["candidates"]:
            xs = c.get("xscore") or {}
            if xs:
                items.append((xs, c.get("llm_semantic_single"), c.get("grade"), c.get("llm_semantic")))
    models = sorted({m for xs, _, _, _ in items for m in xs})
    print(f"split={split} pairs_with_xscore={len(items)} models={models}")

    per_model = {}
    for m in models:
        vals = [xs[m] for xs, _, _, _ in items if xs.get(m) is not None]
        nulls = sum(1 for xs, _, _, _ in items if m in xs and xs.get(m) is None)
        dist = [vals.count(k) for k in range(4)]
        per_model[m] = {
            "n": len(vals), "nulls": nulls,
            "mean": round(statistics.mean(vals), 3) if vals else None,
            "dist_0_3": dist,
        }
    print(json.dumps({"per_model": per_model}, ensure_ascii=False, indent=2))

    matrix = {}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            pairs = [(xs[a], xs[b]) for xs, _, _, _ in items
                     if xs.get(a) is not None and xs.get(b) is not None]
            matrix[f"{a} vs {b}"] = agree(pairs)
    print(json.dumps({"pairwise": matrix}, ensure_ascii=False, indent=2))

    vs_cons = {}
    for m in models:
        pairs = [(xs[m], cons) for xs, _, _, cons in items
                 if xs.get(m) is not None and cons is not None]
        vs_cons[m] = agree(pairs)
    old_vs_cons = [(s, cons) for _, s, _, cons in items if s is not None and cons is not None]
    cons_vs_grade = [(cons, g) for _, _, g, cons in items if cons is not None and g is not None]
    old_vs_grade = [(s, g) for _, s, g, _ in items if s is not None and g is not None]
    summary = {
        "judge_vs_consensus": vs_cons,
        "old_single_vs_consensus": agree(old_vs_cons),
        "consensus_vs_det_grade": agree(cons_vs_grade),
        "old_single_vs_det_grade": agree(old_vs_grade),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out = {"split": split, "n_pairs": len(items), "per_model": per_model,
           "pairwise": matrix, **summary}
    (IN_DIR / f"_xscore_report_{split}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "heldout")
