                       
"""OOD evaluation: behavior slice (track A, production truth) + ranking slice
(tracks A and B) under panel-C consensus semantic labels.

Gain per candidate: 0.5*(grade/2)+0.5*(llm_semantic/3) when a constraint grade
exists, else llm_semantic/3 (constraint-free classes). Orders compared:
``pos`` (production lexical order as retrieved), ``vec`` (dense cosine),
``rrf`` (rank fusion k=60 of pos and vec), ``ltr`` (current models/ltr
artifact). nDCG@5/@10 macro-averaged over queries with >=1 positive-gain
candidate. Internal research tool.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_ltr import candidate_features              


def gain(c) -> float | None:
    sem = c.get("llm_semantic")
    if sem is None:
        return None
    sem_v = float(sem) / 3.0
    grade = c.get("grade")
    if grade is None:
        return sem_v
    return 0.5 * (float(grade) / 2.0) + 0.5 * sem_v


def ndcg(order: list[int], gains: dict[int, float], k: int) -> float:
    dcg = sum(gains.get(idx, 0.0) / math.log2(r + 2) for r, idx in enumerate(order[:k]))
    ideal = sorted(gains.values(), reverse=True)
    idcg = sum(g / math.log2(r + 2) for r, g in enumerate(ideal[:k]))
    return dcg / idcg if idcg else 0.0


def orders_for(row: dict, track: str, ltr_scorer) -> dict[str, list[int]] | None:
    cands = row.get(track) or []
    if not cands:
        return None
    n = len(cands)
    gains = {}
    for i, c in enumerate(cands):
        g = gain(c)
        if g:
            gains[i] = g
    if not gains:
        return None
    pos_order = sorted(range(n), key=lambda i: cands[i].get("pos") or 999)
    vec_order = sorted(range(n), key=lambda i: -(cands[i].get("vec_score") or -1))
    rrf = defaultdict(float)
    for r, i in enumerate(pos_order):
        rrf[i] += 1.0 / (60 + r + 1)
    for r, i in enumerate(vec_order):
        rrf[i] += 1.0 / (60 + r + 1)
    rrf_order = sorted(range(n), key=lambda i: -rrf[i])
    out = {"pos": pos_order, "vec": vec_order, "rrf": rrf_order}
    if ltr_scorer is not None:
        sub = dict(row)
        sub["candidates"] = cands
        feats, _lex, _vec = candidate_features(sub)
        try:
            scores = ltr_scorer(feats.tolist())
            out["ltr"] = sorted(range(n), key=lambda i: -scores[i])
        except Exception:
            pass
    return out


def main() -> None:
    ltr_scorer = None
    try:
        from dataset_recommender.retrieval.vector_recall import load_ltr_scorer
        ltr_scorer = load_ltr_scorer()
        print("ltr scorer:", "loaded" if ltr_scorer else "unavailable")
    except Exception as e:
        print("ltr scorer load failed:", e)

    rows = [json.loads(l) for l in (ROOT / "research/ood_eval/synth_ood_vec.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]

    print("\n== behavior slice (track A, production truth) ==")
    print(f"{'class':<20}{'n':>4}{'有候选':>6}{'降级救回':>8}{'弃权':>6}")
    for cls in sorted({r["ood_class"] for r in rows}):
        rs = [r for r in rows if r["ood_class"] == cls]
        have = sum(1 for r in rs if r["n_returned"] > 0)
        deg = sum(1 for r in rs if r["degraded"])
        ab = sum(1 for r in rs if r["abstain"])
        print(f"{cls:<20}{len(rs):>4}{have:>6}{deg:>8}{ab:>6}")

    for track in ("candidates", "candidates_rescue"):
        print(f"\n== ranking slice ({track}) ==")
        per_cls = defaultdict(lambda: defaultdict(list))
        for row in rows:
            oo = orders_for(row, track, ltr_scorer)
            if oo is None:
                continue
            cands = row[track]
            gains = {i: gain(c) for i, c in enumerate(cands) if gain(c)}
            for name, order in oo.items():
                per_cls[row["ood_class"]][f"{name}@5"].append(ndcg(order, gains, 5))
                per_cls[row["ood_class"]][f"{name}@10"].append(ndcg(order, gains, 10))
        names = sorted({m for d in per_cls.values() for m in d})
        header = f"{'class':<20}{'q':>4}" + "".join(f"{m:>10}" for m in names)
        print(header)
        overall = defaultdict(list)
        for cls in sorted(per_cls):
            metrics = per_cls[cls]
            nq = len(metrics[names[0]]) if names else 0
            line = f"{cls:<20}{nq:>4}"
            for m in names:
                vals = metrics[m]
                overall[m].extend(vals)
                line += f"{statistics.mean(vals):>10.4f}" if vals else f"{'-':>10}"
            print(line)
        line = f"{'OVERALL':<20}{max((len(v) for v in overall.values()), default=0):>4}"
        for m in names:
            line += f"{statistics.mean(overall[m]):>10.4f}" if overall[m] else f"{'-':>10}"
        print(line)


if __name__ == "__main__":
    main()
