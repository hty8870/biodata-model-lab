                       
"""Offline learning-to-rank experiment over the synthetic graded query pairs.

Trains and compares two ranking models on one shared feature set — a pointwise
gradient-boosted regressor (scikit-learn) and a pairwise RankNet-style MLP
(PyTorch) — then reports nDCG@5 / nDCG@10 / MRR on the held-out split against
three production-rule baselines (lex order, linear fusion, reciprocal rank
fusion), plus a feature ablation.

Features deliberately contain only signals available at inference time
(lex_score, vec_score, rank position and per-dimension match flags); the
deterministic judge grade and the LLM semantic score are labels only and never
enter the feature vector, so the reported metrics measure ranking rather than
label recall.

Deliberately standalone: reads only the JSONL pair files under
``research/ltr_data/`` (stdlib + numpy + scikit-learn + PyTorch), imports nothing
from the project's ``src/``, makes no network or LLM calls, and reproduces the same
weights for a fixed ``--seed`` (up to CPU reduction order — see ``--deterministic``
below for the bit-exact mode the promotion pipeline uses).

``--deterministic`` (off by default, so standalone runs stay bit-for-bit what they
were) additionally pins the *numerical* environment: single-threaded PyTorch plus
``random``/``numpy``/``torch`` seeding from the same ``--seed``. A fixed seed alone
does not make a CPU run reproducible — the intra-op thread count decides the
reduction order of the pairwise loss, so the same seed lands on a different
minimum in every run (measured: heldout nDCG@10 sample SD ≈ 5.7e-4 across retrains,
which is the same order as the promotion gate's tolerances). The promotion
pipeline (``scripts/ltr_promote.py candidate``) always passes this flag.

With ``--export DIR`` the full-feature RankNet MLP is also written out as
``ltr_mlp.pt`` (state_dict) and ``ltr_spec.json`` (feature contract, training
normalisation stats, held-out metrics). The exported artefacts are then re-loaded
and re-scored against the in-training ranking; a per-query nDCG@10 mismatch above
1e-6 fails the run.

Input: one JSON object per line, ``{"query_id", "query", "candidates": [...]}``.
Each candidate carries ``uid``, 1-based ``pos``, ``lex_score``, an optional
``vec_score`` (dense-embedding cosine; absent or null counts as 0 with a
``has_vec`` companion flag), a deterministic constraint ``grade`` (2 = every
constraint met, 1 = exactly one violated, 0 = two or more) and a per-dimension
``matched`` map; ``llm_semantic`` (0-3, possibly absent) is the optional LLM
relevance score. Split files are read in layers of increasing richness —
``synth_{split}_vec.jsonl`` > ``_labeled.jsonl`` > ``_nl.jsonl`` >
``synth_{split}.jsonl`` — where a higher-priority row overrides the same query_id
and lower layers fill in the rest, because the augmentation jobs write their
layers incrementally.

Features (``--feature-set``, 9 floats per candidate by default): lex_score
normalised by the query maximum, vec_score and its has_vec flag, 1/pos, the four
fixed ``matched`` slots (species / tissue / disease / has_raw_data; a missing key
counts as 0), and the query's has_raw_data_required flag.

``--feature-set`` selects where those nine inputs come from — the column *contract*
(9 names, same order) is identical for ``f0`` and ``f1``; only the provenance of
four of them changes:

- ``f0`` (default, unchanged): read ``inv_pos`` from the candidate's stored
  ``pos`` and the ``matched`` map / ``has_raw_data_required`` flag from the row as
  harvested (``matched`` was judged against the *intended* constraints).
- ``f1``: read the production-replayed overlay written by
  ``research/ltr_data/exp_20260929/build_f1_features.py``: ``inv_pos`` is the
  position in the row's own pool (1/(i+1), the index the fusion scorer actually
  sees), and ``matched`` / ``raw_required`` come from the *parsed* intent
  (``f1_matched`` / ``f1_raw_required``). Missing overlay fields fail loudly
  rather than silently falling back to f0.
- ``f2``: ``f1`` plus eight query-condition columns (``F2_QUERY_FEATURES``, read
  from the row's ``f2_query`` list). Experimental: there is no production
  consumer for these names, so ``--export`` refuses f2 (see below).
- ``f3``: read the row's per-candidate ``f3_feats`` overlay (9 floats in
  ``FEATURE_NAMES`` order) written by ``research/ltr_data/build_f3_features.py``
  — the production-replayed survivor pool scored by the *same* constructor the
  inference path uses (``retrieval/ltr_features.ltr_feature_rows``), so the
  columns are bit-identical to what production scores at serve time. Missing or
  malformed overlay raises rather than silently degrading to f0/f1 semantics.
  Because the features are production-identical, ``--export`` **allows** f3
  (that is the point of single-sourcing); the spec records ``feature_set: f3``.

Labels are resolved from ``--label`` (sets both) or the decoupled
``--train-label`` / ``--eval-label``: ``det`` = grade/2; ``sem`` = llm_semantic/3
(candidates without a semantic score are dropped); ``mix`` = 0.5*grade/2 +
0.5*llm_semantic/3 when a semantic score exists, else grade/2. Gains for nDCG use
2**label - 1.

Extra evaluation sets can be pinned with ``--eval-set NAME=PATH`` (a single JSONL
file, no layer merging) so that one training run reports on several fixed sets;
``--per-query-out`` dumps the per-query nDCG@10 detail those sets need.

Usage:
  python scripts/train_ltr.py --label det
  python scripts/train_ltr.py --label mix --seed 20260910
  python scripts/train_ltr.py --label mix --export models/ltr
  python scripts/train_ltr.py --in-dir <exp> --feature-set f1 \
      --train-label sem --eval-label sem --eval-set oodonly=<path>
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

AGENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN_DIR = AGENT_ROOT / "research" / "ltr_data"
DEFAULT_SEED = 20260910

                                                                                 
                                                           
LAYER_SUFFIXES = ("vec", "labeled", "nl", "")

                                                                                 
                                                 
DIM_SLOTS = ("species", "tissue", "disease", "has_raw_data")

FEATURE_NAMES = (
    "lex_norm",
    "vec_norm",
    "has_vec",
    "inv_pos",
    "matched_species",
    "matched_tissue",
    "matched_disease",
    "matched_has_raw_data",
    "raw_required",
)

                                                                                 
                                                                                
                                                                            
                                                   
F2_QUERY_FEATURES = (
    "q_has_species",
    "q_has_tissue",
    "q_has_disease",
    "q_has_raw",
    "q_unresolved_ratio",
    "q_len_chars",
    "q_lex_max",
    "q_lexvec_disagree",
)

                                                                                    
                                                                       
FEATURE_SETS: "dict[str, tuple[str, ...]]" = {
    "f0": FEATURE_NAMES,
    "f1": FEATURE_NAMES,
    "f2": FEATURE_NAMES + F2_QUERY_FEATURES,
    "f3": FEATURE_NAMES,
}

_CONSTRAINT_COLS = (4, 5, 6, 7, 8)


def feature_names(feature_set: str) -> "tuple[str, ...]":
    """Column names of a feature set; unknown names raise (no silent f0 fallback)."""
    try:
        return FEATURE_SETS[feature_set]
    except KeyError:
        raise ValueError(f"未知 feature-set：{feature_set!r}（可选 {'/'.join(FEATURE_SETS)}）") from None


def ablation_cols(feature_set: str) -> "dict[str, tuple[int, ...]]":
    """Ablation column groups, sized to the feature set (``full`` = every column)."""
    names = feature_names(feature_set)
    return {
        "constraint": _CONSTRAINT_COLS,
        "constraint+lex": (0,) + _CONSTRAINT_COLS,
        "full": tuple(range(len(names))),
    }

LINEAR_ALPHA = 0.5
RRF_K = 60
PAIR_CAP = 200_000
RANKNET_HIDDEN = (32, 16)
RANKNET_EPOCHS = 400
RANKNET_LR = 0.02

ScoreFn = Callable[[np.ndarray], np.ndarray]
OrderFn = Callable[[dict], np.ndarray]


def _read_jsonl(path: Path) -> list[dict]:
    """Read one JSON object per line, skipping blank or torn trailing lines."""
    rows: list[dict] = []
    malformed = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                                                                              
                                                                                     
                malformed += 1
    if malformed:
        print(f"  ! skipped {malformed} malformed line(s) in {path.name}")
    return rows


def _layer_path(in_dir: Path, stem: str, suffix: str) -> Path:
    name = f"synth_{stem}_{suffix}.jsonl" if suffix else f"synth_{stem}.jsonl"
    return in_dir / name


def load_split(in_dir: Path, stem: str) -> tuple[list[dict], dict]:
    """Merge the split's existing layers, richest first, filling gaps from plainer ones."""
    layers = []
    for suffix in LAYER_SUFFIXES:
        path = _layer_path(in_dir, stem, suffix)
        if path.exists():
            layers.append((path.name, _read_jsonl(path)))
    if not layers:
        names = ", ".join(_layer_path(in_dir, stem, s).name for s in LAYER_SUFFIXES)
        raise FileNotFoundError(f"none of {names} exists in {in_dir}")
                                                                                 
                                                                                  
                        
    merged: list[dict] = []
    index: dict[str, int] = {}
    for _name, rows in reversed(layers):
        for row in rows:
            qid = row.get("query_id")
            if qid and qid in index:
                merged[index[qid]] = row
            else:
                if qid:
                    index[qid] = len(merged)
                merged.append(row)
    info = {"layers": {name: len(rows) for name, rows in layers}, "queries": len(merged)}
    return merged, info


def merge_mix_in(train_rows: list[dict], paths: list[Path]) -> tuple[list[dict], dict]:
    """Merge extra training rows (e.g. OOD xscore exports) into the train split.

    Rows dedupe by query_id against the base split and each other; on collision
    the row with more candidates wins (a same-id double can be an A/B rescore).
    Extra rows never touch the heldout split — mixing into evaluation would leak."""
    merged = list(train_rows)
    index = {row.get("query_id"): i for i, row in enumerate(merged) if row.get("query_id")}
    stats: dict[str, dict] = {}
    for path in paths:
        rows = _read_jsonl(path)
        added = replaced = 0
        for row in rows:
            qid = row.get("query_id")
            if not qid:
                merged.append(row)                                         
                added += 1
                continue
            if qid in index:
                if len(row.get("candidates") or []) > len(merged[index[qid]].get("candidates") or []):
                    merged[index[qid]] = row
                    replaced += 1
            else:
                index[qid] = len(merged)
                merged.append(row)
                added += 1
        stats[path.name] = {"rows": len(rows), "added": added, "replaced": replaced}
    return merged, stats


def candidate_features(row: dict, feature_set: str = "f0") -> tuple[np.ndarray, list[float], list[float]]:
    """Build the feature matrix (n_candidates x n_features), lex scores and vec scores.

    Vec scores are returned with NaN for missing values so the fusion baselines can
    tell "no embedding for this candidate" from "embedding scored zero".

    ``feature_set`` picks the provenance of four columns (see the module docstring):
    f0 reads the harvested ``pos`` / ``matched`` / ``has_raw_data_required``; f1 and
    f2 read the production-replay overlay (``f1_matched`` / ``f1_raw_required`` plus
    the row index for ``inv_pos``); f3 reads the whole nine-column row from the
    candidate's ``f3_feats`` overlay (production-constructed, bit-identical to
    inference). Overlay fields missing under f1/f2/f3 raise instead of degrading
    to f0 — a silent fallback would make the arms indistinguishable.
    """
    names = feature_names(feature_set)
    candidates = row.get("candidates") or []
    lex = [float(c.get("lex_score") or 0.0) for c in candidates]
    lex_max = max(lex) if lex else 0.0
    if feature_set == "f0":
        raw_required = 1.0 if row.get("has_raw_data_required") else 0.0
    elif feature_set in ("f1", "f2"):
        if "f1_raw_required" not in row:
            raise ValueError(f"feature-set={feature_set} 需要行级 f1_raw_required（query_id={row.get('query_id')!r}）")
        raw_required = 1.0 if row.get("f1_raw_required") else 0.0
    else:
                                                    
        raw_required = 0.0
    query_block: list[float] = []
    if feature_set == "f2":
        block = row.get("f2_query")
        if not isinstance(block, (list, tuple)) or len(block) != len(F2_QUERY_FEATURES):
            raise ValueError(
                f"feature-set=f2 需要行级 f2_query（{len(F2_QUERY_FEATURES)} 个值，"
                f"query_id={row.get('query_id')!r}）"
            )
        query_block = [float(value) for value in block]
    vec: list[float] = []
    feats = []
    for index, (cand, score) in enumerate(zip(candidates, lex)):
        vec_score = cand.get("vec_score")
        missing = vec_score is None
        vec.append(float("nan") if missing else float(vec_score))
        if feature_set == "f3":
            overlay = cand.get("f3_feats")
            if not isinstance(overlay, (list, tuple)) or len(overlay) != len(FEATURE_NAMES):
                raise ValueError(
                    f"feature-set=f3 需要候选自带 f3_feats（{len(FEATURE_NAMES)} 个 float，"
                    f"uid={cand.get('uid')!r}）"
                )
            row9 = [float(value) for value in overlay]
            if not all(np.isfinite(row9)):
                raise ValueError(f"feature-set=f3 的 f3_feats 含非有限值（uid={cand.get('uid')!r}）")
            feats.append(row9)
            continue
        if feature_set == "f0":
            matched = cand.get("matched") or {}
            inv_pos = 1.0 / float(cand.get("pos") or 1)
        else:
            matched = cand.get("f1_matched")
            if matched is None:
                raise ValueError(
                    f"feature-set={feature_set} 需要候选自带 f1_matched（uid={cand.get('uid')!r}）"
                )
            inv_pos = 1.0 / float(index + 1)
        feats.append([
            (score / lex_max) if lex_max > 0 else 0.0,
            0.0 if missing else float(vec_score),
            0.0 if missing else 1.0,
            inv_pos,
            *(1.0 if matched.get(dim) else 0.0 for dim in DIM_SLOTS),
            raw_required,
            *query_block,
        ])
    matrix = np.asarray(feats, dtype=np.float32).reshape(len(feats), len(names))
    return matrix, lex, vec


def label_value(cand: dict, mode: str) -> float | None:
    """Relevance label for one candidate; None means the sample is dropped."""
    grade = float(cand.get("grade") or 0) / 2.0
    if mode == "det":
        return grade
    semantic = cand.get("llm_semantic")
    if mode == "sem":
        return None if semantic is None else float(semantic) / 3.0
    if semantic is None:
        return grade
    return 0.5 * grade + 0.5 * float(semantic) / 3.0


def build_training_matrix(
    rows: list[dict], mode: str, feature_set: str = "f0"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Stack labelled candidates into (X, y, group) arrays; ``group`` indexes the query row."""
    xs: list[np.ndarray] = []
    ys: list[float] = []
    groups: list[int] = []
    dropped = 0
    semantic_seen = 0
    vec_seen = 0
    candidates_seen = 0
    for group, row in enumerate(rows):
        feats, _lex, _vec = candidate_features(row, feature_set)
        for i, cand in enumerate(row.get("candidates") or []):
            candidates_seen += 1
            if cand.get("llm_semantic") is not None:
                semantic_seen += 1
            if cand.get("vec_score") is not None:
                vec_seen += 1
            value = label_value(cand, mode)
            if value is None:
                dropped += 1
                continue
            xs.append(feats[i])
            ys.append(value)
            groups.append(group)
    matrix = np.asarray(xs, dtype=np.float32).reshape(len(xs), len(feature_names(feature_set)))
    stats = {
        "labelled_candidates": len(ys),
        "dropped_unlabelled": dropped,
        "candidates_seen": candidates_seen,
        "semantic_coverage": round(semantic_seen / candidates_seen, 4) if candidates_seen else 0.0,
        "vec_coverage": round(vec_seen / candidates_seen, 4) if candidates_seen else 0.0,
    }
    return matrix, np.asarray(ys, dtype=np.float64), np.asarray(groups, dtype=np.int64), stats


def eval_units(rows: list[dict], mode: str, feature_set: str = "f0") -> list[dict]:
    """Per-query arrays needed for ranking metrics, restricted to labelled candidates.

    ``query_id`` / ``cluster_id`` / ``uids`` ride along for the paired-analysis tooling
    (``--per-query-out``); they never enter a metric computation."""
    units = []
    for row in rows:
        feats, lex, vec = candidate_features(row, feature_set)
        candidates = row.get("candidates") or []
        labels = [label_value(c, mode) for c in candidates]
        valid = [i for i, value in enumerate(labels) if value is not None]
        if not valid:
            continue
        units.append({
            "feats": feats[valid],
            "gains": np.asarray([2.0 ** labels[i] - 1.0 for i in valid], dtype=np.float64),
            "lex": np.asarray([lex[i] for i in valid], dtype=np.float64),
            "vec": np.asarray([vec[i] for i in valid], dtype=np.float64),
            "pos": np.asarray([float(candidates[i].get("pos") or 1) for i in valid], dtype=np.float64),
            "query_id": row.get("query_id"),
            "cluster_id": row.get("cluster_id"),
            "uids": [str(candidates[i].get("uid") or "") for i in valid],
        })
    return units


def _minmax(values: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1]; missing (non-finite) entries become 0."""
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    low = values[finite].min()
    high = values[finite].max()
    if high - low < 1e-12:
        return np.where(finite, 0.5, 0.0)
    return np.where(finite, (values - low) / (high - low), 0.0)


def _rank_desc(values: np.ndarray) -> np.ndarray:
    """1-based average ranks, highest value ranked first (ties share a rank)."""
    order = np.argsort(-values, kind="stable")
    ranked = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and ranked[j + 1] == ranked[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    return ranks


def lex_order(unit: dict) -> np.ndarray:
    """The retrieval pipeline's own order: lex_score descending, then original pos."""
    return np.lexsort((unit["pos"], -unit["lex"]))


def linear_order(unit: dict) -> np.ndarray:
    """Production linear fusion of min-max lex and vec at alpha=0.5.

    A query with no vec scores gets all-zero vec_norm, so its score stays monotone
    in lex_norm and the query degrades to pure lex order."""
    score = LINEAR_ALPHA * _minmax(unit["lex"]) + (1.0 - LINEAR_ALPHA) * _minmax(unit["vec"])
    return np.argsort(-score, kind="stable")


def rrf_order(unit: dict) -> np.ndarray:
    """Reciprocal rank fusion of the lex and vec rankings at k=RRF_K.

    Candidates without a vec score rank last; a query with no vec scores at all
    fuses the lex ranking alone."""
    lex_rank = _rank_desc(unit["lex"])
    if np.isfinite(unit["vec"]).any():
        vec_rank = _rank_desc(np.where(np.isfinite(unit["vec"]), unit["vec"], -np.inf))
        score = 1.0 / (RRF_K + lex_rank) + 1.0 / (RRF_K + vec_rank)
    else:
        score = 1.0 / (RRF_K + lex_rank)
    return np.argsort(-score, kind="stable")


BASELINES: dict[str, OrderFn] = {
    "lex order": lex_order,
    f"linear a={LINEAR_ALPHA}": linear_order,
    f"rrf k={RRF_K}": rrf_order,
}


def _ndcg_at_k(gains_in_order: np.ndarray, k: int) -> float:
    m = min(k, len(gains_in_order))
    if m == 0:
        return 0.0
    discount = 1.0 / np.log2(np.arange(2, m + 2))
    dcg = float(np.dot(gains_in_order[:m], discount))
    ideal = np.sort(gains_in_order)[::-1]
    idcg = float(np.dot(ideal[:m], discount))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(units: list[dict], order_fn: OrderFn) -> dict:
    """Mean nDCG@5 / nDCG@10 / MRR over queries; MRR skips queries with no positive gain."""
    ndcg5 = ndcg10 = mrr = 0.0
    mrr_queries = 0
    for unit in units:
        gains = unit["gains"][order_fn(unit)]
        ndcg5 += _ndcg_at_k(gains, 5)
        ndcg10 += _ndcg_at_k(gains, 10)
        hits = np.nonzero(gains > 0)[0]
        if hits.size:
            mrr += 1.0 / float(hits[0] + 1)
            mrr_queries += 1
    n = len(units)
    return {
        "nDCG@5": round(ndcg5 / n, 4) if n else None,
        "nDCG@10": round(ndcg10 / n, 4) if n else None,
        "MRR": round(mrr / mrr_queries, 4) if mrr_queries else None,
        "queries": n,
        "mrr_queries": mrr_queries,
    }


def model_order(score_fn: ScoreFn) -> OrderFn:
    def _order(unit: dict) -> np.ndarray:
        return np.argsort(-score_fn(unit["feats"]), kind="stable")

    return _order


def train_pointwise(X: np.ndarray, y: np.ndarray, cols: tuple[int, ...], seed: int) -> ScoreFn:
    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(X[:, cols], y)

    def _score(feats: np.ndarray) -> np.ndarray:
        return model.predict(feats[:, cols])

    return _score


def _pair_indices(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Within-query pairs with differing labels, as (i, j, sign(y_i - y_j)) arrays."""
    by_group: dict[int, list[int]] = {}
    for idx, group in enumerate(groups):
        by_group.setdefault(int(group), []).append(idx)
    left: list[int] = []
    right: list[int] = []
    signs: list[float] = []
    for members in by_group.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                ia, ib = members[a], members[b]
                if y[ia] == y[ib]:
                    continue
                left.append(ia)
                right.append(ib)
                signs.append(1.0 if y[ia] > y[ib] else -1.0)
    i_arr = np.asarray(left, dtype=np.int64)
    j_arr = np.asarray(right, dtype=np.int64)
    s_arr = np.asarray(signs, dtype=np.float32)
    if len(i_arr) > PAIR_CAP:
        keep = rng.choice(len(i_arr), size=PAIR_CAP, replace=False)
        i_arr, j_arr, s_arr = i_arr[keep], j_arr[keep], s_arr[keep]
    return i_arr, j_arr, s_arr


def _build_mlp(input_dim: int) -> torch.nn.Sequential:
    """RankNet MLP: input_dim -> RANKNET_HIDDEN -> 1, ReLU between layers."""
    layers: list[torch.nn.Module] = []
    last = input_dim
    for width in RANKNET_HIDDEN:
        layers += [torch.nn.Linear(last, width), torch.nn.ReLU()]
        last = width
    layers.append(torch.nn.Linear(last, 1))
    return torch.nn.Sequential(*layers)


def enable_deterministic_mode(seed: int) -> dict:
    """Pin the numerical environment so that "same data + same seed" means bit-identical weights.

    Seeding alone is not enough: the pairwise-loss reduction and the matmuls run through
    intra-op thread pools, and the summation order varies per run — that is the measured
    ~5.7e-4 heldout nDCG@10 spread across retrains. Forcing one thread makes the reduction
    order fixed; the seeds cover the remaining consumers (``random``, ``numpy`` and the
    torch initialiser). ``torch.use_deterministic_algorithms`` is attempted as an extra
    guarantee, but is not required here (the MLP runs on plain CPU ops) and never fatal:
    if this torch build refuses, single-thread + seeds stand on their own.

    Returns the applied settings for the run log; caller decides whether to print it.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    settings = {"seed": seed, "torch_threads": torch.get_num_threads(), "deterministic_algorithms": False}
    try:
        torch.use_deterministic_algorithms(True)
        settings["deterministic_algorithms"] = True
    except Exception as exc:                                                 
        settings["deterministic_algorithms_error"] = repr(exc)
    return settings


def train_ranknet(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cols: tuple[int, ...],
    seed: int,
) -> tuple[ScoreFn | None, dict, dict | None]:
    """Fit the pairwise RankNet MLP.

    Returns (score_fn, info, artifacts). score_fn and artifacts are None when no
    differing-label pair exists, so there is nothing to train or export; otherwise
    artifacts carries the state_dict and the normalisation stats used at inference."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    i_arr, j_arr, s_arr = _pair_indices(y, groups, rng)
    info = {"pairs": int(len(i_arr)), "epochs": RANKNET_EPOCHS, "lr": RANKNET_LR}
    if len(i_arr) == 0:
        info["note"] = "no differing-label pairs; model not trained"
        return None, info, None

    sub = X[:, cols]
    mean = sub.mean(axis=0)
    std = np.where(sub.std(axis=0) < 1e-8, 1.0, sub.std(axis=0))
    normalized = ((sub - mean) / std).astype(np.float32)

    net = _build_mlp(len(cols))
    optimizer = torch.optim.Adam(net.parameters(), lr=RANKNET_LR)
    features = torch.from_numpy(normalized)
    i_t = torch.from_numpy(i_arr)
    j_t = torch.from_numpy(j_arr)
    s_t = torch.from_numpy(s_arr)
    for _ in range(RANKNET_EPOCHS):
        optimizer.zero_grad()
        scores = net(features).squeeze(1)
        loss = -torch.nn.functional.logsigmoid(s_t * (scores[i_t] - scores[j_t])).mean()
        loss.backward()
        optimizer.step()
    info["final_loss"] = round(float(loss.item()), 4)
    net.eval()

    def _score(feats: np.ndarray) -> np.ndarray:
        scaled = ((feats[:, cols] - mean) / std).astype(np.float32)
        with torch.no_grad():
            return net(torch.from_numpy(scaled)).squeeze(1).numpy()

    artifacts = {
        "state_dict": {key: value.detach().clone() for key, value in net.state_dict().items()},
        "mean": mean,
        "std": std,
        "cols": cols,
    }
    return _score, info, artifacts


def _per_query_ndcg10(units: list[dict], order_fn: OrderFn) -> np.ndarray:
    return np.asarray([_ndcg_at_k(u["gains"][order_fn(u)], 10) for u in units], dtype=np.float64)


def export_ranknet(
    export_dir: Path,
    artifacts: dict,
    label_mode: str,
    seed: int,
    train_counts: dict,
    heldout_metrics: dict,
    feature_set: str = "f0",
) -> dict:
    """Write the RankNet MLP state_dict and its feature contract into ``export_dir``."""
    export_dir.mkdir(parents=True, exist_ok=True)
    names = feature_names(feature_set)
    torch.save(artifacts["state_dict"], export_dir / "ltr_mlp.pt")
    spec = {
        "feature_order": [names[c] for c in artifacts["cols"]],
        "feature_set": feature_set,
        "label_mode": label_mode,
        "seed": seed,
        "feature_mean": [float(value) for value in artifacts["mean"]],
        "feature_std": [float(value) for value in artifacts["std"]],
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "train_counts": train_counts,
        "heldout_metrics": heldout_metrics,
        "model": {
            "type": "ranknet_mlp",
            "input_dim": len(artifacts["cols"]),
            "hidden_layers": list(RANKNET_HIDDEN),
            "activation": "relu",
            "output_dim": 1,
            "loss": "pairwise_logistic",
        },
    }
    (export_dir / "ltr_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return spec


def selfcheck_export(export_dir: Path, units: list[dict], expected_order: OrderFn,
                     feature_set: str = "f0") -> float:
    """Re-load the exported MLP and spec, re-score heldout, return max |delta nDCG@10|.

    The normalisation stats are cast back to float32 exactly as training used them,
    so a faithful round-trip reproduces the in-training per-query ranking."""
    spec = json.loads((export_dir / "ltr_spec.json").read_text(encoding="utf-8"))
    names = feature_names(feature_set)
    cols = tuple(names.index(name) for name in spec["feature_order"])
    mean = np.asarray(spec["feature_mean"], dtype=np.float32)
    std = np.asarray(spec["feature_std"], dtype=np.float32)
    net = _build_mlp(spec["model"]["input_dim"])
    net.load_state_dict(torch.load(export_dir / "ltr_mlp.pt", map_location="cpu", weights_only=True))
    net.eval()

    def _score(feats: np.ndarray) -> np.ndarray:
        scaled = ((feats[:, cols] - mean) / std).astype(np.float32)
        with torch.no_grad():
            return net(torch.from_numpy(scaled)).squeeze(1).numpy()

    reloaded = _per_query_ndcg10(units, model_order(_score))
    expected = _per_query_ndcg10(units, expected_order)
    if reloaded.size == 0:
        return 0.0
    return float(np.max(np.abs(reloaded - expected)))


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _parse_eval_sets(entries: "list[str]") -> "dict[str, Path]":
    """Parse repeated ``NAME=PATH`` options into an ordered mapping (duplicate names rejected)."""
    out: "dict[str, Path]" = {}
    for entry in entries:
        name, sep, raw = entry.partition("=")
        name = name.strip()
        if not sep or not name or not raw.strip():
            raise SystemExit(f"! --eval-set 需要 NAME=PATH 形式，收到 {entry!r}")
        if name in out:
            raise SystemExit(f"! --eval-set 名字重复：{name!r}")
        out[name] = Path(raw.strip())
    return out


def _per_query_rows(eval_set: str, units: list[dict], orders: "dict[str, OrderFn | None]") -> "list[dict]":
    """One row per eval query: nDCG@10 under each ranking plus the top-10 uids.

    The uids let the paired-analysis tooling attach per-candidate facts that the
    trainer deliberately knows nothing about (e.g. hard-constraint violations judged
    against the row's intended constraints) without re-implementing the ranking."""
    rows: "list[dict]" = []
    for unit in units:
        row: dict = {
            "eval_set": eval_set,
            "query_id": unit.get("query_id"),
            "cluster_id": unit.get("cluster_id"),
            "n_valid": int(len(unit["gains"])),
            "ndcg10": {},
            "top10_uids": {},
        }
        for key, order_fn in orders.items():
            if order_fn is None:
                continue
            order = order_fn(unit)
            row["ndcg10"][key] = round(_ndcg_at_k(unit["gains"][order], 10), 6)
            uids = unit.get("uids") or []
            row["top10_uids"][key] = [uids[i] for i in order[:10]] if uids else []
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    parser.add_argument(
        "--label",
        choices=("det", "sem", "mix"),
        default=None,
        help="set the train AND eval label at once (kept for compatibility); mutually exclusive "
        "with --train-label / --eval-label",
    )
    parser.add_argument("--train-label", choices=("det", "sem", "mix"), default=None,
                        help="label used to build the training pairs (default: --label, else mix)")
    parser.add_argument("--eval-label", choices=("det", "sem", "mix"), default=None,
                        help="label held FIXED across arms when comparing training targets "
                             "(default: same as the train label)")
    parser.add_argument("--feature-set", choices=tuple(FEATURE_SETS), default="f0",
                        help="feature provenance: f0 = as harvested (default, unchanged), "
                             "f1 = production-replayed overlay, f2 = f1 + query-condition block, "
                             "f3 = production-constructed f3_feats overlay (export allowed)")
    parser.add_argument("--eval-feature-set", choices=tuple(FEATURE_SETS), default=None,
                        help="feature provenance for EVALUATION only (default: same as "
                             "--feature-set). Lets a model trained on one recipe be scored under "
                             "another recipe's semantics — e.g. train f0, serve f3 (production). "
                             "Column count must match the train feature set (fail-closed).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="export the full-feature RankNet MLP (ltr_mlp.pt + ltr_spec.json) to this directory",
    )
    parser.add_argument(
        "--mix-in",
        type=Path,
        action="append",
        default=[],
        help="extra labelled JSONL to merge into the TRAIN split only (repeatable); "
        "dedupes by query_id, more candidates wins. Never touches heldout.",
    )
    parser.add_argument(
        "--eval-set",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="extra fixed evaluation set: one JSONL file read as-is (no layer merging), "
             "repeatable; reported alongside the in-dir heldout split",
    )
    parser.add_argument(
        "--per-query-out",
        type=Path,
        default=None,
        help="write per-query nDCG@10 detail (one JSON object per eval set x query) for the "
             "paired-analysis tooling; requires the same files --eval-set points at",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="pin the numerical environment (single-threaded torch + random/numpy/torch seeds "
        "from --seed) so that reruns on the same data are bit-identical; off by default",
    )
    args = parser.parse_args()

    if args.label is not None and (args.train_label or args.eval_label):
        raise SystemExit("! --label 与 --train-label/--eval-label 互斥：前者同时设两个标签")
    train_label = args.train_label or args.label or "mix"
    eval_label = args.eval_label or args.label or train_label
    feature_set = args.feature_set
    eval_feature_set = args.eval_feature_set or feature_set
    names = feature_names(feature_set)
    if len(feature_names(eval_feature_set)) != len(names):
        raise SystemExit(
            f"! --eval-feature-set={eval_feature_set}（{len(feature_names(eval_feature_set))} 列）"
            f"与 --feature-set={feature_set}（{len(names)} 列）列数不同——"
            "评测特征矩阵无法被训练侧的打分函数消费（fail-closed）")
    ablations = ablation_cols(feature_set)
    extra_eval_paths = _parse_eval_sets(args.eval_set)
    for eval_name, eval_path in extra_eval_paths.items():
        if not eval_path.is_file():
            raise SystemExit(f"! --eval-set {eval_name} 指向的文件不存在：{eval_path}")

    deterministic = enable_deterministic_mode(args.seed) if args.deterministic else None
    in_dir: Path = args.in_dir
    out_path: Path = args.out or (in_dir / "_ltr_metrics.json")
    started = time.time()

    train_rows, train_src = load_split(in_dir, "train")
    held_rows, held_src = load_split(in_dir, "heldout")
    mix_stats: dict = {}
    if args.mix_in:
        train_rows, mix_stats = merge_mix_in(train_rows, args.mix_in)
        train_src = {**train_src, "mix_in": mix_stats}
    X, y, groups, label_stats = build_training_matrix(train_rows, train_label, feature_set)
    units = eval_units(held_rows, eval_label, eval_feature_set)
    extra_units = {
        name: (eval_units(_read_jsonl(path), eval_label, eval_feature_set), path)
        for name, path in extra_eval_paths.items()
    }

    print(f"train_label={train_label} eval_label={eval_label} feature_set={feature_set} "
          f"eval_feature_set={eval_feature_set} seed={args.seed} in_dir={in_dir}")
    if deterministic is not None:
        print(f"  deterministic: {deterministic}")
    print(f"  train: {len(train_rows)} queries, {label_stats['labelled_candidates']} labelled candidates "
          f"({label_stats['dropped_unlabelled']} dropped), vec coverage {label_stats['vec_coverage']}, "
          f"semantic coverage {label_stats['semantic_coverage']}")
    print(f"  heldout: {len(held_rows)} queries, {len(units)} usable for ranking")
    for name, (set_units, path) in extra_units.items():
        print(f"  eval-set {name}: {len(set_units)} usable for ranking ({path})")
    print(f"  layers: train={train_src['layers']} heldout={held_src['layers']}")
    if mix_stats:
        print(f"  mix-in: {mix_stats} -> train now {len(train_rows)} queries")

    results: dict = {"baselines": {}, "models": {"hist_gbdt": {}, "ranknet_mlp": {}}}
    table: list[tuple[str, str, dict]] = []
    full_artifacts: dict | None = None
    full_order: OrderFn | None = None
    full_info: dict | None = None
    full_metrics: dict | None = None
    hist_full_order: OrderFn | None = None
    ranknet_full_order: OrderFn | None = None
    per_query: "dict[str, dict]" = {}
    if units:
        for name, order_fn in BASELINES.items():
            metrics = evaluate(units, order_fn)
            results["baselines"][name] = metrics
            table.append(("baseline", name, metrics))

    if X.shape[0] == 0 or not units:
        print("  ! no labelled candidates for this --train-label mode; models skipped")
        for name in results["models"]:
            for ablation in ablations:
                results["models"][name][ablation] = {"nDCG@5": None, "nDCG@10": None, "MRR": None}
    else:
        for ablation, cols in ablations.items():
            score_fn = train_pointwise(X, y, cols, args.seed)
            metrics = evaluate(units, model_order(score_fn))
            results["models"]["hist_gbdt"][ablation] = metrics
            table.append(("hist_gbdt", ablation, metrics))
            if ablation == "full":
                hist_full_order = model_order(score_fn)
        for ablation, cols in ablations.items():
            score_fn, info, artifacts = train_ranknet(X, y, groups, cols, args.seed)
            if score_fn is None:
                metrics = {"nDCG@5": None, "nDCG@10": None, "MRR": None, **info}
                order_fn: OrderFn | None = None
            else:
                order_fn = model_order(score_fn)
                metrics = {**evaluate(units, order_fn), **info}
            results["models"]["ranknet_mlp"][ablation] = metrics
            table.append(("ranknet_mlp", ablation, metrics))
            if ablation == "full" and artifacts is not None and order_fn is not None:
                full_artifacts, full_order, full_info, full_metrics = artifacts, order_fn, info, metrics
                ranknet_full_order = order_fn
        if args.per_query_out is not None:
            per_query["heldout"] = _per_query_rows(
                "heldout", units, {"lex": lex_order, "hist_gbdt_full": hist_full_order,
                                   "ranknet_mlp_full": ranknet_full_order})
            for name, (set_units, _path) in extra_units.items():
                per_query[name] = _per_query_rows(
                    name, set_units, {"lex": lex_order, "hist_gbdt_full": hist_full_order,
                                      "ranknet_mlp_full": ranknet_full_order})

    extra_results: dict = {}
    for name, (set_units, _path) in extra_units.items():
        if not set_units or X.shape[0] == 0:
            extra_results[name] = {"queries": len(set_units), "note": "no labelled candidates; skipped"}
            continue
        extra_results[name] = {
            "queries": len(set_units),
            "baselines": {base: evaluate(set_units, fn) for base, fn in BASELINES.items()},
            "ranknet_mlp_full": (
                evaluate(set_units, ranknet_full_order) if ranknet_full_order is not None else None
            ),
            "hist_gbdt_full": (
                evaluate(set_units, hist_full_order) if hist_full_order is not None else None
            ),
        }
    if extra_results:
        print()
        print(f"{'eval-set':<16}{'ranknet_full':>13}{'hist_full':>11}{'lex':>9}{'queries':>9}")
        print("-" * 58)
        for name, block in extra_results.items():
            rn = (block.get("ranknet_mlp_full") or {}).get("nDCG@10")
            hg = (block.get("hist_gbdt_full") or {}).get("nDCG@10")
            lex = (block.get("baselines", {}).get("lex order") or {}).get("nDCG@10")
            print(f"{name:<16}{_fmt(rn):>13}{_fmt(hg):>11}{_fmt(lex):>9}{block.get('queries', 0):>9}")

    print()
    print(f"{'model':<14}{'features':<16}{'nDCG@5':>9}{'nDCG@10':>9}{'MRR':>9}")
    print("-" * 57)
    for name, features, metrics in table:
        print(f"{name:<14}{features:<16}{_fmt(metrics.get('nDCG@5')):>9}"
              f"{_fmt(metrics.get('nDCG@10')):>9}{_fmt(metrics.get('MRR')):>9}")

    if args.export:
        if feature_set == "f2":
                                                                      
                                                                 
                                                   
                                                                     
                                                      
            print("  ! --export 拒绝 feature-set=f2：生产推理侧只有 9 维消费点，实验臂只在实验目录内消化")
            return 1
        if full_artifacts is None or full_order is None or full_info is None or full_metrics is None:
            print("  ! --export given but the full-feature RankNet model was not trained; export skipped")
        else:
            spec = export_ranknet(
                args.export,
                full_artifacts,
                train_label,
                args.seed,
                {"queries": len(train_rows), "pairs": full_info["pairs"]},
                {key: full_metrics.get(key) for key in ("nDCG@5", "nDCG@10", "MRR")},
                feature_set,
            )
            max_delta = selfcheck_export(args.export, units, full_order, feature_set)
            print(f"  export: {args.export} -> ltr_mlp.pt + ltr_spec.json "
                  f"({spec['model']['input_dim']} features: {', '.join(spec['feature_order'])})")
            print(f"  export self-check: max per-query |delta nDCG@10| = {max_delta:.3e}")
            if max_delta > 1e-6:
                print("  ! export self-check FAILED: reloaded ranking differs from in-training ranking")
                return 1

    if args.per_query_out is not None:
        args.per_query_out.parent.mkdir(parents=True, exist_ok=True)
        args.per_query_out.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for rows in per_query.values() for row in rows),
            encoding="utf-8",
        )
        print(f"  per-query: {args.per_query_out} ({sum(len(rows) for rows in per_query.values())} rows)")

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "label_mode": train_label,
        "train_label": train_label,
        "eval_label": eval_label,
        "feature_set": feature_set,
        "eval_feature_set": eval_feature_set,
        "seed": args.seed,
        "deterministic": deterministic,
        "in_dir": str(in_dir),
        "out_path": str(out_path),
        "feature_names": list(names),
        "ablations": {name: [names[c] for c in cols] for name, cols in ablations.items()},
        "data": {
            "train_queries": len(train_rows),
            "heldout_queries": len(held_rows),
            "heldout_ranked_queries": len(units),
            "train_label_stats": label_stats,
            "sources": {"train": train_src, "heldout": held_src},
        },
        "results": results,
        "extra_eval_sets": {name: str(path) for name, path in extra_eval_paths.items()},
        "extra_eval_results": extra_results,
        "elapsed_s": round(time.time() - started, 2),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path} in {payload['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
