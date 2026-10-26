                       
"""System-One 分流器 L0 基线：TF-IDF(char 2-4gram) + LogisticRegression（CPU 零门槛）。

回答的问题（见 研究/Jev分流器可行性-2026-09-19.md）：
  1. 一个零神经网络的线性模型，在 LLM 共识标签上能学到多准？（heldout 一致率/F1）
  2. 置信度能不能校准？（温度缩放前后 ECE）
  3. 级联可不可行？（置信档 覆盖率-准确率曲线 → 选 τ：高置信直采、低置信升级 LLM 共识）
  4. LLM 共识标签本身质量如何？（gold vs LLM 一致率、票间一致率）

输入  research/route_eval/route_labels.jsonl（+ route_corpus.jsonl 补 gold）
输出  research/route_eval/l0/metrics.json + l0/report.md + l0/model.joblib

Usage:
  .venv/Scripts/python.exe research/route_eval/train_l0_router.py
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

AGENT_ROOT = Path(__file__).resolve().parents[2]
LABELS = AGENT_ROOT / "research" / "route_eval" / "route_labels.jsonl"
OUT_DIR = AGENT_ROOT / "research" / "route_eval" / "l0"
CLASSES = ["search", "action", "general"]


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("route_llm") in CLASSES and any(v.get("ok") for v in r.get("votes", [])):
            rows.append(r)
    return rows


def vote_stats(rows: list[dict]) -> dict:
    """LLM 共识自身健康度：票间一致率（前 2 票）、加投率、兜底率、gold 一致率。"""
    n = len(rows)
    agree2 = sum(1 for r in rows if r["decided_by"] == "agree2")
    majority3 = sum(1 for r in rows if r["decided_by"] == "majority3")
    fallback = sum(1 for r in rows if r["decided_by"] == "fallback_general")
    gold_match = sum(1 for r in rows if r.get("route_gold") and r["route_gold"] == r["route_llm"])
    gold_known = sum(1 for r in rows if r.get("route_gold"))
    by_class_gold: dict[str, dict] = {}
    for r in rows:
        g = r.get("route_gold")
        if not g:
            continue
        d = by_class_gold.setdefault(g, {"n": 0, "match": 0})
        d["n"] += 1
        d["match"] += int(g == r["route_llm"])
    return {
        "n": n, "agree2_rate": agree2 / max(n, 1), "majority3_rate": majority3 / max(n, 1),
        "fallback_rate": fallback / max(n, 1),
        "gold_agreement": gold_match / max(gold_known, 1), "gold_known": gold_known,
        "gold_by_class": {k: {**v, "rate": v["match"] / max(v["n"], 1)}
                          for k, v in by_class_gold.items()},
    }


def ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """多分类 ECE（max-prob 口径）：confidence = max prob，accuracy = 是否命中。"""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum():
            total += mask.sum() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total / len(conf))


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """温度缩放：在标定集上对 NLL 做网格 + 黄金分割精修。"""
    def nll(t: float) -> float:
        z = logits / max(t, 1e-3)
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(y_true)), y_true].mean())
    grid = np.linspace(0.3, 6.0, 40)
    best = min(grid, key=nll)
    lo, hi = max(0.1, best - 0.4), best + 0.4
    gr = (5 ** 0.5 - 1) / 2
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    for _ in range(40):
        if nll(c) < nll(d):
            hi = d
        else:
            lo = c
        c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    return float((lo + hi) / 2)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cascade_curve(probs: np.ndarray, y_true: np.ndarray) -> list[dict]:
    """置信档 覆盖率-准确率曲线：按 max-prob 从高到低，逐档报 准确率@覆盖率。"""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    order = np.argsort(-conf)
    correct = (pred[order] == y_true[order]).astype(float)
    curve = []
    n = len(order)
    for cov in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
        k = max(1, int(round(cov * n)))
        curve.append({"coverage": cov, "accuracy": float(correct[:k].mean()),
                      "tau_at_edge": float(conf[order[k - 1]])})
    return curve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=str(LABELS))
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--classes", default="",
                        help="逗号分隔的类别表（默认三分类 search,action,general；"
                             "二元写闸传 write,read）")
    args = parser.parse_args()
    if args.classes:
        CLASSES[:] = [c.strip() for c in args.classes.split(",") if c.strip()]
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    rows = load_rows(Path(args.labels))
    if len(rows) < 100:
        print(f"only {len(rows)} labeled rows; need >=100", file=sys.stderr)
        return 2
    texts = [r["utterance"] for r in rows]
    y = np.array([CLASSES.index(r["route_llm"]) for r in rows])

    stats = vote_stats(rows)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)

    idx = np.arange(len(rows))
    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
        idx, y, test_size=0.4, random_state=args.seed, stratify=y)
    idx_cal, idx_te, y_cal, y_te = train_test_split(
        idx_tmp, y_tmp, test_size=0.5, random_state=args.seed, stratify=y_tmp)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2)),
        ("clf", LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000)),
    ])
    pipe.fit([texts[i] for i in idx_tr], y_tr)

    logits_cal = pipe.decision_function([texts[i] for i in idx_cal])
    logits_te = pipe.decision_function([texts[i] for i in idx_te])
    if logits_cal.ndim == 1:                                   
        logits_cal = np.stack([-logits_cal, logits_cal], axis=1)
        logits_te = np.stack([-logits_te, logits_te], axis=1)
    temp = fit_temperature(logits_cal, y_cal)
    probs_te_raw = softmax(logits_te)
    probs_te_cal = softmax(logits_te / temp)

    pred_te = probs_te_raw.argmax(axis=1)
    acc = accuracy_score(y_te, pred_te)
    f1 = f1_score(y_te, pred_te, average="macro")
    cm = confusion_matrix(y_te, pred_te).tolist()
    per_class = classification_report(y_te, pred_te, target_names=CLASSES,
                                      output_dict=True, zero_division=0)
    metrics = {
        "n_total": len(rows), "n_train": len(idx_tr), "n_cal": len(idx_cal),
        "n_test": len(idx_te), "seed": args.seed,
        "label_health": stats,
        "test_accuracy": float(acc), "test_f1_macro": float(f1),
        "confusion_matrix": {"classes": CLASSES, "matrix": cm},
        "per_class": {c: {"precision": per_class[c]["precision"],
                          "recall": per_class[c]["recall"],
                          "f1": per_class[c]["f1-score"]} for c in CLASSES},
        "temperature": temp,
        "ece_raw": ece(probs_te_raw, y_te), "ece_calibrated": ece(probs_te_cal, y_te),
        "cascade_raw": cascade_curve(probs_te_raw, y_te),
        "cascade_calibrated": cascade_curve(probs_te_cal, y_te),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
                                           
    disagree = [{"qid": r["qid"], "utterance": r["utterance"], "route_gold": r["route_gold"],
                 "route_llm": r["route_llm"], "decided_by": r["decided_by"],
                 "votes": [(v.get("temperature"), v.get("route"), v.get("ok"))
                           for v in r.get("votes", [])]}
                for r in rows if r.get("route_gold") and r["route_gold"] != r["route_llm"]]
    (out_dir / "gold_disagreements.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in disagree), encoding="utf-8")
    joblib.dump({"pipeline": pipe, "classes": CLASSES, "temperature": temp},
                out_dir / "model.joblib")
    print(f"\naccuracy={acc:.4f} f1_macro={f1:.4f} temp={temp:.2f} "
          f"ece {metrics['ece_raw']:.4f}->{metrics['ece_calibrated']:.4f}", flush=True)
    print("cascade(calibrated):", flush=True)
    for row in metrics["cascade_calibrated"]:
        print(f"  cov={row['coverage']:.2f} acc={row['accuracy']:.4f} "
              f"tau~{row['tau_at_edge']:.3f}", flush=True)
    print(f"-> {out_dir / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
