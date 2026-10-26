                       
"""L0 模型生产导出器（2026-09-19）：sklearn joblib → 纯 Python JSON 工件 + parity + 工作点。

做什么：
1. 复现 train_l0_router.py 的同一种子/分层切分，在 **cal** 上选工作点、在 **test** 上报告：
   - 路由模型（级联 kind="cascade"）：τ = cal 置信度第 (1-coverage_goal) 分位
     → test 上报 覆盖率/一致率（口径与 metrics.json 的 cascade_calibrated 一致，
     但「选在 cal、报在 test」杜绝了 metrics.json 选在 test 的轻微乐观）。
   - 检索闸（kind="recall_floor"）：τ = cal 上 retrieve 召回 ≥ recall_goal 的最大阈值
     → test 上报 retrieve 召回 / skip 正确跳过率 / flight 起跑率。
2. 导出纯 Python JSON 工件到 src/dataset_recommender/agent/router_models/。
3. parity：用生产模块 l0_router.score 重载工件，与 sklearn softmax(decision_function/温度)
   在 test 全量 + 边缘字符串上逐点比对，max|Δp| 超容差即失败（不导出半截真相）。

Usage:
  .venv/Scripts/python.exe research/route_eval/export_l0_production.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

AGENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT_ROOT / "src"))

from dataset_recommender.agent import l0_router              

SEED = 20260919
EVAL_DIR = AGENT_ROOT / "research" / "route_eval"
OUT_DIR = AGENT_ROOT / "src" / "dataset_recommender" / "agent" / "router_models"
PARITY_TOL = 1e-9

                                                 
                                                                  
                                                                
                                                       
MODELS = [
    (l0_router.ROUTE_THREE_CLASS, "l0aug_route3", "combined_route3_train.jsonl",
     ["search", "action", "general"],
     {"kind": "cascade", "coverage_goal": 0.90}),
    (l0_router.ROUTE_BINARY, "l0aug_binary", "combined_binary_train.jsonl",
     ["write", "read"],
     {"kind": "cascade", "coverage_goal": 0.90}),
    (l0_router.PRELIM_GATE, "l0aug_prelim", "combined_prelim_train.jsonl",
     ["retrieve", "skip"],
                                                     
                                                            
                                                       
                                             
     {"kind": "recall_floor", "recall_goal": 0.999, "positive": "retrieve"}),
]

EDGE_STRINGS = ["", "a", "ab", "肺癌", "GSE123456", "find human lung scRNA-seq data",
                "fie yan de dan xi bao shu ju", "🧬🧬🧬", "换一换", "  空 格\t和\n换行  ",
                "肺" * 300, "check 10x updates then 入库"]


def load_rows(path: Path, classes: list[str]) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("route_llm") in classes and any(v.get("ok") for v in r.get("votes", [])):
            rows.append(r)
    return rows


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def split_probs(pipe, classes: list[str], temp: float, texts: list[str],
               y: np.ndarray, seed: int):
    """复现 train_l0_router 的 60/20/20 stratify 切分，返回 (cal_probs, y_cal, test_probs, y_te)。"""
    idx = np.arange(len(texts))
    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
        idx, y, test_size=0.4, random_state=seed, stratify=y)
    idx_cal, idx_te, y_cal, y_te = train_test_split(
        idx_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)
    logits_cal = pipe.decision_function([texts[i] for i in idx_cal])
    logits_te = pipe.decision_function([texts[i] for i in idx_te])
    if logits_cal.ndim == 1:                                       
        logits_cal = np.stack([-logits_cal, logits_cal], axis=1)
        logits_te = np.stack([-logits_te, logits_te], axis=1)
    return softmax(logits_cal / temp), y_cal, softmax(logits_te / temp), y_te, idx_te


def pick_cascade_tau(cal_probs: np.ndarray, coverage_goal: float) -> float:
    conf = np.sort(cal_probs.max(axis=1))
    k = max(0, int(np.floor((1.0 - coverage_goal) * len(conf))) - 1)
    return float(conf[min(max(k, 0), len(conf) - 1)])


def pick_recall_floor_tau(cal_probs: np.ndarray, y_cal: np.ndarray, pos_idx: int,
                          recall_goal: float) -> float:
    """cal 上 retrieve 召回 ≥ goal 的最大 P(positive) 阈值。"""
    p_pos = cal_probs[:, pos_idx]
    pos_scores = np.sort(p_pos[y_cal == pos_idx])
    n_miss = int(np.floor((1.0 - recall_goal) * len(pos_scores)))
    return float(pos_scores[min(n_miss, len(pos_scores) - 1)])


def main() -> int:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")                              
    except Exception:
        pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"seed": SEED, "parity_tol": PARITY_TOL, "models": {}}
    l0_router.reset_cache()
    for name, src_dir, labels_file, classes, op_cfg in MODELS:
        bundle = joblib.load(EVAL_DIR / src_dir / "model.joblib")
        pipe, temp = bundle["pipeline"], float(bundle["temperature"])
        rows = load_rows(EVAL_DIR / labels_file, classes)
        texts = [r["utterance"] for r in rows]
        y = np.array([classes.index(r["route_llm"]) for r in rows])
        cal_p, y_cal, te_p, y_te, idx_te = split_probs(pipe, classes, temp, texts, y, SEED)

        if op_cfg["kind"] == "cascade":
            tau = pick_cascade_tau(cal_p, op_cfg["coverage_goal"])
            conf_te, pred_te = te_p.max(axis=1), te_p.argmax(axis=1)
            covered = conf_te >= tau
            operating_point = {
                "kind": "cascade", "tau": tau,
                "coverage_goal": op_cfg["coverage_goal"],
                "test_coverage": float(covered.mean()),
                "test_agreement": float((pred_te[covered] == y_te[covered]).mean()),
                "test_agreement_full": float((pred_te == y_te).mean()),
            }
        else:
            pos_idx = classes.index(op_cfg["positive"])
            tau = pick_recall_floor_tau(cal_p, y_cal, pos_idx, op_cfg["recall_goal"])
            p_pos_te = te_p[:, pos_idx]
            start = p_pos_te >= tau
            is_pos = y_te == pos_idx
            operating_point = {
                "kind": "recall_floor", "tau_retrieve": tau,
                "recall_goal": op_cfg["recall_goal"], "positive": op_cfg["positive"],
                "test_recall": float((start[is_pos]).mean()),
                "test_skip_correct_rate": float((~start[~is_pos]).mean()),
                "test_start_rate": float(start.mean()),
            }

        vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
        artifact = {
            "format": "l0-router-v1",
            "name": name,
            "source": {"joblib": f"research/route_eval/{src_dir}/model.joblib",
                       "labels": f"research/route_eval/{labels_file}",
                       "seed": SEED, "n_rows": len(rows)},
            "classes": classes,
            "temperature": temp,
            "ngram_range": [2, 4], "lowercase": True,
            "vocab": {k: int(v) for k, v in vec.vocabulary_.items()},
            "idf": [float(x) for x in vec.idf_],
            "coef": [[float(x) for x in row] for row in clf.coef_],
            "intercept": [float(x) for x in clf.intercept_],
            "operating_point": operating_point,
        }
        out_path = OUT_DIR / f"{name}.json"
        out_path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")

                                                      
        l0_router.reset_cache()
        model = l0_router._load(name)
        assert model is not None, f"reload failed: {name}"
        max_dp = 0.0
        probe = [texts[i] for i in idx_te] + EDGE_STRINGS
        for text in probe:
            lg = pipe.decision_function([text])
            if lg.ndim == 1:
                lg = np.stack([-lg, lg], axis=1)
            ref = softmax(lg / temp)[0]
            got = model.probs(text)
            max_dp = max(max_dp, float(np.max(np.abs(ref - np.array(got)))))
        report["models"][name] = {
            "artifact": str(out_path.relative_to(AGENT_ROOT)),
            "operating_point": operating_point,
            "parity_max_dp": max_dp, "parity_probes": len(probe),
            "parity_ok": bool(max_dp < PARITY_TOL),
        }
        print(f"[{name}] tau={tau:.6f} op={json.dumps(operating_point)} "
              f"parity_max_dp={max_dp:.3e}", flush=True)
        if max_dp >= PARITY_TOL:
            print(f"PARITY FAIL: {name} max_dp={max_dp}", file=sys.stderr)
            return 2

    report_path = EVAL_DIR / "export_production_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {report_path}", flush=True)
    return 0


import io                                          

if __name__ == "__main__":
    raise SystemExit(main())
