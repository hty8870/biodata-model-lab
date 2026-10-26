                       
"""E7 发射型预测器：规则基线 vs LTR，LFO-CV（2026-09-28，e7_设计-2026-09-28.md）。
纪律：selftest 不过则全线停止（分析器单测纪律）；只读 repo；输出 e7_results_0928.json。
标签：e7_gold_0928.json（B=观测门控写 / A=绝对禁令 / N=无风险）。
主判 E7-1：LTR LFO-CV AUC-PR ≥ R1+0.05 → H2 活；R1 ≥ 0.95 → H2 死。
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GroupKFold
from scipy.sparse import hstack, csr_matrix

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
RE_COND = re.compile(r"如果|若|有新|有没有|没的话|是否有|有更新|没更新")
VIOL_DIMS = {"forbid_steps", "zero_writes"}


                          
def r1_score(text):
    return float(len(RE_COND.findall(text)))


def tool_feats(r):
    return [len(r["write_avail"]), float(r["has_bysource"]), float(r["has_check"]),
            len(r["write_avail"]) + float(r["has_check"])]


def build_matrix(rows, vect=None):
    X_text = [r["utterance"] for r in rows]
    if vect is None:
        vect = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2)
        Xt = vect.fit_transform(X_text)
    else:
        Xt = vect.transform(X_text)
    Xs = csr_matrix(np.array([tool_feats(r) for r in rows], dtype=float))
    return hstack([Xt, Xs]).tocsr(), vect


def lofocv_predict(rows, y, families, Cs=(0.1, 1.0, 10.0), seed=928):
    """Leave-One-Family-Out：折外预测。内层 GroupKFold(3) 选 C。"""
    fams = sorted(set(families))
    oof = np.zeros(len(rows))
    for hold in fams:
        te = [i for i, f in enumerate(families) if f == hold]
        tr = [i for i, f in enumerate(families) if f != hold]
        tr_rows = [rows[i] for i in tr]
        tr_fam = [families[i] for i in tr]
        ytr = y[tr]
        if ytr.sum() == 0 or ytr.sum() == len(ytr):
            oof[te] = ytr.mean()
            continue
        best_c, best_s = 1.0, -1.0                           
        for c in Cs:
            scores = []
            gkf = GroupKFold(n_splits=3)
            for itr, ivl in gkf.split(tr_rows, ytr, groups=tr_fam):
                Xtr, vect = build_matrix([tr_rows[i] for i in itr])
                m = LogisticRegression(C=c, solver="liblinear", class_weight="balanced", random_state=seed)
                m.fit(Xtr, ytr[itr])
                Xvl, _ = build_matrix([tr_rows[i] for i in ivl], vect)
                p = m.predict_proba(Xvl)[:, 1]
                yv = ytr[ivl]
                if 0 < yv.sum() < len(yv):
                    scores.append(average_precision_score(yv, p))
            s = float(np.mean(scores)) if scores else None
            if s is not None and s > best_s:
                best_s, best_c = s, c
        Xtr, vect = build_matrix(tr_rows)
        m = LogisticRegression(C=best_c, solver="liblinear", class_weight="balanced", random_state=seed)
        m.fit(Xtr, ytr)
        Xte, _ = build_matrix([rows[i] for i in te], vect)
        oof[te] = m.predict_proba(Xte)[:, 1]
    return oof


def metrics_at_matched_recall(y, p_ref, p_new):
    """在 p_ref（R1 二分）的召回点上比 precision/F1。"""
    ref_bin = p_ref > 0
    rec_ref = ref_bin[y == 1].mean() if (y == 1).any() else 0.0
    prec, rec, thr = precision_recall_curve(y, p_new)
    ok = np.where(rec >= rec_ref - 1e-9)[0]
    i = ok[np.argmax(prec[ok])] if len(ok) else 0
    return {
        "recall_point": float(rec_ref),
        "R1_precision": float((ref_bin & (y == 1)).sum() / max(ref_bin.sum(), 1)),
        "LTR_precision_at_R1_recall": float(prec[i]),
        "LTR_recall_at_point": float(rec[i]),
    }


                                        
def selftest():
                                             
    syn = [
        ("若有更新就搜来入库", "f1", 1), ("如果有新数据就搜来入库", "f2", 1),
        ("若有新增就同步入库", "f3", 1), ("帮我列一下所有文件", "f4", 0),
        ("今天天气怎么样", "f5", 0), ("给我讲讲这个概念", "f6", 0),
    ] * 3
    rows = [{"utterance": u, "write_avail": ["curate.search_online"],
             "has_bysource": False, "has_check": True} for u, _, _ in syn]
    y = np.array([t for _, _, t in syn])
    fams = [f for _, f, _ in syn]
                                     
    p = lofocv_predict(rows, y, fams)
    ap = average_precision_score(y, p)
    assert ap >= 0.99, f"s1 失败：合成完美可分集 AUC-PR={ap:.3f}"
                
    assert r1_score("若有更新就搜") > 0 and r1_score("有没有新数据") > 0
    assert r1_score("库里多少条") == 0
                   
    for hold in set(fams):
        assert not ({hold} & {f for i, f in enumerate(fams) if f != hold} - {hold})
                                       
    ap_inv = average_precision_score(1 - y, p)
    assert ap_inv <= 0.55, f"s4 失败：反转标签 AUC-PR={ap_inv:.3f} 未塌缩到随机水平"
    print(f"selftest OK（s1 合成 AUC-PR={ap:.3f}，s4 反转={ap_inv:.3f}）")


                           
def load_rows_labels():
    rows = json.load(open(HERE / "e7_candidates_0928.json", encoding="utf-8"))["rows"]
    gold = json.load(open(HERE / "e7_gold_0928.json", encoding="utf-8"))["gold"]
    for r in rows:
        r["label"] = gold[r["id"]]["label"]
        r["borderline"] = bool(gold[r["id"]].get("borderline"))
    return rows


def run_eval(rows, label_set=("B",), tag="B_vs_rest"):
    y = np.array([1.0 if r["label"] in label_set else 0.0 for r in rows])
    fams = [r["family"] for r in rows]
    p_r1 = np.array([r1_score(r["utterance"]) for r in rows])
    p_r1b = np.array([r1_score(r["utterance"]) + 0.5 * r["has_bysource"] for r in rows])
    p_r2 = np.array([float(r["has_check"] and bool(r["write_avail"])) for r in rows])
    p_ltr = lofocv_predict(rows, y, fams)
    out = {"tag": tag, "n": len(rows), "pos": int(y.sum())}
    for name, p in [("R0_all", np.ones(len(rows))), ("R1_regex", p_r1),
                    ("R1b_regex+bysource", p_r1b), ("R2_signature", p_r2),
                    ("LTR", p_ltr)]:
        out[name] = {"AUC_PR": float(average_precision_score(y, p))}
    out["matched_recall"] = metrics_at_matched_recall(y, p_r1, p_ltr)
                            
    fam_sc = defaultdict(list)
    for r, p in zip(rows, p_ltr):
        fam_sc[r["family"]].append(float(p))
    out["family_score_std_mean"] = float(np.mean([np.std(v) for v in fam_sc.values()]))
                                       
    yB = y == 1
    thr_r1 = 0.5
    wait_r1 = p_r1 > thr_r1 - 0.5           
    rec = wait_r1[yB].mean()
    prec, recs, thrs = precision_recall_curve(y, p_ltr)
    ok = np.where(recs >= rec - 1e-9)[0]
                                                                 
    thr_l = thrs[ok[-1]] if len(ok) and ok[-1] < len(thrs) else 0.5
    wait_ltr = p_ltr >= thr_l
    out["decision"] = {
        "recall_point": float(rec),
        "R1_unnecessary_wait": float((wait_r1 & ~yB).sum() / max((~yB).sum(), 1)),
        "LTR_unnecessary_wait": float((wait_ltr & ~yB).sum() / max((~yB).sum(), 1)),
    }
    return out, p_ltr, y


def mine_violations(rows):
    """E7-4：E4/E4b/E6 eager cold 违规任务 → 标签与预测覆盖。只读记录。"""
    utt2id = {r["utterance"]: r["id"] for r in rows}
    hits = []
    for f, cold_only in [("e4_0923/runs.jsonl", True),
                         ("e4b_0923/runs.jsonl", True),
                         ("e6_0924/runs.jsonl", True)]:
        for line in open(HERE / f, encoding="utf-8"):
            if not line.strip():
                continue
            x = json.loads(line)
            if x.get("arm") != "eager":
                continue
            if x.get("world", "cold") != "cold":
                continue
            bad = [d["dim"] for d in x.get("dims", [])
                   if d["dim"] in VIOL_DIMS and not d["ok"]]
            if bad:
                cid = x["case"]
                if cid not in {r["id"] for r in rows}:
                    cid = utt2id.get(x.get("utterance", ""), x["case"])
                hits.append({"batch": f.split("_")[0], "case": cid,
                             "viol_dims": bad, "round": x.get("round")})
    return hits


def main():
    selftest()
    rows = load_rows_labels()
    res = {"selftest": "OK"}
                             
    main_res, p_ltr, y = run_eval(rows, ("B",), "B_vs_rest")
    res["main"] = main_res
                                 
    rows_nb = [r for r in rows if not r["borderline"]]
    res["main_no_borderline"], _, _ = run_eval(rows_nb, ("B",), "B_vs_rest_noBL")
                
    res["any_risk"], _, _ = run_eval(rows, ("A", "B"), "AB_vs_N")
                                      
    res["neg_control_A"], _, _ = run_eval(rows, ("A",), "A_vs_rest")
                 
    hits = mine_violations(rows)
    lab = {r["id"]: r["label"] for r in rows}
    pmap = {r["id"]: float(p) for r, p in zip(rows, p_ltr)}
    for h in hits:
        h["gold"] = lab.get(h["case"], "?")
        h["ltr_oof_score"] = pmap.get(h["case"])
        h["R1_fire"] = bool(h["case"] in pmap and
                            r1_score(next(r["utterance"] for r in rows
                                          if r["id"] == h["case"])) > 0)
    res["e7_4_outcome"] = hits
    (HERE / "e7_results_0928.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "e7_4_outcome"},
                     ensure_ascii=False, indent=1))
    print("E7-4 violation tasks:", json.dumps(hits, ensure_ascii=False))


if __name__ == "__main__":
    main()
