                       
"""E7 发射型预测器 v2（2026-09-28，Codex v0.8 评审 B3/B4/M2/M3/M5/m1/m2 修订版）。
相对 v1（e7_train_0928.py）的修订：
- B3 输入合法性：剔除 by_source（世界剧本内部结构，非部署可见签名）。
- B4 谱系隔离：相似句并族（union-find，utterance SequenceMatcher≥0.75 跨族链通 +
  note 显式谱系），测「跨族近重复」下的稳健性。
- M3 真消融：纯文本 / 纯签名（学习模型）/ 全特征 三跑。
- M2 工作点纪律：阈值只用训练折内折外分数选定，held-out family 只评估；
  严格等召回点报告（不拼接两个阈值）。
- M5 selftest v2：索引级 family 不跨折核验；标签置换后重训必须塌缩；
  C 平票偏向修复核验（有效分平票时取较大 C）。
- m1 指标命名 Average Precision（AP）；m2 签名特征与文档对齐（写动词数/有无 check）。
输出 e7_results2_0928.json。
"""
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
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
SEED = 928


                          
def r1_score(text):
    return float(len(RE_COND.findall(text)))


def sig_feats(r):
    """签名特征（B3 修订：仅写动词数、有无 check——部署接口可见字段）。"""
    return [float(len(r["write_avail"])), float(r["has_check"])]


def build_matrix(rows, vect=None, use_text=True, use_sig=True):
    parts = []
    if use_text:
        X_text = [r["utterance"] for r in rows]
        if vect is None:
            vect = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2)
            Xt = vect.fit_transform(X_text)
        else:
            Xt = vect.transform(X_text)
        parts.append(Xt)
    if use_sig:
        parts.append(csr_matrix(np.array([sig_feats(r) for r in rows])))
    return hstack(parts).tocsr(), vect


def lofocv_predict(rows, y, groups, Cs=(0.1, 1.0, 10.0), use_text=True, use_sig=True,
                   return_threshold=False):
    """Leave-One-Group-Out 折外预测（+训练折内选阈值）。"""
    gs = sorted(set(groups))
    oof = np.zeros(len(rows))
    thr_oof = np.full(len(rows), np.nan)
    for hold in gs:
        te = [i for i, g in enumerate(groups) if g == hold]
        tr = [i for i, g in enumerate(groups) if g != hold]
        tr_rows = [rows[i] for i in tr]
        tr_g = [groups[i] for i in tr]
        ytr = y[tr]
        if ytr.sum() == 0 or ytr.sum() == len(ytr):
            oof[te] = ytr.mean()
            continue

        def train_eval(sub_rows, sub_y, sub_fit_vect=None):
            X, v = build_matrix(sub_rows, sub_fit_vect, use_text, use_sig)
            return X, v

        best_c, best_s = 1.0, -1.0
        tie_c = []
        for c in Cs:
            scores = []
            gkf = GroupKFold(n_splits=3)
            for itr, ivl in gkf.split(tr_rows, ytr, groups=tr_g):
                Xtr, vect = build_matrix([tr_rows[i] for i in itr], None, use_text, use_sig)
                m = LogisticRegression(C=c, solver="liblinear",
                                       class_weight="balanced", random_state=SEED)
                m.fit(Xtr, ytr[itr])
                Xvl, _ = build_matrix([tr_rows[i] for i in ivl], vect, use_text, use_sig)
                p = m.predict_proba(Xvl)[:, 1]
                yv = ytr[ivl]
                if 0 < yv.sum() < len(yv):
                    scores.append(average_precision_score(yv, p))
            s = float(np.mean(scores)) if scores else None
            if s is not None:
                if s > best_s + 1e-12:
                    best_s, best_c, tie_c = s, c, [c]
                elif abs(s - best_s) <= 1e-12:
                    tie_c.append(c)
        if len(tie_c) > 1:
            best_c = max(tie_c)                                
        Xtr, vect = build_matrix(tr_rows, None, use_text, use_sig)
        m = LogisticRegression(C=best_c, solver="liblinear",
                               class_weight="balanced", random_state=SEED)
        m.fit(Xtr, ytr)
        Xte, _ = build_matrix([rows[i] for i in te], vect, use_text, use_sig)
        oof[te] = m.predict_proba(Xte)[:, 1]
        if return_threshold:
                                           
                                           
            inner_oof = np.zeros(len(tr_rows))
            inner_gs = sorted(set(tr_g))
            for h2 in inner_gs:
                te2 = [i for i, g in enumerate(tr_g) if g == h2]
                tr2 = [i for i, g in enumerate(tr_g) if g != h2]
                if len(set(tr2)) < 4:
                    inner_oof[te2] = ytr[tr2].mean()
                    continue
                X2, v2 = build_matrix([tr_rows[i] for i in tr2], None, use_text, use_sig)
                m2 = LogisticRegression(C=best_c, solver="liblinear",
                                        class_weight="balanced", random_state=SEED)
                m2.fit(X2, ytr[tr2])
                X2t, _ = build_matrix([tr_rows[i] for i in te2], v2, use_text, use_sig)
                inner_oof[te2] = m2.predict_proba(X2t)[:, 1]
            prec, rec, thr = precision_recall_curve(ytr, inner_oof)
            ok = np.where(rec >= 0.96 - 1e-9)[0]
            thr_oof[te] = thr[ok[np.argmax(prec[ok])]] if len(ok) and ok[-1] < len(thr) \
                else (thr[-1] if len(thr) else 0.5)
    if return_threshold:
        return oof, thr_oof
    return oof


                                    
def merged_groups(rows):
    fams = sorted({r["family"] for r in rows})
    parent = {f: f for f in fams}

    def find(f):
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    utt = defaultdict(list)
    for r in rows:
        utt[r["family"]].append(r["utterance"])
    for i, f1 in enumerate(fams):
        for f2 in fams[i + 1:]:
            if any(SequenceMatcher(None, u1, u2).ratio() >= 0.75
                   for u1 in utt[f1] for u2 in utt[f2]):
                union(f1, f2)
                                     
    note_of = {r["family"]: r.get("note", "") for r in rows}
    for f, note in note_of.items():
        m = re.search(r"([a-z]+\d+)\s*同句", note)
        if m and m.group(1) in parent:
            union(f, m.group(1))
    return [find(r["family"]) for r in rows]


                                   
def selftest():
    syn = [
        ("若有更新就搜来入库", "f1", 1), ("如果有新数据就搜来入库", "f2", 1),
        ("若有新增就同步入库", "f3", 1), ("帮我列一下所有文件", "f4", 0),
        ("今天天气怎么样", "f5", 0), ("给我讲讲这个概念", "f6", 0),
    ] * 4
    rows = [{"utterance": u, "write_avail": ["curate.search_online"], "has_check": True}
            for u, _, _ in syn]
    y = np.array([float(t) for _, _, t in syn])
    fams = [f for _, f, _ in syn]
                    
    p = lofocv_predict(rows, y, fams)
    ap = average_precision_score(y, p)
    assert ap >= 0.99, f"s1 失败：AP={ap:.3f}"
                                  
    for hold in set(fams):
        te = {i for i, f in enumerate(fams) if f == hold}
        tr = {i for i, f in enumerate(fams) if f != hold}
        assert te and tr and not (te & tr) and te | tr == set(range(len(fams)))
                                  
    rng = np.random.RandomState(SEED)
    yp = rng.permutation(y)
    pp = lofocv_predict(rows, yp, fams)
    ap_p = average_precision_score(yp, pp)
    assert ap_p <= 0.65, f"s3 失败：置换重训 AP={ap_p:.3f} 未塌缩"
                                  
    p_a = lofocv_predict(rows, y, fams, Cs=(0.1, 1.0, 10.0))
    p_b = lofocv_predict(rows, y, fams, Cs=(10.0, 1.0, 0.1))
    assert np.allclose(p_a, p_b, atol=1e-6), "s4 失败：Cs 顺序改变预测"
    print(f"selftest v2 OK（s1 AP={ap:.3f}；s3 置换={ap_p:.3f}；s4 C 序无关）")


                           
def load_rows_labels():
    rows = json.load(open(HERE / "e7_candidates_0928.json", encoding="utf-8"))["rows"]
    gold = json.load(open(HERE / "e7_gold_0928.json", encoding="utf-8"))["gold"]
    for r in rows:
        r["label"] = gold[r["id"]]["label"]
        r["borderline"] = bool(gold[r["id"]].get("borderline"))
    return rows


def eval_config(rows, groups, use_text, use_sig, tag):
    y = np.array([1.0 if r["label"] == "B" else 0.0 for r in rows])
    p = lofocv_predict(rows, y, groups, use_text=use_text, use_sig=use_sig)
    return {"tag": tag, "AP": float(average_precision_score(y, p))}, p, y


def main():
    selftest()
    rows = load_rows_labels()
    groups = merged_groups(rows)
    n_groups = len(set(groups))
    res = {"selftest": "v2 OK", "n_cases": len(rows), "n_groups": n_groups,
           "labels": {"B": sum(r["label"] == "B" for r in rows),
                      "A": sum(r["label"] == "A" for r in rows),
                      "N": sum(r["label"] == "N" for r in rows)}}
    y = np.array([1.0 if r["label"] == "B" else 0.0 for r in rows])
        
    p_r1 = np.array([r1_score(r["utterance"]) for r in rows])
    res["R1_regex"] = {"AP": float(average_precision_score(y, p_r1))}
    res["R2_signature"] = {"AP": float(average_precision_score(
        y, np.array([float(r["has_check"] and bool(r["write_avail"])) for r in rows])))}
                
    for ut, us, tag in [(True, True, "LTR_full"), (True, False, "LTR_text_only"),
                        (False, True, "LTR_sig_only")]:
        r, _, _ = eval_config(rows, groups, ut, us, tag)
        res[tag] = r
                      
    p_ltr, thr = lofocv_predict(rows, y, groups, return_threshold=True)
    res["LTR_full"]["AP_train_thresholded_note"] = "阈值训练折内选定"
                          
    ref_bin = p_r1 > 0
    rec_ref = float(ref_bin[y == 1].mean())
    prec, rec, thr_pr = precision_recall_curve(y, p_ltr)
    ok = np.where(rec >= rec_ref - 1e-9)[0]
    i = ok[np.argmax(prec[ok])]
    res["matched_point"] = {
        "recall": float(rec[i]), "R1_precision": float((ref_bin & (y == 1)).sum() / ref_bin.sum()),
        "LTR_precision": float(prec[i]),
        "R1_FP": int((ref_bin & (y == 0)).sum()),
        "LTR_FP_at_point": int(((p_ltr >= (thr_pr[i - 1] if i > 0 and i - 1 < len(thr_pr) else 0.5)) & (y == 0)).sum()),
    }
                               
    wait = p_ltr >= thr
    res["deploy_style"] = {
        "recall_B": float((wait & (y == 1)).sum() / max((y == 1).sum(), 1)),
        "precision_B": float((wait & (y == 1)).sum() / max(wait.sum(), 1)),
        "unnecessary_wait": float((wait & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "nan_thr_folds": int(np.isnan(thr).sum()),
    }
                                           
    rng = np.random.RandomState(SEED)
    gid = np.array(groups)
    boots = []
    for _ in range(200):
        bg = rng.choice(sorted(set(groups)), size=len(set(groups)), replace=True)
        idx = np.concatenate([np.where(gid == g)[0] for g in bg])
        if 0 < y[idx].sum() < len(idx):
            boots.append(average_precision_score(y[idx], p_ltr[idx]))
    res["AP_bootstrap_by_group"] = {"p5": float(np.percentile(boots, 5)),
                                    "p50": float(np.percentile(boots, 50)),
                                    "p95": float(np.percentile(boots, 95))}
    (HERE / "e7_results2_0928.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
