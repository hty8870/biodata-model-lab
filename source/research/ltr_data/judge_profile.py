                       
"""xdata-reuse WP1: 判官画像 + 标签噪声天花板（纯本地统计，零网络零 LLM）。

输入：research/ltr_data/synth_train_xscore.jsonl（三联 C 三判官）
      research/ltr_data/synth_heldout_xscore.jsonl（五判官金标准）
输出：_judge_profile.json（全部统计表）+ _judge_disagreement_cases.jsonl（分歧案例）
      并打印人类可读摘要。
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parent


def load(name):
    return [json.loads(l) for l in (DATA / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n))
    vy = sum((ry[i] - my) ** 2 for i in range(n))
    if vx == 0 or vy == 0:
        return None
    return round(cov / math.sqrt(vx * vy), 4)


def quad_kappa(a, b, k=4):
    n = len(a)
    if n == 0:
        return None
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1
    ha = [sum(row) for row in obs]
    hb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / ((k - 1) ** 2)
            num += w * obs[i][j]
            den += w * ha[i] * hb[j] / n
    if den == 0:
        return None
    return round(1 - num / den, 4)


def agree(a, b):
    n = len(a)
    if n == 0:
        return {}
    exact = sum(1 for x, y in zip(a, b) if x == y) / n
    pm1 = sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / n
    mae = sum(abs(x - y) for x, y in zip(a, b)) / n
    return {"n": n, "exact": round(exact, 4), "pm1": round(pm1, 4), "mae": round(mae, 4)}


def aligned(rows, m1, m2):
    a, b = [], []
    for r in rows:
        for c in r["candidates"]:
            xs = c.get("xscore") or {}
            s1, s2 = xs.get(m1), xs.get(m2)
            if s1 is None or s2 is None:
                continue
            a.append(s1)
            b.append(s2)
    return a, b


def profile_split(name, rows):
    judges = sorted({m for r in rows for c in r["candidates"] for m in (c.get("xscore") or {})})
    out = {"n_queries": len(rows), "judges": judges}
    n_pairs = 0
    dist = {m: collections.Counter() for m in judges}
    fused_counter = collections.Counter()
    single_counter = collections.Counter()
                        
    sig = {m: {"score": [], "pos": [], "lex": [], "vec": [], "grade": []} for m in judges}
    fused_vs_single = {"fused": [], "single": []}
    for r in rows:
        for c in r["candidates"]:
            n_pairs += 1
            xs = c.get("xscore") or {}
            fused = c.get("llm_semantic")
            single = c.get("llm_semantic_single")
            if fused is not None:
                fused_counter[fused] += 1
            if single is not None:
                single_counter[single] += 1
            if fused is not None and single is not None:
                fused_vs_single["fused"].append(fused)
                fused_vs_single["single"].append(single)
            for m in judges:
                s = xs.get(m)
                if s is None:
                    dist[m]["None"] += 1
                    continue
                dist[m][s] += 1
                sig[m]["score"].append(s)
                sig[m]["pos"].append(c.get("pos") or 0)
                sig[m]["lex"].append(c.get("lex_score") or 0.0)
                sig[m]["vec"].append(c.get("vec_score") or 0.0)
                sig[m]["grade"].append(c.get("grade") or 0)
    out["n_pairs"] = n_pairs
    out["judge_dist"] = {m: {str(k): v for k, v in sorted(dist[m].items(), key=lambda x: str(x[0]))} for m in judges}
    out["judge_mean"] = {m: round(sum(k * v for k, v in dist[m].items() if isinstance(k, int)) /
                                  max(1, sum(v for k, v in dist[m].items() if isinstance(k, int))), 4) for m in judges}
    out["fused_dist"] = {str(k): v for k, v in sorted(fused_counter.items())}
    out["single_dist"] = {str(k): v for k, v in sorted(single_counter.items())}
           
    pw = {}
    for i, m1 in enumerate(judges):
        for m2 in judges[i + 1:]:
            a, b = aligned(rows, m1, m2)
            pw[f"{m1} vs {m2}"] = {**agree(a, b), "quad_kappa": quad_kappa(a, b), "spearman": spearman(a, b)}
    out["pairwise"] = pw
                            
    vs_fused = {}
    for m in judges:
        a, b = [], []
        for r in rows:
            for c in r["candidates"]:
                s = (c.get("xscore") or {}).get(m)
                f = c.get("llm_semantic")
                if s is None or f is None:
                    continue
                a.append(s)
                b.append(f)
        vs_fused[m] = {**agree(a, b), "quad_kappa": quad_kappa(a, b)}
    out["judge_vs_fused"] = vs_fused
    out["fused_vs_old_single"] = {**agree(fused_vs_single["fused"], fused_vs_single["single"]),
                                  "quad_kappa": quad_kappa(fused_vs_single["fused"], fused_vs_single["single"])}
                              
    bias = {}
    for m in judges:
        s = sig[m]
        bias[m] = {"vs_pos": spearman(s["score"], s["pos"]),
                   "vs_lex": spearman(s["score"], s["lex"]),
                   "vs_vec": spearman(s["score"], s["vec"]),
                   "vs_grade": spearman(s["score"], s["grade"])}
    out["judge_bias_spearman"] = bias
    return out


def noise_ceiling(rows):
    """留一判官法：用其余判官的中位数（四舍五入）预测被留判官。
    解释：任何模型对「单一判官」的一致性上限 ≈ 判官之间的互预测能力；
    对「融合共识」的一致性上限 ≈ 共识对单判官的预测能力。"""
    loo = {"exact": 0, "pm1": 0, "ae": 0.0, "n": 0}
    cons = {"exact": 0, "pm1": 0, "ae": 0.0, "n": 0}
    for r in rows:
        for c in r["candidates"]:
            xs = {m: s for m, s in (c.get("xscore") or {}).items() if s is not None}
            if len(xs) < 2:
                continue
            fused = c.get("llm_semantic")
            for m, s in xs.items():
                others = [v for k, v in xs.items() if k != m]
                if not others:
                    continue
                pred = round(sorted(others)[len(others) // 2] if len(others) % 2 else
                             (sorted(others)[len(others) // 2 - 1] + sorted(others)[len(others) // 2]) / 2)
                loo["n"] += 1
                loo["exact"] += int(pred == s)
                loo["pm1"] += int(abs(pred - s) <= 1)
                loo["ae"] += abs(pred - s)
                if fused is not None:
                    cons["n"] += 1
                    cons["exact"] += int(fused == s)
                    cons["pm1"] += int(abs(fused - s) <= 1)
                    cons["ae"] += abs(fused - s)
    for d in (loo, cons):
        if d["n"]:
            d["exact"] = round(d["exact"] / d["n"], 4)
            d["pm1"] = round(d["pm1"] / d["n"], 4)
            d["mae"] = round(d["ae"] / d["n"], 4)
        del d["ae"]
    return {"leave_one_judge_out_median": loo, "fused_predicts_single_judge": cons}


def kendall_w(rows):
    """逐 query 的 Kendall's W（判官对候选排序的协和度，平局用平均秩、未做平局校正——近似值），按 query 取均值。"""
    ws = []
    for r in rows:
        cands = r["candidates"]
        n = len(cands)
        if n < 3:
            continue
        judges = sorted({m for c in cands for m in (c.get("xscore") or {}) if (c.get("xscore") or {}).get(m) is not None})
        m = len(judges)
        if m < 3:
            continue
        rank_sum = [0.0] * n
        for jm in judges:
            vals = [(c.get("xscore") or {}).get(jm) for c in cands]
            order = sorted(range(n), key=lambda i: (vals[i] is None, vals[i]))
            rk = [0.0] * n
            i = 0
            while i < n:
                j = i
                while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                    j += 1
                avg = (i + j) / 2.0 + 1
                for k in range(i, j + 1):
                    rk[order[k]] = avg
                i = j + 1
            for i in range(n):
                rank_sum[i] += rk[i]
        mean_rs = sum(rank_sum) / n
        S = sum((rs - mean_rs) ** 2 for rs in rank_sum)
        denom = m * m * (n ** 3 - n)
        if denom:
            ws.append(12 * S / denom)
    return {"n_queries": len(ws), "mean_W": round(sum(ws) / len(ws), 4) if ws else None}


def dump_disagreement(rows, path, cap_spread1=200):
    n2 = n1 = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            for c in r["candidates"]:
                sp = c.get("xscore_spread")
                keep = sp is not None and sp >= 2
                if not keep and sp == 1 and c.get("llm_semantic") != c.get("llm_semantic_single") and n1 < cap_spread1:
                    keep = True
                if not keep:
                    continue
                fh.write(json.dumps({
                    "query_id": r.get("query_id"), "query": r.get("query"), "uid": c.get("uid"),
                    "xscore": c.get("xscore"), "spread": sp, "fused": c.get("llm_semantic"),
                    "old_single": c.get("llm_semantic_single"), "reason": c.get("llm_reason"),
                    "pos": c.get("pos"), "lex_score": c.get("lex_score"), "vec_score": c.get("vec_score"),
                }, ensure_ascii=False) + chr(10))
                if sp is not None and sp >= 2:
                    n2 += 1
                else:
                    n1 += 1
    return {"spread_ge2": n2, "spread1_fused_ne_single_sampled": n1}


def main():
    report = {}
    for name in ["train", "heldout"]:
        rows = load(f"synth_{name}_xscore.jsonl")
        rep = profile_split(name, rows)
        rep["noise_ceiling"] = noise_ceiling(rows)
        rep["kendall_W"] = kendall_w(rows)
        report[name] = rep
    rows_all = load("synth_train_xscore.jsonl") + load("synth_heldout_xscore.jsonl")
    report["disagreement_dump"] = dump_disagreement(rows_all, DATA / "_judge_disagreement_cases.jsonl")
    out = DATA / "_judge_profile.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, rep in report.items():
        if not isinstance(rep, dict) or "judges" not in rep:
            continue
        print(f"== {name}: {rep['n_queries']} queries / {rep['n_pairs']} pairs / judges={rep['judges']}")
        print("  mean:", json.dumps(rep["judge_mean"]))
        print("  fused_vs_old_single:", json.dumps(rep["fused_vs_old_single"]))
        print("  noise_ceiling:", json.dumps(rep["noise_ceiling"]))
        print("  kendall_W:", json.dumps(rep["kendall_W"]))
        print("  vs_fused:", json.dumps({m: {"exact": v.get("exact"), "pm1": v.get("pm1")} for m, v in rep["judge_vs_fused"].items()}))
        print("  bias(pos/lex/vec/grade):", json.dumps(rep["judge_bias_spearman"]))
    print("disagreement_dump:", json.dumps(report["disagreement_dump"]))
    print("written:", out)


if __name__ == "__main__":
    main()
