                       
"""xdata-reuse WP4: embedding / cross-encoder 模型离线基准（纯本地，零网络零 API）。

金标：synth_heldout_xscore.jsonl 五判官融合 llm_semantic（0-3）。
排序宇宙：每个 query 的已收割候选集（~9 条/query）——衡量「候选集内排序质量」，
不衡量全库召回（这一点写进输出 caveat）。
现役基线：(a) 文件内 vec_score（生产 MiniLM 全库索引的余弦，直接排序复算）；
          (b) 同模型本地重编码（校验 harness 与生产一致性）。
模型清单全部来自本机 HF 缓存（HF_HUB_OFFLINE=1，绝不下载）。
输出：_embedding_bench.json + 打印表格。向量缓存 _emb_cache/（gitignored）。
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sys
import time
from pathlib import Path

                                           
                                               
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DATA = Path(__file__).resolve().parent
CACHE = DATA / "_emb_cache"
CACHE.mkdir(exist_ok=True)

EMB_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-m3",
    "Qwen/Qwen3-Embedding-0.6B",
]
CE_MODELS = ["BAAI/bge-reranker-v2-m3", "Qwen/Qwen3-Reranker-0.6B"]


def ndcg(order, labels, k):
    dcg = sum(labels.get(u, 0) / math.log2(i + 2) for i, u in enumerate(order[:k]))
    ideal = sorted(labels.values(), reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    return (dcg / idcg) if idcg > 0 else None


def load_gold():
    rows = [json.loads(l) for l in (DATA / "synth_heldout_xscore.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    gold = []
    for r in rows:
        labels = {c["uid"]: c["llm_semantic"] for c in r["candidates"]
                  if c.get("llm_semantic") is not None}
        vecs = {c["uid"]: c.get("vec_score") for c in r["candidates"]}
        gold.append({"query_id": r["query_id"], "query": r["query"],
                     "labels": labels, "vec_score": vecs})
    return gold


def load_doc_texts(need_uids):
    from dataset_recommender.app.runtime_paths import get_app_paths
    from dataset_recommender.corpus.corpus import load_full_corpus
    from dataset_recommender.retrieval.retriever import RetrievedCandidate
    from dataset_recommender.retrieval.vector_index import MAX_TEXT_CHARS
    from dataset_recommender.retrieval.vector_recall import _candidate_text
    root = Path(get_app_paths().data_root)
    records = load_full_corpus(root / "database" / "base", root)
    texts = {}
    for rec in records:
        raw = getattr(rec, "raw", None)
        uid = str(raw.get("dataset_uid") or "") if isinstance(raw, dict) else ""
        if uid and uid in need_uids:
            t = _candidate_text(RetrievedCandidate(record=rec, score=0.0))
            texts[uid] = t[:MAX_TEXT_CHARS] if len(t) > MAX_TEXT_CHARS else t
    return texts


def _cache_name(model):
    return CACHE / (model.replace("/", "__") + ".json.gz")


def embed_with(model_name, queries, doc_texts):
    """返回 (query_vecs: dict qid->vec, doc_vecs: dict uid->vec)；带 gzip JSON 缓存。"""
    cp = _cache_name(model_name)
    need_q = set(queries)
    need_d = set(doc_texts)
    cached_q, cached_d = {}, {}
    if cp.exists():
        try:
            with gzip.open(cp, "rt", encoding="utf-8") as fh:
                blob = json.load(fh)
            cached_q, cached_d = blob.get("queries", {}), blob.get("docs", {})
        except Exception:
            cached_q, cached_d = {}, {}
    miss_q = {k: v for k, v in queries.items() if k not in cached_q}
    miss_d = {k: v for k, v in doc_texts.items() if k not in cached_d}
    if miss_q or miss_d:
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        model = SentenceTransformer(_resolve(model_name))
        if miss_d:
            keys = list(miss_d)
            vecs = model.encode([miss_d[k] for k in keys], batch_size=16,
                                normalize_embeddings=True, show_progress_bar=False)
            for k, v in zip(keys, vecs):
                cached_d[k] = [round(float(x), 6) for x in v]
        if miss_q:
            keys = list(miss_q)
            kwargs = {}
            if "Qwen3-Embedding" in model_name:
                try:
                    vecs = model.encode([miss_q[k] for k in keys], batch_size=16,
                                        normalize_embeddings=True, show_progress_bar=False,
                                        prompt_name="query")
                except TypeError:
                    vecs = model.encode([miss_q[k] for k in keys], batch_size=16,
                                        normalize_embeddings=True, show_progress_bar=False)
                cached_q.update({k: [round(float(x), 6) for x in v] for k, v in zip(keys, vecs)})
            else:
                vecs = model.encode([miss_q[k] for k in keys], batch_size=16,
                                    normalize_embeddings=True, show_progress_bar=False, **kwargs)
                cached_q.update({k: [round(float(x), 6) for x in v] for k, v in zip(keys, vecs)})
        with gzip.open(cp, "wt", encoding="utf-8") as fh:
            json.dump({"queries": cached_q, "docs": cached_d}, fh)
        print(f"  encoded {len(miss_q)}q+{len(miss_d)}d in {time.time()-t0:.0f}s -> {cp.name}", flush=True)
    return cached_q, cached_d


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def eval_embedding_model(model_name, gold, doc_texts):
    queries = {g["query_id"]: g["query"] for g in gold}
    qv, dv = embed_with(model_name, queries, doc_texts)
    n3, n10, p3 = [], [], []
    for g in gold:
        order = sorted(g["labels"], key=lambda u: -dot(qv[g["query_id"]], dv[u]))
        v3, v10 = ndcg(order, g["labels"], 3), ndcg(order, g["labels"], 10)
        if v3 is not None:
            n3.append(v3)
        if v10 is not None:
            n10.append(v10)
        p3.append(sum(1 for u in order[:3] if g["labels"].get(u, 0) >= 2) / 3)
    return {"ndcg3": round(sum(n3) / len(n3), 4), "ndcg10": round(sum(n10) / len(n10), 4),
            "p3_rel2": round(sum(p3) / len(p3), 4), "n_queries": len(n3)}


def eval_production_vecscore(gold):
    """现役生产基线：直接用文件里的 vec_score（生产 MiniLM 全库索引余弦）排序复算。"""
    n3, n10 = [], []
    for g in gold:
        order = sorted(g["labels"], key=lambda u: -(g["vec_score"].get(u) or -1))
        v3, v10 = ndcg(order, g["labels"], 3), ndcg(order, g["labels"], 10)
        if v3 is not None:
            n3.append(v3)
        if v10 is not None:
            n10.append(v10)
    return {"ndcg3": round(sum(n3) / len(n3), 4), "ndcg10": round(sum(n10) / len(n10), 4)}


def _resolve(model_name):
    """现役 MiniLM 钉死用仓内生产同一份模型文件（vector_recall.default_model_dir 单一真源）。"""
    if "paraphrase-multilingual-MiniLM" in model_name:
        from dataset_recommender.retrieval.vector_recall import default_model_dir
        p = default_model_dir()
        if p.exists():
            return str(p)
    return model_name


def _resolve_ce(model_name):
    if "bge-reranker-v2-m3" in model_name:
        from dataset_recommender.retrieval.vector_recall import default_cross_encoder_dir
        p = default_cross_encoder_dir()
        if p.exists():
            return str(p)
    return model_name


def eval_cross_encoder(model_name, gold, doc_texts):
    from sentence_transformers import CrossEncoder
    t0 = time.time()
    model = CrossEncoder(_resolve_ce(model_name))
    pairs, idx = [], []
    for gi, g in enumerate(gold):
        for uid in g["labels"]:
            pairs.append((g["query"], doc_texts[uid]))
            idx.append((gi, uid))
    scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
    print(f"  CE scored {len(pairs)} pairs in {time.time()-t0:.0f}s", flush=True)
    per_q = {}
    for (gi, uid), s in zip(idx, scores):
        per_q.setdefault(gi, {})[uid] = float(s)
    n3, n10, p3 = [], [], []
    for gi, g in enumerate(gold):
        order = sorted(g["labels"], key=lambda u: -per_q[gi][u])
        v3, v10 = ndcg(order, g["labels"], 3), ndcg(order, g["labels"], 10)
        if v3 is not None:
            n3.append(v3)
        if v10 is not None:
            n10.append(v10)
        p3.append(sum(1 for u in order[:3] if g["labels"].get(u, 0) >= 2) / 3)
    return {"ndcg3": round(sum(n3) / len(n3), 4), "ndcg10": round(sum(n10) / len(n10), 4),
            "p3_rel2": round(sum(p3) / len(p3), 4), "n_queries": len(n3)}


def main():
    gold = load_gold()
    need_uids = {u for g in gold for u in g["labels"]}
    print(f"gold: {len(gold)} queries, {sum(len(g['labels']) for g in gold)} pairs, {len(need_uids)} unique uids", flush=True)
    doc_texts = load_doc_texts(need_uids)
    missing = need_uids - set(doc_texts)
    if missing:
        print(f"WARN: {len(missing)} uids missing doc text, dropped from eval", flush=True)
        gold = [{"query_id": g["query_id"], "query": g["query"],
                 "labels": {u: s for u, s in g["labels"].items() if u in doc_texts},
                 "vec_score": g["vec_score"]} for g in gold]
        gold = [g for g in gold if len(g["labels"]) >= 2]
        print(f"gold after drop: {len(gold)} queries", flush=True)
    report = {"caveat": "排序宇宙=每 query 已收割候选集（~9 条），衡量候选集内排序质量，不衡量全库召回。",
              "n_queries": len(gold), "models": {}}
    report["models"]["production_vec_score_from_file"] = eval_production_vecscore(gold)
    print("production_vec_score_from_file:", report["models"]["production_vec_score_from_file"], flush=True)
    for m in EMB_MODELS:
        print(f"[emb] {m}", flush=True)
        try:
            report["models"][f"emb:{m}"] = eval_embedding_model(m, gold, doc_texts)
            print("  ->", report["models"][f"emb:{m}"], flush=True)
        except Exception as exc:
            report["models"][f"emb:{m}"] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            print("  ERROR:", report["models"][f"emb:{m}"]["error"], flush=True)
    for m in CE_MODELS:
        print(f"[ce] {m}", flush=True)
        try:
            report["models"][f"ce:{m}"] = eval_cross_encoder(m, gold, doc_texts)
            print("  ->", report["models"][f"ce:{m}"], flush=True)
        except Exception as exc:
            report["models"][f"ce:{m}"] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            print("  ERROR:", report["models"][f"ce:{m}"]["error"], flush=True)
    out = DATA / "_embedding_bench.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written:", out)


if __name__ == "__main__":
    main()
