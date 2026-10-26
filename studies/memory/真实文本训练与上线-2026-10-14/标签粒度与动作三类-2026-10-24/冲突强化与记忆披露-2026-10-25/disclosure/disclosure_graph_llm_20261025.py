                      
"""第 4 线真实图构建器接口（v2）：按 split 批量请求 typed 边。

边界：
- 预定通道 = 本机 Ollama `batiai/qwen3.8-27b:iq3`；**本阶段不调用**。
- 输入只有记忆正文/标题/links/来源元数据；不含任务、required IDs、oracle 边。
- 候选 pair 不做数值/实体过滤：同项目同标题的全部 pair 都进候选，因此排序型更新
  （无数值）与 human/mouse 冲突仍有被检出的机会。
- 输出类型限定 context/exception/conflict/equivalent/sibling/none；oracle 只用于评分诊断。
- 旧版无绑定回执的 ingest/cache 只供诊断；真实发布使用 pilot_v4 独立语义闸。
- 规则构造器保留为 `rules_fallback`（dry-run 用），标签必须明确；不得用 gold 补边。

命令：
  python disclosure_graph_llm_20261025.py selftest
  python disclosure_graph_llm_20261025.py prepare-graph --split dev --batch-size 8
  python disclosure_graph_llm_20261025.py ingest-graph --split dev --responses out/graph/dev/responses
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from disclosure_schema_20261025 import (              
    DATA, OUT, json_write, jsonl_read, load_groups, sha256,
)


def publicize_items(groups, id_map):
    out = []
    for g in groups:
        for i, a in enumerate(g["atoms"]):
            out.append({**a, "id": id_map["items"][a["id"]], "links": []})
    return out


def load_v3_groups():
    id_map = json.loads((DATA / "id_map.json").read_text(encoding="utf-8"))
    out = []
    for g in load_groups():
        out.append({**g, "group_id": id_map["groups"][g["group_id"]],
                    "family_id": g["family_id"],
                    "split": "dev" if g["family_id"] in ("F01", "F02", "F03", "F04") else "final",
                    "atoms": [{**a, "id": id_map["items"][a["id"]], "links": []}
                              for a in g["atoms"]],
                    "atomic_fact_ids": {id_map["items"][k]: v
                                        for k, v in g["atomic_fact_ids"].items()}})
    return out, id_map


def load_v3_items():
    groups, id_map = load_v3_groups()
    return [a for g in groups for a in g["atoms"]], id_map
import disclosure_compiler_20261025 as compiler_line              

GRAPH_CHANNEL = "ollama"
GRAPH_MODEL = "batiai/qwen3.8-27b:iq3"
GRAPH_ROOT = OUT / "graph"
BATCH_SIZE = 8
EDGE_TYPES = ("context", "exception", "conflict", "equivalent", "sibling", "none")
                                  
SYMMETRIC = {"conflict", "equivalent", "sibling"}
MAX_PAIRS_PER_SLOT = 8
MAX_CROSS_PAIRS_PER_ITEM = 4

GRAPH_RELATION_CONTRACT = (
    "固定六类：context=条件/语境限定或追加义务，默认义务仍保留；"
    "exception=条件分支替换默认动作，或明确说默认不适用；两类均从条件/例外指向默认。"
    "因此条件句本身不自动等于context或exception，要看默认动作是否被替换；"
    "同时/额外增加动作而未取消默认是context。不能给默认擅自补‘仅’或其他排他性。"
    "conflict=同对象同时段不可同时满足，或有明确生效时间的更新；二者均属conflict，"
    "只在basis说明是同时矛盾还是时间更新，不存在第七种边类型。"
    "equivalent=语义等价；sibling=独立的共同义务/有序流程成员；none=无直接语义依赖。"
    "条件与另一未触发例外之间不因同标题自动建边。"
    "conflict/equivalent/sibling/none使用undirected；context/exception必须有向。"
)
GRAPH_INSTRUCTION = (
    "你是记忆关系标注器。只按给定原子正文判断，不引入新义务或排他范围。"
    + GRAPH_RELATION_CONTRACT +
    "每个pair恰好一项，basis简短引用原文并解释行为差异。只输出裸JSON，不要Markdown或解释前缀："
    "{\"edges\":[{\"pair_id\":\"...\",\"type\":\"context|exception|conflict|equivalent|sibling|none\","
    "\"direction\":\"a_to_b|b_to_a|undirected\",\"confidence\":0.0,\"basis\":\"...\"}]}。"
)

_VALUE_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:分钟|小时|天|周|个月|条|个|份|MB|GB|%|字)")
_TIMEISH = re.compile(r"\d{4}-\d{2}|\d+\s*(?:分钟|小时|天|周|个月)")
_EXC = ("除非", "除外", "不适用", "不受此限")
_COND = ("若", "如果", "仅当")


                                                                 

def candidate_pairs(items: list[dict]) -> list[dict]:
    """内容候选：同项目同标题全配对 + 同项目共享来源的跨槽配对。不做数值过滤。"""
    by_slot: dict[tuple, list[dict]] = defaultdict(list)
    for item in items:
        by_slot[(item["project_id"], str(item["title"]).strip())].append(item)
    pairs, seen = [], set()

    def add(a: dict, b: dict, basis: str) -> None:
        if a["id"] == b["id"]:
            return
        key = tuple(sorted((a["id"], b["id"])))
        if key in seen:
            return
        seen.add(key)
        pairs.append({"pair_id": f"{key[0]}::{key[1]}", "a_id": key[0], "b_id": key[1],
                      "basis": basis})

    for (project, title), group in sorted(by_slot.items()):
        group = sorted(group, key=lambda i: i["id"])
        for a, b in list(combinations(group, 2))[:MAX_PAIRS_PER_SLOT]:
            add(a, b, f"same_slot:{title}")
    by_link: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        for link in item["links"]:
            by_link[link].append(item)
    for link, group in by_link.items():
        group = sorted(group, key=lambda i: i["id"])
        for i, a in enumerate(group):
            for b in group[i + 1:][:MAX_CROSS_PAIRS_PER_ITEM]:
                if a["project_id"] == b["project_id"] and a["title"] != b["title"]:
                    add(a, b, f"shared_source:{link}")
    return sorted(pairs, key=lambda p: p["pair_id"])


def rules_edges(items: list[dict]) -> list[dict]:
    """规则构造（fallback / dry-run）：只依赖内容，标注 edges_source=rules_fallback。"""
    by_slot: dict[tuple, list[dict]] = defaultdict(list)
    for item in items:
        by_slot[(item["project_id"], str(item["title"]).strip())].append(item)
    edges: dict[tuple, dict] = {}

    def add(src: dict, dst: dict, etype: str, basis: str) -> None:
        if src["id"] == dst["id"]:
            return
        edges.setdefault((etype, src["id"], dst["id"]),
                         {"type": etype, "from": src["id"], "to": dst["id"], "basis": basis,
                          "confidence": None, "edges_source": "rules_fallback"})

    for _, group in sorted(by_slot.items()):
        for a in group:
            for b in group:
                if a is b:
                    continue
                va, vb = set(_VALUE_RE.findall(a["text"])), set(_VALUE_RE.findall(b["text"]))
                if va and vb and va != vb and a["id"] < b["id"]:
                    add(a, b, "conflict", "same_slot_numeric_diff")
                elif va and va == vb and set(a["links"]) != set(b["links"]) and a["id"] < b["id"]:
                    add(a, b, "equivalent", "same_slot_same_value_diff_source")
                a_exc = any(m in a["text"] for m in _EXC)
                b_exc = any(m in b["text"] for m in _EXC)
                if a_exc and not b_exc and not any(m in b["text"] for m in _COND):
                    add(a, b, "exception", "exception_marker_targets_slot_rule")
                a_scope = any(m in a["text"] for m in _COND) or _TIMEISH.search(a["text"])
                b_plain = not (any(m in b["text"] for m in _COND) or _TIMEISH.search(b["text"])
                               or any(m in b["text"] for m in _EXC))
                if a_scope and b_plain:
                    add(a, b, "context", "slot_scope_fact_targets_rule")
    for a in items:
        for b in items:
            if a["id"] < b["id"] and a["title"] == b["title"]                     and "必须" in a["text"] and "必须" in b["text"]                     and not any(m in a["text"] + b["text"] for m in ("若", "如果", "不适用")):
                add(a, b, "sibling", "multi_sibling_shared_obligation_set")
    for a in items:
        for b in items:
            if a["id"] >= b["id"] or a["title"] == b["title"] or a["project_id"] != b["project_id"]:
                continue
            shared = set(a["links"]) & set(b["links"])
            if shared:
                add(a, b, "context", f"shared_source:{sorted(shared)[0]}")
    return sorted(edges.values(), key=lambda e: (e["type"], e["from"], e["to"]))


                                                                 

def build_graph_requests(items: list[dict], split: str, batch_size: int = BATCH_SIZE) -> list[dict]:
    by_id = {i["id"]: i for i in items}
    pairs = candidate_pairs(items)
    requests = []
    for index in range(0, len(pairs), batch_size):
        chunk = pairs[index:index + batch_size]
        payload = {"split": split, "pairs": [{
            "pair_id": p["pair_id"],
            "a": {"item_id": p["a_id"], "title": by_id[p["a_id"]]["title"],
                  "text": by_id[p["a_id"]]["text"], "links": by_id[p["a_id"]]["links"]},
            "b": {"item_id": p["b_id"], "title": by_id[p["b_id"]]["title"],
                  "text": by_id[p["b_id"]]["text"], "links": by_id[p["b_id"]]["links"]},
        } for p in chunk]}
        prompt = (GRAPH_INSTRUCTION + "\n----- 输入（只含记忆原子）-----\n"
                  + json.dumps(payload, ensure_ascii=False) + "\n")
        assert not any(k in prompt for k in ("required_facts", "gold", "question",
                                             "oracle", "target_group"))
        requests.append({"request_id": f"graph-{split}-b{index // batch_size:02d}",
                         "split": split, "pair_ids": [p["pair_id"] for p in chunk],
                         "prompt": prompt, "prompt_sha256": sha256(prompt),
                         "input_sha256": sha256(payload)})
    return requests


def parse_graph_output(text: str) -> tuple[list[dict], list[str]]:
    if not isinstance(text, str) or not text.strip():
        return [], ["empty_output"]
    start, depth, best = None, 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                best = text[start:i + 1]
    if best is None:
        return [], ["no_json_object"]
    try:
        value = json.loads(best)
    except json.JSONDecodeError:
        return [], ["bad_json"]
    edges = value.get("edges") if isinstance(value, dict) else None
    if not isinstance(edges, list):
        return [], ["missing_edges"]
    return [e for e in edges if isinstance(e, dict)], []


def ingest_graph(responses_dir: Path, split: str, items: list[dict]) -> dict:
    by_id = {i["id"]: i for i in items}
    requests = build_graph_requests(items, split)
    labels, batch_rows, parse_errors = [], [], []
    for request in requests:
        path = Path(responses_dir) / f"{request['request_id']}.txt"
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        parsed, errors = parse_graph_output(text)
        parse_errors.extend(errors)
        batch_rows.append({"request_id": request["request_id"], "found": path.exists(),
                           "labels": len(parsed), "parse_errors": errors,
                           "prompt_sha256": request["prompt_sha256"],
                           "raw_sha256": sha256(text) if text else None})
        labels.extend(parsed)
    edges = []
    for label in labels:
        pair_id = label.get("pair_id")
        etype = label.get("type")
        if etype not in EDGE_TYPES or not pair_id:
            continue
        a_id, _, b_id = pair_id.partition("::")
        if a_id not in by_id or b_id not in by_id:
            continue
        if etype == "none":
            continue
        direction = label.get("direction") if label.get("direction") in ("a_to_b", "b_to_a") \
            else "undirected"
        if etype in SYMMETRIC or direction == "undirected":
            src, dst = sorted((a_id, b_id))
        else:
            src, dst = (a_id, b_id) if direction == "a_to_b" else (b_id, a_id)
        edges.append({"type": etype, "from": src, "to": dst,
                      "basis": str(label.get("basis") or "")[:200],
                      "confidence": label.get("confidence"),
                      "pair_id": pair_id, "edges_source": "llm_graph"})
    unique = {(e["type"], e["from"], e["to"]): e for e in edges}
    edges = sorted(unique.values(), key=lambda e: (e["type"], e["from"], e["to"]))
    return {"schema": "graph-view-v2", "split": split, "edges": edges,
            "candidates": candidate_pairs(items), "batches": batch_rows,
            "parse_errors": parse_errors,
            "model": {"channel": "ingest", "model": "ingested_response_files", "called": False},
            "real": False, "placeholder": True,
            "block_reason": "legacy_unbound_receipts_use_pilot_v4_semantic_publication"}


                                                              

def _key(edge: dict, symmetric: bool) -> tuple:
    if symmetric or edge["type"] in SYMMETRIC:
        a, b = sorted((edge["from"], edge["to"]))
        return (edge["type"], a, b)
    return (edge["type"], edge["from"], edge["to"])


def diagnostics(edges: list[dict], oracle: list[dict], candidates: list[dict]) -> dict:
    candidate_pairs_set = {tuple(sorted((c["a_id"], c["b_id"]))) for c in candidates}
    oracle_edges = [e for e in oracle if e["from"] != e["to"]]
    candidate_hits = [e for e in oracle_edges
                      if tuple(sorted((e["from"], e["to"]))) in candidate_pairs_set]
    built_keys = {_key(e, e["type"] in SYMMETRIC) for e in edges}
    gold_keys = {_key(e, e["type"] in SYMMETRIC) for e in oracle_edges}
    per_type = {}
    for etype in ("context", "exception", "conflict", "equivalent", "sibling"):
        b = {k for k in built_keys if k[0] == etype}
        g = {k for k in gold_keys if k[0] == etype}
        per_type[etype] = {
            "built": len(b), "oracle": len(g), "matched": len(b & g),
            "precision": round(len(b & g) / len(b), 4) if b else None,
            "recall": round(len(b & g) / len(g), 4) if g else None,
        }
    return {
        "candidate_pairs": len(candidates),
        "candidate_recall": round(len(candidate_hits) / len(oracle_edges), 4)
        if oracle_edges else None,
        "candidate_missed": sorted([e["type"], e["from"], e["to"]]
                                   for e in oracle_edges if e not in candidate_hits),
        "built": len(built_keys), "oracle": len(gold_keys),
        "matched": len(built_keys & gold_keys), "per_type": per_type,
        "missed": sorted([list(k) for k in gold_keys - built_keys]),
        "spurious": sorted([list(k) for k in built_keys - gold_keys]),
        "note": "oracle 只用于评分诊断；候选与边均不含 gold。",
    }


                                                              

def save_graph(graph: dict) -> dict:
    folder = GRAPH_ROOT / graph["split"]
    folder.mkdir(parents=True, exist_ok=True)
    json_write(folder / "edges.json", {k: v for k, v in graph.items() if k != "candidates"})
    json_write(folder / "candidates.json", graph["candidates"])
    return {"edges": str(folder / "edges.json"), "sha256": sha256(graph["edges"])}


def load_edges(items: list[dict], split: str) -> dict:
    folder = GRAPH_ROOT / split
    path = folder / "edges.json"
    if path.exists():
        graph = json.loads(path.read_text(encoding="utf-8"))
        graph["real"] = False
        graph["placeholder"] = True
        graph["block_reason"] = "legacy_unbound_receipts_use_pilot_v4_semantic_publication"
        return graph
    return {"schema": "graph-view-v2", "split": split, "edges": rules_edges(items),
            "candidates": candidate_pairs(items), "real": False,
            "model": {"channel": "none", "model": "rules_fallback_v1", "called": False}}


                                                                    

def selftest() -> dict:
    items, id_map = load_v3_items()
    groups = []
    for g in load_groups():
        g2 = {**g, "atoms": [{**a, "id": id_map["items"][a["id"]], "links": []} for a in g["atoms"]],
              "atomic_fact_ids": {id_map["items"][k]: v for k, v in g["atomic_fact_ids"].items()}}
        groups.append(g2)
    by_id = {g["group_id"]: g for g in groups}
    results = {}
    candidates = candidate_pairs(items)
    oracle = oracle_edges(groups)
    diag = diagnostics(rules_edges(items), oracle, candidates)

                                                
    assert diag["candidate_recall"] == 1.0, diag["candidate_missed"]
    pub = id_map["items"]
    missed = {(t, pub[a], pub[b]) for t, a, b in [
        ("conflict", "F01-G4-A1", "F01-G4-A2"), ("conflict", "F03-G4-A1", "F03-G4-A2"),
        ("conflict", "F06-G4-A1", "F06-G4-A2"), ("conflict", "F07-G4-A1", "F07-G4-A2")]}
                              
    candidate_set = {tuple(sorted((c["a_id"], c["b_id"]))) for c in candidates}
    for etype, a, b in missed:
        assert tuple(sorted((a, b))) in candidate_set, (etype, a, b)
    results["candidate_recall_includes_ordering_and_species"] = True

                                   
    requests = build_graph_requests(items, "final")
    assert requests and all("gold" not in r["prompt"] and "question" not in r["prompt"]
                            and "oracle" not in r["prompt"] for r in requests)
    results["no_gold_in_prompt"] = True

                                   
    parsed, errors = parse_graph_output(json.dumps({"edges": [
        {"pair_id": "F01-G4-A1::F01-G4-A2", "type": "conflict", "direction": "b_to_a",
         "confidence": 0.8, "basis": "same slot diff value"},
        {"pair_id": "F01-G4-A2::F01-G4-A1", "type": "conflict", "direction": "a_to_b",
         "confidence": 0.8, "basis": "dup"},
        {"pair_id": "F01-G3-A3::F01-G3-A1", "type": "exception", "direction": "a_to_b",
         "confidence": 0.7, "basis": "exception"},
        {"pair_id": "F01-G1-A1::F01-G1-A2", "type": "none", "direction": "undirected"},
    ]}, ensure_ascii=False))
    assert not errors and len(parsed) == 4
    ingested = ingest_graph(Path("/nonexistent"), "dev", items)
    assert ingested["real"] is False and ingested["placeholder"] is True and ingested["edges"] == []
    results["parse_and_ingest"] = True

                                           
    def poisoned(rows):
        out = []
        for item in rows:
            copy_item = dict(item)
            copy_item["gold"] = {"required_facts": ["FAKE"]}
            copy_item["oracle_edges"] = [{"type": "conflict"}]
            out.append(copy_item)
        return out
    assert rules_edges(items) == rules_edges(poisoned(items))
    results["no_gold_edges"] = True

                                                              
    fake_labels = [{"pair_id": f"{e['from']}::{e['to']}", "type": e["type"],
                    "direction": "a_to_b", "confidence": 0.9, "basis": "oracle_like"}
                   for e in oracle if e["from"] != e["to"]]
    fake_edges = []
    for label in fake_labels:
        a, _, b = label["pair_id"].partition("::")
        fake_edges.append({"type": label["type"], "from": a, "to": b,
                           "edges_source": "llm_graph", "pair_id": label["pair_id"]})
    fake_diag = diagnostics(fake_edges, oracle, candidates)
    assert fake_diag["per_type"]["conflict"]["recall"] == 1.0
    assert fake_diag["built"] == fake_diag["oracle"]
    results["diagnostics_direction"] = True

                                  
    rules = rules_edges(items)
    assert rules and all(e["edges_source"] == "rules_fallback" for e in rules)
    assert load_edges(items, "final")["real"] is False
    results["fallback_labeled"] = True

    return {"ok": True, "results": results,
            "candidates": len(candidates), "requests_final": len(requests),
            "rules": {k: diag[k] for k in ("built", "oracle", "matched", "candidate_recall")},
            "model": {"channel": GRAPH_CHANNEL, "model": GRAPH_MODEL}}


def oracle_edges(groups: list[dict]) -> list[dict]:
    """与 arms 里 oracle_edges 同口径；此处内联避免循环导入。"""
    out = []
    for group in groups:
        atoms = group["atoms"]
        fact_atom = {}
        for atom in atoms:
            for fid in group["atomic_fact_ids"][atom["id"]]:
                fact_atom[fid] = atom["id"]
        by_type = defaultdict(list)
        for fact in group["facts"]:
            by_type[fact["type"]].append(fact)
        obligations = [fact_atom[f["fact_id"]] for f in by_type["obligation"]]
        obligation_atoms = set(obligations)
        for f in by_type["exception"]:
            for target in obligations:
                out.append({"type": "exception", "from": fact_atom[f["fact_id"]], "to": target})
        for f in by_type["condition"]:
            for target in obligations:
                out.append({"type": "context", "from": fact_atom[f["fact_id"]], "to": target})
        for f in by_type["time"]:
            if fact_atom[f["fact_id"]] in obligation_atoms:
                continue
            for target in obligations:
                out.append({"type": "context", "from": fact_atom[f["fact_id"]], "to": target})
        if group["profile"] == "multi_obligation":
            for i, a in enumerate(obligations):
                for b in obligations[i + 1:]:
                    out.append({"type": "sibling", "from": a, "to": b,
                                "basis": "multi_sibling"})
        if group["profile"] == "conflict_update":
            for i, a in enumerate(obligations):
                for b in obligations[i + 1:]:
                    out.append({"type": "conflict", "from": a, "to": b})
        for i, atom in enumerate(atoms):
            for other in atoms[i + 1:]:
                va = set(_VALUE_RE.findall(atom["text"]))
                vb = set(_VALUE_RE.findall(other["text"]))
                if va and va == vb and set(atom["links"]) != set(other["links"]):
                    out.append({"type": "equivalent", "from": atom["id"], "to": other["id"]})
    return out


def _cli(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] == "selftest":
        print(json.dumps(selftest(), ensure_ascii=False, indent=2))
        return 0
    groups, _id_map = load_v3_groups()
    if argv[1] == "prepare-graph":
        split = argv[argv.index("--split") + 1] if "--split" in argv else "dev"
        batch_size = int(argv[argv.index("--batch-size") + 1]) if "--batch-size" in argv else BATCH_SIZE
        items = [a for g in groups if g["split"] == split for a in g["atoms"]]
        requests = build_graph_requests(items, split, batch_size)
        folder = GRAPH_ROOT / split / "requests"
        folder.mkdir(parents=True, exist_ok=True)
        for request in requests:
            (folder / f"{request['request_id']}.prompt.txt").write_text(request["prompt"],
                                                                        encoding="utf-8")
        json_write(folder / "plan.json", {
            "split": split, "batch_size": batch_size, "calls": len(requests),
            "candidates": len(candidate_pairs(items)),
            "model": {"channel": GRAPH_CHANNEL, "model": GRAPH_MODEL},
            "requests": [{k: r[k] for k in ("request_id", "prompt_sha256", "input_sha256",
                                            "pair_ids")} for r in requests]})
        print(json.dumps({"ok": True, "split": split, "calls": len(requests),
                          "folder": str(folder)}, ensure_ascii=False, indent=2))
        return 0
    if argv[1] == "collect-graph":
        if "--enable-llm" not in argv:
            print(json.dumps({"ok": False, "error": "拒绝调用：需要显式 --enable-llm"},
                             ensure_ascii=False))
            return 2
        split = argv[argv.index("--split") + 1] if "--split" in argv else "dev"
        batch_size = int(argv[argv.index("--batch-size") + 1]) if "--batch-size" in argv else BATCH_SIZE
        max_calls = int(__import__("os").environ.get("MAX_CALLS", "16"))
        max_tokens = int(__import__("os").environ.get("MAX_TOKENS", "400000"))
        items = [a for g in groups if g["split"] == split for a in g["atoms"]]
        requests = build_graph_requests(items, split, batch_size)
        _, channel = compiler_line.load_batch_channel(enable_llm=True)
        result = compiler_line.collect_batch_requests(requests, channel, GRAPH_ROOT / split,
                                                      max_calls=max_calls, max_tokens=max_tokens)
        result["usage"] = compiler_line.batch_usage_summary(GRAPH_ROOT / split)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if argv[1] == "ingest-graph":
        split = argv[argv.index("--split") + 1] if "--split" in argv else "dev"
        responses = Path(argv[argv.index("--responses") + 1])
        items = [a for g in groups if g["split"] == split for a in g["atoms"]]
        graph = ingest_graph(responses, split, items)
        oracle = oracle_edges([g for g in groups if g["split"] == split])
        graph["diagnostics"] = diagnostics(graph["edges"], oracle, graph["candidates"])
        saved = save_graph(graph)
        print(json.dumps({"ok": True, "split": split, "edges": len(graph["edges"]),
                          "candidate_recall": graph["diagnostics"]["candidate_recall"], **saved},
                         ensure_ascii=False, indent=2))
        return 0
    raise SystemExit(f"未知命令：{argv[1]}")


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv))
