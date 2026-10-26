                      
"""第 4 线真实编译器接口（v2）：按 split 编译、侧表 provenance、分层发布审核。

边界：
- 预定编译通道 = 本机 Ollama `batiai/qwen3.8-27b:iq3`；审核通道必须与编译不同。
- **本阶段不调用任何模型**。`prepare-compiled` 只生成请求；`ingest-compiled` 只吃已落盘响应。
  没有真实响应时，dry-run 用 `placeholder_view`，并全程标 `real=False`，不得当臂结果。
- 编译输入只有：split/generation 的 active 原子正文、标题、links、来源元数据。
  不含任务、question、oracle、required IDs、gold。
- 输出每逻辑组 `card_id/text/source_item_ids/generation`；完整 provenance 进侧表，
  主 prompt 只保留紧凑标记（card id + generation）与来源 id 列表，不塞 payload。
- 发布 = 机械闸（schema/source mapping/全义务覆盖/数值/实体/极性）+ 独立语义审核。
  逐字/二元组闸降级为 `mechanical_lower_bound` 诊断，不再一票否决合法改写。
- 旧版无绑定回执的 ingest/cache 永久只供诊断；真实发布使用 pilot_v4 的回执与独立语义闸。

命令：
  python disclosure_compiler_20261025.py selftest
  python disclosure_compiler_20261025.py prepare-compiled --split dev --batch-size 8
  python disclosure_compiler_20261025.py ingest-compiled --split dev --responses out/compiled/dev/responses
  python disclosure_compiler_20261025.py lookup --card-id F01-G1
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from disclosure_schema_20261025 import (              
    DATA, EXP, OUT, bigrams, estimate_tokens, json_write, jsonl_read, jsonl_write,
    load_groups, sha256,
)


def publicize_groups(groups, id_map):
    """v3 公开视图：原子/复合/来源 ID 全部换成 m-/s-/g-，links 置空。"""
    out = []
    for g in groups:
        ng = {k: v for k, v in g.items()}
        ng["group_id"] = id_map["groups"][g["group_id"]]
        ng["split"] = "dev" if g["family_id"] in ("F01", "F02", "F03", "F04") else "final"
        ng["atoms"] = [{**a, "id": id_map["items"][a["id"]], "links": []}
                       for a in g["atoms"]]
        ng["compound"] = {**g["compound"], "id": id_map["items"][g["compound"]["id"]],
                          "links": []}
        ng["sources"] = [{**s, "source_id": id_map["sources"][s["source_id"]]}
                         for s in g["sources"]]
        ng["atomic_fact_ids"] = {
            id_map["items"][aid]: fids for aid, fids in g["atomic_fact_ids"].items()}
        out.append(ng)
    return out


def load_v3_groups():
    id_map = json.loads((DATA / "id_map.json").read_text(encoding="utf-8"))
    return publicize_groups(load_groups(), id_map), id_map

COMPILER_CHANNEL = "ollama"
COMPILER_MODEL = "batiai/qwen3.8-27b:iq3"
REVIEWER_CHANNEL = "chat"                     
REVIEWER_MODEL = "deepseek-v4-flash"
BATCH_SIZE = 8
COMPILE_ROOT = OUT / "compiled"

                               
ENTITY_LEXICON = ("人类", "小鼠", "大鼠", "人", "鼠", "物种", "样本", "细胞")
POLARITY_CLASSES = {
    "forbid": ("禁止", "不得", "不要", "不许", "不允许", "不能"),
    "require": ("必须", "最多", "至少", "应当", "只", "默认", "需要"),
    "prefer": ("优先", "尽量", "建议"),
}
POLARITY_MARKERS = {m for ms in POLARITY_CLASSES.values() for m in ms}
_VALUE_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:分钟|小时|天|周|个月|条|个|份|MB|GB|%|字)")

COMPILE_INSTRUCTION = (
    "你是记忆编译器的机械执行器。只根据输入的 active 原子与来源，把每个逻辑组合并成一张"
    "常规段落卡。不得引入输入中没有的义务、实体、条件或极性；不得省略任何义务；允许改写措辞。"
    "每组恰好输出一张卡：card_id 必须逐字等于该组 group_id；source_item_ids 必须完整列出该组"
    "全部 atoms[].item_id，不填 sources[].source_id；generation 必须是输入顶层 generation 的同一个整数。"
    "只输出 JSON：{\"cards\":[{\"card_id\":\"...\",\"text\":\"...\","
    "\"source_item_ids\":[\"...\"],\"generation\":__GENERATION__}]}。不要输出解释或 Markdown。"
)
REVIEW_INSTRUCTION = (
    "你是独立来源保真审核员，不参与编译，也不判断来源规则是否合理或内部一致。"
    "判断候选卡是否保留全部来源义务、实体、条件、时限、极性及来源间差异，不得新增或删除信息。"
    "来源原子彼此矛盾时，忠实卡必须保留这种矛盾，不得替用户选边、加优先级或化解冲突；"
    "不能仅因来源本身矛盾而拒绝忠实保留它的卡。用且/分号并列保留原有矛盾不是新增义务。"
    "允许合并共同主语、时间前缀及改写，不能将省略的共同前缀误判为删除条件。"
    "source_checks逐原子审核；evidence_in_card须取候选卡中实际存在的最小充分逐字片段，"
    "可以依靠卡文前文共同主语解释该片段，不能机械照抄来源句。"
    "只输出JSON，全部键保留：{\"card_id\":\"...\",\"serving\":true|false,"
    "\"all_obligations_preserved\":true|false,\"added_obligations\":[],"
    "\"entity_errors\":[],\"condition_or_scope_errors\":[],\"polarity_errors\":[],"
    "\"rationale\":\"逐条比较的理由\",\"source_checks\":[{\"item_id\":\"...\","
    "\"faithful\":true|false,\"evidence_in_card\":\"候选卡原文片段\"}]}。"
)


                                                                

def _chunks(rows: list, size: int):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def compile_batches(groups: list[dict], split: str, generation: int,
                    batch_size: int = BATCH_SIZE) -> list[dict]:
    """按 split 分批；不按 bank 重复调用。"""
    chosen = [g for g in groups if g["split"] == split]
    batches = []
    for index, chunk in enumerate(_chunks(chosen, batch_size)):
        batches.append({
            "request_id": f"compile-{split}-{generation}-b{index:02d}",
            "split": split, "generation": generation,
            "group_ids": [g["group_id"] for g in chunk],
            "input": {
                "split": split, "generation": generation,
                "groups": [{
                    "group_id": g["group_id"], "slot": g["slot"],
                    "atoms": [{"item_id": a["id"], "text": a["text"], "links": a["links"]}
                              for a in g["atoms"]],
                    "sources": [{"source_id": s["source_id"], "title": s["title"],
                                 "quote": s["quote"]} for s in g["sources"]],
                } for g in chunk],
            },
        })
    for batch in batches:
        batch["prompt"] = (COMPILE_INSTRUCTION.replace("__GENERATION__", str(int(generation)))
                           + "\n----- 输入（只含原子与来源）-----\n"
                           + json.dumps(batch["input"], ensure_ascii=False) + "\n")
        batch["prompt_sha256"] = sha256(batch["prompt"])
        batch["input_sha256"] = sha256(batch["input"])
        assert not any(k in batch["prompt"] for k in ("required_facts", "gold", "question",
                                                      "oracle", "target_group"))
    return batches


def parse_compile_output(text: str, batch: dict) -> tuple[list[dict], list[str]]:
    errors = []
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
    cards = value.get("cards") if isinstance(value, dict) else None
    if not isinstance(cards, list):
        return [], ["missing_cards"]
    out = []
    for card in cards:
        if not isinstance(card, dict):
            errors.append("card_not_object")
            continue
        out.append({k: card.get(k) for k in ("card_id", "text", "source_item_ids", "generation")})
    return out, errors


                                                               

def _entities(text: str) -> set[str]:
    return {e for e in ENTITY_LEXICON if e in text}


def _polarity_classes(text: str) -> set[str]:
    return {name for name, markers in POLARITY_CLASSES.items()
            if any(m in text for m in markers)}


def _numbers(text: str) -> set[str]:
    return set(_VALUE_RE.findall(text))


def mechanical_gate(cards: list[dict], groups: list[dict], generation: int,
                    requested_ids: list[str]) -> dict:
    by_group = {g["group_id"]: g for g in groups}
    seen, rows, errors = {}, [], []
    for card in cards:
        cid = card.get("card_id")
        text = card.get("text")
        sources = card.get("source_item_ids")
        gen = card.get("generation")
        row = {"card_id": cid, "generation": gen, "errors": []}
        if cid not in requested_ids or cid not in by_group:
            row["errors"].append("unknown_card_id")
        elif cid in seen:
            row["errors"].append("duplicate_card_id")
        seen[cid] = True
        if not isinstance(text, str) or not text.strip():
            row["errors"].append("empty_text")
            text = ""
        if not isinstance(sources, list) or not sources or \
                not all(isinstance(s, str) for s in sources):
            row["errors"].append("bad_source_item_ids")
            sources = []
        if gen != generation:
            row["errors"].append("generation_mismatch")
        group = by_group.get(cid)
        if group is None:
            rows.append(row)
            continue
        atom_ids = {a["id"] for a in group["atoms"]}
        unknown = [s for s in sources if s not in atom_ids]
        if unknown:
            row["errors"].append("unknown_source_item")
        atom_text = "；".join(a["text"] for a in group["atoms"] if a["id"] in set(sources))
        fact_holder = {fid: [a for a in group["atomic_fact_ids"] if fid in group["atomic_fact_ids"][a]]
                       for fid in {f["fact_id"] for f in group["facts"]}}
        covered = {fid for fid, holders in fact_holder.items()
                   if holders and set(holders) & set(sources)}
        coverage = len(covered) / len(fact_holder) if fact_holder else None
        if coverage != 1.0:
            row["errors"].append("coverage_incomplete")
        card_numbers, source_numbers = _numbers(text), _numbers(atom_text)
        fabricated = sorted(card_numbers - source_numbers)
        if fabricated:
            row["errors"].append("fabricated_value")
        card_entities, source_entities = _entities(text), _entities(atom_text)
        added_entities = sorted(card_entities - source_entities)
        if added_entities:
            row["errors"].append("entity_substitution")
        card_pol, source_pol = _polarity_classes(text), _polarity_classes(atom_text)
        if card_pol - source_pol:
            row["errors"].append("polarity_class_added")
        if source_pol - card_pol:
            row["errors"].append("polarity_class_missing")
        verbatim = sum(1 for a in group["atoms"] if a["id"] in set(sources)
                       and a["text"].rstrip("。") in text)
        cited_atoms = [a for a in group["atoms"] if a["id"] in set(sources)]
        source_grams = set()
        for atom in cited_atoms:
            source_grams |= bigrams(atom["text"])
        novel = bigrams(text.replace("【", "").replace("】", "")) - source_grams
        row.update({
            "coverage": round(coverage, 4) if coverage is not None else None,
            "missing_fact_ids": sorted(set(fact_holder) - covered),
            "fabricated_values": fabricated,
            "added_entities": added_entities,
            "polarity_added": sorted(card_pol - source_pol),
            "polarity_missing": sorted(source_pol - card_pol),
            "source_mapping_ok": not unknown,
            "verbatim_ratio": round(verbatim / len(cited_atoms), 4) if cited_atoms else None,
            "novel_bigram_count": len(novel),
            "mechanical_lower_bound_pass": bool(verbatim == len(cited_atoms) and not novel),
            "mechanical_pass": not row["errors"],
        })
        rows.append(row)
    missing_cards = [gid for gid in requested_ids if gid not in seen]
    if missing_cards:
        errors.append(f"missing_cards:{missing_cards}")
    return {"rows": rows, "errors": errors,
            "all_mechanical_pass": not errors and all(r.get("mechanical_pass") for r in rows),
            "coverage_mean": round(sum(r.get("coverage") or 0 for r in rows) / len(rows), 4)
            if rows else None,
            "verbatim_ratio_mean": round(sum(r.get("verbatim_ratio") or 0 for r in rows) / len(rows), 4)
            if rows else None}


                                                                  

def build_review_requests(cards: list[dict], groups: list[dict], generation: int) -> list[dict]:
    by_group = {g["group_id"]: g for g in groups}
    requests = []
    for card in cards:
        group = by_group.get(card.get("card_id"))
        if group is None:
            continue
        cited = [a for a in group["atoms"] if a["id"] in set(card.get("source_item_ids") or [])]
        payload = {
            "card_id": card["card_id"], "generation": generation,
            "card_text": card.get("text", ""),
            "source_atoms": [{"item_id": a["id"], "text": a["text"]} for a in cited],
            "sources": [{"source_id": s["source_id"], "quote": s["quote"]}
                        for s in group["sources"]],
        }
        prompt = (REVIEW_INSTRUCTION + "\n----- 待审 -----\n"
                  + json.dumps(payload, ensure_ascii=False) + "\n")
        assert not any(k in prompt for k in ("required_facts", "gold", "question", "oracle"))
        requests.append({"review_id": f"review-{card['card_id']}-{generation}",
                         "card_id": card["card_id"], "prompt": prompt,
                         "prompt_sha256": sha256(prompt), "payload": payload})
    return requests


def parse_review_output(text: str) -> tuple[dict | None, str]:
    if not isinstance(text, str) or not text.strip():
        return None, "empty_output"
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
        return None, "no_json_object"
    try:
        value = json.loads(best)
    except json.JSONDecodeError:
        return None, "bad_json"
    if not isinstance(value, dict):
        return None, "not_object"
    return value, "ok"


def publish(cards: list[dict], gate: dict, reviews: dict[str, dict] | None,
            *, reviewer_available: bool) -> dict:
    """发布 = 机械闸 ∧ 独立语义审核。审核缺位时不得当作通过（placeholder 除外）。"""
    gate_by_id = {r["card_id"]: r for r in gate["rows"]}
    rows = []
    for card in cards:
        cid = card.get("card_id")
        mech = gate_by_id.get(cid, {})
        review = (reviews or {}).get(cid)
        if reviewer_available:
            review_ok = bool(review and review.get("serving") is True
                             and review.get("all_obligations_preserved") is True
                             and not review.get("added_obligations")
                             and not review.get("entity_errors")
                             and not review.get("condition_or_scope_errors")
                             and not review.get("polarity_errors"))
            review_source = "independent_reviewer"
        else:
            review_ok = bool(mech.get("mechanical_pass"))
            review_source = "placeholder_not_called"
        serving = bool(mech.get("mechanical_pass")) and review_ok
        rows.append({
            "card_id": cid, "mechanical_pass": bool(mech.get("mechanical_pass")),
            "mechanical_errors": mech.get("errors", []),
            "review_source": review_source,
            "review_serving": review_ok,
            "serving": serving,
            "review": review,
        })
    return {"rows": rows, "serving_ids": [r["card_id"] for r in rows if r["serving"]],
            "blocked_ids": [r["card_id"] for r in rows if not r["serving"]],
            "all_serving": bool(rows) and all(r["serving"] for r in rows)}


                                                                 

def build_view(cards: list[dict], groups: list[dict], split: str, generation: int,
               publish_result: dict, *, compiler: dict, real: bool) -> dict:
    by_group = {g["group_id"]: g for g in groups}
    gate_by_id = {r["card_id"]: r for r in publish_result["rows"]}
    items, side = [], {}
    for card in cards:
        gid = card.get("card_id")
        group = by_group.get(gid)
        if group is None:
            continue
        cited = [a for a in group["atoms"] if a["id"] in set(card.get("source_item_ids") or [])]
        source_sids = sorted({sid for a in cited for sid in a["links"]})
        marker = f"〔{gid}·g{generation}〕"
        text = str(card.get("text") or "").strip()
        if marker not in text:
            text = text + marker
        items.append({
            "id": f"C-{gid}", "scope_level": "project",
            "project_id": group["compound"]["project_id"], "kind": "memory",
            "title": group["slot"], "text": text, "trigger": group["trigger"],
            "tier": "explicit", "permanence": group["compound"]["permanence"],
            "payload": {}, "links": source_sids, "revision": generation,
        })
        side[gid] = {
            "card_id": gid, "group_id": gid, "split": split, "generation": generation,
            "source_item_ids": list(card.get("source_item_ids") or []),
            "source_source_ids": source_sids,
            "text_sha256": sha256(text), "raw_text": text,
            "compiler": compiler,
            "serving": gate_by_id.get(gid, {}).get("serving", False),
            "mechanical": {k: gate_by_id.get(gid, {}).get(k) for k in
                           ("mechanical_pass", "coverage", "verbatim_ratio",
                            "novel_bigram_count", "mechanical_lower_bound_pass")},
            "review_source": gate_by_id.get(gid, {}).get("review_source"),
        }
    return {"schema": "compiled-view-v2", "split": split, "generation": generation,
            "cards": items, "side_table": side,
            "compiler": compiler.get("model", "placeholder"),
            "compiler_is_llm": bool(compiler.get("called")),
            "real": bool(real)}


def lookup(view: dict, card_id: str, groups: list[dict]) -> dict:
    side = view["side_table"].get(card_id)
    if not side:
        return {"found": False, "card_id": card_id}
    by_group = {g["group_id"]: g for g in groups}
    group = by_group.get(side["group_id"])
    atoms = {a["id"]: a for a in group["atoms"]} if group else {}
    return {"found": True, "card_id": card_id, "generation": side["generation"],
            "source_item_ids": side["source_item_ids"],
            "source_items": [{"item_id": i, "text": atoms[i]["text"], "links": atoms[i]["links"]}
                             for i in side["source_item_ids"] if i in atoms],
            "compiler": side["compiler"], "serving": side["serving"]}


def placeholder_view(groups: list[dict], split: str, generation: int = 0) -> dict:
    """没有真实编译响应时的接线视图：逐组拼接原子，明确标记非真实。"""
    chosen = [g for g in groups if g["split"] == split]
    cards = [{"card_id": g["group_id"],
              "text": "。".join(a["text"].rstrip("。") for a in g["atoms"]) + "。",
              "source_item_ids": [a["id"] for a in g["atoms"]],
              "generation": generation} for g in chosen]
    gate = mechanical_gate(cards, chosen, generation, [g["group_id"] for g in chosen])
    publication = publish(cards, gate, None, reviewer_available=False)
    view = build_view(cards, chosen, split, generation, publication,
                      compiler={"channel": "none", "model": "placeholder-template-v2",
                                "called": False, "prompt_sha256": None}, real=False)
    view["gate"] = gate
    view["publication"] = publication
    view["placeholder"] = True
    return view


def save_view(view: dict) -> dict:
    folder = COMPILE_ROOT / view["split"]
    folder.mkdir(parents=True, exist_ok=True)
    json_write(folder / "view.json", {k: v for k, v in view.items()
                                      if k not in ("gate", "publication")})
    json_write(folder / "side_table.json", view["side_table"])
    if "gate" in view:
        json_write(folder / "gate.json", view["gate"])
    if "publication" in view:
        json_write(folder / "publication.json", view["publication"])
    return {"view": str(folder / "view.json"),
            "sha256": sha256({k: v for k, v in view.items() if k != "cards"})}


def load_view(groups: list[dict], split: str, generation: int = 0) -> dict:
    """旧视图只供诊断；真实发布必须走 pilot_v4 的回执与独立保真闸。"""
    folder = COMPILE_ROOT / split
    view_path = folder / "view.json"
    if view_path.exists():
        view = json.loads(view_path.read_text(encoding="utf-8"))
        for name in ("side_table", "gate", "publication"):
            path = folder / f"{name}.json"
            if path.exists():
                view[name] = json.loads(path.read_text(encoding="utf-8"))
        view["real"] = False
        view["placeholder"] = True
        view["block_reason"] = "legacy_unbound_receipts_use_pilot_v4_semantic_publication"
        if view.get("generation") != generation:
            view["stale_generation"] = True
        return view
    return placeholder_view(groups, split, generation)


def ingest_responses(responses_dir: Path, split: str, groups: list[dict],
                     generation: int = 0) -> dict:
    """吃已落盘响应（每 batch 一个 .txt，文件名含 request_id），产出可发布视图。"""
    chosen = [g for g in groups if g["split"] == split]
    batches = compile_batches(chosen, split, generation)
    cards, batch_rows = [], []
    for batch in batches:
        path = Path(responses_dir) / f"{batch['request_id']}.txt"
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        parsed, errors = parse_compile_output(text, batch)
        batch_rows.append({"request_id": batch["request_id"], "found": path.exists(),
                           "cards": len(parsed), "parse_errors": errors,
                           "prompt_sha256": batch["prompt_sha256"],
                           "raw_sha256": sha256(text) if text else None})
        cards.extend(parsed)
    gate = mechanical_gate(cards, chosen, generation, [g["group_id"] for g in chosen])
    review_requests = build_review_requests(cards, chosen, generation)
    reviews = {}
    if review_requests:
        review_path = Path(responses_dir) / "reviews.jsonl"
        for row in jsonl_read(review_path):
            parsed, _ = parse_review_output(row.get("raw_text") or "")
            if parsed and parsed.get("card_id"):
                reviews[parsed["card_id"]] = parsed
    reviewer_available = bool(reviews)
    publication = publish(cards, gate, reviews, reviewer_available=reviewer_available)
    compiler = {"channel": "ingest",
                "model": "ingested_response_files", "called": False,
                "prompt_sha256": [b["prompt_sha256"] for b in batches],
                "batches": batch_rows}
    view = build_view(cards, chosen, split, generation, publication,
                      compiler=compiler, real=False)
    view["gate"] = gate
    view["publication"] = publication
    view["real"] = False
    view["placeholder"] = True
    view["block_reason"] = "legacy_unbound_receipts_use_pilot_v4_semantic_publication"
    return view




                                                                          

def collect_batch_requests(requests: list[dict], channel, out_dir: Path, *,
                           max_calls: int = 40, max_tokens: int = 400_000,
                           max_attempts: int = 2) -> dict:
    """逐 attempt 断点采集：raw 落盘、usage/latency 原样记录、成功不覆盖、预算到顶即停。

    编译器与图构建共用同一实现；调用方只传 request 列表与已建通道。
    本函数不判断模型内容，只保证可审计的落盘与恢复。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint.jsonl"
    prior = jsonl_read(checkpoint)
    succeeded = {e["request_id"] for e in prior if e.get("succeeded")}
    attempts: dict[str, int] = {}
    for entry in prior:
        attempts[entry["request_id"]] = attempts.get(entry["request_id"], 0) + 1
    calls = sum(1 for e in prior if (e.get("attempt") or 0) >= 1)
    tokens = sum((e.get("usage") or {}).get("total_tokens") or 0 for e in prior)
    stopped = None
    new_entries = []
    for request in requests:
        rid = request["request_id"]
        if rid in succeeded:
            continue
        attempt = attempts.get(rid, 0)
        while attempt < max_attempts:
            if calls >= max_calls:
                stopped = "max_calls"
                break
            if tokens >= max_tokens:
                stopped = "max_tokens"
                break
            outcome = channel.complete(request["prompt"])
            calls += 1
            tokens += (outcome.get("usage") or {}).get("total_tokens") or 0
            attempt += 1
            text = outcome.get("text") or ""
            (out_dir / f"{rid}.txt").write_text(text, encoding="utf-8")
            entry = {
                "schema": "batch-call-v1", "request_id": rid, "attempt": attempt,
                "ts": time_str(), "prompt_sha256": request["prompt_sha256"],
                "prompt_chars": len(request["prompt"]),
                "succeeded": bool(outcome.get("succeeded")),
                "status": outcome.get("status"), "error_kind": outcome.get("error_kind"),
                "usage": outcome.get("usage"), "latency_s": outcome.get("latency_s"),
                "provider": outcome.get("provider"), "model": outcome.get("model"),
                "raw_text_sha256": sha256(text) if text else None,
                "raw_text_chars": len(text),
                "cumulative_calls": calls, "cumulative_tokens": tokens,
            }
            with checkpoint.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            new_entries.append(entry)
            if entry["succeeded"]:
                succeeded.add(rid)
                break
        if stopped:
            break
    return {"calls": calls, "tokens": tokens, "new_entries": len(new_entries),
            "succeeded_requests": len(succeeded), "stopped": stopped,
            "checkpoint": str(checkpoint)}


def time_str() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def batch_usage_summary(out_dir: Path) -> dict:
    entries = jsonl_read(Path(out_dir) / "checkpoint.jsonl")
    ok = [e for e in entries if e.get("succeeded")]
    return {"attempts": len(entries), "succeeded": len(ok),
            "calls": sum(1 for e in entries if (e.get("attempt") or 0) >= 1),
            "tokens": sum((e.get("usage") or {}).get("total_tokens") or 0 for e in entries),
            "latency_s": round(sum(e.get("latency_s") or 0 for e in ok), 3),
            "provider_usage_rows": sum(1 for e in ok
                                       if (e.get("usage") or {}).get("source") == "provider")}


def load_batch_channel(*, channel: str | None = None, model: str | None = None,
                       enable_llm: bool = False):
    """拒绝未显式授权的调用；返回 (module, channel)。"""
    if not enable_llm:
        raise SystemExit("拒绝调用：需要显式 --enable-llm；本阶段只允许 prepare/ingest。")
    import importlib.util
    path = EXP.parents[1] / "原子关系跨模型复验与Qwen训练-2026-10-25" / "model_channel.py"
    spec = importlib.util.spec_from_file_location("model_channel_line4_batch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    endpoint = ""
    if (channel or COMPILER_CHANNEL) == "ollama":
        endpoint = __import__("os").environ.get("OLLAMA_ENDPOINT", "")
    else:
        endpoint = __import__("os").environ.get("LLM_BASE_URL", "")
    built = module.make_channel(channel or COMPILER_CHANNEL, model or COMPILER_MODEL,
                                endpoint=endpoint or None)
    return module, built


                                                                    

def selftest() -> dict:
    groups, _id_map = load_v3_groups()
    results = {}
                                                  
    dev = compile_batches(groups, "dev", 0, 8)
    final = compile_batches(groups, "final", 0, 8)
    assert len(dev) == math.ceil(24 / 8) and len(final) == math.ceil(48 / 8)
    assert all("question" not in b["prompt"] and "gold" not in b["prompt"] for b in dev + final)
    results["split_batched"] = True

                                                    
    view = placeholder_view(groups, "dev", 0)
    assert view["gate"]["all_mechanical_pass"]
    assert all(item["payload"] == {} for item in view["cards"])
    assert all("source_item_ids" not in item["text"] for item in view["cards"])
    assert view["real"] is False and view["placeholder"] is True
    results["placeholder_no_provenance"] = True

                        
    group = groups[0]
    card = {"card_id": group["group_id"],
            "text": "单次检索最多返回 5 条结果（改写：每次检索上限 5 条）。",
            "source_item_ids": [a["id"] for a in group["atoms"]], "generation": 0}
    gate = mechanical_gate([card], [group], 0, [group["group_id"]])
    assert gate["all_mechanical_pass"], gate["rows"][0]
    assert gate["rows"][0]["verbatim_ratio"] < 1.0
    assert gate["rows"][0]["mechanical_lower_bound_pass"] is False
    fake_review = {group["group_id"]: {"card_id": group["group_id"], "serving": True,
                                       "all_obligations_preserved": True,
                                       "added_obligations": [], "entity_errors": [],
                                       "condition_or_scope_errors": [], "polarity_errors": []}}
    pub = publish([card], gate, fake_review, reviewer_available=True)
    assert pub["all_serving"] is True
    results["paraphrase_publishable"] = True

                                                                  
                                
    mouse_group = next(g for g in groups if "人类" in g["compound"]["text"])
    human_atoms = [a["id"] for a in mouse_group["atoms"] if "人类" in a["text"]]
    bad_card = {"card_id": mouse_group["group_id"],
                "text": "同时返回人类和小鼠数据。",
                "source_item_ids": human_atoms, "generation": 0}
    bad_gate = mechanical_gate([bad_card], [mouse_group], 0, [mouse_group["group_id"]])
    row = bad_gate["rows"][0]
    assert "entity_substitution" in row["errors"], row["errors"]
    ok_review = {mouse_group["group_id"]: {"card_id": mouse_group["group_id"], "serving": True,
                                           "all_obligations_preserved": True,
                                           "added_obligations": [], "entity_errors": [],
                                           "condition_or_scope_errors": [], "polarity_errors": []}}
    bad_pub = publish([bad_card], bad_gate, ok_review, reviewer_available=True)
    assert bad_pub["all_serving"] is False
    results["human_to_mouse_blocked"] = True

                 
    polarity_group = next(g for g in groups if g["profile"] == "negation")
    flip = {"card_id": polarity_group["group_id"], "text": "可以复用超过 7 天的检索缓存。",
            "source_item_ids": [a["id"] for a in polarity_group["atoms"]], "generation": 0}
    flip_gate = mechanical_gate([flip], [polarity_group], 0, [polarity_group["group_id"]])
    assert any(e.startswith("polarity_") for e in flip_gate["rows"][0]["errors"])
    results["polarity_flip_blocked"] = True

                                       
    lookup_result = lookup(view, groups[0]["group_id"], groups)
    assert lookup_result["found"] and lookup_result["source_items"]
    assert view["side_table"][groups[0]["group_id"]]["source_item_ids"]
    results["side_table_lookup"] = True

                    
    assert COMPILER_CHANNEL != REVIEWER_CHANNEL
    results["reviewer_channel_separate"] = True

                                                 
    import shutil, tempfile
    from disclosure_schema_20261025 import jsonl_read as _read

    class _FakeChannel:
        name = "fake"
        model = "fake-batch"

        def __init__(self, texts):
            self.texts = list(texts)
            self.calls = 0

        def complete(self, prompt):
            self.calls += 1
            text = self.texts.pop(0) if self.texts else '{"cards":[]}'
            return {"text": text, "succeeded": bool(text), "status": "ok" if text else "empty_output",
                    "finish_reason": None,
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3,
                              "total_tokens": 10, "source": "provider"},
                    "latency_s": 0.02, "provider": "fake", "model": "fake-batch",
                    "error_kind": None, "attempted": True, "diagnostics": {}}

    folder = Path(tempfile.mkdtemp(prefix="batch-collect-", dir=str(OUT)))
    requests = dev[:2]
    channel = _FakeChannel(["", '{"cards":[]}'])
    first = collect_batch_requests(requests, channel, folder, max_calls=10, max_tokens=10 ** 6)
    assert channel.calls == 3, channel.calls
    assert first["new_entries"] == 3 and first["succeeded_requests"] == 2
    second = collect_batch_requests(requests, channel, folder, max_calls=10, max_tokens=10 ** 6)
    assert second["new_entries"] == 0 and channel.calls == 3
    folder_cap = Path(tempfile.mkdtemp(prefix="batch-cap-", dir=str(OUT)))
    cap = collect_batch_requests(requests, _FakeChannel([]), folder_cap,
                                 max_calls=1, max_tokens=10 ** 6)
    assert cap["stopped"] == "max_calls" and cap["calls"] == 1
    usage = batch_usage_summary(folder)
                                                                
    assert usage["provider_usage_rows"] == 2 and usage["tokens"] == 30
    assert all(e.get("latency_s") == 0.02 for e in _read(folder / "checkpoint.jsonl")
               if e.get("succeeded"))
    shutil.rmtree(folder, ignore_errors=True)
    shutil.rmtree(folder_cap, ignore_errors=True)
    results["batch_collector_resume_caps_usage"] = True
    return {"ok": True, "results": results,
            "batches": {"dev": len(dev), "final": len(final)},
            "compiler": {"channel": COMPILER_CHANNEL, "model": COMPILER_MODEL},
            "reviewer": {"channel": REVIEWER_CHANNEL, "model": REVIEWER_MODEL}}


                                                               

def _cli(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] == "selftest":
        print(json.dumps(selftest(), ensure_ascii=False, indent=2))
        return 0
    if argv[1] == "prepare-compiled":
        split = argv[argv.index("--split") + 1] if "--split" in argv else "dev"
        batch_size = int(argv[argv.index("--batch-size") + 1]) if "--batch-size" in argv else BATCH_SIZE
        groups, _ = load_v3_groups()
        batches = compile_batches(groups, split, 0, batch_size)
        folder = COMPILE_ROOT / split / "requests"
        folder.mkdir(parents=True, exist_ok=True)
        for batch in batches:
            (folder / f"{batch['request_id']}.prompt.txt").write_text(batch["prompt"],
                                                                      encoding="utf-8")
        json_write(folder / "plan.json", {"split": split, "batch_size": batch_size,
                                          "calls": len(batches),
                                          "compiler": {"channel": COMPILER_CHANNEL,
                                                       "model": COMPILER_MODEL},
                                          "requests": [{k: b[k] for k in
                                                        ("request_id", "prompt_sha256",
                                                         "input_sha256", "group_ids")}
                                                       for b in batches]})
        print(json.dumps({"ok": True, "split": split, "calls": len(batches),
                          "folder": str(folder)}, ensure_ascii=False, indent=2))
        return 0
    if argv[1] == "ingest-compiled":
        split = argv[argv.index("--split") + 1] if "--split" in argv else "dev"
        responses = Path(argv[argv.index("--responses") + 1])
        groups, _ = load_v3_groups()
        view = ingest_responses(responses, split, groups)
        saved = save_view(view)
        print(json.dumps({"ok": view["real"], "split": split,
                          "serving": len(view["publication"]["serving_ids"]),
                          "blocked": view["publication"]["blocked_ids"], **saved},
                         ensure_ascii=False, indent=2))
        return 0 if view["real"] else 3
    if argv[1] == "collect-compiled":
        if "--enable-llm" not in argv:
            print(json.dumps({"ok": False, "error": "拒绝调用：需要显式 --enable-llm"},
                             ensure_ascii=False))
            return 2
        split = argv[argv.index("--split") + 1] if "--split" in argv else "dev"
        batch_size = int(argv[argv.index("--batch-size") + 1]) if "--batch-size" in argv else BATCH_SIZE
        max_calls = int(os.environ.get("MAX_CALLS", "12"))
        max_tokens = int(os.environ.get("MAX_TOKENS", "400000"))
        batches = compile_batches(load_groups(), split, 0, batch_size)
        _, channel = load_batch_channel(enable_llm=True)
        result = collect_batch_requests(batches, channel, COMPILE_ROOT / split,
                                        max_calls=max_calls, max_tokens=max_tokens)
        result["usage"] = batch_usage_summary(COMPILE_ROOT / split)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if argv[1] == "lookup":
        card_id = argv[argv.index("--card-id") + 1]
        groups, _ = load_v3_groups()
        split = next(g["split"] for g in groups if g["group_id"] == card_id)
        view = load_view(groups, split, 0)
        print(json.dumps(lookup(view, card_id, groups), ensure_ascii=False, indent=2))
        return 0
    raise SystemExit(f"未知命令：{argv[1]}")


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv))
