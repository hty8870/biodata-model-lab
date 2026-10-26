from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[5] / 'source'
DATA = ROOT / "data"
sys.path.insert(0, str(REPO / "src"))

from dataset_recommender.agent import knowledge as K              
from dataset_recommender.agent import memory_relations as M              
from dataset_recommender.app.conversation_runs import RunError              
from dataset_recommender.app.conversation_store import ConversationStore              
from dataset_recommender.app.knowledge_service import KnowledgeService              
from dataset_recommender.llm.llm_client import LLMConfig              

OWNER = "bench"
QUOTE = "以后汇报先给结论，引用必须附来源。"
CFG = LLMConfig(enable_llm=True, api_key="synthetic-offline-key",
                base_url="https://model.example/v1", model="offline-bench")
ENDPOINT = hashlib.sha256(f"{CFG.base_url}|{CFG.model}".encode()).hexdigest()[:12]


def dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_entries(n: int, text_len: int, source_count: int, source_len: int,
                 representation: str) -> list[dict]:
    rows = []
    for index in range(n):
        group = index // 4
        sources = [{"id": f"s{group}-{suffix}", "type": "user",
                    "text": f"来源{suffix}：" + "证" * source_len,
                    "source_hash": hashlib.sha256(f"{group}-{suffix}".encode()).hexdigest()}
                   for suffix in range(source_count)]
        provenance = {"type": "knowledge_change", "change_id": f"kc_{group:06d}",
                      "assessment_id": f"ka_{group:06d}", "sources": sources,
                      "quotes": [source["text"][:1000] for source in sources]}
        if representation == "compact":
            provenance["source_refs"] = [{"type": row["type"], "id": row["id"]} for row in sources]
            provenance["sources"] = [{"type": row["type"], "id": row["id"]} for row in sources]
        elif representation != "full":
            raise ValueError(representation)
        rows.append({"id": f"ke_{index:06d}", "kind": "memory",
                     "title": f"原子义务{index}", "text": f"第{index}项" + "要" * text_len,
                     "trigger": "汇报时", "tier": "explicit", "permanence": "persistent",
                     "payload": {}, "links": [], "provenance": provenance,
                     "revision": 1, "status": "active", "project_id": "", "scope_level": "user",
                     "principal": OWNER, "endpoint_fp": ENDPOINT,
                     "created_at": None, "updated_at": None})
    return rows


def candidate() -> dict:
    return {"kind": "memory", "title": "新义务", "text": QUOTE,
            "trigger": "汇报时", "tier": "explicit", "permanence": "persistent",
            "payload": {}, "links": [], "evidence": [QUOTE], "principal": OWNER,
            "endpoint_fp": ENDPOINT, "project_id": "", "scope_level": "user"}


def offline_decision(request: dict) -> dict:
    refs = [request["evidence"][0]["id"]]
    return {"semantic_pairs": [{"old_id": row["id"], "relation": "overlap_incomparable",
                "reason": "离线合同响应。", "old_quote": row["text"][:100],
                "new_quote": request["candidate"]["text"][:100],
                "evidence_refs": refs, "error_code": None}
               for row in request["existing"]],
            "edit_checks": [], "evidence_refs": refs, "error_code": None}


def assess(entries: list[dict]) -> tuple[dict, list[dict]]:
    seen = []
    def predictor(request):
        seen.append(request)
        return offline_decision(request)
    started = time.perf_counter()
    result = M.assess_relations(candidate(), entries,
                                [{"id": "e1", "text": QUOTE}], CFG,
                                predictor=predictor, knowledge=True)
    return {"result": result, "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "batch_count": len(seen)}, seen


def semantic_pairs(calls: list[dict]) -> list[dict]:
    return [row["semantic_pair"] for call in calls for row in call["existing"]]


def seed_change_sources(service: KnowledgeService, source_entries: list[dict]) -> None:
    groups = {}
    for row in source_entries:
        provenance = row["provenance"]
        groups.setdefault(provenance["change_id"], provenance["sources"])
    with service.store.database(write=True) as conn:
        for change_id, sources in groups.items():
            change = {"id": change_id, "project_id": "", "request_id": "req-" + change_id,
                      "request_fingerprint": hashlib.sha256(("request" + change_id).encode()).hexdigest(),
                      "status": "committed", "revision": 2,
                      "fingerprint": hashlib.sha256(("change" + change_id).encode()).hexdigest(),
                      "operations": [], "source_refs": [{"type": row["type"], "id": row["id"]} for row in sources],
                      "sources": copy.deepcopy(sources), "actor": "bench",
                      "created_at": None, "updated_at": None}
            service.store.put_change(conn, OWNER, ENDPOINT, change)


def resolve_compact(service: KnowledgeService, entries: list[dict]) -> list[dict]:
    resolved = copy.deepcopy(entries)
    with service.store.database() as conn:
        for row in resolved:
            provenance = row["provenance"]
            change = service.store.change(conn, OWNER, ENDPOINT, provenance["change_id"])
            sources = change["sources"]
            provenance["sources"] = sources
            provenance.pop("source_refs", None)
    return resolved


def measure(entries: list[dict], with_assess: bool = False,
            source_entries: list[dict] | None = None) -> dict:
    DATA.mkdir(exist_ok=True)
    (DATA / "tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=DATA / "tmp") as tmp:
        root = Path(tmp)
        service = KnowledgeService(ConversationStore(root))
        seed_change_sources(service, source_entries or entries)
        with service.store.database(write=True) as conn:
            for row in entries:
                service.store.put_entry(conn, OWNER, ENDPOINT, row)
        resolution = None
        if source_entries is not None and any("source_refs" in row["provenance"] for row in entries):
            resolution = {"checked": True, "exact_restore": resolve_compact(service, entries) == source_entries}
        prompt = None
        prompt_error = None
        timings = []
        for _ in range(3):
            started = time.perf_counter()
            try:
                value = K.knowledge_context(root, "任意查询", principal=OWNER,
                                            endpoint_fp=ENDPOINT, available_tools=())
                prompt = value
                prompt_error = None
            except RunError as exc:
                prompt_error = exc.code
            timings.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        batches = M._size_fit_batches(candidate(), entries,
                                      [{"id": "e1", "text": QUOTE}], CFG.api_key, True)
        batch_ms = (time.perf_counter() - started) * 1000
        result = {"items": len(entries), "snapshot_chars": len(json.dumps(entries, ensure_ascii=False)),
                  "prompt_chars": len(prompt) if prompt is not None else None,
                  "prompt_error": prompt_error, "provenance_resolution": resolution,
                  "prompt_ms_median": round(statistics.median(timings), 3),
                  "relation_batches": len(batches), "relation_batch_rows": [len(x) for x in batches],
                  "batch_planning_ms": round(batch_ms, 3)}
        if with_assess:
            assessed, calls = assess(entries)
            result["assess"] = assessed
            result["assessed_pairs"] = semantic_pairs(calls)
        return result


def public_metrics(value: dict) -> dict:
    return {key: item for key, item in value.items()
            if key not in {"assess", "assessed_pairs"}}


def selftest() -> dict:
    full = make_entries(4, 20, 1, 100, "full")
    compact = make_entries(4, 20, 1, 100, "compact")
    full_assess, full_calls = assess(full)
    compact_assess, compact_calls = assess(compact)
    positive = (full_assess["result"]["coverage"]["complete"] is True
                and compact_assess["result"]["coverage"]["complete"] is True
                and semantic_pairs(full_calls) == semantic_pairs(compact_calls))
    corrupted = copy.deepcopy(compact)
    corrupted[0]["trigger"] = "已丢失条件"
    _, corrupted_calls = assess(corrupted)
    negative = semantic_pairs(full_calls) != semantic_pairs(corrupted_calls)
    full_measure = measure(full)
    compact_measure = measure(compact, source_entries=full)
    prompt_equal = full_measure["prompt_chars"] == compact_measure["prompt_chars"]
    large_full = make_entries(16, 200, 4, 1000, "full")
    large_compact = make_entries(16, 200, 4, 1000, "compact")
    full_limited = measure(large_full)
    compact_ok = measure(large_compact, with_assess=True, source_entries=large_full)
    budget_control = (full_limited["prompt_error"] == "knowledge_context_limit"
                      and compact_ok["prompt_error"] is None
                      and compact_ok["assess"]["result"]["coverage"]["complete"] is True)
    same_batches = full_limited["relation_batches"] == compact_ok["relation_batches"]
    resolution_ok = all(value["provenance_resolution"]["exact_restore"] is True
                        for value in (compact_measure, compact_ok))
    checks = {"consumer_positive": positive, "semantic_negative_control": negative,
              "compact_is_smaller": compact_measure["snapshot_chars"] < full_measure["snapshot_chars"],
              "prompt_unchanged": prompt_equal, "budget_countercontrol": budget_control,
              "compact_does_not_reduce_relation_batches": same_batches,
              "provenance_exact_restore": resolution_ok}
    return {"checks": checks, "passed": all(checks.values()),
            "full_error": full_assess["result"]["error_code"],
            "compact_error": compact_assess["result"]["error_code"],
            "full_coverage": full_assess["result"]["coverage"],
            "compact_coverage": compact_assess["result"]["coverage"],
            "large_compact_error": compact_ok["assess"]["result"]["error_code"],
            "full": public_metrics(full_measure), "compact": public_metrics(compact_measure),
            "large_full_error": full_limited["prompt_error"],
            "large_compact_batches": compact_ok["relation_batches"]}


def sweep() -> dict:
    rows = []
    for n in (16, 64, 128, 256, 512, 1000):
        for text_len in (200, 1000):
            for source_count in (1, 4):
                full_entries = make_entries(n, text_len, source_count, 1000, "full")
                for representation in ("full", "compact"):
                    entries = make_entries(n, text_len, source_count, 1000, representation)
                    value = measure(entries, source_entries=full_entries)
                    rows.append({"n": n, "text_len": text_len, "source_count": source_count,
                                 "source_len": 1000, "representation": representation,
                                 **public_metrics(value)})
    consumers = {}
    reference = make_entries(64, 200, 1, 1000, "full")
    for representation in ("full", "compact"):
        measured = measure(make_entries(64, 200, 1, 1000, representation),
                           with_assess=True, source_entries=reference)
        consumers[representation] = {
            "coverage": measured["assess"]["result"]["coverage"],
            "error_code": measured["assess"]["result"]["error_code"],
            "elapsed_ms": measured["assess"]["elapsed_ms"],
            "batch_count": measured["assess"]["batch_count"],
            "provenance_resolution": measured["provenance_resolution"],
            "pairs_sha256": hashlib.sha256(dump(measured["assessed_pairs"]).encode()).hexdigest()}
    consumers["semantic_pair_equal"] = (
        consumers["full"]["pairs_sha256"] == consumers["compact"]["pairs_sha256"])
    return {"schema": "atomic-architecture-bench-v1", "synthetic_only": True,
            "rows": rows, "consumer": consumers,
            "source_sha256": {
                "knowledge.py": sha(REPO / "src/dataset_recommender/agent/knowledge.py"),
                "memory_relations.py": sha(REPO / "src/dataset_recommender/agent/memory_relations.py"),
                "knowledge_service.py": sha(REPO / "src/dataset_recommender/app/knowledge_service.py"),
                "knowledge_store.py": sha(REPO / "src/dataset_recommender/app/knowledge_store.py"),
                "script": sha(Path(__file__))}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "run"))
    args = parser.parse_args()
    checked = selftest()
    print(json.dumps({"selftest": checked}, ensure_ascii=False))
    if not checked["passed"]:
        raise SystemExit(2)
    if args.command == "run":
        result = sweep()
        result["selftest"] = checked
        DATA.mkdir(exist_ok=True)
        target = DATA / "architecture-results.json"
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(target), "rows": len(result["rows"]),
                          "consumer": result["consumer"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
