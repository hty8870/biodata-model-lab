                      
"""Audited model transport, immutable call ledger, and fail-closed semantic publication."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import disclosure_compiler_20261025 as compiler
import disclosure_graph_llm_20261025 as graph_builder
from disclosure_schema_20261025 import REPO, estimate_tokens, file_sha256, json_write, jsonl_read, sha256
from pilot_data_20261025 import DATA, EXP, GENERATION

OUT = EXP / "out"
CHANNELS = OUT / "channels"
BATCH = 6
GRAPH_VERSION = 2
REVIEW_VERSION = 2
EDGE_TYPES = ("context", "exception", "conflict", "equivalent", "sibling", "none")
SYMMETRIC = {"conflict", "equivalent", "sibling"}


def artifact_path(kind, name):
    if kind == "compiled" and name == "publication":
        return CHANNELS / kind / "publication_v2_evidence_repaired_v1.json"
    suffix = "_v2" if kind == "graph" or name.startswith("review_") or name == "publication" else ""
    extension = ".json" if name in ("candidate", "publication") else ""
    return CHANNELS / kind / (name + suffix + extension)


def inside(path):
    path = Path(path).resolve()
    if not path.is_relative_to(EXP.resolve()):
        raise ValueError("output_outside_pilot")
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def immutable_json(path, value):
    path = inside(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError("immutable_binding_mismatch:" + path.name)
    else:
        json_write(path, value)


def load():
    return jsonl_read(DATA / "groups.jsonl"), read_json(DATA / "id_map.json")


def source_binding():
    groups, ids = load()
    manifest = read_json(DATA / "manifest.json")
    if manifest["generation"] != GENERATION:
        raise ValueError("stale_source_generation")
    for name, digest in manifest["sha256"].items():
        if file_sha256(DATA / name) != digest:
            raise ValueError("stale_source_manifest:" + name)
    return {"generation": GENERATION,
            "groups_sha256": file_sha256(DATA / "groups.jsonl"), "id_map_sha256": file_sha256(DATA / "id_map.json"),
            "revisions": {ids["items"][a["id"]]: {"revision": a["revision"], "text_sha256": sha256(a["text"])} for g in groups for a in g["atoms"]}}


def public_groups(groups=None, ids=None):
    if groups is None:
        groups, ids = load()
    return [{**g, "group_id": ids["groups"][g["group_id"]], "split": "shared",
             "atoms": [{**a, "id": ids["items"][a["id"]], "links": []} for a in g["atoms"]],
             "compound": {**g["compound"], "id": ids["items"][g["compound"]["id"]]},
             "sources": [{**s, "source_id": ids["sources"][s["source_id"]]} for s in g["sources"]],
             "atomic_fact_ids": {ids["items"][k]: v for k, v in g["atomic_fact_ids"].items()}} for g in groups]


def side_table():
    return {g["group_id"]: {"source_item_ids": [a["id"] for a in g["atoms"]],
                             "source_quotes": {s["source_id"]: s["quote"] for s in g["sources"]},
                             "atom_sources": {a["id"]: g["sources"][i]["source_id"] for i, a in enumerate(g["atoms"])}}
            for g in public_groups()}


def oracle(profile, group, id_map=None):
    ids = [a["id"] for a in group["atoms"]]
    if profile == "multi_obligation":
        return [{"type": "sibling", "from": a, "to": b} for i, a in enumerate(ids) for b in ids[i+1:]]
    if profile in ("conditional", "conditional_exception"):
        additive = len(ids) >= 2 and any(w in group["atoms"][1]["text"] for w in ("同时", "额外", "另外"))
        edges = [{"type": "context" if additive else "exception", "from": ids[1], "to": ids[0]}] if len(ids) >= 2 else []
        if profile == "conditional_exception" and len(ids) >= 3:
            edges.append({"type": "exception", "from": ids[2], "to": ids[0]})
        return edges
    if profile in ("simultaneous", "supersede") and len(ids) == 2:
        return [{"type": "conflict", "from": ids[0], "to": ids[1]}]
    return []


def bind_requests(requests, stage, binding=None):
    binding = binding or source_binding()
    out = []
    for request in requests:
        r = dict(request)
        r.update(stage=stage, source_sha256=sha256(binding), source_generation=binding["generation"], blocked=False, reason=None, placeholder=False)
        r["prompt_sha256"] = sha256(r["prompt"])
        r.setdefault("input_sha256", sha256({"prompt_sha256": r["prompt_sha256"], "source_sha256": r["source_sha256"]}))
        out.append(r)
    return out


def prepare(kind, *, save=True):
    pub = public_groups()
    binding = source_binding()
    if kind == "compiled":
        requests = compiler.compile_batches(pub, "shared", GENERATION, BATCH)
                                                                                          
        pairs = []
    elif kind == "graph":
        items = [a for g in pub for a in g["atoms"]]
        requests = graph_builder.build_graph_requests(items, "shared", BATCH)
        for r in requests:
            r["request_id"] = r["request_id"].replace("graph-shared-", "graph-shared-v2-")
            r["graph_protocol_version"] = GRAPH_VERSION
        pairs = graph_builder.candidate_pairs(items)
    else:
        raise ValueError("unknown_kind")
    requests = bind_requests(requests, kind if kind == "compiled" else "graph-v2", binding)
    if save:
        folder = artifact_path(kind, "requests")
        for r in requests:
            immutable_json(folder / f"{r['request_id']}.json", r)
        immutable_json(folder / "plan.json", dict(schema="pilot-channel-requests-r2", source=binding, requests=requests, candidate_pairs=pairs))
    return requests, pairs


def compile_prepare(split="shared"):
    if split != "shared":
        raise ValueError("shared_library_not_task_split")
    return prepare("compiled")[0]


def graph_prepare(split="shared"):
    if split != "shared":
        raise ValueError("shared_library_not_task_split")
    return prepare("graph")


def model_module():
    path = compiler.EXP.parents[1] / "原子关系跨模型复验与Qwen训练-2026-10-25" / "model_channel.py"
    spec = importlib.util.spec_from_file_location("pilot_existing_model_channel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_format_transport(transport):
    def send(endpoint, headers, payload, timeout):
        return transport(endpoint, headers, {**payload, "format": "json"}, timeout)
    return send


class RuntimeChannel:
    def __init__(self, name, model, *, enable_llm=False, max_output=4096):
        if not enable_llm:
            raise ValueError("explicit_enable_llm_required")
        self.mc = model_module()
        self.name, self.model, self.max_output = name, model, max_output
        if name == "ollama":
            _, self.delegate = compiler.load_batch_channel(channel=name, model=model, enable_llm=True)
            self.delegate.num_predict = max_output
            self.delegate.transport = json_format_transport(self.delegate.transport)
            endpoint, provider = self.delegate.endpoint, self.delegate.provider
            self.config = None
        elif name == "chat":
            from dataclasses import replace
            from dataset_recommender.llm.llm_client import load_llm_config
            cfg = load_llm_config(REPO)
            if cfg.mock_llm or cfg.provider == "mock":
                raise ValueError("mock_llm_forbidden")
            self.config = replace(cfg, model=model, enable_llm=True, max_tokens=max_output, temperature=0.0,
                                  thinking=False)
            endpoint, provider = cfg.base_url, cfg.provider
            self.delegate = None
        else:
            raise ValueError("only_production_chat_or_local_ollama")
        self.identity = dict(channel=name, model=model, provider=provider, endpoint_sha256=sha256(endpoint),
                             execution="authorized_model_call", backend_attested=False,
                             adapter_sha256=file_sha256(Path(__file__)), max_output=max_output,
                             ollama_format="json" if self.delegate is not None else None,
                             chat_thinking=self.config.thinking if self.config is not None else None)

    def complete(self, prompt):
        if self.delegate is not None:
            return self.delegate.complete(prompt)
        from dataset_recommender.llm.llm_client import call_llm
        started = time.perf_counter()
        result = call_llm(prompt, self.config)
        raw = result.raw_response
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        usage = (raw.get("usage") or {}) if isinstance(raw, dict) else {}
        tokens = self.mc.normalize_usage(prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"), total_tokens=usage.get("total_tokens"))
        succeeded = bool(result.succeeded) and result.finish_reason not in ("length", "max_tokens", "content_filter")
        return dict(text=result.text, succeeded=succeeded, attempted=bool(result.attempted),
                    status="ok" if succeeded else "llm_failure", finish_reason=result.finish_reason,
                    usage=tokens, latency_s=time.perf_counter()-started, provider=result.provider, model=result.model,
                    error_kind=None if succeeded else "llm_failure", diagnostics={})

    def sanitize(self, text):
        return self.mc.redact(text, limit=200_000)


class FixtureChannel:
    def __init__(self, fn, model="fixture-generator"):
        self.fn, self.calls, self.max_output = fn, 0, 4096
        self.identity = dict(channel="fixture", model=model, provider="offline", execution="fixture", backend_attested=False, max_output=4096)

    def complete(self, prompt):
        self.calls += 1
        value = self.fn(prompt, self.calls)
        if isinstance(value, dict):
            return value
        return dict(text=value, succeeded=bool(value), attempted=True, status="ok" if value else "empty_output",
                    finish_reason="stop", usage=dict(prompt_tokens=7, completion_tokens=3, total_tokens=10, source="fixture"),
                    latency_s=0.001, provider="offline", model=self.identity["model"], error_kind=None)

    def sanitize(self, text):
        return text


def append_event(path, row):
    prior = jsonl_read(path)
    row = {**row, "previous_sha256": prior[-1]["event_sha256"] if prior else None}
    row["event_sha256"] = sha256(row)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n")
        fh.flush()
        os.fsync(fh.fileno())
    return row


def audit_ledger(folder):
    folder = Path(folder)
    manifest = read_json(folder / "run_manifest.json")
    previous, started, finished = None, {}, {}
    for event in jsonl_read(folder / "ledger.jsonl"):
        checksum = event.get("event_sha256")
        if event.get("previous_sha256") != previous or sha256({k: v for k, v in event.items() if k != "event_sha256"}) != checksum:
            raise ValueError("ledger_chain_mismatch")
        if event["run_sha256"] != sha256(manifest):
            raise ValueError("ledger_run_mismatch")
        rid = event["request_id"]
        if event["event"] == "started":
            if rid in started:
                raise ValueError("duplicate_request_reservation")
            started[rid] = event
        elif event["event"] == "finished":
            if rid not in started or rid in finished:
                raise ValueError("unreserved_or_duplicate_finish")
            record = read_json(folder / f"{rid}.outcome.json")
            if sha256(record) != event["record_sha256"]:
                raise ValueError("ledger_record_mismatch")
            finished[rid] = event
        else:
            raise ValueError("unknown_ledger_event")
        previous = checksum
    return manifest, started, finished


def execute_requests(requests, channel, folder, *, binding=None, max_calls=None, max_tokens=2_000_000, max_new=None):
    folder = inside(folder)
    if not isinstance(channel, (RuntimeChannel, FixtureChannel)):
        raise ValueError("unregistered_transport")
    for r in requests:
        if r.get("blocked") or r.get("placeholder"):
            raise ValueError("blocked_or_fake_request")
        if sha256(r["prompt"]) != r["prompt_sha256"]:
            raise ValueError("prompt_sha_mismatch")
    if len({r["request_id"] for r in requests}) != len(requests):
        raise ValueError("duplicate_request_ids")
    max_calls = len(requests) if max_calls is None else max_calls
    manifest = dict(schema="pilot-call-run-r2", identity=channel.identity, binding=binding or {},
                    requests=[{k: r[k] for k in ("request_id", "prompt_sha256", "input_sha256", "source_sha256", "stage")} for r in requests],
                    max_calls=max_calls, max_tokens=max_tokens, automatic_retry=False)
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / ".writer-lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError("collector_already_locked")
    try:
        immutable_json(folder / "run_manifest.json", manifest)
        _, started, finished = audit_ledger(folder)
        charge = sum(e["budget_charge_tokens"] for e in started.values())
        new = 0
        for request in requests:
            rid = request["request_id"]
            if rid in started:
                continue
            reserve = estimate_tokens(request["prompt"]) + channel.max_output
            if len(started)+new >= max_calls or charge+reserve > max_tokens or max_new is not None and new >= max_new:
                break
            append_event(folder / "ledger.jsonl", dict(event="started", request_id=rid, run_sha256=sha256(manifest),
                         prompt_sha256=request["prompt_sha256"], budget_charge_tokens=reserve, ts=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
            charge += reserve
            try:
                outcome = channel.complete(request["prompt"])
            except Exception as exc:
                outcome = dict(text="", succeeded=False, attempted=True, status="transport_exception", error_kind=type(exc).__name__, usage=None, latency_s=None)
            text = channel.sanitize(outcome.get("text") or "")
            raw_path = folder / f"{rid}.raw.txt"
            with raw_path.open("x", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            usage = outcome.get("usage") or dict(prompt_tokens=None, completion_tokens=None, total_tokens=None, source="none")
            record = {k: outcome.get(k) for k in ("succeeded", "attempted", "status", "error_kind", "finish_reason", "latency_s", "provider", "model")}
            record.update(schema="pilot-call-receipt-r2", request_id=rid, prompt_sha256=request["prompt_sha256"],
                          input_sha256=request["input_sha256"], source_sha256=request["source_sha256"],
                          run_sha256=sha256(manifest), identity=channel.identity, usage=usage, text=text,
                          raw_sha256=file_sha256(raw_path), text_sha256=sha256(text), stage=request["stage"])
                                                                                   
            immutable_json(folder / f"{rid}.outcome.json", record)
            append_event(folder / "ledger.jsonl", dict(event="finished", request_id=rid, run_sha256=sha256(manifest),
                         record_sha256=sha256(record), succeeded=bool(record["succeeded"]), usage=usage))
            new += 1
        _, reserved, done = audit_ledger(folder)
        records = [read_json(folder / f"{rid}.outcome.json") for rid in done]
        return dict(total_requests=len(requests), reserved=len(reserved), completed=len(done), new_calls=new,
                    succeeded=sum(bool(r["succeeded"]) for r in records), failed=sum(not r["succeeded"] for r in records),
                    pending_unknown=sorted(set(reserved)-set(done)), remaining=len(requests)-len(reserved),
                    usage_total_tokens_all_sources=sum((r["usage"] or {}).get("total_tokens") or 0 for r in records),
                    provider_total_tokens=sum((r["usage"] or {}).get("total_tokens") or 0 for r in records if (r["usage"] or {}).get("source") == "provider"),
                    usage_unknown=sum((r["usage"] or {}).get("total_tokens") is None for r in records),
                    budget_charge_tokens=sum(e["budget_charge_tokens"] for e in reserved.values()),
                    execution=channel.identity["execution"], llm_called=isinstance(channel, RuntimeChannel) and bool(new))
    finally:
        lock.rmdir()


def response(folder, request):
    folder = inside(folder)
    manifest, _, finished = audit_ledger(folder)
    rid = request["request_id"]
    if rid not in finished:
        raise ValueError("response_missing_or_pending:"+rid)
    record = read_json(folder / f"{rid}.outcome.json")
    for key in ("prompt_sha256", "input_sha256", "source_sha256", "stage"):
        if record.get(key) != request[key]:
            raise ValueError("bad_provenance:"+key)
    expected = next((r for r in manifest["requests"] if r["request_id"] == rid), None)
    if not expected or any(expected[k] != request[k] for k in ("prompt_sha256", "input_sha256", "source_sha256", "stage")):
        raise ValueError("request_not_in_bound_manifest")
    if record["run_sha256"] != sha256(manifest) or record["identity"] != manifest["identity"] or not record.get("succeeded") or not record.get("attempted"):
        raise ValueError("failed_or_bad_identity")
    if record.get("model") != record["identity"]["model"] or record.get("provider") != record["identity"]["provider"]:
        raise ValueError("returned_model_identity_mismatch")
    raw_path = folder / f"{rid}.raw.txt"
    if file_sha256(raw_path) != record["raw_sha256"] or sha256(record["text"]) != record["text_sha256"] or raw_path.read_text(encoding="utf-8") != record["text"]:
        raise ValueError("raw_sha_mismatch")
    usage = record.get("usage")
    if not isinstance(usage, dict) or "source" not in usage or not record["identity"].get("model"):
        raise ValueError("usage_or_identity_missing")
    return record


def decode(text):
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        raise ValueError("json_only_required")
    if not isinstance(value, dict):
        raise ValueError("object_required")
    return value


def parsed_candidate(kind, records, requests):
    pub = public_groups()
    if kind == "compiled":
        cards = []
        for rec, req in zip(records, requests):
            obj = decode(rec["text"])
            batch = obj.get("cards")
            if set(obj) != {"cards"} or not isinstance(batch, list) or not all(isinstance(c, dict) and set(c) == {"card_id", "text", "source_item_ids", "generation"} for c in batch):
                raise ValueError("compiler_schema")
            if Counter(c["card_id"] for c in batch) != Counter(req["group_ids"]):
                raise ValueError("compiler_batch_coverage")
            cards.extend(batch)
        gate = compiler.mechanical_gate(cards, pub, GENERATION, [g["group_id"] for g in pub])
        if not gate["all_mechanical_pass"]:
            raise ValueError("compiler_mechanical_gate:"+json.dumps(gate, ensure_ascii=False))
        for c in cards:
            g = next(g for g in pub if g["group_id"] == c["card_id"])
            if len(c["source_item_ids"]) != len(set(c["source_item_ids"])) or set(c["source_item_ids"]) != {a["id"] for a in g["atoms"]}:
                raise ValueError("compiler_source_mapping")
        return dict(cards=cards, gate=gate, side_table=side_table())
    _, candidates = prepare("graph", save=False)
    by_pair = {p["pair_id"]: p for p in candidates}
    labels, edges = [], []
    for rec, req in zip(records, requests):
        obj = decode(rec["text"])
        batch = obj.get("edges")
        if set(obj) != {"edges"} or not isinstance(batch, list) or not all(isinstance(e, dict) for e in batch):
            raise ValueError("graph_schema")
        if Counter(e.get("pair_id") for e in batch) != Counter(req["pair_ids"]):
            raise ValueError("graph_batch_coverage")
        for e in batch:
            pair = by_pair[e["pair_id"]]
            typ, direction = e.get("type"), e.get("direction")
            if typ not in EDGE_TYPES or direction not in ("a_to_b", "b_to_a", "undirected") or not isinstance(e.get("basis"), str) or not e["basis"].strip():
                raise ValueError("graph_type_direction_basis")
            if (typ in SYMMETRIC or typ == "none") != (direction == "undirected"):
                raise ValueError("graph_symmetry_direction")
            labels.append(e)
            if typ != "none":
                src, dst = pair["a_id"], pair["b_id"]
                if direction == "b_to_a":
                    src, dst = dst, src
                edges.append(dict(type=typ, **{"from": src, "to": dst}, pair_id=e["pair_id"], basis=e["basis"]))
    expected = [e for g in pub for e in oracle(g["profile"], g)]
    diag = graph_builder.diagnostics(edges, expected, candidates)
    have = {tuple(sorted((e["from"], e["to"]))) for e in edges}
    want = {tuple(sorted((e["from"], e["to"]))) for e in expected}
    diag["connection_quality"] = dict(built=len(have), expected=len(want), matched=len(have & want),
                                      precision=len(have & want)/len(have) if have else None,
                                      recall=len(have & want)/len(want) if want else None,
                                      false_connections=[list(p) for p in sorted(have-want)],
                                      missed_connections=[list(p) for p in sorted(want-have)])
    diag["type_quality_note"] = "Typed labels follow the shared behavior contract v2; connectivity is reported separately. Retrieval expands all accepted edges irrespective of fine type."
    return dict(labels=labels, edges=edges, candidates=candidates, diagnostics=diag)


def ingest(kind, responses_dir, *, save=True):
    requests, _ = prepare(kind, save=save)
    records = [response(responses_dir, r) for r in requests]
    parsed = parsed_candidate(kind, records, requests)
    candidate = dict(schema="pilot-candidate-r2", kind=kind, source=source_binding(), source_sha256=sha256(source_binding()),
                     generation=GENERATION, responses_dir=str(inside(responses_dir).relative_to(EXP)),
                     request_sha256=sha256(requests), response_sha256=[sha256(r) for r in records],
                     identities=[r["identity"] for r in records], content=parsed,
                     real=False, placeholder=True, reason="independent_semantic_review_required")
    if save:
        immutable_json(artifact_path(kind, "candidate"), candidate)
    return candidate


def compile_ingest(split, responses_dir):
    if split != "shared":
        raise ValueError("shared_library_not_task_split")
    return ingest("compiled", responses_dir)


def graph_ingest(split, responses_dir):
    if split != "shared":
        raise ValueError("shared_library_not_task_split")
    return ingest("graph", responses_dir)


def candidate_current(kind):
    candidate = read_json(artifact_path(kind, "candidate"))
    if candidate["generation"] != GENERATION or candidate["source"] != source_binding():
        raise ValueError("stale_source_generation_or_revision")
    fresh = ingest(kind, EXP / candidate["responses_dir"], save=False)
    if fresh != candidate:
        raise ValueError("candidate_provenance_changed")
    return candidate


def review_requests(kind, *, save=True):
    candidate = candidate_current(kind)
    if kind == "compiled":
        reqs = compiler.build_review_requests(candidate["content"]["cards"], public_groups(), GENERATION)
        requests = []
        for r in reqs:
            addition = ("\n每个source_atoms[].item_id恰好一项source_checks；逐项核对条件、例外、时间、"
                        "并列或有序义务；允许共用前文主语，但evidence_in_card必须是候选卡实际存在的片段。"
                        "来源有矛盾时审核是否忠实保留，而非要求来源自洽。给出非空rationale。JSON-only。")
            requests.append(dict(request_id=r["review_id"]+"-v2", prompt=r["prompt"]+addition,
                                 card_id=r["card_id"], review_protocol_version=REVIEW_VERSION,
                                 input_sha256=sha256({"payload": r["payload"], "candidate": sha256(candidate), "review_version": REVIEW_VERSION})))
    else:
        requests = []
        items = {a["id"]: a for g in public_groups() for a in g["atoms"]}
        pairs = {p["pair_id"]: p for p in candidate["content"]["candidates"]}
        labels = candidate["content"]["labels"]
        for i in range(0, len(labels), BATCH):
            batch = labels[i:i+BATCH]
            payload = [{"label": e, "a": items[pairs[e["pair_id"]]["a_id"]], "b": items[pairs[e["pair_id"]]["b_id"]]} for e in batch]
            prompt = ("你是独立来源保真审核员，未参与构图。逐对核对正文、候选边及basis是否忠实；"
                      + graph_builder.GRAPH_RELATION_CONTRACT +
                      "只用上述六类，不要求新增类型。不能把默认擅自解释为‘仅’，不能替来源化解矛盾。"
                      "边存在、类型、方向、basis都须检查，不能仅因端点覆盖判保真。"
                      "若不忠实，serving=false并列错误；忠实则可通过。JSON-only且全部键保留："
                      '{"serving":true,"all_relations_faithful":true,"omissions":[],"extra_edges":[],"type_errors":[],"direction_errors":[],"rationale":"...","pair_checks":[{"pair_id":"...","faithful":true,"rationale":"逐对依据"}]}。'
                      "每个pair必须恰好审核一次。\n"+json.dumps(payload, ensure_ascii=False))
            requests.append(dict(request_id=f"review-graph-v2-g{GENERATION}-b{i//BATCH:02d}", prompt=prompt,
                                 pair_ids=[e["pair_id"] for e in batch], review_protocol_version=REVIEW_VERSION,
                                 input_sha256=sha256({"payload": payload, "candidate": sha256(candidate), "review_version": REVIEW_VERSION})))
    requests = bind_requests(requests, "review-"+kind+"-v2")
    if save:
        folder = artifact_path(kind, "review_requests")
        for r in requests:
            immutable_json(folder / f"{r['request_id']}.json", r)
        immutable_json(folder / "plan.json", dict(review_version=REVIEW_VERSION, candidate_sha256=sha256(candidate), requests=requests))
    return requests


def independent_identity(producer, reviewer):
    return bool(producer.get("model") and reviewer.get("model") and
                producer["model"].strip().casefold() != reviewer["model"].strip().casefold())


def semantic_gate(kind, candidate, reviews, requests, effective_values=None):
    reasons = []
    producer_ids = candidate["identities"]
    if len(reviews) != len(requests):
        reasons.append("missing_review")
    for record, request in zip(reviews, requests):
        if not all(independent_identity(p, record["identity"]) for p in producer_ids):
            reasons.append("reviewer_not_independent")
        value = (effective_values or {}).get(request.get("request_id"), decode(record["text"]))
        if kind == "compiled":
            keys = {"card_id", "serving", "all_obligations_preserved", "added_obligations", "entity_errors", "condition_or_scope_errors", "polarity_errors", "rationale", "source_checks"}
            card = next(c for c in candidate["content"]["cards"] if c["card_id"] == request["card_id"])
            if set(value) != keys or value.get("card_id") != card["card_id"] or value.get("serving") is not True or value.get("all_obligations_preserved") is not True:
                reasons.append("semantic_rejection_or_schema")
            if any(value.get(k) != [] for k in ("added_obligations", "entity_errors", "condition_or_scope_errors", "polarity_errors")):
                reasons.append("semantic_errors")
            checks = value.get("source_checks") or []
            if Counter(x.get("item_id") for x in checks if isinstance(x, dict)) != Counter(card["source_item_ids"]):
                reasons.append("semantic_source_checks_missing")
            if any(x.get("faithful") is not True or not x.get("evidence_in_card") or x["evidence_in_card"] not in card["text"] for x in checks):
                reasons.append("semantic_evidence_missing")
        else:
            keys = {"serving", "all_relations_faithful", "omissions", "extra_edges", "type_errors", "direction_errors", "rationale", "pair_checks"}
            if set(value) != keys or value.get("serving") is not True or value.get("all_relations_faithful") is not True:
                reasons.append("semantic_rejection_or_schema")
            if any(value.get(k) != [] for k in ("omissions", "extra_edges", "type_errors", "direction_errors")):
                reasons.append("semantic_errors")
            checks = value.get("pair_checks") or []
            if Counter(x.get("pair_id") for x in checks if isinstance(x, dict)) != Counter(request["pair_ids"]):
                reasons.append("semantic_pair_checks_missing")
            if any(x.get("faithful") is not True or not str(x.get("rationale") or "").strip() for x in checks):
                reasons.append("semantic_pair_rejected")
        if not isinstance(value.get("rationale"), str) or not value["rationale"].strip():
            reasons.append("semantic_rationale_missing")
    return dict(semantic_pass=not reasons, reasons=sorted(set(reasons)), coverage_is_not_faithfulness=True)


def apply_evidence_quote(value, card, item_id, quote):
    if value.get("serving") is not True or value.get("all_obligations_preserved") is not True or any(value.get(k) != [] for k in ("added_obligations", "entity_errors", "condition_or_scope_errors", "polarity_errors")) or not all(x.get("faithful") is True for x in value["source_checks"]):
        raise ValueError("evidence_repair_cannot_change_semantic_rejection")
    if not isinstance(quote, str) or not quote.strip() or quote not in card["text"]:
        raise ValueError("repair_quote_not_literal")
    result = copy.deepcopy(value)
    matches = [x for x in result["source_checks"] if x["item_id"] == item_id]
    if len(matches) != 1 or matches[0].get("evidence_in_card") and matches[0]["evidence_in_card"] in card["text"]:
        raise ValueError("repair_only_invalid_single_reference")
    matches[0]["evidence_in_card"] = quote
    return result


def evidence_repair_requests(candidate=None, reviews=None, requests=None, *, save=True):
    candidate = candidate if candidate is not None else candidate_current("compiled")
    requests = requests if requests is not None else review_requests("compiled", save=False)
    reviews = reviews if reviews is not None else [response(artifact_path("compiled", "review_responses"), r) for r in requests]
    if len(reviews) != len(requests):
        raise ValueError("repair_missing_base_review")
    atoms = {a["id"]: a for g in public_groups() for a in g["atoms"]}
    out = []
    for record, req in zip(reviews, requests):
        gate = semantic_gate("compiled", candidate, [record], [req])
        if gate["semantic_pass"]:
            continue
        value = decode(record["text"])
        if gate["reasons"] != ["semantic_evidence_missing"] or not all(x.get("faithful") is True for x in value["source_checks"]):
            raise ValueError("repair_not_evidence_only")
        card = next(c for c in candidate["content"]["cards"] if c["card_id"] == req["card_id"])
        for check in value["source_checks"]:
            if check.get("evidence_in_card") and check["evidence_in_card"] in card["text"]:
                continue
            payload = dict(old_review=value, old_review_sha256=sha256(record), source_item=atoms[check["item_id"]],
                           card=card, source_item_id=check["item_id"], derivation_rule="replace_only_evidence_in_card_v1",
                           derivation_code_sha256=file_sha256(Path(__file__)))
            prompt = ("只做引用格式补证，不重新审判语义，不改变原审核任何裁决。给定原审核、对应source原子与完整卡文，"
                      "抽取卡文中实际存在、支持该source义务的最小充分逐字片段。卡文可能把共同主语放在前文；"
                      "禁止补出省略的共同前缀，不能照抄不存在的完整source句。只输出裸JSON"
                      '{"evidence_in_card":"卡文中真实存在的逐字片段"}，不返回其他键；确实不能抽取则返回空字符串。\n'
                      + json.dumps(payload, ensure_ascii=False))
            out.append(dict(request_id="evidence-repair-"+req["request_id"]+"-"+check["item_id"], prompt=prompt,
                            input_sha256=sha256(payload), base_review_id=req["request_id"], base_review_sha256=sha256(record),
                            item_id=check["item_id"], card_id=card["card_id"], reviewer_identity=record["identity"],
                            derivation_code_sha256=payload["derivation_code_sha256"]))
    out = bind_requests(out, "evidence-repair-v1")
    if save:
        folder = CHANNELS / "compiled/evidence_repair_v1/requests"
        for req in out:
            immutable_json(folder / (req["request_id"]+".json"), req)
        immutable_json(folder / "plan.json", dict(max_attempts_per_reference=1, requests=out))
    return out


def collect_evidence_repair(*, enable_llm=False):
    if not enable_llm:
        raise ValueError("explicit_enable_llm_required")
    requests = evidence_repair_requests()
    if not requests:
        return dict(new_calls=0, reason="no_invalid_evidence")
    expected = requests[0]["reviewer_identity"]
    keys = ("channel", "model", "provider", "endpoint_sha256", "chat_thinking")
    if any(any(req["reviewer_identity"].get(k) != expected.get(k) for k in keys) for req in requests):
        raise ValueError("repair_reviewers_differ")
    channel = RuntimeChannel(expected["channel"], expected["model"], enable_llm=True, max_output=300)
    if any(channel.identity.get(k) != expected.get(k) for k in keys):
        raise ValueError("repair_requires_same_reviewer_before_call")
    return execute_requests(requests, channel, CHANNELS / "compiled/evidence_repair_v1/responses", max_calls=len(requests), max_tokens=20_000)


def evidence_repair_views(candidate, reviews, requests, *, save=False):
    repairs = evidence_repair_requests(candidate, reviews, requests, save=save)
    values, lineage, records = {}, [], []
    by_id = {req["request_id"]: rec for req, rec in zip(requests, reviews)}
    for req in repairs:
        rec = response(CHANNELS / "compiled/evidence_repair_v1/responses", req)
        original = by_id[req["base_review_id"]]
        if rec["identity"].get("execution") != "authorized_model_call" or any(rec["identity"].get(k) != original["identity"].get(k) for k in ("channel", "model", "provider", "endpoint_sha256", "chat_thinking")):
            raise ValueError("repair_requires_same_real_independent_reviewer")
        parsed = decode(rec["text"])
        if set(parsed) != {"evidence_in_card"}:
            raise ValueError("repair_output_schema")
        card = next(c for c in candidate["content"]["cards"] if c["card_id"] == req["card_id"])
        value = values.get(req["base_review_id"], decode(original["text"]))
        values[req["base_review_id"]] = apply_evidence_quote(value, card, req["item_id"], parsed["evidence_in_card"])
        records.append(rec)
        lineage.append(dict(base_review_id=req["base_review_id"], old_review_sha256=sha256(original),
                            old_raw_sha256=original["raw_sha256"], repair_request_sha256=sha256(req),
                            repair_receipt_sha256=sha256(rec), item_id=req["item_id"],
                            derivation_rule="replace_only_evidence_in_card_v1", derivation_code_sha256=req["derivation_code_sha256"]))
    view = dict(kind="derived_review_values_not_provider_outcomes", values=values, lineage=lineage)
    path = CHANNELS / "compiled/evidence_repair_v1/derived_views.json"
    if save:
        immutable_json(path, view)
    elif not path.exists() or read_json(path) != view:
        raise ValueError("derived_view_missing_or_tampered")
    return view, records


def publication(kind, review_dir, *, save=True):
    candidate = candidate_current(kind)
    requests = review_requests(kind, save=save)
    reviews = [response(review_dir, r) for r in requests]
    views, repair_records = None, []
    if kind == "compiled":
        views, repair_records = evidence_repair_views(candidate, reviews, requests, save=save)
    gate = semantic_gate(kind, candidate, reviews, requests, views["values"] if views else None)
    authentic = all(i.get("execution") == "authorized_model_call" for i in candidate["identities"] + [r["identity"] for r in reviews+repair_records])
    serving = authentic and gate["semantic_pass"]
    artifact = dict(schema="pilot-publication-r2", kind=kind, source=candidate["source"], generation=GENERATION,
                    candidate_sha256=sha256(candidate), review_dir=str(inside(review_dir).relative_to(EXP)),
                    review_sha256=[sha256(r) for r in reviews], gate=gate, authentic_runtime=authentic,
                    real=bool(serving), placeholder=not serving, fake=not authentic,
                    reason=None if serving else "fixture_permanently_nonreal" if not authentic else "semantic_gate_failed",
                    content=candidate["content"])
    if views is not None:
        artifact["derived_review_views"] = views
        artifact["derived_review_views_sha256"] = sha256(views)
    if save:
        immutable_json(artifact_path(kind, "publication"), artifact)
    return artifact


def load_publication(kind):
    path = artifact_path(kind, "publication")
    if not path.exists():
        return None, "missing_independently_reviewed_"+kind
    try:
        artifact = read_json(path)
        if artifact.get("fake") or artifact.get("placeholder") or not artifact.get("real"):
            return None, "fake_or_unpublished_"+kind
        if artifact["generation"] != GENERATION or artifact["source"] != source_binding():
            return None, "stale_source_generation_or_revision"
        fresh = publication(kind, EXP / artifact["review_dir"], save=False)
        if artifact != fresh or not fresh["real"]:
            return None, "publication_provenance_or_semantics_changed"
        return artifact, None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, "artifact_revalidation_failed:"+type(exc).__name__


def selftest():
    checks = []
    def ck(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)
    pub = public_groups()
    ck("shared12", len(pub) == 12)
    ck("two_atom_conditional_oracle", all(len(oracle(g["profile"], g)) == 1 for g in pub if g["profile"] == "conditional"))
    ck("multi_uses_sibling", all(e["type"] == "sibling" for g in pub if g["profile"] == "multi_obligation" for e in oracle(g["profile"], g)))
    for kind in ("compiled", "graph"):
        requests, pairs = prepare(kind)
        ck(kind+":no_gold", all(not any(k in r["prompt"] for k in ("required_facts", "question", "oracle", "target_group")) for r in requests))
    _, pairs = prepare("graph")
    oracle_edges = [e for g in pub for e in oracle(g["profile"], g)]
    diag = graph_builder.diagnostics([], oracle_edges, pairs)
    ck("candidate_vs_type_quality_separate", diag["candidate_recall"] == 1.0 and diag["matched"] == 0)
    miss = graph_builder.diagnostics([], oracle_edges, [])
    ck("candidate_miss_visible", miss["candidate_recall"] == 0.0)
    with tempfile.TemporaryDirectory(prefix="offline-channel-", dir=inside(OUT)) as tmp:
        folder = Path(tmp)
        reqs = bind_requests([dict(request_id=f"r{i}", prompt=f"safe {i}") for i in range(216)], "offline-queue")
        channel = FixtureChannel(lambda p, n: "" if n == 1 else '{"ok":true}')
        first = execute_requests(reqs, channel, folder, max_calls=216, max_new=11)
        second = execute_requests(reqs, channel, folder, max_calls=216)
        third = execute_requests(reqs, channel, folder, max_calls=216)
        ck("216_queue_resume_failure_denominator", first["new_calls"] == 11 and second["completed"] == 216 and second["failed"] == 1 and third["new_calls"] == 0 and channel.calls == 216)
        ck("failed_usage_retained", second["usage_total_tokens_all_sources"] == 2160 and second["provider_total_tokens"] == 0 and second["execution"] == "fixture")
        try:
            execute_requests([reqs[0] | {"prompt": "changed", "prompt_sha256": sha256("changed")}]+reqs[1:], channel, folder)
        except ValueError:
            ck("resume_sha_hard_reject", True)
        else:
            ck("resume_sha_hard_reject", False)
        try:
            response(folder, reqs[1] | {"source_sha256": "bad"})
        except ValueError:
            ck("bad_provenance_rejected", True)
        else:
            ck("bad_provenance_rejected", False)
        raw = folder / "r1.raw.txt"
        raw.write_text("tampered", encoding="utf-8")
        try:
            response(folder, reqs[1])
        except ValueError:
            ck("raw_tamper_rejected", True)
        else:
            ck("raw_tamper_rejected", False)
    try:
        RuntimeChannel("ollama", "unused", enable_llm=False)
    except ValueError:
        ck("no_authorization_no_channel", True)
    else:
        ck("no_authorization_no_channel", False)
    ck("independent_identity", not independent_identity(dict(channel="x", model="same"), dict(channel="x", model="same")))
    return dict(ok=True, passed=len(checks), checks=checks, llm_called=False, real=False, placeholder=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("selftest", "prepare", "collect", "ingest", "prepare-review", "collect-review", "publish", "status"))
    p.add_argument("--kind", choices=("compiled", "graph"), default="compiled")
    p.add_argument("--responses", type=Path)
    p.add_argument("--channel", choices=("chat", "ollama"), default="ollama")
    p.add_argument("--model", default=compiler.COMPILER_MODEL)
    p.add_argument("--enable-llm", action="store_true")
    p.add_argument("--max-new", type=int)
    p.add_argument("--max-tokens", type=int, default=2_000_000)
    a = p.parse_args()
    if a.command == "selftest":
        result = selftest()
        json_write(OUT / "channel_selftest.json", result)
    elif a.command == "prepare":
        reqs, pairs = prepare(a.kind)
        result = dict(ok=True, calls=len(reqs), candidates=len(pairs), source_sha256=sha256(source_binding()), llm_called=False)
    elif a.command in ("collect", "collect-review"):
        if not a.enable_llm:
            raise ValueError("explicit_enable_llm_required")
        review = a.command == "collect-review"
        reqs = review_requests(a.kind) if review else prepare(a.kind)[0]
        if review:
            candidate = candidate_current(a.kind)
            reviewer_identity = dict(channel=a.channel, model=a.model)
            if any(i.get("execution") != "authorized_model_call" for i in candidate["identities"]):
                raise ValueError("real_review_of_fake_candidate_forbidden")
            if not all(independent_identity(i, reviewer_identity) for i in candidate["identities"]):
                raise ValueError("reviewer_not_independent")
        folder = a.responses or artifact_path(a.kind, "review_responses" if review else "responses")
        channel = RuntimeChannel(a.channel, a.model, enable_llm=a.enable_llm)
        result = execute_requests(reqs, channel, folder, max_new=a.max_new, max_tokens=a.max_tokens)
    elif a.command == "ingest":
        result = ingest(a.kind, a.responses or artifact_path(a.kind, "responses"))
    elif a.command == "prepare-review":
        reqs = review_requests(a.kind)
        result = dict(ok=True, review_requests=len(reqs), llm_called=False)
    elif a.command == "publish":
        result = publication(a.kind, a.responses or artifact_path(a.kind, "review_responses"))
    else:
        artifact, reason = load_publication(a.kind)
        result = dict(real=bool(artifact), placeholder=not artifact, reason=reason, llm_called=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if a.command == "publish" and not result.get("real"):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
