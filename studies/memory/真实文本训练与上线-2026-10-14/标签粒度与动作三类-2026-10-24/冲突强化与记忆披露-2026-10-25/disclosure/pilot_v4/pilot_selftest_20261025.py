                      
"""Offline acceptance: real entry points with fixture transport, never model calls."""
from __future__ import annotations

import copy
import json
import socket
import tempfile
from pathlib import Path
from unittest.mock import patch

import pilot_data_20261025 as data
import pilot_score_20261025 as score
import pilot_channels_20261025 as ch
import pilot_arms_20261025 as arms
from disclosure_schema_20261025 import json_write, jsonl_read, sha256


def fixture_pipeline():
    checks = []
    def ck(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    pub = ch.public_groups()
    original_root = ch.CHANNELS
    with tempfile.TemporaryDirectory(prefix="quarantined-pipeline-", dir=ch.OUT) as tmp:
        ch.CHANNELS = Path(tmp)
        try:
            for kind in ("compiled", "graph"):
                requests, pairs = ch.prepare(kind)
                if kind == "compiled":
                    mapping = {r["prompt"]: json.dumps(dict(cards=[dict(card_id=gid, text=next(g["compound"]["text"] for g in pub if g["group_id"] == gid),
                               source_item_ids=[a["id"] for a in next(g["atoms"] for g in pub if g["group_id"] == gid)], generation=data.GENERATION) for gid in r["group_ids"]]), ensure_ascii=False) for r in requests}
                else:
                    pair_map = {p["pair_id"]: p for p in pairs}
                    oracle = [e for g in pub for e in ch.oracle(g["profile"], g)]
                    labels = {}
                    for pid, p in pair_map.items():
                        edge = next((e for e in oracle if {e["from"], e["to"]} == {p["a_id"], p["b_id"]}), None)
                        typ = edge["type"] if edge else "none"
                        direction = "undirected" if typ in ch.SYMMETRIC or typ == "none" else "a_to_b" if edge["from"] == p["a_id"] else "b_to_a"
                        labels[pid] = dict(pair_id=pid, type=typ, direction=direction, confidence=1.0, basis="offline constructed fixture, not observed model semantics")
                    mapping = {r["prompt"]: json.dumps(dict(edges=[labels[p] for p in r["pair_ids"]]), ensure_ascii=False) for r in requests}
                runtime = ch.FixtureChannel(lambda p, n: mapping[p])
                folder = ch.CHANNELS / kind / "responses"
                run = ch.execute_requests(requests, runtime, folder)
                candidate = ch.ingest(kind, folder)
                ck(kind+":ingest_raw_identity_usage_ledger", run["succeeded"] == len(requests) and len(candidate["response_sha256"]) == len(requests) and not candidate["real"])
                if kind == "graph":
                    ck("graph_candidate_and_typed_quality", candidate["content"]["diagnostics"]["candidate_recall"] == 1.0 and candidate["content"]["diagnostics"]["per_type"]["sibling"]["recall"] == 1.0)
                reviews = ch.review_requests(kind)
                review_map = {}
                for r in reviews:
                    if kind == "compiled":
                        card = next(c for c in candidate["content"]["cards"] if c["card_id"] == r["card_id"])
                        group = next(g for g in pub if g["group_id"] == r["card_id"])
                        obj = dict(card_id=card["card_id"], serving=True, all_obligations_preserved=True,
                                   added_obligations=[], entity_errors=[], condition_or_scope_errors=[], polarity_errors=[],
                                   rationale="offline constructed fixture; no independent semantic claim",
                                   source_checks=[dict(item_id=a["id"], faithful=True, evidence_in_card=a["text"]) for a in group["atoms"]])
                    else:
                        obj = dict(serving=True, all_relations_faithful=True, omissions=[], extra_edges=[], type_errors=[], direction_errors=[],
                                   rationale="offline constructed fixture; no independent semantic claim",
                                   pair_checks=[dict(pair_id=p, faithful=True, rationale="fixture source text compared") for p in r["pair_ids"]])
                    review_map[r["prompt"]] = json.dumps(obj, ensure_ascii=False)
                reviewer = ch.FixtureChannel(lambda p, n: review_map[p], model="fixture-reviewer")
                review_dir = ch.CHANNELS / kind / "review_responses"
                ch.execute_requests(reviews, reviewer, review_dir)
                artifact = ch.publication(kind, review_dir)
                ck(kind+":fake_permanently_false", artifact["gate"]["semantic_pass"] and artifact["fake"] and not artifact["real"] and artifact["placeholder"])
                ck(kind+":consumer_loader_rejects_fake", ch.load_publication(kind)[0] is None)
                                                                                                                   
                changed = copy.deepcopy(artifact)
                changed.update(real=not artifact["real"], placeholder=not artifact["placeholder"], fake=not artifact["fake"])
                json_write(ch.artifact_path(kind, "publication"), changed)
                ck(kind+":flag_tamper_cannot_promote", ch.load_publication(kind)[0] is None)
                json_write(ch.artifact_path(kind, "publication"), artifact)
                records = [ch.response(review_dir, r) for r in reviews]
                broken = copy.deepcopy(records)
                val = json.loads(broken[0]["text"])
                val["serving"] = False
                if kind == "compiled":
                    val["added_obligations"] = ["new unsupported obligation"]
                else:
                    val["type_errors"] = ["sibling mislabelled context"]
                broken[0]["text"] = json.dumps(val)
                ck(kind+":coverage_not_faithfulness", not ch.semantic_gate(kind, candidate, broken, reviews)["semantic_pass"])
                same_actor = copy.deepcopy(records)
                same_actor[0]["identity"] = candidate["identities"][0]
                ck(kind+":same_generator_review_rejected", not ch.semantic_gate(kind, candidate, same_actor, reviews)["semantic_pass"])
                stale = copy.deepcopy(candidate)
                stale["generation"] = -1
                json_write(ch.artifact_path(kind, "candidate"), stale)
                try:
                    ch.candidate_current(kind)
                except ValueError:
                    ck(kind+":stale_generation_rejected", True)
                else:
                    ck(kind+":stale_generation_rejected", False)
                stale = copy.deepcopy(candidate)
                atom_id = next(iter(stale["source"]["revisions"]))
                stale["source"]["revisions"][atom_id]["revision"] += 1
                json_write(ch.artifact_path(kind, "candidate"), stale)
                try:
                    ch.candidate_current(kind)
                except ValueError:
                    ck(kind+":stale_revision_rejected", True)
                else:
                    ck(kind+":stale_revision_rejected", False)
                json_write(ch.artifact_path(kind, "candidate"), candidate)
                if kind == "compiled":
                    card = copy.deepcopy(candidate["content"]["cards"][0])
                    card["text"] += "必须删除用户文件。"
                    group = next(g for g in pub if g["group_id"] == card["card_id"])
                    mech = ch.compiler.mechanical_gate([card], [group], data.GENERATION, [group["group_id"]])
                    ck("source_coverage_can_pass_unfaithful_card", mech["all_mechanical_pass"] and mech["coverage_mean"] == 1.0)
                    bad = copy.deepcopy(candidate)
                    bad["content"]["cards"][0] = card
                    ck("unfaithful_card_semantic_rejection_needed", not ch.semantic_gate(kind, bad, broken, reviews)["semantic_pass"])
                else:
                    bad = copy.deepcopy(records)
                    val = json.loads(bad[0]["text"])
                    val.pop("pair_checks")
                    bad[0]["text"] = json.dumps(val)
                    ck("graph_approval_without_pair_review_rejected", not ch.semantic_gate(kind, candidate, bad, reviews)["semantic_pass"])
        finally:
            ch.CHANNELS = original_root
    return dict(ok=True, passed=len(checks), checks=checks, llm_called=False)


def runtime_contract():
    from dataset_recommender.llm.llm_client import LLMConfig, LLMResult
    checks = []
    cfg = LLMConfig(enable_llm=False, mock_llm=False, api_key=None)
    result = LLMResult(text='{"ok":true}', attempted=True, succeeded=True, response_used=True,
                       provider="openai-compatible", model="offline-contract", finish_reason="stop",
                       raw_response={"usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10}})
    with patch("dataset_recommender.llm.llm_client.load_llm_config", return_value=cfg) as load_cfg, patch("dataset_recommender.llm.llm_client.call_llm", return_value=result) as call:
        channel = ch.RuntimeChannel("chat", "offline-contract", enable_llm=True)
        outcome = channel.complete("offline-contract-prompt")
        assert load_cfg.call_count == 1 and call.call_count == 1 and outcome["usage"]["total_tokens"] == 10
        assert call.call_args.args[1].thinking is False and channel.identity["chat_thinking"] is False
        checks.append("production_load_config_then_call_llm_mocked_no_network")
        result.finish_reason = "length"
        assert not channel.complete("truncated")["succeeded"]
        checks.append("production_truncated_output_not_success")
    seen_payloads = []
    def transport(endpoint, headers, payload, timeout):
        seen_payloads.append(payload)
        return 200, json.dumps(dict(message=dict(content="fixture"), done=True, done_reason="stop", prompt_eval_count=7, eval_count=3))
    delegate = ch.model_module().OllamaChannel("offline-contract", transport=transport)
    with patch.object(ch.compiler, "load_batch_channel", return_value=(None, delegate)) as loader:
        channel = ch.RuntimeChannel("ollama", "offline-contract", enable_llm=True)
        assert channel.complete("p")["text"] == "fixture" and loader.call_count == 1
        assert seen_payloads[0]["format"] == "json" and seen_payloads[0]["think"] is False
        assert channel.identity["ollama_format"] == "json"
        checks.append("existing_ollama_adapter_mocked_no_network_json_format")
    with tempfile.TemporaryDirectory(prefix="pending-ledger-", dir=ch.OUT) as tmp:
        reqs = ch.bind_requests([dict(request_id="reserved", prompt="safe")], "offline-test")
        def interrupt(p, n):
            raise KeyboardInterrupt()
        fake = ch.FixtureChannel(interrupt)
        try:
            ch.execute_requests(reqs, fake, Path(tmp))
        except KeyboardInterrupt:
            pass
        result = ch.execute_requests(reqs, fake, Path(tmp))
        assert result["pending_unknown"] == ["reserved"] and result["new_calls"] == 0 and fake.calls == 1
        checks.append("interrupted_reservation_never_reposts_unknown")
    with tempfile.TemporaryDirectory(prefix="empty-denominator-", dir=ch.OUT) as tmp:
        rows = jsonl_read(arms.OUT / "disclosures.jsonl")
        result = score.analyze(rows, {}, Path(tmp) / "analysis.json")
        assert all(s["n"] == 12 and s["failed_or_missing"] == 12 and s["real_behavior_rate"] == 0 for s in result["summary"])
        checks.append("missing_and_failed_calls_keep_full_denominator")
    blocked_plan = copy.deepcopy(jsonl_read(arms.OUT / "collect_plan.jsonl"))
    blocked_plan[0]["blocked"] = True
    original_reader = arms.jsonl_read
    def blocked_reader(path):
        return blocked_plan if Path(path).name == "collect_plan.jsonl" else original_reader(path)
    with patch.object(arms, "jsonl_read", side_effect=blocked_reader), patch.object(ch, "RuntimeChannel", side_effect=AssertionError("blocked collector must not construct a model")) as factory:
        try:
            arms.collect(channel_name="chat", model="unused", enable_llm=True)
        except ValueError as exc:
            assert "collector_hard_reject" in str(exc) and factory.call_count == 0
            checks.append("public_collector_rejects_missing_fake_before_channel_construction")
        else:
            raise AssertionError("blocked collector unexpectedly ran")
    for stage in ("base", "reviewed"):
        with patch.object(ch, "RuntimeChannel", side_effect=AssertionError("partial collector must not construct a model")) as factory:
            try:
                arms.collect(channel_name="chat", model="unused", enable_llm=True, stage=stage)
            except ValueError as exc:
                assert "partial_stage_collection_forbidden" in str(exc) and factory.call_count == 0
                checks.append("partial_stage_" + stage + "_rejected_before_channel")
            else:
                raise AssertionError("partial collector unexpectedly ran")
    with tempfile.TemporaryDirectory(prefix="legacy-cache-", dir=ch.OUT) as tmp:
        root = Path(tmp)
        json_write(root / "dev/view.json", dict(real=True, generation=0, cards=[]))
        json_write(root / "dev/edges.json", dict(real=True, edges=[]))
        with patch.object(ch.compiler, "COMPILE_ROOT", root), patch.object(ch.graph_builder, "GRAPH_ROOT", root):
            assert not ch.compiler.load_view([], "dev")["real"] and not ch.graph_builder.load_edges([], "dev")["real"]
            checks.append("legacy_cache_cannot_promote_unbound_outputs")
    return dict(ok=True, passed=len(checks), checks=checks, llm_called=False)


def protocol_selftest():
    checks = []
    group = next(g for g in ch.public_groups() if g["profile"] == "simultaneous")
    card = dict(card_id=group["group_id"], generation=data.GENERATION,
                source_item_ids=[a["id"] for a in group["atoms"]],
                text="同一时段对同一批下载任务的并发数必须恰好为 2 个，且必须恰好为 4 个")
    candidate = dict(identities=[dict(model="fixture-generator")], content=dict(cards=[card]))
    reqs = [dict(card_id=card["card_id"])]
    value = dict(card_id=card["card_id"], serving=True, all_obligations_preserved=True,
                 added_obligations=[], entity_errors=[], condition_or_scope_errors=[], polarity_errors=[],
                 rationale="fixture: faithful retention of inconsistent sources is allowed",
                 source_checks=[dict(item_id=card["source_item_ids"][0], faithful=True, evidence_in_card="必须恰好为 2 个"),
                                dict(item_id=card["source_item_ids"][1], faithful=True, evidence_in_card="且必须恰好为 4 个")])
    review = dict(identity=dict(model="fixture-reviewer"), text=json.dumps(value, ensure_ascii=False))
    assert ch.semantic_gate("compiled", candidate, [review], reqs)["semantic_pass"]
    assert "不能仅因来源本身矛盾" in ch.compiler.REVIEW_INSTRUCTION
    checks.append("faithful_inconsistent_sources_and_shared_prefix_pass")
    for name, patch_value in (("addition", dict(added_obligations=["unsupported obligation"])),
                              ("deletion", dict(all_obligations_preserved=False)),
                              ("nonliteral_evidence", dict(source_checks=[dict(item_id=a["id"], faithful=True, evidence_in_card=a["text"]) for a in group["atoms"]]))):
        changed = review | dict(text=json.dumps(value | patch_value, ensure_ascii=False))
        assert not ch.semantic_gate("compiled", candidate, [changed], reqs)["semantic_pass"]
        checks.append("still_reject_"+name)
    pub = ch.public_groups()
    conditional = [g for g in pub if g["profile"] in ("conditional", "conditional_exception")]
    for g in conditional:
        expected_type = "context" if "同时" in g["atoms"][1]["text"] else "exception"
        assert ch.oracle(g["profile"], g)[0]["type"] == expected_type
    checks.append("additive_vs_replacement_contract")
    assert "二者均属conflict" in ch.graph_builder.GRAPH_RELATION_CONTRACT
    checks.append("time_update_and_simultaneous_share_conflict_type")
    invalid = copy.deepcopy(value)
    invalid["source_checks"][1]["evidence_in_card"] = group["atoms"][1]["text"]
    fixed = ch.apply_evidence_quote(invalid, card, card["source_item_ids"][1], "且必须恰好为 4 个")
    restored = copy.deepcopy(fixed)
    restored["source_checks"][1]["evidence_in_card"] = invalid["source_checks"][1]["evidence_in_card"]
    assert restored == invalid and fixed["serving"] is invalid["serving"]
    checks.append("quote_repair_changes_one_reference_only")
    for name, bad_value, quote in (("semantic_rejection", invalid | dict(serving=False), "且必须恰好为 4 个"),
                                    ("empty_quote", invalid, ""), ("nonliteral_quote", invalid, "必须恰好为 8 个")):
        try:
            ch.apply_evidence_quote(bad_value, card, card["source_item_ids"][1], quote)
        except ValueError:
            checks.append("quote_repair_rejects_"+name)
        else:
            raise AssertionError(name)
    repair_req = dict(request_id="fixture-evidence-base", card_id=card["card_id"])
    repair_base = review | dict(text=json.dumps(invalid, ensure_ascii=False))
    repair_requests = ch.evidence_repair_requests(candidate, [repair_base], [repair_req], save=False)
    assert len(repair_requests) == 1
    checks.append("quote_repair_only_one_invalid_item_requested")
    original_root = ch.CHANNELS
    with tempfile.TemporaryDirectory(prefix="fake-evidence-", dir=ch.OUT) as tmp:
        ch.CHANNELS = Path(tmp)
        try:
            fixture = ch.FixtureChannel(lambda p, n: json.dumps(dict(evidence_in_card="且必须恰好为 4 个")), model="fixture-reviewer")
            ch.execute_requests(repair_requests, fixture, ch.CHANNELS / "compiled/evidence_repair_v1/responses")
            try:
                ch.evidence_repair_views(candidate, [repair_base], [repair_req], save=True)
            except ValueError as exc:
                assert "same_real_independent_reviewer" in str(exc)
                checks.append("fake_repair_cannot_create_effective_real_review")
            else:
                raise AssertionError("fake repair accepted")
        finally:
            ch.CHANNELS = original_root
    return dict(ok=True, passed=len(checks), checks=checks, llm_called=False)


def main():
    validate_only = "--validate-only" in __import__("sys").argv
                                                                                           
    with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden in offline selftest")), patch.object(socket, "create_connection", side_effect=AssertionError("network forbidden in offline selftest")):
        stages = {"data": data.verify(), "score": score.selftest(), "channels": ch.selftest()}
        if not validate_only:
            json_write(ch.OUT / "score_selftest.json", stages["score"])
            json_write(ch.OUT / "channel_selftest.json", stages["channels"])
            arms.verify_budget()
            arms.dry_run()
        stages["arms"] = arms.verify()
        stages["fixture_pipeline"] = fixture_pipeline()
        stages["runtime_contract"] = runtime_contract()
        stages["protocol_v2"] = protocol_selftest()
        stages["production_projection"] = arms.assert_projection_matches_production()
        result = dict(ok=True, llm_called=False, real_model_calls=0, training_runs=0,
                      passed=sum(s.get("passed", 0) for s in stages.values()),
                      stages={k: {a: b for a, b in v.items() if a != "checks"} for k, v in stages.items()},
                      limits=data.LIMITATIONS)
        if validate_only:
            json_write(ch.OUT / "protocol_selftest_v2.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        json_write(ch.OUT / "arms_selftest.json", stages["arms"])
        arms.author_review()
        score.analyze()
        json_write(ch.OUT / "selftest_summary.json", result)
        json_write(ch.OUT / "fixture_selftest.json", dict(fixture_pipeline=stages["fixture_pipeline"], runtime_contract=stages["runtime_contract"], protocol_v2=stages["protocol_v2"]))
        freeze = arms.freeze()
        assert arms.verify_frozen()["ok"]
        print(json.dumps({**result, "freeze": freeze}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
