from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[5] / 'source'
DATA = ROOT / "data"
sys.path.insert(0, str(REPO / "src"))

from dataset_recommender.app.conversation_store import ConversationStore              
from dataset_recommender.app.knowledge_service import KnowledgeService              
from dataset_recommender.llm.llm_client import LLMConfig              

OWNER = "user-a"
CFG = LLMConfig(enable_llm=True, api_key="synthetic-contract-key",
                base_url="https://model.example/v1", model="contract-model")
ENDPOINT = hashlib.sha256(f"{CFG.base_url}|{CFG.model}".encode()).hexdigest()[:12]


def body(text: str, **updates) -> dict:
    value = {"kind": "memory", "title": "原始标题", "text": text,
             "trigger": "汇报时", "tier": "explicit", "permanence": "persistent",
             "payload": {}, "links": []}
    value.update(updates)
    return value


def decision(request: dict) -> dict:
    refs = [request["evidence"][0]["id"]]
    result = {"semantic_pairs": [{"old_id": row["id"], "relation": "overlap_incomparable",
                "reason": "离线合同响应。", "old_quote": row["text"][:100],
                "new_quote": request["candidate"]["text"][:100],
                "evidence_refs": refs, "error_code": None}
               for row in request["existing"]],
              "edit_checks": [], "evidence_refs": refs, "error_code": None}
    for edit in request["context"]["edits"]:
        after = edit["after"][0]
        result["edit_checks"].append({"entry_id": edit["entry_id"], "supported": True,
            "reason": "离线编辑响应。", "before_refs": [{"field": "text", "quote": edit["before"]["text"][:100]}],
            "after_refs": [{"entry_id": after["entry_id"], "field": "text", "quote": after["body"]["text"][:100]}],
            "source_refs": [{"id": refs[0], "quote": request["evidence"][0]["text"][:100]}],
            "preserved": [], "error_code": None})
    return result


class Env:
    def __init__(self, root: Path):
        self.service = KnowledgeService(ConversationStore(root))

    def propose(self, operations, text: str, request_id: str):
        receipt = self.service.propose(OWNER, ENDPOINT, request_id=request_id,
            operations=operations, source_refs=[{"type": "user", "text": text}])
        if not receipt.get("ok"):
            raise AssertionError(receipt)
        return receipt

    def assess(self, receipt):
        result = self.service.assess(OWNER, ENDPOINT, change_id=receipt["change_id"],
            expected_change_revision=receipt["change_revision"], cfg=CFG, predictor=decision)
        if not result.get("ok"):
            raise AssertionError(result)
        return result

    def commit(self, receipt):
        result = self.service.commit(OWNER, ENDPOINT, change_id=receipt["change_id"],
            expected_change_revision=receipt["change_revision"], assessment_id=receipt["assessment_id"],
            expected_catalog_generation=receipt["catalog_generation"], human_confirmed=True)
        if not result.get("ok"):
            raise AssertionError(result)
        return result

    def activate(self, value: dict, text: str, request_id: str) -> str:
        receipt = self.propose([{"op": "create", "after": value}], text, request_id)
        self.commit(self.assess(receipt))
        return self.service.change(OWNER, ENDPOINT, receipt["change_id"])["operations"][0]["entry_id"]

    def generation(self) -> int:
        with self.service.store.database() as conn:
            return self.service.store.generation(conn, OWNER, ENDPOINT)


def update_defaults(root: Path) -> dict:
    env = Env(root)
    old = env.activate(body("旧正文", title="保留标题", trigger="仅在草稿"), "旧正文", "old")
    receipt = env.propose([{"op": "update", "entry_id": old, "expected_entry_revision": 1,
        "after": {"kind": "memory", "text": "新正文", "permanence": "persistent"}}],
        "旧正文\n新正文", "update-defaults")
    env.commit(env.assess(receipt))
    entry = env.service.entry(OWNER, ENDPOINT, old)
    observed = {key: entry[key] for key in ("title", "trigger", "tier", "permanence", "payload", "links")}
    return {"observed": observed,
            "current_defaults_reach_active_entry": observed == {"title": "", "trigger": "", "tier": "explicit",
                "permanence": "persistent", "payload": {}, "links": []},
            "safe_partial_update_guaranteed": observed["title"] == "保留标题" and observed["trigger"] == "仅在草稿"}


def provenance_overwrite(root: Path) -> dict:
    env = Env(root)
    old = env.activate(body("汇报用中文；公式用 LaTeX。"), "旧来源：汇报用中文；公式用 LaTeX。", "source-a")
    receipt = env.propose([{"op": "update", "entry_id": old, "expected_entry_revision": 1,
        "after": body("汇报用英文；公式用 LaTeX。")}],
        "汇报用中文；公式用 LaTeX。\n新来源：仅本次把汇报改成英文，公式仍用 LaTeX。", "source-b")
    env.commit(env.assess(receipt))
    entry = env.service.entry(OWNER, ENDPOINT, old)
    sources = entry["provenance"]["sources"]
    return {"source_texts": [row["text"] for row in sources],
            "old_source_retained_in_active_entry": any("旧来源" in row["text"] for row in sources),
            "note": "当前 commit 以最新 change 来源覆盖现役 entry provenance；不是完整历史链。"}


def atom_source_mapping(root: Path) -> dict:
    env = Env(root)
    operations = [{"op": "create", "after": body(text)} for text in ("汇报用中文。", "公式用 LaTeX。")]
    receipt = env.propose(operations, "汇报用中文。\n公式用 LaTeX。", "group-sources")
    env.commit(env.assess(receipt))
    entries = env.service.entries(OWNER, ENDPOINT)["items"]
    source_sets = {tuple(row["id"] for row in entry["provenance"]["sources"]) for entry in entries}
    quote_sets = {tuple(entry["provenance"]["quotes"]) for entry in entries}
    return {"entries": len(entries), "distinct_source_sets": len(source_sets),
            "distinct_quote_sets": len(quote_sets),
            "per_atom_unique_mapping_proven": len(source_sets) == len(entries) == len(quote_sets),
            "note": "同 change 原子共享最新 sources/quotes；精确 atom↔evidence 需要另行保留。"}


def group_and_replay(root: Path) -> dict:
    env = Env(root)
    operations = [{"op": "create", "after": body(f"第{index}项义务。")} for index in range(4)]
    first = env.propose(operations, "四项义务。", "group-commit")
    replay = env.service.propose(OWNER, ENDPOINT, request_id="group-commit",
        operations=operations, source_refs=[{"type": "user", "text": "四项义务。"}])
    before = env.generation()
    env.commit(env.assess(first))
    after = env.generation()
    conflict = env.service.propose(OWNER, ENDPOINT, request_id="group-commit",
        operations=[{"op": "create", "after": body("不同内容。")}],
        source_refs=[{"type": "user", "text": "不同内容。"}])
    return {"same_change_on_replay": replay["change_id"] == first["change_id"],
            "replay_created": replay.get("created"), "generation_delta": after - before,
            "active_entries": len(env.service.entries(OWNER, ENDPOINT)["items"]),
            "fingerprint_conflict": conflict.get("error_code")}


def sequential_stale(root: Path) -> dict:
    env = Env(root)
    first = env.propose([{"op": "create", "after": body("第一项。")},], "第一项。", "first")
    second = env.propose([{"op": "create", "after": body("第二项。")},], "第二项。", "second")
    first_assessed = env.assess(first)
    second_assessed = env.assess(second)
    env.commit(first_assessed)
    rejected = env.service.commit(OWNER, ENDPOINT, change_id=second["change_id"],
        expected_change_revision=second_assessed["change_revision"], assessment_id=second_assessed["assessment_id"],
        expected_catalog_generation=second_assessed["catalog_generation"], human_confirmed=True)
    return {"second_commit_error": rejected.get("error_code"),
            "batch_commit_required_to_avoid_stale": rejected.get("error_code") == "knowledge_assessment_stale"}


def retract_and_move(root: Path) -> dict:
    env = Env(root)
    project_a = env.service.workspace.create_project(OWNER, "项目A")["id"]
    project_b = env.service.workspace.create_project(OWNER, "项目B")["id"]
    value = body("项目规则。")
    receipt = env.service.propose(OWNER, ENDPOINT, project_a, request_id="project-rule",
        operations=[{"op": "create", "after": value}], source_refs=[{"type": "user", "text": "项目规则。", "project_id": project_a}])
    assessed = env.service.assess(OWNER, ENDPOINT, project_a, change_id=receipt["change_id"],
        expected_change_revision=receipt["change_revision"], cfg=CFG, predictor=decision)
    committed = env.service.commit(OWNER, ENDPOINT, project_a, change_id=assessed["change_id"],
        expected_change_revision=assessed["change_revision"], assessment_id=assessed["assessment_id"],
        expected_catalog_generation=assessed["catalog_generation"], human_confirmed=True)
    if not committed.get("ok"):
        raise AssertionError(committed)
    entry_id = env.service.change(OWNER, ENDPOINT, receipt["change_id"])["operations"][0]["entry_id"]
    moved_receipt = env.service.propose(OWNER, ENDPOINT, project_a, request_id="move-rule",
        operations=[{"op": "move", "entry_id": entry_id, "expected_entry_revision": 1,
                     "destination": {"scope_level": "project", "project_id": project_b}}],
        source_refs=[{"type": "user", "text": "项目规则。", "project_id": project_a}])
    moved_assessed = env.service.assess(OWNER, ENDPOINT, project_a, change_id=moved_receipt["change_id"],
        expected_change_revision=moved_receipt["change_revision"], cfg=CFG, predictor=decision)
    if not moved_assessed.get("ok"):
        raise AssertionError(moved_assessed)
    moved = env.service.commit(OWNER, ENDPOINT, project_a, change_id=moved_assessed["change_id"],
        expected_change_revision=moved_assessed["change_revision"], assessment_id=moved_assessed["assessment_id"],
        expected_catalog_generation=moved_assessed["catalog_generation"], human_confirmed=True)
    if not moved.get("ok"):
        raise AssertionError(moved)
    moved_entry = env.service.entry(OWNER, ENDPOINT, entry_id)
    retract_receipt = env.service.propose(OWNER, ENDPOINT, project_b, request_id="retract-rule",
        operations=[{"op": "retract", "entry_id": entry_id, "expected_entry_revision": moved_entry["revision"]}],
        source_refs=[])
    retract_assessed = env.service.assess(OWNER, ENDPOINT, project_b, change_id=retract_receipt["change_id"],
        expected_change_revision=retract_receipt["change_revision"], cfg=CFG, predictor=decision)
    if not retract_assessed.get("ok"):
        raise AssertionError(retract_assessed)
    retracted = env.service.commit(OWNER, ENDPOINT, project_b, change_id=retract_assessed["change_id"],
        expected_change_revision=retract_assessed["change_revision"], assessment_id=retract_assessed["assessment_id"],
        expected_catalog_generation=retract_assessed["catalog_generation"], human_confirmed=True)
    final = env.service.entry(OWNER, ENDPOINT, entry_id)
    return {"move_committed": moved.get("ok"), "moved_project": moved_entry["project_id"],
            "retract_committed": retracted.get("ok"), "final_status": final["status"]}


def run_all() -> dict:
    tests = {}
    with tempfile.TemporaryDirectory() as tmp:
        tests["update_defaults"] = update_defaults(Path(tmp) / "update")
    with tempfile.TemporaryDirectory() as tmp:
        tests["provenance_overwrite"] = provenance_overwrite(Path(tmp) / "provenance")
    with tempfile.TemporaryDirectory() as tmp:
        tests["atom_source_mapping"] = atom_source_mapping(Path(tmp) / "atom-sources")
    with tempfile.TemporaryDirectory() as tmp:
        tests["group_and_replay"] = group_and_replay(Path(tmp) / "group")
    with tempfile.TemporaryDirectory() as tmp:
        tests["sequential_stale"] = sequential_stale(Path(tmp) / "stale")
    with tempfile.TemporaryDirectory() as tmp:
        tests["retract_and_move"] = retract_and_move(Path(tmp) / "move")
    observations = {
        "update_defaults_detected": tests["update_defaults"]["current_defaults_reach_active_entry"] is True
            and tests["update_defaults"]["safe_partial_update_guaranteed"] is False,
        "old_provenance_overwrite_detected": tests["provenance_overwrite"]["old_source_retained_in_active_entry"] is False,
        "shared_atom_source_detected": tests["atom_source_mapping"]["per_atom_unique_mapping_proven"] is False,
        "group_once_and_replay": tests["group_and_replay"]["same_change_on_replay"] is True
            and tests["group_and_replay"]["generation_delta"] == 1
            and tests["group_and_replay"]["active_entries"] == 4
            and tests["group_and_replay"]["fingerprint_conflict"] == "knowledge_request_conflict",
        "sequential_stale_detected": tests["sequential_stale"]["batch_commit_required_to_avoid_stale"] is True,
        "move_retract_contract": tests["retract_and_move"]["moved_project"] != ""
            and tests["retract_and_move"]["final_status"] == "retracted"}
    return {"schema": "atomic-contract-current-behavior-v1", "synthetic_only": True,
            "human_confirmation_is_fixture_not_real_approval": True,
            "observations": observations, "all_current_behavior_checks_passed": all(observations.values()),
            "candidate_readiness": {
                "safe_partial_update": tests["update_defaults"]["safe_partial_update_guaranteed"],
                "complete_provenance_history": tests["provenance_overwrite"]["old_source_retained_in_active_entry"],
                "per_atom_source_mapping": tests["atom_source_mapping"]["per_atom_unique_mapping_proven"]},
            "tests": tests,
            "source_sha256": {
                "knowledge.py": hashlib.sha256((REPO / "src/dataset_recommender/agent/knowledge.py").read_bytes()).hexdigest(),
                "knowledge_service.py": hashlib.sha256((REPO / "src/dataset_recommender/app/knowledge_service.py").read_bytes()).hexdigest(),
                "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "run"))
    args = parser.parse_args()
    result = run_all()
    print(json.dumps({"checks": result["observations"], "candidate_readiness": result["candidate_readiness"]},
                     ensure_ascii=False))
    if not result["all_current_behavior_checks_passed"]:
        raise SystemExit(2)
    if args.command == "run":
        DATA.mkdir(exist_ok=True)
        target = DATA / "contract-results.json"
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
