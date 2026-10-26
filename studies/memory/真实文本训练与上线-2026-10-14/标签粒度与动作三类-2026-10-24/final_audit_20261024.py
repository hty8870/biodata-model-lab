from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from verify_action_experiment_20261024 import manual

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def canonical(row):
    value = row["formal"]
    transformed = []
    for side in ("old", "new"):
        transformed.append(tuple(sorted((value["scopes"][int(index) // 4], value["feature_ids"][int(index) % 4], choice)
                                        for index, choice in value[side].items())))
    return tuple(sorted(transformed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    original = read(ROOT / "original-fit-v2/summary.json")
    assert all(sha(args.repo_root / name) == expected for name, expected in original["source_sha256"].items())
    source206 = read(ROOT.parent / "206题可视化审计-2026-10-24/source-manifest-2026-10-24.json")
    assert all(sha(args.repo_root / item["logical_path"]) == item["sha256"] for item in source206["files"])
    design = read(ROOT / "design-lock.json")
    assert all(sha(ROOT / name) == expected for name, expected in design["inputs_sha256"].items())
    seen = defaultdict(list)
    for split in ("train", "dev", "heldout"):
        for line in (ROOT / "data-v1" / (split + ".jsonl")).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            seen[canonical(row)].append(split)
    cross = [value for value in seen.values() if len(set(value)) > 1]
    assert len(seen) == 1199 and cross == [["dev", "heldout"]]
    rows = [json.loads(line) for line in (ROOT / "confirmation-v2/rows.jsonl").read_text(encoding="utf-8").splitlines()]
    new_keys = {canonical(row) for row in rows}
    assert len(rows) == len(new_keys) == 300 and not new_keys.intersection(seen)
    lock = read(ROOT / "confirmation-v2/lock.json")
    assert lock["script_sha256"] == sha(ROOT / "confirmation_audit_20261024.py")
    checked = {}
    for item in lock["models"]:
        assert sha(ROOT / item["directory"] / "selected.pt") == item["checkpoint_sha256"]
        data = read(ROOT / "confirmation-v2" / (item["name"] + ".json"))
        computed = manual(rows, data["predictions"])
        for key, value in computed.items():
            if key != "matrix":
                assert np.isclose(value, data["summary"][key], atol=1e-10, rtol=0)
        assert computed["matrix"] == data["summary"]["confusion_matrix"]["matrix"]
        checked[item["name"]] = computed
    papers = read(ROOT / "papers/sources-2026-10-24.json")
    assert all(sha(ROOT / "papers" / (item["name"] + ".pdf")) == item["sha256"] for item in papers)
    scripts = list(ROOT.glob("*.py"))
    for path in scripts:
        ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    report = ROOT / "研究报告-标签粒度与动作判断-2026-10-24.md"
    content = report.read_text(encoding="utf-8")
    assert not report.read_bytes().startswith(b"\xef\xbb\xbf")
    anchors = {re.sub(r"[^\w -]", "", line.lstrip("# ").strip()).lower().replace(" ", "-") for line in content.splitlines() if line.startswith("#")}
    links = re.findall(r"\]\(([^)]+)\)", content)
    for link in links:
        if link.startswith("#"):
            assert link[1:] in anchors, link
        elif not link.startswith(("https://", "http://")):
            assert (report.parent / unquote(link.split("#", 1)[0])).is_file(), link
    result = {"passed": True, "original_model_and_source_unchanged": True, "old206_source_unchanged": True,
              "locked_experiment_inputs_unchanged": True, "known_initial_logic_duplicate_pairs": len(cross),
              "fresh_confirmation_canonical_pairs": 300, "no_overlap_with_old1200": True,
              "six_confirmation_weights_and_manual_scores_checked": checked,
              "papers_checked": len(papers), "python_AST_checked": len(scripts), "report_links_checked": len(links),
              "report_sha256": sha(report), "report_bytes": report.stat().st_size,
              "known_limitations": ["initial validation has one logical dev overlap", "program labels, not human gold", "five outer topics and fixed seeds", "R6 execution chosen posthoc"],
              "production_not_changed": True}
    (ROOT / "final-audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "six_confirmation_weights_and_manual_scores_checked"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
