from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2]
PARENT_README = HERE.parent / "README.md"
TOTAL_README = RESEARCH / "README.md"
REPORT = HERE / "结论-Qwen第二seed与未暴露家族复验-2026-10-24.md"
EXTERNAL = RESEARCH / "00-汇报与面谈"
RECEIPT = EXTERNAL / "Qwen第二seed与未暴露家族复验-Codex审查回执-2026-10-24.md"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_targets(text):
    return [match.group(1).split("#", 1)[0] for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text)
            if match.group(1) and not match.group(1).startswith(("http://", "https://", "#"))]


def missing_links(text, exists):
    return [target for target in markdown_targets(text) if not exists(target)]


def selftest():
    assert markdown_targets("[ok](ok.md) [anchor](#section) [web](https://example.invalid)") == ["ok.md"]
    assert missing_links("[ok](ok.md)", lambda target: target == "ok.md") == []
    assert missing_links("[bad](missing.md)", lambda target: target == "ok.md") == ["missing.md"]
    return {"known_good_link": True, "known_bad_link": True}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    tests = selftest()
    required = [REPORT, RECEIPT, HERE / "results" / "replication-decision.json",
                HERE / "runs" / "qwen3-lora-seed42" / "verification.json",
                HERE / "baseline" / "qwen3-lora-seed13-best-dev" / "predictions.json"]
    assert all(path.is_file() for path in required)
    link_problems = {}
    for path in (REPORT, RECEIPT):
        text = path.read_text(encoding="utf-8")
        missing = missing_links(text, lambda target: (path.parent / target).exists())
        if missing:
            link_problems[str(path)] = missing
    assert not link_problems, link_problems
    parent_link = "Qwen第二seed与未暴露家族复验-2026-10-24/结论-Qwen第二seed与未暴露家族复验-2026-10-24.md"
    total_link = "记忆关系轻量模型-2026-10-11/真实文本训练与上线-2026-10-14/" + parent_link
    assert (PARENT_README.parent / parent_link).is_file()
    assert (TOTAL_README.parent / total_link).is_file()
    assert parent_link in PARENT_README.read_text(encoding="utf-8")
    assert total_link in TOTAL_README.read_text(encoding="utf-8")

    json_files = sorted(HERE.rglob("*.json"))
    jsonl_files = sorted(HERE.rglob("*.jsonl"))
    for path in json_files:
        read_json(path)
    jsonl_rows = {}
    for path in jsonl_files:
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    json.loads(line)
                    count += 1
        jsonl_rows[path.relative_to(HERE).as_posix()] = count
    assert jsonl_rows["data-unexposed-v1/rows.jsonl"] == 120
    assert jsonl_rows["data-unexposed-v1/inputs.jsonl"] == 120
    assert jsonl_rows["runs/qwen3-lora-seed42/updates.jsonl"] == 300

    csv_counts = {}
    for name in ("seed-comparison.csv", "family-comparison.csv", "shared-errors.csv", "seed42-errors.csv"):
        with (HERE / "results" / name).open(encoding="utf-8", newline="") as stream:
            csv_counts[name] = sum(1 for _ in csv.DictReader(stream))
    assert csv_counts == {"seed-comparison.csv": 2, "family-comparison.csv": 20,
                          "shared-errors.csv": 8, "seed42-errors.csv": 16}

    result = read_json(HERE / "runs" / "qwen3-lora-seed42" / "result.json")
    checkpoint_files = 0
    for key in ("terminal", "best_dev"):
        checkpoint = result[key]["checkpoint"]
        for name, item in checkpoint["files"].items():
            path = Path(checkpoint["directory"]) / name
            assert path.is_file() and sha256(path) == item["sha256"], f"checkpoint hash mismatch: {path}"
            checkpoint_files += 1
    verification = read_json(HERE / "runs" / "qwen3-lora-seed42" / "verification.json")
    decision = read_json(HERE / "results" / "replication-decision.json")
    assert verification["passed"] is True and verification["problems"] == []
    assert decision["decision"]["status"] == "positive_replication" and decision["no_deploy"] is True
    assert result["no_deploy"] is True
    assert decision["analysis_selftest"] == {"known_answer_metric_pass": True, "known_answer_metric_fail": True,
                                             "positive_boundary": True, "three_stop_branches": True, "gray_branch": True}
    for tag in ("terminal/train", "terminal/dev", "best-dev/train", "best-dev/dev",
                "best-dev/diagnostic-120", "best-dev/confirmation-300"):
        assert verification["checks"]["predictions"][tag]["probability_max_abs_difference"] == 0.0

    report_text = REPORT.read_text(encoding="utf-8")
    assert "收尾时补记" not in report_text and "待填" not in report_text
    secret_pattern = re.compile(r"oc_sk_[A-Za-z0-9_-]+|sk-[A-Za-z0-9]{16,}")
    secret_files = [str(path.relative_to(HERE)) for path in (*json_files, *jsonl_files, REPORT)
                    if secret_pattern.search(path.read_text(encoding="utf-8"))]
    assert not secret_files, secret_files
    payload = {"passed": True, "analysis_selftest": tests,
               "report_links_checked": len(markdown_targets(report_text)),
               "receipt_links_checked": len(markdown_targets(RECEIPT.read_text(encoding="utf-8"))),
               "index_links_checked": 2, "json_files_parsed": len(json_files),
               "jsonl_files_parsed": len(jsonl_files), "jsonl_rows": jsonl_rows,
               "csv_counts": csv_counts, "checkpoint_files_hashed": checkpoint_files,
               "verification_passed": True, "probability_max_abs_difference_six_groups": 0.0,
               "decision": decision["decision"]["status"], "no_deploy": True,
               "secret_pattern_matches": 0, "report_sha256": sha256(REPORT),
               "decision_sha256": sha256(HERE / "results" / "replication-decision.json"),
               "script_sha256": sha256(Path(__file__))}
    (HERE / "results" / "delivery-validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
