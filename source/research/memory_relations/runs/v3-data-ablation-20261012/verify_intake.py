"""正式派发与现成launcher真实preflight对照；不读取audit/终测或改源data树。"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
PREFLIGHT = BASE.parent / "v3-training-preflight-20261012"


def main():
    manifest = json.loads((BASE / "launch-manifest.json").read_text(encoding="utf-8"))
    dispatch = json.loads((PREFLIGHT / "formal-dispatch-20261012.json").read_text(encoding="utf-8"))
    if manifest["input_sha256"] != dispatch["development_sha256"]:
        raise AssertionError("实际开发文件SHA与父派发不符")
    contracts = manifest["training_input_contract"]
    for arm, contract in contracts.items():
        if contract["audit_visible"] is not False or contract["unit"] != "semantic_root_not_descendant_rows":
            raise AssertionError("输入非隔离candidate/root单位")
        if [contract["row_counts"]["train"], len(contract["root_ids_by_partition"]["train"])] != dispatch["train_rows_roots"][arm]:
            raise AssertionError("实际train行/根数不符")
        for role, digest in dispatch["common_partition_sha256"].items():
            from hashlib import sha256
            raw_path = Path(dispatch["frozen_root"]) / arm / f"{role}.jsonl"
            if sha256(raw_path.read_bytes()).hexdigest() != digest:
                raise AssertionError("派发的共享分区原文件SHA不符")
            if len({value["sha256_by_partition"][role] for value in contracts.values()}) != 1:
                raise AssertionError("自身按id排序canonical行契约在三臂不共享")
            if [contract["row_counts"][role], len(contract["root_ids_by_partition"][role])] != dispatch["common_rows_roots"][role]:
                raise AssertionError("共享开发分区行/根数不符")
    result = {"status": "passed", "source": "actual launcher prepare_data/data_contract vs parent independently verified dispatch",
              "development_sha256": manifest["input_sha256"], "common_partition_sha256": dispatch["common_partition_sha256"],
              "train_rows_roots": dispatch["train_rows_roots"], "common_rows_roots": dispatch["common_rows_roots"],
              "parent_readiness_actor": "714", "missing_training_ready_manifest_not_blocking": True,
              "audit_or_terminal_read": False, "human_gold": False}
    with (BASE / "intake-verification.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
