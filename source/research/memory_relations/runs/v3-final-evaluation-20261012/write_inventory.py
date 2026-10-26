"""仅列本scope文件路径/类型；不读fixture内容，不跟随reparse目标。"""
import ast
import json
import stat

import binding as B

B.verify()
entries = []


def walk(directory):
    for path in sorted(directory.iterdir()):
        info = path.lstat()
        relative = path.relative_to(B.BASE).as_posix()
        if getattr(info, "st_file_attributes", 0) & 0x400 or stat.S_ISLNK(info.st_mode):
            entries.append({"path": relative, "type": "reparse_not_followed"})
        elif stat.S_ISDIR(info.st_mode):
            walk(path)
        else:
            entries.append({"path": relative, "type": "file", "bytes": info.st_size})


for path in B.BASE.glob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
walk(B.BASE)
B.save_json(B.BASE / "reports/touched-files-inventory.json", {
    "at": B.now(), "scope": str(B.BASE), "entries": entries,
    "outside_scope_modified": ["协同/认领/kimi-mr3-final-eval-1012.md", "开发日志归档/开发日志.md"],
    "new_python_ast_files": len(list(B.BASE.glob("*.py"))),
    "model_prediction_calls": 0, "fixture_contents_read": False,
    "reparse_targets_followed": False, "final_artifacts_sha256_manifest": "final-output-manifest.json"})
print(json.dumps({"inventory": "reports/touched-files-inventory.json", "entries": len(entries),
                  "reparse_not_followed": sum(r["type"] == "reparse_not_followed" for r in entries),
                  "new_python_ast_files": len(list(B.BASE.glob("*.py"))), "model_prediction_calls": 0}))
