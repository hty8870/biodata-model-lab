                       
"""按 trace 快照回退 mutating curate 操作 + CLI。

回退语义（对快照 finalize 出的三清单）：
- created 里仍存在的文件 → **移入回收站**（`.userdata/recycle/<timestamp>_<name>` +
  manifest 行，action="trace_rollback"）——与 curate.remove 同一「移动而非删除」纪律，
  回退删除也绝不真删字节；
- modified / deleted 且有 preimage 字节 → 写回原字节；
- modified / deleted 但**无 preimage → fail-closed 拒动**，如实列入 unrestorable
  （宁可少退不毁数据）；
- 已不在快照所述状态的文件（如已被人手工动过）如实列入 skipped，不强行覆盖。

默认 **dry-run** 只出计划；`--apply` 才动手。执行结果写回快照 meta：每次都记
rollback_attempted_at；只有本次确有恢复且无 unrestorable/errors 才记 rolled_back_at。
未完整回滚保持未完成态，供上层 fail-closed 挡住更早快照，不伪装成「已回过」。

CLI：
  PYTHONPATH=src python -m dataset_recommender.agent.trace.rollback --snapshot <id> [--root .]
  PYTHONPATH=src python -m dataset_recommender.agent.trace.rollback --snapshot <id> --apply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ...app.runtime_paths import instance_data_dir_for
from ...corpus.corpus import EXTERNAL_DIR_NAME, invalidate_external_cache, publish_corpus_write
from ...corpus.corpus_curation import recycle_dir, recycle_manifest
from ...corpus.net_ledger import append_jsonl, now_iso
from ...corpus.uploads import ingest_critical_section
from .recorder import default_project_root
from .snapshot import SnapshotError, SnapshotStore

__all__ = ["plan_rollback", "apply_rollback", "main"]


def _matches_snapshot(path: Path, expected: dict) -> bool:
    """只恢复操作后版本仍未变化的文件；缺哈希/读取失败均拒绝覆盖。"""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest() == expected.get("sha256")
    except OSError:
        return False


def _seal_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """阻塞存在（conflicts 或 unrestorable）时，计划里不留任何执行项——dry-run 与 apply
    同口径：全部 preflight CAS 通过才存在可执行动作。"""
    if plan.get("conflicts") or plan.get("unrestorable"):
        plan["recycle_created"] = []
        plan["restore_bytes"] = []
    return plan


def _plan(root: Path, store: SnapshotStore, meta: dict) -> dict[str, Any]:
    """回退计划（dry-run 与 apply 共用同一真源——apply 不做计划外的事）。

    计划是**全量 preflight CAS**——每个目标都要求「当前状态
    == 本轮写操作之后的 after-state 指纹」（created/modified 用快照记录里的 sha256；
    deleted 的 after-state 是「不存在」）。任一目标不满足 → 进 `conflicts`（机器可判的
    阻塞集；人读原因仍照旧进 `skipped`），apply 在临界区内据此**整体零修改**——绝不覆盖
    同一账户/其他会话在写之后的新编辑，也不「看起来同 owner」就判可删。
    缺 after 指纹（旧格式/不完整快照）→ `missing_after_state`，明确不可信、需人工，
    而不是隐式 overwrite。"""
    if meta.get("mode") == "patch":
        return _plan_patch(root, store, meta)
    diff = meta.get("diff") or {}
    watch = store.watch_dir
    missing = set(meta.get("preimage_missing") or [])
    plan: dict[str, Any] = {
        "snapshot_id": meta.get("snapshot_id"),
        "verb": meta.get("verb"),
        "finalized": bool(meta.get("finalized")),
        "already_rolled_back_at": meta.get("rolled_back_at"),
        "recycle_created": [],               
        "restore_bytes": [],                                            
        "skipped": [],                                             
        "unrestorable": [],                                                     
        "conflicts": [],                                     
    }

    def _block(entry: dict) -> None:
        plan["conflicts"].append(entry)
        plan["skipped"].append(dict(entry))

    for entry in diff.get("created") or []:
        name = entry["name"]
        target = watch / name
        if not str(entry.get("sha256") or ""):
            _block({"name": name, "code": "missing_after_state",
                    "reason": "快照缺写后指纹，无法做 CAS 校验——需人工核对后再处理"})
        elif not target.is_file():
            _block({"name": name, "code": "state_conflict",
                    "reason": "created 文件已不存在（写后状态已变，可能已被移除）"})
        elif not _matches_snapshot(target, entry):
            _block({"name": name, "code": "state_conflict",
                    "reason": "文件与快照写后状态不一致，拒绝移除"})
        else:
            plan["recycle_created"].append(name)
    for entry in diff.get("modified") or []:
        name = entry["name"]
        if name in missing:
            plan["unrestorable"].append({"name": name, "reason": "modified 且无 preimage 字节"})
            continue
        if not str((entry.get("after") or {}).get("sha256") or ""):
            _block({"name": name, "code": "missing_after_state",
                    "reason": "快照缺写后指纹，无法做 CAS 校验——需人工核对后再处理"})
            continue
        target = watch / name
        if not target.is_file():
            _block({"name": name, "code": "state_conflict",
                    "reason": "modified 文件现已不存在（写后状态已变）"})
        elif not _matches_snapshot(target, entry.get("after") or {}):
            _block({"name": name, "code": "state_conflict",
                    "reason": "文件与快照写后状态不一致，拒绝覆盖"})
        else:
            plan["restore_bytes"].append(name)
    for entry in diff.get("deleted") or []:
        name = entry["name"]
        if name in missing:
            plan["unrestorable"].append({"name": name, "reason": "deleted 且无 preimage 字节"})
            continue
        if (watch / name).is_file():
            _block({"name": name, "code": "state_conflict",
                    "reason": "deleted 文件现已存在（写后状态已变，不覆盖）"})
        else:
            plan["restore_bytes"].append(name)
    return _seal_plan(plan)


def _plan_patch(root: Path, store: SnapshotStore, meta: dict) -> dict[str, Any]:
    """补丁模式回退计划：写目标是登录账户的补丁文件，不是共享 external。

    - `created`（写前补丁文件不存在）→ 归档后删除（恢复「不存在」）；
    - `modified`/`deleted` → 用快照 preimage 字节写回（走 patch_package 的锁与原子写）。
    与目录模式同纪律：**全量 preflight CAS**——现状字节必须等于快照记录的
    写后指纹；任一目标不满足（含缺写后指纹）→ `conflicts` 阻塞，apply 整体零修改——
    同一账户后续会话的新写入绝不被旧快照覆盖。"""
    diff = meta.get("diff") or {}
    target = Path(str(meta.get("target") or ""))
    missing = set(meta.get("preimage_missing") or [])
    plan: dict[str, Any] = {
        "snapshot_id": meta.get("snapshot_id"),
        "verb": meta.get("verb"),
        "finalized": bool(meta.get("finalized")),
        "already_rolled_back_at": meta.get("rolled_back_at"),
        "mode": "patch",
        "target": str(target),
        "recycle_created": [],
        "restore_bytes": [],
        "skipped": [],
        "unrestorable": [],
        "conflicts": [],
    }

    def _block(entry: dict) -> None:
        plan["conflicts"].append(entry)
        plan["skipped"].append(dict(entry))

    from ...corpus.patch_package import PatchError, patch_path_for

    receipt = meta.get("patch_receipt")
    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
        _block({"name": target.name, "code": "missing_mutation_receipt",
                "reason": "快照缺真实补丁写入收据，无法认证归属；需人工核对"})
        return _seal_plan(plan)
    try:
        canonical_target = patch_path_for(root, str(meta.get("scope") or "")).resolve()
    except PatchError:
        canonical_target = None
    if (receipt.get("trusted") is not True or canonical_target != target.resolve()
            or receipt.get("scope") != meta.get("scope")
            or Path(str(receipt.get("target") or "")).resolve() != canonical_target):
        _block({"name": target.name, "code": "untrusted_mutation_receipt",
                "reason": "补丁写入收据不完整或多次写入间有其他会话修改，拒绝整文件回退"})
        return _seal_plan(plan)
    entries = [e for key in ("created", "modified", "deleted") for e in diff.get(key) or []]
    if entries:
        before = receipt.get("before")
        after = receipt.get("after")
        valid = (len(entries) == 1 and entries[0].get("name") == target.name
                 and isinstance(receipt.get("writes"), int) and receipt["writes"] > 0)
        for key in ("created", "modified", "deleted"):
            for entry in diff.get(key) or []:
                if key == "modified":
                    valid = valid and entry.get("before") == before and entry.get("after") == after
                else:
                    state = {k: entry[k] for k in ("sha256", "size") if k in entry}
                    valid = valid and ((before is None and state == after) if key == "created"
                                       else (after is None and state == before))
        if not valid:
            _block({"name": target.name, "code": "missing_after_state",
                    "reason": "快照清单与真实写入指纹不一致，拒绝回退"})
            return _seal_plan(plan)
        if before is not None and not _matches_snapshot(store.preimage_path(
                str(meta.get("snapshot_id") or ""), target.name), before):
            _block({"name": target.name, "code": "state_conflict",
                    "reason": "快照前像与真实写入收据不一致，拒绝写回"})
            return _seal_plan(plan)

    for entry in diff.get("created") or []:
        name = entry["name"]
        if not str(entry.get("sha256") or ""):
            _block({"name": name, "code": "missing_after_state",
                    "reason": "快照缺写后指纹，无法做 CAS 校验——需人工核对后再处理"})
        elif not target.is_file():
            _block({"name": name, "code": "state_conflict",
                    "reason": "created 文件已不存在（写后状态已变，可能已被移除）"})
        elif not _matches_snapshot(target, entry):
            _block({"name": name, "code": "state_conflict",
                    "reason": "补丁文件与快照写后状态不一致，拒绝移除"})
        else:
            plan["recycle_created"].append(name)
    for entry in diff.get("modified") or []:
        name = entry["name"]
        if name in missing:
            plan["unrestorable"].append({"name": name, "reason": "modified 且无 preimage 字节"})
            continue
        if not str((entry.get("after") or {}).get("sha256") or ""):
            _block({"name": name, "code": "missing_after_state",
                    "reason": "快照缺写后指纹，无法做 CAS 校验——需人工核对后再处理"})
            continue
        if not target.is_file():
            _block({"name": name, "code": "state_conflict",
                    "reason": "modified 文件现已不存在（写后状态已变）"})
        elif not _matches_snapshot(target, entry.get("after") or {}):
            _block({"name": name, "code": "state_conflict",
                    "reason": "补丁文件与快照写后状态不一致，拒绝覆盖"})
        else:
            plan["restore_bytes"].append(name)
    for entry in diff.get("deleted") or []:
        name = entry["name"]
        if name in missing:
            plan["unrestorable"].append({"name": name, "reason": "deleted 且无 preimage 字节"})
            continue
        if target.is_file():
            _block({"name": name, "code": "state_conflict",
                    "reason": "deleted 文件现已存在（写后状态已变，不覆盖）"})
        else:
            plan["restore_bytes"].append(name)
    return _seal_plan(plan)


def _apply_patch_plan(root: Path, store: SnapshotStore, meta: dict,
                      plan: dict[str, Any], result: dict[str, Any]) -> None:
    """补丁模式回退执行（在 apply_rollback 的既有结果形状内如实记账）。"""
    from ...corpus.patch_package import PatchError, restore_patch_file

    sid = str(meta.get("snapshot_id") or "")
    scope = str(meta.get("scope") or "")
    target = Path(str(meta.get("target") or ""))
    for name in plan["recycle_created"]:
        try:
            data = target.read_bytes() if target.is_file() else b""
            recycle_name = ""
            if data:
                                                   
                rec_dir = recycle_dir(root)
                rec_dir.mkdir(parents=True, exist_ok=True)
                dest = rec_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{name}"
                while dest.exists():
                    dest = rec_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{name}"
                dest.write_bytes(data)
                append_jsonl(recycle_manifest(root), {
                    "ts": now_iso(), "action": "trace_rollback",
                    "original_path": f"patch:{scope}",
                    "recycle_name": dest.name, "snapshot_id": sid,
                })
                recycle_name = dest.name
            restore_patch_file(root, scope, existed=False)
            result["applied"]["recycled"].append({"name": name, "recycle_name": recycle_name})
        except (OSError, PatchError) as exc:
            result["errors"].append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
    for name in plan["restore_bytes"]:
        try:
            data = store.preimage_path(sid, name).read_bytes()
            restore_patch_file(root, scope, existed=True, data=data)
            result["applied"]["restored"].append({"name": name, "bytes": len(data)})
        except (OSError, PatchError) as exc:
            result["errors"].append({"name": name, "error": f"{type(exc).__name__}: {exc}"})


def plan_rollback(project_root: Path, snapshot_id: str) -> dict[str, Any]:
    """dry-run 计划：纯只读，零副作用。"""
    store = SnapshotStore(Path(project_root))
    meta = store.load(snapshot_id)
    if not meta.get("finalized"):
        raise SnapshotError(f"快照 {snapshot_id} 未 finalize（操作后未 diff），无法回退。")
    return _plan(Path(project_root), store, meta)


def _recycle_one(root: Path, name: str, *, snapshot_id: str) -> str:
    """把 external（用户层）里的一个文件移入回收站（与 corpus_curation.apply_remove 同纪律：
    时间戳前缀防覆盖 + manifest 行；action 如实标 trace_rollback）。返回 recycle_name。"""
    target = instance_data_dir_for(root, EXTERNAL_DIR_NAME) / name
    rec_dir = recycle_dir(root)
    rec_dir.mkdir(parents=True, exist_ok=True)
    dest = rec_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{name}"
    while dest.exists():                       
        dest = rec_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{name}"
    shutil.move(str(target), str(dest))
    append_jsonl(recycle_manifest(root), {
        "ts": now_iso(),
        "action": "trace_rollback",
        "original_path": f"{EXTERNAL_DIR_NAME}/{name}",
        "recycle_name": dest.name,
        "snapshot_id": snapshot_id,
    })
    return dest.name


def apply_rollback(project_root: Path, snapshot_id: str) -> dict[str, Any]:
    """真回退：先出计划（与 dry-run 同一函数），再按计划逐项执行；任何单项失败
    如实记入 errors 并继续其余项（部分回退好过零回退，失败项下次可再来）。
    目录模式持摄取锁，补丁模式持账户补丁锁；元信息重检、CAS、恢复与完成标记同一事务。"""
    root = Path(project_root)
    store = SnapshotStore(root)
    routing = store.load(snapshot_id)
    from ...corpus.patch_package import patch_critical_section

    critical_section = (patch_critical_section(root, str(routing.get("scope") or ""))
                        if routing.get("mode") == "patch" else ingest_critical_section(root))
    with critical_section:
        meta = store.load(snapshot_id)
        if (meta.get("mode"), meta.get("scope")) != (routing.get("mode"), routing.get("scope")):
            raise SnapshotError(f"快照 {snapshot_id} 的回退作用域已变化，请重新读取后重试。")
        if not meta.get("finalized"):
            raise SnapshotError(f"快照 {snapshot_id} 未 finalize（操作后未 diff），无法回退。")
        plan = _plan(root, store, meta)
        conflicts = list(plan.get("conflicts") or [])
        result: dict[str, Any] = {
            "snapshot_id": snapshot_id,
            "dry_run": False,
            "applied": {"recycled": [], "restored": []},
            "skipped": plan["skipped"],
            "unrestorable": plan["unrestorable"],
            "errors": [],
            "conflict": bool(conflicts),
            "conflicts": conflicts,
        }
        if conflicts or plan["unrestorable"]:
            pass
        elif plan.get("mode") == "patch":
            _apply_patch_plan(root, store, meta, plan, result)
        else:
            for name in plan["recycle_created"]:
                try:
                    recycle_name = _recycle_one(root, name, snapshot_id=snapshot_id)
                    result["applied"]["recycled"].append({"name": name, "recycle_name": recycle_name})
                except OSError as exc:
                    result["errors"].append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            for name in plan["restore_bytes"]:
                try:
                    data = store.preimage_path(snapshot_id, name).read_bytes()
                    target = instance_data_dir_for(root, EXTERNAL_DIR_NAME) / name
                    from ...corpus import fs_utils
                    try:
                        fs_utils.assert_writable_dir(target, project_root=root)
                    except fs_utils.WritableRootError as exc:
                        result["errors"].append({
                            "name": name, "error": f"WritableRootError: {exc}"})
                        continue
                    target.write_bytes(data)
                    result["applied"]["restored"].append({"name": name, "bytes": len(data)})
                except OSError as exc:
                    result["errors"].append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
        applied_count = (len(result["applied"]["recycled"])
                         + len(result["applied"]["restored"]))
        if applied_count:
            invalidate_external_cache()
            publish_corpus_write(root)
        attempted_at = now_iso()
        meta["rollback_attempted_at"] = attempted_at
        completed_now = (applied_count > 0 and not result["unrestorable"]
                         and not result["errors"] and not result["conflict"])
        if completed_now and not meta.get("rolled_back_at"):
            meta["rolled_back_at"] = attempted_at
        result["rolled_back"] = bool(meta.get("rolled_back_at"))
        meta["rollback_result"] = {
            "recycled": [e["name"] for e in result["applied"]["recycled"]],
            "restored": [e["name"] for e in result["applied"]["restored"]],
            "skipped": result["skipped"],
            "unrestorable": result["unrestorable"],
            "errors": result["errors"],
            "completed": completed_now,
        }
        store.save_meta(snapshot_id, meta)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dataset_recommender.agent.trace.rollback",
        description="按 trace 快照回退 mutating curate 操作（默认 dry-run；--apply 才动手）。")
    parser.add_argument("--root", default=None, help="项目根（缺省自动解析）")
    parser.add_argument("--snapshot", required=True, help="快照 id")
    parser.add_argument("--apply", action="store_true", help="真执行回退（缺省只出计划）")
    args = parser.parse_args(argv)

    root = Path(args.root) if args.root else default_project_root()
    try:
        payload = apply_rollback(root, args.snapshot) if args.apply else plan_rollback(root, args.snapshot)
    except SnapshotError as exc:
        print(f"回退失败：{exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    if args.apply:
        n_err = len(payload.get("errors") or [])
        n_bad = len(payload.get("unrestorable") or [])
        if payload.get("conflict"):
            print("注意：文件状态或写入归属无法通过校验，冲突项未覆盖。", file=sys.stderr)
            return 1
        if n_err or n_bad:
            print(f"注意：{n_err} 项执行失败、{n_bad} 项无 preimage 被拒动（如实上报，未动手）。",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
