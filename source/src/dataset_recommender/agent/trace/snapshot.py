                       
"""mutating curate 操作的**文件级快照存储**。

快照回答「这次写操作动了哪些文件、之前长什么样」——rollback 的锚：
- `capture`：watch 目录（恒 `database/external/`，base 红线结构性不可达）的
  inventory（name → {sha256, size}）+ `preimage_paths` 点名文件的完整字节
  （remove/restore 类动词的目标在 slots 里；create 类动词传空——新文件回退
  不需要 preimage）。
- `finalize`（操作后）：目录模式按收据过滤差分；账户补丁模式只用真实 RMW 的前像与写后指纹，
  capture/finalize 扫描仅作未归因诊断，空收据不认证变化；三清单写回 meta。
- meta.json 与 preimage 字节落 `database/trace/snapshots/<id>/`（.gitignore 已覆盖）。

无 preimage 的 modified/deleted 文件如实记 `preimage_missing`——rollback 对它
fail-closed 拒动（宁可少退不毁数据）。
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from ...app.runtime_paths import instance_data_dir_for
from .recorder import trace_root

__all__ = ["SnapshotStore", "SnapshotError", "snapshot_store"]

                                                            
                                                        
WATCH_DIR_REL = Path("database") / "external"

_SNAPSHOTS_DIR_NAME = "snapshots"


class SnapshotError(ValueError):
    """快照 id 不存在 / meta 损坏 / 文件名非法（含路径分隔符、..）。"""


def _safe_name(name: Any) -> str:
    """external 文件名校验：只接受裸文件名（拒绝分隔符与 ..——防穿越，与
    corpus_curation._leaf_name 同纪律）。"""
    s = str(name or "").strip()
    if not s or "/" in s or "\\" in s or ".." in s:
        raise SnapshotError(f"非法文件名：{s!r}")
    return s


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _inventory(watch_dir: Path) -> dict[str, dict[str, Any]]:
    """目录现状：{name: {sha256, size}}，只收文件（跳过子目录与不可读项——坏项如实缺席，
    不静默当不存在：读不了的文件不列，diff 时就会显成 deleted，逼出真相）。"""
    inv: dict[str, dict[str, Any]] = {}
    if not watch_dir.is_dir():
        return inv
    for p in sorted(watch_dir.iterdir()):
        if not p.is_file():
            continue
        try:
            data = p.read_bytes()
        except OSError:
            continue
        inv[p.name] = {"sha256": _sha256_bytes(data), "size": len(data)}
    return inv


def _diff(before: dict, after: dict) -> dict[str, list]:
    """inventory diff → created/modified/deleted（各元素携带 after/before 侧的哈希与大小）。"""
    created = [{"name": n, **after[n]} for n in sorted(after) if n not in before]
    modified = [{"name": n, "before": before[n], "after": after[n]}
                for n in sorted(after) if n in before and after[n]["sha256"] != before[n]["sha256"]]
    deleted = [{"name": n, **before[n]} for n in sorted(before) if n not in after]
    return {"created": created, "modified": modified, "deleted": deleted}


class SnapshotStore:
    """快照存储（每项目根一个实例即可；方法自带文件锁之外的天然幂等——id 唯一）。"""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.dir = trace_root(self.project_root) / _SNAPSHOTS_DIR_NAME
                                                                  
                                                                    
        self.watch_dir = instance_data_dir_for(self.project_root, "database/external")

    def _snap_dir(self, snapshot_id: str) -> Path:
        sid = _safe_name(snapshot_id)
        d = self.dir / sid
        if not (d / "meta.json").is_file():
            raise SnapshotError(f"快照不存在：{sid}")
        return d

    def capture(self, verb: str, *, preimage_paths: list[str] | tuple = ()) -> str:
        """操作前调用：落 inventory + 点名文件 preimage 字节，返回 snapshot_id。

        绑定补丁作用域（登录账户请求）时，写目标不是共享 external 而是该账户的
        补丁包文件——快照改记 `mode="patch"`（目标路径 + 写前是否存在 + 整文件字节），
        恢复走 patch_package 的锁与原子写（见 rollback）。未绑定 → 历史目录模式逐位不变。"""
        sid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        d = self.dir / sid
        before_dir = d / "before"
        before_dir.mkdir(parents=True, exist_ok=False)
        from ...corpus.patch_package import current_patch_scope, patch_path_for

        scope = current_patch_scope()
        if scope:
            target = patch_path_for(self.project_root, scope)
            try:
                existed = target.is_file()
                preimage = target.read_bytes() if existed else b""
            except OSError:
                existed, preimage = False, b""
            if existed:
                (before_dir / target.name).write_bytes(preimage)
            meta = {
                "snapshot_id": sid,
                "verb": str(verb or ""),
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "mode": "patch",
                "scope": scope,
                "target": str(target).replace("\\", "/"),
                "existed": bool(existed),
                "inventory": {},
                "preimages": [target.name] if existed else [],
                "finalized": False,
            }
            (d / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            return sid
        preimages: list[str] = []
        for name in preimage_paths:
            safe = _safe_name(name)
            src = self.watch_dir / safe
            if not src.is_file():
                raise SnapshotError(f"preimage 目标不存在：{safe}")
            (before_dir / safe).write_bytes(src.read_bytes())
            preimages.append(safe)
        meta = {
            "snapshot_id": sid,
            "verb": str(verb or ""),
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "watch_dir": str(WATCH_DIR_REL).replace("\\", "/"),
            "inventory": _inventory(self.watch_dir),
            "preimages": sorted(preimages),
            "finalized": False,
        }
        (d / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return sid

    def load(self, snapshot_id: str) -> dict[str, Any]:
        d = self._snap_dir(snapshot_id)
        try:
            return json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotError(f"快照 meta 损坏：{snapshot_id}（{exc}）") from exc

    def save_meta(self, snapshot_id: str, meta: dict) -> None:
        d = self._snap_dir(snapshot_id)
        (d / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    def finalize(self, snapshot_id: str, *,
                 written: list[str] | tuple[str, ...] | None = None,
                 mutation: dict | None = None) -> dict[str, list]:
        """操作后调用：diff 出 created/modified/deleted 写回 meta 并返回（diff 结果同时
        供 state_snapshot 事件载荷——同一真源，不两份）。

        目录模式 `mutation={created, modified, deleted}` 按本步工具自报的名字归属，
        其余进 `unattributed_*`；`written` 为 created-only 兼容形态，两者缺省保留目录差分。
        补丁模式必须传 `mutation.patch`（collect_patch_mutations 的真实 RMW 收据列表）；
        空列表代表零写入，缺收据不认证任何变化、回退按不可信快照拒绝。"""
        meta = self.load(snapshot_id)
        if meta.get("mode") == "patch":
            diff = self._finalize_patch(meta, mutation)
            meta["finalized"] = True
            meta["finalized_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            meta["diff"] = diff
            meta["preimage_missing"] = []
            self.save_meta(snapshot_id, meta)
            return diff
        diff = _diff(meta.get("inventory") or {}, _inventory(self.watch_dir))
        if mutation is not None:
            for key in ("created", "modified", "deleted"):
                claimed = {str(n).strip() for n in (mutation.get(key) or [])
                           if str(n or "").strip()}
                diff[f"unattributed_{key}"] = sorted(
                    e["name"] for e in diff[key] if e["name"] not in claimed)
                diff[key] = [e for e in diff[key] if e["name"] in claimed]
            meta["mutation"] = {
                key: sorted(str(n).strip() for n in (mutation.get(key) or [])
                            if str(n or "").strip())
                for key in ("created", "modified", "deleted")}
        elif written is not None:
            mine = {str(n).strip() for n in written if str(n or "").strip()}
            diff["unattributed_created"] = sorted(
                e["name"] for e in diff["created"] if e["name"] not in mine)
            diff["created"] = [e for e in diff["created"] if e["name"] in mine]
            meta["written"] = sorted(mine)
        preimages = set(meta.get("preimages") or [])
        meta["finalized"] = True
        meta["finalized_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta["diff"] = diff
                                                                  
        meta["preimage_missing"] = sorted(
            ({e["name"] for e in diff["modified"]} | {e["name"] for e in diff["deleted"]})
            - preimages)
        self.save_meta(snapshot_id, meta)
        return diff

    def _finalize_patch(self, meta: dict, mutation: dict | None) -> dict[str, list]:
        """补丁归属只来自真实 RMW 收据；capture/finalize 扫描仅用于未归因留痕。
        同一步的多次 RMW 必须首尾相接，间隙有其他写入时整步回退不可信。"""
        target = Path(str(meta.get("target") or ""))
        name = _safe_name(target.name)
        sid = str(meta["snapshot_id"])
        observed = self._patch_diff(meta)
        diff = {key: [] for key in ("created", "modified", "deleted")}
        for key in tuple(diff):
            diff[f"unattributed_{key}"] = sorted(e["name"] for e in observed[key])
        receipts = mutation.get("patch") if isinstance(mutation, dict) else None
        if not isinstance(receipts, list):
            return diff

        def state(blob: bytes | None) -> dict | None:
            return {"sha256": _sha256_bytes(blob), "size": len(blob)} if blob is not None else None

        def inventory(fingerprint: dict | None) -> dict:
            return {name: fingerprint} if fingerprint is not None else {}

        trusted = all(
            isinstance(r, dict) and r.get("trusted") is True
            and r.get("scope") == meta.get("scope")
            and Path(str(r.get("target") or "")).resolve() == target.resolve()
            and "before" in r and (r["before"] is None or isinstance(r["before"], bytes))
            and "after" in r and (r["after"] is None or (
                isinstance(r["after"], dict) and len(str(r["after"].get("sha256") or "")) == 64
                and isinstance(r["after"].get("size"), int)))
            for r in receipts)
        receipt = meta["patch_receipt"] = {
            "schema_version": 1, "scope": meta.get("scope"), "target": str(target),
            "writes": len(receipts), "trusted": trusted,
        }
        if not trusted or not receipts:
            return diff
        for previous, following in zip(receipts, receipts[1:]):
            if previous["after"] != state(following["before"]):
                receipt["trusted"] = False
                return diff
        before_blob = receipts[0]["before"]
        before, after = state(before_blob), receipts[-1]["after"]
        receipt.update(before=before, after=after)
        captured_blob = (self.preimage_path(sid, name).read_bytes()
                         if meta.get("existed") else None)
        try:
            current_blob = target.read_bytes()
        except FileNotFoundError:
            current_blob = None
        gaps = [_diff(inventory(state(captured_blob)), inventory(before)),
                _diff(inventory(after), inventory(state(current_blob)))]
        diff = _diff(inventory(before), inventory(after))
        for key in ("created", "modified", "deleted"):
            diff[f"unattributed_{key}"] = sorted({e["name"] for gap in gaps for e in gap[key]})
        preimage = self.preimage_path(sid, name)
        if before_blob is not None:
            preimage.write_bytes(before_blob)
        else:
            preimage.unlink(missing_ok=True)
        meta["existed"] = before_blob is not None
        meta["preimages"] = [name] if before_blob is not None else []
        return diff

    def _patch_diff(self, meta: dict) -> dict[str, list]:
        """capture 到当前状态的单文件差分，仅用于未归因诊断，不能认证写入归属。"""
        target = Path(str(meta.get("target") or ""))
        name = target.name
        existed = bool(meta.get("existed"))
        sid = str(meta.get("snapshot_id") or "")
        try:
            now_exists = target.is_file()
            after = target.read_bytes() if now_exists else b""
        except OSError:
            now_exists, after = False, b""
        preimage = b""
        if existed:
            try:
                preimage = self.preimage_path(sid, name).read_bytes()
            except OSError:
                preimage = b""
        before = {"sha256": _sha256_bytes(preimage), "size": len(preimage)}
        diff: dict[str, list] = {"created": [], "modified": [], "deleted": []}
        if now_exists and not existed:
            diff["created"] = [{"name": name, "sha256": _sha256_bytes(after),
                                "size": len(after)}]
        elif now_exists and existed:
            if _sha256_bytes(after) != before["sha256"]:
                diff["modified"] = [{"name": name, "before": before,
                                     "after": {"sha256": _sha256_bytes(after),
                                               "size": len(after)}}]
        elif not now_exists and existed:
            diff["deleted"] = [{"name": name, **before}]
        return diff

    def preimage_path(self, snapshot_id: str, name: str) -> Path:
        return self._snap_dir(snapshot_id) / "before" / _safe_name(name)

    def list_snapshots(self) -> list[dict[str, Any]]:
        """全部快照的轻量清单（id/verb/时间/是否已定稿；meta 损坏的如实带 error 键）。"""
        out: list[dict[str, Any]] = []
        if not self.dir.is_dir():
            return out
        for d in sorted(self.dir.iterdir()):
            if not d.is_dir():
                continue
            try:
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                out.append({"snapshot_id": d.name, "verb": meta.get("verb"),
                            "captured_at": meta.get("captured_at"),
                            "finalized": bool(meta.get("finalized")),
                            "rolled_back_at": meta.get("rolled_back_at")})
            except (OSError, json.JSONDecodeError) as exc:
                out.append({"snapshot_id": d.name, "error": type(exc).__name__})
        return out


def snapshot_store(project_root: Path) -> SnapshotStore:
    return SnapshotStore(project_root)
