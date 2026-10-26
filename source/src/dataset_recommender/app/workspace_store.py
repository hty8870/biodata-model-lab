"""Account-owned conversation projects and their membership metadata.

The store does not upload browser history or local memories. Project identifiers
are stable learning scopes; archiving a project retains its learning records.
"""
from __future__ import annotations

import datetime
import os
import re
import unicodedata
import uuid
from contextlib import contextmanager
from pathlib import Path

from .durable_jobs import JobStoreError, open_database
from .runtime_paths import assert_runtime_path, instance_data_dir_for

MAX_PROJECTS = 100
DEFAULT_PROJECT_COLOR = "#0d9488"
_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,80}")
_COLOR = re.compile(r"#[0-9a-fA-F]{6}")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_projects (
    owner TEXT NOT NULL,
    id TEXT NOT NULL,
    name TEXT NOT NULL,
    color TEXT NOT NULL DEFAULT '#0d9488',
    position INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (owner, id)
);
CREATE TABLE IF NOT EXISTS conversations (
    owner TEXT NOT NULL,
    id TEXT NOT NULL,
    project_id TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (owner, id)
);
CREATE INDEX IF NOT EXISTS conversations_project
    ON conversations(owner, project_id, updated_at);
"""


class WorkspaceError(JobStoreError):
    pass


def workspace_path(project_root: Path) -> Path:
    override = os.environ.get("BIODATA_WORKSPACE_DB", "").strip()
    path = Path(override) if override else instance_data_dir_for(project_root, ".userdata") / "workspace.sqlite"
    return assert_runtime_path(path, error_cls=WorkspaceError)


def identifier(value: str, *, allow_empty: bool = False) -> str:
    value = str(value or "")
    if allow_empty and value == "":
        return ""
    if not _IDENTIFIER.fullmatch(value):
        raise WorkspaceError("invalid_identifier", "项目或对话编号无效。")
    return value


def owner_id(value: str) -> str:
    return identifier(value or "anonymous")


def _name(value: str) -> str:
    value = str(value or "").strip()
    if not value or len(value) > 80 or any(unicodedata.category(ch) in {"Cc", "Cf"} for ch in value):
        raise WorkspaceError("invalid_name", "项目名称须为 1–80 个可见字符。")
    return value


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds")


def project_color(value: str) -> str:
    if not isinstance(value, str) or not _COLOR.fullmatch(value):
        raise WorkspaceError("invalid_color", "项目颜色须为六位十六进制颜色，例如 #0d9488。")
    return value.lower()


def _public_project(row) -> dict:
    return {"id": row["id"], "name": row["name"], "color": row["color"], "position": row["position"], "archived": bool(row["archived"]),
            "created_at": row["created_at"], "updated_at": row["updated_at"]}


class WorkspaceStore:
    def __init__(self, project_root: Path, *, db_path: Path | None = None):
        self.path = assert_runtime_path(db_path, error_cls=WorkspaceError) if db_path is not None else workspace_path(project_root)

    @contextmanager
    def database(self, *, write: bool = False):
        with open_database(self.path, schema=_SCHEMA, write=write) as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(conversation_projects)")}
            if not {"color", "position"} <= columns:
                                                                                              
                if not conn.in_transaction:
                    conn.execute("BEGIN IMMEDIATE")
                columns = {row["name"] for row in conn.execute("PRAGMA table_info(conversation_projects)")}
                if "color" not in columns:
                    conn.execute("ALTER TABLE conversation_projects ADD COLUMN color TEXT NOT NULL DEFAULT '#0d9488'")
                if "position" not in columns:
                    conn.execute("ALTER TABLE conversation_projects ADD COLUMN position INTEGER NOT NULL DEFAULT 0")
                    owners = {}
                    for row in conn.execute("SELECT owner,id FROM conversation_projects ORDER BY owner,updated_at DESC,id").fetchall():
                        position = owners.get(row["owner"], 0)
                        conn.execute("UPDATE conversation_projects SET position=? WHERE owner=? AND id=?",
                                     (position, row["owner"], row["id"]))
                        owners[row["owner"]] = position + 1
            yield conn

    @contextmanager
    def joint_database(self, conversations):
        """文件库 + DELETE/FULL 日志让 SQLite super-journal 原子提交两库，拒绝 WAL。"""
        with self.database():
            pass
        with conversations.database() as conn:
            conn.commit()
            schemas = ['main']
            if self.path.resolve() != conversations.path.resolve():
                conn.execute('ATTACH DATABASE ? AS workspace', (str(self.path),))
                schemas.append('workspace')
            for schema in schemas:
                if conn.execute(f'PRAGMA {schema}.journal_mode').fetchone()[0].lower() != 'delete':
                    raise WorkspaceError('workspace_transaction_unavailable', '共同事务需要持久回滚日志，未修改项目归属。')
                conn.execute(f'PRAGMA {schema}.synchronous=FULL')
            conn.execute('BEGIN IMMEDIATE')
            for schema in schemas:
                if conn.execute(f'PRAGMA {schema}.journal_mode').fetchone()[0].lower() != 'delete':
                    raise WorkspaceError('workspace_transaction_unavailable', '共同事务需要持久回滚日志，未修改项目归属。')
            yield conn

    @contextmanager
    def membership_database(self, conn=None):
        if conn is not None:
            yield conn
        else:
            with self.database(write=True) as db:
                yield db

    @staticmethod
    def membership_schema(conn):
        return 'workspace' if any(row['name'] == 'workspace' for row in conn.execute('PRAGMA database_list')) else 'main'

    @staticmethod
    def _project(conn, owner: str, project_id: str, *, active: bool = True):
        schema = WorkspaceStore.membership_schema(conn)
        row = conn.execute(f"SELECT * FROM {schema}.conversation_projects WHERE owner=? AND id=?",
                           (owner, project_id)).fetchone()
        if row is None or (active and row["archived"]):
            raise WorkspaceError("project_not_found", "项目不存在或已归档。")
        return row

    def list_projects(self, owner: str, *, include_archived: bool = False) -> list[dict]:
        owner = owner_id(owner)
        with self.database() as conn:
            rows = conn.execute(
                "SELECT * FROM conversation_projects WHERE owner=? AND (? OR archived=0) ORDER BY position,created_at,id",
                (owner, int(include_archived))).fetchall()
        return [_public_project(row) for row in rows]

    def get_project(self, owner: str, project_id: str, *, active: bool = True) -> dict:
        owner, project_id = owner_id(owner), identifier(project_id)
        with self.database() as conn:
            return _public_project(self._project(conn, owner, project_id, active=active))

    def create_project(self, owner: str, name: str, *, color: str = DEFAULT_PROJECT_COLOR) -> dict:
        owner, name = owner_id(owner), _name(name)
        color = project_color(color)
        project_id, now = "p_" + uuid.uuid4().hex, now_iso()
        with self.database(write=True) as conn:
            count = conn.execute("SELECT COUNT(*) FROM conversation_projects WHERE owner=? AND archived=0",
                                 (owner,)).fetchone()[0]
            if count >= MAX_PROJECTS:
                raise WorkspaceError("project_limit", "项目数量已达上限，请先归档不用的项目。")
            position = conn.execute("SELECT COALESCE(MAX(position),-1)+1 FROM conversation_projects WHERE owner=?", (owner,)).fetchone()[0]
            conn.execute("INSERT INTO conversation_projects(owner,id,name,color,position,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                         (owner, project_id, name, color, position, now, now))
            return _public_project(self._project(conn, owner, project_id))

    def rename_project(self, owner: str, project_id: str, name: str) -> dict:
        return self.update_project(owner, project_id, name=name)

    def update_project(self, owner: str, project_id: str, *, name: str | None = None,
                       color: str | None = None) -> dict:
        owner, project_id = owner_id(owner), identifier(project_id)
        if name is None and color is None:
            raise WorkspaceError("invalid_project_update", "请提供项目名称或颜色。")
        name = _name(name) if name is not None else None
        color = project_color(color) if color is not None else None
        with self.database(write=True) as conn:
            existing = self._project(conn, owner, project_id)
            conn.execute("UPDATE conversation_projects SET name=?,color=?,updated_at=? WHERE owner=? AND id=?",
                         (existing["name"] if name is None else name, existing["color"] if color is None else color,
                          now_iso(), owner, project_id))
            return _public_project(self._project(conn, owner, project_id))

    def archive_project(self, owner: str, project_id: str, *, archived: bool = True) -> dict:
        owner, project_id = owner_id(owner), identifier(project_id)
        with self.database(write=True) as conn:
            existing = self._project(conn, owner, project_id, active=False)
            if not archived and existing["archived"]:
                count = conn.execute("SELECT COUNT(*) FROM conversation_projects WHERE owner=? AND archived=0",
                                     (owner,)).fetchone()[0]
                if count >= MAX_PROJECTS:
                    raise WorkspaceError("project_limit", "项目数量已达上限。")
            now = now_iso()
            conn.execute("UPDATE conversation_projects SET archived=?,updated_at=? WHERE owner=? AND id=?",
                         (int(archived), now, owner, project_id))
                                                                                    
            return _public_project(self._project(conn, owner, project_id, active=False))

    def reorder_projects(self, owner: str, project_ids: list[str]) -> list[dict]:
        owner = owner_id(owner)
        ordered = [identifier(value) for value in project_ids]
        if len(ordered) != len(set(ordered)):
            raise WorkspaceError("invalid_project_order", "项目排序中不能有重复项目。")
        with self.database(write=True) as conn:
            rows = conn.execute("SELECT id,archived FROM conversation_projects WHERE owner=? ORDER BY position,created_at,id", (owner,)).fetchall()
            active = {row["id"] for row in rows if not row["archived"]}
            if set(ordered) != active:
                raise WorkspaceError("project_order_changed", "项目列表已更新，请刷新后再排序。")
            iterator = iter(ordered)
            now = now_iso()
            for position, row in enumerate(rows):
                project_id = row["id"] if row["archived"] else next(iterator)
                conn.execute("UPDATE conversation_projects SET position=?,updated_at=? WHERE owner=? AND id=?",
                             (position, now, owner, project_id))
        return self.list_projects(owner, include_archived=True)

    def assignments(self, owner: str) -> dict[str, str]:
        owner = owner_id(owner)
        with self.database() as conn:
            rows = conn.execute("SELECT id,project_id FROM conversations WHERE owner=?", (owner,)).fetchall()
        return {row["id"]: row["project_id"] for row in rows}

    def forget_conversation(self, owner: str, conversation_id: str, *, conn=None):
        with self.membership_database(conn) as db:
            schema = self.membership_schema(db)
            db.execute(f'DELETE FROM {schema}.conversations WHERE owner=? AND id=?', (owner_id(owner), identifier(conversation_id)))

    def assign_conversation(self, owner: str, conversation_id: str, project_id: str = "", *, title: str = "", conn=None) -> dict:
        owner = owner_id(owner)
        conversation_id, project_id = identifier(conversation_id), identifier(project_id, allow_empty=True)
        with self.membership_database(conn) as db:
            schema = self.membership_schema(db)
            if project_id:
                self._project(db, owner, project_id)
            now = now_iso()
            db.execute(
                f"""INSERT INTO {schema}.conversations(owner,id,project_id,title,updated_at) VALUES(?,?,?,?,?)
                ON CONFLICT(owner,id) DO UPDATE SET project_id=excluded.project_id,updated_at=excluded.updated_at""",
                (owner, conversation_id, project_id, str(title or "")[:180], now))
        return {"conv_id": conversation_id, "project_id": project_id, "updated_at": now}

    def resolve_conversation(self, owner: str, conversation_id: str, requested_project: str | None = None,
                             *, title: str = "", conn=None) -> str:
        """Capture a validated project for a turn without silently moving an existing conversation."""
        owner = owner_id(owner)
        requested = None if requested_project is None else identifier(requested_project, allow_empty=True)
        with self.membership_database(conn) as db:
            schema = self.membership_schema(db)
            if not conversation_id:
                project_id = requested or ''
                if project_id:
                    self._project(db, owner, project_id)
                return project_id
            conversation_id = identifier(conversation_id)
            row = db.execute(f"SELECT project_id FROM {schema}.conversations WHERE owner=? AND id=?",
                             (owner, conversation_id)).fetchone()
            project_id = row["project_id"] if row is not None else (requested or "")
            if row is not None and requested is not None and requested != project_id:
                raise WorkspaceError("conversation_project_changed", "对话的项目归属已变更，请刷新后继续。")
            if project_id:
                self._project(db, owner, project_id)
            now = now_iso()
            db.execute(
                f"""INSERT INTO {schema}.conversations(owner,id,project_id,title,updated_at) VALUES(?,?,?,?,?)
                ON CONFLICT(owner,id) DO UPDATE SET title=CASE WHEN conversations.title='' THEN excluded.title ELSE conversations.title END,
                updated_at=excluded.updated_at""",
                (owner, conversation_id, project_id, str(title or "")[:180], now))
        return project_id

    def list_conversations(self, owner: str, *, project_id: str | None = None, limit: int = 200) -> list[dict]:
        owner = owner_id(owner)
        project_id = None if project_id is None else identifier(project_id, allow_empty=True)
        with self.database() as conn:
            rows = conn.execute(
                "SELECT id,project_id,title,updated_at FROM conversations WHERE owner=? AND (? IS NULL OR project_id=?) ORDER BY updated_at DESC LIMIT ?",
                (owner, project_id, project_id, max(1, min(int(limit), 500)))).fetchall()
        return [dict(row) for row in rows]
