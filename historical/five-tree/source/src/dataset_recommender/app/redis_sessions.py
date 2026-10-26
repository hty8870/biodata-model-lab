                       
"""Redis 会话后端 —— `accounts` 会话的可选共享存储（默认仍走内存 + JSON 快照）。

为什么存在：默认形态是**单机单进程**（会话存进程内存 + `.userdata/sessions.json` 快照），
多进程 / 多实例部署时会话互不可见。本后端把会话放进 Redis，供 opt-in 形态复用；语义与
`accounts` 现有文件后端**逐条对齐**，避免「换个后端行为悄悄变了」。

存储形状：
- key `biodata:sess:sha256:<digest>`；value 紧凑 JSON `{"user_id","username","expires_at"}`
  （`separators=(",",":")`，与文件快照的紧凑风格一致）。摘要不是 cookie 凭据。
- 新会话 TTL 用 `SET ... EX` 交给 Redis 自己过期；旧明文 key 经 WATCH 事务迁移，
  同时监听新旧键以防撤销竞态，用 PEXPIREAT 保留 TTL 与 expires_at 中较早的到期时间。
  并发迁移冲突限次重读，不把另一请求完成的迁移误判为登出。

fail 语义（与文件后端一致：会话是**可再生**的，丢了 = 重新登录，不丢账户数据）：
- `create` 失败 → 抛 `RedisUnavailable("session_store_unavailable")`（登录必须成功或如实报错，
  不能发一个永远解析不出来的 token）；缺包等更精确的 code 原样透传。
- `resolve` 失败返回 None；过期/畸形条目的清理尽力执行。
- `destroy` 失败抛 RedisUnavailable，绝不把未撤销的会话报告为登出成功。
- 畸形 / 过期条目绝不交给调用方：删掉该 key 再返回 None（对齐文件后端 `_load_sessions`
  丢弃畸形条目的口径）。

本模块只在函数体内惰性引用 accounts 的摘要算法，避免模块加载环；
`RedisUnavailable` 由 `accounts` 翻成 `AccountError`。
"""

from __future__ import annotations

import json
import secrets
import time
from typing import Any, Callable

from . import redis_store

                                                              
_KEY_PREFIX = "biodata:sess:"


class RedisSessionStore:
    """Redis 会话后端。构造只存参数，不建连（建连推迟到首次真实命令，见 `redis_store`）。"""

    def __init__(
        self,
        *,
        ttl_seconds: int,
        token_bytes: int,
        client_factory: "Callable[[], Any] | None" = None,
    ) -> None:
                                                                   
        self._ttl_seconds = int(ttl_seconds)
        self._token_bytes = int(token_bytes)
                                              
        self._client_factory = client_factory

    def _client(self) -> "Any":
        if self._client_factory is not None:
            return self._client_factory()
        return redis_store.get_client()

    @staticmethod
    def _key(token: str) -> str:
        from .accounts import session_token_digest
        return f"{_KEY_PREFIX}sha256:{session_token_digest(token)}"

    def create(self, user: Any) -> str:
        """写入一条会话并返回 token；失败抛 `RedisUnavailable`（登录侧翻成 AccountError）。"""
        token = secrets.token_urlsafe(self._token_bytes)
        payload = {
            "user_id": user.id,
            "username": user.username,
            "expires_at": time.time() + self._ttl_seconds,
        }
        try:
            self._client().set(
                self._key(token),
                json.dumps(payload, separators=(",", ":")),
                ex=self._ttl_seconds,
            )
        except redis_store.redis_error_types() as exc:
            raise redis_store.RedisUnavailable(
                "session_store_unavailable",
                "会话存储暂时不可用，请稍后重试。",
            ) from exc
        return token

    def resolve(self, token: "str | None") -> "dict[str, Any] | None":
        """解析 token → `{"user_id","username","expires_at"}` 或 None。

        Redis 命令失败 → None（fail-open）；条目畸形 / 已过期 → 删 key 后 None。
        """
        if not token or token.startswith("sha256:"):
            return None
        key = self._key(token)
        legacy_key = f"{_KEY_PREFIX}{token}"
        try:
            client = self._client()
            raw = client.get(key)
            if raw is None:
                raw = self._migrate_legacy(client, key, legacy_key)
        except (redis_store.RedisUnavailable, *redis_store.redis_error_types()):
            return None
        if raw is None:
            return None
        sess = self._decode(raw)
        if sess is None or sess["expires_at"] <= time.time():
            self._delete_key(key)
            return None
        return sess

    def _migrate_legacy(self, client: Any, key: str, legacy_key: str) -> Any:
        from redis import WatchError

        for _ in range(3):
            try:
                with client.pipeline() as pipe:
                    pipe.watch(key, legacy_key)
                    current = pipe.get(key)
                    if current is not None:
                        return current
                    raw = pipe.get(legacy_key)
                    if raw is None:
                        return None
                    sess = self._decode(raw)
                    if sess is None or sess["expires_at"] <= time.time():
                        pipe.multi()
                        pipe.delete(legacy_key)
                        pipe.execute()
                        return None
                    started_ms = int(time.time() * 1000)
                    ttl_ms = pipe.pttl(legacy_key)
                    if ttl_ms == -2:
                        return None
                    expires_ms = int(sess["expires_at"] * 1000)
                    if ttl_ms >= 0:
                        expires_ms = min(expires_ms, started_ms + ttl_ms)
                    if expires_ms <= started_ms:
                        return None
                    pipe.multi()
                    pipe.set(key, raw)
                    pipe.pexpireat(key, expires_ms)
                    pipe.delete(legacy_key)
                    pipe.execute()
                    return raw
            except WatchError:
                continue
        return None

    def destroy(self, token: "str | None") -> None:
        """删除会话；存储失败如实抛 RedisUnavailable，由 HTTP 层返回可重试错误。"""
        if not token or token.startswith("sha256:"):
            return
        self._delete_key(self._key(token), f"{_KEY_PREFIX}{token}", required=True)

    def _delete_key(self, key: str, *other_keys: str, required: bool = False) -> None:
        try:
            self._client().delete(key, *other_keys)
        except (redis_store.RedisUnavailable, *redis_store.redis_error_types()) as exc:
            if required:
                if isinstance(exc, redis_store.RedisUnavailable):
                    raise
                raise redis_store.RedisUnavailable(
                    "session_store_unavailable", "会话撤销暂时失败，请稍后重试退出登录。") from exc

    @staticmethod
    def _decode(raw: Any) -> "dict[str, Any] | None":
        """把 Redis 值解成受校验的会话 dict；任何畸形（非 JSON / 结构不符）→ None。

        校验口径与 `accounts._load_sessions` 一致：user_id / username 必须是 str，
        expires_at 必须是数值——不满足就当畸形条目丢弃，绝不把半截数据交给调用方。
        """
        try:
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            data = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
            return None
        if not (
            isinstance(data, dict)
            and isinstance(data.get("user_id"), str)
            and isinstance(data.get("username"), str)
            and isinstance(data.get("expires_at"), (int, float))
        ):
            return None
        return {
            "user_id": data["user_id"],
            "username": data["username"],
            "expires_at": float(data["expires_at"]),
        }
