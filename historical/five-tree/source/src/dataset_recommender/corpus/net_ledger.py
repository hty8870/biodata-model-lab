                       
"""联网/审计 JSONL 账本的**路径、时间戳、加锁追加单一真源**。

真源史：写原语原住 `corpus_net`、经 `corpus_curation` 转发 seam 复用，并被 agent/app
三个包按私有名跨包取用（14 处）；2026-10-01 仓库整理批收敛为本模块公共名——
`corpus_net`/`corpus_curation` 保留私有别名兼容既有内部引用，新消费一律用这里的公共名。

并发账本写必须互斥：sync def 端点走线程池，Windows 上 open("a") 的 seek-to-EOF+write
跨并发句柄非原子，裸写会整行覆盖丢行/撕裂（对抗评审 R2-3/R2-8 实测 20 线程丢 7-13%、
2 线程也丢）。一把进程内锁兜住线程池并发；跨进程（Web↔MCP 双实例）残余风险已知悉
（审计面，不挡主功能）。
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from ..app.runtime_paths import instance_data_dir_for

USERDATA_DIR_NAME = ".userdata"
NET_LEDGER_NAME = "curate_net_ledger.jsonl"


def net_ledger_path(project_root: Path) -> Path:
    """联网账本路径：写盘侧经 runtime_paths 解析用户层（frozen = data_root/.userdata；
    source/portable 与测试注入根 = project_root/.userdata，历史逐字节一致）。"""
    return instance_data_dir_for(Path(project_root), USERDATA_DIR_NAME) / NET_LEDGER_NAME


def now_iso() -> str:
    """本地时区 ISO 秒级时间戳（账本行 ts 字段的统一形态）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


_ledger_lock = threading.Lock()


def append_jsonl(path: Path, entry: dict) -> None:
    """追加一行 JSONL（进程内锁互斥；目录不存在自动创建）。"""
    with _ledger_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


                                                                                                    
                                     
                                                                                                    


def current_ledger_account() -> str:
    """当前账本主体：绑定补丁作用域的登录账户 id；未绑定 = ""（本机匿名 / CLI / MCP /
    冻结评测——历史行为逐位不变）。

    作用域解析复用 `patch_package.current_patch_scope`（Web 中间件与 agent turn 均已绑定），
    不新造主体通道。"""
    from .patch_package import current_patch_scope                            

    return str(current_patch_scope() or "")


def stamp_ledger_account(entry: dict) -> dict:
    """账本行主体打标（写侧唯一锚点）：显式 account_id 优先（后台任务可点名），
    否则取当前作用域，未绑定记 ""。返回新行，不改入参。"""
    row = dict(entry)
    row["account_id"] = str(entry.get("account_id") or current_ledger_account())
    return row


def append_net_ledger(path: Path, entry: dict) -> None:
    """**联网/审计账本专用**追加：写入时钉住主体——读侧（状态摘要）按同一主体过滤，
    账户 B 看不到 A 的最近查询与错误文本。回收站 manifest / 其它台账仍用 `append_jsonl`
    （它们的读取面不经本过滤，打标只会污染形状）。"""
    append_jsonl(path, stamp_ledger_account(entry))


def ledger_visible(entry: dict, account: "str | None" = None) -> bool:
    """账本行读侧可见性（唯一判定）：行主体 == 当前主体（`account` 显式给定时用它）。
    历史行（无 account_id 键）按匿名 "" 计——不暴露给任何登录账户；
    匿名读侧（本机未登录）看到的就是匿名行，行为与历史一致。"""
    owner = str(entry.get("account_id") or "")
    scope = current_ledger_account() if account is None else str(account or "")
    return owner == scope
