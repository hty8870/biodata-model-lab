                       
"""联网工具组：免 key 通用搜索（主力）+ 官方源轻量适配器（对照），统一出口形态。

为什么单列一个模块而不是塞进 corpus_curation.py：
  - corpus_curation.py 是「管护真源」（plan/apply/回收站/token），本模块是「纯联网取数」——
    一律返回 `{ok, items, note_zh?, error?}` 字典、**绝不抛异常炸链**（网络失败/页面结构变了都如实降级），
    两种失败语义不同，分开各守各的纪律；
  - corpus_curation 单向 import 本模块（check_updates 的 10x/ENCODE/HCA/GEO/Zenodo/refine.bio 在线比对走这里）；
    本模块**不得**反向 import corpus_curation（循环 import），故账本路径/时间小助手在此保留一份
    极小复刻（3 个函数、同一账本文件约定），注释在此说明而非隐式重复。

纪律（与 corpus_curation 的联网纪律同口径）：
  - 每个 fetch 都经 `fetch_text_logged` / `fetch_json_logged` 包装：限速 + 追加
    `.userdata/curate_net_ledger.jsonl`（ts/endpoint/query/HTTP 状态/条数，不记秘密）；
  - DuckDuckGo 更严：串行 + 请求间隔 ≥1s、超时 ≤12s（免费 HTML 端点，打重了会被 202/验证码墙）；
  - 429/503 与瞬时连接错误指数退避 ≤3 次，其余 4xx 不重试；不引任何需要付费 key 的服务；
  - 解析只用标准库（re + html），**不引新依赖**；页面结构变了 → 如实降级（ok=False + error），不炸链。

GEO 已接入（2026-08-07，推翻 v1 的「不接」裁决——产品已立项，配方见
《数据源API调研-2026-08-08.md》§4）：走 NCBI E-utilities 官方端点（esearch/esummary，免 key）。
当年三条顾虑的处置：① 「需 email 礼貌声明」——本仓库无对外联系邮箱，按官方建议只带
`tool=biodata_agent` 参数，不编造 email；② 「返回结构深」——两段式封装进
`_geo_esearch_ids` / `_geo_summary_items` 形状闸，响应漂移即如实降级（parse_changed），不硬解析；
③ 「series/study 两层映射价值低」——只取 Series 级（"GSE"[Entry Type] 枚举），与本地
geo.json 快照同口径，check_updates 差分与关键词/物种检索都够用。

GEO 备用通道（2026-08-07 降级施工）：本机到 NCBI 持续不可达时，search_geo /
geo_recent_items 按「NCBI 主通道 → E-GEOD 镜像（BioStudies，≤2016 老数据）→
Europe PMC 文献弱兜底」按序降级，全败如实报 all_channels_failed；每次降级的
note_zh / channel 字段如实写明实际通道（含 E-GEOD 年代局限），绝不假装主通道数据。
"""
from __future__ import annotations

import html as html_lib
import http.client
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "ORGANISM_COMMON",
    "SOURCE_ALIASES",
    "resolve_source_key",
    "search_online_source",
    "search_duckduckgo",
    "search_encode",
    "search_10x",
    "search_hca",
    "search_geo",
    "search_zenodo",
    "search_arrayexpress_items",
    "encode_recent_items",
    "tenx_dataset_items",
    "hca_recent_items",
    "geo_recent_items",
    "zenodo_recent_items",
    "search_refinebio",
    "refinebio_recent_items",
    "fetch_text_logged",
    "fetch_json_logged",
    "parse_ddg_html",
]

                                                                                                
                                  
                                                                                                

                                                     
                                                     
                                    
ORGANISM_COMMON: dict[str, str] = {
    "homo sapiens": "Human",
    "mus musculus": "Mouse",
    "rattus norvegicus": "Rat",
    "danio rerio": "Zebrafish",
    "drosophila melanogaster": "Drosophila",
    "macaca mulatta": "Macaque",
    "macaca fascicularis": "Macaque",
    "callithrix jacchus": "Marmoset",
    "pan troglodytes": "Chimpanzee",
    "gallus gallus": "Chicken",
    "sus scrofa": "Pig",
    "canis lupus familiaris": "Dog",
    "canis familiaris": "Dog",
    "oryctolagus cuniculus": "Rabbit",
    "bos taurus": "Cattle",
    "homo sapien": "Human",                                           
}


                                                                                                
                                                           
                                                      
                                                                                                

from .net_ledger import append_jsonl as _append_jsonl
from .net_ledger import append_net_ledger as _append_net_ledger
from .net_ledger import net_ledger_path as _net_ledger_path
from .net_ledger import now_iso as _now_iso


                                                                                                
                                                
                                                                                                

class _NetError(Exception):
    """模块内部联网失败载具；公开函数统一 catch 成 {ok: False, error}，绝不漏出。"""


_DDG_TIMEOUT = 12                            
_DDG_MIN_INTERVAL = 1.0                          
_DEFAULT_TIMEOUT = 20
_DEFAULT_MIN_INTERVAL = 0.2                                               
_RETRIES = 3
                                                                
                                                    
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_RETRYABLE_HTTP = {429, 503}
_last_request_by_host: dict[str, float] = {}
                                                                             
                                                            
                                                           
                           
_rate_limit_lock = threading.Lock()


def _polite_wait(host: str, min_interval: float) -> None:
    """按 host 限速（不同源各记各的间隔，互不挤占）。持锁做 sleep 判定，并发下间隔同样成立。"""
    with _rate_limit_lock:
                                                              
                                                         
        deadline = _last_request_by_host.get(host, 0.0) + min_interval
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(remaining)
        _last_request_by_host[host] = time.monotonic()


def _raw_request(url: str, *, timeout: int, min_interval: float, headers: dict[str, str],
                 method: str = "GET", body: bytes | None = None,
                 attempts: list[int] | None = None,
                 opener: Any = None) -> tuple[bytes, int]:
    """HTTP request → (body, status), with the shared rate/retry/size policy.

    GET and idempotent read-only POST calls use this same primitive. 429/503 and
    transient transport failures retry at most three times; deterministic body
    and parsing failures do not retry.

    attempts：可选出参（单元素列表），回填实际请求次数——重试发生在函数内部，
    不带回尝试数的话账本只记最终一条，「刚才为什么卡了几秒/是不是多打了对方两次」无从回答。"""
    host = urllib.parse.urlparse(url).netloc
    merged_headers = {"User-Agent": "biodata-agent-curate/1.0"}
    merged_headers.update(headers or {})
    last_exc: Exception | None = None
    for attempt in range(_RETRIES):
        if attempts is not None:
            attempts[0] = attempt + 1
        _polite_wait(host, min_interval)
        try:
            req = urllib.request.Request(url, data=body, headers=merged_headers,
                                         method=str(method or "GET").upper())
            open_fn = opener or urllib.request.urlopen
            with open_fn(req, timeout=timeout) as resp:
                data = resp.read(_MAX_RESPONSE_BYTES + 1)
                if len(data) > _MAX_RESPONSE_BYTES:
                                                                             
                                                                                        
                    raise _NetError(f"响应体超过 {_MAX_RESPONSE_BYTES} 字节上限（{url}）——对端异常，已停止读取、不重试。")
                return data, int(getattr(resp, "status", 200) or 200)
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in _RETRYABLE_HTTP and attempt < _RETRIES - 1:
                time.sleep(1.0 * (2 ** attempt))
                continue
            raise _NetError(f"HTTP {exc.code}（{url}）") from exc
        except (urllib.error.URLError, TimeoutError, OSError, http.client.IncompleteRead) as exc:
                                                                        
                                                          
                                                   
            last_exc = exc
            if attempt < _RETRIES - 1:
                time.sleep(1.0 * (2 ** attempt))
                continue
            raise _NetError(f"{type(exc).__name__}: {exc}（{url}）") from exc
    raise _NetError(str(last_exc))


def _raw_get(url: str, *, timeout: int, min_interval: float, headers: dict[str, str],
             attempts: list[int] | None = None) -> tuple[bytes, int]:
    """Compatibility wrapper for existing GET-only adapters."""
    return _raw_request(url, timeout=timeout, min_interval=min_interval,
                        headers=headers, attempts=attempts)


def fetch_text(
    url: str,
    *,
    timeout: int = _DEFAULT_TIMEOUT,
    min_interval: float = _DEFAULT_MIN_INTERVAL,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    body: bytes | None = None,
    attempts: list[int] | None = None,
    opener: Any = None,
) -> tuple[str, int]:
    """GET → (文本, status)。HTML 页面抓取入口。attempts：重试计数出参（透传 _raw_get）。"""
    raw, status = _raw_request(url, timeout=timeout, min_interval=min_interval,
                               headers=headers or {}, method=method, body=body,
                               attempts=attempts, opener=opener)
    return raw.decode("utf-8", errors="replace"), status


def fetch_json(
    url: str,
    *,
    timeout: int = _DEFAULT_TIMEOUT,
    min_interval: float = _DEFAULT_MIN_INTERVAL,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    body: bytes | None = None,
    attempts: list[int] | None = None,
    opener: Any = None,
) -> tuple[Any, int]:
    """GET → (解析后的 JSON, status)。

    JSON 解析失败是对端改了返回形状，属确定性失败——不再当瞬时错误
    退避重试（白打两次），直接抛 _NetError 如实说明；网络层的 429/503/瞬时错误重试仍由
    _raw_get 负责，次数经 attempts 出参带回给账本。"""
    text, status = fetch_text(
        url, timeout=timeout, min_interval=min_interval, headers=headers,
        method=method, body=body, attempts=attempts, opener=opener,
    )
    try:
        return json.loads(text), status
    except ValueError as exc:
        raise _NetError(f"响应不是合法 JSON（{exc}）。这是对端响应形状问题，不是瞬时抖动，没有重试（{url}）") from exc


def _count_payload(payload: Any) -> int:
    """账本条数口径：搜索响应取结果列表长度；其它非空响应记 1。"""
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("hits", "@graph", "results"):                                      
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
            if key == "hits" and isinstance(value, dict) and isinstance(value.get("hits"), list):
                return len(value["hits"])
    return 1 if payload else 0


def fetch_text_logged(
    url: str,
    *,
    project_root: Path,
    endpoint: str,
    query: str,
    timeout: int = _DEFAULT_TIMEOUT,
    min_interval: float = _DEFAULT_MIN_INTERVAL,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    body: bytes | None = None,
    opener: Any = None,
) -> str:
    """fetch_text 的账本包装：每次联网（含失败）追加一行 curate_net_ledger.jsonl。
    发生重试时条目带 attempts 计数（形状只增不减）。"""
    entry: dict[str, Any] = {"ts": _now_iso(), "endpoint": endpoint, "query": query,
                             "method": str(method or "GET").upper()}
    tries = [1]
    try:
        text, status = fetch_text(url, timeout=timeout, min_interval=min_interval,
                                  headers=headers, method=method, body=body,
                                  attempts=tries, opener=opener)
    except _NetError as exc:
        entry.update({"http_status": None, "records": 0, "error": str(exc)})
        if tries[0] > 1:
            entry["attempts"] = tries[0]
        _append_net_ledger(_net_ledger_path(project_root), entry)
        raise
    entry.update({"http_status": status, "records": 1 if text else 0})
    if tries[0] > 1:
        entry["attempts"] = tries[0]
    _append_net_ledger(_net_ledger_path(project_root), entry)
    return text


def fetch_json_logged(
    url: str,
    *,
    project_root: Path,
    endpoint: str,
    query: str,
    timeout: int = _DEFAULT_TIMEOUT,
    min_interval: float = _DEFAULT_MIN_INTERVAL,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    body: bytes | None = None,
    opener: Any = None,
    fetcher: Any = None,
    ledger_append: Any = None,
    ledger_path: Any = None,
) -> Any:
    """fetch_json 的账本包装（条数取 hits/@graph 长度）。
    发生重试时条目带 attempts 计数（形状只增不减）。"""
    entry: dict[str, Any] = {"ts": _now_iso(), "endpoint": endpoint, "query": query,
                             "method": str(method or "GET").upper()}
    tries = [1]
    append = ledger_append or _append_net_ledger
    path = (ledger_path or _net_ledger_path)(project_root)
    try:
        if fetcher is None:
            payload, status = fetch_json(url, timeout=timeout, min_interval=min_interval,
                                         headers=headers, method=method, body=body,
                                         attempts=tries, opener=opener)
        else:
            payload, status = fetcher(tries)
    except Exception as exc:
        entry.update({"http_status": None, "records": 0, "error": str(exc)})
        if tries[0] > 1:
            entry["attempts"] = tries[0]
        append(path, entry)
        raise
    entry.update({"http_status": status, "records": _count_payload(payload)})
    if tries[0] > 1:
        entry["attempts"] = tries[0]
    append(path, entry)
    return payload


def _fail(error: str, note_zh: str, **extra: Any) -> dict:
    """统一失败形态：ok=False + 机器码 + 中文说明（调用方/前端不需要猜异常类型）。"""
    return {"ok": False, "items": [], "error": error, "note_zh": note_zh, **extra}


                                                                                                
                                                          
                                                                                                

DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
DDG_SOURCE_KEYS = ("ddg", "duckduckgo", "web", "generic")

                               
                                                                                              
                                                                            
                                                         
_DDG_LINK_RES = (
    re.compile(r'class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.S),
    re.compile(r'href="(?P<href>[^"]+)"[^>]*class="result__a"[^>]*>(?P<title>.*?)</a>', re.S),
)
                                                 
_DDG_SNIPPET_RE = re.compile(
    r'<(?P<tag>a|div)[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</(?P=tag)>', re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(fragment: str) -> str:
                                                   
                                    
    return html_lib.unescape(_TAG_RE.sub("", html_lib.unescape(fragment))).strip()


def _ddg_real_url(href: str) -> str:
    """DDG 结果链接多是 /l/?uddg=<urlencoded> 跳转壳，剥壳取真链接；剥不动就原样返回。"""
    href = html_lib.unescape(href)
    if "uddg=" in href:
        qs = urllib.parse.urlparse(href).query or href.split("uddg=", 1)[1]
        uddg = urllib.parse.parse_qs(qs).get("uddg") if "uddg=" in qs else None
        if uddg is None:                          
            uddg = [href.split("uddg=", 1)[1].split("&", 1)[0]]
        if uddg and uddg[0]:
            return urllib.parse.unquote(uddg[0])
    if href.startswith("//"):
        return "https:" + href
    return href


def parse_ddg_html(page: str, *, limit: int = 10) -> list[dict]:
    """DDG HTML 结果页 → [{accession, title, url, snippet}]。解析失败宁可少给，不编造。"""
    links: list[tuple[str, str]] = []
    for rx in _DDG_LINK_RES:
        links = [(m.group("href"), m.group("title")) for m in rx.finditer(page)]
        if links:
            break
    snippets = [m.group("snippet") for m in _DDG_SNIPPET_RE.finditer(page)]
    items: list[dict] = []
    seen: set[str] = set()
    for i, (href, title_frag) in enumerate(links):
        url = _ddg_real_url(href)
        title = _strip_tags(title_frag)
        if not url or not title or url in seen:
            continue
        seen.add(url)
        snippet = _strip_tags(snippets[i]) if i < len(snippets) else ""
        items.append({"accession": "", "title": title, "url": url, "snippet": snippet})
        if len(items) >= limit:
            break
    return items


def search_duckduckgo(
    keywords: str,
    *,
    species: str | None = None,
    limit: int = 10,
    project_root: Path,
) -> dict:
    """通用搜索主力：DDG HTML 端点免 key 抓取（标题/链接/摘要），限速 ≥1s、超时 ≤12s、记账本。"""
    query = " ".join(x for x in [str(keywords or "").strip(), str(species or "").strip()] if x)
    if not query:
        return _fail("empty_query", "通用搜索关键词为空，未发请求。")
    url = f"{DDG_HTML_ENDPOINT}?q={urllib.parse.quote(query)}"
    try:
        page = fetch_text_logged(
            url, project_root=Path(project_root), endpoint=DDG_HTML_ENDPOINT, query=query,
            timeout=_DDG_TIMEOUT, min_interval=_DDG_MIN_INTERVAL,
        )
    except _NetError as exc:
        return _fail("network_error", f"DuckDuckGo 请求失败（{exc}）。可稍后重试，或改用官方来源搜索。")
    items = parse_ddg_html(page, limit=limit)
    if not items:
                                                               
                                                              
                                                                                            
                                                                              
                                                
        if "anomaly.js" in page and "result__a" not in page:
            return _fail(
                "blocked",
                f"DuckDuckGo 触发了人机验证，本机暂时用不了通用搜索"
                f"（查询「{query}」）——这不是关键词没命中。可以改用官方来源搜索，或换个网络环境再试。",
            )
        return _fail(
            "no_results",
            f"DuckDuckGo 对 {query!r} 没有解析出结果（可能是搜索页面改版，也可能是真没有结果）。",
        )
    return {"ok": True, "items": items, "note_zh": f"DuckDuckGo 通用搜索返回 {len(items)} 条。"}


                                                                                                
                                                            
                                                   
                                                      
                                                       
                                                                                                

                                                                              
                                                          
AE_SEARCH_API = "https://www.ebi.ac.uk/biostudies/api/v1/arrayexpress/search"
AE_STUDY_TMPL = "https://www.ebi.ac.uk/biostudies/arrayexpress/studies/{accession}"

ENCODE_SEARCH_API = "https://www.encodeproject.org/search/"
ENCODE_BASE = "https://www.encodeproject.org"
_ENCODE_HEADERS = {"Accept": "application/json"}

                                                 
TENX_DATASETS_URL = "https://www.10xgenomics.com/datasets"

                                                              
                                                         
AZUL_PROJECTS_API = "https://service.azul.data.humancellatlas.org/index/projects"
HCA_STUDY_TMPL = "https://data.humancellatlas.org/explore/projects/{project_id}"


def search_arrayexpress_items(
    keywords: str,
    *,
    species: str | None = None,
    limit: int = 20,
    project_root: Path,
) -> dict:
    """ArrayExpress（BioStudies）轻量 items 搜索：一段式（不带详情富化），供在线比对/清点用。"""
    query = str(keywords or "").strip()
    if not query:
        return _fail("empty_query", "ArrayExpress 搜索关键词为空，未发请求。")
    url = f"{AE_SEARCH_API}?query={urllib.parse.quote(query)}&pageSize={int(limit)}"
    try:
        payload = fetch_json_logged(url, project_root=Path(project_root), endpoint=AE_SEARCH_API, query=query)
    except _NetError as exc:
        return _fail("network_error", f"ArrayExpress 官方 API 请求失败（{exc}）。")
    hits = payload.get("hits") if isinstance(payload, dict) else None
    items: list[dict] = []
    for h in (hits or [])[: int(limit)]:
        if not isinstance(h, dict):
            continue
        acc = str(h.get("accession") or "").strip()
        title = str(h.get("title") or "").strip()
        if not acc or not title:
            continue
        items.append({
            "accession": acc,
            "title": title,
            "url": AE_STUDY_TMPL.format(accession=acc),
            "date": str(h.get("release_date") or "").strip(),
            "snippet": str(h.get("content") or "")[:200],
        })
    sp = str(species or "").strip().lower()
    if sp:                                             
        items = [it for it in items if sp in (it["title"] + " " + it.get("snippet", "")).lower()]
    if not items:
        return _fail("no_results", f"ArrayExpress 查询 {query!r} 没有可用条目。")
    return {"ok": True, "items": items, "note_zh": f"ArrayExpress 官方 API 返回 {len(items)} 条。"}


def _encode_graph_items(payload: Any, *, limit: int) -> list[dict]:
    """ENCODE /search/?format=json 的 @graph → items。缺 accession 的条目丢弃（无法核对）。"""
    graph = payload.get("@graph") if isinstance(payload, dict) else None
    items: list[dict] = []
    for node in (graph or [])[:limit]:
        if not isinstance(node, dict):
            continue
        acc = str(node.get("accession") or "").strip()
        if not acc:
            continue
        title = (str(node.get("description") or "").strip()
                 or str(node.get("assay_title") or "").strip()
                 or acc)
        path = str(node.get("@id") or "").strip()
        url = (ENCODE_BASE + path) if path.startswith("/") else (path or f"{ENCODE_BASE}/experiments/{acc}/")
        date = str(node.get("date_created") or node.get("date_released") or "").strip()[:10]
        lab = ""
        if isinstance(node.get("lab"), dict):
            lab = str(node["lab"].get("title") or "").strip()
        items.append({
            "accession": acc,
            "title": title,
            "url": url,
            "date": date,
            "snippet": " · ".join(x for x in [str(node.get("assay_title") or "").strip(), lab] if x),
        })
    return items


def search_encode(
    keywords: str,
    *,
    species: str | None = None,
    limit: int = 20,
    project_root: Path,
) -> dict:
    """ENCODE REST 适配器：/search/?type=Experiment&format=json（Accept: application/json）。

    species 拼进 searchTerm（ENCODE 全文检索会命中 biosample 的 organism 文本），不做本地再过滤。"""
    terms = [str(keywords or "").strip()]
    if str(species or "").strip():
        terms.append(str(species).strip())
    query = " ".join(t for t in terms if t)
    if not query:
        return _fail("empty_query", "ENCODE 搜索关键词为空，未发请求。")
    url = (f"{ENCODE_SEARCH_API}?type=Experiment&searchTerm={urllib.parse.quote(query)}"
           f"&format=json&limit={int(limit)}")
    try:
        payload = fetch_json_logged(
            url, project_root=Path(project_root), endpoint=ENCODE_SEARCH_API, query=query,
            headers=_ENCODE_HEADERS,
        )
    except _NetError as exc:
                                                                                    
        if str(exc).startswith("HTTP 404"):
            return _fail("no_results", f"ENCODE 查询 {query!r} 没有命中条目。")
        return _fail("network_error", f"ENCODE 官方 API 请求失败（{exc}）。")
    items = _encode_graph_items(payload, limit=int(limit))
    if not items:
        return _fail("no_results", f"ENCODE 查询 {query!r} 没有可用条目。")
    return {"ok": True, "items": items, "note_zh": f"ENCODE 官方 API 返回 {len(items)} 条。"}


                                                                    
                                                                         
                                                     
                                                                    
                                                         
                                                                    
                                                                   
                                                          
TENX_SEARCH_API = "https://www.10xgenomics.com/api/search"
TENX_BASE = "https://www.10xgenomics.com"
_TENX_PAGE_LIMIT = 1000                                 
_TENX_MAX_PAGES = 5                                    

                                                                       
                                                     
_TENX_SPECIES_TAG = {"human": "Human", "mouse": "Mouse"}


def _tenx_api_date(value: object) -> str:
    """publishedAt（Unix 秒，字符串）→ UTC 日历日期；非数字/越界 → ""（不猜）。"""
    try:
        return datetime.fromtimestamp(int(str(value).strip()), tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OverflowError, OSError):
        return ""


def _tenx_api_items(payload: Any) -> list[dict] | None:
    """10x 私有 API 响应 → items。**形状校验闸**：meta/results 缺失、count 非 int、任一条目缺
    title 或 slug/path → 整体 None（fail-closed：私有接口无契约，一条畸形即视为契约漂移，
    不挑挑拣拣凑合用）。调用方负责把 None 翻成如实降级。"""
    if not isinstance(payload, dict):
        return None
    meta = payload.get("meta")
    results = payload.get("results")
    count = meta.get("count") if isinstance(meta, dict) else None
    if not isinstance(results, list) or not isinstance(count, int) or isinstance(count, bool):
        return None
    items: list[dict] = []
    for r in results:
        if not isinstance(r, dict):
            return None
        title = r.get("title")
        slug = r.get("slug") if isinstance(r.get("slug"), str) else ""
        path = r.get("path") if isinstance(r.get("path"), str) else ""
        if not isinstance(title, str) or not title.strip() or not (slug.strip() or path.strip()):
            return None
        accession = slug.strip() or path.rstrip("/").rsplit("/", 1)[-1]
        url_path = path if path.startswith("/") else f"/datasets/{accession}"
        items.append({
            "accession": accession,
            "title": title.strip(),
            "url": TENX_BASE + url_path,
            "date": _tenx_api_date(r.get("publishedAt")),
            "snippet": re.sub(r"\s+", " ", str(r.get("body") or "")).strip()[:200],
                                                                    
            "species": [s for s in (r.get("species") or []) if isinstance(s, str)],
        })
    return items


def tenx_dataset_items(*, project_root: Path) -> dict:
    """拉 10x 官网当前**全量**数据集清单（check_updates 在线比对用）：publishedAt DESC 最新在前，
    meta.count 给官网自报总数。私有接口无契约：形状校验不过 → 如实降级（parse_changed），不炸链。"""
    items: list[dict] = []
    total: int | None = None
    for _ in range(_TENX_MAX_PAGES):
        url = (f"{TENX_SEARCH_API}?document=dataset&sort=publishedAt%20DESC"
               f"&limit={_TENX_PAGE_LIMIT}&offset={len(items)}")
        try:
            payload = fetch_json_logged(
                url, project_root=Path(project_root),
                endpoint=TENX_SEARCH_API, query="10x:datasets-full-list",
            )
        except _NetError as exc:
            return _fail("network_error", f"10x 数据集接口请求失败（{exc}）。")
        page = _tenx_api_items(payload)
        if page is None:
            return _fail(
                "parse_changed",
                "10x 官网数据集接口的响应形状变了（它是官网前端私有接口、无官方契约，可能随时漂移），"
                "这次没能读出条目；可到 https://www.10xgenomics.com/datasets 人工核对。",
            )
        total = int(payload["meta"]["count"])
        items.extend(page)
        if len(items) >= total or not page:
            break
    if not items:
        return _fail("no_results", "10x 官网数据集接口返回了空清单（官网自报总数为 0）。")
    note = f"10x 官网接口当前清单 {len(items)} 条（官网自报总数 {total}，按发布时间倒序全量拉取）。"
    return {"ok": True, "items": items, "note_zh": note, "total": total}


def search_10x(
    keywords: str,
    *,
    species: str | None = None,
    limit: int = 20,
    project_root: Path,
) -> dict:
    """10x 适配器：官网私有搜索 API（search= 全文 + tag[species]= 物种 facet 服务端过滤）。

    物种只钉实测词表（Human/Mouse）；词表外物种不打服务端 tag，回退 species 字段本地子串过滤
    （与 ArrayExpress 轻量支同口径）。形状漂移如实降级，不炸链。"""
    kw = str(keywords or "").strip()
    params = ["document=dataset", f"limit={int(limit)}", "offset=0", "sort=publishedAt%20DESC"]
    if kw:
        params.append("search=" + urllib.parse.quote(kw))
    sp = str(species or "").strip()
    tag = _TENX_SPECIES_TAG.get(sp.lower())
    if tag:
        params.append("tag%5Bspecies%5D=" + urllib.parse.quote(tag))
    url = TENX_SEARCH_API + "?" + "&".join(params)
    try:
        payload = fetch_json_logged(
            url, project_root=Path(project_root), endpoint=TENX_SEARCH_API, query=kw or "(全量清单)",
        )
    except _NetError as exc:
        return _fail("network_error", f"10x 数据集接口请求失败（{exc}）。")
    items = _tenx_api_items(payload)
    if items is None:
        return _fail(
            "parse_changed",
            "10x 官网数据集接口的响应形状变了（它是官网前端私有接口、无官方契约，可能随时漂移），"
            "这次没能读出条目；可到 https://www.10xgenomics.com/datasets 人工核对。",
        )
    if sp and not tag:                              
        items = [it for it in items if sp.lower() in " ".join(it.get("species") or []).lower()]
    items = items[: int(limit)]
    if not items:
        return _fail("no_results", f"10x 数据集里没有匹配 {keywords!r} 的条目。")
    return {"ok": True, "items": items, "note_zh": f"10x 官网接口返回 {len(items)} 条。"}


                                                                                  
                                                                     
                                                     
                                                               
                                  
_HCA_SPECIES_TO_LATIN: dict[str, str] = {}
for _latin, _common in ORGANISM_COMMON.items():
    _HCA_SPECIES_TO_LATIN.setdefault(_common.lower(), _latin.capitalize())
del _latin, _common
_AZUL_PAGE_SIZE = 75                          
_AZUL_MAX_PAGES = 12                                  


class _HCAParseError(_NetError):
    """HCA 响应形状漂移；两个入口分别翻译为自己的既有错误契约。"""


def search_hca_hits(
    keywords: str,
    *,
    species: str | None = None,
    limit: int = 20,
    project_root: Path,
    fetcher: Any = None,
) -> tuple[list[dict], str]:
    """HCA 两入口共用的分页/物种/关键词匹配：返回原始命中和覆盖受限说明。

    fetcher 只注入带账本的传输接缝，分页预算与选择逻辑不得由消费方复制。
    词表内物种用服务端 facet；词表外在计入 limit 前按原始 genusSpecies 匹配。
    """
    terms = [t.lower() for t in str(keywords or "").split() if t.strip()]
    sp = str(species or "").strip()
    latin = _HCA_SPECIES_TO_LATIN.get(sp.lower()) if sp else None
    url: str | None = f"{AZUL_PROJECTS_API}?size={_AZUL_PAGE_SIZE}"
    if latin:
        filters = json.dumps({"genusSpecies": {"is": [latin]}}, separators=(",", ":"))
        url += f"&filters={urllib.parse.quote(filters)}"

    matched: list[dict] = []
    seen: set[str] = set()
    pages = 0
    while url and len(matched) < int(limit) and pages < _AZUL_MAX_PAGES:
        payload = (fetcher or fetch_json_logged)(
            url, project_root=Path(project_root), endpoint=AZUL_PROJECTS_API,
            query=str(keywords or "").strip() or "(全量清单)",
        )
        pages += 1
        hits = payload.get("hits") if isinstance(payload, dict) else None
        if not isinstance(hits, list):
            raise _HCAParseError(
                "HCA（Azul）接口的响应形状变了（该服务无公开 API 文档，可能随版本静默变更），"
                "缺 hits 列表。本次没有拿回任何记录；可到 https://data.humancellatlas.org/ 人工核对。",
            )
        for h in hits:
            if not isinstance(h, dict):
                continue
            entry_id = str(h.get("entryId") or "").strip()
            projects = h.get("projects")
            proj = projects[0] if (isinstance(projects, list) and projects
                                   and isinstance(projects[0], dict)) else {}
            title = str(proj.get("projectTitle") or "").strip()
            desc = re.sub(r"\s+", " ", str(proj.get("projectDescription") or "")).strip()
            species_raw = " ".join(
                str(v) for g in (h.get("donorOrganisms") or []) if isinstance(g, dict)
                for v in (g.get("genusSpecies") or []) if isinstance(v, str)
            )
            if not entry_id or not title or entry_id in seen:
                continue
            text = f"{title} {desc} {entry_id}".lower()
            if not all(t in text for t in terms):
                continue
            if sp and not latin and sp.lower() not in species_raw.lower():
                continue                                 
            seen.add(entry_id)
            matched.append(h)
            if len(matched) >= int(limit):
                break
        pagination = payload.get("pagination") if isinstance(payload, dict) else None
        nxt = pagination.get("next") if isinstance(pagination, dict) else None
                                         
        url = nxt if (isinstance(nxt, str) and nxt.startswith(AZUL_PROJECTS_API)) else None
    limited = bool(url and pages >= _AZUL_MAX_PAGES and len(matched) < int(limit))
    note = (f"HCA 搜索已达到 {_AZUL_MAX_PAGES} 页预算，仍有后续页面未检查；"
            "本次覆盖不完整，不能据此断言没有更多匹配条目。") if limited else ""
    return matched, note


def search_hca(
    keywords: str,
    *,
    species: str | None = None,
    limit: int = 20,
    project_root: Path,
) -> dict:
    """共用 HCA 搜索的轻量投影；管护入口对同一原始命中生成完整记录。"""
    try:
        hits, note = search_hca_hits(keywords, species=species, limit=limit, project_root=project_root)
    except _HCAParseError as exc:
        return _fail("parse_changed", str(exc))
    except _NetError as exc:
        return _fail("network_error", f"HCA（Azul）官方接口请求失败（{exc}）。")
    items: list[dict] = []
    for h in hits:
        project = h["projects"][0]
        date = ""
        dates = h.get("dates")
        if isinstance(dates, list) and dates and isinstance(dates[0], dict):
            date = str(dates[0].get("aggregateSubmissionDate") or "")[:10]
        items.append({
            "accession": str(h["entryId"]).strip(),
            "title": str(project["projectTitle"]).strip(),
            "url": HCA_STUDY_TMPL.format(project_id=str(h["entryId"]).strip()),
            "date": date,
            "snippet": re.sub(r"\s+", " ", str(project.get("projectDescription") or "")).strip()[:200],
        })
    if not items:
        if note:
            return _fail("coverage_limited", note)
        return _fail("no_results", f"HCA（Azul）没有匹配 {keywords!r} 的条目。")
    return {"ok": True, "items": items, "note_zh": f"HCA（Azul）官方接口本地匹配后 {len(items)} 条。" + note}


                                                                                      
                                                                        
                                                                    
                                                               
                                                                              
                                                           
GEO_ESEARCH_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
GEO_ESUMMARY_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
GEO_STUDY_TMPL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"
_GEO_TOOL_PARAM = "tool=biodata_agent"
_GEO_MIN_INTERVAL = 0.34                             
_GEO_RECENT_RELDATE = 90                                                 
                                                          

                                                                
                                                                
                                                               
_GEO_SPECIES_TO_LATIN = _HCA_SPECIES_TO_LATIN

                                                                
                                                                        
                                                                         
                                                              
                                                     
                                         
                                                                             
                                                  
GEO_EGEOD_SEARCH_API = "https://www.ebi.ac.uk/biostudies/api/v1/search"
GEO_EGEOD_LEDGER = "corpus_net:geo_egeod"
GEO_EPMC_SEARCH_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
GEO_EPMC_LEDGER = "corpus_net:geo_europepmc"
_EGEOD_ACC_RE = re.compile(r"^E-GEOD-(\d{1,7})$")
_GSE_MENTION_RE = re.compile(r"GSE\d{3,}")

                                                                
                     
_EGEOD_CHANNEL_ZH = "ArrayExpress 的 GEO 镜像（E-GEOD，只有 2016 年前的老数据）"
_EPMC_CHANNEL_ZH = "Europe PMC 文献兜底（全文提到 GSE 号的文献，不是 GEO 数据集清单）"


def _egeod_hits_to_items(payload: Any, *, limit: int) -> list[dict] | None:
    """BioStudies 通用搜索（E-GEOD 集合）→ items。**形状闸**：缺 hits 列表 → None
    （fail-closed）。单条缺 accession/标题、或 accession 不是 E-GEOD 编号 → 只跳过该条
    （不连累其余，与 hca_recent_items 单条跳过同口径）。"""
    hits = payload.get("hits") if isinstance(payload, dict) else None
    if not isinstance(hits, list):
        return None
    items: list[dict] = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        acc = str(h.get("accession") or "").strip()
        m = _EGEOD_ACC_RE.match(acc)
        title = str(h.get("title") or "").strip()
        if not m or not title:
            continue
        items.append({
            "accession": f"GSE{m.group(1)}",                                          
            "egeod_accession": acc,                                        
            "title": title,
            "url": AE_STUDY_TMPL.format(accession=acc),
            "date": str(h.get("release_date") or "").strip(),
            "snippet": re.sub(r"\s+", " ", str(h.get("content") or "")).strip()[:200],
        })
        if len(items) >= int(limit):
            break
    return items


def _search_geo_egeod(kw: str, *, species: str | None, limit: int, project_root: Path,
                      recent: bool = False) -> dict:
    """GEO 降级通道①：BioStudies 的 E-GEOD 集合。query `E-GEOD <kw>` 服务端过滤（recent 模式
    只查 E-GEOD + release_date 倒序）；物种一律本地子串过滤（BioStudies 无 GEO Organism
    词表 facet，与 ArrayExpress 轻量支同口径）。形状漂移如实 parse_changed，不硬解析。"""
    query = " ".join(x for x in ["E-GEOD", kw.strip()] if x)
    url = f"{GEO_EGEOD_SEARCH_API}?query={urllib.parse.quote(query)}&pageSize={int(limit)}"
    if recent:
        url += "&sortBy=release_date&sortOrder=descending"
    try:
        payload = fetch_json_logged(url, project_root=project_root,
                                    endpoint=GEO_EGEOD_LEDGER, query=query)
    except _NetError as exc:
        return _fail("network_error", f"E-GEOD 镜像（BioStudies）请求失败（{exc}）。")
    items = _egeod_hits_to_items(payload, limit=int(limit))
    if items is None:
        return _fail(
            "parse_changed",
            "E-GEOD 镜像（BioStudies）的响应形状变了（缺 hits 列表），这次没能读出条目；"
            "可到 https://www.ebi.ac.uk/biostudies/arrayexpress 人工核对。",
        )
    sp = str(species or "").strip().lower()
    if sp:
        items = [it for it in items if sp in (it["title"] + " " + it.get("snippet", "")).lower()]
    if not items:
        return _fail(
            "no_results",
            f"镜像里没有匹配 {kw.strip() or '（最近条目）'!r} 的条目——这不代表 GEO 真没有"
            "（镜像 2017 年起停止从 GEO 导入）。",
            channel="egeod_mirror",
        )
    return {
        "ok": True,
        "items": items,
        "channel": "egeod_mirror",
        "note_zh": f"镜像返回 {len(items)} 条（编号已按 E-GEOD-n → GSE-n 换算）。",
    }


def _europepmc_gse_items(payload: Any, *, limit: int) -> list[dict] | None:
    """Europe PMC 搜索响应 → items。**形状闸**：缺 resultList.result 列表 → None
    （fail-closed）。从标题/摘要抠 GSE 号提及，一个 GSE 号一条（跨文献去重）。"""
    result_list = payload.get("resultList") if isinstance(payload, dict) else None
    results = result_list.get("result") if isinstance(result_list, dict) else None
    if not isinstance(results, list):
        return None
    items: list[dict] = []
    seen: set[str] = set()
    for r in results:
        if not isinstance(r, dict):
            continue
        title = str(r.get("title") or "").strip()
        text = f"{title} {r.get('abstractText') or ''}"
        src = str(r.get("source") or "").strip() or "MED"
        pid = str(r.get("id") or "").strip()
        for gse in _GSE_MENTION_RE.findall(text):
            if gse in seen:
                continue
            seen.add(gse)
            items.append({
                "accession": gse,
                "title": title or gse,
                "url": f"https://europepmc.org/abstract/{src}/{pid}" if pid else "",
                "date": str(r.get("pubYear") or "").strip(),
                "snippet": re.sub(r"\s+", " ", str(r.get("abstractText") or "")).strip()[:200],
                "mentioned_in": "Europe PMC 文献全文",                 
            })
            if len(items) >= int(limit):
                return items
    return items


def _search_geo_europepmc(kw: str, *, limit: int, project_root: Path) -> dict:
    """GEO 降级通道②（弱兜底）：Europe PMC 全文检索 `TITLE_ABS:"GSE" AND (kw)` 抠 GSE 号提及。
    只能证明「有文献提到这些 GSE」——不是数据集维度，给不了 GEO 元数据；文献不标数据集
    物种，物种过滤无从谈起，note 如实写明。形状漂移如实 parse_changed。"""
    terms = ['TITLE_ABS:"GSE"']
    if kw.strip():
        terms.append(f"({kw.strip()})")
    query = " AND ".join(terms)
    url = (f"{GEO_EPMC_SEARCH_API}?query={urllib.parse.quote(query)}"
           f"&format=json&resultType=core&pageSize={int(limit)}")
    try:
        payload = fetch_json_logged(url, project_root=project_root,
                                    endpoint=GEO_EPMC_LEDGER, query=query)
    except _NetError as exc:
        return _fail("network_error", f"Europe PMC 请求失败（{exc}）。")
    items = _europepmc_gse_items(payload, limit=int(limit))
    if items is None:
        return _fail(
            "parse_changed",
            "Europe PMC 接口的响应形状变了（缺 resultList.result 列表），这次没能读出条目；"
            "可到 https://europepmc.org/ 人工核对。",
        )
    if not items:
        return _fail(
            "no_results",
            f"文献里也没找到提到 GSE 号的 {kw.strip() or '（最近条目）'!r} 相关条目。",
            channel="europepmc_literature",
        )
    return {
        "ok": True,
        "items": items,
        "channel": "europepmc_literature",
        "note_zh": (f"文献兜底返回 {len(items)} 个 GSE 号（按文献标题/摘要提及抠出，"
                    "只能证明有文献提到，给不了数据集元数据）。"),
    }


def _geo_all_channels_failed(reasons: list[str]) -> dict:
    """三条通道全败的如实失败形态：逐条写清每条通道的败因，不挑一条背锅、不含糊。"""
    labels = ["① NCBI E-utilities（主通道）", f"② {_EGEOD_CHANNEL_ZH}", f"③ {_EPMC_CHANNEL_ZH}"]
    detail = "；".join(f"{lab}——{why}" for lab, why in zip(labels, reasons))
    return _fail(
        "all_channels_failed",
        f"GEO 的三条联网通道这次都没通：{detail}。"
        "可到 https://www.ncbi.nlm.nih.gov/geo/ 人工核对，网络恢复后再说一次即可重试。",
        channel="none",
        channels_tried=["ncbi_eutils", "egeod_mirror", "europepmc_literature"],
    )


def _geo_esearch_ids(payload: Any) -> list[str] | None:
    """esearch JSON → GDS UID 列表。形状校验闸：缺 esearchresult.idlist 列表 → None
    （fail-closed：漂移即视为契约变化，不硬解析）。"""
    result = payload.get("esearchresult") if isinstance(payload, dict) else None
    idlist = result.get("idlist") if isinstance(result, dict) else None
    if not isinstance(idlist, list):
        return None
    return [str(i).strip() for i in idlist if str(i).strip()]


def _geo_doc_to_item(doc: dict) -> dict | None:
    """一条 esummary 文档 → 统一 items 形态；缺 accession/标题 → None（调用方跳过该条，
    不连累其余——与 hca_recent_items 单条跳过同口径）。"""
    acc = str(doc.get("accession") or "").strip()
    title = str(doc.get("title") or "").strip()
    if not acc or not title:
        return None
    return {
        "accession": acc,
        "title": title,
        "url": GEO_STUDY_TMPL.format(accession=acc),
        "date": str(doc.get("pdat") or "").strip().replace("/", "-"),                      
        "snippet": re.sub(r"\s+", " ", str(doc.get("summary") or "")).strip()[:200],
                                                             
        "taxon": str(doc.get("taxon") or "").strip(),
        "gdstype": str(doc.get("gdstype") or "").strip(),
    }


def _geo_summary_items(ids: list[str], *, project_root: Path, query: str) -> list[dict] | None:
    """esummary 批量取详情 → items。形状校验闸：缺 result.uids 列表 → None（fail-closed）；
    网络失败抛 _NetError（调用方统一降级，不在此吞）。"""
    url = (f"{GEO_ESUMMARY_API}?db=gds&id={','.join(ids)}&retmode=json&{_GEO_TOOL_PARAM}")
    payload = fetch_json_logged(
        url, project_root=project_root, endpoint=GEO_ESUMMARY_API, query=query,
        min_interval=_GEO_MIN_INTERVAL,
    )
    result = payload.get("result") if isinstance(payload, dict) else None
    uids = result.get("uids") if isinstance(result, dict) else None
    if not isinstance(uids, list):
        return None
    items: list[dict] = []
    for uid in uids:
        doc = result.get(str(uid))
        if not isinstance(doc, dict):
            continue
        item = _geo_doc_to_item(doc)
        if item is not None:
            items.append(item)
    return items


def _search_geo_ncbi(kw: str, *, species: str | None, limit: int, project_root: Path) -> dict:
    """GEO 主通道：E-utilities 两段式（esearch → esummary 批量取详情）。kw 由编排层
    （search_geo）校验非空。term 组装 `(kw) AND "GSE"[Entry Type]`，词表内物种加
    `AND "学名"[Organism]` 服务端过滤，词表外回退 taxon 原文本地子串过滤。"""
    sp = str(species or "").strip()
    latin = _GEO_SPECIES_TO_LATIN.get(sp.lower()) if sp else None
    term_parts = [f"({kw})"]
    if latin:
        term_parts.append(f'"{latin}"[Organism]')
    term_parts.append('"GSE"[Entry Type]')
    term = " AND ".join(term_parts)
    url = (f"{GEO_ESEARCH_API}?db=gds&term={urllib.parse.quote(term)}"
           f"&retmax={int(limit)}&retmode=json&{_GEO_TOOL_PARAM}")
    try:
        payload = fetch_json_logged(
            url, project_root=project_root, endpoint=GEO_ESEARCH_API, query=term,
            min_interval=_GEO_MIN_INTERVAL,
        )
        ids = _geo_esearch_ids(payload)
        if ids is None:
            return _fail(
                "parse_changed",
                "NCBI E-utilities 的响应形状变了（esearch 缺 esearchresult.idlist），"
                "这次没能读出条目；可到 https://www.ncbi.nlm.nih.gov/geo/ 人工核对。",
            )
        if not ids:
            return _fail("no_results", f"GEO 没有匹配 {kw!r} 的条目。")
        items = _geo_summary_items(ids, project_root=project_root, query=term)
    except _NetError as exc:
        return _fail("network_error", f"NCBI GEO（E-utilities）请求失败（{exc}）。")
    if items is None:
        return _fail(
            "parse_changed",
            "NCBI E-utilities 的响应形状变了（esummary 缺 result.uids），"
            "这次没能读出条目；可到 https://www.ncbi.nlm.nih.gov/geo/ 人工核对。",
        )
    if sp and not latin:                                   
        items = [it for it in items if sp.lower() in it.get("taxon", "").lower()]
    items = items[: int(limit)]
    if not items:
        return _fail("no_results", f"GEO 没有匹配 {kw!r} 的条目。")
    return {"ok": True, "items": items, "channel": "ncbi_eutils",
            "note_zh": f"NCBI GEO 官方接口返回 {len(items)} 条。"}


def search_geo(
    keywords: str,
    *,
    species: str | None = None,
    limit: int = 20,
    project_root: Path,
) -> dict:
    """GEO 轻量 items 搜索（三通道降级编排，2026-08-07 降级施工）。

    主通道 NCBI E-utilities；失败（network_error / parse_changed 形状闸拦）→ 降级①
    E-GEOD 镜像（BioStudies，**只有 2016 年前的老数据**）→ 降级② Europe PMC 文献弱兜底；
    全败 → 如实报「三条通道都没通」（all_channels_failed，逐条列败因）。主通道真没结果
    （no_results）是诚实的完整答案，**不**触发降级。每次降级 note_zh 与 channel 字段
    如实写明实际走的通道（含 E-GEOD 年代局限），绝不假装是主通道数据。形状漂移如实
    降级，不炸链。"""
    kw = str(keywords or "").strip()
    if not kw:
        return _fail("empty_query", "GEO 搜索关键词为空，未发请求。")
    root = Path(project_root)
    primary = _search_geo_ncbi(kw, species=species, limit=limit, project_root=root)
    if primary["ok"] or primary.get("error") == "no_results":
        return primary
    reasons = [str(primary.get("note_zh") or primary.get("error") or "未知原因")]
    mirror = _search_geo_egeod(kw, species=species, limit=limit, project_root=root)
    if mirror["ok"] or mirror.get("error") == "no_results":
        mirror["note_zh"] = (f"NCBI 连不上，本次走了{_EGEOD_CHANNEL_ZH}。"
                             + str(mirror.get("note_zh") or ""))
        return mirror
    reasons.append(str(mirror.get("note_zh") or mirror.get("error") or "未知原因"))
    epmc = _search_geo_europepmc(kw, limit=limit, project_root=root)
    if epmc["ok"] or epmc.get("error") == "no_results":
        epmc["note_zh"] = (f"NCBI 和 E-GEOD 镜像都没通，本次走了{_EPMC_CHANNEL_ZH}。"
                           + str(epmc.get("note_zh") or ""))
        return epmc
    reasons.append(str(epmc.get("note_zh") or epmc.get("error") or "未知原因"))
    return _geo_all_channels_failed(reasons)


                                                                                    
                                                     
                                                                          
                                                                       
                                                               
                                                             
                                                              
                                                                       
                                                                
                                                                
ZENODO_API = "https://zenodo.org/api/records"
ZENODO_RECORD_TMPL = "https://zenodo.org/records/{record_id}"
_ZENODO_MIN_INTERVAL = 3.0                                            
_ZENODO_MAX_SIZE = 25                               

                                                                              
                                                             
                                                       
_ZENODO_SPECIES_TO_LATIN = _HCA_SPECIES_TO_LATIN

                                                       
                  
_ZENODO_SPECIES_RES: "list[tuple[re.Pattern, str]]" = [
    (re.compile(r"\b" + re.escape(latin) + r"\b", re.I), latin)
    for latin in dict.fromkeys(_ZENODO_SPECIES_TO_LATIN.values())
]


def _zenodo_extract_species(*texts: str) -> list[str]:
    """从自由文本抠物种学名（词表内、词边界、大小写不敏感）；抠不到 → []（留空不编）。"""
    combined = " ".join(t for t in texts if t)
    if not combined:
        return []
    return [latin for rx, latin in _ZENODO_SPECIES_RES if rx.search(combined)]


def _zenodo_lucene_quote(term: str) -> str:
    """Lucene 短语引号清洗：剥掉 term 内引号，防用户关键词把字段查询语法顶断。"""
    return re.sub(r"\s+", " ", term.replace('"', " ")).strip()


def _zenodo_items(payload: Any, *, limit: int) -> list[dict] | None:
    """Zenodo /api/records 响应 → items。**形状闸（fail-closed）**：缺 hits.hits 列表、或任一
    条目缺 id(int)/metadata.title(str 非空) → 整体 None——身份字段是公开契约核心（legacy 与
    InvenioRDM 两版共有），漂移即视为我们对形状的理解已过期，不挑挑拣拣凑合用。
    description/keywords/doi/publication_date 缺失容忍（诚实缺省）；resource_type 存在且
    type != "dataset" 的条目只跳过该条（type=dataset 是服务端过滤，混入非 dataset 不算漂移）。"""
    hits = payload.get("hits") if isinstance(payload, dict) else None
    hit_list = hits.get("hits") if isinstance(hits, dict) else None
    if not isinstance(hit_list, list):
        return None
    items: list[dict] = []
    seen: set[int] = set()
    for h in hit_list:
        if not isinstance(h, dict):
            return None
        rid = h.get("id")
        if not isinstance(rid, int) or isinstance(rid, bool):
            return None
        meta = h.get("metadata")
        if not isinstance(meta, dict):
            return None
        title = str(meta.get("title") or "").strip()
        if not title:
            return None
        rtype = meta.get("resource_type")
        if isinstance(rtype, dict):
            rt = str(rtype.get("type") or rtype.get("id") or "").strip().lower()
            if rt and rt != "dataset":
                continue                                     
        if rid in seen:
            continue
        seen.add(rid)
        desc = re.sub(r"\s+", " ", _strip_tags(str(meta.get("description") or ""))).strip()
        doi = str(h.get("doi") or "").strip()
        if not doi:                              
            pids = h.get("pids")
            if isinstance(pids, dict) and isinstance(pids.get("doi"), dict):
                doi = str(pids["doi"].get("identifier") or "").strip()
        date = str(meta.get("publication_date") or "").strip()[:10]
        if not date:                         
            date = str(h.get("created") or "").strip()[:10]
        items.append({
            "accession": str(rid),
            "title": title,
            "url": ZENODO_RECORD_TMPL.format(record_id=rid),
            "date": date,
            "snippet": desc[:200],
                                                 
            "doi": doi,
            "species": _zenodo_extract_species(title, desc),
        })
        if len(items) >= int(limit):
            break
    return items


def search_zenodo(
    keywords: str,
    *,
    species: str | None = None,
    limit: int = 20,
    project_root: Path,
) -> dict:
    """Zenodo 轻量 items 搜索：Lucene 字段限定查询（metadata.title/description 短语 OR +
    type=dataset），不用裸自由词（实测噪声大）。物种（词表内→学名，词表外→原词）AND 进
    title/description 字段查询做服务端文本过滤——Zenodo 无物种结构化字段，这是文本级近似，
    note 如实写明。限速 30 req/min 留余量 20/min；形状漂移如实降级（parse_changed），不炸链。"""
    kw = str(keywords or "").strip()
    if not kw:
        return _fail("empty_query", "Zenodo 搜索关键词为空，未发请求。")
    kw_q = _zenodo_lucene_quote(kw)
    parts = [f'(metadata.title:"{kw_q}" OR metadata.description:"{kw_q}")']
    sp = str(species or "").strip()
    if sp:
        word = _zenodo_lucene_quote(_ZENODO_SPECIES_TO_LATIN.get(sp.lower()) or sp)
        parts.append(f'(metadata.title:"{word}" OR metadata.description:"{word}")')
    query = " AND ".join(parts)
    url = (f"{ZENODO_API}?q={urllib.parse.quote(query)}&type=dataset"
           f"&size={min(int(limit), _ZENODO_MAX_SIZE)}")
    try:
        payload = fetch_json_logged(
            url, project_root=Path(project_root), endpoint=ZENODO_API, query=query,
            min_interval=_ZENODO_MIN_INTERVAL,
        )
    except _NetError as exc:
        return _fail("network_error", f"Zenodo 官方 API 请求失败（{exc}）。")
    items = _zenodo_items(payload, limit=int(limit))
    if items is None:
        return _fail(
            "parse_changed",
            "Zenodo 官方 API 的响应形状变了（缺 hits.hits 列表或条目缺 id/标题），"
            "这次没能读出条目；可到 https://zenodo.org/ 人工核对。",
        )
    if not items:
        return _fail("no_results", f"Zenodo 没有匹配 {kw!r} 的 type=dataset 条目。")
    note = (f"Zenodo 官方 API 返回 {len(items)} 条。Zenodo 是通用开放仓储，生物数据集只占一部分；"
            "物种是从标题/描述文本里抠的、不全，组织/疾病无结构化字段。")
    return {"ok": True, "items": items, "note_zh": note}


                                                                                                
                                                  
                                                                                                

def encode_recent_items(*, project_root: Path, limit: int = 10) -> dict:
    """ENCODE 最近创建的 Experiment 清单（check_updates 用：要的是新，不是主题过滤）。"""
    url = (f"{ENCODE_SEARCH_API}?type=Experiment&format=json&limit={int(limit)}"
           f"&sort=-date_created")
    try:
        payload = fetch_json_logged(
            url, project_root=Path(project_root), endpoint=ENCODE_SEARCH_API,
            query="check_updates:recent", headers=_ENCODE_HEADERS,
        )
    except _NetError as exc:
        if str(exc).startswith("HTTP 404"):                      
            return _fail("no_results", "ENCODE 最近条目为空。")
        return _fail("network_error", f"ENCODE 官方 API 请求失败（{exc}）。")
    items = _encode_graph_items(payload, limit=int(limit))
    if not items:
        return _fail("no_results", "ENCODE 最近条目为空（官方 API 没返回可核对的编号）。")
    return {"ok": True, "items": items}


def hca_recent_items(*, project_root: Path, limit: int = 10) -> dict:
    """HCA（Azul）最近入库的项目清单（check_updates 用：sort=aggregateSubmissionDate&order=desc）。

    Azul 无公开 OpenAPI 文档（/openapi 实测要鉴权）：响应先过形状校验——缺 hits 列表 →
    如实降级（parse_changed），不炸链。单条命中缺 entryId/标题只跳过该条（不连累其余）。"""
    url = (f"{AZUL_PROJECTS_API}?size={int(limit)}"
           f"&sort=aggregateSubmissionDate&order=desc")
    try:
        payload = fetch_json_logged(
            url, project_root=Path(project_root), endpoint=AZUL_PROJECTS_API,
            query="check_updates:recent",
        )
    except _NetError as exc:
        return _fail("network_error", f"HCA（Azul）官方接口请求失败（{exc}）。")
    hits = payload.get("hits") if isinstance(payload, dict) else None
    if not isinstance(hits, list):
        return _fail(
            "parse_changed",
            "HCA（Azul）接口的响应形状变了（该服务无公开 API 文档，可能随版本静默变更），"
            "这次没能读出条目；可到 https://data.humancellatlas.org/ 人工核对。",
        )
    items: list[dict] = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        entry_id = str(h.get("entryId") or "").strip()
        projects = h.get("projects")
        proj = projects[0] if (isinstance(projects, list) and projects
                               and isinstance(projects[0], dict)) else {}
        title = str(proj.get("projectTitle") or "").strip()
        if not entry_id or not title:
            continue
        date = ""
        dates = h.get("dates")
        if isinstance(dates, list) and dates and isinstance(dates[0], dict):
            date = str(dates[0].get("aggregateSubmissionDate") or "")[:10]
        items.append({
            "accession": entry_id,
            "title": title,
            "url": HCA_STUDY_TMPL.format(project_id=entry_id),
            "date": date,
        })
    if not items:
        return _fail("no_results", "HCA（Azul）最近条目为空（官方接口没返回可核对的项目）。")
    return {"ok": True, "items": items, "note_zh": f"HCA（Azul）官方接口返回最近 {len(items)} 条。"}


def _geo_recent_items_ncbi(*, project_root: Path, limit: int = 10) -> dict:
    """GEO 主通道「最近条目」："GSE"[Entry Type] + reldate/datetype=pdat 相对日期窗口
    （调研 §4 实测配方，窗口放宽到 90 天防静默期空窗）。

    esearch 不保证按日期排序：多取一批（4×limit、封顶 200 条）esummary 富化后**本地按 pdat
    倒序**截断，确保「最近」口径成立。形状漂移如实降级（parse_changed），不炸链。"""
    term = '"GSE"[Entry Type]'
    fetch_n = min(max(int(limit) * 4, 50), 200)
    url = (f"{GEO_ESEARCH_API}?db=gds&term={urllib.parse.quote(term)}"
           f"&reldate={_GEO_RECENT_RELDATE}&datetype=pdat&retmax={fetch_n}"
           f"&retmode=json&{_GEO_TOOL_PARAM}")
    try:
        payload = fetch_json_logged(
            url, project_root=project_root, endpoint=GEO_ESEARCH_API,
            query="check_updates:recent", min_interval=_GEO_MIN_INTERVAL,
        )
        ids = _geo_esearch_ids(payload)
        if ids is None:
            return _fail(
                "parse_changed",
                "NCBI E-utilities 的响应形状变了（esearch 缺 esearchresult.idlist），"
                "这次没能读出条目；可到 https://www.ncbi.nlm.nih.gov/geo/ 人工核对。",
            )
        if not ids:
            return _fail("no_results", f"GEO 最近 {_GEO_RECENT_RELDATE} 天没有新公开的 Series。")
        items = _geo_summary_items(ids, project_root=project_root, query="check_updates:recent")
    except _NetError as exc:
        return _fail("network_error", f"NCBI GEO（E-utilities）请求失败（{exc}）。")
    if items is None:
        return _fail(
            "parse_changed",
            "NCBI E-utilities 的响应形状变了（esummary 缺 result.uids），"
            "这次没能读出条目；可到 https://www.ncbi.nlm.nih.gov/geo/ 人工核对。",
        )
    items.sort(key=lambda it: str(it.get("date") or ""), reverse=True)
    items = items[: int(limit)]
    if not items:
        return _fail("no_results", "GEO 最近条目为空（官方接口没返回可核对的编号）。")
    note = (f"NCBI GEO 官方接口返回最近 {len(items)} 条"
            f"（pdat {_GEO_RECENT_RELDATE} 天窗口内按公开日期倒序）。")
    return {"ok": True, "items": items, "channel": "ncbi_eutils", "note_zh": note}


def geo_recent_items(*, project_root: Path, limit: int = 10) -> dict:
    """GEO 最近公开的 Series 清单（check_updates 用，三通道降级编排，口径同 search_geo）。

    主通道 NCBI E-utilities（pdat 窗口）；失败 → 降级① E-GEOD 镜像按 release_date 倒序
    （镜像 ≤2016，「最近」其实是镜像里的最新老数据——2016 年后的 GEO 新数据这次看不到，
    note 如实写明）→ 降级② Europe PMC 文献弱兜底；全败如实报（all_channels_failed）。
    主通道 no_results（窗口真空窗）是诚实答案，不触发降级。"""
    root = Path(project_root)
    primary = _geo_recent_items_ncbi(project_root=root, limit=limit)
    if primary["ok"] or primary.get("error") == "no_results":
        return primary
    reasons = [str(primary.get("note_zh") or primary.get("error") or "未知原因")]
    mirror = _search_geo_egeod("", species=None, limit=limit, project_root=root, recent=True)
    if mirror["ok"] or mirror.get("error") == "no_results":
        mirror["note_zh"] = (f"NCBI 连不上，本次走了{_EGEOD_CHANNEL_ZH}。"
                             + str(mirror.get("note_zh") or "")
                             + "2016 年后的 GEO 新数据这次看不到。")
        return mirror
    reasons.append(str(mirror.get("note_zh") or mirror.get("error") or "未知原因"))
    epmc = _search_geo_europepmc("", limit=limit, project_root=root)
    if epmc["ok"] or epmc.get("error") == "no_results":
        epmc["note_zh"] = (f"NCBI 和 E-GEOD 镜像都没通，本次走了{_EPMC_CHANNEL_ZH}。"
                           + str(epmc.get("note_zh") or ""))
        return epmc
    reasons.append(str(epmc.get("note_zh") or epmc.get("error") or "未知原因"))
    return _geo_all_channels_failed(reasons)


def zenodo_recent_items(*, project_root: Path, limit: int = 10) -> dict:
    """Zenodo 最新 type=dataset 条目（check_updates 用：type=dataset&sort=mostrecent 一页）。

    形状闸与 search_zenodo 同一道（`_zenodo_items`，fail-closed）。注意口径：Zenodo 是
    通用开放仓储，这里拉的是**全领域**最新 dataset 条目（不限生物）——note 如实写明，
    比对侧（corpus_curation）据此措辞。"""
    url = (f"{ZENODO_API}?type=dataset&sort=mostrecent"
           f"&size={min(int(limit), _ZENODO_MAX_SIZE)}")
    try:
        payload = fetch_json_logged(
            url, project_root=Path(project_root), endpoint=ZENODO_API,
            query="check_updates:recent", min_interval=_ZENODO_MIN_INTERVAL,
        )
    except _NetError as exc:
        return _fail("network_error", f"Zenodo 官方 API 请求失败（{exc}）。")
    items = _zenodo_items(payload, limit=int(limit))
    if items is None:
        return _fail(
            "parse_changed",
            "Zenodo 官方 API 的响应形状变了（缺 hits.hits 列表或条目缺 id/标题），"
            "这次没能读出条目；可到 https://zenodo.org/ 人工核对。",
        )
    if not items:
        return _fail("no_results", "Zenodo 最近条目为空（官方 API 没返回可核对的编号）。")
    note = (f"Zenodo 官方 API 返回全领域最新 {len(items)} 条 type=dataset 条目"
            "（通用开放仓储，不限生物领域）。")
    return {"ok": True, "items": items, "note_zh": note}


                                                                                                
                                                                         
                                                              
                                                                                      
                                                               
                                                              
                                                                         
                                                     
                                                        
                      
                                                                                                
REFINEBIO_SEARCH_API = "https://api.refine.bio/v1/search/"
REFINEBIO_EXP_TMPL = "https://www.refine.bio/experiments/{accession}"
_REFINEBIO_MIN_INTERVAL = 1.0                           
_REFINEBIO_MAX_LIMIT = 500                                         


def _refinebio_items(payload: Any, *, limit: int) -> list[dict] | None:
    """refine.bio /v1/search/ 响应 → items。**形状闸（fail-closed）**：缺 results 列表、或任一
    条目缺 accession_code(str 非空)/title(str 非空) → 整体 None——身份字段是公开契约核心，
    漂移即视为我们对形状的理解已过期，不挑挑拣拣凑合用。其余字段缺失容忍（诚实缺省）。"""
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return None
    items: list[dict] = []
    seen: set[str] = set()
    for h in results:
        if not isinstance(h, dict):
            return None
        acc = str(h.get("accession_code") or "").strip()
        title = str(h.get("title") or "").strip()
        if not acc or not title:
            return None
        if acc in seen:                                          
            continue
        seen.add(acc)
        desc = re.sub(r"\s+", " ", str(h.get("description") or "")).strip()
        items.append({
            "accession": acc,
            "title": title,
            "url": REFINEBIO_EXP_TMPL.format(accession=acc),
            "date": str(h.get("source_first_published") or "").strip()[:10],
            "snippet": desc[:200],
                                       
            "alternate_accession_code": str(h.get("alternate_accession_code") or "").strip(),
            "organisms": h.get("organism_names") if isinstance(h.get("organism_names"), list) else [],
            "technology": str(h.get("technology") or "").strip(),
            "num_downloadable_samples": h.get("num_downloadable_samples"),
        })
        if len(items) >= int(limit):
            break
    return items


def refinebio_organism_param(species: str) -> str:
    """通用名 → refine.bio organism 过滤值（UPPER_SNAKE 学名，如 HOMO_SAPIENS）；
    词表外 → ""（不做服务端过滤，不乱猜映射）。

    这是两个入口（本模块 search_refinebio 与
    corpus_curation._search_refinebio）的**同一映射真源**——此前 curation 侧把带空格的
    词表外词当二名法学名透传（"white mouse" → organism=WHITE_MOUSE 假过滤、零结果无提示），
    net 侧则静默不过滤且无提示，同一源两种静默。词表外一律不过滤，由调用方在 note 里如实写明。"""
    text = str(species or "").strip()
    if not text:
        return ""
    latin = _ZENODO_SPECIES_TO_LATIN.get(text.lower())                             
    return latin.upper().replace(" ", "_") if latin else ""


def search_refinebio(
    keywords: str,
    *,
    species: str | None = None,
    limit: int = 20,
    project_root: Path,
) -> dict:
    """refine.bio 轻量 items 搜索：/v1/search/?search=<kw>（ES 模糊 OR 全文，召回含弱相关，
    note 如实写明）；species（词表内→UPPER_SNAKE 学名，词表外→不过滤）AND 进 organism
    服务端过滤。限速 ≤60 req/min；形状漂移如实降级（parse_changed），不炸链。"""
    kw = str(keywords or "").strip()
    if not kw:
        return _fail("empty_query", "refine.bio 搜索关键词为空，未发请求。")
    url = f"{REFINEBIO_SEARCH_API}?search={urllib.parse.quote(kw)}&limit={min(int(limit), _REFINEBIO_MAX_LIMIT)}"
    sp = str(species or "").strip()
    organism = refinebio_organism_param(sp)
    if organism:
        url += f"&organism={organism}"
    species_note = ""
    if sp and not organism:
                                 
                                                   
        species_note = f"注意：物种词「{sp}」不在已知词表里，这次没有按物种过滤。"
    try:
        payload = fetch_json_logged(
            url, project_root=Path(project_root), endpoint=REFINEBIO_SEARCH_API, query=kw,
            min_interval=_REFINEBIO_MIN_INTERVAL,
        )
    except _NetError as exc:
        return _fail("network_error", f"refine.bio 官方 API 请求失败（{exc}）。")
    items = _refinebio_items(payload, limit=int(limit))
    if items is None:
        return _fail(
            "parse_changed",
            "refine.bio 官方 API 的响应形状变了（缺 results 列表或条目缺 accession_code/标题），"
            "这次没能读出条目；可到 https://www.refine.bio/ 人工核对。",
        )
    if not items:
        return _fail("no_results", species_note + f"refine.bio 没有匹配 {kw!r} 的实验。")
    note = (f"refine.bio 官方 API 返回 {len(items)} 条。refine.bio 是 GEO/SRA/ArrayExpress 的"
            "统一加工镜像（与库中 GEO/AE 记录可能指向同一研究）；其全文检索是模糊匹配，"
            "结果含弱相关条目，需人工甄别。")
    return {"ok": True, "items": items, "note_zh": species_note + note}


def refinebio_recent_items(*, project_root: Path, limit: int = 10) -> dict:
    """refine.bio 最新实验条目（check_updates 用：/v1/search/ ordering=-source_first_published
    一页，不带主题词——要的是新，不是主题过滤）。

    形状闸与 search_refinebio 同一道（`_refinebio_items`，fail-closed）。口径如实标注：
    比对的是**全库**最新（不限单细胞/空间切片主题）；且 refine.bio 上游加工实测 2023 年后
    基本停更（最新 source_first_published 2023-02），新增候选通常不会很多。"""
    url = (f"{REFINEBIO_SEARCH_API}?ordering=-source_first_published"
           f"&limit={min(int(limit), _REFINEBIO_MAX_LIMIT)}")
    try:
        payload = fetch_json_logged(
            url, project_root=Path(project_root), endpoint=REFINEBIO_SEARCH_API,
            query="check_updates:recent", min_interval=_REFINEBIO_MIN_INTERVAL,
        )
    except _NetError as exc:
        return _fail("network_error", f"refine.bio 官方 API 请求失败（{exc}）。")
    items = _refinebio_items(payload, limit=int(limit))
    if items is None:
        return _fail(
            "parse_changed",
            "refine.bio 官方 API 的响应形状变了（缺 results 列表或条目缺 accession_code/标题），"
            "这次没能读出条目；可到 https://www.refine.bio/ 人工核对。",
        )
    if not items:
        return _fail("no_results", "refine.bio 最近条目为空（官方 API 没返回可核对的编号）。")
    note = (f"refine.bio 官方 API 返回全库最新 {len(items)} 条实验（不限单细胞/空间切片主题；"
            "上游加工实测 2023 年后基本停更）。")
    return {"ok": True, "items": items, "note_zh": note}


                                                                                                
                             
                                                                                                

                                                      
                                                        
                                                     
                                              
SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
                                
    "ddg": ("ddg",),
    "duckduckgo": ("ddg",),
    "web": ("ddg",),
    "generic": ("ddg",),
    "通用": ("ddg",),
                  
    "ae": ("arrayexpress",),
               
    "cxg": ("cellxgene",),
    "cellxgene discover": ("cellxgene",),
         
    "10x genomics": ("10x",),
    "tenx": ("10x",),
         
    "human cell atlas": ("hca",),
    "azul": ("hca",),
         
    "ncbi geo": ("geo",),
                
    "refine.bio": ("refinebio",),
    "refine bio": ("refinebio",),
                                      
    "single cell expression atlas": ("ebi_scea",),
    "scea": ("ebi_scea",),
                                                                             
    "scp": ("single_cell_portal", "scp"),
    "single cell portal": ("single_cell_portal", "scp"),
    "broad single cell portal": ("single_cell_portal", "scp"),
}


def resolve_source_key(alias: Any, *, valid_keys) -> str | None:
    """口语来源名/别名 → 调用方注册表键；认不出 → None（调用方 fail-closed）。

    三级解析（顺序即各注册表解析器的既有语义）：① 注册表键精确匹配；② label 小写精确
    匹配（valid_keys 为「键 → 带 label 的 spec」注册表 dict 时）；③ SOURCE_ALIASES 别名表，
    取第一个落在 valid_keys 键集内的候选。valid_keys 传注册表 dict 或纯键集容器均可。"""
    text = str(alias or "").strip().lower()
    if not text:
        return None
    if text in valid_keys:
        return text
    for key, spec in (valid_keys.items() if hasattr(valid_keys, "items") else ()):
        if text == str(spec["label"]).lower():
            return key
    for key in SOURCE_ALIASES.get(text, ()):
        if key in valid_keys:
            return key
    return None


                                                          
_ONLINE_SOURCE_KEYS = frozenset({
    "ddg", "arrayexpress", "encode", "10x", "hca", "geo", "zenodo", "refinebio",
})


def search_online_source(
    source: Any,
    keywords: Any,
    species: Any = None,
    limit: int = 20,
    *,
    project_root: Path,
) -> dict:
    """统一出口：`{ok, items:[{accession,title,url,date?,snippet?}], note_zh?, error?}`。

    - source：ddg/web/generic（通用搜索主力）| arrayexpress | encode | 10x | hca | geo | zenodo | refinebio（官方适配器对照）；
    - 任何失败（网络/解析/参数）都落成 ok=False + error 机器码 + note_zh，**绝不抛异常**。"""
    key = resolve_source_key(source, valid_keys=_ONLINE_SOURCE_KEYS)
    if key is None:
        return _fail(
            "unknown_source",
            f"未知来源 {source!r}，可选：通用搜索、arrayexpress、encode、10x、hca、geo、zenodo、refinebio。",
        )
    species_s = str(species or "").strip() or None
    kw = str(keywords or "").strip()
    try:
        if key == "ddg":
            return search_duckduckgo(kw, species=species_s, limit=limit, project_root=Path(project_root))
        if key == "arrayexpress":
            return search_arrayexpress_items(kw, species=species_s, limit=limit, project_root=Path(project_root))
        if key == "encode":
            return search_encode(kw, species=species_s, limit=limit, project_root=Path(project_root))
        if key == "hca":
            return search_hca(kw, species=species_s, limit=limit, project_root=Path(project_root))
        if key == "geo":
            return search_geo(kw, species=species_s, limit=limit, project_root=Path(project_root))
        if key == "zenodo":
            return search_zenodo(kw, species=species_s, limit=limit, project_root=Path(project_root))
        if key == "refinebio":
            return search_refinebio(kw, species=species_s, limit=limit, project_root=Path(project_root))
        return search_10x(kw, species=species_s, limit=limit, project_root=Path(project_root))
    except Exception as exc:                            
        return _fail("unexpected_error", f"{type(exc).__name__}: {exc}")
