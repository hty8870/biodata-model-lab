                       
"""公开基准数据集机制专项离线测试：配置闸、下载/重定向闸、解包闸、解析正反例、记录合同、
只读 verify 对账与端到端夹具。

全部走 fixture 与注入的假 opener/假 transport：不触网（autouse 禁 socket/urlopen），不读
ignored runs，不依赖真实数据。r1 修复对应覆盖：真只读 verify（不改/不建/不删 run 与账本、
保留未知目录）、每跳重定向前置校验（禁域后跳回允许域也拒）、pinned ZIP→extracted→normalized
来源链（改 extracted 再自洽重 normalize 也拒）、run 内统一路径守卫与独占 `.part`、ZIP 重复
成员/Windows 别名/ADS/设备名、extract/normalize 失败入账本、verify 全承诺字段对账与 `--only`
子集语义。

r2 收口对应覆盖：verify 的**失败路径**同样零写（unknown_label 解析失败的 API/CLI 双路，
含未知目录与 sentinel 前后全文件/账本快照不变）、run 基准根自身 symlink/junction 写入拒绝
（fetch/extract/normalize 同一闸，直接公共 API 同样拒）、公共 `extract_archive` 成员别名与
配置校验共用同一预检（runtime `['a.json','A.JSON']` 不得覆盖）、坏 ZIP/CRC 与前置 drift
统一有界入账、OCNLI 严格多数（并列/票不支持 gold 不得称 majority，如实降级披露）。

r3 窄收口对应覆盖：属性读取唯一接缝 `_lstat_or_missing` 仅把真实 ENOENT 当缺失——标准
`os.lstat` EACCES/非 ENOENT 故障注入在根/祖先/派生 parent/target 一律 fail-closed
（`unreadable_path`），任何 mkdir/`.part`/账本前列出前终止；真实 symlink/junction 根与
祖先照旧零写拒绝；`ocnli_vote_status` 纯锚点与解析器逐行同判（含 `no_majority` 保原
label 端到端）。
依赖主机能力的边界显式 skip，而不是伪装通过。
"""
from __future__ import annotations

import email.message
import hashlib
import io
import json
import os
import socket
import sys
import unicodedata
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from research.memory_relations.public_benchmarks import datasets as D              

LIMITS = {"timeout_seconds": 5, "attempts": 3, "retry_backoff_seconds": 0.0,
          "max_entries": 100, "max_entry_bytes": 1 << 20,
          "max_total_uncompressed_bytes": 1 << 22, "min_free_bytes": 0}
SEVEN_RELATIONS = frozenset({"duplicate", "complement", "contextual", "temporary",
                             "supersede", "conflict", "uncertain"})
FROZEN_LABEL_MAPS = {
    "ocnli": {"entailment": "entailment", "neutral": "neutral", "contradiction": "contradiction"},
    "anli": {"e": "entailment", "n": "neutral", "c": "contradiction"},
    "hans": {"entailment": "entailment", "non-entailment": "non-entailment"},
    "contractnli": {"Entailment": "entailment", "Contradiction": "contradiction",
                    "NotMentioned": "neutral"},
}
FROZEN_UNITS = {"ocnli": "prem_id", "anli": "premise_hash", "hans": "template",
                "contractnli": "doc_id"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network call in offline datasets test")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(urllib.request, "urlopen", denied)


class Response:
    """可复用的假响应模板：每次打开都产生全新 reader（重试不共享已耗尽的 BytesIO）。"""

    def __init__(self, body, *, ctype="application/octet-stream", status=200, headers=None,
                 final_url=None):
        self.body = body
        self.status = status
        self.headers = {"content-type": ctype}
        if headers:
            self.headers.update(headers)
        self.final_url = final_url

    def open(self, url):
        return D.OpenResult(self.status, dict(self.headers), io.BytesIO(self.body),
                            self.final_url if self.final_url is not None else url)


class FakeOpener:
    """按 URL 路由的假 opener；值可为 Response / OpenResult / 异常 / handler 回调。"""

    def __init__(self, routes=None, handler=None):
        self.routes = dict(routes or {})
        self.handler = handler
        self.calls = []

    def __call__(self, url, timeout):
        self.calls.append(url)
        entry = self.handler(url, timeout, len(self.calls)) if self.handler else self.routes[url]
        if isinstance(entry, BaseException):
            raise entry
        if isinstance(entry, Response):
            return entry.open(url)
        if entry.final_url is None:
            entry = entry._replace(final_url=url)
        return entry


class FakeHTTPResponse:
    """urllib 重定向模拟用的假响应（code/msg/info/read/geturl）。"""

    def __init__(self, url, code, headers, body):
        self.url = url
        self.code = code
        self.msg = "OK" if code == 200 else "Redirect"
        self.headers = email.message.Message()
        for key, value in headers.items():
            self.headers[key] = value
        self._body = io.BytesIO(body)

    def info(self):
        return self.headers

    def read(self, size=-1):
        return self._body.read(size)

    def geturl(self):
        return self.url

    def close(self):
        self._body.close()


class FakeTransport(urllib.request.HTTPSHandler):
    """离线 https transport：按 URL 返回 302/200，记录真实被请求过的 URL。"""

    def __init__(self, routes, visited):
        self.routes = dict(routes)
        self.visited = visited

    def https_open(self, request):
        url = request.get_full_url()
        self.visited.append(url)
        route = self.routes.get(url)
        if route is None:
            raise AssertionError("unexpected request to %s" % url)
        status, location, body = route
        headers = {"Location": location} if location else {}
        return FakeHTTPResponse(url, status, headers, body)


def guarded_open(url, routes, allowed):
    visited = []
    opener = D._build_guarded_opener(allowed, [FakeTransport(routes, visited)])
    request = urllib.request.Request(url, headers={"User-Agent": "test"})
    return opener.open(request, timeout=5), visited


def respond(body: bytes, *, ctype="application/octet-stream", status=200, headers=None,
            final_url=None):
    return Response(body, ctype=ctype, status=status, headers=headers, final_url=final_url)


def file_spec(**over):
    spec = {"role": "data", "kind": "direct", "url": "https://cdn.example.test/a.json",
            "dest": "raw/a.json", "max_bytes": 4096, "expect": "jsonl"}
    spec.update(over)
    return spec


def fx_dataset(**over):
    source = {
        "title": "fixture", "home": "https://example.test/home",
        "license": {"name": "CC0", "url": "https://example.test/license"},
        "task": "nli3", "allowed_hosts": ["cdn.example.test"],
        "label_map": {"entailment": "entailment", "neutral": "neutral",
                      "contradiction": "contradiction", "-": None},
        "group_id": {"kind": "field", "field": "prem_id", "fallback": "premise_family_key"},
        "unit_key": "prem_id",
        "files": [], "splits": {},
    }
    source.update(over)
    return source


def fx_config(datasets, **limits_over):
    limits = {**LIMITS, **limits_over}
    return {"schema_version": 1, "default_run": "fx-run", "limits": limits,
            "datasets": datasets}


def write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_pipeline(tmp_path: Path, config: dict, opener, *, validate=True):
    if validate:
        D.validate_config(config)
    config_path = write_config(tmp_path, config)
    run_dir = tmp_path / "run"
    D.stage_fetch(config, run_dir, opener=opener)
    D.stage_extract(config, run_dir)
    manifest = D.normalize_all(config, run_dir, run_dir / "normalized",
                               config_path=config_path)
    return run_dir, manifest, config_path


def zip_bytes(entries: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def ledger_events(run_dir: Path) -> list:
    path = run_dir / "ledger.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_records(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def snapshot_tree(root: Path) -> dict:
    entries = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if path.is_dir():
            entries[rel + "/"] = "dir"
        else:
            entries[rel] = (path.stat().st_size,
                            hashlib.sha256(path.read_bytes()).hexdigest())
    return entries


def set_manifest_field(manifest: dict, path: str, value) -> None:
    parts = path.split(".")
    node = manifest
    for part in parts[:-1]:
        node = node[int(part)] if part.isdigit() else node[part]
    last = parts[-1]
    if last.isdigit():
        node[int(last)] = value
    else:
        node[last] = value


                                                                                

def test_shipped_sources_config_shape_and_frozen_parity():
    config = D.load_config()
    assert set(config["datasets"]) == {"ocnli", "anli", "hans", "contractnli"}
    assert D.GOLD_LABELS.isdisjoint(SEVEN_RELATIONS)
    for name, source in config["datasets"].items():
        assert source["task"] in D.TASK_LABELS
        assert source["unit_key"] == FROZEN_UNITS[name]
        assert {raw: value for raw, value in source["label_map"].items() if value is not None} \
            == FROZEN_LABEL_MAPS[name]
        for spec in source["files"]:
            assert spec["url"].startswith("https://")
            assert len(spec["sha256"]) == 64
            assert D._host_of(spec["url"]) in source["allowed_hosts"]
        for split in source["splits"].values():
            assert split["format"] in D.PARSERS
    assert config["datasets"]["ocnli"]["label_map"]["-"] is None
    hans = config["datasets"]["hans"]
    assert hans["label_map"]["non-entailment"] == "non-entailment"
    assert "contradiction" not in set(hans["label_map"].values())
    contract = config["datasets"]["contractnli"]
    assert contract["label_map"]["NotMentioned"] == "neutral"
    assert contract["expect_hypotheses"] == 17


@pytest.mark.parametrize("mutate, expect_code", [
    (lambda src: src["label_map"].update({"eu": "duplicate"}), "bad_config"),
    (lambda src: src.update({"task": "nli7"}), "bad_config"),
    (lambda src: src.update({"allowed_hosts": []}), "bad_config"),
    (lambda src: src.pop("unit_key"), "bad_config"),
    (lambda src: src["files"][0].update({"url": "http://cdn.example.test/a.json"}), "bad_config"),
    (lambda src: src["files"][0].update({"dest": "../escape.json"}), "bad_relpath"),
    (lambda src: src["files"][0].update({"dest": "raw/a.json "}), "bad_relpath"),
    (lambda src: src["files"][0].update({"expect": "html"}), "bad_config"),
    (lambda src: src["group_id"].update({"kind": "magic"}), "bad_config"),
    (lambda src: src["group_id"].update({"fallback": "uid_prefix"}), "bad_config"),
    (lambda src: src["splits"]["dev"].update({"format": "unknown_fmt"}), "bad_config"),
    (lambda src: src["splits"]["dev"].update({"entry": "x.json"}), "bad_config"),
])
def test_config_rejects_bad_shapes(mutate, expect_code):
    source = fx_dataset(files=[file_spec()],
                        splits={"dev": {"official_split": "dev", "format": "ocnli_jsonl",
                                        "file_role": "data"}})
    mutate(source)
    with pytest.raises(D.DatasetError) as exc:
        D.validate_config(fx_config({"fx": source}))
    assert exc.value.code == expect_code


def test_config_rejects_unsafe_archive_entry():
    source = fx_dataset(
        files=[file_spec(role="archive", kind="archive",
                         url="https://cdn.example.test/a.zip", dest="raw/a.zip",
                         expect="zip", extract={"root": "raw-extracted/fx",
                                                "entries": ["../evil.txt"]})],
        splits={"dev": {"official_split": "dev", "format": "ocnli_jsonl",
                        "entry": "../evil.txt"}})
    with pytest.raises(D.DatasetError) as exc:
        D.validate_config(fx_config({"fx": source}))
    assert exc.value.code == "bad_entry_name"


def test_config_rejects_escape_and_alias_extract_roots():
    traversal = fx_dataset(
        files=[file_spec(role="archive", kind="archive",
                         url="https://cdn.example.test/a.zip", dest="raw/a.zip",
                         expect="zip", extract={"root": "../outside-run",
                                                "entries": ["x.jsonl"]})],
        splits={"dev": {"official_split": "dev", "format": "ocnli_jsonl",
                        "entry": "x.jsonl"}})
    with pytest.raises(D.DatasetError) as exc:
        D.validate_config(fx_config({"fx": traversal}))
    assert exc.value.code == "bad_relpath"
    alias = fx_dataset(
        files=[file_spec(role="archive", kind="archive",
                         url="https://cdn.example.test/a.zip", dest="raw/a.zip",
                         expect="zip", extract={"root": "raw-extracted/fx",
                                                "entries": ["a.json", "A.JSON"]})],
        splits={"dev": {"official_split": "dev", "format": "ocnli_jsonl",
                        "entry": "a.json"}})
    with pytest.raises(D.DatasetError) as exc:
        D.validate_config(fx_config({"fx": alias}))
    assert exc.value.code == "bad_config"


def test_config_rejects_join_on_archive_role():
    source = fx_dataset(
        files=[file_spec(role="archive", kind="archive",
                         url="https://cdn.example.test/a.zip", dest="raw/a.zip",
                         expect="zip", extract={"root": "raw-extracted/fx",
                                                "entries": ["x.jsonl"]})],
        splits={"dev": {"official_split": "dev", "format": "ocnli_jsonl", "entry": "x.jsonl"}},
        joins=[{"role": "archive", "format": "jsonl_keyed", "key_field": "uid",
                "value_field": "v", "metadata_key": "v"}])
    with pytest.raises(D.DatasetError) as exc:
        D.validate_config(fx_config({"fx": source}))
    assert exc.value.code == "bad_config"


                                                                                   

def test_fetch_ok_writes_file_and_ledger(tmp_path):
    body = b'{"a": 1}\n'
    digest = hashlib.sha256(body).hexdigest()
    spec = file_spec(sha256=digest)
    run_dir = tmp_path / "run"
    opener = FakeOpener({spec["url"]: respond(body, ctype="application/json")})
    entry = D.fetch_file(spec, run_dir, dataset="fx", allowed_hosts=["cdn.example.test"],
                         limits=LIMITS, ledger_path=run_dir / "ledger.jsonl", opener=opener)
    assert entry["sha256"] == digest and entry["bytes"] == len(body)
    assert (run_dir / "raw/a.json").read_bytes() == body
    assert [event["status"] for event in ledger_events(run_dir)] == ["ok"]
    assert opener.calls == [spec["url"]]


def test_fetch_skips_cached_when_pin_matches(tmp_path):
    body = b'{"a": 1}\n'
    digest = hashlib.sha256(body).hexdigest()
    spec = file_spec(sha256=digest)
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "raw/a.json").write_bytes(body)
    opener = FakeOpener({})
    entry = D.fetch_file(spec, run_dir, dataset="fx", allowed_hosts=["cdn.example.test"],
                         limits=LIMITS, ledger_path=run_dir / "ledger.jsonl", opener=opener)
    assert entry["skipped"] is True and entry["attempts"] == 0
    assert opener.calls == []


def test_fetch_pin_mismatch_fails_closed(tmp_path):
    spec = file_spec(sha256="0" * 64)
    run_dir = tmp_path / "run"
    opener = FakeOpener({spec["url"]: respond(b'{"a": 1}\n')})
    with pytest.raises(D.DatasetError) as exc:
        D.fetch_file(spec, run_dir, dataset="fx", allowed_hosts=["cdn.example.test"],
                     limits=LIMITS, ledger_path=run_dir / "ledger.jsonl", opener=opener)
    assert exc.value.code == "fetch_failed:sha256_mismatch"
    assert not (run_dir / "raw/a.json").exists()
    assert not (run_dir / "raw/a.json.part").exists()
    assert [event["status"] for event in ledger_events(run_dir)] == ["error"] * 3


def test_fetch_retries_then_succeeds(tmp_path):
    body = b'{"a": 1}\n'
    spec = file_spec(sha256=hashlib.sha256(body).hexdigest())
    run_dir = tmp_path / "run"

    def handler(url, timeout, count):
        if count < 3:
            raise urllib.error.URLError("temporary")
        return respond(body)

    entry = D.fetch_file(spec, run_dir, dataset="fx", allowed_hosts=["cdn.example.test"],
                         limits=LIMITS, ledger_path=run_dir / "ledger.jsonl",
                         opener=FakeOpener(handler=handler))
    assert entry["attempts"] == 3
    assert [event["status"] for event in ledger_events(run_dir)] == ["error", "error", "ok"]


def test_fetch_permanent_failure_records_all_attempts(tmp_path):
    spec = file_spec()
    run_dir = tmp_path / "run"

    def handler(url, timeout, count):
        raise urllib.error.URLError("down")

    with pytest.raises(D.DatasetError) as exc:
        D.fetch_file(spec, run_dir, dataset="fx", allowed_hosts=["cdn.example.test"],
                     limits=LIMITS, ledger_path=run_dir / "ledger.jsonl",
                     opener=FakeOpener(handler=handler))
    assert exc.value.code == "fetch_failed:network_error"
    assert len(ledger_events(run_dir)) == 3


@pytest.mark.parametrize("response, code", [
    (respond(b"<html>nope</html>", ctype="text/html; charset=utf-8"), "html_error_page"),
    (respond(b"<!DOCTYPE html><html>", ctype="text/plain"), "html_error_page"),
    (respond(b"", ctype="application/json"), "empty_body"),
    (respond(b"not jsonl", ctype="application/json"), "not_jsonl"),
])
def test_fetch_rejects_error_pages_and_wrong_magic(tmp_path, response, code):
    spec = file_spec()
    run_dir = tmp_path / "run"
    with pytest.raises(D.DatasetError) as exc:
        D.fetch_file(spec, run_dir, dataset="fx", allowed_hosts=["cdn.example.test"],
                     limits=LIMITS, ledger_path=run_dir / "ledger.jsonl",
                     opener=FakeOpener({spec["url"]: response}))
    assert exc.value.code == "fetch_failed:" + code


def test_fetch_rejects_non_zip_body(tmp_path):
    spec = file_spec(expect="zip", max_bytes=1024)
    run_dir = tmp_path / "run"
    with pytest.raises(D.DatasetError) as exc:
        D.fetch_file(spec, run_dir, dataset="fx", allowed_hosts=["cdn.example.test"],
                     limits=LIMITS, ledger_path=run_dir / "ledger.jsonl",
                     opener=FakeOpener({spec["url"]: respond(b"plain text")}))
    assert exc.value.code == "fetch_failed:not_zip"


def test_fetch_rejects_oversize_declared_and_streamed(tmp_path):
    spec = file_spec(max_bytes=64)
    run_dir = tmp_path / "run"
    declared = respond(b"x" * 16, headers={"content-length": "4096"})
    with pytest.raises(D.DatasetError) as exc:
        D.fetch_file(spec, run_dir, dataset="fx", allowed_hosts=["cdn.example.test"],
                     limits=LIMITS, ledger_path=run_dir / "ledger.jsonl",
                     opener=FakeOpener({spec["url"]: declared}))
    assert exc.value.code == "fetch_failed:oversize_declared"
    streamed = respond(b"x" * 4096)
    with pytest.raises(D.DatasetError) as exc:
        D.fetch_file(spec, run_dir, dataset="fx", allowed_hosts=["cdn.example.test"],
                     limits=LIMITS, ledger_path=run_dir / "ledger.jsonl",
                     opener=FakeOpener({spec["url"]: streamed}))
    assert exc.value.code == "fetch_failed:oversize"


def test_fetch_rejects_host_and_final_url_drift(tmp_path):
    spec = file_spec(url="https://evil.example.test/a.json")
    run_dir = tmp_path / "run"
    with pytest.raises(D.DatasetError) as exc:
        D.fetch_file(spec, run_dir, dataset="fx", allowed_hosts=["cdn.example.test"],
                     limits=LIMITS, ledger_path=run_dir / "ledger.jsonl",
                     opener=FakeOpener({}))
    assert exc.value.code == "fetch_failed:host_not_allowed"
    spec = file_spec()
    with pytest.raises(D.DatasetError) as exc:
        D.fetch_file(spec, run_dir, dataset="fx", allowed_hosts=["cdn.example.test"],
                     limits=LIMITS, ledger_path=run_dir / "ledger.jsonl",
                     opener=FakeOpener({spec["url"]: respond(
                         b"{}", final_url="https://evil.example.test/a.json")}))
    assert exc.value.code == "fetch_failed:host_not_allowed"


def test_fetch_rejects_bad_status(tmp_path):
    spec = file_spec()
    run_dir = tmp_path / "run"
    with pytest.raises(D.DatasetError) as exc:
        D.fetch_file(spec, run_dir, dataset="fx", allowed_hosts=["cdn.example.test"],
                     limits=LIMITS, ledger_path=run_dir / "ledger.jsonl",
                     opener=FakeOpener({spec["url"]: respond(b"nope", status=404)}))
    assert exc.value.code == "fetch_failed:http_status"


def test_fetch_rejects_existing_part_and_hardlinked_target(tmp_path):
    body = b'{"a": 1}\n'
    spec = file_spec()
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "raw/a.json.part").write_bytes(b"KEEP")
    opener = FakeOpener({spec["url"]: respond(body)})
    with pytest.raises(D.DatasetError) as exc:
        D.fetch_file(spec, run_dir, dataset="fx", allowed_hosts=["cdn.example.test"],
                     limits=LIMITS, ledger_path=run_dir / "ledger.jsonl", opener=opener)
    assert exc.value.code == "fetch_failed:part_exists"
    assert (run_dir / "raw/a.json.part").read_bytes() == b"KEEP"
    assert not (run_dir / "raw/a.json").exists()
    assert len(opener.calls) == LIMITS["attempts"]


def test_fetch_rejects_hardlinked_target(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"SAFE")
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    try:
        os.link(outside, run_dir / "raw/a.json")
    except OSError:
        pytest.skip("hardlink not permitted on this host")
    spec = file_spec()
    with pytest.raises(D.DatasetError) as exc:
        D.fetch_file(spec, run_dir, dataset="fx", allowed_hosts=["cdn.example.test"],
                     limits=LIMITS, ledger_path=run_dir / "ledger.jsonl",
                     opener=FakeOpener({spec["url"]: respond(b'{"a": 1}\n')}))
    assert exc.value.code == "fetch_failed:hardlinked_target"
    assert outside.read_bytes() == b"SAFE"


                                                                                     

def test_redirect_guard_validates_each_hop_before_request():
    allowed = ["allowed.example"]
    start = "https://allowed.example/start"
    second = "https://allowed.example/end"
    response, visited = guarded_open(start, {start: (302, second, b""),
                                             second: (200, None, b"hello")}, allowed)
    assert response.read() == b"hello"
    assert visited == [start, second]

    blocked = "https://blocked.example/secret"
    with pytest.raises(D.DatasetError) as exc:
        guarded_open(start, {start: (302, blocked, b"")}, allowed)
    assert exc.value.code == "host_not_allowed"

    start3 = "https://allowed.example/s3"
    with pytest.raises(D.DatasetError) as exc:
        guarded_open(start3, {start3: (302, "http://allowed.example/insecure", b"")}, allowed)
    assert exc.value.code == "bad_url"

    with pytest.raises(D.DatasetError) as exc:
        D._check_redirect_target("ftp://allowed.example/x", allowed)
    assert exc.value.code == "bad_url"


def test_redirect_guard_blocks_mid_chain_blocked_host_before_request():
    allowed = ["allowed.example"]
    start = "https://allowed.example/start"
    hop = "https://blocked.example/hop"
    end = "https://allowed.example/end"
    visited = []
    opener = D._build_guarded_opener(allowed, [FakeTransport(
        {start: (302, hop, b""), hop: (302, end, b""), end: (200, None, b"ok")}, visited)])
    request = urllib.request.Request(start, headers={"User-Agent": "test"})
    with pytest.raises(D.DatasetError) as exc:
        opener.open(request, timeout=5)
    assert exc.value.code == "host_not_allowed"
    assert visited == [start]


def test_redirect_guard_limits_hop_count():
    allowed = ["a.example"]
    routes = {}
    for index in range(8):
        routes["https://a.example/r%d" % index] = (302, "https://a.example/r%d" % (index + 1), b"")
    routes["https://a.example/r8"] = (200, None, b"ok")
    visited = []
    opener = D._build_guarded_opener(allowed, [FakeTransport(routes, visited)])
    request = urllib.request.Request("https://a.example/r0", headers={"User-Agent": "test"})
    with pytest.raises((D.DatasetError, urllib.error.HTTPError)) as exc:
        opener.open(request, timeout=5)
    if isinstance(exc.value, D.DatasetError):
        assert exc.value.code == "too_many_redirects"
    assert len(visited) <= D._MAX_REDIRECTS + 2


                                                                                     

def test_extract_writes_allowlisted_entry(tmp_path):
    archive = tmp_path / "a.zip"
    archive.write_bytes(zip_bytes({"dir/ok.txt": b"hello"}))
    dest = tmp_path / "out"
    records = D.extract_archive(archive, dest, ["dir/ok.txt"], limits=LIMITS, dataset="fx",
                                ledger_path=tmp_path / "ledger.jsonl", run_dir=tmp_path)
    assert (dest / "dir/ok.txt").read_bytes() == b"hello"
    assert records[0]["sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert ledger_events(tmp_path)[0]["event"] == "extract"


@pytest.mark.parametrize("name", [
    "../evil.txt", "/abs.txt", "C:/drive.txt", "a/../../b.txt", "a\\b.txt", "..",
    "a//b.txt", "CON.txt", "sub/CON/x.txt", "d/", "dir/a.txt:stream", "a.", "a ",
    "COM\u00b9.txt", "com1.txt", "aux", "LPT9.log", "nul.tar.gz",
])
def test_extract_rejects_unsafe_names(tmp_path, name):
    archive = tmp_path / "a.zip"
    archive.write_bytes(zip_bytes({name: b"x"}))
    dest = tmp_path / "out"
    with pytest.raises(D.DatasetError) as exc:
        D.extract_archive(archive, dest, [name], limits=LIMITS, dataset="fx",
                          ledger_path=tmp_path / "ledger.jsonl", run_dir=tmp_path)
    assert exc.value.code == "bad_entry_name"
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path / "abs.txt").exists()


def test_extract_rejects_duplicate_central_entries(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        handle.writestr("ok.json", b"first")
        handle.writestr("ok.json", b"second")
    archive = tmp_path / "a.zip"
    archive.write_bytes(buffer.getvalue())
    with pytest.raises(D.DatasetError) as exc:
        D.extract_archive(archive, tmp_path / "out", ["ok.json"], limits=LIMITS, dataset="fx",
                          ledger_path=tmp_path / "ledger.jsonl", run_dir=tmp_path)
    assert exc.value.code == "duplicate_entry"
    assert not (tmp_path / "out/ok.json").exists()


def test_extract_rejects_symlink_entry(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        info = zipfile.ZipInfo("trap/link")
        info.external_attr = (0o120777 << 16)
        handle.writestr(info, b"target")
    with pytest.raises(D.DatasetError) as exc:
        D.extract_archive(archive, tmp_path / "out", ["trap/link"], limits=LIMITS,
                          dataset="fx", ledger_path=tmp_path / "ledger.jsonl",
                          run_dir=tmp_path)
    assert exc.value.code == "entry_symlink"


def test_extract_rejects_symlinked_parent_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    dest = tmp_path / "out"
    dest.mkdir()
    try:
        (dest / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink not permitted on this host")
    archive = tmp_path / "a.zip"
    archive.write_bytes(zip_bytes({"link/f.txt": b"x"}))
    with pytest.raises(D.DatasetError) as exc:
        D.extract_archive(archive, dest, ["link/f.txt"], limits=LIMITS, dataset="fx",
                          ledger_path=tmp_path / "ledger.jsonl", run_dir=tmp_path)
    assert exc.value.code == "unsafe_path"
    assert not (outside / "f.txt").exists()


def test_extract_rejects_root_outside_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    archive = tmp_path / "a.zip"
    archive.write_bytes(zip_bytes({"ok.txt": b"x"}))
    with pytest.raises(D.DatasetError) as exc:
        D.extract_archive(archive, tmp_path / "outside", ["ok.txt"], limits=LIMITS,
                          dataset="fx", ledger_path=tmp_path / "ledger.jsonl",
                          run_dir=run_dir)
    assert exc.value.code == "path_escape"
    assert not (tmp_path / "outside" / "ok.txt").exists()


def test_extract_rejects_missing_and_oversize(tmp_path):
    archive = tmp_path / "a.zip"
    archive.write_bytes(zip_bytes({"ok.txt": b"x" * 4096}))
    with pytest.raises(D.DatasetError) as exc:
        D.extract_archive(archive, tmp_path / "out", ["nope.txt"], limits=LIMITS, dataset="fx",
                          ledger_path=tmp_path / "ledger.jsonl", run_dir=tmp_path)
    assert exc.value.code == "entry_missing"
    tight = {**LIMITS, "max_entry_bytes": 8192, "max_total_uncompressed_bytes": 1024}
    with pytest.raises(D.DatasetError) as exc:
        D.extract_archive(archive, tmp_path / "out", ["ok.txt"], limits=tight, dataset="fx",
                          ledger_path=tmp_path / "ledger.jsonl", run_dir=tmp_path)
    assert exc.value.code == "entry_oversize"
    assert not (tmp_path / "out/ok.txt").exists()


def test_extract_ignores_non_allowlisted_entries(tmp_path):
    archive = tmp_path / "a.zip"
    archive.write_bytes(zip_bytes({"ok.txt": b"ok", "../sneaky.txt": b"bad"}))
    dest = tmp_path / "out"
    D.extract_archive(archive, dest, ["ok.txt"], limits=LIMITS, dataset="fx",
                      ledger_path=tmp_path / "ledger.jsonl", run_dir=tmp_path)
    assert (dest / "ok.txt").read_bytes() == b"ok"
    assert not (tmp_path / "sneaky.txt").exists()


def test_extract_rejects_entry_count_cap(tmp_path):
    archive = tmp_path / "a.zip"
    archive.write_bytes(zip_bytes({"a.txt": b"a", "b.txt": b"b"}))
    tight = {**LIMITS, "max_entries": 1}
    with pytest.raises(D.DatasetError) as exc:
        D.extract_archive(archive, tmp_path / "out", ["a.txt"], limits=tight, dataset="fx",
                          ledger_path=tmp_path / "ledger.jsonl", run_dir=tmp_path)
    assert exc.value.code == "archive_too_many_entries"


def test_extract_rejects_existing_part_hardlink(tmp_path):
    outside = tmp_path / "sentinel.txt"
    outside.write_bytes(b"SAFE")
    dest = tmp_path / "out"
    dest.mkdir()
    try:
        os.link(outside, dest / "ok.txt.part")
    except OSError:
        pytest.skip("hardlink not permitted on this host")
    archive = tmp_path / "a.zip"
    archive.write_bytes(zip_bytes({"ok.txt": b"payload"}))
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(D.DatasetError) as exc:
        D.extract_archive(archive, dest, ["ok.txt"], limits=LIMITS, dataset="fx",
                          ledger_path=ledger, run_dir=tmp_path)
    assert exc.value.code == "part_exists"
    assert outside.read_bytes() == b"SAFE"
    events = ledger_events(tmp_path)
    assert events and events[-1]["status"] == "error" and events[-1]["code"] == "part_exists"


def test_extract_failure_writes_bounded_ledger(tmp_path):
    archive = tmp_path / "a.zip"
    archive.write_bytes(zip_bytes({"ok.txt": b"x"}))
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(D.DatasetError):
        D.extract_archive(archive, tmp_path / "out", ["missing.txt"], limits=LIMITS,
                          dataset="fx", ledger_path=ledger, run_dir=tmp_path)
    events = ledger_events(tmp_path)
    assert events[-1]["event"] == "extract" and events[-1]["status"] == "error"
    assert events[-1]["code"] == "entry_missing" and events[-1]["partial"] == 0


                                                                                   

def ocnli_rows():
    return [
        {"level": "easy", "sentence1": "前提甲", "sentence2": "假设甲", "label": "entailment",
         "genre": "lit", "prem_id": "lit_1", "id": 0, "label0": "entailment",
         "label1": "entailment", "label2": "entailment", "label3": "entailment",
         "label4": "entailment"},
        {"sentence1": "前提乙", "sentence2": "假设乙", "label": "neutral", "id": 1,
         "label0": "neutral", "label1": None, "label2": None, "label3": None, "label4": None},
        {"sentence1": "前提丙", "sentence2": "假设丙", "label": "neutral", "id": 2,
         "label0": None, "label1": None, "label2": None, "label3": None, "label4": None},
        {"sentence1": "前提丁", "sentence2": "假设丁", "label": "-", "id": 3, "prem_id": "lit_1",
         "label0": "entailment", "label1": "neutral", "label2": "contradiction",
         "label3": "unknown", "label4": "unrelated"},
        {"sentence1": "前提戊", "sentence2": "假设戊", "id": 4},
        {"sentence1": "前提己", "sentence2": "假设己", "label": "entailment", "id": 5,
         "label0": "entailment", "label1": "entailment", "label2": "entailment",
         "label3": "entailment", "label4": "entailment"},
    ]


def ocnli_config(url: str, dest: str = "raw/ocnli-dev.json", sha=None):
    spec = {"role": "dev", "kind": "direct", "url": url, "dest": dest, "expect": "jsonl",
            "max_bytes": 1 << 20}
    if sha:
        spec["sha256"] = sha
    return fx_config({"ocnli": fx_dataset(
        files=[spec],
        splits={"dev": {"official_split": "dev", "format": "ocnli_jsonl",
                        "file_role": "dev"}})})


def test_ocnli_vote_semantics_and_null_prem_id(tmp_path):
    rows = ocnli_rows()
    body = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode("utf-8")
    url = "https://cdn.example.test/ocnli-dev.json"
    config = ocnli_config(url, sha=hashlib.sha256(body).hexdigest())
    run_dir, manifest, _config_path = run_pipeline(
        tmp_path, config, FakeOpener({url: respond(body, ctype="application/json")}))
    split = manifest["datasets"]["ocnli"]["splits"]["dev"]
    assert split["counts"]["rows"] == 6
    assert split["counts"]["gold_label"] == {"entailment": 2, "neutral": 2}
    assert split["counts"]["gold_status"] == {"majority": 2, "single": 2, "no_consensus": 1,
                                              "missing": 1}
    assert split["counts"]["group_id_source"] == {"raw": 2, "derived_premise_family": 4}
    assert split["counts"]["votes_valid"] == {"5": 2, "1": 1, "0": 2, "3": 1}
    assert split["counts"]["missing_gold"] == 2 and split["counts"]["no_consensus"] == 1
    records = read_records(run_dir / "normalized/ocnli.dev.jsonl")
    first, second, third, fourth, fifth, sixth = records
    assert first["gold_label"] == "entailment" and first["group_id"] == "lit_1"
    assert first["metadata"]["gold_status"] == "majority"
    assert first["metadata"]["label_method"] == "majority_vote"
    assert first["metadata"]["prem_id"] == "lit_1"
    assert first["metadata"]["group_id_source"] == "raw"
    assert second["metadata"]["gold_status"] == "single"
    assert second["metadata"]["label_method"] == "single_annotation"
    assert third["metadata"]["gold_status"] == "single"
    assert fourth["gold_label"] is None
    assert fourth["metadata"]["gold_status"] == "no_consensus"
    assert fourth["metadata"]["votes"] == ["entailment", "neutral", "contradiction",
                                           "unknown", "unrelated"]
    assert fourth["metadata"]["votes_valid_count"] == 3
    assert fifth["gold_label"] is None and fifth["metadata"]["gold_status"] == "missing"
    assert fifth["metadata"]["prem_id"] is None
    assert fifth["metadata"]["group_id_source"] == "derived_premise_family"
    assert fifth["group_id"] == D.premise_family_key("前提戊")
                                                                     
    assert sixth["gold_label"] == "entailment"
    assert sixth["metadata"]["prem_id"] is None
    assert sixth["metadata"]["group_id_source"] == "derived_premise_family"
    assert sixth["group_id"] == D.premise_family_key("前提己")


def test_ocnli_unknown_label_fails_closed_and_logs(tmp_path):
    rows = [{"sentence1": "p", "sentence2": "h", "label": "maybe", "id": 0, "prem_id": "g"}]
    body = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
    url = "https://cdn.example.test/ocnli-dev.json"
    config = ocnli_config(url)
    config_path = write_config(tmp_path, config)
    run_dir = tmp_path / "run"
    D.stage_fetch(config, run_dir, opener=FakeOpener({url: respond(body)}))
    with pytest.raises(D.DatasetError) as exc:
        D.normalize_all(config, run_dir, run_dir / "normalized", config_path=config_path)
    assert exc.value.code == "unknown_label"
    events = ledger_events(run_dir)
    assert events[-1]["event"] == "normalize" and events[-1]["status"] == "error"
    assert events[-1]["code"] == "unknown_label" and events[-1]["partial"] == 0


def test_premise_family_key_matches_protocol_algorithm():
    text = "Same  premise.\u00a0Extra"
    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    expected = "premise:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    assert D.premise_family_key(text) == expected
    assert D.premise_family_key("Same premise.") == D.premise_family_key("Same  premise. ")
    with pytest.raises(D.DatasetError) as exc:
        D.premise_family_key("")
    assert exc.value.code == "empty_premise"


                                                                                  

def anli_source(archive_url: str, join_url: str):
    return fx_dataset(
        task="nli3",
        label_map={"e": "entailment", "n": "neutral", "c": "contradiction"},
        group_id={"kind": "premise_family_key"}, unit_key="premise_hash",
        joins=[{"role": "verifier_labels", "format": "jsonl_keyed", "key_field": "uid",
                "value_field": "verifier labels", "metadata_key": "verifier_labels"}],
        files=[
            {"role": "archive", "kind": "archive", "url": archive_url,
             "dest": "raw/anli.zip", "max_bytes": 1 << 20, "expect": "zip",
             "extract": {"root": "raw-extracted/anli",
                         "entries": ["anli_v1.0/R1/dev.jsonl"]}},
            {"role": "verifier_labels", "kind": "direct", "url": join_url,
             "dest": "raw/verifier.jsonl", "max_bytes": 1 << 20, "expect": "jsonl"},
        ],
        splits={"R1/dev": {"official_split": "R1/dev", "format": "anli_jsonl",
                           "entry": "anli_v1.0/R1/dev.jsonl"}})


def test_anli_verifier_agreement_semantics(tmp_path):
    rows = [
        {"uid": "u1", "context": "ctx one", "hypothesis": "h one", "label": "e",
         "reason": "为什么", "genre": "wikipedia", "emturk": False, "tag": "r1_dev",
         "model_label": "n"},
        {"uid": "u2", "context": "ctx two", "hypothesis": "h two", "label": "e"},
        {"uid": "u3", "context": "ctx three", "hypothesis": "h three", "label": "e"},
        {"uid": "u4", "context": "ctx four", "hypothesis": "h four", "label": "c"},
    ]
    body = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
    join_body = b"\n".join([
        json.dumps({"uid": "u1", "verifier labels": ["e", "e"]}).encode(),
        json.dumps({"uid": "u2", "verifier labels": ["e", "c"]}).encode(),
        json.dumps({"uid": "u3", "verifier labels": ["c", "c"]}).encode(),
    ]) + b"\n"
    archive_url = "https://cdn.example.test/anli.zip"
    join_url = "https://cdn.example.test/verifier.jsonl"
    config = fx_config({"anli": anli_source(archive_url, join_url)})
    run_dir, manifest, _config_path = run_pipeline(
        tmp_path, config, FakeOpener({archive_url: respond(zip_bytes(
            {"anli_v1.0/R1/dev.jsonl": body})), join_url: respond(join_body)}))
    split = manifest["datasets"]["anli"]["splits"]["R1/dev"]
    assert split["counts"]["rows"] == 4
    assert split["counts"]["gold_status"] == {"adjudicated": 1, "single": 3}
    assert split["counts"]["join_present"] == {"verifier_labels": 3}
    records = read_records(run_dir / "normalized/anli.R1-dev.jsonl")
    methods = [record["metadata"]["label_method"] for record in records]
    assert methods == ["writer_with_verifiers", "writer_only", "writer_only", "writer_only"]
    agree_counts = [record["metadata"]["verifier_agree_count"] for record in records]
    assert agree_counts == [2, 1, 0, None]
    assert records[0]["metadata"]["gold_status"] == "adjudicated"
    assert records[1]["metadata"]["gold_status"] == "single"
    assert records[0]["metadata"]["round"] == "r1"
    assert records[0]["metadata"]["verifier_labels"] == ["e", "e"]
    assert records[0]["metadata"]["group_id_source"] == "derived_premise_family"
    assert records[3]["metadata"]["verifier_labels"] is None


def test_anli_unknown_label_char_fails(tmp_path):
    body = json.dumps({"uid": "x", "context": "c", "hypothesis": "h", "label": "z"}).encode()
    url = "https://cdn.example.test/dev.jsonl"
    config = fx_config({"anli": fx_dataset(
        label_map={"e": "entailment", "n": "neutral", "c": "contradiction"},
        group_id={"kind": "premise_family_key"}, unit_key="premise_hash",
        files=[{"role": "dev", "kind": "direct", "url": url, "dest": "raw/dev.jsonl",
                "max_bytes": 1 << 20, "expect": "jsonl"}],
        splits={"R1/dev": {"official_split": "R1/dev", "format": "anli_jsonl",
                           "file_role": "dev"}})})
    config_path = write_config(tmp_path, config)
    run_dir = tmp_path / "run"
    D.stage_fetch(config, run_dir, opener=FakeOpener({url: respond(body)}))
    with pytest.raises(D.DatasetError) as exc:
        D.normalize_all(config, run_dir, run_dir / "normalized", config_path=config_path)
    assert exc.value.code == "unknown_label"


                                                                                  

HANS_HEADER = ("gold_label\tsentence1_binary_parse\tsentence2_binary_parse\tsentence1_parse"
               "\tsentence2_parse\tsentence1\tsentence2\tpairID\theuristic\tsubcase\ttemplate")


def hans_txt(rows):
    lines = [HANS_HEADER]
    for row in rows:
        lines.append("\t".join([row["gold_label"], "(p)", "(h)", "(P)", "(H)", row["sentence1"],
                                row["sentence2"], row["pairID"], row["heuristic"],
                                row["subcase"], row["template"]]))
    return ("\n".join(lines) + "\n").encode("utf-8")


def hans_config(txt_url: str, jsonl_url: str, sha=None):
    txt_spec = {"role": "evaluation", "kind": "direct", "url": txt_url,
                "dest": "raw/hans-eval.txt", "max_bytes": 1 << 20, "expect": "text"}
    if sha:
        txt_spec["sha256"] = sha
    return fx_config({"hans": fx_dataset(
        task="entailment2",
        label_map={"entailment": "entailment", "non-entailment": "non-entailment"},
        group_id={"kind": "field", "field": "template"}, unit_key="template",
        files=[txt_spec,
               {"role": "evaluation_jsonl", "kind": "direct", "url": jsonl_url,
                "dest": "raw/hans-eval.jsonl", "max_bytes": 1 << 20, "expect": "jsonl"}],
        splits={"evaluation": {"official_split": "evaluation", "format": "hans_tsv",
                               "file_role": "evaluation",
                               "crosscheck_role": "evaluation_jsonl"}})})


def hans_rows():
    return [
        {"gold_label": "entailment", "sentence1": "The president advised the doctor .",
         "sentence2": "The president advised the doctor .", "pairID": "ex0",
         "heuristic": "lexical_overlap", "subcase": "ln_subject/object_swap", "template": "temp1"},
        {"gold_label": "non-entailment", "sentence1": "The president advised the doctor .",
         "sentence2": "The doctor advised the president .", "pairID": "ex1",
         "heuristic": "lexical_overlap", "subcase": "ln_subject/object_swap", "template": "temp1"},
    ]


def test_hans_generated_labels_and_crosscheck(tmp_path):
    rows = hans_rows()
    txt = hans_txt(rows)
    jsonl = ("\n".join(json.dumps({"pairID": row["pairID"], "gold_label": row["gold_label"]})
                       for row in rows) + "\n").encode()
    txt_url = "https://cdn.example.test/hans-eval.txt"
    jsonl_url = "https://cdn.example.test/hans-eval.jsonl"
    config = hans_config(txt_url, jsonl_url, sha=hashlib.sha256(txt).hexdigest())
    run_dir, manifest, _config_path = run_pipeline(
        tmp_path, config, FakeOpener({txt_url: respond(txt, ctype="text/plain"),
                                      jsonl_url: respond(jsonl)}))
    split = manifest["datasets"]["hans"]["splits"]["evaluation"]
    assert split["counts"]["rows"] == 2
    assert split["counts"]["gold_label"] == {"entailment": 1, "non-entailment": 1}
    assert split["counts"]["gold_status"] == {"generated": 2}
    assert split["omitted_fields"] == ["sentence1_binary_parse", "sentence2_binary_parse",
                                       "sentence1_parse", "sentence2_parse"]
    records = read_records(run_dir / "normalized/hans.evaluation.jsonl")
    assert records[1]["gold_label"] == "non-entailment"
    assert records[1]["gold_label"] != "contradiction"
    assert records[1]["task"] == "entailment2"
    assert records[0]["metadata"]["gold_status"] == "generated"
    assert records[0]["metadata"]["label_method"] == "template_rule"
    assert records[0]["group_id"] == "temp1" and records[0]["metadata"]["group_id_source"] == "raw"


def test_hans_crosscheck_mismatch_fails(tmp_path):
    rows = hans_rows()
    txt = hans_txt(rows)
    jsonl = json.dumps({"pairID": "ex0", "gold_label": "entailment"}).encode() + b"\n"
    txt_url = "https://cdn.example.test/hans-eval.txt"
    jsonl_url = "https://cdn.example.test/hans-eval.jsonl"
    config = hans_config(txt_url, jsonl_url)
    config_path = write_config(tmp_path, config)
    run_dir = tmp_path / "run"
    D.stage_fetch(config, run_dir, opener=FakeOpener({txt_url: respond(txt),
                                                      jsonl_url: respond(jsonl)}))
    with pytest.raises(D.DatasetError) as exc:
        D.normalize_all(config, run_dir, run_dir / "normalized", config_path=config_path)
    assert exc.value.code == "crosscheck_mismatch"


def test_hans_rejects_duplicate_pair_id(tmp_path):
    rows = hans_rows()
    rows[1]["pairID"] = "ex0"
    txt = hans_txt(rows)
    jsonl = json.dumps({"pairID": "ex0", "gold_label": "entailment"}).encode() + b"\n"
    txt_url = "https://cdn.example.test/hans-eval.txt"
    jsonl_url = "https://cdn.example.test/hans-eval.jsonl"
    config = hans_config(txt_url, jsonl_url)
    config_path = write_config(tmp_path, config)
    run_dir = tmp_path / "run"
    D.stage_fetch(config, run_dir, opener=FakeOpener({txt_url: respond(txt),
                                                      jsonl_url: respond(jsonl)}))
    with pytest.raises(D.DatasetError) as exc:
        D.normalize_all(config, run_dir, run_dir / "normalized", config_path=config_path)
    assert exc.value.code == "duplicate_pair_id"


def test_hans_rejects_out_of_space_label(tmp_path):
    rows = [{"gold_label": "neutral", "sentence1": "p", "sentence2": "h", "pairID": "ex0",
             "heuristic": "lexical_overlap", "subcase": "s", "template": "temp1"}]
    txt = hans_txt(rows)
    jsonl = json.dumps({"pairID": "ex0", "gold_label": "neutral"}).encode() + b"\n"
    txt_url = "https://cdn.example.test/hans-eval.txt"
    jsonl_url = "https://cdn.example.test/hans-eval.jsonl"
    config = hans_config(txt_url, jsonl_url)
    config_path = write_config(tmp_path, config)
    run_dir = tmp_path / "run"
    D.stage_fetch(config, run_dir, opener=FakeOpener({txt_url: respond(txt),
                                                      jsonl_url: respond(jsonl)}))
    with pytest.raises(D.DatasetError) as exc:
        D.normalize_all(config, run_dir, run_dir / "normalized", config_path=config_path)
    assert exc.value.code == "unknown_label"


def test_hans_bad_row_and_header_fail(tmp_path):
    txt_url = "https://cdn.example.test/hans-eval.txt"
    jsonl_url = "https://cdn.example.test/hans-eval.jsonl"
    config = hans_config(txt_url, jsonl_url)
    config_path = write_config(tmp_path, config)
    bad_row = (HANS_HEADER + "\nex0\tentailment\n").encode()
    run_dir = tmp_path / "run"
    D.stage_fetch(config, run_dir, opener=FakeOpener({txt_url: respond(bad_row),
                                                      jsonl_url: respond(b"{}")}))
    with pytest.raises(D.DatasetError) as exc:
        D.normalize_all(config, run_dir, run_dir / "normalized", config_path=config_path)
    assert exc.value.code == "bad_row"
    bad_header = (HANS_HEADER.replace("subcase", "SUB") + "\n").encode()
    run_dir2 = tmp_path / "run2"
    D.stage_fetch(config, run_dir2, opener=FakeOpener({txt_url: respond(bad_header),
                                                       jsonl_url: respond(b"{}")}))
    with pytest.raises(D.DatasetError) as exc:
        D.normalize_all(config, run_dir2, run_dir2 / "normalized", config_path=config_path)
    assert exc.value.code == "bad_header"


                                                                                         

def contract_doc(doc_id, hypotheses, *, bad_index=None):
    annotations = {}
    for index, hyp_id in enumerate(hypotheses):
        if bad_index is not None and index == 0:
            annotations[hyp_id] = {"choice": "Entailment", "spans": [bad_index]}
        elif index % 2 == 0:
            annotations[hyp_id] = {"choice": "Entailment", "spans": [0]}
        else:
            annotations[hyp_id] = {"choice": "NotMentioned", "spans": []}
    return {"id": doc_id, "file_name": "f%d.pdf" % doc_id, "text": "契约文本%d正文" % doc_id,
            "spans": [[0, 3], [3, 6]], "annotation_sets": [{"annotations": annotations}],
            "document_type": "search-pdf", "url": "https://example.test/%d" % doc_id}


def contract_config(*, expect_hypotheses=2, archive_url=None):
    archive_url = archive_url or "https://cdn.example.test/contract.zip"
    return fx_config({"contractnli": fx_dataset(
        task="nli3",
        label_map={"Entailment": "entailment", "Contradiction": "contradiction",
                   "NotMentioned": "neutral"},
        group_id={"kind": "field", "field": "doc_id"}, unit_key="doc_id",
        expect_hypotheses=expect_hypotheses,
        files=[{"role": "archive", "kind": "archive", "url": archive_url,
                "dest": "raw/contract.zip", "max_bytes": 1 << 20, "expect": "zip",
                "extract": {"root": "raw-extracted/contractnli",
                            "entries": ["contract-nli/train.json"]}}],
        splits={"train": {"official_split": "train", "format": "contractnli_json",
                          "entry": "contract-nli/train.json"}})})


def contract_body(documents, hypotheses):
    return json.dumps({
        "documents": documents,
        "labels": {hyp: {"short_description": "s-%s" % hyp, "hypothesis": "假设 %s" % hyp}
                   for hyp in hypotheses},
    }, ensure_ascii=False).encode("utf-8")


def test_contract_records_full_hypotheses_and_spans(tmp_path):
    hypotheses = ["nda-1", "nda-2"]
    body = contract_body([contract_doc(7, hypotheses), contract_doc(9, hypotheses)],
                         hypotheses)
    archive_url = "https://cdn.example.test/contract.zip"
    config = contract_config(archive_url=archive_url)
    run_dir, manifest, _config_path = run_pipeline(
        tmp_path, config, FakeOpener({archive_url: respond(zip_bytes(
            {"contract-nli/train.json": body}))}))
    split = manifest["datasets"]["contractnli"]["splits"]["train"]
    assert split["counts"]["rows"] == 4
    assert split["counts"]["gold_label"] == {"entailment": 2, "neutral": 2}
    assert split["counts"]["gold_status"] == {"single": 4}
    assert split["counts"]["groups"] == 2
    records = read_records(run_dir / "normalized/contractnli.train.jsonl")
    first = records[0]
    assert first["id"] == "contractnli:train:7:nda-1"
    assert first["premise"] == "契约文本7正文"
    assert first["hypothesis"] == "假设 nda-1"
    assert first["gold_label"] == "entailment"
    assert first["group_id"] == "7" and first["metadata"]["doc_id"] == "7"
    assert first["metadata"]["hypothesis_id"] == "nda-1"
    assert first["metadata"]["spans"] == [{"text": "契约文", "start": 0, "end": 3}]
    assert first["metadata"]["gold_status"] == "single"
    assert first["metadata"]["label_method"] == "single_annotation"
    assert first["metadata"]["group_id_source"] == "raw"
    assert first["evidence"]["row"] == "doc:7/hyp:nda-1"
    assert first["evidence"]["source_url"] == archive_url
    neutral = records[1]
    assert neutral["gold_label"] == "neutral"
    assert neutral["metadata"]["original_label"] == "NotMentioned"
    assert neutral["metadata"]["spans"] == []


def test_contract_whitespace_span_is_skipped_and_counted(tmp_path):
    document = {"id": 7, "file_name": "f.pdf", "text": "契约 文本",
                "spans": [[0, 2], [2, 3], [3, 5]],
                "annotation_sets": [{"annotations": {
                    "nda-1": {"choice": "Entailment", "spans": [0, 1]},
                    "nda-2": {"choice": "Entailment", "spans": [1]}}}],
                "document_type": "search-pdf", "url": "https://example.test/7"}
    hypotheses = ["nda-1", "nda-2"]
    body = contract_body([document], hypotheses)
    archive_url = "https://cdn.example.test/contract.zip"
    config = contract_config(archive_url=archive_url)
    run_dir, manifest, _config_path = run_pipeline(
        tmp_path, config, FakeOpener({archive_url: respond(zip_bytes(
            {"contract-nli/train.json": body}))}))
    split = manifest["datasets"]["contractnli"]["splits"]["train"]
    assert split["counts"]["rows"] == 2
    assert split["counts"]["spans_skipped_whitespace"] == 2
    records = read_records(run_dir / "normalized/contractnli.train.jsonl")
    assert records[0]["metadata"]["spans"] == [{"text": "契约", "start": 0, "end": 2}]
    assert records[1]["metadata"]["spans"] == []
    assert records[1]["gold_label"] == "entailment"


def test_contract_bad_span_index_fails(tmp_path):
    hypotheses = ["nda-1", "nda-2"]
    body = contract_body([contract_doc(1, hypotheses, bad_index=5)], hypotheses)
    archive_url = "https://cdn.example.test/contract.zip"
    config = contract_config(archive_url=archive_url)
    config_path = write_config(tmp_path, config)
    run_dir = tmp_path / "run"
    D.stage_fetch(config, run_dir, opener=FakeOpener({archive_url: respond(
        zip_bytes({"contract-nli/train.json": body}))}))
    D.stage_extract(config, run_dir)
    with pytest.raises(D.DatasetError) as exc:
        D.normalize_all(config, run_dir, run_dir / "normalized", config_path=config_path)
    assert exc.value.code == "bad_span_index"


def test_contract_missing_hypothesis_or_count_mismatch_fails(tmp_path):
    hypotheses = ["nda-1", "nda-2"]
    document = contract_doc(1, hypotheses)
    del document["annotation_sets"][0]["annotations"]["nda-2"]
    body = contract_body([document], hypotheses)
    archive_url = "https://cdn.example.test/contract.zip"
    config = contract_config(archive_url=archive_url)
    config_path = write_config(tmp_path, config)
    run_dir = tmp_path / "run"
    D.stage_fetch(config, run_dir, opener=FakeOpener({archive_url: respond(
        zip_bytes({"contract-nli/train.json": body}))}))
    D.stage_extract(config, run_dir)
    with pytest.raises(D.DatasetError) as exc:
        D.normalize_all(config, run_dir, run_dir / "normalized", config_path=config_path)
    assert exc.value.code == "hypotheses_mismatch"
    body = contract_body([contract_doc(1, hypotheses)], hypotheses)
    config = contract_config(expect_hypotheses=3, archive_url=archive_url)
    config_path = write_config(tmp_path, config)
    run_dir2 = tmp_path / "run2"
    D.stage_fetch(config, run_dir2, opener=FakeOpener({archive_url: respond(
        zip_bytes({"contract-nli/train.json": body}))}))
    D.stage_extract(config, run_dir2)
    with pytest.raises(D.DatasetError) as exc:
        D.normalize_all(config, run_dir2, run_dir2 / "normalized", config_path=config_path)
    assert exc.value.code == "hypotheses_mismatch"


                                                                                  

def record_kwargs(**over):
    kwargs = {"dataset": "fx", "official_split": "dev", "task": "nli3", "ref": "1",
              "premise": "p", "hypothesis": "h", "original_label": "entailment",
              "gold_status": "majority", "label_map": {"entailment": "entailment"},
              "group_rule": {"kind": "field", "field": "gid"}, "unit_key": "gid",
              "group_id_source": "raw", "candidates": {"gid": "g1"}, "metadata": {"gid": "g1"},
              "evidence": {"source_url": "https://x/y", "file": "raw/f.json",
                           "sha256": "0" * 64, "row": 1}}
    kwargs.update(over)
    return kwargs


def test_build_record_contract_and_unit_rules():
    record = D.build_record(**record_kwargs())
    assert list(record) == list(D.RECORD_KEYS)
    assert list(record["metadata"]) == ["gold_status", "original_label", "group_id_source", "gid"]
    assert record["group_id"] == "g1"
    assert list(record["evidence"]) == ["source_url", "file", "sha256", "row"]
    family = D.premise_family_key("p")
    derived = D.build_record(**record_kwargs(
        group_rule={"kind": "field", "field": "gid", "fallback": "premise_family_key"},
        unit_key="gid", candidates={}, metadata={"gid": None},
        group_id_source="derived_premise_family"))
    assert derived["group_id"] == family
    assert derived["metadata"]["gid"] is None
    hashed = D.build_record(**record_kwargs(
        group_rule={"kind": "premise_family_key"}, unit_key="premise_hash",
        candidates={}, metadata={"premise_hash": family},
        group_id_source="derived_premise_family"))
    assert hashed["group_id"] == family


@pytest.mark.parametrize("over, code", [
    ({"original_label": "bogus"}, "unknown_label"),
    ({"original_label": "duplicate"}, "unknown_label"),
    ({"gold_status": "maybe"}, "invalid_gold_status"),
    ({"gold_status": "missing"}, "gold_status_mismatch"),
    ({"original_label": None, "gold_status": "majority"}, "gold_status_mismatch"),
    ({"original_label": "contradiction", "label_map": {"contradiction": "contradiction"},
      "task": "entailment2"}, "bad_gold_label"),
    ({"premise": ""}, "empty_premise"),
    ({"hypothesis": None}, "empty_hypothesis"),
    ({"ref": ""}, "bad_ref"),
    ({"metadata": {"other": 1}}, "unit_mismatch"),
    ({"group_id_source": "bogus"}, "bad_group_id_source"),
    ({"metadata": {"gid": None}, "group_id_source": "raw"}, "unit_mismatch"),
    ({"evidence": {"source_url": "https://x/y", "file": "raw/f.json", "sha256": "zz",
                   "row": 1}}, "invalid_evidence"),
    ({"evidence": {"source_url": "", "file": "raw/f.json", "sha256": "0" * 64,
                   "row": 1}}, "invalid_evidence"),
    ({"evidence": {"source_url": "https://x/y", "file": "raw/f.json", "sha256": "0" * 64}},
     "invalid_evidence"),
])
def test_build_record_rejections(over, code):
    with pytest.raises(D.DatasetError) as exc:
        D.build_record(**record_kwargs(**over))
    assert exc.value.code == code


def test_build_record_bounds_metadata_and_evidence():
    with pytest.raises(D.DatasetError) as exc:
        D.build_record(**record_kwargs(metadata={"gid": "g1", "blob": "x" * 20000}))
    assert exc.value.code == "metadata_oversize"
    deep = {"gid": "g1"}
    node = deep
    for _ in range(8):
        node["nested"] = {}
        node = node["nested"]
    with pytest.raises(D.DatasetError) as exc:
        D.build_record(**record_kwargs(metadata=deep))
    assert exc.value.code == "metadata_oversize"


def test_gold_status_null_consistency_is_enforced():
    missing = D.build_record(**record_kwargs(original_label=None, gold_status="missing"))
    assert missing["gold_label"] is None
    generated = D.build_record(**record_kwargs(gold_status="generated"))
    assert generated["gold_label"] == "entailment"


                                                                                          

def test_diagnostics_count_cross_group_split_and_pair_duplicates(tmp_path):
    premise = "同一前提文本"
    dev_rows = [{"sentence1": premise, "sentence2": "同一假设", "label": "entailment",
                 "prem_id": "g1", "id": 0}]
    test_rows = [{"sentence1": premise, "sentence2": "同一假设", "id": 0}]
    dev_body = ("\n".join(json.dumps(row, ensure_ascii=False) for row in dev_rows)
                + "\n").encode()
    test_body = ("\n".join(json.dumps(row, ensure_ascii=False) for row in test_rows)
                 + "\n").encode()
    dev_url = "https://cdn.example.test/dev.json"
    test_url = "https://cdn.example.test/test.json"
    config = fx_config({"ocnli": fx_dataset(
        files=[{"role": "dev", "kind": "direct", "url": dev_url, "dest": "raw/dev.json",
                "max_bytes": 1 << 20, "expect": "jsonl"},
               {"role": "test", "kind": "direct", "url": test_url, "dest": "raw/test.json",
                "max_bytes": 1 << 20, "expect": "jsonl"}],
        splits={"dev": {"official_split": "dev", "format": "ocnli_jsonl", "file_role": "dev"},
                "test": {"official_split": "test", "format": "ocnli_jsonl",
                         "file_role": "test"}})})
    run_dir, manifest, _config_path = run_pipeline(
        tmp_path, config, FakeOpener({dev_url: respond(dev_body), test_url: respond(test_body)}))
    diag = manifest["datasets"]["ocnli"]["diagnostics"]
    assert diag["same_premise_cross_group"]["count"] == 1
    assert diag["same_premise_cross_split"]["count"] == 1
    assert diag["directed_pair_duplicates"]["count"] == 1
    assert diag["same_premise_cross_group"]["refs"][0].startswith("premise:")


def test_verify_is_read_only_and_preserves_unknown_files(tmp_path):
    rows = ocnli_rows()
    body = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
    url = "https://cdn.example.test/ocnli-dev.json"
    config = ocnli_config(url, sha=hashlib.sha256(body).hexdigest())
    run_dir, _manifest, config_path = run_pipeline(
        tmp_path, config, FakeOpener({url: respond(body)}))
    (run_dir / "unknown-sentinel.txt").write_text("keep me", encoding="utf-8")
    (run_dir / "_verify-tmp").mkdir()
    (run_dir / "_verify-tmp" / "keep.txt").write_text("old artifact", encoding="utf-8")
    before = snapshot_tree(run_dir)
    ledger_before = (run_dir / "ledger.jsonl").read_bytes()
    assert D.verify_run(config, run_dir, config_path=config_path) == []
    assert snapshot_tree(run_dir) == before
    assert (run_dir / "ledger.jsonl").read_bytes() == ledger_before
    assert (run_dir / "unknown-sentinel.txt").read_text(encoding="utf-8") == "keep me"
    assert (run_dir / "_verify-tmp" / "keep.txt").read_text(encoding="utf-8") == "old artifact"


@pytest.mark.parametrize("mutate", [
    lambda m: set_manifest_field(m, "datasets.ocnli.license.name", "FORGED"),
    lambda m: set_manifest_field(m, "datasets.ocnli.revision", "FORGED"),
    lambda m: set_manifest_field(m, "datasets.ocnli.files.0.url", "https://evil.example/x"),
    lambda m: set_manifest_field(m, "datasets.ocnli.splits.dev.counts.groups", 999),
    lambda m: set_manifest_field(m, "datasets.ocnli.splits.dev.counts.gold_status.missing", 0),
    lambda m: set_manifest_field(m, "datasets.ocnli.splits.dev.counts.spans_skipped_whitespace", 7),
    lambda m: set_manifest_field(m, "datasets.ocnli.splits.dev.normalized.path",
                                 "normalized/other.jsonl"),
    lambda m: set_manifest_field(m, "datasets.ocnli.splits.dev.omitted_fields", ["x"]),
    lambda m: set_manifest_field(m, "datasets.ocnli.diagnostics.same_premise_cross_group.count", 5),
    lambda m: set_manifest_field(m, "catalog.label_map_sha256", "0" * 64),
    lambda m: set_manifest_field(m, "totals.records", 1),
    lambda m: set_manifest_field(m, "datasets.ocnli.extracted", [{"entry": "x", "path": "y",
                                                                  "bytes": 1, "sha256": "0" * 64}]),
])
def test_verify_rejects_forged_manifest_fields(tmp_path, mutate):
    rows = ocnli_rows()
    body = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
    url = "https://cdn.example.test/ocnli-dev.json"
    config = ocnli_config(url, sha=hashlib.sha256(body).hexdigest())
    run_dir, _manifest, config_path = run_pipeline(
        tmp_path, config, FakeOpener({url: respond(body)}))
    manifest_path = run_dir / "manifest.json"
    forged = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(forged)
    manifest_path.write_text(json.dumps(forged, ensure_ascii=False, indent=2, sort_keys=True)
                             + "\n", encoding="utf-8")
    problems = D.verify_run(config, run_dir, config_path=config_path)
    assert any(problem.startswith("manifest_field_mismatch") for problem in problems), problems


def test_verify_rejects_tampered_extracted_after_renormalize(tmp_path):
    hypotheses = ["nda-1", "nda-2"]
    body = contract_body([contract_doc(7, hypotheses)], hypotheses)
    archive_url = "https://cdn.example.test/contract.zip"
    config = contract_config(archive_url=archive_url)
    run_dir, _manifest, config_path = run_pipeline(
        tmp_path, config, FakeOpener({archive_url: respond(zip_bytes(
            {"contract-nli/train.json": body}))}))
    extracted = run_dir / "raw-extracted/contractnli/contract-nli/train.json"
    forged = json.loads(extracted.read_text(encoding="utf-8"))
    forged["documents"][0]["annotation_sets"][0]["annotations"]["nda-1"]["choice"] = "Contradiction"
    extracted.write_text(json.dumps(forged, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(D.DatasetError) as exc:
        D.normalize_all(config, run_dir, run_dir / "normalized", config_path=config_path)
    assert exc.value.code == "extracted_drift"
    events = ledger_events(run_dir)
    assert events[-1]["event"] == "normalize" and events[-1]["status"] == "error"
    assert events[-1]["code"] == "extracted_drift" and events[-1]["stage"] == "precondition"
    before = snapshot_tree(run_dir)
    assert D.verify_run(config, run_dir, config_path=config_path) == \
        ["normalize_failed:extracted_drift"]
    assert snapshot_tree(run_dir) == before


def test_verify_only_scope_is_explicit_and_subset(tmp_path):
    body_a = b'{"sentence1": "pa", "sentence2": "ha", "label": "entailment", "id": 0, "prem_id": "g"}\n'
    body_b = b'{"sentence1": "pb", "sentence2": "hb", "label": "neutral", "id": 0, "prem_id": "g"}\n'
    url_a, url_b = "https://cdn.example.test/a.json", "https://cdn.example.test/b.json"
    config = fx_config({
        "alpha": fx_dataset(title="alpha", files=[file_spec(role="data", url=url_a,
                                                            dest="raw/a.json")],
                            splits={"dev": {"official_split": "dev",
                                            "format": "ocnli_jsonl", "file_role": "data"}}),
        "beta": fx_dataset(title="beta", files=[file_spec(role="data", url=url_b,
                                                          dest="raw/b.json")],
                           splits={"dev": {"official_split": "dev",
                                           "format": "ocnli_jsonl", "file_role": "data"}}),
    })
    run_dir, _manifest, config_path = run_pipeline(
        tmp_path, config, FakeOpener({url_a: respond(body_a), url_b: respond(body_b)}))
    manifest_path = run_dir / "manifest.json"
    forged = json.loads(manifest_path.read_text(encoding="utf-8"))
    set_manifest_field(forged, "datasets.beta.revision", "FORGED")
    manifest_path.write_text(json.dumps(forged, ensure_ascii=False, indent=2, sort_keys=True)
                             + "\n", encoding="utf-8")
    assert D.verify_run(config, run_dir, only=["alpha"], config_path=config_path) == []
    problems = D.verify_run(config, run_dir, config_path=config_path)
    assert any("datasets.beta.revision" in problem for problem in problems)
    D.stage_verify(config, run_dir, only=["alpha"], config_path=config_path)
    assert D.verify_run(config, run_dir, only=["alpha"], config_path=config_path) == []


def test_normalize_is_deterministic_and_verify_catches_tampering(tmp_path):
    rows = ocnli_rows()
    body = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
    url = "https://cdn.example.test/ocnli-dev.json"
    config = ocnli_config(url, sha=hashlib.sha256(body).hexdigest())
    run_dir, manifest, config_path = run_pipeline(
        tmp_path, config, FakeOpener({url: respond(body)}))
    assert D.verify_run(config, run_dir, config_path=config_path) == []
    again = run_dir / "normalized-again"
    manifest2 = D.normalize_all(config, run_dir, again, config_path=config_path, write=False)
    split_one = manifest["datasets"]["ocnli"]["splits"]["dev"]["normalized"]
    split_two = manifest2["datasets"]["ocnli"]["splits"]["dev"]["normalized"]
    assert split_one["sha256"] == split_two["sha256"]
    assert (run_dir / "normalized/ocnli.dev.jsonl").read_bytes() == \
        (again / "ocnli.dev.jsonl").read_bytes()
    target = run_dir / "normalized/ocnli.dev.jsonl"
    target.write_bytes(target.read_bytes() + b"\n")
    problems = D.verify_run(config, run_dir, config_path=config_path)
    assert any(problem.startswith("normalized_file_drift") for problem in problems)
    raw_file = run_dir / "raw/ocnli-dev.json"
    raw_file.write_bytes(raw_file.read_bytes() + b"\n")
    problems = D.verify_run(config, run_dir, config_path=config_path)
    assert problems == ["normalize_failed:pin_mismatch"]


def test_normalize_rejects_existing_part_and_outside_output_root(tmp_path):
    rows = ocnli_rows()
    body = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
    url = "https://cdn.example.test/ocnli-dev.json"
    config = ocnli_config(url)
    config_path = write_config(tmp_path, config)
    run_dir = tmp_path / "run"
    D.stage_fetch(config, run_dir, opener=FakeOpener({url: respond(body)}))
    out_dir = run_dir / "normalized"
    out_dir.mkdir(parents=True)
    (out_dir / "ocnli.dev.jsonl.part").write_bytes(b"KEEP")
    with pytest.raises(D.DatasetError) as exc:
        D.normalize_all(config, run_dir, out_dir, config_path=config_path)
    assert exc.value.code == "part_exists"
    assert (out_dir / "ocnli.dev.jsonl.part").read_bytes() == b"KEEP"
    with pytest.raises(D.DatasetError) as exc:
        D.normalize_all(config, run_dir, tmp_path / "outside", config_path=config_path)
    assert exc.value.code == "path_escape"


def test_totals_pins_and_manifest_fields(tmp_path):
    rows = ocnli_rows()
    body = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
    url = "https://cdn.example.test/ocnli-dev.json"
    config = ocnli_config(url, sha=hashlib.sha256(body).hexdigest())
    run_dir, manifest, _config_path = run_pipeline(
        tmp_path, config, FakeOpener({url: respond(body)}))
    assert manifest["totals"] == {"records": 6, "missing_gold": 2, "no_consensus": 1}
    pins = json.loads((run_dir / "pins.json").read_text(encoding="utf-8"))
    assert pins == {"ocnli/dev": hashlib.sha256(body).hexdigest()}
    stored = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert stored["datasets"]["ocnli"]["license"]["name"] == "CC0"
    assert stored["catalog"]["label_map"]["ocnli"]["-"] is None
    assert len(stored["catalog"]["label_map_sha256"]) == 64
    assert stored["datasets"]["ocnli"]["splits"]["dev"]["counts"]["group_id_source"] == \
        {"raw": 2, "derived_premise_family": 4}


def test_cli_verify_and_unknown_dataset(tmp_path, capsys):
    rows = ocnli_rows()
    body = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
    url = "https://cdn.example.test/ocnli-dev.json"
    config = ocnli_config(url, sha=hashlib.sha256(body).hexdigest())
    run_dir, _manifest, config_path = run_pipeline(
        tmp_path, config, FakeOpener({url: respond(body)}))
    assert D.main(["verify", "--config", str(config_path), "--run", str(run_dir)]) == 0
    output = capsys.readouterr().out
    assert "verify scope=full" in output and "verify problems=0" in output
    assert D.main(["fetch", "--config", str(config_path), "--run", str(run_dir),
                   "--only", "nope"]) == 2
    assert "unknown_dataset" in capsys.readouterr().out


                                                                                     


def _junction(link: Path, target: Path) -> bool:
    if os.name != "nt":
        return False
    import subprocess
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                            capture_output=True)
    return result.returncode == 0


def make_dir_link(link: Path, target: Path, *, prefer_junction: bool = False) -> None:
    """创建目录链接（symlink 优先；可指定 junction）；主机两种机制都不行时显式 skip。"""
    if prefer_junction and _junction(link, target):
        return
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if not prefer_junction and _junction(link, target):
        return
    pytest.skip("directory link not permitted on this host")


def test_extract_rejects_linked_run_root(tmp_path):
    outside = tmp_path / "linked-outside"
    outside.mkdir()
    run_dir = tmp_path / "linked-run"
    make_dir_link(run_dir, outside)
    archive = tmp_path / "a.zip"
    archive.write_bytes(zip_bytes({"ok.json": b"payload"}))
    with pytest.raises(D.DatasetError) as exc:
        D.extract_archive(archive, run_dir / "out", ["ok.json"], limits=LIMITS, dataset="fx",
                          ledger_path=run_dir / "ledger.jsonl", run_dir=run_dir)
    assert exc.value.code == "unsafe_root"
    assert not (outside / "out").exists()
    assert not (outside / "ledger.jsonl").exists()


def test_extract_rejects_junction_run_root(tmp_path):
    outside = tmp_path / "junction-outside"
    outside.mkdir()
    run_dir = tmp_path / "junction-run"
    make_dir_link(run_dir, outside, prefer_junction=True)
    archive = tmp_path / "a.zip"
    archive.write_bytes(zip_bytes({"ok.json": b"payload"}))
    with pytest.raises(D.DatasetError) as exc:
        D.extract_archive(archive, run_dir / "out", ["ok.json"], limits=LIMITS, dataset="fx",
                          ledger_path=run_dir / "ledger.jsonl", run_dir=run_dir)
    assert exc.value.code == "unsafe_root"
    assert list(outside.iterdir()) == []


def test_extract_rejects_nested_junction_inside_run(tmp_path):
    outside = tmp_path / "junction-outside"
    outside.mkdir()
    dest = tmp_path / "out"
    dest.mkdir()
    make_dir_link(dest / "link", outside, prefer_junction=True)
    archive = tmp_path / "a.zip"
    archive.write_bytes(zip_bytes({"link/f.txt": b"x"}))
    with pytest.raises(D.DatasetError) as exc:
        D.extract_archive(archive, dest, ["link/f.txt"], limits=LIMITS, dataset="fx",
                          ledger_path=tmp_path / "ledger.jsonl", run_dir=tmp_path)
    assert exc.value.code == "unsafe_path"
    assert not (outside / "f.txt").exists()


def test_mutating_entries_reject_linked_run_root(tmp_path):
    outside = tmp_path / "linked-outside"
    outside.mkdir()
    run_dir = tmp_path / "linked-run"
    make_dir_link(run_dir, outside)
    spec = file_spec()
    with pytest.raises(D.DatasetError) as exc:
        D.fetch_file(spec, run_dir, dataset="fx", allowed_hosts=["cdn.example.test"],
                     limits=LIMITS, ledger_path=run_dir / "ledger.jsonl",
                     opener=FakeOpener({spec["url"]: respond(b'{"a": 1}\n')}))
    assert exc.value.code == "unsafe_root"
    body = ("\n".join(json.dumps(row, ensure_ascii=False) for row in ocnli_rows())
            + "\n").encode()
    url = "https://cdn.example.test/ocnli-dev.json"
    config = ocnli_config(url)
    with pytest.raises(D.DatasetError) as exc:
        D.stage_fetch(config, run_dir, opener=FakeOpener({url: respond(body)}))
    assert exc.value.code == "unsafe_root"
    with pytest.raises(D.DatasetError) as exc:
        D.normalize_all(config, run_dir, run_dir / "normalized")
    assert exc.value.code == "unsafe_root"
    assert list(outside.iterdir()) == []


def test_extract_rejects_alias_entries_public_api(tmp_path):
    archive = tmp_path / "a.zip"
    archive.write_bytes(zip_bytes({"a.json": b"lower", "A.JSON": b"UPPER"}))
    with pytest.raises(D.DatasetError) as exc:
        D.extract_archive(archive, tmp_path / "out", ["a.json", "A.JSON"], limits=LIMITS,
                          dataset="fx", ledger_path=tmp_path / "ledger.jsonl",
                          run_dir=tmp_path)
    assert exc.value.code == "alias_entry"
    assert not (tmp_path / "out").exists()
    events = ledger_events(tmp_path)
    assert events[-1]["event"] == "extract" and events[-1]["status"] == "error"
    assert events[-1]["code"] == "alias_entry" and events[-1]["partial"] == 0


def test_extract_bad_zip_writes_ledger_event(tmp_path):
    archive = tmp_path / "a.zip"
    archive.write_bytes(b"not a zip archive")
    with pytest.raises(D.DatasetError) as exc:
        D.extract_archive(archive, tmp_path / "out", ["ok.json"], limits=LIMITS, dataset="fx",
                          ledger_path=tmp_path / "ledger.jsonl", run_dir=tmp_path)
    assert exc.value.code == "bad_zip"
    assert not (tmp_path / "out").exists()
    events = ledger_events(tmp_path)
    assert events[-1]["event"] == "extract" and events[-1]["status"] == "error"
    assert events[-1]["code"] == "bad_zip" and events[-1]["partial"] == 0


def test_extract_crc_corruption_writes_ledger_event(tmp_path):
    payload = b"payload-bytes-0123456789"
    data = bytearray(zip_bytes({"ok.json": payload}))
    index = bytes(data).find(payload)
    assert index > 0, "stored 模式 zip 应可定位并破坏 CRC 源字节"
    data[index] ^= 0x01
    archive = tmp_path / "a.zip"
    archive.write_bytes(bytes(data))
    with pytest.raises(D.DatasetError) as exc:
        D.extract_archive(archive, tmp_path / "out", ["ok.json"], limits=LIMITS, dataset="fx",
                          ledger_path=tmp_path / "ledger.jsonl", run_dir=tmp_path)
    assert exc.value.code == "bad_zip"
    events = ledger_events(tmp_path)
    assert events[-1]["event"] == "extract" and events[-1]["status"] == "error"
    assert events[-1]["code"] == "bad_zip" and events[-1]["partial"] == 0
    assert not (tmp_path / "out" / "ok.json").exists()
    assert not (tmp_path / "out" / "ok.json.part").exists()


def test_extract_missing_source_writes_ledger_event(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(D.DatasetError) as exc:
        D.extract_archive(run_dir / "raw/a.zip", run_dir / "out", ["ok.json"], limits=LIMITS,
                          dataset="fx", ledger_path=run_dir / "ledger.jsonl",
                          run_dir=run_dir)
    assert exc.value.code == "missing_raw_file"
    events = ledger_events(run_dir)
    assert events[-1]["event"] == "extract" and events[-1]["status"] == "error"
    assert events[-1]["code"] == "missing_raw_file"


def _ocnli_body(rows) -> bytes:
    return ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()


def test_verify_unknown_label_failure_paths_are_read_only(tmp_path, capsys):
    good_body = _ocnli_body(ocnli_rows())
    bad_body = _ocnli_body([{"sentence1": "p", "sentence2": "h", "label": "maybe",
                             "id": 0, "prem_id": "g"}])
    url = "https://cdn.example.test/ocnli-dev.json"
    config = ocnli_config(url, sha=hashlib.sha256(good_body).hexdigest())
    run_dir, _manifest, _config_path = run_pipeline(
        tmp_path, config, FakeOpener({url: respond(good_body)}))
    (run_dir / "raw/ocnli-dev.json").write_bytes(bad_body)
    bad_config = ocnli_config(url, sha=hashlib.sha256(bad_body).hexdigest())
    bad_config_path = tmp_path / "bad-sources.json"
    bad_config_path.write_text(json.dumps(bad_config, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    (run_dir / "unknown-sentinel.txt").write_text("keep me", encoding="utf-8")
    (run_dir / "_verify-tmp").mkdir()
    (run_dir / "_verify-tmp" / "keep.txt").write_text("old artifact", encoding="utf-8")
    before = snapshot_tree(run_dir)
    ledger_before = (run_dir / "ledger.jsonl").read_bytes()
    problems = D.verify_run(bad_config, run_dir, config_path=bad_config_path)
    assert problems == ["normalize_failed:unknown_label"]
    assert snapshot_tree(run_dir) == before
    assert (run_dir / "ledger.jsonl").read_bytes() == ledger_before
    assert D.main(["verify", "--config", str(bad_config_path), "--run", str(run_dir)]) == 1
    output = capsys.readouterr().out
    assert "normalize_failed:unknown_label" in output
    assert "verify problems=0" not in output
    assert snapshot_tree(run_dir) == before
    assert (run_dir / "ledger.jsonl").read_bytes() == ledger_before
    assert (run_dir / "_verify-tmp" / "keep.txt").read_text(encoding="utf-8") == "old artifact"
    assert (run_dir / "unknown-sentinel.txt").read_text(encoding="utf-8") == "keep me"


def test_ocnli_majority_requires_strict_vote_support(tmp_path):
    rows = [
                                                               
        {"sentence1": "p1", "sentence2": "h1", "label": "neutral", "id": 0, "prem_id": "g1",
         "label0": "entailment", "label1": "contradiction", "label2": "neutral",
         "label3": None, "label4": None},
                                            
        {"sentence1": "p2", "sentence2": "h2", "label": "neutral", "id": 1, "prem_id": "g2",
         "label0": "entailment", "label1": "entailment", "label2": None, "label3": None,
         "label4": None},
                                         
        {"sentence1": "p3", "sentence2": "h3", "label": "entailment", "id": 2, "prem_id": "g3",
         "label0": "entailment", "label1": "entailment", "label2": "neutral",
         "label3": "contradiction", "label4": None},
                           
        {"sentence1": "p4", "sentence2": "h4", "label": "entailment", "id": 3, "prem_id": "g4",
         "label0": "entailment", "label1": "entailment", "label2": "entailment",
         "label3": "neutral", "label4": "contradiction"},
    ]
    body = _ocnli_body(rows)
    url = "https://cdn.example.test/ocnli-dev.json"
    config = ocnli_config(url, sha=hashlib.sha256(body).hexdigest())
    run_dir, manifest, _config_path = run_pipeline(
        tmp_path, config, FakeOpener({url: respond(body)}))
    split = manifest["datasets"]["ocnli"]["splits"]["dev"]
    assert split["counts"]["gold_status"] == {"majority": 1, "single": 3}
    assert split["counts"]["votes_valid"] == {"3": 1, "2": 1, "4": 1, "5": 1}
    records = read_records(run_dir / "normalized/ocnli.dev.jsonl")
    assert records[0]["gold_label"] == "neutral"
    assert records[0]["metadata"]["gold_status"] == "single"
    assert records[0]["metadata"]["label_method"] == "no_majority"
    assert records[0]["metadata"]["votes"] == ["entailment", "contradiction", "neutral",
                                               None, None]
    assert records[1]["gold_label"] == "neutral"
    assert records[1]["metadata"]["label_method"] == "no_majority"
    assert records[2]["metadata"]["votes_valid_count"] == 4
    assert records[2]["metadata"]["label_method"] == "no_majority"
    assert records[3]["gold_label"] == "entailment"
    assert records[3]["metadata"]["gold_status"] == "majority"
    assert records[3]["metadata"]["label_method"] == "majority_vote"
                                                         
                                   
    for row, record in zip(rows, records):
        decision = D.ocnli_vote_status(
            row.get("label"), [row.get(key) for key in D.OCNLI_VOTE_KEYS])
        assert record["metadata"]["gold_status"] == decision["gold_status"]
        assert record["metadata"]["label_method"] == decision["label_method"]
        assert record["metadata"]["votes_valid_count"] == decision["votes_valid_count"]
        assert record["metadata"]["original_label"] == row["label"]
        assert record["gold_label"] == config["datasets"]["ocnli"]["label_map"][row["label"]]


                                                                                                           


def patch_lstat_permission(monkeypatch, denied_path: Path) -> None:
    """仅对指定路径注入 PermissionError(EACCES)（标准 os.lstat 边界；其余调用真实）。"""
    real_lstat = os.lstat
    denied = os.path.normcase(os.path.abspath(str(denied_path)))

    def stat_io(path, *args, **kwargs):
        if os.path.normcase(os.path.abspath(str(path))) == denied:
            raise PermissionError(13, "Permission denied (fault injection)", str(path))
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", stat_io)


@pytest.mark.parametrize("link_kind", ["symlink", "junction"])
def test_lstat_io_fault_on_linked_root_fails_closed(tmp_path, monkeypatch, link_kind):
    outside = tmp_path / "outside"
    outside.mkdir()
    run_dir = tmp_path / "linked-run"
    make_dir_link(run_dir, outside, prefer_junction=(link_kind == "junction"))
    archive = tmp_path / "a.zip"
    archive.write_bytes(zip_bytes({"ok.json": b"OUTSIDE-WRITE"}))
    url = "https://cdn.example.test/ocnli-dev.json"
    body = _ocnli_body(ocnli_rows())
    config = ocnli_config(url)
    spec = file_spec()
    patch_lstat_permission(monkeypatch, run_dir)
    before = snapshot_tree(outside)
    cases = [
        lambda: D.extract_archive(archive, run_dir / "out", ["ok.json"], limits=LIMITS,
                                  dataset="fx", ledger_path=run_dir / "ledger.jsonl",
                                  run_dir=run_dir),
        lambda: D.fetch_file(spec, run_dir, dataset="fx", allowed_hosts=["cdn.example.test"],
                             limits=LIMITS, ledger_path=run_dir / "ledger.jsonl",
                             opener=FakeOpener({spec["url"]: respond(b'{"a": 1}\n')})),
        lambda: D.normalize_all(config, run_dir, run_dir / "normalized"),
        lambda: D.stage_fetch(config, run_dir, opener=FakeOpener({url: respond(body)})),
    ]
    for case in cases:
        with pytest.raises(D.DatasetError) as exc:
            case()
        assert exc.value.code == "unreadable_path"
    assert snapshot_tree(outside) == before


def test_lstat_io_fault_on_plain_root_fails_closed(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "raw/a.json").write_bytes(b'{"a": 1}\n')
    url = "https://cdn.example.test/ocnli-dev.json"
    body = _ocnli_body(ocnli_rows())
    config = ocnli_config(url)
    patch_lstat_permission(monkeypatch, run_dir)
    before = snapshot_tree(run_dir)
    cases = [
        lambda: D.stage_fetch(config, run_dir, opener=FakeOpener({url: respond(body)})),
        lambda: D.fetch_file(file_spec(), run_dir, dataset="fx",
                             allowed_hosts=["cdn.example.test"], limits=LIMITS,
                             ledger_path=run_dir / "ledger.jsonl",
                             opener=FakeOpener({file_spec()["url"]: respond(b"{}\n")})),
        lambda: D.normalize_all(config, run_dir, run_dir / "normalized"),
    ]
    for case in cases:
        with pytest.raises(D.DatasetError) as exc:
            case()
        assert exc.value.code == "unreadable_path"
    assert snapshot_tree(run_dir) == before


def test_lstat_io_fault_on_extract_targets_refused_before_any_output(tmp_path, monkeypatch):
    archive = tmp_path / "a.zip"
    archive.write_bytes(zip_bytes({"ok.json": b"payload"}))
    dest = tmp_path / "out"
    patch_lstat_permission(monkeypatch, dest / "ok.json")
    with pytest.raises(D.DatasetError) as exc:
        D.extract_archive(archive, dest, ["ok.json"], limits=LIMITS, dataset="fx",
                          ledger_path=tmp_path / "ledger.jsonl", run_dir=tmp_path)
    assert exc.value.code == "unreadable_path"
    assert not dest.exists()
    assert not list(tmp_path.rglob("*.part"))
    events = ledger_events(tmp_path)
    assert events[-1]["event"] == "extract" and events[-1]["status"] == "error"
    assert events[-1]["code"] == "unreadable_path" and events[-1]["partial"] == 0


def test_lstat_io_fault_on_existing_parent_component_fails_closed(tmp_path, monkeypatch):
    archive = tmp_path / "a.zip"
    archive.write_bytes(zip_bytes({"sub/ok.json": b"payload"}))
    dest = tmp_path / "out"
    dest.mkdir()
    patch_lstat_permission(monkeypatch, dest)
    with pytest.raises(D.DatasetError) as exc:
        D.extract_archive(archive, dest, ["sub/ok.json"], limits=LIMITS, dataset="fx",
                          ledger_path=tmp_path / "ledger.jsonl", run_dir=tmp_path)
    assert exc.value.code == "unreadable_path"
    assert not (dest / "sub").exists()
    assert not list(dest.rglob("*.part"))


def test_lstat_io_fault_on_normalize_output_root_fails_closed(tmp_path, monkeypatch):
    body = _ocnli_body(ocnli_rows())
    url = "https://cdn.example.test/ocnli-dev.json"
    config = ocnli_config(url)
    config_path = write_config(tmp_path, config)
    run_dir = tmp_path / "run"
    D.stage_fetch(config, run_dir, opener=FakeOpener({url: respond(body)}))
    out_dir = run_dir / "normalized"
    patch_lstat_permission(monkeypatch, out_dir)
    before = snapshot_tree(run_dir)
    with pytest.raises(D.DatasetError) as exc:
        D.normalize_all(config, run_dir, out_dir, config_path=config_path)
    assert exc.value.code == "unreadable_path"
    assert not out_dir.exists()
    after = snapshot_tree(run_dir)
    assert set(after) == set(before)
    for key in before:
        if key != "ledger.jsonl":
            assert after[key] == before[key]
    assert after["ledger.jsonl"][0] > before["ledger.jsonl"][0]             
    events = ledger_events(run_dir)
    assert events[-1]["event"] == "normalize" and events[-1]["status"] == "error"
    assert events[-1]["code"] == "unreadable_path"


@pytest.mark.parametrize("link_kind", ["symlink", "junction"])
def test_linked_ancestor_of_run_root_rejected(tmp_path, link_kind):
    outside = tmp_path / "ancestor-outside"
    outside.mkdir()
    anchor = tmp_path / "ancestor-link"
    make_dir_link(anchor, outside, prefer_junction=(link_kind == "junction"))
    run_dir = anchor / "run"
    run_dir.mkdir(parents=True)
    archive = tmp_path / "a.zip"
    archive.write_bytes(zip_bytes({"ok.json": b"payload"}))
    before = snapshot_tree(outside)
    with pytest.raises(D.DatasetError) as exc:
        D.extract_archive(archive, run_dir / "out", ["ok.json"], limits=LIMITS, dataset="fx",
                          ledger_path=run_dir / "ledger.jsonl", run_dir=run_dir)
    assert exc.value.code == "unsafe_root"
    assert snapshot_tree(outside) == before


def test_ocnli_vote_status_is_public_pure_anchor():
    assert "ocnli_vote_status" in D.__all__
    cases = [
        (None, [None] * 5, "missing", "no_label", 0, 0),
        ("-", ["entailment", "neutral", "contradiction", "unknown", "unrelated"],
         "no_consensus", "majority_vote", 3, 0),
        ("neutral", ["entailment", "contradiction", "neutral", None, None],
         "single", "no_majority", 3, 1),
        ("neutral", ["entailment", "entailment", None, None, None],
         "single", "no_majority", 2, 0),
        ("entailment", ["entailment", "entailment", "neutral", "contradiction", None],
         "single", "no_majority", 4, 2),
        ("entailment", ["entailment", "entailment", "entailment", "neutral",
                        "contradiction"], "majority", "majority_vote", 5, 3),
        ("neutral", ["neutral", None, None, None, None],
         "single", "single_annotation", 1, 1),
    ]
    for original, votes, status, method, valid_count, agree_count in cases:
        snapshot = list(votes)
        assert D.ocnli_vote_status(original, votes) == {
            "gold_status": status, "label_method": method,
            "votes_valid_count": valid_count, "votes_agree_count": agree_count}
        assert votes == snapshot             
