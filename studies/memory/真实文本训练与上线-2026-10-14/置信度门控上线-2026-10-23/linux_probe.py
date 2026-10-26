from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import resource
import time
from pathlib import Path

BUNDLE = Path("/bundle")
PIN = "65a8f9290a01ee038671a9bea068d2047e0897eac5937ecf16cb71bec16ad7b7"
spec = importlib.util.spec_from_file_location("isolated_model_worker", "/probe/model_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
request = json.loads(Path("/probe/request.json").read_text())
start = time.perf_counter()
manifest = worker.read_relation_manifest(BUNDLE, PIN, verify_files=True)
scorer = worker._load_relation_runtime(BUNDLE, manifest)
loaded = time.perf_counter() - start
outputs = []
for _ in range(2):
    buffer = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(buffer):
        worker._relation_serve_loop(scorer, manifest, stream=io.BytesIO((json.dumps(request) + "\n").encode()))
    result = json.loads(buffer.getvalue())
    assert result["ok"], result
    outputs.append({"seconds": time.perf_counter() - started, "result": result})
assert outputs[0]["result"] == outputs[1]["result"]
stat = {}
for name in ("memory.current", "memory.peak", "memory.swap.current", "memory.events"):
    path = Path("/sys/fs/cgroup") / name
    if path.exists():
        stat[name] = path.read_text().strip()
print(json.dumps({"passed": True, "device": "cpu", "manifest_sha256": PIN, "load_seconds": loaded, "inferences": outputs, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "cgroup": stat, "cpu_container_limit": 1, "limitation": "Isolated runtime test, no production user data or API credentials mounted."}, sort_keys=True))
