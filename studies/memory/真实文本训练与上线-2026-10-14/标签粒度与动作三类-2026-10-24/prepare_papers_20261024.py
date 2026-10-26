from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "chen-label-granularity-2019": "https://arxiv.org/pdf/1901.07012v1",
    "pirovano-fine-training-v2-2026": "https://arxiv.org/pdf/2509.05130v2",
    "george-neurips-2020": "https://proceedings.neurips.cc/paper_files/paper/2020/file/e0688d13958a19e087e123148555e4b4-Paper.pdf",
    "george-supplement-2020": "https://proceedings.neurips.cc/paper_files/paper/2020/file/e0688d13958a19e087e123148555e4b4-Supplemental.pdf",
}


def main():
    directory = ROOT / "papers"
    directory.mkdir(exist_ok=True)
    receipts = []
    for name, url in SOURCES.items():
        path = directory / (name + ".pdf")
        if not path.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "Research reading/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read(20 * 1024 * 1024 + 1)
            assert data.startswith(b"%PDF") and len(data) <= 20 * 1024 * 1024
            with path.open("xb") as stream:
                stream.write(data)
        data = path.read_bytes()
        with fitz.open(path) as document:
            pages = [f"\n## Page {i + 1}\n\n" + page.get_text(sort=True) for i, page in enumerate(document)]
        text = "".join(pages)
        target = directory / (name + ".txt")
        target.write_text(text, encoding="utf-8", newline="\n")
        receipt = {"name": name, "url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                   "pages": len(pages), "text_chars": len(text), "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
        receipts.append(receipt)
        (directory / "sources-2026-10-24.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
