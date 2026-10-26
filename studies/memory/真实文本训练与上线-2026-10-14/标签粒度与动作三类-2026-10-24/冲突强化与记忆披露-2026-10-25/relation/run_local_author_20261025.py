from __future__ import annotations

from functools import partial

import build_conflict_data as B


if __name__ == "__main__":
    B.OllamaChannel = partial(B.OllamaChannel, timeout=600, num_ctx=16384)
    raise SystemExit(B.main())
