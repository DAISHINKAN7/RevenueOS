"""Minimal .env loader.

Sourcing .env from a Makefile recipe is fragile: an unquoted `#`, a stray
quote, or CRLF line endings can abort the whole recipe silently. Loading it in
Python instead makes every entrypoint behave identically regardless of shell,
and a malformed line is skipped with a warning rather than killing the process.

No dependency on python-dotenv.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env", override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE pairs. Real environment variables win unless `override`."""
    p = Path(path)
    loaded: dict[str, str] = {}
    if not p.exists():
        return loaded

    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip().lstrip("\ufeff")          # tolerate a BOM
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip("\r")
        # Strip matching quotes; only strip a trailing comment when unquoted.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].strip()
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded