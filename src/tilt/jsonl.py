"""Small JSONL helpers shared by the pipeline stages."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

log = logging.getLogger(__name__)


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield records; skip blank lines and a truncated or corrupt line (logged).

    A run killed mid-write can leave a partial last line; skipping it lets the
    resume logic regenerate that record.
    """
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                log.warning("%s:%d: skipping unparseable line", path, lineno)
                continue
            if isinstance(rec, dict):
                yield rec


def append_jsonl(f, rec: dict) -> None:
    """Write one record to an open text file and flush, so a crash loses at most one line."""
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    f.flush()


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def ensure_trailing_newline(path: str | Path) -> None:
    """If a previous run died mid-line, terminate that line before appending."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as f:
            f.seek(-1, 2)
            last = f.read(1)
        if last != b"\n":
            with path.open("a", encoding="utf-8") as f:
                f.write("\n")
