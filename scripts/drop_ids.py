"""Remove records by id from JSONL files, in place.

Usage:
    uv run python scripts/drop_ids.py <ids-file> <file.jsonl> [<file.jsonl> ...]

<ids-file> holds one id per line (blank lines and lines starting with # are
ignored). Every record whose "id" is in the list is removed; all other lines are
kept verbatim and in order. Each file is rewritten atomically (temp file in the
same directory, then rename). Prints the number of records removed per file.
Used to invalidate downstream artefacts for questions whose inputs changed (for
example re-translated Chinese records), so the resume-safe stages regenerate them.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


def load_ids(path: Path) -> set[str]:
    return {s for line in path.read_text(encoding="utf-8").splitlines()
            if (s := line.strip()) and not s.startswith("#")}


def drop_ids(path: Path, ids: set[str]) -> tuple[int, int]:
    """Rewrite `path` without the records whose id is in `ids`. Returns (removed, kept)."""
    kept_lines, removed = [], 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    rec = None
                if isinstance(rec, dict) and rec.get("id") in ids:
                    removed += 1
                    continue
            kept_lines.append(line if line.endswith("\n") else line + "\n")
    if removed:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.writelines(kept_lines)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    return removed, sum(1 for line in kept_lines if line.strip())


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        sys.exit(__doc__)
    ids = load_ids(Path(argv[0]))
    missing = False
    for p in map(Path, argv[1:]):
        if not p.exists():
            print(f"{p}: not found", file=sys.stderr)
            missing = True
            continue
        removed, kept = drop_ids(p, ids)
        print(f"{p}: removed {removed}, kept {kept}")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
