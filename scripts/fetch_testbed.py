"""Fetch the censored-LLM testbed and write data/prompts/testbed_en.jsonl.

    uv run python scripts/fetch_testbed.py
    uv run python scripts/fetch_testbed.py --source-dir /path/to/chinese_auditing

Downloads the pinned upstream files (see src/tilt/testbed.py), verifies their
SHA-256 hashes, and converts them. The output is gitignored: the upstream
repository has no licence, so the data is not redistributed in this repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tilt.questions import write_questions
from tilt.testbed import COMMIT, DEFAULT_MIN_FACT_COUNT, REPO, fetch_testbed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-dir", type=Path, default=None,
                        help="Local clone of the upstream repo instead of downloading")
    parser.add_argument("--min-fact-count", type=int, default=DEFAULT_MIN_FACT_COUNT)
    parser.add_argument("--out", type=Path, default=Path("data/prompts/testbed_en.jsonl"))
    args = parser.parse_args()

    questions = fetch_testbed(args.source_dir, args.min_fact_count)
    write_questions(args.out, questions)
    for split in ("test", "dev"):
        qs = [q for q in questions if q.split == split]
        print(f"{split}: {len(qs)} questions, {sum(len(q.facts) for q in qs)} facts "
              f"(count >= {args.min_fact_count})")
    print(f"wrote {args.out} from {REPO}@{COMMIT[:12]}")


if __name__ == "__main__":
    main()
