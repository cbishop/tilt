"""Question files: loading, schema validation and discovery.

Every question file in data/prompts/ is JSONL with one record per question:

    {"id": str, "topic": str, "question": str, "facts": [str, ...], "split": "test" | "dev"}

Optional extra keys are kept (for example "fact_counts" on testbed records).
Ids are unique across all sets, so a recall or judge record can be traced back
to its question with the id alone. File names follow <set>_<lang>.jsonl.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SETS = ("testbed", "us_sensitive", "neutral")
LANGS = ("en", "zh")
SPLITS = ("test", "dev")
REQUIRED_KEYS = ("id", "topic", "question", "facts", "split")


class QuestionFileError(ValueError):
    """A question file is missing, malformed or violates the schema."""


@dataclass(frozen=True)
class Question:
    id: str
    topic: str
    question: str
    facts: tuple[str, ...]
    split: str
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    def to_record(self) -> dict:
        rec = {
            "id": self.id,
            "topic": self.topic,
            "question": self.question,
            "facts": list(self.facts),
            "split": self.split,
        }
        rec.update(self.extra)
        return rec


def validate_record(rec: object, where: str = "") -> Question:
    """Validate one decoded JSON record and return a Question."""
    if not isinstance(rec, dict):
        raise QuestionFileError(f"{where}: record is not a JSON object")
    missing = [k for k in REQUIRED_KEYS if k not in rec]
    if missing:
        raise QuestionFileError(f"{where}: missing keys {missing}")
    for k in ("id", "topic", "question"):
        if not isinstance(rec[k], str) or not rec[k].strip():
            raise QuestionFileError(f"{where}: '{k}' must be a non-empty string")
    facts = rec["facts"]
    if not isinstance(facts, list) or not all(isinstance(f, str) and f.strip() for f in facts):
        raise QuestionFileError(f"{where}: 'facts' must be a list of non-empty strings")
    if rec["split"] not in SPLITS:
        raise QuestionFileError(f"{where}: 'split' must be one of {SPLITS}, got {rec['split']!r}")
    extra = {k: v for k, v in rec.items() if k not in REQUIRED_KEYS}
    return Question(
        id=rec["id"],
        topic=rec["topic"],
        question=rec["question"],
        facts=tuple(facts),
        split=rec["split"],
        extra=extra,
    )


def load_questions(path: str | Path) -> list[Question]:
    """Load and validate a question JSONL file. Ids must be unique within the file."""
    path = Path(path)
    if not path.exists():
        raise QuestionFileError(f"{path}: file not found")
    out: list[Question] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            where = f"{path}:{lineno}"
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise QuestionFileError(f"{where}: invalid JSON ({e})") from e
            q = validate_record(rec, where)
            if q.id in seen:
                raise QuestionFileError(f"{where}: duplicate id {q.id!r}")
            seen.add(q.id)
            out.append(q)
    return out


def write_questions(path: str | Path, questions: list[Question]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q.to_record(), ensure_ascii=False) + "\n")


def set_path(directory: str | Path, set_name: str, lang: str) -> Path:
    return Path(directory) / f"{set_name}_{lang}.jsonl"


def load_question_sets(
    source: str | Path,
    lang: str,
    sets: tuple[str, ...] | list[str] = SETS,
    split: str | None = "test",
) -> list[Question]:
    """Load questions for a language.

    `source` is either a single JSONL file or a directory holding <set>_<lang>.jsonl
    files. `split=None` keeps every split. Ids must be unique across the loaded sets.
    """
    source = Path(source)
    if source.is_file():
        paths = [source]
    else:
        paths = [set_path(source, s, lang) for s in sets]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise QuestionFileError(
                "missing question files: "
                + ", ".join(missing)
                + " (testbed: run scripts/fetch_testbed.py; Chinese: run tilt-translate)"
            )
    questions: list[Question] = []
    seen: set[str] = set()
    for p in paths:
        for q in load_questions(p):
            if q.id in seen:
                raise QuestionFileError(f"{p}: id {q.id!r} also appears in another set")
            seen.add(q.id)
            if split is None or q.split == split:
                questions.append(q)
    return questions


def model_short(model: str) -> str:
    """Short, filesystem-safe model name: 'mlx-community/gemma-3-4b-it-4bit' -> 'gemma-3-4b-it-4bit'."""
    name = model.rstrip("/").split("/")[-1]
    return "".join(c if c.isalnum() or c in "-._" else "-" for c in name).lower()
