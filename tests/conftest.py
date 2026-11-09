import json
from pathlib import Path

import pytest


def write_jsonl(path: Path, records: list) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                    encoding="utf-8")
    return path


def qrec(i: str, n_facts: int = 2, split: str = "test", **extra) -> dict:
    rec = {"id": i, "topic": "t", "question": f"Question {i}?",
           "facts": [f"fact {k} of {i}" for k in range(n_facts)], "split": split}
    rec.update(extra)
    return rec


class FakeMessage:
    def __init__(self, text: str, stop_reason: str = "end_turn", usage: dict | None = None):
        self.content = [type("Block", (), {"type": "text", "text": text})()]
        self.stop_reason = stop_reason
        if usage is not None:
            self.usage = type("Usage", (), dict(usage))()


class FakeClient:
    """Mimics client.messages.create; replies come from a callable or a list."""

    def __init__(self, reply):
        self.calls = []
        self._reply = reply
        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                r = outer._reply(kwargs) if callable(outer._reply) else outer._reply.pop(0)
                if isinstance(r, Exception):
                    raise r
                return r if isinstance(r, FakeMessage) else FakeMessage(r)

        self.messages = _Messages()


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr("tilt.llm.time.sleep", lambda s: None)
