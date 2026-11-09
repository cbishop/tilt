import json

import pytest
from conftest import FakeClient, FakeMessage

from tilt.llm import ParseError, call_json, call_json_with_usage


def parse_ok(obj):
    if "x" not in obj:
        raise ParseError("missing x")
    return obj["x"]


def call(client, **kw):
    return call_json_with_usage(client, system="s", user="u", schema={}, parse=parse_ok,
                                sleep=lambda s: None, **kw)


def test_usage_extracted_from_message():
    msg = FakeMessage(json.dumps({"x": 1}), usage={
        "input_tokens": 120, "output_tokens": 30, "cache_read_input_tokens": 5,
        "cache_creation_input_tokens": 7})
    value, usage = call(FakeClient([msg]))
    assert value == 1
    assert usage == {"input_tokens": 120, "output_tokens": 30, "cache_read_input_tokens": 5,
                     "cache_creation_input_tokens": 7, "attempts": 1}


def test_usage_summed_across_retries():
    client = FakeClient([
        FakeMessage("not json", usage={"input_tokens": 100, "output_tokens": 10}),
        FakeMessage("{", stop_reason="max_tokens",
                    usage={"input_tokens": 100, "output_tokens": 50}),
        FakeMessage(json.dumps({"x": 2}), usage={"input_tokens": 100, "output_tokens": 20}),
    ])
    value, usage = call(client)
    assert value == 2 and len(client.calls) == 3
    assert usage["input_tokens"] == 300 and usage["output_tokens"] == 80
    assert usage["cache_read_input_tokens"] == 0 and usage["attempts"] == 3


def test_api_error_without_message_adds_nothing():
    import anthropic
    import httpx

    err = anthropic.APIConnectionError(request=httpx.Request("POST", "http://x"))
    client = FakeClient([err, FakeMessage(json.dumps({"x": 3}),
                                          usage={"input_tokens": 9, "output_tokens": 4})])
    value, usage = call(client)
    assert value == 3
    assert (usage["input_tokens"], usage["output_tokens"], usage["attempts"]) == (9, 4, 2)


def test_zeros_when_usage_missing():
    value, usage = call(FakeClient([json.dumps({"x": 4})]))
    assert value == 4
    assert usage == {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                     "cache_creation_input_tokens": 0, "attempts": 1}


def test_failure_carries_usage_and_call_json_unchanged():
    client = FakeClient(lambda kw: FakeMessage("garbage", usage={"input_tokens": 10,
                                                                  "output_tokens": 1}))
    with pytest.raises(ParseError) as ei:
        call(client, attempts=3)
    assert ei.value.usage["input_tokens"] == 30 and ei.value.usage["attempts"] == 3
    assert call_json(FakeClient([json.dumps({"x": 5})]), system="s", user="u", schema={},
                     parse=parse_ok) == 5
