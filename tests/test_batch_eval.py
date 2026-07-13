"""Tests for the asynchronous Batch API evaluation path (e.g. Maritaca).

The live-request path is exercised elsewhere; here we cover the batch helpers
and the end-to-end `_run_batch_requests` flow with LiteLLM's batch functions
mocked, so no network calls are made.
"""

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from latamqa.eval_mcq import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROPMT_TEMPLATE,
    _batch_client,
    _custom_id_to_idx,
    _extract_batch_content,
    _run_batch_requests,
    is_batch_endpoint,
)


def test_batch_client_fails_fast_without_api_key(monkeypatch):
    """An empty key (e.g. unexpanded $MARITACA_API_KEY) stops the run cleanly."""
    monkeypatch.delenv("MARITACA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Empty string mimics `--llm_api_key "$MARITACA_API_KEY"` from a shell where
    # the variable is unset; it must be treated as absent, not passed through.
    with pytest.raises(SystemExit):
        _batch_client("", "https://chat.maritaca.ai/api")


def test_is_batch_endpoint():
    assert is_batch_endpoint("https://chat.maritaca.ai/api") is True
    assert is_batch_endpoint("https://chat.maritaca.ai/api/") is True
    assert is_batch_endpoint("https://api.openai.com/v1") is False
    assert is_batch_endpoint("http://localhost:11434") is False
    assert is_batch_endpoint(None) is False
    assert is_batch_endpoint("") is False


def test_custom_id_to_idx():
    assert _custom_id_to_idx("req-0") == 0
    assert _custom_id_to_idx("req-42") == 42
    assert _custom_id_to_idx("req-x") is None
    assert _custom_id_to_idx("foo") is None
    assert _custom_id_to_idx(None) is None
    assert _custom_id_to_idx(7) is None


def test_extract_batch_content_success():
    record = {"custom_id": "req-0", "response": {"status_code": 200, "body": {"choices": [{"message": {"content": " B "}}]}}}
    assert _extract_batch_content(record) == "B"


def test_extract_batch_content_error_field():
    record = {"custom_id": "req-0", "error": {"message": "boom"}}
    result = _extract_batch_content(record)
    assert isinstance(result, Exception)


def test_extract_batch_content_non_200():
    record = {"custom_id": "req-0", "response": {"status_code": 429, "body": {"error": "rate limited"}}}
    result = _extract_batch_content(record)
    assert isinstance(result, Exception)


def test_extract_batch_content_malformed():
    record = {"custom_id": "req-0", "response": {"status_code": 200, "body": {}}}
    result = _extract_batch_content(record)
    assert isinstance(result, Exception)


def _make_task(question, options):
    return {
        "item": {"article_id": "a1"},
        "question": question,
        "options": options,
        "correct_letter": "A",
    }


def test_run_batch_requests_maps_by_custom_id(tmp_path):
    """A completed batch maps out-of-order answers and errors back to task order."""
    tasks = [
        _make_task("Q0?", ["a0", "b0", "c0", "d0"]),
        _make_task("Q1?", ["a1", "b1", "c1", "d1"]),
        _make_task("Q2?", ["a2", "b2", "c2", "d2"]),
    ]

    # Output file: req-2 and req-0 succeed, deliberately out of input order.
    output_jsonl = "\n".join(
        [
            json.dumps(
                {"custom_id": "req-2", "response": {"status_code": 200, "body": {"choices": [{"message": {"content": "C"}}]}}}
            ),
            json.dumps(
                {"custom_id": "req-0", "response": {"status_code": 200, "body": {"choices": [{"message": {"content": "A"}}]}}}
            ),
        ]
    )
    # Error file: req-1 failed.
    error_jsonl = json.dumps({"custom_id": "req-1", "error": {"message": "content policy"}})

    batch_obj = SimpleNamespace(
        id="batch-1",
        status="completed",
        output_file_id="file-out",
        error_file_id="file-err",
        request_counts=SimpleNamespace(completed=2, failed=1),
    )

    def fake_file_content(file_id):
        return SimpleNamespace(text=output_jsonl if file_id == "file-out" else error_jsonl)

    client = mock.Mock()
    client.files.create.return_value = SimpleNamespace(id="file-in")
    client.batches.create.return_value = batch_obj
    client.batches.retrieve.return_value = batch_obj
    client.files.content.side_effect = fake_file_content

    with mock.patch("latamqa.eval_mcq._batch_client", return_value=client):
        responses = _run_batch_requests(
            "openai/sabia-4-thinking",
            DEFAULT_PROPMT_TEMPLATE,
            tasks,
            temperature=0.0,
            llm_api_key="key",
            llm_uri="https://chat.maritaca.ai/api",
            results_dir=tmp_path,
            tag="es-la_regional_test",
            poll_interval=0,
        )

    # Answers land in dataset order despite out-of-order output; the errored
    # question is surfaced as an exception, not a silent None.
    assert responses[0] == "A"
    assert isinstance(responses[1], Exception)
    assert responses[2] == "C"

    # The batch was created once, already completed, so no polling occurred.
    client.batches.retrieve.assert_not_called()
    client.batches.create.assert_called_once()
    _, batch_kwargs = client.batches.create.call_args
    assert batch_kwargs["completion_window"] == "24h"
    assert batch_kwargs["endpoint"] == "/v1/chat/completions"
    assert batch_kwargs["input_file_id"] == "file-in"

    # The input JSONL was written with one line per task, bare model name, and
    # positional custom_ids.
    input_path = tmp_path / "batch_input_es-la_regional_test.jsonl"
    lines = [json.loads(ln) for ln in input_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 3
    assert [ln["custom_id"] for ln in lines] == ["req-0", "req-1", "req-2"]
    assert all(ln["body"]["model"] == "sabia-4-thinking" for ln in lines)
    assert all(ln["url"] == "/v1/chat/completions" for ln in lines)
    # Enough token budget for a thinking model to reason before answering.
    assert all(ln["body"]["max_tokens"] == DEFAULT_MAX_TOKENS for ln in lines)


def test_run_batch_requests_polls_until_terminal(tmp_path):
    """An in-progress batch is polled via batches.retrieve until it completes."""
    tasks = [_make_task("Q0?", ["a", "b", "c", "d"])]
    output_jsonl = json.dumps(
        {"custom_id": "req-0", "response": {"status_code": 200, "body": {"choices": [{"message": {"content": "D"}}]}}}
    )

    in_progress = SimpleNamespace(id="b", status="in_progress", output_file_id=None, error_file_id=None, request_counts=None)
    completed = SimpleNamespace(
        id="b",
        status="completed",
        output_file_id="file-out",
        error_file_id=None,
        request_counts=SimpleNamespace(completed=1, failed=0),
    )

    client = mock.Mock()
    client.files.create.return_value = SimpleNamespace(id="file-in")
    client.batches.create.return_value = in_progress
    # First poll still in progress, second poll completed.
    client.batches.retrieve.side_effect = [in_progress, completed]
    client.files.content.return_value = SimpleNamespace(text=output_jsonl)

    with (
        mock.patch("latamqa.eval_mcq._batch_client", return_value=client),
        mock.patch("latamqa.eval_mcq.time.sleep") as sleep,
    ):
        responses = _run_batch_requests(
            "openai/sabia-4-thinking",
            DEFAULT_PROPMT_TEMPLATE,
            tasks,
            temperature=0.0,
            llm_api_key="key",
            llm_uri="https://chat.maritaca.ai/api",
            results_dir=tmp_path,
            tag="poll_test",
            poll_interval=5,
        )

    assert responses[0] == "D"
    assert client.batches.retrieve.call_count == 2
    sleep.assert_called_with(5)
