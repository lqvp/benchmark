"""Unit tests for benchmark.py retry, error categories, and main flow."""

import json
from unittest.mock import Mock, call, patch

import pytest

import benchmark
from benchmark import ErrorCategory, _request_with_retry, benchmark_model, get_api_key


def _make_response(status_code: int, text: str = "") -> Mock:
    """Build a minimal mocked requests.Response."""
    response = Mock()
    response.status_code = status_code
    response.text = text
    response.content = text.encode("utf-8")
    if text:
        response.json.return_value = json.loads(text)
    else:
        response.json.return_value = {}
    return response


def _make_prompt(text: str = "prompt") -> dict[str, str]:
    return {"id": "test_prompt", "category": "test", "text": text}


def test_retry_on_503():
    """503 responses are retried up to 3 times with exponential backoff."""
    side_effect = [
        _make_response(503),
        _make_response(503),
        _make_response(503),
        _make_response(200, '{"ok": true}'),
    ]
    with patch("benchmark.requests.request", side_effect=side_effect) as mock_request, patch(
        "benchmark.time.sleep"
    ) as mock_sleep:
        response = _request_with_retry("GET", "http://example.com")

    assert response.status_code == 200
    assert mock_request.call_count == 4
    mock_sleep.assert_has_calls([call(1), call(2), call(4)])


def test_no_retry_on_410():
    """410 is treated as permanent and not retried."""
    response = _make_response(410, '{"error": {"message": "model deprecated"}}')
    with patch("benchmark.requests.request", return_value=response) as mock_request, patch(
        "benchmark.time.sleep"
    ) as mock_sleep:
        result = _request_with_retry("GET", "http://example.com")

    assert result.status_code == 410
    assert mock_request.call_count == 1
    mock_sleep.assert_not_called()


def test_no_retry_on_400_no_providers():
    """400 is treated as permanent and not retried."""
    response = _make_response(400, '{"error": {"message": "no provider available"}}')
    with patch("benchmark.requests.request", return_value=response) as mock_request, patch(
        "benchmark.time.sleep"
    ) as mock_sleep:
        result = _request_with_retry("GET", "http://example.com")

    assert result.status_code == 400
    assert mock_request.call_count == 1
    mock_sleep.assert_not_called()


def test_timeout_retry():
    """Timeout exceptions are retried before returning a successful response."""
    side_effect = [
        benchmark.requests.exceptions.Timeout("timed out"),
        benchmark.requests.exceptions.Timeout("timed out"),
        _make_response(200, '{"ok": true}'),
    ]
    with patch("benchmark.requests.request", side_effect=side_effect) as mock_request, patch(
        "benchmark.time.sleep"
    ) as mock_sleep:
        response = _request_with_retry("GET", "http://example.com")

    assert response.status_code == 200
    assert mock_request.call_count == 3
    mock_sleep.assert_has_calls([call(1), call(2)])


def test_get_api_key_from_env(monkeypatch):
    """KIMCHI_API_KEY environment variable is returned first."""
    monkeypatch.setenv("KIMCHI_API_KEY", "env-secret")
    assert get_api_key() == "env-secret"


def test_get_api_key_from_config(tmp_path, monkeypatch):
    """API key is read from the opencode config when env var is absent."""
    monkeypatch.delenv("KIMCHI_API_KEY", raising=False)
    config_dir = tmp_path / ".config" / "opencode"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "opencode.json"
    config_file.write_text(
        json.dumps({"provider": {"kimchi": {"options": {"apiKey": "cfg-secret"}}}})
    )

    with patch("benchmark.Path.home", return_value=tmp_path):
        assert get_api_key() == "cfg-secret"


def test_get_api_key_missing(tmp_path, monkeypatch):
    """A clear error is raised when no API key can be found."""
    monkeypatch.delenv("KIMCHI_API_KEY", raising=False)
    with patch("benchmark.Path.home", return_value=tmp_path):
        with pytest.raises(RuntimeError, match="KIMCHI_API_KEY not found"):
            get_api_key()


def test_benchmark_model_410_deprecated():
    """HTTP 410 is classified as DEPRECATED."""
    response = _make_response(410, '{"error": {"message": "model deprecated"}}')
    with patch("benchmark.requests.request", return_value=response):
        result = benchmark_model("old-model", "key", _make_prompt(), 0, 256, 0.3)

    assert not result.success
    assert result.prompt_id == "test_prompt"
    assert result.prompt_category == "test"
    assert result.run_index == 0
    assert result.error_category == ErrorCategory.DEPRECATED


def test_benchmark_model_400_no_provider_unavailable():
    """HTTP 400 with a 'no provider' body is classified as UNAVAILABLE."""
    response = _make_response(400, '{"error": {"message": "no provider available"}}')
    with patch("benchmark.requests.request", return_value=response):
        result = benchmark_model("missing-model", "key", _make_prompt(), 0, 256, 0.3)

    assert not result.success
    assert result.error_category == ErrorCategory.UNAVAILABLE


def test_benchmark_model_503_transient():
    """HTTP 503 is classified as TRANSIENT after retries are exhausted."""
    side_effect = [
        _make_response(503),
        _make_response(503),
        _make_response(503),
        _make_response(503),
    ]
    with patch("benchmark.requests.request", side_effect=side_effect), patch("benchmark.time.sleep"):
        result = benchmark_model("busy-model", "key", _make_prompt(), 0, 256, 0.3)

    assert not result.success
    assert result.error_category == ErrorCategory.TRANSIENT


def test_benchmark_model_timeout():
    """Request timeout is classified as TIMEOUT."""
    with patch(
        "benchmark.requests.request",
        side_effect=benchmark.requests.exceptions.Timeout("timed out"),
    ), patch("benchmark.time.sleep"):
        result = benchmark_model("slow-model", "key", _make_prompt(), 0, 256, 0.3)

    assert not result.success
    assert result.error_category == ErrorCategory.TIMEOUT


def test_main_returns_zero_if_any_success():
    """main returns 0 when at least one model succeeded."""
    ok_result = benchmark.BenchmarkResult(
        model="ok",
        prompt_id="single_prompt",
        prompt_category="custom",
        success=True,
        total_time_sec=1.0,
        tokens_per_sec=10.0,
    )
    fail_result = benchmark.BenchmarkResult(
        model="fail",
        prompt_id="single_prompt",
        prompt_category="custom",
        success=False,
    )

    with patch("benchmark.get_api_key", return_value="key"), patch(
        "benchmark.fetch_available_models", return_value=["ok", "fail"]
    ), patch("benchmark.benchmark_model", side_effect=[ok_result, fail_result]), patch(
        "benchmark.print_aggregates"
    ), patch("benchmark.print_per_prompt_results"), patch("benchmark.print_summary"), patch(
        "benchmark.print_comparison"
    ), patch("benchmark.load_previous_results", return_value={}), patch("benchmark.write_output"):
        assert benchmark.main(["--models", "ok,fail", "--single-prompt"]) == 0


def test_main_returns_one_if_all_fail():
    """main returns 1 when every model failed."""
    fail_result = benchmark.BenchmarkResult(
        model="fail",
        prompt_id="single_prompt",
        prompt_category="custom",
        success=False,
    )

    with patch("benchmark.get_api_key", return_value="key"), patch(
        "benchmark.fetch_available_models", return_value=["fail"]
    ), patch("benchmark.benchmark_model", return_value=fail_result), patch(
        "benchmark.print_aggregates"
    ), patch("benchmark.print_per_prompt_results"), patch("benchmark.print_summary"), patch(
        "benchmark.print_comparison"
    ), patch("benchmark.load_previous_results", return_value={}), patch("benchmark.write_output"):
        assert benchmark.main(["--models", "fail", "--single-prompt"]) == 1
