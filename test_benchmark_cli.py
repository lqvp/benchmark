"""CLI/usability tests for benchmark.py."""

import json
from unittest.mock import patch

import pytest

import benchmark
from benchmark import DEFAULT_MAX_TOKENS, DEFAULT_MODELS, DEFAULT_PROMPT, DEFAULT_RUNS, DEFAULT_TEMPERATURE, main, parse_args


def test_parse_args_defaults(monkeypatch):
    """Default argument values fall back to constants when env/config are absent."""
    for name in [
        "BENCHMARK_MODELS",
        "BENCHMARK_PROMPT",
        "BENCHMARK_MAX_TOKENS",
        "BENCHMARK_TEMPERATURE",
        "BENCHMARK_MAX_WORKERS",
        "BENCHMARK_OUTPUT",
        "BENCHMARK_FORMAT",
        "BENCHMARK_RUNS",
        "BENCHMARK_PROMPTS_COUNT",
        "BENCHMARK_PROMPTS_SEED",
    ]:
        monkeypatch.delenv(name, raising=False)

    args = parse_args([])

    assert args.models == DEFAULT_MODELS
    assert args.prompt == DEFAULT_PROMPT
    assert args.max_tokens == DEFAULT_MAX_TOKENS
    assert args.temperature == pytest.approx(DEFAULT_TEMPERATURE)
    assert args.workers == benchmark.MAX_WORKERS
    assert args.runs == DEFAULT_RUNS
    assert args.output == benchmark.Path("benchmark_results")
    assert args.format == "json"
    assert args.dry_run is False
    assert args.verbose is False
    assert args.single_prompt is False
    assert args.prompts_file is None
    assert args.prompts_count is None
    assert args.prompts_seed is None
    assert args.no_per_prompt_table is False


def test_cli_args_override_env(monkeypatch):
    """Explicit CLI flags take precedence over environment variables."""
    monkeypatch.setenv("BENCHMARK_MODELS", "env-a,env-b")
    monkeypatch.setenv("BENCHMARK_PROMPT", "env prompt")
    monkeypatch.setenv("BENCHMARK_MAX_TOKENS", "999")
    monkeypatch.setenv("BENCHMARK_TEMPERATURE", "0.9")
    monkeypatch.setenv("BENCHMARK_MAX_WORKERS", "8")
    monkeypatch.setenv("BENCHMARK_FORMAT", "csv")
    monkeypatch.setenv("BENCHMARK_RUNS", "5")

    args = parse_args([
        "--models", "cli-x,cli-y",
        "--prompt", "cli prompt",
        "--max-tokens", "111",
        "--temperature", "0.5",
        "--workers", "2",
        "--format", "markdown",
        "--runs", "3",
    ])

    assert args.models == ["cli-x", "cli-y"]
    assert args.prompt == "cli prompt"
    assert args.max_tokens == 111
    assert args.temperature == pytest.approx(0.5)
    assert args.workers == 2
    assert args.format == "markdown"
    assert args.runs == 3


def test_main_accepts_argv_list(capsys, monkeypatch):
    """main can be invoked with an explicit argv list and prints progress."""
    monkeypatch.setenv("KIMCHI_API_KEY", "key")
    result = benchmark.BenchmarkResult(
        model="m1",
        prompt_id="single_prompt",
        prompt_category="custom",
        run_index=0,
        success=True,
        total_time_sec=2.4,
        tokens_per_sec=105.0,
    )

    with patch("benchmark.fetch_available_models", return_value=["m1"]), patch(
        "benchmark.benchmark_model", return_value=result
    ), patch("benchmark.write_output") as write_output:
        rc = main(["--models", "m1", "--single-prompt"])

    assert rc == 0
    write_output.assert_called_once()
    out = capsys.readouterr().out
    assert "[1/1]" in out
    assert "m1" in out
    assert "OK" in out
    assert "run=0" in out
    assert "2.4s" in out
    assert "105" in out


def test_dry_run_skips_api_calls(capsys, monkeypatch):
    """--dry-run exits 0 after printing config without making API calls."""
    monkeypatch.delenv("KIMCHI_API_KEY", raising=False)

    with patch("benchmark.get_api_key") as get_key, patch(
        "benchmark.fetch_available_models"
    ) as fetch, patch("benchmark.benchmark_model") as bm:
        rc = main([
            "--models", "alpha,beta",
            "--prompt", "hello world",
            "--single-prompt",
            "--runs", "2",
            "--dry-run",
            "--verbose",
        ])

    assert rc == 0
    get_key.assert_not_called()
    fetch.assert_not_called()
    bm.assert_not_called()

    out = capsys.readouterr().out
    assert "Dry run" in out
    assert "alpha" in out
    assert "beta" in out
    assert "hello world" in out
    assert "Runs: 2" in out
    assert "Total jobs: 4" in out
    assert "json" in out.lower()


def test_progress_output_for_failed_model(capsys, monkeypatch):
    """Progress lines include error category for failed models."""
    monkeypatch.setenv("KIMCHI_API_KEY", "key")
    result = benchmark.BenchmarkResult(
        model="old",
        prompt_id="single_prompt",
        prompt_category="custom",
        run_index=0,
        success=False,
        error_category=benchmark.ErrorCategory.DEPRECATED,
    )

    with patch("benchmark.fetch_available_models", return_value=["old"]), patch(
        "benchmark.benchmark_model", return_value=result
    ), patch("benchmark.write_output"):
        rc = main(["--models", "old", "--single-prompt"])

    assert rc == 1
    out = capsys.readouterr().out
    assert "[1/1]" in out
    assert "old" in out
    assert "FAIL" in out
    assert "deprecated" in out.lower()


def test_single_prompt_uses_custom_prompt():
    """--single-prompt resolves to a single custom prompt entry."""
    args = parse_args(["--single-prompt", "--prompt", "custom prompt text"])
    prompts = benchmark._resolve_prompts(args)

    assert len(prompts) == 1
    assert prompts[0]["id"] == "single_prompt"
    assert prompts[0]["category"] == "custom"
    assert prompts[0]["text"] == "custom prompt text"


def test_prompts_count_sampling():
    """--prompts-count samples deterministically when --prompts-seed is given."""
    args = parse_args(["--prompts-count", "3", "--prompts-seed", "42"])
    prompts = benchmark._resolve_prompts(args)

    assert len(prompts) == 3

    args2 = parse_args(["--prompts-count", "3", "--prompts-seed", "42"])
    prompts2 = benchmark._resolve_prompts(args2)
    assert [p["id"] for p in prompts] == [p["id"] for p in prompts2]


def test_prompts_file_supports_text_and_prompt_keys(tmp_path):
    """Prompts file entries are accepted with either 'text' or 'prompt' keys."""
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps([
            {"id": "text_key", "category": "coding", "text": "from text"},
            {"id": "prompt_key", "category": "math", "prompt": "from prompt"},
            {"id": "both", "category": "reasoning", "text": "preferred", "prompt": "ignored"},
        ])
    )

    prompts = benchmark._load_prompts_file(path)
    assert len(prompts) == 3
    assert prompts[0]["text"] == "from text"
    assert prompts[1]["text"] == "from prompt"
    assert prompts[2]["text"] == "preferred"


def test_runs_must_be_positive():
    """--runs < 1 is rejected."""
    with pytest.raises(SystemExit):
        parse_args(["--runs", "0"])
