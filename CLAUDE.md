# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a small Python CLI for benchmarking Kimchi inference models through the OpenAI-compatible chat completions API at `https://llm.kimchi.dev/openai/v1`. The implementation lives almost entirely in `benchmark.py`, with pytest test modules alongside it.

## Development Commands

- Install dependencies: `pip install -r requirements.txt`
- Run all tests: `pytest`
- Run a single test file: `pytest test_benchmark.py`
- Run a single test: `pytest test_benchmark.py::test_retry_on_503`
- Run the benchmark (requires API key): `KIMCHI_API_KEY=<key> python benchmark.py`
- Dry run without API calls: `python benchmark.py --dry-run --verbose`

The only runtime dependency is `requests`. Tests use `pytest` and the standard library.

## API Authentication

The benchmark needs a Kimchi API key, resolved in this order:

1. `KIMCHI_API_KEY` environment variable
2. `provider.kimchi.options.apiKey` in `~/.config/opencode/opencode.json`

Use `KIMCHI_API_KEY` in tests and CI to avoid touching user config.

## Code Architecture

`benchmark.py` is organized into a few conceptual layers:

- **Configuration (`parse_args`, `_resolve_*`, `_load_prompts_file`)**: CLI args override environment variables, which override an optional JSON config file (`--config`).
- **Prompt resolution (`_resolve_prompts`, `DEFAULT_PROMPTS`)**: Prompts can come from the built-in prompt suite, a single custom prompt (`--single-prompt`), or a JSON prompts file. Each prompt is a dict with `id`, `category`, and `text`.
- **Execution (`benchmark_model`, `_request_with_retry`, `fetch_available_models`)**: Requests stream from `/chat/completions` with `stream: true` and `include_usage: true`. `_request_with_retry` retries transient HTTP errors (5xx, 429) and timeouts with exponential backoff, but treats 410 and 400 "no provider" as permanent.
- **Metrics (`compute_metrics`, `aggregate_model_results`, `compute_*_summary`)**: Raw timing and token usage from the API are turned into per-run metrics (`tokens_per_sec`, `ms_per_token`, `generation_time_sec`) and then into per-model medians and per-prompt/per-category summaries.
- **Output (`write_*_output`)**: Results are written as JSON (default), CSV, Markdown, or a self-contained styled HTML report. The output path extension implies the format when `--format` is omitted.
- **Comparison (`load_previous_results`, `compare_results`)**: If a previous JSON results file exists at the output path, per-model medians are compared to produce `new`, `improved`, `regressed`, `unchanged`, or `missing` statuses.

Key data structures:

- `BenchmarkResult`: one row per `(model, prompt, run_index)`.
- `ModelAggregate`: median metrics and success counts across all prompts for one model.

## Key Behaviors Worth Knowing

- The model list is intersected with the live `/models` endpoint. If the API is unreachable, `--models` is used as a fallback.
- `--workers` controls concurrency via `ThreadPoolExecutor`.
- `--runs` repeats every `(model, prompt)` pair that many times; the detailed output records `run_index`.
- Time-to-first-token is measured from request start to the first SSE chunk containing either `content` or `reasoning_content`.
- Computed metrics are skipped when required raw values are missing, zero, or non-positive.
- Exit code is `0` if any model had at least one successful run, otherwise `1`.

## Testing Notes

- Tests heavily use `unittest.mock.patch` to mock `requests.request`, `benchmark.benchmark_model`, and API helpers.
- Error-category tests rely on mocked HTTP status codes; do not rely on a real API key in unit tests.
- `test_benchmark_cli.py` and `test_benchmark_metrics.py` cover CLI parsing, prompt resolution, metric math, and output writers.
