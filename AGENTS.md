# AGENTS.md

Compact guidance for OpenCode sessions in this repo.

## What this is

A single-file Python CLI that benchmarks Kimchi inference models through the OpenAI-compatible chat completions API at `https://llm.kimchi.dev/openai/v1`. Implementation lives in `benchmark.py`; tests are `test_benchmark*.py` alongside it.

## Development commands

```bash
pip install -r requirements.txt   # runtime dep is requests>=2.32.0
pytest                            # all tests
pytest test_benchmark.py::test_retry_on_503   # single test
pytest test_benchmark_cli.py      # CLI / env / main() flow
pytest test_benchmark_metrics.py  # metrics, outputs, comparison
```

Run the real benchmark only with an API key:

```bash
KIMCHI_API_KEY=<key> python benchmark.py
python benchmark.py --dry-run --verbose   # validate config without API calls
```

No lint, formatter, typechecker, build tool, or task runner is configured. Python >= 3.10 required (uses PEP 604 `X | Y` unions).

## API authentication

Resolve order:

1. `KIMCHI_API_KEY` environment variable
2. `~/.config/opencode/opencode.json` → `provider.kimchi.options.apiKey`

Use `KIMCHI_API_KEY` in tests and CI to avoid touching user config.

## CI: GitHub Actions workflow

`.github/workflows/benchmark.yaml` runs the benchmark periodically and commits the
regenerated artifacts back to the repository.

- **Triggers**: `workflow_dispatch` (manual) and a daily schedule (`cron: "0 0 * * *"`, 00:00 UTC).
- **Runner**: `ubuntu-latest`, Python 3.11.
- **Secret**: `KIMCHI_API_KEY` must be set in the repository's Actions secrets; the workflow exposes it to the run via `env:` so the existing `get_api_key()` resolver picks it up.
- **Artifacts regenerated**: both `index.json` and `index.html` are produced by running `python benchmark.py --format json` and `python benchmark.py --format html` in sequence.
- **Exit-code handling**: each benchmark step uses `continue-on-error: true` so that an all-fail run (exit code 1) does not fail the workflow; setup or uncaught errors still fail it.
- **Commit/push**: only commits when `index.json` / `index.html` actually change, with message `chore: update benchmark results [actions]`.

## Architecture in `benchmark.py`

- **Config (`parse_args`, `_resolve_*`, `_load_prompts_file`)**: CLI flags override env vars (`BENCHMARK_*`) override optional JSON `--config`.
- **Prompts (`_resolve_prompts`, `DEFAULT_PROMPTS`)**: built-in prompt bank (10 prompts), `--single-prompt`, or a JSON prompts file (`--prompts-file`). Each prompt has `id`, `category`, `text`.
- **Execution (`benchmark_model`, `_request_with_retry`, `fetch_available_models`)**: streaming requests to `/chat/completions`; retries 5xx/429/timeouts with exponential backoff; 400 and 410 are permanent.
- **Metrics (`compute_metrics`, `aggregate_model_results`, `compute_*_summary`)**: per-run metrics (`tokens_per_sec`, `ms_per_token`, `generation_time_sec`) plus per-model medians and per-prompt/per-category summaries.
- **Output (`write_*_output`)**: JSON, CSV, Markdown, or self-contained styled HTML with inline-SVG charts (no external CDN). Format inferred from `--output` extension if `--format` omitted.
- **Comparison (`load_previous_results`, `compare_results`)**: if the JSON output path already exists, per-model medians get classified as `new` / `improved` / `regressed` / `unchanged` / `missing`.

## Key behaviors

- Default models: `glm-5.2-fp8`, `kimi-k2.6`, `kimi-k2.7`, `minimax-m2.7`, `minimax-m3`, `nemotron-3-ultra-fp4`, `deepseek-v4-flash`. Pass `--models` or set `BENCHMARK_MODELS` to override.
- Model list is intersected with `/models`; if unreachable, `--models` is used as fallback.
- Concurrency: `ThreadPoolExecutor` controlled by `--workers` (default 4).
- `--runs` repeats every `(model, prompt)` pair; detailed output records `run_index`.
- Time-to-first-token is request start to first SSE chunk containing `content` or `reasoning_content`.
- Computed metrics are skipped when required raw values are missing, zero, or non-positive.
- Exit code is `0` if any model had at least one successful run, else `1`.
- Checked-in result artifacts `index.json` and `index.html` are intentionally part of the repo record; do not add them to `.gitignore`.

## Testing notes

- No `conftest.py`, no fixtures, no snapshots, no integration tests.
- Tests use `unittest.mock.patch` heavily for `requests.request`, `benchmark_model`, `fetch_available_models`, `get_api_key`, `write_output`, etc.
- Do not rely on real API calls in tests; do not run `python benchmark.py` without a key (or `--dry-run`) in CI.

## Repo conventions

- Flat single-module layout; do not add `__init__.py` unless packaging the tool.
- Default prompts and models are hardcoded at the top of `benchmark.py`.
- Branch is `master` (not `main`).
- Remote: `github.com:lqvp/benchmark.git` (MIT, 2026).
