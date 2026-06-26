# Kimchi Model Benchmark

Benchmark Kimchi inference models through the OpenAI-compatible chat completions API across a diverse prompt suite.

## Requirements

- Python 3.10+
- `requests`

## Setup

```bash
pip install -r requirements.txt
```

## API Key

The script reads the Kimchi API key from one of these sources (in order):

1. `KIMCHI_API_KEY` environment variable
2. `provider.kimchi.options.apiKey` in `~/.config/opencode/opencode.json`

## Usage

Run the full benchmark against the default model list and built-in prompt suite:

```bash
python benchmark.py
```

Use CLI flags to configure the run:

```bash
# Benchmark specific models with a single custom prompt
python benchmark.py --models minimax-m3,kimi-k2.7 --single-prompt --prompt "Explain quantum computing in one paragraph"

# Run each (model, prompt) pair multiple times for stability
python benchmark.py --runs 3 --workers 2

# Use only a random sample of the built-in prompts
python benchmark.py --prompts-count 5 --prompts-seed 42

# Load prompts from a JSON file
python benchmark.py --prompts-file prompts.json

# Tune generation settings
python benchmark.py --max-tokens 512 --temperature 0.5

# Write results as CSV, Markdown, or HTML
python benchmark.py --output results.csv --format csv
python benchmark.py --output report.md --format markdown
python benchmark.py --output report.html --format html

# Print configuration and exit without making API calls
python benchmark.py --dry-run --verbose
```

Environment variables are supported and have lower precedence than CLI flags:

```bash
BENCHMARK_PROMPT="Explain quantum computing in one paragraph" python benchmark.py --single-prompt
BENCHMARK_MODELS="minimax-m3,kimi-k2.7" python benchmark.py
BENCHMARK_RUNS=3 python benchmark.py
BENCHMARK_FORMAT=csv python benchmark.py
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--models` | 6 default models | Comma-separated list of models to benchmark |
| `--single-prompt` | disabled | Use only the prompt from `--prompt` instead of the built-in prompt suite |
| `-p`, `--prompt` | palindromic substring prompt | Prompt used when `--single-prompt` is set; also read from `BENCHMARK_PROMPT` |
| `--prompts-file` | (none) | JSON file containing prompts (list of strings or objects with `id`/`category`/`text`, `prompt` is also accepted) |
| `-n`, `--prompts-count` | all prompts | Number of prompts to sample from the built-in set |
| `--prompts-seed` | (none) | Random seed for `--prompts-count` sampling |
| `--runs` | `1` | Number of repeated runs per `(model, prompt)` pair |
| `--max-tokens` | `256` | Maximum completion tokens per request |
| `-t`, `--temperature` | `0.3` | Sampling temperature |
| `-w`, `--workers` | `4` | Number of parallel workers |
| `-o`, `--output` | `benchmark_results` | Output base path; extension implies `--format` when omitted |
| `--format` | inferred from output suffix | Output file format: `json`, `csv`, `markdown`, or `html` |
| `--dry-run` | disabled | Print configuration and model/prompt list, then exit without API calls |
| `-v`, `--verbose` | disabled | Print extra diagnostic information |
| `--no-per-prompt-table` | disabled | Skip the verbose per-prompt results table |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KIMCHI_API_KEY` | (from opencode config) | Kimchi API key |
| `BENCHMARK_MODELS` | 6 default models | Comma-separated list of models to test |
| `BENCHMARK_PROMPT` | palindromic substring prompt | Prompt used with `--single-prompt` |
| `BENCHMARK_PROMPTS_FILE` | (none) | Path to a JSON prompts file |
| `BENCHMARK_PROMPTS_COUNT` | (all) | Number of built-in prompts to sample |
| `BENCHMARK_PROMPTS_SEED` | (none) | Random seed for prompt sampling |
| `BENCHMARK_RUNS` | `1` | Repeated runs per `(model, prompt)` pair |
| `BENCHMARK_MAX_TOKENS` | `256` | Maximum completion tokens per request |
| `BENCHMARK_TEMPERATURE` | `0.3` | Sampling temperature |
| `BENCHMARK_MAX_WORKERS` | `4` | Number of parallel workers |
| `BENCHMARK_OUTPUT` | `benchmark_results` | Output base path |
| `BENCHMARK_FORMAT` | `json` | Output file format: `json`, `csv`, `markdown`, or `html` |

## Output

- A per-model aggregated results table is printed to stdout.
- Per-prompt progress lines are printed as each run finishes, including the run index.
- A summary section is printed with success rate, median tokens/sec, median total time, and median time-to-first-token.
- A comparison table is printed when a previous JSON results file exists.
- Detailed results are written to a file based on `--output` and `--format`:
  - `benchmark_results.json` (default)
  - `benchmark_results.csv`
  - `benchmark_results.md`
  - `benchmark_results.html`

Use `--output` to change the base path; the appropriate extension is added automatically unless the path already has one matching the selected format.

### Repeated runs

When `--runs` is greater than `1`, every `(model, prompt)` pair is executed that many times. Aggregates remain per-model across all runs, while the detailed output records the `run_index` of each individual result.

## Metrics

### Raw metrics

- **Total(s)**: total elapsed time for the request
- **TFT(s)**: time to first visible/generated token (streaming), measured at the first chunk that contains `content` or `reasoning_content`
- **In / Out / Total**: token usage reported by the API

### Computed metrics

Computed only when the required raw values are available and positive:

- **Gen(s)**: generation time, `total_time_sec - time_to_first_token_sec`
- **Tok/s**: completion throughput, `completion_tokens / generation_time_sec`
- **TotTok/s**: total throughput, `total_tokens / total_time_sec`
- **ms/tok**: latency per completion token, `generation_time_sec / completion_tokens * 1000`

## Previous-Run Comparison

If `benchmark_results.json` exists before the benchmark starts, the script loads the previous per-model aggregates and compares them to the current run. The JSON output contains a `comparisons` object with per-model status and deltas.

Possible comparison statuses:

- `new`: model was not present in the previous run
- `improved`: overall faster/lower latency and/or higher throughput
- `regressed`: overall slower/higher latency and/or lower throughput
- `unchanged`: score is near zero (no meaningful change)
- `missing`: model or comparable metrics are absent from either run

## Output Formats

### JSON

The default output includes:

- `prompts`: the effective prompt list used
- `summary`: overall aggregate statistics
- `aggregates`: per-model `ModelAggregate` objects
- `prompt_summary`: per-prompt aggregate statistics
- `category_summary`: per-category aggregate statistics
- `results`: list of per-run `BenchmarkResult` objects
- `comparisons`: previous-run deltas (when a previous file exists)

### CSV

Two sections:

1. `# Aggregated results` — one row per model with median metrics.
2. `# Per-prompt results` — one row per run, including `run_index` and `prompt_category`.

### Markdown

A Markdown document with a summary section and a per-model aggregated results table suitable for pasting into reports or issue comments.

### HTML

A self-contained, styled HTML report with:

- Summary cards for models tested, success rate, and median timing/throughput metrics.
- A sort-ready aggregated results table per model.
- A detailed per-prompt results table with run index, status, timings, token counts, and error categories.
- A previous-run comparison table when a prior JSON results file exists.

The HTML file includes inline CSS and has no external dependencies, so it can be opened directly in a browser or attached to issues and emails.
