"""Tests for benchmark metrics, output writers, and previous-run comparison."""

import json

import pytest

from benchmark import (
    BenchmarkResult,
    ModelAggregate,
    aggregate_model_results,
    compare_results,
    compute_category_summary,
    compute_metrics,
    compute_prompt_summary,
    compute_summary,
    load_previous_results,
    write_csv_output,
    write_json_output,
    write_markdown_output,
)


def make_result(
    model: str = "test-model",
    prompt_id: str = "prompt-1",
    prompt_category: str = "coding",
    run_index: int = 0,
    success: bool = True,
    total_time: float | None = 10.0,
    time_to_first_token: float | None = 2.0,
    prompt_tokens: int | None = 20,
    completion_tokens: int | None = 40,
    total_tokens: int | None = 60,
) -> BenchmarkResult:
    return BenchmarkResult(
        model=model,
        prompt_id=prompt_id,
        prompt_category=prompt_category,
        run_index=run_index,
        success=success,
        total_time_sec=total_time,
        time_to_first_token_sec=time_to_first_token,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def test_compute_metrics_full():
    """All derived throughput metrics are computed from raw timing and tokens."""
    result = make_result(total_time=10.0, time_to_first_token=2.0, completion_tokens=40, total_tokens=60)
    compute_metrics(result)

    assert result.generation_time_sec == pytest.approx(8.0)
    assert result.tokens_per_sec == pytest.approx(5.0)
    assert result.total_tokens_per_sec == pytest.approx(6.0)
    assert result.ms_per_token == pytest.approx(200.0)


def test_compute_metrics_missing_total_time():
    """No metrics are derived when total_time_sec is missing."""
    result = make_result(total_time=None)
    compute_metrics(result)

    assert result.generation_time_sec is None
    assert result.tokens_per_sec is None
    assert result.total_tokens_per_sec is None
    assert result.ms_per_token is None


def test_compute_metrics_zero_completion_tokens():
    """Per-completion throughput is skipped when no completion tokens were generated."""
    result = make_result(completion_tokens=0)
    compute_metrics(result)

    assert result.generation_time_sec is not None
    assert result.tokens_per_sec is None
    assert result.ms_per_token is None
    assert result.total_tokens_per_sec is not None


def test_compute_metrics_non_positive_generation_time():
    """Generation-derived metrics are skipped if total_time is not greater than TFT."""
    result = make_result(total_time=2.0, time_to_first_token=2.0)
    compute_metrics(result)

    assert result.generation_time_sec is None
    assert result.tokens_per_sec is None


def test_aggregate_model_results():
    """ModelAggregate medians are computed across a model's results."""
    results = [
        make_result(model="m1", total_time=5.0, time_to_first_token=1.0, completion_tokens=20, total_tokens=30),
        make_result(model="m1", total_time=15.0, time_to_first_token=3.0, completion_tokens=40, total_tokens=60),
        make_result(model="m1", success=False),
    ]
    for r in results:
        compute_metrics(r)

    agg = aggregate_model_results("m1", results)

    assert agg.total_runs == 3
    assert agg.successful_runs == 2
    assert agg.success_rate == pytest.approx(66.67, rel=1e-3)
    assert agg.median_total_time_sec == pytest.approx(10.0)
    assert agg.median_time_to_first_token_sec == pytest.approx(2.0)
    assert agg.categories_tested == ["coding"]


def test_compute_summary():
    """Summary contains success rate and medians across model aggregates."""
    results = [
        make_result(model="fast", total_time=5.0, completion_tokens=10, total_tokens=20),
        make_result(model="slow", total_time=15.0, completion_tokens=10, total_tokens=20),
        make_result(model="fail", success=False),
    ]
    for r in results:
        compute_metrics(r)

    aggregates = [
        aggregate_model_results("fast", [r for r in results if r.model == "fast"]),
        aggregate_model_results("slow", [r for r in results if r.model == "slow"]),
        aggregate_model_results("fail", [r for r in results if r.model == "fail"]),
    ]
    summary = compute_summary(aggregates)

    assert summary["total_models"] == 3
    assert summary["fully_successful"] == 2
    assert summary["success_rate_percent"] == pytest.approx(66.67, rel=1e-3)
    assert summary["median_tokens_per_sec"] == pytest.approx(2.0513, rel=1e-3)
    assert summary["median_total_time_sec"] == pytest.approx(10.0)


def test_compute_prompt_and_category_summary():
    """Prompt and category summaries group results correctly."""
    results = [
        make_result(prompt_id="p1", prompt_category="coding", total_time=4.0, completion_tokens=20),
        make_result(prompt_id="p1", prompt_category="coding", total_time=6.0, completion_tokens=20),
        make_result(prompt_id="p2", prompt_category="math", total_time=10.0, completion_tokens=20),
    ]
    for r in results:
        compute_metrics(r)

    prompt_summary = compute_prompt_summary(results)
    assert prompt_summary["p1"]["total_runs"] == 2
    assert prompt_summary["p1"]["successful"] == 2
    assert prompt_summary["p2"]["total_runs"] == 1

    category_summary = compute_category_summary(results)
    assert category_summary["coding"]["total_runs"] == 2
    assert category_summary["math"]["total_runs"] == 1


def test_write_json_output(tmp_path):
    """JSON output includes results, summaries, prompt/category breakdowns, and comparisons."""
    result = make_result(model="m1", total_time=10.0, time_to_first_token=2.0, completion_tokens=40)
    compute_metrics(result)
    agg = aggregate_model_results("m1", [result])
    summary = compute_summary([agg])
    comparisons = {"m1": {"status": "new"}}

    path = tmp_path / "out.json"
    write_json_output(path, DEFAULT_PROMPTS, [result], [agg], summary, comparisons)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["summary"]["fully_successful"] == 1
    assert data["aggregates"][0]["model"] == "m1"
    assert data["results"][0]["model"] == "m1"
    assert data["results"][0]["tokens_per_sec"] == pytest.approx(5.0)
    assert data["results"][0]["run_index"] == 0
    assert "prompt_summary" in data
    assert "category_summary" in data
    assert data["comparisons"]["m1"]["status"] == "new"


def test_write_csv_output(tmp_path):
    """CSV output has aggregate and per-run sections with all metrics."""
    result = make_result(model="m1", total_time=10.0, time_to_first_token=2.0, completion_tokens=40, run_index=1)
    compute_metrics(result)
    agg = aggregate_model_results("m1", [result])

    path = tmp_path / "out.csv"
    write_csv_output(path, [agg], [result])

    text = path.read_text(encoding="utf-8")
    assert "# Aggregated results" in text
    assert "# Per-prompt results" in text
    assert "run_index" in text
    assert "m1" in text
    assert "5.0" in text


def test_write_markdown_output(tmp_path):
    """Markdown output contains a summary and an aggregate table."""
    result = make_result(model="m1", total_time=10.0, time_to_first_token=2.0, completion_tokens=40)
    compute_metrics(result)
    agg = aggregate_model_results("m1", [result])
    summary = compute_summary([agg])

    path = tmp_path / "out.md"
    write_markdown_output(path, [agg], summary)

    text = path.read_text(encoding="utf-8")
    assert "# Benchmark Results" in text
    assert "| Model |" in text
    assert "Tok/s" in text
    assert "ms/tok" in text
    assert "m1" in text


def test_load_previous_results(tmp_path):
    """Previous aggregates are loaded and keyed by model name."""
    data = {
        "prompts": [],
        "aggregates": [
            {"model": "m1", "tokens_per_sec": 5.0, "success_rate": 100.0},
            {"model": "m2", "success_rate": 50.0},
        ],
    }
    path = tmp_path / "benchmark_results.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    previous = load_previous_results(path)
    assert previous["m1"]["tokens_per_sec"] == 5.0
    assert "m2" in previous


def test_load_previous_results_missing(tmp_path):
    """Missing files return an empty dictionary."""
    assert load_previous_results(tmp_path / "does_not_exist.json") == {}


def test_compare_results_new_and_missing():
    """Models absent from previous run are new; absent from current are missing."""
    result = make_result(model="m1", total_time=10.0, time_to_first_token=2.0, completion_tokens=40)
    compute_metrics(result)
    agg = aggregate_model_results("m1", [result])
    previous = {"m2": {"model": "m2", "success": True}}

    comparisons = compare_results([agg], previous)
    assert comparisons["m1"]["status"] == "new"
    assert comparisons["m2"]["status"] == "missing"


def test_compare_results_improved():
    """Faster total time, lower TFT, and higher throughput are marked improved."""
    result = make_result(model="m1", total_time=8.0, time_to_first_token=1.5, completion_tokens=80)
    compute_metrics(result)
    agg = aggregate_model_results("m1", [result])
    previous = {
        "m1": {
            "model": "m1",
            "success": True,
            "median_total_time_sec": 10.0,
            "median_time_to_first_token_sec": 2.0,
            "median_tokens_per_sec": 5.0,
        }
    }

    comparisons = compare_results([agg], previous)
    comp = comparisons["m1"]
    assert comp["status"] == "improved"
    assert comp["median_total_time_sec_delta"] == pytest.approx(-2.0)
    assert comp["median_time_to_first_token_sec_delta"] == pytest.approx(-0.5)
    assert comp["median_tokens_per_sec_delta"] == pytest.approx(7.3077, rel=1e-3)


def test_compare_results_regressed():
    """Slower total time, higher TFT, and lower throughput are marked regressed."""
    result = make_result(model="m1", total_time=12.0, time_to_first_token=2.5, completion_tokens=20)
    compute_metrics(result)
    agg = aggregate_model_results("m1", [result])
    previous = {
        "m1": {
            "model": "m1",
            "success": True,
            "median_total_time_sec": 10.0,
            "median_time_to_first_token_sec": 2.0,
            "median_tokens_per_sec": 5.0,
        }
    }

    comparisons = compare_results([agg], previous)
    assert comparisons["m1"]["status"] == "regressed"


def test_compare_results_unchanged():
    """Near-zero score is reported as unchanged."""
    result = make_result(model="m1", total_time=10.0, time_to_first_token=2.0, completion_tokens=40)
    compute_metrics(result)
    agg = aggregate_model_results("m1", [result])
    previous = {
        "m1": {
            "model": "m1",
            "success": True,
            "median_total_time_sec": 10.0,
            "median_time_to_first_token_sec": 2.0,
            "median_tokens_per_sec": 5.0,
        }
    }

    comparisons = compare_results([agg], previous)
    assert comparisons["m1"]["status"] == "unchanged"


def test_compare_results_missing_metrics():
    """Runs missing comparable metrics are marked missing."""
    result = make_result(model="m1", total_time=10.0, completion_tokens=40)
    compute_metrics(result)
    agg = aggregate_model_results("m1", [result])
    previous = {"m1": {"model": "m1", "success": True, "median_total_time_sec": 10.0}}

    comparisons = compare_results([agg], previous)
    assert comparisons["m1"]["status"] == "missing"


DEFAULT_PROMPTS = [
    {"id": "p1", "category": "coding", "text": "hello"},
]
