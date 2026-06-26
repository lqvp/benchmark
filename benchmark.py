#!/usr/bin/env python3
"""Benchmark all available Kimchi models via the OpenAI-compatible API."""

from __future__ import annotations

import argparse
import csv
import enum
import html
import io
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import requests


class ErrorCategory(str, enum.Enum):
    DEPRECATED = "DEPRECATED"
    UNAVAILABLE = "UNAVAILABLE"
    TRANSIENT = "TRANSIENT"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


API_BASE_URL = "https://llm.kimchi.dev/openai/v1"
MODELS_ENDPOINT = f"{API_BASE_URL}/models"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat/completions"

# Prompt bank — diverse categories to reduce single-prompt variance

DEFAULT_PROMPT = (
    "Write a Python function that finds the longest palindromic substring of a given string. "
    "Explain the time and space complexity of your solution. Keep the answer concise."
)

DEFAULT_PROMPTS: list[dict[str, str]] = [
    {
        "id": "coding_palindrome",
        "category": "coding",
        "text": DEFAULT_PROMPT,
    },
    {
        "id": "coding_debug",
        "category": "debugging",
        "text": (
            "Find and fix all bugs in this Python code, then explain each fix:\n\n"
            "def merge_sorted(a, b):\n"
            "    result = []\n"
            "    i = j = 0\n"
            "    while i < len(a) and j < len(b):\n"
            "        if a[i] < b[j]:\n"
            "            result.append(a[i])\n"
            "            i += 1\n"
            "        else:\n"
            "            result.append(b[j])\n"
            "    result.extend(a[i:])\n"
            "    result.extend(b[j:])\n"
            "    return result"
        ),
    },
    {
        "id": "math_reasoning",
        "category": "math",
        "text": (
            "A factory produces widgets. Machine A produces 120 widgets/hour with a 2% defect rate. "
            "Machine B produces 80 widgets/hour with a 5% defect rate. "
            "If both run for 8 hours, what is the overall defect rate for the combined output? "
            "Show your work step by step."
        ),
    },
    {
        "id": "logic_puzzle",
        "category": "reasoning",
        "text": (
            "You have 9 coins; one is counterfeit and slightly heavier. "
            "Using a balance scale, what is the minimum number of weighings needed to always identify "
            "the counterfeit coin? Describe the strategy."
        ),
    },
    {
        "id": "data_structures",
        "category": "cs_concepts",
        "text": (
            "Compare hash tables and balanced binary search trees. "
            "For each, give the average-case time complexity for insert, delete, and lookup, "
            "then describe two concrete scenarios where you would prefer one over the other."
        ),
    },
    {
        "id": "system_design",
        "category": "design",
        "text": (
            "Design a URL shortener service that handles 10,000 writes/second and 100,000 reads/second. "
            "Describe the key components, the data model, and how you would scale it. Be concise."
        ),
    },
    {
        "id": "sql_query",
        "category": "coding",
        "text": (
            "Given tables: orders(id, customer_id, amount, created_at) and "
            "customers(id, name, country). "
            "Write a SQL query that returns the top 3 countries by total revenue in the last 30 days, "
            "including the number of orders and average order value per country."
        ),
    },
    {
        "id": "concurrency",
        "category": "cs_concepts",
        "text": (
            "Explain the difference between a mutex and a semaphore. "
            "Give a concrete Python example of a race condition and show how to fix it "
            "using threading.Lock. Keep it brief."
        ),
    },
    {
        "id": "regex_task",
        "category": "coding",
        "text": (
            "Write a Python function using regex that validates and parses a log line of the format:\n"
            "  [2024-01-15 08:23:11] ERROR (module.submodule): message text here\n"
            "Return a dict with keys: timestamp, level, module, message. "
            "Handle malformed lines gracefully."
        ),
    },
    {
        "id": "algorithm_complexity",
        "category": "reasoning",
        "text": (
            "Explain why quicksort has O(n²) worst-case but O(n log n) average-case time complexity. "
            "What input pattern triggers the worst case, and how does randomized pivot selection mitigate it?"
        ),
    },
]

MAX_WORKERS = 4
REQUEST_TIMEOUT = 120
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.3
DEFAULT_RUNS = 1
DEFAULT_FORMAT = "json"
DEFAULT_PROMPTS_COUNT = len(DEFAULT_PROMPTS)
OUTPUT_FORMATS = ("json", "csv", "markdown", "html")

DEFAULT_MODELS = [
    "glm-5.2-fp8",
    "kimi-k2.6",
    "kimi-k2.7",
    "minimax-m2.7",
    "minimax-m3",
    "nemotron-3-ultra-fp4",
]

MODELS = [m.strip() for m in os.environ.get("BENCHMARK_MODELS", "").split(",") if m.strip()] or DEFAULT_MODELS


@dataclass
class BenchmarkResult:
    model: str
    prompt_id: str
    prompt_category: str
    success: bool
    run_index: int = 0
    total_time_sec: float | None = None
    time_to_first_token_sec: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    generation_time_sec: float | None = None
    tokens_per_sec: float | None = None
    total_tokens_per_sec: float | None = None
    ms_per_token: float | None = None
    response_snippet: str = ""
    error: str = ""
    error_category: str = ""


@dataclass
class ModelAggregate:
    """Median metrics across all prompt runs for a single model."""

    model: str
    total_runs: int
    successful_runs: int
    success_rate: float
    median_total_time_sec: float | None = None
    median_time_to_first_token_sec: float | None = None
    median_tokens_per_sec: float | None = None
    median_ms_per_token: float | None = None
    median_total_tokens_per_sec: float | None = None
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    categories_tested: list[str] = field(default_factory=list)

    def to_display_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_model_results(model: str, results: list[BenchmarkResult]) -> ModelAggregate:
    """Compute median metrics across multiple prompt runs for one model."""
    total = len(results)
    successes = [r for r in results if r.success]
    success_count = len(successes)

    def med(values: list[float | None]) -> float | None:
        cleaned = [v for v in values if v is not None]
        return median(cleaned) if cleaned else None

    agg = ModelAggregate(
        model=model,
        total_runs=total,
        successful_runs=success_count,
        success_rate=round(success_count / total * 100, 2) if total else 0.0,
        categories_tested=sorted({r.prompt_category for r in results}),
    )
    if successes:
        agg.median_total_time_sec = med([r.total_time_sec for r in successes])
        agg.median_time_to_first_token_sec = med([r.time_to_first_token_sec for r in successes])
        agg.median_tokens_per_sec = med([r.tokens_per_sec for r in successes])
        agg.median_ms_per_token = med([r.ms_per_token for r in successes])
        agg.median_total_tokens_per_sec = med([r.total_tokens_per_sec for r in successes])
        pt = [r.prompt_tokens for r in successes if r.prompt_tokens is not None]
        ct = [r.completion_tokens for r in successes if r.completion_tokens is not None]
        agg.total_prompt_tokens = sum(pt) if pt else None
        agg.total_completion_tokens = sum(ct) if ct else None
    return agg


def get_api_key() -> str:
    env_key = os.environ.get("KIMCHI_API_KEY")
    if env_key:
        return env_key

    config_path = Path.home() / ".config" / "opencode" / "opencode.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            key = config.get("provider", {}).get("kimchi", {}).get("options", {}).get("apiKey")
            if key:
                return key
        except (json.JSONDecodeError, OSError):
            pass

    raise RuntimeError(
        "KIMCHI_API_KEY not found. Set the KIMCHI_API_KEY environment variable "
        "or ensure provider.kimchi.options.apiKey is set in ~/.config/opencode/opencode.json"
    )


def _request_with_retry(method: str, url: str, **kwargs: Any) -> requests.Response:
    retries = 3
    last_response: requests.Response | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.request(method, url, **kwargs)
            if response.status_code in (400, 410):
                return response
            if response.status_code >= 500 or response.status_code == 429:
                last_response = response
                if attempt < retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                return response
            return response
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
                continue
            raise


def fetch_available_models(api_key: str) -> list[str]:
    response = _request_with_retry(
        "GET",
        MODELS_ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return sorted(item["id"] for item in data.get("data", []))


def compute_metrics(result: BenchmarkResult) -> None:
    if result.total_time_sec is not None and result.total_time_sec > 0:
        if result.total_tokens is not None and result.total_tokens > 0:
            result.total_tokens_per_sec = result.total_tokens / result.total_time_sec

    if result.total_time_sec is None or result.time_to_first_token_sec is None:
        return
    if result.total_time_sec <= 0 or result.time_to_first_token_sec <= 0:
        return

    generation_time = result.total_time_sec - result.time_to_first_token_sec
    if generation_time <= 0:
        return
    result.generation_time_sec = generation_time

    if result.completion_tokens is not None and result.completion_tokens > 0:
        result.tokens_per_sec = result.completion_tokens / generation_time
        result.ms_per_token = (generation_time / result.completion_tokens) * 1000


def benchmark_model(
    model: str,
    api_key: str,
    prompt_entry: dict[str, str],
    run_index: int,
    max_tokens: int,
    temperature: float,
    verbose: bool = False,
) -> BenchmarkResult:
    """Run a single benchmark against one model with one prompt."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_entry["text"]}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    result = BenchmarkResult(
        model=model,
        prompt_id=prompt_entry["id"],
        prompt_category=prompt_entry["category"],
        success=False,
        run_index=run_index,
    )
    start_time = time.perf_counter()
    first_token_time: float | None = None
    chunks: list[str] = []

    response: requests.Response | None = None
    try:
        response = _request_with_retry(
            "POST",
            CHAT_ENDPOINT,
            headers=headers,
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            body = ""
            try:
                parsed = response.json()
                error_value = parsed.get("error") if isinstance(parsed, dict) else parsed
                if isinstance(error_value, dict):
                    body = error_value.get("message") or json.dumps(error_value, ensure_ascii=False)
                elif isinstance(error_value, list):
                    body = json.dumps(error_value, ensure_ascii=False)
                else:
                    body = str(error_value) if error_value is not None else response.text
            except ValueError:
                body = response.text or response.content.decode("utf-8", errors="replace")
            result.error = f"HTTP {response.status_code}: {str(body)[:300]}".strip()
            if response.status_code == 410:
                result.error_category = ErrorCategory.DEPRECATED
            elif response.status_code == 400 and "no provider" in body.lower():
                result.error_category = ErrorCategory.UNAVAILABLE
            elif response.status_code >= 500 or response.status_code == 429:
                result.error_category = ErrorCategory.TRANSIENT
            else:
                result.error_category = ErrorCategory.UNKNOWN
            return result

        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data: "):
                continue
            data_str = decoded[len("data: "):]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or ""

            # Measure first visible/generated token, not the usage-only chunk.
            if first_token_time is None and (content or reasoning):
                first_token_time = time.perf_counter()

            if reasoning:
                chunks.append(reasoning)
            if content:
                chunks.append(content)

            usage = chunk.get("usage")
            if usage:
                result.prompt_tokens = usage.get("prompt_tokens")
                result.completion_tokens = usage.get("completion_tokens")
                result.total_tokens = usage.get("total_tokens")

        result.total_time_sec = time.perf_counter() - start_time
        result.time_to_first_token_sec = (
            first_token_time - start_time if first_token_time else None
        )
        result.response_snippet = "".join(chunks).strip().replace("\n", " ")[:200]
        result.success = True
        compute_metrics(result)
    except requests.exceptions.Timeout as exc:
        result.error_category = ErrorCategory.TIMEOUT
        result.error = f"Request error: {exc}"
    except requests.exceptions.ConnectionError as exc:
        result.error_category = ErrorCategory.TRANSIENT
        result.error = f"Request error: {exc}"
    except requests.RequestException as exc:
        result.error_category = ErrorCategory.UNKNOWN
        result.error = f"Request error: {exc}"
    except Exception as exc:  # noqa: BLE001
        result.error_category = ErrorCategory.UNKNOWN
        result.error = f"Unexpected error: {type(exc).__name__}: {exc}"
    finally:
        if response is not None:
            response.close()

    return result


def _float_values(results: list[BenchmarkResult], attr: str) -> list[float]:
    values: list[float] = []
    for result in results:
        value = getattr(result, attr)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def _compute_group_summary(
    results: list[BenchmarkResult],
    key_attr: str,
) -> dict[str, dict[str, Any]]:
    """Compute aggregate statistics grouped by an attribute (e.g. prompt_id or category)."""
    groups: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        key = str(getattr(result, key_attr) or "")
        groups.setdefault(key, []).append(result)

    summaries: dict[str, dict[str, Any]] = {}
    for key, items in groups.items():
        successful = [r for r in items if r.success]
        total = len(items)
        ok_count = len(successful)

        total_times = _float_values(successful, "total_time_sec")
        tfts = _float_values(successful, "time_to_first_token_sec")
        tps_values = _float_values(successful, "tokens_per_sec")
        ms_values = _float_values(successful, "ms_per_token")
        completion_tokens = _float_values(successful, "completion_tokens")

        summaries[key] = {
            "total_runs": total,
            "successful": ok_count,
            "failed": total - ok_count,
            "success_rate_percent": round(ok_count / total * 100, 2) if total else 0.0,
            "median_total_time_sec": round(median(total_times), 4) if total_times else None,
            "median_time_to_first_token_sec": round(median(tfts), 4) if tfts else None,
            "median_tokens_per_sec": round(median(tps_values), 4) if tps_values else None,
            "median_ms_per_token": round(median(ms_values), 4) if ms_values else None,
            "median_completion_tokens": round(median(completion_tokens), 4) if completion_tokens else None,
        }

    return summaries


def compute_prompt_summary(results: list[BenchmarkResult]) -> dict[str, dict[str, Any]]:
    return _compute_group_summary(results, "prompt_id")


def compute_category_summary(results: list[BenchmarkResult]) -> dict[str, dict[str, Any]]:
    return _compute_group_summary(results, "prompt_category")


def print_aggregates(aggregates: list[ModelAggregate]) -> None:
    """Print aggregated per-model results table sorted by median total time."""
    sorted_agg = sorted(
        aggregates,
        key=lambda a: a.median_total_time_sec if a.median_total_time_sec is not None else float("inf"),
    )
    header = (
        f"{'Model':<28} {'OK/N':<7} {'SR%':<7} {'Med Total(s)':<14} "
        f"{'Med TFT(s)':<12} {'Med Tok/s':<11} {'Med ms/tok':<11} {'Categories'}"
    )
    print("\n=== Aggregated Results (median across prompts) ===")
    print(header)
    print("-" * len(header))

    for a in sorted_agg:
        ok_n = f"{a.successful_runs}/{a.total_runs}"
        sr = f"{a.success_rate:.1f}"
        total = f"{a.median_total_time_sec:.2f}" if a.median_total_time_sec is not None else "-"
        tft = f"{a.median_time_to_first_token_sec:.2f}" if a.median_time_to_first_token_sec is not None else "-"
        tps = f"{a.median_tokens_per_sec:.2f}" if a.median_tokens_per_sec is not None else "-"
        ms = f"{a.median_ms_per_token:.2f}" if a.median_ms_per_token is not None else "-"
        cats = ",".join(a.categories_tested)
        print(f"{a.model:<28} {ok_n:<7} {sr:<7} {total:<14} {tft:<12} {tps:<11} {ms:<11} {cats}")


def print_per_prompt_results(results: list[BenchmarkResult]) -> None:
    """Print a detailed per-prompt breakdown."""
    print("\n=== Per-Prompt Results ===")
    header = (
        f"{'Model':<28} {'Prompt ID':<24} {'Run':<4} {'OK':<5} {'Total(s)':<10} "
        f"{'TFT(s)':<10} {'Tok/s':<10} {'ms/tok':<10} {'Note'}"
    )
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda r: (r.model, r.prompt_id, r.run_index)):
        status = "OK" if r.success else "FAIL"
        total = f"{r.total_time_sec:.2f}" if r.total_time_sec else "-"
        tft = f"{r.time_to_first_token_sec:.2f}" if r.time_to_first_token_sec else "-"
        tps = f"{r.tokens_per_sec:.2f}" if r.tokens_per_sec is not None else "-"
        ms = f"{r.ms_per_token:.2f}" if r.ms_per_token is not None else "-"
        note = r.response_snippet[:40] if r.success else r.error[:40]
        print(
            f"{r.model:<28} {r.prompt_id:<24} {r.run_index:<4} {status:<5} "
            f"{total:<10} {tft:<10} {tps:<10} {ms:<10} {note}"
        )


def compute_summary(aggregates: list[ModelAggregate]) -> dict[str, Any]:
    total = len(aggregates)
    ok_count = sum(1 for a in aggregates if a.success_rate == 100.0)
    all_tps = [a.median_tokens_per_sec for a in aggregates if a.median_tokens_per_sec is not None]
    all_total = [a.median_total_time_sec for a in aggregates if a.median_total_time_sec is not None]
    all_tft = [a.median_time_to_first_token_sec for a in aggregates if a.median_time_to_first_token_sec is not None]
    return {
        "total_models": total,
        "fully_successful": ok_count,
        "success_rate_percent": round(sum(a.success_rate for a in aggregates) / total, 2) if total else 0.0,
        "median_tokens_per_sec": round(median(all_tps), 4) if all_tps else None,
        "median_total_time_sec": round(median(all_total), 4) if all_total else None,
        "median_time_to_first_token_sec": round(median(all_tft), 4) if all_tft else None,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\nSummary")
    print("-" * 40)
    print(f"Models tested:        {summary['total_models']}")
    print(f"Fully successful:     {summary['fully_successful']}")
    print(f"Overall success rate: {summary['success_rate_percent']:.2f}%")
    tps = f"{summary['median_tokens_per_sec']:.4f}" if summary["median_tokens_per_sec"] is not None else "-"
    total = f"{summary['median_total_time_sec']:.4f}" if summary["median_total_time_sec"] is not None else "-"
    tft = (
        f"{summary['median_time_to_first_token_sec']:.4f}"
        if summary["median_time_to_first_token_sec"] is not None
        else "-"
    )
    print(f"Median tokens/s:      {tps}")
    print(f"Median total(s):      {total}")
    print(f"Median TFT(s):        {tft}")


def load_previous_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        agg_list = data.get("aggregates", [])
        return {a["model"]: a for a in agg_list if "model" in a}
    except (json.JSONDecodeError, OSError, TypeError):
        return {}


def compare_results(
    aggregates: list[ModelAggregate],
    previous_by_model: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    comparisons: dict[str, dict[str, Any]] = {}
    current_models = {a.model for a in aggregates}
    previous_models = set(previous_by_model.keys())

    for agg in aggregates:
        model = agg.model
        if model not in previous_by_model:
            comparisons[model] = {"status": "new"}
            continue

        prev = previous_by_model[model]
        metrics = [
            ("median_total_time_sec", agg.median_total_time_sec, prev.get("median_total_time_sec"), -1.0),
            ("median_time_to_first_token_sec", agg.median_time_to_first_token_sec, prev.get("median_time_to_first_token_sec"), -1.0),
            ("median_tokens_per_sec", agg.median_tokens_per_sec, prev.get("median_tokens_per_sec"), 1.0),
        ]
        required = [v for _name, cur, prev_v, _direction in metrics for v in (cur, prev_v)]
        if any(v is None for v in required):
            comparisons[model] = {"status": "missing"}
            continue

        score = 0.0
        usable = 0
        deltas: dict[str, float | None] = {}
        for name, cur, prev_v, direction in metrics:
            delta = cur - prev_v
            deltas[f"{name}_delta"] = delta
            if prev_v and prev_v > 0:
                score += direction * (delta / prev_v)
                usable += 1

        if usable == 0:
            status = "missing"
        elif score > 1e-9:
            status = "improved"
        elif score < -1e-9:
            status = "regressed"
        else:
            status = "unchanged"

        comparisons[model] = {"status": status, **deltas}

    for prev_model in previous_models - current_models:
        comparisons[prev_model] = {"status": "missing"}

    return comparisons


def print_comparison(comparisons: dict[str, dict[str, Any]]) -> None:
    if not comparisons:
        return
    header = f"{'Model':<28} {'Status':<12} {'Total Δ':<12} {'TFT Δ':<12} {'Tok/s Δ':<12}"
    print("\nComparison to previous run")
    print(header)
    print("-" * len(header))

    def fmt(value: Any) -> str:
        return f"{value:+.4f}" if value is not None else "-"

    for model in sorted(comparisons):
        comp = comparisons[model]
        status = comp.get("status", "missing")
        print(
            f"{model:<28} {status:<12} "
            f"{fmt(comp.get('median_total_time_sec_delta')):<12} "
            f"{fmt(comp.get('median_time_to_first_token_sec_delta')):<12} "
            f"{fmt(comp.get('median_tokens_per_sec_delta')):<12}"
        )


def write_json_output(
    path: Path,
    prompts: list[dict[str, str]],
    results: list[BenchmarkResult],
    aggregates: list[ModelAggregate],
    summary: dict[str, Any],
    comparisons: dict[str, dict[str, Any]],
) -> None:
    output: dict[str, Any] = {
        "prompts": prompts,
        "summary": summary,
        "aggregates": [asdict(a) for a in aggregates],
        "prompt_summary": compute_prompt_summary(results),
        "category_summary": compute_category_summary(results),
        "results": [asdict(r) for r in results],
    }
    if comparisons:
        output["comparisons"] = comparisons
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")


def _round_floats(row: dict[str, Any]) -> dict[str, Any]:
    float_keys = (
        "total_time_sec",
        "time_to_first_token_sec",
        "generation_time_sec",
        "tokens_per_sec",
        "total_tokens_per_sec",
        "ms_per_token",
        "median_total_time_sec",
        "median_time_to_first_token_sec",
        "median_tokens_per_sec",
        "median_ms_per_token",
        "median_total_tokens_per_sec",
    )
    for key in float_keys:
        if row.get(key) is not None:
            row[key] = round(row[key], 6)
    return row


def write_csv_output(path: Path, aggregates: list[ModelAggregate], results: list[BenchmarkResult]) -> None:
    agg_fields = [
        "model",
        "total_runs",
        "successful_runs",
        "success_rate",
        "median_total_time_sec",
        "median_time_to_first_token_sec",
        "median_tokens_per_sec",
        "median_ms_per_token",
        "median_total_tokens_per_sec",
        "total_prompt_tokens",
        "total_completion_tokens",
    ]
    detail_fields = [
        "model",
        "prompt_id",
        "prompt_category",
        "run_index",
        "success",
        "total_time_sec",
        "time_to_first_token_sec",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "generation_time_sec",
        "tokens_per_sec",
        "ms_per_token",
        "error_category",
        "error",
    ]
    buf = io.StringIO()
    buf.write("# Aggregated results\n")
    w = csv.DictWriter(buf, fieldnames=agg_fields)
    w.writeheader()
    for a in aggregates:
        row = _round_floats(asdict(a))
        w.writerow({k: row.get(k, "") for k in agg_fields})

    buf.write("\n# Per-prompt results\n")
    w2 = csv.DictWriter(buf, fieldnames=detail_fields)
    w2.writeheader()
    for r in results:
        row = _round_floats(asdict(r))
        w2.writerow({k: row.get(k, "") for k in detail_fields})

    path.write_text(buf.getvalue(), encoding="utf-8")


def write_markdown_output(
    path: Path,
    aggregates: list[ModelAggregate],
    summary: dict[str, Any],
) -> None:
    headers = ["Model", "OK/N", "SR%", "Med Total(s)", "Med TFT(s)", "Med Tok/s", "Med ms/tok"]
    lines: list[str] = ["# Benchmark Results", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Overall success rate: {summary['success_rate_percent']:.2f}%")
    tps = f"{summary['median_tokens_per_sec']:.4f}" if summary["median_tokens_per_sec"] is not None else "-"
    total = f"{summary['median_total_time_sec']:.4f}" if summary["median_total_time_sec"] is not None else "-"
    tft = (
        f"{summary['median_time_to_first_token_sec']:.4f}"
        if summary["median_time_to_first_token_sec"] is not None
        else "-"
    )
    lines.append(f"- Median tokens/s: {tps}")
    lines.append(f"- Median total(s): {total}")
    lines.append(f"- Median TFT(s): {tft}")
    lines.append("")
    lines.append("## Aggregated Results (median across prompts)")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    def c(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v).replace("|", "\\|")

    for a in sorted(aggregates, key=lambda x: x.median_total_time_sec or float("inf")):
        ok_n = f"{a.successful_runs}/{a.total_runs}"
        lines.append(
            f"| {a.model} | {ok_n} | {c(a.success_rate)} | "
            f"{c(a.median_total_time_sec)} | {c(a.median_time_to_first_token_sec)} | "
            f"{c(a.median_tokens_per_sec)} | {c(a.median_ms_per_token)} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _html_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return html.escape(str(value))


def write_html_output(
    path: Path,
    aggregates: list[ModelAggregate],
    results: list[BenchmarkResult],
    summary: dict[str, Any],
    comparisons: dict[str, dict[str, Any]],
) -> None:
    """Write a styled HTML report with summary, aggregate, and per-prompt tables."""

    def row(cells: list[Any], header: bool = False) -> str:
        tag = "th" if header else "td"
        return "<tr>" + "".join(f"<{tag}>{_html_value(c)}</{tag}>" for c in cells) + "</tr>"

    sorted_agg = sorted(
        aggregates,
        key=lambda a: a.median_total_time_sec if a.median_total_time_sec is not None else float("inf"),
    )

    agg_rows = "\n".join(
        row([
            a.model,
            f"{a.successful_runs}/{a.total_runs}",
            f"{a.success_rate:.1f}",
            a.median_total_time_sec,
            a.median_time_to_first_token_sec,
            a.median_tokens_per_sec,
            a.median_total_tokens_per_sec,
            a.median_ms_per_token,
            ",".join(a.categories_tested),
        ])
        for a in sorted_agg
    )

    detail_rows = "\n".join(
        row([
            r.model,
            r.prompt_id,
            r.prompt_category,
            r.run_index,
            "OK" if r.success else "FAIL",
            r.total_time_sec,
            r.time_to_first_token_sec,
            r.tokens_per_sec,
            r.ms_per_token,
            r.prompt_tokens,
            r.completion_tokens,
            r.error_category or "",
        ])
        for r in sorted(results, key=lambda r: (r.model, r.prompt_id, r.run_index))
    )

    comparison_rows = ""
    if comparisons:
        comparison_rows = "\n".join(
            row([
                model,
                comp.get("status", "missing"),
                comp.get("median_total_time_sec_delta"),
                comp.get("median_time_to_first_token_sec_delta"),
                comp.get("median_tokens_per_sec_delta"),
            ])
            for model, comp in sorted(comparisons.items())
        )

    tps = _html_value(summary.get("median_tokens_per_sec"))
    total = _html_value(summary.get("median_total_time_sec"))
    tft = _html_value(summary.get("median_time_to_first_token_sec"))

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kimchi Benchmark Results</title>
<style>
:root {{
  --bg: #f7f8fa;
  --card: #ffffff;
  --text: #1f2328;
  --muted: #57606a;
  --border: #d0d7de;
  --accent: #0969da;
  --success: #1a7f37;
  --danger: #cf222e;
  --warning: #9a6700;
}}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
  margin: 0;
  padding: 2rem 1rem;
}}
.container {{
  max-width: 1200px;
  margin: 0 auto;
}}
h1 {{
  font-size: 1.75rem;
  margin-bottom: 0.25rem;
}}
.subtitle {{
  color: var(--muted);
  margin-bottom: 1.5rem;
}}
.summary {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}}
.card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  padding: 1rem;
  box-shadow: 0 1px 2px rgba(31,35,40,0.04);
}}
.card .label {{
  font-size: 0.875rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.025em;
}}
.card .value {{
  font-size: 1.5rem;
  font-weight: 600;
  margin-top: 0.25rem;
}}
.section {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  padding: 1.25rem;
  margin-bottom: 1.5rem;
  overflow-x: auto;
}}
.section h2 {{
  font-size: 1.25rem;
  margin-top: 0;
  margin-bottom: 1rem;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}}
th, td {{
  padding: 0.6rem 0.75rem;
  text-align: left;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}}
th {{
  background: var(--bg);
  font-weight: 600;
  position: sticky;
  top: 0;
}}
tr:hover td {{
  background: #f6f8fa;
}}
.status-ok {{ color: var(--success); font-weight: 600; }}
.status-fail {{ color: var(--danger); font-weight: 600; }}
.status-improved {{ color: var(--success); font-weight: 600; }}
.status-regressed {{ color: var(--danger); font-weight: 600; }}
.status-unchanged {{ color: var(--muted); }}
.status-new {{ color: var(--accent); font-weight: 600; }}
.status-missing {{ color: var(--warning); }}
.footer {{
  color: var(--muted);
  font-size: 0.875rem;
  text-align: center;
  margin-top: 2rem;
}}
</style>
</head>
<body>
<div class="container">
  <h1>Kimchi Benchmark Results</h1>
  <p class="subtitle">Generated on {html.escape(time.strftime("%Y-%m-%d %H:%M:%S %Z"))}</p>

  <div class="summary">
    <div class="card">
      <div class="label">Models tested</div>
      <div class="value">{summary.get('total_models', 0)}</div>
    </div>
    <div class="card">
      <div class="label">Fully successful</div>
      <div class="value">{summary.get('fully_successful', 0)}</div>
    </div>
    <div class="card">
      <div class="label">Overall success rate</div>
      <div class="value">{summary.get('success_rate_percent', 0.0):.2f}%</div>
    </div>
    <div class="card">
      <div class="label">Median tok/s</div>
      <div class="value">{tps}</div>
    </div>
    <div class="card">
      <div class="label">Median total (s)</div>
      <div class="value">{total}</div>
    </div>
    <div class="card">
      <div class="label">Median TFT (s)</div>
      <div class="value">{tft}</div>
    </div>
  </div>

  <div class="section">
    <h2>Aggregated Results (median across prompts)</h2>
    <table>
      <thead>
        {row(["Model", "OK/N", "SR%", "Med Total(s)", "Med TFT(s)", "Med Tok/s", "Med TotTok/s", "Med ms/tok", "Categories"], header=True)}
      </thead>
      <tbody>
        {agg_rows}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>Per-Prompt Results</h2>
    <table>
      <thead>
        {row(["Model", "Prompt", "Category", "Run", "Status", "Total(s)", "TFT(s)", "Tok/s", "ms/tok", "In", "Out", "Error"], header=True)}
      </thead>
      <tbody>
        {detail_rows}
      </tbody>
    </table>
  </div>
"""

    if comparison_rows:
        html_doc += f"""
  <div class="section">
    <h2>Comparison to Previous Run</h2>
    <table>
      <thead>
        {row(["Model", "Status", "Total Δ", "TFT Δ", "Tok/s Δ"], header=True)}
      </thead>
      <tbody>
        {comparison_rows}
      </tbody>
    </table>
  </div>
"""

    html_doc += """
  <div class="footer">
    Generated by Kimchi Model Benchmark
  </div>
</div>
</body>
</html>
"""

    path.write_text(html_doc, encoding="utf-8")


def write_output(
    results: list[BenchmarkResult],
    aggregates: list[ModelAggregate],
    prompts: list[dict[str, str]],
    summary: dict[str, Any],
    comparisons: dict[str, dict[str, Any]],
    output_format: str,
    output_path: Path,
) -> Path:
    suffix_map = {"json": ".json", "csv": ".csv", "markdown": ".md", "html": ".html"}
    expected_suffix = suffix_map.get(output_format, ".json")
    path = output_path if output_path.suffix.lower() == expected_suffix else output_path.with_suffix(expected_suffix)

    if output_format == "json":
        write_json_output(path, prompts, results, aggregates, summary, comparisons)
    elif output_format == "csv":
        write_csv_output(path, aggregates, results)
    elif output_format == "markdown":
        write_markdown_output(path, aggregates, summary)
    elif output_format == "html":
        write_html_output(path, aggregates, results, summary, comparisons)
    else:
        raise ValueError(f"Unsupported format: {output_format}")
    return path


def _load_config(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError, TypeError):
        return {}


def _resolve_default(
    env_name: str,
    config: dict[str, Any],
    config_key: str,
    hard_default: Any,
    type_func: Any = str,
) -> Any:
    env_value = os.environ.get(env_name)
    if env_value:
        try:
            return type_func(env_value)
        except (ValueError, TypeError):
            pass
    config_value = config.get(config_key)
    if config_value is not None:
        try:
            return type_func(config_value)
        except (ValueError, TypeError):
            pass
    return hard_default


def _split_models(value: str) -> list[str]:
    return [m.strip() for m in value.split(",") if m.strip()]


def _infer_format(output_path: Path) -> str:
    suffix = output_path.suffix.lower()
    return {
        ".json": "json",
        ".csv": "csv",
        ".md": "markdown",
        ".markdown": "markdown",
        ".html": "html",
        ".htm": "html",
    }.get(suffix, "json")


def _load_prompts_file(path: Path) -> list[dict[str, str]]:
    """Load prompts from a JSON file.

    Accepts either:
      - a list of strings  →  auto-assigns id/category
      - a list of objects with 'id', 'category', and either 'text' or 'prompt'
        (prefer 'text' if both are present)
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Prompts file must be a JSON array, got {type(data).__name__}")
    result: list[dict[str, str]] = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            result.append({"id": f"custom_{i}", "category": "custom", "text": item})
        elif isinstance(item, dict):
            text = item.get("text") or item.get("prompt")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Prompt entry {i} must have a non-empty 'text' or 'prompt' key")
            result.append({
                "id": str(item.get("id", f"custom_{i}")),
                "category": str(item.get("category", "custom")),
                "text": text,
            })
        else:
            raise ValueError(f"Prompt entry {i} must be a string or object")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:] if __name__ == "__main__" else []

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)
    config = _load_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Benchmark Kimchi inference models across multiple prompts",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--models",
        "-m",
        type=_split_models,
        default=_resolve_default("BENCHMARK_MODELS", config, "models", DEFAULT_MODELS, lambda s: _split_models(str(s))),
        help="Comma-separated list of models to benchmark",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default=_resolve_default("BENCHMARK_PROMPT", config, "prompt", DEFAULT_PROMPT),
        help="Prompt used when --single-prompt is specified",
    )
    parser.add_argument(
        "--single-prompt",
        action="store_true",
        help="Use only --prompt instead of the built-in prompt suite",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Path to a JSON file containing prompts (list of strings or objects with id/category/text)",
    )
    parser.add_argument(
        "--prompts-count",
        "-n",
        type=int,
        default=_resolve_default("BENCHMARK_PROMPTS_COUNT", config, "prompts_count", None, int),
        help="Number of prompts to sample from the built-in set (default: all)",
    )
    parser.add_argument(
        "--prompts-seed",
        type=int,
        default=None,
        help="Random seed for --prompts-count sampling (for reproducibility)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=_resolve_default("BENCHMARK_RUNS", config, "runs", DEFAULT_RUNS, int),
        help="Number of repeated runs per (model, prompt) pair",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=_resolve_default("BENCHMARK_MAX_TOKENS", config, "max_tokens", DEFAULT_MAX_TOKENS, int),
        help="Maximum completion tokens per request",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=_resolve_default("BENCHMARK_TEMPERATURE", config, "temperature", DEFAULT_TEMPERATURE, float),
        help="Sampling temperature",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=_resolve_default("BENCHMARK_MAX_WORKERS", config, "workers", MAX_WORKERS, int),
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=_resolve_default("BENCHMARK_OUTPUT", config, "output", "benchmark_results", Path),
        help="Output base path",
    )
    parser.add_argument(
        "--format",
        choices=OUTPUT_FORMATS,
        default=_resolve_default("BENCHMARK_FORMAT", config, "format", None),
        help="Output format",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit without making API calls")
    parser.add_argument("--no-per-prompt-table", action="store_true", help="Skip the verbose per-prompt results table")

    args = parser.parse_args(argv)

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    if args.format is None:
        args.format = _infer_format(args.output)
    return args


def _resolve_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    """Return the effective prompt list based on CLI args."""
    if args.prompts_file:
        prompts = _load_prompts_file(args.prompts_file)
    elif args.single_prompt:
        prompts = [{"id": "single_prompt", "category": "custom", "text": args.prompt}]
    else:
        prompts = list(DEFAULT_PROMPTS)

    if args.prompts_count and args.prompts_count < len(prompts):
        rng = random.Random(args.prompts_seed)
        prompts = rng.sample(prompts, args.prompts_count)

    return prompts


def _print_progress(completed: int, total: int, result: BenchmarkResult) -> None:
    status = "OK" if result.success else "FAIL"
    prefix = (
        f"  [{completed}/{total}] {result.model} [{result.prompt_id}] run={result.run_index} {status}"
    )
    if result.success and result.total_time_sec is not None and result.tokens_per_sec is not None:
        print(f"{prefix} {result.total_time_sec:.1f}s {result.tokens_per_sec:.1f} tok/s")
    elif result.success and result.total_time_sec is not None:
        print(f"{prefix} {result.total_time_sec:.1f}s")
    else:
        print(f"{prefix} {result.error_category or 'ERR'}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompts = _resolve_prompts(args)

    suffix_map = {"json": ".json", "csv": ".csv", "markdown": ".md", "html": ".html"}
    expected_suffix = suffix_map[args.format]
    effective_output_path = (
        args.output if args.output.suffix.lower() == expected_suffix else args.output.with_suffix(expected_suffix)
    )

    if args.dry_run:
        print("Dry run — no API calls will be made")
        print(f"Models ({len(args.models)}): {', '.join(args.models)}")
        print(f"Prompts ({len(prompts)}):")
        for p in prompts:
            print(f"  [{p['id']}] ({p['category']}) {p['text'][:80]}...")
        print(f"Runs: {args.runs}")
        print(f"Total jobs: {len(args.models) * len(prompts) * args.runs}")
        print(f"Max tokens: {args.max_tokens} | Temperature: {args.temperature} | Workers: {args.workers}")
        print(f"Output: {effective_output_path} ({args.format})")
        return 0

    api_key = get_api_key()

    previous_results_path = args.output.with_suffix(".json")
    previous_results = load_previous_results(previous_results_path)
    if previous_results:
        print(f"Loaded previous results for {len(previous_results)} model(s) from {previous_results_path}")

    print("Fetching available models from Kimchi API...")
    try:
        available_models = fetch_available_models(api_key)
        print(f"Found {len(available_models)} models from API.")
    except requests.RequestException as exc:
        print(f"Warning: could not fetch live model list: {exc}", file=sys.stderr)
        available_models = args.models.copy()

    models_to_test = [m for m in available_models if m in args.models] or args.models
    total_tasks = len(models_to_test) * len(prompts) * args.runs
    print(
        f"\nBenchmarking {len(models_to_test)} models × {len(prompts)} prompts × {args.runs} runs "
        f"= {total_tasks} requests | {args.workers} workers\n"
    )

    # Build all (model, prompt, run_index) tasks
    tasks = [
        (model, prompt, run_index)
        for model in models_to_test
        for prompt in prompts
        for run_index in range(args.runs)
    ]

    all_results: list[BenchmarkResult] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                benchmark_model,
                model,
                api_key,
                prompt,
                run_index,
                args.max_tokens,
                args.temperature,
                args.verbose,
            ): (model, prompt, run_index)
            for model, prompt, run_index in tasks
        }
        for future in as_completed(future_map):
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                model, prompt, run_index = future_map[future]
                result = BenchmarkResult(
                    model=model,
                    prompt_id=prompt["id"],
                    prompt_category=prompt["category"],
                    run_index=run_index,
                    success=False,
                    error_category=ErrorCategory.UNKNOWN,
                    error=f"Future error: {type(exc).__name__}: {exc}",
                )
            completed += 1
            _print_progress(completed, total_tasks, result)
            all_results.append(result)

    # Aggregate per model
    aggregates: list[ModelAggregate] = []
    for model in models_to_test:
        model_results = [r for r in all_results if r.model == model]
        aggregates.append(aggregate_model_results(model, model_results))

    print_aggregates(aggregates)
    if not args.no_per_prompt_table:
        print_per_prompt_results(all_results)

    summary = compute_summary(aggregates)
    print_summary(summary)

    comparisons = compare_results(aggregates, previous_results)
    if comparisons:
        print_comparison(comparisons)

    output_file = write_output(
        all_results,
        aggregates,
        prompts,
        summary,
        comparisons,
        args.format,
        effective_output_path,
    )
    print(f"\nDetailed results saved to {output_file.resolve()} ({args.format})")
    return 0 if any(a.successful_runs > 0 for a in aggregates) else 1


if __name__ == "__main__":
    sys.exit(main())
