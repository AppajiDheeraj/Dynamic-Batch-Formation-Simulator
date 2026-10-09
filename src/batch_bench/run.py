from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any

from .schedulers import RequestSpec, SystemClock, make_policy, run_scheduler


STRATEGIES = ("fixed", "dynamic", "continuous")


def _integer(value: Any, name: str, line_number: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"line {line_number}: {name} must be an integer of at least {minimum}")
    return value


def load_workload(path: Path, max_requests: int | None = None) -> list[RequestSpec]:
    if max_requests is not None and max_requests < 1:
        raise ValueError("max requests must be at least 1")
    requests: list[RequestSpec] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: each row must be a JSON object")
            request_id = row.get("id")
            prompt = row.get("prompt")
            if not isinstance(request_id, str) or not request_id.strip():
                raise ValueError(f"line {line_number}: id must be a non-empty string")
            if request_id in seen:
                raise ValueError(f"line {line_number}: duplicate request id {request_id}")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"line {line_number}: prompt must be a non-empty string")
            arrival_ms = _integer(row.get("arrival_ms"), "arrival_ms", line_number, 0)
            max_new_tokens = _integer(row.get("max_new_tokens"), "max_new_tokens", line_number, 1)
            seen.add(request_id)
            requests.append(RequestSpec(request_id, prompt, arrival_ms, max_new_tokens))
            if max_requests is not None and len(requests) == max_requests:
                break
    if not requests:
        raise ValueError("workload must contain at least one request")
    return sorted(requests, key=lambda request: request.arrival_ms)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def run_one(args: argparse.Namespace) -> None:
    from .model import QwenEngine

    requests = load_workload(args.workload, args.max_requests)
    engine = QwenEngine(args.model)
    engine.warm_up()
    clock = SystemClock()
    policy = make_policy(args.strategy, args.batch_size, args.dynamic_wait_ms)
    completed, steps = run_scheduler(requests, policy, engine, clock)

    result_dir = args.output_dir / args.strategy
    result_dir.mkdir(parents=True, exist_ok=True)
    request_rows = [asdict(request) for request in completed]
    step_rows = [{"strategy": args.strategy, **asdict(step)} for step in steps]
    _atomic_csv(
        result_dir / "requests.csv",
        request_rows,
        [
            "id",
            "arrival_ms",
            "started_ms",
            "first_token_ms",
            "finished_ms",
            "queue_ms",
            "ttft_ms",
            "latency_ms",
            "prompt_tokens",
            "output_tokens",
            "output_text",
        ],
    )
    _atomic_csv(
        result_dir / "steps.csv",
        step_rows,
        ["strategy", "step", "start_ms", "duration_ms", "batch_size"],
    )

    first_arrival = min(request.arrival_ms for request in completed)
    makespan_ms = max(request.finished_ms for request in completed) - first_arrival
    total_output_tokens = sum(request.output_tokens for request in completed)
    latencies = [request.latency_ms for request in completed]
    ttfts = [request.ttft_ms for request in completed]
    summary = {
        "strategy": args.strategy,
        "model": args.model,
        "request_count": len(completed),
        "batch_size": args.batch_size,
        "dynamic_wait_ms": args.dynamic_wait_ms,
        "makespan_ms": makespan_ms,
        "output_tokens": total_output_tokens,
        "output_token_throughput_per_s": total_output_tokens / (makespan_ms / 1000),
        "request_throughput_per_s": len(completed) / (makespan_ms / 1000),
        "p50_latency_ms": median(latencies),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "p50_ttft_ms": median(ttfts),
        "p95_ttft_ms": _percentile(ttfts, 0.95),
        "peak_cuda_memory_bytes": engine.peak_memory_bytes(),
    }
    _atomic_json(result_dir / "summary.json", summary)
    print(f"{args.strategy}: wrote {result_dir}")


def _run_all(args: argparse.Namespace) -> None:
    expected_ids = {request.id for request in load_workload(args.workload, args.max_requests)}
    for strategy in STRATEGIES:
        command = [
            sys.executable,
            "-m",
            "batch_bench.run",
            "--strategy",
            strategy,
            "--workload",
            str(args.workload),
            "--model",
            args.model,
            "--batch-size",
            str(args.batch_size),
            "--dynamic-wait-ms",
            str(args.dynamic_wait_ms),
            "--output-dir",
            str(args.output_dir),
        ]
        if args.max_requests is not None:
            command.extend(["--max-requests", str(args.max_requests)])
        subprocess.run(command, check=True)
        with (args.output_dir / strategy / "requests.csv").open(newline="", encoding="utf-8") as handle:
            result_ids = [row["id"] for row in csv.DictReader(handle)]
        if len(result_ids) != len(set(result_ids)) or set(result_ids) != expected_ids:
            raise RuntimeError(f"{strategy} results do not match the workload request IDs")
    from .plot import create_plots

    create_plots(args.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real LLM batching benchmarks.")
    parser.add_argument("--strategy", choices=(*STRATEGIES, "all"), default="all")
    parser.add_argument("--workload", type=Path, default=Path("workloads/prompts.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dynamic-wait-ms", type=float, default=25)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-requests", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.strategy == "all":
            _run_all(args)
        else:
            run_one(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
