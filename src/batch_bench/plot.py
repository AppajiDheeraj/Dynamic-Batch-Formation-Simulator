from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median


STRATEGIES = ("fixed", "dynamic", "continuous")


def _bar(
    path: Path,
    title: str,
    ylabel: str,
    values: list[list[float]],
    labels: list[str],
    errors: list[list[list[float]]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    x = range(len(STRATEGIES))
    width = 0.8 / len(values)
    for offset, (series, label) in enumerate(zip(values, labels)):
        positions = [position - 0.4 + width / 2 + offset * width for position in x]
        yerr = None if errors is None else errors[offset]
        axis.bar(positions, series, width, label=label, yerr=yerr, capsize=4)
    axis.set_xticks(list(x), STRATEGIES)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    if len(values) > 1:
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _median_range(values: list[float]) -> tuple[float, list[float]]:
    middle = median(values)
    return middle, [middle - min(values), max(values) - middle]


def create_plots(output_dir: Path, repetitions: int = 1) -> None:
    run_dirs = [output_dir] if repetitions == 1 else [output_dir / f"run_{index}" for index in range(1, repetitions + 1)]
    summaries: dict[str, list[dict]] = {strategy: [] for strategy in STRATEGIES}
    batches: dict[str, list[float]] = {strategy: [] for strategy in STRATEGIES}
    for strategy in STRATEGIES:
        for run_dir in run_dirs:
            summary_path = run_dir / strategy / "summary.json"
            steps_path = run_dir / strategy / "steps.csv"
            if not summary_path.exists() or not steps_path.exists():
                raise ValueError(f"missing results for {strategy}")
            with summary_path.open(encoding="utf-8") as handle:
                summaries[strategy].append(json.load(handle))
            with steps_path.open(newline="", encoding="utf-8") as handle:
                sizes = [float(row["batch_size"]) for row in csv.DictReader(handle)]
            if not sizes:
                raise ValueError(f"no token steps found for {strategy}")
            batches[strategy].append(sum(sizes) / len(sizes))

    def series(key: str) -> tuple[list[float], list[list[float]]]:
        stats = [_median_range([summary[key] for summary in summaries[strategy]]) for strategy in STRATEGIES]
        return [value for value, _ in stats], [[error[0] for _, error in stats], [error[1] for _, error in stats]]

    throughput, throughput_error = series("output_token_throughput_per_s")
    p50_latency, p50_latency_error = series("p50_latency_ms")
    p95_latency, p95_latency_error = series("p95_latency_ms")
    p50_ttft, p50_ttft_error = series("p50_ttft_ms")
    p95_ttft, p95_ttft_error = series("p95_ttft_ms")
    batch_stats = [_median_range(batches[strategy]) for strategy in STRATEGIES]
    average_batches = [value for value, _ in batch_stats]
    batch_error = [[error[0] for _, error in batch_stats], [error[1] for _, error in batch_stats]]
    suffix = "" if repetitions == 1 else f" (median of {repetitions} runs)"

    _bar(
        output_dir / "throughput.png",
        "Output-token throughput" + suffix,
        "tokens per second",
        [throughput],
        ["output tokens"],
        [throughput_error],
    )
    _bar(
        output_dir / "latency.png",
        "Request latency" + suffix,
        "milliseconds",
        [
            p50_latency,
            p95_latency,
        ],
        ["p50", "p95"],
        [p50_latency_error, p95_latency_error],
    )
    _bar(
        output_dir / "ttft.png",
        "Time to first token" + suffix,
        "milliseconds",
        [
            p50_ttft,
            p95_ttft,
        ],
        ["p50", "p95"],
        [p50_ttft_error, p95_ttft_error],
    )
    _bar(
        output_dir / "batch_size.png",
        "Average active batch size" + suffix,
        "requests",
        [average_batches],
        ["average"],
        [batch_error],
    )
    print(f"wrote graphs to {output_dir}")
