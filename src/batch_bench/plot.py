from __future__ import annotations

import csv
import json
from pathlib import Path


STRATEGIES = ("fixed", "dynamic", "continuous")


def _bar(path: Path, title: str, ylabel: str, values: list[list[float]], labels: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    x = range(len(STRATEGIES))
    width = 0.8 / len(values)
    for offset, (series, label) in enumerate(zip(values, labels)):
        positions = [position - 0.4 + width / 2 + offset * width for position in x]
        axis.bar(positions, series, width, label=label)
    axis.set_xticks(list(x), STRATEGIES)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    if len(values) > 1:
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def create_plots(output_dir: Path) -> None:
    summaries = []
    average_batches = []
    for strategy in STRATEGIES:
        summary_path = output_dir / strategy / "summary.json"
        steps_path = output_dir / strategy / "steps.csv"
        if not summary_path.exists() or not steps_path.exists():
            raise ValueError(f"missing results for {strategy}")
        with summary_path.open(encoding="utf-8") as handle:
            summaries.append(json.load(handle))
        with steps_path.open(newline="", encoding="utf-8") as handle:
            batches = [float(row["batch_size"]) for row in csv.DictReader(handle)]
        if not batches:
            raise ValueError(f"no token steps found for {strategy}")
        average_batches.append(sum(batches) / len(batches))

    _bar(
        output_dir / "throughput.png",
        "Output-token throughput",
        "tokens per second",
        [[summary["output_token_throughput_per_s"] for summary in summaries]],
        ["output tokens"],
    )
    _bar(
        output_dir / "latency.png",
        "Request latency",
        "milliseconds",
        [
            [summary["p50_latency_ms"] for summary in summaries],
            [summary["p95_latency_ms"] for summary in summaries],
        ],
        ["p50", "p95"],
    )
    _bar(
        output_dir / "ttft.png",
        "Time to first token",
        "milliseconds",
        [
            [summary["p50_ttft_ms"] for summary in summaries],
            [summary["p95_ttft_ms"] for summary in summaries],
        ],
        ["p50", "p95"],
    )
    _bar(
        output_dir / "batch_size.png",
        "Average active batch size",
        "requests",
        [average_batches],
        ["average"],
    )
    print(f"wrote graphs to {output_dir}")
