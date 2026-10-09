from __future__ import annotations

import csv
import json

from batch_bench import plot


def test_create_plots_reads_each_strategy_result(tmp_path, monkeypatch) -> None:
    for index, strategy in enumerate(plot.STRATEGIES, 1):
        result_dir = tmp_path / strategy
        result_dir.mkdir()
        (result_dir / "summary.json").write_text(
            json.dumps(
                {
                    "output_token_throughput_per_s": index,
                    "p50_latency_ms": index,
                    "p95_latency_ms": index,
                    "p50_ttft_ms": index,
                    "p95_ttft_ms": index,
                }
            ),
            encoding="utf-8",
        )
        with (result_dir / "steps.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["batch_size"])
            writer.writeheader()
            writer.writerow({"batch_size": index})

    outputs = []
    monkeypatch.setattr(plot, "_bar", lambda path, *args: outputs.append(path.name))

    plot.create_plots(tmp_path)

    assert outputs == ["throughput.png", "latency.png", "ttft.png", "batch_size.png"]


def test_create_plots_uses_median_and_run_range(tmp_path, monkeypatch) -> None:
    for run, value in enumerate((1, 100, 3), 1):
        for strategy in plot.STRATEGIES:
            result_dir = tmp_path / f"run_{run}" / strategy
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "output_token_throughput_per_s": value,
                        "p50_latency_ms": value,
                        "p95_latency_ms": value,
                        "p50_ttft_ms": value,
                        "p95_ttft_ms": value,
                    }
                ),
                encoding="utf-8",
            )
            with (result_dir / "steps.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["batch_size"])
                writer.writeheader()
                writer.writerow({"batch_size": value})

    calls = []
    monkeypatch.setattr(plot, "_bar", lambda *args: calls.append(args))

    plot.create_plots(tmp_path, repetitions=3)

    assert calls[0][3] == [[3, 3, 3]]
    assert calls[0][5] == [[[2, 2, 2], [97, 97, 97]]]
