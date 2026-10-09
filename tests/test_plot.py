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
