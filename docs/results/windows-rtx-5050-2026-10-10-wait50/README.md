# RTX 5050 benchmark evidence (50 ms dynamic wait)

This folder contains the measurements behind the eight dense and sparse figures in `docs/figures/` and `docs/EXPERIMENTAL_REPORT.md`. The complete run used three repetitions of each strategy on each workload (18 cases). The exploratory wait sweep is separate and is not included in the plotted medians.

## Files

- `results/{dense,sparse}/run_{1,2,3}/{fixed,dynamic,continuous}/requests.csv`: one row per request, including arrival, start, first-token, finish, and derived times.
- `results/{dense,sparse}/run_{1,2,3}/{fixed,dynamic,continuous}/steps.csv`: active batch size for each generated-token step.
- `results/{dense,sparse}/run_{1,2,3}/{fixed,dynamic,continuous}/summary.json`: per-run percentile and throughput metrics.
- `results/{dense,sparse}/*.png`: plots from those summaries and step files. Identical copies appear in `docs/figures/` for the report.
- `metadata.json`, `nvidia-smi.txt`, and `run.log`: configuration, device details, and execution log.
- `tuning-summary.csv`, `tuning-run.log`, and `tuning/`: exploratory single-run wait sweep.

Each run completed 30 distinct requests and 960 output tokens. Latency is `finished_ms - arrival_ms`; time to first token (TTFT) is `first_token_ms - arrival_ms`. Both include queueing. Output-token throughput is 960 tokens divided by the makespan from first scheduled arrival to last completion. A figure bar is the median of the three per-run metrics; its whisker spans the minimum and maximum. Average active batch size is the mean step batch size within each run, then the median of those three means.

The sparse continuous TTFT bars are **34.53 ms p50** and **45.51 ms p95**, not zero. They appear close to the axis because fixed and dynamic TTFT exceed 900 ms on the same linear scale. Sparse continuous total latency is **736.47 ms p50** and **1151.14 ms p95** because generation continues after the first token. Numeric bar labels make the small nonzero TTFT visible.

## Regenerate plots and report

From the repository root, install the project dependencies as documented in the root README, then run:

```python
from pathlib import Path
from batch_bench.plot import create_plots

evidence = Path("docs/results/windows-rtx-5050-2026-10-10-wait50/results")
for workload in ("dense", "sparse"):
    create_plots(evidence / workload, repetitions=3)
```

Copy the eight regenerated PNG files to `docs/figures/` using the `{workload}_{metric}.png` names used by the Markdown report. `docs/EXPERIMENTAL_REPORT.md` contains the numerical tables and interpretations; `docs/EXPERIMENTAL_REPORT.pdf` is the rendered copy.

## Verification

The audit checked all 18 `requests.csv` and `steps.csv` files against the per-run summaries: request counts and IDs, output-token counts, timestamp order, timing formulas, p50/p95 values, makespan, throughput, and batch-size bounds. All eight plot series and error ranges were checked against the same saved inputs. These are measurements for this specific GPU, model, workloads, and run date; the wait sweep selected 50 ms on these workloads, not on a separate validation set.
