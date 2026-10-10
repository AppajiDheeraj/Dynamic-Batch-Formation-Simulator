# Archived RTX 5050 benchmark evidence

This folder preserves the earlier run that used a 25 ms dynamic wait. It is not the run reported in `docs/EXPERIMENTAL_REPORT.md`. The reported 50 ms run and its figures are in [`../windows-rtx-5050-2026-10-10-wait50/`](../windows-rtx-5050-2026-10-10-wait50/).

Each value below is the median of three runs, calculated from this folder's `summary.json` files. Peak memory converts bytes to MiB with 1 MiB equal to 1,048,576 bytes.

## Dense workload

| Strategy | Tokens/s | Latency p50 (ms) | Latency p95 (ms) | TTFT p50 (ms) | TTFT p95 (ms) | Peak CUDA memory (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| Fixed | 118.32 | 3686.74 | 6932.36 | 2890.82 | 6155.18 | 993.29 |
| Dynamic | 120.09 | 4040.71 | 7265.09 | 3521.04 | 6383.40 | 993.29 |
| Continuous | 165.11 | 2815.03 | 4941.50 | 2014.44 | 4092.68 | 1023.61 |

## Sparse workload

| Strategy | Tokens/s | Latency p50 (ms) | Latency p95 (ms) | TTFT p50 (ms) | TTFT p95 (ms) | Peak CUDA memory (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| Fixed | 70.37 | 1686.95 | 2484.21 | 765.96 | 1245.59 | 993.29 |
| Dynamic | 69.19 | 1686.12 | 2752.52 | 687.55 | 1381.99 | 993.29 |
| Continuous | 73.80 | 1060.20 | 1680.94 | 46.25 | 73.07 | 1023.61 |

The raw request, step, summary, plot, device, and run-log files remain unchanged in this archive.
