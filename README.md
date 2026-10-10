# Dynamic Batch Formation Simulator

[![CI/CD](https://img.shields.io/github/actions/workflow/status/AppajiDheeraj/Dynamic-Batch-Formation-Simulator/docker.yml?branch=main&style=for-the-badge&logo=githubactions&logoColor=white&label=CI%20%2F%20CD)](https://github.com/AppajiDheeraj/Dynamic-Batch-Formation-Simulator/actions/workflows/docker.yml)
[![Docker pulls](https://img.shields.io/docker/pulls/appajidheeraj/dynamic-batch-formation-simulator?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/appajidheeraj/dynamic-batch-formation-simulator)
[![GitHub release](https://img.shields.io/github/v/release/AppajiDheeraj/Dynamic-Batch-Formation-Simulator?style=for-the-badge&logo=github)](https://github.com/AppajiDheeraj/Dynamic-Batch-Formation-Simulator/releases/latest)
[![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

This project compares fixed, dynamic, and continuous batching with real LLM inference. Each run loads the same Qwen model and uses the same saved requests. Only the scheduling rule changes.

## What it uses

- Python 3.11 in the Docker image
- PyTorch and Hugging Face Transformers
- `Qwen/Qwen2.5-0.5B-Instruct` in FP16
- Docker Compose and an NVIDIA GPU
- Matplotlib for graphs

There is no HTTP server. There is no `curl`, Ollama, LM Studio, or vLLM process. A Python runner sends the saved workload straight to the shared inference engine.

## Windows setup

Install these items on the RTX laptop:

1. The current NVIDIA driver.
2. WSL 2.
3. Docker Desktop with the WSL 2 engine.
4. Git.

Open PowerShell and check the GPU first:

```powershell
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

Clone the repository and build the image:

```powershell
git clone https://github.com/AppajiDheeraj/Dynamic-Batch-Formation-Simulator.git
cd Dynamic-Batch-Formation-Simulator
docker compose build
```

Or pull the prebuilt image from Docker Hub:

```powershell
docker pull appajidheeraj/dynamic-batch-formation-simulator:latest
```

Run the CPU-only scheduler tests:

```powershell
docker compose run --rm benchmark pytest -q
```

Run a small GPU smoke test. This downloads the model on the first run:

```powershell
docker compose run --rm benchmark python -m batch_bench.run --strategy fixed --max-requests 2 --batch-size 2
```

Run the full comparison:

```powershell
docker compose run --rm benchmark
```

The command benchmarks two 30-request workloads. The dense workload sends a request every 20 ms. The sparse workload sends a request every 400 ms. Each uses a repeating mix of 16, 32, and 48 output-token limits.

Each strategy runs three times in a separate process. The strategy order rotates between repetitions, but runs remain sequential. Graphs show the median and min-to-max error bars. A failed run stops the command.

## Output

Each strategy writes these files under `results/<workload>/run_<number>/<strategy>/`:

- `requests.csv` has request timing, token counts, and generated text.
- `steps.csv` has token-step time and active batch size.
- `summary.json` has throughput, p50 and p95 timing, makespan, and peak CUDA memory.

The full run creates four graphs under both `results/dense/` and `results/sparse/`:

- `throughput.png`: output-token throughput
- `latency.png`: p50 and p95 request latency
- `ttft.png`: p50 and p95 time to first token
- `batch_size.png`: average active batch size

## Scheduling rules

Fixed batching waits for a full batch. It runs that closed batch until every request ends. The last saved batch may be smaller.

Dynamic batching waits until the batch fills or the oldest waiting request reaches the wait limit. It then runs a closed batch until every request ends.

Continuous batching checks for free slots before every token step. It admits waiting requests without waiting for the other active requests to end.

ORCA provides the iteration-level scheduling idea used by the continuous policy. vLLM is a reference implementation of production continuous batching. This project does not copy either codebase and does not claim to match vLLM. It keeps one simple inference path so the three student-written schedulers can be compared.

## Limits

The engine recomputes each active sequence for every token and does not use a KV cache. This is slower than a production engine, but all three strategies use the same path. Prompts are truncated to 128 tokens. The workload controls each request's output limit.

Test the chosen batch size on the target 3 GB to 4 GB RTX GPU. Lower `--batch-size` if CUDA runs out of memory.

## Useful options

```text
--strategy fixed|dynamic|continuous|all
--workload workloads/prompts.jsonl
--model Qwen/Qwen2.5-0.5B-Instruct
--batch-size 4
--dynamic-wait-ms 25
--max-requests 2
--output-dir results
--repetitions 3
```

`--repetitions` is available with `--strategy all`. The default is one run for quick checks.

## GitHub workflow

The `Docker` workflow tests and builds every pull request. Pushes to `main` publish `latest` and commit-SHA tags to Docker Hub; version tags such as `v0.1.0` also publish the matching semantic-version tag.

Repository maintainers must configure a Docker Hub access token as the `DOCKERHUB_TOKEN` GitHub Actions secret. The Docker Hub username is stored as the `DOCKERHUB_USERNAME` repository variable.

The measured graphs used in the report are stored under `docs/figures/`. Raw CSV and JSON results stay under the ignored `results/` directory.

## References

- [ORCA: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [vLLM](https://github.com/vllm-project/vllm)
- [Docker](https://www.docker.com/)
