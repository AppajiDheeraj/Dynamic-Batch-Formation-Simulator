FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY workloads ./workloads

RUN pip install --no-cache-dir ".[test]"

CMD ["python", "-m", "batch_bench.run", "--strategy", "all"]
