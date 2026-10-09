from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    for workload in ("dense", "sparse"):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "batch_bench.run",
                "--strategy",
                "all",
                "--workload",
                f"workloads/{workload}.jsonl",
                "--repetitions",
                "3",
                "--output-dir",
                str(Path("results") / workload),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
