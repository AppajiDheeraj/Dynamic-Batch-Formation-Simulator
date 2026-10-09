from __future__ import annotations

import pytest

from batch_bench.run import load_workload


@pytest.mark.parametrize(
    "content",
    [
        "not json\n",
        "[]\n",
        '{"id":"a","prompt":"x","arrival_ms":-1,"max_new_tokens":1}\n',
        '{"id":"a","prompt":"x","arrival_ms":0,"max_new_tokens":0}\n',
        '{"id":"","prompt":"x","arrival_ms":0,"max_new_tokens":1}\n',
        '{"id":"a","prompt":"","arrival_ms":0,"max_new_tokens":1}\n',
    ],
)
def test_malformed_workload_rows_are_rejected(tmp_path, content: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        load_workload(path)


def test_duplicate_request_ids_are_rejected(tmp_path) -> None:
    path = tmp_path / "duplicate.jsonl"
    path.write_text(
        '{"id":"a","prompt":"one","arrival_ms":0,"max_new_tokens":1}\n'
        '{"id":"a","prompt":"two","arrival_ms":1,"max_new_tokens":1}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_workload(path)
