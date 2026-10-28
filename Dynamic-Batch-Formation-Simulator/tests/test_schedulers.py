from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from batch_bench.schedulers import ActiveRequest, RequestSpec, make_policy, run_scheduler


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def now_ms(self) -> float:
        return self.value

    def sleep_ms(self, duration_ms: float) -> None:
        self.value += duration_ms

    def advance(self, duration_ms: float) -> None:
        self.value += duration_ms


class FakeEngine:
    def __init__(self, clock: FakeClock, lengths: dict[str, int]) -> None:
        self.clock = clock
        self.lengths = lengths

    def activate(self, spec: RequestSpec, started_ms: float) -> ActiveRequest:
        return ActiveRequest(spec, [1], [], 1, started_ms)

    def step(self, active: Sequence[ActiveRequest]) -> None:
        self.clock.advance(1)
        for request in active:
            request.generated_ids.append(2)
            request.eos_reached = len(request.generated_ids) >= self.lengths[request.spec.id]

    def decode(self, request: ActiveRequest) -> str:
        return "x" * len(request.generated_ids)


def request(request_id: str, arrival_ms: int = 0, tokens: int = 10) -> RequestSpec:
    return RequestSpec(request_id, request_id, arrival_ms, tokens)


def run(strategy: str, requests: list[RequestSpec], lengths: dict[str, int], batch_size: int = 2, wait: int = 10):
    clock = FakeClock()
    completed, steps = run_scheduler(
        requests,
        make_policy(strategy, batch_size, wait),
        FakeEngine(clock, lengths),
        clock,
    )
    return {item.id: item for item in completed}, steps


def test_fixed_keeps_a_closed_batch_and_runs_final_partial_batch() -> None:
    completed, steps = run(
        "fixed",
        [request("a"), request("b"), request("c")],
        {"a": 1, "b": 3, "c": 1},
    )

    assert completed["a"].started_ms == 0
    assert completed["b"].started_ms == 0
    assert completed["c"].started_ms == completed["b"].finished_ms == 3
    assert [step.batch_size for step in steps] == [2, 1, 1, 1]


def test_dynamic_starts_when_capacity_is_reached() -> None:
    completed, _ = run(
        "dynamic",
        [request("a", 0), request("b", 3)],
        {"a": 1, "b": 1},
    )

    assert completed["a"].started_ms == 3
    assert completed["b"].started_ms == 3


def test_dynamic_waits_for_timeout_even_when_no_future_request_exists() -> None:
    completed, _ = run("dynamic", [request("a")], {"a": 1})

    assert completed["a"].started_ms == 10


def test_continuous_refills_before_long_request_finishes() -> None:
    completed, steps = run(
        "continuous",
        [request("long", 0), request("new", 1)],
        {"long": 3, "new": 1},
    )

    assert completed["new"].started_ms == 1
    assert completed["new"].started_ms < completed["long"].finished_ms
    assert [step.batch_size for step in steps] == [1, 2, 1]


def test_requests_are_not_lost_duplicated_or_admitted_early() -> None:
    requests = [request("a", 0), request("b", 2), request("c", 5)]
    completed, _ = run("continuous", requests, {"a": 2, "b": 1, "c": 1})

    assert set(completed) == {"a", "b", "c"}
    assert len(completed) == len(requests)
    assert all(result.started_ms >= result.arrival_ms for result in completed.values())
    assert all(result.queue_ms >= 0 for result in completed.values())
    assert all(result.ttft_ms >= 0 for result in completed.values())
    assert all(result.latency_ms >= 0 for result in completed.values())


def test_dynamic_wait_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        make_policy("dynamic", 2, math.nan)
