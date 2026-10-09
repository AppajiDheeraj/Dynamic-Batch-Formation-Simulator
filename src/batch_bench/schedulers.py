from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol, Sequence


@dataclass(frozen=True)
class RequestSpec:
    id: str
    prompt: str
    arrival_ms: float
    max_new_tokens: int


@dataclass
class ActiveRequest:
    spec: RequestSpec
    input_ids: list[int]
    generated_ids: list[int]
    prompt_tokens: int
    started_ms: float
    first_token_ms: float | None = None
    finished_ms: float | None = None
    eos_reached: bool = False


@dataclass(frozen=True)
class CompletedRequest:
    id: str
    arrival_ms: float
    started_ms: float
    first_token_ms: float
    finished_ms: float
    queue_ms: float
    ttft_ms: float
    latency_ms: float
    prompt_tokens: int
    output_tokens: int
    output_text: str


@dataclass
class RunState:
    future: deque[RequestSpec]
    waiting: deque[RequestSpec] = field(default_factory=deque)
    active: list[ActiveRequest] = field(default_factory=list)
    completed: list[CompletedRequest] = field(default_factory=list)


@dataclass(frozen=True)
class AdmissionDecision:
    count: int = 0
    wake_at_ms: float | None = None


@dataclass(frozen=True)
class StepRecord:
    step: int
    start_ms: float
    duration_ms: float
    batch_size: int


class Clock(Protocol):
    def now_ms(self) -> float: ...

    def sleep_ms(self, duration_ms: float) -> None: ...


class InferenceEngine(Protocol):
    def activate(self, spec: RequestSpec, started_ms: float) -> ActiveRequest: ...

    def step(self, active: Sequence[ActiveRequest]) -> None: ...

    def decode(self, request: ActiveRequest) -> str: ...


class Policy(Protocol):
    def decide(self, state: RunState, now_ms: float) -> AdmissionDecision: ...


class SystemClock:
    def __init__(self) -> None:
        self._start_ns = time.perf_counter_ns()

    def now_ms(self) -> float:
        return (time.perf_counter_ns() - self._start_ns) / 1_000_000

    def sleep_ms(self, duration_ms: float) -> None:
        if duration_ms > 0:
            time.sleep(duration_ms / 1000)


@dataclass(frozen=True)
class FixedPolicy:
    batch_size: int

    def decide(self, state: RunState, now_ms: float) -> AdmissionDecision:
        del now_ms
        if state.active:
            return AdmissionDecision()
        if len(state.waiting) >= self.batch_size:
            return AdmissionDecision(self.batch_size)
        if not state.future and state.waiting:
            return AdmissionDecision(len(state.waiting))
        return AdmissionDecision()


@dataclass(frozen=True)
class DynamicPolicy:
    batch_size: int
    wait_ms: float

    def decide(self, state: RunState, now_ms: float) -> AdmissionDecision:
        if state.active or not state.waiting:
            return AdmissionDecision()
        if len(state.waiting) >= self.batch_size:
            return AdmissionDecision(self.batch_size)
        deadline = state.waiting[0].arrival_ms + self.wait_ms
        if now_ms >= deadline:
            return AdmissionDecision(len(state.waiting))
        return AdmissionDecision(wake_at_ms=deadline)


@dataclass(frozen=True)
class ContinuousPolicy:
    batch_size: int

    def decide(self, state: RunState, now_ms: float) -> AdmissionDecision:
        del now_ms
        return AdmissionDecision(min(self.batch_size - len(state.active), len(state.waiting)))


def make_policy(strategy: str, batch_size: int, dynamic_wait_ms: float) -> Policy:
    if batch_size < 1:
        raise ValueError("batch size must be at least 1")
    if dynamic_wait_ms < 0:
        raise ValueError("dynamic wait must not be negative")
    if strategy == "fixed":
        return FixedPolicy(batch_size)
    if strategy == "dynamic":
        return DynamicPolicy(batch_size, dynamic_wait_ms)
    if strategy == "continuous":
        return ContinuousPolicy(batch_size)
    raise ValueError(f"unknown strategy: {strategy}")


def _release_arrivals(state: RunState, now_ms: float) -> None:
    while state.future and state.future[0].arrival_ms <= now_ms:
        state.waiting.append(state.future.popleft())


def _assert_lifecycle(state: RunState, expected_ids: frozenset[str]) -> None:
    ids = [request.id for request in state.future]
    ids.extend(request.id for request in state.waiting)
    ids.extend(request.spec.id for request in state.active)
    ids.extend(request.id for request in state.completed)
    if len(ids) != len(set(ids)) or frozenset(ids) != expected_ids:
        raise RuntimeError("a request was lost or exists in more than one lifecycle collection")


def run_scheduler(
    requests: Sequence[RequestSpec],
    policy: Policy,
    engine: InferenceEngine,
    clock: Clock,
) -> tuple[list[CompletedRequest], list[StepRecord]]:
    ordered = sorted(requests, key=lambda request: request.arrival_ms)
    expected_ids = frozenset(request.id for request in ordered)
    if len(expected_ids) != len(ordered):
        raise ValueError("request IDs must be unique")
    state = RunState(future=deque(ordered))
    steps: list[StepRecord] = []
    _assert_lifecycle(state, expected_ids)

    while state.future or state.waiting or state.active:
        now_ms = clock.now_ms()
        _release_arrivals(state, now_ms)
        decision = policy.decide(state, now_ms)
        if decision.count < 0 or decision.count > len(state.waiting):
            raise RuntimeError("policy returned an invalid admission count")
        for _ in range(decision.count):
            spec = state.waiting.popleft()
            if spec.arrival_ms > now_ms:
                raise RuntimeError("policy admitted a request before its arrival")
            state.active.append(engine.activate(spec, now_ms))
        _assert_lifecycle(state, expected_ids)

        if state.active:
            step_start = clock.now_ms()
            batch_size = len(state.active)
            token_counts = [len(request.generated_ids) for request in state.active]
            engine.step(state.active)
            step_end = clock.now_ms()
            steps.append(StepRecord(len(steps), step_start, step_end - step_start, batch_size))

            survivors: list[ActiveRequest] = []
            for request, previous_count in zip(state.active, token_counts):
                if len(request.generated_ids) != previous_count + 1:
                    raise RuntimeError("the inference engine must generate one token per active request")
                if request.first_token_ms is None:
                    request.first_token_ms = step_end
                if request.eos_reached or len(request.generated_ids) >= request.spec.max_new_tokens:
                    request.finished_ms = step_end
                    state.completed.append(
                        CompletedRequest(
                            id=request.spec.id,
                            arrival_ms=request.spec.arrival_ms,
                            started_ms=request.started_ms,
                            first_token_ms=request.first_token_ms,
                            finished_ms=step_end,
                            queue_ms=request.started_ms - request.spec.arrival_ms,
                            ttft_ms=request.first_token_ms - request.spec.arrival_ms,
                            latency_ms=step_end - request.spec.arrival_ms,
                            prompt_tokens=request.prompt_tokens,
                            output_tokens=len(request.generated_ids),
                            output_text=engine.decode(request),
                        )
                    )
                else:
                    survivors.append(request)
            state.active = survivors
            _assert_lifecycle(state, expected_ids)
            continue

        wake_times = []
        if state.future:
            wake_times.append(state.future[0].arrival_ms)
        if decision.wake_at_ms is not None:
            wake_times.append(decision.wake_at_ms)
        if wake_times:
            clock.sleep_ms(max(0, min(wake_times) - clock.now_ms()))
            continue
        if state.waiting:
            raise RuntimeError("scheduler cannot make progress")

    return state.completed, steps
