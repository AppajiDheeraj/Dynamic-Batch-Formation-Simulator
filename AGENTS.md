# AGENTS.md

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

LLMs often pick an interpretation silently and run with it. This principle forces explicit reasoning:

- **State assumptions explicitly** — If uncertain, ask rather than guess
- **Present multiple interpretations** — Don't pick silently when ambiguity exists
- **Push back when warranted** — If a simpler approach exists, say so
- **Stop when confused** — Name what's unclear and ask for clarification

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

Combat the tendency toward overengineering:

- No features beyond what was asked
- No abstractions for single-use code
- No "flexibility" or "configurability" that wasn't requested
- No error handling for impossible scenarios
- If 200 lines could be 50, rewrite it

**The test:** Would a senior engineer say this is overcomplicated? If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting
- Don't refactor things that aren't broken
- Match existing style, even if you'd do it differently
- If you notice unrelated dead code, mention it — don't delete it

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused
- Don't remove pre-existing dead code unless asked

**The test:** Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform imperative tasks into verifiable goals:

| Instead of... | Transform to... |
|--------------|-----------------|
| "Add validation" | "Write tests for invalid inputs, then make them pass" |
| "Fix the bug" | "Write a test that reproduces it, then make it pass" |
| "Refactor X" | "Ensure tests pass before and after" |

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let the LLM loop independently. Weak criteria ("make it work") require constant clarification.

## Engineering principles

- Prefer simple, readable, flat code with minimal indirection.
- Search for existing implementations and installed libraries before creating new helpers or abstractions.
- Abstract when it prevents meaningful drift and makes the result simpler to maintain. Avoid speculative or one-use abstraction layers.
- Use idiomatic Codes and validate untrusted data at trust boundaries.
- Prefer established project helpers and libraries over hand-rolled implementations.
- Verify user-visible changes on every affected platform available locally and report any platform not tested.

## Project: Dynamic Batch Formation Simulator

### Objective

Build an original, Dockerized benchmarking framework that implements and compares three LLM inference scheduling strategies:

- **Fixed batching:** wait for a fixed-size group, then process the whole batch together.
- **Dynamic batching:** form batches from queued requests using configurable size and wait-time limits.
- **Continuous batching:** admit new requests as active sequences finish, scheduling work at token/iteration granularity.

The project must make the behavioral and performance differences between the strategies measurable and easy to explain. It is a simulator/benchmarking assignment; do not copy an existing implementation from vLLM, ORCA, or another repository.

### Required deliverables

The repository is complete only when it contains:

- Working Python 3.10+ source code for all three batching strategies.
- A reproducible workload generator with request arrival times and varying input/output sequence lengths.
- A common benchmark runner so every strategy receives the same workload and resource assumptions.
- Raw experiment results and performance comparison graphs.
- A `Dockerfile` and Docker Compose configuration that run the benchmark without a host Python setup.
- A clear `README.md` with architecture, setup, Docker commands, experiment commands, output locations, and interpretation guidance.
- An experimental report in PDF form covering understanding, design, workflow, experiments, results, limitations, and conclusions.
- A GitHub-ready source repository. Never commit secrets, generated caches, virtual environments, or large model weights.

### Evaluation metrics

At minimum, record and compare:

- Throughput (requests/second and, where modeled, tokens/second).
- End-to-end latency, including mean and percentile values such as p50, p95, and p99.
- Queue/waiting time.
- Time to first token when the simulation models token-level generation.
- Makespan and resource utilization.
- Batch-size behavior over time.

Graphs must be generated from saved experiment data rather than hand-entered values. Use fixed random seeds and preserve experiment parameters with each result so runs are reproducible.

### Implementation boundaries

- Prefer a lightweight discrete-event simulator using Python, NumPy, Pandas, and Matplotlib. Use PyTorch only if an actual tensor/model workload is intentionally benchmarked.
- Keep one request model and one metrics pipeline shared by all strategies; vary only scheduling behavior.
- Separate simulated time from wall-clock runtime and label results clearly. Never present simulated measurements as real GPU inference measurements.
- Use identical workloads, capacity assumptions, warm-up rules, and metric definitions when comparing strategies.
- Make workload parameters configurable at the command line or in one small configuration file: request count, arrival rate, input/output length distribution, batch capacity, batching timeout, and random seed.
- Validate configuration and untrusted input at the boundary. Fail with an actionable message when a configuration is invalid.
- Keep the default experiment small enough to run on a CPU in Docker. Optional GPU experiments must not be required for basic setup or grading.
- Add the smallest useful automated tests for scheduler invariants and metric calculations. In particular, requests must not be lost, duplicated, processed before arrival, or reported with negative timing values.
- Do not add FastAPI, Flask, Prometheus, Grafana, JMeter, Kafka, or other infrastructure unless a concrete project requirement later calls for it.

### Definition of done

Before calling the project complete, verify:

1. The test suite passes.
2. One command builds the Docker image and one documented Docker Compose command runs the full default experiment.
3. The default run executes all three strategies against the same seeded workload.
4. The run writes machine-readable raw results plus labeled comparison graphs to documented locations.
5. A clean checkout can reproduce the documented outputs using only Docker and Docker Compose.
6. The README and report describe the exact implementation and actual results; they must not claim unrun experiments.
7. Every locally available affected platform is checked, and any platform not tested is stated explicitly.

### Academic and submission constraints

- Implementation is worth 12 marks, documentation/submission 5 marks, and understanding/presentation 8 marks.
- Submission deadline: **November 1, 2026 at 11:59 pm**, via Moodle only.
- Obtain mentor approval after demonstrating that the code runs.
- Upload the source to GitHub with a proper README and share it with the mentor.
- Final submission archive format: `<firstname>_<rollnumber>.zip`.
- The archive's master folder must contain exactly the project folder, the report PDF, and `readme.txt`.
- A detailed presentation/mid-semester evaluation is expected around the third week of November 2026.

### Primary references

- ORCA (OSDI 2022): https://www.usenix.org/conference/osdi22/presentation/yu
- vLLM: https://github.com/vllm-project/vllm
- Docker documentation: https://www.docker.com/

Use these references to understand scheduling concepts and terminology. Cite borrowed ideas in the report and README, but write original code suited to this simulator.
