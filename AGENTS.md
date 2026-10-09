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

### Problem statement

Implement **fixed batching**, **dynamic batching**, and **continuous batching** strategies for LLM inference and compare their performance inside a Docker environment.

- **Expected output:** Dockerized benchmarking framework, performance comparison graphs, experimental report.
- **Mandatory:** `Dockerfile`, Docker Compose configuration, `README.md` with setup instructions, source code repository, experimental evaluation with graphs.
- **References:** ORCA (OSDI '22) https://www.usenix.org/conference/osdi22/presentation/yu · vLLM https://github.com/vllm-project/vllm · Docker https://www.docker.com/

### TA guidance (meeting)

- This is **not a simulation**. Load a real LLM and run real inference.
- Use a GPU model that fits in **3–4 GB VRAM** on an **NVIDIA RTX** GPU, running **inside our Docker container**.
- Run the tests **sequentially**, one batching strategy at a time, end to end.
- ORCA was cited as a reference (it introduced iteration-level scheduling, i.e. continuous batching). vLLM is also a another reference.
- Final results need proper graphs.

### Assignment rules

- **Marks (25):** Implementation 12 · Documentation and submission 5 · Understanding and presentation 8.
- **Deadline:** November 01, 2026, 11:59 pm, Moodle only. Mid-sem evaluation and detailed presentation around the 3rd week of November 2026.
- Understand and implement the problem (topic, problem, technology adopted, tool used).
- Report (PDF): understanding, workflow, and results generated.
- Upload source to GitHub with a proper README and share with the mentor. **Do not copy existing source code.**
- Code must be running and **approved by the mentor**.
- Any one member submits. Zip name: `<firstname>_<rollnumber>.zip` (e.g. `xyz_191CS101.zip`). Master folder holds exactly 3 items: project folder, report (PDF), `readme.txt`.

### Tools

If the problem statement names a tool or version, follow it over this list. Course-wide pool (use only what this project needs): Python 3.10+, PyTorch, NumPy, Pandas, SciPy, Matplotlib, Git · Docker and Docker Compose · FastAPI or Flask · nginx or Envoy · Prometheus and Grafana · InfluxDB · Apache JMeter · Kafka, MQTT (Mosquitto) · SQLite · AWS EC2 or free cloud GPU (Colab, Kaggle) where specified · Ollama or llama.cpp, ngrok, Flower (flwr) for the LLM/FL projects that need them. (SUMO/TraCI and CloudSim Plus are for other groups' projects.)

### Team and repository

- Repo: https://github.com/AppajiDheeraj/Dynamic-Batch-Formation-Simulator
- GitHub collaborators: `AppajiDheeraj` (owner), `AshleshPrabhu`, `rishi746`, `Sachin210506`
