Dynamic Batch Formation Simulator
Team 16

Team members
- Appaji Nagaraja Dheeraj, 241CS110, leader
- Rishinandan D R, 241CS145
- Ashlesh Prabhu, 241CS214
- Sachin Rohra, 241CS251

Mentor
- Balaji, balajink252is009@nitk.edu.in

Contents
- Dynamic-Batch-Formation-Simulator/   project folder with source code, Docker files,
                                        workloads, tests, report sources, and raw results
- EXPERIMENTAL_REPORT.pdf              project report
- readme.txt                           this file

GitHub repository
https://github.com/AppajiDheeraj/Dynamic-Batch-Formation-Simulator

Requirements
- NVIDIA GPU with a current driver
- Docker with the NVIDIA container runtime (Docker Desktop with WSL 2 on Windows)

Run with the prebuilt image

cd Dynamic-Batch-Formation-Simulator
docker pull appajidheeraj/dynamic-batch-formation-simulator:latest
docker compose run --rm benchmark

Build and run from source

cd Dynamic-Batch-Formation-Simulator
docker compose build
docker compose run --rm benchmark

Run the CPU-only tests

cd Dynamic-Batch-Formation-Simulator
docker compose run --rm benchmark pytest -q

Results
- Fresh runs are written to Dynamic-Batch-Formation-Simulator/results/
- Raw data for the reported run:
  Dynamic-Batch-Formation-Simulator/docs/results/windows-rtx-5050-2026-10-10-wait50/
- Report source (LaTeX): Dynamic-Batch-Formation-Simulator/docs/EXPERIMENTAL_REPORT.tex

See Dynamic-Batch-Formation-Simulator/README.md for all options and details.
