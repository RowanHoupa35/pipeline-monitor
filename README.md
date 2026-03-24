# LLM-Powered Pipeline Monitor

> An intelligent monitoring dashboard for ML training pipelines — powered by a local LLM that automatically detects anomalies, scores run health, and generates human-readable alerts.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20%2F%20qwen2.5:3b-black?logo=ollama)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project simulates a real-world MLOps scenario: multiple training runs of a **DistilBERT fake-news detection model** are executed, failures are injected (OOM crashes, timeouts, accuracy drift), and an intelligent agent parses the logs, computes a health score per run, and calls a **local LLM via Ollama** to generate a natural language anomaly report — all visualised in a live Streamlit dashboard.

```
┌─────────────────┐     JSON logs      ┌──────────────┐     score + alerts     ┌───────────────┐
│   Simulator     │ ─────────────────► │  Log Parser  │ ──────────────────────► │ Health Scorer │
│  (DistilBERT    │                    │              │                          │  (0–100 pts)  │
│   fake runs)    │                    └──────────────┘                          └───────┬───────┘
└─────────────────┘                                                                      │
                                                                                         ▼
                                                                               ┌─────────────────┐
                                                                               │   LLM Agent     │
                                                                               │  (Ollama local) │
                                                                               └────────┬────────┘
                                                                                        │
                                                                                        ▼
                                                                               ┌─────────────────┐
                                                                               │ Streamlit        │
                                                                               │ Dashboard        │
                                                                               └─────────────────┘
```

---

## Features

- **Run Simulator** — generates realistic DistilBERT training logs with injected failure modes:
  - `OOM` — CUDA out-of-memory crash
  - `Timeout` — wall-clock budget exceeded mid-epoch
  - `Accuracy drift` — validation accuracy regresses across epochs
  - `Clean` — healthy baseline run

- **Health Scorer** — computes a composite score (0–100) per run across 4 dimensions:

  | Dimension | Weight | Description |
  |-----------|--------|-------------|
  | Completion | 30 pts | Did all epochs finish? |
  | Final accuracy | 30 pts | Scaled against 0.70 baseline |
  | Accuracy trend | 20 pts | Improving or regressing? |
  | Error severity | 20 pts | Penalty for OOM / timeout / warnings |

- **LLM Agent** — sends a structured digest to a local `qwen2.5:3b` model via Ollama and receives a Markdown report with executive summary, root-cause analysis, and recommended actions

- **Streamlit Dashboard** — live visualisations:
  - KPI cards (total runs, avg score, healthy/critical counts)
  - Health score bar chart with healthy/critical thresholds
  - Accuracy-per-epoch curves for all runs
  - Run summary table with colour-coded scores
  - Expandable alert panels per run
  - One-click LLM alert generation

---

## Project Structure

```
pipeline_monitor/
├── agent/
│   ├── log_parser.py        # Parses JSON run logs into RunRecord dataclasses
│   ├── health_scorer.py     # Computes health scores and generates alerts
│   └── llm_agent.py         # Calls Ollama REST API, fallback rule-based summary
├── simulator/
│   └── run_pipeline.py      # Simulates DistilBERT runs with injected failures
├── dashboard/
│   └── app.py               # Streamlit UI
├── logs/                    # Auto-generated JSON run logs (git-ignored)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Ollama](https://ollama.com/download) installed and running locally

```bash
# Pull the model (one-time, ~1.9 GB)
ollama pull qwen2.5:3b
```

### Run with Docker

```bash
git clone https://github.com/<your-username>/pipeline-monitor.git
cd pipeline-monitor/pipeline_monitor

docker-compose up --build
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **Linux users:** if you get a Docker permission error, run:
> ```bash
> sudo usermod -aG docker $USER && newgrp docker
> ```

### Run without Docker

```bash
pip install -r requirements.txt

# Generate simulation logs
python -m simulator.run_pipeline

# Launch dashboard
streamlit run dashboard/app.py
```

---

## Configuration

All settings are controlled via environment variables (or `docker-compose.yml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Model to use for alert generation |

To use a lighter model (if RAM is limited):
```bash
ollama pull qwen2.5:0.5b
OLLAMA_MODEL=qwen2.5:0.5b docker-compose up
```

---

## How It Works

### 1. Simulation
The simulator generates `n` training runs (configurable via the dashboard slider). Each run produces a structured JSON log:

```json
{
  "run_id": "run_001_a3f9bc",
  "failure_mode": "accuracy_drift",
  "status": "completed",
  "config": { "batch_size": 8, "learning_rate": 3e-5, "epochs": 5 },
  "epoch_metrics": [
    { "epoch": 1, "accuracy": 0.703, "val_loss": 0.541, "f1": 0.698 },
    ...
  ],
  "events": [ ... ]
}
```

### 2. Health Scoring
Each log is parsed and scored. Runs are classified as:
- ✅ **Healthy** — score ≥ 70
- ⚠️ **Degraded** — score 40–69
- ❌ **Critical** — score < 40

### 3. LLM Alert
The agent sends a compact digest of all runs to `qwen2.5:3b` via the Ollama REST API. The model returns a structured Markdown report. If Ollama is unavailable, a deterministic rule-based fallback generates the report instead.

---

## Applied Fixes (v2)

During development, the following issues were identified and resolved:

| Fix | Problem | Solution |
|-----|---------|----------|
| P1 | CUDA OOM on 8 GiB GPU | `batch_size` 16 → 8 |
| P2 | Accuracy drift in early epochs | Added `warmup_steps=500` |
| P3 | Timeout at epoch 2 | `max_len` 128 → 64 |
| P4 | Model not converged at epoch 3 | `epochs` 3 → 5 |
| P5 | Slow convergence | `learning_rate` 2e-5 → 3e-5 |

Post-fix results (15 runs): **avg score 66.5/100 · 0 critical · 93% of runs ≥ baseline accuracy 0.70**

---

## Stack

| Component | Technology |
|-----------|-----------|
| Model | DistilBERT (`distilbert-base-uncased`) |
| Dataset | [liar2](https://huggingface.co/datasets/chengxuphd/liar2) |
| LLM | Ollama · qwen2.5:3b (local, no API key) |
| Dashboard | Streamlit + Plotly |
| Containerisation | Docker + Docker Compose |
| Logs | Structured JSON |

---

## License

MIT — feel free to use, adapt, and build on this project.
