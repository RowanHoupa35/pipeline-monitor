# LLM-Powered Pipeline Monitor

> An intelligent monitoring dashboard for ML training pipelines — powered by a local LLM that automatically detects anomalies, scores run health, and generates human-readable streaming alerts.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20%2F%20qwen3:8b-black?logo=ollama)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project simulates a real-world MLOps scenario: multiple training runs of a **DistilBERT fake-news detection model** are executed, failures are injected (OOM crashes, timeouts, accuracy drift), and an intelligent agent parses the logs, computes a health score per run, and calls a **local LLM via Ollama** to generate a natural language anomaly report — all visualised in a live Streamlit dashboard with **real-time streaming output**.

```
┌─────────────────┐     JSON logs      ┌──────────────┐     score + alerts     ┌───────────────┐
│   Simulator     │ ─────────────────► │  Log Parser  │ ──────────────────────► │ Health Scorer │
│  (DistilBERT    │                    │              │                          │  (0–100 pts)  │
│   fake runs)    │                    └──────────────┘                          └───────┬───────┘
└─────────────────┘                                                                      │
                                                                                         ▼
                                                                               ┌─────────────────┐
                                                                               │   LLM Agent     │
                                                                               │ (Ollama/qwen3)  │
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

- **Run Simulator** — generates realistic DistilBERT training logs with two modes:
  - `apply_fixes=True` *(default)* — v2 patches applied, all runs complete successfully
  - `apply_fixes=False` — raw failure injection:
    - `OOM` — CUDA out-of-memory crash at a random early epoch
    - `Timeout` — wall-clock budget exceeded mid-run
    - `Accuracy drift` — validation accuracy regresses across epochs
    - `Clean` — healthy baseline run

- **Health Scorer** — computes a composite score (0–100) per run across **6 dimensions**:

  | Dimension | Weight | Description |
  |-----------|--------|-------------|
  | Completion | 25 pts | Did all epochs finish? |
  | Final accuracy | 20 pts | Scaled against 0.70 baseline |
  | Final F1 | 10 pts | F1 score scaled against 0.68 baseline |
  | Accuracy trend | 15 pts | Improving or regressing across epochs? |
  | Val loss quality | 15 pts | Final val_loss level (10 pts) + train/val gap overfitting indicator (5 pts) |
  | Error severity | 15 pts | Penalty for OOM / timeout / warnings |

- **LLM Agent** — sends a structured digest to a local `qwen3:8b` model via Ollama, with:
  - System/user message split for cleaner context
  - Thinking mode disabled (`think: false`) for direct, clean output
  - **Real-time streaming** — response appears token by token in the dashboard

- **Streamlit Dashboard** — live visualisations:
  - KPI cards (total runs, avg score, healthy/critical counts)
  - Health score bar chart with healthy/critical thresholds
  - Accuracy-per-epoch curves for all runs
  - Run summary table with colour-coded scores, **F1 and val_loss columns**
  - Expandable detail panels with **6-dimension score breakdown** per run
  - Detected alert warnings per run
  - One-click streaming LLM alert generation
  - Toggle to switch between fixed (v2) and raw failure modes

---

## Project Structure

```
pipeline_monitor/
├── agent/
│   ├── __init__.py          # Package exports
│   ├── log_parser.py        # Parses JSON run logs into RunRecord dataclasses
│   ├── health_scorer.py     # Computes 6-dimension health scores and alerts
│   └── llm_agent.py         # Ollama REST API, streaming generator, fallback summary
├── simulator/
│   └── run_pipeline.py      # Simulates DistilBERT runs — fixed or raw failure modes
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
# Pull the model (one-time, ~5 GB)
ollama pull qwen3:8b

# Start Ollama server
ollama serve
```

### Run with Docker

```bash
git clone https://github.com/RowanHoupa35/pipeline-monitor.git
cd pipeline-monitor

docker-compose up --build
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

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
| `OLLAMA_MODEL` | `qwen3:8b` | Model to use for alert generation |

To use a different model:
```bash
OLLAMA_MODEL=qwen3:14b streamlit run dashboard/app.py
```

---

## How It Works

### 1. Simulation
The simulator generates `n` training runs (configurable via the dashboard slider). Each run produces a structured JSON log. Use the **"Apply v2 fixes"** checkbox to toggle between:
- **Checked** — all failure modes resolve successfully (v2 patches active)
- **Unchecked** — raw failures injected: OOM crashes, timeouts, real accuracy regression

```json
{
  "run_id": "run_001_a3f9bc",
  "failure_mode": "oom",
  "status": "failed",
  "error": { "type": "oom", "epoch": 2, "detail": "CUDA out of memory at epoch 2: batch_size=16 exhausted 8 GiB GPU RAM." },
  "epoch_metrics": [
    { "epoch": 1, "accuracy": 0.703, "val_loss": 0.541, "f1": 0.698, ... }
  ]
}
```

### 2. Health Scoring
Each log is parsed and scored across 6 dimensions. Runs are classified as:
- ✅ **Healthy** — score ≥ 70
- ⚠️ **Degraded** — score 40–69
- ❌ **Critical** — score < 40

The **val_loss quality** dimension also flags potential overfitting via the train/val loss gap.

### 3. LLM Alert
The agent sends a compact digest of all runs to `qwen3:8b` via the Ollama REST API. The response **streams token by token** into the dashboard. If Ollama is unavailable, a deterministic rule-based fallback generates the report instead.

---

## Changelog

### v2.0
| Change | Detail |
|--------|--------|
| **Model upgrade** | `qwen2.5:3b` → `qwen3:8b` — significantly better reasoning and instruction following |
| **Real failure injection** | Simulator now supports `apply_fixes=False` — OOM, timeout and accuracy drift produce actual failed runs with `status: "failed"` |
| **Enriched health scorer** | 4 dimensions → 6 dimensions: added **Final F1** (10 pts) and **Val loss quality** (15 pts, includes overfitting gap detection) |
| **Streaming LLM output** | Alert text streams token by token via `st.write_stream()` — no more silent waiting |
| **Dashboard improvements** | New F1 & val_loss columns in the summary table; 6-dimension score breakdown in each run expander |
| **Prompt quality** | System/user message split; thinking mode disabled; prompt instructs the model to be accessible to non-technical audiences |

### v1.0
Initial release — Ollama/qwen2.5:3b, 4-dimension scorer, batch alert generation.

---

## Applied Fixes (v2 mode)

When `apply_fixes=True`, the simulator uses these corrected hyperparameters:

| Fix | Problem | Solution |
|-----|---------|----------|
| P1 | CUDA OOM on 8 GiB GPU | `batch_size` 16 → 8 |
| P2 | Accuracy drift in early epochs | Added `warmup_steps=500` |
| P3 | Timeout at epoch 2 | `max_len` 128 → 64 |
| P4 | Model not converged at epoch 3 | `epochs` 3 → 5 |
| P5 | Slow convergence | `learning_rate` 2e-5 → 3e-5 |

---

## Stack

| Component | Technology |
|-----------|-----------|
| Model | DistilBERT (`distilbert-base-uncased`) |
| Dataset | [liar2](https://huggingface.co/datasets/chengxuphd/liar2) |
| LLM | Ollama · qwen3:8b (local, no API key) |
| Dashboard | Streamlit + Plotly |
| Containerisation | Docker + Docker Compose |
| Logs | Structured JSON |

---

## License

MIT — feel free to use, adapt, and build on this project.
