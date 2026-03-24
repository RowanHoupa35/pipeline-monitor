"""
LLM agent: sends a structured anomaly digest to a local Ollama instance
and returns a human-readable alert summary and recommended action.

Provider: Ollama (local, no API key required)
Model:    qwen2.5:3b  (configurable via OLLAMA_MODEL env var)
Endpoint: http://localhost:11434  (network_mode: host in Docker)
"""

import json
import os
from dataclasses import asdict

import httpx

from .log_parser import RunRecord
from .health_scorer import HealthReport


_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_MODEL       = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")


# ── Prompt builder ──────────────────────────────────────────────────────────

def _build_prompt(records: list[RunRecord], reports: list[HealthReport]) -> str:
    digest = []
    for rec, rep in zip(records, reports):
        digest.append({
            "run_id": rec.run_id,
            "status": rec.status,
            "failure_mode": rec.failure_mode,
            "health_score": rep.score,
            "health_status": rep.status,
            "epochs_completed": f"{rec.epochs_completed}/{rec.expected_epochs}",
            "final_accuracy": rec.final_accuracy,
            "accuracy_trend": rec.accuracy_trend,
            "score_breakdown": rep.breakdown,
            "detected_alerts": rep.alerts,
            "error": asdict(rec.error) if rec.error else None,
        })

    return f"""You are an MLOps monitoring agent. Below is a JSON digest of recent \
DistilBERT training runs on the fake-news detection pipeline (liar2 dataset, baseline \
accuracy ~0.70).

Each run has a health score (0-100) and detected anomalies. Your job is to:
1. Write a concise **executive summary** (3-5 sentences) of the overall pipeline health.
2. List each critical or degraded run with a one-line root-cause and **recommended action**.
3. End with a **priority alert** (one sentence) if any run scored below 40.

Be specific, technical, and actionable. Format your response in Markdown.

--- PIPELINE DIGEST ---
{json.dumps(digest, indent=2)}
--- END DIGEST ---"""


# ── Public API ──────────────────────────────────────────────────────────────

def generate_alert(
    records: list[RunRecord],
    reports: list[HealthReport],
    model: str = _MODEL,
) -> str:
    """
    Call the local Ollama REST API and return a Markdown-formatted alert summary.
    Falls back to a rule-based summary if Ollama is unreachable.
    """
    prompt = _build_prompt(records, reports)
    url = f"{_OLLAMA_HOST.rstrip('/')}/api/chat"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    try:
        response = httpx.post(url, json=payload, timeout=120.0)
        response.raise_for_status()
        return response.json()["message"]["content"]

    except httpx.ConnectError as e:
        return _fallback_summary(records, reports, reason=f"Ollama unreachable ({e})")
    except httpx.HTTPStatusError as e:
        return _fallback_summary(records, reports, reason=f"Ollama error {e.response.status_code}: {e.response.text[:120]}")
    except Exception as e:
        return _fallback_summary(records, reports, reason=str(e))


def _fallback_summary(
    records: list[RunRecord],
    reports: list[HealthReport],
    reason: str = "LLM unavailable",
) -> str:
    """Generate a rule-based summary when Ollama is not accessible."""
    lines = [
        f"## Pipeline Monitor Report _(LLM offline – {reason})_\n",
        f"**Runs analysed:** {len(records)}",
    ]

    healthy  = [r for r in reports if r.status == "healthy"]
    degraded = [r for r in reports if r.status == "degraded"]
    critical = [r for r in reports if r.status == "critical"]

    lines.append(
        f"**Status:** {len(healthy)} healthy · {len(degraded)} degraded · {len(critical)} critical\n"
    )

    if critical or degraded:
        lines.append("### Anomalies detected\n")
        for rep in critical + degraded:
            lines.append(f"**{rep.run_id}** — score {rep.score}/100 ({rep.status})")
            for alert in rep.alerts:
                lines.append(f"  - {alert}")
            lines.append("")

    if critical:
        lines.append(
            f"> **PRIORITY ALERT:** {len(critical)} run(s) are in CRITICAL state "
            "and require immediate investigation."
        )

    return "\n".join(lines)
