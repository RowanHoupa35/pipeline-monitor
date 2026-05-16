"""
LLM agent: sends a structured anomaly digest to a local Ollama instance
and returns a human-readable alert summary and recommended actions.

Provider: Ollama (local, no API key required)
Model:    qwen3:8b (configurable via OLLAMA_MODEL env var)
Endpoint: http://localhost:11434 (network_mode: host in Docker)
"""

import json
import os
import re
from dataclasses import asdict
from typing import Generator

import httpx

from .log_parser import RunRecord
from .health_scorer import HealthReport


_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_MODEL       = os.environ.get("OLLAMA_MODEL", "qwen3:8b")

_SYSTEM_PROMPT = """You are an MLOps monitoring agent specialized in DistilBERT training pipelines \
on the fake-news detection task (liar2 dataset, baseline accuracy ~0.70).

Given the JSON digest below, write a report in English structured as follows:
1. A concise **executive summary** describing the overall pipeline health in clear, natural language.
2. For each critical or degraded run: a root cause and a concrete **recommended action**.
3. A **priority alert**  if any run scored below 40.

Be specific, technical, and actionable but also keep in mind that the audience may not be full technical so adjust it.."""


# ── Prompt builder ──────────────────────────────────────────────────────────

def _build_user_prompt(records: list[RunRecord], reports: list[HealthReport]) -> str:
    digest = []
    for rec, rep in zip(records, reports):
        digest.append({
            "run_id":           rec.run_id,
            "status":           rec.status,
            "failure_mode":     rec.failure_mode,
            "health_score":     rep.score,
            "health_status":    rep.status,
            "epochs_completed": f"{rec.epochs_completed}/{rec.expected_epochs}",
            "final_accuracy":   rec.final_accuracy,
            "accuracy_trend":   rec.accuracy_trend,
            "score_breakdown":  rep.breakdown,
            "detected_alerts":  rep.alerts,
            "error":            asdict(rec.error) if rec.error else None,
        })

    return (
        "Below is a JSON digest of recent DistilBERT training runs "
        "on the fake-news detection pipeline.\n\n"
        f"--- PIPELINE DIGEST ---\n{json.dumps(digest, indent=2)}\n--- END DIGEST ---"
    )


# ── Public API ──────────────────────────────────────────────────────────────

def _make_payload(prompt: str, model: str, stream: bool) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream":  stream,
        "think":   False,
        "options": {"num_predict": 2048},
    }


def generate_alert_stream(
    records: list[RunRecord],
    reports: list[HealthReport],
    model: str = _MODEL,
) -> Generator[str, None, None]:
    """
    Yield text chunks from the Ollama streaming API.
    Falls back to yielding the full fallback summary in one shot on error.
    """
    prompt = _build_user_prompt(records, reports)
    url    = f"{_OLLAMA_HOST.rstrip('/')}/api/chat"

    try:
        with httpx.stream("POST", url, json=_make_payload(prompt, model, stream=True),
                          timeout=180.0) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
                if chunk.get("done"):
                    break

    except httpx.ConnectError as e:
        yield _fallback_summary(records, reports, reason=f"Ollama unreachable ({e})")
    except httpx.HTTPStatusError as e:
        yield _fallback_summary(
            records, reports,
            reason=f"Ollama error {e.response.status_code}: {e.response.text[:120]}",
        )
    except Exception as e:
        yield _fallback_summary(records, reports, reason=str(e))


def generate_alert(
    records: list[RunRecord],
    reports: list[HealthReport],
    model: str = _MODEL,
) -> str:
    """Non-streaming version — joins all chunks and strips qwen3 thinking blocks."""
    content = "".join(generate_alert_stream(records, reports, model))
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


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
