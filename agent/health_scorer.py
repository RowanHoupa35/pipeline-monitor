"""
Computes a health score (0–100) for each run.

Score breakdown:
  Completion     30 pts  – did the run finish all epochs?
  Final accuracy 30 pts  – scaled around the 0.70 baseline
  Accuracy trend 20 pts  – is the model improving across epochs?
  Error severity 20 pts  – penalise OOM / timeout / drift warnings
"""

from dataclasses import dataclass
from typing import Optional

from .log_parser import RunRecord


ACCURACY_BASELINE = 0.70   # from the existing notebook results
ACCURACY_CEIL     = 0.85   # 100% accuracy score target


@dataclass
class HealthReport:
    run_id: str
    score: float           # 0–100
    status: str            # "healthy" | "degraded" | "critical"
    breakdown: dict        # sub-scores
    alerts: list[str]      # human-readable alert strings


def score_run(record: RunRecord) -> HealthReport:
    alerts = []
    breakdown = {}

    # ── 1. Completion (30 pts) ────────────────────────────────────────────────
    completion_ratio = record.epochs_completed / max(record.expected_epochs, 1)
    completion_score = round(completion_ratio * 30, 1)
    breakdown["completion"] = completion_score

    if record.status == "failed":
        alerts.append(
            f"Run failed at epoch {record.error.epoch}/{record.expected_epochs}"
            f" – {record.error.type.upper()}: {record.error.detail}"
        )

    # ── 2. Final accuracy (30 pts) ────────────────────────────────────────────
    if record.final_accuracy is not None:
        acc = record.final_accuracy
        # Linear scale: baseline→0 pts, ceil→30 pts; clip to [0, 30]
        acc_score = (acc - ACCURACY_BASELINE) / (ACCURACY_CEIL - ACCURACY_BASELINE) * 30
        acc_score = round(max(0.0, min(30.0, acc_score)), 1)

        if acc < ACCURACY_BASELINE - 0.05:
            alerts.append(f"Accuracy {acc:.3f} is significantly below baseline ({ACCURACY_BASELINE})")
        elif acc < ACCURACY_BASELINE:
            alerts.append(f"Accuracy {acc:.3f} is slightly below baseline ({ACCURACY_BASELINE})")
    else:
        acc_score = 0.0
        alerts.append("No accuracy metrics recorded (run may have crashed before first epoch)")

    breakdown["accuracy"] = acc_score

    # ── 3. Accuracy trend (20 pts) ────────────────────────────────────────────
    trend = record.accuracy_trend
    if trend is None:
        trend_score = 10.0  # neutral if only one epoch
    elif trend >= 0:
        trend_score = round(min(20.0, 10.0 + trend * 200), 1)
    else:
        trend_score = round(max(0.0, 10.0 + trend * 200), 1)
        if trend < -0.02:
            alerts.append(f"Accuracy drift: -{abs(trend):.3f} over {record.epochs_completed} epochs")

    breakdown["trend"] = trend_score

    # ── 4. Error severity (20 pts) ────────────────────────────────────────────
    error_score = 20.0
    if record.error:
        penalty = {"oom": 20, "timeout": 15, "accuracy_drift": 10}.get(record.error.type, 10)
        error_score = max(0.0, 20.0 - penalty)

    # Extra penalty per WARNING event
    n_warnings = len([e for e in record.warnings if e.get("level") == "WARNING"])
    error_score = max(0.0, error_score - n_warnings * 2)
    breakdown["errors"] = round(error_score, 1)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    total = completion_score + acc_score + trend_score + error_score
    total = round(total, 1)

    if total >= 70:
        status = "healthy"
    elif total >= 40:
        status = "degraded"
    else:
        status = "critical"

    return HealthReport(
        run_id=record.run_id,
        score=total,
        status=status,
        breakdown=breakdown,
        alerts=alerts,
    )


def score_all(records: list[RunRecord]) -> list[HealthReport]:
    return [score_run(r) for r in records]
