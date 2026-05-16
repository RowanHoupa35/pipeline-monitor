"""
Computes a health score (0–100) for each run.

Score breakdown:
  Completion     25 pts  – did the run finish all epochs?
  Final accuracy 20 pts  – scaled around the 0.70 baseline
  Final F1       10 pts  – F1 score scaled around the 0.68 baseline
  Accuracy trend 15 pts  – is the model improving across epochs?
  Val loss       15 pts  – final val_loss level (10 pts) + train/val gap (5 pts)
  Error severity 15 pts  – penalise OOM / timeout / drift warnings
"""

from dataclasses import dataclass
from typing import Optional

from .log_parser import RunRecord


ACCURACY_BASELINE = 0.70
ACCURACY_CEIL     = 0.85

F1_BASELINE = 0.68
F1_CEIL     = 0.83

VAL_LOSS_GOOD   = 0.30   # val_loss ≤ this → full level score
VAL_LOSS_BAD    = 0.70   # val_loss ≥ this → 0 level score
OVERFIT_GAP_OK  = 0.05   # train/val gap ≤ this → full gap score
OVERFIT_GAP_BAD = 0.25   # train/val gap ≥ this → 0 gap score


@dataclass
class HealthReport:
    run_id: str
    score: float           # 0–100
    status: str            # "healthy" | "degraded" | "critical"
    breakdown: dict        # sub-scores per dimension
    alerts: list[str]      # human-readable alert strings


def score_run(record: RunRecord) -> HealthReport:
    alerts = []
    breakdown = {}

    # ── 1. Completion (25 pts) ────────────────────────────────────────────────
    completion_ratio = record.epochs_completed / max(record.expected_epochs, 1)
    completion_score = round(completion_ratio * 25, 1)
    breakdown["completion"] = completion_score

    if record.status == "failed":
        alerts.append(
            f"Run failed at epoch {record.error.epoch}/{record.expected_epochs}"
            f" – {record.error.type.upper()}: {record.error.detail}"
        )

    # ── 2. Final accuracy (20 pts) ────────────────────────────────────────────
    if record.final_accuracy is not None:
        acc = record.final_accuracy
        acc_score = (acc - ACCURACY_BASELINE) / (ACCURACY_CEIL - ACCURACY_BASELINE) * 20
        acc_score = round(max(0.0, min(20.0, acc_score)), 1)

        if acc < ACCURACY_BASELINE - 0.05:
            alerts.append(f"Accuracy {acc:.3f} is significantly below baseline ({ACCURACY_BASELINE})")
        elif acc < ACCURACY_BASELINE:
            alerts.append(f"Accuracy {acc:.3f} is slightly below baseline ({ACCURACY_BASELINE})")
    else:
        acc_score = 0.0
        alerts.append("No accuracy metrics recorded (run may have crashed before first epoch)")

    breakdown["accuracy"] = acc_score

    # ── 3. Final F1 (10 pts) ─────────────────────────────────────────────────
    final_f1 = record.epoch_metrics[-1].f1 if record.epoch_metrics else None
    if final_f1 is not None:
        f1_score = (final_f1 - F1_BASELINE) / (F1_CEIL - F1_BASELINE) * 10
        f1_score = round(max(0.0, min(10.0, f1_score)), 1)

        if final_f1 < F1_BASELINE - 0.05:
            alerts.append(f"F1 score {final_f1:.3f} is significantly below threshold ({F1_BASELINE})")
        elif final_f1 < F1_BASELINE:
            alerts.append(f"F1 score {final_f1:.3f} is below acceptable threshold")
    else:
        f1_score = 0.0

    breakdown["f1"] = f1_score

    # ── 4. Accuracy trend (15 pts) ────────────────────────────────────────────
    trend = record.accuracy_trend
    if trend is None:
        trend_score = 7.5  # neutral if only one epoch
    elif trend >= 0:
        trend_score = round(min(15.0, 7.5 + trend * 150), 1)
    else:
        trend_score = round(max(0.0, 7.5 + trend * 150), 1)
        if trend < -0.02:
            alerts.append(f"Accuracy drift: -{abs(trend):.3f} over {record.epochs_completed} epochs")

    breakdown["trend"] = trend_score

    # ── 5. Val loss quality (15 pts) ─────────────────────────────────────────
    val_loss_score = 0.0
    if record.epoch_metrics:
        last           = record.epoch_metrics[-1]
        final_val_loss = last.val_loss
        gap            = last.val_loss - last.train_loss

        # Val loss level: how low is the final validation loss? (10 pts)
        if final_val_loss <= VAL_LOSS_GOOD:
            val_level = 10.0
        elif final_val_loss >= VAL_LOSS_BAD:
            val_level = 0.0
        else:
            val_level = (VAL_LOSS_BAD - final_val_loss) / (VAL_LOSS_BAD - VAL_LOSS_GOOD) * 10
        val_level = round(val_level, 1)

        # Train/val gap: overfitting indicator (5 pts)
        if gap <= OVERFIT_GAP_OK:
            gap_score = 5.0
        elif gap >= OVERFIT_GAP_BAD:
            gap_score = 0.0
        else:
            gap_score = (OVERFIT_GAP_BAD - gap) / (OVERFIT_GAP_BAD - OVERFIT_GAP_OK) * 5
        gap_score = round(gap_score, 1)

        val_loss_score = val_level + gap_score

        if final_val_loss > 0.55:
            alerts.append(f"High val_loss {final_val_loss:.3f} — potential convergence issue")
        if gap > 0.20:
            alerts.append(f"Train/val loss gap {gap:.3f} — potential overfitting")

    breakdown["val_loss"] = round(val_loss_score, 1)

    # ── 6. Error severity (15 pts) ────────────────────────────────────────────
    error_score = 15.0
    if record.error:
        penalty = {"oom": 15, "timeout": 12, "accuracy_drift": 7}.get(record.error.type, 7)
        error_score = max(0.0, 15.0 - penalty)

    n_warnings = len([e for e in record.warnings if e.get("level") == "WARNING"])
    error_score = max(0.0, error_score - n_warnings * 1)
    breakdown["errors"] = round(error_score, 1)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    total = completion_score + acc_score + f1_score + trend_score + val_loss_score + error_score
    total = round(total, 1)

    status = "healthy" if total >= 70 else ("degraded" if total >= 40 else "critical")

    return HealthReport(
        run_id=record.run_id,
        score=total,
        status=status,
        breakdown=breakdown,
        alerts=alerts,
    )


def score_all(records: list[RunRecord]) -> list[HealthReport]:
    return [score_run(r) for r in records]
