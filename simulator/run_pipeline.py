"""
Simulates multiple DistilBERT training runs with injected failure modes:
  - OOM (out-of-memory crash)
  - Timeout (run exceeds wall-clock budget)
  - Accuracy drift (validation accuracy regresses across epochs)
  - Clean run (baseline)

Two modes:
  apply_fixes=True  (default) — v2 fixes applied, all runs complete successfully
  apply_fixes=False           — raw failures injected: OOM crashes, timeouts, real accuracy regression

Each run writes a structured JSON log to ../logs/
"""

import json
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ── Failure modes ──────────────────────────────────────────────────────────────

FAILURE_MODES = ["clean", "clean", "oom", "timeout", "accuracy_drift", "clean"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit(events: list, level: str, message: str, **extra):
    events.append({"timestamp": _now(), "level": level, "message": message, **extra})


def simulate_run(run_id: str, failure_mode: str, epochs: int = 5, apply_fixes: bool = True) -> dict:
    """Return a complete run record (dict) with structured events."""
    events = []
    start_ts = _now()

    config = {
        "model": "distilbert-base-uncased",
        "dataset": "chengxuphd/liar2",
        "epochs": epochs,
        "batch_size":    8   if apply_fixes else 16,    # P1: was 16, OOM on 8 GiB GPU
        "learning_rate": 3e-5 if apply_fixes else 2e-5, # P5: was 2e-5, slow convergence
        "warmup_steps":  500  if apply_fixes else 0,    # P2: was 0, early accuracy drift
        "max_len":       64   if apply_fixes else 128,  # P3: was 128, epoch timeout
    }

    _emit(events, "INFO", "Run started", run_id=run_id, failure_mode=failure_mode, config=config)

    epoch_metrics = []
    base_accuracy    = 0.70
    lr_gain_per_epoch = 0.014 if apply_fixes else 0.008
    base_loss        = 0.55
    completed        = False
    error            = None

    # Pre-determine crash epoch for unfixed failures
    crash_epoch = None
    if not apply_fixes:
        if failure_mode == "oom":
            crash_epoch = random.randint(1, max(1, epochs // 2))
        elif failure_mode == "timeout":
            crash_epoch = random.randint(2, max(2, epochs - 1))

    for epoch in range(1, epochs + 1):
        _emit(events, "INFO", f"Epoch {epoch}/{epochs} started", epoch=epoch)

        # ── Crash injection (unfixed) ──────────────────────────────────────────

        if failure_mode == "oom" and not apply_fixes and epoch == crash_epoch:
            error = {
                "type": "oom",
                "epoch": epoch,
                "detail": (
                    f"CUDA out of memory at epoch {epoch}: "
                    "batch_size=16 exhausted 8 GiB GPU RAM."
                ),
            }
            _emit(events, "ERROR", f"CUDA OOM at epoch {epoch} — run aborted",
                  epoch=epoch, error_type="oom")
            break

        if failure_mode == "timeout" and not apply_fixes and epoch == crash_epoch:
            error = {
                "type": "timeout",
                "epoch": epoch,
                "detail": (
                    f"Epoch {epoch} exceeded wall-clock budget: "
                    "max_len=128 tokenisation too slow."
                ),
            }
            _emit(events, "ERROR", f"Timeout at epoch {epoch} — run aborted",
                  epoch=epoch, error_type="timeout")
            break

        # ── Fix-active log messages ────────────────────────────────────────────

        if apply_fixes:
            if failure_mode == "timeout":
                _emit(
                    events, "INFO",
                    "P3 fix active: max_len=64 — epoch duration within wall-clock budget, continuing",
                    epoch=epoch, fix="max_len_reduced",
                )
            elif failure_mode == "oom":
                _emit(
                    events, "INFO",
                    "P1 fix active: batch_size=8 — OOM condition resolved, continuing training",
                    epoch=epoch, fix="batch_size_reduced",
                )

        # ── Simulate epoch duration ────────────────────────────────────────────

        time.sleep(random.uniform(0.05, 0.15))

        # ── Generate metrics ───────────────────────────────────────────────────

        noise = random.gauss(0, 0.012)

        if failure_mode == "accuracy_drift":
            if apply_fixes:
                # P2+P5: warmup + higher LR → stable convergence, no drift
                warmup_penalty = max(0.0, 0.015 - (epoch - 1) * 0.015)
                accuracy = base_accuracy + (epoch - 1) * (lr_gain_per_epoch * 0.8) - warmup_penalty + noise
            else:
                # Unfixed: genuine regression (-0.025/epoch)
                accuracy = base_accuracy - (epoch - 1) * 0.025 + noise
        else:
            # Logarithmic saturation: gains taper off as accuracy approaches ceiling 0.85
            raw_gain   = (epoch - 1) * lr_gain_per_epoch
            saturation = 1 - (base_accuracy + raw_gain - 0.70) / (0.85 - 0.70)
            saturation = max(0.2, saturation)
            accuracy   = base_accuracy + raw_gain * saturation + noise

        train_loss = base_loss - (epoch - 1) * 0.08 + random.gauss(0, 0.015)
        val_loss   = train_loss + random.gauss(0.04, 0.01)
        f1         = accuracy - random.uniform(0.005, 0.02)
        precision  = f1 + random.gauss(0, 0.01)
        recall     = f1 + random.gauss(0, 0.01)
        auc_roc    = accuracy + random.gauss(0.07, 0.005)

        metrics = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            "accuracy":   round(accuracy, 4),
            "f1":         round(f1, 4),
            "precision":  round(precision, 4),
            "recall":     round(recall, 4),
            "auc_roc":    round(min(auc_roc, 0.99), 4),
        }
        epoch_metrics.append(metrics)

        # ── Accuracy drift check ───────────────────────────────────────────────

        if failure_mode == "accuracy_drift" and epoch >= 2:
            prev_acc = epoch_metrics[-2]["accuracy"]
            drop = prev_acc - accuracy
            if drop > 0.01:
                _emit(
                    events, "WARNING",
                    f"Accuracy drift detected: -{drop:.3f} vs previous epoch",
                    epoch=epoch,
                    accuracy=round(accuracy, 4),
                    previous_accuracy=round(prev_acc, 4),
                    drop=round(drop, 4),
                    error_type="accuracy_drift",
                )
            elif apply_fixes:
                _emit(
                    events, "INFO",
                    "P2 fix active: warmup_steps=500 — accuracy stable, no drift detected",
                    epoch=epoch, fix="warmup_steps_added",
                )

        _emit(events, "INFO", f"Epoch {epoch}/{epochs} completed", **metrics)

    else:
        completed = True
        _emit(events, "INFO", "Run completed successfully", run_id=run_id)

    end_ts = _now()

    record = {
        "run_id":        run_id,
        "failure_mode":  failure_mode,
        "status":        "completed" if completed else "failed",
        "error":         error,
        "started_at":    start_ts,
        "ended_at":      end_ts,
        "config":        config,
        "epoch_metrics": epoch_metrics,
        "events":        events,
    }
    return record


def run_all(n_runs: int = 6, apply_fixes: bool = True) -> list[Path]:
    """Simulate n_runs and write each to a JSON file. Returns list of log paths."""
    modes = [FAILURE_MODES[i % len(FAILURE_MODES)] for i in range(n_runs)]
    random.shuffle(modes)

    paths = []
    for i, mode in enumerate(modes):
        run_id = f"run_{i+1:03d}_{uuid.uuid4().hex[:6]}"
        print(f"[simulator] {run_id}  mode={mode}  apply_fixes={apply_fixes}")
        record = simulate_run(run_id, mode, apply_fixes=apply_fixes)

        log_path = LOGS_DIR / f"{run_id}.json"
        log_path.write_text(json.dumps(record, indent=2))
        paths.append(log_path)
        print(f"           → {log_path.name}  status={record['status']}")

    print(f"\n[simulator] Done – {len(paths)} logs written to {LOGS_DIR}")
    return paths


if __name__ == "__main__":
    run_all()
