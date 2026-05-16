"""
Parses structured JSON logs produced by the simulator.
Returns normalised RunRecord dataclasses ready for health scoring.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    accuracy: float
    f1: float
    precision: float
    recall: float
    auc_roc: float


@dataclass
class RunError:
    type: str          # "oom" | "timeout" | "accuracy_drift"
    epoch: int
    detail: str


@dataclass
class RunRecord:
    run_id: str
    failure_mode: str
    status: str        # "completed" | "failed"
    started_at: str
    ended_at: str
    config: dict
    epoch_metrics: list[EpochMetrics]
    error: Optional[RunError]
    warnings: list[dict] = field(default_factory=list)
    raw_events: list[dict] = field(default_factory=list)

    # ── Derived helpers ───────────────────────────────────────────────────────

    @property
    def final_accuracy(self) -> Optional[float]:
        return self.epoch_metrics[-1].accuracy if self.epoch_metrics else None

    @property
    def accuracy_trend(self) -> Optional[float]:
        """Δaccuracy from first to last epoch (positive = improving)."""
        if len(self.epoch_metrics) < 2:
            return None
        return self.epoch_metrics[-1].accuracy - self.epoch_metrics[0].accuracy

    @property
    def epochs_completed(self) -> int:
        return len(self.epoch_metrics)

    @property
    def expected_epochs(self) -> int:
        return self.config.get("epochs", 3)


def parse_log_file(path: Path) -> RunRecord:
    data = json.loads(path.read_text())

    epoch_metrics = [
        EpochMetrics(**{k: m[k] for k in EpochMetrics.__dataclass_fields__})
        for m in data.get("epoch_metrics", [])
    ]

    raw_error = data.get("error")
    error = RunError(**raw_error) if raw_error else None

    warnings = [
        e for e in data.get("events", [])
        if e.get("level") in ("WARNING", "ERROR", "CRITICAL")
    ]

    return RunRecord(
        run_id=data["run_id"],
        failure_mode=data["failure_mode"],
        status=data["status"],
        started_at=data["started_at"],
        ended_at=data["ended_at"],
        config=data["config"],
        epoch_metrics=epoch_metrics,
        error=error,
        warnings=warnings,
        raw_events=data.get("events", []),
    )


def load_all_runs(logs_dir: Path) -> list[RunRecord]:
    paths = sorted(logs_dir.glob("*.json"))
    records = [parse_log_file(p) for p in paths]
    return sorted(records, key=lambda r: r.started_at)
