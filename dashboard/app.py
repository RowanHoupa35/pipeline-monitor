"""
Streamlit dashboard for the LLM-Powered Pipeline Monitor.

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Make sure agent package is importable when running from the dashboard dir
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from agent import load_all_runs, score_all, generate_alert_stream
from simulator.run_pipeline import run_all

LOGS_DIR = Path(__file__).parent.parent / "logs"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Pipeline Monitor",
    page_icon="🔬",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Pipeline Monitor")
    st.markdown("**DistilBERT · Fake News Detection**")
    st.divider()

    n_runs = st.slider("Runs to simulate", 3, 12, 6)
    apply_fixes = st.checkbox("Apply v2 fixes (all runs succeed)", value=True)
    if st.button("▶ Run simulator", use_container_width=True):
        with st.spinner("Simulating training runs…"):
            run_all(n_runs, apply_fixes=apply_fixes)
        st.session_state.pop("alert_text", None)  # invalidate stale LLM alert
        st.success("Simulation complete!")
        st.rerun()

    st.divider()
    if st.button("🤖 Generate LLM alert", use_container_width=True):
        st.session_state["generate_alert"] = True

    st.divider()
    st.caption("Stack: DistilBERT · Ollama · qwen3:8b · Docker · Streamlit")

# ── Load data ──────────────────────────────────────────────────────────────────
log_files = list(LOGS_DIR.glob("*.json"))

if not log_files:
    st.info("No logs found. Use the sidebar to run the simulator first.")
    st.stop()

records = load_all_runs(LOGS_DIR)
reports = score_all(records)

# ── KPI row ────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
scores = [r.score for r in reports]
statuses = [r.status for r in reports]

col1.metric("Total runs", len(records))
col2.metric(
    "Avg health score",
    f"{sum(scores)/len(scores):.1f}",
    delta=None,
)
col3.metric("Healthy", statuses.count("healthy"), delta=None)
col4.metric(
    "Critical",
    statuses.count("critical"),
    delta=None,
    delta_color="inverse",
)

st.divider()

# ── Run table ──────────────────────────────────────────────────────────────────
st.subheader("Run Summary")

table_rows = []
for rec, rep in zip(records, reports):
    status_icon = {"healthy": "✅", "degraded": "⚠️", "critical": "❌"}.get(rep.status, "?")
    last = rec.epoch_metrics[-1] if rec.epoch_metrics else None
    table_rows.append({
        "Run ID": rec.run_id,
        "Status": f"{status_icon} {rep.status}",
        "Score": rep.score,
        "Failure mode": rec.failure_mode,
        "Epochs": f"{rec.epochs_completed}/{rec.expected_epochs}",
        "Accuracy": f"{rec.final_accuracy:.4f}" if rec.final_accuracy else "–",
        "F1": f"{last.f1:.4f}" if last else "–",
        "Val loss": f"{last.val_loss:.4f}" if last else "–",
        "Trend": f"{rec.accuracy_trend:+.4f}" if rec.accuracy_trend is not None else "–",
        "Alerts": len(rep.alerts),
    })

df = pd.DataFrame(table_rows)
st.dataframe(
    df.style.background_gradient(subset=["Score"], cmap="RdYlGn", vmin=0, vmax=100),
    width="stretch",
    hide_index=True,
)

# ── Charts ─────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

# Health scores bar chart
with col_left:
    st.subheader("Health Scores")
    fig_scores = px.bar(
        df,
        x="Run ID",
        y="Score",
        color="Score",
        color_continuous_scale="RdYlGn",
        range_color=[0, 100],
        text="Score",
    )
    fig_scores.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(t=20, b=40),
        height=300,
    )
    fig_scores.add_hline(y=70, line_dash="dot", line_color="green", annotation_text="Healthy threshold")
    fig_scores.add_hline(y=40, line_dash="dot", line_color="red", annotation_text="Critical threshold")
    st.plotly_chart(fig_scores, width="stretch")

# Accuracy curves per run
with col_right:
    st.subheader("Accuracy per Epoch")
    fig_acc = go.Figure()
    palette = px.colors.qualitative.Plotly

    for i, rec in enumerate(records):
        if not rec.epoch_metrics:
            continue
        xs = [m.epoch for m in rec.epoch_metrics]
        ys = [m.accuracy for m in rec.epoch_metrics]
        fig_acc.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=f"{rec.run_id} ({rec.failure_mode})",
                line=dict(color=palette[i % len(palette)]),
            )
        )

    fig_acc.add_hline(
        y=0.70, line_dash="dot", line_color="gray", annotation_text="Baseline 0.70"
    )
    fig_acc.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Validation Accuracy",
        margin=dict(t=20, b=40),
        height=300,
        legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig_acc, width="stretch")

# ── Per-run detail ─────────────────────────────────────────────────────────────
st.subheader("Run Details & Alerts")
for rec, rep in zip(records, reports):
    icon = {"healthy": "✅", "degraded": "⚠️", "critical": "❌"}.get(rep.status, "?")
    with st.expander(f"{icon} {rec.run_id} — score {rep.score}/100"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Completion", f"{rec.epochs_completed}/{rec.expected_epochs}")
        c2.metric("Score", f"{rep.score}/100")
        c3.metric("Mode", rec.failure_mode)

        # Score breakdown per dimension
        st.markdown("**Score breakdown:**")
        bd = rep.breakdown
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.metric("Completion", f"{bd.get('completion', 0)}/25")
        b2.metric("Accuracy",   f"{bd.get('accuracy',   0)}/20")
        b3.metric("F1",         f"{bd.get('f1',         0)}/10")
        b4.metric("Trend",      f"{bd.get('trend',      0)}/15")
        b5.metric("Val loss",   f"{bd.get('val_loss',   0)}/15")
        b6.metric("Errors",     f"{bd.get('errors',     0)}/15")

        if rep.alerts:
            st.markdown("**Detected alerts:**")
            for alert in rep.alerts:
                st.warning(alert)

        if rec.epoch_metrics:
            epoch_df = pd.DataFrame(
                [
                    {
                        "Epoch": m.epoch,
                        "Train loss": m.train_loss,
                        "Val loss": m.val_loss,
                        "Accuracy": m.accuracy,
                        "F1": m.f1,
                        "AUC-ROC": m.auc_roc,
                    }
                    for m in rec.epoch_metrics
                ]
            )
            st.dataframe(epoch_df, hide_index=True, width="stretch")

# ── LLM Alert section ──────────────────────────────────────────────────────────
st.divider()
st.subheader("🤖 LLM-Generated Alert")

if st.session_state.get("generate_alert"):
    st.session_state["generate_alert"] = False
    alert_text = st.write_stream(generate_alert_stream(records, reports))
    st.session_state["alert_text"] = alert_text
elif "alert_text" in st.session_state:
    st.markdown(st.session_state["alert_text"])
else:
    st.info("Click **Generate LLM alert** in the sidebar to get an AI-powered anomaly summary.")
