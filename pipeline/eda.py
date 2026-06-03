"""
Exploratory data analysis for a cleaned experiment DataFrame.

Produces a structured, JSON-serializable EDA report:
  * treatment vs control group sizes,
  * per-arm metric distribution summaries,
  * assignment timeline over time,
  * a list of flagged "obvious" anomalies.

The numeric computation is kept separate from plotting so the statistics are
unit-testable without a rendering backend. Plot helpers return Plotly figures
and import Plotly lazily.

IMPORTANT: the anomaly flags here are heuristic descriptive checks meant to
catch obviously broken data (empty arm, degenerate metric, etc.). They are NOT
formal statistical tests. The formal assumption tests (SRM, SUTVA, novelty)
live in the assumption-testing stage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# Heuristic thresholds for anomaly flags. These are descriptive, not inferential.
NEAR_EMPTY_ARM_PROP = 0.01
POINT_MASS_FRAC = 0.5
EXTREME_OUTLIER_IQR_MULT = 3.0
HIGH_SKEW = 2.0
TIMELINE_GAP_FRAC = 0.25


def summarize_group_sizes(df: pd.DataFrame) -> dict[str, Any]:
    """Counts and proportions for the control (0) and treatment (1) arms."""
    n = len(df)
    n_control = int((df["treatment"] == 0).sum())
    n_treatment = int((df["treatment"] == 1).sum())
    return {
        "n_total": n,
        "n_control": n_control,
        "n_treatment": n_treatment,
        "prop_control": n_control / n if n else 0.0,
        "prop_treatment": n_treatment / n if n else 0.0,
    }


def _describe(series: pd.Series) -> dict[str, Any]:
    """Summary statistics for a numeric series (NaNs dropped)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {k: None for k in
                ("n", "mean", "std", "min", "p25", "median", "p75", "max", "skew")}
    std = float(s.std(ddof=1)) if s.size > 1 else 0.0
    # Skew is undefined for zero-variance data; skip to avoid precision warnings.
    skew = float(stats.skew(s)) if (s.size > 2 and std > 0) else 0.0
    return {
        "n": int(s.size),
        "mean": float(s.mean()),
        "std": std,
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
        "skew": skew,
    }


def summarize_metric(df: pd.DataFrame) -> dict[str, Any]:
    """Per-arm and overall metric distribution summaries."""
    return {
        "overall": _describe(df["metric"]),
        "control": _describe(df.loc[df["treatment"] == 0, "metric"]),
        "treatment": _describe(df.loc[df["treatment"] == 1, "metric"]),
    }


def summarize_timeline(df: pd.DataFrame) -> dict[str, Any]:
    """Assignment counts per calendar day, per arm."""
    ts = pd.to_datetime(df["timestamp"])
    day = ts.dt.normalize()
    grouped = (
        pd.DataFrame({"day": day, "treatment": df["treatment"].to_numpy()})
        .groupby(["day", "treatment"])
        .size()
        .unstack(fill_value=0)
    )
    per_day: dict[str, dict[str, int]] = {}
    for d, row in grouped.iterrows():
        per_day[str(pd.Timestamp(d).date())] = {
            "control": int(row.get(0, 0)),
            "treatment": int(row.get(1, 0)),
        }

    day_span = int((day.max() - day.min()).days) + 1 if len(day) else 0
    return {
        "start": str(ts.min()),
        "end": str(ts.max()),
        "n_active_days": int(day.nunique()),
        "day_span": day_span,
        "assignments_per_day": per_day,
    }


def flag_anomalies(
    df: pd.DataFrame,
    group_sizes: dict[str, Any],
    metric_summary: dict[str, Any],
    timeline: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Flag obviously broken data. Heuristic and descriptive (not statistical tests).

    Each anomaly is ``{"code", "severity", "message"}`` with severity in
    {"high", "medium", "low"}.
    """
    anomalies: list[dict[str, Any]] = []
    n = group_sizes["n_total"]

    if group_sizes["n_control"] == 0 or group_sizes["n_treatment"] == 0:
        anomalies.append({
            "code": "empty_arm",
            "severity": "high",
            "message": (
                f"One arm is empty (control={group_sizes['n_control']}, "
                f"treatment={group_sizes['n_treatment']}). No comparison is possible."
            ),
        })
    elif n and min(group_sizes["prop_control"], group_sizes["prop_treatment"]) < NEAR_EMPTY_ARM_PROP:
        anomalies.append({
            "code": "near_empty_arm",
            "severity": "high",
            "message": (
                f"One arm holds <{NEAR_EMPTY_ARM_PROP:.0%} of users "
                f"(control={group_sizes['prop_control']:.1%}, "
                f"treatment={group_sizes['prop_treatment']:.1%})."
            ),
        })

    for arm in ("overall", "control", "treatment"):
        d = metric_summary[arm]
        if d["n"] and d["n"] > 1 and d["std"] == 0:
            anomalies.append({
                "code": "constant_metric",
                "severity": "high",
                "message": f"Metric is constant within the {arm} group (zero variance).",
            })

    metric = pd.to_numeric(df["metric"], errors="coerce").dropna()
    if not metric.empty:
        vc = metric.value_counts(normalize=True)
        top_frac = float(vc.iloc[0])
        if top_frac > POINT_MASS_FRAC:
            anomalies.append({
                "code": "point_mass",
                "severity": "medium",
                "message": (
                    f"{top_frac:.0%} of metric values equal {vc.index[0]:g}. "
                    "If this is a binary/conversion metric this may be expected."
                ),
            })

        q1, q3 = metric.quantile(0.25), metric.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lo = q1 - EXTREME_OUTLIER_IQR_MULT * iqr
            hi = q3 + EXTREME_OUTLIER_IQR_MULT * iqr
            n_out = int(((metric < lo) | (metric > hi)).sum())
            if n_out > 0:
                anomalies.append({
                    "code": "extreme_outliers",
                    "severity": "low",
                    "message": (
                        f"{n_out} metric value(s) lie beyond "
                        f"{EXTREME_OUTLIER_IQR_MULT:g}\u00d7IQR fences "
                        f"([{lo:.3g}, {hi:.3g}]). Inspect before modeling."
                    ),
                })

        overall_skew = metric_summary["overall"]["skew"]
        if overall_skew is not None and abs(overall_skew) > HIGH_SKEW:
            anomalies.append({
                "code": "high_skew",
                "severity": "low",
                "message": (
                    f"Metric is highly skewed (skew={overall_skew:.2f}). "
                    "Mean-based comparisons may be sensitive to the tail."
                ),
            })

    span = timeline["day_span"]
    active = timeline["n_active_days"]
    if span > 1 and active > 0:
        empty_frac = (span - active) / span
        if empty_frac > TIMELINE_GAP_FRAC:
            anomalies.append({
                "code": "timeline_gaps",
                "severity": "medium",
                "message": (
                    f"{span - active} of {span} days in the window had zero "
                    f"assignments ({empty_frac:.0%}). Check for logging gaps or "
                    "ramp/pause periods."
                ),
            })

    return anomalies


def run_eda(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run the full EDA pass on a cleaned experiment DataFrame.

    Expects the output of ``pipeline.ingest.run_ingest`` (treatment is 0/1,
    one row per user). Returns a JSON-serializable report.
    """
    group_sizes = summarize_group_sizes(df)
    metric_summary = summarize_metric(df)
    timeline = summarize_timeline(df)
    anomalies = flag_anomalies(df, group_sizes, metric_summary, timeline)
    return {
        "group_sizes": group_sizes,
        "metric": metric_summary,
        "timeline": timeline,
        "anomalies": anomalies,
        "n_anomalies": len(anomalies),
    }


# --------------------------------------------------------------------------- #
# Plotting helpers. Plotly is imported lazily so the report above can be       #
# computed and tested without a rendering backend installed.                   #
# --------------------------------------------------------------------------- #

_ARM_LABEL = {0: "control", 1: "treatment"}


def plot_group_sizes(df: pd.DataFrame):
    """Bar chart of control vs treatment group sizes. Returns a Plotly Figure."""
    import plotly.graph_objects as go

    g = summarize_group_sizes(df)
    return go.Figure(
        data=[go.Bar(
            x=["control", "treatment"],
            y=[g["n_control"], g["n_treatment"]],
            text=[g["n_control"], g["n_treatment"]],
            textposition="auto",
        )],
        layout=go.Layout(
            title="Group sizes", xaxis_title="arm", yaxis_title="users"
        ),
    )


def plot_metric_histogram(df: pd.DataFrame, nbins: int = 50):
    """Overlaid metric histogram for each arm. Returns a Plotly Figure."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for arm, label in _ARM_LABEL.items():
        vals = df.loc[df["treatment"] == arm, "metric"].dropna()
        fig.add_trace(go.Histogram(x=vals, name=label, nbinsx=nbins, opacity=0.6))
    fig.update_layout(
        barmode="overlay",
        title="Metric distribution by arm",
        xaxis_title="metric",
        yaxis_title="count",
    )
    return fig


def plot_metric_box(df: pd.DataFrame):
    """Box plot of the metric per arm. Returns a Plotly Figure."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for arm, label in _ARM_LABEL.items():
        vals = df.loc[df["treatment"] == arm, "metric"].dropna()
        fig.add_trace(go.Box(y=vals, name=label, boxpoints="outliers"))
    fig.update_layout(title="Metric box plot by arm", yaxis_title="metric")
    return fig


def plot_timeline(df: pd.DataFrame):
    """Assignments per day per arm. Returns a Plotly Figure."""
    import plotly.graph_objects as go

    ts = pd.to_datetime(df["timestamp"]).dt.normalize()
    frame = pd.DataFrame({"day": ts, "treatment": df["treatment"].to_numpy()})
    fig = go.Figure()
    for arm, label in _ARM_LABEL.items():
        counts = frame.loc[frame["treatment"] == arm].groupby("day").size()
        fig.add_trace(go.Scatter(
            x=counts.index, y=counts.to_numpy(), mode="lines+markers", name=label
        ))
    fig.update_layout(
        title="Assignments over time",
        xaxis_title="day",
        yaxis_title="assignments",
    )
    return fig


def save_eda_plots(df: pd.DataFrame, outdir: str) -> list[str]:
    """Write all EDA plots to HTML files in ``outdir``. Returns the file paths."""
    import os

    os.makedirs(outdir, exist_ok=True)
    figs = {
        "group_sizes": plot_group_sizes(df),
        "metric_histogram": plot_metric_histogram(df),
        "metric_box": plot_metric_box(df),
        "timeline": plot_timeline(df),
    }
    paths = []
    for name, fig in figs.items():
        path = os.path.join(outdir, f"{name}.html")
        fig.write_html(path)
        paths.append(path)
    return paths
