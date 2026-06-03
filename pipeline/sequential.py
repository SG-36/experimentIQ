"""
Sequential testing with always-valid p-values (mixture SPRT).

Classical fixed-horizon tests are only valid if you look once, at a
pre-committed sample size. If you peek repeatedly and stop when p<0.05, your
false-positive rate inflates badly. The mixture Sequential Probability Ratio
Test (mSPRT) produces an *always-valid* p-value: you may monitor it after every
observation and stop whenever you like, and the type-I error is still bounded
by alpha (Robbins; Johari, Pekelis & Walsh, "Always Valid Inference").

Test set-up
-----------
H0: the treatment-vs-control difference in means theta = 0.

Let D_n be the running difference in means and V_n its variance
(s_T^2/n_T + s_C^2/n_C). Mixing the alternative over theta ~ N(0, tau^2) gives
the mixture likelihood ratio (a test martingale under H0):

    Lambda_n = sqrt(V_n / (V_n + tau^2))
               * exp( D_n^2 * tau^2 / (2 * V_n * (V_n + tau^2)) )

The always-valid p-value is the running statistic

    p_n = min(1, 1 / max_{k <= n} Lambda_k),

which is non-increasing. Reject H0 the first time Lambda_n >= 1/alpha
(equivalently p_n <= alpha). By Ville's inequality this controls type-I error
at alpha over an unbounded horizon.

Honest caveats
--------------
  * tau (the mixing standard deviation, i.e. the scale of effects you want to
    detect) is a tuning parameter. It affects POWER and *when* you cross, but
    NOT validity: type-I error is controlled for any tau fixed in advance. Power
    is best when tau is near the true effect size. The default sets
    tau^2 = Var(metric); pass tau (or an MDE) for better-targeted power.
  * The mSPRT assumes known variance. We plug in the running sample variance,
    which is the standard practical approximation and is accurate once each arm
    has a moderate number of observations.
  * All computation is done in log space for numerical stability.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_ALPHA = 0.05
MIN_ARM_SIZE = 2


def _log_lambda(d: np.ndarray, v: np.ndarray, tau2: float) -> np.ndarray:
    """Log of the mSPRT mixture likelihood ratio (elementwise)."""
    return 0.5 * np.log(v / (v + tau2)) + (d**2 * tau2) / (2.0 * v * (v + tau2))


def run_sequential_test(
    df: pd.DataFrame,
    metric_col: str = "metric",
    alpha: float = DEFAULT_ALPHA,
    tau: float | None = None,
    target_sample_size: int | None = None,
) -> dict[str, Any]:
    """
    Run an mSPRT over the experiment in timestamp order.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned experiment data with ``treatment``, ``timestamp`` and the
        metric column. Pass ``metric_col="metric_cuped"`` to run on the
        CUPED-adjusted metric.
    metric_col : str
        Which column to test (default ``"metric"``).
    alpha : float
        Significance level; the rejection boundary is ``1/alpha``.
    tau : float, optional
        Mixing standard deviation (scale of effects to detect). Defaults to
        ``sqrt(Var(metric))``. Affects power, not validity.
    target_sample_size : int, optional
        A planned total sample size to annotate (the fixed-horizon analysis
        point). Does not affect the test; used for reporting/plotting.

    Returns
    -------
    dict
        Time series (``sample_sizes``, ``effect``, ``log_lambda``,
        ``p_values``) plus a summary (crossing point, final p-value, decision).
    """
    work = df[["treatment", "timestamp", metric_col]].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna(subset=["treatment", "timestamp", metric_col])
    work = work.sort_values("timestamp", kind="stable").reset_index(drop=True)

    n = len(work)
    metric = work[metric_col].to_numpy(dtype=float)
    is_t = (work["treatment"].to_numpy() == 1).astype(float)
    is_c = (work["treatment"].to_numpy() == 0).astype(float)

    if tau is None:
        var_all = float(np.var(metric, ddof=1)) if n > 1 else 0.0
        tau2 = var_all if var_all > 0 else 1.0
    else:
        tau2 = float(tau) ** 2

    # Running per-arm count, sum, sum-of-squares via cumulative sums.
    n_t = np.cumsum(is_t)
    n_c = np.cumsum(is_c)
    sum_t = np.cumsum(metric * is_t)
    sum_c = np.cumsum(metric * is_c)
    sumsq_t = np.cumsum(metric**2 * is_t)
    sumsq_c = np.cumsum(metric**2 * is_c)

    valid = (n_t >= MIN_ARM_SIZE) & (n_c >= MIN_ARM_SIZE)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_t = sum_t / n_t
        mean_c = sum_c / n_c
        var_t = (sumsq_t - n_t * mean_t**2) / np.maximum(n_t - 1, 1)
        var_c = (sumsq_c - n_c * mean_c**2) / np.maximum(n_c - 1, 1)
        v_n = var_t / n_t + var_c / n_c
        d_n = mean_t - mean_c

    valid = valid & np.isfinite(v_n) & (v_n > 0)

    idx = np.where(valid)[0]
    if idx.size == 0:
        return {
            "status": "skip",
            "alpha": alpha,
            "tau": float(np.sqrt(tau2)),
            "n_total": n,
            "sample_sizes": [],
            "effect": [],
            "log_lambda": [],
            "p_values": [],
            "crossed": False,
            "explanation": "Not enough data in both arms to run a sequential test.",
            "recommendation": "Collect at least a few observations per arm.",
        }

    sample_sizes = (idx + 1).astype(int)
    d_valid = d_n[idx]
    v_valid = v_n[idx]
    log_lambda = _log_lambda(d_valid, v_valid, tau2)

    # Always-valid p-value: running max of log-lambda -> non-increasing p.
    running_max = np.maximum.accumulate(log_lambda)
    p_values = np.minimum(1.0, np.exp(-running_max))

    boundary = np.log(1.0 / alpha)
    crossed_mask = log_lambda >= boundary
    crossed = bool(crossed_mask.any())

    summary: dict[str, Any] = {
        "status": "ok",
        "alpha": alpha,
        "tau": float(np.sqrt(tau2)),
        "metric_col": metric_col,
        "n_total": int(n),
        "n_treatment": int(n_t[-1]),
        "n_control": int(n_c[-1]),
        "sample_sizes": sample_sizes.tolist(),
        "effect": d_valid.tolist(),
        "log_lambda": log_lambda.tolist(),
        "p_values": p_values.tolist(),
        "final_effect": float(d_valid[-1]),
        "final_p_value": float(p_values[-1]),
        "crossed": crossed,
        "target_sample_size": target_sample_size,
    }

    if crossed:
        first = int(np.argmax(crossed_mask))
        summary["crossing_sample_size"] = int(sample_sizes[first])
        summary["crossing_effect"] = float(d_valid[first])
        summary["crossing_p_value"] = float(p_values[first])
        summary["decision"] = "reject_h0"
        explanation = (
            f"The always-valid p-value crossed alpha={alpha} at "
            f"{sample_sizes[first]} cumulative observations, where the estimated "
            f"effect was {d_valid[first]:+.4f}. Because this is an mSPRT, you may "
            "stop here and report a significant effect without inflating the "
            f"false-positive rate. Final estimate: {d_valid[-1]:+.4f} "
            f"(always-valid p={p_values[-1]:.4f})."
        )
        recommendation = (
            "You can stop the experiment and declare a statistically significant "
            "result; confirm the effect direction/size is practically meaningful."
        )
    else:
        summary["decision"] = "no_decision"
        explanation = (
            f"The always-valid p-value never crossed alpha={alpha} "
            f"(final p={p_values[-1]:.4f}, final effect {d_valid[-1]:+.4f}). With "
            "an always-valid test this is NOT evidence of no effect; you simply "
            "have not accumulated enough signal yet."
        )
        recommendation = (
            "Keep collecting data (the test stays valid as it grows), or stop for "
            "futility if the plausible effect is too small to matter."
        )

    if target_sample_size is not None:
        at_target = sample_sizes >= target_sample_size
        if at_target.any():
            ti = int(np.argmax(at_target))
            summary["decision_at_target"] = {
                "sample_size": int(sample_sizes[ti]),
                "effect": float(d_valid[ti]),
                "p_value": float(p_values[ti]),
                "significant": bool(p_values[ti] <= alpha),
            }

    summary["explanation"] = explanation
    summary["recommendation"] = recommendation
    return summary


def plot_sequential_test(result: dict[str, Any]):
    """
    Plot the mSPRT statistic (log Lambda) over cumulative sample size with the
    rejection boundary. Returns a Plotly Figure. Plotly is imported lazily.
    """
    import plotly.graph_objects as go

    if result.get("status") != "ok":
        return go.Figure(layout=go.Layout(title="Sequential test: insufficient data"))

    x = result["sample_sizes"]
    log_lambda = result["log_lambda"]
    boundary = float(np.log(1.0 / result["alpha"]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=log_lambda, mode="lines", name="log Λ (mSPRT)"))
    fig.add_hline(
        y=boundary,
        line_dash="dash",
        line_color="red",
        annotation_text=f"reject boundary (α={result['alpha']})",
    )
    if result.get("crossed"):
        cs = result["crossing_sample_size"]
        fig.add_vline(x=cs, line_dash="dot", line_color="green",
                      annotation_text=f"cross @ n={cs}")
    if result.get("target_sample_size"):
        fig.add_vline(x=result["target_sample_size"], line_dash="dot",
                      line_color="gray", annotation_text="target n")

    fig.update_layout(
        title="Sequential test statistic over time",
        xaxis_title="cumulative sample size",
        yaxis_title="log Λ (mixture likelihood ratio)",
    )
    return fig
