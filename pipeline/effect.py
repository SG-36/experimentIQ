"""
Headline treatment-effect estimation (fixed-horizon inference).

This is the number a published A/B analysis actually reports: the estimated
difference between treatment and control, with a confidence interval and a
p-value. The other modules diagnose validity (assumptions), reduce variance
(CUPED), or monitor continuously (sequential mSPRT); this one produces the
point estimate + uncertainty you compare against an expected/published result.

It picks the right test for the metric type:

  * Binary / conversion metrics (values in {0, 1}) -> two-proportion z-test.
    Reports absolute lift (percentage points) and relative lift (%). The
    p-value uses the pooled-variance SE (standard for testing H0: p_t = p_c);
    the confidence interval uses the unpooled SE.
  * Continuous metrics -> Welch's t-test (unequal variances, Welch-Satterthwaite
    degrees of freedom). Does NOT assume equal variances across arms.

To get a tighter interval from a variance-reduced metric, pass
``metric_col="metric_cuped"`` (after running CUPED). Note: treating the CUPED
output as raw data slightly understates variance because theta is estimated,
but the effect is negligible at the sample sizes where CUPED is used.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_ALPHA = 0.05
MIN_ARM_SIZE = 2


def _is_binary(values: np.ndarray) -> bool:
    """True if every (non-null) value is 0 or 1."""
    uniq = set(np.unique(values).tolist())
    return uniq.issubset({0.0, 1.0})


def _continuous_effect(
    treatment: np.ndarray, control: np.ndarray, alpha: float
) -> dict[str, Any]:
    """Welch's t-test for a difference in means with a confidence interval."""
    n_t, n_c = treatment.size, control.size
    mean_t, mean_c = float(treatment.mean()), float(control.mean())
    var_t = float(treatment.var(ddof=1))
    var_c = float(control.var(ddof=1))
    effect = mean_t - mean_c

    se = float(np.sqrt(var_t / n_t + var_c / n_c))
    if se == 0:
        return {
            "method": "welch_t",
            "effect": effect, "std_error": 0.0, "test_statistic": float("nan"),
            "dof": float("nan"), "p_value": float("nan"),
            "ci_low": effect, "ci_high": effect,
        }
    # Welch-Satterthwaite degrees of freedom.
    dof = (var_t / n_t + var_c / n_c) ** 2 / (
        (var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1)
    )
    t_stat = effect / se
    p_value = float(2 * stats.t.sf(abs(t_stat), dof))
    t_crit = float(stats.t.ppf(1 - alpha / 2, dof))
    return {
        "method": "welch_t",
        "effect": effect,
        "std_error": se,
        "test_statistic": float(t_stat),
        "dof": float(dof),
        "p_value": p_value,
        "ci_low": effect - t_crit * se,
        "ci_high": effect + t_crit * se,
    }


def _proportion_effect(
    treatment: np.ndarray, control: np.ndarray, alpha: float
) -> dict[str, Any]:
    """Two-proportion z-test (pooled SE for the p-value, unpooled for the CI)."""
    n_t, n_c = treatment.size, control.size
    x_t, x_c = float(treatment.sum()), float(control.sum())
    p_t, p_c = x_t / n_t, x_c / n_c
    effect = p_t - p_c

    p_pool = (x_t + x_c) / (n_t + n_c)
    se_pool = float(np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c)))
    se_unpool = float(np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c))

    if se_pool == 0:
        z, p_value = float("nan"), float("nan")
    else:
        z = effect / se_pool
        p_value = float(2 * stats.norm.sf(abs(z)))
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    return {
        "method": "two_proportion_z",
        "effect": effect,
        "std_error": se_unpool,
        "test_statistic": z if se_pool else float("nan"),
        "dof": None,
        "p_value": p_value,
        "ci_low": effect - z_crit * se_unpool,
        "ci_high": effect + z_crit * se_unpool,
        "rate_control": p_c,
        "rate_treatment": p_t,
    }


def estimate_effect(
    df: pd.DataFrame,
    metric_col: str = "metric",
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """
    Estimate the treatment effect with a confidence interval and p-value.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned experiment data with ``treatment`` (0/1) and the metric column.
    metric_col : str
        Column to analyze (e.g. ``"metric"`` or ``"metric_cuped"``).
    alpha : float
        Significance level; the CI is a ``(1 - alpha)`` interval.

    Returns
    -------
    dict
        Point estimate, std error, test statistic, p-value, confidence
        interval, per-arm summaries, relative lift, and a plain-English readout.
    """
    treatment = pd.to_numeric(
        df.loc[df["treatment"] == 1, metric_col], errors="coerce"
    ).dropna().to_numpy(dtype=float)
    control = pd.to_numeric(
        df.loc[df["treatment"] == 0, metric_col], errors="coerce"
    ).dropna().to_numpy(dtype=float)

    n_t, n_c = treatment.size, control.size
    if n_t < MIN_ARM_SIZE or n_c < MIN_ARM_SIZE:
        return {
            "status": "skip",
            "metric_col": metric_col,
            "n_treatment": int(n_t),
            "n_control": int(n_c),
            "explanation": "Both arms need at least two observations to estimate an effect.",
            "recommendation": "Collect more data in the under-filled arm.",
        }

    both = np.concatenate([treatment, control])
    binary = _is_binary(both)
    core = (
        _proportion_effect(treatment, control, alpha)
        if binary
        else _continuous_effect(treatment, control, alpha)
    )

    mean_c = float(control.mean())
    mean_t = float(treatment.mean())
    rel_lift = (core["effect"] / mean_c) if mean_c not in (0.0,) else float("nan")
    # Relative-lift CI by scaling the absolute CI (valid when control mean > 0).
    if mean_c > 0:
        rel_ci_low = core["ci_low"] / mean_c
        rel_ci_high = core["ci_high"] / mean_c
    else:
        rel_ci_low = rel_ci_high = float("nan")

    significant = bool(core["p_value"] < alpha) if np.isfinite(core["p_value"]) else False
    ci_pct = int(round((1 - alpha) * 100))

    metric_kind = "conversion rate" if binary else "mean"
    effect_str = (
        f"{core['effect'] * 100:+.2f} pp" if binary else f"{core['effect']:+.4f}"
    )
    explanation = (
        f"Estimated treatment effect on the {metric_kind}: {effect_str} "
        f"({ci_pct}% CI [{core['ci_low']:+.4f}, {core['ci_high']:+.4f}], "
        f"p={core['p_value']:.4f}, {core['method']}). "
        f"Control {metric_kind}={mean_c:.4f}, treatment={mean_t:.4f}"
    )
    if np.isfinite(rel_lift):
        explanation += f"; relative lift {rel_lift * 100:+.2f}%."
    else:
        explanation += "."

    if significant:
        recommendation = (
            f"The effect is statistically significant at alpha={alpha}. Confirm "
            "the magnitude is practically meaningful and that the assumption "
            "checks (SRM/SUTVA/novelty) passed before acting."
        )
    else:
        recommendation = (
            f"The effect is not statistically significant at alpha={alpha}; the "
            f"{ci_pct}% CI includes zero. This is not proof of no effect - check "
            "whether the experiment was powered for the effect size you care about."
        )

    return {
        "status": "ok",
        "metric_col": metric_col,
        "metric_type": "binary" if binary else "continuous",
        "method": core["method"],
        "alpha": alpha,
        "n_treatment": int(n_t),
        "n_control": int(n_c),
        "mean_control": mean_c,
        "mean_treatment": mean_t,
        "effect": core["effect"],
        "std_error": core["std_error"],
        "test_statistic": core["test_statistic"],
        "dof": core["dof"],
        "p_value": core["p_value"],
        "ci_low": core["ci_low"],
        "ci_high": core["ci_high"],
        "relative_lift": rel_lift,
        "relative_ci_low": rel_ci_low,
        "relative_ci_high": rel_ci_high,
        "significant": significant,
        "explanation": explanation,
        "recommendation": recommendation,
    }
