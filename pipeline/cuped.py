"""
CUPED variance reduction.

CUPED (Controlled-experiment Using Pre-Experiment Data) reduces the variance of
the outcome metric by partialling out variance that is explained by
pre-experiment covariates, WITHOUT changing the expected treatment effect.

Method
------
Given pre-experiment covariates X and outcome Y, fit the OLS coefficients
``theta`` of (centered) Y on (centered) X, then define

    Y_cuped = Y - (X - mean(X)) @ theta

Because the covariates are mean-centered, ``E[Y_cuped] == E[Y]``, so the
treatment-vs-control difference (the ATE) is preserved in expectation. When X
is balanced across arms (true for genuinely pre-experiment covariates under
randomization) the point estimate of the ATE is essentially unchanged, while
the residual variance drops by the fraction of outcome variance X explains.

theta is estimated on the POOLED sample (both arms together). This is the
standard CUPED choice and keeps the ATE unbiased as long as X is unaffected by
treatment.

Honest caveats
--------------
  * CUPED is only valid for PRE-experiment covariates (measured before
    assignment / unaffected by treatment). This module cannot verify that; it
    assumes the caller supplied valid covariates. As a guardrail it reports
    each covariate's standardized mean difference across arms so gross
    imbalance (a sign the covariate is post-treatment or randomization broke)
    is visible.
  * Variance reduction is reported in-sample and equals the regression R^2.
  * No imputation is performed. Rows missing any selected covariate cannot be
    adjusted; their ``metric_cuped`` falls back to the original ``metric`` and
    the count is reported.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["user_id", "treatment", "timestamp", "metric"]


def select_cuped_covariates(
    df: pd.DataFrame,
    covariates: list[str] | None = None,
) -> list[str]:
    """
    Choose usable CUPED covariates.

    If ``covariates`` is given, keep those that exist and are numeric.
    Otherwise default to every non-required, numeric, non-constant column.
    """
    if covariates is None:
        candidates = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    else:
        candidates = [c for c in covariates if c in df.columns]

    usable: list[str] = []
    for c in candidates:
        col = pd.to_numeric(df[c], errors="coerce")
        non_null = col.dropna()
        if non_null.empty:
            continue
        if non_null.nunique() <= 1:  # constant column carries no information
            continue
        usable.append(c)
    return usable


def _standardized_mean_diff(df: pd.DataFrame, col: str) -> float:
    """Standardized mean difference of a covariate across arms (balance check)."""
    t = pd.to_numeric(df.loc[df["treatment"] == 1, col], errors="coerce").dropna()
    c = pd.to_numeric(df.loc[df["treatment"] == 0, col], errors="coerce").dropna()
    if t.size < 2 or c.size < 2:
        return float("nan")
    pooled_sd = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
    if pooled_sd == 0:
        return 0.0
    return float((t.mean() - c.mean()) / pooled_sd)


def _ate(metric: pd.Series, treatment: pd.Series) -> float:
    """Difference in mean metric (treatment - control)."""
    t = metric[treatment == 1]
    c = metric[treatment == 0]
    if t.empty or c.empty:
        return float("nan")
    return float(t.mean() - c.mean())


def run_cuped(
    df: pd.DataFrame,
    covariates: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply CUPED variance reduction to a cleaned experiment DataFrame.

    Adds a ``metric_cuped`` column (the original ``metric`` is left untouched)
    and returns a report describing the covariates used, the fitted theta, the
    variance reduction achieved, and confirmation that the ATE is preserved.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned experiment data (output of ``pipeline.ingest.run_ingest``).
    covariates : list[str], optional
        Covariate columns to use. Defaults to all numeric, non-constant,
        non-required columns.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        The DataFrame with an added ``metric_cuped`` column and the CUPED
        report. If no usable covariates exist the metric is copied through
        unchanged and ``status == "skipped"``.
    """
    out = df.copy()
    out["metric"] = pd.to_numeric(out["metric"], errors="coerce")

    used = select_cuped_covariates(out, covariates)
    var_original = float(out["metric"].var(ddof=1)) if len(out) > 1 else 0.0
    ate_original = _ate(out["metric"], out["treatment"])

    if not used:
        out["metric_cuped"] = out["metric"]
        return out, {
            "status": "skipped",
            "covariates_used": [],
            "reason": "No usable pre-experiment covariates were found.",
            "variance_original": var_original,
            "variance_cuped": var_original,
            "variance_reduction": 0.0,
            "variance_reduction_pct": 0.0,
            "ate_original": ate_original,
            "ate_cuped": ate_original,
            "n_adjusted": 0,
            "n_unadjusted": int(len(out)),
            "explanation": (
                "CUPED was skipped because no pre-experiment covariates are "
                "available to explain outcome variance. The metric is passed "
                "through unchanged."
            ),
            "recommendation": (
                "If you have pre-experiment measurements (e.g. the same metric "
                "from before the test, prior activity), include them as columns "
                "to gain precision."
            ),
        }

    # Complete-case mask: every selected covariate present and metric present.
    cov_numeric = out[used].apply(pd.to_numeric, errors="coerce")
    complete_mask = cov_numeric.notna().all(axis=1) & out["metric"].notna()
    n_adjusted = int(complete_mask.sum())
    n_unadjusted = int((~complete_mask).sum())

    out["metric_cuped"] = out["metric"]

    balance = {c: _standardized_mean_diff(out, c) for c in used}

    theta: dict[str, float] = {c: 0.0 for c in used}
    if n_adjusted >= len(used) + 2:
        X = cov_numeric.loc[complete_mask].to_numpy(dtype=float)
        y = out.loc[complete_mask, "metric"].to_numpy(dtype=float)
        x_mean = X.mean(axis=0)
        Xc = X - x_mean
        yc = y - y.mean()
        coef, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
        theta = {c: float(b) for c, b in zip(used, coef)}
        adjustment = Xc @ coef
        out.loc[complete_mask, "metric_cuped"] = y - adjustment

    var_cuped = float(out["metric_cuped"].var(ddof=1)) if len(out) > 1 else 0.0
    ate_cuped = _ate(out["metric_cuped"], out["treatment"])
    reduction = (1.0 - var_cuped / var_original) if var_original > 0 else 0.0
    # In-sample reduction is >= 0; clip tiny negative float noise.
    reduction = max(0.0, reduction)

    status = "applied" if n_adjusted >= len(used) + 2 else "skipped"
    worst_imbalance = max(
        (abs(v) for v in balance.values() if not np.isnan(v)), default=0.0
    )

    if status == "applied":
        explanation = (
            f"CUPED used {len(used)} covariate(s) ({', '.join(used)}) to remove "
            f"pre-experiment variance. Outcome variance fell from "
            f"{var_original:.4f} to {var_cuped:.4f}, a {reduction:.1%} reduction "
            f"(equivalent to needing ~{reduction:.0%} fewer samples for the same "
            "precision). The treatment effect is preserved: ATE "
            f"{ate_original:+.4f} -> {ate_cuped:+.4f}."
        )
        recommendation = (
            "Use metric_cuped for the effect estimate and confidence intervals "
            "to gain precision."
        )
        if worst_imbalance > 0.1:
            recommendation += (
                f" NOTE: a covariate is imbalanced across arms (max |SMD|="
                f"{worst_imbalance:.2f}). Verify it is genuinely pre-experiment; "
                "if it is affected by treatment, CUPED on it would bias the ATE."
            )
    else:
        explanation = (
            "CUPED was skipped: too few complete-covariate rows to fit a stable "
            f"adjustment ({n_adjusted} usable rows for {len(used)} covariate(s))."
        )
        recommendation = "Collect more complete covariate data, then re-run CUPED."

    report = {
        "status": status,
        "covariates_used": used,
        "theta": theta,
        "covariate_balance_smd": balance,
        "variance_original": var_original,
        "variance_cuped": var_cuped,
        "variance_reduction": reduction,
        "variance_reduction_pct": reduction * 100.0,
        "ate_original": ate_original,
        "ate_cuped": ate_cuped,
        "n_adjusted": n_adjusted,
        "n_unadjusted": n_unadjusted,
        "explanation": explanation,
        "recommendation": recommendation,
    }
    return out, report
