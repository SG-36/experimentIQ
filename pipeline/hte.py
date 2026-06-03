"""
Heterogeneous treatment effects (HTE) via EconML's CausalForestDML.

Estimates the Conditional Average Treatment Effect (CATE) - how the treatment
effect varies across users as a function of their covariates - using a causal
forest with double machine learning (orthogonalized nuisance models). Produces:

  * a per-user CATE estimate (added as a ``cate`` column),
  * a distribution summary of the CATE,
  * the top covariates driving the heterogeneity (forest feature importances),
  * a reliability flag when there are fewer than ``min_per_arm`` observations
    per arm (causal forests are data-hungry; small samples give noisy CATEs).

Honest caveats
--------------
  * CATE estimates are noisy and model-dependent. Feature importances indicate
    which covariates the forest used to split on heterogeneity; they are
    suggestive, not a hypothesis test of heterogeneity.
  * DML assumes unconfoundedness given the covariates. In a randomized A/B test
    treatment is independent of covariates, so this holds by design; on
    observational data the usual caveats apply.
  * Requires pre-experiment covariates. With none, the module skips (a single
    global ATE is all you can estimate).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["user_id", "treatment", "timestamp", "metric"]
DEFAULT_MIN_PER_ARM = 500


def select_hte_covariates(
    df: pd.DataFrame,
    covariates: list[str] | None = None,
) -> list[str]:
    """Choose usable covariates (numeric, non-constant, non-required)."""
    if covariates is None:
        candidates = [c for c in df.columns if c not in REQUIRED_COLUMNS and c != "cate"]
    else:
        candidates = [c for c in covariates if c in df.columns]

    usable: list[str] = []
    for c in candidates:
        col = pd.to_numeric(df[c], errors="coerce").dropna()
        if col.empty or col.nunique() <= 1:
            continue
        usable.append(c)
    return usable


def _cate_summary(cate: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(cate)),
        "std": float(np.std(cate, ddof=1)) if cate.size > 1 else 0.0,
        "min": float(np.min(cate)),
        "p10": float(np.quantile(cate, 0.10)),
        "p25": float(np.quantile(cate, 0.25)),
        "median": float(np.median(cate)),
        "p75": float(np.quantile(cate, 0.75)),
        "p90": float(np.quantile(cate, 0.90)),
        "max": float(np.max(cate)),
        "frac_positive": float(np.mean(cate > 0)),
    }


def run_hte(
    df: pd.DataFrame,
    covariates: list[str] | None = None,
    outcome_col: str = "metric",
    treatment_col: str = "treatment",
    min_per_arm: int = DEFAULT_MIN_PER_ARM,
    n_estimators: int = 200,
    min_samples_leaf: int = 20,
    random_state: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Estimate heterogeneous treatment effects with CausalForestDML.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned experiment data with covariates.
    covariates : list[str], optional
        Covariates to model heterogeneity over. Defaults to all numeric,
        non-constant, non-required columns.
    outcome_col, treatment_col : str
        Outcome and treatment column names.
    min_per_arm : int
        Below this many observations per arm the result is flagged unreliable.
    n_estimators, min_samples_leaf, random_state : int
        Causal-forest hyperparameters (also used for the RF nuisance models).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        The DataFrame with an added ``cate`` column (NaN for rows that could
        not be scored) and a report. Status is ``"skip"`` when HTE cannot be
        estimated, ``"warn"`` when estimated but under-powered, else ``"ok"``.
    """
    out = df.copy()
    out["cate"] = np.nan
    used = select_hte_covariates(out, covariates)

    n_t_all = int((out[treatment_col] == 1).sum())
    n_c_all = int((out[treatment_col] == 0).sum())

    if not used:
        return out, {
            "status": "skip",
            "reliable": False,
            "covariates_used": [],
            "reason": "No usable covariates; cannot model heterogeneity.",
            "n_treatment": n_t_all,
            "n_control": n_c_all,
            "explanation": (
                "Heterogeneous treatment effects were skipped: without "
                "covariates only a single global ATE can be estimated."
            ),
            "recommendation": "Add pre-experiment covariates to explore who responds differently.",
        }

    cov_numeric = out[used].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(out[outcome_col], errors="coerce")
    t = pd.to_numeric(out[treatment_col], errors="coerce")
    complete = cov_numeric.notna().all(axis=1) & y.notna() & t.notna()

    n_used = int(complete.sum())
    n_dropped = int((~complete).sum())
    sub = out.loc[complete]
    n_t = int((sub[treatment_col] == 1).sum())
    n_c = int((sub[treatment_col] == 0).sum())

    # Need both arms and enough rows to fit a forest at all.
    if n_t < 2 or n_c < 2 or n_used < 4 * min_samples_leaf:
        return out, {
            "status": "skip",
            "reliable": False,
            "covariates_used": used,
            "reason": "Too few complete observations to fit a causal forest.",
            "n_treatment": n_t,
            "n_control": n_c,
            "n_used": n_used,
            "explanation": (
                f"Only {n_used} complete rows ({n_t} treatment / {n_c} control) "
                "were available - not enough to fit a causal forest."
            ),
            "recommendation": "Collect more data before estimating heterogeneity.",
        }

    from econml.dml import CausalForestDML
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    X = cov_numeric.loc[complete].to_numpy(dtype=float)
    Y = y.loc[complete].to_numpy(dtype=float)
    T = t.loc[complete].to_numpy(dtype=int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = CausalForestDML(
            model_y=RandomForestRegressor(
                n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            ),
            model_t=RandomForestClassifier(
                n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            ),
            discrete_treatment=True,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        est.fit(Y, T, X=X)
        cate = est.effect(X)
        importances = np.asarray(est.feature_importances_, dtype=float)

    out.loc[complete, "cate"] = cate

    order = np.argsort(importances)[::-1]
    drivers = [
        {"covariate": used[i], "importance": float(importances[i])}
        for i in order[:3]
    ]
    summary = _cate_summary(np.asarray(cate, dtype=float))

    reliable = min(n_t, n_c) >= min_per_arm
    status = "ok" if reliable else "warn"

    # Rough heterogeneity signal: CATE spread relative to the outcome scale.
    outcome_sd = float(np.std(Y, ddof=1)) if Y.size > 1 else 0.0
    spread_ratio = (summary["std"] / outcome_sd) if outcome_sd > 0 else 0.0

    explanation = (
        f"CausalForestDML estimated per-user treatment effects over "
        f"{len(used)} covariate(s). Average effect (ATE) = {summary['mean']:+.4f}; "
        f"CATE ranges [{summary['min']:+.3f}, {summary['max']:+.3f}] with "
        f"{summary['frac_positive']:.0%} of users estimated to benefit. The "
        f"strongest heterogeneity driver is '{drivers[0]['covariate']}' "
        f"(importance {drivers[0]['importance']:.2f})."
    )
    if not reliable:
        explanation += (
            f" RELIABILITY WARNING: only {min(n_t, n_c)} observations in the "
            f"smaller arm (< {min_per_arm}). Causal-forest CATEs are likely "
            "noisy; treat the heterogeneity findings as exploratory."
        )
        recommendation = (
            "Do not target users based on these CATEs yet; collect more data "
            f"(aim for >= {min_per_arm} per arm) before acting on heterogeneity."
        )
    elif spread_ratio < 0.05:
        recommendation = (
            "Heterogeneity looks small relative to noise; a single ATE is "
            "probably an adequate summary, but inspect the top drivers."
        )
    else:
        recommendation = (
            "There is meaningful effect variation. Inspect the top drivers and "
            "consider segment-level decisions, validating on a holdout."
        )

    report = {
        "status": status,
        "reliable": reliable,
        "covariates_used": used,
        "n_treatment": n_t,
        "n_control": n_c,
        "n_used": n_used,
        "n_dropped": n_dropped,
        "ate": summary["mean"],
        "cate_summary": summary,
        "heterogeneity_drivers": drivers,
        "spread_ratio": spread_ratio,
        "explanation": explanation,
        "recommendation": recommendation,
    }
    return out, report


def plot_cate_distribution(df: pd.DataFrame, report: dict[str, Any] | None = None):
    """Histogram of the estimated CATE across users. Returns a Plotly Figure."""
    import plotly.graph_objects as go

    cate = pd.to_numeric(df["cate"], errors="coerce").dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=cate, nbinsx=50, name="CATE"))
    if not cate.empty:
        fig.add_vline(
            x=float(cate.mean()), line_dash="dash", line_color="red",
            annotation_text=f"ATE={cate.mean():+.3f}",
        )
    fig.update_layout(
        title="Distribution of conditional average treatment effects (CATE)",
        xaxis_title="estimated treatment effect",
        yaxis_title="users",
    )
    return fig
