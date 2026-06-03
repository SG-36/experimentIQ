"""
Assumption testing for an experiment DataFrame.

Three checks, each returning a pass/fail result with a plain-English
explanation of what the check means and what to do if it fails:

  * SRM (Sample Ratio Mismatch) - chi-square goodness-of-fit test comparing
    the observed arm split to the intended split.
  * SUTVA - flags any user_id that appears in BOTH the treatment and control
    arms (interference / contamination between units).
  * Novelty effect - splits assignees into an early and a late cohort and
    tests whether the treatment effect differs between them (difference-in-
    differences z-test).

Statistical notes / honest caveats:
  * SRM uses scipy.stats.chisquare (Pearson goodness-of-fit, 1 dof for two
    arms). The conventional SRM alarm threshold is a very small alpha (0.001),
    because SRM is a serious validity threat and you want few false alarms.
    An exact binomial test is an equally valid alternative; for the sample
    sizes typical of A/B tests the two agree closely.
  * The SUTVA check here only detects the *observable* violation the build
    plan specifies (same user_id in both arms). It cannot detect network
    spillover or other interference that is not visible in the assignment
    table. IMPORTANT: run_ingest() deduplicates user_ids, so this check must
    be run on the RAW (pre-ingest) data to be meaningful - see
    run_assumption_tests(..., raw_df=...).
  * The novelty check is a difference-in-differences of the early-cohort vs
    late-cohort treatment effect, with a normal-approximation z-test. It does
    not adjust for covariates; it is a screening check, not a final model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_SRM_ALPHA = 0.001
DEFAULT_NOVELTY_ALPHA = 0.05
MIN_CELL_SIZE = 2


def _arm_effect(df: pd.DataFrame) -> tuple[float, float, int, int]:
    """
    Difference in mean metric (treatment - control) and its standard error.

    Returns ``(effect, se, n_control, n_treatment)``. ``effect``/``se`` are
    NaN when either arm has fewer than ``MIN_CELL_SIZE`` observations.
    """
    control = pd.to_numeric(df.loc[df["treatment"] == 0, "metric"], errors="coerce").dropna()
    treatment = pd.to_numeric(df.loc[df["treatment"] == 1, "metric"], errors="coerce").dropna()
    n_c, n_t = int(control.size), int(treatment.size)
    if n_c < MIN_CELL_SIZE or n_t < MIN_CELL_SIZE:
        return float("nan"), float("nan"), n_c, n_t
    effect = float(treatment.mean() - control.mean())
    # Welch-style SE of the difference in means.
    se = float(np.sqrt(treatment.var(ddof=1) / n_t + control.var(ddof=1) / n_c))
    return effect, se, n_c, n_t


def check_srm(
    df: pd.DataFrame,
    expected_treatment_ratio: float = 0.5,
    alpha: float = DEFAULT_SRM_ALPHA,
) -> dict[str, Any]:
    """
    Sample Ratio Mismatch check via chi-square goodness-of-fit.

    Compares observed (control, treatment) counts against the counts implied
    by ``expected_treatment_ratio``. A small p-value means the realized split
    deviates from the intended one more than chance allows - a sign of a
    broken randomization or logging pipeline.
    """
    n_control = int((df["treatment"] == 0).sum())
    n_treatment = int((df["treatment"] == 1).sum())
    n_total = n_control + n_treatment

    if n_total == 0 or expected_treatment_ratio <= 0 or expected_treatment_ratio >= 1:
        return {
            "name": "Sample Ratio Mismatch (SRM)",
            "status": "skip",
            "p_value": None,
            "n_control": n_control,
            "n_treatment": n_treatment,
            "expected_treatment_ratio": expected_treatment_ratio,
            "observed_treatment_ratio": (n_treatment / n_total) if n_total else None,
            "explanation": "Not enough data or an invalid expected ratio; SRM not evaluated.",
            "recommendation": "Provide both arms and a 0<ratio<1 expected split.",
        }

    f_obs = [n_control, n_treatment]
    f_exp = [n_total * (1 - expected_treatment_ratio), n_total * expected_treatment_ratio]
    chi2, p_value = stats.chisquare(f_obs=f_obs, f_exp=f_exp)
    observed_ratio = n_treatment / n_total
    passed = p_value >= alpha

    explanation = (
        "SRM checks whether users actually landed in each arm at the rate you "
        "intended. Here the split was "
        f"{n_control} control / {n_treatment} treatment "
        f"(observed treatment share {observed_ratio:.3f} vs intended "
        f"{expected_treatment_ratio:.3f}); chi-square={chi2:.2f}, p={p_value:.2e}."
    )
    if passed:
        explanation += " The split is consistent with the intended ratio."
        recommendation = "No action needed for SRM."
    else:
        explanation += (
            " The split deviates from the intended ratio by more than chance. "
            "SRM usually points to a bug (faulty randomization, a broken "
            "redirect/feature flag, bot filtering hitting one arm, or logging "
            "loss), NOT a real treatment effect."
        )
        recommendation = (
            "Do not trust the headline result. Investigate the assignment and "
            "logging pipeline, find where users were lost or misassigned, fix "
            "it, and re-run the experiment before interpreting the metric."
        )

    return {
        "name": "Sample Ratio Mismatch (SRM)",
        "status": "pass" if passed else "fail",
        "chi_square": float(chi2),
        "p_value": float(p_value),
        "alpha": alpha,
        "n_control": n_control,
        "n_treatment": n_treatment,
        "expected_treatment_ratio": expected_treatment_ratio,
        "observed_treatment_ratio": observed_ratio,
        "explanation": explanation,
        "recommendation": recommendation,
    }


def check_sutva(df: pd.DataFrame) -> dict[str, Any]:
    """
    SUTVA / contamination check: flag any user_id assigned to both arms.

    Per the build plan this is the observable SUTVA violation: a single
    user_id appearing in both treatment and control means assignment is not
    unit-clean (the same unit was exposed to both conditions), which breaks
    the stable-unit-treatment-value assumption.

    NOTE: this must be run on RAW data. After run_ingest() deduplicates
    user_ids, no user can appear twice and this check trivially passes.
    """
    arms_per_user = df.groupby("user_id")["treatment"].nunique()
    crossover_users = arms_per_user[arms_per_user > 1].index.tolist()
    n_crossover = len(crossover_users)
    n_users = int(arms_per_user.size)
    passed = n_crossover == 0

    if passed:
        explanation = (
            f"No user_id appears in both arms ({n_users} unique users checked). "
            "Each unit was exposed to a single condition, as SUTVA requires."
        )
        recommendation = "No action needed for this SUTVA check."
    else:
        frac = n_crossover / n_users if n_users else 0.0
        explanation = (
            f"{n_crossover} user_id(s) ({frac:.2%}) appear in BOTH the treatment "
            "and control arms. The same unit was exposed to both conditions, so "
            "the arms are contaminated and the comparison is no longer clean. "
            "This violates SUTVA (no interference / one well-defined treatment "
            "per unit)."
        )
        recommendation = (
            "Find why users cross arms (e.g. logged-out vs logged-in IDs, "
            "re-randomization on each visit, device vs account keys). Fix the "
            "unit of randomization, then decide per analysis plan whether to "
            "drop crossover users or re-run; do not silently keep them."
        )

    return {
        "name": "SUTVA (no cross-arm contamination)",
        "status": "pass" if passed else "fail",
        "n_crossover_users": n_crossover,
        "n_users": n_users,
        "crossover_user_ids": crossover_users[:20],
        "explanation": explanation,
        "recommendation": recommendation,
    }


def check_novelty(
    df: pd.DataFrame,
    alpha: float = DEFAULT_NOVELTY_ALPHA,
) -> dict[str, Any]:
    """
    Novelty-effect check: does the treatment effect differ for early vs late
    assignees?

    Users are split at the median assignment timestamp into an early and a
    late cohort. The treatment effect (mean metric difference) is estimated in
    each cohort, and a difference-in-differences z-test asks whether the two
    effects differ. A novelty (or primacy) effect shows up as a significant
    difference - classically the early cohort's effect is inflated and decays.
    """
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp", "metric", "treatment"])

    if work.empty:
        return {
            "name": "Novelty effect (early vs late)",
            "status": "skip",
            "p_value": None,
            "explanation": "No usable rows with timestamp + metric; novelty not evaluated.",
            "recommendation": "Provide timestamped outcomes to run the novelty check.",
        }

    cutoff = work["timestamp"].median()
    early = work[work["timestamp"] <= cutoff]
    late = work[work["timestamp"] > cutoff]

    eff_early, se_early, ec, et = _arm_effect(early)
    eff_late, se_late, lc, lt = _arm_effect(late)

    if any(np.isnan(v) for v in (eff_early, se_early, eff_late, se_late)):
        return {
            "name": "Novelty effect (early vs late)",
            "status": "skip",
            "p_value": None,
            "cohort_sizes": {
                "early": {"control": ec, "treatment": et},
                "late": {"control": lc, "treatment": lt},
            },
            "explanation": (
                "At least one early/late x arm cell had too few observations "
                f"(min cell size {MIN_CELL_SIZE}) to estimate an effect."
            ),
            "recommendation": "Collect more data or widen the cohort window before testing novelty.",
        }

    dd = eff_early - eff_late
    se_dd = float(np.sqrt(se_early**2 + se_late**2))
    z = dd / se_dd if se_dd > 0 else 0.0
    p_value = float(2 * stats.norm.sf(abs(z)))
    passed = p_value >= alpha

    decaying = eff_early > eff_late
    direction = "decays over time (classic novelty effect)" if decaying else "grows over time (primacy / ramp-up)"

    explanation = (
        "This compares the treatment effect for users who entered early vs "
        f"late. Early-cohort effect={eff_early:.3f}, late-cohort effect="
        f"{eff_late:.3f} (difference={dd:+.3f}, z={z:.2f}, p={p_value:.3f})."
    )
    if passed:
        explanation += (
            " The two effects are statistically indistinguishable, so there is "
            "no strong sign of a time-varying (novelty) effect."
        )
        recommendation = "No action needed; the effect looks stable over time."
    else:
        explanation += (
            f" The effect differs between cohorts and {direction}. A single "
            "pooled estimate would then misstate the long-run effect."
        )
        recommendation = (
            "Do not report the pooled effect as the steady-state effect. Let the "
            "experiment run longer and estimate the effect on later, settled "
            "users (or model the time trend explicitly) before deciding."
        )

    return {
        "name": "Novelty effect (early vs late)",
        "status": "pass" if passed else "fail",
        "effect_early": eff_early,
        "effect_late": eff_late,
        "difference": dd,
        "z_statistic": float(z),
        "p_value": p_value,
        "alpha": alpha,
        "cohort_sizes": {
            "early": {"control": ec, "treatment": et},
            "late": {"control": lc, "treatment": lt},
        },
        "explanation": explanation,
        "recommendation": recommendation,
    }


def run_assumption_tests(
    df: pd.DataFrame,
    raw_df: pd.DataFrame | None = None,
    expected_treatment_ratio: float = 0.5,
    srm_alpha: float = DEFAULT_SRM_ALPHA,
    novelty_alpha: float = DEFAULT_NOVELTY_ALPHA,
) -> dict[str, Any]:
    """
    Run all three assumption checks on a (cleaned) experiment DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned experiment data (output of ``pipeline.ingest.run_ingest``).
        Used for SRM and the novelty check.
    raw_df : pd.DataFrame, optional
        The raw, pre-ingest data. If provided, the SUTVA check runs on it so
        cross-arm user_ids are still visible (ingest deduplicates them). If
        omitted, SUTVA runs on ``df`` and will only catch crossovers that
        survived ingestion (normally none).
    expected_treatment_ratio : float
        Intended share of users in the treatment arm (default 0.5).
    srm_alpha, novelty_alpha : float
        Significance thresholds for the SRM and novelty checks.

    Returns
    -------
    dict
        ``{"srm", "sutva", "novelty", "overall", "n_failed"}`` where each check
        is its own result dict and ``overall`` is "pass"/"fail"/"warn".
    """
    sutva_source = raw_df if raw_df is not None else df

    srm = check_srm(df, expected_treatment_ratio=expected_treatment_ratio, alpha=srm_alpha)
    sutva = check_sutva(sutva_source)
    novelty = check_novelty(df, alpha=novelty_alpha)

    sutva["ran_on_raw_data"] = raw_df is not None
    if raw_df is None:
        sutva["note"] = (
            "SUTVA ran on already-ingested data; cross-arm user_ids may have "
            "been removed during deduplication. Pass raw_df to detect them."
        )

    checks = [srm, sutva, novelty]
    statuses = [c["status"] for c in checks]
    n_failed = sum(s == "fail" for s in statuses)
    if "fail" in statuses:
        overall = "fail"
    elif "skip" in statuses:
        overall = "warn"
    else:
        overall = "pass"

    return {
        "srm": srm,
        "sutva": sutva,
        "novelty": novelty,
        "overall": overall,
        "n_failed": n_failed,
    }
