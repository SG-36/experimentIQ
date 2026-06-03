"""
Generate synthetic A/B experiment datasets with known ground truth.

Scenarios (combinable via CLI flags):
  clean_experiment, srm_injected, novelty_effect, noncompliance,
  heterogeneous_effects, no_effect
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["user_id", "treatment", "timestamp", "metric"]
DEFAULT_COVARIATES = ["prior_sessions", "account_age_days"]


def _base_users(
    n_users: int,
    seed: int,
    treatment_ratio: float = 0.5,
    start_date: datetime | None = None,
) -> pd.DataFrame:
    """Build user-level skeleton with assignment and covariates."""
    rng = np.random.default_rng(seed)
    if start_date is None:
        start_date = datetime(2024, 1, 1)

    user_ids = [f"user_{i:06d}" for i in range(n_users)]
    treatment = (rng.random(n_users) < treatment_ratio).astype(int)

    # Stagger assignments over 21 days
    day_offsets = rng.integers(0, 21, size=n_users)
    timestamps = [start_date + timedelta(days=int(d)) for d in day_offsets]

    prior_sessions = rng.poisson(5, size=n_users).astype(float)
    account_age_days = rng.exponential(180, size=n_users).astype(float)
    # age proxy for heterogeneous effects
    age_years = rng.integers(18, 70, size=n_users).astype(float)

    return pd.DataFrame(
        {
            "user_id": user_ids,
            "treatment": treatment,
            "timestamp": timestamps,
            "prior_sessions": prior_sessions,
            "account_age_days": account_age_days,
            "age_years": age_years,
        }
    )


def _add_metric(
    df: pd.DataFrame,
    seed: int,
    true_ate: float = 0.15,
    covariate_correlations: tuple[float, float] = (0.4, 0.2),
    heterogeneous: bool = False,
    novelty: bool = False,
    noncompliance_rate: float = 0.0,
) -> pd.DataFrame:
    """Add outcome metric with optional effect modifiers and violations."""
    rng = np.random.default_rng(seed)
    out = df.copy()

    # Baseline metric from covariates
    z1 = (out["prior_sessions"] - out["prior_sessions"].mean()) / (
        out["prior_sessions"].std() + 1e-8
    )
    z2 = (out["account_age_days"] - out["account_age_days"].mean()) / (
        out["account_age_days"].std() + 1e-8
    )
    base = 1.0 + covariate_correlations[0] * 0.3 * z1 + covariate_correlations[1] * 0.2 * z2
    noise = rng.normal(0, 0.5, size=len(out))
    metric = base + noise

    # Treatment effect
    effect = np.zeros(len(out))
    if heterogeneous:
        young = out["age_years"] < 30
        effect = np.where(young, true_ate, 0.0)
    else:
        effect = np.full(len(out), true_ate)

    if novelty:
        # Week tercile based on timestamp
        ts = pd.to_datetime(out["timestamp"])
        tercile = pd.qcut(ts.rank(method="first"), 3, labels=[1, 2, 3]).astype(int)
        decay = {1: 1.0, 2: 0.6, 3: 0.25}  # week1 strong, week3 weak
        multiplier = tercile.map(decay).to_numpy()
        effect = effect * multiplier

    # Noncompliance: some control-labeled users actually receive treatment effect
    treat_lift = effect * (out["treatment"] == 1).to_numpy().astype(float)
    metric = metric + treat_lift

    # Noncompliance: inflate outcomes for a subset of control-labeled users
    if noncompliance_rate > 0:
        control_mask = out["treatment"] == 0
        n_control = int(control_mask.sum())
        n_cross = int(n_control * noncompliance_rate)
        if n_cross > 0:
            control_idx = np.where(control_mask)[0]
            cross_idx = rng.choice(control_idx, size=n_cross, replace=False)
            ctrl_mean = metric[control_mask].mean()
            treat_mean = metric[out["treatment"] == 1].mean()
            crossover_bump = max(1.15, (treat_mean - ctrl_mean) + true_ate * 5.0)
            metric[cross_idx] = metric[cross_idx] + crossover_bump
    out["metric"] = metric
    if "_actual_treatment" in out.columns:
        out = out.drop(columns=["_actual_treatment"])

    return out


def generate_experiment(
    n_users: int = 2000,
    seed: int = 42,
    scenario: str = "clean_experiment",
    treatment_ratio: float | None = None,
    inject_outliers: bool = False,
    inject_duplicates: bool = False,
    inject_missing_covariates: bool = False,
) -> pd.DataFrame:
    """
    Generate a synthetic experiment DataFrame for a named scenario.

    Parameters
    ----------
    n_users : int
        Number of unique users.
    seed : int
        Random seed for reproducibility.
    scenario : str
        One of: clean_experiment, srm_injected, novelty_effect,
        noncompliance, heterogeneous_effects, no_effect.
    treatment_ratio : float, optional
        Override assignment probability to treatment arm.
    inject_outliers : bool
        If True, inflate metric for 1% of users by 10x mean.
    inject_duplicates : bool
        If True, duplicate 5% of rows with new timestamps.
    inject_missing_covariates : bool
        If True, set 10% of covariate values to NaN.

    Returns
    -------
    pd.DataFrame
        Experiment data with required columns plus covariates.
    """
    ratio = 0.5
    true_ate = 0.15
    heterogeneous = False
    novelty = False
    noncompliance_rate = 0.0

    if scenario == "clean_experiment":
        pass
    elif scenario == "srm_injected":
        ratio = 0.6
    elif scenario == "novelty_effect":
        novelty = True
        true_ate = 0.20
    elif scenario == "noncompliance":
        noncompliance_rate = 0.10
    elif scenario == "heterogeneous_effects":
        heterogeneous = True
        true_ate = 0.20
    elif scenario == "no_effect":
        true_ate = 0.0
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    if treatment_ratio is not None:
        ratio = treatment_ratio

    df = _base_users(n_users, seed, treatment_ratio=ratio)
    df = _add_metric(
        df,
        seed + 1,
        true_ate=true_ate,
        heterogeneous=heterogeneous,
        novelty=novelty,
        noncompliance_rate=noncompliance_rate,
    )

    rng = np.random.default_rng(seed + 2)
    if inject_outliers:
        n_out = max(1, int(0.01 * len(df)))
        idx = rng.choice(len(df), size=n_out, replace=False)
        df.loc[df.index[idx], "metric"] *= 10

    if inject_duplicates:
        n_dup = max(1, int(0.05 * len(df)))
        dup_rows = df.sample(n=n_dup, random_state=seed).copy()
        dup_rows["timestamp"] = dup_rows["timestamp"] + pd.Timedelta(hours=1)
        df = pd.concat([df, dup_rows], ignore_index=True)

    if inject_missing_covariates:
        cov_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
        for col in cov_cols:
            mask = rng.random(len(df)) < 0.1
            df.loc[mask, col] = np.nan

    return df


def main() -> None:
    """CLI entry point to write synthetic CSV files."""
    parser = argparse.ArgumentParser(description="Generate synthetic experiment data")
    parser.add_argument(
        "--scenario",
        default="clean_experiment",
        choices=[
            "clean_experiment",
            "srm_injected",
            "novelty_effect",
            "noncompliance",
            "heterogeneous_effects",
            "no_effect",
        ],
    )
    parser.add_argument("--n-users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outliers", action="store_true")
    parser.add_argument("--duplicates", action="store_true")
    parser.add_argument("--missing-covariates", action="store_true")
    args = parser.parse_args()

    df = generate_experiment(
        n_users=args.n_users,
        seed=args.seed,
        scenario=args.scenario,
        inject_outliers=args.outliers,
        inject_duplicates=args.duplicates,
        inject_missing_covariates=args.missing_covariates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
