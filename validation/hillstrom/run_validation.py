"""
Validate the ExperimentIQ pipeline against the Hillstrom e-mail experiment.

Runs the applicable modules on real data and compares the output to (a) a direct
computation on the same file (exactness) and (b) the canonical published figures
(see expected_results.md). Prints a PASS/FAIL table.

Run from the project root:
    python validation/hillstrom/run_validation.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.assumptions import run_assumption_tests  # noqa: E402
from pipeline.cuped import run_cuped  # noqa: E402
from pipeline.effect import estimate_effect  # noqa: E402
from pipeline.hte import run_hte  # noqa: E402
from pipeline.ingest import run_ingest  # noqa: E402
from validation.hillstrom.adapter import (  # noqa: E402
    build_experiment_frame,
    covariate_columns,
    load_raw,
)

# Canonical published figures (see expected_results.md).
PUB_VISIT_CONTROL = 0.106
PUB_VISIT_MENS = 0.183
PUB_VISIT_LIFT = 0.077

_results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str) -> None:
    _results.append((name, passed, detail))
    flag = "PASS" if passed else "FAIL"
    print(f"[{flag}] {name}: {detail}")


def main() -> int:
    warnings.filterwarnings("ignore")
    raw = load_raw()
    print(f"Loaded {len(raw):,} rows.\n")

    # ----- 1. Headline: visit, Men's E-Mail vs No E-Mail -----
    frame, mapping = build_experiment_frame(raw, outcome="visit", contrast="mens_vs_none")
    cleaned, ingest_report = run_ingest(frame, treatment_mapping=mapping)
    print(f"Ingest: {ingest_report['rows_in']:,} -> {ingest_report['rows_out']:,} rows "
          f"(encoding: {ingest_report['treatment_encoding']['method']}); "
          f"overall={ingest_report['overall']}\n")

    eff = estimate_effect(cleaned, metric_col="metric")

    # Direct (ground-truth) computation on the same rows.
    direct_c = cleaned.loc[cleaned["treatment"] == 0, "metric"].mean()
    direct_t = cleaned.loc[cleaned["treatment"] == 1, "metric"].mean()
    direct_effect = direct_t - direct_c

    check(
        "visit: two-proportion uses binary path",
        eff["method"] == "two_proportion_z" and eff["metric_type"] == "binary",
        f"method={eff['method']}",
    )
    check(
        "visit: effect matches direct groupby (exactness)",
        abs(eff["effect"] - direct_effect) < 1e-9,
        f"pipeline={eff['effect']:.6f} vs direct={direct_effect:.6f}",
    )
    check(
        "visit: control rate matches published ~10.6%",
        abs(direct_c - PUB_VISIT_CONTROL) < 0.005,
        f"obtained={direct_c:.4f}, published≈{PUB_VISIT_CONTROL}",
    )
    check(
        "visit: men's rate matches published ~18.3%",
        abs(direct_t - PUB_VISIT_MENS) < 0.005,
        f"obtained={direct_t:.4f}, published≈{PUB_VISIT_MENS}",
    )
    check(
        "visit: lift matches published ~+7.7pp and is significant",
        abs(eff["effect"] - PUB_VISIT_LIFT) < 0.006 and eff["significant"],
        f"lift={eff['effect'] * 100:.2f}pp, p={eff['p_value']:.2e}, CI="
        f"[{eff['ci_low'] * 100:.2f}, {eff['ci_high'] * 100:.2f}]pp",
    )

    # ----- 2. Assumptions: SRM should pass at 0.5 -----
    assumptions = run_assumption_tests(cleaned, raw_df=frame, expected_treatment_ratio=0.5)
    srm = assumptions["srm"]
    check(
        "SRM passes (clean ~50/50 randomization)",
        srm["status"] == "pass",
        f"obs ratio={srm['observed_treatment_ratio']:.4f}, p={srm['p_value']:.3f}",
    )
    check(
        "SUTVA passes (no cross-arm users)",
        assumptions["sutva"]["status"] == "pass",
        f"crossover users={assumptions['sutva']['n_crossover_users']}",
    )

    # ----- 3. Conversion lift (Men's vs None) -----
    frame_c, mapping_c = build_experiment_frame(raw, outcome="conversion", contrast="mens_vs_none")
    cleaned_c, _ = run_ingest(frame_c, treatment_mapping=mapping_c)
    eff_c = estimate_effect(cleaned_c)
    check(
        "conversion: e-mail lifts conversion, significant",
        eff_c["effect"] > 0 and eff_c["significant"],
        f"control={eff_c['mean_control']:.4f}, mens={eff_c['mean_treatment']:.4f}, "
        f"p={eff_c['p_value']:.2e}",
    )

    # ----- 4. CUPED on spend. Honest result: this dataset has no strong
    # pre-period predictor of the (99% zero) 2-week spend, so the correct
    # outcome is a ~0% reduction with the ATE preserved. We validate those
    # properties, not a fabricated reduction.
    frame_s, mapping_s = build_experiment_frame(raw, outcome="spend", contrast="mens_vs_none")
    cleaned_s, _ = run_ingest(frame_s, treatment_mapping=mapping_s)
    adjusted, cuped_report = run_cuped(
        cleaned_s, covariates=["history", "recency", "mens", "womens", "newbie"]
    )
    ate_preserved = abs(cuped_report["ate_cuped"] - cuped_report["ate_original"]) < 0.05
    check(
        "CUPED applies cleanly and preserves the ATE (reduction is data-limited here)",
        cuped_report["status"] == "applied" and ate_preserved,
        f"reduction={cuped_report['variance_reduction_pct']:.2f}% "
        f"(history~spend corr≈0.02, spend is ~99% zeros), "
        f"ATE {cuped_report['ate_original']:+.3f} -> {cuped_report['ate_cuped']:+.3f}",
    )

    # ----- 5. Heterogeneity: mens/womens drive response (any e-mail vs none) -----
    frame_h, mapping_h = build_experiment_frame(raw, outcome="visit", contrast="any_vs_none")
    cleaned_h, _ = run_ingest(frame_h, treatment_mapping=mapping_h)
    covs = covariate_columns(cleaned_h)
    scored, hte_report = run_hte(
        cleaned_h, covariates=covs, n_estimators=100, min_samples_leaf=50, min_per_arm=500
    )
    drivers = [d["covariate"] for d in hte_report["heterogeneity_drivers"]]
    check(
        "HTE recovers purchase-history heterogeneity (mens/womens in top drivers)",
        any(d in ("mens", "womens") for d in drivers),
        f"top drivers={drivers}",
    )

    # ----- Summary -----
    n_pass = sum(p for _, p, _ in _results)
    print(f"\n=== {n_pass}/{len(_results)} checks passed ===")
    return 0 if n_pass == len(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
