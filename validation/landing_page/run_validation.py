"""
Validate the ExperimentIQ pipeline against the Udacity landing-page experiment.

This dataset's value is its real per-user timestamps and its documented
*null* conclusion. It checks two things the Hillstrom validation cannot:

  1. The time-based modules (sequential testing, novelty) run on real data.
  2. Calibration: faced with a true-null experiment, the pipeline must NOT
     manufacture a significant effect — including under continuous monitoring.

Runs the applicable modules, compares to a direct computation and to the
canonical published figures (see expected_results.md), and prints a PASS/FAIL
table.

Run from the project root:
    python validation/landing_page/run_validation.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.assumptions import run_assumption_tests  # noqa: E402
from pipeline.effect import estimate_effect  # noqa: E402
from pipeline.ingest import run_ingest  # noqa: E402
from pipeline.sequential import run_sequential_test  # noqa: E402
from validation.landing_page.adapter import (  # noqa: E402
    build_experiment_frame,
    load_raw,
)

# Canonical published figures (see expected_results.md).
PUB_CONTROL_CONV = 0.1204
PUB_TREATMENT_CONV = 0.1188
EXPECTED_MISMATCH_DROPPED = 3893

_results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str) -> None:
    _results.append((name, passed, detail))
    flag = "PASS" if passed else "FAIL"
    print(f"[{flag}] {name}: {detail}")


def main() -> int:
    warnings.filterwarnings("ignore")
    raw = load_raw()
    print(f"Loaded {len(raw):,} raw rows.\n")

    frame, info = build_experiment_frame(raw)
    cleaned, ingest_report = run_ingest(frame)
    print(
        f"Ingest: {ingest_report['rows_in']:,} -> {ingest_report['rows_out']:,} rows "
        f"(encoding: {ingest_report['treatment_encoding']['method']}); "
        f"overall={ingest_report['overall']}\n"
    )

    # ----- 1. Ingestion handles the documented data-quality quirks -----
    check(
        "ingest: drops the 3,893 group/landing_page mismatch rows",
        info["n_mismatched_dropped"] == EXPECTED_MISMATCH_DROPPED,
        f"dropped={info['n_mismatched_dropped']} (expected {EXPECTED_MISMATCH_DROPPED})",
    )
    check(
        "ingest: auto-detects control/treatment labels and dedups users",
        ingest_report["treatment_encoding"]["method"] == "auto_label"
        and ingest_report["rows_out"] == info["n_after_mismatch_clean"] - 1,
        f"method={ingest_report['treatment_encoding']['method']}, "
        f"rows {info['n_after_mismatch_clean']} -> {ingest_report['rows_out']} "
        f"(1 duplicate user removed)",
    )

    # ----- 2. Headline effect: conversion, new vs old page -----
    eff = estimate_effect(cleaned, metric_col="metric")
    direct_c = cleaned.loc[cleaned["treatment"] == 0, "metric"].mean()
    direct_t = cleaned.loc[cleaned["treatment"] == 1, "metric"].mean()

    check(
        "effect: binary metric uses two-proportion path",
        eff["method"] == "two_proportion_z" and eff["metric_type"] == "binary",
        f"method={eff['method']}",
    )
    check(
        "effect: matches direct groupby (exactness)",
        abs(eff["effect"] - (direct_t - direct_c)) < 1e-9,
        f"pipeline={eff['effect']:.6f} vs direct={direct_t - direct_c:.6f}",
    )
    check(
        "effect: control conversion matches published ~12.04%",
        abs(direct_c - PUB_CONTROL_CONV) < 0.003,
        f"obtained={direct_c:.4f}, published≈{PUB_CONTROL_CONV}",
    )
    check(
        "effect: treatment conversion matches published ~11.88%",
        abs(direct_t - PUB_TREATMENT_CONV) < 0.003,
        f"obtained={direct_t:.4f}, published≈{PUB_TREATMENT_CONV}",
    )
    check(
        "calibration: new page is NOT significantly better (matches conclusion)",
        (not eff["significant"]) and eff["ci_low"] < 0 < eff["ci_high"],
        f"effect={eff['effect'] * 100:.2f}pp, p={eff['p_value']:.3f}, CI="
        f"[{eff['ci_low'] * 100:.2f}, {eff['ci_high'] * 100:.2f}]pp",
    )

    # ----- 3. SRM passes (clean ~50/50 randomization) -----
    assumptions = run_assumption_tests(cleaned, raw_df=frame, expected_treatment_ratio=0.5)
    srm = assumptions["srm"]
    check(
        "SRM passes (clean ~50/50 randomization)",
        srm["status"] == "pass",
        f"obs ratio={srm['observed_treatment_ratio']:.4f}, p={srm['p_value']:.3f}",
    )

    # ----- 4. Novelty on REAL timestamps: stable effect -> pass -----
    novelty = assumptions["novelty"]
    check(
        "novelty (real timestamps): no time-varying effect -> pass",
        novelty["status"] == "pass",
        f"early={novelty['effect_early']:+.4f}, late={novelty['effect_late']:+.4f}, "
        f"p={novelty['p_value']:.3f}",
    )

    # ----- 5. Sequential on REAL timestamps: true null never crosses -----
    seq = run_sequential_test(cleaned, metric_col="metric")
    check(
        "sequential (real timestamps): true null never crosses under continuous monitoring",
        seq["status"] == "ok" and (not seq["crossed"]) and seq["decision"] == "no_decision",
        f"crossed={seq['crossed']}, final p={seq['final_p_value']:.3f}, "
        f"peeked at all {seq['n_total']:,} observations",
    )

    # ----- 6. Sequential power sanity (semi-synthetic perturbation) -----
    # Inject a real +~2.3pp lift into the treatment arm (keep real timestamps)
    # to confirm the mSPRT *does* fire on signal — i.e. check #5 is genuine
    # type-I control, not an inert test.
    rng = np.random.default_rng(0)
    pert = cleaned.copy()
    t_zero = pert.index[(pert["treatment"] == 1) & (pert["metric"] == 0)]
    n_flip = int(0.025 * int((pert["treatment"] == 1).sum()))
    flip = rng.choice(t_zero, size=n_flip, replace=False)
    pert.loc[flip, "metric"] = 1
    seq_pert = run_sequential_test(pert, metric_col="metric")
    check(
        "sequential power sanity: a real injected lift DOES cross (test is not inert)",
        seq_pert["crossed"] and seq_pert["decision"] == "reject_h0",
        f"crossed at n={seq_pert.get('crossing_sample_size'):,}, "
        f"final effect={seq_pert['final_effect']:+.4f}",
    )

    # ----- Summary -----
    n_pass = sum(p for _, p, _ in _results)
    print(f"\n=== {n_pass}/{len(_results)} checks passed ===")
    return 0 if n_pass == len(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
