"""
Adapter: Udacity landing-page dataset -> ExperimentIQ schema.

The raw file is already close to our schema (user_id, timestamp, group,
landing_page, converted). Two domain-specific issues must be handled before the
generic pipeline can run:

  * Group / landing-page mismatches. By design ``control`` should see
    ``old_page`` and ``treatment`` should see ``new_page``, but a few thousand
    rows disagree (a control user served the new page, or vice versa). These are
    logging/assignment errors the canonical analysis removes; we drop them here
    and report the count.
  * Duplicate user_ids. A handful of users appear more than once. We deliberately
    KEEP the duplicates so the generic ``run_ingest`` deduplication
    (first-assignment-wins) is exercised on real data.

The remaining mapping is direct:
  * treatment <- group (the auto label encoder maps control->0, treatment->1),
  * timestamp <- real per-user timestamp (enables sequential + novelty modules),
  * metric    <- converted (binary).

There are no usable pre-treatment covariates (``landing_page`` is collinear with
the arm), so CUPED / HTE are not exercised by this dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "ab_data.csv"


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the raw landing-page CSV."""
    return pd.read_csv(path)


def _consistent_mask(raw: pd.DataFrame) -> pd.Series:
    """Rows where the assigned arm and the served page agree."""
    treatment_ok = (raw["group"] == "treatment") & (raw["landing_page"] == "new_page")
    control_ok = (raw["group"] == "control") & (raw["landing_page"] == "old_page")
    return treatment_ok | control_ok


def build_experiment_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Map the raw frame to the ExperimentIQ schema.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        The experiment frame (``user_id``, ``treatment`` raw label,
        ``timestamp``, ``metric``) and a small ``info`` dict recording how many
        mismatched rows were dropped (for reporting in the validation script).
    """
    keep = _consistent_mask(raw)
    n_mismatched = int((~keep).sum())
    clean = raw.loc[keep].copy()

    frame = pd.DataFrame(
        {
            "user_id": clean["user_id"].astype(str),
            "treatment": clean["group"].astype(str),
            "timestamp": pd.to_datetime(clean["timestamp"], errors="coerce"),
            "metric": pd.to_numeric(clean["converted"], errors="coerce"),
        }
    )

    info = {
        "n_raw": int(len(raw)),
        "n_mismatched_dropped": n_mismatched,
        "n_after_mismatch_clean": int(len(frame)),
    }
    return frame, info
