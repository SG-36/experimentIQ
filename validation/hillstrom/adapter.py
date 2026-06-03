"""
Adapter: Hillstrom e-mail dataset -> ExperimentIQ schema.

The raw file has no user_id and no per-user timestamp, and the treatment column
(`segment`) has three arms. This adapter:

  * synthesizes a ``user_id`` (row index) and a constant ``timestamp`` (the
    dataset records no per-user time, so time-based modules are not exercised),
  * exposes the outcome of interest as ``metric`` (visit / conversion / spend),
  * keeps the pre-treatment covariates (recency, history, mens, womens, newbie)
    and one-hot-encodes zip_code and channel,
  * returns the raw two/three-arm frame plus the ``treatment_mapping`` for the
    requested contrast, to be passed straight into ``pipeline.ingest.run_ingest``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hillstrom.csv"
SYNTHETIC_TIMESTAMP = "2008-03-01"

# Pre-treatment (baseline) covariates available before the e-mail was sent.
BASE_COVARIATES = ["recency", "history", "mens", "womens", "newbie"]

CONTRASTS: dict[str, dict[str, int]] = {
    "mens_vs_none": {"No E-Mail": 0, "Mens E-Mail": 1},
    "womens_vs_none": {"No E-Mail": 0, "Womens E-Mail": 1},
    "any_vs_none": {"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 1},
}


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the raw Hillstrom CSV."""
    return pd.read_csv(path)


def build_experiment_frame(
    raw: pd.DataFrame,
    outcome: str = "visit",
    contrast: str = "mens_vs_none",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Map the raw frame to the ExperimentIQ schema for a given outcome and contrast.

    Parameters
    ----------
    raw : pd.DataFrame
        Output of :func:`load_raw`.
    outcome : str
        One of ``"visit"``, ``"conversion"``, ``"spend"``.
    contrast : str
        Key of :data:`CONTRASTS` selecting which arms to compare.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        The experiment frame (``user_id``, ``treatment`` raw label,
        ``timestamp``, ``metric``, covariates) and the ``treatment_mapping``
        to pass to ``run_ingest``.
    """
    if outcome not in {"visit", "conversion", "spend"}:
        raise ValueError(f"Unknown outcome: {outcome}")
    if contrast not in CONTRASTS:
        raise ValueError(f"Unknown contrast: {contrast}")

    frame = pd.DataFrame(
        {
            "user_id": [f"cust_{i:06d}" for i in range(len(raw))],
            "treatment": raw["segment"].astype(str),
            "timestamp": pd.Timestamp(SYNTHETIC_TIMESTAMP),
            "metric": pd.to_numeric(raw[outcome], errors="coerce"),
        }
    )

    for col in BASE_COVARIATES:
        frame[col] = pd.to_numeric(raw[col], errors="coerce")

    # One-hot encode the categorical baseline covariates into numeric columns.
    for prefix, col in (("zip", "zip_code"), ("channel", "channel")):
        dummies = pd.get_dummies(raw[col].astype(str), prefix=prefix).astype(int)
        frame = pd.concat([frame, dummies], axis=1)

    return frame, CONTRASTS[contrast]


def covariate_columns(frame: pd.DataFrame) -> list[str]:
    """Numeric covariate columns present in an adapted frame."""
    required = {"user_id", "treatment", "timestamp", "metric"}
    return [c for c in frame.columns if c not in required]
