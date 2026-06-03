"""
Data ingestion and validation.

Loads experiment data, validates the schema, and checks for nulls, duplicate
user_ids, and invalid treatment values. Returns a cleaned DataFrame alongside
a validation report.

Cleaning decisions (all are reported in the validation report):
  * Rows with nulls in any required column are dropped (unusable).
  * Rows with a treatment value other than 0 or 1 are dropped (unusable).
  * Duplicate user_ids are resolved by keeping the earliest row per user
    (first assignment wins); no aggregation of metric or treatment is done.

No outlier handling, imputation, or variance/assumption testing happens here.
Those belong to later stages of the pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["user_id", "treatment", "timestamp", "metric"]
VALID_TREATMENT_VALUES = (0, 1)

# Common label spellings for the two arms, used to auto-encode real-world
# treatment columns when no explicit mapping is supplied. Matching is done on
# lower-cased, stripped strings.
CONTROL_ALIASES = {
    "0", "control", "ctrl", "c", "a", "off", "baseline", "holdout",
    "false", "no", "original", "default", "comparison",
}
TREATMENT_ALIASES = {
    "1", "treatment", "treat", "t", "b", "variant", "test", "on",
    "exposed", "true", "yes", "new", "experiment",
}


def encode_treatment(
    series: pd.Series,
    treatment_mapping: dict[Any, int] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Encode a real-world treatment column to 0 (control) / 1 (treatment).

    Handles the common ways assignment is recorded in practice: it is already
    0/1, booleans, or text labels such as ``"control"/"treatment"`` or
    ``"A"/"B"``. When the values cannot be interpreted unambiguously (e.g. more
    than two arms, or unrecognized labels) it returns an all-NaN column and an
    ``"unencodable"`` report so the caller can fail with actionable guidance
    rather than silently dropping every row.

    Parameters
    ----------
    series : pd.Series
        The raw treatment column.
    treatment_mapping : dict, optional
        Explicit map from raw values to 0/1, e.g.
        ``{"control": 0, "variant_b": 1}``. Takes precedence over detection.

    Returns
    -------
    tuple[pd.Series, dict]
        The encoded (float 0.0/1.0/NaN) series and an encoding report.
    """
    non_null = series.dropna()
    distinct_raw = list(pd.unique(non_null))
    report: dict[str, Any] = {
        "distinct_raw_values": [_jsonable(v) for v in distinct_raw[:20]],
        "n_distinct_arms": len(distinct_raw),
    }

    if treatment_mapping is not None:
        encoded = pd.to_numeric(series.map(treatment_mapping), errors="coerce")
        report["method"] = "explicit_mapping"
        report["mapping"] = {str(k): v for k, v in treatment_mapping.items()}
        return encoded, report

    # Booleans -> 1/0.
    if pd.api.types.is_bool_dtype(series) or (
        len(distinct_raw) > 0 and all(isinstance(v, (bool, np.bool_)) for v in distinct_raw)
    ):
        encoded = pd.to_numeric(series.map({True: 1, False: 0}), errors="coerce")
        report["method"] = "boolean"
        report["mapping"] = {"True": 1, "False": 0}
        return encoded, report

    # Numeric treatment codes.
    numeric = pd.to_numeric(series, errors="coerce")
    num_non_null = numeric.dropna()
    if not num_non_null.empty:
        num_values = set(num_non_null.unique())
        if num_values.issubset({0, 1}):
            report["method"] = "numeric_0_1"
            return numeric, report
        if {0, 1}.issubset(num_values):
            # Numeric column dominated by 0/1 but with stray invalid codes
            # (data-entry errors). Keep the 0/1 encoding; the invalid-treatment
            # check downstream drops the stray rows.
            report["method"] = "numeric_0_1"
            report["note"] = (
                "Treatment is numeric containing 0 and 1 plus other value(s) "
                f"{sorted(num_values - {0, 1})}; non-0/1 rows are dropped."
            )
            return numeric, report

    # Two text labels that map cleanly onto control/treatment aliases.
    norm = non_null.astype(str).str.strip().str.lower()
    labels = list(pd.unique(norm))
    if len(labels) == 2:
        c_match = [l for l in labels if l in CONTROL_ALIASES]
        t_match = [l for l in labels if l in TREATMENT_ALIASES]
        if len(c_match) == 1 and len(t_match) == 1 and c_match[0] != t_match[0]:
            mapping = {c_match[0]: 0, t_match[0]: 1}
            normalized_full = series.astype(str).str.strip().str.lower()
            encoded = pd.to_numeric(normalized_full.map(mapping), errors="coerce")
            report["method"] = "auto_label"
            report["mapping"] = {c_match[0]: 0, t_match[0]: 1}
            return encoded, report

    report["method"] = "unencodable"
    if len(distinct_raw) > 2:
        report["reason"] = (
            f"Treatment column has {len(distinct_raw)} distinct arms "
            f"{report['distinct_raw_values']}. This pipeline compares exactly two "
            "arms; pass treatment_mapping selecting the control (0) and treatment "
            "(1) arms (other arms become NaN and are dropped)."
        )
    else:
        report["reason"] = (
            f"Could not interpret treatment values {report['distinct_raw_values']}. "
            "Pass treatment_mapping to map raw values to 0 (control) and 1 (treatment)."
        )
    return pd.Series(np.nan, index=series.index, dtype="float64"), report


def _jsonable(value: Any) -> Any:
    """Coerce a raw cell to a JSON/plain-text friendly representation."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def get_covariate_columns(df: pd.DataFrame) -> list[str]:
    """Return the optional covariate columns (everything not required)."""
    return [c for c in df.columns if c not in REQUIRED_COLUMNS]


def validate_schema(
    df: pd.DataFrame,
    column_mapping: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Check that required columns exist and coerce dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        Raw experiment data.
    column_mapping : dict, optional
        Maps canonical names (user_id, treatment, timestamp, metric) to the
        actual column names in ``df``.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        A copy with canonical column names and coerced dtypes, plus the
        schema section of the validation report. When required columns are
        missing the frame is returned unchanged with ``status == "fail"``.
    """
    work = df.copy()

    if column_mapping:
        # Map actual column names -> canonical names.
        inv = {actual: canonical for canonical, actual in column_mapping.items()}
        work = work.rename(columns=inv)

    missing = [c for c in REQUIRED_COLUMNS if c not in work.columns]
    if missing:
        return work, {
            "status": "fail",
            "missing_columns": missing,
            "message": f"Missing required columns: {missing}",
        }

    work["user_id"] = work["user_id"].astype(str)
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work["metric"] = pd.to_numeric(work["metric"], errors="coerce")
    # Treatment is encoded separately (see encode_treatment) so that real-world
    # label columns ("control"/"treatment", booleans, etc.) are handled.

    return work, {
        "status": "pass",
        "missing_columns": [],
        "message": "All required columns present.",
    }


def check_invalid_treatment(df: pd.DataFrame) -> dict[str, Any]:
    """Report non-null treatment values that are not 0 or 1."""
    treatment = df["treatment"]
    invalid_mask = treatment.notna() & ~treatment.isin(VALID_TREATMENT_VALUES)
    n_invalid = int(invalid_mask.sum())
    invalid_values = sorted(treatment[invalid_mask].unique().tolist())
    return {
        "status": "warn" if n_invalid else "pass",
        "n_invalid_rows": n_invalid,
        "invalid_values": invalid_values,
        "message": (
            f"{n_invalid} row(s) had treatment values outside {{0, 1}}: "
            f"{invalid_values}. These rows are dropped."
            if n_invalid
            else "All treatment values are 0 or 1."
        ),
    }


def check_nulls(df: pd.DataFrame) -> dict[str, Any]:
    """Report nulls in required columns."""
    null_counts = {c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS}
    n_rows_with_nulls = int(df[REQUIRED_COLUMNS].isna().any(axis=1).sum())
    return {
        "status": "warn" if n_rows_with_nulls else "pass",
        "n_rows_with_nulls": n_rows_with_nulls,
        "null_counts": null_counts,
        "message": (
            f"{n_rows_with_nulls} row(s) had nulls in required columns "
            f"({null_counts}). These rows are dropped."
            if n_rows_with_nulls
            else "No nulls in required columns."
        ),
    }


def check_duplicate_users(df: pd.DataFrame) -> dict[str, Any]:
    """Report how many user_ids appear more than once."""
    n_rows = len(df)
    n_unique = df["user_id"].nunique()
    n_duplicate_rows = n_rows - n_unique
    n_affected_users = int((df["user_id"].value_counts() > 1).sum())
    return {
        "status": "warn" if n_duplicate_rows else "pass",
        "n_duplicate_rows": n_duplicate_rows,
        "n_affected_users": n_affected_users,
        "message": (
            f"{n_affected_users} user_id(s) appeared more than once "
            f"({n_duplicate_rows} extra row(s)). Earliest row per user is kept."
            if n_duplicate_rows
            else "All user_ids are unique."
        ),
    }


def run_ingest(
    df: pd.DataFrame,
    column_mapping: dict[str, str] | None = None,
    treatment_mapping: dict[Any, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Validate and clean raw experiment data.

    Runs schema validation, treatment encoding, null detection,
    invalid-treatment detection, and duplicate-user detection, then produces a
    cleaned DataFrame:
      1. encode the treatment column to 0/1 (labels/booleans supported),
      2. drop rows with nulls in required columns,
      3. drop rows with invalid treatment values,
      4. keep the earliest row per duplicated user_id.

    Parameters
    ----------
    df : pd.DataFrame
        Raw experiment data.
    column_mapping : dict, optional
        Maps canonical column names to the actual column names in ``df``.
    treatment_mapping : dict, optional
        Explicit map from raw treatment values to 0 (control) / 1 (treatment),
        e.g. ``{"control": 0, "variant_b": 1}``. If omitted, the encoder
        auto-detects numeric 0/1, booleans, and common text labels.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        The cleaned DataFrame and the validation report. If the schema check or
        treatment encoding fails, the (uncleaned) frame is returned with a
        report whose ``overall`` status is ``"fail"``.
    """
    report: dict[str, Any] = {}

    work, schema = validate_schema(df, column_mapping)
    report["schema"] = schema
    if schema["status"] == "fail":
        report["overall"] = "fail"
        report["rows_in"] = len(df)
        report["rows_out"] = 0
        return work, report

    encoded, enc_report = encode_treatment(work["treatment"], treatment_mapping)
    work["treatment"] = encoded
    enc_report["status"] = "fail" if enc_report["method"] == "unencodable" else "pass"
    enc_report["message"] = enc_report.get(
        "reason",
        f"Treatment encoded via '{enc_report['method']}'"
        + (f" ({enc_report['mapping']})." if "mapping" in enc_report else "."),
    )
    report["treatment_encoding"] = enc_report
    if enc_report["status"] == "fail":
        report["overall"] = "fail"
        report["rows_in"] = len(df)
        report["rows_out"] = 0
        return work, report

    report["nulls"] = check_nulls(work)
    report["invalid_treatment"] = check_invalid_treatment(work)

    rows_in = len(work)

    null_mask = work[REQUIRED_COLUMNS].isna().any(axis=1)
    invalid_mask = work["treatment"].notna() & ~work["treatment"].isin(VALID_TREATMENT_VALUES)
    clean = work.loc[~(null_mask | invalid_mask)].copy()

    report["duplicates"] = check_duplicate_users(clean)
    clean = (
        clean.sort_values("timestamp", kind="stable")
        .drop_duplicates(subset="user_id", keep="first")
        .reset_index(drop=True)
    )

    clean["treatment"] = clean["treatment"].astype(int)

    statuses = [
        report["schema"]["status"],
        report["treatment_encoding"]["status"],
        report["nulls"]["status"],
        report["invalid_treatment"]["status"],
        report["duplicates"]["status"],
    ]
    if "fail" in statuses or clean.empty:
        report["overall"] = "fail"
    elif "warn" in statuses:
        report["overall"] = "warn"
    else:
        report["overall"] = "pass"

    report["rows_in"] = rows_in
    report["rows_out"] = len(clean)
    report["n_users"] = len(clean)
    report["covariate_columns"] = get_covariate_columns(clean)

    return clean, report
