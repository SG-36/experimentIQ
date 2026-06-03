# ExperimentIQ

Open-source A/B test analysis platform that automates the data scientist workflow—from raw experiment data to assumption-checked causal analysis and a plain-language report.

## Status

Implemented so far, each with unit tests on synthetic data:

| Stage | Module | What it does |
|-------|--------|--------------|
| Ingestion & validation | `pipeline/ingest.py` | Schema/null/duplicate/invalid-treatment checks → clean df + report |
| EDA | `pipeline/eda.py` | Group sizes, metric distributions, timeline, heuristic anomaly flags + Plotly plots |
| Assumption testing | `pipeline/assumptions.py` | SRM, SUTVA, novelty-effect checks (pass/fail + plain-English explanations) |
| CUPED | `pipeline/cuped.py` | Variance reduction from pre-experiment covariates |

Planned next: sequential testing, heterogeneous treatment effects, LLM report, and the validation layer.

## Quick start

```bash
cd experimentiq
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -v   # full suite
```

## Input format

CSV with required columns:

| Column | Description |
|--------|-------------|
| `user_id` | Unique user identifier |
| `treatment` | `0` = control, `1` = treatment |
| `timestamp` | Assignment time |
| `metric` | Outcome (e.g. retention, revenue) |

Optional: any number of pre-experiment covariate columns.

## Synthetic data

```bash
python -m data.synthetic_experiment --scenario clean_experiment --output /tmp/clean.csv
python scripts/generate_fixtures.py  # regenerate tests/fixtures/
```

Scenarios: `clean_experiment`, `srm_injected`, `novelty_effect`, `noncompliance`, `heterogeneous_effects`, `no_effect`.

## Ingestion API

```python
from pipeline.ingest import run_ingest
import pandas as pd

df = pd.read_csv("your_experiment.csv", parse_dates=["timestamp"])
cleaned, report = run_ingest(df)
```

`run_ingest` returns a cleaned DataFrame and a validation `report` dict with
pass/warn/fail status for the schema, null, invalid-treatment, and duplicate
checks plus an `overall` status. Cleaning is intentionally minimal and fully
reported:

- rows with nulls in required columns are dropped,
- rows with treatment values other than `0`/`1` are dropped,
- duplicate `user_id`s are resolved by keeping the earliest row per user.

Optional covariate columns are passed through untouched.

## EDA API

```python
from pipeline.eda import run_eda, save_eda_plots

report = run_eda(cleaned)                 # JSON-serializable summary
paths = save_eda_plots(cleaned, "/tmp/eda")  # writes Plotly HTML files
```

`run_eda` returns `group_sizes`, per-arm `metric` summaries, a `timeline`, and
a list of `anomalies`. The anomaly flags (empty/near-empty arm, constant
metric, point mass, extreme outliers, high skew, timeline gaps) are **heuristic
descriptive checks, not statistical tests** — thresholds are named constants at
the top of the module. Plotly is imported lazily, so the numeric report works
without a rendering backend.

## Assumption-testing API

```python
from pipeline.assumptions import run_assumption_tests

results = run_assumption_tests(
    cleaned,
    raw_df=raw,                    # needed so SUTVA can see cross-arm user_ids
    expected_treatment_ratio=0.5,
)
```

Returns `srm`, `sutva`, and `novelty` result dicts plus an `overall` status.
Each check carries a pass/fail `status`, the underlying statistics, and a plain
English `explanation` + `recommendation`.

- **SRM** — chi-square goodness-of-fit on the arm split (default alarm
  `alpha=0.001`).
- **SUTVA** — flags any `user_id` in both arms. Because `run_ingest`
  deduplicates `user_id`s, pass the raw (pre-ingest) frame as `raw_df` for this
  check to be meaningful.
- **Novelty** — splits assignees early vs late at the median timestamp and runs
  a difference-in-differences z-test on the treatment effect.

## CUPED API

```python
from pipeline.cuped import run_cuped

adjusted, report = run_cuped(cleaned)   # uses all numeric covariates by default
# adjusted["metric_cuped"] holds the variance-reduced metric
```

CUPED partials out pre-experiment covariate variance via OLS:
`metric_cuped = metric - (X - mean(X)) @ theta`. It adds a `metric_cuped` column
(the original `metric` is untouched), preserves the ATE in expectation, and
reports the variance reduction achieved (= regression R²), the fitted `theta`,
and a covariate-balance check. If no usable covariates are present it skips and
says why. **CUPED is only valid for genuinely pre-experiment covariates** (not
affected by treatment); the balance report surfaces gross imbalance as a
guardrail.

## License

MIT (add license file when ready for public release)
