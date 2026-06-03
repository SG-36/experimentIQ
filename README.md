# ExperimentIQ

**An open-source A/B test analysis pipeline that takes you from a raw experiment
CSV to a trustworthy, plain-English decision.**

Running an A/B test is easy; *trusting* the result is hard. A naive "is the
difference significant?" misses the things that actually invalidate experiments:
a broken randomization, users leaking across arms, a novelty bump that fades, or
peeking at the data until it looks significant. ExperimentIQ runs the checks a
careful data scientist would, explains each one in clear language, and can hand
the whole thing to an LLM to write the final readout.

It is **validated on real, published experiments** — it reproduces the Hillstrom
e-mail lifts and the Udacity landing-page null result (see
[Validation](#validation)).

---

## What it does

Each stage is a small, independent function. Run the ones you need; they compose.

| # | Stage | Module | What you get |
|---|-------|--------|--------------|
| 1 | **Ingest & validate** | `pipeline/ingest.py` | A clean dataframe + a report. Encodes real-world treatment labels (`control`/`treatment`, `A`/`B`, booleans…), drops nulls/invalid rows, de-duplicates users. |
| 2 | **Explore (EDA)** | `pipeline/eda.py` | Group sizes, per-arm metric distributions, an assignment timeline, and heuristic anomaly flags — plus optional Plotly charts. |
| 3 | **Check assumptions** | `pipeline/assumptions.py` | **SRM** (sample-ratio mismatch), **SUTVA** (users in both arms), and **novelty** (effect fading over time) — each pass/fail with an explanation. |
| 4 | **Reduce variance (CUPED)** | `pipeline/cuped.py` | A variance-reduced metric using pre-experiment covariates, with the lift in precision reported. The treatment effect is preserved. |
| 5 | **Estimate the effect** | `pipeline/effect.py` | The headline number: treatment effect, confidence interval, p-value, and relative lift. Auto-picks Welch's t-test or a two-proportion z-test. |
| 6 | **Monitor sequentially** | `pipeline/sequential.py` | An **always-valid p-value** (mixture SPRT) so you can peek continuously and stop early without inflating false positives. |
| 7 | **Find who responds (HTE)** | `pipeline/hte.py` | Per-user treatment effects (CATE) via a causal forest, the covariates driving heterogeneity, and a reliability flag. |
| 8 | **Write the report** | `pipeline/report.py` | A plain-English ship/hold recommendation from Claude that honestly flags any failed assumption. |

Stages 1–7 run **fully offline**. Only the final LLM report (stage 8) needs an
API key.

---

## Quick start

```bash
git clone <your-fork-url> experimentiq
cd experimentiq
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q          # 95 tests, ~5s, no API key or network needed
```

## Input format

A CSV (or any `pandas.DataFrame`) with four required columns:

| Column | Description |
|--------|-------------|
| `user_id` | Unique identifier for the unit of randomization |
| `treatment` | The arm. `0/1`, `control`/`treatment`, `A`/`B`, booleans, etc. — auto-encoded |
| `timestamp` | When the user was assigned (enables the timeline, novelty, and sequential modules) |
| `metric` | The outcome — continuous (e.g. revenue) or binary (e.g. converted) |

Any extra columns are treated as **pre-experiment covariates** and used by CUPED
and the heterogeneity module.

## The 60-second example

```python
import pandas as pd
from pipeline.ingest import run_ingest
from pipeline.assumptions import run_assumption_tests
from pipeline.cuped import run_cuped
from pipeline.effect import estimate_effect
from pipeline.sequential import run_sequential_test
from pipeline.hte import run_hte
from pipeline.report import generate_report

raw = pd.read_csv("your_experiment.csv", parse_dates=["timestamp"])

# 1. Ingest & validate (handles real-world treatment labels)
cleaned, ingest_report = run_ingest(raw)

# 2. Assumption checks (pass `raw` so SUTVA can still see cross-arm user_ids)
assumptions = run_assumption_tests(cleaned, raw_df=raw)

# 3. CUPED variance reduction, then the headline effect on the adjusted metric
adjusted, cuped_report = run_cuped(cleaned)
effect = estimate_effect(adjusted, metric_col="metric_cuped")

# 4. Always-valid (sequential) view + who-responds-differently
sequential = run_sequential_test(cleaned)
scored, hte_report = run_hte(cleaned)

# 5. Plain-English report (needs ANTHROPIC_API_KEY — see below)
out = generate_report({
    "ingest": ingest_report, "assumptions": assumptions, "cuped": cuped_report,
    "effect": effect, "sequential": sequential, "hte": hte_report,
})
print(out["report"])
```

No data yet? Generate some:

```bash
python -m data.synthetic_experiment --scenario clean_experiment --output /tmp/clean.csv
```

Scenarios: `clean_experiment`, `srm_injected`, `novelty_effect`,
`heterogeneous_effects`, `no_effect`.

---

## Features in depth

Every function returns a plain dict with the raw statistics **and** a human-readable
`explanation` + `recommendation`, so results are easy to log, serialize, or show.

### 1. Ingest & validate — `run_ingest(df, column_mapping=None, treatment_mapping=None)`

Returns `(cleaned_df, report)`. The report gives a pass/warn/fail status for the
schema, treatment-encoding, null, invalid-treatment, and duplicate checks, plus
an `overall`. Cleaning is intentionally minimal and fully reported:

- the `treatment` column is encoded to `0`/`1`,
- rows with nulls in required columns are dropped,
- rows with a treatment value other than `0`/`1` are dropped,
- duplicate `user_id`s are resolved by keeping each user's **earliest** row.

**Real-world treatment encoding.** Exports rarely store `treatment` as `0/1`.
The encoder auto-handles numeric `0/1`, booleans, and two-label columns like
`control`/`treatment`, `A`/`B`, `on`/`off`. For custom names — or to pick two
arms out of a multi-variant test — pass an explicit map:

```python
cleaned, report = run_ingest(df, treatment_mapping={"control": 0, "variant_b": 1})
```

If the column can't be interpreted unambiguously (**more than two arms** or
unrecognized labels), ingestion **fails loudly with guidance** instead of
silently dropping rows.

**Scope (by design):** one row per user (pre-aggregated data needs an adapter
first), and exactly two arms per analysis.

### 2. EDA — `run_eda(df)` / `save_eda_plots(df, outdir)`

Returns `group_sizes`, per-arm `metric` summaries, a `timeline`, and a list of
`anomalies` (empty/near-empty arm, constant metric, point mass, extreme
outliers, high skew, timeline gaps). These flags are **heuristic descriptive
checks, not statistical tests**. `save_eda_plots` writes interactive Plotly HTML;
Plotly is imported lazily so the numeric report works without it.

### 3. Assumption testing — `run_assumption_tests(df, raw_df=None, expected_treatment_ratio=0.5)`

Returns `srm`, `sutva`, `novelty`, and an `overall` status.

- **SRM** — chi-square goodness-of-fit on the arm split (default alarm
  `alpha=0.001`). A mismatch usually means a bug, not a real effect.
- **SUTVA** — flags any `user_id` appearing in both arms. Because `run_ingest`
  de-duplicates users, **pass the raw frame as `raw_df`** for this to be meaningful.
- **Novelty** — splits users early vs late at the median timestamp and runs a
  difference-in-differences z-test to see if the effect is fading.

### 4. CUPED — `run_cuped(df, covariates=None)`

Adds a `metric_cuped` column (the original `metric` is untouched) computed as
`metric - (X - mean(X)) @ theta`. Preserves the ATE in expectation and reports
the variance reduction (= regression R²), the fitted `theta`, and a
covariate-balance check. Skips with an explanation if no usable covariates exist.
**Only valid for genuinely pre-experiment covariates** — the balance report
surfaces gross imbalance as a guardrail.

### 5. Effect estimation — `estimate_effect(df, metric_col="metric", alpha=0.05)`

The headline result you compare against a published number. Auto-selects the test:

- **binary / conversion** metrics → two-proportion z-test (absolute lift in pp +
  relative lift %),
- **continuous** metrics → Welch's t-test (unequal variances).

Pass `metric_col="metric_cuped"` after CUPED for a tighter interval. Significance
is consistent with the CI (significant ⟺ the CI excludes zero).

### 6. Sequential testing — `run_sequential_test(df, alpha=0.05, tau=None, target_sample_size=None)`

A mixture Sequential Probability Ratio Test (mSPRT) producing an **always-valid
p-value**: monitor after every observation and stop whenever you like without
inflating the false-positive rate. Reports the running effect, the
log-likelihood-ratio statistic, where (if ever) it `crossed` the `1/alpha`
boundary, and the `decision`. `plot_sequential_test(result)` charts it.

`tau` (the scale of effects to detect) affects **power and timing, not
validity** — type-I error is controlled for any pre-specified `tau`. Not crossing
is *not* proof of no effect; it means not enough signal yet.

### 7. Heterogeneous effects — `run_hte(df, covariates=None, min_per_arm=500)`

Estimates each user's Conditional Average Treatment Effect (CATE) with EconML's
double-ML `CausalForestDML`, adds a `cate` column, and reports the CATE
distribution and the top heterogeneity drivers (forest feature importances).
Flagged unreliable (`status="warn"`) when an arm has fewer than `min_per_arm`
observations; skips with no covariates. CATEs are noisy and model-dependent — the
drivers are *suggestive*, not a formal test of heterogeneity.

### 8. LLM report — `generate_report(results, api_key=None)`

Sends a compact JSON summary of the stage outputs to Claude
(`claude-sonnet-4-20250514`) and returns a report covering the headline result,
trustworthiness (explicitly flagging any SRM/SUTVA/novelty failure), supporting
detail, and a ship/hold recommendation. Long arrays (the sequential time series)
are stripped before sending.

**Setting the API key** — the only part that needs one:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."      # or add it to ~/.zshrc
```

Or pass it inline: `generate_report(results, api_key="sk-ant-...")`. Without a
key, the call is skipped and the fully-built prompt is returned so you can run it
manually.

---

## Validation

ExperimentIQ is checked against experiments whose answers are independently known
— synthetic tests prove the code does what we coded; this proves it recovers
results a skeptic already trusts. Full write-up: [`validation/README.md`](validation/README.md).

| Dataset | What it exercises | Result |
|---------|-------------------|--------|
| **Hillstrom e-mail** (64k users, covariates) | effect, SRM/SUTVA, CUPED, heterogeneity | Reproduces the published +7.7pp visit lift and recovers the documented purchase-history heterogeneity — **10/10 checks** |
| **Udacity landing page** (294k visits, real timestamps) | sequential + novelty, calibration on a true null | Reproduces the ~12.0%/11.9% conversion rates and correctly finds **no** effect; the mSPRT never falsely fires over 290k peeks — **11/11 checks** |

```bash
python validation/hillstrom/fetch.py     && python validation/hillstrom/run_validation.py
python validation/landing_page/fetch.py  && python validation/landing_page/run_validation.py
```

---

## Project structure

```
experimentiq/
├── pipeline/            # the 8 analysis stages (one module each)
├── data/                # synthetic data generator (CLI + importable)
├── tests/               # unit tests (one per stage), run with pytest
├── validation/          # reproductions on real public datasets
│   ├── hillstrom/       #   e-mail experiment (effect, CUPED, HTE)
│   └── landing_page/    #   landing-page experiment (sequential, novelty)
├── requirements.txt
└── README.md
```

## Testing

```bash
pytest -q          # 95 tests, ~5s; no API key or network required
```

Every stage is unit-tested on synthetic data with known ground truth — SRM fires
on the injected-imbalance scenario, the novelty check fires on the decay
scenario, the causal forest recovers the planted heterogeneity, and the effect
estimator matches SciPy's Welch and two-proportion results. The LLM call is faked
in tests, so no key is needed.

## Dependencies

`pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, and `econml` power the
analysis; `plotly` drives the optional charts; `anthropic` is only needed for the
LLM report. All are pinned in `requirements.txt`.

## License

MIT (add a license file before public release).
