# Expected results — Udacity landing-page experiment (`ab_data.csv`)

A public e-commerce A/B test (Udacity Data Analyst Nanodegree). ~294k page
visits over ~3 weeks of January 2017. A new landing page (`treatment` /
`new_page`) is compared to the existing page (`control` / `old_page`) on
**conversion** (did the user pay). **Crucially, every row carries a real
per-user timestamp**, so this dataset exercises the time-based modules
(sequential testing, novelty) that the Hillstrom file cannot.

## Why this dataset

| Property | Value for validation |
| --- | --- |
| Real per-user timestamps | Exercises **sequential** + **novelty** modules on real data |
| Known data-quality quirks | Exercises **ingestion** (mismatch + duplicate cleaning) |
| Documented conclusion: **no significant effect** | Tests **calibration** — the pipeline must *not* invent an effect |

## Raw shape and cleaning (ground truth)

| Step | Count |
| --- | --- |
| Raw rows | 294,478 |
| Rows where `group` and `landing_page` disagree (dropped) | 3,893 |
| After mismatch cleaning | 290,585 |
| Duplicate `user_id` removed by ingest (first-wins) | 1 (user 773192) |
| Final analysis rows | 290,584 |

The mismatch rows (e.g. a `control` user served `new_page`) are
logging/assignment errors that the canonical analysis removes. The duplicate
user is deliberately left in the adapter output so the generic ingestion
deduplication is exercised on real data.

## Published / canonical figures

| Quantity | Published ≈ | Source |
| --- | --- | --- |
| Control conversion (old page) | 0.1204 | Udacity project; widely reproduced |
| Treatment conversion (new page) | 0.1188 | " |
| Difference (new − old) | ≈ −0.0016 | " |
| Statistical conclusion | **Not significant** (p ≈ 0.19, CI includes 0) | " |
| Randomization split | ≈ 50/50 (SRM passes) | " |

The canonical conclusion of this project is **fail to reject the null**: the new
page does *not* convert better than the old page.

## What each module should produce

* **Ingestion** — auto-detects the `control`/`treatment` labels, drops the
  3,893 mismatch rows (in the adapter) and the 1 duplicate user (in ingest).
* **Effect estimation** — binary metric → two-proportion z-test; control ≈ 12.0%,
  treatment ≈ 11.9%, effect ≈ −0.16pp, **not significant**, CI brackets 0.
* **SRM** — observed treatment share ≈ 0.500 → **pass**.
* **Novelty (real timestamps)** — early-cohort and late-cohort effects are
  statistically indistinguishable → **pass** (no time-varying effect).
* **Sequential / mSPRT (real timestamps)** — monitored after every one of the
  ~290k observations, the always-valid p-value **never crosses** α=0.05
  (final p = 1.0). This is the headline property: continuous monitoring of a
  true-null experiment does **not** produce a false rejection.
* **Sequential power sanity (semi-synthetic)** — when a real +≈2.3pp lift is
  injected into the treatment arm (keeping the real timestamps), the same mSPRT
  **does** cross early (around n ≈ 8k). This proves the "never crosses" result
  above is genuine type-I control, not an inert test.

## Honest caveats

* This is a teaching dataset distributed by Udacity, not a peer-reviewed
  publication; the "published" figures are the canonical, widely-reproduced
  results of that project, which we recompute directly from the file.
* There are **no usable pre-treatment covariates** (`landing_page` is collinear
  with the arm), so CUPED and HTE are not exercised here — that is what the
  Hillstrom dataset covers.
* The novelty check uses a coarse median-timestamp split; "pass" means no strong
  evidence of a time-varying effect, not proof that none exists.
