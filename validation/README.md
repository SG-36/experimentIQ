# Validation layer

This folder validates the ExperimentIQ pipeline on **real, published experiments**
where the correct answer is independently known. Synthetic tests prove the code
does what we coded; this proves it recovers results a skeptic already trusts.

Each dataset lives in its own subfolder with a fetch script, an adapter (maps the
real schema to the `user_id / treatment / timestamp / metric` contract), a runner
that executes the applicable modules, and an `expected_results.md` documenting the
ground truth. Raw data is cached in `validation/data/` (gitignored).

## At a glance

The two datasets are deliberately complementary — together they cover every
module:

| Dataset | Shape | Modules exercised | Headline result | Checks |
|---------|-------|-------------------|-----------------|--------|
| **Hillstrom** | 64k users, covariates, **no timestamps** | effect, SRM, SUTVA, CUPED, HTE | Reproduces the published **+7.7pp** visit lift; recovers documented heterogeneity | **10/10** |
| **Udacity landing page** | 294k visits, **real timestamps**, no covariates | effect, SRM, **sequential**, **novelty** | Reproduces the **null result** (new page not better); mSPRT never falsely fires | **11/11** |

Each subfolder runs the same way:

```bash
python validation/<name>/fetch.py           # download raw data -> validation/data/
python validation/<name>/run_validation.py  # run modules, print a PASS/FAIL table
```

---

## Hillstrom e-mail experiment (`hillstrom/`)

**Dataset:** Kevin Hillstrom's *MineThatData E-Mail Analytics and Data Mining
Challenge* (2008). 64,000 customers randomized ~evenly into **No E-Mail**,
**Men's E-Mail**, **Women's E-Mail**; outcomes `visit`, `conversion`, `spend`
over the following two weeks. User-level, three arms, with pre-treatment
covariates — so it exercises the full causal stack.

**Run it:**
```bash
python validation/hillstrom/fetch.py          # downloads to validation/data/
python validation/hillstrom/run_validation.py # runs modules, prints PASS/FAIL
```

### Results — obtained vs. published (10/10 checks pass)

Primary contrast: **Men's E-Mail vs No E-Mail** (the women's arm is dropped via
`treatment_mapping`, exercising the two-of-three-arm path). Ingestion: 64,000 →
42,613 rows (women's arm removed), treatment encoded via explicit mapping.

| Check | Published / expected | Obtained | Status |
|-------|----------------------|----------|--------|
| Visit rate, control | ~10.6% | **10.62%** | ✅ |
| Visit rate, Men's E-Mail | ~18.3% | **18.28%** | ✅ |
| Visit lift (Men's − control) | ~+7.7 pp, significant | **+7.66 pp**, 95% CI [7.00, 8.32], p=5.7e-112 | ✅ |
| Effect == direct `groupby` | exact | 0.076590 == 0.076590 | ✅ |
| Test chosen for binary metric | two-proportion z | two-proportion z | ✅ |
| Conversion rate (control → Men's) | ~0.57% → ~1.25% | **0.57% → 1.25%**, p=1.5e-13 | ✅ |
| SRM (expected ratio 0.5) | PASS (clean randomization) | ratio 0.5000, p=0.996 → **PASS** | ✅ |
| SUTVA | PASS (no cross-arm users) | 0 crossover users → **PASS** | ✅ |
| Heterogeneity drivers | `mens`/`womens` purchase history | top drivers: **womens**, history, recency | ✅ |
| CUPED | applies, preserves ATE | applied, ATE +0.770 → +0.768 | ✅ |

**Headline:** the pipeline reproduces Hillstrom's published visit/conversion
lifts essentially exactly, the effect estimate matches a direct computation to
numerical precision, SRM correctly confirms clean randomization, and the causal
forest independently rediscovers the documented purchase-history heterogeneity.

### Honest caveats (where the dataset limits us)

- **CUPED reduction ≈ 0% — and that is the correct result.** `spend` is ~99%
  zeros over the two-week window and `history` (prior-year spend) correlates with
  it at only r≈0.02, so there is no pre-period variance for CUPED to remove. We
  validate the properties CUPED *must* hold (it applies cleanly and preserves the
  ATE) rather than a manufactured reduction. CUPED's value is real but
  data-dependent; this dataset simply lacks a strong pre-period predictor.
- **Sequential & novelty checks are not validated here.** The dataset has no
  per-user timestamps (a constant date is synthesized), so the time-based modules
  have nothing meaningful to act on. **These are covered by the landing-page
  dataset below.**
- **`overall = "warn"` on ingest is expected:** dropping the women's arm leaves
  those rows with an unmapped (null) treatment, which the null check reports as a
  warning before dropping them — exactly the intended, transparent behavior.

---

## Udacity landing-page experiment (`landing_page/`)

**Dataset:** the widely-used Udacity *Analyze A/B Test Results* file
(`ab_data.csv`): ~294k e-commerce page visits over ~3 weeks of January 2017,
comparing a **new landing page** (treatment) to the **old page** (control) on
**conversion**. Every row carries a **real per-user timestamp**, so this dataset
covers exactly what Hillstrom cannot: the **sequential** and **novelty** modules.
Its documented conclusion is a **null result** — the new page is *not* better —
which makes it a calibration test: the pipeline must refuse to invent an effect.

**Run it:**
```bash
python validation/landing_page/fetch.py          # downloads to validation/data/
python validation/landing_page/run_validation.py # runs modules, prints PASS/FAIL
```

### Results — obtained vs. published (11/11 checks pass)

Ingestion: 294,478 raw rows → 290,585 after dropping 3,893 group/landing-page
mismatch rows (in the adapter) → 290,584 after ingest removes 1 duplicate user.
Treatment labels (`control`/`treatment`) are auto-detected.

| Check | Published / expected | Obtained | Status |
|-------|----------------------|----------|--------|
| Mismatch rows dropped | 3,893 | **3,893** | ✅ |
| Label auto-encoding + dedup | control/treatment, 1 dup removed | auto_label, 290,585 → 290,584 | ✅ |
| Control conversion | ~12.04% | **12.04%** | ✅ |
| Treatment conversion | ~11.88% | **11.88%** | ✅ |
| Effect == direct `groupby` | exact | −0.001578 == −0.001578 | ✅ |
| Test chosen for binary metric | two-proportion z | two-proportion z | ✅ |
| **Calibration: new page NOT significant** | fail to reject null | −0.16 pp, p=0.190, 95% CI [−0.39, 0.08] pp | ✅ |
| SRM (expected ratio 0.5) | PASS (clean ~50/50) | ratio 0.5001, p=0.947 → **PASS** | ✅ |
| **Novelty (real timestamps)** | no time-varying effect | early −0.0003 vs late −0.0028, p=0.300 → **PASS** | ✅ |
| **Sequential (real timestamps)** | true null never crosses | 290,584 continuous peeks, never crossed, final p=1.0 | ✅ |
| Sequential power sanity | a real lift *does* cross | +2.3 pp injected → crosses at n=7,765 | ✅ |

**Headline:** on a real timestamped experiment with a documented null result, the
pipeline (a) reproduces the published ~12.0% / ~11.9% conversion rates exactly,
(b) correctly declines to call the new page a winner, and (c) — the reason this
dataset exists — runs the **sequential mSPRT over all ~290k observations without a
single false rejection under continuous monitoring**, while the novelty check
finds no time-varying effect. A labeled semi-synthetic perturbation (a real
+2.3 pp lift) confirms the sequential test *does* fire on genuine signal, so the
"never crosses" result is real type-I control, not an inert test.

### Honest caveats (where the dataset limits us)

- **No usable pre-treatment covariates.** `landing_page` is collinear with the
  arm, so CUPED and HTE are not exercised here — those are covered by Hillstrom.
- **Teaching dataset, not a peer-reviewed publication.** The "published" figures
  are the canonical, widely-reproduced results of the Udacity project, which we
  recompute directly from the file.
- **The novelty split is coarse** (median timestamp). "Pass" means no strong
  evidence of a time-varying effect, not proof that none exists.
