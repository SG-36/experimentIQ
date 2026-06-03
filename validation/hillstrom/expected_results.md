# Hillstrom e-mail experiment — expected results (ground truth)

Source: Kevin Hillstrom, *MineThatData E-Mail Analytics and Data Mining
Challenge* (2008). 64,000 customers randomized roughly evenly into three arms:
**No E-Mail**, **Men's E-Mail**, **Women's E-Mail**. Outcomes measured over the
following two weeks: `visit` (binary), `conversion` (binary), `spend` (USD).

These figures are the widely-reproduced values from Hillstrom's analysis and
countless replications. Our validation checks two things:

1. **Exactness** — our pipeline's effect estimate must match a direct
   `groupby` computation on this exact file (to numerical precision).
2. **Published agreement** — those values must match the canonical figures
   below (within a small tolerance).

## Visit rate by arm (canonical)

| Arm | Visit rate |
|-----|-----------|
| No E-Mail | ~10.6% |
| Men's E-Mail | ~18.3% |
| Women's E-Mail | ~15.1% |

**Headline finding:** sending e-mail produces a large, highly significant lift
in visit rate. Men's E-Mail vs No E-Mail ≈ **+7.7 percentage points**
(p ≪ 0.001). This is the primary result we reproduce.

## Conversion rate by arm (canonical)

| Arm | Conversion rate |
|-----|-----------------|
| No E-Mail | ~0.57% |
| Men's E-Mail | ~1.25% |
| Women's E-Mail | ~0.88% |

E-mail roughly doubles conversion; the lift is smaller in absolute terms but
significant.

## Randomization

Assignment is ~1/3 per arm by design, so within any two-arm contrast the split
is ~50/50. **SRM must PASS** against an expected ratio of 0.5.

## Heterogeneity (documented)

The documented interaction is that **Men's E-Mail is most effective for
customers with a men's-purchase history, and Women's E-Mail for women's-history
customers**. A causal-forest CATE model should therefore surface `mens` /
`womens` (purchase-history flags) among the top heterogeneity drivers.

## Modules not exercised here

The dataset has **no per-user timestamps** (a constant date is synthesized), so
the **sequential** and **novelty** checks are not meaningfully validated on this
dataset — those are covered by the [landing-page dataset](../landing_page/expected_results.md).
CUPED is validated on `spend` using `history` (prior-year spend) as the
pre-experiment covariate (the reduction is ~0% here, which is the correct result
for this data — see `validation/README.md` for why).
