# Statistics Cheat Sheet

One page. The definitions, formulas, and decision rules that come up in every ML screen.

## Descriptives
| Statistic | Use / trap |
| --- | --- |
| Mean | symmetric data; dragged by outliers |
| Median | skewed data / outliers — report this when skew > ~1 |
| Mode | categorical |
| Variance / SD | spread; sample uses **n−1** (Bessel) |
| IQR | robust spread = Q3−Q1; outliers beyond Q1−1.5·IQR / Q3+1.5·IQR |
| Skew | right (mean > median) vs left; log-transform fixes right skew |

## Distributions
| Situation | Distribution |
| --- | --- |
| # successes in n trials, p known | Binomial(n, p) |
| Rare-event counts per unit | Poisson(λ) — mean = variance |
| Time between rare events | Exponential(λ) — **memoryless** |
| Sums/means of many causes | Normal (via CLT) |
| Waiting for k-th event | Gamma(k, λ) |
| Uniform ignorance | Uniform(a, b) |
| Counts, overdispersed | Negative Binomial (Poisson with extra variance) |

## The central results
- **LLN:** sample mean → μ as n grows.
- **CLT:** sampling distribution of the mean → Normal(μ, σ/√n) regardless of population shape (n ≳ 30).
- **SD vs SE:** SD = spread of data; **SE = SD/√n** = spread of the *mean*.

## Sampling & intervals
```
95% CI for a mean:  x̄ ± 1.96 · s/√n     (t-distribution when n < 30)
Proportion CI:      p̂ ± 1.96 · √(p̂(1-p̂)/n)
CI interpretation:  "95% of intervals constructed this way contain the true parameter"
                    (NOT "95% chance the parameter is in this one")
Bootstrap:          resample w/ replacement B times → percentile interval (any statistic)
```

## Hypothesis testing
```
H0 / H1 (alternative)
α = P(reject H0 | H0 true)      — Type I, conventionally 0.05
β = P(fail to reject | H1 true) — Type II; Power = 1 − β, conventionally ≥ 0.8
p-value = P(data this extreme | H0)   — NOT P(H0 | data)
```
| Test | Use |
| --- | --- |
| t-test (2-sample) | compare two means |
| paired t-test | before/after on same units |
| z-test (proportions) | A/B conversion rates |
| χ² test | categorical association / goodness of fit |
| ANOVA | 3+ group means |
| Mann-Whitney / Wilcoxon | non-parametric versions (rank-based) |

**Effect size always travels with significance:** Cohen's d = (μ₁−μ₂)/σ; a significant p with tiny effect is a business "no."

## A/B test arithmetic (the formulas to know cold)
```
SE of proportion:  √(p̂(1-p̂)/n)
Two-prop z:        z = (p̂₂-p̂₁) / √(p̄(1-p̄)(1/n₁+1/n₂))
Sample size:       n ≈ 16·p(1-p)/δ²   (α=0.05, power=0.8, per group)
                       — halve δ → 4× the sample
Peeking:           sequential looks inflate α (0.05 → ~0.25); fix horizon or use
                    sequential design (alpha-spending / mSPRT)
```

## Bayes
```
P(A|B) = P(B|A)·P(A) / P(B)
Posterior ∝ Likelihood × Prior
```
**The medical-test classic:** prevalence 0.1%, sensitivity 99%, specificity 99% → P(disease|+) = 0.99·0.001 / (0.99·0.001 + 0.01·0.999) ≈ **9%**. Base rates dominate rare-event posteriors.

## Correlation & friends
```
Pearson r:  linear association ∈ [-1, 1];  r² = variance explained
Spearman:   rank correlation (monotonic, outlier-robust)
r ≠ causation: confounders, selection, chance (ice cream ↔ drownings via summer)
Simpson's paradox: subgroup trends can REVERSE when pooled — check for
                   lurking grouping variables before quoting r
```

## Regression stats
```
R² = 1 - SSE/SST         — vs the mean-baseline; rises with ANY added feature
Adjusted R²              — penalizes parameters (in-sample honesty)
Omitted variable bias    — relevant missing feature correlated with included ones
Heteroscedasticity       — non-constant error variance → SEs wrong; log y or robust SEs
Multicollinearity        — correlated features → unstable coefficients; VIF > ~10 flags
```

## The brainteaser answers (verified by simulation in the notebooks)
| Puzzle | Answer |
| --- | --- |
| Expected flips to first head | 1/p = 2 |
| P(shared birthday, 23 people) | ≈ 50.7% |
| Monty Hall — switch? | Yes: 2/3 vs 1/3 |
| P(at least one 6 in 4 rolls) | 1 − (5/6)⁴ ≈ 51.8% |
| 100 people board, seat puzzle | 1/2 |

## Vocabulary one-liners
| Term | 15-second version |
| --- | --- |
| iid | independent + identically distributed; breaks with time series, repeated users, leakage |
| Confidence vs prediction interval | mean vs single observation; PI always wider |
| Simpson's paradox | pooled trend reverses subgroups |
| Spurious correlation | correlation via confounder or chance |
| Power analysis | pick n to detect effect d with prob 1−β — before the experiment |
| Parametric vs non-parametric | assumes a shape vs ranks — power vs safety |
| Conjugate prior | prior family that keeps the posterior in-family (Beta-Binomial) |
| MLE | parameters maximizing likelihood of observed data |
