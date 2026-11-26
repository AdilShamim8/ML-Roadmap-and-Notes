# Probability Cheat Sheet

One page. The rules, the famous distributions' facts, and the puzzle patterns — including the combinatorics that make them computable on a whiteboard.

## The axioms and the four rules
```
0 ≤ P(A) ≤ 1;  P(S) = 1;  P(A∪B) disjoint = P(A)+P(B)
Complement:    P(Aᶜ) = 1 − P(A)
Union:         P(A∪B) = P(A) + P(B) − P(A∩B)
Conditional:   P(A|B) = P(A∩B)/P(B)
Independence:  P(A∩B) = P(A)P(B)   ⟺ P(A|B) = P(A)
```
**Independence vs mutually exclusive — the classic confusion:** mutually exclusive events *interact* (one excludes the other); independent events co-occur proportionally. Two events with positive probability cannot be both.

## Law of total probability & Bayes
```
P(B) = Σ P(B|Aᵢ)P(Aᵢ)                    — the denominator, pre-computed
P(A|B) = P(B|A)P(A) / P(B)              — Bayes
P(Aᶜ|B) vs P(A|B): the complement ratio = P(B|Aᶜ)P(Aᶜ) : P(B|A)P(A)
```
**Base-rate discipline:** with prior 0.1% and 99% accurate test, the posterior is ~9% — the denominator is dominated by false positives from the huge healthy class. Rare-event alarms are mostly false *by arithmetic, not by bad design*.

## Expectation & variance algebra
```
E[c] = c;   E[cX] = cE[X];   E[X+Y] = E[X]+E[Y]  (ALWAYS — even dependent!)
Var(cX) = c²Var(X)
Var(X+Y) = Var(X)+Var(Y) + 2Cov(X,Y)     — independence drops the covariance
E[XY] = E[X]E[Y] only if independent
Var(X) = E[X²] − (E[X])²
Stein: E[g(X)(X−μ)] = σ²E[g'(X)] — for the "E[X·(X−μ)]" family
```
**Why you care in ML:** bias-variance decomposition, portfolio/ensemble variance reduction, and gradient-descent noise are all just this algebra.

## Counting (the puzzle engine)
```
Permutations (ordered):      n! / (n−k)!
Combinations (unordered):    C(n,k) = n!/(k!(n−k)!)
With replacement:            nᵏ
Stars and bars (k identical into n bins): C(n+k−1, k−1)
"At least one":              1 − P(none)      ← the single most useful trick
Complement of matching:      derangements !n ≈ n!/e
```

## The famous puzzles, formula-level
| Puzzle | Method | Answer |
| --- | --- | --- |
| Coin flips until first head | Geometric, E = 1/p | 2 |
| Flips until HH vs HT | Patterns in sequences | HH = 6, HT = 4 (asymmetry!) |
| P(shared birthday, k people) | 1 − Π(365−i)/365 | 50.7% at k=23; 99.9% at k=60 |
| P(at least one 6, 4 dice) | 1 − (5/6)⁴ | 51.8% |
| Monty Hall | Switching wins iff first pick wrong | 2/3 vs 1/3 |
| Two children, one is a girl | Sample-space pruning | 1/2 (girl known *specific*) vs 1/3 (at-least-one phrasing) |
| 100 prisoners light bulb | Strategy loops | ~31% survival |
| Expected value of max of 2 dice | Enumerate or E[max] = Σ P(max ≥ k) | 161/36 ≈ 4.47 |

## Distributions — the fact sheet
| Distribution | PMF/PDF core | Mean | Variance |
| --- | --- | --- | --- |
| Bernoulli(p) | P(1)=p | p | p(1−p) |
| Binomial(n,p) | C(n,k)pᵏ(1−p)ⁿ⁻ᵏ | np | np(1−p) |
| Geometric(p) | (1−p)ᵏ⁻¹p | 1/p | (1−p)/p² |
| Poisson(λ) | e⁻λλᵏ/k! | λ | **λ** (mean = variance!) |
| Uniform(a,b) | 1/(b−a) | (a+b)/2 | (b−a)²/12 |
| Exponential(λ) | λe^(−λx) | 1/λ | 1/λ² (memoryless) |
| Normal(μ,σ²) | (1/σ√2π)e^(−(x−μ)²/2σ²) | μ | σ² |
| Beta(α,β) | x^(α−1)(1−x)^(β−1)/B | α/(α+β) | conjugate to Binomial |

**The 68–95–99.7 rule:** ±1σ = 68%, ±2σ = 95%, ±3σ = 99.7%. Standardization: z = (x−μ)/σ.

## Inequalities (when exact is impossible)
```
Markov:   P(X ≥ a) ≤ E[X]/a
Chebyshev: P(|X−μ| ≥ kσ) ≤ 1/k²         — any distribution, k σ away
Jensen:   E[φ(X)] ≥ φ(E[X]) for convex φ (e.g., E[X²] ≥ (E[X])²)
```
Chebyshev is the bound to quote when someone asks "what if it's NOT normal?"

## Conditional-probability traps
- **Gambler's fallacy vs memorylessness:** fair coins have no memory; P(HH...H then T) = 1/2. The *waiting time* distribution resets every flip.
- **Selection bias:** "average of observed" ≠ "average" — conditioning on observation changes the distribution (survivorship, M&M problems).
- **Simpson's reversal** lives here too: a conditional trend can flip under a different conditioning set.
- **Confusion of the inverse:** P(test+|disease) ≠ P(disease|test+) — the medical-test question, every single time.

## Simulation discipline (the interview flex)
When a puzzle resists analysis: **simulate it.** 100k trials of any of the above takes 3 lines and lands within ±0.3% of the analytic answer — the notebooks do exactly this for the birthday problem, Monty Hall, and the geometric expectation. *"I'd simulate to sanity-check"* is a sentence interviewers remember.
