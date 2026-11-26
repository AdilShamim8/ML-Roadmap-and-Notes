# Question Bank Index — Every Question, By Round and Difficulty

The complete index of interview questions across this section: **~200 questions**, organized the way you'll meet them — by round type. Use it as the diagnostic (Day 0: attempt cold, every miss maps to a notebook), the spaced-repetition deck (any question you fumble gets re-tested tomorrow), and the final-week gauntlet (Day 4 of the [Final Week Plan](../6.%20The%20Final%20Push/Final%20Week%20Preparation%20Plan.md)).

**Difficulty: ● easy (screen), ●● medium (on-site), ●●● hard (senior bar).** Every entry links to the notebook with the full answer.

---

## 1. Statistics & Probability (Fundamentals Review)

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 1 | Mean vs median — when do you report which? | ● | [Statistics & Probability](../1.%20Fundamentals%20Review/Statistics%20and%20Probability.ipynb) |
| 2 | Name the distribution for [count/rare-event/wait-time] scenarios | ● | 〃 |
| 3 | State the CLT and why it matters for ML | ●● | 〃 |
| 4 | Standard deviation vs standard error | ● | 〃 |
| 5 | What is a p-value? (precisely) | ●● | 〃 |
| 6 | Type I vs Type II error, power, and the tradeoff | ● | 〃 |
| 7 | Two-proportion z-test by hand (A/B test) | ●● | 〃 |
| 8 | What does a 95% CI mean? (the correct phrasing) | ●● | 〃 |
| 9 | The 99%-accurate medical test question (Bayes) | ●● | 〃 |
| 10 | Which operations commute for E and Var? | ●● | 〃 |
| 11 | Expected coin flips until first head | ●● | 〃 |
| 12 | Birthday problem with 23 people | ●● | 〃 |
| 13 | Monty Hall — should you switch, and why? | ●● | 〃 |
| 14 | Give an example where correlation misleads (Simpson's) | ●●● | 〃 |
| 15 | CI for a median with no formula — bootstrap | ●● | 〃 |
| 16 | Why n−1 in sample variance? | ●● | 〃 |
| 17 | LLN vs CLT | ●● | 〃 |

## 2. Linear Algebra (Fundamentals Review)

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 18 | Dot product / cosine similarity — and when cosine beats Euclidean | ● | [Linear Algebra](../1.%20Fundamentals%20Review/Linear%20Algebra.ipynb) |
| 19 | L1 vs L2 norms and the geometry of their unit balls | ●● | 〃 |
| 20 | What does a matrix do geometrically? Determinant as volume? | ●● | 〃 |
| 21 | Rank — say it precisely; why rank-deficiency breaks OLS | ●● | 〃 |
| 22 | Eigenvectors/eigenvalues, and the symmetric-matrix guarantee | ●● | 〃 |
| 23 | Explain SVD; why is it everywhere? (Eckart–Young) | ●●● | 〃 |
| 24 | Walk through PCA; what is a "component"? | ●● | 〃 |
| 25 | Why does λI make XᵀX invertible (ridge)? | ●●● | 〃 |
| 26 | Least squares geometrically (projection, Xᵀr = 0) | ●●● | 〃 |
| 27 | Manhattan vs Euclidean vs Mahalanobis | ●● | 〃 |
| 28 | Why can't you invert a non-square matrix? Pseudo-inverse? | ●● | 〃 |
| 29 | What does condition number tell you? | ●●● | 〃 |

## 3. Python & Coding Patterns (Coding Rounds)

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 30 | Two-pointer: pair with target sum in O(n) | ● | [Python Coding Round](../2.%20Coding%20Rounds/Python%20Coding%20Round.ipynb) |
| 31 | Sliding window: longest substring without repeats | ●● | 〃 |
| 32 | Prefix sums: O(1) range-sum queries | ● | 〃 |
| 33 | Top-K frequent elements with a heap | ●● | 〃 |
| 34 | Binary search on the answer (first bad version) | ●●● | 〃 |
| 35 | Floyd's cycle detection (fast/slow pointers) | ●● | 〃 |
| 36 | Merge overlapping intervals | ●● | 〃 |
| 37 | BFS shortest path in an unweighted graph | ●● | 〃 |
| 38 | Palindrome / anagram / FizzBuzz one-liners | ● | 〃 |
| 39 | Kadane's max subarray | ●● | 〃 |
| 40 | DP with O(1) space (climbing stairs, house robber) | ●● | 〃 |
| 41 | Backtracking: subsets and permutations | ●●● | 〃 |
| 42 | In-place linked-list reversal | ● | 〃 |
| 43 | Spiral matrix order | ●● | 〃 |
| 44 | Events-per-user-per-hour grouping drill | ●● | 〃 |

## 4. Python Puzzles & Edge Cases

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 45 | The mutable default argument — what happens and why | ● | [Python Puzzles](../2.%20Coding%20Rounds/Python%20Puzzles%20and%20Edge%20Cases.ipynb) |
| 46 | `is` vs `==`; integer caching; string interning | ●● | 〃 |
| 47 | Closures and late binding — the loop-lambda trap | ●● | 〃 |
| 48 | `False == False in [False]` — comparison chaining | ●● | 〃 |
| 49 | Shallow vs deep copy — nested lists | ● | 〃 |
| 50 | Why is `+=` string concat O(n²)? | ●● | 〃 |
| 51 | Four counting idioms ending at defaultdict | ● | 〃 |
| 52 | The GIL — can threads speed up CPU work? (3.13 note) | ●● | 〃 |
| 53 | Runtime class patching; shared class attributes | ●● | 〃 |
| 54 | Why can't lists be dict keys? (hashability) | ● | 〃 |
| 55 | Sort stability + multi-key sorts | ● | 〃 |
| 56 | Generator vs list comprehension — when and why | ● | 〃 |

## 5. Pandas

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 57 | loc vs iloc; combining filters (& with parens) | ● | [Pandas Interview Qs](../2.%20Coding%20Rounds/Pandas%20Interview%20Questions.ipynb) |
| 58 | Groupby split-apply-combine with multiple aggregations | ● | 〃 |
| 59 | transform vs apply (group-mean imputation) | ●● | 〃 |
| 60 | Melt/pivot reshape round-trip | ●● | 〃 |
| 61 | Merge with validate=; the row-explosion check | ●● | 〃 |
| 62 | Grouped rolling windows (the sort-first trap) | ●● | 〃 |
| 63 | value_counts / cut / qcut binning | ● | 〃 |
| 64 | .map(function) vs .str vectorized — with timing | ●● | 〃 |
| 65 | The null-profiling report in four lines | ● | 〃 |
| 66 | Datetime .dt feature extraction | ● | 〃 |
| 67 | Top-N per group (sort + head) | ●● | 〃 |
| 68 | Fluent query/assign chaining | ●● | 〃 |
| 69 | "Doesn't fit in RAM" — usecols, chunksize | ●● | 〃 |

## 6. NumPy

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 70 | Broadcasting rules, right-to-left; the (3,4)+(3,) fix | ●● | [NumPy Interview Qs](../2.%20Coding%20Rounds/NumPy%20Interview%20Questions.ipynb) |
| 71 | Why is NumPy faster (three stacked reasons)? | ●● | 〃 |
| 72 | axis=0 vs axis=1; keepdims | ● | 〃 |
| 73 | Views vs copies; np.shares_memory | ●● | 〃 |
| 74 | Boolean masking + np.where / np.select | ● | 〃 |
| 75 | reshape(-1); ravel vs flatten; transpose shares memory | ●● | 〃 |
| 76 | solve vs inv — the numerical advice | ●● | 〃 |
| 77 | NaN-safe ops; interpolation of NaNs | ● | 〃 |
| 78 | One-liners: z-scores, ReLU, stable softmax, pairwise distances, one-hot | ●●● | 〃 |
| 79 | int8/float16 precision traps | ●● | 〃 |
| 80 | arange vs linspace; the new random API (default_rng) | ● | 〃 |

## 7. SQL

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 81 | WHERE vs HAVING | ● | [SQL Interview Qs](../2.%20Coding%20Rounds/SQL%20Interview%20Questions.ipynb) |
| 82 | The NOT IN + NULL trap | ●● | 〃 |
| 83 | LEFT JOIN anti-join (departments with no employees) | ●● | 〃 |
| 84 | Self-join for employee/manager pairs | ●● | 〃 |
| 85 | Subquery vs CTE vs correlated — same question thrice | ●● | 〃 |
| 86 | NULL logic: COUNT, COALESCE, NULLIF divide-by-zero | ●● | 〃 |
| 87 | CASE-based pivot (conditional aggregation) | ●● | 〃 |
| 88 | Top-N per group, pre-window and window versions | ●●● | 〃 |
| 89 | ROW_NUMBER vs RANK vs DENSE_RANK | ●● | [SQL Joins & Windows](../2.%20Coding%20Rounds/SQL%20Joins%20and%20Window%20Functions.ipynb) |
| 90 | LAG/LEAD for time-between-events and MoM growth | ●● | 〃 |
| 91 | Running totals / moving averages; ROWS vs RANGE | ●●● | 〃 |
| 92 | NTILE / PERCENT_RANK bucketing | ●● | 〃 |
| 93 | FIRST_VALUE / LAST_VALUE default-frame gotcha | ●●● | 〃 |
| 94 | Sessionization (flag + running sum, three CTEs) | ●●● | 〃 |
| 95 | Duplicate-key join explosion — causes and defenses | ●●● | 〃 |
| 96 | SQL window ↔ pandas translation | ●● | 〃 |

## 8. ML Coding (From Scratch)

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 97 | Implement k-means; validate vs sklearn | ●● | [ML Coding Round](../2.%20Coding%20Rounds/ML%20Coding%20Round.ipynb) |
| 98 | Implement KNN; distance choices | ● | 〃 |
| 99 | Best decision-tree split by information gain | ●●● | 〃 |
| 100 | Logistic regression with GD (gradients stated) | ●●● | 〃 |
| 101 | AUC via the rank formula (tie handling) | ●●● | 〃 |

## 9. ML Core Concepts

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 102 | Train/validation/test — what each is FOR; one-use rule | ● | [ML Core Concepts](../3.%20Machine%20Learning%20Interviews/ML%20Core%20Concepts.ipynb) |
| 103 | Overfitting: definition, diagnosis (the gap), three cures | ● | 〃 |
| 104 | What does CV actually estimate? (+ split variants) | ●● | 〃 |
| 105 | Covariate vs label vs concept drift (+ PSI) | ●●● | 〃 |
| 106 | Walk through the ML project lifecycle (with loops) | ●● | 〃 |
| 107 | Supervised vs unsupervised vs self-supervised (2026) | ● | 〃 |
| 108 | Parametric vs non-parametric — which do you pick? | ●● | 〃 |

## 10. Supervised Learning

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 109 | Linear regression assumptions, ranked by what they break | ●● | [Supervised Learning Qs](../3.%20Machine%20Learning%20Interviews/Supervised%20Learning%20Questions.ipynb) |
| 110 | Why is logistic regression called regression? | ● | 〃 |
| 111 | Cross-entropy vs MSE for classification | ●● | 〃 |
| 112 | SVM margins, C, the kernel trick (inner products) | ●●● | 〃 |
| 113 | KNN's three choices; curse of dimensionality | ●● | 〃 |
| 114 | Why "naive" Bayes works anyway; smoothing | ●● | 〃 |
| 115 | Imbalanced classification: the five-layer answer | ●●● | 〃 |

## 11. Trees & Boosting

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 116 | Gini vs entropy (computed) | ●● | [Trees & Boosting Qs](../3.%20Machine%20Learning%20Interviews/Trees%20and%20Boosting%20Questions.ipynb) |
| 117 | Why don't trees need scaling? (and no extrapolation) | ●● | 〃 |
| 118 | Bagging variance reduction, measured | ●●● | 〃 |
| 119 | Gradient boosting, the additive story, exactly | ●●● | 〃 |
| 120 | XGBoost vs LightGBM vs CatBoost | ●● | 〃 |
| 121 | Which hyperparameters matter, and how you tune them | ●● | 〃 |
| 122 | Gain vs permutation vs SHAP importance | ●●● | 〃 |

## 12. Unsupervised Learning

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 123 | Choosing k: elbow vs silhouette vs the null test | ●● | [Unsupervised Qs](../3.%20Machine%20Learning%20Interviews/Unsupervised%20Learning%20Questions.ipynb) |
| 124 | k-means vs hierarchical vs DBSCAN selection table | ●● | 〃 |
| 125 | PCA vs t-SNE vs UMAP vs autoencoders | ●● | 〃 |
| 126 | The four PCA follow-ups (centering, n-components, whiten, failure) | ●● | 〃 |
| 127 | Anomaly detection one-liners (iForest, OC-SVM, LOF) | ●● | 〃 |
| 128 | Support/confidence/lift | ● | 〃 |

## 13. Evaluation & Metrics

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 129 | ROC-AUC vs PR-AUC under imbalance | ●● | [Model Evaluation Qs](../3.%20Machine%20Learning%20Interviews/Model%20Evaluation%20Questions.ipynb) |
| 130 | Threshold selection as cost minimization | ●●● | 〃 |
| 131 | Calibration: what, when, how to fix | ●●● | 〃 |
| 132 | RMSE/MAE/R² and the outlier-leverage demo | ●● | 〃 |
| 133 | Leakage-aware evaluation rules (5) | ●●● | 〃 |
| 134 | Bootstrap CI on the headline metric | ●● | 〃 |
| 135 | Macro vs micro F1; kappa; MCC | ●● | 〃 |

## 14. Regularization & Bias-Variance

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 136 | The bias-variance decomposition, precisely | ●● | [Regularization & B-V](../3.%20Machine%20Learning%20Interviews/Regularization%20and%20Bias-Variance.ipynb) |
| 137 | L1 vs L2 geometry → sparsity vs stability | ●●● | 〃 |
| 138 | Choosing λ; the one-standard-error rule | ●● | 〃 |
| 139 | Double descent — when the U-curve lies | ●●● | 〃 |

## 15. Gradient Descent & Optimization

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 140 | Batch vs SGD vs mini-batch — what differs | ●● | [Gradient Descent](../3.%20Machine%20Learning%20Interviews/Gradient%20Descent%20from%20Scratch.ipynb) |
| 141 | Learning-rate pathologies and the 2/L bound | ●● | 〃 |
| 142 | Momentum: why it damps zig-zag | ●● | 〃 |
| 143 | Adam: what it adapts; AdamW distinction | ●●● | 〃 |
| 144 | Saddles vs local minima in deep nets | ●●● | 〃 |

## 16. Feature Engineering

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 145 | Categorical encoding decision tree | ●● | [Feature Engineering Qs](../3.%20Machine%20Learning%20Interviews/Feature%20Engineering%20Questions.ipynb) |
| 146 | Which scaler, and when scaling doesn't matter | ●● | 〃 |
| 147 | Target encoding without leakage (3 protections) | ●●● | 〃 |
| 148 | Cyclical datetime encoding | ● | 〃 |
| 149 | Leakage-safe lags (shift before rolling) | ●●● | 〃 |
| 150 | Feature selection: filter/wrapper/embedded + CV-inside | ●● | 〃 |

## 17. A/B Testing & Causal Inference

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 151 | Power analysis before the test (n for a lift) | ●● | [A/B Testing & Causal](../3.%20Machine%20Learning%20Interviews/A%20B%20Testing%20and%20Causal%20Inference.ipynb) |
| 152 | Why peeking breaks α (and the fixes) | ●●● | 〃 |
| 153 | CUPED variance reduction | ●●● | 〃 |
| 154 | Simpson's reversal on treatment effects | ●●● | 〃 |
| 155 | The causal toolkit one-liners (DiD, IV, RDD, IPW, uplift) | ●● | 〃 |
| 156 | p=0.07 — ship or kill? | ●● | 〃 |

## 18. Deep Learning

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 157 | Why nonlinearity is non-optional | ● | [Deep Learning Qs](../4.%20Deep%20Learning%20and%20Domains/Deep%20Learning%20Questions.ipynb) |
| 158 | Backprop equations for a 2-layer net; verify by finite differences | ●●● | 〃 |
| 159 | Vanishing/exploding gradients + the fixes | ●● | 〃 |
| 160 | Batch norm (train/inference asymmetry), dropout (inverted) | ●● | 〃 |
| 161 | Attention QKV — the equation and the story | ●●● | 〃 |
| 162 | CNN/RNN/transformer one-breath definitions | ●● | 〃 |
| 163 | Xavier vs He init | ●● | 〃 |

## 19. NLP & LLMs

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 164 | Tokenization tradeoffs (BPE spirit) | ●● | [NLP Qs](../4.%20Deep%20Learning%20and%20Domains/NLP%20Interview%20Questions.ipynb) |
| 165 | TF-IDF formula + intuition | ● | 〃 |
| 166 | Static vs contextual embeddings | ●● | 〃 |
| 167 | The fine-tuning ladder (SFT → LoRA → DPO); fine-tune vs RAG | ●● | 〃 |
| 168 | RAG's six components; hybrid retrieval + RRF | ●●● | 〃 |
| 169 | Why do LLMs hallucinate; temperature/top-p | ●● | 〃 |
| 170 | Cross-encoder vs bi-encoder (rerank vs retrieve) | ●● | 〃 |

## 20. Time Series

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 171 | Components; additive vs multiplicative | ● | [Time Series Qs](../4.%20Deep%20Learning%20and%20Domains/Time%20Series%20Interview%20Questions.ipynb) |
| 172 | Stationarity: ADF + KPSS, and fixing it (d) | ●● | 〃 |
| 173 | ARIMA (p,d,q) justification via ACF/PACF/AIC | ●● | 〃 |
| 174 | Time-series validation (rolling origin, no shuffling) | ●●● | 〃 |
| 175 | MASE and the naive-baseline discipline | ●● | 〃 |
| 176 | Forecasting with GBMs (the supervised conversion) | ●● | 〃 |

## 21. System Design

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 177 | The 6-step design frame | ●● | [ML System Design](../5.%20System%20Design/ML%20System%20Design.ipynb) |
| 178 | Design real-time fraud detection (label delay, latency budget) | ●●● | 〃 |
| 179 | Design a recommendation system (two-stage, position bias) | ●●● | 〃 |
| 180 | Design churn prediction (expected-value ranking) | ●● | 〃 |
| 181 | Training/serving skew — definition and fixes | ●●● | 〃 |
| 182 | Design a RAG system over 10M docs (chunk→embed→retrieve→rerank→generate) | ●●● | [LLM & RAG Design](../5.%20System%20Design/LLM%20and%20RAG%20System%20Design.ipynb) |
| 183 | RAG evaluation stack (retrieval first) | ●●● | 〃 |
| 184 | Prompt injection, ACLs, and the RAG security layer | ●●● | 〃 |

## 22. Behavioral & Project (Final Push)

| # | Question | Diff | Answer in |
| --- | --- | --- | --- |
| 185 | Tell me about yourself (the 90-second pitch) | ● | [Behavioral & STAR](../6.%20The%20Final%20Push/Behavioral%20and%20STAR%20Answers.md) |
| 186 | The failure story (failure→ownership→change→redemption) | ●● | 〃 |
| 187 | A disagreement with a teammate/PM | ●● | 〃 |
| 188 | Your biggest professional impact | ●● | 〃 |
| 189 | The weakness question (real weakness + system + win) | ●● | 〃 |
| 190 | Walk me through your project (the 5-why drill) | ●●● | [Deep-Dive Prep](../6.%20The%20Final%20Push/Project%20Deep-Dive%20Prep.md) |
| 191 | Why are you leaving / why this role? | ● | [Behavioral & STAR](../6.%20The%20Final%20Push/Behavioral%20and%20STAR%20Answers.md) |
| 192 | "Walk me through your process" (lifecycle question) | ●● | [ML Core Concepts](../3.%20Machine%20Learning%20Interviews/ML%20Core%20Concepts.ipynb) |

---

## How to Use This Bank

1. **Diagnostic (Day 0):** go section by section; for each question say the answer aloud in 60 seconds. Mark every question you can't — that's your personal weakness map.
2. **Spaced repetition:** re-test your marked questions tomorrow, in 3 days, in a week. The bank is static; your marks should move.
3. **Final-week gauntlet (Day 4):** the whole bank, cold, fast — 15 seconds per question. Anything still marked goes to the cheat-sheet reread.
4. **Pair study:** a friend reads the question, you answer, they score against the linked notebook's model answer. Swap. One hour of this beats three of solo rereading.

**Coverage check:** 192 indexed questions across 22 topics — the notebooks contain additional follow-up chains beyond what's listed here; the bank is the rehearsal surface, the notebooks are the depth.
