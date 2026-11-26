# Machine Learning Fundamentals Cheat Sheet

One page. The theory tables that span every ML round — the one sheet to reread the morning of the loop.

## The learning-problem taxonomy
| Setting | Data | Examples |
| --- | --- | --- |
| Supervised | (X, y) pairs | churn, fraud, forecasting |
| Unsupervised | X only | clustering, PCA, anomaly, association rules |
| Semi-supervised | few labels + lots of unlabeled | self-training, label propagation |
| Self-supervised | labels manufactured from data | next-token, masked modeling, contrastive (the LLM engine) |
| RL / bandits | delayed reward via interaction | recommendations, control |

## Bias-variance, one breath each
```
Bias     = systematic error from wrong assumptions (linear model on curve)
Variance = sensitivity to which sample you drew (deep tree)
Noise    = irreducible σ²
Test error = Bias² + Variance + Noise
Cures: more data, less capacity, regularization, CV, ensembling
Double descent: past the interpolation threshold, error can fall again (over-parameterized regime)
```

## Model families, the decision table
| Family | Strengths | Weaknesses | Reach when |
| --- | --- | --- | --- |
| Linear/logistic | interpretable, fast, baseline | linear boundary | always first |
| Trees | mixed features, no scaling, interactions | unstable, no extrapolation | nonlinear tabular |
| RF (bagging) | robust, OOB validation, low tuning | bigger, less interpretable | strong default |
| GBDT (XGB/LGBM/Cat) | SOTA tabular, handles NaN | tuning, extrapolation | production tabular |
| SVM | margins, kernels, small-n | n² kernel, no probs | small nonlinear n |
| KNN | no training, local | slow serve, curse of dim | small data, smooth local |
| Naive Bayes | one pass, sparse text | correlated features, uncalibrated | text baseline |
| Deep nets | images/text/audio, transfer | data+compute hungry | perceptual/sequence data |

## Regularization
```
L1 (lasso):  Σ|w| — diamond corners → SPARSITY (selection); unstable under correlation
L2 (ridge):  Σw²  — disk → SHRINKAGE (stability); keeps all features
ElasticNet:  blend — sparsity + correlation tolerance
λ by CV; one-standard-error rule (simplest within 1 SE of best)
Ridge geometry: (XᵀX+λI) — eigenvalues shift up by λ, always invertible
Scale features BEFORE penalizing (penalty is scale-blind)
Also regularization: early stopping, dropout, augmentation, tree depth limits, label smoothing
```

## Validation (the section that fails candidacies)
```
train      → fit weights
validation → every decision you make (model choice, hyperparams, early stop)
test       → ONE use, at the end — "how will this pipeline do in production?"
CV estimates the MODELING PROCEDURE, not a specific fit (± std across folds)
Splits must match the data-generating process:
  classification imbalance → stratified · rows share entities → GroupKFold
  time series → TimeSeriesSplit / rolling origin · never shuffle temporal data
Everything data-dependent goes INSIDE the pipeline (scaler, imputer, selector, encoder)
```

## Metrics, matched to the problem
| Situation | Metric | Why |
| --- | --- | --- |
| Balanced classes, symmetric cost | accuracy | simple, honest here |
| Imbalanced, positives matter | PR-AUC, recall@precision | ROC-AUC flatters under imbalance |
| Threshold to be set vs cost | ROC/PR curve + cost grid | threshold is a business decision |
| Probabilities consumed | calibration (Brier, reliability) + log loss | AUC 0.95 can be miscalibrated |
| Regression, outlier-sensitive | MAE / Huber | robust |
| Regression, punish big misses | RMSE | quadratic penalty |
| Regression, comparable across series | MASE | vs naive baseline |
| Multi-class imbalance | macro-F1 | minority classes count |
| Ranking / retrieval | MRR, NDCG, recall@k | position-aware |

## Imbalance, the five-layer answer
1. Measure right (PR not ROC) 2. resample **inside CV folds** only 3. `class_weight='balanced'` 4. move threshold by cost 5. maybe ranking + human review, not classification.

## The drift trilogy
```
covariate:  P(X) changes, P(y|X) same → performance degrades at edges; retrain
label:      P(y) changes, P(y|X) same → thresholds/calibration stale; recalibrate
concept:    P(y|X) changes → the mapping is wrong; retrain + rethink features
Monitor: PSI per feature (<0.1 stable, >0.25 act) + metrics where labels arrive
```

## Leakage, the five doors (all demonstrated in the repo's notebook)
```
1. target-derived features        ("days_to_churn" in a churn model)
2. fitting transformers pre-split (scaler/imputer/encoder on full data)
3. feature selection outside CV   (mutual-info on all data, then split)
4. temporal inversion             (future info in lags; rolling without shift)
5. grouped rows split randomly    (same user in train and test)
Rule: if the number looks too good, one of these five doors is open.
```

## Gradient descent, the update
```
w ← w − η∇L
batch (exact, slow) · SGD (noisy, escapes saddles) · mini-batch (production)
momentum: v ← βv + g; w ← w − ηv   (damps zig-zag, β=0.9)
Adam: EMAs of g and g², bias-corrected, per-param steps (β₁=0.9, β₂=0.999)
AdamW: Adam + DECOUPLED weight decay
stability: η < 2/L (L = largest eigenvalue of the Hessian/curvature scale)
schedule: warmup → cosine; lr scales ~linearly with batch size
```

## Feature engineering, the priority order
1. Domain-informed ratios/interactions (price-per-sqft) 2. time features (lags, recency, cyclical sin/cos) 3. aggregates per entity (counts, means) 4. missingness indicators 5. encoding (one-hot <15 levels; target-encode high-cardinality **in-pipeline**) 6. transforms (log1p skewed). Selection: embedded (L1/tree importance) inside CV; permutation/SHAP for honesty.

## The morning-of-the-loop 15-second drill
Cover the right column; recite: CV estimates what? (the procedure) · L1 buys what? (sparsity) · PR vs ROC when? (imbalance) · MASE < 1 means? (beats naive) · skew vs drift vs bug triage order? (reproduce offline → PSI → label delay) · ridge makes what invertible? (XᵀX+λI) · attention formula? (softmax(QKᵀ/√d)V) · the five leakage doors? (derived, pre-split, pre-CV-select, temporal, grouped). All fifteen right — you're ready.
