# Project Deep-Dive Prep — Surviving the Technical Interrogation

"Walk me through your project" is where interviewers spend 30+ minutes drilling five levels deep. The candidates who fail aren't dishonest — they're *unprepared for depth*: they built something once and never rehearsed explaining every decision to a skeptic. This guide applies the **5-why drill** to this repo's two capstones so you can study the pattern, then apply it to your own work.

**The rule of the drill:** for every line on your resume, be ready to answer **five consecutive "why?" questions** without hand-waving. Each level: Level 1 = what you did, Level 2 = why that choice, Level 3 = what alternatives you rejected and why, Level 4 = what breaks / when it fails, Level 5 = what you'd change with more time/resources.

---

## Project 1 — End-to-End ML Project (Titanic Survival Classification)

**The one-liner (memorize):** "A full-lifecycle classification project — EDA through serving-ready pipeline — where every preprocessing step lives inside the cross-validation folds, with a five-model bake-off and honest error analysis. 0.79 CV AUC."

### The 5-Why Chain, Rehearsed

**L1 — What did you build?**
A binary classification pipeline for Titanic survival: 891 rows, mixed numeric/categorical features, ~20% missing ages. ColumnTransformer-based preprocessing, five model families compared, stratified 5-fold CV, per-segment error analysis. Data and seeds shipped in-repo for one-command reproduction.

**L2 — Why [leakage-safe pipelines / that model / that metric]?**
- *Why all transformers inside CV folds?* "Because fitting the scaler/imputer/encoder on the full dataset before splitting lets validation data leak into training. I measured this illusion in my leakage notebook: up to 15 AUC points of fake skill. Inside the pipeline, each fold fits its own preprocessing — the honest number is the only number."
- *Why [winning model family]?* "Gradient boosting gave the best CV score (state the number), but the margin over logistic regression was within one standard deviation of fold variance — so I report both and note that on 891 rows, the simpler model is the defensible default. The tie itself is a finding."
- *Why AUC?** "The classes are ~62/38 — mild imbalance. I track accuracy for communication and AUC for threshold-free comparison; PR-AUC would matter at 1% positives, here it doesn't change the story."

**L3 — What alternatives did you reject?**
- "Deep learning: 891 rows can't support it — I'd be memorizing noise; even a small net overfit immediately in early trials. Trees/linear models are the right complexity class for this data size."
- "Dropping the Age column (20% missing): rejected — missingness itself was predictive (I added a missing-indicator and imputed by class-and-sex medians *within folds*). Dropping would have discarded signal."
- "Feature engineering extravaganzas (ticket patterns, cabin decks): rejected — each addition must beat the pipeline without it in CV, and on this sample size most added variance, not signal."

**L4 — Where does it break?**
- "It's trained on 1912 demographics — it encodes 'women and children first' as signal; any modern application would be concept drift by definition. As a portfolio piece that's fine; as a production pattern, the *pipeline discipline* transfers, not the model."
- "With ~180 positive-class rows, every CV number carries wide uncertainty — I report fold std, and anyone quoting this project's AUC to three decimals is over-reading it."
- "If the serving traffic had feature distributions wildly different from training (only third-class passengers?), the calibration would drift — I'd monitor PSI per feature."

**L5 — What would you change with more time?**
"Three things, in order: (1) nested CV for unbiased model selection — I used the same folds to compare and to estimate; nested is the correct-but-expensive version. (2) SHAP-based error analysis per segment instead of aggregate confusion matrices. (3) A tiny FastAPI serving wrapper with a model-registry pattern — the project stops at the pipeline, and the serving story is where I'd extend it."

### The Hostile-Probe Appendix (Questions They Actually Ask)

- *"Why not SMOTE for the imbalance?"* — "62/38 isn't the regime where resampling pays; SMOTE on mixed-type features with 20% missingness adds synthetic noise to solve a problem the class weights already address. I'd revisit at 90/10."
- *"Your AUC interval overlaps the logistic model — so why ship the tree?"* — "Exactly the right challenge; my honest answer is 'on this data, don't — ship the simpler model.' The tree wins the bake-off only if I trust the point estimate, which I don't at this sample size."
- *"What feature mattered most, and how do you know it's causal?"* — "Sex and class dominate every importance measure — but importance is predictive credit, not causation, and gain-based importance overstates correlated features. I'd answer 'association,' never 'causal,' without an experiment."

---

## Project 2 — End-to-End Regression Project (California Housing)

**The one-liner:** "A regression pipeline predicting median house values by district on 20,640 rows: spatial-feature engineering, target log-transformation justified by skew, leakage-safe CV, and an error analysis that found the model's failure geography. RMSE in log-space, honest."

### The 5-Why Chain, Rehearsed

**L1 — What did you build?**
District-level regression on the California housing dataset (20,640 districts, 8 features): EDA revealing heavy target skew and geographic structure, engineered features (rooms-per-household, bedroom ratio, proximity-to-ocean encoding), gradient-boosted trees vs regularized linear baselines, 5-fold CV, and a residuals-by-geography analysis.

**L2 — Why [log target / those features / RMSE]?**
- *Why log-transform the target?* "The target is right-skewed (median ~$180K, tail past $500K). In raw space, RMSE is owned by expensive districts; log-space makes errors proportional — a 10% error in a $100K district counts the same as 10% in a $600K one. That matches how pricing errors actually hurt, and residuals became homoscedastic."
- *Why rooms-per-household rather than raw rooms?* "Raw totals conflate district size with housing stock; per-household ratios isolate the actual signal (crowding). Domain reasoning first, then CV confirmed it carried incremental information."
- *Why RMSE (log) as headline?* "It's the Kaggle-standard metric for this dataset, which makes results comparable, and in log space it maps to relative error — interpretable to stakeholders ('within ~12% median error'). I report MAE alongside for robustness."

**L3 — What alternatives did you reject?**
- "Kriging/spatial-GAM models: the geographic structure is real (the error map shows it), but the project's scope was the standard tabular pipeline; I noted spatial modeling as the extension."
- "Outlier removal (the $500K cap districts): rejected — the cap is a *data collection artifact* (values censored at $500K), and dropping censored rows biases the sample toward cheap districts. Keeping them and noting the ceiling is the honest choice."
- "Neural nets: same reasoning as Project 1 — 20K rows, 8 tabular features, GBDT's home turf; a net would need to justify itself and couldn't."

**L4 — Where does it break?**
- "The $500K cap: districts AT the ceiling are under-predicted by construction — the model learned that prices 'approach but rarely exceed' $500K. Any use case caring about the top of the market inherits that bias."
- "It's 1990 census data — California's housing market is concept drift incarnate; the value is the *method*, not the model."
- "District-level aggregation hides within-district variance — a $2M street next to a $300K tract shares one label; predictions are about districts, not houses."

**L5 — What would you change?**
"(1) Spatial cross-validation — random folds leak geographic neighbors (adjacent districts are near-duplicates); spatial blocking is the correct split. (2) Quantile regression or conformal intervals instead of point predictions — pricing decisions want uncertainty. (3) A drift-monitoring sketch: the error map is exactly what I'd re-generate in production monthly."

### The Hostile-Probe Appendix

- *"Is your CV honest given spatial autocorrelation?"* — "Good catch, no — random folds put geographic neighbors in train and test, which flatters the score. Spatial CV is on my improvement list; the honest number is lower and I can say by roughly how much from the neighbor-overlap structure."
- *"Why GBDT over random forest?"* — "Both within noise here, honestly. GBDT won the comparison on this seed, and boosting's sequential correction usually edges bagging on smooth regression targets at this size — but the defensible claim is 'ensemble > linear', not 'GBDT > RF'."
- *"Which feature would you drop last?"* — "Median income — it dominates importance by a wide margin; dropping it costs more than dropping everything else combined. Which is also a fairness flag: income correlates with protected attributes, so a real deployment needs a proxy-fairness analysis."

---

## Applying the Drill to Your Own Projects

Work through this worksheet per project (30 minutes each, once):

1. **Write the one-liner** — 25 words: problem, approach, stack, headline number, honest caveat.
2. **List every choice you made** — model, split, metric, features, preprocessing, scope cuts. That list is your L2 queue.
3. **For each choice, write the rejected alternative + the reason.** No alternative written = a question you'll fumble.
4. **Name the failure modes** — where the model is wrong, when the approach breaks, what the data can't support. (Your error analysis section is the source.)
5. **Write the three upgrades** — what more time/data/tools would buy. Ordered.
6. **Say the whole thing out loud, timed.** Two minutes for the walkthrough; 30–60 seconds per drill-down. Record once, listen once, fix the three worst moments, stop.

**The meta-skill the drill builds:** interviewers don't actually need you to be right at level 5 — they need to watch you *stay calm and reasoned as depth increases*. A rehearsed L1–L3 with a graceful "and here's where my certainty runs out" at L4 reads as senior; improvised confidence that collapses at L2 reads as junior no matter how good the project was.

---

*This drill pairs with the [Behavioral guide](Behavioral%20and%20STAR%20Answers.md) (stories 1, 9, 13 reference these projects) and the [Mock Interviews](Mock%20Interview%20-%20ML%20Engineer.md) — the deep-dive segment there runs this exact pattern.*
