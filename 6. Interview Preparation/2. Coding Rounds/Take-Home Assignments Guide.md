# Take-Home Assignments Guide — How They're Graded and How to Score Full Marks

Take-homes look like data-science problems. They are actually **judgment tests**: given a week, can you do work that looks like a careful engineer's? Interviewers grade a rubric, not a leaderboard number. This guide is that rubric, turned around into a playbook.

---

## 1. What You're Actually Being Graded On

Typical rubric weights, compiled from how ML teams describe their take-home scoring:

| Dimension | Weight | What full marks looks like |
| --- | --- | --- |
| Problem understanding & framing | 20% | You restate the business question, define the target, and *state assumptions explicitly* |
| Data handling & EDA | 15% | Purposeful EDA that drives decisions — not a chart dump |
| Modeling approach & justification | 20% | A sensible baseline → 2–3 defensible candidates, with a stated selection metric aligned to the business cost |
| Code quality & reproducibility | 20% | Clean repo, README, `requirements.txt`/`environment.yml`, one-command run, fixed seeds |
| Validation rigor (leakage, splits) | 15% | Correct CV design, leakage guards, honest test-set discipline |
| Communication | 10% | A short write-up: findings, tradeoffs, what you'd do with more time |

**The pattern to internalize:** 60% of the grade (framing + validation + communication + code quality) is *not* modeling skill. Candidates lose take-homes on sloppiness, not on model choice.

---

## 2. The Six-Phase Workflow (and Its Time Budget)

For a 7-day assignment, budget roughly:

1. **Day 1 — Read and frame (don't open a notebook yet).** What is the business decision behind the metric? What's the target? What would a null/naive answer be? Write a one-paragraph problem statement and your assumptions. *Deliverable: `README.md` skeleton.*
2. **Day 2 — EDA with intent.** Every plot should end in a decision: "target is imbalanced 1:23 → I will use PR-AUC and class weights", "this feature is 40% missing and correlates with nothing → I'll test dropping it." *Deliverable: 5–10 decision-linked findings.*
3. **Day 3 — Baseline + pipeline skeleton.** A trivial baseline (majority class / mean / single feature) and a sklearn `Pipeline` with a `ColumnTransformer`. The baseline number is what makes every later number interpretable.
4. **Days 4–5 — Two or three defensible models.** Linear (interpretable) + gradient boosting (performance) covers 90% of take-homes. Tune *lightly* — a small `GridSearchCV`/`optuna` study, not a 500-trial marathon.
5. **Day 6 — Validation hardening + error analysis.** Re-check leakage paths, plot errors by segment, calibrate thresholds to business costs, and compute confidence intervals on the headline metric (bootstrap).
6. **Day 7 — Polish and write-up.** README, requirements, seeds, cleanup, final report. Never leave this to the last two hours.

---

## 3. The Non-Negotiables (Instant-Fail Zone)

- **Leakage.** Target-derived features, fitting scaler/imputer on the full data before splitting, time-traveling features (using future information) in temporal data. Reviewers *look* for this. The Level-3 notebook [Data Leakage](../../3.%20Learn%20Practical%20Concepts/16.%20Data%20Leakage/Data%20Leakage.ipynb) shows each of these happening — re-run it before you start.
- **No test-set discipline.** Touching the test set more than once is the second thing senior reviewers check. Keep a final holdout untouched until the last day.
- **Unreproducible repo.** No seeds, absolute paths, missing dependencies, "works on my machine" errors on the reviewer's laptop. One command — `make all` or `python src/main.py` — must go raw data → results.
- **Fabricated or rounded-up numbers.** If a metric is 0.812 report 0.812. Reviewers re-run your code; a mismatch between the write-up and the output of your own repo is fatal to trust.

---

## 4. The Write-Up Skeleton (One Page Wins)

Structure the report like an executive memo, in this order:

1. **TL;DR (3 sentences):** the business question, the headline result with CI, the recommendation.
2. **Approach:** framing, key assumptions (bulleted), target/metric definition and *why that metric maps to business value*.
3. **What the data showed:** the 4–6 EDA findings that changed your plan — each one line, linked to the decision it drove.
4. **Modeling:** baseline → final; a small table (model, CV metric ± std, test metric). Name the runner-up and why you didn't pick it — this signals judgment better than a leaderboard.
5. **Error analysis & limitations:** where the model is weak, which segments, what you'd collect next.
6. **With more time:** 2–3 concrete next steps (feature source, external data, deployment concern). Honest scoping signals seniority.

---

## 5. Red Flags in Your Own Work (Self-Review Checklist)

Before submitting, walk this list — every item is a real rejection reason:

- [ ] Assumptions stated up front (missing-value meaning, units, timeline)
- [ ] Every EDA plot has a takeaway sentence, not just a caption
- [ ] Baseline exists and is referenced in every results comparison
- [ ] Split strategy matches the data (temporal split for time series; group split if rows share users/patients)
- [ ] All preprocessing inside a `Pipeline`/`ColumnTransformer` — no manual fit on full data
- [ ] Cross-validation results reported with ± std, not a single fold
- [ ] Final numbers reproduced from a clean run (`Kernel → Restart & Run All` equivalent)
- [ ] Seeds fixed everywhere randomness enters (numpy, sklearn, splits)
- [ ] `README.md`: how to run, repo map, key decisions
- [ ] `requirements.txt` pinned (`pandas==…`, not `pandas>=…`)
- [ ] No dead code, no commented-out experiments, no 200-cell kitchen-sink notebook
- [ ] Threshold/business-cost discussion if the problem is imbalanced classification
- [ ] The write-up and the repo numbers agree exactly

---

## 6. Interview: Defending the Take-Home

The follow-up interview drills your take-home. Expect:

- **"Walk me through your approach"** — 90 seconds, framing-first, ending with the headline result.
- **"Why model X and not Y?"** — you need the runner-up's tradeoff memorized (see the [Project Deep-Dive Prep](../6.%20The%20Final%20Push/Project%20Deep-Dive%20Prep.md) 5-why drill).
- **"What would break first in production?"** — drift, retraining cadence, latency, monitoring. The [Model Serving](../../3.%20Learn%20Practical%20Concepts/17.%20Model%20Serving/Model%20Serving.ipynb) and [ML System Design](../5.%20System%20Design/ML%20System%20Design.ipynb) notebooks are the vocabulary source.
- **"Biggest limitation?"** — have one ready and be candid; "it needs external feature X which we didn't have" is a strong answer. "None" is a rejection.
- **A hostile detail probe** — one number, one feature, one line of code. Know your repo cold enough to open any file and explain it.

---

## 7. Time-Boxing Rules When Everything Goes Wrong

- Stuck on dirty data after day 2? Cap cleanup effort, document the data problems in the report, and move to modeling — *documented messiness beats an unfinished model.*
- Model won't beat baseline? That **is** a finding. Report it honestly with hypotheses — sometimes the honest answer is "the signal isn't in these features," and saying so with evidence is a pass; hiding it with a leaked feature is a fail.
- Over budget? Cut hyperparameter tuning first, then number of models — never cut validation, documentation, or the error analysis. Those are the highest-weight rubric rows.

---

*This guide pairs with the [End-to-End ML Project](../../5.%20Pushing%20it%20with%20Projects/End-to-End%20ML%20Project/README.md) and [End-to-End Regression Project](../../5.%20Pushing%20it%20with%20Projects/End-to-End%20Regression%20Project/README.md) capstones — they were built to be defensible take-home exemplars: shipped data, one-command reproducibility, documented decisions.*
