# 6. Interview Preparation — From Roadmap to Offer Letter

Levels 1–5 of this roadmap teach you machine learning. This section closes the loop: **getting selected**. It maps every round you will actually face in 2026 hiring loops — screening, coding, ML theory, system design, and behavioral — and gives you executed, verifiable answers for each one.

> **How this section is different from Levels 1–5:** those notebooks *teach concepts*. These notebooks *rehearse selection*. Every answer is written the way a strong candidate says it out loud — 60–90 seconds, structured, with a concrete number or tradeoff in every response. Code cells are executed so every number you quote in an interview is a number you have actually run.

## The 2026 Interview Loop

A typical ML Engineer / Data Scientist loop now has five stages. Each subfolder below maps to one:

| Stage | What they test | Typical length | Where to prepare here |
| --- | --- | --- | --- |
| 1. Recruiter / screening call | Fit, salary, timeline, one survival question per area | 20–30 min | [6. The Final Push](#6-the-final-push) |
| 2. Coding rounds | Python fluency, Pandas/NumPy under time pressure, SQL joins & window functions, implement-an-algorithm | 45–60 min × 1–3 | [2. Coding Rounds](#2-coding-rounds) |
| 3. ML depth rounds | Bias–variance, regularization, metrics, algorithm internals, "walk me through your project" | 45–60 min × 2–4 | [3. Machine Learning Interviews](#3-machine-learning-interviews) + [1. Fundamentals Review](#1-fundamentals-review) |
| 4. System design | Feature stores, training pipelines, serving, monitoring; increasingly **LLM/RAG systems** | 45–60 min × 1–2 | [5. System Design](#5-system-design) |
| 5. Behavioral / project deep-dive | STAR stories, conflict, failure, ownership; 5-why drilling on your own projects | 30–45 min | [6. The Final Push](#6-the-final-push) |

Data-scientist-heavy loops add **A/B testing and causal inference**; quant-adjacent loops add probability brainteasers — both are covered in [3. Machine Learning Interviews](#3-machine-learning-interviews) and [1. Fundamentals Review](#1-fundamentals-review).

## Folder Guide

### 1. Fundamentals Review
The math that comes up *inside* ML rounds — not as exams, but as the vocabulary of every answer.

- [Statistics and Probability.ipynb](1.%20Fundamentals%20Review/Statistics%20and%20Probability.ipynb) — distributions, CLT, hypothesis tests, Bayes, and the probability puzzles interviewers actually ask
- [Linear Algebra.ipynb](1.%20Fundamentals%20Review/Linear%20Algebra.ipynb) — the five decompositions asked about, what PCA/SVD/ridge actually do to a matrix

### 2. Coding Rounds
Where most rejections happen. Executed drills, not reading material.

- [Python Coding Round.ipynb](2.%20Coding%20Rounds/Python%20Coding%20Round.ipynb) — the 20 patterns that cover most Python screens (two-pointer, sliding window, prefix sums, …)
- [Python Puzzles and Edge Cases.ipynb](2.%20Coding%20Rounds/Python%20Puzzles%20and%20Edge%20Cases.ipynb) — mutable defaults, `is` vs `==`, closures, GIL: the trap questions
- [Pandas Interview Questions.ipynb](2.%20Coding%20Rounds/Pandas%20Interview%20Questions.ipynb) — groupby-apply, merge gotchas, rolling windows, the "split-apply-combine in 60 seconds" drills
- [NumPy Interview Questions.ipynb](2.%20Coding%20Rounds/NumPy%20Interview%20Questions.ipynb) — broadcasting, vectorization speed proof, axis semantics, views vs copies
- [SQL Interview Questions.ipynb](2.%20Coding%20Rounds/SQL%20Interview%20Questions.ipynb) — selects, joins, aggregations on real tables
- [SQL Joins and Window Functions.ipynb](2.%20Coding%20Rounds/SQL%20Joins%20and%20Window%20Functions.ipynb) — the modern DS filter: `ROW_NUMBER`, `LAG`, running totals, sessionization
- [ML Coding Round.ipynb](2.%20Coding%20Rounds/ML%20Coding%20Round.ipynb) — implement k-means, k-NN, decision-tree split, and logistic regression *from scratch* in an interview-safe 20 lines each
- [Take-Home Assignments Guide.md](2.%20Coding%20Rounds/Take-Home%20Assignments%20Guide.md) — how take-homes are graded and the rubric that scores full marks

### 3. Machine Learning Interviews
Theory rounds, organized so every answer is quotable.

- [ML Core Concepts.ipynb](3.%20Machine%20Learning%20Interviews/ML%20Core%20Concepts.ipynb) — train/test/val, overfitting, cross-validation, data drift: the round-one warm-ups
- [Supervised Learning Questions.ipynb](3.%20Machine%20Learning%20Interviews/Supervised%20Learning%20Questions.ipynb) — linear/logistic internals, SVM kernels, KNN distance choices
- [Trees and Boosting Questions.ipynb](3.%20Machine%20Learning%20Interviews/Trees%20and%20Boosting%20Questions.ipynb) — Gini vs entropy, why XGBoost wins tables, LightGBM histogram trick
- [Unsupervised Learning Questions.ipynb](3.%20Machine%20Learning%20Interviews/Unsupervised%20Learning%20Questions.ipynb) — clustering choices, elbow vs silhouette, PCA vs t-SNE vs UMAP
- [Model Evaluation Questions.ipynb](3.%20Machine%20Learning%20Interviews/Model%20Evaluation%20Questions.ipynb) — ROC vs PR, calibration, imbalanced thresholds: with executed proof
- [Regularization and Bias-Variance.ipynb](3.%20Machine%20Learning%20Interviews/Regularization%20and%20Bias-Variance.ipynb) — L1 vs L2 geometry, lambda selection, the "double descent" follow-up
- [Gradient Descent from Scratch.ipynb](3.%20Machine%20Learning%20Interviews/Gradient%20Descent%20from%20Scratch.ipynb) — GD, SGD, mini-batch, momentum, Adam: coded and converged on real data
- [Feature Engineering Questions.ipynb](3.%20Machine%20Learning%20Interviews/Feature%20Engineering%20Questions.ipynb) — encoding, leakage-safe pipelines, target encoding done right
- [A/B Testing and Causal Inference.ipynb](3.%20Machine%20Learning%20Interviews/A%20B%20Testing%20and%20Causal%20Inference.ipynb) — power, MDE, peaking, CUPED, uplift: the DS-loop differentiator

### 4. Deep Learning and Domains
- [Deep Learning Questions.ipynb](4.%20Deep%20Learning%20and%20Domains/Deep%20Learning%20Questions.ipynb) — backprop by hand, vanishing gradients, batch norm, dropout, optimizers — with a NumPy-only neural net that trains live
- [NLP Interview Questions.ipynb](4.%20Deep%20Learning%20and%20Domains/NLP%20Interview%20Questions.ipynb) — TF-IDF math, embeddings, attention, tokenizers, and 2026-era LLM questions
- [Time Series Interview Questions.ipynb](4.%20Deep%20Learning%20and%20Domains/Time%20Series%20Interview%20Questions.ipynb) — stationarity tests, ARIMA orders, validation splits that don't leak

### 5. System Design
- [ML System Design.ipynb](5.%20System%20Design/ML%20System%20Design.ipynb) — the 6-step frame, then 3 full designs (fraud, recsys, churn) with capacity numbers
- [LLM and RAG System Design.ipynb](5.%20System%20Design/LLM%20and%20RAG%20System%20Design.ipynb) — chunking, embedding choice, vector indexes, RAG eval, guardrails: the 2026 favorite

### 6. The Final Push
- [Behavioral and STAR Answers.md](6.%20The%20Final%20Push/Behavioral%20and%20STAR%20Answers.md) — 15 written-out STAR answers incl. the failure story
- [Resume and Portfolio Guide.md](6.%20The%20Final%20Push/Resume%20and%20Portfolio%20Guide.md) — the X-Y-Z bullet formula, applied to this repo's own projects
- [Project Deep-Dive Prep.md](6.%20The%20Final%20Push/Project%20Deep-Dive%20Prep.md) — the 5-why drill on the Titanic and California Housing capstones
- [Mock Interview - ML Engineer.md](6.%20The%20Final%20Push/Mock%20Interview%20-%20ML%20Engineer.md) — full 5-round mock with model answers
- [Mock Interview - Data Scientist.md](6.%20The%20Final%20Push/Mock%20Interview%20-%20Data%20Scientist.md) — full loop with SQL + A/B testing emphasis
- [Mock Interview - ML System Design.md](6.%20The%20Final%20Push/Mock%20Interview%20-%20ML%20System%20Design.md) — 90-minute design drill with grading rubric
- [Final Week Preparation Plan.md](6.%20The%20Final%20Push/Final%20Week%20Preparation%20Plan.md) — day-by-day plan for the last 7 days
- [Offer Negotiation.md](6.%20The%20Final%20Push/Offer%20Negotiation.md) — timelines, leverage, and scripts that don't blow up offers

### 7. Question Banks and Cheat Sheets
- [Question Bank Index.md](7.%20Question%20Banks%20and%20Cheat%20Sheets/Question%20Bank%20Index.md) — every question across the section, indexed by round and difficulty
- [Cheat Sheets/](7.%20Question%20Banks%20and%20Cheat%20Sheets/Cheat%20Sheets/) — one-page refreshers: pandas, NumPy, scikit-learn, SQL, statistics, probability, deep learning, NLP, time series, ML system design

## How to Use This Section

1. **Diagnose first (Day 0).** Open the [Question Bank Index](7.%20Question%20Banks%20and%20Cheat%20Sheets/Question%20Bank%20Index.md) and attempt questions cold. Every question you miss maps to a notebook.
2. **One round-type per day.** Don't read these like a textbook — rehearse them like a script. Cover the answer, say yours out loud, then compare.
3. **Re-run the code cells before onsite rounds.** Quoting a number you personally executed ("I measured the target-leakage illusion at 15 AUC points") lands differently than reciting a blog.
4. **Finish with the mock interviews.** They are written as real loops — do them timed, in one sitting, before the [Final Week Plan](6.%20The%20Final%20Push/Final%20Week%20Preparation%20Plan.md).

## Suggested Timeline

| Weeks out | Focus |
| --- | --- |
| 4+ | Fundamentals Review + one Coding Rounds notebook per week |
| 3 | Machine Learning Interviews (the whole folder) |
| 2 | Deep Learning and Domains + System Design |
| 1 | Mock interviews + cheat sheets + behavioral |
| Final 7 days | [Final Week Preparation Plan](6.%20The%20Final%20Push/Final%20Week%20Preparation%20Plan.md) only |
