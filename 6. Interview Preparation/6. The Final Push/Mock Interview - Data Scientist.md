# Mock Interview — Data Scientist (Full Loop, ~3 Hours, Timed)

The DS loop differs from the MLE loop in weight: **SQL is live-coded, A/B testing is drilled hard, product sense is a scored round of its own, and the ML depth questions lean statistical** (causation, experiment design, inference) rather than systems. This mock mirrors that weighting. Run it timed, in segments, grading honestly against the model answers.

**Setup:** timer, notebook, no notes during segments. SQL segment needs you to actually *write* queries — pen and paper or a blank editor, no autocomplete.

---

## Segment 1 — SQL Round (45 min, Live-Coded)

Schema (memorize the shape; the interviewer shows tables):

```sql
users        (user_id, country, signup_date, plan)
events       (user_id, ts, event_name, properties_json)
subscriptions(user_id, plan, start_date, end_date, price_monthly)
```

**Q1 (8 min): "DAU — daily active users, last 30 days, by country."**

```sql
SELECT u.country, DATE(e.ts) AS day, COUNT(DISTINCT e.user_id) AS dau
FROM events e
JOIN users u ON u.user_id = e.user_id
WHERE e.ts >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY u.country, DATE(e.ts)
ORDER BY dau DESC;
```
**Grades:** `COUNT(DISTINCT ...)` not `COUNT(*)` (a user with 50 events is one DAU) · join before aggregation · the follow-up *"why is COUNT(*) wrong here?"* answered instantly.
*Your score: ___ /3*

**Q2 (10 min): "Week-3 retention of the January signup cohort."**

```sql
WITH january AS (
  SELECT user_id, MIN(DATE(ts)) AS first_day
  FROM events GROUP BY user_id
  HAVING MIN(DATE(ts)) BETWEEN '2026-01-01' AND '2026-01-31'
)
SELECT
  COUNT(DISTINCT CASE WHEN DATE(e.ts) BETWEEN j.first_day + 21 AND j.first_day + 27
                      THEN e.user_id END) AS retained_w3,
  COUNT(DISTINCT j.user_id) AS cohort
FROM january j
LEFT JOIN events e ON e.user_id = j.user_id;
```
**Grades:** cohort defined by *first-ever* activity (the CTE), not signup_date alone · week 3 as a *window* (days 21–27), not a single day · LEFT JOIN so non-retained users still count in the denominator.
*Your score: ___ /3*

**Q3 (12 min): "Users whose average session count per week grew in the last 4 weeks vs the 4 weeks before."**

```sql
WITH weekly AS (
  SELECT user_id,
         DATE_TRUNC('week', ts) AS wk,
         COUNT(*) AS sessions
  FROM events WHERE event_name = 'session_start'
  GROUP BY user_id, DATE_TRUNC('week', ts)
)
SELECT user_id
FROM (
  SELECT user_id,
         AVG(CASE WHEN wk >= CURRENT_DATE - INTERVAL '4 weeks' THEN sessions END) AS recent,
         AVG(CASE WHEN wk <  CURRENT_DATE - INTERVAL '4 weeks' THEN sessions END) AS prior
  FROM weekly
  WHERE wk >= CURRENT_DATE - INTERVAL '8 weeks'
  GROUP BY user_id
) t
WHERE recent > prior;
```
**Grades:** window-bucketing before comparison · conditional aggregation (AVG over CASE) instead of two queries · NULL-handling awareness ("users with no recent weeks have NULL recent — excluded by the comparison; is that right? yes — they churned").
*Your score: ___ /3*

**Q4 (15 min, the boss question): "Monthly retention curve — for each monthly cohort, % active in each subsequent month, one row per cohort × month-since-signup."**

```sql
WITH first AS (
  SELECT user_id, DATE_TRUNC('month', MIN(ts)) AS cohort_month
  FROM events GROUP BY user_id
),
activity AS (
  SELECT user_id, DATE_TRUNC('month', ts) AS active_month
  FROM events GROUP BY user_id, DATE_TRUNC('month', ts)
)
SELECT f.cohort_month,
       (a.active_month - f.cohort_month) AS months_since,   -- dialect-dependent
       COUNT(DISTINCT a.user_id) * 1.0 /
         (SELECT COUNT(*) FROM first f2 WHERE f2.cohort_month = f.cohort_month) AS pct_active
FROM first f
JOIN activity a ON a.user_id = f.user_id AND a.active_month >= f.cohort_month
GROUP BY f.cohort_month, months_since
ORDER BY f.cohort_month, months_since;
```
**Grades:** cohort logic separated from activity logic (two CTEs) · month-difference arithmetic · the denominator subquery (or window) per cohort · **say out loud**: "in production I'd validate the current month's cohort is partial."
*Your score: ___ /4*

**SQL segment pass bar: 9/13.** Write every query before looking at the answers — reading them is worth ~20% of doing them.

---

## Segment 2 — A/B Testing Drill (45 min)

**Scenario the interviewer reads:** *"We changed the checkout button from 'Buy now' to 'Complete purchase'. 50/50 experiment, 2M users, ran 14 days. Conversion: 3.42% (control) vs 3.51% (treatment), p = 0.04. Ship it?"*

**The strong-pass answer, in the order to deliver it:**

1. **Effect size & CI, not just p:** "+0.09pp is a 2.6% relative lift. The 95% CI is roughly [0.004pp, 0.176pp] — the upper bound is a 5% relative lift. It's a real but small effect."
2. **Guardrails:** "What happened to revenue per user, refund rate, and time-to-purchase? A conversion lift that hurts AOV can be net-negative."
3. **Novelty & segments:** "Was the lift uniform, or concentrated in first-week users (novelty) or one platform (a bug masked by aggregation)? I'd cut the results by tenure and platform."
4. **Duration logic:** "14 days covers two full weekly cycles — good; and with 2M users we're powered for effects much smaller than 0.09pp, so this wasn't an underpowered fluke."
5. **Decision:** "With a positive CI lower bound and clean guardrails, I'd ship — but I'd want the segment cuts first. If the lift is all novelty, it decays in a month and the change isn't worth the code churn."

**Follow-up drill (answer each in 30 seconds):**
- "p = 0.06 instead — kill it?" — "No — look at the CI and cost of the two errors. A cheap reversible change with CI [−0.01, +0.19]pp often ships; a risky restructure with the same CI doesn't."
- "The experiment needs 600k users and the feature team wants results in 3 days on 200k. Options?" — "Accept lower power and *say what we can detect* at that n; or reduce variance — CUPED with pre-period conversion typically cuts needed n 20–40%; or ship as a holdout decision and evaluate with more days, not more users."
- "Treatment users can invite control users. Problem?" — "Interference: the treatment spills (shared carts), violating SUTVA. Options: cluster randomization (by friend-group or geography) or switchback design."
- "Metric moved for power users only. Good or bad?" — "Investigate heterogeneity: it might be the real story (feature serves power users) or an artifact (they generate most events). Slice, then decide — but never *post-hoc cherry-pick* a winning segment without acknowledging the multiple-comparisons problem."

**Scoring:** 1 point per numbered element + 1 per clean follow-up. **Pass: 6/8.** *Your score: ___*

---

## Segment 3 — Product Sense / Case (30 min)

**Prompt:** "Engagement (weekly active users) dropped 8% last week. Figure out why."

**The structured investigation the interviewer wants to hear:**

1. **Clarify the metric:** "Ways WAU is computed — what counts as active? Did the *definition* or instrumentation change? First check: is this real or a tracking regression?" (Surprisingly often: tracking.)
2. **Decompose:** "8% overall — is it spread evenly, or concentrated? Cut by platform, geography, tenure, acquisition channel, and *new vs returning*. A drop in new-user activation looks completely different from a churn spike among existing users."
3. **Time patterns:** "Gradual decline vs cliff? A cliff at a specific timestamp points to a release; a gradual slide points to a seasonal pattern or a competitor."
4. **External checks:** "Holiday/weekend composition of the week? A major event? A competitor launch? An app-store outage?"
5. **Hypotheses + evidence, ranked:** release regression (check rollout correlation), tracking bug (check event-volume by event type), seasonal (year-over-year comparison), competitor (search/social signals), genuine dissatisfaction (support tickets, app ratings).
6. **Close with the plan:** "I'd pull the segment cuts first — they usually name the culprit — then match each candidate hypothesis to its expected signature in the data and check them in order of likelihood × cheapness."

**Scoring:** instrumentation check · decomposition named with ≥4 cut dimensions · cliff-vs-gradual reasoning · external factors · at least 3 concrete hypotheses with distinguishing evidence · a closing plan. **Pass: 5/6.** *Your score: ___*

---

## Segment 4 — ML & Statistics Depth, DS-Flavored (45 min)

**Q1. "You fit a model to predict which users will churn. The retention team asks: 'if we run our campaign on the model's top decile, how many churns will we prevent?' What's wrong with the question?"**
*Pass:* prediction ≠ intervention — the model ranks who's *likely* to churn; it says nothing about who *responds* to the campaign.
*5/5:* uplift framing — model P(churn) × P(saves | campaign) separately, or the two-model/padded-outcome uplift approach; propose a holdout within the campaign to *measure* the causal effect; Qini curve as the evaluation.
*Your score: ___*

**Q2. "Feature X correlates 0.6 with the target. Your teammate adds it and the CV score jumps 0.05. Green light?"**
*Pass:* correlation ≠ license — check what X is: if it's computed *after* the target exists (post-hoc feature), it's leakage.
*5/5:* "I check the feature's availability-at-prediction-time, its relationship to the target's timeline, and whether the jump survives a temporal split. Then SHAP/permutation importance for sanity — and a domain plausibility read. A 0.05 CV jump from one feature is *more* suspicious than delightful."
*Your score: ___*

**Q3. "Explain confidence intervals to a marketing VP in 45 seconds."**
*Model:* "If we re-ran this campaign 100 times and computed the range each time, about 95 of those ranges would contain the true lift. Our one range — 1% to 3% — is one draw of that process: we're pretty confident the true effect is positive, less confident it's above 2%." *The skill: correct, short, zero jargon, stops talking.*
*Your score: ___*

**Q4. "What's the difference between R² and adjusted R², and which do you report?"**
*Pass:* R² rises with any added feature, even noise; adjusted penalizes parameter count.
*5/5:* "For model *comparison* with differing feature counts, adjusted or CV-based; for communication to stakeholders, plain R² with the caveat. In practice I report CV metrics — in-sample corrections are training wheels for the real thing."
*Your score: ___*

---

## Segment 5 — Behavioral (30 min)

Run 3 stories from the [Behavioral guide](Behavioral%20and%20STAR%20Answers.md): the metric-translation conflict (story 10), the ambiguous-requirements take-home (story 3), and the data-mistake story (13). DS loops specifically probe *how you handle being wrong about numbers* — the data-mistake story is near-guaranteed to be probed; make sure its redemption arc is tight.

---

## Scoring the Loop

| Segment | Pass bar | Strong pass |
| --- | --- | --- |
| SQL | 9/13 | 11/13 with Q4 clean |
| A/B testing | 6/8 | 7/8 including the interference follow-up |
| Product sense | 5/6 | 6/6 with hypotheses ranked by likelihood × cost |
| ML/stats depth | 3 passes | 2 at 5/5 (uplift + leakage questions are the differentiators) |
| Behavioral | all stories quantified | conflict story ends in changed process |

**Study routing by weak segment:** SQL → [SQL Interview Questions](../2.%20Coding%20Rounds/SQL%20Interview%20Questions.ipynb) + [Joins and Window Functions](../2.%20Coding%20Rounds/SQL%20Joins%20and%20Window%20Functions.ipynb) · A/B → [A/B Testing and Causal Inference](../3.%20Machine%20Learning%20Interviews/A%20B%20Testing%20and%20Causal%20Inference.ipynb) · Product sense → segment 3 above, re-run with a different metric drop · Stats → [Statistics and Probability](../1.%20Fundamentals%20Review/Statistics%20and%20Probability.ipynb) · Uplift → the causal-inference section of the A/B notebook.

**Same calibration note as the MLE mock:** the "I don't know, here's how I'd find out" moments are being scored. In DS loops they're joined by a second scored behavior — **do you reach for data before opinions** in the product-sense segment. Candidates who say "I'd check whether that's true" before theorizing read as data scientists; the reverse order reads as a consultant.
