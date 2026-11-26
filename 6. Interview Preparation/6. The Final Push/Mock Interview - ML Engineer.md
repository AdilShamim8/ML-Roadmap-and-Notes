# Mock Interview — ML Engineer (Full Loop, ~3 Hours, Timed)

This is a rehearsal document, not reading material. Run it **timed, in one sitting per segment**, ideally with a friend as interviewer or a timer + honest self-grading. The model answers are calibrated to strong-pass level — compare your spoken answer against them immediately after each segment. The grading rubric at the end tells you whether you're ready.

**Setup:** timer, blank notebook, no notes during segments. Answers spoken out loud, then written self-scores. Segments match the [section README's loop map](../README.md).

---

## Segment 1 — Recruiter Screen (20 min)

**Interviewer script (friend reads these):**

1. "Tell me about yourself." *(90 seconds max — the elevator pitch)*
2. "What are you working on right now?"
3. "What made you apply here?"
4. "What's your compensation expectation?"
5. "Timeline — when could you start?"

**Model answer for #1 (the pitch skeleton — fill in your specifics):**

> "I'm a machine learning engineer focused on the full lifecycle — I care as much about honest validation and serving as about modeling. Over the past year I built out an executed ML roadmap: a hundred-plus notebooks where every claim is demonstrated in code, plus two end-to-end capstones — a classification pipeline and a regression pipeline — both with leakage-safe validation and error analysis. Before that I worked on [your real background]. What I'm looking for is [team's domain] at real scale — and this role matches because [specific fact about the team]."

**Self-check:** Did you state a specialty, name concrete artifacts, and land on *their* team — all inside 90 seconds, no biography drift?

**Compensation question — the calibrated move:** give a researched range ("Based on the market for [level] in [location], I'm targeting X to Y — where in that range depends on the total package") — not a single number, not "whatever you offer." Never anchor below your researched floor.

---

## Segment 2 — Coding Round (60 min)

**Problem (set the clock):**

> "Given a stream of user events — `(user_id, timestamp, event_type)` — write a function that computes, for each user, the number of distinct sessions, where a session ends after 30 minutes of inactivity. Then extend it: find each user's longest session length in events."

Write it cold. Then compare against the reference approach:

**The expected shape of the solution (Python):**

```python
from collections import defaultdict
from datetime import datetime, timedelta

GAP = timedelta(minutes=30)

def sessionize(events):
    by_user = defaultdict(list)
    for user, ts, _ in events:
        by_user[user].append(ts)
    sessions = defaultdict(list)          # user -> [session lengths in events]
    for user, stamps in by_user.items():
        stamps.sort()                      # interviewers watch for this
        count, length = 1, 1
        for prev, cur in zip(stamps, stamps[1:]):
            if cur - prev > GAP:
                sessions[user].append(length); count += 1; length = 1
            else:
                length += 1
        sessions[user].append(length)
    return sessions                        # len(sessions[u]) = session count; max = longest
```

**What the interviewer grades (score yourself 1–5 each):**
- [ ] Clarifying questions first (timestamps sorted? format? timezone? duplicates?) — *asking two before coding is the pass bar*
- [ ] Stated O(n log n) and *why* (the sort dominates)
- [ ] Grouped by user before processing (the obvious-in-retrospect step people miss under pressure)
- [ ] Handled the edge cases aloud (single event = one session; empty input = {})
- [ ] Narrated while coding, tested with a tiny example

**Follow-up they ask after you finish:** "10 billion events, doesn't fit on one machine — now what?" — Expected riff: sort by (user, time) is parallelizable (MapReduce/Spark: shuffle by user, reduce per user); or maintain sessions *streaming* (per-user last-seen state in a key-value store, emit session-close events on timeout); the streaming answer is the senior one — mention both, name the tradeoff (batch reprocessing vs stateful recency).

---

## Segment 3 — ML Depth Round 1: Fundamentals (60 min)

Speak each answer aloud, 60–90 seconds, then grade against the model.

**Q1. "Walk me through how you'd validate a churn model where you have 18 months of monthly data per customer."**
*Pass answer:* temporal validation — train on earliest months, validate on later, walk-forward across several cutpoints; never shuffle (customer overlap between train/test leaks identity); metric = recall@capacity or PR-AUC given churn's typical imbalance; baseline = "everyone who churned last month churns this month."
*5/5 answer adds:* label definition matters (churn = N consecutive inactive months? define N first); feature as-of discipline (no post-hoc features); probability calibration if the retention team ranks by expected value.
*Your score: ___*

**Q2. "Your model's AUC is 0.85 offline, 0.72 in production. Debug."**
*Pass answer:* data mismatch (training-serving skew: features computed differently), distribution drift (covariate or concept), or evaluation optimism (leakage offline).
*5/5 answer gives the ordered triage:* (1) compare feature distributions train vs production (PSI) — skew or drift; (2) reproduce a production prediction offline — if it differs, it's a serving bug, not a model problem; (3) check label delay/maturity in the production metric; (4) only then consider retraining. *"I've seen this be a feature-parity bug more often than actual drift"* is the line that lands.
*Your score: ___*

**Q3. "Explain gradient boosting to a product manager."**
*Pass answer:* no jargon — "a team of weak rules, each new rule fixing what the team currently gets wrong, learning slowly so no single rule dominates."
*5/5 answer:* the 30-second version, then *stop* — the mistake is depth-creep into loss functions when the audience is PM. Reading the audience IS the skill being tested.
*Your score: ___*

**Q4. "L1 vs L2 regularization — when does each win?"**
*Pass:* L1 → sparsity/feature selection (diamond corners); L2 → stability under correlated features.
*5/5 adds:* the geometric picture in one sentence; ElasticNet as the blend; "scaling before penalizing" unprompted; the double-descent aside if the interviewer is senior.
*Your score: ___*

**Q5. "How would you detect and handle data drift in production?"**
*Pass:* monitor feature distributions (PSI/KL) + performance where labels exist; alert; retrain.
*5/5 adds:* the three drift kinds (covariate/label/concept) with different responses; label-delay-aware monitoring; champion-challenger instead of blind retrain; "retraining on drifted data without understanding *why* just launders the drift into the model."
*Your score: ___*

---

## Segment 4 — ML Depth Round 2: Project Deep-Dive (45 min)

**Interviewer script:** "Pick your strongest project. Walk me through it." Then drill with five consecutive whys on whatever you say (the pattern from the [Deep-Dive Prep guide](Project%20Deep-Dive%20Prep.md)).

Run it against your own project now. If your five-why chain broke before level 4 — that's the finding; go back to the guide's worksheet and fix exactly that chain.

**Scoring:** did you (a) open with the one-liner, (b) state a rejected alternative for every choice, (c) volunteer a failure mode before being asked, (d) stay calm when the interviewer pushed on a weak spot? Each yes = 1 point. 3+/4 = pass.

---

## Segment 5 — System Design (60 min)

**Prompt (set the clock, whiteboard/notebook):**

> "Design a real-time coupon-recommendation system for a food-delivery app: 30M users, 500K restaurant pages, show 3 personalized coupons at checkout, <150ms p99, retrain daily. Coupons have inventory limits and margins."

**The 90-minute (compressed to 60) expected flow:**

1. **Clarify (5 min):** objective — maximize redemption margin, not clicks; inventory constraints; what's the fallback (popular coupons); what data exists (order history, browsing, coupon history).
2. **Two-stage (10 min):** candidate retrieval (top ~200 coupons the user is eligible for, embedding-based + popularity) → ranking (GBDT/NN on user×coupon features with margin-aware objective). Name the two-stage pattern explicitly.
3. **Data & features (10 min):** user order history aggregations, coupon redemption history, restaurant affinity, time-of-day patterns; feature store for online serving parity; the training/serving skew trap named unprompted.
4. **Serving (10 min):** precomputed user embeddings refreshed nightly + on-the-fly feature assembly at checkout; p99 budget arithmetic out loud (retrieval 20ms, features 15ms, ranker 10ms, network, slack).
5. **Constraints (10 min):** inventory — a global counter with soft reservation, decrement on serve, reconciliation; margin — expected-margin objective (P(redeem) × margin) not P(redeem); cold-start coupons → exploration traffic.
6. **Evaluation & lifecycle (10 min):** offline (calibration, expected-margin on replayed logs) → online A/B (redemption rate, margin/order, guardrails: latency, complaint rate); champion-challenger; drift on redemption distributions.

**Self-grading rubric (1 point each):** clarified the *business* objective first · named two-stage retrieval→ranking · did latency arithmetic aloud · addressed inventory/margin constraints without prompting · mentioned training-serving skew · closed with A/B + monitoring plan. **5+/6 = strong pass.**

---

## Segment 6 — Behavioral (45 min)

Interviewer script: pick 4 from the [Behavioral guide's](Behavioral%20and%20STAR%20Answers.md) question map — we recommend: a failure, a conflict, an ambiguity story, and "why this company."

Answer aloud, 2 minutes each, STAR-structured. Then the brutal part — grade yourself on:
- [ ] Result quantified in every story
- [ ] "I" not "we" in the Action beats
- [ ] No blame of teammates/timeline/data in the failure story
- [ ] Ended each with the lesson or the change
- [ ] Under 2:15 per answer (rambling = anxiety signal)

---

## Scoring the Whole Loop

| Segment | Pass bar | Strong pass |
| --- | --- | --- |
| Recruiter screen | pitch < 90s, comp range ready | specific team-fact landing |
| Coding | 3/5 rubric | 5/5 + distributed follow-up solid |
| ML depth 1 | 3 questions at pass level | two questions at 5/5 |
| ML depth 2 | 4-why chain survives | 5-why chain + volunteered failure mode |
| System design | 3/6 rubric | 5/6 with arithmetic |
| Behavioral | all boxes checked | stories differentiated + quantified |

**Interpretation:** all segments at pass bar = you're ready to interview. Any segment below pass = that's tomorrow's study block (the segment names map 1:1 to folders 2–5 of this section). Two+ segments below = repeat the whole mock in three days after targeted study.

**The calibration note that matters:** every interviewer here was checking *how you handle the moments you didn't know* — the "let me think about that" pause, the reasoned guess with its assumptions named, the "I don't know, but here's how I'd find out." Those moments, handled well, are worth more than any prepared answer.
