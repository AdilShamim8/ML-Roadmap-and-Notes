# Mock Interview — ML System Design (90 Minutes, Graded)

The senior round. One prompt, 90 minutes, a grader. This mock is the [ML System Design](../5.%20System%20Design/ML%20System%20Design.ipynb) and [LLM/RAG Design](../5.%20System%20Design/LLM%20and%20RAG%20System%20Design.ipynb) notebooks compressed into a simulation you run under time pressure. Do it on a whiteboard or blank notebook. The grading rubric at the end is the same shape real interviewers score against.

**Setup:** timer at 90 minutes, notebook, the 6-step frame written at the top of your page *before* you start (yes, actually do this — it forces the structure). No peeking at the notebooks until the debrief.

---

## The Prompt (Read Once, Start the Clock)

> "Design the ML system behind a 'Customers who bought this also bought...' feature for a mid-size e-commerce platform.
>
> Context: 20M SKUs, 40M monthly active users, catalog changes constantly (5% of SKUs turnover weekly), 80% of traffic is anonymous (not logged in). Product page → a rail of 12 recommended items. Availability target 99.9%, p99 latency 120ms for the recommendation block. The current system is a hand-tuned 'people who viewed X viewed Y' co-occurrence table, and the business suspects it's leaving money on the table.
>
> You have 90 minutes. Where would you like to start?"

---

## Phase Guide (How the 90 Minutes Should Break)

**0–10 min — Clarify and frame.** The questions a strong candidate asks, roughly in this order:

- *Objective:* what's the north star — click-through on the rail? add-to-cart? revenue per session? And what's the *guardrail* (cannibalization of search, diversity)?
- *The incumbent:* co-occurrence table — "that's implicit-feedback collaborative filtering, hand-rolled" — the new system must beat it *measurably* (A/B against the incumbent, not against nothing).
- *Anonymous traffic:* 80% anonymous changes everything — no user history → item-side signals dominate (item-item similarity works for anonymous; user-embedding approaches need a session representation or fallback).
- *Latency:* 120ms p99 *for the block* — includes the network hop; the model budget inside is maybe 40–60ms.
- *The catalog churn:* 5% weekly turnover → new items have no interaction history → cold-start handling is a first-class requirement, not an afterthought.
- *Availability:* 99.9% → what's the fallback when the recommender is down? (Answer to have: serve the co-occurrence table or a popularity fallback — *the existing system becomes the failure mode*, which is elegant.)

**10–25 min — Data.** Interaction logs (views, add-to-cart, purchases — implicit, abundant, position-biased); item catalog (categories, price, text, images); session context (referrer, device, time). The two traps to name: **position bias** (items shown get clicked — correction or accept bias in training data) and **feedback loops** (the recommender trains on its own outputs — exploration traffic + held-out control). Sampling/windowing: train on a rolling 30–90 days; older interactions decay in relevance for a churning catalog.

**25–50 min — Modeling, two-stage.** 
- *Retrieval:* item-item collaborative filtering (embeddings via matrix factorization or two-tower) + content-based similarity (text/image embeddings for new items — the cold-start answer) + popularity/trending per category. Union → ~200 candidates.
- *Ranking:* GBDT or small NN over user-item cross features (category affinity, price-band affinity, session context, item freshness) → top 12 with a diversity pass (category caps; 12 phone cases in a row is a failed recommendation even at great CTR).
- *Training:* daily retrain on rolling window; embeddings refreshed nightly; the *ranking* model on yesterday's logged interactions with position-bias correction (or at least the caveat said aloud).

**50–65 min — Serving architecture.** Precompute item-item top-K (the classic trick: 20M SKUs × 50 neighbors ≈ 1B entries, ~50–100GB in a KV store — *feasible*; anonymous users get item-anchored retrieval directly). At request time: product page → item ID → precomputed neighbor list → light ranking on session context → 12 items. p99 arithmetic **out loud**: KV fetch ~10ms + ranker ~15ms + network/hops ~30ms + slack. Fallback chain: ranker down → precomputed order; KV down → category popularity; total outage → hide the rail (page still works).

**65–80 min — Evaluation and lifecycle.** Offline: recall@k of retrieval, ranked CTR proxies on replayed logs, embedding sanity checks. Online: A/B against the incumbent with CTR/AOV/revenue-per-session + diversity guardrail; champion-challenger thereafter. Cold-start metrics separately (new-SKU CTR vs catalog average). Monitoring: embedding drift, feature freshness, latency dashboards, and the *feedback-loop audit* (periodic holdout of un-personalized traffic to keep an unbiased data stream).

**80–90 min — Close.** Tradeoffs and the ship plan: "I'd ship item-CF retrieval + light ranker first — it fits the latency budget, handles anonymous traffic, and beats the co-occurrence incumbent on paper; the two-tower + cross-features ranker is the v2. Cost is roughly [name the infra: one KV cluster + nightly batch training + an inference service]." Decisions with reasons, costs named, no hedging.

---

## Grading Rubric (Score Yourself, 1 Point Each)

**Framing (4)**
- [ ] Asked the objective/north-star metric before modeling
- [ ] Identified the incumbent as the A/B baseline
- [ ] Surfaced the anonymous-traffic constraint and its implication
- [ ] Named the cold-start problem from catalog churn unprompted

**Data (3)**
- [ ] Distinguished implicit feedback and named position bias
- [ ] Named the feedback-loop problem with a mitigation (exploration/holdout)
- [ ] Chose a training window and justified it

**Modeling (4)**
- [ ] Explicitly two-stage: retrieval then ranking
- [ ] Content-based path for new items
- [ ] Diversity in the final list, not just CTR
- [ ] Said what the ranker features are, concretely

**Serving (4)**
- [ ] Precompute-what-you-can principle (item-item top-K offline)
- [ ] Latency arithmetic done out loud
- [ ] Fallback chain specified at each layer
- [ ] Storage/feasibility numbers estimated (the 1B-entries arithmetic)

**Evaluation & lifecycle (3)**
- [ ] Offline metrics separated from online A/B
- [ ] Cold-start evaluated as its own metric
- [ ] Retraining cadence + drift monitoring named

**Communication (2)**
- [ ] 6-step structure visible in the flow
- [ ] Closed with tradeoffs, costs, and a ship decision

**Total: 20 points.** 12+ = pass (senior bar at most companies), 16+ = strong senior, 19+ = staff-level signal. Below 12: the phase guide above *is* your study plan — re-read the matching section of the [ML System Design notebook](../5.%20System%20Design/ML%20System%20Design.ipynb), then re-run this mock with a different prompt (swap "recommendation rail" for "search ranking" or "fraud rules + ML hybrid" and the same frame applies).

---

## The Variants to Practice (One Per Day After This)

The same 6-step frame, different stress points:

1. **Search ranking** (relevance + business weights) — stresses the *evaluation* phase (interleaving experiments, NDCG).
2. **Fraud rules + ML hybrid** — stresses *label delay* and *threshold economics* (see the fraud design in the notebook).
3. **RAG over support tickets** — stresses retrieval quality and *grounding* (the whole [LLM/RAG notebook](../5.%20System%20Design/LLM%20and%20RAG%20System%20Design.ipynb)).
4. **Dynamic pricing** — stresses *causality* (you never observe the counterfactual price) and *guardrails* (bounds, kill switches).
5. **Notification timing/suppression** — stresses *uplift thinking* and fatigue guardrails.

One frame, five rehearsals — by the third variant the structure is reflex and your attention goes entirely to the problem's actual difficulty, which is precisely the level interviewers are testing for.
