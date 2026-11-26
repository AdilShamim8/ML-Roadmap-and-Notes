# ML System Design Cheat Sheet

One page. The frame, the numbers you should be able to derive, and the cross-cutting answers — the [design notebooks](../../5.%20System%20Design/ML%20System%20Design.ipynb) compressed to a rehearsal card.

## The 6-step frame (write the headers first — always)
```
1. CLARIFY   business objective, metric, scale, latency, cost, evaluation
2. DATA      sources, volume, labels (+delay!), freshness, quality
3. MODELING  baseline → 2-3 candidates → simplest-that-could-work + upgrade path
4. EVALUATION offline metrics aligned to decision + online A/B plan + guardrails
5. SERVING   batch vs online, feature store, latency budget arithmetic, fallbacks
6. LIFECYCLE drift monitoring, retraining cadence, champion/challenger, rollback
```
**Spend 5+ minutes on 1–2 before saying "model."** Data-flow-first reads senior.

## The reference architectures
| Problem | Shape |
| --- | --- |
| Fraud / risk (real-time) | streaming features + GBDT, <100ms, rule-engine fallback, delayed labels handled |
| Recommendations | two-stage: retrieval (ANN, ~1k candidates) → ranker (heavy model, top-N + diversity) |
| Churn / CRM (batch) | weekly batch scoring, expected-value ranking (P×value), reasons to CRM |
| Search ranking | candidate gen → rank (relevance + business weights) → interleaved A/B evals |
| RAG / doc Q&A | chunk → embed → hybrid retrieve → rerank → generate with citations |
| Pricing | causal constraints, bounded outputs, kill switches |

## Capacity arithmetic (practice these out loud)
```
storage/day   = events_per_day × bytes_per_row
labels/day    = events × positive_rate          (maturity = label_delay days)
QPS           = DAU × actions / 86400 × peak_factor (~3-5×)
embedding RAM = n_vectors × dims × 4 bytes      (40M × 768d ≈ 115GB fp32)
ANN options   = HNSW (RAM-heavy, ~ms) · IVF-PQ (compressed, recall tradeoff)
latency split = network + feature_fetch + model + slack — say the numbers
retrain cadence = alert_threshold / drift_rate   (PSI 0.10 / 0.02 per month ≈ 5 months)
```

## The cross-cutting questions (raised unprompted = points)
| Concern | The answer to have |
| --- | --- |
| Training/serving skew | one feature definition, two runtimes via feature store; parity tests |
| Batch vs online scoring | stale-cheap vs fresh-latency; pick per problem, say the tradeoff |
| Label delay | train on matured labels only; delayed-feedback/PU corrections |
| Cold start | content-based + exploration traffic (bandit); popularity fallback |
| Feedback loops | model changes behavior changes tomorrow's data → exploration + held-out control + IPW |
| Position bias | clicks correlate with placement → IPW or position model (recsys/search) |
| Retraining | drift-triggered > scheduled; champion/challenger with shadow traffic |
| Privacy | PII minimization, consent, deletion propagation (feature lineage), adverse-action reasons |

## Monitoring (the layer that makes it production)
```
data:      PSI per feature (<0.1 ok, 0.1-0.25 watch, >0.25 act), null rates, schema
model:     score distribution drift, calibration over time, segment-level metrics
system:    latency p50/p99, error rate, throughput
business:  the north-star metric + guardrails (complaints, revenue, diversity)
response:  alert → diagnose (skew vs drift vs bug) → retrain or rollback → postmortem
```

## Evaluation discipline
```
offline:  CV with the RIGHT split (temporal/grouped), metrics aligned to decision cost,
          calibration if probabilities are consumed
online:   A/B with pre-registered primary metric + guardrails; A/A test to validate
          the pipeline; CUPED to cut variance; sequential design to allow early stops
champion/challenger: shadow traffic → canary → full rollout; instant rollback
```

## LLM/RAG-specific (the 2026 layer)
```
chunk:  400–800 tokens, 10–20% overlap or structure-aware; parent-small trick
retrieve: hybrid dense+BM25 → RRF fusion (score = Σ 1/(60+rank))
rerank: cross-encoder top-50 → 8      (+180ms for 5–15 quality points)
generate: temp 0–0.2, cite-or-refuse system prompt, numbered-citation validation
evaluate: retrieval recall@k FIRST → faithfulness → human sample
secure:  retrieved docs are untrusted input (injection); ACL-filter at query time
budget:  embed 12ms + retrieve 25ms + rerank 180ms + TTFT 450ms + stream 1800ms ≈ 2.5s of 3s
```

## The closing move (every design round)
End with: **decisions, costs, and the ship order** — "I'd ship [simple thing] first because [constraint], keep [stronger thing] as challenger; infra is roughly [nodes/services]; risks are [X, Y] with mitigations [A, B]." A design that ends in hedging ends the loop badly.

## Rehearsal protocol
1. Whiteboard the 6 headers before reading the prompt details.
2. Do every piece of arithmetic aloud.
3. Name one failure mode per component before the interviewer asks.
4. One full design per day for a week: fraud → recsys → churn → search → RAG (the [mock](../../6.%20The%20Final%20Push/Mock%20Interview%20-%20ML%20System%20Design.md) rotates exactly these).
