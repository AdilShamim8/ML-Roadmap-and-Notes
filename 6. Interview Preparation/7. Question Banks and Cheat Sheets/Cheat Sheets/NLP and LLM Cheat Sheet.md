# NLP and LLM Cheat Sheet

One page. Both eras: the classical pipeline that still gets tested, and the LLM stack that dominates 2026 rounds.

## Tokenization
| Level | Mechanism | Tradeoff |
| --- | --- | --- |
| Word | whitespace split | simple; vocab explosion; OOV |
| Subword (BPE/WordPiece/SentencePiece) | learned merges; frequent words whole, rare decompose | no OOV; vocab-vs-length balance (~30–50k merges) |
| Character | every char a token | no OOV; long sequences, must learn morphology |

`"unbelievable" → ["un", "believ", "able"]` — the tokenizer is baked into the pretrained model; you don't choose it separately.

## TF-IDF (write the formula)
```
tfidf(t,d) = tf(t,d) × idf(t),    idf(t) = log(N / df(t))
```
Rare-but-present = informative; everywhere ("the") ≈ 0. Hand-crafted, interpretable, no semantics ("car" ≠ "automobile"). Still the strong baseline for short text + linear models.

## Embeddings
- **Distributional hypothesis:** a word is characterized by its context.
- **word2vec/GloVe** — static: ONE vector per word ("bank" = one point, both senses).
- **Contextual (BERT/GPT/LLM-era)** — vector per *occurrence*: "bank" in "river bank" ≠ "bank loan". **Static→contextual is the one-sentence history of NLP.**
- **Sentence embeddings** — bi-encoders, mean-pooled or distilled; the retrieval workhorse.

## The transformer block
```
Attention(Q,K,V) = softmax(QKᵀ/√d)·V
x → Self-Attention → +residual → LayerNorm → FFN → +residual → LayerNorm
```
- Q=query · K=key · V=value; √d keeps softmax gradients alive.
- Multi-head: h low-dim attentions, concatenated — heads specialize.
- Positional encodings inject order (sin/cos or learned).
- Parallel over sequence (vs RNN sequential) + O(1) path between any two tokens.
- **BERT** = bidirectional encoder (mask-and-predict; embeddings/understanding). **GPT** = causal decoder (next-token; generation). Modern LLMs = decoder stacks.

## The fine-tuning ladder (2026 edition)
| Rung | What | Cost |
| --- | --- | --- |
| Prompting / in-context | examples in the prompt | 0 params |
| SFT | instruction-response pairs | full or LoRA |
| LoRA | low-rank adapters ΔW = BA on frozen base (rank r ≪ d) | 0.1–1% params |
| QLoRA | LoRA on 4-bit quantized base | 70B on one GPU |
| DPO / RLHF | preference alignment; DPO = "classify preferred over rejected" | reference-model + prefs |

**Fine-tune vs RAG (the decision table):** fine-tune for *style/format/domain behavior*; RAG for *knowledge that changes or must be cited*. Production default: RAG first, LoRA later, on top.

## RAG pipeline (the six components)
```
ingest → parse → chunk(400–800 tok, 10–20% overlap or structure-aware)
      → embed → index (HNSW ANN) → [query] → hybrid retrieve (dense+BM25, RRF)
      → rerank (cross-encoder, top-50→8) → assemble context (dedupe, lost-in-
      middle ordering, numbered citations) → generate (temp 0–0.2, cite-or-refuse)
```
- **Hybrid + RRF:** dense misses rare keywords (error codes); BM25 misses paraphrase; fuse: score = Σ 1/(60+rank).
- **Reranking:** cross-encoder scores query+doc jointly — 5–15 quality points, the cheapest big win.
- **Evaluation, in order:** retrieval recall@k → faithfulness/groundedness → end-to-end human sample. *"RAG quality is 80% retrieval quality."*

## LLM mechanics & ops
| Concept | The one-liner |
| --- | --- |
| Temperature | flattens (→creative) / sharpens (→deterministic) the sampling distribution |
| Top-p (nucleus) | sample from smallest token set with cumulative prob ≥ p |
| Top-k | sample from k most likely tokens |
| Hallucination | plausible-next-token ≠ verified fact; mitigate: RAG, low temp, citations, self-consistency |
| Long context vs RAG | quadratic attention cost + lost-in-the-middle; RAG wins for fresh/large corpora |
| Quantization | int8/int4 weights (GPTQ/AWQ); 2–4× memory cut, small quality cost |
| KV cache | cached attention keys/values — the inference-speed trick |
| Speculative decoding | small model drafts, big model verifies — 2–3× generation |
| Agents | LLM + tools + loop (plan→act→observe); reliability = function-calling accuracy + evals |
| Structured output | JSON-schema/function-calling constrained decoding |

## Classical NLP you still must know
```
Stemming (chop: studies→studi) vs Lemmatization (dictionary: studies→study)
POS tagging / NER — sequence labeling (BIO tags)
n-gram LM: P(wₙ|wₙ₋₁...wₙ₋ₙ₊₁); smoothing (Laplace +α) keeps unseen ≠ 0
BLEU: n-gram precision (MT) · ROUGE: recall-leaning (summarization) — proxies, not semantics
```

## Metrics quick-map
| Task | Standard metrics |
| --- | --- |
| Classification | accuracy, F1 (macro under imbalance) |
| IR / retrieval | recall@k, MRR, NDCG |
| Generation | BLEU/ROUGE (proxy) + human/LLM-judge (real) |
| RAG | recall@k (retrieval) + faithfulness + answer relevance |
| Embeddings | cosine similarity; retrieval evals |

## The 2026 interview phrases that land
- "Bi-encoder to retrieve, cross-encoder to rerank" — the two-stage stack.
- "I evaluate retrieval before generation — no LLM fixes documents it never saw."
- "Retrieved documents are untrusted input" — prompt injection awareness.
- "DPO is the 2026 default for preference tuning — simpler than RLHF, no reward model in the loop."
- "The tokenizer, not the model, sets your cost ceiling — tokens ≈ ¾ words for English."
