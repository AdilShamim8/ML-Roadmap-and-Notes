# Deep Learning Cheat Sheet

One page. The architecture math, the training recipes, and the vocabulary — calibrated to what DL rounds actually probe.

## The layer equations (write these cold)
```
Dense:      z = Wx + b;        a = σ(z)
Sigmoid:    σ(z) = 1/(1+e⁻ᶻ);          σ' = σ(1−σ)
tanh:       tanh' = 1 − tanh²
ReLU:       max(0, z);          ReLU' = 1 if z>0 else 0     (dying ReLU at z≤0)
Softmax:    softmax(z)ᵢ = eᶻⁱ/Σeᶻʲ    (subtract max for stability)
BCE grad:   δ = ŷ − y            ← the magic: sigmoid+BCE gives this clean form
```

## Backprop, the recipe
```
forward:  z₁ = XW₁+b₁ → a₁ = f(z₁) → z₂ = a₁W₂+b₂ → L
backward: δ₂ = ∂L/∂z₂ (for BCE+sigmoid: ŷ−y)
          dW₂ = a₁ᵀδ₂;  db₂ = Σδ₂
          δ₁ = (δ₂W₂ᵀ) ⊙ f'(z₁)
          dW₁ = Xᵀδ₁;  db₁ = Σδ₁
chain rule, matrix form; verify with finite differences
```
(All executed in pure NumPy, gradients checked to 1e-7, in the [DL notebook](../../4.%20Deep%20Learning%20and%20Domains/Deep%20Learning%20Questions.ipynb).)

## Initialization
| Activation | Init | Variance |
| --- | --- | --- |
| tanh / sigmoid | Xavier/Glorot | 2/(fan_in + fan_out) |
| ReLU family | He | 2/fan_in |
- Keeps activation variance ≈ 1 through depth; wrong init → vanishing/exploding from layer 1.

## Normalization & regularization stack
```
BatchNorm:  normalize per-feature per-batch → learnable γ, β
            training uses batch stats; inference uses RUNNING stats (the asymmetry!)
LayerNorm:  normalize per-sample across features — transformers/RNNs
Dropout:    zero units w.p. p in training; scale by 1/(1−p)  ("inverted" dropout)
Weight decay: L2 on weights; AdamW applies it decoupled from adaptive step
Early stopping: stop on val loss — "free lunch" regularizer
Augmentation: cheapest regularizer of all
Label smoothing: targets 1−ε/ε — fights overconfidence
```

## Optimizers (all in the [GD notebook](../../3.%20Machine%20Learning%20Interviews/Gradient%20Descent%20from%20Scratch.ipynb))
```
SGD:        w ← w − ηg
Momentum:   v ← βv + g;  w ← w − ηv        (β=0.9; damps zig-zag)
Nesterov:   gradient evaluated AFTER the momentum step (lookahead)
RMSProp:    v ← β₂v + (1−β₂)g²;  w ← w − ηg/√(v+ε)     (per-param scaling)
Adam:       m,v = EMAs of g, g²;  bias-corrected m̂, v̂;  w ← w − ηm̂/(√v̂+ε)
            (β₁=0.9, β₂=0.999 — the default)
AdamW:      Adam + decoupled weight decay (the correct L2 for adaptive methods)
Schedules:  warmup → cosine decay (or step); linear lr-scaling with batch size
```

## Architectures in one line each
| Architecture | Mechanism | Use |
| --- | --- | --- |
| CNN | learned filters slide over input; weight sharing + locality | images, signals |
| ResNet block | y = F(x) + x — identity shortcut | deep vision |
| U-Net | encoder-decoder with skip connections | segmentation |
| RNN/LSTM | recurrence + gates (input/forget/output) | legacy sequence |
| Transformer | attention, parallel over sequence | language, everything 2026 |
| Autoencoder | bottleneck reconstruction | compression, anomaly via recon error |
| GAN | generator vs discriminator adversarial | generation |
| Diffusion | iterative denoising | image generation |

## Attention (the equation to know)
```
Attention(Q,K,V) = softmax(QKᵀ/√d)·V
Q=query "what am I looking for"; K=key "what I offer"; V=value "what I pass"
√d scaling: dot products grow with d → softmax saturates → dead gradients
Multi-head: h parallel low-dim attentions, concat + project (specialization)
Positional encoding injects order (sin/cos or learned)
Self-attention: Q,K,V from the same sequence
```

## Loss functions by task
| Task | Loss |
| --- | --- |
| Binary classification | BCE (with logits for stability) |
| Multi-class | categorical cross-entropy |
| Multi-label | sigmoid + BCE per label |
| Regression | MSE ( outliers→MAE/Huber) |
| Segmentation (imbalanced) | Dice loss / focal loss |
| Embedding similarity | contrastive / triplet |

## Training debugging protocol (the "my net doesn't train" drill)
1. **Loss sanity:** random-chance level? (log(n_classes) for CE) — if below chance, labels are shuffled.
2. **Overfit 10 samples** to ~zero loss — if you can't, the bug is in the code, not the hyperparameters.
3. **Gradient norms per layer** — vanishing (→ init/norm/residuals) or exploding (→ clipping, lr).
4. **lrfinder:** short lr sweep, pick the steepest-descent region, then ~10× below the divergence point.
5. **Val loss up, train loss down** → regularize (dropout/augment/early stop) or get data.
6. **Both flat** → bigger model, better init, check the loss wiring, more epochs.

## The quick-fire add-ons
- **Mixed precision:** fp16/bf16 compute, fp32 master weights, loss scaling (fp16); 2–3× on tensor cores.
- **Gradient accumulation:** simulate large batch on small GPU memory.
- **Gradient clipping:** cap ‖g‖ (by norm, ~1.0) — RNN and transformer standard.
- **Zero/NaN loss:** fp16 overflow, bad log of 0, lr too high — in that order of likelihood.
- **Batch size ↔ lr:** linear scaling rule with warmup (Goyal et al.).
- **TTA:** test-time augmentation — average predictions over flips/crops.
- **Transfer learning:** freeze backbone → train head → unfreeze with lower lr.
