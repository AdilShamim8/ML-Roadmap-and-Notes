# NumPy Cheat Sheet

One page. Array-first thinking, the interview-grade idiom for each.

## Creation
```python
np.zeros((3, 4)); np.ones(5); np.full((2, 2), 7); np.eye(3)
np.arange(0, 10, 2)          # step-spaced
np.linspace(0, 1, 5)         # count-spaced (endpoint included)
rng = np.random.default_rng(0)        # the modern generator — say this, not np.random.seed
rng.normal(0, 1, (3, 3)); rng.integers(0, 10, 5); rng.uniform(0, 1, 5)
```

## Broadcasting (right-to-left rule)
```
(3, 4) + (4,)     ✓ trailing dims match after stretch
(3, 1) + (1, 4)   ✓ both stretch -> (3, 4)
(3, 4) + (3,)     ✗ 4 vs 3 mismatch -> reshape to (3, 1)
```
Each pair of trailing dims must be **equal or 1** (or absent). Fix failures with `.reshape(-1, 1)` (column) or `[None, :]` (row).

## Axis semantics
```python
M.sum(axis=0)    # collapse rows -> one value per COLUMN (down)
M.sum(axis=1)    # collapse cols -> one value per ROW (across)
M.mean(axis=0, keepdims=True)   # preserve shape for broadcasting
```
Mnemonic: **axis = the direction that disappears.**

## Selection & filtering
```python
x[x > 3]                          # boolean mask (returns a COPY)
x[(x > 1) & (x < 5)]              # & | with parens — same trap as pandas
np.where(cond, a_val, b_val)      # vectorized if/else
np.select([c1, c2], [v1, v2], default=v3)
x.argmax() / x.argmin()           # INDEX of extreme, not the value
np.isnan(x); np.nanmean(x)        # NaN-safe family
idx = np.argsort(x); x[idx]       # sort by permutation
```

## Views vs copies
```python
v = a[1:4]        # basic slice -> VIEW (mutating v mutates a!)
c = a[1:4].copy() # explicit copy
f = a[a > 2]      # fancy indexing -> COPY
np.shares_memory(a, v)   # the check to say out loud
a.reshape(-1, 1)  # usually a view
a.T               # view
a.flatten()       # always copy; a.ravel() usually a view
```

## The one-liners interviews love
```python
# row z-scores
(z := (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True))
# ReLU
np.maximum(x, 0)
# stable softmax
e = np.exp(v - v.max()); e / e.sum()
# pairwise squared distances  X:(n,d) Y:(m,d) -> (n,m)
((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
# one-hot labels -> (n, k)
(labels[:, None] == np.arange(k)).astype(int)
```

## Linear algebra
```python
A @ b                    # matmul operator (prefer over np.dot in 2026)
np.linalg.solve(A, b)    # Ax=b — NEVER inv(A) @ b (slower + unstable)
np.linalg.lstsq(X, y)    # least squares
np.linalg.norm(v); np.linalg.norm(A, axis=1)   # vector / row norms
np.linalg.eigh(S)        # symmetric eigendecomposition (covariance!)
U, S, Vt = np.linalg.svd(X, full_matrices=False)   # SVD
np.linalg.cond(A)        # condition number (instability check)
```

## Aggregations & utility
```python
x.cumsum(); x.cumprod()
np.unique(x, return_counts=True)
np.clip(x, lo, hi); np.round(x, 2)
np.interp(missing_idx, good_idx, good_vals)   # 1-D linear interpolation
np.bincount(labels)       # class counts, fast
a.reshape(3, 4); a.ravel(); np.concatenate([a, b]); np.vstack / np.hstack
```

## dtype discipline
```python
x.astype(np.float32)      # explicit casts — never accidental
np.int8 max 127           # wraps silently past it
np.float16                # 30000 + 1 == 30000 (precision vanishes)
```

## Performance idioms (the "why" to say)
- Vectorize: contiguous typed memory + C loops + SIMD — measured ~50–200× over Python loops.
- Views cost O(1); copies cost O(n) — know which one you just made.
- `np.einsum` for multi-operand contractions when one line beats three.
