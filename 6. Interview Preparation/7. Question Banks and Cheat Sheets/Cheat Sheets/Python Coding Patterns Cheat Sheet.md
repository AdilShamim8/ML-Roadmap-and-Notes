# Python Coding Patterns Cheat Sheet

One page. The 20 patterns from the [coding round notebook](../../2.%20Coding%20Rounds/Python%20Coding%20Round.ipynb), one recognizable line each — pattern recognition is the skill under time pressure.

## The pattern → first-move map
| Problem signal | Pattern | First move |
| --- | --- | --- |
| Sorted array, find pair/sum | Two pointers | converge from both ends |
| Contiguous subarray/substring | Sliding window | expand right, shrink left |
| Many range-sum queries | Prefix sums | precompute cumulative |
| "K most frequent / top K" | Heap | `heapq.nlargest(k, ...)` |
| Monotone yes/no predicate, min/max satisfying | Binary search on answer | `while lo < hi: mid` |
| Linked list cycle / midpoint | Fast & slow pointers | `slow, fast = slow.next, fast.next.next` |
| Overlapping ranges | Sort + merge intervals | sort by start, merge on overlap |
| Shortest path, unweighted | BFS | deque, mark visited at enqueue |
| Count ways / min cost / feasibility | DP | define state, recurrence, base — in words first |
| All subsets / permutations / combinations | Backtracking | choose → explore → un-choose |
| Parse with pairs (roman, nested) | Lookup table + lookahead | map + one-pass with rule |
| Matrix layers (spiral, rotate) | Boundaries or rotate-and-pop | shrink walls / pop-row-rotate |
| "Group by key" | dict-of-dicts / defaultdict | `defaultdict(lambda: defaultdict(int))` |

## Complexity table (say it before they ask)
| Structure/algorithm | Time | Space |
| --- | --- | --- |
| Hash map get/set | O(1) avg | O(n) |
| Binary search | O(log n) | O(1) |
| Sort | O(n log n) | O(n) |
| BFS/DFS | O(V+E) | O(V) |
| Heap push/pop | O(log n) | O(n) |
| Two pointers / sliding window | O(n) | O(1) |
| Subsets / permutations | O(2ⁿ) / O(n!) | — |
| kadane (max subarray) | O(n) | O(1) |

## The one-liners that must be instant
```python
s[::-1]                                  # reverse string/list
" ".join(s.split()[::-1])                # reverse words
[c.lower() for c in s if c.isalnum()]    # normalize for palindrome
Counter(a) == Counter(b)                 # anagram check
heapq.nlargest(k, d.items(), key=kv[1])  # top-k
itertools.combinations(xs, 2)            # pairs
enumerate(xs, start=1)                   # index + value
zip(a, b, strict=True)                   # parallel iteration w/ length check
df.groupby(g).agg(...)                   # (pandas, but say it in python rounds too)
```

## Data-structure reflexes
```python
# counting / grouping — four ways, prefer the last
d.get(k, 0) + 1  ·  setdefault  ·  Counter  ·  defaultdict(int)

# ordered unique
list(dict.fromkeys(xs))                  # dedupe, keep order (3.7+)

# priority queue
import heapq; heapq.heappush(h, (priority, item)); heapq.heappop(h)

# fixed-size deque (sliding window max, recent-K)
from collections import deque; q = deque(maxlen=k)

# union-find (connectivity problems)
parent = list(range(n))
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b): parent[find(a)] = find(b)
```

## The trap checklist (run before declaring done)
```python
[] and None and 0 and ""        # falsy edge cases — which are valid input?
x = float("inf"); x = float("-inf")   # init min/max correctly
range(len(s) - 1)               # off-by-one in pair loops
visited AT ENQUEUE (BFS)        # not at dequeue — else duplicates enqueue
path.copy() in backtracking     # or you append a mutating reference
mutable default args = NO       # def f(x, acc=None): acc = acc or []
is vs ==: only 'is None'        # identity for singletons only
```

## Interview protocol (the 6 beats)
1. **Restate + edge cases** — "empty input? negatives? duplicates? bounds?"
2. **Brute force + complexity, then improve** — show the path.
3. **Narrate while coding.**
4. **Tiny test out loud** — one normal case, one edge case.
5. **State time AND space** — "what if n doubles?"
6. **Stuck? Say the brute force, look for sortedness / hashing / monotone predicates.**

## Complexity estimation drills
- Nested loop over same array but inner starts at i → **O(n²)/2 = O(n²)**, but say "i,j pairs" — sometimes the answer is O(n log n) with sort+two-pointer.
- Hashing inside a loop → O(n × cost-of-hash); strings hash in O(len).
- Slicing `a[i:j]` costs **O(j−i)** — `s = s + x` in a loop is quadratic; use join.
- Recursion: T(n) = 2T(n/2) + O(n) → O(n log n) (merge sort); T(n) = T(n/2) + O(1) → O(log n).
