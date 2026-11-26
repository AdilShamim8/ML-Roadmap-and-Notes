# Pandas Cheat Sheet

One page. The 80% you use daily, with the interview-grade idiom for each.

## First moves on any DataFrame
```python
df.head()          # look
df.shape           # rows, cols
df.dtypes          # types — spot object-cols hiding numbers
df.describe()      # numeric summary
df.info()          # types + nulls + memory, one call
```

## Selecting
```python
df["col"]                     # Series
df[["a", "b"]]                # columns
df.loc[mask, ["a", "b"]]      # label/boolean rows + cols  ← the interview default
df.iloc[0:5, 0:3]             # positions
df.query("age > 30 and city == 'Pune'")   # SQL-ish, chain-friendly
```
Filters combine with `&` `|` and **parentheses** — `and` raises, missing parens misparse.

## Missing data
```python
df.isna().mean()                          # % null per column
df.dropna(subset=["col"])                 # drop where col null
df["col"].fillna(df["col"].median())      # fill
df["col"].ffill().bfill()                 # forward/back fill (time series)
df["col"].isna().astype(int)              # missingness indicator — often a feature!
```

## GroupBy (split → apply → combine)
```python
df.groupby("g")["v"].mean()                          # one stat
df.groupby("g").agg(n=("v", "count"),                # named multi-stat
                    mean_v=("v", "mean"),
                    max_w=("w", "max"))
df.groupby("g")["v"].transform("mean")               # result ALIGNED to rows (fill by group)
df.groupby("g")["v"].cumsum()                        # running total within group
df.groupby("g").apply(fn, include_groups=False)      # arbitrary function
df.groupby(["g1", "g2"], observed=True)              # multi-key; observed kills empty cats
```
**transform = same length** (per-row group stat) · **agg = one row per group**. The #1 interview distinction.

## Reshape
```python
df.pivot_table(index="i", columns="c", values="v", aggfunc="mean")  # wide (aggregates dups)
long = wide.reset_index().melt(id_vars="i", var_name="c", value_name="v")  # wide -> long
df.set_index("k").unstack()           # inner index level -> columns
df.stack()                            # columns -> index level
df.T                                  # transpose
```

## Joins
```python
a.merge(b, on="k", how="left", validate="many_to_one")
# how: inner (default) | left | right | outer | cross
# validate: '1:1' | '1:m' | 'm:1' | 'm:m' — let pandas enforce your assumption
a.join(b, how="left")                 # index-based
pd.concat([a, b])                     # stack rows (axis=0) or cols (axis=1)
```
Row count changed after merge? **Check key uniqueness first**: `b["k"].duplicated().sum()`.

## Sorting & ranking
```python
df.sort_values(["a", "b"], ascending=[True, False])
df["rank"] = df.groupby("g")["v"].rank(method="dense", ascending=False)
df.nlargest(10, "v")                  # top rows
df.groupby("g").head(2)               # top-2 per group (after sorting!) — the classic
```

## Strings & dates
```python
df["s"].str.lower().str.contains("ml").str.split(",").str[0]
df["d"] = pd.to_datetime(df["d"])
df["d"].dt.year / .month / .day / .dayofweek / .day_name() / .hour
df["d"].dt.to_period("M")             # month bucket
s.rolling(7).mean()                   # window over rows (sort first!)
```

## New columns
```python
df["flag"] = np.where(cond, 1, 0)                       # binary
df["cat"]  = np.select([c1, c2], ["a", "b"], default="c")  # multi-way
df = df.assign(x2=lambda d: d["x"] ** 2)               # chain-friendly
df["band"] = pd.cut(df["v"], bins=[0, 25, 50, 100], labels=["lo", "mid", "hi"])
df["q"]    = pd.qcut(df["v"], 4, labels=False)         # quartile
```

## Big-data survival
```python
pd.read_csv(path, usecols=[...], dtype={"id": "int32"}, parse_dates=["d"])
pd.read_csv(path, chunksize=100_000)    # iterate chunks, accumulate stats
df["col"] = df["col"].astype("category")  # repeated strings: big memory win
```

## Style flags interviewers watch for
- Chains wrapped in `( ... )` with a method per line — not 12 intermediate dataframes
- `inplace=True` avoided (deprecated direction of the library; assign instead)
- `df.copy()` when you'll mutate a slice
- No loops over rows (`iterrows` is a confession) — vectorize or `.map`
