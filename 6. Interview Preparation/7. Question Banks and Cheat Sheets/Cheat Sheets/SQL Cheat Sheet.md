# SQL Cheat Sheet

One page. The query grammar that survives screens, ordered the way you'd write a query.

## The evaluation order (internalize once, never fumble again)
```
FROM → JOIN → WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → LIMIT
```
WHERE can't see aggregates (they don't exist yet); HAVING can. SELECT aliases don't exist for WHERE (they do for ORDER BY in most engines).

## The SELECT skeleton
```sql
SELECT col, AGG(other) AS alias
FROM t
JOIN t2 ON t.k = t2.k            -- inner / LEFT / RIGHT / FULL
WHERE row_condition              -- before grouping
GROUP BY col
HAVING agg_condition             -- after grouping
ORDER BY alias DESC
LIMIT 20;
```

## Joins, with the traps
```sql
FROM a JOIN b ON a.k = b.k          -- matched rows only
FROM a LEFT JOIN b ON a.k = b.k     -- all a + matched b (NULL where unmatched)
WHERE b.k IS NULL                   -- the ANTI-JOIN: rows of a with NO match
```
- Duplicate keys on the many side **multiply rows** — check `GROUP BY k HAVING COUNT(*) > 1` before joining.
- NULL never matches `= NULL` — use `IS NULL` / `IS NOT NULL`.
- `NOT IN (subquery)` returns zero rows if the subquery has any NULL — use `NOT EXISTS`.

## Aggregation
```sql
COUNT(*)                  -- all rows
COUNT(col)                -- non-null rows only  ← the interview trap
COUNT(DISTINCT col)
SUM / AVG / MIN / MAX
SUM(CASE WHEN cond THEN 1 ELSE 0 END)      -- conditional count
SUM(CASE WHEN status='shipped' THEN amount END)  -- conditional sum (ELSE NULL -> skipped)
GROUP BY grouping_sets / ROLLUP(c1, c2) / CUBE(c1, c2)   -- subtotal families
```

## NULL tools
```sql
COALESCE(a, b, 0)          -- first non-null
NULLIF(x, 0)               -- NULL when x=0: x / NULLIF(y, 0) = safe divide
```

## CASE (pivot + bucket)
```sql
CASE WHEN age < 18 THEN 'minor'
     WHEN age < 65 THEN 'adult'
     ELSE 'senior' END
-- column pivot:
SUM(CASE WHEN status = 'shipped'   THEN 1 ELSE 0 END) AS shipped,
SUM(CASE WHEN status = 'returned'  THEN 1 ELSE 0 END) AS returned
```

## Subqueries vs CTEs
```sql
WITH recent AS (          -- CTE: named, readable, chainable — the interview default
  SELECT user_id, MAX(ts) AS last_seen FROM events GROUP BY user_id
)
SELECT u.name FROM users u JOIN recent r ON u.id = r.user_id;

-- correlated (per-row; fine in interviews, flag cost in production):
SELECT name, salary FROM employees e
WHERE salary > (SELECT AVG(salary) FROM employees e2
                WHERE e2.dept_id = e.dept_id);

-- scalar subquery in SELECT/HAVING:
SELECT dept_id, AVG(salary) FROM employees
GROUP BY dept_id
HAVING AVG(salary) > (SELECT AVG(salary) FROM employees);
```

## Window functions (decorate, don't collapse)
```sql
ROW_NUMBER() OVER w   -- 1,2,3,4  (dedup, top-N per group; arbitrary tie-break)
RANK()       OVER w   -- 1,2,2,4  (ties share, gaps follow)
DENSE_RANK() OVER w   -- 1,2,2,3  (ties share, no gaps; "Nth distinct")
LAG(x) OVER w / LEAD(x) OVER w        -- previous/next row value
SUM(x) OVER (ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)  -- running total
AVG(x) OVER (ORDER BY t ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)          -- trailing 7
NTILE(4) OVER (ORDER BY v)            -- quartile buckets
FIRST_VALUE(x) OVER w / LAST_VALUE(x) OVER (w + UNBOUNDED FOLLOWING)  -- last needs the frame!
WINDOW w AS (PARTITION BY g ORDER BY t)
```
**Window functions evaluate after WHERE** — filtering on `rn` requires a CTE/subquery.

## Top-N per group (the canonical)
```sql
WITH ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY g ORDER BY v DESC) AS rn FROM t
)
SELECT * FROM ranked WHERE rn <= 2;
```

## Dates (dialect-aware — say which you're in)
```sql
-- SQLite: dates are ISO TEXT
substr(ts, 1, 7)                    -- 'YYYY-MM'
date(ts, '+7 days'); julianday(a) - julianday(b)
-- PostgreSQL:
DATE_TRUNC('month', ts)             -- month bucket
ts::date; EXTRACT(dow FROM ts); AGE(a, b)
-- MySQL:
DATE_FORMAT(ts, '%Y-%m'); DATEDIFF(a, b)
```

## Sessionization (flag + running sum)
```sql
WITH gaps AS (SELECT user_id, ts,
        LAG(ts) OVER (PARTITION BY user_id ORDER BY ts) AS prev FROM events),
flag AS (SELECT user_id, ts,
        CASE WHEN prev IS NULL
              OR ts - prev > INTERVAL '30 minutes' THEN 1 ELSE 0 END AS flag FROM gaps)
SELECT user_id, SUM(flag) OVER (PARTITION BY user_id ORDER BY ts
       ROWS UNBOUNDED PRECEDING) AS session_no FROM flag;
```

## Rapid vocabulary
| Term | One-liner |
| --- | --- |
| UNION vs UNION ALL | dedupe (sort cost) vs keep dups (fast) |
| DELETE vs TRUNCATE | logged + WHERE-able vs page-drop, fast, no WHERE |
| Index | B-tree read speed at write cost — index the queries you run |
| PK vs UNIQUE | one identity per table, no NULLs vs any count, NULLs allowed |
| ACID | atomic, consistent, isolated, durable |
| EXPLAIN | ask the engine its plan — scan vs index seek, first tool when slow |
| Transaction isolation | READ COMMITTED default; SERIALIZABLE = no anomalies, less concurrency |
| OLTP vs OLAP | row-store writes vs column-store analytics (star schema) |
