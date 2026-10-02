# SQL Cheat Sheet

A dense reference for analytical SQL: query anatomy, NULLs, aggregation, joins, CTEs, window functions, common analysis patterns, dates, and running SQL from Python with DuckDB and sqlite3. Syntax is DuckDB's (close to PostgreSQL); dialect differences are listed at the end.

Related chapters: [SQL and data acquisition](../chapters/02-data-science-workflow/01-sql-and-data-acquisition.md) · [Data cleaning](../chapters/02-data-science-workflow/02-data-cleaning.md) · [Experimentation and A/B testing](../chapters/02-data-science-workflow/05-experimentation-and-ab-testing.md)

The examples use three tables: `users(user_id, signup_at, country, device)`, `orders(order_id, user_id, order_ts, category, amount_usd, status)`, and `events(session_id, user_id, event_type, ts)`.

## Query anatomy and evaluation order

```sql
SELECT   category, COUNT(*) AS n, SUM(amount_usd) AS revenue   -- 5. compute output columns
FROM     orders                                                  -- 1. source rows (and JOINs)
WHERE    status = 'completed'                                    -- 2. filter rows
GROUP BY category                                                -- 3. form groups
HAVING   COUNT(*) >= 100                                         -- 4. filter groups
ORDER BY revenue DESC, category                                  -- 6. sort (tie-breaker!)
LIMIT    10;                                                     -- 7. keep the first rows
```

| Rule | Because |
|---|---|
| Can't use a `SELECT` alias in `WHERE` | `WHERE` runs before `SELECT` |
| Can't put aggregates in `WHERE`; use `HAVING` | Groups don't exist yet in `WHERE` |
| Can't filter on a window function in `WHERE`; use a CTE or `QUALIFY` | Windows are computed in the `SELECT` step |
| `ORDER BY` may use aliases | It runs after `SELECT` |
| Without `ORDER BY`, row order is undefined | Tables are sets; add a tie-breaker for reproducible output |

## Filtering and NULL

| Task | SQL |
|---|---|
| Comparison, ranges, lists | `amount_usd > 100`, `amount_usd BETWEEN 10 AND 50` (inclusive), `country IN ('US', 'UK')` |
| Pattern match | `email LIKE '%@example.com'` (`%` any run, `_` one char); `ILIKE` is case-insensitive |
| Missing values | `coupon IS NULL`, `coupon IS NOT NULL` (never `= NULL`) |
| NULL-safe inequality | `coupon IS DISTINCT FROM 'WELCOME10'` (keeps NULL rows) |
| Replace NULL | `COALESCE(revenue, 0)`, `NULLIF(denominator, 0)` (turns 0 into NULL to avoid division errors) |
| Conditional value | `CASE WHEN amount_usd >= 100 THEN 'large' WHEN amount_usd >= 30 THEN 'medium' ELSE 'small' END` |

Three-valued logic: any comparison with `NULL` is `NULL`, and `WHERE` keeps only `TRUE`. `NOT IN (subquery)` returns no rows if the subquery contains a `NULL`; prefer `NOT EXISTS`.

## Aggregation

| Function | Note |
|---|---|
| `COUNT(*)` vs `COUNT(col)` | rows vs non-null values |
| `COUNT(DISTINCT user_id)` | distinct non-null values |
| `SUM`, `AVG`, `MIN`, `MAX` | skip NULLs; `AVG(COALESCE(x, 0))` if NULL means zero |
| `MEDIAN(x)`, `QUANTILE_CONT(x, 0.9)` | DuckDB; PostgreSQL: `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x)` |
| `STDDEV_SAMP(x)`, `VAR_SAMP(x)`, `CORR(x, y)` | sample statistics |
| `COUNT(*) FILTER (WHERE status = 'refunded')` | conditional aggregate (portable form: `SUM(CASE WHEN ... THEN 1 ELSE 0 END)`) |
| `AVG((status = 'refunded')::INT)` | share of rows meeting a condition |
| `STRING_AGG(category, ', ')`, `LIST(category)` | collect values into a string or list |
| `GROUP BY ROLLUP (country, device)` | adds subtotals and a grand total (NULL in rolled-up columns) |
| `GROUP BY ALL` | DuckDB: group by every non-aggregated column |

## Joins

| Join | Keeps | Pattern |
|---|---|---|
| `INNER JOIN` | matching pairs only | `FROM orders o JOIN users u ON u.user_id = o.user_id` |
| `LEFT JOIN` | all left rows; NULLs where no match | all users, with orders if any |
| `FULL OUTER JOIN` | all rows from both | reconciling two sources |
| Semi join | left rows with a match, once | `WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.user_id)` |
| Anti join | left rows without a match | `WHERE NOT EXISTS (...)`, or `LEFT JOIN ... WHERE o.order_id IS NULL` |
| `CROSS JOIN` | every pair | building a grid (dates × categories) |
| `USING (user_id)` | shorthand for equal-named keys | merges the key into one column |

Join size: an equality join returns $\sum_k r_k s_k$ rows, where $r_k$ and $s_k$ count key $k$ on each side. If the key isn't unique on the "one" side, rows **fan out** and sums inflate. Check before joining:

```sql
SELECT COUNT(*) AS n_rows, COUNT(DISTINCT user_id) AS n_keys FROM users;   -- must be equal
```

Put conditions on the *right* table of a `LEFT JOIN` in the `ON` clause. In `WHERE`, they turn it into an inner join:

```sql
SELECT u.user_id, COUNT(o.order_id) AS orders_7d
FROM users u
LEFT JOIN orders o
       ON o.user_id = u.user_id
      AND o.order_ts < u.signup_at + INTERVAL 7 DAY      -- in ON: users with no orders survive
GROUP BY u.user_id;
```

## Subqueries and CTEs

```sql
WITH revenue_per_user AS (               -- name each step; read top to bottom
    SELECT user_id, SUM(amount_usd) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY user_id
)
SELECT u.country, AVG(COALESCE(r.revenue, 0)) AS revenue_per_customer
FROM users u
LEFT JOIN revenue_per_user r USING (user_id)
GROUP BY u.country;
```

| Form | Example |
|---|---|
| Scalar subquery | `SELECT (SELECT COUNT(*) FROM orders) AS n_orders` |
| Derived table | `FROM (SELECT ...) AS t` |
| Filter subquery | `WHERE user_id IN (SELECT user_id FROM vip)` |
| Correlated subquery | `WHERE amount_usd > (SELECT AVG(amount_usd) FROM orders o2 WHERE o2.category = o.category)` |
| View | `CREATE OR REPLACE VIEW orders_clean AS SELECT ...` (`TEMP VIEW` for session-only) |

## Window functions

`function(...) OVER (PARTITION BY ... ORDER BY ... frame)` computes a value per row from related rows, without collapsing them.

| Function | Returns |
|---|---|
| `ROW_NUMBER()` | 1, 2, 3, ... (ties broken arbitrarily: add a tie-breaker) |
| `RANK()`, `DENSE_RANK()` | ranks with ties; `RANK` skips numbers after ties |
| `NTILE(10)` | bucket number 1 to 10 (deciles) |
| `PERCENT_RANK()`, `CUME_DIST()` | relative rank in [0, 1] |
| `LAG(x, 1)`, `LEAD(x, 1)` | value from the previous or next row |
| `FIRST_VALUE(x)`, `LAST_VALUE(x)` | first or last value in the frame |
| `SUM(x) OVER (...)` | running or windowed sum (any aggregate works) |

| Frame | Meaning |
|---|---|
| (none, with `ORDER BY`) | `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`: running total; ties share a value |
| `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` | the current row and the 6 rows before it |
| `RANGE BETWEEN INTERVAL 6 DAYS PRECEDING AND CURRENT ROW` | a true 7-day window, even with missing days |
| `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` | the whole partition (needed for a correct `LAST_VALUE`) |

```sql
SELECT user_id, order_id, order_ts,
       ROW_NUMBER() OVER w                              AS order_number,
       date_diff('day', LAG(order_ts) OVER w, order_ts) AS days_since_prev,
       SUM(amount_usd) OVER w                           AS lifetime_value_so_far
FROM orders
WINDOW w AS (PARTITION BY user_id ORDER BY order_ts, order_id);
```

## Common analysis patterns

**First (or latest) row per group**, also used to deduplicate keeping the newest version:

```sql
SELECT *
FROM orders
QUALIFY ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_ts) = 1;   -- DuckDB, Snowflake, BigQuery
-- elsewhere: wrap in a CTE and filter WHERE rn = 1
```

**Top N per group:**

```sql
SELECT category, order_id, amount_usd
FROM orders
QUALIFY RANK() OVER (PARTITION BY category ORDER BY amount_usd DESC) <= 3;
```

**Share of total:**

```sql
SELECT category, SUM(amount_usd) AS revenue,
       SUM(amount_usd) / SUM(SUM(amount_usd)) OVER () AS share   -- window over the aggregate
FROM orders GROUP BY category;
```

**Funnel with conditional aggregation:**

```sql
SELECT COUNT(DISTINCT session_id)                                            AS sessions,
       COUNT(DISTINCT session_id) FILTER (WHERE event_type = 'add_to_cart')  AS carts,
       COUNT(DISTINCT session_id) FILTER (WHERE event_type = 'checkout')     AS checkouts,
       COUNT(DISTINCT session_id) FILTER (WHERE event_type = 'purchase')     AS purchases
FROM events;
```

**Daily series with no missing days** (a date spine), then a 7-day moving average:

```sql
WITH days AS (
    SELECT CAST(d AS DATE) AS day
    FROM generate_series(DATE '2024-01-01', DATE '2024-12-31', INTERVAL 1 DAY) AS t(d)
),
daily AS (
    SELECT CAST(order_ts AS DATE) AS day, SUM(amount_usd) AS revenue
    FROM orders GROUP BY 1
)
SELECT day, COALESCE(revenue, 0) AS revenue,
       AVG(COALESCE(revenue, 0)) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma_7d
FROM days LEFT JOIN daily USING (day)
ORDER BY day;
```

**Signup cohorts and retention:**

```sql
WITH cohort AS (
    SELECT user_id, date_trunc('month', signup_at) AS cohort_month FROM users
),
activity AS (
    SELECT DISTINCT o.user_id, c.cohort_month,
           date_diff('month', c.cohort_month, date_trunc('month', o.order_ts)) AS months_since
    FROM orders o JOIN cohort c USING (user_id)
)
SELECT cohort_month, months_since, COUNT(*) AS active_users
FROM activity
GROUP BY ALL ORDER BY cohort_month, months_since;
```

**Pivot** (categories to columns):

```sql
SELECT country,
       SUM(amount_usd) FILTER (WHERE category = 'books')       AS books,
       SUM(amount_usd) FILTER (WHERE category = 'electronics') AS electronics
FROM orders JOIN users USING (user_id)
GROUP BY country;
-- DuckDB also has: PIVOT orders ON category USING SUM(amount_usd) GROUP BY user_id;
```

**Data-quality checks:**

```sql
SELECT COUNT(*) - COUNT(DISTINCT order_id)                       AS duplicate_keys,
       COUNT(*) FILTER (WHERE amount_usd IS NULL)                AS missing_amounts,
       COUNT(*) FILTER (WHERE amount_usd < 0)                    AS negative_amounts,
       MIN(order_ts) AS first_ts, MAX(order_ts) AS last_ts
FROM orders;

SELECT COUNT(*) AS orphan_orders                                  -- foreign keys with no parent
FROM orders o WHERE NOT EXISTS (SELECT 1 FROM users u WHERE u.user_id = o.user_id);
```

## Dates, times, and strings (DuckDB)

| Task | SQL |
|---|---|
| Parse text with an offset | `CAST('2024-11-23 11:49:42+05:30' AS TIMESTAMPTZ)` |
| Parse with a format | `strptime('23/11/2024', '%d/%m/%Y')` |
| Epoch milliseconds to timestamp | `epoch_ms(ts_ms)`; seconds: `to_timestamp(ts_s)` |
| Session time zone | `SET TimeZone = 'UTC'` (affects `date_trunc` and display of `TIMESTAMPTZ`) |
| Convert to local wall-clock time | `order_ts AT TIME ZONE 'Europe/Berlin'` |
| Truncate | `date_trunc('month', ts)`, `CAST(ts AS DATE)` |
| Parts | `extract(hour FROM ts)`, `dayofweek(ts)` (0 = Sunday), `strftime(ts, '%Y-%m')` |
| Differences | `date_diff('day', start_ts, end_ts)`, `end_ts - start_ts` (an interval) |
| Arithmetic | `ts + INTERVAL 7 DAY`, `ts - INTERVAL 1 HOUR` |
| Clean strings | `lower(trim(category))`, `upper(country)`, `replace(s, ',', '')` |
| Substrings and regex | `substr(s, 1, 4)`, `regexp_matches(s, '^[A-Z]{2}$')`, `regexp_replace(s, '\s+', ' ', 'g')` |
| Concatenate | `first_name || ' ' || last_name`, `concat(a, b)` |
| Cast | `CAST(x AS DOUBLE)`, `x::INTEGER`, `TRY_CAST(x AS INTEGER)` (NULL instead of error) |

## DuckDB and sqlite3 from Python

```python
import duckdb
import pandas as pd

con = duckdb.connect()                          # in memory; duckdb.connect("shop.duckdb") for a file
con.sql("SET TimeZone = 'UTC'")

con.register("orders", orders_df)               # query a pandas DataFrame as a table (no copy)
df = con.sql("SELECT status, COUNT(*) AS n FROM orders GROUP BY status").df()   # -> pandas
n = con.sql("SELECT COUNT(*) FROM orders").fetchone()[0]                        # -> Python scalar

con.sql("SELECT * FROM 'orders.parquet' WHERE amount_usd > 100")                # query files directly
con.sql("SELECT * FROM read_csv('raw/*.csv', header = true)")                   # globs work too
con.sql("COPY (SELECT * FROM orders) TO 'orders.parquet' (FORMAT parquet)")     # write files

country = "UK"                                   # user input: always pass as a parameter
con.execute("SELECT COUNT(*) FROM users WHERE country = ?", [country]).fetchone()

con.sql("CREATE TABLE orders_tbl AS SELECT * FROM orders_df")   # copy a DataFrame in (found by variable name)
```

```python
import sqlite3

lite = sqlite3.connect("shop.db")                # or ":memory:"
orders_df.to_sql("orders", lite, index=False, if_exists="replace")
df = pd.read_sql("SELECT * FROM orders WHERE status = ?", lite, params=("completed",))
lite.close()
```

| Use | DuckDB | sqlite3 |
|---|---|---|
| Workload | analytics: scans, aggregations, joins over many rows | transactions: many small reads and writes |
| Storage | columnar | row-oriented |
| Reads DataFrames, Parquet, CSV directly | yes | no |
| Installed with Python | no (`pip install duckdb`) | yes |

## Dialect differences that bite

| Behavior | DuckDB | PostgreSQL | SQLite |
|---|---|---|---|
| `7 / 2` | `3.5` | `3` (integer division) | `3` |
| Integer division explicitly | `7 // 2` | `7 / 2` | `7 / 2` |
| `QUALIFY` | yes | no (use a CTE) | no |
| `FILTER (WHERE ...)` on aggregates | yes | yes | yes (3.30+) |
| Selecting non-grouped columns | error | error | allowed, returns an arbitrary row's value |
| Median | `MEDIAN(x)` | `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x)` | none built in |
| Types | strict | strict | loose ("type affinity"): `'10' > 9` comparisons can surprise |
| Reuse aliases later in the same `SELECT` | yes | no | no |

## Gotchas

- **Fan-out**: joining to a table whose key repeats duplicates rows and inflates sums. Check `COUNT(*)` vs `COUNT(DISTINCT key)` and row counts before and after.
- **NULLs disappear silently** in `WHERE` comparisons, `<>`, and `NOT IN`. Use `IS NULL`, `IS DISTINCT FROM`, and `NOT EXISTS`.
- **`LEFT JOIN` + `WHERE` on the right table** = inner join. Move the condition into `ON`.
- **Averages need a named denominator**: per buyer or per customer? Use `COALESCE(x, 0)` deliberately.
- **`ROWS` frames count rows, not days**: fill missing days with a date spine, or use a `RANGE` interval frame.
- **Time zones**: parse offsets, store UTC, set the session time zone before `date_trunc`.
- **Integer division** in PostgreSQL and SQLite: multiply by `1.0` before dividing rates.
- **Nondeterministic order**: add a tie-breaker to `ORDER BY` and to window `ORDER BY`s used for `ROW_NUMBER()`.
- **Don't pull everything into pandas**: filter and aggregate in SQL; fetch only the small result.
