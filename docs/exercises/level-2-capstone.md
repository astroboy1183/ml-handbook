# Level 2 capstone: a full data science investigation

> **Level 2 · Capstone** · ⏱️ 10–15 h · Prerequisites: all of [Level 2](../chapters/02-data-science-workflow/index.md)

This capstone puts the whole workflow together on one realistic problem: query a messy database with SQL, clean it, explore it, analyze an A/B test end to end, and write a one-page recommendation that a busy executive could act on. It's also the gate to the Intermediate tier. Finish it without notes before moving on to [Level 3](../chapters/03-ml-fundamentals/index.md).

## The scenario

You've just joined ShopCo, the online store from Level 2, as its data scientist. On your first day, the Head of Growth gives you two requests.

1. **A 2024 health check.** "How did 2024 go? Where's revenue coming from, how concentrated is it, and where are we losing customers? I've heard our data has problems, so tell me what you had to fix."
2. **A decision on the carousel test.** From 6 to 26 January 2025, ShopCo tested a "recommended for you" carousel shown to new customers right after signup. "The product manager says it won and wants to ship it this week. Is that right? I need a recommendation I can defend."

The experiment was pre-registered as follows:

| Item | Plan |
|---|---|
| Unit of randomization | New customer (`user_id`), assigned at signup |
| Population | Customers who signed up between 6 and 26 January 2025 |
| Variants | `control` (no carousel) and `treatment` (carousel), 50/50 |
| Primary metric | Share of assigned customers with a **completed** order within **7 days** of assignment |
| Guardrails | Order value among converting customers; share of 7-day orders cancelled or refunded |
| Test | Two-sided, $\alpha = 0.05$, power 80% |
| Minimum detectable effect | +1.2 percentage points from a baseline of 10.8% |
| Duration | 3 weeks of assignment, plus 7 days of follow-up |

## The data

Expand the generator, save it as `make_shop_data.py`, and run it (or run the cells below). It writes a DuckDB database file, `shopco.duckdb`, with four tables of **raw** data:

- **users**: one row per customer (in theory): `user_id`, `signup_at` (UTC, as text), `country`, `device`, `channel`, `age`. Includes 2024 signups and the January 2025 experiment signups.
- **events**: website and app events for 2024 sessions: `event_id`, `session_id`, `user_id`, `event_type` (`page_view`, `add_to_cart`, `checkout`, `purchase`), `ts_ms` (Unix epoch milliseconds).
- **orders**: one row per order, 2024 to February 2025: `order_id`, `user_id`, `order_ts` (text, mixed formats), `category`, `amount_usd`, `n_items`, `status`, `coupon`.
- **assignments**: the experiment's assignment log: `user_id`, `experiment`, `variant`, `assigned_at` (UTC, as text), `app_version` (`web` or an Android app version).

Assume the data has every problem you met in Level 2. It may also have one you haven't met.

??? example "The capstone data generator: make_shop_data.py (expand and run this first)"

    ```python
    """Generate the Level 2 capstone database: ShopCo's 2024 history plus a January 2025 A/B test.

    Usage:
        python make_shop_data.py                    # writes shopco.duckdb in the current directory
        from make_shop_data import make_capstone_data
        tables = make_capstone_data()               # dict of pandas DataFrames

    The data is synthetic and deliberately messy, like the data in the Level 2 chapters.
    """
    from pathlib import Path

    import duckdb
    import numpy as np
    import pandas as pd


    def make_shop(n_users=5000, seed=42):
        """Messy, seeded synthetic data for ShopCo: returns (users, events, orders)."""
        rng = np.random.default_rng(seed)
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2025-01-01", tz="UTC")

        # ---------- users ----------
        uid = np.arange(1001, 1001 + n_users)
        country = rng.choice(["US", "UK", "DE", "IN", "BR"], n_users, p=[.40, .20, .15, .15, .10])
        p_mobile = pd.Series(country).map({"US": .35, "UK": .80, "DE": .45, "IN": .80, "BR": .60}).to_numpy()
        device = np.where(rng.random(n_users) < p_mobile, "mobile",
                          rng.choice(["desktop", "tablet"], n_users, p=[.85, .15]))
        channel = rng.choice(["organic", "paid_search", "social", "email", "referral"],
                             n_users, p=[.35, .25, .20, .10, .10])
        signup = start + pd.to_timedelta(rng.uniform(0, 330, n_users), unit="D")
        age_true = np.clip(rng.normal(36, 11, n_users), 18, 80).round()
        age = age_true.copy()
        age[rng.random(n_users) < np.where(device == "mobile", .15, .04)] = np.nan  # mobile form skips age
        age[rng.choice(n_users, 25, replace=False)] = -1                           # "unknown" sentinel
        age[rng.choice(n_users, 4, replace=False)] = [230, 199, 340, 2]            # typos
        variants = {"US": ["us", "USA", "United States"], "UK": ["uk", "U.K.", "United Kingdom"],
                    "DE": ["de", "Germany"], "IN": ["India", " IN"], "BR": ["Brazil", "br "]}
        messy = rng.random(n_users) < 0.2
        country_raw = [rng.choice(variants[c]) if m else c for c, m in zip(country, messy)]
        users = pd.DataFrame({
            "user_id": uid,
            "signup_at": signup.strftime("%Y-%m-%d %H:%M:%S"),    # UTC, stored as text
            "country": country_raw, "device": device, "channel": channel, "age": age,
        })
        users = pd.concat([users, users.sample(60, random_state=seed)])  # double-submitted forms

        # ---------- sessions and funnel events ----------
        lam = pd.Series(channel).map({"email": 9, "referral": 7, "organic": 6,
                                      "social": 4, "paid_search": 4}).to_numpy()
        n_sess = rng.poisson(lam * rng.gamma(2, 0.5, n_users))
        s_user = np.repeat(np.arange(n_users), n_sess)
        s_dev, s_cty = device[s_user], country[s_user]
        # visits follow a daily rhythm in the customer's *local* time, peaking in the evening
        hour_w = np.array([3, 2, 1, 1, 1, 1, 2, 3, 4, 5, 5, 6, 7, 6, 6, 6, 6, 7, 8, 10, 11, 10, 8, 5])
        local_s = rng.choice(24, len(s_user), p=hour_w / hour_w.sum()) * 3600 + rng.uniform(0, 3600, len(s_user))
        utc_off = pd.Series(s_cty).map({"US": -5, "UK": 0, "DE": 1, "IN": 5.5, "BR": -3}).to_numpy()
        day = (signup[s_user] + (end - signup[s_user]) * rng.random(len(s_user))).floor("D")
        s_start = day + pd.to_timedelta(local_s - utc_off * 3600, unit="s")
        s_start = s_start.where(s_start > signup[s_user], signup[s_user] + pd.Timedelta(minutes=5))
        s_start = s_start.where(s_start < end - pd.Timedelta(hours=1), end - pd.Timedelta(hours=1))
        p_buy = np.select([s_dev == "desktop", s_dev == "tablet"], [.85, .70], .40) + .12 * (s_cty == "UK")
        cart = rng.random(len(s_user)) < .35
        checkout = cart & (rng.random(len(s_user)) < .60)
        purchase = checkout & (rng.random(len(s_user)) < p_buy)
        t_cart = s_start + pd.to_timedelta(rng.uniform(30, 600, len(s_user)), unit="s")
        t_chk = t_cart + pd.to_timedelta(rng.uniform(30, 300, len(s_user)), unit="s")
        t_buy = t_chk + pd.to_timedelta(rng.uniform(10, 120, len(s_user)), unit="s")
        sid = np.arange(1, len(s_user) + 1)
        parts = []
        for name, mask, t in [("page_view", np.ones(len(s_user), bool), s_start), ("add_to_cart", cart, t_cart),
                              ("checkout", checkout, t_chk), ("purchase", purchase, t_buy)]:
            parts.append(pd.DataFrame({"session_id": sid[mask], "user_id": uid[s_user][mask],
                                       "event_type": name, "ts_ms": t[mask].as_unit("ms").asi8}))
        events = pd.concat(parts).sort_values("ts_ms", ignore_index=True)
        events = pd.concat([events, events.sample(frac=.01, random_state=seed)])  # client retries
        events = events.sort_values("ts_ms", ignore_index=True)
        events.insert(0, "event_id", np.arange(1, len(events) + 1))

        # ---------- orders (one per purchase session) ----------
        o_idx = np.flatnonzero(purchase)
        n_o = len(o_idx)
        cats = np.array(["electronics", "home", "fashion", "books", "beauty"])
        cat = rng.choice(cats, n_o, p=[.20, .25, .25, .15, .15])
        median = pd.Series(cat).map({"electronics": 120, "home": 60, "fashion": 45,
                                     "books": 18, "beauty": 30}).to_numpy()
        n_items = 1 + rng.poisson(.6, n_o)
        spend = median * n_items ** .7 * np.exp(.01 * (age_true[s_user][o_idx] - 36))
        amount = np.round(spend * rng.lognormal(0, .5, n_o), 2)
        status = rng.choice(["completed", "cancelled", "refunded"], n_o, p=[.90, .05, .05])
        amount[(status == "refunded") & (rng.random(n_o) < .5)] *= -1          # some refunds stored negative
        amount[rng.random(n_o) < .015] = np.nan                                 # lost in a migration
        amount[rng.choice(n_o, 4, replace=False)] = 99999.0                     # QA test orders
        cat_raw = cat.astype(object)
        r = rng.random(n_o)
        cat_raw[r < .10] = np.char.title(cat[r < .10].astype(str))
        cat_raw[(r >= .10) & (r < .15)] = np.char.upper(cat[(r >= .10) & (r < .15)].astype(str))
        cat_raw[(r >= .15) & (r < .20)] = np.char.add(cat[(r >= .15) & (r < .20)].astype(str), " ")
        coupon = rng.choice(np.array([None, "WELCOME10", "SPRING20", "FREESHIP"], dtype=object),
                            n_o, p=[.80, .08, .06, .06])
        # timestamps: web writes UTC with "Z"; the mobile app writes local time with an offset
        t = pd.Series(t_buy[o_idx])
        o_dev, o_cty = s_dev[o_idx], s_cty[o_idx]
        ts = t.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        tzs = {"US": "America/New_York", "UK": "Europe/London", "DE": "Europe/Berlin",
               "IN": "Asia/Kolkata", "BR": "America/Sao_Paulo"}
        for c, tz in tzs.items():
            m = (o_dev == "mobile") & (o_cty == c)
            local = t[m].dt.tz_convert(tz).dt.strftime("%Y-%m-%d %H:%M:%S%z")
            ts[m] = local.str[:-2] + ":" + local.str[-2:]
        orders = pd.DataFrame({
            "order_id": np.arange(500001, 500001 + n_o), "user_id": uid[s_user][o_idx],
            "order_ts": ts.to_numpy(), "category": cat_raw, "amount_usd": amount,
            "n_items": n_items, "status": status, "coupon": coupon,
        })
        orphans = orders.sample(15, random_state=seed).assign(user_id=lambda d: d.user_id + 90000,
                                                              order_id=lambda d: d.order_id + 50000)
        orders = pd.concat([orders, orphans, orders.sample(frac=.02, random_state=seed + 1)])
        orders = orders.sample(frac=1, random_state=seed).reset_index(drop=True)
        return users.reset_index(drop=True), events, orders


    def make_capstone_data(n_users=8000, seed=2043):
        """Return a dict of raw tables: users, events, orders, assignments."""
        users, events, orders = make_shop(n_users=n_users, seed=seed)
        rng = np.random.default_rng(seed + 1)

        # ---------- January 2025: the "recommended for you" carousel experiment ----------
        start = pd.Timestamp("2025-01-06", tz="UTC")
        n = 24_000                                    # new customers who signed up during the 3-week test
        uid = np.arange(200_001, 200_001 + n)
        assigned = start + pd.to_timedelta(rng.uniform(0, 21, n), unit="D")
        country = rng.choice(["US", "UK", "DE", "IN", "BR"], n, p=[.40, .20, .15, .15, .10])
        device = rng.choice(["mobile", "desktop", "tablet"], n, p=[.55, .38, .07])
        channel = rng.choice(["organic", "paid_search", "social", "email", "referral"], n, p=[.35, .25, .20, .10, .10])
        app = np.where(device == "mobile", rng.choice(["5.0.2", "5.1.0"], n, p=[.6, .4]), "web")
        variant = np.where(rng.random(n) < 0.5, "treatment", "control")
        base = np.select([device == "desktop", device == "tablet"], [.16, .14], .09)
        p = base * np.where(variant == "treatment", 1.10, 1.0)          # true effect: +10% relative
        buys = rng.random(n) < p
        late = ~buys & (rng.random(n) < .03)                            # buys, but after the 7-day window
        t_order = np.where(buys, rng.uniform(0, 7, n), rng.uniform(7.5, 14, n))
        order_at = assigned + pd.to_timedelta(t_order, unit="D")
        has_order = buys | late
        k = int(has_order.sum())

        new_users = pd.DataFrame({
            "user_id": uid, "signup_at": (assigned - pd.Timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "country": country, "device": device, "channel": channel,
            "age": np.where(rng.random(n) < .1, np.nan, np.clip(rng.normal(36, 11, n), 18, 80).round()),
        })
        cats = np.array(["electronics", "home", "fashion", "books", "beauty"])
        cat = rng.choice(cats, k, p=[.20, .25, .25, .15, .15])
        median = pd.Series(cat).map({"electronics": 120, "home": 60, "fashion": 45, "books": 18, "beauty": 30}).to_numpy()
        n_items = 1 + rng.poisson(.6, k)
        amount = np.round(median * n_items ** .7 * rng.lognormal(0, .5, k), 2)
        status = rng.choice(["completed", "cancelled", "refunded"], k, p=[.90, .05, .05])
        amount[(status == "refunded") & (rng.random(k) < .5)] *= -1
        cat_raw = cat.astype(object)
        r = rng.random(k)
        cat_raw[r < .10] = np.char.title(cat[r < .10].astype(str))
        t = pd.Series(order_at[has_order])
        ts = t.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        tzs = {"US": "America/New_York", "UK": "Europe/London", "DE": "Europe/Berlin",
               "IN": "Asia/Kolkata", "BR": "America/Sao_Paulo"}
        for c, tz in tzs.items():
            m = (device[has_order] == "mobile") & (country[has_order] == c)
            local = t[m].dt.tz_convert(tz).dt.strftime("%Y-%m-%d %H:%M:%S%z")
            ts[m] = local.str[:-2] + ":" + local.str[-2:]
        new_orders = pd.DataFrame({
            "order_id": np.arange(800_001, 800_001 + k), "user_id": uid[has_order], "order_ts": ts.to_numpy(),
            "category": cat_raw, "amount_usd": amount, "n_items": n_items, "status": status,
            "coupon": rng.choice(np.array([None, "WELCOME10"], dtype=object), k, p=[.7, .3]),
        })

        assignments = pd.DataFrame({
            "user_id": uid, "experiment": "reco-carousel-v1", "variant": variant,
            "assigned_at": assigned.strftime("%Y-%m-%dT%H:%M:%SZ"), "app_version": app,
        })
        # Bug: app 5.1.0 failed to log most treatment assignments during the first week
        lost = ((assignments["variant"] == "treatment") & (assignments["app_version"] == "5.1.0")
                & (assigned < start + pd.Timedelta(days=7)) & (rng.random(n) < .9))
        assignments = assignments[~lost]
        # Cookie resets: a few users were re-assigned, sometimes to the other variant
        flip = assignments.sample(60, random_state=seed).copy()
        flip["variant"] = np.where(flip["variant"] == "treatment", "control", "treatment")
        flip["assigned_at"] = (pd.to_datetime(flip["assigned_at"]) + pd.Timedelta(hours=5)).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        assignments = pd.concat([assignments, flip, assignments.sample(frac=.01, random_state=seed + 2)])
        assignments = assignments.sample(frac=1, random_state=seed).reset_index(drop=True)

        users = pd.concat([users, new_users], ignore_index=True)
        orders = pd.concat([orders, new_orders], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
        return {"users": users, "events": events, "orders": orders, "assignments": assignments}


    def write_database(path="shopco.duckdb", **kwargs):
        """Write the raw tables into a DuckDB database file, replacing any previous copy."""
        path = Path(path)
        path.unlink(missing_ok=True)
        tables = make_capstone_data(**kwargs)
        with duckdb.connect(str(path)) as con:
            for name, df in tables.items():
                con.register("tmp_df", df)
                con.execute(f"CREATE TABLE {name} AS SELECT * FROM tmp_df")
                con.unregister("tmp_df")
        return {name: len(df) for name, df in tables.items()}
    ```

```python
sizes = write_database("shopco.duckdb")
print(sizes)
```

```text
{'users': 32060, 'events': 75251, 'orders': 9437, 'assignments': 23530}
```

## What to do

Work in order; later parts build on earlier ones. Write down your question before each query, and check important numbers two ways.

### Part A: SQL (DuckDB from Python)

1. For each table, compare the row count with the number of distinct keys. Count orders whose `user_id` doesn't exist in `users`.
2. Create provisional clean views in SQL (deduplicated, standardized countries and categories, parsed timestamps, test orders removed). Compute 2024 completed-order revenue and order counts by month, in UTC.
3. Compute the 2024 funnel by device: sessions, carts, checkouts, purchases, session conversion, and checkout-to-purchase rate. Deduplicate events first.
4. What share of 2024 revenue came from the top 10% of 2024 signups by revenue (counting customers with no orders as zero)? What share of them bought anything?
5. With window functions, find the median number of days between a customer's first and second completed orders in 2024.

### Part B: Cleaning

1. Write a cleaning function for all four tables that logs every rule with rows before and after. Decide what to do about duplicates, orphan orders, test orders, negative amounts, invalid ages, mixed time zones, inconsistent strings, duplicated assignment rows, and customers who appear in both variants. Justify each decision in a sentence.
2. Write data contracts for `orders` and `assignments`, and show that they fail on the raw data and pass on the clean data.
3. Determine whether missing age is plausibly MCAR, and say what you'd do if an analysis needed age.

### Part C: Exploratory analysis

1. Describe the distribution of 2024 order values, overall and by category. Choose summaries and scales that suit the shape.
2. Compare 2024 session conversion across countries, then within device. Is any country comparison distorted by device mix? Quantify it with standardization.
3. Write two testable hypotheses suggested by the data, each with its evidence, metric, and test.

### Part D: The A/B test

1. Verify the pre-registered sample size with a power calculation.
2. Build the analysis table: one row per assigned customer, with variant, assignment time, app version, device, and whether they had a completed order within 7 days after assignment. Do the join in SQL.
3. Check for a sample ratio mismatch. If you find one, locate its cause by slicing the data, and decide how to analyze the test validly. Explain why your fix doesn't bias the comparison.
4. Analyze the primary metric: a two-proportion z-test from scratch and with statsmodels, a 95% confidence interval for the difference, and the relative lift. Compare your corrected estimate with the naive one.
5. Check both guardrails, with an appropriate test or a bootstrap interval.
6. Compute the cumulative lift and p-value after each day of assignment. What would the product manager have concluded by peeking on day 3? On the first day with $p < 0.05$?
7. Look at the effect by device, with confidence intervals, and say what you can and can't conclude.

### Part E: Report and reproducibility

1. Write a one-page report for the Head of Growth: answer first, pyramid structure, every number with its denominator and uncertainty, limitations, and next steps with owners. Include the 2024 health-check findings as context.
2. Include at least two explanatory charts with action titles.
3. Make the whole analysis rerunnable with one command, and print a fingerprint of the key result table.

## Deliverables

1. **Code**: a script (or a small package with a `run_all.py`) that rebuilds everything from `shopco.duckdb`. Notebooks are fine for exploration, but they must run cleanly top to bottom.
2. **Clean data**: Parquet files for the four cleaned tables, plus the cleaning log.
3. **Figures**: at least three, saved as PNG files.
4. **Report**: `report.md`, at most one page (about 600 words) plus an appendix.
5. **README**: the question, how to run the analysis, and where the report is.

## Acceptance checklist

- [ ] Every SQL answer in Part A runs from Python against `shopco.duckdb`, and at least one key number is cross-checked in pandas.
- [ ] Key uniqueness is checked for every table before any join, and row counts are compared before and after joins.
- [ ] Timestamps are parsed with their offsets and stored as UTC; monthly figures use UTC boundaries.
- [ ] The cleaning log lists every rule with row counts, and each rule has a one-sentence justification.
- [ ] Data contracts fail on the raw data and pass on the clean data.
- [ ] The missing-data mechanism for age is assessed with evidence, not assumed.
- [ ] The EDA uses summaries suited to skewed data and checks a country comparison for confounding by device.
- [ ] The A/B analysis checks for a sample ratio mismatch *before* looking at the primary metric, explains any mismatch, and fixes it using only pre-treatment information applied to both arms.
- [ ] The primary result is reported as a difference with a 95% confidence interval and a relative lift, computed from scratch and confirmed with a library.
- [ ] Both guardrails are checked, and "not significant" is not presented as "no effect".
- [ ] The peeking analysis explains why the pre-registered horizon matters.
- [ ] Segment results are presented with intervals and treated as hypotheses, not findings.
- [ ] The report states the recommendation in its first two sentences and fits on one page.
- [ ] Every chart has an action title and supports a claim in the text.
- [ ] One command reproduces every number, and the result fingerprint is identical across two runs.

## Hints

??? tip "Hint for Part A"
    Use `duckdb.connect("shopco.duckdb")`, then `SET TimeZone = 'UTC'`. `CAST(order_ts AS TIMESTAMPTZ)` parses both timestamp formats. For revenue concentration, `NTILE(10) OVER (ORDER BY revenue DESC)` assigns deciles; build the per-customer revenue with a `LEFT JOIN` from users so non-buyers count. For the second order, `ROW_NUMBER()` and `LAG()` over `PARTITION BY user_id ORDER BY order_ts`, then filter on the row number in an outer query.

??? tip "Hint for Part B"
    Events have a unique `event_id` but can still be duplicated: compare the columns that describe the event. In the assignment log, distinguish exact duplicate rows (safe to drop) from customers who appear with *both* variants (you can't know which experience they had). A missing `amount_usd` should survive cleaning; be careful with filters like `amount_usd < 10000`, which drop missing values silently.

??? tip "Hint for Part C"
    Build a one-row-per-session table by pivoting events. For standardization, apply each country's device-specific conversion rates to one common device mix, and compare with the overall rates.

??? tip "Hint for Part D, the analysis table"
    Put the time window in the join condition: `LEFT JOIN orders o ON o.user_id = a.user_id AND o.status = 'completed' AND o.order_ts >= a.assigned_at AND o.order_ts < a.assigned_at + INTERVAL 7 DAY`. If you filter on order columns in `WHERE` instead, customers without orders disappear.

??? tip "Hint for Part D, the sample ratio mismatch"
    Compute the share of treatment users within slices of every pre-treatment variable you have: app version, device, and assignment week. A healthy slice is near 50%. Once you find the broken slice, any exclusion rule must be based only on information known before treatment, and must be applied to *both* arms.

??? tip "Hint for Part E"
    Draft the first two sentences of the report before anything else, then check that every later paragraph supports one of them. For the revenue projection, multiply the yearly number of new customers by the lift and by the average order value of converting customers, and give a range using the confidence interval.

## Solution

When you're done, or truly stuck, compare with the [worked solution](solutions/level-2-capstone.md). It includes the full code, every output, and a sample one-page report.
