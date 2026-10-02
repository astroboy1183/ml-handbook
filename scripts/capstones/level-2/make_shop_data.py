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


if __name__ == "__main__":
    print(write_database())
