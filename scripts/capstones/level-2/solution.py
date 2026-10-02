"""Level 2 capstone: a full worked solution, as one script.

Run from any directory after placing make_shop_data.py on the import path, for example:
    cd scripts/capstones/level-2
    python solution.py
It writes shopco.duckdb, clean/*.parquet, and figures/*.png into the current directory,
and prints every result shown on the solution page.
"""
import matplotlib

matplotlib.use("Agg")

from make_shop_data import write_database  # noqa: E402

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
Path("figures").mkdir(exist_ok=True)
Path("clean").mkdir(exist_ok=True)

def save_and_show(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=120, bbox_inches="tight")
    plt.show()

print(write_database("shopco.duckdb"))
con = duckdb.connect("shopco.duckdb")
con.sql("SET TimeZone = 'UTC'")

print(con.sql("""
SELECT 'users' AS tbl, COUNT(*) AS n_rows, COUNT(DISTINCT user_id) AS n_keys, 'user_id' AS key FROM users
UNION ALL SELECT 'orders', COUNT(*), COUNT(DISTINCT order_id), 'order_id' FROM orders
UNION ALL SELECT 'events', COUNT(*), COUNT(DISTINCT (session_id, event_type, ts_ms)), 'event content' FROM events
UNION ALL SELECT 'assignments', COUNT(*), COUNT(DISTINCT user_id), 'user_id' FROM assignments
""").df())
print(con.sql("""
SELECT COUNT(*) AS orphan_orders
FROM orders o WHERE NOT EXISTS (SELECT 1 FROM users u WHERE u.user_id = o.user_id)
""").df())

con.sql("""
CREATE OR REPLACE TEMP VIEW users_clean AS
SELECT DISTINCT user_id, CAST(signup_at AS TIMESTAMP) AS signup_at,
    CASE upper(trim(country))
        WHEN 'USA' THEN 'US' WHEN 'UNITED STATES' THEN 'US' WHEN 'U.K.' THEN 'UK'
        WHEN 'UNITED KINGDOM' THEN 'UK' WHEN 'GERMANY' THEN 'DE' WHEN 'INDIA' THEN 'IN'
        WHEN 'BRAZIL' THEN 'BR' ELSE upper(trim(country)) END AS country,
    device, channel, CASE WHEN age BETWEEN 13 AND 100 THEN age END AS age
FROM users
""")
con.sql("""
CREATE OR REPLACE TEMP VIEW orders_clean AS
SELECT DISTINCT order_id, user_id, CAST(order_ts AS TIMESTAMPTZ) AS order_ts,
       lower(trim(category)) AS category, abs(amount_usd) AS amount_usd, n_items, status
FROM orders
WHERE (amount_usd IS NULL OR amount_usd <> 99999)
  AND user_id IN (SELECT user_id FROM users)
""")
monthly = con.sql("""
SELECT date_trunc('month', order_ts) AS month, COUNT(*) AS orders, ROUND(SUM(amount_usd), 2) AS revenue_usd
FROM orders_clean
WHERE status = 'completed' AND order_ts < TIMESTAMPTZ '2025-01-01 00:00:00+00'
GROUP BY month ORDER BY month
""").df()
print(monthly.assign(month=monthly["month"].dt.strftime("%Y-%m")).to_string(index=False))

funnel = con.sql("""
WITH ev AS (SELECT DISTINCT session_id, user_id, event_type FROM events)
SELECT u.device,
       COUNT(DISTINCT session_id) AS sessions,
       COUNT(DISTINCT session_id) FILTER (WHERE event_type = 'add_to_cart') AS carts,
       COUNT(DISTINCT session_id) FILTER (WHERE event_type = 'checkout') AS checkouts,
       COUNT(DISTINCT session_id) FILTER (WHERE event_type = 'purchase') AS purchases,
       ROUND(100.0 * purchases / sessions, 1) AS session_conv_pct,
       ROUND(100.0 * purchases / checkouts, 1) AS checkout_to_purchase_pct
FROM ev JOIN users_clean u USING (user_id)
GROUP BY u.device ORDER BY sessions DESC
""").df()
print(funnel.to_string(index=False))

print(con.sql("""
WITH rev AS (
    SELECT u.user_id, COALESCE(SUM(o.amount_usd) FILTER (WHERE o.status = 'completed'), 0) AS revenue
    FROM users_clean u
    LEFT JOIN orders_clean o ON o.user_id = u.user_id AND o.order_ts < TIMESTAMPTZ '2025-01-01 00:00:00+00'
    WHERE u.signup_at < TIMESTAMP '2025-01-01'
    GROUP BY u.user_id
),
deciles AS (SELECT revenue, NTILE(10) OVER (ORDER BY revenue DESC) AS decile FROM rev)
SELECT ROUND(100 * SUM(revenue) FILTER (WHERE decile = 1) / SUM(revenue), 1) AS top10_share_pct,
       ROUND(100 * AVG((revenue > 0)::INT), 1) AS pct_customers_with_revenue
FROM deciles
""").df())

print(con.sql("""
WITH seq AS (
    SELECT user_id, order_ts,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_ts) AS n,
           LAG(order_ts) OVER (PARTITION BY user_id ORDER BY order_ts) AS prev_ts
    FROM orders_clean
    WHERE status = 'completed' AND order_ts < TIMESTAMPTZ '2025-01-01 00:00:00+00'
)
SELECT COUNT(*) AS customers_with_2nd_order,
       MEDIAN(date_diff('day', prev_ts, order_ts)) AS median_days_1st_to_2nd
FROM seq WHERE n = 2
""").df())

COUNTRY_MAP = {"US": "US", "USA": "US", "UNITED STATES": "US", "UK": "UK", "U.K.": "UK",
               "UNITED KINGDOM": "UK", "DE": "DE", "GERMANY": "DE", "IN": "IN", "INDIA": "IN",
               "BR": "BR", "BRAZIL": "BR"}
log = []

def note(table, rule, before, after):
    log.append({"table": table, "rule": rule, "rows_before": before, "rows_after": after})

raw = {name: con.table(name).df() for name in ["users", "events", "orders", "assignments"]}

u = raw["users"]
n0 = len(u); u = u.drop_duplicates(); note("users", "drop exact duplicates", n0, len(u))
u["signup_at"] = pd.to_datetime(u["signup_at"], utc=True)
u["country"] = u["country"].str.strip().str.upper().map(COUNTRY_MAP)
assert u["country"].notna().all(), "unmapped country spelling"
bad_age = ~u["age"].between(13, 100) & u["age"].notna()
note("users", f"invalid age -> missing ({bad_age.sum()} values)", len(u), len(u))
u["age"] = u["age"].where(u["age"].between(13, 100))

ev = raw["events"]
n0 = len(ev); ev = ev.drop_duplicates(subset=["session_id", "user_id", "event_type", "ts_ms"])
note("events", "drop retried duplicates (same content, new event_id)", n0, len(ev))
ev["ts"] = pd.to_datetime(ev["ts_ms"], unit="ms", utc=True)

o = raw["orders"]
n0 = len(o); o = o.drop_duplicates(); note("orders", "drop exact duplicates", n0, len(o))
assert o["order_id"].is_unique
n0 = len(o); o = o[o["user_id"].isin(u["user_id"])]; note("orders", "drop orphan orders", n0, len(o))
n0 = len(o); o = o[o["amount_usd"].ne(99999.0)]; note("orders", "drop QA test orders (99999)", n0, len(o))
o["order_ts"] = pd.to_datetime(o["order_ts"], utc=True, format="ISO8601")
o["category"] = o["category"].str.strip().str.lower()
o["amount_usd"] = o["amount_usd"].abs()

a = raw["assignments"]
n0 = len(a); a = a.drop_duplicates(); note("assignments", "drop exact duplicates", n0, len(a))
a["assigned_at"] = pd.to_datetime(a["assigned_at"], utc=True)
in_both = a.groupby("user_id")["variant"].nunique().loc[lambda s: s > 1].index
n0 = len(a); a = a[~a["user_id"].isin(in_both)]; note("assignments", "drop users seen in both variants", n0, len(a))
n0 = len(a); a = a.sort_values("assigned_at").drop_duplicates("user_id"); note("assignments", "keep first assignment per user", n0, len(a))

print(pd.DataFrame(log).to_string(index=False))
for name, df in [("users", u), ("events", ev), ("orders", o), ("assignments", a)]:
    df.to_parquet(f"clean/{name}.parquet", index=False)

def validate(df, contract):
    problems = []
    for col, rules in contract.items():
        s = df[col]
        if rules.get("unique") and s.duplicated().any():
            problems.append(f"{col}: duplicates")
        if rules.get("nullable") is False and s.isna().any():
            problems.append(f"{col}: nulls")
        if "allowed" in rules and not set(s.dropna().astype(str)) <= rules["allowed"]:
            problems.append(f"{col}: unexpected values {sorted(set(s.dropna().astype(str)) - rules['allowed'])[:3]}")
        if "min" in rules and (s < rules["min"]).any():
            problems.append(f"{col}: values below {rules['min']}")
        if "max" in rules and (s > rules["max"]).any():
            problems.append(f"{col}: values above {rules['max']}")
    return problems

ORDERS = {"order_id": {"unique": True, "nullable": False}, "order_ts": {"nullable": False},
          "category": {"allowed": {"electronics", "home", "fashion", "books", "beauty"}},
          "status": {"allowed": {"completed", "cancelled", "refunded"}},
          "amount_usd": {"min": 0, "max": 10_000}}
ASSIGNMENTS = {"user_id": {"unique": True, "nullable": False},
               "variant": {"allowed": {"control", "treatment"}}}
print("orders raw:        ", validate(raw["orders"], ORDERS))
print("orders clean:      ", validate(o, ORDERS))
print("assignments raw:   ", validate(raw["assignments"], ASSIGNMENTS))
print("assignments clean: ", validate(a, ASSIGNMENTS))

from scipy.stats import chi2_contingency

hist = u[u["signup_at"] < "2025-01-01"]
table = pd.crosstab(hist["device"], hist["age"].isna())
print((hist.groupby("device")["age"].apply(lambda s: s.isna().mean()) * 100).round(1).to_dict(),
      f"chi-square p = {chi2_contingency(table)[1]:.1e}")

done24 = o[o["status"].eq("completed") & (o["order_ts"] < "2025-01-01")]
amt = done24["amount_usd"].dropna()
print(f"n={len(amt):,}  mean={amt.mean():.2f}  median={amt.median():.2f}  "
      f"skew={stats.skew(amt):.2f}  skew(log)={stats.skew(np.log(amt)):.2f}")
print(done24.groupby("category")["amount_usd"].median().round(2).sort_values(ascending=False).to_dict())

sess = (ev.pivot_table(index=["session_id", "user_id"], columns="event_type", values="ts",
                       aggfunc="size", fill_value=0).reset_index().rename_axis(columns=None)
          .merge(u[["user_id", "device", "country"]], on="user_id"))
sess["converted"] = sess["purchase"] > 0
dm = sess[sess["device"].isin(["desktop", "mobile"])]
rates = dm.pivot_table(index="country", columns="device", values="converted", aggfunc="mean")
rates["overall"] = dm.groupby("country")["converted"].mean()
mix = pd.crosstab(dm["country"], dm["device"], normalize="index")
rates["standardized"] = (rates[["desktop", "mobile"]] * mix.mean()).sum(axis=1)
rates["mobile_share"] = mix["mobile"]
print((100 * rates).round(1).sort_values("overall"))

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

n_needed = NormalIndPower().solve_power(proportion_effectsize(0.108 + 0.012, 0.108), alpha=0.05, power=0.8)
print(f"needed per group: {n_needed:,.0f}")

con.register("assign_clean", a)
con.register("orders_py", o)
exp = con.sql("""
SELECT a.user_id, a.variant, a.app_version, a.assigned_at,
       date_diff('day', TIMESTAMPTZ '2025-01-06 00:00:00+00', a.assigned_at) AS day,
       u.device,
       COUNT(o.order_id) > 0 AS converted,
       SUM(o.amount_usd) AS revenue_usd
FROM assign_clean a
JOIN (SELECT user_id, device FROM users_clean) u USING (user_id)
LEFT JOIN orders_py o
       ON o.user_id = a.user_id AND o.status = 'completed'
      AND o.order_ts >= a.assigned_at AND o.order_ts < a.assigned_at + INTERVAL 7 DAY
GROUP BY ALL
""").df()
print(len(exp), exp["variant"].value_counts().to_dict())

counts = exp["variant"].value_counts().sort_index()
print(counts.to_dict(), f"SRM p = {stats.chisquare(counts)[1]:.1e}")
exp["week"] = exp["day"] // 7
share_t = exp.pivot_table(index="app_version", columns="week", values="variant",
                          aggfunc=lambda s: (s == "treatment").mean())
print((100 * share_t).round(1))

valid = exp[~((exp["app_version"] == "5.1.0") & (exp["week"] == 0))].copy()
c2 = valid["variant"].value_counts().sort_index()
print(c2.to_dict(), f"SRM p after exclusion = {stats.chisquare(c2)[1]:.2f}")

from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

def two_prop(df):
    g = df.groupby("variant")["converted"].agg(["sum", "count"])
    xt, nt = g.loc["treatment"]
    xc, nc = g.loc["control"]
    pt, pc = xt / nt, xc / nc
    pool = (xt + xc) / (nt + nc)
    z = (pt - pc) / np.sqrt(pool * (1 - pool) * (1 / nt + 1 / nc))
    se = np.sqrt(pt * (1 - pt) / nt + pc * (1 - pc) / nc)
    return {"control": pc, "treatment": pt, "diff_pts": 100 * (pt - pc),
            "ci_low": 100 * (pt - pc - 1.96 * se), "ci_high": 100 * (pt - pc + 1.96 * se),
            "rel_lift": (pt - pc) / pc, "z": z, "p": 2 * stats.norm.sf(abs(z))}

res = pd.DataFrame({"naive (SRM-affected)": two_prop(exp), "corrected": two_prop(valid)}).T
print(res.round(4))

g = valid.groupby("variant")["converted"].agg(["sum", "count"])
z_sm, p_sm = proportions_ztest(g.loc[["treatment", "control"], "sum"], g.loc[["treatment", "control"], "count"])
lo, hi = confint_proportions_2indep(g.loc["treatment", "sum"], g.loc["treatment", "count"],
                                    g.loc["control", "sum"], g.loc["control", "count"], method="wald")
print(f"statsmodels check: z = {z_sm:.3f}, p = {p_sm:.4f}, CI [{100 * lo:.2f}, {100 * hi:.2f}] points")

buyers = valid[valid["converted"]]
aov_t = np.sort(buyers.loc[buyers["variant"].eq("treatment"), "revenue_usd"].to_numpy())   # sorted: DuckDB row order varies
aov_c = np.sort(buyers.loc[buyers["variant"].eq("control"), "revenue_usd"].to_numpy())
rng = np.random.default_rng(0)
boot = [rng.choice(aov_t, len(aov_t)).mean() - rng.choice(aov_c, len(aov_c)).mean() for _ in range(2000)]
print(f"revenue per converting user: treatment {aov_t.mean():.2f}, control {aov_c.mean():.2f} USD; "
      f"Welch p = {stats.ttest_ind(aov_t, aov_c, equal_var=False)[1]:.2f}; "
      f"bootstrap 95% CI [{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}]")

win = o.merge(valid[["user_id", "variant", "assigned_at"]], on="user_id")
win = win[(win["order_ts"] >= win["assigned_at"]) & (win["order_ts"] < win["assigned_at"] + pd.Timedelta(days=7))]
bad = win.assign(bad=win["status"].ne("completed")).groupby("variant")["bad"].agg(["sum", "count"])
print("cancelled or refunded share of 7-day orders:", (bad["sum"] / bad["count"]).round(4).to_dict(),
      f"p = {proportions_ztest(bad['sum'], bad['count'])[1]:.2f}")

days = np.arange(1, 22)
rows = []
for d in days:
    r = two_prop(valid[valid["day"] < d])
    rows.append({"through_day": d, "diff_pts": r["diff_pts"], "p": r["p"]})
path = pd.DataFrame(rows)
print(path.iloc[[1, 2, 4, 6, 9, 13, 20]].round(4).to_string(index=False))
print("days on which a daily look showed p < 0.05:", path.loc[path["p"] < 0.05, "through_day"].tolist())

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(path["through_day"], path["diff_pts"], marker="o", ms=3, color="#4C72B0")
axes[0].axhline(0, color="gray", lw=1)
axes[0].set(xlabel="days of assignment included", ylabel="lift (percentage points)",
            title="The cumulative lift is noisy early on")
axes[1].plot(path["through_day"], path["p"], marker="o", ms=3, color="#C44E52")
axes[1].axhline(0.05, color="black", ls="--", lw=1)
axes[1].set(yscale="log", xlabel="days of assignment included", ylabel="p-value (log scale)",
            title="Early looks suggested no effect, or worse")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
save_and_show(fig, "cumulative_lift")

seg = []
for dev, part in valid.groupby("device"):
    r = two_prop(part)
    seg.append({"device": dev, "n": len(part), **{k: r[k] for k in ["control", "treatment", "diff_pts", "ci_low", "ci_high"]}})
seg = pd.DataFrame(seg).set_index("device")
print(seg.round(3))

overall = res.loc["corrected"]
fig, ax = plt.subplots(figsize=(7, 2.8))
labels = list(seg.index) + ["all users"]
est = list(seg["diff_pts"]) + [overall["diff_pts"]]
lo_ = list(seg["ci_low"]) + [overall["ci_low"]]
hi_ = list(seg["ci_high"]) + [overall["ci_high"]]
for i, (m, l, h) in enumerate(zip(est, lo_, hi_)):
    ax.errorbar(m, i, xerr=[[m - l], [h - m]], fmt="o", color="#C44E52" if i == len(est) - 1 else "#4C72B0", capsize=3)
ax.axvline(0, color="gray", lw=1)
ax.set_yticks(range(len(labels)), labels)
ax.set_xlabel("lift in 7-day conversion (percentage points), 95% CI")
ax.set_title("The carousel's lift is positive overall; segments are too small to tell apart", loc="left", fontsize=10)
ax.spines[["top", "right", "left"]].set_visible(False)
save_and_show(fig, "lift_by_device")

import hashlib

fp = hashlib.sha256(pd.util.hash_pandas_object(res.round(6), index=True).to_numpy().tobytes()).hexdigest()[:12]
print("result fingerprint:", fp)
