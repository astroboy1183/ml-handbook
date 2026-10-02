# pandas Cheat Sheet

A dense reference for everyday pandas (2.x and 3.x): reading data, selecting, cleaning, grouping, joining, reshaping, and time series, with SQL and polars translations.

Related chapters: [pandas](../chapters/00-python-for-data/05-pandas.md) · [Data cleaning](../chapters/02-data-science-workflow/02-data-cleaning.md) · [Exploratory data analysis](../chapters/02-data-science-workflow/03-exploratory-data-analysis.md)

```python
import numpy as np
import pandas as pd
```

## Reading and writing

| Task | Code |
|---|---|
| CSV | `pd.read_csv(path, parse_dates=["date"], dtype={"category": "category"}, usecols=[...], na_values=["?"])` |
| Write CSV | `df.to_csv(path, index=False)` |
| Parquet (keeps dtypes, columnar, compressed) | `pd.read_parquet(path, columns=[...])`, `df.to_parquet(path)` |
| JSON Lines | `pd.read_json(path, lines=True)`, `df.to_json(path, orient="records", lines=True, date_format="iso")` |
| Nested JSON | `pd.json_normalize(records, sep="_")` |
| SQL | `pd.read_sql("SELECT ...", con)`, `df.to_sql("table", con, index=False)` |
| Excel | `pd.read_excel(path, sheet_name=0)` (needs `openpyxl`) |
| Large CSV in pieces | `for chunk in pd.read_csv(path, chunksize=100_000): ...` |

## Inspecting

| Code | Shows |
|---|---|
| `df.shape`, `df.dtypes`, `df.info()` | size, types, non-null counts, memory |
| `df.head()`, `df.tail()`, `df.sample(5, random_state=0)` | rows |
| `df.describe()`, `df.describe(include="all")` | summary statistics |
| `df["col"].value_counts(normalize=True, dropna=False)` | category shares, including NaN |
| `df.nunique()`, `df.isna().sum()`, `df.isna().mean()` | distinct values, missing counts and shares |
| `df.duplicated(subset=[...]).sum()` | duplicate count |
| `df.select_dtypes("number")` | only numeric columns |
| `df.corr(numeric_only=True)` | correlation matrix |
| `df.memory_usage(deep=True)` | true memory, including strings |

## Selecting

| Code | Meaning |
|---|---|
| `df["a"]` / `df[["a", "b"]]` | one column (Series) / several (DataFrame) |
| `df.loc[rows, cols]` | **by label**; slices **include** the end: `df.loc[0:2, "a":"c"]` |
| `df.iloc[rows, cols]` | **by position**; slices **exclude** the end: `df.iloc[0:2, 0:3]` |
| `df.loc[mask, ["a", "b"]]` | boolean mask rows, chosen columns |
| `df.at[label, "a"]`, `df.iat[0, 0]` | fast single value |
| `df.set_index("id")`, `df.reset_index()` | move a column into/out of the index |

## Filtering

```python
df[(df["amount"] > 20) & (df["category"] == "toys")]       # & | ~ with parentheses, never and/or
df[df["category"].isin(["toys", "games"])]
df[df["amount"].between(20, 50)]                           # inclusive on both ends
df[~df["city"].str.contains("test", case=False, na=False)]
df.query("amount > 20 and category == 'toys'")             # string expression; @var for Python vars
df.nlargest(5, "amount"); df.nsmallest(5, "amount")
```

## Creating columns and method chaining

```python
out = (
    df
    .assign(amount=lambda t: t["amount"].fillna(0),          # lambda sees the current frame
            big=lambda t: t["amount"] > 25)
    .query("amount > 0")
    .sort_values("amount", ascending=False)
    .pipe(lambda t: t.head(10))                               # any function of a DataFrame
)
df["tier"] = np.where(df["amount"] > 20, "high", "low")        # vectorized if/else
df["band"] = np.select([df["amount"] < 10, df["amount"] < 40], ["low", "mid"], default="high")
df = df.rename(columns={"amount": "amt"})
```

## Missing data

| Code | Notes |
|---|---|
| `df.isna()`, `df.notna()` | NaN, None, NaT, and pd.NA all count as missing |
| `df.dropna(subset=["a"])`, `df.dropna(thresh=3)` | drop rows missing `a` / rows with fewer than 3 non-missing |
| `df["a"].fillna(df["a"].median())` | constant or statistic fill |
| `df["a"].fillna(df.groupby("g")["a"].transform("median"))` | group-wise fill |
| `df["a"].ffill()`, `.bfill()`, `.interpolate()` | time-ordered fills |
| `pd.Series([1, None, 3], dtype="Int64")` | nullable integer: stays integer, missing is `<NA>` |
| `s.mean()`, `s.sum()` | skip NaN by default (`skipna=True`); `s.sum(min_count=1)` returns NaN if all missing |

!!! warning "Missing values change your denominators"
    `df["a"].mean()` averages only the non-missing values. Always report how many values were missing, and think about *why* they are missing before filling them.

## Strings (`.str`) and categoricals

```python
df["city"].str.strip().str.title()                       # " pune" -> "Pune"
df["city"].str.lower().str.replace(r"\s+", " ", regex=True)
df["city"].str.contains("pune", case=False, na=False)
df["sku"].str.split("-", expand=True)                    # into columns
df["price_txt"].str.replace(r"[^0-9.]", "", regex=True).astype(float)   # "USD 1,200" -> 1200.0
pd.to_numeric(df["x"], errors="coerce")                  # bad values -> NaN (count them!)
df["category"] = df["category"].astype("category")       # less memory, faster groupby
df["category"].cat.categories; df["size"] = pd.Categorical(df["size"], categories=["S", "M", "L"], ordered=True)
```

## Dates and time series

| Code | Notes |
|---|---|
| `pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")` | parse; bad values become `NaT` |
| `s.dt.year`, `.dt.month`, `.dt.day_name()`, `.dt.hour`, `.dt.dayofweek` | parts (Monday = 0) |
| `s.dt.floor("D")`, `s.dt.to_period("M")` | truncate to day / month period |
| `ts.resample("W").sum()` | needs a DatetimeIndex; `"D"`, `"W"`, `"MS"` (month start), `"h"` |
| `ts.rolling(7, min_periods=1).mean()` | rolling window |
| `s.shift(1)`, `s.diff()`, `s.pct_change(fill_method=None)` | lags and changes |
| `ts.tz_localize("UTC").tz_convert("America/New_York")` | attach a zone, then convert |
| `pd.date_range("2024-01-01", periods=8, freq="D")` | generate dates |
| `(d2 - d1).dt.days` | Timedelta to integer days |

## Group by (split-apply-combine)

```python
g = df.groupby("category").agg(                          # named aggregation: new_col=(col, func)
    n=("order_id", "count"),
    revenue=("amount", "sum"),
    avg=("amount", "mean"),
)
df["share"] = df["amount"] / df.groupby("category")["amount"].transform("sum")   # same length as df
df.groupby("category").filter(lambda t: len(t) >= 3)    # keep whole groups that pass
df.groupby(["category", "city"], observed=True).size()  # observed=True: only seen category combos
df.groupby("customer_id")["amount"].cumsum()            # running total within group
df.groupby("customer_id")["date"].rank(method="first")  # order within group
```

| Method | Output shape | Use for |
|---|---|---|
| `agg` | one row per group | summaries |
| `transform` | same as input | group statistics as new columns, group-wise fill |
| `filter` | subset of input rows | dropping small or irrelevant groups |
| `apply` | flexible (slow) | last resort for complex per-group logic |

## Merge, join, concat

```python
m = orders.merge(customers, on="customer_id", how="left",
                 validate="many_to_one",   # raises if customers has duplicate keys
                 indicator=True)           # adds _merge: both / left_only / right_only
m["_merge"].value_counts()
orders.merge(products, left_on="sku", right_on="product_sku", suffixes=("", "_prod"))
pd.concat([df1, df2], ignore_index=True)    # stack rows
pd.concat([df1, df2], axis=1)               # side by side, aligned on index
```

| `how=` | Keeps |
|---|---|
| `inner` | keys in both |
| `left` / `right` | all keys from the left / right table |
| `outer` | all keys from both |
| `cross` | every combination |

!!! warning "Row explosion"
    If a key is duplicated on both sides, a merge returns every pairing (many-to-many). Check `len()` before and after, and use `validate=` to catch it early.

## Reshaping

```python
wide = df.pivot_table(index="customer_id", columns="category", values="amount",
                      aggfunc="sum", fill_value=0)          # long -> wide, aggregates duplicates
long = wide.reset_index().melt(id_vars="customer_id", var_name="category", value_name="amount")
df.pivot(index="order_id", columns="category", values="amount")   # no aggregation; fails on duplicates
pd.crosstab(df["category"], df["city"], normalize="index", margins=True)
wide.stack(); long.set_index(["customer_id", "category"]).unstack()
df.explode("tags")                                           # list column -> one row per element
```

## Sorting, ranking, binning, sampling

```python
df.sort_values(["category", "amount"], ascending=[True, False])
df["amount"].rank(ascending=False, method="dense")
pd.cut(df["amount"], bins=[0, 20, 40, 100])            # fixed edges, right-inclusive
pd.qcut(df["amount"], q=4)                             # quantile bins
df.drop_duplicates(subset=["customer_id"], keep="first")
df.sample(frac=0.5, random_state=0)
```

## Copy-on-Write (pandas 3)

In pandas 3, **Copy-on-Write** is always on: any DataFrame or Series derived from another behaves as a copy. Modifying a subset never changes the parent, and the old `SettingWithCopyWarning` is gone.

```python
sub = df[df["amount"] > 20]
sub.loc[:, "amount"] = 0                   # changes sub only, never df
df.loc[df["amount"] > 50, "flag"] = True   # the one correct way to update df in place
df["a"][0] = 1                             # chained assignment: never updates df in pandas 3
```

The default text dtype in pandas 3 is `str` (Arrow-backed when `pyarrow` is installed) rather than `object`.

## Performance

- **Vectorize**: column arithmetic, `.str`, `.dt`, `np.where`, and `np.select` beat `apply(axis=1)` by orders of magnitude.
- **Categoricals** for low-cardinality strings cut memory and speed up `groupby`.
- **Parquet** over CSV: faster reads, types preserved, and you can read only some columns.
- Downcast with `pd.to_numeric(s, downcast="integer")` or `astype("int32")`.
- Read only what you need: `usecols=`, `columns=`, `chunksize=`.
- Consider **polars** (multi-threaded, lazy query optimization) or **DuckDB** (SQL on DataFrames and Parquet) when data is large or pandas is slow.

## SQL ↔ pandas ↔ polars

| SQL | pandas | polars |
|---|---|---|
| `SELECT a, b FROM t` | `t[["a", "b"]]` | `t.select("a", "b")` |
| `WHERE x > 5` | `t[t["x"] > 5]` | `t.filter(pl.col("x") > 5)` |
| `SELECT *, x*2 AS y` | `t.assign(y=t["x"] * 2)` | `t.with_columns((pl.col("x") * 2).alias("y"))` |
| `GROUP BY g` + `SUM(x)` | `t.groupby("g")["x"].sum()` | `t.group_by("g").agg(pl.col("x").sum())` |
| `COUNT(*)` per group | `t.groupby("g").size()` | `t.group_by("g").agg(pl.len())` |
| `HAVING SUM(x) > 10` | `.agg(s=("x", "sum")).query("s > 10")` | `.agg(pl.col("x").sum().alias("s")).filter(pl.col("s") > 10)` |
| `ORDER BY x DESC LIMIT 5` | `t.nlargest(5, "x")` | `t.sort("x", descending=True).head(5)` |
| `LEFT JOIN u USING (k)` | `t.merge(u, on="k", how="left")` | `t.join(u, on="k", how="left")` |
| `UNION ALL` | `pd.concat([t, u])` | `pl.concat([t, u])` |
| `SELECT DISTINCT g` | `t["g"].drop_duplicates()` | `t.select("g").unique()` |
| `COALESCE(x, 0)` | `t["x"].fillna(0)` | `pl.col("x").fill_null(0)` |
| `CASE WHEN x>5 THEN 'hi' ELSE 'lo' END` | `np.where(t["x"] > 5, "hi", "lo")` | `pl.when(pl.col("x") > 5).then(pl.lit("hi")).otherwise(pl.lit("lo"))` |
| `SUM(x) OVER (PARTITION BY g)` | `t.groupby("g")["x"].transform("sum")` | `pl.col("x").sum().over("g")` |
| `ROW_NUMBER() OVER (PARTITION BY g ORDER BY d)` | `t.sort_values("d").groupby("g").cumcount() + 1` | `pl.col("d").rank("ordinal").over("g")` |
| `LAG(x) OVER (ORDER BY d)` | `t.sort_values("d")["x"].shift(1)` | `pl.col("x").shift(1)` (after sorting) |

```python
import polars as pl
res = (
    pl.from_pandas(df).lazy()                 # build a query plan; nothing runs yet
    .filter(pl.col("amount") > 10)
    .group_by("category")
    .agg(pl.col("amount").sum().alias("revenue"), pl.len().alias("n"))
    .sort("revenue", descending=True)
    .collect()                                # optimize and execute
)
```

## Common gotchas

| Gotcha | Fix |
|---|---|
| `and`/`or` in masks raise "truth value is ambiguous" | use `&`, `|`, `~` with parentheses |
| `loc` slice includes the end, `iloc` excludes it | know which one you're using |
| Chained assignment `df["a"][i] = v` does nothing | `df.loc[i, "a"] = v` |
| A merge silently duplicated rows | `validate="one_to_one"` / `"many_to_one"`; compare row counts |
| Numbers read as strings (`"1,200"`) | clean with `.str.replace`, then `pd.to_numeric(errors="coerce")` |
| `std()` differs from NumPy | pandas uses `ddof=1`, NumPy `ddof=0` |
| Dates parsed as strings | `parse_dates=` or `pd.to_datetime`; give `format=` for speed and safety |
| `groupby` drops NaN keys | `groupby(..., dropna=False)` |
| Index misalignment after filtering produces NaN in assignment | align on index deliberately or `reset_index(drop=True)` |
| `apply(axis=1)` is very slow | vectorize |
