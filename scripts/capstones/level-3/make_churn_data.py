"""Generate the Level 3 capstone data: a synthetic churn data set for a subscription telecom company.

Usage:
    python make_churn_data.py            # writes churn.csv in the current directory
    from make_churn_data import make_churn_data
    df = make_churn_data()               # a pandas DataFrame, one row per customer

The data are synthetic, seeded, and realistically imperfect: mixed column types, missing
values (some missing at random, some not), and about 20% churners.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def make_churn_data(n=10_000, seed=42):
    """Return a DataFrame of n customers with a binary `churned` label (1 = cancelled next month)."""
    rng = np.random.default_rng(seed)

    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    extra = np.select([contract == "one-year", contract == "two-year"], [12, 24], 0)
    tenure = np.clip(np.round(rng.gamma(1.5, 14, n) + extra), 1, 72).astype(int)
    internet = rng.choice(["fiber", "dsl", "none"], n, p=[0.45, 0.35, 0.20])
    n_products = 1 + rng.binomial(4, 0.3, n)
    monthly = (20 + 50 * (internet == "fiber") + 30 * (internet == "dsl") + 5 * n_products
               + rng.normal(0, 8, n)).clip(15, None).round(2)
    payment = rng.choice(["credit card", "bank transfer", "electronic check", "mailed check"], n,
                         p=[0.30, 0.25, 0.30, 0.15])
    autopay = np.where(np.isin(payment, ["credit card", "bank transfer"]),
                       rng.random(n) < 0.7, rng.random(n) < 0.1)
    tickets = rng.poisson(0.4 + 0.6 * (internet == "fiber"), n)
    satisfaction = np.clip(np.round(3.6 - 0.4 * tickets + rng.normal(0, 1, n)), 1, 5)
    last_login = rng.exponential(10, n).round(1)
    age = np.clip(np.round(rng.normal(45, 14, n)), 18, 90)
    region = rng.choice(["north", "south", "east", "west"], n)

    logit = (-1.75 - 0.045 * tenure + 0.02 * (monthly - 65)
             + 1.1 * (contract == "month-to-month") - 0.9 * (contract == "two-year")
             + 0.45 * (payment == "electronic check") + 0.35 * (internet == "fiber")
             + 0.30 * tickets - 0.4 * autopay + 0.035 * last_login
             - 0.45 * (satisfaction - 3) - 0.10 * (n_products - 2) - 0.005 * (age - 45))
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    # Missing values. Survey answers are missing more often for unhappy customers (not at random);
    # login tracking and age have gaps unrelated to anything (completely at random).
    sat_missing = rng.random(n) < np.where(satisfaction <= 2, 0.35, 0.10)
    login_missing = rng.random(n) < 0.06
    age_missing = rng.random(n) < 0.04

    df = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "contract": contract,
        "payment_method": payment,
        "internet_service": internet,
        "autopay": autopay,
        "num_products": n_products,
        "support_tickets_90d": tickets,
        "satisfaction_score": pd.array(np.where(sat_missing, np.nan, satisfaction), dtype="Int64"),
        "days_since_last_login": np.where(login_missing, np.nan, last_login),
        "age": pd.array(np.where(age_missing, np.nan, age), dtype="Int64"),
        "region": region,
        "churned": churned,
    })
    return df


if __name__ == "__main__":
    out = Path("churn.csv")
    make_churn_data().to_csv(out, index=False)
    print(f"wrote {out.resolve()}")
