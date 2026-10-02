"""Generate the Level 4 capstone dataset: synthetic personal-loan applications and defaults.

Usage:
    python make_loan_data.py                 # writes loans.csv in the current directory
    from make_loan_data import make_loans
    df = make_loans()                        # pandas DataFrame

The data is synthetic and seeded. Default risk has nonlinear effects (a credit-score
cliff, a U-shaped utilization effect, saturating late payments), interactions (debt
ratio matters more on 60-month loans, small-business loans are riskier for low
incomes), two categorical features, missing values, and a few pure-noise columns.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def make_loans(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    income = rng.lognormal(np.log(58), 0.45, n).round(1)                       # thousands of USD
    credit = np.clip(rng.normal(680, 60, n), 450, 850).round()
    dti = np.clip(rng.beta(2.2, 5, n) + 0.08 * (income < 35), 0, 0.95).round(3)
    utilization = rng.beta(2, 3, n).round(3)
    late = rng.poisson(0.25 + 1.2 * (credit < 620), n)
    emp_years = rng.exponential(6, n).round(1)
    loan_amount = np.clip(rng.lognormal(np.log(12), 0.6, n), 1, 60).round(1)   # thousands of USD
    term = rng.choice([36, 60], n, p=[0.65, 0.35])
    lines = rng.poisson(8, n)
    home = rng.choice(["rent", "mortgage", "own"], n, p=[0.45, 0.4, 0.15])
    purpose = rng.choice(["debt_consolidation", "home_improvement", "car", "small_business", "other"],
                         n, p=[0.45, 0.15, 0.15, 0.1, 0.15])
    noise = rng.normal(size=(n, 3))

    logit = (
        -4.0
        + 1.6 / (1 + np.exp((credit - 615) / 12))                 # cliff below about 615
        - 0.006 * (credit - 680)
        + 16.0 * (utilization - 0.45) ** 2                        # U-shape around 45%
        + 1.1 * np.log1p(late)                                    # saturating late payments
        + 2.5 * dti * (1 + 1.2 * (term == 60))                    # interaction: DTI x long term
        - 0.6 * np.log(income / 58)
        + 0.9 * ((purpose == "small_business") & (income < 50))  # interaction: purpose x income
        + 0.25 * (home == "rent")
        - 0.03 * np.minimum(emp_years, 10)
        + 0.15 * np.log(loan_amount / income)
    )
    default = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "income_kusd": income, "credit_score": credit, "debt_to_income": dti,
        "utilization": utilization, "late_payments_2y": late, "employment_years": emp_years,
        "loan_amount_kusd": loan_amount, "term_months": term, "open_credit_lines": lines,
        "home_ownership": home, "purpose": purpose,
        "noise_1": noise[:, 0].round(3), "noise_2": noise[:, 1].round(3), "noise_3": noise[:, 2].round(3),
        "default": default,
    })
    # Missing values: employment years more often missing for renters; some credit scores missing
    miss_emp = rng.random(n) < np.where(home == "rent", 0.12, 0.05)
    df.loc[miss_emp, "employment_years"] = np.nan
    df.loc[rng.random(n) < 0.03, "credit_score"] = np.nan
    return df


if __name__ == "__main__":
    out = Path("loans.csv")
    make_loans().to_csv(out, index=False)
    print(f"wrote {out.resolve()}")
