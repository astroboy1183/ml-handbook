# Level 0 Capstone: Analyze a Dataset End to End

> **Level 0 · Capstone** · ⏱️ 6–10 hours · Uses: every Level 0 chapter

You've learned the tools one at a time. This capstone asks you to use them together, the way you will on real projects: set up a reproducible project, load messy data, find and fix its problems, answer questions with NumPy and pandas, make clear charts, and write up what you found.

## The scenario

You've just joined the analytics team of **CityCycle**, a bike-sharing company. The operations manager, Wei, sends you a year of hourly rental data and a short message:

> "We're planning next year's maintenance schedule and staffing. Can you tell me when demand is highest and lowest, how much weather matters, and whether anything in the data looks off? I need something I can show the regional director on Friday."

The data comes from an export that has had problems before. Expect some.

## The data

Generate the dataset with this code. It's seeded, so everyone gets the same data. Save it as `data/raw/rentals.csv` and treat that file as read-only.

```python
import numpy as np
import pandas as pd

def make_rentals(seed=7):
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h")
    n = len(hours)
    doy = hours.dayofyear.to_numpy()
    hour = hours.hour.to_numpy()
    weekend = hours.dayofweek.to_numpy() >= 5

    temp = 12 - 10 * np.cos(2 * np.pi * (doy - 15) / 366) + 4 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 2.5, n)
    rain = rng.random(n) < np.where((doy > 270) | (doy < 90), 0.18, 0.08)
    humidity = np.clip(55 + 25 * rain + rng.normal(0, 10, n), 15, 100)

    commute = np.exp(-((hour - 8) ** 2) / 2) * 260 + np.exp(-((hour - 17.5) ** 2) / 3) * 300
    leisure = np.exp(-((hour - 14) ** 2) / 12) * 220
    base = np.where(weekend, 0.25 * commute + 1.4 * leisure, commute + 0.5 * leisure) + 15
    weather = np.clip(1 + 0.035 * (temp - 12), 0.2, 1.6) * np.where(rain, 0.45, 1.0)
    rentals = rng.poisson(base * weather * 1.08 ** ((doy - 1) / 30.5))

    df = pd.DataFrame({
        "timestamp": hours.strftime("%Y-%m-%d %H:%M"),
        "temp_c": temp.round(1),
        "humidity": humidity.round(0),
        "rain": np.where(rain, "yes", "no"),
        "rentals": rentals,
    })
    # Realistic export problems
    df.loc[rng.choice(n, 120, replace=False), "temp_c"] = np.nan              # sensor gaps
    df.loc[rng.choice(n, 40, replace=False), "rain"] = rng.choice(["Yes", "YES", "y"], 40)
    glitch = (hours >= "2024-08-12") & (hours < "2024-08-15")
    df.loc[glitch, "rentals"] = -1                                            # counter outage
    df = pd.concat([df, df.sample(60, random_state=seed)]).sort_index(kind="stable")  # duplicated rows
    return df.reset_index(drop=True)

rentals = make_rentals()
```

Columns: `timestamp` (local time, as text), `temp_c` (°C), `humidity` (%), `rain` (whether it rained that hour), and `rentals` (bikes rented that hour).

## Tasks

### Part 1: Set up the project

1. Create a project folder with the layout from [Setting up your environment](../chapters/00-python-for-data/02-environment-setup.md): `data/raw`, `data/processed`, `notebooks`, `src`, and `reports`.
2. Create an environment, and pin your dependencies in a lock file.
3. Put the generator in `src/make_data.py`, and write the raw CSV from it.

### Part 2: Load, inspect, and clean

4. Load the CSV with correct dtypes (the timestamp must be a datetime).
5. Find **every** data-quality problem. There are at least four kinds. For each, report how many rows are affected and decide what to do about it.
6. Write the cleaned data to `data/processed/rentals.parquet`. Put the cleaning in a function in `src/`, so it's reproducible, not scattered across notebook cells.

### Part 3: Answer Wei's questions

7. **Daily rhythm:** What does the average hourly demand look like on weekdays versus weekends? When are the peaks?
8. **Seasonality:** How do total daily rentals change across the year? Which month is busiest, and which is quietest?
9. **Weather:** How much do rain and temperature affect demand? Compare like with like: rainy versus dry hours *at the same time of day*.
10. **Staffing:** List the 5 busiest hours-of-the-week (for example, "Tuesday 17:00") by average rentals.
11. **Growth:** Is there a trend over the year beyond seasonality? (Hint: compare the same calendar weeks, or look at a 28-day rolling mean.)

### Part 4: Communicate

12. Make **four** explanatory charts (not exploratory ones) that answer questions 7–9 and 11, each with a takeaway title.
13. Write a half-page summary for Wei: three key findings, one recommendation, and the data-quality issues with what you did about each.

## Deliverables

- The project folder, with `src/make_data.py`, `src/clean.py`, a lock file, and a README.
- `notebooks/01-analysis.ipynb`, which runs top to bottom after "Restart kernel and run all".
- `reports/` with the four charts as PNGs and `summary.md`.

## Acceptance checklist

- [ ] Deleting `.venv` and `data/processed/` and rebuilding from the README reproduces every number.
- [ ] All data-quality problems are found, counted, and handled, with a reason for each decision.
- [ ] No Python loops over rows: everything is vectorized pandas or NumPy.
- [ ] Every chart has a takeaway title, labeled axes with units, and a colorblind-safe palette, and every bar chart starts at zero.
- [ ] The weather comparison controls for time of day.
- [ ] The summary is readable by someone who has never used Python.

## Hints

??? tip "Hint 1: finding the problems"
    Start with `df.info()`, `df.describe()`, `df.duplicated().sum()`, and `df["rain"].value_counts()`. Look at the minimum of `rentals`. Can a count be negative? Check whether every hour of the year appears exactly once.

??? tip "Hint 2: what to do with the outage"
    Rentals of −1 don't mean "minus one bike". They mean "the counter wasn't working." Replace them with NaN, not 0: zero would claim nobody rode a bike for three days, which would distort the August totals and averages.

??? tip "Hint 3: comparing rain fairly"
    Rain is more common in some seasons, and demand depends heavily on the hour. A raw rainy-versus-dry average mixes those effects. Group by hour of day *and* rain, then compare within each hour, for example as a ratio.

??? tip "Hint 4: separating trend from season"
    A 28-day rolling mean of daily totals smooths out weekly cycles. To separate growth from seasonality, compare months that have similar weather, such as April and October, or divide each day's rentals by what the temperature alone would predict.

## Solution

Try the whole capstone before looking. When you're done, compare your work with the [worked solution](solutions/level-0-capstone.md).
