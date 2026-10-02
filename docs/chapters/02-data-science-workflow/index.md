# Level 2: The Data Science Workflow

> **Level 2 · Overview** · ⏱️ ~3–5 weeks at about an hour a day · Prerequisites: [Level 0](../00-python-for-data/index.md) and [Level 1](../01-math-foundations/index.md) (especially [Statistics](../01-math-foundations/05-statistics.md))

Level 2 is where you put the tools and the math to work on real questions. You learn the full data science workflow: get the data with SQL, clean it, explore it, engineer features, run and analyze experiments, and communicate what you found so that people act on it.

## What this level covers

Real data is messy, scattered across tables, and full of traps. This level teaches the craft of turning it into trustworthy answers.

You start with **SQL**, the language most of the world's data lives behind, and the other ways data arrives: files, APIs, and (carefully) the web. Then you **clean** data: fix types, handle missing values with an understanding of *why* they're missing, deal with outliers, duplicates, inconsistent strings, and time zones, and write checks so bad data can't sneak back in. **Exploratory data analysis** teaches you to interrogate a dataset systematically, read relationships correctly, and avoid the classic traps of correlation and Simpson's paradox. **Feature engineering** turns raw columns into inputs a model can use, and it is where you first meet data leakage, the most common way ML projects fool themselves.

The last two chapters are about decisions. **Experimentation and A/B testing** shows you how to measure the effect of a change properly: design, power analysis, analysis, the pitfalls that invalidate tests, bandits, and an introduction to causal inference when you can't randomize. **Communicating results** teaches you to turn an analysis into a recommendation that the right audience understands and trusts, and to make your work reproducible.

## What you'll be able to do

- Write SQL with joins, CTEs, and window functions, and run it from Python against SQLite and DuckDB.
- Clean a messy dataset systematically, choose an imputation strategy based on the missing-data mechanism, and validate the result.
- Run a structured EDA, measure relationships with the right correlation, and spot confounding and Simpson's paradox.
- Encode, scale, transform, and combine features without leaking information from the test set.
- Design an A/B test with a power analysis, analyze it correctly, and avoid peeking and other pitfalls.
- Estimate effects without randomization using difference-in-differences, and reason about confounders with DAGs.
- Write a clear, answer-first report with effective charts, backed by a reproducible analysis under version control.

## Chapters

| # | Chapter | What it covers | Time |
|---|---|---|---|
| 1 | [SQL and data acquisition](01-sql-and-data-acquisition.md) | Relational thinking, SELECT to HAVING, joins, CTEs, window functions, sqlite3 and DuckDB, file formats, APIs, scraping ethics | ~60 min read + 4–5 h practice |
| 2 | [Data cleaning](02-data-cleaning.md) | Tidy data, parsing types, MCAR/MAR/MNAR and imputation, outliers, duplicates, messy strings, time zones, validation | ~60 min read + 4–5 h practice |
| 3 | [Exploratory data analysis](03-exploratory-data-analysis.md) | An EDA checklist, distributions, bivariate relationships, Pearson and Spearman, causation, Simpson's paradox, segments, hypotheses | ~55 min read + 3–4 h practice |
| 4 | [Feature engineering](04-feature-engineering.md) | Categorical encodings, scaling, log and Box-Cox, binning, date and text features, interactions, feature selection, leakage | ~60 min read + 4–5 h practice |
| 5 | [Experimentation and A/B testing](05-experimentation-and-ab-testing.md) | Randomization, metrics and guardrails, power analysis, analysis, peeking, novelty effects, bandits, DAGs, difference-in-differences | ~65 min read + 4–5 h practice |
| 6 | [Communicating results](06-communicating-results.md) | Audiences, the pyramid principle, effective charts, reports, dashboards, reproducibility, notebooks vs scripts, version control | ~50 min read + 2–3 h practice |

## How to study this level

- **Work with messy data, on purpose.** Every chapter generates realistic, imperfect data. Don't clean it up before you start; the mess is the lesson.
- **Write down your question first.** Before any query or chart, write the question it answers. Analysis without a question wanders.
- **Check your numbers two ways.** Compute a key figure in SQL and again in pandas. Compute a test statistic by hand and again with `scipy`. Disagreement means a bug, and finding it is valuable practice.
- **Be suspicious of good news.** A huge effect, a near-perfect correlation, or a feature that predicts the target almost exactly is usually a bug, a confounder, or leakage. Investigate before you celebrate.
- **Practice the write-up, not just the code.** For each chapter, write a three-sentence summary of a result as if for a manager. Communication is a skill you train, like SQL.
- **Keep the references open:** the [SQL](../../cheatsheets/sql.md), [pandas](../../cheatsheets/pandas.md), and [probability and statistics](../../cheatsheets/probability-statistics.md) cheat sheets.

!!! tip "This is the job"
    Many working data scientists spend most of their time on exactly what this level covers: getting data, cleaning it, exploring it, testing changes, and explaining results. Models get the headlines, but this workflow decides whether the models are trustworthy.

## Capstone

The level, and the Beginner tier, ends with **[a full data science investigation](../../exercises/level-2-capstone.md)**: you query an e-commerce database with SQL, clean it, explore it, analyze an A/B test end to end, and write a one-page recommendation. Finish it without notes, and you're ready for the Intermediate tier, starting with [Level 3: ML fundamentals](../03-ml-fundamentals/index.md).
