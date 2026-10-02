# Probability and Statistics Cheat Sheet

A dense reference for the distributions, rules, estimators, intervals, and tests you'll use in data science and ML, with the SciPy and statsmodels calls for each.

Related chapters: [Probability](../chapters/01-math-foundations/04-probability.md) · [Statistics](../chapters/01-math-foundations/05-statistics.md) · [Information theory and optimization](../chapters/01-math-foundations/06-information-theory-and-optimization.md) · [Experimentation and A/B testing](../chapters/02-data-science-workflow/05-experimentation-and-ab-testing.md)

```python
import numpy as np
from scipy import stats
rng = np.random.default_rng(0)
```

## Rules of probability

| Rule | Formula |
|---|---|
| Complement | $P(A^c) = 1 - P(A)$ |
| Union | $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ |
| Conditional | $P(A \mid B) = P(A \cap B) / P(B)$ |
| Multiplication | $P(A \cap B) = P(A \mid B)\,P(B)$ |
| Independence | $P(A \cap B) = P(A)\,P(B)$, equivalently $P(A \mid B) = P(A)$ |
| Total probability | $P(B) = \sum_j P(B \mid A_j)\,P(A_j)$ over a partition $A_1, \dots, A_k$ |
| Bayes' theorem | $P(A \mid B) = P(B \mid A)\,P(A) / P(B)$ |
| "At least one" in $k$ independent tries | $1 - (1 - p)^k$ |

**Bayes with natural frequencies.** Imagine 1,000,000 cases. Multiply by the base rate to split them, apply the sensitivity and false positive rate to each group, and divide true positives by all positives. Precision depends on the base rate; sensitivity doesn't.

## Discrete distributions

| Distribution | Story | PMF $P(X = k)$ | Mean | Variance | SciPy |
|---|---|---|---|---|---|
| Bernoulli$(p)$ | one yes/no trial | $p^k(1-p)^{1-k}$, $k \in \{0, 1\}$ | $p$ | $p(1-p)$ | `stats.bernoulli(p)` |
| Binomial$(n, p)$ | successes in $n$ independent trials | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | `stats.binom(n, p)` |
| Geometric$(p)$ | trials until the first success | $(1-p)^{k-1}p$, $k \geq 1$ | $1/p$ | $(1-p)/p^2$ | `stats.geom(p)` |
| Negative binomial$(r, p)$ | failures before the $r$-th success; overdispersed counts | $\binom{k+r-1}{k}p^r(1-p)^k$ | $r(1-p)/p$ | $r(1-p)/p^2$ | `stats.nbinom(r, p)` |
| Poisson$(\lambda)$ | events in a fixed window at rate $\lambda$ | $\lambda^k e^{-\lambda}/k!$ | $\lambda$ | $\lambda$ | `stats.poisson(lam)` |
| Categorical$(\mathbf{p})$ | one draw from $K$ classes | $p_k$ | (one-hot) $\mathbf{p}$ | $\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ | `rng.choice(K, p=p)` |
| Multinomial$(n, \mathbf{p})$ | counts per class in $n$ draws | $\frac{n!}{\prod_k x_k!}\prod_k p_k^{x_k}$ | $n\mathbf{p}$ | $n(\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top)$ | `stats.multinomial(n, p)` |
| Discrete uniform on $\{a, \dots, b\}$ | a fair die | $1/(b - a + 1)$ | $(a+b)/2$ | $((b-a+1)^2 - 1)/12$ | `stats.randint(a, b + 1)` |

## Continuous distributions

| Distribution | Story / use | PDF | Mean | Variance | SciPy |
|---|---|---|---|---|---|
| Uniform$(a, b)$ | equally likely on an interval | $\frac{1}{b-a}$ on $[a, b]$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | `stats.uniform(loc=a, scale=b-a)` |
| Normal$(\mu, \sigma^2)$ | sums of many small effects; noise | $\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ | `stats.norm(loc=mu, scale=sigma)` |
| Exponential$(\lambda)$ | waiting time at rate $\lambda$; memoryless | $\lambda e^{-\lambda x}$, $x \geq 0$ | $1/\lambda$ | $1/\lambda^2$ | `stats.expon(scale=1/lam)` |
| Gamma$(k, \theta)$ | sum of $k$ exponential waits; positive skewed values | $\frac{x^{k-1}e^{-x/\theta}}{\Gamma(k)\theta^k}$ | $k\theta$ | $k\theta^2$ | `stats.gamma(a=k, scale=theta)` |
| Beta$(a, b)$ | a probability or rate; conjugate prior for Bernoulli | $\propto x^{a-1}(1-x)^{b-1}$ on $[0, 1]$ | $\frac{a}{a+b}$ | $\frac{ab}{(a+b)^2(a+b+1)}$ | `stats.beta(a, b)` |
| Lognormal | $e^{Z}$ with $Z \sim \mathcal{N}(\mu, \sigma^2)$: revenue, incomes, durations | $\frac{1}{x\sigma\sqrt{2\pi}}e^{-(\ln x-\mu)^2/2\sigma^2}$ | $e^{\mu + \sigma^2/2}$ | $(e^{\sigma^2} - 1)e^{2\mu + \sigma^2}$ | `stats.lognorm(s=sigma, scale=np.exp(mu))` |
| Student's $t_\nu$ | standardized mean with estimated sd; heavy tails | (see SciPy) | $0$ ($\nu > 1$) | $\frac{\nu}{\nu - 2}$ ($\nu > 2$) | `stats.t(df)` |
| Chi-square $\chi^2_k$ | sum of $k$ squared standard normals | (see SciPy) | $k$ | $2k$ | `stats.chi2(df)` |
| $F_{d_1, d_2}$ | ratio of scaled chi-squares; ANOVA | (see SciPy) | $\frac{d_2}{d_2 - 2}$ | (see SciPy) | `stats.f(d1, d2)` |
| Multivariate normal $\mathcal{N}(\boldsymbol{\mu}, \mathbf{\Sigma})$ | correlated normal vectors | $\propto e^{-\frac12(\mathbf{x}-\boldsymbol{\mu})^\top\mathbf{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})}$ | $\boldsymbol{\mu}$ | $\mathbf{\Sigma}$ | `stats.multivariate_normal(mu, Sigma)` |

**Normal facts.** $P(\lvert Z \rvert < 1) \approx 0.683$, $P(\lvert Z \rvert < 1.96) = 0.95$, $P(\lvert Z \rvert < 3) \approx 0.997$. One-sided 95%: $z = 1.645$. Standardize with $Z = (X - \mu)/\sigma$.

**Relationships.** Binomial $\approx$ Poisson$(np)$ for large $n$, small $p$. Binomial $\approx \mathcal{N}(np, np(1-p))$ when $np$ and $n(1-p)$ are both above about 10. Poisson counts ↔ exponential gaps. $t_\nu \to \mathcal{N}(0, 1)$ as $\nu \to \infty$.

## The scipy.stats interface

| Method | Returns |
|---|---|
| `d.pmf(k)`, `d.pdf(x)`, `d.logpdf(x)` | mass, density, log-density |
| `d.cdf(x)`, `d.sf(x)` | $P(X \leq x)$, $P(X > x)$ (use `sf` for accurate tails) |
| `d.ppf(q)`, `d.isf(q)` | quantile: the $x$ with $P(X \leq x) = q$; inverse survival |
| `d.mean()`, `d.var()`, `d.std()`, `d.interval(0.95)` | moments; central interval |
| `d.rvs(size, random_state=rng)` | random samples |
| `stats.norm.fit(data)` | maximum likelihood fit of the parameters |

Watch the parameterizations: `scale` means sd for `norm`, $1/\lambda$ for `expon`, and $\theta$ for `gamma`; `uniform` covers `[loc, loc + scale]`.

## Expectation, variance, and covariance

$$
\mathbb{E}[aX + bY + c] = a\mathbb{E}[X] + b\mathbb{E}[Y] + c \quad \text{(always, even if dependent)}
$$

$$
\operatorname{Var}(X) = \mathbb{E}[X^2] - \mu^2, \qquad \operatorname{Var}(aX + b) = a^2\operatorname{Var}(X)
$$

$$
\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X, Y), \qquad \operatorname{Cov}(X, Y) = \mathbb{E}[XY] - \mu_X\mu_Y
$$

$$
\rho_{XY} = \frac{\operatorname{Cov}(X, Y)}{\sigma_X\sigma_Y} \in [-1, 1], \qquad \operatorname{Cov}(\mathbf{A}\mathbf{X}) = \mathbf{A}\,\mathbf{\Sigma}\,\mathbf{A}^\top
$$

Independent ⇒ uncorrelated, but not the reverse ($Y = X^2$ with symmetric $X$). Marginal: $p(x) = \sum_y p(x, y)$. Conditional: $p(y \mid x) = p(x, y)/p(x)$.

## Limit theorems

- **Law of large numbers:** $\bar{X}_n \to \mu$ as $n \to \infty$ (needs a finite mean).
- **Standard error of the mean:** $\sigma/\sqrt{n}$. Four times the data halves the error.
- **Central limit theorem:** $\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \approx \mathcal{N}(0, 1)$ for large $n$ (needs i.i.d. data with finite variance). Skewed data need larger $n$; heavy tails (Cauchy) break it.

## Descriptive statistics

| Statistic | Formula or code | Notes |
|---|---|---|
| Mean | $\bar{x} = \frac{1}{n}\sum_i x_i$ | sensitive to outliers |
| Median | `np.median(x)` | robust |
| Sample variance | $s^2 = \frac{1}{n-1}\sum_i(x_i - \bar{x})^2$ | `np.var(x, ddof=1)`; pandas `.var()` uses `ddof=1` by default, NumPy uses `ddof=0` |
| Quantiles, IQR | `np.percentile(x, [25, 75])` | IQR $= Q_3 - Q_1$, robust spread |
| Skewness | `stats.skew(x)` | $> 0$: long right tail (mean > median) |
| Standard score | $z_i = (x_i - \bar{x})/s$ | |
| Correlation | `np.corrcoef(x, y)`, `stats.spearmanr(x, y)` | Pearson: linear; Spearman: monotonic, rank-based |

## Estimation

| Concept | Formula |
|---|---|
| Bias | $\mathbb{E}[\hat{\theta}] - \theta$ |
| Mean squared error | $\mathbb{E}[(\hat{\theta} - \theta)^2] = \operatorname{Bias}^2 + \operatorname{Var}(\hat{\theta})$ |
| Likelihood, log-likelihood | $L(\theta) = \prod_i p(x_i \mid \theta)$, $\ell(\theta) = \sum_i \log p(x_i \mid \theta)$ |
| MLE | $\hat{\theta} = \arg\max_\theta \ell(\theta)$ |

**Common MLEs.** Bernoulli: $\hat{p} = k/n$. Poisson: $\hat{\lambda} = \bar{x}$. Exponential: $\hat{\lambda} = 1/\bar{x}$. Normal: $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{n}\sum_i(x_i - \bar{x})^2$ (biased). Linear regression with normal errors: least squares.

**Losses as negative log-likelihoods.** Normal noise → MSE. Laplace noise → MAE. Bernoulli → binary cross-entropy. Categorical → cross-entropy. L2 penalty → normal prior; L1 penalty → Laplace prior.

## Confidence intervals

General shape: **estimate ± critical value × standard error**. "95%" means 95% of intervals from repeated studies contain the true value.

| Quantity | Standard error | 95% interval | Code |
|---|---|---|---|
| Mean ($\sigma$ unknown) | $s/\sqrt{n}$ | $\bar{x} \pm t_{0.975, n-1}\,s/\sqrt{n}$ | `stats.ttest_1samp(x, 0).confidence_interval()` |
| Proportion | $\sqrt{\hat{p}(1-\hat{p})/n}$ | Wald: $\hat{p} \pm 1.96\,\text{SE}$; prefer Wilson | `statsmodels.stats.proportion.proportion_confint(k, n, method="wilson")` |
| Difference of means | $\sqrt{s_A^2/n_A + s_B^2/n_B}$ | $\bar{x}_A - \bar{x}_B \pm t^*\,\text{SE}$ (Welch df) | `stats.ttest_ind(a, b, equal_var=False).confidence_interval()` |
| Difference of proportions | $\sqrt{\frac{\hat{p}_A(1-\hat{p}_A)}{n_A} + \frac{\hat{p}_B(1-\hat{p}_B)}{n_B}}$ | $\hat{p}_A - \hat{p}_B \pm 1.96\,\text{SE}$ | `statsmodels.stats.proportion.confint_proportions_2indep` |
| Regression coefficient | $\sqrt{\hat{\sigma}^2[(\mathbf{X}^\top\mathbf{X})^{-1}]_{jj}}$, $\hat{\sigma}^2 = \text{RSS}/(n-p)$ | $\hat{\beta}_j \pm t_{0.975, n-p}\,\text{SE}_j$ | `sm.OLS(y, X).fit().conf_int()` |
| Anything else | bootstrap | percentile or BCa | `stats.bootstrap((x,), stat, method="BCa")` |

**Bootstrap recipe.** Resample the independent units (rows, users, time blocks) with replacement, $B$ times (1,000–10,000); recompute the statistic each time; the standard deviation of the results is the SE, and their 2.5th and 97.5th percentiles are a 95% interval.

## Hypothesis testing

1. State $H_0$ and $H_1$; choose $\alpha$ (often 0.05) **before** looking at the data.
2. Compute a test statistic and its p-value: $P(\text{a statistic at least this extreme} \mid H_0)$.
3. Reject $H_0$ if $p < \alpha$. Report the effect size and confidence interval either way.

| | $H_0$ true | $H_0$ false |
|---|---|---|
| Reject | Type I error (rate $\alpha$) | correct (rate = power $= 1 - \beta$) |
| Don't reject | correct | Type II error (rate $\beta$) |

**A p-value is not** the probability that $H_0$ is true, not the probability the result is a fluke, and not a measure of effect size. Under $H_0$, p-values are uniform on $[0, 1]$.

## Which test?

| Question | Data | Test | Statistic | Code |
|---|---|---|---|---|
| Mean equals $\mu_0$? | one numeric sample | one-sample t | $t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}$, df $n - 1$ | `stats.ttest_1samp(x, mu0)` |
| Two group means equal? | two independent samples | Welch's t | $t = \frac{\bar{x}_A - \bar{x}_B}{\sqrt{s_A^2/n_A + s_B^2/n_B}}$ | `stats.ttest_ind(a, b, equal_var=False)` |
| Paired before/after? | matched pairs | paired t | one-sample t on differences | `stats.ttest_rel(after, before)` |
| Two groups, non-normal or ordinal? | two independent samples | Mann–Whitney U | rank sums | `stats.mannwhitneyu(a, b)` |
| Paired, non-normal? | matched pairs | Wilcoxon signed-rank | signed ranks | `stats.wilcoxon(after, before)` |
| 3+ group means equal? | $k$ samples | one-way ANOVA | $F = \frac{\text{between MS}}{\text{within MS}}$, df $(k-1, N-k)$ | `stats.f_oneway(*groups)` |
| 3+ groups, non-normal? | $k$ samples | Kruskal–Wallis | ranks | `stats.kruskal(*groups)` |
| Two proportions equal? | successes and trials | two-proportion z | $z = \frac{\hat{p}_A - \hat{p}_B}{\sqrt{\hat{p}(1-\hat{p})(1/n_A + 1/n_B)}}$, pooled $\hat{p}$ | `statsmodels.stats.proportion.proportions_ztest(k, n)` |
| Two categorical variables independent? | contingency table | chi-square | $\chi^2 = \sum\frac{(O - E)^2}{E}$, df $(r-1)(c-1)$ | `stats.chi2_contingency(table)` |
| Small counts in a 2×2 table? | contingency table | Fisher's exact | exact | `stats.fisher_exact(table)` |
| Data fit a distribution? | counts per category | chi-square goodness of fit | $\sum\frac{(O - E)^2}{E}$ | `stats.chisquare(obs, f_exp)` |
| Linear association? | two numeric variables | Pearson correlation | $t = r\sqrt{\frac{n-2}{1-r^2}}$ | `stats.pearsonr(x, y)` |
| Monotonic association? | two numeric or ordinal variables | Spearman correlation | ranks | `stats.spearmanr(x, y)` |
| Is a sample normal? | one numeric sample | Shapiro–Wilk (or a Q-Q plot) | | `stats.shapiro(x)` |
| Regression coefficient zero? | linear model | t-test on $\hat{\beta}_j$ | $t = \hat{\beta}_j/\text{SE}_j$, df $n - p$ | `sm.OLS(y, X).fit().summary()` |

Expected cell counts below about 5 make the chi-square approximation unreliable; use Fisher's exact test. With very large samples, every test is "significant": judge the effect size.

## Multiple testing

| Method | Controls | Rule | Code |
|---|---|---|---|
| None | nothing | $p < \alpha$; with $m$ true nulls, $P(\text{any false positive}) = 1 - (1-\alpha)^m$ | |
| Bonferroni | family-wise error rate | $p < \alpha/m$ | `multipletests(p, alpha, "bonferroni")` |
| Holm | family-wise error rate, more power | step-down Bonferroni | `multipletests(p, alpha, "holm")` |
| Benjamini–Hochberg | false discovery rate | largest $k$ with $p_{(k)} \leq \frac{k}{m}\alpha$; reject the $k$ smallest | `multipletests(p, alpha, "fdr_bh")` |

`multipletests` is in `statsmodels.stats.multitest`. Hidden multiple testing: many metrics, many segments, many model variants, and peeking at a running A/B test.

## Bayesian updating

$$
p(\theta \mid \text{data}) \propto p(\text{data} \mid \theta)\,p(\theta)
$$

| Likelihood | Conjugate prior | Posterior after data |
|---|---|---|
| Bernoulli/binomial: $k$ successes in $n$ | Beta$(a, b)$ | Beta$(a + k,\ b + n - k)$ |
| Poisson: counts $x_1..x_n$ | Gamma$(\alpha, \text{rate } \beta)$ | Gamma$(\alpha + \sum x_i,\ \beta + n)$ |
| Normal, known $\sigma^2$: mean $\mu$ | $\mathcal{N}(\mu_0, \tau^2)$ | normal, precision $\frac{1}{\tau^2} + \frac{n}{\sigma^2}$, mean = precision-weighted average of $\mu_0$ and $\bar{x}$ |

A 95% **credible interval** (`post.ppf([0.025, 0.975])`) contains the parameter with 95% posterior probability. $P(B > A)$: sample from both posteriors and compare, `np.mean(post_B.rvs(N) > post_A.rvs(N))`.

## Information theory

$$
H(p) = -\sum_x p(x)\log p(x), \quad H(p, q) = -\sum_x p(x)\log q(x), \quad D_{\text{KL}}(p \parallel q) = \sum_x p(x)\log\frac{p(x)}{q(x)} \geq 0
$$

Average cross-entropy on one-hot labels = negative log-likelihood. `stats.entropy(p)` gives $H(p)$; `stats.entropy(p, q)` gives $D_{\text{KL}}(p \parallel q)$; add `base=2` for bits.

## Common mistakes

- Flipping a conditional: $P(\text{positive} \mid \text{disease}) \neq P(\text{disease} \mid \text{positive})$.
- Adding variances (or treating data as independent) when observations are correlated, such as repeated rows per user.
- Mixing up `ddof` between NumPy and pandas.
- Reading "not significant" as "no effect", or "significant" as "important".
- Testing many things and reporting the winner without correction.
- Using a normal approximation with tiny samples or extreme proportions.
