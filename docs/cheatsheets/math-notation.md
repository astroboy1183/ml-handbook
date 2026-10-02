# Math Notation Cheat Sheet

How to read the symbols in ML papers, textbooks, and this handbook, how to say them aloud, and the key formulas from Level 1, each with its NumPy equivalent.

Related chapters: [Linear algebra](../chapters/01-math-foundations/01-linear-algebra.md) · [Matrix decompositions](../chapters/01-math-foundations/02-matrix-decompositions.md) · [Calculus and gradients](../chapters/01-math-foundations/03-calculus-and-gradients.md) · [Probability](../chapters/01-math-foundations/04-probability.md) · [Statistics](../chapters/01-math-foundations/05-statistics.md) · [Information theory and optimization](../chapters/01-math-foundations/06-information-theory-and-optimization.md)

## Conventions used in this handbook

| Notation | Meaning |
|---|---|
| $x$, $a$, $\lambda$ | a scalar (a single number): lowercase italic |
| $\mathbf{x}$, $\mathbf{w}$ | a vector: bold lowercase; a column by default |
| $\mathbf{X}$, $\mathbf{A}$ | a matrix: bold uppercase (plain $X$ when it's clear) |
| $X$, $Y$ | a random variable: uppercase (context separates it from a matrix) |
| $x_i$, $\mathbf{x}_i$ | the $i$-th entry of $\mathbf{x}$, or the $i$-th sample (a vector) |
| $A_{ij}$ | the entry of $\mathbf{A}$ in row $i$, column $j$ |
| $n$, $d$ | number of samples, number of features |
| $\mathbf{X} \in \mathbb{R}^{n \times d}$ | the design matrix: one row per sample, one column per feature |
| $\theta$, or $\mathbf{w}$ and $b$ | model parameters (weights and bias) |
| $\hat{y}$, $\hat{\theta}$ | a prediction or an estimate ("y hat", "theta hat") |
| $\mathcal{L}$ | a loss function |
| $\eta$ | learning rate |
| $\sigma(z)$ | the sigmoid $1/(1 + e^{-z})$; but $\sigma$ alone is usually a standard deviation |
| $\theta^*$, $\mathbf{w}^*$ | the optimal (or true) value |
| $\mathbf{x}^{(i)}$ | sample $i$ in some texts, to keep subscripts free for features |

## Greek letters

| Letter | Name | Typical meaning in ML and statistics |
|---|---|---|
| $\alpha$ | alpha | significance level; regularization strength; a step size |
| $\beta$ | beta | regression coefficients; Type II error rate; a Lagrange multiplier or inverse temperature |
| $\gamma$ | gamma | discount factor (RL); a kernel width; momentum |
| $\delta$, $\Delta$ | delta | a small change; $\Delta$ also "difference" |
| $\epsilon$, $\varepsilon$ | epsilon | a tiny number (tolerance); noise or error term |
| $\zeta$ | zeta | rarely used; sometimes slack variables |
| $\eta$ | eta | learning rate |
| $\theta$, $\Theta$ | theta | model parameters; an angle |
| $\kappa$ | kappa | condition number |
| $\lambda$, $\Lambda$ | lambda | eigenvalue (and $\mathbf{\Lambda}$, the diagonal matrix of them); regularization strength; Poisson or exponential rate; Lagrange multiplier |
| $\mu$ | mu | mean |
| $\nu$ | nu | degrees of freedom |
| $\xi$ | xi | slack variables (SVMs); a generic random quantity |
| $\pi$, $\Pi$ | pi | 3.14159...; a policy (RL); mixture weights; $\prod$ is a product (see below) |
| $\rho$ | rho | correlation |
| $\sigma$, $\Sigma$ | sigma | standard deviation; sigmoid; singular value; $\mathbf{\Sigma}$ is a covariance matrix or the diagonal matrix of the SVD; $\sum$ is a sum |
| $\tau$ | tau | temperature; a time constant; prior scale |
| $\phi$, $\Phi$ | phi | a feature map $\phi(\mathbf{x})$; $\Phi$ is the standard normal CDF |
| $\chi$ | chi | as in the chi-square ($\chi^2$) distribution |
| $\psi$ | psi | a generic function |
| $\omega$, $\Omega$ | omega | an outcome; $\Omega$ is the sample space; $O$ and $\Omega$ also bound growth rates |

## Sets, numbers, and logic

| Symbol | Read as | Example |
|---|---|---|
| $\mathbb{R}$ | the real numbers | $x \in \mathbb{R}$ |
| $\mathbb{R}^d$ | vectors of $d$ real numbers | $\mathbf{x} \in \mathbb{R}^3$ |
| $\mathbb{R}^{m \times n}$ | $m$-by-$n$ real matrices | $\mathbf{W} \in \mathbb{R}^{64 \times 10}$ |
| $\mathbb{N}$, $\mathbb{Z}$ | natural numbers, integers | counts, labels |
| $\in$, $\notin$ | "is in", "is not in" | $y \in \{0, 1\}$ |
| $\{\, x : x > 0 \,\}$ | "the set of $x$ such that $x > 0$" | |
| $[a, b]$, $(a, b)$ | closed interval (includes ends), open interval | $p \in [0, 1]$ |
| $A \cup B$, $A \cap B$ | union ("or"), intersection ("and") | |
| $A \subseteq B$ | "$A$ is a subset of $B$" | |
| $\lvert A \rvert$ | size of a set; absolute value of a number | |
| $\forall$, $\exists$ | "for all", "there exists" | $\forall i$: for every $i$ |
| $\Rightarrow$, $\iff$ | "implies", "if and only if" | |
| $:=$ or $\triangleq$ | "is defined as" | $\bar{x} := \frac{1}{n}\sum_i x_i$ |
| $\approx$, $\propto$ | "approximately equal", "proportional to" | posterior $\propto$ likelihood × prior |
| $\sim$ | "is distributed as" | $X \sim \mathcal{N}(0, 1)$ |
| $f: \mathbb{R}^n \to \mathbb{R}^m$ | "$f$ maps $n$-vectors to $m$-vectors" | |
| $O(n^2)$ | "order $n$ squared": grows at most like $n^2$ | cost of an algorithm, size of an error |

## Reading sums, products, and optimization

$\sum_{i=1}^{n} x_i$ reads "the sum, for $i$ from 1 to $n$, of $x_i$". It's a `for` loop that accumulates: `sum(x[i] for i in range(n))`, or `x.sum()`.

$\prod_{i=1}^{n} p_i$ reads "the product, for $i$ from 1 to $n$, of $p_i$": `np.prod(p)`. Products of probabilities underflow, which is why likelihoods are computed as sums of logs: $\log\prod_i p_i = \sum_i \log p_i$.

Double sums nest like loops: $\sum_{i}\sum_{j} A_{ij}$ adds every entry (`A.sum()`). A subscript without limits, such as $\sum_i$, means "over all valid $i$".

$\max_{x} f(x)$ is the largest *value* of $f$. $\arg\max_{x} f(x)$ is the *input* that achieves it. The same goes for $\min$ and $\arg\min$:

$$
\hat{\mathbf{w}} = \arg\min_{\mathbf{w}}\ \frac{1}{n}\sum_{i=1}^{n}\left(y_i - \mathbf{w}^\top\mathbf{x}_i\right)^2
$$

reads "w hat is the w that minimizes the mean squared error". In NumPy, `np.max(f_values)` versus `np.argmax(f_values)`.

"Subject to" (often "s.t.") introduces constraints: $\min_{\mathbf{x}} f(\mathbf{x})$ s.t. $g(\mathbf{x}) = 0$.

$\mathbf{1}[\text{condition}]$ is the **indicator**: 1 if the condition holds, else 0. In code, `(condition).astype(int)`. The **Kronecker delta** $\delta_{ij}$ is 1 if $i = j$, else 0: the entries of the identity matrix.

## Linear algebra

| Notation | Read as | Meaning | NumPy |
|---|---|---|---|
| $\mathbf{x}^\top$, $\mathbf{A}^\top$ | "x transpose" | swap rows and columns | `x.T`, `A.T` |
| $\mathbf{a}^\top\mathbf{b}$, $\mathbf{a} \cdot \mathbf{b}$, $\langle \mathbf{a}, \mathbf{b} \rangle$ | "a dot b" | $\sum_i a_i b_i$ | `a @ b` |
| $\mathbf{a}\mathbf{b}^\top$ | "outer product" | matrix with entries $a_i b_j$ | `np.outer(a, b)` |
| $\mathbf{A}\mathbf{B}$ | "A times B" | matrix product | `A @ B` |
| $\mathbf{A} \odot \mathbf{B}$ | "Hadamard product" | entry-wise product | `A * B` |
| $\lVert \mathbf{x} \rVert$, $\lVert \mathbf{x} \rVert_2$ | "norm of x" | Euclidean length $\sqrt{\sum_i x_i^2}$ | `np.linalg.norm(x)` |
| $\lVert \mathbf{x} \rVert_1$ | "L1 norm" | $\sum_i \lvert x_i \rvert$ | `np.linalg.norm(x, 1)` |
| $\lVert \mathbf{x} \rVert_\infty$ | "max norm" | $\max_i \lvert x_i \rvert$ | `np.linalg.norm(x, np.inf)` |
| $\lVert \mathbf{A} \rVert_F$ | "Frobenius norm" | $\sqrt{\sum_{ij} A_{ij}^2}$ | `np.linalg.norm(A, "fro")` |
| $\mathbf{I}$, $\mathbf{I}_d$ | "identity" | ones on the diagonal | `np.eye(d)` |
| $\mathbf{A}^{-1}$ | "A inverse" | $\mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$ | `np.linalg.solve(A, b)` rather than `inv` |
| $\mathbf{A}^{+}$ | "pseudoinverse" | least squares / minimum-norm inverse | `np.linalg.pinv(A)` |
| $\det(\mathbf{A})$, $\lvert \mathbf{A} \rvert$ | "determinant" | volume scaling factor | `np.linalg.det(A)` |
| $\operatorname{tr}(\mathbf{A})$ | "trace" | sum of the diagonal | `np.trace(A)` |
| $\operatorname{rank}(\mathbf{A})$ | "rank" | number of independent columns | `np.linalg.matrix_rank(A)` |
| $\operatorname{diag}(\mathbf{v})$ | "diag of v" | diagonal matrix from a vector | `np.diag(v)` |
| $\mathbf{A} \succeq 0$, $\mathbf{A} \succ 0$ | "positive semi-definite", "positive definite" | $\mathbf{x}^\top\mathbf{A}\mathbf{x} \geq 0$ (or $> 0$) | `np.linalg.eigvalsh(A).min() >= 0` |
| $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$ | "SVD" | rotate, stretch, rotate | `np.linalg.svd(A, full_matrices=False)` |
| $\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^\top$ | "eigendecomposition" (symmetric) | stretch along orthogonal axes | `np.linalg.eigh(A)` |
| $\kappa(\mathbf{A})$ | "condition number" | $\sigma_{\max}/\sigma_{\min}$ | `np.linalg.cond(A)` |

## Calculus

| Notation | Read as | Meaning |
|---|---|---|
| $f'(x)$, $\frac{df}{dx}$ | "f prime", "d f d x" | derivative: slope of $f$ at $x$ |
| $f''(x)$, $\frac{d^2 f}{dx^2}$ | "second derivative" | rate of change of the slope (curvature) |
| $\frac{\partial f}{\partial x_i}$ | "partial f partial x i" | derivative in $x_i$ with other variables held fixed |
| $\nabla f$, $\nabla_{\mathbf{w}} \mathcal{L}$ | "grad f", "gradient of L with respect to w" | vector of partial derivatives; points uphill |
| $\mathbf{J}$, $\frac{\partial \mathbf{f}}{\partial \mathbf{x}}$ | "Jacobian" | matrix $J_{ij} = \partial f_i / \partial x_j$ |
| $\mathbf{H}$, $\nabla^2 f$ | "Hessian" | matrix of second partials $\partial^2 f / \partial x_i \partial x_j$ |
| $\lim_{h \to 0}$ | "the limit as h goes to zero" | the value approached as $h$ shrinks |
| $\int_a^b f(x)\,dx$ | "the integral from a to b of f" | area under $f$ between $a$ and $b$ |
| $\exp(x)$, $e^x$ | "e to the x" | `np.exp(x)` |
| $\log x$, $\ln x$ | "log", "natural log" | natural log in ML unless a base is shown; `np.log` |
| $\log_2 x$ | "log base 2" | information in bits; `np.log2` |

Key rules: $(fg)' = f'g + fg'$; chain rule $\frac{d}{dx}f(g(x)) = f'(g(x))\,g'(x)$; $\nabla_{\mathbf{x}}(\mathbf{a}^\top\mathbf{x}) = \mathbf{a}$; $\nabla_{\mathbf{x}}(\mathbf{x}^\top\mathbf{A}\mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$; $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

## Probability and statistics

| Notation | Read as | Meaning |
|---|---|---|
| $P(A)$, $\Pr(A)$ | "probability of A" | |
| $P(A \mid B)$ | "probability of A given B" | $P(A \cap B)/P(B)$ |
| $p(x)$, $f(x)$ | "p of x" | PMF or PDF |
| $p(x \mid \theta)$, $p_\theta(x)$ | "p of x given theta" | a model with parameters $\theta$ |
| $F(x)$ | "CDF" | $P(X \leq x)$ |
| $\mathbb{E}[X]$, $\mathbb{E}_{x \sim p}[f(x)]$ | "expected value", "expectation over x drawn from p" | probability-weighted average |
| $\operatorname{Var}(X)$, $\sigma^2$ | "variance" | $\mathbb{E}[(X - \mu)^2]$ |
| $\operatorname{Cov}(X, Y)$ | "covariance" | $\mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$ |
| $\mathcal{N}(\mu, \sigma^2)$ | "normal with mean mu, variance sigma squared" | note: variance, while SciPy's `scale` is the sd |
| $X \perp Y$ | "X is independent of Y" | |
| i.i.d. | "independent and identically distributed" | |
| $\bar{x}$ | "x bar" | sample mean |
| $s^2$ | "s squared" | sample variance with $n - 1$ |
| $H_0$, $H_1$ | "null", "alternative" hypothesis | |
| $H(p)$, $H(p, q)$ | "entropy", "cross-entropy" | |
| $D_{\text{KL}}(p \parallel q)$ | "KL divergence from q to p" | $\sum_x p(x)\log\frac{p(x)}{q(x)}$ |

## Key formulas

**Linear algebra.**

$$
\mathbf{a}^\top\mathbf{b} = \lVert \mathbf{a} \rVert\lVert \mathbf{b} \rVert\cos\theta, \qquad
\cos\theta = \frac{\mathbf{a}^\top\mathbf{b}}{\lVert \mathbf{a} \rVert\lVert \mathbf{b} \rVert}, \qquad
\text{proj}_{\mathbf{a}}\mathbf{b} = \frac{\mathbf{a}^\top\mathbf{b}}{\mathbf{a}^\top\mathbf{a}}\,\mathbf{a}
$$

$$
(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top\mathbf{A}^\top, \qquad (\mathbf{A}\mathbf{B})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}, \qquad \det(\mathbf{A}\mathbf{B}) = \det(\mathbf{A})\det(\mathbf{B})
$$

$$
\begin{pmatrix} a & b \\ c & d \end{pmatrix}^{-1} = \frac{1}{ad - bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}
$$

**Least squares.** Normal equations, solution, and projection (hat) matrix:

$$
\mathbf{X}^\top\mathbf{X}\hat{\mathbf{w}} = \mathbf{X}^\top\mathbf{y}, \qquad
\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}, \qquad
\mathbf{P} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top
$$

$$
\nabla_{\mathbf{w}}\frac{1}{n}\lVert \mathbf{X}\mathbf{w} - \mathbf{y} \rVert^2 = \frac{2}{n}\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y}), \qquad
\operatorname{Cov}(\hat{\mathbf{w}}) = \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}
$$

**Eigenvalues and SVD.** $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$; $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$; $\sum_i\lambda_i = \operatorname{tr}(\mathbf{A})$; $\prod_i\lambda_i = \det(\mathbf{A})$. Best rank-$k$ approximation: $\mathbf{A}_k = \sum_{i=1}^{k}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$, with $\lVert \mathbf{A} - \mathbf{A}_k \rVert_F = \sqrt{\sum_{i>k}\sigma_i^2}$. PCA: the principal directions are the right singular vectors of the centered data, with variances $\sigma_i^2/(n-1)$.

**Taylor expansion.**

$$
f(\mathbf{x} + \mathbf{h}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top\mathbf{h} + \frac{1}{2}\mathbf{h}^\top\mathbf{H}(\mathbf{x})\,\mathbf{h}
$$

**Gradient descent.** $\theta_{t+1} = \theta_t - \eta\nabla\mathcal{L}(\theta_t)$, stable on a quadratic for $\eta < 2/\lambda_{\max}(\mathbf{H})$; steps needed grow with $\kappa = \lambda_{\max}/\lambda_{\min}$. Newton's step: $\mathbf{h} = -\mathbf{H}^{-1}\nabla f$.

**Probability.**

$$
P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}, \qquad P(B) = \sum_j P(B \mid A_j)P(A_j)
$$

$$
\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y], \qquad \operatorname{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2, \qquad \operatorname{Var}(aX + b) = a^2\operatorname{Var}(X)
$$

$$
\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X, Y), \qquad \rho = \frac{\operatorname{Cov}(X, Y)}{\sigma_X\sigma_Y}, \qquad \operatorname{Var}(\mathbf{w}^\top\mathbf{X}) = \mathbf{w}^\top\mathbf{\Sigma}\mathbf{w}
$$

$$
\operatorname{SE}(\bar{X}) = \frac{\sigma}{\sqrt{n}}, \qquad \bar{X} \approx \mathcal{N}\!\left(\mu, \frac{\sigma^2}{n}\right) \text{ (CLT)}
$$

**Information theory.**

$$
H(p) = -\sum_x p(x)\log p(x), \qquad H(p, q) = -\sum_x p(x)\log q(x), \qquad D_{\text{KL}}(p \parallel q) = H(p, q) - H(p) \geq 0
$$

$$
\text{softmax}(\mathbf{z})_c = \frac{e^{z_c}}{\sum_j e^{z_j}}, \qquad \nabla_{\mathbf{z}}\left(-\log\text{softmax}(\mathbf{z})_y\right) = \mathbf{p} - \mathbf{y}_{\text{onehot}}
$$

**Constrained optimization.** For $\min f(\mathbf{x})$ subject to $g(\mathbf{x}) = 0$: form $\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) - \lambda g(\mathbf{x})$ and solve $\nabla_{\mathbf{x}}\mathcal{L} = \mathbf{0}$, $g(\mathbf{x}) = 0$. At the optimum, $\nabla f = \lambda\nabla g$.

## Reading a formula: a worked example

Take the L2-regularized logistic regression objective:

$$
\hat{\mathbf{w}} = \arg\min_{\mathbf{w}}\ -\frac{1}{n}\sum_{i=1}^{n}\Big[y_i\log\sigma(\mathbf{w}^\top\mathbf{x}_i) + (1 - y_i)\log\big(1 - \sigma(\mathbf{w}^\top\mathbf{x}_i)\big)\Big] + \lambda\lVert \mathbf{w} \rVert_2^2
$$

Read it from the outside in: "w hat is the weight vector that minimizes the average, over the $n$ training examples, of the log loss, plus lambda times the squared L2 norm of the weights." Then from the inside out: $\mathbf{w}^\top\mathbf{x}_i$ is a score (a dot product), $\sigma(\cdot)$ turns it into a probability, the bracket is the log-probability of the observed label, the minus sign and average make it a negative log-likelihood, and $\lambda\lVert \mathbf{w} \rVert_2^2$ penalizes large weights. In NumPy:

```python
import numpy as np

def objective(w, X, y, lam):
    p = 1 / (1 + np.exp(-(X @ w)))
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + lam * w @ w
```

Each piece of notation maps to one piece of code. When a formula looks intimidating, do this translation, then check it with small numbers.
