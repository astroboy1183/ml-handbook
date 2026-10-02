# NumPy Cheat Sheet

A dense reference for the NumPy 2 features you use every day: arrays, indexing, broadcasting, aggregation, linear algebra, and random numbers.

Related chapters: [NumPy](../chapters/00-python-for-data/04-numpy.md) · [Linear algebra](../chapters/01-math-foundations/01-linear-algebra.md) · [Matrix decompositions](../chapters/01-math-foundations/02-matrix-decompositions.md)

```python
import numpy as np
rng = np.random.default_rng(0)   # one seeded generator per script
```

## Creating arrays

| Code | Result |
|---|---|
| `np.array([1, 2, 3])` | from a list; dtype inferred (`int64`) |
| `np.array(lst, dtype=np.float32)` | explicit dtype |
| `np.zeros((2, 3))`, `np.ones(3)`, `np.full((2, 2), 7)` | constant arrays (default `float64`) |
| `np.empty((2, 2))` | uninitialized memory (fast, garbage values) |
| `np.zeros_like(a)`, `np.ones_like(a)` | same shape and dtype as `a` |
| `np.eye(3)`, `np.identity(3)` | identity matrix |
| `np.arange(0, 10, 2)` | `[0 2 4 6 8]` (stop excluded) |
| `np.linspace(0, 1, 5)` | `[0. 0.25 0.5 0.75 1.]` (stop included) |
| `np.meshgrid(xs, ys)` | coordinate grids for plotting a function of two variables |
| `np.tile([1, 2], 2)` / `np.repeat([1, 2], 2)` | `[1 2 1 2]` / `[1 1 2 2]` |
| `np.asarray(x)` | converts without copying if already an array |

## Attributes and dtypes

| Attribute | Meaning (for `x = np.arange(12).reshape(3, 4)`) |
|---|---|
| `x.shape` | `(3, 4)` |
| `x.ndim`, `x.size` | `2`, `12` |
| `x.dtype`, `x.itemsize` | `int64`, `8` bytes per element |
| `x.nbytes` | `96` |
| `x.strides` | `(32, 8)`: bytes to step to the next row, next column |
| `x.flags["C_CONTIGUOUS"]` | row-major contiguous? (`x.T` is F-contiguous instead) |

| dtype | Range or precision | Typical use |
|---|---|---|
| `bool` | True/False | masks |
| `int8`/`int16`/`int32`/`int64` | int8: −128..127; int32 max 2,147,483,647 | counts, ids, labels |
| `uint8` | 0..255 | images |
| `float32` | ~7 significant digits, eps ≈ 1.19e-7 | deep learning, big arrays |
| `float64` | ~16 significant digits, eps ≈ 2.2e-16 | default for science and stats |
| `complex128` | complex numbers | FFT |
| `str_` / `object` | text / arbitrary Python objects | avoid for numeric work |

```python
x.astype(np.float32)                 # cast (always copies)
np.iinfo(np.int32).max               # 2147483647
np.finfo(np.float32).eps             # 1.1920929e-07
np.array([1, 2], dtype=np.int8) + np.int8(127)   # [-128 -127]  silent overflow wraps around!
```

## Reshaping and combining

| Code | Shape result (from `x.shape == (3, 4)`) |
|---|---|
| `x.reshape(2, 6)`, `x.reshape(2, -1)` | `(2, 6)`; `-1` means "infer this one" |
| `x.ravel()` / `x.flatten()` | `(12,)`; `ravel` returns a view when possible, `flatten` always copies |
| `x.T`, `x.transpose()` | `(4, 3)` (a view) |
| `x[:, np.newaxis, :]`, `x[:, None, :]` | `(3, 1, 4)` |
| `np.expand_dims(x, 0)` | `(1, 3, 4)` |
| `np.squeeze(a)` | removes all length-1 axes |
| `np.moveaxis(a, 0, -1)` | moves an axis, e.g. `(2, 3, 4)` to `(3, 4, 2)` |
| `np.concatenate([x, x], axis=0)` | `(6, 4)`: joins along an *existing* axis |
| `np.stack([x, x])` | `(2, 3, 4)`: joins along a *new* axis |
| `np.vstack`, `np.hstack` | `(6, 4)`, `(3, 8)` |
| `np.column_stack([a, b])` | 1-D arrays become columns |
| `np.split(x, 2, axis=1)` | two `(3, 2)` arrays |

## Indexing

| Kind | Example | Returns |
|---|---|---|
| Single element | `x[1, 2]` | scalar |
| Row / column | `x[-1]`, `x[:, 1]` | 1-D **view** |
| Slice with step | `x[::2, ::-1]` | **view** |
| Fancy (integer arrays) | `x[[0, 2]]`, `x[[0, 2], [1, 3]]` | **copy**; the second picks elements (0,1) and (2,3) |
| Boolean mask | `x[x % 2 == 0]` | **copy**, always 1-D |
| Assignment with a mask | `x[x > 9] = 0` | modifies `x` in place |
| Conditional select | `np.where(x > 5, x, -1)` | new array |
| Positions of True | `np.nonzero(m)`, `np.argwhere(m)` | index tuple / `(k, ndim)` array |
| Sort rows by key | `np.take_along_axis(x, idx, axis=1)` | gather by index array |

!!! warning "Views vs copies"
    Slices are **views**: writing to them changes the original. Fancy and boolean indexing return **copies**. Check with `np.shares_memory(a, b)`. Call `.copy()` when you need independence.

    ```python
    v = x[:2]; v[0, 0] = 99      # x[0, 0] is now 99
    c = x[[0, 1]]; c[0, 0] = 99  # x unchanged
    ```

## Broadcasting rules

Compare shapes **right to left**. Two dimensions are compatible if they are **equal** or **one of them is 1**. Missing leading dimensions count as 1. Size-1 dimensions are stretched (without copying).

| Shape A | Shape B | Result |
|---|---|---|
| `(3, 4)` | `(4,)` | `(3, 4)` |
| `(3, 1)` | `(4,)` | `(3, 4)` |
| `(5, 1, 3)` | `(1, 5, 3)` | `(5, 5, 3)` |
| `(3, 2)` | `(3,)` | **error**: 2 vs 3 |
| `(3, 2)` | `(3, 1)` | `(3, 2)` |

```python
Z = (X - X.mean(axis=0)) / X.std(axis=0)          # standardize columns: (n,d) - (d,)
D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))   # (n,n) pairwise distances
row_means = x.mean(axis=1, keepdims=True)          # (3, 1): broadcasts back against (3, 4)
```

## Aggregations and axes

`axis=k` means "collapse axis `k`". For a 2-D array, `axis=0` aggregates **down the rows** (one result per column), and `axis=1` aggregates **across columns** (one result per row).

| Function | Notes |
|---|---|
| `sum`, `prod`, `mean`, `min`, `max` | `x.sum(axis=0)` → `[12 15 18 21]`; `x.sum(axis=1)` → `[6 22 38]` |
| `argmin`, `argmax` | flat index unless `axis` given; `np.unravel_index(x.argmax(), x.shape)` → `(2, 3)` |
| `cumsum`, `cumprod`, `diff` | running totals; `np.diff([1, 4, 9, 16])` → `[3 5 7]` |
| `var`, `std` | **population** by default (`ddof=0`); use `ddof=1` for the sample version |
| `median`, `percentile(x, [25, 50, 75])`, `quantile(x, 0.5)` | percentiles take 0–100, quantiles take 0–1 |
| `nanmean`, `nansum`, `nanstd`, `nanmax` | ignore NaN (plain `mean` returns `nan`) |
| `any`, `all`, `count_nonzero(x > 5)` | logical reductions |
| `ptp` | max − min |
| `keepdims=True` | keeps the reduced axis with length 1 for broadcasting |

```python
np.var([1, 2, 3, 4])           # 1.25   (divide by n)
np.var([1, 2, 3, 4], ddof=1)   # 1.667  (divide by n - 1)
```

## Element-wise math (ufuncs)

| Category | Functions |
|---|---|
| Arithmetic | `+ - * / // % **`, `np.add`, `np.multiply`, `np.power` |
| Exp/log | `np.exp`, `np.log`, `np.log2`, `np.log10`, `np.log1p` (accurate `log(1+x)` for small x), `np.expm1` |
| Rounding | `np.round` (rounds half to even: `round(2.5) = 2.0`, `round(3.5) = 4.0`), `np.floor`, `np.ceil` |
| Comparison | `np.maximum`, `np.minimum` (element-wise), `np.clip(a, lo, hi)` |
| Other | `np.abs`, `np.sign`, `np.sqrt`, `np.sin`, `np.tanh` |
| Float checks | `np.isnan`, `np.isinf`, `np.isfinite`, `np.isclose`, `np.allclose` |

```python
0.1 + 0.2 == 0.3            # False
np.isclose(0.1 + 0.2, 0.3)  # True: always compare floats with a tolerance
np.nan == np.nan            # False: use np.isnan
```

## Sorting, searching, sets, counting

| Code | Result for `s = np.array([3, 1, 2, 3])` |
|---|---|
| `np.sort(s)` / `s.sort()` | sorted copy / sort in place |
| `np.argsort(s)` | `[1 2 0 3]`: indices that would sort |
| `np.unique(s, return_counts=True)` | `([1 2 3], [1 1 2])` |
| `np.partition(a, k)` | k smallest first, unordered (O(n), faster than sort for top-k) |
| `np.searchsorted([1, 3, 5, 7], 4)` | `2`: insertion point in a sorted array |
| `np.digitize(vals, bins)` | bin index of each value |
| `np.isin([1, 2, 5], [2, 5])` | `[False True True]` |
| `np.intersect1d`, `np.union1d`, `np.setdiff1d` | set operations, sorted output |
| `np.bincount([0, 1, 1, 3])` | `[1 2 0 1]`: counts of non-negative ints |
| `np.histogram(a, bins=3)` | `(counts, bin_edges)` |

## Linear algebra

| Code | Meaning |
|---|---|
| `A @ B`, `np.matmul(A, B)` | matrix product (use `@`, not `*`, which is element-wise) |
| `np.dot(u, v)`, `u @ v` | dot product of 1-D arrays |
| `np.outer(u, v)` | outer product $\mathbf{u}\mathbf{v}^\top$ |
| `np.einsum("ij,j->i", A, b)` | explicit index notation; here $A\mathbf{b}$ |
| `np.linalg.solve(A, b)` | solve $A\mathbf{x} = \mathbf{b}$ (prefer to `inv(A) @ b`) |
| `np.linalg.inv(A)`, `np.linalg.pinv(A)` | inverse, pseudo-inverse |
| `np.linalg.det(A)`, `np.trace(A)` | determinant, trace |
| `np.linalg.norm(v)`, `norm(v, 1)`, `norm(v, np.inf)`, `norm(A, "fro")` | L2, L1, max, Frobenius norms |
| `np.linalg.eig(A)` / `np.linalg.eigh(S)` | eigen-decomposition; use `eigh` for symmetric matrices (sorted ascending, real) |
| `np.linalg.svd(A, full_matrices=False)` | `U, S, Vt` with `A = U @ np.diag(S) @ Vt` |
| `np.linalg.qr(A)` | `Q, R` |
| `np.linalg.lstsq(X, y, rcond=None)` | least squares; returns `(coef, residuals, rank, sv)` |
| `np.linalg.matrix_rank(A)`, `np.linalg.cond(A)` | rank, condition number |

```python
A = np.array([[3.0, 1.0], [1.0, 2.0]]); b = np.array([9.0, 8.0])
np.linalg.solve(A, b)                    # [2. 3.]
X = np.c_[np.ones(4), [0, 1, 2, 3]]      # add an intercept column
coef, *_ = np.linalg.lstsq(X, [1, 3, 5, 7.0], rcond=None)   # [1. 2.]
```

## Random numbers (Generator API)

```python
rng = np.random.default_rng(42)          # seed once; pass rng around
rng.integers(0, 10, size=3)              # ints in [0, 10)
rng.random(2)                            # uniform [0, 1)
rng.uniform(low, high, size)             # uniform [low, high)
rng.normal(loc=0, scale=1, size=(n, d))  # Gaussian
rng.binomial(n=10, p=0.3, size=3); rng.poisson(lam=4, size=3); rng.exponential(scale=2.0, size=2)
rng.choice(["a", "b", "c"], size=4, p=[0.5, 0.3, 0.2])    # weighted sampling
rng.choice(n, size=k, replace=False)     # sample without replacement
rng.permutation(5); rng.shuffle(arr)     # permuted copy / shuffle in place
child_rngs = rng.spawn(2)                # independent streams for parallel work
```

!!! tip "Legacy vs modern"
    Prefer `np.random.default_rng(seed)` over the legacy global `np.random.seed(0)` + `np.random.rand(...)`. A generator object makes randomness explicit, avoids hidden global state, and has better statistical properties. Note that `exponential` takes the **scale** $1/\lambda$, not the rate $\lambda$.

## Saving and loading

| Code | Use |
|---|---|
| `np.save("a.npy", x)` / `np.load("a.npy")` | one array, exact dtype and shape |
| `np.savez_compressed("b.npz", x=x, y=y)`; `d = np.load("b.npz"); d["x"]` | several named arrays |
| `np.savetxt("c.csv", x, delimiter=",", fmt="%d")` / `np.loadtxt(..., delimiter=",")` | text (slower, loses dtype) |
| `np.genfromtxt(path, delimiter=",")` | text with missing values (become `nan`) |

For tables with mixed types, use pandas and Parquet instead.

## Performance tips

- **Vectorize.** Replace Python loops over elements with whole-array operations or ufuncs. Typical speedups are 10–100×.
- **Preallocate** with `np.empty`/`np.zeros` instead of appending in a loop; `np.append` copies the whole array each time.
- **Use the right dtype.** `float32` halves memory versus `float64` when precision allows.
- **Reduce with `axis`** rather than looping over rows.
- **Watch for hidden copies:** fancy indexing, `astype`, `flatten`, and non-contiguous reshapes all copy. `np.ascontiguousarray` makes a C-ordered copy when a library needs one.
- **Use `@` and `np.linalg`** (backed by optimized BLAS/LAPACK) instead of hand-written matrix loops.
- **`np.einsum`** expresses complicated contractions clearly; pass `optimize=True` for big ones.

## Common gotchas

| Gotcha | Fix |
|---|---|
| `a * b` on 2-D arrays is element-wise, not matrix multiplication | use `a @ b` |
| `x.std()` is the population std (`ddof=0`) | use `ddof=1` for sample std (pandas defaults to `ddof=1`) |
| Shape `(n,)` vs `(n, 1)` broadcast to `(n, n)` by accident | check `.shape`; use `reshape(-1, 1)` or `[:, None]` deliberately |
| Writing to a slice modified the original | `.copy()` the slice |
| `np.nan == np.nan` is False; `mean` with NaN gives NaN | `np.isnan`, `np.nanmean` |
| Integer overflow wraps silently (`int8` 127 + 1 = −128) | use wider dtypes; check `np.iinfo` |
| Integer arrays truncate assigned floats (`a[0] = 2.7` stores 2) | create float arrays when you need floats |
| `np.round(2.5)` is `2.0` (round half to even) | expected; use `np.floor(x + 0.5)` for half-up |
| `np.linalg.inv(A) @ b` is slower and less accurate | `np.linalg.solve(A, b)` |
| `np.empty` contains garbage | fill it before reading |
| Comparing floats with `==` | `np.isclose` / `np.allclose` |
