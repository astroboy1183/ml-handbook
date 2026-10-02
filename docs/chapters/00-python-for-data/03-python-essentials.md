# Python Essentials for Data Work

> **Level 0 · Chapter 3** · ⏱️ ~50 min read · Prerequisites: basic Python (variables, loops, functions), [Setting up your environment](02-environment-setup.md)

You already know some Python. This chapter sharpens it for data work: which data structure to use and what it costs, comprehensions, functions done properly, dataclasses, generators for data too big to fit in memory, type hints, working with files and paths, and the classic traps that cause silent bugs in analysis code.

## Why it matters

Wei wrote a script to remove customers who had already received a promotion from a mailing list. It worked in testing on 1,000 customers. In production, with 2 million customers and 500,000 previous recipients, it ran all night and was still going in the morning.

The bug was one line:

<!-- skip-run -->
```python
to_send = [c for c in customers if c not in already_sent]
```

`already_sent` was a list. Checking `c not in already_sent` scans the whole list, all 500,000 entries, for each of the 2 million customers: around a trillion comparisons. Changing one thing, `already_sent = set(already_sent)`, made each check almost instant, and the script finished in under a second.

Python makes it easy to write code that's correct but catastrophically slow, or fast but subtly wrong. Knowing what each data structure costs, and the handful of classic traps, is what separates analysis code that works on a sample from code that works on real data.

## Concepts

### Everything is an object, and names are labels

In Python, every value (a number, a string, a list, a function) is an **object** living in memory. A variable is just a **name** that refers to an object. Assignment never copies an object; it attaches another name to it.

```python
a = [1, 2, 3]
b = a          # b is a second name for the SAME list
b.append(4)
print(a)
print(a is b)
```

```text
[1, 2, 3, 4]
True
```

This is called **aliasing**, and it causes some of the most confusing bugs in data code. When you pass a list or a DataFrame to a function and the function modifies it, the caller's object changes too. To get an independent copy, copy it explicitly: `list(a)`, `a.copy()`, or `copy.deepcopy(a)` for nested structures.

Objects are either **mutable** (can change in place: lists, dicts, sets, NumPy arrays, DataFrames) or **immutable** (can't: ints, floats, strings, tuples, frozensets). Operations on immutable objects always create new objects, so aliasing them is harmless.

`is` checks whether two names refer to the *same object*. `==` checks whether two objects have *equal values*. Use `==` for comparing values, and `is` only for `None` (`if x is None`).

### Data structures and what they cost

Choosing the right container is the single biggest performance decision in plain Python code. The cost of an operation is described with **big-O notation**: how the time grows with the number of items $n$. $O(1)$ means constant time (it doesn't grow with $n$), and $O(n)$ means it grows in proportion to $n$.

| Structure | Ordered? | Mutable? | Lookup by | `x in s` | Typical use |
|---|---|---|---|---|---|
| `list` | yes | yes | position, $O(1)$ | $O(n)$ | sequences you iterate over or append to |
| `tuple` | yes | no | position, $O(1)$ | $O(n)$ | fixed records, dict keys, multiple return values |
| `dict` | insertion order | yes | key, $O(1)$ average | $O(1)$ average | mapping keys to values, counting, grouping |
| `set` | no | yes | — | $O(1)$ average | membership tests, deduplication, set algebra |

Why are dict and set lookups $O(1)$? They're **hash tables**. Python computes a number from each key, its **hash**, and uses it to jump straight to the slot where the key should be, instead of scanning. This only works if the hash never changes, which is why dict keys and set elements must be immutable (hashable). You can use a tuple as a key, but not a list.

A list, by contrast, is an array of references stored in a row. Indexing by position is instant, because the address is computed directly. But searching for a value means checking each element in turn.

!!! tip "Rule of thumb"
    If you test membership (`x in collection`) more than once, use a set or a dict. If you need to look items up by an identifier, use a dict keyed by that identifier.

### Comprehensions

A **comprehension** builds a collection from an iterable in one expression. It's usually clearer, and a little faster, than the equivalent loop.

```python
prices = [12.5, 8.0, 15.25, 3.0]

with_tax = [round(p * 1.2, 2) for p in prices]           # list comprehension
cheap = [p for p in prices if p < 10]                     # with a filter
by_index = {i: p for i, p in enumerate(prices)}           # dict comprehension
rounded = {round(p) for p in prices}                      # set comprehension

print(with_tax)
print(cheap)
print(by_index)
print(rounded)
```

```text
[15.0, 9.6, 18.3, 3.6]
[8.0, 3.0]
{0: 12.5, 1: 8.0, 2: 15.25, 3: 3.0}
{8, 3, 12, 15}
```

The pattern is `[expression for item in iterable if condition]`. If you need nested loops or a complex condition, a regular `for` loop is often more readable. Clarity beats cleverness.

Swap the square brackets for parentheses and you get a **generator expression**, which produces items one at a time instead of building a list. More on why that matters below.

### Functions done properly

Python functions are flexible. A few features matter a lot in data work.

**Default arguments, keyword arguments, and keyword-only arguments:**

```python
def summarize(values, *, decimals=2, skip_none=True):
    """Return the mean of values, rounded."""
    if skip_none:
        values = [v for v in values if v is not None]
    return round(sum(values) / len(values), decimals)

print(summarize([1, 2, None, 4]))
print(summarize([1, 2, 4], decimals=0))
```

```text
2.33
2.0
```

The bare `*` makes every argument after it **keyword-only**: callers must write `decimals=0`, not just `0`. This prevents mistakes like `summarize(data, 0, False)`, where nobody can tell what `0` and `False` mean. Libraries like scikit-learn and pandas use keyword-only arguments heavily for exactly this reason.

**`*args` and `**kwargs`** collect any extra positional and keyword arguments into a tuple and a dict. You'll see them most when a function passes options through to another function:

```python
def plot_series(values, **plot_options):
    print(f"plotting {len(values)} points with options {plot_options}")

plot_series([1, 2, 3], color="red", linewidth=2)
```

```text
plotting 3 points with options {'color': 'red', 'linewidth': 2}
```

**Lambdas** are tiny anonymous functions, for when you need a function for a moment, most often as a `key` for sorting:

```python
orders = [("alice", 120), ("bob", 45), ("carol", 300)]
print(sorted(orders, key=lambda order: order[1], reverse=True))
```

```text
[('carol', 300), ('alice', 120), ('bob', 45)]
```

**Functions are objects.** You can store them in variables and dicts, pass them as arguments, and return them from other functions. That's what lets you pass a scoring function to a model-selection tool, or a transformation to `df.apply`.

### Classes and dataclasses

A **class** bundles data and the functions that work on it (its **methods**). You'll *use* classes constantly: every DataFrame, NumPy array, and scikit-learn model is an instance of a class. Writing your own is useful for organizing more complex code.

For classes that mainly hold data, use a **dataclass**. The `@dataclass` decorator writes the boilerplate for you: the `__init__` method, a readable `__repr__`, and equality comparison.

```python
from dataclasses import dataclass, field

@dataclass
class Experiment:
    name: str
    learning_rate: float = 0.01
    tags: list[str] = field(default_factory=list)

    def label(self) -> str:
        return f"{self.name} (lr={self.learning_rate})"

exp = Experiment("baseline", tags=["v1"])
print(exp)
print(exp.label())
print(exp == Experiment("baseline", tags=["v1"]))
```

```text
Experiment(name='baseline', learning_rate=0.01, tags=['v1'])
baseline (lr=0.01)
True
```

Note `field(default_factory=list)`: each instance gets its own new empty list. Writing `tags: list[str] = []` would be an error, for the same reason as the mutable default trap described below. Add `frozen=True` (`@dataclass(frozen=True)`) to make instances immutable and hashable, which is ideal for configuration objects.

### Iterators and generators

An **iterable** is anything you can loop over: lists, strings, dicts, files, ranges. Under the hood, a `for` loop asks the iterable for an **iterator** and calls `next()` on it until it's exhausted.

A **generator** is a function that produces values one at a time, pausing between them. You write one with `yield` instead of `return`:

```python
def read_in_chunks(n_rows, chunk_size):
    """Yield (start, end) row ranges without building them all at once."""
    for start in range(0, n_rows, chunk_size):
        yield start, min(start + chunk_size, n_rows)

for chunk in read_in_chunks(n_rows=10, chunk_size=4):
    print(chunk)
```

```text
(0, 4)
(4, 8)
(8, 10)
```

Each time the loop asks for the next value, the function runs until the next `yield`, hands over the value, and freezes, keeping all its local variables, until it's asked again.

Why this matters for data: a generator holds one item in memory at a time. You can process a 50 GB log file line by line on a laptop with 8 GB of RAM, because you never hold more than a line. Compare the memory a list and a generator need for the same million values:

```python
import sys

squares_list = [x * x for x in range(1_000_000)]
squares_gen = (x * x for x in range(1_000_000))

print(f"list:      {sys.getsizeof(squares_list):>10,} bytes")
print(f"generator: {sys.getsizeof(squares_gen):>10,} bytes")
print(sum(squares_gen) == sum(squares_list))
```

```text
list:       8,448,728 bytes
generator:        200 bytes
True
```

The list's 8 MB is just its array of references: the integer objects themselves take even more. The generator takes a couple of hundred bytes whatever its length. The trade-off: a generator can be consumed only **once**. After the `sum` above, `squares_gen` is exhausted, and looping over it again yields nothing.

`pandas.read_csv(..., chunksize=100_000)` uses this idea to read huge files in pieces, and PyTorch's data loaders use it to stream training batches.

### Type hints

**Type hints** annotate what types a function expects and returns:

```python
def mean_price(prices: list[float], *, ignore_zero: bool = False) -> float:
    ...
```

Python doesn't enforce them at runtime: `mean_price("oops")` still runs, and fails in whatever way it fails. Hints are for people and tools. They document intent, let your editor autocomplete and flag mistakes as you type, and let a type checker such as `mypy` or `pyright` catch whole classes of bugs before you run anything. Use them in any code you'll reuse, especially functions in `src/`.

Common forms: `int`, `float`, `str`, `list[int]`, `dict[str, float]`, `tuple[int, int]`, `int | None` (an int or None), and `Callable[[float], float]` (a function from float to float, imported from `collections.abc`).

### Files and paths

Use **`pathlib`** for file paths. It handles the differences between Windows and Unix separators for you, and its methods read naturally:

```python
from pathlib import Path

data_dir = Path("data") / "raw"           # / joins path parts
csv_path = data_dir / "sales_2024.csv"
print(csv_path)
print(csv_path.name, csv_path.stem, csv_path.suffix)
print(csv_path.with_suffix(".parquet"))
```

```text
data/raw/sales_2024.csv
sales_2024.csv sales_2024 .csv
data/raw/sales_2024.parquet
```

Open files with a **`with` statement**, a **context manager** that guarantees the file is closed even if an error occurs midway:

```python
import json

config = {"model": "logistic", "C": 1.0, "features": ["age", "tenure"]}
path = Path("config.json")

with path.open("w") as f:
    json.dump(config, f, indent=2)

with path.open() as f:
    loaded = json.load(f)

print(loaded == config)
print(path.read_text()[:40])
```

```text
True
{
  "model": "logistic",
  "C": 1.0,
  "
```

For small files, `path.read_text()` and `path.write_text()` are one-line shortcuts. For tabular data you'll almost always use pandas instead, as covered in the [pandas chapter](05-pandas.md).

### Idioms worth knowing by heart

These come up constantly in data code:

```python
from collections import Counter, defaultdict

events = ["click", "view", "click", "buy", "view", "click"]

# Count things
counts = Counter(events)
print(counts.most_common(2))

# Group things
users = [("alice", "uk"), ("bob", "us"), ("carol", "uk")]
by_country = defaultdict(list)
for name, country in users:
    by_country[country].append(name)
print(dict(by_country))

# Loop with an index, or over two sequences in step
for i, (name, country) in enumerate(users, start=1):
    print(i, name, country)
for e, u in zip(events, users):
    pass  # zip stops at the shorter sequence

# Unpacking
first, *rest = [10, 20, 30, 40]
print(first, rest)

# f-strings with formatting
ratio = 0.123456
print(f"{ratio:.1%}  {1234567.891:,.2f}  {42:>6d}")
```

```text
[('click', 3), ('view', 2)]
{'uk': ['alice', 'carol'], 'us': ['bob']}
1 alice uk
2 bob us
3 carol uk
10 [20, 30, 40]
12.3%  1,234,567.89      42
```

`Counter` counts hashable items. `defaultdict(list)` creates an empty list the first time you access a missing key, which removes the "if key not in dict" boilerplate when grouping. `zip` silently stops at the shorter sequence; in Python 3.10+, pass `strict=True` to raise an error if the lengths differ, which catches misaligned data.

### The classic traps

These bugs are silent: no error, just wrong results.

**1. Mutable default arguments.** Default values are created *once*, when the function is defined, not each time it's called:

```python
def add_tag(tag, tags=[]):         # BUG: one shared list for every call
    tags.append(tag)
    return tags

print(add_tag("a"))
print(add_tag("b"))                # surprise!
```

```text
['a']
['a', 'b']
```

The fix is to use `None` as the default and create the list inside: `def add_tag(tag, tags=None): tags = [] if tags is None else tags`.

**2. Floating-point equality.** Computers store floats in binary, and most decimal fractions can't be represented exactly:

```python
import math

print(0.1 + 0.2)
print(0.1 + 0.2 == 0.3)
print(math.isclose(0.1 + 0.2, 0.3))
```

```text
0.30000000000000004
False
True
```

Never compare floats with `==`. Use `math.isclose` (or `np.isclose` / `np.allclose` for arrays). For money, store integer cents or use `decimal.Decimal`.

**3. Modifying a list while looping over it** skips elements, because the loop's position counter doesn't know items moved:

```python
nums = [1, 2, 2, 3, 4]
for n in nums:
    if n == 2:
        nums.remove(n)
print(nums)          # one 2 survives
```

```text
[1, 2, 3, 4]
```

Build a new list instead: `nums = [n for n in nums if n != 2]`.

**4. Late-binding closures.** Functions created in a loop look up the loop variable when they're *called*, not when they're created:

```python
multipliers = [lambda x: x * i for i in range(3)]
print([m(10) for m in multipliers])          # all use the final i == 2
fixed = [lambda x, i=i: x * i for i in range(3)]
print([m(10) for m in fixed])
```

```text
[20, 20, 20]
[0, 10, 20]
```

Binding `i=i` as a default argument captures the current value. This one shows up when building lists of callbacks or feature transformations.

**5. Integer division and rounding.** `/` always returns a float, while `//` floors. Python's `round` uses **banker's rounding** (round half to even), which surprises people:

```python
print(7 / 2, 7 // 2, -7 // 2)
print(round(0.5), round(1.5), round(2.5))
```

```text
3.5 3 -4
0 2 2
```

`-7 // 2` is `-4` because floor division rounds toward negative infinity, not toward zero. Banker's rounding avoids a systematic upward bias when rounding many values, which is good statistically, but surprising if you expected `round(2.5) == 3`.

## In practice

### Measuring the cost of the wrong data structure

Here's Wei's bug at a smaller scale. We time membership tests with the standard library's `timeit` module:

```python
import timeit

already_sent_list = list(range(50_000))
already_sent_set = set(already_sent_list)
customers = range(40_000, 41_000)       # 1,000 customers to check

t_list = timeit.timeit(lambda: [c for c in customers if c not in already_sent_list], number=1)
t_set = timeit.timeit(lambda: [c for c in customers if c not in already_sent_set], number=1)

print(f"list: {t_list * 1000:9.3f} ms")
print(f"set:  {t_set * 1000:9.3f} ms")
print(f"set is about {t_list / t_set:,.0f}x faster")
```

```text
list:   199.194 ms
set:      0.028 ms
set is about 7,216x faster
```

Your timings will differ, but the ratio will be enormous, and it grows with the size of the list. With 500,000 previous recipients and 2 million customers, the list version does roughly 20,000 times more work than here, which is how a sub-second job became an overnight one.

### Streaming a large file with a generator

Suppose you need the total revenue from a CSV file too big to load into memory. A generator pipeline processes it one line at a time. First we create a sample file:

```python
import csv
import random
from pathlib import Path

random.seed(0)
path = Path("orders.csv")
with path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["order_id", "customer", "amount"])
    for i in range(100_000):
        writer.writerow([i, f"c{random.randint(1, 5000)}", round(random.uniform(5, 500), 2)])

print(f"{path.stat().st_size / 1e6:.1f} MB")
```

```text
1.9 MB
```

Now a pipeline of generators. Each stage pulls one item at a time from the stage before:

```python
def read_rows(path):
    with path.open(newline="") as f:
        yield from csv.DictReader(f)      # one dict per line, lazily

def amounts(rows):
    for row in rows:
        yield float(row["amount"])

total = sum(amounts(read_rows(path)))
print(f"total revenue: {total:,.2f}")
```

```text
total revenue: 25,268,161.96
```

At no point does more than one row live in memory, so this works the same on a 2 MB file or a 200 GB one. `yield from` hands over every item from another iterable, here the CSV reader.

### A typed, documented helper

Code you'll reuse deserves type hints, a docstring, and keyword-only options. This is the standard to aim for in your `src/` folder:

```python
def top_n(counts: dict[str, int], n: int = 3, *, min_count: int = 1) -> list[tuple[str, int]]:
    """Return the n most common items with at least min_count occurrences.

    Ties are broken alphabetically so results are deterministic.
    """
    eligible = [(k, v) for k, v in counts.items() if v >= min_count]
    return sorted(eligible, key=lambda kv: (-kv[1], kv[0]))[:n]

print(top_n({"click": 3, "view": 3, "buy": 1, "share": 2}, n=3, min_count=2))
```

```text
[('click', 3), ('view', 3), ('share', 2)]
```

The sort key `(-count, name)` sorts by count descending, then by name ascending, so ties come out in the same order on every run. Deterministic output makes results reproducible and tests reliable.

## Exercises

### Exercise 1: Predict the output (easy)

Without running it, predict what this prints. Then run it and explain any surprises.

```python
a = [[0] * 3] * 3
a[0][0] = 1
print(a)
```

??? success "Solution"

    It prints `[[1, 0, 0], [1, 0, 0], [1, 0, 0]]`: every row changed. `[[0] * 3] * 3` creates *one* inner list and a list containing three references to it: aliasing again. Changing `a[0][0]` changes the single shared inner list, which appears in all three rows. Build independent rows with a comprehension:

    ```python
    a = [[0] * 3 for _ in range(3)]
    a[0][0] = 1
    print(a)
    ```

    ```text
    [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    ```

    (`[0] * 3` is safe because integers are immutable.)

### Exercise 2: Word frequencies (easy)

Given a block of text, print the five most common words, ignoring case and punctuation. Use `Counter` and a comprehension.

??? success "Solution"

    ```python
    import string
    from collections import Counter

    text = """The model learned the pattern. The pattern was noisy,
    but the model learned it anyway; noisy data is the norm."""

    clean = text.lower().translate(str.maketrans("", "", string.punctuation))
    counts = Counter(clean.split())
    print(counts.most_common(5))
    ```

    ```text
    [('the', 5), ('model', 2), ('learned', 2), ('pattern', 2), ('noisy', 2)]
    ```

    `str.maketrans("", "", string.punctuation)` builds a table that deletes punctuation characters, and `translate` applies it. `most_common` breaks ties by first appearance.

### Exercise 3: A config dataclass (medium)

Write a frozen dataclass `TrainConfig` with fields `model` (str), `learning_rate` (float, default 0.001), `epochs` (int, default 10), and `features` (a tuple of str, default empty). Add a method `to_dict()` that returns a plain dict. Show that you can use a config as a dict key, and that you can't modify it.

??? success "Solution"

    ```python
    from dataclasses import asdict, dataclass

    @dataclass(frozen=True)
    class TrainConfig:
        model: str
        learning_rate: float = 0.001
        epochs: int = 10
        features: tuple[str, ...] = ()

        def to_dict(self) -> dict:
            return asdict(self)

    cfg = TrainConfig("mlp", features=("age", "income"))
    results = {cfg: 0.87}                    # frozen dataclasses are hashable
    print(results[TrainConfig("mlp", features=("age", "income"))])
    print(cfg.to_dict())
    try:
        cfg.epochs = 20
    except Exception as e:
        print(type(e).__name__)
    ```

    ```text
    0.87
    {'model': 'mlp', 'learning_rate': 0.001, 'epochs': 10, 'features': ('age', 'income')}
    FrozenInstanceError
    ```

    `features` is a tuple, not a list, because a frozen dataclass should only contain immutable values. A list field would make the hash fail and could still be changed in place.

### Exercise 4: Find the bug (medium)

This function is meant to normalize each list of scores to the range 0–1 without changing the input. It has two bugs. Find and fix them.

```python
def normalize_all(score_lists, out=[]):
    for scores in score_lists:
        lo, hi = min(scores), max(scores)
        for i in range(len(scores)):
            scores[i] = (scores[i] - lo) / (hi - lo)
        out.append(scores)
    return out
```

??? success "Solution"

    1. **The mutable default `out=[]`** keeps growing across calls, so the second call returns the first call's results too.
    2. **It modifies the caller's lists in place** (`scores[i] = ...`), destroying the input, which the function promised not to do.

    There's also a third, latent bug: if all scores in a list are equal, `hi - lo` is 0 and it divides by zero. A fixed version:

    ```python
    def normalize_all(score_lists):
        out = []
        for scores in score_lists:
            lo, hi = min(scores), max(scores)
            span = hi - lo
            out.append([0.0 if span == 0 else (s - lo) / span for s in scores])
        return out

    data = [[2, 4, 6], [5, 5, 5]]
    print(normalize_all(data))
    print(data)        # input unchanged
    ```

    ```text
    [[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]]
    [[2, 4, 6], [5, 5, 5]]
    ```

### Exercise 5: A streaming aggregation (hard)

Using the `orders.csv` file from the "In practice" section, compute the total amount *per customer* and print the top 3 customers, using generators so that memory use doesn't grow with the number of rows. (Memory may grow with the number of distinct *customers*, which is unavoidable.) Then explain why sorting all orders by amount first would not be possible with constant memory.

??? success "Solution"

    ```python
    import csv
    from collections import defaultdict
    from pathlib import Path

    def read_rows(path):
        with path.open(newline="") as f:
            yield from csv.DictReader(f)

    totals = defaultdict(float)
    for row in read_rows(Path("orders.csv")):
        totals[row["customer"]] += float(row["amount"])

    top3 = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:3]
    for customer, amount in top3:
        print(f"{customer}: {amount:,.2f}")
    ```

    ```text
    c4069: 9,888.44
    c3943: 9,687.06
    c2590: 9,596.29
    ```

    Memory is $O(\text{customers})$, not $O(\text{rows})$: one running total per customer. Sorting all *orders* would require seeing every row before outputting the first one, so you'd have to hold all of them (or use an external, disk-based sort). Aggregations like sums, counts, and maxima can be computed in one streaming pass; global sorts and exact medians can't.

## Check yourself

1. Why does `b = a` followed by `b.append(1)` change `a`?

    ??? note "Answer"

        Assignment doesn't copy. `a` and `b` are two names for the same list object, so changing it through either name changes the one object.

2. Why is `x in my_set` fast and `x in my_list` slow?

    ??? note "Answer"

        Sets are hash tables: the hash of `x` points directly to where it would be stored, so the check takes constant time on average. A list must be scanned element by element, so the check grows with the list's length.

3. Why must dict keys be immutable?

    ??? note "Answer"

        A dict finds a key by its hash. If a key could change after insertion, its hash would change, and the dict could no longer find it.

4. What's the main advantage of a generator over a list, and its main limitation?

    ??? note "Answer"

        It produces items one at a time, so memory use is constant regardless of how many items there are. The limitation is that it can be consumed only once, and you can't index into it or get its length without consuming it.

5. Why is `def f(x, items=[])` dangerous?

    ??? note "Answer"

        The default list is created once, when the function is defined, and shared by every call that uses the default. Changes made in one call persist into the next.

6. How should you compare two floats for equality?

    ??? note "Answer"

        With a tolerance, using `math.isclose` (or `np.isclose`/`np.allclose` for arrays), because floating-point arithmetic introduces tiny rounding errors.

7. What does the bare `*` in `def f(a, *, b=1)` do, and why is it useful?

    ??? note "Answer"

        It makes `b` keyword-only, so callers must write `f(1, b=2)`. This prevents ambiguous calls with unlabeled positional values and lets the function's author reorder or add options safely.

## Key takeaways

- Variables are names for objects. Assignment never copies, so watch for aliasing with mutable objects.
- Use sets and dicts for membership and lookup ($O(1)$), and lists for ordered sequences. The wrong choice can make code thousands of times slower.
- Comprehensions are concise and fast. Generators process data one item at a time in constant memory.
- Use dataclasses for data-holding classes, type hints and keyword-only arguments for reusable functions, and `pathlib` plus `with` for files.
- Know the silent traps: mutable defaults, float equality, modifying while iterating, late-binding closures, and banker's rounding.

## Further reading

- The official Python tutorial, docs.python.org/3/tutorial, especially the sections on data structures and classes.
- *Fluent Python* by Luciano Ramalho. A deep, practical guide to idiomatic Python.
- "Python Data Model" in the Python Language Reference, for what's really going on with objects and names.

## Next

Meet the array library that nearly all numerical Python is built on: [NumPy](04-numpy.md).
