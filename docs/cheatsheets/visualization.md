# Visualization Cheat Sheet

matplotlib and seaborn essentials. The full explanation is in [Data visualization](../chapters/00-python-for-data/06-visualization.md).

## Setup and the object model

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind")   # optional global style

fig, ax = plt.subplots(figsize=(7, 4))                      # one plot
fig, axes = plt.subplots(2, 3, figsize=(12, 6),             # a grid: axes[row, col]
                         sharex=True, sharey=True)
fig.tight_layout()                                          # fix overlapping labels
fig.savefig("chart.png", dpi=150, bbox_inches="tight")     # .svg/.pdf for vector
plt.show()
```

| pyplot (implicit) | Object-oriented (explicit) |
|---|---|
| `plt.plot(x, y)` | `ax.plot(x, y)` |
| `plt.title("t")` | `ax.set_title("t")` |
| `plt.xlabel("x")` | `ax.set_xlabel("x")` |
| `plt.xlim(0, 10)` | `ax.set_xlim(0, 10)` |
| `plt.xticks(...)` | `ax.set_xticks(...)` |
| `plt.legend()` | `ax.legend()` |

## Core matplotlib plots

| Plot | Call | Key options |
|---|---|---|
| Line | `ax.plot(x, y)` | `lw`, `ls="--"`, `marker="o"`, `label` |
| Scatter | `ax.scatter(x, y)` | `s` (size), `c` (color values), `alpha`, `cmap` |
| Bar | `ax.bar(cats, vals)` / `ax.barh(...)` | `width`, `color`, `yerr` |
| Histogram | `ax.hist(x, bins=30)` | `density=True`, `alpha`, `histtype="step"` |
| Box | `ax.boxplot([a, b])` | `labels`, `showfliers` |
| Image / matrix | `ax.imshow(M)` | `cmap`, `vmin`, `vmax`, `aspect` |
| Fill band | `ax.fill_between(x, lo, hi)` | `alpha` (for confidence bands) |
| Reference line | `ax.axhline(y)` / `ax.axvline(x)` | `color`, `ls` |
| Shaded span | `ax.axvspan(x0, x1)` | `alpha` |
| Text | `ax.text(x, y, "s")` / `ax.annotate("s", xy, xytext)` | `ha`, `va`, `arrowprops` |

## Styling details

```python
ax.set_title("Takeaway goes here", loc="left", fontweight="bold")
ax.set(xlabel="Distance (km)", ylabel="Time (min)", ylim=(0, None))
ax.spines[["top", "right"]].set_visible(False)       # remove chart junk
ax.set_xscale("log")                                  # log axis
ax.tick_params(axis="x", rotation=45)
ax.legend(frameon=False, loc="upper left")
ax.grid(axis="y", alpha=0.3)
ax.yaxis.set_major_formatter("{x:,.0f}")              # thousands separators
twin = ax.twinx()                                     # second y-axis (use sparingly)
```

## seaborn: axes-level functions

These draw on one Axes. Pass `ax=` to choose which.

| Purpose | Function | Example |
|---|---|---|
| Histogram / KDE | `histplot`, `kdeplot` | `sns.histplot(df, x="v", hue="g", kde=True, stat="density")` |
| Scatter | `scatterplot` | `sns.scatterplot(df, x="a", y="b", hue="g", size="n", alpha=.6)` |
| Line with CI | `lineplot` | `sns.lineplot(df, x="t", y="v", hue="g", errorbar=("ci", 95))` |
| Category means | `barplot`, `pointplot` | `sns.barplot(df, x="g", y="v", errorbar=("ci", 95))` |
| Counts | `countplot` | `sns.countplot(df, y="cat", order=df["cat"].value_counts().index)` |
| Distributions by group | `boxplot`, `violinplot`, `stripplot` | `sns.boxplot(df, x="g", y="v")` |
| Matrix | `heatmap` | `sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)` |
| Regression fit | `regplot` | `sns.regplot(df, x="a", y="b", ci=None)` |

## seaborn: figure-level functions (small multiples)

These create their own figure and return a `FacetGrid`. Use `col=`, `row=`, and `col_wrap=`.

| Function | Wraps | Example |
|---|---|---|
| `relplot` | scatter, line | `sns.relplot(df, x="a", y="b", col="g", kind="scatter")` |
| `displot` | hist, kde, ecdf | `sns.displot(df, x="v", col="g", row="h", bins=20)` |
| `catplot` | box, bar, violin, strip | `sns.catplot(df, x="g", y="v", col="h", kind="box")` |
| `lmplot` | regplot | `sns.lmplot(df, x="a", y="b", col="g", col_wrap=3)` |
| `pairplot` | all pairs | `sns.pairplot(df[cols + ["target"]], hue="target")` |

```python
g = sns.relplot(df, x="a", y="b", col="g", height=3, aspect=1.2)
g.set_titles("{col_name}")
g.set_axis_labels("A (units)", "B (units)")
g.figure.suptitle("Title", y=1.03)
```

## Which chart for which question

| Question | Chart |
|---|---|
| Change over time | line; add a rolling mean for noisy series |
| Compare categories | bar (horizontal for long labels), sorted, starting at 0 |
| One distribution | histogram (try several bin counts), KDE, ECDF |
| Distributions across groups | box, violin, overlaid density, small multiples |
| Relationship of two numbers | scatter (`alpha` for overplotting, `hexbin` for very many points) |
| Many pairwise relationships | pair plot, correlation heatmap |
| Parts of a whole | stacked bar, or a sorted bar of shares; pie only for 2–3 parts |
| Model results | predicted vs actual scatter, residual plot, confusion matrix heatmap |

## Color

| Data type | Colormap kind | Good choices |
|---|---|---|
| Ordered values | sequential | `viridis`, `cividis`, `rocket`, `Blues` |
| Values around a midpoint | diverging | `vlag`, `RdBu`, `coolwarm`, with `center=` |
| Categories | qualitative | `sns.color_palette("colorblind")`, `tab10` |

```python
palette = sns.color_palette("colorblind", 3)
sns.palplot(palette)                        # preview a palette
ax.scatter(x, y, c=values, cmap="viridis"); fig.colorbar(ax.collections[0], ax=ax)
```

Avoid `jet` and rainbow maps, and don't rely on red versus green alone. Add markers, line styles, or direct labels as a second cue.

## pandas plotting shortcuts

```python
df["col"].hist(bins=30)
df.hist(bins=30, figsize=(12, 8))                     # every numeric column
df["cat"].value_counts().plot.barh()
df.plot(x="date", y=["a", "b"], figsize=(10, 4))      # lines
df.plot.scatter(x="a", y="b", alpha=0.5)
df.boxplot(column="v", by="g")
```

All of these return a matplotlib Axes, so you can keep customizing.

## Plotly (interactive)

```python
import plotly.express as px

fig = px.scatter(df, x="a", y="b", color="g", hover_data=["id"], log_x=True)
fig = px.line(df, x="date", y="v", color="g")
fig = px.histogram(df, x="v", color="g", barmode="overlay", nbins=40)
fig.update_layout(title="Title", template="plotly_white")
fig.show()
fig.write_html("chart.html")
```

## Explanatory chart checklist

- [ ] The title states the takeaway, not just the variables.
- [ ] Axes are labeled, with units.
- [ ] Bars start at zero.
- [ ] Categories are sorted (unless they have a natural order).
- [ ] Lines are labeled directly instead of with a legend, where possible.
- [ ] The palette is colorblind-safe, with a second cue (markers or line styles).
- [ ] Clutter is removed: top and right spines, heavy grids, 3D effects.
- [ ] Text is large enough to read when the image is shrunk into a slide.
