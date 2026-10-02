"""Run every Python code block in a chapter and check its shown output.

Blocks run in order in one shared namespace, like a notebook. When a
```python block is followed by a ```text block, that text block is the
expected output. With --fix, expected outputs are replaced by the real ones.

Figures: each plt.show() saves the current figure as
docs/assets/figures/<level-dir>/<chapter-stem>-fig<N>.png, numbered in order.

A block preceded by the line <!-- skip-run --> is not executed. Use it for
illustrative blocks, such as ones that download data. A block meant to raise
an error passes when its shown output ends with the same exception line.

Usage:
    python scripts/run_chapter.py docs/chapters/00-python-for-data/04-numpy.md [--fix]
"""

import contextlib
import io
import os
import re
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FENCE = re.compile(r"^(?P<indent>[ \t]*)```(?P<lang>[\w+-]*)\s*$")


def parse_blocks(lines):
    """Return a list of fenced blocks: dicts with lang, indent, start, end."""
    blocks, i = [], 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if m and m.group("lang"):
            indent = m.group("indent")
            j = i + 1
            while j < len(lines) and lines[j].rstrip() != indent + "```":
                j += 1
            blocks.append({"lang": m.group("lang"), "indent": indent, "start": i, "end": j})
            i = j + 1
        else:
            i += 1
    return blocks


def body(lines, b):
    n = len(b["indent"])
    return "\n".join(l[n:] if l.startswith(b["indent"]) else l.lstrip() for l in lines[b["start"] + 1 : b["end"]])


def normalize(text):
    return "\n".join(l.rstrip() for l in text.strip("\n").splitlines())


def main():
    path = Path(sys.argv[1]).resolve()
    fix = "--fix" in sys.argv
    lines = path.read_text().splitlines()
    blocks = parse_blocks(lines)

    level_dir = path.parent.name
    docs_dir = next(p for p in path.parents if p.name == "docs")
    fig_dir = docs_dir / "assets" / "figures" / level_dir
    fig_count = [0]

    def show(*args, **kwargs):
        fig_count[0] += 1
        fig_dir.mkdir(parents=True, exist_ok=True)
        out = fig_dir / f"{path.stem}-fig{fig_count[0]}.png"
        plt.gcf().savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
        plt.close("all")
        print(f"[figure saved: {out.name}]", file=sys.__stderr__)

    plt.show = show
    os.chdir(Path(os.environ.get("RUN_DIR", "/tmp")))
    ns = {"__name__": "__main__"}
    replacements, problems, ran = [], 0, 0

    for k, b in enumerate(blocks):
        if b["lang"] != "python":
            continue
        src = body(lines, b)
        prev = [l.strip() for l in lines[: b["start"]] if l.strip()]
        if prev and prev[-1] == "<!-- skip-run -->":
            continue
        nxt = blocks[k + 1] if k + 1 < len(blocks) else None
        expect = None
        if nxt and nxt["lang"] == "text" and all(not l.strip() for l in lines[b["end"] + 1 : nxt["start"]]):
            expect = nxt
        buf = io.StringIO()
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(src, f"{path.name}:{b['start'] + 1}", "exec"), ns)
            ok = True
        except Exception:
            ok = False
            tb = traceback.format_exc().strip().splitlines()
            shown = normalize(body(lines, expect)).splitlines() if expect else []
            if shown and shown[-1].strip() == tb[-1].strip():
                ran += 1
                continue
            problems += 1
            print(f"\n✗ ERROR in block at line {b['start'] + 1}:\n  " + "\n  ".join(tb[-4:]))
        ran += 1
        dt = time.time() - t0
        if dt > 20:
            print(f"  (slow block at line {b['start'] + 1}: {dt:.0f}s)")
        got = buf.getvalue()
        if not ok or expect is None:
            continue
        want = body(lines, expect)
        if normalize(want) != normalize(got):
            problems += 1
            if not normalize(got):
                print(f"\n! block at line {b['start'] + 1} printed nothing, but a text block follows")
                continue
            print(f"\n≠ output differs at line {expect['start'] + 1}:")
            print("  --- shown ---\n  " + "\n  ".join(normalize(want).splitlines()[:12]))
            print("  --- actual ---\n  " + "\n  ".join(normalize(got).splitlines()[:12]))
            replacements.append((expect, got))

    if fix and replacements:
        for expect, got in sorted(replacements, key=lambda r: -r[0]["start"]):
            ind = expect["indent"]
            new = [ind + l if l else l for l in normalize(got).splitlines()]
            lines[expect["start"] + 1 : expect["end"]] = new
        path.write_text("\n".join(lines) + "\n")
        print(f"\nFixed {len(replacements)} output block(s).")
    print(f"\n{path.name}: ran {ran} blocks, {problems} issue(s), {fig_count[0]} figure(s).")


if __name__ == "__main__":
    main()
