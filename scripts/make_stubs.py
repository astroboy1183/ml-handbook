"""Create "coming soon" pages for roadmap chapters and capstones not yet written.

Reads docs/roadmap.md, and for every linked chapter or capstone file that
doesn't exist, writes a short placeholder listing what the page will cover.
Placeholders carry a marker line so they can be found and replaced later.

Usage:
    python scripts/make_stubs.py           # create missing placeholders
    python scripts/make_stubs.py --list    # list current placeholders
"""

import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
MARKER = "<!-- stub: coming soon -->"


def main():
    roadmap = (DOCS / "roadmap.md").read_text()
    if "--list" in sys.argv:
        for p in sorted(DOCS.rglob("*.md")):
            if MARKER in p.read_text():
                print(p.relative_to(DOCS))
        return

    created = 0
    for level in re.split(r"\n### ", roadmap)[1:]:
        title = level.split("\n", 1)[0]
        for name, path, concepts in re.findall(r"^\| \[([^\]]+)\]\(([^)]+)\) \| (.+?) \|$", level, re.M):
            created += write_stub(path, name, title, concepts.split(" · "))
        cap = re.search(r"\*\*Capstone:\*\* \[([^\]]+)\]\(([^)]+)\)", level)
        if cap:
            created += write_stub(cap.group(2), f"Capstone: {cap.group(1)}", title, [])
            solution = cap.group(2).replace("exercises/", "exercises/solutions/")
            created += write_stub(solution, f"Solution: {cap.group(1)}", title, [])
        index = re.search(r"\]\((chapters/[^/]+)/", level)
        if index:
            created += write_stub(index.group(1) + "/index.md", title, title, [])
    print(f"created {created} placeholder page(s)")


def write_stub(rel_path, name, level_title, concepts):
    path = DOCS / rel_path
    if path.exists():
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    depth = len(Path(rel_path).parts) - 1
    up = "../" * depth
    lines = [f"# {name}", "", MARKER, "", f"> **{level_title}**", "",
             "!!! info \"Coming soon\"",
             f"    This page is being written. See the [full roadmap]({up}roadmap.md) for the whole syllabus.", ""]
    if concepts:
        lines += ["## What this chapter will cover", ""] + [f"- {c.strip()}" for c in concepts] + [""]
    path.write_text("\n".join(lines))
    return 1


if __name__ == "__main__":
    main()
