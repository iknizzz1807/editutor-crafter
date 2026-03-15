#!/usr/bin/env python3
"""
Fix headings without space after # (artifact from emoji removal)
"""

import re
from pathlib import Path

DOCS_BASE = Path("/home/ikniz/Work/Coding/AI_MachineLearning/editutor-crafter/data/architecture-docs")


def process_file(md_path: Path) -> int:
    text = md_path.read_text()
    # Fix: "#Something" -> "# Something"
    new_text = re.sub(r'^(#{1,6})([^#\s])', r'\1 \2', text, flags=re.MULTILINE)

    removed = len(text) - len(new_text)
    if removed > 0:
        md_path.write_text(new_text)
    return removed


def main():
    projects = sorted(DOCS_BASE.iterdir())
    total = 0
    modified = 0

    for proj_dir in projects:
        if not proj_dir.is_dir():
            continue
        index_md = proj_dir / "index.md"
        if not index_md.exists():
            continue

        fixed = process_file(index_md)
        if fixed > 0:
            modified += 1
            total += fixed
            print(f"  [{proj_dir.name}] fixed {fixed} chars")

    print(f"\nTotal: {total} chars fixed in {modified} files")


if __name__ == "__main__":
    main()
