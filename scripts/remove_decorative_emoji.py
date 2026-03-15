#!/usr/bin/env python3
"""
Remove decorative emoji from markdown headings ONLY.
Does NOT touch code blocks.
"""

import re
from pathlib import Path

DOCS_BASE = Path("data/architecture-docs")

DECORATIVE = {'🎯', '📚', '🏗️', '🏗', '🌐', '🏁', '🚀', '💡', '🧠', '📍', '🔄', '💾', '🧱'}


def is_code_fence(line: str) -> bool:
    return bool(re.match(r'^(`{3,}|~{3,})', line.strip()))


def process_file(md_path: Path) -> int:
    text = md_path.read_text()
    lines = text.split('\n')
    new_lines = []
    in_code = False
    removed_total = 0

    for line in lines:
        stripped = line.strip()

        # Track code fence state
        if is_code_fence(line):
            in_code = not in_code
            new_lines.append(line)
            continue

        # Only process headings OUTSIDE code blocks
        if not in_code and stripped.startswith('#'):
            new_line = line
            for emoji in DECORATIVE:
                # Remove emoji from heading
                new_line = new_line.replace(emoji, '')
            # Clean up double spaces
            new_line = re.sub(r'  +', ' ', new_line)
            # Ensure space after #
            new_line = re.sub(r'^(#{1,6})([^#\s])', r'\1 \2', new_line)

            removed_total += len(line) - len(new_line)
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    new_text = '\n'.join(new_lines)

    if removed_total > 0:
        md_path.write_text(new_text)

    return removed_total


def main():
    total = 0
    modified = 0

    for proj_dir in sorted(DOCS_BASE.iterdir()):
        if not proj_dir.is_dir():
            continue
        index_md = proj_dir / "index.md"
        if not index_md.exists():
            continue

        removed = process_file(index_md)
        if removed > 0:
            modified += 1
            total += removed
            print(f"  [{proj_dir.name}] -{removed} chars")

    print(f"\nTotal: {total} chars removed from {modified} files")


if __name__ == "__main__":
    main()
