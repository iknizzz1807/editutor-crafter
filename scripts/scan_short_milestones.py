#!/usr/bin/env python3
"""
Scan all index.md files and find milestones with < 10000 chars content.
Handles duplicate MS_ID markers correctly.
"""

import re
from pathlib import Path

DOCS_BASE = Path("data/architecture-docs")


def extract_milestones(text):
    """Extract all milestones with their content lengths."""
    # Find all MS_ID markers with positions
    all_markers = [(m.group(1), m.start()) for m in re.finditer(r'<!-- MS_ID: ([\w-]+) -->', text)]

    # Get unique ms_ids in order of first appearance
    seen = set()
    unique_markers = []
    for ms_id, pos in all_markers:
        if ms_id not in seen:
            seen.add(ms_id)
            unique_markers.append((ms_id, pos))

    milestones = []
    for i, (ms_id, start_pos) in enumerate(unique_markers):
        # Find content start (skip past the marker line)
        content_start = text.find('\n', start_pos) + 1

        # Find content end (next unique milestone or end of content)
        if i + 1 < len(unique_markers):
            next_pos = unique_markers[i + 1][1]
        else:
            # Last milestone - find END_MS or end of file
            end_match = re.search(r'<!-- END_MS -->', text[content_start:])
            if end_match:
                next_pos = content_start + end_match.start()
            else:
                next_pos = len(text)

        content = text[content_start:next_pos].strip()

        # Check if content starts with CRITERIA_JSON (empty content)
        is_empty_criteria = content.startswith('[[CRITERIA_JSON:')

        milestones.append({
            'ms_id': ms_id,
            'content_len': len(content),
            'is_empty_criteria': is_empty_criteria,
            'preview': content[:100]
        })

    return milestones


def main():
    issues = []

    for proj_dir in sorted(DOCS_BASE.iterdir()):
        if not proj_dir.is_dir():
            continue

        index_md = proj_dir / "index.md"
        if not index_md.exists():
            continue

        text = index_md.read_text()
        milestones = extract_milestones(text)

        for ms in milestones:
            # Check for short content (< 10000 chars)
            if ms['content_len'] < 10000:
                issues.append({
                    'project': proj_dir.name,
                    'ms_id': ms['ms_id'],
                    'len': ms['content_len'],
                    'is_criteria': ms['is_empty_criteria'],
                    'preview': ms['preview'][:80]
                })

    # Sort by content length
    issues.sort(key=lambda x: x['len'])

    print(f"Found {len(issues)} milestones with < 10000 chars:\n")
    for issue in issues:
        marker = "⚠️ CRITERIA_ONLY" if issue['is_criteria'] else ""
        print(f"[{issue['project']}] {issue['ms_id']}: {issue['len']} chars {marker}")
        print(f"  preview: {issue['preview'][:60]}...")
        print()


if __name__ == "__main__":
    main()
