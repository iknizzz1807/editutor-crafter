#!/usr/bin/env python3
"""
Fix orphan SVGs: inject diagram refs into the most relevant safe position
in the corresponding milestone/module section of index.md.

Safe positions = between paragraphs (blank lines), never inside code blocks,
never in the middle of a sentence.
"""

import json
import re
import sys
from pathlib import Path

DOCS_BASE = Path("/home/ikniz/Work/Coding/AI_MachineLearning/editutor-crafter/data/architecture-docs")


def extract_json_from_block(block: str) -> dict | None:
    start = block.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(block[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(block[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None


def extract_blueprint_from_log(log_path: Path) -> dict | None:
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return None
    match = re.search(r"\[RESPONSE\].*?NODE: Architect\n(.*?)(?=\n={60,}|\Z)", text, re.DOTALL)
    if not match:
        return None
    return extract_json_from_block(match.group(1))


def extract_tdd_blueprint_from_log(log_path: Path) -> dict | None:
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return None
    match = re.search(r"\[RESPONSE\].*?NODE: TDD Planner\n(.*?)(?=\n={60,}|\Z)", text, re.DOTALL)
    if not match:
        return None
    return extract_json_from_block(match.group(1))


def parse_milestone_sections(md_text: str) -> list[dict]:
    markers = list(re.finditer(r"<!-- MS_ID: ([\w-]+) -->", md_text))
    sections = []
    for i, m in enumerate(markers):
        start = m.start()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(md_text)
        sections.append({"ms_id": m.group(1), "start": start, "end": end})
    return sections


def get_orphan_svgs(proj_dir: Path, md_text: str) -> list[dict]:
    diag_dir = proj_dir / "diagrams"
    if not diag_dir.exists():
        return []
    orphans = []
    for svg in sorted(diag_dir.glob("*.svg")):
        diag_id = svg.stem
        if f"./diagrams/{diag_id}.svg" not in md_text and f"diagrams/{diag_id}.svg" not in md_text:
            orphans.append({"diag_id": diag_id, "svg_ref": f"./diagrams/{diag_id}.svg"})
    for item in sorted(diag_dir.iterdir()):
        if item.is_dir() and (item / "index.svg").exists():
            diag_id = item.name
            if f"./diagrams/{diag_id}/index.svg" not in md_text and f"./diagrams/{diag_id}.svg" not in md_text:
                orphans.append({"diag_id": diag_id, "svg_ref": f"./diagrams/{diag_id}/index.svg"})
    return orphans


def fix_broken_refs(md_text: str, proj_dir: Path) -> tuple[str, int]:
    diag_dir = proj_dir / "diagrams"
    fixes = 0
    def replacer(m):
        nonlocal fixes
        diag_id = m.group(1)
        if not (diag_dir / f"{diag_id}.svg").exists() and (diag_dir / diag_id / "index.svg").exists():
            fixes += 1
            return f"./diagrams/{diag_id}/index.svg"
        return m.group(0)
    md_text = re.sub(r"\./diagrams/([\w-]+)\.svg", replacer, md_text)
    return md_text, fixes


def is_code_fence(line: str) -> bool:
    """True if line is a fenced code block delimiter (``` or ~~~), not inline backticks."""
    s = line.strip()
    return bool(re.match(r'^(`{3,}|~{3,})', s))


def get_safe_paragraph_ends(section_text: str, section_start: int) -> list[int]:
    """
    Return list of absolute positions that are safe injection points:
    - Between paragraphs (after a blank line sequence)
    - NOT inside a fenced code block (``` or ~~~)
    - The preceding non-empty line ends with sentence-ending punctuation,
      a closing markdown element, or is a heading
    """
    lines = section_text.split("\n")
    in_code = False
    safe_ends = []

    # Build a list of (line_index, absolute_offset) for each line
    offsets = []
    pos = section_start
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1  # +1 for \n

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Track code fence state — only full-line fences, not inline backticks
        if is_code_fence(line):
            in_code = not in_code

        # A safe injection point: current line is blank, not in code block
        if not stripped and not in_code and i > 0:
            # Find the previous non-blank line
            prev_i = i - 1
            while prev_i >= 0 and not lines[prev_i].strip():
                prev_i -= 1

            if prev_i >= 0:
                prev = lines[prev_i].strip()
                # Safe if previous line ends with sentence punctuation,
                # closing code fence, closing list item, table row, heading, or blockquote
                is_safe = (
                    prev.endswith((".", "!", "?", ":", "```", "~~~"))
                    or prev.startswith("#")
                    or prev.startswith("|")
                    or prev.startswith(">")
                    or re.match(r"^[-*+]\s", prev)
                    or re.match(r"^\d+\.\s", prev)
                    or prev.endswith("**")
                    or prev.endswith("*")
                    or prev.endswith("`")
                    or prev.endswith(")")
                    or prev.endswith("]")
                )
                if is_safe:
                    # The injection point is after the blank line sequence
                    # Find end of blank line run
                    j = i
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    # Position = start of the next non-blank line (inject before it)
                    if j < len(lines):
                        safe_ends.append(offsets[j])
                    else:
                        safe_ends.append(pos)
        i += 1

    return safe_ends


STOP_WORDS = {
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are",
    "that", "this", "with", "from", "by", "on", "at", "be", "it", "as",
    "how", "what", "when", "where", "which", "who", "show", "shows",
    "using", "used", "use", "can", "will", "its", "their", "vs", "via",
    "each", "all", "both", "these", "those", "between", "into", "about",
}


def score_block_for_diagram(block: str, diag: dict) -> int:
    title_words = set(re.findall(r'\w+', (diag.get("title") or "").lower()))
    desc_words = set(re.findall(r'\w+', (diag.get("description") or "").lower()))
    keywords = (title_words | desc_words) - STOP_WORDS
    block_lower = block.lower()
    block_words = set(re.findall(r'\w+', block_lower))
    score = 0
    for kw in keywords:
        if len(kw) < 3:
            continue
        if kw in block_words:
            score += 2
        elif kw in block_lower:
            score += 1
    return score


def find_best_safe_position(section_text: str, section_start: int, diag: dict,
                             used_positions: set, has_unbalanced_fences: bool) -> int:
    """
    Find a safe position to inject.
    If doc has unbalanced fences (broken structure), inject at end of section only.
    Otherwise, find the best safe paragraph boundary.
    """
    section_end = section_start + len(section_text)

    if has_unbalanced_fences:
        # Safe: just before the end of the section, staggered by used count
        pos = section_end - len(used_positions)
        used_positions.add(pos)
        return section_end

    safe_points = get_safe_paragraph_ends(section_text, section_start)

    if not safe_points:
        return section_end

    # Score the text block BEFORE each safe point
    scored = []
    prev = section_start
    for sp in safe_points:
        block = section_text[prev - section_start:sp - section_start]
        score = score_block_for_diagram(block, diag)
        scored.append((score, sp))
        prev = sp

    scored.sort(key=lambda x: -x[0])

    for score, pos in scored:
        if pos not in used_positions:
            used_positions.add(pos)
            return pos

    pos = safe_points[-1]
    used_positions.add(pos)
    return pos


def process_project(proj_dir: Path, verbose: bool = False) -> dict:
    stats = {"injected": 0, "broken_fixed": 0, "skipped": 0, "errors": []}

    index_md = proj_dir / "index.md"
    log_path = proj_dir / "llm_traces.log"

    if not index_md.exists():
        stats["errors"].append("no index.md")
        return stats

    md_text = index_md.read_text()

    md_text, broken_fixed = fix_broken_refs(md_text, proj_dir)
    stats["broken_fixed"] = broken_fixed

    blueprint = extract_blueprint_from_log(log_path) if log_path.exists() else None
    tdd_blueprint = extract_tdd_blueprint_from_log(log_path) if log_path.exists() else None

    diag_info = {}
    anchor_to_ms_id = {}
    if blueprint:
        for ms in blueprint.get("milestones", []):
            ms_id = ms.get("id")
            anchor_id = ms.get("anchor_id")
            if ms_id and anchor_id:
                anchor_to_ms_id[anchor_id] = ms_id
        for d in blueprint.get("diagrams", []):
            if isinstance(d, dict) and d.get("id"):
                diag_info[d["id"]] = d

    tdd_diag_to_module = {}
    if tdd_blueprint:
        for mod in tdd_blueprint.get("modules", []):
            mod_id = mod.get("id")
            for d in mod.get("diagrams", []):
                if isinstance(d, dict) and d.get("id"):
                    tdd_diag_to_module[d["id"]] = mod_id
                    diag_info[d["id"]] = d

    sections = parse_milestone_sections(md_text)
    ms_id_to_section = {s["ms_id"]: s for s in sections}

    # Detect unbalanced code fences (broken doc structure from pipeline)
    fence_count = sum(1 for line in md_text.split("\n") if re.match(r'^\s*(```|~~~)', line))
    has_unbalanced_fences = (fence_count % 2 != 0)
    if has_unbalanced_fences:
        stats["errors"].append(f"unbalanced code fences ({fence_count}) — skipping inject")
        if broken_fixed > 0:
            index_md.write_text(md_text)
        return stats

    orphans = get_orphan_svgs(proj_dir, md_text)
    if not orphans and broken_fixed == 0:
        return stats

    injections = []
    section_used: dict[str, set] = {}

    for orphan in orphans:
        diag_id = orphan["diag_id"]
        svg_ref = orphan["svg_ref"]
        diag = diag_info.get(diag_id, {})
        title = diag.get("title") or diag_id

        section = None
        anchor_target = diag.get("anchor_target", "")
        if anchor_target:
            ms_id = anchor_to_ms_id.get(anchor_target, anchor_target)
            section = ms_id_to_section.get(ms_id)

        if not section and diag_id in tdd_diag_to_module:
            section = ms_id_to_section.get(tdd_diag_to_module[diag_id])

        if not section:
            ms_num_match = re.search(r'(?:^|-)m(\d+)-', diag_id)
            if ms_num_match:
                ms_num = int(ms_num_match.group(1))
                if ms_num - 1 < len(sections):
                    section = sections[ms_num - 1]

        if not section and sections:
            section = sections[-1]

        if not section:
            stats["skipped"] += 1
            continue

        sec_id = section["ms_id"]
        used = section_used.setdefault(sec_id, set())
        sec_text = md_text[section["start"]:section["end"]]
        pos = find_best_safe_position(sec_text, section["start"], diag, used, has_unbalanced_fences)

        injections.append((pos, f"![{title}]({svg_ref})\n\n", diag_id))
        if verbose:
            print(f"    inject: {diag_id} → {sec_id}")

    # Apply from end to start
    injections.sort(key=lambda x: x[0], reverse=True)
    for pos, text, diag_id in injections:
        md_text = md_text[:pos] + text + md_text[pos:]
        stats["injected"] += 1

    if stats["injected"] > 0 or stats["broken_fixed"] > 0:
        index_md.write_text(md_text)

    return stats


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None
    projects = sorted(DOCS_BASE.iterdir())
    total_injected = 0
    total_broken = 0
    total_skipped = 0

    for proj_dir in projects:
        if not proj_dir.is_dir() or not (proj_dir / "index.md").exists():
            continue
        if target and proj_dir.name != target:
            continue

        print(f"\n[{proj_dir.name}]")
        stats = process_project(proj_dir, verbose=(target is not None))

        parts = []
        if stats["injected"]: parts.append(f"injected={stats['injected']}")
        if stats["broken_fixed"]: parts.append(f"broken_fixed={stats['broken_fixed']}")
        if stats["skipped"]: parts.append(f"skipped={stats['skipped']}")
        for e in stats["errors"]: parts.append(f"ERROR:{e}")
        print(f"  {' | '.join(parts) if parts else 'clean'}")

        total_injected += stats["injected"]
        total_broken += stats["broken_fixed"]
        total_skipped += stats["skipped"]

    print(f"\n{'='*50}")
    print(f"Total injected:    {total_injected}")
    print(f"Total broken fixed:{total_broken}")
    print(f"Total skipped:     {total_skipped}")


if __name__ == "__main__":
    main()
