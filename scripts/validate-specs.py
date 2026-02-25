#!/usr/bin/env python3
"""
Validate project specs from projects_data/.

Checks:
- All projects have required fields (id, name, domain, difficulty, description)
- All milestones have acceptance_criteria, concepts, skills
- Acceptance criteria are measurable (not vague)
- estimated_hours is string format
- No deprecated fields

Usage:
    python3 scripts/validate-specs.py
    python3 scripts/validate-specs.py --strict  # treat warnings as errors
    python3 scripts/validate-specs.py --project build-react
"""

import argparse
import re
import sys
import yaml
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / ".." / "data"
PROJECTS_DATA_DIR = DATA_DIR / "projects_data"
MAPPING_FILE = DATA_DIR / "project_domain_mapping.json"

VALID_DOMAINS = {
    "ai-ml", "app-dev", "compilers", "data-storage", "distributed",
    "game-dev", "security", "software-engineering", "specialized",
    "systems", "world-scale", "performance-engineering"
}

VALID_LEVELS = {"beginner", "intermediate", "advanced", "expert"}

# Patterns that indicate vague/non-measurable criteria
VAGUE_PATTERNS = [
    r"\bproperly\b", r"\bcorrectly\b", r"\bappropriately\b",
    r"\bsuccessfully\b", r"\bwell\b", r"\bgood\b",
    r"\befficiently\b", r"\bcleanly\b", r"\bsafely\b",
]


def load_project(project_id: str) -> dict | None:
    """Load a single project YAML."""
    path = PROJECTS_DATA_DIR / f"{project_id}.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def list_all_projects() -> list[str]:
    """List all project IDs."""
    return [f.stem for f in PROJECTS_DATA_DIR.glob("*.yaml")]


def check_vague_criteria(text: str) -> list[str]:
    """Check if criteria contains vague words."""
    if not text:
        return []
    found = []
    text_lower = text.lower()
    for pattern in VAGUE_PATTERNS:
        if re.search(pattern, text_lower):
            found.append(re.search(pattern, text_lower).group())
    return found


def validate_project(project_id: str, strict: bool = False) -> tuple[list[str], list[str]]:
    """Validate a single project. Returns (errors, warnings)."""
    errors = []
    warnings = []
    prefix = f"[{project_id}]"

    data = load_project(project_id)
    if not data:
        errors.append(f"{prefix} File not found")
        return errors, warnings

    # === PROJECT-LEVEL CHECKS ===

    # Required fields
    if not data.get("id"):
        errors.append(f"{prefix} Missing 'id' field")
    elif data.get("id") != project_id:
        errors.append(f"{prefix} ID mismatch: file={project_id}, id={data.get('id')}")

    if not data.get("name"):
        errors.append(f"{prefix} Missing 'name' field")

    if not data.get("domain"):
        warnings.append(f"{prefix} Missing 'domain' field")
    elif data.get("domain") not in VALID_DOMAINS:
        warnings.append(f"{prefix} Invalid domain: {data.get('domain')}")

    difficulty = data.get("difficulty", data.get("level"))
    if not difficulty:
        warnings.append(f"{prefix} Missing 'difficulty' field")
    elif difficulty not in VALID_LEVELS:
        warnings.append(f"{prefix} Non-standard difficulty: {difficulty}")

    if not data.get("description") and not data.get("essence"):
        warnings.append(f"{prefix} Missing 'description' or 'essence'")

    # estimated_hours should be string
    eh = data.get("estimated_hours")
    if eh is not None and not isinstance(eh, str):
        errors.append(f"{prefix} estimated_hours should be string, got {type(eh).__name__}")

    # Check for deprecated fields
    if "category" in data:
        errors.append(f"{prefix} Deprecated field 'category' found")
    if "why_expert" in data:
        errors.append(f"{prefix} Deprecated field 'why_expert' found")

    # === MILESTONE CHECKS ===
    milestones = data.get("milestones", [])
    if not milestones:
        warnings.append(f"{prefix} No milestones defined")
        return errors, warnings

    for i, ms in enumerate(milestones):
        ms_prefix = f"{prefix} milestone[{i}]"
        ms_id = ms.get("id", f"ms-{i}")

        # Required milestone fields
        if not ms.get("title"):
            warnings.append(f"{ms_prefix} Missing title")

        if not ms.get("description") and not ms.get("summary"):
            warnings.append(f"{ms_prefix} Missing description/summary")

        # Acceptance criteria
        ac = ms.get("acceptance_criteria", [])
        if not ac:
            warnings.append(f"{ms_prefix} No acceptance_criteria")
        else:
            if isinstance(ac, str):
                ac = [ac]
            for j, criterion in enumerate(ac):
                if not criterion or not isinstance(criterion, str):
                    warnings.append(f"{ms_prefix} AC[{j}] is empty or invalid")
                    continue

                # Check for vague criteria
                vague = check_vague_criteria(criterion)
                if vague:
                    warnings.append(f"{ms_prefix} AC[{j}] contains vague words: {vague}")

                # Check minimum length
                if len(criterion) < 20:
                    warnings.append(f"{ms_prefix} AC[{j}] too short: '{criterion[:50]}...'")

        # Concepts
        concepts = ms.get("concepts", [])
        if not concepts:
            warnings.append(f"{ms_prefix} No concepts defined")
        elif len(concepts) < 2:
            warnings.append(f"{ms_prefix} Only {len(concepts)} concepts (recommend >= 2)")

        # Skills
        skills = ms.get("skills", [])
        if not skills:
            warnings.append(f"{ms_prefix} No skills defined")
        elif len(skills) < 2:
            warnings.append(f"{ms_prefix} Only {len(skills)} skills (recommend >= 2)")

        # Common pitfalls
        pitfalls = ms.get("common_pitfalls", ms.get("pitfalls", []))
        if not pitfalls:
            warnings.append(f"{ms_prefix} No common_pitfalls defined")

        # estimated_hours type in milestone
        ms_eh = ms.get("estimated_hours")
        if ms_eh is not None and not isinstance(ms_eh, str):
            errors.append(f"{ms_prefix} estimated_hours should be string")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate project specs")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    parser.add_argument("--project", help="Validate specific project")
    parser.add_argument("--domain", help="Filter by domain")
    parser.add_argument("--level", help="Filter by level")
    args = parser.parse_args()

    # Get projects to validate
    if args.project:
        projects = [args.project]
    else:
        projects = list_all_projects()

    all_errors = []
    all_warnings = []

    print(f"Validating {len(projects)} projects from {PROJECTS_DATA_DIR}...")
    print()

    for pid in sorted(projects):
        data = load_project(pid)

        # Filter by domain/level if specified
        if args.domain and data and data.get("domain") != args.domain:
            continue
        if args.level and data:
            diff = data.get("difficulty", data.get("level"))
            if diff != args.level:
                continue

        errors, warnings = validate_project(pid, args.strict)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

        # Print per-project summary
        issue_count = len(errors) + (len(warnings) if args.strict else 0)
        if issue_count > 0:
            status = "✗" if errors else "⚠"
            print(f"{status} {pid}: {len(errors)} errors, {len(warnings)} warnings")
        else:
            print(f"✓ {pid}")

    # Summary
    print()
    print("=" * 50)

    if all_warnings:
        print(f"\n⚠ {len(all_warnings)} WARNINGS:")
        for w in all_warnings[:30]:
            print(f"  {w}")
        if len(all_warnings) > 30:
            print(f"  ... and {len(all_warnings) - 30} more")

    if all_errors:
        print(f"\n✗ {len(all_errors)} ERRORS:")
        for e in all_errors[:30]:
            print(f"  {e}")
        if len(all_errors) > 30:
            print(f"  ... and {len(all_errors) - 30} more")

    # Stats
    print(f"\n--- STATS ---")
    print(f"Projects validated: {len(projects)}")

    # Count by domain/level
    by_domain = defaultdict(int)
    by_level = defaultdict(int)
    for pid in projects:
        data = load_project(pid)
        if data:
            by_domain[data.get("domain", "unknown")] += 1
            by_level[data.get("difficulty", data.get("level", "unknown"))] += 1

    print(f"\nBy Domain:")
    for d, c in sorted(by_domain.items()):
        print(f"  {d}: {c}")
    print(f"\nBy Level:")
    for l, c in sorted(by_level.items()):
        print(f"  {l}: {c}")

    # Exit code
    total_issues = len(all_errors) + (len(all_warnings) if args.strict else 0)
    if total_issues == 0:
        print(f"\n✓ ALL VALIDATIONS PASSED ({len(all_warnings)} warnings)")
        sys.exit(0)
    else:
        print(f"\n✗ VALIDATION FAILED: {len(all_errors)} errors, {len(all_warnings)} warnings")
        sys.exit(1)


if __name__ == "__main__":
    main()
