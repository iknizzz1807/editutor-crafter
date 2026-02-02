#!/usr/bin/env python3
"""
Phase 3: Validate standardized YAML data.

Checks:
- All milestones have concepts (>=2), skills (>=2), pitfalls (>=2)
- All projects have id, domain_id, essence, why_important, learning_outcomes, tags
- languages always dict format, estimated_hours always string
- No category or why_expert fields remain
- All milestones have order, project_id
- Resource types in canonical set
- All domain stubs have description
- difficulty_score present

Usage:
    python3 scripts/validate-standardized-data.py
    python3 scripts/validate-standardized-data.py --strict  # treat warnings as errors
"""

import argparse
import sys
from pathlib import Path

import yaml

YAML_PATH = Path(__file__).resolve().parent.parent / "data" / "projects.yaml"

CANONICAL_RESOURCE_TYPES = {
    "article", "tutorial", "documentation", "paper", "video", "book",
    "tool", "specification", "guide", "reference", "course", "repository",
}


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate(data: dict, strict: bool = False) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings)."""
    errors = []
    warnings = []

    domains = data.get("domains", [])
    expert_projects = data.get("expert_projects", {})

    # ── Domain stubs ─────────────────────────────────────────────────────
    all_stub_ids = set()
    for domain in domains:
        for level, proj_list in domain.get("projects", {}).items():
            if not isinstance(proj_list, list):
                continue
            for proj in proj_list:
                pid = proj.get("id", "")
                all_stub_ids.add(pid)

                # Check descriptions
                if not proj.get("description"):
                    warnings.append(f"domain-stub [{pid}]: missing description")

                # Check languages format in stubs
                langs = proj.get("languages")
                if langs is not None and isinstance(langs, list):
                    errors.append(f"domain-stub [{pid}]: languages is flat list, should be dict")

    # ── Expert projects ──────────────────────────────────────────────────
    for pid, proj in expert_projects.items():
        prefix = f"project [{pid}]"

        # Required fields
        if not proj.get("id"):
            errors.append(f"{prefix}: missing id")
        if not proj.get("domain_id"):
            errors.append(f"{prefix}: missing domain_id")

        # Fields that should exist (warn if missing, Phase 2 may not have run yet)
        if not proj.get("essence"):
            warnings.append(f"{prefix}: missing essence")
        if not proj.get("why_important"):
            warnings.append(f"{prefix}: missing why_important")
        if not proj.get("learning_outcomes"):
            warnings.append(f"{prefix}: missing learning_outcomes")
        if not proj.get("tags"):
            warnings.append(f"{prefix}: missing tags")

        # Banned fields
        if "category" in proj:
            errors.append(f"{prefix}: still has 'category' field")
        if "why_expert" in proj:
            errors.append(f"{prefix}: still has 'why_expert' field")

        # estimated_hours type
        eh = proj.get("estimated_hours")
        if eh is not None and not isinstance(eh, str):
            errors.append(f"{prefix}: estimated_hours is {type(eh).__name__}, should be str")

        # languages format
        langs = proj.get("languages")
        if langs is not None and isinstance(langs, list):
            errors.append(f"{prefix}: languages is flat list, should be dict")
        if isinstance(langs, dict):
            if "recommended" not in langs:
                warnings.append(f"{prefix}: languages dict missing 'recommended' key")
            if "also_possible" not in langs:
                warnings.append(f"{prefix}: languages dict missing 'also_possible' key")

        # difficulty_score
        if "difficulty_score" not in proj:
            warnings.append(f"{prefix}: missing difficulty_score")

        # Resource types
        for res in proj.get("resources", []):
            rtype = res.get("type", "")
            if rtype and rtype not in CANONICAL_RESOURCE_TYPES:
                warnings.append(f"{prefix}: resource type '{rtype}' not canonical")

        # ── Milestones ───────────────────────────────────────────────────
        milestones = proj.get("milestones", [])
        for ms_idx, ms in enumerate(milestones):
            ms_prefix = f"{prefix} milestone[{ms_idx}]"

            # Required metadata
            if "order" not in ms:
                errors.append(f"{ms_prefix}: missing order")
            if "project_id" not in ms:
                errors.append(f"{ms_prefix}: missing project_id")

            # estimated_hours type
            ms_eh = ms.get("estimated_hours")
            if ms_eh is not None and not isinstance(ms_eh, str):
                errors.append(f"{ms_prefix}: estimated_hours is {type(ms_eh).__name__}")

            # Content fields (warn if < MIN, Phase 2 might not have run)
            for field, min_count in [("concepts", 2), ("skills", 2), ("pitfalls", 2)]:
                val = ms.get(field, [])
                if not val or not isinstance(val, list):
                    warnings.append(f"{ms_prefix}: missing {field}")
                elif len(val) < min_count:
                    warnings.append(f"{ms_prefix}: {field} has {len(val)} items (want >={min_count})")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate standardized YAML data")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = parser.parse_args()

    print(f"Loading {YAML_PATH}...")
    data = load_yaml(YAML_PATH)

    errors, warnings = validate(data, strict=args.strict)

    if warnings:
        print(f"\n⚠ {len(warnings)} warnings:")
        for w in warnings[:50]:
            print(f"  WARN: {w}")
        if len(warnings) > 50:
            print(f"  ... and {len(warnings) - 50} more warnings")

    if errors:
        print(f"\n✗ {len(errors)} errors:")
        for e in errors[:50]:
            print(f"  ERR: {e}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more errors")

    total_issues = len(errors) + (len(warnings) if args.strict else 0)

    if total_issues == 0:
        print(f"\n✓ Validation passed! ({len(warnings)} warnings)")
    else:
        print(f"\n✗ Validation failed: {len(errors)} errors, {len(warnings)} warnings")

    # Summary stats
    ep = data.get("expert_projects", {})
    total_ms = sum(len(p.get("milestones", [])) for p in ep.values())
    ms_with_concepts = sum(1 for p in ep.values() for ms in p.get("milestones", [])
                          if ms.get("concepts") and len(ms.get("concepts", [])) >= 2)
    ms_with_skills = sum(1 for p in ep.values() for ms in p.get("milestones", [])
                        if ms.get("skills") and len(ms.get("skills", [])) >= 2)
    ms_with_pitfalls = sum(1 for p in ep.values() for ms in p.get("milestones", [])
                          if ms.get("pitfalls") and len(ms.get("pitfalls", [])) >= 2)

    print(f"\n--- Stats ---")
    print(f"Expert projects: {len(ep)}")
    print(f"Total milestones: {total_ms}")
    print(f"Milestones with concepts (>=2): {ms_with_concepts}/{total_ms}")
    print(f"Milestones with skills (>=2): {ms_with_skills}/{total_ms}")
    print(f"Milestones with pitfalls (>=2): {ms_with_pitfalls}/{total_ms}")

    sys.exit(1 if total_issues > 0 else 0)


if __name__ == "__main__":
    main()
