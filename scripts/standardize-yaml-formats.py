#!/usr/bin/env python3
"""
Phase 1: Programmatic YAML fixes for data/projects.yaml.

Fixes applied sequentially:
1. estimated_hours → string (int/float → "15"/"4.5", keep range strings)
2. languages flat list → nested {recommended: [...], also_possible: []}
3. Resource type normalize: docs→documentation, spec→specification
4. Merge why_expert → why_important (if missing why_important, use why_expert, then delete)
5. Delete category field from expert_projects
6. Add missing order + project_id to milestones
7. Add missing top-level id to expert_projects
8. Add missing domain_id via reverse lookup
9. Add difficulty_score mapping
10. Copy descriptions from expert_projects to domain stubs
"""

import copy
import sys
from pathlib import Path

import yaml

# ── Preserve YAML formatting ────────────────────────────────────────────────

class LiteralStr(str):
    pass

class FlowList(list):
    pass

def literal_str_representer(dumper, data):
    if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

def flow_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(LiteralStr, literal_str_representer)
yaml.add_representer(FlowList, flow_list_representer)


# ── Config ───────────────────────────────────────────────────────────────────

RESOURCE_TYPE_MAP = {
    'docs': 'documentation',
    'spec': 'specification',
}

DIFFICULTY_SCORE_MAP = {
    'beginner': 2,
    'intermediate': 4,
    'advanced': 6,
    'expert': 8,
}

YAML_PATH = Path(__file__).resolve().parent.parent / 'data' / 'projects.yaml'


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict):
    """Save YAML with reasonable formatting."""
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(
            data, f,
            default_flow_style=False,
            allow_unicode=True,
            width=120,
            sort_keys=False,
        )


def build_domain_project_lookup(domains: list) -> dict:
    """Build reverse lookup: project_id → domain_id."""
    lookup = {}
    for domain in domains:
        domain_id = domain.get('id', '')
        projects = domain.get('projects', {})
        for level, proj_list in projects.items():
            if not isinstance(proj_list, list):
                continue
            for proj in proj_list:
                pid = proj.get('id', '')
                if pid:
                    lookup[pid] = domain_id
    return lookup


def build_project_difficulty_lookup(domains: list) -> dict:
    """Build reverse lookup: project_id → difficulty level."""
    lookup = {}
    for domain in domains:
        projects = domain.get('projects', {})
        for level, proj_list in projects.items():
            if not isinstance(proj_list, list):
                continue
            for proj in proj_list:
                pid = proj.get('id', '')
                if pid:
                    lookup[pid] = level
    return lookup


# ── Fix functions ────────────────────────────────────────────────────────────

def fix_estimated_hours(expert_projects: dict) -> int:
    """Fix 1: Convert estimated_hours to string."""
    count = 0
    for pid, proj in expert_projects.items():
        if 'estimated_hours' in proj:
            val = proj['estimated_hours']
            if isinstance(val, (int, float)):
                proj['estimated_hours'] = str(int(val)) if val == int(val) else str(val)
                count += 1
        # Also fix milestone-level estimated_hours
        for ms in proj.get('milestones', []):
            if 'estimated_hours' in ms:
                val = ms['estimated_hours']
                if isinstance(val, (int, float)):
                    ms['estimated_hours'] = str(int(val)) if val == int(val) else str(val)
                    count += 1
    return count


def fix_languages_format(expert_projects: dict) -> int:
    """Fix 2: Convert flat language lists to nested format."""
    count = 0
    for pid, proj in expert_projects.items():
        langs = proj.get('languages')
        if langs is None:
            continue
        if isinstance(langs, list):
            # Flat list → nested
            proj['languages'] = {
                'recommended': langs,
                'also_possible': [],
            }
            count += 1
        elif isinstance(langs, dict):
            # Ensure both keys exist
            if 'recommended' not in langs:
                langs['recommended'] = []
            if 'also_possible' not in langs:
                langs['also_possible'] = []
    return count


def fix_resource_types(expert_projects: dict) -> int:
    """Fix 3: Normalize resource type names."""
    count = 0
    for pid, proj in expert_projects.items():
        for res in proj.get('resources', []):
            rtype = res.get('type', '')
            if rtype in RESOURCE_TYPE_MAP:
                res['type'] = RESOURCE_TYPE_MAP[rtype]
                count += 1
    return count


def fix_why_expert_merge(expert_projects: dict) -> int:
    """Fix 4: Merge why_expert into why_important, then delete why_expert."""
    count = 0
    for pid, proj in expert_projects.items():
        if 'why_expert' in proj:
            if not proj.get('why_important'):
                proj['why_important'] = proj['why_expert']
                count += 1
            del proj['why_expert']
    return count


def fix_remove_category(expert_projects: dict) -> int:
    """Fix 5: Remove category field."""
    count = 0
    for pid, proj in expert_projects.items():
        if 'category' in proj:
            del proj['category']
            count += 1
    return count


def fix_milestone_order_and_project_id(expert_projects: dict) -> int:
    """Fix 6: Add missing order and project_id to milestones."""
    count = 0
    for pid, proj in expert_projects.items():
        milestones = proj.get('milestones', [])
        for idx, ms in enumerate(milestones):
            if 'order' not in ms:
                ms['order'] = idx + 1
                count += 1
            if 'project_id' not in ms:
                ms['project_id'] = pid
                count += 1
    return count


def fix_missing_project_id(expert_projects: dict) -> int:
    """Fix 7: Add missing top-level id field."""
    count = 0
    for pid, proj in expert_projects.items():
        if 'id' not in proj:
            proj['id'] = pid
            count += 1
    return count


def fix_missing_domain_id(expert_projects: dict, domain_lookup: dict) -> int:
    """Fix 8: Add missing domain_id from reverse lookup."""
    count = 0
    for pid, proj in expert_projects.items():
        if 'domain_id' not in proj:
            if pid in domain_lookup:
                proj['domain_id'] = domain_lookup[pid]
                count += 1
    return count


def fix_add_difficulty_score(expert_projects: dict, difficulty_lookup: dict) -> int:
    """Fix 9: Add difficulty_score based on difficulty level."""
    count = 0
    for pid, proj in expert_projects.items():
        if 'difficulty_score' not in proj:
            difficulty = proj.get('difficulty', '')
            # Try lookup from domain projects if not in expert_project
            if not difficulty and pid in difficulty_lookup:
                difficulty = difficulty_lookup[pid]
            score = DIFFICULTY_SCORE_MAP.get(difficulty)
            if score is not None:
                proj['difficulty_score'] = score
                count += 1
    return count


def fix_copy_descriptions_to_stubs(domains: list, expert_projects: dict) -> int:
    """Fix 10: Copy description from expert_projects to domain stubs missing descriptions."""
    count = 0
    for domain in domains:
        projects = domain.get('projects', {})
        for level, proj_list in projects.items():
            if not isinstance(proj_list, list):
                continue
            for proj in proj_list:
                pid = proj.get('id', '')
                if not proj.get('description') and pid in expert_projects:
                    expert_desc = expert_projects[pid].get('description')
                    if expert_desc:
                        proj['description'] = expert_desc
                        count += 1
    return count


# ── Fix languages in domain stubs too ────────────────────────────────────────

def fix_domain_stub_languages(domains: list) -> int:
    """Convert flat language lists in domain stubs to nested format."""
    count = 0
    for domain in domains:
        projects = domain.get('projects', {})
        for level, proj_list in projects.items():
            if not isinstance(proj_list, list):
                continue
            for proj in proj_list:
                langs = proj.get('languages')
                if isinstance(langs, list):
                    proj['languages'] = {
                        'recommended': langs,
                        'also_possible': [],
                    }
                    count += 1
    return count


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {YAML_PATH}...")
    data = load_yaml(YAML_PATH)

    domains = data.get('domains', [])
    expert_projects = data.get('expert_projects', {})

    print(f"Found {len(domains)} domains, {len(expert_projects)} expert projects")

    # Build lookups
    domain_lookup = build_domain_project_lookup(domains)
    difficulty_lookup = build_project_difficulty_lookup(domains)

    # Apply fixes sequentially
    fixes = [
        ("Fix 1: estimated_hours → string", lambda: fix_estimated_hours(expert_projects)),
        ("Fix 2: languages flat list → nested", lambda: fix_languages_format(expert_projects)),
        ("Fix 2b: domain stub languages flat → nested", lambda: fix_domain_stub_languages(domains)),
        ("Fix 3: resource type normalize", lambda: fix_resource_types(expert_projects)),
        ("Fix 4: merge why_expert → why_important", lambda: fix_why_expert_merge(expert_projects)),
        ("Fix 5: remove category", lambda: fix_remove_category(expert_projects)),
        ("Fix 6: add milestone order + project_id", lambda: fix_milestone_order_and_project_id(expert_projects)),
        ("Fix 7: add missing project id", lambda: fix_missing_project_id(expert_projects)),
        ("Fix 8: add missing domain_id", lambda: fix_missing_domain_id(expert_projects, domain_lookup)),
        ("Fix 9: add difficulty_score", lambda: fix_add_difficulty_score(expert_projects, difficulty_lookup)),
        ("Fix 10: copy descriptions to stubs", lambda: fix_copy_descriptions_to_stubs(domains, expert_projects)),
    ]

    total_changes = 0
    for name, fix_fn in fixes:
        count = fix_fn()
        total_changes += count
        status = f"({count} changes)" if count > 0 else "(no changes needed)"
        print(f"  {name}: {status}")

    if total_changes == 0:
        print("\nNo changes needed. YAML is already standardized.")
        return

    # Save
    print(f"\nTotal changes: {total_changes}")
    print(f"Saving to {YAML_PATH}...")
    save_yaml(YAML_PATH, data)
    print("Done!")


if __name__ == '__main__':
    main()
