#!/usr/bin/env python3
"""
Comprehensive validation of projects.yaml data completeness.
Checks:
1. All domains have projects at all levels
2. All projects marked detailed=true have specs in expert_projects
3. All expert_projects have complete milestone data
4. No orphaned expert_projects (not referenced in domains)
"""

import yaml
import os
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, '..', 'data', 'projects.yaml')

with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

domains = data.get('domains', [])
expert_projects = data.get('expert_projects', {})

print("=" * 80)
print("EDITUTOR CRAFTER DATA VALIDATION")
print("=" * 80)

# Track all issues
issues = []
warnings = []

# Track all project IDs
all_domain_project_ids = set()
detailed_project_ids = set()

# =============================================================================
# 1. DOMAIN ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("1. DOMAIN COVERAGE")
print("=" * 80)

levels = ['beginner', 'intermediate', 'advanced', 'expert']

for domain in domains:
    domain_id = domain.get('id', 'unknown')
    domain_name = domain.get('name', 'Unknown')
    projects = domain.get('projects', {})

    print(f"\n{'─' * 40}")
    print(f"📁 {domain_name} ({domain_id})")
    print(f"{'─' * 40}")

    total = 0
    detailed_count = 0

    for level in levels:
        level_projects = projects.get(level, [])
        count = len(level_projects)
        total += count

        # Collect project IDs
        for p in level_projects:
            pid = p.get('id')
            if pid:
                all_domain_project_ids.add(pid)
                if p.get('detailed'):
                    detailed_project_ids.add(pid)
                    detailed_count += 1

        # Check for empty levels
        status = "✓" if count > 0 else "⚠ EMPTY"
        print(f"  {level.capitalize():15} {count:3} projects {status}")

        if count == 0:
            warnings.append(f"{domain_id}: No {level} projects")

    print(f"  {'─' * 30}")
    print(f"  Total: {total} projects, {detailed_count} detailed")

# =============================================================================
# 2. DETAILED PROJECT SPEC CHECK
# =============================================================================
print("\n" + "=" * 80)
print("2. DETAILED PROJECT SPECS CHECK")
print("=" * 80)

# Projects marked detailed but missing specs
missing_specs = detailed_project_ids - set(expert_projects.keys())
if missing_specs:
    print(f"\n❌ {len(missing_specs)} projects marked 'detailed: true' but NO SPECS:")
    for pid in sorted(missing_specs):
        print(f"   - {pid}")
        issues.append(f"Missing spec for: {pid}")
else:
    print(f"\n✓ All {len(detailed_project_ids)} detailed projects have specs")

# Orphaned specs (in expert_projects but not referenced in domains)
orphaned_specs = set(expert_projects.keys()) - all_domain_project_ids
if orphaned_specs:
    print(f"\n⚠ {len(orphaned_specs)} orphaned specs (not in any domain):")
    for pid in sorted(orphaned_specs):
        print(f"   - {pid}")
        warnings.append(f"Orphaned spec: {pid}")

# Projects in domains but not marked as detailed
not_detailed = all_domain_project_ids - detailed_project_ids
print(f"\n📋 {len(not_detailed)} projects NOT marked as detailed:")
if len(not_detailed) <= 20:
    for pid in sorted(not_detailed):
        print(f"   - {pid}")

# =============================================================================
# 3. MILESTONE COMPLETENESS CHECK
# =============================================================================
print("\n" + "=" * 80)
print("3. MILESTONE COMPLETENESS CHECK")
print("=" * 80)

required_fields = ['id', 'name', 'description', 'acceptance_criteria']
optional_fields = ['hints', 'pitfalls', 'concepts', 'estimated_hours']
hint_levels = ['level1', 'level2', 'level3']

incomplete_milestones = []
milestone_stats = defaultdict(int)

for project_id, project in expert_projects.items():
    milestones = project.get('milestones', [])

    if not milestones:
        issues.append(f"{project_id}: No milestones defined")
        continue

    milestone_stats['total_projects'] += 1
    milestone_stats['total_milestones'] += len(milestones)

    for i, milestone in enumerate(milestones):
        m_id = milestone.get('id', f"milestone-{i}")
        m_name = milestone.get('name', 'Unnamed')

        missing_required = []
        missing_optional = []

        # Check required fields
        for field in required_fields:
            if not milestone.get(field):
                missing_required.append(field)

        # Check optional fields
        for field in optional_fields:
            if not milestone.get(field):
                missing_optional.append(field)

        # Check hints structure
        hints = milestone.get('hints', {})
        missing_hints = []
        if hints:
            for level in hint_levels:
                if level not in hints:
                    missing_hints.append(level)
        else:
            missing_hints = hint_levels

        # Track issues
        if missing_required:
            issues.append(f"{project_id}/{m_id}: Missing required: {missing_required}")
            incomplete_milestones.append((project_id, m_id, 'required', missing_required))

        if missing_hints:
            milestone_stats['missing_hints'] += 1
        else:
            milestone_stats['complete_hints'] += 1

        if not milestone.get('pitfalls'):
            milestone_stats['missing_pitfalls'] += 1
        else:
            milestone_stats['has_pitfalls'] += 1

        if not milestone.get('concepts'):
            milestone_stats['missing_concepts'] += 1
        else:
            milestone_stats['has_concepts'] += 1

        if not milestone.get('acceptance_criteria'):
            milestone_stats['missing_criteria'] += 1
        else:
            milestone_stats['has_criteria'] += 1

print(f"\n📊 Milestone Statistics:")
print(f"   Total projects with specs: {milestone_stats['total_projects']}")
print(f"   Total milestones: {milestone_stats['total_milestones']}")
print(f"   Avg milestones/project: {milestone_stats['total_milestones'] / max(1, milestone_stats['total_projects']):.1f}")

print(f"\n📋 Field Coverage:")
total_m = milestone_stats['total_milestones']
print(f"   Acceptance criteria: {milestone_stats['has_criteria']}/{total_m} ({100*milestone_stats['has_criteria']/total_m:.1f}%)")
print(f"   Complete hints (3 levels): {milestone_stats['complete_hints']}/{total_m} ({100*milestone_stats['complete_hints']/total_m:.1f}%)")
print(f"   Pitfalls: {milestone_stats['has_pitfalls']}/{total_m} ({100*milestone_stats['has_pitfalls']/total_m:.1f}%)")
print(f"   Concepts: {milestone_stats['has_concepts']}/{total_m} ({100*milestone_stats['has_concepts']/total_m:.1f}%)")

# =============================================================================
# 4. EXPERT PROJECT DETAIL CHECK
# =============================================================================
print("\n" + "=" * 80)
print("4. EXPERT PROJECT DETAIL CHECK")
print("=" * 80)

project_fields = ['name', 'description', 'difficulty', 'estimated_hours', 'prerequisites', 'languages', 'resources', 'milestones']

incomplete_projects = []

for project_id, project in expert_projects.items():
    missing = []
    for field in project_fields:
        val = project.get(field)
        if not val or (isinstance(val, list) and len(val) == 0):
            missing.append(field)

    if missing:
        incomplete_projects.append((project_id, missing))

if incomplete_projects:
    print(f"\n⚠ {len(incomplete_projects)} projects with missing fields:")
    for pid, missing in incomplete_projects[:20]:
        print(f"   {pid}: {missing}")
    if len(incomplete_projects) > 20:
        print(f"   ... and {len(incomplete_projects) - 20} more")
else:
    print(f"\n✓ All {len(expert_projects)} expert projects have complete metadata")

# =============================================================================
# 5. DOMAIN-LEVEL DENSITY CHECK
# =============================================================================
print("\n" + "=" * 80)
print("5. DOMAIN DENSITY ANALYSIS")
print("=" * 80)

print(f"\n{'Domain':<30} {'Beg':>5} {'Int':>5} {'Adv':>5} {'Exp':>5} {'Total':>6} {'Detailed':>9}")
print("─" * 75)

total_all = 0
total_detailed = 0

for domain in domains:
    domain_id = domain.get('id', 'unknown')
    domain_name = domain.get('name', 'Unknown')[:28]
    projects = domain.get('projects', {})

    counts = {}
    detailed = 0
    for level in levels:
        level_projects = projects.get(level, [])
        counts[level] = len(level_projects)
        detailed += sum(1 for p in level_projects if p.get('detailed'))

    total = sum(counts.values())
    total_all += total
    total_detailed += detailed

    density = "🟢" if detailed == total else "🟡" if detailed > total * 0.7 else "🔴"

    print(f"{domain_name:<30} {counts['beginner']:>5} {counts['intermediate']:>5} {counts['advanced']:>5} {counts['expert']:>5} {total:>6} {detailed:>5} {density}")

print("─" * 75)
print(f"{'TOTAL':<30} {'':<5} {'':<5} {'':<5} {'':<5} {total_all:>6} {total_detailed:>9}")
print(f"\nOverall coverage: {total_detailed}/{total_all} = {100*total_detailed/total_all:.1f}%")

# =============================================================================
# 6. SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n🔴 Critical Issues: {len(issues)}")
for issue in issues[:10]:
    print(f"   - {issue}")
if len(issues) > 10:
    print(f"   ... and {len(issues) - 10} more")

print(f"\n🟡 Warnings: {len(warnings)}")
for warning in warnings[:10]:
    print(f"   - {warning}")
if len(warnings) > 10:
    print(f"   ... and {len(warnings) - 10} more")

if len(issues) == 0 and len(warnings) == 0:
    print("\n✅ All checks passed!")
elif len(issues) == 0:
    print("\n✅ No critical issues, but some warnings to address")
else:
    print("\n❌ Critical issues need to be fixed")

print("\n" + "=" * 80)
