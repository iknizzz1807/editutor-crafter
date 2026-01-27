#!/usr/bin/env python3
"""
Check milestone counts and quality per project.
"""

import yaml
import os
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, '..', 'data', 'projects.yaml')

with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

expert_projects = data.get('expert_projects', {})

# Group by milestone count
by_milestone_count = defaultdict(list)

print("=" * 80)
print("MILESTONE COUNT PER PROJECT")
print("=" * 80)

for project_id, project in sorted(expert_projects.items()):
    milestones = project.get('milestones', [])
    count = len(milestones)
    by_milestone_count[count].append(project_id)

print(f"\n{'Milestones':<12} {'Count':<8} Projects")
print("-" * 60)
for count in sorted(by_milestone_count.keys()):
    projects = by_milestone_count[count]
    print(f"{count:<12} {len(projects):<8} {', '.join(projects[:5])}{'...' if len(projects) > 5 else ''}")

# Projects with few milestones (< 3)
print("\n" + "=" * 80)
print("PROJECTS WITH < 3 MILESTONES (may need more)")
print("=" * 80)

for project_id, project in sorted(expert_projects.items()):
    milestones = project.get('milestones', [])
    if len(milestones) < 3:
        print(f"\n❌ {project_id}: {len(milestones)} milestones")
        for m in milestones:
            print(f"   - {m.get('name', 'Unnamed')}")

# Projects with many milestones (> 6)
print("\n" + "=" * 80)
print("PROJECTS WITH > 6 MILESTONES (good depth)")
print("=" * 80)

for project_id, project in sorted(expert_projects.items()):
    milestones = project.get('milestones', [])
    if len(milestones) > 6:
        print(f"\n✓ {project_id}: {len(milestones)} milestones")

# Check for milestones with short hints
print("\n" + "=" * 80)
print("MILESTONES WITH SHORT/MISSING LEVEL3 HINTS")
print("=" * 80)

short_hints = []
for project_id, project in expert_projects.items():
    milestones = project.get('milestones', [])
    for m in milestones:
        hints = m.get('hints', {})
        level3 = hints.get('level3', '')
        if len(str(level3)) < 100:  # Level 3 should be detailed
            short_hints.append((project_id, m.get('name', 'Unnamed'), len(str(level3))))

if short_hints:
    print(f"\n⚠ {len(short_hints)} milestones with short level3 hints (< 100 chars):")
    for pid, mname, length in short_hints[:20]:
        print(f"   {pid}/{mname}: {length} chars")
    if len(short_hints) > 20:
        print(f"   ... and {len(short_hints) - 20} more")
else:
    print("\n✓ All milestones have substantial level3 hints")

# Check pitfalls
print("\n" + "=" * 80)
print("MILESTONES WITHOUT PITFALLS")
print("=" * 80)

no_pitfalls = []
for project_id, project in expert_projects.items():
    milestones = project.get('milestones', [])
    for m in milestones:
        if not m.get('pitfalls'):
            no_pitfalls.append((project_id, m.get('name', 'Unnamed')))

if no_pitfalls:
    print(f"\n⚠ {len(no_pitfalls)} milestones without pitfalls:")
    for pid, mname in no_pitfalls[:20]:
        print(f"   {pid}/{mname}")
else:
    print("\n✓ All milestones have pitfalls defined")

# Summary stats
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

total_milestones = sum(len(p.get('milestones', [])) for p in expert_projects.values())
avg = total_milestones / len(expert_projects)

print(f"\nTotal projects: {len(expert_projects)}")
print(f"Total milestones: {total_milestones}")
print(f"Average milestones/project: {avg:.1f}")
print(f"Projects with < 3 milestones: {len([p for p in expert_projects.values() if len(p.get('milestones', [])) < 3])}")
print(f"Projects with 3-5 milestones: {len([p for p in expert_projects.values() if 3 <= len(p.get('milestones', [])) <= 5])}")
print(f"Projects with > 5 milestones: {len([p for p in expert_projects.values() if len(p.get('milestones', [])) > 5])}")
