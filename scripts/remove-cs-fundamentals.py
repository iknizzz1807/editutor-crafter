#!/usr/bin/env python3
"""Remove CS Fundamentals expert_projects from projects.yaml."""

import yaml
from pathlib import Path

# Projects to remove (from CS Fundamentals domain)
CS_PROJECTS = [
    "linked-list",
    "stack-queue",
    "bst",
    "hash-table",
    "red-black-tree",
    "graph-algos",
    "build-btree"
]

def main():
    projects_file = Path("data/projects.yaml")

    with open(projects_file, 'r') as f:
        data = yaml.safe_load(f)

    removed_count = 0

    if 'expert_projects' in data:
        for project_id in CS_PROJECTS:
            if project_id in data['expert_projects']:
                del data['expert_projects'][project_id]
                print(f"Removed expert_project: {project_id}")
                removed_count += 1
            else:
                print(f"Not found: {project_id}")

    with open(projects_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

    print(f"\nRemoved {removed_count} CS Fundamentals expert_projects")

if __name__ == "__main__":
    main()
