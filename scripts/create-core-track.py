#!/usr/bin/env python3
"""Create core track by removing too-easy and too-niche projects."""

import yaml
from pathlib import Path

# Projects to remove - too easy (15)
TOO_EASY = {
    "systems": ["cat-clone", "wc-clone", "file-copy"],
    "app-dev": ["todo-app", "weather-app", "calculator", "portfolio-site"],
    "data-storage": ["json-db", "kv-memory"],
    "game-dev": ["pong", "snake"],
    "software-engineering": ["documentation-project", "git-workflow", "tdd-kata", "code-review-practice"],
}

# Projects to remove - too niche (9)
TOO_NICHE = {
    "distributed": ["build-blockchain"],
    "specialized": ["chip8-emulator", "simd-library"],
    "systems": ["bootloader", "device-driver", "build-quic"],
    "ai-ml": ["gan"],
    "data-storage": ["graph-db", "data-lakehouse"],
}

# Projects to merge/consolidate (remove these, keep alternatives)
CONSOLIDATE = {
    "software-engineering": ["ci-pipeline"],  # keep ci-cd-pipeline
    "systems": ["shell-basic", "grep-clone"],  # keep mini-shell
}


def main():
    projects_file = Path("data/projects.yaml")

    with open(projects_file, 'r') as f:
        data = yaml.safe_load(f)

    removed = []

    # Combine all removal lists
    all_removals = {}
    for removal_dict in [TOO_EASY, TOO_NICHE, CONSOLIDATE]:
        for domain, projects in removal_dict.items():
            if domain not in all_removals:
                all_removals[domain] = []
            all_removals[domain].extend(projects)

    # Process each domain
    for domain in data['domains']:
        domain_id = domain['id']

        if domain_id not in all_removals:
            continue

        to_remove = set(all_removals[domain_id])

        for level_name in ['beginner', 'intermediate', 'advanced', 'expert']:
            if level_name not in domain.get('projects', {}):
                continue

            level_projects = domain['projects'][level_name]
            if not isinstance(level_projects, list):
                continue

            original_len = len(level_projects)
            domain['projects'][level_name] = [
                p for p in level_projects
                if p['id'] not in to_remove
            ]

            removed_count = original_len - len(domain['projects'][level_name])
            if removed_count > 0:
                removed_ids = [p['id'] for p in level_projects if p['id'] in to_remove]
                for rid in removed_ids:
                    removed.append(f"{domain_id}/{level_name}: {rid}")

    # Save
    with open(projects_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

    print("Removed projects:")
    for r in sorted(removed):
        print(f"  - {r}")

    print(f"\nTotal removed: {len(removed)}")

    # Count remaining
    total = 0
    for domain in data['domains']:
        for level_name in ['beginner', 'intermediate', 'advanced', 'expert']:
            if level_name in domain.get('projects', {}):
                projects = domain['projects'][level_name]
                if isinstance(projects, list):
                    total += len(projects)

    print(f"Remaining projects: {total}")


if __name__ == "__main__":
    main()
