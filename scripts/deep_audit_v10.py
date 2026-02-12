import yaml
import json
from pathlib import Path

YAML_PATH = Path("data/projects.yaml")


def get_all_projects_recursive(data):
    projects = []
    if isinstance(data, dict):
        # Check if this level has a 'projects' key
        if "projects" in data:
            p_block = data["projects"]
            for level in ["beginner", "intermediate", "advanced", "expert"]:
                if level in p_block and isinstance(p_block[level], list):
                    projects.extend(p_block[level])

        # Recurse into subdomains or other dictionaries
        for k, v in data.items():
            if k == "projects":
                continue  # Already handled
            projects.extend(get_all_projects_recursive(v))

    elif isinstance(data, list):
        for item in data:
            projects.extend(get_all_projects_recursive(item))

    return projects


def deep_audit():
    with open(YAML_PATH, "r") as f:
        data = yaml.safe_load(f)

    all_p = get_all_projects_recursive(data.get("domains", []))
    ids = [p.get("id") for p in all_p if p.get("id")]

    expert_ids = list(data.get("expert_projects", {}).keys())

    print(f"Total projects found in hierarchy (including subdomains): {len(all_p)}")
    print(f"Unique project IDs in hierarchy: {len(set(ids))}")
    print(f"Total projects in expert_projects: {len(expert_ids)}")
    print(f"Combined unique IDs: {len(set(ids) | set(expert_ids))}")


if __name__ == "__main__":
    deep_audit()
