import yaml
from pathlib import Path

YAML_PATH = Path("data/projects.yaml")


def find_empty_projects():
    with open(YAML_PATH, "r") as f:
        data = yaml.safe_load(f)

    expert_ids = set(data.get("expert_projects", {}).keys())

    empty = []
    for domain in data.get("domains", []):
        for level in ["beginner", "intermediate", "advanced", "expert"]:
            projs = domain.get("projects", {}).get(level, [])
            for p in projs:
                p_id = p.get("id")
                # A project is empty if it has no milestones AND is not in expert_projects
                if not p.get("milestones") and p_id not in expert_ids:
                    empty.append(f"{domain['name']}/{level}/{p_id}")
    return empty


empty_list = find_empty_projects()
print(f"Projects missing milestones and expert data: {len(empty_list)}")
for p in empty_list:
    print(f" - {p}")
