import yaml


def find_missing_details():
    with open("data/projects.yaml", "r") as f:
        data = yaml.safe_load(f)

    expert_ids = set(data.get("expert_projects", {}).keys())
    missing = []

    for domain in data.get("domains", []):
        for level in ["beginner", "intermediate", "advanced", "expert"]:
            projs = domain.get("projects", {}).get(level, [])
            for p in projs:
                p_id = p.get("id")
                if not p.get("milestones") and p_id not in expert_ids:
                    missing.append((domain["name"], level, p_id, p["name"]))
    return missing


missing = find_missing_details()
print(f"Projects missing all details: {len(missing)}")
for d, l, pid, name in missing:
    print(f" - {d} | {l} | {pid} | {name}")
