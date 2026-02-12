import yaml


def check_orphans():
    with open("data/projects.yaml", "r") as f:
        data = yaml.safe_load(f)

    expert_ids = set(data.get("expert_projects", {}).keys())

    domain_ids = set()
    for domain in data.get("domains", []):
        for level in ["beginner", "intermediate", "advanced", "expert"]:
            projs = domain.get("projects", {}).get(level, [])
            for p in projs:
                domain_ids.add(p.get("id"))

    orphans = expert_ids - domain_ids
    return orphans


orphans = check_orphans()
print(f"Orphaned projects (in expert_projects but NOT in domains): {len(orphans)}")
for o in sorted(list(orphans)):
    print(f" - {o}")
