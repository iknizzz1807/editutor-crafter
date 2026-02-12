import yaml
import json


def audit_new_projects():
    with open("data/projects.yaml", "r") as f:
        data = yaml.safe_load(f)

    new_ids = [
        "ad-exchange-engine",
        "anticheat-kernel-driver",
        "distributed-consensus-raft",
        "high-cardinality-metrics",
        "mlops-pipeline-auto",
        "mmo-engine-core",
    ]

    results = []
    for domain in data.get("domains", []):
        for level in ["beginner", "intermediate", "advanced", "expert"]:
            for p in domain.get("projects", {}).get(level, []):
                if p["id"] in new_ids:
                    # Check fields
                    has_hints = any("hints" in ms for ms in p.get("milestones", []))
                    results.append(
                        {
                            "id": p["id"],
                            "name": p["name"],
                            "ms_count": len(p.get("milestones", [])),
                            "has_hints": has_hints,
                            "fields": list(p.keys()),
                        }
                    )
    return results


res = audit_new_projects()
for r in res:
    print(json.dumps(r, indent=2))
