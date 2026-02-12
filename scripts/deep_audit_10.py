import yaml
import json


def deep_audit():
    with open("data/projects.yaml", "r") as f:
        data = yaml.safe_load(f)

    target_ids = [
        "distributed-consensus-raft",
        "build-os",
        "simple-gc",
        "build-sqlite",
        "binary-patcher",
        "transformer-scratch",
        "grpc-service",
        "build-tcp-stack",
        "build-game-engine",
        "build-docker",
    ]

    audit_results = {}
    for domain in data.get("domains", []):
        for level in ["beginner", "intermediate", "advanced", "expert"]:
            for p in domain.get("projects", {}).get(level, []):
                if p["id"] in target_ids:
                    audit_results[p["id"]] = p

    return audit_results


res = deep_audit()
# Print titles and milestone summaries for review
for pid, pdata in res.items():
    print(f"--- PROJECT: {pid} ---")
    print(f"Essence: {pdata.get('essence')}")
    for i, ms in enumerate(pdata.get("milestones", [])):
        print(
            f"  M{i + 1}: {ms.get('name')} | AC Count: {len(ms.get('acceptance_criteria', []))}"
        )
