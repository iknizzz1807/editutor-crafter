import yaml
import json
from pathlib import Path

YAML_PATH = Path("data/projects.yaml")


def pure_cleanup_v2():
    print("Reading projects.yaml...")
    with open(YAML_PATH, "r") as f:
        data = yaml.safe_load(f)

    if not data:
        print("Error: Could not load YAML data.")
        return

    expert_map = data.get("expert_projects", {})
    print(f"Found {len(expert_map)} projects in expert_projects.")

    def clean_milestone(ms, p_id, idx):
        # Fields to KEEP
        kept_fields = [
            "name",
            "description",
            "acceptance_criteria",
            "pitfalls",
            "concepts",
            "skills",
            "deliverables",
            "estimated_hours",
        ]

        new_ms = {}
        # 1. Set standardized ID
        new_ms["id"] = f"{p_id}-m{idx + 1}"

        # 2. Copy allowed fields
        for field in kept_fields:
            val = ms.get(field)
            if not val and field == "acceptance_criteria":
                val = ms.get("acceptanceCriteria")
            if val:
                new_ms[field] = val

        return new_ms

    def clean_project(p):
        p_id = p.get("id")
        if not p_id:
            return None

        # Merge from expert_map if available
        if p_id in expert_map:
            exp_data = expert_map[p_id]
            # Copy all richness from expert
            for key, val in exp_data.items():
                if val and not p.get(key):
                    p[key] = val

        # Fields to KEEP at project level
        project_fields = [
            "id",
            "name",
            "description",
            "difficulty",
            "estimated_hours",
            "essence",
            "why_important",
            "learning_outcomes",
            "skills",
            "tags",
            "architecture_doc",
            "languages",
            "resources",
            "prerequisites",
        ]

        new_p = {}
        for field in project_fields:
            val = p.get(field)
            if val:
                new_p[field] = val

        # Process Milestones
        if "milestones" in p:
            new_ms_list = []
            for i, ms in enumerate(p["milestones"]):
                new_ms_list.append(clean_milestone(ms, p_id, i))
            new_p["milestones"] = new_ms_list

        return new_p

    def process_recursive(node):
        if isinstance(node, list):
            return [process_recursive(i) for i in node]
        elif isinstance(node, dict):
            if "projects" in node:
                new_projects = {}
                for level in ["beginner", "intermediate", "advanced", "expert"]:
                    if level in node["projects"]:
                        projs = node["projects"][level]
                        cleaned = [clean_project(p) for p in projs]
                        new_projects[level] = [c for c in cleaned if c is not None]
                node["projects"] = new_projects

            for k, v in node.items():
                if k != "projects":
                    node[k] = process_recursive(v)
        return node

    # Clean the hierarchy
    data["domains"] = process_recursive(data.get("domains", []))

    # Remove expert_projects section
    if "expert_projects" in data:
        del data["expert_projects"]

    print("Saving pure projects.yaml...")
    with open(YAML_PATH, "w") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True, width=2000)

    print("Cleanup complete!")


if __name__ == "__main__":
    pure_cleanup_v2()
