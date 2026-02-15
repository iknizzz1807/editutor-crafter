import yaml
import json
import time
import re
from pathlib import Path
from openai import OpenAI

# Setup
client = OpenAI(base_url="http://127.0.0.1:7999/v1", api_key="mythong2005")
MODEL = "gemini_cli/gemini-3-pro-preview"
YAML_PATH = Path("data/projects.yaml")
REPORT_PATH = Path("audit_report.json")


def load_data():
    with open(YAML_PATH, "r") as f:
        data = yaml.safe_load(f)
    with open(REPORT_PATH, "r") as f:
        report = json.load(f)
    return data, report


def fix_projects():
    data, report = load_data()

    # Sort projects by score ascending
    report.sort(key=lambda x: x.get("educational_score", 10))

    # Filter only those with issues
    to_fix = [r for r in report if r.get("issues")]
    print(f"Projects to fix: {len(to_fix)}")

    # Create a lookup map for the data
    all_projects = {}
    for domain in data.get("domains", []):
        for level in ["beginner", "intermediate", "advanced", "expert"]:
            for p in domain.get("projects", {}).get(level, []):
                all_projects[p["id"]] = p

    for i, report_item in enumerate(to_fix):
        p_id = report_item["id"]
        issues = report_item["issues"]
        print(
            f"[{i + 1}/{len(to_fix)}] Fixing {p_id} (Score: {report_item['educational_score']})..."
        )

        target_p = all_projects.get(p_id)
        if not target_p:
            continue

        prompt = f"""
        Fix the following technical issues in this project specification.
        Project: {target_p["name"]}
        Issues: {json.dumps(issues)}
        Current Milestones: {json.dumps(target_p.get("milestones"))}
        
        Rules:
        1. Resolve all logical gaps and inaccuracies.
        2. Keep the same structure (list of milestones).
        3. Ensure every milestone has name, description, and acceptance_criteria.
        
        Return ONLY the updated 'milestones' list as raw JSON.
        """

        try:
            res = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8000,
            )
            # Safe parse
            match = re.search(r"(\[.*\])", res.choices[0].message.content, re.DOTALL)
            if match:
                new_milestones = json.loads(match.group(1))
                # Standardize IDs
                for idx, ms in enumerate(new_milestones):
                    ms["id"] = f"{p_id}-m{idx + 1}"
                target_p["milestones"] = new_milestones
                print(f"  ✓ Fixed.")
            else:
                print(f"  [!] Failed to parse fix for {p_id}")
        except Exception as e:
            print(f"  [X] AI Error: {e}")

        # Intermediate save every 5 projects to be safe
        if (i + 1) % 5 == 0:
            with open(YAML_PATH, "w") as f:
                yaml.dump(data, f, sort_keys=False, allow_unicode=True, width=2000)
            print("  --- Checkpoint saved ---")

    # Final Save
    with open(YAML_PATH, "w") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True, width=2000)
    print("\nAll projects fixed!")


if __name__ == "__main__":
    fix_projects()
