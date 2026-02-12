import yaml, json, time, re, os
from pathlib import Path
from openai import OpenAI

# Cấu hình Proxy
client = OpenAI(base_url="http://127.0.0.1:7999/v1", api_key="mythong2005")
MODEL = "gemini_cli/gemini-3-flash-preview"
YAML_PATH = Path("data/projects.yaml")
REPORT_PATH = Path("audit_report.json")


def get_all_projects():
    with open(YAML_PATH, "r") as f:
        data = yaml.safe_load(f)
    projects = []
    for domain in data.get("domains", []):
        for level in ["beginner", "intermediate", "advanced", "expert"]:
            projs = domain.get("projects", {}).get(level, [])
            projects.extend(projs)
    return projects


def audit_knowledge():
    projects = get_all_projects()
    total = len(projects)
    print(f"Total projects to audit: {total}")

    reports = []
    if REPORT_PATH.exists():
        with open(REPORT_PATH, "r") as f:
            reports = json.load(f)

    audited_ids = {r["id"] for r in reports}
    batch_size = 10

    # Filter projects not yet audited
    remaining = [p for p in projects if p.get("id") not in audited_ids]
    print(f"Remaining projects: {len(remaining)}")

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i : i + batch_size]
        print(f"Auditing batch {i // batch_size + 1}... ({len(reports)}/{total})")

        audit_data = []
        for p in batch:
            audit_data.append(
                {
                    "id": p.get("id"),
                    "name": p.get("name"),
                    "essence": p.get("essence"),
                    "milestones": [
                        {"title": m.get("name"), "ac": m.get("acceptance_criteria")}
                        for m in p.get("milestones", [])
                    ],
                }
            )

        prompt = f"""
        Audit these {len(batch)} software projects for accuracy and education.
        Data: {json.dumps(audit_data)}
        
        Rules:
        1. Identify Logical Gaps (order/missing steps).
        2. Identify Technical Inaccuracies.
        3. Rating (1-10).
        
        Return ONLY raw JSON list: [{{"id": "...", "issues": ["..."], "educational_score": 9}}]
        """

        try:
            res = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}]
            )
            batch_report = res.choices[0].message.content
            match = re.search(r"(\[.*\])", batch_report, re.DOTALL)
            if match:
                batch_json = json.loads(match.group(1))
                reports.extend(batch_json)
                # Atomic write
                with open(REPORT_PATH, "w") as f:
                    json.dump(reports, f, indent=2)
            else:
                print(f"  [!] Regex failed for batch.")
        except Exception as e:
            print(f"  [X] Error: {e}")

    print("\nAudit Complete!")


if __name__ == "__main__":
    audit_knowledge()
