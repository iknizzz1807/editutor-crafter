#!/usr/bin/env python3
import json
import yaml
import subprocess
import os
import re
from pathlib import Path

# --- CONFIG ---
SCRIPT_DIR = Path(__file__).parent.resolve()
YAML_PATH = SCRIPT_DIR / ".." / "data" / "projects.yaml"
AUDIT_PATH = SCRIPT_DIR / ".." / "audit_report.json"
OUTPUT_DIR = SCRIPT_DIR / ".." / "data" / "audited-projects"
CLAUDE_MODEL = "sonnet"

PLATFORM_CONTEXT = """
CONTEXT:
This is a project-based learning platform for systems engineering.
- Projects are divided into sequential Milestones.
- Users must complete and SUBMIT each milestone to progress.
- Each milestone requires:
    1. Clear Description (What to do).
    2. Detailed Acceptance Criteria (AC) (How to verify success).
    3. Pitfalls (Common mistakes/optimization traps).
- Goal: Deep understanding, performance obsession, and "Truth Trusted" technical accuracy.
"""


def load_data():
    with open(YAML_PATH, "r") as f:
        projects_data = yaml.safe_load(f)
    with open(AUDIT_PATH, "r") as f:
        audit_data = json.load(f)
    return projects_data, audit_data


def invoke_claude(prompt):
    """Call Claude CLI with the given prompt."""
    cmd = [
        "claude",
        "-p",
        "--model",
        CLAUDE_MODEL,
        "--dangerously-skip-permissions",
        "--tools",
        "",
        "--output-format",
        "json",
    ]
    try:
        result = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            output = json.loads(result.stdout)
            return output.get("result", "")
        return f"Error: {result.stderr}"
    except Exception as e:
        return f"Exception: {e}"


def audit_projects_batch(batch_data):
    """Audit a list of projects in a single Claude call."""
    print(f">>> Auditing batch of {len(batch_data)} projects...")

    # Prepare the payload for the batch
    input_payload = []
    for project, issues in batch_data:
        input_payload.append(
            {
                "project_id": project["id"],
                "original_definition": project,
                "audit_findings": issues,
            }
        )

    prompt = f"""
{PLATFORM_CONTEXT}

SYSTEM ROLE:
You are a Senior Systems Architect and Technical Auditor. You are known for being extremely critical, rigorous, and obsessed with technical truth.

TASK:
I am providing you with a BATCH of {len(batch_data)} projects. For EACH project, you must:
1. CRITIQUE: Analyze the Original Project and the Audit Findings. Be harsh. Identify logical gaps, security risks, or performance bottlenecks.
2. RECONCILE: Fix all technical inaccuracies.
3. REWRITE: Provide a "TRUTH TRUSTED" version. Ensure Milestones are sequential, Acceptance Criteria (AC) are rigorous and measurable, and Pitfalls are insightful.

STRICT RULE: Your response must be a SINGLE JSON ARRAY. Each element in the array must follow this schema:
{{
  "project_id": "string",
  "critique": "string (bullet points)",
  "fixed_project_yaml": "string (the complete project definition in valid YAML format)"
}}

BATCH DATA:
{json.dumps(input_payload, indent=2)}
"""

    response_text = invoke_claude(prompt)
    try:
        # Claude returns the result as a string within a JSON response due to --output-format json
        # We need to parse that string as a JSON array.
        batch_results = json.loads(response_text)
        return batch_results
    except Exception as e:
        print(f"    ! Failed to parse batch JSON: {e}")
        # Save raw response for debugging
        debug_path = OUTPUT_DIR / f"error_batch_{time.time()}.txt"
        debug_path.write_text(response_text)
        return []


def main():
    projects_data, audit_data = load_data()
    audit_map = {item["id"]: item["issues"] for item in audit_data}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Flatten and filter projects that have audit issues
    projects_to_audit = []
    for domain in projects_data.get("domains", []):
        for difficulty in ["beginner", "intermediate", "advanced", "expert"]:
            projs = domain.get("projects", {}).get(difficulty, [])
            for p in projs:
                if p["id"] in audit_map:
                    projects_to_audit.append((p, audit_map[p["id"]]))

    # Process in batches of 10
    batch_size = 10
    for i in range(0, len(projects_to_audit), batch_size):
        batch = projects_to_audit[i : i + batch_size]
        results = audit_projects_batch(batch)

        for res in results:
            p_id = res.get("project_id")
            if p_id:
                output_path = OUTPUT_DIR / f"{p_id}.md"
                content = f"# AUDIT & FIX: {p_id}\n\n## CRITIQUE\n{res.get('critique')}\n\n## FIXED YAML\n```yaml\n{res.get('fixed_project_yaml')}\n```"
                output_path.write_text(content)
                print(f"  ✓ Saved: {p_id}")


if __name__ == "__main__":
    import time

    main()


if __name__ == "__main__":
    main()
