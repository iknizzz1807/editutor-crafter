#!/usr/bin/env python3
import json
import yaml
import os
import re
import time
from pathlib import Path

try:
    from anthropic import Anthropic
except ImportError:
    print("! Library 'anthropic' not found. Installing...")
    import subprocess

    subprocess.run(["pip", "install", "anthropic"])
    from anthropic import Anthropic

# --- CONFIG ---
SCRIPT_DIR = Path(__file__).parent.resolve()
YAML_PATH = SCRIPT_DIR / ".." / "data" / "projects.yaml"
AUDIT_PATH = SCRIPT_DIR / ".." / "audit_report.json"
OUTPUT_DIR = SCRIPT_DIR / ".." / "data" / "audited-projects"
CLAUDE_MODEL = "claude-opus-4-6"


# Initialize client (Key must be in env)
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("ERROR: ANTHROPIC_API_KEY not found in environment.")
    exit(1)

client = Anthropic(api_key=api_key)

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


def invoke_claude_api(prompt):
    """Call Claude API using streaming to handle long-running generations."""
    print(f"    [API] Sending {len(prompt)} chars to Claude (Streaming enabled)...")
    try:
        start_time = time.time()
        full_response = ""

        # Using stream to bypass the 10-minute non-streaming limit
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=64000,
            temperature=1,
            system="You are a Senior Systems Architect and Technical Auditor. You are known for being extremely critical, rigorous, and obsessed with technical truth. Output ONLY a raw JSON array.",
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                full_response += text

        print(f"\n    [API] Response received in {time.time() - start_time:.1f}s")

        return full_response
    except Exception as e:
        print(f"\n    ! API Exception: {e}")
        return None


def audit_projects_batch(batch_data):
    """Audit a list of projects in a single API call."""
    ids = [p[0]["id"] for p in batch_data]
    print(f"\n>>> Auditing batch: {', '.join(ids)}...")

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

TASK:
I am providing you with a BATCH of projects. For EACH project, you must:
1. CRITIQUE: Analyze the Original Project and the Audit Findings. Be harsh. Identify logical gaps, security risks, or performance bottlenecks.
2. RECONCILE: Fix all technical inaccuracies.
3. REWRITE: Provide a "TRUTH TRUSTED" version. Ensure Milestones are sequential, Acceptance Criteria (AC) are rigorous and measurable, and Pitfalls are insightful.

STRICT RULE: Your response must be a SINGLE JSON ARRAY. 
Each element schema:
{{
  "project_id": "string",
  "critique": "string (bullet points)",
  "fixed_project_yaml": "string (the complete project definition in valid YAML format)"
}}

BATCH DATA:
{json.dumps(input_payload, indent=2)}
"""

    response_text = invoke_claude_api(prompt)
    if not response_text:
        return []

    try:
        # Extract JSON array from response
        json_match = re.search(r"(\[.*\])", response_text, re.DOTALL)
        if json_match:
            clean_json = json_match.group(1)
            return json.loads(clean_json)
        else:
            print("    ! No JSON array found in response.")
            debug_file = f"no_json_found_{int(time.time())}.txt"
            debug_path = OUTPUT_DIR / debug_file
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(response_text)
            print(f"    ! Raw response saved to {debug_path} for inspection.")
            return []
    except Exception as e:
        print(f"    ! Failed to parse batch JSON response: {e}")
        debug_file = f"error_batch_{int(time.time())}.txt"
        debug_path = OUTPUT_DIR / debug_file
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(response_text)
        print(f"    ! Raw response saved to {debug_path}")
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

    print(f"Starting audit for {len(projects_to_audit)} projects in batches of 10...")

    # Process in batches of 10
    batch_size = 10
    for i in range(0, len(projects_to_audit), batch_size):
        batch = projects_to_audit[i : i + batch_size]
        try:
            results = audit_projects_batch(batch)

            for res in results:
                p_id = res.get("project_id")
                if p_id:
                    output_path = OUTPUT_DIR / f"{p_id}.md"
                    content = f"# AUDIT & FIX: {p_id}\n\n## CRITIQUE\n{res.get('critique')}\n\n## FIXED YAML\n```yaml\n{res.get('fixed_project_yaml')}\n```"
                    output_path.write_text(content)
                    print(f"  ✓ Saved: {p_id}")
        except Exception as e:
            print(f"  ! Error processing batch {i // batch_size + 1}: {e}")
            continue


if __name__ == "__main__":
    main()
