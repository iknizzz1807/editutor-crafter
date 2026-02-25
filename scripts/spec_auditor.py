#!/usr/bin/env python3
"""
Spec Auditor - Audit project specs from projects_data/ using LLM.

Usage:
    python3 scripts/spec_auditor.py --projects build-react build-git
    python3 scripts/spec_auditor.py --domain systems
    python3 scripts/spec_auditor.py --level expert
    python3 scripts/spec_auditor.py --all
    python3 scripts/spec_auditor.py --all --batch-size 5  # Batch processing
    python3 scripts/spec_auditor.py --dry-run  # Just list projects to audit

Features:
- Reads from projects_data/*.yaml (source of truth)
- Supports multiple LLM providers (Anthropic, Claude CLI, Gemini proxy)
- Batch processing to reduce API calls
- Detailed audit reports with critique and fixed YAML
- Resume from where it left off (checkpoint)
"""

import argparse
import json
import os
import re
import subprocess
import time
import yaml
from pathlib import Path
from typing import Optional

# --- CONFIG ---
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / ".." / "data"
PROJECTS_DATA_DIR = DATA_DIR / "projects_data"
OUTPUT_DIR = DATA_DIR / "audited-specs"
CHECKPOINT_FILE = OUTPUT_DIR / ".audit_checkpoint.json"

# LLM Provider settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic, claude-cli, gemini-proxy
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
GEMINI_PROXY_URL = os.getenv("GEMINI_PROXY_URL", "http://127.0.0.1:7999/v1")
GEMINI_PROXY_KEY = os.getenv("GEMINI_PROXY_KEY", "mythong2005")

DEFAULT_BATCH_SIZE = 5  # Number of projects per API call


AUDIT_CONTEXT = """
CONTEXT:
This is a project-based learning platform for software engineering.
- Projects are divided into sequential Milestones
- Users submit code for each milestone to progress
- Each milestone has:
  1. Description (what to build)
  2. Acceptance Criteria (how to verify - must be MEASURABLE)
  3. Common Pitfalls (what mistakes to avoid)
  4. Concepts & Skills (what they learn)

AUDIT CRITERIA:
1. **Technical Accuracy**: Are the concepts correct? Any outdated info?
2. **Measurability**: Can acceptance criteria be objectively tested?
3. **Progression**: Do milestones build on each other logically?
4. **Completeness**: Are there gaps in the learning path?
5. **Realism**: Is the scope appropriate for the difficulty level?
6. **Security**: Are there security considerations missing?
7. **Performance**: Are performance considerations addressed?

IMPORTANT: Only flag REAL issues. If something is already good, say it's good - don't invent problems or make trivial changes.
"""


def load_project(project_id: str) -> Optional[dict]:
    """Load a single project from projects_data/."""
    yaml_path = PROJECTS_DATA_DIR / f"{project_id}.yaml"
    if not yaml_path.exists():
        return None
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def list_all_projects() -> list[str]:
    """List all project IDs from projects_data/."""
    return [f.stem for f in PROJECTS_DATA_DIR.glob("*.yaml")]


def filter_projects(domain: str = None, level: str = None) -> list[str]:
    """Filter projects by domain and/or level."""
    projects = []
    for pid in list_all_projects():
        data = load_project(pid)
        if not data:
            continue
        if domain and data.get("domain") != domain:
            continue
        if level and data.get("difficulty", data.get("level")) != level:
            continue
        projects.append(pid)
    return projects


def load_checkpoint() -> set[str]:
    """Load already audited projects from checkpoint."""
    if CHECKPOINT_FILE.exists():
        try:
            return set(json.loads(CHECKPOINT_FILE.read_text()))
        except:
            pass
    return set()


def save_checkpoint(audited: set[str]):
    """Save checkpoint of audited projects."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(list(audited)))


def invoke_anthropic(prompt: str) -> Optional[str]:
    """Call Anthropic API with streaming."""
    try:
        from anthropic import Anthropic
    except ImportError:
        print("  ! anthropic library not installed. Run: pip install anthropic")
        return None

    if not ANTHROPIC_API_KEY:
        print("  ! ANTHROPIC_API_KEY not set")
        return None

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    print(f"  [Anthropic] Sending {len(prompt)} chars...")

    try:
        start = time.time()
        response = ""
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=64000,
            temperature=0.3,
            system="You are a Senior Systems Architect and Technical Auditor. Be critical, rigorous, and precise. Output ONLY valid JSON.",
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                response += text
        print(f"  [Anthropic] Response in {time.time() - start:.1f}s")
        return response
    except Exception as e:
        print(f"  ! Anthropic error: {e}")
        return None


def invoke_claude_cli(prompt: str) -> Optional[str]:
    """Call Claude CLI."""
    print(f"  [Claude CLI] Sending {len(prompt)} chars...")
    try:
        result = subprocess.run(
            ["claude", "-p", "--model", CLAUDE_MODEL, "--dangerously-skip-permissions"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min for batch
        )
        if result.returncode == 0:
            ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
            return ansi_escape.sub("", result.stdout).strip()
        print(f"  ! Claude CLI error: {result.stderr}")
        return None
    except Exception as e:
        print(f"  ! Claude CLI error: {e}")
        return None


def invoke_gemini_proxy(prompt: str) -> Optional[str]:
    """Call Gemini via local proxy."""
    import urllib.request
    import urllib.error

    print(f"  [Gemini Proxy] Sending {len(prompt)} chars...")
    try:
        data = json.dumps({
            "model": "gemini_cli/gemini-3-flash-preview",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 64000,
        }).encode()

        req = urllib.request.Request(
            f"{GEMINI_PROXY_URL}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GEMINI_PROXY_KEY}",
            },
        )

        start = time.time()
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode())
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"  [Gemini Proxy] Response in {time.time() - start:.1f}s")
            return content
    except Exception as e:
        print(f"  ! Gemini proxy error: {e}")
        return None


def invoke_llm(prompt: str) -> Optional[str]:
    """Route to appropriate LLM provider."""
    if LLM_PROVIDER == "claude-cli":
        return invoke_claude_cli(prompt)
    elif LLM_PROVIDER == "gemini-proxy":
        return invoke_gemini_proxy(prompt)
    else:
        return invoke_anthropic(prompt)


def audit_single_project(project_id: str, project_data: dict) -> Optional[dict]:
    """Audit a single project spec (for non-batch mode)."""
    prompt = f"""
{AUDIT_CONTEXT}

PROJECT TO AUDIT: {project_id}

```yaml
{yaml.dump(project_data, allow_unicode=True, default_flow_style=False)}
```

TASK:
1. Analyze this project spec
2. Identify ONLY real issues (not nitpicks)
3. If the spec is already good, just say so - don't force changes

OUTPUT FORMAT (JSON only):
{{
  "project_id": "{project_id}",
  "overall_score": <1-10>,
  "verdict": "good|needs_fix",
  "issues": [
    // Only include REAL issues, not minor nitpicks
    {{"type": "technical|measurability|progression|completeness|security|performance", "severity": "critical|major|minor", "location": "field", "description": "what's wrong", "suggestion": "how to fix"}}
  ],
  "strengths": ["what's done well"],
  "critique": "1-2 sentences",
  "fixed_yaml": "ONLY if verdict is 'needs_fix', otherwise put null"
}}

IMPORTANT: If the spec is solid (score >= 7), set verdict to "good" and fixed_yaml to null. Don't waste time on trivial improvements.
"""

    response = invoke_llm(prompt)
    if not response:
        return None

    try:
        json_match = re.search(r"(\{.*\})", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
    except json.JSONDecodeError as e:
        print(f"  ! JSON parse error: {e}")

    debug_path = OUTPUT_DIR / "debug" / f"{project_id}_{int(time.time())}.txt"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(response)
    print(f"  ! Saved raw response to {debug_path}")
    return None


def audit_batch(batch: list[tuple[str, dict]]) -> list[dict]:
    """Audit multiple projects in a single API call."""
    project_ids = [p[0] for p in batch]
    print(f"  Auditing batch: {', '.join(project_ids)}")

    # Build batch payload
    projects_payload = []
    for project_id, project_data in batch:
        projects_payload.append({
            "project_id": project_id,
            "yaml": yaml.dump(project_data, allow_unicode=True, default_flow_style=False)
        })

    prompt = f"""
{AUDIT_CONTEXT}

BATCH OF PROJECTS TO AUDIT ({len(batch)} projects):

```json
{json.dumps(projects_payload, indent=2)}
```

TASK:
For EACH project, analyze and provide audit results.

OUTPUT FORMAT (JSON array only):
[
  {{
    "project_id": "project-id-here",
    "overall_score": <1-10>,
    "verdict": "good|needs_fix",
    "issues": [
      // Only REAL issues
    ],
    "strengths": ["what's done well"],
    "critique": "1-2 sentences",
    "fixed_yaml": "ONLY if verdict is 'needs_fix', otherwise null"
  }}
]

IMPORTANT:
- If a spec is already good (score >= 7), set verdict to "good" and fixed_yaml to null
- Don't invent problems or make trivial changes
- Output MUST be a valid JSON array with ALL {len(batch)} projects
"""

    response = invoke_llm(prompt)
    if not response:
        return []

    try:
        # Try to find JSON array
        json_match = re.search(r"(\[.*\])", response, re.DOTALL)
        if json_match:
            results = json.loads(json_match.group(1))
            if isinstance(results, list):
                return results
    except json.JSONDecodeError as e:
        print(f"  ! JSON parse error: {e}")

    # Save debug
    debug_path = OUTPUT_DIR / "debug" / f"batch_{int(time.time())}.txt"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(response)
    print(f"  ! Saved raw response to {debug_path}")
    return []


def save_audit_result(project_id: str, result: dict):
    """Save audit result to file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    verdict = result.get('verdict', 'needs_fix')
    issues = result.get('issues', [])
    score = result.get('overall_score', '?')
    fixed_yaml = result.get('fixed_yaml')

    # Save markdown report
    md_path = OUTPUT_DIR / f"{project_id}.md"

    if verdict == 'good' or score >= 7:
        md_content = f"""# Audit Report: {project_id}

**Score:** {score}/10
**Verdict:** ✅ GOOD - No significant issues

## Summary
{result.get('critique', 'Spec is solid.')}

## Strengths
{chr(10).join(f'- {s}' for s in result.get('strengths', [])) or '- Well structured'}

## Minor Issues (if any)
{chr(10).join(f'- {i.get("description", str(i))}' for i in issues if isinstance(i, dict) and i.get('severity') == 'minor') or '- None'}
"""
    else:
        md_content = f"""# Audit Report: {project_id}

**Score:** {score}/10
**Verdict:** ⚠️ NEEDS FIX

## Summary
{result.get('critique', 'Issues found.')}

## Strengths
{chr(10).join(f'- {s}' for s in result.get('strengths', [])) or '- None identified'}

## Issues ({len(issues)})

| Type | Severity | Location | Issue | Suggestion |
|------|----------|----------|-------|------------|
{chr(10).join(f"| {i.get('type', '?')} | {i.get('severity', '?')} | {i.get('location', '?')} | {str(i.get('description', '?'))[:50]} | {str(i.get('suggestion', '-'))[:50]} |" for i in issues if isinstance(i, dict)) if issues else '| - | - | - | - | - |'}

## Fixed YAML
```yaml
{fixed_yaml or '# No fix provided'}
```
"""

    md_path.write_text(md_content)

    # Only save fixed YAML if there's actually a fix
    if fixed_yaml and verdict == 'needs_fix':
        yaml_path = OUTPUT_DIR / f"{project_id}_fixed.yaml"
        yaml_path.write_text(fixed_yaml)

    verdict_icon = "✅" if verdict == 'good' or score >= 7 else "⚠️"
    print(f"  {verdict_icon} {project_id}: Score {score}/10, {len(issues)} issues, {verdict}")


def main():
    parser = argparse.ArgumentParser(description="Audit project specs from projects_data/")
    parser.add_argument("--projects", nargs="+", help="Specific project IDs to audit")
    parser.add_argument("--domain", help="Filter by domain (e.g., systems, ai-ml)")
    parser.add_argument("--level", help="Filter by level (beginner, intermediate, advanced, expert)")
    parser.add_argument("--all", action="store_true", help="Audit all projects")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Projects per API call (default: {DEFAULT_BATCH_SIZE}, set to 1 for no batching)")
    parser.add_argument("--dry-run", action="store_true", help="List projects without auditing")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--provider", choices=["anthropic", "claude-cli", "gemini-proxy"],
                        default="anthropic", help="LLM provider to use")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "audited-specs", help="Output directory")
    args = parser.parse_args()

    global LLM_PROVIDER, OUTPUT_DIR
    LLM_PROVIDER = args.provider
    OUTPUT_DIR = args.output

    # Determine which projects to audit
    if args.projects:
        projects = args.projects
    elif args.all:
        projects = list_all_projects()
    else:
        projects = filter_projects(domain=args.domain, level=args.level)

    if not projects:
        print("No projects found matching criteria.")
        return

    # Resume from checkpoint
    audited = load_checkpoint() if args.resume else set()
    if audited:
        print(f"Resuming: {len(audited)} already audited")

    # Filter out already audited
    projects = [p for p in projects if p not in audited]

    if not projects:
        print("All projects already audited!")
        return

    print(f"=== SPEC AUDITOR ===")
    print(f"Provider: {LLM_PROVIDER}")
    print(f"Projects to audit: {len(projects)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    if args.dry_run:
        print("Projects (dry-run):")
        for p in sorted(projects):
            data = load_project(p)
            if data:
                print(f"  - {p} [{data.get('domain', '?')}/{data.get('difficulty', '?')}]")
        return

    # Prepare projects with data
    projects_with_data = []
    for pid in sorted(projects):
        data = load_project(pid)
        if data:
            projects_with_data.append((pid, data))

    success = 0
    failed = 0

    if args.batch_size <= 1:
        # Single project mode
        for i, (project_id, project_data) in enumerate(projects_with_data, 1):
            print(f"\n[{i}/{len(projects_with_data)}] Auditing: {project_id}")

            result = audit_single_project(project_id, project_data)
            if result:
                save_audit_result(project_id, result)
                audited.add(project_id)
                save_checkpoint(audited)
                success += 1
            else:
                print(f"  ! Audit failed")
                failed += 1
    else:
        # Batch mode
        total_batches = (len(projects_with_data) + args.batch_size - 1) // args.batch_size

        for batch_idx in range(0, len(projects_with_data), args.batch_size):
            batch = projects_with_data[batch_idx:batch_idx + args.batch_size]
            batch_num = batch_idx // args.batch_size + 1

            print(f"\n=== Batch {batch_num}/{total_batches} ({len(batch)} projects) ===")

            results = audit_batch(batch)

            if results:
                for result in results:
                    pid = result.get('project_id')
                    if pid:
                        save_audit_result(pid, result)
                        audited.add(pid)
                        success += 1
            else:
                print(f"  ! Batch failed, falling back to individual audits...")
                # Fallback: audit individually
                for project_id, project_data in batch:
                    result = audit_single_project(project_id, project_data)
                    if result:
                        save_audit_result(project_id, result)
                        audited.add(project_id)
                        success += 1
                    else:
                        failed += 1

            save_checkpoint(audited)

            # Small delay between batches
            if batch_idx + args.batch_size < len(projects_with_data):
                print("  Pausing 2s before next batch...")
                time.sleep(2)

    print(f"\n=== DONE ===")
    print(f"Success: {success}, Failed: {failed}")
    print(f"Total audited: {len(audited)}")


if __name__ == "__main__":
    main()
