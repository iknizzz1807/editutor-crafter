#!/usr/bin/env python3
"""
Spec Auditor - Audit project specs from projects_data/ using LLM.

Usage:
    python3 scripts/spec_auditor.py --projects build-react build-git
    python3 scripts/spec_auditor.py --domain systems
    python3 scripts/spec_auditor.py --level expert
    python3 scripts/spec_auditor.py --all
    python3 scripts/spec_auditor.py --dry-run  # Just list projects to audit

Features:
- Reads from projects_data/*.yaml (source of truth)
- Supports multiple LLM providers (Anthropic, Claude CLI, Gemini proxy)
- Batch processing with retry logic
- Detailed audit reports with critique and fixed YAML
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

# LLM Provider settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic, claude-cli, gemini-proxy
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
GEMINI_PROXY_URL = os.getenv("GEMINI_PROXY_URL", "http://127.0.0.1:7999/v1")
GEMINI_PROXY_KEY = os.getenv("GEMINI_PROXY_KEY", "mythong2005")


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
            max_tokens=32000,
            temperature=0.3,  # Lower temp for consistent auditing
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
            timeout=300,
        )
        if result.returncode == 0:
            # Strip ANSI codes
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
            "max_tokens": 32000,
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
        with urllib.request.urlopen(req, timeout=300) as resp:
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


def audit_project(project_id: str, project_data: dict) -> Optional[dict]:
    """Audit a single project spec."""
    prompt = f"""
{AUDIT_CONTEXT}

PROJECT TO AUDIT: {project_id}

```yaml
{yaml.dump(project_data, allow_unicode=True, default_flow_style=False)}
```

TASK:
1. Analyze this project spec thoroughly
2. Identify ALL issues (technical errors, vague criteria, missing concepts, etc.)
3. Provide a FIXED version

OUTPUT FORMAT (JSON only, no markdown):
{{
  "project_id": "{project_id}",
  "overall_score": <1-10>,
  "issues": [
    {{"type": "technical|measurability|progression|completeness|security|performance", "severity": "critical|major|minor", "location": "field name", "description": "what's wrong", "suggestion": "how to fix"}}
  ],
  "strengths": ["what's done well"],
  "critique": "2-3 sentence summary",
  "fixed_yaml": "complete corrected YAML here"
}}
"""

    response = invoke_llm(prompt)
    if not response:
        return None

    # Extract JSON
    try:
        json_match = re.search(r"(\{.*\})", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
    except json.JSONDecodeError as e:
        print(f"  ! JSON parse error: {e}")

    # Save raw response for debugging
    debug_path = OUTPUT_DIR / "debug" / f"{project_id}_{int(time.time())}.txt"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(response)
    print(f"  ! Saved raw response to {debug_path}")
    return None


def save_audit_result(project_id: str, result: dict, original: dict):
    """Save audit result to file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save markdown report
    md_path = OUTPUT_DIR / f"{project_id}.md"
    md_content = f"""# Audit Report: {project_id}

**Overall Score:** {result.get('overall_score', 'N/A')}/10

## Summary
{result.get('critique', 'No summary available.')}

## Strengths
{chr(10).join(f'- {s}' for s in result.get('strengths', [])) or '- None identified'}

## Issues Found ({len(result.get('issues', []))})

| Type | Severity | Location | Issue | Suggestion |
|------|----------|----------|-------|------------|
{chr(10).join(f"| {i.get('type', '?')} | {i.get('severity', '?')} | {i.get('location', '?')} | {i.get('description', '?')} | {i.get('suggestion', '-')} |" for i in result.get('issues', []))}

## Fixed YAML
```yaml
{result.get('fixed_yaml', '# No fix provided')}
```
"""
    md_path.write_text(md_content)

    # Also save the fixed YAML separately
    if result.get('fixed_yaml'):
        yaml_path = OUTPUT_DIR / f"{project_id}_fixed.yaml"
        yaml_path.write_text(result['fixed_yaml'])

    print(f"  ✓ Saved: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit project specs from projects_data/")
    parser.add_argument("--projects", nargs="+", help="Specific project IDs to audit")
    parser.add_argument("--domain", help="Filter by domain (e.g., systems, ai-ml)")
    parser.add_argument("--level", help="Filter by level (beginner, intermediate, advanced, expert)")
    parser.add_argument("--all", action="store_true", help="Audit all projects")
    parser.add_argument("--dry-run", action="store_true", help="List projects without auditing")
    parser.add_argument("--provider", choices=["anthropic", "claude-cli", "gemini-proxy"],
                        default=LLM_PROVIDER, help="LLM provider to use")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output directory")
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

    print(f"=== SPEC AUDITOR ===")
    print(f"Provider: {LLM_PROVIDER}")
    print(f"Projects to audit: {len(projects)}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    if args.dry_run:
        print("Projects (dry-run):")
        for p in sorted(projects):
            data = load_project(p)
            if data:
                print(f"  - {p} [{data.get('domain', '?')}/{data.get('difficulty', '?')}]")
        return

    # Audit each project
    success = 0
    failed = 0
    for i, project_id in enumerate(sorted(projects), 1):
        print(f"\n[{i}/{len(projects)}] Auditing: {project_id}")

        project_data = load_project(project_id)
        if not project_data:
            print(f"  ! Project not found: {project_id}")
            failed += 1
            continue

        result = audit_project(project_id, project_data)
        if result:
            save_audit_result(project_id, result, project_data)
            score = result.get('overall_score', '?')
            issues = len(result.get('issues', []))
            print(f"  Score: {score}/10, Issues: {issues}")
            success += 1
        else:
            print(f"  ! Audit failed")
            failed += 1

    print(f"\n=== DONE ===")
    print(f"Success: {success}, Failed: {failed}")


if __name__ == "__main__":
    main()
