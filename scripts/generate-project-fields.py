#!/usr/bin/env python3
"""
Generate missing project-level fields (essence, why_important, learning_outcomes,
skills, resources) for all projects using Claude Code CLI or Gemini API.

Usage:
    # Using Claude Code CLI
    python3 scripts/generate-project-fields.py --provider claude

    # With web research (for accurate resource URLs)
    python3 scripts/generate-project-fields.py --provider claude --research

    # Using Gemini API
    python3 scripts/generate-project-fields.py --provider gemini

    # Process specific project
    python3 scripts/generate-project-fields.py --provider claude --project build-interpreter

    # Dry run (preview prompt without writing)
    python3 scripts/generate-project-fields.py --dry-run

    # Resume after interruption (automatic)
    python3 scripts/generate-project-fields.py --provider claude

    # Reset progress and start fresh
    python3 scripts/generate-project-fields.py --provider claude --reset

    # Parallel workers for Claude
    python3 scripts/generate-project-fields.py --provider claude --workers 3

    # Save every N projects
    python3 scripts/generate-project-fields.py --provider claude --save-every 5

Providers:
    claude  — Uses `claude -p` CLI (Claude Max subscription)
    gemini  — Uses Gemini API (requires GEMINI_API_KEY env var)

Requirements:
    claude provider: Claude Code CLI installed and authenticated
    gemini provider: pip install google-genai pyyaml && export GEMINI_API_KEY=...
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / ".." / "data"
YAML_PATH = DATA_DIR / "projects.yaml"
PROGRESS_PATH = DATA_DIR / "project_fields_progress.json"

ALL_FIELDS = ("essence", "why_important", "learning_outcomes", "skills", "resources")


# ---------------------------------------------------------------------------
# YAML helpers – preserve block-scalar style for multiline strings
# ---------------------------------------------------------------------------

class _LiteralStr(str):
    """Tag so the YAML dumper emits a literal block scalar ( | )."""

def _literal_representer(dumper, data):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)

yaml.add_representer(_LiteralStr, _literal_representer)


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_progress(path: Path, progress: dict):
    # Deduplicate before saving
    progress["completed"] = list(dict.fromkeys(progress["completed"]))
    progress["failed"] = list(dict.fromkeys(progress["failed"]))
    # Remove items from failed if they later succeeded
    completed = set(progress["completed"])
    progress["failed"] = [f for f in progress["failed"] if f not in completed]
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


# ---------------------------------------------------------------------------
# Determine which fields are missing for a project
# ---------------------------------------------------------------------------

def get_missing_fields(project: dict) -> list[str]:
    """Return list of field names that are missing or empty for this project.
    'essence' is always included since it's a new field."""
    missing = []
    for field in ALL_FIELDS:
        val = project.get(field)
        if field == "essence":
            # Always generate essence (new field)
            missing.append(field)
        elif field in ("learning_outcomes", "skills"):
            if not val or not isinstance(val, list) or len(val) == 0:
                missing.append(field)
        elif field == "resources":
            if not val or not isinstance(val, list) or len(val) == 0:
                missing.append(field)
        else:
            # String fields: why_important
            if not val or (isinstance(val, str) and not val.strip()):
                missing.append(field)
    return missing


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _summarize_milestones(project: dict) -> str:
    """Build a brief summary of milestones for context."""
    milestones = project.get("milestones", [])
    if not milestones:
        return "(no milestones)"
    lines = []
    for i, m in enumerate(milestones):
        name = m.get("name", f"Milestone {i}")
        desc = m.get("description", "")
        if desc and len(desc) > 120:
            desc = desc[:120] + "..."
        lines.append(f"  {i+1}. {name}: {desc}")
    return "\n".join(lines)


def _format_existing_values(project: dict) -> str:
    """Format existing field values as reference context."""
    parts = []
    for field in ALL_FIELDS:
        val = project.get(field)
        if not val:
            continue
        if field == "resources" and isinstance(val, list):
            res_lines = []
            for r in val:
                if isinstance(r, dict):
                    res_lines.append(f"  - {r.get('name', '?')} | {r.get('url', '?')} | {r.get('type', '?')}")
                else:
                    res_lines.append(f"  - {r}")
            parts.append(f"EXISTING {field.upper()}:\n" + "\n".join(res_lines))
        elif isinstance(val, list):
            parts.append(f"EXISTING {field.upper()}: {', '.join(str(v) for v in val)}")
        else:
            parts.append(f"EXISTING {field.upper()}: {val}")
    return "\n".join(parts) if parts else ""


def build_prompt(project: dict, missing_fields: list[str], research: bool = False) -> str:
    project_name = project.get("name", "")
    project_desc = project.get("description", "")
    difficulty = project.get("difficulty", "")
    milestones_summary = _summarize_milestones(project)
    existing_values = _format_existing_values(project)

    # Languages context
    languages = project.get("languages", {})
    if isinstance(languages, dict):
        rec = languages.get("recommended", languages.get("primary", []))
        if isinstance(rec, list):
            lang_str = ", ".join(rec)
        else:
            lang_str = str(rec)
    elif isinstance(languages, list):
        lang_str = ", ".join(str(l) for l in languages)
    else:
        lang_str = str(languages) if languages else ""

    # Build field-specific instructions
    field_instructions = []
    field_markers = []

    if "essence" in missing_fields:
        field_instructions.append("""ESSENCE (1-2 sentences):
- Describe the fundamental technical nature and core challenge of this project
- Focus on what makes it interesting from an engineering/CS perspective
- NOT a sales pitch — describe the core technical problem being solved
- Example for "Multiplayer Game Server": "Real-time state synchronization, input validation, and conflict resolution across unreliable network connections"
- Example for "Database Engine": "B-tree indexing, page-based storage management, and ACID transaction guarantees through write-ahead logging\"""")
        field_markers.append("[ESSENCE]\nReal-time state synchronization...\n[/ESSENCE]")

    if "why_important" in missing_fields:
        field_instructions.append("""WHY_IMPORTANT (1-2 sentences):
- Explain why building this project matters for a developer's growth
- Focus on practical skills gained and career relevance
- Avoid generic platitudes — be specific to this project""")
        field_markers.append("[WHY_IMPORTANT]\nBuilding this teaches you...\n[/WHY_IMPORTANT]")

    if "learning_outcomes" in missing_fields:
        field_instructions.append("""LEARNING_OUTCOMES (4-8 items, one per line):
- Specific, measurable things the developer will learn
- Start each with an action verb (Implement, Design, Build, Debug, etc.)
- Be concrete — not "understand databases" but "Implement B-tree indexing with node splitting and merging\"""")
        field_markers.append("[LEARNING_OUTCOMES]\nImplement X with Y\nDesign Z for W\n[/LEARNING_OUTCOMES]")

    if "skills" in missing_fields:
        field_instructions.append("""SKILLS (4-8 short tags, one per line):
- Technical skill tags gained from this project
- Short phrases (1-4 words each)
- Example: "TCP/IP Networking", "State Management", "Binary Protocols\"""")
        field_markers.append("[SKILLS]\nTCP/IP Networking\nState Management\n[/SKILLS]")

    if "resources" in missing_fields:
        field_instructions.append("""RESOURCES (3-5 items, one per line, format: name|url|type):
- Each line: resource_name|url|type
- type is one of: article, tutorial, documentation, paper, video, book, tool
- Use REAL, verified URLs that actually exist
- Prefer official documentation, well-known tutorials, authoritative sources
- Example: "Redis Documentation|https://redis.io/docs/|documentation\"""")
        field_markers.append("[RESOURCES]\nOfficial Docs|https://...|documentation\nTutorial Name|https://...|tutorial\n[/RESOURCES]")

    research_instruction = ""
    if research:
        research_instruction = """
IMPORTANT: Before generating fields, search the web for:
- Official documentation and tutorials for the technologies in this project
- Best learning resources and authoritative references
- Verify that all resource URLs are real and accessible
Use what you find to ensure accuracy, especially for resource URLs.
"""

    existing_section = ""
    if existing_values:
        existing_section = f"""
EXISTING VALUES (for reference — do NOT repeat these, generate fresh content):
{existing_values}
"""

    return f"""You are an expert programming tutor building a learning roadmap. Generate the missing fields for the project below.

=== PROJECT CONTEXT ===
PROJECT: {project_name}
DESCRIPTION: {project_desc}
DIFFICULTY: {difficulty}
LANGUAGES: {lang_str}

MILESTONES:
{milestones_summary}
{existing_section}
=== FIELD REQUIREMENTS ===
{research_instruction}
Generate ONLY the following missing fields:

{chr(10).join(field_instructions)}

=== OUTPUT FORMAT ===

{chr(10).join(field_markers)}

=== RULES ===
- Output ONLY the marker blocks requested above
- Do NOT add any text before the first marker or after the last marker
- Do NOT wrap content in markdown fences
- For RESOURCES: use the exact format name|url|type (pipe-separated), one per line
- For LEARNING_OUTCOMES and SKILLS: one item per line, no bullet markers or numbering
- For ESSENCE and WHY_IMPORTANT: plain text, 1-2 sentences
- Ensure technical accuracy
"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(text: str, missing_fields: list[str]) -> dict | None:
    """Parse [FIELD]...[/FIELD] markers from LLM response."""
    result = {}
    marker_map = {
        "essence": "ESSENCE",
        "why_important": "WHY_IMPORTANT",
        "learning_outcomes": "LEARNING_OUTCOMES",
        "skills": "SKILLS",
        "resources": "RESOURCES",
    }

    for field in missing_fields:
        marker = marker_map[field]
        pattern = rf"\[{marker}\]\s*\n?(.*?)\s*\[/{marker}\]"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            print(f"  WARNING: Missing [{marker}] marker in response")
            continue
        raw = match.group(1).strip()
        if not raw:
            print(f"  WARNING: Empty [{marker}] content")
            continue

        if field in ("essence", "why_important"):
            result[field] = raw
        elif field == "learning_outcomes":
            items = [line.strip() for line in raw.split("\n") if line.strip()]
            # Strip leading bullets/numbers
            items = [re.sub(r"^[\d]+[.)]\s*|^[-*•]\s*", "", item).strip() for item in items]
            items = [item for item in items if item]
            result[field] = items
        elif field == "skills":
            items = [line.strip() for line in raw.split("\n") if line.strip()]
            items = [re.sub(r"^[\d]+[.)]\s*|^[-*•]\s*", "", item).strip() for item in items]
            items = [item for item in items if item]
            result[field] = items
        elif field == "resources":
            resources = []
            for line in raw.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Strip leading bullets/numbers
                line = re.sub(r"^[\d]+[.)]\s*|^[-*•]\s*", "", line).strip()
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    resources.append({
                        "name": parts[0],
                        "url": parts[1],
                        "type": parts[2].lower(),
                    })
                elif len(parts) == 2:
                    resources.append({
                        "name": parts[0],
                        "url": parts[1],
                        "type": "article",
                    })
            if resources:
                result[field] = resources

    if not result:
        return None
    return result


def validate_fields(fields: dict, missing_fields: list[str]) -> list[str]:
    """Return list of validation warnings (empty = OK)."""
    warnings = []
    for field in missing_fields:
        if field not in fields:
            warnings.append(f"{field}: not generated")
            continue
        val = fields[field]
        if field in ("essence", "why_important"):
            if len(val) < 20:
                warnings.append(f"{field}: too short ({len(val)} chars, want >=20)")
        elif field == "learning_outcomes":
            if len(val) < 3:
                warnings.append(f"{field}: too few items ({len(val)}, want >=3)")
        elif field == "skills":
            if len(val) < 3:
                warnings.append(f"{field}: too few items ({len(val)}, want >=3)")
        elif field == "resources":
            if len(val) < 2:
                warnings.append(f"{field}: too few items ({len(val)}, want >=2)")
    return warnings


# ---------------------------------------------------------------------------
# Provider: Claude Code CLI
# ---------------------------------------------------------------------------

def call_claude(prompt: str, model: str = "sonnet", research: bool = False, max_retries: int = 3) -> str | None:
    """Call Claude via `claude -p` CLI."""
    claude_path = shutil.which("claude")
    if not claude_path:
        print("ERROR: `claude` CLI not found in PATH. Install Claude Code first.")
        return None

    cmd = [claude_path, "-p", "--output-format", "text"]

    if model:
        cmd.extend(["--model", model])

    if research:
        cmd.extend(["--allowedTools", "WebSearch", "WebFetch"])
    else:
        cmd.extend(["--tools", ""])

    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=180,
            )

            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()

            stderr = result.stderr.strip()
            if "overloaded" in stderr.lower() or "rate" in stderr.lower():
                wait = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif stderr:
                print(f"  Claude CLI error: {stderr[:200]}")
                if attempt < max_retries - 1:
                    time.sleep(10)
            else:
                print(f"  Empty response (exit code {result.returncode})")
                if attempt < max_retries - 1:
                    time.sleep(5)

        except subprocess.TimeoutExpired:
            print(f"  Timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(5)
        except Exception as e:
            print(f"  Exception: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    return None


# ---------------------------------------------------------------------------
# Provider: Gemini API
# ---------------------------------------------------------------------------

def init_gemini(api_key: str, model_name: str = "gemini-2.0-flash", use_grounding: bool = True):
    """Initialize Gemini client and return (client, model_name, tools_config)."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("ERROR: google-genai not installed. Run: pip install google-genai")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    tool_config = None
    if use_grounding:
        try:
            tool_config = types.Tool(google_search=types.GoogleSearch())
        except Exception:
            print("WARNING: Google Search grounding not available, proceeding without it.")
            tool_config = None

    return client, model_name, tool_config


def call_gemini(client, model_name: str, prompt: str, tool_config=None, max_retries: int = 3) -> str | None:
    """Call Gemini API with retries."""
    from google.genai import types

    tools = [tool_config] if tool_config else None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=tools,
                ) if tools else None,
            )
            if response.text:
                return response.text
            if response.candidates:
                parts = response.candidates[0].content.parts
                return "".join(p.text for p in parts if hasattr(p, "text"))
            return None
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif "500" in err_str or "503" in err_str:
                wait = 10 * (attempt + 1)
                print(f"  Server error, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API error: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(5)
    return None


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def collect_projects(data: dict, project_filter: str | None = None) -> list[tuple[str, dict]]:
    """Collect all (project_id, project) tuples from expert_projects."""
    results = []
    expert_projects = data.get("expert_projects", {})
    for proj_id, project in expert_projects.items():
        if project_filter and proj_id != project_filter:
            continue
        results.append((proj_id, project))
    return results


def process_project_claude(project: dict, missing_fields: list[str],
                           model: str, research: bool, dry_run: bool) -> dict | None:
    """Process a single project using Claude CLI."""
    prompt = build_prompt(project, missing_fields, research=research)

    if dry_run:
        print(f"  [DRY RUN] Would send prompt ({len(prompt)} chars)")
        print(f"  Missing fields: {', '.join(missing_fields)}")
        return None

    response_text = call_claude(prompt, model=model, research=research)
    if not response_text:
        print("  ERROR: No response from Claude CLI")
        return None

    return _parse_and_validate(response_text, missing_fields)


def process_project_gemini(client, model_name: str, tool_config,
                           project: dict, missing_fields: list[str], dry_run: bool) -> dict | None:
    """Process a single project using Gemini API."""
    prompt = build_prompt(project, missing_fields, research=(tool_config is not None))

    if dry_run:
        print(f"  [DRY RUN] Would send prompt ({len(prompt)} chars)")
        print(f"  Missing fields: {', '.join(missing_fields)}")
        return None

    response_text = call_gemini(client, model_name, prompt, tool_config)
    if not response_text:
        print("  ERROR: No response from Gemini API")
        return None

    return _parse_and_validate(response_text, missing_fields)


def _parse_and_validate(response_text: str, missing_fields: list[str]) -> dict | None:
    """Parse response markers and validate fields."""
    parsed = parse_response(response_text, missing_fields)
    if not parsed:
        print("  ERROR: Could not parse any response markers")
        debug_path = DATA_DIR / "project_fields_debug_last_response.txt"
        with open(debug_path, "w") as f:
            f.write(response_text)
        print(f"  Raw response saved to {debug_path}")
        return None

    warnings = validate_fields(parsed, missing_fields)
    for w in warnings:
        print(f"  WARNING: {w}")

    return parsed


def apply_fields(project: dict, fields: dict):
    """Apply generated fields to project dict in-place."""
    for field, value in fields.items():
        if field in ("essence", "why_important"):
            project[field] = _LiteralStr(value) if "\n" in value else value
        elif field in ("learning_outcomes", "skills"):
            project[field] = value
        elif field == "resources":
            project[field] = value


def save_yaml(data: dict, path: Path):
    """Save YAML with proper formatting."""
    with open(path, "w") as f:
        yaml.dump(
            data,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=120,
        )


# ---------------------------------------------------------------------------
# Parallel processing (Claude provider)
# ---------------------------------------------------------------------------

def _worker_claude(task: tuple, model: str, research: bool) -> tuple[str, dict | None, list[str], str]:
    """Worker function for parallel Claude processing.
    Returns (proj_id, fields, missing_fields, proj_name)."""
    proj_id, project, missing_fields = task
    proj_name = project.get("name", "unknown")

    prompt = build_prompt(project, missing_fields, research=research)
    response_text = call_claude(prompt, model=model, research=research)

    if not response_text:
        return proj_id, None, missing_fields, proj_name

    fields = _parse_and_validate(response_text, missing_fields)
    return proj_id, fields, missing_fields, proj_name


def process_parallel_claude(pending: list, data: dict, progress: dict, completed_set: set, args):
    """Process projects in parallel using Claude CLI."""
    success_count = 0
    fail_count = 0
    processed_since_save = 0

    expert_projects = data.get("expert_projects", {})

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for proj_id, project, missing_fields in pending:
            task = (proj_id, project, missing_fields)
            future = executor.submit(_worker_claude, task, args.model, args.research)
            futures[future] = proj_id

        for future in as_completed(futures):
            proj_id = futures[future]
            try:
                returned_id, fields, missing_fields, proj_name = future.result()

                if fields:
                    apply_fields(expert_projects[returned_id], fields)
                    progress["completed"].append(returned_id)
                    completed_set.add(returned_id)
                    success_count += 1
                    processed_since_save += 1
                    generated = ", ".join(fields.keys())
                    print(f"  OK {proj_name} [{generated}]")
                else:
                    progress["failed"].append(returned_id)
                    fail_count += 1
                    print(f"  FAIL {proj_name}")

                save_progress(PROGRESS_PATH, progress)

                if processed_since_save >= args.save_every:
                    print(f"  [Saving YAML checkpoint...]")
                    save_yaml(data, YAML_PATH)
                    processed_since_save = 0

            except Exception as e:
                print(f"  EXCEPTION for {proj_id}: {e}")
                traceback.print_exc()
                progress["failed"].append(proj_id)
                fail_count += 1
                save_progress(PROGRESS_PATH, progress)

    return success_count, fail_count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate missing project-level fields using Claude Code CLI or Gemini API"
    )
    parser.add_argument("--provider", type=str, default="claude", choices=["claude", "gemini"],
                        help="LLM provider (default: claude)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (claude: sonnet/opus/haiku, gemini: gemini-2.0-flash)")
    parser.add_argument("--research", action="store_true",
                        help="Enable web research (for accurate resource URLs)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--project", type=str, help="Process only this project ID")
    parser.add_argument("--reset", action="store_true", help="Reset progress and start fresh")
    parser.add_argument("--rpm", type=int, default=15,
                        help="Rate limit: requests per minute (gemini only, default: 15)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (claude only, default: 1)")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save YAML every N projects (default: 10)")
    args = parser.parse_args()

    # Default model per provider
    if args.model is None:
        args.model = "sonnet" if args.provider == "claude" else "gemini-2.0-flash"

    # Provider-specific checks
    if args.provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key and not args.dry_run:
            print("ERROR: GEMINI_API_KEY environment variable not set")
            sys.exit(1)
    elif args.provider == "claude" and not args.dry_run:
        if not shutil.which("claude"):
            print("ERROR: `claude` CLI not found in PATH. Install Claude Code first.")
            sys.exit(1)

    # Load data
    print(f"Loading {YAML_PATH}...")
    with open(YAML_PATH) as f:
        data = yaml.safe_load(f)

    # Backup original YAML before any modifications
    if not args.dry_run:
        backup_path = YAML_PATH.with_suffix(".yaml.bak")
        if not backup_path.exists():
            shutil.copy2(YAML_PATH, backup_path)
            print(f"Backup saved to {backup_path}")
        else:
            print(f"Backup already exists: {backup_path}")

    # Load/reset progress
    if args.reset and PROGRESS_PATH.exists():
        os.remove(PROGRESS_PATH)
        print("Progress reset.")
    progress = load_progress(PROGRESS_PATH)
    # Deduplicate completed list and clear stale failed entries
    # (failed items are retried on next run, so reset the list each run)
    progress["completed"] = list(dict.fromkeys(progress["completed"]))
    progress["failed"] = []
    completed_set = set(progress["completed"])

    # Collect projects and determine missing fields
    all_projects = collect_projects(data, args.project)
    total = len(all_projects)

    # Build pending list: (proj_id, project, missing_fields)
    pending = []
    field_stats = {f: 0 for f in ALL_FIELDS}
    for proj_id, project in all_projects:
        if proj_id in completed_set:
            continue
        missing = get_missing_fields(project)
        if missing:
            pending.append((proj_id, project, missing))
            for f in missing:
                field_stats[f] += 1

    print(f"\nProvider: {args.provider} (model: {args.model})")
    print(f"Research: {'ON' if args.research else 'OFF'}")
    print(f"Total projects: {total}")
    print(f"Already completed: {len(completed_set)}")
    print(f"Pending: {len(pending)}")
    print(f"\nMissing field counts:")
    for field, count in field_stats.items():
        print(f"  {field}: {count}")

    if not pending:
        print("\nNothing to process!")
        return

    # --------------- Claude provider ---------------
    if args.provider == "claude":
        if args.workers > 1 and not args.dry_run:
            print(f"\nWorkers: {args.workers}")
            print(f"Processing {len(pending)} projects in parallel...\n")
            success_count, fail_count = process_parallel_claude(
                pending, data, progress, completed_set, args
            )
        else:
            success_count = 0
            fail_count = 0
            processed_since_save = 0

            print(f"\nProcessing {len(pending)} projects...\n")

            expert_projects = data.get("expert_projects", {})

            for i, (proj_id, project, missing_fields) in enumerate(pending):
                proj_name = project.get("name", "unknown")
                print(f"[{i+1}/{len(pending)}] {proj_id} ({', '.join(missing_fields)})")

                try:
                    new_fields = process_project_claude(
                        project, missing_fields, model=args.model,
                        research=args.research, dry_run=args.dry_run,
                    )

                    if new_fields:
                        apply_fields(expert_projects[proj_id], new_fields)
                        progress["completed"].append(proj_id)
                        completed_set.add(proj_id)
                        success_count += 1
                        processed_since_save += 1
                        generated = ", ".join(new_fields.keys())
                        print(f"  OK [{generated}]")
                    elif args.dry_run:
                        pass
                    else:
                        progress["failed"].append(proj_id)
                        fail_count += 1

                    save_progress(PROGRESS_PATH, progress)

                    if not args.dry_run and processed_since_save >= args.save_every:
                        print(f"  [Saving YAML checkpoint...]")
                        save_yaml(data, YAML_PATH)
                        processed_since_save = 0

                except KeyboardInterrupt:
                    print(f"\n\nInterrupted! Saving progress...")
                    save_progress(PROGRESS_PATH, progress)
                    if processed_since_save > 0:
                        save_yaml(data, YAML_PATH)
                    print(f"Completed: {success_count}, Failed: {fail_count}")
                    sys.exit(0)
                except Exception as e:
                    print(f"  EXCEPTION: {e}")
                    traceback.print_exc()
                    progress["failed"].append(proj_id)
                    fail_count += 1
                    save_progress(PROGRESS_PATH, progress)

    # --------------- Gemini provider ---------------
    elif args.provider == "gemini":
        client = None
        model_name = args.model
        tool_config = None
        if not args.dry_run:
            client, model_name, tool_config = init_gemini(
                os.environ["GEMINI_API_KEY"],
                model_name=args.model,
                use_grounding=args.research,
            )

        min_interval = 60.0 / args.rpm
        last_request_time = 0.0
        processed_since_save = 0
        success_count = 0
        fail_count = 0

        expert_projects = data.get("expert_projects", {})

        print(f"\nProcessing {len(pending)} projects (rate: {args.rpm} RPM)...\n")

        for i, (proj_id, project, missing_fields) in enumerate(pending):
            proj_name = project.get("name", "unknown")
            print(f"[{i+1}/{len(pending)}] {proj_id} ({', '.join(missing_fields)})")

            if not args.dry_run:
                elapsed = time.time() - last_request_time
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)

            try:
                last_request_time = time.time()
                new_fields = process_project_gemini(
                    client, model_name, tool_config,
                    project, missing_fields, dry_run=args.dry_run,
                )

                if new_fields:
                    apply_fields(expert_projects[proj_id], new_fields)
                    progress["completed"].append(proj_id)
                    completed_set.add(proj_id)
                    success_count += 1
                    processed_since_save += 1
                    generated = ", ".join(new_fields.keys())
                    print(f"  OK [{generated}]")
                elif args.dry_run:
                    pass
                else:
                    progress["failed"].append(proj_id)
                    fail_count += 1

                save_progress(PROGRESS_PATH, progress)

                if not args.dry_run and processed_since_save >= args.save_every:
                    print(f"  [Saving YAML checkpoint...]")
                    save_yaml(data, YAML_PATH)
                    processed_since_save = 0

            except KeyboardInterrupt:
                print(f"\n\nInterrupted! Saving progress...")
                save_progress(PROGRESS_PATH, progress)
                if processed_since_save > 0:
                    save_yaml(data, YAML_PATH)
                print(f"Completed: {success_count}, Failed: {fail_count}")
                sys.exit(0)
            except Exception as e:
                print(f"  EXCEPTION: {e}")
                traceback.print_exc()
                progress["failed"].append(proj_id)
                fail_count += 1
                save_progress(PROGRESS_PATH, progress)

    # Final save
    if not args.dry_run:
        print(f"\n[Final YAML save...]")
        save_yaml(data, YAML_PATH)

    save_progress(PROGRESS_PATH, progress)

    print(f"\n{'='*50}")
    print(f"DONE!")
    print(f"  Successful: {success_count}")
    print(f"  Failed:     {fail_count}")
    print(f"  Skipped:    {len(completed_set) - success_count}")
    print(f"  Total:      {total}")

    if fail_count > 0:
        print(f"\nFailed project IDs:")
        for fid in progress["failed"]:
            print(f"  - {fid}")


if __name__ == "__main__":
    main()
