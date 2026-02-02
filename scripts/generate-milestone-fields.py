#!/usr/bin/env python3
"""
Phase 2.1: Generate missing concepts, skills, and pitfalls for milestones.

Uses Claude CLI (`claude -p`) to batch-generate fields per project.
One API call per project containing all milestones needing work.
Resumable via progress JSON file.

Usage:
    python3 scripts/generate-milestone-fields.py
    python3 scripts/generate-milestone-fields.py --workers 3
    python3 scripts/generate-milestone-fields.py --dry-run
    python3 scripts/generate-milestone-fields.py --reset
    python3 scripts/generate-milestone-fields.py --project build-redis
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import yaml


# ── Config ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / ".." / "data"
YAML_PATH = DATA_DIR / "projects.yaml"
PROGRESS_PATH = DATA_DIR / "milestone_fields_progress.json"

MIN_ITEMS = 2  # Minimum items required for concepts/skills/pitfalls

yaml_lock = Lock()


# ── YAML helpers ─────────────────────────────────────────────────────────────

class _LiteralStr(str):
    pass

def _literal_representer(dumper, data):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)

yaml.add_representer(_LiteralStr, _literal_representer)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True,
                  width=120, sort_keys=False)


# ── Progress tracking ────────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_progress(progress: dict):
    progress["completed"] = list(dict.fromkeys(progress["completed"]))
    progress["failed"] = [f for f in dict.fromkeys(progress["failed"])
                          if f not in set(progress["completed"])]
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


# ── Determine what's missing ────────────────────────────────────────────────

def get_milestones_needing_work(project: dict) -> list[tuple[int, dict, list[str]]]:
    """Return list of (index, milestone, missing_fields) for milestones needing generation."""
    results = []
    for idx, ms in enumerate(project.get("milestones", [])):
        missing = []
        for field in ("concepts", "skills", "pitfalls"):
            val = ms.get(field, [])
            if not val or not isinstance(val, list) or len(val) < MIN_ITEMS:
                missing.append(field)
        if missing:
            results.append((idx, ms, missing))
    return results


# ── Prompt building ──────────────────────────────────────────────────────────

def build_prompt(project: dict, milestones_work: list[tuple[int, dict, list[str]]]) -> str:
    project_name = project.get("name", "")
    project_desc = project.get("description", "")
    difficulty = project.get("difficulty", "")

    # Languages context
    languages = project.get("languages", {})
    if isinstance(languages, dict):
        lang_str = ", ".join(languages.get("recommended", []))
    elif isinstance(languages, list):
        lang_str = ", ".join(str(l) for l in languages)
    else:
        lang_str = str(languages) if languages else ""

    # Build milestone sections
    milestone_sections = []
    for idx, ms, missing_fields in milestones_work:
        ms_name = ms.get("name", f"Milestone {idx}")
        ms_desc = ms.get("description", "")
        ms_id = idx + 1  # 1-based for prompt clarity

        fields_needed = []
        for field in missing_fields:
            if field == "concepts":
                fields_needed.append(f"[CONCEPTS_{ms_id}]\\n2-5 technical concepts (one per line)\\n[/CONCEPTS_{ms_id}]")
            elif field == "skills":
                fields_needed.append(f"[SKILLS_{ms_id}]\\n2-5 skill tags (one per line)\\n[/SKILLS_{ms_id}]")
            elif field == "pitfalls":
                fields_needed.append(f"[PITFALLS_{ms_id}]\\n2-5 common pitfalls (one per line)\\n[/PITFALLS_{ms_id}]")

        existing_info = []
        for field in ("concepts", "skills", "pitfalls"):
            val = ms.get(field, [])
            if val and isinstance(val, list) and len(val) > 0:
                existing_info.append(f"  Existing {field}: {', '.join(val)}")

        existing_str = "\n".join(existing_info) if existing_info else "  (none)"

        milestone_sections.append(f"""--- Milestone {ms_id}: {ms_name} ---
Description: {ms_desc}
Existing fields:
{existing_str}
Generate:
{chr(10).join(fields_needed)}""")

    return f"""You are an expert programming tutor. Generate missing fields for milestones in the project below.

=== PROJECT ===
Name: {project_name}
Description: {project_desc}
Difficulty: {difficulty}
Languages: {lang_str}

=== MILESTONES NEEDING FIELDS ===

{chr(10).join(milestone_sections)}

=== RULES ===
- CONCEPTS: Core technical concepts the learner must understand for this milestone (e.g., "Write-ahead logging", "Connection pooling")
- SKILLS: Practical skill tags gained (e.g., "TCP/IP Networking", "Error Handling")
- PITFALLS: Common mistakes beginners make (e.g., "Forgetting to handle connection timeouts", "Not validating input boundaries")
- Each item: one per line, no bullets/numbers, 3-15 words
- Generate 2-5 items per field
- Output ONLY the marker blocks, no other text
- Be specific to each milestone, not generic
"""


# ── Response parsing ─────────────────────────────────────────────────────────

def parse_response(text: str, milestones_work: list[tuple[int, dict, list[str]]]) -> dict[int, dict]:
    """Parse [FIELD_N]...[/FIELD_N] markers. Returns {ms_index: {field: [items]}}."""
    results = {}

    for idx, ms, missing_fields in milestones_work:
        ms_id = idx + 1
        ms_results = {}

        for field in missing_fields:
            marker = field.upper()
            pattern = rf"\[{marker}_{ms_id}\]\s*\n?(.*?)\s*\[/{marker}_{ms_id}\]"
            match = re.search(pattern, text, re.DOTALL)
            if not match:
                continue
            raw = match.group(1).strip()
            if not raw:
                continue

            items = [line.strip() for line in raw.split("\n") if line.strip()]
            # Strip leading bullets/numbers
            items = [re.sub(r"^[\d]+[.)]\s*|^[-*•]\s*", "", item).strip() for item in items]
            items = [item for item in items if item and len(item) > 2]

            if len(items) >= MIN_ITEMS:
                ms_results[field] = items

        if ms_results:
            results[idx] = ms_results

    return results


# ── Claude CLI caller ────────────────────────────────────────────────────────

def call_claude(prompt: str, max_retries: int = 3) -> str | None:
    claude_path = shutil.which("claude")
    if not claude_path:
        print("ERROR: `claude` CLI not found in PATH.")
        return None

    cmd = [claude_path, "-p", "--output-format", "text", "--model", "sonnet", "--tools", ""]

    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True, timeout=180,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()

            stderr = result.stderr.strip()
            if "overloaded" in stderr.lower() or "rate" in stderr.lower():
                wait = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif stderr:
                print(f"  Claude error: {stderr[:200]}")
                if attempt < max_retries - 1:
                    time.sleep(10)
            else:
                print(f"  Empty response (exit {result.returncode})")
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


# ── Process one project ──────────────────────────────────────────────────────

def process_project(pid: str, data: dict, progress: dict, dry_run: bool = False) -> bool:
    """Process a single project. Returns True if successful."""
    proj = data["expert_projects"].get(pid)
    if not proj:
        print(f"  [{pid}] Not found in expert_projects")
        return False

    milestones_work = get_milestones_needing_work(proj)
    if not milestones_work:
        print(f"  [{pid}] No milestones need work")
        return True

    fields_needed = sum(len(mf) for _, _, mf in milestones_work)
    print(f"  [{pid}] {len(milestones_work)} milestones, {fields_needed} fields needed")

    prompt = build_prompt(proj, milestones_work)

    if dry_run:
        print(f"  [{pid}] DRY RUN - prompt length: {len(prompt)} chars")
        print(f"  [{pid}] First 500 chars:\n{prompt[:500]}")
        return True

    response = call_claude(prompt)
    if not response:
        print(f"  [{pid}] FAILED: no response from Claude")
        return False

    parsed = parse_response(response, milestones_work)
    if not parsed:
        print(f"  [{pid}] FAILED: could not parse response")
        # Save debug info
        debug_path = DATA_DIR / f"milestone_debug_{pid}.txt"
        with open(debug_path, "w") as f:
            f.write(response)
        return False

    # Apply results
    milestones = proj.get("milestones", [])
    fields_filled = 0
    for ms_idx, ms_fields in parsed.items():
        if ms_idx < len(milestones):
            for field, items in ms_fields.items():
                milestones[ms_idx][field] = items
                fields_filled += 1

    print(f"  [{pid}] OK: filled {fields_filled}/{fields_needed} fields")

    # Save YAML periodically
    with yaml_lock:
        save_yaml(YAML_PATH, data)
        progress["completed"].append(pid)
        save_progress(progress)

    return fields_filled > 0


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate milestone concepts/skills/pitfalls")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Preview prompts only")
    parser.add_argument("--reset", action="store_true", help="Reset progress")
    parser.add_argument("--project", type=str, help="Process single project")
    parser.add_argument("--save-every", type=int, default=5, help="Save every N projects")
    args = parser.parse_args()

    print(f"Loading {YAML_PATH}...")
    data = load_yaml(YAML_PATH)

    if args.reset and PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()
        print("Progress reset.")

    progress = load_progress()
    completed = set(progress.get("completed", []))

    expert_projects = data.get("expert_projects", {})

    # Determine which projects need work
    if args.project:
        work_pids = [args.project] if args.project in expert_projects else []
    else:
        work_pids = []
        for pid, proj in expert_projects.items():
            if pid in completed:
                continue
            ms_work = get_milestones_needing_work(proj)
            if ms_work:
                work_pids.append(pid)

    print(f"Projects needing milestone fields: {len(work_pids)}")
    print(f"Already completed: {len(completed)}")

    if not work_pids:
        print("Nothing to do!")
        return

    if args.dry_run:
        for pid in work_pids[:3]:
            process_project(pid, data, progress, dry_run=True)
        print(f"\n(Showing 3 of {len(work_pids)} projects in dry-run mode)")
        return

    # Process with parallel workers
    if args.workers <= 1:
        # Sequential
        success = 0
        failed = 0
        for i, pid in enumerate(work_pids):
            print(f"\n[{i+1}/{len(work_pids)}] Processing {pid}...")
            ok = process_project(pid, data, progress)
            if ok:
                success += 1
            else:
                failed += 1
                progress["failed"].append(pid)
                save_progress(progress)
    else:
        # Parallel
        success = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for pid in work_pids:
                f = executor.submit(process_project, pid, data, progress)
                futures[f] = pid

            for i, future in enumerate(as_completed(futures)):
                pid = futures[future]
                try:
                    ok = future.result()
                    if ok:
                        success += 1
                    else:
                        failed += 1
                        progress["failed"].append(pid)
                        save_progress(progress)
                except Exception as e:
                    print(f"  [{pid}] Exception: {e}")
                    failed += 1
                    progress["failed"].append(pid)
                    save_progress(progress)

                if (i + 1) % 10 == 0:
                    print(f"\n--- Progress: {i+1}/{len(work_pids)} (success={success}, failed={failed}) ---\n")

    print(f"\n=== DONE ===")
    print(f"Success: {success}")
    print(f"Failed: {failed}")
    print(f"Total processed: {success + failed}/{len(work_pids)}")


if __name__ == "__main__":
    main()
