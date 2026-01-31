#!/usr/bin/env python3
"""
Generate improved hints for all milestones using Claude Code CLI or Gemini API.

Usage:
    # Using Claude Code CLI (uses your Claude Max subscription, no API key needed)
    python3 scripts/generate-hints.py --provider claude

    # With web research enabled (Claude will search the web for accuracy)
    python3 scripts/generate-hints.py --provider claude --research

    # Using Gemini API
    python3 scripts/generate-hints.py --provider gemini

    # Process specific project
    python3 scripts/generate-hints.py --provider claude --project build-interpreter

    # Dry run (preview without writing)
    python3 scripts/generate-hints.py --dry-run

    # Resume after interruption (automatic — progress is saved)
    python3 scripts/generate-hints.py --provider claude

    # Reset progress and start fresh
    python3 scripts/generate-hints.py --provider claude --reset

    # Parallel workers for Claude (default: 1)
    python3 scripts/generate-hints.py --provider claude --workers 3

    # Save every N milestones (default: 10)
    python3 scripts/generate-hints.py --provider claude --save-every 5

Providers:
    claude  — Uses `claude -p` CLI (Claude Max subscription, free with sub)
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
PROGRESS_PATH = DATA_DIR / "hints_progress.json"


# ---------------------------------------------------------------------------
# YAML helpers – preserve block-scalar style for level3 hints
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
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _resolve_language(project: dict) -> tuple[str, str]:
    """Return (lang_display_str, primary_language) from project languages field."""
    languages = project.get("languages", {})
    if isinstance(languages, dict):
        primary = languages.get("primary", languages.get("recommended", ""))
        if isinstance(primary, list):
            primary_lang = primary[0] if primary else ""
            lang_str = ", ".join(primary)
        elif primary:
            primary_lang = str(primary)
            lang_str = primary_lang
        else:
            all_vals = [str(v) for v in languages.values() if v and not isinstance(v, (dict, list))]
            lang_str = ", ".join(all_vals) if all_vals else "Python"
            primary_lang = all_vals[0] if all_vals else "Python"
    elif isinstance(languages, list):
        lang_str = ", ".join(str(l) for l in languages)
        primary_lang = str(languages[0]) if languages else "Python"
    else:
        lang_str = str(languages) if languages else "Python"
        primary_lang = lang_str
    return lang_str, primary_lang


def build_prompt(project: dict, milestone: dict, research: bool = False) -> str:
    project_name = project.get("name", "")
    project_desc = project.get("description", "")
    lang_str, primary_lang = _resolve_language(project)

    m_name = milestone.get("name", "")
    m_desc = milestone.get("description", "")

    ac = milestone.get("acceptance_criteria", [])
    ac_str = "\n".join(f"- {c}" for c in ac) if ac else "(none)"

    deliverables = milestone.get("deliverables", [])
    del_str = "\n".join(f"- {d}" for d in deliverables) if deliverables else "(none)"

    concepts = milestone.get("concepts", [])
    concepts_str = ", ".join(concepts) if concepts else "(none)"

    pitfalls = milestone.get("pitfalls", [])
    pitfalls_str = ", ".join(pitfalls) if pitfalls else "(none)"

    hints = milestone.get("hints", {})
    cur_l1 = hints.get("level1", "")
    cur_l2 = hints.get("level2", "")
    cur_l3 = hints.get("level3", "")

    # Build the reference section — only include if current hints exist
    ref_section = ""
    if cur_l1 or cur_l2 or cur_l3:
        ref_section = f"""
REFERENCE (previous hints for topic coverage — generate fresh, do NOT paraphrase):
level1: {cur_l1}
level2: {cur_l2}
level3: {cur_l3}
"""

    research_instruction = ""
    if research:
        research_instruction = """
IMPORTANT: Before generating hints, search the web for:
- Official documentation for the relevant libraries/APIs mentioned in this milestone
- Best practices and common implementation patterns for the concepts listed
- Known edge cases and pitfalls specific to this topic
Use what you find to ensure technical accuracy in your hints.
"""

    return f"""You are an expert programming tutor. Generate three progressive hint levels for the milestone below.

=== CONTEXT ===
PROJECT: {project_name}
DESCRIPTION: {project_desc}
LANGUAGES: {lang_str} (use {primary_lang} for all code)

MILESTONE: {m_name}
DESCRIPTION: {m_desc}

ACCEPTANCE CRITERIA:
{ac_str}

DELIVERABLES:
{del_str}

CONCEPTS: {concepts_str}
COMMON PITFALLS: {pitfalls_str}
{ref_section}
=== HINT REQUIREMENTS ===
{research_instruction}
level1 (2-3 sentences, NO code, NO function names):
- Explain the core concept and mental model needed
- Point the learner toward the right approach without revealing the solution
- Mention which pitfall to watch out for if relevant

level2 (10-20 lines):
- Numbered step-by-step plan to implement this milestone
- Name specific functions/classes/data structures to create
- Include pseudo-code for the tricky parts
- Reference the acceptance criteria to show what each step achieves

level3 (30-50 lines of {primary_lang} code):
- A single self-contained code snippet (include imports)
- Add inline comments explaining WHY, not WHAT
- Handle the edge cases mentioned in pitfalls
- The code must directly satisfy the deliverables listed above
- Use idiomatic {primary_lang} patterns and standard library where possible

=== EXAMPLE OUTPUT ===

[LEVEL1]
A hash table maps keys to values using a hash function that converts keys into array indices. Think about what happens when two keys hash to the same index — you need a collision resolution strategy. Start with chaining (linked lists at each bucket) since it's simpler to reason about than open addressing.
[/LEVEL1]
[LEVEL2]
Steps to implement a basic hash table:

1. Create a HashTable class with an array of buckets (start with size 16)
2. Implement _hash(key) -> int: use Python's built-in hash() mod capacity
3. Implement put(key, value):
   - Hash the key to find bucket index
   - If bucket is empty, create a new list with (key, value)
   - If key already exists in bucket, update the value
   - Otherwise append (key, value) to the bucket's list
4. Implement get(key) -> value:
   - Hash key, iterate bucket's list, return value if key matches
   - Raise KeyError if not found
5. Implement delete(key):
   - Find and remove the (key, value) pair from the bucket
6. Track load factor (size / capacity), resize when > 0.75:
   - Create new array with 2x capacity
   - Re-hash all existing entries into new array
[/LEVEL2]
[LEVEL3]
class HashTable:
    def __init__(self, capacity=16):
        self.capacity = capacity
        self.size = 0
        self.buckets = [[] for _ in range(capacity)]

    def _hash(self, key):
        return hash(key) % self.capacity

    def put(self, key, value):
        idx = self._hash(key)
        bucket = self.buckets[idx]

        # Update existing key if found
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return

        # New key — append and check load factor
        bucket.append((key, value))
        self.size += 1

        if self.size / self.capacity > 0.75:
            self._resize()

    def get(self, key):
        idx = self._hash(key)
        for k, v in self.buckets[idx]:
            if k == key:
                return v
        raise KeyError(key)

    def delete(self, key):
        idx = self._hash(key)
        bucket = self.buckets[idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self.size -= 1
                return v
        raise KeyError(key)

    def _resize(self):
        old_buckets = self.buckets
        self.capacity *= 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0
        # Re-hash all existing entries into larger array
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)
[/LEVEL3]

=== RULES ===
- Output ONLY the [LEVEL1]...[/LEVEL1], [LEVEL2]...[/LEVEL2], [LEVEL3]...[/LEVEL3] blocks
- Do NOT add any text before [LEVEL1] or after [/LEVEL3]
- Do NOT wrap code in markdown fences (no ```). Write raw code inside [LEVEL3]
- Do NOT add headings or labels inside the blocks (no "## Level 1" etc.)
- Ensure technical accuracy — use correct API signatures and language semantics
"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(text: str) -> dict | None:
    """Parse [LEVEL1]...[/LEVEL1] etc. markers from LLM response."""
    result = {}
    for level in ("LEVEL1", "LEVEL2", "LEVEL3"):
        pattern = rf"\[{level}\]\s*\n?(.*?)\s*\[/{level}\]"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None
        result[level.lower()] = match.group(1).strip()
    return result


def validate_hints(hints: dict) -> list[str]:
    """Return list of validation warnings (empty = OK).
    Only warns if hints are suspiciously short (likely truncated/failed).
    No upper limit — let the LLM write as much as needed.
    """
    warnings = []
    l1 = hints.get("level1", "")
    l2 = hints.get("level2", "")
    l3 = hints.get("level3", "")

    if len(l1) < 30:
        warnings.append(f"level1 too short ({len(l1)} chars, want >=30)")
    if len(l2) < 50:
        warnings.append(f"level2 too short ({len(l2)} chars, want >=50)")
    if len(l3) < 100:
        warnings.append(f"level3 too short ({len(l3)} chars, want >=100)")
    return warnings


# ---------------------------------------------------------------------------
# Provider: Claude Code CLI
# ---------------------------------------------------------------------------

def call_claude(prompt: str, model: str = "sonnet", research: bool = False, max_retries: int = 3) -> str | None:
    """Call Claude via `claude -p` CLI. Uses Claude Max subscription."""
    claude_path = shutil.which("claude")
    if not claude_path:
        print("ERROR: `claude` CLI not found in PATH. Install Claude Code first.")
        return None

    cmd = [claude_path, "-p", "--output-format", "text"]

    if model:
        cmd.extend(["--model", model])

    # If research is enabled, allow WebSearch + WebFetch tools
    # If not, disable all tools so Claude just generates text (faster)
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
                timeout=180,  # 3 min timeout per request
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

def collect_milestones(data: dict, project_filter: str | None = None) -> list[tuple[str, dict, dict]]:
    """Collect all (milestone_id, project, milestone) tuples from expert_projects."""
    results = []
    expert_projects = data.get("expert_projects", {})
    for proj_id, project in expert_projects.items():
        if project_filter and proj_id != project_filter:
            continue
        milestones = project.get("milestones", [])
        for ms in milestones:
            ms_id = ms.get("id", f"{proj_id}-unknown")
            results.append((ms_id, project, ms))
    return results


def process_milestone_claude(project: dict, milestone: dict, model: str, research: bool, dry_run: bool) -> dict | None:
    """Process a single milestone using Claude CLI."""
    prompt = build_prompt(project, milestone, research=research)

    if dry_run:
        print(f"  [DRY RUN] Would send prompt ({len(prompt)} chars)")
        return None

    response_text = call_claude(prompt, model=model, research=research)
    if not response_text:
        print("  ERROR: No response from Claude CLI")
        return None

    return _parse_and_validate(response_text)


def process_milestone_gemini(client, model_name: str, tool_config, project: dict, milestone: dict, dry_run: bool) -> dict | None:
    """Process a single milestone using Gemini API."""
    prompt = build_prompt(project, milestone, research=(tool_config is not None))

    if dry_run:
        print(f"  [DRY RUN] Would send prompt ({len(prompt)} chars)")
        return None

    response_text = call_gemini(client, model_name, prompt, tool_config)
    if not response_text:
        print("  ERROR: No response from Gemini API")
        return None

    return _parse_and_validate(response_text)


def _parse_and_validate(response_text: str) -> dict | None:
    """Parse response markers and validate hints."""
    parsed = parse_response(response_text)
    if not parsed:
        print("  ERROR: Could not parse response markers")
        debug_path = DATA_DIR / "hints_debug_last_response.txt"
        with open(debug_path, "w") as f:
            f.write(response_text)
        print(f"  Raw response saved to {debug_path}")
        return None

    hints = {
        "level1": parsed["level1"],
        "level2": parsed["level2"],
        "level3": _LiteralStr(parsed["level3"]) if "\n" in parsed["level3"] else parsed["level3"],
    }

    warnings = validate_hints(hints)
    for w in warnings:
        print(f"  WARNING: {w}")

    return hints


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

def _worker_claude(task: tuple, model: str, research: bool) -> tuple[str, dict | None, str]:
    """Worker function for parallel Claude processing. Returns (ms_id, hints, ms_name)."""
    ms_id, project, milestone = task
    ms_name = milestone.get("name", "unknown")
    proj_id = project.get("id", "unknown")

    prompt = build_prompt(project, milestone, research=research)
    response_text = call_claude(prompt, model=model, research=research)

    if not response_text:
        return ms_id, None, f"{proj_id}/{ms_name}"

    hints = _parse_and_validate(response_text)
    return ms_id, hints, f"{proj_id}/{ms_name}"


def process_parallel_claude(pending: list, data: dict, progress: dict, completed_set: set, args):
    """Process milestones in parallel using Claude CLI."""
    success_count = 0
    fail_count = 0
    processed_since_save = 0

    # Build a lookup from ms_id -> milestone dict for in-place update
    ms_lookup = {ms_id: ms for ms_id, _, ms in pending}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for task in pending:
            ms_id = task[0]
            future = executor.submit(_worker_claude, task, args.model, args.research)
            futures[future] = ms_id

        for future in as_completed(futures):
            ms_id = futures[future]
            try:
                returned_id, hints, label = future.result()

                if hints:
                    ms_lookup[returned_id]["hints"] = hints
                    progress["completed"].append(returned_id)
                    completed_set.add(returned_id)
                    success_count += 1
                    processed_since_save += 1
                    print(f"  OK {label} (L1:{len(hints['level1'])}c L2:{len(hints['level2'])}c L3:{len(hints['level3'])}c)")
                else:
                    progress["failed"].append(returned_id)
                    fail_count += 1
                    print(f"  FAIL {label}")

                save_progress(PROGRESS_PATH, progress)

                if processed_since_save >= args.save_every:
                    print(f"  [Saving YAML checkpoint...]")
                    save_yaml(data, YAML_PATH)
                    processed_since_save = 0

            except Exception as e:
                print(f"  EXCEPTION for {ms_id}: {e}")
                traceback.print_exc()
                progress["failed"].append(ms_id)
                fail_count += 1
                save_progress(PROGRESS_PATH, progress)

    return success_count, fail_count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate improved hints for milestones using Claude Code CLI or Gemini API"
    )
    parser.add_argument("--provider", type=str, default="claude", choices=["claude", "gemini"],
                        help="LLM provider (default: claude)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (claude: sonnet/opus/haiku, gemini: gemini-2.0-flash)")
    parser.add_argument("--research", action="store_true",
                        help="Enable web research (Claude: WebSearch tool, Gemini: Google Search grounding)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--project", type=str, help="Process only this project ID")
    parser.add_argument("--reset", action="store_true", help="Reset progress and start fresh")
    parser.add_argument("--rpm", type=int, default=15,
                        help="Rate limit: requests per minute (gemini only, default: 15)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (claude only, default: 1)")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save YAML every N milestones (default: 10)")
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
    completed_set = set(progress["completed"])

    # Collect milestones
    all_milestones = collect_milestones(data, args.project)
    total = len(all_milestones)
    pending = [(mid, proj, ms) for mid, proj, ms in all_milestones if mid not in completed_set]

    print(f"Provider: {args.provider} (model: {args.model})")
    print(f"Research: {'ON' if args.research else 'OFF'}")
    print(f"Total milestones: {total}")
    print(f"Already completed: {len(completed_set)}")
    print(f"Pending: {len(pending)}")

    if not pending:
        print("Nothing to process!")
        return

    # --------------- Claude provider ---------------
    if args.provider == "claude":
        if args.workers > 1 and not args.dry_run:
            print(f"Workers: {args.workers}")
            print(f"\nProcessing {len(pending)} milestones in parallel...\n")
            success_count, fail_count = process_parallel_claude(
                pending, data, progress, completed_set, args
            )
        else:
            success_count = 0
            fail_count = 0
            processed_since_save = 0

            print(f"\nProcessing {len(pending)} milestones...\n")

            for i, (ms_id, project, milestone) in enumerate(pending):
                proj_id = project.get("id", "unknown")
                ms_name = milestone.get("name", "unknown")
                print(f"[{i+1}/{len(pending)}] {proj_id} / {ms_name}")

                try:
                    new_hints = process_milestone_claude(
                        project, milestone, model=args.model,
                        research=args.research, dry_run=args.dry_run,
                    )

                    if new_hints:
                        milestone["hints"] = new_hints
                        progress["completed"].append(ms_id)
                        completed_set.add(ms_id)
                        success_count += 1
                        processed_since_save += 1
                        print(f"  OK (L1:{len(new_hints['level1'])}c L2:{len(new_hints['level2'])}c L3:{len(new_hints['level3'])}c)")
                    elif args.dry_run:
                        pass
                    else:
                        progress["failed"].append(ms_id)
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
                    progress["failed"].append(ms_id)
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

        print(f"\nProcessing {len(pending)} milestones (rate: {args.rpm} RPM)...\n")

        for i, (ms_id, project, milestone) in enumerate(pending):
            proj_id = project.get("id", "unknown")
            ms_name = milestone.get("name", "unknown")
            print(f"[{i+1}/{len(pending)}] {proj_id} / {ms_name}")

            if not args.dry_run:
                elapsed = time.time() - last_request_time
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)

            try:
                last_request_time = time.time()
                new_hints = process_milestone_gemini(
                    client, model_name, tool_config,
                    project, milestone, dry_run=args.dry_run,
                )

                if new_hints:
                    milestone["hints"] = new_hints
                    progress["completed"].append(ms_id)
                    completed_set.add(ms_id)
                    success_count += 1
                    processed_since_save += 1
                    print(f"  OK (L1:{len(new_hints['level1'])}c L2:{len(new_hints['level2'])}c L3:{len(new_hints['level3'])}c)")
                elif args.dry_run:
                    pass
                else:
                    progress["failed"].append(ms_id)
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
                progress["failed"].append(ms_id)
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
        print(f"\nFailed milestone IDs:")
        for fid in progress["failed"]:
            print(f"  - {fid}")


if __name__ == "__main__":
    main()
