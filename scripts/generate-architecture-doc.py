#!/usr/bin/env python3
"""
Generate architecture documents for projects using a multi-pass LLM pipeline.

The pipeline generates a comprehensive architecture guide (.md) for each project,
with optional draw.io diagrams converted to SVG.

Pipeline passes:
  1. Skeleton — Generate document outline (sections, subsections, summaries)
  2. Sections — Generate each section in detail (parallelizable)
  3. Consistency — Check and fix naming/type consistency across all sections
  4. Diagrams — Generate draw.io XML for each diagram, convert to SVG

Usage:
    # Generate for a single project
    python3 scripts/generate-architecture-doc.py --provider claude --project build-interpreter

    # Generate for all projects that don't have a doc yet
    python3 scripts/generate-architecture-doc.py --provider claude --all

    # With web research enabled
    python3 scripts/generate-architecture-doc.py --provider claude --research --project tetris

    # Parallel section generation (Claude)
    python3 scripts/generate-architecture-doc.py --provider claude --workers 3

    # Using Gemini API
    python3 scripts/generate-architecture-doc.py --provider gemini --project build-interpreter

    # Skip diagram generation
    python3 scripts/generate-architecture-doc.py --provider claude --project build-interpreter --no-diagrams

    # Dry run (preview without writing)
    python3 scripts/generate-architecture-doc.py --provider claude --project build-interpreter --dry-run

Providers:
    claude  — Uses `claude -p` CLI (Claude Max subscription, free with sub)
    gemini  — Uses Gemini API (requires GEMINI_API_KEY env var)

Requirements:
    claude provider: Claude Code CLI installed and authenticated
    gemini provider: pip install google-genai pyyaml && export GEMINI_API_KEY=...
    diagrams (optional): draw.io CLI (`drawio`) for SVG conversion
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
ARCH_DOCS_DIR = DATA_DIR / "architecture-docs"


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

class _LiteralStr(str):
    """Tag so the YAML dumper emits a literal block scalar ( | )."""

def _literal_representer(dumper, data):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)

yaml.add_representer(_LiteralStr, _literal_representer)


# ---------------------------------------------------------------------------
# LLM Provider: Claude Code CLI
# ---------------------------------------------------------------------------

def call_claude(prompt: str, model: str = "sonnet", research: bool = False, max_retries: int = 3, timeout: int = 300) -> str | None:
    """Call Claude via `claude -p` CLI."""
    claude_path = shutil.which("claude")
    if not claude_path:
        print("ERROR: `claude` CLI not found in PATH.")
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
                timeout=timeout,
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
# LLM Provider: Gemini API
# ---------------------------------------------------------------------------

def init_gemini(api_key: str, model_name: str = "gemini-2.0-flash", use_grounding: bool = True):
    """Initialize Gemini client."""
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
            print("WARNING: Google Search grounding not available.")
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
                config=types.GenerateContentConfig(tools=tools) if tools else None,
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


def call_llm(prompt: str, provider: str, model: str, research: bool,
             gemini_client=None, gemini_model=None, gemini_tools=None,
             timeout: int = 300) -> str | None:
    """Unified LLM call dispatcher."""
    if provider == "claude":
        return call_claude(prompt, model=model, research=research, timeout=timeout)
    elif provider == "gemini":
        return call_gemini(gemini_client, gemini_model, prompt, gemini_tools)
    return None


# ---------------------------------------------------------------------------
# Project data helpers
# ---------------------------------------------------------------------------

def load_project(data: dict, project_id: str) -> dict | None:
    """Load a project from expert_projects."""
    return data.get("expert_projects", {}).get(project_id)


def get_project_context(project: dict) -> str:
    """Build a context string from project data for prompts."""
    name = project.get("name", "")
    desc = project.get("description", "")
    difficulty = project.get("difficulty", "")

    # Languages
    languages = project.get("languages", {})
    if isinstance(languages, dict):
        recommended = languages.get("recommended", [])
        also_possible = languages.get("also_possible", [])
        lang_str = ", ".join(recommended) if isinstance(recommended, list) else str(recommended)
        if also_possible:
            alt_str = ", ".join(also_possible) if isinstance(also_possible, list) else str(also_possible)
            lang_str += f" (also: {alt_str})"
    elif isinstance(languages, list):
        lang_str = ", ".join(str(l) for l in languages)
    else:
        lang_str = str(languages) if languages else "Not specified"

    # Primary language for code examples
    if isinstance(languages, dict):
        rec = languages.get("recommended", [])
        primary_lang = rec[0] if isinstance(rec, list) and rec else "Python"
    elif isinstance(languages, list) and languages:
        primary_lang = str(languages[0])
    else:
        primary_lang = "Python"

    # Prerequisites
    prereqs = project.get("prerequisites", [])
    prereq_str = "\n".join(f"- {p.get('name', p) if isinstance(p, dict) else p}" for p in prereqs) if prereqs else "None"

    # Resources
    resources = project.get("resources", [])
    res_str = "\n".join(
        f"- [{r.get('name', '')}]({r.get('url', '')}) ({r.get('type', '')})"
        for r in resources
    ) if resources else "None"

    # Milestones summary
    milestones = project.get("milestones", [])
    ms_lines = []
    for i, m in enumerate(milestones, 1):
        ms_name = m.get("name", "")
        ms_desc = m.get("description", "")
        ac = m.get("acceptance_criteria", [])
        ac_str = "\n    ".join(f"- {c}" for c in ac)
        concepts = ", ".join(m.get("concepts", []))
        pitfalls = ", ".join(m.get("pitfalls", []))
        deliverables = "\n    ".join(f"- {d}" for d in m.get("deliverables", []))

        ms_lines.append(f"""  Milestone {i}: {ms_name}
    Description: {ms_desc}
    Acceptance Criteria:
    {ac_str}
    Concepts: {concepts}
    Pitfalls: {pitfalls}
    Deliverables:
    {deliverables}""")

    milestones_str = "\n\n".join(ms_lines)

    return f"""PROJECT: {name}
DESCRIPTION: {desc}
DIFFICULTY: {difficulty}
LANGUAGES: {lang_str}
PRIMARY LANGUAGE (for code examples): {primary_lang}

PREREQUISITES:
{prereq_str}

RESOURCES:
{res_str}

MILESTONES:
{milestones_str}"""


# ---------------------------------------------------------------------------
# Pass 1: Skeleton generation
# ---------------------------------------------------------------------------

SKELETON_PROMPT = """You are an expert software architect and technical writer. Generate a document outline (skeleton) for an architecture guide for the following project.

{context}

=== REQUIREMENTS ===

Generate a JSON object with the following structure:

{{
  "title": "Architecture Guide Title",
  "overview": "A 2-3 sentence summary of what this architecture doc covers",
  "sections": [
    {{
      "id": "section-id",
      "title": "Section Title",
      "summary": "What this section covers (1-2 sentences)",
      "subsections": [
        {{
          "id": "subsection-id",
          "title": "Subsection Title",
          "summary": "What this subsection covers"
        }}
      ]
    }}
  ],
  "diagrams": [
    {{
      "id": "diagram-id",
      "title": "Diagram Title",
      "description": "What this diagram shows",
      "type": "flowchart|sequence|class|component|data-flow"
    }}
  ]
}}

=== GUIDELINES ===

The architecture document should:
1. Start with an Overview section explaining the big picture
2. Cover the System Architecture (high-level component layout and data flow)
3. Have a section for each major component/module with:
   - Purpose and responsibilities
   - Key types/interfaces/data structures
   - Design decisions and trade-offs
4. Cover how components interact (data flow, communication patterns)
5. Include Error Handling strategy
6. Include Testing Strategy
7. End with Extension Points (future improvements)

Each section should map to one or more milestones from the project.
The document should help a learner understand the full picture BEFORE they start coding.

Suggest 2-4 diagrams that would be most helpful:
- System overview / component diagram
- Data flow diagram
- Key class/type relationships
- Sequence diagram for important flows

=== OUTPUT ===
Output ONLY the JSON object, no markdown fences, no explanation before or after.
"""


def generate_skeleton(project: dict, provider: str, model: str, research: bool, **llm_kwargs) -> dict | None:
    """Pass 1: Generate document skeleton."""
    context = get_project_context(project)
    prompt = SKELETON_PROMPT.format(context=context)

    response = call_llm(prompt, provider, model, research, timeout=180, **llm_kwargs)
    if not response:
        print("  ERROR: No response for skeleton generation")
        return None

    # Extract JSON from response (handle possible markdown fences)
    json_text = response.strip()
    if json_text.startswith("```"):
        json_text = re.sub(r"^```\w*\n?", "", json_text)
        json_text = re.sub(r"\n?```$", "", json_text)

    try:
        skeleton = json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"  ERROR: Failed to parse skeleton JSON: {e}")
        debug_path = DATA_DIR / "archdoc_debug_skeleton.txt"
        with open(debug_path, "w") as f:
            f.write(response)
        print(f"  Raw response saved to {debug_path}")
        return None

    # Validate required fields
    if "title" not in skeleton or "sections" not in skeleton:
        print("  ERROR: Skeleton missing required fields (title, sections)")
        return None

    return skeleton


# ---------------------------------------------------------------------------
# Pass 2: Section generation
# ---------------------------------------------------------------------------

SECTION_PROMPT = """You are an expert software architect writing a section of an architecture guide.

=== PROJECT CONTEXT ===
{context}

=== DOCUMENT OUTLINE ===
Title: {doc_title}
Overview: {doc_overview}

Full outline:
{outline}

=== YOUR TASK ===
Write the following section in detail:

Section: {section_title}
Summary: {section_summary}
{subsections_info}

=== GUIDELINES ===
- Write in Markdown format
- Start with a ## heading for the section title
- Use ### for subsections
- Include code examples in the project's primary language using fenced code blocks (```language)
- Include TypeScript/language-appropriate type definitions and interfaces
- Explain design decisions and trade-offs
- Reference other sections when relevant (e.g., "as described in the Parser section")
- Use blockquotes (>) for important tips or warnings
- Use tables where appropriate (e.g., for operator precedence, token types)
- Keep explanations clear and educational — the reader is a learner building this project
- Include inline code (`backticks`) for variable names, function names, types
- If a diagram is relevant to this section, include a placeholder: ![Diagram Title](./diagrams/diagram-id.svg)

=== CONTENT TO WRITE ===
{previous_sections_summary}

Write ONLY the markdown content for this section. Do not include the document title or any content outside this section.
Do NOT wrap your output in markdown fences. Just write the raw markdown.
"""


def build_outline_str(skeleton: dict) -> str:
    """Build a readable outline string from skeleton."""
    lines = []
    for s in skeleton.get("sections", []):
        lines.append(f"- {s['title']}: {s.get('summary', '')}")
        for sub in s.get("subsections", []):
            lines.append(f"  - {sub['title']}: {sub.get('summary', '')}")
    return "\n".join(lines)


def generate_section(section: dict, skeleton: dict, project: dict,
                     previous_summaries: str, provider: str, model: str,
                     research: bool, **llm_kwargs) -> str | None:
    """Pass 2: Generate a single section."""
    context = get_project_context(project)
    outline = build_outline_str(skeleton)

    subsections = section.get("subsections", [])
    if subsections:
        sub_info = "Subsections:\n" + "\n".join(
            f"  - {s['title']}: {s.get('summary', '')}" for s in subsections
        )
    else:
        sub_info = ""

    prev_info = ""
    if previous_summaries:
        prev_info = f"\nPrevious sections summary (for continuity):\n{previous_summaries}\n"

    prompt = SECTION_PROMPT.format(
        context=context,
        doc_title=skeleton.get("title", ""),
        doc_overview=skeleton.get("overview", ""),
        outline=outline,
        section_title=section.get("title", ""),
        section_summary=section.get("summary", ""),
        subsections_info=sub_info,
        previous_sections_summary=prev_info,
    )

    response = call_llm(prompt, provider, model, research, timeout=300, **llm_kwargs)
    if not response:
        print(f"  ERROR: No response for section '{section.get('title', '')}'")
        return None

    # Strip any accidental markdown fence wrapping
    content = response.strip()
    if content.startswith("```markdown"):
        content = content[len("```markdown"):].strip()
    if content.startswith("```md"):
        content = content[len("```md"):].strip()
    if content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    return content


# ---------------------------------------------------------------------------
# Pass 3: Consistency check (sliding window)
# ---------------------------------------------------------------------------

CONSISTENCY_EXTRACT_PROMPT = """You are an expert technical editor. Analyze the following two sections of an architecture document and extract a naming conventions registry.

=== PROJECT ===
{project_name}: {project_desc}

=== ESTABLISHED CONVENTIONS (from previous sections) ===
{existing_conventions}

=== SECTION A: {section_a_title} ===
{section_a}

=== SECTION B: {section_b_title} ===
{section_b}

=== TASK ===
1. List all type names, struct names, interface names, function/method names, constants, and key terminology used in both sections.
2. Identify any INCONSISTENCIES between the two sections (or with established conventions):
   - Same concept with different names (e.g., `TxID` vs `tx_id` vs `TransactionID`)
   - Same type/struct with different field names
   - Contradictory descriptions of the same component
   - Mismatched cross-references
3. For each inconsistency, pick the BEST canonical name (prefer the one used more frequently or defined first).
4. Output a JSON object with this structure:

{{
  "conventions": {{
    "types": {{"CanonicalName": "description"}},
    "methods": {{"canonicalName": "description"}},
    "constants": {{"CANONICAL_NAME": "description"}},
    "terminology": {{"preferred term": "avoid these alternatives"}}
  }},
  "corrections": [
    {{
      "find": "exact string to find",
      "replace": "exact string to replace with",
      "reason": "why this change"
    }}
  ]
}}

=== RULES ===
- Output ONLY the JSON object. No explanation before or after.
- Only include REAL inconsistencies, not style differences that are valid (e.g., Go exported vs unexported names).
- `find` and `replace` must be exact strings that can be used for find-and-replace.
- Be precise: `find` should include enough context to avoid false replacements (e.g., include surrounding code).
- Do NOT suggest renaming things that are intentionally different (e.g., a `Coordinator` type vs a `coordinator` variable).
"""

CONSISTENCY_FINAL_PROMPT = """You are an expert technical editor. Review these corrections for an architecture document and filter out any that are wrong or dangerous.

=== PROJECT ===
{project_name}: {project_desc}

=== ALL PROPOSED CORRECTIONS ===
{all_corrections}

=== TASK ===
Review each correction and output ONLY the ones that are:
1. Genuinely fixing an inconsistency (not just style preference)
2. Safe to apply (won't break code examples)
3. Not duplicates

Output a JSON array of the approved corrections:
[
  {{
    "find": "exact string to find",
    "replace": "exact string to replace with"
  }}
]

If no corrections are needed, output: []

Output ONLY the JSON array. No explanation.
"""


def consistency_check(full_doc: str, section_contents: list, project: dict,
                      provider: str, model: str, research: bool, **llm_kwargs) -> str | None:
    """Pass 3: Sliding window consistency check across section pairs."""
    if len(section_contents) < 2:
        print("  Only 1 section, skipping consistency check")
        return full_doc

    project_name = project.get("name", "")
    project_desc = project.get("description", "")

    all_corrections = []
    conventions_so_far = "(none yet — this is the first pair)"

    # Slide through pairs of adjacent sections
    for i in range(len(section_contents) - 1):
        section_a, content_a = section_contents[i]
        section_b, content_b = section_contents[i + 1]

        title_a = section_a.get("title", f"Section {i+1}")
        title_b = section_b.get("title", f"Section {i+2}")

        print(f"    Checking: {title_a} <-> {title_b}...")

        prompt = CONSISTENCY_EXTRACT_PROMPT.format(
            project_name=project_name,
            project_desc=project_desc,
            existing_conventions=conventions_so_far,
            section_a_title=title_a,
            section_a=content_a[:40000],  # Cap at ~10K tokens per section
            section_b_title=title_b,
            section_b=content_b[:40000],
        )

        response = call_llm(prompt, provider, model, research, timeout=300, **llm_kwargs)
        if not response:
            print(f"    WARNING: No response for pair {title_a} <-> {title_b}")
            continue

        # Parse JSON from response
        json_text = response.strip()
        if json_text.startswith("```"):
            json_text = re.sub(r"^```\w*\n?", "", json_text)
            json_text = re.sub(r"\n?```$", "", json_text)

        try:
            result = json.loads(json_text)
        except json.JSONDecodeError:
            print(f"    WARNING: Could not parse JSON for pair {title_a} <-> {title_b}")
            continue

        # Accumulate conventions
        conventions = result.get("conventions", {})
        if conventions:
            conventions_so_far = json.dumps(conventions, indent=2)

        # Accumulate corrections
        corrections = result.get("corrections", [])
        if corrections:
            all_corrections.extend(corrections)
            print(f"    Found {len(corrections)} corrections")
        else:
            print(f"    No issues found")

    if not all_corrections:
        print("  No corrections needed")
        return full_doc

    print(f"\n  Total raw corrections: {len(all_corrections)}")

    # Final review: filter out bad corrections
    print(f"  Filtering corrections...")
    filter_prompt = CONSISTENCY_FINAL_PROMPT.format(
        project_name=project_name,
        project_desc=project_desc,
        all_corrections=json.dumps(all_corrections, indent=2)[:50000],
    )

    response = call_llm(filter_prompt, provider, model, research, timeout=180, **llm_kwargs)
    if not response:
        print("  WARNING: Could not filter corrections, applying all raw corrections")
        approved = all_corrections
    else:
        json_text = response.strip()
        if json_text.startswith("```"):
            json_text = re.sub(r"^```\w*\n?", "", json_text)
            json_text = re.sub(r"\n?```$", "", json_text)
        try:
            approved = json.loads(json_text)
        except json.JSONDecodeError:
            print("  WARNING: Could not parse filtered corrections, applying all raw corrections")
            approved = all_corrections

    if not approved:
        print("  All corrections filtered out, no changes needed")
        return full_doc

    # Apply corrections
    corrected_doc = full_doc
    applied = 0
    for correction in approved:
        find_str = correction.get("find", "")
        replace_str = correction.get("replace", "")
        if find_str and replace_str and find_str != replace_str:
            if find_str in corrected_doc:
                corrected_doc = corrected_doc.replace(find_str, replace_str)
                applied += 1

    print(f"  Applied {applied}/{len(approved)} corrections")
    return corrected_doc


# ---------------------------------------------------------------------------
# Pass 4: Diagram generation
# ---------------------------------------------------------------------------

DIAGRAM_PROMPT = """You are an expert at creating draw.io (diagrams.net) XML diagrams.

Generate a draw.io XML diagram for:

Title: {diagram_title}
Description: {diagram_description}
Type: {diagram_type}

=== PROJECT CONTEXT ===
{project_name}: {project_desc}

=== ARCHITECTURE CONTEXT ===
The following is the relevant section of the architecture document:
{relevant_section}

=== REQUIREMENTS ===
- Use draw.io mxGraph XML format
- Use a dark theme color scheme:
  - Background: transparent
  - Node fill: #1a1a2e or #16213e or #0f3460
  - Node stroke: #3fb950 (green accent)
  - Text color: #e6edf3 (light gray)
  - Arrow/edge color: #8b949e
  - Highlight color: #3fb950 for important elements
- Use clean, readable layout
- Include all relevant components/types/flows from the architecture doc
- Make it visually clear and educational

=== OUTPUT ===
Output ONLY the draw.io XML. No explanation before or after.
Do NOT wrap in markdown fences.
Start with <?xml or <mxfile and end with the closing tag.
"""


def generate_diagram(diagram: dict, project: dict, full_doc: str,
                     provider: str, model: str, research: bool,
                     max_retries: int = 2, **llm_kwargs) -> str | None:
    """Pass 4: Generate a draw.io XML diagram."""
    # Extract relevant section from the document
    relevant = full_doc[:3000]  # First part for context

    prompt = DIAGRAM_PROMPT.format(
        diagram_title=diagram.get("title", ""),
        diagram_description=diagram.get("description", ""),
        diagram_type=diagram.get("type", "component"),
        project_name=project.get("name", ""),
        project_desc=project.get("description", ""),
        relevant_section=relevant,
    )

    for attempt in range(max_retries + 1):
        response = call_llm(prompt, provider, model, research, timeout=180, **llm_kwargs)
        if not response:
            print(f"  ERROR: No response for diagram '{diagram.get('title', '')}'")
            return None

        xml_text = response.strip()
        # Strip any markdown fences
        if xml_text.startswith("```"):
            xml_text = re.sub(r"^```\w*\n?", "", xml_text)
            xml_text = re.sub(r"\n?```$", "", xml_text)

        # Basic XML validation
        if "<mxfile" in xml_text or "<mxGraphModel" in xml_text:
            # Ensure it ends properly
            if "</mxfile>" in xml_text or "</mxGraphModel>" in xml_text:
                return xml_text

        if attempt < max_retries:
            print(f"  Diagram XML invalid, retrying (attempt {attempt + 2}/{max_retries + 1})...")
            prompt = f"The previous output was not valid draw.io XML. Please try again.\n\n{prompt}"
        else:
            print(f"  ERROR: Could not generate valid diagram XML after {max_retries + 1} attempts")
            return None


def convert_drawio_to_svg(drawio_path: Path, svg_path: Path) -> bool:
    """Convert a .drawio file to .svg using draw.io CLI."""
    drawio_cli = shutil.which("drawio") or shutil.which("draw.io")
    if not drawio_cli:
        # Try common paths
        for candidate in ["/usr/bin/drawio", "/usr/local/bin/drawio",
                          "/Applications/draw.io.app/Contents/MacOS/draw.io"]:
            if Path(candidate).exists():
                drawio_cli = candidate
                break

    if not drawio_cli:
        print("  WARNING: draw.io CLI not found, skipping SVG conversion")
        return False

    try:
        result = subprocess.run(
            [drawio_cli, "--export", "--format", "svg", "--output", str(svg_path), str(drawio_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return True
        print(f"  WARNING: draw.io export failed: {result.stderr[:200]}")
        return False
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  WARNING: draw.io export error: {e}")
        return False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_project(project_id: str, project: dict, args,
                    gemini_client=None, gemini_model=None, gemini_tools=None) -> bool:
    """Run the full pipeline for a single project."""
    llm_kwargs = {}
    if args.provider == "gemini":
        llm_kwargs = {
            "gemini_client": gemini_client,
            "gemini_model": gemini_model,
            "gemini_tools": gemini_tools,
        }

    project_name = project.get("name", project_id)
    print(f"\n{'='*60}")
    print(f"PROJECT: {project_name} ({project_id})")
    print(f"{'='*60}")

    # Output directory
    out_dir = ARCH_DOCS_DIR / project_id
    diagrams_dir = out_dir / "diagrams"
    out_file = out_dir / "index.md"

    # Check if doc already exists
    if out_file.exists() and not args.force:
        print(f"  Architecture doc already exists at {out_file}")
        print(f"  Use --force to regenerate")
        return True

    if args.dry_run:
        print(f"  [DRY RUN] Would generate architecture doc to {out_file}")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Pass 1: Skeleton ---
    print(f"\n  [Pass 1/4] Generating skeleton...")
    skeleton = generate_skeleton(
        project, args.provider, args.model, args.research, **llm_kwargs
    )
    if not skeleton:
        print("  FAILED: Could not generate skeleton")
        return False

    sections = skeleton.get("sections", [])
    diagrams = skeleton.get("diagrams", [])
    print(f"  Skeleton: {len(sections)} sections, {len(diagrams)} diagrams")

    # Save skeleton for debugging
    skeleton_path = out_dir / "skeleton.json"
    with open(skeleton_path, "w") as f:
        json.dump(skeleton, f, indent=2)

    # --- Pass 2: Sections ---
    print(f"\n  [Pass 2/4] Generating {len(sections)} sections...")
    section_contents = []
    previous_summaries = ""

    if args.workers > 1 and args.provider == "claude":
        # Parallel section generation (less context continuity but faster)
        print(f"  Using {args.workers} parallel workers")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for section in sections:
                future = executor.submit(
                    generate_section, section, skeleton, project,
                    "", args.provider, args.model, args.research, **llm_kwargs
                )
                futures[future] = section

            for future in as_completed(futures):
                section = futures[future]
                try:
                    content = future.result()
                    if content:
                        section_contents.append((section, content))
                        print(f"    OK: {section.get('title', '')} ({len(content)} chars)")
                    else:
                        print(f"    FAIL: {section.get('title', '')}")
                except Exception as e:
                    print(f"    EXCEPTION for {section.get('title', '')}: {e}")

        # Sort by original order
        section_order = {s["id"]: i for i, s in enumerate(sections)}
        section_contents.sort(key=lambda x: section_order.get(x[0].get("id", ""), 999))
    else:
        # Sequential section generation (better context continuity)
        for i, section in enumerate(sections):
            title = section.get("title", f"Section {i+1}")
            print(f"    [{i+1}/{len(sections)}] {title}...")

            content = generate_section(
                section, skeleton, project, previous_summaries,
                args.provider, args.model, args.research, **llm_kwargs
            )

            if content:
                section_contents.append((section, content))
                # Build summary of previous sections for context
                # Take first 200 chars of each section as summary
                summary = content[:200].replace("\n", " ").strip()
                previous_summaries += f"\n- {title}: {summary}..."
                print(f"    OK ({len(content)} chars)")
            else:
                print(f"    FAIL")

            # Rate limiting for Gemini
            if args.provider == "gemini":
                time.sleep(60.0 / args.rpm)

    if not section_contents:
        print("  FAILED: No sections generated")
        return False

    # Assemble document
    doc_title = skeleton.get("title", f"{project_name} Architecture Guide")
    doc_overview = skeleton.get("overview", "")

    parts = [f"# {doc_title}\n"]
    if doc_overview:
        parts.append(f"\n## Overview\n\n{doc_overview}\n")

    # Add a note about the document
    parts.append(
        "\n> This guide is meant to help you understand the big picture before "
        "diving into each milestone. Refer back to it whenever you need context "
        "on how components connect.\n"
    )

    for section, content in section_contents:
        parts.append(f"\n{content}\n")

    full_doc = "\n".join(parts)
    print(f"\n  Assembled document: {len(full_doc)} chars")

    # --- Pass 3: Consistency check (sliding window) ---
    print(f"\n  [Pass 3/4] Running consistency check ({len(section_contents)} sections)...")
    checked_doc = consistency_check(
        full_doc, section_contents, project, args.provider, args.model, args.research, **llm_kwargs
    )
    if checked_doc:
        full_doc = checked_doc
    else:
        print("  WARNING: Consistency check failed, keeping original")

    # --- Pass 4: Diagrams ---
    if not args.no_diagrams and diagrams:
        print(f"\n  [Pass 4/4] Generating {len(diagrams)} diagrams...")
        diagrams_dir.mkdir(parents=True, exist_ok=True)

        for diagram in diagrams:
            diag_id = diagram.get("id", "diagram")
            diag_title = diagram.get("title", "")
            print(f"    {diag_title} ({diag_id})...")

            xml = generate_diagram(
                diagram, project, full_doc,
                args.provider, args.model, args.research, **llm_kwargs
            )
            if xml:
                # Save .drawio file
                drawio_path = diagrams_dir / f"{diag_id}.drawio"
                with open(drawio_path, "w") as f:
                    f.write(xml)
                print(f"    Saved {drawio_path.name}")

                # Try to convert to SVG
                svg_path = diagrams_dir / f"{diag_id}.svg"
                if convert_drawio_to_svg(drawio_path, svg_path):
                    print(f"    Converted to {svg_path.name}")
                else:
                    print(f"    SVG conversion skipped (install draw.io CLI for auto-conversion)")
            else:
                print(f"    FAIL: Could not generate diagram")
    else:
        if args.no_diagrams:
            print(f"\n  [Pass 4/4] Skipped (--no-diagrams)")
        else:
            print(f"\n  [Pass 4/4] No diagrams in skeleton")

    # Write final document
    with open(out_file, "w") as f:
        f.write(full_doc)
    print(f"\n  Written: {out_file} ({len(full_doc)} chars)")

    return True


def update_yaml(data: dict, project_id: str, doc_path: str):
    """Update projects.yaml with architecture_doc path."""
    expert_projects = data.get("expert_projects", {})
    if project_id in expert_projects:
        expert_projects[project_id]["architecture_doc"] = doc_path

    # Save YAML
    with open(YAML_PATH, "w") as f:
        yaml.dump(
            data,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=120,
        )


def get_all_project_ids(data: dict) -> list[str]:
    """Get all project IDs from expert_projects."""
    return list(data.get("expert_projects", {}).keys())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate architecture documents for projects using LLM pipeline"
    )
    parser.add_argument("--provider", type=str, default="claude", choices=["claude", "gemini"],
                        help="LLM provider (default: claude)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (claude: sonnet/opus/haiku, gemini: gemini-2.0-flash)")
    parser.add_argument("--research", action="store_true",
                        help="Enable web research for accuracy")
    parser.add_argument("--project", type=str,
                        help="Generate for a specific project ID")
    parser.add_argument("--all", action="store_true",
                        help="Generate for all projects without an architecture doc")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if doc already exists")
    parser.add_argument("--no-diagrams", action="store_true",
                        help="Skip diagram generation (Pass 4)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for section generation (default: 1)")
    parser.add_argument("--rpm", type=int, default=15,
                        help="Rate limit: requests per minute (gemini only, default: 15)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing")
    args = parser.parse_args()

    if not args.project and not args.all:
        print("ERROR: Specify --project <id> or --all")
        parser.print_help()
        sys.exit(1)

    # Default model per provider
    if args.model is None:
        args.model = "sonnet" if args.provider == "claude" else "gemini-2.0-flash"

    # Provider checks
    if args.provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key and not args.dry_run:
            print("ERROR: GEMINI_API_KEY environment variable not set")
            sys.exit(1)
    elif args.provider == "claude" and not args.dry_run:
        if not shutil.which("claude"):
            print("ERROR: `claude` CLI not found in PATH.")
            sys.exit(1)

    # Load data
    print(f"Loading {YAML_PATH}...")
    with open(YAML_PATH) as f:
        data = yaml.safe_load(f)

    # Initialize Gemini if needed
    gemini_client = gemini_model = gemini_tools = None
    if args.provider == "gemini" and not args.dry_run:
        gemini_client, gemini_model, gemini_tools = init_gemini(
            os.environ["GEMINI_API_KEY"],
            model_name=args.model,
            use_grounding=args.research,
        )

    # Determine projects to process
    if args.project:
        project_ids = [args.project]
    else:
        # --all: get projects without architecture docs
        all_ids = get_all_project_ids(data)
        if args.force:
            project_ids = all_ids
        else:
            project_ids = [
                pid for pid in all_ids
                if not data["expert_projects"][pid].get("architecture_doc")
            ]

    print(f"\nProvider: {args.provider} (model: {args.model})")
    print(f"Research: {'ON' if args.research else 'OFF'}")
    print(f"Workers: {args.workers}")
    print(f"Projects to process: {len(project_ids)}")

    if not project_ids:
        print("No projects to process!")
        return

    success = 0
    failed = 0

    for pid in project_ids:
        project = load_project(data, pid)
        if not project:
            print(f"\nWARNING: Project '{pid}' not found in expert_projects, skipping")
            failed += 1
            continue

        try:
            ok = process_project(
                pid, project, args,
                gemini_client=gemini_client,
                gemini_model=gemini_model,
                gemini_tools=gemini_tools,
            )

            if ok and not args.dry_run:
                doc_path = f"architecture-docs/{pid}/index.md"
                update_yaml(data, pid, doc_path)
                print(f"  Updated YAML: architecture_doc = {doc_path}")
                success += 1
            elif ok:
                success += 1
            else:
                failed += 1

        except KeyboardInterrupt:
            print(f"\n\nInterrupted!")
            print(f"  Completed: {success}, Failed: {failed}")
            sys.exit(0)
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"  Successful: {success}")
    print(f"  Failed:     {failed}")
    print(f"  Total:      {len(project_ids)}")


if __name__ == "__main__":
    main()
