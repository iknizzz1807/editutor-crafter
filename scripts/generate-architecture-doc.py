#!/usr/bin/env python3
"""
Generate architecture documents for projects using a multi-pass LLM pipeline.

The pipeline generates a comprehensive architecture guide (.md) for each project,
with optional draw.io diagrams converted to SVG.

Pipeline passes (per project):
  1. Skeleton — Generate document outline (sections, subsections, summaries)
  2. Sections — Generate each section sequentially (with naming conventions tracking)
  3. Diagrams — Generate draw.io XML for each diagram, convert to SVG

Usage:
    # Generate for a single project
    python3 scripts/generate-architecture-doc.py --provider claude --project build-interpreter

    # Generate for all projects that don't have a doc yet
    python3 scripts/generate-architecture-doc.py --provider claude --all

    # With web research enabled
    python3 scripts/generate-architecture-doc.py --provider claude --research --project tetris

    # Process 3 projects in parallel
    python3 scripts/generate-architecture-doc.py --provider claude --workers 3 --all

    # Using Gemini API
    python3 scripts/generate-architecture-doc.py --provider gemini --project build-interpreter

    # Skip diagram generation
    python3 scripts/generate-architecture-doc.py --provider claude --project build-interpreter --no-diagrams

    # Dry run (preview without writing)
    python3 scripts/generate-architecture-doc.py --provider claude --project build-interpreter --dry-run

Providers:
    claude    — Uses `claude -p` CLI (Claude Max subscription, free with sub, slower due to process spawn)
    anthropic — Uses Anthropic SDK directly (requires ANTHROPIC_API_KEY, faster, no CLI overhead)
    gemini    — Uses Gemini API (requires GEMINI_API_KEY env var)

Requirements:
    claude provider: Claude Code CLI installed and authenticated
    anthropic provider: pip install anthropic pyyaml && export ANTHROPIC_API_KEY=...
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
import threading
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
# LLM Provider: Anthropic API (direct SDK — no CLI overhead)
# ---------------------------------------------------------------------------

_anthropic_client = None

def init_anthropic(api_key: str):
    """Initialize Anthropic client."""
    global _anthropic_client
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic not installed. Run: pip install anthropic")
        sys.exit(1)
    _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def call_anthropic(prompt: str, model: str = "claude-sonnet-4-20250514", max_retries: int = 3, timeout: int = 300) -> str | None:
    """Call Anthropic API directly (no CLI spawn overhead)."""
    MODEL_MAP = {
        "sonnet": "claude-sonnet-4-20250514",
        "opus": "claude-opus-4-20250514",
        "haiku": "claude-haiku-4-20250307",
    }
    model_id = MODEL_MAP.get(model, model)

    for attempt in range(max_retries):
        try:
            response = _anthropic_client.messages.create(
                model=model_id,
                max_tokens=16384,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )
            text = "".join(b.text for b in response.content if b.type == "text")
            if text.strip():
                return text.strip()
            return None

        except Exception as e:
            err_str = str(e)
            if "overloaded" in err_str.lower() or "rate" in err_str.lower() or "529" in err_str or "429" in err_str:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif "500" in err_str or "503" in err_str:
                wait = 10 * (attempt + 1)
                print(f"  Server error, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Anthropic API error: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(5)
    return None


# ---------------------------------------------------------------------------
# LLM Provider: Gemini API
# ---------------------------------------------------------------------------

def init_gemini(api_key: str, model_name: str = "gemini-2.5-flash-lite", use_grounding: bool = True):
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
    elif provider == "anthropic":
        return call_anthropic(prompt, model=model, timeout=timeout)
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

PREREQUISITES:
{prereq_str}

RESOURCES:
{res_str}

MILESTONES:
{milestones_str}"""


# ---------------------------------------------------------------------------
# Pass 1: Skeleton generation
# ---------------------------------------------------------------------------

SKELETON_PROMPT = """You are an expert software architect creating a design document outline. This follows the Google Design Doc style used at major tech companies — prose-heavy, focused on decisions and trade-offs, with ZERO code blocks.

{context}

=== REQUIREMENTS ===

Generate a JSON object with the following structure:

{{
  "title": "Project Name: Design Document",
  "overview": "A 2-3 sentence summary of what this system does and the key architectural challenge it solves",
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
      "description": "What this diagram shows and what components/flows to include",
      "type": "flowchart|sequence|class|component|state-machine",
      "relevant_sections": ["section-id-1", "section-id-2"]
    }}
  ]
}}

=== DOCUMENT FORMAT: GOOGLE DESIGN DOC STYLE ===

This is a DESIGN DOCUMENT — the kind a senior engineer writes before a team starts coding. It contains NO code blocks whatsoever. Everything is explained through:
- Prose paragraphs explaining concepts, decisions, and rationale
- Tables describing data structures, interfaces, state transitions, message formats
- Numbered algorithm steps for procedures
- Diagrams for visual understanding
- Concrete walk-through examples narrated in prose

The reader should finish this doc knowing WHAT to build, WHY each decision was made, and HOW components interact — then be able to write the code themselves.

=== SECTION STRUCTURE ===

Follow this structure (adapt section names to the specific project):

1. **Context and Problem Statement** — What problem are we solving? Why is it hard? What existing approaches exist?
2. **Goals and Non-Goals** — What this system must do, and explicitly what it does NOT do
3. **High-Level Architecture** — Component overview, responsibilities, how they connect (with diagram)
4. **Data Model** — All key types/structures described as tables (Name | Type | Description), relationships between them
5. **Component Design sections** (one per major component) — each containing:
   - Responsibility and scope
   - Interface: what it receives, what it returns, described as a table (Method | Input | Output | Description)
   - Internal behavior described as numbered algorithm steps
   - State transitions described as a table (Current State × Event → New State + Action)
   - Design decisions: what alternatives were considered, why this approach was chosen
6. **Interactions and Data Flow** — How components communicate, message formats as tables, sequence of operations
7. **Error Handling and Edge Cases** — Failure modes, detection, recovery strategies
8. **Testing Strategy** — What properties to verify, what scenarios to test
9. **Future Extensions** — What could be added later, what the design accommodates

Each section should map to one or more milestones from the project.

=== DIAGRAM GUIDELINES ===

Suggest 4-8 diagrams:
- System component diagram (always include)
- Data model / type relationship diagram
- State machine diagrams for stateful components
- Sequence diagrams for key interactions
- Flowcharts for complex procedures

Each diagram's "relevant_sections" should list which section IDs it belongs to.

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

SECTION_PROMPT_SEQUENTIAL = """You are an expert software architect writing a section of a design document in the Google Design Doc style.

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

=== NAMING CONVENTIONS (MUST follow these exactly) ===
{naming_conventions}

=== AVAILABLE DIAGRAMS ===
{diagrams_info}

=== WRITING RULES — READ CAREFULLY ===

**ZERO CODE BLOCKS.** This document must contain absolutely NO code blocks (no ``` fences). Not even declarations, not even signatures, not even pseudo-code in code fences. Everything is expressed through:

1. **Prose paragraphs** — Explain concepts, decisions, rationale, and behavior in natural language. Be extremely detailed. Write as if you're a senior engineer explaining the system to a new team member at a whiteboard.

2. **Tables** — Use markdown tables for ALL structured information:
   - Data structures: Name | Type | Description (one row per field)
   - Interfaces/APIs: Method Name | Parameters | Returns | Description
   - State machines: Current State | Event | Next State | Action Taken
   - Message formats: Field | Type | Description
   - Comparison tables: Option | Pros | Cons | Chosen?
   - Error tables: Failure Mode | Detection | Recovery

3. **Numbered algorithm steps** — For procedures/algorithms, use numbered lists in prose (NOT code). Example:
   1. The coordinator generates a unique transaction ID
   2. It writes an INIT record to the transaction log (fsync to ensure durability)
   3. It sends a PREPARE message to each participant containing the transaction ID and operation list
   4. It starts a timeout timer for the voting phase
   ... etc.

4. **Concrete walk-through examples** — Narrate specific scenarios in prose: "Consider a transaction T1 involving participants A, B, and C. The coordinator sends PREPARE to all three. Participant A checks its local locks, finds no conflict, and votes COMMIT. Meanwhile, participant B detects a write-write conflict and votes ABORT..."

5. **Blockquotes** — For key design insights, warnings, or important principles:
   > The critical insight here is that once a participant votes YES, it enters the uncertainty window...

6. **Diagrams** — Reference available diagrams where they aid understanding.

WHAT TO EXPLAIN IN DEPTH:
- WHY each design decision was made and what alternatives were rejected (with reasoning)
- The responsibility of each component: what it owns, what data it holds, what it processes
- Every data structure: list ALL fields with their types and purpose in a table
- Every interface method: parameters, return values, preconditions, postconditions, side effects — in a table
- Every state machine: all states, all transitions, trigger events, actions taken — in a table
- Algorithm logic: step-by-step numbered procedures explained in prose
- Data flow: what enters, how it transforms, where it exits, what gets persisted
- Error scenarios: what can fail, how failures are detected, how recovery works
- Concrete examples that walk through real scenarios

WHAT TO AVOID:
- Code blocks of ANY kind (no ``` fences anywhere)
- Language-specific syntax (don't write "func", "struct", "interface" as keywords — describe them in prose/tables)
- Implementation details that belong in the code itself
- Generic advice ("make sure to handle errors") — instead describe SPECIFIC error scenarios and handling

Use inline `backticks` freely for names (type names, method names, field names, constants) within prose and tables.

=== FORMATTING ===
- Markdown format
- Start with ## for section title, ### for subsections
- Use tables extensively (this is the primary way to convey structured information)
- Use blockquotes for key insights and design principles
- Use numbered lists for procedures and algorithms
- Use bold for key concepts on first introduction
- Reference diagrams: ![Diagram Title](./diagrams/DIAGRAM-ID.svg) — only from AVAILABLE DIAGRAMS above
- Use EXACTLY the names from NAMING CONVENTIONS. Do NOT invent alternatives.

=== CONTEXT FROM PREVIOUS SECTIONS ===
{previous_sections_summary}

=== OUTPUT FORMAT ===
First write the markdown content for this section.
Then, after the markdown content, write a line containing ONLY `---CONVENTIONS---`
After that marker, write a JSON object listing all type names, method names, constants, and key terms you used:

---CONVENTIONS---
{{"types": {{"TypeName": "fields: Field1 Type, Field2 Type, ..."}}, "methods": {{"methodName(params) returns": "brief description"}}, "constants": {{"CONSTANT_NAME": "value or brief description"}}, "terms": {{"preferred term": "brief description"}}}}

The JSON must be on a SINGLE line after the marker. Include ALL names you introduced or referenced.
"""


def build_outline_str(skeleton: dict) -> str:
    """Build a readable outline string from skeleton."""
    lines = []
    for s in skeleton.get("sections", []):
        lines.append(f"- {s['title']}: {s.get('summary', '')}")
        for sub in s.get("subsections", []):
            lines.append(f"  - {sub['title']}: {sub.get('summary', '')}")
    return "\n".join(lines)


def _strip_markdown_fences(content: str) -> str:
    """Strip accidental markdown fence wrapping from LLM output."""
    content = content.strip()
    if content.startswith("```markdown"):
        content = content[len("```markdown"):].strip()
    if content.startswith("```md"):
        content = content[len("```md"):].strip()
    if content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()
    return content


def _merge_conventions(accumulated: dict, new_conventions: dict) -> dict:
    """Merge new naming conventions into accumulated conventions."""
    for category in ("types", "methods", "constants", "terms"):
        existing = accumulated.get(category, {})
        incoming = new_conventions.get(category, {})
        existing.update(incoming)
        accumulated[category] = existing
    return accumulated


def _format_conventions(conventions: dict) -> str:
    """Format accumulated conventions as readable text for the prompt."""
    if not conventions or all(len(v) == 0 for v in conventions.values()):
        return "(No conventions established yet — you are defining them for the first section. Choose clear, consistent names and signatures.)"

    lines = []
    if conventions.get("types"):
        lines.append("Types/Structs/Interfaces (use these EXACT names and fields):")
        for name, desc in conventions["types"].items():
            lines.append(f"  - `{name}`: {desc}")
    if conventions.get("methods"):
        lines.append("Functions/Methods (use these EXACT signatures):")
        for name, desc in conventions["methods"].items():
            lines.append(f"  - `{name}`: {desc}")
    if conventions.get("constants"):
        lines.append("Constants/Enums (use these EXACT names):")
        for name, desc in conventions["constants"].items():
            lines.append(f"  - `{name}`: {desc}")
    if conventions.get("terms"):
        lines.append("Terminology (use these preferred terms):")
        for name, desc in conventions["terms"].items():
            lines.append(f"  - \"{name}\": {desc}")
    return "\n".join(lines)


def _build_diagrams_info(skeleton: dict, section_id: str) -> str:
    """Build diagram info string for a section, showing which diagrams are available."""
    diagrams = skeleton.get("diagrams", [])
    if not diagrams:
        return "(No diagrams planned for this document)"

    lines = []
    for d in diagrams:
        relevant = d.get("relevant_sections", [])
        marker = " ← RELEVANT TO THIS SECTION" if section_id in relevant else ""
        lines.append(
            f"- ID: `{d['id']}` | Title: {d.get('title', '')} | "
            f"Type: {d.get('type', '')}{marker}\n"
            f"  Description: {d.get('description', '')}"
        )

    lines.append(
        "\nTo reference a diagram, use EXACTLY: ![Diagram Title](./diagrams/DIAGRAM-ID.svg)"
        "\nOnly reference diagrams marked as RELEVANT TO THIS SECTION."
    )
    return "\n".join(lines)


def generate_section_sequential(section: dict, skeleton: dict, project: dict,
                                previous_summaries: str, conventions: dict,
                                provider: str, model: str,
                                research: bool, **llm_kwargs) -> tuple[str | None, dict]:
    """Pass 2 (sequential mode): Generate section with conventions tracking.
    Returns (content, new_conventions)."""
    context = get_project_context(project)
    outline = build_outline_str(skeleton)

    section_id = section.get("id", "")

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

    diagrams_info = _build_diagrams_info(skeleton, section_id)

    prompt = SECTION_PROMPT_SEQUENTIAL.format(
        context=context,
        doc_title=skeleton.get("title", ""),
        doc_overview=skeleton.get("overview", ""),
        outline=outline,
        section_title=section.get("title", ""),
        section_summary=section.get("summary", ""),
        subsections_info=sub_info,
        naming_conventions=_format_conventions(conventions),
        diagrams_info=diagrams_info,
        previous_sections_summary=prev_info,
    )

    response = call_llm(prompt, provider, model, research, timeout=300, **llm_kwargs)
    if not response:
        print(f"  ERROR: No response for section '{section.get('title', '')}'")
        return None, {}

    response = response.strip()

    # Split content and conventions
    marker = "---CONVENTIONS---"
    new_conventions = {}

    if marker in response:
        parts = response.split(marker, 1)
        content = parts[0].strip()
        conventions_text = parts[1].strip()

        # Parse conventions JSON
        # Handle possible markdown fence around JSON
        if conventions_text.startswith("```"):
            conventions_text = re.sub(r"^```\w*\n?", "", conventions_text)
            conventions_text = re.sub(r"\n?```$", "", conventions_text)

        # Take first line if multi-line (we asked for single line)
        first_line = conventions_text.split("\n")[0].strip()
        if not first_line.startswith("{"):
            # Try full text
            first_line = conventions_text.strip()

        try:
            new_conventions = json.loads(first_line)
        except json.JSONDecodeError:
            # Try the full text
            try:
                new_conventions = json.loads(conventions_text)
            except json.JSONDecodeError:
                print(f"    (could not parse conventions JSON, continuing)")
    else:
        content = response

    content = _strip_markdown_fences(content)
    return content, new_conventions


# ---------------------------------------------------------------------------
# Pass 3: Diagram generation
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
    print(f"\n  [Pass 1/3] Generating skeleton...")
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
    print(f"\n  [Pass 2/3] Generating {len(sections)} sections...")
    section_contents = []
    previous_summaries = ""

    # Always sequential section generation with conventions tracking
    print(f"  Sequential mode (with naming conventions tracking)")
    accumulated_conventions = {}

    for i, section in enumerate(sections):
        title = section.get("title", f"Section {i+1}")
        print(f"    [{i+1}/{len(sections)}] {title}...")

        content, new_conventions = generate_section_sequential(
            section, skeleton, project, previous_summaries,
            accumulated_conventions,
            args.provider, args.model, args.research, **llm_kwargs
        )

        if content:
            section_contents.append((section, content))
            # Merge conventions
            if new_conventions:
                accumulated_conventions = _merge_conventions(accumulated_conventions, new_conventions)
                conv_count = sum(len(v) for v in accumulated_conventions.values())
                print(f"    OK ({len(content)} chars, {conv_count} conventions tracked)")
            else:
                print(f"    OK ({len(content)} chars)")
            # Build summary of previous sections for context
            summary = content[:200].replace("\n", " ").strip()
            previous_summaries += f"\n- {title}: {summary}..."
        else:
            print(f"    FAIL")

        # Rate limiting for Gemini
        if args.provider == "gemini":
            time.sleep(60.0 / args.rpm)

    # Save conventions for reference
    if accumulated_conventions:
        conv_path = out_dir / "naming-conventions.json"
        with open(conv_path, "w") as f:
            json.dump(accumulated_conventions, f, indent=2, ensure_ascii=False)
        print(f"  Naming conventions saved to {conv_path}")

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

    # --- Pass 3: Diagrams ---
    if not args.no_diagrams and diagrams:
        print(f"\n  [Pass 3/3] Generating {len(diagrams)} diagrams...")
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
            print(f"\n  [Pass 3/3] Skipped (--no-diagrams)")
        else:
            print(f"\n  [Pass 3/3] No diagrams in skeleton")

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
    parser.add_argument("--provider", type=str, default="claude", choices=["claude", "anthropic", "gemini"],
                        help="LLM provider: claude (CLI), anthropic (SDK, fast), gemini (default: claude)")
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
                        help="Number of projects to process in parallel (default: 1). Each project runs sections sequentially for naming consistency.")
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
        if args.provider in ("claude", "anthropic"):
            args.model = "sonnet"
        else:
            args.model = "gemini-2.5-flash-lite"

    # Provider checks
    if args.provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key and not args.dry_run:
            print("ERROR: GEMINI_API_KEY environment variable not set")
            sys.exit(1)
    elif args.provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key and not args.dry_run:
            print("ERROR: ANTHROPIC_API_KEY environment variable not set")
            sys.exit(1)
    elif args.provider == "claude" and not args.dry_run:
        if not shutil.which("claude"):
            print("ERROR: `claude` CLI not found in PATH.")
            sys.exit(1)

    # Load data
    print(f"Loading {YAML_PATH}...")
    with open(YAML_PATH) as f:
        data = yaml.safe_load(f)

    # Initialize providers
    gemini_client = gemini_model = gemini_tools = None
    if args.provider == "anthropic" and not args.dry_run:
        init_anthropic(os.environ["ANTHROPIC_API_KEY"])
    elif args.provider == "gemini" and not args.dry_run:
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
    yaml_lock = threading.Lock()

    def _process_one(pid: str) -> bool:
        """Process a single project (thread-safe)."""
        project = load_project(data, pid)
        if not project:
            print(f"\nWARNING: Project '{pid}' not found in expert_projects, skipping")
            return False

        try:
            ok = process_project(
                pid, project, args,
                gemini_client=gemini_client,
                gemini_model=gemini_model,
                gemini_tools=gemini_tools,
            )

            if ok and not args.dry_run:
                doc_path = f"architecture-docs/{pid}/index.md"
                with yaml_lock:
                    update_yaml(data, pid, doc_path)
                print(f"  Updated YAML: architecture_doc = {doc_path}")
            return ok

        except Exception as e:
            print(f"  EXCEPTION for {pid}: {e}")
            traceback.print_exc()
            return False

    if args.workers > 1:
        print(f"\n  Running {args.workers} projects in parallel...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process_one, pid): pid for pid in project_ids}
            try:
                for future in as_completed(futures):
                    pid = futures[future]
                    try:
                        if future.result():
                            success += 1
                        else:
                            failed += 1
                    except Exception as e:
                        print(f"  EXCEPTION for {pid}: {e}")
                        failed += 1
            except KeyboardInterrupt:
                print(f"\n\nInterrupted! Cancelling remaining projects...")
                executor.shutdown(wait=False, cancel_futures=True)
                print(f"  Completed: {success}, Failed: {failed}")
                sys.exit(0)
    else:
        for pid in project_ids:
            try:
                if _process_one(pid):
                    success += 1
                else:
                    failed += 1
            except KeyboardInterrupt:
                print(f"\n\nInterrupted!")
                print(f"  Completed: {success}, Failed: {failed}")
                sys.exit(0)

    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"  Successful: {success}")
    print(f"  Failed:     {failed}")
    print(f"  Total:      {len(project_ids)}")


if __name__ == "__main__":
    main()
