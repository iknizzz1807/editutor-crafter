#!/usr/bin/env python3
import os, json, re, subprocess, time, yaml, argparse, operator
from typing import Annotated, List, TypedDict, Dict, Any, Optional, Union, cast
from pathlib import Path
from string import Template
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None
from langchain_core.messages import HumanMessage, SystemMessage

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / ".." / "data"
YAML_PATH = DATA_DIR / "projects.yaml"
PROJECTS_DATA_DIR = DATA_DIR / "projects_data"
DEFAULT_OUTPUT = DATA_DIR / "architecture-docs"


def load_project_meta(project_id):
    """Load project metadata from projects_data/ folder (preferred) or projects.yaml (fallback)."""
    # Try projects_data/ first
    yaml_file = PROJECTS_DATA_DIR / f"{project_id}.yaml"
    if yaml_file.exists():
        with open(yaml_file) as f:
            return yaml.safe_load(f)

    # Fallback to projects.yaml
    if YAML_PATH.exists():
        with open(YAML_PATH) as f:
            data = yaml.safe_load(f)
            for d in data.get("domains", []):
                for level in ["beginner", "intermediate", "advanced", "expert"]:
                    for p in d.get("projects", {}).get(level, []):
                        if p.get("id") == project_id:
                            return p
    return None


OUTPUT_BASE = DEFAULT_OUTPUT  # Will be overridden by --output flag if provided
D2_EXAMPLES_DIR = SCRIPT_DIR / ".." / "d2_examples"
INSTRUCTIONS_DIR = SCRIPT_DIR / "instructions"


def load_instruction(name):
    path = INSTRUCTIONS_DIR / f"{name}.md"
    return path.read_text() if path.exists() else f"You are a master {name}."


DOMAIN_PROFILES_DIR = INSTRUCTIONS_DIR / ".." / "domain_profiles"


def load_domain_map():
    map_path = DOMAIN_PROFILES_DIR / "domain_map.yaml"
    if map_path.exists():
        with open(map_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def load_domain_profile(meta):
    """Load domain profile + depth calibration for a project."""
    domain = meta.get("domain", "")
    level = meta.get("level", "intermediate")
    domain_map = load_domain_map()
    profile_name = domain_map.get(domain, "specialized")
    profile_path = DOMAIN_PROFILES_DIR / f"{profile_name}.md"
    profile_content = profile_path.read_text() if profile_path.exists() else ""

    return f"""--- DOMAIN PROFILE ({domain} / {level}) ---
{profile_content}

--- DEPTH CALIBRATION ---
Project level: **{level}**
- beginner: explain everything, skip hardware/formal, focus patterns + API
- intermediate: domain concepts, system-level view, tradeoff analysis
- advanced: deep internals, production concerns, failure modes, optimization
- expert: full depth, byte/bit-level where relevant, research papers
"""


def load_d2_docs():
    """Load ALL documentation files in the d2_examples directory for maximum context."""
    docs = []
    if D2_EXAMPLES_DIR.exists():
        for file_path in sorted(D2_EXAMPLES_DIR.glob("*")):
            if file_path.is_file() and file_path.suffix in [".md", ".txt"]:
                docs.append(
                    f"--- DOCUMENT: {file_path.name} ---\n{file_path.read_text()}"
                )
    return "\n\n".join(docs) if docs else "Standard D2 documentation."


print(">>> Loading D2 docs...", flush=True)
D2_REFERENCE = load_d2_docs()
print(f">>> D2 docs loaded ({len(D2_REFERENCE)} chars)", flush=True)
INSTR_ARCHITECT = load_instruction("architect")
INSTR_EDUCATOR = load_instruction("educator")
INSTR_ARTIST = load_instruction("artist")
INSTR_TDD_PLANNER = load_instruction("tdd_planner")
INSTR_TDD_WRITER = load_instruction("tdd_writer")
INSTR_TDD_ARTIST = load_instruction("tdd_artist")
INSTR_BIBLIOGRAPHER = load_instruction("bibliographer")
print(">>> Instructions loaded", flush=True)

# --- LLM SETUP ---
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "sonnet")  # haiku, sonnet, opus
KILO_MODEL = os.getenv("KILO_MODEL", "kilo/minimax/minimax-m2.5")  # Default kilo model
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "lEJimXOmklwc8z4iQPF0g2yClh9NBs4D")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-2411")

# Global variables (will be initialized by init_llm_provider())
LLM_PROVIDER = None
LLM = None
USE_ANTHROPIC = False


def init_llm_provider():
    """Initialize LLM provider based on environment variables or command line flags."""
    global LLM_PROVIDER, LLM, CLAUDE_MODEL, KILO_MODEL, USE_ANTHROPIC, MISTRAL_MODEL
    print(">>> Initializing LLM provider...", flush=True)

    USE_ANTHROPIC = os.getenv("USE_ANTHROPIC", "false").lower() == "true"
    USE_CLAUDE_CLI = os.getenv("USE_CLAUDE_CLI", "false").lower() == "true"
    USE_KILO_CLI = os.getenv("USE_KILO_CLI", "false").lower() == "true"
    USE_MISTRAL = os.getenv("USE_MISTRAL", "false").lower() == "true"
    USE_MIXED_CLAUDE_GEMINI = (
        os.getenv("USE_MIXED_CLAUDE_GEMINI", "false").lower() == "true"
    )
    USE_MIXED_HEAVY_CLAUDE = (
        os.getenv("USE_MIXED_HEAVY_CLAUDE", "false").lower() == "true"
    )
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "sonnet")
    KILO_MODEL = os.getenv("KILO_MODEL", KILO_MODEL)
    MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", MISTRAL_MODEL)

    if USE_MIXED_CLAUDE_GEMINI or USE_MIXED_HEAVY_CLAUDE:
        LLM_PROVIDER = (
            "mixed-claude-gemini" if USE_MIXED_CLAUDE_GEMINI else "mixed-heavy-claude"
        )
        print(f">>> Provider: MIXED ({LLM_PROVIDER})", flush=True)
        LLM = ChatOpenAI(
            base_url="http://127.0.0.1:7999/v1",
            api_key=os.getenv("GEMINI_PROXY_API_KEY", "mythong2005"),
            model="gemini_cli/gemini-3-flash-preview",
            temperature=1,
            max_completion_tokens=64000,
        )
    elif USE_MISTRAL:
        LLM_PROVIDER = "mistral"
        print(f">>> Provider: MISTRAL ({MISTRAL_MODEL})", flush=True)
        LLM = ChatOpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=os.getenv("MISTRAL_API_KEY"),
            model=MISTRAL_MODEL,
            temperature=1,
            max_completion_tokens=64000,
        )
    elif USE_KILO_CLI:
        LLM_PROVIDER = "kilo-cli"
        print(f">>> Provider: KILO CLI (Model: {KILO_MODEL})", flush=True)
    elif USE_CLAUDE_CLI:
        LLM_PROVIDER = "claude-cli"
        print(f">>> Provider: CLAUDE CODE CLI (Model: {CLAUDE_MODEL})", flush=True)
    elif USE_ANTHROPIC and ChatAnthropic:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        # For Anthropic, we use max_tokens and the 'betas' parameter
        LLM = ChatAnthropic(
            model=ANTHROPIC_MODEL,
            temperature=1,
            max_tokens=64000,
            api_key=api_key,
            timeout=1800,
            betas=["context-1m-2025-08-07"],  # Correct way to enable 1M context
        )
        LLM_PROVIDER = "anthropic"
        print(f">>> Provider: ANTHROPIC ({ANTHROPIC_MODEL})", flush=True)
    else:
        print(">>> Setting up local proxy (Gemini)...", flush=True)
        LLM = ChatOpenAI(
            base_url="http://127.0.0.1:7999/v1",
            api_key=os.getenv("GEMINI_PROXY_API_KEY", "mythong2005"),
            model="gemini_cli/gemini-3-flash-preview",
            temperature=1,
            max_completion_tokens=64000,
        )
        LLM_PROVIDER = "local-proxy"


def replace_reducer(old, new):
    return new


def extract_json(text):
    if not text:
        return None
    try:
        text = str(text)
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return None


def strip_ansi_codes(text):
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", str(text))


def invoke_kilo_cli(messages, max_retries=3):
    system_prompt = ""
    user_prompt = ""
    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_prompt = str(msg.content)
        elif isinstance(msg, HumanMessage):
            user_prompt = str(msg.content)

    full_prompt = (
        f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER REQUEST:\n{user_prompt}"
        if system_prompt
        else user_prompt
    )

    for attempt in range(max_retries):
        try:
            cmd = ["kilo", "run"]
            if KILO_MODEL:
                cmd.extend(["--model", KILO_MODEL])
            cmd.append(full_prompt)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            if result.returncode == 0:

                class MockResponse:
                    def __init__(self, content):
                        self.content = content

                return MockResponse(result.stdout)
            time.sleep((attempt + 1) * 5)
        except Exception:
            time.sleep((attempt + 1) * 5)
    raise Exception("Kilo CLI failed")


def invoke_claude_cli(messages, max_retries=3):
    system_prompt = ""
    user_prompt = ""
    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_prompt = str(msg.content)
        elif isinstance(msg, HumanMessage):
            user_prompt = str(msg.content)

    for attempt in range(max_retries):
        try:
            cmd = [
                "claude",
                "-p",
                "--model",
                CLAUDE_MODEL,
                "--dangerously-skip-permissions",
                "--tools",
                "",
            ]
            if system_prompt:
                cmd.extend(["--system-prompt", system_prompt])
            result = subprocess.run(
                cmd, input=user_prompt, capture_output=True, text=True, timeout=1200
            )
            if result.returncode == 0:

                class MockResponse:
                    def __init__(self, content):
                        self.content = strip_ansi_codes(content)

                return MockResponse(result.stdout)
            time.sleep((attempt + 1) * 20)
        except Exception:
            time.sleep((attempt + 1) * 20)
    raise Exception("Claude CLI failed")


def safe_invoke(messages, max_retries=5, invoke_kwargs=None, provider_override=None):
    actual_provider = provider_override or LLM_PROVIDER

    # If the user explicitly requested Anthropic API, use it for everything
    # except when we are forced to a local-proxy (which usually means specific Gemini logic)
    if LLM_PROVIDER == "anthropic" and actual_provider != "local-proxy":
        actual_provider = "anthropic"

    if actual_provider == "claude-cli":
        return invoke_claude_cli(messages, max_retries)
    if actual_provider == "kilo-cli":
        return invoke_kilo_cli(messages, max_retries)

    kwargs = invoke_kwargs or {}
    if "timeout" not in kwargs:
        kwargs["timeout"] = 1200

    for attempt in range(max_retries):
        try:
            if LLM is None:
                raise Exception("LLM not initialized")
            result = LLM.invoke(messages, **kwargs)
            if result is None or (
                hasattr(result, "content") and result.content is None
            ):
                raise Exception("LLM returned None")
            return result
        except Exception as e:
            err_str = str(e).lower()
            transient = [
                "quota",
                "limit",
                "timeout",
                "429",
                "500",
                "502",
                "503",
                "504",
                "internal server error",
            ]
            if any(t in err_str for t in transient):
                time.sleep((attempt + 1) * 20)
            else:
                raise e
    raise Exception("LLM invocation failed")


class GraphState(TypedDict):
    project_id: Annotated[str, replace_reducer]
    meta: Annotated[Dict[str, Any], replace_reducer]
    blueprint: Annotated[Dict[str, Any], replace_reducer]
    accumulated_md: Annotated[str, replace_reducer]
    current_ms_index: Annotated[int, replace_reducer]
    diagrams_to_generate: Annotated[List[Dict[str, Any]], replace_reducer]
    diagram_attempt: Annotated[int, replace_reducer]
    current_diagram_code: Annotated[Optional[str], replace_reducer]
    current_diagram_meta: Annotated[Optional[Dict[str, Any]], replace_reducer]
    last_error: Annotated[Optional[str], replace_reducer]
    status: Annotated[str, replace_reducer]
    phase: Annotated[str, replace_reducer]  # atlas, tdd, done
    knowledge_map: Annotated[List[str], replace_reducer]
    advanced_contexts: Annotated[List[str], replace_reducer]
    tdd_blueprint: Annotated[Dict[str, Any], replace_reducer]
    tdd_accumulated_md: Annotated[str, replace_reducer]
    tdd_current_mod_index: Annotated[int, replace_reducer]
    tdd_diagrams_to_generate: Annotated[List[Dict[str, Any]], replace_reducer]
    external_reading: Annotated[str, replace_reducer]
    running_criteria: Annotated[List[Dict[str, Any]], replace_reducer]


# --- CHECKPOINT MANAGER ---
class CheckpointManager:
    @staticmethod
    def get_path(project_id):
        return OUTPUT_BASE / project_id / "checkpoint.json"

    @staticmethod
    def save(state: Dict[str, Any]):
        path = CheckpointManager.get_path(state["project_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: v for k, v in state.items()}
        path.write_text(json.dumps(serializable, indent=2))

    @staticmethod
    def load(project_id) -> Optional[Dict]:
        path = CheckpointManager.get_path(project_id)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except:
                return None
        return None


def architect_node(state: GraphState):
    if state.get("blueprint") and state["blueprint"].get("milestones"):
        print(f"  [Agent: Architect] Resuming blueprint for {state['project_id']}...")
        return {"status": "writing", "phase": "atlas"}

    print(f"  [Agent: Architect] Blueprinting {state['project_id']}...", flush=True)
    meta = state["meta"]
    full_yaml_meta = yaml.dump(meta, allow_unicode=True, default_flow_style=False)
    domain_profile = load_domain_profile(meta)

    prompt = f"""
{domain_profile}

{INSTR_ARCHITECT}

--- FULL PROJECT SPECIFICATION (YAML) ---
{full_yaml_meta}

TASK: Output ONLY raw JSON for the Blueprint.
Include every milestone from YAML — 1:1 mapping.
Plan 2-3 diagrams per milestone.
For each milestone: misconception, reveal, cascade (3-5 connections, 1+ cross-domain).
"""
    invoke_args = {}
    if LLM_PROVIDER in [
        "local-proxy",
        "mistral",
        "mixed-claude-gemini",
        "mixed-heavy-claude",
    ]:
        invoke_args["response_format"] = {"type": "json_object"}

    res = safe_invoke(
        [
            SystemMessage(
                content="You are a Principal Systems Architect. Output ONLY valid JSON."
            ),
            HumanMessage(content=prompt),
        ],
        invoke_kwargs=invoke_args,
        provider_override=(
            "claude-cli" if LLM_PROVIDER and LLM_PROVIDER.startswith("mixed") else None
        ),
    )
    print(f"  [Agent: Architect] Response received: {len(res.content)} chars")
    blueprint = extract_json(res.content)
    if not blueprint:
        raise ValueError("Architect failed to return valid JSON")

    proj_dir = OUTPUT_BASE / state["project_id"]
    proj_dir.mkdir(parents=True, exist_ok=True)
    header = f"# {blueprint.get('title', state['project_id'])}\n\n{blueprint.get('overview', '')}\n\n"

    new_state = {
        "blueprint": blueprint,
        "diagrams_to_generate": blueprint.get("diagrams", []),
        "status": "writing",
        "phase": "atlas",
        "accumulated_md": header,
    }
    CheckpointManager.save({**state, **new_state})
    return new_state


def knowledge_mapper_node(state: GraphState):
    print(f"  [Agent: Knowledge Mapper] Updating map...", flush=True)
    last_content = state["accumulated_md"].split("\n\n")[-1]
    prompt = f"Extract 3-5 key technical concepts from: {last_content}. Output ONLY comma-separated list."
    res = safe_invoke([HumanMessage(content=prompt)], provider_override="local-proxy")
    new_terms = [t.strip() for t in str(res.content).split(",") if t.strip()]
    adv_tags = re.findall(r"\[\[ADVANCED_CONTEXT:(.*?)\]\]", state["accumulated_md"])
    return {
        "knowledge_map": list(set(state.get("knowledge_map", []) + new_terms)),
        "advanced_contexts": list(set(state.get("advanced_contexts", []) + adv_tags)),
    }


def writer_node(state: GraphState):
    idx = state["current_ms_index"]
    blueprint = state["blueprint"]
    milestones = blueprint.get("milestones", [])
    if idx >= len(milestones):
        return {"status": "visualizing", "phase": "atlas"}

    ms = milestones[idx]
    ms_id = ms.get("id", f"ms-{idx}")
    ms_title = ms.get("title", f"Milestone {idx + 1}")
    marker = f"<!-- MS_ID: {ms_id} -->"
    if marker in state["accumulated_md"]:
        print(f"  [Agent: Educator] Skipping already generated: {ms_title}")
        return {"current_ms_index": idx + 1}

    print(f"  [Agent: Educator] Writing Atlas Node: {ms_title}...")
    is_claude = LLM_PROVIDER == "mixed-heavy-claude" or LLM_PROVIDER == "claude-cli"

    full_yaml_meta = yaml.dump(
        state["meta"], allow_unicode=True, default_flow_style=False
    )
    domain_profile = load_domain_profile(state["meta"])

    diag_list = "\n".join(
        [
            f"- ID: {d.get('id', '?')} | Title: {d.get('title', '?')}"
            for d in blueprint.get("diagrams", [])
            if isinstance(d, dict)
        ]
    )

    ms_misconception = ms.get("misconception", "N/A")
    ms_reveal = ms.get("reveal", "N/A")
    ms_cascade = ms.get("cascade", [])
    ms_cascade_str = (
        "\n".join(f"  - {c}" for c in ms_cascade) if ms_cascade else "  (none)"
    )
    ms_yaml_criteria = ms.get("yaml_acceptance_criteria", [])
    ms_criteria_str = (
        "\n".join(f"  - {c}" for c in ms_yaml_criteria)
        if ms_yaml_criteria
        else "  (none)"
    )

    prompt = f"""
{domain_profile}

{INSTR_EDUCATOR}

--- GROUND TRUTH PROJECT SPEC (YAML) ---
{full_yaml_meta}

--- FULL ATLAS SO FAR ---
{state["accumulated_md"]}

--- TASK ---
Write Atlas content for milestone: {ms_title}
Summary: {ms.get("summary", ms.get("description", ""))}

Revelation inputs from Architect:
- Misconception: {ms_misconception}
- Reveal: {ms_reveal}

Knowledge Cascade connections to surface:
{ms_cascade_str}

YAML Acceptance Criteria (must be covered):
{ms_criteria_str}

Available diagrams ({{{{DIAGRAM:id}}}}):
{diag_list}

Quality check: Does the reader understand WHY this exists, HOW it works, and WHAT ELSE connects to it?
End with [[CRITERIA_JSON: {{"milestone_id": "{ms_id}", "criteria": [...]}} ]]
"""

    res = safe_invoke(
        [HumanMessage(content=prompt)],
        provider_override="claude-cli" if is_claude else None,
    )
    raw_content = str(res.content).strip()

    # Extract criteria
    new_criteria_list = list(state.get("running_criteria", []))
    criteria_match = re.search(
        r"\[\[CRITERIA_JSON:\s*(\{.*?\})\s*\]\]", raw_content, re.DOTALL
    )
    if criteria_match:
        extracted = extract_json(criteria_match.group(1))
        if extracted:
            new_criteria_list.append(extracted)

    content = re.sub(
        r"\[\[CRITERIA_JSON:.*?\]\]", "", raw_content, flags=re.DOTALL
    ).strip()

    # Dynamic diagrams
    new_diagrams = state["diagrams_to_generate"]
    dynamic_orders = re.findall(r"\[\[DYNAMIC_DIAGRAM:(.*?)\|(.*?)\|(.*?)\]\]", content)
    for d_id, d_title, d_desc in dynamic_orders:
        if not any(d["id"] == d_id.strip() for d in new_diagrams):
            new_diagrams.append(
                {
                    "id": d_id.strip(),
                    "title": d_title.strip(),
                    "description": d_desc.strip(),
                    "anchor_target": ms_id,
                }
            )

    content = re.sub(
        r"\[\[DYNAMIC_DIAGRAM:(.*?)\|.*?\|.*?\]\]", r"{{DIAGRAM:\1}}", content
    )
    full_content = f"\n\n{marker}\n{content}\n<!-- END_MS -->\n"
    print(f"    ✓ Content generated: {len(full_content)} chars")

    new_state = {
        "accumulated_md": state["accumulated_md"] + full_content,
        "current_ms_index": idx + 1,
        "diagrams_to_generate": new_diagrams,
        "running_criteria": new_criteria_list,
        "status": "writing",
        "phase": "atlas",
    }
    CheckpointManager.save({**state, **new_state})
    return new_state


def compiler_node(state: GraphState):
    diag = state["current_diagram_meta"]
    code = state["current_diagram_code"]
    if not diag or not code:
        return {"last_error": None}

    proj_dir = OUTPUT_BASE / state["project_id"]
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "diagrams").mkdir(exist_ok=True)
    d2_path = proj_dir / "diagrams" / f"{diag['id']}.d2"
    d2_path.write_text(re.sub(r'icon:\s*"(https?://.*?)"', "", code))

    res = subprocess.run(
        ["d2", "--layout=elk", str(d2_path), str(d2_path.with_suffix(".svg"))],
        capture_output=True,
        text=True,
    )
    is_tdd = state.get("phase") == "tdd"

    if res.returncode == 0:
        print(f"    ✓ Success: {diag['id']}")
        link = f"\n![{diag.get('title')}](./diagrams/{diag['id']}.svg)\n"
        if is_tdd:
            new_state = {
                "tdd_accumulated_md": state["tdd_accumulated_md"].replace(
                    f"{{{{DIAGRAM:{diag['id']}}}}}", link
                ),
                "tdd_diagrams_to_generate": state["tdd_diagrams_to_generate"][1:],
                "diagram_attempt": 0,
                "last_error": None,
                "current_diagram_code": None,
                "current_diagram_meta": None,
                "status": "tdd_visualizing",
                "phase": "tdd",
            }
        else:
            new_state = {
                "accumulated_md": state["accumulated_md"].replace(
                    f"{{{{DIAGRAM:{diag['id']}}}}}", link
                ),
                "diagrams_to_generate": state["diagrams_to_generate"][1:],
                "diagram_attempt": 0,
                "last_error": None,
                "current_diagram_code": None,
                "current_diagram_meta": None,
                "status": "visualizing",
                "phase": "atlas",
            }
        CheckpointManager.save({**state, **new_state})
        return new_state
    else:
        print(f"    ✗ Failed (Attempt {state['diagram_attempt']}), retrying...")
        if state["diagram_attempt"] >= 5:
            if is_tdd:
                new_state = {
                    "tdd_diagrams_to_generate": state["tdd_diagrams_to_generate"][1:],
                    "diagram_attempt": 0,
                    "last_error": None,
                    "current_diagram_code": None,
                    "current_diagram_meta": None,
                }
            else:
                new_state = {
                    "diagrams_to_generate": state["diagrams_to_generate"][1:],
                    "diagram_attempt": 0,
                    "last_error": None,
                    "current_diagram_code": None,
                    "current_diagram_meta": None,
                }
            CheckpointManager.save({**state, **new_state})
            return new_state
        return {"last_error": res.stderr}


def visualizer_node(state: GraphState):
    if not state.get("diagrams_to_generate"):
        return {"status": "tdd_planning", "phase": "tdd"}
    diag = state["diagrams_to_generate"][0]
    proj_dir = OUTPUT_BASE / state["project_id"]
    svg_path = proj_dir / "diagrams" / f"{diag['id']}.svg"
    # Skip if SVG already exists (already compiled successfully)
    if svg_path.exists():
        print(f"  [Agent: Artist] Skipping existing diagram: {diag['title']}")
        return {"diagrams_to_generate": state["diagrams_to_generate"][1:]}

    attempt = state["diagram_attempt"] + 1
    print(f"  [Agent: Artist] Drawing: {diag['title']} (Attempt {attempt})...")
    prompt = f"{INSTR_ARTIST}\nDOCS:\n{D2_REFERENCE}\nCONTEXT:\n{state['accumulated_md']}\nTASK: Draw '{diag['title']}': {diag.get('description')}"
    if state.get("last_error"):
        prompt += f"\nFIX ERROR: {state['last_error']}\nFAILED CODE:\n{state.get('current_diagram_code')}"

    res = safe_invoke(
        [
            SystemMessage(content="Output ONLY raw D2 code."),
            HumanMessage(content=prompt),
        ]
    )
    code = re.sub(r"```d2\n?|```", "", str(res.content)).strip()
    return {
        "current_diagram_code": code,
        "current_diagram_meta": diag,
        "diagram_attempt": attempt,
        "phase": "atlas",
    }


def tdd_planner_node(state: GraphState):
    if state.get("tdd_blueprint"):
        return {"status": "tdd_writing", "phase": "tdd"}
    print(f"  [Agent: TDD Orchestrator] Planning TDD...")
    full_yaml_meta = yaml.dump(
        state["meta"], allow_unicode=True, default_flow_style=False
    )
    domain_profile = load_domain_profile(state["meta"])

    prompt = f"""
{domain_profile}

{INSTR_TDD_PLANNER}

--- GROUND TRUTH PROJECT SPEC (YAML) ---
{full_yaml_meta}

--- PEDAGOGICAL ATLAS (completed) ---
{state["accumulated_md"]}

TASK: Output ONLY raw JSON for TDD Blueprint.
Follow DOMAIN PROFILE for diagram types and spec detail level.
    Minimum 20 diagrams total. Include implementation phases with hours.
"""
    res = safe_invoke([HumanMessage(content=prompt)])
    print(
        f"  [Agent: TDD Orchestrator] TDD Blueprint received: {len(res.content)} chars"
    )
    tdd_blueprint = extract_json(res.content)

    if not tdd_blueprint:
        raise ValueError("TDD Planner failed")

    tdd_diags = []
    for mod in tdd_blueprint.get("modules", []):
        for d in mod.get("diagrams", []):
            d["anchor_target"] = mod["id"]
            tdd_diags.append(d)

    new_state = {
        "tdd_blueprint": tdd_blueprint,
        "tdd_accumulated_md": f"\n\n# TDD\n\n{tdd_blueprint.get('design_vision')}\n\n",
        "tdd_current_mod_index": 0,
        "tdd_diagrams_to_generate": tdd_diags,
        "status": "tdd_writing",
        "phase": "tdd",
    }
    CheckpointManager.save({**state, **new_state})
    return new_state


def tdd_writer_node(state: GraphState):
    idx = state["tdd_current_mod_index"]
    modules = state["tdd_blueprint"].get("modules", [])
    if idx >= len(modules):
        return {"status": "tdd_visualizing", "phase": "tdd"}

    mod = modules[idx]
    mod_id = mod.get("id", f"mod-{idx}")
    mod_name = mod.get("name", f"Module {idx + 1}")
    marker = f"<!-- TDD_MOD_ID: {mod_id} -->"
    if marker in state["tdd_accumulated_md"]:
        return {"tdd_current_mod_index": idx + 1}

    print(f"  [Agent: TDD Writer] Writing Spec: {mod_name}...")
    full_yaml_meta = yaml.dump(
        state["meta"], allow_unicode=True, default_flow_style=False
    )
    domain_profile = load_domain_profile(state["meta"])
    mod_diag_ids = ", ".join([d.get("id", "?") for d in mod.get("diagrams", [])])

    prompt = f"""
{domain_profile}

{INSTR_TDD_WRITER}

--- GROUND TRUTH PROJECT SPEC (YAML) ---
{full_yaml_meta}

--- PEDAGOGICAL ATLAS ---
{state["accumulated_md"]}

--- TDD PROGRESS ---
{state["tdd_accumulated_md"]}

--- TASK ---
Full Technical Design Specification for module: {mod_name}
Description: {mod.get("description", "")}
Specs: {json.dumps(mod.get("specs", {}), indent=2)}
Phases: {json.dumps(mod.get("implementation_phases", []), indent=2)}

Diagrams ({{{{DIAGRAM:id}}}}): {mod_diag_ids}

Quality check: Could an engineer implement this module from this spec alone?
End with [[CRITERIA_JSON: {{"module_id": "{mod_id}", "criteria": [...]}} ]]
"""
    res = safe_invoke([HumanMessage(content=prompt)])
    raw_content = str(res.content).strip()

    # Extract criteria
    new_criteria_list = list(state.get("running_criteria", []))
    criteria_match = re.search(
        r"\[\[CRITERIA_JSON:\s*(\{.*?\})\s*\]\]", raw_content, re.DOTALL
    )
    if criteria_match:
        extracted = extract_json(criteria_match.group(1))
        if extracted:
            new_criteria_list.append(extracted)

    content = re.sub(
        r"\[\[CRITERIA_JSON:.*?\]\]", "", raw_content, flags=re.DOTALL
    ).strip()

    # Dynamic diagrams
    new_diags = state["tdd_diagrams_to_generate"]
    dynamic = re.findall(r"\[\[DYNAMIC_DIAGRAM:(.*?)\|(.*?)\|(.*?)\]\]", content)
    for d_id, d_title, d_desc in dynamic:
        if not any(d["id"] == d_id.strip() for d in new_diags):
            new_diags.append(
                {
                    "id": d_id.strip(),
                    "title": d_title.strip(),
                    "description": d_desc.strip(),
                    "anchor_target": mod_id,
                }
            )

    content = re.sub(
        r"\[\[DYNAMIC_DIAGRAM:(.*?)\|.*?\|.*?\]\]", r"{{DIAGRAM:\1}}", content
    )
    full_content = f"\n\n{marker}\n{content}\n"
    print(f"    ✓ TDD Module generated: {len(full_content)} chars")

    new_state = {
        "tdd_accumulated_md": state["tdd_accumulated_md"] + full_content,
        "tdd_current_mod_index": idx + 1,
        "tdd_diagrams_to_generate": new_diags,
        "running_criteria": new_criteria_list,
        "status": "tdd_writing",
        "phase": "tdd",
    }
    CheckpointManager.save({**state, **new_state})
    return new_state


def tdd_visualizer_node(state: GraphState):
    if not state.get("tdd_diagrams_to_generate"):
        return {"status": "done", "phase": "done"}
    diag = state["tdd_diagrams_to_generate"][0]
    proj_dir = OUTPUT_BASE / state["project_id"]
    svg_path = proj_dir / "diagrams" / f"{diag['id']}.svg"
    # Skip if SVG already exists (already compiled successfully)
    if svg_path.exists():
        print(f"  [Agent: TDD Artist] Skipping existing diagram: {diag['title']}")
        return {"tdd_diagrams_to_generate": state["tdd_diagrams_to_generate"][1:]}

    attempt = state["diagram_attempt"] + 1
    print(f"  [Agent: TDD Artist] Drawing: {diag['title']} (Attempt {attempt})...")
    prompt = f"{INSTR_TDD_ARTIST}\nDOCS:\n{D2_REFERENCE}\nCONTEXT:\n{state['accumulated_md']}\n{state['tdd_accumulated_md']}\nTASK: Draw '{diag['title']}'"
    res = safe_invoke(
        [
            SystemMessage(content="Output ONLY raw D2 code."),
            HumanMessage(content=prompt),
        ],
        provider_override="local-proxy",
    )
    code = re.sub(r"```d2\n?|```", "", str(res.content)).strip()
    return {
        "current_diagram_code": code,
        "current_diagram_meta": diag,
        "diagram_attempt": attempt,
        "phase": "tdd",
    }


def bibliographer_node(state: GraphState):
    print(f"  [Agent: Bibliographer] Curating...")
    prompt = f"{INSTR_BIBLIOGRAPHER}\nCONTENT:\n{state['accumulated_md']}\n{state['tdd_accumulated_md']}"
    res = safe_invoke([HumanMessage(content=prompt)], provider_override="local-proxy")
    return {"external_reading": str(res.content).strip(), "status": "done"}


# --- GRAPH ---
workflow = StateGraph(GraphState)
workflow.add_node("architect", architect_node)
workflow.add_node("knowledge_mapper", knowledge_mapper_node)
workflow.add_node("writer", writer_node)
workflow.add_node("visualizer", visualizer_node)
workflow.add_node("compiler", compiler_node)
workflow.add_node("tdd_planner", tdd_planner_node)
workflow.add_node("tdd_writer", tdd_writer_node)
workflow.add_node("tdd_visualizer", tdd_visualizer_node)
workflow.add_node("bibliographer", bibliographer_node)

workflow.add_edge(START, "architect")
workflow.add_edge("architect", "writer")


def route_writer(state):
    return (
        "knowledge_mapper"
        if state["current_ms_index"] < len(state["blueprint"].get("milestones", []))
        else "visualizer"
    )


workflow.add_conditional_edges(
    "writer",
    route_writer,
    {"knowledge_mapper": "knowledge_mapper", "visualizer": "visualizer"},
)
workflow.add_edge("knowledge_mapper", "writer")


def route_visualizer(state):
    return "compiler" if state.get("diagrams_to_generate") else "tdd_planner"


workflow.add_conditional_edges(
    "visualizer",
    route_visualizer,
    {"compiler": "compiler", "tdd_planner": "tdd_planner"},
)


def route_tdd_writer(state):
    return (
        "tdd_writer"
        if state["tdd_current_mod_index"]
        < len(state["tdd_blueprint"].get("modules", []))
        else "tdd_visualizer"
    )


workflow.add_conditional_edges(
    "tdd_writer",
    route_tdd_writer,
    {"tdd_writer": "tdd_writer", "tdd_visualizer": "tdd_visualizer"},
)


def route_tdd_visualizer(state):
    return "compiler" if state.get("tdd_diagrams_to_generate") else "bibliographer"


workflow.add_conditional_edges(
    "tdd_visualizer",
    route_tdd_visualizer,
    {"compiler": "compiler", "bibliographer": "bibliographer"},
)


def route_compiler(state):
    is_tdd = state.get("phase") == "tdd"
    if state.get("last_error") and state.get("diagram_attempt", 0) > 0:
        return "tdd_retry" if is_tdd else "visual_retry"
    return "tdd_next" if is_tdd else "visual_next"


workflow.add_conditional_edges(
    "compiler",
    route_compiler,
    {
        "visual_retry": "visualizer",
        "visual_next": "visualizer",
        "tdd_retry": "tdd_visualizer",
        "tdd_next": "tdd_visualizer",
    },
)

workflow.add_edge("tdd_planner", "tdd_writer")
workflow.add_edge("bibliographer", END)
app = workflow.compile()


def generate_project(project_id):
    global OUTPUT_BASE
    print(f"\n>>> V17.1 ATLAS STARTING: {project_id}")
    meta = load_project_meta(project_id)
    if not meta:
        return print(f"Error: {project_id} not found.")

    checkpoint = CheckpointManager.load(project_id)
    initial_state = {
        "project_id": project_id,
        "meta": meta,
        "blueprint": {},
        "accumulated_md": "",
        "current_ms_index": 0,
        "diagrams_to_generate": [],
        "diagram_attempt": 0,
        "last_error": None,
        "status": "planning",
        "phase": "atlas",
        "current_diagram_code": None,
        "current_diagram_meta": None,
        "tdd_blueprint": {},
        "tdd_accumulated_md": "",
        "tdd_current_mod_index": 0,
        "tdd_diagrams_to_generate": [],
        "external_reading": "",
        "knowledge_map": [],
        "advanced_contexts": [],
        "running_criteria": [],
    }
    if checkpoint:
        print(f">>> Resuming phase: {checkpoint.get('phase')}")
        for k, v in checkpoint.items():
            if v is not None:
                initial_state[k] = v

    final_state = app.invoke(cast(GraphState, initial_state))

    proj_dir = OUTPUT_BASE / project_id
    proj_dir.mkdir(parents=True, exist_ok=True)
    full = (
        final_state.get("accumulated_md", "")
        + "\n\n"
        + final_state.get("tdd_accumulated_md", "")
        + "\n\n"
        + final_state.get("external_reading", "")
    )
    (proj_dir / "index.md").write_text(full)

    criteria = final_state.get("running_criteria", [])
    if criteria:
        (proj_dir / "synced_criteria.json").write_text(
            json.dumps(criteria, indent=2, ensure_ascii=False)
        )
        print(f"  ✓ Synced criteria: {len(criteria)} entries")

    cp_path = CheckpointManager.get_path(project_id)
    if cp_path.exists():
        cp_path.unlink()

    subprocess.run(
        ["npm", "run", "generate:html", "--", project_id],
        cwd=SCRIPT_DIR / ".." / "web",
        capture_output=True,
    )
    print(f"  ✓ MASTERPIECE COMPLETE: {project_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", nargs="+", required=True)
    parser.add_argument("--claude-cli", action="store_true")
    parser.add_argument("--mistral", action="store_true", help="Use Mistral AI")
    parser.add_argument(
        "--mixed-claude-gemini",
        action="store_true",
        help="Architect=Claude CLI, Others=Gemini Proxy",
    )
    parser.add_argument(
        "--mixed-heavy-claude",
        action="store_true",
        help="Architect+Educator=Claude CLI, Artist=Gemini Proxy",
    )
    parser.add_argument("--kilo-cli", action="store_true", help="Use Kilo CLI")
    parser.add_argument("--claude-model", default=None)
    parser.add_argument("--mistral-model", default=None, help="Mistral model name")
    parser.add_argument(
        "--kilo-model", default=None, help="Model for Kilo (provider/model)"
    )
    parser.add_argument("--anthropic", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Log raw LLM responses")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: data/architecture-docs)",
    )
    args = parser.parse_args()
    if args.claude_cli:
        os.environ["USE_CLAUDE_CLI"] = "true"
    if args.mistral:
        os.environ["USE_MISTRAL"] = "true"
    if args.mixed_claude_gemini:
        os.environ["USE_MIXED_CLAUDE_GEMINI"] = "true"
    if args.mixed_heavy_claude:
        os.environ["USE_MIXED_HEAVY_CLAUDE"] = "true"
    if args.kilo_cli:
        os.environ["USE_KILO_CLI"] = "true"
    if args.claude_model:
        os.environ["CLAUDE_MODEL"] = args.claude_model
    if args.mistral_model:
        os.environ["MISTRAL_MODEL"] = args.mistral_model
    if args.kilo_model:
        os.environ["KILO_MODEL"] = args.kilo_model
    if args.anthropic:
        os.environ["USE_ANTHROPIC"] = "true"
    if args.debug:
        os.environ["DEBUG_MODE"] = "true"
    if args.output:
        OUTPUT_BASE = Path(args.output).resolve()
    else:
        OUTPUT_BASE = DEFAULT_OUTPUT
    print(f">>> Output directory: {OUTPUT_BASE}", flush=True)
    init_llm_provider()
    for p in args.projects:
        generate_project(p)
