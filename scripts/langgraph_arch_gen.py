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
from pydantic import SecretStr

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
INSTR_SYSTEM_DIAGRAM_ARTIST = load_instruction("system_diagram_artist")
INSTR_PROJECT_STRUCTURE = load_instruction("project_structure")
INSTR_PROJECT_CHARTER = load_instruction("project_charter")
print(">>> Instructions loaded", flush=True)

# --- FEATURE FLAGS ---
ENABLE_SYSTEM_DIAGRAM = os.getenv("ENABLE_SYSTEM_DIAGRAM", "false").lower() == "true"
USE_1M = os.getenv("USE_1M", "false").lower() == "true"
MAX_SYSTEM_DIAGRAM_ITERATIONS = 10

# --- LLM SETUP ---
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "sonnet")  # haiku, sonnet, opus
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "lEJimXOmklwc8z4iQPF0g2yClh9NBs4D")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-2411")

# Global variables (will be initialized by init_llm_provider())
LLM_PROVIDER = None
LLM = None
LLM_FALLBACK = None  # Gemini local-proxy, used for diagram retries when main provider is claude-cli
USE_ANTHROPIC = False


def init_llm_provider():
    """Initialize LLM provider based on environment variables or command line flags."""
    global LLM_PROVIDER, LLM, LLM_FALLBACK, CLAUDE_MODEL, USE_ANTHROPIC, MISTRAL_MODEL
    print(">>> Initializing LLM provider...", flush=True)

    USE_ANTHROPIC = os.getenv("USE_ANTHROPIC", "false").lower() == "true"
    USE_CLAUDE_CLI = os.getenv("USE_CLAUDE_CLI", "false").lower() == "true"
    USE_MISTRAL = os.getenv("USE_MISTRAL", "false").lower() == "true"
    USE_MIXED_CLAUDE_GEMINI = (
        os.getenv("USE_MIXED_CLAUDE_GEMINI", "false").lower() == "true"
    )
    USE_MIXED_HEAVY_CLAUDE = (
        os.getenv("USE_MIXED_HEAVY_CLAUDE", "false").lower() == "true"
    )
    USE_ARCHITECT_CLAUDE = os.getenv("USE_ARCHITECT_CLAUDE", "false").lower() == "true"
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "sonnet")
    MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", MISTRAL_MODEL)

    if USE_MIXED_CLAUDE_GEMINI or USE_MIXED_HEAVY_CLAUDE or USE_ARCHITECT_CLAUDE:
        if USE_MIXED_CLAUDE_GEMINI:
            LLM_PROVIDER = "mixed-claude-gemini"
        elif USE_MIXED_HEAVY_CLAUDE:
            LLM_PROVIDER = "mixed-heavy-claude"
        else:
            LLM_PROVIDER = "architect-claude"
        print(f">>> Provider: MIXED ({LLM_PROVIDER})", flush=True)
        LLM = ChatOpenAI(
            base_url="http://127.0.0.1:7999/v1",
            api_key=SecretStr(os.getenv("GEMINI_PROXY_API_KEY", "mythong2005")),
            model="gemini_cli/gemini-3-flash-preview",
            temperature=1,
            max_completion_tokens=64000,
        )
    elif USE_MISTRAL:
        LLM_PROVIDER = "mistral"
        print(f">>> Provider: MISTRAL ({MISTRAL_MODEL})", flush=True)
        LLM = ChatOpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=SecretStr(os.getenv("MISTRAL_API_KEY") or ""),
            model=MISTRAL_MODEL,
            temperature=1,
            max_completion_tokens=64000,
        )
    elif USE_CLAUDE_CLI:
        LLM_PROVIDER = "claude-cli"
        print(f">>> Provider: CLAUDE CODE CLI (Model: {CLAUDE_MODEL})", flush=True)
        # Init Gemini fallback for diagram retries + system diagram
        LLM_FALLBACK = ChatOpenAI(
            base_url="http://127.0.0.1:7999/v1",
            api_key=SecretStr(os.getenv("GEMINI_PROXY_API_KEY", "mythong2005")),
            model="gemini_cli/gemini-3-flash-preview",
            temperature=1,
            max_completion_tokens=64000,
        )
        print(">>> Gemini fallback initialized for diagram retries", flush=True)
    elif USE_ANTHROPIC and ChatAnthropic:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        # For Anthropic, we use max_tokens and the 'betas' parameter
        LLM = ChatAnthropic(
            model=ANTHROPIC_MODEL,
            temperature=1,
            max_tokens=64000,
            api_key=SecretStr(api_key),
            timeout=1800,
            betas=["context-1m-2025-08-07"],  # Correct way to enable 1M context
        )
        LLM_PROVIDER = "anthropic"
        print(f">>> Provider: ANTHROPIC ({ANTHROPIC_MODEL})", flush=True)
    else:
        print(">>> Setting up local proxy (Gemini)...", flush=True)
        LLM = ChatOpenAI(
            base_url="http://127.0.0.1:7999/v1",
            api_key=SecretStr(os.getenv("GEMINI_PROXY_API_KEY", "mythong2005")),
            model="gemini_cli/gemini-3-flash-preview",
            temperature=1,
            max_completion_tokens=64000,
        )
        LLM_PROVIDER = "local-proxy"


def replace_reducer(old, new):
    return new


def get_val(obj: Any, fields: List[str], default: Any = None) -> Any:
    """Flexible field getter for LLM-generated objects."""
    if not isinstance(obj, dict):
        return default
    for f in fields:
        if f in obj and obj[f]:
            return obj[f]
    return default


def get_id(obj: Dict[str, Any], default: str) -> str:
    """Helper to handle various ID field names from different LLMs."""
    return str(get_val(obj, ["id", "diagram_id", "anchor_id", "mod_id"], default))


def extract_json(text):
    if not text:
        return None
    text = str(text)
    # Fix invalid escape sequences LLMs commonly generate inside JSON strings
    text = re.sub(r"\\0(?![0-9a-fA-F])", r"\\\\0", text)  # \0 -> \\0 (but not \0x hex)
    text = re.sub(r"\\x([0-9a-fA-F]{2})", r"\\u00\1", text)  # \xNN -> \u00NN
    text = re.sub(r",(\s*[}\]])", r"\1", text)          # trailing commas ,} or ,]
    # Remove ALL remaining invalid JSON escapes iteratively (handles cascades like \\' -> \' -> ')
    # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
    for _ in range(3):
        prev = text
        text = re.sub(r'\\([^"\\\/bfnrtu\n\r\t])', lambda m: m.group(1), text)
        if text == prev:
            break

    # Find the first potential start of JSON
    start_brace = text.find("{")
    start_bracket = text.find("[")

    if start_brace == -1 and start_bracket == -1:
        return None

    # Decide which structure to look for first
    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        start_char, end_char, start_idx = "{", "}", start_brace
    else:
        start_char, end_char, start_idx = "[", "]", start_bracket

    count = 0
    for i in range(start_idx, len(text)):
        if text[i] == start_char:
            count += 1
        elif text[i] == end_char:
            count -= 1
            if count == 0:
                json_str = text[start_idx : i + 1]
                try:
                    return json.loads(json_str)
                except:
                    # If this block isn't valid JSON, keep searching
                    continue

    # Fallback to regex if bracket counting fails to find a valid object
    try:
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return None


def strip_code_fence(text: str) -> str:
    """Remove wrapping ```markdown ... ``` or ``` ... ``` that LLMs sometimes add."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip()


def strip_ansi_codes(text):
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", str(text))


def log_llm_interaction(
    project_id, node_label, provider, messages, response_text=None, error=None
):
    if os.getenv("DEBUG_MODE", "false").lower() != "true":
        return

    # Create log in the project output directory
    if project_id:
        log_dir = OUTPUT_BASE / project_id
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "llm_traces.log"
    else:
        log_path = Path("llm_traces.log")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 80

    if response_text is None and error is None:
        input_str = ""
        for msg in messages:
            role = "System" if hasattr(msg, "type") and msg.type == "system" else "User"
            content = getattr(msg, "content", str(msg))
            input_str += f"\n--- {role} ---\n{content}\n"

        log_entry = f"\n{separator}\n[REQUEST] TIMESTAMP: {timestamp} | NODE: {node_label} | PROVIDER: {provider}\n{separator}\nINPUT:{input_str}\n{separator}\n"
    elif error:
        log_entry = f"\n[ERROR] TIMESTAMP: {timestamp} | NODE: {node_label}\nERROR: {error}\n{separator}\n"
    else:
        log_entry = f"\n[RESPONSE] TIMESTAMP: {timestamp} | NODE: {node_label}\nOUTPUT:\n{response_text}\n{separator}\n"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)


def invoke_claude_cli(
    messages, node_label="unknown", project_id=None, max_retries=3, model_override=None
):
    model = model_override or CLAUDE_MODEL
    system_prompt = "You are a specialized technical content engine. Output ONLY the requested Markdown or JSON. NO conversational text, NO preambles (e.g. 'Here is the...', 'I will now...'), NO acknowledgments. If writing a milestone, start directly with the content."
    user_prompt = ""
    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_prompt += "\n" + str(msg.content)
        elif isinstance(msg, HumanMessage):
            user_prompt = str(msg.content)

    log_llm_interaction(project_id, node_label, f"claude-cli ({model})", messages)

    for attempt in range(max_retries):
        try:
            cmd = [
                "claude",
                "-p",
                f"--model={model}",
                "--dangerously-skip-permissions",
                "--tools",
                "",
                "--system-prompt",
                system_prompt,
            ]
            result = subprocess.run(
                cmd, input=user_prompt, capture_output=True, text=True, timeout=1200
            )
            if result.returncode == 0:
                content = strip_ansi_codes(result.stdout).strip()

                # Filter out obvious conversational noise from CLI tool itself
                lines = content.split("\n")
                filtered_lines = [
                    l
                    for l in lines
                    if not l.startswith("Thinking Process:") and not l.strip() == ""
                ]
                content = "\n".join(filtered_lines)

                log_llm_interaction(
                    project_id,
                    node_label,
                    f"claude-cli ({model})",
                    messages,
                    response_text=content,
                )

                class MockResponse:
                    def __init__(self, content):
                        self.content = content

                return MockResponse(content)

            err_msg = f"Attempt {attempt + 1} failed with return code {result.returncode}. Stderr: {result.stderr}"
            log_llm_interaction(
                project_id, node_label, f"claude-cli", messages, error=err_msg
            )
            time.sleep((attempt + 1) * 20)
        except Exception as e:
            log_llm_interaction(
                project_id, node_label, f"claude-cli", messages, error=str(e)
            )
            time.sleep((attempt + 1) * 20)
    raise Exception("Claude CLI failed")


def safe_invoke(
    messages,
    node_label="unknown",
    project_id=None,
    max_retries=5,
    invoke_kwargs=None,
    provider_override=None,
    model_override=None,
):
    # When running claude-cli, use Gemini fallback if provider_override="local-proxy", else claude-cli
    if LLM_PROVIDER == "claude-cli":
        if provider_override == "local-proxy" and LLM_FALLBACK is not None:
            log_llm_interaction(
                project_id, node_label, "local-proxy (fallback)", messages
            )
            kwargs = invoke_kwargs or {}
            if "timeout" not in kwargs:
                kwargs["timeout"] = 1200
            return LLM_FALLBACK.invoke(messages, **kwargs)
        return invoke_claude_cli(
            messages, node_label, project_id, max_retries, model_override=model_override
        )

    actual_provider = provider_override or LLM_PROVIDER

    if actual_provider == "claude-cli":
        return invoke_claude_cli(
            messages, node_label, project_id, max_retries, model_override=model_override
        )

    log_llm_interaction(project_id, node_label, actual_provider, messages)

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

            log_llm_interaction(
                project_id,
                node_label,
                actual_provider,
                messages,
                response_text=str(result.content),
            )
            return result
        except Exception as e:
            err_str = str(e).lower()
            log_llm_interaction(
                project_id, node_label, actual_provider, messages, error=err_str
            )
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
    primary_language: Annotated[str, replace_reducer]  # From architect decision
    accumulated_md: Annotated[str, replace_reducer]
    current_ms_index: Annotated[int, replace_reducer]
    diagrams_to_generate: Annotated[List[Dict[str, Any]], replace_reducer]
    diagram_attempt: Annotated[int, replace_reducer]
    current_diagram_code: Annotated[Optional[str], replace_reducer]
    current_diagram_meta: Annotated[Optional[Dict[str, Any]], replace_reducer]
    last_error: Annotated[Optional[str], replace_reducer]
    status: Annotated[str, replace_reducer]
    phase: Annotated[str, replace_reducer]  # atlas, tdd, system_diagram, done

    tdd_blueprint: Annotated[Dict[str, Any], replace_reducer]
    tdd_accumulated_md: Annotated[str, replace_reducer]
    tdd_current_mod_index: Annotated[int, replace_reducer]
    tdd_diagrams_to_generate: Annotated[List[Dict[str, Any]], replace_reducer]
    external_reading: Annotated[str, replace_reducer]
    running_criteria: Annotated[List[Dict[str, Any]], operator.add]
    explained_concepts: Annotated[List[str], replace_reducer]
    # System Overview Diagram
    system_diagram_d2: Annotated[Optional[str], replace_reducer]
    system_diagram_iteration: Annotated[int, replace_reducer]
    system_diagram_done: Annotated[bool, replace_reducer]
    # Project Structure
    project_structure_md: Annotated[str, replace_reducer]
    # Project Charter
    project_charter_md: Annotated[str, replace_reducer]


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
Explicitly list 'prerequisites' (assumed known vs. must teach first).
Plan diagrams generously — every concept that benefits from visualization should have one. Minimum 2 per milestone, but add as many as the complexity demands. Diagrams are cheap; confusion is not.
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
        node_label="Architect",
        project_id=state["project_id"],
        invoke_kwargs=invoke_args,
        provider_override=(
            "claude-cli"
            if LLM_PROVIDER
            and (LLM_PROVIDER.startswith("mixed") or LLM_PROVIDER == "architect-claude")
            else None
        ),
        model_override="default[1m]" if USE_1M else "default",
    )
    print(f"  [Agent: Architect] Initial response received: {len(res.content)} chars")
    blueprint = extract_json(res.content)
    if not blueprint:
        raise ValueError("Architect failed to return valid JSON")

    warnings = _validate_blueprint(blueprint, meta)

    # --- INTELLIGENT REFINEMENT LOOP ---
    if warnings:
        print(
            f"  [Agent: Architect] Detected {len(warnings)} issues. Requesting refinement...",
            flush=True,
        )
        refine_prompt = f"""
        You are an elite Auditor and Systems Architect. I just reviewed your previous Blueprint and found several CRITICAL issues that must be fixed to meet "Perfect Engineer" standards.
        
        --- ISSUES DETECTED ---
        {chr(10).join(["- " + w for w in warnings])}
        
        --- YOUR PREVIOUS (FLAWED) BLUEPRINT ---
        {json.dumps(blueprint, indent=2)}
        
        TASK:
        1. Address EVERY warning listed above.
        2. Ensure all prerequisites are explicitly listed.
        3. Output the FULL CORRECTED JSON. Do not include conversational text.
        """

        res_refined = safe_invoke(
            [
                SystemMessage(
                    content="You are a Master Architect. Output ONLY corrected raw JSON."
                ),
                HumanMessage(content=refine_prompt),
            ],
            node_label="Architect (Refinement)",
            project_id=state["project_id"],
            invoke_kwargs=invoke_args,
            provider_override=(
                "claude-cli"
                if LLM_PROVIDER and str(LLM_PROVIDER).startswith("mixed")
                else None
            ),
        )
        print(
            f"  [Agent: Architect] Refined response received: {len(res_refined.content)} chars"
        )
        refined_blueprint = extract_json(res_refined.content)
        if refined_blueprint:
            blueprint = refined_blueprint
            # Re-validate once for logging, but accept it
            warnings = _validate_blueprint(blueprint, meta)

    proj_dir = OUTPUT_BASE / state["project_id"]
    proj_dir.mkdir(parents=True, exist_ok=True)
    header = f"# {blueprint.get('title', state['project_id'])}\n\n{blueprint.get('overview', '')}\n\n"

    # Extract primary language from blueprint
    primary_language = _get_primary_language(blueprint, meta)
    print(f"  [Agent: Architect] Primary language: {primary_language}")

    impl = blueprint.get("implementation", {})
    if impl:
        print(f"    Rationale: {impl.get('rationale', 'N/A')}")

    new_state = {
        "blueprint": blueprint,
        "primary_language": primary_language,
        "diagrams_to_generate": blueprint.get("diagrams", []),
        "status": "writing",
        "phase": "atlas",
        "accumulated_md": header,
    }
    CheckpointManager.save({**state, **new_state})
    return new_state


def _validate_blueprint(blueprint, meta):
    """Post-generation validation. Call inside architect_node after extract_json."""
    warnings = []

    # Check prerequisites exist
    if not blueprint.get("prerequisites"):
        warnings.append(
            "Missing 'prerequisites' — reader knowledge assumptions will be unclear"
        )

    # Check implementation language decision
    impl = blueprint.get("implementation", {})
    if not impl.get("primary_language"):
        warnings.append("Missing 'implementation.primary_language' — will default to C")

    # Check concept count
    for ms in blueprint.get("milestones", []):
        count = ms.get("concept_count", 0)
        if count > 5:
            warnings.append(
                f"Milestone '{ms.get('title')}' has {count} concepts — consider splitting"
            )

    for w in warnings:
        print(f"  [VALIDATION WARN] {w}")

    return warnings


def _get_primary_language(blueprint: Dict, meta: Dict) -> str:
    """Extract primary language from blueprint, with fallback logic."""
    impl = blueprint.get("implementation", {})
    primary = impl.get("primary_language")

    if primary:
        return primary

    # Fallback 1: Check YAML recommendations
    langs = meta.get("languages", {})
    recommended = langs.get("recommended", [])
    if recommended:
        # Pick first recommended language
        return recommended[0]

    # Fallback 2: Default based on domain
    domain = meta.get("domain", "systems")
    domain_defaults = {
        "systems": "C",
        "database": "C",
        "compiler": "Rust",
        "distributed": "Go",
        "web": "TypeScript",
        "ml": "Python",
        "game": "C++",
    }
    return domain_defaults.get(domain, "C")


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

    # --- QUALITY GATE / SKIP LOGIC ---
    if marker in state["accumulated_md"]:
        # Extract the content for this milestone to check its validity
        parts = state["accumulated_md"].split(marker)
        if len(parts) > 1:
            ms_content = parts[1].split("<!-- END_MS -->")[0].strip()
            # If content is substantial (>1000 chars) and doesn't look like a preamble
            bad_starts = ["Let me", "I will", "I'll now", "I am now"]
            if len(ms_content) > 1000 and not any(
                ms_content.startswith(s) for s in bad_starts
            ):
                print(f"  [Agent: Educator] Skipping already generated: {ms_title}")
                return {"current_ms_index": idx + 1}
            else:
                print(
                    f"  [Agent: Educator] Existing content for {ms_title} is invalid or too thin. Rewriting..."
                )
                # Remove the failed segment from accumulated_md to allow rewrite
                before = parts[0]
                after = (
                    parts[1].split("<!-- END_MS -->", 1)[1]
                    if "<!-- END_MS -->" in parts[1]
                    else ""
                )
                state["accumulated_md"] = before + after

    print(f"  [Agent: Educator] Writing Atlas Node: {ms_title}...")
    is_claude = LLM_PROVIDER == "mixed-heavy-claude" or LLM_PROVIDER == "claude-cli"

    full_yaml_meta = yaml.dump(
        state["meta"], allow_unicode=True, default_flow_style=False
    )
    domain_profile = load_domain_profile(state["meta"])

    diag_list = "\n".join(
        [
            f"- ID: {get_id(d, 'unknown')} | Title: {d.get('title', '?')} | Type: {d.get('diagram_type', 'architecture')}"
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

    # Get acceptance criteria from ORIGINAL YAML (not blueprint - Architect doesn't copy them)
    yaml_milestones = state["meta"].get("milestones", [])
    yaml_ms = None
    for ym in yaml_milestones:
        if ym.get("id") == ms_id or ym.get("title") == ms_title:
            yaml_ms = ym
            break

    ms_yaml_criteria = yaml_ms.get("acceptance_criteria", []) if yaml_ms else []
    if isinstance(ms_yaml_criteria, str):
        ms_yaml_criteria = [ms_yaml_criteria]
    ms_criteria_str = (
        "\n".join(f"  - {c}" for c in ms_yaml_criteria)
        if ms_yaml_criteria
        else "  (none specified)"
    )

    context_md = state["accumulated_md"]

    prereqs = blueprint.get("prerequisites", {})
    must_teach = prereqs.get("must_teach_first", [])
    assumed = prereqs.get("assumed_known", [])

    ms_prereqs = []
    for p in must_teach:
        when = p.get("when", "").lower()
        if (
            ms_title.lower() in when
            or f"milestone {idx}" in when
            or f"milestone {idx + 1}" in when
            or ms_id.lower() in when
        ):
            ms_prereqs.append(p)

    prereqs_str = (
        "\n".join(
            f"  - MUST place [[EXPLAIN:{p['concept'].lower().replace(' ', '-')}|{p['concept']}]] marker in your text where this concept first appears (depth: {p.get('depth', 'basic')})"
            for p in ms_prereqs
        )
        if ms_prereqs
        else "  (none — mark any tangential concept you encounter with [[EXPLAIN:concept-id|description]])"
    )

    assumed_str = ", ".join(assumed) if assumed else "(not specified)"
    already_explained = ", ".join(state.get("explained_concepts", [])[-20:])

    # Get primary language for code examples
    primary_language = state.get("primary_language", "C")
    impl = blueprint.get("implementation", {})
    language_rationale = impl.get("rationale", "")

    # PROMPT RE-ENGINEERING: Move instructions to the end to avoid "Lost in the Middle"
    prompt = f"""
{domain_profile}

--- GROUND TRUTH PROJECT SPEC (YAML) ---
{full_yaml_meta}

--- IMPLEMENTATION LANGUAGE (BINDING) ---
Primary Language: **{primary_language}**
Rationale: {language_rationale}

ALL code examples in this milestone MUST use {primary_language}. This is a BINDING decision.
- Use {primary_language} syntax for all structs, functions, and code blocks
- Follow {primary_language} naming conventions
- Pseudocode allowed for algorithm explanation, but follow with {primary_language} implementation

--- READER CONTEXT ---
Assumed known: {assumed_str}
Concepts Architect flagged for this milestone:
{prereqs_str}
Concepts already explained in earlier milestones: {already_explained or "(none yet)"}

--- RECENT ATLAS CONTEXT ---
{context_md}

--- YOUR TASK ---
Write the full chapter for: **{ms_title}**
Summary: {ms.get("summary", ms.get("description", ""))}

Revelation inputs from Architect:
- Misconception: {ms_misconception}
- Reveal: {ms_reveal}

Knowledge Cascade connections to surface:
{ms_cascade_str}

YAML Acceptance Criteria (ANCHOR - must be covered, but refine based on your content):
{ms_criteria_str}

Available diagrams (use {{{{DIAGRAM:id}}}}):
{diag_list}

{INSTR_EDUCATOR}

CRITICAL RULES:
1. **NO CONVERSATION**: Output ONLY the Markdown content. Do NOT say "I will now write..." or "Let me read...". Start immediately with the content.
2. **MINIMUM LENGTH**: This is an {state["meta"].get("difficulty", "intermediate")} project. Write at least 10,000 characters of deep technical content.
3. **EXACT ID**: Use the exact milestone ID '{ms_id}' in your CRITERIA_JSON block.
4. **EXPLAIN MARKERS**: As you write, whenever you introduce a concept that is important but tangential (would bloat the narrative if explained inline), place `[[EXPLAIN:concept-id|Short description of concept]]` right where it first appears. The Explainer agent will replace it with a Foundation block. Use this freely — it is NOT optional.

After writing the content, generate CRITERIA_JSON:
- Start from YAML Acceptance Criteria above as your base
- Refine each criterion to be more specific and technical based on what you actually wrote
- Add new criteria for any important technical details you covered that weren't in the original
- Each criterion must be measurable and testable
- Remove any criteria that became irrelevant

End with [[CRITERIA_JSON: {{"milestone_id": "{ms_id}", "criteria": [...]}} ]]
"""

    res = safe_invoke(
        [HumanMessage(content=prompt)],
        node_label=f"Educator (MS: {ms_title})",
        project_id=state["project_id"],
        provider_override="claude-cli" if is_claude else None,
    )
    raw_content = str(res.content).strip()

    # ---- Extract criteria ----
    new_criteria_to_add = []
    criteria_tags = re.findall(r"\[\[CRITERIA_JSON:(.*?)\]\]", raw_content, re.DOTALL)
    for tag_content in criteria_tags:
        extracted = extract_json(tag_content)
        if extracted:
            new_criteria_to_add.append(extracted)

    content = re.sub(
        r"\[\[CRITERIA_JSON:.*?\]\]", "", raw_content, flags=re.DOTALL
    ).strip()

    # ---- Dynamic diagrams ----
    new_diagrams = list(state["diagrams_to_generate"])
    for match in re.finditer(
        r"\[\[DYNAMIC_DIAGRAM:([\w-]+)\|([^|]+)(?:\|type:([\w-]+))?\|([^\]]+)\]\]",
        content,
    ):
        d_id, d_title, d_type, d_desc = match.groups()
        if not any(d["id"] == d_id.strip() for d in new_diagrams):
            new_diagrams.append(
                {
                    "id": d_id.strip(),
                    "title": d_title.strip(),
                    "diagram_type": (d_type or "architecture").strip(),
                    "description": d_desc.strip(),
                    "anchor_target": ms_id,
                }
            )

    content = re.sub(
        r"\[\[DYNAMIC_DIAGRAM:(.*?)\]\]",
        lambda m: f"{{{{DIAGRAM:{m.group(1)}}}}}",
        content,
    )

    full_content = f"\n\n{marker}\n{content}\n<!-- END_MS -->\n"
    print(f"    ✓ Content generated: {len(full_content)} chars")

    new_state = {
        "accumulated_md": state["accumulated_md"] + full_content,
        "current_ms_index": idx + 1,
        "diagrams_to_generate": new_diagrams,
        "running_criteria": new_criteria_to_add,  # ONLY RETURN NEW ONES (operator.add will handle)
        "status": "visualizing",
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
        diag_id = diag["id"]
        print(f"    ✓ Success: {diag_id}")
        link = f"\n![{diag.get('title')}](./diagrams/{diag_id}.svg)\n"
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
    diag_id = get_id(diag, "unknown")

    # Safety check for missing ID
    if diag_id == "unknown":
        print(f"  [Agent: Artist] Warning: Skipping invalid diagram metadata: {diag}")
        return {"diagrams_to_generate": state["diagrams_to_generate"][1:]}

    svg_path = proj_dir / "diagrams" / f"{diag_id}.svg"

    # Skip if SVG already exists
    if (
        svg_path.exists()
        and f"{{{{DIAGRAM:{diag_id}}}}}" not in state["accumulated_md"]
    ):
        print(
            f"  [Agent: Artist] Skipping existing diagram: {diag.get('title', 'Unknown')}"
        )
        return {"diagrams_to_generate": state["diagrams_to_generate"][1:]}

    attempt = state["diagram_attempt"] + 1
    is_retry = state.get("last_error") and attempt > 1

    # --- Build type-aware instructions ---
    diag_type = diag.get("diagram_type", "architecture")
    trace_example = diag.get("trace_example", "")

    type_hint = f"\nDIAGRAM TYPE: {diag_type}"
    type_hint += f"\nFollow the '{diag_type}' routing rules from your instructions."
    if trace_example:
        type_hint += f"\nTRACE EXAMPLE (use these EXACT values): {trace_example}"

    if is_retry:
        # ---- RETRY: Minimal context — fix the D2 error only ----
        print(
            f"  [Agent: Artist] RETRY {attempt}: Fixing '{diag.get('title', 'Diagram')}'..."
        )
        prompt = f"""{INSTR_ARTIST}

D2 REFERENCE DOCS:
{D2_REFERENCE}
{type_hint}

TASK: Fix the D2 compilation error below. Keep the same structure and content — only fix what caused the error.

DIAGRAM: '{diag.get("title", "Diagram")}'

FAILED CODE:
```d2
{state.get("current_diagram_code", "")}
```

COMPILER ERROR:
{state["last_error"]}

Output ONLY the corrected D2 code."""

    else:
        # ---- FIRST ATTEMPT: Full context ----
        print(
            f"  [Agent: Artist] Drawing: {diag.get('title', 'Diagram')} (Attempt {attempt})..."
        )
        context_full = state["accumulated_md"]

        # Provide the satellite map code if available for consistency
        satellite_map_marker = "![diag-satellite-overview]"
        satellite_context = ""
        if satellite_map_marker in state["accumulated_md"]:
            satellite_context = "\nREFERENCE SATELLITE MAP: Use consistent IDs from the existing Master Map found in the beginning of CONTEXT."

        prompt = (
            f"{INSTR_ARTIST}\n"
            f"DOCS:\n{D2_REFERENCE}\n"
            f"CONTEXT:\n{context_full}\n"
            f"{type_hint}{satellite_context}\n"
            f"TASK: Draw '{diag.get('title', 'Untitled')}': {diag.get('description', '')}"
        )

    res = safe_invoke(
        [
            SystemMessage(content="Output ONLY raw D2 code."),
            HumanMessage(content=prompt),
        ],
        node_label=f"Artist (Diag: {diag.get('title')})",
        project_id=state["project_id"],
        provider_override="local-proxy" if is_retry else None,
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
    is_claude = LLM_PROVIDER == "mixed-heavy-claude" or LLM_PROVIDER == "claude-cli"
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
    res = safe_invoke(
        [HumanMessage(content=prompt)],
        node_label="TDD Planner",
        project_id=state["project_id"],
        provider_override="claude-cli" if is_claude else None,
    )
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

    # --- QUALITY GATE / SKIP LOGIC ---
    if marker in state["tdd_accumulated_md"]:
        parts = state["tdd_accumulated_md"].split(marker)
        if len(parts) > 1:
            mod_content = parts[1].split("<!-- END_TDD_MOD -->")[0].strip()
            if len(mod_content) > 1000 and not any(
                mod_content.startswith(s) for s in ["Let me", "I will", "I'll now"]
            ):
                return {"tdd_current_mod_index": idx + 1}
            else:
                print(
                    f"  [Agent: TDD Writer] Content for {mod_name} is too thin. Rewriting..."
                )
                before = parts[0]
                after = (
                    parts[1].split("<!-- END_TDD_MOD -->", 1)[1]
                    if "<!-- END_TDD_MOD -->" in parts[1]
                    else ""
                )
                state["tdd_accumulated_md"] = before + after

    print(f"  [Agent: TDD Writer] Writing Spec: {mod_name}...")
    full_yaml_meta = yaml.dump(
        state["meta"], allow_unicode=True, default_flow_style=False
    )
    domain_profile = load_domain_profile(state["meta"])
    mod_diag_ids = ", ".join([d.get("id", "?") for d in mod.get("diagrams", [])])

    # Get primary language for code examples
    primary_language = state.get("primary_language", "C")
    impl = state.get("blueprint", {}).get("implementation", {})
    language_rationale = impl.get("rationale", "")
    style_guide = impl.get("style_guide", "")

    prompt = f"""
{domain_profile}

--- GROUND TRUTH PROJECT SPEC (YAML) ---
{full_yaml_meta}

--- IMPLEMENTATION LANGUAGE (BINDING) ---
Primary Language: **{primary_language}**
Rationale: {language_rationale}
Style Guide: {style_guide}

ALL code in this TDD MUST use {primary_language}. This is a BINDING decision.
- Struct/class definitions in {primary_language} syntax
- Function signatures in {primary_language} syntax  
- Memory layouts with {primary_language} types
- Follow {primary_language} naming conventions

--- PEDAGOGICAL ATLAS ---
{state["accumulated_md"]}

--- TDD PROGRESS ---
{state["tdd_accumulated_md"]}

--- TASK ---
Full Technical Design Specification for module: **{mod_name}**
Description: {mod.get("description", "")}
Specs: {json.dumps(mod.get("specs", {}), indent=2)}
Phases: {json.dumps(mod.get("implementation_phases", []), indent=2)}

Diagrams ({{{{DIAGRAM:id}}}}): {mod_diag_ids}

{INSTR_TDD_WRITER}

CRITICAL RULES:
1. **NO CONVERSATION**: Output ONLY the technical specification. Do NOT introduce yourself or the task.
2. **BYTE-LEVEL PRECISION**: If this is an Expert project, you MUST provide tables for binary layouts, memory addresses, and register states.
3. **MINIMUM LENGTH**: Output at least 8,000 characters of deep technical specification.
4. **EXACT ID**: Use the exact module ID '{mod_id}' in your CRITERIA_JSON.

End with [[CRITERIA_JSON: {{"module_id": "{mod_id}", "criteria": [...]}} ]]
"""
    res = safe_invoke(
        [HumanMessage(content=prompt)],
        node_label=f"TDD Writer (Mod: {mod_name})",
        project_id=state["project_id"],
    )
    raw_content = str(res.content).strip()

    # ---- Extract criteria ----
    new_criteria_to_add = []
    criteria_tags = re.findall(r"\[\[CRITERIA_JSON:(.*?)\]\]", raw_content, re.DOTALL)
    for tag_content in criteria_tags:
        extracted = extract_json(tag_content)
        if extracted:
            new_criteria_to_add.append(extracted)

    content = re.sub(
        r"\[\[CRITERIA_JSON:.*?\]\]", "", raw_content, flags=re.DOTALL
    ).strip()

    # Dynamic diagrams
    new_diags = list(state["tdd_diagrams_to_generate"])
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
    full_content = f"\n\n{marker}\n{content}\n<!-- END_TDD_MOD -->\n"
    print(f"    ✓ TDD Module generated: {len(full_content)} chars")

    new_state = {
        "tdd_accumulated_md": state["tdd_accumulated_md"] + full_content,
        "tdd_current_mod_index": idx + 1,
        "tdd_diagrams_to_generate": new_diags,
        "running_criteria": new_criteria_to_add,  # ONLY RETURN NEW ONES
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
    diag_id = get_id(diag, "unknown")
    if diag_id == "unknown":
        print(
            f"  [Agent: TDD Artist] Warning: Skipping invalid diagram metadata: {diag}"
        )
        return {"tdd_diagrams_to_generate": state["tdd_diagrams_to_generate"][1:]}

    svg_path = proj_dir / "diagrams" / f"{diag_id}.svg"

    if (
        svg_path.exists()
        and f"{{{{DIAGRAM:{diag_id}}}}}" not in state["tdd_accumulated_md"]
    ):
        print(
            f"  [Agent: TDD Artist] Skipping existing diagram: {diag.get('title', 'Unknown')}"
        )
        return {"tdd_diagrams_to_generate": state["tdd_diagrams_to_generate"][1:]}

    attempt = state["diagram_attempt"] + 1
    is_retry = state.get("last_error") and attempt > 1

    # Type-aware instructions
    diag_type = diag.get("diagram_type", diag.get("type", "architecture"))
    trace_example = diag.get("trace_example", "")

    type_hint = f"\nDIAGRAM TYPE: {diag_type}"
    type_hint += f"\nFollow the '{diag_type}' routing rules from your instructions."
    if trace_example:
        type_hint += f"\nTRACE EXAMPLE (use these EXACT values): {trace_example}"

    if is_retry:
        # RETRY: Minimal context
        print(
            f"  [Agent: TDD Artist] RETRY {attempt}: Fixing '{diag.get('title', 'Diagram')}'..."
        )
        prompt = f"""{INSTR_TDD_ARTIST}

D2 REFERENCE DOCS:
{D2_REFERENCE}
{type_hint}

TASK: Fix the D2 compilation error. Keep same structure — only fix the error.

DIAGRAM: '{diag.get("title", "Diagram")}'

FAILED CODE:
```d2
{state.get("current_diagram_code", "")}
```

COMPILER ERROR:
{state["last_error"]}

Output ONLY the corrected D2 code."""

    else:
        # FIRST ATTEMPT: Full context
        print(
            f"  [Agent: TDD Artist] Drawing: {diag.get('title', 'Diagram')} (Attempt {attempt})..."
        )
        atlas_full = state["accumulated_md"]
        tdd_full = state["tdd_accumulated_md"]

        # Satellite consistency
        satellite_context = "\nREFERENCE SATELLITE MAP: Use consistent IDs from the Master Map found in the ATLAS CONTEXT."

        prompt = (
            f"{INSTR_TDD_ARTIST}\n"
            f"DOCS:\n{D2_REFERENCE}\n"
            f"CONTEXT:\n{atlas_full}\n{tdd_full}\n"
            f"{type_hint}{satellite_context}\n"
            f"TASK: Draw '{diag.get('title', 'Untitled')}': {diag.get('description', '')}"
        )

    res = safe_invoke(
        [
            SystemMessage(content="Output ONLY raw D2 code."),
            HumanMessage(content=prompt),
        ],
        node_label=f"TDD Artist (Diag: {diag.get('title')})",
        project_id=state["project_id"],
        provider_override="local-proxy" if is_retry else None,
    )
    code = re.sub(r"```d2\n?|```", "", str(res.content)).strip()
    return {
        "current_diagram_code": code,
        "current_diagram_meta": diag,
        "diagram_attempt": attempt,
        "phase": "tdd",
    }


# --- SYSTEM OVERVIEW DIAGRAM NODES ---
def system_diagram_writer_node(state: GraphState):
    """
    Generate initial system overview diagram from Atlas + TDD content.
    Uses Gemini local proxy by default for cost efficiency.
    """
    if not ENABLE_SYSTEM_DIAGRAM:
        return {"phase": "done", "status": "done"}

    print(f"  [Agent: System Diagram Writer] Creating initial system overview...")
    atlas_content = state["accumulated_md"]
    tdd_content = state["tdd_accumulated_md"]
    project_name = state["meta"].get("name", state["project_id"])
    primary_language = state.get("primary_language", "C")

    prompt = f"""{INSTR_SYSTEM_DIAGRAM_ARTIST}

--- PROJECT: {project_name} ---
Primary Language: {primary_language}

--- ATLAS CONTENT (Educational) ---
{atlas_content}

--- TDD CONTENT (Technical Specs) ---
{tdd_content}

--- D2 REFERENCE ---
{D2_REFERENCE}

TASK: Create a comprehensive system overview diagram that captures ALL major components and relationships from the content above.

QUALITY CHECKLIST (ALL 10 must pass - you are the ONLY generation, no retries):
1. ☐ Every major component has: name, file reference (e.g., "pager.c"), milestone link
2. ☐ Struct/class definitions show: byte offsets, field types, total size
3. ☐ Methods show: return type, parameters, brief description
4. ☐ Data flow arrows labeled with: type, size, example value
5. ☐ Scale indicators present ("4KB page", "64 bytes", "cache line")
6. ☐ 2D grid layout used (horizontal layers + vertical detail within)
7. ☐ No overlapping nodes, readable in PDF
8. ☐ All milestone IDs from Atlas included
9. ☐ Code blocks use primary language ({primary_language})
10. ☐ At least 3 levels of detail (layer → component → struct/method)

REQUIREMENTS:
- This diagram must be IMPLEMENTATION-READY (code-able blueprint)
- An engineer should be able to implement directly from this diagram
- Use ONLY light themes (0, 1, 3, 4, 5, 6, 100, 104) - NO dark themes
- DO NOT use `near` key at non-root level
- DO NOT use `near` with object references when using dagre layout
- Ensure all md block strings are properly closed

Output ONLY valid D2 code. No markdown fences, no explanations."""

    is_claude = LLM_PROVIDER == "mixed-heavy-claude" or LLM_PROVIDER == "claude-cli"
    res = safe_invoke(
        [HumanMessage(content=prompt)],
        node_label="System Diagram Writer",
        project_id=state["project_id"],
        provider_override="claude-cli" if is_claude else "local-proxy",
    )
    d2_code = re.sub(r"```d2\n?|```", "", str(res.content)).strip()

    print(f"    ✓ Initial D2 generated: {len(d2_code)} chars")
    return {
        "system_diagram_d2": d2_code,
        "system_diagram_iteration": 0,
        "system_diagram_done": False,
        "phase": "system_diagram",
        "status": "system_diagram_refining",
    }


def system_diagram_refiner_node(state: GraphState):
    """
    Review and refine the system diagram until complete.
    Agent decides when done based on completeness check.
    Max 10 iterations.
    """
    iteration = state.get("system_diagram_iteration", 0) + 1
    current_d2 = state.get("system_diagram_d2", "")
    atlas_content = state["accumulated_md"]
    tdd_content = state["tdd_accumulated_md"]
    project_name = state["meta"].get("name", state["project_id"])

    print(
        f"  [Agent: System Diagram Refiner] Iteration {iteration}/{MAX_SYSTEM_DIAGRAM_ITERATIONS}..."
    )

    # Get primary language for diagram code blocks
    primary_language = state.get("primary_language", "C")

    prompt = f"""{INSTR_SYSTEM_DIAGRAM_ARTIST}

--- PROJECT: {project_name} ---
Primary Language: {primary_language}

--- CURRENT D2 DIAGRAM ---
```d2
{current_d2}
```

--- D2 REFERENCE ---
{D2_REFERENCE}

QUALITY CHECKLIST (ALL must pass for done=true):
1. ☐ Every major component has: name, file reference (e.g., "pager.c"), link to milestone
2. ☐ Struct/class definitions show: byte offsets, field types, total size
3. ☐ Methods show: return type, parameters, brief description
4. ☐ Data flow arrows labeled with: type, size, example value
5. ☐ Scale indicators present ("4KB page", "64 bytes", "cache line")
6. ☐ 2D grid layout used (horizontal layers + vertical detail within)
7. ☐ No overlapping nodes, readable in PDF
8. ☐ All milestone IDs from Atlas included with links
9. ☐ Code blocks use primary language ({primary_language})
10. ☐ At least 3 levels of detail (layer → component → struct/method)

OUTPUT JSON (no markdown, just raw JSON):
{{
  "done": true/false,
  "d2": "complete updated D2 code here (only if not done)",
  "feedback": "list which checklist items failed and what you fixed",
  "checklist_passed": [1, 2, 3, ...]
}}

CRITICAL RULES:
- Set done=true ONLY when ALL 10 checklist items pass
- If ANY item fails → done=false, output FULL improved D2 code
- The D2 code MUST compile successfully
- This diagram must be IMPLEMENTATION-READY (code-able blueprint)
- An engineer should be able to implement directly from this diagram"""

    invoke_args = {}
    if LLM_PROVIDER in [
        "local-proxy",
        "mistral",
        "mixed-claude-gemini",
        "mixed-heavy-claude",
    ]:
        invoke_args["response_format"] = {"type": "json_object"}

    res = safe_invoke(
        [HumanMessage(content=prompt)],
        node_label=f"System Diagram Refiner (iter {iteration})",
        project_id=state["project_id"],
        invoke_kwargs=invoke_args,
        provider_override="local-proxy",
    )

    result = extract_json(res.content)
    if not result:
        # Fallback: try to extract D2 directly if JSON parse fails
        raw = str(res.content).strip()
        d2_match = re.search(r'"d2"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
        if d2_match:
            result = {
                "done": "done" in raw.lower() and '"done":true' in raw.replace(" ", ""),
                "d2": d2_match.group(1).replace("\\n", "\n").replace('\\"', '"'),
                "feedback": "Extracted from malformed JSON",
            }

    if result:
        is_done = (
            result.get("done", False) or iteration >= MAX_SYSTEM_DIAGRAM_ITERATIONS
        )
        new_d2 = result.get("d2", current_d2)
        feedback = result.get("feedback", "")

        # Clean D2 code
        new_d2 = re.sub(r"```d2\n?|```", "", new_d2).strip()

        print(f"    Feedback: {feedback[:100]}...")
        print(f"    Done: {is_done}, D2 size: {len(new_d2)} chars")

        return {
            "system_diagram_d2": new_d2,
            "system_diagram_iteration": iteration,
            "system_diagram_done": is_done,
            "phase": "system_diagram" if not is_done else "done",
            "status": "system_diagram_refining" if not is_done else "done",
        }

    # If JSON extraction failed completely, mark as done to avoid infinite loop
    print(f"    [WARN] Failed to parse refiner output, marking as done")
    return {
        "system_diagram_iteration": iteration,
        "system_diagram_done": True,
        "phase": "done",
        "status": "done",
    }


def system_diagram_renderer_node(state: GraphState):
    """
    Render the final system diagram to SVG.
    """
    d2_code = state.get("system_diagram_d2", "")
    if not d2_code:
        return {"phase": "done"}

    proj_dir = OUTPUT_BASE / state["project_id"]
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "diagrams").mkdir(exist_ok=True)

    d2_path = proj_dir / "diagrams" / "system-overview.d2"
    # Clean icon URLs that cause D2 issues
    clean_d2 = re.sub(r'icon:\s*"(https?://.*?)"', "", d2_code)
    d2_path.write_text(clean_d2)

    svg_path = d2_path.with_suffix(".svg")
    res = subprocess.run(
        ["d2", "--layout=elk", str(d2_path), str(svg_path)],
        capture_output=True,
        text=True,
    )

    if res.returncode == 0:
        print(f"  ✓ System Overview Diagram rendered: {svg_path}")
        # Add to accumulated_md
        link = f"\n\n## System Overview\n\n![System Overview](./diagrams/system-overview.svg)\n"
        return {
            "accumulated_md": state["accumulated_md"] + link,
            "phase": "done",
            "status": "done",
        }
    else:
        print(f"  [WARN] D2 render failed: {res.stderr[:200]}")
        return {"phase": "done", "status": "done"}


def explainer_node(state: GraphState):
    """
    Process [[EXPLAIN:id|description]] markers left by the Educator.
    Generate concise explanations and replace markers with Foundation blocks.
    Uses a cheap/fast model since explanations are short.
    """
    full_md = state["accumulated_md"]
    segments = full_md.split("<!-- END_MS -->")

    if len(segments) < 2:
        return {}

    last_ms_content = segments[-2]
    markers = re.findall(r"\[\[EXPLAIN:([\w-]+)\|(.*?)\]\]", last_ms_content)

    if not markers:
        return {}  # Nothing to explain — pass through

    # Skip concepts already explained in earlier milestones
    already_explained = set(state.get("explained_concepts", []))
    new_markers = [(mid, desc) for mid, desc in markers if mid not in already_explained]

    if not new_markers:
        # Remove duplicate markers silently from this segment
        processed_segment = last_ms_content
        for mid, desc in markers:
            processed_segment = processed_segment.replace(
                f"[[EXPLAIN:{mid}|{desc}]]", ""
            )

        segments[-2] = processed_segment
        return {"accumulated_md": "<!-- END_MS -->".join(segments)}

    print(f"  [Agent: Explainer] Generating {len(new_markers)} explanations...")

    # Batch all explanations in one LLM call
    level = state["meta"].get("level", "intermediate")
    concept_list = "\n".join(
        f"{i + 1}. **{mid}**: {desc}" for i, (mid, desc) in enumerate(new_markers)
    )

    prompt = f"""You are a technical concept explainer. Reader level: '{level}'.

For each concept below, write an explanation covering:
1. What it IS (definition in plain language)
2. WHY the reader needs it right now (context for this project)
3. ONE key insight or mental model to remember

Use as much depth as the concept demands — simple concepts can be brief, complex ones deserve more. Use concrete examples where possible. This is a Foundation sidebar, not a chapter — stay focused but never sacrifice clarity for brevity.

Concepts:
{concept_list}

Output format — for EACH concept, output EXACTLY:
===CONCEPT:concept-id===
Your explanation here.
===END===

No other text outside these blocks."""

    res = safe_invoke(
        [HumanMessage(content=prompt)],
        node_label="Explainer",
        project_id=state["project_id"],
    )

    # Parse explanations
    explanations = {}
    for match in re.finditer(
        r"===CONCEPT:([\w-]+)===\s*(.*?)\s*===END===", str(res.content), re.DOTALL
    ):
        explanations[match.group(1)] = match.group(2).strip()

    # Replace markers in the current segment
    processed_segment = last_ms_content
    for mid, desc in markers:
        marker_text = f"[[EXPLAIN:{mid}|{desc}]]"
        if mid in explanations:
            title = re.split(r"[—\(\,]", desc)[0].strip()
            block = f"\n> **🔑 Foundation: {title}**\n> \n> {explanations[mid]}\n"
            processed_segment = processed_segment.replace(marker_text, block)
        elif mid in already_explained:
            processed_segment = processed_segment.replace(marker_text, "")
        else:
            print(f"    [WARN] No explanation generated for: {mid}")
            processed_segment = processed_segment.replace(marker_text, "")

    new_explained = list(
        already_explained | {mid for mid, _ in new_markers if mid in explanations}
    )

    segments[-2] = processed_segment
    return {
        "accumulated_md": "<!-- END_MS -->".join(segments),
        "explained_concepts": new_explained,
    }


def project_charter_node(state: GraphState):
    """Generate the Project Charter — placed at the very beginning of the document."""
    print(f"  [Agent: Project Charter] Writing charter...")
    full_yaml_meta = yaml.dump(
        state["meta"], allow_unicode=True, default_flow_style=False
    )
    prompt = f"""
{INSTR_PROJECT_CHARTER}

--- PROJECT SPEC (YAML) ---
{full_yaml_meta}

--- PEDAGOGICAL ATLAS (educator content — milestones, concepts taught) ---
{state["accumulated_md"]}

--- TDD CONTENT (modules, implementation phases, hours estimates) ---
{state["tdd_accumulated_md"]}

TASK: Output ONLY the Project Charter markdown. Start directly with `# 🎯 Project Charter`.
"""
    res = safe_invoke(
        [HumanMessage(content=prompt)],
        node_label="Project Charter",
        project_id=state["project_id"],
    )
    charter_md = strip_code_fence(str(res.content))
    print(f"  [Agent: Project Charter] Generated {len(charter_md)} chars")
    return {"project_charter_md": charter_md}


def project_structure_node(state: GraphState):
    """Synthesize unified project directory structure from all TDD modules."""
    print(f"  [Agent: Project Structure] Synthesizing...")

    project_name = state["meta"].get("name", state["project_id"])
    prompt = f"""
{INSTR_PROJECT_STRUCTURE}

--- PROJECT NAME ---
{project_name}

--- COMPLETE TDD CONTENT (contains all modules with file structures) ---
{state["tdd_accumulated_md"]}

--- PEDAGOGICAL CONTEXT (for milestone mapping) ---
{state["accumulated_md"]}

TASK: Output ONLY the project structure markdown (no preamble, no conversation).
"""

    res = safe_invoke(
        [HumanMessage(content=prompt)],
        node_label="Project Structure",
        project_id=state["project_id"],
    )
    structure_md = str(res.content).strip()
    print(f"  [Agent: Project Structure] Generated {len(structure_md)} chars")
    return {"project_structure_md": structure_md}


def bibliographer_node(state: GraphState):
    print(f"  [Agent: Bibliographer] Curating...")
    prompt = f"{INSTR_BIBLIOGRAPHER}\nCONTENT:\n{state['accumulated_md']}\n{state['tdd_accumulated_md']}\n{state.get('project_structure_md', '')}"
    res = safe_invoke(
        [HumanMessage(content=prompt)],
        node_label="Bibliographer",
    )
    return {"external_reading": str(res.content).strip(), "status": "done"}


def spec_syncer_node(state: GraphState):
    """
    Save AI-generated criteria to a separate JSON file.
    Does NOT modify the original YAML spec - user-authored criteria are preserved.
    The synced_criteria.json file is for reference/debugging only.
    """
    project_id = state["project_id"]
    proj_dir = OUTPUT_BASE / project_id

    synced_data = state.get("running_criteria", [])
    if not synced_data:
        print("  [Agent: Syncer] No synchronized criteria found. Skipping.")
        return {}

    criteria_file = proj_dir / "synced_criteria.json"
    criteria_file.write_text(json.dumps(synced_data, indent=2, ensure_ascii=False))
    print(f"  ✓ Saved {len(synced_data)} criteria sets to synced_criteria.json")
    print(f"    (Original YAML spec left unchanged)")
    return {}


# --- GRAPH ---
workflow = StateGraph(GraphState)
workflow.add_node("architect", architect_node)

workflow.add_node("writer", writer_node)
workflow.add_node("explainer", explainer_node)
workflow.add_node("visualizer", visualizer_node)
workflow.add_node("compiler", compiler_node)
workflow.add_node("tdd_planner", tdd_planner_node)
workflow.add_node("tdd_writer", tdd_writer_node)
workflow.add_node("tdd_visualizer", tdd_visualizer_node)
workflow.add_node("system_diagram_writer", system_diagram_writer_node)
workflow.add_node("system_diagram_refiner", system_diagram_refiner_node)
workflow.add_node("system_diagram_renderer", system_diagram_renderer_node)
workflow.add_node("project_charter", project_charter_node)
workflow.add_node("project_structure", project_structure_node)
workflow.add_node("bibliographer", bibliographer_node)
workflow.add_node("spec_syncer", spec_syncer_node)

workflow.add_edge(START, "architect")
workflow.add_edge("architect", "writer")


def route_writer(state):
    has_more = state["current_ms_index"] < len(state["blueprint"].get("milestones", []))
    if not has_more:
        return "visualizer"
    # Return list of nodes to execute in parallel
    return "explainer"


workflow.add_conditional_edges(
    "writer",
    route_writer,
    {
        "explainer": "explainer",
        "visualizer": "visualizer",
    },
)
workflow.add_edge("explainer", "writer")


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
    if state.get("tdd_diagrams_to_generate"):
        return "compiler"
    if ENABLE_SYSTEM_DIAGRAM:
        return "system_diagram_writer"
    return ["project_charter", "project_structure"]


workflow.add_conditional_edges(
    "tdd_visualizer",
    route_tdd_visualizer,
    {
        "compiler": "compiler",
        "system_diagram_writer": "system_diagram_writer",
        "project_charter": "project_charter",
        "project_structure": "project_structure",
    },
)


def route_system_diagram_refiner(state):
    """Loop until done or max iterations reached."""
    if state.get("system_diagram_done"):
        return "system_diagram_renderer"
    if state.get("system_diagram_iteration", 0) >= MAX_SYSTEM_DIAGRAM_ITERATIONS:
        return "system_diagram_renderer"
    return "system_diagram_refiner"


workflow.add_edge("system_diagram_writer", "system_diagram_refiner")
workflow.add_conditional_edges(
    "system_diagram_refiner",
    route_system_diagram_refiner,
    {
        "system_diagram_refiner": "system_diagram_refiner",
        "system_diagram_renderer": "system_diagram_renderer",
    },
)
workflow.add_edge("system_diagram_renderer", "project_charter")
workflow.add_edge("system_diagram_renderer", "project_structure")
workflow.add_edge("project_charter", "bibliographer")
workflow.add_edge("project_structure", "bibliographer")


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
workflow.add_edge("bibliographer", "spec_syncer")
workflow.add_edge("spec_syncer", END)
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
        "primary_language": "C",  # Will be set by architect
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
        "running_criteria": [],
        "explained_concepts": [],
        "system_diagram_d2": None,
        "system_diagram_iteration": 0,
        "system_diagram_done": False,
        "project_structure_md": "",
        "project_charter_md": "",
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
        final_state.get("project_charter_md", "")
        + "\n\n---\n\n"
        + final_state.get("external_reading", "")
        + "\n\n---\n\n"
        + final_state.get("accumulated_md", "")
        + "\n\n"
        + final_state.get("tdd_accumulated_md", "")
        + "\n\n"
        + final_state.get("project_structure_md", "")
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
    parser.add_argument("--anthropic", action="store_true")
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Use Gemini (Local Proxy) for full pipeline",
    )
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
    parser.add_argument(
        "--architect-claude",
        action="store_true",
        help="Architect=Claude CLI, All other nodes=Gemini Proxy",
    )
    parser.add_argument(
        "--claude-model", default="sonnet", help="Claude CLI model (default: sonnet)"
    )
    parser.add_argument(
        "--1m",
        dest="use_1m",
        action="store_true",
        help="Use 1M context window (appends [1m] to model)",
    )
    parser.add_argument("--anthropic-model", default=None)
    parser.add_argument("--debug", action="store_true", help="Log raw LLM responses")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: data/architecture-docs)",
    )
    parser.add_argument(
        "--system-diagram",
        action="store_true",
        help="Enable system overview diagram generation (uses Gemini local proxy)",
    )
    args = parser.parse_args()

    if args.claude_cli:
        model = args.claude_model + ("[1m]" if args.use_1m else "")
        os.environ["USE_CLAUDE_CLI"] = "true"
        os.environ["CLAUDE_MODEL"] = model
        print(f">>> Claude CLI: ENABLED (model={model})", flush=True)
    if args.anthropic:
        os.environ["USE_ANTHROPIC"] = "true"
    if args.gemini:
        # Implicitly falls back to local-proxy in init_llm_provider
        pass
    if args.mixed_claude_gemini:
        os.environ["USE_MIXED_CLAUDE_GEMINI"] = "true"
    if args.mixed_heavy_claude:
        os.environ["USE_MIXED_HEAVY_CLAUDE"] = "true"
    if args.architect_claude:
        os.environ["USE_ARCHITECT_CLAUDE"] = "true"

    if args.anthropic_model:
        os.environ["ANTHROPIC_MODEL"] = args.anthropic_model

    if args.debug:
        os.environ["DEBUG_MODE"] = "true"
    if args.system_diagram:
        os.environ["ENABLE_SYSTEM_DIAGRAM"] = "true"
        ENABLE_SYSTEM_DIAGRAM = True
        print(">>> System Overview Diagram: ENABLED", flush=True)
    if args.use_1m:
        os.environ["USE_1M"] = "true"
        USE_1M = True
        print(">>> 1M Context Window: ENABLED", flush=True)
    if args.output:
        OUTPUT_BASE = Path(args.output).resolve()
    else:
        OUTPUT_BASE = DEFAULT_OUTPUT
    print(f">>> Output directory: {OUTPUT_BASE}", flush=True)
    init_llm_provider()
    for p in args.projects:
        generate_project(p)
