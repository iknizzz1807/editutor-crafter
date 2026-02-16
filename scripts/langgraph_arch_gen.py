#!/usr/bin/env python3
import os, json, re, subprocess, time, yaml, argparse, operator
from typing import Annotated, List, TypedDict, Dict, Any, Optional, Union
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
DEFAULT_OUTPUT = DATA_DIR / "architecture-docs"
D2_EXAMPLES_DIR = SCRIPT_DIR / ".." / "d2_examples"
INSTRUCTIONS_DIR = SCRIPT_DIR / "instructions"


def load_instruction(name):
    path = INSTRUCTIONS_DIR / f"{name}.md"
    return path.read_text() if path.exists() else f"You are a master {name}."


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
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
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
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "sonnet")  # Re-read from env
    KILO_MODEL = os.getenv("KILO_MODEL", KILO_MODEL)  # Re-read from env or keep default
    MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", MISTRAL_MODEL)

    if USE_MIXED_CLAUDE_GEMINI or USE_MIXED_HEAVY_CLAUDE:
        LLM_PROVIDER = (
            "mixed-claude-gemini" if USE_MIXED_CLAUDE_GEMINI else "mixed-heavy-claude"
        )
        mode_desc = (
            "Architect=Claude, Others=Gemini"
            if USE_MIXED_CLAUDE_GEMINI
            else "Architect+Educator=Claude, Artist=Gemini"
        )
        print(
            f">>> Provider: MIXED ({mode_desc})",
            flush=True,
        )
        # We need to initialize the local proxy for the non-claude parts
        LLM = ChatOpenAI(
            base_url="http://127.0.0.1:7999/v1",
            api_key="mythong2005",
            model="gemini_cli/gemini-3-pro-preview",
            temperature=1,
            max_tokens=64000,
        )
    elif USE_MISTRAL:
        LLM_PROVIDER = "mistral"
        print(f">>> Provider: MISTRAL ({MISTRAL_MODEL})", flush=True)
        LLM = ChatOpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=MISTRAL_API_KEY,
            model=MISTRAL_MODEL,
            temperature=1,
            max_tokens=64000,
        )
    elif USE_KILO_CLI:
        LLM_PROVIDER = "kilo-cli"
        print(f">>> Provider: KILO CLI (Model: {KILO_MODEL})", flush=True)
    elif USE_CLAUDE_CLI:
        LLM_PROVIDER = "claude-cli"
        print(f">>> Provider: CLAUDE CODE CLI (Model: {CLAUDE_MODEL})", flush=True)
    elif USE_ANTHROPIC and ChatAnthropic:
        print(">>> Setting up Anthropic...", flush=True)
        LLM = ChatAnthropic(
            model=ANTHROPIC_MODEL,
            temperature=1,
            max_tokens=64000,  # 64K tokens output limit
        )
        LLM_PROVIDER = "anthropic"
        print(f">>> Provider: ANTHROPIC ({ANTHROPIC_MODEL})", flush=True)
    else:
        print(">>> Setting up local proxy (Gemini)...", flush=True)
        LLM = ChatOpenAI(
            base_url="http://127.0.0.1:7999/v1",
            api_key="mythong2005",
            model="gemini_cli/gemini-3-pro-preview",
            temperature=1,
            max_tokens=64000,  # 64K tokens output limit
        )
        LLM_PROVIDER = "local-proxy"
        if USE_ANTHROPIC and not ChatAnthropic:
            print(
                ">>> Warning: langchain-anthropic not installed. Falling back to local proxy."
            )
        print(">>> Provider: LOCAL PROXY (Gemini 3 Pro)", flush=True)


def replace_reducer(old, new):
    return new


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
    # Knowledge Tracking
    knowledge_map: Annotated[List[str], replace_reducer]
    advanced_contexts: Annotated[List[str], replace_reducer]
    # TDD Fields
    tdd_blueprint: Annotated[Dict[str, Any], replace_reducer]
    tdd_accumulated_md: Annotated[str, replace_reducer]
    tdd_current_mod_index: Annotated[int, replace_reducer]
    tdd_diagrams_to_generate: Annotated[List[Dict[str, Any]], replace_reducer]
    # Bibliographer
    external_reading: Annotated[str, replace_reducer]


def extract_json(text):
    if not text:
        return None
    try:
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return None


def strip_ansi_codes(text):
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def invoke_kilo_cli(messages, max_retries=3):
    """Invoke Kilo CLI with exponential backoff."""
    system_prompt = ""
    user_prompt = ""

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_prompt = msg.content
        elif isinstance(msg, HumanMessage):
            user_prompt = msg.content

    # Kilo doesn't have a distinct system prompt flag in 'run', so we combine them.
    full_prompt = (
        f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER REQUEST:\n{user_prompt}"
        if system_prompt
        else user_prompt
    )

    for attempt in range(max_retries):
        try:
            # cmd = ["kilo", "run", "--model", KILO_MODEL, full_prompt]
            # Use 'run' with the prompt.
            # NOTE: We are passing the prompt as an argument.
            # If the prompt is huge, we might hit shell arg limits, but subprocess.run
            # usually handles large args better than shell=True.
            cmd = ["kilo", "run"]
            if KILO_MODEL:
                cmd.extend(["--model", KILO_MODEL])
            cmd.append(full_prompt)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1200,
            )

            if result.returncode == 0:

                class MockResponse:
                    def __init__(self, content):
                        self.content = content

                return MockResponse(result.stdout)
            else:
                err_str = result.stderr.lower()
                print(f"    ! Kilo CLI Error: {result.stderr[:200]}...")
                # Retry logic for potential networking/API blips
                time.sleep((attempt + 1) * 5)
        except Exception as e:
            print(f"    ! Kilo CLI Exception: {e}")
            time.sleep((attempt + 1) * 5)

    raise Exception("Max retries exceeded for Kilo CLI invocation.")


def invoke_claude_cli(messages, max_retries=3):
    """Invoke Claude Code CLI with exponential backoff."""
    # Extract system message and human message from LangChain format
    system_prompt = ""
    user_prompt = ""

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_prompt = msg.content
        elif isinstance(msg, HumanMessage):
            user_prompt = msg.content

    for attempt in range(max_retries):
        try:
            # Build the claude command - use stdin for long prompts
            # Disable tools to prevent interactive prompts that hang the process
            cmd = [
                "claude",
                "-p",
                "--model",
                CLAUDE_MODEL,
                "--dangerously-skip-permissions",
                "--tools",
                "",
            ]

            # Add system prompt if present via flag
            if system_prompt:
                cmd.extend(["--system-prompt", system_prompt])

            # Use stdin for user prompt to avoid argument length limit
            result = subprocess.run(
                cmd,
                input=user_prompt,
                capture_output=True,
                text=True,
                timeout=1200,
            )

            if result.returncode == 0:
                # Create a mock response object compatible with LangChain
                class MockResponse:
                    def __init__(self, content):
                        self.content = strip_ansi_codes(content)

                return MockResponse(result.stdout)
            else:
                err_str = result.stderr.lower()
                # Check for transient errors
                transient_errors = [
                    "quota",
                    "limit",
                    "timeout",
                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                    "internal server error",
                    "api_error",
                ]
                if any(err in err_str for err in transient_errors):
                    wait_time = (attempt + 1) * 20
                    print(
                        f"    ! Claude CLI transient error. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    error_msg = result.stderr if result.stderr else result.stdout
                    raise Exception(
                        f"Claude CLI failed (RC={result.returncode}): {error_msg}"
                    )

        except subprocess.TimeoutExpired:
            wait_time = (attempt + 1) * 20
            print(f"    ! Claude CLI timeout. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            err_str = str(e).lower()
            transient_errors = [
                "quota",
                "limit",
                "timeout",
                "429",
                "500",
                "502",
                "503",
                "504",
                "internal server error",
                "api_error",
            ]
            if any(err in err_str for err in transient_errors):
                wait_time = (attempt + 1) * 20
                print(f"    ! Claude CLI transient error. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

    raise Exception("Max retries exceeded for Claude CLI invocation.")


def safe_invoke(messages, max_retries=5, invoke_kwargs=None, provider_override=None):
    """Invoke LLM with exponential backoff to handle quota and transient server errors."""
    actual_provider = provider_override or LLM_PROVIDER

    if actual_provider == "claude-cli":
        return invoke_claude_cli(messages, max_retries)
    if actual_provider == "kilo-cli":
        return invoke_kilo_cli(messages, max_retries)

    kwargs = invoke_kwargs or {}
    for attempt in range(max_retries):
        try:
            result = LLM.invoke(messages, **kwargs)
            if result is None or (
                hasattr(result, "content") and result.content is None
            ):
                raise Exception("LLM returned None response")
            return result
        except Exception as e:
            err_str = str(e).lower()
            transient_errors = [
                "quota",
                "limit",
                "timeout",
                "429",
                "500",
                "502",
                "503",
                "504",
                "internal server error",
                "api_error",
                "nonetype",
                "iterable",
                "none",
                "empty",
            ]
            if any(err in err_str for err in transient_errors):
                wait_time = (attempt + 1) * 20
                print(f"    ! LLM transient error. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Max retries exceeded for LLM invocation.")


def architect_node(state: GraphState):
    print(f"  [Agent: Architect] Blueprinting {state['project_id']}...", flush=True)
    meta = state["meta"]
    prompt = f"""
    {INSTR_ARCHITECT}
    PROJECT: {meta.get("name")}\nDESC: {meta.get("description")}
    TASK: Output ONLY raw JSON for the Blueprint. 
    
    CRITICAL: Do NOT include any conversational text or markdown blocks outside the JSON.
    
    EXAMPLE JSON FORMAT (STRICT):
    {{
      "title": "Build a Garbage Collector",
      "overview": "Summary...",
      "technical_contract": {{ "structs": [], "interfaces": [] }},
      "milestones": [ {{ "id": "ms-1", "title": "...", "summary": "..." }} ],
      "diagrams": [ {{ "id": "diag-01", "title": "...", "description": "...", "anchor_target": "ms-1" }} ]
    }}
    Plan 10-15 diagrams. Ensure unique IDs.
    """
    # Use JSON mode if supported by local proxy or Mistral
    invoke_args = {}
    if LLM_PROVIDER in [
        "local-proxy",
        "mistral",
        "mixed-claude-gemini",
        "mixed-heavy-claude",
    ]:
        invoke_args["response_format"] = {"type": "json_object"}

    print(f"  [Agent: Architect] Calling LLM (provider={LLM_PROVIDER})...", flush=True)
    res = safe_invoke(
        [
            SystemMessage(
                content="You are a Master Architect. Output ONLY valid JSON."
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
        res = safe_invoke(
            [
                SystemMessage(
                    content="You are a Master Architect. Output ONLY raw JSON code block."
                ),
                HumanMessage(content=prompt),
            ],
            provider_override=(
                "claude-cli"
                if LLM_PROVIDER and str(LLM_PROVIDER).startswith("mixed")
                else None
            ),
        )
        print(f"  [Agent: Architect] Retry response received: {len(res.content)} chars")
        blueprint = extract_json(res.content)

    if not blueprint:
        raise ValueError("Architect failed to return valid JSON after retries")

    # Incremental write
    proj_dir = OUTPUT_BASE / state["project_id"]
    proj_dir.mkdir(parents=True, exist_ok=True)
    header = f"# {blueprint.get('title', state['project_id'])}\n\n{blueprint.get('overview', '')}\n\n"

    return {
        "blueprint": blueprint,
        "diagrams_to_generate": blueprint.get("diagrams", []),
        "status": "writing",
        "accumulated_md": header,
    }


def knowledge_mapper_node(state: GraphState):
    """Scan the last generated content for key technical concepts to add to the knowledge map."""
    print(f"  [Agent: Knowledge Mapper] Updating map...", flush=True)

    last_content = state["accumulated_md"].split("\n\n")[-1]

    prompt = f"""
    Extract 3-5 key technical concepts/keywords from the following text that were explained in detail.
    Output ONLY a comma-separated list of terms.
    
    TEXT:
    {last_content}
    """

    res = safe_invoke([HumanMessage(content=prompt)], provider_override="local-proxy")
    new_terms = [t.strip() for t in str(res.content).split(",") if t.strip()]

    # Also extract Advanced Context tags
    adv_tags = re.findall(r"\[\[ADVANCED_CONTEXT:(.*?)\]\]", state["accumulated_md"])

    return {
        "knowledge_map": list(set(state.get("knowledge_map", []) + new_terms)),
        "advanced_contexts": list(set(state.get("advanced_contexts", []) + adv_tags)),
    }


def writer_node(state: GraphState):
    idx = state["current_ms_index"]
    blueprint = state["blueprint"]
    milestones = blueprint.get("milestones", [])
    diagrams = blueprint.get("diagrams", [])

    if idx >= len(milestones):
        return {"status": "visualizing"}

    ms = milestones[idx]
    ms_title = ms.get("title") or f"Milestone {idx + 1}"
    print(f"  [Agent: Educator] Writing Atlas Node: {ms_title}...")

    # Determine Provider for this node
    is_claude = LLM_PROVIDER == "mixed-heavy-claude" or LLM_PROVIDER == "claude-cli"

    # CONTEXT OPTIMIZATION STRATEGY:
    # If Claude: Small Input (Summary + Last MS) -> Big Output
    # If Gemini: Big Input (Full History) -> Specialized Output
    if is_claude and state["accumulated_md"]:
        # Get only the last milestone content to maintain style without bloating input
        history_parts = state["accumulated_md"].split("\n\n")
        last_ms_context = "\n\n".join(history_parts[-2:])  # Take last 2 blocks
        context_to_send = f"(...previous omitted for brevity...)\n\n{last_ms_context}"
    else:
        context_to_send = state["accumulated_md"]

    diag_list = "\n".join(
        [
            f"- ID: {d['id']} | Title: {d['title']} | Desc: {d['description']}"
            for d in diagrams
        ]
    )

    prompt = f"""
    {INSTR_EDUCATOR}
    TASK: Write content for: {ms_title}.
    SUMMARY: {ms.get("summary", "")}
    ANCHOR_ID: {ms.get("id", f"ms-{idx}")}
    KNOWLEDGE MAP (Compressed Memory): {state.get("knowledge_map", [])}
    PREVIOUS CONTEXT: {context_to_send}
    
    IMPORTANT: 
    1. You MUST use these specific Diagram IDs when inserting diagrams using the {{{{DIAGRAM:id}}}} syntax:
    {diag_list}
    
    2. DYNAMIC DIAGRAMS: If you find a new complex technical point that absolutely needs a visualization but IS NOT in the list above, you can ORDER a new one using this syntax:
    [[DYNAMIC_DIAGRAM:new-id|Title|Detailed Description of what to draw]]
    """
    res = safe_invoke(
        [HumanMessage(content=prompt)],
        invoke_kwargs={"temperature": 0.7},
        provider_override="claude-cli" if is_claude else None,
    )
    content = str(res.content).strip()

    # --- DYNAMIC DISCOVERY LOGIC ---
    new_diagrams = state["diagrams_to_generate"]
    # Pattern: [[DYNAMIC_DIAGRAM:id|title|description]]
    dynamic_orders = re.findall(r"\[\[DYNAMIC_DIAGRAM:(.*?)\|(.*?)\|(.*?)\]\]", content)

    for d_id, d_title, d_desc in dynamic_orders:
        d_id = d_id.strip()
        # Only add if not already in the list
        if not any(d["id"] == d_id for d in new_diagrams):
            print(f"    + Dynamic Diagram Ordered: {d_id}")
            new_diagrams.append(
                {
                    "id": d_id,
                    "title": d_title.strip(),
                    "description": d_desc.strip(),
                    "anchor_target": ms.get("id", f"ms-{idx}"),
                }
            )

    # Replace the order syntax with a standard marker for the final MD
    content = re.sub(
        r"\[\[DYNAMIC_DIAGRAM:(.*?)\|.*?\|.*?\]\]", r"{{DIAGRAM:\1}}", content
    )

    full_content = f"\n\n{content}\n"
    print(f"    ✓ Content generated: {len(full_content)} chars")

    return {
        "accumulated_md": state["accumulated_md"] + full_content,
        "current_ms_index": idx + 1,
        "diagrams_to_generate": new_diagrams,
    }


def compiler_node(state: GraphState):
    global OUTPUT_BASE
    diag = state["current_diagram_meta"]
    code = state["current_diagram_code"]
    if not diag or not code:
        return {"last_error": None}

    proj_dir = OUTPUT_BASE / state["project_id"]
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "diagrams").mkdir(exist_ok=True)
    d2_path = proj_dir / "diagrams" / f"{diag.get('id', 'diag')}.d2"

    code = re.sub(r'icon:\s*"(https?://.*?)"', "", code)
    d2_path.write_text(code)
    res = subprocess.run(
        ["d2", "--layout=elk", str(d2_path), str(d2_path.with_suffix(".svg"))],
        capture_output=True,
        text=True,
    )

    if res.returncode == 0:
        print(f"    ✓ Success: {diag.get('id')}")
        img_link = (
            f"\n![{diag.get('title', 'Diagram')}](./diagrams/{diag.get('id')}.svg)\n"
        )

        # Decide which MD to update
        if state.get("status") == "tdd_visualizing":
            return {
                "tdd_accumulated_md": state["tdd_accumulated_md"].replace(
                    f"{{{{DIAGRAM:{diag.get('id')}}}}}", img_link
                ),
                "tdd_diagrams_to_generate": state["tdd_diagrams_to_generate"][1:],
                "diagram_attempt": 0,
                "last_error": None,
                "current_diagram_code": None,
                "current_diagram_meta": None,
            }
        else:
            return {
                "accumulated_md": state["accumulated_md"].replace(
                    f"{{{{DIAGRAM:{diag.get('id')}}}}}", img_link
                ),
                "diagrams_to_generate": state["diagrams_to_generate"][1:],
                "diagram_attempt": 0,
                "last_error": None,
                "current_diagram_code": None,
                "current_diagram_meta": None,
            }
    else:
        print(f"    ✗ Failed (Attempt {state['diagram_attempt']}), retrying...")
        if state["diagram_attempt"] >= 5:
            if state.get("status") == "tdd_visualizing":
                return {
                    "tdd_diagrams_to_generate": state["tdd_diagrams_to_generate"][1:],
                    "diagram_attempt": 0,
                    "last_error": None,
                    "current_diagram_code": None,
                    "current_diagram_meta": None,
                }
            else:
                return {
                    "diagrams_to_generate": state["diagrams_to_generate"][1:],
                    "diagram_attempt": 0,
                    "last_error": None,
                    "current_diagram_code": None,
                    "current_diagram_meta": None,
                }
        return {"last_error": res.stderr}


def bibliographer_node(state: GraphState):
    print(f"  [Agent: Bibliographer] Curating external resources...", flush=True)

    # Use BOTH accumulated Atlas and TDD text for a complete sweep
    full_text = state["accumulated_md"] + "\n" + state["tdd_accumulated_md"]
    adv_terms = ", ".join(state.get("advanced_contexts", []))

    prompt = f"""
    {INSTR_BIBLIOGRAPHER}
    ADVANCED TERMS TO COVER: {adv_terms}
    
    FULL PROJECT CONTENT (Atlas + TDD):
    {full_text}
    
    TASK: Provide a "Beyond the Atlas" reading list. 
    Focus specifically on the ADVANCED TERMS listed above plus any other foundational giants found in the text.
    """

    res = safe_invoke([HumanMessage(content=prompt)], provider_override="local-proxy")
    return {"external_reading": str(res.content).strip(), "status": "done"}


def tdd_planner_node(state: GraphState):
    print(
        f"  [Agent: TDD Orchestrator] Planning Technical Design Document...", flush=True
    )

    prompt = f"""
    {INSTR_TDD_PLANNER}
    PROJECT META: {state["meta"]}
    ATLAS BLUEPRINT: {state["blueprint"]}
    FULL ATLAS CONTENT: {state["accumulated_md"]}
    
    TASK: Review the COMPLETE pedagogical Atlas content and plan a professional TDD.
    Output ONLY raw JSON.
    """

    res = safe_invoke(
        [
            SystemMessage(
                content="You are a Technical Design Orchestrator. Output ONLY valid JSON."
            ),
            HumanMessage(content=prompt),
        ],
        provider_override="local-proxy",  # Force Gemini Proxy for this step
    )

    print(
        f"  [Agent: TDD Orchestrator] TDD Blueprint received: {len(res.content)} chars"
    )
    tdd_blueprint = extract_json(res.content)
    if not tdd_blueprint:
        raise ValueError("TDD Orchestrator failed to return valid JSON")

    header = f"\n\n# TECHNICAL DESIGN DOCUMENT (TDD)\n\n## Design Vision\n{tdd_blueprint.get('design_vision', '')}\n\n"

    # Collect all TDD diagrams into a flat list for the visualizer to handle
    tdd_diags = []
    for mod in tdd_blueprint.get("modules", []):
        for d in mod.get("diagrams", []):
            d["anchor_target"] = mod["id"]  # Link diagram to module anchor
            tdd_diags.append(d)

    return {
        "tdd_blueprint": tdd_blueprint,
        "tdd_accumulated_md": header,
        "tdd_current_mod_index": 0,
        "tdd_diagrams_to_generate": tdd_diags,
        "status": "tdd_writing",
    }


def tdd_writer_node(state: GraphState):
    idx = state["tdd_current_mod_index"]
    blueprint = state["tdd_blueprint"]
    modules = blueprint.get("modules", [])

    if idx >= len(modules):
        return {"status": "tdd_visualizing"}

    mod = modules[idx]
    print(f"  [Agent: TDD Writer] Writing Spec for Module: {mod.get('name')}...")

    prompt = f"""
    {INSTR_TDD_WRITER}
    PROJECT: {blueprint.get("project_title")}
    MODULE: {mod.get("name")}
    DESCRIPTION: {mod.get("description")}
    INITIAL SPECS: {mod.get("specs")}
    FULL ATLAS HISTORY (for consistency): {state["accumulated_md"]}
    PREVIOUS TDD SPECS: {state["tdd_accumulated_md"]}
    
    TASK: Write a rigorous Technical Design Specification for this module.
    Use markdown headers. Include Pseudo-code. Use diagram markers {{{{DIAGRAM:id}}}} for: 
    {", ".join([d["id"] for d in mod.get("diagrams", [])])}
    
    DYNAMIC DIAGRAMS: If you need a NEW complex technical diagram (Class/Sequence/Flow) for this module, ORDER it:
    [[DYNAMIC_DIAGRAM:tdd-diag-newid|Title|Detailed technical description]]
    """

    res = safe_invoke(
        [HumanMessage(content=prompt)],
        invoke_kwargs={"temperature": 1},
        provider_override="local-proxy",
    )
    content = str(res.content).strip()

    # --- DYNAMIC DISCOVERY LOGIC ---
    new_tdd_diagrams = state["tdd_diagrams_to_generate"]
    dynamic_orders = re.findall(r"\[\[DYNAMIC_DIAGRAM:(.*?)\|(.*?)\|(.*?)\]\]", content)

    for d_id, d_title, d_desc in dynamic_orders:
        d_id = d_id.strip()
        if not any(d["id"] == d_id for d in new_tdd_diagrams):
            print(f"    + Dynamic TDD Diagram Ordered: {d_id}")
            new_tdd_diagrams.append(
                {
                    "id": d_id,
                    "title": d_title.strip(),
                    "description": d_desc.strip(),
                    "anchor_target": mod["id"],
                }
            )

    content = re.sub(
        r"\[\[DYNAMIC_DIAGRAM:(.*?)\|.*?\|.*?\]\]", r"{{DIAGRAM:\1}}", content
    )
    full_content = f"\n\n{content}\n"
    print(f"    ✓ TDD Module generated: {len(full_content)} chars")

    return {
        "tdd_accumulated_md": state["tdd_accumulated_md"] + full_content,
        "tdd_current_mod_index": idx + 1,
        "tdd_diagrams_to_generate": new_tdd_diagrams,
    }


def visualizer_node(state: GraphState):
    if not state.get("diagrams_to_generate"):
        return {"status": "tdd_planning"}

    diag = state["diagrams_to_generate"][0]
    attempt = state["diagram_attempt"] + 1
    print(
        f"  [Agent: Artist] Drawing: {diag.get('title', 'Diagram')} (Attempt {attempt})..."
    )

    prompt = f"""
    {INSTR_ARTIST}
    REFERENCE (FULL D2 DOCUMENTATION): 
    {D2_REFERENCE}
    
    CONTEXT (FULL TECHNICAL HISTORY):
    {state["accumulated_md"]}
    
    TASK: Generate D2 code for the diagram: '{diag.get("title", "Untitled")}'
    DIAGRAM DESCRIPTION: {diag.get("description", "")}
    TARGET ANCHOR (for links): {diag.get("anchor_target", "")}
    """
    if state.get("last_error"):
        prompt += f"""
        
        !!! FIX PREVIOUS COMPILER ERROR !!!
        FAILED CODE:
        ```d2
        {state.get("current_diagram_code")}
        ```
        ERROR MESSAGE:
        {state["last_error"]}
        
        Analyze the FAILED CODE and the ERROR MESSAGE, then provide the corrected D2 code.
        """

    res = safe_invoke(
        [
            SystemMessage(
                content="You are a D2 Master Artist. Output ONLY raw D2 code. No preamble, no explanation."
            ),
            HumanMessage(content=prompt),
        ]
    )
    code = re.sub(r"```d2\n?|```", "", str(res.content)).strip()
    return {
        "current_diagram_code": code,
        "current_diagram_meta": diag,
        "diagram_attempt": attempt,
    }


def tdd_visualizer_node(state: GraphState):
    if not state.get("tdd_diagrams_to_generate"):
        return {"status": "done", "current_diagram_code": None}

    diag = state["tdd_diagrams_to_generate"][0]
    attempt = state["diagram_attempt"] + 1
    print(
        f"  [Agent: TDD Artist] Drawing: {diag.get('title', 'Diagram')} (Attempt {attempt})..."
    )

    prompt = f"""
    {INSTR_TDD_ARTIST}
    REFERENCE (FULL D2 DOCUMENTATION): 
    {D2_REFERENCE}
    
    CONTEXT (FULL TECHNICAL & TDD HISTORY):
    {state["accumulated_md"]}
    {state["tdd_accumulated_md"]}
    
    TASK: Generate detailed D2 code for the technical diagram: '{diag.get("title", "Untitled")}'
    DIAGRAM DESCRIPTION: {diag.get("description", "")}
    TARGET ANCHOR: {diag.get("anchor_target", "")}
    """
    if state.get("last_error"):
        prompt += f"""
        
        !!! FIX PREVIOUS COMPILER ERROR !!!
        FAILED CODE:
        ```d2
        {state.get("current_diagram_code")}
        ```
        ERROR MESSAGE:
        {state["last_error"]}
        
        Analyze the FAILED CODE and the ERROR MESSAGE, then provide the corrected D2 code.
        """

    res = safe_invoke(
        [
            SystemMessage(
                content="You are a Technical D2 Master Artist. Output ONLY raw D2 code."
            ),
            HumanMessage(content=prompt),
        ],
        provider_override="local-proxy",
    )
    code = re.sub(r"```d2\n?|```", "", str(res.content)).strip()
    return {
        "current_diagram_code": code,
        "current_diagram_meta": diag,
        "diagram_attempt": attempt,
    }


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
    if state["current_ms_index"] < len(state["blueprint"].get("milestones", [])):
        return "knowledge_mapper"
    return "visualizer"


workflow.add_conditional_edges(
    "writer",
    route_writer,
    {"knowledge_mapper": "knowledge_mapper", "visualizer": "visualizer"},
)

workflow.add_edge("knowledge_mapper", "writer")


def route_visualizer(state):
    if not state.get("diagrams_to_generate"):
        return "tdd_planner"
    return "compiler"


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
    if not state.get("tdd_diagrams_to_generate"):
        return "bibliographer"
    return "compiler"


workflow.add_conditional_edges(
    "tdd_visualizer",
    route_tdd_visualizer,
    {"compiler": "compiler", "bibliographer": "bibliographer"},
)

workflow.add_edge("bibliographer", END)


def route_compiler(state):
    # Determine phase based on status field instead of cleared metadata
    is_tdd = state.get("status") == "tdd_visualizing"

    if state.get("last_error") and state["diagram_attempt"] > 0:
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

app = workflow.compile()


def generate_project(project_id):
    global OUTPUT_BASE
    print(f"\n>>> V17.1 ATLAS STARTING: {project_id}")
    with open(YAML_PATH) as f:
        data = yaml.safe_load(f)
        meta = next(
            (
                p
                for d in data.get("domains", [])
                for l in ["beginner", "intermediate", "advanced", "expert"]
                for p in d.get("projects", {}).get(l, [])
                if p.get("id") == project_id
            ),
            None,
        )
    if not meta:
        return print(f"Error: {project_id} not found.")

    final_state = app.invoke(
        {
            "project_id": project_id,
            "meta": meta,
            "blueprint": {},
            "accumulated_md": "",
            "current_ms_index": 0,
            "diagrams_to_generate": [],
            "diagram_attempt": 0,
            "last_error": None,
            "status": "planning",
            "current_diagram_code": None,
            "current_diagram_meta": None,
            "tdd_blueprint": {},
            "tdd_accumulated_md": "",
            "tdd_current_mod_index": 0,
            "tdd_diagrams_to_generate": [],
            "external_reading": "",
            "knowledge_map": [],
            "advanced_contexts": [],
        }
    )

    # Save final index.md (Concatenate Atlas + TDD + Further Reading)
    proj_dir = OUTPUT_BASE / project_id
    proj_dir.mkdir(parents=True, exist_ok=True)
    full_content = (
        final_state.get("accumulated_md", "")
        + "\n\n"
        + final_state.get("tdd_accumulated_md", "")
        + "\n\n"
        + final_state.get("external_reading", "")
    )
    with open(proj_dir / "index.md", "w") as f:
        f.write(full_content)

    # Final HTML generation
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
    if args.output:
        OUTPUT_BASE = Path(args.output).resolve()
    else:
        OUTPUT_BASE = DEFAULT_OUTPUT
    print(f">>> Output directory: {OUTPUT_BASE}", flush=True)
    init_llm_provider()
    for p in args.projects:
        generate_project(p)
