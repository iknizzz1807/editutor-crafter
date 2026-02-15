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
print(">>> Instructions loaded", flush=True)

# --- LLM SETUP ---
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "sonnet")  # haiku, sonnet, opus

# Global variables (will be initialized by init_llm_provider())
LLM_PROVIDER = None
LLM = None
USE_ANTHROPIC = False


def init_llm_provider():
    """Initialize LLM provider based on environment variables or command line flags."""
    global LLM_PROVIDER, LLM, CLAUDE_MODEL, USE_ANTHROPIC
    print(">>> Initializing LLM provider...", flush=True)

    USE_ANTHROPIC = os.getenv("USE_ANTHROPIC", "false").lower() == "true"
    USE_CLAUDE_CLI = os.getenv("USE_CLAUDE_CLI", "false").lower() == "true"
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "sonnet")  # Re-read from env

    if USE_CLAUDE_CLI:
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
            cmd = ["claude", "-p", "--model", CLAUDE_MODEL]

            # Add system prompt if present via flag
            if system_prompt:
                cmd.extend(["--system-prompt", system_prompt])

            # Use stdin for user prompt to avoid argument length limit
            result = subprocess.run(
                cmd,
                input=user_prompt,  # Pass via stdin instead of argument
                capture_output=True,
                text=True,
                timeout=1200,  # 10 minute timeout
            )

            if result.returncode == 0:
                # Create a mock response object compatible with LangChain
                class MockResponse:
                    def __init__(self, content):
                        self.content = content

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
                    raise Exception(f"Claude CLI failed: {result.stderr}")

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


def safe_invoke(messages, max_retries=5, invoke_kwargs=None):
    """Invoke LLM with exponential backoff to handle quota and transient server errors."""
    if LLM_PROVIDER == "claude-cli":
        return invoke_claude_cli(messages, max_retries)

    kwargs = invoke_kwargs or {}
    print(
        f"    [safe_invoke] Starting LLM call with {len(messages)} messages...",
        flush=True,
    )
    for attempt in range(max_retries):
        try:
            result = LLM.invoke(messages, **kwargs)
            if result is None or (
                hasattr(result, "content") and result.content is None
            ):
                raise Exception("LLM returned None response")
            print(f"    [safe_invoke] LLM call succeeded", flush=True)
            return result
        except Exception as e:
            print(f"    [safe_invoke] Error: {e}", flush=True)
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
    # Use JSON mode if supported by local proxy (Gemini)
    invoke_args = {}
    if LLM_PROVIDER == "local-proxy":
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
    )
    print(
        f"  [Agent: Architect] LLM response received ({len(res.content)} chars)",
        flush=True,
    )
    blueprint = extract_json(res.content)
    if not blueprint:
        print("    ! Failed with JSON mode, retrying with raw prompt...")
        res = safe_invoke(
            [
                SystemMessage(
                    content="You are a Master Architect. Output ONLY raw JSON code block."
                ),
                HumanMessage(content=prompt),
            ]
        )
        blueprint = extract_json(res.content)

    if not blueprint:
        raise ValueError("Architect failed to return valid JSON after retries")

    return {
        "blueprint": blueprint,
        "diagrams_to_generate": blueprint.get("diagrams", []),
        "status": "writing",
        "accumulated_md": f"# {blueprint.get('title', state['project_id'])}\n\n{blueprint.get('overview', '')}\n\n",
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
    PREVIOUS: {state["accumulated_md"][-15000:]}
    
    IMPORTANT: You MUST use these specific Diagram IDs when inserting diagrams using the {{{{DIAGRAM:id}}}} syntax. 
    Only use diagrams that are relevant to this milestone.
    
    AVAILABLE DIAGRAMS:
    {diag_list}
    """
    res = safe_invoke([HumanMessage(content=prompt)])
    return {
        "accumulated_md": state["accumulated_md"] + f"\n\n{str(res.content).strip()}\n",
        "current_ms_index": idx + 1,
    }


def visualizer_node(state: GraphState):
    if not state.get("diagrams_to_generate"):
        return {"status": "done", "current_diagram_code": None}

    diag = state["diagrams_to_generate"][0]
    attempt = state["diagram_attempt"] + 1
    print(
        f"  [Agent: Artist] Drawing: {diag.get('title', 'Diagram')} (Attempt {attempt})..."
    )

    prompt = f"""
    {INSTR_ARTIST}
    REFERENCE (FULL D2 DOCUMENTATION): 
    {D2_REFERENCE}
    
    CONTEXT (TECHNICAL CONTENT):
    {state["accumulated_md"][-15000:]}
    
    TASK: Generate D2 code for the diagram: '{diag.get("title", "Untitled")}'
    DIAGRAM DESCRIPTION: {diag.get("description", "")}
    TARGET ANCHOR (for links): {diag.get("anchor_target", "")}
    """
    if state.get("last_error"):
        prompt += f"\n\n!!! FIX PREVIOUS COMPILER ERROR: {state['last_error']}\nAnalyze the error and update the D2 code to resolve it."

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
            return {
                "diagrams_to_generate": state["diagrams_to_generate"][1:],
                "diagram_attempt": 0,
                "last_error": None,
                "current_diagram_code": None,
                "current_diagram_meta": None,
            }
        return {"last_error": res.stderr}


# --- GRAPH ---
workflow = StateGraph(GraphState)
workflow.add_node("architect", architect_node)
workflow.add_node("writer", writer_node)
workflow.add_node("visualizer", visualizer_node)
workflow.add_node("compiler", compiler_node)

workflow.add_edge(START, "architect")
workflow.add_edge("architect", "writer")


def route_writer(state):
    return (
        "writer"
        if state["current_ms_index"] < len(state["blueprint"].get("milestones", []))
        else "visualizer"
    )


workflow.add_conditional_edges(
    "writer", route_writer, {"writer": "writer", "visualizer": "visualizer"}
)


def route_visualizer(state):
    if not state.get("diagrams_to_generate"):
        return END
    return "compiler"


workflow.add_conditional_edges(
    "visualizer", route_visualizer, {"compiler": "compiler", END: END}
)


def route_compiler(state):
    return (
        "retry" if state.get("last_error") and state["diagram_attempt"] > 0 else "next"
    )


workflow.add_conditional_edges(
    "compiler", route_compiler, {"retry": "visualizer", "next": "visualizer"}
)

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
        }
    )

    with open(OUTPUT_BASE / project_id / "index.md", "w") as f:
        f.write(final_state.get("accumulated_md", ""))
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
    parser.add_argument("--claude-model", default=None)
    parser.add_argument("--anthropic", action="store_true")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: data/architecture-docs)",
    )
    args = parser.parse_args()
    if args.claude_cli:
        os.environ["USE_CLAUDE_CLI"] = "true"
    if args.claude_model:
        os.environ["CLAUDE_MODEL"] = args.claude_model
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
