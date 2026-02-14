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
OUTPUT_BASE = DATA_DIR / "architecture-docs"
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


D2_REFERENCE = load_d2_docs()
INSTR_ARCHITECT = load_instruction("architect")
INSTR_EDUCATOR = load_instruction("educator")
INSTR_ARTIST = load_instruction("artist")

# --- LLM SETUP ---
USE_ANTHROPIC = os.getenv("USE_ANTHROPIC", "false").lower() == "true"
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

if USE_ANTHROPIC and ChatAnthropic:
    LLM = ChatAnthropic(
        model=ANTHROPIC_MODEL,
        temperature=0.1,
    )
    print(f">>> Provider: ANTHROPIC ({ANTHROPIC_MODEL})")
else:
    LLM = ChatOpenAI(
        base_url="http://127.0.0.1:7999/v1",
        api_key="mythong2005",
        model="gemini_cli/gemini-3-flash-preview",
        temperature=0.1,
    )
    if USE_ANTHROPIC and not ChatAnthropic:
        print(
            ">>> Warning: langchain-anthropic not installed. Falling back to local proxy."
        )
    print(">>> Provider: LOCAL PROXY (Gemini 3 Flash)")


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


def safe_invoke(messages, max_retries=5):
    """Invoke LLM with exponential backoff to handle quota and transient server errors."""
    for attempt in range(max_retries):
        try:
            return LLM.invoke(messages)
        except Exception as e:
            err_str = str(e).lower()
            # Catch Rate Limits (429), Quota, Timeouts, and Internal Server Errors (500, 502, 503, 504)
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
                wait_time = (attempt + 1) * 20  # Wait 20s, 40s, 60s...
                print(
                    f"    ! LLM Provider error (transient). Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
            else:
                raise e
    raise Exception(
        "Max retries exceeded for LLM invocation due to persistent provider errors."
    )


def architect_node(state: GraphState):
    print(f"  [Agent: Architect] Blueprinting {state['project_id']}...")
    meta = state["meta"]
    prompt = f"""
    {INSTR_ARCHITECT}
    PROJECT: {meta.get("name")}\nDESC: {meta.get("description")}
    TASK: Output ONLY raw JSON for the Blueprint. 
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
    res = safe_invoke(
        [
            SystemMessage(
                content="You are a Master Architect. Output ONLY valid JSON."
            ),
            HumanMessage(content=prompt),
        ]
    )
    blueprint = extract_json(res.content)
    if not blueprint:
        raise ValueError("Architect failed to return valid JSON")

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

    # Create a list of available diagram IDs for the Educator to use
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
    content = str(res.content)
    code = re.sub(r"```d2\n?|```", "", content).strip()
    return {
        "current_diagram_code": code,
        "current_diagram_meta": diag,
        "diagram_attempt": attempt,
    }


def compiler_node(state: GraphState):
    diag = state["current_diagram_meta"]
    code = state["current_diagram_code"]
    if not diag or not code:
        return {"last_error": None}

    proj_dir = OUTPUT_BASE / state["project_id"]
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "diagrams").mkdir(exist_ok=True)
    d2_path = proj_dir / "diagrams" / f"{diag.get('id', 'diag')}.d2"

    # Strip remote icons which cause 403 Forbidden in some environments
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
            print(f"    ! Max attempts reached for {diag.get('id')}. Skipping.")
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
    args = parser.parse_args()
    for p in args.projects:
        generate_project(p)
