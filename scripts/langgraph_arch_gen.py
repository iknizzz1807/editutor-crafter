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
    path = D2_EXAMPLES_DIR / "d2_docs.md"
    return path.read_text() if path.exists() else "Standard D2 documentation."


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
    Plan 10-15 diagrams.
    """
    res = LLM.invoke(
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
    if idx >= len(milestones):
        return {"status": "visualizing"}

    ms = milestones[idx]
    ms_title = ms.get("title") or f"Milestone {idx + 1}"
    print(f"  [Agent: Educator] Writing Atlas Node: {ms_title}...")

    prompt = f"""
    {INSTR_EDUCATOR}
    TASK: Write content for: {ms_title}.
    SUMMARY: {ms.get("summary", "")}
    ANCHOR_ID: {ms.get("id", f"ms-{idx}")}
    PREVIOUS: {state["accumulated_md"][-10000:]}
    """
    res = LLM.invoke([HumanMessage(content=prompt)])
    return {
        "accumulated_md": state["accumulated_md"] + f"\n\n{str(res.content).strip()}\n",
        "current_ms_index": idx + 1,
    }


def visualizer_node(state: GraphState):
    if not state["diagrams_to_generate"]:
        return {"status": "done"}

    diag = state["diagrams_to_generate"][0]
    attempt = state["diagram_attempt"] + 1
    print(
        f"  [Agent: Artist] Drawing: {diag.get('title', 'Diagram')} (Attempt {attempt})..."
    )

    prompt = f"""
    {INSTR_ARTIST}
    REFERENCE: {D2_REFERENCE[:50000]}
    CONTEXT: {state["accumulated_md"][-10000:]}
    TASK: Generate D2 for '{diag.get("title", "Untitled")}'
    DESC: {diag.get("description", "")}
    ANCHOR: {diag.get("anchor_target", "")}
    """
    if state.get("last_error"):
        prompt += f"\n\n!!! FIX PREVIOUS ERROR: {state['last_error']}"

    res = LLM.invoke(
        [
            SystemMessage(
                content="You are a D2 Master. Output ONLY raw D2 code. No preamble, no explanation."
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
    diag = state["current_diagram_meta"]
    code = state["current_diagram_code"]
    if not diag or not code:
        return {}

    proj_dir = OUTPUT_BASE / state["project_id"]
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "diagrams").mkdir(exist_ok=True)
    d2_path = proj_dir / "diagrams" / f"{diag.get('id', 'diag')}.d2"

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
        }
    else:
        print(f"    ✗ Failed (Attempt {state['diagram_attempt']}), retrying...")
        if state["diagram_attempt"] >= 5:
            return {
                "diagrams_to_generate": state["diagrams_to_generate"][1:],
                "diagram_attempt": 0,
                "last_error": None,
                "current_diagram_code": None,
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


def route_compiler(state):
    return (
        "retry" if state.get("last_error") and state["diagram_attempt"] > 0 else "next"
    )


workflow.add_edge("visualizer", "compiler")
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
