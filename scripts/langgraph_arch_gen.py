#!/usr/bin/env python3
import os, json, re, subprocess, time, yaml, argparse, operator
from typing import Annotated, List, TypedDict, Dict, Any, Optional, Union
from pathlib import Path
from string import Template
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
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
    docs = []
    for file in ["d2_docs.md", "dagre.txt", "elk.txt", "tala.txt"]:
        path = D2_EXAMPLES_DIR / file
        if path.exists():
            docs.append(f"--- DOC: {file} ---\n{path.read_text()}")
    return "\n\n".join(docs)


D2_REFERENCE = load_d2_docs()
INSTR_ARCHITECT = load_instruction("architect")
INSTR_EDUCATOR = load_instruction("educator")
INSTR_ARTIST = load_instruction("artist")

LLM = ChatOpenAI(
    base_url="http://127.0.0.1:7999/v1",
    api_key="mythong2005",
    model="gemini_cli/gemini-3-flash-preview",
    temperature=0.1,
)


def replace_reducer(old, new):
    return new


# ===========================================================================
# V17.1 - THE SCHEMA-ENFORCED ATLAS (Robust JSON & Fallbacks)
# ===========================================================================


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


# --- HELPERS ---
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


# --- AGENT NODES ---


def architect_node(state: GraphState):
    """Phase 1: Planning with strict JSON schema."""
    print(f"  [Agent: Architect] Blueprinting {state['project_id']}...")
    meta = state["meta"]

    prompt = f"""
    {INSTR_ARCHITECT}
    PROJECT: {meta.get("name")}\nDESC: {meta.get("description")}
    
    TASK: Output ONLY raw JSON for the Blueprint. 
    EXAMPLE JSON FORMAT (STRICT):
    {{
      "title": "Build a Garbage Collector",
      "overview": "Comprehensive summary...",
      "technical_contract": {{ "structs": [], "interfaces": [] }},
      "milestones": [ {{ "id": "ms-1", "title": "Object Anatomy", "summary": "..." }} ],
      "diagrams": [ {{ "id": "diag-01", "title": "Memory Layout", "description": "...", "anchor_target": "ms-1" }} ]
    }}
    Plan 25+ diagrams. Ensure unique IDs like diag-01, diag-02.
    """
    res = LLM.invoke(
        [
            SystemMessage(
                content="You are a Master Architect. Output ONLY valid JSON based on the example."
            ),
            HumanMessage(content=prompt),
        ]
    )
    blueprint = extract_json(res.content)
    if not blueprint:
        raise ValueError("Architect failed to return valid JSON")

    # Validation & Fallbacks
    if "milestones" not in blueprint:
        blueprint["milestones"] = []
    if "diagrams" not in blueprint:
        blueprint["diagrams"] = []

    return {
        "blueprint": blueprint,
        "diagrams_to_generate": blueprint["diagrams"],
        "status": "writing",
        "accumulated_md": f"# {blueprint.get('title', state['project_id'])}\n\n{blueprint.get('overview', '')}\n\n",
    }


def writer_node(state: GraphState):
    """Phase 2: Writing Chapters."""
    idx = state["current_ms_index"]
    blueprint = state["blueprint"]
    milestones = blueprint.get("milestones", [])
    if idx >= len(milestones):
        return {"status": "visualizing"}

    ms = milestones[idx]
    ms_title = ms.get("title") or ms.get("id") or f"Milestone {idx + 1}"
    print(f"  [Agent: Educator] Writing Atlas Node: {ms_title}...")

    prompt = f"""
    {INSTR_EDUCATOR}
    TASK: Write Chapter content for: {ms_title}.
    SUMMARY: {ms.get("summary", "")}
    ANCHOR_ID: {ms.get("id", f"ms-{idx}")}
    CONTRACT: {json.dumps(blueprint.get("technical_contract", {}))}
    PREVIOUS: {state["accumulated_md"][-20000:]}
    """
    res = LLM.invoke([HumanMessage(content=prompt)])
    return {
        "accumulated_md": state["accumulated_md"] + f"\n\n{res.content.strip()}\n",
        "current_ms_index": idx + 1,
    }


def visualizer_node(state: GraphState):
    """Phase 3: Drawing."""
    if not state["diagrams_to_generate"]:
        return {"status": "done"}
    diag = state["diagrams_to_generate"][0]
    attempt = state["diagram_attempt"] + 1
    print(
        f"  [Agent: Artist] Drawing: {diag.get('title', 'Diagram')} (Attempt {attempt})..."
    )

    prompt = f"""
    {INSTR_ARTIST}
    TASK: Generate D2 for: '{diag.get("title", "Untitled")}'
    DESCRIPTION: {diag.get("description", "")}
    TARGET_ANCHOR: {diag.get("anchor_target", "")}
    REFERENCE: {D2_REFERENCE[:8000]}
    CONTEXT: {state["accumulated_md"][-15000:]}
    """
    if state.get("last_error"):
        prompt += f"\n\nFIX ERROR: {state['last_error']}"
    res = LLM.invoke(
        [SystemMessage(content="You are a D2 Master."), HumanMessage(content=prompt)]
    )
    return {
        "current_diagram_code": re.sub(r"```d2\n?|```", "", res.content).strip(),
        "current_diagram_meta": diag,
        "diagram_attempt": attempt,
    }


def compiler_node(state: GraphState):
    diag = state["current_diagram_meta"]
    if not diag:
        return {}
    code = state["current_diagram_code"] or ""
    project_id = state["project_id"]
    proj_dir = OUTPUT_BASE / project_id
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "diagrams").mkdir(exist_ok=True)
    d2_path = proj_dir / "diagrams" / f"{diag.get('id', 'diag')}.d2"

    for s in ["capsule", "plaintext", "record", "sticky_note"]:
        code = code.replace(f"shape: {s}", "shape: rectangle")
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
        tag = f"{{{{DIAGRAM:{diag.get('id')}}}}}"
        return {
            "accumulated_md": state["accumulated_md"].replace(tag, img_link),
            "diagrams_to_generate": state["diagrams_to_generate"][1:],
            "diagram_attempt": 0,
            "last_error": None,
        }
    else:
        print(f"    ✗ Failed, retrying...")
        return {"diagram_attempt": state["diagram_attempt"], "last_error": res.stderr}


# --- GRAPH ---
workflow = StateGraph(GraphState)
workflow.add_node("architect", architect_node)
workflow.add_node("writer", writer_node)
workflow.add_node("visualizer", visualizer_node)
workflow.add_node("compiler", compiler_node)
workflow.add_node("check", lambda x: x)

workflow.add_edge(START, "architect")
workflow.add_edge("architect", "writer")


def route_writer(state):
    milestones = state["blueprint"].get("milestones", [])
    return "writer" if state["current_ms_index"] < len(milestones) else "visualizer"


workflow.add_conditional_edges(
    "writer", route_writer, {"writer": "writer", "visualizer": "visualizer"}
)


def route_compiler(state):
    if state.get("last_error") and state["diagram_attempt"] < 3:
        return "retry"
    return "next"


workflow.add_edge("visualizer", "compiler")
workflow.add_conditional_edges(
    "compiler", route_compiler, {"retry": "visualizer", "next": "check"}
)


def route_check(state):
    return "visualizer" if state.get("diagrams_to_generate") else END


workflow.add_edge("check", "visualizer")
workflow.add_conditional_edges(
    "check", route_check, {"visualizer": "visualizer", END: END}
)

app = workflow.compile()


def generate_project(project_id):
    print(f"\n>>> V17.1 SCHEMA-ENFORCED ATLAS STARTING: {project_id}")
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
    print(f"  ✓ MASTERPIECE V17.1 COMPLETE: {project_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", nargs="+", required=True)
    args = parser.parse_args()
    for p in args.projects:
        generate_project(p)
