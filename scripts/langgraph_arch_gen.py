#!/usr/bin/env python3
import os, json, re, subprocess, time, yaml, argparse
from typing import Annotated, List, TypedDict, Dict, Any, Optional
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / ".." / "data"
YAML_PATH = DATA_DIR / "projects.yaml"
OUTPUT_BASE = DATA_DIR / "architecture-docs"

LLM_MASTER = ChatOpenAI(
    base_url="http://127.0.0.1:7999/v1",
    api_key="mythong2005",
    model="gemini_cli/gemini-3-flash-preview",
    temperature=0.2,
)

D2_GUIDE = """
=== D2 RULES (STRICT) ===
1. SHAPES: [rectangle, square, cylinder, queue, package, step, person, diamond, cloud, class, sql_table].
2. LABELS: Use "double quotes".
3. NO ILLEGAL KEYS: No 'font-weight', 'fill-opacity', 'theme', 'mermaid'.
4. STYLES: Use classes: { id: { style: { fill: "#f3f4f6" } } } and node.class: id
"""


class GraphState(TypedDict):
    project_id: str
    meta: Dict[str, Any]
    blueprint: Dict[str, Any]
    accumulated_md: str
    current_step: str
    current_ms_index: int
    diagrams_to_generate: List[Dict[str, Any]]
    diagram_attempt: int
    current_diagram_meta: Optional[Dict[str, Any]]
    last_error: Optional[str]


# --- AGENT NODES ---


def architect_node(state: GraphState):
    """Phase 1: High-level Educational Planning."""
    print(
        f"  [Agent: Architect] Planning the educational journey for {state['project_id']}..."
    )
    meta = state["meta"]
    prompt = f"""
    PROJECT: {meta["name"]}\nDESC: {meta["description"]}\nMILESTONES: {json.dumps(meta.get("milestones", []))}
    TASK: Design a 'Visual-First' Masterclass Blueprint.
    Output ONLY raw JSON:
    {{
      "title": "Masterclass Title",
      "overview": "The 'Why' and Mental Model plan",
      "suggested_design": {{ "summary": "Goal of the architecture", "components": [] }},
      "milestone_chapters": [ {{ "id": "ms-1", "title": "...", "focus": "..." }} ],
      "diagrams": [ {{ "id": "d1", "title": "...", "description": "...", "type": "...", "placement": "preamble|design|ms_id" }} ]
    }}
    Plan 15+ diagrams. 
    """
    res = LLM_MASTER.invoke(
        [
            SystemMessage(
                content="You are a world-class architect. Output ONLY raw JSON."
            ),
            HumanMessage(content=prompt),
        ]
    )
    blueprint = json.loads(
        re.search(r"(\{.*\}|\[.*\])", res.content, re.DOTALL).group(0)
    )
    return {
        "blueprint": blueprint,
        "diagrams_to_generate": blueprint.get("diagrams", []),
        "current_step": "preamble",
    }


def preamble_node(state: GraphState):
    """Writing Introduction & Epiphany Analogy."""
    print(f"  [Agent: Writer] Writing Introduction & Mental Models...")
    bp = state["blueprint"]
    prompt = f"Write the Preamble for '{bp['title']}'. Analogy first. Why this project matters. Use {{{{DIAGRAM:id}}}} for placement. CONTEXT: {bp['overview']}"
    res = LLM_MASTER.invoke([HumanMessage(content=prompt)])
    return {
        "accumulated_md": f"# {bp['title']}\n\n{res.content.strip()}\n",
        "current_step": "design",
    }


def design_spec_node(state: GraphState):
    """Writing the 'Suggested Design' - THE ARCHITECT'S BLUEPRINT."""
    print(f"  [Agent: Architect] Detailing the Suggested Design (The Blueprint)...")
    bp = state["blueprint"]
    prompt = f"""
    TASK: Write 'The Architect's Blueprint' for this project.
    1. High-Level Abstraction: Explain the core components and their interaction.
    2. Interface Specifications: 
       - List structs/classes with detailed fields and reasons.
       - List key functions/APIs with signatures and pseudocode steps.
    3. Project Structure: Suggested file/folder layout.
    4. Mandatory: Use {{{{DIAGRAM:id}}}} for high-level and component diagrams.
    
    PROJECT CONTEXT: {bp["title"]}
    Output Markdown content.
    """
    res = LLM_MASTER.invoke(
        [
            SystemMessage(content="You are a precise technical designer."),
            HumanMessage(content=prompt),
        ]
    )
    return {
        "accumulated_md": state["accumulated_md"]
        + "\n\n# The Architect's Blueprint\n\n"
        + res.content.strip(),
        "current_step": "milestones",
    }


def milestone_node(state: GraphState):
    """Writing practical Milestone implementation chapters."""
    idx = state["current_ms_index"]
    bp = state["blueprint"]
    chapters = bp.get("milestone_chapters", [])
    if idx >= len(chapters):
        return {"current_step": "visualizing"}

    ch = chapters[idx]
    print(f"  [Agent: Writer] Implementation Milestone: {ch['title']}...")
    prompt = f"Explain Milestone: {ch['title']}. Focus: {ch['focus']}. Requirements: 1. Logic Walkthrough. 2. Pitfalls (⚠️). 3. Infrastructure Code. 4. Core Logic Skeleton (TODOs). 5. Verification Code. 6. Grading Table. Use {{{{DIAGRAM:id}}}}. Previous Docs: {state['accumulated_md'][-20000:]}"
    res = LLM_MASTER.invoke([HumanMessage(content=prompt)])

    return {
        "accumulated_md": state["accumulated_md"]
        + f"\n\n## Milestone: {ch['title']}\n\n{res.content.strip()}\n",
        "current_ms_index": idx + 1,
    }


def visualizer_node(state: GraphState):
    if not state["diagrams_to_generate"]:
        return {"current_step": "done"}
    diag = state["diagrams_to_generate"][0]
    print(f"  [Agent: Visualizer] {diag.get('title')}...")
    prompt = f"Generate D2 for: {diag.get('title')}. {D2_GUIDE}. Description: {diag.get('description')}. Full Context: {state['accumulated_md'][-15000:]}"
    if state.get("last_error"):
        prompt += f"\n\nFIX THIS ERROR: {state['last_error']}"
    res = LLM_MASTER.invoke([HumanMessage(content=prompt)])
    return {
        "current_diagram_code": re.sub(r"```d2\n?|```", "", res.content).strip(),
        "current_diagram_meta": diag,
    }


def compiler_node(state: GraphState):
    diag = state["current_diagram_meta"]
    code = state.get("current_diagram_code", "")
    proj_dir = OUTPUT_BASE / state["project_id"]
    d2_path = proj_dir / "diagrams" / f"{diag.get('id', 'diag')}.d2"
    for s in ["capsule", "plaintext", "record", "sticky_note", "border-radius"]:
        code = code.replace(f"shape: {s}", "shape: rectangle")
    d2_path.write_text(code)
    res = subprocess.run(
        ["d2", "--layout=elk", str(d2_path), str(d2_path.with_suffix(".svg"))],
        capture_output=True,
        text=True,
    )
    if res.returncode == 0:
        print(f"    ✓ Success")
        img_link = f"\n![{diag.get('title')}](./diagrams/{diag.get('id')}.svg)\n"
        tag = f"{{{{DIAGRAM:{diag.get('id')}}}}}"
        updated_md = (
            state["accumulated_md"].replace(tag, img_link)
            if tag in state["accumulated_md"]
            else state["accumulated_md"] + img_link
        )
        return {
            "accumulated_md": updated_md,
            "diagrams_to_generate": state["diagrams_to_generate"][1:],
            "diagram_attempt": 0,
            "last_error": None,
        }
    else:
        print(f"    ✗ Failed, retrying...")
        return {
            "diagram_attempt": state.get("diagram_attempt", 0) + 1,
            "last_error": res.stderr,
        }


# --- GRAPH ---
workflow = StateGraph(GraphState)
workflow.add_node("architect", architect_node)
workflow.add_node("preamble", preamble_node)
workflow.add_node("design", design_spec_node)
workflow.add_node("milestones", milestone_node)
workflow.add_node("visualizer", visualizer_node)
workflow.add_node("compiler", compiler_node)

workflow.add_edge(START, "architect")
workflow.add_conditional_edges("architect", lambda x: "preamble")
workflow.add_edge("preamble", "design")
workflow.add_edge("design", "milestones")
workflow.add_conditional_edges(
    "milestones",
    lambda x: "milestones"
    if x["current_ms_index"] < len(x["blueprint"].get("milestone_chapters", []))
    else "visualizer",
)
workflow.add_edge("visualizer", "compiler")
workflow.add_conditional_edges(
    "compiler",
    lambda x: "visualizer"
    if x.get("last_error") and x.get("diagram_attempt", 0) < 3
    else "visualizer_check",
)
workflow.add_conditional_edges(
    "visualizer", lambda x: END if not x.get("diagrams_to_generate") else "visualizer"
)

app = workflow.compile()


def generate_project(project_id):
    print(f"\n>>> V12.1 ARCHITECT'S BLUEPRINT STARTING: {project_id}")
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
            "current_step": "architect",
            "current_ms_index": 0,
            "diagrams_to_generate": [],
            "diagram_attempt": 0,
            "last_error": None,
        }
    )
    with open(OUTPUT_BASE / project_id / "index.md", "w") as f:
        f.write(final_state["accumulated_md"])
    subprocess.run(
        ["npm", "run", "generate:html", "--", project_id],
        cwd=SCRIPT_DIR / ".." / "web",
        capture_output=True,
    )
    print(f"  ✓ MASTERPIECE V12.1 COMPLETE: {project_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", nargs="+", required=True)
    args = parser.parse_args()
    for p in args.projects:
        generate_project(p)
