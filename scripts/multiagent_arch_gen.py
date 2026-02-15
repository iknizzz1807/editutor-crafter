#!/usr/bin/env python3
import argparse, os, re, json, yaml, time, subprocess
from pathlib import Path
from openai import OpenAI

# --- CONFIGURATION ---
client = OpenAI(base_url="http://127.0.0.1:7999/v1", api_key="mythong2005")
MODEL = "gemini_cli/gemini-3-pro-preview"
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / ".." / "data"
YAML_PATH = DATA_DIR / "projects.yaml"
OUTPUT_BASE = DATA_DIR / "architecture-docs"

D2_GUIDE = """
=== D2 SYNTAX RULES (STRICT) ===
1. SHAPES: [rectangle, square, page, parallelogram, document, cylinder, queue, package, step, callout, stored_data, person, diamond, oval, circle, hexagon, cloud, class, sql_table].
2. LABELS: Always wrap in "double quotes".
3. CONTAINERS: Parent { Child1; Child2 }
4. CLASSES: classes: { style_id: { style: { fill: "#f3f4f6"; stroke: "#374151" } } }; node_id.class: style_id
5. NO ILLEGAL KEYS: No 'font-weight', 'fill-opacity', 'theme', 'mermaid'.
"""

# ===========================================================================
# AGENT CLASSES
# ===========================================================================


class ArchitectAgent:
    def run(self, project_meta):
        prompt = f"""
        Project: {project_meta["name"]}
        Description: {project_meta["description"]}
        Milestones: {json.dumps(project_meta.get("milestones", []))}
        
        TASK: Create a Master Blueprint & Technical Contract.
        Output ONLY raw JSON:
        {{
          "title": "...",
          "overview": "...",
          "contract": {{
            "structs": [ {{ "name": "...", "fields": [ {{ "name": "...", "type": "..." }} ], "description": "..." }} ],
            "interfaces": [ {{ "name": "...", "signature": "...", "description": "..." }} ],
            "constants": [ {{ "name": "...", "value": "...", "description": "..." }} ]
          }},
          "milestones": [ {{ "id": "ms-1", "title": "...", "summary": "..." }} ],
          "diagrams": [ {{ "id": "diag-1", "title": "...", "description": "...", "type": "flowchart", "milestone_id": "ms-1" }} ]
        }}
        Plan at least 10 diagrams.
        """
        return self._call(prompt)

    def _call(self, prompt):
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Master Architect. Output ONLY raw JSON.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        text = res.choices[0].message.content
        try:
            return json.loads(re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL).group(0))
        except:
            return None


class CriticAgent:
    def review(self, blueprint):
        prompt = f"Critique this blueprint for logical gaps or naming mismatches: {json.dumps(blueprint)}. Return a list of issues or 'PASSED'."
        res = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content


class WriterAgent:
    def write_section(self, milestone, blueprint, accumulated_docs, lang):
        prompt = f"""
        WRITING TASK: Milestone '{milestone["title"]}'
        CONTRACT: {json.dumps(blueprint["contract"])}
        SUMMARY: {milestone["summary"]}
        
        RULES:
        1. Use Diagram Tags: {{{{DIAGRAM:id}}}} where appropriate. Available IDs: {[d["id"] for d in blueprint.get("diagrams", [])]}
        2. Mental Model -> ADR -> Pitfalls (⚠️) -> Grading Matrix -> Code Scaffold -> Test Code.
        3. NO MERMAID. Language: {lang}. Use ```{lang} for code.
        
        PREVIOUS CONTEXT (DO NOT REPEAT):
        {accumulated_docs[-20000:]}
        
        Output Markdown for THIS milestone only.
        """
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32000,
        )
        return res.choices[0].message.content


class VisualizerAgent:
    def generate(self, diag, contract, full_context):
        prompt = f"""
        Generate D2 code for: {diag["title"]}
        Description: {diag["description"]}
        CONTRACT: {json.dumps(contract)}
        {D2_GUIDE}
        CONTEXT: {full_context[-10000:]}
        Output ONLY raw D2 code.
        """
        res = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content


# ===========================================================================
# COORDINATOR
# ===========================================================================


class MultiAgentOrchestrator:
    def __init__(self, project_id):
        self.project_id = project_id
        self.output_dir = OUTPUT_BASE / project_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "diagrams").mkdir(exist_ok=True)
        self.architect = ArchitectAgent()
        self.critic = CriticAgent()
        self.writer = WriterAgent()
        self.visualizer = VisualizerAgent()

    def run(self):
        print(f"\n>>> ORCHESTRATING: {self.project_id}")
        meta = self._load_meta()
        if not meta:
            return print("Error: Meta not found.")

        # 1. Blueprint Loop
        print("  - [Agent: Architect] Planning...")
        blueprint = self.architect.run(meta)
        if not blueprint:
            return print("Failed to gen blueprint.")

        print("  - [Agent: Critic] Reviewing Blueprint...")
        critique = self.critic.review(blueprint)
        if "PASSED" not in critique:
            print(f"    ! Reworking blueprint based on: {critique[:50]}...")
            blueprint = self.architect.run(
                {
                    "name": meta["name"],
                    "description": f"{meta['description']}\nFix issues: {critique}",
                    "milestones": meta.get("milestones"),
                }
            )

        with open(self.output_dir / "blueprint.json", "w") as f:
            json.dump(blueprint, f, indent=2)

        # 2. Section Loop
        full_md = f"# {blueprint.get('title', self.project_id)}\n\n{blueprint.get('overview', '')}\n\n"
        lang = meta.get("languages", ["c"])[0]
        if isinstance(lang, dict):
            lang = lang.get("recommended", ["c"])[0]

        for ms in blueprint.get("milestones", []):
            print(f"  - [Agent: Writer] Writing: {ms['title']}...")
            section = self.writer.write_section(ms, blueprint, full_md, lang)
            full_md += f"\n\n## {ms['title']}\n\n{section}\n"

        # 3. Visualizer Loop
        print("  - [Agent: Visualizer] Generating Diagrams...")
        for diag in blueprint.get("diagrams", []):
            print(f"    - Diagram: {diag['title']}")
            d2_path = self.output_dir / "diagrams" / f"{diag['id']}.d2"

            success = False
            for attempt in range(3):
                code = self.visualizer.generate(diag, blueprint["contract"], full_md)
                code = self._sanitize_d2(code)
                d2_path.write_text(code)
                res = subprocess.run(
                    [
                        "d2",
                        "--layout=elk",
                        str(d2_path),
                        str(d2_path.with_suffix(".svg")),
                    ],
                    capture_output=True,
                    text=True,
                )
                if res.returncode == 0:
                    print("      ✓ Success")
                    success = True
                    break
                else:
                    print(f"      ✗ Retry {attempt + 1}")

            tag, link = (
                f"{{{{DIAGRAM:{diag['id']}}}}}",
                f"\n![{diag['title']}](./diagrams/{diag['id']}.svg)\n",
            )
            if tag in full_md:
                full_md = full_md.replace(tag, link)
            elif success:
                full_md += link

        with open(self.output_dir / "index.md", "w") as f:
            f.write(full_md)
        print(f"  - Building Web Assets for {self.project_id}...")
        subprocess.run(
            ["npm", "run", "generate:html", "--", self.project_id],
            cwd=SCRIPT_DIR / ".." / "web",
            capture_output=True,
        )
        print(f"  ✓ DONE: {self.project_id}")

    def _load_meta(self):
        with open(YAML_PATH) as f:
            data = yaml.safe_load(f)
            for d in data.get("domains", []):
                for l in ["beginner", "intermediate", "advanced", "expert"]:
                    for p in d.get("projects", {}).get(l, []):
                        if p["id"] == self.project_id:
                            return p
        return None

    def _sanitize_d2(self, code):
        code = re.sub(r"```d2\n?|```", "", code).strip()
        illegal = [
            "capsule",
            "plaintext",
            "record",
            "sticky_note",
            "double_circle",
            "column",
            "grid",
        ]
        for s in illegal:
            code = code.replace(f"shape: {s}", "shape: rectangle")
        return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", nargs="+", required=True)
    args = parser.parse_args()
    for p in args.projects:
        MultiAgentOrchestrator(p).run()
