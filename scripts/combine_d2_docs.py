import os
import re
from pathlib import Path

BASE_DIR = Path("/home/ikniz/Downloads/d2-docs-master")
DOCS_TOUR = BASE_DIR / "docs" / "tour"
STATIC_D2 = BASE_DIR / "static" / "d2"
BESPOKE_D2 = BASE_DIR / "static" / "bespoke-d2"
OUTPUT_FILE = Path(
    "/home/ikniz/Work/Coding/AI_MachineLearning/editutor-crafter/d2_examples/d2_docs.md"
)

# Priority order for critical syntax
CORE_FILES = [
    "hello-world.md",
    "shapes.md",
    "connections.md",
    "containers.md",
    "style.md",
    "classes.md",
    "vars.md",
    "globs.md",
    "sql-tables.md",
    "uml-classes.md",
    "sequence-diagrams.md",
    "grid-diagrams.md",
    "text.md",
    "icons.md",
    "linking.md",
    "positions.md",
    "composition.md",
    "layers.md",
    "scenarios.md",
    "steps.md",
    "imports.md",
]


def resolve_d2_path(rel_path):
    rel_path = rel_path.strip("'\"")
    # Map @site/static/d2/filename.d2 to local path
    filename = rel_path.split("/")[-1]
    p1 = STATIC_D2 / filename
    if p1.exists():
        return p1
    p2 = BESPOKE_D2 / filename
    if p2.exists():
        return p2
    return None


def extract_d2_examples(content):
    # Find {VarName} patterns
    examples = re.findall(r"\{(\w+)\}", content)
    return list(set(examples))


def process_md_file(file_path):
    raw_content = file_path.read_text()

    # 1. Extract D2 imports
    imports = dict(re.findall(r"import (\w+) from ['\"](.+?)['\"];", raw_content))

    # 2. Clean prose but keep meaningful headers and descriptions
    lines = raw_content.splitlines()
    clean_lines = []
    skip = False
    for line in lines:
        if line.startswith("---"):
            skip = not skip
            continue
        if skip or line.startswith("import "):
            continue

        # Replace components with simple text
        line = re.sub(r"<CodeBlock.*?>", "", line)
        line = re.sub(r"</CodeBlock>", "", line)
        line = re.sub(r"<div.*?>.*?</div>", "", line, flags=re.DOTALL)
        line = re.sub(r":::\w+", "", line)
        line = re.sub(r":::", "", line)

        # Inject D2 code directly into the line if it's a variable reference
        vars_in_line = re.findall(r"\{(\w+)\}", line)
        for v in vars_in_line:
            if v in imports:
                d2_path = resolve_d2_path(imports[v])
                if d2_path:
                    d2_code = d2_path.read_text().strip()
                    line = line.replace(f"{{{v}}}", f"\n```d2\n{d2_code}\n```\n")

        if line.strip():
            clean_lines.append(line)

    return "\n".join(clean_lines)


def main():
    # Only use CORE_FILES to keep it concise but complete
    full_docs = ["# D2 COMPACT REFERENCE (v0.7.1)\n"]

    for filename in CORE_FILES:
        md_file = DOCS_TOUR / filename
        if not md_file.exists():
            continue

        content = process_md_file(md_file)
        full_docs.append(f"\n## {md_file.stem.upper()}\n")
        full_docs.append(content)

    result = "\n".join(full_docs)
    result = re.sub(r"\n{3,}", "\n\n", result)

    # Final safety: truncate if still too long, but here we aim for ~2000 lines
    OUTPUT_FILE.write_text(result)
    print(f"Compact reference created: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
