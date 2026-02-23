import re
import os
import subprocess
from pathlib import Path

# 1. Deduplicate markers in index.md
idx_path = Path("data/architecture-docs/build-os/index.md")
if idx_path.exists():
    content = idx_path.read_text()
    # Remove duplicated consecutive MS_ID markers
    fixed = re.sub(r"(<!-- MS_ID: .*? -->)\n\1", r"\1", content)
    idx_path.write_text(fixed)
    print("Deduplicated markers in index.md")

# 2. Fix D2 syntax in failed diagrams
diag_dir = Path("data/architecture-docs/build-os/diagrams")
for d2_file in diag_dir.glob("*.d2"):
    svg_file = d2_file.with_suffix(".svg")
    # We fix only if SVG is missing (indicates previous failure)
    if not svg_file.exists():
        print(f"Fixing {d2_file.name}...")
        d2_content = d2_file.read_text()

        # FIX A: Pipe collision in block strings
        fixed_d2 = re.sub(
            r"\|(\w+)\n(.*?)\n\s*\|", r"|'\1\n\2\n'|", d2_content, flags=re.DOTALL
        )
        fixed_d2 = re.sub(
            r":\s*\|\n(.*?)\n\s*\|", r": |'md\n\1\n'|", fixed_d2, flags=re.DOTALL
        )

        # FIX B: Mixed apostrophe block string termination (common in tables)
        fixed_d2 = re.sub(
            r"\|'md\n(.*?)\n\s*\|(?!\')", r"|'md\n\1\n'|", fixed_d2, flags=re.DOTALL
        )
        # Fix double pipe incorrectly used in tables
        fixed_d2 = fixed_d2.replace("'| Flag | Bit |", "| Flag | Bit |")
        fixed_d2 = fixed_d2.replace("'|------|-----|", "|------|-----|")

        # FIX C: Reserved keywords
        fixed_d2 = re.sub(r"^left:", "left_side:", fixed_d2, flags=re.MULTILINE)
        fixed_d2 = re.sub(r"^right:", "right_side:", fixed_d2, flags=re.MULTILINE)
        fixed_d2 = fixed_d2.replace("left.", "left_side.")
        fixed_d2 = fixed_d2.replace("right.", "right_side.")
        fixed_d2 = fixed_d2.replace("left ->", "left_side ->")
        fixed_d2 = fixed_d2.replace("right ->", "right_side ->")
        fixed_d2 = fixed_d2.replace("steps:", "details:")

        # FIX D: Near objects (ELK doesn't support)
        # Find 'near: obj_name' and remove it or change to constant
        fixed_d2 = re.sub(r"near:\s+[a-zA-Z0-9_.]+(?!\n)", "near: top-right", fixed_d2)

        # FIX E: Unquoted labels with special characters
        # Find patterns like 'id: Label [info] {' and change to 'id: "Label [info]" {'
        # This is a bit risky but let's try for common patterns
        fixed_d2 = re.sub(
            r'^(\s*)(\w+):\s+([^"\'\n|{]+?)\s*\{',
            r'\1\2: "\3" {',
            fixed_d2,
            flags=re.MULTILINE,
        )

        d2_file.write_text(fixed_d2)

        # Try compiling
        res = subprocess.run(
            ["d2", "--layout=elk", str(d2_file), str(svg_file)],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            print(f"  ✓ Compiled: {svg_file.name}")
        else:
            print(f"  ✗ Still failing: {d2_file.name}")
            # print(res.stderr)

# 3. Update index.md with successfully compiled diagrams
if idx_path.exists():
    content = idx_path.read_text()
    for svg_file in diag_dir.glob("*.svg"):
        diag_id = svg_file.stem
        placeholder = f"{{{{DIAGRAM:{diag_id}}}}}"
        if placeholder in content:
            img_link = f"\n![{diag_id}](./diagrams/{diag_id}.svg)\n"
            content = content.replace(placeholder, img_link)
    idx_path.write_text(content)
    print("Updated index.md with new diagram links")
