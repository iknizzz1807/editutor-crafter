#!/usr/bin/env python3
"""
Evaluate GLM-5 generated docs vs Claude-written baselines.
Spawns up to 10 parallel Claude workers, each reviewing 2 GLM projects at a time
alongside the 2 Claude baseline projects.
Results written to evaluation_results.md
"""

import os
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from itertools import islice

# --- Config ---
BASE_DIR = Path(__file__).parent.parent
ARCH_DOCS = BASE_DIR / "data" / "architecture-docs"
OUTPUT_FILE = BASE_DIR / "evaluation_results.md"

BASELINE_PROJECTS = ["http-server-basic", "build-event-loop"]
EXCLUDE = set(BASELINE_PROJECTS)

MAX_WORKERS = 10
CLAUDE_MODEL = "opus[1m]"

PROMPT_TEMPLATE = """http-basic và event-loop là được Claude viết. Còn lại là được model GLM-5 viết vì tôi thấy Claude đắt quá. Tôi thích Claude, tuy nhiên tôi lo rằng tài liệu viết bằng GLM sẽ không đảm bảo chất lượng, kiến thức, trình bày, giải thích, giáo dục, hướng dẫn hay code.

Bạn đọc và nhận xét giúp tôi nhé, và tôi có thể yên tâm dùng GLM-5 chưa? So sánh chi tiết từng khía cạnh nhé. Check thật chi tiết đừng bỏ qua bất cứ thứ gì.

Dưới đây là nội dung 4 tài liệu:

---
# [BASELINE 1 - Claude] Project: http-server-basic
{baseline1}

---
# [BASELINE 2 - Claude] Project: build-event-loop
{baseline2}

---
# [GLM-5 PROJECT 1] Project: {glm1_name}
{glm1}

---
# [GLM-5 PROJECT 2] Project: {glm2_name}
{glm2}
"""

def read_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def run_claude(prompt: str, projects: list[str]) -> str:
    """Spawn claude CLI subprocess, feed prompt via stdin to avoid ARG_MAX limit."""
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)  # allow nested claude invocation
    result = subprocess.run(
        ["claude", f"--model={CLAUDE_MODEL}", "-p", "--dangerously-skip-permissions"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    if result.returncode != 0:
        return f"ERROR (returncode={result.returncode}):\n{result.stderr}"
    return result.stdout

def evaluate_pair(baseline1: str, baseline2: str, proj1: str, proj2: str, results: list, lock: threading.Lock):
    glm1_path = ARCH_DOCS / proj1 / "index.md"
    glm2_path = ARCH_DOCS / proj2 / "index.md"

    if not glm1_path.exists():
        print(f"[SKIP] {proj1}: index.md not found")
        return
    if not glm2_path.exists():
        print(f"[SKIP] {proj2}: index.md not found")
        return

    glm1 = read_file(glm1_path)
    glm2 = read_file(glm2_path)

    prompt = PROMPT_TEMPLATE.format(
        baseline1=baseline1,
        baseline2=baseline2,
        glm1_name=proj1,
        glm1=glm1,
        glm2_name=proj2,
        glm2=glm2,
    )

    print(f"[START] Evaluating: {proj1} + {proj2}")
    response = run_claude(prompt, [proj1, proj2])
    print(f"[DONE]  Evaluated:  {proj1} + {proj2}")

    entry = f"""
## Batch: {proj1} & {proj2}
_Evaluated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_

{response}

---
"""
    with lock:
        results.append((proj1, proj2, entry))

def chunked(lst, n):
    it = iter(lst)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk

def main():
    # Load baselines
    b1 = read_file(ARCH_DOCS / "http-server-basic" / "index.md")
    b2 = read_file(ARCH_DOCS / "build-event-loop" / "index.md")

    # Get all GLM projects (those with index.md, excluding baselines)
    all_projects = sorted([
        d.name for d in ARCH_DOCS.iterdir()
        if d.is_dir() and d.name not in EXCLUDE and (d / "index.md").exists()
    ])

    print(f"Found {len(all_projects)} GLM projects to evaluate")
    print(f"Projects: {all_projects}")

    # Pair them up (2 per batch)
    pairs = list(chunked(all_projects, 2))
    # Drop incomplete last pair if only 1 project left
    # (still evaluate it by pairing with the first project)
    if pairs and len(pairs[-1]) == 1:
        # pair with first available
        pairs[-1].append(all_projects[0])

    print(f"Total batches: {len(pairs)} | Max parallel workers: {MAX_WORKERS}")

    results = []
    lock = threading.Lock()
    threads = []
    semaphore = threading.Semaphore(MAX_WORKERS)

    def worker(proj1, proj2):
        with semaphore:
            evaluate_pair(b1, b2, proj1, proj2, results, lock)

    for pair in pairs:
        proj1, proj2 = pair[0], pair[1]
        t = threading.Thread(target=worker, args=(proj1, proj2), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Write output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"# GLM-5 vs Claude Documentation Evaluation\n")
        f.write(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n")
        f.write(f"**Baselines (Claude-written):** http-server-basic, build-event-loop\n\n")
        f.write(f"**Evaluated {len(all_projects)} GLM-5 projects in {len(pairs)} batches**\n\n")
        f.write("---\n")

        # Sort by project names for consistent order
        results.sort(key=lambda x: x[0])
        for _, _, entry in results:
            f.write(entry)

    print(f"\nDone! Results written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
