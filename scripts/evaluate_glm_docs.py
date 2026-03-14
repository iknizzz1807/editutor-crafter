#!/usr/bin/env python3
"""
Evaluate documentation quality by scoring each project out of 100.
Each project is evaluated individually with detailed feedback.
"""

import os
import subprocess
import threading
from pathlib import Path
from datetime import datetime

# --- Config ---
BASE_DIR = Path(__file__).parent.parent
ARCH_DOCS = BASE_DIR / "data" / "architecture-docs"
OUTPUT_FILE = BASE_DIR / "evaluation_results.md"

MAX_WORKERS = 10
CLAUDE_MODEL = "opus"

PROMPT_TEMPLATE = """Hãy đánh giá tài liệu hướng dẫn dự án dưới đây và cho điểm trên thang 100.

Yêu cầu đánh giá chi tiết các khía cạnh:
1. **Kiến thức chuyên môn** - Nội dung có chính xác, đầy đủ không?
2. **Cấu trúc và trình bày** - Có dễ hiểu, logic không?
3. **Giải thích** - Các khái niệm có được giải thích rõ ràng không?
4. **Giáo dục và hướng dẫn** - Có phù hợp để học không?
5. **Code mẫu** - Code có chính xác, chạy được không?
6. **Phương pháp sư phạm** - Có theo style giảng dạy tốt không?
   - Có nêu mục tiêu học trước không?
   - Có giải thích "tại sao" không chỉ "cái gì"?
   - Có nối kiến thức cũ với mới không?
   - Có dẫn dắt từ dễ đến khó không?
   - Có giải thích chi tiết các khái niệm, thuật ngữ không?
7. **Tính giao dịch** - Ngôn ngữ có thân thiện, dễ hiểu không? Có khuyến khích người học không?
8. **Context bám sát** - Tài liệu có liên kết từ đầu đến cuối không? Có continuity không hay lộn xộn?
9. **Code bám sát** - Code có khớp với nội dung giải thích không? Hay code và chữ rời rạc, không nhất quán?

Cho điểm từ 0-100 và giải thích chi tiết từng điểm mạnh, điểm yếu.

---
# Project: {project_name}

{content}
"""

def read_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def run_claude(prompt: str, project_name: str) -> str:
    """Spawn claude CLI subprocess, feed prompt via stdin."""
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    cmd = ["claude", "--model", CLAUDE_MODEL, "-p", "--dangerously-skip-permissions"]
    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    if result.returncode != 0:
        return f"ERROR (returncode={result.returncode}):\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    return result.stdout

def evaluate_project(project_name: str, results: list, lock: threading.Lock):
    project_path = ARCH_DOCS / project_name / "index.md"

    if not project_path.exists():
        print(f"[SKIP] {project_name}: index.md not found")
        return

    content = read_file(project_path)

    prompt = PROMPT_TEMPLATE.format(
        project_name=project_name,
        content=content,
    )

    print(f"[START] Evaluating: {project_name}")
    response = run_claude(prompt, project_name)
    print(f"[DONE]  Evaluated:  {project_name}")

    # Extract score if present
    score = "?"
    for line in response.split('\n'):
        if 'điểm' in line.lower() or 'score' in line.lower() or '/100' in line:
            if any(c.isdigit() for c in line):
                # Try to find a number
                import re
                nums = re.findall(r'\d+', line)
                if nums:
                    score = nums[0]
                    break

    entry = f"""
## {project_name} - Score: {score}/100
_Evaluated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_

{response}

---
"""
    with lock:
        results.append((project_name, entry))
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(entry)

def main():
    # Get all projects with index.md
    all_projects = sorted([
        d.name for d in ARCH_DOCS.iterdir()
        if d.is_dir() and (d / "index.md").exists()
    ])

    print(f"Found {len(all_projects)} projects to evaluate")
    print(f"Projects: {all_projects}")

    # Init output file with header
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"# Documentation Quality Evaluation\n")
        f.write(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n")
        f.write(f"**Model:** {CLAUDE_MODEL}\n\n")
        f.write(f"**Evaluated {len(all_projects)} projects**\n\n")
        f.write("---\n")

    results = []
    lock = threading.Lock()
    threads = []
    semaphore = threading.Semaphore(MAX_WORKERS)

    def worker(project_name):
        with semaphore:
            evaluate_project(project_name, results, lock)

    for project_name in all_projects:
        t = threading.Thread(target=worker, args=(project_name,), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"\nDone! Results written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
