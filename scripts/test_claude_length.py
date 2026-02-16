import subprocess
import time


def test_config(name, cmd_args, input_text):
    print(f"\n=== TESTING CONFIG: {name} ===")
    cmd = [
        "claude",
        "-p",
        "--model",
        "sonnet",
        "--dangerously-skip-permissions",
        "--tools",
        "",
    ] + cmd_args

    start_time = time.time()
    result = subprocess.run(cmd, input=input_text, capture_output=True, text=True)
    duration = time.time() - start_time

    output = result.stdout
    print(f"Duration: {duration:.1f}s")
    print(f"Output Length: {len(output)} chars")
    print(f"Sample (First 200 chars):\n{output[:200]}...")
    if len(output) < 500:
        print("!!! WARNING: Output too short.")


prompt = "Write a very long, detailed essay about the internal architecture of SQLite, specifically how the B-Tree and Pager layers interact. Aim for at least 2000 words. Explain every detail."

# Test 1: Current default
test_config("Default", [], prompt)

# Test 2: With --effort high
test_config("Effort High", ["--effort", "high"], prompt)

# Test 3: System Prompt vs User Prompt
test_config(
    "System Prompt Split",
    [
        "--system-prompt",
        "You are a master technical writer. Write extremely long and detailed guides.",
    ],
    prompt,
)
