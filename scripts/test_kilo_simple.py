import subprocess
import time
import os


def test_kilo_run():
    print("Testing 'kilo run' via subprocess...")

    full_prompt = "Say 'Hello from Kilo' and nothing else."
    cmd = ["kilo", "run", full_prompt]

    print(f"Test 1: Standard run (Timeout 10s)...")
    try:
        # Explicitly set stdin to DEVNULL to prevent hanging on stdin read
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL
        )
        print(f"Success! Output: {result.stdout}")
    except subprocess.TimeoutExpired:
        print("FAILED: Timed out.")

    print(f"\nTest 2: With input='' (Timeout 10s)...")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, input=""
        )
        print(f"Success! Output: {result.stdout}")
    except subprocess.TimeoutExpired:
        print("FAILED: Timed out.")

    print(f"\nTest 3: Echo pipe (Timeout 10s)...")
    try:
        # Simulating: echo "prompt" | kilo run
        # Note: kilo run might not accept stdin for the message, but let's check.
        p1 = subprocess.Popen(["echo", full_prompt], stdout=subprocess.PIPE)
        p2 = subprocess.run(
            ["kilo", "run"], stdin=p1.stdout, capture_output=True, text=True, timeout=10
        )
        print(f"Success! Output: {result.stdout}")
    except subprocess.TimeoutExpired:
        print("FAILED: Timed out.")


if __name__ == "__main__":
    test_kilo_run()
