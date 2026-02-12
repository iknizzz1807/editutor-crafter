import yaml
import re
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

YAML_PATH = Path("data/projects.yaml")


def check_link(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        # Try HEAD first for speed
        response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        if response.status_code >= 400:
            # Fallback to GET for sites that block HEAD
            response = requests.get(
                url, headers=headers, timeout=10, allow_redirects=True
            )
        return url, response.status_code
    except Exception as e:
        return url, str(e)


def validate_all_links():
    print("Reading projects.yaml for links...")
    with open(YAML_PATH, "r") as f:
        content = f.read()

    # Find all http/https links using regex
    urls = list(set(re.findall(r'https?://[^\s<>"\'\]\}]+', content)))
    # Clean trailing punctuation
    urls = [u.rstrip(".,") for u in urls]
    urls = list(set(urls))

    print(f"Found {len(urls)} unique URLs. Validating in parallel...")

    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(check_link, urls))

    broken = [r for r in results if not isinstance(r[1], int) or r[1] >= 400]

    print("\n--- LINK VALIDATION REPORT ---")
    print(f"Total Unique Links: {len(urls)}")
    print(f"Broken/Suspect Links: {len(broken)}")

    if broken:
        for url, status in broken:
            print(f"  [X] Status {status} | {url}")
    else:
        print("  ✓ All links are healthy!")


if __name__ == "__main__":
    validate_all_links()
