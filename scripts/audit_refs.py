import yaml
import re
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

YAML_PATH = Path("data/projects.yaml")


def check_url(url):
    try:
        # Some sites block headless requests, so we use a common User-Agent
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        if response.status_code >= 400:
            # Try GET if HEAD fails
            response = requests.get(
                url, headers=headers, timeout=10, allow_redirects=True
            )
        return url, response.status_code
    except Exception as e:
        return url, str(e)


def audit_references():
    with open(YAML_PATH, "r") as f:
        data = yaml.safe_load(f)

    urls = []
    # Collect all URLs from resources
    for domain in data.get("domains", []):
        for level in ["beginner", "intermediate", "advanced", "expert"]:
            for proj in domain.get("projects", {}).get(level, []):
                for res in proj.get("resources", []):
                    if isinstance(res, dict) and res.get("url"):
                        urls.append(res["url"])

    print(f"Found {len(urls)} external links. Checking status...")

    unique_urls = list(set(urls))
    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(check_url, unique_urls))

    broken = [r for r in results if not isinstance(r[1], int) or r[1] >= 400]

    print(f"\nAudit Result:")
    print(f"Total Unique Links: {len(unique_urls)}")
    print(f"Broken Links: {len(broken)}")
    for url, status in broken:
        print(f" - {url} | Status: {status}")


audit_references()
