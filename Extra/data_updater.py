import requests
import pandas as pd
from bs4 import BeautifulSoup
import io
import os
import json
import zipfile
from urllib.parse import urlparse

# Browser-like headers\
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/113.0.0.0 Safari/537.36"
    )
}

# Extra shot-data ZIP URLs
SHOT_URLS = [
    "https://peter-tanner.com/moneypuck/downloads/shots_2007-2023.zip",
    "https://peter-tanner.com/moneypuck/downloads/shots_2024.zip",
]

METADATA_FILE = "metadata.json"

def load_metadata(save_dir):
    path = os.path.join(save_dir, METADATA_FILE)
    if os.path.exists(path):
        return json.load(open(path, "r"))
    return {}

def save_metadata(save_dir, meta):
    path = os.path.join(save_dir, METADATA_FILE)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

def get_csv_links():
    """Scrape MoneyPuck page for CSV links and generate playoff variants."""
    base = "https://moneypuck.com/data.htm"
    r = requests.get(base, headers=HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    scraped = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".csv"):
            scraped.append(requests.compat.urljoin(base, href))

    # Dedupe
    urls = list(dict.fromkeys(scraped))

    # Generate playoff variants
    playoff_urls = []
    for u in urls:
        if "/seasonSummary/" in u and "/regular/" in u:
            playoff_urls.append(u.replace("/regular/", "/playoffs/"))

    urls.extend(playoff_urls)
    urls = list(dict.fromkeys(urls))

    print("Found CSV URLs:")
    for u in urls:
        print("  ", u)
    return urls


def download_data(save_dir="data"):
    os.makedirs(save_dir, exist_ok=True)
    meta = load_metadata(save_dir)
    updated = False

    # 1) CSV files into structured folders
    for url in get_csv_links():
        parsed = urlparse(url).path
        rel = parsed.split("/moneypuck/")[-1].lstrip('/')
        dest = os.path.join(save_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        fn = os.path.basename(dest)
        headers = HEADERS.copy()
        if fn in meta and "Last-Modified" in meta[fn]:
            headers["If-Modified-Since"] = meta[fn]["Last-Modified"]

        print(f"{rel:60}", end=" ")
        try:
            r = requests.get(url, headers=headers)
            if r.status_code == 304:
                print("⟳ not modified, skipping")
                continue
            r.raise_for_status()

            df = pd.read_csv(io.StringIO(r.text))
            tmp = dest + ".tmp"
            df.to_csv(tmp, index=False)
            if os.path.exists(dest):
                os.chmod(dest, 0o666)
                os.remove(dest)
            os.replace(tmp, dest)
            print("✓ saved")

            meta[fn] = {"Last-Modified": r.headers.get("Last-Modified")}
            updated = True
        except Exception as e:
            print(f"✗ failed ({e})")

    # 2) Shot-data ZIPs into shot_data folder
    shot_dir = os.path.join(save_dir, "shot_data")
    os.makedirs(shot_dir, exist_ok=True)

    for url in SHOT_URLS:
        fn = os.path.basename(url)
        dest = os.path.join(shot_dir, fn)
        tmp = dest + ".tmp"

        headers = HEADERS.copy()
        if fn in meta and "Last-Modified" in meta[fn]:
            headers["If-Modified-Since"] = meta[fn]["Last-Modified"]

        print(f"shot_data/{fn:35}", end=" ")
        try:
            r = requests.get(url, headers=headers, stream=True)
            # If not modified and file exists, skip
            if r.status_code == 304 and os.path.exists(dest):
                print("⟳ not modified, skipping")
                continue
            # If not modified but file missing, re-fetch without conditional
            if r.status_code == 304:
                r = requests.get(url, headers=HEADERS, stream=True)
            r.raise_for_status()

            # Write new zip
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            if os.path.exists(dest):
                os.chmod(dest, 0o666)
                os.remove(dest)
            os.replace(tmp, dest)
            print("✓ downloaded")

            # Extract CSVs
            with zipfile.ZipFile(dest, 'r') as z:
                z.extractall(shot_dir)
            print("  • extracted")

            meta[fn] = {"Last-Modified": r.headers.get("Last-Modified")}
            updated = True
        except Exception as e:
            print(f"✗ failed ({e})")

    if updated:
        save_metadata(save_dir, meta)


if __name__ == "__main__":
    download_data()