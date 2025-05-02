import os
import io
import json
import glob
import zipfile
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

# Browser-like headers
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/113.0.0.0 Safari/537.36"
    )
}

# Shot-data ZIP URLs
SHOT_URLS = [
    "https://peter-tanner.com/moneypuck/downloads/shots_2007-2023.zip",
    "https://peter-tanner.com/moneypuck/downloads/shots_2024.zip",
]

METADATA_FILE = "metadata.json"

def load_metadata(save_dir: str) -> dict:
    path = os.path.join(save_dir, METADATA_FILE)
    if os.path.exists(path):
        return json.load(open(path, "r"))
    return {}

def save_metadata(save_dir: str, meta: dict):
    path = os.path.join(save_dir, METADATA_FILE)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

def get_csv_links() -> list[str]:
    """
    Scrape MoneyPuck page for CSV links, then add playoff variants
    only for seasonSummary/.../regular/... paths.
    """
    base = "https://moneypuck.com/data.htm"
    r = requests.get(base, headers=HEADERS); r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # all raw CSV links
    scraped = [
        requests.compat.urljoin(base, a["href"])
        for a in soup.find_all("a", href=True)
        if a["href"].lower().endswith(".csv")
    ]
    urls = list(dict.fromkeys(scraped))

    # add playoffs for seasonSummary/…/regular/…
    playoffs = [
        u.replace("/regular/", "/playoffs/")
        for u in urls
        if "/seasonSummary/" in u and "/regular/" in u
    ]
    urls.extend(playoffs)
    return list(dict.fromkeys(urls))

def download_file(url: str, dest: str, meta: dict) -> bool:
    """
    Download to dest if new or missing. Returns True if we actually wrote a new file.
    Skips HTTP 304 & 404.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    fn = os.path.basename(dest)
    headers = HEADERS.copy()
    if fn in meta and "Last-Modified" in meta[fn]:
        headers["If-Modified-Since"] = meta[fn]["Last-Modified"]

    r = requests.get(url, headers=headers, stream=True)
    if r.status_code == 304:
        print(f"{fn:30} ⟳ not modified")
        return False
    if r.status_code == 404:
        print(f"{fn:30} ⟳ not found, skipping")
        return False
    r.raise_for_status()

    # atomic write
    tmp = dest + ".tmp"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(8_192):
            f.write(chunk)
    os.replace(tmp, dest)

    # record new Last-Modified
    lm = r.headers.get("Last-Modified", datetime.utcnow().isoformat())
    meta[fn] = {"Last-Modified": lm}
    print(f"{fn:30} ✓ downloaded")
    return True

def extract_zip(path: str, out_dir: str):
    """Unzip path into out_dir."""
    with zipfile.ZipFile(path, "r") as z:
        z.extractall(out_dir)
    print(f"    • extracted {os.path.basename(path)} → {out_dir}")

def assert_integrity(shots_dir: str):
    """For each game in shots_dir/*.csv, ensure #SOG + #MISS == total attempts."""
    # 1) load all shot_data CSVs
    paths = glob.glob(os.path.join(shots_dir, "*.csv"))
    df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)

    # 2) flag SOG vs MISS in one shot
    df = df.assign(
        is_sog = ((df["event"] == "SHOT") & (df["shotWasOnGoal"] == 1)) | (df["event"] == "GOAL"),
        is_miss= (df["event"] == "MISS")
    )

    # 3) group and compare
    check = (
        df.groupby("game_id")
          .agg(
              total_events=pd.NamedAgg(column="shotID", aggfunc="size"),
              sog         =pd.NamedAgg(column="is_sog",   aggfunc="sum"),
              miss        =pd.NamedAgg(column="is_miss",  aggfunc="sum"),
          )
          .reset_index()
    )
    check["counted"] = check["sog"] + check["miss"]

    bad = check[check["total_events"] != check["counted"]]
    if not bad.empty:
        print(f"\nIntegrity failures: {len(bad)} games\n")
        for _, row in bad.head(5).iterrows():
            gid = row['game_id']
            sub = df[df['game_id'] == gid]
            print(f"Game {gid}: total={len(sub)}  SOG={sub['is_sog'].sum()}  MISS={sub['is_miss'].sum()}")
        raise AssertionError(f"Integrity check failed for {len(bad)} games")

    print("✔ Integrity check passed for all games")

def run_nightly(save_dir="data"):
    meta = load_metadata(save_dir)
    updated = False

    # 1) Download all MoneyPuck CSVs
    for url in get_csv_links():
        fn   = os.path.basename(url)
        dest = os.path.join(save_dir, fn)
        if download_file(url, dest, meta):
            updated = True

    # 2) Download & extract shot_data zips
    shots_dir = os.path.join(save_dir, "shot_data")
    os.makedirs(shots_dir, exist_ok=True)
    for url in SHOT_URLS:
        fn   = os.path.basename(url)
        dest = os.path.join(shots_dir, fn)
        if download_file(url, dest, meta):
            updated = True
            extract_zip(dest, shots_dir)

    # 3) Verify integrity of just the shot_data CSVs
    assert_integrity(shots_dir)

    # 4) Save updated metadata
    if updated:
        save_metadata(save_dir, meta)

if __name__ == "__main__":
    run_nightly()
