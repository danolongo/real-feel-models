#!/usr/bin/env python3
"""
rf.v1.0.0.data_pipeline.download_data
Download and prepare the Cresci-2017 Twitter bot detection dataset.

Usage:
  python download_data.py
  python download_data.py --output_dir /custom/path

Output: datasets/cresci_2017_merged.csv with columns 'text' and 'label'
  - label=0: human (genuine accounts)
  - label=1: bot (all bot categories)
"""

import logging
import re
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DIRECT_DOWNLOAD_URL = "https://botometer.osome.iu.edu/bot-repository/datasets/cresci-2017/cresci-2017.csv.zip"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Folder → label mapping for the Cresci-2017 directory structure
CRESCI_LABEL_MAP = {
    "genuine_accounts.csv": 0,
    "fake_followers.csv": 1,
    "social_spambots_1.csv": 1,
    "social_spambots_2.csv": 1,
    "social_spambots_3.csv": 1,
    "traditional_spambots_1.csv": 1,
    "traditional_spambots_2.csv": 1,
    "traditional_spambots_3.csv": 1,
    "traditional_spambots_4.csv": 1,
}

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
_MENTION_RE = re.compile(r"@\w+")


def preprocess_text(text: str) -> str:
    if not text or str(text).strip() in ("nan", "None", ""):
        return "[EMPTY]"
    text = str(text).strip()
    text = _URL_RE.sub("http://url.removed", text)
    text = _MENTION_RE.sub("@user", text)
    text = " ".join(text.split())
    return text or "[EMPTY]"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_zip(url: str, raw_dir: Path) -> bool:
    """Stream-download a zip from url and extract into raw_dir."""
    try:
        import requests
    except ImportError:
        log.error("'requests' library not installed. Run: uv sync")
        return False

    log.info(f"Downloading from: {url}")
    log.info("This may take a few minutes (~150 MB)...")

    try:
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
    except Exception as exc:
        log.error(f"Download failed: {exc}")
        return False

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    chunks = []

    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        if chunk:
            chunks.append(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(
                    f"\r  {downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB ({pct:.0f}%)",
                    end="", flush=True,
                )

    print()
    raw_zip = b"".join(chunks)
    log.info(f"Downloaded {len(raw_zip) / 1e6:.1f} MB")

    log.info(f"Extracting to {raw_dir} ...")
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(BytesIO(raw_zip)) as zf:
            zf.extractall(raw_dir)
        log.info("Outer zip extracted.")
    except zipfile.BadZipFile as exc:
        log.error(f"Bad zip file: {exc}")
        return False

    # Extract nested *.csv.zip files (e.g. genuine_accounts.csv.zip → genuine_accounts.csv/)
    nested_zips = list(raw_dir.rglob("*.csv.zip"))
    if nested_zips:
        log.info(f"Found {len(nested_zips)} nested zip(s) — extracting...")
        for nested in nested_zips:
            try:
                with zipfile.ZipFile(nested) as zf:
                    zf.extractall(nested.parent)
                log.info(f"  Extracted: {nested.name}")
                nested.unlink()  # remove the inner zip to save space
            except zipfile.BadZipFile as exc:
                log.warning(f"  Could not extract {nested.name}: {exc}")

    return True


# ---------------------------------------------------------------------------
# Parse the Cresci-2017 folder structure
# ---------------------------------------------------------------------------

def _find_cresci_root(search_root: Path) -> Optional[Path]:
    """BFS search for directory containing Cresci-2017 category folders."""
    known_folders = set(CRESCI_LABEL_MAP.keys())
    queue = [search_root]
    visited: set = set()
    depth = 0
    while queue and depth < 4:
        next_queue = []
        for path in queue:
            if path in visited:
                continue
            visited.add(path)
            if not path.is_dir():
                continue
            children = {p.name for p in path.iterdir() if p.is_dir()}
            if children & known_folders:
                return path
            next_queue.extend(p for p in path.iterdir() if p.is_dir())
        queue = next_queue
        depth += 1
    return None


def load_cresci_from_dir(root: Path) -> Optional[Tuple[List[str], List[int]]]:
    """Walk a local directory and load all Cresci-2017 category tweets."""
    import pandas as pd

    actual_root = _find_cresci_root(root)
    if actual_root is None:
        log.error(
            f"Could not locate Cresci-2017 folder structure under {root}. "
            "Expected folders like 'genuine_accounts.csv/', 'fake_followers.csv/', etc."
        )
        return None

    log.info(f"Cresci-2017 root found at: {actual_root}")

    all_texts: List[str] = []
    all_labels: List[int] = []

    for folder_name, label in CRESCI_LABEL_MAP.items():
        folder_path = actual_root / folder_name
        if not folder_path.exists():
            log.warning(f"  Folder not found, skipping: {folder_path}")
            continue

        tweets_csv = folder_path / "tweets.csv"
        if not tweets_csv.exists():
            log.warning(f"  No tweets.csv in {folder_path}, skipping.")
            continue

        try:
            df = pd.read_csv(tweets_csv, low_memory=False)
        except Exception as exc:
            log.warning(f"  Could not read {tweets_csv}: {exc}")
            continue

        if "text" not in df.columns:
            log.warning(f"  No 'text' column in {tweets_csv}. Columns: {list(df.columns)}")
            continue

        before = len(all_texts)
        for raw_text in df["text"].dropna():
            text = preprocess_text(str(raw_text))
            if len(text) < 10:
                continue
            all_texts.append(text)
            all_labels.append(label)

        added = len(all_texts) - before
        label_str = "human" if label == 0 else "bot"
        log.info(f"  {folder_name}: {added} {label_str} tweets loaded")

    if not all_texts:
        return None

    return all_texts, all_labels


# ---------------------------------------------------------------------------
# Save merged CSV
# ---------------------------------------------------------------------------

def save_merged_csv(texts: List[str], labels: List[int], output_path: Path) -> None:
    import pandas as pd

    df = pd.DataFrame({"text": texts, "label": labels})

    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    after = len(df)
    if before != after:
        log.info(f"Removed {before - after} duplicate rows. {after} unique samples remain.")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    bot_count = (df["label"] == 1).sum()
    human_count = (df["label"] == 0).sum()
    log.info(f"Saved {len(df)} rows to {output_path}")
    log.info(f"  Human (label=0): {human_count}  |  Bot (label=1): {bot_count}")
    log.info(f"  Bot ratio: {bot_count / len(df) * 100:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and prepare Cresci-2017 for bot detection training."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Root directory for datasets/ folder. Defaults to rf.v1.0.0/.",
    )
    args = parser.parse_args()

    if DIRECT_DOWNLOAD_URL == "DIRECT_DOWNLOAD_URL":
        log.error(
            "No download URL set. Open download_data.py and replace "
            "DIRECT_DOWNLOAD_URL with your direct link."
        )
        sys.exit(1)

    base_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent.parent
    datasets_dir = base_dir / "datasets"
    raw_dir = datasets_dir / "cresci_2017_raw"
    output_csv = datasets_dir / "cresci_2017_merged.csv"

    log.info("=" * 60)
    log.info("Cresci-2017 Twitter Bot Detection — Data Downloader")
    log.info("=" * 60)
    log.info(f"Output CSV: {output_csv}")

    if output_csv.exists():
        log.info(f"Merged CSV already exists at {output_csv} — nothing to do.")
        log.info("Delete it and re-run if you want to rebuild.")
        return

    if raw_dir.exists():
        log.info(f"Raw data already extracted at {raw_dir} — skipping download.")
    elif not download_zip(DIRECT_DOWNLOAD_URL, raw_dir):
        sys.exit(1)

    result = load_cresci_from_dir(raw_dir)
    if result is None:
        log.error("Failed to parse dataset from extracted files.")
        sys.exit(1)

    texts, labels = result
    save_merged_csv(texts, labels, output_csv)

    log.info("")
    log.info("Done! To train the model, run:")
    log.info(f"  python train.py --config production --data_path {output_csv}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
