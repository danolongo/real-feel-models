#!/usr/bin/env python3
"""
rf.v1.0.0.data-pipeline.download_data
Download and prepare the Cresci-2017 Twitter bot detection dataset.

Strategy:
  1. Try HuggingFace datasets hub (best UX, automatic caching)
  2. Fall back to direct Zenodo download (official canonical source)

Output: datasets/cresci_2017_merged.csv with columns 'text' and 'label'
  - label=0: human (genuine accounts)
  - label=1: bot (all bot categories)

Usage:
  python download_data.py
  python download_data.py --output_dir /custom/path
  python download_data.py --method hf        # HuggingFace only
  python download_data.py --method zenodo    # Zenodo direct download only
"""

import argparse
import logging
import os
import sys
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HuggingFace dataset identifiers to try (in priority order)
HF_DATASET_CANDIDATES = [
    # Cresci-2017 mirrors on the Hub
    "twitterbot/cresci-2017",
    "cresci-2017/cresci-2017",
    "bot-detection/cresci-2017",
]

# Zenodo record for Cresci-2017 (DOI: 10.5281/zenodo.1482079)
ZENODO_RECORD_ID = "1482079"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Folder → label mapping for the Cresci-2017 directory structure
CRESCI_LABEL_MAP = {
    "genuine_accounts.csv": 0,       # human
    "fake_followers.csv": 1,         # bot
    "social_spambots_1.csv": 1,
    "social_spambots_2.csv": 1,
    "social_spambots_3.csv": 1,
    "traditional_spambots_1.csv": 1,
    "traditional_spambots_2.csv": 1,
    "traditional_spambots_3.csv": 1,
    "traditional_spambots_4.csv": 1,
}

# ---------------------------------------------------------------------------
# Text preprocessing (matches DatasetLoader.preprocess_text in data.py)
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
# HuggingFace download path
# ---------------------------------------------------------------------------

def try_hf_download() -> Optional[Tuple[List[str], List[int]]]:
    """
    Attempt to load the Cresci-2017 dataset from HuggingFace Hub.
    Returns (texts, labels) on success or None on failure.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        log.warning("'datasets' library not installed — skipping HuggingFace path.")
        return None

    for dataset_id in HF_DATASET_CANDIDATES:
        log.info(f"Trying HuggingFace dataset: {dataset_id}")
        try:
            ds = load_dataset(dataset_id)
            return _extract_hf_texts_labels(ds)
        except Exception as exc:
            log.debug(f"  {dataset_id} failed: {exc}")

    # Generic fallback: look for any split that has text+label columns
    log.info("Named Cresci mirrors not found. Trying generic 'text'+'label' HF search...")
    generic_candidates = [
        "valurank/Twitter-Bots-Accounts",
        "social-media-ai/twitter-bots",
        "papluca/twitter-bot-detection",
    ]
    for dataset_id in generic_candidates:
        log.info(f"Trying generic HF dataset: {dataset_id}")
        try:
            ds = load_dataset(dataset_id)
            result = _extract_hf_texts_labels(ds)
            if result:
                log.info(f"  SUCCESS: {dataset_id} ({len(result[0])} samples)")
                return result
        except Exception as exc:
            log.debug(f"  {dataset_id} failed: {exc}")

    return None


def _extract_hf_texts_labels(ds) -> Optional[Tuple[List[str], List[int]]]:
    """
    Given a HuggingFace DatasetDict, extract texts and labels.
    Handles multiple column naming conventions.
    """
    try:
        import pandas as pd
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        import pandas as pd

    all_texts: List[str] = []
    all_labels: List[int] = []

    # Combine all available splits
    for split_name, split_data in ds.items():
        df = split_data.to_pandas()

        # Identify text column
        text_col = None
        for candidate in ("text", "tweet", "content", "tweet_text"):
            if candidate in df.columns:
                text_col = candidate
                break

        # Identify label column
        label_col = None
        for candidate in ("label", "bot", "is_bot", "class", "account_type"):
            if candidate in df.columns:
                label_col = candidate
                break

        if text_col is None or label_col is None:
            log.warning(
                f"  Split '{split_name}' missing text/label columns "
                f"(found: {list(df.columns)}). Skipping."
            )
            continue

        for _, row in df.iterrows():
            raw_text = row[text_col]
            raw_label = row[label_col]

            text = preprocess_text(raw_text)
            if len(text) < 10:
                continue

            # Normalise label to 0/1
            if isinstance(raw_label, (int, float)):
                label = int(raw_label)
            elif isinstance(raw_label, str):
                low = raw_label.lower()
                if low in ("human", "genuine", "0"):
                    label = 0
                elif low in ("bot", "fake", "spam", "1"):
                    label = 1
                else:
                    log.debug(f"  Unknown label value '{raw_label}' — skipping row.")
                    continue
            else:
                continue

            all_texts.append(text)
            all_labels.append(label)

        log.info(f"  Split '{split_name}': {len(all_texts)} samples accumulated")

    if not all_texts:
        return None

    return all_texts, all_labels


# ---------------------------------------------------------------------------
# Zenodo / direct download path
# ---------------------------------------------------------------------------

def try_zenodo_download(raw_dir: Path) -> Optional[Tuple[List[str], List[int]]]:
    """
    Download the Cresci-2017 zip from Zenodo and extract into raw_dir.
    Returns (texts, labels) on success or None on failure.
    """
    try:
        import requests
    except ImportError:
        log.warning("'requests' library not installed — skipping Zenodo path.")
        return None

    log.info(f"Fetching Zenodo record metadata: record {ZENODO_RECORD_ID}")
    try:
        meta_resp = requests.get(ZENODO_API_URL, timeout=30)
        meta_resp.raise_for_status()
        record = meta_resp.json()
    except Exception as exc:
        log.error(f"Failed to fetch Zenodo metadata: {exc}")
        return None

    # Find the zip file in the record's files list
    files = record.get("files", [])
    zip_entry = None
    for f in files:
        fname = f.get("key", "") or f.get("filename", "")
        if fname.lower().endswith(".zip"):
            zip_entry = f
            break

    if zip_entry is None:
        log.error("No zip file found in Zenodo record. Files available:")
        for f in files:
            log.error(f"  {f.get('key') or f.get('filename', '?')}")
        return None

    # Zenodo API v2 uses 'links.self' for the download URL
    download_url = (
        zip_entry.get("links", {}).get("self")
        or zip_entry.get("links", {}).get("download")
    )
    if not download_url:
        # Construct manually
        fname = zip_entry.get("key") or zip_entry.get("filename")
        download_url = f"https://zenodo.org/record/{ZENODO_RECORD_ID}/files/{fname}"

    log.info(f"Downloading Cresci-2017 zip from: {download_url}")
    log.info("This may take a few minutes (~150 MB)...")

    try:
        resp = requests.get(download_url, stream=True, timeout=120)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunks = []

        for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
            if chunk:
                chunks.append(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Progress: {downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)

        print()  # newline after progress
        raw_zip = b"".join(chunks)
        log.info(f"Downloaded {len(raw_zip) / 1e6:.1f} MB")

    except Exception as exc:
        log.error(f"Download failed: {exc}")
        return None

    # Extract the zip
    log.info(f"Extracting to {raw_dir} ...")
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(BytesIO(raw_zip)) as zf:
            zf.extractall(raw_dir)
        log.info("Extraction complete.")
    except zipfile.BadZipFile as exc:
        log.error(f"Bad zip file: {exc}")
        return None

    return load_cresci_from_dir(raw_dir)


# ---------------------------------------------------------------------------
# Parse the Cresci-2017 folder structure on disk
# ---------------------------------------------------------------------------

def load_cresci_from_dir(root: Path) -> Optional[Tuple[List[str], List[int]]]:
    """
    Walk a local directory and find all Cresci-2017 category folders.
    Handles nested structures (zip may unpack into a subdirectory).
    """
    import pandas as pd

    # The zip may unpack into a subdirectory — find the actual root
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

        # Cresci-2017 tweets.csv has no header; tweet text is column index 2.
        # Files use Latin-1 encoding (legacy Twitter export, ~2010-2013).
        try:
            df = pd.read_csv(
                tweets_csv,
                header=None,
                encoding="latin-1",
                low_memory=False,
                on_bad_lines="skip",
            )
        except Exception as exc:
            log.warning(f"  Could not read {tweets_csv}: {exc}")
            continue

        if df.shape[1] < 3:
            log.warning(f"  Unexpected column count ({df.shape[1]}) in {tweets_csv}, skipping.")
            continue

        before = len(all_texts)
        for raw_text in df.iloc[:, 2].dropna():
            text = preprocess_text(str(raw_text))
            # Skip very short or pure retweet headers
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


def _find_cresci_root(search_root: Path) -> Optional[Path]:
    """
    Recursively search for a directory that contains at least one of the
    known Cresci-2017 category folders.
    """
    known_folders = set(CRESCI_LABEL_MAP.keys())

    # BFS up to depth 4
    queue = [search_root]
    visited = set()
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


# ---------------------------------------------------------------------------
# Save merged CSV
# ---------------------------------------------------------------------------

def save_merged_csv(texts: List[str], labels: List[int], output_path: Path) -> None:
    try:
        import pandas as pd
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        import pandas as pd

    df = pd.DataFrame({"text": texts, "label": labels})

    # Deduplicate identical (text, label) pairs
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    after = len(df)
    if before != after:
        log.info(f"Removed {before - after} duplicate rows. {after} unique samples remain.")

    # Shuffle for good measure
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and prepare Cresci-2017 for bot detection training."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Root directory for datasets/ folder. "
            "Defaults to two levels above this script (rf.v1.0.0/)."
        ),
    )
    parser.add_argument(
        "--method",
        choices=["auto", "hf", "zenodo"],
        default="auto",
        help=(
            "Download method. 'auto' tries HuggingFace first, then Zenodo. "
            "'hf' = HuggingFace only. 'zenodo' = Zenodo direct download only."
        ),
    )
    parser.add_argument(
        "--local_dir",
        default=None,
        help=(
            "Path to a locally extracted Cresci-2017 directory. "
            "Skips all downloads and processes the local files directly."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve output directory
    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        # Default: rf.v1.0.0/ (two levels above this script)
        base_dir = Path(__file__).resolve().parent.parent

    datasets_dir = base_dir / "datasets"
    raw_dir = datasets_dir / "cresci_2017_raw"
    output_csv = datasets_dir / "cresci_2017_merged.csv"

    log.info("=" * 60)
    log.info("Cresci-2017 Twitter Bot Detection — Data Downloader")
    log.info("=" * 60)
    log.info(f"Base directory : {base_dir}")
    log.info(f"Output CSV     : {output_csv}")
    log.info(f"Method         : {args.method}")

    texts: Optional[List[str]] = None
    labels: Optional[List[int]] = None

    # ── Path 0: local directory provided ────────────────────────────────────
    if args.local_dir:
        log.info(f"Processing local Cresci-2017 directory: {args.local_dir}")
        result = load_cresci_from_dir(Path(args.local_dir))
        if result:
            texts, labels = result
        else:
            log.error("Failed to load data from local directory.")
            sys.exit(1)

    # ── Path 1: HuggingFace ──────────────────────────────────────────────────
    elif args.method in ("auto", "hf"):
        log.info("Attempting HuggingFace datasets download...")
        result = try_hf_download()
        if result:
            texts, labels = result
            log.info(f"HuggingFace download succeeded: {len(texts)} samples")
        elif args.method == "hf":
            log.error("HuggingFace download failed and no fallback requested.")
            sys.exit(1)
        else:
            log.warning("HuggingFace download failed. Falling back to Zenodo...")

    # ── Path 2: Zenodo (fallback or explicit) ────────────────────────────────
    if texts is None and args.method in ("auto", "zenodo"):
        log.info("Attempting Zenodo direct download...")
        result = try_zenodo_download(raw_dir)
        if result:
            texts, labels = result
            log.info(f"Zenodo download succeeded: {len(texts)} samples")
        else:
            log.error(
                "Zenodo download also failed.\n"
                "You can manually download Cresci-2017 from:\n"
                "  https://zenodo.org/record/1482079\n"
                "Then run: python download_data.py --local_dir /path/to/extracted/cresci"
            )
            sys.exit(1)

    if texts is None or labels is None:
        log.error("No data was loaded. Exiting.")
        sys.exit(1)

    # ── Save ─────────────────────────────────────────────────────────────────
    save_merged_csv(texts, labels, output_csv)

    log.info("")
    log.info("Done! To train the model, run:")
    log.info(f"  python training-pipeline/train_ensemble.py --data_path {output_csv}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
