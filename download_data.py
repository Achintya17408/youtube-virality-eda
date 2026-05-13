"""
[OPTIONAL] Download YouTube-8M TFRecord shards to a local cache.

youtube_eda.py streams data DIRECTLY from Google Cloud Storage by default,
so this script is NOT required to run the EDA.  Use it only if you want
faster subsequent runs by caching shards on local disk.

The YouTube-8M dataset is publicly hosted by Google at:
  gs://youtube8m-ml/2/video/train/train{XX}.tfrecord

File naming: base-62 two-character suffix (0-9, A-Z, a-z) → 3,844 total files
             ~944 videos per file · ~3.6 M videos total

Authentication: None required — bucket is publicly readable.

Usage
-----
  python download_data.py                 # default: 1,100 files (~1 M videos)
  python download_data.py --num_files 200 # smaller sample, ~190 K videos
  python download_data.py --workers 4     # reduce parallelism on slow connections

Storage  : ~4.8 MB per file → 1,100 files ≈ 5.3 GB  (need ~6 GB free)
Time est.: ~14 min at 50 Mbps with default 8 parallel workers

Note: If disk space is tight, simply run youtube_eda.py without downloading —
it will stream all 1 M+ videos from GCS at ~5 MB/s per shard.
"""

import argparse
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Base-62 alphabet that Google uses to name the shards
_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
ALL_SHARD_NAMES = [f"train{a}{b}" for a in _CHARS for b in _CHARS]  # 3,844


def _download_one(client_factory, name: str) -> tuple[str, str]:
    dest = os.path.join(DATA_DIR, f"{name}.tfrecord")
    if os.path.exists(dest):
        return name, "skip"
    bucket = client_factory().bucket("youtube8m-ml")
    blob = bucket.blob(f"2/video/train/{name}.tfrecord")
    blob.download_to_filename(dest)
    return name, "ok"


def main():
    parser = argparse.ArgumentParser(description="Download YouTube-8M TFRecord files")
    parser.add_argument("--num_files", type=int, default=1_100,
                        help="Number of shard files to download (default: 1100 ≈ 1 M videos)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel download threads (default: 8)")
    args = parser.parse_args()

    try:
        from google.cloud import storage
    except ImportError:
        raise SystemExit(
            "google-cloud-storage not installed.\n"
            "Run: pip install google-cloud-storage"
        )

    def client_factory():
        return storage.Client.create_anonymous_client()

    already = {
        f.replace(".tfrecord", "")
        for f in os.listdir(DATA_DIR)
        if f.endswith(".tfrecord") and f.startswith("train")
    }
    print(f"Already downloaded : {len(already):,} files")

    need = [n for n in ALL_SHARD_NAMES[:args.num_files] if n not in already]
    if not need:
        print(f"Nothing to do — already have {len(already):,} ≥ {args.num_files:,} files.")
        return

    print(f"Target             : {args.num_files:,} files")
    print(f"Need to download   : {len(need):,} files")
    print(f"Estimated size     : {len(need)*4.8:.0f} MB")
    print(f"Parallel workers   : {args.workers}")
    print()

    done = 0
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_download_one, client_factory, n): n for n in need}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                _, status = fut.result()
            except Exception as e:
                status = "error"
                errors.append((name, str(e)))
            done += 1
            if done % 100 == 0 or done == len(need):
                print(f"  [{done:5d}/{len(need)}] downloaded  (last: {name}  {status})")

    total = len(already) + done - len(errors)
    print(f"\nFinished. Total train files in data/: {total:,}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for n, e in errors[:5]:
            print(f"  {n}: {e}")


if __name__ == "__main__":
    main()
