#!/usr/bin/env python3
"""
download_lichess_data.py

Downloads a set of monthly Lichess standard-rated game archives (.pgn.zst),
with:
    - resumable downloads (safe to re-run after a dropped connection --
      picks up where it left off instead of restarting a 30GB file from zero)
    - SHA256 checksum verification against Lichess's published sums, so a
      silently-corrupted download doesn't waste hours of downstream processing
    - skip-if-already-verified, so re-running the script after it's finished
      (or partially finished) is fast and safe

Usage:
    # the corpora behind the results table:
    python -m src.download_lichess_data --corpus 1.2m --output-dir data
    python -m src.download_lichess_data --corpus 98k --output-dir data

    # an explicit set of months:
    python -m src.download_lichess_data --months 2025-01,2025-02 --output-dir data

    # skip checksum verification (faster, less safe):
    python -m src.download_lichess_data --output-dir data --skip-checksum
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

# 2023-01 alone is the `98k` corpus; the full 18-month set is the `1.2m`
# corpus, ~510GB compressed.
CORPORA = {
    "98k": ["2023-01"],
    "1.2m": [
        "2023-01", "2025-01", "2025-02", "2025-03", "2025-04", "2025-05",
        "2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11",
        "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05",
    ],
}

DEFAULT_MONTHS = CORPORA["1.2m"]

def filename_for_month(month: str) -> str:
    return f"lichess_db_standard_rated_{month}.pgn.zst"


def fetch_checksums(base_url: str) -> dict:
    """
    Fetches and parses Lichess's sha256sums.txt (standard `sha256sum`
    output format: "<64-hex-char-hash>  <filename>" per line).
    Returns {} and prints a warning if it can't be fetched or parsed --
    verification is then skipped gracefully rather than blocking downloads.
    """
    url = f"{base_url}/sha256sums.txt"
    try:
        with urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except URLError as e:
        print(f"  WARNING: couldn't fetch checksums from {url}: {e}")
        print("  Proceeding WITHOUT checksum verification.")
        return {}

    checksums = {}
    for line in text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        hash_hex, fname = parts
        fname = fname.lstrip("*")  # some sha256sum output marks binary mode with a leading *
        if len(hash_hex) == 64:
            checksums[fname] = hash_hex.lower()

    if not checksums:
        print(f"  WARNING: fetched {url} but found no parseable checksum lines.")
        print("  Proceeding WITHOUT checksum verification.")

    return checksums


def sha256_of_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_one(url: str, dest: Path, retries: int) -> bool:
    """Runs wget -c (resumable) against the given URL. Returns True on success."""
    cmd = [
        "wget",
        "-c", # continue/resume partial downloads
        "--tries", str(retries),
        "--timeout", "30",
        "--retry-connrefused",
        "-O", str(dest),
        url,
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Download Lichess monthly game archives.")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--corpus", choices=sorted(CORPORA), default=None,
                         help="Named corpus from the results table: 98k (one month) "
                              "or 1.2m (18 months, ~510GB). Mutually exclusive with --months.")
    parser.add_argument("--months", default=None,
                         help="Comma-separated list, e.g. 2025-01,2025-02. "
                              "Defaults to the 1.2m corpus (18 months, ~510GB).")
    parser.add_argument("--base-url", default="https://database.lichess.org/standard",
                         help="Override for testing against a different host.")
    parser.add_argument("--skip-checksum", action="store_true",
                         help="Skip SHA256 verification (faster, less safe).")
    parser.add_argument("--redownload-on-failed-checksum", action="store_true",
                         help="If a checksum fails, delete the file and retry once "
                              "automatically. Default: just warn and leave it for you "
                              "to handle, since deleting a 30GB file is a big action "
                              "to take without asking.")
    parser.add_argument("--retries", type=int, default=20,
                         help="wget retry count per file for transient network failures.")
    args = parser.parse_args()
    
    if args.corpus and args.months:
        sys.exit("Pass either --corpus or --months, not both.")
    if args.corpus:
        months = CORPORA[args.corpus]
    elif args.months:
        months = [m.strip() for m in args.months.split(",")]
    else:
        months = DEFAULT_MONTHS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(months)} month(s) to {output_dir.resolve()}")
    print()

    checksums = {} if args.skip_checksum else fetch_checksums(args.base_url)
    print()

    results = []  # (month, status) where status in {"ok", "checksum_fail", "download_fail"}

    for i, month in enumerate(months, start=1):
        fname = filename_for_month(month)
        url = f"{args.base_url}/{fname}"
        dest = output_dir / fname

        print(f"[{i}/{len(months)}] {fname}")

        expected_hash = checksums.get(fname)

        # Fast path: already downloaded and already verified correct -- skip entirely.
        if dest.exists() and expected_hash:
            print("  already exists -- verifying checksum before deciding whether to re-fetch...")
            actual_hash = sha256_of_file(dest)
            if actual_hash == expected_hash:
                print("  OK (already complete and verified) -- skipping")
                results.append((month, "ok"))
                print()
                continue
            else:
                print("  checksum MISMATCH on existing file -- will re-download")

        ok = download_one(url, dest, args.retries)
        if not ok:
            print(f"  DOWNLOAD FAILED for {fname} -- will need to be retried")
            results.append((month, "download_fail"))
            print()
            continue

        if expected_hash:
            print("  verifying checksum...")
            actual_hash = sha256_of_file(dest)
            if actual_hash == expected_hash:
                print("  checksum OK")
                results.append((month, "ok"))
            else:
                print(f"  CHECKSUM MISMATCH")
                print(f"    expected: {expected_hash}")
                print(f"    actual:   {actual_hash}")
                if args.redownload_on_failed_checksum:
                    print("  deleting and retrying once...")
                    dest.unlink()
                    ok2 = download_one(url, dest, args.retries)
                    if ok2 and sha256_of_file(dest) == expected_hash:
                        print("  retry succeeded, checksum OK")
                        results.append((month, "ok"))
                    else:
                        print("  retry FAILED -- file is still bad")
                        results.append((month, "checksum_fail"))
                else:
                    print(f"  Leaving the file in place. To retry: delete {dest} and re-run this script.")
                    results.append((month, "checksum_fail"))
        else:
            results.append((month, "ok"))  # no checksum available, trust the download

        print()

    # ---- summary ----
    print("=" * 60)
    ok_count = sum(1 for _, s in results if s == "ok")
    print(f"Done: {ok_count}/{len(months)} files OK")
    for month, status in results:
        if status != "ok":
            print(f"  {month}: {status}")
    print("=" * 60)

    if ok_count < len(months):
        print("\nRe-run this exact command to retry only the failed/missing files --")
        print("already-verified files will be skipped automatically.")
        sys.exit(1)


if __name__ == "__main__":
    main()
