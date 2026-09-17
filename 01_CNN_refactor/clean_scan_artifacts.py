"""Purge un-accepted scan artifacts from /tmp.

Lifecycle recap (see bingo_scan.py and cnn_reader.py):
  * Every successful /scan-card write grayscale cell crops + auto.json to
    /tmp/phone_learning_pending/<scan_id>/ (cnn_reader._persist_pending_cells).
  * When you SAVE a reviewed scan (/cards), _save_learning_cells copies the
    labeled crops into 01_CNN_refactor/learning_cells/<number>/ AND deletes
    the pending dir — so accepted cells live only in learning_cells.
  * When you hit "Cancel — discard this scan", /scan/reject deletes that
    scan's pending dir immediately.
  * So anything still in /tmp/phone_learning_pending was NEVER accepted and
    is safe to delete; it can never enter the retraining corpus.

Usage:
    python3 clean_scan_artifacts.py [--min-age DAYS] [--debug] [--dry-run]

    --min-age DAYS  only purge pending scans older than this many days
                    (0 = all). Also applies to the debug dump age.
    --debug         additionally clear /tmp/cnn_reader_debug (in_/overlay_/
                    orig_ dumps: pure debug, no training value).
    --dry-run       print what would be removed without deleting.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

PENDING_DIR = Path("/tmp/phone_learning_pending")
DEBUG_DIR = Path("/tmp/cnn_reader_debug")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, add_help=False)
    ap.add_argument("--min-age", type=float, default=0.0,
                    help="only purge entries older than this many days (default 0 = all)")
    ap.add_argument("--debug", action="store_true",
                    help="also clear /tmp/cnn_reader_debug")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cutoff = time.time() - args.min_age * 86400
    dry = args.dry_run
    removed = kept = 0

    if PENDING_DIR.is_dir():
        for entry in sorted(PENDING_DIR.iterdir()):
            if not entry.is_dir():
                continue
            if args.min_age and entry.stat().st_mtime > cutoff:
                kept += 1
                continue
            if dry:
                print(f"would remove pending scan {entry.name}")
            else:
                shutil.rmtree(entry, ignore_errors=True)
            removed += 1

    if args.debug and DEBUG_DIR.is_dir():
        if dry:
            print(f"would clear {DEBUG_DIR}")
        else:
            shutil.rmtree(DEBUG_DIR, ignore_errors=True)

    tag = " (dry run)" if dry else ""
    print(f"{removed} pending scan(s) removed, {kept} kept within --min-age{tag}")
    if args.debug:
        print(f"debug dir {'would be' if dry else ''} cleared")
    return 0


if __name__ == "__main__":
    sys.exit(main())