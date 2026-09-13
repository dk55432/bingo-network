"""Pure helpers for the confirmed-cell learning loop (no heavy deps).

Kept dependency-free (stdlib only) so the correctness-critical naming /
provenance / correction-diff rules are unit-testable in the fast CI job
without torch.  The scanning server (bingo_scan.py) and the retrain script
(train_learning.py) import these so the audit rules live in exactly one
place.

Confirmed cell filenames are the corpus's self-audit record:
    <scan_id>_<git_sha>_c<cid>_r<r>c<c>.jpg        confirmed as scanned
    <scan_id>_<git_sha>_c<cid>_r<r>c<c>_x.jpg      user CORRECTED the value

- the first token is the scan's unix-epoch timestamp, so the provenance
  gate (MIN_SCAN_TS) and re-run idempotency still work on the new names,
- the embedded git short SHA says WHICH code version produced the crop,
  so a corpus harvested under a buggy reader can be audited/excluded.
"""

import json
import subprocess

# A corrected crop gets a trailing "_x" so train-time weighting and ad-hoc
# audits can distinguish "user actually changed the number" from
# "confirmed as scanned" without re-parsing grids.
CORRECTED_TAG = "_x"


def scan_epoch_of(cell_name: str):
    """Scan unix-epoch embedded at the start of a confirmed-cell filename,
    or None if the name doesn't start with a numeric scan id."""
    first = cell_name.split("_", 1)[0]
    return int(first) if first.isdigit() else None


def is_corrected_name(cell_name: str) -> bool:
    """True when this confirmed cell's number was corrected by the user
    (filename ends in the "_x" tag; other "_x" substrings — e.g. inside a
    future hash token — don't count)."""
    stem = cell_name.rsplit(".", 1)[0]
    return stem.endswith(CORRECTED_TAG)


def save_cell_name(scan_id, sha, cid, r, c, corrected: bool) -> str:
    """Confirmed-cell filename carrying scan epoch + reader git SHA (+ a
    correction tag when the user changed the number)."""
    return "{0}_{1}_c{2}_r{3}c{4}{5}.jpg".format(
        scan_id, sha, cid, r, c, CORRECTED_TAG if corrected else "")


def cell_was_corrected(auto_value, confirmed_value) -> bool:
    """True when the user's final number differs from what the reader
    auto-recognized at scan time (auto None -> a value supplied counts as a
    correction; equal values are a pure confirmation; free space/None is
    never a correction)."""
    if confirmed_value is None:
        return False
    if auto_value is None:
        return True
    return auto_value != confirmed_value


def repo_short_sha(repo_dir):
    """Current repo HEAD short SHA, or 'nogit' if unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "nogit"


def dump_auto_json(path, auto_grids) -> bool:
    """Write the scan-time auto grid values next to the pending crops so a
    later /cards confirmation can diff them.  Best-effort; returns False on
    failure (callers degrade to 'confirmed as scanned' tagging)."""
    try:
        with open(path, "w") as fh:
            json.dump(auto_grids, fh)
        return True
    except Exception:
        return False


def load_auto_json(path):
    """Read the auto grids written at scan time; None when missing/bogus."""
    try:
        with open(path) as fh:
            grids = json.load(fh)
        if isinstance(grids, list):
            return grids
    except Exception:
        pass
    return None