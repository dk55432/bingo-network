"""Torch-free tests for the confirmed-cell learning-audit rules.

These rules (filename provenance, correction tagging, staleness gating,
correction diffing) are correctness-critical and deliberately live in a
dependency-free module so they can be tested in the fast CI job without
torch / cv2 / the FastAPI app.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "01_CNN_refactor"))

from learning_audit import (  # noqa: E402
    CORRECTED_TAG,
    cell_was_corrected,
    dump_auto_json,
    is_corrected_name,
    load_auto_json,
    repo_short_sha,
    save_cell_name,
    scan_epoch_of,
)


def test_scan_epoch_of_parses_new_and_legacy_names():
    assert scan_epoch_of("1789233548_c2_r3c0.jpg") == 1789233548
    assert scan_epoch_of("1789233548_59bc940_c1_r0c2_x.jpg") == 1789233548
    assert scan_epoch_of("notnumeric.jpg") is None


def test_is_corrected_name_only_tags_only_tagged():
    assert is_corrected_name("1789233548_59bc940_c1_r0c2_x.jpg")
    assert not is_corrected_name("1789233548_59bc940_c1_r0c2.jpg")
    assert not is_corrected_name("legacy_1789233548_c2_r3c0.jpg")
    # '_x' elsewhere (never produced by save_cell_name) must not count
    assert not is_corrected_name("1789233548_xa1c2_c1_r0c0.jpg")


def test_save_cell_name_embeds_scan_sha_and_tag():
    plain = save_cell_name(1789233548, "59bc940", 2, 3, 0, False)
    assert plain == "1789233548_59bc940_c2_r3c0.jpg"
    corr = save_cell_name(1789233548, "59bc940", 2, 3, 0, True)
    assert corr.endswith(f"c2_r3c0{CORRECTED_TAG}.jpg")
    # first token stays the scan epoch so the staleness gate still works
    assert scan_epoch_of(corr) == 1789233548


def test_cell_was_corrected_rules():
    assert cell_was_corrected(17, 18)          # user changed the number
    assert cell_was_corrected(None, 42)        # reader blanked it, user set it
    assert not cell_was_corrected(42, 42)      # pure confirmation
    assert not cell_was_corrected(17, None)    # free/unspecified never counts


def test_auto_json_roundtrip_and_best_effort():
    grids = [[[17, 42]], [[1]]]
    assert dump_auto_json("/tmp/auto_json_test.json", grids)
    assert load_auto_json("/tmp/auto_json_test.json") == grids
    # missing/bogus files degrade to None, never raise
    assert load_auto_json("/tmp/definitely_missing_auto.json") is None
    import os
    with open("/tmp/auto_json_test.json", "w") as fh:
        fh.write("not json")
    assert load_auto_json("/tmp/auto_json_test.json") is None
    os.remove("/tmp/auto_json_test.json")


def test_repo_short_sha_returns_string():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    sha = repo_short_sha(here)
    assert isinstance(sha, str) and len(sha) in (7, len("nogit"))