# CNN Phone-Card Reader — Capture Doc

This file is the durable record for a fresh agent (or a fresh session) to get up
to speed on the machine-learning card-reading pipeline in `01_CNN_refactor/`
without the original author's memory. Read this first; the rest of the repo
(code + this file) is the source of truth.

Last updated: 2026-08-31.

## What the pipeline does

`bingo_scan.py` exposes the scan API. With `READER=cnn`, a photo of a full bingo
sheet (typically 3 cards stacked vertically) is normalized, card header bands
are auto-detected, each card is divided into a 5x5 grid of cells, and each cell
is classified by a small CNN to a bingo number (1..75). Results are shown for
human review, and confirmed cells are saved to a `learning_cells/` corpus for
periodic retraining.

## The retrain loop (how to retrain)

Run from `01_CNN_refactor/`:

```bash
/Users/davidkohn/Downloads/bingo-network/.venv/bin/python train_learning.py
```

`train_learning.py` does, in order:

1. `backup_learning()` — tars `learning_cells/` into `learning_backups/learning_<ts>.tar.gz` (keeps last 20). This is the confirmed-cell corpus snapshot.
2. `build()` (from `train_phone_cells.py`) — rebuilds the base whole-cell dataset into `/tmp/phone_cells_v2` from the phone photos:
   - Batch A: `phone_sheets/` (sheets 1..30)
   - Batch B: `phone_sheets3/` **minus sheet 1** (bad frame)
   - SPLITS: sheets 1..24 → train, 25..30 → valid
   - Each sheet's cells are extracted via `sheet_to_cells_teal()` and cross-referenced against ground-truth `parse_truth()`; blank/no-truth cells are dropped.
3. `merge_learning()` — copies every confirmed cell from `learning_cells/<number>/*.jpg` into `/tmp/phone_cells_v2/train/<number>/`. Filenames embed the scan id, so re-runs are idempotent.
4. Fine-tunes `cell_classifier_phone.pth` with **early stopping**: lr 1e-4
   (Adam, StepLR step=10 gamma=0.5, batch=32), up to **30 epochs**, tracking the
   best valid accuracy. Stops early when valid accuracy hasn't improved for
   `PATIENCE` (6) consecutive epochs, then restores and saves the best-epoch
   weights (so a long run never overfits — the epoch count is self-tuning; the
   plateau is typically around epoch 15-20 for the current corpus size).
5. Saves the new checkpoint as `cell_classifier_phone.pth`, old one → `.pth.bak`.

Verification output is a `classification_report` on the valid split. Last retrain
was 917 → 2175 confirmed cells.

## How confirmed cells are produced (the learning loop)

1. `POST /scan-card` (CNN path) writes each cell crop to `/tmp/phone_learning_pending/<scan_id>/` and returns `scan_id` (and `debug.ts`).
2. `templates/scan.html` stores `lastScanId`.
3. When the user confirms, `POST /cards {game_id, player_id, grids, scan_id}` → `_save_learning_cells` pairs grid i (card id = i+1) with the pending crops, writing `learning_cells/<number>/<scan_id>_c<cid>_r<r>c<c>.jpg`. Best-effort — failures are non-fatal.

## Gray / washed-out sheets — the 3-tap assist

Some photos (pale gray sheets photographed small/under-lit) have no detectable
header color and auto-scan fails. Rather than read the desk as cards, the reader
raises an actionable error and the UI offers **manual assisted scan**:

- `POST /scan-assist` re-processes the original photo with the user's taps.
- The user taps (or drags) the **top edge of each of the 3 gray bars** in `scan.html`. Taps are % coords, converted via `assistImg.naturalHeight`.
- `POST /scan-assist` takes `band_tops` (pixel rows, 3 taps) → `cells_from_forced()` pins the card geometry. **Requires exactly 3 readable cards** (1 or 2 → HTTP 422 "assisted scan couldn't verify all 3 cards").

So a gray sheet is always scanned via the assist path; the auto path only flags it
(`debug.partial_sheet` when <3 cards detected) and steers the user there.

## Grid alignment (critical detail)

Cell row boundaries are found by "snapping" equal division toward printed grid
lines. This was a real bug: the original snap used a full-image-width mean
(`(255 - gray).mean(axis=1)`), which **dilutes the thin dark grid lines into
noise** (a 2px line at brightness ~30 averaged over ~950 columns of white cells).
The result was uneven rows and a systematic "the number I want is one cell below
where I'm looking" error.

The fix (`photo_sheet_cells.py`): a per-card **brightness-dip** signal computed in
a narrow vertical strip (12px wide, centered on the card). Grid lines are clearly
visible in a narrow strip, and the dip score (how much darker a row is than its
5-row neighbors) peaks sharply at each line. The snap now finds real grid lines.

Geometery keys:
- `_grid_dip(gray, x0, x1, top, bottom, strip_w=12)` — the dip signal.
- `_snap(bounds, signal, lo, hi, radius=40)` — radius 40 (larger overshoots to adjacent card boundaries).

## Model / decode mapping

- `CellClassifier` (in `train_phone_cells.py`) is a small CNN; 76 output classes (index 0 unused).
- `cnn_reader.` maps logits to numbers via `IDX_TO_NUM` (lexicographically sorted class order — index i = i-th smallest label).
- `decode_card` does a per-column **linear_sum_assignment** between the 5 non-FREE cells and the 15 legal numbers (COLUMN_RANGES: col→(1-15),(16-30),(31-45),(46-60),(61-75)), maximizing total log-probability, no repeats.
- FREE cell is (2,2) — exempt from decoding.
- `needs_review` is set when mean conf < 0.25 or min conf < 0.08. The review gate is the correctness backstop (model has inherent ~0.66 valid acc, hypersensitive to ±5px crop shifts).

## Files that matter

- `cnn_reader.py` — `read_sheet_bytes`, `_read_sheet`, `cells_from`, `cells_from_forced`, `cell_logits`, `decode_card`, `_persist_pending_cells`.
- `photo_sheet_cells.py` — `teal_card_bboxes`, `_grid_dip`, `_snap`, `_card_y_extent`, `_frame_bright_extent`, `sheet_to_cells_teal`, `sheet_to_cells_forced`, `forced_card_bboxes`.
- `extract_scan_cells.py` — `_BAND_FAMILIES`, `_header_hue_spec`, `_sheet_structure`, `sheet_to_cells`.
- `train_learning.py`, `train_phone_cells.py`.
- `bingo_scan.py` — `POST /scan-card`, `POST /scan-assist`, `GET /scan-debug/{ts}.png`, `GET /scan-debug-in/{ts}.png`, `POST /cards` (confirm), `_save_learning_cells`.
- `templates/scan.html` — assist panel, `handleScanSuccess`, `partial_sheet` steering.

## Debug dumps (regression set)

`_read_sheet` dumps the normalized input + a green-cell-box overlay to
`/tmp/cnn_reader_debug/in_<ts>.png` / `overlay_<ts>.png`. To test a reader change,
re-run `cnn_reader._read_sheet()` over the dump set and verify card counts don't
regress (24 dumps: gray working / gray fail / gray uneven / 1-card / 2-card; pink;
orange; green). `GET /scan-debug-in/{ts}.png` serves the tap image for the assist.

This check is now automated in CI: a curated 14-dump golden corpus lives in
`regression_dumps/` (committed) and is replayed by `test_scan_regression.py`,
comparing card counts, partial_sheet, error presence, and exact per-cell grids
against `regression_dumps/manifest.json`. Regenerate the golden only after an
intentional improvement: `python test_scan_regression.py --rewrite-manifest`.

## Holdover / known limitations

- Sheet-bottom clamp on bright tables (making the last card's region never extend off the sheet) is unsolved — ink-fraction alone doesn't stop at the sheet edge.
- First/last row of each card are structurally smaller/larger (band detection doesn't perfectly align with the grid); the 3 middle rows are now well-aligned.
- CPU-only training is slow; a dedicated machine (e.g. Raspberry Pi) is the intended host.

## Migration to a new host (e.g. Raspberry Pi)

> **Status: the migration is DONE — the Raspberry Pi (`pi@slice-of-pi`,
> `~/bingo-network/`) is now the canonical training host.** The Mac's copies of
> `cell_classifier_phone.pth` and `learning_cells/` are STALE relative to the
> Pi's — treat the Pi as the source of truth for the model + corpus. The Pi also
> has the current `cell_classifier_phone.pth` checked in to git. The Pi's
> `requirements.txt` was at one point overwritten with a system-wide
> `pip freeze > requirements.txt` (insider the venv mistake) — that has since
> been replaced with a clean direct-deps manifest (see the Python deps section
> below).

In git (portable via `git clone`/`git fetch`): all code + `cell_classifier_phone.pth` +
`scan_card_numbers.txt` (ground truth). After a clone + `pip install`, the scanner
runs immediately — the trained `.pth` is enough for inference.

**NOT in git (gitignored) — must be copied manually to retrain on the new host:**
- `phone_sheets/`  (30 photos, batch A)
- `phone_sheets3/` (30 photos, batch B; sheet 1 excluded by the retrainer)
- `learning_cells/` (the confirmed-cell corpus — grow it on the new host)

Safe to leave behind: `phone_sheets2/` (not used by `train_phone_cells.build()`),
`learning_backups/` (older corpus tarballs, only for rollback), `scans/` (old
tesseract-era output, unused by the CNN server).

Copy from the old host (this Mac) to the Pi (`pi@<ip>`, path `~/bingo-network/`):

```bash
ssh pi@<ip> 'mkdir -p ~/bingo-network/01_CNN_refactor'
# from the old host's 01_CNN_refactor/:
tar -C . -czf - phone_sheets phone_sheets3 learning_cells \
  | ssh pi@<ip> 'mkdir -p ~/bingo-network/01_CNN_refactor && tar -C ~/bingo-network/01_CNN_refactor -xzf -'
```

Verify:
```bash
ssh pi@<ip> 'for d in phone_sheets phone_sheets3 learning_cells; do
  echo "$d: $(ls ~/bingo-network/01_CNN_refactor/$d | wc -l) entries"; done'
# expect 30 / 30 / 75
```

Set up Python deps on the new host (the repo `.venv` is not portable).
`requirements.txt` at the repo root is a **clean direct-deps manifest** (hand
curated — fastapi, uvicorn, python-multipart, pydantic, jinja2, websockets,
numpy, pillow, opencv-python-headless, pytesseract, scikit-learn, scipy).

The CNN reader and retrainer additionally need **torch + torchvision**, which are
NOT in `requirements.txt` (they're CPU-vs-CUDA-sensitive and pulled from a
dedicated wheel index). Install the small CPU build explicitly:

```bash
cd ~/bingo-network
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Venv hygiene (learned the hard way on the Pi):** always install/run through the
venv — use `.venv/bin/pip`, `.venv/bin/python`, and `.venv/bin/uvicorn` (or
`.venv/bin/activate`). Do NOT run bare `pip`/`python -m pip`/`pip freeze`, which
hit the system Python and both pollute the whole OS with project packages AND
dump Debian system packages into `requirements.txt` (that happened once; the
resulting 323-line freeze was replaced with the clean manifest above). If you
ever blunder the venv again, rebuild it fresh:
`rm -rf .venv && python3 -m venv .venv && .venv/bin/pip install -r
requirements.txt` (+ the torch line) and re-copy the three gitignored data dirs.

Without torch installed, `READER=cnn` will fail at import — the tesseract
(`default`) reader works without it.

Run the server on the new host with READER=cnn:

```bash
cd ~/bingo-network
READER=cnn .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

Retrain on the new host (confirm you copied the three dirs first):

```bash
cd ~/bingo-network/01_CNN_refactor
../.venv/bin/python train_learning.py
```
