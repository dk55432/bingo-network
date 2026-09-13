# bingo-network — Roadmap

Prioritized work items, roughly in impact order. Triage notes from Sep 2026.

> About the "99%": that number is classifier accuracy on held-out cell
> crops, NOT end-to-end scan accuracy. The e2e ceiling is geometry
> (header detection, row/col lattice, sheet-bottom clamp, blur gating).

## 1. Game persistence + a stable address
**Both parts DONE Sep 2026.**

**Persistence.** `GameManager` was pure in-memory, so a restart/crash wiped
every game. Now durable state (games, players, cards + marks, called/current
numbers, status, winning pattern, waiting room) lives in SQLite
(`data/bingo.sqlite3`, gitignored):

- `game_store.GameStore` — whole-world snapshot (`save_all`, full-table
  replace so deletions propagate) + single-game `upsert`.
- `game.py` — `Game.to_persistable()` / `GameManager.snapshot()` /
  `Game.from_persisted()` / `GameManager.restore_from()`. Connection state
  (websockets, connected flags) is never persisted; players re-attach by
  stable `player_id` via the existing `reconnect` flow.
- `main.py` — restore on startup (lifespan), crash-proof immediate writes
  on `create_game` and `submit_number`, plus an unconditional 5s autosave
  (a hall's game set is tiny; no mutation-tracking to miss a code path).
- Path overridable with `BINGO_STATE_DB` env var.
- `test_persistence.py` (fast CI job, no torch): snapshot→store→restore
  round-trip including win detection on a restored board.

**Stable address.** Tailscale Funnel on the Orange Pi gives a permanent
public HTTPS URL that never depends on the LAN IP:

    https://orangepizero3.tail3fad8c.ts.net/

No domain buy, no CA certs, no open router ports — QR code links stay valid
forever (links are built from `window.location.origin`, so no code change).
Full setup/ops/troubleshooting notes live in NOTES.md ("Tailscale Funnel…").

## 2. Scanner regression tests + CI  (DONE Sep 2026)
- `test_scan_regression.py` replays 14 committed normalized dumps
  (`01_CNN_refactor/regression_dumps/`) through `cnn_reader._read_sheet`
  against a golden manifest: card counts, partial_sheet, error presence,
  and exact per-cell grids.
- `.github/workflows/scanner-ci.yml` runs it on ubuntu (CPU torch).
- Regenerate the golden deliberately (`python test_scan_regression.py
  --rewrite-manifest`) only after an intentional improvement; a reader
  change that shifts detection/decode fails CI.
- Note: `sample_bingo_cards/` is historically tracked but is out-of-domain
  for the CNN reader (8.5% baseline there) — not a usable e2e set. The
  strong e2e data (`phone_sheets/`) is gitignored/100+MB.

## 3. Retrain loop automation + eval gate
- Retention: `train_learning.py` fine-tunes on confirmed cells; a
  `MIN_SCAN_TS` provenance gate keeps stale/mis-segmented cells out.
- Ship gate: evaluate candidate `.pth` vs the current one on the valid
  split and refuse to deploy on regression (the last retrain shipped
  without this).
- Record the git SHA next to each confirmed cell (self-auditing corpus).
- Revisit correction-tagging so retraining can focus on user-corrected
  cells rather than all confirms.

## 4. Legacy cleanup
`parse_bingo_sheet.py` (2518 lines), `pipeline.py`, `00_card_scan_refactor/`,
`defunct_tests/`, `recognize_cards.py`, tesseract-era eval scripts are dead
weight vs the CNN reader. Removes confusion, 23 TODO/FIXMEs.

## 5. Accessibility (the stated mission)
Large high-contrast current-number display, vibration on new call,
screen-reader semantics, color-blind-safe marking, "unmarked numbers
remaining" per card.

## 6. Game-day observability
A `/health` (or minimal admin view): model version + retrain date, scan
success/failure counts, game/player counts — so a hall-day problem is
diagnosable in seconds.

## Environment / ops notes (Orange Pi, 192.168.1.175)
- Server = systemd `fastapi-bingo.service` with a repo drop-in
  (`deploy/fastapi-bingo.service.d/override.conf`) setting `READER=cnn`.
- Pi pulls via a passphrase-less read-only repo deploy key
  (`~/.ssh/bingo_deploy`, pinned in `~/.ssh/config`); the interactive
  account key stays passphrase-locked behind the agent.
- Only the Mac can retrain right now (has `phone_sheets/`); ship the new
  `.pth` through git and restart the service on the Pi.