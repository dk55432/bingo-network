# Bingo Buddy — Real-time bingo card tracking with phone scanning

A real-time multiplayer bingo app: a host runs games from any browser, players
join on their phones, and everyone's cards are tracked automatically as numbers
are called. A printed card is photographed with the phone's camera, OCR'd, and
entered into the game — no manual data entry, no special equipment.

The project's primary motivation is **accessibility**: bingo halls often depend
on hearing the caller announce numbers. This app can read cards out loud, mark
them automatically, and vibrate/announce on new calls, so players who are deaf,
hard-of-hearing, or low-vision can follow the game on their own terms.

## Features

- **Real-time multiplayer** over WebSockets (FastAPI backend, browser clients)
- **Host console** — create games, set winning patterns, broadcast numbers,
  announce winners
- **Automatic card scanning** — photograph a bingo sheet (single cards or a
  stacked strip), get an OCR'd card back in seconds
  - CNN-based whole-card digit reader (`READER=cnn`), with a Tesseract
    fallback and a SIFT/homography-assisted path
  - Review-and-correct UI before a card enters play (strict column-range
    validation catches OCR mistakes)
  - Corrected cells feed a self-auditing training corpus for future retrains
- **Winning-pattern engine** — line, blackout, postage stamp, six-pack, and any
  arbitrary mask pattern, composed via `patterns_config.txt`
- **Game persistence** — SQLite-backed state survives restarts and crashes
  (`data/bingo.sqlite3`, gitignored)
- **Waiting room** — players who join mid-game are queued and auto-promoted
  when the next game starts
- **Reconnect support** — host and players survive browser reloads and network
  blips
- **Accessibility** — large high-contrast current-number display, screen-reader
  semantics, color-blind-safe card marking, audible chimes and vibration on new
  calls, "N numbers left per card" readout
- **No accounts, no database setup** — self-host on any always-on box (tested on
  a Raspberry Pi / Orange Pi and macOS) and hand out a QR code

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-cnn.txt   # core app + CNN reader
READER=cnn uvicorn main:app --reload --host 0.0.0.0
```

`READER=cnn` selects the recommended CNN scanner (see *Choosing the card
reader* below). If you skip it, the app falls back to the legacy single-card
Tesseract path — which also needs the `tesseract` binary installed:

```bash
# Debian/Ubuntu (incl. the Orange Pi/Raspberry Pi)
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

Open the landing page at `http://<host-ip>:8000/`:

- **Host a game:** `/create_game` — new game page, QR code, and a join link
- **Host console:** `/host`
- **Player view:** the join link from the host page (or `/join`)
- **Scan a card:** `/scan`

Run the test suite:

```bash
.venv/bin/python -m pytest
```

## Choosing the card reader

The card-scanner endpoint (`POST /scan-card`) selects its reader via the
`READER` environment variable:

| `READER` | Pipeline |
| -------- | -------- |
| `cnn` (recommended) | EXIF-correct normalization, teal-band geometry, whole-cell CNN digit model with column-constrained decode (~64% cell-accurate on held-out phone photos vs ~23% for Tesseract) |
| `homography` | SIFT + homography header detection, per-card column projection |
| `tesseract` (default) | Legacy contour/corner → Hough grid → Tesseract path |

The CNN model weights (`01_CNN_refactor/cell_classifier_phone.pth`) and header
templates are checked in. No training is needed to use it.

## Architecture overview

```
Browser clients (host + N players)
        │  WebSocket (/ws)  +  REST (scan API, templates)
        ▼
main.py ── FastAPI app, connection manager, game lifecycle handlers
   │
   ├── game.py          Game / Player / GameManager state machine + win checking
   ├── game_store.py    SQLite persistence (snapshot + per-game upsert)
   ├── bingo_scan.py    POST /scan-card, /scan-assist, /cards (attach scanned card)
   ├── patterns_parser  /patterns_config.txt → winning-pattern objects
   ├── pipeline.py      tesseract-era CV pipeline (contour, warp, grid slice, OCR)
   └── 01_CNN_refactor/ CNN card reader (cnn_reader.py), training + eval tooling
```

Connection state is never persisted — players re-attach to a live game by
stable `player_id` over the existing reconnect flow. All durable game state
(games, players, cards + marks, called numbers, winning pattern) lives in
SQLite and is snapshotted on every meaningful mutation plus a 5-second autosave.

## Project layout

| Path | Contents |
| ---- | -------- |
| `main.py` | FastAPI app, WebSocket endpoint, route handlers |
| `game*.py`, `player.py`, `bingo_card*.py` | Game state + rules |
| `winning_pattern.py`, `patterns_parser.py`, `pattern_helpers.py`, `patterns_config.txt` | Winning-pattern engine |
| `bingo_scan.py`, `pipeline.py` | Card-scanning API + CV pipeline |
| `connection_manager.py` | WebSocket connection bookkeeping |
| `01_CNN_refactor/` | CNN digit reader, retrain loop, eval gate, corpus audit |
| `templates/`, `static/` | Browser UI (Jinja2 + vanilla JS) |
| `test_*.py` | Test suite (game logic, patterns, persistence, scan regression) |
| `.github/workflows/scanner-ci.yml` | CI: golden scan-regression + persistence round-trip |
| `ROADMAP.md`, `CNN_REBUILD.md`, `ENGINEERING_DECISIONS.md` | Design notes and decisions |

## License

[MIT](LICENSE)