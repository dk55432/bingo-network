# Bingo Server

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload --host 0.0.0.0
```

Open:

Host:
http://localhost:8000/host

Join:
http://localhost:8000/join
