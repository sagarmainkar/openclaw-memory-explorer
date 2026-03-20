# Memory Explorer

Web UI to browse, search, and manage OpenClaw memory chunks stored in SQLite databases.

## Tech Stack

- Python 3.12, FastAPI, uvicorn
- SQLite3 for memory storage
- Alpine.js + Tailwind CSS for the frontend
- Dockerized deployment

## Quick Start

### Docker

```bash
docker build -t memory-explorer .
docker run -d \
  -p 12089:12089 \
  -v /path/to/memories:/data \
  --env-file .env \
  memory-explorer
```

The application will be available at `http://localhost:12089`.

### Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | API key for Gemini integration |

### Mount Paths

| Container Path | Purpose |
|---|---|
| `/data` | Directory containing SQLite memory databases |

## Local Development

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 12089 --reload
```
