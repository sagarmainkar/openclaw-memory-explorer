<div align="center">

# OpenClaw Memory Explorer

### Browse, search, and manage your AI agent's memory

<br>

<img src="https://raw.githubusercontent.com/sagarmainkar/openclaw-memory-explorer/main/static/lobster.svg" width="120" alt="OpenClaw Lobster">

<br><br>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://docker.com)

**A web-based tool for inspecting, searching, and managing [OpenClaw](https://github.com/nichochar/open-claw) memory databases. Replicate the exact same hybrid search your agent uses, upload new memories from files and images, edit chunks inline, and simulate what your agent will find.**

[Features](#features) &middot; [Quick Start](#quick-start) &middot; [Screenshots](#screenshots) &middot; [How It Works](#how-it-works) &middot; [Configuration](#configuration) &middot; [Contributing](#contributing)

</div>

---

## Features

| Feature | Description |
|---------|-------------|
| **Browse** | Sortable table of all memory chunks with resizable columns, inline expand, and bulk operations |
| **Search** | Hybrid vector + keyword search replicating OpenClaw's exact behavior (0.35 cutoff, 6 max results, 70/30 weights) with confidence scores and score breakdowns |
| **Create** | Add individual memory chunks with live token counter (400 token / 1600 char limit) |
| **Upload** | Upload `.md`, `.pdf`, or images (`.jpg`, `.png`) — images are processed by AI vision (Ollama/kimi-k2.5) to extract text |
| **Edit** | Edit any chunk inline with re-embedding on save |
| **Delete** | Delete chunks from DB only, or delete with source file to prevent re-indexing |
| **Tag** | Bulk-tag chunks or tag at upload time — tags are embedded in the text so they improve search relevance |
| **Part Linking** | Multi-chunk uploads are auto-labeled `[Part 1 of 3 from: source]` so the agent knows related chunks exist |
| **Source Viewer** | Click any source path to view the original file (renders images inline) |
| **File Browser** | Browse and connect to any SQLite database on the host |
| **Themes** | 5 built-in themes: Dark, Midnight, Forest, Ember, Light (persisted in localStorage) |

## Quick Start

### Docker Compose (recommended)

1. **Clone the repo:**

```bash
git clone https://github.com/sagarmainkar/openclaw-memory-explorer.git
cd openclaw-memory-explorer
```

2. **Create `.env`:**

```bash
echo "GEMINI_API_KEY=your-gemini-api-key" > .env
```

3. **Edit `docker-compose.yml`** to match your paths:

```yaml
volumes:
  - /path/to/your/.openclaw/memory:/data/memory       # memory SQLite DB
  - /path/to/vec0.so:/data/vec0.so:ro                  # sqlite-vec extension
  - /path/to/your/.openclaw/workspace:/workspace        # workspace files
  - /home/youruser:/host:ro                             # host browsing
```

4. **Start:**

```bash
docker compose up -d --build
```

5. **Open:** `http://localhost:12089`

### Local Development

```bash
# Install dependencies
uv sync

# Run
DB_PATH=/path/to/main.sqlite \
VEC_PATH=/path/to/vec0.so \
GEMINI_API_KEY=your-key \
uv run uvicorn app:app --host 0.0.0.0 --port 12089 --reload
```

## How It Works

### Search — Replicating OpenClaw Exactly

Memory Explorer uses the **exact same hybrid search algorithm** as OpenClaw:

```
score = 0.7 * vector_similarity + 0.3 * keyword_relevance
```

| Parameter | Value | Source |
|-----------|-------|--------|
| `maxResults` | 6 | `agents/memory-search.ts` |
| `minScore` | 0.35 | `agents/memory-search.ts` |
| `vectorWeight` | 0.7 | `agents/memory-search.ts` |
| `textWeight` | 0.3 | `agents/memory-search.ts` |
| `candidateMultiplier` | 4 | Fetches 24 candidates, re-ranks to top 6 |
| `chunkTokens` | 400 | ~1,600 chars per chunk |
| `chunkOverlap` | 80 | ~320 chars overlap between adjacent chunks |

The search results show a **confidence bar** and label each result as either "OpenClaw would return this" or "Below threshold — OpenClaw would discard", so you can simulate exactly what your agent will find.

### Image Upload — AI Vision Processing

When you upload an image, it's sent to **Ollama** (running locally or as a cloud proxy) for text extraction and description:

```
Image bytes → Ollama kimi-k2.5 → Structured text → Chunked → Gemini embedding → Stored
```

The original image is saved to the workspace so you can view it later via the source viewer.

### Chunk Relationships — Part Labels

When a file produces multiple chunks, each is auto-labeled:

```
[Part 1 of 3 from: memory/trip-day3.png]
...chunk content...
```
```
[Final part of 3 from: memory/trip-day3.png]
...chunk content...
```

This way, when the agent finds one chunk via search, it knows related chunks exist and can search for them.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `/data/memory/main.sqlite` | Path to the SQLite memory database |
| `VEC_PATH` | `/data/vec0.so` | Path to the sqlite-vec extension |
| `DATA_DIR` | `/host` | Root directory for file browsing |
| `WORKSPACE_DIR` | `/workspace` | OpenClaw workspace (for source files) |
| `GEMINI_API_KEY` | *required* | Google Gemini API key for embeddings |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama API endpoint for vision |
| `OLLAMA_VISION_MODEL` | `kimi-k2.5:cloud` | Ollama model for image processing |

### Gemini Rate Limits

The built-in rate limiter respects free-tier Gemini limits:

| Limit | Value |
|-------|-------|
| Requests per minute | 100 |
| Tokens per minute | 30,000 |
| Requests per day | 1,000 |

Rate limit status is displayed in the UI during search and upload operations.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.12, FastAPI, uvicorn |
| **Database** | SQLite3 + [sqlite-vec](https://github.com/asg017/sqlite-vec) for vector search + FTS5 for keyword search |
| **Embeddings** | Google Gemini `gemini-embedding-001` (3072 dimensions) |
| **Vision** | Ollama with kimi-k2.5 (or any vision-capable model) |
| **Frontend** | Single HTML file — Alpine.js + Tailwind CSS (CDN, no build step) |
| **PDF** | PyMuPDF for text extraction |
| **Container** | Docker with `restart: unless-stopped` |

## Project Structure

```
openclaw-memory-explorer/
├── app.py              # FastAPI server — all API routes
├── db.py               # SQLite + vec0 + FTS5 — hybrid search engine
├── embeddings.py       # Gemini embedding client + rate limiter
├── chunker.py          # Text chunker (400 tokens, 80 overlap)
├── static/
│   ├── index.html      # Single-page UI (Alpine.js + Tailwind)
│   └── lobster.svg     # OpenClaw logo
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Contributing

Contributions welcome! This project is intentionally simple — single HTML file, no build step, no framework complexity.

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Submit a PR

## License

MIT

---

<div align="center">

Built with care for the [OpenClaw](https://github.com/nichochar/open-claw) community.

</div>
