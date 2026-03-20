# Memory Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Dockerized web tool to browse, search, and manage OpenClaw memory chunks in any SQLite memory database.

**Architecture:** FastAPI backend serves a single `index.html` (Alpine.js + Tailwind CDN). The backend connects to a SQLite DB with the `vec0.so` extension for vector search, calls Gemini `gemini-embedding-001` for embeddings, and exposes REST endpoints. Runs as an always-on Docker container on port 12089.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, httpx (Gemini API), pymupdf (PDF), sqlite3 + vec0.so, Alpine.js, Tailwind CSS CDN, Docker

---

## File Structure

```
/srv/work/memory-explorer/
├── app.py                  # FastAPI server: all API routes + static file serving
├── chunker.py              # Text chunking logic (400 tokens, 80 overlap, line-based)
├── embeddings.py           # Gemini embedding API client with rate limiting
├── db.py                   # SQLite connection manager (vec0.so loading, all DB queries)
├── static/
│   └── index.html          # Single-page UI (Alpine.js + Tailwind CDN)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml           # uv project config with dependencies
├── .env                     # GEMINI_API_KEY (not committed)
├── .gitignore
└── README.md
```

**Responsibilities:**

| File | Responsibility |
|------|---------------|
| `db.py` | Open SQLite, load vec0.so, CRUD for chunks/files/FTS/vec tables, embedding cache |
| `embeddings.py` | Call Gemini API, rate limiter (100 RPM, 30K TPM, 1K RPD), token counting |
| `chunker.py` | Split text into chunks by line accumulation, 400 tokens / 80 overlap |
| `app.py` | FastAPI routes, file upload handling, PDF text extraction, serve static files |
| `static/index.html` | Tabbed UI: Browse, Search, Create, Upload. Alpine.js reactivity, Tailwind styling |

---

## OpenClaw Memory DB Constants (from source code analysis)

These MUST match OpenClaw's behavior exactly:

```python
CHUNK_TOKENS = 400           # max tokens per chunk
CHUNK_OVERLAP = 80           # overlap tokens between adjacent chunks
CHARS_PER_TOKEN = 4          # approximation used by OpenClaw
MAX_CHUNK_CHARS = 1600       # CHUNK_TOKENS * CHARS_PER_TOKEN
OVERLAP_CHARS = 320          # CHUNK_OVERLAP * CHARS_PER_TOKEN
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMS = 3072
MIN_SCORE = 0.35             # OpenClaw's default cutoff
MAX_RESULTS = 6              # OpenClaw's default
VECTOR_WEIGHT = 0.7          # hybrid search weight
TEXT_WEIGHT = 0.3            # hybrid search weight
CANDIDATE_MULTIPLIER = 4     # fetch 24 candidates, re-rank to 6
```

## Gemini Rate Limits

```python
GEMINI_RPM = 100       # requests per minute
GEMINI_TPM = 30_000    # tokens per minute
GEMINI_RPD = 1_000     # requests per day
```

## OpenClaw Exact Formulas (from source code)

### Chunk ID Generation
From `manager-embedding-ops.ts:873-875`:
```
chunk_id = sha256(f"{source}:{path}:{start_line}:{end_line}:{text_hash}:{model}")
```
Where `text_hash = sha256(chunk_text)`.

### BM25 Rank-to-Score Conversion
From `hybrid.ts:46-55`:
```python
def bm25_rank_to_score(rank: float) -> float:
    if not math.isfinite(rank):
        return 1 / (1 + 999)
    if rank < 0:
        relevance = -rank
        return relevance / (1 + relevance)
    return 1 / (1 + rank)
```

### Hybrid Score Merging
From `hybrid.ts:127-137` — NO normalization. Scores used directly:
```python
score = vector_weight * vector_score + text_weight * text_score
```
Where `vector_score = 1 - cosine_distance` and `text_score = bm25_rank_to_score(bm25_rank)`.

### FTS Query Construction
From `hybrid.ts:33-44` — tokenize, quote, AND-join:
```python
import re
tokens = re.findall(r'[\w]+', query, re.UNICODE)
fts_query = " AND ".join(f'"{t}"' for t in tokens if t)
```

### Relaxed minScore Fallback
From `manager.ts:346-366` — when strict filter returns 0 results but keyword results exist:
```python
if len(results) == 0 and len(keyword_hit_ids) > 0:
    relaxed_min = min(min_score, text_weight)
    results = [r for r in candidates if r.score >= relaxed_min and r.id in keyword_hit_ids]
```

### Notes on Disabled Features
OpenClaw also has MMR (Maximal Marginal Relevance) re-ranking and temporal decay scoring, but both are **disabled by default** (`DEFAULT_MMR_ENABLED = false`, `DEFAULT_TEMPORAL_DECAY_ENABLED = false`). Memory Explorer does not implement these.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env`
- Create: `README.md`

- [ ] **Step 1: Initialize git repo**

```bash
cd /srv/work/memory-explorer
git init
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "memory-explorer"
version = "0.1.0"
description = "Web UI to browse, search, and manage OpenClaw memory SQLite databases"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "httpx>=0.27.0",
    "pymupdf>=1.25.0",
    "python-multipart>=0.0.9",
    "sse-starlette>=2.0.0",
]
```

- [ ] **Step 3: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
*.egg-info/
uv.lock
.env
```

- [ ] **Step 4: Create .env file (never committed)**

```bash
echo "GEMINI_API_KEY=$GEMINI_API_KEY" > /srv/work/memory-explorer/.env
```

The API key must be set as an env var before this step.

- [ ] **Step 5: Create README.md**

Brief usage instructions: docker compose up, port 12089, mount paths, env vars.

- [ ] **Step 6: Install dependencies with uv**

```bash
cd /srv/work/memory-explorer
uv sync
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore README.md
git commit -m "feat: project scaffolding with uv and dependencies"
```

---

### Task 2: Database Layer (`db.py`)

**Files:**
- Create: `/srv/work/memory-explorer/db.py`

This is the core data layer. All SQLite operations go through here.

- [ ] **Step 1: Write db.py**

Must implement:

```python
import math

class MemoryDB:
    def __init__(self, db_path: str, vec_extension_path: str):
        """Open SQLite connection, load vec0.so extension, enable WAL mode.
        Open with check_same_thread=False for FastAPI async.
        If vec0.so fails to load, set self.vec_available = False and
        log a warning (fall back to no vector search)."""

    def list_chunks(self) -> list[dict]:
        """Return all chunks: id, path, source, start_line, end_line,
        text, updated_at (as ISO string), hash, model.
        Ordered by path ASC, start_line ASC."""

    def get_chunk(self, chunk_id: str) -> dict | None:
        """Return single chunk by id with full text."""

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete chunk from chunks, chunks_fts, chunks_vec.
        Returns True if deleted, False if not found."""

    def vector_search(self, embedding: list[float], limit: int) -> list[dict]:
        """Search chunks_vec by cosine distance. Return id + distance.
        Uses struct.pack with little-endian: struct.pack(f'<{len(emb)}f', *emb)
        Fetches `limit` results."""

    def fts_search(self, query: str, limit: int) -> list[dict]:
        """Build FTS query: tokenize with re.findall(r'[\\w]+', query),
        quote each token, join with AND.
        Search chunks_fts using FTS5 MATCH with bm25().
        Return id + bm25_rank."""

    @staticmethod
    def bm25_rank_to_score(rank: float) -> float:
        """Convert BM25 rank to [0,1] score.
        if not finite: return 1 / (1 + 999)
        if rank < 0: relevance = -rank; return relevance / (1 + relevance)
        else: return 1 / (1 + rank)"""

    def hybrid_search(self, embedding: list[float], query: str,
                      max_results: int = 6, min_score: float = 0.35,
                      vector_weight: float = 0.7, text_weight: float = 0.3,
                      candidate_multiplier: int = 4) -> list[dict]:
        """
        1. candidates = max_results * candidate_multiplier (capped at 200)
        2. Fetch `candidates` from vector_search → convert distance to score: 1 - distance
        3. Fetch `candidates` from fts_search → convert rank to score via bm25_rank_to_score()
        4. Merge: for each unique chunk id, combine:
           score = vector_weight * vec_score + text_weight * fts_score
           (NO normalization — use scores directly as OpenClaw does)
        5. Sort by score DESC, filter by min_score, take top max_results
        6. RELAXED FALLBACK: if 0 results pass min_score but FTS had hits:
           relaxed_min = min(min_score, text_weight)
           return candidates with score >= relaxed_min that were in FTS results
        Each result: {id, path, start_line, end_line, text, score, vec_score, fts_score, updated_at}
        """

    def insert_chunk(self, chunk: dict, embedding: list[float]):
        """Insert into chunks, chunks_fts, chunks_vec, embedding_cache.
        chunk dict: id, path, source, start_line, end_line, hash, model, text.
        Embedding stored as json.dumps(embedding) in chunks.embedding column.
        Vector blob uses little-endian: struct.pack(f'<{len(emb)}f', *emb)
        Sets updated_at = current epoch ms (int(time.time() * 1000))."""

    def upsert_file(self, path: str, source: str, content_hash: str, size: int):
        """Insert or update files table entry. Sets mtime = current epoch ms."""

    def get_stats(self) -> dict:
        """Return counts: total_chunks, total_files, total_cached_embeddings."""

    def close(self):
        """Close the connection."""
```

- [ ] **Step 2: Test manually**

```bash
cd /srv/work/memory-explorer
uv run python -c "
from db import MemoryDB
db = MemoryDB('/home/ubuntu/.openclaw/memory/main.sqlite',
              '/home/ubuntu/openclaw/node_modules/.pnpm/sqlite-vec-linux-x64@0.1.7-alpha.2/node_modules/sqlite-vec-linux-x64/vec0.so')
stats = db.get_stats()
print(f'Stats: {stats}')
chunks = db.list_chunks()
print(f'Chunks: {len(chunks)}')
print(f'First: {chunks[0][\"path\"]} lines {chunks[0][\"start_line\"]}-{chunks[0][\"end_line\"]}')
db.close()
"
```

Expected: Stats show chunk/file/cache counts matching the current database state.

- [ ] **Step 3: Commit**

```bash
git add db.py
git commit -m "feat: SQLite database layer with vec0 vector search and hybrid ranking"
```

---

### Task 3: Gemini Embeddings Client (`embeddings.py`)

**Files:**
- Create: `/srv/work/memory-explorer/embeddings.py`

- [ ] **Step 1: Write embeddings.py**

```python
import time
import httpx
from collections import deque

class RateLimiter:
    """Sliding window rate limiter for Gemini API."""

    def __init__(self, rpm: int = 100, tpm: int = 30_000, rpd: int = 1_000):
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self.minute_requests: deque  # (timestamp,) entries
        self.minute_tokens: deque   # (timestamp, token_count) entries
        self.day_requests: deque    # (timestamp,) entries

    def wait_if_needed(self, estimated_tokens: int) -> dict:
        """Check all 3 limits. If any would be exceeded, calculate wait time.
        Returns {ok: bool, wait_seconds: float, reason: str}."""

    def record(self, tokens: int):
        """Record a completed request."""

    def status(self) -> dict:
        """Return current usage: rpm_used, tpm_used, rpd_used, rpm_limit, tpm_limit, rpd_limit."""


class GeminiEmbedder:
    """Generate embeddings using Gemini embedding API."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self.api_key = api_key
        self.model = model
        self.client = httpx.Client(timeout=30.0)
        self.limiter = RateLimiter()

    def embed(self, text: str) -> list[float]:
        """
        1. Estimate tokens (~len(text) / 4)
        2. Check rate limiter; if blocked, raise with wait info
        3. POST to https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent
           Body: {"model": "models/{model}", "content": {"parts": [{"text": text}]}}
           Query param: ?key={api_key}
        4. Record usage
        5. Return embedding values list (3072 floats)
        """

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts sequentially, respecting rate limits.
        Yields progress info between calls via callback if provided."""

    def status(self) -> dict:
        """Return rate limiter status."""
```

Key details:
- Gemini embedding endpoint: `POST /v1beta/models/gemini-embedding-001:embedContent`
- Request body: `{"model": "models/gemini-embedding-001", "content": {"parts": [{"text": "..."}]}}`
- Response: `{"embedding": {"values": [0.1, 0.2, ...]}}`
- API key passed as query param `?key=`
- Token estimation: `len(text) // 4` (same approximation as OpenClaw)

- [ ] **Step 2: Test with real API call**

```bash
cd /srv/work/memory-explorer
source .env && uv run python -c "
from embeddings import GeminiEmbedder
import os
e = GeminiEmbedder(os.environ['GEMINI_API_KEY'])
result = e.embed('test query about vedic astrology')
print(f'Dims: {len(result)}')
print(f'First 5: {result[:5]}')
print(f'Status: {e.status()}')
"
```

Expected: 3072 dimensions, rate limiter shows 1 RPM used.

- [ ] **Step 3: Commit**

```bash
git add embeddings.py
git commit -m "feat: Gemini embedding client with sliding window rate limiter"
```

---

### Task 4: Text Chunker (`chunker.py`)

**Files:**
- Create: `/srv/work/memory-explorer/chunker.py`

- [ ] **Step 1: Write chunker.py**

```python
import hashlib

CHUNK_TOKENS = 400
CHUNK_OVERLAP = 80
CHARS_PER_TOKEN = 4
MAX_CHUNK_CHARS = CHUNK_TOKENS * CHARS_PER_TOKEN  # 1600
OVERLAP_CHARS = CHUNK_OVERLAP * CHARS_PER_TOKEN    # 320
EMBEDDING_MODEL = "gemini-embedding-001"


def chunk_text(text: str, path: str, source: str = "memory") -> list[dict]:
    """
    Split text into chunks matching OpenClaw's chunking behavior
    (from internal.ts:334-416).

    Algorithm (line-accumulation, NOT heading-aware):
    1. Split text into lines (preserving line endings)
    2. Accumulate lines into current chunk
    3. When adding a line would exceed MAX_CHUNK_CHARS, flush current chunk
    4. Carry overlap: new chunk starts with last OVERLAP_CHARS worth of
       lines from previous chunk
    5. If a single line exceeds MAX_CHUNK_CHARS, hard-split at char boundary
    6. Track line numbers (1-based)

    Returns list of dicts with chunk ID matching OpenClaw's formula:
    {
        id: sha256(f"{source}:{path}:{start_line}:{end_line}:{text_hash}:{model}"),
        path: str,
        source: str,
        start_line: int,
        end_line: int,
        text: str,
        hash: sha256(text),
        model: "gemini-embedding-001"
    }
    """


def chunk_markdown(text: str, path: str) -> list[dict]:
    """Convenience wrapper for .md files. source="memory"."""
    return chunk_text(text, path, source="memory")


def estimate_tokens(text: str) -> int:
    """Estimate token count: len(text) // CHARS_PER_TOKEN."""
    return len(text) // CHARS_PER_TOKEN
```

Key details:
- Chunk ID: `sha256(f"{source}:{path}:{start_line}:{end_line}:{text_hash}:{model}")`
- Chunk hash (text_hash): `sha256(text)`
- Line numbers are 1-based
- No heading-aware splitting — pure line-accumulation matching OpenClaw's internal.ts
- If a single line exceeds MAX_CHUNK_CHARS, hard-split at char boundary

- [ ] **Step 2: Test chunker manually**

```bash
cd /srv/work/memory-explorer
uv run python -c "
from chunker import chunk_text, estimate_tokens
text = open('/home/ubuntu/.openclaw/workspace/MEMORY.md').read()
chunks = chunk_text(text, 'MEMORY.md')
print(f'Chunks: {len(chunks)}')
for c in chunks[:3]:
    print(f'  lines {c[\"start_line\"]}-{c[\"end_line\"]}: {len(c[\"text\"])} chars ({estimate_tokens(c[\"text\"])} tokens)')
"
```

Expected: ~29 chunks (matching OpenClaw's count for MEMORY.md). Each chunk ≤1600 chars.

- [ ] **Step 3: Commit**

```bash
git add chunker.py
git commit -m "feat: line-based text chunker matching OpenClaw's 400-token/80-overlap settings"
```

---

### Task 5: FastAPI Server (`app.py`)

**Files:**
- Create: `/srv/work/memory-explorer/app.py`

- [ ] **Step 1: Write app.py**

Endpoints:

```python
# --- App setup ---
# FastAPI app with CORS
# On startup: read DB_PATH, VEC_PATH, GEMINI_API_KEY from env
# Initialize MemoryDB and GeminiEmbedder
# Serve static/index.html at GET /
# DATA_DIR env var defaults to "/data" (for listing .sqlite files)

# --- API Routes ---

GET /api/stats
    # Returns: {total_chunks, total_files, total_cached_embeddings, db_path}

GET /api/chunks
    # Returns: list of all chunks (id, path, start_line, end_line,
    #          text preview (first 150 chars), updated_at as ISO, char_count)

GET /api/chunks/{chunk_id}
    # Returns: full chunk data including complete text

DELETE /api/chunks/{chunk_id}
    # Delete a chunk from all tables. Returns {deleted: bool}

GET /api/search?q={query}
    # 1. Call embedder.embed(query)
    # 2. Call db.hybrid_search(embedding, query)
    # 3. Return results with scores (including vec_score and fts_score),
    #    plus metadata:
    #    {results: [...], openclaw_defaults: {min_score: 0.35, max_results: 6},
    #     rate_limit_status: {...}}

POST /api/chunks
    # Body: {text: str, path: str, source: str}
    # 1. Validate text length <= MAX_CHUNK_CHARS (1600)
    # 2. Generate embedding via Gemini
    # 3. Compute chunk id using OpenClaw formula
    # 4. Insert into DB (all 4 tables)
    # 5. Return created chunk

POST /api/upload
    # Multipart form: file (UploadFile), path_prefix (str, optional)
    # 1. Detect file type (.md or .pdf)
    # 2. Extract text (direct read for .md, pymupdf for .pdf)
    # 3. Chunk the text using chunker.py
    # 4. Return chunks preview: {chunks: [...], total_tokens: int}
    #    (does NOT embed yet - just preview)

POST /api/upload/confirm
    # Body: {chunks: list[dict]} (the chunks from preview, possibly edited/removed)
    # 1. For each chunk: embed via Gemini (respecting rate limits)
    # 2. Insert into DB
    # 3. Return progress via SSE (using sse-starlette EventSourceResponse):
    #    Event name: "progress"
    #    Data: {chunk_index: int, total: int, status: str, rate_limit_status: dict}
    #    Event name: "done"
    #    Data: {inserted: int}
    #    Event name: "rate_limited"
    #    Data: {wait_seconds: float, reason: str}

GET /api/rate-limit
    # Returns current Gemini rate limiter status

GET /api/files
    # List .sqlite files found recursively under DATA_DIR

POST /api/connect
    # Body: {db_path: str}
    # Switch the active DB connection to a different SQLite file
    # Validates file exists and has chunks table
    # Returns new stats
```

Key details:
- `DB_PATH` env var defaults to `/data/memory/main.sqlite`
- `VEC_PATH` env var defaults to `/data/vec0.so`
- `DATA_DIR` env var defaults to `/data`
- `GEMINI_API_KEY` env var required
- PDF text extraction: `pymupdf.open(stream=bytes, filetype="pdf") -> page.get_text()`
- Global `MemoryDB` and `GeminiEmbedder` instances, re-created on `/api/connect`
- SSE uses `sse-starlette` package for proper EventSource format

- [ ] **Step 2: Test server starts**

```bash
cd /srv/work/memory-explorer
source .env && \
DB_PATH=/home/ubuntu/.openclaw/memory/main.sqlite \
VEC_PATH=/home/ubuntu/openclaw/node_modules/.pnpm/sqlite-vec-linux-x64@0.1.7-alpha.2/node_modules/sqlite-vec-linux-x64/vec0.so \
uv run uvicorn app:app --host 0.0.0.0 --port 12089 &
sleep 2
curl -s http://localhost:12089/api/stats | python3 -m json.tool
kill %1
```

Expected: JSON with chunk/file/cache counts matching the database.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: FastAPI server with browse, search, create, delete, upload, and DB switching"
```

---

### Task 6: Frontend UI (`static/index.html`)

**Files:**
- Create: `/srv/work/memory-explorer/static/index.html`

- [ ] **Step 1: Write index.html**

Single HTML file with:
- `<script src="https://cdn.jsdelivr.net/npm/alpinejs@3/dist/cdn.min.js" defer>`
- `<script src="https://cdn.tailwindcss.com">`

**Layout structure:**

```
┌─────────────────────────────────────────────────────────┐
│ Memory Explorer          [SQLite path input] [Connect]  │
│                          Stats: N chunks, N files       │
├────────┬────────┬─────────┬──────────┐                  │
│ Browse │ Search │ Create  │ Upload   │  ← tabs          │
├────────┴────────┴─────────┴──────────┴──────────────────┤
│                                                         │
│  (tab content area)                                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Browse tab:**
- Table with columns: Path, Lines, Preview (first 120 chars), Size, Updated
- Click any row → expands to show full chunk text in a monospace code block
- Each row has a delete button (trash icon, confirms before deleting)
- Sort by clicking column headers (Alpine.js computed sort)

**Search tab:**
- Input field + Search button
- Below: info bar showing OpenClaw defaults: "min_score: 0.35 | max_results: 6 | hybrid: 70% vector + 30% keyword"
- Results as cards, each showing:
  - Path + lines
  - Score as a colored progress bar (red < 0.35, yellow 0.35-0.6, green > 0.6)
  - Breakdown: "vec: 0.82 | fts: 0.45 | combined: 0.71"
  - A dashed line at the 0.35 cutoff between results
  - Label: "OpenClaw would return this" / "Below threshold — OpenClaw discards"
  - Expandable full text
- Rate limit status indicator in corner

**Create tab:**
- Text area with live counter: "342 / 1600 chars (85 / 400 tokens)"
- Counter turns red when over limit
- Path input (e.g., `memory/my-note.md`)
- Source dropdown: "memory" (default)
- Start line / end line inputs (auto-calculated, editable)
- "Create Chunk" button (disabled when over limit or empty)
- Success message with chunk ID after creation

**Upload tab:**
- File input accepting `.md, .pdf`
- Optional path prefix input (default: filename becomes path)
- "Upload & Preview" button → shows table of chunks that will be created:
  - Chunk #, start_line, end_line, preview (first 100 chars), char count, token estimate
  - Each row has a remove (X) button to exclude individual chunks
- "Confirm & Embed" button → starts embedding process:
  - Progress bar: "Embedding chunk 3 of 29..."
  - Rate limit status: "RPM: 3/100 | TPM: 1,200/30,000 | RPD: 45/1,000"
  - If rate limited: "Paused — resuming in 12s (RPM limit)"
  - Uses EventSource to stream SSE progress events
  - Final: "Done! 29 chunks created."

**Styling notes:**
- Dark theme (Tailwind: bg-gray-900, text-gray-100)
- Monospace for chunk text (font-mono)
- Subtle borders (border-gray-700), rounded cards (rounded-lg)
- Responsive but primarily designed for desktop

- [ ] **Step 2: Test in browser**

```bash
cd /srv/work/memory-explorer
source .env && \
DB_PATH=/home/ubuntu/.openclaw/memory/main.sqlite \
VEC_PATH=/home/ubuntu/openclaw/node_modules/.pnpm/sqlite-vec-linux-x64@0.1.7-alpha.2/node_modules/sqlite-vec-linux-x64/vec0.so \
uv run uvicorn app:app --host 0.0.0.0 --port 12089
```

Open `http://<server-ip>:12089` and verify all 4 tabs work.

- [ ] **Step 3: Commit**

```bash
git add static/index.html
git commit -m "feat: single-page UI with browse, search, create, upload tabs"
```

---

### Task 7: Docker Setup

**Files:**
- Create: `/srv/work/memory-explorer/Dockerfile`
- Create: `/srv/work/memory-explorer/docker-compose.yml`

- [ ] **Step 1: Copy vec0.so to a stable location**

The vec0.so path inside `node_modules/.pnpm/` is fragile (version in path). Copy it:

```bash
cp /home/ubuntu/openclaw/node_modules/.pnpm/sqlite-vec-linux-x64@0.1.7-alpha.2/node_modules/sqlite-vec-linux-x64/vec0.so /home/ubuntu/.openclaw/memory/vec0.so
```

- [ ] **Step 2: Write Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 12089

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "12089"]
```

- [ ] **Step 3: Write docker-compose.yml**

```yaml
services:
  memory-explorer:
    build: .
    container_name: memory-explorer
    ports:
      - "12089:12089"
    volumes:
      - /home/ubuntu/.openclaw/memory:/data/memory
      - /home/ubuntu/.openclaw/memory/vec0.so:/data/vec0.so:ro
    environment:
      - DB_PATH=/data/memory/main.sqlite
      - VEC_PATH=/data/vec0.so
      - DATA_DIR=/data
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    restart: unless-stopped
```

- [ ] **Step 4: Build and run**

```bash
cd /srv/work/memory-explorer
docker compose up -d --build
```

- [ ] **Step 5: Verify container is running**

```bash
docker ps | grep memory-explorer
curl -s http://localhost:12089/api/stats | python3 -m json.tool
```

Expected: Container running with `restart: unless-stopped`, stats endpoint returns data.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: Docker setup with always-on container on port 12089"
```

---

### Task 8: End-to-End Verification

- [ ] **Step 1: Test Browse** — `curl /api/chunks`, verify entries with updated_at timestamps
- [ ] **Step 2: Test Search** — `curl '/api/search?q=K+Saaga+compliance'`, verify results have score + vec_score + fts_score, cutoff at 0.35
- [ ] **Step 3: Test Create** — `POST /api/chunks` with a test chunk, verify it appears in browse
- [ ] **Step 4: Test Delete** — `DELETE /api/chunks/{id}` on the test chunk, verify it's gone
- [ ] **Step 5: Test Upload .md** — Upload a small .md file, preview chunks, confirm embedding, verify progress SSE
- [ ] **Step 6: Test Upload .pdf** — Upload a small PDF, verify text extraction + chunking
- [ ] **Step 7: Test DB switch** — `POST /api/connect` with a different path, verify stats change
- [ ] **Step 8: Test rate limit display** — Run multiple searches quickly, verify status updates
- [ ] **Step 9: Test container persistence** — `docker restart memory-explorer`, verify still accessible
