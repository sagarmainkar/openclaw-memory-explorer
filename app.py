from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import os
import json
import hashlib
import time
import asyncio
import sqlite3

from db import MemoryDB
from embeddings import GeminiEmbedder
from chunker import chunk_text, estimate_tokens, MAX_CHUNK_CHARS

# --- Globals ---
DB_PATH = os.environ.get("DB_PATH", "/data/memory/main.sqlite")
VEC_PATH = os.environ.get("VEC_PATH", "/data/vec0.so")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
EMBEDDING_MODEL = "gemini-embedding-001"

db: MemoryDB | None = None
embedder: GeminiEmbedder | None = None

app = FastAPI(title="Memory Explorer")

# CORS — allow all origins (local tool)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    global db, embedder
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is required")
    db = MemoryDB(DB_PATH, VEC_PATH)
    embedder = GeminiEmbedder(GEMINI_API_KEY, model=EMBEDDING_MODEL)


# --- Static files ---

@app.get("/", include_in_schema=False)
async def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "Frontend not yet built. API is available at /docs."})


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


# --- API Routes ---

@app.get("/api/stats")
async def get_stats():
    try:
        stats = db.get_stats()
        stats["db_path"] = DB_PATH
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chunks")
async def list_chunks():
    try:
        chunks = db.list_chunks()
        # Return lightweight listing — no full text
        result = []
        for c in chunks:
            result.append({
                "id": c["id"],
                "path": c["path"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "preview": c["text"][:150],
                "char_count": len(c["text"]),
                "updated_at": c["updated_at"],
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chunks/{chunk_id}")
async def get_chunk(chunk_id: str):
    try:
        chunk = db.get_chunk(chunk_id)
        if chunk is None:
            raise HTTPException(status_code=404, detail="Chunk not found")
        return chunk
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/chunks/{chunk_id}")
async def delete_chunk(chunk_id: str):
    try:
        deleted = db.delete_chunk(chunk_id)
        return {"deleted": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search")
async def search(q: str = ""):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")
    try:
        embedding = embedder.embed(q)
        results = db.hybrid_search(embedding, q)
        return {
            "results": results,
            "openclaw_defaults": {
                "min_score": 0.35,
                "max_results": 6,
                "vector_weight": 0.7,
                "text_weight": 0.3,
            },
            "rate_limit_status": embedder.status(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chunks")
async def create_chunk(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    text = body.get("text", "")
    path = body.get("path", "")
    source = body.get("source", "memory")

    if not text:
        raise HTTPException(status_code=400, detail="'text' is required")
    if not path:
        raise HTTPException(status_code=400, detail="'path' is required")
    if len(text) > MAX_CHUNK_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text length {len(text)} exceeds maximum {MAX_CHUNK_CHARS} chars",
        )

    try:
        embedding = embedder.embed(text)

        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        start_line = 1
        end_line = text.count("\n") + 1

        chunk_id_raw = f"{source}:{path}:{start_line}:{end_line}:{text_hash}:{EMBEDDING_MODEL}"
        chunk_id = hashlib.sha256(chunk_id_raw.encode("utf-8")).hexdigest()

        chunk = {
            "id": chunk_id,
            "path": path,
            "source": source,
            "start_line": start_line,
            "end_line": end_line,
            "text": text,
            "hash": text_hash,
            "model": EMBEDDING_MODEL,
        }

        db.insert_chunk(chunk, embedding)
        return chunk
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    path_prefix: str = Form(None),
):
    try:
        file_bytes = await file.read()
        filename = file.filename or "unknown"
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            import pymupdf
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
        elif ext in (".md", ".txt", ".markdown"):
            text = file_bytes.decode("utf-8")
        else:
            text = file_bytes.decode("utf-8")

        path = path_prefix if path_prefix else f"memory/{filename}"

        chunks = chunk_text(text, path)

        total_tokens = 0
        chunk_summaries = []
        for i, c in enumerate(chunks):
            token_est = estimate_tokens(c["text"])
            total_tokens += token_est
            chunk_summaries.append({
                "index": i,
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "preview": c["text"][:100],
                "char_count": len(c["text"]),
                "token_estimate": token_est,
            })

        return {
            "chunks": chunk_summaries,
            "full_chunks": chunks,  # needed for confirm step
            "path": path,
            "total_tokens": total_tokens,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload/confirm")
async def upload_confirm(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    chunks = body.get("chunks", [])
    path = body.get("path", "")

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks provided")

    async def event_generator():
        inserted = 0
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            # Check rate limiter
            estimated_tokens = len(chunk.get("text", "")) // 4
            check = embedder.limiter.wait_if_needed(estimated_tokens)
            if not check["ok"]:
                yield {
                    "event": "rate_limited",
                    "data": json.dumps({
                        "wait_seconds": check["wait_seconds"],
                        "reason": check["reason"],
                    }),
                }
                await asyncio.sleep(check["wait_seconds"])

            try:
                # Embed the chunk text
                embedding = embedder.embed(chunk["text"])

                # Insert into DB
                db.insert_chunk(chunk, embedding)
                inserted += 1

                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "chunk_index": i,
                        "total": total,
                        "status": "embedded",
                        "rate_limit_status": embedder.status(),
                    }),
                }
            except Exception as e:
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "chunk_index": i,
                        "error": str(e),
                    }),
                }

        yield {
            "event": "done",
            "data": json.dumps({"inserted": inserted}),
        }

    return EventSourceResponse(event_generator())


@app.get("/api/rate-limit")
async def rate_limit():
    return embedder.status()


@app.get("/api/files")
async def list_files():
    files = []
    if os.path.isdir(DATA_DIR):
        for root, dirs, filenames in os.walk(DATA_DIR):
            for f in filenames:
                if f.endswith(".sqlite"):
                    files.append(os.path.join(root, f))
    return {"files": sorted(files)}


@app.post("/api/connect")
async def connect_db(request: Request):
    global db

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    db_path = body.get("db_path", "")
    if not db_path:
        raise HTTPException(status_code=400, detail="'db_path' is required")

    if not os.path.exists(db_path):
        raise HTTPException(status_code=400, detail=f"File not found: {db_path}")

    # Validate it has a chunks table
    try:
        test_conn = sqlite3.connect(db_path)
        tables = test_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchall()
        test_conn.close()
        if not tables:
            raise HTTPException(status_code=400, detail="Database does not have a 'chunks' table")
    except sqlite3.Error as e:
        raise HTTPException(status_code=400, detail=f"Invalid SQLite database: {e}")

    # Close old DB and open new one
    try:
        if db is not None:
            db.close()
        db = MemoryDB(db_path, VEC_PATH)
        stats = db.get_stats()
        stats["db_path"] = db_path
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
