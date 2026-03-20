import math
import json
import re
import struct
import time
import hashlib
import sqlite3
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class MemoryDB:
    def __init__(self, db_path: str, vec_extension_path: str):
        """Open SQLite connection, load vec0.so extension, enable WAL mode.
        Open with check_same_thread=False for FastAPI async.
        If vec0.so fails to load, set self.vec_available = False and
        log a warning (fall back to no vector search)."""
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")

        self.vec_available = False
        try:
            self.conn.enable_load_extension(True)
            self.conn.load_extension(vec_extension_path)
            self.conn.enable_load_extension(False)
            self.vec_available = True
        except Exception as e:
            logger.warning(f"Failed to load vec0.so extension: {e}. Vector search disabled.")

    def list_chunks(self) -> list[dict]:
        """Return all chunks: id, path, source, start_line, end_line,
        text, updated_at (as ISO string from epoch ms), hash, model.
        Ordered by path ASC, start_line ASC."""
        rows = self.conn.execute(
            "SELECT id, path, source, start_line, end_line, text, updated_at, hash, model "
            "FROM chunks ORDER BY path ASC, start_line ASC"
        ).fetchall()
        return [self._chunk_row_to_dict(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> dict | None:
        """Return single chunk by id with full text."""
        row = self.conn.execute(
            "SELECT id, path, source, start_line, end_line, text, updated_at, hash, model "
            "FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        return self._chunk_row_to_dict(row)

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete chunk from chunks, chunks_fts, chunks_vec.
        Returns True if deleted, False if not found."""
        row = self.conn.execute("SELECT id FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if row is None:
            return False
        self.conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
        self.conn.execute("DELETE FROM chunks_fts WHERE id = ?", (chunk_id,))
        if self.vec_available:
            try:
                self.conn.execute("DELETE FROM chunks_vec WHERE id = ?", (chunk_id,))
            except Exception as e:
                logger.warning(f"Failed to delete from chunks_vec: {e}")
        self.conn.commit()
        return True

    def vector_search(self, embedding: list[float], limit: int) -> list[dict]:
        """Search chunks_vec by cosine distance. Return id + distance.
        Uses struct.pack with little-endian: struct.pack(f'<{len(emb)}f', *emb)
        Fetches `limit` results."""
        if not self.vec_available:
            return []
        n = len(embedding)
        blob = struct.pack(f"<{n}f", *embedding)
        rows = self.conn.execute(
            "SELECT id, vec_distance_cosine(embedding, ?) AS distance "
            "FROM chunks_vec "
            "ORDER BY distance ASC "
            "LIMIT ?",
            (blob, limit),
        ).fetchall()
        return [{"id": row["id"], "distance": row["distance"]} for row in rows]

    def fts_search(self, query: str, limit: int) -> list[dict]:
        """Build FTS query: tokenize with re.findall(r'[\\w]+', query),
        quote each token, join with AND.
        Search chunks_fts using FTS5 bm25() ranking.
        Return id + bm25_rank."""
        tokens = re.findall(r"[\w]+", query)
        if not tokens:
            return []
        quoted = [f'"{t}"' for t in tokens]
        fts_query = " AND ".join(quoted)
        rows = self.conn.execute(
            "SELECT id, path, source, start_line, end_line, text, bm25(chunks_fts) AS rank "
            "FROM chunks_fts "
            "WHERE chunks_fts MATCH ? "
            "ORDER BY rank ASC "
            "LIMIT ?",
            (fts_query, limit),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "bm25_rank": row["rank"],
                "path": row["path"],
                "source": row["source"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "text": row["text"],
            }
            for row in rows
        ]

    @staticmethod
    def bm25_rank_to_score(rank: float) -> float:
        """Convert BM25 rank to [0,1] score.
        Match OpenClaw's hybrid.ts:46-55 exactly:
        if not finite: return 1 / (1 + 999)
        if rank < 0: relevance = -rank; return relevance / (1 + relevance)
        else: return 1 / (1 + rank)"""
        if not math.isfinite(rank):
            return 1 / (1 + 999)
        if rank < 0:
            relevance = -rank
            return relevance / (1 + relevance)
        return 1 / (1 + rank)

    def hybrid_search(
        self,
        embedding: list[float],
        query: str,
        max_results: int = 6,
        min_score: float = 0.35,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        candidate_multiplier: int = 4,
    ) -> list[dict]:
        """
        Match OpenClaw's exact hybrid search behavior:
        1. candidates = min(200, max_results * candidate_multiplier)
        2. Fetch `candidates` from vector_search -> score = 1 - distance
        3. Fetch `candidates` from fts_search -> score = bm25_rank_to_score(rank)
        4. Merge: for each unique chunk id:
           combined_score = vector_weight * vec_score + text_weight * fts_score
           (NO normalization -- use scores directly)
        5. Sort by score DESC, filter by min_score, take top max_results
        6. RELAXED FALLBACK (from manager.ts:346-366): if 0 results pass min_score
           but FTS had hits:
           relaxed_min = min(min_score, text_weight)
           return candidates with score >= relaxed_min that were in FTS result set
        Each result: {id, path, start_line, end_line, text, score, vec_score, fts_score, updated_at}
        """
        candidates = min(200, max(1, max_results * candidate_multiplier))

        # Fetch vector results
        vec_results = self.vector_search(embedding, candidates) if self.vec_available else []

        # Fetch FTS results
        fts_results = self.fts_search(query, candidates)

        # Build lookup by id
        by_id: dict[str, dict] = {}

        for vr in vec_results:
            vec_score = 1 - vr["distance"]
            by_id[vr["id"]] = {
                "id": vr["id"],
                "vec_score": vec_score,
                "fts_score": 0.0,
            }

        for fr in fts_results:
            fts_score = self.bm25_rank_to_score(fr["bm25_rank"])
            if fr["id"] in by_id:
                by_id[fr["id"]]["fts_score"] = fts_score
            else:
                by_id[fr["id"]] = {
                    "id": fr["id"],
                    "vec_score": 0.0,
                    "fts_score": fts_score,
                }

        # Compute combined scores
        merged = []
        for entry in by_id.values():
            combined = vector_weight * entry["vec_score"] + text_weight * entry["fts_score"]
            merged.append({**entry, "score": combined})

        # Sort by score descending
        merged.sort(key=lambda x: x["score"], reverse=True)

        # Strict filter
        strict = [m for m in merged if m["score"] >= min_score]

        if len(strict) > 0 or len(fts_results) == 0:
            results = strict[:max_results]
        else:
            # Relaxed fallback: lower threshold for FTS-matched results
            relaxed_min = min(min_score, text_weight)
            fts_keys = set()
            for fr in fts_results:
                # Build key from chunk metadata to match OpenClaw's behavior
                chunk_info = self._get_chunk_info(fr["id"])
                if chunk_info:
                    key = f"{chunk_info['source']}:{chunk_info['path']}:{chunk_info['start_line']}:{chunk_info['end_line']}"
                    fts_keys.add(key)

            relaxed = []
            for m in merged:
                chunk_info = self._get_chunk_info(m["id"])
                if chunk_info:
                    key = f"{chunk_info['source']}:{chunk_info['path']}:{chunk_info['start_line']}:{chunk_info['end_line']}"
                    if key in fts_keys and m["score"] >= relaxed_min:
                        relaxed.append(m)
            results = relaxed[:max_results]

        # Enrich results with chunk data
        enriched = []
        for r in results:
            chunk = self.get_chunk(r["id"])
            if chunk:
                enriched.append({
                    "id": r["id"],
                    "path": chunk["path"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "text": chunk["text"],
                    "score": r["score"],
                    "vec_score": r["vec_score"],
                    "fts_score": r["fts_score"],
                    "updated_at": chunk["updated_at"],
                })
        return enriched

    def _get_chunk_info(self, chunk_id: str) -> dict | None:
        """Get source, path, start_line, end_line for a chunk."""
        row = self.conn.execute(
            "SELECT source, path, start_line, end_line FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "source": row["source"],
            "path": row["path"],
            "start_line": row["start_line"],
            "end_line": row["end_line"],
        }

    def insert_chunk(self, chunk: dict, embedding: list[float]):
        """Insert into chunks, chunks_fts, chunks_vec, embedding_cache.
        chunk dict: id, path, source, start_line, end_line, hash, model, text.
        Store embedding as json.dumps(embedding) in chunks.embedding column.
        Vector blob: struct.pack(f'<{len(emb)}f', *emb) for chunks_vec.
        embedding_cache entry: provider='gemini', model=chunk['model'],
          provider_key='memory-explorer', hash=chunk['hash'],
          embedding=json.dumps(embedding), dims=len(embedding),
          updated_at=epoch_ms.
        Sets updated_at = int(time.time() * 1000)."""
        now_ms = int(time.time() * 1000)
        embedding_json = json.dumps(embedding)

        self.conn.execute(
            "INSERT INTO chunks (id, path, source, start_line, end_line, hash, model, text, embedding, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                chunk["id"],
                chunk["path"],
                chunk["source"],
                chunk["start_line"],
                chunk["end_line"],
                chunk["hash"],
                chunk["model"],
                chunk["text"],
                embedding_json,
                now_ms,
            ),
        )

        self.conn.execute(
            "INSERT INTO chunks_fts (text, id, path, source, model, start_line, end_line) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                chunk["text"],
                chunk["id"],
                chunk["path"],
                chunk["source"],
                chunk["model"],
                chunk["start_line"],
                chunk["end_line"],
            ),
        )

        if self.vec_available:
            n = len(embedding)
            blob = struct.pack(f"<{n}f", *embedding)
            self.conn.execute(
                "INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)",
                (chunk["id"], blob),
            )

        self.conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (provider, model, provider_key, hash, embedding, dims, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "gemini",
                chunk["model"],
                "memory-explorer",
                chunk["hash"],
                embedding_json,
                len(embedding),
                now_ms,
            ),
        )

        self.conn.commit()

    def upsert_file(self, path: str, source: str, content_hash: str, size: int):
        """Insert or update files table entry. Sets mtime = current epoch ms."""
        now_ms = int(time.time() * 1000)
        self.conn.execute(
            "INSERT OR REPLACE INTO files (path, source, hash, mtime, size) "
            "VALUES (?, ?, ?, ?, ?)",
            (path, source, content_hash, now_ms, size),
        )
        self.conn.commit()

    def get_stats(self) -> dict:
        """Return counts: total_chunks, total_files, total_cached_embeddings."""
        total_chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        total_files = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        total_cached = self.conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0]
        return {
            "total_chunks": total_chunks,
            "total_files": total_files,
            "total_cached_embeddings": total_cached,
        }

    def close(self):
        """Close the connection."""
        self.conn.close()

    def _chunk_row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a chunk row to a dictionary with ISO-formatted updated_at."""
        epoch_ms = row["updated_at"]
        iso_str = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).isoformat()
        return {
            "id": row["id"],
            "path": row["path"],
            "source": row["source"],
            "start_line": row["start_line"],
            "end_line": row["end_line"],
            "text": row["text"],
            "updated_at": iso_str,
            "hash": row["hash"],
            "model": row["model"],
        }
