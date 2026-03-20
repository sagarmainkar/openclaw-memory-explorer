import hashlib

CHUNK_TOKENS = 400
CHUNK_OVERLAP = 80
CHARS_PER_TOKEN = 4
MAX_CHUNK_CHARS = CHUNK_TOKENS * CHARS_PER_TOKEN  # 1600
OVERLAP_CHARS = CHUNK_OVERLAP * CHARS_PER_TOKEN    # 320
EMBEDDING_MODEL = "gemini-embedding-001"


def _sha256(text: str) -> str:
    """Return the hex SHA-256 digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_chunk_id(source: str, path: str, start_line: int, end_line: int, text_hash: str, model: str) -> str:
    """
    Produce a chunk ID matching OpenClaw's formula
    (manager-embedding-ops.ts:873-875).
    """
    raw = f"{source}:{path}:{start_line}:{end_line}:{text_hash}:{model}"
    return _sha256(raw)


def _hard_split_line(line: str, start_line_num: int, path: str, source: str, model: str) -> list[dict]:
    """
    Hard-split a single line that exceeds MAX_CHUNK_CHARS into multiple
    chunks at character boundaries.  Each sub-chunk shares the same
    1-based line number.
    """
    chunks: list[dict] = []
    offset = 0
    while offset < len(line):
        piece = line[offset : offset + MAX_CHUNK_CHARS]
        text_hash = _sha256(piece)
        chunks.append({
            "id": _make_chunk_id(source, path, start_line_num, start_line_num, text_hash, model),
            "path": path,
            "source": source,
            "start_line": start_line_num,
            "end_line": start_line_num,
            "text": piece,
            "hash": text_hash,
            "model": model,
        })
        offset += MAX_CHUNK_CHARS
    return chunks


def chunk_text(text: str, path: str, source: str = "memory") -> list[dict]:
    """
    Split text into chunks matching OpenClaw's chunking behavior
    (internal.ts:334-416).

    Algorithm (line-accumulation, NOT heading-aware):
    1. Split text into lines (preserving line endings).
    2. Accumulate lines into the current chunk.
    3. When adding a line would exceed MAX_CHUNK_CHARS, flush the
       current chunk.
    4. Carry overlap: new chunk starts with the last OVERLAP_CHARS
       worth of lines from the previous chunk.
    5. If a single line exceeds MAX_CHUNK_CHARS, hard-split at char
       boundary.
    6. Track line numbers (1-based).

    Returns a list of dicts with keys:
        id, path, source, start_line, end_line, text, hash, model
    """
    if not text:
        return []

    model = EMBEDDING_MODEL

    # Split into lines, preserving line endings.
    # We use splitlines(keepends=True) so trailing \n is kept per line.
    lines = text.splitlines(keepends=True)

    chunks: list[dict] = []
    current_lines: list[str] = []       # accumulated lines for the current chunk
    current_start: int = 1              # 1-based start line of the current chunk
    current_len: int = 0                # total char length of current_lines

    for i, line in enumerate(lines):
        line_num = i + 1  # 1-based

        # If the single line itself exceeds MAX_CHUNK_CHARS, we need to
        # flush whatever we have, then hard-split the long line.
        if len(line) > MAX_CHUNK_CHARS:
            # Flush the current accumulator first (if non-empty).
            if current_lines:
                chunk_text_str = "".join(current_lines)
                if chunk_text_str:
                    text_hash = _sha256(chunk_text_str)
                    chunks.append({
                        "id": _make_chunk_id(source, path, current_start,
                                             current_start + len(current_lines) - 1,
                                             text_hash, model),
                        "path": path,
                        "source": source,
                        "start_line": current_start,
                        "end_line": current_start + len(current_lines) - 1,
                        "text": chunk_text_str,
                        "hash": text_hash,
                        "model": model,
                    })
                current_lines = []
                current_len = 0

            # Hard-split the oversized line.
            chunks.extend(_hard_split_line(line, line_num, path, source, model))

            # After a hard-split, there's no meaningful overlap to carry.
            # The next chunk starts fresh from the next line.
            current_start = line_num + 1
            continue

        # Would adding this line exceed the limit?
        if current_len + len(line) > MAX_CHUNK_CHARS and current_lines:
            # --- Flush the current chunk ---
            chunk_text_str = "".join(current_lines)
            end_line = current_start + len(current_lines) - 1

            if chunk_text_str:
                text_hash = _sha256(chunk_text_str)
                chunks.append({
                    "id": _make_chunk_id(source, path, current_start, end_line,
                                         text_hash, model),
                    "path": path,
                    "source": source,
                    "start_line": current_start,
                    "end_line": end_line,
                    "text": chunk_text_str,
                    "hash": text_hash,
                    "model": model,
                })

            # --- Carry overlap ---
            # Collect lines from the END of the flushed chunk whose total
            # length is <= OVERLAP_CHARS.
            overlap_lines: list[str] = []
            overlap_len = 0
            for ol in reversed(current_lines):
                if overlap_len + len(ol) > OVERLAP_CHARS:
                    break
                overlap_lines.insert(0, ol)
                overlap_len += len(ol)

            # New chunk starts with the overlap lines.
            # The start_line of the new chunk corresponds to where the
            # first overlap line was in the original text.
            if overlap_lines:
                overlap_start_index = len(current_lines) - len(overlap_lines)
                current_start = current_start + overlap_start_index
            else:
                current_start = line_num

            current_lines = list(overlap_lines)
            current_len = overlap_len

        # Accumulate the line.
        if not current_lines and current_len == 0:
            # Starting fresh — record start line.
            current_start = line_num
        current_lines.append(line)
        current_len += len(line)

    # --- Flush remaining lines ---
    if current_lines:
        chunk_text_str = "".join(current_lines)
        if chunk_text_str:
            end_line = current_start + len(current_lines) - 1
            text_hash = _sha256(chunk_text_str)
            chunks.append({
                "id": _make_chunk_id(source, path, current_start, end_line,
                                     text_hash, model),
                "path": path,
                "source": source,
                "start_line": current_start,
                "end_line": end_line,
                "text": chunk_text_str,
                "hash": text_hash,
                "model": model,
            })

    return chunks


def chunk_markdown(text: str, path: str) -> list[dict]:
    """Convenience wrapper for .md files. source='memory'."""
    return chunk_text(text, path, source="memory")


def estimate_tokens(text: str) -> int:
    """Estimate token count: len(text) // CHARS_PER_TOKEN."""
    return len(text) // CHARS_PER_TOKEN
