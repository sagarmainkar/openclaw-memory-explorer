import time
import httpx
from collections import deque


class RateLimiter:
    """Sliding window rate limiter for Gemini API."""

    def __init__(self, rpm: int = 100, tpm: int = 30_000, rpd: int = 1_000):
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self.minute_requests: deque[float] = deque()
        self.minute_tokens: deque[tuple[float, int]] = deque()
        self.day_requests: deque[float] = deque()

    def wait_if_needed(self, estimated_tokens: int) -> dict:
        """Check all 3 limits. If any would be exceeded, calculate wait time.
        Purge entries older than 60s (for minute windows) or 86400s (for day window).
        Returns {ok: bool, wait_seconds: float, reason: str}."""
        now = time.monotonic()

        # Purge expired entries
        while self.minute_requests and now - self.minute_requests[0] >= 60:
            self.minute_requests.popleft()
        while self.minute_tokens and now - self.minute_tokens[0][0] >= 60:
            self.minute_tokens.popleft()
        while self.day_requests and now - self.day_requests[0] >= 86400:
            self.day_requests.popleft()

        # Check RPM
        if len(self.minute_requests) >= self.rpm:
            wait = 60 - (now - self.minute_requests[0])
            return {"ok": False, "wait_seconds": max(0.0, wait), "reason": "rpm"}

        # Check TPM
        current_tokens = sum(t[1] for t in self.minute_tokens)
        if current_tokens + estimated_tokens > self.tpm:
            wait = 60 - (now - self.minute_tokens[0][0])
            return {"ok": False, "wait_seconds": max(0.0, wait), "reason": "tpm"}

        # Check RPD
        if len(self.day_requests) >= self.rpd:
            wait = 86400 - (now - self.day_requests[0])
            return {"ok": False, "wait_seconds": max(0.0, wait), "reason": "rpd"}

        return {"ok": True, "wait_seconds": 0.0, "reason": ""}

    def record(self, tokens: int):
        """Record a completed request with timestamp and token count."""
        now = time.monotonic()
        self.minute_requests.append(now)
        self.minute_tokens.append((now, tokens))
        self.day_requests.append(now)

    def status(self) -> dict:
        """Return current usage: rpm_used, tpm_used, rpd_used, rpm_limit, tpm_limit, rpd_limit."""
        now = time.monotonic()

        # Purge expired entries before reporting
        while self.minute_requests and now - self.minute_requests[0] >= 60:
            self.minute_requests.popleft()
        while self.minute_tokens and now - self.minute_tokens[0][0] >= 60:
            self.minute_tokens.popleft()
        while self.day_requests and now - self.day_requests[0] >= 86400:
            self.day_requests.popleft()

        return {
            "rpm_used": len(self.minute_requests),
            "tpm_used": sum(t[1] for t in self.minute_tokens),
            "rpd_used": len(self.day_requests),
            "rpm_limit": self.rpm,
            "tpm_limit": self.tpm,
            "rpd_limit": self.rpd,
        }


class GeminiEmbedder:
    """Generate embeddings using Gemini embedding API."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self.api_key = api_key
        self.model = model
        self.client = httpx.Client(timeout=30.0)
        self.limiter = RateLimiter()

    def embed(self, text: str) -> list[float]:
        """
        1. Estimate tokens: len(text) // 4
        2. Check rate limiter; if not ok, sleep for wait_seconds then re-check
        3. POST to Gemini embedContent endpoint
        4. Parse response embedding values
        5. Record usage in rate limiter
        6. Return embedding values list (3072 floats)
        Raise httpx.HTTPStatusError on API errors.
        """
        estimated_tokens = len(text) // 4

        # Wait for rate limiter clearance
        while True:
            check = self.limiter.wait_if_needed(estimated_tokens)
            if check["ok"]:
                break
            time.sleep(check["wait_seconds"])

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"
        body = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
        }

        response = self.client.post(
            url,
            params={"key": self.api_key},
            json=body,
        )
        response.raise_for_status()

        values = response.json()["embedding"]["values"]

        self.limiter.record(estimated_tokens)

        return values

    def status(self) -> dict:
        """Return rate limiter status."""
        return self.limiter.status()
