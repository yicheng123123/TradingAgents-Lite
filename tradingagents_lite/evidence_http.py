"""Bounded single-thread source acquisition with immutable, checked disk cache."""
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class SourceError(ValueError):
    """An unavailable source, without pretending that the source returned no documents."""


@dataclass(frozen=True)
class Payload:
    url: str
    body: bytes
    ingested_at: datetime

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.body).hexdigest()


class CachedHTTP:
    def __init__(self, root: Path, *, offline: bool = False, sec_user_agent: str,
                 opener=urlopen, sleep=time.sleep):
        self.root = root
        self.offline = offline
        self.sec_user_agent = sec_user_agent
        self.opener = opener
        self.sleep = sleep

    def get(self, url: str, cap: int = 12_000_000, *, store_body: bool = True) -> Payload:
        parts = urlsplit(url)
        sec = parts.hostname in {"www.sec.gov", "data.sec.gov"}
        if (parts.scheme != "https" or parts.username or parts.password or
                parts.hostname not in {"www.sec.gov", "data.sec.gov", "www.apple.com",
                                       "blogs.microsoft.com", "news.microsoft.com", "www.microsoft.com"}):
            raise SourceError("URL is outside the source allowlist")
        if sec and (not self.sec_user_agent.strip() or any(c in self.sec_user_agent for c in "\r\n")):
            raise SourceError("Set a declared SEC User-Agent")
        key = hashlib.sha256(url.encode()).hexdigest()
        loc = self.root / f"{key}.json"
        if loc.is_file():
            metadata = json.loads(loc.read_text())
            body = (self.root / f"{key}.body").read_bytes()
            if (metadata["url"] != url or len(body) > cap or
                    hashlib.sha256(body).hexdigest() != metadata["content_hash"]):
                raise SourceError("Cached source checksum mismatch")
            return Payload(url, body, datetime.fromisoformat(metadata["ingested_at"]))
        if self.offline:
            raise SourceError("offline_cache_miss")

        # Single request stream with a full pause between SEC requests, including retries.
        for attempt in range(3):
            self.sleep(1.25 if sec else .26)
            delay = 2 ** attempt
            try:
                with self.opener(Request(url, headers={"User-Agent": self.sec_user_agent if sec else
                    "TradingAgents-Lite/0.1 (independent historical evidence research)",
                    "Accept-Encoding": "identity"}), timeout=20) as response:
                    if response.geturl() != url and urlsplit(response.geturl()).hostname != parts.hostname:
                        raise SourceError("Unexpected source redirect")
                    body = response.read(cap + 1)
                if not body or len(body) > cap:
                    raise SourceError("Source payload empty or exceeds the configured bound")
                break
            except HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise SourceError(f"http_{exc.code}") from exc
                retry = exc.headers.get("Retry-After", "")
                try:
                    wait = float(retry) if retry.isdecimal() else (
                        parsedate_to_datetime(retry) - datetime.now(UTC)).total_seconds()
                except (ValueError, TypeError):
                    wait = 0
                if wait > 60:
                    raise SourceError(f"http_{exc.code}_retry_later") from exc
                delay = max(delay, wait)
            except (URLError, TimeoutError, OSError) as exc:
                if attempt == 2:
                    raise SourceError("network_unavailable") from exc
            self.sleep(delay)
        payload = Payload(url, body, datetime.now(UTC))
        if store_body:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / f"{key}.body").write_bytes(payload.body)
            loc.write_text(json.dumps({"url": url, "ingested_at": payload.ingested_at.isoformat(),
                                       "content_hash": payload.content_hash}) + "\n")
        return payload
