"""Publisher archive adapter: store dated metadata, never article bodies."""
from datetime import date, datetime
from html.parser import HTMLParser
import hashlib
import json
import re
from urllib.parse import urljoin

from .data import NY
from .evidence import Evidence, next_session_proxy
from .evidence_http import CachedHTTP, SourceError

APPLE_ARCHIVE = "https://www.apple.com/newsroom/archive/?page="
MS_ARCHIVES = ["https://blogs.microsoft.com/blog/{year}/{month:02}/",
               "https://news.microsoft.com/source/{year}/{month:02}/"]


class AppleIndex(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.items = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag != "a" or "result__item" not in attrs.get("class", "").split():
            return
        path = attrs.get("href", "")
        label = attrs.get("aria-label", "")
        match = re.fullmatch(r"/newsroom/(20[0-9]{2})/([0-9]{2})/[A-Za-z0-9-]+/", path)
        if not match:
            return
        parts = label.split(" - ", 2)
        if len(parts) != 3:
            return
        try:
            day = datetime.strptime(parts[0], "%B %d, %Y").date()
        except ValueError:
            return
        if (day.year, day.month) != (int(match[1]), int(match[2])) or not parts[2].strip():
            return
        self.items.append({"url": urljoin(APPLE_ARCHIVE, path), "published_date": day.isoformat(),
                           "title": parts[2].strip(), "category": parts[1], "publisher": "Apple Newsroom"})


def _projection(http: CachedHTTP, url: str, parser_factory=AppleIndex) -> dict:
    key = hashlib.sha256(url.encode()).hexdigest()
    path = http.root / f"{key}.news.json"
    if path.is_file():
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = envelope["metadata"]
        if (payload["url"] != url or
                hashlib.sha256(_canonical(payload)).hexdigest() != envelope["metadata_sha256"]):
            raise SourceError("news_metadata_cache_checksum_mismatch")
        return payload
    result = http.get(url, cap=900_000, store_body=False)
    parser = parser_factory()
    parser.feed(result.body.decode("utf-8"))
    items = parser.items
    if not items:
        raise SourceError("no_recognized_publisher_metadata")
    payload = {"url": url, "source_sha256": result.content_hash, "bytes": len(result.body),
               "ingested_at": result.ingested_at.isoformat(), "items": items}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metadata": payload, "metadata_sha256":
                    hashlib.sha256(_canonical(payload)).hexdigest()}, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return payload


def _canonical(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def collect_apple(http: CachedHTTP, start: date, end: date, sessions: tuple[date, ...],
                  pages: list[int]) -> tuple[list[Evidence], dict]:
    records = {}
    requested, failures = [], []
    for page in dict.fromkeys(pages):
        url = APPLE_ARCHIVE + str(page)
        try:
            data = _projection(http, url)
            requested.append({k: data[k] for k in ("url", "source_sha256", "bytes", "ingested_at")}
                             | {"retention": "metadata_only",
                                "projection_sha256": hashlib.sha256(_canonical(data)).hexdigest()})
            for item in data["items"]:
                day = date.fromisoformat(item["published_date"])
                if not start <= day <= end:
                    continue
                at = next_session_proxy(day, sessions)
                record = Evidence(
                    ticker="AAPL", source_type="news", document_id=item["url"], source_url=item["url"],
                    ingested_at=datetime.fromisoformat(data["ingested_at"]), eligible_at=at,
                    availability_basis="date_only_publisher_archive_next_session",
                    pit_level="historical_proxy" if at else "unavailable",
                    content_hash=hashlib.sha256(_canonical(item)).hexdigest(), excerpt=item["title"][:1800],
                    source_metadata={k: v for k, v in item.items() if k != "url"}
                        | {"hash_scope": "canonical_publisher_metadata"})
                records.setdefault(record.evidence_id, record)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            failures.append({"url": url, "code": str(exc) if isinstance(exc, SourceError) else type(exc).__name__})
    return list(records.values()), {
        "provider": "Apple Newsroom", "ticker": "AAPL", "source_type": "news", "requested": requested,
        "failures": failures, "records": len(records), "status": ("partial" if failures else "available")
        if records else "unavailable", "scope": "issuer archive dated headlines only",
    }


def audit_microsoft(http: CachedHTTP, months: list[tuple[int, int]]) -> dict:
    """Record availability outcomes only. No validated historic Microsoft decoder is active."""
    path = http.root / "microsoft-news-probes.json"
    urls = [template.format(year=y, month=m) for y, m in months for template in MS_ARCHIVES]
    if path.is_file():
        results = json.loads(path.read_text())
        if [r["url"] for r in results] == urls:
            return _microsoft_result(results)
    results = []
    for url in urls:
        try:
            payload = http.get(url, cap=900_000, store_body=False)
            results.append({"url": url, "code": "archive_content_not_validated",
                            "source_sha256": payload.content_hash, "bytes": len(payload.body)})
        except SourceError as exc:
            results.append({"url": url, "code": str(exc)})
    if not http.offline:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2) + "\n")
    return _microsoft_result(results)


def _microsoft_result(results: list[dict]) -> dict:
    return {"provider": "Microsoft corporate blogs / Source", "ticker": "MSFT", "source_type": "news",
            "status": "unavailable", "reason": "no_validated_historical_archive", "requested": results,
            "records": 0}


# Curated primary-source discovery URLs. Dates/titles are parsed and checked
# from the publisher. This is not a complete historical Microsoft news feed.
MS_SECURITY_ROOT = "https://www.microsoft.com/en-us/security/blog/"
MS_SECURITY_ARTICLES = (
    "2024/01/25/midnight-blizzard-guidance-for-responders-on-nation-state-attack/",
    "2024/02/14/cyber-signals-navigating-cyberthreats-and-strengthening-defenses-in-the-era-of-ai/",
    "2024/03/13/microsoft-copilot-for-security-is-generally-available-on-april-1-2024-with-new-capabilities/",
    "2024/04/03/get-end-to-end-protection-with-microsofts-unified-security-operations-platform-now-in-public-preview/",
)


class MicrosoftArticle(HTMLParser):
    def __init__(self, url: str):
        super().__init__(convert_charrefs=True)
        self.url = url
        self.meta = {}

    def handle_starttag(self, tag, attrs):
        if tag == "meta":
            attrs = dict(attrs)
            if attrs.get("property") in {"og:title", "article:published_time", "article:modified_time"}:
                self.meta[attrs["property"]] = attrs.get("content", "")

    @property
    def items(self):
        title = self.meta.get("og:title", "").strip()
        timestamp = self.meta.get("article:published_time", "")
        if not title or not timestamp:
            return []
        stamp = datetime.fromisoformat(timestamp)
        if stamp.utcoffset() is None:
            return []
        return [{"url": self.url, "title": title, "published_at": stamp.isoformat(),
                 "publisher": "Microsoft Security Blog",
                 "modified_at_reported": self.meta.get("article:modified_time")}]


def collect_microsoft(http: CachedHTTP, start: date, end: date,
                      sessions: tuple[date, ...]) -> tuple[list[Evidence], dict]:
    records, requested, failures = [], [], []
    for path in MS_SECURITY_ARTICLES:
        url = MS_SECURITY_ROOT + path
        url_day = date.fromisoformat(path[:10].replace("/", "-"))
        if not start <= url_day <= end:
            continue
        try:
            data = _projection(http, url, lambda: MicrosoftArticle(url))
            requested.append({k: data[k] for k in ("url", "source_sha256", "bytes", "ingested_at")}
                             | {"retention": "metadata_only",
                                "projection_sha256": hashlib.sha256(_canonical(data)).hexdigest()})
            item = data["items"][0]
            published = datetime.fromisoformat(item["published_at"])
            public_day = max(url_day, published.astimezone(NY).date())
            if not start <= public_day <= end:
                continue
            reported_revision = item.get("modified_at_reported")
            revision = datetime.fromisoformat(reported_revision) if reported_revision else None
            if revision is not None and revision.utcoffset() is None:
                raise SourceError("publisher_revision_time_has_no_timezone")
            version_day = max(public_day, revision.astimezone(NY).date()) if revision else public_day
            at = next_session_proxy(version_day, sessions)
            records.append(Evidence(
                ticker="MSFT", source_type="news", document_id=url, source_url=url,
                published_at=published, ingested_at=datetime.fromisoformat(data["ingested_at"]),
                eligible_at=at, availability_basis="next_session_after_publisher_timestamp_and_reported_revision",
                content_hash=hashlib.sha256(_canonical(item)).hexdigest(), excerpt=item["title"][:1800],
                pit_level="historical_proxy" if at else "unavailable",
                source_metadata={"title": item["title"], "publisher": item["publisher"],
                                 "published_date": str(public_day),
                                 "modified_at_reported": item["modified_at_reported"],
                                 "hash_scope": "canonical_publisher_metadata"}))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            failures.append({"url": url, "code": str(exc) if isinstance(exc, SourceError) else type(exc).__name__})
    return records, {"provider": "Microsoft Security Blog", "ticker": "MSFT", "source_type": "news",
                     "requested": requested, "failures": failures, "records": len(records),
                     "status": ("partial" if failures else "available")
                     if any(r.eligible_at is not None for r in records) else "unavailable",
                     "unavailable_versions": sum(r.eligible_at is None for r in records),
                     "scope": "four preselected issuer security posts, not a market news index"}
