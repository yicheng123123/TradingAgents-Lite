"""Read-only BigModel console data, kept in the caller's session."""

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import os
import re
from typing import Callable

import httpx


ORIGIN = "https://open.bigmodel.cn"
BALANCE_PATH = "/api/biz/account/query-customer-account-report"
PACKAGES_PATH = "/api/biz/tokenAccounts/list/my"
CACHE_TTL = timedelta(seconds=60)
CONSOLE_TZ = timezone(timedelta(hours=8))
MODEL = "glm-5.3"
STATUS_ISSUES = {401: "Authentication required", 403: "Access denied", 429: "Rate limited"}


@dataclass(frozen=True)
class AccountBalance:
    available_cny: Decimal | None
    spend_cny: Decimal | None
    granted_cny: Decimal | None


@dataclass(frozen=True)
class ResourcePackage:
    remaining_tokens: int | None
    available_tokens: int | None
    expires_at: datetime


@dataclass(frozen=True)
class ResourcePackages:
    verified: tuple[ResourcePackage, ...]
    complete: bool

    def active(self, now: datetime) -> tuple[ResourcePackage, ...]:
        return tuple(package for package in self.verified if package.expires_at > now)


@dataclass(frozen=True)
class AccountSection[T]:
    value: T | None = None
    synced_at: datetime | None = None
    issue: str | None = None

    def is_fresh(self, now: datetime) -> bool:
        return self.synced_at is not None and timedelta(0) <= now - self.synced_at < CACHE_TTL


@dataclass(frozen=True)
class AccountSnapshot:
    balance: AccountSection[AccountBalance] = field(default_factory=AccountSection)
    packages: AccountSection[ResourcePackages] = field(default_factory=AccountSection)
    key_tag: bytes | None = field(default=None, repr=False)


class AccountQueryError(ValueError):
    """An error with a fixed user-facing label, independent of response text."""


def _number(value: object) -> Decimal | None:
    if type(value) not in (str, int, Decimal):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    if not number.is_finite() or (not number.is_zero() and abs(number.adjusted()) > 30):
        return None
    return number


def _tokens(value: object) -> int | None:
    number = _number(value)
    if number is None or number < 0 or number != number.to_integral_value():
        return None
    return int(number)


def _balance(body: dict) -> AccountBalance:
    if body.get("success") is not True or not isinstance(data := body.get("data"), dict):
        raise AccountQueryError("Unrecognized balance response")
    amounts = AccountBalance(
        available_cny=_number(data.get("availableBalance")),
        spend_cny=_number(data.get("totalSpendAmount")),
        granted_cny=_number(data.get("giveAmount")),
    )
    if all(value is None for value in (amounts.available_cny, amounts.spend_cny, amounts.granted_cny)):
        raise AccountQueryError("Unrecognized balance fields")
    return amounts


def _expiration(value: object) -> datetime | None:
    try:
        if type(value) in (str, int) and str(value).isdigit() and len(str(value)) in (10, 13):
            epoch = int(value)
            return datetime.fromtimestamp(epoch / 1000 if len(str(value)) == 13 else epoch, UTC)
        if isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", value):
            stamp = datetime.fromisoformat(value)
            return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=CONSOLE_TZ)
    except (ValueError, OSError, OverflowError):
        return None
    return None


def _models(row: dict) -> set[str]:
    value = row.get("suitableModel") or row.get("suitableScene")
    names = re.split(r"[,，、]", value) if isinstance(value, str) else value
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        return set()
    names = {name.strip().casefold() for name in names}
    if not all(re.fullmatch(r"glm-[a-z0-9][a-z0-9.\-]*", name) for name in names):
        return set()
    return names


def _packages(body: dict, now: datetime) -> ResourcePackages:
    rows = body.get("rows")
    if body.get("code") != 200 or not isinstance(rows, list):
        raise AccountQueryError("Unrecognized resource package response")
    # The console's total is a row count, separate from any purchased Token allocation.
    complete = _tokens(body.get("total")) == len(rows)
    verified = []
    for row in rows:
        if not isinstance(row, dict):
            complete = False
            continue
        label = str(row.get("resourcePackageName") or "").casefold()
        purpose = str(row.get("tokenPurpose") or "").casefold()
        category = f"{label} {purpose}"
        if any(marker in category for marker in ("coding", "glm code", "编程", "private_instance")):
            continue
        status = row.get("displayStatus") or row.get("status")
        expires_at = _expiration(row.get("packageExpirationTime"))
        if status in ("EXPIRED", "USED_UP") or (expires_at is not None and expires_at <= now):
            continue
        unit = row.get("consumeType")
        models = _models(row)
        if unit == "TIMES" or (models and MODEL not in models):
            continue
        if not label or unit != "TOKENS" or MODEL not in models or status != "EFFECTIVE" or expires_at is None:
            complete = False
            continue
        remaining = _tokens(row.get("tokenBalance"))
        available = _tokens(row.get("availableBalance"))
        if remaining is not None and available is not None and available > remaining:
            available = None
        verified.append(ResourcePackage(remaining, available, expires_at))
    return ResourcePackages(tuple(verified), complete)


def _get(client: httpx.Client, url: str, *, params: dict | None = None) -> dict:
    try:
        client.cookies.clear()
        response = client.get(ORIGIN + url, params=params)
    except httpx.TimeoutException:
        raise AccountQueryError("Network timeout") from None
    except (httpx.RequestError, UnicodeError):
        raise AccountQueryError("Network or credential error") from None
    status = response.status_code
    if status in STATUS_ISSUES:
        raise AccountQueryError(STATUS_ISSUES[status])
    if 300 <= status < 400:
        raise AccountQueryError("Redirect blocked")
    if status != 200:
        raise AccountQueryError("HTTP request failed")
    try:
        body = response.json(parse_float=Decimal)
    except ValueError:
        raise AccountQueryError("Invalid JSON response") from None
    if not isinstance(body, dict):
        raise AccountQueryError("Unrecognized response")
    code = body.get("code")
    if code in (1000, 1001):
        raise AccountQueryError("Authentication required")
    if type(code) is int and code in STATUS_ISSUES:
        raise AccountQueryError(STATUS_ISSUES[code])
    if body.get("success") is False or code not in (None, 200):
        raise AccountQueryError("Console rejected the request")
    return body


def _refresh[T](section: AccountSection[T], fetch: Callable[[], T],
                now: datetime) -> AccountSection[T]:
    if section.is_fresh(now):
        return section
    try:
        return AccountSection(fetch(), datetime.now(UTC))
    except AccountQueryError as error:
        return replace(section, issue=str(error))


def account_snapshot(previous: AccountSnapshot | None = None, *, refresh: bool = False,
                     transport: httpx.BaseTransport | None = None) -> AccountSnapshot:
    """Read the session view; only an explicit refresh can contact BigModel."""
    key = os.getenv("TALITE_API_KEY", "").strip()
    key_tag = sha256(key.encode()).digest() if key else None
    current = previous if previous is not None and previous.key_tag == key_tag else AccountSnapshot(key_tag=key_tag)
    if not refresh:
        return current
    if not key or not re.fullmatch(r"[!-~]+", key):
        issue = "TALITE_API_KEY is missing" if not key else "Invalid TALITE_API_KEY format"
        unavailable = AccountSection(issue=issue)
        return AccountSnapshot(unavailable, unavailable, key_tag)
    now = datetime.now(UTC)
    if current.balance.is_fresh(now) and current.packages.is_fresh(now):
        return current
    with httpx.Client(headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
                      timeout=6, follow_redirects=False, trust_env=False, transport=transport) as client:
        return replace(
            current,
            balance=_refresh(current.balance, lambda: _balance(_get(client, BALANCE_PATH)), now),
            packages=_refresh(current.packages, lambda: _packages(_get(
                client, PACKAGES_PATH, params={"pageNum": 1, "pageSize": 50, "filterEnabled": "false"},
            ), now), now),
        )
