from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest

from tradingagents_lite.zhipu_account import (
    BALANCE_PATH, CACHE_TTL, MODEL, PACKAGES_PATH, account_snapshot,
)


@pytest.fixture(autouse=True)
def local_key(monkeypatch):
    monkeypatch.setenv("TALITE_API_KEY", "mock-account-key")


def package(**changes):
    return {
        "resourcePackageName": "Model inference package",
        "displayStatus": "EFFECTIVE",
        "consumeType": "TOKENS",
        "suitableModel": MODEL,
        "packageExpirationTime": "2099-04-05 23:00:00",
        "tokenBalance": "2100",
        "availableBalance": 2000,
        **changes,
    }


def test_console_response_uses_fixed_https_bearer_and_disjoint_units(monkeypatch):
    monkeypatch.setenv("TALITE_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("HTTPS_PROXY", "http://example.invalid:9000")
    calls = []

    def handler(request):
        calls.append(request.url.path)
        assert request.method == "GET" and request.url.scheme == "https"
        assert request.url.host == "open.bigmodel.cn"
        assert request.headers["Authorization"] == "Bearer mock-account-key"
        assert "cookie" not in request.headers
        if request.url.path == BALANCE_PATH:
            return httpx.Response(200, headers={"Set-Cookie": "ignored=1"}, json={
                "code": 200, "success": True,
                "data": {"availableBalance": "0.00", "totalSpendAmount": 4.25, "giveAmount": "1.00"},
            })
        assert dict(request.url.params) == {"pageNum": "1", "pageSize": "50", "filterEnabled": "false"}
        rows = [
            package(),
            package(packageExpirationTime="2000-01-01T00:00:00+08:00"),
            package(displayStatus="EXPIRED"),
            package(suitableModel="glm-5.3-trial"),
            package(resourcePackageName="GLM Coding Plan"),
            package(consumeType="TIMES"),
        ]
        return httpx.Response(200, json={"code": 200, "rows": rows, "total": len(rows)})

    transport = httpx.MockTransport(handler)
    initial = account_snapshot(transport=transport)
    assert calls == [] and initial.balance.value is None
    result = account_snapshot(initial, refresh=True, transport=transport)
    assert calls == [BALANCE_PATH, PACKAGES_PATH]
    assert result.balance.value.available_cny == Decimal("0.00")
    assert result.balance.value.spend_cny == Decimal("4.25")
    assert result.balance.value.granted_cny == Decimal("1.00")
    assert result.packages.value.complete
    active, = result.packages.value.active(datetime.now(UTC))
    assert (active.remaining_tokens, active.available_tokens) == (2100, 2000)
    assert active.expires_at.astimezone(UTC).isoformat() == "2099-04-05T15:00:00+00:00"
    assert result.packages.value.active(active.expires_at) == ()
    assert account_snapshot(result, refresh=True, transport=transport) is result
    assert calls == [BALANCE_PATH, PACKAGES_PATH]


def test_missing_and_changed_key_discard_private_cache(monkeypatch):
    transport = httpx.MockTransport(lambda request: httpx.Response(
        200, json={"success": True, "data": {"availableBalance": 0}},
    ))
    original = account_snapshot(refresh=True, transport=transport)
    assert original.balance.value is not None
    monkeypatch.delenv("TALITE_API_KEY")
    result = account_snapshot(original, refresh=True, transport=httpx.MockTransport(
        lambda request: pytest.fail("Missing credential must not trigger a request"),
    ))
    assert result.balance.value is None
    assert result.balance.issue == "TALITE_API_KEY is missing"
    monkeypatch.setenv("TALITE_API_KEY", "another-account-key")
    assert account_snapshot(original).balance.value is None


@pytest.mark.parametrize(("status", "issue"), [
    (401, "Authentication required"), (403, "Access denied"), (429, "Rate limited"),
    (307, "Redirect blocked"),
])
def test_errors_and_redirect_are_sanitized(status, issue):
    def handler(request):
        assert request.url.host == "open.bigmodel.cn"
        return httpx.Response(
            status, headers={"Location": "https://example.invalid"}, text="private response",
        )

    snapshot = account_snapshot(refresh=True, transport=httpx.MockTransport(handler))
    for section in (snapshot.balance, snapshot.packages):
        assert section.issue == issue and section.synced_at is None and section.value is None


def test_stale_data_survives_timeout_and_sections_refresh_independently():
    snapshot = account_snapshot(refresh=True, transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=(
            {"success": True, "data": {"availableBalance": 7}} if request.url.path == BALANCE_PATH
            else {"code": 200, "rows": [], "total": 0}
        )),
    ))
    old_balance = replace(snapshot.balance, synced_at=datetime.now(UTC) - CACHE_TTL)
    old_snapshot = replace(snapshot, balance=old_balance)

    def timeout(request):
        assert request.url.path == BALANCE_PATH
        raise httpx.ReadTimeout("private response")

    stale = account_snapshot(old_snapshot, refresh=True, transport=httpx.MockTransport(timeout))
    assert stale.balance.issue == "Network timeout"
    assert stale.balance.value == old_balance.value and stale.balance.synced_at == old_balance.synced_at
    assert not stale.balance.is_fresh(datetime.now(UTC))
    assert stale.packages == snapshot.packages


@pytest.mark.parametrize(("payload", "issue"), [
    ({"code": 1001, "success": False}, "Authentication required"),
    ({"success": True, "data": {"newAmount": 0}}, "Unrecognized balance fields"),
    ({"success": True, "data": [{"availableBalance": 0}]}, "Unrecognized balance response"),
])
def test_business_error_or_schema_change(payload, issue):
    result = account_snapshot(refresh=True, transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=payload),
    ))
    assert result.balance.value is None and result.balance.issue == issue
    assert result.packages.value is None


def test_partial_balances_and_uncertain_package_scope():
    packages = [
        package(suitableModel="不适用于 GLM-5.3-Flash"),
        package(suitableModel="所有按 tokens 计费的模型"),
        package(packageExpirationTime="2099-04-05"),
        package(availableBalance=None),
    ]
    result = account_snapshot(refresh=True, transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=(
            {"success": True, "data": {"availableBalance": None, "totalSpendAmount": 0}}
            if request.url.path == BALANCE_PATH else {"code": 200, "rows": packages, "total": 20}
        )),
    ))
    assert result.balance.value.available_cny is None
    assert result.balance.value.spend_cny == Decimal(0)
    assert not result.packages.value.complete
    active, = result.packages.value.active(datetime.now(UTC))
    assert active.available_tokens is None


def test_flash_scope_accepts_console_active_rows_and_excludes_other_models():
    rows = [
        package(status="EFFECTIVE", displayStatus="ACTIVE", tokenBalance=12345,
                availableBalance=12000, suitableModel="glm-5.3-flash"),
        package(suitableModel="glm-5.3-flashx", tokenBalance=3000000),
        package(suitableModel="glm-5.3", tokenBalance=2000000),
        package(suitableModel="all token models"),
    ]
    result = account_snapshot(refresh=True, transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=(
            {"success": True, "data": {"availableBalance": 0}}
            if request.url.path == BALANCE_PATH else {"code": 200, "rows": rows, "total": len(rows)}
        )),
    ))
    assert result.packages.value.complete is False
    active, = result.packages.value.active(datetime.now(UTC))
    assert (active.remaining_tokens, active.available_tokens) == (12345, 12000)
