"""Persistent, conservative token accounting for every real LLM HTTP attempt."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import os
from pathlib import Path
import sqlite3
from typing import Any


GLOBAL_TOKEN_LIMIT = 15_000_000
PHASE_TOKEN_LIMITS = {
    "B1": 1_000_000,
    "A1": 2_500_000,
    "A2": 1_500_000,
    "A3": 2_000_000,
    "B2": 1_000_000,
    "B3": 4_000_000,
    "Reserve": 3_000_000,
}
SMOKE_TOKEN_LIMIT = 120_000
PROTOCOL_TOKEN_ALLOWANCE = 128
DEFAULT_LEDGER_PATH = Path(__file__).resolve().parent.parent / "data" / "token-budget.sqlite"


class TokenBudgetExceeded(RuntimeError):
    """Raised before network I/O when a reservation cannot fit the remaining budget."""

    def __init__(self, reason: str, required: int, available: int):
        super().__init__("token_budget_exhausted")
        self.code = "token_budget_exhausted"
        self.reason = reason
        self.required = required
        self.available = available


class TokenBudget:
    """SQLite ledger with atomic pre-call reservations and conservative settlement.

    A failed call or a response without usage remains charged at its full reservation.
    Cached and reasoning tokens are recorded as detail fields; they are not added again
    when the provider includes them in prompt_tokens or completion_tokens.
    """

    def __init__(self, db_path: str | Path | None = None, *,
                 global_limit: int = GLOBAL_TOKEN_LIMIT,
                 phase_limits: dict[str, int] | None = None,
                 experiment_limit: int | None = None):
        configured_path = os.getenv("TALITE_TOKEN_BUDGET_DB")
        self.db_path = Path(db_path or configured_path or DEFAULT_LEDGER_PATH)
        self.global_limit = global_limit
        self.phase_limits = dict(phase_limits or PHASE_TOKEN_LIMITS)
        self.experiment_limit = experiment_limit
        if global_limit <= 0 or any(limit <= 0 for limit in self.phase_limits.values()):
            raise ValueError("Token budget limits must be positive")
        if experiment_limit is not None and experiment_limit <= 0:
            raise ValueError("Experiment token limit must be positive")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS token_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    model TEXT NOT NULL,
                    logical_call INTEGER NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    is_retry INTEGER NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    request_bytes INTEGER NOT NULL,
                    reservation_tokens INTEGER NOT NULL,
                    charged_tokens INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    reasoning_tokens INTEGER,
                    cached_tokens INTEGER,
                    reasoning_in_output INTEGER,
                    usage_known INTEGER NOT NULL DEFAULT 0,
                    usage_missing INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT,
                    deny_reason TEXT
                )
            """)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS token_attempts_phase_idx "
                "ON token_attempts(phase, status)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS token_attempts_experiment_idx "
                "ON token_attempts(experiment_id, status)"
            )

    @staticmethod
    def reservation_size(request_bytes: int, max_output_tokens: int, *,
                         protocol_allowance: int = PROTOCOL_TOKEN_ALLOWANCE,
                         separate_reasoning_tokens: int | None = None) -> int:
        if min(request_bytes, max_output_tokens, protocol_allowance) < 0:
            raise ValueError("Reservation inputs must be non-negative")
        reasoning = max_output_tokens if separate_reasoning_tokens is None else separate_reasoning_tokens
        if reasoning < 0:
            raise ValueError("Reasoning reservation must be non-negative")
        return request_bytes + protocol_allowance + max_output_tokens + reasoning

    @staticmethod
    def _used(connection: sqlite3.Connection, where: str = "", values: tuple = ()) -> int:
        query = "SELECT COALESCE(SUM(charged_tokens), 0) AS total FROM token_attempts"
        if where:
            query += " WHERE " + where
        return int(connection.execute(query, values).fetchone()["total"])

    def capacity(self, *, experiment_id: str, phase: str) -> dict[str, int | bool]:
        if phase not in self.phase_limits:
            raise ValueError(f"Unknown token budget phase: {phase}")
        with self._connection() as connection:
            global_used = self._used(connection)
            phase_used = self._used(connection, "phase = ?", (phase,))
            experiment_used = self._used(connection, "experiment_id = ?", (experiment_id,))
        phase_limit = self.phase_limits[phase]
        experiment_limit = self.experiment_limit
        available = min(
            self.global_limit - global_used,
            phase_limit - phase_used,
            experiment_limit - experiment_used if experiment_limit is not None else self.global_limit,
        )
        return {
            "global_limit": self.global_limit,
            "global_used": global_used,
            "global_remaining": max(0, self.global_limit - global_used),
            "phase": phase,
            "phase_limit": phase_limit,
            "phase_used": phase_used,
            "phase_remaining": max(0, phase_limit - phase_used),
            "experiment_id": experiment_id,
            "experiment_limit": experiment_limit or 0,
            "experiment_used": experiment_used,
            "experiment_remaining": max(0, experiment_limit - experiment_used)
            if experiment_limit is not None else self.global_limit,
            "available": max(0, available),
            "can_continue": available > 0,
        }

    def can_reserve(self, required_tokens: int, *, experiment_id: str, phase: str) -> dict[str, Any]:
        if required_tokens < 0:
            raise ValueError("Required tokens must be non-negative")
        capacity = self.capacity(experiment_id=experiment_id, phase=phase)
        capacity["required_tokens"] = required_tokens
        capacity["fits"] = required_tokens <= capacity["available"]
        return capacity

    def reserve(self, *, experiment_id: str, task_id: str, phase: str, model: str,
                logical_call: int, attempt_number: int, request_sha256: str,
                request_bytes: int, max_output_tokens: int,
                protocol_allowance: int = PROTOCOL_TOKEN_ALLOWANCE,
                separate_reasoning_tokens: int | None = None) -> tuple[int, int]:
        if phase not in self.phase_limits:
            raise ValueError(f"Unknown token budget phase: {phase}")
        reservation = self.reservation_size(
            request_bytes, max_output_tokens, protocol_allowance=protocol_allowance,
            separate_reasoning_tokens=separate_reasoning_tokens,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            global_used = self._used(connection)
            phase_used = self._used(connection, "phase = ?", (phase,))
            experiment_used = self._used(connection, "experiment_id = ?", (experiment_id,))
            remaining = min(
                self.global_limit - global_used,
                self.phase_limits[phase] - phase_used,
                self.experiment_limit - experiment_used if self.experiment_limit is not None else self.global_limit,
            )
            if reservation > remaining:
                if global_used + reservation > self.global_limit:
                    reason = "global_limit"
                elif phase_used + reservation > self.phase_limits[phase]:
                    reason = "phase_limit"
                else:
                    reason = "experiment_limit"
                cursor = connection.execute("""
                    INSERT INTO token_attempts (
                        created_at, experiment_id, task_id, phase, model, logical_call,
                        attempt_number, is_retry, request_sha256, request_bytes,
                        reservation_tokens, charged_tokens, status, usage_known,
                        usage_missing, deny_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'denied', 0, 0, ?)
                """, (datetime.now(UTC).isoformat(), experiment_id, task_id, phase, model,
                      logical_call, attempt_number, int(attempt_number > 1), request_sha256,
                      request_bytes, reservation, reason))
                connection.commit()
                raise TokenBudgetExceeded(reason, reservation, max(0, remaining))
            cursor = connection.execute("""
                INSERT INTO token_attempts (
                    created_at, experiment_id, task_id, phase, model, logical_call,
                    attempt_number, is_retry, request_sha256, request_bytes,
                    reservation_tokens, charged_tokens, status, usage_known, usage_missing
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'reserved', 0, 0)
            """, (datetime.now(UTC).isoformat(), experiment_id, task_id, phase, model,
                  logical_call, attempt_number, int(attempt_number > 1), request_sha256,
                  request_bytes, reservation, reservation))
            connection.commit()
            return int(cursor.lastrowid), reservation
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def settle_usage(self, attempt_id: int, *, input_tokens: int, output_tokens: int,
                     reasoning_tokens: int | None, cached_tokens: int | None,
                     status: str = "succeeded", error_code: str | None = None) -> int:
        if status not in {"succeeded", "failed"}:
            raise ValueError("Usage settlement status must be succeeded or failed")
        prompt, output = max(0, input_tokens), max(0, output_tokens)
        reasoning = max(0, reasoning_tokens) if reasoning_tokens is not None else None
        cached = max(0, cached_tokens) if cached_tokens is not None else None
        reasoning_in_output = reasoning is None or reasoning <= output
        charged = prompt + output + (max(0, reasoning - output) if reasoning is not None else 0)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM token_attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
            if row is None or row["status"] != "reserved":
                connection.rollback()
                raise ValueError("Token reservation is missing or already settled")
            connection.execute("""
                UPDATE token_attempts SET charged_tokens = ?, status = ?, input_tokens = ?,
                    output_tokens = ?, reasoning_tokens = ?, cached_tokens = ?,
                    reasoning_in_output = ?, usage_known = 1, usage_missing = 0, error_code = ?
                WHERE id = ?
            """, (charged, status, prompt, output, reasoning, cached,
                  int(reasoning_in_output), error_code, attempt_id))
        return charged

    def settle_unknown(self, attempt_id: int, *, error_code: str | None,
                       status: str = "usage_missing", input_tokens: int | None = None,
                       output_tokens: int | None = None, reasoning_tokens: int | None = None,
                       cached_tokens: int | None = None) -> int:
        if status not in {"failed", "usage_missing"}:
            raise ValueError("Unknown usage status must be failed or usage_missing")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, reservation_tokens FROM token_attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
            if row is None or row["status"] != "reserved":
                connection.rollback()
                raise ValueError("Token reservation is missing or already settled")
            connection.execute("""
                UPDATE token_attempts SET charged_tokens = reservation_tokens, status = ?,
                    input_tokens = ?, output_tokens = ?, reasoning_tokens = ?, cached_tokens = ?,
                    usage_known = 0, usage_missing = 1, error_code = ? WHERE id = ?
            """, (status,
                  max(0, input_tokens) if input_tokens is not None else None,
                  max(0, output_tokens) if output_tokens is not None else None,
                  max(0, reasoning_tokens) if reasoning_tokens is not None else None,
                  max(0, cached_tokens) if cached_tokens is not None else None,
                  error_code, attempt_id))
            return int(row["reservation_tokens"])

    def snapshot(self, *, experiment_id: str, phase: str) -> dict[str, Any]:
        capacity = self.capacity(experiment_id=experiment_id, phase=phase)
        with self._connection() as connection:
            aggregate = connection.execute("""
                SELECT COUNT(*) AS attempts,
                    SUM(CASE WHEN status != 'denied' THEN 1 ELSE 0 END) AS http_attempts,
                    SUM(CASE WHEN status = 'failed' OR status = 'usage_missing' THEN 1 ELSE 0 END) AS failed_attempts,
                    SUM(CASE WHEN status = 'denied' THEN 1 ELSE 0 END) AS denied_attempts,
                    SUM(CASE WHEN usage_missing = 1 THEN 1 ELSE 0 END) AS missing_usage_attempts,
                    SUM(CASE WHEN usage_known = 1 THEN input_tokens ELSE 0 END) AS input_tokens,
                    SUM(CASE WHEN usage_known = 1 THEN output_tokens ELSE 0 END) AS output_tokens,
                    SUM(CASE WHEN usage_known = 1 THEN reasoning_tokens ELSE 0 END) AS reasoning_tokens,
                    SUM(CASE WHEN usage_known = 1 THEN cached_tokens ELSE 0 END) AS cached_tokens,
                    SUM(CASE WHEN usage_known = 1 AND reasoning_tokens IS NULL THEN 1 ELSE 0 END) AS missing_reasoning_details,
                    SUM(CASE WHEN usage_known = 1 AND cached_tokens IS NULL THEN 1 ELSE 0 END) AS missing_cached_details,
                    SUM(CASE WHEN usage_known = 1 THEN charged_tokens ELSE 0 END) AS usage_known_tokens,
                    SUM(charged_tokens) AS conservatively_charged_tokens
                FROM token_attempts WHERE experiment_id = ?
            """, (experiment_id,)).fetchone()
            retry_count = connection.execute(
                "SELECT COUNT(*) AS total FROM token_attempts "
                "WHERE experiment_id = ? AND is_retry = 1", (experiment_id,)
            ).fetchone()["total"]
        return {
            **capacity,
            "attempts_recorded": aggregate["attempts"] or 0,
            "http_attempts": aggregate["http_attempts"] or 0,
            "failed_attempts": aggregate["failed_attempts"] or 0,
            "denied_attempts": aggregate["denied_attempts"] or 0,
            "missing_usage_attempts": aggregate["missing_usage_attempts"] or 0,
            "retries": retry_count or 0,
            "input_tokens": aggregate["input_tokens"] or 0,
            "output_tokens": aggregate["output_tokens"] or 0,
            "reasoning_tokens": aggregate["reasoning_tokens"] or 0,
            "cached_tokens": aggregate["cached_tokens"] or 0,
            "missing_reasoning_details": aggregate["missing_reasoning_details"] or 0,
            "missing_cached_details": aggregate["missing_cached_details"] or 0,
            "usage_known_tokens": aggregate["usage_known_tokens"] or 0,
            "conservatively_charged_tokens": aggregate["conservatively_charged_tokens"] or 0,
            "ledger": self.db_path.name,
        }

    def attempts(self, *, experiment_id: str | None = None) -> list[dict[str, Any]]:
        with self._connection() as connection:
            if experiment_id is None:
                rows = connection.execute("SELECT * FROM token_attempts ORDER BY id").fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM token_attempts WHERE experiment_id = ? ORDER BY id",
                    (experiment_id,),
                ).fetchall()
        return [dict(row) for row in rows]
