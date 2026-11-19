"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    MAX_TEXT_CHARS = 2000  # tránh log phình to vì input dài

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._inputs: dict[str, dict] = {}

    @staticmethod
    def _key(user_id: str, request_id: str | None) -> str:
        return request_id or f"user:{user_id}"

    def _clip(self, text: str | None) -> str:
        text = text or ""
        if len(text) > self.MAX_TEXT_CHARS:
            return text[: self.MAX_TEXT_CHARS] + f"... [truncated {len(text) - self.MAX_TEXT_CHARS} chars]"
        return text

    @staticmethod
    def _redact(text: str) -> str:
        """Audit log không được trở thành nơi rò secret: che secret/PII ở output."""
        try:
            from guardrails.output_guardrails import content_filter

            return content_filter(text)["redacted"]
        except Exception:  # pragma: no cover - log phải luôn ghi được
            return text

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Lưu input + thời điểm bắt đầu theo request_id/user_id. Trả về request_id."""
        request_id = request_id or uuid.uuid4().hex[:12]
        key = self._key(user_id, request_id)
        self._open[key] = time.perf_counter()
        self._inputs[key] = {
            "request_id": request_id,
            "user_id": user_id,
            "input": self._clip(text),
            "input_chars": len(text or ""),
            "started_at": utc_now_iso(),
        }
        # Cho phép record_output(request_id=None) tìm lại request gần nhất của user
        self._open[self._key(user_id, None)] = self._open[key]
        self._inputs[self._key(user_id, None)] = self._inputs[key]
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Lưu output, quyết định theo lớp, latency; append vào self.logs."""
        key = self._key(user_id, request_id)
        meta = self._inputs.pop(key, None)
        started = self._open.pop(key, None)
        if meta is None:  # record_output mà chưa có record_input
            meta = {
                "request_id": request_id,
                "user_id": user_id,
                "input": None,
                "input_chars": 0,
                "started_at": None,
            }
        # dọn cả khoá theo request_id lẫn bản sao theo user (nếu là bản gần nhất)
        if meta.get("request_id"):
            rk = self._key(user_id, meta["request_id"])
            self._inputs.pop(rk, None)
            self._open.pop(rk, None)
        user_key = self._key(user_id, None)
        if self._inputs.get(user_key) is meta:
            self._inputs.pop(user_key, None)
            self._open.pop(user_key, None)

        latency_ms = (
            round((time.perf_counter() - started) * 1000, 2) if started is not None else None
        )
        entry = {
            **meta,
            "output": self._clip(self._redact(text)),
            "blocked": bool(blocked),
            "layer": layer,
            "latency_ms": latency_ms,
            "finished_at": utc_now_iso(),
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
