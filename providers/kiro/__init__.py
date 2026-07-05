"""Kiro AI provider — free AI provider with multi-account support.

Uses AWS CodeWhisperer's ``GenerateAssistantResponse`` API via the
Kiro runtime endpoint (``runtime.us-east-1.kiro.dev``) with the
AWS EventStream binary protocol.

Supported authentication methods:
- Kiro OAuth login (device code flow)
- AWS Builder ID
- AWS IAM Identity Center
- Import Token (direct)

Multiple accounts can be configured as a JSON session list in the KIRO_SESSIONS
environment variable, each with a priority for fallback ordering.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from config.settings import Settings
from providers.base import BaseProvider, ProviderConfig

from .transport import (
    DEFAULT_PROFILE_ARN,
    kiro_stream,
    list_kiro_models_via_q_api,
)


class KiroProvider(BaseProvider):
    """Kiro AI provider with multi-account priority-based fallback.

    Uses the CodeWhisperer ``GenerateAssistantResponse`` EventStream protocol
    instead of the OpenAI Chat Completions API.

    Sessions are loaded from ``settings.kiro_sessions`` (JSON array) and
    optionally from ``settings.kiro_access_token`` as a single-token fallback.

    Each session dict supports:
    - ``access_token`` (str) — the API token
    - ``auth_type`` (str) — one of ``oauth``, ``aws_builder_id``,
      ``aws_iam_identity_center``, ``import``
    - ``priority`` (int) — higher values tried first (default 50)
    - ``name`` (str) — optional human label for display
    """

    def __init__(self, config: ProviderConfig, settings: Settings | None = None):
        super().__init__(config)
        self._settings = settings
        self._sessions: list[dict[str, Any]] = []
        self._http_client: httpx.AsyncClient | None = None
        self._load_sessions()

    def _load_sessions(self) -> None:
        """Load and sort sessions from settings, falling back to single token."""
        sessions: list[dict[str, Any]] = []

        if self._settings and self._settings.kiro_sessions:
            try:
                parsed = json.loads(self._settings.kiro_sessions)
                if isinstance(parsed, list):
                    for s in parsed:
                        token = s.get("access_token", "")
                        if token and token.strip():
                            sessions.append(s)
            except (json.JSONDecodeError, TypeError):
                pass

        token = self._config.api_key
        if token and token.strip():
            if not any(s.get("access_token") == token for s in sessions):
                sessions.append({
                    "id": str(uuid.uuid4()),
                    "name": "Default Token",
                    "access_token": token,
                    "auth_type": "import",
                    "priority": 50,
                })

        sessions.sort(key=lambda s: s.get("priority", 0) or 0, reverse=True)
        self._sessions = sessions

    def _get_profile_arn(self) -> str:
        """Return the profile ARN from config, or the Builder ID default."""
        return (
            getattr(self._config, "kiro_profile_arn", None)
            or DEFAULT_PROFILE_ARN
        )

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream with multi-session fallback on auth/rate-limit errors."""
        if not self._sessions:
            from providers.exceptions import AuthenticationError
            raise AuthenticationError(
                "No Kiro sessions configured. Add a token or OAuth session "
                "via KIRO_ACCESS_TOKEN or KIRO_SESSIONS env vars."
            )

        thinking = self._is_thinking_enabled(request, thinking_enabled)
        profile_arn = self._get_profile_arn()

        last_error: Exception | None = None
        for idx, session in enumerate(self._sessions):
            token = session.get("access_token", "")
            if not token or not token.strip():
                continue

            try:
                async for event in kiro_stream(
                    token=token,
                    request=request,
                    input_tokens=input_tokens,
                    request_id=request_id,
                    thinking_enabled=thinking,
                    profile_arn=profile_arn,
                    timeout=httpx.Timeout(
                        self._config.http_read_timeout,
                        connect=self._config.http_connect_timeout,
                    ),
                ):
                    yield event
                return

            except Exception as exc:
                last_error = exc
                status = _extract_status(exc)

                # Auth / rate-limit / server errors -> try next session
                if status in (401, 403, 429) or (status and 500 <= status < 600):
                    from loguru import logger
                    session_name = session.get("name", f"session-{idx}")
                    logger.warning(
                        "KIRO session '{}' failed (HTTP {}), trying next session "
                        "({}/{} remaining)",
                        session_name, status,
                        len(self._sessions) - idx - 1,
                        len(self._sessions),
                    )
                    continue
                raise

        if last_error is not None:
            raise last_error
        from providers.exceptions import ServiceUnavailableError
        raise ServiceUnavailableError("All Kiro sessions failed")

    async def list_model_ids(self) -> frozenset[str]:
        """Return Kiro model IDs from the Q API, with fallback defaults."""
        if not self._sessions:
            return self._fallback_model_ids()

        token = self._sessions[0].get("access_token", "")
        if not token:
            return self._fallback_model_ids()

        profile_arn = self._get_profile_arn()
        try:
            models = await list_kiro_models_via_q_api(
                token=token,
                profile_arn=profile_arn,
                timeout_s=15.0,
            )
            if models:
                ids: set[str] = set()
                for m in models:
                    mid = m.get("modelId", "") or m.get("modelName", "")
                    if mid:
                        ids.add(mid)
                if ids:
                    return frozenset(ids)
        except Exception:
            pass

        return self._fallback_model_ids()

    def _fallback_model_ids(self) -> frozenset[str]:
        """Return known Kiro model IDs as static fallback."""
        return frozenset({
            "claude-sonnet-4.5",
            "claude-opus-4.5",
            "claude-haiku-4.5",
            "glm-5",
            "minimax-text-01",
            "deepseek-chat",
            "deepseek-reasoner",
        })

    async def cleanup(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


def _extract_status(exc: Exception) -> int | None:
    """Extract HTTP status code from an exception."""
    response = getattr(exc, "response", None)
    if response is not None:
        return getattr(response, "status_code", None)
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status
    return getattr(exc, "status", None)
