"""Kiro AI / AWS CodeWhisperer streaming transport.

Converts Anthropic Messages API requests to CodeWhisperer
``GenerateAssistantResponse`` payloads, sends them over EventStream,
and converts the binary frames back to Anthropic SSE events.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from loguru import logger

from core.anthropic.streaming import AnthropicStreamLedger
from providers.kiro.eventstream import EventStreamFrame, decode_eventstream

DEFAULT_PROFILE_ARN = "arn:aws:codewhisperer:us-east-1:638616132270:profile/AAAACCCCXXXX"

KIRO_RUNTIME_URL = "https://runtime.us-east-1.kiro.dev/generateAssistantResponse"

_DEFAULT_HEADERS: dict[str, str] = {
    "Content-Type": "application/json",
    "Accept": "application/vnd.amazon.eventstream",
    "X-Amz-Target": "AmazonCodeWhispererStreamingService.GenerateAssistantResponse",
    "User-Agent": (
        "aws-sdk-js/1.0.0 ua/2.1 os/windows#10.0.26200 "
        "lang/js md/nodejs#22.21.1 api/codewhispererruntime#1.0.0 m/N,E"
    ),
    "x-amz-user-agent": "aws-sdk-js/1.0.0",
    "x-amzn-kiro-agent-mode": "vibe",
    "x-amzn-codewhisperer-optout": "true",
    "amz-sdk-request": "attempt=1; max=1",
}

_STOP_REASON_MAP: dict[str, str] = {
    "END_TURN": "end_turn",
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
    "CONTENT_FILTERED": "stop_sequence",
    "ERROR": "error",
    "TOOL_USE": "tool_use",
}


def build_session_headers(token: str) -> dict[str, str]:
    headers = dict(_DEFAULT_HEADERS)
    headers["Authorization"] = f"Bearer {token}"
    headers["amz-sdk-invocation-id"] = str(uuid.uuid4())
    return headers


def _resolve_kiro_model(model: str) -> str:
    upstream = model
    for suffix in ("-thinking-agentic", "-agentic", "-thinking"):
        if upstream.endswith(suffix):
            upstream = upstream[: -len(suffix)]
            break
    return upstream


def build_kiro_payload(
    model: str,
    messages: list[Any],
    system: str | list | None = None,
    tools: list[Any] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    thinking_enabled: bool = False,
    thinking_budget: int = 16000,
    profile_arn: str = DEFAULT_PROFILE_ARN,
) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    current_content: str = ""
    has_current = False

    for msg in messages:
        role = _msg_role(msg)
        content = _get_message_content(msg)
        if role == "user":
            if has_current:
                history.append({"userInputMessage": {"content": current_content, "modelId": model}})
            current_content = content
            has_current = True
        elif role == "assistant":
            if has_current:
                history.append({"userInputMessage": {"content": current_content, "modelId": model}})
                current_content = ""
            history.append({"assistantResponseMessage": {"content": content}})

    full_content = _build_system_prefix(system, thinking_enabled, thinking_budget)
    if current_content:
        full_content = (full_content + "\n\n" if full_content else "") + current_content

    current_msg: dict[str, Any] = {
        "userInputMessage": {"content": full_content or "continue", "modelId": model, "origin": "AI_EDITOR"}
    }
    if tools:
        specs = _convert_tools(tools)
        if specs:
            current_msg["userInputMessage"]["userInputMessageContext"] = {"tools": specs}

    payload: dict[str, Any] = {
        "conversationState": {
            "chatTriggerType": "MANUAL",
            "conversationId": str(uuid.uuid4()),
            "currentMessage": current_msg,
            "history": history,
        },
    }
    if profile_arn:
        payload["profileArn"] = profile_arn
    inf: dict[str, Any] = {}
    if max_tokens:
        inf["maxTokens"] = max_tokens
    if temperature is not None:
        inf["temperature"] = temperature
    if top_p is not None:
        inf["topP"] = top_p
    if inf:
        payload["inferenceConfig"] = inf
    return payload


def _msg_role(msg: Any) -> str:
    return msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")


def _get_message_content(msg: Any) -> str:
    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                bt = block.get("type", "")
                if bt == "text":
                    texts.append(block.get("text", ""))
                elif bt == "tool_use":
                    inp = json.dumps(block.get("input", {})) if block.get("input") else ""
                    texts.append(f"[Tool call: {block.get('name', '')}({inp})]")
                elif bt == "tool_result":
                    texts.append(f"[Tool result: {_extract_tool_result_text(block.get('content', ''))}]")
            elif hasattr(block, "type"):
                if block.type == "text" and hasattr(block, "text"):
                    texts.append(block.text)
        return "\n\n".join(texts)
    return str(content)


def _extract_tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") if isinstance(c, dict) and "text" in c else str(c) for c in content)
    return str(content)


def _build_system_prefix(system: str | list | None, thinking_enabled: bool, thinking_budget: int) -> str:
    parts: list[str] = []
    if thinking_enabled:
        safe = max(1, min(32000, thinking_budget))
        parts.append(f"<thinking_mode>enabled</thinking_mode>\n<max_thinking_length>{safe}</max_thinking_length>")
    if system:
        if isinstance(system, str):
            parts.append(system)
        elif isinstance(system, list):
            texts = [s.get("text", "") if isinstance(s, dict) else str(s) for s in system]
            parts.append("\n".join(t for t in texts if t))
    return "\n\n".join(parts)


def _convert_tools(tools: list[Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name", "")
            desc = tool.get("description", f"Tool: {name}")
            schema = tool.get("input_schema", {})
        else:
            name = getattr(tool, "name", "")
            desc = getattr(tool, "description", f"Tool: {name}")
            schema = getattr(tool, "input_schema", {})
        if not name:
            continue
        specs.append({
            "toolSpecification": {
                "name": name,
                "description": desc,
                "inputSchema": {"json": schema or {"type": "object", "properties": {}, "required": []}},
            }
        })
    return specs


# =============================================================================
# Response streamer
# =============================================================================


class KiroResponseStreamer:
    """Read an EventStream response from the Kiro runtime and emit Annie SSE."""

    def __init__(self, response: httpx.Response, request_model: str, input_tokens: int = 0):
        self._response = response
        self._ledger = AnthropicStreamLedger(f"msg_{uuid.uuid4().hex[:12]}", request_model, input_tokens)
        self._buffer = bytearray()
        self._accumulated_text = ""
        self._completion_tokens: int = 0
        self._tool_index = 0

    async def stream_sse(self) -> AsyncIterator[str]:
        yield self._ledger.message_start()
        try:
            async for chunk in self._response.aiter_bytes():
                self._buffer.extend(chunk)
                for ev in self._drain_frames():
                    yield ev
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Kiro EventStream read error: {}", exc)
            if self._buffer:
                try:
                    for fr in decode_eventstream(bytes(self._buffer)):
                        for ev in self._frame_events(fr):
                            yield ev
                except Exception:
                    pass
        if not self._ledger.has_terminal_message():
            for ev in self._stop_events("END_TURN"):
                yield ev

    def _drain_frames(self) -> list[str]:
        """Parse complete frames from the buffer and return SSE event strings."""
        data = bytes(self._buffer)
        offset = 0
        events: list[str] = []
        while offset < len(data):
            if offset + 12 > len(data):
                break
            tl = int.from_bytes(data[offset:offset+4], "big")
            if tl < 12 or offset + tl > len(data):
                break
            try:
                for fr in decode_eventstream(data[offset:offset+tl]):
                    events.extend(self._frame_events(fr))
            except Exception:
                pass
            offset += tl
        self._buffer = bytearray(data[offset:])
        return events

    def _frame_events(self, fr: EventStreamFrame) -> list[str]:
        """Convert one frame to SSE event strings."""
        ev = fr.event_type
        p = fr.payload or {}
        out: list[str] = []

        if ev == "assistantResponseEvent":
            text = p.get("content", "")
            if text:
                out.extend(self._ledger.ensure_text_block())
                out.append(self._ledger.emit_text_delta(text))
                self._accumulated_text += text

        elif ev == "reasoningContentEvent":
            r = p.get("reasoningContentEvent", p)
            rtext = r.get("text", "") if isinstance(r, dict) else (str(r) if r else "")
            if rtext:
                out.extend(self._ledger.ensure_thinking_block())
                out.append(self._ledger.emit_thinking_delta(rtext))

        elif ev == "toolUseEvent":
            tus = p.get("toolUse", p)
            if not isinstance(tus, list):
                tus = [tus]
            for tu in tus:
                if not isinstance(tu, dict):
                    continue
                tid = tu.get("toolUseId", f"toolu_{uuid.uuid4().hex[:12]}")
                tname = tu.get("name", "")
                tinp = tu.get("input", {})
                idx = self._tool_index
                self._tool_index += 1
                out.extend(self._ledger.close_content_blocks())
                out.append(self._ledger.start_tool_block(idx, tid, tname))
                out.append(self._ledger.emit_tool_delta(idx, json.dumps(tinp) if tinp else "{}"))
                out.append(self._ledger.stop_tool_block(idx))

        elif ev in ("metadataEvent",):
            sr = p.get("stopReason", "")
            if sr:
                out.extend(self._stop_events(sr))

        return out

    def _stop_events(self, stop_reason_raw: str) -> list[str]:
        if self._ledger.has_terminal_message():
            return []
        reason = _STOP_REASON_MAP.get(stop_reason_raw.upper(), "end_turn")
        ct = self._completion_tokens or max(1, len(self._accumulated_text) // 4) if self._accumulated_text else 0
        out: list[str] = []
        out.extend(self._ledger.close_all_blocks())
        out.append(self._ledger.message_delta(reason, output_tokens=ct))
        out.append(self._ledger.message_stop())
        return out


async def kiro_stream(
    token: str,
    request: Any,
    input_tokens: int = 0,
    request_id: str | None = None,
    thinking_enabled: bool = False,
    profile_arn: str = DEFAULT_PROFILE_ARN,
    timeout: httpx.Timeout | None = None,
) -> AsyncIterator[str]:
    model = _resolve_kiro_model(getattr(request, "model", "auto"))
    messages = getattr(request, "messages", [])
    system = getattr(request, "system", None)
    tools = getattr(request, "tools", None)
    max_tokens = getattr(request, "max_tokens", 4096)
    temperature = getattr(request, "temperature", None)
    top_p = getattr(request, "top_p", None)
    extra_body = getattr(request, "extra_body", None) or {}
    if isinstance(extra_body, dict) and "kiro_profile_arn" in extra_body:
        profile_arn = extra_body["kiro_profile_arn"]
    thinking_budget = 16000
    thinking = getattr(request, "thinking", None)
    if thinking:
        budget = thinking.get("budget_tokens") if isinstance(thinking, dict) else getattr(thinking, "budget_tokens", None)
        if budget:
            thinking_budget = int(budget)
    payload = build_kiro_payload(
        model=model, messages=messages, system=system, tools=tools,
        max_tokens=max_tokens, temperature=temperature, top_p=top_p,
        thinking_enabled=thinking_enabled, thinking_budget=thinking_budget,
        profile_arn=profile_arn,
    )
    headers = build_session_headers(token)
    async with httpx.AsyncClient(timeout=timeout or httpx.Timeout(300.0, connect=10.0)) as client:
        response = await client.post(KIRO_RUNTIME_URL, json=payload, headers=headers)
        streamer = KiroResponseStreamer(response, request_model=model, input_tokens=input_tokens)
        async for ev in streamer.stream_sse():
            yield ev


async def list_kiro_models_via_q_api(
    token: str, profile_arn: str = DEFAULT_PROFILE_ARN, timeout_s: float = 15.0,
) -> list[dict[str, Any]] | None:
    url = f"https://q.us-east-1.amazonaws.com/ListAvailableModels?origin=AI_EDITOR"
    if profile_arn:
        url += f"&profileArn={profile_arn}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "aws-sdk-js/1.0.0 ua/2.1 os/windows#10.0.26200 lang/js md/nodejs#22.21.1",
        "x-amz-user-agent": "aws-sdk-js/1.0.0",
        "x-amzn-kiro-agent-mode": "vibe",
        "x-amzn-codewhisperer-optout": "true",
        "amz-sdk-invocation-id": str(uuid.uuid4()),
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                return resp.json().get("models", [])
            logger.warning("Kiro ListAvailableModels returned HTTP {}: {}", resp.status_code, resp.text[:300])
            return None
    except Exception as exc:
        logger.warning("Kiro ListAvailableModels failed: {}", exc)
        return None
