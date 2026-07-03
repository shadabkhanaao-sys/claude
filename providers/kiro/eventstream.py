r"""AWS EventStream binary frame decoder.

AWS EventStream is the binary framing used by CodeWhisperer, Q,
Kinesis, and other AWS streaming APIs.

Frame format::

    +-------------+------------+-------------+----------+-------------+
    | total_len   | hdr_len    | prelude_crc | headers  | payload     |
    | (u32 BE)    | (u32 BE)   | (u32 BE)    | (bytes)  | (bytes)     |
    +-------------+------------+-------------+----------+-------------+
    \--- prelude (12 bytes) ---/             \--- message body ---/
                                            \--- crc (u32 BE, last 4 bytes) ---

Each header:
    name_len (u8) | name (utf-8) | type (u8) | value (type-dependent)

Header types (only those used by CodeWhisperer):
    6 = ByteBuf
    7 = String
    8 = Int16
    9 = Int32
"""

from __future__ import annotations

import json
from typing import Any


# Header value type constants
_HEADER_TYPE_STRING = 7
_HEADER_TYPE_INT16 = 8
_HEADER_TYPE_INT32 = 9
_HEADER_TYPE_BYTE_BUF = 6


class EventStreamFrame:
    """A decoded EventStream frame with headers and optional JSON payload."""

    __slots__ = ("headers", "payload", "event_type", "_raw_payload")

    def __init__(
        self,
        headers: dict[str, Any],
        payload: dict[str, Any] | None,
        event_type: str,
        raw_payload: bytes,
    ):
        self.headers = headers
        self.payload = payload
        self.event_type = event_type
        self._raw_payload = raw_payload

    def __repr__(self) -> str:
        return (
            f"EventStreamFrame(event_type={self.event_type!r}, "
            f"headers={self.headers}, "
            f"payload_keys={list(self.payload) if self.payload else None})"
        )


def decode_eventstream(data: bytes) -> list[EventStreamFrame]:
    """Decode a complete EventStream binary message into frames.

    Args:
        data: Raw binary EventStream payload (may contain multiple frames).

    Returns:
        List of decoded :class:`EventStreamFrame` instances.

    Raises:
        ValueError: On malformed framing or truncated data.
    """
    frames: list[EventStreamFrame] = []
    offset = 0

    while offset < len(data):
        if offset + 12 > len(data):
            break

        total_length = int.from_bytes(data[offset : offset + 4], "big")
        headers_length = int.from_bytes(data[offset + 4 : offset + 8], "big")

        if total_length < 12 or offset + total_length > len(data):
            break

        # Decode headers
        headers: dict[str, Any] = {}
        hdr_offset = offset + 12
        hdr_end = hdr_offset + headers_length

        while hdr_offset < hdr_end:
            consumed, name, value = _decode_header(data, hdr_offset, hdr_end)
            if consumed == 0:
                break
            headers[name] = value
            hdr_offset += consumed

        # Decode payload (everything between headers and CRC)
        payload_start = offset + 12 + headers_length
        payload_end = offset + total_length - 4  # Exclude message CRC
        raw_payload = data[payload_start:payload_end]

        payload: dict[str, Any] | None = None
        if raw_payload.strip():
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError:
                payload = None

        event_type = headers.get(":event-type", headers.get("event-type", "unknown"))

        frames.append(EventStreamFrame(
            headers=headers,
            payload=payload,
            event_type=event_type,
            raw_payload=raw_payload,
        ))

        offset += total_length

    return frames


def _decode_header(
    data: bytes, offset: int, limit: int
) -> tuple[int, str, Any]:
    """Decode one header entry from the binary stream.

    Returns (consumed_bytes, name, value).
    Returns (0, "", None) on error.
    """
    if offset >= limit:
        return 0, "", None

    name_len = data[offset]
    offset += 1
    if offset + name_len > limit or name_len == 0:
        return 0, "", None

    name = data[offset : offset + name_len].decode("utf-8", errors="replace")
    offset += name_len

    if offset >= limit:
        return 0, "", None

    hdr_type = data[offset]
    offset += 1

    if hdr_type == _HEADER_TYPE_STRING:
        if offset + 2 > limit:
            return 1 + name_len + 1, name, ""
        val_len = int.from_bytes(data[offset : offset + 2], "big")
        offset += 2
        if offset + val_len > limit:
            return 1 + name_len + 1 + 2, name, ""
        value = data[offset : offset + val_len].decode("utf-8", errors="replace")
        offset += val_len
        return 1 + name_len + 1 + 2 + val_len, name, value

    elif hdr_type == _HEADER_TYPE_INT16:
        if offset + 2 > limit:
            return 1 + name_len + 1, name, 0
        value = int.from_bytes(data[offset : offset + 2], "big", signed=True)
        offset += 2
        return 1 + name_len + 1 + 2, name, value

    elif hdr_type == _HEADER_TYPE_INT32:
        if offset + 4 > limit:
            return 1 + name_len + 1, name, 0
        value = int.from_bytes(data[offset : offset + 4], "big", signed=True)
        offset += 4
        return 1 + name_len + 1 + 4, name, value

    elif hdr_type == _HEADER_TYPE_BYTE_BUF:
        if offset + 2 > limit:
            return 1 + name_len + 1, name, b""
        val_len = int.from_bytes(data[offset : offset + 2], "big")
        offset += 2
        if offset + val_len > limit:
            return 1 + name_len + 1 + 2, name, b""
        value = data[offset : offset + val_len]
        offset += val_len
        return 1 + name_len + 1 + 2 + val_len, name, value

    else:
        # Unknown header type — skip one byte and hope for the best
        return 1 + name_len + 1, name, None

