"""Unit tests for sanitize_tool_result_user_messages (mixed tool_result+text fold)."""

from __future__ import annotations

from core.anthropic.native_messages_request import sanitize_tool_result_user_messages


def test_tool_result_only_turn_passes_through_unchanged() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "42"},
            ],
        }
    ]
    out = sanitize_tool_result_user_messages(messages)
    assert out == messages


def test_string_content_tool_result_folds_trailing_text_by_joining() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "42"},
                {"type": "text", "text": "CRITICAL: respond with text only"},
            ],
        }
    ]
    out = sanitize_tool_result_user_messages(messages)
    assert len(out[0]["content"]) == 1
    result = out[0]["content"][0]
    assert result["type"] == "tool_result"
    assert result["content"] == "42\n\nCRITICAL: respond with text only"


def test_list_content_tool_result_folds_extras_as_appended_blocks() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "output"}],
                },
                {"type": "text", "text": "reminder"},
            ],
        }
    ]
    out = sanitize_tool_result_user_messages(messages)
    result = out[0]["content"][0]
    assert result["type"] == "tool_result"
    assert result["content"] == [
        {"type": "text", "text": "output"},
        {"type": "text", "text": "reminder"},
    ]


def test_multiple_tool_results_fold_extras_into_last_one() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "a"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "b"},
                {"type": "text", "text": "reminder"},
            ],
        }
    ]
    out = sanitize_tool_result_user_messages(messages)
    content = out[0]["content"]
    assert len(content) == 2
    assert content[0]["content"] == "a"
    assert content[1]["content"] == "b\n\nreminder"


def test_plain_text_and_assistant_messages_are_no_ops() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]
    assert sanitize_tool_result_user_messages(messages) == messages


def test_non_list_input_is_returned_unchanged() -> None:
    assert sanitize_tool_result_user_messages("not-a-list") == "not-a-list"
    assert sanitize_tool_result_user_messages(None) is None


def test_non_tool_result_extras_are_preserved_as_separate_blocks() -> None:
    """A non-text extra (e.g. an image) is appended, not merged into text."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "a"},
                {"type": "image", "source": {"type": "base64", "data": "..."}},
            ],
        }
    ]
    out = sanitize_tool_result_user_messages(messages)
    result = out[0]["content"][0]
    assert result["content"] == [
        {"type": "text", "text": "a"},
        {"type": "image", "source": {"type": "base64", "data": "..."}},
    ]
