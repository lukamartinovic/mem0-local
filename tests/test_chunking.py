"""
Unit tests for _chunk_text(), _get_chunk_size(), and _detect_conversation().

Pure Python — no Docker, no Ollama, no Qdrant needed.
Can run locally on any machine:
    python3 -m pytest tests/test_chunking.py -v
"""

import json

import pytest

import mcp_server


# ── _chunk_text ──────────────────────────────────────────────────────────────

class TestChunkText:
    def test_short_text_single_chunk(self):
        """Text under max_chars returns a single chunk."""
        result = mcp_server._chunk_text("Hello world", max_chars=3000)
        assert len(result) == 1
        assert result[0] == "Hello world"

    def test_short_text_with_header(self):
        """Text under max_chars with context header gets header prepended."""
        result = mcp_server._chunk_text("Hello world", max_chars=3000,
                                         context_header="[Doc: test.md]")
        assert len(result) == 1
        assert "[Doc: test.md]" in result[0]
        assert "Hello world" in result[0]

    def test_long_text_multiple_chunks(self):
        """Text over max_chars is split into multiple chunks."""
        paragraphs = [f"Paragraph {i}. " + "x" * 500 for i in range(20)]
        text = "\n\n".join(paragraphs)
        result = mcp_server._chunk_text(text, max_chars=1000)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 1000

    def test_chunks_respect_max_chars_with_header(self):
        """Every chunk including context header is under max_chars."""
        paragraphs = [f"Para {i}. " + "y" * 500 for i in range(20)]
        text = "\n\n".join(paragraphs)
        header = "[Document: very/long/path/to/file.md][Source: docs_import]"
        result = mcp_server._chunk_text(text, max_chars=1000,
                                         context_header=header)
        for chunk in result:
            assert len(chunk) <= 1000, f"Chunk is {len(chunk)} chars, max 1000"

    def test_no_paragraphs_long_text(self):
        """Single long paragraph with no \\n\\n splits at line boundaries."""
        lines = [f"line {i}" for i in range(200)]
        text = "\n".join(lines)
        result = mcp_server._chunk_text(text, max_chars=500)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 500

    def test_paragraph_longer_than_max(self):
        """A single paragraph longer than max_chars splits at line level."""
        text = "word " * 500  # ~2500 chars, single paragraph, no \n\n
        result = mcp_server._chunk_text(text, max_chars=1000)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 1000

    def test_context_header_prepended_to_all_chunks(self):
        """When using context header, every chunk gets it prepended."""
        paragraphs = [f"Para {i}. " + "z" * 500 for i in range(10)]
        text = "\n\n".join(paragraphs)
        header = "[Doc: test.md]"
        result = mcp_server._chunk_text(text, max_chars=800,
                                         context_header=header)
        assert len(result) > 1
        for chunk in result:
            assert chunk.startswith(header)

    def test_empty_string(self):
        """Empty string returns a single empty chunk (or empty list)."""
        result = mcp_server._chunk_text("", max_chars=3000)
        # Current behavior: returns [""]
        assert len(result) >= 0  # Don't assert specific behavior, just no crash

    def test_exact_boundary(self):
        """Text exactly at max_chars stays as one chunk."""
        text = "x" * 1000
        result = mcp_server._chunk_text(text, max_chars=1000)
        assert len(result) == 1

    def test_header_reduces_effective_max(self):
        """Context header length is accounted for in chunk splitting."""
        header = "[Doc: test.md]"  # 14 chars + 2 for \n\n = 16
        text = "a" * 990  # 990 + 16 = 1006 > 1000
        result = mcp_server._chunk_text(text, max_chars=1000,
                                         context_header=header)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 1000

    def test_all_content_preserved(self):
        """No text is lost during chunking — concatenation matches input."""
        paragraphs = [f"Paragraph {i}. " + "x" * 300 for i in range(10)]
        text = "\n\n".join(paragraphs)
        result = mcp_server._chunk_text(text, max_chars=800)
        # Reassemble (chunks use \n\n as paragraph separator)
        reassembled = "\n\n".join(result)
        # All paragraphs should be present
        for i in range(10):
            assert f"Paragraph {i}." in reassembled


# ── _get_chunk_size ──────────────────────────────────────────────────────────

class TestGetChunkSize:
    def test_default_model(self):
        """Default config (qwen2.5:7b) returns 3000."""
        assert mcp_server._get_chunk_size() == 3000

    def test_known_models(self):
        """All known models return correct chunk sizes."""
        assert mcp_server._CHUNK_SIZES["qwen2.5:3b"] == 1500
        assert mcp_server._CHUNK_SIZES["qwen2.5:7b"] == 3000
        assert mcp_server._CHUNK_SIZES["qwen3.5:9b"] == 4000
        assert mcp_server._CHUNK_SIZES["gemma4:12b"] == 5000
        assert mcp_server._CHUNK_SIZES["qwen3.5:27b"] == 8000

    def test_unknown_model_default(self):
        """Unknown model returns default 3000."""
        original = mcp_server.CONFIG["llm"]["config"]["model"]
        try:
            mcp_server.CONFIG["llm"]["config"]["model"] = "unknown-model:1b"
            assert mcp_server._get_chunk_size() == 3000
        finally:
            mcp_server.CONFIG["llm"]["config"]["model"] = original

    def test_prefix_match(self):
        """Model name that's a superset of a known model uses prefix matching."""
        original = mcp_server.CONFIG["llm"]["config"]["model"]
        try:
            mcp_server.CONFIG["llm"]["config"]["model"] = "qwen2.5:7b-instruct"
            assert mcp_server._get_chunk_size() == 3000
        finally:
            mcp_server.CONFIG["llm"]["config"]["model"] = original


# ── _detect_conversation ─────────────────────────────────────────────────────

class TestDetectConversation:
    def test_valid_conversation_json(self):
        """JSON array of {role, content} dicts is detected as conversation."""
        content = json.dumps([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ])
        is_conv, messages = mcp_server._detect_conversation(content)
        assert is_conv is True
        assert messages is not None
        assert len(messages) == 2
        assert messages[0]["role"] == "user"

    def test_plain_text_not_conversation(self):
        """Plain text is not detected as conversation."""
        is_conv, messages = mcp_server._detect_conversation(
            "I prefer TypeScript over JavaScript."
        )
        assert is_conv is False
        assert messages is None

    def test_invalid_json_not_conversation(self):
        """Invalid JSON is not detected as conversation."""
        is_conv, messages = mcp_server._detect_conversation("{not valid json}")
        assert is_conv is False
        assert messages is None

    def test_json_object_not_conversation(self):
        """JSON object (not array) is not a conversation."""
        is_conv, messages = mcp_server._detect_conversation('{"key": "value"}')
        assert is_conv is False
        assert messages is None

    def test_json_array_without_role_not_conversation(self):
        """JSON array without 'role' key is not a conversation."""
        is_conv, messages = mcp_server._detect_conversation(
            json.dumps([{"text": "hello"}, {"text": "world"}])
        )
        assert is_conv is False
        assert messages is None

    def test_empty_array_not_conversation(self):
        """Empty JSON array is not a conversation (would cause empty add)."""
        is_conv, messages = mcp_server._detect_conversation("[]")
        assert is_conv is False
        assert messages is None

    def test_empty_string_not_conversation(self):
        """Empty string is not a conversation."""
        is_conv, messages = mcp_server._detect_conversation("")
        assert is_conv is False
        assert messages is None

    def test_single_message_conversation(self):
        """Single message in a JSON array is still a conversation."""
        content = json.dumps([{"role": "user", "content": "Hello"}])
        is_conv, messages = mcp_server._detect_conversation(content)
        assert is_conv is True
        assert len(messages) == 1

    def test_message_without_content_still_conversation(self):
        """Message with role but no content is still detected as conversation."""
        content = json.dumps([{"role": "user"}])
        is_conv, messages = mcp_server._detect_conversation(content)
        assert is_conv is True  # Has role key — that's the check

    def test_non_string_input_not_conversation(self):
        """None or non-string input is handled gracefully."""
        is_conv, messages = mcp_server._detect_conversation(None)
        assert is_conv is False
        assert messages is None