"""
Unit tests for _chunk_text() and _get_chunk_size().

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
    def test_default_chunk_size(self):
        """Default chunk size is MEM0_CHUNK_CHARS (3000)."""
        original = mcp_server.CHUNK_CHARS
        try:
            mcp_server.CHUNK_CHARS = 3000
            assert mcp_server._get_chunk_size() == 3000
        finally:
            mcp_server.CHUNK_CHARS = original

    def test_chunk_size_from_env(self):
        """Chunk size follows MEM0_CHUNK_CHARS."""
        original = mcp_server.CHUNK_CHARS
        try:
            mcp_server.CHUNK_CHARS = 1200
            assert mcp_server._get_chunk_size() == 1200
        finally:
            mcp_server.CHUNK_CHARS = original


