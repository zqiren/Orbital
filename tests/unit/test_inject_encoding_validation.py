"""Tests for InjectRequest encoding validation.

Verifies that the field_validator on InjectRequest.content rejects content
containing U+FFFD replacement characters (encoding corruption) while
accepting valid ASCII and valid Chinese (UTF-8) content.
"""

import pytest
from pydantic import ValidationError

from agent_os.api.routes.agents_v2 import InjectRequest


class TestInjectRequestEncodingValidation:
    """InjectRequest.content must reject U+FFFD replacement characters."""

    def test_ascii_content_passes(self):
        req = InjectRequest(content="Hello, world!")
        assert req.content == "Hello, world!"

    def test_chinese_content_passes(self):
        req = InjectRequest(content="你好世界")
        assert req.content == "你好世界"

    def test_mixed_chinese_ascii_passes(self):
        req = InjectRequest(content="Hello 你好 world 世界")
        assert req.content == "Hello 你好 world 世界"

    def test_replacement_char_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            InjectRequest(content="Hello \ufffd world")
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "invalid UTF-8" in errors[0]["msg"]

    def test_multiple_replacement_chars_rejected(self):
        with pytest.raises(ValidationError):
            InjectRequest(content="\ufffd\ufffd\ufffd")

    def test_replacement_char_mixed_with_chinese_rejected(self):
        with pytest.raises(ValidationError):
            InjectRequest(content="你好\ufffd世界")

    def test_empty_content_passes(self):
        req = InjectRequest(content="")
        assert req.content == ""

    def test_optional_fields_unaffected(self):
        req = InjectRequest(content="valid", target="@sub", nonce="abc123")
        assert req.content == "valid"
        assert req.target == "@sub"
        assert req.nonce == "abc123"
