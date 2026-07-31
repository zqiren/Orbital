# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""ReadTool pagination + window-relative size cap.

Before this, ReadTool capped every read at a fixed 100_000 chars with no way
to continue — anything past the cap was PERMANENTLY unreadable (1.32% of
production reads hit it). Two changes are under test:

  * the cap is derived from the active model's context window, not a constant
  * ``offset``/``limit`` let the agent page past the cap, and the truncation
    notice names the offset to resume from
"""

import re

import pytest

from agent_os.agent.tools.read import (
    _DEFAULT_MAX_CHARS,
    _MAX_CAP_CHARS,
    _MIN_CAP_CHARS,
    ReadTool,
    _cap_for_window,
)


@pytest.fixture
def workspace(tmp_path):
    return str(tmp_path)


def _write_lines(tmp_path, name, count, text="line"):
    """Write ``count`` numbered lines; returns the full content."""
    content = "".join(f"{text}{i}\n" for i in range(count))
    (tmp_path / name).write_text(content, encoding="utf-8")
    return content


def _next_offset(content):
    """Pull the offset= the truncation notice tells the agent to use."""
    m = re.search(r"offset=(\d+)", content)
    assert m, f"truncation notice names no offset: {content[-300:]!r}"
    return int(m.group(1))


# A read can end with either of two trailing notices, and the distinction
# matters to the agent: hitting the size cap is "you were cut off", honoring
# an explicit `limit` is "here is what you asked for". Tests that only care
# about the body strip whichever one is present.
_NOTICE = re.compile(r"\n\[(?:TRUNCATED|Showing lines)[^\]]*\]\Z")


def _body(content):
    """The file bytes a read returned, minus any trailing notice."""
    return _NOTICE.sub("", content)


def _has_notice(content):
    return bool(_NOTICE.search(content))


# ---------------------------------------------------------------------------
# offset / limit correctness
# ---------------------------------------------------------------------------

class TestOffsetLimit:

    def test_no_offset_or_limit_reads_whole_file(self, tmp_path, workspace):
        content = _write_lines(tmp_path, "f.txt", 10)
        tool = ReadTool(workspace=workspace)
        assert tool.execute(path="f.txt").content == content

    def test_offset_skips_leading_lines(self, tmp_path, workspace):
        _write_lines(tmp_path, "f.txt", 10)
        tool = ReadTool(workspace=workspace)
        result = tool.execute(path="f.txt", offset=3)

        assert result.content.startswith("line3\n")
        assert result.content.endswith("line9\n")
        assert "line2" not in result.content

    def test_limit_caps_line_count(self, tmp_path, workspace):
        _write_lines(tmp_path, "f.txt", 10)
        tool = ReadTool(workspace=workspace)
        result = tool.execute(path="f.txt", limit=4)

        assert _body(result.content) == "line0\nline1\nline2\nline3\n"

    def test_offset_and_limit_together_select_a_window(self, tmp_path, workspace):
        _write_lines(tmp_path, "f.txt", 10)
        tool = ReadTool(workspace=workspace)
        result = tool.execute(path="f.txt", offset=4, limit=3)

        assert _body(result.content) == "line4\nline5\nline6\n"

    def test_offset_is_zero_indexed(self, tmp_path, workspace):
        _write_lines(tmp_path, "f.txt", 5)
        tool = ReadTool(workspace=workspace)
        assert tool.execute(path="f.txt", offset=0, limit=1).content.startswith("line0")

    def test_limit_larger_than_file_returns_whole_file(self, tmp_path, workspace):
        content = _write_lines(tmp_path, "f.txt", 10)
        tool = ReadTool(workspace=workspace)
        result = tool.execute(path="f.txt", limit=9999)

        assert result.content == content
        assert "TRUNCATED" not in result.content

    def test_window_reaching_eof_gets_no_truncation_notice(self, tmp_path, workspace):
        _write_lines(tmp_path, "f.txt", 10)
        tool = ReadTool(workspace=workspace)
        result = tool.execute(path="f.txt", offset=7, limit=3)

        assert result.content == "line7\nline8\nline9\n"
        assert "TRUNCATED" not in result.content

    def test_offset_past_eof_reports_line_count(self, tmp_path, workspace):
        _write_lines(tmp_path, "f.txt", 10)
        tool = ReadTool(workspace=workspace)
        result = tool.execute(path="f.txt", offset=99)

        assert "past end of file" in result.content
        assert "10 lines" in result.content

    def test_offset_exactly_at_eof_is_past_eof(self, tmp_path, workspace):
        _write_lines(tmp_path, "f.txt", 10)
        tool = ReadTool(workspace=workspace)
        assert "past end of file" in tool.execute(path="f.txt", offset=10).content

    def test_empty_file_reads_empty(self, tmp_path, workspace):
        (tmp_path / "empty.txt").write_text("", encoding="utf-8")
        tool = ReadTool(workspace=workspace)
        assert tool.execute(path="empty.txt").content == ""

    def test_bogus_offset_and_limit_are_ignored(self, tmp_path, workspace):
        content = _write_lines(tmp_path, "f.txt", 10)
        tool = ReadTool(workspace=workspace)

        assert tool.execute(path="f.txt", offset=None, limit=None).content == content
        assert tool.execute(path="f.txt", offset="nope").content == content
        assert tool.execute(path="f.txt", offset=-5).content == content
        assert tool.execute(path="f.txt", limit=0).content == content

    def test_string_offset_from_model_is_coerced(self, tmp_path, workspace):
        _write_lines(tmp_path, "f.txt", 10)
        tool = ReadTool(workspace=workspace)
        assert tool.execute(path="f.txt", offset="3").content.startswith("line3\n")

    def test_no_per_line_truncation(self, tmp_path, workspace):
        """A 2_063-char line (LESSONS.md's shape) must come back intact."""
        long_line = "L" * 2_063
        (tmp_path / "mem.md").write_text(long_line + "\nshort\n", encoding="utf-8")
        tool = ReadTool(workspace=workspace)
        assert long_line in tool.execute(path="mem.md").content


# ---------------------------------------------------------------------------
# The truncation notice must be actionable
# ---------------------------------------------------------------------------

class TestTruncationNotice:

    def test_notice_names_lines_shown_and_total(self, tmp_path, workspace):
        _write_lines(tmp_path, "f.txt", 100)
        tool = ReadTool(workspace=workspace)
        result = tool.execute(path="f.txt", limit=10)

        assert "lines 0-10 of 100" in result.content
        assert "offset=10" in result.content

    def test_honoring_an_explicit_limit_is_not_reported_as_truncation(
        self, tmp_path, workspace,
    ):
        """A bounded read got exactly what it asked for — saying TRUNCATED
        there tells the agent something went wrong when nothing did, which is
        the same class of misleading signal this change exists to remove."""
        _write_lines(tmp_path, "f.txt", 100)
        tool = ReadTool(workspace=workspace)

        bounded = tool.execute(path="f.txt", limit=10).content
        assert "TRUNCATED" not in bounded
        assert "as requested" in bounded
        assert _next_offset(bounded) == 10          # still pageable

    def test_hitting_the_size_cap_is_reported_as_truncation(
        self, tmp_path, workspace,
    ):
        """The cap ending a read IS a cut-off, and must say so + name why."""
        (tmp_path / "big.txt").write_text(
            "".join("z" * 199 + "\n" for _ in range(1_000)), encoding="utf-8",
        )
        tool = ReadTool(workspace=workspace, context_window=1_000)  # 20_000 floor

        capped = tool.execute(path="big.txt").content
        assert "TRUNCATED" in capped
        assert "size cap" in capped

    def test_next_offset_actually_continues(self, tmp_path, workspace):
        """The whole point: following the notice must recover the rest."""
        content = _write_lines(tmp_path, "f.txt", 50)
        tool = ReadTool(workspace=workspace)

        seen = ""
        offset = 0
        for _ in range(20):  # bounded so a broken offset can't hang the suite
            result = tool.execute(path="f.txt", offset=offset, limit=7)
            seen += _body(result.content)
            if not _has_notice(result.content):
                break
            offset = _next_offset(result.content)
        assert seen == content

    def test_next_offset_continues_past_the_size_cap(self, tmp_path, workspace):
        """Paging past the CAP (not just past `limit`) — the 1.32% case."""
        # 20_000-char cap (the floor) with 400-char lines → ~49 lines/read.
        content = "".join("c" * 399 + "\n" for _ in range(300))  # 120_000 chars
        (tmp_path / "big.txt").write_text(content, encoding="utf-8")
        tool = ReadTool(workspace=workspace, context_window=1_000)  # → floor

        seen = ""
        offset = 0
        for _ in range(50):
            result = tool.execute(path="big.txt", offset=offset)
            body, sep, _notice = result.content.partition("\n[TRUNCATED")
            seen += body
            if not sep:
                break
            next_offset = _next_offset(result.content)
            assert next_offset > offset, "notice must advance, not loop"
            offset = next_offset
        assert seen == content

    def test_notice_fits_inside_the_cap(self, tmp_path, workspace):
        """Body + notice must stay under the cap.

        Session._cap_tool_result caps again at append time and cuts at the
        LAST NEWLINE within its cap — which would eat the trailing notice and
        strand the agent. Keeping the whole result under our own cap keeps the
        two caps from fighting.
        """
        content = "".join("d" * 99 + "\n" for _ in range(5_000))  # 500_000 chars
        (tmp_path / "big.txt").write_text(content, encoding="utf-8")

        for window in (1_000, 128_000, 1_000_000):
            tool = ReadTool(workspace=workspace, context_window=window)
            result = tool.execute(path="big.txt")
            assert "TRUNCATED" in result.content
            assert len(result.content) <= _cap_for_window(window)

    def test_single_oversized_line_is_cut_but_still_advances(self, tmp_path, workspace):
        """One line bigger than the whole budget: cut it, don't return nothing."""
        (tmp_path / "one.txt").write_text("y" * 60_000 + "\ntail\n", encoding="utf-8")
        tool = ReadTool(workspace=workspace, context_window=1_000)  # 20_000 cap

        result = tool.execute(path="one.txt")
        assert result.content.startswith("y" * 1_000)
        assert "TRUNCATED" in result.content
        assert _next_offset(result.content) == 1

        assert tool.execute(path="one.txt", offset=1).content == "tail\n"


# ---------------------------------------------------------------------------
# Window-relative cap
# ---------------------------------------------------------------------------

class TestWindowRelativeCap:

    def test_none_window_falls_back_to_historical_default(self):
        assert _cap_for_window(None) == _DEFAULT_MAX_CHARS == 100_000
        assert _cap_for_window(0) == _DEFAULT_MAX_CHARS

    def test_none_window_warns(self, caplog):
        with caplog.at_level("WARNING"):
            ReadTool(workspace="/tmp")
        assert any("context_window" in r.getMessage() for r in caplog.records)

    def test_cap_scales_with_the_window(self):
        # 10% of window * 4 chars/token
        assert _cap_for_window(128_000) == 51_200
        assert _cap_for_window(200_000) == 80_000

    def test_cap_is_floored(self):
        assert _cap_for_window(1) == _MIN_CAP_CHARS == 20_000
        assert _cap_for_window(50_000) == _MIN_CAP_CHARS

    def test_cap_is_ceilinged(self):
        assert _cap_for_window(1_047_576) == _MAX_CAP_CHARS == 400_000
        assert _cap_for_window(10_000_000) == _MAX_CAP_CHARS

    def test_read_result_survives_the_session_cap_intact(self, tmp_path, workspace):
        """ReadTool caps at 10% of window; Session._cap_tool_result at 30%.

        A read result must pass through the session cap UNCHANGED — otherwise
        the session's own truncation eats the "use offset=N" hint and the
        agent is stranded at the cap with no way to continue.

        Windows span the real range in config/providers.json (32_768 is the
        smallest configured). Below ~16_667 tokens the 20_000-char FLOOR would
        outgrow the session's 30% cap, but no configured model is that small.
        """
        from agent_os.agent.session import Session

        content = "".join("f" * 99 + "\n" for _ in range(6_000))  # 600_000 chars
        (tmp_path / "big.txt").write_text(content, encoding="utf-8")

        for window in (32_768, 128_000, 200_000, 400_000, 1_047_576):
            tool = ReadTool(workspace=workspace, context_window=window)
            result = tool.execute(path="big.txt").content
            assert "TRUNCATED" in result, window
            assert Session._cap_tool_result(result, window) == result, window

    def test_small_window_truncates_where_large_window_does_not(self, tmp_path, workspace):
        content = "".join("e" * 99 + "\n" for _ in range(600))  # 60_000 chars
        (tmp_path / "f.txt").write_text(content, encoding="utf-8")

        small = ReadTool(workspace=workspace, context_window=128_000)  # 51_200
        assert "TRUNCATED" in small.execute(path="f.txt").content

        large = ReadTool(workspace=workspace, context_window=1_000_000)  # 400_000
        assert large.execute(path="f.txt").content == content


# ---------------------------------------------------------------------------
# Other branches must be untouched
# ---------------------------------------------------------------------------

class TestOtherBranchesUnaffected:

    def test_directory_listing_ignores_offset_and_limit(self, tmp_path, workspace):
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "b.txt").write_text("b", encoding="utf-8")
        tool = ReadTool(workspace=workspace)

        plain = tool.execute(path=".").content
        paged = tool.execute(path=".", offset=1, limit=1).content
        assert plain == paged
        assert "a.txt" in paged and "b.txt" in paged

    def test_image_read_ignores_offset_and_limit(self, tmp_path, workspace):
        import struct
        import zlib

        def _chunk(tag, data):
            return (struct.pack(">I", len(data)) + tag + data
                    + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

        ihdr = struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0)
        raw = b"".join(b"\x00" + b"\xff\x00\x00" * 4 for _ in range(4))
        png = (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
               + _chunk(b"IDAT", zlib.compress(raw)) + _chunk(b"IEND", b""))
        (tmp_path / "img.png").write_bytes(png)

        tool = ReadTool(workspace=workspace)
        plain = tool.execute(path="img.png")
        paged = tool.execute(path="img.png", offset=2, limit=1)

        assert isinstance(paged.content, list)
        assert paged.content == plain.content


# ---------------------------------------------------------------------------
# Tool schema — the model has to know it can page
# ---------------------------------------------------------------------------

class TestSchema:

    def test_offset_and_limit_are_optional_integer_params(self):
        tool = ReadTool(workspace="/tmp")
        props = tool.parameters["properties"]

        assert props["offset"]["type"] == "integer"
        assert props["limit"]["type"] == "integer"
        assert tool.parameters["required"] == ["path"]

    def test_schema_teaches_the_pagination_contract(self):
        tool = ReadTool(workspace="/tmp")
        props = tool.parameters["properties"]

        assert "TRUNCATED" in props["offset"]["description"]
        assert "offset" in props["offset"]["description"]
        assert "0-indexed" in props["offset"]["description"]
        assert "offset" in props["limit"]["description"]
        # Reaches the model: property descriptions ship in the function schema.
        schema = tool.schema()["function"]["parameters"]["properties"]
        assert schema["offset"]["description"] == props["offset"]["description"]

    def test_path_description_is_unchanged(self):
        """Locked byte-exactly by test_scope_aware_tool_descriptions.py."""
        tool = ReadTool(workspace="/tmp")
        assert tool.parameters["properties"]["path"]["description"] == (
            "Path within your workspace. Use a relative path like 'src/main.py' "
            "or 'docs/notes.md'. Do NOT start with '/' and do NOT pass an absolute path."
        )
