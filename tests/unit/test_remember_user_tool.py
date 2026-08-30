# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 073 — RememberAboutUserTool (§4.1/§4.2) and the Layer-1
non-interference invariant (D4): the user memory file is capped at write time
in the tool and never enters the shared budget arithmetic."""

import re
import threading
from datetime import date

from agent_os.agent.tools.remember_user import (
    MAX_CHARS,
    MAX_LINES,
    RememberAboutUserTool,
)


def _tool(tmp_path, project_name="test-project"):
    return RememberAboutUserTool(
        user_memory_path=str(tmp_path / "user_memory.md"),
        project_name=project_name,
    )


def _content(tmp_path) -> str:
    path = tmp_path / "user_memory.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


class TestAppend:
    def test_append_creates_file_with_bullet_line(self, tmp_path):
        result = _tool(tmp_path).execute(fact="Works as a PM at Tencent")
        assert "Noted" in result.content
        content = _content(tmp_path)
        assert content.startswith("- Works as a PM at Tencent ")
        assert content.endswith("-->\n")

    def test_origin_stamp_carries_project_and_date(self, tmp_path):
        _tool(tmp_path, project_name="orbital-marketing").execute(
            fact="Likes mechanical keyboards")
        line = _content(tmp_path).splitlines()[0]
        assert f"<!--from:orbital-marketing {date.today().isoformat()}-->" in line

    def test_second_fact_appends_on_its_own_line(self, tmp_path):
        tool = _tool(tmp_path)
        tool.execute(fact="Fact one")
        tool.execute(fact="Fact two")
        lines = _content(tmp_path).splitlines()
        assert len(lines) == 2
        assert lines[0].startswith("- Fact one ")
        assert lines[1].startswith("- Fact two ")

    def test_whitespace_is_normalized(self, tmp_path):
        _tool(tmp_path).execute(fact="  Works   at\tTencent  ")
        assert _content(tmp_path).startswith("- Works at Tencent ")


class TestRejection:
    def test_empty_fact_rejected(self, tmp_path):
        result = _tool(tmp_path).execute(fact="   ")
        assert "Error" in result.content
        assert _content(tmp_path) == ""

    def test_missing_fact_rejected(self, tmp_path):
        result = _tool(tmp_path).execute()
        assert "Error" in result.content

    def test_multiline_fact_rejected(self, tmp_path):
        result = _tool(tmp_path).execute(fact="Line one\nLine two")
        assert "Error" in result.content
        assert "single line" in result.content
        assert _content(tmp_path) == ""


class TestDedup:
    def test_exact_refile_is_a_noop(self, tmp_path):
        tool = _tool(tmp_path)
        tool.execute(fact="Works at Tencent")
        result = tool.execute(fact="Works at Tencent")
        assert "Already on file" in result.content
        assert len(_content(tmp_path).splitlines()) == 1

    def test_dedup_ignores_the_origin_stamp(self, tmp_path):
        # The same fact re-stated in another project/on another day must not
        # duplicate — the stamp is provenance, not identity.
        (tmp_path / "user_memory.md").write_text(
            "- Works at Tencent <!--from:other-project 2020-01-01-->\n",
            encoding="utf-8")
        result = _tool(tmp_path).execute(fact="Works at Tencent")
        assert "Already on file" in result.content
        assert len(_content(tmp_path).splitlines()) == 1


class TestCaps:
    def test_line_cap_refuses_without_truncating(self, tmp_path):
        existing = "".join(
            f"- Fact {i} <!--from:p 2026-01-01-->\n" for i in range(MAX_LINES))
        (tmp_path / "user_memory.md").write_text(existing, encoding="utf-8")
        result = _tool(tmp_path).execute(fact="One fact too many")
        assert "Error" in result.content
        assert "prune" in result.content
        # Refuse means REFUSE: nothing dropped, nothing appended.
        assert _content(tmp_path) == existing

    def test_char_cap_refuses_without_truncating(self, tmp_path):
        filler = "x" * 560
        existing = "".join(
            f"- {filler}{i} <!--from:p 2026-01-01-->\n" for i in range(10))
        assert len(existing) < MAX_CHARS
        assert len(existing) + 100 > MAX_CHARS - 50  # the append will overflow
        (tmp_path / "user_memory.md").write_text(existing, encoding="utf-8")
        result = _tool(tmp_path).execute(fact="y" * 100)
        assert "Error" in result.content
        assert _content(tmp_path) == existing

    def test_append_allowed_below_both_caps(self, tmp_path):
        existing = "".join(
            f"- Fact {i} <!--from:p 2026-01-01-->\n"
            for i in range(MAX_LINES - 1))
        (tmp_path / "user_memory.md").write_text(existing, encoding="utf-8")
        result = _tool(tmp_path).execute(fact="The last one that fits")
        assert "Noted" in result.content
        assert len(_content(tmp_path).splitlines()) == MAX_LINES


class TestConcurrency:
    def test_bounded_retry_refuses_while_lock_held(self, tmp_path):
        # acquire() is non-blocking; the tool retries 3 times, 100ms apart,
        # then refuses. Hold the lock for the whole window and the call must
        # come back busy with the file untouched — never block forever.
        from agent_os.utils.file_lock import session_lock
        path = tmp_path / "user_memory.md"
        path.write_text("- Existing <!--from:p 2026-01-01-->\n", encoding="utf-8")
        lock = session_lock(str(path))
        lock.acquire()
        try:
            result = _tool(tmp_path).execute(fact="Cannot land")
        finally:
            lock.release()
        assert "busy" in result.content
        assert _content(tmp_path) == "- Existing <!--from:p 2026-01-01-->\n"

    def test_concurrent_appends_do_not_interleave(self, tmp_path):
        # A thundering herd may legitimately exhaust the bounded retry (that
        # refusal is the test above); the invariant HERE is that whatever
        # lands, lands whole — every winner's line intact, every refusal
        # leaving no trace, no spliced bytes.
        facts = [f"Concurrent fact number {i}" for i in range(8)]
        results = {}
        barrier = threading.Barrier(len(facts))

        def worker(fact):
            barrier.wait()
            results[fact] = _tool(tmp_path).execute(fact=fact)

        threads = [threading.Thread(target=worker, args=(f,)) for f in facts]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        succeeded = {f for f, r in results.items() if "Noted" in r.content}
        for fact, result in results.items():
            assert "Noted" in result.content or "busy" in result.content, (
                f"{fact!r}: {result.content}")
        assert succeeded, "at least one append must win the lock"
        lines = _content(tmp_path).splitlines()
        assert sorted(lines) == sorted(
            f"- {f} <!--from:test-project {date.today().isoformat()}-->"
            for f in succeeded)
        # Every line is intact — bullet, fact, stamp — nothing spliced.
        pattern = re.compile(
            r"^- Concurrent fact number \d <!--from:test-project "
            r"\d{4}-\d{2}-\d{2}-->$")
        assert all(pattern.match(line) for line in lines)


class TestLayer1NonInterference:
    """Spec 073 D4: the user memory file must not enter FILE_BUDGETS /
    budgets_for_window / _LAYER1_FILES. Golden-value assertions so ANY change
    to the shared arithmetic fails here."""

    GOLDEN_BUDGETS = {
        "decisions": {"soft": 7000, "hard": 9000},
        "lessons": {"soft": 5000, "hard": 6000},
        "state": {"soft": 1800, "hard": 2000},
        "index": {"soft": 1500, "hard": 2000},
    }

    def test_file_budgets_byte_identical(self):
        from agent_os.agent.memory_entries import FILE_BUDGETS
        assert FILE_BUDGETS == self.GOLDEN_BUDGETS

    def test_budgets_for_window_byte_identical(self):
        from agent_os.agent.memory_entries import budgets_for_window
        # Floors path (no / generous window) — exactly the four keys.
        assert budgets_for_window(None) == self.GOLDEN_BUDGETS
        assert budgets_for_window(1_000_000) == self.GOLDEN_BUDGETS
        # Tiny-window scaling path: total_hard=19000, cap=0.25*40000=10000,
        # scale=10/19 — the exact pre-073 arithmetic over the four keys.
        scaled = budgets_for_window(40_000)
        assert set(scaled) == set(self.GOLDEN_BUDGETS)
        scale = (0.25 * 40_000) / 19_000
        for key, floors in self.GOLDEN_BUDGETS.items():
            assert scaled[key]["hard"] == int(floors["hard"] * scale)
            assert scaled[key]["soft"] == int(floors["soft"] * scale)

    def test_layer1_files_unchanged(self):
        from agent_os.agent.context import _LAYER1_FILES
        assert _LAYER1_FILES == (
            ("state", "PROJECT_STATE.md"),
            ("decisions", "DECISIONS.md"),
            ("lessons", "LESSONS.md"),
            ("index", "INDEX.md"),
        )
