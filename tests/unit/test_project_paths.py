"""
Tests for agent_os.agent.project_paths.ProjectPaths.

Covers every accessor. Uses os.path.normpath for cross-platform path comparison.
"""

import os
import pytest

from agent_os.agent.project_paths import ProjectPaths


WS = "/tmp/test_workspace"
WS_NORM = os.path.normpath(WS)


def p(*parts):
    """Helper: normpath join from WS_NORM."""
    return os.path.normpath(os.path.join(WS_NORM, *parts))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_orbital_dir(self):
        pp = ProjectPaths(WS)
        assert os.path.normpath(pp.orbital_dir) == p("orbital")

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            ProjectPaths("")

    def test_none_raises_value_error(self):
        # None fails the truthiness check and raises ValueError
        with pytest.raises((TypeError, ValueError)):
            ProjectPaths(None)

    def test_windows_style_path(self):
        # Should not crash on Windows-style paths
        pp = ProjectPaths("C:\\workspace")
        expected = os.path.normpath(os.path.join("C:\\workspace", "orbital"))
        assert os.path.normpath(pp.orbital_dir) == expected


# ---------------------------------------------------------------------------
# Flat memory files
# ---------------------------------------------------------------------------

class TestFlatMemoryFiles:
    def setup_method(self):
        self.pp = ProjectPaths(WS)

    def test_project_state(self):
        assert os.path.normpath(self.pp.project_state) == p("orbital", "PROJECT_STATE.md")

    def test_decisions(self):
        assert os.path.normpath(self.pp.decisions) == p("orbital", "DECISIONS.md")

    def test_lessons(self):
        assert os.path.normpath(self.pp.lessons) == p("orbital", "LESSONS.md")

    def test_session_log(self):
        assert os.path.normpath(self.pp.session_log) == p("orbital", "SESSION_LOG.md")

    def test_context(self):
        assert os.path.normpath(self.pp.context) == p("orbital", "CONTEXT.md")


# ---------------------------------------------------------------------------
# Instructions sub-directory
# ---------------------------------------------------------------------------

class TestInstructions:
    def setup_method(self):
        self.pp = ProjectPaths(WS)

    def test_instructions_dir(self):
        assert os.path.normpath(self.pp.instructions_dir) == p("orbital", "instructions")

    def test_project_goals(self):
        assert os.path.normpath(self.pp.project_goals) == p("orbital", "instructions", "project_goals.md")

    def test_user_directives(self):
        assert os.path.normpath(self.pp.user_directives) == p("orbital", "instructions", "user_directives.md")


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class TestSessions:
    def setup_method(self):
        self.pp = ProjectPaths(WS)

    def test_sessions_dir(self):
        assert os.path.normpath(self.pp.sessions_dir) == p("orbital", "sessions")

    def test_session_file(self):
        assert os.path.normpath(self.pp.session_file("abc")) == p("orbital", "sessions", "abc.jsonl")

    def test_session_file_different_id(self):
        assert os.path.normpath(self.pp.session_file("sess-xyz-99")) == p("orbital", "sessions", "sess-xyz-99.jsonl")


# ---------------------------------------------------------------------------
# Sub-agents
# ---------------------------------------------------------------------------

class TestSubAgents:
    def setup_method(self):
        self.pp = ProjectPaths(WS)

    def test_sub_agent_dir(self):
        assert os.path.normpath(self.pp.sub_agent_dir("claude-code")) == p("orbital", "sub_agents", "claude-code")

    def test_sub_agent_dir_other(self):
        assert os.path.normpath(self.pp.sub_agent_dir("gemini")) == p("orbital", "sub_agents", "gemini")


# ---------------------------------------------------------------------------
# Tool results
# ---------------------------------------------------------------------------

class TestToolResults:
    def setup_method(self):
        self.pp = ProjectPaths(WS)

    def test_tool_results_dir(self):
        assert os.path.normpath(self.pp.tool_results_dir("sess_1")) == p("orbital", "tool-results", "sess_1")

    def test_tool_results_dir_other(self):
        assert os.path.normpath(self.pp.tool_results_dir("my-session")) == p("orbital", "tool-results", "my-session")


# ---------------------------------------------------------------------------
# Output sub-directories
# ---------------------------------------------------------------------------

class TestOutput:
    def setup_method(self):
        self.pp = ProjectPaths(WS)

    def test_output_dir(self):
        assert os.path.normpath(self.pp.output_dir) == p("orbital", "output")

    def test_screenshots_dir(self):
        assert os.path.normpath(self.pp.screenshots_dir) == p("orbital", "output", "screenshots")

    def test_pdfs_dir(self):
        assert os.path.normpath(self.pp.pdfs_dir) == p("orbital", "output", "pdfs")

    def test_shell_output_dir(self):
        assert os.path.normpath(self.pp.shell_output_dir) == p("orbital", "output", "shell-output")


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

class TestSkills:
    def setup_method(self):
        self.pp = ProjectPaths(WS)

    def test_skills_dir(self):
        assert os.path.normpath(self.pp.skills_dir) == p("orbital", "skills")

    def test_skill_dir(self):
        assert os.path.normpath(self.pp.skill_dir("my-skill")) == p("orbital", "skills", "my-skill")

    def test_skill_dir_other(self):
        assert os.path.normpath(self.pp.skill_dir("another-skill")) == p("orbital", "skills", "another-skill")


# ---------------------------------------------------------------------------
# Tmp and approval history
# ---------------------------------------------------------------------------

class TestMisc:
    def setup_method(self):
        self.pp = ProjectPaths(WS)

    def test_tmp_dir(self):
        assert os.path.normpath(self.pp.tmp_dir) == p("orbital", ".tmp")

    def test_approval_history(self):
        assert os.path.normpath(self.pp.approval_history) == p("orbital", "approval_history.jsonl")


# ---------------------------------------------------------------------------
# Platform correctness: os.path.join usage (no manual slash concatenation)
# ---------------------------------------------------------------------------

class TestPlatformCorrectness:
    def test_orbital_dir_uses_os_join(self):
        # Verify path is built correctly on both Windows and POSIX
        pp = ProjectPaths(WS)
        # The result must equal what os.path.join would produce
        expected = os.path.normpath(os.path.join(WS, "orbital"))
        assert os.path.normpath(pp.orbital_dir) == expected

    def test_session_file_uses_os_join(self):
        pp = ProjectPaths(WS)
        expected = os.path.normpath(os.path.join(WS, "orbital", "sessions", "abc.jsonl"))
        assert os.path.normpath(pp.session_file("abc")) == expected

    def test_nested_output_uses_os_join(self):
        pp = ProjectPaths(WS)
        expected = os.path.normpath(os.path.join(WS, "orbital", "output", "screenshots"))
        assert os.path.normpath(pp.screenshots_dir) == expected
