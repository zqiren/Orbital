"""
Owns all Orbital on-disk path construction. Callers must use this class rather
than constructing paths directly. See TASK-01.
"""

import os

# ---------------------------------------------------------------------------
# Directory and file name constants — single place of truth
# ---------------------------------------------------------------------------

_ORBITAL = "orbital"
_INSTRUCTIONS = "instructions"
_SESSIONS = "sessions"
_SUB_AGENTS = "sub_agents"
_TOOL_RESULTS = "tool-results"
_OUTPUT = "output"
_SCREENSHOTS = "screenshots"
_PDFS = "pdfs"
_SHELL_OUTPUT = "shell-output"
_SKILLS = "skills"
_TMP = ".tmp"
_PROJECT_STATE = "PROJECT_STATE.md"
_DECISIONS = "DECISIONS.md"
_LESSONS = "LESSONS.md"
_SESSION_LOG = "SESSION_LOG.md"
_CONTEXT = "CONTEXT.md"
_PROJECT_GOALS = "project_goals.md"
_USER_DIRECTIVES = "user_directives.md"
_APPROVAL_HISTORY = "approval_history.jsonl"


class ProjectPaths:
    """Pure path calculator for all Orbital on-disk locations.

    No I/O is performed; callers are responsible for creating directories.
    """

    def __init__(self, workspace: str) -> None:
        if not workspace:
            raise ValueError("workspace must be a non-empty string")
        self._workspace = workspace

    # ------------------------------------------------------------------
    # Root
    # ------------------------------------------------------------------

    @property
    def orbital_dir(self) -> str:
        return os.path.join(self._workspace, _ORBITAL)

    # ------------------------------------------------------------------
    # Flat memory files
    # ------------------------------------------------------------------

    @property
    def project_state(self) -> str:
        return os.path.join(self.orbital_dir, _PROJECT_STATE)

    @property
    def decisions(self) -> str:
        return os.path.join(self.orbital_dir, _DECISIONS)

    @property
    def lessons(self) -> str:
        return os.path.join(self.orbital_dir, _LESSONS)

    @property
    def session_log(self) -> str:
        return os.path.join(self.orbital_dir, _SESSION_LOG)

    @property
    def context(self) -> str:
        return os.path.join(self.orbital_dir, _CONTEXT)

    # ------------------------------------------------------------------
    # Instructions sub-directory
    # ------------------------------------------------------------------

    @property
    def instructions_dir(self) -> str:
        return os.path.join(self.orbital_dir, _INSTRUCTIONS)

    @property
    def project_goals(self) -> str:
        return os.path.join(self.instructions_dir, _PROJECT_GOALS)

    @property
    def user_directives(self) -> str:
        return os.path.join(self.instructions_dir, _USER_DIRECTIVES)

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    @property
    def sessions_dir(self) -> str:
        return os.path.join(self.orbital_dir, _SESSIONS)

    def session_file(self, session_id: str) -> str:
        return os.path.join(self.sessions_dir, session_id + ".jsonl")

    # ------------------------------------------------------------------
    # Sub-agents
    # ------------------------------------------------------------------

    @property
    def sub_agents_dir(self) -> str:
        """Parent directory containing all sub-agent transcript directories."""
        return os.path.join(self.orbital_dir, _SUB_AGENTS)

    def sub_agent_dir(self, handle: str) -> str:
        return os.path.join(self.orbital_dir, _SUB_AGENTS, handle)

    # ------------------------------------------------------------------
    # Tool results
    # ------------------------------------------------------------------

    def tool_results_dir(self, session_id: str) -> str:
        return os.path.join(self.orbital_dir, _TOOL_RESULTS, session_id)

    # ------------------------------------------------------------------
    # Output sub-directories
    # ------------------------------------------------------------------

    @property
    def output_dir(self) -> str:
        return os.path.join(self.orbital_dir, _OUTPUT)

    @property
    def screenshots_dir(self) -> str:
        return os.path.join(self.output_dir, _SCREENSHOTS)

    @property
    def pdfs_dir(self) -> str:
        return os.path.join(self.output_dir, _PDFS)

    @property
    def shell_output_dir(self) -> str:
        return os.path.join(self.output_dir, _SHELL_OUTPUT)

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    @property
    def skills_dir(self) -> str:
        return os.path.join(self.orbital_dir, _SKILLS)

    def skill_dir(self, name: str) -> str:
        return os.path.join(self.skills_dir, name)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @property
    def tmp_dir(self) -> str:
        return os.path.join(self.orbital_dir, _TMP)

    @property
    def approval_history(self) -> str:
        return os.path.join(self.orbital_dir, _APPROVAL_HISTORY)
