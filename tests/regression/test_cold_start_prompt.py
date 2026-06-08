import os

from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext


def _ctx(tmp_path, **kw):
    base = dict(
        workspace=str(tmp_path), model="m", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=["read", "write"], os_type="macos",
        datetime_now="2026-06-08T00:00:00", project_name="Demo",
        project_instructions="",
    )
    base.update(kw)
    return PromptContext(**base)


def test_cold_start_emits_scan_stages(tmp_path):
    # No project_goals.md present + cold_start=True -> 3-stage scan prompt.
    section = PromptBuilder()._onboarding_or_directive(_ctx(tmp_path, cold_start=True))
    assert "COLD-START" in section.upper()
    assert "skeleton" in section.lower()
    # Stage 3 ownership rules: must NOT instruct writing user_directives.md.
    assert "user_directives.md" not in section


def test_non_cold_start_keeps_reactive_onboarding(tmp_path):
    section = PromptBuilder()._onboarding_or_directive(_ctx(tmp_path, cold_start=False))
    assert "ONBOARDING MODE" in section
    assert "COLD-START" not in section.upper()


def test_existing_goals_still_directive(tmp_path):
    gp = tmp_path / "orbital" / "instructions"
    os.makedirs(gp, exist_ok=True)
    (gp / "project_goals.md").write_text("Mission: do X")
    section = PromptBuilder()._onboarding_or_directive(_ctx(tmp_path, cold_start=True))
    # Goals exist -> directive wins even under cold_start (idempotent re-entry).
    assert "PROJECT DIRECTIVE" in section
