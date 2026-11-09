"""Spec 167: the manager prompt's goals section is the directive or nothing.

The interview (ONBOARDING MODE) and the card-driven cold-start scan are gone;
only the import scan's guess flavour still has a section of its own.
"""
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


def test_the_import_scan_emits_its_stages(tmp_path):
    section = PromptBuilder()._onboarding_or_directive(_ctx(tmp_path, cold_start="guess:en"))
    assert "COLD-START" in section.upper()
    assert "skeleton" in section.lower()
    # Must NOT instruct writing user_directives.md.
    assert "user_directives.md" not in section


def test_no_goals_and_no_scan_is_no_section(tmp_path):
    for cold_start in (False, True):
        assert PromptBuilder()._onboarding_or_directive(_ctx(tmp_path, cold_start=cold_start)) is None
    assert PromptBuilder()._onboarding_or_directive(_ctx(tmp_path, is_scratch=True)) is None


def test_existing_goals_still_directive(tmp_path):
    gp = tmp_path / "orbital" / "instructions"
    os.makedirs(gp, exist_ok=True)
    (gp / "project_goals.md").write_text("Mission: do X")
    section = PromptBuilder()._onboarding_or_directive(_ctx(tmp_path))
    assert "PROJECT DIRECTIVE" in section
