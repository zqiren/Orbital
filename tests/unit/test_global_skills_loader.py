# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Global Orbital skills (<data dir>/skills): the loader, the manager's
prompt listing and the per-project `/` catalog.

Every test uses its own tmp workspace, fake HOME and global folder; the
process-wide configured folder is restored after each test.
"""

from __future__ import annotations

import os

import pytest

from agent_os.agent import skills as skills_mod
from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext
from agent_os.agent.skill_catalog import SkillCatalog
from agent_os.agent.skills import SkillLoader


def _skill(root, dir_name, text):
    d = os.path.join(str(root), dir_name)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "SKILL.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


FM = "---\nname: {name}\ndescription: {desc}\n---\n\nBody.\n"


def _ctx(ws):
    return PromptContext(
        workspace=str(ws), model="m", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=["read", "write", "edit", "shell"],
        os_type="macos", datetime_now="2026-11-17T10:00:00",
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    home = tmp_path / "home"
    gdir = tmp_path / "data" / "skills"
    ws.mkdir()
    home.mkdir()
    # Restored at teardown whatever a test configures.
    monkeypatch.setattr(skills_mod, "_global_skills_dir", None)
    monkeypatch.setenv("ORBITAL_SKILLS_HOME", str(home))
    return ws, home, gdir


# ---- configuration ------------------------------------------------------------


def test_unconfigured_by_default_and_configure_makes_absolute(env, monkeypatch, tmp_path):
    assert skills_mod.global_skills_dir() is None
    monkeypatch.chdir(tmp_path)
    skills_mod.configure_global_skills_dir("rel/skills")
    assert skills_mod.global_skills_dir() == os.path.join(str(tmp_path), "rel", "skills")
    skills_mod.configure_global_skills_dir(None)
    assert skills_mod.global_skills_dir() is None


def test_project_paths_helper_is_data_dir_skills(tmp_path):
    from agent_os.agent.project_paths import global_skills_dir
    assert global_skills_dir(str(tmp_path)) == os.path.join(str(tmp_path), "skills")


# ---- loader -------------------------------------------------------------------


def test_scan_shared_global_lists_frontmatter_and_heading_skills(env):
    ws, _home, gdir = env
    _skill(gdir, "pdf-tidy", FM.format(name="pdf-tidy", desc="Tidy PDFs."))
    _skill(gdir, "old-style", "# Old Style\nHeading form.\n")
    _skill(gdir, ".staging-x", FM.format(name="hidden", desc="Never listed."))
    os.makedirs(gdir / "no-skill-md")
    rows = SkillLoader(str(ws), global_dir=str(gdir)).scan_shared()
    assert [r["dir_name"] for r in rows] == ["old-style", "pdf-tidy"]
    assert rows[1]["name"] == "pdf-tidy"
    assert rows[1]["description"] == "Tidy PDFs."
    assert rows[1]["path"] == os.path.join(str(gdir), "pdf-tidy", "SKILL.md")


def test_scan_stays_project_only(env):
    ws, _home, gdir = env
    _skill(gdir, "g", FM.format(name="g", desc="G."))
    assert SkillLoader(str(ws), global_dir=str(gdir)).scan() == []


def test_scan_shared_global_missing_folder_is_empty(env):
    ws, _home, gdir = env
    assert SkillLoader(str(ws), global_dir=str(gdir)).scan_shared() == []
    assert SkillLoader(str(ws), global_dir=None).scan_shared() == []


def test_global_skill_md_linking_out_is_skipped(env, tmp_path):
    ws, _home, gdir = env
    outside = tmp_path / "secret.md"
    outside.write_text(FM.format(name="leak", desc="L."), encoding="utf-8")
    d = gdir / "leak"
    os.makedirs(d)
    try:
        os.symlink(str(outside), str(d / "SKILL.md"))
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform")
    assert SkillLoader(str(ws), global_dir=str(gdir)).scan_shared() == []


def test_loader_defaults_to_configured_folder(env):
    ws, _home, gdir = env
    _skill(gdir, "g", FM.format(name="g", desc="G."))
    skills_mod.configure_global_skills_dir(str(gdir))
    assert [r["name"] for r in SkillLoader(str(ws)).scan_shared()] == ["g"]


def test_project_skill_of_same_name_shadows_global(env):
    ws, _home, gdir = env
    _skill(ws / "orbital" / "skills", "mine", FM.format(name="Shared", desc="Project one."))
    _skill(gdir, "shared", FM.format(name="shared", desc="Global one."))
    _skill(gdir, "other", FM.format(name="other", desc="O."))
    loader = SkillLoader(str(ws), global_dir=str(gdir))
    visible = loader.scan_shared(shadowed_by=loader.scan())
    assert [r["name"] for r in visible] == ["other"]


# ---- manager prompt -----------------------------------------------------------


_PROJECT_ONLY_SKILLS = (
    "## Skills\n\n"
    "If a skill below matches your task, read its SKILL.md before you start: it holds\n"
    "validated steps and known pitfalls from earlier runs.\n\n"
    "Skills available:\n"
    "- Invoices: File invoices. (at orbital/skills/invoices/SKILL.md)"
)


def test_prompt_without_global_skills_is_byte_identical(env):
    ws, _home, gdir = env
    _skill(ws / "orbital" / "skills", "invoices", FM.format(name="Invoices", desc="File invoices."))
    baseline = PromptBuilder(workspace=str(ws)).build(_ctx(ws))
    assert _PROJECT_ONLY_SKILLS in baseline[1]
    # Configured, folder missing.
    skills_mod.configure_global_skills_dir(str(gdir))
    assert PromptBuilder(workspace=str(ws)).build(_ctx(ws)) == baseline
    # Configured, folder exists but empty.
    os.makedirs(gdir)
    assert PromptBuilder(workspace=str(ws)).build(_ctx(ws)) == baseline


def test_prompt_without_any_skills_keeps_planning_discipline(env):
    ws, _home, gdir = env
    baseline = PromptBuilder(workspace=str(ws)).build(_ctx(ws))
    os.makedirs(gdir)
    skills_mod.configure_global_skills_dir(str(gdir))
    assert PromptBuilder(workspace=str(ws)).build(_ctx(ws)) == baseline
    assert "## Planning Discipline" in baseline[1]


def test_global_skill_changes_only_the_skills_section(env):
    ws, _home, gdir = env
    _skill(ws / "orbital" / "skills", "invoices", FM.format(name="Invoices", desc="File invoices."))
    before = PromptBuilder(workspace=str(ws))
    parts_before = before.semi_stable_parts(_ctx(ws))
    cached_before, _semi, dyn_before = before.build(_ctx(ws))

    gpath = _skill(gdir, "pdf-tidy", FM.format(name="pdf-tidy", desc="Tidy PDFs."))
    skills_mod.configure_global_skills_dir(str(gdir))
    after = PromptBuilder(workspace=str(ws))
    parts_after = after.semi_stable_parts(_ctx(ws))
    cached_after, _semi2, dyn_after = after.build(_ctx(ws))

    assert cached_after == cached_before
    assert dyn_after == dyn_before
    assert [label for label, _ in parts_after] == [label for label, _ in parts_before]
    for (label, text_b), (_l, text_a) in zip(parts_before, parts_after):
        if label != "skills":
            assert text_a == text_b, label
    skills_text = dict(parts_after)["skills"]
    assert skills_text == _PROJECT_ONLY_SKILLS + f"\n- pdf-tidy: Tidy PDFs. (at {gpath})"


def test_global_skill_alone_replaces_planning_discipline_with_listing(env):
    ws, _home, gdir = env
    gpath = _skill(gdir, "pdf-tidy", FM.format(name="pdf-tidy", desc="Tidy PDFs."))
    skills_mod.configure_global_skills_dir(str(gdir))
    text = dict(PromptBuilder(workspace=str(ws)).semi_stable_parts(_ctx(ws)))["skills"]
    assert text.startswith("## Skills\n")
    assert text.endswith(f"Skills available:\n- pdf-tidy: Tidy PDFs. (at {gpath})")


def test_prompt_hides_global_skill_shadowed_by_project(env):
    ws, _home, gdir = env
    _skill(ws / "orbital" / "skills", "invoices", FM.format(name="Invoices", desc="File invoices."))
    _skill(gdir, "invoices", FM.format(name="invoices", desc="Global copy."))
    skills_mod.configure_global_skills_dir(str(gdir))
    text = dict(PromptBuilder(workspace=str(ws)).semi_stable_parts(_ctx(ws)))["skills"]
    assert text == _PROJECT_ONLY_SKILLS


# ---- `/` catalog --------------------------------------------------------------


def test_catalog_lists_global_with_source_label(env):
    ws, home, gdir = env
    gpath = _skill(gdir, "pdf-tidy", FM.format(name="pdf-tidy", desc="Tidy PDFs."))
    entries = SkillCatalog(str(ws), home=str(home), global_dir=str(gdir)).scan()
    assert entries == [{
        "key": "pdf-tidy", "name": "pdf-tidy", "description": "Tidy PDFs.",
        "source": "global", "path": gpath, "dir_name": "pdf-tidy",
    }]


def test_catalog_uses_configured_global_folder_by_default(env):
    ws, home, gdir = env
    _skill(gdir, "g", FM.format(name="g", desc="G."))
    assert SkillCatalog(str(ws), home=str(home)).scan() == []
    skills_mod.configure_global_skills_dir(str(gdir))
    assert [e["source"] for e in SkillCatalog(str(ws), home=str(home)).scan()] == ["global"]


def test_catalog_precedence_project_and_workspace_beat_global_beats_home(env):
    ws, home, gdir = env
    _skill(ws / "orbital" / "skills", "a", FM.format(name="alpha", desc="project"))
    _skill(ws / ".claude" / "skills", "b", FM.format(name="beta", desc="workspace"))
    _skill(gdir, "alpha", FM.format(name="Alpha", desc="global"))
    _skill(gdir, "beta", FM.format(name="beta", desc="global"))
    _skill(gdir, "gamma", FM.format(name="gamma", desc="global"))
    _skill(home / ".claude" / "skills", "gamma", FM.format(name="gamma", desc="home"))
    entries = SkillCatalog(str(ws), home=str(home), global_dir=str(gdir)).scan()
    by_key = {e["key"]: (e["source"], e["description"]) for e in entries}
    assert by_key == {
        "alpha": ("project", "project"),
        "beta": ("workspace", "workspace"),
        "gamma": ("global", "global"),
    }


def test_catalog_resolve_accepts_a_global_path(env):
    ws, home, gdir = env
    gpath = _skill(gdir, "g", FM.format(name="g", desc="G."))
    cat = SkillCatalog(str(ws), home=str(home), global_dir=str(gdir))
    assert cat.resolve(gpath)["source"] == "global"


def _symlink_or_skip(target, link):
    os.makedirs(os.path.dirname(str(link)), exist_ok=True)
    try:
        os.symlink(str(target), str(link))
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform")


# ---- home-folder skills (shared with the sub-agents) ---------------------------


def test_prompt_lists_home_skills_after_global_sorted_with_paths(env):
    ws, home, gdir = env
    _skill(ws / "orbital" / "skills", "invoices", FM.format(name="Invoices", desc="File invoices."))
    gpath = _skill(gdir, "pdf-tidy", FM.format(name="pdf-tidy", desc="Tidy PDFs."))
    zpath = _skill(home / ".codex" / "skills", "zeta", FM.format(name="zeta", desc="Z."))
    apath = _skill(home / ".claude" / "skills", "alpha", FM.format(name="alpha", desc="A."))
    mpath = _skill(home / ".agents" / "skills", "mid", FM.format(name="mid", desc="M."))
    skills_mod.configure_global_skills_dir(str(gdir))
    text = dict(PromptBuilder(workspace=str(ws)).semi_stable_parts(_ctx(ws)))["skills"]
    assert text == _PROJECT_ONLY_SKILLS + "".join([
        f"\n- pdf-tidy: Tidy PDFs. (at {gpath})",
        f"\n- alpha: A. (at {apath})",
        f"\n- mid: M. (at {mpath})",
        f"\n- zeta: Z. (at {zpath})",
    ])


def test_prompt_shadowing_project_then_global_then_home(env):
    ws, home, gdir = env
    _skill(ws / "orbital" / "skills", "a", FM.format(name="one", desc="project"))
    _skill(gdir, "one", FM.format(name="one", desc="global"))
    _skill(gdir, "two", FM.format(name="two", desc="global"))
    _skill(home / ".agents" / "skills", "two", FM.format(name="Two", desc="home"))
    _skill(home / ".agents" / "skills", "three", FM.format(name="three", desc="agents"))
    _skill(home / ".claude" / "skills", "three", FM.format(name="three", desc="claude"))
    skills_mod.configure_global_skills_dir(str(gdir))
    rows = SkillLoader(str(ws)).scan_shared(shadowed_by=SkillLoader(str(ws)).scan())
    assert [(r["name"], r["description"]) for r in rows] == [("two", "global"), ("three", "agents")]


def test_symlinked_home_skill_listed_once(env):
    ws, home, _gdir = env
    real = _skill(home / ".agents" / "skills", "shared", FM.format(name="shared", desc="S."))
    _symlink_or_skip(os.path.dirname(real), home / ".claude" / "skills" / "shared")
    rows = SkillLoader(str(ws)).scan_shared()
    assert [(r["source"], r["path"]) for r in rows] == [("agents", real)]


def test_prompt_skips_home_skill_without_description(env):
    ws, home, _gdir = env
    _skill(home / ".claude" / "skills", "bare", "---\nname: bare\n---\n")
    assert SkillLoader(str(ws)).scan_shared() == []


def test_all_home_skills_off_and_no_global_is_byte_identical(env):
    ws, home, gdir = env
    _skill(ws / "orbital" / "skills", "invoices", FM.format(name="Invoices", desc="File invoices."))
    empty_home = env[1].parent / "empty-home"
    empty_home.mkdir()
    os.environ["ORBITAL_SKILLS_HOME"] = str(empty_home)
    baseline = PromptBuilder(workspace=str(ws)).build(_ctx(ws))
    os.environ["ORBITAL_SKILLS_HOME"] = str(home)
    _skill(home / ".claude" / "skills", "c1", FM.format(name="c1", desc="C."))
    _skill(home / ".codex" / "skills", "x1", FM.format(name="x1", desc="X."))
    skills_mod.configure_global_skills_dir(str(gdir))
    assert PromptBuilder(workspace=str(ws)).build(_ctx(ws)) != baseline
    skills_mod.set_skill_enabled("claude:c1", False)
    skills_mod.set_skill_enabled("codex:x1", False)
    assert PromptBuilder(workspace=str(ws)).build(_ctx(ws)) == baseline


def test_listing_is_identical_across_builds(env):
    ws, home, gdir = env
    for i in (3, 1, 2):
        _skill(home / ".claude" / "skills", f"s{i}", FM.format(name=f"s{i}", desc="D."))
    first = PromptBuilder(workspace=str(ws)).build(_ctx(ws))
    assert PromptBuilder(workspace=str(ws)).build(_ctx(ws)) == first
    assert first[1].index("- s1:") < first[1].index("- s2:") < first[1].index("- s3:")


# ---- switches -----------------------------------------------------------------


def test_switches_default_on_and_persist_sorted(env, tmp_path):
    _ws, _home, gdir = env
    skills_mod.configure_global_skills_dir(str(gdir))
    assert skills_mod.read_disabled_skills() == set()
    skills_mod.set_skill_enabled("codex:b", False)
    skills_mod.set_skill_enabled("claude:a", False)
    path = tmp_path / "data" / "disabled-skills.json"
    import json
    assert json.loads(path.read_text(encoding="utf-8")) == ["claude:a", "codex:b"]
    skills_mod.set_skill_enabled("codex:b", True)
    assert skills_mod.read_disabled_skills() == {"claude:a"}
    assert [n for n in os.listdir(path.parent) if n.endswith(".tmp")] == []


def test_corrupt_or_unconfigured_switch_file_means_all_on(env, tmp_path):
    _ws, _home, gdir = env
    assert skills_mod.read_disabled_skills() == set()
    skills_mod.configure_global_skills_dir(str(gdir))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "disabled-skills.json").write_text("{not json", encoding="utf-8")
    assert skills_mod.read_disabled_skills() == set()


def test_switched_off_skill_leaves_prompt_and_catalog(env):
    ws, home, gdir = env
    _skill(gdir, "g", FM.format(name="g", desc="G."))
    _skill(home / ".claude" / "skills", "c", FM.format(name="c", desc="C."))
    skills_mod.configure_global_skills_dir(str(gdir))
    skills_mod.set_skill_enabled("claude:c", False)
    skills_mod.set_skill_enabled("global:g", False)
    assert SkillLoader(str(ws)).scan_shared() == []
    assert SkillCatalog(str(ws)).scan() == []
    text = dict(PromptBuilder(workspace=str(ws)).semi_stable_parts(_ctx(ws)))["skills"]
    assert text.startswith("## Planning Discipline")


def test_switching_off_one_lets_a_different_same_named_skill_through(env):
    ws, home, gdir = env
    _skill(gdir, "dup", FM.format(name="dup", desc="global"))
    _skill(home / ".claude" / "skills", "dup", FM.format(name="dup", desc="claude"))
    skills_mod.configure_global_skills_dir(str(gdir))
    assert [r["description"] for r in SkillCatalog(str(ws)).scan()] == ["global"]
    skills_mod.set_skill_enabled("global:dup", False)
    assert [r["description"] for r in SkillCatalog(str(ws)).scan()] == ["claude"]


def test_switched_off_folder_is_not_relisted_through_its_symlink(env):
    ws, home, gdir = env
    real = _skill(home / ".agents" / "skills", "s", FM.format(name="s", desc="S."))
    _symlink_or_skip(os.path.dirname(real), home / ".claude" / "skills" / "s")
    skills_mod.configure_global_skills_dir(str(gdir))
    skills_mod.set_skill_enabled("agents:s", False)
    assert SkillCatalog(str(ws)).scan() == []
    assert SkillLoader(str(ws)).scan_shared() == []


# ---- the Settings page groups ---------------------------------------------------


def test_page_groups_order_dedupe_and_hidden_by(env):
    from agent_os.agent.skill_catalog import shared_skill_groups
    _ws, home, gdir = env
    _skill(gdir, "dup", FM.format(name="dup", desc="global"))
    real = _skill(home / ".agents" / "skills", "x", FM.format(name="x", desc="X."))
    _symlink_or_skip(os.path.dirname(real), home / ".claude" / "skills" / "x")
    _skill(home / ".claude" / "skills", "dup", FM.format(name="dup", desc="claude"))
    _skill(home / ".claude" / "skills", ".system", FM.format(name="sys", desc="S."))
    skills_mod.configure_global_skills_dir(str(gdir))
    skills_mod.set_skill_enabled("agents:x", False)
    groups = shared_skill_groups()
    assert [g["source"] for g in groups] == ["global", "agents", "claude", "codex"]
    by = {g["source"]: g for g in groups}
    assert by["codex"]["exists"] is False and by["codex"]["skills"] == []
    assert by["agents"]["skills"] == [{
        "id": "agents:x", "name": "x", "description": "X.", "source": "agents",
        "agent": "shared", "linked_from": ["claude"],
        "path": real, "dir_name": "x", "editable": False, "enabled": False,
        "hidden_by": None,
    }]
    assert [r["agent"] for r in by["claude"]["skills"]] == ["claude-code"]
    assert by["global"]["skills"][0]["agent"] == "orbital"
    assert by["global"]["skills"][0]["linked_from"] == []
    claude = by["claude"]["skills"]
    assert [(r["id"], r["enabled"], r["hidden_by"]) for r in claude] == [
        ("claude:dup", True, "global:dup")]
    assert by["global"]["skills"][0]["editable"] is True
