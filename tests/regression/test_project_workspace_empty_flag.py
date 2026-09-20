from agent_os.api.routes.agents_v2 import _redact_project


def test_empty_workspace_flagged_true(tmp_path):
    p = {"project_id": "p1", "workspace": str(tmp_path), "api_key": ""}
    assert _redact_project(p)["is_empty_workspace"] is True


def test_nonempty_workspace_flagged_false(tmp_path):
    (tmp_path / "README.md").write_text("hi")
    p = {"project_id": "p1", "workspace": str(tmp_path), "api_key": ""}
    assert _redact_project(p)["is_empty_workspace"] is False


def test_orbital_only_workspace_still_empty(tmp_path):
    # An orbital/ scaffold dir alone does not count as "imported content".
    (tmp_path / "orbital").mkdir()
    p = {"project_id": "p1", "workspace": str(tmp_path), "api_key": ""}
    assert _redact_project(p)["is_empty_workspace"] is True


def _seed(tmp_path, name="Demo"):
    from agent_os.daemon_v2.agent_md_seeder import AGENT_MD_TEMPLATE
    (tmp_path / "AGENTS.md").write_text(
        AGENT_MD_TEMPLATE.format(project_name=name, agent_name=name),
        encoding="utf-8",
    )


def test_seeded_agents_md_does_not_make_a_new_project_nonempty(tmp_path):
    # Project creation seeds AGENTS.md at the workspace ROOT. It is Orbital's
    # own scaffold, exactly like orbital/ — counting it as user content put the
    # "Scan this workspace?" card on every brand-new project and replaced the
    # onboarding flow with the imported-project scan.
    (tmp_path / "orbital").mkdir()
    _seed(tmp_path)
    p = {"project_id": "p1", "workspace": str(tmp_path), "api_key": ""}
    assert _redact_project(p)["is_empty_workspace"] is True


def test_seeded_agents_md_still_recognized_after_project_rename(tmp_path):
    # The seeded file carries the project/agent name it was created with; a
    # later rename must not turn the scaffold into "user content".
    _seed(tmp_path, name="Old name")
    p = {"project_id": "p1", "name": "New name", "workspace": str(tmp_path), "api_key": ""}
    assert _redact_project(p)["is_empty_workspace"] is True


def test_historical_seeded_agents_md_is_scaffold_too(tmp_path):
    from agent_os.daemon_v2.agent_md_seeder import _TEMPLATE_PRE_ASKS
    (tmp_path / "AGENTS.md").write_text(
        _TEMPLATE_PRE_ASKS.format(project_name="x", agent_name="x"), encoding="utf-8",
    )
    p = {"project_id": "p1", "workspace": str(tmp_path), "api_key": ""}
    assert _redact_project(p)["is_empty_workspace"] is True


def test_user_authored_agents_md_counts_as_content(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# My repo\n\nBuild with make.\n", encoding="utf-8")
    p = {"project_id": "p1", "workspace": str(tmp_path), "api_key": ""}
    assert _redact_project(p)["is_empty_workspace"] is False


def test_seeded_agents_md_beside_real_files_is_nonempty(tmp_path):
    _seed(tmp_path)
    (tmp_path / "main.py").write_text("print('hi')")
    p = {"project_id": "p1", "workspace": str(tmp_path), "api_key": ""}
    assert _redact_project(p)["is_empty_workspace"] is False
