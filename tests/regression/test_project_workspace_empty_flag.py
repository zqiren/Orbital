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
