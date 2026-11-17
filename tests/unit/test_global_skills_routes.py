# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The global skills API (/api/v2/skills) and the project upload it shares
code with. A fake HOME and a tmp data dir per test; nothing reads the real
~/.claude, ~/.codex or ~/.agents."""

from __future__ import annotations

import io
import os
import zipfile

import pytest
from fastapi.testclient import TestClient

from agent_os.agent import skills as skills_mod

FM = "---\nname: {name}\ndescription: {desc}\n---\n\nBody.\n"


def _skill(root, dir_name, text):
    d = os.path.join(str(root), dir_name)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "SKILL.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("USERPROFILE", str(h))
    monkeypatch.setenv("ORBITAL_SKILLS_HOME", str(h))
    return h


@pytest.fixture
def client(tmp_path, home, monkeypatch):
    # Recorded before create_app configures it, so teardown restores it.
    monkeypatch.setattr(skills_mod, "_global_skills_dir", None)
    from agent_os.api.app import create_app
    app = create_app(data_dir=str(tmp_path / "data"))
    return TestClient(app)


@pytest.fixture
def gdir(tmp_path):
    return tmp_path / "data" / "skills"


def _project(client, tmp_path, name):
    ws = str(tmp_path / f"ws_{name}")
    os.makedirs(ws, exist_ok=True)
    resp = client.post("/api/v2/projects", json={
        "name": name, "workspace": ws, "model": "gpt-4", "api_key": "test-key"})
    assert resp.status_code == 201, resp.text
    return resp.json()["project_id"], ws


def _upload(client, filename, data, overwrite=None):
    form = {} if overwrite is None else {"overwrite": overwrite}
    return client.post("/api/v2/skills", files={"file": (filename, data)}, data=form)


# ---- wiring -------------------------------------------------------------------


def test_create_app_configures_the_data_dir_skills_folder(client, gdir):
    assert skills_mod.global_skills_dir() == os.path.abspath(str(gdir))


def test_nothing_is_created_until_the_first_add(client, gdir):
    assert client.get("/api/v2/skills").status_code == 200
    assert not os.path.exists(gdir)


# ---- list ---------------------------------------------------------------------


def test_list_groups_global_and_home_folders(client, home, gdir):
    _skill(gdir, "pdf-tidy", FM.format(name="pdf-tidy", desc="Tidy PDFs."))
    cpath = _skill(home / ".claude" / "skills", "frontend", FM.format(name="frontend", desc="F."))
    body = client.get("/api/v2/skills").json()
    groups = body["groups"]
    assert [g["source"] for g in groups] == ["global", "agents", "claude", "codex"]
    g, a, c, x = groups
    assert g["editable"] is True and g["exists"] is True
    assert g["path"] == os.path.abspath(str(gdir))
    assert g["skills"] == [{
        "id": "global:pdf-tidy", "name": "pdf-tidy", "description": "Tidy PDFs.",
        "source": "global",
        "path": os.path.join(os.path.abspath(str(gdir)), "pdf-tidy", "SKILL.md"),
        "dir_name": "pdf-tidy", "editable": True, "enabled": True, "hidden_by": None,
        "agent": "orbital", "linked_from": [],
    }]
    assert c["editable"] is False and c["exists"] is True
    assert c["path"] == os.path.join(str(home), ".claude", "skills")
    assert c["skills"] == [{
        "id": "claude:frontend", "name": "frontend", "description": "F.", "source": "claude",
        "path": cpath, "dir_name": "frontend", "editable": False, "enabled": True,
        "hidden_by": None, "agent": "claude-code", "linked_from": [],
    }]
    assert x["exists"] is False and x["skills"] == []
    assert a["exists"] is False


# ---- switch -------------------------------------------------------------------


def test_switch_off_hides_from_every_project_and_back_on(client, tmp_path, home):
    p1, _ = _project(client, tmp_path, "one")
    p2, _ = _project(client, tmp_path, "two")
    cpath = _skill(home / ".claude" / "skills", "frontend", FM.format(name="frontend", desc="F."))
    before = os.stat(cpath).st_mtime_ns
    for pid in (p1, p2):
        assert [r["source"] for r in client.get(f"/api/v2/projects/{pid}/skill-catalog").json()] == ["claude"]
    resp = client.put("/api/v2/skills/enabled", json={"id": "claude:frontend", "enabled": False})
    assert resp.status_code == 200
    assert resp.json() == {"id": "claude:frontend", "enabled": False}
    for pid in (p1, p2):
        assert client.get(f"/api/v2/projects/{pid}/skill-catalog").json() == []
    row = client.get("/api/v2/skills").json()["groups"][2]["skills"][0]
    assert row["enabled"] is False
    assert client.put("/api/v2/skills/enabled",
                      json={"id": "claude:frontend", "enabled": True}).status_code == 200
    assert len(client.get(f"/api/v2/projects/{p1}/skill-catalog").json()) == 1
    # The home folder itself is never written.
    assert os.stat(cpath).st_mtime_ns == before
    assert sorted(os.listdir(home / ".claude" / "skills")) == ["frontend"]


def test_switch_errors(client):
    assert client.put("/api/v2/skills/enabled",
                      json={"id": "claude:nope", "enabled": False}).status_code == 404
    assert client.put("/api/v2/skills/enabled",
                      json={"id": "project:x", "enabled": False}).status_code == 400
    assert client.put("/api/v2/skills/enabled",
                      json={"id": "nocolon", "enabled": False}).status_code == 400
    assert client.put("/api/v2/skills/enabled", json={"id": "claude:x"}).status_code == 422


def test_delete_drops_the_switch(client, gdir):
    assert _upload(client, "a.md", b"# Tidy\nT.\n").status_code == 201
    assert client.put("/api/v2/skills/enabled",
                      json={"id": "global:tidy", "enabled": False}).status_code == 200
    assert skills_mod.read_disabled_skills() == {"global:tidy"}
    assert client.delete("/api/v2/skills/tidy").status_code == 200
    assert skills_mod.read_disabled_skills() == set()
    row = _upload(client, "a.md", b"# Tidy\nT.\n").json()
    assert row["enabled"] is True and row["id"] == "global:tidy"


# ---- add ----------------------------------------------------------------------


def test_add_md_heading_form(client, gdir):
    resp = _upload(client, "SKILL.md", b"# PDF Tidy\nTidy scanned PDFs.\n")
    assert resp.status_code == 201, resp.text
    row = resp.json()
    assert row["dir_name"] == "pdf-tidy"
    assert row["name"] == "PDF Tidy"
    assert row["source"] == "global" and row["editable"] is True
    assert row["replaced"] is False
    with open(os.path.join(str(gdir), "pdf-tidy", "SKILL.md"), encoding="utf-8") as f:
        assert f.read() == "# PDF Tidy\nTidy scanned PDFs.\n"


def test_add_md_frontmatter_only(client, gdir):
    resp = _upload(client, "x.md", FM.format(name="invoice-flow", desc="File invoices.").encode())
    assert resp.status_code == 201, resp.text
    assert resp.json()["dir_name"] == "invoice-flow"
    assert resp.json()["description"] == "File invoices."


def test_add_cjk_name_keeps_letters(client, gdir):
    resp = _upload(client, "x.md", FM.format(name="整理 发票", desc="把发票归档。").encode("utf-8"))
    assert resp.status_code == 201, resp.text
    assert resp.json()["dir_name"] == "整理-发票"
    assert os.path.isfile(os.path.join(str(gdir), "整理-发票", "SKILL.md"))


def test_add_rejects_bad_shapes(client):
    assert _upload(client, "x.txt", b"# A\nB\n").status_code == 400
    assert _upload(client, "x.md", b"no heading here\n").status_code == 400
    assert _upload(client, "x.md", b"# !!!\nPunctuation only.\n").status_code == 400
    assert _upload(client, "x.zip", b"not a zip").status_code == 400
    assert _upload(client, "x.zip", _zip({"readme.txt": b"hi"})).status_code == 400


def test_add_zip_slip_is_400_and_writes_nothing(client, gdir, tmp_path):
    data = _zip({"../../evil/SKILL.md": b"# Evil\nE.\n"})
    assert _upload(client, "x.zip", data).status_code == 400
    assert not os.path.exists(tmp_path / "evil")
    assert not os.path.exists(gdir) or os.listdir(gdir) == []


def test_add_conflict_then_overwrite(client, gdir):
    assert _upload(client, "a.md", b"# Tidy\nFirst.\n").status_code == 201
    resp = _upload(client, "a.md", b"# Tidy\nSecond.\n")
    assert resp.status_code == 409
    assert resp.json()["detail"] == "Skill already exists: tidy"
    resp = _upload(client, "a.md", b"# Tidy\nSecond.\n", overwrite="true")
    assert resp.status_code == 201, resp.text
    assert resp.json()["replaced"] is True
    with open(os.path.join(str(gdir), "tidy", "SKILL.md"), encoding="utf-8") as f:
        assert "Second." in f.read()
    # Only the skill folder is there: no staging or trash left behind.
    assert os.listdir(gdir) == ["tidy"]
    assert [n for n in os.listdir(os.path.dirname(str(gdir))) if "skills-" in n] == []


def test_add_zip_folder_replaces_whole_folder_on_overwrite(client, gdir):
    v1 = _zip({"My Tool/SKILL.md": b"# My Tool\nV1.\n", "My Tool/old.py": b"x"})
    resp = _upload(client, "t.zip", v1)
    assert resp.status_code == 201, resp.text
    assert resp.json()["dir_name"] == "my-tool"
    assert os.path.isfile(os.path.join(str(gdir), "my-tool", "old.py"))
    v2 = _zip({"My Tool/SKILL.md": b"# My Tool\nV2.\n", "My Tool/new.py": b"y"})
    assert _upload(client, "t.zip", v2).status_code == 409
    assert _upload(client, "t.zip", v2, overwrite="true").status_code == 201
    assert sorted(os.listdir(os.path.join(str(gdir), "my-tool"))) == ["SKILL.md", "new.py"]


def test_add_zip_folder_is_named_from_the_skill_not_the_zip(client, gdir):
    # macOS `zip` stores names without the UTF-8 flag, so a zip's folder
    # name can arrive garbled; the skill's own name is read as UTF-8.
    data = _zip({"whatever-folder/SKILL.md": "# 发票整理\n把发票按月份归档。\n".encode("utf-8"),
                 "whatever-folder/notes.txt": b"x"})
    resp = _upload(client, "t.zip", data)
    assert resp.status_code == 201, resp.text
    assert resp.json()["dir_name"] == "发票整理"
    assert sorted(os.listdir(os.path.join(str(gdir), "发票整理"))) == ["SKILL.md", "notes.txt"]


def test_add_zip_root_skill_md(client, gdir):
    resp = _upload(client, "t.zip", _zip({"SKILL.md": FM.format(name="rooty", desc="R.").encode()}))
    assert resp.status_code == 201, resp.text
    assert os.listdir(os.path.join(str(gdir), "rooty")) == ["SKILL.md"]


def test_add_size_caps(client, monkeypatch):
    from agent_os.api.routes import skills_global
    monkeypatch.setattr(skills_global, "MAX_UPLOAD_BYTES", 100)
    assert _upload(client, "a.md", b"# A\n" + b"x" * 200).status_code == 413
    monkeypatch.setattr(skills_global, "MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
    monkeypatch.setattr(skills_global, "MAX_UNPACKED_BYTES", 1000)
    big = _zip({"s/SKILL.md": b"# S\nS.\n", "s/blob.bin": b"\0" * 5000})
    assert _upload(client, "a.zip", big).status_code == 413
    monkeypatch.setattr(skills_global, "MAX_UNPACKED_BYTES", 50 * 1024 * 1024)
    monkeypatch.setattr(skills_global, "MAX_FILES", 3)
    many = _zip({"s/SKILL.md": b"# S\nS.\n", **{f"s/{i}.txt": b"x" for i in range(5)}})
    assert _upload(client, "a.zip", many).status_code == 413


# ---- delete -------------------------------------------------------------------


def test_delete_global_skill(client, gdir):
    assert _upload(client, "a.md", b"# Tidy\nT.\n").status_code == 201
    resp = client.delete("/api/v2/skills/tidy")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": "tidy"}
    assert not os.path.exists(os.path.join(str(gdir), "tidy"))
    assert client.delete("/api/v2/skills/tidy").status_code == 404


def test_delete_never_touches_home_folders(client, home):
    cpath = _skill(home / ".claude" / "skills", "frontend", FM.format(name="frontend", desc="F."))
    assert client.delete("/api/v2/skills/frontend").status_code == 404
    assert os.path.isfile(cpath)


def test_delete_rejects_escaping_names(client, gdir, tmp_path):
    os.makedirs(gdir)
    victim = tmp_path / "data" / "victim"
    os.makedirs(victim)
    for name in ("..", "%2E%2E", ".", "..%2Fvictim"):
        # "." collapses to the collection URL (405); the rest are refused.
        assert client.delete(f"/api/v2/skills/{name}").status_code in (400, 404, 405), name
    assert os.path.isdir(victim)
    assert os.path.isdir(gdir)


def test_delete_symlinked_skill_removes_link_not_target(client, gdir, tmp_path):
    target = tmp_path / "elsewhere" / "linked"
    _skill(target.parent, "linked", FM.format(name="linked", desc="L."))
    os.makedirs(gdir)
    try:
        os.symlink(str(target), os.path.join(str(gdir), "linked"))
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform")
    assert client.delete("/api/v2/skills/linked").status_code == 200
    assert not os.path.lexists(os.path.join(str(gdir), "linked"))
    assert os.path.isfile(target / "SKILL.md")


# ---- every project sees it ----------------------------------------------------


def test_global_skill_reaches_every_projects_catalog_and_project_shadows(client, tmp_path):
    p1, ws1 = _project(client, tmp_path, "one")
    p2, ws2 = _project(client, tmp_path, "two")
    assert _upload(client, "a.md", FM.format(name="pdf-tidy", desc="Global.").encode()).status_code == 201
    for pid in (p1, p2):
        rows = client.get(f"/api/v2/projects/{pid}/skill-catalog").json()
        assert [(r["name"], r["source"]) for r in rows] == [("pdf-tidy", "global")]
    # The project's own skill list stays project-only.
    assert client.get(f"/api/v2/projects/{p1}/skills").json() == []
    _skill(os.path.join(ws2, "orbital", "skills"), "pdf-tidy",
           FM.format(name="pdf-tidy", desc="Project two's own."))
    rows = client.get(f"/api/v2/projects/{p2}/skill-catalog").json()
    assert [(r["source"], r["description"]) for r in rows] == [("project", "Project two's own.")]
    rows = client.get(f"/api/v2/projects/{p1}/skill-catalog").json()
    assert [r["source"] for r in rows] == ["global"]


# ---- copy a project skill to global -------------------------------------------


def test_copy_project_skill_to_global(client, tmp_path, gdir):
    pid, ws = _project(client, tmp_path, "src")
    src = os.path.join(ws, "orbital", "skills", "Invoice Flow")
    _skill(os.path.dirname(src), "Invoice Flow", FM.format(name="Invoice Filing", desc="File them."))
    with open(os.path.join(src, "helper.py"), "w", encoding="utf-8") as f:
        f.write("print(1)\n")
    body = {"project_id": pid, "dir_name": "Invoice Flow"}
    resp = client.post("/api/v2/skills/from-project", json=body)
    assert resp.status_code == 201, resp.text
    # Named from the skill, as every global install is.
    assert resp.json()["dir_name"] == "invoice-filing"
    assert sorted(os.listdir(os.path.join(str(gdir), "invoice-filing"))) == ["SKILL.md", "helper.py"]
    assert os.path.isdir(src)  # the project copy stays
    assert client.post("/api/v2/skills/from-project", json=body).status_code == 409
    resp = client.post("/api/v2/skills/from-project", json={**body, "overwrite": True})
    assert resp.status_code == 201 and resp.json()["replaced"] is True
    assert client.post("/api/v2/skills/from-project",
                       json={"project_id": pid, "dir_name": "nope"}).status_code == 404
    assert client.post("/api/v2/skills/from-project",
                       json={"project_id": "missing", "dir_name": "x"}).status_code == 404
    assert client.post("/api/v2/skills/from-project",
                       json={"project_id": pid, "dir_name": ".."}).status_code in (400, 404)


# ---- the project upload is unchanged by the shared helper ----------------------


def test_project_upload_md_unchanged(client, tmp_path):
    pid, ws = _project(client, tmp_path, "p")
    resp = client.post(f"/api/v2/projects/{pid}/skills",
                       files={"file": ("x.md", b"# My Skill\nDoes it.\n")})
    assert resp.status_code == 201, resp.text
    path = os.path.join(ws, "orbital", "skills", "my-skill", "SKILL.md")
    assert resp.json() == {"name": "My Skill", "description": "Does it.",
                           "path": path, "dir_name": "my-skill"}
    again = client.post(f"/api/v2/projects/{pid}/skills",
                        files={"file": ("x.md", b"# My Skill\nDoes it.\n")})
    assert again.status_code == 409
    assert again.json()["detail"] == "Skill already exists: my-skill"
    bad = client.post(f"/api/v2/projects/{pid}/skills", files={"file": ("x.txt", b"x")})
    assert bad.status_code == 400 and bad.json()["detail"] == "File must be .zip or .md"


def test_project_upload_zip_unchanged(client, tmp_path):
    pid, ws = _project(client, tmp_path, "z")
    data = _zip({"My Tool/SKILL.md": b"# My Tool\nTool.\n", "My Tool/a.py": b"x"})
    resp = client.post(f"/api/v2/projects/{pid}/skills", files={"file": ("t.zip", data)})
    assert resp.status_code == 201, resp.text
    # The project route keeps the zip's folder name as-is.
    assert resp.json()["dir_name"] == "My Tool"
    assert sorted(os.listdir(os.path.join(ws, "orbital", "skills", "My Tool"))) == ["SKILL.md", "a.py"]
    slip = client.post(f"/api/v2/projects/{pid}/skills",
                       files={"file": ("t.zip", _zip({"../x/SKILL.md": b"# X\nX.\n"}))})
    assert slip.status_code == 400
    assert slip.json()["detail"] == "Zip contains path traversal entry"
    none = client.post(f"/api/v2/projects/{pid}/skills",
                       files={"file": ("t.zip", _zip({"a.txt": b"x"}))})
    assert none.status_code == 400 and none.json()["detail"] == "No SKILL.md found in zip"
