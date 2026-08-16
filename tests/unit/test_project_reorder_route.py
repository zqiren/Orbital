# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Manual project ordering — POST /projects/reorder + the GET /projects sort.

Spec 056. Mounts the agents_v2 router over a real ``ProjectStore`` in a tmp
dir, so the ``sort_key`` round-trip through ``projects.json`` is exercised for
real rather than mocked.

The scratch-first invariant is the must-keep here: no reorder, however the
client phrases it, may float a project above the pinned Quick Tasks row.
"""

import os
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes.agents_v2 import configure, router
from agent_os.daemon_v2.project_store import ProjectStore


@pytest.fixture
def store(tmp_path):
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    return ProjectStore(data_dir)


@pytest.fixture
def client(store):
    app = FastAPI()
    configure(
        project_store=store,
        agent_manager=MagicMock(),
        ws_manager=MagicMock(),
    )
    app.include_router(router)
    return TestClient(app)


def _new_project(store, tmp_path, name, **extra):
    workspace = str(tmp_path / "ws" / name)
    os.makedirs(workspace, exist_ok=True)
    return store.create_project({"name": name, "workspace": workspace, **extra})


def _names(response) -> list[str]:
    return [p["name"] for p in response.json()]


def _listed_names(client) -> list[str]:
    r = client.get("/api/v2/projects")
    assert r.status_code == 200
    return _names(r)


# ---------------------------------------------------------------------------
# Baseline: nothing changes until someone drags
# ---------------------------------------------------------------------------


class TestDefaultOrder:
    def test_keyless_projects_keep_creation_order(self, client, store, tmp_path):
        for name in ("A", "B", "C"):
            _new_project(store, tmp_path, name)
        # A fresh install has no sort_key anywhere and must look exactly as it
        # did before this feature existed: creation (dict-insertion) order.
        assert _listed_names(client) == ["A", "B", "C"]

    def test_scratch_is_pinned_first_without_any_sort_key(self, client, store, tmp_path):
        _new_project(store, tmp_path, "A")
        _new_project(store, tmp_path, "Quick", is_scratch=True)
        assert _listed_names(client) == ["Quick", "A"]


# ---------------------------------------------------------------------------
# The reorder endpoint
# ---------------------------------------------------------------------------


class TestReorder:
    def test_reorder_changes_the_listed_order(self, client, store, tmp_path):
        ids = {n: _new_project(store, tmp_path, n) for n in ("A", "B", "C")}
        r = client.post(
            "/api/v2/projects/reorder",
            json={"ordered_ids": [ids["C"], ids["A"], ids["B"]]},
        )
        assert r.status_code == 200
        assert _listed_names(client) == ["C", "A", "B"]

    def test_response_is_the_canonical_list_in_the_new_order(self, client, store, tmp_path):
        ids = {n: _new_project(store, tmp_path, n) for n in ("A", "B")}
        r = client.post(
            "/api/v2/projects/reorder", json={"ordered_ids": [ids["B"], ids["A"]]},
        )
        # The caller can adopt the response directly — a racing GET converges.
        assert _names(r) == ["B", "A"]
        assert _names(r) == _listed_names(client)

    def test_sort_key_is_the_position_and_is_persisted(self, client, store, tmp_path):
        ids = {n: _new_project(store, tmp_path, n) for n in ("A", "B", "C")}
        client.post(
            "/api/v2/projects/reorder",
            json={"ordered_ids": [ids["C"], ids["B"], ids["A"]]},
        )
        assert store.get_project(ids["C"])["sort_key"] == 0
        assert store.get_project(ids["B"])["sort_key"] == 1
        assert store.get_project(ids["A"])["sort_key"] == 2

    def test_order_survives_a_store_reload(self, client, store, tmp_path):
        ids = {n: _new_project(store, tmp_path, n) for n in ("A", "B", "C")}
        client.post(
            "/api/v2/projects/reorder",
            json={"ordered_ids": [ids["B"], ids["C"], ids["A"]]},
        )
        # projects.json is the persistence story (survives restart + reinstall).
        reloaded = ProjectStore(store._data_dir)
        by_id = {p["project_id"]: p for p in reloaded.list_projects()}
        assert by_id[ids["B"]]["sort_key"] == 0
        assert by_id[ids["C"]]["sort_key"] == 1
        assert by_id[ids["A"]]["sort_key"] == 2

    def test_reorder_is_idempotent(self, client, store, tmp_path):
        ids = {n: _new_project(store, tmp_path, n) for n in ("A", "B")}
        payload = {"ordered_ids": [ids["B"], ids["A"]]}
        client.post("/api/v2/projects/reorder", json=payload)
        client.post("/api/v2/projects/reorder", json=payload)
        assert _listed_names(client) == ["B", "A"]

    def test_empty_order_is_accepted_and_changes_nothing(self, client, store, tmp_path):
        for name in ("A", "B"):
            _new_project(store, tmp_path, name)
        r = client.post("/api/v2/projects/reorder", json={"ordered_ids": []})
        assert r.status_code == 200
        assert _listed_names(client) == ["A", "B"]

    def test_unknown_id_is_rejected_and_writes_nothing(self, client, store, tmp_path):
        ids = {n: _new_project(store, tmp_path, n) for n in ("A", "B")}
        r = client.post(
            "/api/v2/projects/reorder",
            json={"ordered_ids": [ids["B"], "proj_ghost", ids["A"]]},
        )
        assert r.status_code == 404
        assert "proj_ghost" in r.json()["detail"]
        # A stale id means a stale order — nothing is half-applied.
        assert "sort_key" not in store.get_project(ids["B"])
        assert _listed_names(client) == ["A", "B"]

    def test_deleting_a_project_leaves_the_survivors_ordered(self, client, store, tmp_path):
        ids = {n: _new_project(store, tmp_path, n) for n in ("A", "B", "C")}
        client.post(
            "/api/v2/projects/reorder",
            json={"ordered_ids": [ids["C"], ids["B"], ids["A"]]},
        )
        # No delete-janitoring: the survivors' keys stay valid, merely sparse.
        store.delete_project(ids["B"])
        assert _listed_names(client) == ["C", "A"]


# ---------------------------------------------------------------------------
# ProjectStore.reorder_projects
# ---------------------------------------------------------------------------


class TestStoreReorder:
    def test_returns_the_ids_it_stamped_and_skips_the_rest(self, store, tmp_path):
        a = _new_project(store, tmp_path, "A")
        b = _new_project(store, tmp_path, "B")
        assert store.reorder_projects([b, "proj_ghost", a]) == [b, a]
        # Positions are the ordered_ids indices, gaps included — the sort only
        # cares about relative order.
        assert store.get_project(b)["sort_key"] == 0
        assert store.get_project(a)["sort_key"] == 2

    def test_a_subset_reorder_leaves_unlisted_keys_alone(self, client, store, tmp_path):
        a = _new_project(store, tmp_path, "A")
        b = _new_project(store, tmp_path, "B")
        store.reorder_projects([b, a])  # b=0, a=1
        store.reorder_projects([a])     # a=0; b is unlisted and keeps its 0
        assert store.get_project(a)["sort_key"] == 0
        assert store.get_project(b)["sort_key"] == 0
        # Callers are expected to send the complete list. A subset can collide
        # positions like this; the documented degradation is that the list
        # route's stable sort breaks the tie by creation order, never a crash
        # or a random shuffle.
        assert _listed_names(client) == ["A", "B"]

    def test_whole_order_lands_in_one_save(self, store, tmp_path, monkeypatch):
        ids = [_new_project(store, tmp_path, n) for n in ("A", "B", "C")]
        saves = []
        real_save = store._save
        monkeypatch.setattr(store, "_save", lambda: (saves.append(1), real_save())[1])
        store.reorder_projects(list(reversed(ids)))
        # Atomic-enough: a partial write can never strand two shared keys.
        assert len(saves) == 1


# ---------------------------------------------------------------------------
# Legacy / keyless fallback
# ---------------------------------------------------------------------------


class TestKeylessFallback:
    def test_keyless_projects_settle_below_placed_ones_in_creation_order(
        self, client, store, tmp_path,
    ):
        ids = {n: _new_project(store, tmp_path, n) for n in ("A", "B", "C", "D")}
        # Only C and A have ever been dragged.
        client.post(
            "/api/v2/projects/reorder", json={"ordered_ids": [ids["C"], ids["A"]]},
        )
        # B and D are keyless — bottom, still in creation order (stable sort).
        assert _listed_names(client) == ["C", "A", "B", "D"]

    def test_a_newly_created_project_lands_at_the_bottom(self, client, store, tmp_path):
        ids = {n: _new_project(store, tmp_path, n) for n in ("A", "B")}
        client.post(
            "/api/v2/projects/reorder", json={"ordered_ids": [ids["B"], ids["A"]]},
        )
        _new_project(store, tmp_path, "New")
        # Matches today's append-on-create behaviour in the frontend.
        assert _listed_names(client) == ["B", "A", "New"]

    @pytest.mark.parametrize("junk", [True, False, "2", None, [], {}])
    def test_a_non_numeric_sort_key_is_treated_as_unplaced(
        self, client, store, tmp_path, junk,
    ):
        ids = {n: _new_project(store, tmp_path, n) for n in ("A", "B")}
        client.post("/api/v2/projects/reorder", json={"ordered_ids": [ids["B"]]})
        # A hand-edited / corrupt projects.json must not scramble the list or
        # 500 the route. `True` is the sharp edge: bool is an int subclass.
        store.update_project(ids["A"], {"sort_key": junk})
        assert _listed_names(client) == ["B", "A"]


# ---------------------------------------------------------------------------
# The scratch-first invariant — the must-keep
# ---------------------------------------------------------------------------


class TestScratchInvariant:
    def test_reorder_cannot_move_a_project_above_scratch(self, client, store, tmp_path):
        scratch = _new_project(store, tmp_path, "Quick", is_scratch=True)
        ids = {n: _new_project(store, tmp_path, n) for n in ("A", "B")}
        # A client asking for A and B ahead of the scratch project.
        client.post(
            "/api/v2/projects/reorder",
            json={"ordered_ids": [ids["A"], ids["B"], scratch]},
        )
        assert _listed_names(client) == ["Quick", "A", "B"]

    def test_scratch_stays_first_even_with_the_worst_sort_key(
        self, client, store, tmp_path,
    ):
        scratch = _new_project(store, tmp_path, "Quick", is_scratch=True)
        ids = {n: _new_project(store, tmp_path, n) for n in ("A", "B")}
        # Directly stamp the scratch project with a last-place key, bypassing
        # the endpoint entirely. Pinning is a property of the sort, not of
        # request validation, so it must hold anyway.
        store.update_project(scratch, {"sort_key": 999})
        store.update_project(ids["A"], {"sort_key": 0})
        store.update_project(ids["B"], {"sort_key": 1})
        assert _listed_names(client) == ["Quick", "A", "B"]

    def test_scratch_stays_first_when_it_is_the_only_keyless_project(
        self, client, store, tmp_path,
    ):
        _new_project(store, tmp_path, "Quick", is_scratch=True)
        ids = {n: _new_project(store, tmp_path, n) for n in ("A", "B")}
        # Every regular project placed, scratch keyless (+inf) — it still wins
        # because ``not is_scratch`` is the first sort component.
        client.post(
            "/api/v2/projects/reorder", json={"ordered_ids": [ids["B"], ids["A"]]},
        )
        assert _listed_names(client) == ["Quick", "B", "A"]
