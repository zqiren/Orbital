# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the read-only onboarding import scanner (backlog #34).

All sources are built as fixture home directories under a tmp_path — a fake
``~/.claude/projects``, a fake ``~/.codex`` with a tiny ``state_5.sqlite`` index
and ``sessions/`` JSONL, and a fake ``obsidian.json``. Assertions cover the
locked hard-rules: candidates found, dead-path candidates dropped, dedup across
sessions/sources, Obsidian ranked below agent projects, and metadata-only
extraction (no transcript message bodies surfaced).
"""

import json
import os
import sqlite3

import pytest

from agent_os.onboarding import scan_importable_projects
from agent_os.onboarding.import_scanner import (
    SOURCE_CLAUDE_CODE,
    SOURCE_CODEX,
    SOURCE_OBSIDIAN,
)

# A sentinel we stuff into transcript message bodies. If it ever appears in any
# candidate field, the scanner read a body it should never have touched.
SECRET_BODY = "SECRET_MESSAGE_BODY_DO_NOT_READ"


# --------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------


def _encode_claude_dir(cwd: str) -> str:
    """Claude Code's lossy dir-name encoding (real path is read from the cwd
    key, so the exact encoding does not matter — only uniqueness does)."""
    return cwd.replace("/", "-")


def _write_claude_project(projects_dir, real_cwd, n_sessions=1, with_body=True):
    """Create a Claude Code project dir with `n_sessions` transcripts whose
    records carry the `cwd` key (and, optionally, a message body sentinel)."""
    d = os.path.join(projects_dir, _encode_claude_dir(real_cwd))
    os.makedirs(d, exist_ok=True)
    for i in range(n_sessions):
        records = [
            {"type": "queue-operation", "operation": "start"},  # no cwd
            {
                "type": "user",
                "cwd": real_cwd,
                "message": {"role": "user", "content": SECRET_BODY} if with_body else None,
            },
        ]
        with open(os.path.join(d, f"session-{i}.jsonl"), "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    return d


def _write_codex_index(codex_dir, rows):
    """Create a minimal ``state_5.sqlite`` with the columns the scanner reads
    plus message-derived columns it must NOT read."""
    os.makedirs(codex_dir, exist_ok=True)
    con = sqlite3.connect(os.path.join(codex_dir, "state_5.sqlite"))
    con.execute(
        "CREATE TABLE threads ("
        "id TEXT, rollout_path TEXT, cwd TEXT, updated_at INTEGER, "
        "first_user_message TEXT, preview TEXT, title TEXT)"
    )
    for i, (cwd, updated_at) in enumerate(rows):
        con.execute(
            "INSERT INTO threads VALUES (?,?,?,?,?,?,?)",
            (f"id-{i}", f"/rollout-{i}.jsonl", cwd, updated_at,
             SECRET_BODY, SECRET_BODY, SECRET_BODY),
        )
    con.commit()
    con.close()


def _write_obsidian(config_path, vaults):
    """vaults: list of (path, ts_ms)."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    data = {
        "vaults": {
            f"vault{i}": {"path": p, "ts": ts, "open": True}
            for i, (p, ts) in enumerate(vaults)
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


@pytest.fixture
def home(tmp_path):
    """A fixture HOME with the three source roots pre-created (empty)."""
    h = tmp_path / "home"
    (h / ".claude" / "projects").mkdir(parents=True)
    (h / ".codex").mkdir(parents=True)
    return h


def _scan(home, tmp_path):
    """Run the scanner against the fixture home's default-derived locations."""
    obsidian_cfg = str(tmp_path / "obsidian" / "obsidian.json")
    return scan_importable_projects(
        claude_projects_dir=str(home / ".claude" / "projects"),
        codex_dir=str(home / ".codex"),
        obsidian_config_path=obsidian_cfg,
    )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_finds_candidates_from_all_three_sources(home, tmp_path):
    live_cc = str(tmp_path / "cc-project")
    os.makedirs(live_cc)
    _write_claude_project(str(home / ".claude" / "projects"), live_cc, n_sessions=2)

    live_codex = str(tmp_path / "codex-project")
    os.makedirs(live_codex)
    _write_codex_index(str(home / ".codex"), [(live_codex, 1_777_000_000)])

    live_vault = str(tmp_path / "my-vault")
    os.makedirs(live_vault)
    _write_obsidian(str(tmp_path / "obsidian" / "obsidian.json"),
                    [(live_vault, 1_777_000_000_000)])

    cands = _scan(home, tmp_path)
    by_source = {c.source: c for c in cands}

    assert set(by_source) == {SOURCE_CLAUDE_CODE, SOURCE_CODEX, SOURCE_OBSIDIAN}
    assert by_source[SOURCE_CLAUDE_CODE].path == live_cc
    assert by_source[SOURCE_CLAUDE_CODE].session_count == 2
    assert by_source[SOURCE_CLAUDE_CODE].name == "cc-project"
    assert by_source[SOURCE_CODEX].path == live_codex
    assert by_source[SOURCE_CODEX].session_count == 1
    assert by_source[SOURCE_OBSIDIAN].path == live_vault
    assert by_source[SOURCE_OBSIDIAN].session_count == 0
    # Every candidate carries a real, populated last_activity.
    assert all(c.last_activity for c in cands)


def test_dead_paths_are_dropped(home, tmp_path):
    """A project/vault whose target folder no longer exists is filtered out."""
    live = str(tmp_path / "live-project")
    os.makedirs(live)
    dead = str(tmp_path / "deleted-project")  # never created on disk

    _write_claude_project(str(home / ".claude" / "projects"), live)
    _write_claude_project(str(home / ".claude" / "projects"), dead)
    _write_codex_index(str(home / ".codex"),
                       [(live, 1_777_000_000), (dead, 1_777_000_000)])
    _write_obsidian(str(tmp_path / "obsidian" / "obsidian.json"),
                    [(str(tmp_path / "dead-vault"), 1_777_000_000_000)])

    cands = _scan(home, tmp_path)
    paths = {c.path for c in cands}
    assert live in paths
    assert dead not in paths
    assert str(tmp_path / "dead-vault") not in paths


def test_dedupe_across_sessions_and_sources(home, tmp_path):
    """Same real folder from two Claude sessions + a Codex thread collapses to
    one candidate; session counts sum and the agent source wins the label."""
    shared = str(tmp_path / "shared-project")
    os.makedirs(shared)

    _write_claude_project(str(home / ".claude" / "projects"), shared, n_sessions=3)
    _write_codex_index(str(home / ".codex"),
                       [(shared, 1_777_000_000), (shared, 1_777_100_000)])

    cands = _scan(home, tmp_path)
    for_shared = [c for c in cands if c.path == shared]
    assert len(for_shared) == 1
    merged = for_shared[0]
    # 3 claude sessions + 2 codex threads.
    assert merged.session_count == 5
    assert merged.source in {SOURCE_CLAUDE_CODE, SOURCE_CODEX}
    assert set(merged.sources) == {SOURCE_CLAUDE_CODE, SOURCE_CODEX}


def test_obsidian_ranked_below_agent_projects(home, tmp_path):
    """A very-recent vault still ranks below any agent project."""
    older_cc = str(tmp_path / "older-cc")
    os.makedirs(older_cc)
    _write_claude_project(str(home / ".claude" / "projects"), older_cc)
    _write_codex_index(str(home / ".codex"), [(older_cc, 1_000_000_000)])

    fresh_vault = str(tmp_path / "fresh-vault")
    os.makedirs(fresh_vault)
    # Vault ts far in the future — recency must not float it above agent projects.
    _write_obsidian(str(tmp_path / "obsidian" / "obsidian.json"),
                    [(fresh_vault, 2_000_000_000_000)])

    cands = _scan(home, tmp_path)
    assert cands[-1].source == SOURCE_OBSIDIAN
    assert cands[0].source in {SOURCE_CLAUDE_CODE, SOURCE_CODEX}


def test_within_tier_sorted_newest_first(home, tmp_path):
    old = str(tmp_path / "old-codex")
    new = str(tmp_path / "new-codex")
    os.makedirs(old)
    os.makedirs(new)
    _write_codex_index(str(home / ".codex"),
                       [(old, 1_600_000_000), (new, 1_800_000_000)])

    cands = _scan(home, tmp_path)
    codex = [c for c in cands if c.source == SOURCE_CODEX]
    assert [c.path for c in codex] == [new, old]


def test_metadata_only_no_message_bodies_surfaced(home, tmp_path):
    """The scanner must never surface transcript message bodies. Every source
    fixture embeds a SECRET_BODY sentinel in message/preview/title fields; it
    must not appear anywhere in the returned candidates."""
    live_cc = str(tmp_path / "cc")
    live_codex = str(tmp_path / "cx")
    os.makedirs(live_cc)
    os.makedirs(live_codex)

    _write_claude_project(str(home / ".claude" / "projects"), live_cc, with_body=True)
    _write_codex_index(str(home / ".codex"), [(live_codex, 1_777_000_000)])

    cands = _scan(home, tmp_path)
    blob = json.dumps([c.to_dict() for c in cands])
    assert SECRET_BODY not in blob


def test_codex_index_columns_are_metadata_only(home, tmp_path, monkeypatch):
    """Guard the Codex reader's SQL: it must select only metadata columns and
    never first_user_message/preview/title."""
    live = str(tmp_path / "cx-cols")
    os.makedirs(live)
    _write_codex_index(str(home / ".codex"), [(live, 1_777_000_000)])

    seen_sql = []
    real_connect = sqlite3.connect

    def spy_connect(*a, **k):
        con = real_connect(*a, **k)
        con.set_trace_callback(seen_sql.append)  # fires for every executed SQL
        return con

    monkeypatch.setattr(sqlite3, "connect", spy_connect)
    _scan(home, tmp_path)

    joined = " ".join(seen_sql).lower()
    assert "threads" in joined
    for forbidden in ("first_user_message", "preview", "title"):
        assert forbidden not in joined


def test_codex_falls_back_to_sessions_without_index(home, tmp_path):
    """No state_5.sqlite → scanner derives candidates from session_meta JSONL."""
    live = str(tmp_path / "cx-fallback")
    os.makedirs(live)
    sessions = home / ".codex" / "sessions" / "2026" / "06" / "06"
    sessions.mkdir(parents=True)
    meta = {
        "type": "session_meta",
        "timestamp": "2026-06-06T13:05:00.161Z",
        "payload": {"cwd": live, "id": "abc"},
    }
    body = {"type": "event_msg", "payload": {"content": SECRET_BODY}}
    with open(sessions / "rollout-x.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        f.write(json.dumps(body) + "\n")

    cands = _scan(home, tmp_path)
    codex = [c for c in cands if c.source == SOURCE_CODEX]
    assert len(codex) == 1
    assert codex[0].path == live
    assert SECRET_BODY not in json.dumps([c.to_dict() for c in cands])


def test_missing_sources_yield_empty_not_error(tmp_path):
    """Pointing at non-existent source locations returns [] rather than raising."""
    cands = scan_importable_projects(
        claude_projects_dir=str(tmp_path / "nope-claude"),
        codex_dir=str(tmp_path / "nope-codex"),
        obsidian_config_path=str(tmp_path / "nope" / "obsidian.json"),
    )
    assert cands == []


def test_symlinked_folder_dedupes_to_real_path(home, tmp_path):
    """A Claude project and a Codex thread naming a symlink vs. its target
    collapse via realpath."""
    real = str(tmp_path / "real-target")
    os.makedirs(real)
    link = str(tmp_path / "link-to-target")
    os.symlink(real, link)

    _write_claude_project(str(home / ".claude" / "projects"), real)
    _write_codex_index(str(home / ".codex"), [(link, 1_777_000_000)])

    cands = _scan(home, tmp_path)
    # Both resolve to the same realpath → one merged candidate.
    agentish = [c for c in cands if c.source in {SOURCE_CLAUDE_CODE, SOURCE_CODEX}]
    assert len(agentish) == 1
    assert agentish[0].session_count == 2
