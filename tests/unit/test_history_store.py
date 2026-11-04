# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Spec 065: the history store and the tracked-set rules (no watcher)."""
import json
import os

import pytest

from agent_os.history import diffing, rules
from agent_os.history.store import HistoryStore, sha256


# ── rules ─────────────────────────────────────────────────────────────


class TestDenyList:
    @pytest.mark.parametrize("rel", [
        "node_modules/x/y.js", "a/node_modules/z.js", ".git/HEAD", "src/__pycache__/m.pyc",
        ".venv/lib/x.py", "venv/bin/activate",
        "orbital/sessions/abc.jsonl", "orbital/history/index.json", "orbital/backups/x/PROJECT_STATE.md",
        "sub/orbital/PROJECT_STATE.md",  # a nested project's store
        "orbital/ASKS.md", "orbital/DECISIONS_ARCHIVE.md",
        "~$report.docx", ".~lock.sheet.xlsx#", "a/.notes.md.swp", ".#notes.md", "notes.md~",
        "build.tmp", ".orbital-restore-ab12.tmp",
    ])
    def test_denied(self, rel):
        assert rules.is_denied(rel)

    @pytest.mark.parametrize("rel", [
        "notes.md", "src/main.py", "orbital.md", "docs/orbital-notes.md",
        "orbital/PROJECT_STATE.md", "orbital/DECISIONS.md", "orbital/LESSONS.md", "orbital/INDEX.md",
    ])
    def test_tracked(self, rel):
        assert not rules.is_denied(rel)

    def test_nested_registered_workspace_denied(self):
        assert rules.is_denied("inner/notes.md", ("inner",))
        assert rules.is_denied("inner", ("inner",))
        assert not rules.is_denied("inner2/notes.md", ("inner",))


class TestClassify:
    def test_text_binary_size_office(self, tmp_path):
        (tmp_path / "a.md").write_text("hi\n")
        (tmp_path / "b.png").write_bytes(b"\x89PNG\x00\x00data")
        (tmp_path / "big.txt").write_bytes(b"x" * (rules.TEXT_MAX_BYTES + 1))
        (tmp_path / "deck.pptx").write_bytes(b"PK\x03\x04\x00\x00" + b"y" * (2 * 1024 * 1024))
        assert rules.classify(str(tmp_path / "a.md"), "a.md") is None
        assert rules.classify(str(tmp_path / "b.png"), "b.png") == "binary"
        assert rules.classify(str(tmp_path / "big.txt"), "big.txt") == "size"
        # Office documents are tracked whole up to 20 MB, NULs and all.
        assert rules.classify(str(tmp_path / "deck.pptx"), "deck.pptx") is None


class TestListing:
    def test_listing_excludes_store_deps_nested_and_gitignored(self, tmp_path):
        ws = tmp_path
        for rel in ["notes.md", "src/a.py", "node_modules/p/i.js", "orbital/sessions/s.jsonl",
                    "orbital/history/index.json", "orbital/PROJECT_STATE.md", "inner/orbital/x.md",
                    "inner/own.md", "dist/bundle.js", "deep/orbital/y.md"]:
            p = ws / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x\n")
        (ws / ".gitignore").write_text("dist/\n")  # not a git repo: --no-require-git
        listing = set(rules.list_tracked(str(ws), ("inner",)))
        assert listing == {"notes.md", "src/a.py", ".gitignore"}

    def test_no_ripgrep_is_unavailable_not_walk(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rules, "find_ripgrep", lambda: None)
        with pytest.raises(rules.ListingUnavailable):
            rules.list_tracked(str(tmp_path))


# ── store ─────────────────────────────────────────────────────────────


A = {"kind": "outside"}


class TestStore:
    def test_record_dedupes_and_counts_lines(self, tmp_path):
        s = HistoryStore(str(tmp_path))
        s.load()
        v1 = s.record("n.md", b"a\nb\n", attrib=A, kind="baseline")
        assert v1["kind"] == "baseline" and v1["prev_sha"] is None
        assert s.record("n.md", b"a\nb\n", attrib=A) is None  # equal hash → nothing
        v2 = s.record("n.md", b"a\nc\nd\n", attrib=A)
        assert v2["kind"] == "modify" and v2["prev_sha"] == v1["sha"]
        assert (v2["added"], v2["removed"]) == (2, 1)
        assert s.get_blob(v2["sha"]) == b"a\nc\nd\n"
        # restore never writes a new blob for content already stored
        blobs = sum(len(f) for _d, _s, f in os.walk(s.blobs_dir))
        s.record("n.md", b"a\nb\n", attrib=A, kind="restore")
        assert sum(len(f) for _d, _s, f in os.walk(s.blobs_dir)) == blobs

    def test_delete_keeps_content(self, tmp_path):
        s = HistoryStore(str(tmp_path))
        s.load()
        v1 = s.record("n.md", b"keep me\n", attrib=A)
        d = s.record("n.md", None, attrib=A)
        assert d["kind"] == "delete" and d["sha"] is None and d["prev_sha"] == v1["sha"]
        assert s.current("n.md") is None
        assert s.get_blob(v1["sha"]) == b"keep me\n"

    def test_index_rebuilt_and_torn_line_skipped(self, tmp_path):
        s = HistoryStore(str(tmp_path))
        s.load()
        s.record("a.md", b"1\n", attrib=A)
        s.record("a.md", b"2\n", attrib=A)
        os.remove(s.index_path)
        with open(s.versions_path, "ab") as f:
            f.write(b'{"v": 99, "path": "a.md", "sha"')  # crash mid-append
        s2 = HistoryStore(str(tmp_path))
        s2.load()
        assert s2.current("a.md")["sha"] == sha256(b"2\n")
        assert len(s2.versions) == 2
        assert s2.record("a.md", b"3\n", attrib=A)["v"] == 3

    def test_names_latest_wins(self, tmp_path):
        s = HistoryStore(str(tmp_path))
        s.load()
        v = s.record("a.md", b"1\n", attrib=A)
        s.set_name("a.md", v["sha"], "draft")
        s.set_name("a.md", v["sha"], "发布稿")
        s2 = HistoryStore(str(tmp_path))
        s2.load()
        assert s2.names[("a.md", v["sha"])] == "发布稿"

    def test_cap_prunes_oldest_never_current_or_named(self, tmp_path):
        s = HistoryStore(str(tmp_path), cap_bytes=6000)
        s.load()
        named = s.record("keep.md", os.urandom(800).hex().encode(), attrib=A)
        s.set_name("keep.md", named["sha"], "milestone")
        s.record("keep.md", os.urandom(800).hex().encode(), attrib=A)
        olds = [s.record("a.md", os.urandom(800).hex().encode(), attrib=A) for _ in range(8)]
        assert s.total_bytes > 6000
        dropped = s.prune_if_needed()
        assert dropped > 0 and s.total_bytes <= 6000
        shas = {v["sha"] for v in s.versions}
        assert named["sha"] in shas  # named survives
        assert olds[-1]["sha"] in shas  # a.md's current survives
        assert olds[0]["sha"] not in shas  # oldest went first
        assert not s.has_blob(olds[0]["sha"])
        assert s.get_blob(s.current("keep.md")["sha"])  # keep.md current intact
        # the rewritten file still loads
        s2 = HistoryStore(str(tmp_path))
        s2.load()
        assert {v["sha"] for v in s2.versions} == shas


class TestCompose:
    def test_group_view_tags_editor_per_line(self):
        base = b"one\ntwo\nthree\n"
        v1 = b"one\nTWO\nthree\n"            # chain[0] replaces line 2
        v2 = b"one\nTWO\nthree\nfour\n"       # chain[1] appends
        lines = diffing.compose_view(base, [v1, v2])
        ops = [(ln["op"], ln["text"], ln["editor"]) for ln in lines]
        assert ("del", "two", 0) in ops
        assert ("add", "TWO", 0) in ops
        assert ("add", "four", 1) in ops
        assert ("same", "one", None) in ops

    def test_jsonl_is_plain_json(self, tmp_path):
        s = HistoryStore(str(tmp_path))
        s.load()
        s.record("a.md", "中文\n".encode(), attrib=A)
        with open(s.versions_path, encoding="utf-8") as f:
            rec = json.loads(f.readline())
        assert set(rec) >= {"v", "ts", "path", "sha", "prev_sha", "kind", "size", "added", "removed", "attrib"}
