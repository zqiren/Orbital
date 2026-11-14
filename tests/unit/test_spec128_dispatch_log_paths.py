# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 128: pass the original session-log paths at dispatch, read on demand.

A worker's recap carries conversational text only (tool calls, results and
lifecycle rows are filtered out). The agent taking over after a pin switch
therefore had no way to find the evidence of what ran before. The fix is a
small path block appended to the dispatch message, after the recap and
before the message itself: the prior agents' ORIGINAL native session logs
(Claude Code's ``~/.claude/projects/<slug>/<id>.jsonl``, Codex's rollout
JSONL), the canonical Orbital chat log, and — only when a native log cannot
be resolved — the exact Orbital worker transcript from this chat's dispatch
markers, labelled as a partial capture. Paths only: nothing is read or
injected, no tools are enumerated.
"""

from __future__ import annotations

import json
import os
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.recap import (
    RECAP_CAP_BYTES,
    SOURCE_LOGS_HEADER,
    build_recap,
    format_source_logs,
    source_log_entries,
)

SID = "proj_r_1a2b3c4d"
CHAT_LOG = "/ws/orbital/sessions/proj_r_1a2b3c4d.jsonl"


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def _user(content, target=None, kind=None):
    m = {"role": "user", "content": content, "source": "user"}
    if target:
        m["target"] = target
    if kind:
        m["_kind"] = kind
    return m


def _assistant(content):
    return {"role": "assistant", "content": content, "source": "management"}


def _marker(handle, dispatch_id="d0", transcript="/ws/orbital/sub_agents/{h}/t1.jsonl",
            kind=None):
    path = transcript.format(h=handle)
    row = {
        "role": "system", "source": "daemon",
        "content": f'[Sub-agent] Message sent to {handle}: "hi". Transcript: {path}',
        "_meta": {"dispatch_id": dispatch_id, "handle": handle,
                  "transcript_path": path},
    }
    if kind:
        row["_kind"] = kind
    return row


def _completed(handle, reply):
    display = (f"[Sub-agent] {handle} completed. Summary: {reply[:500]}. "
               f"Transcript: /ws/orbital/sub_agents/{handle}/t1.jsonl.")
    return {"role": "system", "source": "daemon", "content": display,
            "_meta": {"event": "sub_agent_terminal", "kind": "completed",
                      "display_content": display, "reply": reply}}


def _record(session_id, **extra):
    return {"session_id": session_id, "model": "m", "last_used_at": "t", **extra}


def _resolver(table):
    """``native_path_for(handle, record)`` backed by a ``session_id -> path`` map."""
    calls = []

    def resolve(handle, record):
        calls.append((handle, record.get("session_id")))
        return table.get(record.get("session_id"))

    resolve.calls = calls
    return resolve


# ---------------------------------------------------------------------------
# A. source_log_entries — which paths, labelled how
# ---------------------------------------------------------------------------


class TestSourceLogEntries:

    def test_a_to_b_hands_over_a_native_log_and_the_chat_log_only(self):
        """claude-code worked first; codex takes over. Codex gets claude's
        ORIGINAL native log and the chat log — no transcript, no bodies."""
        rows = [
            _user("U1", target="claude-code"), _marker("claude-code", "d1"),
            _completed("claude-code", "C1 done"),
            _user("U2", target="codex"),
        ]
        resolve = _resolver({"cc-1": "/Users/x/.claude/projects/-ws/cc-1.jsonl"})
        entries = source_log_entries(
            rows, "codex", fresh=True,
            thread_history={"claude-code": [_record("cc-1")]},
            chat_log_path=CHAT_LOG, native_path_for=resolve)
        assert entries == [
            ("Original agent session log (claude-code)",
             "/Users/x/.claude/projects/-ws/cc-1.jsonl"),
            ("Orbital chat log", CHAT_LOG),
        ]

    def test_unresolvable_native_log_falls_back_to_the_markers_exact_transcript(self):
        rows = [
            _marker("claude-code", "d1", "/ws/orbital/sub_agents/claude-code/old.jsonl"),
            _completed("claude-code", "C1"),
        ]
        entries = source_log_entries(
            rows, "codex", fresh=True,
            thread_history={"claude-code": [_record("cc-gone")]},
            chat_log_path=CHAT_LOG, native_path_for=_resolver({}))
        assert entries[0] == (
            "Orbital worker transcript (claude-code, partial capture)",
            "/ws/orbital/sub_agents/claude-code/old.jsonl")
        # Never the .latest pointer: the marker's exact file is the source.
        assert all(".latest" not in (p or "") for _, p in entries)

    def test_marker_only_handle_without_a_record_uses_its_transcript(self):
        """An ACP/Pi worker never stores a native path; its transcript is the
        only on-disk evidence and is labelled as partial."""
        rows = [_marker("pi", "d1"), _completed("pi", "P1")]
        entries = source_log_entries(
            rows, "codex", fresh=True, thread_history={},
            chat_log_path=CHAT_LOG, native_path_for=_resolver({}))
        assert entries == [
            ("Orbital worker transcript (pi, partial capture)",
             "/ws/orbital/sub_agents/pi/t1.jsonl"),
            ("Orbital chat log", CHAT_LOG),
        ]

    def test_record_with_no_native_log_and_no_marker_is_unavailable_not_fabricated(self):
        entries = source_log_entries(
            [], "codex", fresh=True,
            thread_history={"claude-code": [_record("cc-pruned")]},
            chat_log_path=CHAT_LOG, native_path_for=_resolver({}))
        assert entries[0] == ("Original agent session log (claude-code)", None)
        rendered = format_source_logs(entries)
        assert "Original agent session log (claude-code): unavailable" in rendered
        assert "cc-pruned" not in rendered

    def test_resumed_worker_does_not_get_its_own_live_thread(self):
        """codex resuming its own thread: its latest record IS that thread.
        An older record (a thread lost to a reset) is still a prior log."""
        resolve = _resolver({"x-old": "/codex/rollout-old.jsonl",
                             "x-live": "/codex/rollout-live.jsonl"})
        # Every completed turn rewrites the live record (seen live: three
        # rows for one claude-code thread) — all of them are the live thread.
        entries = source_log_entries(
            [_marker("codex", "d1"), _completed("codex", "X1")], "codex",
            fresh=False,
            thread_history={"codex": [_record("x-old"), _record("x-live"),
                                      _record("x-live"), _record("x-live")]},
            chat_log_path=CHAT_LOG, native_path_for=resolve)
        assert ("Original agent session log (codex)", "/codex/rollout-old.jsonl") in entries
        assert all(p != "/codex/rollout-live.jsonl" for _, p in entries)
        # Its own current transcript is not offered back to it either.
        assert all("sub_agents/codex" not in (p or "") for _, p in entries)

    def test_fresh_respawn_keeps_every_distinct_old_native_log_of_its_own(self):
        """A reset starts a new thread; the lost threads' logs stay known."""
        resolve = _resolver({"x-1": "/codex/rollout-1.jsonl",
                             "x-2": "/codex/rollout-2.jsonl"})
        entries = source_log_entries(
            [_marker("codex", "d1")], "codex", fresh=True,
            thread_history={"codex": [_record("x-1"), _record("x-2")]},
            chat_log_path=CHAT_LOG, native_path_for=resolve)
        paths = [p for _, p in entries]
        assert paths == ["/codex/rollout-1.jsonl", "/codex/rollout-2.jsonl", CHAT_LOG]

    def test_repeated_records_and_markers_deduplicate_by_exact_path(self):
        rows = [_marker("claude-code", "d1"), _marker("claude-code", "d2"),
                _marker("claude-code", "d3")]
        # background_loss rewrites the same record; three markers, one file.
        history = {"claude-code": [_record("cc-1"), _record("cc-1", background_loss=True)]}
        native = source_log_entries(
            rows, "codex", fresh=True, thread_history=history,
            chat_log_path=CHAT_LOG,
            native_path_for=_resolver({"cc-1": "/c/cc-1.jsonl"}))
        assert [p for _, p in native] == ["/c/cc-1.jsonl", CHAT_LOG]
        fallback = source_log_entries(
            rows, "codex", fresh=True, thread_history={},
            chat_log_path=CHAT_LOG, native_path_for=_resolver({}))
        assert [p for _, p in fallback] == [
            "/ws/orbital/sub_agents/claude-code/t1.jsonl", CHAT_LOG]

    def test_memory_record_and_tidy_turns_add_no_sources(self):
        rows = [_marker("claude-code", "d1", "/ws/orbital/sub_agents/claude-code/mem.jsonl",
                        kind="memory_record"),
                _marker("claude-code", "d2", "/ws/orbital/sub_agents/claude-code/tidy.jsonl",
                        kind="memory_tidy")]
        entries = source_log_entries(
            rows, "codex", fresh=True, thread_history={},
            chat_log_path=CHAT_LOG, native_path_for=_resolver({}))
        assert entries == [("Orbital chat log", CHAT_LOG)]

    def test_unknown_transcript_placeholder_is_not_a_path(self):
        rows = [_marker("claude-code", "d1", "unknown")]
        entries = source_log_entries(
            rows, "codex", fresh=True, thread_history={},
            chat_log_path=CHAT_LOG, native_path_for=_resolver({}))
        assert entries == [("Orbital chat log", CHAT_LOG)]

    def test_no_sources_at_all_means_no_entries(self):
        assert source_log_entries([], "codex", fresh=True, thread_history={},
                                  chat_log_path=None,
                                  native_path_for=_resolver({})) == []

    def test_resolution_is_metadata_only(self):
        """The resolver sees (handle, record) and nothing else: no transcript
        bodies, no message contents are passed to it."""
        resolve = _resolver({"cc-1": "/c/cc-1.jsonl"})
        source_log_entries(
            [_user("secret paste " * 50)], "codex", fresh=True,
            thread_history={"claude-code": [_record("cc-1")]},
            chat_log_path=CHAT_LOG, native_path_for=resolve)
        assert resolve.calls == [("claude-code", "cc-1")]


class TestFormatSourceLogs:

    def test_block_shape_and_json_escaped_paths(self):
        weird = '/Users/x/Desktop/my "quoted"\nproject/log.jsonl'
        text = format_source_logs([
            ("Original agent session log (codex)", "/codex/rollout-1.jsonl"),
            ("Orbital chat log", weird),
        ])
        assert text.startswith(SOURCE_LOGS_HEADER)
        assert '- Original agent session log (codex): "/codex/rollout-1.jsonl"' in text
        assert f"- Orbital chat log: {json.dumps(weird, ensure_ascii=False)}" in text
        assert "\n\n" == text[-2:]
        # The raw newline inside the filename never breaks the block.
        assert weird not in text

    def test_unicode_path_is_kept_readable_and_whole(self):
        path = "/Users/周/桌面/项目/orbital/sessions/x.jsonl"
        text = format_source_logs([("Orbital chat log", path)])
        assert f'"{path}"' in text

    def test_empty_entries_render_nothing(self):
        assert format_source_logs([]) == ""

    def test_windows_paths_render_with_forward_slashes_not_doubled_backslashes(self):
        # 2026-10-22, Windows: JSON doubled every backslash
        # ("D:\\\\proj\\\\..."), which is not a path an agent can open.
        text = format_source_logs([
            ("Orbital chat log", "D:\\orbital-1022\\orbital\\sessions\\k.jsonl"),
            ("Original agent session log (codex)",
             "C:\\Users\\me\\.codex\\sessions\\2026\\10\\23\\rollout-x.jsonl"),
            ("UNC log", "\\\\server\\share\\log.jsonl"),
        ])
        assert '"D:/orbital-1022/orbital/sessions/k.jsonl"' in text
        assert '"C:/Users/me/.codex/sessions/2026/10/23/rollout-x.jsonl"' in text
        assert '"//server/share/log.jsonl"' in text
        assert "\\\\" not in text

    def test_posix_path_with_a_backslash_in_its_name_is_untouched(self):
        text = format_source_logs([("Orbital chat log", "/tmp/odd\\name/log.jsonl")])
        assert json.dumps("/tmp/odd\\name/log.jsonl") in text


# ---------------------------------------------------------------------------
# B. Dispatch — the block rides every dispatch, after the recap
# ---------------------------------------------------------------------------


class _Session:
    def __init__(self, rows, *, history=None, filepath=CHAT_LOG):
        self.rows = rows
        self.sub_agent_thread_history = history or {}
        self.sub_agent_threads = {h: r[-1] for h, r in (history or {}).items() if r}
        self.filepath = filepath

    def get_messages(self):
        return list(self.rows)


def _manager(session, native=None):
    from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
    observer = MagicMock(on_message_routed=AsyncMock())
    mgr = SubAgentManager(process_manager=MagicMock(), lifecycle_observer=observer)
    mgr._dispatch_async = AsyncMock()
    mgr._session_resolver = lambda pid, sid: session
    if native is not None:
        mgr._native_log_path = native
    return mgr


def _prompt(message, dispatch_id="d9"):
    from agent_os.daemon_v2.sub_agent_manager import _QueuedPrompt
    return _QueuedPrompt(message=message, dispatch_id=dispatch_id,
                         transcript_path="/ws/orbital/sub_agents/codex/t1.jsonl",
                         initiator="user_pinned")


def _adapter(status="fresh"):
    return SimpleNamespace(_resume_status=(status, None))


class TestDispatchCarriesThePaths:

    @pytest.mark.asyncio
    async def test_block_sits_between_recap_and_message(self):
        rows = [_user("U1", target="claude-code"), _marker("claude-code", "d1"),
                _completed("claude-code", "C1 done"), _user("U2", target="codex")]
        session = _Session(rows, history={"claude-code": [_record("cc-1")]})
        mgr = _manager(session, native=lambda pid, h, r, **kw: "/c/cc-1.jsonl")
        await mgr._dispatch_prompt_locked(_adapter(), _prompt("take over"),
                                          "proj", SID, "codex")
        sent = mgr._dispatch_async.await_args.args[3]
        recap = build_recap(rows, "codex", fresh=True)
        assert sent.startswith(recap)
        assert sent.endswith("take over")
        block = sent[len(recap):-len("take over")]
        assert block == format_source_logs([
            ("Original agent session log (claude-code)", "/c/cc-1.jsonl"),
            ("Orbital chat log", CHAT_LOG),
        ])
        # Pointers only: no transcript bodies, no tool enumeration.
        assert "tool" not in block.lower()
        # The timeline marker still previews the user's own words.
        obs = mgr._lifecycle_observer.on_message_routed
        assert obs.await_args.kwargs["message_preview"] == "take over"

    @pytest.mark.asyncio
    async def test_recap_eviction_never_drops_the_block(self):
        rows = [_user(f"message number {i} " + "x" * 400) for i in range(200)]
        rows.append(_marker("claude-code", "d1"))
        session = _Session(rows, history={"claude-code": [_record("cc-1")]})
        mgr = _manager(session, native=lambda pid, h, r, **kw: "/c/cc-1.jsonl")
        await mgr._dispatch_prompt_locked(_adapter(), _prompt("go"),
                                          "proj", SID, "codex")
        sent = mgr._dispatch_async.await_args.args[3]
        assert "message number 1 " not in sent  # recap cap still bites
        assert '"/c/cc-1.jsonl"' in sent
        assert f'"{CHAT_LOG}"' in sent
        assert len(sent.encode("utf-8")) < RECAP_CAP_BYTES + 1000

    @pytest.mark.asyncio
    async def test_fifo_drain_resolves_at_dispatch_time(self):
        """A record that lands while the prompt waits in the FIFO is in the
        block when the prompt actually goes out."""
        session = _Session([_marker("codex", "d1")], history={})
        mgr = _manager(session, native=lambda pid, h, r, **kw: f"/c/{r['session_id']}.jsonl")
        session.sub_agent_thread_history["claude-code"] = [_record("cc-late")]
        await mgr._dispatch_prompt_locked(_adapter("resumed"), _prompt("next"),
                                          "proj", SID, "codex")
        sent = mgr._dispatch_async.await_args.args[3]
        assert '"/c/cc-late.jsonl"' in sent

    @pytest.mark.asyncio
    async def test_native_resolution_failure_never_blocks_the_dispatch(self):
        session = _Session([_user("hi")], history={"claude-code": [_record("cc-1")]})

        def _boom(pid, h, r, **kw):
            raise RuntimeError("store unreadable")

        mgr = _manager(session, native=_boom)
        await mgr._dispatch_prompt_locked(_adapter(), _prompt("still goes"),
                                          "proj", SID, "codex")
        sent = mgr._dispatch_async.await_args.args[3]
        assert sent.endswith("still goes")
        assert "user: hi" in sent  # the recap survives a path failure

    @pytest.mark.asyncio
    async def test_memory_record_turn_gets_no_block(self):
        """Spec 112 A6: the end-of-turn memory reminder is bookkeeping, not
        a handover — it must not invite the worker to go reading logs."""
        reminder = "[Orbital memory] Your last turn is ending. Files changed: a.txt"
        rows = [_marker("claude-code", "d1"), _completed("claude-code", "C1"),
                _user(reminder, target="codex", kind="memory_record")]
        session = _Session(rows, history={"claude-code": [_record("cc-1")]})
        mgr = _manager(session, native=lambda pid, h, r, **kw: "/c/cc-1.jsonl")
        await mgr._dispatch_prompt_locked(_adapter("resumed"), _prompt(reminder),
                                          "proj", SID, "codex")
        sent = mgr._dispatch_async.await_args.args[3]
        assert sent.endswith(reminder)
        assert SOURCE_LOGS_HEADER not in sent and "/c/cc-1.jsonl" not in sent
        # The same text typed by the user (no bookkeeping tag) does get it.
        rows[-1] = _user(reminder, target="codex")
        await mgr._dispatch_prompt_locked(_adapter("resumed"), _prompt(reminder),
                                          "proj", SID, "codex")
        assert '"/c/cc-1.jsonl"' in mgr._dispatch_async.await_args.args[3]

    @pytest.mark.asyncio
    async def test_session_without_path_metadata_dispatches_as_before(self):
        bare = SimpleNamespace(get_messages=lambda: [])
        mgr = _manager(bare)
        await mgr._dispatch_prompt_locked(_adapter("resumed"), _prompt("again"),
                                          "proj", SID, "codex")
        assert mgr._dispatch_async.await_args.args[3] == "again"

    @pytest.mark.asyncio
    async def test_path_resolution_runs_off_the_loop(self):
        session = _Session([], history={"claude-code": [_record("cc-1")]})
        seen = []

        def native(pid, h, r, **kw):
            seen.append(threading.current_thread())
            return "/c/cc-1.jsonl"

        mgr = _manager(session, native=native)
        await mgr._dispatch_prompt_locked(_adapter(), _prompt("go"), "proj", SID, "codex")
        assert seen and all(t is not threading.main_thread() for t in seen)


# ---------------------------------------------------------------------------
# C. SubAgentManager._native_log_path — per-transport resolution
# ---------------------------------------------------------------------------


def _registry(transport):
    runtime = SimpleNamespace(transport=transport)
    return SimpleNamespace(get=lambda handle: SimpleNamespace(runtime=runtime))


class TestNativeLogPath:

    def _mgr(self, transport, workspace="/ws"):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        store = SimpleNamespace(get_project=lambda pid: {"workspace": workspace})
        return SubAgentManager(process_manager=MagicMock(),
                               registry=_registry(transport), project_store=store)

    def test_codex_uses_the_saved_rollout_path(self, tmp_path):
        rollout = tmp_path / "rollout-2026-10-22T10-00-00-T1.jsonl"
        rollout.write_text("{}\n")
        mgr = self._mgr("codex-appserver")
        assert mgr._native_log_path("proj", "codex", _record(
            "T1", rollout_path=str(rollout))) == str(rollout)

    def test_codex_falls_back_to_the_exact_thread_id_lookup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))
        d = tmp_path / "sessions" / "2026" / "10" / "22"
        d.mkdir(parents=True)
        found = d / "rollout-2026-10-22T10-00-00-T1.jsonl"
        found.write_text("{}\n")
        mgr = self._mgr("codex-appserver")
        assert mgr._native_log_path("proj", "codex", _record(
            "T1", rollout_path=str(tmp_path / "gone.jsonl"))) == str(found)
        assert mgr._native_log_path("proj", "codex", _record("T-missing")) is None

    def test_claude_resolves_through_the_configured_root_and_slug(self, tmp_path, monkeypatch):
        from agent_os.agent.transports.sdk_transport import SDKTransport
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
        ws = tmp_path / "ws"
        ws.mkdir()
        slug = SDKTransport.claude_project_slug(str(ws))
        store = tmp_path / "cfg" / "projects" / slug
        store.mkdir(parents=True)
        log = store / "0000-cc.jsonl"
        log.write_text("{}\n")
        mgr = self._mgr("acp-sdk-no", workspace=str(ws))  # unknown hint -> SDK lookup
        mgr._registry = _registry("sdk")
        assert mgr._native_log_path("proj", "claude-code", _record("0000-cc")) == str(log)
        assert mgr._native_log_path("proj", "claude-code", _record("0000-nope")) is None

    def test_provider_owned_ids_are_not_paths(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
        for hint in ("acp-sdk", "pi-rpc"):
            mgr = self._mgr(hint)
            assert mgr._native_log_path("proj", "x", _record("some-id")) is None

    def test_workspace_falls_back_to_the_chat_log_location(self, tmp_path, monkeypatch):
        """No project store (or no workspace on it): the chat log lives at
        ``<workspace>/orbital/sessions/<id>.jsonl``, so the slug still resolves."""
        from agent_os.agent.transports.sdk_transport import SDKTransport
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
        ws = tmp_path / "ws"
        (ws / "orbital" / "sessions").mkdir(parents=True)
        slug = SDKTransport.claude_project_slug(str(ws))
        store = tmp_path / "cfg" / "projects" / slug
        store.mkdir(parents=True)
        (store / "0000-cc.jsonl").write_text("{}\n")
        mgr = SubAgentManager(process_manager=MagicMock())
        chat = str(ws / "orbital" / "sessions" / "s.jsonl")
        assert mgr._native_log_path(
            "proj", "claude-code", _record("0000-cc"), chat_log_path=chat,
        ) == str(store / "0000-cc.jsonl")


# ---------------------------------------------------------------------------
# D. Transports: a path-returning resolver behind the existing boolean checks
# ---------------------------------------------------------------------------


class TestTransportResolvers:

    def test_codex_resume_source_path_and_exists_agree(self, tmp_path, monkeypatch):
        from agent_os.agent.transports.codex_transport import CodexTransport
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))
        rollout = tmp_path / "rollout-2026-06-06T21-19-11-T9.jsonl"
        rollout.write_text("{}\n")
        rec = {"session_id": "T9", "rollout_path": str(rollout)}
        assert CodexTransport.resume_source_path(rec) == str(rollout)
        assert CodexTransport.resume_source_exists(rec) is True
        d = tmp_path / "sessions" / "2026" / "06" / "06"
        d.mkdir(parents=True)
        globbed = d / "rollout-2026-06-06T21-19-11-T8.jsonl"
        globbed.write_text("{}\n")
        assert CodexTransport.resume_source_path({"session_id": "T8"}) == str(globbed)
        assert CodexTransport.resume_source_exists({"session_id": "T8"}) is True
        assert CodexTransport.resume_source_path({"session_id": "T7"}) is None
        assert CodexTransport.resume_source_exists({"session_id": "T7"}) is False
        assert CodexTransport.resume_source_path({}) is None

    def test_claude_resume_source_path_and_exists_agree(self, tmp_path, monkeypatch):
        from agent_os.agent.transports.sdk_transport import SDKTransport
        root = tmp_path / "projects"
        ws = tmp_path / "ws"
        ws.mkdir()
        slug = SDKTransport.claude_project_slug(str(ws))
        (root / slug).mkdir(parents=True)
        primary = root / slug / "aaaa.jsonl"
        primary.write_text("{}\n")
        (root / "-elsewhere").mkdir()
        drifted = root / "-elsewhere" / "bbbb.jsonl"
        drifted.write_text("{}\n")
        assert SDKTransport.resume_source_path(
            str(ws), "aaaa", projects_root=str(root)) == str(primary)
        assert SDKTransport.resume_source_path(
            str(ws), "bbbb", projects_root=str(root)) == str(drifted)
        assert SDKTransport.resume_source_path(
            str(ws), "cccc", projects_root=str(root)) is None
        assert SDKTransport.resume_source_exists(
            str(ws), "aaaa", projects_root=str(root)) is True
        assert SDKTransport.resume_source_exists(
            str(ws), "bbbb", projects_root=str(root)) is True
        assert SDKTransport.resume_source_exists(
            str(ws), "cccc", projects_root=str(root)) is False

    def test_claude_lookup_honours_claude_config_dir(self, tmp_path, monkeypatch):
        from agent_os.agent.transports.sdk_transport import SDKTransport
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
        ws = tmp_path / "ws"
        ws.mkdir()
        slug = SDKTransport.claude_project_slug(str(ws))
        store = tmp_path / "cfg" / "projects" / slug
        store.mkdir(parents=True)
        (store / "dddd.jsonl").write_text("{}\n")
        assert SDKTransport.resume_source_path(str(ws), "dddd") == str(store / "dddd.jsonl")


# ---------------------------------------------------------------------------
# E. Session keeps every thread record, not just the last one per handle
# ---------------------------------------------------------------------------


class TestSessionThreadHistory:

    def test_history_keeps_every_record_and_threads_keeps_the_last(self, tmp_path):
        from agent_os.agent.session import Session
        ws = str(tmp_path)
        s = Session.new("proj_x_1", ws)
        s.append({"role": "user", "content": "hi"})
        s.set_sub_agent_thread("codex", session_id="T1", rollout_path="/r/1.jsonl")
        s.set_sub_agent_thread("codex", session_id="T2", rollout_path="/r/2.jsonl")
        s.set_sub_agent_thread("claude-code", session_id="C1")
        assert s.get_sub_agent_thread("codex")["session_id"] == "T2"
        assert [r["session_id"] for r in s.sub_agent_thread_history["codex"]] == ["T1", "T2"]
        assert [r["session_id"] for r in s.sub_agent_thread_history["claude-code"]] == ["C1"]
        assert s.filepath == os.path.join(ws, "orbital", "sessions", "proj_x_1.jsonl")

        loaded = Session.load(s.filepath)
        assert loaded.get_sub_agent_thread("codex")["session_id"] == "T2"
        assert [r["session_id"] for r in loaded.sub_agent_thread_history["codex"]] == ["T1", "T2"]
        assert loaded.sub_agent_thread_history["codex"][0]["rollout_path"] == "/r/1.jsonl"
        # Meta rows never become conversation.
        assert all(m.get("role") != "meta" for m in loaded.get_messages())
        assert loaded.filepath == s.filepath
