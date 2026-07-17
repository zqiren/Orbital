# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Codex live model-list fetcher (TASK-live-model-config).

The sub-agent settings dropdown must offer the ChatGPT account's ACTUAL
model list instead of free text — a free-text override (`gpt-5.6`, valid in
Codex desktop but not through the CLI's ChatGPT-account gate) 400s on every
dispatch. The fetcher speaks the same version-pinned app-server JSON-RPC the
transport uses: initialize → initialized → model/list, parsing
``result.data[].id``.

Layering mirrors the failure modes: a pure protocol layer unit-tested over
in-memory streams (cross-platform), a spawn layer that never raises (any
failure → None so the settings page degrades to free text), and a TTL cache
so the settings page doesn't spawn codex on every load.
"""

import asyncio
import json
import os
import stat
import sys

import pytest

from agent_os.agent.transports import codex_models


# ---------------------------------------------------------------------------
# Protocol layer — in-memory streams
# ---------------------------------------------------------------------------

class FakeWriter:
    """Duck-typed StreamWriter capturing written lines."""

    def __init__(self):
        self.lines: list[dict] = []

    def write(self, data: bytes) -> None:
        for raw in data.decode("utf-8").splitlines():
            if raw.strip():
                self.lines.append(json.loads(raw))

    async def drain(self) -> None:
        return None


def _feed(reader: asyncio.StreamReader, *objs: dict) -> None:
    for obj in objs:
        reader.feed_data((json.dumps(obj) + "\n").encode("utf-8"))


INIT_RESULT = {"jsonrpc": "2.0", "id": 1,
               "result": {"userAgent": "codex/0.125.0"}}
MODEL_LIST_RESULT = {
    "jsonrpc": "2.0", "id": 2,
    "result": {"data": [{"id": "gpt-5.5"}, {"id": "gpt-5.4-mini"},
                        {"id": "gpt-5.3-codex"}],
               "nextCursor": None},
}


class TestProtocol:

    @pytest.mark.asyncio
    async def test_returns_model_ids_and_speaks_pinned_handshake(self):
        reader = asyncio.StreamReader()
        _feed(reader, INIT_RESULT, MODEL_LIST_RESULT)
        reader.feed_eof()
        writer = FakeWriter()

        ids = await codex_models.read_model_ids(reader, writer, timeout=5.0)

        assert ids == ["gpt-5.5", "gpt-5.4-mini", "gpt-5.3-codex"]
        methods = [m.get("method") for m in writer.lines]
        assert methods == ["initialize", "initialized", "model/list"]
        # Same includeHidden=False the transport's startup resolution uses.
        assert writer.lines[2]["params"] == {"includeHidden": False}

    @pytest.mark.asyncio
    async def test_skips_notifications_and_garbage_lines(self):
        reader = asyncio.StreamReader()
        reader.feed_data(b"not json at all\n")
        _feed(reader, {"jsonrpc": "2.0", "method": "thread/started",
                       "params": {}})
        _feed(reader, INIT_RESULT)
        _feed(reader, {"jsonrpc": "2.0", "method": "thread/tokenUsage/updated",
                       "params": {}})
        _feed(reader, MODEL_LIST_RESULT)
        reader.feed_eof()

        ids = await codex_models.read_model_ids(reader, FakeWriter(),
                                                timeout=5.0)
        assert ids == ["gpt-5.5", "gpt-5.4-mini", "gpt-5.3-codex"]

    @pytest.mark.asyncio
    async def test_raises_on_eof_before_response(self):
        reader = asyncio.StreamReader()
        reader.feed_eof()
        with pytest.raises(Exception):
            await codex_models.read_model_ids(reader, FakeWriter(),
                                              timeout=5.0)

    @pytest.mark.asyncio
    async def test_raises_on_malformed_result(self):
        """A response with no data list must raise (spawn layer maps any
        raise to None) rather than silently returning []."""
        reader = asyncio.StreamReader()
        _feed(reader, INIT_RESULT,
              {"jsonrpc": "2.0", "id": 2, "result": {"unexpected": True}})
        reader.feed_eof()
        with pytest.raises(Exception):
            await codex_models.read_model_ids(reader, FakeWriter(),
                                              timeout=5.0)


# ---------------------------------------------------------------------------
# Spawn layer — fake codex binary (POSIX; Windows CI covers the protocol
# layer above, the spawn seam is a thin asyncio.create_subprocess_exec)
# ---------------------------------------------------------------------------

FAKE_CODEX = """#!/usr/bin/env python3
import json, sys
for line in sys.stdin:
    try:
        msg = json.loads(line)
    except Exception:
        continue
    rpc_id = msg.get("id")
    method = msg.get("method")
    if method == "initialize":
        print(json.dumps({"jsonrpc": "2.0", "id": rpc_id,
                          "result": {"userAgent": "codex/0.125.0"}}), flush=True)
    elif method == "model/list":
        print(json.dumps({"jsonrpc": "2.0", "id": rpc_id, "result": {
            "data": [{"id": "gpt-5.5"}, {"id": "gpt-5.4-mini"}],
            "nextCursor": None}}), flush=True)
"""


def _write_fake_codex(tmp_path) -> str:
    path = tmp_path / "fake-codex"
    path.write_text(FAKE_CODEX)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return str(path)


@pytest.mark.skipif(sys.platform == "win32",
                    reason="shebang fake binary; protocol layer covers win32")
class TestFetch:

    @pytest.mark.asyncio
    async def test_fetch_returns_ids_from_fake_binary(self, tmp_path):
        binary = _write_fake_codex(tmp_path)
        ids = await codex_models.fetch_codex_models(binary, timeout=10.0)
        assert ids == ["gpt-5.5", "gpt-5.4-mini"]

    @pytest.mark.asyncio
    async def test_fetch_returns_none_when_binary_missing(self, tmp_path):
        ids = await codex_models.fetch_codex_models(
            str(tmp_path / "no-such-binary"), timeout=5.0)
        assert ids is None

    @pytest.mark.asyncio
    async def test_fetch_returns_none_on_garbage_binary(self, tmp_path):
        path = tmp_path / "garbage-codex"
        path.write_text("#!/bin/sh\necho not-json\nexit 0\n")
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
        ids = await codex_models.fetch_codex_models(str(path), timeout=5.0)
        assert ids is None


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------

class TestCache:

    @pytest.fixture(autouse=True)
    def _clean_cache(self):
        codex_models.clear_codex_models_cache()
        yield
        codex_models.clear_codex_models_cache()

    @pytest.mark.asyncio
    async def test_success_is_cached(self, monkeypatch):
        calls = []

        async def fake_fetch(binary, *, timeout=10.0):
            calls.append(binary)
            return ["gpt-5.5"]

        monkeypatch.setattr(codex_models, "fetch_codex_models", fake_fetch)
        assert await codex_models.get_codex_models_cached("codex") == ["gpt-5.5"]
        assert await codex_models.get_codex_models_cached("codex") == ["gpt-5.5"]
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_failure_cached_then_retried_after_failure_ttl(
            self, monkeypatch):
        calls = []

        async def fake_fetch(binary, *, timeout=10.0):
            calls.append(binary)
            return None

        clock = [1000.0]
        monkeypatch.setattr(codex_models, "fetch_codex_models", fake_fetch)
        monkeypatch.setattr(codex_models.time, "monotonic",
                            lambda: clock[0])

        assert await codex_models.get_codex_models_cached(
            "codex", failure_ttl=60.0) is None
        assert await codex_models.get_codex_models_cached(
            "codex", failure_ttl=60.0) is None
        assert len(calls) == 1  # failure is negative-cached

        clock[0] += 61.0
        assert await codex_models.get_codex_models_cached(
            "codex", failure_ttl=60.0) is None
        assert len(calls) == 2  # retried after the failure TTL

    @pytest.mark.asyncio
    async def test_success_refetched_after_ttl(self, monkeypatch):
        calls = []

        async def fake_fetch(binary, *, timeout=10.0):
            calls.append(binary)
            return ["gpt-5.5"]

        clock = [1000.0]
        monkeypatch.setattr(codex_models, "fetch_codex_models", fake_fetch)
        monkeypatch.setattr(codex_models.time, "monotonic",
                            lambda: clock[0])

        await codex_models.get_codex_models_cached("codex", ttl=600.0)
        clock[0] += 601.0
        await codex_models.get_codex_models_cached("codex", ttl=600.0)
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_cache_keyed_by_binary(self, monkeypatch):
        calls = []

        async def fake_fetch(binary, *, timeout=10.0):
            calls.append(binary)
            return [binary]

        monkeypatch.setattr(codex_models, "fetch_codex_models", fake_fetch)
        assert await codex_models.get_codex_models_cached("codex-a") == ["codex-a"]
        assert await codex_models.get_codex_models_cached("codex-b") == ["codex-b"]
        assert calls == ["codex-a", "codex-b"]

    @pytest.mark.asyncio
    async def test_clear_forces_refetch(self, monkeypatch):
        calls = []

        async def fake_fetch(binary, *, timeout=10.0):
            calls.append(binary)
            return ["gpt-5.5"]

        monkeypatch.setattr(codex_models, "fetch_codex_models", fake_fetch)
        await codex_models.get_codex_models_cached("codex")
        codex_models.clear_codex_models_cache()
        await codex_models.get_codex_models_cached("codex")
        assert len(calls) == 2


class TestOversizedLines:
    @pytest.mark.asyncio
    async def test_survives_notification_line_beyond_reader_limit(self):
        # Same regression class as CodexTransport._read_loop: a server
        # notification line >limit must be skipped, not crash the probe.
        reader = asyncio.StreamReader(limit=2**16)
        _feed(reader, INIT_RESULT)
        _feed(reader, {"jsonrpc": "2.0", "method": "noise/event",
                       "params": {"blob": "x" * (2**17)}})  # 128 KiB line
        _feed(reader, MODEL_LIST_RESULT)
        reader.feed_eof()

        ids = await codex_models.read_model_ids(reader, FakeWriter(),
                                                timeout=5.0)
        assert ids == ["gpt-5.5", "gpt-5.4-mini", "gpt-5.3-codex"]
