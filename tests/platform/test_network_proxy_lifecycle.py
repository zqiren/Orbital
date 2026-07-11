# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for the network-proxy lifecycle (zombie-proxy fix).

The proxy must survive the death of whatever event loop created it —
ShellTool._run_async spins up throwaway asyncio.run() loops per command.
"""

import asyncio
import socket
import sys
import threading
import time

import pytest


class TestNetworkLoop:
    def test_singleton_returns_same_instance(self):
        from agent_os.platform.shared.network_loop import NetworkLoop

        assert NetworkLoop.get() is NetworkLoop.get()

    def test_run_executes_on_dedicated_thread(self):
        from agent_os.platform.shared.network_loop import NetworkLoop

        async def where_am_i():
            return threading.current_thread().name

        async def go():
            return await NetworkLoop.get().run(where_am_i())

        name = asyncio.run(go())
        assert name == "orbital-network-loop"

    def test_run_survives_caller_loop_exit(self):
        """A coroutine scheduled from loop A still works after loop A closes,
        because it ran on the network loop, not on A."""
        from agent_os.platform.shared.network_loop import NetworkLoop

        results = []

        async def remember(x):
            results.append(x)
            return x

        async def go():
            return await NetworkLoop.get().run(remember("first"))

        assert asyncio.run(go()) == "first"      # loop A now closed
        assert asyncio.run(go()) == "first"      # loop B works identically
        assert results == ["first", "first"]


class TestProxySurvivesCreatorLoop:
    def test_proxy_serves_after_creator_loop_closes(self):
        """THE zombie-proxy regression test.

        Pre-fix: the socket connects (kernel backlog) but no bytes ever
        arrive — recv() times out. Post-fix: instant 403 for a blocked
        domain, served by the network loop.
        """
        from agent_os.platform.shared.network import NetworkProxy
        from agent_os.platform.types import NetworkRules

        def create_in_ephemeral_loop() -> "NetworkProxy":
            async def go():
                p = NetworkProxy(project_id="zombie_test")
                p.set_rules(NetworkRules(mode="allowlist", domains=["allowed.example"]))
                await p.start()
                return p

            return asyncio.run(go())  # creator loop is CLOSED on return

        p = create_in_ephemeral_loop()
        try:
            with socket.create_connection(("127.0.0.1", p.port), timeout=2) as s:
                s.sendall(b"CONNECT x.com:443 HTTP/1.1\r\nHost: x.com:443\r\n\r\n")
                s.settimeout(2)
                data = s.recv(1024)
            assert b"403" in data, f"expected 403, got: {data!r}"
        finally:
            asyncio.run(p.stop())

    def test_stop_works_from_a_different_loop(self):
        from agent_os.platform.shared.network import NetworkProxy

        async def make():
            p = NetworkProxy(project_id="stop_test")
            await p.start()
            return p

        p = asyncio.run(make())
        port = p.port
        asyncio.run(p.stop())  # different asyncio.run() loop than start()
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=1)


@pytest.mark.skipif(sys.platform != "darwin", reason="MacOSProvider is darwin-only")
class TestRunCommandDoesNotBlockLoop:
    @pytest.mark.asyncio
    async def test_event_loop_stays_responsive_during_run_command(
        self, monkeypatch, tmp_path
    ):
        from agent_os.platform.macos import provider as macos_provider

        prov = macos_provider.MacOSPlatformProvider()

        try:
            def slow_run(*args, **kwargs):          # stands in for sandbox-exec
                time.sleep(0.6)

                class R:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""

                return R()

            monkeypatch.setattr(macos_provider.subprocess, "run", slow_run)

            ticks = []

            async def ticker():
                for _ in range(6):
                    ticks.append(time.monotonic())
                    await asyncio.sleep(0.1)

            t = asyncio.create_task(ticker())
            result = await prov.run_command("p1", "/bin/echo", ["hi"], str(tmp_path))
            await t

            assert result.exit_code == 0
            gaps = [b - a for a, b in zip(ticks, ticks[1:])]
            assert max(gaps) < 0.4, f"event loop starved during run_command: {gaps}"
        finally:
            await prov.teardown()


async def _raw_request(port: int, payload: bytes) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(payload)
    await writer.drain()
    data = await asyncio.wait_for(reader.read(4096), timeout=2)
    writer.close()
    return data


class TestLegibleBlockedResponse:
    @pytest.mark.asyncio
    async def test_blocked_connect_gets_403_with_policy_body(self):
        from agent_os.platform.shared.network import NetworkProxy
        from agent_os.platform.types import NetworkRules

        p = NetworkProxy(project_id="legible_test")
        p.set_rules(NetworkRules(mode="allowlist", domains=["allowed.example"]))
        await p.start()
        try:
            data = await _raw_request(
                p.port, b"CONNECT x.com:443 HTTP/1.1\r\nHost: x.com:443\r\n\r\n"
            )
            assert b"403" in data
            assert b"X-Orbital-Blocked: policy" in data
            assert b"Orbital network policy" in data
            assert b"x.com" in data
            assert b"browser tool" in data
        finally:
            await p.stop()

    @pytest.mark.asyncio
    async def test_blocked_plain_http_gets_policy_body(self):
        from agent_os.platform.shared.network import NetworkProxy
        from agent_os.platform.types import NetworkRules

        p = NetworkProxy(project_id="legible_http_test")
        p.set_rules(NetworkRules(mode="allowlist", domains=["allowed.example"]))
        await p.start()
        try:
            data = await _raw_request(
                p.port,
                b"GET http://x.com/ HTTP/1.1\r\nHost: x.com\r\n\r\n",
            )
            assert b"403" in data
            assert b"Orbital network policy" in data
        finally:
            await p.stop()


class TestExpandedDefaults:
    @pytest.mark.parametrize(
        "domain",
        [
            # GitHub, properly wildcarded (api./gist./codeload. were blocked before)
            "github.com",
            "api.github.com",
            "gist.github.com",
            "codeload.github.com",
            "raw.githubusercontent.com",
            "objects.githubusercontent.com",
            # providers the product actually ships
            "api.moonshot.cn",
            "api.moonshot.ai",
            "api.minimaxi.com",
            "api.minimax.io",
            # package ecosystems
            "pypi.org",
            "files.pythonhosted.org",
            "registry.npmjs.org",
            "registry.yarnpkg.com",
            "crates.io",
            "static.crates.io",
            "index.crates.io",
            "proxy.golang.org",
            "sum.golang.org",
            # model downloads
            "huggingface.co",
            "cdn-lfs.huggingface.co",
            "hf.co",
            "cdn-lfs-us-1.hf.co",
        ],
    )
    def test_default_allowlist_covers(self, domain):
        from agent_os.platform.shared.network import NetworkProxy
        from agent_os.platform.types import DEFAULT_ALLOWLIST_DOMAINS

        assert NetworkProxy._matches_any(domain, DEFAULT_ALLOWLIST_DOMAINS), domain

    @pytest.mark.parametrize(
        "domain",
        [
            "x.com",
            "evil.example",
            "github.com.evil.example",   # suffix spoof must not match
            "notgithub.com",
        ],
    )
    def test_default_allowlist_blocks(self, domain):
        from agent_os.platform.shared.network import NetworkProxy
        from agent_os.platform.types import DEFAULT_ALLOWLIST_DOMAINS

        assert not NetworkProxy._matches_any(domain, DEFAULT_ALLOWLIST_DOMAINS), domain
