# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for the network-proxy lifecycle (zombie-proxy fix).

The proxy must survive the death of whatever event loop created it —
ShellTool._run_async spins up throwaway asyncio.run() loops per command.
"""

import asyncio
import socket
import threading

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
