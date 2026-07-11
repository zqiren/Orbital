# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A single process-lifetime event loop for network proxy servers.

NetworkProxy servers must outlive whichever event loop happens to create
them: ShellTool._run_async executes provider calls inside short-lived
``asyncio.run()`` loops on worker threads, and a server bound to such a
loop becomes a zombie when the loop closes (the listening socket leaks,
the kernel keeps accepting connections, nothing ever serves them).

All proxy servers therefore run on this one daemon-thread loop, whose
lifetime matches the process.
"""

import asyncio
import threading


class NetworkLoop:
    """Singleton owner of the dedicated network event loop."""

    _instance: "NetworkLoop | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="orbital-network-loop",
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def get(cls) -> "NetworkLoop":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    async def run(self, coro):
        """Execute *coro* on the network loop; awaitable from any other loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return await asyncio.wrap_future(future)
