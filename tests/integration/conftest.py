# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pytest configuration for the live-daemon integration harness.

Registers the ``live_daemon`` marker so these tests can be opted into
with ``-m live_daemon``. The marker is declared here (instead of in
``pyproject.toml``) to keep the harness self-contained.
"""

from __future__ import annotations

import pytest

from tests.integration.harness import ApiClient, DaemonProcess


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live_daemon: test spawns a real uvicorn subprocess (skip with "
        "'-m \"not live_daemon\"')",
    )


# -- Fixtures -------------------------------------------------------------- #


@pytest.fixture(scope="module")
def daemon():
    """Spawn a real daemon for a module's worth of tests."""
    d = DaemonProcess()
    d.start()
    try:
        yield d
    finally:
        d.shutdown()


@pytest.fixture
async def api_client(daemon):
    """Async ApiClient bound to the module-scoped daemon."""
    async with ApiClient(daemon.base_url) as client:
        yield client
