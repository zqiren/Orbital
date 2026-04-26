# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration tests package.

Holds the live-daemon test harness (see ``tests/integration/harness``) and
tests that exercise a real uvicorn subprocess. These tests are gated behind
the ``live_daemon`` pytest marker.
"""
