# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Minimal interactive agent for transport testing."""
import sys
while True:
    try:
        line = input()
        print(f"Echo: {line}")
        sys.stdout.flush()
    except EOFError:
        break
