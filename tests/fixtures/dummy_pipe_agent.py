# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Minimal pipe-mode agent for transport testing. No LLM needed."""
import sys, json
msg = ""
for i, arg in enumerate(sys.argv):
    if arg == "-p" and i + 1 < len(sys.argv):
        msg = sys.argv[i + 1]
print(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": f"Echo: {msg}"}]}}))
print(json.dumps({"type": "result", "subtype": "success", "session_id": "test-session-001"}))
