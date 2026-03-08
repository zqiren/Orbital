# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Minimal ACP agent for transport testing. Implements JSON-RPC 2.0 over stdio."""
import json
import sys


def write_response(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    session_id = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = msg.get("method", "")
        msg_id = msg.get("id")
        params = msg.get("params", {})

        if method == "initialize":
            write_response({
                "jsonrpc": "2.0", "id": msg_id,
                "result": {
                    "protocolVersion": 1,
                    "agentInfo": {"name": "dummy-acp", "version": "0.1.0"},
                    "agentCapabilities": {},
                },
            })
        elif method == "session/new":
            session_id = "dummy-sess-001"
            write_response({
                "jsonrpc": "2.0", "id": msg_id,
                "result": {"sessionId": session_id},
            })
        elif method == "session/prompt":
            prompt_text = ""
            for item in params.get("prompt", []):
                if item.get("kind") == "text":
                    prompt_text = item.get("text", "")
            # Send a text update notification
            write_response({
                "jsonrpc": "2.0", "method": "session/update",
                "params": {
                    "sessionId": session_id,
                    "update": {"kind": "text", "text": f"Echo: {prompt_text}"},
                },
            })
            # Send prompt response
            write_response({
                "jsonrpc": "2.0", "id": msg_id,
                "result": {"stopReason": "endTurn"},
            })
        elif method == "shutdown":
            write_response({
                "jsonrpc": "2.0", "id": msg_id, "result": {},
            })
            break


if __name__ == "__main__":
    main()
