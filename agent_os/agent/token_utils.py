# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Token estimation utilities shared across context and session."""

from __future__ import annotations

import json


def estimate_message_tokens(msg: dict) -> float:
    """Estimate token count for a message, handling multimodal content correctly.

    For detail:low images, count 85 tokens flat instead of len(base64)/4.
    """
    content = msg.get("content")
    if isinstance(content, list):
        tokens = 0.0
        for block in content:
            if isinstance(block, dict):
                if block.get("type") in ("image_url", "image"):
                    tokens += 85  # detail:low = 85 tokens flat
                elif block.get("type") == "text":
                    tokens += len(block.get("text", "")) / 4
                else:
                    tokens += len(json.dumps(block)) / 4
            else:
                tokens += len(str(block)) / 4
        # Add overhead for role, meta, etc.
        msg_copy = {k: v for k, v in msg.items() if k != "content"}
        tokens += len(json.dumps(msg_copy)) / 4
        return tokens
    return len(json.dumps(msg)) / 4
