# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""RequestCredentialTool — request website credentials via secure modal."""

import json

from .base import Tool, ToolResult


class RequestCredentialTool(Tool):
    """Request user credentials for a website. Triggers secure input modal."""

    def __init__(self, credential_store):
        self.name = "request_credential"
        self.description = (
            "Request website login credentials from the user. "
            "If credentials already exist, returns secret tokens. "
            "Otherwise, triggers a secure input modal (values never enter chat)."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Identifier for the credential (e.g., 'twitter', 'amazon')",
                },
                "domain": {
                    "type": "string",
                    "description": "Website domain (e.g., 'twitter.com')",
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to request (e.g., ['username', 'password'])",
                },
                "reason": {
                    "type": "string",
                    "description": "Why credentials are needed — shown to user",
                },
            },
            "required": ["name", "domain", "fields", "reason"],
        }
        self._credential_store = credential_store

    def execute(self, **arguments) -> ToolResult:
        try:
            name = arguments["name"]
            domain = arguments["domain"]
            fields = arguments["fields"]
            reason = arguments["reason"]

            # Check if credential already exists
            existing = self._credential_store.get_metadata(name)
            if existing is not None:
                tokens = {f: f"<secret:{name}.{f}>" for f in fields}
                return ToolResult(content=json.dumps({
                    "status": "ready",
                    "name": name,
                    "tokens": tokens,
                    "message": f"Credential '{name}' is stored. Use the <secret:> tokens.",
                }))

            # Credential doesn't exist — return pending with meta signal
            return ToolResult(
                content=json.dumps({
                    "status": "pending",
                    "name": name,
                    "domain": domain,
                    "fields": fields,
                    "reason": reason,
                    "message": "Waiting for user to provide credentials via secure modal.",
                }),
                meta={
                    "credential_request": True,
                    "name": name,
                    "domain": domain,
                    "fields": fields,
                    "reason": reason,
                },
            )
        except Exception as e:
            return ToolResult(content=f"Error: {str(e)}")
