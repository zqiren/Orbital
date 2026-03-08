# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Custom FastAPI middleware for the Agent OS daemon."""

import json

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RelayRedactionMiddleware(BaseHTTPMiddleware):
    """Strip sensitive keys from JSON responses when request arrives via relay.

    When a request contains the X-Via-Relay header, any ``api_key`` field
    in the top-level JSON response (or in list items) is replaced with
    ``"***"``.  This prevents API keys from leaking to the mobile client.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        if request.headers.get("x-via-relay") != "true":
            return response

        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return response

        # Read the response body
        body_chunks = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                body_chunks.append(chunk)
            else:
                body_chunks.append(chunk.encode("utf-8"))
        body_bytes = b"".join(body_chunks)

        try:
            data = json.loads(body_bytes)
        except (json.JSONDecodeError, ValueError):
            return Response(
                content=body_bytes,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        redacted = _redact(data)
        new_body = json.dumps(redacted)

        # Drop content-length so Starlette recalculates it for the new body
        new_headers = {
            k: v for k, v in response.headers.items()
            if k.lower() != "content-length"
        }

        return Response(
            content=new_body,
            status_code=response.status_code,
            headers=new_headers,
            media_type=response.media_type,
        )


def _redact(obj):
    """Recursively strip api_key from dicts and lists."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "api_key":
                out[k] = "***"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(item) for item in obj]
    return obj
