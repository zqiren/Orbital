# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared pieces of the six bot adapters (Telegram, Slack, Discord, Feishu,
DingTalk, WeCom). Each bot holds one OUTBOUND connection to its service;
nothing here listens on a port and nothing ever sends a reply into a chat.

Only the standard library, ``httpx`` and ``websockets`` (both already daemon
dependencies) are used. Secrets reach this code through ``ctx.secret`` and
must never appear in a log line or an exception message: every message built
here passes through :func:`redact`, and httpx's own request log line (which
would carry Telegram's token, it lives in the URL path) is redacted by a
filter installed once.
"""

from __future__ import annotations

import logging
import os
import re
from collections import OrderedDict
from datetime import datetime, timezone

logger = logging.getLogger("agent_os.sources.bots")

#: Test-only endpoint override: ``AGENT_OS_SOURCES_FAKE_TELEGRAM=http://127.0.0.1:9999``
#: points that type's adapter at a local protocol fake. Never set by the app,
#: never reachable from the catalogue or the connect form.
FAKE_ENV_PREFIX = "AGENT_OS_SOURCES_FAKE_"

HTTP_TIMEOUT_S = 20.0
USER_AGENT = "Orbital-sources"

PLACEHOLDERS = {
    "image": "[image]", "photo": "[image]", "picture": "[image]", "sticker": "[sticker]",
    "file": "[file]", "document": "[file]", "audio": "[audio]", "voice": "[voice]",
    "video": "[video]", "media": "[video]", "video_note": "[video]", "animation": "[animation]",
    "location": "[location]", "contact": "[contact]", "poll": "[poll]", "card": "[card]",
    "interactive": "[card]", "share_chat": "[shared chat]", "share_user": "[shared contact]",
    "merge_forward": "[forwarded messages]", "post": "[rich text]", "richText": "[rich text]",
}



def placeholder(kind: str) -> str:
    """A short typed stand-in for a message that is not plain text."""
    return PLACEHOLDERS.get(kind) or f"[{kind or 'message'}]"


def endpoint(type_: str, default: str) -> str:
    """The service's base address, unless a test points it at a fake."""
    v = os.environ.get(FAKE_ENV_PREFIX + type_.upper())
    return v.rstrip("/") if v else default


def redact(text, *secrets) -> str:
    """``text`` with every secret (and anything shaped like a Telegram bot
    token) replaced by a fixed mask."""
    s = "" if text is None else str(text)
    for sec in secrets:
        if sec and len(str(sec)) >= 4:
            s = s.replace(str(sec), "••••")
    return _scrub(s)


_TG_TOKEN_RE = re.compile(r"(bot)\d{5,}:[A-Za-z0-9_-]{20,}")
#: One-time connection tickets / keys some services put in the WebSocket URL
#: (Slack ``ticket``, DingTalk ``ticket``, Feishu ``access_key``/``ticket``).
_QUERY_SECRET_RE = re.compile(r"((?:ticket|access_key|token|secret)=)[^&\s\"']+", re.I)


#: Secrets inside JSON frames (Discord identify ``token``, WeCom subscribe
#: ``secret``, Feishu/DingTalk ``AppSecret``/``clientSecret``): websockets logs
#: the start of every frame at DEBUG. No closing quote needed (truncated logs).
_JSON_SECRET_RE = re.compile(
    r"""(["'](?:token|secret|appsecret|app_secret|clientsecret|client_secret)["']\s*:\s*["'])[^"']*""", re.I)


def _scrub(msg: str) -> str:
    msg = _TG_TOKEN_RE.sub(r"\1••••", msg)
    msg = _QUERY_SECRET_RE.sub(r"\1••••", msg)
    return _JSON_SECRET_RE.sub(r"\1••••", msg)


class _RedactFilter(logging.Filter):
    """httpx logs ``HTTP Request: GET https://api.telegram.org/bot<token>/…``
    at INFO and the daemon's file log takes INFO; websockets logs the
    handshake request line (with a connection ticket) at DEBUG. Rewrite the
    record so neither reaches a handler."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            msg = record.getMessage()
        except Exception:
            return True
        clean = _scrub(msg)
        if clean != msg:
            record.msg = clean
            record.args = None
        return True


_FILTER = _RedactFilter()


def install_log_redaction() -> None:
    """Idempotent: attach the redaction filter to the HTTP/WebSocket
    libraries' loggers."""
    for name in ("httpx", "httpcore", "websockets.client"):
        lg = logging.getLogger(name)
        if _FILTER not in lg.filters:
            lg.addFilter(_FILTER)


def http_client(timeout: float = HTTP_TIMEOUT_S, **kw):
    import httpx
    install_log_redaction()
    headers = {"User-Agent": USER_AGENT}
    headers.update(kw.pop("headers", {}) or {})
    return httpx.AsyncClient(timeout=timeout, headers=headers, **kw)


def host_of(url: str) -> str:
    from urllib.parse import urlparse
    return urlparse(url).hostname or url


class NameCache:
    """One lookup per id: names of senders, groups and channels when the
    event carries only ids. A failed lookup is cached too (as ``None``) so a
    service that refuses the lookup is asked once, not per message."""

    def __init__(self, cap: int = 1000):
        self._d: OrderedDict = OrderedDict()
        self._cap = cap

    def peek(self, key):
        return self._d.get(key)

    def put(self, key, value) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self._cap:
            self._d.popitem(last=False)

    async def get(self, key, loader):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        try:
            value = await loader()
        except Exception:
            value = None
        self.put(key, value)
        return value


class SeenIds:
    """A small per-connection memory of delivered ids, so a service that
    re-delivers an event (Slack retries, DingTalk resends an un-acked
    callback) is emitted once even before the core's own de-duplication."""

    def __init__(self, cap: int = 2000):
        self._d: OrderedDict = OrderedDict()
        self._cap = cap

    def add(self, key) -> bool:
        """True when ``key`` is new."""
        if key in self._d:
            return False
        self._d[key] = True
        while len(self._d) > self._cap:
            self._d.popitem(last=False)
        return True


def split_text(text: str, fallback: str) -> tuple[str, str]:
    """(title, body) for a chat message: the first non-empty line is the
    title, the whole text is the body. An empty text uses ``fallback``."""
    s = (text or "").strip()
    if not s:
        return fallback, fallback
    first = next((ln.strip() for ln in s.splitlines() if ln.strip()), s)
    return first, s


def iso_from_epoch(value, *, millis: bool = False) -> str | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if millis:
        v /= 1000.0
    try:
        return datetime.fromtimestamp(v, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OverflowError, OSError, ValueError):
        return None


def close_code(exc) -> int | None:
    """The close code a ``websockets`` ConnectionClosed carries, across
    library versions (``rcvd.code`` in 13+, ``code`` before)."""
    rcvd = getattr(exc, "rcvd", None)
    code = getattr(rcvd, "code", None) if rcvd is not None else None
    if code is None:
        try:
            code = getattr(exc, "code", None)
        except Exception:
            code = None
    return code if isinstance(code, int) else None


_TLS = None


def tls_context():
    """The CA bundle httpx already trusts (certifi), unless SSL_CERT_FILE
    says otherwise. Python's own default store is empty on a python.org
    build without "Install Certificates", which made every ``wss://``
    handshake fail certificate verification on this Mac (2026-11-16)."""
    global _TLS
    if _TLS is None:
        import ssl
        cafile = os.environ.get("SSL_CERT_FILE")
        if not cafile:
            try:
                import certifi
                cafile = certifi.where()
            except Exception:
                cafile = None
        _TLS = ssl.create_default_context(cafile=cafile)
    return _TLS


def ws_connect(url: str, **kw):
    """``websockets.connect`` with our defaults. The caller owns pings
    (``ping_interval=None``) only where the protocol has its own heartbeat;
    otherwise the library's protocol-level pings keep the socket honest."""
    import websockets
    install_log_redaction()
    if url.startswith("wss://") and "ssl" not in kw:
        kw["ssl"] = tls_context()
    kw.setdefault("open_timeout", 20)
    kw.setdefault("close_timeout", 5)
    kw.setdefault("max_size", 8 * 1024 * 1024)
    return websockets.connect(url, **kw)


def ws_invalid_status():
    """The handshake-refused exception class across websockets versions
    (``InvalidStatus`` in 14+, ``InvalidStatusCode`` before)."""
    import websockets
    return getattr(websockets, "InvalidStatus", None) or getattr(websockets, "InvalidStatusCode", OSError)


def ws_status_headers(exc) -> dict:
    """Response headers of a refused handshake (``exc.response.headers`` in
    14+, ``exc.headers`` before)."""
    hs = getattr(getattr(exc, "response", None), "headers", None)
    if hs is None:
        hs = getattr(exc, "headers", None)
    return hs if hs is not None else {}


class Unreachable(ConnectionError):
    """The service could not be reached / the connection dropped. The
    message carries the host only, never a URL with a token or ticket."""
