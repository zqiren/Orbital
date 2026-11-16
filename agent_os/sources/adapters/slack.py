# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Slack bot, by Socket Mode — an outbound WebSocket, no public address.

Protocol (https://docs.slack.dev/apis/events-api/using-socket-mode):
``apps.connections.open`` with the app-level token (``xapp-``,
``connections:write``) returns a one-time ``wss://`` URL; the server says
``hello``, then sends envelopes ``{"envelope_id", "type": "events_api",
"payload": <Events API callback>}``, each of which must be acknowledged by
sending ``{"envelope_id"}`` back (otherwise Slack retries it). A
``disconnect`` message (``refresh_requested`` every few hours, ``warning``
shortly before, ``link_disabled`` when Socket Mode is switched off) ends the
connection: the core opens a new one with a fresh URL. The bot token
(``xoxb-``) is used for ``auth.test`` (who the bot is) and the name lookups.

Lossy: Slack does not replay events sent while no connection was open, so
the core records gaps.

Arrivals: ``message`` events in an IM with the bot (``dm``) and
``app_mention`` (``at``). The bot's own messages, edits, deletions, joins
and other subtypes that are not a person's new message are ignored.
"""

from __future__ import annotations

import asyncio
import html
import json
import re

from agent_os.sources.adapters import Adapter
from agent_os.sources.adapters._bot_common import (
    NameCache, SeenIds, Unreachable, endpoint, host_of, http_client, iso_from_epoch, placeholder,
    split_text, ws_connect,
)
from agent_os.sources.model import Item, SourceFailure, Verified, VerifyError

API = "https://slack.com/api"
_REJECTED = {"invalid_auth", "not_authed", "account_inactive", "token_revoked", "token_expired",
             "no_permission"}
_USER_SUBTYPES = {None, "file_share", "thread_broadcast"}

#: users:read / channels:read for sender and channel names (channels:history
#: is not needed: only app_mention and message.im are used); the App Home
#: Messages tab must allow messages or nobody can DM the bot.

_USER_RE = re.compile(r"<@([UW][A-Z0-9]+)(?:\|[^>]*)?>")
_CHAN_RE = re.compile(r"<#([CG][A-Z0-9]+)(?:\|([^>]*))?>")
_SPECIAL_RE = re.compile(r"<!(here|channel|everyone)(?:\|[^>]*)?>")
_SUBTEAM_RE = re.compile(r"<!subteam\^[A-Z0-9]+(?:\|([^>]*))?>")
_LINK_RE = re.compile(r"<((?:https?|mailto):[^>|]+)(?:\|([^>]*))?>")


class SlackAdapter(Adapter):
    mode = "stream"
    lossless = False
    api_base: str | None = None
    ping_interval: float | None = 30.0

    def __init__(self):
        self._names = NameCache()

    def _base(self) -> str:
        return self.api_base or endpoint("slack", API)

    async def _api(self, client, token: str, method: str, params: dict | None = None,
                   *, get: bool = False) -> dict:
        import httpx
        base = self._base()
        headers = {"Authorization": f"Bearer {token}"}
        try:
            if get:
                r = await client.get(f"{base}/{method}", params=params or {}, headers=headers)
            else:
                r = await client.post(f"{base}/{method}", data=params or {}, headers=headers)
        except httpx.HTTPError:
            raise Unreachable(host_of(base)) from None
        if r.status_code == 429:
            return {"ok": False, "error": "ratelimited"}
        try:
            data = r.json()
        except ValueError:
            return {"ok": False, "error": f"http_{r.status_code}"}
        return data if isinstance(data, dict) else {"ok": False, "error": "bad_response"}

    # -- connect ------------------------------------------------------------

    async def verify(self, ctx) -> Verified:
        app = (ctx.secret("app") or "").strip()
        bot = (ctx.secret("bot") or "").strip()
        if not app.startswith("xapp-"):
            raise VerifyError("invalid", "app", "the app-level token starts with xapp-")
        if not bot.startswith("xoxb-"):
            raise VerifyError("invalid", "bot", "the bot token starts with xoxb-")
        try:
            async with http_client() as client:
                who = await self._api(client, bot, "auth.test")
                if not who.get("ok"):
                    err = str(who.get("error") or "error")
                    raise VerifyError("rejected" if err in _REJECTED else "error", "bot",
                                      f"Slack refused the bot token ({err})")
                conn = await self._api(client, app, "apps.connections.open")
                if not conn.get("ok"):
                    err = str(conn.get("error") or "error")
                    reason = ("rejected" if err in _REJECTED
                              else "invalid" if err == "not_allowed_token_type" else "error")
                    raise VerifyError(reason, "app", f"Slack refused the app-level token ({err})")
        except Unreachable as e:
            raise VerifyError("unreachable", None, f"cannot reach {e}") from None
        name, team = who.get("user"), who.get("team")
        return Verified(label=" · ".join(p for p in (name, team) if p) or None)

    # -- the socket -----------------------------------------------------------

    async def stream(self, ctx) -> None:
        import websockets
        app = (ctx.secret("app") or "").strip()
        bot = (ctx.secret("bot") or "").strip()
        async with http_client() as client:
            who = await self._api(client, bot, "auth.test")
            if not who.get("ok"):
                if who.get("error") in _REJECTED:
                    raise SourceFailure("token_rejected", {"field": "bot"}, retryable=False)
                if who.get("error") == "ratelimited":
                    raise SourceFailure("rate_limited", {})
                raise Unreachable(f"slack auth.test: {who.get('error')}")
            conn = await self._api(client, app, "apps.connections.open")
            if not conn.get("ok"):
                err = conn.get("error")
                if err in _REJECTED or err == "not_allowed_token_type":
                    raise SourceFailure("token_rejected", {"field": "app"}, retryable=False)
                if err == "ratelimited":
                    raise SourceFailure("rate_limited", {})
                raise Unreachable(f"slack apps.connections.open: {err}")
            me = {"user_id": who.get("user_id"), "bot_id": who.get("bot_id"),
                  "url": who.get("url") or ""}
            seen = SeenIds()
            try:
                async with ws_connect(conn["url"], ping_interval=self.ping_interval) as ws:
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except (TypeError, ValueError):
                            continue
                        if not isinstance(msg, dict):
                            continue
                        mtype = msg.get("type")
                        if msg.get("envelope_id"):
                            # Ack first: Slack retries an envelope not acked in 3 s.
                            await ws.send(json.dumps({"envelope_id": msg["envelope_id"]}))
                        if mtype == "hello":
                            ctx.connected()
                        elif mtype == "disconnect":
                            reason = msg.get("reason")
                            if reason == "warning":
                                continue  # the real disconnect follows
                            if reason == "link_disabled":
                                raise SourceFailure("error", {"reason": "socket_mode_off"})
                            return  # refresh: the core reconnects with a fresh URL
                        elif mtype == "events_api":
                            item = await self._item(client, bot, me, msg.get("payload") or {})
                            if item is not None and seen.add(item.id):
                                await ctx.emit([item])
            except websockets.ConnectionClosed:
                raise Unreachable("slack socket closed") from None
            except Unreachable:
                raise
            except (OSError, websockets.InvalidHandshake, asyncio.TimeoutError):
                raise Unreachable("cannot open the slack socket") from None
            raise Unreachable("slack socket closed")

    # -- one event → one item ---------------------------------------------------

    async def _user_name(self, client, bot: str, uid: str) -> str | None:
        async def _load():
            r = await self._api(client, bot, "users.info", {"user": uid}, get=True)
            u = r.get("user") or {}
            prof = u.get("profile") or {}
            return (prof.get("display_name") or prof.get("real_name") or u.get("real_name")
                    or u.get("name")) or None
        return await self._names.get(("u", uid), _load)

    async def _channel_name(self, client, bot: str, cid: str) -> str | None:
        async def _load():
            r = await self._api(client, bot, "conversations.info", {"channel": cid}, get=True)
            name = (r.get("channel") or {}).get("name")
            return f"#{name}" if name else None
        return await self._names.get(("c", cid), _load)

    async def _plain(self, client, bot: str, me: dict, text: str) -> str:
        async def _user(m):
            uid = m.group(1)
            if uid == me.get("user_id"):
                return ""
            name = await self._user_name(client, bot, uid)
            return f"@{name or uid}"
        out, pos = [], 0
        for m in _USER_RE.finditer(text):
            out.append(text[pos:m.start()])
            out.append(await _user(m))
            pos = m.end()
        out.append(text[pos:])
        s = "".join(out)
        s = _CHAN_RE.sub(lambda m: f"#{m.group(2)}" if m.group(2) else "#channel", s)
        s = _SPECIAL_RE.sub(lambda m: f"@{m.group(1)}", s)
        s = _SUBTEAM_RE.sub(lambda m: m.group(1) or "@group", s)
        s = _LINK_RE.sub(lambda m: m.group(2) or m.group(1), s)
        s = html.unescape(s)
        return re.sub(r"[ \t]{2,}", " ", s).strip()

    async def _item(self, client, bot: str, me: dict, payload: dict) -> Item | None:
        if payload.get("type") != "event_callback":
            return None
        ev = payload.get("event") or {}
        etype = ev.get("type")
        if ev.get("bot_id") or (me.get("user_id") and ev.get("user") == me["user_id"]):
            return None
        if not ev.get("user"):
            return None
        if etype == "message":
            if ev.get("channel_type") != "im" or ev.get("subtype") not in _USER_SUBTYPES:
                return None
            kind, place = "dm", {"kind": "dm"}
        elif etype == "app_mention":
            if ev.get("edited") or ev.get("subtype") not in _USER_SUBTYPES:
                return None
            kind = "at"
            place = {"kind": "channel", "name": await self._channel_name(client, bot, ev.get("channel") or "")}
        else:
            return None
        text = await self._plain(client, bot, me, ev.get("text") or "")
        files = ev.get("files") or []
        if files:
            marks = " ".join(placeholder("image") if str(f.get("mimetype") or "").startswith("image/")
                             else placeholder("file") for f in files if isinstance(f, dict))
            text = f"{marks} {text}".strip()
        title, body = split_text(text, placeholder("message"))
        channel, ts = ev.get("channel") or "", ev.get("ts") or ""
        url = None
        if me.get("url") and channel and ts:
            url = f"{me['url'].rstrip('/')}/archives/{channel}/p{ts.replace('.', '')}"
            if ev.get("thread_ts") and ev["thread_ts"] != ts:
                url += f"?thread_ts={ev['thread_ts']}&cid={channel}"
        return Item(id=f"{channel}:{ts}", kind=kind, title=title, body=body,
                    sender=await self._user_name(client, bot, ev["user"]), place=place, url=url,
                    at=iso_from_epoch(ts))


ADAPTER = SlackAdapter
