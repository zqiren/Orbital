# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Discord bot, by the Gateway WebSocket — outbound, no public address.

Protocol (https://discord.com/developers/docs/events/gateway, read from the
docs repository on 2026-11-16): ``GET /gateway/bot`` gives the URL; the
server sends Hello (op 10, ``heartbeat_interval``); the client heartbeats
(op 1 with the last sequence) first after ``interval × random()``, then every
interval, and treats a missing Heartbeat ACK (op 11) as a dead connection;
it identifies (op 2) or, after a drop, resumes (op 6 with ``session_id`` and
sequence on ``resume_gateway_url``), so events missed during a short drop
are replayed. Reconnect (op 7) and Invalid Session (op 9, ``d`` = resumable)
end the connection; the core reconnects. Close codes 4004 (bad token) and
4010–4014 cannot succeed by retrying: a non-retryable failure.

Intents: GUILDS | GUILD_MESSAGES | DIRECT_MESSAGES. No privileged
MESSAGE_CONTENT intent: Discord still delivers the content of DMs to the
bot and of messages that mention it, which are the only two arrivals, so
the user has no portal toggle to find.

Lossy across longer outages (a session that cannot be resumed loses what
happened meanwhile): the core records gaps.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import sys

from agent_os.sources.adapters import Adapter
from agent_os.sources.adapters._bot_common import (
    NameCache, SeenIds, Unreachable, close_code, endpoint, host_of, http_client, placeholder,
    split_text, ws_connect,
)
from agent_os.sources.model import Item, SourceFailure, Verified, VerifyError

API = "https://discord.com/api/v10"
GATEWAY_QUERY = "v=10&encoding=json"
INTENTS = (1 << 0) | (1 << 9) | (1 << 12)  # GUILDS, GUILD_MESSAGES, DIRECT_MESSAGES
_FATAL_CLOSE = {4010, 4011, 4012, 4013, 4014}
_FRESH_CLOSE = {4007, 4009}  # reconnect, but identify again
_USER_TYPES = {0, 19}  # DEFAULT, REPLY

#: No Message Content Intent: Discord delivers the content of DMs and of
#: messages that mention the bot without it (see the module docstring).

_MENTION_RE = re.compile(r"<@!?(\d+)>")
_ROLE_RE = re.compile(r"<@&(\d+)>")
_CHAN_RE = re.compile(r"<#(\d+)>")
_EMOJI_RE = re.compile(r"<a?:(\w+):\d+>")


class _Zombie(Exception):
    pass


class DiscordAdapter(Adapter):
    mode = "stream"
    lossless = False
    api_base: str | None = None

    def __init__(self):
        # source id -> {"url", "session_id", "seq", "bot_id"}: what a resume needs.
        # In memory only: a session outlives a short drop, not a daemon restart.
        self._resume: dict = {}
        self._channels = NameCache()

    def _base(self) -> str:
        return self.api_base or endpoint("discord", API)

    async def _rest(self, client, token: str, path: str) -> tuple[int, dict]:
        import httpx
        base = self._base()
        try:
            r = await client.get(f"{base}{path}", headers={"Authorization": f"Bot {token}"})
        except httpx.HTTPError:
            raise Unreachable(host_of(base)) from None
        try:
            data = r.json()
        except ValueError:
            data = {}
        return r.status_code, data if isinstance(data, dict) else {}

    # -- connect ------------------------------------------------------------

    async def verify(self, ctx) -> Verified:
        token = (ctx.secret("token") or "").strip()
        if token.lower().startswith("bot "):
            token = token[4:].strip()
        if not token or " " in token:
            raise VerifyError("invalid", "token", "paste the bot token from the Developer Portal")
        try:
            async with http_client() as client:
                status, me = await self._rest(client, token, "/users/@me")
        except Unreachable as e:
            raise VerifyError("unreachable", None, f"cannot reach {e}") from None
        if status == 401:
            raise VerifyError("rejected", "token", "Discord refused this token")
        if status != 200:
            raise VerifyError("error", "token", f"Discord answered HTTP {status}")
        if not me.get("bot"):
            raise VerifyError("invalid", "token", "this is not a bot token")
        return Verified(label=me.get("username") or None)

    # -- the gateway ----------------------------------------------------------

    async def stream(self, ctx) -> None:
        import websockets
        token = (ctx.secret("token") or "").strip()
        if token.lower().startswith("bot "):
            token = token[4:].strip()
        key = ctx.source_id or "_"
        res = self._resume.get(key)
        async with http_client() as client:
            if res:
                url = res["url"]
            else:
                status, g = await self._rest(client, token, "/gateway/bot")
                if status == 401:
                    raise SourceFailure("token_rejected", {"field": "token"}, retryable=False)
                if status == 429:
                    raise SourceFailure("rate_limited", {})
                if status != 200 or not g.get("url"):
                    raise Unreachable(f"discord /gateway/bot answered HTTP {status}")
                url = g["url"]
            ws_url = url.rstrip("/") + "/?" + GATEWAY_QUERY
            hb_state = {"seq": res["seq"] if res else None, "acked": True, "zombie": False}
            seen = SeenIds()
            try:
                async with ws_connect(ws_url, ping_interval=None) as ws:
                    hello = json.loads(await asyncio.wait_for(ws.recv(), 30))
                    if hello.get("op") != 10:
                        raise Unreachable("discord gateway sent no hello")
                    interval = float((hello.get("d") or {}).get("heartbeat_interval") or 41250) / 1000.0
                    hb = asyncio.create_task(self._heartbeat(ws, interval, hb_state))
                    try:
                        if res:
                            await ws.send(json.dumps({"op": 6, "d": {
                                "token": token, "session_id": res["session_id"], "seq": res["seq"]}}))
                        else:
                            await ws.send(json.dumps({"op": 2, "d": {
                                "token": token, "intents": INTENTS,
                                "properties": {"os": sys.platform, "browser": "orbital", "device": "orbital"}}}))
                        end = await self._receive(ctx, client, token, ws, key, hb_state, seen)
                    finally:
                        hb.cancel()
                    if end == "return":
                        return
            except websockets.ConnectionClosed as e:
                self._closed(key, close_code(e))
                return
            except Unreachable:
                raise
            except (OSError, websockets.InvalidHandshake, asyncio.TimeoutError):
                raise Unreachable("cannot open the discord gateway") from None
            if hb_state["zombie"]:
                raise Unreachable("discord heartbeat was not acknowledged")
            self._closed(key, getattr(ws, "close_code", None))

    def _closed(self, key, code) -> None:
        if code == 4004:
            self._resume.pop(key, None)
            raise SourceFailure("token_rejected", {"field": "token"}, retryable=False)
        if code in _FATAL_CLOSE:
            self._resume.pop(key, None)
            raise SourceFailure("error", {"close_code": code}, retryable=False)
        if code in _FRESH_CLOSE:
            self._resume.pop(key, None)
            return
        raise Unreachable(f"discord gateway closed ({code})")

    @staticmethod
    async def _heartbeat(ws, interval: float, st: dict) -> None:
        try:
            await asyncio.sleep(interval * random.random())
            while True:
                if not st["acked"]:
                    st["zombie"] = True
                    await ws.close(4000, "heartbeat")  # non-1000: the session stays resumable
                    return
                st["acked"] = False
                await ws.send(json.dumps({"op": 1, "d": st["seq"]}))
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def _receive(self, ctx, client, token, ws, key, st, seen) -> str | None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except (TypeError, ValueError):
                continue
            op, d = msg.get("op"), msg.get("d")
            if msg.get("s") is not None:
                st["seq"] = msg["s"]
                if key in self._resume:
                    self._resume[key]["seq"] = msg["s"]
            if op == 11:
                st["acked"] = True
            elif op == 1:
                await ws.send(json.dumps({"op": 1, "d": st["seq"]}))
            elif op == 7:
                await ws.close(4000, "reconnect")
                return "return"
            elif op == 9:
                if not d:
                    self._resume.pop(key, None)
                    await asyncio.sleep(random.uniform(1, 5))
                await ws.close(4000, "invalid session")
                return "return"
            elif op == 0:
                t = msg.get("t")
                d = d or {}
                if t == "READY":
                    self._resume[key] = {"url": d.get("resume_gateway_url") or "", "session_id": d.get("session_id"),
                                         "seq": st["seq"], "bot_id": (d.get("user") or {}).get("id")}
                    if not self._resume[key]["url"]:
                        self._resume.pop(key, None)
                    ctx.connected()
                elif t == "RESUMED":
                    ctx.connected()
                elif t == "GUILD_CREATE":
                    for ch in (d.get("channels") or []) + (d.get("threads") or []):
                        if ch.get("id") and ch.get("name"):
                            self._channels.put(ch["id"], ch["name"])
                elif t in ("CHANNEL_CREATE", "CHANNEL_UPDATE", "THREAD_CREATE", "THREAD_UPDATE"):
                    if d.get("id") and d.get("name"):
                        self._channels.put(d["id"], d["name"])
                elif t == "MESSAGE_CREATE":
                    bot_id = (self._resume.get(key) or {}).get("bot_id")
                    item = await self._item(client, token, bot_id, d)
                    if item is not None and seen.add(item.id):
                        await ctx.emit([item])
        return None

    # -- one message → one item -------------------------------------------------

    async def _channel_name(self, client, token, cid: str) -> str | None:
        async def _load():
            status, ch = await self._rest(client, token, f"/channels/{cid}")
            return ch.get("name") if status == 200 else None
        return await self._channels.get(cid, _load)

    async def _item(self, client, token, bot_id, d: dict) -> Item | None:
        author = d.get("author") or {}
        if not bot_id or author.get("id") == bot_id or author.get("bot"):
            return None
        if d.get("type", 0) not in _USER_TYPES:
            return None
        guild_id = d.get("guild_id")
        if guild_id:
            if not any((m or {}).get("id") == bot_id for m in d.get("mentions") or []):
                return None
            name = await self._channel_name(client, token, d.get("channel_id") or "")
            kind, place = "at", {"kind": "channel", "name": f"#{name}" if name else None}
        else:
            kind, place = "dm", {"kind": "dm"}
        names = {}
        for m in d.get("mentions") or []:
            m = m or {}
            names[m.get("id")] = ((m.get("member") or {}).get("nick") or m.get("global_name")
                                  or m.get("username"))
        text = d.get("content") or ""
        text = _MENTION_RE.sub(lambda m: "" if m.group(1) == bot_id else f"@{names.get(m.group(1)) or 'user'}", text)
        text = _ROLE_RE.sub("@role", text)
        text = _CHAN_RE.sub(lambda m: f"#{self._channels.peek(m.group(1)) or 'channel'}", text)
        text = _EMOJI_RE.sub(lambda m: f":{m.group(1)}:", text)
        text = re.sub(r"[ \t]{2,}", " ", text).strip()
        marks = [placeholder("image") if str(a.get("content_type") or "").startswith("image/")
                 else placeholder("file") for a in d.get("attachments") or [] if isinstance(a, dict)]
        marks += [placeholder("sticker") for _ in d.get("sticker_items") or []]
        if marks:
            text = f"{' '.join(marks)} {text}".strip()
        title, body = split_text(text, placeholder("message"))
        sender = ((d.get("member") or {}).get("nick") or author.get("global_name")
                  or author.get("username"))
        at = d.get("timestamp")
        if isinstance(at, str) and at.endswith("+00:00"):
            at = at[:19] + "Z"
        return Item(id=str(d.get("id")), kind=kind, title=title, body=body, sender=sender, place=place,
                    url=f"https://discord.com/channels/{guild_id or '@me'}/{d.get('channel_id')}/{d.get('id')}",
                    at=at if isinstance(at, str) else None)


ADAPTER = DiscordAdapter
