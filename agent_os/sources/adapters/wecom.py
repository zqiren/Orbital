# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""WeCom (企业微信) smart robot, by its API-mode long connection — outbound,
no public address, no SDK.

Protocol, read 2026-11-16 from the WeCom team's own Node SDK
(github.com/WecomTeam/aibot-node-sdk: src/ws.ts, src/types/*.ts) and its
community Python port wecom-aibot-sdk 1.0.8 (ws.py, types.py,
message_handler.py); no official prose document was reachable:

* ``wss://openws.work.weixin.qq.com``; JSON frames ``{"cmd", "headers":
  {"req_id"}, "body"}``; answers carry the request's ``req_id`` plus
  ``errcode``/``errmsg`` and no ``cmd``.
* ``aibot_subscribe`` with ``{"bot_id", "secret"}`` authenticates
  (``errcode`` 0; 853000 "invalid bot_id or secret" — observed live with a
  made-up pair on 2026-11-16).
* ``ping`` every 30 s; an answer resets the missed count; two missed → dead.
* pushes: ``aibot_msg_callback`` (body ``msgid``, ``aibotid``, ``chatid``
  (groups only), ``chattype`` single|group, ``from.userid``, ``msgtype``
  text|image|mixed|voice|file|video, ``text.content``…) and
  ``aibot_event_callback`` (``event.eventtype``; ``disconnected_event`` =
  another connection for this bot was opened, this one is about to close).
  Message callbacks need no acknowledgement (the SDKs send none unless they
  reply, and Orbital never replies).

A smart robot in a group only receives messages that @ it, so a group
message is an ``at`` arrival. The callback carries user ids only and the
robot has no API to resolve names, so sender = the userid and the group has
no name. Lossy: the core records gaps.
"""

from __future__ import annotations

import asyncio
import json
import re
import secrets as _rand
import time

from agent_os.sources.adapters import Adapter
from agent_os.sources.adapters._bot_common import (
    SeenIds, Unreachable, endpoint, iso_from_epoch, placeholder, split_text, ws_connect,
)
from agent_os.sources.model import Item, SourceFailure, Verified, VerifyError

WS_URL = "wss://openws.work.weixin.qq.com"
_REJECTED = {853000}
SUBSCRIBE_TIMEOUT_S = 15.0
OWN_VERIFY_WINDOW_S = 60.0


def _req_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}_{_rand.token_hex(4)}"


class WeComAdapter(Adapter):
    mode = "stream"
    lossless = False
    ws_url: str | None = None
    heartbeat_s: float = 30.0
    max_missed: int = 2

    def __init__(self):
        # bot id -> monotonic time of our last verify: WeCom allows one
        # connection per bot, so a verify (「再检查」) closes the running
        # stream; that disconnect is ours, not another program's.
        self._verified_at: dict = {}

    def _url(self) -> str:
        return self.ws_url or endpoint("wecom", WS_URL)

    async def _subscribe(self, ws, bid: str, bsec: str) -> tuple[int, str]:
        rid = _req_id("aibot_subscribe")
        await ws.send(json.dumps({"cmd": "aibot_subscribe", "headers": {"req_id": rid},
                                  "body": {"bot_id": bid, "secret": bsec}}))

        async def _answer():
            while True:
                f = json.loads(await ws.recv())
                if isinstance(f, dict) and (f.get("headers") or {}).get("req_id") == rid:
                    return int(f.get("errcode", -1)), str(f.get("errmsg") or "")
        return await asyncio.wait_for(_answer(), SUBSCRIBE_TIMEOUT_S)

    @staticmethod
    def _creds(ctx) -> tuple[str, str]:
        return (ctx.config.get("bid") or "").strip(), (ctx.secret("bsec") or "").strip()

    async def verify(self, ctx) -> Verified:
        import websockets
        bid, bsec = self._creds(ctx)
        if not bid:
            raise VerifyError("invalid", "bid", "enter the Bot ID")
        if not bsec:
            raise VerifyError("invalid", "bsec", "enter the Secret")
        self._verified_at[bid] = time.monotonic()
        try:
            async with ws_connect(self._url(), ping_interval=None) as ws:
                code, _msg = await self._subscribe(ws, bid, bsec)
        except (OSError, websockets.InvalidHandshake, websockets.ConnectionClosed,
                asyncio.TimeoutError, ValueError):
            raise VerifyError("unreachable", None, "cannot reach WeCom") from None
        if code == 0:
            return Verified(label=bid)
        if code in _REJECTED:
            raise VerifyError("rejected", "bsec", "WeCom refused this Bot ID / Secret")
        raise VerifyError("error", None, f"WeCom answered errcode {code}")

    async def stream(self, ctx) -> None:
        import websockets
        bid, bsec = self._creds(ctx)
        seen = SeenIds()
        hb = {"missed": 0, "dead": False}
        try:
            async with ws_connect(self._url(), ping_interval=None) as ws:
                code, _msg = await self._subscribe(ws, bid, bsec)
                if code in _REJECTED:
                    raise SourceFailure("token_rejected", {"field": "bsec"}, retryable=False)
                if code != 0:
                    raise Unreachable(f"wecom subscribe answered errcode {code}")
                ctx.connected()
                beat = asyncio.create_task(self._heartbeat(ws, hb))
                try:
                    async for raw in ws:
                        try:
                            f = json.loads(raw)
                        except (TypeError, ValueError):
                            continue
                        if not isinstance(f, dict):
                            continue
                        cmd = f.get("cmd")
                        if not cmd:
                            if str((f.get("headers") or {}).get("req_id") or "").startswith("ping"):
                                hb["missed"] = 0
                            continue
                        if cmd == "aibot_msg_callback":
                            item = self.item(f.get("body") or {})
                            if item is not None and seen.add(item.id):
                                await ctx.emit([item])
                        elif cmd == "aibot_event_callback":
                            ev = ((f.get("body") or {}).get("event") or {})
                            if ev.get("eventtype") == "disconnected_event":
                                at = self._verified_at.get(bid)
                                if at is not None and time.monotonic() - at < OWN_VERIFY_WINDOW_S:
                                    return  # our own check took the connection
                                raise SourceFailure("bot_in_use", {})
                finally:
                    beat.cancel()
        except websockets.ConnectionClosed:
            raise Unreachable("wecom connection closed") from None
        except Unreachable:
            raise
        except (OSError, websockets.InvalidHandshake, asyncio.TimeoutError):
            raise Unreachable("cannot open the wecom connection") from None
        if hb["dead"]:
            raise Unreachable("wecom heartbeat was not answered")
        raise Unreachable("wecom connection closed")

    async def _heartbeat(self, ws, hb: dict) -> None:
        try:
            while True:
                await asyncio.sleep(self.heartbeat_s)
                if hb["missed"] >= self.max_missed:
                    hb["dead"] = True
                    await ws.close()
                    return
                hb["missed"] += 1
                await ws.send(json.dumps({"cmd": "ping", "headers": {"req_id": _req_id("ping")}}))
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    @staticmethod
    def item(b: dict) -> Item | None:
        if not isinstance(b, dict) or not b.get("msgid"):
            return None
        chattype = b.get("chattype")
        if chattype == "single":
            kind, place = "dm", {"kind": "dm"}
        elif chattype == "group":
            kind, place = "at", {"kind": "group", "name": None}
        else:
            return None
        mtype = b.get("msgtype")
        if mtype == "text":
            text = str((b.get("text") or {}).get("content") or "")
        elif mtype == "voice":
            text = f"{placeholder('voice')} {(b.get('voice') or {}).get('content') or ''}"
        elif mtype == "mixed":
            bits = []
            for it in ((b.get("mixed") or {}).get("msg_item") or []):
                if not isinstance(it, dict):
                    continue
                if it.get("msgtype") == "text":
                    bits.append(str((it.get("text") or {}).get("content") or ""))
                else:
                    bits.append(placeholder(it.get("msgtype") or "message"))
            text = " ".join(bits)
        else:
            text = placeholder(mtype or "message")
        text = re.sub(r"[ \t]{2,}", " ", text).strip()
        if kind == "at":
            # The text keeps the leading "@<robot name>"; the robot's name is
            # not in the callback, so drop the first @word.
            text = re.sub(r"^@\S+\s*", "", text, count=1).strip() or text
        title, body = split_text(text, placeholder(mtype or "message"))
        ct = b.get("create_time")
        at = iso_from_epoch(ct, millis=isinstance(ct, (int, float)) and ct > 1e12)
        return Item(id=str(b["msgid"]), kind=kind, title=title, body=body,
                    sender=((b.get("from") or {}).get("userid")) or None, place=place, url=None, at=at)


ADAPTER = WeComAdapter
