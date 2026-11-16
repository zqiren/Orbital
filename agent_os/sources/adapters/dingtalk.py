# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""DingTalk bot, by Stream Mode — outbound, no public address, no SDK.

Protocol (official SDK github.com/open-dingtalk/dingtalk-stream-sdk-python,
``stream.py`` / ``frames.py`` / ``chatbot.py``, read 2026-11-16, and
https://open.dingtalk.com/document/direction/stream-mode-protocol-access-description):
``POST https://api.dingtalk.com/v1.0/gateway/connections/open`` with
``clientId``/``clientSecret`` and the subscriptions returns ``endpoint`` +
``ticket``; the client opens ``<endpoint>?ticket=<ticket>``. Frames are JSON
``{"specVersion", "type": "SYSTEM"|"EVENT"|"CALLBACK", "headers":
{"messageId", "topic", "contentType", …}, "data": "<json string>"}``. Every
frame is answered with ``{"code": 200, "headers": {"messageId",
"contentType"}, "message": "OK", "data": …}``; SYSTEM ``ping`` echoes its
data, SYSTEM ``disconnect`` means the server will close (the core reconnects).
An un-acknowledged callback is re-delivered, so acks go out first.

Arrivals: CALLBACK ``/v1.0/im/bot/messages/get``: ``conversationType`` "1"
(single chat, ``dm``) or "2" (group: the bot only receives @ messages,
``at``). The callback carries ``senderNick`` and ``conversationTitle``, so no
name lookup is needed. Stable id = ``msgId``.

Verify = ``POST /v1.0/oauth2/accessToken`` (``appKey``/``appSecret``).
Lossy (no replay documented): the core records gaps.
"""

from __future__ import annotations

import asyncio
import json
import re
from urllib.parse import quote_plus

from agent_os.sources.adapters import Adapter
from agent_os.sources.adapters._bot_common import (
    SeenIds, Unreachable, endpoint, host_of, http_client, iso_from_epoch, placeholder, split_text,
    ws_connect,
)
from agent_os.sources.model import Item, SourceFailure, Verified, VerifyError

API = "https://api.dingtalk.com"
BOT_TOPIC = "/v1.0/im/bot/messages/get"
_REJECTED_CODES = {"invalidClientIdOrSecret", "authFailed", "invalidClientId", "invalidClientSecret",
                   "Forbidden.AccessDenied.AccessTokenPermissionDenied"}


class DingTalkAdapter(Adapter):
    mode = "stream"
    lossless = False
    api_base: str | None = None
    ping_interval: float | None = 60.0  # protocol-level pings, as the SDK's keepalive

    def _base(self) -> str:
        return self.api_base or endpoint("dingtalk", API)

    async def _post(self, client, path: str, body: dict) -> tuple[int, dict]:
        import httpx
        base = self._base()
        try:
            r = await client.post(f"{base}{path}", json=body)
        except httpx.HTTPError:
            raise Unreachable(host_of(base)) from None
        try:
            data = r.json()
        except ValueError:
            data = {}
        return r.status_code, data if isinstance(data, dict) else {}

    async def verify(self, ctx) -> Verified:
        cid = (ctx.config.get("cid") or "").strip()
        csec = (ctx.secret("csec") or "").strip()
        if not cid:
            raise VerifyError("invalid", "cid", "enter the Client ID (AppKey)")
        if not csec:
            raise VerifyError("invalid", "csec", "enter the Client Secret")
        try:
            async with http_client() as client:
                status, data = await self._post(client, "/v1.0/oauth2/accessToken",
                                                {"appKey": cid, "appSecret": csec})
        except Unreachable as e:
            raise VerifyError("unreachable", None, f"cannot reach {e}") from None
        if status == 200 and data.get("accessToken"):
            return Verified(label=cid)
        if data.get("code") in _REJECTED_CODES or status in (400, 401, 403):
            raise VerifyError("rejected", "csec", "DingTalk refused this Client ID / Client Secret")
        raise VerifyError("error", None, f"DingTalk answered HTTP {status}")

    async def stream(self, ctx) -> None:
        import websockets
        cid = (ctx.config.get("cid") or "").strip()
        csec = (ctx.secret("csec") or "").strip()
        async with http_client() as client:
            status, conn = await self._post(client, "/v1.0/gateway/connections/open", {
                "clientId": cid, "clientSecret": csec,
                "subscriptions": [{"type": "CALLBACK", "topic": BOT_TOPIC}],
                "ua": "orbital-sources", "localIp": ""})
        if status != 200 or not conn.get("endpoint") or not conn.get("ticket"):
            if conn.get("code") in _REJECTED_CODES or status in (401, 403):
                raise SourceFailure("token_rejected", {"field": "csec"}, retryable=False)
            if status == 429:
                raise SourceFailure("rate_limited", {})
            raise Unreachable(f"dingtalk connections/open answered HTTP {status}")
        url = f"{conn['endpoint']}?ticket={quote_plus(conn['ticket'])}"
        seen = SeenIds()
        try:
            async with ws_connect(url, ping_interval=self.ping_interval) as ws:
                ctx.connected()
                async for raw in ws:
                    try:
                        frame = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    if not isinstance(frame, dict):
                        continue
                    end = await self._frame(ctx, ws, frame, seen)
                    if end:
                        return
        except websockets.ConnectionClosed:
            raise Unreachable("dingtalk stream closed") from None
        except Unreachable:
            raise
        except (OSError, websockets.InvalidHandshake, asyncio.TimeoutError):
            raise Unreachable("cannot open the dingtalk stream") from None
        raise Unreachable("dingtalk stream closed")

    async def _frame(self, ctx, ws, frame: dict, seen: SeenIds) -> bool:
        ftype = frame.get("type")
        headers = frame.get("headers") or {}
        topic = headers.get("topic")
        ack_headers = {"contentType": "application/json", "messageId": headers.get("messageId")}
        if ftype == "SYSTEM":
            await ws.send(json.dumps({"code": 200, "headers": ack_headers, "message": "OK",
                                      "data": frame.get("data") or ""}))
            return topic == "disconnect"
        if ftype not in ("CALLBACK", "EVENT"):
            return False
        await ws.send(json.dumps({"code": 200, "headers": ack_headers, "message": "OK",
                                  "data": json.dumps({"response": None})}))
        if ftype != "CALLBACK" or topic != BOT_TOPIC:
            return False
        try:
            data = json.loads(frame.get("data") or "{}")
        except (TypeError, ValueError):
            return False
        item = self.item(data)
        if item is not None and seen.add(item.id):
            await ctx.emit([item])
        return False

    @staticmethod
    def item(d: dict) -> Item | None:
        if not isinstance(d, dict) or not d.get("msgId"):
            return None
        if d.get("senderId") and d.get("senderId") == d.get("chatbotUserId"):
            return None
        ctype = str(d.get("conversationType") or "")
        if ctype == "1":
            kind, place = "dm", {"kind": "dm"}
        elif ctype == "2":
            if d.get("isInAtList") is False:
                return None
            kind, place = "at", {"kind": "group", "name": d.get("conversationTitle") or None}
        else:
            return None
        mtype = d.get("msgtype")
        content = d.get("content") if isinstance(d.get("content"), dict) else {}
        if mtype == "text":
            text = str((d.get("text") or {}).get("content") or "")
        elif mtype == "richText":
            bits = []
            for el in content.get("richText") or []:
                if not isinstance(el, dict):
                    continue
                if el.get("text"):
                    bits.append(str(el["text"]))
                elif el.get("pictureDownloadCode") or el.get("downloadCode"):
                    bits.append(placeholder("image"))
            text = " ".join(bits)
        elif mtype == "audio" and content.get("recognition"):
            text = f"{placeholder('voice')} {content['recognition']}"
        else:
            text = placeholder(mtype or "message")
        # DingTalk already leaves the @bot itself out of a group message's
        # text (the SDK and nanobot use it as is).
        text = re.sub(r"[ \t]{2,}", " ", text).strip()
        title, body = split_text(text, placeholder(mtype or "message"))
        return Item(id=str(d["msgId"]), kind=kind, title=title, body=body,
                    sender=d.get("senderNick") or None, place=place, url=None,
                    at=iso_from_epoch(d.get("createAt"), millis=True))


ADAPTER = DingTalkAdapter
