# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Feishu (and Lark) bot, by the long connection — outbound, no public
address, no SDK.

Long connection (ported from the official MIT SDK ``lark-oapi``,
``lark_oapi/ws/client.py`` + ``pb/pbbp2.proto``, read 2026-11-16):
``POST /callback/ws/endpoint {"AppID", "AppSecret"}`` returns ``data.URL``
(a ``wss://`` address whose query carries ``service_id``) and
``ClientConfig.PingInterval`` (seconds). Frames are protobuf ``pbbp2.Frame``
(SeqID 1, LogID 2, service 3, method 4, repeated Header{key 1, value 2} 5,
payload_encoding 6, payload_type 7, payload 8, LogIDNew 9). ``method`` 0 =
control (``type`` header ``ping``/``pong``; a pong may carry a new
ClientConfig), 1 = data (``type`` ``event``/``card``; ``message_id``,
``sum``/``seq`` for a payload split over several frames). Each data frame is
acknowledged by sending the same frame back with payload ``{"code": 200}``
and a ``biz_rt`` header. The codec is ~60 lines below; it needs no library.

Events (https://open.feishu.cn/document/server-docs/im-v1/message/events/receive):
``im.message.receive_v1``: ``event.message.chat_type`` ``p2p`` (``dm``) or
``group`` with the bot among ``mentions`` (``at``). Stable id =
``message_id``. Names: ``GET /open-apis/im/v1/chats/{chat_id}`` (group name)
and ``GET /open-apis/contact/v3/users/{open_id}`` (sender), one cached
lookup each; without those permissions the item carries no name.

QR connect: the scan-to-create app registration nanobot uses
(``accounts.feishu.cn/oauth/v1/app/registration``, a device-code flow:
``init`` → ``begin`` gives ``device_code`` + ``verification_uri_complete``
(the QR) → ``poll`` answers ``authorization_pending`` until the user approves,
then ``client_id`` + ``client_secret``). The flow has no separate "scanned"
answer, so the states are pending → done | expired | failed.

Lossy: Feishu does not replay events missed while no connection was open
(not confirmed by docs; nanobot treats it so). The core records gaps.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse, urlsplit, urlunsplit

from agent_os.sources.adapters import Adapter
from agent_os.sources.adapters._bot_common import (
    NameCache, SeenIds, Unreachable, endpoint, host_of, http_client, iso_from_epoch, placeholder,
    split_text, ws_connect, ws_invalid_status, ws_status_headers,
)
from agent_os.sources.model import Item, SourceFailure, Verified, VerifyError

OPEN = {"feishu": "https://open.feishu.cn", "lark": "https://open.larksuite.com"}
ACCOUNTS = {"feishu": "https://accounts.feishu.cn", "lark": "https://accounts.larksuite.com"}
REGISTRATION_PATH = "/oauth/v1/app/registration"
_BAD_CREDENTIALS = {10003, 10014, 10013}  # tenant_access_token: invalid param / app secret invalid
_WS_BAD_APP = {1000040344, 1000040345, 1000040346, 514, 403}
_WS_CONN_LIMIT = 1000040350

#: The two read permissions a bot needs to receive messages.


# ── pbbp2.Frame codec (proto2; see the module docstring) ─────────────────────

def _uvarint(n: int) -> bytes:
    n &= 0xFFFFFFFFFFFFFFFF
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _read_uvarint(buf: bytes, i: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        if i >= len(buf):
            raise ValueError("truncated varint")
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _ld(field: int, data: bytes) -> bytes:
    return _uvarint((field << 3) | 2) + _uvarint(len(data)) + data


def _vi(field: int, value: int) -> bytes:
    return _uvarint(field << 3) + _uvarint(value)


def encode_frame(f: dict) -> bytes:
    out = (_vi(1, int(f.get("SeqID") or 0)) + _vi(2, int(f.get("LogID") or 0))
           + _vi(3, int(f.get("service") or 0)) + _vi(4, int(f.get("method") or 0)))
    for k, v in f.get("headers") or []:
        out += _ld(5, _ld(1, str(k).encode()) + _ld(2, str(v).encode()))
    for num, key in ((6, "payload_encoding"), (7, "payload_type")):
        if f.get(key):
            out += _ld(num, str(f[key]).encode())
    if f.get("payload") is not None:
        out += _ld(8, bytes(f["payload"]))
    if f.get("LogIDNew"):
        out += _ld(9, str(f["LogIDNew"]).encode())
    return out


def _fields(buf: bytes):
    i = 0
    while i < len(buf):
        key, i = _read_uvarint(buf, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            val, i = _read_uvarint(buf, i)
        elif wt == 2:
            n, i = _read_uvarint(buf, i)
            if i + n > len(buf):
                raise ValueError("truncated field")
            val, i = buf[i:i + n], i + n
        elif wt == 1:
            val, i = buf[i:i + 8], i + 8
        elif wt == 5:
            val, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"wire type {wt}")
        yield num, wt, val


def decode_frame(buf: bytes) -> dict:
    f = {"SeqID": 0, "LogID": 0, "service": 0, "method": 0, "headers": [], "payload": b""}
    names = {1: "SeqID", 2: "LogID", 3: "service", 4: "method"}
    for num, wt, val in _fields(buf):
        if num in names and wt == 0:
            f[names[num]] = val - (1 << 64) if num in (3, 4) and val >= (1 << 63) else val
        elif num == 5 and wt == 2:
            k = v = ""
            for hn, hwt, hv in _fields(val):
                if hn == 1 and hwt == 2:
                    k = hv.decode("utf-8", "replace")
                elif hn == 2 and hwt == 2:
                    v = hv.decode("utf-8", "replace")
            f["headers"].append((k, v))
        elif num == 6 and wt == 2:
            f["payload_encoding"] = val.decode("utf-8", "replace")
        elif num == 7 and wt == 2:
            f["payload_type"] = val.decode("utf-8", "replace")
        elif num == 8 and wt == 2:
            f["payload"] = bytes(val)
        elif num == 9 and wt == 2:
            f["LogIDNew"] = val.decode("utf-8", "replace")
    return f


def _header(f: dict, key: str) -> str | None:
    for k, v in f.get("headers") or []:
        if k == key:
            return v
    return None


# ── the adapter ────────────────────────────────────────────────────────────

class FeishuAdapter(Adapter):
    mode = "stream"
    lossless = False
    #: tests: one base for every Feishu host (open API, accounts, ws endpoint),
    #: or ``{"feishu": url, "lark": url}``
    api_base = None
    default_ping_s: float = 120.0

    def __init__(self):
        self._names = NameCache()
        self._tokens: dict = {}  # (domain, app_id) -> (token, expires_monotonic)

    def _test_base(self, domain: str) -> str | None:
        if isinstance(self.api_base, dict):
            return self.api_base.get(domain)
        return self.api_base

    def _open(self, domain: str) -> str:
        return self._test_base(domain) or endpoint("feishu", OPEN.get(domain, OPEN["feishu"]))

    def _accounts(self, domain: str) -> str:
        return self._test_base(domain) or endpoint("feishu", ACCOUNTS.get(domain, ACCOUNTS["feishu"]))

    async def _json(self, client, method: str, url: str, **kw) -> tuple[int, dict]:
        import httpx
        try:
            r = await client.request(method, url, **kw)
        except httpx.HTTPError:
            raise Unreachable(host_of(url)) from None
        try:
            data = r.json()
        except ValueError:
            data = {}
        return r.status_code, data if isinstance(data, dict) else {}

    async def _tenant_token(self, client, domain: str, app_id: str, secret: str,
                            *, fresh: bool = False) -> tuple[int, str | None]:
        """(feishu code, token)."""
        key = (domain, app_id)
        hit = self._tokens.get(key)
        if hit and not fresh and hit[1] > time.monotonic():
            return 0, hit[0]
        _s, data = await self._json(client, "POST",
                                    f"{self._open(domain)}/open-apis/auth/v3/tenant_access_token/internal",
                                    json={"app_id": app_id, "app_secret": secret})
        code = data.get("code")
        tok = data.get("tenant_access_token")
        if code == 0 and tok:
            ttl = max(60, int(data.get("expire") or 7200) - 300)
            self._tokens[key] = (tok, time.monotonic() + ttl)
            return 0, tok
        return int(code) if isinstance(code, int) else -1, None

    async def _bot_info(self, client, domain: str, token: str) -> dict:
        _s, data = await self._json(client, "GET", f"{self._open(domain)}/open-apis/bot/v3/info",
                                    headers={"Authorization": f"Bearer {token}"})
        return data.get("bot") if data.get("code") == 0 and isinstance(data.get("bot"), dict) else {}

    # -- connect ------------------------------------------------------------

    async def _check(self, client, app_id: str, secret: str, domain: str | None):
        """(domain, bot) or VerifyError. A fresh connection with no domain
        tries Feishu, then Lark (an app id belongs to one of them)."""
        domains = [domain] if domain in OPEN else ["feishu", "lark"]
        first_code = None
        for d in domains:
            code, tok = await self._tenant_token(client, d, app_id, secret, fresh=True)
            if tok:
                # Bot info gives the label and the bot's open_id. It is not a
                # gate: an app created by the QR flow may answer differently,
                # and the long connection is the real test.
                return d, await self._bot_info(client, d, tok)
            first_code = first_code if first_code is not None else code
        if first_code in _BAD_CREDENTIALS:
            raise VerifyError("rejected", "appsec" if first_code == 10014 else "appid",
                              "Feishu refused this App ID / App Secret")
        raise VerifyError("error", None, f"Feishu answered code {first_code}")

    async def verify(self, ctx) -> Verified:
        app_id = (ctx.config.get("appid") or "").strip()
        secret = (ctx.secret("appsec") or "").strip()
        if not app_id:
            raise VerifyError("invalid", "appid", "enter the App ID")
        if not secret:
            raise VerifyError("invalid", "appsec", "enter the App Secret")
        try:
            async with http_client() as client:
                domain, bot = await self._check(client, app_id, secret, ctx.config.get("domain"))
        except Unreachable as e:
            raise VerifyError("unreachable", None, f"cannot reach {e}") from None
        return Verified(label=bot.get("app_name") or None,
                        config={"domain": domain} if domain != "feishu" else {})

    # -- QR: scan to create the app ---------------------------------------------

    async def qr_start(self, ctx) -> dict:
        domain = "feishu"
        base = self._accounts(domain) + REGISTRATION_PATH
        async with http_client() as client:
            _s, init = await self._json(client, "POST", base, data={"action": "init"})
            if "client_secret" not in (init.get("supported_auth_methods") or []):
                raise Unreachable("feishu app registration is not available")
            _s, begin = await self._json(client, "POST", base, data={
                "action": "begin", "archetype": "PersonalAgent", "auth_method": "client_secret",
                "request_user_info": "open_id"})
        code, url = begin.get("device_code"), begin.get("verification_uri_complete")
        if not code or not url:
            raise Unreachable("feishu app registration did not start")
        p = urlsplit(url)
        if p.scheme == "https" and p.netloc in ("open.feishu.cn", "open.larksuite.com") and p.path == "/page/launcher":
            url = urlunsplit(p._replace(path="/page/cli"))  # what nanobot shows for PersonalAgent codes
        expires_in = int(begin.get("expires_in") or 600)
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        return {"handle": {"device_code": code, "domain": domain, "deadline": time.time() + expires_in},
                "qr_text": url, "expires_at": expires_at.strftime("%Y-%m-%dT%H:%M:%SZ")}

    async def qr_poll(self, ctx, handle) -> dict:
        if not isinstance(handle, dict) or time.time() >= float(handle.get("deadline") or 0):
            return {"state": "expired"}
        domain = handle.get("domain") or "feishu"
        async with http_client() as client:
            _s, res = await self._json(client, "POST", self._accounts(domain) + REGISTRATION_PATH, data={
                "action": "poll", "device_code": handle["device_code"], "tp": "ob_app"})
            if ((res.get("user_info") or {}).get("tenant_brand")) == "lark":
                domain = handle["domain"] = "lark"
            cid, csec = res.get("client_id"), res.get("client_secret")
            if cid and csec:
                label = None
                try:
                    _code, tok = await self._tenant_token(client, domain, cid, csec, fresh=True)
                    if tok:
                        label = (await self._bot_info(client, domain, tok)).get("app_name")
                except Unreachable:
                    pass
                cfg = {"appid": cid}
                return {"state": "done", "label": label, "config": cfg, "secrets": {"appsec": csec}}
        err = res.get("error")
        if err == "expired_token":
            return {"state": "expired"}
        if err in ("access_denied", "invalid_grant"):
            return {"state": "failed"}
        return {"state": "pending"}

    # -- the long connection ------------------------------------------------------

    async def stream(self, ctx) -> None:
        import websockets
        app_id = (ctx.config.get("appid") or "").strip()
        secret = (ctx.secret("appsec") or "").strip()
        domain = ctx.config.get("domain") if ctx.config.get("domain") in OPEN else "feishu"
        async with http_client() as client:
            code, tok = await self._tenant_token(client, domain, app_id, secret)
            if not tok:
                if code in _BAD_CREDENTIALS:
                    raise SourceFailure("token_rejected", {"field": "appsec"}, retryable=False)
                raise Unreachable(f"feishu token answered code {code}")
            bot = await self._bot_info(client, domain, tok)
            bot_open_id = bot.get("open_id")
            _s, ep = await self._json(client, "POST", f"{self._open(domain)}/callback/ws/endpoint",
                                      json={"AppID": app_id, "AppSecret": secret}, headers={"locale": "zh"})
            ecode = ep.get("code")
            if ecode != 0:
                if ecode == _WS_CONN_LIMIT:
                    raise SourceFailure("bot_in_use", {})
                if ecode in _WS_BAD_APP:
                    raise SourceFailure("token_rejected", {"field": "appsec"}, retryable=False)
                raise Unreachable(f"feishu ws endpoint answered code {ecode}")
            data = ep.get("data") or {}
            url = data.get("URL") or ""
            ping_s = float(((data.get("ClientConfig") or {}).get("PingInterval")) or self.default_ping_s)
            try:
                service_id = int(parse_qs(urlparse(url).query).get("service_id", ["0"])[0])
            except ValueError:
                service_id = 0
            state = {"ping": ping_s}
            seen = SeenIds()
            parts: dict = {}
            try:
                async with ws_connect(url, ping_interval=None) as ws:
                    ctx.connected()
                    pinger = asyncio.create_task(self._pinger(ws, service_id, state))
                    try:
                        async for raw in ws:
                            if isinstance(raw, str):
                                continue
                            try:
                                f = decode_frame(raw)
                            except ValueError:
                                continue
                            await self._frame(ctx, client, ws, f, parts, seen, state,
                                              domain, app_id, secret, bot_open_id)
                    finally:
                        pinger.cancel()
            except websockets.ConnectionClosed:
                raise Unreachable("feishu long connection closed") from None
            except ws_invalid_status() as e:
                hs = ws_status_headers(e)
                status = str(hs.get("handshake-status") or "")
                if status == "514" and str(hs.get("handshake-autherrcode") or "") == str(_WS_CONN_LIMIT):
                    raise SourceFailure("bot_in_use", {}) from None
                if status in ("403", "514"):
                    raise SourceFailure("token_rejected", {"field": "appsec"}, retryable=False) from None
                raise Unreachable("feishu refused the long connection") from None
            except Unreachable:
                raise
            except (OSError, websockets.InvalidHandshake, asyncio.TimeoutError):
                raise Unreachable("cannot open the feishu long connection") from None
            raise Unreachable("feishu long connection closed")

    async def _pinger(self, ws, service_id: int, state: dict) -> None:
        try:
            while True:
                await asyncio.sleep(state["ping"])
                await ws.send(encode_frame({"service": service_id, "method": 0,
                                            "headers": [("type", "ping")]}))
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def _frame(self, ctx, client, ws, f, parts, seen, state, domain, app_id, secret, bot_open_id):
        ftype = _header(f, "type")
        if f["method"] == 0:
            if ftype == "pong" and f.get("payload"):
                try:
                    conf = json.loads(f["payload"])
                    if conf.get("PingInterval"):
                        state["ping"] = float(conf["PingInterval"])
                except (ValueError, TypeError, AttributeError):
                    pass
            return
        if f["method"] != 1:
            return
        payload = f.get("payload") or b""
        try:
            total = int(_header(f, "sum") or 1)
            seq = int(_header(f, "seq") or 0)
        except ValueError:
            total, seq = 1, 0
        if total > 1:
            mid = _header(f, "message_id") or ""
            buf = parts.setdefault(mid, [None] * total)
            if 0 <= seq < total:
                buf[seq] = payload
            if any(p is None for p in buf):
                return
            payload = b"".join(parts.pop(mid))
        ack = dict(f)
        ack["headers"] = list(f["headers"]) + [("biz_rt", "0")]
        ack["payload"] = b'{"code":200}'
        await ws.send(encode_frame(ack))
        if ftype != "event":
            return
        try:
            ev = json.loads(payload)
        except ValueError:
            return
        item = await self._item(client, domain, app_id, secret, bot_open_id, ev)
        if item is not None and seen.add(item.id):
            await ctx.emit([item])

    # -- one event → one item -----------------------------------------------------

    async def _get_name(self, client, domain, app_id, secret, path: str, pick) -> str | None:
        async def _load():
            _c, tok = await self._tenant_token(client, domain, app_id, secret)
            if not tok:
                return None
            _s, data = await self._json(client, "GET", f"{self._open(domain)}{path}",
                                        headers={"Authorization": f"Bearer {tok}"})
            return pick(data.get("data") or {}) if data.get("code") == 0 else None
        return await self._names.get(path, _load)

    async def _item(self, client, domain, app_id, secret, bot_open_id, ev: dict) -> Item | None:
        header = ev.get("header") or {}
        if header.get("event_type") != "im.message.receive_v1":
            return None
        body = ev.get("event") or {}
        msg = body.get("message") or {}
        sender = body.get("sender") or {}
        if sender.get("sender_type") not in (None, "user"):
            return None
        sender_open_id = (sender.get("sender_id") or {}).get("open_id")
        if bot_open_id and sender_open_id == bot_open_id:
            return None
        mentions = [m for m in msg.get("mentions") or [] if isinstance(m, dict)]
        chat_type = msg.get("chat_type")
        if chat_type == "p2p":
            kind, place = "dm", {"kind": "dm"}
        elif chat_type == "group":
            # Without the "all group messages" permission Feishu only delivers
            # group messages that @ the bot; check anyway when its id is known.
            if bot_open_id and not any((m.get("id") or {}).get("open_id") == bot_open_id for m in mentions):
                return None
            if not bot_open_id and not mentions:
                return None
            chat_id = msg.get("chat_id") or ""
            name = await self._get_name(client, domain, app_id, secret, f"/open-apis/im/v1/chats/{chat_id}",
                                        lambda d: d.get("name"))
            kind, place = "at", {"kind": "group", "name": name}
        else:
            return None
        text = self._text(msg, mentions, bot_open_id)
        title, body_text = split_text(text, placeholder(msg.get("message_type") or "message"))
        who = None
        if sender_open_id:
            who = await self._get_name(client, domain, app_id, secret,
                                       f"/open-apis/contact/v3/users/{sender_open_id}?user_id_type=open_id",
                                       lambda d: (d.get("user") or {}).get("name"))
        return Item(id=str(msg.get("message_id")), kind=kind, title=title, body=body_text, sender=who,
                    place=place, url=None, at=iso_from_epoch(msg.get("create_time"), millis=True))

    @staticmethod
    def _text(msg: dict, mentions: list, bot_open_id) -> str:
        mtype = msg.get("message_type")
        try:
            content = json.loads(msg.get("content") or "{}")
        except ValueError:
            content = {}
        if not isinstance(content, dict):
            content = {}
        if mtype == "text":
            text = str(content.get("text") or "")
        elif mtype == "post":
            post = content if "content" in content else next(
                (v for v in content.values() if isinstance(v, dict)), {})
            lines = [str(post.get("title") or "")] if post.get("title") else []
            for para in post.get("content") or []:
                bits = []
                for el in para or []:
                    if not isinstance(el, dict):
                        continue
                    tag = el.get("tag")
                    if tag in ("text", "a", "md"):
                        bits.append(str(el.get("text") or ""))
                    elif tag == "at":
                        bits.append("" if el.get("user_id") == bot_open_id else f"@{el.get('user_name') or ''}")
                    elif tag in ("img", "media"):
                        bits.append(placeholder("image" if tag == "img" else "video"))
                lines.append("".join(bits))
            text = "\n".join(lines)
        else:
            return placeholder(mtype or "message")
        for m in mentions:
            key = m.get("key")
            if not key:
                continue
            is_bot = (m.get("id") or {}).get("open_id") == bot_open_id
            text = text.replace(key, "" if is_bot else f"@{m.get('name') or ''}")
        return re.sub(r"[ \t]{2,}", " ", text).strip()


ADAPTER = FeishuAdapter
