# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Discord bot source against a local REST + Gateway fake.

Frame shapes, all from the official docs repository
(github.com/discord/discord-api-docs, developers/events/gateway.mdx,
gateway-events.mdx, topics/opcodes-and-status-codes.mdx, read 2026-11-16):
Hello op 10 ``{"heartbeat_interval"}``, Heartbeat op 1 ``d = last seq``,
ACK op 11, Identify op 2 ``{token, intents, properties}``, Resume op 6
``{token, session_id, seq}``, Dispatch op 0 ``{t, s, d}`` with READY
(``session_id``, ``resume_gateway_url``, ``user``), RESUMED, GUILD_CREATE
(``channels``), MESSAGE_CREATE (message object: ``author``, ``member``,
``mentions``, ``guild_id``, ``content``, ``attachments``, ``type``),
Reconnect op 7, Invalid Session op 9 ``d = resumable``, close codes
4004/4007/4009/4014. REST ``GET /users/@me`` 401 body ``{"message":
"401: Unauthorized", "code": 0}`` observed live on 2026-11-16 with a
made-up token.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from agent_os.sources.adapters import discord as dc
from agent_os.sources.model import SourceFailure, VerifyError
from tests.unit._sources_bot_fakes import FakeHTTP, FakeWS, Run, verify_ctx

TOKEN = "MTI4FAKEFAKEFAKE.GxYz12.fakeDiscordTokenForTests0123456789"
BOT_ID = "1280000000000000001"


class GatewayFake:
    def __init__(self):
        self.http = FakeHTTP()
        self.ws = FakeWS(self._conn)
        self.received: list[dict] = []
        self.ack = True
        self.hb_ms = 40
        self.outbox: asyncio.Queue | None = None
        self.seq = 0
        self.replay: list[dict] = []  # dispatched after a successful resume
        self.resume_ok = True
        self.http.route("GET", "/api/v10/users/@me", self._me)
        self.http.route("GET", "/api/v10/gateway/bot", self._gw)
        self.http.route("GET", "/api/v10/channels/*", self._chan)

    async def start(self):
        self.http.start()
        await self.ws.start()
        self.outbox = asyncio.Queue()
        return self

    async def stop(self):
        self.http.stop()
        await self.ws.stop()

    def _auth(self, req):
        return req.headers.get("authorization") == f"Bot {TOKEN}"

    def _me(self, req):
        if not self._auth(req):
            return 401, {"message": "401: Unauthorized", "code": 0}
        return 200, {"id": BOT_ID, "username": "orbital-bot", "bot": True}

    def _gw(self, req):
        if not self._auth(req):
            return 401, {"message": "401: Unauthorized", "code": 0}
        return 200, {"url": self.ws.url, "shards": 1, "session_start_limit": {"remaining": 999}}

    def _chan(self, req):
        cid = req.path.rsplit("/", 1)[1]
        if cid == "77":
            return 200, {"id": "77", "name": "random", "type": 0}
        return 403, {"message": "Missing Access", "code": 50001}

    def dispatch(self, t, d):
        self.outbox.put_nowait(("D", t, d))

    def control(self, frame):
        self.outbox.put_nowait(("RAW", frame))

    def close(self, code):
        self.outbox.put_nowait(("CLOSE", code))

    async def _conn(self, ws):
        async def send_dispatch(t, d):
            self.seq += 1
            await ws.send(json.dumps({"op": 0, "t": t, "s": self.seq, "d": d}))

        await ws.send(json.dumps({"op": 10, "d": {"heartbeat_interval": self.hb_ms}}))

        async def reader():
            async for raw in ws:
                m = json.loads(raw)
                self.received.append(m)
                if m["op"] == 1 and self.ack:
                    await ws.send(json.dumps({"op": 11}))
                elif m["op"] == 2:
                    await send_dispatch("READY", {"v": 10, "session_id": "sess-1",
                                                  "resume_gateway_url": self.ws.url + "/resume",
                                                  "user": {"id": BOT_ID, "username": "orbital-bot", "bot": True}})
                    await send_dispatch("GUILD_CREATE", {"id": "9", "name": "Acme Dev",
                                                         "channels": [{"id": "55", "name": "support", "type": 0}]})
                elif m["op"] == 6:
                    if not self.resume_ok:
                        await ws.send(json.dumps({"op": 9, "d": False}))
                        continue
                    for ev in self.replay:
                        await send_dispatch("MESSAGE_CREATE", ev)
                    self.replay = []
                    await send_dispatch("RESUMED", {})
        rt = asyncio.create_task(reader())
        try:
            while True:
                get = asyncio.ensure_future(self.outbox.get())
                await asyncio.wait([get, rt], return_when=asyncio.FIRST_COMPLETED)
                if not get.done():
                    get.cancel()
                    return
                what = get.result()
                if what[0] == "D":
                    await send_dispatch(what[1], what[2])
                elif what[0] == "RAW":
                    await ws.send(json.dumps(what[1]))
                elif what[0] == "CLOSE":
                    await ws.close(what[1])
                    return
        finally:
            rt.cancel()

    def ops(self, op):
        return [m for m in self.received if m["op"] == op]


def message(mid, content, *, guild=True, author_id="501", mentions=(), **extra):
    m = {"id": mid, "channel_id": "55" if guild else "800", "type": 0, "content": content,
         "timestamp": "2026-11-16T08:00:00.000000+00:00",
         "author": {"id": author_id, "username": "jordan", "global_name": "Jordan"},
         "mentions": [{"id": i, "username": n} for i, n in mentions], "attachments": [], **extra}
    if guild:
        m["guild_id"] = "9"
        m["member"] = {"nick": "Jordan (support)"}
    return m


@pytest.fixture
async def fake():
    f = await GatewayFake().start()
    yield f
    await f.stop()


@pytest.fixture
def adapter(fake, monkeypatch):
    a = dc.DiscordAdapter()
    a.api_base = fake.http.url + "/api/v10"
    monkeypatch.setattr(dc.random, "uniform", lambda a_, b_: 0.0)
    return a


SECRETS = {"token": TOKEN}


# ── verify ─────────────────────────────────────────────────────────────────

async def test_verify_labels_the_bot(adapter):
    assert (await adapter.verify(verify_ctx(SECRETS))).label == "orbital-bot"


async def test_verify_rejects_a_bad_token_on_its_field(adapter):
    with pytest.raises(VerifyError) as e:
        await adapter.verify(verify_ctx({"token": "MTI4.wrong.token"}))
    assert (e.value.field, e.value.reason) == ("token", "rejected")


async def test_verify_rejects_a_user_token(adapter, fake):
    fake.http.route("GET", "/api/v10/users/@me", lambda r: (200, {"id": "1", "username": "me"}))
    with pytest.raises(VerifyError) as e:
        await adapter.verify(verify_ctx(SECRETS))
    assert e.value.reason == "invalid"


# ── identify, arrivals ─────────────────────────────────────────────────────

async def test_identify_asks_for_no_privileged_intent(adapter, fake):
    run = Run(adapter, secrets=SECRETS).start()
    await run.until(lambda: run.connects == 1)
    ident = fake.ops(2)[0]["d"]
    assert ident["token"] == TOKEN and ident["intents"] == 4609  # GUILDS|GUILD_MESSAGES|DIRECT_MESSAGES
    assert not ident["intents"] & (1 << 15)
    await run.cancel()


async def test_dm_and_guild_mention_become_items(adapter, fake):
    run = Run(adapter, secrets=SECRETS).start()
    await run.until(lambda: run.connects == 1)
    fake.dispatch("MESSAGE_CREATE", message("m1", "the build is red again", guild=False))
    fake.dispatch("MESSAGE_CREATE", message(
        "m2", f"<@{BOT_ID}> customer says exports time out, cc <@502> in <#55> <:fire:123>",
        mentions=[(BOT_ID, "orbital-bot"), ("502", "kai")]))
    await run.until(lambda: len(run.items) == 2)
    a, b = run.items
    assert (a.kind, a.sender, a.place, a.id) == ("dm", "Jordan", {"kind": "dm"}, "m1")
    assert a.url == "https://discord.com/channels/@me/800/m1" and a.at == "2026-11-16T08:00:00Z"
    assert (b.kind, b.sender, b.place) == ("at", "Jordan (support)", {"kind": "channel", "name": "#support"})
    assert b.title == "customer says exports time out, cc @kai in #support :fire:"
    await run.cancel()


async def test_unmentioned_own_other_bot_and_system_messages_are_ignored(adapter, fake):
    run = Run(adapter, secrets=SECRETS).start()
    await run.until(lambda: run.connects == 1)
    fake.dispatch("MESSAGE_CREATE", message("x1", "just chatting"))
    fake.dispatch("MESSAGE_CREATE", message("x2", "mine", guild=False, author_id=BOT_ID))
    fake.dispatch("MESSAGE_CREATE", {**message("x3", "beep", guild=False), "author": {"id": "9", "username": "b", "bot": True}})
    fake.dispatch("MESSAGE_CREATE", {**message("x4", "", mentions=[(BOT_ID, "o")]), "type": 7})  # member join
    fake.dispatch("MESSAGE_UPDATE", message("x5", f"<@{BOT_ID}> edited", mentions=[(BOT_ID, "o")]))
    fake.dispatch("MESSAGE_CREATE", message("x6", "real", guild=False))
    await run.until(lambda: len(run.items) == 1)
    await asyncio.sleep(0.15)
    assert [i.id for i in run.items] == ["x6"]
    await run.cancel()


async def test_attachments_unknown_channels_and_duplicate_delivery(adapter, fake):
    run = Run(adapter, secrets=SECRETS).start()
    await run.until(lambda: run.connects == 1)
    att = message("a1", "", guild=False, attachments=[{"content_type": "image/png"}, {"content_type": "text/plain"}])
    fake.dispatch("MESSAGE_CREATE", att)
    fake.dispatch("MESSAGE_CREATE", att)  # delivered twice
    fake.dispatch("MESSAGE_CREATE", {**message("a2", f"<@{BOT_ID}> hi", mentions=[(BOT_ID, "o")]), "channel_id": "77"})
    fake.dispatch("MESSAGE_CREATE", {**message("a3", f"<@{BOT_ID}> hi", mentions=[(BOT_ID, "o")]), "channel_id": "78"})
    await run.until(lambda: len(run.items) == 3)
    await asyncio.sleep(0.1)
    assert [i.id for i in run.items] == ["a1", "a2", "a3"]
    assert run.items[0].title == "[image] [file]"
    assert run.items[1].place["name"] == "#random"  # one REST lookup
    assert run.items[2].place == {"kind": "channel", "name": None}  # lookup refused
    await run.cancel()


# ── heartbeats, resume, invalid session, close codes ───────────────────────

async def test_heartbeats_carry_the_sequence(adapter, fake):
    run = Run(adapter, secrets=SECRETS).start()
    await run.until(lambda: run.connects == 1)
    await run.until(lambda: len(fake.ops(1)) >= 3)
    assert fake.ops(1)[-1]["d"] == fake.seq
    await run.cancel()


async def test_a_missing_ack_drops_the_connection_and_the_next_one_resumes(adapter, fake):
    run = Run(adapter, secrets=SECRETS).start()
    await run.until(lambda: run.connects == 1)
    fake.dispatch("MESSAGE_CREATE", message("r1", "before", guild=False))
    await run.until(lambda: len(run.items) == 1)
    fake.ack = False
    err = await run.result()
    assert isinstance(err, ConnectionError)  # zombie connection → the core reconnects
    fake.ack = True
    seq_before = fake.seq
    fake.replay = [message("r2", "missed while down", guild=False)]
    run2 = Run(adapter, secrets=SECRETS).start()
    await run2.until(lambda: run2.connects == 1)
    resume = fake.ops(6)[-1]["d"]
    assert resume == {"token": TOKEN, "session_id": "sess-1", "seq": seq_before}
    assert fake.ws.paths[-1].startswith("/resume")  # resume_gateway_url
    assert [i.id for i in run2.items] == ["r2"]  # replayed once, r1 not again
    await run2.cancel()


async def test_reconnect_op7_resumes(adapter, fake):
    run = Run(adapter, secrets=SECRETS).start()
    await run.until(lambda: run.connects == 1)
    fake.control({"op": 7, "d": None})
    assert await run.result() is None
    run2 = Run(adapter, secrets=SECRETS).start()
    await run2.until(lambda: run2.connects == 1)
    assert len(fake.ops(6)) == 1 and len(fake.ops(2)) == 1
    await run2.cancel()


async def test_invalid_session_identifies_again(adapter, fake):
    run = Run(adapter, secrets=SECRETS).start()
    await run.until(lambda: run.connects == 1)
    fake.close(4000)  # an unknown drop: resumable
    assert isinstance(await run.result(), ConnectionError)
    fake.resume_ok = False  # the server answers the resume with op 9 d=false
    run2 = Run(adapter, secrets=SECRETS).start()
    assert await run2.result() is None
    run3 = Run(adapter, secrets=SECRETS).start()
    await run3.until(lambda: run3.connects == 1)
    assert len(fake.ops(2)) == 2  # fresh identify
    await run3.cancel()


@pytest.mark.parametrize("code,expect", [(4004, "token_rejected"), (4014, "error"), (4013, "error")])
async def test_non_recoverable_close_codes_stop_the_source(adapter, fake, code, expect):
    run = Run(adapter, secrets=SECRETS).start()
    await run.until(lambda: run.connects == 1)
    fake.close(code)
    err = await run.result()
    assert isinstance(err, SourceFailure) and err.code == expect and not err.retryable


async def test_session_timeout_close_starts_fresh(adapter, fake):
    run = Run(adapter, secrets=SECRETS).start()
    await run.until(lambda: run.connects == 1)
    fake.close(4009)
    assert await run.result() is None
    run2 = Run(adapter, secrets=SECRETS).start()
    await run2.until(lambda: run2.connects == 1)
    assert len(fake.ops(2)) == 2 and not fake.ops(6)
    await run2.cancel()


async def test_a_revoked_token_before_connecting_is_non_retryable(adapter, fake):
    err = await Run(adapter, secrets={"token": "MTI4.revoked.x"}).start().result()
    assert isinstance(err, SourceFailure) and err.code == "token_rejected" and not err.retryable


async def test_the_token_never_reaches_a_log_line(adapter, fake, caplog):
    caplog.set_level(logging.DEBUG)
    run = Run(adapter, secrets=SECRETS).start()
    await run.until(lambda: run.connects == 1)
    await run.cancel()
    ours = "\n".join(r.getMessage() for r in caplog.records if not r.name.startswith("websockets.server"))
    assert '"op": 2' in ours  # websockets logged the identify frame…
    assert TOKEN not in ours and TOKEN[:12] not in ours  # …without the token
