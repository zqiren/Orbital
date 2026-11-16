# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Slack bot source against a local Web API + Socket Mode fake.

Frame shapes:
- ``hello`` / ``disconnect`` (reasons ``warning``, ``refresh_requested``,
  ``link_disabled``) / envelope ``{"envelope_id", "type": "events_api",
  "payload", "accepts_response_payload"}`` / ack ``{"envelope_id"}``:
  official docs https://docs.slack.dev/apis/events-api/using-socket-mode
  (read 2026-11-16).
- ``message`` (``channel_type: "im"``, ``subtype``, ``bot_id``) and
  ``app_mention`` event bodies, ``<@U…>`` / ``<#C…|name>`` / ``<url|label>``
  markup: official Events API docs (docs.slack.dev/reference/events/…),
  from memory of those pages, cross-read against nanobot
  channels/slack/runtime.py.
- Web API ``{"ok": false, "error": "invalid_auth"}``: observed live from
  slack.com on 2026-11-16 with made-up tokens (auth.test and
  apps.connections.open); ``not_allowed_token_type`` from the
  apps.connections.open reference page.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from agent_os.sources.adapters import slack as sl
from agent_os.sources.model import SourceFailure, VerifyError
from tests.unit._sources_bot_fakes import FakeHTTP, FakeWS, Run, verify_ctx

APP = "xapp-" + "1-A0FAKE-1234567890-abcdefabcdefabcdefabcdef"  # split so secret scanners do not flag a test value
BOT = "xoxb-" + "1111111111-2222222222-FaKeBoTtOkEnFoRtEsTs12"  # split so secret scanners do not flag a test value
BOT_UID = "U0BOT"


class SlackFake:
    def __init__(self):
        self.http = FakeHTTP()
        self.ws = FakeWS(self._conn)
        self.acks: list[str] = []
        self.frames: asyncio.Queue | None = None
        self.app_error: str | None = None
        self.bot_error: str | None = None
        self.users = {"U1": "Jordan", "U2": "Kai"}
        self.channels = {"C1": "support"}
        self.http.route("POST", "/api/auth.test", self._auth_test)
        self.http.route("POST", "/api/apps.connections.open", self._open)
        self.http.route("GET", "/api/users.info", self._user)
        self.http.route("GET", "/api/conversations.info", self._chan)

    async def start(self):
        self.http.start()
        await self.ws.start()
        self.frames = asyncio.Queue()
        return self

    async def stop(self):
        self.http.stop()
        await self.ws.stop()

    def _tok(self, req):
        return (req.headers.get("authorization") or "").removeprefix("Bearer ")

    def _auth_test(self, req):
        if self.bot_error or self._tok(req) != BOT:
            return 200, {"ok": False, "error": self.bot_error or "invalid_auth"}
        return 200, {"ok": True, "url": "https://acme.slack.com/", "team": "acme", "user": "orbital",
                     "team_id": "T1", "user_id": BOT_UID, "bot_id": "B0BOT"}

    def _open(self, req):
        tok = self._tok(req)
        if self.app_error:
            return 200, {"ok": False, "error": self.app_error}
        if tok.startswith("xoxb-"):
            return 200, {"ok": False, "error": "not_allowed_token_type"}
        if tok != APP:
            return 200, {"ok": False, "error": "invalid_auth"}
        return 200, {"ok": True, "url": self.ws.url + "/link/?ticket=t-secret-ticket&app_id=A1"}

    def _user(self, req):
        uid = req.query.get("user")
        if uid not in self.users:
            return 200, {"ok": False, "error": "user_not_found"}
        return 200, {"ok": True, "user": {"id": uid, "name": uid.lower(),
                                          "profile": {"display_name": self.users[uid]}}}

    def _chan(self, req):
        cid = req.query.get("channel")
        if cid not in self.channels:
            return 200, {"ok": False, "error": "missing_scope"}
        return 200, {"ok": True, "channel": {"id": cid, "name": self.channels[cid]}}

    async def _conn(self, ws):
        await ws.send(json.dumps({"type": "hello", "num_connections": 1,
                                  "connection_info": {"app_id": "A1"}}))

        async def reader():
            async for m in ws:
                self.acks.append(json.loads(m)["envelope_id"])
        rt = asyncio.create_task(reader())
        try:
            while True:
                get = asyncio.ensure_future(self.frames.get())
                await asyncio.wait([get, rt], return_when=asyncio.FIRST_COMPLETED)
                if not get.done():  # the client went away
                    get.cancel()
                    return
                frame = get.result()
                if frame == "CLOSE":
                    await ws.close()
                    return
                if frame == "DROP":
                    ws.transport.abort() if hasattr(ws, "transport") else await ws.close(1011)
                    return
                await ws.send(json.dumps(frame))
        finally:
            rt.cancel()

    def send(self, frame):
        self.frames.put_nowait(frame)


_n = 0


def envelope(event: dict, *, env_id=None, retry=0):
    global _n
    _n += 1
    return {"envelope_id": env_id or f"env-{_n}", "type": "events_api", "accepts_response_payload": False,
            "retry_attempt": retry, "retry_reason": "",
            "payload": {"type": "event_callback", "team_id": "T1", "event": event}}


def im(text, ts="1763280000.000100", **extra):
    return {"type": "message", "channel_type": "im", "channel": "D1", "user": "U1", "text": text,
            "ts": ts, "event_ts": ts, **extra}


def mention(text, ts="1763280100.000200", **extra):
    return {"type": "app_mention", "channel": "C1", "user": "U2", "text": text, "ts": ts,
            "event_ts": ts, **extra}


@pytest.fixture
async def fake():
    f = await SlackFake().start()
    yield f
    await f.stop()


@pytest.fixture
def adapter(fake):
    a = sl.SlackAdapter()
    a.api_base = fake.http.url + "/api"
    return a


SECRETS = {"app": APP, "bot": BOT}


# ── verify ─────────────────────────────────────────────────────────────────

async def test_verify_checks_both_tokens_and_labels_bot_and_team(adapter):
    v = await adapter.verify(verify_ctx(SECRETS))
    assert v.label == "orbital · acme"


@pytest.mark.parametrize("secrets,field,reason", [
    ({"app": "nope", "bot": BOT}, "app", "invalid"),
    ({"app": APP, "bot": "xoxp-user"}, "bot", "invalid"),
    ({"app": APP, "bot": "xoxb-wrong"}, "bot", "rejected"),
    ({"app": "xapp-wrong", "bot": BOT}, "app", "rejected"),
])
async def test_verify_maps_each_failure_to_its_field(adapter, secrets, field, reason):
    with pytest.raises(VerifyError) as e:
        await adapter.verify(verify_ctx(secrets))
    assert (e.value.field, e.value.reason) == (field, reason)
    for s in secrets.values():
        assert s not in e.value.message


# ── arrivals ───────────────────────────────────────────────────────────────

async def test_dm_and_mention_become_items_and_every_envelope_is_acked(adapter, fake):
    run = Run(adapter, secrets=SECRETS).start()
    await run.until(lambda: run.connects == 1)
    fake.send(envelope(im("payment page spins forever"), env_id="e1"))
    fake.send(envelope(mention("<@U0BOT> customer says <#C9|exports> time out, see <https://x.io/a|log> &amp; fix"),
                       env_id="e2"))
    await run.until(lambda: len(run.items) == 2)
    a, b = run.items
    assert (a.kind, a.sender, a.place, a.id) == ("dm", "Jordan", {"kind": "dm"}, "D1:1763280000.000100")
    assert a.url == "https://acme.slack.com/archives/D1/p1763280000000100"
    assert (b.kind, b.sender, b.place) == ("at", "Kai", {"kind": "channel", "name": "#support"})
    assert b.title == "customer says #exports time out, see log & fix"
    await run.until(lambda: fake.acks == ["e1", "e2"])
    await run.cancel()


async def test_own_bot_edits_subtypes_and_channel_messages_are_ignored(adapter, fake):
    run = Run(adapter, secrets=SECRETS).start()
    await run.until(lambda: run.connects == 1)
    for ev in (im("mine", user=BOT_UID), im("from a bot", bot_id="B9"),
               im("edited", subtype="message_changed"), im("bye", subtype="message_deleted"),
               im("joined", subtype="channel_join"),
               {**im("in a channel"), "channel_type": "channel"},
               mention("<@U0BOT> again", edited={"user": "U2", "ts": "1"}),
               {"type": "reaction_added", "user": "U1"}):
        fake.send(envelope(ev))
    fake.send(envelope(im("real", ts="1763280999.000001")))
    await run.until(lambda: len(run.items) == 1)
    await asyncio.sleep(0.2)
    assert [i.title for i in run.items] == ["real"]
    assert len(fake.acks) == 9  # ignored envelopes are acked too
    await run.cancel()


async def test_a_retried_envelope_is_emitted_once(adapter, fake):
    run = Run(adapter, secrets=SECRETS).start()
    await run.until(lambda: run.connects == 1)
    ev = im("same one")
    fake.send(envelope(ev, env_id="a"))
    fake.send(envelope(ev, env_id="b", retry=1))
    await run.until(lambda: len(fake.acks) == 2)
    assert len(run.items) == 1
    await run.cancel()


async def test_files_get_placeholders_and_unknown_names_fall_back(adapter, fake):
    run = Run(adapter, secrets=SECRETS).start()
    await run.until(lambda: run.connects == 1)
    fake.send(envelope(im("", subtype="file_share", user="U7",
                          files=[{"mimetype": "image/png"}, {"mimetype": "application/pdf"}])))
    fake.send(envelope(mention("<@U0BOT> ping", channel="C404", ts="1763280200.0003")))
    await run.until(lambda: len(run.items) == 2)
    assert run.items[0].title == "[image] [file]" and run.items[0].sender is None
    assert run.items[1].place == {"kind": "channel", "name": None}
    await run.cancel()


# ── disconnects ────────────────────────────────────────────────────────────

async def test_refresh_requested_returns_so_the_core_reconnects_with_a_fresh_url(adapter, fake):
    run = Run(adapter, secrets=SECRETS).start()
    await run.until(lambda: run.connects == 1)
    fake.send({"type": "disconnect", "reason": "warning", "debug_info": {"host": "wss-1"}})
    fake.send(envelope(im("before refresh")))
    fake.send({"type": "disconnect", "reason": "refresh_requested", "debug_info": {"host": "wss-1"}})
    assert await run.result() is None  # a clean end, not a failure
    assert [i.title for i in run.items] == ["before refresh"]
    # next connection (what the core does): new URL from apps.connections.open, no duplicate
    opens = len(fake.http.calls("/api/apps.connections.open"))
    run2 = Run(adapter, secrets=SECRETS).start()
    await run2.until(lambda: run2.connects == 1)
    assert len(fake.http.calls("/api/apps.connections.open")) == opens + 1
    fake.send(envelope(im("after refresh", ts="1763281000.0001")))
    await run2.until(lambda: len(run2.items) == 1)
    assert run2.items[0].title == "after refresh"
    await run2.cancel()


async def test_a_dropped_socket_is_an_unreachable_end(adapter, fake):
    run = Run(adapter, secrets=SECRETS).start()
    await run.until(lambda: run.connects == 1)
    fake.send("DROP")
    err = await run.result()
    assert isinstance(err, ConnectionError)


async def test_socket_mode_switched_off_is_a_failure(adapter, fake):
    run = Run(adapter, secrets=SECRETS).start()
    await run.until(lambda: run.connects == 1)
    fake.send({"type": "disconnect", "reason": "link_disabled"})
    err = await run.result()
    assert isinstance(err, SourceFailure) and err.params == {"reason": "socket_mode_off"}


async def test_a_revoked_token_is_a_non_retryable_failure_naming_its_field(adapter, fake):
    fake.app_error = "invalid_auth"
    err = await Run(adapter, secrets=SECRETS).start().result()
    assert isinstance(err, SourceFailure) and err.code == "token_rejected"
    assert err.params == {"field": "app"} and not err.retryable
    fake.app_error, fake.bot_error = None, "token_revoked"
    err = await Run(adapter, secrets=SECRETS).start().result()
    assert err.code == "token_rejected" and err.params == {"field": "bot"}


async def test_tokens_and_the_socket_ticket_never_reach_a_log_line(adapter, fake, caplog):
    caplog.set_level(logging.DEBUG)
    run = Run(adapter, secrets=SECRETS).start()
    await run.until(lambda: run.connects == 1)
    fake.send(envelope(im("hello")))
    await run.until(lambda: len(run.items) == 1)
    await run.cancel()
    ours = "\n".join(r.getMessage() for r in caplog.records
                     if not r.name.startswith("websockets.server"))  # the fake's own log
    assert "GET /link/?ticket=" in ours  # websockets did log the handshake…
    for s in (APP, BOT, "t-secret-ticket"):
        assert s not in ours  # …without the ticket
