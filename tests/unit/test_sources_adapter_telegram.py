# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Telegram bot source against a local Bot API fake.

Frame shapes (all from the official Bot API reference,
https://core.telegram.org/bots/api — getMe, getWebhookInfo, getUpdates,
Update, Message, MessageEntity, the ``{"ok", "result" | "error_code",
"description", "parameters": {"retry_after"}}`` envelope). Error texts for
401 ("Unauthorized"), 404 ("Not Found" for a malformed token) and the two
409 conflicts are the ones the real API returns (the 401/404 bodies were
also read from the live API with a made-up token on 2026-11-16; the 409
wordings are from the Bot API docs/community knowledge, not observed here).
"""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest

from agent_os.sources.adapters import telegram as tg
from agent_os.sources.core import SourcesCore
from agent_os.sources.model import SourceFailure, VerifyError
from agent_os.sources.registry import AdapterRegistry
from agent_os.sources.secrets import MemorySecretStore
from tests.unit._sources_bot_fakes import FakeHTTP, Run, verify_ctx

TOKEN = "7123456789:AAH-fakeTokenForTests_0123456789abc"
BOT = {"id": 7123456789, "is_bot": True, "first_name": "Orbital", "username": "orbital_test_bot"}


class TgFake:
    """Bot API over HTTP: /bot<token>/<method>."""

    def __init__(self):
        self.http = FakeHTTP()
        self.updates: list[dict] = []
        self.cond = threading.Condition()
        self.webhook = ""
        self.conflict: str | None = None
        self.unauthorized = False
        self.offsets: list = []
        self.next_id = 1000
        self.http.route("POST", "/bot*", self._dispatch)

    def start(self):
        self.http.start()
        return self

    def push(self, message: dict) -> int:
        with self.cond:
            self.next_id += 1
            self.updates.append({"update_id": self.next_id, "message": message})
            self.cond.notify_all()
            return self.next_id

    def _dispatch(self, req):
        token, _, method = req.path[len("/bot"):].partition("/")
        if token != TOKEN or self.unauthorized:
            return 401, {"ok": False, "error_code": 401, "description": "Unauthorized"}
        body = req.json or {}
        if method == "getMe":
            return 200, {"ok": True, "result": BOT}
        if method == "getWebhookInfo":
            return 200, {"ok": True, "result": {"url": self.webhook, "pending_update_count": 0}}
        if method == "getUpdates":
            if self.webhook:
                return 409, {"ok": False, "error_code": 409, "description":
                             "Conflict: can't use getUpdates method while webhook is active; "
                             "use deleteWebhook to delete the webhook first"}
            if self.conflict:
                return 409, {"ok": False, "error_code": 409, "description": self.conflict}
            offset = body.get("offset")
            self.offsets.append(offset)
            timeout = min(float(body.get("timeout") or 0), 1.0)
            with self.cond:
                if offset:
                    self.updates = [u for u in self.updates if u["update_id"] >= offset]
                if not self.updates and timeout:
                    self.cond.wait(timeout)
                return 200, {"ok": True, "result": list(self.updates)}
        return 404, {"ok": False, "error_code": 404, "description": "Not Found"}


def dm(text="把 v0.22 的发布说明翻成英文", mid=11, **extra):
    return {"message_id": mid, "date": 1763280000,
            "from": {"id": 42, "is_bot": False, "first_name": "Mira", "last_name": "Chen"},
            "chat": {"id": 42, "type": "private", "first_name": "Mira"}, "text": text, **extra}


def group(text, mid=21, **extra):
    return {"message_id": mid, "date": 1763280100,
            "from": {"id": 43, "is_bot": False, "first_name": "Tom"},
            "chat": {"id": -1001234567890, "type": "supergroup", "title": "Acme Dev"},
            "text": text, **extra}


@pytest.fixture
def fake():
    f = TgFake().start()
    yield f
    f.http.stop()


@pytest.fixture
def adapter(fake):
    a = tg.TelegramAdapter()
    a.api_base = fake.http.url
    a.long_poll_s = 1
    return a


# ── verify ─────────────────────────────────────────────────────────────────

async def test_verify_returns_the_bot_handle(adapter):
    v = await adapter.verify(verify_ctx({"token": TOKEN}))
    assert v.label == "@orbital_test_bot"


@pytest.mark.parametrize("bad,reason", [("not-a-token", "invalid"),
                                        ("7123456789:AAH-wrongTokenForTests_000000000", "rejected")])
async def test_verify_marks_the_token_field(adapter, bad, reason):
    with pytest.raises(VerifyError) as e:
        await adapter.verify(verify_ctx({"token": bad}))
    assert e.value.field == "token" and e.value.reason == reason
    assert bad not in e.value.message


async def test_verify_refuses_a_bot_with_a_webhook_and_never_removes_it(adapter, fake):
    fake.webhook = "https://example.com/hook"
    with pytest.raises(VerifyError) as e:
        await adapter.verify(verify_ctx({"token": TOKEN}))
    assert e.value.field == "token" and "webhook" in e.value.message
    assert not [r for r in fake.http.requests if r.path.endswith("/deleteWebhook")]


async def test_verify_unreachable_is_not_a_token_problem():
    a = tg.TelegramAdapter()
    a.api_base = "http://127.0.0.1:9"  # nothing listens
    with pytest.raises(VerifyError) as e:
        await a.verify(verify_ctx({"token": TOKEN}))
    assert e.value.reason == "unreachable" and TOKEN not in e.value.message


# ── arrivals ───────────────────────────────────────────────────────────────

async def test_dm_and_group_mention_become_items(adapter, fake):
    run = Run(adapter, secrets={"token": TOKEN}).start()
    await run.until(lambda: run.connects == 1)
    fake.push(dm())
    fake.push(group("@orbital_test_bot staging returns 502 again\nsecond line"))
    await run.until(lambda: len(run.items) == 2)
    a, b = run.items
    assert (a.kind, a.sender, a.place, a.id) == ("dm", "Mira Chen", {"kind": "dm"}, "42:11")
    assert a.title == "把 v0.22 的发布说明翻成英文" and a.url is None and a.at == "2025-11-16T08:00:00Z"
    assert (b.kind, b.sender, b.place) == ("at", "Tom", {"kind": "group", "name": "Acme Dev"})
    assert b.title == "staging returns 502 again" and "second line" in b.body
    assert b.url == "https://t.me/c/1234567890/21"
    await run.cancel()


async def test_unmentioned_group_talk_own_messages_and_channels_are_ignored(adapter, fake):
    run = Run(adapter, secrets={"token": TOKEN}).start()
    await run.until(lambda: run.connects == 1)
    fake.push(group("just chatting, @someone_else look", mid=30))
    fake.push({**dm("echo", mid=31), "from": {"id": BOT["id"], "is_bot": True, "first_name": "Orbital"}})
    fake.push({"message_id": 32, "date": 1, "chat": {"id": -100555, "type": "channel", "title": "News"},
               "text": "@orbital_test_bot hi"})
    fake.push(dm("last", mid=33))
    await run.until(lambda: len(run.items) >= 1)
    await asyncio.sleep(0.2)
    assert [i.id for i in run.items] == ["42:33"]
    await run.cancel()


async def test_text_mention_entity_counts_and_media_get_a_placeholder(adapter, fake):
    run = Run(adapter, secrets={"token": TOKEN}).start()
    await run.until(lambda: run.connects == 1)
    fake.push(group("Orbital please look", mid=40,
                    entities=[{"type": "text_mention", "offset": 0, "length": 7, "user": BOT}]))
    fake.push({**dm("", mid=41), "text": None, "photo": [{"file_id": "x"}], "caption": "screenshot"})
    fake.push({**dm("", mid=42), "text": None, "voice": {"file_id": "v"}})
    await run.until(lambda: len(run.items) == 3)
    assert run.items[0].kind == "at"
    assert run.items[1].title == "[image] screenshot"
    assert run.items[2].title == "[voice]"
    await run.cancel()


# ── failures ───────────────────────────────────────────────────────────────

async def test_a_revoked_token_is_a_non_retryable_failure(adapter, fake):
    fake.unauthorized = True
    run = Run(adapter, secrets={"token": TOKEN}).start()
    err = await run.result()
    assert isinstance(err, SourceFailure) and err.code == "token_rejected" and not err.retryable


async def test_a_webhook_set_later_fails_with_its_own_code(adapter, fake):
    run = Run(adapter, secrets={"token": TOKEN}).start()
    await run.until(lambda: run.connects == 1)
    fake.webhook = "https://example.com/hook"
    err = await run.result()
    assert isinstance(err, SourceFailure) and err.code == "webhook_set" and err.retryable


async def test_another_poller_is_bot_in_use(adapter, fake):
    fake.conflict = ("Conflict: terminated by other getUpdates request; make sure that only one "
                     "bot instance is running")
    run = Run(adapter, secrets={"token": TOKEN}).start()
    err = await run.result()
    assert isinstance(err, SourceFailure) and err.code == "bot_in_use"


async def test_a_dropped_server_ends_the_stream_without_the_token(adapter, fake):
    run = Run(adapter, secrets={"token": TOKEN}).start()
    await run.until(lambda: run.connects == 1)
    fake.http.stop()
    err = await run.result(timeout=10)
    assert isinstance(err, ConnectionError) and TOKEN not in str(err)


# ── through the core: offset across a restart, duplicates, no gaps ─────────

class _Send:
    def __init__(self):
        self.n = 0

    async def __call__(self, text, project_id):
        self.n += 1
        return {"project_id": "proj_a", "session_id": f"s{self.n}", "by": "model"}


def _core(tmp_path, adapter, secrets):
    async def _info(pid, sid):
        return pid, "done"
    return SourcesCore(str(tmp_path), registry=AdapterRegistry(overrides={"telegram": adapter}),
                       secret_store=secrets, send=_Send(), session_info=_info)


async def _wait(cond, timeout=5.0):
    async def _w():
        while not cond():
            await asyncio.sleep(0.02)
    await asyncio.wait_for(_w(), timeout)


async def test_offset_survives_a_restart_and_nothing_is_sent_twice(adapter, fake, tmp_path):
    secrets = MemorySecretStore()
    core = _core(tmp_path, adapter, secrets)
    core.load()
    sid = (await core.create({"type": "telegram", "kinds": {"dm": {}, "at": {}},
                              "secrets": {"token": TOKEN}}))["id"]
    core._running = True
    await core.supervise_streams()
    first = fake.push(dm("one", mid=1))
    fake.push(dm("two", mid=2))
    await _wait(lambda: len(core.store.rt(sid)["entries"]) == 2)
    await core.stop()
    assert core.store.rt(sid)["state"]["offset"] == first + 2

    # A redelivery (Telegram resends what was not confirmed) and a fresh core.
    fake.updates.insert(0, {"update_id": first, "message": dm("one", mid=1)})
    core2 = _core(tmp_path, adapter, secrets)
    core2.load()
    core2._running = True
    calls_before = len(fake.offsets)
    await core2.supervise_streams()
    fake.push(dm("three", mid=3))
    await _wait(lambda: len(core2.store.rt(sid)["entries"]) == 3)
    assert fake.offsets[calls_before] == first + 2  # resumed after what was confirmed
    titles = sorted(e["title"] for e in core2.store.rt(sid)["entries"])
    assert titles == ["one", "three", "two"]
    assert (await core2.detail(sid))["history"][0]["kind"] == "item"  # no gap: lossless
    await core2.stop()


async def test_the_token_never_reaches_a_log_line(adapter, fake, caplog):
    caplog.set_level(logging.DEBUG)
    run = Run(adapter, secrets={"token": TOKEN}).start()
    await run.until(lambda: run.connects == 1)
    fake.push(dm())
    await run.until(lambda: len(run.items) == 1)
    await run.cancel()
    await adapter.verify(verify_ctx({"token": TOKEN}))
    text = caplog.text
    assert "getUpdates" in text  # httpx did log its request lines…
    assert TOKEN not in text and TOKEN.split(":")[1] not in text  # …without the token
