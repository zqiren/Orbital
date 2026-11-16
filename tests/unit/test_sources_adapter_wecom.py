# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""WeCom smart-robot source against a local long-connection fake.

Frame shapes (no official prose document was reachable; confidence in each):
- ``aibot_subscribe`` ``{"cmd", "headers": {"req_id"}, "body": {"bot_id",
  "secret"}}`` and its answer ``{"headers": {"req_id"}, "errcode",
  "errmsg"}``: WeCom team Node SDK (github.com/WecomTeam/aibot-node-sdk
  src/ws.ts) + the Python port; the failure answer (errcode 853000 "invalid
  bot_id or secret") was observed live on 2026-11-16 with a made-up pair.
  High.
- ``ping`` with a ``req_id`` and its ``errcode: 0`` answer: both SDKs.
  Medium-high (the live check did not run long enough to see one).
- ``aibot_msg_callback`` body (``msgid``, ``aibotid``, ``chatid``,
  ``chattype``, ``from.userid``, ``msgtype``, ``text.content``,
  ``mixed.msg_item``, ``voice.content``): the Node SDK's
  src/types/message.ts. Medium (typed, never seen on the wire).
- ``aibot_event_callback`` with ``event.eventtype: "disconnected_event"``:
  the Python port's ws.py. Medium.
- That group text starts with "@<robot name>": assumption (low).
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from agent_os.sources.adapters import wecom as wc
from agent_os.sources.model import SourceFailure, VerifyError
from tests.unit._sources_bot_fakes import FakeWS, Run, verify_ctx

BID = "aibFakeBotId0000000001"
BSEC = "FakeWeComBotSecretForTests0123456789"


class WeComFake:
    def __init__(self):
        self.ws = FakeWS(self._conn)
        self.subscribes: list[dict] = []
        self.pings = 0
        self.answer_pings = True
        self.out: asyncio.Queue | None = None

    async def start(self):
        await self.ws.start()
        self.out = asyncio.Queue()
        return self

    async def stop(self):
        await self.ws.stop()

    async def _conn(self, ws):
        async def reader():
            async for raw in ws:
                f = json.loads(raw)
                rid = f["headers"]["req_id"]
                if f["cmd"] == "aibot_subscribe":
                    self.subscribes.append(f["body"])
                    ok = f["body"] == {"bot_id": BID, "secret": BSEC}
                    await ws.send(json.dumps({"headers": {"req_id": rid}, "errcode": 0 if ok else 853000,
                                              "errmsg": "ok" if ok else "invalid bot_id or secret"}))
                elif f["cmd"] == "ping":
                    self.pings += 1
                    if self.answer_pings:
                        await ws.send(json.dumps({"headers": {"req_id": rid}, "errcode": 0, "errmsg": "ok"}))
        rt = asyncio.create_task(reader())
        try:
            while True:
                get = asyncio.ensure_future(self.out.get())
                await asyncio.wait([get, rt], return_when=asyncio.FIRST_COMPLETED)
                if not get.done():
                    get.cancel()
                    return
                frame = get.result()
                if frame == "DROP":
                    await ws.close(1011)
                    return
                await ws.send(json.dumps(frame, ensure_ascii=False))
        finally:
            rt.cancel()

    def push(self, body: dict, cmd="aibot_msg_callback"):
        self.out.put_nowait({"cmd": cmd, "headers": {"req_id": f"cb_{body.get('msgid', 'e')}"}, "body": body})


def msg(msgid, *, group=False, text="整理一下本周客户反馈", msgtype="text", **extra):
    b = {"msgid": msgid, "aibotid": BID, "chattype": "group" if group else "single",
         "from": {"userid": "zhangsan"}, "msgtype": msgtype, "response_url": "https://example.invalid/r"}
    if group:
        b["chatid"] = "wrkSFfCgAAxxx"
    if msgtype == "text":
        b["text"] = {"content": text}
    b.update(extra)
    return b


@pytest.fixture
async def fake():
    f = await WeComFake().start()
    yield f
    await f.stop()


@pytest.fixture
def adapter(fake):
    a = wc.WeComAdapter()
    a.ws_url = fake.ws.url
    a.heartbeat_s = 0.05
    return a


CFG = {"bid": BID}
SECRETS = {"bsec": BSEC}


async def test_verify_subscribes_and_labels_with_the_bot_id(adapter, fake):
    assert (await adapter.verify(verify_ctx(SECRETS, CFG))).label == BID
    assert fake.subscribes == [{"bot_id": BID, "secret": BSEC}]


@pytest.mark.parametrize("cfg,sec,field,reason", [
    (CFG, {"bsec": "wrong"}, "bsec", "rejected"),
    ({"bid": ""}, SECRETS, "bid", "invalid"),
    (CFG, {"bsec": ""}, "bsec", "invalid"),
])
async def test_verify_failures(adapter, cfg, sec, field, reason):
    with pytest.raises(VerifyError) as e:
        await adapter.verify(verify_ctx(sec, cfg))
    assert (e.value.field, e.value.reason) == (field, reason)


async def test_verify_when_nothing_answers():
    a = wc.WeComAdapter()
    a.ws_url = "ws://127.0.0.1:9"
    with pytest.raises(VerifyError) as e:
        await a.verify(verify_ctx(SECRETS, CFG))
    assert e.value.reason == "unreachable"


async def test_single_and_group_messages_become_items(adapter, fake):
    run = Run(adapter, secrets=SECRETS, config=CFG).start()
    await run.until(lambda: run.connects == 1)
    fake.push(msg("m1", create_time=1763280000))
    fake.push(msg("m2", group=True, text="@Orbital 机器人 帮我查一下报价单"))
    fake.push(msg("m3", msgtype="mixed", mixed={"msg_item": [{"msgtype": "text", "text": {"content": "看下这个"}},
                                                             {"msgtype": "image", "image": {"url": "u"}}]}))
    fake.push(msg("m4", msgtype="voice", voice={"content": "明天开会"}))
    fake.push(msg("m5", msgtype="file", file={"url": "u"}))
    await run.until(lambda: len(run.items) == 5)
    a, b, c, d, e = run.items
    assert (a.kind, a.id, a.sender, a.place, a.title, a.at) == (
        "dm", "m1", "zhangsan", {"kind": "dm"}, "整理一下本周客户反馈", "2025-11-16T08:00:00Z")
    assert (b.kind, b.place, b.title) == ("at", {"kind": "group", "name": None}, "机器人 帮我查一下报价单")
    assert c.title == "看下这个 [image]" and d.title == "[voice] 明天开会" and e.title == "[file]"
    await run.cancel()


async def test_events_and_duplicates(adapter, fake):
    run = Run(adapter, secrets=SECRETS, config=CFG).start()
    await run.until(lambda: run.connects == 1)
    fake.push({"msgid": "ev1", "event": {"eventtype": "enter_chat"}, "msgtype": "event"}, cmd="aibot_event_callback")
    fake.push(msg("dup"))
    fake.push(msg("dup"))
    fake.push(msg("last"))
    await run.until(lambda: len(run.items) == 2)
    await asyncio.sleep(0.1)
    assert [i.id for i in run.items] == ["dup", "last"]
    await run.cancel()


async def test_heartbeats_and_a_dead_connection(adapter, fake):
    run = Run(adapter, secrets=SECRETS, config=CFG).start()
    await run.until(lambda: fake.pings >= 3)
    assert not run.task.done()
    fake.answer_pings = False
    err = await run.result()
    assert isinstance(err, ConnectionError)


async def test_another_connection_taking_over_is_bot_in_use(adapter, fake):
    run = Run(adapter, secrets=SECRETS, config=CFG).start()
    await run.until(lambda: run.connects == 1)
    adapter._verified_at.clear()
    fake.push({"msgid": "d", "msgtype": "event", "event": {"eventtype": "disconnected_event"}},
              cmd="aibot_event_callback")
    err = await run.result()
    assert isinstance(err, SourceFailure) and err.code == "bot_in_use" and err.retryable


async def test_our_own_check_taking_over_is_a_clean_end(adapter, fake):
    run = Run(adapter, secrets=SECRETS, config=CFG).start()
    await run.until(lambda: run.connects == 1)
    await adapter.verify(verify_ctx(SECRETS, CFG))
    fake.push({"msgid": "d", "msgtype": "event", "event": {"eventtype": "disconnected_event"}},
              cmd="aibot_event_callback")
    assert await run.result() is None


async def test_rejected_secret_is_non_retryable_and_drop_is_unreachable(adapter, fake):
    err = await Run(adapter, secrets={"bsec": "rotated"}, config=CFG).start().result()
    assert isinstance(err, SourceFailure) and err.code == "token_rejected" and not err.retryable
    adapter.heartbeat_s = 30
    run = Run(adapter, secrets=SECRETS, config=CFG).start()
    await run.until(lambda: run.connects == 1)
    fake.out.put_nowait("DROP")
    assert isinstance(await run.result(), ConnectionError)


async def test_the_secret_never_reaches_a_log_line(adapter, fake, caplog):
    caplog.set_level(logging.DEBUG)
    run = Run(adapter, secrets=SECRETS, config=CFG).start()
    await run.until(lambda: run.connects == 1)
    await run.cancel()
    ours = "\n".join(r.getMessage() for r in caplog.records if not r.name.startswith("websockets.server"))
    assert "aibot_subscribe" in ours  # websockets logged the subscribe frame…
    assert BSEC not in ours and BSEC[:10] not in ours  # …without the secret
