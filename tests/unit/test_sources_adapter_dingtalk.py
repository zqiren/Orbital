# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""DingTalk bot source against a local Open API + Stream Mode fake.

Frame shapes:
- ``POST /v1.0/gateway/connections/open`` body/answer (``endpoint``,
  ``ticket``), ``<endpoint>?ticket=``, frame ``{"specVersion", "type",
  "headers": {"messageId", "topic", …}, "data": "<json>"}``, the ack
  ``{"code": 200, "headers": {"messageId", "contentType"}, "message",
  "data"}``, SYSTEM ``ping``/``disconnect``: official SDK
  dingtalk-stream-sdk-python (stream.py, frames.py, handlers.py).
- Bot message callback fields (``msgId``, ``conversationType`` "1"/"2",
  ``conversationTitle``, ``senderNick``, ``senderId``, ``chatbotUserId``,
  ``isInAtList``, ``msgtype``, ``text.content``, ``content.richText``,
  ``createAt``): the same SDK's chatbot.py ``ChatbotMessage.from_dict``.
- ``oauth2/accessToken`` 400 ``{"code": "invalidClientIdOrSecret"}`` and
  ``connections/open`` 401 ``{"code": "authFailed"}``: observed live on
  2026-11-16 with made-up credentials.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from agent_os.sources.adapters import dingtalk as dt
from agent_os.sources.model import SourceFailure, VerifyError
from tests.unit._sources_bot_fakes import FakeHTTP, FakeWS, Run, verify_ctx

CID = "dingfakeclientid0001"
CSEC = "FakeDingTalkClientSecretForTests_0123456789"


class DingFake:
    def __init__(self):
        self.http = FakeHTTP()
        self.ws = FakeWS(self._conn)
        self.acks: list[dict] = []
        self.subscriptions = None
        self.out: asyncio.Queue | None = None
        self.http.route("POST", "/v1.0/oauth2/accessToken", self._token)
        self.http.route("POST", "/v1.0/gateway/connections/open", self._open)

    async def start(self):
        self.http.start()
        await self.ws.start()
        self.out = asyncio.Queue()
        return self

    async def stop(self):
        self.http.stop()
        await self.ws.stop()

    def _token(self, req):
        b = req.json or {}
        if (b.get("appKey"), b.get("appSecret")) != (CID, CSEC):
            return 400, {"requestid": "r", "code": "invalidClientIdOrSecret", "message": "无效的clientId或者clientSecret"}
        return 200, {"accessToken": "at-fake", "expireIn": 7200}

    def _open(self, req):
        b = req.json or {}
        self.subscriptions = b.get("subscriptions")
        if (b.get("clientId"), b.get("clientSecret")) != (CID, CSEC):
            return 401, {"requestid": "r", "code": "authFailed", "message": "鉴权失败"}
        return 200, {"endpoint": self.ws.url + "/connect", "ticket": "TICKET-SECRET/+1"}

    async def _conn(self, ws):
        async def reader():
            async for raw in ws:
                self.acks.append(json.loads(raw))
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

    _n = 0

    def callback(self, data: dict, *, mid=None, topic=dt.BOT_TOPIC):
        DingFake._n += 1
        self.out.put_nowait({"specVersion": "1.0", "type": "CALLBACK",
                             "headers": {"appId": "a", "connectionId": "c", "contentType": "application/json",
                                         "messageId": mid or f"mid-{DingFake._n}", "time": "1763280000000",
                                         "topic": topic},
                             "data": json.dumps(data, ensure_ascii=False)})

    def system(self, topic, data="{}"):
        self.out.put_nowait({"specVersion": "1.0", "type": "SYSTEM",
                             "headers": {"contentType": "application/json", "messageId": f"sys-{topic}",
                                         "topic": topic}, "data": data})


def bot_msg(msg_id, *, group=False, text="明早提醒我看合同修改", msgtype="text", **extra):
    d = {"msgId": msg_id, "conversationType": "2" if group else "1", "conversationId": "cid1",
         "senderNick": "陈雨", "senderId": "$:LWCP_v1:$abc", "senderStaffId": "staff1",
         "chatbotUserId": "$:LWCP_v1:$bot", "robotCode": CID, "createAt": 1763280000123,
         "msgtype": msgtype, "sessionWebhook": "https://oapi.dingtalk.com/robot/sendBySession?session=SECRET"}
    if msgtype == "text":
        d["text"] = {"content": text}
    if group:
        d.update({"conversationTitle": "客户成功群", "isInAtList": True})
    d.update(extra)
    return d


@pytest.fixture
async def fake():
    f = await DingFake().start()
    yield f
    await f.stop()


@pytest.fixture
def adapter(fake):
    a = dt.DingTalkAdapter()
    a.api_base = fake.http.url
    return a


CFG = {"cid": CID}
SECRETS = {"csec": CSEC}


async def test_verify_uses_the_access_token_call(adapter):
    assert (await adapter.verify(verify_ctx(SECRETS, CFG))).label == CID


@pytest.mark.parametrize("cfg,sec,field,reason", [
    ({"cid": "wrong"}, SECRETS, "csec", "rejected"),
    (CFG, {"csec": "wrong"}, "csec", "rejected"),
    ({"cid": ""}, SECRETS, "cid", "invalid"),
    (CFG, {"csec": ""}, "csec", "invalid"),
])
async def test_verify_failures_mark_a_field(adapter, cfg, sec, field, reason):
    with pytest.raises(VerifyError) as e:
        await adapter.verify(verify_ctx(sec, cfg))
    assert (e.value.field, e.value.reason) == (field, reason)


async def test_dm_and_group_at_become_items_and_every_frame_is_acked(adapter, fake):
    run = Run(adapter, secrets=SECRETS, config=CFG).start()
    await run.until(lambda: run.connects == 1)
    assert fake.subscriptions == [{"type": "CALLBACK", "topic": "/v1.0/im/bot/messages/get"}]
    assert fake.ws.paths[0] == "/connect?ticket=TICKET-SECRET%2F%2B1"
    fake.callback(bot_msg("m1"), mid="cb-1")
    fake.callback(bot_msg("m2", group=True, text=" 退款邮件模板要换成新政策"), mid="cb-2")
    await run.until(lambda: len(run.items) == 2)
    a, b = run.items
    assert (a.kind, a.id, a.sender, a.place, a.title) == ("dm", "m1", "陈雨", {"kind": "dm"}, "明早提醒我看合同修改")
    assert a.at == "2025-11-16T08:00:00Z" and a.url is None
    assert (b.kind, b.place, b.title) == ("at", {"kind": "group", "name": "客户成功群"}, "退款邮件模板要换成新政策")
    await run.until(lambda: len(fake.acks) == 2)
    ack = fake.acks[0]
    assert ack["code"] == 200 and ack["headers"]["messageId"] == "cb-1"
    assert json.loads(ack["data"]) == {"response": None}
    await run.cancel()


async def test_pings_are_answered_and_other_topics_ignored(adapter, fake):
    run = Run(adapter, secrets=SECRETS, config=CFG).start()
    await run.until(lambda: run.connects == 1)
    fake.system("ping", data='{"opaque": "x"}')
    fake.callback(bot_msg("c1"), topic="/v1.0/card/instances/callback")
    fake.callback(bot_msg("own", senderId="$:LWCP_v1:$bot"))
    fake.callback(bot_msg("g0", group=True, isInAtList=False))
    fake.callback(bot_msg("rt", msgtype="richText", content={"richText": [{"text": "看图"}, {"pictureDownloadCode": "p"}]}))
    fake.callback(bot_msg("pic", msgtype="picture", content={"downloadCode": "d"}))
    await run.until(lambda: len(run.items) == 2)
    await asyncio.sleep(0.1)
    assert [i.title for i in run.items] == ["看图 [image]", "[image]"]
    ping_ack = fake.acks[0]
    assert ping_ack["headers"]["messageId"] == "sys-ping" and ping_ack["data"] == '{"opaque": "x"}'
    assert len(fake.acks) == 6
    await run.cancel()


async def test_a_redelivered_callback_is_emitted_once(adapter, fake):
    run = Run(adapter, secrets=SECRETS, config=CFG).start()
    await run.until(lambda: run.connects == 1)
    fake.callback(bot_msg("same"), mid="cb-a")
    fake.callback(bot_msg("same"), mid="cb-a")
    await run.until(lambda: len(fake.acks) == 2)
    assert len(run.items) == 1
    await run.cancel()


async def test_server_disconnect_ends_the_stream_cleanly(adapter, fake):
    run = Run(adapter, secrets=SECRETS, config=CFG).start()
    await run.until(lambda: run.connects == 1)
    fake.system("disconnect")
    assert await run.result() is None
    assert fake.acks[-1]["headers"]["messageId"] == "sys-disconnect"


async def test_a_dropped_socket_is_unreachable(adapter, fake):
    run = Run(adapter, secrets=SECRETS, config=CFG).start()
    await run.until(lambda: run.connects == 1)
    fake.out.put_nowait("DROP")
    assert isinstance(await run.result(), ConnectionError)


async def test_revoked_credentials_are_non_retryable(adapter, fake):
    err = await Run(adapter, secrets={"csec": "rotated"}, config=CFG).start().result()
    assert isinstance(err, SourceFailure) and err.code == "token_rejected" and not err.retryable


async def test_secret_and_ticket_never_reach_a_log_line(adapter, fake, caplog):
    caplog.set_level(logging.DEBUG)
    run = Run(adapter, secrets=SECRETS, config=CFG).start()
    await run.until(lambda: run.connects == 1)
    fake.callback(bot_msg("log"))
    await run.until(lambda: len(run.items) == 1)
    await run.cancel()
    ours = "\n".join(r.getMessage() for r in caplog.records if not r.name.startswith("websockets.server"))
    assert "/connect?ticket=" in ours
    for s in (CSEC, "TICKET-SECRET", "TICKET-SECRET%2F%2B1"):
        assert s not in ours
