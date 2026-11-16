# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Feishu bot source against a local Open API + long-connection fake.

Frame shapes and where they come from:
- ``pbbp2.Frame`` / ``Header`` field numbers and types: the official MIT SDK
  lark-oapi, ``lark_oapi/ws/pb/pbbp2_pb2.py`` (serialized descriptor), and
  ``lark_oapi/ws/client.py`` for ping/pong/data handling, ``sum``/``seq``
  fragments and the ``{"code": 200}`` + ``biz_rt`` ack. The golden ping bytes
  below are standard protobuf encoding of that message, worked by hand.
- ``POST /callback/ws/endpoint`` → ``{"code": 0, "data": {"URL",
  "ClientConfig": {"PingInterval", …}}}``: lark-oapi ``ws/model.py``; the
  error shape ``{"code": 1000040346, "msg": "app_id is invalid"}`` and
  ``tenant_access_token`` → ``{"code": 10003, "msg": "invalid param"}`` were
  observed live on 2026-11-16 with made-up credentials.
- ``im.message.receive_v1`` event body (``header.event_type``,
  ``event.sender.sender_id.open_id``, ``event.message.{message_id, chat_id,
  chat_type, message_type, content, mentions[{key, id.open_id, name}]}``):
  open.feishu.cn docs for the event, from memory, cross-read with nanobot
  channels/feishu/runtime.py. ``bot/v3/info``, ``im/v1/chats/{id}``,
  ``contact/v3/users/{id}`` response shapes: Open API docs from memory.
- App registration ``init`` / ``begin`` / ``poll`` (``authorization_pending``
  HTTP 400, ``invalid_grant``): observed live on 2026-11-16 (anonymous, no
  account); the success shape ``client_id`` + ``client_secret`` and
  ``access_denied``/``expired_token`` are from nanobot feishu/runtime.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import pytest

from agent_os.sources.adapters import feishu as fs
from agent_os.sources.core import SourcesCore
from agent_os.sources.model import SourceFailure, VerifyError
from agent_os.sources.registry import AdapterRegistry
from agent_os.sources.secrets import MemorySecretStore
from tests.unit._sources_bot_fakes import FakeHTTP, FakeWS, Run, verify_ctx

APP_ID = "cli_a1b2c3d4e5f60718"
SECRET = "FakeFeishuAppSecretForTests0123"
BOT_OPEN_ID = "ou_bot000000000000000000000000001"
TENANT = "t-faketenanttoken0001"


# ── the frame codec ─────────────────────────────────────────────────────────

def test_ping_frame_encodes_like_the_sdk():
    golden = b"\x08\x00\x10\x00\x18\x21\x20\x00\x2a\x0c\x0a\x04type\x12\x04ping"
    assert fs.encode_frame({"service": 33, "method": 0, "headers": [("type", "ping")]}) == golden


def test_data_frame_round_trips_and_unknown_fields_are_skipped():
    f = {"SeqID": 7, "LogID": 300, "service": 33, "method": 1, "payload": "中文".encode(),
         "headers": [("type", "event"), ("message_id", "m1"), ("sum", "1"), ("seq", "0")],
         "LogIDNew": "log-1"}
    raw = fs.encode_frame(f) + b"\x78\x05" + b"\x82\x01\x02hi"  # field 15 varint, field 16 bytes
    back = fs.decode_frame(raw)
    for k in ("SeqID", "LogID", "service", "method", "headers", "payload", "LogIDNew"):
        assert back[k] == f[k]


def test_a_truncated_frame_is_refused():
    with pytest.raises(ValueError):
        fs.decode_frame(fs.encode_frame({"payload": b"x" * 40})[:-5])


# ── the fake service ────────────────────────────────────────────────────────

class FeishuFake:
    def __init__(self):
        self.http = FakeHTTP()
        self.ws = FakeWS(self._conn)
        self.acks: list[dict] = []
        self.pings = 0
        self.out: asyncio.Queue | None = None
        self.endpoint_code = 0
        self.ping_interval = 120
        self.reg_state = "pending"
        self.http.route("POST", "/open-apis/auth/v3/tenant_access_token/internal", self._token)
        self.http.route("GET", "/open-apis/bot/v3/info", self._bot)
        self.http.route("POST", "/callback/ws/endpoint", self._endpoint)
        self.http.route("GET", "/open-apis/im/v1/chats/*", self._chat)
        self.http.route("GET", "/open-apis/contact/v3/users/*", self._user)
        self.http.route("POST", "/oauth/v1/app/registration", self._reg)

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
        if b.get("app_id") != APP_ID:
            return 200, {"code": 10003, "data": {}, "msg": "invalid param"}
        if b.get("app_secret") != SECRET:
            return 200, {"code": 10014, "msg": "app secret invalid"}
        return 200, {"code": 0, "msg": "ok", "tenant_access_token": TENANT, "expire": 7200}

    def _bearer_ok(self, req):
        return req.headers.get("authorization") == f"Bearer {TENANT}"

    def _bot(self, req):
        if not self._bearer_ok(req):
            return 400, {"code": 99991661, "msg": "Missing access token"}
        return 200, {"code": 0, "msg": "ok", "bot": {"activate_status": 2, "app_name": "Mira 的 Orbital",
                                                     "open_id": BOT_OPEN_ID}}

    def _endpoint(self, req):
        b = req.json or {}
        if self.endpoint_code:
            return 200, {"code": self.endpoint_code, "msg": "refused", "data": {"URL": ""}}
        if b.get("AppID") != APP_ID or b.get("AppSecret") != SECRET:
            return 200, {"code": 1000040346, "msg": "app_id is invalid", "data": {"URL": ""}}
        return 200, {"code": 0, "msg": "ok", "data": {
            "URL": self.ws.url + "/ws/v2?device_id=dev1&access_key=SECRETACCESSKEY&service_id=33&ticket=tick-secret",
            "ClientConfig": {"ReconnectCount": -1, "ReconnectInterval": 120, "ReconnectNonce": 30,
                             "PingInterval": self.ping_interval}}}

    def _chat(self, req):
        if not self._bearer_ok(req):
            return 400, {"code": 99991661}
        cid = req.path.rsplit("/", 1)[1]
        if cid == "oc_group1":
            return 200, {"code": 0, "data": {"name": "产品讨论群"}}
        return 400, {"code": 232011, "msg": "no permission"}

    def _user(self, req):
        oid = req.path.rsplit("/", 1)[1]
        names = {"ou_wangkai": "王凯", "ou_lina": "李娜"}
        if oid in names:
            return 200, {"code": 0, "data": {"user": {"name": names[oid], "open_id": oid}}}
        return 400, {"code": 41050, "msg": "no user authority"}

    def _reg(self, req):
        a = req.form.get("action")
        if a == "init":
            return 200, {"nonce": "n", "supported_auth_methods": ["client_secret", "private_key_jwt"]}
        if a == "begin":
            return 200, {"device_code": "DEV-CODE-123", "user_code": "ABCD-EFGH",
                         "verification_uri": "https://open.feishu.cn/page/launcher",
                         "verification_uri_complete": "https://open.feishu.cn/page/launcher?user_code=ABCD-EFGH&x=1",
                         "expires_in": 3600, "interval": 5}
        if a == "poll":
            if req.form.get("device_code") != "DEV-CODE-123":
                return 400, {"error": "invalid_grant", "code": 20079}
            if self.reg_state == "pending":
                return 400, {"error": "authorization_pending", "error_description": "", "code": 20094}
            if self.reg_state == "denied":
                return 400, {"error": "access_denied"}
            return 200, {"client_id": APP_ID, "client_secret": SECRET, "user_info": {"open_id": "ou_me"}}
        return 400, {"error": "unsupported"}

    async def _conn(self, ws):
        async def reader():
            async for raw in ws:
                f = fs.decode_frame(raw)
                hs = dict(f["headers"])
                if f["method"] == 0 and hs.get("type") == "ping":
                    self.pings += 1
                    await ws.send(fs.encode_frame({"service": 33, "method": 0, "headers": [("type", "pong")],
                                                   "payload": json.dumps({"PingInterval": 0.05}).encode()}))
                elif f["method"] == 1:
                    self.acks.append({"headers": hs, "payload": json.loads(f["payload"]), "seq_id": f["SeqID"]})
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
                await ws.send(frame)
        finally:
            rt.cancel()

    _seq = 0

    def event(self, ev: dict, *, split: int = 1, mid: str | None = None):
        FeishuFake._seq += 1
        data = json.dumps(ev, ensure_ascii=False).encode()
        mid = mid or f"msg-{FeishuFake._seq}"
        chunk = -(-len(data) // split)
        for i in range(split):
            self.out.put_nowait(fs.encode_frame({
                "SeqID": FeishuFake._seq, "LogID": 1, "service": 33, "method": 1,
                "headers": [("type", "event"), ("message_id", mid), ("sum", str(split)), ("seq", str(i)),
                            ("trace_id", "tr")],
                "payload": data[i * chunk:(i + 1) * chunk]}))


def receive(message_id, *, chat_type="p2p", text="帮我看下支付页", sender="ou_lina", mentions=(),
            message_type="text", content=None, chat_id="oc_group1"):
    return {"schema": "2.0",
            "header": {"event_id": f"ev-{message_id}", "event_type": "im.message.receive_v1",
                       "create_time": "1763280000000", "app_id": APP_ID, "tenant_key": "tk"},
            "event": {"sender": {"sender_id": {"open_id": sender, "user_id": "u", "union_id": "on"},
                                 "sender_type": "user", "tenant_key": "tk"},
                      "message": {"message_id": message_id, "create_time": "1763280000123",
                                  "chat_id": chat_id, "chat_type": chat_type, "message_type": message_type,
                                  "content": content if content is not None else json.dumps({"text": text}, ensure_ascii=False),
                                  "mentions": [{"key": k, "id": {"open_id": oid}, "name": n, "tenant_key": "tk"}
                                               for k, oid, n in mentions]}}}


@pytest.fixture
async def fake():
    f = await FeishuFake().start()
    yield f
    await f.stop()


@pytest.fixture
def adapter(fake):
    a = fs.FeishuAdapter()
    a.api_base = fake.http.url
    return a


CFG = {"appid": APP_ID}
SECRETS = {"appsec": SECRET}


# ── verify ─────────────────────────────────────────────────────────────────

async def test_verify_labels_with_the_app_name(adapter):
    v = await adapter.verify(verify_ctx(SECRETS, CFG))
    assert v.label == "Mira 的 Orbital" and v.config == {}


@pytest.mark.parametrize("cfg,sec,field,reason", [
    ({"appid": "cli_wrong"}, SECRETS, "appid", "rejected"),
    (CFG, {"appsec": "wrong"}, "appsec", "rejected"),
    ({"appid": ""}, SECRETS, "appid", "invalid"),
    (CFG, {"appsec": ""}, "appsec", "invalid"),
])
async def test_verify_marks_the_field(adapter, cfg, sec, field, reason):
    with pytest.raises(VerifyError) as e:
        await adapter.verify(verify_ctx(sec, cfg))
    assert (e.value.field, e.value.reason) == (field, reason)


async def test_a_lark_app_is_found_on_the_lark_host(fake):
    nowhere = FakeHTTP().start()
    nowhere.route("POST", "/open-apis/auth/v3/tenant_access_token/internal",
                  lambda r: (200, {"code": 10003, "data": {}, "msg": "invalid param"}))
    try:
        a = fs.FeishuAdapter()
        a.api_base = {"feishu": nowhere.url, "lark": fake.http.url}
        v = await a.verify(verify_ctx(SECRETS, CFG))
        assert v.config == {"domain": "lark"}
    finally:
        nowhere.stop()


# ── arrivals ───────────────────────────────────────────────────────────────

async def test_dm_and_group_mention_become_items_and_each_frame_is_acked(adapter, fake):
    run = Run(adapter, secrets=SECRETS, config=CFG).start()
    await run.until(lambda: run.connects == 1)
    fake.event(receive("om_1"))
    fake.event(receive("om_2", chat_type="group", sender="ou_wangkai", text="@_user_1 把这周的用户反馈整理成一页",
                       mentions=[("@_user_1", BOT_OPEN_ID, "Mira 的 Orbital")]))
    await run.until(lambda: len(run.items) == 2)
    a, b = run.items
    assert (a.kind, a.id, a.sender, a.place, a.title) == ("dm", "om_1", "李娜", {"kind": "dm"}, "帮我看下支付页")
    assert a.at == "2025-11-16T08:00:00Z"
    assert (b.kind, b.sender, b.place, b.title) == ("at", "王凯", {"kind": "group", "name": "产品讨论群"},
                                                    "把这周的用户反馈整理成一页")
    await run.until(lambda: len(fake.acks) == 2)
    ack = fake.acks[0]
    assert ack["payload"] == {"code": 200} and "biz_rt" in ack["headers"] and ack["headers"]["type"] == "event"
    await run.cancel()


async def test_other_mentions_own_messages_and_app_senders_are_ignored(adapter, fake):
    run = Run(adapter, secrets=SECRETS, config=CFG).start()
    await run.until(lambda: run.connects == 1)
    fake.event(receive("om_x1", chat_type="group", text="@_user_1 看一下", mentions=[("@_user_1", "ou_someone", "Kai")]))
    fake.event(receive("om_x2", sender=BOT_OPEN_ID))
    app_sender = receive("om_x3")
    app_sender["event"]["sender"]["sender_type"] = "app"
    fake.event(app_sender)
    other = receive("om_x4")
    other["header"]["event_type"] = "im.chat.member.bot.added_v1"
    fake.event(other)
    fake.event(receive("om_x5", text="real"))
    await run.until(lambda: len(run.items) == 1)
    await asyncio.sleep(0.1)
    assert [i.id for i in run.items] == ["om_x5"]
    assert len(fake.acks) == 5  # every data frame is acknowledged
    await run.cancel()


async def test_non_text_rich_text_split_payloads_and_duplicates(adapter, fake):
    run = Run(adapter, secrets=SECRETS, config=CFG).start()
    await run.until(lambda: run.connects == 1)
    fake.event(receive("om_img", message_type="image", content=json.dumps({"image_key": "img_1"})))
    post = {"title": "周报", "content": [[{"tag": "text", "text": "第一段 "}, {"tag": "at", "user_id": BOT_OPEN_ID}],
                                       [{"tag": "img", "image_key": "k"}, {"tag": "text", "text": "截图"}]]}
    fake.event(receive("om_post", message_type="post", content=json.dumps(post, ensure_ascii=False),
                       sender="ou_unknown"), split=3)
    fake.event(receive("om_dup", text="once"))
    fake.event(receive("om_dup", text="once"))
    await run.until(lambda: len(run.items) == 3)
    await asyncio.sleep(0.1)
    img, p, dup = run.items
    assert img.title == "[image]"
    assert p.title == "周报" and "第一段" in p.body and "[image]截图" in p.body and p.sender is None
    assert dup.id == "om_dup"
    await run.cancel()


async def test_unknown_group_names_fall_back(adapter, fake):
    run = Run(adapter, secrets=SECRETS, config=CFG).start()
    await run.until(lambda: run.connects == 1)
    fake.event(receive("om_g", chat_type="group", chat_id="oc_secret", mentions=[("@_user_1", BOT_OPEN_ID, "bot")],
                       text="@_user_1 hi"))
    await run.until(lambda: len(run.items) == 1)
    assert run.items[0].place == {"kind": "group", "name": None} and run.items[0].title == "hi"
    await run.cancel()


async def test_pings_follow_the_interval_the_server_sets(adapter, fake):
    fake.ping_interval = 0.05
    run = Run(adapter, secrets=SECRETS, config=CFG).start()
    await run.until(lambda: fake.pings >= 3)
    await run.cancel()


# ── failures ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("code,expect,retry", [(1000040346, "token_rejected", False),
                                               (1000040350, "bot_in_use", True)])
async def test_endpoint_refusals(adapter, fake, code, expect, retry):
    fake.endpoint_code = code
    err = await Run(adapter, secrets=SECRETS, config=CFG).start().result()
    assert isinstance(err, SourceFailure) and err.code == expect and err.retryable is retry


async def test_a_changed_secret_is_non_retryable(adapter, fake):
    err = await Run(adapter, secrets={"appsec": "rotated"}, config=CFG).start().result()
    assert isinstance(err, SourceFailure) and err.code == "token_rejected" and not err.retryable


async def test_a_dropped_connection_is_unreachable(adapter, fake):
    run = Run(adapter, secrets=SECRETS, config=CFG).start()
    await run.until(lambda: run.connects == 1)
    fake.out.put_nowait("DROP")
    assert isinstance(await run.result(), ConnectionError)


async def test_secrets_never_reach_a_log_line(adapter, fake, caplog):
    caplog.set_level(logging.DEBUG)
    run = Run(adapter, secrets=SECRETS, config=CFG).start()
    await run.until(lambda: run.connects == 1)
    fake.event(receive("om_log"))
    await run.until(lambda: len(run.items) == 1)
    await run.cancel()
    ours = "\n".join(r.getMessage() for r in caplog.records if not r.name.startswith("websockets.server"))
    assert "/ws/v2?" in ours
    for s in (SECRET, "SECRETACCESSKEY", "tick-secret", TENANT):
        assert s not in ours


# ── QR: scan to create ─────────────────────────────────────────────────────

async def test_qr_flow_pending_then_done(adapter, fake):
    started = await adapter.qr_start(verify_ctx({}))
    assert started["qr_text"] == "https://open.feishu.cn/page/cli?user_code=ABCD-EFGH&x=1"
    assert started["expires_at"].endswith("Z")
    h = started["handle"]
    assert (await adapter.qr_poll(verify_ctx({}), h)) == {"state": "pending"}
    fake.reg_state = "done"
    done = await adapter.qr_poll(verify_ctx({}), h)
    assert done == {"state": "done", "label": "Mira 的 Orbital", "config": {"appid": APP_ID},
                    "secrets": {"appsec": SECRET}}


async def test_qr_denied_and_expired(adapter, fake):
    h = (await adapter.qr_start(verify_ctx({})))["handle"]
    fake.reg_state = "denied"
    assert (await adapter.qr_poll(verify_ctx({}), h))["state"] == "failed"
    assert (await adapter.qr_poll(verify_ctx({}), {**h, "deadline": time.time() - 1}))["state"] == "expired"


async def test_qr_through_the_core_creates_a_source_without_the_browser_seeing_the_secret(adapter, fake, tmp_path):
    secrets = MemorySecretStore()
    core = SourcesCore(str(tmp_path), registry=AdapterRegistry(overrides={"feishu": adapter}),
                       secret_store=secrets)
    core.load()
    core.registry._cache["feishu"] = (adapter, None)
    started = await core.qr_start("feishu")
    assert started["state"] == "pending" and "secret" not in json.dumps(started).lower()
    fake.reg_state = "done"
    got = await core.qr_get(started["qr_id"])
    assert got == {"qr_id": started["qr_id"], "state": "done", "label": "Mira 的 Orbital"}
    detail = await core.create({"type": "feishu", "kinds": {"dm": {}, "at": {}}, "qr_id": started["qr_id"]})
    assert detail["label"] == "Mira 的 Orbital" and detail["config"]["appid"] == APP_ID
    assert secrets.get(detail["id"], "appsec") == SECRET
    assert SECRET not in json.dumps(detail)
