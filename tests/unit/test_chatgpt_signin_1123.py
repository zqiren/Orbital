# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sign in with ChatGPT (spec 1123): the sign-in module, the token broker,
request shaping on the route, plan-usage errors, and the loop's no-fallback
rule. OpenAI is mocked everywhere (httpx.MockTransport + a test RSA key);
no real token appears in this file.
"""

from __future__ import annotations

import asyncio
import json
import time
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from agent_os.providers_auth import chatgpt
from agent_os.providers_auth.token_cards import (
    NeedsSignInError,
    TokenBroker,
    dump_bundle,
    parse_bundle,
)
from agent_os.agent.providers.types import LLMError, ErrorCategory, plan_usage_error
from agent_os.daemon_v2.settings_store import mask_key

ISSUED = "oaiapp_test_issued_client_0000000"

_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_JWK = json.loads(RSAAlgorithm.to_jwk(_KEY.public_key()))
_JWK.update({"kid": "k1", "alg": "RS256", "use": "sig"})


def _id_token(*, aud=ISSUED, nonce="", sub="user-1", email="person@example.com",
              key=_KEY, iss=chatgpt.ISSUER, exp_in=3600):
    now = int(time.time())
    return jwt.encode({"iss": iss, "aud": aud, "sub": sub, "email": email,
                       "nonce": nonce, "iat": now, "exp": now + exp_in},
                      key, algorithm="RS256", headers={"kid": "k1"})


class FakeOpenAI:
    """auth.openai.com token/JWKS/revoke + api.openai.com/v1/models."""

    def __init__(self, *, scope=chatgpt.SCOPE, id_aud=ISSUED, nonce_ok=True,
                 sign_key=_KEY):
        self.scope = scope
        self.id_aud = id_aud
        self.nonce_ok = nonce_ok
        self.sign_key = sign_key
        self.requests: list[tuple[str, dict]] = []
        self.nonce = ""
        self.refresh_status = 200
        self.refresh_body = None
        self.revoke_status = 200
        self.counter = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()} \
            if request.method == "POST" else {}
        self.requests.append((url, form))
        if url == chatgpt.JWKS_URL:
            return httpx.Response(200, json={"keys": [_JWK]})
        if url == chatgpt.TOKEN_URL and form.get("grant_type") == "authorization_code":
            nonce = self.nonce if self.nonce_ok else "other"
            return httpx.Response(200, json={
                "access_token": "at-1", "refresh_token": "rt-1",
                "id_token": _id_token(aud=self.id_aud, nonce=nonce, key=self.sign_key),
                "token_type": "Bearer", "expires_in": 3600, "scope": self.scope,
            })
        if url == chatgpt.TOKEN_URL and form.get("grant_type") == "refresh_token":
            if self.refresh_status != 200:
                return httpx.Response(self.refresh_status, json=self.refresh_body or {})
            self.counter += 1
            return httpx.Response(200, json={
                "access_token": f"at-r{self.counter}",
                "refresh_token": f"rt-r{self.counter}",
                "expires_in": 3600, "scope": chatgpt.SCOPE,
            })
        if url == chatgpt.REVOKE_URL:
            return httpx.Response(self.revoke_status)
        return httpx.Response(404)


def _browser(params_for_callback):
    """An ``open_browser`` that plays the user: it reads the authorize URL and
    calls the loopback callback with what ``params_for_callback`` returns."""
    seen = {}

    def open_browser(url: str):
        q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        seen.update(q)

        async def hit():
            extra = params_for_callback(q)
            async with httpx.AsyncClient() as c:
                await c.get(q["redirect_uri"], params=extra)
        asyncio.get_running_loop().create_task(hit())
    return open_browser, seen


def _ok_callback(q):
    return {"code": "the-code", "state": q["state"], "client_id": ISSUED,
            "scope": chatgpt.SCOPE, "iss": chatgpt.ISSUER}


async def _sign_in(fake: FakeOpenAI, callback=_ok_callback, **kw):
    open_browser, seen = _browser(lambda q: (setattr(fake, "nonce", q["nonce"]),
                                             callback(q))[1])
    async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as http:
        bundle = await chatgpt.sign_in(host_id="urn:uuid:host-1", http=http,
                                       open_browser=open_browser, timeout=10, **kw)
    return bundle, seen


# ---------------------------------------------------------------------------
# 1. Sign-in module
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_sign_in_registers_orbital_on_the_official_route():
    fake = FakeOpenAI()
    bundle, seen = await _sign_in(fake)
    assert seen["client_id"] == "dynamic_agent_client"
    assert seen["agent_name_hint"] == "Orbital"
    assert seen["ext_agent_host_id"] == "urn:uuid:host-1"
    assert seen["resource"] == "https://api.openai.com/v1"
    assert seen["code_challenge_method"] == "S256" and seen["nonce"] and seen["state"]
    assert set(seen["scope"].split()) >= {"openid", "offline_access",
                                          "chatgpt.tokens.use.direct"}
    redirect = urlparse(seen["redirect_uri"])
    assert redirect.hostname == "127.0.0.1" and redirect.path == "/auth/callback"
    # Never the legacy Codex route.
    assert "1455" not in seen["redirect_uri"]
    assert all("chatgpt.com" not in u for u, _ in fake.requests)
    exchange = next(f for u, f in fake.requests if f.get("grant_type") == "authorization_code")
    assert exchange["client_id"] == ISSUED  # the issued one, not dynamic_agent_client
    assert exchange["resource"] == chatgpt.RESOURCE
    assert exchange["redirect_uri"] == seen["redirect_uri"]
    assert bundle["client_id"] == ISSUED and bundle["host_id"] == "urn:uuid:host-1"
    assert bundle["access_token"] == "at-1" and bundle["refresh_token"] == "rt-1"
    assert bundle["email"] == "person@example.com" and bundle["sub"] == "user-1"
    assert bundle["expires_at"] > time.time() + 3000


@pytest.mark.asyncio
async def test_reauth_uses_the_saved_client_id_and_hint_without_name_hint():
    fake = FakeOpenAI()
    _bundle, seen = await _sign_in(fake, client_id=ISSUED, id_token_hint="old-idt",
                                   expected_sub="user-1")
    assert seen["client_id"] == ISSUED
    assert seen["id_token_hint"] == "old-idt"
    assert "agent_name_hint" not in seen


@pytest.mark.asyncio
async def test_reauth_refuses_a_different_account():
    fake = FakeOpenAI()
    with pytest.raises(chatgpt.ChatGPTAuthError) as e:
        await _sign_in(fake, client_id=ISSUED, expected_sub="someone-else")
    assert e.value.status == 409


@pytest.mark.asyncio
async def test_missing_plan_usage_scope_is_refused_clearly():
    fake = FakeOpenAI(scope="openid profile email offline_access")
    with pytest.raises(chatgpt.ChatGPTAuthError) as e:
        await _sign_in(fake, callback=lambda q: {**_ok_callback(q),
                                                 "scope": "openid email"})
    assert e.value.code == "plan_usage_not_granted"
    assert "plan" in e.value.message.lower()


@pytest.mark.parametrize("fake_kw,what", [
    ({"id_aud": "someone-elses-client"}, "audience"),
    ({"nonce_ok": False}, "nonce"),
    ({"sign_key": rsa.generate_private_key(public_exponent=65537, key_size=2048)},
     "signature"),
])
@pytest.mark.asyncio
async def test_id_token_is_verified(fake_kw, what):
    with pytest.raises(chatgpt.ChatGPTAuthError) as e:
        await _sign_in(FakeOpenAI(**fake_kw))
    assert e.value.code == "signin_failed", what


@pytest.mark.asyncio
async def test_access_denied_stops_without_exchange():
    fake = FakeOpenAI()
    with pytest.raises(chatgpt.ChatGPTAuthError) as e:
        await _sign_in(fake, callback=lambda q: {"error": "access_denied",
                                                 "state": q["state"]})
    assert e.value.code == "signin_denied"
    assert not any(f.get("grant_type") for _, f in fake.requests)


@pytest.mark.asyncio
async def test_state_mismatch_is_refused():
    fake = FakeOpenAI()
    with pytest.raises(chatgpt.ChatGPTAuthError) as e:
        await _sign_in(fake, callback=lambda q: {**_ok_callback(q), "state": "forged"})
    assert e.value.code == "signin_failed"
    assert not any(f.get("grant_type") for _, f in fake.requests)


@pytest.mark.asyncio
async def test_cancel_and_timeout():
    cancel = asyncio.Event()
    asyncio.get_running_loop().call_later(0.2, cancel.set)
    with pytest.raises(chatgpt.ChatGPTAuthError) as e:
        await chatgpt.sign_in(host_id="h", open_browser=lambda u: None,
                              timeout=10, cancel=cancel)
    assert e.value.code == "signin_cancelled"
    with pytest.raises(chatgpt.ChatGPTAuthError) as e:
        await chatgpt.sign_in(host_id="h", open_browser=lambda u: None, timeout=0.2)
    assert e.value.code == "signin_timeout" and e.value.status == 504


@pytest.mark.asyncio
async def test_refresh_rotates_and_sends_the_documented_fields():
    fake = FakeOpenAI()
    old = {"provider": "chatgpt", "client_id": ISSUED, "refresh_token": "rt-1",
           "access_token": "at-1", "expires_at": 0, "email": "p@example.com"}
    async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as http:
        new = await chatgpt.refresh_tokens(old, http=http)
    form = fake.requests[-1][1]
    assert form == {"grant_type": "refresh_token", "client_id": ISSUED,
                    "refresh_token": "rt-1", "resource": chatgpt.RESOURCE}
    assert new["refresh_token"] == "rt-r1" and new["access_token"] == "at-r1"
    assert new["expires_at"] > time.time() + 3000
    assert new["email"] == "p@example.com"


@pytest.mark.parametrize("status,body,terminal", [
    (400, {"error": "invalid_grant"}, True),
    (401, {"error": {"code": "refresh_token_expired"}}, True),
    (400, {"error": "refresh_token_reused"}, True),
    (503, {}, False),
    (429, {}, False),
])
@pytest.mark.asyncio
async def test_refresh_errors_terminal_or_not(status, body, terminal):
    fake = FakeOpenAI()
    fake.refresh_status, fake.refresh_body = status, body
    async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as http:
        with pytest.raises(chatgpt.ChatGPTAuthError) as e:
            await chatgpt.refresh_tokens({"client_id": ISSUED, "refresh_token": "x"},
                                         http=http)
    assert e.value.terminal is terminal


@pytest.mark.asyncio
async def test_revoke_sends_refresh_token_with_hint_and_client():
    fake = FakeOpenAI()
    async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)) as http:
        await chatgpt.revoke({"client_id": ISSUED, "refresh_token": "rt-9"}, http=http)
        url, form = fake.requests[-1]
        assert url == "https://auth.openai.com/api/accounts/oauth/revoke"
        assert form == {"token": "rt-9", "token_type_hint": "refresh_token",
                        "client_id": ISSUED}
        fake.revoke_status = 503
        with pytest.raises(chatgpt.ChatGPTAuthError):
            await chatgpt.revoke({"client_id": ISSUED, "refresh_token": "rt-9"}, http=http)


def test_model_list_keeps_listed_models_in_server_order():
    data = {"models": [
        {"slug": "gpt-6.2-sol", "display_name": "GPT-6.2 Sol", "visibility": "list",
         "context_window": 500000},
        {"slug": "gpt-reserve", "visibility": "hide"},
        {"slug": "gpt-6-luna", "display_name": "GPT-6 Luna", "visibility": "list",
         "context_window": 272000, "supported_in_api": True},
        {"slug": "codex-auto-review", "visibility": "hidden"},
        {"slug": "no-api", "visibility": "list", "supported_in_api": False},
    ]}
    out = chatgpt.parse_model_list(data)
    assert [m["id"] for m in out] == ["gpt-6.2-sol", "gpt-6-luna"]
    assert out[0] == {"id": "gpt-6.2-sol", "display_name": "GPT-6.2 Sol",
                      "context_window": 500000}


def test_host_id_is_stable_per_install(tmp_path):
    a = chatgpt.load_or_create_host_id(str(tmp_path))
    b = chatgpt.load_or_create_host_id(str(tmp_path))
    assert a == b and a.startswith("urn:uuid:")


# ---------------------------------------------------------------------------
# 2. Token bundle + broker
# ---------------------------------------------------------------------------

def _bundle(**kw):
    b = {"provider": "chatgpt", "client_id": ISSUED, "host_id": "h", "sub": "u",
         "email": "person@example.com", "access_token": "at-0",
         "refresh_token": "rt-0", "id_token": "idt", "expires_at": time.time() + 3600,
         "scopes": chatgpt.SCOPE.split()}
    b.update(kw)
    return b


def test_bundle_format_masks_to_nothing_secret_on_an_older_build():
    raw = dump_bundle(_bundle(access_token="SECRET-ACCESS", refresh_token="SECRET-REFRESH"))
    assert raw.startswith('{"kind": "oauth-token-set"') and raw.endswith('"v": 1}')
    masked = mask_key(raw)  # what an older build shows for this card
    assert "SECRET" not in masked
    assert parse_bundle(raw)["access_token"] == "SECRET-ACCESS"
    assert parse_bundle("sk-plain-key") is None
    assert parse_bundle('{"not": "a bundle"}') is None


class MemStore:
    def __init__(self, raw):
        self.raw = {"card_c": raw}
        self.health = []
        self.writes = []

    def secret_for(self, cid):
        return self.raw.get(cid, "")

    def set_card_secret(self, cid, raw):
        self.writes.append(raw)
        self.raw[cid] = raw

    def record_card_health(self, cid, *, verified=False, error=None):
        self.health.append((cid, error))


@pytest.mark.asyncio
async def test_broker_returns_fresh_token_without_network():
    calls = []

    async def refresher(b):
        calls.append(b)
        raise AssertionError("must not refresh")
    store = MemStore(dump_bundle(_bundle()))
    assert await TokenBroker(store, refresher=refresher).access_token("card_c") == "at-0"
    assert calls == []


@pytest.mark.asyncio
async def test_broker_refreshes_near_expiry_and_persists_before_use():
    store = MemStore(dump_bundle(_bundle(expires_at=time.time() + 60)))

    async def refresher(b):
        assert b["refresh_token"] == "rt-0"
        return {**b, "access_token": "at-1", "refresh_token": "rt-1",
                "expires_at": time.time() + 3600}
    token = await TokenBroker(store, refresher=refresher).access_token("card_c")
    assert token == "at-1"
    stored = parse_bundle(store.raw["card_c"])
    assert stored["refresh_token"] == "rt-1" and stored["access_token"] == "at-1"


@pytest.mark.asyncio
async def test_broker_does_not_hand_out_a_token_it_could_not_persist():
    store = MemStore(dump_bundle(_bundle(expires_at=0)))

    def boom(cid, raw):
        raise RuntimeError("keychain write failed")
    store.set_card_secret = boom

    async def refresher(b):
        return {**b, "access_token": "at-1", "refresh_token": "rt-1",
                "expires_at": time.time() + 3600}
    with pytest.raises(RuntimeError):
        await TokenBroker(store, refresher=refresher).access_token("card_c")


@pytest.mark.asyncio
async def test_concurrent_401s_refresh_once_rotating_token_never_races():
    store = MemStore(dump_bundle(_bundle()))
    calls = []

    async def refresher(b):
        calls.append(b["refresh_token"])
        await asyncio.sleep(0.05)
        return {**b, "access_token": "at-1", "refresh_token": "rt-1",
                "expires_at": time.time() + 3600}
    broker = TokenBroker(store, refresher=refresher)
    tokens = await asyncio.gather(*[
        broker.access_token("card_c", force=True, failed_token="at-0")
        for _ in range(5)])
    assert tokens == ["at-1"] * 5
    assert calls == ["rt-0"]


@pytest.mark.asyncio
async def test_terminal_refresh_marks_needs_signin_once_and_stops_trying():
    store = MemStore(dump_bundle(_bundle(expires_at=0)))
    calls = []

    async def refresher(b):
        calls.append(1)
        raise chatgpt.ChatGPTAuthError("needs_signin", chatgpt.NEEDS_SIGNIN_MESSAGE,
                                       status=401, terminal=True)
    broker = TokenBroker(store, refresher=refresher)
    for _ in range(3):
        with pytest.raises(NeedsSignInError) as e:
            await broker.access_token("card_c")
        assert e.value.final and e.value.code == "needs_signin"
        assert e.value.category == ErrorCategory.ABORT
    assert calls == [1]                     # no loop
    assert len(store.health) == 1
    assert store.health[0][1]["code"] == "needs_signin"
    # A fresh sign-in (new refresh token) is tried again.
    store.raw["card_c"] = dump_bundle(_bundle(expires_at=0, refresh_token="rt-new"))
    with pytest.raises(NeedsSignInError):
        await broker.access_token("card_c")
    assert calls == [1, 1]


@pytest.mark.asyncio
async def test_transient_refresh_failure_is_not_needs_signin():
    store = MemStore(dump_bundle(_bundle(expires_at=0)))

    async def refresher(b):
        raise chatgpt.ChatGPTAuthError("refresh_unreachable", "offline", terminal=False)
    with pytest.raises(LLMError) as e:
        await TokenBroker(store, refresher=refresher).access_token("card_c")
    assert not isinstance(e.value, NeedsSignInError) and not e.value.final
    assert store.health == []


# ---------------------------------------------------------------------------
# 3. Request shaping on the ChatGPT route (regressions from the spike)
# ---------------------------------------------------------------------------

from unittest.mock import AsyncMock  # noqa: E402

import openai  # noqa: E402

from agent_os.agent.providers.openai_compat import LLMProvider  # noqa: E402
from agent_os.config.provider_registry import ProviderRegistry  # noqa: E402

ROUTE_FORBIDDEN = {"temperature", "max_output_tokens", "top_p", "metadata",
                   "previous_response_id", "user", "background", "truncation"}


def _status_error(status, body):
    req = httpx.Request("POST", "https://api.openai.com/v1/responses")
    resp = httpx.Response(status, json=body, request=req)
    cls = {400: openai.BadRequestError, 401: openai.AuthenticationError,
           403: openai.PermissionDeniedError, 429: openai.RateLimitError}.get(
        status, openai.APIStatusError)
    return cls(f"Error code: {status} - {body}", response=resp,
               body=body.get("error", body))


def _events(text="PONG"):
    from openai.types.responses import Response
    resp = Response.model_validate({
        "id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-6.2-sol",
        "status": "completed", "output": [{
            "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}],
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
                  "input_tokens_details": {"cached_tokens": 8},
                  "output_tokens_details": {"reasoning_tokens": 0}},
    })
    return [
        {"type": "response.output_text.delta", "delta": text, "item_id": "msg_1",
         "output_index": 0, "content_index": 0},
        {"type": "response.completed", "response": resp},
    ]


class _Stream:
    def __init__(self, events):
        self._events = events

    def __aiter__(self):
        async def gen():
            for e in self._events:
                yield _obj(e)
        return gen()


def _obj(d):
    if isinstance(d, dict):
        class O(dict):
            __getattr__ = dict.get
        return O({k: _obj(v) if isinstance(v, dict) else v for k, v in d.items()})
    return d


def _route(sent):
    """A fake of the ChatGPT route's admission rules (spike, 2026-11-23)."""
    async def create(**body):
        sent.append(body)
        if body.get("stream") is not True:
            raise _status_error(400, {"detail": "Stream must be set to true"})
        if any(i.get("role") == "system" for i in body["input"]):
            raise _status_error(400, {"detail": "System messages are not allowed"})
        if ROUTE_FORBIDDEN & set(body):
            raise _status_error(400, {"detail": "Unsupported parameter"})
        return _Stream(_events())
    return create


def _chatgpt_provider(**kw):
    data = ProviderRegistry().get_provider_data("chatgpt")
    info = ProviderRegistry().get_model_info("chatgpt", "gpt-6.2-sol")
    p = LLMProvider("gpt-6.2-sol", "at-0", data["base_url"], sdk=data["sdk"],
                    reasoning=info.reasoning, provider="chatgpt",
                    session_header=data["session_header"],
                    system_role=kw.pop("system_role", data["system_role"]),
                    stream_only=kw.pop("stream_only", data["stream_only"]), **kw)
    return p


MSGS = [{"role": "system", "content": "You are Orbital."},
        {"role": "user", "content": "Reply with PONG"}]


@pytest.mark.asyncio
async def test_regression_system_rows_go_out_as_developer_messages():
    sent = []
    p = _chatgpt_provider()
    p._openai_client.responses.create = AsyncMock(side_effect=_route(sent))
    chunks = [c async for c in p.stream(MSGS)]
    assert any(c.text == "PONG" for c in chunks)
    assert [i.get("role") for i in sent[-1]["input"]] == ["developer", "user"]
    # Without the registry flag the route refuses (the spike's 400).
    sent2 = []
    q = _chatgpt_provider(system_role="system")
    q._openai_client.responses.create = AsyncMock(side_effect=_route(sent2))
    with pytest.raises(LLMError) as e:
        [c async for c in q.stream(MSGS)]
    assert e.value.status_code == 400
    assert "System messages are not allowed" in e.value.message


@pytest.mark.asyncio
async def test_regression_complete_streams_on_a_stream_only_route():
    sent = []
    p = _chatgpt_provider()
    p._openai_client.responses.create = AsyncMock(side_effect=_route(sent))
    out = await p.complete(MSGS)
    assert out.text == "PONG" and out.usage.cache_read_tokens == 8
    assert sent[-1]["stream"] is True and sent[-1]["store"] is False
    # Without the flag, complete() sends stream:false and the route refuses.
    q = _chatgpt_provider(stream_only=False)
    q._openai_client.responses.create = AsyncMock(side_effect=_route([]))
    with pytest.raises(LLMError) as e:
        await q.complete(MSGS)
    assert e.value.status_code == 400 and "Stream must be set to true" in e.value.message


@pytest.mark.asyncio
async def test_route_body_never_carries_refused_fields_and_sends_session_id():
    sent = []
    p = _chatgpt_provider()
    p.bind_session_id(lambda: "proj-sess-1")
    create = AsyncMock(side_effect=_route(sent))
    p._openai_client.responses.create = create
    tools = [{"type": "function", "function": {"name": "read", "parameters": {
        "type": "object", "properties": {}}}}]
    [c async for c in p.stream(MSGS, tools=tools)]
    body = sent[-1]
    assert not (ROUTE_FORBIDDEN & set(body))
    assert {t["type"] for t in body["tools"]} == {"function"}
    assert body["store"] is False and body["stream"] is True
    assert create.call_args.kwargs["extra_headers"] == {"session-id": "proj-sess-1"}


@pytest.mark.asyncio
async def test_token_source_refreshes_before_the_call_and_once_on_401():
    asked = []

    async def source(force=False, failed_token=None):
        asked.append((force, failed_token))
        return "at-2" if force else "at-1"
    p = _chatgpt_provider(token_source=source)
    seen_keys = []
    calls = {"n": 0}

    async def create(**body):
        seen_keys.append(p.api_key)
        calls["n"] += 1
        if calls["n"] == 1:
            raise _status_error(401, {"detail": "token expired"})
        return _Stream(_events())
    p._openai_client.responses.create = AsyncMock(side_effect=create)
    p.update_api_key = lambda k: setattr(p, "api_key", k)  # keep the mock client
    chunks = [c async for c in p.stream(MSGS)]
    assert any(c.text == "PONG" for c in chunks)
    assert asked == [(False, None), (True, "at-1")]
    assert seen_keys == ["at-1", "at-2"]

    # A second 401 is not retried again.
    async def always401(**body):
        raise _status_error(401, {"detail": "nope"})
    p._openai_client.responses.create = AsyncMock(side_effect=always401)
    with pytest.raises(LLMError) as e:
        [c async for c in p.stream(MSGS)]
    assert e.value.status_code == 401
    assert p._openai_client.responses.create.call_count == 2


@pytest.mark.asyncio
async def test_needs_signin_from_the_token_source_is_final():
    async def source(force=False, failed_token=None):
        raise NeedsSignInError(chatgpt.NEEDS_SIGNIN_MESSAGE)
    p = _chatgpt_provider(token_source=source)
    p._openai_client.responses.create = AsyncMock(side_effect=AssertionError)
    with pytest.raises(LLMError) as e:
        await p.complete(MSGS)
    assert e.value.final and e.value.code == "needs_signin"


# ---------------------------------------------------------------------------
# 4. Plan-usage errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status,code,ours,final", [
    (429, "subscription_sharing_usage_limit_exceeded", "usage_limit_reached", True),
    (403, "subscription_sharing_user_not_eligible", "not_eligible", True),
    (401, "subscription_sharing_invalid_user", "needs_signin", True),
    (400, "subscription_sharing_unsupported_capability", "provider_error", True),
    (503, "subscription_sharing_usage_unavailable", "provider_error", False),
])
@pytest.mark.asyncio
async def test_plan_usage_errors_map_to_clear_final_reasons(status, code, ours, final):
    p = _chatgpt_provider()

    async def create(**body):
        raise _status_error(status, {"error": {"code": code, "message": "x",
                                               "param": "tools"}})
    p._openai_client.responses.create = AsyncMock(side_effect=create)
    with pytest.raises(LLMError) as e:
        await p.complete(MSGS)
    assert e.value.code == ours and e.value.final is final
    assert e.value.status_code == status
    if final:
        assert e.value.category == ErrorCategory.ABORT
    else:
        assert e.value.category == ErrorCategory.RETRY  # same card, never rotate
    assert "subscription_sharing" not in e.value.message  # a human sentence


def test_in_stream_failure_with_a_plan_code_is_final():
    from agent_os.agent.providers.responses_adapter import raise_response_error
    with pytest.raises(LLMError) as e:
        raise_response_error("subscription_sharing_usage_limit_exceeded", "limit")
    assert e.value.final and e.value.code == "usage_limit_reached"


def test_detail_shaped_refusal_is_unwrapped():
    from agent_os.agent.providers.types import unwrap_provider_message
    assert unwrap_provider_message(
        "Error code: 400 - {'detail': 'System messages are not allowed'}"
    ) == "System messages are not allowed"


def test_unknown_codes_are_untouched():
    assert plan_usage_error("rate_limit_exceeded") is None
    err = LLMError("x", status_code=429)
    assert err.category == ErrorCategory.TRANSIENT and not err.final


# ---------------------------------------------------------------------------
# 5. Loop: a final refusal never falls back to another card
# ---------------------------------------------------------------------------

from tests.unit.test_loop_auth_fallback import (  # noqa: E402
    AlwaysErrorProvider, SuccessProvider, _make_loop, _system_texts,
)
from agent_os.agent.session import Session, persist_user_row  # noqa: E402


@pytest.mark.parametrize("err", [
    NeedsSignInError(chatgpt.NEEDS_SIGNIN_MESSAGE),
    plan_usage_error("subscription_sharing_usage_limit_exceeded"),
    plan_usage_error("subscription_sharing_user_not_eligible"),
])
@pytest.mark.asyncio
async def test_final_errors_stop_the_run_without_any_fallback(tmp_path, err):
    session = Session.new("final", str(tmp_path))
    primary = AlwaysErrorProvider("gpt-6.2-sol", err, api_key="at-x")
    auth = SuccessProvider("global-model", api_key="sk-global", provider="openai")
    fb = SuccessProvider("fb-model", api_key="sk-fb", provider="openai")
    health = []
    loop = _make_loop(tmp_path, session, primary, fallbacks=[fb], auth=auth)
    loop._on_credential_error = lambda *a: health.append(a)
    persist_user_row(session, "hello")
    await loop.run()
    assert loop._llm_failed
    assert auth._call_count == 0 and fb._call_count == 0
    assert primary._call_count == 1
    texts = " ".join(_system_texts(session))
    assert err.message.rstrip(".") in texts and ".." not in texts
    if err.status_code in (401, 403):
        assert health and health[0][2] == err.code
