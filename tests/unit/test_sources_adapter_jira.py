# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Jira source adapter against a local HTTP server standing in for Jira Cloud.

Response shapes: Atlassian's REST v3 reference
(https://developer.atlassian.com/cloud/jira/platform/rest/v3/api-group-myself/,
…/api-group-issue-search/#api-rest-api-3-search-jql-post — the enhanced search
answers ``{"issues": [...], "nextPageToken"?, "isLast"}`` and returns only
the fields asked for; …/api-group-issue-comments/), ADF mention nodes
(https://developer.atlassian.com/cloud/jira/platform/apis/document/nodes/mention/).
The 403 ``X-Seraph-LoginReason: AUTHENTICATION_DENIED`` (CAPTCHA) header is
from Atlassian's basic-auth docs. Not checked against a real site.
"""

from __future__ import annotations

import base64
import http.server
import json
import threading

import pytest

from agent_os.sources.adapters.jira import JiraAdapter, adf_text, mentions, normalise_site
from agent_os.sources.context import SourceContext
from agent_os.sources.model import SourceFailure, VerifyError

ME = "5b10ac8d82e05b22cc7d4ef5"
TOKEN = "ATATT3x-secret-token"


def _adf(*paras):
    return {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": list(p)} for p in paras]}


def _t(s):
    return {"type": "text", "text": s}


def _m(account_id, text):
    return {"type": "mention", "attrs": {"id": account_id, "text": text}}


class FakeJira:
    def __init__(self):
        self.status = {}          # path → forced status
        self.headers = {}
        self.assigned = []
        self.mention_candidates = []
        self.comments = {}        # key → full comment list (served by /comment)
        self.reject_mention_jql = False
        self.requests = []
        jira = self

        class H(http.server.BaseHTTPRequestHandler):
            def _auth_ok(self):
                want = "Basic " + base64.b64encode(f"mira@acme.com:{TOKEN}".encode()).decode()
                return self.headers.get("Authorization") == want

            def _send(self, status, body=None, headers=None):
                data = json.dumps(body).encode() if body is not None else b""
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _handle(self, method):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"null") if n else None
                path = self.path.split("?")[0]
                jira.requests.append((method, self.path, body))
                if path in jira.status:
                    return self._send(jira.status[path], {"errorMessages": ["x"]}, jira.headers.get(path))
                if not self._auth_ok():
                    return self._send(401, {"errorMessages": ["Unauthorized"]})
                if method == "GET" and path == "/rest/api/3/myself":
                    return self._send(200, {"accountId": ME, "emailAddress": "mira@acme.com",
                                            "displayName": "Mira Chen", "active": True})
                if method == "POST" and path == "/rest/api/3/search/jql":
                    jql = body["jql"]
                    if jql.startswith("assignee = currentUser()"):
                        return self._send(200, {"issues": jira.assigned, "isLast": True})
                    if "comment ~ currentUser()" in jql:
                        if jira.reject_mention_jql:
                            return self._send(400, {"errorMessages": ["The operator '~' is not supported"]})
                        return self._send(200, {"issues": jira.mention_candidates, "isLast": True})
                    if jql.startswith("updated >= -1d"):
                        return self._send(200, {"issues": jira.mention_candidates, "isLast": True})
                if method == "GET" and path.startswith("/rest/api/3/issue/") and path.endswith("/comment"):
                    key = path.split("/")[5]
                    cs = jira.comments.get(key, [])
                    return self._send(200, {"comments": cs, "total": len(cs), "maxResults": 50, "startAt": 0})
                return self._send(404, {"errorMessages": ["no"]})

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

            def log_message(self, *a):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.site = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def jira():
    j = FakeJira()
    yield j
    j.close()


def _issue(key, summary, *, desc=None, comments=None, total=None, status="In Progress"):
    f = {"summary": summary, "status": {"name": status}, "updated": "2026-11-16T07:20:00.000+0000",
         "reporter": {"accountId": "x", "displayName": "Li Na"}, "description": desc}
    if comments is not None:
        f["comment"] = {"comments": comments, "total": total if total is not None else len(comments),
                        "maxResults": len(comments), "startAt": 0}
    return {"id": "100" + key[-1], "key": key, "fields": f}


def _comment(cid, body, author_id="other", name="Kai Wu"):
    return {"id": cid, "author": {"accountId": author_id, "displayName": name}, "body": body,
            "created": "2026-11-16T07:30:00.000+0000", "updated": "2026-11-16T07:30:00.000+0000"}


def _ctx(site, *, mail="mira@acme.com", token=TOKEN, kinds=None, state=None):
    secrets = {"token": token}
    return SourceContext(source_id="src_j", type_="jira", config={"site": site, "mail": mail},
                         kinds=kinds or {"assigned": {}}, account=None, secrets_fn=secrets.get,
                         state=state if state is not None else {})


def test_site_normalisation():
    assert normalise_site("acme") == ("https://acme.atlassian.net", "acme.atlassian.net")
    assert normalise_site("https://acme.atlassian.net/jira/software/projects") == \
        ("https://acme.atlassian.net", "acme.atlassian.net")
    assert normalise_site("http://acme.atlassian.net") == ("https://acme.atlassian.net", "acme.atlassian.net")
    assert normalise_site("http://127.0.0.1:9000") == ("http://127.0.0.1:9000", "127.0.0.1:9000")
    assert normalise_site("") is None


def test_adf_text_and_mentions():
    doc = _adf([_t("hi "), _m(ME, "@Mira Chen"), _t(" please look")], [_t("line 2")])
    assert adf_text(doc) == "hi @Mira Chen please look\nline 2\n"
    assert mentions(doc, ME) and not mentions(doc, "someone-else")
    assert mentions(f"see [~accountid:{ME}]", ME)  # v2 wiki markup


async def test_verify_ok_returns_site_label(jira):
    v = await JiraAdapter().verify(_ctx(jira.site))
    assert v.label == jira.site.split("//")[1]
    assert v.config == {"site": jira.site}
    assert jira.requests[0][:2] == ("GET", "/rest/api/3/myself")


async def test_verify_maps_failures_to_fields(jira):
    with pytest.raises(VerifyError) as e:
        await JiraAdapter().verify(_ctx(jira.site, token="wrong"))
    assert (e.value.reason, e.value.field) == ("rejected", "token")
    assert TOKEN not in e.value.message and "wrong" not in e.value.message
    jira.status["/rest/api/3/myself"] = 403
    jira.headers["/rest/api/3/myself"] = {"X-Seraph-LoginReason": "AUTHENTICATION_DENIED"}
    with pytest.raises(VerifyError) as e:
        await JiraAdapter().verify(_ctx(jira.site))
    assert (e.value.reason, e.value.field) == ("rejected", "mail")
    jira.status["/rest/api/3/myself"] = 404
    with pytest.raises(VerifyError) as e:
        await JiraAdapter().verify(_ctx(jira.site))
    assert (e.value.reason, e.value.field) == ("not_found", "site")
    with pytest.raises(VerifyError) as e:
        await JiraAdapter().verify(_ctx(jira.site, mail="not-an-email"))
    assert e.value.field == "mail"


def _no_dns(monkeypatch, host):
    # The resolver's own failure, without asking the system resolver.
    import socket
    real = socket.getaddrinfo

    def fake(h, *a, **k):
        if (h.decode() if isinstance(h, bytes) else h) == host:
            raise socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided, or not known")
        return real(h, *a, **k)
    monkeypatch.setattr(socket, "getaddrinfo", fake)


async def test_verify_bad_site_dns_and_refused(monkeypatch):
    _no_dns(monkeypatch, "no-such-site.invalid")
    with pytest.raises(VerifyError) as e:
        await JiraAdapter().verify(_ctx("https://no-such-site.invalid"))
    assert (e.value.reason, e.value.field) == ("not_found", "site")
    with pytest.raises(VerifyError) as e:
        await JiraAdapter().verify(_ctx("http://127.0.0.1:9"))  # discard port: refused
    assert (e.value.reason, e.value.field) == ("unreachable", "site")


async def test_poll_assigned_uses_enhanced_search(jira):
    jira.assigned = [_issue("WEB-12", "Cart total wrong", desc=_adf([_t("Remove an item; total stays.")]))]
    items = await JiraAdapter().poll(_ctx(jira.site))
    assert len(items) == 1
    it = items[0]
    assert (it.id, it.kind, it.subject, it.title) == ("assigned:WEB-12", "assigned", "WEB-12", "WEB-12 Cart total wrong")
    assert it.url == f"{jira.site}/browse/WEB-12"
    assert "Status: In Progress" in it.body and "Remove an item" in it.body
    assert it.at == "2026-11-16T07:20:00Z"
    search = [r for r in jira.requests if r[1] == "/rest/api/3/search/jql"]
    assert search and search[0][0] == "POST"
    assert "updated >=" in search[0][2]["jql"] and "summary" in search[0][2]["fields"]
    assert not any(r[1].startswith("/rest/api/3/search?") or r[1] == "/rest/api/3/search"
                   for r in jira.requests)


async def test_poll_mention_keeps_only_real_mentions_of_me(jira):
    jira.mention_candidates = [
        _issue("WEB-7", "Checkout copy", comments=[
            _comment("10001", _adf([_t("hey "), _m(ME, "@Mira Chen"), _t(" can you check?")])),
            _comment("10002", _adf([_t("unrelated")])),
            _comment("10003", _adf([_m(ME, "@Mira Chen")]), author_id=ME, name="Mira Chen"),  # myself
        ]),
        _issue("WEB-8", "Spec", desc=_adf([_t("owner: "), _m(ME, "@Mira Chen")]), comments=[]),
        _issue("WEB-9", "Text match only", comments=[_comment("10009", _adf([_t(ME)]))]),
        # more comments than the search embedded: fetched from /comment
        _issue("WEB-5", "Long thread", comments=[_comment("10050", _adf([_t("old")]))], total=2),
    ]
    jira.comments["WEB-5"] = [_comment("10051", _adf([_m(ME, "@Mira Chen"), _t(" ping")])),
                              _comment("10050", _adf([_t("old")]))]
    state = {}
    items = await JiraAdapter().poll(_ctx(jira.site, kinds={"mention": {}}, state=state))
    assert [i.id for i in items] == ["mention:WEB-7:10001", "mention:WEB-8:description", "mention:WEB-5:10051"]
    assert items[0].sender == "Kai Wu" and "can you check?" in items[0].body
    assert items[0].url.endswith("/browse/WEB-7?focusedCommentId=10001")
    assert state["me"]["id"] == ME
    # the account id is remembered: the next poll does not ask "myself" again
    before = sum(1 for r in jira.requests if r[1] == "/rest/api/3/myself")
    await JiraAdapter().poll(_ctx(jira.site, kinds={"mention": {}}, state=state))
    assert sum(1 for r in jira.requests if r[1] == "/rest/api/3/myself") == before


async def test_mention_jql_refused_falls_back_to_recent_issues(jira):
    jira.reject_mention_jql = True
    jira.mention_candidates = [_issue("WEB-7", "Checkout", comments=[
        _comment("10001", _adf([_m(ME, "@Mira Chen")]))])]
    items = await JiraAdapter().poll(_ctx(jira.site, kinds={"mention": {}}))
    assert [i.id for i in items] == ["mention:WEB-7:10001"]


async def test_poll_failures(jira):
    with pytest.raises(SourceFailure) as f:
        await JiraAdapter().poll(_ctx(jira.site, token="revoked"))
    assert f.value.code == "token_rejected" and f.value.retryable is False
    jira.status["/rest/api/3/search/jql"] = 429
    with pytest.raises(SourceFailure) as f:
        await JiraAdapter().poll(_ctx(jira.site))
    assert f.value.code == "rate_limited" and f.value.retryable
    jira.status["/rest/api/3/search/jql"] = 503
    with pytest.raises(SourceFailure) as f:
        await JiraAdapter().poll(_ctx(jira.site))
    assert f.value.code == "unreachable"
    with pytest.raises(SourceFailure) as f:  # nothing listening
        await JiraAdapter().poll(_ctx("http://127.0.0.1:9", state={"me": {"key": "x", "id": ME}}))
    assert f.value.code == "unreachable"
