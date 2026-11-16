# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Task sources: the reference adapters (website, check command, GitHub)."""

from __future__ import annotations

import http.server
import json
import sys
import threading

import pytest

from agent_os.sources.adapters.command import CommandAdapter, parse_items
from agent_os.sources.adapters.github import GitHubAdapter
from agent_os.sources.adapters.website import WebsiteAdapter
from agent_os.sources.context import RunResult, SourceContext
from agent_os.sources.core import SourcesCore
from agent_os.sources.model import Item, Resolve, SourceFailure, VerifyError
from agent_os.sources.registry import AdapterRegistry
from agent_os.sources.secrets import MemorySecretStore


def _ctx(type_="website", config=None, kinds=None, account=None, command=None, state=None):
    return SourceContext(source_id="src_00000000", type_=type_, config=config or {},
                         kinds=kinds or {}, account=account, secrets_fn=lambda k: None,
                         state=state if state is not None else {}, command=command)


# ── website ────────────────────────────────────────────────────────────────

class _Site:
    def __init__(self):
        self.status = 200
        self.body = "<h1>立即购买</h1>"
        site = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                data = site.body.encode()
                self.send_response(site.status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def site():
    s = _Site()
    yield s
    s.close()


async def test_website_verify_normalises_and_labels():
    v = await WebsiteAdapter().verify(_ctx(config={"url": "shop.acme.com"}))
    assert v.label == "shop.acme.com" and v.config == {"url": "https://shop.acme.com"}
    with pytest.raises(VerifyError) as e:
        await WebsiteAdapter().verify(_ctx(config={"url": "ftp://x"}))
    assert e.value.field == "url"


async def test_website_down_once_silent_while_down_resolved_on_recovery(site):
    a = WebsiteAdapter()
    a.retry_delay_s = 0
    state = {}
    ctx = lambda: _ctx(config={"url": site.url, "must_contain": "立即购买"}, kinds={"down": {}}, state=state)
    assert await a.poll(ctx()) == []
    site.status = 503
    first = await a.poll(ctx())
    assert len(first) == 1 and isinstance(first[0], Item)
    assert first[0].kind == "down" and "HTTP 503" in first[0].body
    assert await a.poll(ctx()) == []          # still down: nothing
    site.status = 200
    site.body = "sold out"                    # up, but the text is gone: still down
    assert await a.poll(ctx()) == []
    site.body = "<h1>立即购买</h1>"
    back = await a.poll(ctx())
    assert len(back) == 1 and isinstance(back[0], Resolve)
    assert back[0].subject == first[0].subject
    assert await a.poll(ctx()) == []


async def test_website_retries_once_before_calling_it_down(site):
    a = WebsiteAdapter()
    a.retry_delay_s = 0.01
    calls = []
    real = a._probe

    async def flaky(url, must):
        calls.append(1)
        return "no answer" if len(calls) == 1 else await real(url, must)
    a._probe = flaky
    assert await a.poll(_ctx(config={"url": site.url}, kinds={"down": {}})) == []
    assert len(calls) == 2


async def test_website_incident_through_the_core_is_one_entry_closed_on_recovery(tmp_path, site):
    adapter = WebsiteAdapter()
    adapter.retry_delay_s = 0
    sent = []

    async def send(text, pid):
        sent.append(text)
        return {"project_id": "p", "session_id": "s1", "by": "model"}

    async def sessions(pid, sid):
        return pid, "running"
    core = SourcesCore(str(tmp_path), registry=AdapterRegistry({"website": adapter}),
                       secret_store=MemorySecretStore(), send=send, session_info=sessions)
    core.load()
    sid = (await core.create({"type": "website", "config": {"url": site.url}}))["id"]
    cfg = core.store.get(sid)
    await core._poll_one(cfg)
    site.status = 500
    await core._poll_one(cfg)   # down → one arrival (no baseline for a check)
    await core._poll_one(cfg)   # still down → nothing
    await core.dispatch_once()
    site.status = 200
    await core._poll_one(cfg)   # recovered → same entry
    h = (await core.detail(sid))["history"]
    assert len(h) == 1 and len(sent) == 1
    assert h[0]["state"] == "routed" and h[0]["recovered_at"] is not None
    assert sent[0].splitlines()[0] == f"127.0.0.1 is down"


# ── check command ──────────────────────────────────────────────────────────

def test_command_output_reads_only_the_item_fields():
    items = parse_items(json.dumps([
        {"id": 7, "title": "seven", "project": "proj_x", "agent": "codex", "cwd": "/",
         "permissions": "all", "url": "https://x/7"},
        {"id": "", "title": "no id"},
        {"id": "t", "title": ""},
        "junk",
    ]))
    assert len(items) == 1
    it = items[0]
    assert (it.id, it.title, it.url, it.kind, it.subject) == ("7", "seven", "https://x/7", "item", "7")
    assert not hasattr(it, "project") and not hasattr(it, "agent")
    assert parse_items(json.dumps({"items": [{"id": "a", "title": "A"}]}))[0].id == "a"
    assert parse_items("") == []
    with pytest.raises(ValueError):
        parse_items("not json")
    with pytest.raises(ValueError):
        parse_items('{"x": 1}')


def _py(code: str) -> str:
    return f'"{sys.executable}" -c "{code}"'


async def test_command_runs_through_the_shell_and_maps_failures(tmp_path):
    a = CommandAdapter()
    ok = _py("import json;print(json.dumps([{'id':'1','title':'one'}]))")
    items = await a.poll(_ctx("command", command={"run": ok, "cwd": str(tmp_path)}))
    assert [i.id for i in items] == ["1"]
    with pytest.raises(SourceFailure) as e:
        await a.poll(_ctx("command", command={"run": _py("import sys;sys.exit(3)")}))
    assert e.value.code == "command_exit" and e.value.params == {"code": 3}
    with pytest.raises(SourceFailure) as e:
        await a.poll(_ctx("command", command={"run": _py("print('hello')")}))
    assert e.value.code == "command_output"


async def test_command_verify_never_runs_it(tmp_path):
    marker = tmp_path / "ran"
    a = CommandAdapter()
    await a.verify(_ctx("command", command={"run": f"touch {marker}"}))
    assert not marker.exists()
    with pytest.raises(VerifyError):
        await a.verify(_ctx("command", command={"run": "x", "cwd": str(tmp_path / "missing")}))


# ── GitHub (the CLI is faked) ──────────────────────────────────────────────

class FakeGh(GitHubAdapter):
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    async def _run(self, ctx, args, *, token=None, timeout=60):
        self.calls.append((tuple(args), token))
        for key, res in self.answers:
            if all(k in args for k in key):
                return res
        return RunResult(1, "", "unknown")


def _rows(*nums, repo="acme/web"):
    return json.dumps([{"number": n, "title": f"issue {n}", "url": f"https://github.com/{repo}/issues/{n}",
                        "repository": {"nameWithOwner": repo}, "updatedAt": "2026-11-16T07:00:00Z",
                        "author": {"login": "li-na"}, "body": "b", "isPullRequest": False} for n in nums])


async def test_github_poll_pins_the_account_token_and_maps_items():
    gh = FakeGh([
        (("auth", "token", "--user", "mira"), RunResult(0, "gho_TOKEN\n", "")),
        (("--assignee",), RunResult(0, _rows(1, 2), "")),
        (("--mentions",), RunResult(0, _rows(2), "")),
    ])
    items = await gh.poll(_ctx("github", kinds={"assigned": {}, "mention": {}}, account="mira"))
    assert [i.id for i in items] == ["assigned:acme/web#1", "assigned:acme/web#2", "mention:acme/web#2"]
    assert items[1].subject == items[2].subject == "acme/web#2"  # same issue: merges in the core
    assert items[0].title == "#1 issue 1" and items[0].sender == "li-na"
    search_calls = [c for c in gh.calls if "search" in c[0]]
    assert search_calls and all(tok == "gho_TOKEN" for _, tok in search_calls)
    assert all("@me" in args for args, _ in search_calls)


async def test_github_signed_out_account_fails_as_signed_out():
    gh = FakeGh([(("auth", "token"), RunResult(1, "", "no account"))])
    with pytest.raises(SourceFailure) as e:
        await gh.poll(_ctx("github", kinds={"assigned": {}}, account="ghost"))
    assert e.value.code == "cli_signed_out" and e.value.params["account"] == "ghost"


async def test_github_verify_checks_repo_and_label():
    gh = FakeGh([
        (("auth", "token"), RunResult(0, "t", "")),
        (("repo", "view", "acme/web"), RunResult(0, "{}", "")),
        (("label", "list"), RunResult(0, json.dumps([{"name": "bug"}]), "")),
    ])
    v = await gh.verify(_ctx("github", kinds={"label": {"repo": "acme/web", "label": "bug"}}, account="mira"))
    assert v.label == "mira"
    with pytest.raises(VerifyError) as e:
        await gh.verify(_ctx("github", kinds={"label": {"repo": "acme/web", "label": "nope"}}, account="mira"))
    assert e.value.field == "label" and e.value.reason == "not_found"
    with pytest.raises(VerifyError) as e:
        await gh.verify(_ctx("github", kinds={"label": {"repo": "acme/gone", "label": "bug"}}, account="mira"))
    assert e.value.field == "repo"


async def test_github_label_repo_gone_is_repo_not_found():
    gh = FakeGh([
        (("auth", "token"), RunResult(0, "t", "")),
        (("repo", "view"), RunResult(1, "", "GraphQL: Could not resolve to a Repository with the name 'acme/gone'.")),
    ])
    with pytest.raises(SourceFailure) as e:
        await gh.poll(_ctx("github", kinds={"label": {"repo": "acme/gone", "label": "bug"}}, account="mira"))
    assert e.value.code == "repo_not_found" and e.value.params == {"repo": "acme/gone"}


# ── language of titles Orbital composes ────────────────────────────────────

async def test_website_title_follows_the_interface_language(site):
    a = WebsiteAdapter()
    a.retry_delay_s = 0
    site.status = 503
    ctx = _ctx(config={"url": site.url}, kinds={"down": {}})
    ctx.lang = "zh"
    assert (await a.poll(ctx))[0].title == "127.0.0.1 打不开"
    ctx2 = _ctx(config={"url": site.url}, kinds={"down": {}})
    assert ctx2.lang == "en"
    assert (await a.poll(ctx2))[0].title == "127.0.0.1 is down"


async def test_core_hands_adapters_the_interface_language(tmp_path):
    seen = []

    class A(WebsiteAdapter):
        async def poll(self, ctx):
            seen.append(ctx.lang)
            return []
    core = SourcesCore(str(tmp_path), registry=AdapterRegistry({"website": A()}),
                       secret_store=MemorySecretStore(), lang_fn=lambda: "zh")
    core.load()
    sid = (await core.create({"type": "website", "config": {"url": "https://x.example"}}))["id"]
    await core._poll_one(core.store.get(sid))
    assert seen == ["zh"]
