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


def _rest_rows(*nums, repo="acme/web", pr=()):
    """The same issues as ``_rows`` in the REST issue list's shape."""
    rows = []
    for n in nums:
        kind = "pull" if n in pr else "issues"
        r = {"number": n, "title": f"issue {n}", "html_url": f"https://github.com/{repo}/{kind}/{n}",
             "url": f"https://api.github.com/repos/{repo}/issues/{n}",
             "repository_url": f"https://api.github.com/repos/{repo}", "updated_at": "2026-11-16T07:00:00Z",
             "user": {"login": "li-na"}, "body": "b"}
        if n in pr:
            r["pull_request"] = {"url": f"https://api.github.com/repos/{repo}/pulls/{n}"}
        rows.append(r)
    return json.dumps(rows)


def _search_rows_pr(*nums, repo="acme/web", pr=()):
    return json.dumps([{"number": n, "title": f"issue {n}",
                        "url": f"https://github.com/{repo}/{'pull' if n in pr else 'issues'}/{n}",
                        "repository": {"nameWithOwner": repo}, "updatedAt": "2026-11-16T07:00:00Z",
                        "author": {"login": "li-na"}, "body": "b", "isPullRequest": n in pr} for n in nums])


def _lang_ctx(lang, **kw):
    ctx = _ctx("github", **kw)
    ctx.lang = lang
    return ctx


async def test_github_label_reads_the_repo_issue_list_not_search():
    gh = FakeGh([
        (("auth", "token"), RunResult(0, "gho_T\n", "")),
        (("api", "repos/acme/web/issues"), RunResult(0, _rest_rows(3, 4, pr=(4,)), "")),
    ])
    items = await gh.poll(_ctx("github", kinds={"label": {"repo": "acme/web", "label": "bug"}}, account="mira"))
    assert [i.id for i in items] == ["label:acme/web#3", "label:acme/web#4"]
    api = [args for args, tok in gh.calls if "api" in args]
    assert len(api) == 1 and gh.calls[-1][1] == "gho_T"
    args = api[0]
    assert "GET" in args and "labels=bug" in args and "state=open" in args
    assert "sort=updated" in args and "direction=desc" in args and "per_page=50" in args
    # no search and no per-poll repo view on this path
    assert not any("search" in a or "view" in a for a, _ in gh.calls)


async def test_github_same_issue_via_search_and_rest_is_the_same_item():
    a = GitHubAdapter._item("label", json.loads(_search_rows_pr(7, 8, pr=(8,)))[0])
    b = GitHubAdapter._item_rest("label", json.loads(_rest_rows(7, 8, pr=(8,)))[0], "acme/web")
    assert a == b
    a = GitHubAdapter._item("label", json.loads(_search_rows_pr(7, 8, pr=(8,)))[1])
    b = GitHubAdapter._item_rest("label", json.loads(_rest_rows(7, 8, pr=(8,)))[1], "acme/web")
    assert a == b and a.url.endswith("/pull/8")


async def test_github_rest_ids_use_githubs_spelling_of_the_repo():
    # a repo typed in another case still yields the ids search gave
    row = json.loads(_rest_rows(5, repo="Acme/Web"))[0]
    assert GitHubAdapter._item_rest("label", row, "acme/web").id == "label:Acme/Web#5"
    row.pop("repository_url")
    assert GitHubAdapter._item_rest("label", row, "acme/web").id == "label:acme/web#5"


async def test_github_upgrade_from_search_to_rest_brings_no_duplicate_arrivals(tmp_path):
    """A v0.23.0 source recorded its label items through search; the first
    polls on the REST list record nothing new (same ids, the core's seen
    set), and a really new issue still arrives once."""
    answers = [
        (("auth", "token"), RunResult(0, "t", "")),
        (("repo", "view", "acme/web"), RunResult(0, "{}", "")),
        (("label", "list"), RunResult(0, json.dumps([{"name": "bug"}]), "")),
        (("api", "repos/acme/web/issues"), RunResult(0, _rest_rows(1, 2, 3), "")),
    ]

    class Upgraded(FakeGh):
        old = True

        async def poll(self, ctx):
            if self.old:  # what v0.23.0 produced: search rows through _item
                return [self._item("label", r) for r in json.loads(_search_rows_pr(1, 2, 3))]
            return await super().poll(ctx)

    gh = Upgraded(answers)
    gh.DESCRIPTOR = None
    sent = []

    async def send(text, project_id):
        sent.append(text)
        return {"project_id": "proj_a", "session_id": f"s{len(sent)}", "by": "model"}

    core = SourcesCore(str(tmp_path), registry=AdapterRegistry(overrides={"github": gh}),
                       secret_store=MemorySecretStore(), send=send)
    core.load()
    sid = (await core.create({"type": "github", "account": "mira",
                              "kinds": {"label": {"repo": "acme/web", "label": "bug"}}}))["id"]
    await core._poll_one(core.store.get(sid))  # baseline on the old path
    await core._poll_one(core.store.get(sid))
    before = list(core.store.rt(sid)["entries"])
    gh.old = False
    await core._poll_one(core.store.get(sid))
    await core._poll_one(core.store.get(sid))
    assert core.store.rt(sid)["entries"] == before
    assert any("api" in a for a, _ in gh.calls)
    gh.answers[-1] = (("api", "repos/acme/web/issues"), RunResult(0, _rest_rows(4, 1, 2, 3), ""))
    await core._poll_one(core.store.get(sid))
    new = [e for e in core.store.rt(sid)["entries"] if e not in before]
    assert len(new) == 1 and "#4" in json.dumps(new[0], ensure_ascii=False)


async def test_github_label_with_a_comma_stays_on_search():
    gh = FakeGh([
        (("auth", "token"), RunResult(0, "t", "")),
        (("repo", "view", "acme/web"), RunResult(0, "{}", "")),
        (("search", "--label", "a,b"), RunResult(0, _rows(9), "")),
    ])
    items = await gh.poll(_ctx("github", kinds={"label": {"repo": "acme/web", "label": "a,b"}}, account="mira"))
    assert [i.id for i in items] == ["label:acme/web#9"]
    assert not any("api" in a for a, _ in gh.calls)


async def test_github_label_repo_gone_is_repo_not_found():
    gh = FakeGh([
        (("auth", "token"), RunResult(0, "t", "")),
        (("api", "repos/acme/gone/issues"), RunResult(
            1, '{"message":"Not Found","status":"404"}', "gh: Not Found (HTTP 404)")),
    ])
    with pytest.raises(SourceFailure) as e:
        await gh.poll(_ctx("github", kinds={"label": {"repo": "acme/gone", "label": "bug"}}, account="mira"))
    assert e.value.code == "repo_not_found" and e.value.params == {"repo": "acme/gone"}


async def test_github_label_rest_other_failures_keep_their_code():
    gh = FakeGh([
        (("auth", "token"), RunResult(0, "t", "")),
        (("api", "repos/acme/web/issues"), RunResult(1, "", "gh: Bad credentials (HTTP 401)")),
    ])
    with pytest.raises(SourceFailure) as e:
        await gh.poll(_ctx("github", kinds={"label": {"repo": "acme/web", "label": "bug"}}, account="mira"))
    assert e.value.code == "auth_expired"


async def test_github_comma_label_repo_gone_via_search_is_still_repo_not_found():
    gh = FakeGh([
        (("auth", "token"), RunResult(0, "t", "")),
        (("repo", "view"), RunResult(1, "", "GraphQL: Could not resolve to a Repository with the name 'acme/gone'.")),
    ])
    with pytest.raises(SourceFailure) as e:
        await gh.poll(_ctx("github", kinds={"label": {"repo": "acme/gone", "label": "a,b"}}, account="mira"))
    assert e.value.code == "repo_not_found" and e.value.params == {"repo": "acme/gone"}


async def test_github_body_line_is_unchanged_whatever_the_interface_language():
    # what the agent receives stays as before (localizing it is out of scope)
    answers = [
        (("auth", "token"), RunResult(0, "t", "")),
        (("api", "repos/acme/web/issues"), RunResult(0, _rest_rows(1, 2, pr=(2,)), "")),
        (("--assignee",), RunResult(0, _search_rows_pr(5, 6, pr=(6,)), "")),
    ]
    kinds = {"label": {"repo": "acme/web", "label": "bug"}, "assigned": {}}
    for lang in ("zh", "en"):
        items = await FakeGh(answers).poll(_lang_ctx(lang, kinds=kinds, account="mira"))
        assert {i.id: i.body for i in items} == {
            "assigned:acme/web#5": "Issue in acme/web\n\nb", "assigned:acme/web#6": "Pull request in acme/web\n\nb",
            "label:acme/web#1": "Issue in acme/web\n\nb", "label:acme/web#2": "Pull request in acme/web\n\nb"}


def test_github_polls_every_minute():
    from agent_os.sources.catalog import _BASE
    assert GitHubAdapter.interval_s == 60
    assert _BASE["github"]["interval_s"] == 60


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
