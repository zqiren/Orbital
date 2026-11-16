# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""GitLab source adapter against a fake ``glab`` executable on PATH.

Where the fake's shapes come from:
- ``glab auth status --all --output json`` when not signed in: run on this Mac
  (glab 1.130.0, scratch HOME): ``{"hosts":[{"host":"gitlab.com","state":"error",
  "error":"GET https://gitlab.com/api/v4/user: 401 {message: 401 Unauthorized}",
  "token_source":"none",...}]}``, exit 1.
- Signed-in row: ASSUMED ``"state":"success"`` plus ``"user"`` (json tags
  ``state``/``user`` found in the glab binary; not observed — no account).
  The adapter falls back to ``glab api user`` when ``user`` is absent.
- ``glab api … user`` / ``todos``: GitLab REST docs
  (https://docs.gitlab.com/api/users/#retrieve-the-current-user,
  https://docs.gitlab.com/api/todos/). Unauthenticated ``glab api`` output
  (``{"message":"401 Unauthorized"}`` + stderr ``glab: 401 Unauthorized
  (HTTP 401)``, exit 1) and an unknown host (``dial tcp: lookup …: no such
  host``) were run on this Mac.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap

import pytest

from agent_os.sources.adapters.gitlab import GitLabAdapter, join_account, split_account
from agent_os.sources.context import SourceContext
from agent_os.sources.model import SourceFailure, VerifyError

pytestmark = pytest.mark.skipif(sys.platform == "win32",
                                reason="the fake glab is a #! script (POSIX only)")

_FAKE = textwrap.dedent("""\
    #!{python}
    import json, os, sys
    state = json.load(open(os.environ["FAKE_GLAB_STATE"]))
    args = sys.argv[1:]
    with open(os.environ["FAKE_GLAB_STATE"] + ".calls", "a") as f:
        f.write(json.dumps({{"args": args, "cwd": os.getcwd(),
                             "no_prompt": os.environ.get("GLAB_NO_PROMPT")}}) + "\\n")
    def out(code, stdout="", stderr=""):
        sys.stdout.write(stdout); sys.stderr.write(stderr); sys.exit(code)
    if args[:2] == ["auth", "status"]:
        out(state.get("status_code", 1), json.dumps(state["status"]))
    if args[0] == "api":
        host = args[args.index("--hostname") + 1]
        endpoint = [a for a in args[1:] if not a.startswith("-") and a != host
                    and "=" not in a and a != "GET"][0]
        h = state["hosts"].get(host)
        if h is None:
            out(1, "", "ERROR Get \\"https://%s/api/v4/user\\": dial tcp: lookup %s: no such host." % (host, host))
        if not h.get("user"):
            out(1, '{{"message":"401 Unauthorized"}}', "glab: 401 Unauthorized (HTTP 401)")
        if endpoint == "user":
            out(0, json.dumps({{"id": 1, "username": h["user"], "name": h["user"].title()}}))
        if endpoint == "todos":
            if h.get("todos_error"):
                out(1, "", h["todos_error"])
            out(0, json.dumps(h.get("todos", [])))
    out(2, "", "unknown command")
""")


def _todo(tid, action, iid, *, mr=False, path="acme/web", title="Cart total wrong", body=None):
    kind = "MergeRequest" if mr else "Issue"
    seg = "merge_requests" if mr else "issues"
    return {
        "id": tid,
        "project": {"id": 2, "name": "web", "name_with_namespace": "Acme / web",
                    "path": "web", "path_with_namespace": path},
        "author": {"id": 9, "name": "Li Na", "username": "li-na"},
        "action_name": action,
        "target_type": kind,
        "target": {"id": 300 + iid, "iid": iid, "project_id": 2, "title": title,
                   "description": "Steps: remove an item.", "state": "opened",
                   "web_url": f"https://gitlab.com/{path}/-/{seg}/{iid}"},
        "target_url": f"https://gitlab.com/{path}/-/{seg}/{iid}",
        "body": body if body is not None else title,
        "state": "pending",
        "created_at": "2026-11-16T07:52:35.225Z",
        "updated_at": "2026-11-16T07:52:35.225Z",
    }


@pytest.fixture
def glab(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "glab"
    exe.write_text(_FAKE.format(python=sys.executable))
    exe.chmod(0o755)
    state_file = tmp_path / "glab.json"

    class G:
        state = {"status": {"hosts": []}, "status_code": 1, "hosts": {}}

        def save(self):
            state_file.write_text(json.dumps(self.state))

        def calls(self):
            p = str(state_file) + ".calls"
            if not os.path.exists(p):
                return []
            return [json.loads(line) for line in open(p)]

    g = G()
    g.save()
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + "/usr/bin" + os.pathsep + "/bin")
    monkeypatch.setenv("FAKE_GLAB_STATE", str(state_file))
    return g


def _ctx(account="mira", kinds=None):
    return SourceContext(source_id="src_1", type_="gitlab", config={}, kinds=kinds or {"assigned": {}},
                         account=account, secrets_fn=lambda k: None, state={})


def test_account_value_round_trips_host():
    assert split_account("mira") == ("mira", "gitlab.com")
    assert split_account("mira@gitlab.acme.com") == ("mira", "gitlab.acme.com")
    assert join_account("mira", "gitlab.com") == "mira"
    assert join_account("mira", "gitlab.acme.com") == "mira@gitlab.acme.com"


async def test_not_installed(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert (await GitLabAdapter().accounts()).state == "not_installed"
    with pytest.raises(VerifyError) as e:
        await GitLabAdapter().verify(_ctx())
    assert e.value.reason == "cli_missing"
    with pytest.raises(SourceFailure) as f:
        await GitLabAdapter().poll(_ctx())
    assert f.value.code == "cli_missing" and f.value.params == {"cli": "glab"}


async def test_installed_but_not_signed_in(glab):
    # the exact output of this Mac's glab 1.130.0 with no token
    glab.state["status"] = {"hosts": [{"host": "gitlab.com", "state": "error",
                                       "error": "GET https://gitlab.com/api/v4/user: 401 {message: 401 Unauthorized}",
                                       "token_source": "none", "git_protocol": "ssh"}]}
    glab.state["hosts"] = {"gitlab.com": {}}
    glab.save()
    acc = await GitLabAdapter().accounts()
    assert acc.state == "signed_out" and acc.accounts == []
    with pytest.raises(VerifyError) as e:
        await GitLabAdapter().verify(_ctx("mira"))
    assert e.value.reason == "signed_out" and e.value.field == "account"


async def test_accounts_lists_each_signed_in_host(glab):
    glab.state["status"] = {"hosts": [
        {"host": "gitlab.com", "state": "success", "user": "mira", "token_source": "keyring"},
        {"host": "gitlab.acme.com", "state": "success", "token_source": "keyring"},  # no user field
        {"host": "old.acme.com", "state": "error", "error": "401", "token_source": "config"},
    ]}
    glab.state["hosts"] = {"gitlab.com": {"user": "mira"}, "gitlab.acme.com": {"user": "mchen"}}
    glab.save()
    acc = await GitLabAdapter().accounts()
    assert acc.state == "signed_in"
    assert acc.accounts == [
        {"login": "mira", "host": "gitlab.com", "active": True},
        {"login": "mchen@gitlab.acme.com", "host": "gitlab.acme.com", "active": False},
    ]


async def test_verify_pins_host_and_user_and_never_touches_the_default(glab):
    glab.state["hosts"] = {"gitlab.acme.com": {"user": "mchen"}}
    glab.save()
    v = await GitLabAdapter().verify(_ctx("mchen@gitlab.acme.com"))
    assert v.label == "mchen@gitlab.acme.com"
    calls = glab.calls()
    assert calls and all("--hostname" in c["args"] for c in calls)
    assert not any(c["args"][:2] in (["config", "set"], ["auth", "login"]) for c in calls)
    assert all(c["no_prompt"] == "1" for c in calls)
    assert all(c["cwd"] == os.path.realpath(os.path.expanduser("~")) or c["cwd"] == os.path.expanduser("~")
               for c in calls)
    # someone signed glab in to that host as another user: refuse, don't read theirs
    glab.state["hosts"] = {"gitlab.acme.com": {"user": "boss"}}
    glab.save()
    with pytest.raises(VerifyError) as e:
        await GitLabAdapter().verify(_ctx("mchen@gitlab.acme.com"))
    assert e.value.reason == "signed_out"
    with pytest.raises(SourceFailure) as f:
        await GitLabAdapter().poll(_ctx("mchen@gitlab.acme.com"))
    assert f.value.code == "cli_signed_out" and f.value.params["account"] == "mchen@gitlab.acme.com"


async def test_unknown_host_is_unreachable(glab):
    with pytest.raises(VerifyError) as e:
        await GitLabAdapter().verify(_ctx("mira@gitlab.gone.example"))
    assert e.value.reason == "unreachable"
    with pytest.raises(SourceFailure) as f:
        await GitLabAdapter().poll(_ctx("mira@gitlab.gone.example"))
    assert f.value.code == "unreachable" and f.value.params == {"host": "gitlab.gone.example"}


async def test_poll_one_call_maps_todos_to_kinds(glab):
    glab.state["hosts"] = {"gitlab.com": {"user": "mira", "todos": [
        _todo(905, "review_requested", 12, mr=True, title="Fix cart total"),
        _todo(904, "mentioned", 7, body="@mira can you look at this?"),
        _todo(903, "assigned", 7),
        _todo(902, "build_failed", 11, mr=True),      # not one of our kinds
        _todo(901, "directly_addressed", 8, title="Checkout copy"),
        _todo(900, "marked", 9),                      # "added a to-do": not ours
    ]}}
    glab.save()
    a = GitLabAdapter()
    items = await a.poll(_ctx(kinds={"assigned": {}, "mention": {}, "review": {}}))
    assert [(i.id, i.kind) for i in items] == [
        ("todo:905", "review"), ("todo:904", "mention"), ("todo:903", "assigned"), ("todo:901", "mention")]
    review, mention, assigned, _ = items
    assert review.title == "!12 Fix cart total" and review.subject == "acme/web!12"
    assert review.url == "https://gitlab.com/acme/web/-/merge_requests/12"
    assert mention.subject == assigned.subject == "acme/web#7"  # same issue → merges in the core
    assert "@mira can you look at this?" in mention.body and mention.sender == "li-na"
    assert assigned.at == "2026-11-16T07:52:35.225Z"
    todo_calls = [c for c in glab.calls() if "todos" in c["args"]]
    assert len(todo_calls) == 1
    assert "state=pending" in todo_calls[0]["args"] and "GET" in todo_calls[0]["args"]
    # only switched-on kinds come back
    items = await a.poll(_ctx(kinds={"review": {}}))
    assert [i.id for i in items] == ["todo:905"]


async def test_token_expired_mid_life_reads_as_signed_out(glab):
    glab.state["hosts"] = {"gitlab.com": {}}  # token revoked: every call 401
    glab.save()
    with pytest.raises(SourceFailure) as f:
        await GitLabAdapter().poll(_ctx("mira"))
    assert f.value.code == "cli_signed_out" and f.value.retryable


async def test_rate_limited_todos(glab):
    glab.state["hosts"] = {"gitlab.com": {"user": "mira", "todos_error": "glab: 429 Too Many Requests (HTTP 429)"}}
    glab.save()
    with pytest.raises(SourceFailure) as f:
        await GitLabAdapter().poll(_ctx("mira"))
    assert f.value.code == "rate_limited"
