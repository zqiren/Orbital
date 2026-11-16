# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""GitLab, through the GitLab CLI (``glab``) the user is already signed in to.

glab keeps one sign-in per GitLab host (gitlab.com or a self-managed
instance). An account here is that host's signed-in user; its value is the
username for gitlab.com and ``username@host`` for any other host. Every call
names the host with ``--hostname``, so glab's default host (its ``host``
config key) is never changed, and each call first checks that the host is
still signed in as the pinned user: if someone signed glab in as another
user, the source stops with ``cli_signed_out`` instead of reading the other
account.

Kinds come from one call, the to-do list (``GET /todos?state=pending``,
newest first), which GitLab fills when you are assigned, mentioned or asked
to review: ``assigned`` ← ``assigned``; ``mention`` ← ``mentioned`` /
``directly_addressed``; ``review`` ← ``review_requested`` /
``approval_required``. The item id is the to-do's own id. Orbital only reads
the list; it never marks a to-do done.
"""

from __future__ import annotations

import json
import os
import shutil

from agent_os.sources.adapters import Adapter
from agent_os.sources.context import CommandNotFound, CommandTimeout
from agent_os.sources.model import Accounts, Item, SourceFailure, Verified, VerifyError

DEFAULT_HOST = "gitlab.com"
PER_PAGE = 100

_ACTION_KIND = {
    "assigned": "assigned",
    "mentioned": "mention",
    "directly_addressed": "mention",
    "review_requested": "review",
    "approval_required": "review",
}
_ACTION_WORDS = {
    "assigned": "Assigned to you",
    "mentioned": "You were mentioned",
    "directly_addressed": "You were mentioned",
    "review_requested": "Review requested",
    "approval_required": "Approval required",
}
_ENV = {
    "NO_COLOR": "1",
    "NO_PROMPT": "1",
    "GLAB_NO_PROMPT": "1",
    "GLAB_CHECK_UPDATE": "false",
    "GLAB_SEND_TELEMETRY": "false",
    "GLAB_PAGER": "cat",
    "PAGER": "cat",
}


def _glab() -> str | None:
    return shutil.which("glab")


def split_account(account: str) -> tuple[str, str]:
    """``"mira"`` → (mira, gitlab.com); ``"mira@gitlab.acme.com"`` →
    (mira, gitlab.acme.com). GitLab usernames cannot contain ``@``."""
    a = (account or "").strip()
    if "@" in a:
        user, host = a.rsplit("@", 1)
        if user and host:
            return user, host.lower()
    return a, DEFAULT_HOST


def join_account(user: str, host: str) -> str:
    return user if host == DEFAULT_HOST else f"{user}@{host}"


def _classify(text: str) -> str:
    s = (text or "").lower()
    if "has not been authenticated" in s or "401" in s or "unauthorized" in s or "auth login" in s:
        return "signed_out"
    if "429" in s or "rate limit" in s or "too many requests" in s:
        return "rate_limited"
    if ("no such host" in s or "dial tcp" in s or "timeout" in s or "connection refused" in s
            or "network" in s or "tls handshake" in s):
        return "unreachable"
    return "error"


class GitLabAdapter(Adapter):
    mode = "poll"
    interval_s = 300

    # -- the CLI ------------------------------------------------------------

    async def _run(self, run, args, *, timeout=60):
        glab = _glab()
        if glab is None:
            raise CommandNotFound("glab")
        # cwd = home: inside a git checkout glab would otherwise pick that
        # repository's host for commands without --hostname.
        return await run([glab, *args], env=dict(_ENV), timeout=timeout, cwd=os.path.expanduser("~"))

    async def _whoami(self, run, host: str) -> tuple[str | None, str]:
        """(username or None, failure class) for ``host``."""
        try:
            res = await self._run(run, ["api", "--hostname", host, "user"], timeout=30)
        except CommandTimeout:
            return None, "unreachable"
        if res.code != 0:
            return None, _classify(res.stderr + "\n" + res.stdout)
        try:
            data = json.loads(res.stdout or "{}")
        except ValueError:
            return None, "error"
        user = data.get("username") if isinstance(data, dict) else None
        return (user if isinstance(user, str) and user else None), ("ok" if user else "error")

    # -- accounts -----------------------------------------------------------

    async def accounts(self) -> Accounts:
        from agent_os.sources.context import run_argv
        if _glab() is None:
            return Accounts(state="not_installed")
        try:
            res = await self._run(run_argv, ["auth", "status", "--all", "--output", "json"], timeout=40)
        except CommandNotFound:
            return Accounts(state="not_installed")
        except CommandTimeout:
            return Accounts(state="signed_out")
        # glab exits 1 when any configured host fails; the JSON is still printed.
        try:
            data = json.loads(res.stdout or "{}")
        except ValueError:
            data = {}
        rows = data.get("hosts") if isinstance(data, dict) else None
        accounts = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            host = (row.get("host") or "").strip().lower()
            if not host or row.get("error") or row.get("token_source") == "none":
                continue
            if row.get("state") not in (None, "success", "ok", "authenticated"):
                continue
            user = next((row[k] for k in ("user", "username", "login")
                         if isinstance(row.get(k), str) and row.get(k)), None)
            if user is None:
                user, _ = await self._whoami(run_argv, host)
            if user:
                accounts.append({"login": join_account(user, host), "host": host, "active": False})
        if accounts:
            preferred = next((a for a in accounts if a["host"] == DEFAULT_HOST), accounts[0])
            preferred["active"] = True
        accounts.sort(key=lambda a: (not a["active"], a["login"].lower()))
        return Accounts(state="signed_in" if accounts else "signed_out", accounts=accounts)

    # -- verify -------------------------------------------------------------

    async def _check_account(self, run, account: str) -> tuple[str, str]:
        user, host = split_account(account)
        if not user:
            raise VerifyError("invalid", "account", "pick an account")
        try:
            who, why = await self._whoami(run, host)
        except CommandNotFound:
            raise VerifyError("cli_missing", "account", "the GitLab CLI (glab) is not installed")
        if who is None:
            if why == "unreachable":
                raise VerifyError("unreachable", "account", f"{host} did not answer")
            if why == "signed_out":
                raise VerifyError("signed_out", "account", f"{account} is not signed in to glab")
            raise VerifyError("error", "account", f"glab could not read the account on {host}")
        if who.lower() != user.lower():
            raise VerifyError("signed_out", "account",
                              f"glab is signed in to {host} as {who} now, not {user}")
        return user, host

    async def verify(self, ctx) -> Verified:
        account = (ctx.account or "").strip()
        await self._check_account(ctx.run, account)
        return Verified(label=account)

    # -- poll ---------------------------------------------------------------

    async def poll(self, ctx) -> list:
        account = (ctx.account or "").strip()
        try:
            _user, host = await self._check_account(ctx.run, account)
        except VerifyError as e:
            if e.reason == "cli_missing":
                raise SourceFailure("cli_missing", {"cli": "glab"})
            if e.reason == "unreachable":
                raise SourceFailure("unreachable", {"host": split_account(account)[1]})
            if e.reason == "error":
                raise SourceFailure("error")
            raise SourceFailure("cli_signed_out", {"cli": "glab", "account": account})
        try:
            res = await self._run(ctx.run, ["api", "--hostname", host, "--method", "GET", "todos",
                                            "-f", "state=pending", "-f", f"per_page={PER_PAGE}"])
        except CommandNotFound:
            raise SourceFailure("cli_missing", {"cli": "glab"})
        except CommandTimeout:
            raise SourceFailure("unreachable", {"host": host})
        if res.code != 0:
            code = _classify(res.stderr + "\n" + res.stdout)
            if code == "signed_out":
                raise SourceFailure("cli_signed_out", {"cli": "glab", "account": account})
            raise SourceFailure(code, {"host": host} if code == "unreachable" else {})
        try:
            rows = json.loads(res.stdout or "[]")
        except ValueError:
            raise SourceFailure("error")
        out = []
        for row in rows if isinstance(rows, list) else []:
            item = self._item(row)
            if item is not None and item.kind in ctx.kinds:
                out.append(item)
        return out

    @staticmethod
    def _item(r) -> Item | None:
        if not isinstance(r, dict) or r.get("id") is None:
            return None
        action = r.get("action_name")
        kind = _ACTION_KIND.get(action)
        if kind is None:
            return None
        target = r.get("target") if isinstance(r.get("target"), dict) else {}
        project = r.get("project") if isinstance(r.get("project"), dict) else {}
        path = project.get("path_with_namespace") or project.get("name_with_namespace") or ""
        is_mr = r.get("target_type") == "MergeRequest"
        iid = target.get("iid")
        ref = (f"!{iid}" if is_mr else f"#{iid}") if iid is not None else ""
        title = f"{ref} {target.get('title') or r.get('body') or ''}".strip()
        subject = f"{path}{ref}" if path and ref else f"todo:{r['id']}"
        note = r.get("body") or ""
        lines = [f"{'Merge request' if is_mr else 'Issue'} in {path}".rstrip(),
                 _ACTION_WORDS.get(action, action)]
        if note and note.strip() != (target.get("title") or "").strip():
            lines += ["", note]
        if target.get("description"):
            lines += ["", target["description"]]
        author = r.get("author") if isinstance(r.get("author"), dict) else {}
        return Item(
            id=f"todo:{r['id']}",
            kind=kind,
            subject=subject,
            title=title or f"To-do {r['id']}",
            body="\n".join(lines),
            url=r.get("target_url") or target.get("web_url"),
            sender=author.get("username") or author.get("name"),
            at=r.get("created_at"),
        )


ADAPTER = GitLabAdapter
