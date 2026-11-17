# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""GitHub, through the GitHub CLI the user is already signed in to.

The source is pinned to one account: every call runs with that account's
token (``gh auth token --user <login>``) in ``GH_TOKEN``, so the terminal's
active account is never changed and several accounts can coexist. The token
lives only in the child process's environment for the call; it is never
stored or logged.

Kinds: assigned to me / mentions me (across every repository the account
sees, ``gh search issues --assignee|--mentions @me``, pull requests
included) and with a label in one repository. The label kind reads the
repository's own issue list (``gh api repos/{repo}/issues?labels=…``), which
shows a new issue at once, where search lags behind its index; a label
whose name holds a comma (the list's separator) still goes through search.
Both give the same items (ids ``kind:repo#n``). Each poll re-reads the most
recently updated open items; the core records each id once.
"""

from __future__ import annotations

import json
import shutil

from agent_os.sources.adapters import Adapter
from agent_os.sources.context import CommandNotFound, CommandTimeout
from agent_os.sources.model import Accounts, Choice, Item, SourceFailure, Verified, VerifyError

HOST = "github.com"
SEARCH_LIMIT = 50
_FIELDS = "number,title,url,repository,updatedAt,author,body,isPullRequest"


def _gh() -> str | None:
    return shutil.which("gh")


def _classify(stderr: str) -> str:
    s = (stderr or "").lower()
    if "could not resolve to a repository" in s or ("not found" in s and "repo" in s):
        return "repo_not_found"
    if "rate limit" in s or "secondary rate" in s:
        return "rate_limited"
    if "bad credentials" in s or "401" in s or "authentication" in s or "auth login" in s:
        return "auth_expired"
    if "could not resolve host" in s or "connection" in s or "timeout" in s or "network" in s:
        return "unreachable"
    return "error"


class GitHubAdapter(Adapter):
    mode = "poll"
    interval_s = 60

    # -- the CLI ------------------------------------------------------------

    async def _run(self, ctx, args, *, token=None, timeout=60):
        gh = _gh()
        if gh is None:
            raise CommandNotFound("gh")
        env = {"GH_PROMPT_DISABLED": "1", "NO_COLOR": "1", "GH_NO_UPDATE_NOTIFIER": "1"}
        if token:
            env["GH_TOKEN"] = token
            env["GH_HOST"] = HOST
        return await ctx.run([gh, *args], env=env, timeout=timeout)

    async def _token(self, ctx, login: str) -> str:
        """The account's token, or VerifyError(signed_out / cli_missing)."""
        try:
            res = await self._run(ctx, ["auth", "token", "--hostname", HOST, "--user", login], timeout=20)
        except CommandNotFound:
            raise VerifyError("cli_missing", "account", "the GitHub CLI (gh) is not installed")
        except CommandTimeout:
            raise VerifyError("unreachable", "account", "gh did not answer")
        token = res.stdout.strip()
        if res.code != 0 or not token:
            raise VerifyError("signed_out", "account", f"{login} is not signed in to gh")
        return token

    async def _json(self, ctx, args, token):
        try:
            res = await self._run(ctx, args, token=token)
        except CommandNotFound:
            raise SourceFailure("cli_missing", {"cli": "gh"})
        except CommandTimeout as t:
            raise SourceFailure("unreachable", {"host": HOST, "seconds": t.seconds})
        if res.code != 0:
            code = _classify(res.stderr)
            raise SourceFailure(code, {"host": HOST} if code == "unreachable" else {})
        try:
            return json.loads(res.stdout or "null")
        except ValueError:
            raise SourceFailure("error")

    # -- accounts -----------------------------------------------------------

    async def accounts(self) -> Accounts:
        from agent_os.sources.context import run_argv
        gh = _gh()
        if gh is None:
            return Accounts(state="not_installed")
        try:
            res = await run_argv([gh, "auth", "status", "--json", "hosts"], timeout=20,
                                 env={"GH_PROMPT_DISABLED": "1", "NO_COLOR": "1"})
        except (CommandNotFound, CommandTimeout):
            return Accounts(state="not_installed")
        accounts = []
        try:
            data = json.loads(res.stdout or "{}")
            for row in (data.get("hosts") or {}).get(HOST) or []:
                if row.get("login") and row.get("state", "success") == "success":
                    accounts.append({"login": row["login"], "host": HOST,
                                     "active": bool(row.get("active"))})
        except (ValueError, AttributeError):
            accounts = []
        accounts.sort(key=lambda a: (not a["active"], a["login"].lower()))
        return Accounts(state="signed_in" if accounts else "signed_out", accounts=accounts)

    # -- verify -------------------------------------------------------------

    async def verify(self, ctx) -> Verified:
        login = (ctx.account or "").strip()
        if not login:
            raise VerifyError("invalid", "account", "pick an account")
        token = await self._token(ctx, login)
        label = ctx.kinds.get("label")
        if label is not None:
            repo, name = label.get("repo"), label.get("label")
            res = await self._run(ctx, ["repo", "view", repo, "--json", "nameWithOwner"], token=token)
            if res.code != 0:
                raise VerifyError("not_found", "repo", f"cannot find {repo}")
            labels = await self._labels(ctx, repo, token)
            if name not in labels:
                raise VerifyError("not_found", "label", f"{repo} has no label {name}")
        return Verified(label=login)

    async def _labels(self, ctx, repo, token) -> list[str]:
        res = await self._run(ctx, ["label", "list", "-R", repo, "--limit", "300", "--json", "name"],
                              token=token)
        if res.code != 0:
            raise VerifyError("not_found", "repo", f"cannot find {repo}")
        try:
            return [r["name"] for r in json.loads(res.stdout or "[]") if r.get("name")]
        except (ValueError, TypeError, KeyError):
            return []

    # -- choices ------------------------------------------------------------

    async def choices(self, ctx, list_name, params):
        login = (ctx.account or "").strip()
        if not login:
            raise VerifyError("invalid", "account", "pick an account")
        token = await self._token(ctx, login)
        if list_name == "repos":
            res = await self._run(ctx, ["api", "--method", "GET", "user/repos", "-f", "per_page=100",
                                        "-f", "sort=pushed",
                                        "-f", "affiliation=owner,collaborator,organization_member"],
                                  token=token)
            if res.code != 0:
                raise VerifyError("error", None, "could not list repositories")
            try:
                rows = json.loads(res.stdout or "[]")
            except ValueError:
                rows = []
            return [Choice(r["full_name"]) for r in rows if isinstance(r, dict) and r.get("full_name")]
        if list_name == "labels":
            repo = (params or {}).get("repo")
            if not repo:
                raise VerifyError("invalid", "repo", "pick a repository first")
            return [Choice(n) for n in await self._labels(ctx, repo, token)]
        return []

    # -- poll ---------------------------------------------------------------

    async def poll(self, ctx) -> list:
        login = (ctx.account or "").strip()
        try:
            token = await self._token(ctx, login)
        except VerifyError as e:
            if e.reason == "cli_missing":
                raise SourceFailure("cli_missing", {"cli": "gh"})
            if e.reason == "unreachable":
                raise SourceFailure("unreachable", {"host": HOST})
            raise SourceFailure("cli_signed_out", {"cli": "gh", "account": login})
        out: list = []
        base = ["search", "issues", "--state", "open", "--include-prs", "--sort", "updated",
                "--limit", str(SEARCH_LIMIT), "--json", _FIELDS]
        for kind in ("assigned", "mention", "label"):
            params = ctx.kinds.get(kind)
            if params is None:
                continue
            if kind == "assigned":
                args = base + ["--assignee", "@me"]
            elif kind == "mention":
                args = base + ["--mentions", "@me"]
            else:
                repo, name = params.get("repo"), params.get("label")
                if not repo or not name:
                    continue
                if "," not in name:
                    out.extend(await self._label_rest(ctx, repo, name, token))
                    continue
                exists = await self._run(ctx, ["repo", "view", repo, "--json", "nameWithOwner"], token=token)
                if exists.code != 0:
                    code = _classify(exists.stderr)
                    if code in ("error", "repo_not_found"):
                        raise SourceFailure("repo_not_found", {"repo": repo})
                    raise SourceFailure(code)
                args = base + ["--repo", repo, "--label", name]
            rows = await self._json(ctx, args, token)
            for r in rows or []:
                item = self._item(kind, r)
                if item is not None:
                    out.append(item)
        return out

    async def _label_rest(self, ctx, repo: str, name: str, token: str) -> list:
        """Open issues and pull requests carrying the label, newest update
        first, from the repository's issue list. A 404 means the repository
        is gone (or no longer visible to the account)."""
        args = ["api", "--method", "GET", f"repos/{repo}/issues", "-f", f"labels={name}",
                "-f", "state=open", "-f", "sort=updated", "-f", "direction=desc",
                "-f", f"per_page={SEARCH_LIMIT}"]
        try:
            res = await self._run(ctx, args, token=token)
        except CommandNotFound:
            raise SourceFailure("cli_missing", {"cli": "gh"})
        except CommandTimeout as t:
            raise SourceFailure("unreachable", {"host": HOST, "seconds": t.seconds})
        if res.code != 0:
            if "http 404" in (res.stderr or "").lower():
                raise SourceFailure("repo_not_found", {"repo": repo})
            code = _classify(res.stderr)
            raise SourceFailure(code, {"host": HOST} if code == "unreachable" else {})
        try:
            rows = json.loads(res.stdout or "null")
        except ValueError:
            raise SourceFailure("error")
        out = []
        for r in rows if isinstance(rows, list) else []:
            item = self._item_rest("label", r, repo)
            if item is not None:
                out.append(item)
        return out

    @staticmethod
    def _make(kind: str, repo: str, n, title, body, is_pr: bool, url, sender, at) -> Item:
        subject = f"{repo}#{n}"
        return Item(
            id=f"{kind}:{subject}",
            kind=kind,
            subject=subject,
            title=f"#{n} {title or ''}".strip(),
            body=f"{'Pull request' if is_pr else 'Issue'} in {repo}\n\n" + (body or ""),
            url=url,
            sender=sender,
            at=at,
        )

    @classmethod
    def _item(cls, kind: str, r: dict) -> Item | None:
        """One row of ``gh search issues --json``."""
        if not isinstance(r, dict) or not r.get("number"):
            return None
        repo = (r.get("repository") or {}).get("nameWithOwner") or ""
        return cls._make(kind, repo, r["number"], r.get("title"), r.get("body"),
                         bool(r.get("isPullRequest")), r.get("url"),
                         (r.get("author") or {}).get("login"), r.get("updatedAt"))

    @classmethod
    def _item_rest(cls, kind: str, r: dict, repo: str) -> Item | None:
        """One row of the REST issue list. The repository's name comes from
        the row's ``repository_url`` (GitHub's own spelling, as search gives
        it), so the ids match the search path's exactly; the requested name
        is the fallback."""
        if not isinstance(r, dict) or not r.get("number"):
            return None
        api_repo = r.get("repository_url") or ""
        marker = "/repos/"
        canonical = api_repo.split(marker, 1)[1] if marker in api_repo else ""
        return cls._make(kind, canonical or repo, r["number"], r.get("title"), r.get("body"),
                         r.get("pull_request") is not None, r.get("html_url"),
                         (r.get("user") or {}).get("login"), r.get("updated_at"))


ADAPTER = GitHubAdapter
