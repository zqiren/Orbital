# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Jira Cloud, with the user's site address, login e-mail and API token
(basic auth, as Atlassian documents for scripts).

- verify: ``GET /rest/api/3/myself``.
- poll: the enhanced search ``POST /rest/api/3/search/jql`` (the classic
  ``/rest/api/3/search`` was removed from Jira Cloud), newest-updated first,
  over a recent window:
  - ``assigned``: ``assignee = currentUser() AND statusCategory != Done AND
    updated >= -14d``. Id ``assigned:<KEY>``.
  - ``mention``: JQL cannot search mentions (JRACLOUD-27594), so it asks for
    candidates (``comment ~ currentUser() OR description ~ currentUser()``,
    falling back to every issue updated in the last day if the site refuses
    that JQL) and keeps only comments / descriptions whose document really
    holds a mention node of this account. Id ``mention:<KEY>:<comment id>``
    (or ``…:description``).
- Items of one issue share the subject ``<KEY>``, so an assignment and a
  mention of the same issue merge into one task while it is open.

The token is only ever in the request's Authorization header; it is never
logged or put in a message.
"""

from __future__ import annotations

import ipaddress
from datetime import datetime, timezone
from urllib.parse import urlparse

from agent_os.sources.adapters import Adapter
from agent_os.sources.model import Item, SourceFailure, Verified, VerifyError

TIMEOUT_S = 20.0
MAX_RESULTS = 50
COMMENT_PAGE = 50
ASSIGNED_JQL = ("assignee = currentUser() AND statusCategory != Done AND updated >= -14d "
                "ORDER BY updated DESC")
MENTION_JQL = ("(comment ~ currentUser() OR description ~ currentUser()) AND updated >= -7d "
               "ORDER BY updated DESC")
MENTION_FALLBACK_JQL = "updated >= -1d ORDER BY updated DESC"
_FIELDS = ["summary", "status", "updated", "reporter", "description"]


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def normalise_site(raw: str) -> tuple[str, str] | None:
    """What the user typed → (base URL, host label), or None. ``acme`` →
    ``acme.atlassian.net``; scheme and path are dropped; https always
    (plain http only for a server on this computer)."""
    s = (raw or "").strip()
    if not s:
        return None
    if "://" not in s:
        s = "https://" + s
    p = urlparse(s)
    host = (p.hostname or "").lower()
    if not host:
        return None
    if "." not in host and not _is_loopback(host):
        host += ".atlassian.net"
    netloc = host if p.port is None else f"{host}:{p.port}"
    if ":" in host:  # IPv6 literal
        netloc = f"[{host}]" if p.port is None else f"[{host}]:{p.port}"
    scheme = "http" if _is_loopback(host) and p.scheme == "http" else "https"
    return f"{scheme}://{netloc}", netloc


def _iso(value) -> str | None:
    """Jira's ``2026-11-16T07:20:00.000+0000`` → ``2026-11-16T07:20:00Z``."""
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f%z")
    except ValueError:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def adf_text(node) -> str:
    """Atlassian Document Format → plain text (a v2 string passes through)."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(adf_text(n) for n in node)
    if not isinstance(node, dict):
        return ""
    t = node.get("type")
    if t == "text":
        return node.get("text") or ""
    if t == "mention":
        return (node.get("attrs") or {}).get("text") or "@someone"
    if t in ("emoji",):
        return (node.get("attrs") or {}).get("text") or ""
    if t == "hardBreak":
        return "\n"
    inner = adf_text(node.get("content") or [])
    if t in ("paragraph", "heading", "blockquote", "codeBlock", "listItem", "rule", "panel"):
        return inner.rstrip("\n") + "\n"
    return inner


def mentions(node, account_id: str) -> bool:
    if not account_id or node is None:
        return False
    if isinstance(node, str):
        return f"accountid:{account_id}" in node.lower() or f"[~{account_id}]" in node
    if isinstance(node, list):
        return any(mentions(n, account_id) for n in node)
    if not isinstance(node, dict):
        return False
    if node.get("type") == "mention" and (node.get("attrs") or {}).get("id") == account_id:
        return True
    return mentions(node.get("content"), account_id)


class _Refused(Exception):
    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status = status


class JiraAdapter(Adapter):
    mode = "poll"
    interval_s = 300

    def _client(self, ctx):
        import httpx
        site = normalise_site(ctx.config.get("site"))
        if site is None:
            raise VerifyError("invalid", "site", "enter your Jira site address")
        base, _label = site
        mail = (ctx.config.get("mail") or "").strip()
        token = ctx.secret("token") or ""
        return httpx.AsyncClient(base_url=base, auth=(mail, token), timeout=TIMEOUT_S,
                                 follow_redirects=False,
                                 headers={"Accept": "application/json",
                                          "User-Agent": "Orbital-task-source"})

    # -- verify -------------------------------------------------------------

    async def verify(self, ctx) -> Verified:
        import httpx
        site = normalise_site(ctx.config.get("site"))
        if site is None:
            raise VerifyError("invalid", "site", "enter your Jira site address")
        if "@" not in (ctx.config.get("mail") or ""):
            raise VerifyError("invalid", "mail", "enter the e-mail you sign in to Jira with")
        if not ctx.secret("token"):
            raise VerifyError("invalid", "token", "paste an API token")
        _base, label = site
        try:
            async with self._client(ctx) as client:
                resp = await client.get("/rest/api/3/myself")
        except httpx.ConnectError as e:
            if _is_dns_failure(e):
                raise VerifyError("not_found", "site", f"cannot find {label}")
            raise VerifyError("unreachable", "site", f"{label} did not answer")
        except httpx.TimeoutException:
            raise VerifyError("unreachable", "site", f"{label} did not answer")
        except httpx.HTTPError:
            raise VerifyError("unreachable", "site", f"{label} did not answer")
        st = resp.status_code
        if st == 401:
            raise VerifyError("rejected", "token", "Jira refused this e-mail and API token")
        if st == 403:
            if "denied" in (resp.headers.get("X-Seraph-LoginReason") or "").lower():
                raise VerifyError("rejected", "mail",
                                  "Jira wants a browser sign-in first (too many failed attempts)")
            raise VerifyError("rejected", "token", "this account may not use the Jira API")
        if st == 429:
            raise VerifyError("unreachable", "site", "Jira is limiting requests; try again shortly")
        if st >= 500:
            raise VerifyError("unreachable", "site", f"{label} answered with an error ({st})")
        if st != 200:
            raise VerifyError("not_found", "site", f"{label} is not a Jira Cloud site")
        try:
            me = resp.json()
        except ValueError:
            me = None
        if not isinstance(me, dict) or not me.get("accountId"):
            raise VerifyError("not_found", "site", f"{label} is not a Jira Cloud site")
        base, _ = site
        return Verified(label=label, config={"site": base if base.startswith("http://") else label})

    # -- poll ---------------------------------------------------------------

    async def _call(self, client, method, path, **kw):
        import httpx
        try:
            resp = await client.request(method, path, **kw)
        except (httpx.TimeoutException, httpx.TransportError):
            raise SourceFailure("unreachable", {"host": str(client.base_url.host)})
        st = resp.status_code
        if st == 401:
            raise SourceFailure("token_rejected", {"field": "token"}, retryable=False)
        if st == 403:
            raise SourceFailure("auth_expired", {})
        if st == 429:
            raise SourceFailure("rate_limited", {})
        if st >= 500:
            raise SourceFailure("unreachable", {"host": str(client.base_url.host)})
        if st != 200:
            raise _Refused(st)
        try:
            return resp.json()
        except ValueError:
            raise SourceFailure("error")

    async def _search(self, client, jql: str, fields: list[str]) -> list[dict]:
        data = await self._call(client, "POST", "/rest/api/3/search/jql",
                                json={"jql": jql, "maxResults": MAX_RESULTS, "fields": fields})
        issues = data.get("issues") if isinstance(data, dict) else None
        return [i for i in issues or [] if isinstance(i, dict) and i.get("key")]

    async def _me(self, ctx, client) -> str:
        key = f"{(ctx.config.get('site') or '').lower()}|{(ctx.config.get('mail') or '').lower()}"
        me = ctx.state.get("me") if isinstance(ctx.state.get("me"), dict) else {}
        if me.get("key") == key and me.get("id"):
            return me["id"]
        try:
            data = await self._call(client, "GET", "/rest/api/3/myself")
        except _Refused:
            raise SourceFailure("error")
        account_id = data.get("accountId") if isinstance(data, dict) else None
        if not account_id:
            raise SourceFailure("error")
        ctx.state["me"] = {"key": key, "id": account_id}
        return account_id

    async def poll(self, ctx) -> list:
        site = normalise_site(ctx.config.get("site"))
        if site is None:
            raise SourceFailure("error")
        base, _ = site
        out: list = []
        try:
            async with self._client(ctx) as client:
                me = await self._me(ctx, client)
                if "assigned" in ctx.kinds:
                    for issue in await self._search(client, ASSIGNED_JQL, _FIELDS):
                        out.append(self._issue_item("assigned", issue, base))
                if "mention" in ctx.kinds:
                    out.extend(await self._mentions(client, base, me))
        except _Refused:
            raise SourceFailure("error")
        return out

    async def _mentions(self, client, base: str, me: str) -> list:
        fields = _FIELDS + ["comment"]
        try:
            issues = await self._search(client, MENTION_JQL, fields)
        except _Refused as r:
            if r.status != 400:
                raise
            issues = await self._search(client, MENTION_FALLBACK_JQL, fields)
        out = []
        for issue in issues:
            key = issue["key"]
            f = issue.get("fields") or {}
            if mentions(f.get("description"), me):
                out.append(self._mention_item(issue, base, None, f.get("description"),
                                              None, _iso(f.get("updated"))))
            block = f.get("comment") if isinstance(f.get("comment"), dict) else {}
            comments = [c for c in block.get("comments") or [] if isinstance(c, dict)]
            total = block.get("total")
            if isinstance(total, int) and total > len(comments):
                try:
                    data = await self._call(client, "GET", f"/rest/api/3/issue/{key}/comment",
                                            params={"orderBy": "-created", "maxResults": COMMENT_PAGE})
                    comments = [c for c in (data or {}).get("comments") or [] if isinstance(c, dict)]
                except _Refused:
                    pass
            for c in comments:
                author = c.get("author") if isinstance(c.get("author"), dict) else {}
                if author.get("accountId") == me or not c.get("id"):
                    continue
                if mentions(c.get("body"), me):
                    out.append(self._mention_item(issue, base, str(c["id"]), c.get("body"),
                                                  author.get("displayName"), _iso(c.get("created"))))
        return out

    @staticmethod
    def _issue_item(kind: str, issue: dict, base: str) -> Item:
        key = issue["key"]
        f = issue.get("fields") or {}
        status = (f.get("status") or {}).get("name") if isinstance(f.get("status"), dict) else None
        reporter = (f.get("reporter") or {}).get("displayName") if isinstance(f.get("reporter"), dict) else None
        lines = [f"Jira issue {key}"]
        if status:
            lines.append(f"Status: {status}")
        if reporter:
            lines.append(f"Reporter: {reporter}")
        desc = adf_text(f.get("description")).strip()
        if desc:
            lines += ["", desc]
        return Item(id=f"{kind}:{key}", kind=kind, subject=key,
                    title=f"{key} {f.get('summary') or ''}".strip(), body="\n".join(lines),
                    url=f"{base}/browse/{key}", sender=reporter, at=_iso(f.get("updated")))

    @staticmethod
    def _mention_item(issue, base, comment_id, doc, author, at) -> Item:
        key = issue["key"]
        f = issue.get("fields") or {}
        where = "a comment" if comment_id else "the description"
        lines = [f"Jira issue {key}: you were mentioned in {where}"
                 + (f" by {author}" if author else ""), "", adf_text(doc).strip()]
        url = f"{base}/browse/{key}"
        if comment_id:
            url += f"?focusedCommentId={comment_id}"
        return Item(id=f"mention:{key}:{comment_id or 'description'}", kind="mention", subject=key,
                    title=f"{key} {f.get('summary') or ''}".strip(), body="\n".join(lines),
                    url=url, sender=author, at=at)


def _is_dns_failure(e: Exception) -> bool:
    s = str(e).lower()
    return any(m in s for m in ("name or service not known", "nodename nor servname", "getaddrinfo",
                                "no address associated", "name resolution", "errno 8", "11001"))


ADAPTER = JiraAdapter
