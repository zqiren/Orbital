# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Website check: an address and, optionally, text the page must contain.

A state-change source. Going down opens one item (one task); staying down
is silent; coming back resolves that same item (history shows 已恢复, no new
task). "Down" = no answer, an HTTP status of 400 or more, or the page
without the required text — each confirmed by one retry a few seconds later
inside the same check, so a single dropped request never opens a task.

The up/down state lives in the source's adapter state, so a daemon restart
neither re-opens an incident nor forgets one. Unlike polled account types
there is no baseline: a site that is down at the first check opens a task.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlparse

from agent_os.sources.adapters import Adapter
from agent_os.sources.model import Item, Resolve, Verified, VerifyError

READ_CAP = 2_000_000
TIMEOUT_S = 15.0
RETRY_DELAY_S = 5.0


def normalise_url(raw: str) -> str:
    s = (raw or "").strip()
    if s and "://" not in s:
        s = "https://" + s
    return s


class WebsiteAdapter(Adapter):
    mode = "poll"
    interval_s = 120
    baseline = False
    retry_delay_s = RETRY_DELAY_S

    async def verify(self, ctx) -> Verified:
        url = normalise_url(ctx.config.get("url"))
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise VerifyError("invalid", "url", "enter a web address")
        return Verified(label=parsed.hostname, config={"url": url})

    async def _probe(self, url: str, must: str) -> str | None:
        """None when the page is fine, else a short reason."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_S, follow_redirects=True,
                                         headers={"User-Agent": "Orbital-website-check"}) as client:
                async with client.stream("GET", url) as resp:
                    status = resp.status_code
                    body = b""
                    if must:
                        async for chunk in resp.aiter_bytes():
                            body += chunk
                            if len(body) >= READ_CAP:
                                break
        except httpx.HTTPError as e:
            return f"no answer ({type(e).__name__})"
        if status >= 400:
            return f"HTTP {status}"
        if must and must not in body.decode(resp.encoding or "utf-8", errors="replace"):
            return f"the page no longer contains “{must}”"
        return None

    async def poll(self, ctx) -> list:
        url = normalise_url(ctx.config.get("url"))
        must = (ctx.config.get("must_contain") or "").strip()
        reason = await self._probe(url, must)
        if reason is not None and self.retry_delay_s:
            await asyncio.sleep(self.retry_delay_s)
            reason = await self._probe(url, must)
        st = ctx.state
        host = urlparse(url).hostname or url
        now = ctx.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        if reason is not None:
            if st.get("down_since"):
                return []  # still down: silent
            st["down_since"] = now
            subject = f"down:{now}"
            st["subject"] = subject
            title = f"{host} 打不开" if ctx.lang == "zh" else f"{host} is down"
            return [Item(id=subject, kind="down", subject=subject, title=title,
                         body=f"Address: {url}\nWhat failed: {reason}\nSince: {now}", url=url)]
        if st.get("down_since"):
            subject = st.get("subject") or f"down:{st['down_since']}"
            st.pop("down_since", None)
            st.pop("subject", None)
            return [Resolve(subject=subject, at=now)]
        return []


ADAPTER = WebsiteAdapter
