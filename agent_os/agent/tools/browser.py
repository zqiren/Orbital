# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""BrowserTool — async browser automation tool with 23 action handlers.

Dispatches browser actions via Playwright through BrowserManager,
translates errors to human-friendly messages, handles batching,
captures screenshots, and wraps results with safety.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import asdict
from urllib.parse import quote_plus as _url_encode

from .base import Tool, ToolResult
from .browser_refs import serialize_snapshot, resolve_ref, RefMap, SnapshotStats
from .browser_safety import (
    validate_url_pre_navigation,
    validate_url_post_navigation,
    wrap_untrusted_content,
    detect_secrets,
    substitute_secrets,
)

logger = logging.getLogger(__name__)


async def _get_ax_tree(page) -> dict | None:
    """Get the accessibility tree as a dict using CDP.

    Playwright >= 1.49 removed page.accessibility.snapshot(). This function
    uses the Chrome DevTools Protocol to retrieve the full accessibility tree
    and convert it into the same dict structure that serialize_snapshot expects.
    """
    try:
        client = await page.context.new_cdp_session(page)
        result = await client.send("Accessibility.getFullAXTree")
        await client.detach()
    except Exception:
        return None

    nodes = result.get("nodes", [])
    if not nodes:
        return None

    node_map = {n["nodeId"]: n for n in nodes}

    def _build(node_id: str) -> dict | None:
        n = node_map.get(node_id)
        if not n:
            return None
        if n.get("ignored"):
            children = []
            for cid in n.get("childIds", []):
                child = _build(cid)
                if child:
                    children.append(child)
            if len(children) == 1:
                return children[0]
            if children:
                return {"role": "generic", "children": children}
            return None

        role_val = n.get("role", {}).get("value", "")
        # Skip internal Chrome text nodes
        if role_val in ("StaticText", "InlineTextBox"):
            return None

        entry: dict = {
            "role": role_val,
            "name": n.get("name", {}).get("value") or None,
        }
        props = {}
        for p in n.get("properties", []):
            pval = p.get("value", {})
            if isinstance(pval, dict):
                props[p["name"]] = pval.get("value", "")
            else:
                props[p["name"]] = pval
        if "level" in props:
            entry["level"] = props["level"]
        if "checked" in props:
            entry["checked"] = props["checked"] == "true"
        if "pressed" in props:
            entry["pressed"] = props["pressed"] == "true"
        if "value" in props:
            entry["value"] = props["value"]

        children = []
        for cid in n.get("childIds", []):
            child = _build(cid)
            if child:
                children.append(child)
        if children:
            entry["children"] = children
        return entry

    root = _build(nodes[0]["nodeId"]) if nodes else None
    return root


BROWSER_WRITE_ACTIONS = frozenset({
    "click", "type", "fill", "press", "hover", "select",
    "drag", "upload_file",
})

BROWSER_OBSERVATION_ACTIONS = frozenset({
    "snapshot", "screenshot", "extract", "search_page", "wait",
    "go_back", "go_forward", "scroll", "navigate", "reload",
    "tab_new", "tab_switch", "tab_close", "done", "pdf",
    "search", "fetch",
})

_EXTRACT_MAX_CHARS = 10_000
_WAIT_TIMEOUT_MS = 30_000
BROWSER_ACTION_TIMEOUT = 60

# JS to extract Google search results.
# NOTE: Google changes DOM structure regularly. If extraction returns empty,
# the agent can fall back to navigate → snapshot → read accessibility tree.
_SEARCH_EXTRACT_JS = """
() => {
    const results = [];
    // Google organic results
    const items = document.querySelectorAll('div.g');
    for (const item of items) {
        const titleEl = item.querySelector('h3');
        const linkEl = item.querySelector('a[href]');
        const snippetEl = item.querySelector('[data-sncf], .VwiC3b, [style*="-webkit-line-clamp"]');
        if (titleEl && linkEl) {
            results.push({
                title: titleEl.textContent || '',
                url: linkEl.href || '',
                snippet: snippetEl ? snippetEl.textContent || '' : '',
            });
        }
    }
    return results.slice(0, 10);
}
"""

# JS to extract readable text content from a page.
_FETCH_EXTRACT_JS = """
() => {
    // Remove non-content elements
    const remove = ['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript'];
    for (const tag of remove) {
        document.querySelectorAll(tag).forEach(el => el.remove());
    }
    // Try article/main first, fall back to body
    const main = document.querySelector('article') || document.querySelector('main') || document.body;
    return main ? main.innerText : '';
}
"""


class BrowserTool(Tool):
    """Async browser automation tool with flat action-discriminated schema."""

    name = "browser"
    is_async = True
    description = (
        "Full browser automation. Call snapshot to see the page, then target elements by ref.\n\n"
        "IMPORTANT: If a browser action returns errors, empty results, or timeouts on 2+ "
        "consecutive attempts with different inputs, the action is likely broken (e.g., site "
        "DOM changed, site blocking automation, network issue). Do NOT keep retrying the same "
        "action type. Instead:\n"
        "- For search: results will auto-fallback to page snapshot. If that also fails, "
        "try navigating directly to known URLs.\n"
        "- For click/interact: try a different element, selector, or approach.\n"
        "- For navigate: try an alternative URL or skip this site entirely.\n"
        "- If a site appears to be blocking you: report to the user and move on to "
        "alternative sources."
        "\n\nIf clicking, typing, or other element interactions fail 3+ times, the element "
        "is likely not interactive (disabled, hidden, or dynamically removed). Do NOT keep "
        "retrying the same element. Switch strategy: navigate directly to URLs, use search, "
        "use fetch, or try different elements on the page."
    )

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "navigate", "click", "type", "fill", "press", "hover",
                    "select", "scroll", "drag", "upload_file", "snapshot",
                    "screenshot", "extract", "search_page", "evaluate",
                    "go_back", "go_forward", "reload", "wait", "pdf",
                    "tab_new", "tab_switch", "tab_close", "done",
                    "search", "fetch",
                ],
            },
            "url": {"type": "string"},
            "query": {"type": "string", "description": "Search query (for search action)"},
            "ref": {"type": "string", "description": "Element ref from snapshot (e.g. 'e5')"},
            "text": {"type": "string"},
            "key": {"type": "string"},
            "value": {"type": "string"},
            "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
            "amount": {"type": "integer"},
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"ref": {"type": "string"}, "value": {"type": "string"}},
                    "required": ["ref", "value"],
                },
            },
            "start_ref": {"type": "string"},
            "end_ref": {"type": "string"},
            "file_path": {"type": "string"},
            "javascript": {"type": "string"},
            "seconds": {"type": "number"},
            "text_gone": {"type": "string", "description": "Wait for text to disappear"},
            "selector": {"type": "string", "description": "Wait for CSS selector to appear"},
            "url_pattern": {"type": "string", "description": "Wait for URL to match pattern"},
            "load_state": {
                "type": "string",
                "enum": ["load", "domcontentloaded", "networkidle"],
                "description": "Wait for page load state",
            },
            "tab_id": {"type": "string"},
            "interactive_only": {"type": "boolean"},
            "annotate": {"type": "boolean"},
            "actions": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Batch: array of action objects executed sequentially",
            },
        },
        "required": ["action"],
    }

    def __init__(
        self,
        browser_manager,
        project_id: str,
        workspace: str,
        autonomy_preset: str,
        session_id: str = "default",
        user_credential_store=None,
        vision_enabled: bool = False,
        project_dir_name: str = "",
    ):
        self._bm = browser_manager
        self._project_id = project_id
        self._workspace = workspace
        self._autonomy_preset = autonomy_preset
        self._session_id = session_id
        self._vision_enabled = vision_enabled
        self._project_dir_name = project_dir_name
        self._resolver = self._make_resolver(user_credential_store)
        self._action_failure_tracker: dict[str, list[str]] = {}

    def on_run_start(self) -> None:
        """Reset per-run state. Called by ToolRegistry.reset_run_state()."""
        self._action_failure_tracker = {}

    def _track_result(self, action: str, result: ToolResult) -> ToolResult:
        """Track consecutive failures per action and append advisory note at threshold."""
        is_success = self._is_successful_result(action, result)

        if is_success:
            self._action_failure_tracker.pop(action, None)
            return result

        tracker = self._action_failure_tracker.setdefault(action, [])
        summary = self._summarize_result(action, result)
        tracker.append(summary)

        consecutive = len(tracker)

        # HARD REFUSAL at 5+ failures — stop executing this action type
        if consecutive >= 5 and self._is_element_action(action):
            recent = "; ".join(tracker[-3:])
            self._action_failure_tracker.pop(action, None)  # Reset so agent can try later if strategy changes
            return ToolResult(
                content=(
                    f"BLOCKED: browser:{action} has failed {consecutive} consecutive times. "
                    f"This action is not working on this page. Recent failures: {recent}\n\n"
                    f"Do NOT retry {action} on this page. Instead:\n"
                    f"- Navigate directly to the target URL if you know it\n"
                    f"- Use search to find the content via a different route\n"
                    f"- Use fetch to read the page content as text\n"
                    f"- Report to the user that this content is not accessible via browser interaction\n\n"
                    f"If you need to interact with this page, try scroll to find different elements, "
                    f"or navigate away and come back."
                ),
                meta={"blocked_action": action, "failure_count": consecutive},
            )

        # Existing advisory at 3+ failures (kept as early warning)
        if consecutive >= 3:
            recent = "; ".join(tracker[-3:])
            content = result.content if isinstance(result.content, str) else str(result.content)
            result = ToolResult(
                content=content + (
                    f"\n\n[Note: browser:{action} has failed {consecutive} consecutive "
                    f"times this session. Recent results: {recent}. "
                    f"This action may not be working. Consider a different approach.]"
                ),
                meta=result.meta,
            )

        return result

    @staticmethod
    def _is_successful_result(action: str, result: ToolResult) -> bool:
        """Determine if a tool result represents success for the given action type."""
        content = result.content if isinstance(result.content, str) else str(result.content)
        meta = result.meta or {}

        if content.startswith("Error"):
            return False
        if "timed out" in content.lower():
            return False

        if action == "search":
            if meta.get("search_fallback"):
                return bool(content and len(content) > 100)
            if meta.get("result_count", 0) > 0:
                return True
            if "No results found" in content:
                return False
            return True

        if action == "navigate":
            return "Error" not in content
        if action in ("click", "type", "fill", "press", "select"):
            return "not found" not in content.lower() and "Error" not in content
        if action in ("snapshot", "screenshot"):
            return bool(content) and "empty" not in content.lower()

        return "Error" not in content

    @staticmethod
    def _summarize_result(action: str, result: ToolResult) -> str:
        """Extract a short failure summary from the result."""
        content = result.content if isinstance(result.content, str) else str(result.content)
        first_line = content.split("\n")[0][:80]
        return first_line

    @staticmethod
    def _is_element_action(action: str) -> bool:
        """Actions that target specific DOM elements."""
        return action in (
            "click", "type", "fill", "press", "hover",
            "select", "drag", "check", "uncheck", "upload_file",
        )

    @staticmethod
    def _make_resolver(store):
        """Build a callable that resolves secret keys like 'gmail.email' from the credential store."""
        if store is None:
            def _noop(key: str) -> str | None:
                return None
            return _noop

        def _resolve(key: str) -> str | None:
            if "." in key:
                name, field = key.split(".", 1)
                return store.get_value(name, field)
            return None
        return _resolve

    async def _collect_page_signals(self, page, requested_url: str | None = None) -> dict:
        """Collect objective page signals after navigation/snapshot."""
        signals = {}
        try:
            # Has password field
            signals["has_password_field"] = await page.evaluate(
                '!!document.querySelector("input[type=password]:not([hidden])")'
            )
            # Captcha detection
            captcha = await page.evaluate('''() => {
                const frames = document.querySelectorAll("iframe");
                for (const f of frames) {
                    const src = (f.src || "").toLowerCase();
                    if (src.includes("recaptcha")) return "recaptcha";
                    if (src.includes("hcaptcha")) return "hcaptcha";
                    if (src.includes("turnstile")) return "turnstile";
                }
                return null;
            }''')
            signals["has_captcha_iframe"] = captcha
            # Visible text snippet (first 200 chars)
            signals["visible_text_snippet"] = (await page.evaluate(
                '(document.body && document.body.innerText || "").substring(0, 200)'
            )) or ""
            # Input and form counts
            signals["input_count"] = await page.evaluate(
                'document.querySelectorAll("input:not([type=hidden])").length'
            )
            signals["form_count"] = await page.evaluate(
                'document.querySelectorAll("form").length'
            )
            # Redirect detection
            current_url = page.url
            if requested_url:
                from urllib.parse import urlparse
                req_host = urlparse(requested_url).netloc
                cur_host = urlparse(current_url).netloc
                signals["was_redirected"] = req_host != cur_host or current_url != requested_url
                signals["redirected_from"] = requested_url if signals["was_redirected"] else None
            else:
                signals["was_redirected"] = False
                signals["redirected_from"] = None
            # HTTP status (not reliably available after page load, set None)
            signals["http_status"] = None
        except Exception:
            # If page is crashed or navigating, return partial signals
            pass
        return signals

    @staticmethod
    async def _wait_for_stable(page, timeout_ms: int = 5000) -> None:
        """Wait for networkidle to let JS-rendered content settle.

        Capped by timeout to avoid hanging on sites with persistent
        connections (WebSockets, analytics, long-polling).
        """
        try:
            await page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except Exception:
            pass  # partial page is better than failing

    async def execute(self, **arguments) -> ToolResult:
        action = arguments.get("action", "")
        if action == "batch":
            return await self._execute_batch(arguments.get("actions", []))
        return await self._dispatch(action, arguments)

    # Alias so callers using the old name still work
    execute_async = execute

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, action: str, args: dict) -> ToolResult:
        try:
            handler = getattr(self, f"_action_{action}", None)
            if handler is None:
                return ToolResult(content=f"Unknown browser action: {action}")
            result = await asyncio.wait_for(handler(args), timeout=BROWSER_ACTION_TIMEOUT)
            return self._track_result(action, result)
        except asyncio.TimeoutError:
            result = ToolResult(
                content=f"Browser action '{action}' timed out after {BROWSER_ACTION_TIMEOUT}s. "
                        f"The page may be unresponsive or waiting for user interaction.",
                meta={"error": "timeout", "action": action}
            )
            return self._track_result(action, result)
        except Exception as e:
            result = self._translate_error(e, action, args)
            return self._track_result(action, result)

    # ------------------------------------------------------------------
    # Navigation actions
    # ------------------------------------------------------------------

    async def _action_navigate(self, args: dict) -> ToolResult:
        url = args.get("url", "")
        error = validate_url_pre_navigation(url)
        if error:
            return ToolResult(content=error)

        page = await self._bm.get_page(self._project_id)
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await self._wait_for_stable(page, timeout_ms=10000)

        final_url = page.url
        error = validate_url_post_navigation(final_url)
        if error:
            await page.goto("about:blank")
            return ToolResult(content=error)

        self._bm.clear_ref_map(self._project_id, id(page))
        title = await page.title()
        screenshot_path = await self._bm.capture_screenshot(page, self._workspace, self._session_id, project_dir_name=self._project_dir_name)
        signals = await self._collect_page_signals(page, url)

        # Blocker detection: flag pages requiring authentication or verification
        is_blocker = bool(
            signals.get("has_password_field") or signals.get("has_captcha_iframe")
        )
        if is_blocker:
            return ToolResult(
                content=f"Navigated to: {title} ({final_url}) — ⚠ Page appears to require authentication or verification.",
                meta={"url": final_url, "title": title, "screenshot_path": screenshot_path,
                      "page_signals": signals, "blocker_detected": True},
            )

        return ToolResult(
            content=f"Navigated to: {title} ({final_url})",
            meta={"url": final_url, "title": title, "screenshot_path": screenshot_path,
                  "page_signals": signals},
        )

    async def _action_go_back(self, args: dict) -> ToolResult:
        page = await self._bm.get_page(self._project_id)
        await page.go_back()
        await self._wait_for_stable(page, timeout_ms=10000)
        self._bm.clear_ref_map(self._project_id, id(page))
        url = page.url
        title = await page.title()
        screenshot_path = await self._bm.capture_screenshot(page, self._workspace, self._session_id, project_dir_name=self._project_dir_name)
        return ToolResult(
            content=f"Went back to: {title} ({url})",
            meta={"url": url, "title": title, "screenshot_path": screenshot_path},
        )

    async def _action_go_forward(self, args: dict) -> ToolResult:
        page = await self._bm.get_page(self._project_id)
        await page.go_forward()
        await self._wait_for_stable(page, timeout_ms=10000)
        self._bm.clear_ref_map(self._project_id, id(page))
        url = page.url
        title = await page.title()
        screenshot_path = await self._bm.capture_screenshot(page, self._workspace, self._session_id, project_dir_name=self._project_dir_name)
        return ToolResult(
            content=f"Went forward to: {title} ({url})",
            meta={"url": url, "title": title, "screenshot_path": screenshot_path},
        )

    async def _action_reload(self, args: dict) -> ToolResult:
        page = await self._bm.get_page(self._project_id)
        await page.reload()
        await self._wait_for_stable(page, timeout_ms=10000)
        self._bm.clear_ref_map(self._project_id, id(page))
        url = page.url
        title = await page.title()
        screenshot_path = await self._bm.capture_screenshot(page, self._workspace, self._session_id, project_dir_name=self._project_dir_name)
        return ToolResult(
            content=f"Reloaded: {title} ({url})",
            meta={"url": url, "title": title, "screenshot_path": screenshot_path},
        )

    # ------------------------------------------------------------------
    # Interaction actions
    # ------------------------------------------------------------------

    async def _action_click(self, args: dict) -> ToolResult:
        page = await self._bm.get_page(self._project_id)
        ref_map = self._bm.get_ref_map(self._project_id, id(page))
        if not ref_map:
            return ToolResult(content="No snapshot taken yet. Run snapshot first to see the page.")
        locator = await resolve_ref(ref_map, args.get("ref", ""), page)
        if args.get("double_click"):
            await locator.dblclick()
        else:
            await locator.click()
        await self._wait_for_stable(page)
        screenshot_path = await self._bm.capture_screenshot(page, self._workspace, self._session_id, project_dir_name=self._project_dir_name)
        return ToolResult(
            content=f"Clicked element ref={args.get('ref', '')}",
            meta={"url": page.url, "title": await page.title(), "screenshot_path": screenshot_path},
        )

    async def _action_type(self, args: dict) -> ToolResult:
        page = await self._bm.get_page(self._project_id)
        ref_map = self._bm.get_ref_map(self._project_id, id(page))
        if not ref_map:
            return ToolResult(content="No snapshot taken yet. Run snapshot first to see the page.")
        locator = await resolve_ref(ref_map, args.get("ref", ""), page)
        text = args.get("text", "")
        display_text = text
        if detect_secrets(text):
            text = substitute_secrets(text, self._resolver)
        await locator.fill(text)
        await self._wait_for_stable(page)
        screenshot_path = await self._bm.capture_screenshot(page, self._workspace, self._session_id, project_dir_name=self._project_dir_name)
        return ToolResult(
            content=f"Typed '{display_text}' into element",
            meta={"url": page.url, "title": await page.title(), "screenshot_path": screenshot_path},
        )

    async def _action_fill(self, args: dict) -> ToolResult:
        page = await self._bm.get_page(self._project_id)
        ref_map = self._bm.get_ref_map(self._project_id, id(page))
        if not ref_map:
            return ToolResult(content="No snapshot taken yet. Run snapshot first.")
        fields = args.get("fields", [])
        results = []
        for field in fields:
            try:
                locator = await resolve_ref(ref_map, field["ref"], page)
                value = field["value"]
                display_value = value
                if detect_secrets(value):
                    value = substitute_secrets(value, self._resolver)
                await locator.fill(value)
                results.append(f"  {field['ref']}: filled with '{display_value}'")
            except Exception as e:
                results.append(
                    f"  {field['ref']}: ERROR — "
                    f"{self._translate_error_message(e, {'ref': field['ref']})}"
                )
        await self._wait_for_stable(page)
        screenshot_path = await self._bm.capture_screenshot(page, self._workspace, self._session_id, project_dir_name=self._project_dir_name)
        return ToolResult(
            content=f"Filled {len(fields)} fields:\n" + "\n".join(results),
            meta={"url": page.url, "title": await page.title(), "screenshot_path": screenshot_path},
        )

    async def _action_press(self, args: dict) -> ToolResult:
        page = await self._bm.get_page(self._project_id)
        key = args.get("key", "")
        await page.keyboard.press(key)
        await self._wait_for_stable(page)
        screenshot_path = await self._bm.capture_screenshot(page, self._workspace, self._session_id, project_dir_name=self._project_dir_name)
        return ToolResult(
            content=f"Pressed key: {key}",
            meta={"url": page.url, "title": await page.title(), "screenshot_path": screenshot_path},
        )

    async def _action_hover(self, args: dict) -> ToolResult:
        page = await self._bm.get_page(self._project_id)
        ref_map = self._bm.get_ref_map(self._project_id, id(page))
        if not ref_map:
            return ToolResult(content="No snapshot taken yet. Run snapshot first to see the page.")
        locator = await resolve_ref(ref_map, args.get("ref", ""), page)
        await locator.hover()
        screenshot_path = await self._bm.capture_screenshot(page, self._workspace, self._session_id, project_dir_name=self._project_dir_name)
        return ToolResult(
            content=f"Hovered over element ref={args.get('ref', '')}",
            meta={"url": page.url, "title": await page.title(), "screenshot_path": screenshot_path},
        )

    async def _action_select(self, args: dict) -> ToolResult:
        page = await self._bm.get_page(self._project_id)
        ref_map = self._bm.get_ref_map(self._project_id, id(page))
        if not ref_map:
            return ToolResult(content="No snapshot taken yet. Run snapshot first to see the page.")
        locator = await resolve_ref(ref_map, args.get("ref", ""), page)
        value = args.get("value", "")
        await locator.select_option(value)
        await self._wait_for_stable(page)
        screenshot_path = await self._bm.capture_screenshot(page, self._workspace, self._session_id, project_dir_name=self._project_dir_name)
        return ToolResult(
            content=f"Selected '{value}' in element ref={args.get('ref', '')}",
            meta={"url": page.url, "title": await page.title(), "screenshot_path": screenshot_path},
        )

    async def _action_scroll(self, args: dict) -> ToolResult:
        page = await self._bm.get_page(self._project_id)
        ref = args.get("ref")
        if ref:
            ref_map = self._bm.get_ref_map(self._project_id, id(page))
            if not ref_map:
                return ToolResult(content="No snapshot taken yet. Run snapshot first to see the page.")
            locator = await resolve_ref(ref_map, ref, page)
            await locator.scroll_into_view_if_needed()
        else:
            direction = args.get("direction", "down")
            amount = args.get("amount", 3)
            pixels = amount * 100
            dx, dy = 0, 0
            if direction == "down":
                dy = pixels
            elif direction == "up":
                dy = -pixels
            elif direction == "right":
                dx = pixels
            elif direction == "left":
                dx = -pixels
            await page.mouse.wheel(dx, dy)
        screenshot_path = await self._bm.capture_screenshot(page, self._workspace, self._session_id, project_dir_name=self._project_dir_name)
        return ToolResult(
            content=f"Scrolled {args.get('direction', 'down')}",
            meta={"url": page.url, "title": await page.title(), "screenshot_path": screenshot_path},
        )

    async def _action_drag(self, args: dict) -> ToolResult:
        page = await self._bm.get_page(self._project_id)
        ref_map = self._bm.get_ref_map(self._project_id, id(page))
        if not ref_map:
            return ToolResult(content="No snapshot taken yet. Run snapshot first to see the page.")
        start_locator = await resolve_ref(ref_map, args.get("start_ref", ""), page)
        end_locator = await resolve_ref(ref_map, args.get("end_ref", ""), page)
        await start_locator.drag_to(end_locator)
        await self._wait_for_stable(page)
        screenshot_path = await self._bm.capture_screenshot(page, self._workspace, self._session_id, project_dir_name=self._project_dir_name)
        return ToolResult(
            content=f"Dragged from ref={args.get('start_ref', '')} to ref={args.get('end_ref', '')}",
            meta={"url": page.url, "title": await page.title(), "screenshot_path": screenshot_path},
        )

    def _resolve_upload_path(self, file_path: str) -> str | None:
        """Resolve file_path to an absolute path within the workspace.

        Accepts both relative paths and absolute paths (if within workspace).
        Returns None if outside workspace.
        """
        if os.path.isabs(file_path):
            resolved = os.path.realpath(file_path)
        else:
            resolved = os.path.realpath(os.path.join(self._workspace, file_path))
        workspace_real = os.path.realpath(self._workspace)
        if not resolved.startswith(workspace_real + os.sep) and resolved != workspace_real:
            return None
        return resolved

    async def _action_upload_file(self, args: dict) -> ToolResult:
        page = await self._bm.get_page(self._project_id)
        file_path = args.get("file_path", "")

        # 1. Path validation (workspace confinement)
        resolved = self._resolve_upload_path(file_path)
        if resolved is None:
            return ToolResult(
                content=f"Error: file path outside workspace: {file_path}. "
                        f"Only files within the project workspace can be uploaded."
            )
        if not os.path.isfile(resolved):
            return ToolResult(content=f"Error: file not found: {file_path}")

        # 2. Check for pending FileChooser (from a prior click on upload button)
        chooser = self._bm.consume_file_chooser(self._project_id)
        if chooser:
            await chooser.set_files(resolved)
            await self._wait_for_stable(page)
            screenshot_path = await self._bm.capture_screenshot(
                page, self._workspace, self._session_id,
                project_dir_name=self._project_dir_name,
            )
            return ToolResult(
                content=f"Uploaded {os.path.basename(resolved)} via file chooser",
                meta={"url": page.url, "title": await page.title(),
                      "screenshot_path": screenshot_path},
            )

        # 3. Fallback: locator-based upload (for hidden <input type="file">)
        ref = args.get("ref", "")
        if not ref:
            return ToolResult(
                content="Error: no pending file chooser and no ref provided. "
                        "Click the upload button first, then call upload_file."
            )
        ref_map = self._bm.get_ref_map(self._project_id, id(page))
        if not ref_map:
            return ToolResult(content="No snapshot taken yet. Run snapshot first.")
        locator = await resolve_ref(ref_map, ref, page)
        await locator.set_input_files(resolved)
        await self._wait_for_stable(page)
        screenshot_path = await self._bm.capture_screenshot(
            page, self._workspace, self._session_id,
            project_dir_name=self._project_dir_name,
        )
        return ToolResult(
            content=f"Uploaded {os.path.basename(resolved)} to input ref={ref}",
            meta={"url": page.url, "title": await page.title(),
                  "screenshot_path": screenshot_path},
        )

    # ------------------------------------------------------------------
    # Observation actions
    # ------------------------------------------------------------------

    async def _action_snapshot(self, args: dict) -> ToolResult:
        page = await self._bm.get_page(self._project_id)
        interactive_only = args.get("interactive_only", False)
        ax_tree = await _get_ax_tree(page)
        text, ref_map, stats = serialize_snapshot(ax_tree, interactive_only=interactive_only)
        self._bm.store_ref_map(self._project_id, id(page), ref_map)
        url = page.url
        wrapped = wrap_untrusted_content(text, url) if text else "(empty page)"
        signals = await self._collect_page_signals(page)
        return ToolResult(
            content=wrapped,
            meta={"url": url, "title": await page.title(), "snapshot_stats": asdict(stats),
                  "page_signals": signals},
        )

    async def _action_screenshot(self, args: dict) -> ToolResult:
        page = await self._bm.get_page(self._project_id)
        annotate = args.get("annotate", False)
        if annotate:
            screenshot_path = await self._annotate_screenshot(page)
            if not screenshot_path:
                screenshot_path = await self._bm.capture_screenshot(
                    page, self._workspace, self._session_id,
                    project_dir_name=self._project_dir_name,
                )
        else:
            screenshot_path = await self._bm.capture_screenshot(
                page, self._workspace, self._session_id,
                project_dir_name=self._project_dir_name,
            )
        signals = await self._collect_page_signals(page)
        title = await page.title()
        url = page.url

        # Return multimodal content with embedded image for vision-capable models
        if self._vision_enabled:
            import base64
            try:
                with open(screenshot_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode("ascii")
                content = [
                    {"type": "text", "text": f"Screenshot of {url}. Title: {title}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}},
                ]
            except OSError:
                content = f"Screenshot saved to {screenshot_path}"
        else:
            content = f"Screenshot saved to {screenshot_path}"

        return ToolResult(
            content=content,
            meta={"url": url, "title": title, "screenshot_path": screenshot_path,
                  "page_signals": signals},
        )

    async def _action_extract(self, args: dict) -> ToolResult:
        page = await self._bm.get_page(self._project_id)
        text = await page.inner_text("body")
        if len(text) > _EXTRACT_MAX_CHARS:
            text = text[:_EXTRACT_MAX_CHARS] + "\n[TRUNCATED at 10,000 characters]"
        url = page.url
        wrapped = wrap_untrusted_content(text, url)
        return ToolResult(
            content=wrapped,
            meta={"url": url, "title": await page.title()},
        )

    async def _action_search_page(self, args: dict) -> ToolResult:
        page = await self._bm.get_page(self._project_id)
        query = args.get("text", "").lower()
        if not query:
            return ToolResult(content="No search query provided.")

        ref_map = self._bm.get_ref_map(self._project_id, id(page))
        matches = []
        if ref_map:
            for ref_id, entry in ref_map.items():
                name = entry.name or ""
                if query in name.lower():
                    matches.append(f"  [ref={ref_id}] {entry.role} \"{name}\"")

        if matches:
            return ToolResult(content=f"Found {len(matches)} matching elements:\n" + "\n".join(matches))

        body_text = await page.inner_text("body")
        idx = body_text.lower().find(query)
        if idx >= 0:
            start = max(0, idx - 50)
            end = min(len(body_text), idx + len(query) + 50)
            context = body_text[start:end]
            return ToolResult(content=f"Text found in page content: ...{context}...")

        return ToolResult(content=f"No matches found for '{args.get('text', '')}'")

    # ------------------------------------------------------------------
    # Web search / fetch (single-call, background tab)
    # ------------------------------------------------------------------

    async def _action_search(self, args: dict) -> ToolResult:
        query = args.get("query", "").strip()
        if not query:
            return ToolResult(content="Error: query is required for search action.")

        original_page = await self._bm.get_page(self._project_id)

        search_page = None
        try:
            search_url = f"https://www.google.com/search?q={_url_encode(query)}"
            search_page = await original_page.context.new_page()

            await search_page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            try:
                await search_page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass  # take what we have

            results = await search_page.evaluate(_SEARCH_EXTRACT_JS)

            if not results:
                # Structured extraction failed (selectors may be stale).
                # Fall back to accessibility tree of the already-loaded page.
                try:
                    ax_tree = await _get_ax_tree(search_page)
                    if ax_tree:
                        text, _, _ = serialize_snapshot(ax_tree)
                        if text and text.strip():
                            await search_page.close()
                            search_page = None
                            await original_page.bring_to_front()
                            return ToolResult(
                                content=(
                                    f"Search extraction returned 0 results (selectors may be stale). "
                                    f"Falling back to page snapshot:\n\n{text}"
                                ),
                                meta={"search_fallback": True, "query": query},
                            )
                except Exception:
                    pass  # If snapshot also fails, fall through to original path

                await search_page.close()
                search_page = None
                await original_page.bring_to_front()
                return ToolResult(
                    content=f"No results found for: {query}",
                    meta={"query": query},
                )

            await search_page.close()
            search_page = None
            await original_page.bring_to_front()

            formatted = []
            for i, r in enumerate(results[:10], 1):
                title = r.get("title", "")
                url = r.get("url", "")
                snippet = r.get("snippet", "")
                formatted.append(f"{i}. {title}\n   {url}\n   {snippet}")

            return ToolResult(
                content=f"Search results for: {query}\n\n" + "\n\n".join(formatted),
                meta={"query": query, "result_count": len(results)},
            )

        except Exception as e:
            try:
                if search_page is not None and not search_page.is_closed():
                    await search_page.close()
            except Exception:
                pass
            return ToolResult(content=f"Error searching: {str(e)}")

    async def _action_fetch(self, args: dict) -> ToolResult:
        url = args.get("url", "").strip()
        if not url:
            return ToolResult(content="Error: url is required for fetch action.")

        error = validate_url_pre_navigation(url)
        if error:
            return ToolResult(content=f"Error: URL blocked: {error}")

        original_page = await self._bm.get_page(self._project_id)

        fetch_page = None
        try:
            fetch_page = await original_page.context.new_page()

            await fetch_page.goto(url, wait_until="domcontentloaded", timeout=30000)
            try:
                await fetch_page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass

            text = await fetch_page.evaluate(_FETCH_EXTRACT_JS)

            await fetch_page.close()
            fetch_page = None
            await original_page.bring_to_front()

            if not text or not text.strip():
                return ToolResult(content=f"No readable content extracted from {url}")

            if len(text) > 50000:
                text = text[:50000] + "\n[TRUNCATED at 50,000 characters]"

            return ToolResult(
                content=text,
                meta={"url": url, "chars": len(text)},
            )

        except Exception as e:
            try:
                if fetch_page is not None and not fetch_page.is_closed():
                    await fetch_page.close()
            except Exception:
                pass
            return ToolResult(content=f"Error fetching {url}: {str(e)}")

    async def _action_evaluate(self, args: dict) -> ToolResult:
        page = await self._bm.get_page(self._project_id)
        javascript = args.get("javascript", "")
        result = await page.evaluate(javascript)
        await self._wait_for_stable(page)
        screenshot_path = await self._bm.capture_screenshot(page, self._workspace, self._session_id, project_dir_name=self._project_dir_name)
        content = str(result) if result is not None else "(no return value)"
        return ToolResult(
            content=f"JavaScript result: {content}",
            meta={"url": page.url, "title": await page.title(), "screenshot_path": screenshot_path},
        )

    # ------------------------------------------------------------------
    # Wait action (6 modes)
    # ------------------------------------------------------------------

    async def _action_wait(self, args: dict) -> ToolResult:
        page = await self._bm.get_page(self._project_id)

        if "seconds" in args:
            await asyncio.sleep(args["seconds"])
            return ToolResult(content=f"Waited {args['seconds']} seconds.")

        if "text" in args:
            text = args["text"]
            await page.wait_for_selector(f"text={text}", timeout=_WAIT_TIMEOUT_MS)
            return ToolResult(content=f"Text '{text}' appeared on the page.")

        if "text_gone" in args:
            text = args["text_gone"]
            await page.wait_for_selector(f"text={text}", state="hidden", timeout=_WAIT_TIMEOUT_MS)
            return ToolResult(content=f"Text '{text}' disappeared from the page.")

        if "selector" in args:
            selector = args["selector"]
            await page.wait_for_selector(selector, timeout=_WAIT_TIMEOUT_MS)
            return ToolResult(content=f"Selector '{selector}' appeared on the page.")

        if "url_pattern" in args:
            pattern = args["url_pattern"]
            await page.wait_for_url(pattern, timeout=_WAIT_TIMEOUT_MS)
            return ToolResult(content=f"URL matched pattern '{pattern}'.")

        if "load_state" in args:
            state = args["load_state"]
            await page.wait_for_load_state(state, timeout=_WAIT_TIMEOUT_MS)
            return ToolResult(content=f"Page reached load state '{state}'.")

        return ToolResult(content="No wait condition specified. Provide seconds, text, text_gone, selector, url_pattern, or load_state.")

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    async def _action_pdf(self, args: dict) -> ToolResult:
        page = await self._bm.get_page(self._project_id)
        file_path = args.get("file_path")
        if not file_path:
            from agent_os.agent.project_paths import ProjectPaths
            pdf_dir = ProjectPaths(self._workspace).pdfs_dir
            os.makedirs(pdf_dir, exist_ok=True)
            title = await page.title()
            safe_title = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in (title or "page"))
            file_path = os.path.join(pdf_dir, f"{safe_title}.pdf")
        await page.pdf(path=file_path)
        return ToolResult(
            content=f"PDF saved to {file_path}",
            meta={"file_path": file_path},
        )

    # ------------------------------------------------------------------
    # Tab management
    # ------------------------------------------------------------------

    async def _action_tab_new(self, args: dict) -> ToolResult:
        url = args.get("url")
        page = await self._bm.new_tab(self._project_id, url)
        tab_id = str(id(page))
        page_url = page.url
        title = await page.title()
        return ToolResult(
            content=f"New tab opened (tab_id={tab_id}): {title} ({page_url})",
            meta={"tab_id": tab_id, "url": page_url, "title": title},
        )

    async def _action_tab_switch(self, args: dict) -> ToolResult:
        tab_id = args.get("tab_id", "")
        pages = await self._bm.get_all_pages(self._project_id)
        for page in pages:
            if str(id(page)) == tab_id:
                await page.bring_to_front()
                title = await page.title()
                return ToolResult(
                    content=f"Switched to tab: {title} ({page.url})",
                    meta={"tab_id": tab_id, "url": page.url, "title": title},
                )
        return ToolResult(content=f"Tab {tab_id} not found. Use tab_new to open a new tab.")

    async def _action_tab_close(self, args: dict) -> ToolResult:
        tab_id = args.get("tab_id")
        pages = await self._bm.get_all_pages(self._project_id)
        if tab_id:
            for page in pages:
                if str(id(page)) == tab_id:
                    await page.close()
                    return ToolResult(content=f"Closed tab {tab_id}.")
            return ToolResult(content=f"Tab {tab_id} not found.")
        elif pages:
            page = pages[-1]
            await page.close()
            return ToolResult(content="Closed current tab.")
        return ToolResult(content="No tabs to close.")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------

    async def _action_done(self, args: dict) -> ToolResult:
        text = args.get("text", "Browser task complete.")
        return ToolResult(content=text)

    # ------------------------------------------------------------------
    # Annotated screenshot
    # ------------------------------------------------------------------

    async def _annotate_screenshot(self, page) -> str | None:
        """Inject element overlays, screenshot, remove overlays."""
        ref_map = self._bm.get_ref_map(self._project_id, id(page))
        if not ref_map:
            return None

        await page.evaluate("""() => {
            const overlay = document.createElement('div');
            overlay.id = '__agent_os_annotation_overlay__';
            overlay.style.position = 'fixed';
            overlay.style.top = '0';
            overlay.style.left = '0';
            overlay.style.width = '100%';
            overlay.style.height = '100%';
            overlay.style.pointerEvents = 'none';
            overlay.style.zIndex = '999999';
            document.body.appendChild(overlay);
        }""")

        for ref_id, entry in ref_map.items():
            try:
                locator = await resolve_ref(ref_map, ref_id, page)
                box = await locator.bounding_box()
                if box:
                    await page.evaluate(
                        """([refId, x, y, w, h]) => {
                            const overlay = document.getElementById('__agent_os_annotation_overlay__');
                            const label = document.createElement('div');
                            label.style.cssText = `position:absolute;border:2px solid red;left:${x}px;top:${y}px;width:${w}px;height:${h}px;`;
                            const tag = document.createElement('span');
                            tag.textContent = refId;
                            tag.style.cssText = 'position:absolute;top:-16px;left:0;background:red;color:white;font-size:11px;padding:1px 3px;';
                            label.appendChild(tag);
                            overlay.appendChild(label);
                        }""",
                        [ref_id, box["x"], box["y"], box["width"], box["height"]],
                    )
            except Exception:
                continue

        screenshot_path = await self._bm.capture_screenshot(page, self._workspace, self._session_id, project_dir_name=self._project_dir_name)

        await page.evaluate("() => document.getElementById('__agent_os_annotation_overlay__')?.remove()")

        return screenshot_path

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------

    async def _execute_batch(self, actions: list[dict]) -> ToolResult:
        results = []
        for i, action_args in enumerate(actions):
            action = action_args.get("action", "")

            if (self._autonomy_preset in ("check_in", "supervised")
                    and action in BROWSER_WRITE_ACTIONS):
                pending = actions[i:]
                results.append(
                    f"\n[PAUSED] Action '{action}' requires approval. "
                    f"{len(pending)} actions pending."
                )
                return ToolResult(
                    content="Batch execution:\n" + "\n".join(results),
                    meta={"pending_actions": pending, "paused_at_action": action},
                )

            result = await self._dispatch(action, action_args)
            results.append(f"  [{action}] {result.content[:100]}")

        return ToolResult(content="Batch execution:\n" + "\n".join(results))

    # ------------------------------------------------------------------
    # Error translation
    # ------------------------------------------------------------------

    ERROR_TRANSLATIONS = {
        "Timeout": "Page or element did not respond within the timeout period. The site may be slow.",
        "Target closed": "The page was closed. Use navigate to open a new page.",
        "frame was detached": "The page frame was removed. The page may have been redirected.",
    }

    def _invalidate_stale_ref(self, ref: str) -> None:
        """Remove a stale ref from the current RefMap.

        Called when an action against ``ref`` hits a "waiting for locator" /
        timeout error. Mutating the ref_map in place forces the next
        interaction to fail fast and snapshot before retrying.

        ``BrowserManager.get_ref_map()`` returns a live dict reference (not a
        copy), so ``del current_map[ref]`` removes the ref for subsequent
        calls. If the mock/stub doesn't expose the same object on repeat
        calls, we also mutate any accessible ``_page_state`` ref_maps as a
        best-effort fallback.
        """
        if not ref:
            return
        try:
            # Primary: fetch the live ref_map via the public getter and
            # mutate in place. We probe any page_state that exists for this
            # project so we don't need to re-await get_page() from a sync
            # context.
            page_state = getattr(self._bm, "_page_state", None)
            if isinstance(page_state, dict):
                for state in page_state.values():
                    rm = getattr(state, "ref_map", None)
                    if isinstance(rm, dict) and ref in rm:
                        del rm[ref]
        except Exception:
            pass

        # Fallback for tests/mocks that return a single dict from get_ref_map
        # without exposing ``_page_state``. We pass sentinel values because
        # the signature requires them, but mocks typically ignore args.
        try:
            getter = getattr(self._bm, "get_ref_map", None)
            if callable(getter):
                rm = getter(self._project_id, 0)
                if isinstance(rm, dict) and ref in rm:
                    del rm[ref]
        except Exception:
            pass

    def _translate_error(self, error: Exception, action: str, args: dict) -> ToolResult:
        msg = str(error)
        if "strict mode violation" in msg.lower():
            return ToolResult(content="Multiple elements matched. Run snapshot to see current refs.")
        if "waiting for locator" in msg.lower() or "timeout" in msg.lower():
            ref = args.get("ref", "")
            # Invalidate the stale ref so the agent is forced to re-snapshot
            # before retrying. Non-timeout errors (strict mode, overlay) do
            # NOT hit this branch, so they correctly leave the map untouched.
            self._invalidate_stale_ref(ref)
            return ToolResult(
                content=(
                    f"Element ref {ref} did not become actionable. "
                    f"This element may be disabled, hidden, or removed by page scripts. "
                    f"Try a different element or approach. Run snapshot to get fresh refs."
                )
            )
        if "intercept" in msg.lower() or "pointer" in msg.lower():
            return ToolResult(
                content="Element blocked by overlay or modal. Try dismissing the overlay first (close button or Escape key)."
            )
        for pattern, translation in self.ERROR_TRANSLATIONS.items():
            if pattern.lower() in msg.lower():
                return ToolResult(content=translation)
        return ToolResult(
            content=f"Browser action '{action}' failed: {msg[:200]}. Try a different approach or run snapshot to see current page state."
        )

    def _translate_error_message(self, error: Exception, args: dict | None = None) -> str:
        """Short error message for inline use (e.g., in fill results).

        Accepts optional ``args`` so callers like ``_action_fill`` can pass
        the per-field ``{"ref": "..."}`` dict through to ``_translate_error``.
        That lets timeout errors on a specific field invalidate just that
        ref in the live ref_map.
        """
        result = self._translate_error(error, "", args or {})
        return result.content[:100]
