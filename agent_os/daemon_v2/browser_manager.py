# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Browser lifecycle management for the daemon.

Owns the Patchright browser instance (drop-in Playwright replacement with
anti-detection patches), tracks per-project pages, stores RefMaps, captures
screenshots, and recovers from crashes.
"""

import asyncio
import locale as sys_locale
import logging
import os
import sys
import time
import warnings
from collections import deque
from pathlib import Path

try:
    from patchright.async_api import async_playwright, Browser, BrowserContext, Page
    HAS_PLAYWRIGHT = True
except ImportError:
    async_playwright = None
    HAS_PLAYWRIGHT = False

logger = logging.getLogger(__name__)

# Stealth init script — injected into every browser context to spoof
# automation-detection signals that Patchright does not cover natively.
_STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
Object.defineProperty(navigator, 'plugins', {
    get: () => [1, 2, 3, 4, 5]
});
window.chrome = window.chrome || {};
window.chrome.runtime = window.chrome.runtime || {};
window.chrome.app = {isInstalled: false, InstallState: {DISABLED: 'disabled', INSTALLED: 'installed', NOT_INSTALLED: 'not_installed'}, RunningState: {CANNOT_RUN: 'cannot_run', READY_TO_RUN: 'ready_to_run', RUNNING: 'running'}};
"""

# Common Windows timezone key → IANA mappings
_WIN_TZ_TO_IANA = {
    "Singapore Standard Time": "Asia/Singapore",
    "Malay Peninsula Standard Time": "Asia/Kuala_Lumpur",
    "China Standard Time": "Asia/Shanghai",
    "Taipei Standard Time": "Asia/Taipei",
    "Tokyo Standard Time": "Asia/Tokyo",
    "Korea Standard Time": "Asia/Seoul",
    "India Standard Time": "Asia/Kolkata",
    "Eastern Standard Time": "America/New_York",
    "Central Standard Time": "America/Chicago",
    "Mountain Standard Time": "America/Denver",
    "Pacific Standard Time": "America/Los_Angeles",
    "Alaskan Standard Time": "America/Anchorage",
    "Hawaiian Standard Time": "Pacific/Honolulu",
    "GMT Standard Time": "Europe/London",
    "W. Europe Standard Time": "Europe/Berlin",
    "Romance Standard Time": "Europe/Paris",
    "Central Europe Standard Time": "Europe/Budapest",
    "E. Europe Standard Time": "Europe/Bucharest",
    "Russian Standard Time": "Europe/Moscow",
    "AUS Eastern Standard Time": "Australia/Sydney",
    "New Zealand Standard Time": "Pacific/Auckland",
    "SE Asia Standard Time": "Asia/Bangkok",
    "West Asia Standard Time": "Asia/Tashkent",
    "Arabian Standard Time": "Asia/Dubai",
}

# UTC offset (hours) → IANA timezone fallback
_UTC_OFFSET_TO_IANA = {
    -12: "Etc/GMT+12", -11: "Pacific/Pago_Pago", -10: "Pacific/Honolulu",
    -9: "America/Anchorage", -8: "America/Los_Angeles", -7: "America/Denver",
    -6: "America/Chicago", -5: "America/New_York", -4: "America/Halifax",
    -3: "America/Sao_Paulo", -2: "Atlantic/South_Georgia", -1: "Atlantic/Azores",
    0: "Europe/London", 1: "Europe/Paris", 2: "Europe/Helsinki",
    3: "Europe/Moscow", 4: "Asia/Dubai", 5: "Asia/Karachi",
    5.5: "Asia/Kolkata", 6: "Asia/Dhaka", 7: "Asia/Bangkok",
    8: "Asia/Shanghai", 9: "Asia/Tokyo", 10: "Australia/Sydney",
    11: "Pacific/Noumea", 12: "Pacific/Auckland", 13: "Pacific/Tongatapu",
}


def _detect_locale() -> str:
    """Detect system locale in BCP-47 format (e.g. 'en-US', 'zh-CN').

    On Windows, ``locale.getlocale()`` returns verbose display names like
    'Chinese (Simplified)_China' instead of IETF codes.  We use the
    (deprecated) ``getdefaultlocale()`` which returns correct codes, with
    a validation fallback for future Python versions.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            raw = sys_locale.getdefaultlocale()[0]
        if raw:
            return raw.replace("_", "-")
    except Exception:
        pass
    # Fallback: try getlocale, but only if it looks like a real code
    try:
        raw = sys_locale.getlocale()[0] or ""
        if raw and len(raw) <= 10 and "_" in raw:
            return raw.replace("_", "-")
    except Exception:
        pass
    return "en-US"


def _detect_timezone() -> str:
    """Detect system timezone as IANA identifier (e.g. 'Asia/Shanghai').

    ``time.tzname`` on Windows returns localized display names (e.g.
    '中国标准时间') which Chromium rejects.  We try:
      1. ``tzlocal`` package (if installed)
      2. Windows registry ``TimeZoneKeyName`` → IANA mapping
      3. UTC offset → approximate IANA timezone
    """
    # 1. tzlocal (best, works cross-platform)
    try:
        import tzlocal
        tz = str(tzlocal.get_localzone())
        if "/" in tz:
            return tz
    except Exception:
        pass

    # 2. Windows registry
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\TimeZoneInformation",
            )
            tz_key = winreg.QueryValueEx(key, "TimeZoneKeyName")[0]
            winreg.CloseKey(key)
            iana = _WIN_TZ_TO_IANA.get(tz_key)
            if iana:
                return iana
            logger.debug("Unmapped Windows timezone key: %s", tz_key)
        except Exception:
            pass

    # 3. time.tzname on non-Windows (usually returns valid abbreviations)
    if sys.platform != "win32":
        try:
            tz = time.tzname[0]
            if tz and "/" in tz:
                return tz
        except Exception:
            pass

    # 4. UTC offset fallback
    try:
        offset_hours = -time.timezone / 3600
        iana = _UTC_OFFSET_TO_IANA.get(offset_hours)
        if iana:
            return iana
    except Exception:
        pass

    return "America/New_York"


class _PageState:
    """Per-page state tracking with ring buffer caps."""

    def __init__(self):
        self.ref_map: dict | None = None
        self.console_log: deque = deque(maxlen=500)
        self.errors: deque = deque(maxlen=200)
        self.last_dialog: str | None = None

    def add_console(self, text: str):
        self.console_log.append(text)

    def add_error(self, text: str):
        self.errors.append(text)


class BrowserManager:
    """Daemon-level singleton managing Playwright browser lifecycle."""

    CHROME_FLAGS = [
        "--disable-blink-features=AutomationControlled",
        "--disable-sync",
        "--disable-background-networking",
        "--disable-features=Translate,MediaRouter",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-popup-blocking",
        "--disable-infobars",
        "--disable-dev-shm-usage",
        "--disable-client-side-phishing-detection",
        "--disable-component-extensions-with-background-pages",
        "--force-color-profile=srgb",
        "--start-minimized",
    ]

    def __init__(self, profile_dir: str = None, headless: bool = True):
        self._profile_dir = Path(profile_dir or Path.home() / "orbital" / "browser-profile")
        self._headless = headless
        self._playwright = None
        self._browser = None
        self._context = None
        self._warmup_active = False

        # Per-project page tracking: project_id -> list[page]
        self._project_pages: dict[str, list] = {}

        # Per-page state: page_id (id(page)) -> _PageState
        self._page_state: dict[int, _PageState] = {}

        # Crash recovery tracking
        self._restart_timestamps: deque = deque(maxlen=3)

        # Screenshot counters per session
        self._screenshot_counters: dict[str, int] = {}

        # Pending file chooser per project (from filechooser event interceptor)
        self._pending_file_choosers: dict[str, object] = {}

        # Max concurrent pages
        self._max_pages = 10

    # ------------------------------------------------------------------
    # Browser launch (lazy + persistent context)
    # ------------------------------------------------------------------

    async def ensure_browser(self):
        """Ensure browser is running. Launch if needed. Recover if crashed."""
        if async_playwright is None and self._playwright is None:
            raise RuntimeError("patchright is not installed — install with: pip install patchright")
        if self._context and self._browser and self._browser.is_connected():
            # Browser process is alive, but verify the CDP connection is still
            # responsive (catches stale connections after sleep/wake).
            await self._ensure_browser_alive()
            return self._context

        # Browser is dead or never started
        if self._browser and not self._browser.is_connected():
            logger.warning("Browser crashed — attempting recovery")
            await self._handle_crash()
            return self._context

        return await self._launch()

    async def _get_clean_user_agent(self) -> str | None:
        """Launch browser briefly to get real UA, strip 'Headless', cache result.

        Tries system browsers (channel="chrome", "msedge") before a bare
        chromium.launch() so this works even when bundled chromium hasn't
        been downloaded yet.  Returns None if no browser is available —
        the caller should skip the UA override in that case.
        """
        if hasattr(self, '_cached_clean_ua'):
            return self._cached_clean_ua

        for channel in ("chrome", "msedge", None):
            try:
                kwargs = {"headless": True}
                if channel:
                    kwargs["channel"] = channel
                browser = await self._playwright.chromium.launch(**kwargs)
                page = await browser.new_page()
                raw_ua = await page.evaluate("navigator.userAgent")
                await browser.close()
                clean_ua = raw_ua.replace("HeadlessChrome", "Chrome")
                self._cached_clean_ua = clean_ua
                return clean_ua
            except Exception:
                continue

        logger.warning("Could not obtain clean UA — no browser available for probe")
        return None

    async def _launch(self):
        """Launch Chromium with persistent context and anti-detection flags."""
        self._profile_dir.mkdir(parents=True, exist_ok=True)
        if not self._playwright:
            try:
                self._playwright = await async_playwright().start()
            except FileNotFoundError:
                raise RuntimeError(
                    "Patchright driver (node.exe) not found. "
                    "The browser tool requires the patchright driver to be "
                    "bundled with the application. Reinstall or run: "
                    "pip install patchright"
                )

        # Determine headless from env var override (for CI/testing)
        headless = self._headless
        if os.environ.get("AGENT_OS_BROWSER_HEADLESS", "").lower() in ("1", "true"):
            headless = True
        # HEADED override takes precedence (for demos/debugging)
        if os.environ.get("AGENT_OS_BROWSER_HEADED", "").lower() in ("1", "true"):
            headless = False

        # Override UA in headless mode to remove "HeadlessChrome" detection signal
        ua_override = await self._get_clean_user_agent() if headless else None

        # Detect system locale (BCP-47 format like "en-US", "zh-CN")
        detected_locale = _detect_locale()

        # Detect system timezone (IANA format like "Asia/Shanghai")
        detected_timezone = _detect_timezone()

        launch_kwargs = dict(
            user_data_dir=str(self._profile_dir),
            headless=headless,
            args=self.CHROME_FLAGS,
            ignore_default_args=["--enable-automation"],
            locale=detected_locale,
            timezone_id=detected_timezone,
            user_agent=ua_override,
        )

        # Try system Chrome → Edge → WebKit (macOS) → bundled Chromium
        self._context = None
        channels = [
            ("chrome", "system Chrome"),
            ("msedge", "system Edge"),
        ]
        for channel, label in channels:
            try:
                self._context = await self._playwright.chromium.launch_persistent_context(
                    channel=channel, **launch_kwargs
                )
                logger.info("Browser launched using %s", label)
                break
            except Exception:
                logger.info("%s not available, trying next fallback", label)
        else:
            # macOS: try WebKit (Safari engine) before bundled Chromium
            if sys.platform == "darwin":
                try:
                    webkit_kwargs = dict(
                        user_data_dir=str(self._profile_dir),
                        headless=headless,
                        locale=detected_locale,
                        timezone_id=detected_timezone,
                    )
                    self._context = await self._playwright.webkit.launch_persistent_context(
                        **webkit_kwargs
                    )
                    logger.info("Browser launched using WebKit (Safari)")
                except Exception:
                    logger.info("WebKit not available, trying bundled Chromium")
                    self._context = None

            if self._context is None:
                try:
                    self._context = await self._playwright.chromium.launch_persistent_context(
                        **launch_kwargs
                    )
                    logger.info("Browser launched using bundled Chromium")
                except Exception as exc:
                    raise RuntimeError(
                        "No browser available. Install Chrome or Edge, or run "
                        "'python -m patchright install chromium' to download a "
                        "bundled browser. On macOS, Safari (WebKit) is also supported."
                    ) from exc

        # NOTE: Do NOT use context.add_init_script() — it breaks DNS
        # resolution on Patchright persistent contexts (Windows).  Stealth
        # JS is injected per-page via _apply_stealth() instead.

        self._browser = self._context.browser

        # Check User-Agent for headless leak
        pages = self._context.pages
        if pages:
            try:
                ua = await pages[0].evaluate("navigator.userAgent")
                if "Headless" in ua:
                    logger.warning(
                        "User-Agent contains 'Headless' — site detection possible: %s", ua
                    )
            except Exception:
                pass

        logger.info("Browser launched with profile: %s (locale=%s, tz=%s)",
                     self._profile_dir, detected_locale, detected_timezone)
        return self._context

    # ------------------------------------------------------------------
    # Crash recovery
    # ------------------------------------------------------------------

    async def _handle_crash(self):
        """Handle browser crash. Re-launch with rate limiting."""
        now = time.monotonic()
        self._restart_timestamps.append(now)

        # Rate limit: max 3 restarts in 5 minutes
        if len(self._restart_timestamps) >= 3:
            oldest = self._restart_timestamps[0]
            if now - oldest < 300:
                raise RuntimeError(
                    "Browser keeps crashing (3 times in 5 minutes). "
                    "Check system resources and restart the daemon."
                )

        # Clean up stale state
        self._project_pages.clear()
        self._page_state.clear()
        self._pending_file_choosers.clear()

        # Re-launch
        self._browser = None
        self._context = None
        await self._launch()
        logger.info("Browser recovered after crash")

    async def _ensure_browser_alive(self):
        """Quick health check — verify the browser context is still responsive.

        After sleep/wake the browser process may still be running but the CDP
        connection can be broken.  Accessing ``self._context.pages`` is a
        lightweight round-trip that will throw if the connection is stale.
        """
        try:
            _ = self._context.pages
        except Exception:
            logger.warning("Browser connection stale, relaunching")
            await self._cleanup_stale()
            await self._launch()

    async def _cleanup_stale(self):
        """Safely tear down stale browser/context handles and reset state."""
        self._project_pages.clear()
        self._page_state.clear()
        self._pending_file_choosers.clear()

        if self._context:
            try:
                await self._context.close()
            except Exception:
                pass
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass

        self._browser = None
        self._context = None
        self._playwright = None

    def get_crash_notification(self) -> str:
        """Returns the system message to inject into affected projects."""
        return "Browser was restarted due to a crash. Run snapshot to see current page state."

    # ------------------------------------------------------------------
    # Per-page stealth injection
    # ------------------------------------------------------------------

    async def _apply_stealth(self, page):
        """Inject stealth JS into a page and re-inject on each navigation.

        We cannot use ``context.add_init_script()`` because it breaks DNS
        resolution on Patchright persistent contexts (Windows).  Instead we
        inject per-page and re-inject on every ``load`` event.
        """
        async def _inject():
            try:
                await page.evaluate(_STEALTH_JS)
            except Exception:
                pass  # page may have closed

        # Inject immediately (page is at about:blank)
        await _inject()

        # Re-inject after every navigation
        page.on("load", lambda _: asyncio.ensure_future(_inject()))

    # ------------------------------------------------------------------
    # Filechooser interceptor
    # ------------------------------------------------------------------

    def _setup_page_handlers(self, page, project_id: str):
        """Register event handlers on a new page (console, errors, filechooser, dialog)."""
        state = self._page_state[id(page)]
        page.on("console", lambda msg: state.add_console(msg.text))
        page.on("pageerror", lambda err: state.add_error(str(err)))

        def on_filechooser(chooser):
            self._pending_file_choosers[project_id] = chooser
            logger.debug("File chooser intercepted for project %s", project_id)

        page.on("filechooser", on_filechooser)

        async def _on_dialog(dialog):
            state.last_dialog = f"{dialog.type}: {dialog.message}"
            logger.info("Auto-dismissing dialog: type=%s message=%s", dialog.type, dialog.message)
            await dialog.dismiss()

        page.on("dialog", lambda d: asyncio.ensure_future(_on_dialog(d)))

    def consume_file_chooser(self, project_id: str):
        """Return and remove pending FileChooser, or None."""
        return self._pending_file_choosers.pop(project_id, None)

    # ------------------------------------------------------------------
    # Per-project page management
    # ------------------------------------------------------------------

    async def get_page(self, project_id: str):
        """Get or create the active page for a project."""
        ctx = await self.ensure_browser()

        # Return existing page if alive
        if project_id in self._project_pages:
            pages = self._project_pages[project_id]
            pages = [p for p in pages if not p.is_closed()]
            self._project_pages[project_id] = pages
            if pages:
                return pages[-1]

        # Enforce page cap with LRU eviction
        total_pages = sum(len(ps) for ps in self._project_pages.values())
        if total_pages >= self._max_pages:
            await self._evict_lru_page()

        # Create new page
        page = await ctx.new_page()
        await self._apply_stealth(page)
        self._project_pages.setdefault(project_id, []).append(page)
        self._page_state[id(page)] = _PageState()
        self._setup_page_handlers(page, project_id)

        return page

    async def get_all_pages(self, project_id: str) -> list:
        """Get all open pages (tabs) for a project."""
        if project_id not in self._project_pages:
            return []
        pages = [p for p in self._project_pages[project_id] if not p.is_closed()]
        self._project_pages[project_id] = pages
        return pages

    async def new_tab(self, project_id: str, url: str = None):
        """Open a new tab for a project."""
        ctx = await self.ensure_browser()
        total = sum(len(ps) for ps in self._project_pages.values())
        if total >= self._max_pages:
            await self._evict_lru_page()
        page = await ctx.new_page()
        await self._apply_stealth(page)
        if url:
            await page.goto(url)
        self._project_pages.setdefault(project_id, []).append(page)
        self._page_state[id(page)] = _PageState()
        self._setup_page_handlers(page, project_id)
        return page

    async def close_project_pages(self, project_id: str):
        """Close all pages for a project. Called on agent stop."""
        pages = self._project_pages.pop(project_id, [])
        for page in pages:
            self._page_state.pop(id(page), None)
            if not page.is_closed():
                await page.close()

    async def _evict_lru_page(self):
        """Close the least recently used page across all projects."""
        if not self._project_pages:
            return
        target_pid = max(self._project_pages, key=lambda k: len(self._project_pages[k]))
        pages = self._project_pages[target_pid]
        if pages:
            evicted = pages.pop(0)
            self._page_state.pop(id(evicted), None)
            if not evicted.is_closed():
                await evicted.close()

    # ------------------------------------------------------------------
    # RefMap storage
    # ------------------------------------------------------------------

    def store_ref_map(self, project_id: str, page_id: int, ref_map: dict):
        """Store a RefMap for a specific project+page."""
        if page_id in self._page_state:
            self._page_state[page_id].ref_map = ref_map

    def get_ref_map(self, project_id: str, page_id: int) -> dict | None:
        """Get the current RefMap for a project+page."""
        if page_id in self._page_state:
            return self._page_state[page_id].ref_map
        return None

    def clear_ref_map(self, project_id: str, page_id: int):
        """Invalidate refs (called on navigation, new snapshot)."""
        if page_id in self._page_state:
            self._page_state[page_id].ref_map = None

    # ------------------------------------------------------------------
    # Screenshot capture
    # ------------------------------------------------------------------

    async def capture_screenshot(
        self, page, workspace: str, session_id: str,
        max_width: int = 1400, max_height: int = 900,
        project_dir_name: str = "",
    ) -> str:
        """Capture screenshot, save to workspace, return file path."""
        counter_key = session_id
        step = self._screenshot_counters.get(counter_key, 0) + 1
        self._screenshot_counters[counter_key] = step

        if project_dir_name:
            screenshot_dir = Path(workspace) / "orbital-output" / project_dir_name / "screenshots" / session_id
        else:
            screenshot_dir = Path(workspace) / "orbital-output" / "screenshots" / session_id
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Enforce retention (max 50)
        existing = sorted(screenshot_dir.glob("step_*.png"))
        while len(existing) >= 50:
            existing[0].unlink()
            existing.pop(0)

        file_path = screenshot_dir / f"step_{step:04d}.png"
        await page.screenshot(path=str(file_path), full_page=False)

        return str(file_path)

    def cleanup_screenshots(self, workspace: str, session_id: str = None):
        """Remove screenshot files. Called on project deletion / daemon shutdown."""
        import shutil

        base = Path(workspace) / "orbital-output" / "screenshots"
        if session_id:
            target = base / session_id
            if target.exists():
                shutil.rmtree(target)
        elif base.exists():
            shutil.rmtree(base)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self):
        """Clean shutdown. Called on daemon stop."""
        logger.info("Browser shutdown starting")
        for project_id in list(self._project_pages.keys()):
            logger.debug("Closing pages for project %s", project_id)
            await self.close_project_pages(project_id)
        if self._context:
            try:
                await self._context.close()
                logger.debug("Browser context closed")
            except Exception as exc:
                logger.warning("Error closing browser context: %s", exc)
        if self._playwright:
            try:
                await self._playwright.stop()
                logger.debug("Playwright stopped")
            except Exception as exc:
                logger.warning("Error stopping playwright: %s", exc)
        self._browser = None
        self._context = None
        self._playwright = None
        logger.info("Browser shutdown complete")

    # ------------------------------------------------------------------
    # Browser warmup (headed session for cookie warmup)
    # ------------------------------------------------------------------

    @property
    def warmup_active(self) -> bool:
        return self._warmup_active

    async def launch_warmup(self, url: str):
        """Launch a headed browser for manual cookie warmup.

        Uses a SEPARATE playwright instance so it doesn't interfere with
        the daemon's headless browser.  The user interacts with the browser,
        then closes it.  Cookies persist in the shared profile directory.
        """
        self._warmup_active = True
        try:
            await self._launch_warmup_impl(url)
        finally:
            self._warmup_active = False

    async def _launch_warmup_impl(self, url: str):
        self._profile_dir.mkdir(parents=True, exist_ok=True)

        if async_playwright is None:
            raise RuntimeError(
                "patchright is not installed — install with: pip install patchright"
            )

        try:
            pw = await async_playwright().start()
        except FileNotFoundError:
            raise RuntimeError(
                "Patchright driver (node.exe) not found. "
                "Reinstall or run: pip install patchright"
            )

        launch_kwargs = dict(
            user_data_dir=str(self._profile_dir),
            headless=False,
            args=self.CHROME_FLAGS + ["--force-color-profile=srgb"],
            ignore_default_args=["--enable-automation"],
            locale=_detect_locale(),
            timezone_id=_detect_timezone(),
        )

        # Same Chrome -> Edge -> WebKit (macOS) -> bundled Chromium fallback
        ctx = None
        channels = [
            ("chrome", "system Chrome"),
            ("msedge", "system Edge"),
        ]
        for channel, label in channels:
            try:
                ctx = await pw.chromium.launch_persistent_context(
                    channel=channel, **launch_kwargs
                )
                logger.info("Warmup browser launched using %s", label)
                break
            except Exception:
                logger.info("Warmup: %s not available, trying next", label)
        else:
            # macOS: try WebKit (Safari engine) before bundled Chromium
            if sys.platform == "darwin":
                try:
                    webkit_kwargs = dict(
                        user_data_dir=str(self._profile_dir),
                        headless=False,
                        locale=_detect_locale(),
                        timezone_id=_detect_timezone(),
                    )
                    ctx = await pw.webkit.launch_persistent_context(**webkit_kwargs)
                    logger.info("Warmup browser launched using WebKit (Safari)")
                except Exception:
                    logger.info("Warmup: WebKit not available, trying bundled Chromium")

            if ctx is None:
                try:
                    ctx = await pw.chromium.launch_persistent_context(**launch_kwargs)
                    logger.info("Warmup browser launched using bundled Chromium")
                except Exception as exc:
                    await pw.stop()
                    raise RuntimeError(
                        "No browser available for warmup. Install Chrome or Edge. "
                        "On macOS, Safari (WebKit) is also supported."
                    ) from exc

        # Navigate to the target URL in a new page
        page = await ctx.new_page()
        await self._apply_stealth(page)
        await page.goto(url)

        # Wait for user to close all browser windows
        while ctx.pages:
            try:
                page = ctx.pages[-1]
                await page.wait_for_event("close", timeout=0)
            except Exception:
                break

        try:
            await ctx.close()
        except Exception:
            pass
        await pw.stop()
        logger.info("Warmup browser closed — cookies saved to %s", self._profile_dir)
