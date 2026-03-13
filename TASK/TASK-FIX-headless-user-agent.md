# TASK-FIX-headless-user-agent: Set Realistic User-Agent for Headless Browser

*Created: 2026-03-11*
*Status: READY*
*Severity: Medium (sites detect headless browser and degrade functionality)*
*Complexity: Low*
*Ref: Daemon monitoring session 2026-03-11, Issue 2*
*Depends on: Nothing*

---

## One-Sentence Summary

Set a realistic `user_agent` string in `launch_persistent_context()` to remove "HeadlessChrome" from the User-Agent header, which currently leaks headless mode to every site the browser visits.

## Problem

The browser launches in headless mode via Patchright. The stealth script (`_STEALTH_JS` at line 33) spoofs `navigator.webdriver`, `navigator.languages`, `navigator.plugins`, and `window.chrome`, but does NOT modify the User-Agent string. The UA contains "HeadlessChrome" which sites (X/Twitter, LinkedIn, Cloudflare-protected sites) use to detect and block automation.

```
User-Agent contains 'Headless': Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/145.0.0.0 Safari/537.36
```

## Root Cause

**`user_agent` not set in `launch_persistent_context()`.** Patchright accepts a `user_agent` parameter that overrides both HTTP headers AND `navigator.userAgent`. The code doesn't set it.

**Evidence:** `browser_manager.py` lines 278-285 — all launch kwargs listed, no `user_agent`. The warning at lines 326-329 confirms the UA contains "Headless". The `_STEALTH_JS` (lines 33-41) patches JS APIs but NOT the UA string or HTTP headers.

**Note:** This may contribute to X/Twitter browsing failures (element not actionable, 36 click retries) if Twitter served degraded DOM to detected headless browsers. However, Twitter's anti-automation is multi-layered — treat this as hygiene, not a guaranteed fix for Twitter issues.

## Fix

**File:** `agent_os/daemon_v2/browser_manager.py`

### Step 1: Derive UA dynamically from browser binary

Add a helper method to `BrowserManager` that launches the browser briefly in headed mode, reads the real UA, strips "Headless", and caches the result. This prevents version mismatch between the actual Chromium binary and the spoofed UA (which is itself a detection signal).

```python
async def _get_clean_user_agent(self) -> str:
    """Launch browser briefly to get real UA, strip 'Headless', cache result."""
    # Cache on the instance to avoid repeated launches
    if hasattr(self, '_cached_clean_ua'):
        return self._cached_clean_ua
    
    browser = await self._playwright.chromium.launch(headless=True)
    page = await browser.new_page()
    raw_ua = await page.evaluate("navigator.userAgent")
    await browser.close()
    
    clean_ua = raw_ua.replace("HeadlessChrome", "Chrome")
    self._cached_clean_ua = clean_ua
    return clean_ua
```

### Step 2: Set user_agent in launch_kwargs (headless only)

In `_launch()`, add `user_agent` to the kwargs:

```python
# Only override in headless mode — headed mode uses the browser's real UA
ua_override = await self._get_clean_user_agent() if headless else None

launch_kwargs = dict(
    user_data_dir=str(self._profile_dir),
    headless=headless,
    args=self.CHROME_FLAGS,
    ignore_default_args=["--enable-automation"],
    locale=detected_locale,
    timezone_id=detected_timezone,
    user_agent=ua_override,
)
```

### Key decisions

- **Only override in headless mode.** Headed mode gets a real UA naturally.
- **Derive UA dynamically.** Do NOT hardcode a Chrome version string — it rots on Patchright upgrades and version mismatch is a detection signal.
- **Do NOT add UA override to `_STEALTH_JS`.** The `user_agent` launch parameter already overrides both HTTP headers and `navigator.userAgent`. A JS-level override would be redundant dead code.
- **`launch_warmup()` (line 622) is unaffected.** It uses `headless=False`, so it gets a real UA.

## DO NOT

- Do NOT hardcode a Chrome version string in the UA
- Do NOT modify `_STEALTH_JS` for UA — the launch parameter handles it
- Do NOT change any behavior in headed mode
- Do NOT change `launch_warmup()`

## Tests

Place in `tests/regression/`

```
test_headless_launch_sets_realistic_ua:
  Setup: Launch BrowserManager in headless mode
  Action: Read navigator.userAgent from a page
  Assert: UA does NOT contain "Headless"
  Assert: UA contains "Chrome/"
  Assert: UA version matches the actual browser binary version

test_headed_launch_uses_default_ua:
  Setup: Launch BrowserManager in headed mode (or mock)
  Assert: user_agent kwarg is None (browser uses default)

test_ua_warning_not_emitted_after_fix:
  Setup: Launch BrowserManager in headless mode with UA override
  Action: Check log output
  Assert: "User-Agent contains 'Headless'" warning NOT emitted

test_stealth_js_still_applied_with_ua_override:
  Setup: Launch with UA override
  Action: Check navigator.webdriver, navigator.plugins
  Assert: Stealth patches still active (not broken by UA override)
```

## Manual Smoke Test

1. Launch daemon with headless browser
2. Navigate to `https://httpbin.org/user-agent`
3. Confirm returned UA does not contain "Headless"
4. Navigate to a Cloudflare-protected site — confirm no immediate block

## Cleanup

- Remove ALL debug prints, console.logs, and temporary code
- Search for: `print(f"[DEBUG`, `console.log("[DEBUG`, `TODO`, `HACK`, `TEMP`, `FIXME`
- Remove any you find

## DONE WHEN

- [ ] Headless browser UA does not contain "HeadlessChrome"
- [ ] UA version matches actual browser binary (not hardcoded)
- [ ] Headed mode unchanged
- [ ] All 4 regression tests pass
- [ ] Manual smoke test passes
- [ ] No debug artifacts left in code
