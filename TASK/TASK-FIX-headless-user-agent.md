# TASK-FIX-headless-user-agent: Set Realistic User-Agent for Headless Browser

*Created: 2026-03-11*
*Status: INVESTIGATED*
*Severity: Medium (sites detect headless browser and degrade functionality)*
*Complexity: Low (1-3 lines)*
*Ref: Daemon monitoring session 2026-03-11, Issue 2*
*Depends on: Nothing*

---

## One-Sentence Summary

Set a realistic `user_agent` string in `launch_persistent_context()` to remove "HeadlessChrome" from the User-Agent header, which currently leaks headless mode to every site the browser visits.

## What Happens

```
User-Agent contains 'Headless' — site detection possible: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/145.0.0.0 Safari/537.36
```

The browser is launched in headless mode via Patchright. The stealth script (`_STEALTH_JS` at line 33) spoofs `navigator.webdriver`, `navigator.languages`, `navigator.plugins`, and `window.chrome`, but does NOT modify the User-Agent string. The UA still contains "HeadlessChrome" which sites (especially X/Twitter, LinkedIn, and Cloudflare-protected sites) use to detect and block automation.

## Current Code (browser_manager.py)

Lines 278-285 — `launch_persistent_context()` kwargs:
```python
launch_kwargs = dict(
    user_data_dir=str(self._profile_dir),
    headless=headless,
    args=self.CHROME_FLAGS,
    ignore_default_args=["--enable-automation"],
    locale=detected_locale,
    timezone_id=detected_timezone,
)
```

No `user_agent` parameter is set. Patchright/Playwright uses the browser's default UA, which in headless mode includes "HeadlessChrome".

## Root Cause Analysis

### Root Cause 1: user_agent not set in launch_persistent_context() (CONFIRMED)
Patchright's `launch_persistent_context()` accepts a `user_agent` parameter. Setting it replaces the UA in both HTTP headers AND `navigator.userAgent`. The code doesn't set it.

**Evidence:** Line 278-285 shows all kwargs — no `user_agent` present. The warning at line 326-329 confirms the UA contains "Headless".

### Root Cause 2: Patchright stealth doesn't cover UA (CONFIRMED)
Patchright applies some anti-detection patches at the CDP level, but User-Agent modification requires explicit configuration. It's not part of the stealth layer. The `_STEALTH_JS` only patches JavaScript APIs, not HTTP request headers.

**Evidence:** `_STEALTH_JS` (lines 33-41) patches `navigator.webdriver`, `navigator.languages`, `navigator.plugins`, `window.chrome` — but NOT `navigator.userAgent`.

### Root Cause 3: Contributes to X/Twitter browsing failures (PROBABLE)
The x-manager project experienced "element not actionable" failures on X/Twitter. Twitter likely detected the headless browser via the UA string and served a degraded/different DOM structure (e.g., a "please enable JavaScript" wall, or a simplified mobile layout where elements have different selectors or are non-interactive).

**Evidence:** The stuck browser loop on X/Twitter (36 click retries) may have been partly caused by Twitter serving a different page to headless browsers.

## Proposed Fix

**File:** `agent_os/daemon_v2/browser_manager.py`, in `_launch()`

Add `user_agent` to `launch_kwargs`:

```python
# Realistic User-Agent — matches the headless Chrome version but without "Headless"
# Dynamically derive from the actual browser version to stay current
_HEADLESS_UA_OVERRIDE = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/145.0.0.0 Safari/537.36"
)

launch_kwargs = dict(
    user_data_dir=str(self._profile_dir),
    headless=headless,
    args=self.CHROME_FLAGS,
    ignore_default_args=["--enable-automation"],
    locale=detected_locale,
    timezone_id=detected_timezone,
    user_agent=_HEADLESS_UA_OVERRIDE if headless else None,
)
```

Key decisions:
- **Only override in headless mode.** In headed mode, the browser's real UA is fine.
- **Match the Chrome version.** The UA should match the actual Chromium version to avoid version mismatch detection. Ideally, derive dynamically from the browser binary, but a hardcoded version that's updated with Patchright upgrades is acceptable.
- **Don't override in warmup mode.** The `launch_warmup()` method (line 622) already uses `headless=False`, so it gets a real UA.

### Alternative: Dynamic UA detection

Launch the browser, read the UA, strip "Headless", and set the cleaned UA on the context:

```python
# After launch, before returning context:
pages = self._context.pages
if pages and headless:
    ua = await pages[0].evaluate("navigator.userAgent")
    if "Headless" in ua:
        clean_ua = ua.replace("HeadlessChrome", "Chrome")
        # Can't change UA on persistent context after launch — must relaunch
        # This approach doesn't work for persistent contexts
```

This doesn't work for persistent contexts (UA is set at launch time). Use the static approach.

### Also update _STEALTH_JS to override navigator.userAgent

As a belt-and-suspenders measure, add UA override to the stealth script for pages that read `navigator.userAgent` via JavaScript:

```javascript
// Add to _STEALTH_JS:
Object.defineProperty(navigator, 'userAgent', {
    get: () => navigator.userAgent.replace('HeadlessChrome', 'Chrome')
});
```

However, this is redundant if `user_agent` is set at launch — Playwright overrides both HTTP and JS-level UA. Include only if the `user_agent` launch parameter doesn't cover all cases.

## Impact Assessment

- **Current impact:** Sites with headless detection (X/Twitter, LinkedIn, Cloudflare, Google) may block or degrade browsing. This directly affects the browser tool's usefulness for social media monitoring tasks.
- **Fix impact:** Removes the most obvious headless detection signal. Sites may still detect automation via other means (WebGL fingerprinting, canvas fingerprinting, behavioral analysis), but UA is the lowest-hanging fruit.
- **Risk:** Low. Setting a realistic UA is standard practice for browser automation. No behavioral change for non-headless mode.

## Tests

```
Test: test_headless_launch_sets_realistic_ua
Setup: Launch BrowserManager in headless mode
Action: Read navigator.userAgent from a page
Assert: UA does NOT contain "Headless"
Assert: UA contains "Chrome/"

Test: test_headed_launch_uses_default_ua
Setup: Launch BrowserManager in headed mode (or mock)
Assert: user_agent kwarg is None (browser uses default)

Test: test_ua_warning_not_emitted_after_fix
Setup: Launch BrowserManager in headless mode with UA override
Action: Check log output
Assert: "User-Agent contains 'Headless'" warning NOT emitted

Test: test_stealth_js_still_applied_with_ua_override
Setup: Launch with UA override
Action: Check navigator.webdriver, navigator.plugins
Assert: Stealth patches still active (not broken by UA override)
```

## Cleanup

- Remove ALL debug prints, console.logs, and temporary code
- Search for: `print(f"[DEBUG`, `console.log("[DEBUG`, `TODO`, `HACK`, `TEMP`, `FIXME`
- Remove any you find
