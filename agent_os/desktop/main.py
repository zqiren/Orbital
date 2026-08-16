# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import os
import time
import threading
import socket

# PyInstaller windowed mode (console=False) sets sys.stdout/stderr to None.
# Uvicorn's log formatter calls sys.stderr.isatty() on init and crashes.
# Redirect to a log file so errors are visible for debugging.
if getattr(sys, "frozen", False) and sys.stderr is None:
    _log_dir = os.path.join(os.path.expanduser("~"), "Library", "Logs", "Orbital")
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = open(os.path.join(_log_dir, "orbital-stderr.log"), "w")
    sys.stdout = _log_file
    sys.stderr = _log_file

# Set AppUserModelID so Windows taskbar shows the Orbital icon, not Python's
if sys.platform == "win32":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("orbital.desktop.app")


def find_free_port(preferred: int = 8000) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", preferred))
        sock.close()
        return preferred
    except OSError:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port


def is_already_running(port: int = 8000) -> bool:
    import urllib.request
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v2/settings", timeout=2)
        return resp.status == 200
    except Exception:
        return False


def _frozen_base_dir() -> str:
    """Return the base directory for bundled resources.

    On macOS .app bundles the executable lives in Contents/MacOS/ but
    manually-copied resources (web/, assets/) live in Contents/Resources/.
    On Windows the resources sit alongside the executable.
    """
    exe_dir = os.path.dirname(sys.executable)
    resources_dir = os.path.join(os.path.dirname(exe_dir), "Resources")
    if os.path.isdir(resources_dir):
        return resources_dir
    return exe_dir


def resolve_spa_dir() -> str:
    if getattr(sys, "frozen", False):
        spa = os.path.join(_frozen_base_dir(), "web")
    else:
        spa = os.path.join(os.path.dirname(__file__), "..", "..", "web", "dist")
    return os.path.abspath(spa)


def _inherit_shell_path():
    """Inherit the user's full shell PATH into the daemon process.

    PyInstaller bundles run with a minimal PATH that doesn't include
    user-installed tool directories (homebrew, npm global, ~/.local/bin,
    etc.).  This prevents agent binary discovery via shutil.which().

    We invoke the user's login shell once at startup to capture the
    real PATH, then merge it into os.environ so all downstream code
    (including SetupEngine.resolve_binary()) benefits.
    """
    import subprocess as _sp

    shell = os.environ.get("SHELL", "/bin/bash")
    try:
        result = _sp.run(
            [shell, "-lc", "echo $PATH"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            shell_path = result.stdout.strip()
            current = os.environ.get("PATH", "")
            # Merge: shell paths first, then existing (deduped)
            seen = set()
            merged = []
            for p in shell_path.split(os.pathsep) + current.split(os.pathsep):
                if p and p not in seen:
                    seen.add(p)
                    merged.append(p)
            os.environ["PATH"] = os.pathsep.join(merged)
    except Exception:
        pass  # Fall back to existing PATH silently


def _prepare_bundled_ripgrep():
    """Ensure the bundled macOS ripgrep binary is executable and dequarantined.

    PyInstaller's datas= copies files verbatim but may not preserve the execute
    bit on Unix, and macOS may attach a com.apple.quarantine xattr to files
    that arrive via .dmg. Both would cause the grep tool to fail.

    Idempotent — safe to run even if the bit is already set or the xattr is
    not present. Swallows all errors: grep will fall back to system PATH.
    """
    if sys.platform != "darwin":
        return
    try:
        from agent_os.agent.tools.grep_tool import _find_ripgrep
    except Exception:
        return
    try:
        rg_path = _find_ripgrep()
    except Exception:
        rg_path = None
    if not rg_path or not os.path.isfile(rg_path):
        return
    try:
        mode = os.stat(rg_path).st_mode
        if not (mode & 0o111):
            os.chmod(rg_path, mode | 0o755)
    except Exception:
        pass
    try:
        import subprocess as _sp
        _sp.run(
            ["xattr", "-d", "com.apple.quarantine", rg_path],
            check=False, capture_output=True,
        )
    except Exception:
        pass


def _disable_app_nap():
    """Disable App Nap at daemon startup on macOS (defense-in-depth).

    The primary assertion lives in MacOSPlatformProvider.setup(), but
    the desktop app may start serving before the provider is initialized.
    Multiple assertions are safe — macOS tracks them independently.
    """
    if sys.platform != "darwin":
        return
    try:
        from AppKit import NSProcessInfo

        activity = NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
            0x00FFFFFF,  # NSActivityUserInitiatedAllowingIdleSystemSleep
            "Orbital daemon: maintaining agent connections and background tasks",
        )
        # Store globally to prevent garbage collection
        _disable_app_nap._activity = activity
    except Exception:
        pass


def start_daemon(port: int):
    os.environ["AGENT_OS_SPA_DIR"] = resolve_spa_dir()
    os.environ["AGENT_OS_PORT"] = str(port)
    from agent_os.desktop.migration import DATA_DIR
    os.environ["AGENT_OS_DATA_DIR"] = DATA_DIR
    import uvicorn
    from agent_os.api.app import create_app
    app = create_app(data_dir=DATA_DIR)
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


def wait_for_daemon(port: int, timeout: int = 15) -> bool:
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v2/settings", timeout=2)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def run_smoke_test() -> int:
    """Headless self-check used by CI: boot the daemon, confirm its HTTP server
    answers, then return 0 (pass) or 1 (fail). Starts no GUI / tray / webview.

    This exists to catch a packaged bundle that builds successfully but cannot
    actually start — e.g. a PyInstaller gap where uvicorn's httptools parser is
    missing (``module 'httptools' has no attribute 'HttpRequestParser'``). Such
    a bundle otherwise ships green because CI only checks that the installer
    file was produced, never that it runs. ``wait_for_daemon`` bounds the run to
    its timeout, so this can never hang CI.
    """
    from agent_os.desktop.migration import run_migrations

    run_migrations()
    _inherit_shell_path()
    _prepare_bundled_ripgrep()
    port = find_free_port(8000)
    start_daemon(port)
    ok = wait_for_daemon(port)
    # In a windowed (console=False) bundle sys.stdout may be redirected/None —
    # guard print so the check's exit code remains the source of truth.
    try:
        print(f"SMOKE_TEST {'PASS' if ok else 'FAIL'} port={port}", flush=True)
    except Exception:
        pass
    return 0 if ok else 1


def resolve_icon_path() -> str:
    if getattr(sys, "frozen", False):
        base = _frozen_base_dir()
    else:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # Always use .png — higher quality, works cross-platform.
    # NOTE: this is the *source* PNG for _png_to_hicon() (the Windows HICON
    # fallback decodes it via PIL). Do not make this return .ico — that would
    # reroute the working HICON path through PIL's ICO decoder. The webview
    # window icon is resolved separately by resolve_window_icon_path().
    icon = os.path.join(base, "assets", "icon.png")
    return os.path.abspath(icon)


def resolve_window_icon_path() -> str:
    """Icon path to hand to webview.start(icon=...).

    On Windows, pywebview forwards this to .NET System.Drawing.Icon, which only
    accepts .ico — a .png raises ArgumentException and crashes the main thread
    before any window is created, which in turn kills the daemon thread (GH
    #37/#38). Other backends (Cocoa on macOS, GTK on Linux) accept the .png.
    Falls back to the .png if icon.ico is missing from the bundle.
    """
    png = resolve_icon_path()
    if sys.platform == "win32":
        ico = os.path.join(os.path.dirname(png), "icon.ico")
        if os.path.exists(ico):
            return ico
    return png


def _png_to_hicon(png_path: str, size: int):
    """Convert a PNG file to a Windows HICON handle at the given size."""
    import ctypes
    from PIL import Image as PILImage

    img = PILImage.open(png_path).convert("RGBA").resize((size, size), PILImage.LANCZOS)
    # BGRA byte order for Windows HBITMAP
    pixels = img.tobytes("raw", "BGRA")

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", ctypes.c_uint32),
            ("biWidth", ctypes.c_int32),
            ("biHeight", ctypes.c_int32),
            ("biPlanes", ctypes.c_uint16),
            ("biBitCount", ctypes.c_uint16),
            ("biCompression", ctypes.c_uint32),
            ("biSizeImage", ctypes.c_uint32),
            ("biXPelsPerMeter", ctypes.c_int32),
            ("biYPelsPerMeter", ctypes.c_int32),
            ("biClrUsed", ctypes.c_uint32),
            ("biClrImportant", ctypes.c_uint32),
        ]

    class ICONINFO(ctypes.Structure):
        _fields_ = [
            ("fIcon", ctypes.c_bool),
            ("xHotspot", ctypes.c_uint32),
            ("yHotspot", ctypes.c_uint32),
            ("hbmMask", ctypes.c_void_p),
            ("hbmColor", ctypes.c_void_p),
        ]

    gdi32 = ctypes.windll.gdi32
    user32 = ctypes.windll.user32

    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth = size
    bmi.biHeight = -size  # top-down
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0  # BI_RGB

    # Create color bitmap
    dc = user32.GetDC(0)
    data_ptr = ctypes.c_void_p()
    hbm_color = gdi32.CreateDIBSection(
        dc, ctypes.byref(bmi), 0, ctypes.byref(data_ptr), None, 0)
    ctypes.memmove(data_ptr, pixels, len(pixels))
    user32.ReleaseDC(0, dc)

    # Create mask bitmap (all zeros = fully opaque, alpha in color bitmap)
    hbm_mask = gdi32.CreateBitmap(size, size, 1, 1, None)

    ii = ICONINFO()
    ii.fIcon = True
    ii.hbmMask = hbm_mask
    ii.hbmColor = hbm_color
    hicon = user32.CreateIconIndirect(ctypes.byref(ii))

    gdi32.DeleteObject(hbm_color)
    gdi32.DeleteObject(hbm_mask)
    return hicon


def _set_window_icon():
    """Set the Orbital icon on the pywebview window (Windows only)."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        icon_path = resolve_icon_path()
        if not os.path.exists(icon_path):
            return
        WM_SETICON = 0x0080
        ICON_SMALL = 0
        ICON_BIG = 1
        sm_cxsmicon = user32.GetSystemMetrics(49)  # SM_CXSMICON
        sm_cxicon = user32.GetSystemMetrics(11)     # SM_CXICON
        hicon_sm = _png_to_hicon(icon_path, sm_cxsmicon)
        hicon_lg = _png_to_hicon(icon_path, sm_cxicon)
        if not hicon_sm and not hicon_lg:
            return
        for _ in range(50):
            hwnd = user32.FindWindowW(None, "Orbital")
            if hwnd:
                if hicon_sm:
                    user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon_sm)
                if hicon_lg:
                    user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon_lg)
                break
            time.sleep(0.1)
    except Exception:
        pass


def _is_foreign_url(url: str | None, port: int) -> bool:
    """Return True iff `url` is an http(s) URL whose origin is not this app's.

    Used by open_window()'s `loaded` guard to detect a same-frame navigation
    that escaped the SPA (pywebview only routes NEW-WINDOW navigations to the
    system browser; same-frame navigations sail through unchecked). None,
    empty, `about:blank`, `data:`, `file:`, and the app's own origin
    (127.0.0.1 or localhost, defensively, at `port`) are all "not foreign" —
    only a navigation that would actually leave the SPA counts.

    Origin comparison is a real URL parse (scheme/host/port), not a
    startswith — a lookalike host like `127.0.0.1:{port}.evil.com` must count
    as foreign rather than matching a naive prefix check.
    """
    import urllib.parse

    if not url:
        return False
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        return True
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.hostname not in ("127.0.0.1", "localhost"):
        return True
    try:
        parsed_port = parsed.port
    except ValueError:
        # e.g. "127.0.0.1:8000.evil.com" — host matches but the port segment
        # isn't a real port, so this is not actually our app origin.
        return True
    app_port = parsed_port if parsed_port is not None else (443 if parsed.scheme == "https" else 80)
    return app_port != port


# ------------------------------------------------------------------ drag strip
# macOS-only. See install_titlebar_drag_strip() for the full story; the short
# version is that the frameless window has NO drag region at all without this.

# PyObjC raises if the same ObjC class name is registered twice in one process,
# so the subclass is built once and cached here.
_DRAG_STRIP_CLASS = None

# The band height, once it has been read off a live NSTitlebarContainerView.
# Cached because that view is ABSENT in fullscreen — the measurement is only
# available while the window is in its normal state, so we keep the first
# successful reading rather than re-measuring later and getting the fallback.
_MEASURED_BAND_HEIGHT = None

# Matches --titlebar-h in web/src/index.css. Only used if the live measurement
# is unavailable (fullscreen at install time, or an AppKit shape we don't know).
_TITLEBAR_BAND_FALLBACK = 28.0


def _run_on_main_thread(func):
    """Run `func` on the AppKit main thread, hopping only if we aren't on it.

    Every entry point that touches the drag strip — pywebview's `loaded`,
    `maximized`, and `restored` events — is dispatched on a throw-away thread
    (webview/event.py spawns one per non-locking event), and AppKit geometry
    calls off the main thread raise `NSInternalInconsistencyException: NSWindow
    geometry should only be modified on the main thread!`.

    The hop is fire-and-forget: callers must not depend on `func` having run by
    the time this returns. Blocking on the main queue instead would risk a
    deadlock against pywebview's own main-thread work.
    """
    queue = None
    try:
        import AppKit

        if not AppKit.NSThread.isMainThread():
            queue = AppKit.NSOperationQueue.mainQueue()
    except Exception:
        # No AppKit (non-macOS, or a stripped build) — run inline rather than
        # dropping the work. Off macOS there is no main-thread rule anyway.
        queue = None
    # Deciding where to run and running are kept apart on purpose: folding them
    # into one try would re-invoke `func` on the fallback path if it raised.
    try:
        if queue is None:
            func()
        else:
            queue.addOperationWithBlock_(func)
    except Exception:
        pass


def _double_click_titlebar_action() -> str:
    """What a double-click on the titlebar should do, per system preference.

    macOS stores this in NSGlobalDomain as `AppleActionOnDoubleClick`
    ("Maximize" / "Minimize" / "None"). The key is UNSET on a default install,
    and unset means the system default, which is zoom — so that is also what
    every unrecognized value maps to.
    """
    try:
        import AppKit

        raw = AppKit.NSUserDefaults.standardUserDefaults().stringForKey_(
            "AppleActionOnDoubleClick"
        )
    except Exception:
        return "zoom"
    if raw is None:
        return "zoom"
    normalized = str(raw).strip().lower()
    if normalized == "minimize":
        return "minimize"
    if normalized == "none":
        return "none"
    return "zoom"


def _drag_strip_mouse_down(view, event) -> None:
    """Route a click on the drag strip: double-click gestures, else drag.

    Split out of the ObjC method below so it can be exercised with plain
    doubles — an NSEvent cannot be synthesized meaningfully in a unit test.

    `performWindowDragWithEvent_` is the supported AppKit API for custom drag
    regions: it hands off to the real window-drag loop, so window snapping,
    Spaces, and multi-display all behave natively rather than being
    re-implemented on top of mouse deltas.
    """
    window = view.window()
    if window is None:
        return
    if event.clickCount() == 2:
        action = _double_click_titlebar_action()
        if action == "zoom":
            window.zoom_(view)
        elif action == "minimize":
            window.miniaturize_(view)
        # "none": the user asked for no double-click gesture at all.
        return
    window.performWindowDragWithEvent_(event)


def _drag_strip_class():
    """The transparent NSView that turns the titlebar band into a drag region."""
    global _DRAG_STRIP_CLASS
    if _DRAG_STRIP_CLASS is not None:
        return _DRAG_STRIP_CLASS

    import AppKit

    class _OrbitalTitlebarDragStrip(AppKit.NSView):
        def acceptsFirstMouse_(self, event):
            # Drag an inactive window without a separate click to focus it
            # first, which is how a real titlebar behaves.
            return True

        def mouseDownCanMoveWindow(self):
            # NO, so that mouseDown_ below actually runs. YES would route the
            # click into the window-background drag path instead and skip the
            # double-click handling entirely.
            return False

        def mouseDown_(self, event):
            _drag_strip_mouse_down(self, event)

    _DRAG_STRIP_CLASS = _OrbitalTitlebarDragStrip
    return _DRAG_STRIP_CLASS


def titlebar_band_height(native_window) -> float:
    """Height of the native titlebar band, measured off the live window.

    Read from NSTitlebarContainerView rather than hardcoded so it cannot drift
    from the system metric (28.0 today). The container is a sibling of the
    content view under NSThemeFrame, and is absent in fullscreen — hence the
    module-level cache and the fallback.
    """
    global _MEASURED_BAND_HEIGHT
    if _MEASURED_BAND_HEIGHT is not None:
        return _MEASURED_BAND_HEIGHT
    try:
        theme_frame = native_window.contentView().superview()
        for view in theme_frame.subviews():
            if view.className() != "NSTitlebarContainerView":
                continue
            height = float(view.frame().size.height)
            if height > 0:
                _MEASURED_BAND_HEIGHT = height
                return height
    except Exception:
        pass
    return _TITLEBAR_BAND_FALLBACK


def install_titlebar_drag_strip(native_window):
    """Give the frameless macOS window a drag region. Returns the strip or None.

    Why this is needed at all: pywebview's `frameless` on macOS sets
    `titlebarAppearsTransparent` + NSFullSizeContentView on a still-titled
    window. A *transparent* NSTitlebarContainerView is hit-transparent
    everywhere except its own traffic-light widgets, so every mouse-down in the
    band falls through to the WKWebView and AppKit's titlebar-drag machinery
    never arms. With `easy_drag` correctly off, that leaves the window with no
    drag region whatsoever (bug #60). `-webkit-app-region: drag` is not an
    option — that is a Chromium feature and this shell is WKWebView.

    Three of the four obvious ways to write this silently produce a strip that
    exists and does nothing. All four were measured on a live NSWindow:

    1. The strip must be a subview of the **contentView** (the WKWebView), not
       of NSThemeFrame. AppKit rejects the latter — "adding an unknown
       subview" — and forces it to index [0], behind the webview.
    2. Install on `loaded`, not `before_show`: pywebview only swaps the
       WKWebView in as the contentView inside `webView_didFinishNavigation_`,
       so anything earlier attaches to the throw-away default content view.
    3. `loaded` runs off the main thread (see _run_on_main_thread), and it can
       re-fire — the origin guard reloads the SPA — so this must be idempotent.
    4. **Auto Layout, not autoresizing masks.** WebKitHost reports
       `isFlipped=True` but AppKit lays its subviews out with unflipped math
       (pywebview compensates for exactly this in its own `addSubview_`
       override). With NSViewWidthSizable|NSViewMinYMargin the width tracked a
       resize but the vertical pin did not, leaving a dead band floating in the
       middle of the content. Constraints pin correctly through resize and both
       fullscreen transitions.

    Must be called on the main thread. Failure is always non-fatal: window
    chrome cosmetics never block the app from opening, and a missing strip
    leaves the window dragging no worse than it does today.
    """
    if sys.platform != "darwin":
        return None
    try:
        import AppKit

        content = native_window.contentView()
        if content is None:
            return None

        strip_class = _drag_strip_class()
        for existing in content.subviews():
            if isinstance(existing, strip_class):
                return existing  # trap 3: `loaded` re-fired, already installed

        band = titlebar_band_height(native_window)
        strip = strip_class.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, content.frame().size.width, band)
        )
        strip.setTranslatesAutoresizingMaskIntoConstraints_(False)
        # Positioned above every existing subview so it wins hit-testing inside
        # the webview. The traffic lights still win over it: they live in the
        # titlebar container, which is a sibling of the content view and in
        # front of it, so a full-width strip needs no left inset.
        content.addSubview_positioned_relativeTo_(strip, AppKit.NSWindowAbove, None)
        AppKit.NSLayoutConstraint.activateConstraints_([
            strip.leadingAnchor().constraintEqualToAnchor_(content.leadingAnchor()),
            strip.trailingAnchor().constraintEqualToAnchor_(content.trailingAnchor()),
            strip.topAnchor().constraintEqualToAnchor_(content.topAnchor()),
            strip.heightAnchor().constraintEqualToConstant_(band),
        ])
        return strip
    except Exception:
        return None


def set_drag_strip_hidden(strip, hidden: bool) -> None:
    """Show/hide the drag strip. No-op when there is no strip.

    Hiding is mandatory in fullscreen: the SPA collapses --titlebar-h to 0
    there, so real UI moves into the top band and the strip would swallow its
    clicks. Hidden views drop out of hit-testing entirely, restoring
    click-through to the webview.
    """
    if strip is None:
        return

    def _apply():
        try:
            strip.setHidden_(bool(hidden))
        except Exception:
            pass

    _run_on_main_thread(_apply)


_window = None

# pywebview chrome strings (macOS menu bar, quit dialog, file pickers).
# Keyed off the OS language — the menu bar should match the system, which
# is also what the SPA's own zh auto-detect assumes.
_ZH_LOCALIZATION = {
    'global.quitConfirmation': '确定要退出吗？',
    'global.ok': '好',
    'global.quit': '退出',
    'global.cancel': '取消',
    'global.saveFile': '存储文件',
    'cocoa.menu.about': '关于',
    'cocoa.menu.services': '服务',
    'cocoa.menu.view': '显示',
    'cocoa.menu.edit': '编辑',
    'cocoa.menu.hide': '隐藏',
    'cocoa.menu.hideOthers': '隐藏其他',
    'cocoa.menu.showAll': '全部显示',
    'cocoa.menu.quit': '退出',
    'cocoa.menu.fullscreen': '进入全屏幕',
    'cocoa.menu.cut': '剪切',
    'cocoa.menu.copy': '拷贝',
    'cocoa.menu.paste': '粘贴',
    'cocoa.menu.selectAll': '全选',
    'windows.fileFilter.allFiles': '所有文件',
    'windows.fileFilter.otherFiles': '其他文件类型',
    'linux.openFile': '打开文件',
    'linux.openFiles': '打开多个文件',
    'linux.openFolder': '打开文件夹',
}


def _os_localization() -> dict | None:
    """pywebview localization dict for the OS language (None = English)."""
    import locale as sys_locale
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            lang = (sys_locale.getdefaultlocale()[0] or "").lower()
    except Exception:
        lang = ""
    if lang.startswith("zh"):
        return _ZH_LOCALIZATION
    return None


# Correct Evergreen Runtime client GUID — the installer's old probe used
# {F3017226-FE2A-4295-8BEE-13A6279B0638}, which does not exist, so it
# always reported the runtime as missing.
_WEBVIEW2_RUNTIME_GUID = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
_WEBVIEW2_DOWNLOAD_URL = "https://developer.microsoft.com/microsoft-edge/webview2/"


def _webview2_available() -> bool:
    """True when a WebView2 runtime (any channel) is registered.

    Without it pywebview silently degrades to the IE11 engine (MSHTML),
    which cannot execute the SPA bundle — the user gets a blank window
    with no error. Mirrors pywebview's own registry probe
    (webview/platforms/winforms.py `_is_chromium`).
    """
    if sys.platform != "win32":
        return True
    import winreg

    client_guids = [
        _WEBVIEW2_RUNTIME_GUID,                       # Evergreen runtime
        "{2CD8A007-E189-409D-A2C8-9AF4EF3C72AA}",     # Beta
        "{0D50BFEC-CD6A-4F9A-964C-C7416E3ACB10}",     # Dev
        "{65C35B14-6C1D-4122-AC46-7148CC9D6497}",     # Canary
    ]
    roots = [
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\EdgeUpdate\Clients"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\EdgeUpdate\Clients"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients"),
    ]
    for root, base in roots:
        for guid in client_guids:
            try:
                with winreg.OpenKey(root, base + "\\" + guid) as key:
                    pv, _ = winreg.QueryValueEx(key, "pv")
                    if pv and pv != "0.0.0.0":
                        return True
            except OSError:
                continue
    return False


def _fail_missing_webview2() -> None:
    """Native dialog (bilingual by OS language) + download page, then exit.

    Must not use pywebview for the dialog — pywebview IS the broken layer.
    """
    import ctypes
    import webbrowser

    if _os_localization() is not None:
        title = "Orbital 需要 WebView2"
        msg = (
            "未检测到 Microsoft Edge WebView2 运行时，Orbital 无法显示界面。\n\n"
            "点击“确定”将打开下载页面，安装后请重新启动 Orbital。"
        )
    else:
        title = "Orbital needs WebView2"
        msg = (
            "The Microsoft Edge WebView2 Runtime was not found, so Orbital "
            "cannot display its interface.\n\n"
            "Click OK to open the download page, then restart Orbital after "
            "installing the runtime."
        )
    try:
        ctypes.windll.user32.MessageBoxW(0, msg, title, 0x00000040)  # MB_ICONINFORMATION
    except Exception:
        pass
    try:
        webbrowser.open(_WEBVIEW2_DOWNLOAD_URL)
    except Exception:
        pass
    sys.exit(1)


def open_window(port: int):
    global _window
    import webview

    # Fail LOUDLY when the WebView2 runtime is absent — pywebview would
    # otherwise fall back to the IE11 engine and show a blank window.
    if not _webview2_available():
        _fail_missing_webview2()

    # If window already exists (hidden), just show it
    if _window is not None:
        try:
            _window.show()
        except Exception:
            pass
        return

    # macOS window-chrome state shared between the `loaded` installer and the
    # fullscreen handlers. A dict rather than two locals because the installer
    # writes from the AppKit main queue while the handlers read from pywebview
    # event threads.
    chrome_state = {"strip": None, "fullscreen": False}

    class Api:
        def pick_folder(self):
            result = window.create_file_dialog(
                webview.FOLDER_DIALOG,
                directory=os.path.expanduser("~"),
            )
            return result[0] if result else None

    # Set icon in background thread (needs window to exist first)
    icon_thread = threading.Thread(target=_set_window_icon, daemon=True)
    icon_thread.start()

    def _activate_macos():
        """Force window to foreground on macOS Sequoia.

        pywebview uses the deprecated activateIgnoringOtherApps: API which
        macOS 14+ silently ignores.  This callback fires after the run loop
        starts and uses the modern activateWithOptions: API instead.
        """
        if sys.platform != "darwin":
            return
        import time
        time.sleep(0.5)  # let the NSApp run loop settle
        try:
            import AppKit
            app = AppKit.NSRunningApplication.currentApplication()
            app.activateWithOptions_(AppKit.NSApplicationActivateIgnoringOtherApps)
        except Exception:
            pass

    def _on_closing():
        """Intercept window close — hide instead of closing.

        On macOS: Cmd+Q and Dock→Quit go through applicationShouldTerminate_
        which also fires this event — detect via call stack and allow quit.
        On Windows/Linux: quit happens via system tray menu (os._exit).
        """
        if sys.platform == "darwin":
            # During app termination (Cmd+Q, Dock→Quit), the call stack
            # includes applicationShouldTerminate_.  Allow the quit.
            frame = sys._getframe()
            while frame is not None:
                if frame.f_code.co_name == "applicationShouldTerminate_":
                    return True
                frame = frame.f_back
        # Hide instead of close — app stays alive in tray (Windows) or Dock (macOS)
        try:
            if sys.platform == "darwin":
                window.native.miniaturize_(window.native)
            else:
                window.hide()
        except Exception:
            pass
        return False

    def _reveal_traffic_lights():
        """Put the macOS window buttons back after the frameless style hides them.

        pywebview's `frameless` on macOS is not a truly frameless window: it
        keeps NSTitled|NSClosable|NSMiniaturizable, adds NSFullSizeContentView,
        and makes the titlebar transparent — then hides all three standard
        buttons (webview/platforms/cocoa.py). We want every part of that except
        the last step. Content runs edge-to-edge under an empty titlebar, but
        the traffic lights stay where users expect them, and the still-titled
        style mask keeps the fullscreen titlebar auto-hide (lights hide in
        fullscreen, slide back on hover at the top edge) for free.

        What it does NOT keep for free is drag. A transparent titlebar is
        hit-transparent everywhere except these three buttons, so mouse-downs
        in the band land on the WKWebView and AppKit never starts a window
        drag — bug #60, shipped v0.8.0 → v0.9.0 on the strength of a comment
        here that claimed the opposite. install_titlebar_drag_strip() supplies
        the drag region; see its docstring for the measurements.

        Runs on `before_show` rather than `shown` so the buttons are already
        visible on the first frame — `window.native` doesn't exist until
        webview.start() builds the NSWindow, so this can't happen at
        create_window time.
        """
        if sys.platform != "darwin":
            return
        try:
            import AppKit

            native = window.native
            for button in (
                AppKit.NSWindowCloseButton,
                AppKit.NSWindowMiniaturizeButton,
                AppKit.NSWindowZoomButton,
            ):
                native.standardWindowButton_(button).setHidden_(False)
        except Exception:
            # Never let window chrome cosmetics block the app from opening —
            # worst case the user gets a window with no visible traffic lights
            # but a working Cmd+W / Cmd+M / Dock menu.
            pass

    def _set_chrome_mode(mode: str):
        """Tell the SPA which window chrome it is living under.

        In fullscreen macOS auto-hides the titlebar and re-reveals it as an
        overlay on hover at the top edge, so the traffic-light gutter the SPA
        reserves is dead space there. Flipping this attribute collapses it —
        `--titlebar-h` falls back to 0px for any value that isn't 'mac-inline',
        so the fullscreen case needs no CSS branch of its own.

        The native drag strip has to move in lockstep: with the gutter
        collapsed, real UI sits in the top band, and a strip left visible there
        would swallow its clicks. `set_drag_strip_hidden` does its own
        main-thread hop, because these handlers run off it (below).

        Calling evaluate_js here is safe specifically because `maximized` and
        `restored` are non-locking events, which pywebview runs on their own
        thread (webview/event.py). evaluate_js queues its script onto the main
        thread and then blocks waiting for the result, so calling it from a
        *locking* handler like `before_show` would deadlock the window.
        """
        if not inline_titlebar:
            return
        fullscreen = mode == "fullscreen"
        # Recorded as well as applied: `loaded` installs the strip
        # asynchronously, so it may not exist yet when this first runs.
        chrome_state["fullscreen"] = fullscreen
        set_drag_strip_hidden(chrome_state["strip"], fullscreen)
        try:
            window.evaluate_js(f"document.documentElement.dataset.chrome = '{mode}'")
        except Exception:
            pass

    def _install_drag_strip():
        """Attach the native drag strip once the WKWebView is the contentView.

        Bound to `loaded` because that is the earliest point the real content
        view exists — pywebview swaps it in inside webView_didFinishNavigation_
        — and hopped to the main queue because `loaded` is dispatched on a
        worker thread, where AppKit geometry calls raise. `loaded` re-fires when
        the origin guard reloads the SPA, which is why the installer is
        idempotent.
        """
        if not inline_titlebar:
            return

        def _install():
            # Runs as a block on the AppKit main queue, where an escaping
            # Python exception would surface inside pywebview's run loop.
            try:
                strip = install_titlebar_drag_strip(window.native)
            except Exception:
                return
            if strip is None:
                return
            chrome_state["strip"] = strip
            # Catch up with a fullscreen transition that beat the install.
            set_drag_strip_hidden(strip, chrome_state["fullscreen"])

        _run_on_main_thread(_install)

    def _on_loaded():
        """Origin guard: catch a same-frame navigation that escaped the SPA.

        Fix A (MarkdownContent.tsx) already externalizes known external links
        via target="_blank", which pywebview routes to the system browser. This
        is defense-in-depth for anything that slips past it (a stray anchor
        without target=_blank, a window.location script, a future component) —
        the desktop shell has no browser chrome, so an in-place navigation away
        from the SPA would otherwise strand the user with no way back short of
        quit + reopen. Reloading the SPA below re-fires `loaded`, but at the
        app's own origin, which is a no-op — no re-entry loop.
        """
        import webbrowser

        try:
            current = window.get_current_url()
            if not _is_foreign_url(current, port):
                return
        except Exception:
            # Never let the guard crash pywebview's event dispatch — an
            # unclassifiable URL just means we leave the window alone.
            return
        try:
            webbrowser.open(current)
        except Exception:
            pass
        try:
            window.load_url(app_url)
        except Exception:
            pass

    # macOS only: reclaim the native titlebar. Windows keeps its native frame —
    # WebView2 has no transparent-titlebar equivalent, so matching this there
    # would mean hand-drawing minimize/maximize/close plus drag and snap.
    inline_titlebar = sys.platform == "darwin"

    app_url = f"http://127.0.0.1:{port}"
    if inline_titlebar:
        # Tells the SPA to reserve a gutter for the traffic lights now floating
        # over its top-left corner. A query param rather than `window.pywebview`
        # (which is injected asynchronously) so the gutter is there on the first
        # paint instead of popping in. Windows and the relay/mobile browser
        # serve the same bundle and never see it.
        app_url += "/?chrome=mac-inline"

    _window = webview.create_window(
        title="Orbital",
        url=app_url,
        width=1200,
        height=800,
        min_size=(800, 600),
        text_select=True,
        js_api=Api(),
        frameless=inline_titlebar,
        # Must be False whenever frameless is on: pywebview implements easy_drag
        # by making the *entire* WebKitHost surface a drag handle, so dragging to
        # select text in the chat would drag the window instead. With it off,
        # drag comes from the native (now transparent) titlebar band only.
        easy_drag=False,
    )
    window = _window
    window.events.closing += _on_closing
    window.events.loaded += _on_loaded
    # Also on `loaded` — that is the first moment the WKWebView is the
    # contentView. Ordered after the origin guard so the guard gets first look;
    # the reload it may trigger re-fires this, which the installer absorbs.
    window.events.loaded += _install_drag_strip
    window.events.before_show += _reveal_traffic_lights
    # macOS fires `maximized` on entering fullscreen and `restored` on leaving
    # it (webview/platforms/cocoa.py). `restored` also fires on un-minimize,
    # which is harmless here: a minimized window is never fullscreen.
    window.events.maximized += lambda: _set_chrome_mode("fullscreen")
    window.events.restored += lambda: _set_chrome_mode("mac-inline")
    start_kwargs = {}
    localization = _os_localization()
    if localization:
        start_kwargs["localization"] = localization
    if sys.platform == "win32":
        # Declare the only renderer that can run the SPA. (pywebview still
        # falls back to MSHTML internally if WebView2 vanished after the
        # guard above — the guard is what produces the loud failure.)
        start_kwargs["gui"] = "edgechromium"
    webview.start(icon=resolve_window_icon_path(), func=_activate_macos, **start_kwargs)


def run_sandbox_setup():
    """Headless sandbox setup — called by installer via --setup-sandbox flag."""
    import asyncio
    import logging

    # In frozen mode stderr may be devnull; configure logging to handle that.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger("orbital.setup")

    try:
        from agent_os.platform import create_platform_provider

        provider = create_platform_provider()
        caps = provider.get_capabilities()

        if caps.platform == "null" or caps.setup_complete:
            logger.info("Sandbox setup not needed (platform=%s, complete=%s)", caps.platform, caps.setup_complete)
            sys.exit(0)

        logger.info("Running sandbox setup...")
        result = asyncio.run(provider.setup())

        if result.success:
            logger.info("Sandbox setup completed successfully.")
            sys.exit(0)
        else:
            logger.error("Sandbox setup failed: %s", result.error)
            sys.exit(1)
    except Exception:
        logger.exception("Sandbox setup crashed")
        sys.exit(1)


def run_sandbox_teardown():
    """Headless sandbox teardown — called by uninstaller via --teardown-sandbox flag.

    Cleans up: sandbox user/ACLs, API key from keychain, user credentials.
    CRITICAL: Always exits 0. A failed teardown must never block uninstall.
    """
    import asyncio
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger("orbital.teardown")

    # 1. Sandbox teardown
    try:
        from agent_os.platform import create_platform_provider

        provider = create_platform_provider()
        caps = provider.get_capabilities()

        if caps.platform == "null":
            logger.info("Sandbox teardown not needed (platform=%s)", caps.platform)
        else:
            logger.info("Running sandbox teardown...")
            result = asyncio.run(provider.teardown())

            if result.success:
                logger.info("Sandbox teardown completed successfully.")
            else:
                logger.warning("Sandbox teardown reported failure: %s", result.error)
    except Exception:
        logger.warning("Sandbox teardown crashed (ignored)", exc_info=True)

    # 2. Remove API key from OS keychain
    try:
        from agent_os.daemon_v2.credential_store import ApiKeyStore
        ApiKeyStore().delete_api_key()
        logger.info("API key removed from keychain.")
    except Exception:
        logger.warning("API key cleanup failed (ignored)", exc_info=True)

    # 3. Remove user credentials from OS keychain
    try:
        from agent_os.desktop.migration import DATA_DIR
        from agent_os.daemon_v2.credential_store import UserCredentialStore
        meta_path = os.path.join(DATA_DIR, "credential-meta.json")
        store = UserCredentialStore(meta_path=meta_path)
        creds = store.list_all()
        for cred in creds:
            store.delete(cred["name"])
        if creds:
            logger.info("Removed %d user credential(s) from keychain.", len(creds))
    except Exception:
        logger.warning("User credential cleanup failed (ignored)", exc_info=True)

    # Always exit 0 — never block uninstall
    sys.exit(0)


def _get_log_path() -> str:
    """Return the platform-appropriate log directory path for error messages."""
    if sys.platform == "win32":
        return os.path.join(os.environ.get("APPDATA", ""), "Orbital", "logs")
    elif sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Logs", "Orbital")
    else:
        return os.path.join(os.path.expanduser("~"), ".orbital", "logs")


# Every desktop shell puts its own icon in the Windows notification area, so
# "one Orbital" has to mean one process (bug #54). The name is unprefixed and
# therefore lives in the caller's *session* namespace ("Local\"), not "Global\":
# a second logged-in user (fast user switching, RDP) has their own notification
# area and must get their own Orbital, not a refusal.
SINGLE_INSTANCE_MUTEX_NAME = "OrbitalDesktopShell"

# The mutex is owned by its handle, and a kernel mutex dies with the process
# holding it — that is why this beats the daemon PID file, which needs a
# liveness dance. Parked in a module global so it is never garbage collected.
_single_instance_mutex = None

# Flags main() knows how to act on. Anything else must not reach the GUI path.
_RECOGNIZED_FLAGS = ("--setup-sandbox", "--teardown-sandbox", "--smoke-test")


def _unrecognized_args(argv) -> list:
    """Return the arguments main() does not understand.

    `-psn_*` is what macOS LaunchServices hands a bundled app on launch; it is
    not a request to do anything and must never keep Orbital from starting.
    """
    return [
        arg for arg in argv
        if arg not in _RECOGNIZED_FLAGS and not arg.startswith("-psn_")
    ]


def _acquire_single_instance(name: str = SINGLE_INSTANCE_MUTEX_NAME) -> bool:
    """Claim the desktop-shell mutex. True means this process is the only shell.

    Non-Windows platforms and any ctypes failure return True: the guard must
    never be the reason Orbital refuses to start.
    """
    global _single_instance_mutex

    if sys.platform != "win32":
        return True

    try:
        import ctypes

        ERROR_ALREADY_EXISTS = 183
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.CreateMutexW(None, False, name)
        last_error = kernel32.GetLastError()
        if not handle:
            return True
        _single_instance_mutex = handle
        return last_error != ERROR_ALREADY_EXISTS
    except Exception:
        return True


def _activate_existing_window(title: str = "Orbital") -> bool:
    """Surface the already-running shell's window. True if one was found.

    The running instance may have been hidden by the close-to-tray path
    (`window.hide()`), so this both un-hides and un-minimizes it. FindWindowW
    enumerates by title regardless of visibility.
    """
    if sys.platform != "win32":
        return False

    try:
        import ctypes

        SW_SHOW = 5
        SW_RESTORE = 9
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, title)
        if not hwnd:
            return False
        user32.ShowWindow(hwnd, SW_SHOW)
        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.SetForegroundWindow(hwnd)
        return True
    except Exception:
        return False


def main():
    # Refuse to start the GUI on arguments we do not understand. A stray argv
    # used to fall straight through to the daemon + window + tray path — which
    # is how request_elevation()'s re-exec of the frozen exe would have spawned
    # a second tray icon.
    unknown = _unrecognized_args(sys.argv[1:])
    if unknown:
        try:
            print(
                "Orbital: unrecognized argument(s): " + " ".join(unknown),
                file=sys.stderr,
            )
            print(
                "Usage: Orbital [" + " | ".join(_RECOGNIZED_FLAGS) + "]",
                file=sys.stderr,
            )
        except Exception:
            pass
        sys.exit(2)

    # Handle CLI flags before any daemon/GUI setup
    if "--setup-sandbox" in sys.argv:
        run_sandbox_setup()
        return

    if "--teardown-sandbox" in sys.argv:
        run_sandbox_teardown()
        return

    if "--smoke-test" in sys.argv:
        sys.exit(run_smoke_test())

    # Single-instance guard — before any daemon, window, or tray work, because
    # a second shell means a second tray icon. Deliberately *after* the flag
    # handlers above: those create no tray, and the installer runs
    # --setup-sandbox / --teardown-sandbox while an instance may be live, so
    # they must never be swallowed by this guard.
    if not _acquire_single_instance():
        if _activate_existing_window():
            sys.exit(0)
        # Fail OPEN: the mutex is held but nothing answers to it — a wedged
        # instance with no window. Being locked out of Orbital entirely is a
        # worse bug than the duplicate icon this guard exists to prevent.

    from agent_os.desktop.migration import run_migrations

    PORT = 8000
    run_migrations()
    _inherit_shell_path()
    _disable_app_nap()
    _prepare_bundled_ripgrep()

    if is_already_running(PORT):
        port = PORT
    else:
        port = find_free_port(PORT)
        server, thread = start_daemon(port)

        if not wait_for_daemon(port):
            import webview
            webview.create_window(
                "Orbital \u2014 Error",
                html=f"<h2>Failed to start daemon</h2><p>Check logs in {_get_log_path()}</p>",
            )
            webview.start()
            return

    # Daemon is up — provision Chromium (extract bundled archive, or a
    # backed-off background download) without blocking the UI.
    from agent_os.desktop.migration import provision_browsers_background
    provision_browsers_background()

    def shutdown():
        os._exit(0)

    # On macOS, pystray's Cocoa backend must run on the main thread, but
    # pywebview also requires it.  Initialising pystray from a background
    # thread corrupts AppKit state and causes an NSApplication assertion
    # crash.  macOS .app bundles already get a Dock icon, so the tray is
    # unnecessary — skip it entirely on Darwin.
    if sys.platform != "darwin":
        from agent_os.desktop.tray import start_tray

        tray_thread = threading.Thread(
            target=start_tray,
            args=(port, lambda: open_window(port), shutdown),
            daemon=True,
        )
        tray_thread.start()

    open_window(port)

    if sys.platform == "darwin":
        # macOS: no system tray, so window close = app close.
        # This is standard macOS behavior for non-document-based apps.
        os._exit(0)
    else:
        # Windows/Linux: system tray keeps app alive after window close.
        # User quits via tray menu → shutdown() → os._exit(0).
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            # Any exit from here has to take the icon with it. Windows keeps
            # showing a dead process's tray icon until the user sweeps the
            # pointer over it, so an un-deleted icon is a visible ghost.
            from agent_os.desktop.tray import hide_tray

            hide_tray()


if __name__ == "__main__":
    main()
