# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import os
import time
import threading
import socket


def _get_log_path() -> str:
    """Return the platform-appropriate log directory path for error messages.

    Defined before the stderr redirect below so the redirect and the
    "Failed to start daemon" dialog point at the same directory.
    """
    if sys.platform == "win32":
        return os.path.join(os.environ.get("APPDATA", ""), "Orbital", "logs")
    elif sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Logs", "Orbital")
    else:
        return os.path.join(os.path.expanduser("~"), ".orbital", "logs")


#: Spec 134 A4: the stderr log is appended across launches, and moves to
#: ``orbital-stderr.log.1`` once it passes this size.
STDERR_LOG_MAX_BYTES = 1024 * 1024
#: The first line each launch writes, so the failure dialog reads only this
#: launch's lines.
_LAUNCH_MARKER = "=== Orbital launch"


def _open_stderr_log(log_dir: str):
    """Open ``orbital-stderr.log`` for this launch: append, size-capped.

    It used to be opened ``"w"``, so a relaunch that failed against a frozen
    instance overwrote the frozen run's stderr — the one piece of evidence
    (spec 134 W6). The rotation is best-effort: on Windows a running
    instance may still hold the file, and then this launch just appends.
    """
    path = os.path.join(log_dir, "orbital-stderr.log")
    try:
        if os.path.getsize(path) > STDERR_LOG_MAX_BYTES:
            os.replace(path, path + ".1")
    except OSError:
        pass
    # utf-8: tracebacks can carry non-ANSI text (e.g. Chinese filenames);
    # the platform-default codec would raise while writing the evidence.
    f = open(path, "a", encoding="utf-8", errors="replace")
    f.write(f"\n{_LAUNCH_MARKER} {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"pid {os.getpid()} ===\n")
    f.flush()
    return f


# PyInstaller windowed mode (console=False) sets sys.stdout/stderr to None.
# Uvicorn's log formatter calls sys.stderr.isatty() on init and crashes.
# Redirect to a log file so errors are visible for debugging.
if getattr(sys, "frozen", False) and sys.stderr is None:
    _log_dir = _get_log_path()
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = _open_stderr_log(_log_dir)
    sys.stdout = _log_file
    sys.stderr = _log_file

# Set AppUserModelID so Windows taskbar shows the Orbital icon, not Python's
if sys.platform == "win32":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("orbital.desktop.app")


def find_free_port(preferred: int = 8000) -> int:
    # Probe BOTH addresses the app depends on: uvicorn binds the wildcard, and
    # the health probe + SPA connect via 127.0.0.1. Bind semantics differ per
    # platform. Windows only conflicts EXACT-duplicate binds (specific-over-
    # wildcard and wildcard-over-specific both coexist by default) — so the
    # old loopback-only probe bound 127.0.0.1:8000 clean OVER a foreign
    # 0.0.0.0:8000 socket and uvicorn's own wildcard bind then died on the
    # duplicate with WSAEADDRINUSE (bug #75); the loopback leg also catches a
    # foreign loopback server that would otherwise steal 127.0.0.1 routing
    # from a "successful" wildcard bind. On macOS/Linux wildcard-vs-specific
    # conflicts both ways, so the wildcard leg covers specific-interface
    # occupants too. A port counts as free only when both legs bind cleanly.
    # Bind only, never listen() (listening triggers the macOS app-firewall
    # prompt on unsigned dev builds), and no SO_REUSEADDR (it would mask the
    # very conflicts this probe exists to detect).
    def _binds_cleanly(candidate: int):
        port = candidate
        for addr in ("0.0.0.0", "127.0.0.1"):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind((addr, port))
                port = sock.getsockname()[1]
            except OSError:
                return None
            finally:
                sock.close()
        return port

    port = _binds_cleanly(preferred)
    if port is not None:
        return port
    for _ in range(5):
        port = _binds_cleanly(0)
        if port is not None:
            return port
    # Last resort: a wildcard-bindable ephemeral port; boot_daemon_with_retry
    # covers any residual conflict.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def _local_opener():
    """urllib opener for loopback health probes, bypassing every proxy.

    A bare urlopen honors http_proxy env vars and the Windows registry proxy —
    on machines running Clash/V2Ray/corp agents that routes the 127.0.0.1
    probe through the proxy and reports a healthy daemon as down.
    """
    import urllib.request
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def flush_telemetry_before_exit(port: int, timeout: float = 3.0) -> bool:
    """Ask the in-process daemon to send today's telemetry ping now (spec 111
    A2). Both desktop exit paths end in ``os._exit(0)``, which skips the
    daemon's shutdown hook, so the flush has to happen here, over loopback,
    before the process is torn down. Bounded and silent: a dead daemon or a
    slow network must never delay or block quitting."""
    import urllib.request

    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v2/settings/telemetry/flush",
            data=b"", method="POST",
        )
        resp = _local_opener().open(req, timeout=timeout)
        return 200 <= resp.status < 300
    except Exception:
        return False


def is_already_running(port: int = 8000) -> bool:
    try:
        resp = _local_opener().open(f"http://127.0.0.1:{port}/api/v2/settings", timeout=2)
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
    nvm, ...). Spec 115: the capture lives in ``agent_os.platform.shell_path``
    so Settings → Refresh can re-run the same thing; it probes a login shell
    AND an interactive login shell (``.zshrc`` is where version managers put
    their PATH line) and never blocks launch — a failed probe keeps the
    current PATH.
    """
    from agent_os.platform.shell_path import merge_shell_path_into_environ

    merge_shell_path_into_environ()


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


def wait_for_daemon(port: int, timeout: int = 15, server=None, thread=None) -> bool:
    opener = _local_opener()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = opener.open(f"http://127.0.0.1:{port}/api/v2/settings", timeout=2)
            if resp.status == 200:
                return True
        except Exception:
            pass
        # Fail fast on a dead daemon: a failed uvicorn bind logs ERROR and
        # returns from server.run() without raising, so the thread dies with
        # server.started False — polling HTTP for the rest of the window only
        # delays the error dialog (bug #75).
        if (
            thread is not None
            and not thread.is_alive()
            and (server is None or not getattr(server, "started", False))
        ):
            return False
        time.sleep(0.5)
    return False


def boot_daemon_with_retry(preferred: int):
    """Start the daemon, retrying once on a fresh random port if the first
    attempt dies outright (bug #75).

    The port probe is inherently TOCTOU-racy, and bind failures can have
    probe-invisible causes (Windows excluded port ranges, races) — one retry
    converts every port-conflict variant into a working app. A live-but-slow
    daemon is NOT retried: a second daemon beside it would just lose the
    pid-file race. Returns (port, server, thread); port is None on failure.
    """
    port = find_free_port(preferred)
    server, thread = start_daemon(port)
    if wait_for_daemon(port, server=server, thread=thread):
        return port, server, thread
    if not thread.is_alive():
        port = find_free_port(0)
        server, thread = start_daemon(port)
        if wait_for_daemon(port, server=server, thread=thread):
            return port, server, thread
    return None, server, thread


def _stderr_log_tail(max_lines: int = 5) -> str:
    """Last ERROR lines (or last lines) of orbital-stderr.log, for the failure
    dialog — the generic "check logs" message points at a folder Windows hides
    by default, which cost a full support round-trip on bug #75.
    """
    try:
        try:
            sys.stderr.flush()  # the frozen build's stderr IS this file
        except Exception:
            pass
        path = os.path.join(_get_log_path(), "orbital-stderr.log")
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [line.rstrip() for line in f if line.strip()]
        # The file is appended across launches: only this launch's lines.
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith(_LAUNCH_MARKER):
                lines = lines[i + 1:]
                break
        errors = [line for line in lines if "ERROR" in line]
        return "\n".join((errors or lines)[-max_lines:])
    except Exception:
        return ""


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
    port, _server, _thread = boot_daemon_with_retry(8000)
    ok = port is not None
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


def local_file_entries(paths) -> list[dict]:
    """The composer's path attachments for these picked/dropped paths.

    Desktop app only: a file is attached by its absolute local path, never
    uploaded. Each entry is ``{path, name, size, mime}``; ``path`` is the
    realpath, ``name`` the file's own name. Folders, missing paths and
    anything that is not a regular file are left out.
    """
    import mimetypes
    import stat as _stat
    out: list[dict] = []
    for p in paths or ():
        if not isinstance(p, str) or not p or not os.path.isabs(p):
            continue
        real = os.path.realpath(p)
        try:
            st = os.stat(real)
        except OSError:
            continue
        if not _stat.S_ISREG(st.st_mode):
            continue
        out.append({
            "path": real,
            "name": os.path.basename(p),
            "size": st.st_size,
            "mime": mimetypes.guess_type(real)[0] or "application/octet-stream",
        })
    return out


def pick_files_with(open_dialog) -> list[dict]:
    """Run the native multi-select open dialog and return the picked files
    as :func:`local_file_entries`. ``open_dialog()`` → paths or None (cancel)."""
    try:
        picked = open_dialog()
    except Exception:
        return []
    return local_file_entries(list(picked or ()))


def dropped_file_entries(event) -> list[dict]:
    """The real paths of files dropped on the window, from pywebview's
    serialized drop event (``dataTransfer.files[i].pywebviewFullPath``)."""
    try:
        files = (event or {}).get("dataTransfer", {}).get("files", []) or []
    except AttributeError:
        return []
    paths = [f.get("pywebviewFullPath") for f in files if isinstance(f, dict)]
    return local_file_entries([p for p in paths if p])


# The SPA listens for this event to turn a drop into path attachments.
NATIVE_DROP_EVENT = "orbital:native-drop"


def native_drop_js(entries: list[dict]) -> str:
    """JS that hands dropped files' real paths to the SPA."""
    import json
    return (f"window.dispatchEvent(new CustomEvent({json.dumps(NATIVE_DROP_EVENT)}, "
            f"{{detail: {json.dumps(entries)}}}))")


def pick_files_in(window) -> list[dict]:
    """The native multi-select open dialog of ``window``, as path entries."""
    import webview
    open_kind = getattr(getattr(webview, "FileDialog", None), "OPEN", None)
    if open_kind is None:
        open_kind = webview.OPEN_DIALOG
    return pick_files_with(lambda: window.create_file_dialog(
        open_kind,
        directory=os.path.expanduser("~"),
        allow_multiple=True,
    ))


def install_native_drop(window) -> None:
    """Drag and drop onto a composer attaches by real path (desktop app).

    A file dropped into the web view arrives as a File with no path. With a
    Python drop listener registered, pywebview records the pasteboard's file
    URLs natively and puts each file's real path on the serialized drop event
    (``pywebviewFullPath``); this handler hands those paths to the SPA.
    ``__orbitalNativeDrop`` tells the SPA to wait for them; a file that comes
    back without a path is uploaded as before.

    Runs on every ``loaded``: a reload drops the page's JS listener. After
    one, pywebview may call more than one Python handler per drop; the SPA
    takes the first event per drop and ignores the rest.
    """
    try:
        from webview.dom import DOMEventHandler
    except Exception:
        return

    def _on_drop(event):
        entries = dropped_file_entries(event)
        try:
            window.evaluate_js(native_drop_js(entries))
        except Exception:
            pass

    try:
        window.dom.document.events.drop += DOMEventHandler(_on_drop, False, False)
        window.evaluate_js("window.__orbitalNativeDrop = true")
    except Exception:
        # Older pywebview / no DOM API: drops keep uploading.
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


# ---------------------------------------------------------------------------
# Spec 115 §6 (O2) — the close button hides the window; a Dock click or app
# activation brings it back; Cmd+Q and Dock → Quit still quit.
# ---------------------------------------------------------------------------

#: Whether the window is currently hidden *by the close button* (as opposed
#: to minimized by the user, which AppKit restores on its own).
_close_state: dict[str, bool] = {"hidden": False}
_REOPEN_DELEGATE_CLASS = None


def _close_requests_quit() -> bool:
    """True when the closing event comes from app termination.

    On macOS, Cmd+Q and Dock → Quit go through ``applicationShouldTerminate_``,
    which also fires pywebview's ``closing`` event; walking the stack is how
    the two are told apart.
    """
    frame = sys._getframe()
    while frame is not None:
        if frame.f_code.co_name == "applicationShouldTerminate_":
            return True
        frame = frame.f_back
    return False


# ---------------------------------------------------------------------------
# Spec 134 Part B, decision (a): background jobs do not survive quitting.
# Quitting while one runs asks first; quitting ends every job tree.
# ---------------------------------------------------------------------------

# AppleEvent quit reasons (``kAEQuitReason``) that mean the session is ending.
_AE_QUIT_REASON = 0x77687920        # 'why?'
_AE_SESSION_ENDING = {
    0x6C6F676F,  # 'logo' kAELogOut
    0x726C6763,  # 'rlgc' kAEReallyLogOut
    0x72657374,  # 'rest' kAERestart
    0x72727374,  # 'rrst' kAEShowRestartDialog / really restart
    0x73687574,  # 'shut' kAEShutDown
    0x7273646E,  # 'rsdn' kAEReallyShutDown
}


def _quit_jobs_prompt(n: int) -> tuple[str, str, str, str]:
    """(title, message, quit label, cancel label) in the OS language."""
    if _os_localization() is not None:
        return ("退出 Orbital？", f"还有 {n} 个后台任务正在运行。退出会结束它们。",
                "退出", "取消")
    return ("Quit Orbital?",
            f"{n} background job(s) are still running. Quitting stops them.",
            "Quit", "Cancel")


def _system_is_ending_session() -> bool:
    """True when this quit comes from a logout, restart or shutdown, which a
    confirmation must never block. Only macOS routes those through the same
    quit path; Windows' tray Quit is always the user's own click."""
    if sys.platform != "darwin":
        return False
    try:
        import Foundation

        event = Foundation.NSAppleEventManager.sharedAppleEventManager().currentAppleEvent()
        if event is None:
            return False
        reason = event.attributeDescriptorForKeyword_(_AE_QUIT_REASON)
        return reason is not None and reason.enumCodeValue() in _AE_SESSION_ENDING
    except Exception:
        return False


def _ask_quit_with_jobs(n: int) -> bool:
    """Native Quit / Cancel. True = quit. Any failure to ask quits (a broken
    dialog must never trap the user in the app)."""
    title, message, quit_label, cancel_label = _quit_jobs_prompt(n)
    try:
        if sys.platform == "win32":
            import ctypes

            MB_OKCANCEL, MB_ICONWARNING, MB_SETFOREGROUND, IDOK = 0x1, 0x30, 0x10000, 1
            answer = ctypes.windll.user32.MessageBoxW(
                0, message, title, MB_OKCANCEL | MB_ICONWARNING | MB_SETFOREGROUND)
            return answer == IDOK
        if sys.platform == "darwin":
            if threading.current_thread() is threading.main_thread():
                import AppKit

                alert = AppKit.NSAlert.alloc().init()
                alert.setMessageText_(title)
                alert.setInformativeText_(message)
                alert.addButtonWithTitle_(quit_label)
                alert.addButtonWithTitle_(cancel_label)
                return alert.runModal() == AppKit.NSAlertFirstButtonReturn
            import subprocess

            from agent_os.utils.subprocess_flags import win_no_window_flags

            script = (
                f"display dialog {_applescript_str(message)} "
                f"with title {_applescript_str(title)} "
                f"buttons {{{_applescript_str(cancel_label)}, {_applescript_str(quit_label)}}} "
                f"default button 2 cancel button 1 with icon caution"
            )
            result = subprocess.run(["osascript", "-e", script], capture_output=True,
                                    encoding="utf-8", errors="replace", timeout=600,
                                    creationflags=win_no_window_flags())
            return result.returncode == 0 and quit_label in (result.stdout or "")
    except Exception:
        _boot_logger().exception("Could not show the quit-with-jobs dialog")
    return True


def confirm_quit() -> bool:
    """True when Orbital may quit now: no background job is running, the OS
    is ending the session, or the user chose Quit."""
    from agent_os.daemon_v2 import jobs

    n = jobs.running_job_count()
    if n <= 0 or _system_is_ending_session():
        return True
    return _ask_quit_with_jobs(n)


def _exit_app(port: int) -> None:
    """The desktop's one way out: end every background job tree, flush
    telemetry, release the PID file, exit (``os._exit`` skips the app's
    shutdown hooks, so the jobs are ended here)."""
    from agent_os.daemon_v2 import jobs
    from agent_os.utils import pid_file

    try:
        jobs.stop_all_jobs()
    except Exception:
        _boot_logger().exception("Could not stop background jobs on quit")
    flush_telemetry_before_exit(port)
    pid_file.release_pid_file()  # os._exit skips atexit (spec 134 W7)
    os._exit(0)


def _handle_window_close(window) -> bool:
    """pywebview ``closing`` handler: False keeps the window (hidden), True
    lets it close.

    macOS used to *miniaturize* here: the window shrank into the Dock's
    minimized zone, which users read as "I pressed the wrong button" (#57).
    The convention for a document-less app whose process keeps running
    (Slack, Discord, Spotify) is ``orderOut:`` — pywebview's ``hide()`` — and
    a reopen on Dock click (:func:`_reopen_delegate_class`). The WKWebView is
    not torn down while hidden, so the WebSocket state is live on reshow.
    """
    if sys.platform == "darwin" and _close_requests_quit():
        # Cmd+Q / menu Quit / Dock Quit: False cancels the termination.
        return confirm_quit()
    try:
        window.hide()
        _close_state["hidden"] = True
    except Exception:
        pass
    return False


def _restore_hidden_window(window) -> None:
    """Show the window again if the close button hid it. Idempotent."""
    if not _close_state["hidden"]:
        return
    _close_state["hidden"] = False
    try:
        window.show()
    except Exception:
        pass


def _reopen_delegate_class(base, restore):
    """Subclass pywebview's NSApplication delegate with the reopen hooks.

    ``applicationShouldHandleReopen:hasVisibleWindows:`` fires on a Dock
    click; ``applicationDidBecomeActive:`` also covers Cmd+Tab, which sends no
    reopen. Both return the window only when *we* hid it — a window the user
    minimized is left to AppKit, which deminiaturizes it on its own. Built
    over a plain base in tests; PyObjC infers the selector signatures from
    the ``NSApplicationDelegate`` protocol, as it does for pywebview's own
    ``applicationShouldTerminate_``.
    """

    class OrbitalAppDelegate(base):
        def applicationShouldHandleReopen_hasVisibleWindows_(self, app, has_visible_windows):
            if not has_visible_windows and _close_state["hidden"]:
                restore()
                return False  # handled: do not open an untitled document
            return True

        def applicationDidBecomeActive_(self, notification):
            if _close_state["hidden"]:
                restore()

    return OrbitalAppDelegate


def _install_macos_reopen_handler(restore) -> None:
    """Swap pywebview's ``BrowserView.AppDelegate`` for the reopen-aware
    subclass. Must run before ``webview.start``, which instantiates it."""
    global _REOPEN_DELEGATE_CLASS
    if sys.platform != "darwin":
        return
    try:
        from webview.platforms import cocoa
    except Exception:
        return
    if _REOPEN_DELEGATE_CLASS is None:
        _REOPEN_DELEGATE_CLASS = _reopen_delegate_class(
            cocoa.BrowserView.AppDelegate, restore)
    cocoa.BrowserView.AppDelegate = _REOPEN_DELEGATE_CLASS


def webview_storage_kwargs(data_dir: str) -> dict:
    """``webview.start`` options that keep the window's web storage.

    pywebview defaults to private mode, whose WebView2 profile is a fresh
    temporary folder each launch, so localStorage did not survive a restart:
    the language choice (``orbital.locale``) and the first-journey tour state
    were forgotten every time. Persisted under the app's data dir instead.
    """
    return {"private_mode": False,
            "storage_path": os.path.join(data_dir, "webview")}


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

        def pick_files(self):
            """Composer attach button (desktop app): pick files, get their
            absolute paths. Nothing is uploaded and there is no size cap."""
            return pick_files_in(window)

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
        """Intercept window close — hide instead of closing (spec 115 §6).

        On macOS: Cmd+Q and Dock→Quit go through applicationShouldTerminate_
        which also fires this event — _handle_window_close detects that via
        the call stack and allows the quit; otherwise the window is hidden
        (orderOut) and a Dock click brings it back. On Windows/Linux: hidden
        to the tray; quit happens via the tray menu (os._exit).
        """
        return _handle_window_close(window)

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
    window.events.loaded += lambda: install_native_drop(window)
    window.events.before_show += _reveal_traffic_lights
    # macOS fires `maximized` on entering fullscreen and `restored` on leaving
    # it (webview/platforms/cocoa.py). `restored` also fires on un-minimize,
    # which is harmless here: a minimized window is never fullscreen.
    window.events.maximized += lambda: _set_chrome_mode("fullscreen")
    window.events.restored += lambda: _set_chrome_mode("mac-inline")
    from agent_os.desktop.migration import DATA_DIR
    start_kwargs = webview_storage_kwargs(DATA_DIR)
    localization = _os_localization()
    if localization:
        start_kwargs["localization"] = localization
    if sys.platform == "win32":
        # Declare the only renderer that can run the SPA. (pywebview still
        # falls back to MSHTML internally if WebView2 vanished after the
        # guard above — the guard is what produces the loud failure.)
        start_kwargs["gui"] = "edgechromium"
    # Spec 115 §6: must precede webview.start(), which instantiates the delegate.
    _install_macos_reopen_handler(lambda: _restore_hidden_window(_window))
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


# ---------------------------------------------------------------------------
# Spec 134 A4 — always come back
# ---------------------------------------------------------------------------
#
# A frozen Orbital stays alive (on Windows the close button hides it to the
# tray), keeps the single-instance mutex and the daemon PID file, and every
# relaunch used to either re-show the frozen window or die on an uncaught
# ``DaemonAlreadyRunning``. Now: a frozen instance is detected (hung window,
# or its daemon does not answer), the user is asked "End it and start
# again", and any boot failure is logged at ERROR and shown in a window.

_DAEMON_ANSWER_TIMEOUT_S = 5.0


def _boot_logger():
    import logging

    return logging.getLogger("agent_os.desktop")


def _can_offer_end_and_restart() -> bool:
    """Only the packaged app offers to end another instance. A run from
    source (tests start this module) never kills a process on a click."""
    return bool(getattr(sys, "frozen", False))


def _recorded_daemon() -> tuple[int | None, dict | None]:
    """The PID in the daemon PID file and its identity sidecar."""
    from agent_os.utils import pid_file

    path = pid_file._DEFAULT_PID_PATH
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return None, None
    return pid, pid_file.read_pid_meta(path)


def _daemon_answers(meta: dict | None,
                    timeout: float = _DAEMON_ANSWER_TIMEOUT_S) -> bool:
    """Whether the recorded daemon serves its health route in *timeout* s."""
    port = 8000
    try:
        if meta and meta.get("port"):
            port = int(meta["port"])
    except (TypeError, ValueError):
        pass
    try:
        resp = _local_opener().open(
            f"http://127.0.0.1:{port}/api/v2/settings", timeout=timeout)
        return resp.status == 200
    except Exception:
        return False


def _window_is_hung(title: str = "Orbital") -> bool:
    """Windows: the running shell's window exists and is not answering."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, title)
        return bool(hwnd) and bool(user32.IsHungAppWindow(hwnd))
    except Exception:
        return False


def _existing_instance_is_responsive() -> bool:
    """The running shell can be handed the launch: its window answers and,
    when a daemon is recorded, the daemon answers too."""
    if _window_is_hung():
        return False
    pid, meta = _recorded_daemon()
    if pid is None:
        return True
    return _daemon_answers(meta)


def _unresponsive_prompt() -> tuple[str, str, str, str]:
    """(title, message, end label, cancel label) in the OS language."""
    if _os_localization() is not None:
        return (
            "Orbital 没有响应",
            "Orbital 正在运行，但没有响应。\n\n要结束它并重新启动吗？"
            "它正在运行的命令也会被结束。",
            "结束并重新启动", "取消",
        )
    return (
        "Orbital is not responding",
        "Orbital is running but not responding.\n\nEnd it and start again? "
        "Commands it is running will be stopped too.",
        "End and Start Again", "Cancel",
    )


def _applescript_str(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _confirm_end_unresponsive() -> bool:
    """Native Yes/No: "Orbital is not responding — End it and start again"."""
    title, message, end_label, cancel_label = _unresponsive_prompt()
    try:
        if sys.platform == "win32":
            import ctypes

            MB_YESNO, MB_ICONWARNING, MB_SETFOREGROUND, IDYES = 0x4, 0x30, 0x10000, 6
            answer = ctypes.windll.user32.MessageBoxW(
                0, message, title, MB_YESNO | MB_ICONWARNING | MB_SETFOREGROUND)
            return answer == IDYES
        if sys.platform == "darwin":
            import subprocess

            from agent_os.utils.subprocess_flags import win_no_window_flags

            script = (
                f"display dialog {_applescript_str(message)} "
                f"with title {_applescript_str(title)} "
                f"buttons {{{_applescript_str(cancel_label)}, {_applescript_str(end_label)}}} "
                f"default button 2 cancel button 1 with icon caution"
            )
            result = subprocess.run(
                ["osascript", "-e", script], capture_output=True,
                encoding="utf-8", errors="replace", timeout=600,
                creationflags=win_no_window_flags(),
            )
            return result.returncode == 0 and end_label in (result.stdout or "")
    except Exception:
        _boot_logger().exception("Could not show the not-responding dialog")
    return False


def _looks_like_orbital(proc) -> bool:
    # Only packaged builds offer to end a process, and there the daemon is
    # the Orbital binary itself, never a Python interpreter: a reused PID now
    # held by someone's python script must not be ended.
    try:
        name = (proc.name() or "").lower()
    except Exception:
        return False
    return name.startswith("orbital")


def _end_process_tree(pid: int, meta: dict | None) -> bool:
    """End the unresponsive instance and what it started. True when gone.

    Only the process the PID file describes: with a sidecar its create time
    must match (a reused PID is left alone and counts as gone); without one
    (an older build) it must at least look like Orbital.
    """
    import psutil

    try:
        proc = psutil.Process(pid)
        if meta and meta.get("pid") == pid and meta.get("create_time") is not None:
            if abs(proc.create_time() - float(meta["create_time"])) > 1.0:
                return True
        elif not _looks_like_orbital(proc):
            _boot_logger().error("PID %d does not look like Orbital; not ending it", pid)
            return False
        procs = proc.children(recursive=True) + [proc]
    except psutil.NoSuchProcess:
        return True
    except Exception:
        _boot_logger().exception("Could not inspect PID %d", pid)
        return False
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
        except Exception:
            _boot_logger().warning("Could not end PID %s", getattr(p, "pid", "?"))
    _gone, alive = psutil.wait_procs(procs, timeout=10)
    if proc in alive:
        _boot_logger().error("PID %d did not exit after being ended", pid)
        return False
    _boot_logger().warning("Ended the unresponsive Orbital (PID %d)", pid)
    return True


def _offer_end_and_restart(pid: int, meta: dict | None) -> bool:
    """Ask, then end. True when the old instance is gone and boot may go on."""
    if not _can_offer_end_and_restart():
        return False
    if not _confirm_end_unresponsive():
        _boot_logger().info("User kept the unresponsive Orbital (PID %d)", pid)
        return False
    return _end_process_tree(pid, meta)


def _show_boot_failure(headline: str, message: str | None = None) -> None:
    """The "could not start" window (pywebview), with the stderr tail."""
    import html as _html
    import webview

    detail = _stderr_log_tail()
    message_html = f"<p>{_html.escape(message)}</p>" if message else ""
    detail_html = (
        f"<pre style='white-space:pre-wrap'>{_html.escape(detail)}</pre>"
        if detail
        else ""
    )
    webview.create_window(
        "Orbital \u2014 Error",
        html=f"<h2>{_html.escape(headline)}</h2>{message_html}{detail_html}"
        f"<p>Check logs in {_get_log_path()}</p>",
    )
    webview.start()


def _boot_or_report(preferred: int, _retried: bool = False):
    """Boot the daemon; on failure log at ERROR, show a window, return None.

    ``DaemonAlreadyRunning`` means another instance owns the PID file. If its
    daemon answers, this launch opens a window on it. If not, the user may
    end it and this launch boots in its place.
    """
    from agent_os.utils.pid_file import DaemonAlreadyRunning

    try:
        port, _server, _thread = boot_daemon_with_retry(preferred)
    except DaemonAlreadyRunning as exc:
        _boot_logger().error("Orbital could not start: %s", exc)
        pid, meta = _recorded_daemon()
        # No PID in the error: another instance held the claim lock and had
        # not recorded one yet. Use what the file says now, if anything.
        owner = exc.existing_pid if exc.existing_pid is not None else pid
        if owner is None:
            _show_boot_failure(
                "Orbital is already starting",
                "Another Orbital is starting. Wait a moment, then start "
                "Orbital again.",
            )
            return None
        if meta is not None and meta.get("pid") == owner and _daemon_answers(meta):
            return int(meta.get("port") or preferred)
        if not _retried and _offer_end_and_restart(owner, meta):
            return _boot_or_report(preferred, _retried=True)
        _show_boot_failure(
            "Orbital is already running",
            f"Another Orbital (PID {owner}) is running but not "
            "responding. End it in Task Manager (Windows) or Activity Monitor "
            "(macOS), then start Orbital again.",
        )
        return None
    except Exception as exc:
        _boot_logger().exception("Orbital could not start")
        try:
            import traceback

            traceback.print_exc()  # the frozen build's stderr is the log file
        except Exception:
            pass
        _show_boot_failure("Failed to start daemon", str(exc) or type(exc).__name__)
        return None
    if port is None:
        _boot_logger().error("Orbital could not start: the daemon did not answer")
        _show_boot_failure("Failed to start daemon")
    return port


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
        if _existing_instance_is_responsive():
            if _activate_existing_window():
                sys.exit(0)
        else:
            # Spec 134 W6: the running instance is frozen (hung window, or a
            # daemon that does not answer). Re-showing it is no help; offer
            # to end it and start again.
            pid, meta = _recorded_daemon()
            if pid is None or not _offer_end_and_restart(pid, meta):
                _activate_existing_window()
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
        port = _boot_or_report(PORT)
        if port is None:
            return

    # Daemon is up — provision Chromium (extract bundled archive, or a
    # backed-off background download) without blocking the UI.
    from agent_os.desktop.migration import provision_browsers_background
    provision_browsers_background()

    def shutdown():
        _exit_app(port)

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
            kwargs={"confirm_quit_fn": confirm_quit},
            daemon=True,
        )
        tray_thread.start()

    open_window(port)

    if sys.platform == "darwin":
        # macOS: no system tray, so window close = app close.
        # This is standard macOS behavior for non-document-based apps.
        _exit_app(port)
    else:
        # Windows/Linux: system tray keeps app alive after window close.
        # User quits via tray menu → shutdown() → _exit_app().
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
