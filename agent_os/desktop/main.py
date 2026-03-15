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


def resolve_icon_path() -> str:
    if getattr(sys, "frozen", False):
        base = _frozen_base_dir()
        # Use .png on macOS (.ico is Windows-only)
        if sys.platform == "darwin":
            icon = os.path.join(base, "assets", "icon.png")
        else:
            icon = os.path.join(base, "assets", "icon.ico")
    else:
        ext = "icon.png" if sys.platform == "darwin" else "icon.ico"
        icon = os.path.join(os.path.dirname(__file__), "..", "..", "assets", ext)
    return os.path.abspath(icon)


def _set_window_icon():
    """Set the Orbital icon on the pywebview window (Windows only)."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        icon_path = resolve_icon_path()
        if not os.path.exists(icon_path):
            return
        IMAGE_ICON = 1
        LR_LOADFROMFILE = 0x0010
        LR_DEFAULTSIZE = 0x0040
        WM_SETICON = 0x0080
        hicon = user32.LoadImageW(0, icon_path, IMAGE_ICON, 0, 0,
                                  LR_LOADFROMFILE | LR_DEFAULTSIZE)
        if not hicon:
            return
        for _ in range(50):
            hwnd = user32.FindWindowW(None, "Orbital")
            if hwnd:
                user32.SendMessageW(hwnd, WM_SETICON, 0, hicon)
                user32.SendMessageW(hwnd, WM_SETICON, 1, hicon)
                break
            time.sleep(0.1)
    except Exception:
        pass


_window = None


def open_window(port: int):
    global _window
    import webview

    # If window already exists (hidden), just show it
    if _window is not None:
        try:
            _window.show()
        except Exception:
            pass
        return

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

    _window = webview.create_window(
        title="Orbital",
        url=f"http://127.0.0.1:{port}",
        width=1200,
        height=800,
        min_size=(800, 600),
        text_select=True,
        js_api=Api(),
    )
    window = _window
    window.events.closing += _on_closing
    webview.start(icon=resolve_icon_path(), func=_activate_macos)


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


def main():
    # Handle CLI flags before any daemon/GUI setup
    if "--setup-sandbox" in sys.argv:
        run_sandbox_setup()
        return

    if "--teardown-sandbox" in sys.argv:
        run_sandbox_teardown()
        return

    from agent_os.desktop.migration import run_migrations

    PORT = 8000
    run_migrations()
    _disable_app_nap()

    if is_already_running(PORT):
        open_window(PORT)
        return

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


if __name__ == "__main__":
    main()
