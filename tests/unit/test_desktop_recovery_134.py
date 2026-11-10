# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 134 A4 — Orbital always comes back.

- A boot failure (``DaemonAlreadyRunning`` above all) is caught in the
  desktop entry point, logged at ERROR and shown in a window, instead of
  killing the new process with an uncaught traceback (W6).
- A frozen instance can be ended from the relaunch: "Orbital is not
  responding — End it and start again".
- ``orbital-stderr.log`` is appended (size-capped), so a failed relaunch no
  longer overwrites the frozen run's evidence.
- The PID file is released before every ``os._exit`` (W7).
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import psutil
import pytest

from agent_os.desktop import main
from agent_os.utils.pid_file import DaemonAlreadyRunning


@pytest.fixture
def shown():
    calls = []
    with patch.object(main, "_show_boot_failure",
                      side_effect=lambda headline, message=None: calls.append((headline, message))):
        yield calls


# ---------------------------------------------------------------------------
# boot failures are caught, logged and shown
# ---------------------------------------------------------------------------


def test_daemon_already_running_is_logged_and_shown(shown, caplog):
    with patch.object(main, "boot_daemon_with_retry", side_effect=DaemonAlreadyRunning(4321)), \
         patch.object(main, "_recorded_daemon", return_value=(4321, None)), \
         patch.object(main, "_can_offer_end_and_restart", return_value=False), \
         caplog.at_level(logging.ERROR, logger="agent_os.desktop"):
        assert main._boot_or_report(8000) is None
    assert any(r.levelno == logging.ERROR and "4321" in r.getMessage() for r in caplog.records)
    assert shown and shown[0][0] == "Orbital is already running"
    assert "4321" in shown[0][1]


def test_any_other_boot_failure_is_logged_and_shown(shown, caplog):
    with patch.object(main, "boot_daemon_with_retry", side_effect=RuntimeError("projects.json is corrupt")), \
         caplog.at_level(logging.ERROR, logger="agent_os.desktop"):
        assert main._boot_or_report(8000) is None
    assert any(r.levelno == logging.ERROR for r in caplog.records)
    assert shown == [("Failed to start daemon", "projects.json is corrupt")]


def test_a_daemon_that_answers_gets_a_window_instead(shown):
    meta = {"pid": 4321, "create_time": 1.0, "port": "8123"}
    with patch.object(main, "boot_daemon_with_retry", side_effect=DaemonAlreadyRunning(4321)), \
         patch.object(main, "_recorded_daemon", return_value=(4321, meta)), \
         patch.object(main, "_daemon_answers", return_value=True):
        assert main._boot_or_report(8000) == 8123
    assert shown == []


def test_end_and_start_again_boots_in_its_place(shown):
    boots = [DaemonAlreadyRunning(4321), (8000, object(), object())]

    def fake_boot(port):
        result = boots.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    with patch.object(main, "boot_daemon_with_retry", side_effect=fake_boot), \
         patch.object(main, "_recorded_daemon", return_value=(4321, None)), \
         patch.object(main, "_daemon_answers", return_value=False), \
         patch.object(main, "_can_offer_end_and_restart", return_value=True), \
         patch.object(main, "_confirm_end_unresponsive", return_value=True), \
         patch.object(main, "_end_process_tree", return_value=True) as ended:
        assert main._boot_or_report(8000) == 8000
    ended.assert_called_once_with(4321, None)
    assert shown == []


def test_declining_keeps_the_old_instance(shown):
    with patch.object(main, "boot_daemon_with_retry", side_effect=DaemonAlreadyRunning(4321)), \
         patch.object(main, "_recorded_daemon", return_value=(4321, None)), \
         patch.object(main, "_daemon_answers", return_value=False), \
         patch.object(main, "_can_offer_end_and_restart", return_value=True), \
         patch.object(main, "_confirm_end_unresponsive", return_value=False), \
         patch.object(main, "_end_process_tree") as ended:
        assert main._boot_or_report(8000) is None
    ended.assert_not_called()
    assert shown and shown[0][0] == "Orbital is already running"


def test_no_readable_pid_never_offers_to_end_anything(shown):
    # The claim lock timed out and the PID file held no PID: another
    # instance is mid-start. Nothing to end, and no nonsense PID shown.
    with patch.object(main, "boot_daemon_with_retry", side_effect=DaemonAlreadyRunning(None)), \
         patch.object(main, "_recorded_daemon", return_value=(None, None)), \
         patch.object(main, "_can_offer_end_and_restart", return_value=True), \
         patch.object(main, "_confirm_end_unresponsive", return_value=True) as confirm, \
         patch.object(main, "_end_process_tree") as ended:
        assert main._boot_or_report(8000) is None
    confirm.assert_not_called()
    ended.assert_not_called()
    assert len(shown) == 1
    headline, message = shown[0]
    for nonsense in ("None", "-1", "PID"):
        assert nonsense not in headline + (message or "")


def test_no_pid_in_the_error_uses_the_pid_file_reread(shown):
    # By the time the window code runs, the other claim may have finished.
    meta = {"pid": 4321, "create_time": 1.0, "port": "8123"}
    with patch.object(main, "boot_daemon_with_retry", side_effect=DaemonAlreadyRunning(None)), \
         patch.object(main, "_recorded_daemon", return_value=(4321, meta)), \
         patch.object(main, "_daemon_answers", return_value=True):
        assert main._boot_or_report(8000) == 8123
    assert shown == []


def test_a_source_run_never_offers_to_kill():
    # Tests launch this module; a click must never end the user's app there.
    with patch.object(main, "_confirm_end_unresponsive") as confirm:
        assert main._offer_end_and_restart(4321, None) is False
    confirm.assert_not_called()


def test_the_prompt_is_bilingual():
    with patch.object(main, "_os_localization", return_value=None):
        title, message, end, _ = main._unresponsive_prompt()
    assert title == "Orbital is not responding" and "End it and start again" in message
    with patch.object(main, "_os_localization", return_value={"x": "y"}):
        title, message, end, _ = main._unresponsive_prompt()
    assert "没有响应" in title and end == "结束并重新启动"


def test_applescript_strings_are_escaped():
    assert main._applescript_str('a "b" \\c') == '"a \\"b\\" \\\\c"'


# ---------------------------------------------------------------------------
# ending the frozen instance: only the process the PID file describes
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX sleep")
def test_end_refuses_a_process_that_is_not_orbital_without_a_sidecar():
    proc = subprocess.Popen(["sleep", "30"])
    try:
        assert main._end_process_tree(proc.pid, None) is False
        assert proc.poll() is None
    finally:
        proc.kill()
        proc.wait()


def test_end_never_ends_a_python_process_without_a_sidecar():
    # Packaged builds are the only ones that offer to end a process, and their
    # daemon is the Orbital binary: a reused PID held by a python script stays.
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert main._end_process_tree(proc.pid, None) is False
        assert proc.poll() is None
    finally:
        proc.kill()
        proc.wait()


def test_end_kills_the_recorded_process_and_its_children():
    proc = subprocess.Popen([sys.executable, "-c",
                             "import subprocess, sys, time;"
                             "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']);"
                             "time.sleep(30)"])
    try:
        deadline = time.monotonic() + 5
        while not psutil.Process(proc.pid).children() and time.monotonic() < deadline:
            time.sleep(0.05)
        child = psutil.Process(proc.pid).children()[0]
        meta = {"pid": proc.pid, "create_time": psutil.Process(proc.pid).create_time()}
        assert main._end_process_tree(proc.pid, meta) is True
        proc.wait(timeout=5)
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    finally:
        if proc.poll() is None:
            proc.kill()


def test_end_leaves_a_reused_pid_alone_and_counts_it_as_gone():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        meta = {"pid": proc.pid, "create_time": psutil.Process(proc.pid).create_time() - 3600}
        assert main._end_process_tree(proc.pid, meta) is True
        assert proc.poll() is None
    finally:
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------
# stderr log: appended, capped, the dialog reads this launch only
# ---------------------------------------------------------------------------


def test_stderr_log_is_appended_across_launches(tmp_path):
    first = main._open_stderr_log(str(tmp_path))
    first.write("ERROR: the frozen run's evidence\n")
    first.close()
    second = main._open_stderr_log(str(tmp_path))
    second.close()
    text = (tmp_path / "orbital-stderr.log").read_text(encoding="utf-8")
    assert "the frozen run's evidence" in text
    assert text.count(main._LAUNCH_MARKER) == 2


def test_stderr_log_rotates_past_the_cap(tmp_path):
    log = tmp_path / "orbital-stderr.log"
    log.write_bytes(b"x" * (main.STDERR_LOG_MAX_BYTES + 1))
    main._open_stderr_log(str(tmp_path)).close()
    assert (tmp_path / "orbital-stderr.log.1").stat().st_size > main.STDERR_LOG_MAX_BYTES
    assert log.stat().st_size < 200


def test_failure_dialog_tail_reads_only_this_launch(tmp_path):
    log = tmp_path / "orbital-stderr.log"
    log.write_text(
        f"{main._LAUNCH_MARKER} 2026-10-25 pid 1 ===\n"
        "ERROR: old launch problem\n"
        f"{main._LAUNCH_MARKER} 2026-10-26 pid 2 ===\n"
        "ERROR: this launch problem\n",
        encoding="utf-8",
    )
    with patch.object(main, "_get_log_path", return_value=str(tmp_path)):
        tail = main._stderr_log_tail()
    assert "this launch problem" in tail and "old launch problem" not in tail


# ---------------------------------------------------------------------------
# W7: the PID file is released before every os._exit
# ---------------------------------------------------------------------------


def test_every_os_exit_in_main_releases_the_pid_file_first():
    # Spec 134 Part B: main() leaves only through _exit_app (which also ends
    # the background jobs); that one os._exit releases the PID file first.
    src = Path(main.__file__).read_text(encoding="utf-8")
    main_src = src[src.index("def main():"):]
    assert "os._exit(0)" not in main_src
    assert main_src.count("_exit_app(port)") == 2   # tray shutdown + macOS window close
    lines = src[src.index("def _exit_app("):src.index("def _handle_window_close(")].splitlines()
    exits = [n for n, line in enumerate(lines) if line.strip() == "os._exit(0)"]
    assert len(exits) == 1
    assert "release_pid_file()" in lines[exits[0] - 1], lines[exits[0] - 2:exits[0] + 1]
