# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Windows install step (agent_os/update_windows.py), platform mocked: the
exact Setup command line, the UAC / ready-mutex wait, the child cleanup, and
the contract between the Python side, main.py and installer/agentos-setup.iss.
The real run is docs/verification/in-app-upgrade-windows.md."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent_os import update_windows as uw
from agent_os.update_windows import WindowsInstaller, setup_command
from agent_os.updater import UpdateError

ROOT = Path(__file__).resolve().parents[2]
ISS = (ROOT / "installer" / "agentos-setup.iss").read_text(encoding="utf-8-sig")
MAIN = (ROOT / "agent_os" / "desktop" / "main.py").read_text(encoding="utf-8")

APP = Path(r"C:\Orbital")
EXE = Path(r"C:\Orbital\bin\Orbital.exe")


class FakeProc:
    def __init__(self, exit_after=None, code=0):
        self.polls = 0
        self.exit_after = exit_after
        self.code = code

    def poll(self):
        self.polls += 1
        if self.exit_after is not None and self.polls >= self.exit_after:
            return self.code
        return None


def installer(*, proc=None, ready_after=None, popen_exc=None, ended=None, wait_s=5.0):
    calls = {"popen": [], "probes": 0}
    ended = ended if ended is not None else []

    def popen(cmd, **kw):
        calls["popen"].append((cmd, kw))
        if popen_exc:
            exc = popen_exc.pop(0) if isinstance(popen_exc, list) else popen_exc
            if exc is not None:
                raise exc
        return proc or FakeProc()

    def probe(name):
        assert name == "OrbitalUpdateSetupReady"
        calls["probes"] += 1
        return ready_after is not None and calls["probes"] >= ready_after

    inst = WindowsInstaller((APP, EXE), popen=popen, mutex_probe=probe,
                            end_children=lambda d: ended.append(d) or 1,
                            sleep=lambda s: None, uac_wait_s=wait_s)
    return inst, calls, ended


def winerror(code):
    exc = OSError(f"winerror {code}")
    exc.winerror = code
    return exc


class TestCommand:
    def test_exact_command_line(self, tmp_path):
        cmd = setup_command(Path(r"C:\Users\A B\AppData\Local\Orbital\updates\Orbital-0.24.0.exe"),
                            EXE, Path(r"C:\Users\A B\AppData\Local\Orbital\updates\setup-0.24.0.log"))
        assert cmd == (
            r'"C:\Users\A B\AppData\Local\Orbital\updates\Orbital-0.24.0.exe" '
            "/SILENT /SP- /SUPPRESSMSGBOXES /NORESTART /CLOSEAPPLICATIONS "
            "/FORCECLOSEAPPLICATIONS /NORESTARTAPPLICATIONS "
            r'/LOG="C:\Users\A B\AppData\Local\Orbital\updates\setup-0.24.0.log" '
            r'/ORBITALUPDATE=1 /ORBITALRELAUNCH="C:\Orbital\bin\Orbital.exe"')

    def test_progress_stays_visible(self):
        # /SILENT shows Setup's progress window; /VERYSILENT would hide it.
        assert "/SILENT" in uw.SETUP_FLAGS and "/VERYSILENT" not in uw.SETUP_FLAGS

    def test_a_quote_in_a_path_is_refused(self):
        with pytest.raises(UpdateError):
            setup_command(Path('C:\\a"b\\x.exe'), EXE, Path("C:\\l.log"))

    def test_install_layout(self, tmp_path):
        (tmp_path / "bin").mkdir()
        exe = tmp_path / "bin" / "Orbital.exe"
        exe.write_text("")
        assert uw.install_layout(str(exe)) is None            # no uninstaller: portable copy
        (tmp_path / "unins000.exe").write_text("")
        assert uw.install_layout(str(exe)) == (tmp_path, exe)
        assert uw.install_layout(str(tmp_path / "python.exe")) is None


class TestHandoff:
    async def test_go_once_setup_is_past_uac(self, tmp_path):
        inst, calls, ended = installer(ready_after=3)
        assert await inst.handoff(tmp_path / "Orbital-0.24.0.exe", "0.24.0", tmp_path) == "go"
        cmd, kw = calls["popen"][0]
        assert cmd.startswith(f'"{tmp_path / "Orbital-0.24.0.exe"}" /SILENT')
        flags = kw["creationflags"]
        assert flags & uw.DETACHED_PROCESS and flags & uw.CREATE_NEW_PROCESS_GROUP
        assert flags & uw.CREATE_BREAKAWAY_FROM_JOB
        assert ended == [APP]                                  # install-dir children ended

    async def test_uac_declined_stays(self, tmp_path):
        inst, calls, ended = installer(proc=FakeProc(exit_after=2, code=2))
        assert await inst.handoff(tmp_path / "s.exe", "0.24.0", tmp_path) == "cancelled"
        assert ended == []                                     # nothing touched

    async def test_no_answer_within_the_wait_stays(self, tmp_path):
        inst, calls, ended = installer(wait_s=0.05)
        assert await inst.handoff(tmp_path / "s.exe", "0.24.0", tmp_path) == "cancelled"

    async def test_breakaway_denied_retries_without_it(self, tmp_path):
        inst, calls, _ = installer(ready_after=1, popen_exc=[winerror(5), None])
        assert await inst.handoff(tmp_path / "s.exe", "0.24.0", tmp_path) == "go"
        assert len(calls["popen"]) == 2
        assert not calls["popen"][1][1]["creationflags"] & uw.CREATE_BREAKAWAY_FROM_JOB

    async def test_elevation_error_is_an_install_failure(self, tmp_path):
        inst, _, _ = installer(popen_exc=winerror(740))
        with pytest.raises(UpdateError) as err:
            await inst.handoff(tmp_path / "s.exe", "0.24.0", tmp_path)
        assert err.value.code == "install"

    async def test_prepare_is_the_hash_only(self, tmp_path):
        inst, _, _ = installer()
        f = tmp_path / "Orbital-0.24.0.exe"
        assert await inst.prepare(f, "0.24.0") == f

    def test_not_installed_cannot_update(self):
        # A dev run's interpreter is not {app}\bin\Orbital.exe.
        assert WindowsInstaller().unsupported_reason() == "not-installed"

    def test_under_is_a_real_path_prefix(self, tmp_path):
        (tmp_path / "app" / "bin").mkdir(parents=True)
        (tmp_path / "app2").mkdir()
        assert uw._under(str(tmp_path / "app" / "bin" / "node.exe"), str(tmp_path / "app"))
        assert not uw._under(str(tmp_path / "app2" / "node.exe"), str(tmp_path / "app"))


class TestContract:
    """The names the three sides agree on. Changing one without the others
    breaks the Windows upgrade in a way no test on macOS would notice."""

    def test_app_mutex(self):
        assert f"AppMutex={uw.APP_MUTEX}" in ISS
        assert re.search(rf'SINGLE_INSTANCE_MUTEX_NAME = "{uw.APP_MUTEX}"', MAIN)
        assert f"CheckForMutexes('{uw.APP_MUTEX}')" in ISS

    def test_ready_mutex_and_params(self):
        assert f"CreateMutex('{uw.READY_MUTEX}')" in ISS
        assert f"{{param:{uw.UPDATE_PARAM}|0}}" in ISS
        assert f"{{param:{uw.RELAUNCH_PARAM}|}}" in ISS

    def test_relaunch_runs_as_the_original_user(self):
        assert "ExecAsOriginalUser(Exe" in ISS

    def test_app_id_is_the_implicit_value(self):
        # AppId defaults to AppName; an explicit AppId must equal it exactly,
        # or upgrades register a second Orbital beside the first.
        app_id = re.search(r"^AppId=(.*)$", ISS, re.M).group(1).strip()
        app_name = re.search(r"^AppName=(.*)$", ISS, re.M).group(1).strip()
        assert app_id == app_name == "Orbital"

    def test_the_manual_launch_entry_still_skips_silent_runs(self):
        assert re.search(r'Description: "Launch Orbital"; Flags: nowait postinstall skipifsilent', ISS)
