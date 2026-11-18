# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""macOS install step (agent_os/update_macos.py): the verification gate, the
install-location checks, prepare's cleanup, and the REAL swap helper script
run against fake bundles in a temp dir (success, rollback, old app that never
exits, open that fails)."""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from agent_os import update_macos
from agent_os.update_macos import HELPER_SCRIPT, MacInstaller, verify_bundle
from agent_os.updater import UpdateError

# The macOS step is POSIX through and through (statvfs, uid, symlinks, /bin/sh).
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="macOS install step")

TEAM = "UG8R3KRXSU"
REFERENCE = {"identifier": "com.orbital.desktop", "team_id": TEAM, "authority": []}


def make_bundle(path: Path, *, version="0.24.0", bundle_id="com.orbital.desktop",
                marker="new") -> Path:
    (path / "Contents" / "MacOS").mkdir(parents=True)
    with open(path / "Contents" / "Info.plist", "wb") as fh:
        plistlib.dump({"CFBundleIdentifier": bundle_id,
                       "CFBundleShortVersionString": version}, fh)
    (path / "Contents" / "marker").write_text(marker)
    return path


class FakeRun:
    """codesign / spctl / hdiutil / ditto stand-in. ``results`` maps a tool
    name to (returncode, stdout, stderr)."""

    def __init__(self, **results):
        self.results = {
            "verify": (0, "", ""),
            "display": (0, "", f"Identifier=com.orbital.desktop\nTeamIdentifier={TEAM}\n"
                               "Authority=Developer ID Application: X (UG8R3KRXSU)\n"),
            "spctl": (0, "", "accepted\nsource=Notarized Developer ID\n"),
            "attach": (0, "", ""), "detach": (0, "", ""), "ditto": (0, "", ""),
            **results,
        }
        self.calls = []
        self.dmg_app: Path | None = None

    def __call__(self, cmd, timeout=0):
        self.calls.append(cmd)
        tool = Path(cmd[0]).name
        if tool == "codesign":
            key = "verify" if "--verify" in cmd else "display"
        elif tool == "hdiutil":
            key = cmd[1]
            if key == "attach" and self.results["attach"][0] == 0 and self.dmg_app:
                mnt = Path(cmd[cmd.index("-mountpoint") + 1])
                shutil.copytree(self.dmg_app, mnt / "Orbital.app", symlinks=True)
            if key == "detach" and self.results["detach"][0] == 0:
                shutil.rmtree(Path(cmd[-1]) / "Orbital.app", ignore_errors=True)
        elif tool == "ditto":
            key = "ditto"
            if self.results["ditto"][0] == 0:
                shutil.copytree(cmd[-2], cmd[-1], symlinks=True)
        else:
            key = tool
        rc, out, err = self.results[key]
        return subprocess.CompletedProcess(cmd, rc, out, err)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

class TestVerifyBundle:
    def check(self, bundle, run=None, **kw):
        verify_bundle(bundle, expected_version=kw.get("expected", "0.24.0"),
                      current_version=kw.get("current", "0.23.1"),
                      reference=kw.get("reference", REFERENCE), run=run or FakeRun())

    def test_a_good_bundle_passes_every_check(self, tmp_path):
        run = FakeRun()
        self.check(make_bundle(tmp_path / "Orbital.app"), run)
        tools = [" ".join(c[:3]) for c in run.calls]
        assert any("--verify --deep" in t for t in tools)
        assert any(c[0].endswith("spctl") and c[1:4] == ["-a", "-t", "exec"] for c in run.calls)

    def test_wrong_team_id_is_refused(self, tmp_path):
        run = FakeRun(display=(0, "", "Identifier=com.orbital.desktop\nTeamIdentifier=EVILTEAM00\n"))
        with pytest.raises(UpdateError) as err:
            self.check(make_bundle(tmp_path / "Orbital.app"), run)
        assert err.value.code == "verify" and "team id" in err.value.detail

    def test_unsigned_bundle_is_refused(self, tmp_path):
        run = FakeRun(verify=(1, "", "code object is not signed at all"))
        with pytest.raises(UpdateError) as err:
            self.check(make_bundle(tmp_path / "Orbital.app"), run)
        assert "codesign" in err.value.detail

    def test_ad_hoc_signature_is_refused(self, tmp_path):
        run = FakeRun(display=(0, "", "Identifier=com.orbital.desktop\nTeamIdentifier=not set\n"))
        with pytest.raises(UpdateError):
            self.check(make_bundle(tmp_path / "Orbital.app"), run)

    def test_a_running_app_without_a_team_refuses_everything(self, tmp_path):
        with pytest.raises(UpdateError):
            self.check(make_bundle(tmp_path / "Orbital.app"),
                       reference={"identifier": "com.orbital.desktop", "team_id": ""})

    def test_gatekeeper_rejection_is_refused(self, tmp_path):
        run = FakeRun(spctl=(3, "", "rejected\nsource=Unnotarized Developer ID"))
        with pytest.raises(UpdateError) as err:
            self.check(make_bundle(tmp_path / "Orbital.app"), run)
        assert "Gatekeeper" in err.value.detail

    def test_other_bundle_id_is_refused(self, tmp_path):
        with pytest.raises(UpdateError):
            self.check(make_bundle(tmp_path / "Orbital.app", bundle_id="com.evil.app"))

    @pytest.mark.parametrize("version,expected,current", [
        ("0.24.1", "0.24.0", "0.23.1"),   # not what was offered
        ("0.23.1", "0.23.1", "0.23.1"),   # not newer
        ("0.22.0", "0.22.0", "0.23.1"),   # a downgrade
    ])
    def test_version_must_match_and_be_newer(self, tmp_path, version, expected, current):
        with pytest.raises(UpdateError):
            self.check(make_bundle(tmp_path / "Orbital.app", version=version),
                       expected=expected, current=current)

    def test_a_symlink_out_of_the_bundle_is_refused(self, tmp_path):
        bundle = make_bundle(tmp_path / "Orbital.app")
        os.symlink("/etc", bundle / "Contents" / "escape")
        with pytest.raises(UpdateError) as err:
            self.check(bundle)
        assert "symlink" in err.value.detail

    def test_symlinks_inside_the_bundle_are_fine(self, tmp_path):
        bundle = make_bundle(tmp_path / "Orbital.app")
        (bundle / "Contents" / "Frameworks" / "X.framework" / "Versions" / "A").mkdir(parents=True)
        os.symlink("A", bundle / "Contents" / "Frameworks" / "X.framework" / "Versions" / "Current")
        self.check(bundle)


class TestLocation:
    def test_running_bundle(self):
        assert update_macos.running_bundle("/Applications/Orbital.app/Contents/MacOS/Orbital") \
            == Path("/Applications/Orbital.app")
        assert update_macos.running_bundle("/usr/local/bin/python3") is None

    def test_translocated_is_refused(self):
        p = Path("/private/var/folders/x/AppTranslocation/ABC/d/Orbital.app")
        assert update_macos.location_problem(p) == "translocated"

    def test_read_only_volume_is_refused(self, tmp_path, monkeypatch):
        bundle = make_bundle(tmp_path / "Orbital.app")
        monkeypatch.setattr(update_macos.os, "statvfs",
                            lambda p: types.SimpleNamespace(f_flag=os.ST_RDONLY))
        assert update_macos.location_problem(bundle) == "read-only"

    @pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0,
                        reason="root can write anywhere")
    def test_unwritable_folder_is_refused(self, tmp_path):
        parent = tmp_path / "Apps"
        bundle = make_bundle(parent / "Orbital.app")
        parent.chmod(0o555)
        try:
            assert update_macos.location_problem(bundle) == "folder-not-writable"
        finally:
            parent.chmod(0o755)

    def test_a_writable_own_folder_is_fine(self, tmp_path):
        assert update_macos.location_problem(make_bundle(tmp_path / "Orbital.app")) is None

    def test_unsigned_running_app_cannot_update(self, tmp_path):
        bundle = make_bundle(tmp_path / "Orbital.app")
        run = FakeRun(display=(0, "", "Identifier=com.orbital.desktop\nTeamIdentifier=not set\n"))
        assert MacInstaller(bundle, run=run).unsupported_reason() == "unsigned"
        assert MacInstaller(bundle, run=FakeRun()).unsupported_reason() is None
        assert MacInstaller(None, run=FakeRun()).unsupported_reason() == "not-a-bundle"


# ---------------------------------------------------------------------------
# prepare / handoff / startup
# ---------------------------------------------------------------------------

class TestPrepare:
    def build(self, tmp_path, **run_results):
        apps = tmp_path / "Applications"
        installed = make_bundle(apps / "Orbital.app", version="0.23.1", marker="old")
        run = FakeRun(**run_results)
        run.dmg_app = make_bundle(tmp_path / "dmgsrc" / "Orbital.app")
        cache = tmp_path / "cache"
        cache.mkdir()
        dmg = cache / "Orbital-0.24.0.dmg"
        dmg.write_bytes(b"dmg")
        inst = MacInstaller(installed, run=run, current_version=lambda: "0.23.1", retry_delay=0)
        return inst, run, dmg, apps

    async def test_stages_beside_the_installed_app_and_verifies(self, tmp_path):
        inst, run, dmg, apps = self.build(tmp_path)
        staged = await inst.prepare(dmg, "0.24.0")
        assert staged == apps / ".Orbital-update-0.24.0" / "Orbital.app"
        assert (staged / "Contents" / "marker").read_text() == "new"
        attach = next(c for c in run.calls if c[1] == "attach")
        assert {"-nobrowse", "-readonly", "-noautoopen"} <= set(attach)
        ditto = next(c for c in run.calls if c[0].endswith("ditto"))
        assert "--noqtn" in ditto
        assert any(c[1] == "detach" for c in run.calls if c[0].endswith("hdiutil"))
        assert not list(dmg.parent.glob("mnt-*"))       # mount point removed
        assert (apps / "Orbital.app" / "Contents" / "marker").read_text() == "old"

    async def test_a_failed_gate_removes_the_staging_folder(self, tmp_path):
        inst, run, dmg, apps = self.build(tmp_path, spctl=(3, "", "rejected"))
        with pytest.raises(UpdateError):
            await inst.prepare(dmg, "0.24.0")
        assert not (apps / ".Orbital-update-0.24.0").exists()
        assert (apps / "Orbital.app" / "Contents" / "marker").read_text() == "old"

    async def test_a_disk_image_that_will_not_mount(self, tmp_path):
        inst, run, dmg, apps = self.build(tmp_path, attach=(1, "", "corrupt image"))
        with pytest.raises(UpdateError) as err:
            await inst.prepare(dmg, "0.24.0")
        assert err.value.code == "verify"
        assert sum(1 for c in run.calls if c[1:2] == ["attach"]) == 3   # retried
        assert not (apps / ".Orbital-update-0.24.0").exists()

    async def test_handoff_starts_a_helper_that_outlives_us(self, tmp_path):
        inst, run, dmg, apps = self.build(tmp_path)
        staged = await inst.prepare(dmg, "0.24.0")
        spawned = {}

        def popen(cmd, **kw):
            spawned.update(cmd=cmd, **kw)
        inst._popen = popen
        assert await inst.handoff(staged, "0.24.0", dmg.parent) == "go"
        assert spawned["start_new_session"] is True        # not our process group
        assert spawned["cmd"][:2] == ["/bin/sh", str(dmg.parent / "apply-update.sh")]
        assert spawned["cmd"][2:] == [str(os.getpid()), str(apps / "Orbital.app"), str(staged),
                                      str(staged.parent / "previous.app"), str(dmg.parent),
                                      "0.24.0"]
        assert "ORBITAL_UPDATE_OPEN" not in spawned["env"]
        assert (dmg.parent / "apply-update.sh").read_text() == HELPER_SCRIPT

    def test_startup_on_the_new_version_signals_the_helper(self, tmp_path):
        inst = MacInstaller(None)
        inst.on_startup(tmp_path, {"target": "0.24.0"}, "0.24.0")
        assert (tmp_path / "booted-0.24.0").read_text() == str(os.getpid())
        inst.on_startup(tmp_path, {"target": "0.24.0"}, "0.23.1")
        assert not (tmp_path / "booted-0.23.1").exists()

    def test_cleanup_sweeps_only_our_leftovers(self, tmp_path):
        apps = tmp_path / "Applications"
        installed = make_bundle(apps / "Orbital.app", marker="old")
        (apps / ".Orbital-update-0.24.0" / "Orbital.app").mkdir(parents=True)
        (apps / "Other.app").mkdir()
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "booted-0.24.0").write_text("1")
        (cache / "apply-update.sh").write_text("#!/bin/sh")
        (cache / "apply-update.log").write_text("kept for Send feedback")
        MacInstaller(installed).cleanup(cache)
        assert sorted(p.name for p in apps.iterdir()) == ["Orbital.app", "Other.app"]
        assert [p.name for p in cache.iterdir()] == ["apply-update.log"]


# ---------------------------------------------------------------------------
# The real helper script
# ---------------------------------------------------------------------------

OPEN_STUB = r'''#!/bin/sh
# Test stand-in for /usr/bin/open: records the call, then acts per STUB_MODE.
echo "$1 $(cat "$1/Contents/marker")" >> "$STUB_LOG"
marker="$(cat "$1/Contents/marker")"
case "$STUB_MODE" in
  boot) [ "$marker" = new ] && echo up > "$STUB_BOOTED" ;;
  hang) [ "$marker" = new ] && [ -x "$1/Contents/MacOS/Stub" ] && ("$1/Contents/MacOS/Stub" &) ;;
  fail) [ "$marker" = new ] && exit 1 ;;
esac
exit 0
'''


class TestHelperScript:
    def layout(self, tmp_path, mode, *, compiled_stub=False):
        apps = tmp_path / "Applications"
        app = make_bundle(apps / "Orbital.app", version="0.23.1", marker="old")
        stage = apps / ".Orbital-update-0.24.0"
        new = make_bundle(stage / "Orbital.app", marker="new")
        if compiled_stub:
            if shutil.which("cc") is None:
                pytest.skip("no C compiler for the fake app process")
            src = tmp_path / "s.c"
            src.write_text("#include <unistd.h>\nint main(){sleep(600);return 0;}\n")
            subprocess.run(["cc", "-o", str(new / "Contents" / "MacOS" / "Stub"), str(src)],
                           check=True)
        cache = tmp_path / "cache"
        cache.mkdir()
        helper = cache / "apply-update.sh"
        helper.write_text(HELPER_SCRIPT)
        stub = tmp_path / "open-stub.sh"
        stub.write_text(OPEN_STUB)
        stub.chmod(0o755)
        env = {**os.environ, "ORBITAL_UPDATE_OPEN": str(stub), "STUB_MODE": mode,
               "STUB_LOG": str(tmp_path / "open.log"),
               "STUB_BOOTED": str(cache / "booted-0.24.0"),
               "ORBITAL_UPDATE_WAIT_EXIT": "20", "ORBITAL_UPDATE_WAIT_BOOT": "6"}
        return types.SimpleNamespace(apps=apps, app=app, stage=stage, new=new, cache=cache,
                                     helper=helper, env=env, log=tmp_path / "open.log")

    def run_helper(self, lay, old_pid):
        return subprocess.run(
            ["/bin/sh", str(lay.helper), str(old_pid), str(lay.app), str(lay.new),
             str(lay.stage / "previous.app"), str(lay.cache), "0.24.0"],
            env=lay.env, capture_output=True, text=True, timeout=60)

    def exited_pid(self):
        p = subprocess.Popen(["/usr/bin/true"])
        p.wait()
        return p.pid

    def opened(self, lay):
        return lay.log.read_text().split("\n")[:-1] if lay.log.exists() else []

    def test_swap_waits_for_the_old_app_then_keeps_the_new_one(self, tmp_path):
        lay = self.layout(tmp_path, "boot")
        old = subprocess.Popen(["/bin/sleep", "1.5"])
        threading.Thread(target=old.wait, daemon=True).start()   # reap: no zombie
        started = time.monotonic()
        proc = self.run_helper(lay, old.pid)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert time.monotonic() - started >= 1.0                  # it waited
        assert (lay.app / "Contents" / "marker").read_text() == "new"
        assert not lay.stage.exists()                             # old bundle deleted
        assert self.opened(lay) == [f"{lay.app} new"]
        assert not (lay.cache / "result.json").exists()

    def test_a_new_app_that_never_comes_up_is_rolled_back(self, tmp_path):
        lay = self.layout(tmp_path, "hang", compiled_stub=True)
        proc = self.run_helper(lay, self.exited_pid())
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert (lay.app / "Contents" / "marker").read_text() == "old"
        assert not lay.stage.exists()
        assert self.opened(lay) == [f"{lay.app} new", f"{lay.app} old"]
        assert '"result": "rolled_back"' in (lay.cache / "result.json").read_text()
        # The hung new process was ended before the swap back.
        ps = subprocess.run(["ps", "-axo", "comm="], capture_output=True, text=True).stdout
        assert str(lay.apps) not in ps

    def test_open_failing_rolls_back(self, tmp_path):
        lay = self.layout(tmp_path, "fail")
        proc = self.run_helper(lay, self.exited_pid())
        assert proc.returncode == 1
        assert (lay.app / "Contents" / "marker").read_text() == "old"
        assert '"rolled_back"' in (lay.cache / "result.json").read_text()

    def test_an_old_app_that_never_exits_is_left_alone(self, tmp_path):
        lay = self.layout(tmp_path, "boot")
        lay.env["ORBITAL_UPDATE_WAIT_EXIT"] = "2"
        proc = self.run_helper(lay, os.getpid())   # this test process: never exits
        assert proc.returncode == 1
        assert (lay.app / "Contents" / "marker").read_text() == "old"
        assert not lay.stage.exists()
        assert self.opened(lay) == []
        assert '"aborted"' in (lay.cache / "result.json").read_text()

    def test_a_missing_staged_app_reopens_the_old_one(self, tmp_path):
        lay = self.layout(tmp_path, "boot")
        shutil.rmtree(lay.new)
        proc = self.run_helper(lay, self.exited_pid())
        assert proc.returncode == 1
        assert (lay.app / "Contents" / "marker").read_text() == "old"
        assert self.opened(lay) == [f"{lay.app} old"]
