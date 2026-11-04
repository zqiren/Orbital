# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The background-job launcher Orbital ships into a workspace (spec 134 Part B).

Codex and Pi run commands in their own shells, not through Orbital's ``shell``
tool, so they cannot pass ``background: true``. Instead they run
``orbital/bin/bg "<command>"`` (Windows: ``orbital\\bin\\bg.cmd "<command>"``).
The launcher:

* makes ``orbital/jobs/<job_id>/`` — the folder ``JobRegistry`` watches;
* starts the command detached in its own session (POSIX ``setsid``) or, on
  Windows, through WMI so it is outside the calling CLI's Job Object; Codex
  kills its exec process group when a turn ends (verified live 2026-10-27:
  a ``setsid`` launcher job survives the turn end and the app-server exit);
* appends the command's output to ``output.log`` and writes ``exit`` with
  the exit code when the command ends, since Orbital is not its parent;
* writes ``job.json`` last (the registry ignores a folder until it exists),
  attributed with ``ORBITAL_PROJECT_ID`` / ``ORBITAL_SESSION_ID`` /
  ``ORBITAL_AGENT``, which Orbital sets in the worker's environment;
* prints the same start text as ``shell(background=true)``.

The texts live here rather than as data files so the packaged app needs no
PyInstaller ``datas`` entry for them. ``ensure_launcher`` rewrites a file only
when its content differs.
"""

from __future__ import annotations

import logging
import os
import stat

from agent_os.agent.project_paths import ProjectPaths

logger = logging.getLogger(__name__)

# POSIX sh. The detached runner traps TERM/INT/HUP with a no-op handler (not
# SIG_IGN, which the command would inherit) so a group kill
# (``kill -TERM -- -<pid>``) ends the command while the runner survives long
# enough to record its exit code.
BG_SH = r'''#!/bin/sh
# Orbital background-job launcher. Usage: orbital/bin/bg "<command>"
# Starts the command detached, writes its output to orbital/jobs/<id>/output.log
# and tells Orbital, which messages you when it ends. Written by Orbital.
set -u
if [ $# -eq 0 ]; then
  echo 'usage: orbital/bin/bg "<command>"' >&2
  exit 2
fi
clean() { printf '%s' "$1" | LC_ALL=C tr -cd 'A-Za-z0-9_.:\200-\377-'; }
here=$(cd "$(dirname "$0")" && pwd)
jobs_dir=$(dirname "$here")/jobs
id=job_$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')
dir=$jobs_dir/$id
mkdir -p "$dir" || exit 1
printf '%s\n' "$*" > "$dir/cmd"
: > "$dir/output.log"
if [ $# -eq 1 ]; then set -- /bin/sh -c "$1"; fi
runner='d=$1; shift
trap ":" TERM INT HUP
"$@" </dev/null >>"$d/output.log" 2>&1 &
c=$!
while :; do wait "$c"; rc=$?; kill -0 "$c" 2>/dev/null || break; done
printf "%s %s\n" "$rc" "$(date +%s)" > "$d/exit.tmp" && mv "$d/exit.tmp" "$d/exit"'
if command -v setsid >/dev/null 2>&1; then
  setsid /bin/sh -c "$runner" orbital-bg "$dir" "$@" </dev/null >/dev/null 2>&1 &
else
  /usr/bin/perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV or exit 127' \
    /bin/sh -c "$runner" orbital-bg "$dir" "$@" </dev/null >/dev/null 2>&1 &
fi
pid=$!
sid=$(clean "${ORBITAL_SESSION_ID:-}")
printf '{"id": "%s", "group_id": "%s", "project_id": "%s", "session_id": "%s", "agent": "%s", "pid": %s, "started_at": %s, "source": "launcher", "state": "running"}\n' \
  "$id" "$(clean "${ORBITAL_JOB_GROUP:-}")" "$(clean "${ORBITAL_PROJECT_ID:-}")" "$sid" \
  "$(clean "${ORBITAL_AGENT:-}")" "$pid" "$(date +%s)" > "$dir/job.json.tmp" \
  && mv "$dir/job.json.tmp" "$dir/job.json"
echo "Started background job $id (pid $pid)."
echo "Output: orbital/jobs/$id/output.log"
if [ -n "$sid" ]; then
  echo "Orbital will send you a message when it finishes, and check in every 15 minutes while it runs. End your turn now; do not wait, sleep or poll for it."
else
  echo "ORBITAL_SESSION_ID is not set, so Orbital is not tracking this job and no message will come when it ends."
fi
echo "To check early: read the output file. To stop it: kill -TERM -- -$pid"
'''

BG_CMD = '@echo off\r\n' \
    'powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass ' \
    '-File "%~dp0bg.ps1" %*\r\n'

# Windows. Win32_Process.Create (WMI) starts the runner outside the calling
# process's Job Object, so a CLI that kills its job at turn end does not take
# the job with it. The WMI-created process gets the account's default
# environment, not the calling shell's; bg.ps1 saves the caller's PATH in the
# job folder and the runner restores it, since without it a safe-mode job
# (worker account) could not find the per-user python or node the sandboxed
# shell had on its PATH (Windows pass 2026-10-27).
#
# The runner is a batch file (bg-run.cmd), not one compound ``cmd /c "(cmd) &
# echo !errorlevel! …"`` line: a command that ran ``exit`` ended that cmd.exe
# before it wrote the exit code, and its ``/v:on`` ate every ``!`` in the
# command (Windows pass 2026-10-27). The runner starts the command in its own
# cmd.exe and records that child's exit code. It reads the command from the
# job's ``cmd`` file with delayed expansion off, and hands it on through
# ``!c!``, which cmd expands after parsing, so the command is parsed once,
# by the child, exactly as before.
#
# WMI starts conhost.exe, which hosts the runner's cmd.exe. On the Windows
# test machine the process Win32_Process.Create makes cannot itself start
# cmd.exe or powershell.exe ("Access is denied"; python, ping and the like
# start fine), so a job running ``powershell …``, ``cmd /c …`` or npm's own
# nested cmd failed; a process one level below it is not restricted. With
# ShowWindow 0 the conhost window stays hidden. Its command line names the
# job folder, which the registry's identity check looks for.
BG_PS1 = r'''# Orbital background-job launcher. Usage: orbital\bin\bg.cmd "<command>"
# Written by Orbital.
$ErrorActionPreference = 'Stop'
$cmd = ($args -join ' ')
if (-not $cmd) { [Console]::Error.WriteLine('usage: orbital\bin\bg.cmd "<command>"'); exit 2 }
function Clean([string]$v) { if ($v) { return ($v -replace '[^\w.:-]', '') } else { return '' } }
$bin = Split-Path -Parent $MyInvocation.MyCommand.Path
$jobs = Join-Path (Split-Path -Parent $bin) 'jobs'
$id = 'job_' + (-join ((1..8) | ForEach-Object { '{0:x}' -f (Get-Random -Maximum 16) }))
$dir = Join-Path $jobs $id
New-Item -ItemType Directory -Force -Path $dir | Out-Null
[IO.File]::WriteAllText((Join-Path $dir 'cmd'), $cmd + "`n")
[IO.File]::WriteAllText((Join-Path $dir 'path'), $env:PATH + "`n")
[IO.File]::WriteAllText((Join-Path $dir 'output.log'), '')
$line = 'conhost.exe cmd.exe /d /s /c ""' + (Join-Path $bin 'bg-run.cmd') + '" "' + $dir + '""'
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = $line; CurrentDirectory = (Get-Location).Path; ProcessStartupInformation = $startup }
if ($r.ReturnValue -ne 0) { [Console]::Error.WriteLine("could not start the job (Win32_Process.Create returned $($r.ReturnValue))"); exit 1 }
$procId = [int]$r.ProcessId
$sid = Clean $env:ORBITAL_SESSION_ID
$job = [ordered]@{ id = $id; group_id = (Clean $env:ORBITAL_JOB_GROUP); project_id = (Clean $env:ORBITAL_PROJECT_ID);
  session_id = $sid; agent = (Clean $env:ORBITAL_AGENT); pid = $procId;
  started_at = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds(); source = 'launcher'; state = 'running' }
[IO.File]::WriteAllText((Join-Path $dir 'job.json.tmp'), ($job | ConvertTo-Json -Compress))
Move-Item -Force (Join-Path $dir 'job.json.tmp') (Join-Path $dir 'job.json')
Write-Output "Started background job $id (pid $procId)."
Write-Output "Output: orbital/jobs/$id/output.log"
if ($sid) {
  Write-Output 'Orbital will send you a message when it finishes, and check in every 15 minutes while it runs. End your turn now; do not wait, sleep or poll for it.'
} else {
  Write-Output 'ORBITAL_SESSION_ID is not set, so Orbital is not tracking this job and no message will come when it ends.'
}
Write-Output "To check early: read the output file. To stop it: taskkill /T /F /PID $procId"
'''

# ``delims^=^ eol^=`` turns off both FOR /F's field split and its ``;``
# comment character, so the first line of ``cmd`` arrives whole.
BG_RUN_CMD = '\r\n'.join([
    '@echo off',
    'rem Orbital background-job runner, started by bg.ps1. Usage: bg-run.cmd <job folder>',
    'rem Runs the job\'s command in its own cmd.exe, appends its output to output.log',
    'rem and writes its exit code to exit. Written by Orbital.',
    'setlocal DisableDelayedExpansion',
    'set "d=%~1"',
    'set "c="',
    'for /f usebackq^ delims^=^ eol^= %%L in ("%d%\\cmd") do if not defined c set "c=%%L"',
    'set "p="',
    'if exist "%d%\\path" for /f usebackq^ delims^=^ eol^= %%L in ("%d%\\path") do if not defined p set "p=%%L"',
    'if defined p set "PATH=%p%"',
    'setlocal EnableDelayedExpansion',
    'cmd.exe /d /s /c "(!c!) >>"!d!\\output.log" 2>&1"',
    'set "rc=!errorlevel!"',
    '>"!d!\\exit.tmp" echo !rc! 0',
    'move /y "!d!\\exit.tmp" "!d!\\exit" >nul',
    '',
])

#: file name -> (content, executable)
LAUNCHER_FILES: dict[str, tuple[str, bool]] = {
    "bg": (BG_SH, True),
    "bg.cmd": (BG_CMD, False),
    "bg.ps1": (BG_PS1, False),
    "bg-run.cmd": (BG_RUN_CMD, False),
}


def worker_instructions(os_type: str) -> str:
    """The paragraph a Codex / Pi worker gets about the launcher."""
    launcher = ('orbital\\bin\\bg.cmd "<command>"' if os_type == "windows"
                else 'orbital/bin/bg "<command>"')
    return (
        "Long-running commands: never wait on a command that can take more than "
        "a couple of minutes (a build, a test suite, training, a download, a "
        f"server, a long script). Start it from the workspace root with `{launcher}` "
        "instead. It returns at once with a job id, the output file and the "
        "command that stops it. Then end your turn: Orbital sends you a message "
        "with the exit code and the last output lines when the job finishes, and "
        "checks in every 15 minutes while it runs. Do not sleep or poll for it."
    )


def launcher_path(workspace: str, os_type: str) -> str:
    """The launcher a worker on *os_type* runs."""
    name = "bg.cmd" if os_type == "windows" else "bg"
    return os.path.join(ProjectPaths(workspace).bin_dir, name)


def ensure_launcher(workspace: str) -> str | None:
    """Write the launchers into ``<workspace>/orbital/bin/``; return the dir.

    Never raises: a workspace Orbital cannot write to just has no launcher
    (the worker's attempt then fails visibly in its own shell).
    """
    if not workspace:
        return None
    bin_dir = ProjectPaths(workspace).bin_dir
    try:
        os.makedirs(bin_dir, exist_ok=True)
        for name, (content, executable) in LAUNCHER_FILES.items():
            path = os.path.join(bin_dir, name)
            data = content.encode("utf-8")
            try:
                with open(path, "rb") as f:
                    same = f.read() == data
            except OSError:
                same = False
            if not same:
                tmp = path + ".tmp"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, path)
            if executable:
                mode = os.stat(path).st_mode
                want = mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
                if mode != want:
                    os.chmod(path, want)
        return bin_dir
    except OSError:
        logger.warning("could not write the job launcher into %s", bin_dir,
                       exc_info=True)
        return None
