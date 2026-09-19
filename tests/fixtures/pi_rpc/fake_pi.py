# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A fake ``pi --mode rpc`` child for the PiRPCTransport tests.

Speaks the same LF-delimited JSONL protocol as the released pi 0.85.1 but
never talks to a provider: every turn replays a scenario file, normally one
of the redacted records captured from the real binary next to this script.

Controlled entirely through environment variables:

FAKE_PI_RECORD            append one JSON line per spawn / command / child here
FAKE_PI_ENV_KEYS          comma list of env var names to copy into the record
FAKE_PI_STDERR            text written to stderr at startup
FAKE_PI_EXIT_AT_START     exit with this code before answering anything
FAKE_PI_EXIT_WITHOUT_SESSION  exit 2 when started without ``--session``
FAKE_PI_KNOWN_SESSIONS    comma list of session ids ``--session`` can open
FAKE_PI_RESUME_MODE       "mismatch": open a new id instead of exiting 1
FAKE_PI_NEW_SESSION_ID    id of a freshly created session
FAKE_PI_SCENARIO          JSONL replayed for every prompt

Scenario directives (lines that are not protocol records):
{"__sleep__": s}            pause
{"__raw__": "text"}         write text + LF verbatim (malformed records)
{"__big_line__": n}         write one valid record of about n bytes
{"__exit__": code}          exit immediately
{"__spawn_child__": true}   start a long-running grandchild (a "tool")
{"__wait_for__": type}      block until a command of that type arrives
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
_out_lock = threading.Lock()


def _arg(argv, flag):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _write(data: bytes) -> None:
    with _out_lock:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()


def emit(obj) -> None:
    _write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))


def record(entry) -> None:
    path = os.environ.get("FAKE_PI_RECORD")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def _read_stdin(commands: "queue.Queue") -> None:
    for line in sys.stdin.buffer:
        try:
            cmd = json.loads(line.decode("utf-8"))
        except ValueError:
            emit({"type": "response", "command": "parse", "success": False,
                  "error": "Failed to parse command"})
            continue
        record({"event": "command", "cmd": cmd})
        commands.put(cmd)
    commands.put(None)


def _state(session_id: str, session_dir: str | None) -> dict:
    with open(os.path.join(HERE, "get_state_response.json"), encoding="utf-8") as f:
        data = json.load(f)["data"]
    data["sessionId"] = session_id
    data["sessionFile"] = os.path.join(session_dir or HERE, f"{session_id}.jsonl")
    return data


def _scenario() -> list:
    path = os.environ.get("FAKE_PI_SCENARIO")
    if not path:
        return [{"type": "response", "command": "prompt", "success": True},
                {"type": "agent_start"}, {"type": "agent_settled"}]
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _wait_for(commands, wanted: str) -> dict:
    while True:
        cmd = commands.get()
        if cmd is None:
            sys.exit(0)
        if cmd.get("type") == wanted:
            return cmd
        if cmd.get("type") == "prompt":
            emit({"id": cmd.get("id"), "type": "response", "command": "prompt",
                  "success": False, "error": "Agent is already processing."})


def _run_prompt(cmd: dict, commands) -> None:
    abort_id = None
    children = []
    for rec in _scenario():
        if "__sleep__" in rec:
            time.sleep(rec["__sleep__"])
        elif "__raw__" in rec:
            _write((rec["__raw__"] + "\n").encode("utf-8"))
        elif "__big_line__" in rec:
            emit({"type": "message_update", "assistantMessageEvent": {
                "type": "text_delta", "contentIndex": 0, "delta": "x" * rec["__big_line__"]}})
        elif "__exit__" in rec:
            sys.stdout.flush()
            os._exit(rec["__exit__"])
        elif "__spawn_child__" in rec:
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
            children.append(child)
            record({"event": "child", "pid": child.pid})
        elif "__wait_for__" in rec:
            got = _wait_for(commands, rec["__wait_for__"])
            if got.get("type") == "abort":
                abort_id = got.get("id")
        elif rec.get("type") == "response" and rec.get("command") == "prompt" and "id" in rec:
            emit({**rec, "id": cmd.get("id")})
        elif rec.get("type") == "response" and rec.get("command") == "prompt":
            emit({**rec, "id": cmd.get("id")})
        elif rec.get("type") == "response" and rec.get("command") == "abort":
            emit({**rec, "id": abort_id})
        else:
            emit(rec)


def main() -> int:
    argv = sys.argv[1:]
    env_keys = [k for k in os.environ.get("FAKE_PI_ENV_KEYS", "").split(",") if k]
    append = _arg(argv, "--append-system-prompt")
    append_text = None
    if append is not None:
        append_text = (open(append, encoding="utf-8").read()
                       if os.path.isfile(append) else append)
    requested = _arg(argv, "--session")
    record({"event": "spawn", "argv": argv, "cwd": os.getcwd(), "pid": os.getpid(),
            "env": {k: os.environ.get(k) for k in env_keys},
            "append_prompt": append_text})
    if os.environ.get("FAKE_PI_STDERR"):
        sys.stderr.write(os.environ["FAKE_PI_STDERR"] + "\n")
        sys.stderr.flush()
    if os.environ.get("FAKE_PI_EXIT_AT_START"):
        return int(os.environ["FAKE_PI_EXIT_AT_START"])
    if requested is None and os.environ.get("FAKE_PI_EXIT_WITHOUT_SESSION"):
        return 2

    session_id = os.environ.get("FAKE_PI_NEW_SESSION_ID", "fresh-session-1")
    if requested is not None:
        known = [s for s in os.environ.get("FAKE_PI_KNOWN_SESSIONS", "").split(",") if s]
        if requested in known:
            session_id = requested
        elif os.environ.get("FAKE_PI_RESUME_MODE") != "mismatch":
            sys.stderr.write(f"No session found matching '{requested}'\n")
            sys.stderr.flush()
            return 1

    commands: "queue.Queue" = queue.Queue()
    threading.Thread(target=_read_stdin, args=(commands,), daemon=True).start()
    session_dir = _arg(argv, "--session-dir")
    while True:
        cmd = commands.get()
        if cmd is None:
            return 0
        ctype = cmd.get("type")
        if ctype == "get_state":
            emit({"id": cmd.get("id"), "type": "response", "command": "get_state",
                  "success": True, "data": _state(session_id, session_dir)})
        elif ctype == "prompt":
            _run_prompt(cmd, commands)
        elif ctype == "abort":
            emit({"id": cmd.get("id"), "type": "response", "command": "abort", "success": True})
        elif ctype == "extension_ui_response":
            continue
        else:
            emit({"id": cmd.get("id"), "type": "response", "command": ctype,
                  "success": False, "error": f"Unknown command: {ctype}"})


if __name__ == "__main__":
    sys.exit(main())
