#!/usr/bin/env python3
"""Isolated dev daemon: own PID file, explicit data dir, in-memory keyring.

Used by scripts/i18n-editor.sh, and handy on its own whenever you need a repo
daemon that coexists with the packaged Orbital.app and survives a headless start:

  .venv/bin/python scripts/dev_daemon_isolated.py --port 8331 --data-dir ~/.orbital-i18n-editor/data

- The PID singleton is redirected from ~/orbital/daemon.pid to <data-dir>/dev-daemon.pid,
  so the packaged app (or another dev daemon) keeps running untouched.
- keyring is replaced with a write-capable in-memory backend, so nothing ever
  touches the macOS Keychain (whose access prompt hangs a headless daemon).
  Consequence: API keys saved through the UI live only as long as this process.
- Keep --data-dir outside ~/Desktop: a daemon running sandboxed tools against a
  Desktop folder can make macOS revoke the terminal's Desktop access.
"""
import argparse
import os
import pathlib
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--port", type=int, default=8331)
parser.add_argument("--data-dir", required=True)
parser.add_argument("--host", default="127.0.0.1")
args = parser.parse_args()

root = pathlib.Path(__file__).resolve().parents[1]
data = pathlib.Path(args.data_dir).expanduser().resolve()
data.mkdir(parents=True, exist_ok=True)
os.chdir(root)
sys.path.insert(0, str(root))
os.environ.setdefault("AGENT_OS_TELEMETRY_DISABLED", "1")
os.environ.setdefault("AGENT_OS_DATA_DIR", str(data))

import keyring  # noqa: E402
from keyring.backend import KeyringBackend  # noqa: E402
from keyring.errors import PasswordDeleteError  # noqa: E402


class MemoryKeyring(KeyringBackend):
    """Write-capable keyring that never leaves this process."""

    priority = 1
    _store: dict = {}

    def get_password(self, service, username):
        return self._store.get((service, username))

    def set_password(self, service, username, password):
        self._store[(service, username)] = password

    def delete_password(self, service, username):
        if (service, username) not in self._store:
            raise PasswordDeleteError("no such item")
        del self._store[(service, username)]


keyring.set_keyring(MemoryKeyring())

import agent_os.utils.pid_file as pid_file  # noqa: E402

pid_file._DEFAULT_PID_PATH = data / "dev-daemon.pid"

import uvicorn  # noqa: E402
from agent_os.api.app import create_app  # noqa: E402

print(f"[dev-daemon] code={root} data={data} port={args.port} pid={os.getpid()} keyring=in-memory", flush=True)
uvicorn.run(create_app(data_dir=str(data)), host=args.host, port=args.port, log_level="info")
