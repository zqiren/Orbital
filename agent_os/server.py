# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run the daemon on a server: the desktop app without its window and tray.

    python -m agent_os.server [--host 127.0.0.1] [--port 8000] [--data-dir DIR]

The daemon has no authentication of its own, so it binds to loopback unless
told otherwise. Anything wider belongs behind a proxy that authenticates.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_PORT = 8000


def default_data_dir() -> str:
    return os.environ.get("AGENT_OS_DATA_DIR") or str(Path.home() / ".orbital")


def default_spa_dir() -> str:
    return os.environ.get("AGENT_OS_SPA_DIR") or str(
        Path(__file__).resolve().parent.parent / "web" / "dist"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m agent_os.server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("AGENT_OS_PORT") or DEFAULT_PORT),
    )
    parser.add_argument("--data-dir", default=default_data_dir())
    return parser.parse_args(argv)


def configure_environment(args: argparse.Namespace) -> None:
    """Set what the desktop entry point sets before it creates the app."""
    os.environ["AGENT_OS_HEADLESS"] = "1"
    os.environ["AGENT_OS_PORT"] = str(args.port)
    os.environ["AGENT_OS_DATA_DIR"] = args.data_dir
    os.environ["AGENT_OS_SPA_DIR"] = default_spa_dir()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_environment(args)
    os.makedirs(args.data_dir, exist_ok=True)

    import uvicorn

    from agent_os.api.app import create_app
    from agent_os.platform.shell_path import merge_shell_path_into_environ

    merge_shell_path_into_environ()
    uvicorn.run(
        create_app(data_dir=args.data_dir),
        host=args.host, port=args.port, log_level="info",
    )


if __name__ == "__main__":
    main()
