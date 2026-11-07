# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The headless entry point: safe defaults and the environment it sets up."""

import os
from pathlib import Path

import pytest

from agent_os import server


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("AGENT_OS_PORT", "AGENT_OS_DATA_DIR", "AGENT_OS_SPA_DIR",
                 "AGENT_OS_HEADLESS"):
        monkeypatch.delenv(name, raising=False)


def test_binds_to_loopback_by_default():
    assert server.parse_args([]).host == "127.0.0.1"


def test_defaults_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("AGENT_OS_PORT", "9123")
    monkeypatch.setenv("AGENT_OS_DATA_DIR", "/srv/orbital")
    args = server.parse_args([])
    assert args.port == 9123
    assert args.data_dir == "/srv/orbital"


def test_arguments_override_the_environment(monkeypatch):
    monkeypatch.setenv("AGENT_OS_PORT", "9123")
    args = server.parse_args(["--port", "8100", "--host", "0.0.0.0"])
    assert args.port == 8100
    assert args.host == "0.0.0.0"


def test_data_dir_defaults_to_the_home_folder():
    assert server.parse_args([]).data_dir == str(Path.home() / ".orbital")


def test_configure_environment_marks_the_daemon_headless():
    server.configure_environment(server.parse_args(["--port", "8100"]))
    assert os.environ["AGENT_OS_HEADLESS"] == "1"
    assert os.environ["AGENT_OS_PORT"] == "8100"
    assert os.environ["AGENT_OS_SPA_DIR"].endswith(os.path.join("web", "dist"))


def test_configure_environment_keeps_a_custom_spa_dir(monkeypatch):
    monkeypatch.setenv("AGENT_OS_SPA_DIR", "/opt/spa")
    server.configure_environment(server.parse_args([]))
    assert os.environ["AGENT_OS_SPA_DIR"] == "/opt/spa"
