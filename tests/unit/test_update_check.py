"""Update-availability checker (notify-only tier)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os import update_check
from agent_os.update_check import UpdateChecker, is_newer


class TestIsNewer:
    @pytest.mark.parametrize(
        "latest,current,expected",
        [
            ("0.9.0", "0.8.4", True),
            ("v0.9.0", "0.8.4", True),
            ("1.0.0", "0.9.9", True),
            ("0.8.4", "0.8.4", False),
            ("0.8.3", "0.8.4", False),
            ("0.9", "0.8.4", True),        # short form padded
            ("garbage", "0.8.4", False),   # unparseable never notifies
            ("0.9.0", "garbage", False),
            ("0.9.0", "0.0.0", False),     # version-module floor: never nag
        ],
    )
    def test_cases(self, latest, current, expected):
        assert is_newer(latest, current) is expected


def make_checker(fetch_result=None, fetch_exc=None, enabled=True, current="0.8.4"):
    ws = MagicMock()
    fetch = AsyncMock(return_value=fetch_result)
    if fetch_exc:
        fetch.side_effect = fetch_exc
    checker = UpdateChecker(ws, latest_url="http://stub/latest", enabled=enabled, fetch=fetch)
    checker.status["current"] = current
    return checker, ws, fetch


class TestUpdateChecker:
    async def test_newer_version_updates_status_and_broadcasts_once(self):
        checker, ws, _ = make_checker(
            {"version": "0.9.0", "url": "https://gh/rel"},
        )
        await checker.run_check()
        assert checker.status["update_available"] is True
        assert checker.status["latest"] == "0.9.0"
        ws.broadcast_global.assert_called_once_with(
            {"type": "update.available", "version": "0.9.0", "url": "https://gh/rel"}
        )
        # 6h re-check of the SAME version must not re-announce (the user may
        # have dismissed the pill).
        await checker.run_check()
        assert ws.broadcast_global.call_count == 1

    async def test_next_version_announces_again(self):
        checker, ws, fetch = make_checker({"version": "0.9.0", "url": "u"})
        await checker.run_check()
        fetch.return_value = {"version": "0.10.0", "url": "u2"}
        await checker.run_check()
        assert ws.broadcast_global.call_count == 2
        assert checker.status["latest"] == "0.10.0"

    async def test_same_or_older_version_is_silent(self):
        checker, ws, _ = make_checker({"version": "0.8.4", "url": "u"})
        await checker.run_check()
        assert checker.status["update_available"] is False
        ws.broadcast_global.assert_not_called()

    async def test_fetch_failure_is_silent(self):
        checker, ws, _ = make_checker(fetch_exc=OSError("offline"))
        await checker.run_check()  # must not raise
        assert checker.status["update_available"] is False
        ws.broadcast_global.assert_not_called()

    async def test_disabled_never_fetches(self):
        checker, ws, fetch = make_checker({"version": "9.9.9", "url": "u"}, enabled=False)
        await checker.run_check()
        fetch.assert_not_awaited()
        checker.start()
        assert checker._task is None

    def test_env_override_enables_in_dev(self, monkeypatch):
        monkeypatch.setenv("AGENT_OS_UPDATE_CHECK", "1")
        assert update_check._default_enabled() is True
        monkeypatch.setenv("AGENT_OS_UPDATE_CHECK", "0")
        assert update_check._default_enabled() is False

    def test_default_disabled_when_not_frozen(self, monkeypatch):
        monkeypatch.delenv("AGENT_OS_UPDATE_CHECK", raising=False)
        assert update_check._default_enabled() is False  # test runs unfrozen
