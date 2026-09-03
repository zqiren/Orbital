# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""TokenDance OAuth-key recovery contract (api-key-oauth.md#recover-key).

A failed gateway call may carry a ``TokenDance-Recovery-Action`` response
header; ``_classify_error`` turns the three documented actions into
actionable LLMError messages and leaves everything else untouched.
"""

import httpx
import pytest

from agent_os.agent.providers.openai_compat import _classify_error
from agent_os.agent.providers.types import LLMError


class _FakeResponse:
    def __init__(self, headers: dict[str, str]):
        self.headers = httpx.Headers(headers)


class _FakeStatusError(Exception):
    def __init__(self, message: str, status_code: int, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        if headers is not None:
            self.response = _FakeResponse(headers)


def _classify(exc: Exception) -> LLMError:
    with pytest.raises(LLMError) as ei:
        _classify_error(exc)
    return ei.value


def test_top_up_balance_appends_hint_and_keeps_status():
    err = _classify(_FakeStatusError(
        "insufficient balance", 402,
        {"TokenDance-Recovery-Action": "top_up_balance"},
    ))
    assert "top up" in err.message
    assert "still valid" in err.message
    assert "insufficient balance" in err.message
    assert err.status_code == 402


def test_reauthorize_points_at_global_settings_reconnect():
    err = _classify(_FakeStatusError(
        "key disabled", 401,
        {"TokenDance-Recovery-Action": "reauthorize_api_key"},
    ))
    assert "re-connect TokenDance in Global Settings" in err.message


def test_quota_action_offers_wait_or_reauthorize():
    err = _classify(_FakeStatusError(
        "quota exceeded", 429,
        {"TokenDance-Recovery-Action": "api_key_quota"},
    ))
    assert "quota" in err.message
    assert "refresh" in err.message


def test_header_is_case_insensitive():
    # httpx.Headers lookups are case-insensitive; the wire may downcase.
    err = _classify(_FakeStatusError(
        "nope", 401, {"tokendance-recovery-action": "reauthorize_api_key"},
    ))
    assert "re-connect TokenDance" in err.message


def test_unrecognized_action_falls_through_verbatim():
    # Their contract: unknown actions must be treated as ordinary errors.
    err = _classify(_FakeStatusError(
        "some error", 403, {"TokenDance-Recovery-Action": "brand_new_action"},
    ))
    assert err.message == "some error"


def test_absent_header_and_absent_response_untouched():
    err = _classify(_FakeStatusError("plain", 500, {}))
    assert err.message == "plain"
    err = _classify(_FakeStatusError("no response attr", 500))
    assert err.message == "no response attr"
