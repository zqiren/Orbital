# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Typed LLM provider/credential errors + classifier.

Start paths (cold-start scan, /agents/start, inject auto-start, triggers) and
the mid-run loop-error branch all funnel provider failures through here so the
frontend receives a stable machine ``code`` it can translate into an
actionable message ("add an API key in Settings") instead of a swallowed 500
or a log-only traceback.

Codes are part of the HTTP/WS contract with the frontend — keep them stable:
  missing_api_key      no key on the project, credential store, or settings
  invalid_api_key      provider rejected the key (401/403)
  model_not_found      provider returned 404 for the model/endpoint
  provider_unreachable network-level failure reaching base_url
  provider_error       anything else (raw message passed through)
"""


class ProviderConfigError(Exception):
    """Provider construction/credential failure with a stable machine code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def classify_llm_error(exc: BaseException) -> tuple[str, str]:
    """Map an exception from provider construction or an LLM call to
    ``(code, message)``.

    Classification is attribute-based (``status_code``) rather than
    isinstance-based so it covers both the openai and anthropic SDK error
    hierarchies (and any wrapper that preserves ``status_code``) without
    importing either.
    """
    if isinstance(exc, ProviderConfigError):
        return exc.code, exc.message

    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return "invalid_api_key", str(exc)
    if status == 404:
        return "model_not_found", str(exc)

    # openai.APIConnectionError / anthropic.APIConnectionError (and timeout
    # subclasses) carry no status_code; match on the class name.
    if "Connection" in type(exc).__name__:
        return "provider_unreachable", str(exc)

    return "provider_error", str(exc)
