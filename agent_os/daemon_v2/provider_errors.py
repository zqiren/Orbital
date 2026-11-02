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
  missing_card         the project points at a credential card that no longer
                       exists and there is no default card
  missing_api_key      the resolved credential card carries no key
  invalid_api_key      provider rejected the key (401, or a 403 that says so)
  model_not_found      provider returned 404 for the model/endpoint
  provider_unreachable network-level failure reaching base_url
  provider_error       any other failure from the LLM call path (raw message
                       passed through)
  internal_error       an exception that did NOT come from the LLM call path —
                       Orbital's own bug (e.g. a file it could not read). The
                       UI must not send the user to provider settings for it.
"""

from agent_os.agent.providers.types import (
    is_rejected_credential, unwrap_provider_message,
)

# Modules whose exceptions are, by construction, failures of the LLM call
# path: the provider SDKs, their HTTP transport, and Orbital's own provider
# layer (which wraps SDK errors in LLMError / ContextOverflowError — a timeout
# LLMError carries ``status_code=None``).
_LLM_PATH_MODULES = ("openai", "anthropic", "httpx", "httpcore",
                     "agent_os.agent.providers")


def _from_llm_call_path(exc: BaseException) -> bool:
    if hasattr(exc, "status_code"):
        return True
    module = type(exc).__module__ or ""
    return any(module == m or module.startswith(m + ".") for m in _LLM_PATH_MODULES)


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

    Spec 141 §1 C: only exceptions from the LLM call path keep the provider
    codes; everything else is ``internal_error`` (the old fallthrough
    labelled e.g. a ``UnicodeDecodeError`` reading a memory file as a
    provider failure and sent the user to settings).
    """
    if isinstance(exc, ProviderConfigError):
        return exc.code, exc.message

    if not _from_llm_call_path(exc):
        return "internal_error", f"{type(exc).__name__}: {exc}"

    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        if is_rejected_credential(status, str(exc)):
            return "invalid_api_key", str(exc)
        # A region gate, a privacy setting: the key is fine, say what was said.
        return "provider_error", unwrap_provider_message(str(exc))
    if status == 404:
        return "model_not_found", str(exc)

    # openai.APIConnectionError / anthropic.APIConnectionError (and timeout
    # subclasses) carry no status_code; match on the class name.
    if "Connection" in type(exc).__name__:
        return "provider_unreachable", str(exc)

    return "provider_error", str(exc)
