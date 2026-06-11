# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pricing-table API (Budget Piece 3 / Task P3-D).

Two endpoints, backend-only (the editor UI is a later task):

``GET /api/v2/pricing``
    The resolved rates table with PER-FIELD origin, covering every
    provider/model with pricing in ``providers.json`` PLUS any extra models
    present only in the override file (forward-compat). Resolution agrees
    EXACTLY with ``agent_os.agent.pricing.resolve_rates`` — this route does not
    reimplement the merge; it calls ``pricing.pricing_table_with_origin``,
    which shares ``resolve_rates``'s internals. Also reports the resolved
    override-file path as a read-only debug field.

``PUT /api/v2/pricing/overrides``
    A validated, atomic, **full-replace** write of ``pricing-overrides.json``.

PUT semantics — full replace (NOT merge)
----------------------------------------
The body becomes the entire new override file. This matches an editor that
loads the current overrides, edits in place, and saves the whole document; a
field is reverted to its default simply by omitting it from the PUT body (the
frontend's per-field revert), and there is no separate delete verb to reason
about. Merge semantics would make "revert" ambiguous (you'd need an explicit
null/sentinel to unset a key), so we deliberately chose full-replace.

The override path is the same module-level ``_OVERRIDE_FILE`` that
``pricing.py`` resolves from ``AGENT_OS_DATA_DIR`` (injectable by tests via
``pricing._OVERRIDE_FILE``). The route never hardcodes a path — it reads
``pricing.override_file_path()`` so a GET after a PUT round-trips, and so the
mtime-invalidation from Piece 1 makes the new rates take effect WITHOUT a
daemon restart.

Validation (codes-only 400 bodies, mirroring ``{"code": "invalid_window"}``)
----------------------------------------------------------------------------
- Rate values must be numeric and non-negative (``invalid_rate``).
- ``_currency`` must be a string (``invalid_currency``).
- The top-level body and each provider/model entry must be a JSON object
  (``invalid_shape``).
- Unknown provider/model keys are ALLOWED (forward-compat). ``_currency`` and
  ``_default`` keys are allowed per the override schema.
"""

import json
import os
import tempfile

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from agent_os.agent import pricing

router = APIRouter(prefix="/api/v2")


# A bare APIRouter with no injected dependencies: pricing resolution reads the
# checked-in providers.json and the module-level override path directly. The
# ``configure`` shim exists only for symmetry with the other routers' wiring in
# app.py (and as a forward hook should this route ever need injected state).
def configure() -> None:  # pragma: no cover - no state to inject today
    return None


# ---------------------------------------------------------------------------
# GET — resolved table + per-field origin + override path
# ---------------------------------------------------------------------------

@router.get("/pricing")
async def get_pricing():
    """Return the resolved rates table with per-field origin.

    Shape (see ``pricing.pricing_table_with_origin``)::

        {
          "override_path": "/abs/path/pricing-overrides.json",
          "providers": {
            "<provider>": {
              "currency": "USD",
              "currency_origin": "default" | "override",
              "models": {
                "<model>": {
                  "input_per_1m":        {"value": 3.0,  "origin": "default"},
                  "cached_input_per_1m": {"value": 0.3,  "origin": "default"},
                  "cache_write_per_1m":  {"value": 3.75, "origin": "default"},
                  "output_per_1m":       {"value": 15.0, "origin": "default"}
                }
              }
            }
          }
        }
    """
    table = pricing.pricing_table_with_origin()
    table["override_path"] = pricing.override_file_path()
    return table


# ---------------------------------------------------------------------------
# PUT — validated, atomic, full-replace write of the override file
# ---------------------------------------------------------------------------

def _error(code: str) -> JSONResponse:
    """Codes-only 400 body, matching the existing error convention."""
    return JSONResponse(status_code=400, content={"code": code})


def _is_number(value) -> bool:
    # bool is a subclass of int — reject it as a rate value.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_overrides(body) -> str | None:
    """Validate a sparse override document. Return an error code or None.

    Mirrors the override-file schema: a dict keyed by provider, then model (or
    ``_default``), each carrying any subset of the four per-1M rate fields. A
    ``_currency`` string may sit at the provider level. Unknown provider/model
    keys are accepted (forward-compat).
    """
    if not isinstance(body, dict):
        return "invalid_shape"

    for provider, provider_block in body.items():
        if not isinstance(provider_block, dict):
            return "invalid_shape"
        for key, entry in provider_block.items():
            if key == "_currency":
                if not isinstance(entry, str):
                    return "invalid_currency"
                continue
            # key is a model name or "_default" → a rate-field dict.
            if not isinstance(entry, dict):
                return "invalid_shape"
            for field, value in entry.items():
                # Unknown rate-field names are tolerated (forward-compat), but
                # whatever value is present must be a non-negative number.
                if not _is_number(value):
                    return "invalid_rate"
                if value < 0:
                    return "invalid_rate"
    return None


def _atomic_write_json(path: str, data: dict) -> None:
    """Write ``data`` as JSON to ``path`` atomically (tmp + rename).

    Shifts the file mtime forward after the rename as a monotonic-forward
    nudge for the mtime-keyed override cache in pricing.py. Real invalidation
    relies on sub-second mtime resolution (APFS/ext4/NTFS all provide it); a
    constant shift cannot separate two writes on a 1s-granularity filesystem.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=".pricing-overrides-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    # Guarantee the mtime advances so the override cache invalidates.
    st = os.stat(path)
    os.utime(path, (st.st_atime, st.st_mtime + 1.0))


@router.put("/pricing/overrides")
async def put_pricing_overrides(request: Request):
    """Full-replace the override file with a validated sparse document.

    Returns the freshly-resolved pricing table (same shape as GET) on success
    so the caller sees the post-write origins without a second round-trip. On
    validation failure (or unparseable JSON) returns a codes-only 400 — we read
    the raw body ourselves so every malformed shape lands as ``400`` with a
    ``code``, not FastAPI's default ``422``.
    """
    try:
        raw = await request.body()
        body = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _error("invalid_json")

    err = _validate_overrides(body)
    if err is not None:
        return _error(err)

    path = pricing.override_file_path()
    _atomic_write_json(path, body)

    table = pricing.pricing_table_with_origin()
    table["override_path"] = path
    return table
