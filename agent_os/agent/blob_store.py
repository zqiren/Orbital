# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Content-addressed store for tool-result bytes (spec 066 phase 1).

Tool output that used to be written once per reference — base64 images
inlined into session JSONL rows, and one archive file per tool call even when
the content was byte-identical to an earlier call — is stored here ONCE, named
by its SHA-256, and referenced by hash:

    {workspace}/orbital/tool-results/blobs/<sha[:2]>/<sha256>.<ext>

The store only ever ADDS files. Nothing here deletes, moves or rewrites data
that existed before; savings start with data written after this module
shipped. A blob is written atomically (temp file + ``os.replace``), so a crash
leaves either the whole blob or none of it.

Session rows keep a reference instead of the base64 payload (see
``deinline_row_images``). The reference is shaped so an Orbital that predates
this store degrades gracefully: the image block becomes a plain text block
naming the blob's path, and the machine-readable reference rides in ``_meta``,
which every provider adapter strips before a request. Readers that know about
the store call ``rehydrate_row_images`` and get back the exact original row.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import os
import re
import uuid

logger = logging.getLogger(__name__)

# Relative to the orbital/ dir. The blobs are tool-result content (browser
# screenshots, images read by the agent, archived text results), so they live
# with the rest of the tool output rather than beside the durable memory files.
_BLOBS_RELDIR = os.path.join("tool-results", "blobs")

# ``_meta`` keys carrying the image references of a de-inlined row.
META_KEY = "image_blobs"
# Set when de-inlining had to CREATE ``_meta``, so rehydration can drop it
# again and hand back a row without a ``_meta`` key, exactly as it was.
_META_CREATED_KEY = "image_blobs_created_meta"

_DATA_URL_RE = re.compile(r"^data:([A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/=]*)$")

_EXT_BY_MIME = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "image/x-icon": "ico",
    "image/tiff": "tiff",
    "image/svg+xml": "svg",
}


def blobs_dir(orbital_dir: str) -> str:
    """Root of the blob store for the project whose orbital/ dir is given."""
    return os.path.join(orbital_dir, _BLOBS_RELDIR)


def blob_path(orbital_dir: str, sha256: str, ext: str) -> str:
    """Absolute path of the blob with this hash and extension."""
    return os.path.join(blobs_dir(orbital_dir), sha256[:2], f"{sha256}.{ext}")


def blob_relpath(sha256: str, ext: str) -> str:
    """Workspace-relative path of a blob (for human/model-facing text)."""
    return "/".join(("orbital", "tool-results", "blobs", sha256[:2], f"{sha256}.{ext}"))


def _matches(path: str, data: bytes, sha256: str) -> bool:
    """True when the file at ``path`` holds exactly ``data``."""
    try:
        if os.path.getsize(path) != len(data):
            return False
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest() == sha256
    except OSError:
        return False


def put_bytes(orbital_dir: str, data: bytes, ext: str) -> tuple[str, str]:
    """Store ``data`` once; return ``(sha256, absolute_path)``.

    An existing blob with the same hash is reused when its bytes still match
    (the common, free case). A blob whose bytes no longer match its name —
    edited in place by hand — is replaced atomically with the right bytes.
    Raises ``OSError`` when the store cannot be written.
    """
    sha256 = hashlib.sha256(data).hexdigest()
    path = blob_path(orbital_dir, sha256, ext)
    if _matches(path, data, sha256):
        return sha256, path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return sha256, path


def read_verified(orbital_dir: str, sha256: str, ext: str) -> bytes | None:
    """Bytes of a blob, or None when it is missing or no longer matches its hash."""
    path = blob_path(orbital_dir, sha256, ext)
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if hashlib.sha256(data).hexdigest() != sha256:
        logger.warning("Blob %s does not match its hash; treating it as missing", path)
        return None
    return data


# ---------------------------------------------------------------------------
# Session rows: base64 images <-> blob references
# ---------------------------------------------------------------------------

def _inline_image(block) -> tuple[str, str, bytes] | None:
    """``(mime, payload, raw)`` for an image_url block holding a canonical
    base64 data URL, else None.

    Canonical means re-encoding the decoded bytes reproduces the payload
    exactly — the guarantee that rehydration hands back the identical string.
    Anything else (wrapped lines, missing padding, a remote URL) stays inline.
    """
    if not isinstance(block, dict) or block.get("type") != "image_url":
        return None
    image_url = block.get("image_url")
    if not isinstance(image_url, dict):
        return None
    url = image_url.get("url")
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    m = _DATA_URL_RE.match(url)
    if m is None:
        return None
    mime, payload = m.group(1), m.group(2)
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return None
    if base64.b64encode(raw).decode("ascii") != payload:
        return None
    return mime, payload, raw


def has_inline_images(row) -> bool:
    """True when the row carries at least one base64 image in its content."""
    if not isinstance(row, dict):
        return False
    content = row.get("content")
    if not isinstance(content, list):
        return False
    return any(_inline_image(b) is not None for b in content)


def _placeholder_text(sha256: str, ext: str) -> str:
    return f"[image stored outside this log: {blob_relpath(sha256, ext)}]"


def deinline_row_images(row: dict, orbital_dir: str) -> dict:
    """Return ``row`` with every inline base64 image moved to the blob store.

    The caller's row is never mutated: when there is anything to move, a new
    row (new content list, new ``_meta`` dict) is returned; otherwise the very
    same object comes back. Each moved image leaves a text block naming the
    blob, plus an entry under ``_meta.image_blobs`` holding the hash, the MIME
    type and the original block with its URL swapped for a reference —
    everything ``rehydrate_row_images`` needs to restore the row exactly.

    A blob that cannot be written leaves that image inline: a row is never
    stored with a reference to bytes that are not on disk.
    """
    if not has_inline_images(row):
        return row
    meta = row.get("_meta")
    if meta is not None and not isinstance(meta, dict):
        return row
    content = list(row["content"])
    refs = list(meta.get(META_KEY) or []) if meta else []
    moved = False
    for index, block in enumerate(content):
        parsed = _inline_image(block)
        if parsed is None:
            continue
        mime, _payload, raw = parsed
        ext = _EXT_BY_MIME.get(mime, "bin")
        try:
            sha256, _path = put_bytes(orbital_dir, raw, ext)
        except OSError:
            logger.warning("Could not store an image blob; keeping it inline", exc_info=True)
            continue
        template = dict(block)
        template["image_url"] = {
            k: (f"blob:sha256:{sha256}" if k == "url" else v)
            for k, v in block["image_url"].items()
        }
        content[index] = {"type": "text", "text": _placeholder_text(sha256, ext)}
        refs.append({
            "index": index,
            "sha256": sha256,
            "ext": ext,
            "mime": mime,
            "block": template,
        })
        moved = True
    if not moved:
        return row
    new_meta = dict(meta) if meta is not None else {_META_CREATED_KEY: True}
    new_meta[META_KEY] = refs
    out = dict(row)
    out["content"] = content
    out["_meta"] = new_meta
    return out


def rehydrate_row_images(row: dict, orbital_dir: str) -> dict:
    """Inverse of ``deinline_row_images``: put the base64 images back.

    Returns a new row equal (key order included) to the one that was
    de-inlined. A reference whose blob is missing or damaged is left as its
    placeholder text block and its ``_meta`` entry is kept, so nothing is lost
    and a restored blob store brings the image back. Rows with no references
    come back as the very same object.
    """
    if not isinstance(row, dict):
        return row
    meta = row.get("_meta")
    if not isinstance(meta, dict) or not meta.get(META_KEY):
        return row
    content = row.get("content")
    if not isinstance(content, list):
        return row
    content = list(content)
    unresolved = []
    for ref in meta[META_KEY]:
        try:
            index = ref["index"]
            sha256 = ref["sha256"]
            ext = ref["ext"]
            mime = ref["mime"]
            template = ref["block"]
        except (KeyError, TypeError):
            unresolved.append(ref)
            continue
        raw = read_verified(orbital_dir, sha256, ext)
        if raw is None or not (0 <= index < len(content)) or not isinstance(template, dict):
            unresolved.append(ref)
            continue
        data_url = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
        block = dict(template)
        block["image_url"] = {
            k: (data_url if k == "url" else v)
            for k, v in (template.get("image_url") or {}).items()
        }
        content[index] = block
    out = dict(row)
    out["content"] = content
    new_meta = dict(meta)
    if unresolved:
        new_meta[META_KEY] = unresolved
        out["_meta"] = new_meta
        return out
    new_meta.pop(META_KEY, None)
    if new_meta.pop(_META_CREATED_KEY, False) and not new_meta:
        del out["_meta"]
    else:
        out["_meta"] = new_meta
    return out
