# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""File browsing endpoints for Agent OS v2 API.

Provides directory listing, file content preview, upload, and download
within project workspaces.
"""

import base64
import mimetypes
import os

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/v2")

# ---- Dependency holders ----

_project_store = None


def configure(project_store):
    """Called by app factory to inject dependencies."""
    global _project_store
    _project_store = project_store


MAX_PREVIEW_BYTES = 512_000  # 500KB
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}


def _resolve_path(project_id: str, path: str):
    """Resolve a relative path within the project workspace.

    Returns (workspace, target) or raises HTTPException on error.
    """
    project = _project_store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    workspace = project["workspace"]
    target = os.path.normpath(os.path.join(workspace, path))

    # Path traversal protection
    if not target.startswith(os.path.normpath(workspace)):
        raise HTTPException(status_code=400, detail="Path outside workspace")

    return workspace, target


@router.get("/projects/{project_id}/files")
async def list_files(project_id: str, path: str = ""):
    _workspace, target = _resolve_path(project_id, path)

    if not os.path.isdir(target):
        raise HTTPException(status_code=404, detail="Directory not found")

    entries = []
    for name in sorted(os.listdir(target)):
        full = os.path.join(target, name)
        if os.path.isdir(full):
            entries.append({"name": name, "type": "directory"})
        else:
            try:
                stat = os.stat(full)
                entries.append({
                    "name": name,
                    "type": "file",
                    "size": stat.st_size,
                    "modified_at": stat.st_mtime,
                })
            except OSError:
                entries.append({"name": name, "type": "file", "size": 0})
    return {"path": path, "entries": entries}


@router.get("/projects/{project_id}/files/content")
async def get_file_content(project_id: str, path: str):
    _workspace, target = _resolve_path(project_id, path)

    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail="File not found")

    size = os.path.getsize(target)
    ext = os.path.splitext(target)[1].lower()
    mime_type = mimetypes.guess_type(target)[0] or "application/octet-stream"

    # Image files: return base64 encoded
    if ext in IMAGE_EXTENSIONS:
        try:
            with open(target, "rb") as f:
                data = f.read()
            content_b64 = base64.b64encode(data).decode("ascii")
        except OSError as e:
            raise HTTPException(status_code=400, detail=f"Cannot read file: {e}")
        return {
            "path": path,
            "content": content_b64,
            "type": "image",
            "mime": mime_type,
            "size": size,
        }

    # Try reading as text
    try:
        with open(target, "r", encoding="utf-8") as f:
            content = f.read(MAX_PREVIEW_BYTES)
        truncated = size > MAX_PREVIEW_BYTES
        return {
            "path": path,
            "content": content,
            "type": "text",
            "size": size,
            "truncated": truncated,
        }
    except (UnicodeDecodeError, ValueError):
        # Binary file (not text, not image) — include base64 for relay download
        max_download_bytes = 50 * 1024 * 1024
        content_b64 = ""
        if size <= max_download_bytes:
            try:
                with open(target, "rb") as f:
                    content_b64 = base64.b64encode(f.read()).decode("ascii")
            except OSError:
                pass
        return {
            "path": path,
            "type": "binary",
            "mime": mime_type,
            "size": size,
            "content": content_b64,
            "download_url": f"/api/v2/projects/{project_id}/files/download?path={path}",
        }
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Cannot read file: {e}")


@router.post("/projects/{project_id}/files/upload")
async def upload_file(project_id: str, file: UploadFile, path: str = "/uploads/"):
    workspace, target_dir = _resolve_path(project_id, path.lstrip("/"))

    # Read file content with size limit
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")

    # Create upload directory if needed
    os.makedirs(target_dir, exist_ok=True)

    # Sanitize filename: only keep the basename to prevent path traversal via filename
    safe_name = os.path.basename(file.filename or "upload")
    if not safe_name:
        safe_name = "upload"

    dest = os.path.join(target_dir, safe_name)

    # Verify the destination is still within workspace
    if not os.path.normpath(dest).startswith(os.path.normpath(workspace)):
        raise HTTPException(status_code=400, detail="Path outside workspace")

    with open(dest, "wb") as f:
        f.write(data)

    rel_path = os.path.join(path.lstrip("/"), safe_name)
    return {"path": rel_path, "size": len(data)}


@router.get("/projects/{project_id}/files/download")
async def download_file(project_id: str, path: str):
    _workspace, target = _resolve_path(project_id, path)

    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail="File not found")

    mime_type = mimetypes.guess_type(target)[0] or "application/octet-stream"
    return FileResponse(target, media_type=mime_type)
