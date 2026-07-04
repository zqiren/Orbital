# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Event↔project linkage store (Spec 011 §0.4/§2; Task G1).

A flat ``{event_id: project_id}`` map persisted at
``orbital-data/calendar_links.json``. The external calendar stays the source of
truth for events; this file is Orbital metadata layered on top — which project
an event belongs to. Loaded lazily on first access, written atomically
(temp-file + ``os.replace``) so a crash mid-write never corrupts the map.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


class Linkage:
    def __init__(self, data_dir: str = "orbital-data"):
        self._data_dir = data_dir
        self._links: dict[str, str] | None = None  # None until first load

    def _path(self) -> str:
        return os.path.join(self._data_dir, "calendar_links.json")

    def _ensure_loaded(self) -> None:
        if self._links is None:
            self._load()

    def _load(self) -> None:
        self._links = {}
        path = self._path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self._links = {
                    str(k): str(v) for k, v in data.items() if v is not None
                }
        except (json.JSONDecodeError, OSError, ValueError):
            logger.warning("failed to load calendar linkage from %s", path, exc_info=True)
            self._links = {}

    def get(self, event_id: str) -> str | None:
        self._ensure_loaded()
        assert self._links is not None
        return self._links.get(event_id)

    def all(self) -> dict[str, str]:
        self._ensure_loaded()
        assert self._links is not None
        return dict(self._links)

    def link(self, event_id: str, project_id: str) -> None:
        self._ensure_loaded()
        assert self._links is not None
        self._links[event_id] = project_id
        self._save()

    def unlink(self, event_id: str) -> None:
        self._ensure_loaded()
        assert self._links is not None
        self._links.pop(event_id, None)
        self._save()

    def _save(self) -> None:
        os.makedirs(self._data_dir, exist_ok=True)
        path = self._path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._links, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)  # atomic on POSIX + Windows
