# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Task sources: the one pipeline every type shares.

A source is something outside the daemon watches. Arrivals are detected by
adapter code with no model call; the first model call for an arrival is the
Workbench's normal routing (``POST /workbench/tasks``'s path, project and
agent on 自动).

The pipeline, for every type:

* **De-duplication** — each item id is recorded once per source; an adapter
  may re-read an overlapping window every time.
* **Baseline** — a polled kind's first successful poll records what is
  already there as seen, so connecting does not flood the Workbench.
* **Merge** — an item whose subject matches an entry whose task is still
  open (waiting, held, being sent, unrouted, or a session not yet done)
  joins that entry (count + updates) instead of opening a new task.
* **Hold** — while a source is paused its arrivals are recorded ``held``;
  one catch-up sends each at most once, or they are dismissed.
* **One at a time** — a single dispatcher sends one arrival at a time, and
  a source's next arrival waits while the session its previous one became
  is still running: a burst never starts N agents.
* **Material, not instructions** — see ``framing.py``.
* **Gaps** — lossy long-connection sources record when they were not
  connected (including while the daemon was not running).
* **Failure** — an adapter's ``SourceFailure`` puts the source in a failed
  state with a code and the actions that can help; nothing repairs it by
  itself except the scheduled re-check of retryable failures.

Runner: one tick loop evaluates wall-clock due-ness each tick (the
TriggerManager's pattern: survives sleep, catches up after it), polls run
concurrently up to a small bound, long connections are supervised tasks with
backoff, and every subprocess / blocking call runs off the loop.
"""

from __future__ import annotations

import asyncio
import logging
import random
import secrets as _pyrandom
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from agent_os.sources import catalog, framing
from agent_os.sources.context import SourceContext
from agent_os.sources.model import Item, Resolve, SourceFailure, VerifyError
from agent_os.sources.store import SourcesStore, safe_source_id

logger = logging.getLogger(__name__)

TICK_S = 15.0
POLL_CONCURRENCY = 3
POLL_TIMEOUT_S = 150.0
DISPATCH_IDLE_S = 5.0
GAP_MIN_S = 600
SEEN_CAP = 5000
ENTRY_CAP = 500
UPDATES_CAP = 20
HISTORY_DEFAULT = 50
HISTORY_MAX = 200
MAX_BACKOFF_S = 3600
STREAM_MAX_BACKOFF_S = 300
ALIVE_PERSIST_S = 60

ACTIVE_STATES = ("waiting", "routing", "held", "unrouted")

TYPE_WORDS = {
    "github": "GitHub", "gitlab": "GitLab", "jira": "Jira", "email": "E-mail",
    "calendar": "Calendar", "website": "Website check", "telegram": "Telegram",
    "slack": "Slack", "discord": "Discord", "feishu": "Feishu", "dingtalk": "DingTalk",
    "wecom": "WeCom", "command": "Check command",
}

# VerifyError.reason at a later check → failure code (retryable?)
_REASON_FAILURE = {
    "rejected": ("token_rejected", False),
    "not_found": ("repo_not_found", True),
    "signed_out": ("cli_signed_out", True),
    "cli_missing": ("cli_missing", True),
    "permission": ("calendar_permission", True),
    "unreachable": ("unreachable", True),
    "invalid": ("error", True),
    "error": ("error", True),
}


class ApiError(Exception):
    def __init__(self, status: int, detail):
        super().__init__(str(detail))
        self.status = status
        self.detail = detail


class SendError(Exception):
    """The Workbench refused the task. ``reason``: ``routing_failed`` |
    ``missing_api_key`` | ``no_project`` | ``send_failed``."""

    def __init__(self, reason: str, status: int = 400, detail=None):
        super().__init__(reason)
        self.reason = reason
        self.status = status
        self.detail = detail if detail is not None else reason


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(value) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{_pyrandom.token_hex(6)}"


class SourcesCore:
    """Owns the sources store, the runner and the dispatcher.

    Injected:
      * ``send(text, project_id|None) -> {"project_id", "session_id", "by"}``
        (async; raises :class:`SendError`) — the Workbench task path;
      * ``session_info(project_id, session_id) -> (project_id, state)``
        (async) — where the session is now (it may have been moved) and
        ``"running"`` | ``"needs_you"`` | ``"done"``;
      * ``project_name(project_id) -> str | None``;
      * ``broadcast(payload)`` — every connected client.
    """

    def __init__(self, data_dir: str, *, registry, secret_store, send=None,
                 session_info=None, project_name=None, broadcast=None, now_fn=None,
                 first_tick_delay_s: float = 0.0, tick_s: float = TICK_S, lang_fn=None,
                 session_refresh=None):
        self.store = SourcesStore(data_dir)
        self.registry = registry
        self.secrets = secret_store
        self._send = send
        self._session_info = session_info
        # Called before a history read: session lookups re-read once.
        self._session_refresh = session_refresh
        self._project_name = project_name or (lambda pid: None)
        self._broadcast = broadcast or (lambda payload: None)
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self.first_tick_delay_s = first_tick_delay_s
        self._lang_fn = lang_fn or (lambda: "en")
        self.tick_s = tick_s
        self._running = False
        self._tick_task: asyncio.Task | None = None
        self._dispatch_task: asyncio.Task | None = None
        self._streams: dict[str, asyncio.Task] = {}
        self._poll_locks: dict[str, asyncio.Lock] = {}
        self._polls_inflight: set[str] = set()
        self._wake_dispatch = asyncio.Event()
        self._wake_tick = asyncio.Event()
        self._session_index: dict[str, tuple[str, str]] = {}
        self._alive_saved: dict[str, datetime] = {}
        self._qr: dict[str, dict] = {}
        self._poll_sem = asyncio.Semaphore(POLL_CONCURRENCY)
        self._loaded = False

    # ================================================================
    # lifecycle
    # ================================================================

    def load(self) -> None:
        """Blocking: read the store, rebuild the session index, settle what
        a stop left half-done. Idempotent."""
        if self._loaded:
            return
        self.store.load()
        now = self._now()
        for cfg in self.store.config:
            sid = cfg["id"]
            rt = self.store.rt(sid)
            for e in rt["entries"]:
                if e.get("session_id"):
                    self._session_index[e["session_id"]] = (sid, e["id"])
                if e.get("state") == "routing":
                    # The daemon stopped mid-send. At most once: never resent
                    # by itself; the user can pick a project for it.
                    e["state"] = "unrouted"
                    e["reason"] = "send_failed"
            conn = rt.setdefault("conn", {"up": False, "down_since": None, "alive_at": None})
            if conn.get("up"):
                # It was connected when the daemon stopped: the time since
                # it was last seen alive is a gap.
                conn["up"] = False
                conn["down_since"] = conn.get("alive_at") or _iso(now)
        self._loaded = True

    async def start(self) -> None:
        if not self._loaded:
            await asyncio.to_thread(self.load)
        self._running = True
        self._tick_task = asyncio.create_task(self._tick_loop())
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        self._running = False
        tasks = [t for t in (self._tick_task, self._dispatch_task) if t is not None]
        tasks += list(self._streams.values())
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._streams.clear()
        self._tick_task = self._dispatch_task = None
        now = self._now()
        for cfg in list(self.store.config):
            rt = self.store.rt(cfg["id"])
            conn = rt.get("conn") or {}
            if conn.get("up"):
                conn["alive_at"] = _iso(now)
            try:
                await self.store.save_runtime(cfg["id"])
            except Exception:
                logger.warning("sources: could not save %s at stop", cfg["id"], exc_info=True)

    # ================================================================
    # small helpers
    # ================================================================

    def _cfg(self, source_id: str) -> dict:
        cfg = self.store.get(source_id) if safe_source_id(source_id) else None
        if cfg is None:
            raise ApiError(404, {"code": "not_found"})
        return cfg

    def _entry(self, type_: str) -> dict:
        e = self.registry.entry(type_)
        if e is None:
            raise ApiError(404, {"code": "unknown_type"})
        return e

    def _display_name(self, cfg: dict) -> str:
        if cfg.get("name"):
            return cfg["name"]
        if cfg.get("type") == "website":
            host = urlparse((cfg.get("config") or {}).get("url") or "").hostname
            if host:
                return host
        return TYPE_WORDS.get(cfg.get("type"), cfg.get("type") or "")

    def _is_stream(self, cfg: dict) -> bool:
        adapter = self.registry.adapter(cfg["type"])
        return adapter is not None and getattr(adapter, "mode", "poll") == "stream"

    def _lossy(self, cfg: dict) -> bool:
        adapter = self.registry.adapter(cfg["type"])
        if adapter is None or getattr(adapter, "mode", "poll") != "stream":
            return False
        return not getattr(adapter, "lossless", False)

    def _interval(self, cfg: dict) -> float:
        if cfg["type"] == "command":
            return float(max(60, int((cfg.get("command") or {}).get("interval_s") or 300)))
        adapter = self.registry.adapter(cfg["type"])
        iv = getattr(adapter, "interval_s", None) or (self.registry.entry(cfg["type"]) or {}).get("interval_s")
        return float(iv or 300)

    def _runnable(self, cfg: dict) -> bool:
        """Whether the runner may touch the service at all. A registered
        source the user never turned on never runs; a failure that can only
        fail again waits for the user."""
        if cfg.get("is_new"):
            return False
        if self.registry.adapter(cfg["type"]) is None:
            return False
        failure = self.store.rt(cfg["id"]).get("failure")
        return not (failure and not failure.get("retryable", True))

    async def _secrets_for(self, source_id: str, fields: list[str]) -> dict:
        def _read():
            return {f: self.secrets.get(source_id, f) for f in fields}
        return await asyncio.to_thread(_read)

    async def _ctx(self, cfg: dict, **kw) -> SourceContext:
        entry = self.registry.entry(cfg["type"]) or {}
        vals = await self._secrets_for(cfg["id"], catalog.secret_fields(entry))
        rt = self.store.rt(cfg["id"])
        return SourceContext(source_id=cfg["id"], type_=cfg["type"], config=cfg.get("config") or {},
                             kinds=cfg.get("kinds") or {}, account=cfg.get("account"),
                             secrets_fn=vals.get, state=rt.setdefault("state", {}),
                             command=cfg.get("command"), label=cfg.get("label"),
                             lang=self._lang(), **kw)

    def _lang(self) -> str:
        try:
            return self._lang_fn() or "en"
        except Exception:
            return "en"

    async def _persist(self, source_id: str, *, config: bool = False) -> None:
        try:
            if config:
                await self.store.save_config()
            await self.store.save_runtime(source_id)
        except Exception:
            logger.warning("sources: could not save %s", source_id, exc_info=True)

    def _emit(self, payload: dict) -> None:
        try:
            self._broadcast(payload)
        except Exception:
            logger.debug("sources: broadcast failed", exc_info=True)

    def _push_source(self, cfg: dict) -> None:
        self._emit({"type": "sources.changed", "source": self.summary(cfg)})

    async def _push_entry(self, cfg: dict, entry: dict) -> None:
        self._emit({"type": "sources.entry", "source_id": cfg["id"],
                    "entry": await self._entry_json(entry)})

    # ================================================================
    # read shapes
    # ================================================================

    def _failure_json(self, cfg: dict) -> dict | None:
        f = self.store.rt(cfg["id"]).get("failure")
        if not f:
            return None
        entry = self.registry.entry(cfg["type"]) or {}
        method = (entry.get("connect") or {}).get("method")
        code = f.get("code")
        actions = []
        if code in ("cli_signed_out", "cli_missing") or (code == "auth_expired" and method == "cli_account"):
            actions.append("sign_in")
        if code == "calendar_permission":
            actions.append("open_settings")
        if code == "token_rejected" or (code == "auth_expired" and method in ("fields", "qr")):
            actions.append("reconnect")
        if f.get("retryable", True):
            actions.append("check_again")
        actions.append("remove")
        return {"code": code, "params": f.get("params") or {}, "at": f.get("at"),
                "actions": actions}

    def summary(self, cfg: dict) -> dict:
        rt = self.store.rt(cfg["id"])
        reason = self.registry.unavailable_reason(cfg["type"])
        failure = self._failure_json(cfg)
        entries = rt.get("entries") or []
        held = sum(1 for e in entries if e.get("state") == "held")
        waiting = sum(1 for e in entries if e.get("state") in ("waiting", "routing"))
        entry = self.registry.entry(cfg["type"]) or {}
        return {
            "id": cfg["id"], "type": cfg["type"], "family": entry.get("family"),
            "name": cfg.get("name") or "", "display_name": self._display_name(cfg),
            "label": cfg.get("label"), "on": bool(cfg.get("on")),
            "is_new": bool(cfg.get("is_new")),
            "registered_by": self._registered_by_json(cfg.get("registered_by")),
            "status": "unavailable" if reason else ("failed" if failure else "ok"),
            "failure": failure, "last_check_at": rt.get("last_check_at"),
            "waiting": waiting, "held": held,
            "catch_up": ({"count": held} if cfg.get("on") and rt.get("catch_up_offer") and held
                         else None),
            "created_at": cfg.get("created_at"),
        }

    def _registered_by_json(self, rb: dict | None) -> dict | None:
        if not rb:
            return None
        out = dict(rb)
        if out.get("project_id"):
            out["project_name"] = self._project_name(out["project_id"])
        else:
            out.setdefault("project_name", None)
        return out

    async def _session_now(self, entry: dict) -> tuple[str | None, str | None]:
        pid, sid = entry.get("project_id"), entry.get("session_id")
        if not (pid and sid) or self._session_info is None:
            return pid, None
        try:
            new_pid, state = await self._session_info(pid, sid)
        except Exception:
            logger.debug("sources: session info failed", exc_info=True)
            return pid, None
        if new_pid and new_pid != pid:
            entry["project_id"] = new_pid  # moved (spec 180): history follows it
        return new_pid or pid, state

    async def _entry_json(self, e: dict) -> dict:
        if e.get("kind") == "gap":
            return {"id": e["id"], "kind": "gap", "start": e.get("start"), "end": e.get("end")}
        state = e.get("state")
        session_state = None
        pid = e.get("project_id")
        if state == "routed":
            pid, session_state = await self._session_now(e)
        count = int(e.get("count") or 1)
        return {
            "id": e["id"], "kind": "item", "state": state, "title": e.get("title"),
            "sender": e.get("sender"), "place": e.get("place"), "url": e.get("url"),
            "arrived_at": e.get("arrived_at"), "updated_at": e.get("updated_at"),
            "count": count,
            "updates": ([{"title": u.get("title"), "sender": u.get("sender"), "at": u.get("at")}
                         for u in reversed(e.get("parts") or [])][:UPDATES_CAP]
                        if count > 1 else []),
            "project_id": pid, "project_name": self._project_name(pid) if pid else None,
            "routed_by": e.get("routed_by"), "session_id": e.get("session_id"),
            "session_state": session_state, "recovered_at": e.get("recovered_at"),
            "reason": e.get("reason"),
        }

    def _gaps(self, cfg: dict) -> list[dict]:
        if not self._lossy(cfg):
            return []
        rt = self.store.rt(cfg["id"])
        out = [g for g in rt.get("gaps") or []]
        conn = rt.get("conn") or {}
        now = self._now()
        if not conn.get("up") and conn.get("down_since") and not cfg.get("is_new"):
            start = _parse(conn["down_since"])
            if start and (now - start).total_seconds() >= GAP_MIN_S:
                out.append({"id": "gap_open", "kind": "gap", "start": conn["down_since"], "end": None})
        return out

    async def detail(self, source_id: str, *, limit: int = HISTORY_DEFAULT,
                     before: str | None = None) -> dict:
        cfg = self._cfg(source_id)
        limit = max(1, min(HISTORY_MAX, int(limit or HISTORY_DEFAULT)))
        if self._session_refresh is not None:
            try:
                self._session_refresh()
            except Exception:
                logger.debug("sources: session refresh failed", exc_info=True)
        rt = self.store.rt(cfg["id"])
        rows = [dict(e, kind=e.get("kind") or "item") for e in rt.get("entries") or []]
        rows += [dict(g, kind="gap") for g in self._gaps(cfg)]
        rows.sort(key=lambda r: (r.get("arrived_at") or r.get("start") or ""), reverse=True)
        if before:
            idx = next((i for i, r in enumerate(rows) if r.get("id") == before), None)
            rows = rows[idx + 1:] if idx is not None else rows
        page, more = rows[:limit], len(rows) > limit
        by_id = {e["id"]: e for e in rt.get("entries") or []}
        history = []
        for r in page:
            history.append(await self._entry_json(by_id.get(r["id"], r)))
        entry = self.registry.entry(cfg["type"]) or {}
        out = self.summary(cfg)
        out.update({
            "kinds": cfg.get("kinds") or {},
            "account": cfg.get("account"),
            "config": dict(cfg.get("config") or {}),
            "secrets": {f: {"set": f in (cfg.get("secret_fields") or []), "masked": "••••••••••••"}
                        for f in catalog.secret_fields(entry)},
            "command": ({"run": cfg["command"].get("run"), "cwd": cfg["command"].get("cwd"),
                         "interval_s": int(self._interval(cfg))} if cfg.get("command") else None),
            "instruction": cfg.get("instruction") or "",
            "history": history, "history_more": more,
        })
        return out

    def list(self) -> dict:
        rows = [self.summary(c) for c in reversed(self.store.config)]
        waiting = sum(r["waiting"] for r in rows if r["on"])
        return {"sources": rows, "waiting": waiting}

    def catalogue(self, skill_url: str | None = None) -> dict:
        types = self.registry.catalogue()
        if skill_url:
            for t in types:
                t["agent_prompt"] = catalog.with_skill_url(t.get("agent_prompt"), skill_url)
        return {"platform": catalog.current_platform(), "types": types}

    def mark_for_session(self, session_id) -> dict | None:
        """The Workbench card's source mark (thread-safe read)."""
        if not isinstance(session_id, str):
            return None
        hit = self._session_index.get(session_id)
        if hit is None:
            return None
        sid, aid = hit
        cfg = self.store.get(sid)
        if cfg is None:
            return {"source_id": sid, "type": None, "name": None, "arrival_id": aid, "removed": True}
        return {"source_id": sid, "type": cfg["type"], "name": self._display_name(cfg),
                "arrival_id": aid}

    # ================================================================
    # validation
    # ================================================================

    def _validate_kinds(self, entry: dict, kinds) -> dict:
        if kinds is None:
            kinds = catalog.default_kinds(entry)
        if not isinstance(kinds, dict):
            raise ApiError(422, {"code": "invalid", "field": "kinds", "message": "kinds must be an object"})
        spec = {k["key"]: k for k in entry.get("kinds", [])}
        out = {}
        for key, params in kinds.items():
            if key not in spec:
                raise ApiError(422, {"code": "invalid", "field": "kinds",
                                     "message": f"unknown kind {key!r}"})
            params = params if isinstance(params, dict) else {}
            clean = {}
            for p in spec[key].get("params", []):
                v = params.get(p["key"], p.get("default"))
                if v is None or v == "":
                    if p.get("required"):
                        raise ApiError(422, {"code": "invalid", "field": p["key"],
                                             "message": f"{key} needs {p['key']}"})
                    continue
                if p.get("type") == "bool":
                    v = bool(v)
                elif p.get("options"):
                    if v not in p["options"]:
                        try:
                            v = type(p["options"][0])(v)
                        except (TypeError, ValueError):
                            pass
                    if v not in p["options"]:
                        raise ApiError(422, {"code": "invalid", "field": p["key"],
                                             "message": f"{p['key']} must be one of {p['options']}"})
                else:
                    v = framing.clean_line(v, 300)
                clean[p["key"]] = v
            out[key] = clean
        if not out:
            raise ApiError(422, {"code": "invalid", "field": "kinds", "message": "pick at least one kind"})
        return out

    def _validate_config(self, entry: dict, config, *, partial_base: dict | None = None) -> dict:
        config = config if isinstance(config, dict) else {}
        out = dict(partial_base or {})
        for f in catalog.plain_fields(entry):
            if f["key"] in config:
                v = config[f["key"]]
                out[f["key"]] = framing.clean_line(v, 2000) if v is not None else ""
            if not out.get(f["key"]):
                if f.get("default") is not None:
                    out[f["key"]] = f["default"]
                elif not f.get("optional"):
                    raise ApiError(422, {"code": "invalid", "field": f["key"],
                                         "message": f"{f['key']} is required"})
                else:
                    out[f["key"]] = ""
        return out

    def _validate_secrets(self, entry: dict, secrets, *, required: bool) -> dict:
        secrets = secrets if isinstance(secrets, dict) else {}
        out = {}
        for key in catalog.secret_fields(entry):
            v = secrets.get(key)
            v = v.strip() if isinstance(v, str) else ""
            if not v and required:
                raise ApiError(422, {"code": "invalid", "field": key, "message": f"{key} is required"})
            if v:
                out[key] = v
        return out

    @staticmethod
    def _verify_detail(e: VerifyError) -> dict:
        return {"code": "verify_failed", "field": e.field, "reason": e.reason,
                "message": e.message}

    async def _verify(self, adapter, ctx) -> object:
        try:
            return await asyncio.wait_for(adapter.verify(ctx), timeout=POLL_TIMEOUT_S)
        except VerifyError as e:
            raise ApiError(400, self._verify_detail(e))
        except asyncio.TimeoutError:
            raise ApiError(400, {"code": "verify_failed", "field": None, "reason": "unreachable",
                                 "message": "the check timed out"})
        except ApiError:
            raise
        except Exception as e:
            logger.warning("sources: verify of %s failed", ctx.type, exc_info=True)
            raise ApiError(400, {"code": "verify_failed", "field": None, "reason": "error",
                                 "message": type(e).__name__})

    def _validate_command(self, command) -> dict:
        if not isinstance(command, dict) or not isinstance(command.get("run"), str) \
                or not command["run"].strip():
            raise ApiError(422, {"code": "invalid", "field": "command",
                                 "message": "command.run is required"})
        run = command["run"].strip()
        if len(run) > 2000 or "\x00" in run:
            raise ApiError(422, {"code": "invalid", "field": "command", "message": "command.run is too long"})
        cwd = command.get("cwd")
        cwd = cwd.strip() if isinstance(cwd, str) and cwd.strip() else None
        try:
            iv = int(command.get("interval_s") or 300)
        except (TypeError, ValueError):
            raise ApiError(422, {"code": "invalid", "field": "interval_s", "message": "interval_s must be a number"})
        return {"run": run, "cwd": cwd, "interval_s": max(60, min(86400, iv))}

    # ================================================================
    # create / register / change
    # ================================================================

    async def create(self, body: dict, *, registered_by: dict | None = None) -> dict:
        """Verify, then store. Nothing is kept when the check fails."""
        type_ = body.get("type")
        if not isinstance(type_, str):
            raise ApiError(422, {"code": "invalid", "field": "type", "message": "type is required"})
        entry = self._entry(type_)
        external = registered_by is not None
        if type_ == "command" and not external:
            raise ApiError(422, {"code": "invalid", "field": "type",
                                 "message": "check commands are registered through /sources/register"})
        if self.registry.unavailable_reason(type_) is not None:
            raise ApiError(409, {"code": "type_unavailable",
                                 "reason": self.registry.unavailable_reason(type_)})
        adapter = self.registry.adapter(type_)
        method = (entry.get("connect") or {}).get("method")
        name = framing.clean_line(body.get("name"), 80)
        kinds = ({"down": {}} if type_ == "website" else {"item": {}} if type_ == "command"
                 else self._validate_kinds(entry, body.get("kinds")))
        qr = None
        if body.get("qr_id"):
            qr = self._qr.get(body["qr_id"])
            if not qr or qr.get("type") != type_ or qr.get("state") != "done":
                raise ApiError(422, {"code": "invalid", "field": "qr_id", "message": "the QR sign-in is not done"})
        config = self._validate_config(entry, (qr or {}).get("config") or body.get("config"))
        secrets = self._validate_secrets(entry, (qr or {}).get("secrets") or body.get("secrets"),
                                         required=method in ("fields", "qr"))
        account = None
        if method == "cli_account":
            account = body.get("account")
            if not isinstance(account, str) or not account.strip():
                raise ApiError(422, {"code": "invalid", "field": "account", "message": "pick an account"})
            account = account.strip()
        command = self._validate_command(body.get("command")) if type_ == "command" else None
        instruction = framing.clean_body(body.get("instruction"), framing.INSTRUCTION_CAP)
        ctx = SourceContext(source_id=None, type_=type_, config=config, kinds=kinds, account=account,
                            secrets_fn=secrets.get, state={}, command=command)
        verified = await self._verify(adapter, ctx)
        sid = _new_id("src")
        if secrets:
            def _store():
                for k, v in secrets.items():
                    self.secrets.set(sid, k, v)
            try:
                await asyncio.to_thread(_store)
            except Exception as e:
                await asyncio.to_thread(lambda: [self.secrets.delete(sid, k) for k in secrets])
                raise ApiError(500, {"code": "keychain_failed", "message": str(e)})
        now = _iso(self._now())
        cfg = {
            "id": sid, "type": type_, "name": name,
            "on": not external, "is_new": external,
            "kinds": kinds, "account": account,
            "config": {**config, **(getattr(verified, "config", None) or {})},
            "secret_fields": sorted(secrets),
            "label": getattr(verified, "label", None) or None,
            "instruction": instruction, "command": command,
            "registered_by": registered_by, "created_at": now,
        }
        self.store.add(cfg)
        rt = self.store.rt(sid)
        rt["state"] = dict(ctx.state)
        rt["next_check_at"] = now
        if qr:
            self._qr.pop(body["qr_id"], None)
        await self._persist(sid, config=True)
        logger.info("sources: %s %s (%s)", "registered" if external else "created", sid, type_)
        self._push_source(cfg)
        self._wake_tick.set()
        return await self.detail(sid)

    async def register(self, body: dict) -> dict:
        """An agent or a script registers a source: OFF and marked new."""
        rb = body.get("registered_by") if isinstance(body.get("registered_by"), dict) else {}
        pid = rb.get("project_id") if isinstance(rb.get("project_id"), str) and rb.get("project_id") else None
        if pid and self._project_name(pid) is None:
            pid = None
        registered_by = {
            "by": "agent" if pid else "outside",
            "agent": framing.clean_line(rb.get("agent"), 60) or None,
            "project_id": pid,
            "session_id": framing.clean_line(rb.get("session_id"), 120) or None,
            "at": _iso(self._now()),
        }
        return await self.create(body, registered_by=registered_by)

    async def patch(self, source_id: str, body: dict) -> dict:
        cfg = self._cfg(source_id)
        entry = self._entry(cfg["type"])
        rt = self.store.rt(cfg["id"])
        changed_config = False
        reverify = False
        new = dict(cfg)
        if "name" in body:
            new["name"] = framing.clean_line(body.get("name"), 80)
            changed_config = True
        if "instruction" in body:
            new["instruction"] = framing.clean_body(body.get("instruction"), framing.INSTRUCTION_CAP)
            changed_config = True
        if "kinds" in body and cfg["type"] not in ("website", "command"):
            kinds = self._validate_kinds(entry, body.get("kinds"))
            new["kinds"] = kinds
            changed_config = reverify = True
        if "account" in body and (entry.get("connect") or {}).get("method") == "cli_account":
            acc = body.get("account")
            if not isinstance(acc, str) or not acc.strip():
                raise ApiError(422, {"code": "invalid", "field": "account", "message": "pick an account"})
            new["account"] = acc.strip()
            changed_config = reverify = True
        if "config" in body:
            new["config"] = self._validate_config(entry, body.get("config"),
                                                  partial_base=cfg.get("config") or {})
            changed_config = reverify = True
        if reverify and not cfg.get("is_new"):
            adapter = self.registry.adapter(cfg["type"])
            if adapter is not None:
                ctx = await self._ctx(new)
                verified = await self._verify(adapter, ctx)
                if getattr(verified, "label", None):
                    new["label"] = verified.label
                new["config"] = {**(new.get("config") or {}), **(getattr(verified, "config", None) or {})}
                rt["failure"] = None
                rt["fail_count"] = 0
        if "kinds" in new and new.get("kinds") != cfg.get("kinds"):
            # A kind switched on (or re-pointed, e.g. another label) starts
            # from now: its first poll is a baseline again.
            old = cfg.get("kinds") or {}
            rt["baselined"] = [k for k in rt.get("baselined") or []
                               if k in new["kinds"] and new["kinds"].get(k) == old.get(k)]
        if "on" in body:
            on = bool(body.get("on"))
            if on and not cfg.get("on"):
                if any(e.get("state") == "held" for e in rt.get("entries") or []):
                    rt["catch_up_offer"] = True
                if cfg.get("is_new"):
                    new["is_new"] = False
                    rt["next_check_at"] = _iso(self._now())
            new["on"] = on
            changed_config = True
        cfg.clear()
        cfg.update(new)
        if reverify or "on" in body:
            rt["next_check_at"] = rt.get("next_check_at") or _iso(self._now())
        await self._persist(cfg["id"], config=changed_config)
        self._push_source(cfg)
        self._wake_tick.set()
        self._wake_dispatch.set()
        return await self.detail(cfg["id"])

    async def reconnect(self, source_id: str, body: dict) -> dict:
        cfg = self._cfg(source_id)
        entry = self._entry(cfg["type"])
        adapter = self.registry.adapter(cfg["type"])
        if adapter is None:
            raise ApiError(409, {"code": "type_unavailable"})
        qr = None
        if body.get("qr_id"):
            qr = self._qr.get(body["qr_id"])
            if not qr or qr.get("state") != "done":
                raise ApiError(422, {"code": "invalid", "field": "qr_id", "message": "the QR sign-in is not done"})
        config = self._validate_config(entry, (qr or {}).get("config") or body.get("config"),
                                       partial_base=cfg.get("config") or {})
        secrets = self._validate_secrets(entry, (qr or {}).get("secrets") or body.get("secrets"),
                                         required=True)
        ctx = SourceContext(source_id=cfg["id"], type_=cfg["type"], config=config,
                            kinds=cfg.get("kinds") or {}, account=cfg.get("account"),
                            secrets_fn=secrets.get, state=self.store.rt(cfg["id"]).setdefault("state", {}),
                            command=cfg.get("command"))
        verified = await self._verify(adapter, ctx)

        def _store():
            for k, v in secrets.items():
                self.secrets.set(cfg["id"], k, v)
        try:
            await asyncio.to_thread(_store)
        except Exception as e:
            raise ApiError(500, {"code": "keychain_failed", "message": str(e)})
        cfg["config"] = {**config, **(getattr(verified, "config", None) or {})}
        cfg["secret_fields"] = sorted(set(cfg.get("secret_fields") or []) | set(secrets))
        if getattr(verified, "label", None):
            cfg["label"] = verified.label
        rt = self.store.rt(cfg["id"])
        rt["failure"] = None
        rt["fail_count"] = 0
        rt["next_check_at"] = _iso(self._now())
        if qr:
            self._qr.pop(body["qr_id"], None)
        await self._restart_stream(cfg)
        await self._persist(cfg["id"], config=True)
        self._push_source(cfg)
        self._wake_tick.set()
        return await self.detail(cfg["id"])

    async def delete(self, source_id: str) -> dict:
        cfg = self._cfg(source_id)
        task = self._streams.pop(cfg["id"], None)
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        entry = self.registry.entry(cfg["type"]) or {}
        fields = set(cfg.get("secret_fields") or []) | set(catalog.secret_fields(entry))
        await asyncio.to_thread(lambda: [self.secrets.delete(cfg["id"], f) for f in fields])
        self.store.remove(cfg["id"])
        await self.store.save_config()
        await self.store.delete_runtime(cfg["id"])
        logger.info("sources: removed %s (%s)", cfg["id"], cfg["type"])
        self._emit({"type": "sources.removed", "source_id": cfg["id"]})
        return {"status": "deleted", "id": cfg["id"]}

    async def catch_up(self, source_id: str) -> dict:
        cfg = self._cfg(source_id)
        rt = self.store.rt(cfg["id"])
        held = [e for e in rt.get("entries") or [] if e.get("state") == "held"]
        held.sort(key=lambda e: e.get("arrived_at") or "")
        for e in held:
            e["state"] = "waiting"
            e["queued_at"] = _iso(self._now())
        rt["catch_up_offer"] = False
        await self._persist(cfg["id"])
        for e in held:
            await self._push_entry(cfg, e)
        self._push_source(cfg)
        self._wake_dispatch.set()
        return await self.detail(cfg["id"])

    async def dismiss_held(self, source_id: str) -> dict:
        cfg = self._cfg(source_id)
        rt = self.store.rt(cfg["id"])
        held = [e for e in rt.get("entries") or [] if e.get("state") == "held"]
        for e in held:
            e["state"] = "dismissed"
            e.pop("parts_body", None)
        rt["catch_up_offer"] = False
        await self._persist(cfg["id"])
        for e in held:
            await self._push_entry(cfg, e)
        self._push_source(cfg)
        return await self.detail(cfg["id"])

    async def route_arrival(self, source_id: str, arrival_id: str, project_id: str) -> dict:
        cfg = self._cfg(source_id)
        rt = self.store.rt(cfg["id"])
        e = next((x for x in rt.get("entries") or [] if x.get("id") == arrival_id), None)
        if e is None:
            raise ApiError(404, {"code": "not_found"})
        if e.get("state") != "unrouted":
            raise ApiError(409, {"code": "not_unrouted"})
        if not isinstance(project_id, str) or not project_id:
            raise ApiError(422, {"code": "invalid", "field": "project_id", "message": "project_id is required"})
        e["state"] = "routing"
        await self._push_entry(cfg, e)
        try:
            result = await self._send(self._task_text(cfg, e), project_id)
        except SendError as err:
            e["state"] = "unrouted"
            e["reason"] = err.reason
            await self._persist(cfg["id"])
            await self._push_entry(cfg, e)
            raise ApiError(err.status, err.detail)
        self._mark_routed(cfg, e, result, by="user")
        await self._persist(cfg["id"])
        await self._push_entry(cfg, e)
        self._push_source(cfg)
        return await self._entry_json(e)

    # ================================================================
    # checks
    # ================================================================

    async def check(self, source_id: str) -> dict:
        """「再检查」: the same check the runner does on its interval, now."""
        cfg = self._cfg(source_id)
        if cfg.get("is_new"):
            # Registered from outside and never switched on: switching it on
            # in the panel is the only way it first runs.
            raise ApiError(409, {"code": "not_on_yet"})
        adapter = self.registry.adapter(cfg["type"])
        if adapter is None:
            raise ApiError(409, {"code": "type_unavailable"})
        rt = self.store.rt(cfg["id"])
        if getattr(adapter, "mode", "poll") == "stream":
            ctx = await self._ctx(cfg)
            try:
                verified = await asyncio.wait_for(adapter.verify(ctx), timeout=POLL_TIMEOUT_S)
            except VerifyError as e:
                code, retry = _REASON_FAILURE.get(e.reason, ("error", True))
                self._set_failure(cfg, SourceFailure(code, {"field": e.field} if e.field else {},
                                                     retryable=retry))
            except Exception:
                logger.warning("sources: check of %s failed", cfg["id"], exc_info=True)
                self._set_failure(cfg, SourceFailure("error"))
            else:
                if getattr(verified, "label", None):
                    cfg["label"] = verified.label
                rt["failure"] = None
                rt["fail_count"] = 0
                await self._restart_stream(cfg)
            await self._persist(cfg["id"], config=True)
            self._push_source(cfg)
        else:
            rt["failure"] = (rt.get("failure") or None)
            await self._poll_one(cfg, manual=True)
        return await self.detail(cfg["id"])

    def _set_failure(self, cfg: dict, f: SourceFailure) -> None:
        rt = self.store.rt(cfg["id"])
        prev = rt.get("failure")
        rt["failure"] = {"code": f.code, "params": f.params, "retryable": f.retryable,
                         "at": (prev or {}).get("at") if prev and prev.get("code") == f.code
                         else _iso(self._now())}
        rt["fail_count"] = int(rt.get("fail_count") or 0) + 1
        if not prev or prev.get("code") != f.code:
            logger.info("sources: %s failed: %s", cfg["id"], f.code)

    async def _poll_one(self, cfg: dict, *, manual: bool = False) -> None:
        sid = cfg["id"]
        lock = self._poll_locks.setdefault(sid, asyncio.Lock())
        async with lock:
            if self.store.get(sid) is None:
                return
            adapter = self.registry.adapter(cfg["type"])
            if adapter is None:
                return
            rt = self.store.rt(sid)
            now = self._now()
            interval = self._interval(cfg)
            try:
                ctx = await self._ctx(cfg)
                events = await asyncio.wait_for(adapter.poll(ctx), timeout=POLL_TIMEOUT_S)
            except asyncio.CancelledError:
                raise
            except SourceFailure as f:
                self._set_failure(cfg, f)
                backoff = min(MAX_BACKOFF_S, interval * (2 ** min(6, rt["fail_count"] - 1)))
                rt["next_check_at"] = _iso(now + timedelta(seconds=backoff))
            except asyncio.TimeoutError:
                self._set_failure(cfg, SourceFailure("error", {"timeout": True}))
                rt["next_check_at"] = _iso(now + timedelta(seconds=interval))
            except Exception as e:
                logger.warning("sources: poll of %s (%s) failed: %s", sid, cfg["type"],
                               type(e).__name__, exc_info=True)
                self._set_failure(cfg, SourceFailure("error"))
                backoff = min(MAX_BACKOFF_S, interval * (2 ** min(6, rt["fail_count"] - 1)))
                rt["next_check_at"] = _iso(now + timedelta(seconds=backoff))
            else:
                if rt.get("failure"):
                    logger.info("sources: %s recovered", sid)
                rt["failure"] = None
                rt["fail_count"] = 0
                rt["state"] = ctx.state
                await self._ingest(cfg, events or [], poll=True, persist=False)
                rt["last_check_at"] = _iso(self._now())
                jitter = random.uniform(-0.1, 0.1) * interval
                rt["next_check_at"] = _iso(self._now() + timedelta(seconds=interval + jitter))
            await self._persist(sid)
            self._push_source(cfg)

    # ================================================================
    # the arrival pipeline
    # ================================================================

    async def _is_open(self, e: dict) -> bool:
        state = e.get("state")
        if state in ACTIVE_STATES:
            return True
        if state != "routed" or e.get("recovered_at"):
            return False
        _pid, sstate = await self._session_now(e)
        return sstate in ("running", "needs_you")

    async def ingest(self, source_id: str, events: list) -> None:
        """Hand arrivals to the pipeline (streams use this via ``ctx.emit``)."""
        cfg = self.store.get(source_id)
        if cfg is None:
            return
        await self._ingest(cfg, events, poll=False, persist=True)

    async def _ingest(self, cfg: dict, events: list, *, poll: bool, persist: bool) -> None:
        sid = cfg["id"]
        rt = self.store.rt(sid)
        adapter = self.registry.adapter(cfg["type"])
        baselining = poll and getattr(adapter, "baseline", True)
        baselined = set(rt.get("baselined") or [])
        kinds = cfg.get("kinds") or {}
        seen = rt.setdefault("seen", {})
        entries = rt.setdefault("entries", [])
        now = _iso(self._now())
        touched: list[dict] = []
        new_waiting = False
        for ev in events:
            if isinstance(ev, Resolve):
                subj = str(ev.subject)
                for e in reversed(entries):
                    if e.get("subject") == subj and not e.get("recovered_at"):
                        e["recovered_at"] = ev.at or now
                        e["updated_at"] = now
                        touched.append(e)
                        break
                continue
            if not isinstance(ev, Item):
                continue
            item_id = framing.clean_line(ev.id, 300)
            if not item_id or ev.kind not in kinds:
                continue
            if item_id in seen:
                continue
            seen[item_id] = now
            if baselining and ev.kind not in baselined:
                continue  # already there when the kind was switched on
            subject = framing.clean_line(ev.subject, 300) or item_id
            part = {"title": framing.clean_title(ev.title), "sender": framing.clean_line(ev.sender, 100) or None,
                    "place": ev.place if isinstance(ev.place, dict) else None,
                    "body": framing.clean_body(ev.body), "url": framing.clean_url(ev.url),
                    "at": ev.at or now}
            target = None
            for e in reversed(entries):
                if e.get("subject") == subject and e.get("kind", "item") == "item":
                    target = e if await self._is_open(e) else None
                    break
            if target is not None:
                parts = target.setdefault("parts", [])
                if target.get("state") in ("waiting", "held", "unrouted"):
                    parts.append(part)
                else:
                    # Already sent: the update is recorded on the entry.
                    parts.append({k: v for k, v in part.items() if k != "body"})
                del parts[:-UPDATES_CAP]
                target["count"] = int(target.get("count") or 1) + 1
                target["updated_at"] = now
                touched.append(target)
                continue
            state = "waiting" if cfg.get("on") else "held"
            e = {"id": _new_id("arr"), "kind": "item", "subject": subject, "item_kind": ev.kind,
                 "state": state, "title": part["title"], "sender": part["sender"],
                 "place": part["place"], "url": part["url"], "arrived_at": now,
                 "updated_at": now, "count": 1, "parts": [part], "project_id": None,
                 "routed_by": None, "session_id": None, "recovered_at": None, "reason": None}
            entries.append(e)
            touched.append(e)
            new_waiting = new_waiting or state == "waiting"
        if poll:
            rt["baselined"] = sorted(set(kinds))
        while len(seen) > SEEN_CAP:
            seen.pop(next(iter(seen)))
        self._trim_entries(rt)
        if persist:
            await self._persist(sid)
        seen_ids = set()
        for e in touched:
            if e["id"] not in seen_ids:
                seen_ids.add(e["id"])
                await self._push_entry(cfg, e)
        if touched:
            self._push_source(cfg)
        if new_waiting:
            self._wake_dispatch.set()

    @staticmethod
    def _trim_entries(rt: dict) -> None:
        entries = rt.get("entries") or []
        if len(entries) <= ENTRY_CAP:
            return
        extra = len(entries) - ENTRY_CAP
        keep = []
        for e in entries:
            if extra > 0 and e.get("state") not in ("waiting", "routing", "held"):
                extra -= 1
                continue
            keep.append(e)
        rt["entries"] = keep

    # ================================================================
    # dispatcher: one arrival at a time
    # ================================================================

    def _task_text(self, cfg: dict, e: dict) -> str:
        return framing.task_text(
            title=e.get("title") or "", source_name=cfg.get("name") or "",
            type_word=TYPE_WORDS.get(cfg["type"], cfg["type"]),
            label=cfg.get("label"), kind=e.get("item_kind") or "item", url=e.get("url"),
            instruction=cfg.get("instruction") or "", parts=e.get("parts") or [])

    def _mark_routed(self, cfg: dict, e: dict, result: dict, *, by: str) -> None:
        e["state"] = "routed"
        e["reason"] = None
        e["project_id"] = result.get("project_id")
        e["session_id"] = result.get("session_id")
        rb = result.get("by")
        e["routed_by"] = by if by == "user" else ("fallback" if rb == "fallback" else "model")
        e["routed_at"] = _iso(self._now())
        for p in e.get("parts") or []:
            p.pop("body", None)  # sent: the session has it now
        if e["session_id"]:
            self._session_index[e["session_id"]] = (cfg["id"], e["id"])
        self.store.rt(cfg["id"])["active"] = {"project_id": e["project_id"],
                                              "session_id": e["session_id"]}

    async def _source_busy(self, cfg: dict) -> bool:
        rt = self.store.rt(cfg["id"])
        if any(e.get("state") == "routing" for e in rt.get("entries") or []):
            return True
        active = rt.get("active")
        if not active or not active.get("session_id"):
            return False
        probe = {"project_id": active.get("project_id"), "session_id": active.get("session_id")}
        _pid, state = await self._session_now(probe)
        if probe["project_id"] != active.get("project_id"):
            active["project_id"] = probe["project_id"]
        return state == "running"

    async def _next_to_send(self):
        candidates = []
        for cfg in self.store.config:
            if not cfg.get("on"):
                continue
            for e in self.store.rt(cfg["id"]).get("entries") or []:
                if e.get("state") == "waiting":
                    candidates.append((e.get("queued_at") or e.get("arrived_at") or "", cfg, e))
                    break  # oldest waiting of this source (entries are oldest first)
        candidates.sort(key=lambda c: c[0])
        for _key, cfg, e in candidates:
            if not await self._source_busy(cfg):
                return cfg, e
        return None

    async def dispatch_once(self) -> bool:
        """Send the next sendable arrival, if any. True when one was sent
        (or failed); the loop calls this; tests call it directly."""
        if self._send is None:
            return False
        nxt = await self._next_to_send()
        if nxt is None:
            return False
        cfg, e = nxt
        e["state"] = "routing"
        await self._persist(cfg["id"])
        await self._push_entry(cfg, e)
        self._push_source(cfg)
        try:
            result = await self._send(self._task_text(cfg, e), None)
        except SendError as err:
            e["state"] = "unrouted"
            e["reason"] = err.reason
            logger.info("sources: %s arrival %s not sent (%s)", cfg["id"], e["id"], err.reason)
        except Exception:
            logger.warning("sources: sending %s failed", e["id"], exc_info=True)
            e["state"] = "unrouted"
            e["reason"] = "send_failed"
        else:
            self._mark_routed(cfg, e, result or {}, by="auto")
            logger.info("sources: %s arrival %s -> %s/%s (%s)", cfg["id"], e["id"],
                        e.get("project_id"), e.get("session_id"), e.get("routed_by"))
        if self.store.get(cfg["id"]) is not None:
            await self._persist(cfg["id"])
            await self._push_entry(cfg, e)
            self._push_source(cfg)
        return True

    async def _dispatch_loop(self) -> None:
        while self._running:
            try:
                sent = await self.dispatch_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("sources: dispatcher error", exc_info=True)
                sent = False
            if sent:
                continue
            self._wake_dispatch.clear()
            try:
                await asyncio.wait_for(self._wake_dispatch.wait(), timeout=DISPATCH_IDLE_S)
            except asyncio.TimeoutError:
                pass

    # ================================================================
    # runner: polls on a wall-clock tick, streams supervised
    # ================================================================

    def _due(self, cfg: dict, now: datetime) -> bool:
        nxt = _parse(self.store.rt(cfg["id"]).get("next_check_at"))
        return nxt is None or nxt <= now

    async def run_due_polls(self) -> None:
        now = self._now()
        due = [c for c in list(self.store.config)
               if self._runnable(c) and not self._is_stream(c) and self._due(c, now)
               and c["id"] not in self._polls_inflight]

        async def _one(cfg):
            self._polls_inflight.add(cfg["id"])
            try:
                async with self._poll_sem:
                    await self._poll_one(cfg)
            finally:
                self._polls_inflight.discard(cfg["id"])
        if due:
            await asyncio.gather(*[_one(c) for c in due], return_exceptions=True)

    async def supervise_streams(self) -> None:
        live_ids = set()
        for cfg in list(self.store.config):
            if not self._is_stream(cfg) or not self._runnable(cfg):
                continue
            live_ids.add(cfg["id"])
            task = self._streams.get(cfg["id"])
            if task is None or task.done():
                self._streams[cfg["id"]] = asyncio.create_task(self._stream_runner(cfg["id"]))
        for sid in list(self._streams):
            if sid not in live_ids:
                t = self._streams.pop(sid)
                t.cancel()
        # heartbeat: what a gap is measured from if the daemon stops
        now = self._now()
        for sid in live_ids:
            conn = self.store.rt(sid).get("conn") or {}
            if conn.get("up"):
                conn["alive_at"] = _iso(now)
                last = self._alive_saved.get(sid)
                if last is None or (now - last).total_seconds() >= ALIVE_PERSIST_S:
                    self._alive_saved[sid] = now
                    await self._persist(sid)

    async def _restart_stream(self, cfg: dict) -> None:
        t = self._streams.pop(cfg["id"], None)
        if t is not None:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        if self._running and self._is_stream(cfg) and self._runnable(cfg):
            self._streams[cfg["id"]] = asyncio.create_task(self._stream_runner(cfg["id"]))

    def _on_connected(self, sid: str) -> None:
        cfg = self.store.get(sid)
        if cfg is None:
            return
        rt = self.store.rt(sid)
        conn = rt.setdefault("conn", {})
        now = self._now()
        start = _parse(conn.get("down_since"))
        if start and self._lossy(cfg) and (now - start).total_seconds() >= GAP_MIN_S:
            gap = {"id": _new_id("gap"), "start": conn["down_since"], "end": _iso(now)}
            rt.setdefault("gaps", []).append(gap)
            del rt["gaps"][:-200]
            self._emit({"type": "sources.entry", "source_id": sid,
                        "entry": {"id": gap["id"], "kind": "gap", "start": gap["start"], "end": gap["end"]}})
        conn.update({"up": True, "down_since": None, "alive_at": _iso(now)})
        rt["failure"] = None
        rt["fail_count"] = 0
        rt["last_check_at"] = _iso(now)
        asyncio.ensure_future(self._persist(sid))
        self._push_source(cfg)

    def _on_disconnected(self, sid: str) -> None:
        rt = self.store.runtime.get(sid)
        if rt is None:
            return
        conn = rt.setdefault("conn", {})
        if conn.get("up") or not conn.get("down_since"):
            conn["down_since"] = _iso(self._now())
        conn["up"] = False

    async def _stream_runner(self, sid: str) -> None:
        backoff = 1.0
        while self._running:
            cfg = self.store.get(sid)
            if cfg is None or not self._runnable(cfg):
                return
            adapter = self.registry.adapter(cfg["type"])
            began = self._now()
            try:
                ctx = await self._ctx(cfg, emit_fn=lambda evs, _sid=sid: self.ingest(_sid, evs),
                                      connected_fn=lambda _sid=sid: self._on_connected(_sid))
                await adapter.stream(ctx)
                quick_fail = None
            except asyncio.CancelledError:
                self._on_disconnected(sid)
                raise
            except SourceFailure as f:
                self._on_disconnected(sid)
                self._set_failure(cfg, f)
                await self._persist(sid)
                self._push_source(cfg)
                if not f.retryable:
                    return
                quick_fail = f
            except Exception as e:
                logger.info("sources: stream %s ended: %s", sid, type(e).__name__)
                quick_fail = e
            self._on_disconnected(sid)
            lasted = (self._now() - began).total_seconds()
            backoff = 1.0 if lasted > 60 else min(STREAM_MAX_BACKOFF_S, backoff * 2)
            rt = self.store.rt(sid)
            if quick_fail is not None and not isinstance(quick_fail, SourceFailure):
                rt["fail_count"] = int(rt.get("fail_count") or 0) + 1
                if rt["fail_count"] >= 3 and not rt.get("failure"):
                    self._set_failure(cfg, SourceFailure("unreachable", {}))
                    self._push_source(cfg)
            await self._persist(sid)
            await asyncio.sleep(backoff)

    async def _tick_loop(self) -> None:
        if self.first_tick_delay_s:
            try:
                await asyncio.sleep(self.first_tick_delay_s)
            except asyncio.CancelledError:
                return
        while self._running:
            try:
                await self.supervise_streams()
                await self.run_due_polls()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("sources: tick failed", exc_info=True)
            self._wake_tick.clear()
            try:
                await asyncio.wait_for(self._wake_tick.wait(), timeout=self.tick_s)
            except asyncio.TimeoutError:
                pass

    async def command_path(self, source_id: str) -> str | None:
        """The script a check command runs: the first word of ``command.run``
        that is an existing path (``~`` expanded, quotes stripped), else
        its ``cwd``; None when there is neither."""
        import os
        import shlex
        cfg = self._cfg(source_id)
        cmd = cfg.get("command") or {}
        if cfg.get("type") != "command" or not cmd:
            return None
        cwd = os.path.expanduser(cmd.get("cwd") or "~")
        try:
            words = shlex.split(cmd.get("run") or "", posix=True)
        except ValueError:
            words = (cmd.get("run") or "").split()

        def _find():
            for w in words:
                w = os.path.expanduser(w.strip("'\""))
                if not w or w.startswith("-"):
                    continue
                p = w if os.path.isabs(w) else os.path.join(cwd, w)
                if (os.sep in w or "/" in w) and os.path.exists(p):
                    return os.path.realpath(p)
            return os.path.realpath(cwd) if cmd.get("cwd") and os.path.isdir(cwd) else None
        return await asyncio.to_thread(_find)

    # ================================================================
    # accounts, choices, QR
    # ================================================================

    async def accounts(self, type_: str) -> dict:
        entry = self._entry(type_)
        connect = entry.get("connect") or {}
        if connect.get("method") != "cli_account":
            raise ApiError(422, {"code": "invalid", "field": "type", "message": "not a CLI type"})
        adapter = self.registry.adapter(type_)
        if adapter is None:
            raise ApiError(409, {"code": "type_unavailable"})
        acc = await adapter.accounts()
        return {"type": type_, "cli": connect.get("cli"), "state": acc.state,
                "accounts": acc.accounts, "sign_in_command": connect.get("sign_in_command"),
                "install_command": connect.get("install_command")}

    async def choices(self, body: dict) -> dict:
        list_name = body.get("list")
        params = body.get("params") if isinstance(body.get("params"), dict) else {}
        if body.get("source_id"):
            cfg = self._cfg(body["source_id"])
            adapter = self.registry.adapter(cfg["type"])
            if adapter is None:
                raise ApiError(409, {"code": "type_unavailable"})
            ctx = await self._ctx(cfg)
        else:
            type_ = body.get("type")
            entry = self._entry(type_ if isinstance(type_, str) else "")
            adapter = self.registry.adapter(type_)
            if adapter is None:
                raise ApiError(409, {"code": "type_unavailable"})
            secrets = self._validate_secrets(entry, body.get("secrets"), required=False)
            config = body.get("config") if isinstance(body.get("config"), dict) else {}
            ctx = SourceContext(source_id=None, type_=type_, config=config, kinds={},
                                account=body.get("account"), secrets_fn=secrets.get, state={})
        try:
            items = await asyncio.wait_for(adapter.choices(ctx, list_name, params), timeout=POLL_TIMEOUT_S)
        except VerifyError as e:
            raise ApiError(400, self._verify_detail(e))
        except asyncio.TimeoutError:
            raise ApiError(400, {"code": "verify_failed", "field": None, "reason": "unreachable",
                                 "message": "timed out"})
        return {"items": [c.to_json() for c in items or []]}

    async def qr_start(self, type_: str) -> dict:
        entry = self._entry(type_ if isinstance(type_, str) else "")
        if (entry.get("connect") or {}).get("method") != "qr":
            raise ApiError(422, {"code": "invalid", "field": "type", "message": "no QR sign-in for this type"})
        adapter = self.registry.adapter(type_)
        if adapter is None:
            raise ApiError(409, {"code": "type_unavailable"})
        ctx = SourceContext(source_id=None, type_=type_, config={}, kinds={}, account=None,
                            secrets_fn=None, state={})
        try:
            started = await adapter.qr_start(ctx)
        except NotImplementedError:
            raise ApiError(409, {"code": "type_unavailable"})
        qr_id = _new_id("qr")
        self._qr[qr_id] = {"type": type_, "handle": started.get("handle"), "state": "pending",
                           "label": None, "config": None, "secrets": None}
        return {"qr_id": qr_id, "state": "pending", "qr_text": started.get("qr_text"),
                "expires_at": started.get("expires_at")}

    async def qr_get(self, qr_id: str) -> dict:
        q = self._qr.get(qr_id)
        if q is None:
            raise ApiError(404, {"code": "not_found"})
        if q["state"] == "pending":
            adapter = self.registry.adapter(q["type"])
            ctx = SourceContext(source_id=None, type_=q["type"], config={}, kinds={}, account=None,
                                secrets_fn=None, state={})
            try:
                res = await adapter.qr_poll(ctx, q["handle"])
            except Exception:
                logger.warning("sources: QR poll failed", exc_info=True)
                res = {"state": "failed"}
            state = res.get("state") or "failed"
            if state != q["state"]:
                q["state"] = state
                self._emit({"type": "sources.qr", "qr_id": qr_id, "state": state})
            if state == "done":
                q.update(label=res.get("label"), config=res.get("config") or {},
                         secrets=res.get("secrets") or {})
        return {"qr_id": qr_id, "state": q["state"], "label": q.get("label")}


# The installed core, for the Workbench feed's source mark (read from a
# worker thread; plain dict reads).
_installed: SourcesCore | None = None


def install(core: SourcesCore | None) -> None:
    global _installed
    _installed = core


def mark_for_session(session_id) -> dict | None:
    core = _installed
    if core is None:
        return None
    try:
        return core.mark_for_session(session_id)
    except Exception:
        return None
