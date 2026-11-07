# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Worker-run onboarding for a keyless project.

When Orbital's own agent has no key (``AgentManager.manager_ready`` false),
the first-run onboarding — the interview, or the cold-start workspace scan —
runs on the project's pinned worker instead of failing with
``missing_api_key``. The worker gets the same text the management agent
would (``agent_os.agent.onboarding_prompts``) and answers through the
ordinary pinned direct-send path, so its reply lands as today's
``sub_agent_terminal`` completed row: the first bubble of the session.

Mirrors ``TriggerManager._fire_on_worker``: mint the session, build the
Session object, put a row on disk BEFORE the dispatch (the terminal event
hydrates the session from disk and is dropped when no JSONL matches), then
``SubAgentManager.send``. The row here is a daemon system row, never a user
row — the user did not type the instruction.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from agent_os.agent import onboarding_prompts
from agent_os.daemon_v2.provider_errors import ProviderConfigError

logger = logging.getLogger(__name__)


class WorkerDispatchError(RuntimeError):
    """The worker refused the onboarding dispatch (not signed in, too old…).

    The session exists and carries the failure row; ``session_id`` names it.
    """

    def __init__(self, message: str, session_id: str):
        super().__init__(message)
        self.session_id = session_id


def build_instruction(slug: str, project: dict, mode: str,
                      skeleton_text: str | None = None, *,
                      guess_lang: str | None = None,
                      multi_agent: bool = False,
                      language_signals: dict | None = None) -> str:
    """The onboarding instruction ``slug`` receives for ``mode``.
    ``guess_lang`` (spec 160): the import scan's guess-first flavour, in that
    reply language; None keeps the card's propose-first scan.
    ``language_signals``: the user's language signals; given, the
    instruction carries the ``## LANGUAGE`` section and the worker picks the
    language it speaks."""
    from agent_os.onboarding.user_language import language_section

    workspace = project.get("workspace", "")
    if mode == "scan" and guess_lang is not None:
        return onboarding_prompts.worker_guess_first_instruction(
            slug, workspace, skeleton_text or "", lang=guess_lang,
            multi_agent=multi_agent,
            language=(language_section(language_signals, import_lang=guess_lang)
                      if language_signals else None))
    tail = f"\n\n{language_section(language_signals)}" if language_signals else ""
    if mode == "scan":
        return onboarding_prompts.worker_cold_start_instruction(
            slug, workspace, skeleton_text or "") + tail
    return onboarding_prompts.worker_interview_instruction(
        slug, workspace,
        project_name=project.get("name", "") or "",
        project_instructions=project.get("instructions", "") or "",
        is_scratch=bool(project.get("is_scratch")),
    ) + tail


def _language_signals(agent_manager) -> dict | None:
    """The user's language signals from the manager's settings store."""
    try:
        from agent_os.onboarding.user_language import signals_from_settings
        store = getattr(agent_manager, "_settings_store", None)
        return signals_from_settings(store.get() if store is not None else None)
    except Exception:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def start_worker_onboarding(agent_manager, sub_agent_manager,
                                  project_id: str, slug: str, *, mode: str,
                                  skeleton_text: str | None = None,
                                  ws_manager=None, guess_lang: str | None = None,
                                  multi_agent: bool = False,
                                  title: str | None = None) -> dict:
    """Mint a session pinned to ``slug`` and hand it the onboarding.

    ``mode`` is ``"interview"`` or ``"scan"``; ``guess_lang`` /
    ``multi_agent`` / ``title`` are the spec 160 import scan's. Returns the
    route's 201 body.
    Raises ``WorkerDispatchError`` when ``send`` refuses the dispatch (the
    session then holds a ``sub_agent_dispatch_failed`` row, the same row the
    pinned inject route writes), and ``RuntimeError`` when no sub-agent
    manager is wired.
    """
    from agent_os.agent.session import Session

    if sub_agent_manager is None:
        raise RuntimeError(f"no sub-agent manager available to run '{slug}'")
    project = agent_manager._project_store.get_project(project_id) or {}
    instruction = build_instruction(slug, project, mode, skeleton_text,
                                    guess_lang=guess_lang, multi_agent=multi_agent,
                                    language_signals=_language_signals(agent_manager))

    minted = await agent_manager.new_session(project_id)
    session_id = minted["session_id"]
    try:
        config = await agent_manager._abuild_agent_config_from_project(project_id)
    except ProviderConfigError:
        # A card reference that no longer resolves: the manager could not
        # start here either, but the worker needs no card — record the
        # session without provider metadata.
        config = None
    if config is not None:
        session = agent_manager._new_session_object(
            config, session_id, cold_start=(mode == "scan"))
    else:
        session = Session.new(
            session_id, project.get("workspace", ""), session_id=session_id,
            origin="cold_start" if mode == "scan" else "chat",
        )
    agent_manager._wire_session_observers(session, project_id, session_id)
    if title:
        # Spec 160 round 3: the import scan's session keeps "项目梳理" (a
        # final name; the user's first reply does not retitle it).
        session.set_name(title)
    session.set_pinned_target(slug)
    session.append({
        "role": "system",
        "source": "daemon",
        "content": f"[Onboarding] {slug} is setting up this project",
        "timestamp": _now(),
        "_meta": {
            "event": "worker_onboarding",
            "handle": slug,
            "mode": mode,
            "suppress_wake": True,
        },
    })

    result = await sub_agent_manager.send(
        project_id, slug, instruction,
        session_id=session_id, initiator="user_pinned",
    )
    if isinstance(result, str) and result.startswith("Error"):
        logger.warning("Worker onboarding dispatch to %s failed for %s/%s: %s",
                       slug, project_id, session_id, result)
        failure_meta = {
            "event": "sub_agent_dispatch_failed",
            "handle": slug,
            "suppress_wake": True,
        }
        meta_reader = getattr(sub_agent_manager, "last_start_failure_meta", None)
        start_meta = (meta_reader(project_id, slug) or {}) if callable(meta_reader) else {}
        if isinstance(start_meta, dict):
            failure_meta.update({
                k: start_meta[k] for k in ("reason", "min_version", "version")
                if start_meta.get(k) is not None
            })
        session.append({
            "role": "system",
            "source": "daemon",
            "content": f"[Sub-agent] {slug} could not start: {result}",
            "timestamp": _now(),
            "_meta": failure_meta,
        })
        if ws_manager is not None:
            ws_manager.broadcast(project_id, {
                "type": "sub_agent.failed",
                "project_id": project_id,
                "handle": slug,
                "reason": result,
                "session_id": session_id,
            })
        raise WorkerDispatchError(result, session_id)

    logger.info("Onboarding (%s) dispatched directly to '%s' (session=%s)",
                mode, slug, session_id)
    return {
        "status": "ok",
        "session_id": session_id,
        "session_uuid": minted.get("session_uuid", session_id),
        "runner": slug,
    }
