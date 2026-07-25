// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * ScopeChip — the Quick Tasks cross-project read-scope control (Spec 012 §2c).
 *
 * Renders ONLY for the scratch ("Quick Tasks") project and is always visible
 * in the chat header area — its permanent presence *is* the reminder of how
 * Quick Tasks works. The chip label reflects the ACTIVE session's scope:
 *
 *   Access: All projects ▾    (mode 'all')
 *   Access: 2 projects ▾      (mode 'selected')
 *   Access: This workspace ▾  (mode 'off')
 *
 * Opening the chip shows a popover with the master toggle ("Read all
 * projects"), a project multi-select (the already-loaded App-level project
 * list from props, excluding the scratch project itself), and a fixed
 * footnote: writes always stay in the Quick Tasks workspace.
 *
 * Scope is PER-SESSION: the chip re-fetches whenever `sessionId` changes.
 * Updates are optimistic — the chip flips immediately, the PUT runs in the
 * background, and a failure reverts the chip and shows a transient error
 * notice. Backend contract (locked, Task A3):
 *   GET/PUT /api/v2/agents/{project_id}/sessions/{session_id}/scope
 *   body/response: { mode: 'all'|'selected'|'off', selected_project_ids: [] }
 *   PUT returns the stored record; 409 when the project is not scratch.
 *
 * New sessions (and sessions the backend hasn't materialized yet, whose GET
 * fails) display the backend default: all projects ON.
 */

import { useEffect, useRef, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { api } from '../config';
import type { Project, SessionScope } from '../types';
import { useT } from '../i18n/useT';

export interface ScopeChipProps {
  /** The project whose chat is being viewed. The chip renders only when
   *  `project.is_scratch` is true. */
  project: Project;
  /** The App-level projects list (already loaded by ChatTab's parent). The
   *  scratch project itself is excluded from the multi-select. */
  projects: Project[];
  /** Active session id (route.sessionId). Scope is per-session; the chip
   *  re-fetches when this changes. Undefined (no session yet) shows the
   *  backend default and disables edits (there is nothing to PUT against). */
  sessionId?: string;
}

/** Backend default for new scratch sessions: all projects ON. */
const DEFAULT_SCOPE: SessionScope = { mode: 'all', selected_project_ids: [] };

function scopeUrl(projectId: string, sessionId: string): string {
  return `/api/v2/agents/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}/scope`;
}

/** Defensive shape check — tolerate a bad/partial payload without crashing. */
function isScopeRecord(value: unknown): value is SessionScope {
  if (!value || typeof value !== 'object') return false;
  const v = value as Record<string, unknown>;
  return (
    (v.mode === 'all' || v.mode === 'selected' || v.mode === 'off') &&
    Array.isArray(v.selected_project_ids)
  );
}

export default function ScopeChip({ project, projects, sessionId }: ScopeChipProps) {
  const t = useT();
  const projectId = project.project_id;

  const [scope, setScope] = useState<SessionScope>(DEFAULT_SCOPE);
  const [open, setOpen] = useState(false);
  const [errorVisible, setErrorVisible] = useState(false);

  const btnRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
  const errorTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Monotonic write epoch: a stale PUT's response/revert must not clobber the
  // optimistic state of a newer toggle. (Plain ref bookkeeping — deliberately
  // NOT a closure var mutated inside a setState updater; see CLAUDE.md.)
  const writeEpochRef = useRef(0);

  // Per-session fetch: re-read the scope whenever the active session changes.
  useEffect(() => {
    if (!project.is_scratch) return;
    // No session yet (blank project / still resolving): show the backend
    // default rather than a stale record from the previous session.
    setScope(DEFAULT_SCOPE);
    if (!sessionId) return;
    let cancelled = false;
    api<SessionScope>(scopeUrl(projectId, sessionId))
      .then((record) => {
        if (!cancelled && isScopeRecord(record)) setScope(record);
      })
      .catch(() => {
        // New sessions may not be materialized server-side yet — the backend
        // default (all) is the honest display; nothing to surface to the user.
        if (!cancelled) setScope(DEFAULT_SCOPE);
      });
    return () => {
      cancelled = true;
    };
  }, [project.is_scratch, projectId, sessionId]);

  // Close on outside click.
  useEffect(() => {
    if (!open) return;
    function handleClickOutside(e: MouseEvent) {
      if (
        popoverRef.current &&
        !popoverRef.current.contains(e.target as Node) &&
        btnRef.current &&
        !btnRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [open]);

  // Close on Escape.
  useEffect(() => {
    if (!open) return;
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false);
    }
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [open]);

  // Clear the error auto-dismiss timer on unmount.
  useEffect(() => {
    return () => {
      if (errorTimerRef.current) clearTimeout(errorTimerRef.current);
    };
  }, []);

  // Chip renders ONLY for the scratch (Quick Tasks) project.
  if (!project.is_scratch) return null;

  // The multi-select lists every OTHER project (scratch excluded — reading
  // your own workspace is not a scope decision).
  const selectable = projects.filter(
    (p) => p.project_id !== projectId && !p.is_scratch,
  );

  function showError() {
    setErrorVisible(true);
    if (errorTimerRef.current) clearTimeout(errorTimerRef.current);
    errorTimerRef.current = setTimeout(() => setErrorVisible(false), 4000);
  }

  /** Optimistic PUT: flip the chip now, persist in the background, revert on
   *  failure. The stored record from the response is authoritative. */
  function applyScope(next: SessionScope) {
    if (!sessionId) return;
    const prev = scope;
    const epoch = writeEpochRef.current + 1;
    writeEpochRef.current = epoch;
    setScope(next); // optimistic
    api<SessionScope>(scopeUrl(projectId, sessionId), {
      method: 'PUT',
      body: JSON.stringify(next),
    })
      .then((stored) => {
        if (writeEpochRef.current === epoch && isScopeRecord(stored)) {
          setScope(stored);
        }
      })
      .catch(() => {
        if (writeEpochRef.current === epoch) {
          setScope(prev); // revert
          showError();
        }
      });
  }

  function handleMasterToggle() {
    if (scope.mode === 'all') {
      // Master OFF: fall back to the explicit selection when one exists,
      // else to workspace-only.
      applyScope(
        scope.selected_project_ids.length > 0
          ? { mode: 'selected', selected_project_ids: scope.selected_project_ids }
          : { mode: 'off', selected_project_ids: [] },
      );
    } else {
      // Master ON: keep the explicit selection in the record so toggling
      // back off restores it.
      applyScope({ mode: 'all', selected_project_ids: scope.selected_project_ids });
    }
  }

  function handleProjectToggle(id: string) {
    // Base selection the click mutates: under 'all' every row is checked, so
    // unchecking one narrows to an explicit selection of everything else.
    const base =
      scope.mode === 'all'
        ? selectable.map((p) => p.project_id)
        : scope.mode === 'selected'
          ? scope.selected_project_ids
          : [];
    const nextIds = base.includes(id)
      ? base.filter((x) => x !== id)
      : [...base, id];
    applyScope(
      nextIds.length === 0
        ? { mode: 'off', selected_project_ids: [] }
        : { mode: 'selected', selected_project_ids: nextIds },
    );
  }

  const label =
    scope.mode === 'all'
      ? t('scopeChip.label.all')
      : scope.mode === 'off'
        ? t('scopeChip.label.off')
        : t(
            scope.selected_project_ids.length === 1
              ? 'scopeChip.label.selected.one'
              : 'scopeChip.label.selected.other',
            { n: scope.selected_project_ids.length },
          );

  return (
    <div className="relative inline-block" data-testid="scope-chip">
      <button
        ref={btnRef}
        type="button"
        disabled={!sessionId}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={t('scopeChip.chip.aria')}
        data-testid="scope-chip-trigger"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full border border-border bg-card text-xs font-medium text-secondary hover:text-primary hover:bg-card-hover transition-colors disabled:opacity-50 disabled:cursor-default"
      >
        <span data-testid="scope-chip-label">{label}</span>
        <ChevronDown size={12} aria-hidden="true" className="shrink-0" />
      </button>

      {errorVisible && (
        <span
          role="alert"
          data-testid="scope-chip-error"
          className="absolute left-0 top-full mt-1 z-50 whitespace-nowrap rounded-md border border-error/30 bg-error/10 px-2 py-1 text-[11px] text-error"
        >
          {t('scopeChip.error.update')}
        </span>
      )}

      {open && (
        <div
          ref={popoverRef}
          role="dialog"
          aria-label={t('scopeChip.popover.aria')}
          data-testid="scope-chip-popover"
          className="absolute left-0 top-full mt-1 z-50 w-64 rounded-md border border-border bg-elevated shadow-md p-3"
        >
          {/* Master toggle */}
          <label className="flex items-center gap-2 text-xs text-primary cursor-pointer">
            <input
              type="checkbox"
              checked={scope.mode === 'all'}
              onChange={handleMasterToggle}
              data-testid="scope-chip-master-toggle"
              className="accent-accent"
            />
            <span className="font-medium">{t('scopeChip.toggle.all')}</span>
          </label>

          {/* Project multi-select */}
          <div className="mt-2 pt-2 border-t border-border">
            <div className="text-2xs uppercase tracking-wide text-muted mb-1">
              {t('scopeChip.projects.heading')}
            </div>
            {selectable.length === 0 ? (
              <p className="text-xs text-secondary">
                {t('scopeChip.projects.empty')}
              </p>
            ) : (
              <ul className="max-h-48 overflow-y-auto">
                {selectable.map((p) => {
                  const checked =
                    scope.mode === 'all' ||
                    (scope.mode === 'selected' &&
                      scope.selected_project_ids.includes(p.project_id));
                  return (
                    <li key={p.project_id}>
                      <label className="flex items-center gap-2 py-1 text-xs text-primary cursor-pointer">
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() => handleProjectToggle(p.project_id)}
                          data-testid={`scope-chip-project-${p.project_id}`}
                          className="accent-accent"
                        />
                        <span className="truncate">{p.name}</span>
                      </label>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>

          {/* Fixed footnote — the write guarantee. */}
          <p className="mt-2 pt-2 border-t border-border text-[11px] text-secondary">
            {t('scopeChip.footnote')}
          </p>
        </div>
      )}
    </div>
  );
}
