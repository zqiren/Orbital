// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * ImportProjectsStep — first-run onboarding import surface (backlog #34).
 *
 * A confirm-each list of candidate projects discovered from other CLI agents
 * (Claude Code, Codex) and Obsidian vaults on disk. Nothing is created
 * automatically:
 *
 *   1. Disclose-then-scan — a one-line disclosure that Orbital will read other
 *      apps' local data is shown BEFORE any scan; the scan runs only when the
 *      user clicks "Scan".
 *   2. The GET /api/v2/onboarding/importable-projects route returns ranked,
 *      deduplicated, path-verified candidates (metadata only — never
 *      conversation content).
 *   3. Each row the user confirms calls the EXISTING POST /api/v2/projects with
 *      the real folder as the workspace (link-only) — reusing the standard
 *      project-creation flow (which triggers the cold-start scan) verbatim.
 */

import { useCallback, useState } from 'react';
import { Loader2, Search, Check, FolderPlus } from 'lucide-react';
import { api, ApiError } from '../config';
import type { Project } from '../types';
import { useT } from '../i18n/useT';

/** Candidate shape returned by /api/v2/onboarding/importable-projects. */
export interface ImportCandidate {
  source: 'claude-code' | 'codex' | 'obsidian' | string;
  name: string;
  path: string;
  session_count: number;
  last_activity: string | null;
  /** Spec 084: the folder is already a project's workspace (no Add offered). */
  already_imported?: boolean;
}

interface ImportableResponse {
  candidates: ImportCandidate[];
}

type ScanPhase = 'idle' | 'scanning' | 'done' | 'error';
type RowStatus = 'idle' | 'adding' | 'added' | 'error';

/**
 * Spec 084: why an Add failed, keyed by HTTP status so the row can say
 * something true. A 409 is a name collision the backend will return forever
 * (retry cannot fix it); a 400 is a folder that vanished between scan and
 * click; anything else is genuinely retryable.
 */
type AddFailure = 'conflict' | 'gone' | 'failed';

function classifyAddFailure(err: unknown): AddFailure {
  if (err instanceof ApiError) {
    if (err.status === 409) return 'conflict';
    if (err.status === 400) return 'gone';
  }
  return 'failed';
}

function addFailureKey(f: AddFailure): 'import.addConflict' | 'import.addGone' | 'import.addFailed' {
  switch (f) {
    case 'conflict':
      return 'import.addConflict';
    case 'gone':
      return 'import.addGone';
    default:
      return 'import.addFailed';
  }
}

interface ImportProjectsStepProps {
  /** Called after a candidate is successfully created (link-only). */
  onProjectCreated?: (project: Project) => void;
}

function sourceLabel(t: ReturnType<typeof useT>, source: string): string {
  switch (source) {
    case 'claude-code':
      return t('import.source.claudeCode');
    case 'codex':
      return t('import.source.codex');
    case 'obsidian':
      return t('import.source.obsidian');
    default:
      return source;
  }
}

/** "N sessions" for agent projects; "Vault" for Obsidian (no sessions). */
function metaLabel(t: ReturnType<typeof useT>, c: ImportCandidate): string {
  if (c.source === 'obsidian') return t('import.vault');
  const n = c.session_count;
  return n === 1
    ? t('import.sessions.one', { n })
    : t('import.sessions.other', { n });
}

export default function ImportProjectsStep({ onProjectCreated }: ImportProjectsStepProps) {
  const t = useT();
  const [phase, setPhase] = useState<ScanPhase>('idle');
  const [candidates, setCandidates] = useState<ImportCandidate[]>([]);
  // Keyed by candidate path (paths are unique post-dedup).
  const [rowStatus, setRowStatus] = useState<Record<string, RowStatus>>({});
  const [rowFailure, setRowFailure] = useState<Record<string, AddFailure>>({});

  const handleScan = useCallback(async () => {
    setPhase('scanning');
    try {
      const data = await api<ImportableResponse>('/api/v2/onboarding/importable-projects');
      setCandidates(data.candidates ?? []);
      setPhase('done');
    } catch {
      setPhase('error');
    }
  }, []);

  const handleAdd = useCallback(
    async (c: ImportCandidate) => {
      setRowStatus((s) => ({ ...s, [c.path]: 'adding' }));
      try {
        // Reuse the standard project-creation flow verbatim: link-only, the
        // real folder IS the workspace. Backend triggers the cold-start scan.
        // auto_unique_name (spec 084): the name is the folder basename, not
        // something the user typed, so a taken name becomes name-2 rather
        // than a 409 the wizard could never get past.
        const project = await api<Project>('/api/v2/projects', {
          method: 'POST',
          body: JSON.stringify({ name: c.name, workspace: c.path, auto_unique_name: true }),
        });
        setRowStatus((s) => ({ ...s, [c.path]: 'added' }));
        onProjectCreated?.(project);
      } catch (err) {
        setRowFailure((s) => ({ ...s, [c.path]: classifyAddFailure(err) }));
        setRowStatus((s) => ({ ...s, [c.path]: 'error' }));
      }
    },
    [onProjectCreated],
  );

  return (
    <div data-testid="wizard-import-group">
      <div className="flex items-center gap-2 mb-1">
        <FolderPlus className="w-4 h-4 text-accent" />
        <h2 className="text-sm font-medium text-primary">{t('import.heading')}</h2>
      </div>
      {/* Disclose-then-scan: the disclosure is always shown before the scan. */}
      <p className="text-xs text-secondary mb-3" data-testid="import-disclosure">
        {t('import.disclosure')}
      </p>

      {phase === 'idle' && (
        <button
          type="button"
          onClick={handleScan}
          data-testid="import-scan"
          className="border border-border text-primary text-sm font-medium rounded-lg px-4 py-2 hover:bg-surface transition-all duration-150 inline-flex items-center gap-2"
        >
          <Search className="w-4 h-4" />
          {t('import.scan')}
        </button>
      )}

      {phase === 'scanning' && (
        <div className="inline-flex items-center gap-2 text-sm text-secondary">
          <Loader2 className="w-4 h-4 animate-spin" />
          {t('import.scanning')}
        </div>
      )}

      {phase === 'error' && (
        <div className="text-sm text-error" role="alert" data-testid="import-error">
          {t('import.error')}{' '}
          <button
            type="button"
            onClick={handleScan}
            className="underline hover:no-underline"
            data-testid="import-rescan"
          >
            {t('import.rescan')}
          </button>
        </div>
      )}

      {phase === 'done' && candidates.length === 0 && (
        <p className="text-sm text-secondary" data-testid="import-empty">
          {t('import.empty')}
        </p>
      )}

      {phase === 'done' && candidates.length > 0 && (
        <ul className="space-y-2" data-testid="import-list">
          {candidates.map((c) => {
            const status = rowStatus[c.path] ?? 'idle';
            // Spec 084: an already-imported folder shows the same check state
            // as a row added in this session, labelled honestly.
            const imported = status === 'added' || c.already_imported === true;
            return (
              <li
                key={c.path}
                data-testid={`import-candidate-${c.path}`}
                className="border border-border rounded-lg px-3 py-2.5"
              >
                <div className="flex items-center gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-primary truncate">
                        {c.name}
                      </span>
                      <span className="shrink-0 text-[10px] uppercase tracking-wide text-secondary border border-border rounded px-1.5 py-0.5">
                        {sourceLabel(t, c.source)}
                      </span>
                    </div>
                    <p className="text-xs text-secondary truncate" title={c.path}>
                      {c.path}
                    </p>
                    <p className="text-xs text-secondary/80">{metaLabel(t, c)}</p>
                  </div>
                  {imported ? (
                    <span
                      className="shrink-0 inline-flex items-center gap-1 text-xs text-success"
                      data-testid={`import-added-${c.path}`}
                    >
                      <Check className="w-3.5 h-3.5" />
                      {status === 'added' ? t('import.added') : t('import.alreadyImported')}
                    </span>
                  ) : (
                    <button
                      type="button"
                      data-testid={`import-add-${c.path}`}
                      disabled={status === 'adding'}
                      onClick={() => handleAdd(c)}
                      className="shrink-0 text-xs font-medium text-white bg-accent rounded-lg px-3 py-1.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-60 inline-flex items-center gap-1.5 max-md:min-h-[44px]"
                    >
                      {status === 'adding' ? (
                        <>
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                          {t('import.adding')}
                        </>
                      ) : (
                        t('import.add')
                      )}
                    </button>
                  )}
                </div>
                {status === 'error' && (
                  <p
                    className="mt-2 text-xs text-error"
                    role="alert"
                    data-testid={`import-add-error-${c.path}`}
                  >
                    {t(addFailureKey(rowFailure[c.path] ?? 'failed'))}
                  </p>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
