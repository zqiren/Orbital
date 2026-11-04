// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * UnsandboxedRunsSection — Project Settings → 沙盒外运行 / Runs outside the
 * sandbox (spec 117 D5/D6). The same shape as NetworkAccessSection:
 *
 *  - the executables the user answered "always allow" for, each with a
 *    remove (saved through the project PUT as a replace-list);
 *  - the parked `unsandboxed_run` asks a hands-off run auto-denied, each with
 *    the four answers. Answers go to the dedicated resolve route — the
 *    daemon runs the command (or not) and records the consequence — and the
 *    caller applies the visible consequence locally.
 *
 * Owns no state: SettingsView holds both lists and decides what to persist.
 */

import { useState } from 'react';
import { useT } from '../i18n/useT';
import type { PendingUnsandboxedRun, UnsandboxedRunDecision } from '../types';

/** The executable a command runs: its first token (spec 117 D5). */
export function commandExecutable(command: string): string {
  return command.trim().split(/\s+/)[0] ?? '';
}

/**
 * The program "Always allow" would allowlist. The daemon names it
 * (`executable`) — not always the first word (`cmd /c curl.exe x` →
 * `curl.exe`), and `""` when the command cannot be allowlisted at all
 * (chained, piped, interpreter-wrapped): the caller then offers no "Always
 * allow". Only an older daemon that omits the field falls back to the first
 * word.
 */
export function allowlistExecutable(command: string, executable?: string | null): string {
  return typeof executable === 'string' ? executable : commandExecutable(command);
}

interface UnsandboxedRunsSectionProps {
  executables: string[];
  pendingRuns: PendingUnsandboxedRun[];
  onRemoveExecutable: (exe: string) => void;
  /** Resolves on success; a rejection keeps the row and shows an error. */
  onResolve: (run: PendingUnsandboxedRun, decision: UnsandboxedRunDecision | 'deny') => Promise<void>;
}

export function UnsandboxedRunsSection({
  executables,
  pendingRuns,
  onRemoveExecutable,
  onResolve,
}: UnsandboxedRunsSectionProps) {
  const t = useT();
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function resolve(run: PendingUnsandboxedRun, decision: UnsandboxedRunDecision | 'deny') {
    setBusyId(run.id);
    setError(null);
    try {
      await onResolve(run, decision);
    } catch {
      setError(t('settings.unsandboxed.resolveError'));
    } finally {
      setBusyId(null);
    }
  }

  const answerClass =
    'text-xs font-medium border rounded-lg px-2.5 py-1 transition-all duration-150 disabled:opacity-50 max-md:min-h-[44px]';

  return (
    // Heading + intro come from the enclosing SettingsSection.
    <div data-testid="unsandboxed-runs-section">
      {pendingRuns.length > 0 && (
        <div className="mb-3">
          <h4 className="text-xs font-semibold text-primary mb-1.5">
            {t('settings.unsandboxed.pendingTitle')}
          </h4>
          <div className="flex flex-col gap-2">
            {pendingRuns.map((run) => {
              const exe = allowlistExecutable(run.command, run.executable);
              const busy = busyId === run.id;
              return (
                <div
                  key={run.id}
                  className="surface-float-static px-3 py-2 space-y-2"
                >
                  <code className="block text-xs font-mono text-primary whitespace-pre-wrap break-all">
                    {run.command}
                  </code>
                  {run.reason && (
                    <p className="text-xs text-secondary">{run.reason}</p>
                  )}
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => resolve(run, 'once')}
                      className={`${answerClass} text-accent border-border hover:bg-accent/5`}
                    >
                      {t('approval.unsandboxed.allowOnce')}
                    </button>
                    {exe && (
                      <button
                        type="button"
                        disabled={busy}
                        onClick={() => resolve(run, 'always')}
                        className={`${answerClass} text-accent border-border hover:bg-accent/5`}
                      >
                        {t('approval.unsandboxed.alwaysAllow', { exe })}
                      </button>
                    )}
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => resolve(run, 'disable_safe_mode')}
                      className={`${answerClass} text-secondary border-border hover:bg-secondary/5`}
                    >
                      {t('approval.unsandboxed.disableSafeMode')}
                    </button>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => resolve(run, 'deny')}
                      className={`${answerClass} text-error border-error/40 hover:bg-error/5`}
                    >
                      {t('settings.unsandboxed.deny')}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
          {error && <p className="text-xs text-error mt-2">{error}</p>}
        </div>
      )}

      {executables.length === 0 ? (
        <p className="text-xs text-secondary/60 italic">{t('settings.unsandboxed.empty')}</p>
      ) : (
        <div className="flex flex-col gap-1">
          {executables.map((exe) => (
            <div key={exe} className="flex items-center gap-2 py-1 max-md:min-h-[44px]">
              <span className="text-sm font-mono text-primary flex-1 truncate">{exe}</span>
              <button
                type="button"
                aria-label={t('settings.unsandboxed.removeAria', { exe })}
                onClick={() => onRemoveExecutable(exe)}
                className="text-xs font-medium text-error border border-error/40 rounded-lg px-2.5 py-1 hover:bg-error/5 transition-all duration-150 max-md:min-h-[44px]"
              >
                {t('settings.unsandboxed.remove')}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
