// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../config';
import { keepShownOrder, type HistoryStatus, type ImportableResponse } from '../utils/onboarding';

export interface OnboardingScan {
  history: HistoryStatus | null;
  importable: ImportableResponse | null;
  /** The importable list could not be fetched at all. */
  error: boolean;
  /** No more updates are coming: the scan is ready, polling gave up, or the
   *  fetch failed. Before this, an empty list may still fill in. */
  settled: boolean;
  /** Fetch again from scratch (the error line's Try again). */
  retry: () => void;
}

interface Options {
  pollMs?: number;
  maxMs?: number;
}

/**
 * The agent-history scan behind the wizard's Agents and Projects steps and the
 * import dialog. GET /onboarding/history first — the call that starts the
 * daemon's background scan — then the importable list, whose candidates are
 * there at once while the groups arrive when the scan is ready. Polls both
 * while either says `status == "scanning"` (every 1.5 s, at most 90 s). Once the scan is
 * ready the folder rows are rebuilt from it, so giving up early leaves the
 * first, noisier list; reading Codex's compressed rollouts made a cold scan
 * take 10–14 s on a dev Mac, and a heavier history takes longer. The list
 * also says scanning while the daemon's walk for git repos runs (batch 1115:
 * bounded at 60 s, inside the 90 s here), each answer carrying the repos
 * found so far; rows already shown keep their place (keepShownOrder).
 */
export function useOnboardingScan({ pollMs = 1500, maxMs = 90000 }: Options = {}): OnboardingScan {
  const [history, setHistory] = useState<HistoryStatus | null>(null);
  const [importable, setImportable] = useState<ImportableResponse | null>(null);
  const [error, setError] = useState(false);
  const [settled, setSettled] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const optsRef = useRef({ pollMs, maxMs });

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    // The list as last shown: rows that arrive later never reorder it.
    let shown: ImportableResponse | null = null;
    const started = Date.now();
    const { pollMs: every, maxMs: limit } = optsRef.current;

    const tick = async () => {
      const h = await api<HistoryStatus>('/api/v2/onboarding/history').catch(() => null);
      if (cancelled) return;
      const historyOk = !!h && typeof h === 'object' && 'by_agent' in h;
      if (historyOk) setHistory(h);
      // The folder list can be "ready" from its first pass while the history
      // scan behind the conversation count is still running; stopping on the
      // list alone froze the step at "Found in 0 conversations".
      const historyScanning = historyOk && (h as { status?: string }).status === 'scanning';
      let data: ImportableResponse | null = null;
      try {
        data = await api<ImportableResponse>('/api/v2/onboarding/importable-projects');
      } catch {
        if (cancelled) return;
        setError(true);
        setSettled(true);
        return;
      }
      if (cancelled) return;
      const body: ImportableResponse = keepShownOrder(shown, {
        status: data?.status,
        candidates: Array.isArray(data?.candidates) ? data.candidates : [],
        groups: Array.isArray(data?.groups) ? data.groups : [],
      });
      shown = body;
      setImportable(body);
      if ((body.status === 'scanning' || historyScanning) && Date.now() - started < limit) {
        timer = setTimeout(() => { void tick(); }, every);
      } else {
        setSettled(true);
      }
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [attempt]);

  const retry = useCallback(() => {
    setError(false);
    setSettled(false);
    setAttempt((a) => a + 1);
  }, []);

  return { history, importable, error, settled, retry };
}
