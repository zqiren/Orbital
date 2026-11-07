// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * ImportProjectsStep — the project checklist built from the user's agent
 * history (spec 138, wizard step 2 and the "Import projects from your agents"
 * dialog).
 *
 * Everything is offered checked and the user unchecks: folders they worked in
 * (git repos, folders with AGENTS.md / CLAUDE.md, or 3+ conversations), then
 * — collapsed and unchecked — folders with only one or two conversations, then
 * groups of scattered conversations that belong together. A group needs a
 * folder before it can be imported; without one it stays on the list but is
 * neither sent nor counted. One POST /onboarding/import creates them all
 * (link-only: Orbital writes nothing into a folder until it is first used).
 *
 * Candidates render at once; groups arrive when the scan is ready (the
 * `scan` prop polls). Data comes from `useOnboardingScan` so the wizard can
 * decide whether to show this step before it does.
 */

import { useState } from 'react';
import { Check, ChevronDown, ChevronRight, FolderOpen, Loader2 } from 'lucide-react';
import { api } from '../config';
import { useT } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import TokenCostSign from './TokenCostSign';
import type { StringKey } from '../i18n/strings';
import { formatRelativeTime } from '../utils/relativeTime';
import {
  agentDisplayName,
  buildImportRequest,
  folderKey,
  groupFolder,
  groupKey,
  isFolderChecked,
  isGroupChecked,
  landingProjectId,
  uncheckedCount,
  type GroupReason,
  type ImportCandidate,
  type ImportGroup,
  type ImportResult,
  type TickOverrides,
} from '../utils/onboarding';
import { sendTelemetry } from '../utils/telemetry';
import type { OnboardingScan } from '../hooks/useOnboardingScan';
import FolderBrowserPanel from './FolderBrowserPanel';

interface ImportProjectsStepProps {
  scan: OnboardingScan;
  /** After a successful import: what was created, and the project to land in
   *  (the most recently active one; null when nothing was created). */
  onImported: (result: ImportResult, landingProjectId: string | null) => void;
  onBack?: () => void;
  /** Leave without importing. The wizard labels it Skip, the dialog Cancel. */
  onSkip?: () => void;
  skipLabel?: string;
  /** The dialog's layout: the list scrolls and the actions sit in a footer
   *  below it, always in view (a long list used to hide Cancel and Import). */
  scrollable?: boolean;
  /** Spec 160 round 2 (the dialog only): who reads the imported projects
   *  once — Orbital with the key, else the user's own agent (product name).
   *  Null/absent: nobody, no sign. The wizard imports before its key step,
   *  so it passes nothing. */
  scanRunner?: { key: boolean; agent: string | null } | null;
  /** The wizard: its import comes before the key step, so the read waits
   *  for that step's Save or Skip (``defer_scan`` on the request). */
  deferScan?: boolean;
}

const basename = (p: string) => p.replace(/\\/g, '/').split('/').filter(Boolean).pop() ?? p;

/** "Claude Code 182 · Codex 41" — product names and numbers, not translated. */
function agentCounts(byAgent: Record<string, number> | undefined): string {
  return Object.entries(byAgent ?? {})
    .filter(([, n]) => n > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([slug, n]) => `${agentDisplayName(slug)} ${n}`)
    .join(' · ');
}

const REASON_KEY: Record<string, StringKey> = {
  same_skill: 'importProjects.reason.sameSkill',
  same_files: 'importProjects.reason.sameFiles',
  same_request: 'importProjects.reason.sameRequest',
  same_folder: 'importProjects.reason.sameFolder',
};

function reasonText(t: ReturnType<typeof useT>, r: GroupReason): string | null {
  const key = REASON_KEY[r.kind];
  return key ? t(key, { value: r.value, count: r.count }) : null;
}

const checkboxClass = 'mt-0.5 h-4 w-4 shrink-0 accent-accent cursor-pointer';

export default function ImportProjectsStep({
  scan,
  onImported,
  onBack,
  onSkip,
  skipLabel,
  scrollable = false,
  scanRunner,
  deferScan = false,
}: ImportProjectsStepProps) {
  const t = useT();
  const { locale } = useLocale();
  const [ticks, setTicks] = useState<TickOverrides>({});
  const [chosen, setChosen] = useState<Record<string, string>>({});
  const [showMinor, setShowMinor] = useState(false);
  const [browsingGroup, setBrowsingGroup] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  // Round 2 opt-out (the dialog's sign): off = import without the read.
  const [scanOn, setScanOn] = useState(true);
  const [importError, setImportError] = useState(false);

  const data = scan.importable;
  const candidates = data?.candidates ?? [];
  const groups = data?.groups ?? [];
  const major = candidates.filter((c) => !c.minor);
  const minor = candidates.filter((c) => c.minor);
  const request = data ? buildImportRequest(data, ticks, chosen) : { folders: [], groups: [] };
  const count = request.folders.length + request.groups.length;
  const scanning = !scan.settled && data?.status === 'scanning';

  const toggle = (key: string, checked: boolean) =>
    setTicks((prev) => ({ ...prev, [key]: checked }));

  async function handleImport() {
    if (!data || count === 0 || importing) return;
    setImporting(true);
    setImportError(false);
    sendTelemetry('projects_unchecked', { n: uncheckedCount(data, ticks) });
    try {
      const result = await api<ImportResult>('/api/v2/onboarding/import', {
        method: 'POST',
        // Spec 160: the language the background read replies in.
        body: JSON.stringify({
          ...request,
          lang: locale,
          ...(scanRunner && !scanOn ? { import_scan: false } : {}),
          ...(deferScan ? { defer_scan: true } : {}),
        }),
      });
      const safe: ImportResult = {
        created: Array.isArray(result?.created) ? result.created : [],
        skipped: Array.isArray(result?.skipped) ? result.skipped : [],
      };
      onImported(safe, landingProjectId(safe, request, data));
    } catch {
      setImportError(true);
    } finally {
      setImporting(false);
    }
  }

  function renderFolder(c: ImportCandidate) {
    const checked = isFolderChecked(c, ticks);
    const counts = c.source === 'obsidian' && !c.by_agent ? t('importProjects.vault') : agentCounts(c.by_agent ?? { [c.source]: c.session_count });
    const body = (
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          <span className="min-w-0 flex-1 truncate text-sm font-medium text-primary">{c.name}</span>
          {c.already_imported && (
            <span className="shrink-0 text-xs text-success" data-testid={`import-imported-${c.path}`}>
              {t('importProjects.alreadyImported')}
            </span>
          )}
          {c.last_activity && (
            <span className="shrink-0 text-xs text-secondary">{formatRelativeTime(c.last_activity, t)}</span>
          )}
        </div>
        <p className="truncate font-mono text-[11px] text-secondary" title={c.path}>{c.path}</p>
        {counts && <p className="text-xs text-secondary/80">{counts}</p>}
      </div>
    );
    if (c.already_imported) {
      return (
        <li key={c.path} data-testid={`import-candidate-${c.path}`} className="flex items-start gap-3 surface-float-static surface-float-nested px-3 py-2.5 opacity-75">
          <Check className="mt-0.5 h-4 w-4 shrink-0 text-success" aria-hidden="true" />
          {body}
        </li>
      );
    }
    return (
      <li key={c.path} data-testid={`import-candidate-${c.path}`}>
        <label className="flex cursor-pointer items-start gap-3 surface-float surface-float-nested px-3 py-2.5 max-md:min-h-[44px]">
          <input
            type="checkbox"
            checked={checked}
            onChange={(e) => toggle(folderKey(c.path), e.target.checked)}
            data-testid={`import-check-${c.path}`}
            className={checkboxClass}
          />
          {body}
        </label>
      </li>
    );
  }

  function renderGroup(g: ImportGroup) {
    const checked = isGroupChecked(g, ticks);
    const folder = groupFolder(g, chosen);
    const agents = Object.keys(g.by_agent ?? {})
      .filter((s) => (g.by_agent[s] ?? 0) > 0)
      .map((s) => agentDisplayName(s))
      .join(', ');
    const meta = t(g.session_count === 1 ? 'importProjects.group.meta.one' : 'importProjects.group.meta.other', {
      n: g.session_count,
      agents,
      when: formatRelativeTime(g.last_activity, t),
    });
    const reasons = (g.reasons ?? []).map((r) => reasonText(t, r)).filter((s): s is string => !!s);
    const titleId = `import-group-title-${g.id}`;
    return (
      <li key={g.id} data-testid={`import-group-${g.id}`} className="surface-float-static surface-float-nested px-3 py-2.5">
        <div className="flex items-start gap-3">
          <input
            type="checkbox"
            checked={checked}
            onChange={(e) => toggle(groupKey(g.id), e.target.checked)}
            aria-labelledby={titleId}
            data-testid={`import-group-check-${g.id}`}
            className={checkboxClass}
          />
          <div className="min-w-0 flex-1">
            <p
              id={titleId}
              className={`truncate text-sm font-medium ${folder ? 'text-primary' : 'text-warning-text'}`}
              data-testid={`import-group-title-${g.id}`}
            >
              {folder ? basename(folder) : t('importProjects.group.needsFolder')}
            </p>
            <p className="text-xs text-secondary">{meta}</p>
            {reasons.length > 0 && (
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {reasons.map((r) => (
                  <span key={r} className="max-w-full truncate rounded-full border border-border bg-sidebar px-2 py-0.5 text-[11px] text-secondary">
                    {r}
                  </span>
                ))}
              </div>
            )}
            <div className="mt-2 flex items-center gap-2">
              <span className="shrink-0 text-xs text-secondary">{t('importProjects.group.folder')}</span>
              <span
                className={`min-w-0 flex-1 truncate text-xs ${folder ? 'font-mono text-primary' : 'text-warning-text'}`}
                title={folder ?? undefined}
              >
                {folder ?? t('importProjects.group.chooseHint')}
              </span>
              <button
                type="button"
                onClick={() => setBrowsingGroup((cur) => (cur === g.id ? null : g.id))}
                aria-expanded={browsingGroup === g.id}
                data-testid={`import-group-choose-${g.id}`}
                className="shrink-0 inline-flex items-center gap-1 rounded-lg border border-border px-2.5 py-1 text-xs font-medium text-primary transition-colors duration-150 hover:bg-surface max-md:min-h-[44px]"
              >
                <FolderOpen className="h-3.5 w-3.5" aria-hidden="true" />
                {folder ? t('importProjects.group.change') : t('importProjects.group.choose')}
              </button>
            </div>
            {browsingGroup === g.id && (
              <div className="mt-2 overflow-hidden rounded-lg border border-border" data-testid={`import-group-browser-${g.id}`}>
                <FolderBrowserPanel
                  compact
                  suggestedFolderName={folder ? basename(folder) : ''}
                  onSelect={(path) => {
                    setChosen((prev) => ({ ...prev, [g.id]: path }));
                    setBrowsingGroup(null);
                  }}
                />
              </div>
            )}
          </div>
        </div>
      </li>
    );
  }

  const nothing = scan.settled && !scan.error && candidates.length === 0 && groups.length === 0;

  const content = (
    <>
      {scan.error && (
        <div className="mb-4 text-sm text-error" role="alert" data-testid="import-error">
          {t('importProjects.error')}{' '}
          <button type="button" onClick={scan.retry} className="underline hover:no-underline" data-testid="import-retry">
            {t('importProjects.retry')}
          </button>
        </div>
      )}

      {!data && !scan.error && (
        <div className="mb-4 inline-flex items-center gap-2 text-sm text-secondary">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          {t('importProjects.scanning')}
        </div>
      )}

      {nothing && (
        <p className="mb-4 text-sm text-secondary" data-testid="import-empty">{t('importProjects.empty')}</p>
      )}

      {(major.length > 0 || minor.length > 0) && (
        <section className="mb-5" aria-labelledby="import-folders-heading">
          <h2 id="import-folders-heading" className="mb-2 text-sm font-medium text-primary">
            {t('importProjects.folders.heading')}
          </h2>
          {major.length > 0 && <ul className="space-y-2" data-testid="import-list">{major.map(renderFolder)}</ul>}
          {minor.length > 0 && (
            <>
              <button
                type="button"
                onClick={() => setShowMinor((v) => !v)}
                aria-expanded={showMinor}
                data-testid="import-minor-toggle"
                className="mt-2 inline-flex items-center gap-1 text-xs font-medium text-secondary transition-colors duration-150 hover:text-primary max-md:min-h-[44px]"
              >
                {showMinor ? <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" /> : <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />}
                {t(minor.length === 1 ? 'importProjects.minor.toggle.one' : 'importProjects.minor.toggle.other', { n: minor.length })}
              </button>
              {showMinor && <ul className="mt-2 space-y-2" data-testid="import-minor-list">{minor.map(renderFolder)}</ul>}
            </>
          )}
        </section>
      )}

      {(groups.length > 0 || scanning) && (
        <section className="mb-5" aria-labelledby="import-groups-heading">
          <h2 id="import-groups-heading" className="text-sm font-medium text-primary">
            {t('importProjects.groups.heading')}
          </h2>
          <p className="mb-2 text-xs text-secondary">{t('importProjects.groups.sub')}</p>
          {groups.length > 0 && <ul className="space-y-2" data-testid="import-groups">{groups.map(renderGroup)}</ul>}
          {scanning && (
            <div className="mt-1 inline-flex items-center gap-2 text-xs text-secondary" data-testid="import-groups-scanning">
              <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
              {t('importProjects.scanning')}
            </div>
          )}
        </section>
      )}

      {importError && (
        <p className="mb-2 text-right text-xs text-error" role="alert" data-testid="import-failed">
          {t('importProjects.importFailed')}
        </p>
      )}

    </>
  );

  const actions = (
    <div className={`flex items-center gap-3${scrollable ? '' : ' mt-6'}`} data-testid="import-actions">
      {onBack && (
        <button
          type="button"
          onClick={onBack}
          data-testid="import-back"
          className="text-sm font-medium text-secondary transition-colors duration-150 hover:text-primary max-md:min-h-[44px]"
        >
          {t('global.back')}
        </button>
      )}
      <div className="ml-auto flex items-center gap-3">
        {onSkip && (
          <button
            type="button"
            onClick={onSkip}
            data-testid="import-skip"
            className="rounded-lg border border-border px-5 py-2.5 text-sm font-medium text-primary transition-all duration-150 hover:bg-surface"
          >
            {skipLabel ?? t('wizard.skip')}
          </button>
        )}
        {scanRunner && (
          <TokenCostSign
            n={count}
            kind={scanRunner.key ? 'importKey' : 'importAgent'}
            agent={scanRunner.agent}
            scan={scanOn}
            onScanChange={setScanOn}
          />
        )}
        <button
          type="button"
          onClick={handleImport}
          disabled={count === 0 || importing}
          data-testid="import-submit"
          className="inline-flex items-center gap-2 rounded-lg bg-accent px-5 py-2.5 text-sm font-medium text-white transition-all duration-150 hover:bg-accent/90 disabled:opacity-50"
        >
          {importing && <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />}
          {importing
            ? t('importProjects.importing')
            : t(count === 1 ? 'importProjects.import.one' : 'importProjects.import.other', { n: count })}
        </button>
      </div>
    </div>
  );

  if (scrollable) {
    return (
      <div data-testid="import-projects" className="flex min-h-0 flex-1 flex-col">
        <div data-testid="import-dialog-scroll" className="min-h-0 flex-1 overflow-y-auto px-6 pt-3 pb-4">
          {content}
        </div>
        <div className="border-t border-border px-6 py-3">{actions}</div>
      </div>
    );
  }

  return (
    <div data-testid="import-projects">
      {content}
      {actions}
    </div>
  );
}
