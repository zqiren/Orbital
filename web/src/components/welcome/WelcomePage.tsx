// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The Welcome page (batch 1123): the first run on ONE page, replacing the
 * four-page setup wizard (Agents → Projects → key → Accounts).
 *
 * Single column: a header, three bands that fold (see foldModel), and a
 * fixed footer with one button.
 *
 *  - Agents: every agent detected on this computer (AgentsBand).
 *  - Projects: only when the history scan found something. One line,
 *    "导入 N 个项目", with a live checkbox; 查看全部 opens the existing
 *    checklist (ImportProjectsStep, embedded) as the working area.
 *  - The Orbital Agent: open by default, optional. The connect screen
 *    (ConnectPanel, embedded) with DeepSeek preselected.
 *
 * Start is enabled when an agent is ready (installed AND signed in) or the
 * Orbital Agent is connected. It saves a passing key (making it the default
 * card), or else persists `setup_api_key_skipped` so this page does not come
 * back; then fires the import of what is ticked WITHOUT waiting for it
 * (utils/backgroundImport) and hands over to App, which lands on the
 * Workbench at once. Nothing is hidden or written about agents here: hiding one
 * stays in Settings.
 *
 * Telemetry keeps the `wizard_step_reached` allowlist: `agents` on mount,
 * `key` when a key or a sign-in passes, `done` at Start.
 */

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Check, ChevronDown, ChevronRight, Globe, Loader2 } from 'lucide-react';
import { api } from '../../config';
import { useCredentialCards } from '../../hooks/useCredentialCards';
import { useSubAgentList } from '../../hooks/useSubAgentList';
import { useOnboardingScan } from '../../hooks/useOnboardingScan';
import type { CredentialCard } from '../../types';
import { useT } from '../../i18n/useT';
import { useLocale } from '../../i18n/LocaleContext';
import { LOCALES } from '../../i18n/locales';
import { hasImportables, NO_CHOICES, uncheckedCount, type ImportChoices } from '../../utils/onboarding';
import { isEmptyImport, startBackgroundImport } from '../../utils/backgroundImport';
import { sendTelemetry } from '../../utils/telemetry';
import ConnectPanel, { type ConnectFooterCtx } from '../connect/ConnectPanel';
import ProviderLogo from '../connect/ProviderLogo';
import HintSign from '../HintSign';
import ImportProjectsStep from '../ImportProjectsStep';
import Select from '../Select';
import AgentGlyph from './AgentGlyph';
import AutoHeight from './AutoHeight';
import { AgentsBandBody, AgentsSummary } from './AgentsBand';
import {
  canStart, closeReview, engageOrb, INITIAL_FOLD, isFolded, isReviewing, toggleBand, toggleReview,
  type FoldInputs, type FoldState,
} from './foldModel';
import {
  defaultImportCount, importAgents, importCount, importLineTick, importRequestOf, readyAgentCount, toggleImportLine,
} from './welcomeModel';

/** The provider the Orbital band opens on (the prototype's, both languages). */
const DEFAULT_PROVIDER = 'deepseek';

interface WelcomePageProps {
  onComplete: () => void;
}

/** What the page needs from the connect panel, reported after each render. */
interface ConnState {
  connected: boolean;
  canSubmit: boolean;
  busy: boolean;
  error: string | null;
  provider: string;
  model: string;
}

const NO_CONN: ConnState = { connected: false, canSubmit: false, busy: false, error: null, provider: '', model: '' };

function sameConn(a: ConnState, b: ConnState): boolean {
  return a.connected === b.connected && a.canSubmit === b.canSubmit && a.busy === b.busy
    && a.error === b.error && a.provider === b.provider && a.model === b.model;
}

/** Rendered as the panel's footer: hands its state up after the panel renders. */
function ConnectReporter({ ctx, onReport }: { ctx: ConnectFooterCtx; onReport: (ctx: ConnectFooterCtx) => void }) {
  useLayoutEffect(() => { onReport(ctx); });
  return null;
}

function LocaleToggle() {
  const t = useT();
  const { locale, setLocale } = useLocale();
  return (
    <div className="inline-flex shrink-0 items-center gap-1.5">
      <Globe className="h-3.5 w-3.5 text-secondary" aria-hidden="true" />
      <Select
        aria-label={t('global.language')}
        data-testid="wizard-locale-select"
        value={locale}
        onChange={(e) => setLocale(e.target.value as typeof locale)}
        className="rounded-lg border border-border bg-card px-2 py-1 text-caption text-primary transition-all duration-150 focus:border-accent focus:outline-none"
      >
        {LOCALES.map((l) => (
          <option key={l.code} value={l.code}>{l.label}</option>
        ))}
      </Select>
    </div>
  );
}

function Chevron({ open }: { open: boolean }) {
  return open
    ? <ChevronDown aria-hidden="true" className="h-3.5 w-3.5 shrink-0 text-secondary" strokeWidth={2} />
    : <ChevronRight aria-hidden="true" className="h-3.5 w-3.5 shrink-0 text-secondary" strokeWidth={2} />;
}

const foldedRow =
  'surface-float surface-float-nested flex h-12 w-full min-w-0 items-center gap-2.5 !rounded-[12px] px-3.5 text-left';
const bandHead = 'flex h-9 w-full min-w-0 items-center gap-2 text-left';

function LineCheckbox({ state }: { state: 'all' | 'some' | 'none' }) {
  return (
    <span
      aria-hidden="true"
      className={`inline-grid h-4 w-4 shrink-0 place-items-center rounded-[4px] border transition-colors duration-150 ${
        state === 'none' ? 'border-muted bg-card' : 'border-accent bg-accent text-white'
      }`}
    >
      {state === 'all' && <Check className="h-3 w-3" strokeWidth={3} />}
      {state === 'some' && <span className="h-[2px] w-2 rounded bg-white" />}
    </span>
  );
}

export default function WelcomePage({ onComplete }: WelcomePageProps) {
  const t = useT();
  const scan = useOnboardingScan();
  const agents = useSubAgentList();
  const { cards, defaultCardId, refresh: refreshCards } = useCredentialCards();
  const defaultCard = cards.find((c) => c.id === defaultCardId) ?? undefined;

  const [choices, setChoices] = useState<ImportChoices>(NO_CHOICES);
  const [fold, setFold] = useState<FoldState>(INITIAL_FOLD);
  const [conn, setConn] = useState<ConnState>(NO_CONN);
  const submitRef = useRef<() => void>(() => {});
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  // ── the Start flow's phases (refs: read across the panel's callbacks) ──
  /** Start pressed with an unsaved key: waiting for the panel's save. */
  const awaitingKeyRef = useRef(false);
  const sawBusyRef = useRef(false);
  /** The panel reported done for this Start: its busy ending is not a failure
   *  (the panel leaves busy right after onDone, before the checks below). */
  const doneSeenRef = useRef(false);
  /** The key (or sign-in) is saved and is what Orbital runs on. */
  const keySavedRef = useRef(false);

  useEffect(() => { sendTelemetry('wizard_step_reached', { step: 'agents' }); }, []);
  const keyReportedRef = useRef(false);
  useEffect(() => {
    if (!conn.connected || keyReportedRef.current) return;
    keyReportedRef.current = true;
    sendTelemetry('wizard_step_reached', { step: 'key' });
  }, [conn.connected]);

  const onReport = useCallback((ctx: ConnectFooterCtx) => {
    submitRef.current = ctx.submit;
    const next: ConnState = {
      connected: ctx.connected, canSubmit: ctx.canSubmit, busy: ctx.busy, error: ctx.error,
      provider: ctx.provider, model: ctx.model,
    };
    setConn((prev) => (sameConn(prev, next) ? prev : next));
  }, []);

  const data = scan.importable;
  const found = !!data && hasImportables(data);
  const showProjects = found || scan.error || (!scan.settled && !found);
  const ready = readyAgentCount(agents.entries);
  const inputs: FoldInputs = { ready, connected: conn.connected, hasProjects: found };
  const agentsFolded = isFolded('agents', fold, inputs);
  const orbFolded = isFolded('orb', fold, inputs);
  const reviewing = isReviewing(fold, inputs);
  const n = importCount(data, choices);
  const gateOpen = canStart(inputs) && agents.entries !== null;
  const noneInstalled = agents.entries !== null && !agents.error && !agents.entries.some((e) => e.installed);

  // Esc closes the review.
  useEffect(() => {
    if (!reviewing) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !e.defaultPrevented) setFold((s) => closeReview(s));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [reviewing]);

  // ── Start ──
  const finish = useCallback(async () => {
    // No key: persist the choice so the first-run gate stops showing this page.
    if (!keySavedRef.current) {
      try {
        await api('/api/v2/settings', { method: 'PUT', body: JSON.stringify({ setup_api_key_skipped: true }) });
      } catch {
        setStartError(t('wizard.skipKey.failed'));
        setStarting(false);
        return;
      }
    }
    // The import runs in the background: Start lands on the Workbench at
    // once, the sidebar shows a spinner until it answers (backgroundImport).
    const request = importRequestOf(data, choices);
    if (data && !isEmptyImport(request)) {
      sendTelemetry('projects_unchecked', { n: uncheckedCount(data, choices.ticks) });
      void startBackgroundImport(request);
    }
    sendTelemetry('wizard_step_reached', { step: 'done' });
    onComplete();
  }, [data, choices, onComplete, t]);

  // The panel saved the key (or finished a sign-in): make it the card
  // Orbital runs on, then carry on with Start.
  const handleConnected = useCallback(async (card: CredentialCard) => {
    void refreshCards();
    let ok = false;
    try {
      let isDefault = card.is_default;
      if (!isDefault) {
        const res = await api<{ default_card_id: string | null }>(
          `/api/v2/settings/cards/${encodeURIComponent(card.id)}/default`,
          { method: 'PUT' },
        ).catch(() => null);
        isDefault = res?.default_card_id === card.id;
      }
      const settings = await api<{ llm: { api_key_set: boolean } }>('/api/v2/settings');
      ok = settings.llm.api_key_set || isDefault;
    } catch {
      ok = false;
    }
    if (!awaitingKeyRef.current) return;
    awaitingKeyRef.current = false;
    if (!ok) {
      setStartError(t('wizard.apiKey.notSaved'));
      setStarting(false);
      return;
    }
    keySavedRef.current = true;
    await finish();
  }, [finish, refreshCards, t]);

  // A save that failed ends without onDone: the panel went busy and came back.
  useEffect(() => {
    if (!awaitingKeyRef.current || doneSeenRef.current) return;
    if (conn.busy) { sawBusyRef.current = true; return; }
    if (sawBusyRef.current) {
      sawBusyRef.current = false;
      awaitingKeyRef.current = false;
      setStarting(false);
    }
  }, [conn.busy]);

  const start = useCallback(() => {
    if (!gateOpen || starting) return;
    setStartError(null);
    setStarting(true);
    if (conn.connected && !keySavedRef.current && conn.canSubmit) {
      awaitingKeyRef.current = true;
      sawBusyRef.current = false;
      doneSeenRef.current = false;
      submitRef.current();
      return;
    }
    void finish();
  }, [gateOpen, starting, conn.connected, conn.canSubmit, finish]);

  // ── bands ──
  const agentsBand = agentsFolded ? (
    <button
      type="button"
      className={foldedRow}
      onClick={() => setFold((s) => toggleBand(s, 'agents', inputs))}
      aria-expanded={false}
      aria-label={noneInstalled ? t('wizard.steps.agents') : t('welcome.agents.title')}
      data-testid="welcome-band-agents"
    >
      <span className="min-w-0 flex-1"><AgentsSummary entries={agents.entries} /></span>
      <Chevron open={false} />
    </button>
  ) : (
    <div data-testid="welcome-band-agents">
      <button
        type="button"
        className={bandHead}
        onClick={() => setFold((s) => toggleBand(s, 'agents', inputs))}
        aria-expanded={true}
      >
        <span className="min-w-0 flex-1 truncate text-body font-medium text-primary">
          {noneInstalled ? t('wizard.steps.agents') : t('welcome.agents.title')}
        </span>
        <Chevron open />
      </button>
      <AgentsBandBody entries={agents.entries} error={agents.error} reload={agents.reload} />
    </div>
  );

  const lineTick = importLineTick(data, choices);
  const lineN = n || defaultImportCount(data, choices);
  const lineAgents = importAgents(data);
  const scanning = !scan.settled;
  const projectsBand = showProjects && (
    <div
      data-testid="welcome-band-projects"
      className={`surface-float-static surface-float-nested !rounded-[12px] ${reviewing ? 'flex min-h-0 flex-1 flex-col' : ''}`}
    >
      {scan.error && !found ? (
        <div className="flex h-12 items-center gap-2 px-3.5 text-body" role="alert" data-testid="welcome-projects-error">
          <span className="min-w-0 flex-1 truncate text-secondary">{t('importProjects.error')}</span>
          <button
            type="button"
            onClick={scan.retry}
            data-testid="welcome-projects-retry"
            className="shrink-0 font-medium text-accent-text hover:underline max-md:min-h-[44px]"
          >
            {t('importProjects.retry')}
          </button>
        </div>
      ) : !found ? (
        <div className="flex h-12 items-center gap-2 px-3.5 text-body text-secondary" data-testid="welcome-projects-scanning">
          <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
          <span className="truncate">{t('importProjects.scanning')}</span>
        </div>
      ) : (
        <>
          <div className="flex h-12 min-w-0 items-center gap-2.5 px-3.5">
            <button
              type="button"
              role="checkbox"
              aria-checked={lineTick === 'all' ? true : lineTick === 'some' ? 'mixed' : false}
              onClick={() => setChoices((prev) => toggleImportLine(prev, data))}
              data-testid="welcome-import-check"
              className="flex min-w-0 items-center gap-2.5 max-md:min-h-[44px]"
            >
              <LineCheckbox state={lineTick} />
              <span className="truncate text-body font-medium text-primary" data-testid="welcome-import-label">
                {t(lineN === 1 ? 'importProjects.import.one' : 'importProjects.import.other', { n: lineN })}
              </span>
            </button>
            <span className="flex shrink-0 items-center gap-0.5">
              {lineAgents.slice(0, 4).map((slug) => <AgentGlyph key={slug} slug={slug} size={14} />)}
              {lineAgents.length > 4 && <span className="ml-0.5 text-micro text-secondary">+{lineAgents.length - 4}</span>}
            </span>
            {scanning && <Loader2 aria-label={t('importProjects.scanning')} className="h-3.5 w-3.5 shrink-0 animate-spin text-secondary" />}
            <span className="flex-1" />
            <button
              type="button"
              onClick={() => setFold((s) => toggleReview(s, inputs))}
              aria-expanded={reviewing}
              data-testid="welcome-review"
              className="inline-flex shrink-0 items-center gap-1 text-caption text-secondary transition-colors duration-150 hover:text-primary max-md:min-h-[44px]"
            >
              {reviewing ? t('welcome.review.done') : (
                <>
                  {t('welcome.review')}
                  <ChevronDown aria-hidden="true" className="h-3 w-3" />
                </>
              )}
            </button>
          </div>
          {reviewing && (
            <div className="min-h-0 flex-1 overflow-y-auto border-t border-border/60 px-3 pb-3 pt-3" data-testid="welcome-review-list">
              <ImportProjectsStep
                scan={scan}
                embedded
                choices={[choices, setChoices]}
                onImported={() => {}}
              />
            </div>
          )}
        </>
      )}
    </div>
  );

  const optionalTag = !noneInstalled && !conn.connected && (
    <span className="shrink-0 rounded-full bg-card-hover px-2 py-0.5 text-micro text-secondary" data-testid="welcome-orb-optional">
      {t('welcome.optional')}
    </span>
  );
  const orbValue = conn.connected && (
    <span className="flex min-w-0 items-center gap-1.5" data-testid="welcome-orb-value">
      <ProviderLogo id={conn.provider} size={16} />
      <span className="truncate font-mono text-caption text-primary">{conn.model}</span>
      <Check aria-hidden="true" className="h-3.5 w-3.5 shrink-0 text-success-text" strokeWidth={2.2} />
    </span>
  );
  const orbBand = (
    <div data-testid="welcome-band-orb">
      {orbFolded ? (
        <button
          type="button"
          className={foldedRow}
          onClick={() => setFold((s) => toggleBand(s, 'orb', inputs))}
          aria-expanded={false}
        >
          <AgentGlyph slug="orbital" size={20} />
          <span className="shrink-0 text-body font-medium text-primary">{t('wizard.steps.key')}</span>
          {optionalTag}
          <span className="flex-1" />
          {orbValue}
          <Chevron open={false} />
        </button>
      ) : (
        <div className="flex h-9 min-w-0 items-center gap-2">
          <button
            type="button"
            className="flex min-w-0 items-center gap-2 text-left"
            onClick={() => setFold((s) => toggleBand(s, 'orb', inputs))}
            aria-expanded={true}
          >
            <AgentGlyph slug="orbital" size={20} />
            <span className="truncate text-body font-medium text-primary">{t('wizard.steps.key')}</span>
            {optionalTag}
          </button>
          <HintSign label={t('connect.wizard.introLabel')} testId="welcome-orb-hint">
            {t('connect.wizard.intro')}
          </HintSign>
          <button
            type="button"
            className="flex h-full flex-1 items-center justify-end"
            onClick={() => setFold((s) => toggleBand(s, 'orb', inputs))}
            aria-label={t('wizard.steps.key')}
          >
            <Chevron open />
          </button>
        </div>
      )}
      {/* Mounted while folded: a typed key, a sign-in and the Start flow live here. */}
      <div className={orbFolded ? 'hidden' : 'mt-2'}>
        <ConnectPanel
          embedded
          initialProvider={DEFAULT_PROVIDER}
          replaceCardId={defaultCard && !defaultCard.key_set && !defaultCard.read_only ? defaultCard.id : null}
          onEngage={() => setFold((s) => engageOrb(s))}
          onCardStored={() => { void refreshCards(); }}
          onCardRemoved={() => { void refreshCards(); }}
          onDone={(card) => {
            if (awaitingKeyRef.current) doneSeenRef.current = true;
            void handleConnected(card);
          }}
          footer={(ctx) => <ConnectReporter ctx={ctx} onReport={onReport} />}
        />
      </div>
    </div>
  );

  const label = n > 0
    ? t(n === 1 ? 'welcome.start.import.one' : 'welcome.start.import.other', { n })
    : t('welcome.start');
  const busy = starting || (awaitingKeyRef.current && conn.busy);
  const error = startError ?? (starting ? null : conn.error);

  return (
    // pt-titlebar: a full-screen surface outside the app shell clears the
    // macOS titlebar band itself (the native titlebar eats clicks there).
    <div className="flex h-dvh items-center justify-center bg-background px-4 py-6 pt-titlebar" data-testid="welcome-page">
      <div
        className={`surface-float-static flex w-full max-w-[640px] flex-col overflow-hidden max-h-[calc(100dvh-48px-var(--titlebar-h,0px))] ${
          reviewing ? 'h-[calc(100dvh-48px-var(--titlebar-h,0px))]' : ''
        }`}
        data-testid="welcome-card"
      >
        <div className="flex shrink-0 items-center justify-between gap-4 px-8 pb-3 pt-7 max-md:px-5">
          <div className="flex min-w-0 items-center gap-2.5">
            <img src="/icon-192.png" alt="" aria-hidden="true" className="h-7 w-7 shrink-0 rounded-md" />
            <h1 className="truncate text-heading font-semibold text-primary">{t('sidebar.wordmark')}</h1>
          </div>
          <LocaleToggle />
        </div>
        <div className={`flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-8 pb-2 pt-1 max-md:px-5 ${reviewing ? '' : ''}`}>
          {reviewing ? agentsBand : <AutoHeight>{agentsBand}</AutoHeight>}
          {projectsBand}
          {reviewing ? orbBand : <AutoHeight>{orbBand}</AutoHeight>}
        </div>
        <div className="flex shrink-0 items-center gap-3 px-8 pb-6 pt-4 max-md:px-5">
          <p className="min-w-0 flex-1 text-right text-caption text-error-text" role={error ? 'alert' : undefined} data-testid="welcome-start-error">
            {error}
          </p>
          <HintSign label={t('welcome.privacy')} align="right" testId="welcome-privacy">
            {t('telemetry.wizard.disclosure')}
          </HintSign>
          <button
            type="button"
            onClick={start}
            disabled={!gateOpen || busy}
            aria-busy={busy || undefined}
            data-testid="welcome-start"
            className="inline-flex h-10 shrink-0 items-center gap-2 whitespace-nowrap rounded-[10px] bg-accent px-5 text-body font-medium text-white transition-all duration-150 hover:bg-accent/90 disabled:opacity-45 max-md:min-h-[44px]"
          >
            {busy && <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />}
            {label}
          </button>
        </div>
      </div>
    </div>
  );
}
