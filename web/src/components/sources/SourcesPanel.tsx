// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Task sources (任务来源) — the panel in the Workbench's docked pane (a
 * bottom sheet on a phone), the place to connect the outside services the
 * daemon watches and to govern them. Approved prototype:
 * BACKLOG/specs/177-assets/sources-prototype-v4.html.
 *
 * Views, one at a time with a back control: the list (or, while nothing is
 * connected, the guide: three beats and what can start right away), the
 * connect list (我的账号 / 我的机器人 / 网站检查, from the catalogue), a connect
 * form, a source, a source's reconnect form. Nothing here runs an agent or
 * takes a prompt: connecting is fixed forms, governing is switches, the
 * instruction and the history.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { ChevronRight, Folder, Plus, X } from 'lucide-react';
import type { Project } from '../../types';
import { useT } from '../../i18n/useT';
import { useLocale } from '../../i18n/LocaleContext';
import { projectDisplayName } from '../../utils/projectDisplayName';
import SourceMark from './SourceMark';
import ConnectForm from './ConnectForm';
import SourceDetail from './SourceDetail';
import { Hint, Switch } from './parts';
import { connectGroups } from './formModel';
import { fetchAccounts, patchSource } from './sourcesApi';
import { loadCatalog, loadDetail, putDetail, setOpenDetail, useSources } from './sourcesStore';
import { relTime, sourceName, typeName } from './sourcesText';
import type { AccountsResponse, CatalogType, SourceSummary } from './types';
import './sources.css';

export type SourcesView =
  | { v: 'list' }
  | { v: 'add' }
  | { v: 'new'; type: string }
  | { v: 'detail'; id: string; arrival?: string | null }
  | { v: 're'; id: string };

export interface SourcesFocus {
  sourceId: string;
  arrivalId: string | null;
  /** A new value re-opens the same source (a second click on a card's mark). */
  nonce: number;
}

export interface SourcesPanelProps {
  phone: boolean;
  projects: readonly Project[];
  focus: SourcesFocus | null;
  onClose: () => void;
  onOpenSession: (projectId: string, sessionId: string) => void;
}

const CLI_TYPES = ['github', 'gitlab'];

function SourceRow({ s, flash, onOpen }: { s: SourceSummary; flash: boolean; onOpen: () => void }) {
  const t = useT();
  const fresh = s.is_new && !s.on;
  const by = s.registered_by;
  const acc = fresh || s.type === 'command'
    ? (by?.project_name || (by ? t('sources.ext.outside') : ''))
    : s.type === 'website' ? '' : (s.label ?? '');
  const failed = s.status === 'failed';
  return (
    <div
      role="button"
      tabIndex={0}
      className={`srow${s.on ? '' : ' is-off'}${flash ? ' flash' : ''}`}
      onClick={onOpen}
      onKeyDown={(e) => { if (e.key === 'Enter' && e.target === e.currentTarget) onOpen(); }}
      data-testid="src-row"
      data-id={s.id}
    >
      <SourceMark type={s.type} off={!s.on} />
      <span className="nmw">
        <span className="nm">{sourceName(t, s)}</span>
        {acc && <span className="ev">{acc}</span>}
      </span>
      {fresh && <span className="newp" data-testid="src-row-new">{t('sources.ext.new')}</span>}
      {s.waiting > 0 && <span className="wc" data-testid="src-row-waiting">{t('sources.waiting', { n: s.waiting })}</span>}
      <span className="lt">
        {failed && <span className="sd warn" data-testid="src-row-failed" />}
        {relTime(t, s.last_check_at)}
      </span>
      <Switch
        on={s.on}
        label={t('sources.toggle')}
        disabled={s.status === 'unavailable'}
        onChange={(on) => {
          patchSource(s.id, { on }).then(putDetail).catch(() => {});
        }}
        testId="src-row-switch"
      />
    </div>
  );
}

function KindRow({ typeDef, accounts, onPick }: { typeDef: CatalogType; accounts: AccountsResponse | null; onPick: () => void }) {
  const t = useT();
  const cli = typeDef.connect.method === 'cli_account';
  const acc = cli && accounts
    ? accounts.state === 'signed_in' ? (accounts.accounts.find((a) => a.active) ?? accounts.accounts[0])?.login ?? ''
      : t(accounts.state === 'not_installed' ? 'sources.cli.missing' : 'sources.cli.signedOut')
    : '';
  return (
    <button type="button" className="kr" onClick={onPick} data-testid="src-kind-row" data-type={typeDef.type}>
      <SourceMark type={typeDef.type} />
      <span className="nm">{typeName(t, typeDef.type)}</span>
      {acc && <span className="acc">{acc}</span>}
      {typeDef.type === 'calendar' && <Hint text={t('sources.calendar.hint')} />}
      <span className="sp" />
      <span className="cv"><ChevronRight size={14} strokeWidth={2} aria-hidden="true" /></span>
    </button>
  );
}

export default function SourcesPanel({ phone, projects, focus, onClose, onOpenSession }: SourcesPanelProps) {
  const t = useT();
  const { locale } = useLocale();
  const store = useSources();
  const [stack, setStack] = useState<SourcesView[]>([]);
  const [flash, setFlash] = useState<string | null>(null);
  const [toastText, setToastText] = useState<string | null>(null);
  const [accounts, setAccounts] = useState<Record<string, AccountsResponse | null>>({});
  // A card's mark opens its source at the entry it came from (a new focus
  // replaces the stack; adjusted during render, not in an effect).
  const [seenFocus, setSeenFocus] = useState<SourcesFocus | null>(null);
  if (focus && focus !== seenFocus) {
    setSeenFocus(focus);
    setStack([{ v: 'detail', id: focus.sourceId, arrival: focus.arrivalId }]);
  }
  const view: SourcesView = useMemo(() => stack[stack.length - 1] ?? { v: 'list' }, [stack]);

  useEffect(() => { void loadCatalog(); }, []);
  const catalog = store.catalog;
  const typeOf = useCallback((type: string) => catalog?.types.find((x) => x.type === type) ?? null, [catalog]);

  const detailId = view.v === 'detail' || view.v === 're' ? view.id : null;
  useEffect(() => {
    setOpenDetail(detailId);
    if (detailId) void loadDetail(detailId);
    return () => setOpenDetail(null);
  }, [detailId]);

  const refreshAccounts = useCallback(async (type: string) => {
    try {
      const a = await fetchAccounts(type);
      setAccounts((m) => ({ ...m, [type]: a }));
    } catch {
      setAccounts((m) => ({ ...m, [type]: m[type] ?? null }));
    }
  }, []);
  // The CLI accounts are read where they are shown: the guide, the connect
  // list, a CLI form, a CLI source.
  const cliShown = useMemo(() => {
    if (view.v === 'list' && store.loaded && store.list.length === 0) return CLI_TYPES;
    if (view.v === 'add') return CLI_TYPES;
    if (view.v === 'new' && CLI_TYPES.includes(view.type)) return [view.type];
    if (view.v === 'detail') {
      const s = store.list.find((x) => x.id === view.id);
      if (s && CLI_TYPES.includes(s.type)) return [s.type];
    }
    return [];
  }, [view, store.loaded, store.list]);
  const cliKey = cliShown.join(',');
  useEffect(() => {
    if (!catalog) return;
    for (const ty of cliKey ? cliKey.split(',') : []) {
      if (typeOf(ty)?.available) void refreshAccounts(ty);
    }
  }, [cliKey, refreshAccounts, typeOf, catalog]);

  const toast = useCallback((s: string) => setToastText(s), []);
  useEffect(() => {
    if (!toastText) return;
    const id = setTimeout(() => setToastText(null), 2200);
    return () => clearTimeout(id);
  }, [toastText]);
  useEffect(() => {
    if (!flash) return;
    const id = setTimeout(() => setFlash(null), 2300);
    return () => clearTimeout(id);
  }, [flash]);

  const push = (v: SourcesView) => setStack((s) => [...s, v]);
  const back = useCallback(() => setStack((s) => s.slice(0, -1)), []);

  // Esc steps back (a menu open above takes Esc first and stops it).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || e.defaultPrevented) return;
      const tg = e.target as HTMLElement | null;
      if (tg && (tg.tagName === 'INPUT' || tg.tagName === 'TEXTAREA')) return;
      if (stack.length) back(); else onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [stack.length, back, onClose]);

  const groups = useMemo(() => connectGroups(catalog, locale), [catalog, locale]);
  const header = (title: string, opts: { back?: boolean; plus?: boolean }) => (
    <div className={`wb-ph${opts.back ? ' hb' : ''}`}>
      {opts.back && (
        <button type="button" className="ib" onClick={back} title={t('sources.back')} aria-label={t('sources.back')} data-testid="src-back">
          <ChevronRight size={16} strokeWidth={2.2} style={{ transform: 'rotate(180deg)' }} aria-hidden="true" />
        </button>
      )}
      <span className="pt" data-testid="src-title">{title}</span>
      {opts.plus && (
        <button type="button" className="ib" onClick={() => push({ v: 'add' })} title={t('sources.add')} aria-label={t('sources.add')} data-testid="src-plus">
          <Plus size={16} strokeWidth={2} aria-hidden="true" />
        </button>
      )}
      <button type="button" className="ib" onClick={onClose} title={t('sources.close')} aria-label={t('sources.close')} data-testid="src-close">
        <X size={16} strokeWidth={1.9} aria-hidden="true" />
      </button>
    </div>
  );
  const kindRow = (ty: CatalogType) => (
    <KindRow key={ty.type} typeDef={ty} accounts={accounts[ty.type] ?? null} onPick={() => push({ v: 'new', type: ty.type })} />
  );

  let body: React.ReactNode;
  if (view.v === 'new' || view.v === 're') {
    const s = view.v === 're' ? store.details[view.id] : null;
    const ty = typeOf(view.v === 'new' ? view.type : s?.type ?? '');
    body = !ty || (view.v === 're' && !s) ? <div className="src-view">{header('', { back: true })}</div> : (
      <ConnectForm
        key={`${view.v}:${ty.type}`}
        typeDef={ty}
        source={s}
        accounts={accounts[ty.type] ?? null}
        onAccountsCheck={() => refreshAccounts(ty.type)}
        onBack={back}
        onClose={onClose}
        onDone={(d) => {
          if (view.v === 're') { setStack((st) => [...st.slice(0, -1)]); return; }
          setStack([]);
          setFlash(d.id);
        }}
      />
    );
  } else if (view.v === 'detail') {
    const s = store.details[view.id];
    body = !s ? <div className="src-view">{header('', { back: true })}</div> : (
      <SourceDetail
        key={s.id}
        source={s}
        typeDef={typeOf(s.type)}
        accounts={accounts[s.type] ?? null}
        onAccountsCheck={() => refreshAccounts(s.type)}
        projects={projects}
        focusArrival={view.arrival ?? null}
        onBack={back}
        onReconnect={() => push({ v: 're', id: s.id })}
        onOpenSession={onOpenSession}
        onRemoved={() => setStack([])}
        toast={toast}
      />
    );
  } else if (view.v === 'add') {
    body = (
      <div className="src-view is-in" data-testid="src-add">
        {header(t('sources.add'), { back: true })}
        <div className="src-body">
          <div className="kinds2">
            {groups.account.length > 0 && <><div className="kg type-label">{t('sources.group.account')}</div><div className="kc">{groups.account.map(kindRow)}</div></>}
            {groups.bot.length > 0 && <><div className="kg type-label">{t('sources.group.bot')}</div><div className="kc">{groups.bot.map(kindRow)}</div></>}
            {groups.check.length > 0 && <div className="kc kc-solo">{groups.check.map(kindRow)}</div>}
          </div>
        </div>
      </div>
    );
  } else {
    const has = store.list.length > 0;
    // What can start with nothing filled in: a CLI found signed in, Calendar on a Mac.
    const ready = [...groups.account].filter((ty) => ty.available && (
      ty.connect.method === 'none' || (ty.connect.method === 'cli_account' && accounts[ty.type]?.state === 'signed_in')));
    // Until the CLIs have answered, the first step is not known: show none.
    const asking = !catalog || groups.account.some((ty) => ty.available && ty.connect.method === 'cli_account' && accounts[ty.type] === undefined);
    body = (
      <div className="src-view" data-testid="src-list">
        {header(t('sources.title'), { plus: has })}
        <div className="src-body">
          {has ? (
            <div className="slist">
              {store.list.map((s) => (
                <SourceRow key={s.id} s={s} flash={flash === s.id} onOpen={() => push({ v: 'detail', id: s.id })} />
              ))}
            </div>
          ) : store.loaded ? (
            <div className="guide" data-testid="src-guide">
              <div className="gbeats">
                <div className="gb">
                  <div className="gpic"><span className="gstack">
                    {(locale === 'zh' ? ['github', 'feishu', 'dingtalk'] : ['github', 'slack', 'telegram']).map((ty) => <SourceMark key={ty} type={ty} size="md" />)}
                  </span></div>
                  <div className="gl">{t('sources.guide.1')}</div>
                </div>
                <span className="garr"><ChevronRight size={14} strokeWidth={2} aria-hidden="true" /></span>
                <div className="gb">
                  <div className="gpic"><span className="gcard"><span className="gdot" /><span className="gln"><i /><i /></span></span></div>
                  <div className="gl">{t('sources.guide.2')}</div>
                </div>
                <span className="garr"><ChevronRight size={14} strokeWidth={2} aria-hidden="true" /></span>
                <div className="gb">
                  <div className="gpic"><span className="gproj"><Folder size={12} strokeWidth={1.9} aria-hidden="true" /><span className="t">{(() => { const pr = projects.find((p) => !p.is_scratch) ?? projects[0]; return pr ? projectDisplayName(pr, t) : 'Orbital'; })()}</span></span></div>
                  <div className="gl">{t('sources.guide.3')}</div>
                </div>
              </div>
              {asking ? null : ready.length > 0 ? (
                <>
                  <div className="kg type-label">{t('sources.guide.ready')}</div>
                  <div className="kc">{ready.map(kindRow)}</div>
                  <button type="button" className="gmore" onClick={() => push({ v: 'add' })} data-testid="src-guide-more">
                    {t('sources.guide.more')}<ChevronRight size={13} strokeWidth={2} aria-hidden="true" />
                  </button>
                </>
              ) : (
                <div className="gfoot">
                  <button type="button" className="go inline" onClick={() => push({ v: 'add' })} data-testid="src-guide-add">{t('sources.add')}</button>
                </div>
              )}
            </div>
          ) : null}
        </div>
      </div>
    );
  }

  return (
    <div className={`src-panel${phone ? ' is-phone' : ''}`} data-testid="src-panel" data-view={view.v}>
      {body}
      {toastText && <div className="src-toast" role="status">{toastText}</div>}
    </div>
  );
}
