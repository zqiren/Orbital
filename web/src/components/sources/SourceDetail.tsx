// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Task sources — one source, opened: its name, the pause switch and ⋯ in the
 * header; then what failed (with only the actions the daemon offers), whose
 * account, what was registered from outside, what counts as an arrival, the
 * instruction that travels with every item, and the history.
 *
 * History rows (newest first): the title is the one link to the session the
 * item became; under it the project (spec 180's picker moves a wrong route),
 * the session's state as plain text, merged updates behind «n 条更新». Bots
 * add who sent it and where; lossy bots show their connection gaps. Held
 * items are quiet, with one catch-up; the queue is one quiet row.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ChevronDown, ChevronLeft, Folder, MoreHorizontal, RotateCw, Sparkles, Trash2, Unplug, User, Users, X,
} from 'lucide-react';
import type { Project } from '../../types';
import { useT } from '../../i18n/useT';
import { useLocale } from '../../i18n/LocaleContext';
import { moveSession } from '../../hooks/useWorkbenchFeed';
import { projectDisplayName } from '../../utils/projectDisplayName';
import { canRevealInFileManager, revealLabelKey } from '../../utils/clientPlatform';
import { AnchoredProjectPicker } from '../workbench/ProjectPicker';
import SourceMark from './SourceMark';
import { AccountField, AgentLogo, AnchoredMenu, CommandRow, Dots, KindsCard, MenuOption, PickCard, Switch } from './parts';
import { buildHistoryRows, dateLabel, gapLabel, type HistoryRow } from './historyModel';
import {
  catchUp, checkSource, deleteSource, dismissHeld, fetchChoices, openSettingsPane, patchSource, revealSource,
  routeArrival, sourceError, type SourcePatch,
} from './sourcesApi';
import { dropSource, loadDetail, loadOlder, patchLocal, putDetail } from './sourcesStore';
import { failureText, labelText, placeName, placeholderText, relTime, sourceName } from './sourcesText';
import type { AccountsResponse, CatalogType, ChoiceItem, HistoryItem, SourceDetail as Detail } from './types';
import { pickOf, pickPayload, pickState, togglePick, type Kinds } from './formModel';

export interface SourceDetailProps {
  source: Detail;
  typeDef: CatalogType | null;
  accounts: AccountsResponse | null;
  onAccountsCheck: () => Promise<void>;
  projects: readonly Project[];
  /** The entry to show (from a Workbench card); highlighted for a moment. */
  focusArrival: string | null;
  onBack: () => void;
  onReconnect: () => void;
  onOpenSession: (projectId: string, sessionId: string) => void;
  onRemoved: () => void;
  toast: (text: string) => void;
}

/** The instruction: saved as typed (no send). */
function Instruction({ source, toast }: { source: Detail; toast: (s: string) => void }) {
  const t = useT();
  const [text, setText] = useState(source.instruction ?? '');
  const ref = useRef<HTMLTextAreaElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sent = useRef(source.instruction ?? '');
  const latest = useRef(text);
  const idRef = useRef(source.id);

  // Another device changed it: follow, unless this field has unsaved typing.
  useEffect(() => {
    if (timer.current) return;
    if ((source.instruction ?? '') !== sent.current) { sent.current = source.instruction ?? ''; latest.current = sent.current; setText(sent.current); }
  }, [source.instruction]);

  // Grows with its text; re-measured when the pane's width settles (it
  // slides open from zero width).
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const fit = () => { el.style.height = 'auto'; el.style.height = `${Math.max(20, el.scrollHeight)}px`; };
    fit();
    if (typeof ResizeObserver === 'undefined') return;
    let w = el.clientWidth;
    let raf = 0;
    // Measured on the next frame, not inside the observer (WebKit reports a loop otherwise).
    const ro = new ResizeObserver(() => { if (el.clientWidth !== w) { w = el.clientWidth; cancelAnimationFrame(raf); raf = requestAnimationFrame(fit); } });
    ro.observe(el);
    return () => { ro.disconnect(); cancelAnimationFrame(raf); };
  }, [text]);

  const save = () => {
    if (timer.current) { clearTimeout(timer.current); timer.current = null; }
    const v = latest.current;
    if (v === sent.current) return;
    sent.current = v;
    patchSource(idRef.current, { instruction: v }).then(putDetail).catch(() => toast(t('sources.error.action')));
  };
  // Flush on leaving the source.
  useEffect(() => () => { if (timer.current) save(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="pk ins" data-testid="src-instruction">
      <label className="cap" htmlFor={`src-ins-${source.id}`}>{t('sources.instr.cap')}</label>
      <textarea
        id={`src-ins-${source.id}`}
        ref={ref}
        rows={1}
        value={text}
        maxLength={4000}
        placeholder={t('sources.instr.ph')}
        onChange={(e) => {
          latest.current = e.target.value;
          setText(e.target.value);
          patchLocal(source.id, { instruction: e.target.value });
          if (timer.current) clearTimeout(timer.current);
          timer.current = setTimeout(save, 700);
        }}
        onBlur={save}
      />
    </div>
  );
}

function FailLine({ source, typeDef, checking, onCheck, onReconnect, onRemove }: {
  source: Detail; typeDef: CatalogType | null; checking: boolean; onCheck: () => void; onReconnect: () => void;
  onRemove: (anchor: HTMLElement) => void;
}) {
  const t = useT();
  const f = source.failure!;
  const [signEl, setSignEl] = useState<HTMLButtonElement | null>(null);
  const [signOpen, setSignOpen] = useState(false);
  const cliMissing = f.code === 'cli_missing';
  const cmd = cliMissing ? typeDef?.connect.install_command : typeDef?.connect.sign_in_command;
  // No one-line install on this platform: the CLI's own page.
  const installUrl = cliMissing && !cmd ? typeDef?.connect.install_url : null;
  return (
    <div className={`fail${checking ? ' checking' : ''}`} data-testid="src-fail" data-code={f.code}>
      <span className="sd warn" />
      <span className="ft" title={failureText(t, f, source.type)}>{failureText(t, f, source.type)}</span>
      {f.at && <span className="fa">{relTime(t, f.at)}</span>}
      {f.actions.includes('sign_in') && cmd && (
        <button ref={setSignEl} type="button" className={`tbtn acc${signOpen ? ' is-open' : ''}`} onClick={() => setSignOpen((o) => !o)}>
          {t(cliMissing ? 'sources.cli.howInstall' : 'sources.cli.howSignIn')}
        </button>
      )}
      {f.actions.includes('sign_in') && installUrl && (
        <a className="tbtn acc" href={installUrl} target="_blank" rel="noopener noreferrer" data-testid="src-fail-install">{t('sources.cli.download')}</a>
      )}
      {f.actions.includes('open_settings') && (
        <button type="button" className="tbtn acc" onClick={() => void openSettingsPane()} data-testid="src-fail-settings">{t('sources.action.open_settings')}</button>
      )}
      {f.actions.includes('reconnect') && (
        <button type="button" className="tbtn acc" onClick={onReconnect} data-testid="src-fail-reconnect">{t('sources.action.reconnect')}</button>
      )}
      {f.actions.includes('check_again') && (
        <button type="button" className="tbtn acc" onClick={onCheck} disabled={checking} data-testid="src-fail-check">
          {checking ? <Dots /> : t('sources.action.check_again')}
        </button>
      )}
      {f.actions.includes('remove') && !f.actions.includes('check_again') && (
        <button type="button" className="tbtn danger" onClick={(e) => onRemove(e.currentTarget)}>{t('sources.delete')}</button>
      )}
      {signOpen && cmd && (
        <AnchoredMenu anchor={signEl} onClose={() => setSignOpen(false)} width={240} alignRight>
          <CommandRow cmd={cmd} />
        </AnchoredMenu>
      )}
    </div>
  );
}

/** A built-in an agent registered: who added it and its fields, secrets masked. */
function RegisteredCard({ source, typeDef }: { source: Detail; typeDef: CatalogType | null }) {
  const t = useT();
  const fields = typeDef?.connect.fields ?? [];
  return (
    <div className="pk xo by" data-testid="src-registered">
      <FromRow source={source} />
      {fields.map((f) => (
        <div className="xr" key={f.key}>
          <span className="cap">{labelText(t, f.label_key)}</span>
          <span className="xv"><span className="t mono">
            {f.secret ? (source.secrets?.[f.key]?.set ? source.secrets[f.key].masked || '••••••••••••' : '—') : String(source.config?.[f.key] ?? '—')}
          </span></span>
        </div>
      ))}
    </div>
  );
}

function FromRow({ source }: { source: Detail }) {
  const t = useT();
  const by = source.registered_by!;
  const where = by.project_name || (by.by === 'outside' || !by.project_id ? t('sources.ext.outside') : '');
  const who = by.agent ? (by.agent === 'claude-code' ? 'Claude Code' : by.agent === 'codex' ? 'Codex' : by.agent) : null;
  return (
    <div className="xr">
      <span className="cap">{t('sources.ext.from')}</span>
      <span className="xv">
        {by.agent && <AgentLogo slug={by.agent} label={who ?? undefined} />}
        <span className="t">{[who, where].filter(Boolean).join(' · ')}</span>
        {by.at && <span className="xt">{relTime(t, by.at)}</span>}
      </span>
    </div>
  );
}

/** A check command an agent registered: exactly what runs, read-only. */
function CommandCard({ source, toast }: { source: Detail; toast: (s: string) => void }) {
  const t = useT();
  return (
    <div className="pk xo" data-testid="src-command">
      {source.registered_by && <FromRow source={source} />}
      <div className="xr">
        <span className="cap">{t('sources.ext.runs')}</span>
        <span className="xv"><code className="xc">{source.command?.run ?? ''}</code></span>
      </div>
      {canRevealInFileManager() && (
        <div className="xf">
          <button type="button" className="tbtn" onClick={() => { revealSource(source.id).catch(() => toast(t('sources.error.action'))); }} data-testid="src-reveal">
            {t(revealLabelKey())}
          </button>
        </div>
      )}
    </div>
  );
}

function ItemRow({ source, row, projects, hl, onOpenSession, toast }: {
  source: Detail; row: Extract<HistoryRow, { kind: 'item' }>; projects: readonly Project[]; hl: boolean;
  onOpenSession: (p: string, s: string) => void; toast: (s: string) => void;
}) {
  const t = useT();
  const { locale } = useLocale();
  const a: HistoryItem = row.entry;
  const [projEl, setProjEl] = useState<HTMLButtonElement | null>(null);
  const [picker, setPicker] = useState(false);
  const [openUps, setOpenUps] = useState(false);
  const elRef = useRef<HTMLDivElement>(null);
  const now = new Date();
  useEffect(() => { if (hl) elRef.current?.scrollIntoView?.({ block: 'nearest' }); }, [hl]);

  const routed = a.state === 'routed' && !!a.project_id;
  const place = a.place ? (
    <>
      <span className="plc">
        {a.place.kind === 'dm' ? <User size={11} strokeWidth={2} aria-hidden="true" />
          : a.place.kind === 'group' ? <Users size={11} strokeWidth={2} aria-hidden="true" /> : null}
        <span className="t">{placeName(t, a.place)}</span>
      </span>
      <span className="dotsep">·</span>
    </>
  ) : null;

  async function move(to: string) {
    setPicker(false);
    if (!a.project_id || !a.session_id || to === a.project_id) return;
    try {
      await moveSession(a.project_id, a.session_id, to);
      void loadDetail(source.id);
    } catch { toast(t('sources.error.action')); }
  }
  async function route(to: string) {
    setPicker(false);
    try { await routeArrival(source.id, a.id, to); void loadDetail(source.id); } catch { toast(t('sources.error.action')); }
  }

  let t2: React.ReactNode = null;
  if (routed) {
    const auto = a.routed_by === 'model' || a.routed_by === 'fallback';
    const st = a.session_state ?? 'done';
    t2 = (
      <>
        <button ref={setProjEl} type="button" className={`rp${picker ? ' is-open' : ''}`} aria-haspopup="listbox" aria-expanded={picker} onClick={() => setPicker((p) => !p)} data-testid="src-arv-project">
          {auto ? <Sparkles size={10} strokeWidth={2} aria-hidden="true" /> : <Folder size={11} strokeWidth={1.9} aria-hidden="true" />}
          <span className="t">{(() => { const pr = projects.find((p) => p.project_id === a.project_id); return pr ? projectDisplayName(pr, t) : a.project_name ?? ''; })()}</span>
          <ChevronDown size={9} strokeWidth={2.4} aria-hidden="true" />
        </button>
        <span className="dotsep">·</span>
        <span className="ss" data-testid="src-arv-state">
          <span className={`sd ${a.recovered_at ? '' : st === 'running' ? 'run' : st === 'needs_you' ? 'need' : ''}`} />
          {a.recovered_at ? t('sources.st.recovered') : t(`sources.ss.${st}` as 'sources.ss.done')}
        </span>
      </>
    );
  } else if (a.state === 'routing') {
    t2 = <span className="rp picking"><Sparkles size={10} strokeWidth={2} aria-hidden="true" /><span className="t">{t('sources.auto')}</span></span>;
  } else if (a.state === 'unrouted') {
    t2 = (
      <>
        <span className="why warn">{t(`sources.unrouted.${a.reason ?? 'send_failed'}` as 'sources.unrouted.send_failed')}</span>
        <button ref={setProjEl} type="button" className={`chip${picker ? ' is-open' : ''}`} onClick={() => setPicker((p) => !p)} data-testid="src-arv-pick">
          <span className="ic"><Folder size={12} strokeWidth={1.9} aria-hidden="true" /></span>
          <span className="nm">{t('sources.pickProject')}</span>
          <span className="cv"><ChevronDown size={10} strokeWidth={2.2} aria-hidden="true" /></span>
        </button>
      </>
    );
  } else if (a.state === 'held' || a.state === 'dismissed') {
    t2 = <span className="why">{t('sources.st.held')}</span>;
  }
  const ups = a.count > 1 ? (
    <>
      <span className="dotsep">·</span>
      <button type="button" className={`mg${openUps ? ' open' : ''}`} aria-expanded={openUps} onClick={() => setOpenUps((o) => !o)} data-testid="src-arv-updates">
        {t('sources.updates', { n: a.count })}<ChevronDown size={10} strokeWidth={2.2} aria-hidden="true" />
      </button>
    </>
  ) : null;
  const title = <>{a.sender ? <><span className="snd">{a.sender}</span>{locale === 'zh' ? '：' : ': '}</> : null}{a.title}</>;
  return (
    <div ref={elRef} className={`arv${row.quiet ? ' quiet' : ''}${hl ? ' hl' : ''}`} data-testid="src-arv" data-state={a.state} data-id={a.id}>
      <div className="arv-tx">
        {routed && a.session_id ? (
          <button type="button" className="t1" title={t('sources.openSession')} onClick={() => onOpenSession(a.project_id!, a.session_id!)} data-testid="src-arv-title">{title}</button>
        ) : <span className="t1">{title}</span>}
        {(place || t2 || ups) && <span className="t2">{place}{t2}{ups}</span>}
      </div>
      <span className="hx-time">{dateLabel(a.arrived_at, now, locale, (k, v) => t(k as 'sources.time.today', v))}</span>
      {openUps && a.updates.length > 0 && (
        <div className="ups">
          {a.updates.map((u, i) => (
            <div className="u" key={i}><span className="t">{u.title}</span><span className="hx-time">{relTime(t, u.at)}</span></div>
          ))}
        </div>
      )}
      {picker && (
        <AnchoredProjectPicker
          anchor={projEl}
          projects={projects}
          selectedId={routed ? a.project_id : null}
          onPick={(pid) => void (routed ? move(pid) : route(pid))}
          onClose={() => setPicker(false)}
          testId="src-pop-project"
        />
      )}
    </div>
  );
}

export default function SourceDetail({
  source, typeDef, accounts, onAccountsCheck, projects, focusArrival, onBack, onReconnect, onOpenSession, onRemoved, toast,
}: SourceDetailProps) {
  const t = useT();
  const { locale } = useLocale();
  const [name, setName] = useState(source.name);
  const [checking, setChecking] = useState(false);
  const [hl, setHl] = useState<string | null>(focusArrival);
  const moreRef = useRef<HTMLButtonElement>(null);
  const [menu, setMenu] = useState<null | 'more' | 'confirm'>(null);
  const [menuAnchor, setMenuAnchor] = useState<HTMLElement | null>(null);
  useEffect(() => { setName(source.name); }, [source.id, source.name]);
  useEffect(() => {
    setHl(focusArrival);
    if (!focusArrival) return;
    const id = setTimeout(() => setHl(null), 3200);
    return () => clearTimeout(id);
  }, [focusArrival]);

  const rows = useMemo(() => buildHistoryRows(source.history), [source.history]);
  const now = new Date();
  const tr = (k: string, v?: Record<string, string | number>) => t(k as 'sources.time.today', v);
  const isCli = typeDef?.connect.method === 'cli_account';
  // 重新连接: types with a secret (a token, a password) or a QR scan.
  const formType = typeDef?.connect.method === 'qr' || !!typeDef?.connect.fields?.some((f) => f.secret);

  async function patch(p: SourcePatch) {
    try { putDetail(await patchSource(source.id, p)); } catch (e) {
      const err = sourceError(e);
      toast(err.message || t('sources.error.action'));
      void loadDetail(source.id);
    }
  }
  async function act(fn: () => Promise<Detail>) {
    try { putDetail(await fn()); } catch { toast(t('sources.error.action')); }
  }
  async function check() {
    setChecking(true);
    try { putDetail(await checkSource(source.id)); } catch { toast(t('sources.error.action')); } finally { setChecking(false); }
  }
  async function remove() {
    setMenu(null);
    try { await deleteSource(source.id); dropSource(source.id); onRemoved(); } catch { toast(t('sources.error.action')); }
  }
  const saveName = () => { if (name.trim() !== source.name) void patch({ name: name.trim() }); };
  const loadChoices = async (list: string, params: Record<string, unknown>) =>
    (await fetchChoices({ source_id: source.id, list, params })).items ?? [];

  // Calendar: which calendars, read when the source opens (nothing stored = every one ticked).
  const pick = typeDef ? pickOf(typeDef) : null;
  const [pickItems, setPickItems] = useState<ChoiceItem[] | null>(null);
  const pickReq = useRef<{ id: string; p: Promise<ChoiceItem[]> } | null>(null);
  useEffect(() => {
    if (!pick) return;
    let dead = false;
    if (pickReq.current?.id !== source.id) {
      setPickItems(null);
      pickReq.current = { id: source.id, p: fetchChoices({ source_id: source.id, list: pick.choices, params: {} }).then((r) => r.items ?? []) };
    }
    pickReq.current.p.then((items) => { if (!dead) setPickItems(items); }).catch(() => { if (!dead) setPickItems([]); });
    return () => { dead = true; };
  }, [source.id, pick]);
  const pickCard = pick ? (
    <PickCard
      items={pickItems}
      on={pickState(source.config?.[pick.key], pickItems ?? []).on}
      label={t('sources.calendar.list')}
      onToggle={(v) => {
        const next = togglePick(pickState(source.config?.[pick.key], pickItems ?? []), v);
        if (!next.on.length) return; // the last one stays ticked
        const value = pickPayload(next);
        const config = { ...source.config };
        if (value === null) delete config[pick.key]; else config[pick.key] = value;
        putDetail({ ...source, config });
        void patch({ config: { [pick.key]: value } });
      }}
    />
  ) : null;

  const kindsCard = typeDef && source.type !== 'command' ? (
    <KindsCard
      typeDef={typeDef}
      kinds={source.kinds as Kinds}
      onChange={(kinds) => { putDetail({ ...source, kinds }); void patch({ kinds }); }}
      who={!isCli && source.label && source.type !== 'website' ? { label: source.label } : null}
      loadChoices={loadChoices}
    />
  ) : null;
  const websiteCard = source.type === 'website' ? (
    <div className="pk wf" data-testid="src-website">
      {(typeDef?.connect.fields ?? []).map((f) => (
        <WebsiteField key={f.key} label={labelText(t, f.label_key)} value={String(source.config?.[f.key] ?? '')} placeholder={placeholderText(t, f.placeholder, f.optional)}
          mono={f.key === 'url'} onSave={(v) => void patch({ config: { ...source.config, [f.key]: v } })} />
      ))}
    </div>
  ) : null;
  const catchup = source.catch_up && source.on ? (
    <span className="cu" data-testid="src-catchup">
      <button type="button" className="cub" onClick={() => void act(() => catchUp(source.id))}>
        {t('sources.catchUp.n', { n: source.catch_up.count })}<span>·</span><b>{t('sources.catchUp.go')}</b>
      </button>
      <button type="button" className="cux" title={t('sources.catchUp.skip')} aria-label={t('sources.catchUp.skip')} onClick={() => void act(() => dismissHeld(source.id))}>
        <X size={12} strokeWidth={2.2} aria-hidden="true" />
      </button>
    </span>
  ) : null;

  return (
    <div className="src-view is-in" data-testid="src-detail" data-id={source.id}>
      <div className="wb-ph hb">
        <button type="button" className="ib" onClick={onBack} title={t('sources.back')} aria-label={t('sources.back')} data-testid="src-back">
          <ChevronLeft size={16} strokeWidth={2.2} aria-hidden="true" />
        </button>
        <span className="pgl"><SourceMark type={source.type} size="md" /></span>
        <input
          className="pname"
          value={name}
          placeholder={sourceName(t, { ...source, name: '' })}
          aria-label={sourceName(t, source)}
          onChange={(e) => setName(e.target.value)}
          onBlur={saveName}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.nativeEvent.isComposing && e.keyCode !== 229) e.currentTarget.blur(); }}
          data-testid="src-name"
        />
        <Switch on={source.on} label={t('sources.toggle')} disabled={source.status === 'unavailable'} onChange={(on) => { putDetail({ ...source, on }); void patch({ on }); }} testId="src-detail-switch" />
        <button ref={moreRef} type="button" className={`ib${menu ? ' is-open' : ''}`} title={t('sources.more')} aria-label={t('sources.more')} aria-haspopup="menu"
          onClick={() => { setMenuAnchor(moreRef.current); setMenu((m) => (m ? null : 'more')); }} data-testid="src-more">
          <MoreHorizontal size={16} strokeWidth={2} aria-hidden="true" />
        </button>
      </div>
      <div className="src-body">
        <div className="sdet">
          {source.status === 'failed' && source.failure && (
            <FailLine source={source} typeDef={typeDef} checking={checking} onCheck={() => void check()} onReconnect={onReconnect}
              onRemove={(el) => { setMenuAnchor(el); setMenu('confirm'); }} />
          )}
          {source.registered_by && source.type !== 'command' && <RegisteredCard source={source} typeDef={typeDef} />}
          {isCli && typeDef && (
            <AccountField typeDef={typeDef} accounts={accounts} value={source.account} onPick={(login) => void patch({ account: login })} onCheck={onAccountsCheck} />
          )}
          {websiteCard}
          {source.type === 'command' && <CommandCard source={source} toast={toast} />}
          {pickCard}
          {kindsCard}
          <Instruction source={source} toast={toast} />
          {(rows.length > 0 || catchup) && (
            <div className="hist" data-testid="src-history">
              <div className="hlr"><span>{t('sources.recent')}</span>{catchup}</div>
              {rows.map((r) => r.kind === 'gap' ? (
                <div key={r.id} className="gap" title={t('sources.gap')} data-testid="src-gap">
                  <Unplug size={11} strokeWidth={2} aria-hidden="true" /><span>{gapLabel(r.entry, now, locale, tr)}</span>
                </div>
              ) : r.kind === 'wait' ? (
                <div key="wait" className="arv quiet" data-testid="src-wait">
                  <div className="arv-tx"><span className="t1">{r.title}</span><span className="t2"><span className="why">{t('sources.waiting', { n: r.count })}</span></span></div>
                  <span className="hx-time">{dateLabel(r.at, now, locale, tr)}</span>
                </div>
              ) : (
                <ItemRow key={r.id} source={source} row={r} projects={projects} hl={hl === r.id} onOpenSession={onOpenSession} toast={toast} />
              ))}
              {source.history_more && (
                <div className="older"><button type="button" className="tbtn" onClick={() => void loadOlder(source.id)}>{t('sources.older')}</button></div>
              )}
            </div>
          )}
        </div>
      </div>
      {menu && (
        <AnchoredMenu anchor={menuAnchor} onClose={() => setMenu(null)} width={menu === 'confirm' ? 300 : 210} alignRight testId="src-more-menu">
          {menu === 'confirm' ? (
            <div className="pconf">
              <span className="rc">{t('sources.delete.q')}</span>
              <button type="button" className="tbtn" onClick={() => setMenu(null)}>{t('sources.cancel')}</button>
              <button type="button" className="tbtn danger" onClick={() => void remove()} data-testid="src-delete-yes">{t('sources.delete')}</button>
            </div>
          ) : (
            <>
              {!isCli && source.label && source.type !== 'website' && (
                <div className="wb-opt is-static"><span className="wb-gl"><SourceMark type={source.type} size="md" /></span><span className="wb-tx"><span className="wb-n1">{source.label}</span></span></div>
              )}
              {formType && (
                <MenuOption glyph={<RotateCw size={14} strokeWidth={1.9} aria-hidden="true" />} label={t('sources.action.reconnect')} onSelect={() => { setMenu(null); onReconnect(); }} testId="src-menu-reconnect" />
              )}
              <MenuOption danger glyph={<Trash2 size={14} strokeWidth={1.9} aria-hidden="true" />} label={t('sources.delete')} onSelect={() => setMenu('confirm')} testId="src-menu-delete" />
            </>
          )}
        </AnchoredMenu>
      )}
    </div>
  );
}

function WebsiteField({ label, value, placeholder, mono, onSave }: {
  label: string; value: string; placeholder: string; mono: boolean; onSave: (v: string) => void;
}) {
  const [v, setV] = useState(value);
  useEffect(() => { setV(value); }, [value]);
  return (
    <div className="fl">
      <label className="cap">{label}</label>
      <div className="fin">
        <input className={`wfi${mono ? ' mono' : ''}`} value={v} placeholder={placeholder} spellCheck={false} autoComplete="off"
          onChange={(e) => setV(e.target.value)}
          onBlur={() => { if (v.trim() !== value) onSave(v.trim()); }}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.nativeEvent.isComposing && e.keyCode !== 229) e.currentTarget.blur(); }} />
      </div>
    </div>
  );
}
