// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Task sources — small pieces shared by the connect forms and a source's
 * view: the switch, an anchored menu (the Workbench's popover, in a portal),
 * the hint sign, the agent marks, the CLI account field and the card of
 * arrival kinds (rendered from the catalogue).
 */

import {
  useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode,
} from 'react';
import { createPortal } from 'react-dom';
import { ArrowUpRight, Check, ChevronDown, Copy, Folder, Info, Plus, Tag } from 'lucide-react';
import { useT } from '../../i18n/useT';
import { useCopyToClipboard } from '../../hooks/useCopyToClipboard';
import { getAgentIcon, MAIN_AGENT_HANDLE } from '../../utils/agentIcons';
import { placePopover } from '../workbench/ProjectPicker';
import SourceMark from './SourceMark';
import { accountLine, labelText } from './sourcesText';
import { kindsReady, setParam, toggleKind, type Kinds } from './formModel';
import type { AccountsResponse, CatalogParam, CatalogType, ChoiceItem } from './types';

export function Switch({ on, onChange, label, disabled, testId }: {
  on: boolean; onChange: (on: boolean) => void; label: string; disabled?: boolean; testId?: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      aria-label={label}
      title={label}
      className={`sw${on ? ' on' : ''}`}
      disabled={disabled}
      data-testid={testId}
      onClick={(e) => { e.stopPropagation(); onChange(!on); }}
    />
  );
}

export function Dots() {
  return <span className="dots" aria-hidden="true"><span /><span /><span /></span>;
}

/** A menu under (or above) its anchor, in a portal; closes on a press outside, Esc, scroll, resize. */
export function AnchoredMenu({ anchor, onClose, width = 240, children, testId, alignRight }: {
  anchor: HTMLElement | null; onClose: () => void; width?: number; children: ReactNode; testId?: string; alignRight?: boolean;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number; maxHeight: number | undefined } | null>(null);
  const closeRef = useRef(onClose);
  useLayoutEffect(() => { closeRef.current = onClose; });

  const place = useCallback(() => {
    const pop = hostRef.current?.firstElementChild as HTMLElement | null;
    if (!anchor || !pop) return;
    const r = anchor.getBoundingClientRect();
    const size = { w: pop.offsetWidth || width, h: pop.offsetHeight };
    const p = placePopover(r, size, { w: window.innerWidth, h: window.innerHeight }, 'below');
    if (alignRight) p.left = Math.max(8, Math.min(r.right - size.w, window.innerWidth - size.w - 8));
    setPos((cur) => (cur && cur.left === p.left && cur.top === p.top && cur.maxHeight === p.maxHeight ? cur : p));
  }, [anchor, width, alignRight]);

  useLayoutEffect(() => { place(); });

  useEffect(() => {
    const onPointer = (e: Event) => {
      const path = e.composedPath();
      if (hostRef.current && path.includes(hostRef.current)) return;
      if (anchor && path.includes(anchor)) return;
      closeRef.current();
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.stopPropagation(); closeRef.current(); } };
    const onScroll = (e: Event) => {
      if (hostRef.current && e.target instanceof Node && hostRef.current.contains(e.target)) return;
      closeRef.current();
    };
    const onResize = () => closeRef.current();
    document.addEventListener('pointerdown', onPointer);
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', onResize);
    return () => {
      document.removeEventListener('pointerdown', onPointer);
      window.removeEventListener('keydown', onKey, true);
      window.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', onResize);
    };
  }, [anchor]);

  const style: CSSProperties = pos
    ? { left: pos.left, top: pos.top, maxHeight: pos.maxHeight, width }
    : { left: -9999, top: -9999, width };
  return createPortal(
    <div ref={hostRef} className="wb-pophost" onClick={(e) => e.stopPropagation()} onMouseDown={(e) => e.stopPropagation()}>
      <div className="wb-pop is-anchored src-pop" role="menu" style={style} data-testid={testId}>{children}</div>
    </div>,
    document.body,
  );
}

export function MenuOption({ label, onSelect, on, glyph, note, danger, mono, testId }: {
  label: ReactNode; onSelect: () => void; on?: boolean; glyph?: ReactNode; note?: ReactNode; danger?: boolean; mono?: boolean; testId?: string;
}) {
  return (
    <button type="button" role="menuitemradio" aria-checked={!!on} className={`wb-opt${danger ? ' is-danger' : ''}`} onClick={onSelect} data-testid={testId}>
      {glyph !== undefined && <span className="wb-gl">{glyph}</span>}
      <span className="wb-tx"><span className={`wb-n1${mono ? ' mono' : ''}`}>{label}{note}</span></span>
      <span className="wb-ck">{on ? <Check size={14} strokeWidth={2.4} aria-hidden="true" /> : null}</span>
    </button>
  );
}

/** A needed hint, behind a small sign: hover on desktop, tap on a phone. */
export function Hint({ text }: { text: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const [at, setAt] = useState<{ left: number; top: number } | null>(null);
  const show = () => {
    const r = ref.current?.getBoundingClientRect();
    if (r) setAt({ left: Math.max(8, Math.min(r.left - 117, window.innerWidth - 258)), top: r.bottom + 6 });
  };
  return (
    <span
      ref={ref}
      className="src-hint"
      tabIndex={0}
      role="img"
      aria-label={text}
      onMouseEnter={show}
      onMouseLeave={() => setAt(null)}
      onFocus={show}
      onBlur={() => setAt(null)}
      onClick={(e) => { e.stopPropagation(); if (at) setAt(null); else show(); }}
    >
      <Info size={12} strokeWidth={2} aria-hidden="true" />
      {at && createPortal(<div className="src-tip" style={at}>{text}</div>, document.body)}
    </span>
  );
}

export function AgentLogo({ slug, label }: { slug: string | null; label?: string }) {
  const s = slug || MAIN_AGENT_HANDLE;
  const icon = getAgentIcon(s);
  return (
    <span className={`av${s === MAIN_AGENT_HANDLE ? ' is-orb' : ''}`} title={label}>
      {icon.src ? <img src={icon.src} alt={label ?? s} /> : null}
    </span>
  );
}

/** Where to get a CLI when this platform has no one-line install (the CLI's own page). */
export function InstallLink({ url, children }: { url: string; children?: ReactNode }) {
  const t = useT();
  return (
    <div className="accmd">
      <a className="lout" href={url} target="_blank" rel="noopener noreferrer" data-testid="src-install-link">
        {t('sources.cli.download')}<ArrowUpRight size={11} strokeWidth={2.2} aria-hidden="true" />
      </a>
      <span className="sp" />
      {children}
    </div>
  );
}

/** A command with a copy control (sign-in / install in Terminal). */
export function CommandRow({ cmd, children }: { cmd: string; children?: ReactNode }) {
  const t = useT();
  const { copied, copy, supported } = useCopyToClipboard(1400);
  return (
    <div className="accmd">
      <span className="mono">{cmd}</span>
      {supported && (
        <button type="button" className={`ib${copied ? ' ok' : ''}`} title={t(copied ? 'sources.copied' : 'sources.copy')} aria-label={t('sources.copy')} onClick={() => void copy(cmd)}>
          {copied ? <Check size={14} strokeWidth={2.2} aria-hidden="true" /> : <Copy size={14} strokeWidth={1.9} aria-hidden="true" />}
        </button>
      )}
      {children}
    </div>
  );
}

const AVATAR_COLORS = ['#539AF8', '#22A06B', '#E5893A', '#8B5CF6'];
export function Avatar({ login, index }: { login: string; index: number }) {
  return <span className="avt" style={{ background: AVATAR_COLORS[Math.max(0, index) % AVATAR_COLORS.length] }}>{(login[0] ?? '?').toUpperCase()}</span>;
}

/**
 * The account a GitHub / GitLab source uses: whose items arrive. Signed in:
 * the account, a picker of the CLI's accounts and 添加另一个账号. Signed out or
 * not installed: says so, with the command for Terminal and 检查 (re-reads).
 */
export function AccountField({ typeDef, accounts, value, onPick, onCheck, bad }: {
  typeDef: CatalogType;
  accounts: AccountsResponse | null;
  value: string | null;
  onPick: (login: string) => void;
  onCheck: () => Promise<void>;
  /** The check refused this account: the line under it. */
  bad?: string | null;
}) {
  const t = useT();
  const [rowEl, setRowEl] = useState<HTMLButtonElement | null>(null);
  const [open, setOpen] = useState(false);
  const [adding, setAdding] = useState(false);
  const [checking, setChecking] = useState(false);
  const host = typeDef.connect.host ?? '';
  const list = accounts?.accounts ?? [];
  const who = value ?? list.find((a) => a.active)?.login ?? list[0]?.login ?? null;
  const state = accounts?.state ?? null;
  const signedIn = state === 'signed_in' && !!who;
  const cmd = state === 'not_installed'
    ? accounts?.install_command || typeDef.connect.install_command || ''
    : accounts?.sign_in_command || typeDef.connect.sign_in_command || '';
  const installUrl = state === 'not_installed' && !cmd ? typeDef.connect.install_url ?? '' : '';
  const line = (login: string) => accountLine(login, list.find((a) => a.login === login)?.host ?? host);
  const whoLine = who ? line(who) : null;

  async function check() {
    setChecking(true);
    try { await onCheck(); } finally { setChecking(false); setAdding(false); }
  }

  return (
    <div className={`pk acf${bad ? ' bad' : ''}`} data-testid="src-account">
      <div className="cap">{t('sources.account.cap')}</div>
      {signedIn ? (
        <button ref={setRowEl} type="button" className={`acrow${open ? ' is-open' : ''}`} aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen((o) => !o)} data-testid="src-account-row">
          <Avatar login={who!} index={list.findIndex((a) => a.login === who)} />
          <span className="acn"><span className="l">{whoLine!.name}</span>{whoLine!.host && <span className="h">{whoLine!.host}</span>}</span>
          <span className="cv"><ChevronDown size={12} strokeWidth={2.2} aria-hidden="true" /></span>
        </button>
      ) : (
        <div className="acrow off" data-testid="src-account-off">
          <span className="avt none"><SourceMark type={typeDef.type} size="sm" /></span>
          <span className="acn">
            <span className="l">{state === null ? '…' : t(state === 'not_installed' ? 'sources.cli.missing' : 'sources.cli.signedOut')}</span>
            <span className="h">{host}</span>
          </span>
        </div>
      )}
      {state !== null && (!signedIn || adding) && (cmd || installUrl) && (() => {
        const checkBtn = (
          <button type="button" className="tbtn acc" onClick={() => void check()} disabled={checking} data-testid="src-account-check">
            {checking ? <Dots /> : t('sources.account.check')}
          </button>
        );
        return cmd ? <CommandRow cmd={cmd}>{checkBtn}</CommandRow> : <InstallLink url={installUrl}>{checkBtn}</InstallLink>;
      })()}
      {bad && <div className="ferr" role="alert" data-testid="src-field-error">{bad}</div>}
      {open && (
        <AnchoredMenu anchor={rowEl} onClose={() => setOpen(false)} width={280} testId="src-account-menu">
          {list.map((a, i) => (
            <MenuOption
              key={a.login}
              glyph={<Avatar login={a.login} index={i} />}
              label={(() => { const l = line(a.login); return <>{l.name}{l.host && <span className="src-oh">{l.host}</span>}</>; })()}
              on={a.login === who}
              onSelect={() => { setOpen(false); setAdding(false); onPick(a.login); }}
            />
          ))}
          <div className="wb-sep" />
          <MenuOption glyph={<Plus size={14} strokeWidth={2} aria-hidden="true" />} label={t('sources.account.add')} onSelect={() => { setOpen(false); setAdding(true); }} />
        </AnchoredMenu>
      )}
    </div>
  );
}

type LoadChoices = (list: string, params: Record<string, unknown>) => Promise<ChoiceItem[]>;

/** A param picker: values the service lists (never free text), or fixed options. */
function ParamChip({ typeDef, kinds, kind, param, onChange, loadChoices, bad }: {
  typeDef: CatalogType; kinds: Kinds; kind: string; param: CatalogParam;
  onChange: (k: Kinds) => void; loadChoices: LoadChoices; bad?: boolean;
}) {
  const t = useT();
  const [chipEl, setChipEl] = useState<HTMLButtonElement | null>(null);
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<ChoiceItem[] | null>(null);
  const [failed, setFailed] = useState(false);
  const val = kinds[kind]?.[param.key];
  const shown = val === undefined || val === null || val === '' ? null : String(val);

  async function openMenu() {
    if (open) { setOpen(false); return; }
    setOpen(true);
    if (param.options) { setItems(param.options.map((o) => ({ value: String(o), label: String(o) }))); return; }
    if (!param.choices) return;
    setItems(null); setFailed(false);
    try {
      setItems(await loadChoices(param.choices, kinds[kind] ?? {}));
    } catch { setFailed(true); setItems([]); }
  }
  const icon = param.choices === 'labels' ? <Tag size={12} strokeWidth={2} aria-hidden="true" /> : <Folder size={13} strokeWidth={1.9} aria-hidden="true" />;
  const mono = param.choices === 'repos' || param.choices === 'labels';
  return (
    <>
      <button ref={setChipEl} type="button" className={`chip${shown ? ' is-set' : ''}${open ? ' is-open' : ''}${bad ? ' is-bad' : ''}`} aria-invalid={bad || undefined} onClick={() => void openMenu()} data-testid={`src-param-${param.key}`}>
        <span className="ic">{icon}</span>
        <span className={`nm${shown && mono ? ' mono' : ''}`}>{shown ?? labelText(t, param.label_key)}</span>
        <span className="cv"><ChevronDown size={11} strokeWidth={2.2} aria-hidden="true" /></span>
      </button>
      {open && (
        <AnchoredMenu anchor={chipEl} onClose={() => setOpen(false)} width={mono ? 260 : 220} testId={`src-param-menu-${param.key}`}>
          {items === null ? <div className="src-pop-empty">…</div>
            : items.length === 0 ? <div className="src-pop-empty">{failed ? t('sources.error.action') : '—'}</div>
              : items.map((it) => (
                <MenuOption
                  key={it.value}
                  glyph={param.choices === 'labels' ? undefined : <Folder size={13} strokeWidth={1.9} aria-hidden="true" />}
                  label={it.label}
                  mono={mono}
                  on={shown === it.value}
                  onSelect={() => { setOpen(false); onChange(setParam(typeDef, kinds, kind, param.key, param.options ? coerce(param, it.value) : it.value)); }}
                />
              ))}
        </AnchoredMenu>
      )}
    </>
  );
}

const coerce = (p: CatalogParam, v: string) => (p.options && typeof p.options[0] === 'number' ? Number(v) : v);

/** The number inside a kind's pill («开始前 30 分钟»): a fixed-options param. */
function InlineNumber({ typeDef, kinds, kind, param, onChange }: {
  typeDef: CatalogType; kinds: Kinds; kind: string; param: CatalogParam; onChange: (k: Kinds) => void;
}) {
  const t = useT();
  const [numEl, setNumEl] = useState<HTMLSpanElement | null>(null);
  const [open, setOpen] = useState(false);
  const cur = kinds[kind]?.[param.key] ?? param.default;
  return (
    <>
      <span
        ref={setNumEl}
        role="button"
        tabIndex={0}
        className={`num${open ? ' is-open' : ''}`}
        onClick={(e) => { e.stopPropagation(); setOpen((o) => !o); }}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); setOpen((o) => !o); } }}
        data-testid={`src-num-${param.key}`}
      >{String(cur ?? '')}</span>
      {open && (
        <AnchoredMenu anchor={numEl} onClose={() => setOpen(false)} width={150}>
          {(param.options ?? []).map((o) => (
            <MenuOption
              key={String(o)}
              label={t('sources.mins', { n: String(o) })}
              on={String(cur) === String(o)}
              onSelect={() => {
                setOpen(false);
                const on = kinds[kind] ? kinds : toggleKind(typeDef, kinds, kind);
                onChange(setParam(typeDef, on, kind, param.key, o));
              }}
            />
          ))}
        </AnchoredMenu>
      )}
    </>
  );
}

/**
 * What counts as an arrival, under «收进工作台» with whose items they are.
 * Several kinds: one pill each (the last one stays on), their params as
 * pickers under them. One kind: its params are the controls (e-mail: folder
 * + unread only). One kind and no params (the website check): nothing.
 */
export function KindsCard({ typeDef, kinds, onChange, who, loadChoices, bad }: {
  typeDef: CatalogType; kinds: Kinds; onChange: (k: Kinds) => void;
  who?: { label: string } | null; loadChoices: LoadChoices;
  /** A refusal that names one of the params (repo, label, folder). */
  bad?: { field: string; text: string } | null;
}) {
  const t = useT();
  const single = typeDef.kinds.length === 1 ? typeDef.kinds[0] : null;
  if (single && !single.params.length) return null;
  const isBad = (p: CatalogParam) => bad?.field === p.key;
  const badLine = bad && typeDef.kinds.some((k) => k.params.some(isBad))
    ? <div className="ferr" role="alert" data-testid="src-field-error">{bad.text}</div> : null;
  const cap = (
    <div className="cap">
      <span>{t('sources.pick.cap')}</span>
      <span className="sp" />
      {who?.label && <span className="who"><SourceMark type={typeDef.type} size="sm" /><span className="t">{who.label}</span></span>}
    </div>
  );
  const boolPill = (kind: string, p: CatalogParam) => {
    const on = !!(kinds[kind]?.[p.key] ?? p.default);
    return (
      <button key={p.key} type="button" className={`pill2${on ? ' on' : ''}`} aria-pressed={on} onClick={() => onChange(setParam(typeDef, kinds, kind, p.key, !on))} data-testid={`src-param-${p.key}`}>
        {on && <Check size={12} strokeWidth={2.4} aria-hidden="true" />}<span>{labelText(t, p.label_key)}</span>
      </button>
    );
  };
  if (single) {
    return (
      <div className="pk" data-testid="src-kinds">
        {cap}
        <div className="pills2">
          {single.params.map((p) => p.type === 'bool' ? boolPill(single.key, p)
            : <ParamChip key={p.key} typeDef={typeDef} kinds={kinds} kind={single.key} param={p} onChange={onChange} loadChoices={loadChoices} bad={isBad(p)} />)}
        </div>
        {badLine}
      </div>
    );
  }
  const foot: ReactNode[] = [];
  for (const k of typeDef.kinds) {
    if (!kinds[k.key]) continue;
    for (const p of k.params) {
      if (p.options && labelHasN(t, k.label_key)) continue;
      if (p.depends_on && !kinds[k.key]?.[p.depends_on]) continue;
      foot.push(p.type === 'bool' ? boolPill(k.key, p)
        : <ParamChip key={`${k.key}.${p.key}`} typeDef={typeDef} kinds={kinds} kind={k.key} param={p} onChange={onChange} loadChoices={loadChoices} bad={isBad(p)} />);
    }
  }
  return (
    <div className="pk" data-testid="src-kinds" data-ready={kindsReady(typeDef, kinds) ? 'true' : 'false'}>
      {cap}
      <div className="pills2">
        {typeDef.kinds.map((k) => {
          const on = !!kinds[k.key];
          const inline = k.params.find((p) => p.options) && labelHasN(t, k.label_key) ? k.params.find((p) => p.options)! : null;
          const label = inline
            ? splitN(labelText(t, k.label_key, { n: '\u0000' }), <InlineNumber typeDef={typeDef} kinds={kinds} kind={k.key} param={inline} onChange={onChange} />)
            : labelText(t, k.label_key);
          return (
            <button key={k.key} type="button" className={`pill2${on ? ' on' : ''}`} aria-pressed={on} onClick={() => onChange(toggleKind(typeDef, kinds, k.key))} data-testid={`src-kind-${k.key}`}>
              {on && <Check size={12} strokeWidth={2.4} aria-hidden="true" />}<span>{label}</span>
            </button>
          );
        })}
      </div>
      {foot.length > 0 && <div className="pk-foot">{foot}</div>}
      {badLine}
    </div>
  );
}

const labelHasN = (t: ReturnType<typeof useT>, key: string) => labelText(t, key, { n: '\u0000' }).includes('\u0000');
function splitN(text: string, node: ReactNode): ReactNode {
  const [a, b] = text.split('\u0000');
  return <>{a}{node}{b}</>;
}
