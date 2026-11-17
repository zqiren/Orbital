// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Task sources — a connect form, drawn from the type's catalogue entry, and
 * the same form for 重新连接 (secrets blank, settings and history kept).
 *
 * Order (prototype `detailView` draft + `credBlock`): 交给 Agent 设置 (types
 * with fields), the one setup link, the fields (secrets masked, an eye to
 * show), or the QR scan with its fallback to fields; the CLI account; what
 * counts as an arrival (calendar: which calendars above it, all ticked). One primary button at the foot, 添加并开启: nothing
 * exists until the daemon's check passes; a refusal marks the field it
 * failed on.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowUpRight, Check, ChevronDown, ChevronLeft, ChevronUp, Copy, Eye, EyeOff, X } from 'lucide-react';
import { QRCodeSVG } from 'qrcode.react';
import { useT } from '../../i18n/useT';
import { useLocale } from '../../i18n/LocaleContext';
import { useWebSocket } from '../../hooks/useWebSocket';
import type { WebSocketEvent } from '../../types';
import { useCopyToClipboard } from '../../hooks/useCopyToClipboard';
import SourceMark from './SourceMark';
import { AccountField, AgentLogo, Dots, KindsCard, PickCard } from './parts';
import {
  createBody, credentials, draftReady, fieldRows, formFields, initDraft, markedField, pickOf, pickState, setField, togglePick,
  usesQr, verifyMessageKey, type Draft,
} from './formModel';
import {
  createSource, fetchChoices, fetchQr, openSettingsPane, reconnectSource, sourceError, startQr,
} from './sourcesApi';
import { putDetail } from './sourcesStore';
import { cjkSpacing, hasKey, labelText, placeholderText, typeName } from './sourcesText';
import type { AccountsResponse, CatalogField, CatalogType, ChoiceItem, SourceDetail, SourceError } from './types';

interface VerifyState { st: 'checking' | 'fail'; field: string | null; text: string | null; reason?: string | null }

export interface ConnectFormProps {
  typeDef: CatalogType;
  /** Reconnect an existing source: the same fields, the same check. */
  source?: SourceDetail | null;
  accounts: AccountsResponse | null;
  onAccountsCheck: () => Promise<void>;
  onDone: (d: SourceDetail) => void;
  onBack: () => void;
  onClose: () => void;
}

/** 交给 Agent 设置: peek at the prompt, copy it; the page runs nothing. */
function AgentPromptCard({ prompt }: { prompt: string }) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const { copied, copy, supported } = useCopyToClipboard(1800);
  return (
    <>
      <div className="pk agc" data-testid="src-agent-card">
        <div className="agh">
          <span className="agm">
            {['claude-code', 'codex', 'orbital'].map((s) => <AgentLogo key={s} slug={s} />)}
          </span>
          <span className="agt">{t('sources.agent.title')}</span>
          <span className="sp" />
          <button type="button" className="tbtn agp" aria-expanded={open} onClick={() => setOpen((o) => !o)} data-testid="src-agent-peek">
            {t(open ? 'sources.agent.hide' : 'sources.agent.peek')}
            {open ? <ChevronUp size={10} strokeWidth={2.4} aria-hidden="true" /> : <ChevronDown size={10} strokeWidth={2.4} aria-hidden="true" />}
          </button>
          {supported && (
            <button type="button" className={`agb${copied ? ' ok' : ''}`} onClick={() => void copy(prompt)} data-testid="src-agent-copy">
              {copied ? <Check size={12} strokeWidth={2.6} aria-hidden="true" /> : <Copy size={12} strokeWidth={2} aria-hidden="true" />}
              <span>{t(copied ? 'sources.agent.copied' : 'sources.agent.copy')}</span>
            </button>
          )}
        </div>
        {open && <div className="ptxt" data-testid="src-agent-prompt">{prompt}</div>}
      </div>
      <div className="agor"><span>{t('sources.agent.or')}</span></div>
    </>
  );
}

function FieldInput({ typeDef, field, draft, bad, onChange }: {
  typeDef: CatalogType; field: CatalogField; draft: Draft; bad: string | null; onChange: (k: string, v: string) => void;
}) {
  const t = useT();
  const [shown, setShown] = useState(false);
  const v = (field.secret ? draft.secrets[field.key] : draft.config[field.key]) ?? '';
  const mono = !['acct', 'mail', 'must_contain'].includes(field.key);
  const id = `src-f-${typeDef.type}-${field.key}`;
  const ph = placeholderText(t, field.placeholder, field.optional);
  return (
    <div className={`fl${bad ? ' bad' : ''}`}>
      <label className="cap" htmlFor={id}>{labelText(t, field.label_key)}</label>
      <div className="fin">
        <input
          id={id}
          className={`wfi${mono ? ' mono' : ''}`}
          type={field.secret && !shown ? 'password' : 'text'}
          value={v}
          placeholder={ph}
          spellCheck={false}
          autoComplete="off"
          autoCapitalize="off"
          data-testid={`src-field-${field.key}`}
          aria-invalid={bad ? true : undefined}
          onChange={(e) => onChange(field.key, e.target.value)}
          onKeyDown={(e) => {
            if (e.key !== 'Enter' || e.nativeEvent.isComposing || e.keyCode === 229) return;
            e.preventDefault(); e.currentTarget.blur();
          }}
        />
        {field.secret && (
          <button type="button" className="eye" title={t(shown ? 'sources.hide' : 'sources.show')} aria-label={t(shown ? 'sources.hide' : 'sources.show')} onClick={() => setShown((s) => !s)}>
            {shown ? <EyeOff size={14} strokeWidth={1.8} aria-hidden="true" /> : <Eye size={14} strokeWidth={1.8} aria-hidden="true" />}
          </button>
        )}
      </div>
      {bad && <div className="ferr" role="alert" data-testid="src-field-error">{bad}</div>}
    </div>
  );
}

function FieldsCard({ typeDef, draft, verify, onChange }: {
  typeDef: CatalogType; draft: Draft; verify: VerifyState | null; onChange: (k: string, v: string) => void;
}) {
  const bad = (k: string) => (verify?.st === 'fail' && verify.field === k ? verify.text : null);
  return (
    <div className="pk wf" data-testid="src-fields">
      {fieldRows(formFields(typeDef)).map((r) => Array.isArray(r) ? (
        <div className="fpair" key={r[0].key}>
          {r.map((f) => <FieldInput key={f.key} typeDef={typeDef} field={f} draft={draft} bad={bad(f.key)} onChange={onChange} />)}
        </div>
      ) : <FieldInput key={r.key} typeDef={typeDef} field={r} draft={draft} bad={bad(r.key)} onChange={onChange} />)}
    </div>
  );
}

/** Feishu: scanning creates the bot app for the user. pending → done (or expired / failed: tap for a new code). */
function QrCard({ typeDef, draft, setDraft, onUnavailable }: {
  typeDef: CatalogType; draft: Draft; setDraft: (f: (d: Draft) => Draft) => void; onUnavailable: () => void;
}) {
  const t = useT();
  const ws = useWebSocket();
  const [text, setText] = useState<string | null>(null);
  const [label, setLabel] = useState<string | null>(null);
  const [round, setRound] = useState(0);
  const qrId = draft.qrId;
  const state = draft.qrState;
  const unavailableRef = useRef(onUnavailable);
  useEffect(() => { unavailableRef.current = onUnavailable; });

  useEffect(() => {
    let dead = false;
    startQr(typeDef.type)
      .then((q) => { if (!dead) { setText(q.qr_text); setDraft((d) => ({ ...d, qrId: q.qr_id, qrState: q.state })); } })
      .catch(() => { if (!dead) unavailableRef.current(); });
    return () => { dead = true; };
  }, [typeDef.type, round, setDraft]);

  // The scan's progress: pushed (sources.qr) and polled, whichever is first.
  useEffect(() => {
    if (!qrId || state === 'done' || state === 'expired' || state === 'failed') return;
    const apply = (st: Draft['qrState'], lb?: string | null) => {
      if (lb) setLabel(lb);
      setDraft((d) => (d.qrId === qrId && d.qrState !== st ? { ...d, qrState: st } : d));
    };
    const onEvent = (e: WebSocketEvent) => {
      const ev = e as { qr_id?: string; state?: Draft['qrState']; label?: string | null };
      if (ev.qr_id === qrId && ev.state) apply(ev.state, ev.label);
    };
    ws.on('sources.qr', onEvent);
    const id = window.setInterval(() => {
      fetchQr(qrId).then((q) => apply(q.state, q.label)).catch(() => {});
    }, 1500);
    return () => { ws.off('sources.qr', onEvent); window.clearInterval(id); };
  }, [qrId, state, ws, setDraft]);

  if (state === 'done') {
    return (
      <div className="pk qrc" data-testid="src-qr" data-state="done">
        <div className="qr-done">
          <SourceMark type={typeDef.type} size="xl" />
          <span className="t"><Check size={14} strokeWidth={2.4} aria-hidden="true" />{label ?? typeName(t, typeDef.type)}</span>
        </div>
      </div>
    );
  }
  const dead = state === 'expired' || state === 'failed';
  return (
    <div className="pk qrc" data-testid="src-qr" data-state={state ?? 'loading'}>
      <button
        type="button"
        className={`qr${dead ? ' expired' : ''}`}
        onClick={() => { if (dead) setRound((r) => r + 1); }}
        aria-label={t('sources.qr.scan')}
      >
        {text ? <QRCodeSVG value={text} size={146} level="M" bgColor="#ffffff" fgColor="#111111" /> : <Dots />}
      </button>
      <div className="qr-cap">
        {t(dead ? 'sources.qr.expired' : 'sources.qr.scan')}
      </div>
    </div>
  );
}

export default function ConnectForm({ typeDef, source, accounts, onAccountsCheck, onDone, onBack, onClose }: ConnectFormProps) {
  const t = useT();
  const { locale } = useLocale();
  const reconnect = !!source;
  const [draft, setDraftState] = useState<Draft>(() => {
    const d = initDraft(typeDef);
    if (!source) return d;
    // Reconnect: the non-secret fields kept, secrets blank.
    const config: Record<string, string> = { ...d.config };
    for (const f of formFields(typeDef)) if (!f.secret && source.config?.[f.key] != null) config[f.key] = String(source.config[f.key]);
    return { ...d, kinds: source.kinds, account: source.account, config };
  });
  const setDraft = useCallback((f: (d: Draft) => Draft) => setDraftState(f), []);
  const [verify, setVerify] = useState<VerifyState | null>(null);
  const svc = typeName(t, typeDef.type);
  const method = typeDef.connect.method;
  const qr = usesQr(typeDef, draft);
  const pick = reconnect ? null : pickOf(typeDef);
  const [pickItems, setPickItems] = useState<ChoiceItem[] | null>(null);
  const pickReq = useRef<Promise<ChoiceItem[]> | null>(null);

  // A CLI type defaults to the CLI's active account (until one is picked).
  const defaultLogin = method === 'cli_account' && accounts?.state === 'signed_in'
    ? (accounts.accounts.find((x) => x.active) ?? accounts.accounts[0])?.login ?? null
    : null;
  const effective: Draft = useMemo(
    () => (draft.account || !defaultLogin ? draft : { ...draft, account: defaultLogin }), [draft, defaultLogin]);

  // A change to what the refusal named clears its line.
  const clearIf = (k: string) => { if (verify?.st === 'fail' && verify.field === k) setVerify(null); };

  function onField(k: string, v: string) {
    setDraftState((d) => setField(typeDef, d, k, v));
    if (verify?.st === 'fail' && (verify.field === k || verify.field === null)) setVerify(null);
  }

  /** A refusal (connect, or reading the list) as the line it shows. */
  const failState = (err: SourceError): VerifyState => {
    const field = markedField(typeDef, err.field);
    const secret = !!formFields(typeDef).find((f) => f.key === field)?.secret;
    const key = verifyMessageKey(typeDef.type, err.reason ?? (err.code === 'invalid' ? 'invalid' : null), secret);
    return { st: 'fail', field, text: hasKey(key) ? cjkSpacing(t(key, { svc }), svc, t(key)) : t('sources.verify.error'), reason: err.reason };
  };

  // Calendar: its calendars, all ticked. Opening the form is the user's
  // gesture, so reading them may ask macOS for access (once). One read per
  // form, also under StrictMode's double effect.
  useEffect(() => {
    if (!pick) return;
    let dead = false;
    pickReq.current ??= fetchChoices({ type: typeDef.type, list: pick.choices, params: {} }).then((r) => r.items ?? []);
    pickReq.current.then((items) => {
      if (dead) return;
      setPickItems(items);
      setDraftState((d) => ({ ...d, pick: pickState(null, items) }));
    }).catch((e) => {
      if (dead) return;
      // Unreadable (no access): no list; connecting means every calendar, and asks again.
      setPickItems([]);
      setDraftState((d) => ({ ...d, pick: null }));
      setVerify(failState(sourceError(e)));
    });
    return () => { dead = true; };
  }, [pick, typeDef.type]); // eslint-disable-line react-hooks/exhaustive-deps

  const loadChoices = useCallback(async (list: string, params: Record<string, unknown>) => {
    const { config, secrets } = credentials(typeDef, effective);
    const res = await fetchChoices({ type: typeDef.type, list, account: effective.account, params, config, secrets });
    return res.items ?? [];
  }, [typeDef, effective]);

  const ready = draftReady(typeDef, effective) && verify?.st !== 'checking';

  async function submit() {
    if (!ready) return;
    setVerify({ st: 'checking', field: null, text: null });
    try {
      let d: SourceDetail;
      if (source) {
        const { config, secrets } = credentials(typeDef, draft);
        d = await reconnectSource(source.id, { config, secrets, qr_id: qr ? draft.qrId : null });
      } else {
        d = await createSource(createBody(typeDef, effective));
      }
      putDetail(d);
      setVerify(null);
      onDone(d);
    } catch (e) {
      setVerify(failState(sourceError(e)));
    }
  }

  const prompt = !reconnect && (method === 'fields' || method === 'qr') && typeDef.family !== 'check' && typeDef.agent_prompt
    ? (locale === 'zh' ? typeDef.agent_prompt.zh : typeDef.agent_prompt.en) || typeDef.agent_prompt.en
    : null;
  const linkKey = `sources.lk.${typeDef.type}`;
  const link = typeDef.setup_link && !qr ? (
    <a className="lout" href={typeDef.setup_link} target="_blank" rel="noopener noreferrer" data-testid="src-setup-link">
      {hasKey(linkKey) ? t(linkKey) : t('sources.lk.default')}<ArrowUpRight size={11} strokeWidth={2.2} aria-hidden="true" />
    </a>
  ) : null;
  const formError = verify?.st === 'fail' && !verify.field ? verify.text : null;

  return (
    <div className="src-view is-in" data-testid={reconnect ? 'src-reconnect' : 'src-connect'} data-type={typeDef.type}>
      <div className="wb-ph hb">
        <button type="button" className="ib" onClick={onBack} title={t(reconnect ? 'sources.backForm' : 'sources.back')} aria-label={t(reconnect ? 'sources.backForm' : 'sources.back')} data-testid="src-back">
          <ChevronLeft size={16} strokeWidth={2.2} aria-hidden="true" />
        </button>
        <span className="pgl"><SourceMark type={typeDef.type} size="md" /></span>
        {reconnect ? <span className="pt">{source?.name || svc}</span> : (
          <input
            className="pname"
            value={draft.name}
            placeholder={svc}
            aria-label={svc}
            onChange={(e) => setDraftState((d) => ({ ...d, name: e.target.value }))}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.nativeEvent.isComposing && e.keyCode !== 229) e.currentTarget.blur(); }}
          />
        )}
        <button type="button" className="ib" onClick={onClose} title={t('sources.close')} aria-label={t('sources.close')}>
          <X size={16} strokeWidth={1.9} aria-hidden="true" />
        </button>
      </div>
      <div className="src-body">
        <div className="sdet">
          {prompt && <AgentPromptCard prompt={prompt} />}
          {method === 'qr' && qr && (
            <>
              <QrCard typeDef={typeDef} draft={draft} setDraft={setDraft} onUnavailable={() => setDraftState((d) => ({ ...d, manual: true }))} />
              {draft.qrState !== 'done' && (
                <div className="alt">
                  <button type="button" className="tbtn" onClick={() => setDraftState((d) => ({ ...d, manual: true, qrState: null, qrId: null }))} data-testid="src-qr-manual">{t('sources.qr.manual')}</button>
                </div>
              )}
            </>
          )}
          {(method === 'fields' || (method === 'qr' && !qr)) && (
            <>
              {link}
              <FieldsCard typeDef={typeDef} draft={draft} verify={verify} onChange={onField} />
              {method === 'qr' && (
                <div className="alt">
                  <button type="button" className="tbtn" onClick={() => { setVerify(null); setDraftState((d) => ({ ...d, manual: false })); }} data-testid="src-qr-again">{t('sources.qr.scanAgain')}</button>
                </div>
              )}
            </>
          )}
          {method === 'cli_account' && (
            <AccountField
              typeDef={typeDef}
              accounts={accounts}
              value={effective.account}
              onPick={(login) => { clearIf('account'); setDraftState((d) => ({ ...d, account: login })); }}
              onCheck={onAccountsCheck}
              bad={verify?.st === 'fail' && verify.field === 'account' ? verify.text : null}
            />
          )}
          {pick && (
            <PickCard
              items={pickItems}
              on={draft.pick && draft.pick !== 'loading' ? draft.pick.on : []}
              label={t('sources.calendar.list')}
              onToggle={(v) => setDraftState((d) => (d.pick && d.pick !== 'loading' ? { ...d, pick: togglePick(d.pick, v) } : d))}
            />
          )}
          {!reconnect && (
            <KindsCard
              typeDef={typeDef}
              kinds={draft.kinds}
              onChange={(kinds) => { if (verify?.st === 'fail' && verify.field) setVerify(null); setDraftState((d) => ({ ...d, kinds })); }}
              loadChoices={loadChoices}
              bad={verify?.st === 'fail' && verify.field && verify.text ? { field: verify.field, text: verify.text } : null}
            />
          )}
        </div>
      </div>
      <div className="src-foot">
        {formError && (
          <div className="ferr" role="alert" data-testid="src-form-error">
            {formError}
            {verify?.reason === 'permission' && (
              <button type="button" className="tbtn acc" onClick={() => void openSettingsPane()} data-testid="src-form-settings">{t('sources.action.open_settings')}</button>
            )}
          </div>
        )}
        <button
          type="button"
          className={`go${verify?.st === 'checking' ? ' busy' : ''}`}
          disabled={!ready}
          onClick={() => void submit()}
          data-testid="src-submit"
        >
          {verify?.st === 'checking' ? <Dots /> : t(reconnect ? 'sources.connect' : 'sources.addOn')}
        </button>
      </div>
    </div>
  );
}
