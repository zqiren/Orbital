// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Connect a model provider (batch 1123, owner-approved "option C").
 *
 * One screen: logo tiles, then the picked provider's panel below them.
 *
 *  - Key providers: one key field. The connection is tested on PASTE, on
 *    ENTER (not while an IME composes), when the field LOSES FOCUS with an
 *    untested key, and on Next / Save. Typing alone never tests. Next / Save
 *    is enabled whenever a key is present; pressed with an untested key it
 *    tests first and only advances on a pass. A failure stays under the
 *    field (short reason, the provider's sentence behind a sign); editing
 *    the key clears it. A click that blurs the field (Next, a tile, the
 *    region toggle) lands first; the blur test runs after it.
 *  - Two-region providers carry a 中国 / 国际 toggle inside the key field.
 *    Switching marks the key untested and hides the model row.
 *  - After a pass, ONE model row appears under the key in the same frame,
 *    preselected from the live list; its list opens inside the card.
 *  - One-click sign-ins (ChatGPT, TokenDance): one button → waiting (reopen,
 *    cancel) → account row + model row. The daemon stores that card itself,
 *    so Next / Save afterwards only writes a changed model.
 *
 * A key is stored only when its test passes (the daemon enforces it too).
 * There is no <form> here: the panel renders inside the wizard and inside a
 * Settings dialog that may sit under another form, and a nested form
 * swallows submit. Enter is handled on the fields themselves.
 */
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { Ban, CircleCheck, KeyRound, Loader2, LogOut, RotateCw, Server } from 'lucide-react';
import type { CredentialCard, LLMSdk, ProviderCheckResult } from '../../types';
import { api } from '../../config';
import { useT } from '../../i18n/useT';
import { useLocale } from '../../i18n/LocaleContext';
import HintSign from '../HintSign';
import Select from '../Select';
import ProviderTiles from './ProviderTiles';
import ModelRow from './ModelRow';
import GetKeyLink from './GetKeyLink';
import ProviderLogo from './ProviderLogo';
import {
  cancelSignIn,
  checkConnection,
  deleteCardById,
  reopenSignIn,
  saveCard,
  signIn,
  type Failure,
  type SigninKind,
} from './connectApi';
import { CHATGPT, consoleUrl, CUSTOM, hasRegions, initialRegion, type Region } from './providerMeta';
import { shortReason } from './failure';
import { useCatalog } from './useCatalog';

export interface ConnectFooterCtx {
  /** Next / Save may be pressed. */
  canSubmit: boolean;
  /** Testing-then-saving, or saving. */
  busy: boolean;
  submit: () => void;
  /** A refusal that is not about the key (it is drawn under the field). */
  error: string | null;
}

export interface ConnectPanelProps {
  /** Edit this saved card: no tiles, the provider is fixed. */
  card?: CredentialCard | null;
  /** Wizard: save onto this (keyless default) card instead of creating one. */
  replaceCardId?: string | null;
  /** Start "Sign in again" at once (a revoked sign-in row's action). */
  autoSignIn?: boolean;
  /** A card was stored: by a save, or daemon-side by a sign-in. */
  onCardStored?: (card: CredentialCard) => void;
  /** A signed-in card was signed out (deleted daemon-side). */
  onCardRemoved?: (cardId: string) => void;
  /** Next / Save finished; the card is ready to use. */
  onDone: (card: CredentialCard) => void;
  /** Next / Save and anything around it. */
  footer: (ctx: ConnectFooterCtx) => ReactNode;
}

type KeyStatus = 'idle' | 'testing' | 'ok' | 'fail';
type SigninState = 'idle' | 'waiting' | 'ok' | 'refused' | 'error';

/** Sign-in refusals that mean "this account cannot be used here". */
const REFUSED_CODES = new Set(['not_eligible', 'plan_usage_not_granted']);

function FailLine({ failure, testId }: { failure: Failure; testId: string }) {
  const t = useT();
  return (
    <p className="mt-2 flex items-center gap-1.5 text-body text-error-text" data-testid={testId} role="alert">
      <span>{shortReason(failure, t)}</span>
      {failure.message && (
        <HintSign label={t('connect.fail.details')} testId={`${testId}-sign`}>
          <span className="block max-w-[260px] break-words text-primary">{failure.message}</span>
        </HintSign>
      )}
    </p>
  );
}

function RegionToggle({ value, onChange }: { value: Region; onChange: (r: Region) => void }) {
  const t = useT();
  return (
    <div
      role="group"
      aria-label={t('llm.field.region')}
      className="inline-flex shrink-0 gap-0.5 rounded-[8px] bg-card-hover p-0.5"
      data-testid="connect-region"
    >
      {(['china', 'global'] as const).map((r) => (
        <button
          type="button"
          key={r}
          aria-pressed={value === r}
          data-testid={`connect-region-${r}`}
          onClick={() => onChange(r)}
          className={`h-6 whitespace-nowrap rounded-[6px] px-2 text-caption ${
            value === r
              ? 'bg-segment-on text-segment-on-text shadow-[0_1px_2px_rgb(0_0_0/0.08)]'
              : 'text-secondary hover:text-primary'
          }`}
        >
          {r === 'china' ? t('connect.region.china') : t('connect.region.global')}
        </button>
      ))}
    </div>
  );
}

/** Model ids from a pass, and the one to preselect. */
function pickModel(models: string[], preferred: string | null | undefined, fallback: string | null | undefined): string {
  if (preferred && models.includes(preferred)) return preferred;
  if (fallback && (models.length === 0 || models.includes(fallback))) return fallback;
  return models[0] ?? fallback ?? '';
}

function labelsOf(r: ProviderCheckResult | { models: string[]; model_details?: ProviderCheckResult['model_details'] }) {
  const out: Record<string, string> = {};
  for (const d of r.model_details ?? []) if (d.display_name) out[d.id] = d.display_name;
  return out;
}

export default function ConnectPanel({ card, replaceCardId, autoSignIn, onCardStored, onCardRemoved, onDone, footer }: ConnectPanelProps) {
  const t = useT();
  const { locale } = useLocale();
  const catalog = useCatalog();
  const editing = !!card;

  // ---- what is picked ----
  const [provider, setProvider] = useState('');
  const [method, setMethod] = useState<'key' | 'signin'>('key');
  const [moreOpen, setMoreOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [region, setRegion] = useState<Region>('global');

  // ---- the key ----
  const [key, setKey] = useState('');
  /** Editing: the card's stored key stands in until a new one is typed. */
  const [storedKey, setStoredKey] = useState(false);
  const [keyStatus, setKeyStatus] = useState<KeyStatus>('idle');
  const [failure, setFailure] = useState<Failure | null>(null);
  const [customUrl, setCustomUrl] = useState('');
  const [customSdk, setCustomSdk] = useState<LLMSdk>('openai');

  // ---- the model ----
  const [models, setModels] = useState<string[]>([]);
  const [labels, setLabels] = useState<Record<string, string>>({});
  const [model, setModel] = useState('');
  const [modelOpen, setModelOpen] = useState(false);
  /** The provider lists no models and the daemon asked for one: it is typed. */
  const [needsModel, setNeedsModel] = useState(false);

  // ---- one-click sign-in ----
  const [signin, setSignin] = useState<SigninState>('idle');
  const [signinFailure, setSigninFailure] = useState<Failure | null>(null);
  const [signedCard, setSignedCard] = useState<CredentialCard | null>(null);
  const [email, setEmail] = useState<string | null>(null);
  const [signoutError, setSignoutError] = useState<Failure | null>(null);

  // ---- submit ----
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  /** Bumped by every change that makes an in-flight test or sign-in stale. */
  const tokenRef = useRef(0);
  /** Next / Save pressed while (or before) a test: save once it passes. */
  const afterRef = useRef(false);
  /** The key a blur left untested, waiting for the click under way to land. */
  const pendingBlurRef = useRef<string | null>(null);
  const pointerDownRef = useRef(false);
  const signinAbortRef = useRef<AbortController | null>(null);
  const keyInputRef = useRef<HTMLInputElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const pasteSoonRef = useRef(false);
  /** The latest runCheck, for the timers and document listeners. */
  const runCheckRef = useRef<() => Promise<void>>(async () => {});
  /** Mirrors afterRef for drawing Next's busy state. */
  const [awaitingPass, setAwaitingPass] = useState(false);

  const info = catalog?.find((p) => p.id === provider);
  const isSignin = method === 'signin';
  const isCustom = provider === CUSTOM;
  const signinKind: SigninKind | null = info?.signin ?? (provider === CHATGPT ? 'chatgpt' : null);

  // The state the timers and listeners read: always the last committed one.
  const live = useRef({ key, keyStatus, provider, region, customUrl, customSdk, storedKey, model, submitting, needsModel });
  useLayoutEffect(() => {
    live.current = { key, keyStatus, provider, region, customUrl, customSdk, storedKey, model, submitting, needsModel };
    runCheckRef.current = () => runCheck();
  });

  // ---- edit: start already connected ----
  const [initialized, setInitialized] = useState(false);
  useEffect(() => {
    if (initialized || !catalog || !card) return;
    setInitialized(true);
    const pid = card.provider && catalog.some((p) => p.id === card.provider) ? card.provider : CUSTOM;
    setProvider(pid);
    setRegion(card.region === 'china' ? 'china' : 'global');
    setModel(card.model || '');
    if (pid === CUSTOM) {
      setCustomUrl(card.base_url || '');
      setCustomSdk(card.sdk || 'openai');
    }
    const signinCard = card.auth === 'signin' || pid === CHATGPT;
    setMethod(signinCard ? 'signin' : 'key');
    if (signinCard) {
      setSignedCard(card);
      setEmail(card.account?.email ?? null);
      if (card.needs_signin || autoSignIn) {
        setSignin('idle');
        if (autoSignIn) void startSignin(pid === CHATGPT ? 'chatgpt' : 'tokendance', card.id);
      } else {
        setSignin('ok');
        void loadCardModels(card, true);
      }
      return;
    }
    setStoredKey(card.key_set);
    if (card.key_set && !card.last_error) {
      setKeyStatus('ok');
      void loadCardModels(card, true);
    } else if (card.last_error) {
      setKeyStatus('fail');
      setFailure({ code: card.last_error.code, status: card.last_error.status, message: card.last_error.message });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [catalog, card, initialized]);

  /** The saved card's list, without spending a completion call. */
  async function loadCardModels(c: CredentialCard, open: boolean) {
    const token = tokenRef.current;
    let list: string[] = [];
    let names: Record<string, string> = {};
    try {
      const res = await api<{ models?: string[]; details?: ProviderCheckResult['model_details'] }>(
        '/api/v2/providers/models',
        { method: 'POST', body: JSON.stringify({ card_id: c.id }) },
      );
      list = res?.models ?? [];
      names = labelsOf({ models: list, model_details: res?.details });
    } catch {
      list = [];
    }
    if (token !== tokenRef.current) return;
    if (c.model && !list.includes(c.model)) list = [c.model, ...list];
    setModels(list);
    setLabels(names);
    setModelOpen(open);
  }

  // ---- pointer tracking: a click that blurs the key field lands first ----
  const flushBlur = useCallback(() => {
    const pending = pendingBlurRef.current;
    pendingBlurRef.current = null;
    const s = live.current;
    if (pending === null || s.key !== pending || s.keyStatus !== 'idle' || s.submitting) return;
    if (document.activeElement === keyInputRef.current) return;
    void runCheckRef.current();
  }, []);

  useEffect(() => {
    const down = () => { pointerDownRef.current = true; };
    const up = () => {
      pointerDownRef.current = false;
      // After the click this pointer makes has been dispatched.
      if (pendingBlurRef.current !== null) window.setTimeout(flushBlur, 0);
    };
    document.addEventListener('pointerdown', down, true);
    document.addEventListener('pointerup', up, true);
    document.addEventListener('pointercancel', up, true);
    return () => {
      document.removeEventListener('pointerdown', down, true);
      document.removeEventListener('pointerup', up, true);
      document.removeEventListener('pointercancel', up, true);
    };
  }, [flushBlur]);

  // The More search closes on a press outside it, and on Escape.
  useEffect(() => {
    if (!moreOpen) return;
    const onDoc = (e: PointerEvent) => {
      const root = rootRef.current?.querySelector('[data-connect-tiles]');
      if (root && !e.composedPath().includes(root)) setMoreOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setMoreOpen(false); };
    document.addEventListener('pointerdown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('pointerdown', onDoc);
      document.removeEventListener('keydown', onKey);
    };
  }, [moreOpen]);

  useEffect(() => () => signinAbortRef.current?.abort(), []);

  /** Anything about the credential changed: the last verdict no longer holds.
   *  `keepBlur`: a region switch that blurred the field still tests after it. */
  function invalidate(keepBlur = false) {
    tokenRef.current += 1;
    afterRef.current = false;
    setAwaitingPass(false);
    if (!keepBlur) pendingBlurRef.current = null;
    setKeyStatus('idle');
    setFailure(null);
    setModelOpen(false);
    setSubmitError(null);
  }

  function pickProvider(id: string) {
    if (editing) return;
    setMoreOpen(false);
    setQuery('');
    if (id === provider) {
      // TokenDance after "I have a key": its tile brings the sign-in back.
      if (method === 'key' && id === 'tokendance' && keyStatus !== 'ok') {
        setMethod('signin');
        invalidate();
      }
      return;
    }
    invalidate();
    signinAbortRef.current?.abort();
    if (signin === 'waiting' && signinKind) void cancelSignIn(signinKind);
    const p = catalog?.find((x) => x.id === id);
    setProvider(id);
    setKey('');
    setStoredKey(false);
    setModels([]);
    setLabels({});
    setModel('');
    setNeedsModel(false);
    setSignin('idle');
    setSigninFailure(null);
    setSignedCard(null);
    setEmail(null);
    setSignoutError(null);
    setMethod(p && (p.auth === 'signin' || p.signin) ? 'signin' : 'key');
    setRegion(initialRegion(p, locale));
    if (id === CUSTOM) {
      setCustomUrl('');
      setCustomSdk('openai');
    }
  }

  function changeRegion(r: Region) {
    if (r === region) return;
    setRegion(r);
    invalidate(true);
  }

  function editKey(value: string) {
    setKey(value);
    invalidate();
  }

  /** The connection test, on exactly these inputs. */
  async function runCheck(over?: { key?: string; region?: Region }) {
    const s = live.current;
    const p = catalog?.find((x) => x.id === s.provider);
    if (!p) return;
    const k = (over?.key ?? s.key).trim();
    const r = over?.region ?? s.region;
    const useStored = !k && s.storedKey && !!card;
    if (!k && !useStored && !(p.id === CUSTOM && s.customUrl.trim())) return;
    const token = ++tokenRef.current;
    pendingBlurRef.current = null;
    setKeyStatus('testing');
    setFailure(null);
    setModelOpen(false);
    setSubmitError(null);
    const res = await checkConnection({
      provider: p.id,
      region: r,
      apiKey: k || undefined,
      cardId: useStored ? card!.id : undefined,
      baseUrl: s.customUrl.trim() || undefined,
      sdk: s.customSdk,
      model: s.needsModel && s.model.trim() ? s.model.trim() : null,
    }, p).catch((): ProviderCheckResult => ({
      ok: false, status: null, code: null, message: '', models: [], default_model: null,
    }));
    if (token !== tokenRef.current) return;
    if (!res.ok) {
      afterRef.current = false;
      setAwaitingPass(false);
      if (res.code === 'missing_model') setNeedsModel(true);
      setKeyStatus('fail');
      setFailure({ code: res.code, status: res.status, message: res.message });
      return;
    }
    const typed = s.needsModel ? s.model.trim() : '';
    let list = res.models.length ? res.models : (res.default_model ? [res.default_model] : []);
    if (typed && !list.includes(typed)) list = [typed, ...list];
    // A model the user typed because the provider lists none is the one tested.
    const chosen = typed || pickModel(list, s.model || card?.model, res.default_model ?? p.default_model);
    setModels(list);
    setLabels(labelsOf(res));
    setModel(chosen);
    setKeyStatus('ok');
    if (afterRef.current) {
      afterRef.current = false;
      setAwaitingPass(false);
      await save({ model: chosen, key: k, region: r, useStored });
    }
  }

  /** Store the card (create, replace or edit). Only ever after a pass. */
  async function save(v: { model: string; key: string; region: Region; useStored: boolean }) {
    const s = live.current;
    setSubmitting(true);
    setSubmitError(null);
    try {
      let fields: ('provider' | 'region' | 'model' | 'api_key' | 'base_url' | 'sdk')[] | undefined;
      if (card) {
        fields = [];
        if (v.model !== card.model) fields.push('model');
        if (v.region !== (card.region === 'china' ? 'china' : 'global')) fields.push('region');
        if (v.key) fields.push('api_key');
        if (s.provider === CUSTOM) {
          if (s.customUrl.trim() !== (card.base_url || '')) fields.push('base_url');
          if (s.customSdk !== (card.sdk || 'openai')) fields.push('sdk');
        }
        if (fields.length === 0) {
          onDone(card);
          return;
        }
      }
      const res = await saveCard({
        cardId: card?.id ?? replaceCardId ?? null,
        provider: s.provider,
        region: v.region,
        model: v.model,
        apiKey: v.key || undefined,
        baseUrl: s.customUrl.trim(),
        sdk: s.customSdk,
        fields: card ? fields : undefined,
      });
      onCardStored?.(res.card);
      setKey('');
      setStoredKey(true);
      onDone(res.card);
    } catch (err) {
      const f = err as Failure & { test?: { code: string | null; status: number | null; message: string } };
      if (f && f.code === 'connection_test_failed') {
        // The daemon tested again and it failed: the verdict, under the field.
        setKeyStatus('fail');
        setFailure({ code: null, status: null, message: f.message });
      } else {
        setSubmitError(f?.message || t('llm.saveError'));
      }
    } finally {
      setSubmitting(false);
    }
  }

  async function startSignin(kind: SigninKind, cardId: string | null) {
    signinAbortRef.current?.abort();
    const controller = new AbortController();
    signinAbortRef.current = controller;
    const token = ++tokenRef.current;
    setSignin('waiting');
    setSigninFailure(null);
    setSignoutError(null);
    setModelOpen(false);
    try {
      const res = await signIn(kind, cardId, controller.signal);
      if (token !== tokenRef.current) return;
      setSignedCard(res.card);
      setEmail(res.email);
      onCardStored?.(res.card);
      setModel(res.card.model || '');
      if (res.test && !res.test.ok && res.test.code && REFUSED_CODES.has(res.test.code)) {
        // OpenAI says this account or plan can't be used by other apps: the
        // card was stored red. Offer a key or another account instead.
        setSigninFailure({ code: res.test.code, status: res.test.status, message: res.test.message });
        setSignin('refused');
        return;
      }
      setSignin('ok');
      if (res.test && !res.test.ok) {
        setSigninFailure({ code: res.test.code, status: res.test.status, message: res.test.message });
      }
      await loadCardModels(res.card, false);
    } catch (err) {
      if (token !== tokenRef.current) return;
      const f = err as Failure;
      if (f.code === 'signin_cancelled') {
        setSignin(cardId ? (signedCard ? 'ok' : 'idle') : 'idle');
        return;
      }
      setSigninFailure(f);
      setSignin(f.code && REFUSED_CODES.has(f.code) ? 'refused' : 'error');
    } finally {
      if (signinAbortRef.current === controller) signinAbortRef.current = null;
    }
  }

  function cancelSignin() {
    if (!signinKind) return;
    tokenRef.current += 1;
    signinAbortRef.current?.abort();
    signinAbortRef.current = null;
    void cancelSignIn(signinKind);
    setSignin(card && signedCard && !card.needs_signin ? 'ok' : 'idle');
  }

  async function reopenBrowser() {
    if (!signinKind) return;
    const ok = await reopenSignIn(signinKind);
    if (!ok) {
      // The daemon cannot reopen: start over, which opens the browser again.
      void cancelSignIn(signinKind);
      void startSignin(signinKind, signedCard?.id ?? null);
    }
  }

  /** A refused sign-in stored a card that cannot run: remove it before moving on. */
  async function dropRefusedCard(): Promise<void> {
    const c = signedCard;
    if (!c || (card && c.id === card.id)) return;
    setSignedCard(null);
    try {
      await deleteCardById(c.id);
      onCardRemoved?.(c.id);
    } catch {
      /* best effort: the row stays red in the list, with its delete action */
    }
  }

  async function signOut() {
    if (!signedCard) return;
    setSignoutError(null);
    try {
      await deleteCardById(signedCard.id);
      onCardRemoved?.(signedCard.id);
      tokenRef.current += 1;
      setSignedCard(null);
      setEmail(null);
      setModel('');
      setModels([]);
      setModelOpen(false);
      setSignin('idle');
      setSigninFailure(null);
    } catch (err) {
      setSignoutError(err as Failure);
    }
  }

  // ---- Next / Save ----
  const keyPresent = !!key.trim() || (editing && storedKey) || (isCustom && !!customUrl.trim());
  const canSubmit = !!provider && !submitting && (
    isSignin ? signin === 'ok' && !!signedCard && !!model.trim() : keyPresent && (keyStatus !== 'ok' || !!model.trim())
  );
  const busy = submitting || (keyStatus === 'testing' && awaitingPass);

  async function submit() {
    if (!canSubmit) return;
    pendingBlurRef.current = null;
    setSubmitError(null);
    if (isSignin) {
      const c = signedCard!;
      if (model.trim() && model.trim() !== c.model) {
        setSubmitting(true);
        try {
          const res = await saveCard({ cardId: c.id, provider: c.provider, region: 'global', model: model.trim(), fields: ['model'] });
          onCardStored?.(res.card);
          onDone(res.card);
        } catch (err) {
          setSubmitError((err as Failure)?.message || t('llm.saveError'));
        } finally {
          setSubmitting(false);
        }
        return;
      }
      onDone(c);
      return;
    }
    const s = live.current;
    if (s.keyStatus === 'ok') {
      await save({ model: s.model.trim(), key: s.key.trim(), region: s.region, useStored: !s.key.trim() && s.storedKey });
      return;
    }
    afterRef.current = true;
    setAwaitingPass(true);
    if (s.keyStatus === 'testing') return; // the running test saves on its pass
    await runCheck();
  }

  // ---- key field events ----
  function onKeyPaste(e: React.ClipboardEvent<HTMLInputElement>) {
    const text = (e.clipboardData?.getData('text') ?? '').trim();
    if (!text) {
      // Some paste paths (WebKit autofill) carry no clipboard text: test on the input that follows.
      pasteSoonRef.current = true;
      return;
    }
    e.preventDefault();
    const el = e.currentTarget;
    const start = el.selectionStart ?? el.value.length;
    const end = el.selectionEnd ?? start;
    const next = el.value.slice(0, start) + text + el.value.slice(end);
    editKey(next);
    live.current = { ...live.current, key: next, keyStatus: 'idle' };
    void runCheck({ key: next });
  }

  function onKeyChange(value: string) {
    editKey(value);
    if (pasteSoonRef.current && value.trim()) {
      pasteSoonRef.current = false;
      live.current = { ...live.current, key: value, keyStatus: 'idle' };
      void runCheck({ key: value });
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key !== 'Enter') return;
    // An IME confirming a candidate also sends Enter: that is not a submit.
    if (e.nativeEvent.isComposing || e.keyCode === 229) return;
    e.preventDefault();
    if ((key.trim() || (isCustom && customUrl.trim()) || (editing && storedKey)) && (keyStatus === 'idle' || keyStatus === 'fail')) {
      void runCheck();
    }
  }

  function onKeyBlur() {
    const s = live.current;
    const credential = s.key.trim() || (s.provider === CUSTOM && s.customUrl.trim());
    if (!credential || s.keyStatus !== 'idle') return;
    pendingBlurRef.current = s.key;
    // A click is under way (Next, a tile, the region toggle): let it land first.
    if (!pointerDownRef.current) window.setTimeout(flushBlur, 0);
  }

  // ---- render ----
  if (!catalog || (editing && !initialized)) {
    return (
      <div className="flex items-center gap-2 py-4 text-body text-secondary">
        <Loader2 className="h-4 w-4 animate-spin" />
        {t('llm.loading')}
      </div>
    );
  }

  const url = info ? consoleUrl(info, region) : null;
  const showRegion = hasRegions(info);
  const accountMono = !isSignin;

  const keyGroup = (
    <div
      data-testid="connect-key-group"
      className={`rounded-[11px] border bg-sidebar transition-[border-color,box-shadow] duration-150 ${
        keyStatus === 'fail'
          ? 'border-error/55'
          : 'border-border focus-within:border-accent focus-within:shadow-[var(--float-focus)]'
      }`}
    >
      {isCustom && (
        <>
          <div className="flex h-[42px] items-center gap-2 pl-3 pr-1.5">
            <Server aria-hidden="true" className="h-3.5 w-3.5 shrink-0 text-muted" strokeWidth={1.6} />
            <input
              type="text"
              value={customUrl}
              data-testid="connect-custom-url"
              aria-label={t('llm.field.baseUrl')}
              placeholder="https://…/v1"
              spellCheck={false}
              autoComplete="off"
              onChange={(e) => { setCustomUrl(e.target.value); invalidate(); }}
              onKeyDown={onKeyDown}
              onBlur={onKeyBlur}
              className="h-full min-w-0 flex-1 bg-transparent font-mono text-caption text-primary outline-none placeholder:font-sans placeholder:text-body placeholder:text-muted"
            />
            <Select
              aria-label={t('llm.field.sdk')}
              data-testid="connect-custom-sdk"
              value={customSdk}
              onChange={(e) => { setCustomSdk(e.target.value as LLMSdk); invalidate(); }}
              className="h-7 shrink-0 rounded-[7px] bg-card-hover px-2 text-caption text-secondary outline-none"
            >
              <option value="openai">{t('llm.sdk.openai')}</option>
              <option value="openai-responses">{t('llm.sdk.openaiResponses')}</option>
              <option value="anthropic">{t('llm.sdk.anthropic')}</option>
            </Select>
          </div>
          <div className="mx-3 h-px bg-border" />
        </>
      )}
      <div className="flex h-[42px] items-center gap-2 pl-3 pr-1.5">
        <KeyRound aria-hidden="true" className="h-3.5 w-3.5 shrink-0 text-muted" strokeWidth={1.6} />
        <input
          ref={keyInputRef}
          type="password"
          value={key}
          data-testid="connect-key"
          aria-label={t('llm.field.apiKey')}
          aria-invalid={keyStatus === 'fail' || undefined}
          placeholder={editing && storedKey && card?.key_masked ? card.key_masked : t('connect.key.placeholder')}
          autoComplete="off"
          spellCheck={false}
          disabled={card?.read_only}
          autoFocus={!editing && !isCustom}
          onChange={(e) => onKeyChange(e.target.value)}
          onPaste={onKeyPaste}
          onKeyDown={onKeyDown}
          onBlur={onKeyBlur}
          className="h-full min-w-0 flex-1 bg-transparent font-mono tracking-[0.06em] text-body text-primary outline-none placeholder:font-sans placeholder:tracking-normal placeholder:text-muted"
        />
        {keyStatus === 'testing' && (
          <Loader2 aria-label={t('llm.testing')} data-testid="connect-key-testing" className="h-3.5 w-3.5 shrink-0 animate-spin text-secondary" />
        )}
        {keyStatus === 'ok' && (
          <CircleCheck aria-label={t('connect.key.passed')} data-testid="connect-key-ok" className="h-3.5 w-3.5 shrink-0 text-success-text" strokeWidth={1.75} />
        )}
        {showRegion && <RegionToggle value={region} onChange={changeRegion} />}
        {url && <GetKeyLink url={url} />}
      </div>
      {needsModel && keyStatus !== 'ok' && (
        <>
          <div className="mx-3 h-px bg-border" />
          <label className="flex h-[42px] items-center gap-2 px-3">
            <span className="shrink-0 text-body text-secondary">{t('llm.field.model')}</span>
            <input
              type="text"
              value={model}
              data-testid="connect-model-typed"
              spellCheck={false}
              autoComplete="off"
              onChange={(e) => { setModel(e.target.value); invalidate(); }}
              onKeyDown={onKeyDown}
              className="min-w-0 flex-1 bg-transparent font-mono text-caption text-primary outline-none"
            />
          </label>
        </>
      )}
      {keyStatus === 'ok' && (
        <>
          <div className="mx-3 h-px bg-border" />
          <ModelRow
            models={models}
            labels={labels}
            value={model}
            open={modelOpen}
            mono={accountMono}
            onToggle={() => setModelOpen((v) => !v)}
            onChange={(m) => { setModel(m); setModelOpen(false); }}
          />
        </>
      )}
    </div>
  );

  const signinLabel = signinKind === 'chatgpt' ? t('connect.signin.chatgpt') : t('connect.signin.tokendance');
  const signinBlock = (
    <div>
      {signin === 'idle' && (
        <button
          type="button"
          data-testid="connect-signin"
          onClick={() => signinKind && void startSignin(signinKind, signedCard?.id ?? null)}
          className="flex h-11 w-full items-center justify-center gap-2 rounded-[11px] bg-primary px-5 text-body font-medium text-card hover:opacity-90"
        >
          <ProviderLogo id={signinKind === 'chatgpt' ? 'openai' : 'tokendance'} size={18} />
          {editing && card?.needs_signin ? t('connect.signin.again') : signinLabel}
        </button>
      )}
      {signin === 'waiting' && (
        <div className="flex items-center gap-2 @md:gap-3" data-testid="connect-signin-waiting">
          <div className="flex h-11 min-w-0 flex-1 items-center justify-center gap-2 rounded-[11px] bg-primary/80 px-4 text-body font-medium text-card">
            <Loader2 aria-hidden="true" className="h-3.5 w-3.5 shrink-0 animate-spin" />
            <span className="truncate">{t('connect.signin.waiting')}</span>
          </div>
          <button
            type="button"
            data-testid="connect-signin-reopen"
            aria-label={t('connect.signin.reopen')}
            title={t('connect.signin.reopen')}
            onClick={() => void reopenBrowser()}
            className="inline-grid h-9 w-9 shrink-0 place-items-center rounded-[9px] text-secondary hover:bg-card-hover hover:text-primary"
          >
            <RotateCw aria-hidden="true" className="h-4 w-4" strokeWidth={1.6} />
          </button>
          <button
            type="button"
            data-testid="connect-signin-cancel"
            onClick={cancelSignin}
            className="shrink-0 text-body font-medium text-secondary hover:text-primary max-md:min-h-[44px]"
          >
            {t('connect.signin.cancel')}
          </button>
        </div>
      )}
      {signin === 'ok' && (
        <div className="rounded-[12px] bg-sidebar" data-testid="connect-account">
          <div className="flex h-11 items-center gap-2.5 pl-3 pr-1.5">
            <ProviderLogo id={provider} size={20} />
            <span className="min-w-0 flex-1 truncate text-body font-medium text-primary" data-testid="connect-account-email">
              {email || signedCard?.key_masked || signedCard?.name}
            </span>
            <button
              type="button"
              data-testid="connect-signout"
              aria-label={t('connect.signout')}
              title={t('connect.signout')}
              onClick={() => void signOut()}
              className="inline-grid h-[26px] w-[26px] shrink-0 place-items-center rounded-[7px] text-muted hover:bg-card-hover hover:text-primary"
            >
              <LogOut aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.6} />
            </button>
          </div>
          <div className="mx-3 h-px bg-border" />
          <ModelRow
            models={models.length ? models : (model ? [model] : [])}
            labels={labels}
            value={model}
            open={modelOpen}
            mono={false}
            onToggle={() => setModelOpen((v) => !v)}
            onChange={(m) => { setModel(m); setModelOpen(false); }}
          />
        </div>
      )}
      {signin === 'refused' && (
        <div className="flex items-start gap-2.5 rounded-[11px] bg-sidebar px-3.5 py-3" data-testid="connect-signin-refused">
          <Ban aria-hidden="true" className="mt-[3px] h-3.5 w-3.5 shrink-0 text-error-text" strokeWidth={1.75} />
          <div className="min-w-0 flex-1">
            <div className="flex items-start gap-1.5">
              <span className="text-body font-medium text-primary">{t('connect.signin.refused')}</span>
              {signinFailure?.message && (
                <HintSign label={t('connect.fail.details')} align="right" testId="connect-signin-refused-sign">
                  <span className="block max-w-[260px] break-words text-primary">{signinFailure.message}</span>
                </HintSign>
              )}
            </div>
            <div className="mt-2 flex flex-wrap gap-2">
              <button
                type="button"
                data-testid="connect-use-openai-key"
                onClick={() => { void dropRefusedCard().then(() => pickProvider('openai')); }}
                className="inline-flex h-[30px] items-center gap-2 rounded-[8px] bg-card-hover px-3 text-caption font-medium text-primary hover:bg-nav-hover"
              >
                <ProviderLogo id="openai" size={14} />
                {t('connect.signin.useKey')}
              </button>
              <button
                type="button"
                data-testid="connect-signin-other"
                onClick={() => {
                  if (!signinKind) return;
                  const keep = card && signedCard?.id === card.id ? card.id : null;
                  void dropRefusedCard().then(() => startSignin(signinKind, keep));
                }}
                className="inline-flex h-[30px] items-center rounded-[8px] border border-border px-3 text-caption font-medium text-secondary hover:text-primary"
              >
                {t('connect.signin.other')}
              </button>
            </div>
          </div>
        </div>
      )}
      {signin === 'error' && (
        <>
          <button
            type="button"
            data-testid="connect-signin"
            onClick={() => signinKind && void startSignin(signinKind, signedCard?.id ?? null)}
            className="flex h-11 w-full items-center justify-center gap-2 rounded-[11px] bg-primary px-5 text-body font-medium text-card hover:opacity-90"
          >
            <ProviderLogo id={signinKind === 'chatgpt' ? 'openai' : 'tokendance'} size={18} />
            {signinLabel}
          </button>
          {signinFailure && <FailLine failure={signinFailure} testId="connect-signin-error" />}
        </>
      )}
      {signin === 'ok' && signinFailure && <FailLine failure={signinFailure} testId="connect-signin-test" />}
      {signoutError && <FailLine failure={signoutError} testId="connect-signout-error" />}
      {provider === 'tokendance' && !editing && signin !== 'ok' && signin !== 'waiting' && info?.key_entry && (
        <div className="mt-2 flex justify-end">
          <button
            type="button"
            data-testid="connect-have-key"
            onClick={() => { setMethod('key'); invalidate(); }}
            className="text-caption text-secondary hover:text-primary"
          >
            {t('connect.haveKey')}
          </button>
        </div>
      )}
    </div>
  );

  return (
    <div ref={rootRef} className="@container" data-testid="connect-panel">
      {!editing && (
        <div data-connect-tiles>
          <ProviderTiles
            catalog={catalog}
            selected={provider}
            moreOpen={moreOpen}
            query={query}
            onPick={pickProvider}
            onToggleMore={() => { setMoreOpen((v) => !v); setQuery(''); }}
            onQuery={setQuery}
          />
        </div>
      )}
      {provider && (
        <div className={editing ? '' : 'mt-5'} data-testid={`connect-provider-${provider}`}>
          {isSignin ? signinBlock : (
            <>
              {keyGroup}
              {failure && keyStatus === 'fail' && (
                <div className="flex flex-wrap items-center gap-x-3">
                  <FailLine failure={failure} testId="connect-key-error" />
                  {editing && provider === 'tokendance' && info?.signin && (
                    <button
                      type="button"
                      data-testid="connect-tokendance-resignin"
                      onClick={() => { setMethod('signin'); setSignin('idle'); void startSignin('tokendance', card!.id); }}
                      className="mt-2 text-body font-medium text-accent-text hover:underline"
                    >
                      {t('connect.signin.again')}
                    </button>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      )}
      {footer({ canSubmit, busy, submit: () => void submit(), error: submitError })}
    </div>
  );
}
