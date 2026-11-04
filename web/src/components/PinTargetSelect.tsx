// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Spec 074 — the composer pin control. The sticky selection IS the session's
// sub-agent pin: choosing a worker PATCHes `pinned_target`, and while pinned
// every composer send dispatches straight to that worker with zero
// management-agent turns. Selecting Orbital (the manager) back is the unpin.
//
// Design (aligned 2026-08-31 via the "Composer Pin Header" mockup): a
// permanent logo-only mark fused to the input row's left edge — Orbital's own
// mark at rest, the pinned agent's mark while pinned. Colorless chrome: no
// tint, no name; identity comes from the avatar itself, the tooltip, and the
// "Message {agent}…" placeholder. Hidden entirely when no sub-agents are
// installed: zero surface for users without workers.
//
// Spec 099 (approved mockup, 2026-09-27): each row carries an availability
// dot and, for claude-code / codex, what is left of the subscription's
// 5-hour window (weekly when tighter). A hover card beside the menu shows
// every window with reset times; phones get both windows stacked in the row
// instead. The collapsed control stays colorless — it only gains a corner
// dot when the pinned worker needs attention (low, limited, not ready,
// running).
//
// Keyless onboarding (2026-10-21): with no API key Orbital's own agent cannot
// run. Its row stays first but says "API key required", and choosing it opens
// Global Settings at the LLM cards instead of unpinning onto a manager that
// would only fail.

// Spec 149 (design "C2 with chevron", 2026-11-02): in the composer, each
// worker row carries a second muted line "<model> · <thinking>" with a small
// chevron. The line is its own control, beside (not inside) the row's option
// button: clicking it swaps the side usage card, in place and at the same
// width, for that worker's Model + Thinking panel (this conversation only).
// The Orbital row's line is the project's credential card; its panel picks
// another (the same project PUT as Project Settings). Clicking the line again,
// Esc, or anywhere else in the menu swaps the usage card back; Esc never
// closes the menu while the panel shows. Phones have no side card, so the
// panel opens inline under its row.

import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { ChevronDown } from 'lucide-react';
import MessageAvatar from './MessageAvatar';
import {
  labelFor,
  modelLineText,
  type ParamsPatch,
  type WorkerModelInfo,
} from '../hooks/useConversationModels';
import { useT } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import {
  refreshAgentAvailability,
  useAgentAvailability,
} from '../hooks/useAgentAvailability';
import {
  ageParts,
  formatResetTime,
  leftTone,
  relativeParts,
  statusDot,
  summarizeQuota,
  windowLeft,
  type DotColor,
  type QuotaTone,
  type ShownWindow,
} from '../utils/quotaDisplay';
import type { AgentAvailability, QuotaWindowKind } from '../types';

/** Result of resolving one composer send against the session pin. */
export interface ResolvedSendTarget {
  /** Slug to dispatch to, or undefined for the management agent. */
  target?: string;
  /** Message text, always verbatim — a leading `@slug` is ordinary text. */
  content: string;
  /** True whenever `target` is set: the sticky dropdown pin is the composer's
   *  only direct-to-worker send (spec 091). The backend maps every `target`
   *  send to initiator="user_pinned" — the wake-suppressed dispatch class. */
  pinned: boolean;
}

/**
 * Target resolution (spec 091, replacing spec 074 §3.2's @mention
 * precedence): the sticky pin applies, otherwise the management agent. There
 * is no `@` parse. `@codex do X` sent to Orbital is an ordinary request for
 * Orbital to dispatch (its supervised path); sent while pinned, it reaches
 * the pinned worker as typed. Switching who you talk to is the dropdown.
 */
export function resolveSendTarget(
  text: string,
  pinnedTarget: string | null | undefined,
): ResolvedSendTarget {
  if (pinnedTarget) {
    return { target: pinnedTarget, content: text, pinned: true };
  }
  return { target: undefined, content: text, pinned: false };
}

interface PinTargetSelectProps {
  /** Installed sub-agents (App-level `/agents/available`, built-in excluded). */
  agents: Array<{ slug: string; name: string }>;
  /** Currently pinned slug, or null when talking to Orbital (the manager). */
  value: string | null;
  /** Fired with the new slug, or null when Orbital is selected (unpin). */
  onChange: (slug: string | null) => void;
  disabled?: boolean;
  /**
   * Spec 079 — where this control is mounted.
   *
   * `'composer'` (the default, and ChatView's only spelling) keeps the
   * edge-fusion geometry described above: negative margins that pull the mark
   * through the composer card's padding, a full-height left-rounded button
   * with a divider on its right, and a menu that opens upward off a control
   * sitting at the bottom of the viewport.
   *
   * `'standalone'` is the same menu as an ordinary inline form control — a
   * self-contained bordered pill that claims no space it wasn't given. Used by
   * the queue composer's option row and the automation form, neither of which
   * has a card edge to fuse to.
   */
  variant?: 'composer' | 'standalone';
  /** Accessible name + tooltip for the manager (unassigned) state. Defaults to
   *  the chat pin's wording; the queue and automation surfaces say "runs this"
   *  rather than "talking to". */
  managerLabel?: string;
  /** Spec 099: slugs with an open turn in this session (from the shared
   *  useSubAgentStatus pipeline) — their dot pulses green. */
  runningSlugs?: readonly string[];
  /**
   * The project's management agent has a key (`project.manager_ready`).
   * Default true. When false the Orbital row carries an "API key required"
   * pill and choosing it opens Global Settings → LLM (the
   * `open-global-settings` event, detail.section 'llm') instead of
   * calling onChange(null).
   */
  managerReady?: boolean;
  /** Spec 149, composer only: the model line on each row and its panel.
   *  Absent (queue, automations) = the rows exactly as before. */
  models?: PinModels;
  'data-testid'?: string;
}

/** Spec 149: what the composer hands the menu for the model lines. */
export interface PinModels {
  /** Per worker slug; a worker missing here gets no line. */
  workers: Record<string, WorkerModelInfo>;
  /** False until the conversation exists: lines show, but cannot change. */
  editable: boolean;
  /** Set / clear (null) a worker's model or thinking; resolves false on failure. */
  onWorkerChange: (slug: string, patch: ParamsPatch) => Promise<boolean>;
  /** Orbital's row: the project's credential card. Absent = no line. */
  manager?: {
    cards: Array<{ id: string; name: string }>;
    /** The project's `card_id`; null follows the global default card. */
    cardId: string | null;
    /** The global default card's name, for "Default (…)". */
    defaultName: string | null;
    onChange: (cardId: string | null) => Promise<boolean>;
  };
  /** The menu opened: re-read the choices and this conversation's picks. */
  onMenuOpen?: () => void;
}

const MENU_ROW =
  'w-full flex items-center gap-2.5 px-3 py-2 text-[13px] text-primary text-left ' +
  'hover:bg-card-hover/50 max-md:min-h-[52px]';

const ELEVATED = 'surface-float-pop';

/** Agents whose subscription quota Orbital can read (spec 099 §2.1). */
const QUOTA_SLUGS = new Set(['claude-code', 'codex']);

type T = ReturnType<typeof useT>;

// ---------------------------------------------------------------------------
// Dot
// ---------------------------------------------------------------------------

const DOT_FILL: Record<DotColor, string> = {
  green: 'bg-success text-success',
  amber: 'bg-warning text-warning',
  red: 'bg-error text-error',
  grey: 'bg-card shadow-[inset_0_0_0_1.5px_var(--color-muted)]',
};

const DOT_LABEL = {
  green: 'pinAgent.status.running',
  amber: 'pinAgent.status.low',
  red: 'pinAgent.status.limited',
  grey: 'pinAgent.status.needsLogin',
} as const;

function StatusDot({ color, pulse, testId, t }: {
  color: DotColor; pulse: boolean; testId: string; t: T;
}) {
  return (
    <span
      data-testid={testId}
      data-color={color}
      data-pulse={pulse ? 'true' : 'false'}
      role="img"
      aria-label={t(DOT_LABEL[color])}
      className={`absolute -right-[3px] -bottom-[3px] w-[9px] h-[9px] rounded-full
                  border-2 border-card box-content ${DOT_FILL[color]}`}
    >
      {pulse && (
        <span
          aria-hidden
          className="absolute -inset-[3px] rounded-full border-2 border-current
                     motion-safe:animate-ping [animation-duration:1.6s] motion-reduce:opacity-40"
        />
      )}
    </span>
  );
}

function AvatarWithDot({ slug, dot, dotTestId, t }: {
  slug?: string;
  dot: { color: DotColor; pulse: boolean } | null;
  dotTestId: string;
  t: T;
}) {
  return (
    <div className="relative shrink-0">
      <MessageAvatar variant="agent" agentHandle={slug} />
      {dot && <StatusDot {...dot} testId={dotTestId} t={t} />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Row usage text
// ---------------------------------------------------------------------------

type LineTone = 'ok' | 'low' | 'out' | 'muted';
interface Line { text: string; tone: LineTone }

const LINE_CLASS: Record<LineTone, string> = {
  ok: 'text-secondary',
  low: 'text-warning-text',
  out: 'text-error-text',
  muted: 'text-muted',
};

function windowText(w: ShownWindow, t: T): string {
  return t('pinAgent.quota.left', {
    window: t(w.kind === 'weekly' ? 'pinAgent.quota.window.weekly' : 'pinAgent.quota.window.fiveHour'),
    n: w.left,
  });
}

/** [top line, optional second line]: the desktop row shows the first, a
 *  phone row stacks both (no hover to hide the other window behind). */
function usageLines(
  slug: string, info: AgentAvailability | undefined, now: number,
  t: T, locale: 'en' | 'zh',
): Line[] {
  if (!info) return [];
  // Spec 116 D4: "needs update" BEFORE the generic not-ready branch — a CLI
  // too old to know `auth status` reads as signed-out too, and the user
  // must hear "update", not "login".
  if (info.update_needed) return [{ text: t('pinAgent.status.needsUpdate'), tone: 'ok' }];
  if (info.ready === false) return [{ text: t('pinAgent.status.needsLogin'), tone: 'ok' }];
  if (!QUOTA_SLUGS.has(slug)) return [];
  const s = summarizeQuota(info.quota, now);
  if (s.kind === 'none') return [{ text: t('pinAgent.quota.noReading'), tone: 'muted' }];
  if (s.kind === 'stale') return [{ text: t('pinAgent.quota.noCurrent'), tone: 'muted' }];
  const second = (o: ShownWindow | null): Line[] =>
    o ? [{ text: windowText(o, t), tone: 'muted' }] : [];
  if (s.kind === 'limited') {
    const text = s.resetsAt
      ? t('pinAgent.quota.limitReached', { time: formatResetTime(s.resetsAt, now, locale) })
      : t('pinAgent.quota.limitReachedNoTime');
    return [{ text, tone: 'out' }, ...second(s.other)];
  }
  return [{ text: windowText(s.shown, t), tone: s.tone }, ...second(s.other)];
}

// ---------------------------------------------------------------------------
// Hover card
// ---------------------------------------------------------------------------

const CARD_WINDOWS: Array<[QuotaWindowKind, Parameters<T>[0]]> = [
  ['five_hour', 'pinAgent.card.window.fiveHour'],
  ['weekly', 'pinAgent.card.window.weekly'],
  ['weekly_opus', 'pinAgent.card.window.weeklyOpus'],
  ['weekly_sonnet', 'pinAgent.card.window.weeklySonnet'],
];

const BAR_FILL: Record<QuotaTone, string> = {
  ok: 'bg-accent', low: 'bg-warning', out: 'bg-error',
};

const titleCase = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

function unitText(unit: 'minutes' | 'hours' | 'days', n: number, prefix: 'age' | 'in', t: T) {
  return t(`pinAgent.quota.${prefix}.${unit}` as Parameters<T>[0], { n });
}

function hasCard(slug: string | null, info: AgentAvailability | undefined): boolean {
  if (!slug || !info) return false;
  return info.update_needed === true || info.ready === false
    || QUOTA_SLUGS.has(slug) || slug === 'cursor';
}

function QuotaCard({ slug, name, info, now, t, locale }: {
  slug: string; name: string; info: AgentAvailability; now: number;
  t: T; locale: 'en' | 'zh';
}) {
  const shell = (header: ReactNode, body: ReactNode) => (
    <div
      role="tooltip"
      data-testid="pin-quota-card"
      className={`${ELEVATED} w-[300px] px-3.5 py-3 flex flex-col gap-2.5 text-left`}
    >
      <div className="flex items-center gap-2">{header}</div>
      {body}
    </div>
  );
  const note = (text: string) => <p className="m-0 text-xs text-secondary">{text}</p>;

  if (info.update_needed || info.ready === false) {
    return shell(
      <>
        <AvatarWithDot slug={slug} dot={{ color: 'grey', pulse: false }} dotTestId="pin-card-dot" t={t} />
        <b className="text-[13px] font-semibold">{name}</b>
      </>,
      note(info.update_needed
        ? t('pinAgent.card.needsUpdate', {
            name,
            version: info.version?.match(/\d+\.\d+\.\d+/)?.[0] ?? info.version ?? '?',
            min: info.min_version ?? '?',
          })
        : t('pinAgent.card.needsLogin', { name })),
    );
  }
  if (!QUOTA_SLUGS.has(slug)) {
    return shell(
      <>
        <MessageAvatar variant="agent" agentHandle={slug} />
        <b className="text-[13px] font-semibold">{name}</b>
      </>,
      note(t('pinAgent.card.cursor')),
    );
  }

  const quota = info.quota;
  const header = (
    <>
      <MessageAvatar variant="agent" agentHandle={slug} />
      <b className="text-[13px] font-semibold">{t('pinAgent.card.usage', { name })}</b>
      {quota?.plan && (
        <span className="ml-auto text-[11px] text-muted">
          {t('pinAgent.card.plan', { plan: titleCase(quota.plan) })}
        </span>
      )}
    </>
  );
  if (!quota) {
    return shell(header, note(t(slug === 'codex'
      ? 'pinAgent.card.noReadingCodex' : 'pinAgent.card.noReadingClaude')));
  }

  const blocks = CARD_WINDOWS.flatMap(([kind, labelKey]) => {
    const w = quota.windows?.find((x) => x.kind === kind);
    if (!w) return [];
    const left = windowLeft(w, now);
    const resets = w.resets_at ? new Date(w.resets_at) : null;
    const time = resets ? formatResetTime(resets, now, locale) : null;
    let sub: string;
    if (left === null) sub = t('pinAgent.card.void');
    else if (left <= 0 && time) sub = t('pinAgent.card.reached', { time });
    else if (time && resets) {
      const rel = relativeParts(resets, now);
      sub = rel
        ? t('pinAgent.card.remainRel', { n: left, time, rel: unitText(rel.unit, rel.n, 'in', t) })
        : t('pinAgent.card.remain', { n: left, time });
    } else sub = t('pinAgent.card.remainNoReset', { n: left });
    const tone = left === null ? 'ok' : leftTone(left);
    return [(
      <div key={kind} className="flex flex-col gap-1">
        <div className="flex justify-between gap-2 text-xs">
          <b className="font-medium">{t(labelKey)}</b>
          <span className="text-secondary tabular-nums">{left === null ? '—' : `${left}%`}</span>
        </div>
        <div className="h-1.5 rounded-[3px] bg-card-hover overflow-hidden">
          <i
            className={`block h-full rounded-[3px] ${BAR_FILL[tone]}`}
            style={{ width: `${left ?? 0}%` }}
          />
        </div>
        <div className="text-[11px] text-muted tabular-nums">{sub}</div>
      </div>
    )];
  });

  const age = ageParts(quota.observed_at, now);
  const fromReply = slug === 'claude-code';
  const footer = age === null ? null : age.unit === 'now'
    ? t(fromReply ? 'pinAgent.card.updatedNowReply' : 'pinAgent.card.updatedNow')
    : t(fromReply ? 'pinAgent.card.updatedAgoReply' : 'pinAgent.card.updatedAgo',
      { ago: unitText(age.unit, age.n, 'age', t) });

  return shell(header, (
    <>
      {blocks}
      {footer && (
        <div className="text-[11px] text-muted border-t border-border pt-2">{footer}</div>
      )}
    </>
  ));
}

// ---------------------------------------------------------------------------
// Spec 149: model + thinking panel (the side card's other state)
// ---------------------------------------------------------------------------

function Radio({ checked, onClick, testId, children }: {
  checked: boolean; onClick: () => void; testId: string; children: ReactNode;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={checked}
      data-testid={testId}
      onClick={onClick}
      className="flex w-full min-w-0 items-center gap-2.5 rounded-md px-2 py-1.5 text-left text-[13px] text-primary
                 hover:bg-card-hover/50 focus-visible:outline-none focus-visible:bg-card-hover/50 max-md:min-h-[44px]"
    >
      <span
        aria-hidden
        className={`grid h-3.5 w-3.5 shrink-0 place-items-center rounded-full border ${
          checked ? 'border-accent' : 'border-muted/70'
        }`}
      >
        {checked && <span className="h-1.5 w-1.5 rounded-full bg-accent" />}
      </span>
      {children}
    </button>
  );
}

function Section({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="px-2 text-[11px] font-medium text-secondary">{label}</span>
      {children}
    </div>
  );
}

function PanelFooter({ text, error, t }: { text: string; error: boolean; t: T }) {
  return (
    <p className="m-0 border-t border-border/60 px-2 pt-2 text-[11px] leading-snug text-muted">
      {error && (
        <span role="alert" data-testid="pin-model-error" className="block text-error-text">
          {t('pinAgent.model.saveFailed')}
        </span>
      )}
      {text}
    </p>
  );
}

/** A long id (MiniMax's `m:minimax:MiniMax-M3:v:thinking`) wraps, never
 *  truncates — at its `:` / `/` separators first, anywhere only as a last resort. */
const VALUE_TEXT = 'min-w-0 [overflow-wrap:anywhere]';

function breakable(text: string): ReactNode {
  const parts = text.split(/(?<=[:/])/);
  return parts.length < 2 ? text : parts.map((p, i) => (
    <span key={i}>{p}{i < parts.length - 1 && <wbr />}</span>
  ));
}

function WorkerPanel({ info, onChange, error, t }: {
  info: WorkerModelInfo;
  onChange: (patch: ParamsPatch) => void;
  error: boolean;
  t: T;
}) {
  const settingsLabel = info.settingsModel ? labelFor(info, info.settingsModel) : null;
  const level = info.effort ?? info.settingsEffort;
  // A pick the list no longer offers still shows, checked.
  const listed = info.models
    ? (info.model && !info.models.some((m) => m.value === info.model)
      ? [...info.models, { value: info.model, label: info.model }]
      : info.models)
    : null;
  return (
    <div className="flex flex-col gap-2.5" data-testid={`pin-model-panel-${info.slug}`}>
      <Section label={t('pinAgent.model.section.model')}>
        <div role="radiogroup" aria-label={t('pinAgent.model.section.model')}>
          <Radio
            checked={info.model === null}
            onClick={() => onChange({ model: null })}
            testId="pin-model-opt-default"
          >
            <span className="shrink-0">{t('pinAgent.model.default')}</span>
            {settingsLabel && (
              <span className={`${VALUE_TEXT} text-muted`}>
                {breakable(t('pinAgent.model.defaultValue', { value: settingsLabel }))}
              </span>
            )}
          </Radio>
          {listed
            ? listed.map((m) => (
                <Radio
                  key={m.value}
                  checked={info.model === m.value}
                  onClick={() => onChange({ model: m.value })}
                  testId={`pin-model-opt-${m.value}`}
                >
                  <span className={VALUE_TEXT}>{breakable(m.label)}</span>
                </Radio>
              ))
            : <FreeTextModel key={info.slug} info={info} onChange={onChange} t={t} />}
        </div>
      </Section>
      {info.levels && info.levels.length > 0 && (
        <Section label={t('pinAgent.model.section.thinking')}>
          <div
            role="radiogroup"
            aria-label={t('pinAgent.model.section.thinking')}
            className="mx-2 flex gap-0.5 rounded-lg border border-border bg-nav p-0.5"
          >
            {info.levels.map((l) => (
              <button
                key={l}
                type="button"
                role="radio"
                aria-checked={level === l}
                data-testid={`pin-model-level-${l}`}
                // Picking the Settings level (or the current pick again)
                // clears the override: the conversation follows Settings.
                onClick={() => onChange({
                  effort: l === info.effort || l === info.settingsEffort ? null : l,
                })}
                className={`flex-auto whitespace-nowrap rounded-md px-1 py-1 text-[11px] font-medium transition-colors duration-150 max-md:min-h-[36px] ${
                  level === l
                    ? 'bg-segment-on text-segment-on-text shadow-[0_1px_2px_rgb(0_0_0/0.06)]'
                    : 'text-secondary hover:text-primary'
                }`}
              >
                {l}
              </button>
            ))}
          </div>
        </Section>
      )}
      <PanelFooter text={t('pinAgent.model.scope.worker')} error={error} t={t} />
    </div>
  );
}

/** Free-text agents (MiniMax, Pi, Cursor…): Default, or a typed model id,
 *  applied on Enter or when the field loses focus. */
function FreeTextModel({ info, onChange, t }: {
  info: WorkerModelInfo; onChange: (patch: ParamsPatch) => void; t: T;
}) {
  const [draft, setDraft] = useState(info.model ?? '');
  const inputRef = useRef<HTMLInputElement>(null);
  const commit = () => {
    const v = draft.trim();
    if (v !== (info.model ?? '')) onChange({ model: v || null });
  };
  const custom = info.model !== null;
  return (
    <div
      className="flex w-full min-w-0 items-center gap-2.5 rounded-md px-2 py-1 text-[13px] max-md:min-h-[44px]"
      data-testid="pin-model-opt-other"
    >
      <button
        type="button"
        role="radio"
        aria-checked={custom}
        aria-label={t('pinAgent.model.other')}
        onClick={() => inputRef.current?.focus()}
        className={`grid h-3.5 w-3.5 shrink-0 place-items-center rounded-full border focus-visible:outline-none ${
          custom ? 'border-accent' : 'border-muted/70'
        }`}
      >
        {custom && <span className="h-1.5 w-1.5 rounded-full bg-accent" />}
      </button>
      <input
        ref={inputRef}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') { e.preventDefault(); commit(); }
        }}
        placeholder={t('pinAgent.model.other')}
        aria-label={t('pinAgent.model.other')}
        spellCheck={false}
        data-testid="pin-model-other-input"
        className="min-w-0 flex-1 rounded-md border border-border bg-card px-2 py-1 text-[13px] text-primary
                   placeholder:text-muted focus:border-accent focus:outline-none max-md:text-base"
      />
    </div>
  );
}

function ManagerPanel({ manager, onChange, error, t }: {
  manager: NonNullable<PinModels['manager']>;
  onChange: (cardId: string | null) => void;
  error: boolean;
  t: T;
}) {
  return (
    <div className="flex flex-col gap-2.5" data-testid="pin-model-panel-orbital">
      <Section label={t('pinAgent.model.section.provider')}>
        <div role="radiogroup" aria-label={t('pinAgent.model.section.provider')}>
          <Radio
            checked={manager.cardId === null}
            onClick={() => onChange(null)}
            testId="pin-model-card-default"
          >
            <span className="shrink-0">{t('pinAgent.model.default')}</span>
            {manager.defaultName && (
              <span className={`${VALUE_TEXT} text-muted`}>
                {t('pinAgent.model.defaultValue', { value: manager.defaultName })}
              </span>
            )}
          </Radio>
          {manager.cards.map((c) => (
            <Radio
              key={c.id}
              checked={manager.cardId === c.id}
              onClick={() => onChange(c.id)}
              testId={`pin-model-card-${c.id}`}
            >
              <span className={VALUE_TEXT}>{c.name}</span>
            </Radio>
          ))}
        </div>
      </Section>
      <PanelFooter text={t('pinAgent.model.scope.manager')} error={error} t={t} />
    </div>
  );
}

/** The Orbital row's line: the card the project runs on. */
function managerCardName(manager: PinModels['manager']): string | null {
  if (!manager) return null;
  if (manager.cardId === null) return manager.defaultName;
  return manager.cards.find((c) => c.id === manager.cardId)?.name ?? null;
}

/** Date.now(), re-read every 30 s while `active` (ages, reset voiding). */
function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const id = setInterval(() => setNow(Date.now()), 30_000);
    return () => clearInterval(id);
  }, [active]);
  return active ? now : Date.now();
}

export default function PinTargetSelect({
  agents,
  value,
  onChange,
  disabled,
  variant = 'composer',
  managerLabel,
  runningSlugs,
  managerReady = true,
  models,
  'data-testid': testId,
}: PinTargetSelectProps) {
  const t = useT();
  const { locale } = useLocale();
  const [open, setOpen] = useState(false);
  // The composer can sit far enough right that the menu and the card beside it
  // run off a narrow window. Slide them back on screen, as far as the left edge
  // allows. Every render: which card shows changes the width.
  const menuWrapRef = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const wrap = menuWrapRef.current;
    if (!wrap) return;
    wrap.style.transform = '';
    const { left, right } = wrap.getBoundingClientRect();
    const shift = Math.min(right - (window.innerWidth - 8), left - 8);
    if (shift > 0) wrap.style.transform = `translateX(-${shift}px)`;
  });
  // Whose hover card shows while the menu is open (defaults to the pinned
  // worker's), and whether the collapsed control is hovered.
  const [cardFor, setCardFor] = useState<string | null>(null);
  const [controlHover, setControlHover] = useState(false);
  // Spec 149: whose card is in its model state (undefined = the usage card;
  // null = Orbital's), and whether the last change there failed.
  const [modelFor, setModelFor] = useState<string | null | undefined>(undefined);
  const [modelError, setModelError] = useState(false);
  const availability = useAgentAvailability();
  const now = useNow(open || controlHover);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDocMouseDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      // Spec 149: Esc first returns the card to usage; the menu stays open.
      if (modelFor !== undefined) setModelFor(undefined);
      else setOpen(false);
    };
    document.addEventListener('mousedown', onDocMouseDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('mousedown', onDocMouseDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open, modelFor]);

  // No sub-agents installed → no pin surface at all.
  if (agents.length === 0) return null;

  // A stale pin (agent uninstalled since) still renders so the user can see
  // and clear it — the avatar lookup falls back to a monogram badge, and the
  // menu appends a bare-slug row instead of silently showing Orbital while
  // the session is actually still pinned.
  const known = agents.some((a) => a.slug === value);
  const pinnedName = value ? (agents.find((a) => a.slug === value)?.name ?? value) : null;

  const pick = (slug: string | null) => {
    setOpen(false);
    if (slug === null && !managerReady) {
      // Unpinning onto a manager with no key would only fail; send the user
      // to where the key goes. App's listener scrolls to the cards block.
      window.dispatchEvent(
        new CustomEvent('open-global-settings', { detail: { section: 'llm' } }),
      );
      return;
    }
    onChange(slug);
  };

  const toggle = () => {
    // Decided from the rendered `open`, never inside a setState updater
    // (updaters must stay pure; CLAUDE.md React anti-patterns).
    setModelFor(undefined);
    if (!open) {
      setCardFor(value);
      setControlHover(false);
      // One backend-cached REST hit; also kicks the codex read-through.
      refreshAgentAvailability();
      models?.onMenuOpen?.();
    }
    setOpen(!open);
  };

  const standalone = variant === 'standalone';
  const managerText = managerLabel ?? t('pinAgent.tooltipManager');
  const running = new Set(runningSlugs ?? []);
  const dotFor = (slug: string) => statusDot({
    ready: availability[slug]?.ready,
    running: running.has(slug),
    quota: QUOTA_SLUGS.has(slug) ? availability[slug]?.quota : undefined,
    now,
  });
  const nameOf = (slug: string) => agents.find((a) => a.slug === slug)?.name ?? slug;
  const card = (slug: string | null) => (slug && hasCard(slug, availability[slug]) ? (
    <QuotaCard
      slug={slug}
      name={nameOf(slug)}
      info={availability[slug]}
      now={now}
      t={t}
      locale={locale}
    />
  ) : null);

  // --- Spec 149: the model line and its panel -------------------------------
  const pinModels = standalone ? undefined : models;
  const managerName = managerCardName(pinModels?.manager);
  const lineText = (slug: string | null): string | null => {
    if (!pinModels) return null;
    if (slug === null) return managerName;
    const info = pinModels.workers[slug];
    return info ? modelLineText(info, t('pinAgent.model.default')) : null;
  };
  const editable = (slug: string | null) =>
    slug === null ? Boolean(pinModels?.manager) : Boolean(pinModels?.editable);
  const openModel = (slug: string | null) => {
    setModelError(false);
    setCardFor(slug);
    setModelFor(modelFor === slug ? undefined : slug);
  };
  const changeWorker = (slug: string, patch: ParamsPatch) => {
    setModelError(false);
    void pinModels?.onWorkerChange(slug, patch).then((ok) => { if (!ok) setModelError(true); });
  };
  const changeCard = (cardId: string | null) => {
    setModelError(false);
    void pinModels?.manager?.onChange(cardId).then((ok) => { if (!ok) setModelError(true); });
  };
  const panelFor = (slug: string | null): ReactNode => {
    if (!pinModels) return null;
    if (slug === null) {
      return pinModels.manager
        ? <ManagerPanel manager={pinModels.manager} onChange={changeCard} error={modelError} t={t} />
        : null;
    }
    const info = pinModels.workers[slug];
    return info
      ? <WorkerPanel info={info} onChange={(p) => changeWorker(slug, p)} error={modelError} t={t} />
      : null;
  };
  /** The side card's model state: same shell and width as the usage card. */
  const modelCard = (slug: string | null) => {
    const panel = panelFor(slug);
    if (!panel) return null;
    const name = slug === null ? t('pinAgent.orbital') : nameOf(slug);
    return (
      <div
        role="dialog"
        aria-label={t('pinAgent.model.aria', { agent: name })}
        data-testid="pin-model-card"
        data-card-for={slug ?? 'orbital'}
        // A long live model list scrolls inside the card on a short window.
        className={`${ELEVATED} w-[300px] max-h-[calc(100dvh-7rem)] overflow-y-auto px-1.5 py-3 flex flex-col gap-2.5 text-left`}
      >
        <div className="flex items-center gap-2 px-2">
          <MessageAvatar variant="agent" agentHandle={slug ?? undefined} />
          <b className="text-[13px] font-semibold">{name}</b>
        </div>
        {panel}
      </div>
    );
  };
  const hoverRow = (slug: string | null) => {
    // While a panel shows, the card stays put: the pointer may cross other
    // rows on its way to it.
    if (modelFor !== undefined) return;
    setCardFor(slug);
  };
  /** The row's second line, a control of its own (or plain text before the
   *  conversation exists). */
  const modelLine = (slug: string | null, name: string, text: string) => {
    const key = slug ?? 'orbital';
    if (!editable(slug)) {
      return (
        <span data-testid={`pin-model-line-${key}`} className="mt-px text-xs text-muted whitespace-nowrap max-md:truncate max-md:max-w-full">
          {text}
        </span>
      );
    }
    const active = modelFor === slug;
    return (
      <button
        type="button"
        data-model-line
        data-testid={`pin-model-line-${key}`}
        aria-expanded={active}
        aria-label={`${t('pinAgent.model.aria', { agent: name })}: ${text}`}
        title={text}
        onClick={(e) => { e.stopPropagation(); openModel(slug); }}
        className={`group/line pointer-events-auto relative z-10 mt-px inline-flex max-w-full items-center rounded-sm
                    text-xs whitespace-nowrap transition-colors duration-150 hover:text-primary
                    focus-visible:outline-none focus-visible:text-primary max-md:py-1 ${
                      active ? 'text-primary' : 'text-muted'}`}
      >
        <span className="min-w-0 decoration-dotted decoration-1 underline-offset-[3px] group-hover/line:underline group-focus-visible/line:underline max-md:truncate">
          {text}
        </span>
        <ChevronDown size={11} aria-hidden className="ml-0.5 shrink-0 text-muted" />
      </button>
    );
  };
  /** A row WITH a line: the option is a full-row button underneath, the
   *  visible content passes clicks through to it, and the line sits above
   *  as its own control — no control nested inside another. */
  const lineRow = (opts: {
    slug: string | null; name: string; text: string; selected: boolean;
    avatar: ReactNode; trailing: ReactNode; label: string; highlight: boolean;
  }) => (
    <div
      key={opts.slug ?? 'orbital'}
      data-testid={`pin-row-${opts.slug ?? 'orbital'}`}
      onMouseEnter={() => hoverRow(opts.slug)}
      className={`${MENU_ROW} relative group/row has-[>button[role=option]:focus-visible]:bg-card-hover/50 ${
        opts.highlight ? 'md:bg-card-hover/50' : ''}`}
    >
      <button
        type="button"
        role="option"
        aria-selected={opts.selected}
        aria-label={opts.label}
        onClick={() => pick(opts.slug)}
        onFocus={() => hoverRow(opts.slug)}
        className="absolute inset-0 h-full w-full focus-visible:outline-none"
      />
      <span className="pointer-events-none contents">
        {opts.avatar}
        <span className="flex min-w-0 flex-1 flex-col items-start">
          <span className="font-medium whitespace-nowrap">{opts.name}</span>
          {modelLine(opts.slug, opts.name, opts.text)}
        </span>
        {opts.trailing}
      </span>
    </div>
  );
  /** Phone: no side card, so the panel opens under its row. */
  const inlinePanel = (slug: string | null) => (modelFor === slug && pinModels ? (
    <div
      data-model-inline
      data-testid={`pin-model-inline-${slug ?? 'orbital'}`}
      className="md:hidden border-y border-border/60 bg-card/60 px-1.5 py-2"
    >
      {panelFor(slug)}
    </div>
  ) : null);

  const pinnedHasCard = hasCard(value, value ? availability[value] : undefined);
  const controlDot = value ? dotFor(value) : null;
  const menuCard = !open ? null : modelFor !== undefined ? modelCard(modelFor) : card(cardFor);
  const hoverCard = !open && controlHover ? card(value) : null;

  const orbitalTrailing = (
    <span className="ml-auto flex items-center gap-2 text-xs text-muted whitespace-nowrap">
      {managerReady ? (
        <span>{t('pinAgent.managerRole')}</span>
      ) : (
        <span
          data-testid="pin-orbital-key-required"
          className="rounded-full border border-warning/40 bg-warning/10 px-2 py-px text-[11px] font-medium text-warning-text"
        >
          {t('pinAgent.orbital.keyRequired')}
        </span>
      )}
      <span className="w-3 text-center">{value === null ? '✓' : ''}</span>
    </span>
  );
  const orbitalText = lineText(null);

  return (
    // Composer: -ml-3/-my-2 pull the control through the composer card's
    // padding so the mark fuses to the card's left edge, full row height
    // (self-stretch), per the aligned mockup. rounded-l matches the card's
    // inner radius. Standalone drops all of that and sits inline.
    <div
      ref={rootRef}
      className={
        standalone
          ? 'relative flex shrink-0'
          : 'relative self-stretch flex shrink-0 -ml-3 -my-2'
      }
      data-testid={testId ?? 'pin-target-select'}
      data-tour="pin-select"
    >
      <button
        type="button"
        onClick={toggle}
        onMouseEnter={() => setControlHover(true)}
        onMouseLeave={() => setControlHover(false)}
        disabled={disabled}
        aria-label={managerLabel ?? t('pinAgent.aria')}
        aria-haspopup="listbox"
        aria-expanded={open}
        // The usage card replaces the native tooltip when there is one; two
        // tooltips on one hover would stack.
        title={pinnedHasCard ? undefined : pinnedName
          ? t('pinAgent.tooltipPinned', { name: pinnedName })
          : managerText}
        className={
          standalone
            ? `flex items-center gap-1 px-1.5 py-1 rounded-md border border-border
               hover:bg-card-hover/50 disabled:opacity-50 focus-visible:outline-none
               focus-visible:bg-card-hover/50 max-md:min-h-[36px]`
            : `flex items-center gap-1 pl-3 pr-1.5 rounded-l-[14px] border-r border-border
               hover:bg-card-hover/50 disabled:opacity-50 focus-visible:outline-none
               focus-visible:bg-card-hover/50 max-md:min-w-[48px]`
        }
      >
        <AvatarWithDot
          slug={value ?? undefined}
          dot={controlDot}
          dotTestId="pin-control-dot"
          t={t}
        />
        <ChevronDown size={10} className="text-muted shrink-0" />
      </button>

      {hoverCard && (
        <div
          className={`absolute ${standalone ? 'top-full mt-1' : 'bottom-full mb-2'}
                     left-0 z-50 max-md:hidden pointer-events-none`}
        >
          {hoverCard}
        </div>
      )}

      {open && (
        // Menu and hover card share one positioned wrapper so the card sits
        // beside the menu, bottom-aligned with it (top-aligned when the
        // standalone menu opens downward).
        <div
          ref={menuWrapRef}
          className={`absolute ${standalone ? 'top-full mt-1 items-start' : 'bottom-full mb-2 items-end'}
                     left-0 z-50 flex gap-2`}
        >
          <div
            role="listbox"
            aria-label={managerLabel ?? t('pinAgent.aria')}
            // Spec 149: a press anywhere in the menu but a line or an open
            // phone panel returns the card to usage.
            onMouseDown={pinModels ? (e) => {
              const el = e.target as HTMLElement;
              if (!el.closest('[data-model-line], [data-model-inline]')) setModelFor(undefined);
            } : undefined}
            // Standalone opens downward: it is an ordinary field in the middle of
            // a form, where an upward menu would clip against the surface above.
            className={`${ELEVATED} min-w-[300px] max-w-[calc(100vw-2rem)] overflow-hidden ${
              pinModels ? 'max-md:max-h-[calc(100dvh-7rem)] max-md:overflow-y-auto' : ''}`}
          >
            {orbitalText ? lineRow({
              slug: null,
              name: t('pinAgent.orbital'),
              text: orbitalText,
              selected: value === null,
              avatar: <MessageAvatar variant="agent" />,
              trailing: orbitalTrailing,
              label: t('pinAgent.orbital'),
              highlight: false,
            }) : (
              <button
                type="button"
                role="option"
                aria-selected={value === null}
                onClick={() => pick(null)}
                onMouseEnter={() => hoverRow(null)}
                onFocus={() => hoverRow(null)}
                className={MENU_ROW}
              >
                <MessageAvatar variant="agent" />
                <span className="font-medium whitespace-nowrap">{t('pinAgent.orbital')}</span>
                {orbitalTrailing}
              </button>
            )}
            {inlinePanel(null)}
            <div className="h-px bg-border/60" aria-hidden />
            {agents.map((a) => {
              const lines = usageLines(a.slug, availability[a.slug], now, t, locale);
              const highlight = cardFor === a.slug && Boolean(menuCard);
              const avatar = (
                <AvatarWithDot
                  slug={a.slug}
                  dot={dotFor(a.slug)}
                  dotTestId={`pin-dot-${a.slug}`}
                  t={t}
                />
              );
              const trailing = (
                <span className="ml-auto flex items-center gap-2 text-xs text-muted whitespace-nowrap tabular-nums">
                  {lines[0] && (
                    <span
                      data-testid={`pin-quota-${a.slug}`}
                      data-tone={lines[0].tone}
                      className={`max-md:hidden ${LINE_CLASS[lines[0].tone]}`}
                    >
                      {lines[0].text}
                    </span>
                  )}
                  {lines[0] && (
                    <span
                      data-testid={`pin-quota-stack-${a.slug}`}
                      className="hidden max-md:flex flex-col items-end gap-px"
                    >
                      {lines.map((l, i) => (
                        <span key={i} className={LINE_CLASS[l.tone]}>{l.text}</span>
                      ))}
                    </span>
                  )}
                  <span className="w-3 text-center">{value === a.slug ? '✓' : ''}</span>
                </span>
              );
              const text = lineText(a.slug);
              if (text) {
                return [
                  lineRow({
                    slug: a.slug,
                    name: a.name,
                    text,
                    selected: value === a.slug,
                    avatar,
                    trailing,
                    label: [a.name, lines[0]?.text].filter(Boolean).join(', '),
                    highlight,
                  }),
                  <span key={`${a.slug}-inline`} className="contents">{inlinePanel(a.slug)}</span>,
                ];
              }
              return (
                <button
                  key={a.slug}
                  type="button"
                  role="option"
                  aria-selected={value === a.slug}
                  onClick={() => pick(a.slug)}
                  onMouseEnter={() => hoverRow(a.slug)}
                  onFocus={() => hoverRow(a.slug)}
                  className={`${MENU_ROW} ${highlight ? 'md:bg-card-hover/50' : ''}`}
                >
                  {avatar}
                  <span className="font-medium whitespace-nowrap">{a.name}</span>
                  {trailing}
                </button>
              );
            })}
            {value && !known && (
              <button
                type="button"
                role="option"
                aria-selected
                onClick={() => pick(value)}
                onMouseEnter={() => hoverRow(null)}
                className={MENU_ROW}
              >
                <MessageAvatar variant="agent" agentHandle={value} />
                <span className="font-medium">{value}</span>
                <span className="ml-auto flex items-center gap-2 text-xs text-muted">
                  <span className="w-3 text-center">✓</span>
                </span>
              </button>
            )}
          </div>
          {menuCard && <div className="max-md:hidden">{menuCard}</div>}
        </div>
      )}
    </div>
  );
}
