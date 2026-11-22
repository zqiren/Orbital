// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Who may use one connected connector: a compact trigger on the connector
 * row ("All agents", or the chosen agents' logos) that opens a small menu —
 * All / Selected, then the agents from GET /api/v2/connectors/agents with a
 * tick each. The manager ("Orbital") is one of them.
 *
 * The control only reports the next access value; ConnectorSettings saves it
 * (optimistic, reverted on failure). Unsupported agents are listed but can
 * never be ticked; why sits behind a hover sign, not as visible text. The
 * agent list stays rendered in both modes (dimmed under "All") so switching
 * never changes the menu's height, and the trigger has a fixed width so the
 * row does not move either.
 */

import { useEffect, useId, useRef, useState } from 'react';
import { Check, ChevronDown } from 'lucide-react';
import type { ConnectorAccess, ConnectorAgent } from '../types';
import { useT } from '../i18n/useT';
import { AgentLogo } from './history/AgentLogo';
import HintSign from './HintSign';

export const DEFAULT_ACCESS: ConnectorAccess = { mode: 'all', agents: [] };

/** The ids that actually get the connector under 'selected', in list order. */
export function tickedAgents(access: ConnectorAccess, agents: ConnectorAgent[]): string[] {
  const chosen = new Set(access.agents);
  return agents.filter((a) => a.supported && chosen.has(a.id)).map((a) => a.id);
}

/**
 * The access value after picking a mode. Back to 'all' keeps the selection
 * (the daemon holds it for next time). To 'selected' restores the last
 * selection, or, when there is none, starts with every supported agent
 * ticked, so the switch itself takes nothing away from anyone.
 */
export function accessForMode(
  mode: ConnectorAccess['mode'],
  access: ConnectorAccess,
  agents: ConnectorAgent[],
): ConnectorAccess {
  if (mode === 'all') return { mode: 'all', agents: access.agents };
  const restored = tickedAgents(access, agents);
  return {
    mode: 'selected',
    agents: restored.length > 0 ? access.agents : agents.filter((a) => a.supported).map((a) => a.id),
  };
}

/** The access value after ticking or unticking one agent (selected mode). */
export function accessWithAgent(
  access: ConnectorAccess,
  id: string,
  on: boolean,
): ConnectorAccess {
  const rest = access.agents.filter((x) => x !== id);
  return { mode: 'selected', agents: on ? [...rest, id] : rest };
}

const MAX_LOGOS = 3;

interface ConnectorAccessControlProps {
  connectorId: string;
  connectorName: string;
  access: ConnectorAccess;
  /** null = loading; 'error' = the agent list did not load. */
  agents: ConnectorAgent[] | null | 'error';
  onChange: (next: ConnectorAccess) => void;
}

export default function ConnectorAccessControl({
  connectorId,
  connectorName,
  access,
  agents,
  onChange,
}: ConnectorAccessControlProps) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const menuId = useId();
  const list = Array.isArray(agents) ? agents : [];
  // The ones that can be ticked first (the daemon's order within each part):
  // unsupported agents interleaved by name break up the list being scanned.
  const rows = [...list.filter((a) => a.supported), ...list.filter((a) => !a.supported)];
  const all = access.mode === 'all';
  const ticked = tickedAgents(access, list);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (rootRef.current && !e.composedPath().includes(rootRef.current)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDoc);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  let summary: React.ReactNode;
  if (all) {
    summary = <span className="truncate">{t('connectors.access.all')}</span>;
  } else if (ticked.length === 0) {
    summary = <span className="truncate">{t('connectors.access.none')}</span>;
  } else {
    const shown = ticked.slice(0, MAX_LOGOS);
    const nameOf = (id: string) => list.find((a) => a.id === id)?.name ?? id;
    summary = (
      <span className="flex items-center gap-1">
        {shown.map((id) => (
          <AgentLogo key={id} slug={id} title={nameOf(id)} />
        ))}
        {ticked.length > MAX_LOGOS && (
          <span className="text-micro tabular-nums text-secondary">+{ticked.length - MAX_LOGOS}</span>
        )}
      </span>
    );
  }

  const segment = (mode: ConnectorAccess['mode'], label: string) => {
    const on = access.mode === mode;
    return (
      <button
        type="button"
        role="radio"
        aria-checked={on}
        data-testid={`connector-access-mode-${mode}-${connectorId}`}
        onClick={() => {
          if (!on) onChange(accessForMode(mode, access, list));
        }}
        className={`flex-1 min-w-0 truncate rounded-md px-2 py-1 text-caption font-medium transition-colors duration-150 max-md:min-h-[36px] ${
          on
            ? 'bg-segment-on text-segment-on-text shadow-[0_1px_2px_rgb(0_0_0/0.06)]'
            : 'text-secondary hover:text-primary'
        }`}
      >
        {label}
      </button>
    );
  };

  return (
    // On a phone the menu hangs from the whole card (the nearest positioned
    // ancestor once this wrapper stops being one), so it can use the card's
    // width instead of running off the left edge from a right-hand trigger.
    <div ref={rootRef} className="md:relative">
      <button
        type="button"
        data-testid={`connector-access-${connectorId}`}
        aria-label={t('connectors.access.aria', { name: connectorName })}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        onClick={() => setOpen((v) => !v)}
        className="inline-flex h-7 w-[96px] items-center justify-end gap-1 rounded-lg px-2 text-caption font-medium text-secondary hover:bg-card-hover hover:text-primary transition-colors duration-150 aria-expanded:bg-card-hover max-md:min-h-[44px]"
      >
        {summary}
        <ChevronDown size={12} aria-hidden="true" className="shrink-0 text-muted" />
      </button>

      {open && (
        <div
          id={menuId}
          role="dialog"
          aria-label={t('connectors.access.aria', { name: connectorName })}
          data-testid={`connector-access-menu-${connectorId}`}
          className="surface-float-pop absolute right-0 top-full z-40 mt-1 w-[248px] p-1.5 max-md:left-2 max-md:right-2 max-md:w-auto"
        >
          <div role="radiogroup" className="flex gap-0.5 rounded-lg bg-nav p-0.5">
            {segment('all', t('connectors.access.all'))}
            {segment('selected', t('connectors.access.selected'))}
          </div>

          {agents === 'error' ? (
            <p className="px-2 py-2 text-caption text-error" role="alert">
              {t('connectors.access.loadError')}
            </p>
          ) : (
            <ul className="mt-1">
              {rows.map((a) => {
                const checked = a.supported && (all || ticked.includes(a.id));
                const disabled = all || !a.supported;
                return (
                  <li key={a.id}>
                    <label
                      data-testid={`connector-access-agent-${connectorId}-${a.id}`}
                      className={`flex items-center gap-2.5 rounded-lg px-2 py-1.5 text-body max-md:min-h-[44px] ${
                        disabled ? 'cursor-default' : 'cursor-pointer hover:bg-card-hover/60'
                      }`}
                    >
                      <input
                        type="checkbox"
                        className="sr-only peer"
                        checked={checked}
                        disabled={disabled}
                        onChange={(e) => {
                          // Never trust the disabled attribute alone: an
                          // unsupported agent must not reach the selection.
                          if (disabled) return;
                          onChange(accessWithAgent(access, a.id, e.target.checked));
                        }}
                        data-testid={`connector-access-check-${connectorId}-${a.id}`}
                      />
                      <span
                        aria-hidden="true"
                        className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-[5px] border transition-colors duration-150 peer-focus-visible:ring-2 peer-focus-visible:ring-accent/50 ${
                          checked ? 'border-accent bg-accent text-white' : 'border-border'
                        } ${all ? 'opacity-50' : ''}`}
                      >
                        {checked && <Check size={11} strokeWidth={3} />}
                      </span>
                      <span className={`flex min-w-0 flex-1 items-center gap-2 ${a.supported ? '' : 'opacity-50'}`}>
                        <AgentLogo slug={a.id} title={a.name} />
                        <span className="truncate text-primary">{a.name}</span>
                      </span>
                      {!a.supported && (
                        <HintSign
                          label={t('connectors.access.whyAria')}
                          align="right"
                          testId={`connector-access-why-${connectorId}-${a.id}`}
                        >
                          {t('connectors.access.unsupported')}
                        </HintSign>
                      )}
                    </label>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
