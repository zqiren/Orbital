// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The Workbench composer (spec 154 §5.6): a raised card at the top of the
 * page — the text, then one row with the project chip, the agent chip and
 * send. Both chips start on 自动 / Auto; routing happens on send, never while
 * typing. After a send the text clears and both chips reset to 自动, so a
 * per-task pick never sticks (and never moves the project's own pin).
 *
 *  - Project chip: folder icon + name. Picker: 自动, then the projects in
 *    sidebar order (excluded projects are not on the Workbench at all).
 *  - Agent chip: once picked, only the agent's logo and a chevron (name in
 *    the tooltip). Picker: 自动, then Orbital and the available workers, each
 *    with logo, name and model line, then one muted row that opens Global
 *    Settings at user memory, where the user tells Orbital which agent does
 *    which kind of work (the project's manager reads it when it delegates).
 *
 * Keyless (no API key, 2026-11-05): routing needs Orbital's own model, so the
 * project chip never shows 自动. It starts on the last project sent to from
 * this browser (else Quick Tasks), and the picker's 自动 row stays visible
 * but not selectable, with a note; a click on it opens the model settings.
 */

import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import { ArrowUp, Check, ChevronDown, Folder, Settings2, Sparkles } from 'lucide-react';
import type { Project } from '../../types';
import { useT } from '../../i18n/useT';
import { api } from '../../config';
import { getAgentIcon, MAIN_AGENT_HANDLE } from '../../utils/agentIcons';
import { projectDisplayName } from '../../utils/projectDisplayName';
import {
  modelLineText, workerModelInfo, type SubAgentModelEntry,
} from '../../hooks/useConversationModels';
import { restoreComposer, useWorkbenchFeed } from '../../hooks/useWorkbenchFeed';
import type { SendChoice } from './sendTask';

const LAST_PROJECT_KEY = 'orbital.wbLastProject';

/** The last project a Workbench task was sent to from this browser. */
export function readLastProject(): string | null {
  try {
    return window.localStorage.getItem(LAST_PROJECT_KEY);
  } catch {
    return null;
  }
}

export function rememberLastProject(projectId: string): void {
  try {
    window.localStorage.setItem(LAST_PROJECT_KEY, projectId);
  } catch {
    /* private mode: the default just falls back to Quick Tasks */
  }
}

export interface WorkbenchComposerProps {
  projects: Project[];
  /** No API key: no routing, so no 自动 project (see the header). */
  keyless?: boolean;
  agents: Array<{ slug: string; name: string }>;
  /** Orbital's own model line (the default card's model). */
  managerModel?: string;
  /** Resolves when the send settled; false = the text came back. */
  onSend: (choice: SendChoice) => Promise<boolean>;
}

type Pop = 'none' | 'project' | 'agent';

function AgentMark({ slug, size = 16 }: { slug: string; size?: number }) {
  const icon = getAgentIcon(slug);
  const orb = slug === MAIN_AGENT_HANDLE;
  return (
    <span className={`wb-av${orb ? ' is-orb' : ''}`} style={{ width: size, height: size }}>
      {icon.src ? <img src={icon.src} alt="" /> : <span style={{ fontSize: 9, color: icon.color }}>{icon.monogram}</span>}
    </span>
  );
}

export default function WorkbenchComposer({
  projects, keyless = false, agents, managerModel, onSend,
}: WorkbenchComposerProps) {
  const t = useT();
  const feed = useWorkbenchFeed();
  const [text, setText] = useState('');
  const [projectId, setProjectId] = useState<string | null>(null);
  const [lastProject, setLastProject] = useState<string | null>(readLastProject);
  const [agent, setAgent] = useState<string | null>(null);
  const [pop, setPop] = useState<Pop>('none');
  const [entries, setEntries] = useState<SubAgentModelEntry[] | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const agentChipRef = useRef<HTMLButtonElement>(null);

  // A send that could not start puts its text back (§5.6 step 6), and a
  // reload restores an unsent draft — whichever page is open at the time.
  useEffect(() => {
    if (feed.composerRestore === null) return;
    setText(feed.composerRestore);
    restoreComposer(null);
  }, [feed.composerRestore]);

  // Close a picker on Esc and on a press outside it.
  useEffect(() => {
    if (pop === 'none') return;
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Escape') setPop('none');
    };
    const onPointer = (e: Event) => {
      const wrap = wrapRef.current;
      if (wrap && !e.composedPath().includes(wrap)) setPop('none');
    };
    window.addEventListener('keydown', onKey);
    document.addEventListener('pointerdown', onPointer);
    return () => {
      window.removeEventListener('keydown', onKey);
      document.removeEventListener('pointerdown', onPointer);
    };
  }, [pop]);

  // The model lines load when the agent picker opens.
  useEffect(() => {
    if (pop !== 'agent') return;
    let cancelled = false;
    api<SubAgentModelEntry[]>('/api/v2/settings/sub-agents')
      .then((list) => { if (!cancelled) setEntries(Array.isArray(list) ? list : []); })
      .catch(() => { if (!cancelled) setEntries([]); });
    return () => { cancelled = true; };
  }, [pop]);

  // Grow with the text, up to the CSS cap.
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${el.scrollHeight}px`;
  }, [text]);

  // Keyless: the chip stands on a real project from the start.
  const keylessDefault = keyless
    ? (projects.find((p) => p.project_id === lastProject)
      ?? projects.find((p) => p.is_scratch)
      ?? projects[0])?.project_id ?? null
    : null;
  const effectiveProjectId = projectId ?? keylessDefault;
  const picked = effectiveProjectId ? projects.find((p) => p.project_id === effectiveProjectId) ?? null : null;
  const projectLabel = picked ? projectDisplayName(picked, t) : t('workbench.composer.auto');
  const agentLabel = agent
    ? (agent === MAIN_AGENT_HANDLE ? 'Orbital' : agents.find((a) => a.slug === agent)?.name ?? agent)
    : t('workbench.composer.auto');

  function modelLine(slug: string): string | null {
    if (slug === MAIN_AGENT_HANDLE) return managerModel || null;
    const entry = entries?.find((e) => e.slug === slug);
    if (!entry) return null;
    // Item 158: a picked project's own worker pick shows here too.
    const info = workerModelInfo(entry, picked?.sub_agent_params?.[slug]);
    return info ? modelLineText(info, t('pinAgent.model.default')) : null;
  }

  async function send() {
    const body = text.trim();
    if (!body) return;
    const choice: SendChoice = { text: body, projectId: effectiveProjectId, agent };
    if (effectiveProjectId) {
      rememberLastProject(effectiveProjectId);
      setLastProject(effectiveProjectId);
    }
    setText('');
    setProjectId(null);
    setAgent(null);
    setPop('none');
    await onSend(choice);
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void send();
    }
  }

  const ck = (on: boolean) => (
    <span className="wb-ck">{on ? <Check size={14} strokeWidth={2.4} aria-hidden="true" /> : null}</span>
  );

  const agentLeft = agentChipRef.current?.offsetLeft ?? 84;

  return (
    <div className="wb-composer" ref={wrapRef}>
      <div className="wb-box wb-layer">
        <textarea
          ref={inputRef}
          className="wb-input"
          rows={2}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKey}
          placeholder={t('workbench.composer.placeholder')}
          aria-label={t('workbench.composer.placeholder')}
          data-testid="wb-composer-input"
        />
        <div className="wb-foot">
          <button
            type="button"
            className={`wb-chip${picked ? ' is-set' : ''}`}
            aria-haspopup="listbox"
            aria-expanded={pop === 'project'}
            aria-label={`${t('workbench.composer.project')}: ${projectLabel}`}
            title={projectLabel}
            data-testid="wb-chip-project"
            onClick={() => setPop(pop === 'project' ? 'none' : 'project')}
          >
            <span className="inline-flex shrink-0"><Folder size={13} strokeWidth={1.9} aria-hidden="true" /></span>
            <span className="wb-nm">{projectLabel}</span>
            <span className="wb-cv"><ChevronDown size={11} strokeWidth={2.2} aria-hidden="true" /></span>
          </button>
          <button
            ref={agentChipRef}
            type="button"
            className={`wb-chip${agent ? ' is-set' : ''}`}
            aria-haspopup="listbox"
            aria-expanded={pop === 'agent'}
            aria-label={`${t('workbench.composer.agent')}: ${agentLabel}`}
            title={agentLabel}
            data-testid="wb-chip-agent"
            onClick={() => setPop(pop === 'agent' ? 'none' : 'agent')}
          >
            {agent ? (
              <AgentMark slug={agent} />
            ) : (
              <>
                <span className="inline-flex shrink-0"><Sparkles size={13} strokeWidth={1.9} aria-hidden="true" /></span>
                <span className="wb-nm">{t('workbench.composer.auto')}</span>
              </>
            )}
            <span className="wb-cv"><ChevronDown size={11} strokeWidth={2.2} aria-hidden="true" /></span>
          </button>
          <span className="flex-1" />
          <button
            type="button"
            className="wb-send"
            aria-label={t('workbench.composer.send')}
            title={t('workbench.composer.send')}
            data-testid="wb-composer-send"
            disabled={!text.trim()}
            onClick={() => void send()}
          >
            <ArrowUp size={15} strokeWidth={2.2} aria-hidden="true" />
          </button>
        </div>
      </div>

      {pop === 'project' && (
        <div className="wb-pop" role="listbox" aria-label={t('workbench.composer.project')} style={{ left: 8 }} data-testid="wb-pop-project">
          {keyless ? (
            // Routing needs a key: 自动 stays visible, not selectable; the
            // click goes where the key is added.
            <button
              type="button"
              role="option"
              aria-selected={false}
              aria-disabled="true"
              className="wb-opt is-off"
              data-testid="wb-opt-auto"
              onClick={() => {
                setPop('none');
                window.dispatchEvent(new CustomEvent('open-global-settings', { detail: { section: 'llm' } }));
              }}
            >
              <span className="wb-gl"><Sparkles size={13} strokeWidth={1.9} aria-hidden="true" /></span>
              <span className="wb-tx"><span className="wb-n1">{t('workbench.composer.auto')}</span></span>
              <span className="wb-note">{t('workbench.composer.needsKey')}</span>
              {ck(false)}
            </button>
          ) : (
            <button type="button" role="option" aria-selected={!projectId} className="wb-opt" data-testid="wb-opt-auto" onClick={() => { setProjectId(null); setPop('none'); }}>
              <span className="wb-gl"><Sparkles size={13} strokeWidth={1.9} aria-hidden="true" /></span>
              <span className="wb-tx"><span className="wb-n1">{t('workbench.composer.auto')}</span></span>
              {ck(!projectId)}
            </button>
          )}
          <div className="wb-sep" />
          {projects.map((p) => (
            <button
              key={p.project_id}
              type="button"
              role="option"
              aria-selected={effectiveProjectId === p.project_id}
              className="wb-opt"
              title={projectDisplayName(p, t)}
              onClick={() => { setProjectId(p.project_id); setPop('none'); }}
            >
              <span className="wb-gl"><Folder size={13} strokeWidth={1.9} aria-hidden="true" /></span>
              <span className="wb-tx"><span className="wb-n1">{projectDisplayName(p, t)}</span></span>
              {ck(effectiveProjectId === p.project_id)}
            </button>
          ))}
        </div>
      )}

      {pop === 'agent' && (
        <div className="wb-pop" role="listbox" aria-label={t('workbench.composer.agent')} style={{ left: Math.max(8, agentLeft + 18) }} data-testid="wb-pop-agent">
          <button type="button" role="option" aria-selected={!agent} className="wb-opt" onClick={() => { setAgent(null); setPop('none'); }}>
            <span className="wb-gl"><Sparkles size={13} strokeWidth={1.9} aria-hidden="true" /></span>
            <span className="wb-tx"><span className="wb-n1">{t('workbench.composer.auto')}</span></span>
            {ck(!agent)}
          </button>
          <div className="wb-sep" />
          {[{ slug: MAIN_AGENT_HANDLE, name: 'Orbital' }, ...agents].map((a) => {
            const line = modelLine(a.slug);
            return (
              <button
                key={a.slug}
                type="button"
                role="option"
                aria-selected={agent === a.slug}
                className="wb-opt"
                onClick={() => { setAgent(a.slug); setPop('none'); }}
              >
                <AgentMark slug={a.slug} size={22} />
                <span className="wb-tx">
                  <span className="wb-n1">{a.name}</span>
                  {line && <span className="wb-n2">{line}</span>}
                </span>
                {ck(agent === a.slug)}
              </button>
            );
          })}
          {/* Who does what lives in user memory (Global Settings), which the
              project's manager reads. Pinned to the bottom, so a long worker
              list never hides it. */}
          <div className="wb-pop-foot">
            <div className="wb-sep" />
            <button
              type="button"
              className="wb-opt wb-opt-muted"
              data-testid="wb-agent-prefs"
              onClick={() => {
                setPop('none');
                window.dispatchEvent(new CustomEvent('open-global-settings', { detail: { section: 'user-memory' } }));
              }}
            >
              <span className="wb-gl"><Settings2 size={13} strokeWidth={1.9} aria-hidden="true" /></span>
              <span className="wb-tx"><span className="wb-n1">{t('workbench.composer.whoDoesWhat')}</span></span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
