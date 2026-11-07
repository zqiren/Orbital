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
 * Attachments (2026-11-05): a Plus button left of the project chip, paste and
 * drop onto the box, chips above the chip row — the chat composer's caps,
 * chip and copy. The composer only STAGES files (`deferUpload`): the send
 * uploads them once the project is known (after routing on 自动). The text
 * clears at once; the chips clear only after the send went out, so a send
 * that could not start keeps them. A sending task's chips stay on show
 * (uploading) but are not part of the next task.
 *
 * Routing found no project (2026-11-07): the text (and the staged chips) are
 * back, one error line heads the box and the project picker opens. The line
 * goes on the next send or when a project is picked.
 *
 * Keyless (no API key, 2026-11-05): routing needs Orbital's own model, so the
 * project chip never shows 自动. It starts on the last project sent to from
 * this browser (else Quick Tasks), and the picker's 自动 row stays visible
 * but not selectable, with a note; a click on it opens the model settings.
 */

import { useEffect, useRef, useState, type DragEvent, type KeyboardEvent } from 'react';
import { ArrowUp, Check, ChevronDown, Folder, Plus, Settings2, Sparkles } from 'lucide-react';
import type { Project } from '../../types';
import { useT } from '../../i18n/useT';
import { api } from '../../config';
import { getAgentIcon, MAIN_AGENT_HANDLE } from '../../utils/agentIcons';
import { projectDisplayName } from '../../utils/projectDisplayName';
import {
  modelLineText, workerModelInfo, type SubAgentModelEntry,
} from '../../hooks/useConversationModels';
import { restoreComposer, useWorkbenchFeed } from '../../hooks/useWorkbenchFeed';
import { MAX_ATTACHMENT_BYTES, useAttachments } from '../../hooks/useAttachments';
import AttachmentChip from '../AttachmentChip';
import type { SendChoice, SendDeps } from './sendTask';

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
  /** Resolves when the send settled; false = the text came back,
   *  'unrouted' = it came back because 自动 found no project. `uploadAll`
   *  uploads the choice's staged files once the project is known. */
  onSend: (choice: SendChoice, uploadAll: NonNullable<SendDeps['uploadAll']>) => Promise<boolean | 'unrouted'>;
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
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [attError, setAttError] = useState<string | null>(null);
  const [routeError, setRouteError] = useState(false);
  const {
    attachments, removeAttachment, retryAttachment, uploadAll,
    handleFilePickerChange, handlePaste, handleDrop,
  } = useAttachments('', { deferUpload: true, onError: setAttError });
  // Chips of a task still being sent: shown, but not part of the next task.
  const [inFlight, setInFlight] = useState<ReadonlySet<string>>(() => new Set());
  const [dragDepth, setDragDepth] = useState(0);
  // What the next send takes: staged, not sending, under the size cap.
  const sendable = attachments.filter((a) => !inFlight.has(a.id) && a.size <= MAX_ATTACHMENT_BYTES);

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
    const files = sendable.map((a) => ({ id: a.id, filename: a.filename }));
    if (!body && files.length === 0) return;
    const choice: SendChoice = { text: body, projectId: effectiveProjectId, agent, files };
    if (effectiveProjectId) {
      rememberLastProject(effectiveProjectId);
      setLastProject(effectiveProjectId);
    }
    setText('');
    setProjectId(null);
    setAgent(null);
    setPop('none');
    setAttError(null);
    setRouteError(false);
    const ids = files.map((f) => f.id);
    if (ids.length > 0) setInFlight((cur) => new Set([...cur, ...ids]));
    let ok = false;
    try {
      const result = await onSend(choice, uploadAll);
      ok = result === true;
      if (result === 'unrouted') {
        setRouteError(true);
        setPop('project');
      }
    } finally {
      // Gone only once the task went out; otherwise they stay for the retry.
      if (ok) for (const id of ids) removeAttachment(id);
      if (ids.length > 0) {
        setInFlight((cur) => {
          const next = new Set(cur);
          for (const id of ids) next.delete(id);
          return next;
        });
      }
    }
  }

  const hasFiles = (e: DragEvent) => !!e.dataTransfer?.types?.includes('Files');

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    // While an IME (e.g. Pinyin) is composing, Enter commits the candidate —
    // it belongs to the input method, not to us. Don't submit mid-composition.
    if (e.nativeEvent.isComposing || e.keyCode === 229) return;
    if (e.key === 'Enter' && !e.shiftKey) {
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
      <div
        className={`wb-box wb-layer${dragDepth > 0 ? ' is-drop' : ''}`}
        data-testid="wb-composer-box"
        onDragEnter={(e) => { if (!hasFiles(e)) return; e.preventDefault(); setDragDepth((n) => n + 1); }}
        onDragOver={(e) => { if (hasFiles(e)) e.preventDefault(); }}
        onDragLeave={(e) => { if (!hasFiles(e)) return; e.preventDefault(); setDragDepth((n) => Math.max(0, n - 1)); }}
        onDrop={(e) => { setDragDepth(0); setAttError(null); handleDrop(e); }}
      >
        {routeError && (
          <div className="wb-atterr" role="alert" data-testid="wb-route-error">
            {t('workbench.composer.routeFailed')}
          </div>
        )}
        <textarea
          ref={inputRef}
          className="wb-input"
          rows={2}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKey}
          onPaste={(e) => { setAttError(null); handlePaste(e); }}
          placeholder={t('workbench.composer.placeholder')}
          aria-label={t('workbench.composer.placeholder')}
          data-testid="wb-composer-input"
        />
        {attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 max-h-[100px] overflow-y-auto" data-testid="wb-chip-strip">
            {attachments.map((a) => {
              const sending = inFlight.has(a.id);
              // Staged = no badge (the shared chip spins on 'pending').
              const status = a.status === 'pending' ? (sending ? 'uploading' : undefined) : a.status;
              return (
                <AttachmentChip
                  key={a.id}
                  filename={a.filename}
                  mime={a.mime}
                  size={a.size}
                  status={status}
                  thumbnailUrl={a.thumbnailUrl}
                  errorMessage={a.errorMessage}
                  onRemove={sending ? undefined : () => removeAttachment(a.id)}
                  onRetry={a.status === 'error' && !sending && a.size <= MAX_ATTACHMENT_BYTES
                    ? () => retryAttachment(a.id) : undefined}
                />
              );
            })}
          </div>
        )}
        {attError && <div className="wb-atterr" role="alert" data-testid="wb-attach-error">{attError}</div>}
        <div className="wb-foot">
          <input
            type="file"
            multiple
            ref={fileInputRef}
            className="hidden"
            onChange={(e) => { setAttError(null); handleFilePickerChange(e); }}
            data-testid="wb-attach-input"
          />
          <button
            type="button"
            className="wb-plus"
            aria-label={t('chat.attachFiles')}
            title={t('chat.attachFiles')}
            data-testid="wb-attach"
            onClick={() => fileInputRef.current?.click()}
          >
            <Plus size={16} strokeWidth={2} aria-hidden="true" />
          </button>
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
            disabled={!text.trim() && sendable.length === 0}
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
              onClick={() => { setProjectId(p.project_id); setRouteError(false); setPop('none'); }}
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
