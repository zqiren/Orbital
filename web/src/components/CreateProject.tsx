// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState, useEffect, useId, useRef } from 'react';
import { ChevronDown, ChevronRight, Folder, FolderPlus, Lightbulb, Loader2, X } from 'lucide-react';
import type {
  Autonomy,
  ProjectCreateRequest,
} from '../types';
import LLMProviderSettings from './LLMProviderSettings';
import FolderBrowserPanel from './FolderBrowserPanel';
import { ApiError } from '../config';
import { useT } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';

interface CreateProjectProps {
  onSubmit: (data: ProjectCreateRequest) => Promise<void>;
  onCancel: () => void;
}

interface FormErrors {
  name?: string;
  workspace?: string;
}

const AUTONOMY_OPTIONS: {
  value: Autonomy;
  titleKey: StringKey;
  descriptionKey: StringKey;
}[] = [
  {
    value: 'hands_off',
    titleKey: 'autonomy.handsOff.title',
    descriptionKey: 'autonomy.handsOff.desc',
  },
  {
    value: 'check_in',
    titleKey: 'autonomy.checkIn.title',
    descriptionKey: 'autonomy.checkIn.desc',
  },
  {
    value: 'supervised',
    titleKey: 'autonomy.supervised.title',
    descriptionKey: 'autonomy.supervised.desc',
  },
];

/** Last path segment, cross-platform — the picked folder's display name. */
function basename(path: string): string {
  const parts = path.trim().replace(/[\\/]+$/, '').split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] || '';
}

export default function CreateProject({
  onSubmit,
  onCancel,
}: CreateProjectProps) {
  const t = useT();
  const [name, setName] = useState('');
  const [agentName, setAgentName] = useState('');
  const [workspace, setWorkspace] = useState('');
  const [instructions, setInstructions] = useState('');
  const [autonomy, setAutonomy] = useState<Autonomy>('hands_off');
  const [budgetLimit, setBudgetLimit] = useState('');
  const [errors, setErrors] = useState<FormErrors>({});
  const [pickerExpanded, setPickerExpanded] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const titleId = useId();
  const panelRef = useRef<HTMLDivElement>(null);
  const pickerRef = useRef<HTMLDivElement>(null);
  const nameInputRef = useRef<HTMLInputElement>(null);
  // The element focused before the modal mounted, restored when it unmounts.
  const prevFocusRef = useRef<HTMLElement | null>(null);

  // Esc-to-close. The modal is mounted only while open (App routes to
  // 'create'), so there is no `open` guard to check.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      // Escape raised inside the embedded folder picker belongs to the
      // picker — it cancels the inline "New folder" editor. The picker's own
      // handler neither stops propagation nor preventDefaults, so this
      // listener has to scope itself out; otherwise one Escape would both
      // cancel the folder editor AND tear down the modal, discarding
      // everything already typed into the form.
      //
      // composedPath(), NOT `picker.contains(e.target)`: the picker's Escape
      // handler unmounts the new-folder input, and React flushes that at the
      // microtask checkpoint the browser runs between listeners — so by the
      // time the event reaches `window` the target is already detached and
      // `contains` says false. The path is captured at dispatch time and
      // still records where the key was actually pressed.
      const picker = pickerRef.current;
      if (picker && e.composedPath().includes(picker)) return;
      onCancel();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onCancel]);

  // Autofocus the first field, then restore focus to the opener on unmount.
  //
  // Deliberately NOT React's `autoFocus` prop, and deliberately two frames
  // late — both because this app ships on pywebview→WKWebView, not Chromium:
  //  - the panel mounts mid `animate-slide-up`, i.e. translated 100% DOWN and
  //    off-screen. WebKit's focus scroll-into-view targets an element's
  //    VISUAL position, so focusing while it is still down there scrolls the
  //    document to chase it. `preventScroll` suppresses that; Chromium never
  //    showed it because it uses the layout position.
  //  - `autoFocus` fires synchronously on mount, before the animating layer
  //    has painted, and accepts no `preventScroll` option. The double rAF
  //    guarantees a paint landed before focus moves.
  useEffect(() => {
    prevFocusRef.current = document.activeElement as HTMLElement | null;
    let inner = 0;
    const outer = requestAnimationFrame(() => {
      inner = requestAnimationFrame(() => {
        nameInputRef.current?.focus({ preventScroll: true });
      });
    });
    return () => {
      cancelAnimationFrame(outer);
      if (inner) cancelAnimationFrame(inner);
      const prev = prevFocusRef.current;
      prevFocusRef.current = null;
      // Same rationale on restore — don't let refocusing the opener scroll.
      if (prev && document.contains(prev)) prev.focus({ preventScroll: true });
    };
  }, []);

  // Focus trap: keep Tab / Shift+Tab cycling within the modal.
  function handleTrapTab(e: React.KeyboardEvent<HTMLDivElement>) {
    if (e.key !== 'Tab') return;
    const panel = panelRef.current;
    if (!panel) return;
    const focusables = Array.from(
      panel.querySelectorAll<HTMLElement>(
        'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ),
    );
    if (focusables.length === 0) {
      e.preventDefault();
      return;
    }
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    const active = document.activeElement as HTMLElement | null;
    if (e.shiftKey) {
      if (active === first || !panel.contains(active)) {
        e.preventDefault();
        last.focus();
      }
    } else if (active === last || !panel.contains(active)) {
      e.preventDefault();
      first.focus();
    }
  }

  /** The folder is only ever set from the embedded browser (pick an existing
   * folder, or create one under the browsed path) — there is no free-text
   * path field here, so it is always a real directory the daemon listed or
   * just made. Name and folder are independent: picking a folder never
   * touches the name, so a name error (e.g. a 409 collision) is left alone. */
  function handleWorkspaceSelect(path: string) {
    setWorkspace(path);
    setErrors((prev) => ({ ...prev, workspace: undefined }));
    setPickerExpanded(false);
  }

  function handleNameChange(value: string) {
    setName(value);
    setErrors((prev) => ({ ...prev, name: undefined }));
  }

  function validate(): FormErrors {
    const e: FormErrors = {};
    if (!name.trim()) e.name = t('createProject.name.required');
    if (!workspace.trim()) e.workspace = t('createProject.workspace.required');
    return e;
  }

  const canSubmit = name.trim() !== '' && workspace.trim() !== '';

  async function handleSubmit(ev: React.FormEvent) {
    ev.preventDefault();
    if (submitting) return;
    const validationErrors = validate();
    setErrors(validationErrors);
    if (Object.keys(validationErrors).length > 0) return;

    setSubmitError(null);
    setSubmitting(true);
    try {
      await onSubmit({
        name: name.trim(),
        workspace: workspace.trim(),
        instructions: instructions.trim() || undefined,
        autonomy,
        agent_name: agentName.trim() || undefined,
        budget_limit_usd: budgetLimit ? parseFloat(budgetLimit) : undefined,
      });
    } catch (err) {
      // agent_name collisions (likelier now that the name is often
      // auto-derived from a folder basename) surface inline on the name
      // field — never auto-suffixed, since the name is user-visible identity.
      if (err instanceof ApiError && err.status === 409) {
        setErrors((prev) => ({ ...prev, name: err.detail }));
      } else {
        setSubmitError(err instanceof ApiError ? err.detail : t('createProject.submitError'));
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={(e) => { if (e.target === e.currentTarget) onCancel(); }}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onKeyDown={handleTrapTab}
        className="bg-background rounded-xl shadow-xl border border-border w-full max-w-[560px] max-h-[85vh] flex flex-col mx-4 animate-slide-up max-md:max-w-full max-md:max-h-full max-md:h-full max-md:mx-0 max-md:rounded-none"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-border shrink-0">
          <h2 id={titleId} className="text-sm font-semibold text-primary">{t('createProject.title')}</h2>
          <button
            type="button"
            onClick={onCancel}
            className="text-secondary hover:text-primary transition-all duration-150 p-1 max-md:min-h-[44px] max-md:min-w-[44px] max-md:flex max-md:items-center max-md:justify-center"
          >
            <X size={16} />
          </button>
        </div>

        {/* Scrollable body — only this area scrolls when Advanced is expanded */}
        <form id="create-project-form" onSubmit={handleSubmit} className="flex-1 overflow-y-auto min-h-0 px-5 py-4 space-y-5">
          {/* Project name — first, typed by the user, never derived. */}
          <div>
            <label className="block text-sm font-medium text-primary mb-1.5">
              {t('createProject.name.label')}
            </label>
            <div className="relative">
              <Folder
                size={16}
                aria-hidden="true"
                className="absolute left-3 top-1/2 -translate-y-1/2 text-secondary pointer-events-none"
              />
              <input
                ref={nameInputRef}
                type="text"
                value={name}
                onChange={(e) => handleNameChange(e.target.value)}
                placeholder={t('createProject.name.placeholder')}
                className="w-full text-sm bg-sidebar border border-border rounded-lg pl-9 pr-3 py-2.5 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
              />
            </div>
            {errors.name && (
              <p className="text-xs text-error mt-1">{errors.name}</p>
            )}
          </div>

          {/* Folder — a separate field, chosen by browsing (no path input in
              the default view; the browser panel keeps its own manual-path
              row for people who want to type one). */}
          <div>
            <label className="block text-sm font-medium text-primary mb-1.5">
              {t('createProject.workspace.label')}
            </label>
            {workspace ? (
              <div
                data-testid="create-project-folder-row"
                className="flex items-center gap-3 bg-sidebar border border-border rounded-lg px-3 py-2.5"
              >
                <Folder size={16} aria-hidden="true" className="shrink-0 text-accent" />
                <div className="min-w-0 flex-1">
                  <span className="block text-sm font-medium text-primary truncate">
                    {basename(workspace) || workspace}
                  </span>
                  <span className="block text-xs font-mono text-secondary truncate">
                    {workspace}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => setPickerExpanded((v) => !v)}
                  className="shrink-0 text-sm font-medium text-accent hover:text-accent/80 transition-all duration-150 max-md:min-h-[44px]"
                >
                  {t('createProject.folder.change')}
                </button>
              </div>
            ) : !pickerExpanded && (
              <button
                type="button"
                onClick={() => setPickerExpanded(true)}
                className="w-full flex flex-col items-center justify-center gap-1.5 border border-dashed border-secondary/35 rounded-lg px-4 py-7 text-center hover:border-accent/50 hover:bg-accent/5 transition-all duration-150"
              >
                <FolderPlus size={20} aria-hidden="true" className="text-secondary" />
                <span className="text-sm font-medium text-primary">
                  {t('createProject.folder.choose')}
                </span>
                <span className="text-xs text-secondary max-w-[340px]">
                  {t('createProject.folder.chooseHint')}
                </span>
              </button>
            )}
            {errors.workspace && (
              <p className="text-xs text-error mt-1">{errors.workspace}</p>
            )}
            {pickerExpanded && (
              <div
                ref={pickerRef}
                className={`border border-border rounded-lg overflow-hidden ${workspace ? 'mt-2' : ''}`}
              >
                <FolderBrowserPanel
                  compact
                  onSelect={handleWorkspaceSelect}
                  suggestedFolderName={name}
                />
              </div>
            )}
          </div>

          {/* What a project is — stated once, at the moment of the decision. */}
          <div className="flex items-start gap-3 bg-sidebar rounded-lg px-3.5 py-3">
            <Lightbulb size={16} aria-hidden="true" className="shrink-0 mt-0.5 text-secondary" />
            <p className="text-xs leading-relaxed text-secondary">
              {t('createProject.hint')}
            </p>
          </div>

          {/* LLM info: renders only the no-api-key warning; nothing otherwise */}
          <LLMProviderSettings mode="wizard" />

          {/* Advanced options (collapsed by default): Agent Name, Instructions, Autonomy, Budget */}
          <div>
            <button
              type="button"
              onClick={() => setAdvancedOpen((v) => !v)}
              className="flex items-center gap-1.5 text-sm font-medium text-secondary hover:text-primary transition-all duration-150"
            >
              {advancedOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              {t('createProject.advanced.label')}
            </button>

            {advancedOpen && (
              <div className="space-y-5 pt-4">
                {/* Agent Name */}
                <div>
                  <label className="block text-sm font-medium text-primary mb-1.5">
                    {t('createProject.agentName.label')} <span className="text-secondary font-normal">{t('createProject.agentName.optional')}</span>
                  </label>
                  <input
                    type="text"
                    value={agentName}
                    onChange={(e) => setAgentName(e.target.value)}
                    placeholder={t('createProject.agentName.placeholder')}
                    className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
                  />
                  <p className="text-xs text-secondary mt-1">
                    {t('createProject.agentName.hint')}
                  </p>
                </div>

                {/* Instructions */}
                <div>
                  <label className="block text-sm font-medium text-primary mb-1.5">
                    {t('createProject.instructions.label')}
                  </label>
                  <textarea
                    rows={5}
                    value={instructions}
                    onChange={(e) => setInstructions(e.target.value)}
                    placeholder={t('createProject.instructions.placeholder')}
                    className="w-full text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150 resize-y"
                  />
                </div>

                {/* Autonomy Level */}
                <div>
                  <label className="block text-sm font-medium text-primary mb-2">
                    {t('autonomy.level.label')}
                  </label>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    {AUTONOMY_OPTIONS.map((opt) => (
                      <button
                        key={opt.value}
                        type="button"
                        onClick={() => setAutonomy(opt.value)}
                        className={`text-left border rounded-lg p-3 transition-all duration-150 max-md:min-h-[44px] ${
                          autonomy === opt.value
                            ? 'border-accent bg-accent/5'
                            : 'border-border hover:border-secondary/40'
                        }`}
                      >
                        <span className="text-sm font-medium text-primary block">
                          {t(opt.titleKey)}
                        </span>
                        <span className="text-xs text-secondary mt-1 block">
                          {t(opt.descriptionKey)}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Budget Limit */}
                <div>
                  <label className="block text-sm font-medium text-primary mb-1.5">
                    {t('createProject.budget.label')} <span className="text-secondary font-normal">{t('createProject.agentName.optional')}</span>
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    value={budgetLimit}
                    onChange={(e) => setBudgetLimit(e.target.value)}
                    placeholder={t('createProject.budget.placeholder')}
                    className="w-48 text-sm bg-sidebar border border-border rounded-lg px-3 py-2 text-primary placeholder:text-secondary/60 focus:outline-none focus:border-accent transition-all duration-150"
                  />
                  <p className="text-xs text-secondary mt-1">
                    {t('createProject.budget.hint')}
                  </p>
                </div>
              </div>
            )}
          </div>

          {submitError && (
            <p className="text-sm text-error">{submitError}</p>
          )}
        </form>

        {/* Footer */}
        <div className="border-t border-border px-5 py-3 shrink-0 flex items-center justify-end gap-2 max-md:flex-col-reverse">
          <button
            type="button"
            onClick={onCancel}
            className="text-sm text-secondary hover:text-primary transition-all duration-150 px-4 py-2 max-md:w-full max-md:min-h-[44px]"
          >
            {t('createProject.cancel')}
          </button>
          <button
            type="submit"
            form="create-project-form"
            disabled={submitting || !canSubmit}
            className="inline-flex items-center justify-center gap-2 bg-accent text-white text-sm font-medium rounded-lg px-5 py-2.5 hover:bg-accent/90 transition-all duration-150 disabled:opacity-50 max-md:w-full max-md:min-h-[44px]"
          >
            {submitting && <Loader2 size={14} className="animate-spin" />}
            {t('createProject.deploy')}
          </button>
        </div>
      </div>
    </div>
  );
}
