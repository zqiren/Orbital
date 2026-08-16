// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Calendar, FolderOpen, Pencil, Plus, Trash2 } from 'lucide-react';
import { useTriggers, type TriggerDraft } from '../hooks/useTriggers';
import type { Trigger } from '../types';
import { translate, useT } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import AutomationForm from './AutomationForm';
import { describeCron, type Translate } from './scheduleFormat';

interface AutomationsListProps {
  projectId: string;
}

function formatLastFired(last: string | null): string {
  if (!last) return '—';
  try {
    const d = new Date(last);
    if (isNaN(d.getTime())) return '—';
    return d.toLocaleString();
  } catch {
    return '—';
  }
}

/**
 * The house switch for a trigger, lifted from TriggerStrip and upgraded to
 * role="switch" (the a11y shape TelemetrySettings already uses) now that it is
 * the primary control rather than a detail-panel extra.
 */
function Toggle({
  checked,
  label,
  testId,
  onChange,
  disabled,
}: {
  checked: boolean;
  label: string;
  testId: string;
  onChange: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      data-testid={testId}
      disabled={disabled}
      onClick={onChange}
      className={`relative w-9 h-5 shrink-0 rounded-full transition-colors duration-150 disabled:opacity-50 ${
        checked ? 'bg-accent' : 'bg-border'
      }`}
    >
      <span
        aria-hidden="true"
        className={`absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-white transition-all duration-150 ${
          checked ? 'translate-x-4' : ''
        }`}
      />
    </button>
  );
}

function AutomationRow({
  trigger,
  onToggle,
  onDelete,
  onEdit,
  describe,
}: {
  trigger: Trigger;
  onToggle: () => void;
  onDelete: () => Promise<void>;
  onEdit: () => void;
  describe: (cron: string) => string | null;
}) {
  const t = useT();
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const isSchedule = trigger.type === 'schedule';
  const cron = trigger.schedule?.cron ?? '';
  // Prefer a caption re-derived from the cron in the reader's language over the
  // stored one, which is frozen in whatever language wrote it (the agent's).
  const described = isSchedule ? describe(cron) : null;
  const condition = isSchedule
    ? (described ?? trigger.schedule?.human ?? cron ?? '—')
    : (trigger.watch_path ?? '—');
  const conditionLabel = isSchedule ? t('trigger.schedule') : t('trigger.watching');
  // Mono is for identifiers — a raw cron or a path. A phrased schedule is prose.
  const conditionMono = !isSchedule || described === null;

  const handleDelete = async () => {
    setDeleting(true);
    try {
      await onDelete();
    } finally {
      setDeleting(false);
      setConfirmDelete(false);
    }
  };

  return (
    // Same card family as the queue items — identical radius, border weight and
    // padding — but deliberately FLAT (no shadow, lighter hairline). An
    // automation is a definition, not work in flight: it produces queue items
    // rather than being one. Keeping it in the family but one step back in the
    // depth hierarchy says "related, different kind of thing" without needing a
    // second visual language. The left slot reinforces it: queue items carry a
    // state dot, automations carry a type icon.
    <div
      className="flex flex-col gap-2 rounded-xl border border-border/50 bg-card px-4 py-3"
      data-testid={`automation-card-${trigger.id}`}
    >
      <div className="flex items-center gap-2">
        {isSchedule ? (
          <Calendar className="w-3.5 h-3.5 shrink-0 text-muted" aria-hidden />
        ) : (
          <FolderOpen className="w-3.5 h-3.5 shrink-0 text-muted" aria-hidden />
        )}
        <span className="flex-1 truncate text-[13px] font-medium text-primary">
          {trigger.name}
        </span>
        {/* Dot + label, matching the queue's state markers, instead of a
            tinted pill. Enabled/disabled stays legible because a paused
            automation is a silent failure mode. */}
        <span
          className={`inline-flex shrink-0 items-center gap-1.5 text-[11px] ${
            trigger.enabled ? 'text-secondary' : 'text-muted'
          }`}
          data-testid={`automation-status-${trigger.id}`}
        >
          <span
            aria-hidden="true"
            className={`h-1.5 w-1.5 shrink-0 rounded-full ${
              trigger.enabled ? 'bg-success' : 'bg-idle'
            }`}
          />
          {trigger.enabled ? t('automations.on') : t('automations.off')}
        </span>
        <Toggle
          checked={trigger.enabled}
          label={t('trigger.toggleAria', { name: trigger.name })}
          testId={`automation-toggle-${trigger.id}`}
          onChange={onToggle}
        />
        <button
          type="button"
          onClick={onEdit}
          aria-label={t('trigger.editAria', { name: trigger.name })}
          data-testid={`automation-edit-${trigger.id}`}
          className="shrink-0 p-1 text-secondary hover:text-primary transition-colors max-md:min-h-[44px] max-md:min-w-[44px] max-md:flex max-md:items-center max-md:justify-center"
        >
          <Pencil size={14} />
        </button>
        <button
          type="button"
          onClick={() => setConfirmDelete(true)}
          aria-label={t('trigger.deleteAria')}
          data-testid={`automation-delete-${trigger.id}`}
          className="shrink-0 p-1 text-secondary hover:text-error transition-colors max-md:min-h-[44px] max-md:min-w-[44px] max-md:flex max-md:items-center max-md:justify-center"
        >
          <Trash2 size={14} />
        </button>
      </div>

      {/* Two-step inline confirm — the established destructive pattern for
          triggers (TriggerStrip shipped it), not a native confirm(). */}
      {confirmDelete && (
        <div
          className="flex items-center gap-2 pl-[22px]"
          data-testid={`automation-delete-confirm-${trigger.id}`}
        >
          <span className="text-xs text-secondary">{t('trigger.deleteConfirm')}</span>
          <button
            type="button"
            onClick={() => setConfirmDelete(false)}
            disabled={deleting}
            className="text-xs text-secondary hover:text-primary transition-colors"
          >
            {t('trigger.delete.cancelBtn')}
          </button>
          <button
            type="button"
            onClick={handleDelete}
            disabled={deleting}
            data-testid={`automation-delete-confirm-btn-${trigger.id}`}
            className="text-xs text-error hover:text-error/80 transition-colors"
          >
            {deleting ? t('trigger.deleting') : t('trigger.delete.confirmBtn')}
          </button>
        </div>
      )}

      <div className="flex flex-col gap-1 pl-[22px]">
        <p className="text-[11px] text-muted">
          <span className="mr-1.5 uppercase tracking-wide">{conditionLabel}</span>
          <span
            className={`text-secondary ${conditionMono ? 'font-mono' : ''}`}
            data-testid={`automation-condition-${trigger.id}`}
          >
            {condition}
          </span>
        </p>
        <p className="text-[11px] text-muted">
          <span className="mr-1.5 uppercase tracking-wide">{t('automations.lastFired')}</span>
          <span
            className="font-mono tabular-nums text-secondary"
            data-testid={`automation-last-fired-${trigger.id}`}
          >
            {formatLastFired(trigger.last_triggered)}
          </span>
          {trigger.trigger_count > 0 && (
            <>
              <span className="mx-1.5 uppercase tracking-wide">{t('trigger.runs')}</span>
              <span
                className="font-mono tabular-nums text-secondary"
                data-testid={`automation-runs-${trigger.id}`}
              >
                {trigger.trigger_count}
              </span>
            </>
          )}
        </p>
      </div>
    </div>
  );
}

/**
 * The Automations pane: the one place a project's triggers are managed.
 *
 * Toggle / edit / delete / create all live here. Rows update IN PLACE from the
 * `trigger.updated` WS event — before that event existed, a disable arrived as
 * `trigger.deleted` and the row disappeared until the next refetch, which is
 * why this surface shipped read-only.
 */
export default function AutomationsList({ projectId }: AutomationsListProps) {
  const t = useT();
  const { locale } = useLocale();
  const {
    triggers,
    loading,
    fetchTriggers,
    createTrigger,
    updateTrigger,
    toggleTrigger,
    deleteTrigger,
  } = useTriggers(projectId);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void fetchTriggers();
  }, [fetchTriggers]);

  // Bound to `locale`, not the unstable `t` identity (i18n house rule).
  const describe = useMemo(() => {
    const tr: Translate = (k, v) => translate(locale, k, v);
    return (cron: string) => (cron ? describeCron(cron, tr) : null);
  }, [locale]);

  const handleToggle = useCallback(
    async (trigger: Trigger) => {
      setError(null);
      try {
        await toggleTrigger(trigger.id, !trigger.enabled);
      } catch (err) {
        setError(err instanceof Error ? err.message : t('trigger.form.saveFailed'));
      }
    },
    // t is unstable; the message only needs the locale bound.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [toggleTrigger, locale],
  );

  const handleDelete = useCallback(
    async (triggerId: string) => {
      setError(null);
      try {
        await deleteTrigger(triggerId);
      } catch (err) {
        setError(err instanceof Error ? err.message : t('trigger.form.saveFailed'));
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [deleteTrigger, locale],
  );

  const handleSubmit = useCallback(
    async (draft: TriggerDraft) => {
      if (editingId) {
        // `type` is not editable, and `enabled` belongs to the row toggle —
        // sending either from the form would let a stale value clobber it.
        const { type: _type, enabled: _enabled, ...patch } = draft;
        void _type;
        void _enabled;
        await updateTrigger(editingId, patch);
        setEditingId(null);
      } else {
        await createTrigger(draft);
        setCreating(false);
      }
    },
    [createTrigger, updateTrigger, editingId],
  );

  const editing = editingId ? triggers.find((tr) => tr.id === editingId) : undefined;

  return (
    <div className="flex flex-col gap-3" data-testid="automations-pane">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs text-secondary">
          {triggers.length > 0
            ? t(triggers.length === 1 ? 'trigger.summary.one' : 'trigger.summary.other', {
                n: triggers.length,
              })
            : ''}
        </span>
        <button
          type="button"
          onClick={() => {
            setEditingId(null);
            setCreating(true);
          }}
          data-testid="automations-new"
          className="inline-flex items-center gap-1.5 text-xs font-medium text-accent hover:text-accent/80 transition-colors max-md:min-h-[44px]"
        >
          <Plus size={14} aria-hidden />
          {t('trigger.new')}
        </button>
      </div>

      {error && (
        <p
          className="text-xs text-error border border-error/40 bg-error/5 rounded px-2.5 py-2"
          data-testid="automations-error"
        >
          {error}
        </p>
      )}

      {creating && (
        <AutomationForm onSubmit={handleSubmit} onCancel={() => setCreating(false)} />
      )}

      {loading && triggers.length === 0 && (
        <p className="text-sm text-secondary px-1 italic" data-testid="automations-loading">
          {t('automations.loading')}
        </p>
      )}

      {!loading && triggers.length === 0 && !creating && (
        <p className="text-sm text-secondary px-1 italic" data-testid="automations-empty">
          {t('automations.empty')}
        </p>
      )}

      {triggers.length > 0 && (
        <div className="flex flex-col gap-2" data-testid="automations-list">
          {triggers.map((trigger) =>
            editing && editing.id === trigger.id ? (
              <AutomationForm
                key={trigger.id}
                trigger={editing}
                onSubmit={handleSubmit}
                onCancel={() => setEditingId(null)}
              />
            ) : (
              <AutomationRow
                key={trigger.id}
                trigger={trigger}
                describe={describe}
                onToggle={() => void handleToggle(trigger)}
                onDelete={() => handleDelete(trigger.id)}
                onEdit={() => {
                  setCreating(false);
                  setEditingId(trigger.id);
                }}
              />
            ),
          )}
        </div>
      )}
    </div>
  );
}
