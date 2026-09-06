// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useMemo, useState } from 'react';
import { translate, useT } from '../i18n/useT';
import { useLocale } from '../i18n/LocaleContext';
import type { Trigger } from '../types';
import type { TriggerDraft } from '../hooks/useTriggers';
import ScheduleInput from './ScheduleInput';
import PinTargetSelect from './PinTargetSelect';
import Select from './Select';
import { FIELD, FIELD_MULTILINE } from './fieldStyles';
import {
  cronForDraft,
  draftFromCron,
  emptyScheduleDraft,
  humanForCron,
  type ScheduleDraft,
} from './scheduleFormat';

/** Longest derived name. Past this the row's own truncation takes over, and a
 *  name that has to be truncated to be read is not doing its job anyway. */
const DERIVED_NAME_MAX = 48;

/**
 * A name for an automation whose author did not give it one.
 *
 * The prompt is the definition; the name is a label for it. Deriving from the
 * first line rather than the whole thing keeps a multi-paragraph brief from
 * turning into a 48-character run-on.
 *
 * Pure and un-translated on purpose: every character it returns came from the
 * user's own prompt, so there is nothing here to localise (the i18n rule for
 * non-React helpers is about UI chrome, and this produces none).
 */
export function deriveName(task: string): string {
  const firstLine = task.trim().split('\n')[0]!.replace(/\s+/g, ' ').trim();
  return firstLine.length > DERIVED_NAME_MAX
    ? `${firstLine.slice(0, DERIVED_NAME_MAX).trimEnd()}…`
    : firstLine;
}

interface AutomationFormProps {
  /** Edit target. Absent = create. */
  trigger?: Trigger;
  onSubmit: (draft: TriggerDraft) => Promise<void>;
  onCancel: () => void;
  /**
   * Spec 079 — installed sub-agents offered as the runner for this
   * automation. Empty (the default) hides the picker entirely, so a user with
   * no workers sees the form exactly as it is today.
   */
  agents?: Array<{ slug: string; name: string }>;
}

/**
 * One form for create and edit — they differ only in what seeds the state and
 * in whether the type can still be chosen. Type is fixed after create: a
 * schedule and a file-watch automation have different required fields and arm
 * different machinery, so "changing" one is really creating another.
 *
 * Cron/timezone validity is decided by the server (croniter + pytz live in
 * Python only). Its 400 detail is shown inline, verbatim — a backend string,
 * not UI chrome, so it isn't translated.
 */
export default function AutomationForm({ trigger, onSubmit, onCancel, agents = [] }: AutomationFormProps) {
  const t = useT();
  const { locale } = useLocale();
  const isEdit = trigger !== undefined;

  const [type, setType] = useState<'schedule' | 'file_watch'>(
    trigger?.type ?? 'schedule',
  );
  const [name, setName] = useState(trigger?.name ?? '');
  const [task, setTask] = useState(trigger?.task ?? '');
  const [schedule, setSchedule] = useState<ScheduleDraft>(() =>
    trigger?.schedule
      ? draftFromCron(trigger.schedule.cron, trigger.schedule.timezone)
      : emptyScheduleDraft(),
  );
  const [watchPath, setWatchPath] = useState(trigger?.watch_path ?? '');
  const [patterns, setPatterns] = useState((trigger?.patterns ?? []).join(', '));
  const [recursive, setRecursive] = useState(trigger?.recursive ?? false);
  const [debounce, setDebounce] = useState(String(trigger?.debounce_seconds ?? 5));
  // Spec 079 — who runs this automation. Unlike the queue composer's per-item
  // choice this one is part of the record, so edit seeds it from the trigger.
  const [agent, setAgent] = useState<string | null>(trigger?.agent ?? null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  // The caption stored in `schedule.human` is built from the catalog in the
  // CURRENT locale — bound to `locale`, not to the unstable `t` identity.
  const humanize = useMemo(
    () => (cron: string) => humanForCron(cron, (k, v) => translate(locale, k, v)),
    [locale],
  );

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (saving) return;

    // Name is optional. It was the first, largest and most prominent field on
    // the form while being the least load-bearing thing on it: the prompt is
    // what defines the automation, and the name only labels it. Left blank it
    // is taken from the prompt, which is why the prompt check comes first —
    // there is no name to derive without one.
    if (!task.trim()) return setError(t('trigger.form.promptRequired'));

    const draft: TriggerDraft = {
      name: name.trim() || deriveName(task),
      type,
      task: task.trim(),
      // Always sent, null included: on edit an omitted field would leave a
      // previously chosen worker in place, making "back to Orbital" impossible
      // to express (the PATCH route reads presence, not nullness).
      agent,
    };
    if (type === 'schedule') {
      const cron = cronForDraft(schedule);
      if (!cron) return setError(t('trigger.form.cronRequired'));
      draft.schedule = { cron, human: humanize(cron), timezone: schedule.timezone };
    } else {
      if (!watchPath.trim()) return setError(t('trigger.form.watchPathRequired'));
      draft.watch_path = watchPath.trim();
      draft.patterns = patterns
        .split(',')
        .map((p) => p.trim())
        .filter(Boolean);
      draft.recursive = recursive;
      draft.debounce_seconds = Math.max(0, Number(debounce) || 0);
    }

    setError(null);
    setSaving(true);
    try {
      await onSubmit(draft);
    } catch (err) {
      setError(err instanceof Error ? err.message : t('trigger.form.saveFailed'));
    } finally {
      setSaving(false);
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      data-testid="automation-form"
      className="flex flex-col gap-4 rounded-xl border border-border bg-card px-4 py-4"
    >
      <p className="text-[13px] font-medium text-primary">
        {isEdit ? t('trigger.form.title.edit') : t('trigger.new')}
      </p>

      <label className="flex flex-col gap-1">
        <span className="text-[11px] uppercase tracking-wide text-secondary">
          {t('trigger.form.name')}
        </span>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={t('trigger.form.namePlaceholder')}
          aria-label={t('trigger.form.name')}
          data-testid="automation-form-name"
          className={FIELD}
        />
      </label>

      <div className="flex flex-col gap-1">
        <span className="text-[11px] uppercase tracking-wide text-secondary">
          {t('trigger.form.type')}
        </span>
        {isEdit ? (
          <span className="text-sm text-secondary" data-testid="automation-form-type-fixed">
            {type === 'schedule'
              ? t('trigger.form.type.schedule')
              : t('trigger.form.type.fileWatch')}
          </span>
        ) : (
          <Select
            value={type}
            onChange={(e) => setType(e.target.value as 'schedule' | 'file_watch')}
            aria-label={t('trigger.form.type')}
            data-testid="automation-form-type"
            className={`${FIELD} w-full`}
          >
            <option value="schedule">{t('trigger.form.type.schedule')}</option>
            <option value="file_watch">{t('trigger.form.type.fileWatch')}</option>
          </Select>
        )}
      </div>

      {type === 'schedule' ? (
        <ScheduleInput value={schedule} onChange={setSchedule} />
      ) : (
        <div className="flex flex-col gap-3">
          <label className="flex flex-col gap-1">
            <span className="text-[11px] uppercase tracking-wide text-secondary">
              {t('trigger.form.watchPath')}
            </span>
            <input
              type="text"
              value={watchPath}
              onChange={(e) => setWatchPath(e.target.value)}
              placeholder="incoming"
              aria-label={t('trigger.form.watchPath')}
              data-testid="automation-form-watch-path"
              className={`${FIELD} font-mono`}
            />
            <span className="text-[11px] text-muted">{t('trigger.form.watchPathHint')}</span>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[11px] uppercase tracking-wide text-secondary">
              {t('trigger.form.patterns')}
            </span>
            <input
              type="text"
              value={patterns}
              onChange={(e) => setPatterns(e.target.value)}
              placeholder="*.jpg, *.png"
              aria-label={t('trigger.form.patterns')}
              data-testid="automation-form-patterns"
              className={`${FIELD} font-mono`}
            />
            <span className="text-[11px] text-muted">{t('trigger.form.patternsHint')}</span>
          </label>
          <div className="flex flex-wrap items-end gap-4">
            <label className="flex items-center gap-2 text-sm text-primary">
              <input
                type="checkbox"
                checked={recursive}
                onChange={(e) => setRecursive(e.target.checked)}
                data-testid="automation-form-recursive"
                className="accent-accent"
              />
              {t('trigger.form.recursive')}
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[11px] uppercase tracking-wide text-secondary">
                {t('trigger.form.debounce')}
              </span>
              <input
                type="number"
                min={0}
                value={debounce}
                onChange={(e) => setDebounce(e.target.value)}
                aria-label={t('trigger.form.debounce')}
                data-testid="automation-form-debounce"
                className={`${FIELD} w-24`}
              />
            </label>
          </div>
        </div>
      )}

      {agents.length > 0 && (
        <div className="flex flex-col gap-1" data-testid="automation-form-agent-field">
          <span className="text-[11px] uppercase tracking-wide text-secondary">
            {t('automation.agent.label')}
          </span>
          <div className="flex items-center gap-2">
            <PinTargetSelect
              agents={agents}
              value={agent}
              onChange={setAgent}
              variant="standalone"
              managerLabel={t('automation.agent.aria')}
              data-testid="automation-form-agent"
            />
            <span className="text-xs text-secondary">
              {agent
                ? (agents.find((a) => a.slug === agent)?.name ?? agent)
                : t('pinAgent.orbital')}
            </span>
          </div>
        </div>
      )}

      <label className="flex flex-col gap-1">
        <span className="text-[11px] uppercase tracking-wide text-secondary">
          {t('trigger.task')}
        </span>
        <textarea
          value={task}
          onChange={(e) => setTask(e.target.value)}
          rows={3}
          placeholder={t('trigger.form.promptPlaceholder')}
          aria-label={t('trigger.task')}
          data-testid="automation-form-prompt"
          className={`${FIELD_MULTILINE} resize-y leading-relaxed`}
        />
      </label>

      {error && (
        <p
          className="text-xs text-error border border-error/40 bg-error/5 rounded px-2.5 py-2"
          data-testid="automation-form-error"
        >
          {error}
        </p>
      )}

      {/* A footer, not a pair of buttons at the end of a stack. Negative
          margins pull it out to the card's edges so the rule reads as the
          bottom of the form; text-xs was the same size as the field HINTS for
          what is the terminal action of a card this tall. */}
      <div className="-mx-4 -mb-4 mt-1 flex items-center gap-2 border-t border-border/60 px-4 py-3">
        <button
          type="submit"
          disabled={saving}
          data-testid="automation-form-save"
          className="text-[13px] font-medium px-4 py-2 rounded-lg bg-accent text-white disabled:opacity-60 transition-colors max-md:min-h-[44px]"
        >
          {saving
            ? t('trigger.form.saving')
            : isEdit
              ? t('trigger.form.save')
              : t('trigger.form.create')}
        </button>
        <button
          type="button"
          onClick={onCancel}
          disabled={saving}
          data-testid="automation-form-cancel"
          className="text-[13px] text-secondary hover:text-primary transition-colors px-3 py-2 max-md:min-h-[44px]"
        >
          {t('trigger.form.cancel')}
        </button>
      </div>
    </form>
  );
}
