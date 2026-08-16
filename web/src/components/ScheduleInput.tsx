// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useT } from '../i18n/useT';
import Select from './Select';
import { FIELD } from './fieldStyles';
import {
  PRESET_LABEL_KEYS,
  SCHEDULE_PRESETS,
  cronForDraft,
  describeCron,
  timezoneOptions,
  weekdayName,
  type ScheduleDraft,
  type SchedulePreset,
} from './scheduleFormat';

function Label({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-[11px] uppercase tracking-wide text-secondary">{children}</span>
  );
}

/**
 * Preset dropdown + time picker + timezone, with a raw-cron escape hatch.
 *
 * Shared by create and edit — the two differ only in what seeds the draft.
 * The preview line under the controls is the caption that will be stored as
 * `schedule.human`, so what the user reads here is exactly what the row will
 * say afterwards.
 */
export default function ScheduleInput({
  value,
  onChange,
}: {
  value: ScheduleDraft;
  onChange: (next: ScheduleDraft) => void;
}) {
  const t = useT();
  const set = (patch: Partial<ScheduleDraft>) => onChange({ ...value, ...patch });

  const cron = cronForDraft(value);
  const described = describeCron(cron, t);
  const showTime = value.preset !== 'hourly' && value.preset !== 'custom';

  return (
    <div className="flex flex-col gap-3" data-testid="schedule-input">
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1">
          <Label>{t('trigger.form.repeat')}</Label>
          <Select
            value={value.preset}
            onChange={(e) => set({ preset: e.target.value as SchedulePreset })}
            aria-label={t('trigger.form.repeat')}
            data-testid="schedule-preset"
            className={`${FIELD} w-40`}
          >
            {SCHEDULE_PRESETS.map((preset) => (
              <option key={preset} value={preset}>
                {t(PRESET_LABEL_KEYS[preset])}
              </option>
            ))}
          </Select>
        </label>

        {value.preset === 'weekly' && (
          <label className="flex flex-col gap-1">
            <Label>{t('trigger.form.dayOfWeek')}</Label>
            <Select
              value={value.weekday}
              onChange={(e) => set({ weekday: Number(e.target.value) })}
              aria-label={t('trigger.form.dayOfWeek')}
              data-testid="schedule-weekday"
              className={`${FIELD} w-32`}
            >
              {[0, 1, 2, 3, 4, 5, 6].map((d) => (
                <option key={d} value={d}>
                  {weekdayName(d, t)}
                </option>
              ))}
            </Select>
          </label>
        )}

        {value.preset === 'monthly' && (
          <label className="flex flex-col gap-1">
            <Label>{t('trigger.form.dayOfMonth')}</Label>
            {/* Capped at 28: a "31st" schedule silently skips four months a
                year, which reads as a broken automation. */}
            <Select
              value={value.dayOfMonth}
              onChange={(e) => set({ dayOfMonth: Number(e.target.value) })}
              aria-label={t('trigger.form.dayOfMonth')}
              data-testid="schedule-dom"
              className={`${FIELD} w-20`}
            >
              {Array.from({ length: 28 }, (_, i) => i + 1).map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </Select>
          </label>
        )}

        {showTime && (
          <label className="flex flex-col gap-1">
            <Label>{t('trigger.form.time')}</Label>
            <input
              type="time"
              value={value.time}
              onChange={(e) => set({ time: e.target.value })}
              aria-label={t('trigger.form.time')}
              data-testid="schedule-time"
              className={`${FIELD} w-28`}
            />
          </label>
        )}

        {value.preset === 'hourly' && (
          <label className="flex flex-col gap-1">
            <Label>{t('trigger.form.minute')}</Label>
            <input
              type="number"
              min={0}
              max={59}
              value={value.minute}
              onChange={(e) =>
                set({ minute: Math.max(0, Math.min(59, Number(e.target.value) || 0)) })
              }
              aria-label={t('trigger.form.minute')}
              data-testid="schedule-minute"
              className={`${FIELD} w-20`}
            />
          </label>
        )}

        <label className="flex min-w-0 flex-col gap-1">
          <Label>{t('trigger.form.timezone')}</Label>
          <Select
            value={value.timezone}
            onChange={(e) => set({ timezone: e.target.value })}
            aria-label={t('trigger.form.timezone')}
            data-testid="schedule-timezone"
            className={`${FIELD} w-56 max-w-full`}
          >
            {timezoneOptions(value.timezone).map((zone) => (
              <option key={zone} value={zone}>
                {zone}
              </option>
            ))}
          </Select>
        </label>
      </div>

      {value.preset === 'custom' && (
        <label className="flex flex-col gap-1">
          <Label>{t('trigger.form.cron')}</Label>
          <input
            type="text"
            value={value.cron}
            onChange={(e) => set({ cron: e.target.value })}
            placeholder="0 7 * * *"
            aria-label={t('trigger.form.cron')}
            data-testid="schedule-cron"
            className={`${FIELD} font-mono`}
          />
          <span className="text-[11px] text-muted">{t('trigger.form.cronHint')}</span>
        </label>
      )}

      {/* Exactly what the row will read afterwards. For a custom cron with no
          preset phrasing, that is the expression itself. */}
      <p className="text-[11px] text-secondary" data-testid="schedule-preview">
        <span className="mr-1.5 uppercase tracking-wide text-muted">
          {t('trigger.form.preview')}
        </span>
        {described ?? <span className="font-mono">{cron || '—'}</span>}
      </p>
    </div>
  );
}
