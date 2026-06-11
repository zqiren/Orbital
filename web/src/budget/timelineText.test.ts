// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Logic gate for the budget-trip timeline TEXT (P3-G). Tests the SAME
// `budgetTimelineText` implementation ChatView's budget_event branch renders
// (web/src/budget/timelineText.ts) — no duplicated composition here.

import { describe, it, expect } from 'vitest';
import { translate } from '../i18n/useT';
import { budgetTimelineText, type TimelineTr } from './timelineText';

// React code binds tr to the active locale; mirror that for the zh cases.
const zhTr: TimelineTr = (k, v) => translate('zh', k, v);

describe('budgetTimelineText', () => {
  it('en pause: "queue paused" with spend/limit and window word', () => {
    const txt = budgetTimelineText({ action: 'pause', window: 'daily', spend: 1.2, limit: 1.0, currency: 'USD' });
    expect(txt).toContain('queue paused');
    expect(txt).toContain('$1.20');
    expect(txt).toContain('$1.00');
    expect(txt).toContain('today');
  });

  it('en stop: "queue stopped"', () => {
    const txt = budgetTimelineText({ action: 'stop', window: 'weekly', spend: 5, limit: 4, currency: 'USD' });
    expect(txt).toContain('queue stopped');
    expect(txt).toContain('this week');
  });

  it('en no-limit: omits the " of {limit}" clause', () => {
    const txt = budgetTimelineText({ action: 'pause', window: 'monthly', spend: 7, limit: null, currency: 'USD' });
    expect(txt).toContain('$7.00');
    expect(txt).not.toContain(' of ');
  });

  it('omitting locale and translator yields byte-identical English output', () => {
    const item = { action: 'pause' as const, window: 'daily' as const, spend: 1.2, limit: 1.0, currency: 'USD' };
    const enTr: TimelineTr = (k, v) => translate('en', k, v);
    expect(budgetTimelineText(item)).toBe(budgetTimelineText(item, 'en', enTr));
  });

  it('zh pause: localized message, window word leads, CNY symbol', () => {
    const txt = budgetTimelineText(
      { action: 'pause', window: 'daily', spend: 8, limit: 6, currency: 'CNY' },
      'zh',
      zhTr,
    );
    expect(txt).toContain('队列已暂停');
    expect(txt).toContain('今日');
    expect(txt).toContain('¥8.00');
    expect(txt).toContain('¥6.00');
  });

  it('zh stop: distinct stopped wording', () => {
    const txt = budgetTimelineText(
      { action: 'stop', window: 'total', spend: 100, limit: 80, currency: 'CNY' },
      'zh',
      zhTr,
    );
    expect(txt).toContain('队列已停止');
    expect(txt).toContain('总计');
  });
});
