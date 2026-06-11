// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import {
  buildOverrideDoc,
  cellKey,
  effectiveCell,
  effectiveCurrency,
  emptyStaged,
  errorCodeToKey,
  hasStagedChanges,
  validateRateInput,
  validationReasonToKey,
  type PricingTable,
  type StagedEdits,
} from './pricing';

// A two-provider resolved table. openai/gpt-4o has ONE persisted override
// (output), the rest are defaults. anthropic has a persisted override currency
// AND a persisted input override on _default. This is the "other providers'
// overrides we must not lose" fixture.
function makeTable(): PricingTable {
  return {
    override_path: '/data/pricing-overrides.json',
    providers: {
      openai: {
        currency: 'USD',
        currency_origin: 'default',
        models: {
          'gpt-4o': {
            input_per_1m: { value: 2.5, origin: 'default' },
            cached_input_per_1m: { value: 1.25, origin: 'default' },
            cache_write_per_1m: { value: 3.0, origin: 'default' },
            output_per_1m: { value: 99.0, origin: 'override' }, // persisted override
          },
          _default: {
            input_per_1m: { value: 1.0, origin: 'default' },
            cached_input_per_1m: { value: 0.5, origin: 'default' },
            cache_write_per_1m: { value: 1.0, origin: 'default' },
            output_per_1m: { value: 2.0, origin: 'default' },
          },
        },
      },
      anthropic: {
        currency: 'CNY',
        currency_origin: 'override', // persisted currency override
        models: {
          _default: {
            input_per_1m: { value: 7.0, origin: 'override' }, // persisted override
            cached_input_per_1m: { value: 0.7, origin: 'default' },
            cache_write_per_1m: { value: 8.75, origin: 'default' },
            output_per_1m: { value: 21.0, origin: 'default' },
          },
        },
      },
    },
  };
}

describe('buildOverrideDoc — full-replace reconstruction', () => {
  it('preserves ALL persisted overrides across providers when nothing is staged', () => {
    const table = makeTable();
    const doc = buildOverrideDoc(table, emptyStaged());

    // openai keeps its output override and nothing else.
    expect(doc).toEqual({
      openai: { 'gpt-4o': { output_per_1m: 99.0 } },
      anthropic: { _currency: 'CNY', _default: { input_per_1m: 7.0 } },
    });
  });

  it('DATA-LOSS GUARD: editing one provider keeps the other provider’s overrides', () => {
    const table = makeTable();
    // Stage a NEW override on openai/gpt-4o input — must not wipe anthropic.
    const staged: StagedEdits = {
      ...emptyStaged(),
      rates: { [cellKey('openai', 'gpt-4o', 'input_per_1m')]: 3.33 },
    };
    const doc = buildOverrideDoc(table, staged);

    // anthropic's persisted currency + input override survive untouched.
    expect(doc.anthropic).toEqual({ _currency: 'CNY', _default: { input_per_1m: 7.0 } });
    // openai now carries BOTH the new input and the pre-existing output override.
    expect(doc.openai).toEqual({ 'gpt-4o': { input_per_1m: 3.33, output_per_1m: 99.0 } });
  });

  it('a staged revert removes EXACTLY one field, leaving siblings intact', () => {
    const table = makeTable();
    const staged: StagedEdits = {
      ...emptyStaged(),
      reverts: { [cellKey('openai', 'gpt-4o', 'output_per_1m')]: true },
    };
    const doc = buildOverrideDoc(table, staged);

    // openai/gpt-4o had only the output override → reverting it drops the model
    // (and the now-empty provider block) entirely.
    expect(doc.openai).toBeUndefined();
    // anthropic is wholly unaffected.
    expect(doc.anthropic).toEqual({ _currency: 'CNY', _default: { input_per_1m: 7.0 } });
  });

  it('revert of one field on a multi-override model keeps the other override', () => {
    const table = makeTable();
    // Add a second persisted override on anthropic/_default (output), then revert input.
    table.providers.anthropic.models._default.output_per_1m = { value: 50, origin: 'override' };
    const staged: StagedEdits = {
      ...emptyStaged(),
      reverts: { [cellKey('anthropic', '_default', 'input_per_1m')]: true },
    };
    const doc = buildOverrideDoc(table, staged);
    expect(doc.anthropic).toEqual({ _currency: 'CNY', _default: { output_per_1m: 50 } });
  });

  it('a staged edit overrides a persisted override value', () => {
    const table = makeTable();
    const staged: StagedEdits = {
      ...emptyStaged(),
      rates: { [cellKey('openai', 'gpt-4o', 'output_per_1m')]: 12.0 },
    };
    const doc = buildOverrideDoc(table, staged);
    expect(doc.openai).toEqual({ 'gpt-4o': { output_per_1m: 12.0 } });
  });

  it('a staged currency edit promotes a default-currency provider to override', () => {
    const table = makeTable();
    const staged: StagedEdits = {
      ...emptyStaged(),
      currencies: { openai: 'EUR' },
    };
    const doc = buildOverrideDoc(table, staged);
    // openai now carries _currency AND keeps its output override.
    expect(doc.openai).toEqual({ _currency: 'EUR', 'gpt-4o': { output_per_1m: 99.0 } });
  });

  it('a staged-edit-then-revert on the SAME field yields no field (revert wins)', () => {
    const table = makeTable();
    const k = cellKey('openai', 'gpt-4o', 'output_per_1m');
    const staged: StagedEdits = {
      ...emptyStaged(),
      rates: { [k]: 7 },
      reverts: { [k]: true },
    };
    const doc = buildOverrideDoc(table, staged);
    // Revert wins → openai/gpt-4o loses its only override → provider dropped.
    expect(doc.openai).toBeUndefined();
  });

  it('produces an empty doc when every override is reverted and nothing staged', () => {
    const table: PricingTable = {
      override_path: '/x',
      providers: {
        openai: {
          currency: 'USD',
          currency_origin: 'default',
          models: {
            'gpt-4o': {
              input_per_1m: { value: 1, origin: 'override' },
              cached_input_per_1m: { value: 1, origin: 'default' },
              cache_write_per_1m: { value: 1, origin: 'default' },
              output_per_1m: { value: 1, origin: 'default' },
            },
          },
        },
      },
    };
    const doc = buildOverrideDoc(table, {
      ...emptyStaged(),
      reverts: { [cellKey('openai', 'gpt-4o', 'input_per_1m')]: true },
    });
    expect(doc).toEqual({});
  });
});

describe('effectiveCell', () => {
  it('returns the resolved cell verbatim with no staged edits', () => {
    const table = makeTable();
    const c = effectiveCell(table, emptyStaged(), 'openai', 'gpt-4o', 'output_per_1m');
    expect(c).toEqual({ value: 99.0, origin: 'override', staged: false, reverted: false });
  });

  it('reflects a staged rate edit as override + staged', () => {
    const table = makeTable();
    const staged: StagedEdits = {
      ...emptyStaged(),
      rates: { [cellKey('openai', 'gpt-4o', 'input_per_1m')]: 4.4 },
    };
    const c = effectiveCell(table, staged, 'openai', 'gpt-4o', 'input_per_1m');
    expect(c).toEqual({ value: 4.4, origin: 'override', staged: true, reverted: false });
  });

  it('reflects a staged revert as default + reverted', () => {
    const table = makeTable();
    const staged: StagedEdits = {
      ...emptyStaged(),
      reverts: { [cellKey('openai', 'gpt-4o', 'output_per_1m')]: true },
    };
    const c = effectiveCell(table, staged, 'openai', 'gpt-4o', 'output_per_1m');
    expect(c.origin).toBe('default');
    expect(c.reverted).toBe(true);
  });
});

describe('effectiveCurrency', () => {
  it('returns the resolved currency + origin with no staged edit', () => {
    const table = makeTable();
    expect(effectiveCurrency(table, emptyStaged(), 'anthropic')).toEqual({
      value: 'CNY',
      origin: 'override',
      staged: false,
    });
  });
  it('reflects a staged currency edit', () => {
    const table = makeTable();
    const staged: StagedEdits = { ...emptyStaged(), currencies: { openai: 'GBP' } };
    expect(effectiveCurrency(table, staged, 'openai')).toEqual({
      value: 'GBP',
      origin: 'override',
      staged: true,
    });
  });
});

describe('hasStagedChanges', () => {
  it('false for empty staged', () => {
    expect(hasStagedChanges(emptyStaged())).toBe(false);
  });
  it('true with any staged rate / revert / currency', () => {
    expect(hasStagedChanges({ ...emptyStaged(), rates: { a: 1 } })).toBe(true);
    expect(hasStagedChanges({ ...emptyStaged(), reverts: { a: true } })).toBe(true);
    expect(hasStagedChanges({ ...emptyStaged(), currencies: { p: 'USD' } })).toBe(true);
  });
});

describe('errorCodeToKey', () => {
  it('maps each known backend code to its i18n key', () => {
    expect(errorCodeToKey('invalid_rate')).toBe('pricing.error.invalidRate');
    expect(errorCodeToKey('invalid_currency')).toBe('pricing.error.invalidCurrency');
    expect(errorCodeToKey('invalid_shape')).toBe('pricing.error.invalidShape');
    expect(errorCodeToKey('invalid_json')).toBe('pricing.error.invalidJson');
  });
  it('falls back to the generic key for an unknown code', () => {
    expect(errorCodeToKey('something_new')).toBe('pricing.error.generic');
    expect(errorCodeToKey('')).toBe('pricing.error.generic');
  });
});

describe('validateRateInput', () => {
  it('accepts a non-negative number', () => {
    expect(validateRateInput('3.5')).toEqual({ ok: true, value: 3.5 });
    expect(validateRateInput(' 0 ')).toEqual({ ok: true, value: 0 });
  });
  it('rejects empty input', () => {
    expect(validateRateInput('')).toEqual({ ok: false, reason: 'empty' });
    expect(validateRateInput('   ')).toEqual({ ok: false, reason: 'empty' });
  });
  it('rejects non-numeric input', () => {
    expect(validateRateInput('abc')).toEqual({ ok: false, reason: 'nan' });
    expect(validateRateInput('1.2.3')).toEqual({ ok: false, reason: 'nan' });
    expect(validateRateInput('Infinity')).toEqual({ ok: false, reason: 'nan' });
  });
  it('rejects negative input', () => {
    expect(validateRateInput('-1')).toEqual({ ok: false, reason: 'negative' });
  });
});

describe('validationReasonToKey', () => {
  it('maps reasons to i18n keys', () => {
    expect(validationReasonToKey('nan')).toBe('pricing.error.notNumber');
    expect(validationReasonToKey('negative')).toBe('pricing.error.negative');
    expect(validationReasonToKey('empty')).toBe('pricing.error.empty');
  });
});
