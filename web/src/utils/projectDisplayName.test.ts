// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it } from 'vitest';
import { translate } from '../i18n/useT';
import { projectDisplayName } from './projectDisplayName';

const zh = (k: Parameters<typeof translate>[1]) => translate('zh', k);
const en = (k: Parameters<typeof translate>[1]) => translate('en', k);

describe('projectDisplayName', () => {
  it('renders the scratch project through the catalog in each locale', () => {
    const scratch = { name: 'Quick Tasks', is_scratch: true };
    expect(projectDisplayName(scratch, en)).toBe('Quick Tasks');
    expect(projectDisplayName(scratch, zh)).toBe('快速任务');
  });

  it('leaves every other project name alone', () => {
    expect(projectDisplayName({ name: 'Orbital-marketing', is_scratch: false }, zh)).toBe('Orbital-marketing');
    expect(projectDisplayName({ name: 'Orbital-marketing' }, zh)).toBe('Orbital-marketing');
  });
});
