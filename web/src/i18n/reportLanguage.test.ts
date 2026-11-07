// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { afterEach, describe, expect, it, vi } from 'vitest';

const apiMock = vi.fn();
vi.mock('../config', () => ({ api: (...args: unknown[]) => apiMock(...args) }));

import { reportUiLanguage } from './reportLanguage';

afterEach(() => {
  apiMock.mockReset();
  vi.restoreAllMocks();
});

describe('reportUiLanguage', () => {
  it('sends the interface language and the device languages', async () => {
    vi.spyOn(navigator, 'languages', 'get').mockReturnValue(['ja-JP', 'en-US']);
    apiMock.mockResolvedValue({});
    await reportUiLanguage('zh');
    expect(apiMock).toHaveBeenCalledTimes(1);
    const [path, opts] = apiMock.mock.calls[0];
    expect(path).toBe('/api/v2/settings/ui-language');
    expect(opts.method).toBe('PUT');
    expect(JSON.parse(opts.body)).toEqual({ ui_language: 'zh', ui_languages: ['ja-JP', 'en-US'] });
  });

  it('swallows a daemon that is not reachable', async () => {
    apiMock.mockRejectedValue(new Error('offline'));
    await expect(reportUiLanguage('en')).resolves.toBeUndefined();
  });
});
