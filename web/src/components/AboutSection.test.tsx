// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import AboutSection from './AboutSection';
import { LocaleProvider } from '../i18n/LocaleContext';

const INSTALLER = {
  global: 'https://github.com/zqiren/Orbital/releases/latest/download/Orbital-Setup.exe',
  china: 'https://orbital-release.oss-cn-hongkong.aliyuncs.com/releases/Orbital-Setup.exe',
};
let status: unknown;

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal('fetch', () =>
    Promise.resolve({ json: () => Promise.resolve(status) } as Response));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

// The About row's "available" link goes where the pill goes.
describe('AboutSection update link', () => {
  it('downloads the installer, from the mirror the UI language picks', async () => {
    status = { current: '0.8.4', update_available: true, latest: '0.9.0', url: 'https://site', installer: INSTALLER };
    render(<AboutSection />);
    expect((await screen.findByRole('link')).getAttribute('href')).toBe(INSTALLER.global);
    cleanup();
    localStorage.setItem('orbital.locale', 'zh');
    render(<LocaleProvider><AboutSection /></LocaleProvider>);
    expect((await screen.findByRole('link')).getAttribute('href')).toBe(INSTALLER.china);
  });

  it('keeps the website link where there is no installer (Linux)', async () => {
    status = { current: '0.8.4', update_available: true, latest: '0.9.0', url: 'https://site', installer: null };
    render(<AboutSection />);
    expect((await screen.findByRole('link')).getAttribute('href')).toBe('https://site');
  });
});
