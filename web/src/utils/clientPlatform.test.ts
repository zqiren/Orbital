// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { afterEach, describe, expect, it, vi } from 'vitest';

const relay = vi.hoisted(() => ({ value: false }));
vi.mock('../config', () => ({
  get isRelayMode() {
    return relay.value;
  },
}));

import { canRevealInFileManager, clientOS, isWindowsClient, revealLabelKey } from './clientPlatform';

afterEach(() => {
  relay.value = false;
});

describe('clientOS', () => {
  it('prefers userAgentData.platform (Chromium, WebView2)', () => {
    expect(clientOS({ platform: 'MacIntel', userAgentData: { platform: 'Windows' } })).toBe('windows');
    expect(clientOS({ userAgentData: { platform: 'macOS' } })).toBe('mac');
  });

  it('falls back to navigator.platform (WebKit / the pywebview app)', () => {
    expect(clientOS({ platform: 'MacIntel' })).toBe('mac');
    expect(clientOS({ platform: 'Win32' })).toBe('windows');
  });

  it('anything else has no file manager Orbital can drive', () => {
    expect(clientOS({ platform: 'Linux x86_64' })).toBe('other');
    expect(clientOS({ platform: 'iPhone' })).toBe('other');
    expect(clientOS({ userAgentData: { platform: 'Android' } })).toBe('other');
    expect(clientOS({ platform: '' })).toBe('other');
    expect(clientOS({})).toBe('other');
  });
});

describe('labels', () => {
  it('Windows says File Explorer, everything else Finder', () => {
    expect(isWindowsClient({ platform: 'Win32' })).toBe(true);
    expect(revealLabelKey({ platform: 'Win32' })).toBe('fileExplorer.revealExplorer');
    expect(isWindowsClient({ platform: 'MacIntel' })).toBe(false);
    expect(revealLabelKey({ platform: 'MacIntel' })).toBe('fileExplorer.revealFinder');
  });
});

describe('canRevealInFileManager', () => {
  it('is on for a desktop client talking to its own daemon', () => {
    expect(canRevealInFileManager({ platform: 'MacIntel' })).toBe(true);
    expect(canRevealInFileManager({ platform: 'Win32' })).toBe(true);
  });

  it('is off through the relay — Finder would open on the desktop, not the phone', () => {
    relay.value = true;
    expect(canRevealInFileManager({ platform: 'MacIntel' })).toBe(false);
  });

  it('is off on a client with no supported file manager (a phone on the LAN, Linux)', () => {
    expect(canRevealInFileManager({ platform: 'iPhone' })).toBe(false);
    expect(canRevealInFileManager({ platform: 'Linux x86_64' })).toBe(false);
  });
});
