// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { resolveAddress, displayAddress } from './browserAddress';

describe('resolveAddress', () => {
  it('uses a web URL as typed', () => {
    expect(resolveAddress('https://example.com/a?b=1')).toBe('https://example.com/a?b=1');
    expect(resolveAddress(' http://example.com ')).toBe('http://example.com');
  });

  it('adds https to something that reads as a host', () => {
    expect(resolveAddress('example.com')).toBe('https://example.com');
    expect(resolveAddress('news.ycombinator.com/newest')).toBe('https://news.ycombinator.com/newest');
    expect(resolveAddress('example.com:8443/x')).toBe('https://example.com:8443/x');
  });

  it('uses http for localhost and bare IPs', () => {
    expect(resolveAddress('localhost:5174')).toBe('http://localhost:5174');
    expect(resolveAddress('127.0.0.1:8000/api')).toBe('http://127.0.0.1:8000/api');
    expect(resolveAddress('192.168.31.107')).toBe('http://192.168.31.107');
  });

  it('turns words into a search', () => {
    expect(resolveAddress('orbital agent os')).toBe('https://www.google.com/search?q=orbital%20agent%20os');
    expect(resolveAddress('what is a.b c')).toBe('https://www.google.com/search?q=what%20is%20a.b%20c');
  });

  it('refuses non-web schemes and empty input', () => {
    expect(resolveAddress('file:///etc/passwd')).toBeNull();
    expect(resolveAddress('javascript://x')).toBeNull();
    expect(resolveAddress('')).toBeNull();
    expect(resolveAddress('   ')).toBeNull();
  });
});

describe('displayAddress', () => {
  it('hides about:blank and nothing', () => {
    expect(displayAddress('about:blank')).toBe('');
    expect(displayAddress(undefined)).toBe('');
    expect(displayAddress('')).toBe('');
    expect(displayAddress('https://x.y/')).toBe('https://x.y/');
  });
});
