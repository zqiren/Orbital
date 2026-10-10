// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { pickLoginUrl } from './loginUrl';

describe('pickLoginUrl', () => {
  it('returns a plain sign-in URL', () => {
    expect(pickLoginUrl('visit: https://claude.com/cai/oauth/authorize?code=true'))
      .toBe('https://claude.com/cai/oauth/authorize?code=true');
  });

  it.each([
    'Starting local login server on http://localhost:1455.',
    'callback at http://127.0.0.1:8080/cb',
    'http://[::1]:1455/auth',
    'http://LOCALHOST/',
  ])('skips the loopback callback server: %s', line => {
    expect(pickLoginUrl(line)).toBeNull();
  });

  it('takes the first non-loopback URL on a line with both', () => {
    expect(pickLoginUrl('server http://localhost:1455 then https://auth.openai.com/x'))
      .toBe('https://auth.openai.com/x');
  });

  it('drops sentence punctuation after the URL', () => {
    expect(pickLoginUrl('Open (https://auth.openai.com/oauth?a=1).')).toBe('https://auth.openai.com/oauth?a=1');
  });

  it('keeps a host that only starts with "localhost"', () => {
    expect(pickLoginUrl('https://localhost.example.com/login')).toBe('https://localhost.example.com/login');
  });

  it('returns null without a URL', () => {
    expect(pickLoginUrl('Waiting for authentication...')).toBeNull();
  });
});
