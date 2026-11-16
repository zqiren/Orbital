// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it } from 'vitest';
import { translate } from '../../i18n/useT';
import type { StringKey } from '../../i18n/strings';
import { accountLine, cjkSpacing, failureText, placeName } from './sourcesText';
import type { SourceFailure } from './types';

const zh = (k: StringKey, v?: Record<string, string | number>) => translate('zh', k, v);
const en = (k: StringKey, v?: Record<string, string | number>) => translate('en', k, v);
const f = (code: string, params: SourceFailure['params'] = {}): SourceFailure => ({ code, params, at: null, actions: ['remove'] });

describe('failureText', () => {
  it('fills the params the daemon sent', () => {
    expect(failureText(zh, f('folder_not_found', { folder: '其他文件夹' }), 'email')).toBe('找不到文件夹 其他文件夹');
    expect(failureText(en, f('unreachable', { host: 'imap.qq.com' }), 'email')).toBe("Can't reach imap.qq.com");
  });
  it('never shows a bare {placeholder} when a param is missing', () => {
    expect(failureText(zh, f('unreachable'), 'slack')).toBe('连不上 Slack');
    expect(failureText(en, f('repo_not_found'), 'github')).not.toMatch(/[{}]/);
  });
  it('the bot codes have their own words', () => {
    expect(failureText(zh, f('bot_in_use'), 'telegram')).toBe('另一个程序正在使用这个机器人');
    expect(failureText(en, f('webhook_set'), 'telegram')).toMatch(/webhook/i);
  });
  it('an unknown code reads as the generic failure', () => {
    expect(failureText(zh, f('brand_new'), 'github')).toBe('检查出错');
  });
});

describe('placeName', () => {
  it('a group or channel without a name gets the generic word', () => {
    expect(placeName(zh, { kind: 'group', name: null })).toBe('群聊');
    expect(placeName(zh, { kind: 'channel' })).toBe('频道');
    expect(placeName(en, { kind: 'group', name: '产品讨论群' })).toBe('产品讨论群');
    expect(placeName(zh, { kind: 'dm' })).toBe('私聊');
  });
});

describe('accountLine', () => {
  it('a gitlab.com or github.com account is its login over its host', () => {
    expect(accountLine('mira', 'gitlab.com')).toEqual({ name: 'mira', host: 'gitlab.com' });
  });
  it('a self-managed GitLab account reads user@host once', () => {
    expect(accountLine('mira@git.acme.dev', 'git.acme.dev')).toEqual({ name: 'mira@git.acme.dev', host: null });
  });
});

describe('cjkSpacing', () => {
  it('drops a space between two Chinese characters, keeps it next to Latin', () => {
    expect(cjkSpacing('钉钉 不认这个令牌', '钉钉')).toBe('钉钉不认这个令牌');
    expect(cjkSpacing('Telegram 不认这个令牌', 'Telegram')).toBe('Telegram 不认这个令牌');
    expect(cjkSpacing('Slack rejected this token', 'Slack')).toBe('Slack rejected this token');
    expect(cjkSpacing('找不到文件夹 其他文件夹', '邮箱')).toBe('找不到文件夹 其他文件夹');
  });
  it('failure and verify lines with a Chinese service name read without the gap', () => {
    expect(failureText(zh, f('token_rejected', { field: 'csec' }), 'dingtalk')).toBe('钉钉令牌已失效');
  });
});
