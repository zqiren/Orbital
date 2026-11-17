// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, expect, it } from 'vitest';
import {
  connectGroups, createBody, draftReady, fieldRows, initDraft, kindsReady, setField, setParam, toggleKind,
  markedField, pickGroups, pickPayload, pickState, togglePick, verifyMessageKey, websiteHost,
} from './formModel';
import { CALENDAR, EMAIL, FEISHU, GITHUB, TELEGRAM, WEBSITE } from './testFixtures';
import type { Catalog, CatalogType } from './types';

describe('initDraft', () => {
  it('switches on the default kinds with their param defaults', () => {
    expect(initDraft(GITHUB).kinds).toEqual({ assigned: {} });
    expect(initDraft(CALENDAR).kinds).toEqual({ before: { minutes: 30 } });
    expect(initDraft(EMAIL).kinds).toEqual({ new_mail: { folder: 'INBOX', unread_only: true } });
  });
  it('fills field defaults (email port)', () => {
    expect(initDraft(EMAIL).config).toEqual({ port: '993' });
  });
});

describe('kinds', () => {
  it('never turns the last kind off', () => {
    const k = toggleKind(GITHUB, { assigned: {} }, 'assigned');
    expect(k).toEqual({ assigned: {} });
  });
  it('turns a kind on with its defaults and off again', () => {
    let k = toggleKind(GITHUB, { assigned: {} }, 'mention');
    expect(k).toEqual({ assigned: {}, mention: {} });
    k = toggleKind(GITHUB, k, 'assigned');
    expect(k).toEqual({ mention: {} });
  });
  it('a kind with a required param is not ready until it is set', () => {
    let k = toggleKind(GITHUB, { assigned: {} }, 'label');
    expect(kindsReady(GITHUB, k)).toBe(false);
    k = setParam(GITHUB, k, 'label', 'repo', 'acme/web');
    expect(kindsReady(GITHUB, k)).toBe(false);
    k = setParam(GITHUB, k, 'label', 'label', 'bug');
    expect(kindsReady(GITHUB, k)).toBe(true);
  });
  it('changing a param clears the params that depend on it', () => {
    let k = setParam(GITHUB, { label: { repo: 'a/b', label: 'bug' } }, 'label', 'repo', 'c/d');
    expect(k).toEqual({ label: { repo: 'c/d' } });
    k = setParam(GITHUB, k, 'label', 'repo', 'c/d');
    expect(k).toEqual({ label: { repo: 'c/d' } });
  });
});

describe('fields', () => {
  it('pairs side-by-side fields', () => {
    const rows = fieldRows(EMAIL.connect.fields!);
    expect(rows.map((r) => (Array.isArray(r) ? r.map((f) => f.key) : r.key))).toEqual(['acct', 'pass', ['server', 'port']]);
  });
  it('an e-mail address of a known provider fills the server until the user types one', () => {
    let d = setField(EMAIL, initDraft(EMAIL), 'acct', 'mira@gmail.com');
    expect(d.config.server).toBe('imap.gmail.com');
    d = setField(EMAIL, d, 'acct', 'mira@qq.com');
    expect(d.config.server).toBe('imap.qq.com');
    d = setField(EMAIL, d, 'server', 'mail.acme.com');
    d = setField(EMAIL, d, 'acct', 'mira@gmail.com');
    expect(d.config.server).toBe('mail.acme.com');
  });
  it('secrets go to secrets, the rest to config', () => {
    const d = setField(TELEGRAM, initDraft(TELEGRAM), 'token', 'abc');
    expect(d.secrets).toEqual({ token: 'abc' });
    expect(d.config).toEqual({});
  });
});

describe('readiness', () => {
  it('a CLI type needs an account', () => {
    const d = initDraft(GITHUB);
    expect(draftReady(GITHUB, d)).toBe(false);
    expect(draftReady(GITHUB, { ...d, account: 'mira' })).toBe(true);
  });
  it('a fields type needs every field that is not optional', () => {
    let d = initDraft(EMAIL);
    for (const [k, v] of [['acct', 'a@x.com'], ['pass', 'p'], ['server', 'imap.x.com']]) {
      expect(draftReady(EMAIL, d)).toBe(false);
      d = setField(EMAIL, d, k, v);
    }
    expect(draftReady(EMAIL, d)).toBe(true);
    expect(draftReady(EMAIL, setField(EMAIL, d, 'pass', '  '))).toBe(false);
  });
  it('a website needs an address with a host; the text is optional', () => {
    expect(draftReady(WEBSITE, setField(WEBSITE, initDraft(WEBSITE), 'url', 'https://'))).toBe(false);
    expect(draftReady(WEBSITE, setField(WEBSITE, initDraft(WEBSITE), 'url', 'shop.acme.com'))).toBe(true);
  });
  it('feishu by QR is ready once the scan is done; by fields once filled', () => {
    const d = initDraft(FEISHU);
    expect(draftReady(FEISHU, d)).toBe(false);
    expect(draftReady(FEISHU, { ...d, qrId: 'q', qrState: 'done' })).toBe(true);
    const m = setField(FEISHU, setField(FEISHU, { ...d, manual: true }, 'appid', 'cli_1'), 'appsec', 's');
    expect(draftReady(FEISHU, m)).toBe(true);
  });
  it('calendar needs nothing filled once its list is read', () => {
    expect(draftReady(CALENDAR, { ...initDraft(CALENDAR), pick: { all: ['a'], on: ['a'] } })).toBe(true);
  });
});

describe('createBody', () => {
  it('sends the kinds, the account and only the filled values', () => {
    const d = { ...initDraft(GITHUB), account: 'mira' };
    expect(createBody(GITHUB, d)).toEqual({
      type: 'github', name: '', kinds: { assigned: {} }, account: 'mira', config: {}, secrets: {}, qr_id: null, instruction: '',
    });
  });
  it('feishu by QR sends the qr id and no secrets', () => {
    const b = createBody(FEISHU, { ...initDraft(FEISHU), qrId: 'qr_1', qrState: 'done', secrets: { appsec: 'x' } });
    expect(b.qr_id).toBe('qr_1');
    expect(b.secrets).toEqual({});
  });
  it('a website address without a scheme gets https', () => {
    const b = createBody(WEBSITE, setField(WEBSITE, initDraft(WEBSITE), 'url', 'shop.acme.com'));
    expect(b.config).toEqual({ url: 'https://shop.acme.com' });
  });
});

describe('connectGroups', () => {
  const t = (type: string, family: CatalogType['family'], o: Partial<CatalogType> = {}): CatalogType =>
    ({ ...TELEGRAM, type, family, ...o });
  const cat: Catalog = {
    platform: 'win32',
    types: [
      t('telegram', 'bot'), t('feishu', 'bot'), t('slack', 'bot', { available: false, unavailable_reason: 'not_built' }),
      t('github', 'account'), t('calendar', 'account', { available: false, unavailable_reason: 'platform' }),
      t('email', 'account'), t('website', 'check'), t('command', 'external'),
    ],
  };
  it('orders by family and language and hides every type this daemon cannot run (unbuilt or other platform)', () => {
    const zh = connectGroups(cat, 'zh');
    expect(zh.account.map((x) => x.type)).toEqual(['github', 'email']);
    expect(zh.bot.map((x) => x.type)).toEqual(['feishu', 'telegram']);
    expect(zh.check.map((x) => x.type)).toEqual(['website']);
    expect(connectGroups(cat, 'en').bot.map((x) => x.type)).toEqual(['telegram', 'feishu']);
  });
});

describe('verify errors', () => {
  it('names the message for the reason', () => {
    expect(verifyMessageKey('telegram', 'rejected', true)).toBe('sources.verify.token');
    expect(verifyMessageKey('email', 'rejected', true)).toBe('sources.verify.login');
    expect(verifyMessageKey('github', 'not_found', false)).toBe('sources.verify.notFound');
    expect(verifyMessageKey('github', 'whatever', false)).toBe('sources.verify.error');
  });
});

describe('websiteHost', () => {
  it('reads the host from what was typed', () => {
    expect(websiteHost('https://shop.acme.com/path')).toBe('shop.acme.com');
    expect(websiteHost('shop.acme.com')).toBe('shop.acme.com');
    expect(websiteHost('https://')).toBe('');
  });
});

describe('markedField', () => {
  it('names the form field, the account or the kind param a refusal points at', () => {
    expect(markedField(EMAIL, 'pass')).toBe('pass');
    expect(markedField(EMAIL, 'folder')).toBe('folder');
    expect(markedField(GITHUB, 'account')).toBe('account');
    expect(markedField(GITHUB, 'repo')).toBe('repo');
    expect(markedField(GITHUB, 'nothing')).toBeNull();
    expect(markedField(TELEGRAM, null)).toBeNull();
  });
});

describe('the calendar list', () => {
  const ITEMS = [
    { value: 'ic-home', label: 'Home', group: 'iCloud' },
    { value: 'ex-team', label: 'Team', group: 'Exchange' },
    { value: 'ic-kids', label: 'Kids', group: 'iCloud' },
  ];
  it('is being read when the form opens: connect waits', () => {
    const d = initDraft(CALENDAR);
    expect(d.pick).toBe('loading');
    expect(draftReady(CALENDAR, d)).toBe(false);
    expect(initDraft(GITHUB).pick).toBeUndefined();
  });
  it('every calendar ticked sends null (no key: calendars added later are read too)', () => {
    const d = { ...initDraft(CALENDAR), pick: pickState(null, ITEMS) };
    expect(d.pick.on).toEqual(['ic-home', 'ex-team', 'ic-kids']);
    expect(draftReady(CALENDAR, d)).toBe(true);
    const body = createBody(CALENDAR, d);
    expect('calendars' in body.config).toBe(true);
    expect(body.config.calendars).toBeNull();
  });
  it('unticking some sends the exact ids; none ticked cannot connect', () => {
    let p = togglePick(pickState(null, ITEMS), 'ex-team');
    expect(createBody(CALENDAR, { ...initDraft(CALENDAR), pick: p }).config.calendars).toEqual(['ic-home', 'ic-kids']);
    p = togglePick(togglePick(p, 'ic-home'), 'ic-kids');
    expect(p.on).toEqual([]);
    expect(draftReady(CALENDAR, { ...initDraft(CALENDAR), pick: p })).toBe(false);
    // ticking back keeps the listed order
    p = togglePick(togglePick(p, 'ic-kids'), 'ic-home');
    expect(pickPayload(p)).toEqual(['ic-home', 'ic-kids']);
    expect(pickPayload(togglePick(p, 'ex-team'))).toBeNull();
  });
  it('an unreadable list connects every calendar', () => {
    const d = { ...initDraft(CALENDAR), pick: null };
    expect(draftReady(CALENDAR, d)).toBe(true);
    expect(createBody(CALENDAR, d).config.calendars).toBeNull();
  });
  it('a stored list ticks its ids; a deleted calendar is not shown', () => {
    expect(pickState(['ex-team', 'gone'], ITEMS).on).toEqual(['ex-team']);
    expect(pickState(['gone'], ITEMS).on).toEqual([]);
    expect(pickState(undefined, ITEMS).on).toHaveLength(3);
  });
  it('groups by account in the order listed', () => {
    expect(pickGroups(ITEMS).map((g) => [g.group, g.items.map((i) => i.value)])).toEqual([
      ['iCloud', ['ic-home', 'ic-kids']], ['Exchange', ['ex-team']],
    ]);
  });
});
