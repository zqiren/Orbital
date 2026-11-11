// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * ImportProjectsStep — the project checklist from the user's agent history
 * (spec 138): what is checked by default, what "Import N projects" counts,
 * group folders, and the single POST /onboarding/import.
 */

import { render, screen, cleanup, fireEvent, waitFor, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { OnboardingScan } from '../hooks/useOnboardingScan';
import { useState } from 'react';
import { NO_CHOICES, type ImportableResponse, type ImportChoices } from '../utils/onboarding';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock }));
vi.mock('./FolderBrowserPanel', () => ({
  default: ({ onSelect }: { onSelect: (p: string) => void }) => (
    <button type="button" data-testid="fake-browser" onClick={() => onSelect('/Users/x/new-decks')}>
      pick
    </button>
  ),
}));

import ImportProjectsStep from './ImportProjectsStep';

const DAY = 86_400_000;
const iso = (msAgo: number) => new Date(Date.now() - msAgo).toISOString();

const DATA: ImportableResponse = {
  status: 'ready',
  candidates: [
    { source: 'claude-code', name: 'orbital-test', path: '/Users/x/orbital-test', session_count: 223, last_activity: iso(DAY), already_imported: false, kind: 'folder', minor: false, is_git: true, by_agent: { 'claude-code': 182, codex: 41 } },
    { source: 'codex', name: 'website', path: '/Users/x/website', session_count: 9, last_activity: iso(3 * DAY), already_imported: true, kind: 'folder', minor: false, is_git: true, by_agent: { codex: 9 } },
    { source: 'claude-code', name: 'scratch-a', path: '/Users/x/scratch-a', session_count: 2, last_activity: iso(5 * DAY), already_imported: false, kind: 'folder', minor: true, is_git: false, by_agent: { 'claude-code': 2 } },
    { source: 'codex', name: 'scratch-b', path: '/Users/x/scratch-b', session_count: 1, last_activity: iso(6 * DAY), already_imported: false, kind: 'folder', minor: true, is_git: false, by_agent: { codex: 1 } },
  ],
  groups: [
    {
      id: 'grp_decks', session_count: 13, last_activity: iso(2 * DAY), by_agent: { 'claude-code': 9, codex: 4 },
      reasons: [
        { kind: 'same_skill', value: 'pptx', count: 6 },
        { kind: 'same_files', value: 'Q3-review.pptx', count: 4 },
        { kind: 'same_request', value: "write this week's report from", count: 5 },
        { kind: 'same_folder', value: '/Users/x/Desktop/decks', count: 3 },
      ],
      suggested_folder: '/Users/x/Desktop/decks',
      members: [{ agent: 'codex', native_id: 'a' }],
    },
    {
      id: 'grp_loose', session_count: 3, last_activity: iso(4 * DAY), by_agent: { codex: 3 },
      reasons: [{ kind: 'same_request', value: 'summarize the meeting notes', count: 3 }],
      suggested_folder: null,
      members: [{ agent: 'codex', native_id: 'b' }],
    },
  ],
};

function scanOf(over: Partial<OnboardingScan> = {}): OnboardingScan {
  return { history: null, importable: DATA, error: false, settled: true, retry: vi.fn(), ...over };
}

const telemetry: Array<{ event: string; fields: Record<string, unknown> }> = [];
let importResponse: () => Promise<unknown>;

beforeEach(() => {
  telemetry.length = 0;
  apiMock.mockReset();
  importResponse = async () => ({
    created: [
      { project_id: 'p_orb', name: 'orbital-test', workspace: '/Users/x/orbital-test', from: 'folder' },
      { project_id: 'p_decks', name: 'decks', workspace: '/Users/x/Desktop/decks', from: 'group' },
    ],
    skipped: [],
  });
  apiMock.mockImplementation(async (path: string, opts?: { method?: string; body?: string }) => {
    if (path === '/api/v2/telemetry/event') {
      telemetry.push(JSON.parse(opts?.body ?? '{}'));
      return undefined;
    }
    if (path === '/api/v2/onboarding/import') return importResponse();
    return {};
  });
});
afterEach(() => cleanup());

const submit = () => screen.getByTestId('import-submit') as HTMLButtonElement;
const check = (testId: string) => screen.getByTestId(testId) as HTMLInputElement;

describe('ImportProjectsStep — what is offered, and checked', () => {
  it('checks the worked-in folders, shows imported ones as imported, and collapses the minor ones', () => {
    render(<ImportProjectsStep scan={scanOf()} onImported={vi.fn()} />);
    expect(check('import-check-/Users/x/orbital-test').checked).toBe(true);
    expect(screen.queryByTestId('import-check-/Users/x/website')).toBeNull();
    expect(screen.getByTestId('import-imported-/Users/x/website').textContent).toBe('Already imported');
    // per-agent counts, biggest first
    expect(screen.getByTestId('import-candidate-/Users/x/orbital-test').textContent).toContain('Claude Code 182 · Codex 41');

    const toggle = screen.getByTestId('import-minor-toggle');
    expect(toggle.textContent).toBe('2 more folders with one or two conversations');
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect(screen.queryByTestId('import-check-/Users/x/scratch-a')).toBeNull();
    fireEvent.click(toggle);
    expect(check('import-check-/Users/x/scratch-a').checked).toBe(false);
    expect(check('import-check-/Users/x/scratch-b').checked).toBe(false);
  });

  it('renders each group with its folder name, meta line and reason chips', () => {
    render(<ImportProjectsStep scan={scanOf()} onImported={vi.fn()} />);
    const decks = screen.getByTestId('import-group-grp_decks');
    expect(screen.getByTestId('import-group-title-grp_decks').textContent).toBe('decks');
    expect(decks.textContent).toContain('13 conversations · Claude Code, Codex · last one 2d ago');
    expect(decks.textContent).toContain('used the pptx skill');
    expect(decks.textContent).toContain('wrote the same files: Q3-review.pptx');
    expect(decks.textContent).toContain('started with the same request ×5: "write this week\'s report from…"');
    expect(decks.textContent).toContain('ran in the same folder: /Users/x/Desktop/decks');
    expect(check('import-group-check-grp_decks').checked).toBe(true);

    const loose = screen.getByTestId('import-group-title-grp_loose');
    expect(loose.textContent).toBe('Needs a folder');
    expect(loose.className).toContain('warning');
    expect(screen.getByTestId('import-group-grp_loose').textContent).toContain('Choose a folder before importing');
    expect(screen.getByTestId('import-group-choose-grp_loose').textContent).toBe('Choose folder');
    expect(screen.getByTestId('import-group-choose-grp_decks').textContent).toBe('Change');
  });
});

describe('ImportProjectsStep — the Import count', () => {
  it('counts checked folders plus checked groups that have a folder', () => {
    render(<ImportProjectsStep scan={scanOf()} onImported={vi.fn()} />);
    // orbital-test + grp_decks; grp_loose has no folder; minor ones unchecked.
    expect(submit().textContent).toBe('Import 2 projects');

    fireEvent.click(check('import-check-/Users/x/orbital-test'));
    expect(submit().textContent).toBe('Import 1 project');

    fireEvent.click(screen.getByTestId('import-minor-toggle'));
    fireEvent.click(check('import-check-/Users/x/scratch-a'));
    expect(submit().textContent).toBe('Import 2 projects');

    fireEvent.click(check('import-group-check-grp_decks'));
    fireEvent.click(check('import-check-/Users/x/scratch-a'));
    expect(submit().textContent).toBe('Import 0 projects');
    expect(submit().disabled).toBe(true);
  });

  it('a group counts once a folder is chosen through the folder browser', () => {
    render(<ImportProjectsStep scan={scanOf()} onImported={vi.fn()} />);
    fireEvent.click(screen.getByTestId('import-group-choose-grp_loose'));
    fireEvent.click(screen.getByTestId('fake-browser'));
    expect(screen.getByTestId('import-group-title-grp_loose').textContent).toBe('new-decks');
    expect(screen.queryByTestId('import-group-browser-grp_loose')).toBeNull();
    expect(submit().textContent).toBe('Import 3 projects');
  });
});

describe('ImportProjectsStep — cost sign (spec 160 round 2)', () => {
  it('counts the checked rows beside Import, with the key or the agent sentence', () => {
    const { unmount } = render(<ImportProjectsStep scan={scanOf()} onImported={vi.fn()}
      scanRunner={{ key: true, agent: null }} />);
    const sign = screen.getByTestId('token-cost-sign');
    expect(sign.getAttribute('data-kind')).toBe('importKey');
    expect(sign.textContent).toBe(submit().textContent?.match(/\d+/)?.[0]);
    expect(sign.nextElementSibling).toBe(submit());
    unmount();
    render(<ImportProjectsStep scan={scanOf()} onImported={vi.fn()}
      scanRunner={{ key: false, agent: 'Claude Code' }} />);
    expect(screen.getByTestId('token-cost-sign').getAttribute('data-kind')).toBe('importAgent');
  });

  it('round 2: switched off, the import request carries import_scan:false', async () => {
    importResponse = async () => ({ created: [], skipped: [] });
    render(<ImportProjectsStep scan={scanOf()} onImported={vi.fn()}
      scanRunner={{ key: true, agent: null }} />);
    fireEvent.click(screen.getByTestId('token-cost-sign'));
    fireEvent.click(screen.getByTestId('token-cost-switch'));
    await act(async () => { fireEvent.click(submit()); });
    const post = apiMock.mock.calls.find(([p]) => p === '/api/v2/onboarding/import');
    expect(JSON.parse(post?.[1]?.body).import_scan).toBe(false);
  });

  it('round 3: the wizard defers the read to its key step (defer_scan); the dialog does not', async () => {
    for (const defer of [true, false]) {
      apiMock.mockClear();
      importResponse = async () => ({ created: [], skipped: [] });
      const { unmount } = render(<ImportProjectsStep scan={scanOf()} onImported={vi.fn()} deferScan={defer} />);
      await act(async () => { fireEvent.click(submit()); });
      const post = apiMock.mock.calls.find(([p]) => p === '/api/v2/onboarding/import');
      expect(JSON.parse(post?.[1]?.body).defer_scan).toBe(defer ? true : undefined);
      unmount();
    }
  });

  it('no runner (the wizard) shows no sign', () => {
    render(<ImportProjectsStep scan={scanOf()} onImported={vi.fn()} />);
    expect(screen.queryByTestId('token-cost-sign')).toBeNull();
  });
});

describe('ImportProjectsStep — Import', () => {
  it('POSTs one import with the checked folders and groups, reports unchecks, and hands back the landing project', async () => {
    importResponse = async () => ({
      created: [
        { project_id: 'p_loose', name: 'new-decks', workspace: '/Users/x/new-decks', from: 'group' },
        { project_id: 'p_decks', name: 'decks', workspace: '/Users/x/Desktop/decks', from: 'group' },
      ],
      skipped: [],
    });
    const onImported = vi.fn();
    render(<ImportProjectsStep scan={scanOf()} onImported={onImported} />);
    fireEvent.click(screen.getByTestId('import-group-choose-grp_loose'));
    fireEvent.click(screen.getByTestId('fake-browser'));
    fireEvent.click(check('import-check-/Users/x/orbital-test')); // uncheck one

    await act(async () => { fireEvent.click(submit()); });

    const post = apiMock.mock.calls.find(([p]) => p === '/api/v2/onboarding/import');
    expect(post?.[1]?.method).toBe('POST');
    expect(JSON.parse(post?.[1]?.body)).toEqual({
      folders: [],
      groups: [
        { id: 'grp_decks', workspace: '/Users/x/Desktop/decks' },
        { id: 'grp_loose', workspace: '/Users/x/new-decks' },
      ],
      // Spec 160: the UI language the background read replies in.
      lang: 'en',
    });
    expect(telemetry).toContainEqual({ event: 'projects_unchecked', fields: { n: 1 } });
    // grp_decks (2 days ago) is more recent than grp_loose (4 days ago).
    await waitFor(() => expect(onImported).toHaveBeenCalledTimes(1));
    expect(onImported.mock.calls[0][1]).toBe('p_decks');
  });

  it('a failed import says so and stays put', async () => {
    importResponse = async () => { throw new Error('boom'); };
    const onImported = vi.fn();
    render(<ImportProjectsStep scan={scanOf()} onImported={onImported} />);
    await act(async () => { fireEvent.click(submit()); });
    expect(screen.getByTestId('import-failed').textContent).toBe("Couldn't import. Try again.");
    expect(onImported).not.toHaveBeenCalled();
    expect(submit().disabled).toBe(false);
  });

  it('offers Back and Skip when given them', () => {
    const onBack = vi.fn();
    const onSkip = vi.fn();
    render(<ImportProjectsStep scan={scanOf()} onImported={vi.fn()} onBack={onBack} onSkip={onSkip} />);
    fireEvent.click(screen.getByTestId('import-back'));
    fireEvent.click(screen.getByTestId('import-skip'));
    expect(onBack).toHaveBeenCalledTimes(1);
    expect(onSkip).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId('import-skip').textContent).toBe('Skip');
  });
});

describe('ImportProjectsStep — while the scan runs', () => {
  it('shows candidates at once and says groups are still coming', () => {
    render(
      <ImportProjectsStep
        scan={scanOf({ settled: false, importable: { status: 'scanning', candidates: DATA.candidates, groups: [] } })}
        onImported={vi.fn()}
      />,
    );
    expect(screen.getByTestId('import-candidate-/Users/x/orbital-test')).toBeTruthy();
    expect(screen.getByTestId('import-groups-scanning').textContent).toBe('Still reading your conversations…');
  });

  it('an unreadable history offers Try again', () => {
    const retry = vi.fn();
    render(<ImportProjectsStep scan={scanOf({ importable: null, error: true, retry })} onImported={vi.fn()} />);
    fireEvent.click(screen.getByTestId('import-retry'));
    expect(retry).toHaveBeenCalledTimes(1);
  });
});

describe('ImportProjectsStep — rows that arrive after the user started choosing (F2, 2026-11-11)', () => {
  // The scan's first answer: one worked-in folder, no groups yet.
  const FIRST: ImportableResponse = { status: 'scanning', candidates: [DATA.candidates[0]], groups: [] };
  // The history scan lands: a new folder row and a group with a folder.
  const NEW_FOLDER = { ...DATA.candidates[0], name: 'later', path: '/Users/x/later' };
  const READY: ImportableResponse = {
    status: 'ready',
    candidates: [DATA.candidates[0], NEW_FOLDER],
    groups: [DATA.groups![0]],
  };

  it('after an uncheck, rows that show up later arrive unchecked, and only what the user kept is imported', async () => {
    const { rerender } = render(<ImportProjectsStep scan={scanOf({ settled: false, importable: FIRST })} onImported={vi.fn()} />);
    fireEvent.click(check('import-check-/Users/x/orbital-test'));
    expect(check('import-check-/Users/x/orbital-test').checked).toBe(false);

    rerender(<ImportProjectsStep scan={scanOf({ importable: READY })} onImported={vi.fn()} />);
    // The unchecked row stays unchecked; the two that were not on screen
    // when the user chose come in unchecked too (the "5th row" re-checked).
    expect(check('import-check-/Users/x/orbital-test').checked).toBe(false);
    expect(check('import-check-/Users/x/later').checked).toBe(false);
    expect(check('import-group-check-grp_decks').checked).toBe(false);
    expect(submit().disabled).toBe(true);

    // The user can still pick a late row on purpose.
    fireEvent.click(check('import-check-/Users/x/later'));
    await act(async () => { fireEvent.click(submit()); });
    const body = JSON.parse(String(apiMock.mock.calls.find(([p]) => p === '/api/v2/onboarding/import')![1].body));
    expect(body.folders).toEqual(['/Users/x/later']);
    expect(body.groups).toEqual([]);
  });

  it('before the user touches anything, later rows arrive checked (everything offered checked)', () => {
    const { rerender } = render(<ImportProjectsStep scan={scanOf({ settled: false, importable: FIRST })} onImported={vi.fn()} />);
    rerender(<ImportProjectsStep scan={scanOf({ importable: READY })} onImported={vi.fn()} />);
    expect(check('import-check-/Users/x/orbital-test').checked).toBe(true);
    expect(check('import-check-/Users/x/later').checked).toBe(true);
    expect(check('import-group-check-grp_decks').checked).toBe(true);
  });
});

describe('ImportProjectsStep — the feed replaces the list after the user chose (1111 pass 2, N2)', () => {
  const folder = (name: string, over: Partial<ImportableResponse['candidates'][number]> = {}) => ({
    ...DATA.candidates[0], name, path: `/Users/x/${name}`, by_agent: { 'claude-code': 4 }, session_count: 4, ...over,
  });
  // The folder scan's instant first answer: a few rows, some of them minor.
  const FIRST: ImportableResponse = {
    status: 'scanning',
    candidates: [folder('a'), folder('b'), folder('c'), folder('tiny', { minor: true, session_count: 1, is_git: false })],
    groups: [],
  };
  // The analysed list replaces it: more rows, a row under a new path, the
  // minor row now a worked-in folder, groups.
  const REPLACED: ImportableResponse = {
    status: 'scanning',
    candidates: [
      folder('a'), folder('b-renamed', { path: '/Users/x/b/' }), folder('c'), folder('tiny'),
      ...Array.from({ length: 20 }, (_, i) => folder(`new-${i}`)),
    ],
    groups: [DATA.groups![0]],
  };
  const READY: ImportableResponse = { ...REPLACED, status: 'ready', candidates: REPLACED.candidates.slice(0, 10), groups: [{ ...DATA.groups![0], id: 'grp_other' }] };

  const checkedRows = () =>
    screen.queryAllByRole('checkbox').filter((el) => (el as HTMLInputElement).checked).map((el) => el.getAttribute('data-testid'));

  it('unchecking the first rows, then a replacement list and the scan finishing, checks nothing', () => {
    const { rerender } = render(<ImportProjectsStep scan={scanOf({ settled: false, importable: FIRST })} onImported={vi.fn()} />);
    for (const n of ['a', 'b', 'c']) fireEvent.click(check(`import-check-/Users/x/${n}`));
    expect(submit().textContent).toBe('Import 0 projects');

    rerender(<ImportProjectsStep scan={scanOf({ settled: false, importable: REPLACED })} onImported={vi.fn()} />);
    expect(checkedRows()).toEqual([]);
    expect(submit().textContent).toBe('Import 0 projects');

    // The scan flips from running to done (settled, status ready).
    rerender(<ImportProjectsStep scan={scanOf({ settled: true, importable: READY })} onImported={vi.fn()} />);
    expect(checkedRows()).toEqual([]);
    expect(submit().textContent).toBe('Import 0 projects');
    expect(submit().disabled).toBe(true);
  });

  it('a row the user left checked stays checked through the replacement', () => {
    const { rerender } = render(<ImportProjectsStep scan={scanOf({ settled: false, importable: FIRST })} onImported={vi.fn()} />);
    fireEvent.click(check('import-check-/Users/x/a'));
    rerender(<ImportProjectsStep scan={scanOf({ settled: true, importable: READY })} onImported={vi.fn()} />);
    expect(checkedRows()).toEqual(['import-check-/Users/x/c']);
    expect(submit().textContent).toBe('Import 1 project');
  });

  it('the wizard-held choices survive the step remounting (Back and forward) mid-scan', () => {
    function Harness({ data, shown }: { data: ImportableResponse; shown: boolean }) {
      const choices = useState<ImportChoices>(NO_CHOICES);
      return shown ? <ImportProjectsStep scan={scanOf({ settled: false, importable: data })} onImported={vi.fn()} choices={choices} /> : null;
    }
    const { rerender } = render(<Harness data={FIRST} shown />);
    for (const n of ['a', 'b', 'c']) fireEvent.click(check(`import-check-/Users/x/${n}`));
    rerender(<Harness data={FIRST} shown={false} />);
    rerender(<Harness data={REPLACED} shown />);
    expect(checkedRows()).toEqual([]);
    expect(submit().textContent).toBe('Import 0 projects');
  });
});
