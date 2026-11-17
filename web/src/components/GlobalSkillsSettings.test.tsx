// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Global Settings → Skills (skills contract v3): one list of Orbital's own
 * skills and the home-folder skills, each with its source mark, its folder on
 * hover and an on/off switch; delete (after one confirm) only on Orbital's
 * own; a same-name skill that wins greys the row with a hover sign; an empty
 * list is one line + the add button.
 */

import { render, screen, waitFor, cleanup, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { GlobalSkillGroup, GlobalSkill } from '../types';

const { apiMock, MockApiError } = vi.hoisted(() => {
  class MockApiError extends Error {
    constructor(
      public status: number,
      public detail: string,
    ) {
      super(detail);
      this.name = 'ApiError';
    }
  }
  return { apiMock: vi.fn(), MockApiError };
});
vi.mock('../config', () => ({ api: apiMock, ApiError: MockApiError }));

import GlobalSkillsSettings, { skillFolder, skillId, skillOwners } from './GlobalSkillsSettings';

afterEach(() => cleanup());

function skill(source: GlobalSkill['source'], dir: string, overrides: Partial<GlobalSkill> = {}): GlobalSkill {
  const root = source === 'global' ? '/data/skills' : `/u/.${source}/skills`;
  return {
    name: dir,
    description: `About ${dir}`,
    source,
    path: `${root}/${dir}/SKILL.md`,
    dir_name: dir,
    editable: source === 'global',
    id: `${source}:${dir.toLowerCase()}`,
    enabled: true,
    hidden_by: null,
    ...overrides,
  };
}

function groups(rows: Partial<Record<GlobalSkill['source'], GlobalSkill[]>>): GlobalSkillGroup[] {
  return (['global', 'agents', 'claude', 'codex'] as const).map((source) => ({
    source,
    path: source === 'global' ? '/data/skills' : `/u/.${source}/skills`,
    editable: source === 'global',
    exists: (rows[source] ?? []).length > 0,
    skills: rows[source] ?? [],
  }));
}

let current: GlobalSkillGroup[] = [];

function calls(method: string) {
  return apiMock.mock.calls.filter(([, o]) => o?.method === method);
}

beforeEach(() => {
  apiMock.mockReset();
  current = [];
  apiMock.mockImplementation(async (path: string, options?: RequestInit) => {
    if (path === '/api/v2/skills' && !options?.method) return { groups: current };
    if (path === '/api/v2/skills/enabled') return JSON.parse(String(options?.body));
    return {};
  });
});

describe('GlobalSkillsSettings — listing', () => {
  it('lists Orbital and home-folder skills in one list with source marks, folder on hover and a switch each', async () => {
    current = groups({
      global: [skill('global', 'pdf-tidy')],
      agents: [skill('agents', 'sheet-cleanup')],
      claude: [skill('claude', 'frontend-design')],
      codex: [skill('codex', 'release-notes')],
    });
    render(<GlobalSkillsSettings />);
    const list = await screen.findByTestId('global-skills-list');
    const ids = Array.from(list.querySelectorAll(':scope > [data-testid]')).map((e) =>
      e.getAttribute('data-testid'),
    );
    expect(ids).toEqual([
      'skill-global-pdf-tidy',
      'skill-agents-sheet-cleanup',
      'skill-claude-frontend-design',
      'skill-codex-release-notes',
    ]);
    const claude = screen.getByTestId('skill-claude-frontend-design');
    expect(claude.getAttribute('title')).toBe('/u/.claude/skills/frontend-design');
    expect(claude.querySelector('[data-agent="claude-code"]')).not.toBeNull();
    expect(screen.getByTestId('skill-codex-release-notes').querySelector('[data-agent="codex"]')).not.toBeNull();
    expect(screen.getByTestId('skill-global-pdf-tidy').querySelector('[data-agent="orbital"]')).not.toBeNull();
    for (const id of ids) {
      expect(screen.getByTestId(`${id}-switch`).getAttribute('aria-checked')).toBe('true');
    }
    // The path is only on hover, never printed.
    expect(list.textContent).not.toContain('/u/.claude');
  });

  it('offers delete only on Orbital\'s own skills', async () => {
    current = groups({ global: [skill('global', 'pdf-tidy')], claude: [skill('claude', 'frontend-design')] });
    render(<GlobalSkillsSettings />);
    expect(await screen.findByTestId('skill-global-pdf-tidy-delete')).toBeInTheDocument();
    expect(screen.queryByTestId('skill-claude-frontend-design-delete')).not.toBeInTheDocument();
  });

  it('a row another same-name skill wins over is greyed, with the reason behind a hover sign', async () => {
    current = groups({
      global: [skill('global', 'pdf')],
      claude: [skill('claude', 'pdf', { hidden_by: 'global:pdf' })],
    });
    render(<GlobalSkillsSettings />);
    const sign = await screen.findByTestId('skill-claude-pdf-why');
    expect(screen.queryByTestId('skill-global-pdf-why')).not.toBeInTheDocument();
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
    fireEvent.click(sign);
    expect(screen.getByRole('tooltip').textContent).toBe(
      "A skill with the same name takes precedence, so this one isn't used.",
    );
  });

  it('empty: one short line and the add button', async () => {
    current = groups({});
    render(<GlobalSkillsSettings />);
    expect((await screen.findByTestId('global-skills-empty')).textContent).toBe('No skills yet');
    expect(screen.getByTestId('global-skills-upload')).toBeInTheDocument();
    expect(screen.queryByTestId('global-skills-list')).not.toBeInTheDocument();
  });

  it('skillFolder strips SKILL.md for either separator; skillId falls back to source:name', () => {
    expect(skillFolder('/a/b/SKILL.md')).toBe('/a/b');
    expect(skillFolder('C:\\Users\\u\\.codex\\skills\\x\\SKILL.md')).toBe('C:\\Users\\u\\.codex\\skills\\x');
    expect(skillId(skill('claude', 'PDF', { id: undefined }))).toBe('claude:pdf');
  });
});

describe('GlobalSkillsSettings — source marks', () => {
  it('a shared ~/.agents skill linked from Claude Code shows the Claude Code logo, else a folder', () => {
    expect(skillOwners(skill('agents', 'x'))).toEqual([]);
    expect(skillOwners(skill('agents', 'x', { linked_from: ['claude'] }))).toEqual(['claude-code']);
    expect(skillOwners(skill('agents', 'x', { linked_from: ['claude', 'codex'] }))).toEqual(['claude-code', 'codex']);
    expect(skillOwners(skill('global', 'x'))).toEqual(['orbital']);
  });
});

describe('GlobalSkillsSettings — switch', () => {
  it('switching a home skill off PUTs its id and shows it off', async () => {
    current = groups({ claude: [skill('claude', 'frontend-design')] });
    render(<GlobalSkillsSettings />);
    const sw = await screen.findByTestId('skill-claude-frontend-design-switch');
    current = groups({ claude: [skill('claude', 'frontend-design', { enabled: false })] });
    fireEvent.click(sw);
    expect(sw.getAttribute('aria-checked')).toBe('false');
    await waitFor(() =>
      expect(calls('PUT').map(([p, o]) => [p, JSON.parse(String(o.body))])).toEqual([
        ['/api/v2/skills/enabled', { id: 'claude:frontend-design', enabled: false }],
      ]),
    );
    expect(screen.getByTestId('skill-claude-frontend-design-switch').getAttribute('aria-checked')).toBe('false');
  });

  it('a failed switch goes back on and says so', async () => {
    current = groups({ global: [skill('global', 'pdf-tidy')] });
    apiMock.mockImplementation(async (path: string, options?: RequestInit) => {
      if (path === '/api/v2/skills' && !options?.method) return { groups: current };
      if (path === '/api/v2/skills/enabled') throw new MockApiError(404, 'no skill');
      return {};
    });
    render(<GlobalSkillsSettings />);
    const sw = await screen.findByTestId('skill-global-pdf-tidy-switch');
    fireEvent.click(sw);
    expect((await screen.findByTestId('global-skills-error')).textContent).toBe("Couldn't switch this skill.");
    expect(screen.getByTestId('skill-global-pdf-tidy-switch').getAttribute('aria-checked')).toBe('true');
  });
});

describe('GlobalSkillsSettings — delete', () => {
  it('asks once in place, then DELETEs the global skill and refreshes', async () => {
    current = groups({ global: [skill('global', 'pdf-tidy')] });
    render(<GlobalSkillsSettings />);
    fireEvent.click(await screen.findByTestId('skill-global-pdf-tidy-delete'));
    expect(calls('DELETE')).toEqual([]);
    current = groups({});
    fireEvent.click(screen.getByTestId('skill-global-pdf-tidy-confirm'));
    await waitFor(() => expect(calls('DELETE').map(([p]) => p)).toEqual(['/api/v2/skills/pdf-tidy']));
    await screen.findByTestId('global-skills-empty');
  });

  it('cancel keeps the skill and sends nothing', async () => {
    current = groups({ global: [skill('global', 'pdf-tidy')] });
    render(<GlobalSkillsSettings />);
    fireEvent.click(await screen.findByTestId('skill-global-pdf-tidy-delete'));
    fireEvent.click(screen.getByTestId('skill-global-pdf-tidy-cancel'));
    expect(screen.getByTestId('skill-global-pdf-tidy-delete')).toBeInTheDocument();
    expect(calls('DELETE')).toEqual([]);
  });
});

describe('GlobalSkillsSettings — upload', () => {
  it('POSTs the file as multipart and lists the new skill', async () => {
    current = groups({});
    render(<GlobalSkillsSettings />);
    const input = await screen.findByTestId('global-skills-upload');
    apiMock.mockImplementation(async (path: string, options?: RequestInit) => {
      if (path === '/api/v2/skills' && options?.method === 'POST') {
        current = groups({ global: [skill('global', 'pdf-tidy')] });
        return { ...skill('global', 'pdf-tidy'), replaced: false };
      }
      if (path === '/api/v2/skills') return { groups: current };
      return {};
    });
    const file = new File(['# pdf-tidy\nClean PDFs.'], 'SKILL.md', { type: 'text/markdown' });
    fireEvent.change(input, { target: { files: [file] } });
    await screen.findByTestId('skill-global-pdf-tidy');
    const post = calls('POST')[0];
    expect(post[1].body).toBeInstanceOf(FormData);
    expect((post[1].body as FormData).get('file')).toBe(file);
  });

  it('a 409 reads as "already exists" (keyed off the status, not the detail)', async () => {
    current = groups({});
    render(<GlobalSkillsSettings />);
    const input = await screen.findByTestId('global-skills-upload');
    apiMock.mockImplementation(async (path: string, options?: RequestInit) => {
      if (options?.method === 'POST') throw new MockApiError(409, 'Skill already exists: pdf-tidy');
      if (path === '/api/v2/skills') return { groups: current };
      return {};
    });
    fireEvent.change(input, { target: { files: [new File(['x'], 'SKILL.md')] } });
    expect((await screen.findByTestId('global-skills-error')).textContent).toBe(
      'A skill with this name already exists.',
    );
  });
});
