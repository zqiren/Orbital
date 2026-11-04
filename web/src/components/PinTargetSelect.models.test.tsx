// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

/**
 * Spec 149 — the pin menu's model line ("C2 with chevron").
 *
 *  - each worker row shows "<model> · <thinking>" as a control of its own,
 *    not nested in the row's option button;
 *  - clicking the row still picks the recipient and closes the menu;
 *  - clicking the line swaps the side card to the Model / Thinking panel,
 *    clicking it again (or Esc, or elsewhere in the menu) swaps it back, and
 *    Esc does not close the menu while the panel shows;
 *  - choosing applies immediately; Default clears the override;
 *  - Orbital's row shows the project's card and picks another.
 */

import { render, screen, cleanup, fireEvent, act, within } from '@testing-library/react';
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock }));

import PinTargetSelect, { type PinModels } from './PinTargetSelect';
import { __resetAgentAvailability } from '../hooks/useAgentAvailability';
import {
  modelLineText,
  workerModelInfo,
  type WorkerModelInfo,
} from '../hooks/useConversationModels';
import { LocaleProvider } from '../i18n/LocaleContext';

beforeEach(() => {
  apiMock.mockReset();
  apiMock.mockRejectedValue(new Error('offline'));
});

afterEach(() => {
  cleanup();
  act(() => __resetAgentAvailability());
});

const AGENTS = [
  { slug: 'claude-code', name: 'Claude Code' },
  { slug: 'codex', name: 'Codex' },
  { slug: 'minimax-code', name: 'MiniMax Code' },
];

const CLAUDE: WorkerModelInfo = {
  slug: 'claude-code',
  models: [
    { value: 'sonnet', label: 'Sonnet 5.5' },
    { value: 'opus', label: 'Opus 5.5' },
  ],
  levels: ['low', 'medium', 'high', 'xhigh', 'max'],
  settingsModel: 'sonnet',
  settingsEffort: null,
  model: null,
  effort: null,
};
const CODEX: WorkerModelInfo = {
  slug: 'codex', models: [{ value: 'gpt-5.6', label: 'gpt-5.6' }],
  levels: ['low', 'medium', 'high'], settingsModel: null, settingsEffort: null,
  model: 'gpt-5.6', effort: 'low',
};
const MINIMAX: WorkerModelInfo = {
  slug: 'minimax-code', models: null, levels: null,
  settingsModel: 'm:minimax:MiniMax-M3:v:thinking', settingsEffort: null,
  model: null, effort: null,
};

function models(over: Partial<PinModels> = {}): PinModels {
  return {
    workers: { 'claude-code': CLAUDE, codex: CODEX, 'minimax-code': MINIMAX },
    editable: true,
    onWorkerChange: vi.fn().mockResolvedValue(true),
    manager: {
      cards: [{ id: 'c1', name: 'DeepSeek v4.1 · OpenCode Go' }, { id: 'c2', name: 'Kimi K2.5' }],
      cardId: null,
      defaultName: 'DeepSeek v4.1 · OpenCode Go',
      onChange: vi.fn().mockResolvedValue(true),
    },
    onMenuOpen: vi.fn(),
    ...over,
  };
}

function renderMenu(m: PinModels, props: Partial<Parameters<typeof PinTargetSelect>[0]> = {}) {
  const onChange = vi.fn();
  const utils = render(
    <PinTargetSelect agents={AGENTS} value="claude-code" onChange={onChange} models={m} {...props} />,
  );
  fireEvent.click(screen.getByRole('button', { name: 'Choose who this chat talks to' }));
  return { ...utils, onChange };
}

const line = (slug: string) => screen.getByTestId(`pin-model-line-${slug}`);

describe('model line helpers', () => {
  it('reads override, else Settings, else Default', () => {
    expect(modelLineText(CLAUDE, 'Default')).toBe('Sonnet 5.5');
    expect(modelLineText({ ...CLAUDE, model: 'opus', effort: 'high' }, 'Default')).toBe('Opus 5.5 · high');
    expect(modelLineText({ ...CLAUDE, settingsModel: null }, 'Default')).toBe('Default');
    expect(modelLineText(MINIMAX, 'Default')).toBe('m:minimax:MiniMax-M3:v:thinking');
  });

  it('folds a Settings entry: labels, levels, free text', () => {
    const info = workerModelInfo({
      slug: 'claude-code',
      config: { model: 'opus' },
      param_schema: {
        model: { allowed: ['opus'], labels: { opus: 'Opus 5.5' } },
        effort: { allowed: ['low', 'high'] },
        'permission-mode': { allowed: ['plan'] },
      },
    }, { effort: 'high' });
    expect(info).toMatchObject({
      models: [{ value: 'opus', label: 'Opus 5.5' }], levels: ['low', 'high'],
      settingsModel: 'opus', model: null, effort: 'high',
    });
    expect(workerModelInfo({ slug: 'x', param_schema: { model: { allowed: null } } }, undefined)?.models).toBeNull();
    expect(workerModelInfo({ slug: 'y', param_schema: {} }, undefined)).toBeNull();
  });
});

describe('PinTargetSelect model line', () => {
  it('opening the menu asks for fresh choices', () => {
    const m = models();
    renderMenu(m);
    expect(m.onMenuOpen).toHaveBeenCalledTimes(1);
  });

  it('each worker row shows its line with a chevron, outside the option button', () => {
    renderMenu(models());
    expect(line('claude-code').textContent).toBe('Sonnet 5.5');
    expect(line('codex').textContent).toBe('gpt-5.6 · low');
    // Long ids are never cut on desktop.
    expect(line('minimax-code').textContent).toBe('m:minimax:MiniMax-M3:v:thinking');
    expect(line('claude-code').querySelector('svg')).not.toBeNull();
    // The line is a real button and not inside the row's option.
    expect(line('claude-code').tagName).toBe('BUTTON');
    for (const opt of screen.getAllByRole('option')) {
      expect(opt.contains(line('claude-code'))).toBe(false);
      expect(opt.querySelector('button')).toBeNull();
    }
  });

  it('clicking the row still picks the recipient and closes the menu', () => {
    const { onChange } = renderMenu(models());
    fireEvent.click(screen.getByRole('option', { name: /Codex/ }));
    expect(onChange).toHaveBeenCalledWith('codex');
    expect(screen.queryByRole('listbox')).toBeNull();
  });

  it('clicking the line swaps the side card to the panel and back', () => {
    const m = models();
    const { onChange } = renderMenu(m);
    fireEvent.click(line('claude-code'));
    expect(onChange).not.toHaveBeenCalled();
    expect(screen.getByRole('listbox')).toBeTruthy();
    const card = screen.getByTestId('pin-model-card');
    expect(card.getAttribute('data-card-for')).toBe('claude-code');
    expect(line('claude-code').getAttribute('aria-expanded')).toBe('true');
    const radios = within(card).getAllByRole('radio');
    expect(radios.map((r) => r.textContent)).toEqual([
      'Default(Sonnet 5.5)', 'Sonnet 5.5', 'Opus 5.5',
      'low', 'medium', 'high', 'xhigh', 'max',
    ]);
    expect(within(card).getByText('Used by new sessions in this project. Change the default in Settings.')).toBeTruthy();
    fireEvent.click(line('claude-code'));
    expect(screen.queryByTestId('pin-model-card')).toBeNull();
  });

  it('a hover over another row does not move the open panel', () => {
    renderMenu(models());
    fireEvent.click(line('claude-code'));
    fireEvent.mouseEnter(screen.getByTestId('pin-row-codex'));
    expect(screen.getByTestId('pin-model-card').getAttribute('data-card-for')).toBe('claude-code');
  });

  it('Esc returns the card to usage without closing the menu; a second Esc closes it', () => {
    renderMenu(models());
    fireEvent.click(line('codex'));
    expect(screen.getByTestId('pin-model-card')).toBeTruthy();
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByTestId('pin-model-card')).toBeNull();
    expect(screen.getByRole('listbox')).toBeTruthy();
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByRole('listbox')).toBeNull();
  });

  it('a press elsewhere in the menu returns the card to usage', () => {
    renderMenu(models());
    fireEvent.click(line('codex'));
    fireEvent.mouseDown(screen.getByRole('listbox'));
    expect(screen.queryByTestId('pin-model-card')).toBeNull();
  });

  describe('in a window too narrow for the menu and its card', () => {
    const place = (left: number, right: number) =>
      vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect')
        .mockReturnValue({ left, right } as DOMRect);
    const width = window.innerWidth;

    beforeEach(() => { window.innerWidth = 929; });
    afterEach(() => {
      window.innerWidth = width;
      vi.restoreAllMocks();
    });

    it('slides left so the card stays on screen', () => {
      place(498, 1106);
      renderMenu(models());
      fireEvent.click(line('codex'));
      expect(screen.getByRole('listbox').parentElement!.style.transform).toBe('translateX(-185px)');
    });

    it('never slides past the left edge', () => {
      place(100, 1300);
      renderMenu(models());
      expect(screen.getByRole('listbox').parentElement!.style.transform).toBe('translateX(-92px)');
    });

    it('stays put when everything fits', () => {
      place(258, 866);
      renderMenu(models());
      expect(screen.getByRole('listbox').parentElement!.style.transform).toBe('');
    });
  });

  it('choosing applies at once; Default clears the override', () => {
    const m = models();
    renderMenu(m);
    fireEvent.click(line('claude-code'));
    const card = screen.getByTestId('pin-model-card');
    fireEvent.click(within(card).getByTestId('pin-model-opt-opus'));
    expect(m.onWorkerChange).toHaveBeenLastCalledWith('claude-code', { model: 'opus' });
    fireEvent.click(within(card).getByTestId('pin-model-level-high'));
    expect(m.onWorkerChange).toHaveBeenLastCalledWith('claude-code', { effort: 'high' });
    fireEvent.click(within(card).getByTestId('pin-model-opt-default'));
    expect(m.onWorkerChange).toHaveBeenLastCalledWith('claude-code', { model: null });
  });

  it('the current thinking pick, clicked again, goes back to Settings', () => {
    const m = models();
    renderMenu(m);
    fireEvent.click(line('codex'));
    fireEvent.click(within(screen.getByTestId('pin-model-card')).getByTestId('pin-model-level-low'));
    expect(m.onWorkerChange).toHaveBeenLastCalledWith('codex', { effort: null });
  });

  it('a free-text agent offers Default + Other model id and no thinking', () => {
    const m = models();
    renderMenu(m, { value: 'minimax-code' });
    fireEvent.click(line('minimax-code'));
    const card = screen.getByTestId('pin-model-card');
    expect(within(card).queryByText('Thinking')).toBeNull();
    const input = within(card).getByTestId('pin-model-other-input');
    fireEvent.change(input, { target: { value: 'm:minimax:MiniMax-M3:v:fast' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(m.onWorkerChange).toHaveBeenLastCalledWith('minimax-code', { model: 'm:minimax:MiniMax-M3:v:fast' });
  });

  it('a failed save says so in the panel', async () => {
    const m = models({ onWorkerChange: vi.fn().mockResolvedValue(false) });
    renderMenu(m);
    fireEvent.click(line('claude-code'));
    await act(async () => {
      fireEvent.click(within(screen.getByTestId('pin-model-card')).getByTestId('pin-model-opt-opus'));
    });
    expect(within(screen.getByTestId('pin-model-card')).getByTestId('pin-model-error')).toBeTruthy();
  });

  it('on a phone the same panel opens inline under its row', () => {
    renderMenu(models());
    expect(screen.queryByTestId('pin-model-inline-codex')).toBeNull();
    fireEvent.click(line('codex'));
    const inline = screen.getByTestId('pin-model-inline-codex');
    expect(inline.className).toContain('md:hidden');
    expect(within(inline).getByTestId('pin-model-panel-codex')).toBeTruthy();
    // A press inside the inline panel keeps it open.
    fireEvent.mouseDown(within(inline).getByTestId('pin-model-level-high'));
    expect(screen.getByTestId('pin-model-inline-codex')).toBeTruthy();
  });

  it('before the conversation exists the line is text, not a control', () => {
    renderMenu(models({ editable: false }));
    expect(line('claude-code').tagName).toBe('SPAN');
  });

  it('a worker without model info keeps the plain row', () => {
    renderMenu(models({ workers: { 'claude-code': CLAUDE } }));
    expect(screen.queryByTestId('pin-model-line-codex')).toBeNull();
    expect(screen.getByRole('option', { name: /Codex/ }).textContent).toBe('Codex');
  });

  it('the standalone menu (queue, automations) never shows lines', () => {
    renderMenu(models(), { variant: 'standalone' });
    expect(screen.queryByTestId('pin-model-line-claude-code')).toBeNull();
    expect(screen.queryByTestId('pin-model-line-orbital')).toBeNull();
  });
});

describe('PinTargetSelect Orbital row', () => {
  it('shows the project card and lists every card with the project footer', () => {
    const m = models();
    renderMenu(m);
    expect(line('orbital').textContent).toBe('DeepSeek v4.1 · OpenCode Go');
    fireEvent.click(line('orbital'));
    const card = screen.getByTestId('pin-model-card');
    expect(card.getAttribute('data-card-for')).toBe('orbital');
    expect(within(card).getAllByRole('radio').map((r) => r.textContent)).toEqual([
      'Default(DeepSeek v4.1 · OpenCode Go)', 'DeepSeek v4.1 · OpenCode Go', 'Kimi K2.5',
    ]);
    expect(within(card).getByText('Applies to this project.')).toBeTruthy();
    fireEvent.click(within(card).getByTestId('pin-model-card-c2'));
    expect(m.manager!.onChange).toHaveBeenCalledWith('c2');
  });

  it('a project on a named card shows that card', () => {
    renderMenu(models({ manager: { ...models().manager!, cardId: 'c2' } }));
    expect(line('orbital').textContent).toBe('Kimi K2.5');
  });

  it('clicking the Orbital row itself still unpins', () => {
    const { onChange } = renderMenu(models());
    fireEvent.click(screen.getByRole('option', { name: /Orbital/ }));
    expect(onChange).toHaveBeenCalledWith(null);
  });
});

describe('PinTargetSelect model line in Chinese', () => {
  it('renders the panel copy in zh', () => {
    localStorage.setItem('orbital.locale', 'zh');
    render(
      <LocaleProvider>
        <PinTargetSelect agents={AGENTS} value="claude-code" onChange={() => {}} models={models()} />
      </LocaleProvider>,
    );
    fireEvent.click(screen.getByRole('button', { name: '选择此对话的交流对象' }));
    fireEvent.click(line('claude-code'));
    const card = screen.getByTestId('pin-model-card');
    expect(within(card).getByText('本项目的新会话都用它。默认值在设置中修改。')).toBeTruthy();
    expect(within(card).getByText('思考强度')).toBeTruthy();
    localStorage.removeItem('orbital.locale');
  });
});
