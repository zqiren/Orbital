// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import ApprovalCard from './ApprovalCard';

// useAgent reads BASE_URL + the websocket auth context; stub the methods we
// need so the card can render without a live daemon.
const { approveMock, denyMock } = vi.hoisted(() => ({
  approveMock: vi.fn(),
  denyMock: vi.fn(),
}));
vi.mock('../hooks/useAgent', () => ({
  useAgent: () => ({
    approveToolCall: approveMock,
    denyToolCall: denyMock,
  }),
}));

const baseApproval = {
  what: 'Run a shell command',
  tool_name: 'shell',
  tool_call_id: 'call-1',
  tool_args: { command: 'ls' },
  recent_activity: [],
};

describe('ApprovalCard — Deny button visual treatment (FE post-build #1a)', () => {
  it('Deny button uses the destructive style (error-colored border + error text)', () => {
    render(
      <ApprovalCard approval={baseApproval} projectId="p1" sessionId="s1" />,
    );
    const deny = screen.getByRole('button', { name: 'Deny' });
    // Destructive treatment — matches the "Confirm Deny" button on the
    // expanded deny pane at ApprovalCard.tsx:232.
    expect(deny.className).toContain('border-error/40');
    expect(deny.className).toContain('text-error');
    expect(deny.className).toContain('hover:bg-error/5');
    // Must NOT carry the old ghost treatment that made it blend into the
    // card background.
    expect(deny.className).not.toContain('text-secondary');
    expect(deny.className).not.toContain('border-border');
  });

  it('Deny button has 44px minimum tap target on mobile and stays the SAME size as before', () => {
    render(
      <ApprovalCard approval={baseApproval} projectId="p1" sessionId="s1" />,
    );
    const deny = screen.getByRole('button', { name: 'Deny' });
    expect(deny.className).toContain('min-h-[44px]');
    // Layout dims unchanged — only the color treatment moved. 1122 type
    // scale: `text-micro` is the 11px step the button always used.
    expect(deny.className).toContain('text-micro');
    expect(deny.className).toContain('px-2.5');
  });

  it('Auto-approve button keeps its ghost styling — only Deny was made destructive', () => {
    render(
      <ApprovalCard approval={baseApproval} projectId="p1" sessionId="s1" />,
    );
    const autoApprove = screen.getByRole('button', { name: 'Auto-approve 10 min' });
    expect(autoApprove.className).toContain('text-secondary');
    expect(autoApprove.className).toContain('border-border');
    expect(autoApprove.className).not.toContain('border-error/40');
  });
});

describe('ApprovalCard — button row layout (FE post-build #1b)', () => {
  it('button container is NOT sticky-positioned (no more sticky / bottom-0 collisions with adjacent chat rows)', () => {
    render(
      <ApprovalCard approval={baseApproval} projectId="p1" sessionId="s1" />,
    );
    const deny = screen.getByRole('button', { name: 'Deny' });
    const container = deny.parentElement;
    expect(container).not.toBeNull();
    // The previous implementation used sticky bottom-0 with bg-[#FFFBF0]
    // bleed; both are gone so the buttons live in normal flow.
    expect(container!.className).not.toContain('sticky');
    expect(container!.className).not.toContain('bottom-0');
    expect(container!.className).not.toContain('bg-[#FFFBF0]');
    // Layout intent preserved: stacked on mobile, right-justified row on desktop.
    expect(container!.className).toContain('flex-col');
    expect(container!.className).toContain('md:flex-row');
    expect(container!.className).toContain('md:justify-end');
  });
});

describe('ApprovalCard — localized what sentence (China i18n)', () => {
  it('derives the sentence from tool_name/tool_args, not the backend English what', () => {
    render(
      <ApprovalCard approval={baseApproval} projectId="p1" sessionId="s1" />,
    );
    // activity.ran template ("Ran: {command}"), not the wire prose.
    expect(screen.getByText('Ran: ls')).toBeTruthy();
    expect(screen.queryByText('Run a shell command')).toBeNull();
  });

  it('falls back to the wire what for an unknown tool without args', () => {
    render(
      <ApprovalCard
        approval={{ ...baseApproval, tool_name: 'mystery', tool_args: undefined as never, what: 'Do a mystery thing' }}
        projectId="p1"
        sessionId="s1"
      />,
    );
    expect(screen.getByText('Do a mystery thing')).toBeTruthy();
  });
});

// Spec 117 D5: a sandbox denial raises an approval of kind `unsandboxed_run`.
// The card shows the command and the detected reason, and offers three ways
// to run it outside the sandbox (plus the ordinary Deny).
describe('ApprovalCard — unsandboxed_run kind (spec 117)', () => {
  const unsandboxed = {
    what: 'Run outside the sandbox',
    tool_name: 'shell',
    tool_call_id: 'call-9',
    tool_args: { command: 'git push origin main' },
    recent_activity: [],
    kind: 'unsandboxed_run',
    command: 'git push origin main',
    reason: 'Seatbelt denied file-read-data ~/.ssh/config',
  };

  it('shows the command in a code block, the reason, and the three answers', () => {
    render(<ApprovalCard approval={unsandboxed} projectId="p1" sessionId="s1" />);
    const code = screen.getByText('git push origin main');
    expect(code.tagName).toBe('CODE');
    expect(screen.getByText('Seatbelt denied file-read-data ~/.ssh/config')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Allow once' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Always allow git in this project' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Turn off Safe mode for this project' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Deny' })).toBeTruthy();
    // The generic answers do not apply to this kind.
    expect(screen.queryByRole('button', { name: 'Approve' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Auto-approve 10 min' })).toBeNull();
  });

  it.each([
    ['Allow once', 'once'],
    ['Always allow git in this project', 'always'],
    ['Turn off Safe mode for this project', 'disable_safe_mode'],
  ])('"%s" approves with response_payload %s', async (label, payload) => {
    approveMock.mockClear();
    approveMock.mockResolvedValue({});
    const onResolve = vi.fn();
    render(
      <ApprovalCard approval={unsandboxed} projectId="p1" sessionId="s1" onResolve={onResolve} />,
    );
    fireEvent.click(screen.getByRole('button', { name: label }));
    await waitFor(() =>
      expect(approveMock).toHaveBeenCalledWith('p1', 'call-9', undefined, undefined, 's1', payload),
    );
    await waitFor(() => expect(onResolve).toHaveBeenCalledWith('call-9', 'approved'));
  });

  it('Deny still goes through the existing deny route', async () => {
    denyMock.mockClear();
    denyMock.mockResolvedValue({});
    render(<ApprovalCard approval={unsandboxed} projectId="p1" sessionId="s1" />);
    fireEvent.click(screen.getByRole('button', { name: 'Deny' }));
    fireEvent.click(screen.getByRole('button', { name: 'Confirm Deny' }));
    await waitFor(() => expect(denyMock).toHaveBeenCalledWith('p1', 'call-9', 'Denied by user', 's1'));
  });
});

// The daemon names the program "Always allow" would allowlist (`executable`),
// which is not always the command's first word (`cmd /c curl.exe x` →
// `curl.exe`), and "" when the command cannot be allowlisted at all (chained,
// piped, interpreter-wrapped). Older daemons omit the field.
describe('ApprovalCard — unsandboxed_run uses the backend executable', () => {
  const wrapped = {
    what: 'Run outside the sandbox',
    tool_name: 'shell',
    tool_call_id: 'call-10',
    tool_args: { command: 'cmd /c curl.exe https://example.com' },
    recent_activity: [],
    kind: 'unsandboxed_run',
    command: 'cmd /c curl.exe https://example.com',
    reason: 'Sandbox denied network access',
  };

  it('labels "Always allow" with the executable field, not the first word', () => {
    render(
      <ApprovalCard approval={{ ...wrapped, executable: 'curl.exe' }} projectId="p1" sessionId="s1" />,
    );
    expect(screen.getByRole('button', { name: 'Always allow curl.exe in this project' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Always allow cmd in this project' })).toBeNull();
  });

  it('hides "Always allow" when the command cannot be allowlisted (executable "")', () => {
    render(
      <ApprovalCard
        approval={{ ...wrapped, command: 'git pull && npm i', executable: '' }}
        projectId="p1"
        sessionId="s1"
      />,
    );
    expect(screen.queryByRole('button', { name: /^Always allow/ })).toBeNull();
    expect(screen.getByRole('button', { name: 'Allow once' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Turn off Safe mode for this project' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Deny' })).toBeTruthy();
  });

  it('falls back to the first word when the field is absent (older daemon)', () => {
    render(<ApprovalCard approval={wrapped} projectId="p1" sessionId="s1" />);
    expect(screen.getByRole('button', { name: 'Always allow cmd in this project' })).toBeTruthy();
  });
});

// Spec 195: a worker's question the user answers (its turn was the user's
// pinned send or another worker's dispatch) rides the approval card as
// `kind: "question"`: the asking agent's words, an answer box, its options
// when it offers some, Send and Deny. Same chrome and routes as an approval.
describe('ApprovalCard — question kind (spec 195)', () => {
  const question = {
    what: 'Sub-agent dsh asks: 颜色',
    tool_name: 'question',
    tool_call_id: 'q-1',
    tool_args: {
      prompt: '颜色',
      options: [{ id: 'q1', prompt: '用哪种颜色？', options: [{ id: 'red', label: '红' }, { id: 'blue', label: '蓝' }] }],
    },
    recent_activity: [],
    kind: 'question',
  };

  it('shows the question, its options, an answer box, Send and Deny, and nothing of an approval', () => {
    render(<ApprovalCard approval={question} projectId="p1" sessionId="s1" asker="dsh" />);
    expect(screen.getByText('颜色')).toBeTruthy();
    expect(screen.getByText('用哪种颜色？')).toBeTruthy();
    expect(screen.getByRole('button', { name: '红' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '蓝' })).toBeTruthy();
    expect(screen.getByRole('textbox')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Send' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Deny' })).toBeTruthy();
    expect(screen.getByText('Question')).toBeTruthy();
    expect(screen.getByTestId('approval-asker').getAttribute('data-agent')).toBe('dsh');
    expect(screen.queryByRole('button', { name: 'Approve' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Auto-approve 10 min' })).toBeNull();
    expect(screen.queryByText('Action details')).toBeNull();
  });

  it('Send stays off until there is an answer, then sends it as the reply text', async () => {
    approveMock.mockClear();
    approveMock.mockResolvedValue({});
    const onResolve = vi.fn();
    render(<ApprovalCard approval={question} projectId="p1" sessionId="s1" onResolve={onResolve} />);
    const send = screen.getByRole('button', { name: 'Send' }) as HTMLButtonElement;
    expect(send.disabled).toBe(true);
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '蓝绿色' } });
    expect(send.disabled).toBe(false);
    fireEvent.click(send);
    await waitFor(() =>
      expect(approveMock).toHaveBeenCalledWith('p1', 'q-1', '蓝绿色', undefined, 's1', undefined),
    );
    await waitFor(() => expect(onResolve).toHaveBeenCalledWith('q-1', 'approved'));
  });

  it('Enter in the answer box sends it', async () => {
    approveMock.mockClear();
    approveMock.mockResolvedValue({});
    render(<ApprovalCard approval={question} projectId="p1" sessionId="s1" />);
    const box = screen.getByRole('textbox');
    fireEvent.change(box, { target: { value: '红色' } });
    fireEvent.keyDown(box, { key: 'Enter' });
    await waitFor(() =>
      expect(approveMock).toHaveBeenCalledWith('p1', 'q-1', '红色', undefined, 's1', undefined),
    );
  });

  it('an option answers with its id', async () => {
    approveMock.mockClear();
    approveMock.mockResolvedValue({});
    render(<ApprovalCard approval={question} projectId="p1" sessionId="s1" />);
    fireEvent.click(screen.getByRole('button', { name: '蓝' }));
    await waitFor(() =>
      expect(approveMock).toHaveBeenCalledWith('p1', 'q-1', undefined, undefined, 's1', 'blue'),
    );
  });

  it('Deny declines the question in one click', async () => {
    denyMock.mockClear();
    denyMock.mockResolvedValue({});
    const onResolve = vi.fn();
    render(<ApprovalCard approval={question} projectId="p1" sessionId="s1" onResolve={onResolve} />);
    fireEvent.click(screen.getByRole('button', { name: 'Deny' }));
    await waitFor(() => expect(denyMock).toHaveBeenCalledWith('p1', 'q-1', 'Denied by user', 's1'));
    await waitFor(() => expect(onResolve).toHaveBeenCalledWith('q-1', 'denied'));
  });

  it('once answered it collapses to one line', () => {
    render(<ApprovalCard approval={question} projectId="p1" sessionId="s1" resolved="approved" />);
    expect(screen.getByText(/Answered/)).toBeTruthy();
    expect(screen.queryByRole('textbox')).toBeNull();
  });

  it('a question offering no options shows no option buttons', () => {
    render(
      <ApprovalCard
        approval={{ ...question, tool_args: { prompt: '你的名字？' } }}
        projectId="p1"
        sessionId="s1"
      />,
    );
    expect(screen.getByText('你的名字？')).toBeTruthy();
    expect(screen.getAllByRole('button').map((b) => b.textContent)).toEqual(['Deny', 'Send']);
  });
});
