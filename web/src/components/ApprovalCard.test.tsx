// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import ApprovalCard from './ApprovalCard';

// useAgent reads BASE_URL + the websocket auth context; stub the methods we
// need so the card can render without a live daemon.
vi.mock('../hooks/useAgent', () => ({
  useAgent: () => ({
    approveToolCall: vi.fn(),
    denyToolCall: vi.fn(),
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
    // Layout dims unchanged — only the color treatment moved.
    expect(deny.className).toContain('text-[11px]');
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
