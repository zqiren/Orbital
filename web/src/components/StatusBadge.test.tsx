// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Spec 145: the header badge's dot uses the readable status tokens (the
// bright fills read at ~2:1 on light surfaces).

import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import StatusBadge from './StatusBadge';

describe('StatusBadge dot colours', () => {
  it.each([
    ['running', 'bg-success-text'],
    ['waiting', 'bg-success-text'],
    ['queued', 'bg-accent-text'],
    ['error', 'bg-error-text'],
    ['needs_input', 'bg-warning-text'],
    ['pending_approval', 'bg-warning-text'],
    ['idle', 'bg-idle'],
  ] as const)('%s → %s', (status, cls) => {
    render(<StatusBadge status={status} size="sm" />);
    expect(screen.getByTestId('status-dot')).toHaveClass(cls);
  });
});
