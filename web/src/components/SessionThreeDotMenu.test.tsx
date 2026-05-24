// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SessionThreeDotMenu } from './SessionThreeDotMenu';

afterEach(() => cleanup());

const BATCH4_TOOLTIP = 'Coming in Batch 4';

describe('SessionThreeDotMenu', () => {
  it('renders the trigger button with default aria-label', () => {
    render(<SessionThreeDotMenu />);
    expect(screen.getByTestId('session-three-dot-trigger')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /session options/i })).toBeInTheDocument();
  });

  it('accepts a custom aria-label', () => {
    render(<SessionThreeDotMenu ariaLabel="Options for session foo" />);
    expect(screen.getByRole('button', { name: /options for session foo/i })).toBeInTheDocument();
  });

  it('dropdown is not visible initially', () => {
    render(<SessionThreeDotMenu />);
    expect(screen.queryByTestId('session-three-dot-dropdown')).toBeNull();
  });

  it('opens dropdown on button click', async () => {
    render(<SessionThreeDotMenu />);
    await userEvent.click(screen.getByTestId('session-three-dot-trigger'));
    expect(screen.getByTestId('session-three-dot-dropdown')).toBeInTheDocument();
  });

  it('renders Rename, Archive, Delete actions when open', async () => {
    render(<SessionThreeDotMenu />);
    await userEvent.click(screen.getByTestId('session-three-dot-trigger'));

    expect(screen.getByTestId('session-action-rename')).toBeInTheDocument();
    expect(screen.getByTestId('session-action-archive')).toBeInTheDocument();
    expect(screen.getByTestId('session-action-delete')).toBeInTheDocument();
  });

  it('all actions are disabled', async () => {
    render(<SessionThreeDotMenu />);
    await userEvent.click(screen.getByTestId('session-three-dot-trigger'));

    expect(screen.getByTestId('session-action-rename')).toBeDisabled();
    expect(screen.getByTestId('session-action-archive')).toBeDisabled();
    expect(screen.getByTestId('session-action-delete')).toBeDisabled();
  });

  it('all actions carry the Batch-4 tooltip', async () => {
    render(<SessionThreeDotMenu />);
    await userEvent.click(screen.getByTestId('session-three-dot-trigger'));

    expect(screen.getByTestId('session-action-rename')).toHaveAttribute('title', BATCH4_TOOLTIP);
    expect(screen.getByTestId('session-action-archive')).toHaveAttribute('title', BATCH4_TOOLTIP);
    expect(screen.getByTestId('session-action-delete')).toHaveAttribute('title', BATCH4_TOOLTIP);
  });

  it('closes on Escape key', async () => {
    render(<SessionThreeDotMenu />);
    await userEvent.click(screen.getByTestId('session-three-dot-trigger'));
    expect(screen.getByTestId('session-three-dot-dropdown')).toBeInTheDocument();

    await userEvent.keyboard('{Escape}');
    expect(screen.queryByTestId('session-three-dot-dropdown')).toBeNull();
  });

  it('toggles closed on second button click', async () => {
    render(<SessionThreeDotMenu />);
    await userEvent.click(screen.getByTestId('session-three-dot-trigger'));
    expect(screen.getByTestId('session-three-dot-dropdown')).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('session-three-dot-trigger'));
    expect(screen.queryByTestId('session-three-dot-dropdown')).toBeNull();
  });
});
