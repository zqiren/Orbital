// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import CalendarPage from './CalendarPage';

afterEach(() => cleanup());

describe('CalendarPage (Wave 2 placeholder)', () => {
  it('renders the calendar title and the coming-soon copy (global mount)', () => {
    render(<CalendarPage />);
    expect(screen.getByText('Calendar')).toBeInTheDocument();
    expect(screen.getByText(/coming in this branch/i)).toBeInTheDocument();
  });

  it('does NOT mark the project-lens attribute for the global mount', () => {
    render(<CalendarPage />);
    expect(screen.getByTestId('calendar-page')).not.toHaveAttribute('data-project-lens');
  });

  it('marks the project-lens attribute when a projectId is supplied', () => {
    render(<CalendarPage projectId="proj-1" />);
    expect(screen.getByTestId('calendar-page')).toHaveAttribute('data-project-lens', 'true');
  });
});
