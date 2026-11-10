/**
 * Batch 1110 item 1: the autonomy level governs Orbital's own agent only;
 * the ⓘ beside the title says so on hover and on tap.
 */
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { LocaleProvider } from '../i18n/LocaleContext';
import AutonomyScopeHint from './AutonomyScopeHint';

afterEach(() => cleanup());

describe('AutonomyScopeHint', () => {
  it('is a labelled control that opens the sentence on click and closes on Escape', () => {
    render(<LocaleProvider><AutonomyScopeHint /></LocaleProvider>);
    const btn = screen.getByTestId('autonomy-scope-hint');
    expect(btn.getAttribute('aria-label')).toBe('Who this applies to');
    expect(screen.queryByTestId('autonomy-scope-pop')).toBeNull();
    fireEvent.click(btn);
    const pop = screen.getByTestId('autonomy-scope-pop');
    expect(pop.textContent).toMatch(/Applies to the Orbital agent\./);
    expect(pop.textContent).toMatch(/Claude Code, Codex and other agents keep their own permission models/);
    expect(btn.getAttribute('aria-expanded')).toBe('true');
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByTestId('autonomy-scope-pop')).toBeNull();
  });

  it('opens on hover', () => {
    render(<LocaleProvider><AutonomyScopeHint /></LocaleProvider>);
    fireEvent.pointerEnter(screen.getByTestId('autonomy-scope-hint'), { pointerType: 'mouse' });
    expect(screen.getByTestId('autonomy-scope-pop')).toBeTruthy();
  });
});
