// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { isFirstRun, needsSetupWizard, wizardHandoffProjectId, wizardLandingProjectId } from './firstRun';

const scratch = { project_id: 'proj-scratch', is_scratch: true };
const own = (id: string) => ({ project_id: id, is_scratch: false });

describe('isFirstRun', () => {
  it('is true when Quick Tasks is the only project', () => {
    expect(isFirstRun([scratch])).toBe(true);
  });

  it('is false once the user has a project of their own', () => {
    expect(isFirstRun([scratch, own('a')])).toBe(false);
  });

  it('is false for an EMPTY list — that is "not loaded / fetch failed", not "new user"', () => {
    // listProjects() resolves [] on error and the list is [] before the first
    // fetch lands. The daemon always creates Quick Tasks, so a loaded list is
    // never empty; treating [] as first-run would flash the welcome screen at
    // every existing user on every cold start.
    expect(isFirstRun([])).toBe(false);
  });
});

describe('wizardLandingProjectId', () => {
  it('is null when the wizard created nothing (→ open Create Project)', () => {
    expect(wizardLandingProjectId([scratch])).toBeNull();
    expect(wizardLandingProjectId([])).toBeNull();
  });

  it('lands in the most recently added own project when the wizard imported some', () => {
    expect(wizardLandingProjectId([scratch, own('a'), own('b')])).toBe('b');
  });
});

// The wizard's Finish opens this project on a new session (never its newest
// one, which after an import is the background read).
describe('wizardHandoffProjectId', () => {
  it('opens the imported landing project when it is in the list', () => {
    expect(wizardHandoffProjectId([scratch, own('a'), own('b')], 'a')).toBe('a');
  });

  it('falls back to the most recently added own project (landing lost, e.g. a reload mid-wizard)', () => {
    expect(wizardHandoffProjectId([scratch, own('a'), own('b')], null)).toBe('b');
    expect(wizardHandoffProjectId([scratch, own('a'), own('b')], 'gone')).toBe('b');
  });

  it('is null when there is no own project (→ open Create Project)', () => {
    expect(wizardHandoffProjectId([scratch], null)).toBeNull();
  });
});

// Keyless onboarding (contract §1): the first-run gate App.tsx applies.
describe('needsSetupWizard — first-run gate', () => {
  it('shows the wizard when there is no key and the step was never skipped', () => {
    expect(needsSetupWizard({ llm: { api_key_set: false }, setup: { api_key_skipped: false } })).toBe(true);
  });

  it('does not show it once the key step was skipped', () => {
    expect(needsSetupWizard({ llm: { api_key_set: false }, setup: { api_key_skipped: true } })).toBe(false);
  });

  it('does not show it when a key exists, skipped or not', () => {
    expect(needsSetupWizard({ llm: { api_key_set: true }, setup: { api_key_skipped: false } })).toBe(false);
    expect(needsSetupWizard({ llm: { api_key_set: true }, setup: { api_key_skipped: true } })).toBe(false);
  });

  it('an older daemon (no `setup`) reads as not skipped', () => {
    expect(needsSetupWizard({ llm: { api_key_set: false } })).toBe(true);
    expect(needsSetupWizard({ llm: { api_key_set: true } })).toBe(false);
  });

  it('an unreachable daemon (no settings) shows the wizard, as before', () => {
    expect(needsSetupWizard(null)).toBe(true);
  });
});
