// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { describe, it, expect } from 'vitest';
import { isFirstRun, wizardLandingProjectId } from './firstRun';

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
