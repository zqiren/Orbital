// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * In-memory route model for Orbital V1 UI.
 *
 * No URL-based router — navigation is driven entirely by React state.
 * The single `route: Route` object replaces the old `view`/`tab`/`selectedProjectId` triad.
 */

export type Route =
  | { name: 'list' }
  | { name: 'create' }
  | { name: 'blocked' }
  | { name: 'settings' }
  | { name: 'calendar' }
  | { name: 'workbench' }
  | {
      name: 'project';
      projectId: string;
      /**
       * Spec 137: which sub-page of the project is showing. Undefined is the
       * project's only home, the conversation. Every other surface is a page
       * under the shared PageToolbar (Back · `Project / Page`), described by
       * one entry in the page registry (`components/projectPages.tsx`). This
       * replaces the old `tab` + `settings` + `pricing` trio: three ways of
       * saying "which page".
       */
      page?: ProjectPage;
      sessionId?: string;
      /**
       * Which pane of the Tasks page is showing. The name stays `queuePane`
       * (renaming it churns deep links and tests for no user benefit); this
       * only picks between the two panes inside the page. Undefined means
       * 'queue' — the page's default.
       */
      queuePane?: 'queue' | 'automations';
      /**
       * Optional intent to scroll the settings page to a specific section on
       * open (P3-G: the budget corner deep-links here with 'budget'). Any
       * `data-settings-section` id is valid (D1 generalized the one-off budget
       * anchor into the named-section system). Only meaningful on the
       * settings page; consumed (scrolled-to) once by SettingsView.
       */
      settingsAnchor?: string;
      /**
       * Workspace-relative path of the file to preview in the slide-out
       * FilePreviewDrawer (spec 002). Set when a user clicks a clickable path
       * in chat; the drawer is open iff this is non-null. Does NOT change the
       * page — the chat stays mounted underneath. In-memory only (not
       * persisted across reload), like the rest of the route model.
       */
      previewPath?: string;
      /**
       * One-shot composer prefill (Workbench card-tap doorway, spec
       * 2026-07-24 §5.3). ChatTab reads this, threads it into ChatView as
       * `initialDraft`, and clears it back to undefined via `onDraftConsumed`
       * once ChatView has applied it — so it loads into the composer exactly
       * once, never re-applied on a later re-render or page revisit. Never
       * auto-sent and never triggers a session spawn by itself.
       */
      draft?: string;
      /**
       * Spec 137: a built-in composer skill to pick alongside `draft` (the
       * Automations pane's "New automation" opens chat with `/automate`
       * picked). Same one-shot lifetime as `draft`, cleared with it.
       */
      draftSkill?: BuiltinSkillKey;
      /** One-shot: open with this ask quoted in the composer (spec 146). */
      askReply?: { id: string; text: string };
    };

/** Spec 137: the project's sub-pages. Chat (undefined) is the home. */
export type ProjectPage = 'tasks' | 'settings' | 'pricing';

/** Built-in composer skills (spec 137): client-side entries in the `/` picker. */
export type BuiltinSkillKey = 'automate';

export type ProjectRoute = Extract<Route, { name: 'project' }>;

/**
 * Each page's parent page. Undefined = it sits directly under the project
 * (Back goes home). A deeper page nests in the crumb (`Project / Settings /
 * Pricing`) and Back goes up one level.
 */
export const PROJECT_PAGE_PARENT: Record<ProjectPage, ProjectPage | undefined> = {
  tasks: undefined,
  settings: undefined,
  pricing: 'settings',
};

const PROJECT_PAGES: readonly ProjectPage[] = ['tasks', 'settings', 'pricing'];

/** A page key from untrusted input (a notification payload, an old caller), else undefined. */
export function parseProjectPage(value: unknown): ProjectPage | undefined {
  return typeof value === 'string' && (PROJECT_PAGES as readonly string[]).includes(value)
    ? (value as ProjectPage)
    : undefined;
}

/** The pages from the top down to `page` (`['settings', 'pricing']`); empty at home. */
export function pageTrail(page: ProjectPage | undefined): ProjectPage[] {
  const out: ProjectPage[] = [];
  for (let p = page; p !== undefined; p = PROJECT_PAGE_PARENT[p]) out.unshift(p);
  return out;
}

/**
 * The route one level up from the current page: its parent page, or the
 * chat home. A one-shot preview path never survives navigation.
 */
export function routeUp(route: ProjectRoute): ProjectRoute {
  const parent = route.page ? PROJECT_PAGE_PARENT[route.page] : undefined;
  return { ...route, page: parent, previewPath: undefined };
}

/**
 * A stable string for "which screen is this" (ErrorBoundary's reset key):
 * project, page and session. Not a URL — Orbital has no URL router.
 */
export function routeKey(route: Route): string {
  if (route.name !== 'project') return route.name;
  return `project:${route.projectId}:${route.page ?? 'chat'}:${route.sessionId ?? ''}`;
}
