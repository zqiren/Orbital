// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 137 — the project's page registry.
 *
 * Chat is the project's only home; every other surface is a sub-page under
 * the shared PageToolbar (filled Back · `Project / … / Page`). A page is one
 * entry here plus one link that sets `route.page` — nothing becomes a tab
 * again. Where a new page goes (the user's rule, 2026-10-27):
 *
 * - something you SET UP → a Settings section, or a page under Settings
 *   (`parent: 'settings'`) once it outgrows a section;
 * - something that RUNS WITHOUT YOU → a page reached from a status line that
 *   shows its state (Tasks, from the line under the project title);
 * - something you LOOK AT WHILE WORKING → a view in the right panel, not a page.
 *
 * The parent chain itself lives in `route.ts` (`PROJECT_PAGE_PARENT`) so the
 * pure route helpers can walk it without importing React.
 */

import type { Dispatch, ReactNode, SetStateAction } from 'react';
import type { Project } from '../types';
import type { ProjectPage, ProjectRoute, Route } from '../route';
import type { StringKey } from '../i18n/strings';
import { useT } from '../i18n/useT';
import AutomationsList from './AutomationsList';

export interface ProjectPageContext {
  project: Project;
  route: ProjectRoute;
  setRoute: Dispatch<SetStateAction<Route>>;
  /** What App handed ProjectDetail for this page (`pageContent[key]`). */
  content: ReactNode;
  agentsAvailable: Array<{ slug: string; name: string }>;
  /** Back to chat with `text` in the composer and `/automate` picked. */
  onTryInChat: (text: string) => void;
  /** "New automation": back to chat with `/automate` picked. */
  onNewAutomation: () => void;
}

export interface ProjectPageDef {
  key: ProjectPage;
  /** The page's crumb and h1. */
  titleKey: StringKey;
  backTestId: string;
  titleTestId: string;
  /** Applied to the route when Back leaves this page for its parent. */
  backPatch?: Partial<ProjectRoute>;
  render: (ctx: ProjectPageContext) => ReactNode;
}

type QueuePaneKey = 'queue' | 'automations';

// The two panes inside the Tasks page. A parent page should not carry the
// name of one of its children — hence "Tasks" for the page, "Queue" here.
const QUEUE_PANES: { key: QueuePaneKey; labelKey: StringKey }[] = [
  { key: 'queue', labelKey: 'queue.pane.queue' },
  { key: 'automations', labelKey: 'queue.section.automations' },
];

/**
 * The Tasks page body: today's Queue | Automations segmented control directly
 * under the toolbar, then the selected pane. The queue is a live stream and
 * an automation is a standing definition with an edit form — sibling views,
 * not sections of one scroll.
 */
function TasksPage({ project, route, setRoute, content, agentsAvailable, onTryInChat, onNewAutomation }: ProjectPageContext) {
  const t = useT();
  const queuePane: QueuePaneKey = route.queuePane ?? 'queue';
  return (
    <div className="flex flex-col h-full min-h-0" data-testid="tasks-page">
      {/* A segmented control, on a track: a track around both panes is what
          says "one control, two positions" (bare pills read as one card
          floating beside a line of plain text). */}
      <div className="px-6 pt-3 max-md:px-4">
        <div
          className="inline-flex gap-0.5 rounded-lg border border-border bg-nav p-0.5"
          role="tablist"
          aria-label={t('projectDetail.tab.queue')}
        >
          {QUEUE_PANES.map((pane) => {
            const active = queuePane === pane.key;
            return (
              <button
                key={pane.key}
                type="button"
                role="tab"
                aria-selected={active}
                data-testid={`queue-pane-${pane.key}`}
                onClick={() => setRoute({ ...route, page: 'tasks', queuePane: pane.key })}
                className={`text-micro font-medium px-3 py-1 rounded-md transition-colors duration-150 max-md:min-h-[40px] ${
                  active
                    ? 'bg-segment-on text-segment-on-text shadow-[0_1px_2px_rgb(0_0_0/0.06)]'
                    : 'text-secondary hover:text-primary'
                }`}
              >
                {t(pane.labelKey)}
              </button>
            );
          })}
        </div>
      </div>
      <div className="flex-1 min-h-0 overflow-hidden">
        {queuePane === 'automations' ? (
          <div className="h-full overflow-y-auto px-6 py-4 max-md:px-4">
            <AutomationsList
              projectId={project.project_id}
              agents={agentsAvailable}
              onTryInChat={onTryInChat}
              onNewInChat={onNewAutomation}
            />
          </div>
        ) : (
          content
        )}
      </div>
    </div>
  );
}

export const PROJECT_PAGES: Record<ProjectPage, ProjectPageDef> = {
  tasks: {
    key: 'tasks',
    titleKey: 'projectDetail.tab.queue',
    backTestId: 'tasks-back-button',
    titleTestId: 'tasks-title',
    render: (ctx) => <TasksPage {...ctx} />,
  },
  settings: {
    key: 'settings',
    titleKey: 'settingsModal.crumb',
    backTestId: 'settings-back-button',
    titleTestId: 'settings-modal-title',
    render: (ctx) => ctx.content,
  },
  pricing: {
    key: 'pricing',
    titleKey: 'pricing.title',
    backTestId: 'pricing-back-button',
    titleTestId: 'pricing-title',
    // Back from the pricing table lands on the Budget section it came from.
    backPatch: { settingsAnchor: 'budget' },
    render: (ctx) => ctx.content,
  },
};
