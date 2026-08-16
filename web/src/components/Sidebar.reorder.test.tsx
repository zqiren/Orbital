// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

// Drag-END wiring for the sidebar project reorder (spec 056).
//
// jsdom has no layout, so a real pointer drag cannot be simulated — the
// activation constraints (5px mouse distance, 250ms touch long-press) and the
// scroll-vs-drag behaviour on a real phone are NOT covered here and need the
// QR mobile pass. What IS covered is everything downstream of the drop: the
// active/over pair a sensor would produce is fed straight into dnd-kit's
// onDragEnd, and we assert the ordered id list the Sidebar derives from it.
//
// @dnd-kit is therefore stubbed in this file only. Sidebar.test.tsx renders
// the same component against the real library.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import type { ReactNode } from 'react';
import type { DragEndEvent } from '@dnd-kit/core';
import type { Project } from '../types';

afterEach(() => cleanup());

const apiMock = vi.hoisted(() => vi.fn());
vi.mock('../config', () => ({ api: apiMock }));

// Captures the onDragEnd handler the Sidebar hands to DndContext so a test can
// fire a synthetic drop.
const dnd = vi.hoisted(() => ({
  onDragEnd: null as ((event: DragEndEvent) => void) | null,
}));

vi.mock('@dnd-kit/core', () => ({
  DndContext: ({
    children,
    onDragEnd,
  }: {
    children: ReactNode;
    onDragEnd: (event: DragEndEvent) => void;
  }) => {
    dnd.onDragEnd = onDragEnd;
    return <>{children}</>;
  },
  MouseSensor: 'MouseSensor',
  TouchSensor: 'TouchSensor',
  closestCenter: () => [],
  useSensor: (sensor: unknown, options: unknown) => ({ sensor, options }),
  useSensors: (...sensors: unknown[]) => sensors,
}));

vi.mock('@dnd-kit/sortable', async () => {
  const actual = await vi.importActual<typeof import('@dnd-kit/sortable')>(
    '@dnd-kit/sortable',
  );
  return {
    // arrayMove is the ordering primitive under test — keep the real one.
    arrayMove: actual.arrayMove,
    verticalListSortingStrategy: 'vertical',
    SortableContext: ({ children }: { children: ReactNode }) => <>{children}</>,
    useSortable: () => ({
      listeners: {},
      setNodeRef: () => {},
      setActivatorNodeRef: () => {},
      transform: null,
      transition: undefined,
      isDragging: false,
    }),
  };
});

import Sidebar from './Sidebar';

function makeProject(id: string, name: string, extra: Partial<Project> = {}): Project {
  return {
    project_id: id,
    name,
    workspace: `/tmp/${id}`,
    model: '',
    api_key: '',
    base_url: null,
    autonomy: 'hands_off',
    instructions: '',
    ...extra,
  };
}

const scratch = makeProject('scratch-1', 'Quick Tasks', { is_scratch: true });
const alpha = makeProject('p-a', 'Alpha');
const beta = makeProject('p-b', 'Beta');
const gamma = makeProject('p-c', 'Gamma');

type ReorderFn = (orderedIds: string[]) => Promise<unknown>;
let onReorderProjects: ReturnType<typeof vi.fn<ReorderFn>>;

function renderSidebar(projects: Project[]) {
  return render(
    <Sidebar
      projects={projects}
      agentStatuses={{}}
      statusSummaries={{}}
      pendingApprovals={{}}
      route={{ name: 'list' }}
      connectionState="connected"
      onSelectProject={vi.fn()}
      onSelectCalendar={vi.fn()}
      onSelectWorkbench={vi.fn()}
      onNewProject={vi.fn()}
      onSettings={vi.fn()}
      onReorderProjects={onReorderProjects}
    />,
  );
}

function drop(activeId: string, overId: string) {
  dnd.onDragEnd?.({
    active: { id: activeId },
    over: { id: overId },
  } as unknown as DragEndEvent);
}

beforeEach(() => {
  apiMock.mockReset();
  apiMock.mockResolvedValue({ entries: [] });
  dnd.onDragEnd = null;
  onReorderProjects = vi.fn<ReorderFn>().mockResolvedValue([]);
});

describe('Sidebar reorder — drag end', () => {
  it('sends the complete reordered id list when a row is dropped onto another', () => {
    renderSidebar([alpha, beta, gamma]);
    drop('p-c', 'p-a'); // Gamma dragged to the top
    expect(onReorderProjects).toHaveBeenCalledWith(['p-c', 'p-a', 'p-b']);
  });

  it('moves a row down as well as up', () => {
    renderSidebar([alpha, beta, gamma]);
    drop('p-a', 'p-c');
    expect(onReorderProjects).toHaveBeenCalledWith(['p-b', 'p-c', 'p-a']);
  });

  it('omits the pinned scratch project from the persisted order', () => {
    // Quick Tasks is not draggable and is not the user's to position, so it
    // must never appear in the ordered ids the daemon persists.
    renderSidebar([scratch, alpha, beta]);
    drop('p-b', 'p-a');
    expect(onReorderProjects).toHaveBeenCalledWith(['p-b', 'p-a']);
  });

  it('does nothing when a row is dropped on itself', () => {
    renderSidebar([alpha, beta]);
    drop('p-a', 'p-a');
    expect(onReorderProjects).not.toHaveBeenCalled();
  });

  it('does nothing when the drag ends outside any row', () => {
    renderSidebar([alpha, beta]);
    dnd.onDragEnd?.({ active: { id: 'p-a' }, over: null } as unknown as DragEndEvent);
    expect(onReorderProjects).not.toHaveBeenCalled();
  });

  it('does not swallow a rejected write as an unhandled rejection', async () => {
    // The caller (useProjects.reorderProjects) owns the revert; the Sidebar
    // only has to keep the rejection from escaping.
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});
    onReorderProjects = vi.fn<ReorderFn>().mockRejectedValue(new Error('offline'));
    renderSidebar([alpha, beta]);
    drop('p-b', 'p-a');
    await vi.waitFor(() => expect(consoleError).toHaveBeenCalled());
    expect(consoleError.mock.calls[0][0]).toBe('Failed to reorder projects');
    consoleError.mockRestore();
  });

  it('leaves a plain row click routing to the project, not reordering', () => {
    renderSidebar([alpha, beta]);
    fireEvent.click(screen.getByText('Alpha'));
    expect(onReorderProjects).not.toHaveBeenCalled();
  });
});
