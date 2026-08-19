// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useRef } from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import SettingsRail, {
  scrollToSettingsSection,
  type SettingsRailSection,
} from './SettingsRail';

// ---------------------------------------------------------------------------
// jsdom shims. The codebase has no existing IntersectionObserver mock, so we
// use a minimal manual one that records instances and lets tests drive the
// callback. scrollIntoView does not exist on jsdom's HTMLElement at all.
// ---------------------------------------------------------------------------

type IOEntry = { target: Element; isIntersecting: boolean };
type IOCallback = (entries: IOEntry[], observer: MockIntersectionObserver) => void;

const ioInstances: MockIntersectionObserver[] = [];

class MockIntersectionObserver {
  cb: IOCallback;
  observed: Element[] = [];
  constructor(cb: IOCallback) {
    this.cb = cb;
    ioInstances.push(this);
  }
  observe(el: Element) {
    this.observed.push(el);
  }
  unobserve() {}
  disconnect() {}
}

/** Elements scrollIntoView was invoked on, in call order. */
let scrolledTo: Element[] = [];
let scrollSpy: ReturnType<typeof vi.fn>;

beforeEach(() => {
  ioInstances.length = 0;
  scrolledTo = [];
  scrollSpy = vi.fn(function (this: Element) {
    scrolledTo.push(this);
  });
  vi.stubGlobal('IntersectionObserver', MockIntersectionObserver);
  HTMLElement.prototype.scrollIntoView = scrollSpy as unknown as (
    arg?: boolean | ScrollIntoViewOptions,
  ) => void;
});

afterEach(() => {
  vi.unstubAllGlobals();
  delete (HTMLElement.prototype as { scrollIntoView?: unknown }).scrollIntoView;
});

/** The last-created observer is the live one (earlier ones are disconnected). */
function liveObserver(): MockIntersectionObserver {
  expect(ioInstances.length).toBeGreaterThan(0);
  return ioInstances[ioInstances.length - 1];
}

// ---------------------------------------------------------------------------
// Harness: rail + tagged sections inside one container, mirroring how
// GlobalSettings / SettingsView mount it. 'connectors' is declared in the
// sections array but only present in the DOM when withConnectors is set —
// the reserved-id case.
// ---------------------------------------------------------------------------

const SECTIONS: SettingsRailSection[] = [
  { id: 'alpha', labelKey: 'global.language' }, // en: "Language"
  { id: 'beta', labelKey: 'settings.budget.label' }, // en: "Budget"
  { id: 'connectors', labelKey: 'settingsRail.connectors' }, // en: "Connectors"
];

function Harness({ withConnectors = false }: { withConnectors?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  return (
    <div ref={ref} data-testid="settings-scroll-container">
      <SettingsRail sections={SECTIONS} containerRef={ref} />
      <div data-settings-section="alpha">Alpha body</div>
      <div data-settings-section="beta">Beta body</div>
      {withConnectors && <div data-settings-section="connectors">Connectors body</div>}
    </div>
  );
}

function sectionEl(id: string): Element {
  const el = document.querySelector(`[data-settings-section="${id}"]`);
  expect(el).not.toBeNull();
  return el as Element;
}

describe('SettingsRail (desktop)', () => {
  it('renders one entry per tagged section present in the DOM, in DOM order', () => {
    render(<Harness />);
    const rail = screen.getByTestId('settings-rail');
    expect(rail).toBeInTheDocument();
    const labels = screen.getAllByRole('button').map((b) => b.textContent);
    expect(labels).toEqual(['Language', 'Budget']);
  });

  it('does NOT render an entry for a declared id with no matching element (reserved "connectors")', () => {
    render(<Harness />);
    expect(screen.queryByRole('button', { name: 'Connectors' })).not.toBeInTheDocument();
  });

  it('renders the connectors entry once a matching tagged element exists', () => {
    render(<Harness withConnectors />);
    expect(screen.getByRole('button', { name: 'Connectors' })).toBeInTheDocument();
  });

  it('picks up sections that mount late (MutationObserver re-discovery)', async () => {
    render(<Harness />);
    expect(screen.queryByRole('button', { name: 'Connectors' })).not.toBeInTheDocument();
    await act(async () => {
      const el = document.createElement('div');
      el.setAttribute('data-settings-section', 'connectors');
      screen.getByTestId('settings-scroll-container').appendChild(el);
      await Promise.resolve(); // flush the MutationObserver microtask
    });
    expect(screen.getByRole('button', { name: 'Connectors' })).toBeInTheDocument();
  });

  it('clicking an entry scrolls the tagged section into view (shared budget-anchor mechanism)', () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole('button', { name: 'Budget' }));
    expect(scrollSpy).toHaveBeenCalledTimes(1);
    expect(scrollSpy).toHaveBeenCalledWith({ behavior: 'smooth', block: 'start' });
    expect(scrolledTo[0]).toBe(sectionEl('beta'));
  });

  it('clicking an entry highlights it optimistically (aria-current)', () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole('button', { name: 'Budget' }));
    expect(screen.getByRole('button', { name: 'Budget' })).toHaveAttribute('aria-current', 'true');
    expect(screen.getByRole('button', { name: 'Language' })).not.toHaveAttribute('aria-current');
  });

  it('scrollspy highlights the section reported intersecting', () => {
    render(<Harness />);
    const io = liveObserver();
    act(() => {
      io.cb([{ target: sectionEl('beta'), isIntersecting: true }], io);
    });
    expect(screen.getByRole('button', { name: 'Budget' })).toHaveAttribute('aria-current', 'true');
    expect(screen.getByRole('button', { name: 'Language' })).not.toHaveAttribute('aria-current');
  });

  it('when several sections intersect, the topmost (DOM order) wins', () => {
    render(<Harness />);
    const io = liveObserver();
    act(() => {
      io.cb([{ target: sectionEl('beta'), isIntersecting: true }], io);
    });
    act(() => {
      io.cb([{ target: sectionEl('alpha'), isIntersecting: true }], io);
    });
    expect(screen.getByRole('button', { name: 'Language' })).toHaveAttribute('aria-current', 'true');
    expect(screen.getByRole('button', { name: 'Budget' })).not.toHaveAttribute('aria-current');
  });

  it('renders nothing when no tagged sections exist', () => {
    function Empty() {
      const ref = useRef<HTMLDivElement>(null);
      return (
        <div ref={ref}>
          <SettingsRail sections={SECTIONS} containerRef={ref} />
        </div>
      );
    }
    render(<Empty />);
    expect(screen.queryByTestId('settings-rail')).not.toBeInTheDocument();
    expect(screen.queryByTestId('settings-jump-menu')).not.toBeInTheDocument();
  });
});

describe('SettingsRail (mobile jump menu)', () => {
  beforeEach(() => {
    // App-standard isMobile pattern reads matchMedia('(max-width: 767px)').
    vi.stubGlobal(
      'matchMedia',
      vi.fn().mockReturnValue({
        matches: true,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      }),
    );
  });

  it('renders a compact jump menu with an option per present section only', () => {
    render(<Harness />);
    expect(screen.queryByTestId('settings-rail')).not.toBeInTheDocument();
    const select = screen.getByTestId('settings-jump-menu');
    const options = Array.from(select.querySelectorAll('option')).map((o) => o.textContent);
    // First option is the placeholder; no dead 'Connectors' entry.
    expect(options).toEqual(['Jump to section…', 'Language', 'Budget']);
  });

  it('selecting a section scrolls it into view', () => {
    render(<Harness />);
    fireEvent.change(screen.getByTestId('settings-jump-menu'), { target: { value: 'beta' } });
    expect(scrollSpy).toHaveBeenCalledTimes(1);
    expect(scrolledTo[0]).toBe(sectionEl('beta'));
  });
});

// ---------------------------------------------------------------------------
// Grouped rails (project settings). A group heading must never outlive the
// sections it names — the whole point of deriving the runs from the entries
// that survived the DOM filter rather than from the declaration array.
// ---------------------------------------------------------------------------

const GROUPED: SettingsRailSection[] = [
  { id: 'alpha', labelKey: 'global.language', groupKey: 'settings.group.project' },
  { id: 'beta', labelKey: 'settings.budget.label', groupKey: 'settings.group.limits' },
  { id: 'connectors', labelKey: 'settingsRail.connectors', groupKey: 'settings.group.capabilities' },
  { id: 'danger', labelKey: 'settings.danger.title' },
];

function GroupedHarness({ withConnectors = false }: { withConnectors?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  return (
    <div ref={ref} data-testid="settings-scroll-container">
      <SettingsRail sections={GROUPED} containerRef={ref} />
      <div data-settings-section="alpha">Alpha body</div>
      {withConnectors && <div data-settings-section="connectors">Connectors body</div>}
      <div data-settings-section="beta">Beta body</div>
      <div data-settings-section="danger">Danger body</div>
    </div>
  );
}

describe('SettingsRail grouping', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('clusters the desktop rail by group without printing the chapter labels', () => {
    render(<GroupedHarness withConnectors />);
    const rail = screen.getByTestId('settings-rail');
    // DOM order, not declaration order: connectors sits before beta above.
    const labels = screen.getAllByRole('button').map((b) => b.textContent);
    expect(labels).toEqual(['Language', 'Connectors', 'Budget', 'Danger Zone']);
    // One list per run — the clustering IS the grouping on this surface.
    expect(rail.querySelectorAll('ul')).toHaveLength(4);
    // …and the chapter names themselves never reach the rail.
    expect(rail.textContent).not.toContain('Project');
    expect(rail.textContent).not.toContain('Capabilities');
    expect(rail.textContent).not.toContain('Limits & safety');
  });

  it('drops a run entirely when none of its sections are in the DOM', () => {
    render(<GroupedHarness />);
    const rail = screen.getByTestId('settings-rail');
    expect(screen.queryByRole('button', { name: 'Connectors' })).not.toBeInTheDocument();
    // 4 runs minus the absent capabilities one.
    expect(rail.querySelectorAll('ul')).toHaveLength(3);
  });

  it('groups the mobile jump menu with optgroups, leaving ungrouped entries loose', () => {
    vi.stubGlobal(
      'matchMedia',
      vi.fn().mockReturnValue({
        matches: true,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      }),
    );
    render(<GroupedHarness withConnectors />);
    const select = screen.getByTestId('settings-jump-menu');
    const groups = Array.from(select.querySelectorAll('optgroup')).map((g) => [
      g.label,
      Array.from(g.children).map((o) => o.textContent),
    ]);
    expect(groups).toEqual([
      ['Project', ['Language']],
      ['Capabilities', ['Connectors']],
      ['Limits & safety', ['Budget']],
    ]);
    const loose = Array.from(select.children)
      .filter((c) => c.tagName === 'OPTION')
      .map((o) => o.textContent);
    expect(loose).toEqual(['Jump to section…', 'Danger Zone']);
  });
});

describe('scrollToSettingsSection helper', () => {
  it('scrolls the tagged element and returns true', () => {
    const host = document.createElement('div');
    const target = document.createElement('div');
    target.setAttribute('data-settings-section', 'budget');
    host.appendChild(target);
    expect(scrollToSettingsSection(host, 'budget')).toBe(true);
    expect(scrolledTo[0]).toBe(target);
  });

  it('returns false and does not throw for an unknown id', () => {
    const host = document.createElement('div');
    expect(scrollToSettingsSection(host, 'nope')).toBe(false);
    expect(scrollSpy).not.toHaveBeenCalled();
  });
});
