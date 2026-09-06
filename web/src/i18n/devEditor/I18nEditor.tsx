// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Dev-only click-to-translate overlay. Mounted from main.tsx when
// `?i18n=edit` has been visited (see enabled.ts); never part of a build.
//
// Pick mode intercepts clicks (capture phase, before React sees them), finds
// the catalog key(s) behind the clicked element's text, and opens them in the
// side panel. Edits mutate the in-memory catalog so the app re-renders with
// the new wording immediately, and are persisted through the Vite dev plugin
// to docs/i18n/pending/ui-overrides.json for a later fold into strings.ts.
//
// The panel docks left or right, can push the app aside so nothing sits under
// it, and collapses to a corner pill — collapsing also turns picking off so
// the app is usable again; expanding restores the previous picking state.
//
// Its own chrome is intentionally plain English literals: it is a tool for
// editing the catalog, not a surface the catalog covers.

import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { STRINGS, type StringEntry } from '../strings';
import { useLocale } from '../LocaleContext';
import type { Locale } from '../locales';
import { buildTextIndex, findMatches, searchCatalog, type CatalogMatch } from './matchCatalog';

interface Override { en?: string; zh?: string }
type Side = 'left' | 'right';

const PANEL_WIDTH = 340;
const PREFS_KEY = 'orbital.i18nEdit.prefs';

const CATALOG = STRINGS as unknown as Record<string, StringEntry>;
const PRISTINE: Record<string, StringEntry> = {};

function snapshotPristine(): void {
  if (Object.keys(PRISTINE).length) return;
  for (const [k, e] of Object.entries(CATALOG)) PRISTINE[k] = { ...e };
}

/** Apply one override to the live catalog. '' for zh drops it (falls back to en). */
function applyOverride(key: string, o: Override): void {
  const entry = CATALOG[key];
  if (!entry) return;
  if (o.en !== undefined) entry.en = o.en || PRISTINE[key].en;
  if (o.zh !== undefined) {
    if (o.zh === '') delete entry.zh; else entry.zh = o.zh;
  }
}

function restorePristine(key: string): void {
  const p = PRISTINE[key];
  if (!p) return;
  const e = CATALOG[key];
  e.en = p.en;
  if (p.zh === undefined) delete e.zh; else e.zh = p.zh;
}

/** What differs from the shipped catalog for `key`, or null when nothing does. */
function pendingFor(key: string): Override | null {
  const e = CATALOG[key];
  const p = PRISTINE[key];
  const o: Override = {};
  if (e.en !== p.en) o.en = e.en;
  if ((e.zh ?? '') !== (p.zh ?? '')) o.zh = e.zh ?? '';
  return Object.keys(o).length ? o : null;
}

async function postOverride(key: string, o: Override | null): Promise<Record<string, Override>> {
  const res = await fetch('/__i18n/overrides', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key, en: o?.en ?? null, zh: o?.zh ?? null }),
  });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as Record<string, Override>;
}

interface Prefs { open: boolean; side: Side; push: boolean; pick: boolean }
const DEFAULT_PREFS: Prefs = { open: true, side: 'right', push: true, pick: true };

function readPrefs(): Prefs {
  try {
    const raw = localStorage.getItem(PREFS_KEY);
    return raw ? { ...DEFAULT_PREFS, ...(JSON.parse(raw) as Partial<Prefs>) } : DEFAULT_PREFS;
  } catch {
    return DEFAULT_PREFS;
  }
}

function writePrefs(p: Prefs): void {
  try { localStorage.setItem(PREFS_KEY, JSON.stringify(p)); } catch { /* ignore */ }
}

const font = '12px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif';
const mono = '11px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace';
const S = {
  panel: {
    position: 'fixed', top: 0, bottom: 0, width: PANEL_WIDTH, zIndex: 2147483000,
    background: '#fff', color: '#1f2328', boxShadow: '0 0 24px rgba(0,0,0,.10)',
    font, display: 'flex', flexDirection: 'column',
  } as CSSProperties,
  head: { display: 'flex', alignItems: 'center', gap: 4, padding: '6px 8px', borderBottom: '1px solid #d0d7de', background: '#f6f8fa', flexWrap: 'wrap' } as CSSProperties,
  body: { flex: 1, overflow: 'auto', padding: 10 } as CSSProperties,
  foot: { padding: '8px 10px', borderTop: '1px solid #d0d7de', color: '#57606a', background: '#f6f8fa' } as CSSProperties,
  btn: { font, padding: '3px 7px', border: '1px solid #d0d7de', borderRadius: 6, background: '#fff', cursor: 'pointer', color: '#1f2328', whiteSpace: 'nowrap' } as CSSProperties,
  btnOn: { background: '#0969da', borderColor: '#0969da', color: '#fff' } as CSSProperties,
  input: { font, width: '100%', boxSizing: 'border-box', padding: '5px 8px', border: '1px solid #d0d7de', borderRadius: 6, outline: 'none' } as CSSProperties,
  ta: { font: '13px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif', width: '100%', boxSizing: 'border-box', padding: '6px 8px', border: '1px solid #d0d7de', borderRadius: 6, resize: 'vertical', minHeight: 56, outline: 'none' } as CSSProperties,
  key: { font: mono, color: '#0969da', wordBreak: 'break-all' } as CSSProperties,
  muted: { color: '#57606a' } as CSSProperties,
  row: { padding: '6px 8px', border: '1px solid #d0d7de', borderRadius: 6, marginBottom: 6, cursor: 'pointer', background: '#fff' } as CSSProperties,
  rowOn: { borderColor: '#0969da', background: '#ddf4ff' } as CSSProperties,
  label: { display: 'block', margin: '10px 0 4px', fontWeight: 600 } as CSSProperties,
  notice: { padding: '6px 8px', borderRadius: 6, background: '#fff8c5', border: '1px solid #d4a72c', marginTop: 8 } as CSSProperties,
};

export default function I18nEditor() {
  const { locale, setLocale, refresh } = useLocale();
  const [prefs, setPrefs] = useState<Prefs>(readPrefs);
  const { open, side, push, pick } = prefs;
  const pickBeforeCollapse = useRef(prefs.pick);
  const [rev, setRev] = useState(0);
  const [overrides, setOverrides] = useState<Record<string, Override>>({});
  const [matches, setMatches] = useState<CatalogMatch[]>([]);
  const [tried, setTried] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [hover, setHover] = useState<DOMRect | null>(null);
  const [target, setTarget] = useState<DOMRect | null>(null);
  const [status, setStatus] = useState('');
  const timers = useRef<Record<string, number>>({});
  const index = useMemo(() => buildTextIndex(locale), [locale, rev]);

  const update = (patch: Partial<Prefs>) => setPrefs((p) => { const next = { ...p, ...patch }; writePrefs(next); return next; });
  const setPick = (v: boolean) => update({ pick: v });
  const collapse = () => { pickBeforeCollapse.current = pick; update({ open: false, pick: false }); setHover(null); };
  const expand = () => update({ open: true, pick: pickBeforeCollapse.current });

  // Load pending overrides once and apply them so the app shows the edited wording.
  useEffect(() => {
    snapshotPristine();
    fetch('/__i18n/overrides')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then((map: Record<string, Override>) => {
        for (const [k, o] of Object.entries(map)) applyOverride(k, o);
        setOverrides(map);
        setRev((v) => v + 1);
        refresh();
      })
      .catch((err: unknown) => setStatus(`Could not load overrides: ${String(err)}`));
  }, [refresh]);

  // Push the app aside so the panel never covers it (fixed-position app
  // elements such as modals still ignore this; dock to the other side then).
  useEffect(() => {
    const body = document.body.style;
    body.marginLeft = open && push && side === 'left' ? `${PANEL_WIDTH}px` : '';
    body.marginRight = open && push && side === 'right' ? `${PANEL_WIDTH}px` : '';
    return () => { body.marginLeft = ''; body.marginRight = ''; };
  }, [open, push, side]);

  // Element picking: capture-phase listeners run before React's handlers.
  useEffect(() => {
    const isOwn = (t: EventTarget | null) => t instanceof Element && !!t.closest('[data-i18n-editor]');
    const onClick = (e: MouseEvent) => {
      if (isOwn(e.target) || !(e.target instanceof Element)) return;
      if (!pick && !e.altKey) return;
      e.preventDefault();
      e.stopPropagation();
      const res = findMatches(e.target, index);
      setMatches(res.matches);
      setTried(res.tried);
      setSelected(res.matches[0]?.key ?? null);
      setTarget((res.el ?? e.target).getBoundingClientRect());
      if (!open) expand();
    };
    const onMove = (e: MouseEvent) => {
      if (!pick) return;
      if (isOwn(e.target) || !(e.target instanceof Element)) { setHover(null); return; }
      setHover(e.target.getBoundingClientRect());
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && pick) { setPick(false); setHover(null); }
    };
    document.addEventListener('click', onClick, true);
    document.addEventListener('mousemove', onMove, true);
    document.addEventListener('keydown', onKey, true);
    return () => {
      document.removeEventListener('click', onClick, true);
      document.removeEventListener('mousemove', onMove, true);
      document.removeEventListener('keydown', onKey, true);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pick, index, open]);

  useEffect(() => {
    document.body.style.cursor = pick ? 'crosshair' : '';
    return () => { document.body.style.cursor = ''; };
  }, [pick]);

  const edit = (key: string, field: 'en' | 'zh', value: string) => {
    applyOverride(key, { [field]: value });
    setRev((v) => v + 1);
    refresh();
    window.clearTimeout(timers.current[key]);
    timers.current[key] = window.setTimeout(() => {
      postOverride(key, pendingFor(key))
        .then((map) => { setOverrides(map); setStatus(`Saved ${key}`); })
        .catch((err: unknown) => setStatus(`Save failed: ${String(err)}`));
    }, 400);
  };

  const revert = (key: string) => {
    restorePristine(key);
    setRev((v) => v + 1);
    refresh();
    window.clearTimeout(timers.current[key]);
    postOverride(key, null)
      .then((map) => { setOverrides(map); setStatus(`Reverted ${key}`); })
      .catch((err: unknown) => setStatus(`Revert failed: ${String(err)}`));
  };

  const hits = useMemo(() => searchCatalog(query, 40), [query, rev]);
  const pendingKeys = Object.keys(overrides);
  const entry = selected ? CATALOG[selected] : null;
  const pristine = selected ? PRISTINE[selected] : null;

  const box = (r: DOMRect | null, color: string, dashed: boolean) => r && (
    <div data-i18n-editor="box" style={{
      position: 'fixed', left: r.left - 2, top: r.top - 2, width: r.width + 4, height: r.height + 4,
      border: `2px ${dashed ? 'dashed' : 'solid'} ${color}`, borderRadius: 4, pointerEvents: 'none', zIndex: 2147482999,
    }} />
  );

  if (!open) {
    return (
      <div data-i18n-editor="root">
        {box(target, '#0969da', false)}
        <button
          type="button"
          title="Expand the translation editor (restores picking). Alt+click any element also opens it."
          style={{ ...S.btn, position: 'fixed', bottom: 12, [side]: 12, zIndex: 2147483000, boxShadow: '0 2px 8px rgba(0,0,0,.15)', padding: '6px 10px' }}
          onClick={expand}
        >
          i18n editor{pendingKeys.length ? ` · ${pendingKeys.length}` : ''} · picking off
        </button>
      </div>
    );
  }

  return (
    <div data-i18n-editor="root">
      {box(hover, '#54aeff', true)}
      {box(target, '#0969da', false)}
      <aside style={{ ...S.panel, [side]: 0, [side === 'right' ? 'borderLeft' : 'borderRight']: '1px solid #d0d7de' }}>
        <div style={S.head}>
          <strong style={{ marginRight: 4 }}>Translation editor</strong>
          <span style={{ flex: 1 }} />
          <button type="button" style={{ ...S.btn, ...(pick ? S.btnOn : {}) }} onClick={() => setPick(!pick)} title="On: clicks select elements instead of acting. Off: the app works normally; Alt+click still selects. Esc turns it off.">
            {pick ? 'Picking: on' : 'Picking: off'}
          </button>
          {(['en', 'zh'] as Locale[]).map((l) => (
            <button key={l} type="button" style={{ ...S.btn, ...(locale === l ? S.btnOn : {}) }} onClick={() => setLocale(l)}>
              {l === 'en' ? 'EN' : '中文'}
            </button>
          ))}
          <button type="button" style={S.btn} onClick={() => update({ side: side === 'right' ? 'left' : 'right' })} title="Dock the panel on the other side">
            {side === 'right' ? '◧ Dock left' : 'Dock right ◨'}
          </button>
          <button type="button" style={{ ...S.btn, ...(push ? S.btnOn : {}) }} onClick={() => update({ push: !push })} title="Push the app aside so the panel never covers it">
            Push app
          </button>
          <button type="button" style={S.btn} onClick={collapse} title="Collapse to a corner pill and turn picking off">
            Collapse
          </button>
        </div>

        <div style={S.body}>
          {!pick && (
            <div style={S.notice}>
              Picking is off: the app responds to clicks normally. Alt+click an element to pick it, or turn picking back on.
            </div>
          )}
          <input
            style={{ ...S.input, marginTop: pick ? 0 : 8 }}
            placeholder="Search key, English or 中文…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          {query.trim() ? (
            <div style={{ marginTop: 8 }}>
              {hits.length === 0 && <div style={S.muted}>No catalog entry contains that.</div>}
              {hits.map((h) => (
                <div key={h.key} style={{ ...S.row, ...(selected === h.key ? S.rowOn : {}) }} onClick={() => { setSelected(h.key); setMatches([]); }}>
                  <div style={S.key}>{h.key}</div>
                  <div>{h.en}</div>
                  {h.zh && <div style={S.muted}>{h.zh}</div>}
                </div>
              ))}
            </div>
          ) : matches.length > 0 ? (
            <div style={{ marginTop: 8 }}>
              <div style={S.muted}>{matches.length === 1 ? 'Matched 1 key' : `Matched ${matches.length} keys — pick the right one`}</div>
              {matches.map((m) => (
                <div key={m.key} style={{ ...S.row, ...(selected === m.key ? S.rowOn : {}) }} onClick={() => setSelected(m.key)}>
                  <div style={S.key}>{m.key}{m.exact ? '' : ' (placeholder match)'}</div>
                  <div>{CATALOG[m.key]?.en}</div>
                  {CATALOG[m.key]?.zh && <div style={S.muted}>{CATALOG[m.key].zh}</div>}
                </div>
              ))}
            </div>
          ) : tried.length > 0 && !selected ? (
            <div style={{ marginTop: 8, ...S.muted }}>
              No catalog string matched this element. Texts tried:
              <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
                {tried.slice(0, 6).map((t) => <li key={t}>{t.length > 80 ? `${t.slice(0, 80)}…` : t}</li>)}
              </ul>
              Use the search box above, or the text may come from the backend (not translatable here).
            </div>
          ) : !selected ? (
            <div style={{ marginTop: 8, ...S.muted }}>
              {pick
                ? 'Click any button, label or text in the app to edit its translation. Elements highlight as you hover. To navigate the app, turn picking off (Esc).'
                : 'Navigate the app, then turn picking on (or Alt+click) to edit a string.'}
            </div>
          ) : null}

          {selected && entry && pristine && (
            <div style={{ marginTop: 12, paddingTop: 10, borderTop: '1px solid #d0d7de' }}>
              <div style={S.key}>{selected}</div>
              <label style={S.label}>English</label>
              <textarea style={S.ta} value={entry.en} onChange={(e) => edit(selected, 'en', e.target.value)} />
              {entry.en !== pristine.en && <div style={S.muted}>was: {pristine.en}</div>}
              <label style={S.label}>中文</label>
              <textarea style={S.ta} value={entry.zh ?? ''} placeholder="(empty = falls back to English)" onChange={(e) => edit(selected, 'zh', e.target.value)} />
              {(entry.zh ?? '') !== (pristine.zh ?? '') && <div style={S.muted}>was: {pristine.zh ?? '(none)'}</div>}
              {pendingFor(selected) && (
                <button type="button" style={{ ...S.btn, marginTop: 8 }} onClick={() => revert(selected)}>Revert this key</button>
              )}
            </div>
          )}

          {pendingKeys.length > 0 && (
            <details style={{ marginTop: 14 }}>
              <summary style={{ cursor: 'pointer' }}>{pendingKeys.length} pending change{pendingKeys.length === 1 ? '' : 's'}</summary>
              <div style={{ marginTop: 6 }}>
                {pendingKeys.map((k) => (
                  <div key={k} style={{ ...S.row, ...(selected === k ? S.rowOn : {}) }} onClick={() => { setSelected(k); setMatches([]); setQuery(''); }}>
                    <div style={S.key}>{k}</div>
                    {overrides[k].en !== undefined && <div>en: {overrides[k].en}</div>}
                    {overrides[k].zh !== undefined && <div>zh: {overrides[k].zh || '(cleared)'}</div>}
                  </div>
                ))}
                <button
                  type="button"
                  style={S.btn}
                  onClick={() => { navigator.clipboard.writeText(JSON.stringify(overrides, null, 2)).then(() => setStatus('Copied JSON')).catch(() => setStatus('Clipboard blocked')); }}
                >
                  Copy JSON
                </button>
              </div>
            </details>
          )}
        </div>

        <div style={S.foot}>
          <div>{status || 'Edits save to docs/i18n/pending/ui-overrides.json'}</div>
          <div style={{ marginTop: 4 }}>Most strings update live; a few cached ones refresh after switching EN/中文.</div>
        </div>
      </aside>
    </div>
  );
}
