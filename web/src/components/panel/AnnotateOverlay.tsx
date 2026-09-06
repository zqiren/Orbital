// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Spec 078 §5.4 — box-drawing overlay used over the live browser canvas and
 * over image previews. Absolutely positioned to fill its parent (the parent
 * must be position:relative). When `active`, a drag draws a box and a note
 * field appears; Enter calls onAdd. Existing boxes render numbered pins.
 * CONTRACT FILE: props are final; workstream E implements.
 */
import { useCallback, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { useT } from '../../i18n/useT';
import type { AnnotationBox } from '../../utils/annotations';

export interface AnnotateOverlayProps {
  active: boolean;
  boxes: { n: number; box: AnnotationBox }[];
  onAdd: (box: AnnotationBox, note: string) => void;
  onRemove?: (n: number) => void;
}

/** A drag smaller than this in either axis is a click, not a box. */
const MIN_BOX = 6;

function normalize(ax: number, ay: number, bx: number, by: number): AnnotationBox {
  return {
    x: Math.min(ax, bx),
    y: Math.min(ay, by),
    w: Math.abs(bx - ax),
    h: Math.abs(by - ay),
  };
}

export default function AnnotateOverlay({ active, boxes, onAdd, onRemove }: AnnotateOverlayProps) {
  const t = useT();
  const rootRef = useRef<HTMLDivElement | null>(null);
  const startRef = useRef<{ x: number; y: number } | null>(null);
  const [drag, setDrag] = useState<AnnotationBox | null>(null);
  const [pending, setPending] = useState<AnnotationBox | null>(null);
  const [note, setNote] = useState('');

  // Pointer coordinates relative to the overlay's own box. jsdom reports a
  // zero rect, which is harmless: the offsets then equal the client coords.
  const localPoint = useCallback((e: ReactPointerEvent) => {
    const rect = rootRef.current?.getBoundingClientRect();
    return { x: e.clientX - (rect?.left ?? 0), y: e.clientY - (rect?.top ?? 0) };
  }, []);

  const commit = useCallback(() => {
    if (!pending) return;
    onAdd(pending, note.trim());
    setPending(null);
    setNote('');
  }, [pending, note, onAdd]);

  const cancel = useCallback(() => {
    setPending(null);
    setNote('');
  }, []);

  const handlePointerDown = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      if (!active) return;
      // Only the overlay surface starts a drag — clicks on the note field, the
      // Add button or a pin's ✕ must reach those controls.
      if (e.target !== e.currentTarget) return;
      // While annotating, the drag belongs to the overlay and must not reach
      // the live browser canvas / image underneath.
      e.preventDefault();
      e.stopPropagation();
      const p = localPoint(e);
      startRef.current = p;
      setPending(null);
      setNote('');
      setDrag({ x: p.x, y: p.y, w: 0, h: 0 });
      // Not implemented in jsdom, and absent on some older WebKit builds.
      e.currentTarget.setPointerCapture?.(e.pointerId);
    },
    [active, localPoint],
  );

  const handlePointerMove = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      const start = startRef.current;
      if (!active || !start) return;
      e.preventDefault();
      e.stopPropagation();
      const p = localPoint(e);
      setDrag(normalize(start.x, start.y, p.x, p.y));
    },
    [active, localPoint],
  );

  const handlePointerUp = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      const start = startRef.current;
      if (!active || !start) return;
      e.preventDefault();
      e.stopPropagation();
      startRef.current = null;
      e.currentTarget.releasePointerCapture?.(e.pointerId);
      const p = localPoint(e);
      const box = normalize(start.x, start.y, p.x, p.y);
      setDrag(null);
      if (box.w < MIN_BOX || box.h < MIN_BOX) return;
      setPending(box);
    },
    [active, localPoint],
  );

  // The overlay itself is transparent to the pointer unless it is annotating,
  // so the live browser view underneath keeps receiving clicks. Its children
  // opt back in individually, which keeps a pin's ✕ usable at rest.
  const rootStyle: React.CSSProperties = {
    position: 'absolute',
    inset: 0,
    pointerEvents: active ? 'auto' : 'none',
    touchAction: active ? 'none' : undefined,
    cursor: active ? 'crosshair' : undefined,
  };

  const stroke = 'var(--color-accent, #ff3b30)';

  return (
    <div
      ref={rootRef}
      data-testid="annotate-overlay"
      data-active={active ? 'true' : 'false'}
      style={rootStyle}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
    >
      {boxes.map(({ n, box }) => (
        <div
          key={n}
          data-testid={`annotation-box-${n}`}
          style={{
            position: 'absolute',
            left: box.x,
            top: box.y,
            width: box.w,
            height: box.h,
            border: `2px solid ${stroke}`,
            pointerEvents: 'none',
          }}
        >
          <span
            data-testid={`annotation-pin-${n}`}
            style={{
              position: 'absolute',
              left: -2,
              top: -20,
              minWidth: 18,
              height: 18,
              lineHeight: '18px',
              padding: '0 4px',
              textAlign: 'center',
              fontSize: 11,
              fontWeight: 700,
              color: '#fff',
              background: stroke,
              borderRadius: 3,
            }}
          >
            {n}
          </span>
          {onRemove && (
            <button
              type="button"
              aria-label={t('panel.annotation.remove')}
              onClick={(e) => {
                e.stopPropagation();
                onRemove(n);
              }}
              style={{
                position: 'absolute',
                right: -9,
                top: -9,
                width: 18,
                height: 18,
                lineHeight: '16px',
                fontSize: 12,
                color: '#fff',
                background: stroke,
                borderRadius: 9,
                pointerEvents: 'auto',
                cursor: 'pointer',
              }}
            >
              ×
            </button>
          )}
        </div>
      ))}

      {/* The rubber band while the pointer is down. */}
      {drag && (
        <div
          data-testid="annotate-drag"
          style={{
            position: 'absolute',
            left: drag.x,
            top: drag.y,
            width: drag.w,
            height: drag.h,
            border: `2px dashed ${stroke}`,
            pointerEvents: 'none',
          }}
        />
      )}

      {/* The just-drawn box plus its note field, anchored under the box. */}
      {pending && (
        <>
          <div
            data-testid="annotate-pending"
            style={{
              position: 'absolute',
              left: pending.x,
              top: pending.y,
              width: pending.w,
              height: pending.h,
              border: `2px dashed ${stroke}`,
              pointerEvents: 'none',
            }}
          />
          <div
            style={{
              position: 'absolute',
              left: pending.x,
              top: pending.y + pending.h + 6,
              display: 'flex',
              gap: 4,
              alignItems: 'center',
              pointerEvents: 'auto',
              zIndex: 2,
            }}
            onPointerDown={(e) => e.stopPropagation()}
          >
            <input
              autoFocus
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder={t('panel.annotation.note')}
              aria-label={t('panel.annotation.note')}
              data-testid="annotate-note"
              onKeyDown={(e) => {
                // Never let Enter/Esc reach the composer or the panel shell.
                e.stopPropagation();
                if (e.nativeEvent.isComposing || e.keyCode === 229) return;
                if (e.key === 'Enter') {
                  e.preventDefault();
                  commit();
                } else if (e.key === 'Escape') {
                  e.preventDefault();
                  cancel();
                }
              }}
              className="text-[12px] px-2 py-1 rounded border border-border bg-card text-primary outline-none focus:border-accent"
            />
            <button
              type="button"
              onClick={commit}
              className="text-[12px] px-2 py-1 rounded bg-accent text-white font-medium"
            >
              {t('panel.annotation.add')}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
