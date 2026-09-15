// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useEffect, useMemo, useRef } from 'react';
import type { DocumentNavState } from './types';

/** Zoom shared by the paged engines: fit-width by default, manual after +/−. */
export type ZoomState = { mode: 'fit' } | { mode: 'manual'; scale: number };

export const MIN_ZOOM = 0.25;
export const MAX_ZOOM = 4;
const ZOOM_STEP = 1.25;

export function clampZoom(scale: number): number {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, scale));
}

export function zoomIn(current: number): ZoomState {
  return { mode: 'manual', scale: clampZoom(current * ZOOM_STEP) };
}

export function zoomOut(current: number): ZoomState {
  return { mode: 'manual', scale: clampZoom(current / ZOOM_STEP) };
}

/**
 * Publish zoom-only navigation (the Word engines) while `ready`, and clear it
 * on unmount. Actions are stable; they read the latest scale through a ref.
 */
export function useZoomNav(
  onNav: (nav: DocumentNavState | null) => void,
  ready: boolean,
  scale: number,
  zoom: ZoomState,
  setZoom: (zoom: ZoomState) => void,
): void {
  const scaleRef = useRef(scale);
  useEffect(() => {
    scaleRef.current = scale;
  }, [scale]);

  const actions = useMemo(
    () => ({
      zoomIn: () => setZoom(zoomIn(scaleRef.current)),
      zoomOut: () => setZoom(zoomOut(scaleRef.current)),
      fitWidth: () => setZoom({ mode: 'fit' }),
    }),
    [setZoom],
  );

  const fit = zoom.mode === 'fit';
  useEffect(() => {
    if (ready) onNav({ kind: 'zoom', zoomPercent: Math.round(scale * 100), fit, ...actions });
  }, [onNav, ready, scale, fit, actions]);

  useEffect(() => () => onNav(null), [onNav]);
}
