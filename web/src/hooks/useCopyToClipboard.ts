// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * One clipboard implementation for the whole app (BACKLOG spec 068).
 *
 * `supported` is the load-bearing part. `navigator.clipboard` requires a
 * SECURE CONTEXT, and this app is reached three ways that do not agree:
 *
 *   desktop app   http://127.0.0.1:{port}   available  (loopback is trustworthy)
 *   relay/mobile  https://…up.railway.app   available
 *   LAN dev + QR  http://<LAN-IP>:5173      UNDEFINED
 *
 * The third is the project's own mobile-testing flow (see the QR section in
 * CLAUDE.md), and it is exactly where a button that silently does nothing gets
 * mistaken for a broken feature. Callers render the control only when
 * `supported`; the phone still has native long-press selection.
 *
 * WebKit note: `writeText` must be the FIRST await inside the user-gesture
 * handler or the gesture is lost and the write is rejected. Do not add an
 * await before the `copy()` call in a click handler.
 */
export function useCopyToClipboard(resetMs = 2000) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // A row can unmount while the "Copied" state is still counting down (the
  // transcript re-renders constantly); without this the timeout fires against
  // an unmounted component.
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  const copy = useCallback(
    async (text: string): Promise<boolean> => {
      try {
        await navigator.clipboard.writeText(text);
        setCopied(true);
        if (timer.current) clearTimeout(timer.current);
        timer.current = setTimeout(() => setCopied(false), resetMs);
        return true;
      } catch {
        // Denied by permissions policy, or an insecure context that slipped
        // past the `supported` check. Never throw at a click handler.
        return false;
      }
    },
    [resetMs],
  );

  const supported =
    typeof navigator !== 'undefined' &&
    typeof navigator.clipboard?.writeText === 'function';

  return { copied, copy, supported };
}
