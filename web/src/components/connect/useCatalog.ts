// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useEffect, useState } from 'react';
import type { CatalogProvider } from '../../types';
import { fetchCatalog } from './connectApi';

/** The connect screen's provider catalog, fetched once per page load. */
export function useCatalog(): CatalogProvider[] | null {
  const [catalog, setCatalog] = useState<CatalogProvider[] | null>(null);
  useEffect(() => {
    let live = true;
    fetchCatalog().then((c) => { if (live) setCatalog(c); }).catch(() => { if (live) setCatalog([]); });
    return () => { live = false; };
  }, []);
  return catalog;
}
