// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState } from 'react';
import { api } from '../config';
import type { PlatformStatus } from '../types';

interface SetupGateProps {
  setupIssues: string[];
  onComplete: () => void;
}

export default function SetupGate({ setupIssues, onComplete }: SetupGateProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSetup() {
    setLoading(true);
    setError(null);
    try {
      await api<PlatformStatus>('/api/v2/platform/setup', {
        method: 'POST',
      });
      onComplete();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Setup failed. Please try again.',
      );
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background px-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <span className="font-mono text-lg text-primary tracking-tight">
            Orbital
          </span>
        </div>

        <div className="bg-background border border-border rounded-lg p-8">
          <h1 className="text-2xl font-semibold text-primary mb-3">
            Welcome to Orbital
          </h1>
          <p className="text-secondary text-sm leading-relaxed mb-6">
            A secure sandbox environment needs to be configured before you can
            create and manage agents. This requires administrator permissions and
            may take a moment.
          </p>

          {setupIssues.length > 0 && (
            <div className="mb-6 space-y-2">
              {setupIssues.map((issue, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 text-sm text-warning bg-warning/5 border border-warning/20 rounded-lg px-3 py-2"
                >
                  <span className="shrink-0 mt-0.5">!</span>
                  <span>{issue}</span>
                </div>
              ))}
            </div>
          )}

          {error && (
            <div className="mb-6 text-sm text-error bg-error/5 border border-error/20 rounded-lg px-3 py-2">
              {error}
            </div>
          )}

          {loading ? (
            <div className="text-center py-4">
              <p className="text-sm text-secondary">
                Setting up sandbox environment. This may take up to 30
                seconds...
              </p>
            </div>
          ) : (
            <button
              onClick={handleSetup}
              className="w-full bg-accent text-white text-sm font-medium rounded-lg px-4 py-2.5 hover:bg-accent/90 transition-all duration-150"
            >
              {error ? 'Retry Setup' : 'Set Up Sandbox'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
