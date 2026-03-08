// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState, useEffect, useCallback, useRef } from 'react';
import { QRCodeSVG } from 'qrcode.react';
import { api, isRelayMode, ApiError } from '../config';
import {
  Smartphone,
  RefreshCw,
  Trash2,
  Scan,
  Laptop,
  LinkIcon,
  Clock,
  Wifi,
} from 'lucide-react';

interface PairedDevice {
  phone_id: string;
  paired_at: string;
}

type PairState = 'idle' | 'loading' | 'active' | 'error' | 'no-relay';

export default function PairPhone() {
  const [state, setState] = useState<PairState>('idle');
  const [code, setCode] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [devices, setDevices] = useState<PairedDevice[]>([]);
  const [revokingId, setRevokingId] = useState<string | null>(null);
  const [countdown, setCountdown] = useState(300); // 5 min in seconds
  const [relayUrl, setRelayUrl] = useState('http://localhost:3000');
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // LAN mode state
  const [lanUrl, setLanUrl] = useState<string | null>(null);
  const [lanLoading, setLanLoading] = useState(false);
  const [lanError, setLanError] = useState<string | null>(null);

  const fetchLanUrl = useCallback(async () => {
    setLanLoading(true);
    setLanError(null);
    try {
      const result = await api<{ ip: string | null; error?: string }>('/api/v2/network/lan-url');
      if (result.ip) {
        setLanUrl(`${window.location.protocol}//${result.ip}:${window.location.port}`);
      } else {
        setLanUrl(null);
      }
      if (!result.ip && result.error) {
        setLanError(result.error);
      }
    } catch {
      setLanError('Failed to detect LAN address');
      setLanUrl(null);
    } finally {
      setLanLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!isRelayMode) {
      fetchLanUrl();
    }
  }, [fetchLanUrl]);

  const loadDevices = useCallback(async () => {
    try {
      const list = await api<PairedDevice[]>('/api/v2/pairing/devices');
      setDevices(list);
    } catch {
      setDevices([]);
    }
  }, []);

  useEffect(() => {
    loadDevices();
  }, [loadDevices]);

  // Countdown timer when QR is active
  useEffect(() => {
    if (state === 'active') {
      setCountdown(300);
      timerRef.current = setInterval(() => {
        setCountdown((prev) => {
          if (prev <= 1) {
            // Code expired — reset
            if (timerRef.current) clearInterval(timerRef.current);
            setCode(null);
            setState('idle');
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
      return () => {
        if (timerRef.current) clearInterval(timerRef.current);
      };
    }
  }, [state]);

  async function startPairing() {
    setState('loading');
    setError(null);
    setCode(null);
    try {
      const result = await api<{ code: string; relay_url: string }>('/api/v2/pairing/start', {
        method: 'POST',
      });
      setCode(result.code);
      if (result.relay_url) setRelayUrl(result.relay_url);
      setState('active');
    } catch (err) {
      if (err instanceof ApiError && err.status === 503) {
        setState('no-relay');
        setError(
          'Cloud relay is not configured. Set the AGENT_OS_RELAY_URL environment variable and restart the daemon to enable remote pairing.',
        );
      } else {
        setState('error');
        setError(
          err instanceof ApiError ? err.detail : 'Failed to generate pairing code',
        );
      }
    }
  }

  async function revokeDevice(phoneId: string) {
    setRevokingId(phoneId);
    try {
      await api(`/api/v2/pairing/devices/${phoneId}`, { method: 'DELETE' });
      setDevices((prev) => prev.filter((d) => d.phone_id !== phoneId));
    } catch {
      // ignore
    } finally {
      setRevokingId(null);
    }
  }

  function resetPairing() {
    if (timerRef.current) clearInterval(timerRef.current);
    setCode(null);
    setState('idle');
    setError(null);
  }

  const qrValue = code ? `${relayUrl}/pair?code=${code}` : '';
  const minutes = Math.floor(countdown / 60);
  const seconds = countdown % 60;

  // ---- LAN mode: simple QR with LAN URL, no pairing code exchange ----
  if (!isRelayMode) {
    return (
      <div>
        {/* ---- Section header ---- */}
        <div className="flex items-center gap-2.5 mb-5">
          <div className="w-8 h-8 rounded-lg bg-accent/10 flex items-center justify-center">
            <Wifi className="w-4 h-4 text-accent" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-primary leading-tight">
              Mobile Access
            </h3>
            <p className="text-xs text-secondary">
              Open Orbital on your phone via local network
            </p>
          </div>
        </div>

        <div className="bg-sidebar border border-border rounded-xl overflow-hidden">
          {lanLoading ? (
            <div className="p-6 flex items-center justify-center gap-3">
              <RefreshCw className="w-4 h-4 text-secondary animate-spin" />
              <span className="text-sm text-secondary">Detecting LAN address...</span>
            </div>
          ) : lanUrl ? (
            <div className="flex max-md:flex-col">
              {/* Left: instructions */}
              <div className="flex-1 p-6 flex flex-col justify-center">
                <h4 className="text-base font-semibold text-primary mb-4">
                  Scan to open Orbital on your phone
                </h4>
                <ol className="space-y-3">
                  {[
                    'Connect your phone to the same Wi-Fi network',
                    'Open your phone\u2019s camera or QR scanner',
                    'Point it at this QR code to open',
                  ].map((step, i) => (
                    <li key={i} className="flex items-start gap-3">
                      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-accent/15 text-accent text-xs font-bold flex items-center justify-center mt-0.5">
                        {i + 1}
                      </span>
                      <span className="text-sm text-secondary leading-snug">
                        {step}
                      </span>
                    </li>
                  ))}
                </ol>

                {/* Refresh button */}
                <div className="flex items-center gap-3 mt-5 pt-4 border-t border-border">
                  <Wifi className="w-3.5 h-3.5 text-secondary" />
                  <span className="text-xs text-secondary">Local network</span>
                  <button
                    onClick={fetchLanUrl}
                    className="ml-auto flex items-center gap-1.5 text-xs text-accent hover:text-accent/80 transition-colors"
                  >
                    <RefreshCw className="w-3 h-3" />
                    Refresh
                  </button>
                </div>
              </div>

              {/* Right: QR code */}
              <div className="flex-shrink-0 p-6 flex flex-col items-center justify-center border-l border-border bg-[var(--bg-primary,#111)] max-md:border-l-0 max-md:border-t">
                <div className="bg-background p-3 rounded-xl shadow-lg shadow-black/20 max-md:max-w-full">
                  <QRCodeSVG
                    value={lanUrl}
                    size={176}
                    level="M"
                    bgColor="#ffffff"
                    fgColor="#000000"
                    className="max-md:w-full max-md:h-auto"
                  />
                </div>
                <p className="text-xs text-secondary mt-3 font-mono select-all">
                  {lanUrl}
                </p>
              </div>
            </div>
          ) : (
            /* No LAN / error state */
            <div className="p-6 flex items-center gap-5">
              <div className="flex-shrink-0 w-24 h-24 rounded-xl bg-accent/5 border border-accent/10 flex flex-col items-center justify-center gap-1.5">
                <Wifi className="w-8 h-8 text-accent/40" />
                <span className="text-[10px] text-accent/40 font-medium uppercase tracking-wider">
                  LAN
                </span>
              </div>

              <div className="flex-1 min-w-0">
                <h4 className="text-sm font-semibold text-primary mb-1">
                  No LAN network detected
                </h4>
                <p className="text-xs text-secondary leading-relaxed mb-3">
                  {lanError || 'Could not determine a local network address. Make sure your computer is connected to Wi-Fi or Ethernet.'}
                </p>

                <button
                  onClick={fetchLanUrl}
                  className="inline-flex items-center gap-2 bg-accent text-white text-xs font-semibold rounded-lg px-4 py-2 hover:bg-accent/90 transition-all duration-150 max-md:w-full max-md:min-h-[44px] max-md:justify-center"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  Retry
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  // ---- Relay mode: existing pairing code flow ----
  return (
    <div>
      {/* ---- Section header ---- */}
      <div className="flex items-center gap-2.5 mb-5">
        <div className="w-8 h-8 rounded-lg bg-accent/10 flex items-center justify-center">
          <Smartphone className="w-4 h-4 text-accent" />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-primary leading-tight">
            Linked Devices
          </h3>
          <p className="text-xs text-secondary">
            Manage your phone and other paired devices
          </p>
        </div>
      </div>

      {/* ---- QR Pairing Card (WhatsApp-style) ---- */}
      <div className="bg-sidebar border border-border rounded-xl overflow-hidden">
        {/* Top section: instructions + QR side by side */}
        {state === 'active' && code ? (
          <div className="flex max-md:flex-col">
            {/* Left: instructions */}
            <div className="flex-1 p-6 flex flex-col justify-center">
              <h4 className="text-base font-semibold text-primary mb-4">
                Pair your phone
              </h4>
              <ol className="space-y-3">
                {[
                  'Open your phone\u2019s camera or QR scanner',
                  'Point it at this QR code to scan',
                  'Confirm the connection on your phone',
                ].map((step, i) => (
                  <li key={i} className="flex items-start gap-3">
                    <span className="flex-shrink-0 w-5 h-5 rounded-full bg-accent/15 text-accent text-xs font-bold flex items-center justify-center mt-0.5">
                      {i + 1}
                    </span>
                    <span className="text-sm text-secondary leading-snug">
                      {step}
                    </span>
                  </li>
                ))}
              </ol>

              {/* Timer + refresh */}
              <div className="flex items-center gap-3 mt-5 pt-4 border-t border-border">
                <Clock className="w-3.5 h-3.5 text-secondary" />
                <span className="text-xs text-secondary tabular-nums">
                  Expires in{' '}
                  <span
                    className={
                      countdown < 60 ? 'text-error font-medium' : 'text-primary font-medium'
                    }
                  >
                    {minutes}:{seconds.toString().padStart(2, '0')}
                  </span>
                </span>
                <button
                  onClick={startPairing}
                  className="ml-auto flex items-center gap-1.5 text-xs text-accent hover:text-accent/80 transition-colors"
                >
                  <RefreshCw className="w-3 h-3" />
                  New code
                </button>
              </div>
            </div>

            {/* Right: QR code */}
            <div className="flex-shrink-0 p-6 flex flex-col items-center justify-center border-l border-border bg-[var(--bg-primary,#111)] max-md:border-l-0 max-md:border-t">
              <div className="bg-background p-3 rounded-xl shadow-lg shadow-black/20 max-md:max-w-full">
                <QRCodeSVG
                  value={qrValue}
                  size={176}
                  level="M"
                  bgColor="#ffffff"
                  fgColor="#000000"
                  className="max-md:w-full max-md:h-auto"
                />
              </div>
              <p className="text-xs text-secondary mt-3 font-mono tracking-[0.25em] select-all">
                {code}
              </p>
            </div>
          </div>
        ) : (
          /* Idle / loading / error state */
          <div className="p-6 flex items-center gap-5">
            {/* Illustration placeholder */}
            <div className="flex-shrink-0 w-24 h-24 rounded-xl bg-accent/5 border border-accent/10 flex flex-col items-center justify-center gap-1.5">
              <Scan className="w-8 h-8 text-accent/40" />
              <span className="text-[10px] text-accent/40 font-medium uppercase tracking-wider">
                Scan
              </span>
            </div>

            <div className="flex-1 min-w-0">
              <h4 className="text-sm font-semibold text-primary mb-1">
                Use Orbital on your phone
              </h4>
              <p className="text-xs text-secondary leading-relaxed mb-3">
                Pair your phone to monitor agents, approve actions, and send
                messages remotely. Your phone connects through a secure cloud
                relay.
              </p>

              {error && (
                <div className="text-xs text-error bg-error/5 border border-error/15 rounded-lg px-3 py-2 mb-3">
                  {error}
                </div>
              )}

              <button
                onClick={startPairing}
                disabled={state === 'loading'}
                className="inline-flex items-center gap-2 bg-accent text-white text-xs font-semibold rounded-lg px-4 py-2 hover:bg-accent/90 disabled:opacity-50 transition-all duration-150 max-md:w-full max-md:min-h-[44px] max-md:justify-center"
              >
                {state === 'loading' ? (
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <LinkIcon className="w-3.5 h-3.5" />
                )}
                {state === 'loading' ? 'Generating...' : 'Link a Device'}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Cancel button when QR is showing */}
      {state === 'active' && (
        <button
          onClick={resetPairing}
          className="mt-2 text-xs text-secondary hover:text-primary transition-colors"
        >
          Cancel pairing
        </button>
      )}

      {/* ---- Paired Devices List ---- */}
      <div className="mt-6">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-xs font-semibold text-secondary uppercase tracking-wider">
            Paired devices
          </h4>
          {devices.length > 0 && (
            <span className="text-xs text-secondary">
              {devices.length} device{devices.length !== 1 ? 's' : ''}
            </span>
          )}
        </div>

        {devices.length === 0 ? (
          <div className="bg-sidebar border border-border rounded-xl px-5 py-8 flex flex-col items-center text-center">
            <div className="w-10 h-10 rounded-full bg-secondary/5 flex items-center justify-center mb-3">
              <Laptop className="w-5 h-5 text-secondary/40" />
            </div>
            <p className="text-xs text-secondary">
              No devices paired yet. Link a device above to get started.
            </p>
          </div>
        ) : (
          <div className="space-y-1.5">
            {devices.map((device) => {
              const isRevoking = revokingId === device.phone_id;
              const pairedDate = device.paired_at
                ? new Date(device.paired_at)
                : null;

              return (
                <div
                  key={device.phone_id}
                  className={`group flex items-center gap-3 bg-sidebar border border-border rounded-xl px-4 py-3 transition-all duration-150 ${
                    isRevoking ? 'opacity-50' : 'hover:border-secondary/30'
                  }`}
                >
                  <div className="w-8 h-8 rounded-lg bg-accent/10 flex items-center justify-center flex-shrink-0">
                    <Smartphone className="w-4 h-4 text-accent" />
                  </div>

                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-primary font-medium truncate">
                      Phone &middot;{' '}
                      <span className="font-mono text-xs">
                        {device.phone_id.slice(0, 12)}
                      </span>
                    </p>
                    {pairedDate && (
                      <p className="text-xs text-secondary">
                        Linked{' '}
                        {pairedDate.toLocaleDateString(undefined, {
                          month: 'short',
                          day: 'numeric',
                          year: 'numeric',
                        })}
                        {' at '}
                        {pairedDate.toLocaleTimeString(undefined, {
                          hour: '2-digit',
                          minute: '2-digit',
                        })}
                      </p>
                    )}
                  </div>

                  <button
                    onClick={() => revokeDevice(device.phone_id)}
                    disabled={isRevoking}
                    className="flex items-center gap-1.5 text-xs text-secondary hover:text-error opacity-0 group-hover:opacity-100 max-md:opacity-100 transition-all duration-150 rounded-lg px-2.5 py-1.5 hover:bg-error/5 max-md:min-h-[44px]"
                    title="Unlink this device"
                  >
                    <Trash2 className="w-3 h-3" />
                    Unlink
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
