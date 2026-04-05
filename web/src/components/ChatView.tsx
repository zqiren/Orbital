// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useCallback, useEffect, useRef, useState } from 'react';
import { Send, Square, Loader2 } from 'lucide-react';
import { api, apiWithTotal } from '../config';
import { useWebSocket } from '../hooks/useWebSocket';
import { useAgent } from '../hooks/useAgent';
import { transformChatHistory } from '../utils/chatTransform';
import type { DisplayItem } from '../utils/chatTransform';

const CHAT_PAGE_SIZE = 50;
const REST_FALLBACK_DELAY_MS = 500;
const SLASH_COMMANDS = [
  { name: '/new', description: 'Start a fresh session' },
];
import type {
  AgentRunStatus,
  ChatMessage as ChatMessageType,
  StreamDeltaEvent,
  ActivityEvent,
  ApprovalRequestEvent,
  ApprovalResolvedEvent,
  SubAgentMessageEvent,
  UserMessageEvent,
  AgentNotifyEvent,
  WebSocketEvent,
  Project,
} from '../types';
import ChatMessage from './ChatMessage';
import StreamingMessage from './StreamingMessage';
import ActivityBlockComponent from './ActivityBlock';
import ApprovalCard from './ApprovalCard';
import CredentialCard from './CredentialCard';

interface ChatViewProps {
  projectId: string;
  project: Project;
  agentStatus: AgentRunStatus;
  statusTick?: number;
}

interface StreamState {
  text: string;
  source: string;
  isComplete: boolean;
}

interface PendingApproval {
  what: string;
  tool_name: string;
  tool_call_id: string;
  tool_args: Record<string, unknown>;
  recent_activity: ChatMessageType[];
  reasoning?: string;
  resolved?: 'approved' | 'denied';
}

interface RealtimeActivityBlock {
  activities: Array<{
    id: string;
    category: string;
    description: string;
    toolName: string;
    timestamp: string;
  }>;
  startTime: string;
  endTime: string;
}

export default function ChatView({ projectId, project, agentStatus, statusTick }: ChatViewProps) {
  const [items, setItems] = useState<DisplayItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [totalMessages, setTotalMessages] = useState(0);
  const [loadedOffset, setLoadedOffset] = useState(0);
  const [stream, setStream] = useState<StreamState | null>(null);
  const [approvals, setApprovals] = useState<Map<string, PendingApproval>>(new Map());
  const [realtimeBlock, setRealtimeBlock] = useState<RealtimeActivityBlock | null>(null);
  const [inputText, setInputText] = useState('');
  const [showMentionDropdown, setShowMentionDropdown] = useState(false);
  const [mentionFilter, setMentionFilter] = useState('');
  const [mentionAgents, setMentionAgents] = useState<Array<{slug: string; name: string}>>([]);
  const [selectedMentionIndex, setSelectedMentionIndex] = useState(0);
  const [subAgentLoading, setSubAgentLoading] = useState<string | null>(null);
  const [showThinking, setShowThinking] = useState(false);
  const [showCommandDropdown, setShowCommandDropdown] = useState(false);
  const [commandFilter, setCommandFilter] = useState('');
  const [selectedCommandIndex, setSelectedCommandIndex] = useState(0);
  const [injectError, setInjectError] = useState<string | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const localNoncesRef = useRef<Map<string, number>>(new Map());
  const wasRunningRef = useRef(false);
  const { on, off } = useWebSocket();
  const { injectMessage, startAgent, stopAgent, newSession } = useAgent();
  const autoStarted = useRef(false);

  /**
   * REST fallback: fetch the latest assistant message from the REST API.
   * Used when streaming deltas are missed (tunnel drop, late subscribe, etc.)
   * to recover the agent's response.
   */
  const fetchLatestMessage = useCallback(() => {
    setTimeout(() => {
      api<Array<{ role: string; content: string; timestamp?: string }>>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/chat?limit=1`,
      )
        .then((messages) => {
          if (!messages || messages.length === 0) return;
          const latest = messages[messages.length - 1];
          if (latest.role !== 'assistant' || !latest.content) return;
          const restText = latest.content;
          setItems((prevItems) => {
            // Check if the message is already present (by content match on last item)
            const last = prevItems[prevItems.length - 1];
            if (
              last &&
              last.type === 'agent_message' &&
              'content' in last &&
              last.content === restText
            ) {
              return prevItems;
            }
            // Also check if the last agent_message has shorter content (stream was truncated)
            if (
              last &&
              last.type === 'agent_message' &&
              'content' in last &&
              restText.length > last.content.length
            ) {
              const updated = [...prevItems];
              updated[prevItems.length - 1] = { ...last, content: restText };
              return updated;
            }
            // Message not present at all — insert before any trailing user messages
            // so recovered agent responses appear before follow-up questions
            const newMsg = {
              type: 'agent_message' as const,
              content: restText,
              source: 'assistant',
              timestamp: latest.timestamp ?? new Date().toISOString(),
            };
            let insertIdx = prevItems.length;
            while (insertIdx > 0 && prevItems[insertIdx - 1].type === 'user_message') {
              insertIdx--;
            }
            const updated = [...prevItems];
            updated.splice(insertIdx, 0, newMsg);
            return updated;
          });
        })
        .catch(() => {
          // REST fetch failed — best effort
        });
    }, REST_FALLBACK_DELAY_MS);
  }, [projectId]);

  const scrollToBottom = useCallback(() => {
    const el = scrollRef.current;
    if (el) {
      requestAnimationFrame(() => {
        el.scrollTop = el.scrollHeight;
      });
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadData() {
      setLoading(true);
      try {
        const [chatResult, agentsResult] = await Promise.allSettled([
          apiWithTotal<ChatMessageType[]>(
            `/api/v2/agents/${encodeURIComponent(projectId)}/chat?limit=${CHAT_PAGE_SIZE}`,
          ),
          api<Array<{slug: string; name: string; installed: boolean}>>('/api/v2/agents/available'),
        ]);
        if (cancelled) return;

        if (chatResult.status === 'fulfilled') {
          const { data: messages, total } = chatResult.value;
          const transformed = transformChatHistory(messages, project.workspace);
          setItems(transformed);
          setTotalMessages(total);
          setLoadedOffset(CHAT_PAGE_SIZE);
        } else {
          console.error('[ChatView] Failed to load chat history:', chatResult.reason);
          setItems([]);
        }

        if (agentsResult.status === 'fulfilled') {
          setMentionAgents(agentsResult.value.filter(a => a.slug !== 'built-in' && a.installed));
        } else {
          console.error('[ChatView] Failed to load agents:', agentsResult.reason);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadData();
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  const hasMore = totalMessages > loadedOffset;

  async function loadOlderMessages() {
    if (loadingMore || !hasMore) return;
    setLoadingMore(true);
    const el = scrollRef.current;
    const prevHeight = el?.scrollHeight ?? 0;
    try {
      const { data: messages } = await apiWithTotal<ChatMessageType[]>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/chat?limit=${CHAT_PAGE_SIZE}&offset=${loadedOffset}`,
      );
      if (messages.length === 0) return;
      const transformed = transformChatHistory(messages, project.workspace);
      setItems(prev => [...transformed, ...prev]);
      setLoadedOffset(prev => prev + CHAT_PAGE_SIZE);
      // Preserve scroll position after prepending
      requestAnimationFrame(() => {
        if (el) el.scrollTop = el.scrollHeight - prevHeight;
      });
    } catch (err) {
      console.error('[ChatView] Failed to load older messages:', err);
    } finally {
      setLoadingMore(false);
    }
  }

  // Auto-start agent on first open when no messages exist
  useEffect(() => {
    if (!autoStarted.current && !loading && items.length === 0) {
      autoStarted.current = true;
      startAgent(projectId).catch(console.error);
    }
  }, [loading, items.length, projectId, startAgent]);

  // Fix 3A: Flush realtimeBlock on terminal status (idle, error, stopped)
  // Fix 3C: Show "Thinking..." indicator when agent starts running
  useEffect(() => {
    if (agentStatus === 'idle' || agentStatus === 'error' || agentStatus === 'stopped') {
      flushRealtimeBlock();
      setShowThinking(false);
      // Catch-up fetch: if the agent was running, fetch the latest message
      // in case streaming deltas were entirely missed (tunnel down, etc.)
      if (wasRunningRef.current) {
        wasRunningRef.current = false;
        fetchLatestMessage();
      }
    } else if (agentStatus === 'running') {
      wasRunningRef.current = true;
      setShowThinking(true);
    } else if (agentStatus === 'pending_approval') {
      // Fetch pending approval via REST in case the WS event was missed
      api<{ pending: boolean; tool_call_id?: string; tool_name?: string; tool_args?: Record<string, unknown>; what?: string; recent_activity?: ChatMessageType[]; reasoning?: string }>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/pending-approval`,
      )
        .then((result) => {
          if (!result.pending || !result.tool_call_id) return;
          setApprovals((prev) => {
            // Skip if already present (WS event arrived first)
            if (prev.has(result.tool_call_id!)) return prev;
            const next = new Map(prev);
            next.set(result.tool_call_id!, {
              what: result.what ?? '',
              tool_name: result.tool_name ?? '',
              tool_call_id: result.tool_call_id!,
              tool_args: result.tool_args ?? {},
              recent_activity: result.recent_activity ?? [],
              reasoning: result.reasoning,
            });
            return next;
          });
          scrollToBottom();
        })
        .catch(() => {
          // REST fetch failed — best effort
        });
    } else if (agentStatus === 'new_session') {
      // WS event is the single source of truth for session swap
      wasRunningRef.current = false;
      setItems([{
        type: 'agent_notify' as const,
        title: 'New session started',
        body: 'Workspace memory preserved. The agent remembers your project.',
        urgency: 'low' as const,
        timestamp: new Date().toISOString(),
      }]);
      setStream(null);
      setApprovals(new Map());
      setRealtimeBlock(null);
      setShowThinking(false);
      setTotalMessages(0);
      setLoadedOffset(0);
    }
  }, [agentStatus, statusTick, projectId, fetchLatestMessage, scrollToBottom]);

  // On mount, always check for pending approvals via REST.
  // Handles the case where ChatView was unmounted (tab switch to files)
  // and remounted with a stale agentStatus that doesn't trigger the
  // status-change effect above.
  useEffect(() => {
    api<{ pending: boolean; tool_call_id?: string; tool_name?: string; tool_args?: Record<string, unknown>; what?: string; recent_activity?: ChatMessageType[]; reasoning?: string }>(
      `/api/v2/agents/${encodeURIComponent(projectId)}/pending-approval`,
    )
      .then((result) => {
        if (!result.pending || !result.tool_call_id) return;
        setApprovals((prev) => {
          if (prev.has(result.tool_call_id!)) return prev;
          const next = new Map(prev);
          next.set(result.tool_call_id!, {
            what: result.what ?? '',
            tool_name: result.tool_name ?? '',
            tool_call_id: result.tool_call_id!,
            tool_args: result.tool_args ?? {},
            recent_activity: result.recent_activity ?? [],
            reasoning: result.reasoning,
          });
          return next;
        });
        scrollToBottom();
      })
      .catch(() => {});
  }, [projectId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Fix 2B: Status poll fallback — if agent appears stuck as "running"
  // with no stream activity for 15 seconds, poll REST for actual status.
  const lastEventTimeRef = useRef<number>(Date.now());
  useEffect(() => {
    // Reset timer whenever we get any stream delta
    lastEventTimeRef.current = Date.now();
  }, [stream]);

  useEffect(() => {
    if (!agentStatus) return;
    const timer = setInterval(() => {
      api<{ project_id: string; status: string }>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/run-status`,
      )
        .then((result) => {
          if (result.status !== agentStatus) {
            window.dispatchEvent(
              new CustomEvent('agent-status-override', {
                detail: { project_id: projectId, status: result.status },
              }),
            );
          } else if (result.status === 'pending_approval') {
            // Same status but might be a NEW approval — fetch if needed
            api<{ pending: boolean; tool_call_id?: string; tool_name?: string; tool_args?: Record<string, unknown>; what?: string; recent_activity?: ChatMessageType[]; reasoning?: string }>(
              `/api/v2/agents/${encodeURIComponent(projectId)}/pending-approval`,
            ).then((pa) => {
              if (!pa.pending || !pa.tool_call_id) return;
              setApprovals((prev) => {
                if (prev.has(pa.tool_call_id!)) return prev;
                const next = new Map(prev);
                next.set(pa.tool_call_id!, {
                  what: pa.what ?? '',
                  tool_name: pa.tool_name ?? '',
                  tool_call_id: pa.tool_call_id!,
                  tool_args: pa.tool_args ?? {},
                  recent_activity: pa.recent_activity ?? [],
                  reasoning: pa.reasoning,
                });
                return next;
              });
              scrollToBottom();
            }).catch(() => {});
          }
        })
        .catch(() => {});
    }, 5_000);
    return () => clearInterval(timer);
  }, [agentStatus, projectId, scrollToBottom]);

  // Polling safety net: while the agent is running, poll /pending-approval
  // every 5 seconds so approvals surface even if the WS event was missed
  // (relay disconnect, transient drop). This is a fallback — the WebSocket
  // handler `handleApprovalRequest` remains the primary path.
  useEffect(() => {
    if (agentStatus !== 'running') return;

    const timer = setInterval(() => {
      api<{
        pending: boolean;
        tool_call_id?: string;
        tool_name?: string;
        tool_args?: Record<string, unknown>;
        what?: string;
        source?: string;
        recent_activity?: ChatMessageType[];
        reasoning?: string;
      }>(
        `/api/v2/agents/${encodeURIComponent(projectId)}/pending-approval`,
      )
        .then((result) => {
          if (!result.pending || !result.tool_call_id) return;
          setApprovals((prev) => {
            if (prev.has(result.tool_call_id!)) return prev; // dedup
            const next = new Map(prev);
            next.set(result.tool_call_id!, {
              what: result.what ?? '',
              tool_name: result.tool_name ?? '',
              tool_call_id: result.tool_call_id!,
              tool_args: result.tool_args ?? {},
              recent_activity: result.recent_activity ?? [],
              reasoning: result.reasoning,
            });
            return next;
          });
          scrollToBottom();
        })
        .catch(() => {});
    }, 5_000);

    return () => clearInterval(timer);
  }, [agentStatus, projectId, scrollToBottom]);

  useEffect(() => {
    function handleStreamDelta(event: WebSocketEvent) {
      const e = event as StreamDeltaEvent;
      if (e.project_id !== projectId) return;

      if (e.is_final) {
        setStream((prev) => {
          if (!prev) {
            // Stream state is null — intermediate deltas were missed.
            // Trigger REST fallback outside the updater (side-effect free).
            // fetchLatestMessage is called below after setStream.
            return null;
          }
          const finalText = prev.text + e.text;
          if (finalText.trim()) {
            setItems((prevItems) => {
              // Deduplicate: skip if last item has same content
              const last = prevItems[prevItems.length - 1];
              if (last && last.type === 'agent_message' && 'content' in last && last.content === finalText) {
                return prevItems;
              }
              return [
                ...prevItems,
                {
                  type: 'agent_message',
                  content: finalText,
                  source: e.source,
                  timestamp: new Date().toISOString(),
                },
              ];
            });
          }
          return null;
        });
        // Always trigger REST fallback on is_final — whether prev was null
        // (missed all deltas) or non-null (verify streamed text is complete).
        fetchLatestMessage();
        flushRealtimeBlock();
        scrollToBottom();
        return;
      }

      setShowThinking(false);
      setStream((prev) => ({
        text: (prev?.text ?? '') + e.text,
        source: e.source,
        isComplete: false,
      }));
      scrollToBottom();
    }

    function handleActivity(event: WebSocketEvent) {
      const e = event as ActivityEvent;
      if (e.project_id !== projectId) return;
      if (e.category === 'tool_result' || e.category === 'agent_output') return;

      const activity = {
        id: e.id,
        category: e.category,
        description: e.description,
        toolName: e.tool_name,
        timestamp: e.timestamp,
      };

      setRealtimeBlock((prev) => {
        if (!prev) {
          return {
            activities: [activity],
            startTime: e.timestamp,
            endTime: e.timestamp,
          };
        }
        return {
          ...prev,
          activities: [...prev.activities, activity],
          endTime: e.timestamp,
        };
      });
      scrollToBottom();
    }

    function handleApprovalRequest(event: WebSocketEvent) {
      const e = event as ApprovalRequestEvent;
      if (e.project_id !== projectId) return;

      setApprovals((prev) => {
        const next = new Map(prev);
        next.set(e.tool_call_id, {
          what: e.what,
          tool_name: e.tool_name,
          tool_call_id: e.tool_call_id,
          tool_args: e.tool_args,
          recent_activity: e.recent_activity,
          reasoning: e.reasoning,
        });
        return next;
      });

      if (document.hidden && 'Notification' in window && Notification.permission === 'granted') {
        new Notification(`Orbital: ${project.name} needs your approval`, {
          body: e.what,
        });
      } else if (
        document.hidden &&
        'Notification' in window &&
        Notification.permission === 'default'
      ) {
        Notification.requestPermission();
      }

      scrollToBottom();
    }

    function handleApprovalResolved(event: WebSocketEvent) {
      const e = event as ApprovalResolvedEvent;
      if (e.project_id !== projectId) return;

      setApprovals((prev) => {
        const next = new Map(prev);
        const existing = next.get(e.tool_call_id);
        if (existing) {
          next.set(e.tool_call_id, { ...existing, resolved: e.resolution });
        }
        return next;
      });
    }

    function handleSubAgentMessage(event: WebSocketEvent) {
      const e = event as SubAgentMessageEvent;
      if (e.project_id !== projectId) return;

      // Strip ANSI codes and filter empty / "(no response)" content
      const cleaned = (e.content ?? '').replace(/\x1b\[[0-9;]*m/g, '').trim();
      if (!cleaned || cleaned === '(no response)') return;

      setItems((prev) => [
        ...prev,
        {
          type: 'sub_agent_message',
          content: cleaned,
          source: e.source,
          timestamp: e.timestamp,
        },
      ]);
      scrollToBottom();
    }

    function handleUserMessage(event: WebSocketEvent) {
      const e = event as UserMessageEvent;
      if (e.project_id !== projectId) return;

      // Evict nonces older than 30s on each incoming message
      const now = Date.now();
      for (const [n, ts] of localNoncesRef.current) {
        if (ts > 0 && now - ts > 30_000) localNoncesRef.current.delete(n);
      }

      // Skip if this is our own message (nonce matches a local send)
      if (e.nonce && localNoncesRef.current.has(e.nonce)) {
        // Mark as received with timestamp instead of deleting, so relay
        // retries of the same event are still deduped within the TTL window.
        localNoncesRef.current.set(e.nonce, Date.now());
        return;
      }

      setItems((prev) => [
        ...prev,
        {
          type: 'user_message',
          content: e.content,
          timestamp: e.timestamp,
        },
      ]);
      scrollToBottom();
    }

    function handleAgentNotify(event: WebSocketEvent) {
      const e = event as AgentNotifyEvent;
      if (e.project_id !== projectId) return;

      setItems((prev) => [
        ...prev,
        {
          type: 'agent_notify' as const,
          title: e.title,
          body: e.body,
          urgency: e.urgency,
          timestamp: e.timestamp,
        },
      ]);

      // Browser notification for high/normal urgency
      if (e.urgency !== 'low' && 'Notification' in window && Notification.permission === 'granted') {
        new Notification(e.title, { body: e.body });
      }

      scrollToBottom();
    }

    on('chat.stream_delta', handleStreamDelta);
    on('agent.activity', handleActivity);
    on('approval.request', handleApprovalRequest);
    on('approval.resolved', handleApprovalResolved);
    on('chat.sub_agent_message', handleSubAgentMessage);
    on('chat.user_message', handleUserMessage);
    on('agent.notify', handleAgentNotify);

    return () => {
      off('chat.stream_delta', handleStreamDelta);
      off('agent.activity', handleActivity);
      off('approval.request', handleApprovalRequest);
      off('approval.resolved', handleApprovalResolved);
      off('chat.sub_agent_message', handleSubAgentMessage);
      off('chat.user_message', handleUserMessage);
      off('agent.notify', handleAgentNotify);
    };
  }, [projectId, project.name, on, off, scrollToBottom]);

  useEffect(() => {
    scrollToBottom();
  }, [items, scrollToBottom]);

  function flushRealtimeBlock() {
    setRealtimeBlock((prev) => {
      if (prev && prev.activities.length > 0) {
        const block: DisplayItem = {
          type: 'activity_block' as const,
          activities: prev.activities.map((a) => ({
            ...a,
            category: a.category as import('../types').ActivityCategory,
          })),
          startTime: prev.startTime,
          endTime: prev.endTime,
        };
        setItems((prevItems) => {
          // Insert before the last agent_message so the activity block
          // appears between the two text messages, not after both.
          let insertIdx = prevItems.length;
          for (let i = prevItems.length - 1; i >= 0; i--) {
            if (prevItems[i].type === 'agent_message') {
              insertIdx = i;
              break;
            }
          }
          const updated = [...prevItems];
          updated.splice(insertIdx, 0, block);
          return updated;
        });
      }
      return null;
    });
  }

  function adjustTextareaHeight() {
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = 'auto';
      ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
    }
  }

  const filteredAgents = mentionAgents.filter(a =>
    a.slug.toLowerCase().includes(mentionFilter) ||
    a.name.toLowerCase().includes(mentionFilter)
  );

  function handleInputChange(value: string) {
    setInputText(value);
    adjustTextareaHeight();

    // Check for /command trigger (only when input starts with /)
    if (value.startsWith('/')) {
      setShowCommandDropdown(true);
      setCommandFilter(value.slice(1).toLowerCase());
      setSelectedCommandIndex(0);
      // Hide @mention dropdown when in command mode
      setShowMentionDropdown(false);
      setMentionFilter('');
      return;
    }
    setShowCommandDropdown(false);
    setCommandFilter('');

    // Check for @mention trigger
    const atMatch = value.match(/@(\S*)$/);
    if (atMatch) {
      setShowMentionDropdown(true);
      setMentionFilter(atMatch[1].toLowerCase());
      setSelectedMentionIndex(0);
    } else {
      setShowMentionDropdown(false);
      setMentionFilter('');
    }
  }

  function selectMention(slug: string) {
    // Replace @partial with @slug
    const newText = inputText.replace(/@\S*$/, `@${slug} `);
    setInputText(newText);
    setShowMentionDropdown(false);
    setMentionFilter('');
    textareaRef.current?.focus();
  }

  const filteredCommands = SLASH_COMMANDS.filter(c =>
    c.name.toLowerCase().startsWith('/' + commandFilter)
  );

  async function executeNewSession() {
    setInputText('');
    // Echo the command so user sees it was received
    setItems((prev) => [...prev, {
      type: 'user_message' as const,
      content: '/new',
      timestamp: new Date().toISOString(),
    }]);
    setShowCommandDropdown(false);
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    try {
      const result = await newSession(projectId);
      if (result.status === 'no_active_session') {
        setItems((prev) => [...prev, {
          type: 'agent_notify' as const,
          title: 'No active session',
          body: 'Start the agent first, then use /new to reset the session.',
          urgency: 'high' as const,
          timestamp: new Date().toISOString(),
        }]);
      }
      // Otherwise, state clearing happens via WS new_session event
    } catch (err) {
      console.error('[ChatView] /new failed:', err);
    }
  }

  function selectCommand(name: string) {
    setShowCommandDropdown(false);
    setCommandFilter('');
    if (name === '/new') {
      executeNewSession();
      return;
    }
    setInputText(name);
    setTimeout(() => handleSend(), 0);
  }

  async function handleSend() {
    const text = inputText.trim();
    if (!text) return;

    // Slash command: /new
    if (text === '/new') {
      executeNewSession();
      return;
    }

    let target: string | undefined;
    let content = text;
    const atMatch = text.match(/^@([\w-]+)\s+([\s\S]*)/);
    if (atMatch) {
      target = atMatch[1];
      content = atMatch[2];
    }

    // Generate nonce so we can deduplicate the WS echo of our own message
    // crypto.randomUUID() requires secure context (HTTPS/localhost) — use fallback for LAN HTTP
    const nonce = typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    localNoncesRef.current.set(nonce, 0);

    setInputText('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    setItems((prev) => [
      ...prev,
      {
        type: 'user_message',
        content: content,
        timestamp: new Date().toISOString(),
        ...(target && { target }),
      },
    ]);

    if (target) setSubAgentLoading(target);
    setInjectError(null);
    try {
      await injectMessage(projectId, content, target, nonce);
    } catch {
      setInjectError('Failed to send message. Please try again.');
    } finally {
      setSubAgentLoading(null);
    }

    scrollToBottom();
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (showCommandDropdown) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedCommandIndex(i => Math.min(i + 1, filteredCommands.length - 1));
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedCommandIndex(i => Math.max(i - 1, 0));
        return;
      }
      if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault();
        if (filteredCommands.length > 0) {
          selectCommand(filteredCommands[selectedCommandIndex].name);
        }
        return;
      }
      if (e.key === 'Escape') {
        setShowCommandDropdown(false);
        return;
      }
    }
    if (showMentionDropdown) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedMentionIndex(i => Math.min(i + 1, filteredAgents.length - 1));
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedMentionIndex(i => Math.max(i - 1, 0));
        return;
      }
      if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault();
        if (filteredAgents.length > 0) {
          selectMention(filteredAgents[selectedMentionIndex].slug);
        }
        return;
      }
      if (e.key === 'Escape') {
        setShowMentionDropdown(false);
        return;
      }
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 max-md:pb-20">
        {loading && (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-12 bg-sidebar rounded-lg animate-pulse" />
            ))}
          </div>
        )}

        {!loading && hasMore && (
          <button
            onClick={loadOlderMessages}
            disabled={loadingMore}
            className="w-full py-2 text-xs text-secondary hover:text-primary transition-colors disabled:opacity-50"
          >
            {loadingMore ? 'Loading...' : `Load earlier messages (${totalMessages - loadedOffset} more)`}
          </button>
        )}

        {!loading && items.length === 0 && !stream && (
          <div className="text-secondary text-sm text-center mt-12">
            {autoStarted.current
              ? 'Agent is starting...'
              : 'No messages yet. Send a message to get started.'}
          </div>
        )}

        {!loading &&
          items.map((item, index) => {
            const historical = 'isHistorical' in item && item.isHistorical;
            let rendered: React.ReactNode = null;

            if (item.type === 'user_message') {
              rendered = <ChatMessage key={`msg-${index}`} message={item} />;
            } else if (item.type === 'agent_message' || item.type === 'sub_agent_message') {
              rendered = <ChatMessage key={`msg-${index}`} message={item} />;
            } else if (item.type === 'session_separator') {
              return (
                <div key={`sep-${index}`} className="flex items-center gap-3 my-4 px-2 opacity-50">
                  <div className="flex-1 border-t border-border" />
                  <span className="text-xs text-secondary whitespace-nowrap">
                    Previous session
                  </span>
                  <div className="flex-1 border-t border-border" />
                </div>
              );
            } else if (item.type === 'activity_block') {
              rendered = (
                <ActivityBlockComponent
                  key={`act-${index}`}
                  activities={item.activities}
                  startTime={item.startTime}
                  endTime={item.endTime}
                />
              );
            } else if (item.type === 'agent_notify') {
              const urgencyColor = item.urgency === 'high' ? 'border-error/40 bg-error/5' : 'border-accent/30 bg-accent/5';
              rendered = (
                <div key={`notify-${index}`} className={`mb-3 rounded-lg border ${urgencyColor} px-4 py-3`}>
                  <p className="text-sm font-medium text-primary">{item.title}</p>
                  <p className="text-sm text-secondary mt-1">{item.body}</p>
                </div>
              );
            } else if (item.type === 'approval_card') {
              const resolved = approvals.get(item.tool_call_id)?.resolved ?? item.resolved;
              rendered = item.tool_name === 'request_credential' ? (
                <CredentialCard
                  key={`cred-${item.tool_call_id}`}
                  credential={{
                    tool_call_id: item.tool_call_id,
                    name: item.tool_args?.name as string ?? '',
                    domain: item.tool_args?.domain as string ?? '',
                    fields: (item.tool_args?.fields as string[]) ?? [],
                    reason: item.tool_args?.reason as string ?? '',
                    resolved: !!resolved,
                  }}
                  projectId={projectId}
                  onResolve={(toolCallId: string) => {
                    setApprovals((prev) => {
                      const next = new Map(prev);
                      const existing = next.get(toolCallId);
                      if (existing) {
                        next.set(toolCallId, { ...existing, resolved: 'approved' });
                      }
                      return next;
                    });
                  }}
                />
              ) : (
                <ApprovalCard
                  key={`apr-${item.tool_call_id}`}
                  approval={item}
                  projectId={projectId}
                  resolved={resolved}
                  onResolve={(toolCallId: string, resolution: 'approved' | 'denied') => {
                    setApprovals((prev) => {
                      const next = new Map(prev);
                      const existing = next.get(toolCallId);
                      if (existing) {
                        next.set(toolCallId, { ...existing, resolved: resolution });
                      }
                      return next;
                    });
                  }}
                />
              );
            }

            if (!rendered) return null;

            return historical ? (
              <div key={`hist-${index}`} className="opacity-50">
                {rendered}
              </div>
            ) : (
              rendered
            );
          })}

        {!loading && realtimeBlock && realtimeBlock.activities.length > 0 && (
          <ActivityBlockComponent
            activities={realtimeBlock.activities.map((a) => ({
              ...a,
              category: a.category as import('../types').ActivityCategory,
            }))}
            startTime={realtimeBlock.startTime}
            endTime={realtimeBlock.endTime}
          />
        )}

        {!loading &&
          Array.from(approvals.values())
            .filter(
              (a) =>
                !a.resolved &&
                !items.some(
                  (i) => i.type === 'approval_card' && i.tool_call_id === a.tool_call_id,
                ),
            )
            .map((a) =>
              a.tool_name === 'request_credential' ? (
                <CredentialCard
                  key={`rt-cred-${a.tool_call_id}`}
                  credential={{
                    tool_call_id: a.tool_call_id,
                    name: a.tool_args?.name as string ?? '',
                    domain: a.tool_args?.domain as string ?? '',
                    fields: (a.tool_args?.fields as string[]) ?? [],
                    reason: a.tool_args?.reason as string ?? '',
                  }}
                  projectId={projectId}
                  onResolve={(toolCallId: string) => {
                    setApprovals((prev) => {
                      const next = new Map(prev);
                      const existing = next.get(toolCallId);
                      if (existing) {
                        next.set(toolCallId, { ...existing, resolved: 'approved' });
                      }
                      return next;
                    });
                  }}
                />
              ) : (
                <ApprovalCard
                  key={`rt-apr-${a.tool_call_id}`}
                  approval={a}
                  projectId={projectId}
                  resolved={a.resolved}
                  onResolve={(toolCallId: string, resolution: 'approved' | 'denied') => {
                    setApprovals((prev) => {
                      const next = new Map(prev);
                      const existing = next.get(toolCallId);
                      if (existing) {
                        next.set(toolCallId, { ...existing, resolved: resolution });
                      }
                      return next;
                    });
                  }}
                />
              ),
            )}

        {subAgentLoading && (
          <div className="flex justify-start mb-3">
            <div className="max-w-[75%] max-md:max-w-[85%]">
              <div className="flex items-center gap-2 mb-0.5">
                <span className="text-sm font-medium text-secondary">{subAgentLoading}</span>
              </div>
              <div className="bg-background border border-border rounded-lg px-4 py-2 text-sm">
                <span className="inline-flex gap-1 text-secondary">
                  <span className="animate-pulse">●</span>
                  <span className="animate-pulse" style={{animationDelay: '0.2s'}}>●</span>
                  <span className="animate-pulse" style={{animationDelay: '0.4s'}}>●</span>
                </span>
              </div>
            </div>
          </div>
        )}

        {showThinking && !stream && !subAgentLoading && (
          <div className="flex items-center gap-2 px-2 py-1 text-secondary text-sm">
            <Loader2 size={14} className="animate-spin" />
            <span>Thinking...</span>
          </div>
        )}

        {stream && (
          <StreamingMessage
            text={stream.text}
            source={stream.source}
            isComplete={stream.isComplete}
          />
        )}
      </div>

      {injectError && (
        <div className="shrink-0 px-4 py-1">
          <p className="text-xs text-error">{injectError}</p>
        </div>
      )}

      <div className="shrink-0 px-4 pb-4 pt-2 max-md:fixed max-md:bottom-0 max-md:left-0 max-md:right-0 max-md:bg-background max-md:z-[60] max-md:pb-[env(safe-area-inset-bottom,12px)]">
        <div className="relative flex items-center gap-2 bg-background border border-border rounded-lg shadow-lg px-3 py-2">
          {showCommandDropdown && filteredCommands.length > 0 && (
            <div className="absolute bottom-full left-0 mb-1 w-64 bg-zinc-800 border border-zinc-700 rounded-lg shadow-lg overflow-hidden z-50">
              {filteredCommands.map((cmd, i) => (
                <button
                  key={cmd.name}
                  className={`w-full text-left px-3 py-2 text-sm hover:bg-zinc-700 max-md:min-h-[44px] ${
                    i === selectedCommandIndex ? 'bg-zinc-700' : ''
                  }`}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    selectCommand(cmd.name);
                  }}
                >
                  <span className="font-medium text-zinc-200">{cmd.name}</span>
                  <span className="ml-2 text-zinc-500">{cmd.description}</span>
                </button>
              ))}
            </div>
          )}
          {showMentionDropdown && (
            <div className="absolute bottom-full left-0 mb-1 w-64 bg-zinc-800 border border-zinc-700 rounded-lg shadow-lg overflow-hidden z-50">
              {filteredAgents.length === 0 ? (
                <div className="px-3 py-2 text-sm text-zinc-500">No agents available</div>
              ) : (
                filteredAgents.map((agent, i) => (
                  <button
                    key={agent.slug}
                    className={`w-full text-left px-3 py-2 text-sm hover:bg-zinc-700 max-md:min-h-[44px] ${
                      i === selectedMentionIndex ? 'bg-zinc-700' : ''
                    }`}
                    onMouseDown={(e) => {
                      e.preventDefault();
                      selectMention(agent.slug);
                    }}
                  >
                    <span className="font-medium text-zinc-200">@{agent.slug}</span>
                    <span className="ml-2 text-zinc-500">{agent.name}</span>
                  </button>
                ))
              )}
            </div>
          )}
          <textarea
            ref={textareaRef}
            value={inputText}
            onChange={(e) => handleInputChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Send a message..."
            rows={1}
            className="flex-1 resize-none text-sm max-md:text-base bg-transparent focus:outline-none leading-relaxed"
          />
          {(agentStatus === 'running' || agentStatus === 'waiting') ? (
            <>
              <button
                type="button"
                onClick={() => stopAgent(projectId)}
                onTouchEnd={(e) => { e.preventDefault(); stopAgent(projectId); }}
                className="shrink-0 p-1.5 rounded-lg transition-colors duration-150 cursor-pointer text-red-500 hover:bg-red-500/10 max-md:min-h-[44px] max-md:min-w-[44px] max-md:flex max-md:items-center max-md:justify-center"
              >
                <Square size={18} />
              </button>
              <button
                type="button"
                onClick={handleSend}
                onTouchEnd={(e) => { e.preventDefault(); handleSend(); }}
                disabled={!inputText.trim()}
                className={`shrink-0 px-2.5 py-1 rounded-md text-xs font-semibold tracking-wide transition-colors duration-150 max-md:min-h-[44px] max-md:flex max-md:items-center max-md:justify-center ${
                  inputText.trim()
                    ? 'bg-accent text-white hover:bg-accent/85 cursor-pointer'
                    : 'bg-secondary/20 text-secondary/40 cursor-default'
                }`}
              >
                Queue
              </button>
            </>
          ) : (
            <button
              type="button"
              onClick={handleSend}
              onTouchEnd={(e) => { e.preventDefault(); handleSend(); }}
              aria-disabled={!inputText.trim()}
              className={`shrink-0 p-1.5 rounded-lg transition-colors duration-150 cursor-pointer max-md:min-h-[44px] max-md:min-w-[44px] max-md:flex max-md:items-center max-md:justify-center ${
                inputText.trim()
                  ? 'text-accent hover:bg-accent/10'
                  : 'text-secondary opacity-40'
              }`}
            >
              <Send size={18} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
