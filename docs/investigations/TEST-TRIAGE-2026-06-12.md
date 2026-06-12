# Test-suite triage — 2026-06-12 (baseline: commit ad8f645)

Goal: make red mean stop. Every pre-existing failure in the default suite run
was classified (A=environment, B=stale, C=real bug, D=flaky) with evidence,
then acted on per category. **Tests-only change — zero product-code edits.**

## Headline

- Default run before: 83 FAILED + 16 ERRORS (+1 collection error that aborted
  `pytest tests/` entirely, +2 platform files excluded as dangerous, +29
  live tests silently runnable-by-default).
- Default run after: **GREEN twice consecutively** (outputs at the bottom);
  every skip is explicit, marked, and reasoned.
- **Category C (real product bugs): none found.** Every failure resolved from
  a fixture/harness-only fix or had a citable deliberate behavior change.
  Per the seam-3 resume doc's rule 2, each fix was verified to resolve from
  the test side alone — two failures needed deeper root-causing before they
  cleared (test_api_key_refresh client-churn, BUG-005a consumer hook) and
  both turned out to be fixture drift, with the asserted product contract
  verified intact.

## Category counts

| Cat | Count | Meaning | Action taken |
|---|---|---|---|
| A | 24 failing items (+2 excluded hazard files, +29 live tests) | missing resource | resource markers + skip-by-default with reason; docs note with manual run commands |
| B | ~75 | stale vs deliberately-changed behavior | fixtures updated to current contract with commit citations; 6 tests proposed for DELETION (skipped pending review) |
| C | 0 | product bug | — |
| D | 0 currently failing | flaky | two documented candidates verified stable (see D-list) |

## Marker infrastructure added

Registered in `pyproject.toml`, enforced in `tests/conftest.py`, documented in
`docs/TESTING-markers.md` (exact manual-run command per group):

- `requires_windows` — auto-skips off-win32 (acl_teardown_revoke also gets a
  module-level guard because importing it touches `ctypes.windll`).
- `requires_keychain` — opt-in `ORBITAL_KEYCHAIN_TESTS=1`.
- `requires_relay` — auto-lifts when `relay/src/index.ts` exists.
- `live_sandbox` — opt-in `ORBITAL_LIVE_SANDBOX_TESTS=1`; covers the two files
  that run REAL sandbox-exec against ~/Desktop (documented EPERM hazard).
- `live_daemon` — opt-in `ORBITAL_LIVE_DAEMON_TESTS=1`. **Behavior change:**
  these previously ran by default if `claude`/`codex` were on PATH (they are on
  this machine) — a bare `pytest tests/` would have spawned real agents and
  spent live turns. Now skip-by-default.

## C-list (real bugs)

None. Near-misses that were investigated and cleared:

1. `test_api_key_refresh::test_hot_resume_no_op_when_key_unchanged` — client
   was rebuilt despite unchanged key, apparently contradicting the 36046b3
   contract ("no client churn on the unchanged path"). Root cause:
   `FakeSettingsStore` left `llm.provider` as a bare MagicMock attribute, so
   `_provider_config_changed()` saw MagicMock != "custom". Product logic
   verified correct with a complete fake.
2. `test_slot_enforcement::test_session_id_none_does_not_bypass_slot_guard` —
   verified the slot guard still raises "Slot held by session" under
   session_id=None once the mock project store provides a real name; no
   bypass exists.
3. BUG-005a smoke (`test_message_events_reach_consumer_before_send_completes`)
   — "events buffered instead of streamed" was the test's 2-arg on_message
   hook raising TypeError inside the consumer; streaming verified intact
   (sibling test `test_read_stream_yields_events_in_realtime` passes
   unchanged).

Production-code observations surfaced during triage (no code changed; for
follow-up):

- `SubAgentManager.stop_all` docstring still references the deleted
  `DEFAULT_SESSION_ID` (sub_agent_manager.py:1386) — stale doc.
- The seam-3 notes say resolve_sub_agent_approval is the ONE None-tolerant
  SubAgentManager method; `get_pending_sub_agent_approval`
  (sub_agent_manager.py:1293) is a second. Notes/doc out of date.
- `PUT /settings/api-key` maps any keyring failure to a 500; a 503 or a
  structured "keyring unavailable" response would let clients distinguish
  user error from environment.

## Proposed DELETION list (category B, skipped pending your review)

All six are skipped with reason `DELETION CANDIDATE: ...` so the suite is
green while you review; none were deleted.

1. `tests/integration/test_watchdog_full_path.py` (2 tests) — pins the 300s
   `_MAX_IDLE_POLLS` watchdog kill deleted in 54b7ee2 ("no fixed clock may
   kill"); superseded by `tests/regression/test_watchdog_stops_subagents.py`
   (which asserts the constant is GONE — the two files are mutually
   exclusive contracts).
2. `tests/integration/test_new_session_subagent_lifecycle.py` (2 tests) —
   asserts POST /new-session stops sub-agents; new_session is pure-create
   since 8083e35 ("touches no running session"); surviving contract covered
   by `test_new_session_pure_create.py` / `test_new_session_uuid_only.py`.
3. `tests/integration/test_terminate_paths.py::test_new_session_terminates_old_loop`
   — same 8083e35 retirement.
4. `tests/integration/test_terminate_no_leak.py::test_new_session_terminates_in_under_3s_without_leak`
   — same 8083e35 retirement.

## D-list (flaky)

No currently-failing test was flaky. The two candidates documented in
`ACTIVE-seam3-test-sweep-resume.md` ("pass in isolation, fail in full suite"):

- `tests/regression/test_reap_adapter_drop_ordering.py` — passed the full
  triage run AND 3/3 isolation runs. Its historical full-suite failures are
  consistent with the `test_acl_teardown_revoke.py` collection error aborting
  runs (now guarded); no `flaky` marker without reproduced evidence.
- `tests/platform/test_network_proxy.py::TestNetworkProxy::test_wildcard_matching`
  — passed full run + 3/3 isolation. Note: it opens real outbound CONNECTs
  (sub.example.com:443), so it is network-sensitive by design; if it
  reappears, `requires_network` is the right marker.

## Environment notes for whoever runs this next

- The 5 `test_e2e_acp.py` failures and 1 of the 2 `test_api_key_wiring.py`
  failures were artifacts of the *runner's* environment (non-activated venv;
  `AGENT_OS_API_KEY` exported per the project's own debugging notes), not
  pre-existing reds. Both are now environment-proof (sys.executable; contract
  tuple).
- Canonical headless invocation:
  `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring python -m pytest tests/ -q`

## Full classification table (the 83 FAILED from the baseline run)

| # | Test | Cat | Evidence | Action |
|---|------|-----|----------|--------|
| 1 | `integration/test_cancel_message_full_path.py::test_cancel_via_http_idle_loop` | B | same pattern over the HTTP route (433912a) | re-keyed; session_id in body/calls |
| 2 | `integration/test_new_layout_writers.py::test_session_new_writes_to_orbital_sessions` | B | 8083e35: JSONL materializes on first append, not Session.new() (contract in test_session_deferred_creation.py) | test now appends first, then asserts location |
| 3 | `integration/test_new_session_subagent_lifecycle.py::test_new_session_stops_subagents_via_http` | B | asserts POST /new-session stops sub-agents; retired by 8083e35 pure-create ("touches no running session"); covered by test_new_session_pure_create.py | both tests skipped as DELETION CANDIDATES |
| 4 | `integration/test_new_session_subagent_lifecycle.py::test_new_session_subagent_stop_all_timeout_does_not_block` | B | asserts POST /new-session stops sub-agents; retired by 8083e35 pure-create ("touches no running session"); covered by test_new_session_pure_create.py | both tests skipped as DELETION CANDIDATES |
| 5 | `integration/test_stop_removed.py::test_cancel_still_works_and_keeps_session` | B | (pid,"default") + body-less /cancel resolves no-holder -> "no_agent" (433912a) | re-keyed; session_id in request body |
| 6 | `integration/test_stop_resume_roundtrip.py::test_pause_does_not_swap_chat_lands_in_parked_session_resume_continues` | B | fake AgentManager predated dispatcher additions: current_holder_session_id (8083e35), is_onboarding_complete (a1cffaa), minted["session_id"], session_id/queue_state kwargs | fake surface updated to current dispatcher contract |
| 7 | `integration/test_stop_resume_roundtrip.py::test_pause_does_not_mint_chat_session_id_state_field_removed` | B | fake AgentManager predated dispatcher additions: current_holder_session_id (8083e35), is_onboarding_complete (a1cffaa), minted["session_id"], session_id/queue_state kwargs | fake surface updated to current dispatcher contract |
| 8 | `integration/test_stop_resume_roundtrip.py::test_resume_no_parked_attempt_just_kicks_main_loop` | B | fake AgentManager predated dispatcher additions: current_holder_session_id (8083e35), is_onboarding_complete (a1cffaa), minted["session_id"], session_id/queue_state kwargs | fake surface updated to current dispatcher contract |
| 9 | `integration/test_sub_agent_inheritance.py::test_full_inheritance_dispatch_flow` | B | 433912a hard-raise on bare _start_from_registry/stop; fakes also predated 006bcec/7724315 (resume_session_id/model ctor kwargs) | session_id threaded; fake transports accept **kwargs |
| 10 | `integration/test_sub_agent_inheritance.py::test_memory_md_persistence_across_dispatches` | B | 433912a hard-raise on bare _start_from_registry/stop; fakes also predated 006bcec/7724315 (resume_session_id/model ctor kwargs) | session_id threaded; fake transports accept **kwargs |
| 11 | `integration/test_sub_agent_inheritance.py::test_other_sub_agents_memory_visible` | B | 433912a hard-raise on bare _start_from_registry/stop; fakes also predated 006bcec/7724315 (resume_session_id/model ctor kwargs) | session_id threaded; fake transports accept **kwargs |
| 12 | `integration/test_sub_agent_inheritance.py::test_prompt_is_freshly_rendered_per_dispatch` | B | 433912a hard-raise on bare _start_from_registry/stop; fakes also predated 006bcec/7724315 (resume_session_id/model ctor kwargs) | session_id threaded; fake transports accept **kwargs |
| 13 | `integration/test_sub_agent_inheritance.py::test_pty_transport_does_not_create_memory_md` | B | 433912a hard-raise on bare _start_from_registry/stop; fakes also predated 006bcec/7724315 (resume_session_id/model ctor kwargs) | session_id threaded; fake transports accept **kwargs |
| 14 | `integration/test_sub_agent_inheritance.py::test_acp_transport_does_not_create_memory_md` | B | 433912a hard-raise on bare _start_from_registry/stop; fakes also predated 006bcec/7724315 (resume_session_id/model ctor kwargs) | session_id threaded; fake transports accept **kwargs |
| 15 | `integration/test_sub_agent_inheritance.py::test_sdk_transport_still_creates_memory_md` | B | 433912a hard-raise on bare _start_from_registry/stop; fakes also predated 006bcec/7724315 (resume_session_id/model ctor kwargs) | session_id threaded; fake transports accept **kwargs |
| 16 | `integration/test_terminate_no_leak.py::test_new_session_terminates_in_under_3s_without_leak` | B | same split: dispatcher/stop test fixed (433912a keying); new_session-terminates test retired by 8083e35 | fixed + DELETION CANDIDATE skip |
| 17 | `integration/test_terminate_no_leak.py::test_dispatcher_survives_stop_agent_and_is_drained_by_shutdown` | B | same split: dispatcher/stop test fixed (433912a keying); new_session-terminates test retired by 8083e35 | fixed + DELETION CANDIDATE skip |
| 18 | `integration/test_terminate_paths.py::test_stop_agent_terminates_via_loop` | B | stop_agent test: (pid,"default") + bare stop_agent (433912a). new_session test: asserts loop termination retired by 8083e35 pure-create | stop test fixed; new_session test skipped as DELETION CANDIDATE |
| 19 | `integration/test_terminate_paths.py::test_new_session_terminates_old_loop` | B | stop_agent test: (pid,"default") + bare stop_agent (433912a). new_session test: asserts loop termination retired by 8083e35 pure-create | stop test fixed; new_session test skipped as DELETION CANDIDATE |
| 20 | `integration/test_watchdog_full_path.py::test_watchdog_full_path_stops_wedged_subagents` | B | patches _MAX_IDLE_POLLS, DELETED in 54b7ee2 ("no fixed clock may kill"); superseded by test_watchdog_stops_subagents.py which asserts the constant is gone | skipped as DELETION CANDIDATE (review list) |
| 21 | `integration/test_watchdog_full_path.py::test_watchdog_full_path_no_stop_when_subagents_idle` | B | patches _MAX_IDLE_POLLS, DELETED in 54b7ee2 ("no fixed clock may kill"); superseded by test_watchdog_stops_subagents.py which asserts the constant is gone | skipped as DELETION CANDIDATE (review list) |
| 22 | `platform/test_consumer2_wiring.py::TestShellToolNullProviderIntegration::test_echo_with_null_provider` | A | helper defaults os_type="windows" -> powershell cmd; NullProvider is real-subprocess passthrough since initial release -> needs Windows | marked requires_windows |
| 23 | `platform/test_pty_reconciliation.py::TestPlatformProviderABC::test_windows_provider_run_process_accepts_use_pty` | A | imports Windows provider -> ctypes.windll at import time | marked requires_windows |
| 24 | `regression/test_api_key_refresh.py::test_hot_resume_uses_updated_key` | B | bare start_agent mints a fresh uuid (433912a) -> KeyError (pid,"default"); FakeSettingsStore also predated 36046b3 live-resolve (MagicMock provider field forced client rebuild) | explicit session_id; FakeSettingsStore stubs provider/sdk/fallback_models |
| 25 | `regression/test_api_key_refresh.py::test_hot_resume_no_op_when_key_unchanged` | B | bare start_agent mints a fresh uuid (433912a) -> KeyError (pid,"default"); FakeSettingsStore also predated 36046b3 live-resolve (MagicMock provider field forced client rebuild) | explicit session_id; FakeSettingsStore stubs provider/sdk/fallback_models |
| 26 | `regression/test_api_key_wiring.py::TestApiKeyEndpoints::test_get_api_key_status` | A/B | status test: tuple ("none","keyring","settings") never matched product contract environment|keychain|none(+settings) (credential_store.py:74). put test: 500s without functional OS keyring | status tuple fixed to contract (B); put test marked requires_keychain (A) |
| 27 | `regression/test_api_smoke_bugfixes.py::TestBug005b_IdleRaceAPI::test_create_project_start_subagent_check_status` | B | setdefault((pid,"default")) "for back-compat"; list_active hard-raises on None (433912a) | re-keyed; session_id passed |
| 28 | `regression/test_approval_denial_history.py::TestInjectMessageDuringApproval::test_inject_auto_denies_when_paused_for_approval` | B | (pid,"default") handles; bare inject/_on_loop_done are chat-route/no-op under 433912a | re-keyed; session_id passed |
| 29 | `regression/test_approval_denial_history.py::TestInjectMessageDuringApproval::test_inject_direct_appends_when_not_paused` | B | (pid,"default") handles; bare inject/_on_loop_done are chat-route/no-op under 433912a | re-keyed; session_id passed |
| 30 | `regression/test_approval_denial_history.py::TestOnLoopDoneApprovalBeforeQueueDrain::test_queued_messages_not_drained_when_paused_for_approval` | B | (pid,"default") handles; bare inject/_on_loop_done are chat-route/no-op under 433912a | re-keyed; session_id passed |
| 31 | `regression/test_approval_denial_history.py::TestOnLoopDoneApprovalBeforeQueueDrain::test_queued_messages_drained_after_approval_resolved` | B | (pid,"default") handles; bare inject/_on_loop_done are chat-route/no-op under 433912a | re-keyed; session_id passed |
| 32 | `regression/test_approval_pause_status.py::TestOnLoopDonePendingApproval::test_broadcasts_pending_approval_when_paused` | B | bare _on_loop_done/get_run_status against planted (pid,"default") handles (433912a) | re-keyed; session_id passed |
| 33 | `regression/test_approval_pause_status.py::TestOnLoopDonePendingApproval::test_broadcasts_idle_when_not_paused` | B | bare _on_loop_done/get_run_status against planted (pid,"default") handles (433912a) | re-keyed; session_id passed |
| 34 | `regression/test_background_send_exception.py::TestBackgroundSendStrongRef::test_background_send_task_strongly_referenced` | B | helper keyed (pid,"default"); bare send (433912a); 2-tuple transcript key | helper re-keyed; session_id passed |
| 35 | `regression/test_background_send_exception.py::TestBackgroundSendExceptionHandling::test_background_send_exception_surfaces_at_error_level` | B | helper keyed (pid,"default"); bare send (433912a); 2-tuple transcript key | helper re-keyed; session_id passed |
| 36 | `regression/test_background_send_exception.py::TestBackgroundSendExceptionHandling::test_cancelled_background_send_does_not_mark_broken` | B | helper keyed (pid,"default"); bare send (433912a); 2-tuple transcript key | helper re-keyed; session_id passed |
| 37 | `regression/test_cancel_from_approval.py::test_cancel_with_no_pending_approval_still_returns_idle` | B | (pid,"default") + bare cancel_message; idle path resolves "no_agent" (433912a) | re-keyed; session_id passed |
| 38 | `regression/test_cancel_message.py::test_cancel_message_idle_loop` | B | (pid,"default") handles; idle-cancel resolves no-holder -> "no_agent" vs asserted "idle"; broadcast payload pinned session_id="default" (433912a) | re-keyed; session_id passed; payload assert updated |
| 39 | `regression/test_cancel_message.py::test_cancel_message_idempotent` | B | (pid,"default") handles; idle-cancel resolves no-holder -> "no_agent" vs asserted "idle"; broadcast payload pinned session_id="default" (433912a) | re-keyed; session_id passed; payload assert updated |
| 40 | `regression/test_cancel_stops_subagents.py::test_waiting_state_cancel_stops_subagents` | B | (pid,"default") handles + bare-pid _idle_poll_tasks (SessionKey-keyed since multi-loop, agent_manager.py:108); stop_all asserts pinned "default" | re-keyed both dicts; session_id passed |
| 41 | `regression/test_cancel_stops_subagents.py::test_truly_idle_cancel_is_noop` | B | (pid,"default") handles + bare-pid _idle_poll_tasks (SessionKey-keyed since multi-loop, agent_manager.py:108); stop_all asserts pinned "default" | re-keyed both dicts; session_id passed |
| 42 | `regression/test_cancel_stops_subagents.py::test_stop_all_timeout_still_returns_cancelled` | B | (pid,"default") handles + bare-pid _idle_poll_tasks (SessionKey-keyed since multi-loop, agent_manager.py:108); stop_all asserts pinned "default" | re-keyed both dicts; session_id passed |
| 43 | `regression/test_cancel_stops_subagents.py::test_stop_agent_uses_shared_helper` | B | (pid,"default") handles + bare-pid _idle_poll_tasks (SessionKey-keyed since multi-loop, agent_manager.py:108); stop_all asserts pinned "default" | re-keyed both dicts; session_id passed |
| 44 | `regression/test_e2e_bugfixes_smoke.py::TestBug005a_SDKTransportStreaming::test_message_events_reach_consumer_before_send_completes` | B | ProcessManager calls on_message(msg, pid, session_id=...) (process_manager.py:233, 113ff70/006bcec); 2-arg hook raised TypeError in consumer | hook accepts session_id kwarg |
| 45 | `regression/test_first_message_nonce.py::TestQueuedMessageNonce::test_inject_queued_message_preserves_nonce` | B | (pid,"default") handle + bare inject/_on_loop_done (433912a) | re-keyed; session_id passed |
| 46 | `regression/test_first_message_nonce.py::TestOnLoopDoneNonce::test_on_loop_done_drain_includes_nonce` | B | (pid,"default") handle + bare inject/_on_loop_done (433912a) | re-keyed; session_id passed |
| 47 | `regression/test_inject_auto_denies_approval.py::TestInjectAutoDeniesApproval::test_inject_while_paused_auto_denies_and_delivers` | B | handles keyed (pid,"default"); inject_message(None) routes to chat session via _sid_inject, never the planted handle (433912a); asserted _start_loop(session_id="default") | re-keyed; session_id passed; assert updated |
| 48 | `regression/test_inject_auto_denies_approval.py::TestInjectAutoDeniesApproval::test_inject_while_paused_records_denial_in_history` | B | handles keyed (pid,"default"); inject_message(None) routes to chat session via _sid_inject, never the planted handle (433912a); asserted _start_loop(session_id="default") | re-keyed; session_id passed; assert updated |
| 49 | `regression/test_inject_auto_denies_approval.py::TestInjectAutoDeniesApproval::test_inject_while_paused_nonce_preserved` | B | handles keyed (pid,"default"); inject_message(None) routes to chat session via _sid_inject, never the planted handle (433912a); asserted _start_loop(session_id="default") | re-keyed; session_id passed; assert updated |
| 50 | `regression/test_inject_auto_denies_approval.py::TestInjectAutoDeniesApproval::test_inject_while_paused_broadcasts_approval_resolved` | B | handles keyed (pid,"default"); inject_message(None) routes to chat session via _sid_inject, never the planted handle (433912a); asserted _start_loop(session_id="default") | re-keyed; session_id passed; assert updated |
| 51 | `regression/test_last_terminal_event.py::test_stopped_event_records_last_terminal_event` | B | "default" sid + bare stop_agent (passthrough no-op) and bare inject (chat mint) (433912a) | re-keyed; session_id passed |
| 52 | `regression/test_last_terminal_event.py::test_terminal_event_cleared_on_next_inject` | B | "default" sid + bare stop_agent (passthrough no-op) and bare inject (chat mint) (433912a) | re-keyed; session_id passed |
| 53 | `regression/test_queue_during_wait.py::test_user_message_queued_during_waiting` | B | _idle_poll_tasks keyed by bare pid; SessionKey-keyed since multi-loop (agent_manager.py:108) | re-keyed to SessionKey |
| 54 | `regression/test_queued_message_drain_on_loop_done.py::test_on_loop_done_checks_queued_messages` | B | bare _on_loop_done(pid) = passthrough-None handle-miss no-op (433912a) | re-keyed; session_id passed |
| 55 | `regression/test_queued_message_drain_on_loop_done.py::test_on_loop_done_appends_queued_messages_to_session` | B | bare _on_loop_done(pid) = passthrough-None handle-miss no-op (433912a) | re-keyed; session_id passed |
| 56 | `regression/test_queued_message_drain_on_loop_done.py::test_on_loop_done_broadcasts_idle_when_no_queued_messages` | B | bare _on_loop_done(pid) = passthrough-None handle-miss no-op (433912a) | re-keyed; session_id passed |
| 57 | `regression/test_sdk_autonomy_filter.py::TestSubAgentManagerAutonomyWiring::test_update_sub_agent_autonomy` | B | adapters keyed (pid,"default"); update_sub_agent_autonomy hard-raises on None (433912a) | re-keyed; session_id passed |
| 58 | `regression/test_sdk_autonomy_filter.py::TestSubAgentManagerAutonomyWiring::test_update_sub_agent_autonomy_skips_non_sdk_transports` | B | adapters keyed (pid,"default"); update_sub_agent_autonomy hard-raises on None (433912a) | re-keyed; session_id passed |
| 59 | `regression/test_sdk_autonomy_filter.py::TestSubAgentManagerAutonomyWiring::test_update_sub_agent_autonomy_no_adapters` | B | adapters keyed (pid,"default"); update_sub_agent_autonomy hard-raises on None (433912a) | re-keyed; session_id passed |
| 60 | `regression/test_slot_enforcement.py::test_session_id_none_does_not_bypass_slot_guard` | B | start_agent(None) now mints uuid via _sanitize_project_name; MagicMock project store name -> TypeError before guard (433912a) | project store stubbed with real name; guard assert unchanged and passes |
| 61 | `regression/test_stop_from_approval.py::test_stop_from_pending_approval_pops_handle_and_broadcasts` | B | helper default sid="default"; bare stop_agent = passthrough no-op (433912a) | re-keyed; session_id passed |
| 62 | `regression/test_subagent_limits.py::TestBreadthLimit::test_max_breadth_blocks_spawn` | B | _register_adapter keyed (pid,"default"); bare start/stop (433912a) | helper re-keyed to SID; session_id passed |
| 63 | `regression/test_subagent_limits.py::TestBreadthLimit::test_breadth_freed_on_completion` | B | _register_adapter keyed (pid,"default"); bare start/stop (433912a) | helper re-keyed to SID; session_id passed |
| 64 | `regression/test_subagent_limits.py::TestBreadthLimit::test_breadth_under_limit_allows_spawn` | B | _register_adapter keyed (pid,"default"); bare start/stop (433912a) | helper re-keyed to SID; session_id passed |
| 65 | `regression/test_subagent_limits.py::TestBreadthLimit::test_breadth_is_per_project` | B | _register_adapter keyed (pid,"default"); bare start/stop (433912a) | helper re-keyed to SID; session_id passed |
| 66 | `regression/test_user_message_broadcast.py::TestInjectNoncePropagation::test_nonce_in_session_append` | B | same pattern (433912a) | re-keyed; session_id passed |
| 67 | `regression/test_user_message_broadcast.py::TestInjectNoncePropagation::test_no_nonce_omits_field` | B | same pattern (433912a) | re-keyed; session_id passed |
| 68 | `regression/test_v5_02_nonblocking_send.py::TestNonBlockingSend::test_send_returns_immediately` | B | 433912a retired the "default" sentinel; SubAgentManager.send hard-raises on session_id=None; _transcripts now (pid, sid, handle)-keyed | fixtures re-keyed to explicit SID; session_id passed to send() |
| 69 | `regression/test_v5_02_nonblocking_send.py::TestNonBlockingSend::test_send_result_includes_transcript_path` | B | 433912a retired the "default" sentinel; SubAgentManager.send hard-raises on session_id=None; _transcripts now (pid, sid, handle)-keyed | fixtures re-keyed to explicit SID; session_id passed to send() |
| 70 | `regression/test_v5_02_nonblocking_send.py::TestNonBlockingSend::test_send_error_for_unknown_agent` | B | 433912a retired the "default" sentinel; SubAgentManager.send hard-raises on session_id=None; _transcripts now (pid, sid, handle)-keyed | fixtures re-keyed to explicit SID; session_id passed to send() |
| 71 | `regression/test_v5_02_nonblocking_send.py::TestNonBlockingSend::test_send_transcript_unknown_when_no_transcript` | B | 433912a retired the "default" sentinel; SubAgentManager.send hard-raises on session_id=None; _transcripts now (pid, sid, handle)-keyed | fixtures re-keyed to explicit SID; session_id passed to send() |
| 72 | `regression/test_v5_02_nonblocking_send.py::TestNonBlockingSend::test_dispatch_uses_transport_dispatch_when_available` | B | 433912a retired the "default" sentinel; SubAgentManager.send hard-raises on session_id=None; _transcripts now (pid, sid, handle)-keyed | fixtures re-keyed to explicit SID; session_id passed to send() |
| 73 | `regression/test_v5_02_nonblocking_send.py::TestNonBlockingSend::test_dispatch_falls_back_to_background_send` | B | 433912a retired the "default" sentinel; SubAgentManager.send hard-raises on session_id=None; _transcripts now (pid, sid, handle)-keyed | fixtures re-keyed to explicit SID; session_id passed to send() |
| 74 | `regression/test_v5_02_nonblocking_send.py::TestNonBlockingSend::test_background_send_error_is_logged` | B | 433912a retired the "default" sentinel; SubAgentManager.send hard-raises on session_id=None; _transcripts now (pid, sid, handle)-keyed | fixtures re-keyed to explicit SID; session_id passed to send() |
| 75 | `regression/test_v5_05_prompt_sub_agent_section.py::TestSubAgentAwarenessSection::test_section_describes_nonblocking_model` | B | 16f9f42 reworded the section: "does NOT wait" -> "ENDS YOUR TURN"/"AUTOMATICALLY RESUMED" (push-based resume) | assertion updated to current wording |
| 76 | `regression/test_v5_integration.py::test_lifecycle_to_session_injection` | B | LifecycleObserver._inject passes session_id= (Piece 3, 2ce0182); fake_inject(pid, content) TypeError swallowed by except -> nothing injected | fake accepts session_id kwarg |
| 77 | `regression/test_v5_lifecycle_defer.py::TestInjectSystemMessage::test_inject_system_message_idle_appends_directly` | B | (pid,"default") handles; bare inject_system_message (433912a) | re-keyed; session_id passed |
| 78 | `regression/test_v5_lifecycle_defer.py::TestInjectSystemMessage::test_inject_system_message_running_defers` | B | (pid,"default") handles; bare inject_system_message (433912a) | re-keyed; session_id passed |
| 79 | `test_e2e_acp.py::TestACPTransportE2E::test_acp_roundtrip_with_dummy_agent` | A* | spawned bare "python" via PATH; only resolves with venv activated (harness fragility, product fine) | fixture uses sys.executable (kept running, not skipped) |
| 80 | `test_e2e_acp.py::TestACPTransportE2E::test_acp_session_persists` | A* | spawned bare "python" via PATH; only resolves with venv activated (harness fragility, product fine) | fixture uses sys.executable (kept running, not skipped) |
| 81 | `test_e2e_acp.py::TestACPTransportE2E::test_acp_permission_request` | A* | spawned bare "python" via PATH; only resolves with venv activated (harness fragility, product fine) | fixture uses sys.executable (kept running, not skipped) |
| 82 | `test_e2e_acp.py::TestACPTransportE2E::test_acp_stop_sends_shutdown` | A* | spawned bare "python" via PATH; only resolves with venv activated (harness fragility, product fine) | fixture uses sys.executable (kept running, not skipped) |
| 83 | `test_e2e_acp.py::TestACPTransportE2E::test_acp_transport_standalone` | A* | spawned bare "python" via PATH; only resolves with venv activated (harness fragility, product fine) | fixture uses sys.executable (kept running, not skipped) |

Grouped items beyond the 83:

| # | Group | Cat | Evidence | Action |
|---|-------|-----|----------|--------|
| 84 | `tests/e2e/relay/*` (16 setup ERRORS across 4 files) | A | conftest spawns `npx tsx src/index.ts` with cwd=`./relay` — directory does not exist in this checkout (relay deployed on Railway) | `requires_relay`, auto-lifts when source present |
| 85 | `tests/regression/test_acl_teardown_revoke.py` (collection error, aborted the whole run) | A | module import chain reaches `ctypes.windll` (no attribute off-Windows) | module-level skip + `requires_windows` |
| 86 | `tests/platform/test_e2e_agent_isolation.py` (file) | A | spawns REAL sandbox-exec probing actual `~/Desktop` (line ~121); EPERM-locked the dev tree historically | `live_sandbox`, opt-in env var |
| 87 | `tests/platform/test_macos_provider_integration.py` (file, incl. documented env-fail `test_portal_readonly`) | A | same live Seatbelt hazard | `live_sandbox`, opt-in env var |
| 88 | `live_daemon` tests (29 collected across 6 integration files) | A | spawn real claude/codex; cost live turns; previously only deselected by convention | skip-by-default, opt-in `ORBITAL_LIVE_DAEMON_TESTS=1` |

## Green gate (two consecutive runs)

Command (both runs):

```bash
PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring python -m pytest tests/ -q --timeout=300
```

Run 1:

```
3430 passed, 199 skipped, 3954 warnings in 207.31s (0:03:27)
```

Run 2:

```
3430 passed, 199 skipped, 3954 warnings in 206.04s (0:03:26)
```

Identical pass/skip counts across consecutive runs — stable.

