Scenario 1: "Delegate and continue working"
The core v5 promise — user asks management agent to delegate, then keeps working with it while the sub-agent runs.
Turn 1: User → "Ask Claude Code to refactor the auth module to use JWT tokens"
  Expected: Management agent responds quickly (within seconds, not minutes)
            saying it delegated the task. Status shows "idle" for management 
            agent, "running" for claude-code.
            Management session contains: tool_call(agent_message) + tool_result("Message sent...")

Turn 2: User → "While we wait, what's in the README?"
  Expected: Management agent reads README.md and responds with summary.
            This proves it's not blocked by Claude Code.
            Claude Code is still running in the background.

Turn 3: [Claude Code finishes]
  Expected: Management session receives a system message:
            "[Sub-agent] claude-code completed. Summary: ..."
            Management agent loop wakes up (if idle) or queues (if busy).
            Chat timeline shows Claude Code's output interspersed by timestamp 
            with management agent's README discussion.

Turn 4: User → "What did Claude Code change?"
  Expected: Management agent sees the completion system message in its context,
            knows Claude Code worked on auth, and either reads the workspace 
            files or the transcript to answer. It should NOT say "I don't know 
            about any Claude Code activity."

Scenario 2: "Direct @mention then ask management agent about it"
User talks to sub-agent directly, then pivots to management agent expecting it to know what happened.
Turn 1: User → "@claude-code analyze this codebase and list all security issues"
  Expected: Message routes to Claude Code directly.
            Management session receives trace: "[Sub-agent] User sent @claude-code: 'analyze this codebase...'"
            Claude Code output streams to chat in real-time.

Turn 2: [Claude Code responds with security analysis]
  Expected: Management session receives: "[Sub-agent] claude-code completed. Summary: ..."
            Chat timeline shows Claude Code's full output.

Turn 3: User → "Summarize what Claude Code just found and prioritize the top 3 issues"
  Expected: This goes to management agent (no @mention).
            Management agent sees the lifecycle traces in its session.
            It knows Claude Code analyzed security issues.
            It can either read the transcript file (path is in the trace) 
            or read the workspace files Claude Code may have created.
            It should NOT say "I'm not aware of any security analysis."

Scenario 3: "Claude Code needs permission mid-task"
Sub-agent hits a permission gate while the management agent is available for other work.
Turn 1: User → "Ask Claude Code to fix the failing tests"
  Expected: Management agent dispatches, responds with delegation acknowledgement.

Turn 2: [Claude Code reads test files, identifies fix, requests permission to edit test_auth.py]
  Expected: Approval card appears in user's chat for "claude-code wants to edit test_auth.py"
            Management agent is NOT involved — it doesn't wake up to relay the permission.
            Management agent status stays idle.
            Claude Code status shows "waiting for approval" or similar.

Turn 3: User approves the edit
  Expected: Claude Code continues working.
            No management agent involvement.

Turn 4: User → "How's the fix going?" (no @mention — goes to management agent)
  Expected: Management agent checks status via its awareness layers.
            Should report that Claude Code is still working (if still running) 
            or that it completed (if done between turn 3 and 4).

Scenario 4: "Claude Code errors out"
Sub-agent fails and the user needs the management agent to handle recovery.
Turn 1: User → "Ask Claude Code to refactor the entire database layer"
  Expected: Management agent dispatches, responds quickly.

Turn 2: [Claude Code runs for a while, then hits context window limit and errors]
  Expected: Management session receives: "[Sub-agent] claude-code stopped with error: context window exceeded. Task was incomplete. Transcript: ..."
            Management agent loop wakes up.
            Chat shows Claude Code's partial output up to the error.

Turn 3: Management agent (proactively or on user prompt) → reports the error
  Expected: "Claude Code ran into a context limit while refactoring the database layer. 
            It made partial progress — I can check what was completed."
            Management agent can read the transcript or workspace to see what 
            Claude Code accomplished before failing.

Turn 4: User → "What files did it manage to change before crashing?"
  Expected: Management agent reads workspace (git diff or file timestamps) or 
            reads the transcript, reports the partial results.

Scenario 5: "Multiple sub-agents on the same project"
User dispatches two different sub-agents and the management agent tracks both.
Turn 1: User → "Start Claude Code on the backend refactoring and start Aider on the frontend CSS fixes"
  Expected: Management agent makes two agent_message(send) calls.
            Both return immediately.
            Management session has two tool_call/tool_result pairs.
            Both sub-agents start independently.

Turn 2: User → "What's everyone working on?"
  Expected: Management agent calls agent_message(status) for both agents 
            or reads the prompt context section showing both as "running."
            Reports: "Claude Code is working on backend refactoring, 
            Aider is working on CSS fixes."

Turn 3: [Aider finishes first]
  Expected: Management session receives: "[Sub-agent] aider completed. Summary: Fixed 4 CSS issues..."
            Chat timeline shows Aider's output.
            Claude Code still running.

Turn 4: [Claude Code finishes]
  Expected: Management session receives: "[Sub-agent] claude-code completed. Summary: Refactored 3 modules..."
            Chat timeline shows Claude Code's output.

Turn 5: User → "Give me a summary of everything that was done"
  Expected: Management agent sees both completion messages in its session.
            Synthesizes a unified summary from both sub-agents' work.

Scenario 6: "Page refresh preserves everything"
The user refreshes the browser or reconnects from phone mid-session.
Turn 1: User → "@claude-code list all TODO comments in the codebase"
Turn 2: [Claude Code responds with a list of 20 TODOs]
Turn 3: User → "Prioritize the top 5"
  Expected: Management agent responds with prioritized list.

Turn 4: [User refreshes the page]
  Expected: GET /chat returns the complete interleaved timeline:
            - User's @mention message
            - Claude Code's TODO list (from sub-agent transcript)
            - User's "prioritize" message
            - Management agent's prioritized response
            All in correct chronological order with correct source labels.
            No messages missing. No duplicates.

Turn 5: User sends a new message after refresh
  Expected: Everything works normally. Management agent has its session 
            intact. Sub-agent transcripts are readable from disk.

Scenario 7: "Rapid-fire dispatch — user sends multiple tasks quickly"
User doesn't wait for acknowledgement between dispatches.
Turn 1: User → "Ask Claude Code to fix the auth bug"
Turn 2: User → "Also ask Claude Code to update the API docs"  (sent before management agent responds to Turn 1)
  Expected: Management agent processes Turn 1 first (dispatches to Claude Code).
            Turn 2 queues and is processed in the next loop iteration.
            Claude Code receives both messages in its input queue.
            Claude Code processes them sequentially (one at a time).
            Management session has two clean tool_call/tool_result pairs.
            No interleaving, no crash.

Turn 3: [Claude Code finishes first task]
  Expected: Completion notification for "fix auth bug."
            Claude Code starts on "update API docs" from its queue.

Turn 4: [Claude Code finishes second task]
  Expected: Completion notification for "update API docs."

What to check at every turn
For each scenario, the test should verify at each turn:

Management session JSONL — no role=agent messages, clean tool_call/tool_result pairing, system messages present where expected
Sub-agent transcript — output captured, entries have correct source and chunk_type
GET /chat response — merged timeline, correct chronological order, correct source attribution
WebSocket events — chat.sub_agent_message for real-time sub-agent output, sub_agent.completed/sub_agent.error for lifecycle events
Status — GET /run-status or agent_message(status) reflects actual state at each point