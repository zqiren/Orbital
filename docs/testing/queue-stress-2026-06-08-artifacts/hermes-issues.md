# Hermes Agent Memory/Persistence Feature Requests

## Research Summary

**Target**: Nous Research's Hermes Agent (https://github.com/NousResearch/hermes-agent)

**Why Hermes**: Among multiple projects named "Hermes" (e.g., Nous Research's Hermes language models), the Hermes Agent project is the most relevant to agent orchestration/memory because it's an autonomous AI agent framework with explicit session management, context compression, and state persistence features. The repository shows active development around memory management, session splitting, and goal state migration.

**Platform**: GitHub Issues & Discussions

---

## Potential Interviewees (8 Candidates)

### 1. kortylokai-web (Tier 1 - Most Engaged)
- **Issue**: #41478 - "Fix goal state drift across gateway context-compression session splits"
- **URL**: https://github.com/NousResearch/hermes-agent/issues/41478
- **Quote**: "When a session is compressed, Hermes ends the oversized parent session and continues in a new child session. Before this change, active /goal state could remain attached to the old parent session, meaning post-turn goal continuation/status could desynchronize from the live conversation."
- **Context**: Developer submitted a PR to fix goal state drift during session compression, showing deep engagement with memory/persistence issues.

### 2. Issue #41425 (Tier 1 - Active Feature Request)
- **URL**: https://github.com/NousResearch/hermes-agent/issues/41425
- **Topic**: Bounded memory/window size concerns
- **Context**: User raised concerns about bounded memory in the Hermes agent system.

### 3. Issue #32064 (Tier 1 - Memory Feature Request)
- **URL**: https://github.com/NousResearch/hermes-agent/issues/32064
- **Topic**: Bounded memory feature request
- **Context**: User requested improvements to bounded memory handling.

### 4. Memory Persistence Issues (Tier 2 - Multiple Requesters)
- **Search URL**: https://github.com/NousResearch/hermes-agent/issues?q=memory+OR+persist+OR+context+window+OR+state
- **Context**: Multiple users requesting memory persistence, context window improvements, and session state features.

### 5. Context Window Issues (Tier 2)
- **Search URL**: https://github.com/NousResearch/hermes-agent/issues?q=context+window+OR+conversation+history+OR+long+term+memory
- **Context**: Users requesting longer context windows and conversation history persistence.

### 6. Session State Persistence (Tier 2)
- **Search URL**: https://github.com/NousResearch/hermes-agent/issues?q=session+state+persistence+OR+persistent+state
- **Context**: Users requesting persistent session state across conversations.

### 7. Compression-Related Issues (Tier 3)
- **Related Issues**: #18749, #18427, #34035, #38231
- **URL**: https://github.com/NousResearch/hermes-agent/issues?q=is%3Aissue+memory+OR+context+window+OR+conversation+history+OR+long+term+memory+OR+persistent+state+OR+session+OR+state+persistence
- **Context**: Multiple issues related to preserving goals/data across compression session splits.

### 8. Feature Request Discussions (Tier 3)
- **Search URL**: https://github.com/NousResearch/hermes-agent/discussions
- **Context**: Community discussions on memory and persistence features.

---

## Key Findings

1. **Memory Management**: Active development around bounded memory and context window limits
2. **Session Compression**: Multiple PRs/fixes for preserving state across session splits
3. **Goal State**: Specific focus on goal state migration during compression
4. **Persistence**: Users requesting persistent conversation history and session state

## Sources

- https://github.com/NousResearch/hermes-agent/issues/41478
- https://github.com/NousResearch/hermes-agent/issues/41425
- https://github.com/NousResearch/hermes-agent/issues/32064
- https://github.com/NousResearch/hermes-agent/issues?q=memory+OR+persist+OR+context+window+OR+state
- https://github.com/NousResearch/hermes-agent/issues?q=context+window+OR+conversation+history+OR+long+term+memory
- https://github.com/NousResearch/hermes-agent/issues?q=session+state+persistence+OR+persistent+state