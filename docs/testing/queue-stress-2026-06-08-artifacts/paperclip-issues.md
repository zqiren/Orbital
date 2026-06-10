# Paperclip AI — Memory/Persistence Feature Requests

## Verification
**Product**: Paperclip AI — AI agent orchestration platform
**Repo**: https://github.com/paperclipai/paperclip (69.5k stars, 12.9k forks)
**Status**: CONFIRMED — Real AI agent product

---

## Memory-Related GitHub Issues

### Issue #460: feat: Agent Persistent Memory Store
- **URL**: https://github.com/paperclipai/paperclip/issues/460
- **Author**: mvanhorn
- **Status**: Open (primary tracking issue)
- **Summary**: Request for database-backed persistent memory store for agents

> "A database-backed memory store where agents can save and retrieve context across sessions."

---

### Issue #3326: feat: Built-in agent learning journal — file-based institutional memory
- **URL**: https://github.com/paperclipai/paperclip/issues/3326
- **Author**: rudyjellis
- **Status**: Open (detailed feature request)
- **Summary**: File-based learning journal with auto-injection into agent instructions

**Key Quote** (rudyjellis):
> "Agents lose all learned context between runs. This is tracked in #460, #764, and #228, but none of those have shipped yet. Meanwhile, production users are hitting this daily."

**Workaround Described**:
> "We added a LEARNINGS.md file to the agent's managed instructions bundle... The agent reads LEARNINGS.md from the instructions bundle every run, checks it before making decisions, and appends new entries when it learns something."

**Implementation Proposal**:
1. Auto-generate LEARNINGS.md in managed instructions bundle
2. Add "learnings" directive to default templates
3. Board UI: Learning journal viewer
4. Optional: Cross-agent learning sharing

**Update from Production** (rudyjellis):
> "The CEO agent had para-memory-files active and was writing Layer 2 daily notes — but only Layer 2 was being used... Daily notes were bloated and repetitive. 42KB in a single day."

**3-Layer Pattern Proposed**:
| Layer | File | Auto-injected? | Purpose |
|-------|------|---------------|---------|
| LEARNINGS.md | Instructions bundle | ✅ Yes | Curated decisions & facts |
| PARA knowledge graph | Workspace life/ | ❌ On-demand | Entity-level facts |
| PARA daily notes | Workspace memory/ | ❌ On-demand | Raw timeline |

---

### Issue #7706: State persistence bug
- **URL**: https://github.com/paperclipai/paperclip/issues/7706
- **Author**: (see issue)
- **Summary**: Bug report about state persistence

---

### Related Issues
- Issue #764: Cross-agent long-term memory store
- Issue #228: Decision & Architecture Documentation System
- Issue #373: "PROPOSAL: Application logic could save billions of wasted tokens?" (token savings from not re-learning known facts)

---

## Potential Interviewees

| Tier | Handle | Relevance | Issue |
|------|--------|-----------|-------|
| Tier 1 | **mvanhorn** | Proposed Agent Persistent Memory Store feature | #460 |
| Tier 1 | **rudyjellis** | Detailed learning journal proposal with production experience | #3326 |
| Tier 2 | **Omee11** | Commented on #3326, discussed consolidation | #3326 |

---

## Summary
Paperclip AI is a well-established AI agent orchestration platform (69.5k stars) with active feature requests for persistent memory:
- **mvanhorn** proposed the original database-backed memory store idea (#460)
- **rudyjellis** contributed detailed production-use case and file-based workaround (#3326)
- Multiple related issues track different aspects (cross-agent memory, documentation system, token savings)

All sources are real GitHub issues with verified user handles.