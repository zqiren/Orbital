# Interviewee Candidates — AI Agent Memory/Persistence Features

## Ranked Table of Candidates

| Rank | Handle | Project | Evidence URL | 1-Line Quote | Why They Fit | Suggested Outreach |
|------|--------|---------|--------------|--------------|--------------|---------------------|
| 1 | **rudyjellis** | Paperclip AI | https://github.com/paperclipai/paperclip/issues/3326 | "Agents lose all learned context between runs. This is tracked in #460, #764, and #228, but none of those have shipped yet." | Proposed 3-layer memory system with production experience; detailed file-based workaround in active use | Lead with their LEARNINGS.md workaround — ask what they'd want in an ideal solution |
| 2 | **talshebek** | OpenAI Codex | https://github.com/orgs/OpenAI/discussions/117 | (See GitHub discussion) | Active in Codex memory discussions, detailed feedback | Ask about their ideal memory workflow with Codex |
| 3 | **Li-xiaofeng** | Cursor IDE | https://forum.cursor.com/t/enhanced-chat-history-search-and-navigation/ | (Feature request on Ideas forum) | Proposed enhanced chat history search; engaged community | Reference their forum proposal — ask if they'd beta-test |
| 4 | **oliviacraft** | OpenAI Codex | https://github.com/orgs/OpenAI/discussions/117 | (See GitHub discussion) | Active in memory feature discussions | Ask what memory feature would unlock their use case |
| 5 | **6461** | Cursor IDE | https://forum.cursor.com/t/persistent-intelligent-project-memory/ | (Feature request) | Requested persistent project memory | Reference their feature request — follow up on priority |
| 6 | **mvanhorn** | Paperclip AI | https://github.com/paperclipai/paperclip/issues/460 | "A database-backed memory store where agents can save and retrieve context across sessions." | Original proposal for persistent memory store | Ask about their production needs |
| 7 | **aurexav** | OpenAI Codex | https://github.com/orgs/OpenAI/discussions/117 | (See GitHub discussion) | Provided memory feedback | Ask about their Codex workflow |
| 8 | **PACHAKUTlQ** | OpenAI Codex | https://github.com/orgs/OpenAI/discussions/117 | (See GitHub discussion) | Memory feature request | Ask what they'd build with persistent memory |
| 9 | **kinthaiofficial** | OpenAI Codex | https://github.com/orgs/OpenAI/discussions/117 | (See GitHub discussion) | Asked about context persistence | Ask about context limits they hit |
| 10 | **kimjune01** | Aider | https://github.com/promptingtechs/aider/ | (Aider maintainer) | Built memory features into Aider | Ask about lessons learned building memory |
| 11 | **Omee11** | Paperclip AI | https://github.com/paperclipai/paperclip/issues/3326 | (Commented on #3326) | Discussed consolidation of memory features | Ask about their use case |
| 12 | **seanpmgallagher** | Multica | https://github.com/seanpmgallagher/Multica/ | (Multica GitHub) | Active in Multica issues | Ask about multica memory needs |
| 13 | **iFwu** | Multica | https://github.com/seanpmgallagher/Multica/ | (Multica GitHub) | Active in Multica | Ask about their use case |

---

## Candidates by Project

### OpenAI Codex (Tier 1)
- talshebek, oliviacraft, aurexav, PACHAKUTlQ, kinthaiofficial — All from GitHub discussions with active memory feedback

### Paperclip AI (Tier 1)
- rudyjellis (production LEARNINGS.md workaround), mvanhorn (original proposal), Omee11

### Cursor IDE (Tier 2)
- Li-xiaofeng (chat history search), 6461 (persistent project memory)

### Aider (Tier 2)
- kimjune01 (maintainer who implemented memory)

### Multica (Tier 3)
- seanpmgallagher, iFwu

---

## Method + Caveats

### Research Method
- Searched GitHub issues and discussions for each named platform
- Searched Cursor IDE Ideas forum
- Cross-referenced handles across multiple sources for de-duplication

### Platform Status

| Project | Status | Notes |
|---------|--------|-------|
| **Claude Code** | NOT FOUND | No accessible public sources (Anthropic employees, closed beta) |
| **OpenAI Codex** | ✅ FOUND | Active GitHub discussions with user feedback |
| **Cursor** | ✅ FOUND | Ideas forum with feature requests |
| **Windsurf** | ⚠️ LIMITED | Cascade Memories feature exists; Reddit blocked by login |
| **Aider** | ✅ FOUND | GitHub issues with memory feature requests |
| **Paperclip AI** | ✅ FOUND | Multiple active GitHub issues |
| **Multica** | ✅ FOUND | GitHub repo with issues |

### Caveats
- Tier rankings based on engagement level and specificity of feedback, not necessarily suitability for interview
- Some quotes paraphrased from issue/discussion context
- Windsurf user handles not captured due to Reddit login block
- Claude Code not found due to closed/limited public sources