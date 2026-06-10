# OpenAI Codex Memory Feature Discussions - Potential Interviewees

## Source
GitHub Discussion: https://github.com/openai/codex/discussions/12567

## Overview
OpenAI developer jif-oai posted asking for feedback on adding "memories" to Codex. This generated extensive discussion from real developers with detailed opinions on memory features.

## Potential Interviewees with Public Statements

### 1. jif-oai (OpenAI Maintainer)
**Role:** OpenAI engineer working on memories feature
**Quote:** "I'm working on adding memories into Codex and I would love your opinion..."
**Relevance:** Core team working on memory feature - primary contact for technical implementation

### 2. talshebek
**Background:** Researcher experimenting with persistent conversational memory systems since 2023
**Quote:** "I've been experimenting with persistent conversational memory systems for a while, so this direction in Codex is really interesting to see... The idea is to separate reasoning from memory handling."
**Detailed Feedback:** 
- 4/5 rating for citation importance
- Prefers hybrid (automatic + manual trigger)
- Proposes layered memory: project + global + tutoring
- Built experimental memory-first architecture with separate memory agents
- Published detailed architecture: https://github.com/hack-ink/ELF
**Relevance:** Deep technical expertise in memory architecture, clear interest in AI agent memory

### 3. oliviacraft
**Background:** Extensive Claude Code user (CLAUDE.md mechanism), builds rule sets for projects
**Quote:** "For project-level memories, I would prefer manual confirmation — the cost of a wrong project rule persisting silently is high."
**Detailed Feedback:**
- Rating 4/5 for citations
- Per-project more valuable than global for coding
- Published free per-stack rule files: https://gist.github.com/oliviacraft
**Relevance:** Practical experience with persistent context files, clear articulation of memory needs

### 4. aurexav
**Background:** Building external memory system for Codex (ELF system)
**Quote:** "I am also developing an external memory system for Codex... The ELF system includes a time-to-live (TTL) mechanism for rewriting or discarding old or unnecessary memories."
**Project:** https://github.com/hack-ink/ELF
**Relevance:** Already building memory solutions - potential power user

### 5. kinthaiofficial  
**Background:** Detailed analysis of coding agent memory vs conversational memory
**Quote:** "Memory in coding agents is more nuanced than in conversational agents because code-level context has stronger dependencies than natural language."
**Key Points:**
- Proposes symbol-level memory (not just session-level)
- Cross-file dependency tracking
- Ephemeral vs persistent memory distinction
- Memory consolidation after task completion
**Relevance:** Academic/technical perspective on memory architecture

### 6. Kbediako
**Quote:** "5... Hybrid, with an option to decide between one or the other. Probably off by default with phrases or words that can trigger it (similar to skills). Both, global and per project memories."
**Relevance:** Clear user requirement for hybrid approach

### 7. jeemitsha
**Quote:** "Citation importance: 4... Auto-generation of a footgun for sensitive codebases (privacy, NDA scope, half-baked decisions you'd rather not codify yet)."
**Additional Work:** Filed #20138 proposing session-scoped notes panel
**Relevance:** Security considerations for memory

### 8. GeorgeWingg
**Quote:** "2/5 Low priority. I mostly only care if the memory leads to an outdated or wrong answer."
**Relevance:** Skeptical but constructive perspective

### 9. lattwood
**Quote:** "I'd love memories to take into account the git remote of the project folders"
**Relevance:** Practical use case for project-aware memory

## Related GitHub Issues
- #4933: "Want OpenAI to dev user memory feature for Codex"
- #13147: "Add configurable source to the memories feature" (by winnal)
- #24172: "Memories are forgotten"
- #20138: Session-scoped notes panel proposal (by jeemitsha)

## Summary
This discussion shows strong developer interest in Codex memory features with varied technical requirements. Multiple developers have detailed, thoughtful feedback and some are already building their own solutions. These represent ideal potential interviewees who have publicly expressed interest in AI agent memory/persistence.