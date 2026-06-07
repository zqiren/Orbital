# Aider Memory/Persistence Feature Requests - Potential Interviewees

## Source
GitHub Issue: https://github.com/Aider-AI/aider/issues/3607

## Issue: "More control over chat history"
**Opened by:** PACHAKUTlQ
**URL:** https://github.com/Aider-AI/aider/issues/3607

### Problem Statement
Aider includes the whole chat history in context, and the only way to control context is using /clear which deletes the whole chat history. The request is for more granular control over what chat history is put into context.

### Developer Quotes and Context

#### PACHAKUTlQ (Author)
**Quote:** "Allowing more control over what chat history is put into context can save quite a lot tokens and improve response quality (messy context confuses LLM)."
**Proposal:** 
- Run /history to open history.md where all the chat history can be selected/unselected by markdown checkbox
- Only display first 300 characters for every user input and LLM output
- Keep a copy of history.md for backup

**Follow-up:** "I can write this and create PR, but I dont want to waste my time writing a feature you will never merge."

#### EleVicted
**Quote:** "I'm running in to this exact same issue, wanting some of the ease of context management I'm able to easily achieve in Google AI studio. Should be possible to delete specific messages, edit previous or even the LLM replies."
**Context:** Has been searching for where the context file is stored to build their own solution.

#### bartlomiejwolk
**Quote:** "I think that Aider definitely needs to give the user more control over context management. I'm not sure about the checklist approach but imo it's at least an interesting idea."

#### Zhaoyilunnn
**Quote:** "I also vote for more control. Actually --max-chat-history-tokens seems to not work well..."
**Issue:** Reports that even with --max-chat-history-tokens setting, the token limit is exceeded.

#### kimjune01 (Aider Contributor)
**Quote:** "I've been working on this — opened #4940 (backend) and #4941 (commands) that add /topics and /drop-topic behind an opt-in --chat-history-summarizer union-find flag. Would love feedback from anyone who's hit this pain point."
**Relevance:** Core contributor working on memory feature

### Related Issues/PRs
- #4940: Add opt-in union-find chat history summarizer
- #4941: Add /topics and /drop-topic commands

## Summary
Aider users are actively requesting more granular control over chat history persistence. Multiple developers have expressed frustration with the current all-or-nothing approach. PACHAKUTlQ proposed a detailed solution. These are ideal potential interviewees who have publicly expressed interest in AI agent memory/persistence for coding.