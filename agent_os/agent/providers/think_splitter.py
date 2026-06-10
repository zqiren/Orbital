# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Inline ``<think>…</think>`` reasoning splitter.

Some providers (notably MiniMax M-series) emit chain-of-thought reasoning
*inline* in the ``content`` field wrapped in ``<think>…</think>`` rather than
in a dedicated ``reasoning_content`` field. The provider registry encodes this
as ``ReasoningInfo.field is None`` + ``supported=True`` ("inline think"). Orbital
otherwise only separates reasoning when it arrives in a named delta field, so
without this splitter the entire monologue renders as visible message content.

``InlineThinkSplitter`` is a stateful parser that peels the think segments out
of a (possibly chunked) content stream, returning visible text and reasoning
text separately. It tolerates tags split across streaming deltas by buffering a
trailing partial-tag prefix until the next delta resolves it.
"""

from __future__ import annotations


class InlineThinkSplitter:
    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self._in_think = False
        # Holds a tail that might be the start of a tag spanning into the next
        # delta (e.g. "<thi"), so we never emit a partial tag as visible text.
        self._buf = ""

    @staticmethod
    def _partial_tag_suffix_len(buf: str, tag: str) -> int:
        """Length of the longest suffix of ``buf`` that is a proper prefix of
        ``tag`` (1..len(tag)-1). Used only when the full tag is absent — it is
        the amount we must retain in case the tag completes next delta."""
        maxk = min(len(buf), len(tag) - 1)
        for k in range(maxk, 0, -1):
            if buf[-k:] == tag[:k]:
                return k
        return 0

    def feed(self, text: str) -> tuple[str, str]:
        """Process one delta; return ``(visible, reasoning)`` extracted now.

        Characters that may belong to a not-yet-complete tag are retained
        internally and surface on a later ``feed`` or on ``flush``.
        """
        if not text:
            return "", ""
        self._buf += text
        visible: list[str] = []
        reasoning: list[str] = []

        while self._buf:
            if not self._in_think:
                idx = self._buf.find(self.OPEN)
                if idx != -1:
                    visible.append(self._buf[:idx])
                    self._buf = self._buf[idx + len(self.OPEN):]
                    self._in_think = True
                    continue
                keep = self._partial_tag_suffix_len(self._buf, self.OPEN)
                emit_to = len(self._buf) - keep
                visible.append(self._buf[:emit_to])
                self._buf = self._buf[emit_to:]
                break
            else:
                idx = self._buf.find(self.CLOSE)
                if idx != -1:
                    reasoning.append(self._buf[:idx])
                    self._buf = self._buf[idx + len(self.CLOSE):]
                    self._in_think = False
                    continue
                keep = self._partial_tag_suffix_len(self._buf, self.CLOSE)
                emit_to = len(self._buf) - keep
                reasoning.append(self._buf[:emit_to])
                self._buf = self._buf[emit_to:]
                break

        return "".join(visible), "".join(reasoning)

    def flush(self) -> tuple[str, str]:
        """Drain any retained tail at stream end. A buffered partial tag that
        never completed surfaces as visible text (outside think) or reasoning
        (inside an unclosed think block)."""
        rest = self._buf
        self._buf = ""
        if not rest:
            return "", ""
        return ("", rest) if self._in_think else (rest, "")
