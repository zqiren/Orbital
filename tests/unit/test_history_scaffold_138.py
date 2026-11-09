# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 138 shared scaffolding: the injected-prompt cleaner, error-text
classification and the source loader."""

from agent_os.onboarding import history_types as ht
from agent_os.onboarding.sources import load_sources
from agent_os.onboarding.text_clean import (
    KIND_AUTOMATION, KIND_INJECTED, KIND_USER, clean_user_text,
)


class TestCleanUserText:
    def test_plain_text_is_the_users(self):
        assert clean_user_text("  fix the login bug ") == (KIND_USER, "fix the login bug")

    def test_memory_prompt_is_injected(self):
        c = clean_user_text("[Orbital memory] Your last turn is ending. Decide whether anything durable\n…")
        assert c == (KIND_INJECTED, None)

    def test_schedule_trigger_is_an_automation_with_its_task(self):
        c = clean_user_text("[Triggered by schedule 'Daily digest' (every day at 9:00)]\n\nSummarize the inbox")
        assert c == (KIND_AUTOMATION, "Summarize the inbox")

    def test_file_watch_trigger_drops_the_changed_files_line(self):
        c = clean_user_text("[Triggered by file_watch 'docs']\n\nChanged files: a.md, b.md\n\nRebuild the index")
        assert c == (KIND_AUTOMATION, "Rebuild the index")

    def test_recap_keeps_only_the_text_after_the_footer(self):
        text = ("Conversation so far (earlier messages in this chat, provided as context; "
                "prior replies are labeled \"assistant\"):\n\nuser: hi\nassistant: hello"
                "\n\n--- end of conversation so far ---\n\nnow add tests")
        assert clean_user_text(text) == (KIND_USER, "now add tests")

    def test_previous_session_logs_header_and_bullets_are_dropped(self):
        text = "Previous session logs — 2 earlier runs\n- ran tests\n- fixed lint\n\nship it"
        assert clean_user_text(text) == (KIND_USER, "ship it")

    def test_system_reminder_and_pasted_tags_are_stripped(self):
        text = "<system-reminder>ctx</system-reminder>\n<pasted_content id=\"x\">log line</pasted_content> why?"
        assert clean_user_text(text) == (KIND_USER, "log line why?")

    def test_only_a_reminder_is_injected(self):
        assert clean_user_text("<system-reminder>ctx</system-reminder>") == (KIND_INJECTED, None)

    def test_non_string_is_injected(self):
        assert clean_user_text(None) == (KIND_INJECTED, None)


class TestClassifyErrorText:
    def test_claude_session_limit_with_reset_in_zone(self):
        stop = ht.classify_error_text(
            "You've hit your session limit · resets 10pm (Asia/Shanghai)",
            at="2026-10-25T09:00:00+00:00")  # 17:00 in Shanghai
        assert stop.cause == ht.CAUSE_USAGE_LIMIT
        assert stop.resets_at == "2026-10-25T14:00:00+00:00"  # 22:00 Shanghai

    def test_reset_already_passed_today_rolls_to_tomorrow(self):
        stop = ht.classify_error_text("You've hit your usage limit · resets 8am (UTC)",
                                      at="2026-10-25T09:00:00+00:00")
        assert stop.resets_at == "2026-10-26T08:00:00+00:00"

    def test_limit_without_reset(self):
        stop = ht.classify_error_text("You're out of usage credits")
        assert stop.cause == ht.CAUSE_USAGE_LIMIT and stop.resets_at is None

    def test_a_worker_that_is_not_signed_in_reads_as_sign_in(self):
        # Spec 167: a failed import scan shows why it stopped. Verbatim from
        # the dispatch-failed rows of a signed-out worker.
        for text in ("[Sub-agent] minimax-code could not start: Error: adapter start failed: "
                     "Authentication required: Run `mcode login` and try again.",
                     "Error: 'claude-code' is installed but not signed in",
                     "Not logged in · Please run /login"):
            assert ht.classify_error_text(text).cause == ht.CAUSE_NETWORK_AUTH, text

    def test_model_unavailable(self):
        stop = ht.classify_error_text("400 Bad Request: The 'gpt-x' model is not supported when using Codex with a ChatGPT account.")
        assert stop.cause == ht.CAUSE_MODEL_UNAVAILABLE

    def test_a_gateway_refusing_the_model_reads_as_model_unavailable(self):
        # OpenCode Go, verbatim: a retired id, and a model the workspace's
        # privacy Regions do not allow.
        for text in ("LLM error (non-recoverable): Upstream request failed: Model is unavailable.",
                     "The latest version of this model is only available hosted in China "
                     "and requires explicit opt in: https://opencode.ai/workspace/w/go"):
            assert ht.classify_error_text(text).cause == ht.CAUSE_MODEL_UNAVAILABLE

    def test_network_and_auth(self):
        assert ht.classify_error_text("read ECONNRESET").cause == ht.CAUSE_NETWORK_AUTH
        assert ht.classify_error_text("Failed to authenticate. API Error: 403").cause == ht.CAUSE_NETWORK_AUTH

    def test_unrelated_text_is_none(self):
        assert ht.classify_error_text("All tests pass.") is None
        assert ht.classify_error_text(None) is None


def test_load_sources_skips_modules_not_built_yet():
    # Only built readers load; an absent module is not an error.
    assert all(hasattr(s, "agent") for s in load_sources())
