# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Config-store support for params that never reach argv (Task 4).

dsh's model and permission mode are ordinary Settings params, but they are
delivered through the per-spawn composition file rather than the command line.
They therefore carry ``flag_template=None``, which ``build_extra_args`` must
skip instead of ``.format``-ing.
"""

import os

import pytest

from agent_os.daemon_v2.sub_agent_config_store import (
    SCHEMA,
    SubAgentConfigError,
    SubAgentConfigStore,
    _ParamSchema,
    resolve_params,
)


@pytest.fixture
def store(tmp_path):
    return SubAgentConfigStore(os.path.join(str(tmp_path), "sub_agent_config.json"))


class TestFlaglessParams:
    def test_build_extra_args_skips_none_flag_template(self, store):
        """A UI-only param must contribute nothing to argv."""
        store.set("dsh", {"model": "deepseek-v4-pro"})
        assert store.build_extra_args("dsh") == []

    def test_build_extra_args_skips_defaults_with_none_flag_template(self, store):
        """Defaults are applied even with nothing persisted — and still emit
        no argv."""
        assert resolve_params("dsh", store)["permission-mode"] == "workspace-write"
        assert store.build_extra_args("dsh") == []

    def test_flagless_params_do_not_disturb_flagged_agents(self, store):
        store.set("claude-code", {"model": "sonnet"})
        assert store.build_extra_args("claude-code") == ["--model", "sonnet"]

    def test_mixed_schema_emits_only_the_flagged_param(self, store, monkeypatch):
        monkeypatch.setitem(SCHEMA, "mixed", {
            "flagged": _ParamSchema(
                name="flagged", allowed=None, flag_template="--flagged {value}",
            ),
            "silent": _ParamSchema(
                name="silent", allowed=None, flag_template=None,
            ),
        })
        store.set("mixed", {"flagged": "a", "silent": "b"})
        assert store.build_extra_args("mixed") == ["--flagged", "a"]

    def test_flagless_params_are_still_validated(self, store):
        with pytest.raises(SubAgentConfigError):
            store.set("dsh", {"model": "gpt-4"})

    def test_flagless_params_are_surfaced_to_the_ui(self, store):
        schema = store.schema_for("dsh")
        assert schema["model"]["allowed"] == [
            "deepseek-v4-flash", "deepseek-v4-pro",
        ]
        assert schema["model"]["default"] == "deepseek-v4-flash"
        assert schema["permission-mode"]["allowed"] == [
            "workspace-write", "danger-full-access",
        ]
        assert schema["permission-mode"]["default"] == "workspace-write"

    def test_flagless_params_round_trip_through_the_store(self, store):
        store.set("dsh", {"model": "deepseek-v4-pro",
                          "permission-mode": "danger-full-access"})
        assert store.get("dsh") == {
            "model": "deepseek-v4-pro",
            "permission-mode": "danger-full-access",
        }


class TestResolveParams:
    def test_defaults_apply_when_nothing_persisted(self, store):
        assert resolve_params("dsh", store) == {
            "model": "deepseek-v4-flash",
            "permission-mode": "workspace-write",
        }

    def test_persisted_overrides_beat_defaults(self, store):
        store.set("dsh", {"model": "deepseek-v4-pro"})
        assert resolve_params("dsh", store) == {
            "model": "deepseek-v4-pro",
            "permission-mode": "workspace-write",
        }

    def test_works_without_a_store(self):
        """Dispatch must still get the SCHEMA defaults when no store is wired."""
        assert resolve_params("dsh") == {
            "model": "deepseek-v4-flash",
            "permission-mode": "workspace-write",
        }

    def test_unknown_slug_is_empty(self, store):
        assert resolve_params("nope", store) == {}


class TestDshSchema:
    def test_dsh_params_carry_no_argv_flag(self):
        for param in SCHEMA["dsh"].values():
            assert param.flag_template is None

    def test_model_allowlist(self):
        assert SCHEMA["dsh"]["model"].allowed == (
            "deepseek-v4-flash", "deepseek-v4-pro",
        )
        assert SCHEMA["dsh"]["model"].default == "deepseek-v4-flash"

    def test_permission_mode_allowlist(self):
        assert SCHEMA["dsh"]["permission-mode"].allowed == (
            "workspace-write", "danger-full-access",
        )
        assert SCHEMA["dsh"]["permission-mode"].default == "workspace-write"
