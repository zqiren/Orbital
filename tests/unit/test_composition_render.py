# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Per-spawn composition rendering (Task 4).

The renderer is the ONLY route Orbital config takes into a harness that
configures itself through a composition file. It must never string-template:
a persona is arbitrary agent-authored prose and would otherwise be a YAML
injection vector.
"""

import os
import time

import pytest
import yaml

from agent_os.agents.composition import (
    CompositionError,
    gc_stale,
    render_composition,
    unlink_rendered,
)


SHIPPED_TEMPLATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "agent_os", "agents", "assets", "dsh", "cordis.template.yml",
)


@pytest.fixture
def template(tmp_path):
    """A minimal stand-in with the same shape as the shipped template."""
    path = tmp_path / "cordis.template.yml"
    path.write_text(
        "- id: sandbox-policy\n"
        "  name: '@deepseek-ai/dsh-sandbox-policy'\n"
        "  config:\n"
        "    mode: workspace-write\n"
        "\n"
        "- id: bash\n"
        "  name: '@deepseek-ai/dsh-bash-sandbox'\n"
        "  config:\n"
        "    timeoutMs: 60000\n"
        "\n"
        "- id: acp-agent\n"
        "  name: '@deepseek-ai/dsh-acp-demo'\n"
        "  config:\n"
        "    provider: deepseek-official\n"
        "    model: deepseek-v4-flash\n"
        "    persistenceRoot: './.sessions'\n"
        "    workspaceContext:\n"
        "      maxBytes: 65536\n"
        "    persona: |\n"
        "      You are a coding assistant.\n",
        encoding="utf-8",
    )
    return str(path)


def _render(template, tmp_path, **overrides):
    kwargs = {
        "model": "deepseek-v4-pro",
        "sandbox_mode": "danger-full-access",
        "persona": "You are a sub-agent.",
        "persistence_root": str(tmp_path / "out" / "dsh-sessions"),
        "out_dir": str(tmp_path / "out"),
    }
    kwargs.update(overrides)
    return render_composition(template, **kwargs)


def _blocks(path):
    with open(path, "r", encoding="utf-8") as f:
        return {b["id"]: b for b in yaml.safe_load(f)}


class TestRenderComposition:
    def test_sets_model_sandbox_mode_persona_and_persistence_root(
        self, template, tmp_path
    ):
        out = _render(template, tmp_path)
        blocks = _blocks(out)
        assert blocks["sandbox-policy"]["config"]["mode"] == "danger-full-access"
        assert blocks["acp-agent"]["config"]["model"] == "deepseek-v4-pro"
        assert blocks["acp-agent"]["config"]["persona"] == "You are a sub-agent."
        assert blocks["acp-agent"]["config"]["persistenceRoot"] == str(
            tmp_path / "out" / "dsh-sessions"
        )

    def test_persistence_root_is_absolutised(self, template, tmp_path):
        """The template ships a cwd-relative './.sessions' — overwriting it
        with an absolute path is what keeps sessions inside the project."""
        out = _render(template, tmp_path, persistence_root="rel/dsh-sessions")
        root = _blocks(out)["acp-agent"]["config"]["persistenceRoot"]
        assert os.path.isabs(root)
        assert root.endswith(os.path.join("rel", "dsh-sessions"))

    def test_untouched_blocks_survive_verbatim(self, template, tmp_path):
        out = _render(template, tmp_path)
        blocks = _blocks(out)
        assert blocks["bash"]["config"]["timeoutMs"] == 60000
        assert blocks["acp-agent"]["config"]["provider"] == "deepseek-official"
        assert blocks["acp-agent"]["config"]["workspaceContext"]["maxBytes"] == 65536

    def test_returns_an_existing_absolute_path(self, template, tmp_path):
        out = _render(template, tmp_path)
        assert os.path.isabs(out)
        assert os.path.isfile(out)
        assert os.path.basename(out).startswith("cordis-")
        assert out.endswith(".yml")

    def test_creates_out_dir_when_missing(self, template, tmp_path):
        out_dir = tmp_path / "nested" / "sub-agents" / "dsh"
        out = _render(template, tmp_path, out_dir=str(out_dir))
        assert os.path.dirname(out) == str(out_dir)

    def test_each_call_writes_a_distinct_file(self, template, tmp_path):
        paths = {_render(template, tmp_path) for _ in range(8)}
        assert len(paths) == 8
        assert all(os.path.isfile(p) for p in paths)

    def test_concurrent_renders_for_one_handle_do_not_collide(
        self, template, tmp_path
    ):
        """Same-slug dispatch is unrestricted across projects and sessions, so
        two renders into the same dir must not share a filename."""
        a = _render(template, tmp_path, persona="alpha")
        b = _render(template, tmp_path, persona="beta")
        assert a != b
        assert _blocks(a)["acp-agent"]["config"]["persona"] == "alpha"
        assert _blocks(b)["acp-agent"]["config"]["persona"] == "beta"

    def test_template_file_is_not_modified(self, template, tmp_path):
        before = open(template, encoding="utf-8").read()
        _render(template, tmp_path)
        assert open(template, encoding="utf-8").read() == before


class TestPersonaSafety:
    HOSTILE = (
        "You are a sub-agent.\n"
        "Rules: never say \"done\" unless it is.\n"
        "- item: value  # not a comment\n"
        "\tliteral tab & {braces} and {{model}}\n"
        "yaml: [not, a, list]\n"
        "---\n"
        "...\n"
        "trailing spaces here:   \n"
        "unicode: 中文 — em dash, emoji 🚀\n"
        "*anchor &alias |pipe >fold\n"
    )

    def test_yaml_hostile_persona_round_trips_byte_for_byte(
        self, template, tmp_path
    ):
        out = _render(template, tmp_path, persona=self.HOSTILE)
        assert _blocks(out)["acp-agent"]["config"]["persona"] == self.HOSTILE

    def test_persona_cannot_inject_a_sibling_key(self, template, tmp_path):
        attack = "hi\nmodel: attacker-model\npersistenceRoot: /etc\n"
        out = _render(template, tmp_path, persona=attack)
        cfg = _blocks(out)["acp-agent"]["config"]
        assert cfg["persona"] == attack
        assert cfg["model"] == "deepseek-v4-pro"
        assert cfg["persistenceRoot"] != "/etc"

    def test_persona_cannot_inject_a_new_plugin_block(self, template, tmp_path):
        attack = "hi\n- id: evil\n  name: '@evil/plugin'\n"
        out = _render(template, tmp_path, persona=attack)
        with open(out, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert {b["id"] for b in loaded} == {"sandbox-policy", "bash", "acp-agent"}


class TestShippedTemplate:
    def test_shipped_dsh_template_renders(self, tmp_path):
        out = render_composition(
            SHIPPED_TEMPLATE,
            model="deepseek-v4-flash",
            sandbox_mode="workspace-write",
            persona="Orbital sub-agent prompt.",
            persistence_root=str(tmp_path / "dsh-sessions"),
            out_dir=str(tmp_path),
        )
        blocks = _blocks(out)
        assert blocks["acp-agent"]["config"]["model"] == "deepseek-v4-flash"
        assert blocks["sandbox-policy"]["config"]["mode"] == "workspace-write"
        assert blocks["acp-agent"]["config"]["persona"] == (
            "Orbital sub-agent prompt."
        )
        assert os.path.isabs(blocks["acp-agent"]["config"]["persistenceRoot"])
        # Every shipped plugin must survive the round-trip.
        assert len(blocks) == 9


class TestRenderErrors:
    def test_missing_template_raises(self, tmp_path):
        with pytest.raises(CompositionError):
            _render(str(tmp_path / "nope.yml"), tmp_path)

    def test_non_list_template_raises(self, tmp_path):
        bad = tmp_path / "bad.yml"
        bad.write_text("id: not-a-list\n", encoding="utf-8")
        with pytest.raises(CompositionError):
            _render(str(bad), tmp_path)

    def test_block_without_config_gets_one(self, tmp_path):
        bare = tmp_path / "bare.yml"
        bare.write_text(
            "- id: sandbox-policy\n"
            "  name: '@deepseek-ai/dsh-sandbox-policy'\n"
            "- id: acp-agent\n"
            "  name: '@deepseek-ai/dsh-acp-demo'\n",
            encoding="utf-8",
        )
        out = _render(str(bare), tmp_path)
        blocks = _blocks(out)
        assert blocks["sandbox-policy"]["config"]["mode"] == "danger-full-access"
        assert blocks["acp-agent"]["config"]["model"] == "deepseek-v4-pro"


class TestGarbageCollection:
    def _plant(self, d, name, age_days):
        p = os.path.join(str(d), name)
        with open(p, "w", encoding="utf-8") as f:
            f.write("- id: x\n")
        old = time.time() - age_days * 86400
        os.utime(p, (old, old))
        return p

    def test_removes_stale_rendered_configs(self, tmp_path):
        stale = self._plant(tmp_path, "cordis-deadbeef.yml", 9)
        fresh = self._plant(tmp_path, "cordis-cafebabe.yml", 1)
        removed = gc_stale(str(tmp_path), max_age_days=7)
        assert removed == 1
        assert not os.path.exists(stale)
        assert os.path.exists(fresh)

    def test_leaves_unrelated_files_alone(self, tmp_path):
        keep = self._plant(tmp_path, "transcript.jsonl", 99)
        keep2 = self._plant(tmp_path, "cordis-notyaml.txt", 99)
        gc_stale(str(tmp_path), max_age_days=7)
        assert os.path.exists(keep)
        assert os.path.exists(keep2)

    def test_missing_dir_is_a_noop(self, tmp_path):
        assert gc_stale(str(tmp_path / "absent")) == 0


class TestUnlink:
    def test_removes_the_rendered_file(self, template, tmp_path):
        out = _render(template, tmp_path)
        unlink_rendered(out)
        assert not os.path.exists(out)

    def test_is_idempotent_and_tolerates_none(self, template, tmp_path):
        out = _render(template, tmp_path)
        unlink_rendered(out)
        unlink_rendered(out)
        unlink_rendered(None)
        assert not os.path.exists(out)
