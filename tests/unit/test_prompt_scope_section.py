from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext


def _ctx(**over):
    base = dict(
        workspace="/tmp/ws", model="m", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=["read", "glob", "grep"],
        os_type="macos", datetime_now="2026-07-05T00:00:00",
        is_scratch=True,
        scope_projects=[
            {"name": "Orbital-marketing", "path": "/Users/u/Desktop/orbital-marketing"},
            {"name": "Hn-daily", "path": "/Users/u/Desktop/hn-daily"},
        ],
    )
    base.update(over)
    return PromptContext(**base)


def _semi_stable(ctx):
    _, semi, _ = PromptBuilder().build(ctx)
    return semi


def test_scratch_prompt_lists_in_scope_projects():
    semi = _semi_stable(_ctx())
    assert "Cross-Project Read Access" in semi
    assert "Orbital-marketing: /Users/u/Desktop/orbital-marketing" in semi
    assert "Hn-daily: /Users/u/Desktop/hn-daily" in semi
    assert "ABSOLUTE path" in semi


def test_non_scratch_prompt_has_no_scope_section():
    semi = _semi_stable(_ctx(is_scratch=False))
    assert "Cross-Project Read Access" not in semi


def test_scratch_with_empty_scope_has_no_section():
    semi = _semi_stable(_ctx(scope_projects=[]))
    assert "Cross-Project Read Access" not in semi
