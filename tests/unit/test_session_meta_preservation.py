# Meta records (session_start identity, session_kind worker tag) must survive
# the stub-truncation rewrite and compaction — losing them un-tags fanout
# worker sessions and leaks them into the sidebar (bug found 2026-07-04).
import json

from agent_os.agent.session import Session


def _mint(tmp_path, stem="worker_worker_ab12cd34_0_deadbeef"):
    s = Session.new(stem, str(tmp_path))
    s.append_meta(
        "session_kind",
        kind="worker",
        parent_session_id="quick_tasks_11112222",
        fanout_id="ab12cd34",
        task_label="T",
    )
    s.append({"role": "user", "content": "brief", "source": "user"})
    s.append({
        "role": "assistant", "content": "", "source": "management",
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "read", "arguments": "{}"}}],
    })
    s.append({"role": "tool", "content": "X" * 600, "tool_call_id": "c1",
              "source": "management"})
    return s


def _records(tmp_path, stem="worker_worker_ab12cd34_0_deadbeef"):
    p = tmp_path / "orbital" / "sessions" / f"{stem}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_stub_rewrite_preserves_meta_records(tmp_path):
    s = _mint(tmp_path)
    s.replace_tool_results_with_stubs({"c1": "[stub]"})
    recs = _records(tmp_path)
    metas = [r for r in recs if r.get("role") == "meta"]
    assert {m["event"] for m in metas} == {"session_start", "session_kind"}
    assert recs[0]["event"] == "session_start"  # stays the first line
    kind = next(m for m in metas if m["event"] == "session_kind")
    assert kind["kind"] == "worker"
    tool = next(r for r in recs if r.get("role") == "tool")
    assert tool["content"] == "[stub]"  # the rewrite itself still works


def test_compact_preserves_meta_records(tmp_path):
    s = _mint(tmp_path)
    summary = {"role": "system", "content": "[SUMMARY]", "source": "daemon"}
    s._compact(summary, 2)
    recs = _records(tmp_path)
    metas = [r for r in recs if r.get("role") == "meta"]
    assert {m["event"] for m in metas} == {"session_start", "session_kind"}
    assert recs[0]["event"] == "session_start"
    assert any(r.get("content") == "[SUMMARY]" for r in recs)
