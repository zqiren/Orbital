# Pin-to-top persistence (BACKLOG spec 067).
#
# `pinned` rides the SAME session_start meta record as `name`, so the two are
# writers on one line. These lock down (a) the round trip, (b) that the two
# fields are independent rather than clobbering each other, and (c) that a pin
# survives the rewrite paths — which is the property that makes the meta the
# right home for it in the first place.
import json

from agent_os.agent.session import Session


def _mint(tmp_path, stem="chat_pin_deadbeef"):
    s = Session.new(stem, str(tmp_path))
    s.append({"role": "user", "content": "hello there", "source": "user"})
    return s


def _start_meta(tmp_path, stem="chat_pin_deadbeef"):
    p = tmp_path / "orbital" / "sessions" / f"{stem}.jsonl"
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("role") == "meta" and rec.get("event") == "session_start":
            return rec
    raise AssertionError("no session_start meta on disk")


def _path(tmp_path, stem="chat_pin_deadbeef"):
    return str(tmp_path / "orbital" / "sessions" / f"{stem}.jsonl")


def test_pin_defaults_false_and_writes_nothing(tmp_path):
    """An untouched session is unpinned and carries no `pinned` key.

    This is what makes spec 067 migration-free: every pre-067 log on disk has
    no such key, and absence must read as False.
    """
    _mint(tmp_path)
    assert "pinned" not in _start_meta(tmp_path)
    assert Session.load(_path(tmp_path)).pinned is False


def test_set_pinned_round_trips_through_disk(tmp_path):
    s = _mint(tmp_path)
    s.set_pinned(True)
    assert s.pinned is True
    assert _start_meta(tmp_path)["pinned"] is True
    assert Session.load(_path(tmp_path)).pinned is True

    s.set_pinned(False)
    assert _start_meta(tmp_path)["pinned"] is False
    assert Session.load(_path(tmp_path)).pinned is False


def test_pin_and_rename_do_not_clobber_each_other(tmp_path):
    """Both write the same physical line; neither may drop the other's field.

    Guards the reason `_apply_meta_fields` is field-agnostic instead of each
    field owning its own read-modify-write.
    """
    s = _mint(tmp_path)
    s.set_name("Weekly report")
    s.set_pinned(True)
    meta = _start_meta(tmp_path)
    assert meta["name"] == "Weekly report"
    assert meta["pinned"] is True

    # ... and in the other order, on a session loaded fresh from disk.
    loaded = Session.load(_path(tmp_path))
    loaded.set_pinned(False)
    loaded.set_name("Renamed after unpin")
    meta = _start_meta(tmp_path)
    assert meta["name"] == "Renamed after unpin"
    assert meta["pinned"] is False


def test_pin_survives_stub_truncation_and_compaction(tmp_path):
    """The rewrite paths regenerate the file from `_messages`, which excludes
    meta. `_collect_meta_lines` is what carries session_start across — the same
    mechanism that keeps `name`, `origin` and the worker tag alive."""
    s = _mint(tmp_path)
    s.set_pinned(True)
    s.append({
        "role": "assistant", "content": "", "source": "management",
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "read", "arguments": "{}"}}],
    })
    s.append({"role": "tool", "content": "X" * 600, "tool_call_id": "c1",
              "source": "management"})

    s.replace_tool_results_with_stubs({"c1": "[stub]"})
    assert _start_meta(tmp_path)["pinned"] is True
    assert Session.load(_path(tmp_path)).pinned is True


def test_pin_set_before_first_write_lands_via_pending_meta(tmp_path):
    """A session is a file only once it has a message. Pinning one that has
    never been written must stamp the PENDING meta, not silently no-op."""
    s = Session.new("chat_pending_cafebabe", str(tmp_path))
    s.set_pinned(True)
    s.append({"role": "user", "content": "first message", "source": "user"})
    assert _start_meta(tmp_path, "chat_pending_cafebabe")["pinned"] is True
