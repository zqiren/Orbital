# Legacy fallback for worker sessions whose session_kind meta was destroyed
# by pre-Task-1 rewrites: the stem minted by NativeWorkerAdapter is
# f"worker_{sanitize('worker:<id8>-<i>')}_{hex8}" == "worker_worker_..." —
# filter it out of the sidebar scan and the merged-chat aggregation.
import json

from agent_os.daemon_v2.native_worker import is_worker_session_stem
from agent_os.api.routes.agents_v2 import _read_chat_messages


def test_worker_stem_detection():
    assert is_worker_session_stem("worker_worker_d46f3fd1_0_54ff0ce6")
    # A project the user named "worker tracker" must NOT be filtered.
    assert not is_worker_session_stem("worker_tracker_ab12cd34")
    assert not is_worker_session_stem("quick_tasks_92dbcbff")
    # A chat session of a project named "worker worker" (stem = sanitized name + one hex8 suffix)
    # must NOT be filtered. Only the rigid fanout shape is filtered.
    assert not is_worker_session_stem("worker_worker_ab12cd34")


def test_read_chat_messages_skips_worker_files(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "quick_tasks_aaaa1111.jsonl").write_text(
        json.dumps({"role": "user", "content": "hi"}) + "\n")
    (sessions / "worker_worker_ab12cd34_0_deadbeef.jsonl").write_text(
        json.dumps({"role": "user", "content": "worker brief"}) + "\n")
    msgs, total = _read_chat_messages(str(sessions), 0, 0)
    assert total == 1
    assert msgs[0]["content"] == "hi"
