# Session auto-names are derived from the first user message, but machine-
# originated sessions carry machine prefixes there — the queue wraps items in
# "[QUEUE ITEM | …]" + a header contract, and attachments prepend an
# "<attached_files>" block. Deriving the name from the RAW content stored
# machine markup as the display name (worse: word-boundary truncation cut the
# attachment name to "<attached_files>\n-…"). The derivation must strip the
# machine prefixes first so the name is the user's actual text, falling back
# to the file basename for attachment-only messages (bug found 2026-08-12).
import json

from agent_os.agent.session import Session, _derive_name, is_machine_derived_name
from agent_os.queue.dispatcher import QueueDispatcher

QUEUE_HEADER = "[QUEUE ITEM | id=item_ab12 | attempt=1]\n" + QueueDispatcher.HEADER_CONTRACT
ATTACH_BLOCK = (
    "<attached_files>\n"
    "- /uploads/2026-08-12T053000-report.pdf (application/pdf, 1.2 MB)\n"
    "</attached_files>\n\n"
)


# ---------------------------------------------------------------- _derive_name

def test_plain_short_content_passes_through():
    assert _derive_name("write an essay") == "write an essay"


def test_plain_long_content_truncates_at_word_boundary():
    name = _derive_name("word " * 20)
    assert name is not None
    assert name.endswith("…")
    assert len(name) <= 51


def test_queue_wrapper_stripped_to_item_text():
    assert _derive_name(QUEUE_HEADER + "整理 Reddit 线索并更新 tracker") == "整理 Reddit 线索并更新 tracker"


def test_queue_wrapper_with_long_item_text_truncates_the_item_text():
    name = _derive_name(QUEUE_HEADER + "sweep the Discord candidate list " * 5)
    assert name is not None
    assert name.startswith("sweep the Discord")
    assert name.endswith("…")


def test_attachment_block_with_typed_text_names_from_the_text():
    assert _derive_name(ATTACH_BLOCK + "summarize this report") == "summarize this report"


def test_attachment_only_message_falls_back_to_basename():
    # No typed text: the file basename (upload timestamp stripped) is the name.
    assert _derive_name(ATTACH_BLOCK) == "report.pdf"


def test_queue_item_with_staged_files_names_from_item_text():
    # Dispatcher order: attach_prefix + header + contract + item content.
    assert _derive_name(ATTACH_BLOCK + QUEUE_HEADER + "review the doc") == "review the doc"


def test_trigger_prefix_is_NOT_stripped():
    # The frontend classifies trigger sessions from this prefix and extracts
    # the trigger's own name (web/src/lib/sessionLabel.ts) — keep the contract.
    content = "[Triggered by schedule 'Daily check' (every day 09:00)]\n\ndo the thing"
    name = _derive_name(content)
    assert name is not None
    assert name.startswith("[Triggered by schedule 'Daily check'")


def test_unrecognized_contract_degrades_to_raw_head_not_crash():
    # If the queue contract wording ever drifts, we degrade to naming from the
    # bracket-stripped remainder (same visibility as before the fix), not None.
    content = "[QUEUE ITEM | id=item_x | attempt=2]\nSome future contract text here"
    name = _derive_name(content)
    assert name is not None
    assert "QUEUE ITEM" not in name


# -------------------------------------------------- is_machine_derived_name

def test_machine_name_detection():
    assert is_machine_derived_name("<attached_files>\n-…")
    assert is_machine_derived_name("[QUEUE ITEM | id=item_ab12 | atte…")
    assert not is_machine_derived_name("Reddit 线索整理")
    assert not is_machine_derived_name("[Triggered by schedule 'Daily check'…")
    assert not is_machine_derived_name(None)


# ------------------------------------------------------------- Session.append

def test_append_stamps_clean_name_into_meta(tmp_path):
    s = Session.new("proj_11112222", str(tmp_path))
    s.append({"role": "user", "content": QUEUE_HEADER + "clean item text",
              "source": "user"})
    assert s.name == "clean item text"
    path = tmp_path / "orbital" / "sessions" / "proj_11112222.jsonl"
    first = json.loads(path.read_text().splitlines()[0])
    assert first["event"] == "session_start"
    assert first["name"] == "clean item text"


# --------------------------------------------------------------- Session.load

def _write_session(tmp_path, stem, stored_name, first_user_content):
    d = tmp_path / "orbital" / "sessions"
    d.mkdir(parents=True)
    rows = [
        {"role": "meta", "event": "session_start", "session_id": stem,
         "session_uuid": stem, "origin": "queue", "name": stored_name,
         "timestamp": "2026-08-12T00:00:00+00:00"},
        {"role": "user", "content": first_user_content, "source": "user"},
    ]
    path = d / f"{stem}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_load_rederives_when_stored_name_is_machine_markup(tmp_path):
    path = _write_session(tmp_path, "proj_aaaa1111",
                          stored_name="[QUEUE ITEM | id=item_ab12 | atte…",
                          first_user_content=QUEUE_HEADER + "the real item text")
    s = Session.load(str(path))
    assert s.name == "the real item text"


def test_load_keeps_user_rename_verbatim(tmp_path):
    path = _write_session(tmp_path, "proj_bbbb2222",
                          stored_name="My renamed session",
                          first_user_content=QUEUE_HEADER + "the real item text")
    s = Session.load(str(path))
    assert s.name == "My renamed session"


def test_derive_name_drops_the_annotation_quotes_block():
    """Spec 078 §5.4: a quoted annotation must not become the session name."""
    from agent_os.agent.session import _derive_name, _QUOTES_HEADING

    content = (
        "what is this about\n\n" + _QUOTES_HEADING + "\n"
        "[1] AGENTS.md\n    > AGENTS.md — read this first\n    note: explain"
    )
    assert _derive_name(content) == "what is this about"
