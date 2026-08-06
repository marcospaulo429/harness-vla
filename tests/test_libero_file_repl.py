import json

import pytest

from embodiedbench.evaluator.libero_file_repl import (
    LiberoFileProtocolError,
    LiberoFileREPL,
)


def _outcome(value, *, done=False):
    return {"state": {"value": value}, "done": done, "error": None}


def test_one_command_executes_exactly_once_and_commits_all_files(tmp_path):
    calls = []
    repl = LiberoFileREPL(tmp_path, lambda invocation: calls.append(invocation) or _outcome(1))
    repl.step({"name": "move", "target": "bowl"})

    state = repl.process_one()

    assert calls == [{"name": "move", "target": "bowl"}]
    assert state["state"] == {"value": 1}
    assert {path.name for path in tmp_path.iterdir()} == {
        "command_01.json", "state_01.json", "log_01.json", "ledger.jsonl", "status.json"
    }


def test_replay_and_restart_return_persisted_result_without_execution(tmp_path):
    first = LiberoFileREPL(tmp_path, lambda invocation: _outcome(7))
    invocation = {"name": "grasp", "target": "cup"}
    first.step(invocation)
    expected = first.process_one()
    calls = []
    restarted = LiberoFileREPL(tmp_path, lambda invocation: calls.append(invocation))

    assert restarted.step(invocation, turn=1) == expected
    assert restarted.process_one() == expected
    assert calls == []


def test_gap_and_conflicting_replay_fail_closed(tmp_path):
    repl = LiberoFileREPL(tmp_path, lambda invocation: _outcome(1))
    with pytest.raises(LiberoFileProtocolError, match="gap"):
        repl.step({"name": "move"}, turn=2)
    repl.step({"name": "move"})
    repl.process_one()
    with pytest.raises(LiberoFileProtocolError, match="conflicting replay"):
        repl.step({"name": "release"}, turn=1)


def test_states_are_monotonic_and_next_command_waits_for_state(tmp_path):
    repl = LiberoFileREPL(tmp_path, lambda invocation: _outcome(invocation["value"]))
    repl.step({"value": 1})
    with pytest.raises(LiberoFileProtocolError):
        repl.step({"value": 2})
    repl.process_one()
    repl.step({"value": 2})
    repl.process_one()
    assert json.loads((tmp_path / "state_01.json").read_text())["turn"] == 1
    assert json.loads((tmp_path / "state_02.json").read_text())["turn"] == 2


def test_crash_before_commit_can_execute_again_on_retry(tmp_path, monkeypatch):
    calls = []
    repl = LiberoFileREPL(tmp_path, lambda invocation: calls.append(invocation) or _outcome(3))
    repl.step({"name": "move"})
    import embodiedbench.evaluator.libero_file_repl as module

    original = module.write_json_atomic
    crashed = False

    def crash_before_commit(path, payload):
        nonlocal crashed
        if not crashed and path.name == "log_01.json":
            crashed = True
            raise OSError("simulated crash")
        original(path, payload)

    monkeypatch.setattr(module, "write_json_atomic", crash_before_commit)
    with pytest.raises(OSError, match="simulated crash"):
        repl.process_one()
    assert not (tmp_path / "state_01.json").exists()
    result = repl.process_one()
    assert result["state"] == {"value": 3}
    assert len(calls) == 2


def test_restart_repairs_metadata_after_committed_state_without_reexecution(
    tmp_path, monkeypatch
):
    calls = []
    repl = LiberoFileREPL(tmp_path, lambda invocation: calls.append(invocation) or _outcome(4))
    repl.step({"name": "move"})
    import embodiedbench.evaluator.libero_file_repl as module

    original = module.append_jsonl_record

    def crash_after_state(path, record):
        raise OSError("simulated metadata crash")

    monkeypatch.setattr(module, "append_jsonl_record", crash_after_state)
    with pytest.raises(OSError, match="metadata crash"):
        repl.process_one()
    monkeypatch.setattr(module, "append_jsonl_record", original)

    restarted = LiberoFileREPL(tmp_path, lambda invocation: calls.append(invocation))
    state = restarted.process_one()
    assert state["state"] == {"value": 4}
    assert len(calls) == 1
    assert restarted.ledger()[0]["turn"] == 1
    assert json.loads((tmp_path / "status.json").read_text())["last_completed_turn"] == 1


def test_malformed_command_is_rejected_without_execution(tmp_path):
    (tmp_path / "command_01.json").write_text('{"turn": 1}', encoding="utf-8")
    calls = []
    with pytest.raises(LiberoFileProtocolError, match="schema"):
        LiberoFileREPL(tmp_path, lambda invocation: calls.append(invocation)).process_one()
    assert calls == []


def test_files_and_schemas_are_self_contained_and_reconstructable(tmp_path):
    repl = LiberoFileREPL(tmp_path, lambda invocation: _outcome(9, done=True))
    repl.step({"name": "finish"})
    repl.process_one()
    command = json.loads((tmp_path / "command_01.json").read_text())
    state = json.loads((tmp_path / "state_01.json").read_text())
    log = json.loads((tmp_path / "log_01.json").read_text())
    status = json.loads((tmp_path / "status.json").read_text())
    assert command == {"schema_version": 1, "turn": 1, "invocation": {"name": "finish"}}
    assert state["command_sha256"] == log["command_sha256"]
    assert log["invocation"] == command["invocation"]
    assert state["done"] is status["done"] is True
    assert repl.ledger()[0]["turn"] == status["last_completed_turn"] == 1