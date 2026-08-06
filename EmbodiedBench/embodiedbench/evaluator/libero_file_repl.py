"""Idempotent file-mediated REPL for synchronous LIBERO adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from embodiedbench.planner.harness.trace_io import (
    append_jsonl_record,
    load_complete_jsonl,
    write_json_atomic,
    write_text_atomic,
)


SCHEMA_VERSION = 1


class LiberoFileProtocolError(RuntimeError):
    """Raised when persisted protocol files are inconsistent or invalid."""


def _name(kind: str, turn: int) -> str:
    return "%s_%02d.json" % (kind, turn)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiberoFileProtocolError("malformed protocol file: %s" % path.name) from exc
    if not isinstance(payload, dict):
        raise LiberoFileProtocolError("protocol file must contain an object: %s" % path.name)
    return payload


def _command_digest(command: Dict[str, Any]) -> str:
    encoded = json.dumps(
        command, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class LiberoFileREPL:
    """Publish planner turns and execute at most one pending command per call."""

    def __init__(self, directory, executor: Optional[Callable[[Any], Any]] = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.executor = executor
        self.ledger_path = self.directory / "ledger.jsonl"
        self.status_path = self.directory / "status.json"

    def _commands(self) -> Dict[int, Path]:
        commands = {}
        for path in self.directory.glob("command_*.json"):
            suffix = path.stem[len("command_") :]
            if not suffix.isdigit() or int(suffix) < 1 or path.name != _name("command", int(suffix)):
                raise LiberoFileProtocolError("invalid command filename: %s" % path.name)
            commands[int(suffix)] = path
        return commands

    def _validate_command(self, path: Path, expected_turn: int) -> Dict[str, Any]:
        command = _read_json(path)
        if set(command) != {"schema_version", "turn", "invocation"}:
            raise LiberoFileProtocolError("invalid command schema: %s" % path.name)
        if command["schema_version"] != SCHEMA_VERSION or command["turn"] != expected_turn:
            raise LiberoFileProtocolError("command identity mismatch: %s" % path.name)
        if not isinstance(command["invocation"], dict):
            raise LiberoFileProtocolError("command invocation must be an object: %s" % path.name)
        return command

    def _completed(self) -> Dict[int, Dict[str, Any]]:
        completed = {}
        turn = 1
        while True:
            state_path = self.directory / _name("state", turn)
            log_path = self.directory / _name("log", turn)
            if not state_path.exists():
                future_states = []
                for path in self.directory.glob("state_*.json"):
                    suffix = path.stem[len("state_") :]
                    if suffix.isdigit() and int(suffix) > turn:
                        future_states.append(path)
                if future_states:
                    raise LiberoFileProtocolError("state gap at turn %d" % turn)
                break
            if not log_path.exists():
                raise LiberoFileProtocolError("state without log at turn %d" % turn)
            command_path = self.directory / _name("command", turn)
            if not command_path.exists():
                raise LiberoFileProtocolError("state without command at turn %d" % turn)
            command = self._validate_command(command_path, turn)
            digest = _command_digest(command)
            state = _read_json(state_path)
            log = _read_json(log_path)
            required = {"schema_version", "turn", "command_sha256", "done", "error"}
            if not required.issubset(state) or not required.issubset(log):
                raise LiberoFileProtocolError("invalid result schema at turn %d" % turn)
            if any(item.get("schema_version") != SCHEMA_VERSION for item in (state, log)):
                raise LiberoFileProtocolError("result schema version mismatch at turn %d" % turn)
            if any(item.get("turn") != turn or item.get("command_sha256") != digest for item in (state, log)):
                raise LiberoFileProtocolError("conflicting replay at turn %d" % turn)
            completed[turn] = state
            turn += 1
        return completed

    def _sync_metadata(self, completed: Dict[int, Dict[str, Any]]) -> None:
        if not completed:
            return
        canonical = [
            {
                "schema_version": SCHEMA_VERSION,
                "turn": turn,
                "command_sha256": state["command_sha256"],
                "done": state["done"],
                "error": state["error"],
            }
            for turn, state in completed.items()
        ]
        ledger = load_complete_jsonl(self.ledger_path)
        if ledger != canonical[:len(ledger)] or len(ledger) > len(canonical):
            raise LiberoFileProtocolError("ledger conflicts with committed states")
        has_truncated_tail = (
            self.ledger_path.exists()
            and self.ledger_path.read_bytes()
            and not self.ledger_path.read_bytes().endswith(b"\n")
        )
        if has_truncated_tail:
            payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in canonical)
            write_text_atomic(self.ledger_path, payload)
        else:
            for record in canonical[len(ledger):]:
                append_jsonl_record(self.ledger_path, record)
        latest = completed[max(completed)]
        status = {
            "schema_version": SCHEMA_VERSION,
            "last_completed_turn": latest["turn"],
            "done": latest["done"],
            "error": latest["error"],
        }
        if not self.status_path.exists() or _read_json(self.status_path) != status:
            write_json_atomic(self.status_path, status)

    def step(self, invocation: Dict[str, Any], *, turn: Optional[int] = None) -> Dict[str, Any]:
        """Atomically publish one command after the preceding state is committed."""
        if not isinstance(invocation, dict):
            raise ValueError("invocation must be an object")
        completed = self._completed()
        self._sync_metadata(completed)
        commands = self._commands()
        next_turn = len(completed) + 1
        requested_turn = next_turn if turn is None else turn
        if isinstance(requested_turn, bool) or not isinstance(requested_turn, int) or requested_turn < 1:
            raise ValueError("turn must be a positive integer")
        command = {
            "schema_version": SCHEMA_VERSION,
            "turn": requested_turn,
            "invocation": invocation,
        }
        path = self.directory / _name("command", requested_turn)
        if requested_turn > next_turn:
            raise LiberoFileProtocolError("command gap at turn %d" % requested_turn)
        if requested_turn < next_turn or requested_turn in commands:
            persisted = self._validate_command(path, requested_turn)
            if _command_digest(persisted) != _command_digest(command):
                raise LiberoFileProtocolError("conflicting replay at turn %d" % requested_turn)
            return completed.get(requested_turn, persisted)
        if commands and max(commands) >= requested_turn:
            raise LiberoFileProtocolError("pending command sequence is inconsistent")
        write_json_atomic(path, command)
        return command

    def process_one(self) -> Optional[Dict[str, Any]]:
        """Execute and commit one pending command, or return the persisted replay."""
        completed = self._completed()
        self._sync_metadata(completed)
        commands = self._commands()
        next_turn = len(completed) + 1
        if any(turn > next_turn for turn in commands):
            raise LiberoFileProtocolError("command gap at turn %d" % next_turn)
        if next_turn not in commands:
            return completed.get(max(completed)) if completed else None
        command = self._validate_command(commands[next_turn], next_turn)
        if self.executor is None:
            raise RuntimeError("process_one requires an executor")
        outcome = self.executor(command["invocation"])
        if not isinstance(outcome, dict):
            raise TypeError("executor result must be an object")
        done = bool(outcome.get("done", False))
        error = outcome.get("error")
        if error is not None and not isinstance(error, str):
            raise TypeError("executor error must be a string or null")
        digest = _command_digest(command)
        state = {
            "schema_version": SCHEMA_VERSION,
            "turn": next_turn,
            "command_sha256": digest,
            "state": outcome.get("state"),
            "done": done,
            "error": error,
        }
        log = {
            "schema_version": SCHEMA_VERSION,
            "turn": next_turn,
            "command_sha256": digest,
            "invocation": command["invocation"],
            "result": outcome,
            "done": done,
            "error": error,
        }
        write_json_atomic(self.directory / _name("log", next_turn), log)
        write_json_atomic(self.directory / _name("state", next_turn), state)
        completed[next_turn] = state
        self._sync_metadata(completed)
        return state

    def ledger(self):
        """Return all complete advisory ledger records."""
        return load_complete_jsonl(self.ledger_path)


LiberoFilePlanner = LiberoFileREPL
LiberoFileWorker = LiberoFileREPL
