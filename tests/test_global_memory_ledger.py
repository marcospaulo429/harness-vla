import json

import pytest

from embodiedbench.planner.harness.global_memory import (
    GlobalMemoryEvidence,
    GlobalMemoryLedger,
    evidence_from_completed_trace,
)


def _write_trace(path, records, *, final_newline=True):
    payload = "\n".join(json.dumps(record) for record in records)
    if final_newline:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def test_completed_trace_produces_trace_backed_failure_candidate(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    _write_trace(trace_path, [{
        "turn": 3,
        "primitive": "vla_act",
        "termination_reason": "empty_grasp",
        "primitive_postcondition_met": False,
    }])

    evidence = evidence_from_completed_trace(trace_path, run_status="completed")

    assert len(evidence) == 1
    assert evidence[0].kind == "failure_model"
    assert evidence[0].turns == (3,)
    assert len(evidence[0].trace_sha256) == 64


def test_reprocessing_same_trace_is_idempotent(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    ledger_path = tmp_path / "ledger.json"
    _write_trace(trace_path, [{
        "turn": 1,
        "primitive": "move_to",
        "termination_reason": "postcondition_met",
        "primitive_postcondition_met": True,
    }])
    candidate = evidence_from_completed_trace(
        trace_path, run_status="completed"
    )[0]
    ledger = GlobalMemoryLedger()

    assert ledger.add(candidate) is True
    assert ledger.add(candidate) is False
    ledger.save(ledger_path)
    loaded = GlobalMemoryLedger.load(ledger_path)

    assert [item.to_dict() for item in loaded.evidence] == [candidate.to_dict()]


def test_incomplete_run_or_trace_cannot_produce_evidence(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    _write_trace(trace_path, [{"turn": 1}], final_newline=False)

    with pytest.raises(ValueError, match="completed run"):
        evidence_from_completed_trace(trace_path, run_status="incomplete")
    with pytest.raises(ValueError, match="incomplete trace"):
        evidence_from_completed_trace(trace_path, run_status="completed")


def test_evidence_requires_turn_provenance():
    with pytest.raises(ValueError, match="positive trace turns"):
        GlobalMemoryEvidence(
            kind="failure_model",
            text="Observed failure",
            trace_sha256="a" * 64,
            turns=(),
        )


def test_deployment_ledger_rejects_writes(tmp_path):
    ledger = GlobalMemoryLedger(read_only=True)
    candidate = GlobalMemoryEvidence(
        kind="failure_model",
        text="Observed failure",
        trace_sha256="a" * 64,
        turns=(1,),
    )

    with pytest.raises(PermissionError, match="read-only"):
        ledger.add(candidate)
    with pytest.raises(PermissionError, match="read-only"):
        ledger.save(tmp_path / "ledger.json")
