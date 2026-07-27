"""Durable JSONL helpers for incremental Harness traces."""

import json
import os
from pathlib import Path


def initialize_jsonl(path):
    """Create an empty trace, replacing any stale run with the same identity."""
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as trace_file:
        trace_file.flush()
        os.fsync(trace_file.fileno())


def append_jsonl_record(path, record):
    """Durably append one complete JSON record before returning."""
    trace_path = Path(path)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with trace_path.open("a", encoding="utf-8") as trace_file:
        trace_file.write(line)
        trace_file.flush()
        os.fsync(trace_file.fileno())


def write_json_atomic(path, payload):
    """Replace a JSON file atomically after flushing its complete payload."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False)
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, output_path)


def resolve_git_commit(start_path):
    """Resolve HEAD without requiring the git executable in the runtime."""
    current = Path(start_path).resolve()
    for parent in (current, *current.parents):
        git_path = parent / ".git"
        if git_path.is_file():
            pointer = git_path.read_text(encoding="utf-8").strip()
            if pointer.startswith("gitdir:"):
                git_path = (git_path.parent / pointer.split(":", 1)[1].strip()).resolve()
        if not git_path.is_dir():
            continue
        head = (git_path / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head or None
        reference = head.split(":", 1)[1].strip()
        reference_path = git_path / reference
        if reference_path.exists():
            return reference_path.read_text(encoding="utf-8").strip() or None
        packed_refs = git_path / "packed-refs"
        if packed_refs.exists():
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith(("#", "^")):
                    commit, name = line.split(" ", 1)
                    if name == reference:
                        return commit
        return None
    return None


def load_complete_jsonl(path):
    """Load complete records, tolerating only a truncated final line."""
    trace_path = Path(path)
    if not trace_path.exists():
        return []
    payload = trace_path.read_bytes()
    lines = payload.splitlines(keepends=True)
    records = []
    for index, raw_line in enumerate(lines):
        if not raw_line.strip():
            continue
        try:
            records.append(json.loads(raw_line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            is_truncated_final_line = index == len(lines) - 1 and not raw_line.endswith(b"\n")
            if is_truncated_final_line:
                break
            raise
    return records


def summarize_trace_records(records):
    """Reconstruct execution metrics using only persisted turn records."""
    summary = {
        "turn_count": len(records),
        "env_step_count": 0,
        "status_counts": {},
        "primitive_counts": {},
        "postconditions_met": 0,
        "postconditions_failed": 0,
        "termination_reasons": {},
        "task_success": 0.0,
    }
    for record in records:
        status = record.get("status", "missing")
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
        primitive = record.get("primitive")
        if primitive is not None:
            summary["primitive_counts"][primitive] = (
                summary["primitive_counts"].get(primitive, 0) + 1
            )
        step_results = record.get("step_results", [])
        summary["env_step_count"] += len(step_results)
        for step in step_results:
            summary["task_success"] = max(
                summary["task_success"], float(step.get("task_success", 0.0))
            )
        postcondition = record.get("primitive_postcondition_met")
        if postcondition is True:
            summary["postconditions_met"] += 1
        elif postcondition is False:
            summary["postconditions_failed"] += 1
        reason = record.get("termination_reason")
        if reason is not None:
            summary["termination_reasons"][reason] = (
                summary["termination_reasons"].get(reason, 0) + 1
            )
    return summary