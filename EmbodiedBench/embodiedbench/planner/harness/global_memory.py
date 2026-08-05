"""Global Memory for the Harness VLA beta.

Global Memory stores *task-independent* operating knowledge for the fixed
primitive library: reusable success rules and failure models (e.g. empty grasp,
false visual success). In the full framework these are distilled from execution
traces; in this beta they are a **fixed manual seed** (see the paper's Appendix A
example) and are injected as context into the planner prompt.

Task Specific Memory (few-shot bootstrapping traces) is intentionally *not*
implemented in the beta — see ``docs/HARNESS_VLA_NOT_IMPLEMENTED.md``.
"""

from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from embodiedbench.planner.harness.trace_io import load_complete_jsonl, write_json_atomic


# Manual seed rules, adapted from the Harness VLA paper (Appendix A). These are
# task-independent and describe how to use the fixed primitive library well.
SEED_SUCCESS_RULES: List[str] = [
    "Use the vla_act primitive for contact-rich phases such as irregular grasping, "
    "constrained placement, or fixture interaction. After a stable grasp, prefer "
    "analytic primitives (move_to, release) for long transport and precise placement.",
    "Stage the end-effector into a favorable pre-contact pose with move_to / "
    "rotate_wrist before invoking vla_act, then invoke vla_act only for the local "
    "contact-rich operation.",
    "vla_act is retryable: if a contact attempt fails, re-stage the robot with "
    "analytic primitives and invoke vla_act again rather than abandoning the task.",
    "Reserve analytic primitives for non-contact structure: grounding the target, "
    "free-space transport, posture adjustment, and release.",
]

SEED_FAILURE_MODELS: List[str] = [
    "Empty grasp: if the gripper closes but the object does not move with the "
    "end-effector, treat the attempt as an empty grasp. Re-localize the object and "
    "re-stage before retrying vla_act.",
    "False success: do not terminate from visual proximity alone. Rely on the "
    "benchmark success signal and the latest execution feedback before concluding "
    "the task is done.",
    "Unstable staging: if the pre-contact pose does not expose the target in a "
    "reachable configuration, adjust position/orientation with analytic primitives "
    "before invoking vla_act.",
    "Detach during transport: commanding the gripper open (gripper=\"open\" or "
    "release) while holding an object detaches it immediately and the object is "
    "left behind. After a verified grasp, transport with move_to using "
    "gripper=\"close\" (or omit gripper) and open the gripper only at the "
    "destination via release or vla_act place.",
    "Repeated grasp at the same pose: if a grasp attempt fails (empty_grasp or "
    "grasp_unverified), do NOT retry the identical approach. Re-stage first: "
    "change the wrist orientation with rotate_wrist and/or approach the target "
    "from a different offset with move_to, then retry vla_act grasp. Repeating "
    "the same pose reproduces the same failed contact geometry.",
    "Path planning failure during transport: if move_to fails with 'Could not "
    "create path' while holding an object, the direct straight-line path is "
    "blocked (usually by the table surface or the destination object). Do NOT "
    "repeat the same move_to. First lift: move_to the current X,Y with a much "
    "higher Z (10-15 voxels above the destination height), then transport "
    "laterally at that safe height, and only then descend above the "
    "destination before releasing.",
]


@dataclass
class GlobalMemory:
    """Fixed-seed cross-task memory rendered into the planner prompt."""

    success_rules: List[str] = field(default_factory=lambda: list(SEED_SUCCESS_RULES))
    failure_models: List[str] = field(default_factory=lambda: list(SEED_FAILURE_MODELS))

    @classmethod
    def seeded(cls) -> "GlobalMemory":
        """Return a GlobalMemory populated with the fixed manual seed."""
        return cls()

    @classmethod
    def load(cls, path: str) -> "GlobalMemory":
        """Load from a JSON file, falling back to the seed if absent/invalid."""
        if not path or not os.path.exists(path):
            return cls.seeded()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                success_rules=list(data.get("success_rules", SEED_SUCCESS_RULES)),
                failure_models=list(data.get("failure_models", SEED_FAILURE_MODELS)),
            )
        except (OSError, json.JSONDecodeError, TypeError):
            return cls.seeded()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "success_rules": self.success_rules,
                    "failure_models": self.failure_models,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def render(self) -> str:
        """Render the memory as prompt text."""
        lines = ["Global Memory (task-independent operating knowledge):", "", "Success rules:"]
        for i, rule in enumerate(self.success_rules, 1):
            lines.append(f"  {i}. {rule}")
        lines.append("")
        lines.append("Failure models:")
        for i, fm in enumerate(self.failure_models, 1):
            lines.append(f"  {i}. {fm}")
        return "\n".join(lines)


@dataclass(frozen=True)
class GlobalMemoryEvidence:
    """A trace-backed candidate, not an automatically promoted memory entry."""

    kind: str
    text: str
    trace_sha256: str
    turns: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"success_rule", "failure_model"}:
            raise ValueError("evidence kind must be success_rule or failure_model")
        if not self.text.strip():
            raise ValueError("evidence text must be non-empty")
        if len(self.trace_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.trace_sha256
        ):
            raise ValueError("trace_sha256 must be a SHA-256 hex digest")
        if not self.turns or any(
            isinstance(turn, bool) or not isinstance(turn, int) or turn <= 0
            for turn in self.turns
        ):
            raise ValueError("evidence must reference positive trace turns")

    @property
    def identity(self) -> str:
        payload = json.dumps(
            {
                "kind": self.kind,
                "text": " ".join(self.text.split()),
                "trace_sha256": self.trace_sha256,
                "turns": list(self.turns),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "id": self.identity,
            "kind": self.kind,
            "text": self.text,
            "trace_sha256": self.trace_sha256,
            "turns": list(self.turns),
        }


@dataclass
class GlobalMemoryLedger:
    """Persist trace-backed memory candidates without silently promoting them."""

    evidence: List[GlobalMemoryEvidence] = field(default_factory=list)
    read_only: bool = False

    @classmethod
    def load(cls, path: str, *, read_only: bool = False) -> "GlobalMemoryLedger":
        ledger_path = Path(path)
        if not ledger_path.exists():
            return cls(read_only=read_only)
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
        evidence = [
            GlobalMemoryEvidence(
                kind=item["kind"],
                text=item["text"],
                trace_sha256=item["trace_sha256"],
                turns=tuple(item["turns"]),
            )
            for item in data.get("evidence", [])
        ]
        return cls(evidence=evidence, read_only=read_only)

    def add(self, candidate: GlobalMemoryEvidence) -> bool:
        if self.read_only:
            raise PermissionError("deployment Global Memory ledger is read-only")
        if any(existing.identity == candidate.identity for existing in self.evidence):
            return False
        self.evidence.append(candidate)
        self.evidence.sort(key=lambda item: item.identity)
        return True

    def save(self, path: str) -> None:
        if self.read_only:
            raise PermissionError("deployment Global Memory ledger is read-only")
        write_json_atomic(path, {"evidence": [item.to_dict() for item in self.evidence]})


def evidence_from_completed_trace(
    path: str, *, run_status: str
) -> List[GlobalMemoryEvidence]:
    """Extract structured candidates; promotion remains an explicit later decision."""
    if run_status != "completed":
        raise ValueError("only a completed run can produce Global Memory evidence")
    trace_path = Path(path)
    payload = trace_path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise ValueError("incomplete trace cannot produce Global Memory evidence")
    records = load_complete_jsonl(trace_path)
    if not records:
        raise ValueError("empty trace cannot produce Global Memory evidence")
    trace_sha256 = hashlib.sha256(payload).hexdigest()
    candidates = []
    for index, record in enumerate(records, 1):
        reason = record.get("termination_reason")
        primitive = record.get("primitive", "primitive")
        if record.get("primitive_postcondition_met") is False and reason:
            candidates.append(GlobalMemoryEvidence(
                kind="failure_model",
                text=f"{primitive} ended with failed postcondition: {reason}.",
                trace_sha256=trace_sha256,
                turns=(int(record.get("turn", index)),),
            ))
        elif record.get("primitive_postcondition_met") is True:
            candidates.append(GlobalMemoryEvidence(
                kind="success_rule",
                text=f"{primitive} satisfied its postcondition in the recorded physical state.",
                trace_sha256=trace_sha256,
                turns=(int(record.get("turn", index)),),
            ))
    return candidates
