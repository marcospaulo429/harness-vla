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
from dataclasses import dataclass, field
from typing import List


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
